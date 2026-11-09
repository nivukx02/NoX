package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.ActorDetailAllocationEngine
import com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.SkeletonFrame

/**
 * Bounded per-actor 3D reconstruction scheduler.
 *
 * The canonical fusion buffer may contain dozens of actor identities. Running the full detailed
 * optimizer for every actor on every camera frame would turn correctness into thermal throttling.
 * This coordinator therefore retains every identity while scheduling detailed versus standard
 * reconstruction by actor priority and temporal cadence.
 */
class MultiActor3DReconstructionCoordinator(
    private val fusion: MultiViewFusionEngine,
) : AutoCloseable {

    data class ActorResult(
        val actorId: Long,
        val skeleton: SkeletonFrame?,
        val detailed: MultiViewFusionEngine.DetailedResult?,
        val reconstructed: Boolean,
        val predicted: Boolean = false,
        val predictionAgeMillis: Long = 0L,
        val predictionConfidence: Double = 0.0,
        val warnings: List<String>,
    )

    data class Result(
        val timestampNanos: Long,
        val actors: List<ActorResult>,
        val primaryActorId: Long?,
        val primaryDetailed: MultiViewFusionEngine.DetailedResult?,
        val reconstructedActorCount: Int,
        val detailedActorCount: Int,
        val warnings: List<String>,
        val detailStarvationRisk: Double = 0.0,
        val detailFairnessScore: Double = 1.0,
        val maximumDetailStarvationFrames: Int = 0,
    )

    private val lastProcessedTimestampNanos =
        HashMap<Long, Long>()
    private val skeletonPredictor =
        KinematicSkeletonPredictionEngine()
    private var lastPrimaryActorId: Long? = null
    private val actorWorkloadScheduler = ActorWorkloadScheduler()

    fun process(
        timestampNanos: Long,
        actors: List<MultiActorPoseAssociationEngine.Result>,
        allocation: ActorDetailAllocationEngine.Result?,
        customization: MotionCaptureCustomization,
        primaryActorId: Long? = null,
        minimumActorIntervalMillis: Long = 16L,
        maximumDetailedActorsPerFrame: Int = 12,
    ): Result {
        require(timestampNanos >= 0L)
        require(minimumActorIntervalMillis >= 0L)
        require(maximumDetailedActorsPerFrame in 1..64)

        if (actors.isEmpty()) {
            return Result(
                timestampNanos = timestampNanos,
                actors = emptyList(),
                primaryActorId = null,
                primaryDetailed = null,
                reconstructedActorCount = 0,
                detailedActorCount = 0,
                warnings =
                    listOf(
                        "MULTI_ACTOR_3D_NO_ADMITTED_ACTORS",
                    ),
                detailStarvationRisk = 0.0,
                detailFairnessScore = 1.0,
                maximumDetailStarvationFrames = 0,
            )
        }

        val selectedPrimary =
            primaryActorId
                ?: lastPrimaryActorId
                ?: actors.firstOrNull()?.actorId

        lastPrimaryActorId = selectedPrimary

        val orderedActors =
            actors
                .take(
                    customization.maxSubjects.coerceIn(1, 64),
                )
                .sortedByDescending { actor ->
                    val allocationRank =
                        allocation
                            ?.allocation(actor.actorId)
                            ?.rank
                            ?: Int.MAX_VALUE
                    if (actor.actorId == selectedPrimary) {
                        Int.MIN_VALUE
                    } else {
                        allocationRank
                    }
                }

        val schedulerResult =
            actorWorkloadScheduler.schedule(
                timestampNanos = timestampNanos,
                actors = orderedActors,
                allocation = allocation,
                customization = customization,
                primaryActorId = selectedPrimary,
                maximumDetailedActorsPerFrame =
                    minOf(
                        maximumDetailedActorsPerFrame,
                        customization.fullDetailActorBudget.coerceIn(1, 64),
                    ),
            )

        val detailedIds = schedulerResult.selectedActorIds

        val outputs = ArrayList<ActorResult>(
            orderedActors.size,
        )

        orderedActors.forEach { actor ->
                val detail =
                    allocation?.allocation(actor.actorId)

                val cadenceScale =
                    detail?.optionalCadenceScale
                        ?.coerceIn(0.25, 1.0)
                        ?: 1.0

                val targetIntervalMillis =
                    maxOf(
                        minimumActorIntervalMillis,
                        (
                            16.67 /
                                cadenceScale
                            ).toLong(),
                    )

                val last =
                    lastProcessedTimestampNanos[
                        actor.actorId
                    ]

                val due =
                    last == null ||
                        timestampNanos - last >=
                        targetIntervalMillis * 1_000_000L

                val mustReconstruct =
                    actor.actorId ==
                        selectedPrimary

                if (!due && !mustReconstruct) {
                    val configuredPredictionHorizonMillis =
                        (
                            customization
                                .maximumOcclusionPredictionFrames
                                .coerceAtLeast(0)
                                .toDouble() *
                                1_000.0 /
                                customization.targetFps
                                    .coerceIn(15, 240)
                                    .toDouble()
                        ).toLong().coerceAtLeast(1L)

                    val prediction =
                        skeletonPredictor.predict(
                            trackId = actor.actorId,
                            targetTimestampNanos = timestampNanos,
                            maximumPredictionHorizonMillis =
                                configuredPredictionHorizonMillis,
                        )

                    outputs +=
                        ActorResult(
                            actorId = actor.actorId,
                            skeleton = prediction?.skeleton,
                            detailed = null,
                            reconstructed = false,
                            predicted = prediction?.predicted == true,
                            predictionAgeMillis =
                                prediction?.predictionAgeMillis ?: 0L,
                            predictionConfidence =
                                prediction?.confidenceScale ?: 0.0,
                            warnings =
                                listOf(
                                    "ACTOR_3D_RECONSTRUCTION_RATE_LIMITED",
                                ) +
                                    (prediction?.warnings ?: listOf(
                                        "ACTOR_3D_PREDICTION_UNAVAILABLE",
                                    )),
                        )
                    return@forEach
                }

                val wantsDetailed =
                    actor.actorId in detailedIds &&
                        (
                            detail?.fullDetail == true ||
                                detail?.tiers?.contains(
                                    ActorDetailAllocationEngine.DetailTier.SURFACE,
                                ) == true ||
                                actor.actorId ==
                                    selectedPrimary
                            )

                var detailedFailureWarning: String? = null
                var reconstructionFailureWarning: String? = null

                val detailed =
                    if (wantsDetailed) {
                        runCatching {
                            fusion.reconstructDetailed(
                                timestampNanos = timestampNanos,
                                trackId = actor.actorId,
                                subjectType =
                                    actor.subjectType,
                                bodySpacePriors =
                                    bodySpacePriors(actor),
                            )
                        }.onFailure {
                            detailedFailureWarning =
                                "ACTOR_3D_DETAIL_FAILURE_" +
                                    it::class.simpleName.orEmpty()
                                        .take(48)
                        }.getOrNull()
                    } else {
                        null
                    }

                val skeleton =
                    detailed?.skeleton
                        ?: runCatching {
                            fusion.reconstruct(
                                timestampNanos = timestampNanos,
                                trackId = actor.actorId,
                                subjectType =
                                    actor.subjectType,
                                bodySpacePriors =
                                    bodySpacePriors(actor),
                            )
                        }.onFailure {
                            reconstructionFailureWarning =
                                "ACTOR_3D_RECONSTRUCTION_FAILURE_" +
                                    it::class.simpleName.orEmpty()
                                        .take(48)
                        }.getOrNull()

                lastProcessedTimestampNanos[
                    actor.actorId
                ] = timestampNanos

                if (skeleton != null) {
                    skeletonPredictor.observe(skeleton)
                }

                outputs +=
                    ActorResult(
                        actorId = actor.actorId,
                        skeleton = skeleton,
                        detailed = detailed,
                        reconstructed = skeleton != null,
                        warnings =
                            buildList {
                                detailedFailureWarning?.let(::add)
                                reconstructionFailureWarning?.let(::add)
                                if (skeleton == null) {
                                    add(
                                        "ACTOR_3D_RECONSTRUCTION_UNAVAILABLE",
                                    )
                                }
                            },
                    )
            }

        pruneHistory(
            timestampNanos,
            customization,
        )

        return Result(
            timestampNanos = timestampNanos,
            actors = outputs,
            primaryActorId = selectedPrimary,
            primaryDetailed =
                outputs.firstOrNull {
                    it.actorId == selectedPrimary
                }?.detailed,
            reconstructedActorCount =
                outputs.count {
                    it.reconstructed
                },
            detailedActorCount =
                outputs.count {
                    it.detailed != null
                },
            warnings =
                (
                    outputs
                        .flatMap { it.warnings } +
                        schedulerResult.warnings
                    )
                    .distinct()
                    .takeLast(32),
            detailStarvationRisk =
                schedulerResult.starvationRisk,
            detailFairnessScore =
                schedulerResult.fairnessScore,
            maximumDetailStarvationFrames =
                schedulerResult.maximumStarvationFrames,
            meanCaptureAuthorityScore =
                outputs.mapNotNull { actor ->
                    actor.detailed?.captureAuthorityScore
                }.takeIf { it.isNotEmpty() }
                    ?.average()
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0,
            authoritativeActorRatio =
                if (outputs.isEmpty()) {
                    0.0
                } else {
                    outputs.count {
                        it.reconstructed &&
                            (
                                it.detailed?.captureAuthorityBand ==
                                    HighEndCaptureAuthorityEngine.Band.HIGH_CONFIDENCE ||
                                it.detailed?.captureAuthorityBand ==
                                    HighEndCaptureAuthorityEngine.Band.AUTHORITATIVE
                            ) &&
                            it.detailed?.captureAuthorityHardBlocked == false
                    }.toDouble() /
                        outputs.size.toDouble()
                }.coerceIn(0.0, 1.0),
            minimumCrossModalAuthority =
                outputs.mapNotNull { actor ->
                    actor.detailed?.crossModalEpipolarAuthority
                }.minOrNull()?.coerceIn(0.0, 1.0) ?: 0.0,
            minimumCrossModalMetricAuthority =
                outputs.mapNotNull { actor ->
                    actor.detailed?.crossModalMetricAuthority
                }.minOrNull()?.coerceIn(0.0, 1.0) ?: 0.0,
            authorityBlockedActorCount =
                outputs.count {
                    it.detailed?.captureAuthorityHardBlocked == true
                },
            authorityWarnings =
                buildList {
                    val ratio = authoritativeActorRatio
                    val mean = meanCaptureAuthorityScore
                    val minimum = minimumCrossModalAuthority
                    val minimumMetric = minimumCrossModalMetricAuthority
                    if (outputs.isNotEmpty() && ratio < 0.80) {
                        add("MULTI_ACTOR_AUTHORITY_COVERAGE_DEGRADED")
                    }
                    if (mean > 0.0 && mean < 0.70) {
                        add("MULTI_ACTOR_CAPTURE_AUTHORITY_LOW")
                    }
                    if (minimum > 0.0 && minimum < 0.50) {
                        add("MULTI_ACTOR_CROSS_MODAL_AUTHORITY_LIMITED")
                    }
                    if (minimumMetric > 0.0 && minimumMetric < 0.50) {
                        add("MULTI_ACTOR_CROSS_MODAL_METRIC_AUTHORITY_LIMITED")
                    }
                    if (authorityBlockedActorCount > 0) {
                        add(
                            "MULTI_ACTOR_AUTHORITY_BLOCKED_ACTORS=" +
                                authorityBlockedActorCount,
                        )
                    }
                }.distinct(),
        )
    }

    fun reset() {
        lastProcessedTimestampNanos.clear()
        skeletonPredictor.reset()
        actorWorkloadScheduler.reset()
        lastPrimaryActorId = null
    }

    override fun close() {
        reset()
    }

    private fun Int.floorMod(modulus: Int): Int {
        if (modulus <= 0) return 0
        return ((this % modulus) + modulus) % modulus
    }

    private fun bodySpacePriors(
        actor: MultiActorPoseAssociationEngine.Result,
    ): List<BodySpacePriorAlignmentEngine.Prior> =
        actor.worldJoints
            .asSequence()
            .filter {
                it.confidence.isFinite() &&
                    it.confidence > 0.08 &&
                    it.xMeters.isFinite() &&
                    it.yMeters.isFinite() &&
                    it.zMeters.isFinite()
            }
            .distinctBy {
                it.landmarkId
            }
            .take(33)
            .map {
                BodySpacePriorAlignmentEngine.Prior(
                    landmarkId = it.landmarkId,
                    positionMeters = it.positionMeters,
                    confidence = it.confidence.coerceIn(0.0, 1.0),
                )
            }
            .toList()

    private fun pruneHistory(
        timestampNanos: Long,
        customization: MotionCaptureCustomization,
    ) {
        val horizonNanos =
            maxOf(
                2_000_000_000L,
                customization.multiPersonCapacityPolicy
                    .identityRetentionFrames
                    .toLong() *
                    1_000_000_000L /
                    customization.targetFps
                        .coerceIn(15, 240)
                        .toLong(),
            )

        val liveIds =
            lastProcessedTimestampNanos.keys.toSet()
        skeletonPredictor.trackedActorIds()
            .filter { it !in liveIds }
            .forEach { skeletonPredictor.forget(it) }

        lastProcessedTimestampNanos.entries.removeIf {
            timestampNanos -
                it.value >
                horizonNanos
        }

        if (
            lastProcessedTimestampNanos.size >
                customization.maxSubjects.coerceIn(1, 64)
        ) {
            val keep =
                lastProcessedTimestampNanos.entries
                    .sortedByDescending { it.value }
                    .take(
                        customization.maxSubjects
                            .coerceIn(1, 64),
                    )
                    .associate {
                        it.key to it.value
                    }

            lastProcessedTimestampNanos.clear()
            lastProcessedTimestampNanos.putAll(keep)
        }
    }
}
