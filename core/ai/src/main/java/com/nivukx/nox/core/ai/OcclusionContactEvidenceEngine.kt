package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Visibility/contact-aware evidence arbitration for markerless human motion.
 *
 * This engine does not synthesize missing pose. It quantifies which landmarks remain reliable,
 * how much prediction budget should remain, and how close-contact configurations should preserve
 * anatomical evidence without allowing a contact event to become an identity merge.
 */
class OcclusionContactEvidenceEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumLandmarks: Int = 512,
        val maximumContacts: Int = 256,
        val maximumActors: Int = 64,
        val maximumStateAgeNanos: Long = 2_000_000_000L,
        val maximumPredictionMillis: Double = 120.0,
        val hardOcclusionRatio: Double = 0.55,
        val warningOcclusionRatio: Double = 0.30,
        val minimumLandmarkConfidence: Double = 0.05,
        val contactDistanceThreshold: Double = 0.08,
        val contactConfidenceFloor: Double = 0.20,
        val visibilityMemoryAlpha: Double = 0.32,
        val uncertaintyGrowthPer100ms: Double = 0.55,
        val contactSuppressionStrength: Double = 0.82,
        val identitySuppressionStrength: Double = 0.78,
        val frequencyPriorStrength: Double = 0.18,
        val momentumPriorStrength: Double = 0.16,
        val crossViewEvidenceWeight: Double = 0.18,
        val depthEvidenceWeight: Double = 0.12,
        val appearanceEvidenceWeight: Double = 0.10,
        val motionBoundaryEvidenceWeight: Double = 0.08,
        val identitySafetyThreshold: Double = 0.62,
    ) {
        init {
            require(maximumLandmarks > 0)
            require(maximumContacts > 0)
            require(maximumActors > 0)
            require(maximumStateAgeNanos > 0L)
            require(maximumPredictionMillis > 0.0)
            require(hardOcclusionRatio in 0.0..1.0)
            require(warningOcclusionRatio in 0.0..hardOcclusionRatio)
            require(minimumLandmarkConfidence in 0.0..1.0)
            require(contactDistanceThreshold > 0.0)
            require(contactConfidenceFloor in 0.0..1.0)
            require(visibilityMemoryAlpha in 0.01..1.0)
            require(uncertaintyGrowthPer100ms >= 0.0)
            require(contactSuppressionStrength in 0.0..1.0)
            require(identitySuppressionStrength in 0.0..1.0)
            require(frequencyPriorStrength in 0.0..1.0)
            require(momentumPriorStrength in 0.0..1.0)
            require(crossViewEvidenceWeight >= 0.0)
            require(depthEvidenceWeight >= 0.0)
            require(appearanceEvidenceWeight >= 0.0)
            require(motionBoundaryEvidenceWeight >= 0.0)
            require(identitySafetyThreshold in 0.0..1.0)
        }
    }

    data class JointObservation(
        val landmarkId: Int,
        val x: Double,
        val y: Double,
        val confidence: Double,
        val visible: Boolean = true,
        val semanticGroup: String = "body",
        val crossViewSupport: Double = 0.0,
        val depthConfidence: Double = 1.0,
        val appearanceConsistency: Double = 1.0,
        val motionBoundaryConfidence: Double = 1.0,
        val temporalInnovationScore: Double = 1.0,
    ) {
        init {
            require(landmarkId >= 0)
            require(x.isFinite() && y.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(semanticGroup.isNotBlank())
            listOf(
                crossViewSupport,
                depthConfidence,
                appearanceConsistency,
                motionBoundaryConfidence,
                temporalInnovationScore,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
        }
    }

    data class ContactObservation(
        val firstLandmarkId: Int,
        val secondLandmarkId: Int,
        val distanceNormalized: Double,
        val confidence: Double,
    ) {
        init {
            require(firstLandmarkId >= 0)
            require(secondLandmarkId >= 0)
            require(firstLandmarkId != secondLandmarkId)
            require(distanceNormalized.isFinite() && distanceNormalized >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val actorId: Long,
        val landmarks: List<JointObservation>,
        val contacts: List<ContactObservation> = emptyList(),
        val periodicMotionScore: Double = 0.0,
        val motionFrequencyHz: Double = 0.0,
        val momentumConfidence: Double = 0.0,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(actorId >= 0L)
            require(periodicMotionScore in 0.0..1.0)
            require(motionFrequencyHz.isFinite() && motionFrequencyHz >= 0.0)
            require(momentumConfidence in 0.0..1.0)
        }
    }

    data class LandmarkEvidence(
        val landmarkId: Int,
        val visibilityWeight: Double,
        val confidenceWeight: Double,
        val uncertaintyMultiplier: Double,
        val predictionAllowanceMillis: Double,
        val contactProtected: Boolean,
        val semanticGroup: String,
        val crossViewSupport: Double = 0.0,
        val depthConfidence: Double = 0.0,
        val appearanceConsistency: Double = 0.0,
        val motionBoundaryConfidence: Double = 0.0,
        val temporalInnovationScore: Double = 0.0,
    ) {
        val evidenceQuality: Double
            get() = (
                crossViewSupport.coerceIn(0.0, 1.0) *
                    0.30 +
                    depthConfidence.coerceIn(0.0, 1.0) *
                    0.20 +
                    appearanceConsistency.coerceIn(0.0, 1.0) *
                    0.18 +
                    motionBoundaryConfidence.coerceIn(0.0, 1.0) *
                    0.12 +
                    temporalInnovationScore.coerceIn(0.0, 1.0) *
                    0.20
                ).coerceIn(0.0, 1.0)

        val trust: Double
            get() = (
                visibilityWeight *
                    confidenceWeight *
                    (1.0 / uncertaintyMultiplier) *
                    (0.76 + 0.24 * evidenceQuality) *
                    (if (contactProtected) 1.0 else 0.92)
                ).coerceIn(0.0, 1.0)
    }

    data class Result(
        val timestampNanos: Long,
        val actorId: Long,
        val processedLandmarkCount: Int,
        val processedContactCount: Int,
        val meanVisibility: Double,
        val criticalOcclusionRatio: Double,
        val visibilityUncertaintyMultiplier: Double,
        val predictionBudgetRemainingMillis: Double,
        val contactClusterCount: Int,
        val contactPreservationScore: Double,
        val identityMergeSuppression: Double,
        val frequencyRecoveryPrior: Double,
        val momentumRecoveryPrior: Double,
        val captureTrust: Double,
        val landmarkEvidence: List<LandmarkEvidence>,
        val warnings: List<String>,
        val meanEvidenceQuality: Double = 0.0,
        val identitySafetyScore: Double = 0.0,
    )

    private data class LandmarkState(
        var timestampNanos: Long,
        var visibilityMemory: Double,
        var confidenceMemory: Double,
        var consecutiveOccludedFrames: Int,
    )

    private data class ActorState(
        val landmarks: LinkedHashMap<Int, LandmarkState> = LinkedHashMap(),
        var timestampNanos: Long = Long.MIN_VALUE,
    )

    private val actors = LinkedHashMap<Long, ActorState>()
    private var closed = false

    @Synchronized
    fun evaluate(input: Input): Result {
        check(!closed) { "OcclusionContactEvidenceEngine is closed" }

        val existing = actors[input.actorId]
        if (
            existing != null &&
                existing.timestampNanos != Long.MIN_VALUE &&
                input.timestampNanos <= existing.timestampNanos
        ) {
            return staleResult(input, existing)
        }

        val actor = existing ?: ActorState().also {
            if (actors.size >= config.maximumActors) {
                val victim = actors.entries.minByOrNull {
                    it.value.timestampNanos
                }?.key
                if (victim != null) actors.remove(victim)
            }
            actors[input.actorId] = it
        }

        val landmarks = input.landmarks
            .asSequence()
            .filter {
                it.confidence >= config.minimumLandmarkConfidence
            }
            .distinctBy { it.landmarkId }
            .take(config.maximumLandmarks)
            .toList()

        val contacts = input.contacts
            .asSequence()
            .filter {
                it.confidence >= config.contactConfidenceFloor
            }
            .take(config.maximumContacts)
            .toList()

        val contactIds = HashSet<Int>()
        val contactClusters = ArrayList<Set<Int>>()
        for (contact in contacts) {
            if (
                contact.distanceNormalized <=
                    config.contactDistanceThreshold
            ) {
                contactIds += contact.firstLandmarkId
                contactIds += contact.secondLandmarkId
                contactClusters += setOf(
                    contact.firstLandmarkId,
                    contact.secondLandmarkId,
                )
            }
        }

        val evidence = ArrayList<LandmarkEvidence>(landmarks.size)

        var totalVisibility = 0.0
        var totalConfidence = 0.0
        var occludedCount = 0
        var evidenceQualitySum = 0.0

        for (landmark in landmarks) {
            val state =
                actor.landmarks[landmark.landmarkId]
                    ?: LandmarkState(
                        timestampNanos = input.timestampNanos,
                        visibilityMemory = if (landmark.visible) 1.0 else 0.0,
                        confidenceMemory = landmark.confidence,
                        consecutiveOccludedFrames =
                            if (landmark.visible) 0 else 1,
                    )

            val dtMillis =
                if (state.timestampNanos <= 0L) {
                    0.0
                } else {
                    (
                        (
                            input.timestampNanos -
                                state.timestampNanos
                            ).coerceAtLeast(0L) / 1e6
                        ).coerceIn(0.0, 10_000.0)
                }

            val observedVisibility =
                if (landmark.visible && landmark.confidence >=
                    config.minimumLandmarkConfidence
                ) {
                    1.0
                } else {
                    0.0
                }

            state.visibilityMemory +=
                (
                    observedVisibility -
                        state.visibilityMemory
                    ) * config.visibilityMemoryAlpha

            state.confidenceMemory +=
                (
                    landmark.confidence -
                        state.confidenceMemory
                    ) * config.visibilityMemoryAlpha

            state.consecutiveOccludedFrames =
                if (observedVisibility > 0.5) {
                    0
                } else {
                    state.consecutiveOccludedFrames + 1
                }

            state.timestampNanos = input.timestampNanos
            actor.landmarks[landmark.landmarkId] = state

            if (observedVisibility < 0.5) {
                occludedCount++
            }

            totalVisibility += state.visibilityMemory
            totalConfidence += state.confidenceMemory

            val occlusionAgeFactor =
                (
                    state.consecutiveOccludedFrames / 6.0
                ).coerceIn(0.0, 1.0)

            val uncertaintyMultiplier =
                (
                    1.0 +
                        occlusionAgeFactor *
                            config.uncertaintyGrowthPer100ms *
                            max(
                                1.0,
                                dtMillis / 100.0,
                            )
                    ).coerceAtLeast(1.0)

            val predictionAllowance =
                (
                    config.maximumPredictionMillis *
                        state.visibilityMemory *
                        (
                            1.0 -
                                0.55 *
                                    occlusionAgeFactor
                            )
                    ).coerceIn(
                        0.0,
                        config.maximumPredictionMillis,
                    )

            val evidenceQuality = (
                landmark.crossViewSupport * 0.30 +
                    landmark.depthConfidence * 0.20 +
                    landmark.appearanceConsistency * 0.18 +
                    landmark.motionBoundaryConfidence * 0.12 +
                    landmark.temporalInnovationScore * 0.20
                ).coerceIn(0.0, 1.0)

            evidenceQualitySum += evidenceQuality

            evidence +=
                LandmarkEvidence(
                    landmarkId = landmark.landmarkId,
                    visibilityWeight =
                        state.visibilityMemory.coerceIn(0.0, 1.0),
                    confidenceWeight =
                        state.confidenceMemory.coerceIn(0.0, 1.0),
                    uncertaintyMultiplier = uncertaintyMultiplier,
                    predictionAllowanceMillis = predictionAllowance,
                    contactProtected = landmark.landmarkId in contactIds,
                    semanticGroup = landmark.semanticGroup,
                    crossViewSupport =
                        landmark.crossViewSupport,
                    depthConfidence =
                        landmark.depthConfidence,
                    appearanceConsistency =
                        landmark.appearanceConsistency,
                    motionBoundaryConfidence =
                        landmark.motionBoundaryConfidence,
                    temporalInnovationScore =
                        landmark.temporalInnovationScore,
                )
        }

        pruneActor(actor, input.timestampNanos)

        val meanVisibility =
            if (landmarks.isEmpty()) {
                0.0
            } else {
                totalVisibility /
                    landmarks.size.toDouble()
            }

        val criticalOcclusionRatio =
            if (landmarks.isEmpty()) {
                1.0
            } else {
                occludedCount.toDouble() /
                    landmarks.size.toDouble()
            }

        val visibilityUncertaintyMultiplier =
            (
                1.0 +
                    (1.0 - meanVisibility) *
                        (
                            0.85 +
                                criticalOcclusionRatio *
                                    0.85
                            )
                ).coerceIn(1.0, 4.0)

        val frequencyRecoveryPrior =
            (
                input.periodicMotionScore *
                    config.frequencyPriorStrength +
                    if (input.motionFrequencyHz > 0.1) {
                        0.25
                    } else {
                        0.0
                    }
                ).coerceIn(0.0, 1.0)

        val momentumRecoveryPrior =
            (
                input.momentumConfidence *
                    config.momentumPriorStrength
                ).coerceIn(0.0, 1.0)

        val predictionBudgetRemaining =
            (
                config.maximumPredictionMillis *
                    (
                        0.25 +
                            0.75 * meanVisibility +
                            0.20 * frequencyRecoveryPrior +
                            0.16 * momentumRecoveryPrior
                        ) *
                    (
                        1.0 -
                            criticalOcclusionRatio * 0.70
                        )
                ).coerceIn(0.0, config.maximumPredictionMillis)

        val contactPreservation =
            if (contactClusters.isEmpty()) {
                1.0
            } else {
                contacts.map { contact ->
                    val distanceFactor =
                        1.0 -
                            (
                                contact.distanceNormalized /
                                    (config.contactDistanceThreshold * 2.0)
                            ).coerceIn(0.0, 1.0)
                    (
                        distanceFactor * 0.65 +
                            contact.confidence * 0.35
                        ).coerceIn(0.0, 1.0)
                }.average()
            }

        val identityMergeSuppression =
            (
                contactClusters.size
                    .coerceAtMost(4)
                    .toDouble() /
                    4.0
            ).coerceIn(0.0, 1.0) *
                config.identitySuppressionStrength +
                (
                    if (contactClusters.isEmpty()) {
                        0.0
                    } else {
                        1.0
                    } * config.contactSuppressionStrength * 0.5
                )
            .coerceIn(0.0, 1.0)

        val meanEvidenceQuality =
            if (landmarks.isEmpty()) {
                0.0
            } else {
                evidenceQualitySum /
                    landmarks.size.toDouble()
            }

        val identitySafetyScore =
            (
                identityMergeSuppression * 0.46 +
                    contactPreservation * 0.24 +
                    meanEvidenceQuality * 0.18 +
                    (
                        1.0 -
                            criticalOcclusionRatio
                        ) * 0.12
                ).coerceIn(0.0, 1.0)

        val occlusionTrust =
            (
                meanVisibility * 0.32 +
                    (1.0 - criticalOcclusionRatio) * 0.24 +
                    contactPreservation * 0.10 +
                    meanEvidenceQuality * 0.16 +
                    frequencyRecoveryPrior * 0.07 +
                    momentumRecoveryPrior * 0.05 +
                    (
                        predictionBudgetRemaining /
                            config.maximumPredictionMillis
                    ) * 0.06
                ).coerceIn(0.0, 1.0)

        val captureTrust =
            (
                occlusionTrust * 0.52 +
                    (
                        totalConfidence /
                            landmarks.size.coerceAtLeast(1)
                    ) * 0.22 +
                    meanEvidenceQuality * 0.16 +
                    identitySafetyScore * 0.10
                ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (criticalOcclusionRatio >= config.hardOcclusionRatio) {
                add("persistent-occlusion")
            } else if (
                criticalOcclusionRatio >= config.warningOcclusionRatio
            ) {
                add("occlusion-elevated")
            }
            if (predictionBudgetRemaining <=
                config.maximumPredictionMillis * 0.30
            ) {
                add("prediction-budget-near-exhaustion")
            }
            if (contactClusters.isNotEmpty()) {
                add("contact-topology-protected")
            }
            if (identityMergeSuppression >= 0.70) {
                add("identity-merge-suppressed")
            }
            if (
                identitySafetyScore <
                    config.identitySafetyThreshold
            ) {
                add("identity-safety-evidence-degraded")
            }
            if (meanEvidenceQuality < 0.45) {
                add("physical-evidence-quality-degraded")
            }
            if (landmarks.size < 8) {
                add("sparse-landmark-support")
            }
        }

        actor.timestampNanos = input.timestampNanos

        pruneActors(input.timestampNanos)

        return Result(
            timestampNanos = input.timestampNanos,
            actorId = input.actorId,
            processedLandmarkCount = landmarks.size,
            processedContactCount = contacts.size,
            meanVisibility = meanVisibility.coerceIn(0.0, 1.0),
            criticalOcclusionRatio =
                criticalOcclusionRatio.coerceIn(0.0, 1.0),
            visibilityUncertaintyMultiplier =
                visibilityUncertaintyMultiplier,
            predictionBudgetRemainingMillis =
                predictionBudgetRemaining,
            contactClusterCount = contactClusters.size,
            contactPreservationScore =
                contactPreservation.coerceIn(0.0, 1.0),
            identityMergeSuppression =
                identityMergeSuppression.coerceIn(0.0, 1.0),
            frequencyRecoveryPrior =
                frequencyRecoveryPrior.coerceIn(0.0, 1.0),
            momentumRecoveryPrior =
                momentumRecoveryPrior.coerceIn(0.0, 1.0),
            captureTrust = captureTrust.coerceIn(0.0, 1.0),
            landmarkEvidence = evidence,
            warnings = warnings.distinct(),
            meanEvidenceQuality =
                meanEvidenceQuality.coerceIn(0.0, 1.0),
            identitySafetyScore =
                identitySafetyScore.coerceIn(0.0, 1.0),
        )
    }

    fun trackedActorCount(): Int = actors.size

    @Synchronized
    fun reset() {
        actors.clear()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        actors.clear()
        closed = true
    }

    private fun staleResult(
        input: Input,
        existing: ActorState,
    ): Result {
        val latestLandmarks =
            input.landmarks
                .take(config.maximumLandmarks)

        return Result(
            timestampNanos = existing.timestampNanos,
            actorId = input.actorId,
            processedLandmarkCount = 0,
            processedContactCount = 0,
            meanVisibility =
                existing.landmarks.values
                    .map { it.visibilityMemory }
                    .averageOrZero(),
            criticalOcclusionRatio = 0.0,
            visibilityUncertaintyMultiplier = 1.0,
            predictionBudgetRemainingMillis =
                config.maximumPredictionMillis,
            contactClusterCount = 0,
            contactPreservationScore = 1.0,
            identityMergeSuppression = 0.0,
            frequencyRecoveryPrior = 0.0,
            momentumRecoveryPrior = 0.0,
            captureTrust =
                existing.landmarks.values
                    .map { it.confidenceMemory }
                    .averageOrZero(),
            landmarkEvidence = latestLandmarks.map { landmark ->
                LandmarkEvidence(
                    landmarkId = landmark.landmarkId,
                    visibilityWeight =
                        existing.landmarks[landmark.landmarkId]
                            ?.visibilityMemory
                            ?: 0.0,
                    confidenceWeight =
                        existing.landmarks[landmark.landmarkId]
                            ?.confidenceMemory
                            ?: 0.0,
                    uncertaintyMultiplier = 1.0,
                    predictionAllowanceMillis =
                        config.maximumPredictionMillis,
                    contactProtected = false,
                    semanticGroup = landmark.semanticGroup,
                    crossViewSupport = 0.0,
                    depthConfidence = landmark.depthConfidence,
                    appearanceConsistency =
                        landmark.appearanceConsistency,
                    motionBoundaryConfidence =
                        landmark.motionBoundaryConfidence,
                    temporalInnovationScore =
                        landmark.temporalInnovationScore,
                )
            },
            warnings = listOf(
                "timestamp-reversal",
                "stale-frame-rejected",
            ),
        )
    }

    private fun pruneActor(
        actor: ActorState,
        timestampNanos: Long,
    ) {
        actor.landmarks.entries.removeIf {
            timestampNanos -
                it.value.timestampNanos >
                config.maximumStateAgeNanos
        }
        while (actor.landmarks.size > config.maximumLandmarks) {
            actor.landmarks.entries.firstOrNull()?.let {
                actor.landmarks.remove(it.key)
            } ?: break
        }
    }

    private fun pruneActors(
        timestampNanos: Long,
    ) {
        val iterator = actors.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                entry.value.timestampNanos != Long.MIN_VALUE &&
                timestampNanos -
                    entry.value.timestampNanos >
                    config.maximumStateAgeNanos
            ) {
                iterator.remove()
            }
        }
        while (actors.size > config.maximumActors) {
            actors.entries.firstOrNull()?.let {
                actors.remove(it.key)
            } ?: break
        }
    }

    private fun Iterable<Double>.averageOrZero(): Double {
        val values = toList()
        return if (values.isEmpty()) 0.0 else values.average()
    }
}
