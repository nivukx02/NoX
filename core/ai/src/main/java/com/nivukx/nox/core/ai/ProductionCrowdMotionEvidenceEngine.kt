package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Production crowd-motion evidence authority.
 *
 * The purpose of this engine is not to invent additional pose detections. It turns already measured
 * actor tracks into a bounded, uncertainty-aware scheduling and quality signal for dense scenes.
 *
 * Design goals:
 * - preserve all admitted actors instead of allowing a few visually dominant actors to monopolize
 *   expensive reconstruction work;
 * - separate subject motion from common-mode camera motion;
 * - detect close-proximity / interaction pressure before identity swaps become visible downstream;
 * - score short prediction bridges conservatively;
 * - retain a bounded temporal digest rather than image payloads;
 * - produce deterministic decisions from the same evidence;
 * - expose failure attribution instead of collapsing everything into one confidence number.
 *
 * This engine is intentionally model-agnostic and Android-free. It can therefore be reused by the
 * live perception path, post-capture processing path, and native bridge without coupling those
 * surfaces together.
 */
class ProductionCrowdMotionEvidenceEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class ActorState {
        OBSERVED,
        PREDICTED,
        RECOVERING,
        HOLD,
        LOST,
    }

    enum class Bottleneck {
        NONE,
        DETECTOR_CONFIDENCE,
        VISIBILITY,
        TEMPORAL_INNOVATION,
        IDENTITY_RISK,
        CROWD_DENSITY,
        CAMERA_MOTION,
        TIMING,
        RUNTIME_PRESSURE,
    }

    enum class InteractionLevel {
        ISOLATED,
        NEAR,
        CONTACT_RISK,
        HEAVY_OCCLUSION,
    }

    data class Config(
        val maximumActors: Int = 50,
        val hardActorCeiling: Int = 64,
        val historyFrames: Int = 24,
        val staleStateFrames: Int = 120,
        val minimumActorConfidence: Double = 0.18,
        val minimumVisibility: Double = 0.12,
        val minimumPredictionConfidence: Double = 0.16,
        val nearDistanceNormalized: Double = 0.20,
        val contactDistanceNormalized: Double = 0.09,
        val heavyOverlapIou: Double = 0.48,
        val identityRiskPredictionWeight: Double = 0.26,
        val identityRiskInnovationWeight: Double = 0.24,
        val identityRiskCrowdWeight: Double = 0.26,
        val identityRiskOcclusionWeight: Double = 0.16,
        val identityRiskTimingWeight: Double = 0.08,
        val cameraMotionSuppression: Double = 0.72,
        val commonModeRobustness: Double = 0.80,
        val minimumFairnessWeight: Double = 0.55,
        val maximumFairnessWeight: Double = 1.50,
        val minimumSchedulingScore: Double = 0.20,
        val holdScore: Double = 0.26,
        val recoveryScore: Double = 0.46,
        val observedScore: Double = 0.60,
        val robustOutlierSigma: Double = 3.5,
        val historyHalfLifeFrames: Double = 18.0,
    ) {
        init {
            require(maximumActors in 1..hardActorCeiling)
            require(hardActorCeiling in maximumActors..64)
            require(historyFrames >= 4)
            require(staleStateFrames >= historyFrames)
            require(minimumActorConfidence in 0.0..1.0)
            require(minimumVisibility in 0.0..1.0)
            require(minimumPredictionConfidence in 0.0..1.0)
            require(nearDistanceNormalized > 0.0)
            require(contactDistanceNormalized > 0.0)
            require(heavyOverlapIou in 0.0..1.0)
            require(identityRiskPredictionWeight >= 0.0)
            require(identityRiskInnovationWeight >= 0.0)
            require(identityRiskCrowdWeight >= 0.0)
            require(identityRiskOcclusionWeight >= 0.0)
            require(identityRiskTimingWeight >= 0.0)
            require(cameraMotionSuppression in 0.0..1.0)
            require(commonModeRobustness in 0.0..1.0)
            require(minimumFairnessWeight > 0.0)
            require(maximumFairnessWeight >= minimumFairnessWeight)
            require(minimumSchedulingScore in 0.0..1.0)
            require(holdScore in 0.0..1.0)
            require(recoveryScore in holdScore..1.0)
            require(observedScore in recoveryScore..1.0)
            require(robustOutlierSigma > 0.0)
            require(historyHalfLifeFrames > 0.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val actors: List<MultiActorPoseAssociationEngine.Result>,
        val cameraMotionRisk: Double = 0.0,
        val timingConfidence: Double = 1.0,
        val runtimeHeadroom: Double = 1.0,
        val segmentationEvidence: Double = 0.0,
        val segmentationCoverage: Double = 0.0,
        val actorSilhouetteEvidence:
            Map<Long, ActorSilhouetteEvidenceEngine.Evidence> =
            emptyMap(),
    ) {
        init {
            require(timestampNanos >= 0L)
            require(cameraMotionRisk in 0.0..1.0)
            require(timingConfidence in 0.0..1.0)
            require(runtimeHeadroom in 0.0..1.0)
            require(segmentationEvidence in 0.0..1.0)
            require(segmentationCoverage in 0.0..1.0)
        }
    }

    data class PairInteraction(
        val actorA: Long,
        val actorB: Long,
        val centerDistanceNormalized: Double,
        val iou: Double,
        val level: InteractionLevel,
        val occlusionRisk: Double,
    )

    data class ActorEvidence(
        val actorId: Long,
        val state: ActorState,
        val confidence: Double,
        val visibility: Double,
        val predictionConfidence: Double,
        val temporalStability: Double,
        val innovationNormalized: Double,
        val speedNormalizedPerSecond: Double,
        val accelerationNormalizedPerSecond2: Double,
        val motionEnergy: Double,
        val cameraCompensatedMotionEnergy: Double,
        val crowdPressure: Double,
        val identityRisk: Double,
        val interactionLevel: InteractionLevel,
        val worldLandmarkCoverage: Double,
        val worldBoneConsistency: Double,
        val worldGeometryReliability: Double,
        val fairnessWeight: Double,
        val schedulingScore: Double,
        val recommendedBodyDetail: Double,
        val recommendedFaceDetail: Double,
        val recommendedEyeDetail: Double,
        val recommendedHandDetail: Double,
        val recommendedSurfaceDetail: Double,
        val holdDenseDetail: Boolean,
        val silhouetteSupport: Double = 0.0,
        val silhouetteAmbiguity: Double = 0.0,
        val silhouetteReliability: Double = 0.0,
        val appearanceQuality: Double = 0.0,
        val warnings: List<String>,
    )

    data class FrameResult(
        val timestampNanos: Long,
        val actors: List<ActorEvidence>,
        val interactions: List<PairInteraction>,
        val activeActorCount: Int,
        val predictedActorCount: Int,
        val recoveringActorCount: Int,
        val heldActorCount: Int,
        val meanConfidence: Double,
        val meanVisibility: Double,
        val meanIdentitySafety: Double,
        val commonModeVelocityNormalizedPerSecond: Double,
        val differentialMotionScore: Double,
        val cameraCompensatedMotionScore: Double,
        val crowdPressure: Double,
        val interactionPressure: Double,
        val meanWorldLandmarkCoverage: Double,
        val meanWorldBoneConsistency: Double,
        val meanWorldGeometryReliability: Double,
        val segmentationEvidence: Double,
        val segmentationCoverage: Double,
        val actorSilhouetteEvidenceAvailable: Boolean = false,
        val meanActorSilhouetteSupport: Double = 0.0,
        val meanActorSilhouetteAmbiguity: Double = 0.0,
        val meanAppearanceQuality: Double = 0.0,
        val timingConfidence: Double,
        val runtimeHeadroom: Double,
        val fairnessSaturation: Double,
        val globalSchedulingCapacity: Double,
        val bottleneck: Bottleneck,
        val authoritative: Boolean,
        val warnings: List<String>,
        val diagnostics: List<String>,
    )

    data class SchedulerSlice(
        val actorId: Long,
        val bodyBudget: Double,
        val faceBudget: Double,
        val eyeBudget: Double,
        val handBudget: Double,
        val surfaceBudget: Double,
        val deferredSurface: Boolean,
        val deferredFaceEye: Boolean,
        val priorityRank: Int,
    )

    data class SchedulerDecision(
        val timestampNanos: Long,
        val slices: List<SchedulerSlice>,
        val preservedActorCount: Int,
        val deferredActorCount: Int,
        val minimumActorBudget: Double,
        val maximumActorBudget: Double,
        val warnings: List<String>,
    )

    private data class HistoryDigest(
        val timestampNanos: Long,
        val center: Vec2,
        val speed: Double,
        val acceleration: Double,
        val confidence: Double,
        val identitySafety: Double,
    )

    private data class WorldGeometryEvidence(
        val coverage: Double,
        val boneConsistency: Double,
        val bodyScaleMeters: Double,
        val reliability: Double,
    )

    private data class MutableMemory(
        var lastTimestampNanos: Long = Long.MIN_VALUE,
        var ageFrames: Int = 0,
        var deferredFrames: Int = 0,
        var recoveryFrames: Int = 0,
        var lastState: ActorState = ActorState.LOST,
        val history: ArrayDeque<HistoryDigest> = ArrayDeque(),
    )

    private val memories = LinkedHashMap<Long, MutableMemory>()
    private var latest: FrameResult? = null
    private var latestScheduler: SchedulerDecision? = null
    private var lastTimestampNanos = Long.MIN_VALUE
    private var closed = false

    @Synchronized
    fun evaluate(input: Input): FrameResult {
        check(!closed) { "ProductionCrowdMotionEvidenceEngine is closed" }

        if (lastTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos <= lastTimestampNanos
        ) {
            return latest ?: emptyResult(
                input.timestampNanos,
                "NON_MONOTONIC_TIMESTAMP_REJECTED",
            )
        }

        lastTimestampNanos = input.timestampNanos

        val boundedActors =
            input.actors
                .asSequence()
                .filter { it.actorId > 0L }
                .take(config.hardActorCeiling)
                .toList()

        val visibleActors =
            boundedActors.filter {
                it.confidence >= config.minimumActorConfidence &&
                    visibility(it) >= config.minimumVisibility
            }

        val commonModeVelocity =
            robustCommonModeVelocity(
                visibleActors,
                input.cameraMotionRisk,
            )

        val interactions =
            buildInteractions(boundedActors)

        val interactionByActor =
            HashMap<Long, MutableList<PairInteraction>>()
        interactions.forEach { interaction ->
            interactionByActor
                .getOrPut(interaction.actorA) { mutableListOf() }
                .add(interaction)
            interactionByActor
                .getOrPut(interaction.actorB) { mutableListOf() }
                .add(interaction)
        }

        val actorEvidence =
            boundedActors.map { actor ->
                val memory =
                    memories.getOrPut(actor.actorId) { MutableMemory() }

                memory.ageFrames =
                    (memory.ageFrames + 1)
                        .coerceAtMost(config.staleStateFrames)

                val interactionsForActor =
                    interactionByActor[actor.actorId].orEmpty()

                val crowdPressure =
                    actorCrowdPressure(
                        actor = actor,
                        interactions = interactionsForActor,
                    )

                val worldGeometry =
                    worldGeometryEvidence(actor)

                val silhouetteEvidence =
                    input.actorSilhouetteEvidence[actor.actorId]

                val identityRisk =
                    identityRisk(
                        actor = actor,
                        crowdPressure = crowdPressure,
                        input = input,
                        interactions = interactionsForActor,
                        worldGeometry = worldGeometry,
                    )

                val state =
                    classifyState(
                        actor = actor,
                        identityRisk = identityRisk,
                        input = input,
                        memory = memory,
                    )

                if (state == ActorState.RECOVERING) {
                    memory.recoveryFrames =
                        (memory.recoveryFrames + 1).coerceAtMost(240)
                } else {
                    memory.recoveryFrames = 0
                }

                val speed =
                    actor.microMotion.velocityNormalizedPerSecond
                        .coerceAtLeast(0.0)

                val acceleration =
                    actor.microMotion.accelerationNormalizedPerSecond2
                        .coerceAtLeast(0.0)

                val rawMotionEnergy =
                    motionEnergy(
                        speed = speed,
                        acceleration = acceleration,
                        confidence = actor.confidence,
                    )

                val commonMode =
                    commonModeMagnitudeFor(
                        actor = actor,
                        commonVelocity = commonModeVelocity,
                    )

                val compensated =
                    (
                        rawMotionEnergy -
                            commonMode *
                                config.cameraMotionSuppression
                    )
                        .coerceAtLeast(0.0)
                        .coerceIn(0.0, 1.0)

                val fairness =
                    fairnessWeight(memory, actor, input.runtimeHeadroom)

                val scheduling =
                    schedulingScore(
                        actor = actor,
                        state = state,
                        identityRisk = identityRisk,
                        compensatedMotion = compensated,
                        crowdPressure = crowdPressure,
                        fairness = fairness,
                        timingConfidence = input.timingConfidence,
                        runtimeHeadroom = input.runtimeHeadroom,
                        silhouetteEvidence = silhouetteEvidence,
                    )

                val detail =
                    allocateDetail(
                        schedulingScore = scheduling,
                        identityRisk = identityRisk,
                        crowdPressure = crowdPressure,
                        state = state,
                        runtimeHeadroom = input.runtimeHeadroom,
                        silhouetteEvidence = silhouetteEvidence,
                    )

                val evidence =
                    ActorEvidence(
                        actorId = actor.actorId,
                        state = state,
                        confidence = actor.confidence.coerceIn(0.0, 1.0),
                        visibility = visibility(actor),
                        predictionConfidence =
                            actor.predictionConfidence.coerceIn(0.0, 1.0),
                        temporalStability =
                            temporalStability(actor),
                        innovationNormalized =
                            actor.meanMotionInnovationNormalized
                                .coerceAtLeast(0.0),
                        speedNormalizedPerSecond = speed,
                        accelerationNormalizedPerSecond2 = acceleration,
                        motionEnergy = rawMotionEnergy,
                        cameraCompensatedMotionEnergy = compensated,
                        crowdPressure = crowdPressure,
                        identityRisk = identityRisk,
                        interactionLevel =
                            interactionsForActor
                                .maxByOrNull { interactionSeverity(it.level) }
                                ?.level
                                ?: InteractionLevel.ISOLATED,
                        worldLandmarkCoverage = worldGeometry.coverage,
                        worldBoneConsistency = worldGeometry.boneConsistency,
                        worldGeometryReliability = worldGeometry.reliability,
                        fairnessWeight = fairness,
                        schedulingScore = scheduling,
                        recommendedBodyDetail = detail.body,
                        recommendedFaceDetail = detail.face,
                        recommendedEyeDetail = detail.eye,
                        recommendedHandDetail = detail.hand,
                        recommendedSurfaceDetail = detail.surface,
                        holdDenseDetail = detail.holdSurface,
                        silhouetteSupport =
                            silhouetteEvidence?.foregroundSupport
                                ?.coerceIn(0.0, 1.0) ?: 0.0,
                        silhouetteAmbiguity =
                            silhouetteEvidence?.overlapAmbiguity
                                ?.coerceIn(0.0, 1.0) ?: 0.0,
                        silhouetteReliability =
                            silhouetteEvidence?.reliability
                                ?.coerceIn(0.0, 1.0) ?: 0.0,
                        appearanceQuality =
                            actor.appearanceQuality
                                .coerceIn(0.0, 1.0),
                        warnings = (
                            actorWarnings(
                            actor = actor,
                            state = state,
                            identityRisk = identityRisk,
                            crowdPressure = crowdPressure,
                            input = input,
                            ) +
                            silhouetteEvidence?.warnings.orEmpty()
                        ).distinct().takeLast(16),
                    )

                if (scheduling < config.minimumSchedulingScore) {
                    memory.deferredFrames =
                        (memory.deferredFrames + 1).coerceAtMost(240)
                } else {
                    memory.deferredFrames = 0
                }

                memory.lastState = state
                memory.lastTimestampNanos = input.timestampNanos
                appendHistory(
                    memory = memory,
                    actor = actor,
                    evidence = evidence,
                )

                evidence
            }

        pruneMemories(
            activeActorIds = boundedActors.map { it.actorId }.toHashSet(),
        )

        val active =
            actorEvidence.count {
                it.state == ActorState.OBSERVED
            }

        val predicted =
            actorEvidence.count {
                it.state == ActorState.PREDICTED
            }

        val recovering =
            actorEvidence.count {
                it.state == ActorState.RECOVERING
            }

        val held =
            actorEvidence.count {
                it.state == ActorState.HOLD
            }

        val meanConfidence =
            actorEvidence
                .map { it.confidence }
                .averageOrZero()

        val meanVisibility =
            actorEvidence
                .map { it.visibility }
                .averageOrZero()

        val meanIdentitySafety =
            actorEvidence
                .map { 1.0 - it.identityRisk }
                .averageOrZero()

        val meanWorldLandmarkCoverage =
            actorEvidence
                .map { it.worldLandmarkCoverage }
                .averageOrZero()

        val meanWorldBoneConsistency =
            actorEvidence
                .map { it.worldBoneConsistency }
                .averageOrZero()

        val meanWorldGeometryReliability =
            actorEvidence
                .map { it.worldGeometryReliability }
                .averageOrZero()

        val segmentationEvidence =
            input.segmentationEvidence.coerceIn(0.0, 1.0)

        val segmentationCoverage =
            input.segmentationCoverage.coerceIn(0.0, 1.0)

        val actorSilhouetteAvailable =
            input.actorSilhouetteEvidence.isNotEmpty()

        val meanActorSilhouetteSupport =
            if (actorSilhouetteAvailable) {
                actorEvidence
                    .map { it.silhouetteSupport }
                    .averageOrZero()
                    .coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val meanActorSilhouetteAmbiguity =
            if (actorSilhouetteAvailable) {
                actorEvidence
                    .map { it.silhouetteAmbiguity }
                    .averageOrZero()
                    .coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val meanAppearanceQuality =
            if (actorEvidence.isEmpty()) {
                0.0
            } else {
                actorEvidence
                    .map { it.appearanceQuality }
                    .average()
                    .coerceIn(0.0, 1.0)
            }

        val differentialMotion =
            actorEvidence
                .map { it.cameraCompensatedMotionEnergy }
                .averageOrZero()

        val crowdPressure =
            actorEvidence
                .map { it.crowdPressure }
                .averageOrZero()

        val interactionPressure =
            interactions
                .map { it.occlusionRisk }
                .averageOrZero()

        val fairnessSaturation =
            if (actorEvidence.isEmpty()) {
                0.0
            } else {
                val range =
                    config.maximumFairnessWeight -
                        config.minimumFairnessWeight
                actorEvidence
                    .map {
                        if (range <= 1e-9) {
                            1.0
                        } else {
                            (
                                it.fairnessWeight -
                                    config.minimumFairnessWeight
                            ) / range
                        }
                    }
                    .average()
                    .coerceIn(0.0, 1.0)
            }

        val globalSchedulingCapacity =
            (
                input.runtimeHeadroom *
                    input.timingConfidence *
                    (
                        0.52 +
                            meanIdentitySafety * 0.18 +
                            meanVisibility * 0.14 +
                            (1.0 - crowdPressure) * 0.08 +
                            (1.0 - interactionPressure) * 0.08
                    )
                ) *
                (
                    0.90 +
                        segmentationEvidence * 0.10
                )
                .coerceIn(0.0, 1.0)

        val bottleneck =
            bottleneck(
                meanConfidence = meanConfidence,
                meanVisibility = meanVisibility,
                meanIdentitySafety = meanIdentitySafety,
                crowdPressure = crowdPressure,
                interactionPressure = interactionPressure,
                cameraMotionRisk = input.cameraMotionRisk,
                timingConfidence = input.timingConfidence,
                runtimeHeadroom = input.runtimeHeadroom,
            )

        val warnings =
            buildWarnings(
                actorEvidence = actorEvidence,
                interactions = interactions,
                bottleneck = bottleneck,
                input = input,
            )

        val diagnostics =
            listOf(
                "actors=" + actorEvidence.size,
                "observed=" + active,
                "predicted=" + predicted,
                "recovering=" + recovering,
                "held=" + held,
                "interactions=" + interactions.size,
                "meanConfidence=" + metric(meanConfidence),
                "meanVisibility=" + metric(meanVisibility),
                "identitySafety=" + metric(meanIdentitySafety),
                "worldCoverage=" + metric(meanWorldLandmarkCoverage),
                "worldBoneConsistency=" + metric(meanWorldBoneConsistency),
                "worldGeometryReliability=" + metric(meanWorldGeometryReliability),
                "segmentationEvidence=" + metric(segmentationEvidence),
                "segmentationCoverage=" + metric(segmentationCoverage),
                "actorSilhouetteAvailable=" + actorSilhouetteAvailable,
                "meanActorSilhouetteSupport=" +
                    metric(meanActorSilhouetteSupport),
                "meanActorSilhouetteAmbiguity=" +
                    metric(meanActorSilhouetteAmbiguity),
                "meanAppearanceQuality=" +
                    metric(meanAppearanceQuality),
                "commonModeVelocity=" + metric(commonModeVelocity),
                "differentialMotion=" + metric(differentialMotion),
                "crowdPressure=" + metric(crowdPressure),
                "interactionPressure=" + metric(interactionPressure),
                "fairnessSaturation=" + metric(fairnessSaturation),
                "schedulingCapacity=" + metric(globalSchedulingCapacity),
                "bottleneck=" + bottleneck.name,
            )

        val authoritative =
            actorEvidence.isNotEmpty() &&
                meanConfidence >= 0.40 &&
                meanVisibility >= 0.35 &&
                meanIdentitySafety >= 0.55 &&
                input.timingConfidence >= 0.45 &&
                input.runtimeHeadroom >= 0.25

        val result =
            FrameResult(
                timestampNanos = input.timestampNanos,
                actors = actorEvidence.sortedBy { it.actorId },
                interactions = interactions,
                activeActorCount = active,
                predictedActorCount = predicted,
                recoveringActorCount = recovering,
                heldActorCount = held,
                meanConfidence = meanConfidence,
                meanVisibility = meanVisibility,
                meanIdentitySafety = meanIdentitySafety,
                meanWorldLandmarkCoverage =
                    meanWorldLandmarkCoverage,
                meanWorldBoneConsistency =
                    meanWorldBoneConsistency,
                meanWorldGeometryReliability =
                    meanWorldGeometryReliability,
                segmentationEvidence = segmentationEvidence,
                segmentationCoverage = segmentationCoverage,
                actorSilhouetteEvidenceAvailable =
                    actorSilhouetteAvailable,
                meanActorSilhouetteSupport =
                    meanActorSilhouetteSupport,
                meanActorSilhouetteAmbiguity =
                    meanActorSilhouetteAmbiguity,
                meanAppearanceQuality =
                    meanAppearanceQuality,
                commonModeVelocityNormalizedPerSecond =
                    commonModeVelocity,
                differentialMotionScore = differentialMotion,
                cameraCompensatedMotionScore = differentialMotion,
                crowdPressure = crowdPressure,
                interactionPressure = interactionPressure,
                timingConfidence = input.timingConfidence,
                runtimeHeadroom = input.runtimeHeadroom,
                fairnessSaturation = fairnessSaturation,
                globalSchedulingCapacity = globalSchedulingCapacity,
                bottleneck = bottleneck,
                authoritative = authoritative,
                warnings = warnings,
                diagnostics = diagnostics,
            )

        latest = result
        latestScheduler = buildSchedulerDecision(result)
        return result
    }

    @Synchronized
    fun latest(): FrameResult? = latest

    @Synchronized
    fun scheduler(): SchedulerDecision? = latestScheduler

    @Synchronized
    fun actorHistory(actorId: Long): List<FrameDigest> =
        memories[actorId]
            ?.history
            ?.mapIndexed { index, digest ->
                FrameDigest(
                    sequenceIndex = index,
                    timestampNanos = digest.timestampNanos,
                    center = digest.center,
                    speed = digest.speed,
                    acceleration = digest.acceleration,
                    confidence = digest.confidence,
                    identitySafety = digest.identitySafety,
                )
            }
            .orEmpty()

    data class FrameDigest(
        val sequenceIndex: Int,
        val timestampNanos: Long,
        val center: Vec2,
        val speed: Double,
        val acceleration: Double,
        val confidence: Double,
        val identitySafety: Double,
    )

    @Synchronized
    fun reset() {
        check(!closed) { "ProductionCrowdMotionEvidenceEngine is closed" }
        memories.clear()
        latest = null
        latestScheduler = null
        lastTimestampNanos = Long.MIN_VALUE
    }

    override fun close() {
        if (closed) return
        memories.clear()
        latest = null
        latestScheduler = null
        closed = true
    }

    private data class DetailAllocation(
        val body: Double,
        val face: Double,
        val eye: Double,
        val hand: Double,
        val surface: Double,
        val holdSurface: Boolean,
    )

    private fun visibility(
        actor: MultiActorPoseAssociationEngine.Result,
    ): Double {
        if (actor.joints.isEmpty()) return 0.0
        return (
            actor.joints.count { it.confidence >= 0.35 }.toDouble() /
                actor.joints.size.coerceAtLeast(1).toDouble()
        ).coerceIn(0.0, 1.0)
    }

    private fun temporalStability(
        actor: MultiActorPoseAssociationEngine.Result,
    ): Double {
        val motion = actor.microMotion
        val accelerationPenalty =
            (
                motion.accelerationNormalizedPerSecond2 /
                    24.0
            ).coerceIn(0.0, 1.0)
        val innovationPenalty =
            (
                actor.meanMotionInnovationNormalized /
                    0.12
            ).coerceIn(0.0, 1.0)

        return (
            actor.predictionConfidence * 0.28 +
                actor.confidence * 0.42 +
                (1.0 - accelerationPenalty) * 0.12 +
                (1.0 - innovationPenalty) * 0.18
            )
            .coerceIn(0.0, 1.0)
    }

    private fun motionEnergy(
        speed: Double,
        acceleration: Double,
        confidence: Double,
    ): Double {
        val speedTerm =
            (
                speed / 1.6
            ).coerceIn(0.0, 1.0)
        val accelerationTerm =
            (
                acceleration / 14.0
            ).coerceIn(0.0, 1.0)

        return (
            speedTerm * 0.48 +
                accelerationTerm * 0.34 +
                confidence * 0.18
            )
            .coerceIn(0.0, 1.0)
    }

    private fun commonModeMagnitudeFor(
        actor: MultiActorPoseAssociationEngine.Result,
        commonVelocity: Double,
    ): Double {
        val local =
            sqrt(
                actor.velocityXNormalizedPerSecond *
                    actor.velocityXNormalizedPerSecond +
                    actor.velocityYNormalizedPerSecond *
                    actor.velocityYNormalizedPerSecond,
            )
        return min(
            1.0,
            abs(local - commonVelocity) /
                (local + commonVelocity + 0.02),
        )
            .let { divergence ->
                (
                    1.0 -
                        divergence
                )
                    .coerceIn(0.0, 1.0)
            }
            .let { commonLike ->
                (
                    commonLike *
                        config.commonModeRobustness
                )
                    .coerceIn(0.0, 1.0)
            }
    }

    private fun robustCommonModeVelocity(
        actors: List<MultiActorPoseAssociationEngine.Result>,
        cameraMotionRisk: Double,
    ): Double {
        if (actors.isEmpty()) return 0.0

        val values =
            actors
                .map {
                    sqrt(
                        it.velocityXNormalizedPerSecond *
                            it.velocityXNormalizedPerSecond +
                            it.velocityYNormalizedPerSecond *
                            it.velocityYNormalizedPerSecond,
                    )
                }
                .filter { it.isFinite() }
                .sorted()

        if (values.isEmpty()) return 0.0

        val median =
            if (values.size % 2 == 1) {
                values[values.size / 2]
            } else {
                (
                    values[values.size / 2 - 1] +
                        values[values.size / 2]
                    ) * 0.5
            }

        val deviations =
            values.map { abs(it - median) }
                .sorted()

        val mad =
            if (deviations.size % 2 == 1) {
                deviations[deviations.size / 2]
            } else {
                (
                    deviations[deviations.size / 2 - 1] +
                        deviations[deviations.size / 2]
                    ) * 0.5
            }

        val threshold =
            median +
                config.robustOutlierSigma *
                    max(mad, 0.01)

        val inliers =
            values.filter { it <= threshold }

        val robustMean =
            inliers.averageOrZero()

        return (
            robustMean *
                (
                    0.45 +
                        0.55 * cameraMotionRisk
                    )
            )
            .coerceAtLeast(0.0)
    }

    private fun buildInteractions(
        actors: List<MultiActorPoseAssociationEngine.Result>,
    ): List<PairInteraction> {
        if (actors.size < 2) return emptyList()

        val interactions = ArrayList<PairInteraction>(
            (actors.size * (actors.size - 1)) / 6,
        )

        for (i in actors.indices) {
            val a = actors[i]
            for (j in i + 1 until actors.size) {
                val b = actors[j]

                val centerDistance =
                    a.center.distanceTo(b.center)

                val iou =
                    intersectionOverUnion(
                        a.bounds.left,
                        a.bounds.top,
                        a.bounds.right,
                        a.bounds.bottom,
                        b.bounds.left,
                        b.bounds.top,
                        b.bounds.right,
                        b.bounds.bottom,
                    )

                val level =
                    when {
                        iou >= config.heavyOverlapIou ||
                            centerDistance <= config.contactDistanceNormalized ->
                            InteractionLevel.HEAVY_OCCLUSION
                        centerDistance <= config.contactDistanceNormalized * 1.45 ->
                            InteractionLevel.CONTACT_RISK
                        centerDistance <= config.nearDistanceNormalized ->
                            InteractionLevel.NEAR
                        else ->
                            InteractionLevel.ISOLATED
                    }

                if (level == InteractionLevel.ISOLATED) continue

                val occlusionRisk =
                    (
                        iou * 0.56 +
                            (
                                1.0 -
                                    (
                                        centerDistance /
                                            config.nearDistanceNormalized
                                    )
                            ).coerceIn(0.0, 1.0) * 0.44
                    )
                        .coerceIn(0.0, 1.0)

                interactions +=
                    PairInteraction(
                        actorA = a.actorId,
                        actorB = b.actorId,
                        centerDistanceNormalized =
                            centerDistance.coerceAtLeast(0.0),
                        iou = iou,
                        level = level,
                        occlusionRisk = occlusionRisk,
                    )
            }
        }

        return interactions
            .sortedWith(
                compareByDescending<PairInteraction> { it.occlusionRisk }
                    .thenBy { it.actorA }
                    .thenBy { it.actorB },
            )
    }

    private fun actorCrowdPressure(
        actor: MultiActorPoseAssociationEngine.Result,
        interactions: List<PairInteraction>,
    ): Double {
        if (interactions.isEmpty()) return 0.0

        val localPressure =
            interactions
                .map { interactionSeverity(it.level) * it.occlusionRisk }
                .averageOrZero()

        val identityLoss =
            actor.missedFrames.toDouble() /
                8.0

        return (
            localPressure * 0.82 +
                identityLoss.coerceIn(0.0, 1.0) * 0.18
            )
            .coerceIn(0.0, 1.0)
    }

    private fun identityRisk(
        actor: MultiActorPoseAssociationEngine.Result,
        crowdPressure: Double,
        input: Input,
        interactions: List<PairInteraction>,
        worldGeometry: WorldGeometryEvidence,
    ): Double {
        val predictionRisk =
            if (actor.predicted) {
                (
                    1.0 -
                        actor.predictionConfidence
                )
                    .coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val innovationRisk =
            (
                actor.meanMotionInnovationNormalized /
                    0.12
            )
                .coerceIn(0.0, 1.0)

        val occlusionRisk =
            if (interactions.isEmpty()) {
                0.0
            } else {
                interactions
                    .map { it.occlusionRisk }
                    .maxOrNull()
                    ?: 0.0
            }

        val timingRisk =
            1.0 -
                input.timingConfidence

        val confidenceRisk =
            1.0 -
                actor.confidence.coerceIn(0.0, 1.0)

        val worldGeometryRisk =
            if (actor.worldJoints.isEmpty()) {
                0.025
            } else {
                (1.0 - worldGeometry.reliability) * 0.075
            }

        return (
            predictionRisk *
                config.identityRiskPredictionWeight +
                innovationRisk *
                config.identityRiskInnovationWeight +
                crowdPressure *
                config.identityRiskCrowdWeight +
                occlusionRisk *
                config.identityRiskOcclusionWeight +
                timingRisk *
                config.identityRiskTimingWeight +
                confidenceRisk * 0.10 +
                worldGeometryRisk
            )
            .coerceIn(0.0, 1.0)
    }

    private fun classifyState(
        actor: MultiActorPoseAssociationEngine.Result,
        identityRisk: Double,
        input: Input,
        memory: MutableMemory,
    ): ActorState {
        if (actor.confidence < config.minimumActorConfidence * 0.75) {
            return ActorState.HOLD
        }

        if (actor.predicted) {
            return when {
                identityRisk >= 0.78 ->
                    ActorState.HOLD
                identityRisk >= 0.48 ->
                    ActorState.RECOVERING
                actor.predictionConfidence <
                    config.minimumPredictionConfidence ->
                    ActorState.RECOVERING
                else ->
                    ActorState.PREDICTED
            }
        }

        if (identityRisk >= 0.80) return ActorState.HOLD

        if (memory.lastState == ActorState.PREDICTED &&
            identityRisk >= 0.42
        ) {
            return ActorState.RECOVERING
        }

        if (input.runtimeHeadroom < 0.18 &&
            actor.confidence < 0.55
        ) {
            return ActorState.HOLD
        }

        return if (actor.missedFrames > 0) {
            ActorState.RECOVERING
        } else {
            ActorState.OBSERVED
        }
    }

    private fun fairnessWeight(
        memory: MutableMemory,
        actor: MultiActorPoseAssociationEngine.Result,
        runtimeHeadroom: Double,
    ): Double {
        val aging =
            1.0 +
                memory.deferredFrames /
                    24.0

        val predictionBoost =
            if (actor.predicted) {
                1.10
            } else {
                1.0
            }

        val confidenceDamping =
            (
                0.80 +
                    actor.confidence * 0.40
            )
                .coerceIn(0.80, 1.20)

        val runtimeDamping =
            (
                0.72 +
                    runtimeHeadroom * 0.28
            )
                .coerceIn(0.72, 1.0)

        return (
            aging *
                predictionBoost *
                confidenceDamping *
                runtimeDamping
            )
            .coerceIn(
                config.minimumFairnessWeight,
                config.maximumFairnessWeight,
            )
    }

    private fun schedulingScore(
        actor: MultiActorPoseAssociationEngine.Result,
        state: ActorState,
        identityRisk: Double,
        compensatedMotion: Double,
        crowdPressure: Double,
        fairness: Double,
        timingConfidence: Double,
        runtimeHeadroom: Double,
        silhouetteEvidence:
            ActorSilhouetteEvidenceEngine.Evidence?,
    ): Double {
        val confidence =
            actor.confidence.coerceIn(0.0, 1.0)

        val predictionTrust =
            if (actor.predicted) {
                actor.predictionConfidence
                    .coerceIn(0.0, 1.0)
            } else {
                1.0
            }

        val silhouetteFactor =
            silhouetteEvidence?.let {
                (
                    0.78 +
                        it.reliability * 0.22 -
                        it.overlapAmbiguity * 0.10
                    ).coerceIn(0.68, 1.0)
            } ?: 1.0

        val stateWeight =
            when (state) {
                ActorState.OBSERVED -> 1.0
                ActorState.PREDICTED -> 0.88
                ActorState.RECOVERING -> 0.78
                ActorState.HOLD -> 0.50
                ActorState.LOST -> 0.0
            }

        return (
            (
                confidence * 0.23 +
                visibility(actor) * 0.14 +
                predictionTrust * 0.10 +
                compensatedMotion * 0.17 +
                (1.0 - identityRisk) * 0.16 +
                (1.0 - crowdPressure) * 0.05 +
                timingConfidence * 0.06 +
                runtimeHeadroom * 0.06
            ) *
            silhouetteFactor *
            stateWeight *
            fairness.coerceIn(
                config.minimumFairnessWeight,
                config.maximumFairnessWeight,
            ) /
            config.maximumFairnessWeight
        ).coerceIn(0.0, 1.0)
    }

    private fun allocateDetail(
        schedulingScore: Double,
        identityRisk: Double,
        crowdPressure: Double,
        state: ActorState,
        runtimeHeadroom: Double,
        silhouetteEvidence:
            ActorSilhouetteEvidenceEngine.Evidence?,
    ): DetailAllocation {
        val safety =
            (
                1.0 -
                    identityRisk * 0.56 -
                    crowdPressure * 0.18
                )
                .coerceIn(0.25, 1.0)

        val body =
            (
                0.52 +
                    schedulingScore * 0.42
                )
                .coerceIn(0.0, 1.0)

        val face =
            (
                schedulingScore *
                    0.92 *
                    safety
                )
                .coerceIn(0.0, 1.0)

        val eye =
            (
                schedulingScore *
                    0.84 *
                    safety
                )
                .coerceIn(0.0, 1.0)

        val hand =
            (
                schedulingScore *
                    (0.74 + 0.18 * (1.0 - crowdPressure))
                    *
                    safety
                )
                .coerceIn(0.0, 1.0)

        val silhouetteSurfaceScale =
            silhouetteEvidence?.let {
                (
                    0.62 +
                        it.reliability * 0.38 -
                        it.overlapAmbiguity * 0.16
                    ).coerceIn(0.45, 1.0)
            } ?: 1.0

        val surface =
            (
                schedulingScore *
                    0.72 *
                    runtimeHeadroom *
                    safety *
                    silhouetteSurfaceScale
                )
                .coerceIn(0.0, 1.0)

        val holdSurface =
            state == ActorState.HOLD ||
                identityRisk >= 0.72 ||
                runtimeHeadroom < 0.20

        return DetailAllocation(
            body = body,
            face = if (holdSurface) face * 0.72 else face,
            eye = if (holdSurface) eye * 0.66 else eye,
            hand = if (holdSurface) hand * 0.62 else hand,
            surface = if (holdSurface) 0.0 else surface,
            holdSurface = holdSurface,
        )
    }

    private fun worldGeometryEvidence(
        actor: MultiActorPoseAssociationEngine.Result,
    ): WorldGeometryEvidence {
        val joints =
            actor.worldJoints
                .asSequence()
                .filter {
                    it.confidence > 0.05 &&
                        it.xMeters.isFinite() &&
                        it.yMeters.isFinite() &&
                        it.zMeters.isFinite()
                }
                .associateBy { it.landmarkId }

        val referenceIds =
            listOf(
                0,
                11,
                12,
                23,
                24,
                25,
                26,
                27,
                28,
            )

        val coverage =
            referenceIds
                .count { joints[it] != null }
                .toDouble() /
                referenceIds.size.toDouble()

        val pairs =
            listOf(
                0 to 11,
                0 to 12,
                11 to 12,
                11 to 23,
                12 to 24,
                23 to 25,
                24 to 26,
                25 to 27,
                26 to 28,
            )

        val lengths =
            pairs
                .mapNotNull { (a, b) ->
                    val first = joints[a]?.positionMeters
                    val second = joints[b]?.positionMeters
                    if (first == null || second == null) {
                        null
                    } else {
                        first.distanceTo(second)
                            .takeIf { it.isFinite() && it > 1e-5 }
                    }
                }

        if (lengths.size < 3) {
            return WorldGeometryEvidence(
                coverage = coverage.coerceIn(0.0, 1.0),
                boneConsistency = 0.0,
                bodyScaleMeters = 0.0,
                reliability = (
                    coverage * 0.55
                ).coerceIn(0.0, 1.0),
            )
        }

        val sorted = lengths.sorted()
        val median =
            sorted[sorted.size / 2]
        val deviations =
            sorted
                .map { abs(it - median) }
                .sorted()
        val mad =
            deviations[deviations.size / 2]
                .coerceAtLeast(1e-6)

        val relativeSpread =
            (mad / median.coerceAtLeast(1e-6))
                .coerceIn(0.0, 1.0)

        val boneConsistency =
            (
                1.0 -
                    relativeSpread * 2.4
                )
                .coerceIn(0.0, 1.0)

        val reliability =
            (
                coverage * 0.48 +
                    boneConsistency * 0.42 +
                    actor.worldJoints
                        .map { it.confidence }
                        .averageOrZero() * 0.10
            )
                .coerceIn(0.0, 1.0)

        return WorldGeometryEvidence(
            coverage = coverage.coerceIn(0.0, 1.0),
            boneConsistency = boneConsistency,
            bodyScaleMeters = median,
            reliability = reliability,
        )
    }

    private fun actorWarnings(
        actor: MultiActorPoseAssociationEngine.Result,
        state: ActorState,
        identityRisk: Double,
        crowdPressure: Double,
        input: Input,
    ): List<String> =
        buildList {
            if (actor.predicted) {
                add("ACTOR_SHORT_PREDICTION_BRIDGE_ACTIVE")
            }
            if (actor.predictionConfidence < 0.35 &&
                actor.predicted
            ) {
                add("ACTOR_PREDICTION_CONFIDENCE_LOW")
            }
            if (identityRisk >= 0.55) {
                add("ACTOR_IDENTITY_RISK_ELEVATED")
            }
            if (crowdPressure >= 0.55) {
                add("ACTOR_CROWD_OCCLUSION_PRESSURE_ELEVATED")
            }
            if (
                actor.worldJoints.isNotEmpty() &&
                worldGeometryEvidence(actor).reliability < 0.35
            ) {
                add("ACTOR_WORLD_BODY_GEOMETRY_WEAK")
            }
            if (actor.microMotion.accelerationNormalizedPerSecond2 >= 18.0) {
                add("ACTOR_HIGH_ACCELERATION")
            }
            if (input.timingConfidence < 0.50) {
                add("ACTOR_TIMING_EVIDENCE_DEGRADED")
            }
            if (input.runtimeHeadroom < 0.30) {
                add("ACTOR_RUNTIME_HEADROOM_DEGRADED")
            }
            if (state == ActorState.HOLD) {
                add("ACTOR_DETAIL_HELD_PENDING_STRONGER_EVIDENCE")
            }
        }.distinct()

    private fun buildSchedulerDecision(
        frame: FrameResult,
    ): SchedulerDecision {
        val ranked =
            frame.actors
                .sortedWith(
                    compareByDescending<ActorEvidence> {
                        it.schedulingScore
                    }.thenBy {
                        it.actorId
                    },
                )

        val count =
            ranked.size.coerceAtLeast(1)

        val baseBudget =
            (
                frame.globalSchedulingCapacity /
                    count.toDouble().coerceAtLeast(1.0)
            )
                .coerceIn(
                    0.0,
                    1.0,
                )

        val minimum =
            if (ranked.isEmpty()) {
                0.0
            } else {
                ranked.minOf {
                    min(
                        baseBudget,
                        (
                            0.20 +
                                it.schedulingScore * 0.38
                            )
                                .coerceIn(0.10, 0.72),
                    )
                }
            }

        val slices =
            ranked.mapIndexed { rank, actor ->
                val actorBudget =
                    (
                        baseBudget *
                            (
                                0.82 +
                                    actor.fairnessWeight * 0.18
                            )
                        )
                        .coerceIn(
                            0.05,
                            1.0,
                        )

                SchedulerSlice(
                    actorId = actor.actorId,
                    bodyBudget =
                        max(
                            0.20,
                            actor.recommendedBodyDetail,
                        )
                        .coerceIn(0.0, 1.0),
                    faceBudget =
                        min(
                            actor.recommendedFaceDetail,
                            actorBudget,
                        ),
                    eyeBudget =
                        min(
                            actor.recommendedEyeDetail,
                            actorBudget,
                        ),
                    handBudget =
                        min(
                            actor.recommendedHandDetail,
                            actorBudget,
                        ),
                    surfaceBudget =
                        min(
                            actor.recommendedSurfaceDetail,
                            actorBudget,
                        ),
                    deferredSurface =
                        actor.holdDenseDetail ||
                            actor.recommendedSurfaceDetail < 0.18,
                    deferredFaceEye =
                        actor.recommendedFaceDetail < 0.16 &&
                            actor.recommendedEyeDetail < 0.12,
                    priorityRank = rank,
                )
            }

        val deferred =
            slices.count {
                it.deferredSurface ||
                    it.deferredFaceEye
            }

        val warnings =
            buildList {
                if (frame.actors.size >= config.maximumActors) {
                    add("CROWD_ACTOR_TARGET_CAP_REACHED")
                }
                if (deferred > 0) {
                    add("OPTIONAL_DETAIL_DEFERRED_TO_PRESERVE_CORE_BODY_TRACKING")
                }
                if (frame.crowdPressure >= 0.60) {
                    add("DENSE_INTERACTION_REQUIRES_GEOMETRIC_AND_IDENTITY_CAUTION")
                }
                if (frame.timingConfidence < 0.50) {
                    add("TIMING_EVIDENCE_REDUCES_OPTIONAL_DETAIL_BUDGET")
                }
                if (frame.runtimeHeadroom < 0.28) {
                    add("RUNTIME_HEADROOM_REDUCES_DENSE_SURFACE_BUDGET")
                }
            }.distinct()

        return SchedulerDecision(
            timestampNanos = frame.timestampNanos,
            slices = slices,
            preservedActorCount =
                frame.actors.count {
                    !it.holdDenseDetail
                },
            deferredActorCount = deferred,
            minimumActorBudget = minimum,
            maximumActorBudget =
                slices.maxOfOrNull {
                    it.bodyBudget
                } ?: 0.0,
            warnings = warnings,
        )
    }

    private fun appendHistory(
        memory: MutableMemory,
        actor: MultiActorPoseAssociationEngine.Result,
        evidence: ActorEvidence,
    ) {
        memory.history.addLast(
            HistoryDigest(
                timestampNanos = memory.lastTimestampNanos,
                center = actor.center,
                speed = actor.microMotion.velocityNormalizedPerSecond,
                acceleration =
                    actor.microMotion.accelerationNormalizedPerSecond2,
                confidence = actor.confidence,
                identitySafety = 1.0 - evidence.identityRisk,
            ),
        )

        while (memory.history.size > config.historyFrames) {
            memory.history.removeFirst()
        }
    }

    private fun pruneMemories(
        activeActorIds: Set<Long>,
    ) {
        memories.entries.removeIf { (actorId, memory) ->
            actorId !in activeActorIds &&
                memory.ageFrames >= config.staleStateFrames
        }
    }

    private fun bottleneck(
        meanConfidence: Double,
        meanVisibility: Double,
        meanIdentitySafety: Double,
        crowdPressure: Double,
        interactionPressure: Double,
        cameraMotionRisk: Double,
        timingConfidence: Double,
        runtimeHeadroom: Double,
    ): Bottleneck =
        when {
            runtimeHeadroom < 0.22 ->
                Bottleneck.RUNTIME_PRESSURE
            timingConfidence < 0.42 ->
                Bottleneck.TIMING
            cameraMotionRisk > 0.62 ->
                Bottleneck.CAMERA_MOTION
            meanIdentitySafety < 0.44 ->
                Bottleneck.IDENTITY_RISK
            crowdPressure > 0.62 ||
                interactionPressure > 0.64 ->
                Bottleneck.CROWD_DENSITY
            meanVisibility < 0.32 ->
                Bottleneck.VISIBILITY
            meanConfidence < 0.34 ->
                Bottleneck.DETECTOR_CONFIDENCE
            else ->
                Bottleneck.TEMPORAL_INNOVATION
        }

    private fun buildWarnings(
        actorEvidence: List<ActorEvidence>,
        interactions: List<PairInteraction>,
        bottleneck: Bottleneck,
        input: Input,
    ): List<String> =
        buildList {
            when (bottleneck) {
                Bottleneck.RUNTIME_PRESSURE ->
                    add("RUNTIME_PRESSURE_REQUIRES_DEFERRED_OPTIONAL_MODALITIES")
                Bottleneck.TIMING ->
                    add("TIMING_QUALITY_IS_LIMITING_DENSE_TEMPORAL_FUSION")
                Bottleneck.CAMERA_MOTION ->
                    add("COMMON_MODE_CAMERA_MOTION_REQUIRES_STRONGER_COMPENSATION")
                Bottleneck.IDENTITY_RISK ->
                    add("IDENTITY_CONTINUITY_REQUIRES_ADDITIONAL_GEOMETRIC_EVIDENCE")
                Bottleneck.CROWD_DENSITY ->
                    add("CROWD_DENSITY_REDUCES_INDEPENDENT_VIEW_OBSERVABILITY")
                Bottleneck.VISIBILITY ->
                    add("ACTOR_VISIBILITY_IS_LIMITING_EVIDENCE")
                Bottleneck.DETECTOR_CONFIDENCE ->
                    add("DETECTOR_CONFIDENCE_IS_LIMITING_EVIDENCE")
                Bottleneck.TEMPORAL_INNOVATION,
                Bottleneck.NONE ->
                    Unit
            }

            if (actorEvidence.isEmpty()) {
                add("NO_ACTIVE_ACTOR_EVIDENCE")
            }

            val highRisk =
                actorEvidence.count {
                    it.identityRisk >= 0.65
                }

            if (highRisk > 0) {
                add("IDENTITY_RISK_ACTORS=" + highRisk)
            }

            if (interactions.isNotEmpty()) {
                add("INTERACTING_PAIRS=" + interactions.size)
            }

            if (input.runtimeHeadroom < 0.35) {
                add("RUNTIME_HEADROOM_BELOW_DENSE_DETAIL_TARGET")
            }

            if (
                input.segmentationEvidence > 0.0 &&
                input.segmentationEvidence < 0.35
            ) {
                add("SILHOUETTE_EVIDENCE_BELOW_DENSE_DETAIL_TARGET")
            }

            if (input.timingConfidence < 0.65) {
                add("TEMPORAL_EVIDENCE_BELOW_DENSE_DETAIL_TARGET")
            }
        }.distinct()

    private fun interactionSeverity(
        level: InteractionLevel,
    ): Double =
        when (level) {
            InteractionLevel.ISOLATED -> 0.0
            InteractionLevel.NEAR -> 0.35
            InteractionLevel.CONTACT_RISK -> 0.72
            InteractionLevel.HEAVY_OCCLUSION -> 1.0
        }

    private fun intersectionOverUnion(
        leftA: Double,
        topA: Double,
        rightA: Double,
        bottomA: Double,
        leftB: Double,
        topB: Double,
        rightB: Double,
        bottomB: Double,
    ): Double {
        val left =
            max(leftA, leftB)
        val top =
            max(topA, topB)
        val right =
            min(rightA, rightB)
        val bottom =
            min(bottomA, bottomB)

        val intersection =
            (
                right - left
            )
                .coerceAtLeast(0.0) *
                (
                    bottom - top
                )
                .coerceAtLeast(0.0)

        val areaA =
            (
                rightA - leftA
            )
                .coerceAtLeast(0.0) *
                (
                    bottomA - topA
                )
                .coerceAtLeast(0.0)

        val areaB =
            (
                rightB - leftB
            )
                .coerceAtLeast(0.0) *
                (
                    bottomB - topB
                )
                .coerceAtLeast(0.0)

        val union =
            areaA +
                areaB -
                intersection

        if (union <= 1e-12) return 0.0
        return (
            intersection /
                union
            )
            .coerceIn(0.0, 1.0)
    }

    private fun metric(value: Double): String =
        if (value.isFinite()) {
            "%.6f".format(java.util.Locale.US, value)
        } else {
            "NaN"
        }

    private fun emptyResult(
        timestampNanos: Long,
        warning: String,
    ): FrameResult =
        FrameResult(
            timestampNanos = timestampNanos,
            actors = emptyList(),
            interactions = emptyList(),
            activeActorCount = 0,
            predictedActorCount = 0,
            recoveringActorCount = 0,
            heldActorCount = 0,
            meanConfidence = 0.0,
            meanVisibility = 0.0,
            meanIdentitySafety = 0.0,
            commonModeVelocityNormalizedPerSecond = 0.0,
            differentialMotionScore = 0.0,
            cameraCompensatedMotionScore = 0.0,
            crowdPressure = 0.0,
            interactionPressure = 0.0,
            meanWorldLandmarkCoverage = 0.0,
            meanWorldBoneConsistency = 0.0,
            meanWorldGeometryReliability = 0.0,
            segmentationEvidence = 0.0,
            segmentationCoverage = 0.0,
            actorSilhouetteEvidenceAvailable = false,
            meanActorSilhouetteSupport = 0.0,
            meanActorSilhouetteAmbiguity = 0.0,
            meanAppearanceQuality = 0.0,
            timingConfidence = 0.0,
            runtimeHeadroom = 0.0,
            fairnessSaturation = 0.0,
            globalSchedulingCapacity = 0.0,
            bottleneck = Bottleneck.TIMING,
            authoritative = false,
            warnings = listOf(warning),
            diagnostics = listOf(
                "timestampContractViolation=true",
                "outputHeld=true",
            ),
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun ArrayDeque<HistoryDigest>.mapIndexed(
        transform: (Int, HistoryDigest) -> FrameDigest,
    ): List<FrameDigest> {
        val result = ArrayList<FrameDigest>(size)
        var index = 0
        for (digest in this) {
            result += transform(index++, digest)
        }
        return result
    }

}
