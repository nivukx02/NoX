package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.ActorDetailAllocationEngine
import com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine
import com.nivukx.nox.core.model.MotionCaptureCustomization
import kotlin.math.max
import kotlin.math.min

/**
 * Deterministic workload scheduler for dense multi-actor reconstruction.
 *
 * The scheduler separates identity retention from expensive detail execution. Every admitted actor
 * remains eligible for standard reconstruction while expensive face/eye/hand/surface work is
 * allocated with aging, risk, motion and primary-actor priority. This avoids the common failure
 * mode where a small number of visually salient actors consume all accelerator time and other
 * actors silently starve.
 *
 * The scheduler is bounded to the same 64-actor ceiling already used by NoX model contracts and
 * is intentionally independent of detector/provider implementation.
 */
class ActorWorkloadScheduler(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumActors: Int = 64,
        val starvationFrames: Int = 8,
        val maximumPriorityBoost: Double = 0.48,
        val motionWeight: Double = 0.20,
        val occlusionWeight: Double = 0.14,
        val identityWeight: Double = 0.18,
        val confidenceWeight: Double = 0.12,
        val allocationWeight: Double = 0.16,
        val primaryWeight: Double = 0.20,
        val minimumDetailQuantum: Double = 0.25,
        val maximumDetailQuantum: Double = 1.0,
        val historyRetentionFrames: Int = 120,
        val fairnessWeight: Double = 0.22,
        val continuityWeight: Double = 0.14,
        val diversityWeight: Double = 0.08,
    ) {
        init {
            require(maximumActors in 1..64)
            require(starvationFrames >= 1)
            require(maximumPriorityBoost in 0.0..1.0)
            require(motionWeight >= 0.0)
            require(occlusionWeight >= 0.0)
            require(identityWeight >= 0.0)
            require(confidenceWeight >= 0.0)
            require(allocationWeight >= 0.0)
            require(primaryWeight >= 0.0)
            require(minimumDetailQuantum in 0.0..1.0)
            require(maximumDetailQuantum in minimumDetailQuantum..1.0)
            require(historyRetentionFrames >= starvationFrames)
            require(fairnessWeight >= 0.0)
            require(continuityWeight >= 0.0)
            require(diversityWeight >= 0.0)
            require(
                motionWeight +
                    occlusionWeight +
                    identityWeight +
                    confidenceWeight +
                    allocationWeight +
                    primaryWeight +
                    fairnessWeight +
                    continuityWeight +
                    diversityWeight >
                    0.0,
            )
        }
    }

    data class Candidate(
        val actorId: Long,
        val allocationRank: Int = Int.MAX_VALUE,
        val optionalCadenceScale: Double = 1.0,
        val fullDetailRequested: Boolean = false,
        val surfaceRequested: Boolean = false,
        val confidence: Double = 0.0,
        val motionScore: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val identityRisk: Double = 0.0,
        val continuityScore: Double = 1.0,
        val predicted: Boolean = false,
        val primary: Boolean = false,
    ) {
        init {
            require(actorId > 0L)
            require(optionalCadenceScale in 0.0..1.0)
            require(allocationRank >= 0)
            require(confidence in 0.0..1.0)
            require(motionScore in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
            require(identityRisk in 0.0..1.0)
            require(continuityScore in 0.0..1.0)
        }
    }

    data class Decision(
        val actorId: Long,
        val score: Double,
        val detailQuantum: Double,
        val selectedForDetail: Boolean,
        val starvationFrames: Int,
        val reason: String,
    )

    data class Result(
        val timestampNanos: Long,
        val decisions: List<Decision>,
        val selectedActorIds: Set<Long>,
        val admittedActorIds: Set<Long>,
        val heldActorIds: Set<Long>,
        val predictedActorIds: Set<Long>,
        val starvationRisk: Double,
        val fairnessScore: Double,
        val meanPriorityScore: Double,
        val maximumStarvationFrames: Int,
        val warnings: List<String>,
    )

    private data class State(
        var starvationFrames: Int = 0,
        var lastSelectedTimestampNanos: Long = Long.MIN_VALUE,
        var selections: Long = 0L,
        var lastScore: Double = 0.0,
    )

    private val stateByActor = LinkedHashMap<Long, State>()

    @Synchronized
    fun schedule(
        timestampNanos: Long,
        actors: List<MultiActorPoseAssociationEngine.Result>,
        allocation: ActorDetailAllocationEngine.Result?,
        customization: MotionCaptureCustomization,
        primaryActorId: Long?,
        maximumDetailedActorsPerFrame: Int,
    ): Result {
        require(timestampNanos >= 0L)
        require(maximumDetailedActorsPerFrame in 1..64)

        val maxActors =
            min(
                config.maximumActors,
                customization.maxSubjects.coerceIn(1, 64),
            )

        val admitted =
            actors
                .sortedWith(
                    compareByDescending<MultiActorPoseAssociationEngine.Result> {
                        if (it.actorId == primaryActorId) 1 else 0
                    }.thenByDescending {
                        it.confidence
                    }.thenBy {
                        it.actorId
                    },
                )
                .take(maxActors)

        val candidateById =
            admitted.associate { actor ->
                val detail =
                    allocation?.allocation(actor.actorId)
                Candidate(
                    actorId = actor.actorId,
                    allocationRank =
                        detail?.rank ?: Int.MAX_VALUE,
                    optionalCadenceScale =
                        detail?.optionalCadenceScale
                            ?.coerceIn(0.25, 1.0)
                            ?: 1.0,
                    fullDetailRequested =
                        detail?.fullDetail == true,
                    surfaceRequested =
                        detail?.tiers?.contains(
                            ActorDetailAllocationEngine.DetailTier.SURFACE,
                        ) == true,
                    confidence =
                        actor.confidence.coerceIn(0.0, 1.0),
                    motionScore =
                        actor.microMotion.preservedSignal
                            .coerceIn(0.0, 1.0),
                    occlusionRisk =
                        (
                            actor.missedFrames.toDouble() /
                                config.starvationFrames
                                    .coerceAtLeast(1)
                        ).coerceIn(0.0, 1.0),
                    identityRisk =
                        actor.identityAmbiguity.coerceIn(0.0, 1.0),
                    continuityScore =
                        actor.bodySpaceContinuity
                            .coerceIn(0.0, 1.0),
                    predicted = actor.predicted,
                    primary = actor.actorId == primaryActorId,
                )
            }

        val decisions =
            candidateById.values
                .map { candidate ->
                    val state =
                        stateByActor.getOrPut(
                            candidate.actorId,
                            ::State,
                        )

                    val allocationScore =
                        (
                            if (candidate.fullDetailRequested) 1.0 else 0.58
                        ) *
                            (
                                1.0 /
                                    (
                                        1.0 +
                                            candidate
                                                .allocationRank
                                                .coerceAtLeast(0)
                                                .toDouble() /
                                            8.0
                                    )
                            )

                    val fairness =
                        (
                            state.starvationFrames.toDouble() /
                                config.starvationFrames
                                    .coerceAtLeast(1)
                        ).coerceIn(0.0, 1.0)

                    val primary =
                        if (candidate.primary) 1.0 else 0.0

                    val score =
                        (
                            candidate.motionScore *
                                config.motionWeight +
                                candidate.occlusionRisk *
                                config.occlusionWeight +
                                candidate.identityRisk *
                                config.identityWeight +
                                candidate.confidence *
                                config.confidenceWeight +
                                allocationScore *
                                config.allocationWeight +
                                primary *
                                config.primaryWeight +
                                fairness *
                                config.fairnessWeight +
                                candidate.continuityScore *
                                config.continuityWeight +
                                (
                                    1.0 -
                                        candidate.optionalCadenceScale
                                ) *
                                config.diversityWeight
                            ).coerceIn(0.0, 1.0)

                    val boost =
                        min(
                            config.maximumPriorityBoost,
                            fairness *
                                config.maximumPriorityBoost,
                        )

                    val finalScore =
                        (score + boost).coerceIn(0.0, 1.0)

                    state.lastScore = finalScore

                    val quantum =
                        (
                            config.minimumDetailQuantum +
                                (
                                    config.maximumDetailQuantum -
                                        config.minimumDetailQuantum
                                    ) *
                                finalScore
                            ).coerceIn(
                                config.minimumDetailQuantum,
                                config.maximumDetailQuantum,
                            )

                    Decision(
                        actorId = candidate.actorId,
                        score = finalScore,
                        detailQuantum = quantum,
                        selectedForDetail = false,
                        starvationFrames = state.starvationFrames,
                        reason = reason(candidate, fairness, finalScore),
                    )
                }
                .sortedWith(
                    compareByDescending<Decision> { it.score }
                        .thenBy { it.actorId },
                )

        val selectedIds =
            decisions
                .take(
                    min(
                        maximumDetailedActorsPerFrame,
                        decisions.size,
                    ),
                )
                .map { it.actorId }
                .toSet()

        val finalized =
            decisions.map { decision ->
                decision.copy(
                    selectedForDetail =
                        decision.actorId in selectedIds,
                )
            }

        finalized.forEach { decision ->
            val state =
                stateByActor[decision.actorId]
                    ?: return@forEach

            if (decision.selectedForDetail) {
                state.starvationFrames = 0
                state.lastSelectedTimestampNanos = timestampNanos
                state.selections += 1L
            } else {
                state.starvationFrames =
                    min(
                        config.historyRetentionFrames,
                        state.starvationFrames + 1,
                    )
            }
        }

        prune(timestampNanos, maxActors)

        val selectedSet = selectedIds
        val admittedSet = admitted.map { it.actorId }.toSet()
        val held =
            admittedSet -
                selectedSet

        val predicted =
            admitted
                .filter { it.predicted }
                .map { it.actorId }
                .toSet()

        val starvationMaximum =
            stateByActor
                .filterKeys { it in admittedSet }
                .values
                .maxOfOrNull { it.starvationFrames }
                ?: 0

        val starvationRisk =
            (
                starvationMaximum.toDouble() /
                    config.historyRetentionFrames
                        .coerceAtLeast(1)
            ).coerceIn(0.0, 1.0)

        val selectedCount =
            selectedSet.size.coerceAtLeast(1)

        val fairnessScore =
            (
                1.0 -
                    (
                        stateByActor
                            .filterKeys { it in admittedSet }
                            .values
                            .map { it.starvationFrames.toDouble() }
                            .let { values ->
                                if (values.isEmpty()) {
                                    0.0
                                } else {
                                    values.average()
                                }
                            } /
                        config.starvationFrames
                            .coerceAtLeast(1)
                            .toDouble()
                    )
                ).coerceIn(0.0, 1.0)

        val meanPriority =
            finalized
                .map { it.score }
                .averageOrZero()

        val warnings =
            buildList {
                if (admitted.size >= 50) {
                    add("CROWD_50_ACTOR_ENVELOPE_ACTIVE")
                }
                if (held.isNotEmpty()) {
                    add("CROWD_DETAIL_SCHEDULING_ACTIVE")
                }
                if (starvationRisk > 0.65) {
                    add("CROWD_DETAIL_STARVATION_RISK_HIGH")
                }
                if (fairnessScore < 0.55 && admitted.size > 1) {
                    add("CROWD_DETAIL_FAIRNESS_DEGRADED")
                }
                if (predicted.isNotEmpty()) {
                    add("CROWD_PREDICTION_PRESENT")
                }
            }

        return Result(
            timestampNanos = timestampNanos,
            decisions = finalized,
            selectedActorIds = selectedSet,
            admittedActorIds = admittedSet,
            heldActorIds = held,
            predictedActorIds = predicted,
            starvationRisk = starvationRisk,
            fairnessScore = fairnessScore,
            meanPriorityScore = meanPriority,
            maximumStarvationFrames = starvationMaximum,
            warnings = warnings,
        )
    }

    @Synchronized
    fun reset() {
        stateByActor.clear()
    }

    fun stateSnapshot(): Map<Long, Int> =
        synchronized(this) {
            stateByActor.mapValues { it.value.starvationFrames }
        }

    private fun reason(
        candidate: Candidate,
        fairness: Double,
        score: Double,
    ): String {
        if (candidate.primary) return "PRIMARY_ACTOR_PRIORITY"
        if (fairness >= 0.80) return "STARVATION_RECOVERY"
        if (candidate.identityRisk >= 0.65) return "IDENTITY_RISK_PRIORITY"
        if (candidate.occlusionRisk >= 0.55) return "OCCLUSION_RECOVERY_PRIORITY"
        if (candidate.motionScore >= 0.70) return "HIGH_MOTION_PRIORITY"
        if (candidate.surfaceRequested) return "SURFACE_DETAIL_REQUEST"
        if (candidate.fullDetailRequested) return "FULL_DETAIL_REQUEST"
        return if (score >= 0.60) {
            "HIGH_INFORMATION_VALUE"
        } else {
            "FAIRNESS_BALANCED"
        }
    }

    private fun prune(
        timestampNanos: Long,
        maximumLiveActors: Int,
    ) {
        val horizonNanos =
            config.historyRetentionFrames
                .toLong()
                .coerceAtLeast(1L) *
                1_000_000_000L /
                15L

        val stale =
            stateByActor
                .entries
                .filter { (_, state) ->
                    state.lastSelectedTimestampNanos !=
                        Long.MIN_VALUE &&
                        timestampNanos -
                            state.lastSelectedTimestampNanos >
                        horizonNanos
                }
                .map { it.key }

        stale.forEach(stateByActor::remove)

        while (stateByActor.size > maximumLiveActors * 2) {
            val victim =
                stateByActor.entries
                    .minWithOrNull(
                        compareBy<Map.Entry<Long, State>> {
                            it.value.lastSelectedTimestampNanos
                        }.thenBy {
                            it.key
                        },
                    )?.key
                    ?: break
            stateByActor.remove(victim)
        }
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
