package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Bounded 50-actor workload scheduler.
 *
 * Body identity and temporal continuity are protected first. Face, eyes, hands, dense surface,
 * contact and global-4D refinement consume optional budgets. Aging prevents a repeatedly deferred
 * actor from remaining invisible forever.
 */
class CrowdMotionResourceScheduler(
    private val config: Config = Config(),
) {
    enum class Modality {
        BODY,
        FACE,
        EYES,
        HANDS,
        DENSE_SURFACE,
        CONTACT,
        GLOBAL_4D,
    }

    enum class PriorityClass {
        CRITICAL,
        HIGH,
        NORMAL,
        DEFERRED,
    }

    data class Config(
        val targetActors: Int = 50,
        val hardActorCeiling: Int = 64,
        val minimumBodyBudget: Double = 0.74,
        val optionalBudgetFloor: Double = 0.12,
        val fairnessWindowFrames: Long = 12L,
        val maximumDeferralFrames: Long = 30L,
        val faceShareLimit: Double = 0.56,
        val eyeShareLimit: Double = 0.42,
        val handShareLimit: Double = 0.48,
        val denseSurfaceShareLimit: Double = 0.34,
        val global4DShareLimit: Double = 0.28,
        val maximumHistoryEntries: Int = 1_024,
    ) {
        init {
            require(targetActors in 1..hardActorCeiling)
            require(hardActorCeiling in 1..64)
            require(minimumBodyBudget in 0.0..1.0)
            require(optionalBudgetFloor in 0.0..1.0)
            require(fairnessWindowFrames >= 1L)
            require(maximumDeferralFrames >= fairnessWindowFrames)
            require(faceShareLimit in 0.0..1.0)
            require(eyeShareLimit in 0.0..1.0)
            require(handShareLimit in 0.0..1.0)
            require(denseSurfaceShareLimit in 0.0..1.0)
            require(global4DShareLimit in 0.0..1.0)
            require(maximumHistoryEntries in hardActorCeiling..16_384)
        }
    }

    data class ActorDemand(
        val actorId: Long,
        val detectionConfidence: Double,
        val identityContinuity: Double,
        val motionEnergy: Double,
        val occlusionRisk: Double,
        val faceVisibility: Double,
        val eyeVisibility: Double,
        val handVisibility: Double,
        val surfaceVisibility: Double,
        val motionAcceleration: Double = 0.0,
        val motionJerk: Double = 0.0,
        val fairnessAgeFrames: Long,
        val predicted: Boolean,
        val explicitlySelected: Boolean = false,
    ) {
        init {
            require(actorId >= 0L)
            require(detectionConfidence in 0.0..1.0)
            require(identityContinuity in 0.0..1.0)
            require(motionEnergy in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
            require(faceVisibility in 0.0..1.0)
            require(eyeVisibility in 0.0..1.0)
            require(handVisibility in 0.0..1.0)
            require(surfaceVisibility in 0.0..1.0)
            require(
                motionAcceleration.isFinite() &&
                    motionAcceleration >= 0.0,
            )
            require(
                motionJerk.isFinite() &&
                    motionJerk >= 0.0,
            )
            require(fairnessAgeFrames >= 0L)
        }
    }

    data class Budget(
        val body: Double,
        val face: Double,
        val eyes: Double,
        val hands: Double,
        val denseSurface: Double,
        val contact: Double,
        val global4D: Double,
        val priority: PriorityClass,
        val deferredFrames: Long,
        val fairnessBoost: Double,
        val reason: String,
    )

    data class Result(
        val frameIndex: Long,
        val admittedActorIds: List<Long>,
        val heldActorIds: List<Long>,
        val budgets: Map<Long, Budget>,
        val modalityActorCounts: Map<Modality, Int>,
        val computeCreditsUsed: Double,
        val computeCreditsAvailable: Double,
        val fairnessBoostedCount: Int,
        val starvationRisk: Double,
        val overloadRisk: Double,
        val fairnessIndex: Double,
        val trackedHistoryCount: Int,
        val warnings: List<String>,
    )

    private data class History(
        var lastSeenFrame: Long,
        var deferredFrames: Long,
        var servedFrames: Long = 0L,
    )

    private val history = HashMap<Long, History>()
    private var frameIndex = 0L

    @Synchronized
    fun allocate(
        demands: List<ActorDemand>,
        computeCredits: Double,
        thermalHeadroom: Double,
        memoryHeadroom: Double,
        gpuHeadroom: Double,
    ): Result {
        require(computeCredits >= 0.0)
        require(thermalHeadroom in 0.0..1.0)
        require(memoryHeadroom in 0.0..1.0)
        require(gpuHeadroom in 0.0..1.0)

        frameIndex += 1L

        val availableCredits =
            computeCredits *
                thermalHeadroom *
                memoryHeadroom *
                gpuHeadroom

        val bounded =
            demands
                .filter { it.detectionConfidence.isFinite() }
                .distinctBy { it.actorId }
                .take(config.hardActorCeiling)

        val ranked =
            bounded
                .map { actor ->
                    val state =
                        history.getOrPut(actor.actorId) {
                            History(
                                lastSeenFrame = frameIndex,
                                deferredFrames = 0L,
                            )
                        }
                    state.lastSeenFrame = frameIndex

                    val ageBoost =
                        min(
                            0.22,
                            actor.fairnessAgeFrames /
                                config.fairnessWindowFrames.toDouble() *
                                0.18,
                        )

                    val deferBoost =
                        min(
                            0.28,
                            state.deferredFrames /
                                config.maximumDeferralFrames.toDouble() *
                                0.28,
                        )

                    val explicitBoost =
                        if (actor.explicitlySelected) 0.30 else 0.0

                    val predictionPenalty =
                        if (actor.predicted) 0.09 else 0.0

                    val dynamicDifficulty =
                        (
                            min(
                                actor.motionAcceleration /
                                    90.0,
                                1.0,
                            ) * 0.60 +
                                min(
                                    actor.motionJerk /
                                        600.0,
                                    1.0,
                                ) * 0.40
                            ).coerceIn(0.0, 1.0)

                    val score =
                        (
                            0.30 * actor.detectionConfidence +
                                0.22 * actor.identityContinuity +
                                0.18 * actor.motionEnergy +
                                0.08 * dynamicDifficulty +
                                0.10 * actor.faceVisibility +
                                0.07 * actor.eyeVisibility +
                                0.06 * actor.handVisibility +
                                0.07 * actor.surfaceVisibility -
                                0.20 * actor.occlusionRisk -
                                predictionPenalty +
                                ageBoost +
                                deferBoost +
                                explicitBoost
                            ).coerceIn(0.0, 1.50)

                    actor to score
                }
                .sortedWith(
                    compareByDescending<Pair<ActorDemand, Double>> { it.second }
                        .thenBy { it.first.actorId },
                )

        val actorShare =
            if (ranked.isEmpty()) {
                0.0
            } else {
                min(
                    1.0,
                    availableCredits /
                        ranked.size.toDouble().coerceAtLeast(1.0),
                )
            }

        val admitted = ArrayList<Long>(config.targetActors)
        val held = ArrayList<Long>()
        val budgets = LinkedHashMap<Long, Budget>()

        var usedCredits = 0.0
        var bodyCount = 0
        var faceCount = 0
        var eyeCount = 0
        var handCount = 0
        var surfaceCount = 0
        var contactCount = 0
        var global4DCount = 0
        var fairnessBoosted = 0

        for ((actor, rawScore) in ranked) {
            val state = history[actor.actorId]!!
            val fairnessBoost =
                (
                    state.deferredFrames /
                        config.maximumDeferralFrames.toDouble()
                    ).coerceIn(0.0, 1.0)

            val candidateAdmission =
                admitted.size < config.targetActors &&
                    (
                        actor.explicitlySelected ||
                            rawScore >= 0.34 ||
                            admitted.size < 8
                        )

            if (!candidateAdmission) {
                state.deferredFrames =
                    (state.deferredFrames + 1L)
                        .coerceAtMost(config.maximumDeferralFrames * 2L)
                held += actor.actorId
                budgets[actor.actorId] =
                    deferredBudget(
                        state = state,
                        fairnessBoost = fairnessBoost,
                        reason = "capacity-held",
                    )
                continue
            }

            val body =
                max(
                    config.minimumBodyBudget,
                    (
                        actorShare *
                            (
                                0.76 +
                                    0.24 *
                                        actor.detectionConfidence *
                                        actor.identityContinuity
                                )
                        ).coerceIn(0.0, 1.0),
                )

            val face =
                optional(
                    visibility = actor.faceVisibility,
                    actorScore = rawScore / 1.5,
                    fairnessBoost = fairnessBoost,
                    currentCount = faceCount,
                    admittedCount = admitted.size + 1,
                    limit = config.faceShareLimit,
                    scale = 0.74,
                )

            val eyes =
                optional(
                    visibility = actor.eyeVisibility,
                    actorScore = rawScore / 1.5,
                    fairnessBoost = fairnessBoost,
                    currentCount = eyeCount,
                    admittedCount = admitted.size + 1,
                    limit = config.eyeShareLimit,
                    scale = 0.68,
                )

            val hands =
                optional(
                    visibility = actor.handVisibility,
                    actorScore = rawScore / 1.5,
                    fairnessBoost = fairnessBoost,
                    currentCount = handCount,
                    admittedCount = admitted.size + 1,
                    limit = config.handShareLimit,
                    scale = 0.72,
                )

            val surface =
                optional(
                    visibility = actor.surfaceVisibility,
                    actorScore = actor.motionEnergy,
                    fairnessBoost = fairnessBoost * 0.55,
                    currentCount = surfaceCount,
                    admittedCount = admitted.size + 1,
                    limit = config.denseSurfaceShareLimit,
                    scale = 0.56,
                )

            val contact =
                if (
                    actor.occlusionRisk > 0.42 ||
                    actor.handVisibility > 0.55
                ) {
                    (
                        0.34 +
                            0.44 * actor.handVisibility +
                            0.22 * (1.0 - actor.occlusionRisk)
                        ).coerceIn(0.0, 1.0)
                } else {
                    config.optionalBudgetFloor
                }

            val global4D =
                if (
                    actor.motionEnergy > 0.35 &&
                    admitted.size < 20
                ) {
                    min(
                        config.global4DShareLimit,
                        (
                            0.30 +
                                0.42 * actor.motionEnergy +
                                0.18 * actor.identityContinuity +
                                0.10 * fairnessBoost
                            ).coerceIn(0.0, 1.0),
                    )
                } else {
                    config.optionalBudgetFloor
                }

            val actorCost =
                body +
                    face * 0.42 +
                    eyes * 0.34 +
                    hands * 0.46 +
                    surface * 0.72 +
                    contact * 0.18 +
                    global4D * 0.54

            if (
                usedCredits + actorCost > availableCredits &&
                admitted.size >= 8
            ) {
                state.deferredFrames =
                    (state.deferredFrames + 1L)
                        .coerceAtMost(config.maximumDeferralFrames * 2L)
                held += actor.actorId
                budgets[actor.actorId] =
                    deferredBudget(
                        state = state,
                        fairnessBoost = fairnessBoost,
                        reason = "compute-budget-held",
                    )
                continue
            }

            usedCredits += actorCost
            admitted += actor.actorId
            state.deferredFrames = 0L
            state.servedFrames += 1L

            val priority =
                when {
                    actor.explicitlySelected ||
                        actor.identityContinuity > 0.84 ->
                        PriorityClass.CRITICAL
                    actor.motionEnergy > 0.55 ||
                        actor.occlusionRisk > 0.58 ->
                        PriorityClass.HIGH
                    else ->
                        PriorityClass.NORMAL
                }

            if (fairnessBoost > 0.40) fairnessBoosted += 1
            bodyCount += 1
            if (face > config.optionalBudgetFloor) faceCount += 1
            if (eyes > config.optionalBudgetFloor) eyeCount += 1
            if (hands > config.optionalBudgetFloor) handCount += 1
            if (surface > config.optionalBudgetFloor) surfaceCount += 1
            if (contact > config.optionalBudgetFloor) contactCount += 1
            if (global4D > config.optionalBudgetFloor) global4DCount += 1

            budgets[actor.actorId] =
                Budget(
                    body = body,
                    face = face,
                    eyes = eyes,
                    hands = hands,
                    denseSurface = surface,
                    contact = contact,
                    global4D = global4D,
                    priority = priority,
                    deferredFrames = 0L,
                    fairnessBoost = fairnessBoost,
                    reason = "admitted",
                )
        }

        evictHistory()

        val fairnessIndex = fairnessIndex(bounded)
        val overloadRisk =
            if (availableCredits <= 0.0) {
                1.0
            } else {
                (
                    usedCredits /
                        availableCredits.coerceAtLeast(0.0001) -
                        0.75
                    ).coerceAtLeast(0.0) / 0.75
                    .coerceIn(0.0, 1.0)
            }

        val starvationRisk =
            if (held.isEmpty()) {
                0.0
            } else {
                (
                    0.72 *
                        held.size.toDouble() /
                        (held.size + admitted.size)
                            .toDouble()
                            .coerceAtLeast(1.0) +
                        0.28 *
                            (
                                1.0 -
                                    fairnessBoosted.toDouble() /
                                        held.size.toDouble().coerceAtLeast(1.0)
                                ).coerceIn(0.0, 1.0)
                    ).coerceIn(0.0, 1.0)
            }

        val warnings =
            buildList {
                if (demands.size > config.hardActorCeiling) {
                    add("hard-actor-ceiling-rejected-excess-demands")
                }
                if (thermalHeadroom < 0.35) {
                    add("thermal-headroom-low")
                }
                if (memoryHeadroom < 0.35) {
                    add("memory-headroom-low")
                }
                if (gpuHeadroom < 0.30) {
                    add("gpu-headroom-low")
                }
                if (overloadRisk > 0.55) {
                    add("compute-near-saturation")
                }
                if (starvationRisk > 0.40) {
                    add("actor-starvation-risk-elevated")
                }
                if (fairnessIndex < 0.72 && bounded.size >= 4) {
                    add("long-session-actor-fairness-below-target")
                }
            }

        return Result(
            frameIndex = frameIndex,
            admittedActorIds = admitted.sorted(),
            heldActorIds = held.sorted(),
            budgets = budgets.toMap(),
            modalityActorCounts = mapOf(
                Modality.BODY to bodyCount,
                Modality.FACE to faceCount,
                Modality.EYES to eyeCount,
                Modality.HANDS to handCount,
                Modality.DENSE_SURFACE to surfaceCount,
                Modality.CONTACT to contactCount,
                Modality.GLOBAL_4D to global4DCount,
            ),
            computeCreditsUsed = usedCredits,
            computeCreditsAvailable = availableCredits,
            fairnessBoostedCount = fairnessBoosted,
            starvationRisk = starvationRisk,
            overloadRisk = overloadRisk,
            fairnessIndex = fairnessIndex,
            trackedHistoryCount = history.size,
            warnings = warnings,
        )
    }

    @Synchronized
    fun reset() {
        history.clear()
        frameIndex = 0L
    }

    @Synchronized
    fun currentFrame(): Long = frameIndex

    private fun optional(
        visibility: Double,
        actorScore: Double,
        fairnessBoost: Double,
        currentCount: Int,
        admittedCount: Int,
        limit: Double,
        scale: Double,
    ): Double {
        if (visibility < 0.16) return config.optionalBudgetFloor

        val share =
            currentCount /
                admittedCount.toDouble().coerceAtLeast(1.0)

        if (
            share >= limit &&
            fairnessBoost < 0.70
        ) {
            return config.optionalBudgetFloor
        }

        return (
            (
                0.18 +
                    0.48 * visibility +
                    0.24 * actorScore +
                    0.10 * fairnessBoost
                ) * scale
            ).coerceIn(
                config.optionalBudgetFloor,
                1.0,
            )
    }

    private fun deferredBudget(
        state: History,
        fairnessBoost: Double,
        reason: String,
    ): Budget =
        Budget(
            body = 0.0,
            face = 0.0,
            eyes = 0.0,
            hands = 0.0,
            denseSurface = 0.0,
            contact = 0.0,
            global4D = 0.0,
            priority = PriorityClass.DEFERRED,
            deferredFrames = state.deferredFrames,
            fairnessBoost = fairnessBoost,
            reason = reason,
        )

    private fun fairnessIndex(active: List<ActorDemand>): Double {
        if (active.isEmpty()) return 1.0
        val service = active.map { history[it.actorId]?.servedFrames?.toDouble() ?: 0.0 }
        val sum = service.sum()
        val squareSum = service.sumOf { it * it }
        if (squareSum <= 1e-12) return 0.0
        return (
            (sum * sum) /
                (service.size.toDouble() * squareSum)
            ).coerceIn(0.0, 1.0)
    }

    private fun evictHistory() {
        val cutoff =
            frameIndex -
                config.maximumDeferralFrames * 3L

        val iterator = history.entries.iterator()
        while (iterator.hasNext()) {
            if (iterator.next().value.lastSeenFrame < cutoff) {
                iterator.remove()
            }
        }
        if (history.size > config.maximumHistoryEntries) {
            history.entries
                .sortedWith(
                    compareBy<Map.Entry<Long, History>> { it.value.servedFrames }
                        .thenBy { it.value.lastSeenFrame }
                        .thenBy { it.key }
                )
                .take(history.size - config.maximumHistoryEntries)
                .forEach { history.remove(it.key) }
        }
    }
}
