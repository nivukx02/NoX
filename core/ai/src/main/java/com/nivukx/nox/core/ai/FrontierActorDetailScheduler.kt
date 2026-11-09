package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Fairness-aware allocation of expensive per-actor modalities.
 *
 * The allocator is deliberately budget-first rather than loop-first:
 *  - every admitted actor receives a bounded body reservation when the frame budget permits;
 *  - residual body budget is distributed proportionally to evidence and motion priority;
 *  - optional face/eye/hand/surface detail is scaled proportionally across actors so early actors
 *    cannot consume the whole frame and starve actors later in the deterministic ordering;
 *  - starvation age remains a priority input, but priority never silently deletes an otherwise
 *    admissible actor merely because another actor scored higher.
 *
 * This is the runtime policy for the 50-actor target profile. The 64-actor ceiling remains a hard
 * safety boundary and is intentionally preserved.
 */
class FrontierActorDetailScheduler(initialConfig: Config = Config()) {

    @Volatile
    private var config: Config = initialConfig

    data class Config(
        val targetActors: Int = 50,
        val hardActorCeiling: Int = 64,
        val frameBudgetUnits: Double = 256.0,
        val minimumBodyShare: Double = 0.36,
        val starvationAgeFrames: Long = 8L,
        val maximumDetailCadenceDivisor: Int = 8,
        /** Minimum fraction of a body's requested budget reserved before residual allocation. */
        val minimumBodyAllocationFraction: Double = 0.55,
    ) {
        init {
            require(targetActors in 1..hardActorCeiling)
            require(hardActorCeiling <= 64)
            require(frameBudgetUnits > 0.0)
            require(minimumBodyShare in 0.0..1.0)
            require(starvationAgeFrames >= 1L)
            require(maximumDetailCadenceDivisor >= 1)
            require(minimumBodyAllocationFraction in 0.1..1.0)
        }
    }

    @Synchronized
    fun configure(updated: Config) {
        config = updated
    }

    fun currentConfig(): Config = config

    data class ActorDemand(
        val actorId: Int,
        val uncertainty: Double,
        val visibility: Double,
        val motionComplexity: Double,
        val interactionRisk: Double,
        val reconstructionValue: Double,
        val ageSinceDetailFrames: Long,
        val bodyCost: Double = 1.0,
        val faceCost: Double = 0.5,
        val eyeCost: Double = 0.35,
        val handCost: Double = 0.60,
        val surfaceCost: Double = 1.25,
        /** Upstream metric-evidence confidence for this actor's current multi-view solve. */
        val metricEvidenceAuthority: Double = 1.0,
        /** Fraction of candidate cameras with usable, independently supported evidence. */
        val cameraCoverage: Double = 1.0,
        /** Optical motion-blur risk for the actor's current source frames, in [0, 1]. */
        val opticalMotionBlurRisk: Double = 0.0,
    ) {
        init {
            require(actorId >= 0)
            listOf(
                uncertainty,
                visibility,
                motionComplexity,
                interactionRisk,
                reconstructionValue,
            ).forEach { require(it in 0.0..1.0) }
            require(ageSinceDetailFrames >= 0L)
            listOf(
                bodyCost,
                faceCost,
                eyeCost,
                handCost,
                surfaceCost,
            ).forEach { require(it.isFinite() && it > 0.0) }
            require(metricEvidenceAuthority in 0.0..1.0)
            require(cameraCoverage in 0.0..1.0)
            require(opticalMotionBlurRisk in 0.0..1.0)
        }
    }

    data class Allocation(
        val actorId: Int,
        val admitted: Boolean,
        val bodyEveryFrames: Int,
        val faceEveryFrames: Int,
        val eyeEveryFrames: Int,
        val handEveryFrames: Int,
        val surfaceEveryFrames: Int,
        val score: Double,
        val budgetUnits: Double,
    )

    data class Result(
        val allocations: List<Allocation>,
        val admittedActorIds: Set<Int>,
        val bodyCoverage: Double,
        val optionalDetailCoverage: Double,
        val fairnessIndex: Double = 1.0,
        val warnings: List<String>,
    )

    private fun priority(demand: ActorDemand): Double {
        val aging =
            min(
                1.0,
                demand.ageSinceDetailFrames.toDouble() /
                    config.starvationAgeFrames.toDouble(),
            )
        return (
            0.24 * demand.uncertainty +
                0.16 * (1.0 - demand.visibility) +
                0.20 * demand.motionComplexity +
                0.18 * demand.interactionRisk +
                0.12 * demand.reconstructionValue +
                0.07 * demand.metricEvidenceAuthority +
                0.04 * demand.cameraCoverage +
                0.07 * aging
            ).coerceIn(0.0, 1.0)
    }

    fun allocate(
        demands: List<ActorDemand>,
        pressure: Double = 0.0,
    ): Result {
        require(pressure in 0.0..1.0)

        val ordered =
            demands
                .distinctBy { it.actorId }
                .sortedWith(
                    compareByDescending<ActorDemand> {
                        it.ageSinceDetailFrames >=
                            config.starvationAgeFrames
                    }
                        .thenByDescending { priority(it) }
                        .thenBy { it.actorId },
                )

        val severe = pressure >= 0.78
        val maximumBodyShare =
            max(
                config.minimumBodyShare,
                if (severe) 0.52 else 0.36,
            )
        val bodyBudget =
            config.frameBudgetUnits *
                maximumBodyShare.coerceIn(0.0, 1.0)

        var admittedCount =
            min(
                ordered.size,
                config.hardActorCeiling,
            )

        /*
         * Reserve a deterministic body floor for every admitted actor whenever feasible. If an
         * adversarial cost profile makes the full ceiling impossible, reduce admission until the
         * reservation fits. This is the only place where actors are removed after prioritization.
         */
        val bodyReservationFraction =
            if (severe) {
                max(
                    config.minimumBodyAllocationFraction,
                    0.65,
                )
            } else {
                config.minimumBodyAllocationFraction
            }

        while (admittedCount > 1) {
            val reservation =
                ordered
                    .take(admittedCount)
                    .sumOf {
                        it.bodyCost * bodyReservationFraction
                    }
            if (reservation <= bodyBudget + 1.0e-9) {
                break
            }
            admittedCount -= 1
        }

        val admitted = ordered.take(admittedCount)
        if (admitted.isEmpty()) {
            return Result(
                allocations = emptyList(),
                admittedActorIds = emptySet(),
                bodyCoverage = 0.0,
                optionalDetailCoverage = 0.0,
                fairnessIndex = 1.0,
                warnings =
                    buildList {
                        if (ordered.size > config.hardActorCeiling) {
                            add("actor-hard-ceiling-applied")
                        }
                        if (ordered.isNotEmpty()) {
                            add("actor-budget-too-small-for-body-reservation")
                        }
                    },
            )
        }

        val scores =
            admitted.associateWith(::priority)
        val rawBodyReservations =
            admitted.associateWith {
                it.bodyCost *
                    bodyReservationFraction
            }
        val rawReservedBody =
            rawBodyReservations.values.sum()
        val reservationScale =
            if (rawReservedBody <= 1.0e-9) {
                1.0
            } else {
                min(
                    1.0,
                    bodyBudget /
                        rawReservedBody,
                )
            }
        val bodyReservations =
            rawBodyReservations.mapValues { (_, value) ->
                value * reservationScale
            }
        val reservedBody =
            bodyReservations.values.sum()
                .coerceAtMost(bodyBudget)
        val remainingFrameAfterReservation =
            (
                config.frameBudgetUnits -
                    reservedBody
                ).coerceAtLeast(0.0)
        val residualBodyBudget =
            (
                bodyBudget -
                    reservedBody
                ).coerceAtLeast(0.0)
                .coerceAtMost(
                    remainingFrameAfterReservation,
                )

        val bodyResidualDemand =
            admitted.sumOf { demand ->
                demand.bodyCost *
                    (
                        0.50 +
                            (scores[demand] ?: 0.0)
                    )
            }.coerceAtLeast(1.0e-9)

        val bodyGrants =
            admitted.associateWith { demand ->
                val reservation =
                    bodyReservations[demand] ?: 0.0
                val residual =
                    residualBodyBudget *
                        (
                            demand.bodyCost *
                                (
                                    0.50 +
                                        (scores[demand] ?: 0.0)
                                    )
                        ) /
                        bodyResidualDemand
                (
                    reservation +
                        residual
                    ).coerceAtMost(
                        demand.bodyCost *
                            (
                                1.0 +
                                    (scores[demand] ?: 0.0)
                                ),
                    )
            }

        val totalBodyGranted =
            bodyGrants.values.sum()
                .coerceAtMost(config.frameBudgetUnits)
        var remainingForOptional =
            (
                config.frameBudgetUnits -
                    totalBodyGranted
            ).coerceAtLeast(0.0)

        val detailDemand =
            admitted.associateWith { demand ->
                val evidenceMultiplier =
                    (
                        0.60 +
                            0.40 *
                                demand.metricEvidenceAuthority
                        ).coerceIn(0.60, 1.0)
                val opticalDetailMultiplier =
                    (
                        1.0 -
                            0.70 *
                                demand.opticalMotionBlurRisk
                    ).coerceIn(0.35, 1.0)
                val detailMultiplier =
                    (
                        if (severe) 0.35 else 1.0
                        ) *
                        evidenceMultiplier *
                        opticalDetailMultiplier
                val face =
                    demand.faceCost *
                        (
                            0.45 +
                                (scores[demand] ?: 0.0)
                            ) *
                        detailMultiplier
                val eye =
                    demand.eyeCost *
                        (
                            0.35 +
                                (scores[demand] ?: 0.0)
                            ) *
                        detailMultiplier
                val hand =
                    demand.handCost *
                        (
                            0.40 +
                                (scores[demand] ?: 0.0)
                            ) *
                        detailMultiplier
                val surface =
                    demand.surfaceCost *
                        (
                            0.30 +
                                (scores[demand] ?: 0.0)
                            ) *
                        detailMultiplier
                DetailGrant(
                    face = face,
                    eye = eye,
                    hand = hand,
                    surface = surface,
                    total =
                        face +
                            eye +
                            hand +
                            surface,
                )
            }

        val totalDetailDemand =
            detailDemand.values
                .sumOf { it.total }
                .coerceAtLeast(1.0e-9)
        val detailScale =
            min(
                1.0,
                remainingForOptional /
                    totalDetailDemand,
            )

        data class MutableAllocation(
            val demand: ActorDemand,
            val score: Double,
            val body: Double,
            val face: Double,
            val eye: Double,
            val hand: Double,
            val surface: Double,
        )

        val mutable =
            admitted.map { demand ->
                val detail =
                    detailDemand[demand]
                        ?: DetailGrant.ZERO
                val face =
                    detail.face * detailScale
                val eye =
                    detail.eye * detailScale
                val hand =
                    detail.hand * detailScale
                val surface =
                    detail.surface * detailScale
                MutableAllocation(
                    demand = demand,
                    score = scores[demand] ?: 0.0,
                    body = bodyGrants[demand] ?: 0.0,
                    face = face,
                    eye = eye,
                    hand = hand,
                    surface = surface,
                )
            }

        remainingForOptional =
            (
                remainingForOptional -
                    mutable.sumOf {
                        it.face +
                            it.eye +
                            it.hand +
                            it.surface
                    }
            ).coerceAtLeast(0.0)

        val allocations =
            mutable.map { current ->
                val demand = current.demand
                val bodyRatio =
                    current.body /
                        demand.bodyCost
                            .coerceAtLeast(1.0e-9)
                val bodyCadence =
                    cadenceFromRatio(
                        ratio = bodyRatio,
                        fullRatio = 0.95,
                    )

                Allocation(
                    actorId = demand.actorId,
                    admitted = current.body > 1.0e-9,
                    bodyEveryFrames = bodyCadence,
                    faceEveryFrames =
                        cadenceFromGranted(
                            granted = current.face,
                            requested =
                                demand.faceCost *
                                    (
                                        0.45 +
                                            current.score
                                    ),
                        ),
                    eyeEveryFrames =
                        cadenceFromGranted(
                            granted = current.eye,
                            requested =
                                demand.eyeCost *
                                    (
                                        0.35 +
                                            current.score
                                    ),
                        ),
                    handEveryFrames =
                        cadenceFromGranted(
                            granted = current.hand,
                            requested =
                                demand.handCost *
                                    (
                                        0.40 +
                                            current.score
                                    ),
                        ),
                    surfaceEveryFrames =
                        max(
                            bodyCadence,
                            cadenceFromGranted(
                                granted = current.surface,
                                requested =
                                    demand.surfaceCost *
                                        (
                                            0.30 +
                                                current.score
                                        ),
                            ),
                        ),
                    score = current.score,
                    budgetUnits =
                        (
                            current.body +
                                current.face +
                                current.eye +
                                current.hand +
                                current.surface
                            ).coerceAtLeast(0.0),
                )
            }

        val admittedActorIds =
            allocations
                .filter { it.admitted }
                .map { it.actorId }
                .toSet()

        val bodyCoverage =
            if (admitted.isEmpty()) {
                0.0
            } else {
                admittedActorIds.size.toDouble() /
                    admitted.size.toDouble()
            }

        val optionalCoverage =
            if (allocations.isEmpty()) {
                0.0
            } else {
                allocations.count {
                    it.faceEveryFrames <= 2 ||
                        it.handEveryFrames <= 2
                }.toDouble() /
                    allocations.size.toDouble()
            }

        val fairnessIndex =
            jainsFairnessIndex(
                allocations.map { it.budgetUnits },
            )

        val warnings =
            buildList {
                if (admitted.any { it.opticalMotionBlurRisk >= 0.55 }) {
                    add("optional-detail-optical-blur-limited")
                }
                if (ordered.size > config.hardActorCeiling) {
                    add("actor-hard-ceiling-applied")
                }
                if (admittedCount < min(
                        ordered.size,
                        config.hardActorCeiling,
                    )) {
                    add("actor-body-reservation-ceiling-applied")
                }
                if (severe) {
                    add("severe-pressure-optional-detail-shed")
                }
                if (remainingForOptional <= 1.0e-9 &&
                    totalDetailDemand > 0.0
                ) {
                    add("detail-budget-exhausted")
                }
                if (fairnessIndex < 0.75) {
                    add("actor-detail-fairness-limited")
                }
            }

        return Result(
            allocations = allocations,
            admittedActorIds = admittedActorIds,
            bodyCoverage = bodyCoverage,
            optionalDetailCoverage = optionalCoverage,
            fairnessIndex = fairnessIndex,
            warnings = warnings,
        )
    }

    private data class DetailGrant(
        val face: Double,
        val eye: Double,
        val hand: Double,
        val surface: Double,
        val total: Double,
    ) {
        companion object {
            val ZERO =
                DetailGrant(
                    face = 0.0,
                    eye = 0.0,
                    hand = 0.0,
                    surface = 0.0,
                    total = 0.0,
                )
        }
    }

    private fun cadenceFromGranted(
        granted: Double,
        requested: Double,
    ): Int {
        val ratio =
            if (requested <= 1.0e-9) {
                0.0
            } else {
                granted /
                    requested
            }
        return cadenceFromRatio(
            ratio = ratio,
            fullRatio = 0.95,
        )
    }

    private fun cadenceFromRatio(
        ratio: Double,
        fullRatio: Double,
    ): Int {
        val normalized =
            ratio.coerceIn(0.0, 1.0)
        if (normalized >= fullRatio) {
            return 1
        }
        return (
            1.0 +
                (
                    (1.0 - normalized) *
                        (
                            config.maximumDetailCadenceDivisor -
                                1
                            )
                    )
            ).toInt()
                .coerceIn(
                    1,
                    config.maximumDetailCadenceDivisor,
                )
    }

    private fun jainsFairnessIndex(
        values: List<Double>,
    ): Double {
        if (values.isEmpty()) return 1.0
        val sanitized =
            values.map {
                it.coerceAtLeast(0.0)
            }
        val sum =
            sanitized.sum()
        if (sum <= 1.0e-9) return 1.0
        val squared =
            sanitized.sumOf { it * it }
                .coerceAtLeast(1.0e-9)
        return (
            (sum * sum) /
                (
                    sanitized.size.toDouble() *
                        squared
                    )
            ).coerceIn(0.0, 1.0)
    }
}
