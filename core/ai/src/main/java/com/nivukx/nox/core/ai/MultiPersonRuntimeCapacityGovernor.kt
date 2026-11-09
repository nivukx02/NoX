package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.ActorRuntimeBudget
import com.nivukx.nox.core.model.MultiPersonCapacityPolicy
import com.nivukx.nox.core.model.MultiPersonRuntimeBudget
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Bounded multi-person resource governor for high-density markerless motion capture.
 *
 * The governor does not perform detection itself. It prevents downstream engines from accidentally
 * allocating per-subject state without a hard bound, and it distributes body/face/eye/hand/surface
 * budgets according to evidence priority and current device headroom.
 *
 * Target operation is 50 subjects with a 64-subject hard ceiling. This is a capacity contract, not
 * a claim that a particular phone camera or NPU can maintain full-resolution dense inference for
 * fifty people at every requested modality simultaneously. Runtime admission is evidence- and
 * hardware-aware and reports degraded modalities rather than silently fabricating tracking quality.
 */
class MultiPersonRuntimeCapacityGovernor(
    private var policy: MultiPersonCapacityPolicy = MultiPersonCapacityPolicy.fiftyPersonProduction(),
    private val minimumBodyHz: Int = 24,
    private val preferredBodyHz: Int = 60,
    private val fairnessWindowCycles: Long = 10L,
    private val maximumAgingBoost: Double = 0.22,
) {
    private var allocationCycle: Long = 0L
    private val lastAdmittedCycle = HashMap<Long, Long>()
    private val firstSeenCycle = HashMap<Long, Long>()

    init {
        require(fairnessWindowCycles >= 1L)
        require(maximumAgingBoost in 0.0..0.50)
    }
    data class RuntimeHeadroom(
        val cpuHeadroom: Double,
        val gpuHeadroom: Double,
        val npuHeadroom: Double,
        val thermalHeadroom: Double,
        val memoryHeadroom: Double,
    ) {
        init {
            require(cpuHeadroom in 0.0..1.0)
            require(gpuHeadroom in 0.0..1.0)
            require(npuHeadroom in 0.0..1.0)
            require(thermalHeadroom in 0.0..1.0)
            require(memoryHeadroom in 0.0..1.0)
        }

        val composite: Double
            get() =
                min(
                    cpuHeadroom,
                    min(
                        gpuHeadroom,
                        min(
                            npuHeadroom,
                            min(thermalHeadroom, memoryHeadroom),
                        ),
                    ),
                ).coerceIn(0.0, 1.0)
    }

    data class SubjectDemand(
        val trackId: Long,
        val confidence: Double,
        val visibleRatio: Double,
        val motionEnergy: Double,
        val faceRequested: Boolean,
        val eyeRequested: Boolean,
        val handsRequested: Boolean,
        val surfaceRequested: Boolean,
        val interactionPriority: Double = 0.0,
        val identityPriority: Double = 1.0,
    ) {
        init {
            require(trackId >= 0L)
            require(confidence in 0.0..1.0)
            require(visibleRatio in 0.0..1.0)
            require(motionEnergy in 0.0..1.0)
            require(interactionPriority in 0.0..1.0)
            require(identityPriority in 0.0..1.0)
        }

        val priority: Double
            get() =
                (
                    0.26 * confidence +
                        0.18 * visibleRatio +
                        0.18 * motionEnergy +
                        0.20 * interactionPriority +
                        0.18 * identityPriority
                    ).coerceIn(0.0, 1.0)
    }

    data class Allocation(
        val budgets: List<ActorRuntimeBudget>,
        val runtime: MultiPersonRuntimeBudget,
        val admittedSubjectIds: List<Long>,
        val heldSubjectIds: List<Long>,
        val fairnessBoostedSubjectIds: List<Long> = emptyList(),
        val starvationRisk: Double = 0.0,
        val warnings: List<String>,
    )

    fun configure(newPolicy: MultiPersonCapacityPolicy) {
        policy = newPolicy
    }

    fun allocate(
        demands: Collection<SubjectDemand>,
        headroom: RuntimeHeadroom,
        captureFps: Int,
    ): Allocation {
        require(captureFps in 15..240)

        allocationCycle += 1L

        /*
         * Priority-only scheduling can starve lower-confidence actors during long crowded takes.
         * The bounded aging boost increases admission probability for repeatedly deferred identities
         * without allowing weak evidence to overwhelm a high-confidence active actor.
         */
        val unique = demands
            .distinctBy { it.trackId }
            .map { demand ->
                val firstSeen =
                    firstSeenCycle.getOrPut(
                        demand.trackId,
                    ) { allocationCycle }
                val lastServed = lastAdmittedCycle[demand.trackId]
                val ageCycles =
                    if (lastServed == null) {
                        (allocationCycle - firstSeen)
                            .coerceAtLeast(0L)
                            .toDouble()
                    } else {
                        (allocationCycle - lastServed)
                            .coerceAtLeast(0L)
                            .toDouble()
                    }
                val agingBoost =
                    (
                        ageCycles /
                            fairnessWindowCycles.toDouble()
                    ).coerceIn(0.0, 1.0) *
                        maximumAgingBoost
                demand to agingBoost
            }
            .sortedWith(
                compareByDescending<Pair<SubjectDemand, Double>> {
                    (
                        it.first.priority + it.second
                    ).coerceIn(0.0, 1.0)
                }.thenBy { it.first.trackId },
            )

        val admitted = unique
            .take(policy.hardSubjectCeiling)

        val targetOverflow = max(0, unique.size - policy.targetSubjects)
        val hardOverflow = max(0, unique.size - policy.hardSubjectCeiling)
        val admittedIds =
            admitted.map { it.first.trackId }.toSet()
        val fairnessBoosted =
            unique
                .filter { (_, boost) ->
                    boost > 0.001
                }
                .map { it.first.trackId }
                .filter { it in admittedIds }

        val capacityRatio =
            (
                headroom.composite *
                    thermalAdmission(headroom.thermalHeadroom) *
                    memoryAdmission(headroom.memoryHeadroom)
                ).coerceIn(0.15, 1.0)

        val bodyHz =
            (
                minimumBodyHz +
                    (preferredBodyHz - minimumBodyHz) * capacityRatio
                ).roundToInt().coerceIn(15, captureFps)

        admitted.forEach { (demand, _) ->
            lastAdmittedCycle[demand.trackId] = allocationCycle
        }
        lastAdmittedCycle.keys.removeIf { id ->
            unique.none { it.first.trackId == id }
        }
        firstSeenCycle.keys.removeIf { id ->
            unique.none { it.first.trackId == id }
        }

        val maxObservedAge =
            unique.maxOfOrNull { (_, boost) ->
                if (maximumAgingBoost <= 1e-9) 0.0
                else (boost / maximumAgingBoost)
            } ?: 0.0

        val budgets = admitted.mapIndexed { index, entry ->
            val demand = entry.first
            val agingBoost = entry.second
            val rankFactor =
                1.0 -
                    (index.toDouble() /
                        max(1, admitted.size - 1).toDouble())
                        .coerceIn(0.0, 1.0)

            val priority =
                (
                    0.55 *
                        (
                            demand.priority +
                                agingBoost
                        ).coerceIn(0.0, 1.0) +
                        0.45 * rankFactor
                    ).coerceIn(0.0, 1.0)

            val faceHz =
                when {
                    demand.faceRequested && headroom.gpuHeadroom >= 0.40 ->
                        (18 + 42 * priority * headroom.gpuHeadroom)
                            .roundToInt()
                            .coerceIn(12, captureFps)
                    demand.faceRequested -> 15
                    else -> 0
                }

            val eyeHz =
                when {
                    demand.eyeRequested && headroom.npuHeadroom >= 0.50 ->
                        (24 + 72 * priority * headroom.npuHeadroom)
                            .roundToInt()
                            .coerceIn(20, 120)
                    demand.eyeRequested -> 20
                    else -> 0
                }

            val handHz =
                when {
                    demand.handsRequested && headroom.npuHeadroom >= 0.45 ->
                        (15 + 45 * priority * headroom.npuHeadroom)
                            .roundToInt()
                            .coerceIn(12, min(90, captureFps))
                    demand.handsRequested -> 12
                    else -> 0
                }

            val surfaceHz =
                when {
                    demand.surfaceRequested && headroom.gpuHeadroom >= 0.65 &&
                        headroom.memoryHeadroom >= 0.55 ->
                        (6 + 30 * priority * headroom.gpuHeadroom)
                            .roundToInt()
                            .coerceIn(4, min(60, captureFps))
                    demand.surfaceRequested -> 0
                    else -> 0
                }

            val reserveBytes =
                (
                    96 * 1024 +
                        (224 * 1024 * priority) +
                        (if (demand.faceRequested) 48 * 1024 else 0) +
                        (if (demand.eyeRequested) 32 * 1024 else 0) +
                        (if (demand.handsRequested) 96 * 1024 else 0) +
                        (if (demand.surfaceRequested) 256 * 1024 else 0)
                    )
                    .roundToInt()
                    .coerceIn(64 * 1024, 2 * 1024 * 1024)

            ActorRuntimeBudget(
                trackId = demand.trackId,
                ordinal = index,
                bodyInferenceHz = bodyHz,
                faceInferenceHz = faceHz,
                eyeInferenceHz = eyeHz,
                handInferenceHz = handHz,
                denseSurfaceHz = surfaceHz,
                reconstructionPriority = priority,
                reserveStateBytes = reserveBytes,
                occlusionPredictionFrames =
                    if (policy.enableOcclusionPrediction) {
                        (
                            18 +
                                96 *
                                priority *
                                headroom.composite
                            ).roundToInt().coerceIn(8, 120)
                    } else {
                        0
                    },
                identityReserve = index < policy.targetSubjects,
            )
        }

        val totalBytes = budgets.sumOf { it.reserveStateBytes.toLong() }
        val predicted =
            budgets.count { it.occlusionPredictionFrames > 0 }
        val highPriority =
            budgets.count { it.reconstructionPriority >= 0.70 }

        val visualAdmission =
            when {
                admitted.isEmpty() -> 0.0
                headroom.composite >= 0.70 -> 1.0
                headroom.composite >= 0.45 -> 0.82
                else -> 0.56
            }

        val warnings = buildList {
            if (targetOverflow > 0) {
                add(
                    "Scene contains $targetOverflow subject(s) beyond the 50-subject production target; " +
                        "tracks remain bounded and lower-priority modalities may be throttled.",
                )
            }
            if (hardOverflow > 0) {
                add(
                    "Scene exceeded the 64-subject hard ceiling; " +
                        "$hardOverflow track candidate(s) were held outside the reconstruction graph.",
                )
            }
            if (headroom.memoryHeadroom < 0.35) {
                add("Memory headroom is low; dense surface budgets are admission-controlled.")
            }
            if (headroom.thermalHeadroom < 0.35) {
                add("Thermal headroom is low; body tracking is retained while optional modalities may reduce cadence.")
            }
            if (headroom.npuHeadroom < 0.30) {
                add("NPU headroom is low; eye/hand inference cadence is reduced before body tracking is sacrificed.")
            }
            if (
                maxObservedAge >= 0.80 &&
                unique.size > policy.targetSubjects
            ) {
                add(
                    "Crowd scheduling is using aging fairness to prevent prolonged actor starvation.",
                )
            }
        }

        return Allocation(
            budgets = budgets,
            runtime =
                MultiPersonRuntimeBudget(
                    subjectCount = unique.size,
                    targetSubjectCount = policy.targetSubjects,
                    hardSubjectCeiling = policy.hardSubjectCeiling,
                    totalReservedBytes = totalBytes,
                    activeTrackCount = admitted.size,
                    predictedTrackCount = predicted,
                    highPriorityTrackCount = highPriority,
                    visualEvidenceAdmissionRatio = visualAdmission,
                    warning = warnings.firstOrNull(),
                ),
            admittedSubjectIds = admitted.map { it.first.trackId },
            heldSubjectIds =
                unique.drop(policy.hardSubjectCeiling)
                    .map { it.first.trackId },
            fairnessBoostedSubjectIds = fairnessBoosted,
            starvationRisk = maxObservedAge.coerceIn(0.0, 1.0),
            warnings = warnings,
        )
    }

    data class ActorBudgetView(
        val bodyHz: Int,
        val faceHz: Int,
        val eyeHz: Int,
        val handHz: Int,
        val surfaceHz: Int,
        val reconstructionPriority: Double,
        val reserveStateBytes: Int,
    )

    fun reset() {
        allocationCycle = 0L
        lastAdmittedCycle.clear()
        firstSeenCycle.clear()
    }

    private fun thermalAdmission(value: Double): Double =
        when {
            value >= 0.80 -> 1.0
            value >= 0.55 -> 0.90
            value >= 0.35 -> 0.72
            else -> 0.52
        }

    private fun memoryAdmission(value: Double): Double =
        when {
            value >= 0.80 -> 1.0
            value >= 0.60 -> 0.92
            value >= 0.40 -> 0.78
            else -> 0.58
        }

    companion object {
        const val PRODUCTION_TARGET_SUBJECTS = 50
        const val HARD_SUBJECT_CEILING = 64
    }
}
