package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.ActorDetailPriorityMode
import com.nivukx.nox.core.model.MotionCaptureCustomization
import kotlin.math.max
import kotlin.math.min

/**
 * Allocates optional high-cost modalities across a crowded scene without removing canonical body
 * identities. Every actor remains represented in the body/identity graph; only expensive detail
 * workloads are scheduled according to evidence quality, motion, visibility, interaction and
 * identity stability.
 */
class ActorDetailAllocationEngine {

    private var allocationEpoch = 0L
    private val lastDetailedEpoch = LinkedHashMap<Long, Long>()

    enum class DetailTier {
        FULL,
        FACE,
        EYE,
        HAND,
        SURFACE,
        TEMPORAL_PREDICTION,
    }

    data class ActorEvidence(
        val actorId: Long,
        val confidence: Double,
        val visibleRatio: Double,
        val motionEnergy: Double,
        val interactionPriority: Double,
        val identityContinuity: Double,
        val occlusionRisk: Double,
        val screenAreaRatio: Double,
    ) {
        init {
            require(actorId >= 0L)
            require(confidence in 0.0..1.0)
            require(visibleRatio in 0.0..1.0)
            require(motionEnergy in 0.0..1.0)
            require(interactionPriority in 0.0..1.0)
            require(identityContinuity in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
            require(screenAreaRatio in 0.0..1.0)
        }
    }

    data class ActorAllocation(
        val actorId: Long,
        val rank: Int,
        val priorityScore: Double,
        val tiers: Set<DetailTier>,
        val bodyCadenceScale: Double,
        val optionalCadenceScale: Double,
        val preservedDuringOcclusion: Boolean,
    ) {
        val fullDetail: Boolean
            get() = DetailTier.FULL in tiers
    }

    data class Result(
        val allocations: List<ActorAllocation>,
        val fullDetailActorIds: Set<Long>,
        val faceDetailActorIds: Set<Long>,
        val eyeDetailActorIds: Set<Long>,
        val handDetailActorIds: Set<Long>,
        val denseSurfaceActorIds: Set<Long>,
        val predictionActorIds: Set<Long>,
        val warnings: List<String>,
    ) {
        fun allocation(actorId: Long): ActorAllocation? =
            allocations.firstOrNull { it.actorId == actorId }
    }

    @Synchronized
    fun allocate(
        evidence: Collection<ActorEvidence>,
        customization: MotionCaptureCustomization,
    ): Result {
        allocationEpoch += 1L

        val candidates =
            evidence
                .distinctBy { it.actorId }
                .sortedWith(
                    compareByDescending<ActorEvidence> {
                        effectivePriority(
                            evidence = it,
                            mode = customization.actorDetailPriorityMode,
                        )
                    }.thenBy { it.actorId },
                )
                .take(customization.maxSubjects.coerceIn(1, 64))

        val maxActors = candidates.size.coerceAtLeast(1)

        val fullBudget =
            customization.fullDetailActorBudget
                .coerceIn(1, maxActors)
        val faceBudget =
            customization.faceDetailActorBudget
                .coerceIn(1, maxActors)
        val eyeBudget =
            customization.eyeDetailActorBudget
                .coerceIn(1, maxActors)
        val handBudget =
            customization.handDetailActorBudget
                .coerceIn(1, maxActors)
        val surfaceBudget =
            customization.denseSurfaceActorBudget
                .coerceIn(1, maxActors)

        val allocations =
            candidates.mapIndexed { index, actor ->
                val normalizedRank =
                    if (candidates.size <= 1) {
                        1.0
                    } else {
                        1.0 -
                            index.toDouble() /
                            (candidates.size - 1).toDouble()
                    }

                val tiers =
                    buildSet {
                        if (index < fullBudget) add(DetailTier.FULL)
                        if (index < faceBudget) add(DetailTier.FACE)
                        if (index < eyeBudget) add(DetailTier.EYE)
                        if (index < handBudget) add(DetailTier.HAND)
                        if (index < surfaceBudget) add(DetailTier.SURFACE)

                        if (
                            actor.occlusionRisk >= 0.55 ||
                            actor.identityContinuity < 0.55
                        ) {
                            add(DetailTier.TEMPORAL_PREDICTION)
                        }
                    }

                val optionalCadence =
                    (
                        0.35 +
                            0.65 *
                            (
                                0.45 * normalizedRank +
                                    0.25 * actor.visibleRatio +
                                    0.18 * actor.identityContinuity +
                                    0.12 * actor.confidence
                                )
                        ).coerceIn(0.25, 1.0)

                val priorityScore =
                    effectivePriority(
                        evidence = actor,
                        mode = customization.actorDetailPriorityMode,
                    )

                ActorAllocation(
                    actorId = actor.actorId,
                    rank = index,
                    priorityScore = priorityScore,
                    tiers = tiers,
                    bodyCadenceScale =
                        (
                            0.75 +
                                0.25 *
                                (
                                    actor.confidence * 0.50 +
                                        actor.visibleRatio * 0.30 +
                                        actor.identityContinuity * 0.20
                                    )
                            ).coerceIn(0.50, 1.0),
                    optionalCadenceScale =
                        optionalCadence,
                    preservedDuringOcclusion =
                        actor.occlusionRisk >= 0.40 ||
                            customization.actorOcclusionRecovery >= 0.70,
                )
            }

        val warnings = buildList {
            if (evidence.size > customization.maxSubjects) {
                add(
                    "DETAIL_ALLOCATOR_SCENE_EXCEEDS_GLOBAL_SUBJECT_LIMIT",
                )
            }
            if (candidates.size > surfaceBudget && surfaceBudget < maxActors) {
                add("DETAIL_ALLOCATOR_AGING_FAIRNESS_ACTIVE")
            }
            if (candidates.isNotEmpty() &&
                candidates.count {
                    DetailTier.SURFACE in (
                        allocations.first {
                            allocation ->
                            allocation.actorId == it.actorId
                        }.tiers
                    )
                } < candidates.size
            ) {
                add(
                    "DENSE_SURFACE_DETAIL_IS_BUDGETED_BY_ACTOR_PRIORITY",
                )
            }
            if (
                customization.actorDetailPriorityMode ==
                    ActorDetailPriorityMode.UNIFORM &&
                candidates.size > surfaceBudget
            ) {

                add(
                    "UNIFORM_MODE_STILL_USES_EXPLICIT_SURFACE_BUDGET",
                )
            }
        }

        allocations.forEach { allocation ->
            if (
                DetailTier.FULL in allocation.tiers ||
                DetailTier.SURFACE in allocation.tiers ||
                DetailTier.FACE in allocation.tiers ||
                DetailTier.EYE in allocation.tiers ||
                DetailTier.HAND in allocation.tiers
            ) {
                lastDetailedEpoch[allocation.actorId] =
                    allocationEpoch
            }
        }

        while (
            lastDetailedEpoch.size >
                (customization.maxSubjects.coerceIn(1, 64) * 2)
        ) {
            val staleActor =
                lastDetailedEpoch.minByOrNull { it.value }?.key
                    ?: break
            lastDetailedEpoch.remove(staleActor)
        }

        return Result(
            allocations = allocations,
            fullDetailActorIds =
                allocations
                    .filter { DetailTier.FULL in it.tiers }
                    .map { it.actorId }
                    .toSet(),
            faceDetailActorIds =
                allocations
                    .filter { DetailTier.FACE in it.tiers }
                    .map { it.actorId }
                    .toSet(),
            eyeDetailActorIds =
                allocations
                    .filter { DetailTier.EYE in it.tiers }
                    .map { it.actorId }
                    .toSet(),
            handDetailActorIds =
                allocations
                    .filter { DetailTier.HAND in it.tiers }
                    .map { it.actorId }
                    .toSet(),
            denseSurfaceActorIds =
                allocations
                    .filter { DetailTier.SURFACE in it.tiers }
                    .map { it.actorId }
                    .toSet(),
            predictionActorIds =
                allocations
                    .filter {
                        DetailTier.TEMPORAL_PREDICTION in
                            it.tiers
                    }
                    .map { it.actorId }
                    .toSet(),
            warnings = warnings.distinct(),
        )
    }

    private fun effectivePriority(
        evidence: ActorEvidence,
        mode: ActorDetailPriorityMode,
    ): Double {
        val base =
            score(
                evidence = evidence,
                mode = mode,
            )
        val last =
            lastDetailedEpoch[evidence.actorId]
        val age =
            if (last == null) {
                32L
            } else {
                (allocationEpoch - last).coerceAtLeast(0L)
            }

        /*
         * Aging is deliberately bounded. A weak actor should not leapfrog a clearly superior
         * subject, but a persistently skipped actor should eventually receive expensive detail.
         */
        val fairnessBoost =
            min(
                0.18,
                age.toDouble() * 0.006,
            )

        return (base + fairnessBoost)
            .coerceIn(0.0, 1.0)
    }

    private fun score(
        evidence: ActorEvidence,
        mode: ActorDetailPriorityMode,
    ): Double {
        val base =
            when (mode) {
                ActorDetailPriorityMode.BALANCED ->
                    0.26 * evidence.confidence +
                        0.19 * evidence.visibleRatio +
                        0.16 * evidence.motionEnergy +
                        0.12 * evidence.interactionPriority +
                        0.17 * evidence.identityContinuity +
                        0.06 * (1.0 - evidence.occlusionRisk) +
                        0.04 * evidence.screenAreaRatio

                ActorDetailPriorityMode.MOTION_FIRST ->
                    0.42 * evidence.motionEnergy +
                        0.22 * evidence.confidence +
                        0.14 * evidence.visibleRatio +
                        0.12 * evidence.identityContinuity +
                        0.06 * evidence.interactionPriority +
                        0.04 * evidence.screenAreaRatio

                ActorDetailPriorityMode.VISIBILITY_FIRST ->
                    0.46 * evidence.visibleRatio +
                        0.22 * evidence.confidence +
                        0.14 * evidence.screenAreaRatio +
                        0.08 * evidence.identityContinuity +
                        0.06 * evidence.motionEnergy +
                        0.04 * evidence.interactionPriority

                ActorDetailPriorityMode.PROXIMITY_FIRST ->
                    0.44 * evidence.interactionPriority +
                        0.20 * evidence.visibleRatio +
                        0.16 * evidence.motionEnergy +
                        0.10 * evidence.confidence +
                        0.06 * evidence.identityContinuity +
                        0.04 * evidence.screenAreaRatio

                ActorDetailPriorityMode.IDENTITY_FIRST ->
                    0.48 * evidence.identityContinuity +
                        0.18 * evidence.confidence +
                        0.12 * evidence.visibleRatio +
                        0.10 * (1.0 - evidence.occlusionRisk) +
                        0.08 * evidence.motionEnergy +
                        0.04 * evidence.interactionPriority

                ActorDetailPriorityMode.UNIFORM ->
                    0.50 +
                        0.15 * evidence.confidence +
                        0.10 * evidence.visibleRatio +
                        0.10 * evidence.identityContinuity +
                        0.05 * evidence.motionEnergy +
                        0.05 * evidence.interactionPriority +
                        0.05 * evidence.screenAreaRatio
            }

        return min(
            1.0,
            max(0.0, base),
        )
    }
}
