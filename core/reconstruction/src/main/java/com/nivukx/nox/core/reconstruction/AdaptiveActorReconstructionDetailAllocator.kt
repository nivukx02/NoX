package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min

/**
 * Allocates bounded reconstruction detail across actors.
 *
 * Priority is evidence-driven rather than actor-count-driven:
 * uncertainty, occlusion, motion complexity, interaction risk, identity safety, visible coverage,
 * camera support and operator priority all influence the allocation.
 *
 * The allocator produces a plan. It never fabricates geometry or silently discards authoritative
 * body state when detail budgets are constrained.
 */
class AdaptiveActorReconstructionDetailAllocator(
    private val config: Config = Config(),
) {

    enum class Tier {
        HOLD,
        CORE,
        STANDARD,
        FINE,
        ULTRA,
    }

    data class Config(
        val maximumActors: Int = 64,
        val minimumSurfaceSamplesPerActor: Int = 2048,
        val maximumSurfaceSamplesPerActor: Int = 64_000,
        val minimumGaussianBudgetPerActor: Int = 1024,
        val maximumGaussianBudgetPerActor: Int = 96_000,
        val globalSurfaceSampleBudget: Int = 750_000,
        val globalGaussianBudget: Int = 1_500_000,
        val coreBodyHz: Int = 30,
        val standardBodyHz: Int = 45,
        val fineBodyHz: Int = 60,
        val ultraBodyHz: Int = 90,
        val memoryBytesPerSurfaceSample: Int = 64,
        val memoryBytesPerGaussian: Int = 96,
        val uncertaintyWeight: Double = 0.22,
        val occlusionWeight: Double = 0.16,
        val motionWeight: Double = 0.14,
        val interactionWeight: Double = 0.12,
        val visibilityWeight: Double = 0.08,
        val identityWeight: Double = 0.10,
        val cameraSupportWeight: Double = 0.08,
        val operatorPriorityWeight: Double = 0.10,
    ) {
        init {
            require(maximumActors in 1..64)
            require(minimumSurfaceSamplesPerActor >= 256)
            require(maximumSurfaceSamplesPerActor >= minimumSurfaceSamplesPerActor)
            require(minimumGaussianBudgetPerActor >= 128)
            require(maximumGaussianBudgetPerActor >= minimumGaussianBudgetPerActor)
            require(globalSurfaceSampleBudget >= minimumSurfaceSamplesPerActor)
            require(globalGaussianBudget >= minimumGaussianBudgetPerActor)
            require(coreBodyHz >= 1)
            require(standardBodyHz >= coreBodyHz)
            require(fineBodyHz >= standardBodyHz)
            require(ultraBodyHz >= fineBodyHz)
            require(memoryBytesPerSurfaceSample > 0)
            require(memoryBytesPerGaussian > 0)
            listOf(
                uncertaintyWeight,
                occlusionWeight,
                motionWeight,
                interactionWeight,
                visibilityWeight,
                identityWeight,
                cameraSupportWeight,
                operatorPriorityWeight,
            ).forEach { require(it >= 0.0) }
        }
    }

    data class ActorRequest(
        val actorId: Long,
        val operatorPriority: Double,
        val identityConfidence: Double,
        val visibilityRatio: Double,
        val occlusionRatio: Double,
        val motionComplexity: Double,
        val interactionRisk: Double,
        val cameraSupportCount: Int,
        val requestedSurfaceDetail: Boolean = true,
        val requestedFaceDetail: Boolean = true,
        val requestedEyeDetail: Boolean = true,
        val requestedHandDetail: Boolean = true,
        val faceGeometryConfidence: Double = 1.0,
        val eyeGeometryConfidence: Double = 1.0,
        val handGeometryConfidence: Double = 1.0,
        val surfaceGeometryConfidence: Double = 1.0,
        val occlusionRecoveryConfidence: Double = 1.0,
        val identityStability: Double = 1.0,
    ) {
        init {
            require(actorId >= 0L)
            listOf(
                operatorPriority,
                identityConfidence,
                visibilityRatio,
                occlusionRatio,
                motionComplexity,
                interactionRisk,
            ).forEach { require(it in 0.0..1.0) }
            require(cameraSupportCount >= 0)
            listOf(
                faceGeometryConfidence,
                eyeGeometryConfidence,
                handGeometryConfidence,
                surfaceGeometryConfidence,
                occlusionRecoveryConfidence,
                identityStability,
            ).forEach { require(it in 0.0..1.0) }
        }
    }

    data class Allocation(
        val actorId: Long,
        val tier: Tier,
        val priorityScore: Double,
        val surfaceSamplesPerFrame: Int,
        val gaussianBudget: Int,
        val faceMeshBudget: Int,
        val eyeSamplesPerFrame: Int,
        val handSamplesPerFrame: Int,
        val bodyUpdateHz: Int,
        val detailUpdateHz: Int,
        val memoryBudgetBytes: Long,
        val confidence: Double,
        val reasons: List<String>,
    )

    data class Result(
        val allocations: List<Allocation>,
        val totalSurfaceSamplesPerFrame: Int,
        val totalGaussianBudget: Int,
        val totalMemoryBudgetBytes: Long,
        val ultraActorCount: Int,
        val fineOrHigherActorCount: Int,
        val coreActorCount: Int,
        val heldActorCount: Int,
        val budgetUtilization: Double,
        val starvationRisk: Double,
        val warnings: List<String>,
    )

    private val previousTiers =
        LinkedHashMap<Long, Tier>()

    @Synchronized
    fun allocate(
        actors: Collection<ActorRequest>,
        memoryHeadroom: Double,
        gpuHeadroom: Double,
        npuHeadroom: Double,
    ): Result {
        require(memoryHeadroom in 0.0..1.0)
        require(gpuHeadroom in 0.0..1.0)
        require(npuHeadroom in 0.0..1.0)

        val bounded =
            actors
                .distinctBy { it.actorId }
                .take(config.maximumActors)

        if (bounded.isEmpty()) {
            return Result(
                allocations = emptyList(),
                totalSurfaceSamplesPerFrame = 0,
                totalGaussianBudget = 0,
                totalMemoryBudgetBytes = 0L,
                ultraActorCount = 0,
                fineOrHigherActorCount = 0,
                coreActorCount = 0,
                heldActorCount = 0,
                budgetUtilization = 0.0,
                starvationRisk = 0.0,
                warnings = emptyList(),
            )
        }

        val capacityScale =
            (
                0.34 * memoryHeadroom +
                    0.38 * gpuHeadroom +
                    0.28 * npuHeadroom
                )
                .coerceIn(0.08, 1.0)

        val ranked =
            bounded
                .map { actor ->
                    val uncertainty =
                        (
                            1.0 -
                                actor.identityConfidence
                        ).coerceIn(0.0, 1.0)
                    val occlusion =
                        actor.occlusionRatio
                    val cameraSupport =
                        (
                            actor.cameraSupportCount.toDouble() /
                                12.0
                            )
                            .coerceIn(0.0, 1.0)

                    val modalityUncertainty =
                        (
                            0.20 * (1.0 - actor.faceGeometryConfidence) +
                                0.18 * (1.0 - actor.eyeGeometryConfidence) +
                                0.16 * (1.0 - actor.handGeometryConfidence) +
                                0.22 * (1.0 - actor.surfaceGeometryConfidence) +
                                0.14 * (1.0 - actor.occlusionRecoveryConfidence) +
                                0.10 * (1.0 - actor.identityStability)
                            )
                            .coerceIn(0.0, 1.0)

                    val score =
                        (
                            config.uncertaintyWeight *
                                uncertainty +
                                config.occlusionWeight *
                                    occlusion +
                                config.motionWeight *
                                    actor.motionComplexity +
                                config.interactionWeight *
                                    actor.interactionRisk +
                                config.visibilityWeight *
                                    (
                                        1.0 -
                                            actor.visibilityRatio
                                    ).coerceIn(0.0, 1.0) +
                                config.identityWeight *
                                    (
                                        1.0 -
                                            actor.identityConfidence
                                    ).coerceIn(0.0, 1.0) +
                                config.cameraSupportWeight *
                                    cameraSupport +
                                config.operatorPriorityWeight *
                                    actor.operatorPriority +
                                0.14 * modalityUncertainty +
                                hysteresisBonus(actor.actorId)
                            )
                            .coerceIn(0.0, 1.0)

                    actor to score
                }
                .sortedWith(
                    compareByDescending<Pair<ActorRequest, Double>> {
                        it.second
                    }.thenBy {
                        it.first.actorId
                    },
                )

        var remainingSurface =
            (
                config.globalSurfaceSampleBudget *
                    capacityScale
                )
                .toInt()
                .coerceAtLeast(0)

        var remainingGaussian =
            (
                config.globalGaussianBudget *
                    capacityScale
                )
                .toInt()
                .coerceAtLeast(0)

        val allocations =
            ArrayList<Allocation>(
                ranked.size,
            )

        for ((actor, priorityScore) in ranked) {
            val cameraSupport =
                (
                    actor.cameraSupportCount.toDouble() /
                        12.0
                    )
                    .coerceIn(0.0, 1.0)

            val tier =
                tierFor(
                    score = priorityScore,
                    capacityScale = capacityScale,
                )

            val tierScale =
                when (tier) {
                    Tier.HOLD -> 0.10
                    Tier.CORE -> 0.32
                    Tier.STANDARD -> 0.56
                    Tier.FINE -> 0.78
                    Tier.ULTRA -> 1.0
                }

            val desiredSurface =
                (
                    config.minimumSurfaceSamplesPerActor +
                        (
                            config.maximumSurfaceSamplesPerActor -
                                config.minimumSurfaceSamplesPerActor
                            ) * tierScale
                    )
                    .toInt()
                    .coerceAtMost(
                        config.maximumSurfaceSamplesPerActor,
                    )

            val desiredGaussian =
                (
                    config.minimumGaussianBudgetPerActor +
                        (
                            config.maximumGaussianBudgetPerActor -
                                config.minimumGaussianBudgetPerActor
                            ) * tierScale
                    )
                    .toInt()
                    .coerceAtMost(
                        config.maximumGaussianBudgetPerActor,
                    )

            val surface =
                desiredSurface
                    .coerceAtMost(
                        remainingSurface,
                    )

            val gaussian =
                desiredGaussian
                    .coerceAtMost(
                        remainingGaussian,
                    )

            val actualTier =
                if (
                    surface <
                        config.minimumSurfaceSamplesPerActor ||
                    gaussian <
                        config.minimumGaussianBudgetPerActor
                ) {
                    Tier.HOLD
                } else {
                    tier
                }

            val actualSurface =
                if (actualTier == Tier.HOLD) {
                    0
                } else {
                    surface
                }

            val actualGaussian =
                if (actualTier == Tier.HOLD) {
                    0
                } else {
                    gaussian
                }

            if (actualTier != Tier.HOLD) {
                remainingSurface -= actualSurface
                remainingGaussian -= actualGaussian
            }

            val bodyHz =
                when (actualTier) {
                    Tier.HOLD -> 15
                    Tier.CORE -> config.coreBodyHz
                    Tier.STANDARD -> config.standardBodyHz
                    Tier.FINE -> config.fineBodyHz
                    Tier.ULTRA -> config.ultraBodyHz
                }

            val detailHz =
                when (actualTier) {
                    Tier.HOLD -> 5
                    Tier.CORE -> 10
                    Tier.STANDARD -> 20
                    Tier.FINE -> 30
                    Tier.ULTRA -> 45
                }
                    .let {
                        (
                            it *
                                (
                                    0.55 +
                                        0.45 *
                                            capacityScale
                                    )
                            )
                            .toInt()
                            .coerceAtLeast(1)
                    }

            val faceBudget =
                when {
                    !actor.requestedFaceDetail ->
                        0
                    actualTier == Tier.ULTRA ->
                        1_536
                    actualTier == Tier.FINE ->
                        1_024
                    actualTier == Tier.STANDARD ->
                        768
                    actualTier == Tier.CORE ->
                        512
                    else ->
                        256
                }

            val eyeBudget =
                when {
                    !actor.requestedEyeDetail -> 0
                    actualTier == Tier.ULTRA -> 32
                    actualTier == Tier.FINE -> 24
                    actualTier == Tier.STANDARD -> 16
                    actualTier == Tier.CORE -> 10
                    else -> 4
                }

            val handBudget =
                when {
                    !actor.requestedHandDetail -> 0
                    actualTier == Tier.ULTRA -> 42
                    actualTier == Tier.FINE -> 32
                    actualTier == Tier.STANDARD -> 24
                    actualTier == Tier.CORE -> 16
                    else -> 8
                }

            val memory =
                actualSurface.toLong() *
                    config.memoryBytesPerSurfaceSample.toLong() +
                    actualGaussian.toLong() *
                        config.memoryBytesPerGaussian.toLong()

            val confidence =
                (
                    0.42 * actor.identityConfidence +
                        0.28 * actor.visibilityRatio +
                        0.18 *
                            cameraSupport +
                        0.12 * (1.0 - actor.occlusionRatio)
                    )
                    .coerceIn(0.0, 1.0)

            val reasons =
                buildList {
                    if (actor.occlusionRatio >= 0.50) {
                        add("occlusion-priority")
                    }
                    if (actor.motionComplexity >= 0.65) {
                        add("motion-complexity-priority")
                    }
                    if (actor.interactionRisk >= 0.65) {
                        add("interaction-priority")
                    }
                    if (actor.identityConfidence <= 0.65) {
                        add("identity-uncertainty")
                    }
                    if (actor.cameraSupportCount < 4) {
                        add("camera-support-limited")
                    }
                    if (actor.surfaceGeometryConfidence < 0.60) {
                        add("surface-geometry-uncertainty")
                    }
                    if (actor.faceGeometryConfidence < 0.60) {
                        add("face-geometry-uncertainty")
                    }
                    if (actor.eyeGeometryConfidence < 0.60) {
                        add("eye-geometry-uncertainty")
                    }
                    if (actor.handGeometryConfidence < 0.60) {
                        add("hand-geometry-uncertainty")
                    }
                    if (actor.occlusionRecoveryConfidence < 0.60) {
                        add("occlusion-recovery-uncertainty")
                    }
                    if (capacityScale < 0.45) {
                        add("resource-pressure")
                    }
                    if (actualTier == Tier.HOLD) {
                        add("detail-held-body-protection")
                    }
                }.distinct()

            allocations +=
                Allocation(
                    actorId = actor.actorId,
                    tier = actualTier,
                    priorityScore = priorityScore,
                    surfaceSamplesPerFrame = actualSurface,
                    gaussianBudget = actualGaussian,
                    faceMeshBudget = faceBudget,
                    eyeSamplesPerFrame = eyeBudget,
                    handSamplesPerFrame = handBudget,
                    bodyUpdateHz = bodyHz,
                    detailUpdateHz = detailHz,
                    memoryBudgetBytes = memory,
                    confidence = confidence,
                    reasons = reasons,
                )

            previousTiers[actor.actorId] = actualTier
        }

        while (previousTiers.size > config.maximumActors) {
            previousTiers.entries
                .firstOrNull()
                ?.let { previousTiers.remove(it.key) }
        }

        val surfaceUsed =
            allocations.sumOf {
                it.surfaceSamplesPerFrame
            }
        val gaussianUsed =
            allocations.sumOf {
                it.gaussianBudget
            }

        val held =
            allocations.count {
                it.tier == Tier.HOLD
            }

        val starvation =
            (
                allocations
                    .map {
                        when {
                            it.tier == Tier.HOLD -> 1.0
                            it.surfaceSamplesPerFrame <
                                config.minimumSurfaceSamplesPerActor -> 0.75
                            else -> 0.0
                        }
                    }
                    .average()
            )
                .coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (held > 0) {
                    add("reconstruction-detail-held-for-resource-protection")
                }
                if (starvation >= 0.30) {
                    add("reconstruction-detail-starvation-risk")
                }
                if (capacityScale < 0.35) {
                    add("reconstruction-resource-pressure-high")
                }
                if (
                    surfaceUsed >=
                        (
                            config.globalSurfaceSampleBudget *
                                capacityScale *
                                0.96
                            )
                            .toInt()
                ) {
                    add("reconstruction-surface-budget-near-limit")
                }
                if (
                    gaussianUsed >=
                        (
                            config.globalGaussianBudget *
                                capacityScale *
                                0.96
                            )
                            .toInt()
                ) {
                    add("reconstruction-gaussian-budget-near-limit")
                }
            }

        return Result(
            allocations = allocations,
            totalSurfaceSamplesPerFrame = surfaceUsed,
            totalGaussianBudget = gaussianUsed,
            totalMemoryBudgetBytes =
                allocations.sumOf { it.memoryBudgetBytes },
            ultraActorCount =
                allocations.count {
                    it.tier == Tier.ULTRA
                },
            fineOrHigherActorCount =
                allocations.count {
                    it.tier == Tier.FINE ||
                        it.tier == Tier.ULTRA
                },
            coreActorCount =
                allocations.count {
                    it.tier == Tier.CORE ||
                        it.tier == Tier.STANDARD
                },
            heldActorCount = held,
            budgetUtilization =
                (
                    0.50 *
                        (
                            surfaceUsed.toDouble() /
                                (
                                    config.globalSurfaceSampleBudget *
                                        capacityScale
                                    )
                                )
                        +
                    0.50 *
                        (
                            gaussianUsed.toDouble() /
                                (
                                    config.globalGaussianBudget *
                                        capacityScale
                                    )
                                )
                    )
                    .coerceIn(0.0, 1.0),
            starvationRisk = starvation,
            warnings = warnings.distinct(),
        )
    }

    fun reset() {
        synchronized(this) {
            previousTiers.clear()
        }
    }

    private fun tierFor(
        score: Double,
        capacityScale: Double,
    ): Tier =
        when {
            capacityScale < 0.18 -> Tier.CORE
            score >= 0.82 && capacityScale >= 0.72 -> Tier.ULTRA
            score >= 0.62 && capacityScale >= 0.48 -> Tier.FINE
            score >= 0.40 && capacityScale >= 0.28 -> Tier.STANDARD
            else -> Tier.CORE
        }

    private fun hysteresisBonus(
        actorId: Long,
    ): Double =
        when (previousTiers[actorId]) {
            Tier.ULTRA -> 0.055
            Tier.FINE -> 0.040
            Tier.STANDARD -> 0.025
            Tier.CORE -> 0.012
            Tier.HOLD -> 0.0
            null -> 0.0
        }

}
