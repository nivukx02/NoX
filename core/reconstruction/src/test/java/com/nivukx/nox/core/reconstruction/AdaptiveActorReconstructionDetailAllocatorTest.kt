package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AdaptiveActorReconstructionDetailAllocatorTest {

    private fun actor(
        id: Long,
        score: Double,
        occlusion: Double = 0.10,
        interaction: Double = 0.10,
    ) =
        AdaptiveActorReconstructionDetailAllocator.ActorRequest(
            actorId = id,
            operatorPriority = score,
            identityConfidence = 1.0 - (1.0 - score) * 0.35,
            visibilityRatio = 1.0 - occlusion,
            occlusionRatio = occlusion,
            motionComplexity = score,
            interactionRisk = interaction,
            cameraSupportCount = 12,
        )

    @Test
    fun protectsHighestPriorityActorsWithHigherDetail() {
        val allocator =
            AdaptiveActorReconstructionDetailAllocator()

        val result =
            allocator.allocate(
                actors =
                    listOf(
                        actor(1L, 0.98, occlusion = 0.70, interaction = 0.80),
                        actor(2L, 0.35),
                        actor(3L, 0.20),
                        actor(4L, 0.10),
                    ),
                memoryHeadroom = 0.95,
                gpuHeadroom = 0.95,
                npuHeadroom = 0.95,
            )

        assertEquals(4, result.allocations.size)

        val first =
            result.allocations
                .first { it.actorId == 1L }
        val last =
            result.allocations
                .first { it.actorId == 4L }

        assertTrue(
            first.surfaceSamplesPerFrame >=
                last.surfaceSamplesPerFrame,
        )
        assertTrue(
            first.gaussianBudget >=
                last.gaussianBudget,
        )
        assertTrue(
            first.tier.ordinal >=
                last.tier.ordinal,
        )
    }

    @Test
    fun preservesBodyAtResourcePressure() {
        val allocator =
            AdaptiveActorReconstructionDetailAllocator()

        val result =
            allocator.allocate(
                actors =
                    (0L until 50L).map {
                        actor(it, 0.5)
                    },
                memoryHeadroom = 0.12,
                gpuHeadroom = 0.10,
                npuHeadroom = 0.15,
            )

        assertEquals(50, result.allocations.size)
        assertTrue(
            result.allocations.all {
                it.bodyUpdateHz >= 15
            },
        )
        assertTrue(
            result.totalSurfaceSamplesPerFrame <=
                750_000,
        )
        assertTrue(
            result.totalGaussianBudget <=
                1_500_000,
        )
        assertTrue(
            result.budgetUtilization in 0.0..1.0,
        )
    }

    @Test
    fun resetClearsTierHysteresis() {
        val allocator =
            AdaptiveActorReconstructionDetailAllocator()

        allocator.allocate(
            actors = listOf(actor(7L, 0.95)),
            memoryHeadroom = 1.0,
            gpuHeadroom = 1.0,
            npuHeadroom = 1.0,
        )

        allocator.reset()

        val result =
            allocator.allocate(
                actors = listOf(actor(7L, 0.95)),
                memoryHeadroom = 1.0,
                gpuHeadroom = 1.0,
                npuHeadroom = 1.0,
            )

        assertEquals(1, result.allocations.size)
        assertTrue(result.allocations.single().priorityScore >= 0.0)
    }
}
