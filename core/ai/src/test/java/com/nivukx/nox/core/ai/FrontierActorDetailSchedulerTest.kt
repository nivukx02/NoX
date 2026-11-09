package com.nivukx.nox.core.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierActorDetailSchedulerTest {
    private fun demand(id: Int, age: Long = 0L, uncertainty: Double = 0.5) =
        FrontierActorDetailScheduler.ActorDemand(id, uncertainty, 0.7, 0.6, 0.4, 0.8, age)

    @Test
    fun opticalBlurLimitsOptionalDetailWithoutRemovingActorAdmission() {
        val scheduler =
            FrontierActorDetailScheduler(
                FrontierActorDetailScheduler.Config(
                    targetActors = 2,
                    hardActorCeiling = 2,
                    frameBudgetUnits = 256.0,
                ),
            )
        val clear =
            scheduler.allocate(
                demands = listOf(
                    FrontierActorDetailScheduler.ActorDemand(
                        actorId = 1,
                        uncertainty = 0.2,
                        visibility = 0.9,
                        motionComplexity = 0.8,
                        interactionRisk = 0.2,
                        reconstructionValue = 0.9,
                        ageSinceDetailFrames = 0L,
                        opticalMotionBlurRisk = 0.0,
                    ),
                ),
            )
        val blurred =
            scheduler.allocate(
                demands = listOf(
                    FrontierActorDetailScheduler.ActorDemand(
                        actorId = 1,
                        uncertainty = 0.2,
                        visibility = 0.9,
                        motionComplexity = 0.8,
                        interactionRisk = 0.2,
                        reconstructionValue = 0.9,
                        ageSinceDetailFrames = 0L,
                        opticalMotionBlurRisk = 0.80,
                    ),
                ),
            )

        assertTrue(clear.allocations.single().admitted)
        assertTrue(blurred.allocations.single().admitted)
        assertTrue(
            blurred.allocations.single().budgetUnits <=
                clear.allocations.single().budgetUnits,
        )
        assertTrue(
            blurred.warnings.contains(
                "optional-detail-optical-blur-limited",
            ),
        )
    }

    @Test
    fun admitsFiftyActorsWithoutExceedingHardCeiling() {
        val result = FrontierActorDetailScheduler().allocate((0 until 50).map(::demand))
        assertEquals(50, result.admittedActorIds.size)
        assertTrue(result.bodyCoverage > 0.9)
    }

    @Test
    fun sixtyFiveActorsAreBoundedAtSixtyFour() {
        val result = FrontierActorDetailScheduler().allocate((0 until 65).map(::demand))
        assertEquals(64, result.admittedActorIds.size)
        assertTrue(result.warnings.contains("actor-hard-ceiling-applied"))
    }

    @Test
    fun agedActorReceivesPriority() {
        val result = FrontierActorDetailScheduler().allocate(listOf(demand(1, 20L, 0.3), demand(2, 0L, 0.3)))
        val first = result.allocations.first()
        assertEquals(1, first.actorId)
    }

    @Test
    fun fiftyActorAllocationMaintainsNearUniformBodyAdmission() {
        val result =
            FrontierActorDetailScheduler().allocate(
                (0 until 50).map(::demand),
            )

        assertEquals(50, result.allocations.size)
        assertEquals(50, result.admittedActorIds.size)
        assertTrue(result.bodyCoverage >= 0.99)
        assertTrue(result.fairnessIndex >= 0.85)
        assertTrue(
            result.allocations.all {
                it.bodyEveryFrames == 1
            },
        )
    }

    @Test
    fun adversarialBodyCostsStayInsideHardFrameBudget() {
        val demands =
            (0 until 20).map {
                demand(it).copy(
                    bodyCost = 40.0,
                    faceCost = 5.0,
                    eyeCost = 4.0,
                    handCost = 4.0,
                    surfaceCost = 8.0,
                )
            }
        val result =
            FrontierActorDetailScheduler().allocate(demands)

        assertTrue(result.allocations.size < 20)
        assertTrue(
            result.allocations.sumOf { it.budgetUnits } <=
                256.0 + 1.0e-6,
        )
        assertTrue(result.fairnessIndex >= 0.80)
        assertTrue(
            result.warnings.contains(
                "actor-body-reservation-ceiling-applied",
            ),
        )
    }

    @Test
    fun severePressureRetainsBodyBeforeOptionalDetail() {
        val result = FrontierActorDetailScheduler().allocate((0 until 20).map(::demand), pressure = 0.95)
        assertTrue(result.warnings.contains("severe-pressure-optional-detail-shed"))
        assertTrue(result.allocations.all { it.bodyEveryFrames == 1 })
    }
}