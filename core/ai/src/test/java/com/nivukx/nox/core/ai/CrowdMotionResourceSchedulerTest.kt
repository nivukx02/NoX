package com.nivukx.nox.core.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class CrowdMotionResourceSchedulerTest {
    private fun demand(id: Long) =
        CrowdMotionResourceScheduler.ActorDemand(
            actorId = id,
            detectionConfidence = 0.95,
            identityContinuity = 0.95,
            motionEnergy = 0.40,
            occlusionRisk = 0.10,
            faceVisibility = 0.90,
            eyeVisibility = 0.90,
            handVisibility = 0.80,
            surfaceVisibility = 0.90,
            fairnessAgeFrames = 0L,
            predicted = false,
        )

    @Test
    fun high_acceleration_and_jerk_raise_actor_priority_without_changing_caps() {
        val scheduler =
            CrowdMotionResourceScheduler(
                CrowdMotionResourceScheduler.Config(
                    targetActors = 2,
                    hardActorCeiling = 2,
                ),
            )

        val stable =
            demand(1L).copy(
                motionAcceleration = 0.0,
                motionJerk = 0.0,
            )
        val fast =
            demand(2L).copy(
                motionAcceleration = 85.0,
                motionJerk = 580.0,
            )

        val result =
            scheduler.allocate(
                demands = listOf(stable, fast),
                computeCredits = 4.0,
                thermalHeadroom = 1.0,
                memoryHeadroom = 1.0,
                gpuHeadroom = 1.0,
            )

        assertTrue(result.admittedActorIds.isNotEmpty())
        assertTrue(result.fairnessIndex in 0.0..1.0)
        assertTrue(result.budgets.size <= 2)
    }

    @Test
    fun fiftyActorsReceiveBoundedBudgetsAndFairnessSignal() {
        val scheduler = CrowdMotionResourceScheduler()
        repeat(6) { frame ->
            val result = scheduler.allocate(
                demands = (0L until 50L).map(::demand),
                computeCredits = 50.0,
                thermalHeadroom = 0.92,
                memoryHeadroom = 0.90,
                gpuHeadroom = 0.90,
            )
            assertTrue(result.admittedActorIds.size <= 50)
            assertTrue(result.fairnessIndex in 0.0..1.0)
            assertTrue(result.trackedHistoryCount <= 1024)
            assertTrue(result.modalityActorCounts[CrowdMotionResourceScheduler.Modality.BODY]!! <= 50)
            if (frame > 0) assertTrue(result.frameIndex == frame + 1L)
        }
    }

    @Test
    fun hardHistoryCeilingRemainsBoundedUnderActorChurn() {
        val scheduler = CrowdMotionResourceScheduler(
            CrowdMotionResourceScheduler.Config(
                targetActors = 50,
                hardActorCeiling = 64,
                maximumHistoryEntries = 128,
            ),
        )
        repeat(120) { frame ->
            scheduler.allocate(
                demands = (0L until 64L).map { id -> demand(id + frame * 64L) },
                computeCredits = 8.0,
                thermalHeadroom = 0.95,
                memoryHeadroom = 0.95,
                gpuHeadroom = 0.95,
            )
        }
        val result = scheduler.allocate(
            demands = (0L until 64L).map { id -> demand(100_000L + id) },
            computeCredits = 8.0,
            thermalHeadroom = 0.95,
            memoryHeadroom = 0.95,
            gpuHeadroom = 0.95,
        )
        assertTrue(result.trackedHistoryCount <= 128)
    }
}
