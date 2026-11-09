package com.nivukx.nox.core.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierSessionRecoveryCoordinatorTest {
    @Test
    fun outOfMemoryPlanShedsBuffersBeforeResume() {
        val coordinator = FrontierSessionRecoveryCoordinator()
        val checkpoint = coordinator.checkpoint(
            timestampNanos = 10L,
            actorCount = 50,
            cameraCount = 128,
            authoritativeCameraCount = 32,
            reconstructionFrame = 100L,
            activeStages = setOf("body","face","eyes","surface"),
            stateMaterial = "frontier".toByteArray(),
        )
        val plan = coordinator.planRecovery(
            timestampNanos = 11L,
            faultKind = RuntimeFailureContainmentEngine.FaultKind.OUT_OF_MEMORY,
            severity = RuntimeFailureContainmentEngine.Severity.ERROR,
        )
        assertEquals(checkpoint.checkpointId, plan.checkpoint?.checkpointId)
        assertTrue(plan.steps.contains(FrontierSessionRecoveryCoordinator.RecoveryStep.RELEASE_GPU_LEASES))
        assertTrue(plan.steps.contains(FrontierSessionRecoveryCoordinator.RecoveryStep.SHED_TEMPORAL_STATE))
    }

    @Test
    fun checkpointHistoryIsBounded() {
        val coordinator = FrontierSessionRecoveryCoordinator(maximumCheckpoints = 2)
        repeat(4) {
            coordinator.checkpoint(it.toLong()+1L,1,2,2,it.toLong(),emptySet(),byteArrayOf(it.toByte()))
        }
        assertEquals(2, coordinator.checkpointCount())
    }
    @Test
    fun checkpointDigestMustMatchRestorationMaterial() {
        val coordinator = FrontierSessionRecoveryCoordinator()
        val material = "stable".toByteArray()
        val checkpoint = coordinator.checkpoint(
            timestampNanos = 10L,
            actorCount = 2,
            cameraCount = 4,
            authoritativeCameraCount = 2,
            reconstructionFrame = 20L,
            activeStages = setOf("motion"),
            stateMaterial = material,
        )
        assertTrue(coordinator.checkpointMatches(checkpoint, material))
        assertTrue(!coordinator.checkpointMatches(checkpoint, "tampered".toByteArray()))
    }
    @Test
    fun resetDoesNotRecycleCheckpointIds() {
        val coordinator = FrontierSessionRecoveryCoordinator()
        val first = coordinator.checkpoint(
            timestampNanos = 1L,
            actorCount = 1,
            cameraCount = 2,
            authoritativeCameraCount = 2,
            reconstructionFrame = 1L,
            activeStages = setOf("motion"),
            stateMaterial = byteArrayOf(1, 2, 3),
        )
        coordinator.reset()
        val second = coordinator.checkpoint(
            timestampNanos = 2L,
            actorCount = 1,
            cameraCount = 2,
            authoritativeCameraCount = 2,
            reconstructionFrame = 2L,
            activeStages = setOf("motion"),
            stateMaterial = byteArrayOf(4, 5, 6),
        )
        assertTrue(second.checkpointId > first.checkpointId)
    )
}