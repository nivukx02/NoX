package com.nivukx.nox.core.reconstruction

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class FrontierReconstructionCapacitySourceIntegrityTest {
    private fun coordinatorSource(): String {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/core/reconstruction/FrontierReconstructionCoordinator.kt"),
            Path.of("core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/FrontierReconstructionCoordinator.kt"),
            Path.of("../core/reconstruction/src/main/java/com/nivukx/nox/core/reconstruction/FrontierReconstructionCoordinator.kt"),
        )
        return candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("FrontierReconstructionCoordinator.kt was not found")
    }

    @Test
    fun capacityDiagnosticsUseUniqueCameraAndActorIds() {
        val source = coordinatorSource()
        assertTrue(source.contains("val cameraCapacitySelection ="))
        assertTrue(source.contains("items = cameras"))
        assertTrue(source.contains("keyOf = { it.cameraId }"))
        assertTrue(source.contains("val actorCapacitySelection ="))
        assertTrue(source.contains("items = actors"))
        assertTrue(source.contains("keyOf = { it.actorId }"))
        assertTrue(source.contains("cameraInputCount = cameraCapacitySelection.eligibleCountCappedAtOverflow"))
        assertTrue(source.contains("actorInputCount = actorCapacitySelection.eligibleCountCappedAtOverflow"))
    }

    @Test
    fun uniqueIdentityCountingUsesHardCapsAndOverflowWitnesses() {
        val source = coordinatorSource()
        assertTrue(source.contains("maximumItems = ReconstructionFrameCapacityGuard.MAX_CAMERA_INPUTS"))
        assertTrue(source.contains("maximumItems = ReconstructionFrameCapacityGuard.MAX_ACTOR_INPUTS"))
        assertTrue(source.contains("retains at most cap entries and one overflow witness"))
    }
}
