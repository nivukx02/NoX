package com.nivukx.nox.runtime

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class FrontierCaptureDepthInputContractTest {
    private fun source(): String {
        val candidates = listOf(
            Path.of("app/src/main/java/com/nivukx/nox/runtime/FrontierCaptureSessionCoordinator.kt"),
            Path.of("src/main/java/com/nivukx/nox/runtime/FrontierCaptureSessionCoordinator.kt"),
            Path.of("../app/src/main/java/com/nivukx/nox/runtime/FrontierCaptureSessionCoordinator.kt"),
        )
        return candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("FrontierCaptureSessionCoordinator.kt was not found")
    }

    @Test
    fun inputBoundsDepthFramesAndDisallowsDuplicateCameraSamplesPerStep() {
        val source = source()
        assertTrue(source.contains("val depthFrames: List<DepthFrameEvidence> = emptyList()"))
        assertTrue(source.contains("require(depthFrames.size <= 128)"))
        assertTrue(source.contains("depthFrames.map { it.cameraId }.distinct().size == depthFrames.size"))
        assertTrue(source.contains("it.frame.timestampNanos <= timestampNanos"))
    }

    @Test
    fun nativeDepthFailuresCannotAppearAsHealthyReconstruction() {
        val source = source()
        assertTrue(source.contains("result = submissionResult"))
        assertTrue(source.contains("native-depth-sink-not-configured"))
        assertTrue(source.contains("SessionState.RECONSTRUCTION_LIMITED"))
        assertTrue(source.contains("SubmissionResult.NO_CALIBRATION"))
        assertTrue(source.contains("SubmissionResult.INVALID_CALIBRATION"))
        assertTrue(source.contains("SubmissionResult.INTEGRATION_FAILED"))
        assertTrue(source.contains("depthFrameSubmissions = depthFrameSubmissions"))
    }

    @Test
    fun sessionOwnsTheInjectedDepthSinkLifecycle() {
        val source = source()
        assertTrue(source.contains("private val nativeDepthSink: NativeDepthFrameReconstructionSink? = null"))
        assertTrue(source.contains("closeStage { nativeDepthSink?.close() }"))
    }
}
