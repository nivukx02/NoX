package com.nivukx.nox.core.capture

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

class MultiCameraCaptureFleetLifecycleSourceIntegrityTest {
    private fun sourceText(): String {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/core/capture/MultiCameraCaptureFleetCoordinator.kt"),
            Path.of("core/capture/src/main/java/com/nivukx/nox/core/capture/MultiCameraCaptureFleetCoordinator.kt"),
            Path.of("../core/capture/src/main/java/com/nivukx/nox/core/capture/MultiCameraCaptureFleetCoordinator.kt"),
        )
        return candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("MultiCameraCaptureFleetCoordinator.kt was not found")
    }

    @Test
    fun asynchronousOpenCompletionValidatesEntryIdentityAndGeneration() {
        val source = sourceText()
        assertTrue(source.contains("entries[cameraId] !== entry"))
        assertTrue(source.contains("entry.generation != requestGeneration"))
        assertTrue(source.contains("entry.state != CameraState.OPENING"))
        assertTrue(source.contains("entry.state != CameraState.ADMITTED"))
        assertTrue(source.contains("entry.state != CameraState.FAILED"))
        assertTrue(source.contains("camera-open-already-in-progress"))
        assertTrue(source.contains("callbackStatus?.let(onReady)"))
        assertTrue(source.contains("failureStatus?.let(onReady)"))
    }

    @Test
    fun openReservationChecksOwnershipBeforeCallingCameraController() {
        val source = sourceText()
        val requestReservation = source.indexOf("val requestGeneration =")
        val ownershipCheck = source.indexOf("if (closed.get() || entries[cameraId] !== entry)")
        val dispatch = source.indexOf("entry.controller.open(")

        assertTrue(requestReservation >= 0)
        assertTrue(ownershipCheck > requestReservation)
        assertTrue(dispatch > ownershipCheck)
    }

    @Test
    fun depthSinkIsReservedBeforeOpenAndGatesLiveDepthReaderCreation() {
        val source = sourceText()
        assertTrue(source.contains("val depthSink = resolveDepthSink(entry, requestGeneration)"))
        assertTrue(source.contains("entry.depthFrameSink = depthSink"))
        assertTrue(source.contains("enableLiveDepthStream = depthSink != null"))
        assertTrue(source.contains("private fun resolveDepthSink("))
        assertTrue(source.contains("entry.depthFrameSink ?: return"))
    }

    @Test
    fun negotiatedFallbackDoesNotClaimDepthStreamIsActive() {
        val source = sourceText()
        assertTrue(source.contains("val depthStreamActive: Boolean = false"))
        assertTrue(source.contains("entry.depthFrameSink != null"))
        assertTrue(source.contains("!entry.controller.isDepthReaderActive"))
        assertTrue(source.contains("live-depth-output-unavailable-after-session-negotiation"))
        assertTrue(source.contains("depthStreamActive = entry.controller.isDepthReaderActive"))
        assertTrue(source.contains("!entry.controller.isDepthReaderActive"))
    }

    @Test
    fun depthFramesAreForwardedOnlyForTheCurrentReadyEntry() {
        val source = sourceText()
        assertTrue(source.contains("private val depthFrameSinkFactory:"))
        assertTrue(source.contains("entry.controller.depthFlow.collect { frame ->"))
        assertTrue(source.contains("entries[entry.cameraId] === entry"))
        assertTrue(source.contains("entry.generation == requestGeneration"))
        assertTrue(source.contains("entry.state == CameraState.READY"))
        assertTrue(source.contains("runCatching { sink(frame) }"))
    }

    @Test
    fun depthCollectorsAreCancelledBeforeCameraAndFleetShutdown() {
        val source = sourceText()
        assertTrue(source.contains("private fun cancelDepthForwarding(entry: Entry)"))
        assertTrue(source.contains("cancelDepthForwarding(entry)"))
        assertTrue(source.contains("depthScope.cancel()"))
        assertTrue(source.contains("existing?.cancel()"))
        assertTrue(source.contains("depthForwardingFailedGeneration"))
        assertTrue(source.contains("if (currentFailure) cancelDepthForwarding(entry)"))
        assertTrue(source.contains("depth-frame-sink-failed:"))
    }

    @Test
    fun everyStatusSnapshotReportsActualDepthReaderState() {
        val source = sourceText()
        assertTrue(source.contains("val depthStreamActive: Boolean = false"))
        assertTrue(source.split("depthStreamActive = entry.controller.isDepthReaderActive").size - 1 == 3)
        assertTrue(source.contains("live-depth-output-unavailable-after-session-negotiation"))
    }

    @Test
    fun statusSnapshotsComeFromTheValidatedEntry() {
        val source = sourceText()
        assertTrue(source.contains("private fun statusOf(entry: Entry): CameraStatus"))
        assertTrue(source.contains("var alreadyReadyStatus: CameraStatus? = null"))
        assertTrue(source.contains("alreadyReadyStatus?.let {"))
        assertTrue(source.contains("Never invoke client-owned code under the fleet-wide monitor."))
        assertTrue(source.contains("statusOf(entry)"))
    }
}
