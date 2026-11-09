package com.nivukx.nox.core.capture

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class Camera2CaptureControllerLifecycleSourceIntegrityTest {
    private fun source(): String {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/core/capture/Camera2CaptureController.kt"),
            Path.of("core/capture/src/main/java/com/nivukx/nox/core/capture/Camera2CaptureController.kt"),
            Path.of("../core/capture/src/main/java/com/nivukx/nox/core/capture/Camera2CaptureController.kt"),
        )
        return candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("Camera2CaptureController.kt was not found")
    }

    @Test
    fun everyOpenAttemptOwnsAnAtomicGeneration() {
        val text = source()
        assertTrue(text.contains("private val openGeneration = AtomicLong(0L)"))
        assertTrue(text.contains("val requestGeneration = openGeneration.incrementAndGet()"))
        assertTrue(text.contains("openGeneration.incrementAndGet()"))
        assertTrue(text.contains("openGeneration.get() == requestGeneration"))
        assertTrue(text.contains("closeCameraOnlyInternal()"))
        assertTrue(text.contains("if (!isCurrentOpen(requestGeneration)) {\n                            image.close()"))
        assertTrue(text.contains("if (!isCurrentOpen(requestGeneration)) return@runCatching"))
    }

    @Test
    fun staleDeviceAndSessionCallbacksAreDisposedInsteadOfPublished() {
        val text = source()
        assertTrue(text.contains("if (!isCurrentOpen(requestGeneration)) {"))
        assertTrue(text.contains("camera.close()"))
        assertTrue(text.contains("if (!isCurrentOpen(requestGeneration) || device !== camera)"))
        assertTrue(text.contains("requestGeneration = requestGeneration,"))
    }

    @Test
    fun liveDepthConsumerCanEnableReaderWithoutSidecarProfile() {
        val text = source()
        assertTrue(text.contains("enableLiveDepthStream: Boolean = false"))
        assertTrue(text.contains("enableLiveDepthStream = enableLiveDepthStream"))
        assertTrue(text.contains("(profile.enableDepthSidecar || enableLiveDepthStream)"))
        assertTrue(text.contains("if (target != null)"))
    }

    @Test
    fun inactiveDepthReaderStateIsResetWhenOptionalOutputsAreClosed() {
        val text = source()
        assertTrue(text.contains("@Volatile private var activeDepthReader = false"))
        assertTrue(text.contains("val isDepthReaderActive: Boolean get() = activeDepthReader"))
        assertTrue(text.contains("activeDepthReader = depthReader != null"))
        assertTrue(text.contains("depthReader = null\n        activeDepthReader = false"))
    }

    @Test
    fun depthReaderCanStreamWithoutPersistentSidecarFile() {
        val text = source()
        val prepareStart = text.indexOf("private fun prepareOptionalReaders(")
        val prepareEnd = text.indexOf("private fun buildCaptureRequest(", prepareStart)
        assertTrue(prepareStart >= 0)
        assertTrue(prepareEnd > prepareStart)
        val prepare = text.substring(prepareStart, prepareEnd)

        assertTrue(prepare.contains("if (capability == null) return"))
        assertFalse(prepare.contains("if (sidecarBase == null || capability == null) return"))
        assertTrue(prepare.contains("profile.enableDepthSidecar &&"))
        assertTrue(prepare.contains("val target =\n                    sidecarBase?.resolveSibling("))
        assertTrue(prepare.contains("if (target != null)"))
        assertTrue(prepare.contains("depthReader = ImageReader.newInstance("))
        assertTrue(prepare.contains("depthWriter?.append(image)"))
    }

    @Test
    fun analysisRawAndDepthListenersRejectStaleGenerationFrames() {
        val text = source()
        assertTrue(text.contains("private fun createAnalysisReader("))
        assertTrue(text.contains("private fun prepareOptionalReaders("))
        assertTrue(text.contains("if (!isCurrentOpen(requestGeneration)) {"))
    }

    @Test
    fun successfulHighSpeedSessionSignalsReadyOnlyAfterCurrentGenerationIsConfirmed() {
        val text = source()
        assertTrue(text.contains("var highSpeedStarted = false"))
        assertTrue(text.contains("if (isCurrentOpen(requestGeneration) && device === camera)"))
        assertTrue(text.contains("highSpeedSessionActive.set(true)\n                            onReady(null)"))
    }

    @Test
    fun publicCameraOnlyCloseInvalidatesPendingOpenRequests() {
        val text = source()
        assertTrue(text.contains("fun closeCameraOnly() {\n        openGeneration.incrementAndGet()"))
        assertTrue(text.contains("private fun closeCameraOnlyInternal()"))
    }

    @Test
    fun stopCaptureInvalidatesPendingOpensAndAvoidsWorkerSelfWait() {
        val text = source()
        assertTrue(text.contains("fun stopCapture() {\n        // Stopping is terminal for the current open request."))
        assertTrue(text.contains("openGeneration.incrementAndGet()\n\n        if (Thread.currentThread() === thread)"))
        assertTrue(text.contains("val posted = handler.post"))
        assertTrue(text.contains("if (!posted)"))
        assertTrue(text.contains("finally {\n                latch.countDown()"))
    }

    @Test
    fun controllerShutdownCancelsBufferedImageAndDepthQueues() {
        val text = source()
        val closeStart = text.lastIndexOf("override fun close()")
        assertTrue(closeStart >= 0)
        val closeBlock = text.substring(closeStart)
        assertTrue(closeBlock.contains("runCatching { frames.cancel() }"))
        assertTrue(closeBlock.contains("runCatching { depthFrames.cancel() }"))
        assertTrue(closeBlock.contains("onUndeliveredElement = { (_, image) ->"))
        assertTrue(closeBlock.contains("runCatching { image.close() }"))
        assertFalse(closeBlock.contains("runCatching { frames.close() }"))
    }

    @Test
    fun cameraOnlyCloseHandlesAStoppedHandlerWithoutSkippingCleanup() {
        val text = source()
        val closeStart = text.indexOf("private fun closeCameraOnlyInternal()")
        val closeEnd = text.indexOf("private fun closeCameraResources()", closeStart)
        assertTrue(closeStart >= 0)
        assertTrue(closeEnd > closeStart)
        val closeBlock = text.substring(closeStart, closeEnd)
        assertTrue(closeBlock.contains("val posted ="))
        assertTrue(closeBlock.contains("if (!posted)"))
        assertTrue(closeBlock.contains("closeCameraResources()\n                return"))
        assertTrue(closeBlock.contains("finally {\n                        latch.countDown()"))
    }

    @Test
    fun currentDeviceFailureClosesAllCaptureResources() {
        val text = source()
        val disconnected = text.indexOf("override fun onDisconnected(camera: CameraDevice)")
        val onError = text.indexOf("override fun onError(camera: CameraDevice, error: Int)")
        assertTrue(disconnected >= 0)
        assertTrue(onError > disconnected)
        assertTrue(text.indexOf("closeCameraResources()", disconnected) < onError)
        assertTrue(text.indexOf("closeCameraResources()", onError) > onError)
    }



}
