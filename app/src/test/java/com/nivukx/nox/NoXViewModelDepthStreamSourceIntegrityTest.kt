package com.nivukx.nox

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NoXViewModelDepthStreamSourceIntegrityTest {
    private fun source(): String =
        listOf(
            Path.of("app/src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            Path.of("src/main/java/com/nivukx/nox/NoXViewModel.kt"),
            Path.of("../app/src/main/java/com/nivukx/nox/NoXViewModel.kt"),
        ).firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("NoXViewModel.kt was not found")

    @Test
    fun beginRecordingOpensCameraOnceAfterFrameCollectorsAreReady() {
        val text = source()
        val begin = text.indexOf("private fun beginRecording()")
        val end = text.indexOf("private fun stopCapture(", begin)
        assertTrue(begin >= 0)
        assertTrue(end > begin)
        val body = text.substring(begin, end)

        assertTrue(body.contains("val enableLiveDepthStream ="))
        assertTrue(body.contains("openCaptureCamera("))
        assertEquals(0, body.split("camera.open(").size - 1)
        assertTrue(body.indexOf("depthJob = viewModelScope.launch") < body.indexOf("openCaptureCamera("))
    }

    @Test
    fun kotlinSparseTsdfReceivesPerPixelDepthConfidence() {
        val text = source()
        assertTrue(text.contains("depthConfidence = depthFrame.confidence"))
    }

    @Test
    fun nativeDepthConfidenceIsBoundedBeforeCallingStrictNativeApi() {
        val text = source()
        assertTrue(text.contains("NativeDepthConfidencePolicy.resolve("))
        assertTrue(!text.contains("depthConfidence *\n                                    adaptiveDepthDecision"))
    }

    @Test
    fun singleOpenPreservesSidecarPolicyAndRequestsCalibratedDepth() {
        val text = source()
        val helperStart = text.indexOf("private fun openCaptureCamera(")
        val helperEnd = text.indexOf("private fun stopCapture(", helperStart)
        assertTrue(helperStart >= 0)
        assertTrue(helperEnd > helperStart)
        val helper = text.substring(helperStart, helperEnd)

        assertTrue(helper.contains("allowPersistentSensorSidecars ="))
        assertTrue(helper.contains("activeCaptureProfile.enableRawSidecar"))
        assertTrue(helper.contains("activeCaptureProfile.enableDepthSidecar"))
        assertTrue(helper.contains("enableLiveDepthStream = enableLiveDepthStream"))
        assertTrue(helper.contains("camera.isDepthReaderActive"))
        assertTrue(helper.contains("live-depth-output-unavailable-after-session-negotiation"))
    }

    @Test
    fun previewOpenDoesNotAccidentallyEnableDepthByDefault() {
        val text = source()
        val previewStart = text.indexOf("fun startPreview()")
        val previewEnd = text.indexOf("fun discover()", previewStart)
        assertTrue(previewStart >= 0)
        assertTrue(previewEnd > previewStart)
        val preview = text.substring(previewStart, previewEnd)
        assertTrue(preview.contains("camera.open("))
        assertTrue(!preview.contains("enableLiveDepthStream = true"))
    }
}
