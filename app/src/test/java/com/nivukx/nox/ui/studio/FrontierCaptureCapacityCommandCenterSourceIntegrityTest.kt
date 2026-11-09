package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FrontierCaptureCapacityCommandCenterSourceIntegrityTest {
    private fun source(): String {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/ui/studio/FrontierCaptureCapacityCommandCenter.kt"),
            Path.of("app/src/main/java/com/nivukx/nox/ui/studio/FrontierCaptureCapacityCommandCenter.kt"),
            Path.of("../app/src/main/java/com/nivukx/nox/ui/studio/FrontierCaptureCapacityCommandCenter.kt"),
        )
        return candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("FrontierCaptureCapacityCommandCenter.kt was not found")
    }

    @Test
    fun modeledFeasibilityIsNeverUsedAsMeasuredPrecisionFallback() {
        val source = source()
        assertTrue(source.contains("label = \"PRECISION AUTHORITY\""))
        assertTrue(source.contains("label = \"MOTION P95 ERROR\""))
        assertTrue(source.contains("label = \"MODELED FEASIBILITY\""))
        assertTrue(source.contains("telemetry.captureAuthorityPrecisionQualified == true"))
        assertFalse(source.contains("?: telemetry.precisionFeasibilityScore"))
        assertFalse(source.contains("\"HIGH-CONFIDENCE\""))
        assertTrue(source.contains("Modeled feasibility is not measured physical accuracy"))
    }

    @Test
    fun absentExpectedCameraCountIsPresentedAsCapacityNotExpectation() {
        val source = source()
        assertTrue(source.contains("connected.toString() + \" / 128 capacity\""))
        assertTrue(source.contains("connected.toString() + \" / \" + expected + \" expected\""))
        assertFalse(source.contains("(telemetry.camerasExpected ?: 128)"))
    }
}
