package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HighEndPrecisionCockpitSourceIntegrityTest {
    private fun source(): String {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/ui/studio/HighEndPrecisionCockpit.kt"),
            Path.of("app/src/main/java/com/nivukx/nox/ui/studio/HighEndPrecisionCockpit.kt"),
            Path.of("../app/src/main/java/com/nivukx/nox/ui/studio/HighEndPrecisionCockpit.kt"),
        )
        return candidates.firstOrNull(Files::isRegularFile)
            ?.let(Files::readString)
            ?: error("HighEndPrecisionCockpit.kt was not found")
    }

    @Test
    fun compositePrecisionChannelsAreNotLabeledAsPhysicalPrecision() {
        val source = source()
        assertTrue(source.contains("label = \"QUALITY INDEX\""))
        assertTrue(source.contains("captureAuthorityPrecisionQualified == true"))
        assertFalse(source.contains("label = \"PRECISION\""))
    }

    @Test
    fun defaultCameraCeilingIsNotPresentedAsExpectedHardwareCount() {
        val source = source()
        assertTrue(source.contains("\"UNKNOWN / 128 cap\""))
        assertTrue(source.contains("\" / 128 cap\""))
        assertTrue(source.contains("\" expected\""))
        assertFalse(source.contains("telemetry.camerasExpected ?: 128"))
    }
}
