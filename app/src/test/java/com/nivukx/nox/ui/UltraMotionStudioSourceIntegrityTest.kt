package com.nivukx.nox.ui

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class UltraMotionStudioSourceIntegrityTest {
    @Test
    fun studioSourceContainsExactlyOneHelperDefinitionSet() {
        val sourcePath = Path.of(
            "src/main/java/com/nivukx/nox/ui/UltraMotionStudio.kt",
        )
        assertTrue(
            sourcePath.exists(),
            "Motion Studio source must be present for this regression test",
        )
        val source = Files.readString(sourcePath)
        assertEquals(
            1,
            Regex("private enum class StudioWidthTier").findAll(source).count(),
            "StudioWidthTier must be defined exactly once",
        )
        assertEquals(
            1,
            Regex("private fun CaptureControlRail").findAll(source).count(),
            "CaptureControlRail must be defined exactly once",
        )
        assertEquals(
            1,
            Regex("private fun StudioStage").findAll(source).count(),
            "StudioStage must be defined exactly once",
        )
        assertEquals(
            1,
            Regex("private fun CountdownOverlay").findAll(source).count(),
            "CountdownOverlay must be defined exactly once",
        )
    }
}
