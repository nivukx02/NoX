package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class StudioCommandBarSourceIntegrityTest {
    @Test
    fun commandBarKeepsGlobalOperatorActions() {
        val sourcePath = Path.of(
            "src/main/java/com/nivukx/nox/ui/studio/StudioCommandBar.kt",
        )
        assertTrue(sourcePath.exists())
        val source = Files.readString(sourcePath)
        assertEquals(1, Regex("fun StudioCommandBar").findAll(source).count())
        assertTrue(source.contains("CAPTURE"))
        assertTrue(source.contains("EXPORT"))
        assertTrue(source.contains("Search commands and settings"))
        assertTrue(source.contains("\"CAMS\""))
        assertTrue(source.contains("\"SYNC\""))
        assertTrue(source.contains("\"RECON\""))
        assertTrue(source.contains("StudioUiStateReducer.exportEnabled"))
    }
}
