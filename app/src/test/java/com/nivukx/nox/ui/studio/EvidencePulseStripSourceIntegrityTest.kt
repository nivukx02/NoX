package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class EvidencePulseStripSourceIntegrityTest {
    @Test
    fun evidenceStripKeepsAuthoritativeSignalsOnly() {
        val sourcePath = Path.of(
            "src/main/java/com/nivukx/nox/ui/studio/EvidencePulseStrip.kt",
        )
        assertTrue(sourcePath.exists())
        val source = Files.readString(sourcePath)
        assertEquals(1, Regex("fun EvidencePulseStrip").findAll(source).count())
        assertTrue(source.contains("StudioTelemetryState"))
        assertTrue(source.contains("EVIDENCE PULSE"))
        assertTrue(source.contains("SYNC"))
        assertTrue(source.contains("RIG"))
        assertTrue(source.contains("BODY"))
        assertTrue(source.contains("3D"))
        assertTrue(source.contains("OUT"))
        assertTrue(source.contains("StudioUiStateReducer.exportEnabled"))
        assertTrue(source.contains("N/A"))
    }
}
