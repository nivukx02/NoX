package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.assertTrue
import org.junit.Test

class LiquidGlassSurfaceSourceIntegrityTest {
    @Test
    fun glassSurfaceRetainsClearAndStandardMaterials() {
        val sourcePath = Path.of(
            "src/main/java/com/nivukx/nox/ui/studio/LiquidGlassSurface.kt",
        )
        assertTrue(sourcePath.exists())
        val source = Files.readString(sourcePath)
        assertTrue(source.contains("clear"))
        assertTrue(source.contains("surfaceContainer"))
        assertTrue(source.contains("outline"))
        assertTrue(source.contains("topSpecular"))
        assertTrue(source.contains("plateProtection"))
        assertTrue(source.contains("readabilityPlate"))
    }
}
