package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertTrue
import org.junit.Test

class LiquidGlassSurfaceContractTest {

    @Test
    fun liquidGlassSurfaceExposesAccessibleOperatorSemantics() {
        val candidates = listOf(
            Path.of("src/main/java/com/nivukx/nox/ui/studio/LiquidGlassSurface.kt"),
            Path.of("../app/src/main/java/com/nivukx/nox/ui/studio/LiquidGlassSurface.kt"),
            Path.of("app/src/main/java/com/nivukx/nox/ui/studio/LiquidGlassSurface.kt"),
        )
        val source = candidates
            .firstOrNull { Files.isRegularFile(it) }
            ?.let { Files.readString(it) }
            ?: error("LiquidGlassSurface.kt was not found")

        assertTrue(source.contains("accessibilityLabel"))
        assertTrue(source.contains("accessibilityRole"))
        assertTrue(source.contains("accessibilityStateDescription"))
        assertTrue(source.contains("semantics(mergeDescendants = true)"))
    }
}
