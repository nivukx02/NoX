package com.nivukx.nox

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Test

class MainActivitySourceIntegrityTest {
    private fun source(): String {
        val candidates = listOf(
            File("src/main/java/com/nivukx/nox/MainActivity.kt"),
            File("app/src/main/java/com/nivukx/nox/MainActivity.kt"),
        )
        return candidates.firstOrNull { it.isFile }?.readText()
            ?: error("MainActivity.kt was not found from the test working directory")
    }

    @Test
    fun mainActivityHasNoMalformedListGenericParameterDeclarations() {
        assertFalse(
            "MainActivity contains malformed List<T, parameter syntax",
            Regex("""List<String,\s*$""", RegexOption.MULTILINE).containsMatchIn(source()),
        )
    }

    @Test
    fun controlStudioParameterListDoesNotDropRequiredSeparators() {
        val lines = source().split("\n")
        val start = lines.indexOfFirst { it.contains("private fun ControlStudioScreen(") }
        require(start >= 0) { "ControlStudioScreen declaration not found" }

        val end =
            (start + 1 until lines.size)
                .firstOrNull { position ->
                    lines[position].trim() == ") {"
                }
                ?: -1
        require(end > start) { "ControlStudioScreen parameter-list terminator not found" }

        for (position in (start + 1) until end) {
            val current = lines[position].trim()
            val next = lines.getOrNull(position + 1)?.trim().orEmpty()
            if (
                Regex("""^[A-Za-z_]\w*\s*:""").containsMatchIn(current) &&
                Regex("""^[A-Za-z_]\w*\s*:""").containsMatchIn(next)
            ) {
                assertFalse(
                    "Parameter at line ${position + 1} is missing a trailing comma: $current",
                    !current.endsWith(","),
                )
            }
        }
    }
}
