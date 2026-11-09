package com.nivukx.nox.ui

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertTrue
import org.junit.Test

class UltraMotionStudioContractRegressionTest {
    private val sourcePath =
        Path.of("src/main/java/com/nivukx/nox/ui/UltraMotionStudio.kt")

    @Test
    fun studioStageRetainsExtendedTelemetryContract() {
        assertTrue(sourcePath.toFile().exists())
        val source = Files.readString(sourcePath)
        val start = source.indexOf("private fun StudioStage(")
        val end = source.indexOf(") {", start)
        require(start >= 0 && end > start)
        val signature = source.substring(start, end)

        listOf(
            "face3DTemporalConsistencyConfidence: Double",
            "face3DTemporalConsistencyWarnings: List<String>",
            "sessionPresentationSettings: SessionPresentationSettings",
            "recordingCountdown: Int",
        ).forEach { required ->
            assertTrue(
                signature.contains(required),
                "StudioStage contract lost required parameter: $required",
            )
        }
    }

    @Test
    fun studioStageCallSitesProvideSessionAndCountdownState() {
        assertTrue(sourcePath.toFile().exists())
        val source = Files.readString(sourcePath)
        assertTrue(
            source.countOccurrences("sessionPresentationSettings,") >= 2,
            "Both StudioStage call sites must provide session presentation state",
        )
        assertTrue(
            source.countOccurrences("recordingCountdown,") >= 2,
            "Both StudioStage call sites must provide recording countdown state",
        )
    }

    private fun String.countOccurrences(needle: String): Int {
        var index = 0
        var count = 0
        while (true) {
            val next = indexOf(needle, index)
            if (next < 0) return count
            count += 1
            index = next + needle.length
        }
    }
}
