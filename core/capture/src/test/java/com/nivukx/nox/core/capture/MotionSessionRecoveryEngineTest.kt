package com.nivukx.nox.core.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class MotionSessionRecoveryEngineTest {

    @Test
    fun truncatesIncompleteNdjsonTail() {
        val dir = Files.createTempDirectory("nox-recovery").toFile()
        try {
            val file = File(dir, "poses.ndjson")
            file.writeText("""{"timestampNanos":1}
{"timestampNanos":2}
{"timestampNanos":3""")

            val result = MotionSessionRecoveryEngine().recover(dir)

            assertTrue(result.repairedFiles.any {
                it.fileName == "poses.ndjson" && it.truncated
            })
            assertEquals(
                """{"timestampNanos":1}
{"timestampNanos":2}
""",
                file.readText(),
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun completeTailIsNotModified() {
        val dir = Files.createTempDirectory("nox-recovery").toFile()
        try {
            val file = File(dir, "eyes.ndjson")
            file.writeText("""{"timestampNanos":1}
""")

            val result = MotionSessionRecoveryEngine().recover(
                dir,
                MotionSessionRecoveryEngine.Policy(
                    requireMotionOnlyManifest = false,
                ),
            )

            assertTrue(result.repairedFiles.none { it.truncated })
            assertEquals(
                """{"timestampNanos":1}
""",
                file.readText(),
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun completeJsonTailWithoutNewlineIsPreserved() {
        val dir =
            Files.createTempDirectory(
                "nox-recovery",
            ).toFile()
        try {
            val file =
                File(
                    dir,
                    "poses.ndjson",
                )
            file.writeText(
                """{"timestampNanos":1}
{"timestampNanos":2}""",
            )

            val result =
                MotionSessionRecoveryEngine().recover(
                    dir,
                    MotionSessionRecoveryEngine.Policy(
                        requireMotionOnlyManifest = false,
                    ),
                )

            assertTrue(
                result.repairedFiles.none {
                    it.truncated
                },
            )
            assertEquals(
                """{"timestampNanos":1}
{"timestampNanos":2}""",
                file.readText(),
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun mediaArtifactIsReportedAndNeverDeleted() {
        val dir = Files.createTempDirectory("nox-recovery").toFile()
        try {
            val video = File(dir, "capture.mp4")
            video.writeBytes(byteArrayOf(1, 2, 3))

            val result = MotionSessionRecoveryEngine().recover(
                dir,
                MotionSessionRecoveryEngine.Policy(
                    requireMotionOnlyManifest = false,
                ),
            )

            assertTrue(
                result.forbiddenArtifacts.contains("capture.mp4"),
            )
            assertTrue(video.exists())
        } finally {
            dir.deleteRecursively()
        }
    }
}
