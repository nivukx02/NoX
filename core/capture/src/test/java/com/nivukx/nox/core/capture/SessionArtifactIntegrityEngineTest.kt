package com.nivukx.nox.core.capture

import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files

class SessionArtifactIntegrityEngineTest {

    @Test
    fun writeAndVerifySessionArtifacts() {
        val directory =
            Files.createTempDirectory("nox-integrity")
                .toFile()

        try {
            directory.resolve("frames.ndjson")
                .writeText("""{"timestampNanos":100}""")
            directory.resolve("mesh.bin")
                .writeBytes(ByteArray(128) { it.toByte() })

            val engine =
                SessionArtifactIntegrityEngine()

            engine.writeManifest(directory)

            val report =
                engine.verify(directory)

            assertTrue(report.valid)
            assertTrue(report.artifactsChecked >= 2)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun tamperingIsDetected() {
        val directory =
            Files.createTempDirectory("nox-integrity")
                .toFile()

        try {
            directory.resolve("frames.ndjson")
                .writeText("original")

            val engine =
                SessionArtifactIntegrityEngine()

            engine.writeManifest(directory)

            directory.resolve("frames.ndjson")
                .writeText("tampered")

            val report =
                engine.verify(directory)

            assertTrue(!report.valid)
            assertTrue(
                report.mismatchedArtifacts.contains(
                    "frames.ndjson",
                ),
            )
        } finally {
            directory.deleteRecursively()
        }
    }
}
