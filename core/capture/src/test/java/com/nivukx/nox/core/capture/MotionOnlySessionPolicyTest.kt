package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.MotionCaptureMode
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files

class MotionOnlySessionPolicyTest {

    @Test
    fun rejectsVideoAndRawArtifacts() {
        val dir = Files.createTempDirectory("nox-session").toFile()
        try {
            java.io.File(dir, "manifest.json").writeText(
                """{"motionOnly":true,"mode":"MONOCULAR_RGB"}""",
            )
            java.io.File(dir, "reconstruction.ndjson").writeText("{}")
            java.io.File(dir, "take.mp4").writeBytes(byteArrayOf(1))
            java.io.File(dir, "sensor.noxraw").writeBytes(byteArrayOf(1))

            val result = MotionOnlySessionPolicy().validateSession(
                dir,
                MotionCaptureMode.MONOCULAR_RGB,
            )

            assertFalse(result.valid)
            assertTrue(result.violations.any { it.contains("mp4") })
            assertTrue(result.violations.any { it.contains("noxraw") })
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun acceptsStructuredMotionSession() {
        val dir = Files.createTempDirectory("nox-motion-only").toFile()
        try {
            java.io.File(dir, "manifest.json").writeText(
                """{"motionOnly":true,"mode":"MONOCULAR_RGB"}""",
            )
            java.io.File(dir, "reconstruction.ndjson").writeText("{}")
            java.io.File(dir, "avatar_motion.ndjson").writeText("{}")

            val result = MotionOnlySessionPolicy().validateSession(
                dir,
                MotionCaptureMode.MONOCULAR_RGB,
            )

            assertTrue(result.valid)
            assertTrue(result.motionOnly)
        } finally {
            dir.deleteRecursively()
        }
    }
}
