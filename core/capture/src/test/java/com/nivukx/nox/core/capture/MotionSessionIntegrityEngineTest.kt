package com.nivukx.nox.core.capture

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.zip.CRC32C

class MotionSessionIntegrityEngineTest {

    @Test
    fun validMotionOnlySessionPasses() {
        val verifier = MotionSessionIntegrityEngine()
        val payload = "motion".encodeToByteArray()
        val crc = CRC32C().apply { update(payload) }.value

        val report = verifier.audit(
            manifest = MotionSessionIntegrityEngine.SessionManifest(
                schema = 10,
                sessionId = "nox-test",
                motionOnly = true,
                mediaFramesPersisted = false,
                videoEncoder = "none",
                rawSidecar = false,
                depthSidecar = false,
            ),
            frames = listOf(
                MotionSessionIntegrityEngine.FrameRecord(
                    timestampNanos = 1L,
                    chunkOrdinal = 0,
                    payloadBytes = payload,
                    checksum = crc,
                ),
            ),
            artifacts = listOf("manifest.json", "motion.nmx2"),
        )

        assertTrue(report.accepted)
        assertTrue(report.verifiedFrames == 1)
    }

    @Test
    fun videoArtifactAndChecksumMismatchFail() {
        val verifier = MotionSessionIntegrityEngine()
        val report = verifier.audit(
            manifest = MotionSessionIntegrityEngine.SessionManifest(
                schema = 10,
                sessionId = "nox-test",
                motionOnly = true,
                mediaFramesPersisted = false,
                videoEncoder = "none",
                rawSidecar = false,
                depthSidecar = false,
            ),
            frames = listOf(
                MotionSessionIntegrityEngine.FrameRecord(
                    timestampNanos = 2L,
                    chunkOrdinal = 0,
                    payloadBytes = byteArrayOf(1, 2, 3),
                    checksum = 123L,
                ),
            ),
            artifacts = listOf("motion.mp4"),
        )

        assertFalse(report.accepted)
        assertTrue(report.fatalCount > 0)
    }
}
