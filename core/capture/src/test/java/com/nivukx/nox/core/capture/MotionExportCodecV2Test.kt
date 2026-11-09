package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.MotionSampleV2
import com.nivukx.nox.core.model.MotionSessionHeaderV2
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.SessionId
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import com.nivukx.nox.core.model.Vec3
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionExportCodecV2Test {

    @Test
    fun roundTripPreservesMotionContract() {
        val header = MotionSessionHeaderV2(
            sessionId = SessionId("codec-test"),
            mode = MotionCaptureMode.MONOCULAR_RGB,
            subjectType = SubjectType.HUMAN,
            topologyKind = TopologyKind.HUMAN,
            createdAtNanos = 100L,
            targetFps = 60,
            calibrationVersion = "cal-v1",
            modelProfile = "quality-first",
        )

        val sample = MotionSampleV2(
            timestampNanos = 200L,
            trackId = 42L,
            nodeId = 11,
            position = Vec3(0.12, 0.30, 2.20),
            rotation = Quat.IDENTITY,
            linearVelocity = Vec3(0.2, 0.1, 0.0),
            angularVelocity = Vec3(0.0, 0.2, 0.4),
            confidence = 0.94f,
            uncertaintyMeters = 0.08f,
            predicted = false,
        )

        val bytes = ByteArrayOutputStream()
        val summary = MotionExportCodecV2.write(
            output = bytes,
            header = header,
            frames = sequenceOf(
                MotionExportCodecV2.Frame(
                    timestampNanos = 200L,
                    samples = listOf(sample),
                ),
            ),
        )

        val decoded = MotionExportCodecV2.read(
            ByteArrayInputStream(bytes.toByteArray()),
        )

        assertEquals(1, summary.frameCount)
        assertEquals(1L, summary.jointSamples)
        assertEquals("codec-test", decoded.header.sessionId.value)
        assertEquals(1, decoded.frames.size)
        assertEquals(1, decoded.frames[0].samples.size)
        assertEquals(
            sample.nodeId,
            decoded.frames[0].samples[0].nodeId,
        )
        assertTrue(
            kotlin.math.abs(
                decoded.frames[0].samples[0].confidence -
                    sample.confidence,
            ) < 0.001f,
        )
    }
}
