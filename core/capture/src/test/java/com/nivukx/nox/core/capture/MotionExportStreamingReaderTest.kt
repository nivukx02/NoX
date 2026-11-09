package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.MotionSampleV2
import com.nivukx.nox.core.model.MotionSessionHeaderV2
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.SessionId
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files

class MotionExportStreamingReaderTest {

    @Test
    fun readsLongSequenceWithoutDocumentMaterializationContract() {
        val file = Files.createTempFile("nox-nmx2", ".nmx2").toFile()
        try {
            val header = MotionSessionHeaderV2(
                schema = 2,
                sessionId = SessionId("stream-test"),
                mode = MotionCaptureMode.MONOCULAR_RGB,
                subjectType = SubjectType.HUMAN,
                topologyKind = TopologyKind.HUMAN,
                createdAtNanos = 1L,
                targetFps = 60,
                calibrationVersion = "test",
                modelProfile = "production",
                motionOnly = true,
            )

            val frames = (0 until 64).asSequence().map { index ->
                MotionExportCodecV2.Frame(
                    timestampNanos = index * 16_666_666L,
                    samples = listOf(
                        MotionSampleV2(
                            timestampNanos = index * 16_666_666L,
                            trackId = 1L,
                            nodeId = 23,
                            position = Vec3(
                                index.toDouble() * 0.001,
                                1.0,
                                2.0,
                            ),
                            rotation = Quat.IDENTITY,
                            linearVelocity = Vec3(
                                0.06,
                                0.0,
                                0.0,
                            ),
                            angularVelocity = Vec3(
                                0.0,
                                0.0,
                                0.0,
                            ),
                            confidence = 0.95f,
                            uncertaintyMeters = 0.02f,
                            predicted = false,
                        ),
                    ),
                )
            }

            MotionExportCodecV2.writeFile(
                file = file,
                header = header,
                frames = frames,
            )

            MotionExportCodecV2.openStreamingReader(file).use { reader ->
                assertEquals("stream-test", reader.header.sessionId.value)

                var count = 0
                var lastTimestamp = Long.MIN_VALUE

                while (true) {
                    val frame = reader.next() ?: break
                    count += 1
                    assertTrue(frame.timestampNanos >= lastTimestamp)
                    lastTimestamp = frame.timestampNanos
                }

                assertEquals(64, count)
                assertTrue(reader.discoveredIndex().isNotEmpty())
            }
        } finally {
            file.delete()
        }
    }
}
