package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceContour
import com.nivukx.nox.core.model.FaceContourFrame
import com.nivukx.nox.core.model.FaceContourKind
import com.nivukx.nox.core.model.Vec2
import org.junit.Assert.assertEquals
import org.junit.Test

class FaceContourWireCodecTest {
    @Test
    fun roundTripsOrderedContours() {
        val frame = FaceContourFrame(
            timestampNanos = 123L,
            boundsLeft = 10f,
            boundsTop = 20f,
            boundsRight = 210f,
            boundsBottom = 220f,
            yawDegrees = 4f,
            rollDegrees = -2f,
            contours = listOf(
                FaceContour(
                    kind = FaceContourKind.LEFT_EYE,
                    points = listOf(
                        Vec2(30.0, 40.0),
                        Vec2(31.0, 41.0),
                        Vec2(32.0, 42.0),
                    ),
                    confidence = 0.96,
                ),
            ),
        )

        val decoded = FaceContourWireCodec.decode(
            FaceContourWireCodec.encode(frame),
        )

        assertEquals(1, decoded.contours.size)
        assertEquals(FaceContourKind.LEFT_EYE, decoded.contours.first().kind)
        assertEquals(3, decoded.contours.first().points.size)
        assertEquals(31.0, decoded.contours.first().points[1].x, 1e-4)
    }
}


class FaceContourWireCodecFrontierTransportTest {
    @Test
    fun v3RoundTripPreservesCaptureTimingEvidence() {
        val input = FaceContourFrame(
            timestampNanos = 123_456_789L,
            trackingId = 27,
            boundsLeft = 10f,
            boundsTop = 20f,
            boundsRight = 300f,
            boundsBottom = 420f,
            yawDegrees = 12f,
            rollDegrees = -3f,
            contours = listOf(
                FaceContour(
                    kind = FaceContourKind.LEFT_EYE,
                    points = (0 until 32).map {
                        Vec2(
                            40.0 + it * 1.25,
                            80.0 + kotlin.math.sin(it.toDouble() * 0.2) * 3.0,
                        )
                    },
                    confidence = 0.97,
                ),
            ),
            imageWidth = 1920,
            imageHeight = 1080,
            rotationDegrees = 90,
            captureTimingConfidence = 0.93,
            rollingShutterSkewNanos = 7_500_000L,
            frameDurationNanos = 16_666_667L,
            exposureTimeNanos = 2_000_000L,
            rollingShutterDirectionSign = -1,
        )

        val bytes = FaceContourWireCodec.encode(input)
        assertTrue(bytes.size <= 32 * 1024)

        val decoded = FaceContourWireCodec.decode(bytes)

        assertEquals(input.timestampNanos, decoded.timestampNanos)
        assertEquals(input.trackingId, decoded.trackingId)
        assertEquals(input.captureTimingConfidence, decoded.captureTimingConfidence, 1e-6)
        assertEquals(input.rollingShutterSkewNanos, decoded.rollingShutterSkewNanos)
        assertEquals(input.frameDurationNanos, decoded.frameDurationNanos)
        assertEquals(input.exposureTimeNanos, decoded.exposureTimeNanos)
        assertEquals(input.rollingShutterDirectionSign, decoded.rollingShutterDirectionSign)
        assertEquals(
            input.contours.single().points.size,
            decoded.contours.single().points.size,
        )
    }

    @Test
    fun oversizedContourSetIsDeterministicallyResampledUnderTransportBudget() {
        val contours = FaceContourKind.entries.map { kind ->
            FaceContour(
                kind = kind,
                points = (0 until 512).map { index ->
                    Vec2(
                        index.toDouble(),
                        (index * 1.7) + kind.ordinal,
                    )
                },
                confidence = 0.91,
            )
        }
        val input = FaceContourFrame(
            timestampNanos = 10L,
            boundsLeft = 0f,
            boundsTop = 0f,
            boundsRight = 1920f,
            boundsBottom = 1080f,
            yawDegrees = 0f,
            rollDegrees = 0f,
            contours = contours,
            imageWidth = 1920,
            imageHeight = 1080,
        )

        val first = FaceContourWireCodec.encode(input)
        val second = FaceContourWireCodec.encode(input)

        assertEquals(first.toList(), second.toList())
        assertTrue(first.size <= 32 * 1024)

        val decoded = FaceContourWireCodec.decode(first)
        assertEquals(contours.size, decoded.contours.size)
        assertTrue(
            decoded.contours.all {
                it.points.size <= 512
            },
        )
    }
}

class FaceContourWireCodecLegacyV2Test {
    @Test
    fun legacyV2PayloadStillDecodesWithDefaultTimingEvidence() {
        val contour = FaceContour(
            kind = FaceContourKind.LEFT_EYE,
            points = listOf(
                Vec2(10.0, 20.0),
                Vec2(15.0, 20.0),
                Vec2(20.0, 20.0),
            ),
            confidence = 0.88,
        )
        val buffer = java.nio.ByteBuffer
            .allocate(48 + 12 + 24)
            .order(java.nio.ByteOrder.BIG_ENDIAN)

        buffer.putInt(2)
        buffer.putLong(1_000L)
        buffer.putFloat(0f)
        buffer.putFloat(0f)
        buffer.putFloat(100f)
        buffer.putFloat(100f)
        buffer.putFloat(0f)
        buffer.putFloat(0f)
        buffer.putInt(100)
        buffer.putInt(100)
        buffer.putInt(0)
        buffer.putInt(1)
        buffer.putInt(contour.kind.ordinal)
        buffer.putInt(3)
        buffer.putFloat(0.88f)
        contour.points.forEach {
            buffer.putFloat(it.x.toFloat())
            buffer.putFloat(it.y.toFloat())
        }

        val decoded = FaceContourWireCodec.decode(buffer.array())

        assertEquals(1_000L, decoded.timestampNanos)
        assertEquals(1.0, decoded.captureTimingConfidence, 1e-6)
        assertEquals(0L, decoded.rollingShutterSkewNanos)
        assertEquals(0L, decoded.frameDurationNanos)
        assertEquals(0L, decoded.exposureTimeNanos)
    }
}
