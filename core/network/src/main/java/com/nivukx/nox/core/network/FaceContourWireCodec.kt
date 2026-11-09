package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceContour
import com.nivukx.nox.core.model.FaceContourFrame
import com.nivukx.nox.core.model.FaceContourKind
import com.nivukx.nox.core.model.Vec2
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Binary transport for the ordered facial contour channel.
 *
 * The payload is variable length because each contour has a different number of points. It remains
 * comfortably below the 32 KiB NoX packet ceiling for the complete contour set.
 */
object FaceContourWireCodec {
    private const val VERSION = 3
    private const val HEADER_BYTES_V2 = 48
    private const val HEADER_BYTES_V3 = 88
    private const val MAX_PAYLOAD_BYTES = 32 * 1024
    private const val MAX_CONTOURS = 32
    private const val MAX_POINTS_PER_CONTOUR = 512

    fun encode(frame: FaceContourFrame): ByteArray {
        val contours = frame.contours.take(MAX_CONTOURS)
        val maximumPointsPerContour =
            if (contours.isEmpty()) {
                0
            } else {
                (
                    (MAX_PAYLOAD_BYTES - HEADER_BYTES_V3 - 12 * contours.size) /
                        (8 * contours.size)
                    ).coerceIn(1, MAX_POINTS_PER_CONTOUR)
            }
        val encodedPoints = contours.map { contour ->
            resamplePoints(
                points = contour.points,
                maximumPoints = maximumPointsPerContour,
            )
        }
        val bodyBytes = contours.indices.sumOf { index ->
            12 + encodedPoints[index].size * 8
        }
        require(HEADER_BYTES_V3 + bodyBytes <= MAX_PAYLOAD_BYTES) {
            "Face contour payload exceeds the NoX 32 KiB transport budget"
        }
        val buffer = ByteBuffer.allocate(HEADER_BYTES_V3 + bodyBytes)
            .order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(VERSION)
        buffer.putLong(frame.timestampNanos)
        buffer.putInt(frame.trackingId ?: -1)
        buffer.putFloat(frame.boundsLeft)
        buffer.putFloat(frame.boundsTop)
        buffer.putFloat(frame.boundsRight)
        buffer.putFloat(frame.boundsBottom)
        buffer.putFloat(frame.yawDegrees)
        buffer.putFloat(frame.rollDegrees)
        buffer.putInt(frame.imageWidth)
        buffer.putInt(frame.imageHeight)
        buffer.putInt(frame.rotationDegrees)
        buffer.putFloat(frame.captureTimingConfidence.toFloat().coerceIn(0f, 1f))
        buffer.putLong(frame.rollingShutterSkewNanos)
        buffer.putLong(frame.frameDurationNanos)
        buffer.putLong(frame.exposureTimeNanos)
        buffer.putInt(frame.rollingShutterDirectionSign)
        buffer.putInt(contours.size)

        encodedPoints.forEachIndexed { index, points ->
            val contour = contours[index]
            buffer.putInt(contour.kind.ordinal)
            buffer.putInt(points.size)
            buffer.putFloat(contour.confidence.toFloat().coerceIn(0f, 1f))
            points.forEach {
                buffer.putFloat(it.x.toFloat())
                buffer.putFloat(it.y.toFloat())
            }
        }

        return buffer.array()
    }

    fun decode(bytes: ByteArray): FaceContourFrame {
        require(bytes.size >= HEADER_BYTES_V2) { "Face contour frame too small" }
        require(bytes.size <= MAX_PAYLOAD_BYTES) { "Face contour payload exceeds transport budget" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        val version = buffer.int
        require(version == 2 || version == 3) { "Unsupported face contour payload version" }

        val timestamp = buffer.long
        val trackingId =
            if (version >= 3) {
                buffer.int.takeIf { it >= 0 }
            } else {
                null
            }
        val left = buffer.float
        val top = buffer.float
        val right = buffer.float
        val bottom = buffer.float
        val yaw = buffer.float
        val roll = buffer.float
        val imageWidth = buffer.int
        val imageHeight = buffer.int
        val rotationDegrees = buffer.int
        val captureTimingConfidence =
            if (version >= 3) {
                buffer.float.toDouble().coerceIn(0.0, 1.0)
            } else {
                1.0
            }
        val rollingShutterSkewNanos =
            if (version >= 3) buffer.long else 0L
        val frameDurationNanos =
            if (version >= 3) buffer.long else 0L
        val exposureTimeNanos =
            if (version >= 3) buffer.long else 0L
        val rollingShutterDirectionSign =
            if (version >= 3) buffer.int else 1
        require(
            rollingShutterDirectionSign == 1 ||
                rollingShutterDirectionSign == -1,
        )
        val contourCount = buffer.int
        require(contourCount in 0..MAX_CONTOURS) { "Invalid contour count" }

        val contours = ArrayList<FaceContour>(contourCount)
        repeat(contourCount) {
            val kind = FaceContourKind.entries.getOrNull(buffer.int)
                ?: error("Unknown face contour kind")
            val count = buffer.int
            require(count in 0..MAX_POINTS_PER_CONTOUR) { "Invalid contour point count" }
            val confidence = buffer.float.toDouble().coerceIn(0.0, 1.0)
            val points = ArrayList<Vec2>(count)
            repeat(count) {
                points += Vec2(
                    buffer.float.toDouble(),
                    buffer.float.toDouble(),
                )
            }
            contours += FaceContour(
                kind = kind,
                points = points,
                confidence = confidence,
            )
        }

        require(!buffer.hasRemaining()) { "Trailing face contour payload bytes" }

        return FaceContourFrame(
            timestampNanos = timestamp,
            trackingId = trackingId,
            boundsLeft = left,
            boundsTop = top,
            boundsRight = right,
            boundsBottom = bottom,
            yawDegrees = yaw,
            rollDegrees = roll,
            contours = contours,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = rotationDegrees,
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
            rollingShutterDirectionSign = rollingShutterDirectionSign,
        )
    }

    private fun resamplePoints(
        points: List<Vec2>,
        maximumPoints: Int,
    ): List<Vec2> {
        if (points.size <= maximumPoints) return points
        if (maximumPoints <= 1) return listOf(points.first())
        val output = ArrayList<Vec2>(maximumPoints)
        val denominator = (maximumPoints - 1).toDouble()
        val lastIndex = points.lastIndex.toDouble()
        for (index in 0 until maximumPoints) {
            val sourceIndex =
                kotlin.math.round(
                    index.toDouble() *
                        lastIndex /
                        denominator,
                ).toInt().coerceIn(0, points.lastIndex)
            output += points[sourceIndex]
        }
        return output
    }

    }
}
