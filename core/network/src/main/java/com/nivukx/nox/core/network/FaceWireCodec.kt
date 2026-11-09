package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import com.nivukx.nox.core.model.Vec2
import java.nio.ByteBuffer
import java.nio.ByteOrder

object FaceWireCodec {
    private const val VERSION = 1
    private const val HEADER_BYTES = 44

    fun encode(frame: FaceFrame): ByteArray {
        val count = frame.points.size.coerceAtMost(256)
        val buffer = ByteBuffer.allocate(HEADER_BYTES + count * 16)
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
        buffer.putInt(count)

        frame.points.take(count).forEach {
            buffer.putInt(it.index)
            buffer.putFloat(it.point.x.toFloat())
            buffer.putFloat(it.point.y.toFloat())
            buffer.putFloat(it.confidence.toFloat())
        }
        return buffer.array()
    }

    fun decode(bytes: ByteArray): FaceFrame {
        require(bytes.size >= HEADER_BYTES) { "Face frame too small" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        require(buffer.int == VERSION) { "Unsupported face payload version" }

        val timestamp = buffer.long
        val trackingId = buffer.int.takeIf { it >= 0 }
        val left = buffer.float
        val top = buffer.float
        val right = buffer.float
        val bottom = buffer.float
        val yaw = buffer.float
        val roll = buffer.float
        val count = buffer.int

        require(count in 0..256) { "Invalid face point count" }
        require(buffer.remaining() == count * 16) { "Invalid face payload length" }

        val points = ArrayList<FaceMeshPoint>(count)
        repeat(count) {
            val index = buffer.int
            val x = buffer.float.toDouble()
            val y = buffer.float.toDouble()
            val confidence = buffer.float.toDouble()
            points += FaceMeshPoint(index, Vec2(x, y), confidence)
        }

        return FaceFrame(
            timestampNanos = timestamp,
            trackingId = trackingId,
            boundsLeft = left,
            boundsTop = top,
            boundsRight = right,
            boundsBottom = bottom,
            yawDegrees = yaw,
            rollDegrees = roll,
            points = points,
        )
    }
}
