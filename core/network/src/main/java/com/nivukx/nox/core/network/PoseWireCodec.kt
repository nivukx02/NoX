package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.LandmarkSpace
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Compact pose metadata transport. This is intentionally much smaller than video and is safe for
 * the synchronized control/data channel: one 33-landmark frame is only a few hundred bytes.
 */
object PoseWireCodec {
    private const val VERSION = 1
    private const val HEADER_BYTES = 20

    data class Frame(
        val timestampNanos: Long,
        val landmarks: List<LandmarkObservation>,
    )

    fun encode(frame: Frame): ByteArray {
        val count = frame.landmarks.size.coerceAtMost(128)
        val buffer = ByteBuffer.allocate(HEADER_BYTES + count * 20)
            .order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(VERSION)
        buffer.putLong(frame.timestampNanos)
        buffer.putInt(count)

        frame.landmarks.take(count).forEach {
            buffer.putInt(it.landmarkId)
            buffer.putFloat(it.x.toFloat())
            buffer.putFloat(it.y.toFloat())
            buffer.putFloat(it.confidence.coerceIn(0.0, 1.0).toFloat())
        }

        return buffer.array()
    }

    fun decode(bytes: ByteArray): Frame {
        require(bytes.size >= HEADER_BYTES) { "Pose frame too small" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        require(buffer.int == VERSION) { "Unsupported pose payload version" }

        val timestamp = buffer.long
        val count = buffer.int
        require(count in 0..128) { "Invalid pose landmark count" }
        require(buffer.remaining() == count * 20) { "Invalid pose payload length" }

        val landmarks = ArrayList<LandmarkObservation>(count)
        repeat(count) {
            val id = buffer.int
            val x = buffer.float.toDouble()
            val y = buffer.float.toDouble()
            val confidence = buffer.float.toDouble().coerceIn(0.0, 1.0)
            landmarks += LandmarkObservation(
                landmarkId = id,
                x = x,
                y = y,
                confidence = confidence,
                timestampNanos = timestamp,
                space = LandmarkSpace.PIXELS,
            )
        }

        return Frame(timestamp, landmarks)
    }
}
