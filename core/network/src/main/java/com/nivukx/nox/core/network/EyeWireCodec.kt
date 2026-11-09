package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.EyeObservation
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import java.nio.ByteOrder

object EyeWireCodec {
    private const val VERSION = 1
    private const val HEADER_BYTES = 32
    private const val EYE_BYTES = 53

    fun encode(frame: EyeTrackingFrame): ByteArray {
        val buffer = ByteBuffer.allocate(
            HEADER_BYTES + 2 * EYE_BYTES,
        ).order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(VERSION)
        buffer.putLong(frame.timestampNanos)
        buffer.putInt(frame.trackingId ?: -1)
        buffer.putFloat(frame.combinedGaze.x.toFloat())
        buffer.putFloat(frame.combinedGaze.y.toFloat())
        buffer.putFloat(frame.combinedGaze.z.toFloat())
        buffer.putFloat(frame.fixationConfidence.toFloat())

        writeEye(buffer, frame.left)
        writeEye(buffer, frame.right)

        return buffer.array()
    }

    fun decode(bytes: ByteArray): EyeTrackingFrame {
        require(bytes.size == HEADER_BYTES + 2 * EYE_BYTES) {
            "Invalid eye payload length"
        }

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        require(buffer.int == VERSION) { "Unsupported eye payload version" }

        val timestamp = buffer.long
        val trackingId = buffer.int.takeIf { it >= 0 }
        val gaze = Vec3(
            buffer.float.toDouble(),
            buffer.float.toDouble(),
            buffer.float.toDouble(),
        )
        val fixation = buffer.float.toDouble().coerceIn(0.0, 1.0)

        val left = readEye(buffer)
        val right = readEye(buffer)

        return EyeTrackingFrame(
            timestampNanos = timestamp,
            trackingId = trackingId,
            left = left,
            right = right,
            combinedGaze = gaze,
            fixationConfidence = fixation,
        )
    }

    private fun writeEye(buffer: ByteBuffer, eye: EyeObservation?) {
        if (eye == null) {
            buffer.put(0)
            repeat(11) { buffer.putFloat(0f) }
            buffer.putInt(0)
            buffer.putInt(0)
            return
        }

        buffer.put(1)
        buffer.putFloat(eye.center.x.toFloat())
        buffer.putFloat(eye.center.y.toFloat())
        buffer.putFloat(eye.irisCenter?.x?.toFloat() ?: Float.NaN)
        buffer.putFloat(eye.irisCenter?.y?.toFloat() ?: Float.NaN)
        buffer.putFloat(eye.normalizedOffset.x.toFloat())
        buffer.putFloat(eye.normalizedOffset.y.toFloat())
        buffer.putFloat(eye.gazeDirection.x.toFloat())
        buffer.putFloat(eye.gazeDirection.y.toFloat())
        buffer.putFloat(eye.gazeDirection.z.toFloat())
        buffer.putFloat(eye.openness.toFloat())
        buffer.putFloat(eye.confidence.toFloat())
        buffer.putInt(if (eye.blink) 1 else 0)
        buffer.putInt(eye.eye.ordinal)
    }

    private fun readEye(buffer: ByteBuffer): EyeObservation? {
        val present = buffer.get().toInt() != 0
        val center = Vec2(buffer.float.toDouble(), buffer.float.toDouble())
        val irisX = buffer.float.toDouble()
        val irisY = buffer.float.toDouble()
        val offset = Vec2(buffer.float.toDouble(), buffer.float.toDouble())
        val gaze = Vec3(
            buffer.float.toDouble(),
            buffer.float.toDouble(),
            buffer.float.toDouble(),
        ).normalized()
        val openness = buffer.float.toDouble()
        val confidence = buffer.float.toDouble()
        val blink = buffer.int != 0
        val sideOrdinal = buffer.int

        if (!present) return null

        val iris = if (irisX.isNaN() || irisY.isNaN()) null else Vec2(irisX, irisY)
        val side = EyeSide.entries.getOrNull(sideOrdinal) ?: EyeSide.LEFT

        return EyeObservation(
            eye = side,
            center = center,
            irisCenter = iris,
            normalizedOffset = offset,
            openness = openness,
            blink = blink,
            confidence = confidence,
            gazeDirection = gaze,
        )
    }
}
