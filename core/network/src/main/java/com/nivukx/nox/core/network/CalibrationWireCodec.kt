package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * Compact calibration bootstrap packet.
 *
 * It carries calibrated camera intrinsics plus the device's current world-frame orientation.
 * Translation is deliberately treated as a seed only. The actor-based self-calibration stage is
 * responsible for refining inter-camera translation.
 */
object CalibrationWireCodec {
    private const val VERSION = 1
    private const val MAX_ID_BYTES = 96

    data class Frame(
        val deviceId: String,
        val role: DeviceRole,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val accuracy: Int,
    )

    fun encode(frame: Frame): ByteArray {
        val encodedId = frame.deviceId.toByteArray(StandardCharsets.UTF_8)
        val idBytes = encodedId.copyOf(
            minOf(encodedId.size, MAX_ID_BYTES),
        )

        val buffer = ByteBuffer.allocate(
            4 + 4 + idBytes.size +
                4 +
                (9 * 8) +
                2 * 4 +
                3 * 8 +
                4 * 8 +
                4 +
                8,
        ).order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(VERSION)
        buffer.putInt(idBytes.size)
        buffer.put(idBytes)
        buffer.putInt(frame.role.ordinal)

        val k = frame.intrinsics
        buffer.putDouble(k.fx)
        buffer.putDouble(k.fy)
        buffer.putDouble(k.cx)
        buffer.putDouble(k.cy)
        buffer.putDouble(k.k1)
        buffer.putDouble(k.k2)
        buffer.putDouble(k.p1)
        buffer.putDouble(k.p2)
        buffer.putDouble(k.k3)
        buffer.putInt(k.imageWidth)
        buffer.putInt(k.imageHeight)

        val p = frame.pose.position
        buffer.putDouble(p.x)
        buffer.putDouble(p.y)
        buffer.putDouble(p.z)

        val q = frame.pose.rotation
        buffer.putDouble(q.w)
        buffer.putDouble(q.x)
        buffer.putDouble(q.y)
        buffer.putDouble(q.z)

        buffer.putInt(frame.accuracy)
        buffer.putLong(System.nanoTime())
        return buffer.array()
    }

    fun decode(bytes: ByteArray): Frame {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        require(buffer.int == VERSION) { "Unsupported calibration payload version" }

        val idLength = buffer.int
        require(idLength in 1..MAX_ID_BYTES) { "Invalid calibration device id" }
        val idBytes = ByteArray(idLength)
        buffer.get(idBytes)
        val id = String(idBytes, StandardCharsets.UTF_8)

        val role = DeviceRole.entries.getOrNull(buffer.int) ?: DeviceRole.AUXILIARY

        val intrinsics = CameraIntrinsics(
            fx = buffer.double,
            fy = buffer.double,
            cx = buffer.double,
            cy = buffer.double,
            k1 = buffer.double,
            k2 = buffer.double,
            p1 = buffer.double,
            p2 = buffer.double,
            k3 = buffer.double,
            imageWidth = buffer.int,
            imageHeight = buffer.int,
        )

        val position = Vec3(
            buffer.double,
            buffer.double,
            buffer.double,
        )
        val rotation = Quat(
            buffer.double,
            buffer.double,
            buffer.double,
            buffer.double,
        ).normalized()

        val accuracy = buffer.int
        buffer.long

        require(!buffer.hasRemaining()) { "Trailing calibration payload bytes" }

        return Frame(
            deviceId = id,
            role = role,
            intrinsics = intrinsics,
            pose = Pose(position, rotation),
            accuracy = accuracy,
        )
    }
}
