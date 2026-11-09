package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

object NoXWireProtocol {
    private const val MAGIC = 0x4E4F5802
    private const val VERSION = 2
    private const val HEADER_BYTES = 36
    /**
     * Nearby Connections Payload.Type.BYTES is limited to 32 KiB on Android.
     * Keep this contract aligned with the platform instead of relying on a larger
     * local envelope that would be rejected at runtime.
     */
    const val MAX_PACKET_BYTES = 32 * 1024

    enum class Type(val code: Int) {
        HELLO(1),
        PING(2),
        PONG(3),
        ARM(4),
        START(5),
        STOP(6),
        HEARTBEAT(7),
        CALIBRATION(8),
        DEVICE_DESCRIPTOR(9),
        CAPTURE_ACK(10),
        CALIBRATION_FRAME(11),
        POSE_FRAME(12),
        FACE_FRAME(13),
        EYE_FRAME(14),
        ERROR(15),
        FACE_CONTOUR_FRAME(16),
        CALIBRATION_RESULT(17),
        DEPTH_FRAME(18),
        HAND_FRAME(19),
        FACE_MESH_FRAME(20),
        DENSE_LANDMARK_FRAME(21),
        MULTI_ACTOR_POSE_FRAME(22),
        DENSE_LANDMARK_FRAGMENT(23),
        MULTI_ACTOR_POSE_FRAGMENT(24),
        ROUTED_EVIDENCE_FRAME(25),
        FACE_MESH_FRAGMENT(26),
        SECURE_REMOTE_EVIDENCE_FRAME(27),
        TIMING_BEACON_OBSERVATION(28),
        EVIDENCE_FRAGMENT_FEEDBACK(29),
        DEPTH_FRAGMENT(30),
    }

    data class Packet(
        val type: Type,
        val sequence: Long,
        val timestampNanos: Long,
        val payload: ByteArray,
    )

    fun encode(
        type: Type,
        sequence: Long,
        timestampNanos: Long,
        payload: ByteArray = ByteArray(0),
    ): ByteArray {
        require(payload.size <= MAX_PACKET_BYTES - HEADER_BYTES) {
            "NoX control payload exceeds " + (MAX_PACKET_BYTES - HEADER_BYTES) + " bytes"
        }
        val crc = CRC32().apply { update(payload) }.value.toInt()
        return ByteBuffer.allocate(HEADER_BYTES + payload.size)
            .order(ByteOrder.BIG_ENDIAN)
            .apply {
                putInt(MAGIC)
                putInt(VERSION)
                putInt(type.code)
                putLong(sequence)
                putLong(timestampNanos)
                putInt(crc)
                putInt(payload.size)
                put(payload)
            }
            .array()
    }

    fun decode(bytes: ByteArray): Packet {
        require(bytes.size in HEADER_BYTES..MAX_PACKET_BYTES) { "Invalid NoX packet size" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        require(buffer.int == MAGIC) { "Invalid NoX magic" }
        require(buffer.int == VERSION) { "Unsupported NoX protocol version" }

        val typeCode = buffer.int
        val type = Type.entries.firstOrNull { it.code == typeCode }
            ?: error("Unknown NoX packet type=" + typeCode)
        val sequence = buffer.long
        val timestampNanos = buffer.long
        val crc = buffer.int
        val length = buffer.int

        require(length >= 0 && length == bytes.size - HEADER_BYTES) {
            "Invalid NoX payload length"
        }

        val payload = ByteArray(length)
        buffer.get(payload)
        val actual = CRC32().apply { update(payload) }.value.toInt()
        require(actual == crc) { "NoX payload CRC mismatch" }

        return Packet(type, sequence, timestampNanos, payload)
    }
}
