package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.DepthSource
import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DepthFragmentTransportTest {

    @Test
    fun oversizedMetricDepthPayloadRoundTripsLosslessly() {
        val frame =
            DepthFrame(
                timestampNanos = 14_000_000_000L,
                source = DepthSource.SENSOR_DEPTH,
                width = 160,
                height = 90,
                depthMeters =
                    FloatArray(160 * 90) { index ->
                        1.0f +
                            (index % 97) * 0.0025f
                    },
            )

        val encoded =
            requireNotNull(
                DepthWireCodec.encode(
                    deviceId = DeviceId("camera-depth"),
                    frame = frame,
                ),
            )

        val chunks =
            DepthChunkedWireCodec.encode(
                deviceId = DeviceId("camera-depth"),
                payload = encoded,
                streamId = "depth-rig",
                frameSequence = 7L,
            )

        assertTrue(chunks.size >= 2)
        assertTrue(
            chunks.all {
                it.payload.size <
                    NoXWireProtocol.MAX_PACKET_BYTES
            },
        )

        val packets =
            chunks.mapIndexed { index, chunk ->
                DepthFragmentWireCodec.encode(
                    chunk = chunk,
                    packetSequence = 100L + index,
                )
            }

        val reassembler =
            DepthFragmentReassembler()

        var result: DepthWireCodec.Frame? = null

        packets.reversed().forEach { packet ->
            val chunk =
                DepthFragmentWireCodec.decode(
                    NoXWireProtocol.decode(packet),
                )
            result =
                reassembler.accept(
                    endpoint = "peer-depth",
                    chunk = chunk,
                    nowNanos = 1_000L,
                ) ?: result
        }

        val wire =
            requireNotNull(result)

        assertEquals(frame.timestampNanos, wire.timestampNanos)
        assertEquals(frame.width, wire.width)
        assertEquals(frame.height, wire.height)
        assertEquals(
            frame.depthMeters.size,
            wire.depthMeters.size,
        )
        assertEquals(
            frame.depthMeters[1234],
            wire.depthMeters[1234],
        )
    }

    @Test
    fun conflictingDuplicateDepthFragmentIsRejected() {
        val payload =
            byteArrayOf(1, 2, 3, 4, 5, 6)

        val chunk =
            DepthChunkedWireCodec.Chunk(
                deviceId = DeviceId("cam"),
                streamId = "s",
                frameSequence = 1L,
                timestampNanos = 15_000_000_000L,
                chunkIndex = 0,
                chunkCount = 1,
                payload = payload,
                crc32 = DepthChunkedWireCodec.crc32(payload),
            )

        val reassembler =
            DepthFragmentReassembler()

        reassembler.accept(
            "peer",
            chunk,
            1L,
        )

        val conflicting =
            chunk.copy(
                payload =
                    byteArrayOf(
                        9, 8, 7, 6, 5, 4,
                    ),
            )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept(
                "peer",
                conflicting,
                2L,
            )
        }
    }

    @Test
    fun mixedDeviceIdentityDiscardsThePoisonedAssemblyAndReleasesRetainedBytes() {
        val timestamp = 15_000_000_000L
        fun fragment(device: String, index: Int): DepthChunkedWireCodec.Chunk {
            val payload = java.nio.ByteBuffer.allocate(16)
                .order(java.nio.ByteOrder.BIG_ENDIAN)
                .putLong(123L)
                .putLong(timestamp)
                .array()
            return DepthChunkedWireCodec.Chunk(
                deviceId = DeviceId(device),
                streamId = "rig",
                frameSequence = 73L,
                timestampNanos = timestamp,
                chunkIndex = index,
                chunkCount = 2,
                payload = payload,
                crc32 = DepthChunkedWireCodec.crc32(payload),
            )
        }
        val reassembler = DepthFragmentReassembler()
        reassembler.accept("peer", fragment("camera-a", 0), 1L)
        assertEquals(1, reassembler.inFlightAssemblyCount())
        assertEquals(16, reassembler.inFlightBytes())

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept("peer", fragment("camera-b", 1), 2L)
        }
        assertEquals(0, reassembler.inFlightAssemblyCount())
        assertEquals(0, reassembler.inFlightBytes())
    }

    @Test
    fun mismatchedFrameTimestampDiscardsAssemblyInsteadOfRetainingMemory() {
        val timestamp = 15_000_000_000L
        fun fragment(index: Int, fragmentTimestamp: Long): DepthChunkedWireCodec.Chunk {
            val payload = java.nio.ByteBuffer.allocate(16)
                .order(java.nio.ByteOrder.BIG_ENDIAN)
                .putLong(456L)
                .putLong(timestamp)
                .array()
            return DepthChunkedWireCodec.Chunk(
                deviceId = DeviceId("camera-a"),
                streamId = "rig",
                frameSequence = 74L,
                timestampNanos = fragmentTimestamp,
                chunkIndex = index,
                chunkCount = 2,
                payload = payload,
                crc32 = DepthChunkedWireCodec.crc32(payload),
            )
        }
        val reassembler = DepthFragmentReassembler()
        reassembler.accept("peer", fragment(0, timestamp), 1L)

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept("peer", fragment(1, timestamp + 1L), 2L)
        }
        assertEquals(0, reassembler.inFlightAssemblyCount())
        assertEquals(0, reassembler.inFlightBytes())
    }

    @Test
    fun corruptCompletedAssemblyIsRemovedWhenDecodeFails() {
        val timestamp = 15_000_000_000L
        val payload = java.nio.ByteBuffer.allocate(16)
            .order(java.nio.ByteOrder.BIG_ENDIAN)
            .putLong(789L)
            .putLong(timestamp + 1L)
            .array()
        val fragment = DepthChunkedWireCodec.Chunk(
            deviceId = DeviceId("camera-a"),
            streamId = "rig",
            frameSequence = 75L,
            timestampNanos = timestamp,
            chunkIndex = 0,
            chunkCount = 1,
            payload = payload,
            crc32 = DepthChunkedWireCodec.crc32(payload),
        )
        val reassembler = DepthFragmentReassembler()
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept("peer", fragment, 1L)
        }
        assertEquals(0, reassembler.inFlightAssemblyCount())
        assertEquals(0, reassembler.inFlightBytes())
    }


    @Test
    fun senderRejectsDepthFragmentWithIncorrectPayloadCrc() {
        val timestamp = 15_000_000_000L
        val payload = java.nio.ByteBuffer.allocate(16)
            .order(java.nio.ByteOrder.BIG_ENDIAN)
            .putLong(123L)
            .putLong(timestamp)
            .array()
        val chunk = DepthChunkedWireCodec.Chunk(
            deviceId = DeviceId("camera-a"),
            streamId = "rig",
            frameSequence = 76L,
            timestampNanos = timestamp,
            chunkIndex = 0,
            chunkCount = 1,
            payload = payload,
            crc32 = DepthChunkedWireCodec.crc32(payload) xor 1L,
        )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            DepthFragmentWireCodec.encode(chunk, packetSequence = 1L)
        }
    }

    @Test
    fun senderRejectsDepthFragmentWithTimestampDifferentFromPayload() {
        val payloadTimestamp = 15_000_000_000L
        val payload = java.nio.ByteBuffer.allocate(16)
            .order(java.nio.ByteOrder.BIG_ENDIAN)
            .putLong(123L)
            .putLong(payloadTimestamp)
            .array()
        val chunk = DepthChunkedWireCodec.Chunk(
            deviceId = DeviceId("camera-a"),
            streamId = "rig",
            frameSequence = 77L,
            timestampNanos = payloadTimestamp + 1L,
            chunkIndex = 0,
            chunkCount = 1,
            payload = payload,
            crc32 = DepthChunkedWireCodec.crc32(payload),
        )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            DepthFragmentWireCodec.encode(chunk, packetSequence = 2L)
        }
    }

}
