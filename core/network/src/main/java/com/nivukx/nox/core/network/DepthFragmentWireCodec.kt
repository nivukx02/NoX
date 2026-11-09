package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

object DepthFragmentWireCodec {

    private const val VERSION = 1
    private const val HEADER_BYTES = 32
    private const val MAX_DEVICE_ID_BYTES = 64
    private const val MAX_STREAM_ID_BYTES = 96
    private const val MAX_CHUNKS = 8

    fun encode(
        chunk: DepthChunkedWireCodec.Chunk,
        packetSequence: Long,
    ): ByteArray {
        require(packetSequence >= 0L)
        require(chunk.chunkCount in 1..MAX_CHUNKS)
        require(chunk.chunkIndex in 0 until chunk.chunkCount)
        require(
            DepthChunkedWireCodec.crc32(chunk.payload) == chunk.crc32,
        ) {
            "Depth fragment CRC mismatch before transmission"
        }
        require(
            DepthChunkedWireCodec.timestampFromPayload(chunk.payload) ==
                chunk.timestampNanos,
        ) {
            "Depth fragment timestamp mismatch before transmission"
        }

        val device =
            chunk.deviceId.value
                .toByteArray(Charsets.UTF_8)
        val stream =
            chunk.streamId
                .toByteArray(Charsets.UTF_8)

        require(device.size in 1..MAX_DEVICE_ID_BYTES)
        require(stream.size in 1..MAX_STREAM_ID_BYTES)

        val payload =
            ByteBuffer
                .allocate(
                    HEADER_BYTES +
                        device.size +
                        stream.size +
                        chunk.payload.size,
                )
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putShort(device.size.toShort())
                    putShort(stream.size.toShort())
                    putShort(chunk.chunkIndex.toShort())
                    putShort(chunk.chunkCount.toShort())
                    putLong(chunk.frameSequence)
                    putInt(chunk.crc32.toInt())
                    put(device)
                    put(stream)
                    put(chunk.payload)
                }
                .array()

        require(
            payload.size <=
                NoXWireProtocol.MAX_PACKET_BYTES - 36,
        )

        return NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.DEPTH_FRAGMENT,
            sequence = packetSequence,
            timestampNanos = chunk.timestampNanos,
            payload = payload,
        )
    }

    fun decode(
        packet: NoXWireProtocol.Packet,
    ): DepthChunkedWireCodec.Chunk {
        require(
            packet.type ==
                NoXWireProtocol.Type.DEPTH_FRAGMENT,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.remaining() >= HEADER_BYTES)
        require(buffer.int == VERSION)

        val deviceLength =
            buffer.short.toInt() and 0xFFFF
        val streamLength =
            buffer.short.toInt() and 0xFFFF
        val chunkIndex =
            buffer.short.toInt() and 0xFFFF
        val chunkCount =
            buffer.short.toInt() and 0xFFFF
        val frameSequence = buffer.long
        val expectedCrc =
            buffer.int.toLong() and
                0xFFFF_FFFFL

        require(deviceLength in 1..MAX_DEVICE_ID_BYTES)
        require(streamLength in 1..MAX_STREAM_ID_BYTES)
        require(chunkCount in 1..MAX_CHUNKS)
        require(chunkIndex in 0 until chunkCount)
        require(frameSequence >= 0L)
        require(buffer.remaining() >= deviceLength + streamLength + 1)

        val deviceId =
            ByteArray(deviceLength)
                .also(buffer::get)
                .toString(Charsets.UTF_8)

        val streamId =
            ByteArray(streamLength)
                .also(buffer::get)
                .toString(Charsets.UTF_8)

        require(deviceId.isNotBlank())
        require(streamId.isNotBlank())

        val payload =
            ByteArray(buffer.remaining())
                .also(buffer::get)

        require(
            DepthChunkedWireCodec.crc32(payload) ==
                expectedCrc,
        ) {
            "Depth fragment CRC mismatch"
        }

        require(
            DepthChunkedWireCodec
                .timestampFromPayload(payload) ==
                packet.timestampNanos,
        ) {
            "Depth fragment timestamp mismatch"
        }

        return DepthChunkedWireCodec.Chunk(
            deviceId = com.nivukx.nox.core.model.DeviceId(deviceId),
            streamId = streamId,
            frameSequence = frameSequence,
            timestampNanos = packet.timestampNanos,
            chunkIndex = chunkIndex,
            chunkCount = chunkCount,
            payload = payload,
            crc32 = expectedCrc,
        )
    }
}
