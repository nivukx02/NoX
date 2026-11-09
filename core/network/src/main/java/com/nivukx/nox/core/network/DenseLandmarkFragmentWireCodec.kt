package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

/**
 * Serializes a DenseLandmarkChunk into the existing NoX packet envelope.
 *
 * The packet envelope keeps the established 32 KiB safety limit. Fragment payloads therefore remain
 * bounded by DenseLandmarkChunkedWireCodec before reaching Nearby transport.
 */
object DenseLandmarkFragmentWireCodec {

    private const val VERSION = 1
    private const val HEADER_BYTES = 28
    private const val MAX_STREAM_ID_BYTES = 96
    private const val MAX_CHUNKS = 16

    fun encode(
        chunk: DenseLandmarkChunkedWireCodec.Chunk,
        packetSequence: Long,
    ): ByteArray {
        val streamBytes =
            chunk.streamId.toByteArray(Charsets.UTF_8)

        require(streamBytes.size <= MAX_STREAM_ID_BYTES)

        val payload =
            ByteBuffer
                .allocate(
                    HEADER_BYTES +
                        streamBytes.size +
                        chunk.payload.size,
                )
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putShort(streamBytes.size.toShort())
                    putShort(chunk.chunkIndex.toShort())
                    putShort(chunk.chunkCount.toShort())
                    putShort(0)
                    putLong(chunk.frameSequence)
                    putLong(chunk.crc32)
                    put(streamBytes)
                    put(chunk.payload)
                }
                .array()

        require(
            payload.size <=
                NoXWireProtocol.MAX_PACKET_BYTES - 36,
        ) {
            "Dense landmark fragment exceeds NoX transport safety budget"
        }

        return NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            sequence = packetSequence,
            timestampNanos = chunk.timestampNanos,
            payload = payload,
        )
    }

    fun decode(
        packet: NoXWireProtocol.Packet,
    ): DenseLandmarkChunkedWireCodec.Chunk {
        require(
            packet.type ==
                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.remaining() >= HEADER_BYTES) {
            "Dense landmark fragment payload is too small"
        }

        val version = buffer.int
        require(version == VERSION)

        val streamLength =
            buffer.short.toInt() and 0xFFFF
        val chunkIndex =
            buffer.short.toInt() and 0xFFFF
        val chunkCount =
            buffer.short.toInt() and 0xFFFF
        buffer.short
        val frameSequence = buffer.long
        val crc32 = buffer.long

        require(streamLength in 1..MAX_STREAM_ID_BYTES)
        require(chunkCount in 1..MAX_CHUNKS)
        require(chunkIndex in 0 until chunkCount)
        require(frameSequence >= 0L)
        require(crc32 in 0L..0xFFFF_FFFFL)
        require(buffer.remaining() >= streamLength)

        val streamId =
            ByteArray(streamLength)
                .also(buffer::get)
                .toString(Charsets.UTF_8)

        require(streamId.isNotBlank())
        require(buffer.remaining() > 0)

        val payload = ByteArray(buffer.remaining())
        buffer.get(payload)

        val actualCrc =
            CRC32().apply {
                update(payload)
            }.value

        require(actualCrc == crc32) {
            "Dense landmark fragment CRC mismatch"
        }

        require(
            DenseLandmarkWireCodec
                .decode(
                    bytes = payload,
                    timestampNanos = packet.timestampNanos,
                )
                .timestampNanos ==
                packet.timestampNanos,
        ) {
            "Dense landmark fragment payload validation failed"
        }

        return DenseLandmarkChunkedWireCodec.Chunk(
            streamId = streamId,
            frameSequence = frameSequence,
            timestampNanos = packet.timestampNanos,
            chunkIndex = chunkIndex,
            chunkCount = chunkCount,
            payload = payload,
            crc32 = crc32,
        )
    }
}
