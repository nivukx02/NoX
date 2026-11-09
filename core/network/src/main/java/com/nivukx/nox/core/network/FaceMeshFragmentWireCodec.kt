package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

object FaceMeshFragmentWireCodec {

    private const val VERSION = 1
    private const val HEADER_BYTES = 24
    private const val MAX_STREAM_ID_BYTES = 96
    private const val MAX_CHUNKS = 16

    fun encode(
        chunk: FaceMeshChunkedWireCodec.Chunk,
        packetSequence: Long,
    ): ByteArray {
        val streamBytes =
            chunk.streamId.toByteArray(Charsets.UTF_8)

        require(
            streamBytes.size in 1..MAX_STREAM_ID_BYTES,
        )

        val body =
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
                    putInt(chunk.crc32.toInt())
                    put(streamBytes)
                    put(chunk.payload)
                }
                .array()

        require(
            body.size <=
                NoXWireProtocol.MAX_PACKET_BYTES -
                    36,
        )

        return NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
            sequence = packetSequence,
            timestampNanos = chunk.timestampNanos,
            payload = body,
        )
    }

    fun decode(
        packet: NoXWireProtocol.Packet,
    ): FaceMeshChunkedWireCodec.Chunk {
        require(
            packet.type ==
                NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.remaining() >= HEADER_BYTES)
        require(buffer.int == VERSION)

        val streamLength =
            buffer.short.toInt() and 0xFFFF
        val chunkIndex =
            buffer.short.toInt() and 0xFFFF
        val chunkCount =
            buffer.short.toInt() and 0xFFFF
        buffer.short
        val frameSequence =
            buffer.long
        val expectedCrc =
            buffer.int.toLong() and
                0xFFFF_FFFFL

        require(streamLength in 1..MAX_STREAM_ID_BYTES)
        require(chunkCount in 1..MAX_CHUNKS)
        require(chunkIndex in 0 until chunkCount)
        require(frameSequence >= 0L)
        require(buffer.remaining() >= streamLength)

        val streamId =
            ByteArray(streamLength)
                .also(buffer::get)
                .toString(Charsets.UTF_8)

        require(streamId.isNotBlank())

        val payload =
            ByteArray(buffer.remaining())
                .also(buffer::get)

        val actualCrc =
            CRC32()
                .apply { update(payload) }
                .value

        require(
            actualCrc ==
                expectedCrc,
        ) {
            "Face mesh fragment CRC mismatch"
        }

        FaceMeshWireCodec.decode(payload)

        return FaceMeshChunkedWireCodec.Chunk(
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
