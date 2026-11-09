package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshFrame
import java.util.zip.CRC32

/**
 * Lossless byte-level fragmentation for validated face meshes.
 *
 * Face triangle topology is cross-boundary by nature, so semantic point chunking can silently
 * destroy connectivity. This codec therefore validates one complete mesh with FaceMeshWireCodec,
 * then fragments the exact encoded bytes. Reassembly returns the exact original model after one
 * final decode.
 */
object FaceMeshChunkedWireCodec {

    private const val DEFAULT_BYTES_PER_CHUNK = 45 * 1024
    private const val MAX_CHUNKS = 16
    private const val MAX_STREAM_ID_BYTES = 96
    private const val MIN_BYTES_PER_CHUNK = 4_096
    private const val DIRECT_PACKET_PAYLOAD_BUDGET =
        NoXWireProtocol.MAX_PACKET_BYTES - 36 - 24 - MAX_STREAM_ID_BYTES

    data class Chunk(
        val streamId: String,
        val frameSequence: Long,
        val timestampNanos: Long,
        val chunkIndex: Int,
        val chunkCount: Int,
        val payload: ByteArray,
        val crc32: Long,
    ) {
        init {
            require(streamId.isNotBlank())
            require(
                streamId.toByteArray(Charsets.UTF_8).size <=
                    MAX_STREAM_ID_BYTES,
            )
            require(frameSequence >= 0L)
            require(timestampNanos > 0L)
            require(chunkCount in 1..MAX_CHUNKS)
            require(chunkIndex in 0 until chunkCount)
            require(payload.isNotEmpty())
            require(
                payload.size <
                    NoXWireProtocol.MAX_PACKET_BYTES -
                        256,
            )
            require(crc32 in 0L..0xFFFF_FFFFL)
        }
    }

    fun encode(
        frame: FaceMeshFrame,
        streamId: String,
        frameSequence: Long,
        maximumBytesPerChunk: Int =
            DEFAULT_BYTES_PER_CHUNK,
        maximumPayloadBytes: Int =
            DIRECT_PACKET_PAYLOAD_BUDGET,
    ): List<Chunk> {
        require(streamId.isNotBlank())
        require(
            streamId.toByteArray(Charsets.UTF_8).size <=
                MAX_STREAM_ID_BYTES,
        )
        require(frameSequence >= 0L)
        require(
            maximumBytesPerChunk in
                MIN_BYTES_PER_CHUNK..DEFAULT_BYTES_PER_CHUNK,
        )
        require(
            maximumPayloadBytes in
                MIN_BYTES_PER_CHUNK..DIRECT_PACKET_PAYLOAD_BUDGET,
        )
        val effectiveBytesPerChunk =
            minOf(
                maximumBytesPerChunk,
                maximumPayloadBytes,
            )
        require(frame.timestampNanos > 0L)

        val bytes =
            FaceMeshWireCodec.encode(frame)

        val chunkCount =
            (
                bytes.size +
                    effectiveBytesPerChunk -
                    1
                ) / effectiveBytesPerChunk

        require(chunkCount in 1..MAX_CHUNKS) {
            "Face mesh requires too many transport fragments"
        }

        return (0 until chunkCount).map { index ->
            val from =
                index *
                    effectiveBytesPerChunk
            val to =
                minOf(
                    bytes.size,
                    from +
                        effectiveBytesPerChunk,
                )
            val payload =
                bytes.copyOfRange(
                    from,
                    to,
                )

            Chunk(
                streamId = streamId,
                frameSequence = frameSequence,
                timestampNanos = frame.timestampNanos,
                chunkIndex = index,
                chunkCount = chunkCount,
                payload = payload,
                crc32 = crc(payload),
            )
        }
    }

    fun decode(
        chunks: List<Chunk>,
    ): FaceMeshFrame {
        require(chunks.isNotEmpty())

        val first =
            chunks.first()

        require(
            chunks.size ==
                first.chunkCount,
        )

        require(
            chunks.map { it.chunkIndex }.toSet() ==
                (0 until first.chunkCount).toSet(),
        ) {
            "Face mesh fragments are incomplete or duplicated"
        }

        chunks.forEach { chunk ->
            require(
                chunk.streamId ==
                    first.streamId,
            )
            require(
                chunk.frameSequence ==
                    first.frameSequence,
            )
            require(
                chunk.timestampNanos ==
                    first.timestampNanos,
            )
            require(
                crc(chunk.payload) ==
                    chunk.crc32,
            ) {
                "Face mesh fragment CRC mismatch"
            }
        }

        val ordered =
            chunks.sortedBy { it.chunkIndex }
        val encodedSize =
            ordered.sumOf { it.payload.size }
        require(
            encodedSize > 0 &&
                encodedSize <=
                    MAX_CHUNKS *
                        (NoXWireProtocol.MAX_PACKET_BYTES - 256),
        ) {
            "Reassembled face mesh exceeds bounded frame budget"
        }

        val encoded =
            ByteArray(encodedSize)
        var offset = 0
        ordered.forEach { chunk ->
            chunk.payload.copyInto(
                destination = encoded,
                destinationOffset = offset,
            )
            offset += chunk.payload.size
        }

        return FaceMeshWireCodec.decode(
            encoded,
        )
    }

    private fun crc(
        payload: ByteArray,
    ): Long =
        CRC32()
            .apply { update(payload) }
            .value
}
