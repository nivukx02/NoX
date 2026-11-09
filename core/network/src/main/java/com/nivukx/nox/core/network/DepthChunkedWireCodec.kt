package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import java.util.zip.CRC32

object DepthChunkedWireCodec {

    private const val DEFAULT_MAX_PAYLOAD_BYTES = 30 * 1024
    private const val MAX_CHUNKS = 8
    private const val MAX_DEVICE_ID_BYTES = 64
    private const val MAX_STREAM_ID_BYTES = 96

    data class Chunk(
        val deviceId: DeviceId,
        val streamId: String,
        val frameSequence: Long,
        val timestampNanos: Long,
        val chunkIndex: Int,
        val chunkCount: Int,
        val payload: ByteArray,
        val crc32: Long,
    ) {
        init {
            require(deviceId.value.isNotBlank())
            require(
                deviceId.value.toByteArray(Charsets.UTF_8).size <=
                    MAX_DEVICE_ID_BYTES,
            )
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
                payload.size <=
                    DEFAULT_MAX_PAYLOAD_BYTES,
            )
            require(crc32 in 0L..0xFFFF_FFFFL)
        }
    }

    fun encode(
        deviceId: DeviceId,
        payload: ByteArray,
        streamId: String,
        frameSequence: Long,
        maximumPayloadBytes: Int =
            DEFAULT_MAX_PAYLOAD_BYTES,
    ): List<Chunk> {
        require(deviceId.value.isNotBlank())
        require(
            deviceId.value.toByteArray(Charsets.UTF_8).size <=
                MAX_DEVICE_ID_BYTES,
        )
        require(streamId.isNotBlank())
        require(
            streamId.toByteArray(Charsets.UTF_8).size <=
                MAX_STREAM_ID_BYTES,
        )
        require(frameSequence >= 0L)
        require(maximumPayloadBytes in 4 * 1024..DEFAULT_MAX_PAYLOAD_BYTES)
        require(payload.isNotEmpty())

        val chunkCount =
            (
                payload.size +
                    maximumPayloadBytes -
                    1
                ) / maximumPayloadBytes

        require(chunkCount in 1..MAX_CHUNKS) {
            "Depth payload exceeds bounded fragmentation envelope"
        }

        return (0 until chunkCount).map { index ->
            val from =
                index * maximumPayloadBytes
            val to =
                minOf(
                    payload.size,
                    from + maximumPayloadBytes,
                )
            val bytes =
                payload.copyOfRange(
                    from,
                    to,
                )

            Chunk(
                deviceId = deviceId,
                streamId = streamId,
                frameSequence = frameSequence,
                timestampNanos =
                    readTimestamp(payload),
                chunkIndex = index,
                chunkCount = chunkCount,
                payload = bytes,
                crc32 = crc32(bytes),
            )
        }
    }

    fun decode(
        chunks: List<Chunk>,
    ): ByteArray {
        require(chunks.isNotEmpty())

        val first = chunks.first()

        require(chunks.size == first.chunkCount) {
            "Incomplete depth fragment set"
        }

        val indices =
            chunks.map { it.chunkIndex }.toSet()

        require(
            indices ==
                (0 until first.chunkCount).toSet(),
        ) {
            "Depth fragment indices are incomplete or duplicated"
        }

        chunks.forEach { chunk ->
            require(chunk.deviceId == first.deviceId)
            require(chunk.streamId == first.streamId)
            require(chunk.frameSequence == first.frameSequence)
            require(chunk.timestampNanos == first.timestampNanos)
            require(chunk.chunkCount == first.chunkCount)
            require(
                crc32(chunk.payload) ==
                    chunk.crc32,
            ) {
                "Depth fragment CRC mismatch"
            }
        }

        val totalBytes =
            chunks.sumOf { it.payload.size }
        val output =
            ByteArray(totalBytes)
        var offset = 0
        chunks
            .sortedBy { it.chunkIndex }
            .forEach { chunk ->
                chunk.payload.copyInto(
                    destination = output,
                    destinationOffset = offset,
                )
                offset += chunk.payload.size
            }
        return output
    }

    fun timestampFromPayload(
        payload: ByteArray,
    ): Long {
        require(payload.size >= 16) {
            "Depth payload is too small"
        }

        return java.nio.ByteBuffer
            .wrap(payload)
            .order(java.nio.ByteOrder.BIG_ENDIAN)
            .apply {
                position(8)
            }
            .long
    }

    fun crc32(
        payload: ByteArray,
    ): Long =
        CRC32()
            .apply {
                update(payload)
            }
            .value

    private fun readTimestamp(
        payload: ByteArray,
    ): Long =
        timestampFromPayload(payload)
}
