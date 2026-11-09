package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder

object EvidenceFragmentFeedbackWireCodec {

    private const val VERSION = 1
    private const val HEADER_BYTES = 28
    private const val MAX_STREAM_ID_BYTES = 96
    private const val MAX_CHUNKS = 16

    enum class Direction(val code: Int) {
        ACK(1),
        NACK(2),
    }

    data class Feedback(
        val direction: Direction,
        val evidenceType: NoXWireProtocol.Type,
        val streamId: String,
        val frameSequence: Long,
        val chunkCount: Int,
        val receivedMask: Int,
    ) {
        init {
            require(
                evidenceType ==
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT ||
                    evidenceType ==
                    NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT ||
                    evidenceType ==
                    NoXWireProtocol.Type.FACE_MESH_FRAGMENT ||
                    evidenceType ==
                    NoXWireProtocol.Type.DEPTH_FRAGMENT,
            )
            require(streamId.isNotBlank())
            require(
                streamId.toByteArray(Charsets.UTF_8).size <=
                    MAX_STREAM_ID_BYTES,
            )
            require(frameSequence >= 0L)
            require(chunkCount in 1..MAX_CHUNKS)
            val fullMask = fullMask(chunkCount)
            require(receivedMask and fullMask == receivedMask)
        }

        fun missingChunkMask(): Int =
            fullMask(chunkCount) xor receivedMask

        fun missingChunkIndices(): Set<Int> =
            (0 until chunkCount)
                .filter { missingChunkMask() and (1 shl it) != 0 }
                .toSet()
    }

    fun encode(
        feedback: Feedback,
        packetSequence: Long,
        timestampNanos: Long,
    ): ByteArray {
        require(packetSequence >= 0L)
        require(timestampNanos > 0L)

        val stream =
            feedback.streamId
                .toByteArray(Charsets.UTF_8)

        require(stream.size in 1..MAX_STREAM_ID_BYTES)

        val body =
            ByteBuffer
                .allocate(
                    HEADER_BYTES +
                        stream.size,
                )
                .order(ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(VERSION)
                    putInt(feedback.evidenceType.code)
                    putShort(feedback.direction.code.toShort())
                    putShort(feedback.chunkCount.toShort())
                    putLong(feedback.frameSequence)
                    putInt(feedback.receivedMask)
                    putShort(stream.size.toShort())
                    putShort(0)
                    put(stream)
                }
                .array()

        return NoXWireProtocol.encode(
            type =
                NoXWireProtocol.Type.EVIDENCE_FRAGMENT_FEEDBACK,
            sequence = packetSequence,
            timestampNanos = timestampNanos,
            payload = body,
        )
    }

    fun decode(
        packet: NoXWireProtocol.Packet,
    ): Feedback {
        require(
            packet.type ==
                NoXWireProtocol.Type.EVIDENCE_FRAGMENT_FEEDBACK,
        )

        val buffer =
            ByteBuffer
                .wrap(packet.payload)
                .order(ByteOrder.BIG_ENDIAN)

        require(buffer.remaining() >= HEADER_BYTES)

        require(buffer.int == VERSION)

        val evidenceTypeCode = buffer.int
        val evidenceType =
            NoXWireProtocol.Type.entries
                .firstOrNull {
                    it.code == evidenceTypeCode
                }
                ?: error(
                    "Unknown evidence fragment type",
                )

        val directionCode =
            buffer.short.toInt()

        val direction =
            Direction.entries
                .firstOrNull {
                    it.code == directionCode
                }
                ?: error(
                    "Unknown fragment feedback direction",
                )

        val chunkCount =
            buffer.short.toInt() and 0xFFFF
        val frameSequence = buffer.long
        val receivedMask = buffer.int
        val streamLength =
            buffer.short.toInt() and 0xFFFF
        buffer.short

        require(
            evidenceType ==
                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT ||
                evidenceType ==
                NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT ||
                evidenceType ==
                NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
        )
        require(chunkCount in 1..MAX_CHUNKS)
        require(frameSequence >= 0L)
        require(streamLength in 1..MAX_STREAM_ID_BYTES)
        require(buffer.remaining() == streamLength)

        val streamId =
            ByteArray(streamLength)
                .also(buffer::get)
                .toString(Charsets.UTF_8)

        return Feedback(
            direction = direction,
            evidenceType = evidenceType,
            streamId = streamId,
            frameSequence = frameSequence,
            chunkCount = chunkCount,
            receivedMask = receivedMask,
        )
    }

    private fun fullMask(
        chunkCount: Int,
    ): Int =
        (1 shl chunkCount) - 1
}
