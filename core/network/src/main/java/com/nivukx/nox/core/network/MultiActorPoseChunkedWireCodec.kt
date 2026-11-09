package com.nivukx.nox.core.network

/**
 * Lossless bounded fragmentation for multi-actor pose evidence.
 *
 * The base pose codec remains optimized for one packet. This layer makes its 64-actor semantic
 * capacity independent from transport packet size by greedily packing whole actors according to the
 * actual encoded size. No actor is silently dropped.
 */
object MultiActorPoseChunkedWireCodec {

    private const val MAX_CHUNKS = 16
    private const val NOX_HEADER_BYTES = 36
    private const val FRAGMENT_HEADER_BYTES = 24
    private const val MAX_STREAM_ID_BYTES = 96
    private const val MAX_PACKET_PAYLOAD =
        NoXWireProtocol.MAX_PACKET_BYTES -
            NOX_HEADER_BYTES -
            FRAGMENT_HEADER_BYTES -
            MAX_STREAM_ID_BYTES

    data class Chunk(
        val streamId: String,
        val frameSequence: Long,
        val timestampNanos: Long,
        val chunkIndex: Int,
        val chunkCount: Int,
        val payload: ByteArray,
    ) {
        init {
            require(streamId.isNotBlank())
            require(frameSequence >= 0L)
            require(timestampNanos > 0L)
            require(chunkCount in 1..MAX_CHUNKS)
            require(chunkIndex in 0 until chunkCount)
            require(payload.isNotEmpty())
            require(
                payload.size <=
                    MAX_PACKET_PAYLOAD,
            )
        }
    }

    fun encode(
        frame: MultiActorPoseWireCodec.Frame,
        streamId: String,
        frameSequence: Long,
        maximumPayloadBytes: Int =
            MAX_PACKET_PAYLOAD,
    ): List<Chunk> {
        require(streamId.isNotBlank())
        require(frameSequence >= 0L)
        require(frame.timestampNanos > 0L)
        require(
            maximumPayloadBytes in
                1024..MAX_PACKET_PAYLOAD,
        )

        val actors =
            frame.actors
                .distinctBy { it.actorId }

        require(actors.size <= 64) {
            "Multi-actor pose exceeds the 64-actor semantic capacity"
        }
        require(actors.size == frame.actors.size) {
            "Duplicate actor IDs cannot be fragmented"
        }

        val groups =
            ArrayList<List<MultiActorPoseWireCodec.Actor>>()
        var current =
            ArrayList<MultiActorPoseWireCodec.Actor>()

        fun encodedSize(
            candidate:
                List<MultiActorPoseWireCodec.Actor>,
        ): Int =
            MultiActorPoseWireCodec.encodedSize(
                frame.copy(actors = candidate),
            )

        actors.forEach { actor ->
            val candidate =
                ArrayList(current.size + 1)
                    .apply {
                        addAll(current)
                        add(actor)
                    }

            if (
                current.isNotEmpty() &&
                encodedSize(candidate) >
                    maximumPayloadBytes
            ) {
                groups += current.toList()
                current =
                    ArrayList<
                        MultiActorPoseWireCodec.Actor,
                    >(1)
                current += actor
            } else {
                current += actor
            }

            require(
                encodedSize(current) <=
                    maximumPayloadBytes,
            ) {
                "Single multi-actor record exceeds the transport safety budget"
            }
        }

        if (current.isNotEmpty()) {
            groups += current.toList()
        }

        if (groups.isEmpty()) {
            groups += emptyList()
        }

        require(groups.size <= MAX_CHUNKS) {
            "Multi-actor pose requires too many transport chunks"
        }

        return groups.mapIndexed { index, group ->
            Chunk(
                streamId = streamId,
                frameSequence = frameSequence,
                timestampNanos = frame.timestampNanos,
                chunkIndex = index,
                chunkCount = groups.size,
                payload =
                    MultiActorPoseWireCodec.encode(
                        frame.copy(actors = group),
                    ),
            )
        }
    }

    fun decode(
        chunks: List<Chunk>,
    ): MultiActorPoseWireCodec.Frame {
        require(chunks.isNotEmpty())

        val first = chunks.first()
        require(chunks.size == first.chunkCount)

        val expected =
            (0 until first.chunkCount).toSet()
        val actual =
            chunks.map { it.chunkIndex }.toSet()

        require(actual == expected) {
            "Incomplete or duplicated multi-actor pose chunks"
        }

        chunks.forEach { chunk ->
            require(chunk.streamId == first.streamId)
            require(chunk.frameSequence == first.frameSequence)
            require(chunk.timestampNanos == first.timestampNanos)
            require(chunk.chunkCount == first.chunkCount)
        }

        val frames =
            chunks
                .sortedBy { it.chunkIndex }
                .map {
                    MultiActorPoseWireCodec.decode(
                        it.payload,
                    )
                }

        val actors =
            frames.flatMap { it.actors }

        require(
            actors.distinctBy { it.actorId }.size ==
                actors.size,
        ) {
            "Reassembled multi-actor pose contains duplicate actor IDs"
        }

        val base =
            frames.first()

        require(
            frames.all {
                it.timestampNanos ==
                    base.timestampNanos &&
                    it.sourceDeviceId ==
                    base.sourceDeviceId &&
                    it.inputWidth ==
                    base.inputWidth &&
                    it.inputHeight ==
                    base.inputHeight
            },
        ) {
            "Multi-actor pose chunk metadata is inconsistent"
        }

        return base.copy(
            actors = actors,
        )
    }
}
