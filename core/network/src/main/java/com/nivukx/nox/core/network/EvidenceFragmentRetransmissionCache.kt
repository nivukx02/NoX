package com.nivukx.nox.core.network

/**
 * Bounded outbound cache for app-level fragment retransmission.
 *
 * Nearby BYTES delivery is a whole-payload operation and an accepted send does not guarantee remote
 * delivery. This cache lets NoX selectively replay missing evidence fragments with a fresh outer
 * packet sequence while keeping strict frame, byte, retry, and age bounds.
 */
class EvidenceFragmentRetransmissionCache(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumInFlightFrames: Int = 64,
        val maximumInFlightChunksPerFrame: Int = 16,
        val maximumRetainedBytes: Int = 8 * 1024 * 1024,
        val maximumFrameAgeNanos: Long = 500_000_000L,
        val maximumRetriesPerFrame: Int = 3,
    ) {
        init {
            require(maximumInFlightFrames in 1..256)
            require(maximumInFlightChunksPerFrame in 1..16)
            require(maximumRetainedBytes in 64 * 1024..128 * 1024 * 1024)
            require(maximumFrameAgeNanos > 0L)
            require(maximumRetriesPerFrame in 0..8)
        }
    }

    data class Key(
        val endpoint: String,
        val evidenceType: NoXWireProtocol.Type,
        val streamId: String,
        val frameSequence: Long,
    ) {
        init {
            require(endpoint.isNotBlank())
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
            require(frameSequence >= 0L)
        }
    }

    data class Retransmission(
        val packet: ByteArray,
        val reconstructionCritical: Boolean,
        val chunkIndex: Int,
    )

    private data class Entry(
        val packet: NoXWireProtocol.Packet,
        val reconstructionCritical: Boolean,
        val bytes: Int,
    )

    private data class FrameCache(
        val key: Key,
        val chunkCount: Int,
        val firstSeenNanos: Long,
        val chunks: MutableMap<Int, Entry>,
        var retryCount: Int,
    )

    private val frames =
        LinkedHashMap<Key, FrameCache>(
            16,
            0.75f,
            true,
        )
    private var retainedBytes = 0L

    @Synchronized
    fun remember(
        endpoint: String,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
        chunkIndex: Int,
        chunkCount: Int,
        packet: ByteArray,
        reconstructionCritical: Boolean,
        nowNanos: Long,
    ) {
        require(packet.isNotEmpty())
        require(packet.size <= NoXWireProtocol.MAX_PACKET_BYTES)

        expire(nowNanos)

        val decoded =
            NoXWireProtocol.decode(packet)

        require(decoded.type == evidenceType) {
            "Fragment cache packet type mismatch"
        }
        require(evidenceType != NoXWireProtocol.Type.ROUTED_EVIDENCE_FRAME)
        require(chunkCount in 1..config.maximumInFlightChunksPerFrame)
        require(chunkIndex in 0 until chunkCount)

        val key =
            Key(
                endpoint = endpoint,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
            )

        val cache =
            frames[key]
                ?: createFrame(
                    key = key,
                    chunkCount = chunkCount,
                    nowNanos = nowNanos,
                )

        if (cache.chunkCount != chunkCount) {
            // A frame cannot be completed once chunk cardinality changes. Remove stale bytes before
            // rejecting the new fragment so repeated malformed input does not pin this cache key.
            removeFrame(key)
            throw IllegalArgumentException("Fragment cache chunk count changed for an active frame")
        }

        val old = cache.chunks[chunkIndex]
        if (old != null) {
            val sameLogicalFragment =
                old.packet.type == decoded.type &&
                    old.packet.timestampNanos == decoded.timestampNanos &&
                    old.packet.payload.contentEquals(decoded.payload)
            if (sameLogicalFragment) {
                if (reconstructionCritical && !old.reconstructionCritical) {
                    cache.chunks[chunkIndex] =
                        old.copy(reconstructionCritical = true)
                }
                return
            }

            // Never replace a fragment under the same frame/index key with different bytes. Doing
            // so would make an acknowledgement mask ambiguous and could corrupt retransmissions.
            removeFrame(key)
            throw IllegalArgumentException("Conflicting duplicate cached fragment")
        }

        try {
            ensureCapacity(
                additionalBytes = packet.size,
                preserve = key,
            )
        } catch (failure: Throwable) {
            // The current frame may itself have exhausted the byte budget. Remove it as well, or
            // its retained chunks would remain impossible to complete after every retry fails.
            removeFrame(key)
            throw failure
        }

        cache.chunks[chunkIndex] =
            Entry(
                packet =
                    decoded.copy(
                        payload =
                            decoded.payload.copyOf(),
                    ),
                reconstructionCritical =
                    reconstructionCritical,
                bytes = packet.size,
            )

        retainedBytes +=
            packet.size.toLong()
    }

    @Synchronized
    fun handle(
        endpoint: String,
        feedback:
            EvidenceFragmentFeedbackWireCodec.Feedback,
        nowNanos: Long,
        nextPacketSequence: () -> Long,
    ): List<Retransmission> {
        require(endpoint.isNotBlank())
        expire(nowNanos)

        val key =
            Key(
                endpoint = endpoint,
                evidenceType =
                    feedback.evidenceType,
                streamId = feedback.streamId,
                frameSequence =
                    feedback.frameSequence,
            )

        val cache = frames[key]
            ?: return emptyList()

        require(
            cache.chunkCount ==
                feedback.chunkCount,
        ) {
            "Fragment feedback chunk count mismatch"
        }

        when (feedback.direction) {
            EvidenceFragmentFeedbackWireCodec.Direction.ACK -> {
                removeMask(
                    cache = cache,
                    mask = feedback.receivedMask,
                )
                return emptyList()
            }

            EvidenceFragmentFeedbackWireCodec.Direction.NACK -> {
                if (
                    cache.retryCount >=
                        config.maximumRetriesPerFrame
                ) {
                    removeFrame(key)
                    return emptyList()
                }

                cache.retryCount += 1

                return feedback.missingChunkIndices()
                    .mapNotNull { chunkIndex ->
                        cache.chunks[chunkIndex]
                            ?.let { entry ->
                                Triple(
                                    chunkIndex,
                                    entry.reconstructionCritical,
                                    entry,
                                )
                            }
                    }
                    .sortedWith(
                        compareByDescending<
                            Triple<Int, Boolean, Entry>
                        > { it.second }
                            .thenBy { it.first },
                    )
                    .mapNotNull { (chunkIndex, _, entry) ->
                                val sequence =
                                    nextPacketSequence()
                                require(sequence >= 0L)

                                Retransmission(
                                    packet =
                                        NoXWireProtocol.encode(
                                            type =
                                                entry.packet.type,
                                            sequence = sequence,
                                            timestampNanos =
                                                entry.packet.timestampNanos,
                                            payload =
                                                entry.packet.payload,
                                        ),
                                    reconstructionCritical =
                                        entry.reconstructionCritical,
                                    chunkIndex =
                                        chunkIndex,
                                )
                            }
                    }
            }
        }
    }

    @Synchronized
    fun expire(nowNanos: Long) {
        require(nowNanos >= 0L)

        val iterator =
            frames.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                nowNanos -
                    entry.value.firstSeenNanos >=
                    config.maximumFrameAgeNanos
            ) {
                retainedBytes =
                    (
                        retainedBytes -
                            entry.value.chunks.values.sumOf {
                                it.bytes.toLong()
                            }
                        ).coerceAtLeast(0L)
                iterator.remove()
            }
        }
    }

    @Synchronized
    fun clearEndpoint(endpoint: String) {
        require(endpoint.isNotBlank())

        val keys =
            frames.keys
                .filter {
                    it.endpoint == endpoint
                }

        keys.forEach(::removeFrame)
    }

    @Synchronized
    fun clear() {
        frames.clear()
        retainedBytes = 0L
    }

    @Synchronized
    fun inFlightFrameCount(): Int =
        frames.size

    @Synchronized
    fun retainedBytes(): Int =
        retainedBytes
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()

    private fun removeMask(
        cache: FrameCache,
        mask: Int,
    ) {
        val indices =
            cache.chunks.keys
                .filter { index ->
                    mask and
                        (1 shl index) !=
                        0
                }

        indices.forEach { index ->
            val removed =
                cache.chunks.remove(index)
            if (removed != null) {
                retainedBytes =
                    (
                        retainedBytes -
                            removed.bytes.toLong()
                        ).coerceAtLeast(0L)
            }
        }

        if (cache.chunks.isEmpty()) {
            removeFrame(cache.key)
        }
    }

    private fun createFrame(
        key: Key,
        chunkCount: Int,
        nowNanos: Long,
    ): FrameCache {
        if (
            frames.size >=
                config.maximumInFlightFrames
        ) {
            frames.entries
                .firstOrNull()
                ?.key
                ?.let(::removeFrame)
        }

        return FrameCache(
            key = key,
            chunkCount = chunkCount,
            firstSeenNanos = nowNanos,
            chunks = LinkedHashMap(),
            retryCount = 0,
        ).also {
            frames[key] = it
        }
    }

    private fun ensureCapacity(
        additionalBytes: Int,
        preserve: Key,
    ) {
        require(
            additionalBytes <=
                config.maximumRetainedBytes,
        )

        while (
            retainedBytes +
                additionalBytes.toLong() >
                config.maximumRetainedBytes
        ) {
            val victim =
                frames.entries
                    .firstOrNull {
                        it.key != preserve
                    }
                    ?.key
                    ?: break

            removeFrame(victim)
        }

        require(
            retainedBytes +
                additionalBytes.toLong() <=
                config.maximumRetainedBytes
        ) {
            "Fragment retransmission cache byte budget exhausted"
        }
    }

    private fun removeFrame(
        key: Key,
    ) {
        val removed =
            frames.remove(key)
                ?: return

        retainedBytes =
            (
                retainedBytes -
                    removed.chunks.values.sumOf {
                        it.bytes.toLong()
                    }
                ).coerceAtLeast(0L)
    }
}
