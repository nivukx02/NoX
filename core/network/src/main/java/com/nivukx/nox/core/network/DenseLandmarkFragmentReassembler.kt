package com.nivukx.nox.core.network

/**
 * Memory-bounded reassembler for dense landmark fragments.
 *
 * This component intentionally accepts one complete frame at a time. Partial assemblies expire,
 * the global assembly count is bounded, and replacement of old entries is deterministic.
 */
class DenseLandmarkFragmentReassembler(
    private val config: Config = Config(),
) {

    data class FeedbackState(
        val endpoint: String,
        val streamId: String,
        val frameSequence: Long,
        val timestampNanos: Long,
        val chunkCount: Int,
        val receivedMask: Int,
    ) {
        fun missingMask(): Int =
            ((1 shl chunkCount) - 1) xor receivedMask
    }

    data class Config(
        val maximumInFlightAssemblies: Int = 32,
        val maximumInFlightBytes: Int = 16 * 1024 * 1024,
        val maximumAssemblyAgeNanos: Long = 250_000_000L,
    ) {
        init {
            require(maximumInFlightAssemblies in 1..256)
            require(maximumInFlightBytes in 64 * 1024..128 * 1024 * 1024)
            require(maximumAssemblyAgeNanos > 0L)
        }
    }

    private data class Key(
        val endpoint: String,
        val streamId: String,
        val frameSequence: Long,
    )

    private data class Assembly(
        val chunkCount: Int,
        val timestampNanos: Long,
        val firstSeenNanos: Long,
        val chunks: MutableMap<Int, DenseLandmarkChunkedWireCodec.Chunk>,
    )

    private val assemblies =
        LinkedHashMap<Key, Assembly>(
            16,
            0.75f,
            true,
        )
    private var retainedPayloadBytes = 0L

    @Synchronized
    fun accept(
        endpoint: String,
        chunk: DenseLandmarkChunkedWireCodec.Chunk,
        nowNanos: Long,
    ): DenseLandmarkWireCodec.Frame? {
        require(endpoint.isNotBlank())
        require(nowNanos >= 0L)

        expire(nowNanos)

        val key =
            Key(
                endpoint = endpoint,
                streamId = chunk.streamId,
                frameSequence = chunk.frameSequence,
            )

        val assembly =
            assemblies[key]
                ?: createAssembly(
                    key = key,
                    chunk = chunk,
                    nowNanos = nowNanos,
                )

        if (
            assembly.chunkCount != chunk.chunkCount ||
            assembly.timestampNanos != chunk.timestampNanos
        ) {
            // The old assembly can no longer complete consistently. Release its retained bytes
            // before the existing require below reports the malformed incoming fragment.
            val failed = assemblies.remove(key)
            if (failed != null) {
                retainedPayloadBytes =
                    (retainedPayloadBytes - failed.chunks.values.sumOf { it.payload.size.toLong() })
                        .coerceAtLeast(0L)
            }
        }

        require(
            assembly.chunkCount ==
                chunk.chunkCount,
        ) {
            "Dense landmark fragment chunk count changed during assembly"
        }

        require(
            assembly.timestampNanos ==
                chunk.timestampNanos,
        ) {
            "Dense landmark fragment timestamp changed during assembly"
        }

        val existing =
            assembly.chunks[chunk.chunkIndex]

        if (existing != null) {
            if (
                !existing.payload.contentEquals(chunk.payload) ||
                existing.crc32 != chunk.crc32
            ) {
                val failed = assemblies.remove(key)
                if (failed != null) {
                    retainedPayloadBytes =
                        (retainedPayloadBytes - failed.chunks.values.sumOf { it.payload.size.toLong() })
                            .coerceAtLeast(0L)
                }
            }
            require(
                existing.payload.contentEquals(
                    chunk.payload,
                ),
            ) {
                "Conflicting duplicate dense landmark fragment"
            }
            require(existing.crc32 == chunk.crc32) {
                "Conflicting duplicate dense landmark fragment CRC"
            }
            return null
        }

        ensureCapacity(
            additionalBytes = chunk.payload.size,
            preserve = key,
        )
        assembly.chunks[chunk.chunkIndex] =
            chunk
        retainedPayloadBytes +=
            chunk.payload.size.toLong()

        if (
            assembly.chunks.size !=
                assembly.chunkCount
        ) {
            return null
        }

        val ordered =
            assembly.chunks.values
                .sortedBy { it.chunkIndex }

        val frame =
            try {
                DenseLandmarkChunkedWireCodec.decode(ordered)
            } catch (failure: Throwable) {
                // Invalid CRCs or a corrupt completed frame must not retain its partial bytes.
                val failed = assemblies.remove(key)
                if (failed != null) {
                    retainedPayloadBytes =
                        (retainedPayloadBytes - failed.chunks.values.sumOf { it.payload.size.toLong() })
                            .coerceAtLeast(0L)
                }
                throw failure
            }

        assemblies.remove(key)
        retainedPayloadBytes =
            (
                retainedPayloadBytes -
                    assembly.chunks.values.sumOf {
                        it.payload.size.toLong()
                    }
            ).coerceAtLeast(0L)

        return frame
    }

    @Synchronized
    fun expire(
        nowNanos: Long,
    ) {
        expireWithFeedback(nowNanos)
    }

    @Synchronized
    fun expireWithFeedback(
        nowNanos: Long,
    ): List<FeedbackState> {
        require(nowNanos >= 0L)

        val expired =
            ArrayList<FeedbackState>()

        val iterator =
            assemblies.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                nowNanos -
                    entry.value.firstSeenNanos >=
                    config.maximumAssemblyAgeNanos
            ) {
                val mask =
                    entry.value.chunks.keys.fold(0) { acc, index ->
                        acc or (1 shl index)
                    }

                expired +=
                    FeedbackState(
                        endpoint = entry.key.endpoint,
                        streamId = entry.key.streamId,
                        frameSequence = entry.key.frameSequence,
                        timestampNanos =
                            entry.value.timestampNanos,
                        chunkCount =
                            entry.value.chunkCount,
                        receivedMask = mask,
                    )

                retainedPayloadBytes =
                    (
                        retainedPayloadBytes -
                            entry.value.chunks.values.sumOf {
                                it.payload.size.toLong()
                            }
                    ).coerceAtLeast(0L)
                iterator.remove()
            }
        }

        return expired
    }

    @Synchronized
    fun feedbackState(
        endpoint: String,
        streamId: String,
        frameSequence: Long,
    ): FeedbackState? {
        require(endpoint.isNotBlank())
        require(streamId.isNotBlank())
        require(frameSequence >= 0L)

        val key =
            Key(
                endpoint = endpoint,
                streamId = streamId,
                frameSequence = frameSequence,
            )

        val assembly =
            assemblies[key]
                ?: return null

        val mask =
            assembly.chunks.keys.fold(0) { acc, index ->
                acc or (1 shl index)
            }

        return FeedbackState(
            endpoint = endpoint,
            streamId = streamId,
            frameSequence = frameSequence,
            timestampNanos =
                assembly.timestampNanos,
            chunkCount =
                assembly.chunkCount,
            receivedMask = mask,
        )
    }

    @Synchronized
    fun clearEndpoint(
        endpoint: String,
    ) {
        require(endpoint.isNotBlank())
        val iterator =
            assemblies.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key.endpoint == endpoint) {
                retainedPayloadBytes =
                    (
                        retainedPayloadBytes -
                            entry.value.chunks.values.sumOf {
                                it.payload.size.toLong()
                            }
                    ).coerceAtLeast(0L)
                iterator.remove()
            }
        }
    }

    @Synchronized
    fun clear() {
        assemblies.clear()
        retainedPayloadBytes = 0L
    }

    @Synchronized
    fun inFlightAssemblyCount(): Int =
        assemblies.size

    @Synchronized
    fun inFlightBytes(): Int =
        retainedPayloadBytes
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()

    private fun ensureCapacity(
        additionalBytes: Int,
        preserve: Key,
    ) {
        require(additionalBytes > 0)
        require(
            additionalBytes <=
                config.maximumInFlightBytes,
        )

        while (
            assemblies.isNotEmpty() &&
            (
                retainedPayloadBytes +
                    additionalBytes.toLong() >
                    config.maximumInFlightBytes.toLong() ||
                assemblies.size >
                    config.maximumInFlightAssemblies
            )
        ) {
            val victim =
                assemblies.entries
                    .firstOrNull {
                        it.key != preserve
                    }
                    ?.key
                    ?: break

            val removed =
                assemblies.remove(victim)

            if (removed != null) {
                retainedPayloadBytes =
                    (
                        retainedPayloadBytes -
                            removed.chunks.values.sumOf {
                                it.payload.size.toLong()
                            }
                    ).coerceAtLeast(0L)
            }
        }

        require(
            retainedPayloadBytes +
                additionalBytes.toLong() <=
                config.maximumInFlightBytes.toLong(),
        ) {
            "Fragment reassembly byte budget exhausted"
        }
    }

    private fun createAssembly(
        key: Key,
        chunk: DenseLandmarkChunkedWireCodec.Chunk,
        nowNanos: Long,
    ): Assembly {
        if (
            assemblies.size >=
                config.maximumInFlightAssemblies
        ) {
            val oldestKey =
                assemblies.entries
                    .firstOrNull()
                    ?.key
            if (oldestKey != null) {
                val removed =
                    assemblies.remove(oldestKey)
                if (removed != null) {
                    retainedPayloadBytes =
                        (
                            retainedPayloadBytes -
                                removed.chunks.values.sumOf {
                                    it.payload.size.toLong()
                                }
                        ).coerceAtLeast(0L)
                }
            }
        }

        return Assembly(
            chunkCount = chunk.chunkCount,
            timestampNanos = chunk.timestampNanos,
            firstSeenNanos = nowNanos,
            chunks = LinkedHashMap(),
        ).also {
            assemblies[key] = it
        }
    }
}
