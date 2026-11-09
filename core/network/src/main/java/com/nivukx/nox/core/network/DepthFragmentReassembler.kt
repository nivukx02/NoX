package com.nivukx.nox.core.network

class DepthFragmentReassembler(
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
        fun missingChunkMask(): Int =
            ((1 shl chunkCount) - 1) xor receivedMask
    }

    data class Config(
        val maximumInFlightAssemblies: Int = 16,
        val maximumInFlightBytes: Int = 16 * 1024 * 1024,
        val maximumAssemblyAgeNanos: Long = 250_000_000L,
    ) {
        init {
            require(maximumInFlightAssemblies in 1..128)
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
        val deviceId: com.nivukx.nox.core.model.DeviceId,
        val chunkCount: Int,
        val timestampNanos: Long,
        val firstSeenNanos: Long,
        val chunks: MutableMap<Int, DepthChunkedWireCodec.Chunk>,
    )

    private val assemblies =
        LinkedHashMap<Key, Assembly>(
            16,
            0.75f,
            true,
        )
    private var retainedBytes = 0L

    @Synchronized
    fun accept(
        endpoint: String,
        chunk: DepthChunkedWireCodec.Chunk,
        nowNanos: Long,
    ): ByteArray? {
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

        // A stream/frame key may not mix origins or frame metadata. A poisoned assembly is
        // discarded before throwing so malformed fragments cannot pin its in-flight byte budget.
        if (
            assembly.deviceId != chunk.deviceId ||
            assembly.chunkCount != chunk.chunkCount ||
            assembly.timestampNanos != chunk.timestampNanos
        ) {
            removeAssembly(key)
            throw IllegalArgumentException("Inconsistent depth fragment assembly metadata")
        }

        val existing =
            assembly.chunks[chunk.chunkIndex]

        if (existing != null) {
            if (
                !existing.payload.contentEquals(chunk.payload) ||
                existing.crc32 != chunk.crc32 ||
                existing.deviceId != chunk.deviceId
            ) {
                removeAssembly(key)
                throw IllegalArgumentException("Conflicting duplicate depth fragment")
            }
            return null
        }

        ensureCapacity(
            additionalBytes = chunk.payload.size,
            preserve = key,
        )

        assembly.chunks[chunk.chunkIndex] = chunk
        retainedBytes += chunk.payload.size.toLong()

        if (assembly.chunks.size != assembly.chunkCount) {
            return null
        }

        val ordered =
            assembly.chunks.values.sortedBy { it.chunkIndex }

        val payload =
            try {
                DepthChunkedWireCodec.decode(ordered).also { decoded ->
                    require(
                        DepthChunkedWireCodec.timestampFromPayload(decoded) ==
                            assembly.timestampNanos,
                    ) {
                        "Depth assembly payload timestamp mismatch"
                    }
                }
            } catch (failure: Throwable) {
                // CRC, origin, structure and timestamp failures must release retained memory.
                removeAssembly(key)
                throw failure
            }

        removeAssembly(key)
        return payload
    }

    @Synchronized
    fun expire(nowNanos: Long) {
        expireWithFeedback(nowNanos)
    }

    @Synchronized
    fun expireWithFeedback(
        nowNanos: Long,
    ): List<FeedbackState> {
        require(nowNanos >= 0L)

        val expired = ArrayList<FeedbackState>()
        val iterator = assemblies.entries.iterator()

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
                        timestampNanos = entry.value.timestampNanos,
                        chunkCount = entry.value.chunkCount,
                        receivedMask = mask,
                    )

                retainedBytes =
                    (
                        retainedBytes -
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
        val key =
            Key(
                endpoint = endpoint,
                streamId = streamId,
                frameSequence = frameSequence,
            )

        val assembly = assemblies[key] ?: return null
        val mask =
            assembly.chunks.keys.fold(0) { acc, index ->
                acc or (1 shl index)
            }

        return FeedbackState(
            endpoint = endpoint,
            streamId = streamId,
            frameSequence = frameSequence,
            timestampNanos = assembly.timestampNanos,
            chunkCount = assembly.chunkCount,
            receivedMask = mask,
        )
    }

    @Synchronized
    fun clearEndpoint(endpoint: String) {
        require(endpoint.isNotBlank())

        val iterator = assemblies.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key.endpoint == endpoint) {
                retainedBytes =
                    (
                        retainedBytes -
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
        retainedBytes = 0L
    }

    @Synchronized
    fun inFlightAssemblyCount(): Int =
        assemblies.size

    @Synchronized
    fun inFlightBytes(): Int =
        retainedBytes
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()

    private fun createAssembly(
        key: Key,
        chunk: DepthChunkedWireCodec.Chunk,
        nowNanos: Long,
    ): Assembly {
        if (
            assemblies.size >=
                config.maximumInFlightAssemblies
        ) {
            val victim =
                assemblies.entries.firstOrNull()?.key
            if (victim != null) {
                removeAssembly(victim)
            }
        }

        return Assembly(
            deviceId = chunk.deviceId,
            chunkCount = chunk.chunkCount,
            timestampNanos = chunk.timestampNanos,
            firstSeenNanos = nowNanos,
            chunks = LinkedHashMap(),
        ).also {
            assemblies[key] = it
        }
    }

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
            retainedBytes +
                additionalBytes.toLong() >
                config.maximumInFlightBytes
        ) {
            val victim =
                assemblies.entries
                    .firstOrNull {
                        it.key != preserve
                    }
                    ?.key
                    ?: break

            removeAssembly(victim)
        }

        require(
            retainedBytes +
                additionalBytes.toLong() <=
                config.maximumInFlightBytes,
        ) {
            "Depth fragment memory budget exhausted"
        }
    }

    private fun removeAssembly(key: Key) {
        val removed = assemblies.remove(key) ?: return
        retainedBytes =
            (
                retainedBytes -
                    removed.chunks.values.sumOf {
                        it.payload.size.toLong()
                    }
            ).coerceAtLeast(0L)
    }
}
