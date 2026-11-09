package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.FaceMeshFrame

class FaceMeshFragmentReassembler(
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
        val chunkCount: Int,
        val timestampNanos: Long,
        val firstSeenNanos: Long,
        val chunks:
            MutableMap<Int, FaceMeshChunkedWireCodec.Chunk>,
    )

    private val assemblies =
        LinkedHashMap<Key, Assembly>(
            16,
            0.75f,
            true,
        )

    @Synchronized
    fun accept(
        endpoint: String,
        chunk: FaceMeshChunkedWireCodec.Chunk,
        nowNanos: Long,
    ): FaceMeshFrame? {
        require(endpoint.isNotBlank())
        require(nowNanos >= 0L)

        expire(nowNanos)

        val key =
            Key(
                endpoint,
                chunk.streamId,
                chunk.frameSequence,
            )

        val assembly =
            assemblies[key]
                ?: createAssembly(
                    key,
                    chunk,
                    nowNanos,
                )

        if (
            assembly.chunkCount != chunk.chunkCount ||
            assembly.timestampNanos != chunk.timestampNanos
        ) {
            // Drop impossible partial state before the existing checks reject the new fragment.
            assemblies.remove(key)
        }

        require(
            assembly.chunkCount ==
                chunk.chunkCount,
        )
        require(
            assembly.timestampNanos ==
                chunk.timestampNanos,
        )

        val previous =
            assembly.chunks[chunk.chunkIndex]

        if (previous != null) {
            if (
                !previous.payload.contentEquals(chunk.payload) ||
                previous.crc32 != chunk.crc32
            ) {
                assemblies.remove(key)
            }
            require(
                previous.payload.contentEquals(
                    chunk.payload,
                ),
            )
            require(previous.crc32 == chunk.crc32) {
                "Conflicting duplicate face mesh fragment CRC"
            }
            return null
        }

        ensureCapacity(
            additionalBytes = chunk.payload.size,
            preserve = key,
        )
        assembly.chunks[chunk.chunkIndex] =
            chunk

        if (
            assembly.chunks.size !=
                assembly.chunkCount
        ) {
            return null
        }

        val frame =
            try {
                FaceMeshChunkedWireCodec.decode(
                    assembly.chunks.values.sortedBy { it.chunkIndex },
                )
            } catch (failure: Throwable) {
                assemblies.remove(key)
                throw failure
            }

        assemblies.remove(key)
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
                val receivedMask =
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
                        receivedMask =
                            receivedMask,
                    )

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
                endpoint,
                streamId,
                frameSequence,
            )
        val assembly =
            assemblies[key]
                ?: return null

        val receivedMask =
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
            receivedMask =
                receivedMask,
        )
    }

    @Synchronized
    fun clearEndpoint(endpoint: String) {
        require(endpoint.isNotBlank())
        assemblies.entries.removeIf {
            it.key.endpoint == endpoint
        }
    }

    @Synchronized
    fun clear() {
        assemblies.clear()
    }

    @Synchronized
    fun inFlightAssemblyCount(): Int =
        assemblies.size

    @Synchronized
    fun inFlightBytes(): Int =
        assemblies.values.sumOf { assembly ->
            assembly.chunks.values.sumOf { it.payload.size }
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

        fun totalBytes(): Int =
            assemblies.values.sumOf { assembly ->
                assembly.chunks.values.sumOf {
                    it.payload.size
                }
            }

        while (
            assemblies.isNotEmpty() &&
            (
                totalBytes() +
                    additionalBytes >
                    config.maximumInFlightBytes ||
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
            assemblies.remove(victim)
        }

        require(
            totalBytes() +
                additionalBytes <=
                config.maximumInFlightBytes,
        ) {
            "Fragment reassembly byte budget exhausted"
        }
    }

    private fun createAssembly(
        key: Key,
        chunk: FaceMeshChunkedWireCodec.Chunk,
        nowNanos: Long,
    ): Assembly {
        if (
            assemblies.size >=
                config.maximumInFlightAssemblies
        ) {
            assemblies.entries.firstOrNull()
                ?.key
                ?.let(assemblies::remove)
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
