package com.nivukx.nox.core.network

/**
 * Memory-bounded UDP fragment reassembler.
 *
 * Source is an authenticated transport endpoint such as "10.0.0.4:9100". Authentication happens
 * after full logical-message reassembly through SecureRemoteEvidenceCodec, while this layer enforces
 * strict sequence/count/size/expiry bounds before allocating large buffers.
 */
class RemoteDatagramReassembler(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumInFlightAssemblies: Int = 128,
        val maximumInFlightBytes: Int = 32 * 1024 * 1024,
        val maximumAssemblyAgeNanos: Long = 500_000_000L,
    ) {
        init {
            require(maximumInFlightAssemblies in 1..1024)
            require(maximumInFlightBytes in 64 * 1024..256 * 1024 * 1024)
            require(maximumAssemblyAgeNanos > 0L)
        }
    }

    private data class Key(
        val source: String,
        val sequence: Long,
    )

    private data class Assembly(
        val fragmentCount: Int,
        val totalPayloadBytes: Int,
        val firstSeenNanos: Long,
        val fragments: MutableMap<Int, RemoteDatagramFragmentCodec.Fragment>,
    )

    private val assemblies =
        LinkedHashMap<Key, Assembly>(
            32,
            0.75f,
            true,
        )

    private var retainedBytes = 0L

    @Synchronized
    fun accept(
        source: String,
        datagram: ByteArray,
        nowNanos: Long,
    ): ByteArray? {
        require(source.isNotBlank())
        require(nowNanos >= 0L)

        expire(nowNanos)

        val fragment =
            RemoteDatagramFragmentCodec.decode(
                datagram,
            )

        val key =
            Key(
                source = source,
                sequence =
                    fragment.messageSequence,
            )

        val assembly =
            assemblies[key]
                ?: createAssembly(
                    key = key,
                    fragment = fragment,
                    nowNanos = nowNanos,
                )

        require(
            assembly.fragmentCount ==
                fragment.fragmentCount,
        ) {
            "Remote evidence fragment count changed"
        }

        require(
            assembly.totalPayloadBytes ==
                fragment.totalPayloadBytes,
        )

        val existing =
            assembly.fragments[fragment.fragmentIndex]

        if (existing != null) {
            require(
                existing.payload.contentEquals(
                    fragment.payload,
                ),
            ) {
                "Conflicting duplicate remote evidence fragment"
            }
            return null
        }

        require(
            assembly.fragments.size <
                assembly.fragmentCount,
        )

        ensureCapacity(
            additionalBytes =
                fragment.payloadLength,
            preserve = key,
        )

        assembly.fragments[fragment.fragmentIndex] =
            fragment
        retainedBytes +=
            fragment.payloadLength.toLong()

        if (
            assembly.fragments.size !=
                assembly.fragmentCount
        ) {
            return null
        }

        val ordered =
            assembly.fragments.values
                .sortedBy {
                    it.fragmentIndex
                }

        val total =
            ordered.sumOf {
                it.payloadLength
            }

        require(total == assembly.totalPayloadBytes) {
            "Remote evidence total payload mismatch"
        }

        val output =
            ByteArray(total)

        var offset = 0
        ordered.forEach { part ->
            part.payload.copyInto(
                destination = output,
                destinationOffset = offset,
            )
            offset +=
                part.payloadLength
        }

        assemblies.remove(key)
        retainedBytes =
            (
                retainedBytes -
                    total.toLong()
            ).coerceAtLeast(0L)

        return output
    }

    @Synchronized
    fun expire(
        nowNanos: Long,
    ) {
        require(nowNanos >= 0L)

        val iterator =
            assemblies.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                nowNanos -
                    entry.value.firstSeenNanos >=
                    config.maximumAssemblyAgeNanos
            ) {
                retainedBytes =
                    (
                        retainedBytes -
                            entry.value.fragments.values.sumOf {
                                it.payloadLength.toLong()
                            }
                    ).coerceAtLeast(0L)
                iterator.remove()
            }
        }
    }

    @Synchronized
    fun clearSource(
        source: String,
    ) {
        require(source.isNotBlank())

        val iterator =
            assemblies.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key.source == source) {
                retainedBytes =
                    (
                        retainedBytes -
                            entry.value.fragments.values.sumOf {
                                it.payloadLength.toLong()
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
        fragment: RemoteDatagramFragmentCodec.Fragment,
        nowNanos: Long,
    ): Assembly {
        require(
            fragment.payloadLength <=
                config.maximumInFlightBytes,
        )

        if (
            assemblies.size >=
                config.maximumInFlightAssemblies
        ) {
            val victim =
                assemblies.entries
                    .firstOrNull()
                    ?.key

            if (victim != null) {
                val removed =
                    assemblies.remove(victim)
                if (removed != null) {
                    retainedBytes =
                        (
                            retainedBytes -
                                removed.fragments.values.sumOf {
                                    it.payloadLength.toLong()
                                }
                        ).coerceAtLeast(0L)
                }
            }
        }

        return Assembly(
            fragmentCount =
                fragment.fragmentCount,
            totalPayloadBytes =
                totalPayloadBytesFromFragment(
                    fragment,
                ),
            firstSeenNanos = nowNanos,
            fragments = LinkedHashMap(),
        ).also {
            assemblies[key] = it
        }
    }

    private fun ensureCapacity(
        additionalBytes: Int,
        preserve: Key,
    ) {
        require(additionalBytes > 0)

        while (
            assemblies.isNotEmpty() &&
            retainedBytes +
                additionalBytes.toLong() >
                config.maximumInFlightBytes
        ) {
            val victimEntry =
                assemblies.entries
                    .firstOrNull {
                        it.key != preserve
                    }
                    ?: break

            assemblies.remove(victimEntry.key)

            retainedBytes =
                (
                    retainedBytes -
                        victimEntry.value.fragments.values.sumOf {
                            it.payloadLength.toLong()
                        }
                ).coerceAtLeast(0L)
        }

        require(
            retainedBytes +
                additionalBytes.toLong() <=
                config.maximumInFlightBytes,
        ) {
            "Remote datagram memory budget exhausted"
        }
    }

    /*
     * The total logical message size is duplicated in every fragment header. Keeping the helper
     * here makes the expected-size invariant explicit and lets future transport versions change
     * fragmentation without changing reassembly semantics.
     */
    private fun totalPayloadBytesFromFragment(
        fragment: RemoteDatagramFragmentCodec.Fragment,
    ): Int =
        fragment.totalPayloadBytes
}
