package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId

/**
 * Bounded registry of return routes for fragmented evidence arriving through the distributed relay
 * fabric. It is intentionally independent from fragment payload state so a partial frame can expire
 * without retaining the full payload graph.
 */
class FragmentReplyRouteRegistry(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumEntries: Int = 256,
        val maximumAgeNanos: Long = 750_000_000L,
        val maximumRouteHops: Int = 12,
    ) {
        init {
            require(maximumEntries in 1..4096)
            require(maximumAgeNanos > 0L)
            require(maximumRouteHops in 1..12)
        }
    }

    private data class Key(
        val endpoint: String,
        val evidenceType: NoXWireProtocol.Type,
        val streamId: String,
        val frameSequence: Long,
    )

    private data class Entry(
        val route: List<DeviceId>,
        val expiresAtNanos: Long,
    )

    private val entries =
        LinkedHashMap<Key, Entry>(
            32,
            0.75f,
            true,
        )

    @Synchronized
    fun remember(
        endpoint: String,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
        route: List<DeviceId>,
        nowNanos: Long,
    ) {
        require(endpoint.isNotBlank())
        require(streamId.isNotBlank())
        require(frameSequence >= 0L)
        require(route.isNotEmpty())
        require(route.size <= config.maximumRouteHops)
        require(route.distinct().size == route.size)

        expire(nowNanos)

        val key =
            Key(
                endpoint = endpoint,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
            )

        if (
            entries.size >=
                config.maximumEntries &&
            !entries.containsKey(key)
        ) {
            entries.entries.firstOrNull()
                ?.key
                ?.let(entries::remove)
        }

        entries[key] =
            Entry(
                route = route.toList(),
                expiresAtNanos =
                    nowNanos +
                        config.maximumAgeNanos,
            )
    }

    @Synchronized
    fun find(
        endpoint: String,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
        nowNanos: Long,
    ): List<DeviceId>? {
        require(endpoint.isNotBlank())
        require(streamId.isNotBlank())
        require(frameSequence >= 0L)

        expire(nowNanos)

        return entries[
            Key(
                endpoint = endpoint,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
            )
        ]?.route
    }

    @Synchronized
    fun remove(
        endpoint: String,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
    ) {
        entries.remove(
            Key(
                endpoint = endpoint,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
            ),
        )
    }

    @Synchronized
    fun expire(
        nowNanos: Long,
    ) {
        require(nowNanos >= 0L)

        val iterator =
            entries.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                nowNanos >=
                    entry.value.expiresAtNanos
            ) {
                iterator.remove()
            }
        }
    }

    @Synchronized
    fun clearEndpoint(
        endpoint: String,
    ) {
        require(endpoint.isNotBlank())
        entries.entries.removeIf {
            it.key.endpoint == endpoint
        }
    }

    @Synchronized
    fun clear() {
        entries.clear()
    }

    @Synchronized
    fun size(): Int =
        entries.size
}
