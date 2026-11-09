package com.nivukx.nox.core.network

/**
 * Thread-safe connection-state budget for distributed Nearby sessions.
 *
 * The capture architecture supports a 128-camera global envelope. This policy prevents transport
 * bookkeeping from exceeding that envelope and separately bounds concurrent handshake pressure.
 */
class NearbyPeerCapacityPolicy(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumPeers: Int = 128,
        val maximumConcurrentConnections: Int = 16,
    ) {
        init {
            require(maximumPeers in 1..256)
            require(
                maximumConcurrentConnections in
                    1..maximumPeers,
            )
        }
    }

    private val connected =
        LinkedHashSet<String>()
    private val connecting =
        LinkedHashSet<String>()

    @Synchronized
    fun tryReserveConnected(
        endpoint: String,
    ): Boolean {
        require(endpoint.isNotBlank())

        if (connected.contains(endpoint)) {
            return true
        }

        if (
            connected.size >=
                config.maximumPeers
        ) {
            return false
        }

        connected += endpoint
        connecting.remove(endpoint)
        return true
    }

    @Synchronized
    fun tryReserveConnecting(
        endpoint: String,
    ): Boolean {
        require(endpoint.isNotBlank())

        if (
            connected.contains(endpoint) ||
            connecting.contains(endpoint)
        ) {
            return true
        }

        if (
            connecting.size >=
                config.maximumConcurrentConnections
        ) {
            return false
        }

        connecting += endpoint
        return true
    }

    @Synchronized
    fun releaseConnecting(
        endpoint: String,
    ) {
        require(endpoint.isNotBlank())
        connecting.remove(endpoint)
    }

    @Synchronized
    fun release(
        endpoint: String,
    ) {
        require(endpoint.isNotBlank())
        connected.remove(endpoint)
        connecting.remove(endpoint)
    }

    @Synchronized
    fun clear() {
        connected.clear()
        connecting.clear()
    }

    @Synchronized
    fun connectedCount(): Int =
        connected.size

    @Synchronized
    fun connectingCount(): Int =
        connecting.size
}
