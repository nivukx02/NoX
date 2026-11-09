package com.nivukx.nox.core.network

/**
 * Deterministic topology selector for distributed Nearby capture sessions.
 *
 * STAR preserves the strongest per-link bandwidth for small direct groups. CLUSTER is selected for
 * large federations because it permits M-to-N connections and therefore supports NoX routed
 * evidence without forcing every node through one hub.
 */
object NearbyTransportTopologySelector {

    enum class Mode {
        STAR,
        CLUSTER,
        AUTO,
    }

    data class Config(
        val autoClusterThreshold: Int = 32,
    ) {
        init {
            require(autoClusterThreshold in 2..128)
        }
    }

    fun resolve(
        mode: Mode,
        expectedPeerCount: Int,
        config: Config = Config(),
    ): Mode {
        require(expectedPeerCount >= 1)

        return when (mode) {
            Mode.STAR -> Mode.STAR
            Mode.CLUSTER -> Mode.CLUSTER
            Mode.AUTO ->
                if (
                    expectedPeerCount >=
                        config.autoClusterThreshold
                ) {
                    Mode.CLUSTER
                } else {
                    Mode.STAR
                }
        }
    }
}
