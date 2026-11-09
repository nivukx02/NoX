package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals

class NearbyTransportTopologyModeTest {

    @Test
    fun distributedClusterModeUsesClusterStrategy() {
        assertEquals(
            NearbyTransportTopologySelector.Mode.CLUSTER,
            NearbyTransportTopologySelector.resolve(
                mode = NearbyTransportTopologySelector.Mode.CLUSTER,
                expectedPeerCount = 128,
            ),
        )
    }

    @Test
    fun directModePreservesHighBandwidthStarStrategy() {
        assertEquals(
            NearbyTransportTopologySelector.Mode.STAR,
            NearbyTransportTopologySelector.resolve(
                mode = NearbyTransportTopologySelector.Mode.STAR,
                expectedPeerCount = 128,
            ),
        )
    }

    @Test
    fun autoModeUsesStarForSmallSessions() {
        assertEquals(
            NearbyTransportTopologySelector.Mode.STAR,
            NearbyTransportTopologySelector.resolve(
                mode = NearbyTransportTopologySelector.Mode.AUTO,
                expectedPeerCount = 16,
            ),
        )
    }

    @Test
    fun autoModeUsesClusterForLargeFederations() {
        assertEquals(
            NearbyTransportTopologySelector.Mode.CLUSTER,
            NearbyTransportTopologySelector.resolve(
                mode = NearbyTransportTopologySelector.Mode.AUTO,
                expectedPeerCount = 128,
            ),
        )
    }
}
