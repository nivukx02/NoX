package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NearbyPeerCapacityPolicyTest {

    @Test
    fun capacityAcceptsUpToConfiguredDistributedEnvelope() {
        val policy =
            NearbyPeerCapacityPolicy(
                NearbyPeerCapacityPolicy.Config(
                    maximumPeers = 128,
                    maximumConcurrentConnections = 16,
                ),
            )

        repeat(128) { index ->
            assertTrue(
                policy.tryReserveConnected(
                    "endpoint-$index",
                ),
            )
        }

        assertFalse(
            policy.tryReserveConnected(
                "endpoint-overflow",
            ),
        )
    }

    @Test
    fun duplicateReservationsAreIdempotent() {
        val policy =
            NearbyPeerCapacityPolicy()

        assertTrue(
            policy.tryReserveConnected("endpoint-a"),
        )
        assertTrue(
            policy.tryReserveConnected("endpoint-a"),
        )

        assertTrue(
            policy.connectedCount() == 1,
        )
    }

    @Test
    fun connectionAttemptsAreBoundedAndReleased() {
        val policy =
            NearbyPeerCapacityPolicy(
                NearbyPeerCapacityPolicy.Config(
                    maximumPeers = 4,
                    maximumConcurrentConnections = 2,
                ),
            )

        assertTrue(policy.tryReserveConnecting("a"))
        assertTrue(policy.tryReserveConnecting("b"))
        assertFalse(policy.tryReserveConnecting("c"))

        policy.releaseConnecting("a")

        assertTrue(policy.tryReserveConnecting("c"))
    }
}
