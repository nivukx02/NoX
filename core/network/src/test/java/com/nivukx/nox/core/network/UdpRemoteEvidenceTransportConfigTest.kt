package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UdpRemoteEvidenceTransportConfigTest {

    @Test
    fun federated_decode_defaults_are_bounded() {
        val config =
            UdpRemoteEvidenceTransport.Config()

        assertTrue(config.maximumDecodeWorkers in 1..16)
        assertTrue(config.maximumDecodeQueue in 64..16_384)
        assertEquals(
            minOf(
                8,
                Runtime.getRuntime()
                    .availableProcessors()
                    .coerceAtLeast(1),
            ),
            config.maximumDecodeWorkers,
        )
    }

    @Test
    fun custom_decode_budget_never_changes_datagram_safety_contract() {
        val config =
            UdpRemoteEvidenceTransport.Config(
                maximumDecodeWorkers = 4,
                maximumDecodeQueue = 2048,
                maximumDatagramBytes = 1200,
            )

        assertEquals(4, config.maximumDecodeWorkers)
        assertEquals(2048, config.maximumDecodeQueue)
        assertTrue(
            config.maximumDatagramBytes <=
                RemoteDatagramFragmentCodec.MAX_DATAGRAM_BYTES,
        )
    }
}
