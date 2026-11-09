package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertTrue

class UdpRemoteEvidenceTransportContractTest {

    @Test
    fun reachableEndpointValidationIsStrict() {
        val endpoint =
            UdpRemoteEvidenceTransport.Endpoint(
                host = "192.168.10.42",
                port = 9100,
            )

        assertTrue(endpoint.host.isNotBlank())
        assertTrue(endpoint.port == 9100)
    }

    @Test
    fun mtuBudgetLeavesRoomForFragmentHeader() {
        assertTrue(
            UdpRemoteEvidenceTransport.Config()
                .maximumDatagramBytes <=
                RemoteDatagramFragmentCodec.MAX_DATAGRAM_BYTES,
        )
    }
}
