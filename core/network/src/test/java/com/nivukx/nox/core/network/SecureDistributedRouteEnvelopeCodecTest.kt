package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SecureDistributedRouteEnvelopeCodecTest {

    @Test
    fun secureRouteRoundTrips() {
        val key =
            ByteArray(32) { index ->
                (index * 13 + 5).toByte()
            }

        val inner =
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAME,
                sequence = 77L,
                timestampNanos = 30_000L,
                payload = byteArrayOf(4, 5, 6),
            )

        val packet =
            SecureDistributedRouteEnvelopeCodec.encode(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                route =
                    listOf(
                        DeviceId("relay-a"),
                        DeviceId("master"),
                    ),
                packetSequence = 77L,
                timestampNanos = 30_000L,
                payload = inner,
                key = key,
            )

        val decoded =
            SecureDistributedRouteEnvelopeCodec.decode(
                packet,
                key,
            )

        assertEquals(DeviceId("cam-a"), decoded.origin)
        assertEquals(DeviceId("master"), decoded.destination)
        assertEquals(2, decoded.route.size)
        assertEquals(2, decoded.ttl)
        assertTrue(inner.contentEquals(decoded.payload))
    }
}
