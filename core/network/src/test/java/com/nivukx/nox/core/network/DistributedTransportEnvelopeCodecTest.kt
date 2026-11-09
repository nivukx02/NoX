package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedTransportEnvelopeCodecTest {

    @Test
    fun routedEnvelopeRoundTripsWithOrderedRoute() {
        val inner =
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.POSE_FRAME,
                sequence = 10L,
                timestampNanos = 20_000L,
                payload = byteArrayOf(1, 2, 3, 4),
            )

        val encoded =
            DistributedTransportEnvelopeCodec.encode(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                route =
                    listOf(
                        DeviceId("relay-a"),
                        DeviceId("master"),
                    ),
                hopIndex = 0,
                ttl = 2,
                packetSequence = 99L,
                timestampNanos = 20_000L,
                payload = inner,
            )

        val decoded =
            DistributedTransportEnvelopeCodec.decode(
                NoXWireProtocol.decode(encoded),
            )

        assertEquals(DeviceId("cam-a"), decoded.origin)
        assertEquals(DeviceId("master"), decoded.destination)
        assertEquals(2, decoded.route.size)
        assertEquals(DeviceId("relay-a"), decoded.route[0])
        assertEquals(DeviceId("master"), decoded.route[1])
        assertEquals(0, decoded.hopIndex)
        assertEquals(2, decoded.ttl)
        assertTrue(inner.contentEquals(decoded.payload))
    }

    @Test
    fun onlyTheExpectedNextHopCanAdvanceEnvelope() {
        val envelope =
            DistributedTransportEnvelopeCodec.Envelope(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                route =
                    listOf(
                        DeviceId("relay-a"),
                        DeviceId("master"),
                    ),
                hopIndex = 0,
                ttl = 2,
                packetSequence = 99L,
                timestampNanos = 20_000L,
                payload = byteArrayOf(8),
            )

        assertTrue(
            DistributedTransportEnvelopeCodec
                .canAdvance(
                    envelope,
                    DeviceId("relay-a"),
                ),
        )
        assertTrue(
            !DistributedTransportEnvelopeCodec
                .canAdvance(
                    envelope,
                    DeviceId("relay-b"),
                ),
        )
    }

    @Test
    fun destinationStopsForwarding() {
        val envelope =
            DistributedTransportEnvelopeCodec.Envelope(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                route =
                    listOf(
                        DeviceId("relay-a"),
                        DeviceId("master"),
                    ),
                hopIndex = 1,
                ttl = 1,
                packetSequence = 99L,
                timestampNanos = 20_000L,
                payload = byteArrayOf(8),
            )

        assertTrue(
            !DistributedTransportEnvelopeCodec
                .canAdvance(
                    envelope,
                    DeviceId("master"),
                ),
        )
    }
    @Test
    fun reconstructionCriticalFlagSurvivesEncodeDecode() {
        val inner =
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                sequence = 100L,
                timestampNanos = 30_000L,
                payload = byteArrayOf(1, 2, 3),
            )

        val encoded =
            DistributedTransportEnvelopeCodec.encode(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                route =
                    listOf(
                        DeviceId("relay-a"),
                        DeviceId("master"),
                    ),
                hopIndex = 0,
                ttl = 2,
                packetSequence = 101L,
                timestampNanos = 30_000L,
                payload = inner,
                reconstructionCritical = true,
            )

        val decoded =
            DistributedTransportEnvelopeCodec.decode(
                NoXWireProtocol.decode(encoded),
            )

        assertTrue(decoded.reconstructionCritical)
    }

    @Test
    fun legacyDefaultFlagRemainsNonCritical() {
        val envelope =
            DistributedTransportEnvelopeCodec.Envelope(
                origin = DeviceId("cam-a"),
                destination = DeviceId("master"),
                route =
                    listOf(
                        DeviceId("relay-a"),
                        DeviceId("master"),
                    ),
                hopIndex = 0,
                ttl = 2,
                packetSequence = 102L,
                timestampNanos = 31_000L,
                payload = byteArrayOf(1),
            )

        assertTrue(!envelope.reconstructionCritical)
    }


}
