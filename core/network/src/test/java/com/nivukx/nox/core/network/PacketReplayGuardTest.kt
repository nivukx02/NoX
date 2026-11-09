package com.nivukx.nox.core.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PacketReplayGuardTest {

    @Test
    fun duplicatePacketIsRejectedPerPeer() {
        val guard = PacketReplayGuard()
        val packet = NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.START,
            sequence = 10L,
            timestampNanos = 100L,
        )

        assertTrue(guard.accept("peer-a", packet))
        assertFalse(guard.accept("peer-a", packet))

        val diagnostics = guard.diagnostics()
        assertEquals(1L, diagnostics.accepted)
        assertEquals(1L, diagnostics.rejected)
    }

    @Test
    fun samePacketCanBeAcceptedFromDifferentPeers() {
        val guard = PacketReplayGuard()
        val packet = NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.HEARTBEAT,
            sequence = 12L,
            timestampNanos = 200L,
        )

        assertTrue(guard.accept("peer-a", packet))
        assertTrue(guard.accept("peer-b", packet))
    }

    @Test
    fun boundedWindowEvictsOldFingerprints() {
        val guard = PacketReplayGuard(capacityPerPeer = 32)
        val first = NoXWireProtocol.encode(
            NoXWireProtocol.Type.PING,
            sequence = 1L,
            timestampNanos = 1L,
        )

        assertTrue(guard.accept("peer-a", first))

        for (sequence in 2L..40L) {
            val packet = NoXWireProtocol.encode(
                NoXWireProtocol.Type.PING,
                sequence = sequence,
                timestampNanos = sequence,
            )
            assertTrue(guard.accept("peer-a", packet))
        }

        assertTrue(
            guard.accept(
                "peer-a",
                first,
            ),
        )
    }

    @Test
    fun clearingPeerStartsFreshSessionWindow() {
        val guard = PacketReplayGuard()
        val packet = NoXWireProtocol.encode(
            NoXWireProtocol.Type.ARM,
            sequence = 5L,
            timestampNanos = 50L,
        )

        assertTrue(guard.accept("peer-a", packet))
        assertFalse(guard.accept("peer-a", packet))

        guard.clear("peer-a")

        assertTrue(guard.accept("peer-a", packet))
    }

    @Test
    fun emptyPacketDoesNotCreateReplayState() {
        val guard = PacketReplayGuard()

        assertTrue(guard.accept("peer-a", ByteArray(0)))
        assertEquals(0, guard.diagnostics().peers)
    }

    @Test
    fun peerTableIsBoundedAcrossLongLivedEndpointChurn() {
        val guard = PacketReplayGuard(
            capacityPerPeer = 32,
            maximumPeers = 2,
        )

        val a = NoXWireProtocol.encode(
            NoXWireProtocol.Type.PING,
            sequence = 1L,
            timestampNanos = 1L,
        )
        val b = NoXWireProtocol.encode(
            NoXWireProtocol.Type.PING,
            sequence = 2L,
            timestampNanos = 2L,
        )
        val c = NoXWireProtocol.encode(
            NoXWireProtocol.Type.PING,
            sequence = 3L,
            timestampNanos = 3L,
        )

        assertTrue(guard.accept("peer-a", a))
        assertTrue(guard.accept("peer-b", b))
        assertTrue(guard.accept("peer-c", c))

        assertEquals(2, guard.diagnostics().peers)

        assertTrue(
            guard.accept(
                "peer-a",
                a,
            ),
        )
    }

}
