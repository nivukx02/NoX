package com.nivukx.nox.core.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierDistributedTransportOrchestratorTest {
    @Test
    fun autoTopologyUsesClusterForLargeFederationsAndBudgetStaysBounded() {
        val orchestrator = FrontierDistributedTransportOrchestrator()
        val channels = (0 until 128).map { id ->
            FrontierCameraTransportBudgetEngine.CameraChannel(
                cameraId = "c$id",
                authoritative = id < 32,
                quality = 0.95,
                packetLoss = 0.01,
                latencyMillis = 8.0,
                requestedKbps = 8_000.0,
            )
        }
        val result = orchestrator.evaluate(
            channels = channels,
            linkCapacityKbps = 120_000.0,
            timestampNanos = 1L,
        )
        assertEquals(NearbyTransportTopologySelector.Mode.CLUSTER, result.topology)
        assertTrue(result.budget.totalAllocatedKbps <= 120_000.0 + 1e-6)
        assertEquals(128, result.budget.allocations.size)
    }

    @Test
    fun closeIsIdempotentAndRejectsFurtherPeerQualityAccess() {
        val orchestrator = FrontierDistributedTransportOrchestrator()
        orchestrator.observePingSent(
            endpoint = "peer-1",
            sequence = 1L,
            timestampNanos = 1_000L,
        )
        orchestrator.observePong(
            endpoint = "peer-1",
            sequence = 1L,
            timestampNanos = 2_000L,
        )

        orchestrator.close()
        orchestrator.close()

        assertThrows(IllegalStateException::class.java) {
            orchestrator.peerQuality(
                endpoint = "peer-1",
                nowNanos = 3_000L,
            )
        }
    }

    @Test
    fun closeRejectsNewPingAndPayloadSamples() {
        val orchestrator = FrontierDistributedTransportOrchestrator()
        orchestrator.close()

        assertThrows(IllegalStateException::class.java) {
            orchestrator.observePingSent(
                endpoint = "peer-2",
                sequence = 1L,
                timestampNanos = 1_000L,
            )
        }
        assertThrows(IllegalStateException::class.java) {
            orchestrator.observePayloadStarted(
                endpoint = "peer-2",
                payloadId = 1L,
                totalBytes = 100L,
                timestampNanos = 1_000L,
            )
        }
    }

    @Test
    fun peerQualityCanBeObservedThroughUnifiedControlPlane() {
        val orchestrator = FrontierDistributedTransportOrchestrator()
        orchestrator.observePingSent("peer", 1L, 1_000L)
        orchestrator.observePong("peer", 1L, 2_000L)
        val quality = orchestrator.peerQuality("peer", 2_000L)
        assertTrue(quality.transportScore >= 0.0)
    }
}
