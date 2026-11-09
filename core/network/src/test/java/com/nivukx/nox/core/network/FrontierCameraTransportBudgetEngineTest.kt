package com.nivukx.nox.core.network

import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierCameraTransportBudgetEngineTest {
    @Test
    fun totalBandwidthNeverExceedsLinkCapacity() {
        val channels = (0 until 128).map { id ->
            FrontierCameraTransportBudgetEngine.CameraChannel(
                cameraId = "c$id",
                authoritative = id < 32,
                quality = 0.90,
                packetLoss = 0.01,
                latencyMillis = 10.0,
                requestedKbps = 10_000.0,
                actorImportance = (0 until 50).associateWith { 1.0 },
            )
        }
        val result = FrontierCameraTransportBudgetEngine().allocate(
            channels,
            linkCapacityKbps = 80_000.0,
            timestampNanos = 1L,
        )
        assertTrue(result.totalAllocatedKbps <= 80_000.0 + 1e-6)
        assertTrue(result.totalQueueBytes <= 32L * 1024L * 1024L)
    }

    @Test
    fun brokenTransportIsQuarantined() {
        val result = FrontierCameraTransportBudgetEngine().allocate(
            listOf(FrontierCameraTransportBudgetEngine.CameraChannel("bad",true,0.2,0.50,250.0,10_000.0)),
            linkCapacityKbps = 2_000.0,
            timestampNanos = 2L,
        )
        assertTrue(result.allocations.single().tier == FrontierCameraTransportBudgetEngine.TransportTier.QUARANTINED)
    }
}


class FrontierCameraTransportBudgetEvidenceClassTest {
    @Test
    fun evidenceClassReservationsConserveAllocatedBandwidth() {
        val engine = FrontierCameraTransportBudgetEngine()
        val result = engine.allocate(
            channels = listOf(
                FrontierCameraTransportBudgetEngine.CameraChannel(
                    cameraId = "authoritative-a",
                    authoritative = true,
                    quality = 0.98,
                    packetLoss = 0.005,
                    latencyMillis = 6.0,
                    requestedKbps = 18_000.0,
                    actorImportance = (0 until 50).associateWith { 1.0 },
                ),
                FrontierCameraTransportBudgetEngine.CameraChannel(
                    cameraId = "witness-b",
                    authoritative = false,
                    quality = 0.92,
                    packetLoss = 0.01,
                    latencyMillis = 10.0,
                    requestedKbps = 2_000.0,
                    actorImportance = mapOf(4 to 1.0, 11 to 0.9),
                ),
            ),
            linkCapacityKbps = 20_000.0,
            timestampNanos = 1_000L,
        )

        result.allocations.forEach { allocation ->
            assertTrue(
                allocation.timingKbps +
                    allocation.identityKbps +
                    allocation.geometryKbps +
                    allocation.surfaceKbps +
                    allocation.diagnosticsKbps <=
                    allocation.allocatedKbps + 1e-6,
            )
            assertTrue(allocation.timingKbps >= 0.0)
            assertTrue(allocation.identityKbps >= 0.0)
            assertTrue(allocation.geometryKbps >= 0.0)
            assertTrue(allocation.surfaceKbps >= 0.0)
            assertTrue(allocation.diagnosticsKbps >= 0.0)
        }

        assertTrue(result.totalAllocatedKbps <= 20_000.0 + 1e-6)
        assertTrue(result.totalQueueBytes <= 32L * 1024L * 1024L)
    }

    @Test
    fun witnessCameraRetainsProtectedTimingAndIdentityBudgetsUnderPressure() {
        val result = FrontierCameraTransportBudgetEngine().allocate(
            channels = listOf(
                FrontierCameraTransportBudgetEngine.CameraChannel(
                    cameraId = "witness",
                    authoritative = false,
                    quality = 0.95,
                    packetLoss = 0.005,
                    latencyMillis = 8.0,
                    requestedKbps = 1_000.0,
                    actorImportance = mapOf(7 to 1.0),
                ),
            ),
            linkCapacityKbps = 500.0,
            timestampNanos = 2_000L,
        )

        val allocation = result.allocations.single()
        assertTrue(allocation.allocatedKbps > 0.0)
        assertTrue(
            allocation.timingKbps +
                allocation.identityKbps >
                0.0,
        )
    }
}
