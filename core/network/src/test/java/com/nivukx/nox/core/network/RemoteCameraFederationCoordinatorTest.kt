package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceCapabilities
import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RemoteCameraFederationCoordinatorTest {

    private fun node(
        id: Int,
        heartbeat: Long = 1_000L,
    ): RemoteCameraSessionRegistry.CameraNode =
        RemoteCameraSessionRegistry.CameraNode(
            cameraId = DeviceId("federated-" + id),
            endpoint =
                UdpRemoteEvidenceTransport.Endpoint(
                    host = "10.20.0." + ((id % 240) + 1),
                    port = 9_000 + id,
                ),
                capabilities =
                    DeviceCapabilities(
                        model = "federated-camera",
                        manufacturer = "test",
                        androidApi = 37,
                        cameraIds = listOf("0"),
                        supportsRaw = true,
                        supportsConcurrentCamera = true,
                        supportsGyroscope = true,
                        supportsAccelerometer = true,
                        maxAnalysisWidth = 3840,
                        maxAnalysisHeight = 2160,
                        maxFps = 120,
                        supportsHardwareReadoutTimestamp = true,
                        maximumVerifiedConcurrentCameras = 1,
                        minimumExposureTimeNanos = 100_000L,
                        maximumExposureTimeNanos = 16_000_000L,
                    ),
            timingAuthorityScore = 0.94,
            lastHeartbeatNanos = heartbeat,
        )

    @Test
    fun admits_and_bounds_full_128_camera_federation() {
        val federation =
            RemoteCameraFederationCoordinator()

        repeat(128) { index ->
            federation.admit(node(index))
        }

        assertEquals(128, federation.size())
        assertEquals(
            128,
            federation.snapshot().registry.cameraCount,
        )
        assertTrue(
            federation.snapshot().registry.healthyCameraCount == 128,
        )
    }

    @Test
    fun heartbeat_keeps_camera_alive_and_retirement_is_targeted() {
        val federation =
            RemoteCameraFederationCoordinator(
                RemoteCameraFederationCoordinator.Config(
                    maximumSilenceNanos = 100L,
                ),
            )

        federation.admit(node(0, heartbeat = 800L))
        federation.admit(node(1, heartbeat = 950L))
        assertTrue(
            federation.heartbeat(
                DeviceId("federated-0"),
                1_050L,
            ),
        )

        val retired =
            federation.retireStale(
                nowNanos = 1_100L,
            )

        assertEquals(
            listOf(DeviceId("federated-1")),
            retired,
        )
        assertEquals(1, federation.size())
    }

    @Test
    fun transport_budget_respects_authority_and_link_capacity() {
        val federation =
            RemoteCameraFederationCoordinator()

        repeat(16) { index ->
            federation.admit(
                node(
                    id = index,
                    heartbeat = 10_000L,
                ),
            )
        }

        val result =
            federation.allocateTransport(
                linkCapacityKbps = 12_000.0,
                authoritativeCameraIds =
                    (0 until 8).map {
                        DeviceId("federated-" + it)
                    }.toSet(),
                actorImportanceByCamera =
                    (0 until 16).associate { index ->
                        DeviceId("federated-" + index) to
                            mapOf(
                                0 to 1.0,
                                1 to 0.8,
                            )
                    },
                timestampNanos = 10_100L,
            )

        assertTrue(
            result.totalAllocatedKbps <= 12_000.0 + 1e-6,
        )
        assertEquals(16, result.allocations.size)
        assertTrue(
            result.allocations.any {
                it.tier ==
                    FrontierCameraTransportBudgetEngine
                        .TransportTier.FULL_EVIDENCE
            },
        )
    }

    @Test
    fun measuredPacketLossAndLatencyAffectFederationNetworkHealth() {
        val federation = RemoteCameraFederationCoordinator()
        federation.admit(
            node(0).copy(
                packetLossRatio = 0.0,
                latencyMillis = 2.0,
                requestedKbps = 3_500.0,
            ),
        )
        federation.admit(
            node(1).copy(
                packetLossRatio = 0.40,
                latencyMillis = 600.0,
                requestedKbps = 3_500.0,
            ),
        )

        val result = federation.allocateTransport(
            linkCapacityKbps = 8_000.0,
            authoritativeCameraIds = setOf(DeviceId("federated-0")),
            timestampNanos = 2_000L,
        )

        assertTrue(result.networkHealth < 0.80)
        assertTrue(result.warnings.contains("low-quality-camera-links-present"))
        assertTrue(
            !result.warnings.contains("remote-camera-network-telemetry-incomplete"),
        )
    }

    @Test
    fun missingNetworkTelemetryIsNotMisreportedAsPerfect() {
        val federation = RemoteCameraFederationCoordinator()
        federation.admit(node(0))
        federation.admit(node(1))

        val result = federation.allocateTransport(
            linkCapacityKbps = 8_000.0,
            authoritativeCameraIds = setOf(DeviceId("federated-0")),
            timestampNanos = 2_000L,
        )

        assertTrue(result.networkHealth < 0.80)
        assertTrue(
            result.warnings.contains("remote-camera-network-telemetry-incomplete"),
        )
    }

    @Test
    fun liveNetworkTelemetryRefreshesTransportInputsAndInvalidatesOldAllocation() {
        val federation = RemoteCameraFederationCoordinator()
        val firstId = DeviceId("federated-0")
        val secondId = DeviceId("federated-1")
        federation.admit(node(0))
        federation.admit(node(1))

        val initial = federation.allocateTransport(
            linkCapacityKbps = 8_000.0,
            authoritativeCameraIds = setOf(firstId),
            timestampNanos = 1_100L,
        )
        assertTrue(initial.networkHealth < 0.70)

        assertTrue(
            federation.recordNetworkTelemetry(
                cameraId = firstId,
                packetLossRatio = 0.0,
                latencyMillis = 2.0,
                requestedKbps = 3_500.0,
                timestampNanos = 1_200L,
            ),
        )
        assertTrue(
            federation.recordNetworkTelemetry(
                cameraId = secondId,
                packetLossRatio = 0.01,
                latencyMillis = 5.0,
                requestedKbps = 2_500.0,
                timestampNanos = 1_200L,
            ),
        )

        assertEquals(null, federation.snapshot().transport)
        assertEquals(0.0, federation.find(firstId)!!.packetLossRatio)
        assertEquals(2.0, federation.find(firstId)!!.latencyMillis)
        assertEquals(1_200L, federation.find(firstId)!!.lastHeartbeatNanos)

        val refreshed = federation.allocateTransport(
            linkCapacityKbps = 8_000.0,
            authoritativeCameraIds = setOf(firstId),
            timestampNanos = 1_250L,
        )
        assertTrue(refreshed.networkHealth > initial.networkHealth)
        assertTrue(
            !refreshed.warnings.contains(
                "remote-camera-network-telemetry-incomplete",
            ),
        )
    }

    @Test
    fun close_releases_registry_and_future_access_fails() {
        val federation =
            RemoteCameraFederationCoordinator()

        federation.admit(node(3))
        federation.close()

        assertEquals(
            0,
            federation.snapshot().registry.cameraCount,
        )
        assertTrue(
            federation.snapshot().retiredCameraIds.contains(
                DeviceId("federated-3"),
            ),
        )
    }
}
