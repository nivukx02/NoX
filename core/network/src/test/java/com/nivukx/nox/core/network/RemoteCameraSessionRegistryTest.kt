package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceCapabilities
import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RemoteCameraSessionRegistryTest {

    @Test
    fun registryBoundsRemoteCameraEnvelopeAndKeepsIdentityDistinctFromEndpoint() {
        val registry =
            RemoteCameraSessionRegistry(
                RemoteCameraSessionRegistry.Config(
                    maximumCameras = 128,
                ),
            )

        repeat(128) { index ->
            registry.register(
                RemoteCameraSessionRegistry.CameraNode(
                    cameraId = DeviceId("cam-${index}"),
                    endpoint =
                        UdpRemoteEvidenceTransport.Endpoint(
                            host = "10.0.0.${(index % 250) + 1}",
                            port = 9_100 + (index % 100),
                        ),
                    capabilities = capabilities(),
                    timingAuthorityScore = 0.90,
                ),
            )
        }

        assertEquals(128, registry.size())

        assertFailsWith<IllegalStateException> {
            registry.register(
                RemoteCameraSessionRegistry.CameraNode(
                    cameraId = DeviceId("cam-overflow"),
                    endpoint =
                        UdpRemoteEvidenceTransport.Endpoint(
                            host = "10.1.1.1",
                            port = 9100,
                        ),
                    capabilities = capabilities(),
                ),
            )
        }

        assertTrue(
            registry.findByEndpoint(
                UdpRemoteEvidenceTransport.Endpoint(
                    host = "10.0.0.1",
                    port = 9100,
                ),
            ) != null,
        )
    }

    @Test
    fun staleNodesCanBeExpiredWithoutAffectingOtherCameras() {
        val registry =
            RemoteCameraSessionRegistry()

        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = DeviceId("cam-a"),
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.5",
                        port = 9100,
                    ),
                capabilities = capabilities(),
                lastHeartbeatNanos = 100L,
            ),
        )

        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = DeviceId("cam-b"),
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.6",
                        port = 9100,
                    ),
                capabilities = capabilities(),
                lastHeartbeatNanos = 950L,
            ),
        )

        val expired =
            registry.expire(
                nowNanos = 1_000L,
                maximumSilenceNanos = 100L,
            )

        assertEquals(
            listOf(DeviceId("cam-a")),
            expired,
        )
        assertEquals(1, registry.size())
        assertEquals(
            DeviceId("cam-b"),
            registry.all().single().cameraId,
        )
    }


    @Test
    fun nodesThatNeverSendHeartbeatsExpireAfterGracePeriod() {
        val registry = RemoteCameraSessionRegistry()
        val cameraId = DeviceId("cam-never-heartbeat")
        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = cameraId,
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.77",
                        port = 9100,
                    ),
                capabilities = capabilities(),
            ),
        )

        assertTrue(
            registry.expire(
                nowNanos = 1_000L,
                maximumSilenceNanos = 100L,
            ).isEmpty(),
        )
        assertEquals(1, registry.size())

        assertEquals(
            listOf(cameraId),
            registry.expire(
                nowNanos = 1_100L,
                maximumSilenceNanos = 100L,
            ),
        )
        assertEquals(0, registry.size())
    }

    @Test
    fun heartbeatArrivalResetsTheMissingHeartbeatGraceClock() {
        val registry = RemoteCameraSessionRegistry()
        val cameraId = DeviceId("cam-delayed-heartbeat")
        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = cameraId,
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.78",
                        port = 9100,
                    ),
                capabilities = capabilities(),
            ),
        )

        registry.expire(nowNanos = 1_000L, maximumSilenceNanos = 100L)
        assertTrue(registry.heartbeat(cameraId, 1_050L))
        assertTrue(
            registry.expire(
                nowNanos = 1_100L,
                maximumSilenceNanos = 100L,
            ).isEmpty(),
        )
        assertEquals(1, registry.size())
    }

    @Test
    fun networkTelemetryUpdatesLinkMeasurementsAndHeartbeatAtomically() {
        val registry = RemoteCameraSessionRegistry()
        val cameraId = DeviceId("cam-telemetry")
        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = cameraId,
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.79",
                        port = 9100,
                    ),
                capabilities = capabilities(),
                lastHeartbeatNanos = 1_000L,
            ),
        )

        assertTrue(
            registry.recordNetworkTelemetry(
                cameraId = cameraId,
                packetLossRatio = 0.12,
                latencyMillis = 77.0,
                requestedKbps = 2_400.0,
                timestampNanos = 1_200L,
            ),
        )

        val updated = registry.find(cameraId)!!
        assertEquals(0.12, updated.packetLossRatio)
        assertEquals(77.0, updated.latencyMillis)
        assertEquals(2_400.0, updated.requestedKbps)
        assertEquals(1_200L, updated.lastHeartbeatNanos)

        assertFalse(
            registry.recordNetworkTelemetry(
                cameraId = cameraId,
                packetLossRatio = 0.95,
                latencyMillis = 500.0,
                requestedKbps = 9_000.0,
                timestampNanos = 1_100L,
            ),
        )
        val afterStaleSample = registry.find(cameraId)!!
        assertEquals(0.12, afterStaleSample.packetLossRatio)
        assertEquals(77.0, afterStaleSample.latencyMillis)
        assertEquals(2_400.0, afterStaleSample.requestedKbps)
        assertEquals(1_200L, afterStaleSample.lastHeartbeatNanos)

        assertTrue(
            registry.expire(
                nowNanos = 1_299L,
                maximumSilenceNanos = 100L,
            ).isEmpty(),
        )
        assertEquals(
            listOf(cameraId),
            registry.expire(
                nowNanos = 1_300L,
                maximumSilenceNanos = 100L,
            ),
        )
    }

    @Test
    fun invalidNetworkTelemetryIsRejectedBeforeChangingTheRegistry() {
        val registry = RemoteCameraSessionRegistry()
        val cameraId = DeviceId("cam-invalid-telemetry")
        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = cameraId,
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.80",
                        port = 9100,
                    ),
                capabilities = capabilities(),
                lastHeartbeatNanos = 1_000L,
            ),
        )

        assertFailsWith<IllegalArgumentException> {
            registry.recordNetworkTelemetry(
                cameraId = cameraId,
                packetLossRatio = 1.1,
                latencyMillis = 20.0,
                requestedKbps = 1_000.0,
                timestampNanos = 1_100L,
            )
        }
        assertEquals(1_000L, registry.find(cameraId)!!.lastHeartbeatNanos)
        assertEquals(null, registry.find(cameraId)!!.packetLossRatio)
        assertFalse(
            registry.recordNetworkTelemetry(
                cameraId = DeviceId("unknown-camera"),
                packetLossRatio = 0.01,
                latencyMillis = 20.0,
                requestedKbps = 1_000.0,
                timestampNanos = 1_100L,
            ),
        )
    }

    @Test
    fun reconnecting_same_identity_can_change_endpoint_without_duplicate_camera() {
        val registry =
            RemoteCameraSessionRegistry(
                RemoteCameraSessionRegistry.Config(
                    maximumCameras = 4,
                ),
            )

        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = DeviceId("cam-reconnect"),
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.20",
                        port = 9100,
                    ),
                capabilities = capabilities(),
                timingAuthorityScore = 0.92,
                lastHeartbeatNanos = 100L,
            ),
        )

        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = DeviceId("cam-reconnect"),
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.21",
                        port = 9100,
                    ),
                capabilities = capabilities(),
                timingAuthorityScore = 0.95,
                lastHeartbeatNanos = 200L,
            ),
        )

        assertEquals(1, registry.size())
        assertEquals(
            "cam-reconnect",
            registry
                .findByEndpoint(
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.21",
                        port = 9100,
                    ),
                )
                ?.cameraId
                ?.value,
        )
        assertTrue(registry.duplicateIdentityCount() > 0L)
    }

    @Test
    fun endpoint_collision_cannot_reassign_one_endpoint_to_two_camera_identities() {
        val registry =
            RemoteCameraSessionRegistry(
                RemoteCameraSessionRegistry.Config(
                    maximumCameras = 8,
                ),
            )

        val endpoint =
            UdpRemoteEvidenceTransport.Endpoint(
                host = "10.0.0.30",
                port = 9100,
            )

        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = DeviceId("owner-a"),
                endpoint = endpoint,
                capabilities = capabilities(),
            ),
        )

        assertFailsWith<IllegalStateException> {
            registry.register(
                RemoteCameraSessionRegistry.CameraNode(
                    cameraId = DeviceId("owner-b"),
                    endpoint = endpoint,
                    capabilities = capabilities(),
                ),
            )
        }

        assertEquals(
            DeviceId("owner-a"),
            registry.endpointOwner(endpoint),
        )
        assertEquals(1L, registry.endpointCollisionCount())
    }

    @Test
    fun heartbeat_history_is_bounded_and_monotonic() {
        val registry =
            RemoteCameraSessionRegistry(
                RemoteCameraSessionRegistry.Config(
                    maximumHeartbeatHistory = 3,
                ),
            )

        registry.register(
            RemoteCameraSessionRegistry.CameraNode(
                cameraId = DeviceId("cam-heartbeat"),
                endpoint =
                    UdpRemoteEvidenceTransport.Endpoint(
                        host = "10.0.0.40",
                        port = 9100,
                    ),
                capabilities = capabilities(),
            ),
        )

        registry.heartbeat(
            DeviceId("cam-heartbeat"),
            10L,
        )
        registry.heartbeat(
            DeviceId("cam-heartbeat"),
            20L,
        )
        registry.heartbeat(
            DeviceId("cam-heartbeat"),
            20L,
        )
        registry.heartbeat(
            DeviceId("cam-heartbeat"),
            30L,
        )
        registry.heartbeat(
            DeviceId("cam-heartbeat"),
            40L,
        )

        assertEquals(
            listOf(20L, 30L, 40L),
            registry.heartbeatHistory(
                DeviceId("cam-heartbeat"),
            ),
        )
    }

    private fun capabilities() =
        DeviceCapabilities(
            model = "remote-camera",
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
        )
}
