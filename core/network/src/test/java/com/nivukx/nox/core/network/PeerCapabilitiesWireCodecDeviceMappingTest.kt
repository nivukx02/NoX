package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceCapabilities
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PeerCapabilitiesWireCodecDeviceMappingTest {

    @Test
    fun deviceMappingDoesNotInventHardwareTiming() {
        val capabilities =
            DeviceCapabilities(
                model = "test",
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
                supportsHardwareReadoutTimestamp = false,
                maximumVerifiedConcurrentCameras = 2,
                minimumExposureTimeNanos = 100_000L,
                maximumExposureTimeNanos = 20_000_000L,
            )

        val mapped =
            PeerCapabilitiesWireCodec.fromDeviceCapabilities(
                capabilities,
            )

        assertFalse(mapped.supportsHardwareReadoutTelemetry)
        assertFalse(mapped.supportsVisualTimingBeacon)
        assertEquals(2, mapped.maximumVerifiedConcurrentCameras)
        assertEquals(64, mapped.maximumSubjects)
        assertTrue(mapped.supportsDenseLandmarkFragmentation)
        assertTrue(mapped.supportsMultiActorPoseFragmentation)
    }

    @Test
    fun hardwareDeviceMappingRaisesTimingAuthorityOnlyFromAdvertisedHardware() {
        val capabilities =
            DeviceCapabilities(
                model = "reference",
                manufacturer = "test",
                androidApi = 37,
                cameraIds = listOf("0", "1"),
                supportsRaw = true,
                supportsConcurrentCamera = true,
                supportsGyroscope = true,
                supportsAccelerometer = true,
                maxAnalysisWidth = 3840,
                maxAnalysisHeight = 2160,
                maxFps = 120,
                supportsHardwareReadoutTimestamp = true,
                maximumVerifiedConcurrentCameras = 4,
                minimumExposureTimeNanos = 100_000L,
                maximumExposureTimeNanos = 20_000_000L,
            )

        val mapped =
            PeerCapabilitiesWireCodec.fromDeviceCapabilities(
                capabilities,
                timingAuthorityScore = 0.96,
                timingResidualP95Milliseconds = 0.8,
            )

        assertTrue(mapped.supportsHardwareReadoutTelemetry)
        assertEquals(0.96, mapped.timingAuthorityScore)
        assertEquals(0.8, mapped.timingResidualP95Milliseconds)
    }
}
