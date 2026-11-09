package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.DeviceCapabilities
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DeviceCapabilitiesTimingEvidenceTest {

    @Test
    fun hardwareTimingAndConcurrencyAreRepresentedInCanonicalDescriptor() {
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
                maximumVerifiedConcurrentCameras = 2,
                minimumExposureTimeNanos = 100_000L,
                maximumExposureTimeNanos = 33_000_000L,
            )

        assertTrue(capabilities.supportsHardwareReadoutTimestamp)
        assertEquals(2, capabilities.maximumVerifiedConcurrentCameras)
        assertEquals(100_000L, capabilities.minimumExposureTimeNanos)
        assertEquals(33_000_000L, capabilities.maximumExposureTimeNanos)
    }
}
