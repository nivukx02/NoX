package com.nivukx.nox.core.model

import kotlin.test.assertEquals
import org.junit.Test

class LandmarkObservationTimingMetadataTest {
    @Test
    fun timingMetadata_defaults_to_legacy_safe_values() {
        val observation = LandmarkObservation(
            landmarkId = 7,
            x = 120.0,
            y = 80.0,
            confidence = 0.9,
            timestampNanos = 10L,
        )

        assertEquals(0, observation.frameWidth)
        assertEquals(0, observation.frameHeight)
        assertEquals(0L, observation.rollingShutterSkewNanos)
        assertEquals(0L, observation.frameDurationNanos)
        assertEquals(0L, observation.exposureTimeNanos)
        assertEquals(1.0, observation.captureTimingConfidence)
    }

    @Test
    fun timingMetadata_preserves_physical_capture_contract() {
        val observation = LandmarkObservation(
            landmarkId = 3,
            x = 640.0,
            y = 540.0,
            confidence = 0.97,
            timestampNanos = 20L,
            frameWidth = 1920,
            frameHeight = 1080,
            rollingShutterSkewNanos = 12_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
            captureTimingConfidence = 0.94,
        )

        assertEquals(1920, observation.frameWidth)
        assertEquals(1080, observation.frameHeight)
        assertEquals(12_000_000L, observation.rollingShutterSkewNanos)
        assertEquals(33_333_333L, observation.frameDurationNanos)
        assertEquals(8_000_000L, observation.exposureTimeNanos)
        assertEquals(0.94, observation.captureTimingConfidence)
    }
}
