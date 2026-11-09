package com.nivukx.nox.core.model

import kotlin.test.Test
import kotlin.test.assertTrue

class LandmarkObservationCaptureTimingQualityTest {

    @Test
    fun highConfidenceShortExposureProducesStrongTimingQuality() {
        val observation =
            LandmarkObservation(
                landmarkId = 1,
                x = 0.5,
                y = 0.5,
                confidence = 0.98,
                timestampNanos = 10_000_000L,
                space = LandmarkSpace.NORMALIZED,
                frameWidth = 3840,
                frameHeight = 2160,
                rollingShutterSkewNanos = 300_000L,
                frameDurationNanos = 8_333_333L,
                exposureTimeNanos = 1_000_000L,
                captureTimingConfidence = 0.99,
            )

        assertTrue(
            observation.captureTimingQuality >= 0.90,
        )
    }

    @Test
    fun largeRollingSkewAndWeakTimingReduceQuality() {
        val observation =
            LandmarkObservation(
                landmarkId = 2,
                x = 0.5,
                y = 0.5,
                confidence = 0.98,
                timestampNanos = 20_000_000L,
                space = LandmarkSpace.NORMALIZED,
                frameWidth = 1920,
                frameHeight = 1080,
                rollingShutterSkewNanos = 12_000_000L,
                frameDurationNanos = 16_666_667L,
                exposureTimeNanos = 8_000_000L,
                captureTimingConfidence = 0.50,
            )

        assertTrue(
            observation.captureTimingQuality <
                0.70,
        )
    }
}
