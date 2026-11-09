package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class FaceTimingQualityTest {
    @Test
    fun clean_capture_keeps_face_quality_high() {
        val score = FaceLandmarkEngine.temporalQualityScale(
            captureTimingConfidence = 0.98,
            rollingShutterSkewNanos = 0L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
        )

        assertTrue(score > 0.90)
    }

    @Test
    fun degraded_readout_and_exposure_reduce_face_quality() {
        val strong = FaceLandmarkEngine.temporalQualityScale(
            captureTimingConfidence = 0.98,
            rollingShutterSkewNanos = 8_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
        )
        val weak = FaceLandmarkEngine.temporalQualityScale(
            captureTimingConfidence = 0.45,
            rollingShutterSkewNanos = 24_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 16_000_000L,
        )

        assertTrue(weak < strong)
        assertTrue(weak >= 0.05)
    }

    @Test
    fun face_result_exposes_timing_quality_metadata() {
        val frame = com.nivukx.nox.core.model.FaceFrame(
            timestampNanos = 100L,
            trackingId = 4,
            boundsLeft = 10f,
            boundsTop = 20f,
            boundsRight = 210f,
            boundsBottom = 260f,
            yawDegrees = 2f,
            rollDegrees = 1f,
            points = emptyList(),
            captureTimingConfidence = 0.87,
            rollingShutterSkewNanos = 12_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
        )

        assertEquals(0.87, frame.captureTimingConfidence)
        assertEquals(12_000_000L, frame.rollingShutterSkewNanos)
        assertEquals(33_333_333L, frame.frameDurationNanos)
        assertEquals(8_000_000L, frame.exposureTimeNanos)
    }
}
