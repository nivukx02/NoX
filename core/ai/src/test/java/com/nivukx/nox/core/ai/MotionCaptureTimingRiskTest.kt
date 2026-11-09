package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MotionCaptureTimingRiskTest {
    @Test
    fun clean_global_shutter_has_no_readout_risk() {
        val risk = MotionCaptureEngine.rollingShutterRiskFromTelemetry(
            rollingShutterSkewNanos = 0L,
            frameDurationNanos = 33_333_333L,
            captureTimingConfidence = 0.98,
        )

        assertEquals(0.0, risk, 1e-12)
    }

    @Test
    fun rising_readout_ratio_increases_motion_risk() {
        val low = MotionCaptureEngine.rollingShutterRiskFromTelemetry(
            rollingShutterSkewNanos = 4_000_000L,
            frameDurationNanos = 33_333_333L,
            captureTimingConfidence = 0.98,
        )
        val high = MotionCaptureEngine.rollingShutterRiskFromTelemetry(
            rollingShutterSkewNanos = 24_000_000L,
            frameDurationNanos = 33_333_333L,
            captureTimingConfidence = 0.98,
        )

        assertTrue(high > low)
        assertTrue(high <= 1.0)
    }

    @Test
    fun poor_timing_confidence_adds_system_risk_without_exceeding_bounds() {
        val risk = MotionCaptureEngine.rollingShutterRiskFromTelemetry(
            rollingShutterSkewNanos = 8_000_000L,
            frameDurationNanos = 33_333_333L,
            captureTimingConfidence = 0.30,
        )

        assertTrue(risk in 0.0..1.0)
        assertTrue(risk > 0.10)
    }
}
