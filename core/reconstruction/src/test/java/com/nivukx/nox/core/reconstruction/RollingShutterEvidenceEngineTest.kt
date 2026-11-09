package com.nivukx.nox.core.reconstruction

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class RollingShutterEvidenceEngineTest {

    @Test
    fun normalized_sensor_row_can_override_pixel_derived_row() {
        val result =
            RollingShutterEvidenceEngine().evaluate(
                RollingShutterEvidenceEngine.Input(
                    yPx = 540.0,
                    frameHeightPx = 1080,
                    normalizedRow = 0.0,
                    rollingShutterSkewNanos = 12_000_000L,
                    frameDurationNanos = 33_333_333L,
                    exposureTimeNanos = 8_000_000L,
                    captureTimingConfidence = 0.95,
                ),
            )

        assertTrue(result.normalizedRow <= 0.001)
        assertTrue(result.rowOffsetNanos < 0L)
    }

    @Test
    fun center_row_has_zero_temporal_offset() {
        val result = RollingShutterEvidenceEngine().evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 540.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 12_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 8_000_000L,
                captureTimingConfidence = 0.95,
            ),
        )

        assertEquals(0L, result.rowOffsetNanos)
        assertEquals(0.0, result.rollingShutterFraction, 1e-12)
        assertTrue(result.confidenceScale > 0.90)
    }

    @Test
    fun top_and_bottom_rows_have_opposite_signed_offsets() {
        val engine = RollingShutterEvidenceEngine()
        val top = engine.evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 0.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 12_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 8_000_000L,
                captureTimingConfidence = 1.0,
            ),
        )
        val bottom = engine.evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 1080.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 12_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 8_000_000L,
                captureTimingConfidence = 1.0,
            ),
        )

        assertTrue(top.rowOffsetNanos < 0L)
        assertTrue(bottom.rowOffsetNanos > 0L)
        assertTrue(top.rollingShutterFraction < 0.0)
        assertTrue(bottom.rollingShutterFraction > 0.0)
    }

    @Test
    fun degraded_capture_timing_reduces_evidence_confidence() {
        val engine = RollingShutterEvidenceEngine()
        val strong = engine.evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 100.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 20_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 12_000_000L,
                captureTimingConfidence = 0.98,
            ),
        )
        val weak = engine.evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 100.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 20_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 12_000_000L,
                captureTimingConfidence = 0.35,
            ),
        )

        assertTrue(weak.confidenceScale < strong.confidenceScale)
    }


    @Test
    fun activeArrayViewportScalesReportedSkew() {
        val engine = RollingShutterEvidenceEngine()
        val result = engine.evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 540.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 20_000_000L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 8_000_000L,
                captureTimingConfidence = 0.96,
                activeArrayHeightPx = 4000,
                viewportTopPx = 1000,
                viewportHeightPx = 2000,
            ),
        )

        assertEquals(0L, result.rowOffsetNanos)
        assertTrue(result.readoutRatio < 0.40)
    }

    @Test
    fun missing_skew_preserves_zero_correction_and_reports_metadata_gap() {
        val result = RollingShutterEvidenceEngine().evaluate(
            RollingShutterEvidenceEngine.Input(
                yPx = 100.0,
                frameHeightPx = 1080,
                rollingShutterSkewNanos = 0L,
                frameDurationNanos = 33_333_333L,
                exposureTimeNanos = 8_000_000L,
                captureTimingConfidence = 0.90,
            ),
        )

        assertEquals(0L, result.rowOffsetNanos)
        assertEquals(0.0, result.rollingShutterFraction, 1e-12)
        assertTrue(result.warnings.any { it == "ROLLING_SHUTTER_SKEW_UNAVAILABLE" })
    }
}
