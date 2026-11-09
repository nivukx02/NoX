package com.nivukx.nox.core.calibration

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class CalibrationFreeMetricRecoveryEngineTest {

    @Test
    fun known_segment_priors_recover_metric_scale_robustly() {
        val engine = CalibrationFreeMetricRecoveryEngine()

        val result = engine.recover(
            CalibrationFreeMetricRecoveryEngine.Input(
                timestampNanos = 1_000_000_000L,
                calibrated = false,
                metricAnchors = emptyList(),
                segmentObservations = listOf(
                    segment(0.50, 0.90),
                    segment(0.40, 0.72),
                    segment(0.60, 1.08),
                    segment(0.45, 0.81),
                    segment(0.10, 2.80),
                ),
            ),
        )

        assertEquals(
            CalibrationFreeMetricRecoveryEngine.Mode.SCALE_ANCHORED_METRIC,
            result.mode,
        )
        assertEquals(
            1.80,
            requireNotNull(result.scaleMetersPerUnit),
            0.02,
        )
        assertTrue(result.confidence > 0.80)
        assertTrue(result.acceptedSegments >= 4)
        assertTrue(result.rejectedSegments >= 1)
    }

    @Test
    fun missing_scale_anchor_stays_relative_only() {
        val engine = CalibrationFreeMetricRecoveryEngine()

        val result = engine.recover(
            CalibrationFreeMetricRecoveryEngine.Input(
                timestampNanos = 2_000_000_000L,
                calibrated = false,
                metricAnchors = emptyList(),
                segmentObservations = listOf(
                    segment(0.4, 0.72),
                    segment(0.5, 0.90),
                ),
            ),
        )

        assertEquals(
            CalibrationFreeMetricRecoveryEngine.Mode.RELATIVE_ONLY,
            result.mode,
        )
        assertTrue(result.scaleMetersPerUnit == null)
        assertTrue(result.warnings.any { it.contains("metric-anchor") })
    }

    @Test
    fun explicit_metric_anchor_overrides_weak_shape_prior() {
        val engine = CalibrationFreeMetricRecoveryEngine()

        val result = engine.recover(
            CalibrationFreeMetricRecoveryEngine.Input(
                timestampNanos = 3_000_000_000L,
                calibrated = true,
                metricAnchors = listOf(
                    CalibrationFreeMetricRecoveryEngine.MetricAnchor(
                        metricDistanceMeters = 2.0,
                        observedDistanceUnits = 1.0,
                        confidence = 0.98,
                    ),
                ),
                segmentObservations = listOf(
                    segment(0.50, 0.85),
                    segment(0.60, 1.05),
                ),
            ),
        )

        assertEquals(
            CalibrationFreeMetricRecoveryEngine.Mode.CALIBRATED_METRIC,
            result.mode,
        )
        assertEquals(
            2.0,
            requireNotNull(result.scaleMetersPerUnit),
            0.01,
        )
        assertTrue(result.confidence > 0.90)
    }

    @Test
    fun timestamp_regression_does_not_rewrite_scale_state() {
        val engine = CalibrationFreeMetricRecoveryEngine()

        val first = engine.recover(
            CalibrationFreeMetricRecoveryEngine.Input(
                timestampNanos = 5_000_000_000L,
                calibrated = false,
                metricAnchors = emptyList(),
                segmentObservations = listOf(
                    segment(0.5, 0.9),
                    segment(0.4, 0.72),
                ),
            ),
        )

        val stale = engine.recover(
            CalibrationFreeMetricRecoveryEngine.Input(
                timestampNanos = 4_000_000_000L,
                calibrated = false,
                metricAnchors = emptyList(),
                segmentObservations = listOf(
                    segment(0.5, 2.0),
                ),
            ),
        )

        assertEquals(first.timestampNanos, stale.timestampNanos)
        assertEquals(first.scaleMetersPerUnit, stale.scaleMetersPerUnit)
        assertTrue(stale.warnings.any { it.contains("timestamp-reversal") })
    }

    private fun segment(
        observed: Double,
        expectedMeters: Double,
    ) = CalibrationFreeMetricRecoveryEngine.SegmentObservation(
        observedDistanceUnits = observed,
        expectedMetricDistanceMeters = expectedMeters,
        confidence = 0.95,
    )
}
