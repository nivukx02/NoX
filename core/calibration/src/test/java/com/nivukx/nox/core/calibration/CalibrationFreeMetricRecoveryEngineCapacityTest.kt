package com.nivukx.nox.core.calibration

import kotlin.test.Test
import kotlin.test.assertEquals

class CalibrationFreeMetricRecoveryEngineCapacityTest {

    @Test
    fun calibratedMetricRecoveryCanUseMoreThanThirtyTwoAnchors() {
        val anchors =
            (0 until 96).map {
                CalibrationFreeMetricRecoveryEngine.MetricAnchor(
                    metricDistanceMeters = 2.0,
                    observedDistanceUnits = 1.0,
                    confidence = 0.92,
                )
            }

        val result =
            CalibrationFreeMetricRecoveryEngine().recover(
                CalibrationFreeMetricRecoveryEngine.Input(
                    timestampNanos = 1_000_000L,
                    calibrated = true,
                    metricAnchors = anchors,
                    segmentObservations = emptyList(),
                ),
            )

        assertEquals(
            96,
            result.acceptedAnchors,
        )
        assertEquals(
            2.0,
            result.scaleMetersPerUnit,
        )
    }
}
