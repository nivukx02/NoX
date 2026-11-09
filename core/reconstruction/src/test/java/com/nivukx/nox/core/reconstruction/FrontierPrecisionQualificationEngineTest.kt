package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierPrecisionQualificationEngineTest {
    private fun input(
        p95: Double,
        p99: Double,
    ) = FrontierPrecisionQualificationEngine.Input(
        evidenceSamples = 800,
        measuredP95Millimeters = p95,
        measuredP99Millimeters = p99,
        externalReferenceP95Millimeters = null,
        externalReferenceUncertaintyMillimeters = 0.10,
        systemMeasurementUncertaintyMillimeters = 0.10,
        matchedReference = false,
        independentViews = 8,
        baselineMeters = 1.20,
        parallaxDegrees = 18.0,
        calibrationConfidence = 0.99,
        synchronizationConfidence = 0.99,
        viewDiversity = 0.92,
        crossViewAgreement = 0.97,
        occlusionRatio = 0.04,
        identityContinuity = 0.99,
        surfaceTemporalStability = 0.98,
        thermalRisk = 0.20,
        memoryPressure = 0.25,
        timingUncertaintyMicros = 300.0,
        reprojectionP95Pixels = 0.8,
    )

    @Test
    fun measuredTargetFailureUsesExplicitExceedanceBlocker() {
        val result = FrontierPrecisionQualificationEngine().evaluate(
            input(1.10, 2.00),
        )
        assertEquals(
            FrontierPrecisionQualificationEngine.Blocker.MEASUREMENT_EXCEEDS_TARGET,
            result.blocker,
        )
        assertTrue(!result.physicallyQualified)
    }

    @Test
    fun measuredTargetFailureIsNotReportedAsMissingEvidence() {
        val result = FrontierPrecisionQualificationEngine().evaluate(
            input(2.40, 3.80),
        )
        assertTrue(
            result.blocker !=
                FrontierPrecisionQualificationEngine.Blocker.MEASUREMENT_MISSING,
        )
        assertTrue(
            result.recommendations.any {
                it.contains("Reduce measured P95/P99 error")
            },
        )
    }
}
