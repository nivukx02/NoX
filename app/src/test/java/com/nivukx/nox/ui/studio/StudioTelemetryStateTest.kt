package com.nivukx.nox.ui.studio

import org.junit.Assert.assertEquals
import org.junit.Test

class StudioTelemetryStateTest {
    private fun base(
        coverage: Double? = 0.9,
        failureRisk: Double? = 0.1,
    ) = StudioTelemetryState(
        camerasConnected = 2,
        camerasExpected = 2,
        synchronizationConfidence = 0.98,
        calibrationReady = true,
        trackingConfidence = 0.96,
        reconstructionConfidence = 0.94,
        droppedObservations = 0L,
        thermalConstrained = false,
        exportReady = true,
        correlatedEvidenceCoverage = coverage,
        correlatedEvidenceFailureRisk = failureRisk,
        nativeEngineHealthy = true,
    )

    @Test
    fun highFailureRiskIsRejected() {
        assertEquals("REJECTED", base(failureRisk = 0.90).correlatedEvidenceState)
    }

    @Test
    fun lowCoverageOrModerateRiskIsDegraded() {
        assertEquals("DEGRADED", base(coverage = 0.40, failureRisk = 0.20).correlatedEvidenceState)
        assertEquals("DEGRADED", base(coverage = 0.90, failureRisk = 0.60).correlatedEvidenceState)
    }

    @Test
    fun healthyEvidenceIsReady() {
        assertEquals("HEALTHY", base().correlatedEvidenceState)
    }
}
