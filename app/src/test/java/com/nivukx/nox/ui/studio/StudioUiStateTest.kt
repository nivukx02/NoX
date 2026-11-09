package com.nivukx.nox.ui.studio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StudioUiStateTest {
    private fun telemetry(
        exportReady: Boolean?,
        calibrationReady: Boolean?,
        confidence: Double?,
        nativeHealthy: Boolean?,
    ) = StudioTelemetryState(
        camerasConnected = 4,
        camerasExpected = 4,
        synchronizationConfidence = confidence,
        calibrationReady = calibrationReady,
        trackingConfidence = confidence,
        reconstructionConfidence = confidence,
        droppedObservations = 0,
        thermalConstrained = false,
        exportReady = exportReady,
        nativeEngineHealthy = nativeHealthy,
        motionP95ErrorMillimeters = 1.0,
        motionP99ErrorMillimeters = 1.5,
        motionJitterMillimeters = 0.5,
        temporalLagMilliseconds = 4.0,
        calibrationImageCoverage = 0.80,
        calibrationDepthCoverage = 0.60,
        selectedViewDiversity = 0.80,
    )

    @Test
    fun exportRequiresKnownHealthyPrerequisites() {
        assertTrue(StudioUiStateReducer.exportEnabled(telemetry(true, true, .9, true)))
        assertFalse(StudioUiStateReducer.exportEnabled(telemetry(true, true, null, true)))
        assertFalse(StudioUiStateReducer.exportEnabled(telemetry(true, false, .9, true)))
        assertFalse(StudioUiStateReducer.exportEnabled(telemetry(true, true, .9, false)))
    }

    @Test
    fun unknownTelemetryNeverPretendsToBeHealthy() {
        val unknown = StudioTelemetryState(
            camerasConnected = null,
            camerasExpected = null,
            synchronizationConfidence = null,
            calibrationReady = null,
            trackingConfidence = null,
            reconstructionConfidence = null,
            droppedObservations = null,
            thermalConstrained = null,
            exportReady = null,
            nativeEngineHealthy = null,
        )
        assertTrue(unknown.cameraHealth == null)
        assertTrue(unknown.systemConfidence == null)
        assertFalse(StudioUiStateReducer.exportEnabled(unknown))
    }

    @Test
    fun healthSummaryDoesNotPromoteUnknownTelemetryToReady() {
        val unknown = telemetry(true, true, null, null)
        val summary = StudioHealthModel.summarize(unknown)

        assertEquals(StudioHealthBand.UNKNOWN, summary.band)
        assertTrue(summary.warnings.any { it.contains("Native runtime") })
        assertFalse(summary.isOperationallyReady)
    }

    @Test
    fun nativeFailureIsAReleaseBlockingCondition() {
        val failed = telemetry(true, true, .95, false)
        val summary = StudioHealthModel.summarize(failed)

        assertEquals(StudioHealthBand.BLOCKED, summary.band)
        assertTrue(summary.blockers.any { it.contains("Native runtime") })
        assertFalse(summary.isOperationallyReady)
    }

    @Test
    fun strongKnownTelemetryReachesReadyWithoutInventingExtraSignals() {
        val ready = telemetry(true, true, .95, true)
        val summary = StudioHealthModel.summarize(ready)

        assertEquals(StudioHealthBand.READY, summary.band)
        assertTrue(summary.isOperationallyReady)
        assertTrue(summary.knownSignalCount >= 8)
        assertTrue(summary.blockers.isEmpty())
    }

    @Test
    fun capture_profile_selection_is_normalized_to_a_real_policy() {
        val base = StudioUiStateReducer.reduce(
            telemetry = telemetry(true, true, .95, true),
        )
        val selected = StudioUiStateReducer.selectCaptureProfile(
            base,
            "not-real",
        )
        assertEquals("LIVE_ACTION", selected.captureProfileId)
    }

    @Test
    fun customization_panel_is_pure_reducer_state() {
        val base = StudioUiStateReducer.reduce(
            telemetry = telemetry(true, true, .95, true),
        )
        val opened = StudioUiStateReducer.toggleCustomizationPanel(base)
        val closed = StudioUiStateReducer.toggleCustomizationPanel(opened)
        assertTrue(opened.customizationPanelVisible)
        assertFalse(closed.customizationPanelVisible)
    }


    @Test
    fun precision_evidence_is_exposed_as_a_measured_score() {
        val state = telemetry(true, true, .95, true)
        assertTrue((state.precisionEvidenceScore ?: 0.0) > 0.70)
    }

    @Test
    fun precision_warnings_trigger_for_measured_accuracy_limits() {
        val degraded = StudioTelemetryState(
            camerasConnected = 4,
            camerasExpected = 4,
            synchronizationConfidence = .95,
            calibrationReady = true,
            trackingConfidence = .95,
            reconstructionConfidence = .95,
            droppedObservations = 0,
            thermalConstrained = false,
            exportReady = true,
            nativeEngineHealthy = true,
            motionP95ErrorMillimeters = 8.0,
            temporalLagMilliseconds = 40.0,
            calibrationImageCoverage = .10,
            calibrationDepthCoverage = .05,
            selectedViewDiversity = .20,
        )
        val summary = StudioHealthModel.summarize(degraded)
        assertTrue(summary.warnings.any { it.contains("P95 error") })
        assertTrue(summary.warnings.any { it.contains("temporal lag") })
        assertTrue(summary.warnings.any { it.contains("Calibration image coverage") })
        assertTrue(summary.warnings.any { it.contains("camera geometry") })
    }
}
