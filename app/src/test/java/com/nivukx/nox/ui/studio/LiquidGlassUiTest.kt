package com.nivukx.nox.ui.studio

import org.junit.Assert.assertTrue
import org.junit.Test

class LiquidGlassUiTest {
    @Test
    fun liquidGlassLayersDoNotChangeExportTruth() {
        val telemetry = StudioTelemetryState(
            camerasConnected = 4,
            camerasExpected = 4,
            synchronizationConfidence = .9,
            calibrationReady = true,
            trackingConfidence = .9,
            reconstructionConfidence = .9,
            droppedObservations = 0,
            thermalConstrained = false,
            exportReady = true,
            nativeEngineHealthy = true,
        )
        assertTrue(StudioUiStateReducer.exportEnabled(telemetry))
    }
}
