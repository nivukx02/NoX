package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CameraTelemetryState
import org.junit.Assert.assertTrue
import org.junit.Test

class LightingAdaptationEngineTest {

    @Test
    fun stableTelemetryProducesFiniteLightingState() {
        val engine = LightingAdaptationEngine()
        val result = engine.update(
            timestampNanos = 1_000_000_000L,
            telemetry = telemetry(),
            estimatedLux = 320.0,
        )

        assertTrue(result.lighting.lux >= 0.0)
        assertTrue(result.lighting.correlatedColorTemperatureKelvin > 0.0)
        assertTrue(result.inferenceScale in 0.25..1.0)
        assertTrue(result.eyeScale in 0.15..1.0)
    }

    @Test
    fun changingLuxRaisesAdaptationDiagnostics() {
        val engine = LightingAdaptationEngine()
        engine.update(1_000_000_000L, telemetry(), 120.0)
        val result = engine.update(1_033_000_000L, telemetry(), 9_000.0)

        assertTrue(result.lighting.highlightRisk >= 0.0)
        assertTrue(result.lighting.flickerRisk >= 0.0)
        assertTrue(result.diagnostics.isNotEmpty())
    }

    private fun telemetry() = CameraTelemetryState(
        timestampNanos = 1_000_000_000L,
        sensorTimestampNanos = 1_000_000_000L,
        exposureTimeNanos = 8_000_000L,
        sensitivityIso = 200,
        focusDistanceDiopters = 1.0,
        focusConfidence = 0.94,
        aperture = 1.8,
        focalLengthMillimeters = 5.8,
        rollingShutterReadoutNanos = 10_000_000L,
        frameDurationNanos = 16_666_666L,
        zoomRatio = 1.0,
        stabilizationActive = true,
        cameraMotionRisk = 0.05,
    )
}
