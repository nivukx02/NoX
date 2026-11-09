package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.LightingEstimate
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LightingEnvironmentTemporalConsistencyEngineTest {

    @Test
    fun stableLightingRemainsStable() {
        val engine = LightingEnvironmentTemporalConsistencyEngine()
        engine.evaluate(estimate(1_000_000_000L, 800.0, 5200.0))
        val result = engine.evaluate(estimate(1_016_000_000L, 810.0, 5220.0))
        assertEquals(
            LightingEnvironmentTemporalConsistencyEngine.State.STABLE,
            result.state,
        )
        assertTrue(result.continuityScore > 0.70)
    }

    @Test
    fun repeatedIlluminanceJumpsBecomeFlicker() {
        val engine = LightingEnvironmentTemporalConsistencyEngine(
            LightingEnvironmentTemporalConsistencyEngine.Config(
                persistentFlickerFrames = 2,
            ),
        )
        engine.evaluate(estimate(2_000_000_000L, 700.0, 5000.0))
        engine.evaluate(estimate(2_016_000_000L, 2300.0, 5000.0))
        engine.evaluate(estimate(2_032_000_000L, 700.0, 5000.0))
        val result =
            engine.evaluate(estimate(2_048_000_000L, 2300.0, 5000.0))
        assertEquals(
            LightingEnvironmentTemporalConsistencyEngine.State.FLICKER,
            result.state,
        )
        assertTrue(result.warnings.any { it.contains("flicker") })
    }

    @Test
    fun abruptColorTemperatureShiftIsTransition() {
        val engine = LightingEnvironmentTemporalConsistencyEngine()
        engine.evaluate(estimate(3_000_000_000L, 800.0, 3200.0))
        val result =
            engine.evaluate(estimate(3_016_000_000L, 820.0, 6500.0))
        assertEquals(
            LightingEnvironmentTemporalConsistencyEngine.State.TRANSITION,
            result.state,
        )
    }

    @Test
    fun lowConfidenceLightingDegradesTrust() {
        val engine = LightingEnvironmentTemporalConsistencyEngine()
        val result = engine.evaluate(
            estimate(
                timestamp = 4_000_000_000L,
                lux = 800.0,
                cct = 5000.0,
                confidence = 0.10,
            ),
        )
        assertTrue(
            result.state ==
                LightingEnvironmentTemporalConsistencyEngine.State.DEGRADED ||
                result.state ==
                LightingEnvironmentTemporalConsistencyEngine.State.UNKNOWN,
        )
        assertTrue(result.confidence < 0.35)
    }

    private fun estimate(
        timestamp: Long,
        lux: Double,
        cct: Double,
        confidence: Double = 0.92,
    ) = LightingEstimate(
        timestampNanos = timestamp,
        illuminanceLux = lux,
        colorTemperatureKelvin = cct,
        exposureCompensationEv = 0.0,
        ambientLuminance = 0.50,
        directionalLight = Vec3(0.0, 0.0, 1.0),
        directionalIntensity = 0.25,
        confidence = confidence,
        dynamicRangeStops = 8.0,
    )
}