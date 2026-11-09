package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ThermalExtrinsicDriftEngineTest {
    private val engine = ThermalExtrinsicDriftEngine(
        ThermalExtrinsicDriftEngine.Config(
            minimumSamples = 4,
        ),
    )

    private fun sample(
        time: Long,
        temp: Double,
        driftMm: Double = 0.0,
        driftDegrees: Double = 0.0,
    ) = ThermalExtrinsicDriftEngine.Sample(
        cameraId = DeviceId("cam-1"),
        timestampNanos = time,
        temperatureCelsius = temp,
        calibratedPose = Pose(),
        observedPose = Pose(
            position = Vec3(driftMm / 1000.0, 0.0, 0.0),
            rotation = Quat(
                kotlin.math.cos(Math.toRadians(driftDegrees) / 2.0),
                0.0,
                0.0,
                kotlin.math.sin(Math.toRadians(driftDegrees) / 2.0),
            ),
        ),
        calibrationConfidence = 0.98,
        mountIntegrity = 0.98,
    )

    @Test
    fun lowDriftRemainsStable() {
        var result = engine.observe(sample(1L, 25.0))
        result = engine.observe(sample(2L, 25.5, driftMm = 0.05))
        result = engine.observe(sample(3L, 26.0, driftMm = 0.08))
        result = engine.observe(sample(4L, 26.5, driftMm = 0.10))

        assertEquals(
            ThermalExtrinsicDriftEngine.State.STABLE,
            result.state,
        )
        assertTrue(result.translationDriftMm < 0.35)
    }

    @Test
    fun thermalTrendRaisesWarning() {
        var result = engine.observe(sample(10L, 25.0))
        result = engine.observe(sample(11L, 30.0, driftMm = 0.40))
        result = engine.observe(sample(12L, 35.0, driftMm = 0.80))
        result = engine.observe(sample(13L, 40.0, driftMm = 1.10))

        assertTrue(
            result.warnings.any {
                it.contains("THERMAL_TRANSLATION_COEFFICIENT")
            },
        )
    }

    @Test
    fun heating_and_cooling_paths_expose_thermal_hysteresis() {
        val local =
            ThermalExtrinsicDriftEngine(
                ThermalExtrinsicDriftEngine.Config(
                    minimumSamples = 6,
                    warningThermalPathHysteresisMm = 0.20,
                    hardThermalPathHysteresisMm = 2.0,
                ),
            )

        local.observe(sample(50L, 25.0, driftMm = 0.10))
        local.observe(sample(51L, 30.0, driftMm = 0.40))
        local.observe(sample(52L, 35.0, driftMm = 0.80))
        local.observe(sample(53L, 30.0, driftMm = 0.70))
        local.observe(sample(54L, 25.5, driftMm = 0.42))
        val result =
            local.observe(
                sample(
                    55L,
                    30.5,
                    driftMm = 0.95,
                ),
            )

        assertTrue(
            result.thermalPathHysteresisMm >
                0.20,
        )
        assertTrue(
            result.warnings.any {
                it == "THERMAL_PATH_HYSTERESIS_ELEVATED"
            },
        )
    }

    @Test
    fun hardDriftBlocksCamera() {
        var result = engine.observe(sample(20L, 25.0))
        result = engine.observe(sample(21L, 30.0, driftMm = 0.40, driftDegrees = 0.2))
        result = engine.observe(sample(22L, 35.0, driftMm = 0.90, driftDegrees = 0.4))
        result = engine.observe(sample(23L, 40.0, driftMm = 1.70, driftDegrees = 0.8))

        assertEquals(
            ThermalExtrinsicDriftEngine.State.BLOCKED,
            result.state,
        )
    }
    @Test
    fun robustThermalSlopeIgnoresSingleBadCalibrationOutlier() {
        val local =
            ThermalExtrinsicDriftEngine(
                ThermalExtrinsicDriftEngine.Config(
                    minimumSamples = 5,
                    warningDriftRateMmPerCelsius = 0.025,
                    hardDriftRateMmPerCelsius = 0.080,
                ),
            )

        local.observe(sample(30L, 25.0, driftMm = 0.00))
        local.observe(sample(31L, 30.0, driftMm = 0.12))
        local.observe(sample(32L, 35.0, driftMm = 0.24))
        local.observe(sample(33L, 40.0, driftMm = 1.70))
        val result =
            local.observe(sample(34L, 45.0, driftMm = 0.48))

        assertTrue(
            result.translationDriftRateMmPerCelsius < 0.08,
        )
        assertTrue(
            result.warnings.none {
                it == "THERMAL_TRANSLATION_COEFFICIENT_ELEVATED"
            },
        )
    }

    @Test
    fun narrowTemperatureSpanDoesNotCreateFalseThermalConfidence() {
        val local =
            ThermalExtrinsicDriftEngine(
                ThermalExtrinsicDriftEngine.Config(
                    minimumSamples = 4,
                    minimumTemperatureSpanForRateCelsius = 4.0,
                ),
            )

        local.observe(sample(40L, 25.0, driftMm = 0.00))
        local.observe(sample(41L, 25.5, driftMm = 0.08))
        local.observe(sample(42L, 26.0, driftMm = 0.16))
        val result =
            local.observe(sample(43L, 26.4, driftMm = 0.24))

        assertTrue(
            result.temperatureSpanConfidence < 0.50,
        )
        assertTrue(
            result.state !=
                ThermalExtrinsicDriftEngine.State.BLOCKED,
        )
    }

}
