package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.PhotometricCalibration
import com.nivukx.nox.core.model.FrameTelemetry
import com.nivukx.nox.core.model.GpuBackend
import com.nivukx.nox.core.model.ImageFormat
import kotlin.test.Test
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class LightingEnvironmentPhotometricCalibrationTest {

    @Test
    fun calibratedLightingUsesMeasuredBlackWhiteAndScale() {
        val engine = LightingEnvironmentEngine()
        val calibration =
            PhotometricCalibration(
                deviceId = DeviceId("cam-a"),
                scaleLux = 1800.0,
                blackLevel = 8.0 / 255.0,
                whiteLevel = 240.0 / 255.0,
                referenceTemperatureCelsius = 25.0,
                temperatureCoefficient = 0.01,
                calibratedAtNanos = 1_000_000L,
            )

        val normalized =
            engine.normalizeRadiance(
                normalizedLuminance = 0.50,
                telemetry =
                    telemetry(
                        temperatureCelsius = 25.0f,
                    ),
                calibration = calibration,
            )

        assertTrue(normalized > 0.0)
        assertNotEquals(0.50, normalized)
    }

    @Test
    fun thermalCorrectionChangesNormalizedRadiance() {
        val engine = LightingEnvironmentEngine()
        val calibration =
            PhotometricCalibration(
                deviceId = DeviceId("cam-b"),
                scaleLux = 1000.0,
                blackLevel = 0.0,
                whiteLevel = 1.0,
                referenceTemperatureCelsius = 25.0,
                temperatureCoefficient = 0.02,
                calibratedAtNanos = 1_000_000L,
            )

        val room =
            engine.normalizeRadiance(
                normalizedLuminance = 0.50,
                telemetry =
                    telemetry(
                        temperatureCelsius = 25.0f,
                    ),
                calibration = calibration,
            )

        val hot =
            engine.normalizeRadiance(
                normalizedLuminance = 0.50,
                telemetry =
                    telemetry(
                        temperatureCelsius = 35.0f,
                    ),
                calibration = calibration,
            )

        assertTrue(hot != room)
    }

    private fun telemetry(
        temperatureCelsius: Float,
    ) =
        FrameTelemetry(
            timestampNanos = 2_000_000L,
            exposureTimeNanos = 1_500_000L,
            sensorSensitivityIso = 400,
            focusDistanceMeters = null,
            readoutTimestampNanos = 2_100_000L,
            sensorTemperatureCelsius = temperatureCelsius,
        )
}
