package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CrossCameraPhotometricCalibrationEngineTest {

    @Test
    fun robustFitRecoversChannelGainAndOffsetUnderOutliers() {
        val engine =
            CrossCameraPhotometricCalibrationEngine()

        val samples =
            (0 until 80).map { index ->
                val x =
                    (
                        (index % 20) /
                            19.0
                    ).coerceIn(0.03, 0.97)

                CrossCameraPhotometricCalibrationEngine.Sample(
                    source =
                        CrossCameraPhotometricCalibrationEngine.Rgb(
                            x,
                            x * 0.75,
                            x * 0.55,
                        ),
                    target =
                        CrossCameraPhotometricCalibrationEngine.Rgb(
                            (0.04 + 1.20 * x)
                                .coerceIn(0.0, 1.0),
                            (0.02 + 1.08 * x * 0.75)
                                .coerceIn(0.0, 1.0),
                            (0.03 + 0.92 * x * 0.55)
                                .coerceIn(0.0, 1.0),
                        ),
                    confidence = 0.98,
                )
            } +
                listOf(
                    CrossCameraPhotometricCalibrationEngine.Sample(
                        source =
                            CrossCameraPhotometricCalibrationEngine.Rgb(
                                0.9,
                                0.8,
                                0.7,
                            ),
                        target =
                            CrossCameraPhotometricCalibrationEngine.Rgb(
                                0.02,
                                0.98,
                                0.03,
                            ),
                        confidence = 0.20,
                    ),
                )

        val result =
            engine.fit(
                sourceDeviceId = DeviceId("cam-a"),
                referenceDeviceId = DeviceId("cam-b"),
                samples = samples,
            )

        assertTrue(result.accepted)
        assertTrue(abs(result.transform.red.gain - 1.20) < 0.08)
        assertTrue(abs(result.transform.green.gain - 1.08) < 0.08)
        assertTrue(abs(result.transform.blue.gain - 0.92) < 0.08)
        assertTrue(result.consistency >= 0.90)
    }

    @Test
    fun saturatedSamplesAreRejectedBeforeFitting() {
        val engine =
            CrossCameraPhotometricCalibrationEngine()

        val samples =
            (0 until 48).map { index ->
                val x =
                    0.10 +
                        0.75 *
                        (index / 47.0)

                CrossCameraPhotometricCalibrationEngine.Sample(
                    source =
                        CrossCameraPhotometricCalibrationEngine.Rgb(
                            x,
                            x,
                            x,
                        ),
                    target =
                        CrossCameraPhotometricCalibrationEngine.Rgb(
                            x,
                            x,
                            x,
                        ),
                    confidence = 1.0,
                    sourceSaturation = index % 12 == 0,
                    targetSaturation = false,
                )
            }

        val result =
            engine.fit(
                sourceDeviceId = DeviceId("cam-a"),
                referenceDeviceId = DeviceId("cam-b"),
                samples = samples,
            )

        assertTrue(result.accepted)
        assertEquals(
            4,
            result.rejectedSampleCount,
        )
    }

    @Test
    fun insufficientPhotometricVariationFailsClosed() {
        val engine =
            CrossCameraPhotometricCalibrationEngine()

        val result =
            engine.fit(
                sourceDeviceId = DeviceId("cam-a"),
                referenceDeviceId = DeviceId("cam-b"),
                samples =
                    List(32) {
                        CrossCameraPhotometricCalibrationEngine.Sample(
                            source =
                                CrossCameraPhotometricCalibrationEngine.Rgb(
                                    0.5,
                                    0.5,
                                    0.5,
                                ),
                            target =
                                CrossCameraPhotometricCalibrationEngine.Rgb(
                                    0.55,
                                    0.55,
                                    0.55,
                                ),
                            confidence = 0.95,
                        )
                    },
            )

        assertTrue(!result.accepted)
        assertTrue(
            result.warnings.contains(
                "PHOTOMETRIC_SOURCE_VARIATION_INSUFFICIENT",
            ),
        )
    }
    @Test
    fun unequalExposureIsNormalizedInTheCorrectDirection() {
        val engine =
            CrossCameraPhotometricCalibrationEngine()

        val samples =
            (0 until 64).map { index ->
                val x =
                    0.06 +
                        0.88 *
                        (index / 63.0)

                CrossCameraPhotometricCalibrationEngine.Sample(
                    source =
                        CrossCameraPhotometricCalibrationEngine.Rgb(
                            x,
                            x,
                            x,
                        ),
                    target =
                        CrossCameraPhotometricCalibrationEngine.Rgb(
                            (0.02 + 0.50 * x).coerceIn(0.0, 1.0),
                            (0.01 + 0.50 * x).coerceIn(0.0, 1.0),
                            (0.03 + 0.50 * x).coerceIn(0.0, 1.0),
                        ),
                    confidence = 0.98,
                    sourceMetadata =
                        CrossCameraPhotometricCalibrationEngine.CaptureMetadata(
                            exposureNanos = 20_000_000L,
                            iso = 800,
                        ),
                    targetMetadata =
                        CrossCameraPhotometricCalibrationEngine.CaptureMetadata(
                            exposureNanos = 10_000_000L,
                            iso = 400,
                        ),
                )
            }

        val result =
            engine.fit(
                sourceDeviceId = DeviceId("cam-long"),
                referenceDeviceId = DeviceId("cam-short"),
                samples = samples,
            )

        assertTrue(result.accepted)
        assertTrue(
            result.transform.red.gain <
                1.40,
        )
        assertTrue(
            result.transform.red.gain >
                0.80,
        )
    }

}
