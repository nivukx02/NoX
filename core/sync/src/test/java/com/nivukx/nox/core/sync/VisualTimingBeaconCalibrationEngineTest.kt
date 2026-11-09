package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class VisualTimingBeaconCalibrationEngineTest {

    @Test
    fun exposureCenterCorrectionRecoversEventClock() {
        val engine = VisualTimingBeaconCalibrationEngine()
        val device = DeviceId("cam-a")

        val observations =
            (0 until 48).map { index ->
                val start = 1_000_000_000L + index * 40_000_000L
                val exposure = 8_000_000L
                val rolling = 4_000_000L
                val row = if (index % 2 == 0) 0.25 else 0.75
                val center =
                    start +
                        exposure / 2L +
                        ((row - 0.5) * rolling)
                            .toLong()
                val beacon = center + 2_500_000L

                VisualTimingBeaconCalibrationEngine.Observation(
                    deviceId = device,
                    deviceTimestampNanos = start,
                    beaconTimestampNanos = beacon,
                    exposureTimeNanos = exposure,
                    rollingShutterSkewNanos = rolling,
                    normalizedRow = row,
                    decodeConfidence = 0.98,
                )
            }

        val result = engine.calibrate(observations).perDevice.single()

        assertTrue(result.accepted)
        assertTrue(abs(result.fit.clockOffsetNanos - 2_500_000L) < 250_000L)
        assertTrue(result.decodeConfidence > 0.90)
    }

    @Test
    fun lowConfidenceBeaconObservationsAreExcluded() {
        val engine = VisualTimingBeaconCalibrationEngine(
            VisualTimingBeaconCalibrationEngine.Config(
                minimumDecodeConfidence = 0.70,
            ),
        )
        val device = DeviceId("cam-b")

        val observations =
            (0 until 12).map { index ->
                VisualTimingBeaconCalibrationEngine.Observation(
                    deviceId = device,
                    deviceTimestampNanos =
                        2_000_000_000L + index * 33_333_333L,
                    beaconTimestampNanos =
                        2_003_000_000L + index * 33_333_333L,
                    exposureTimeNanos = 6_000_000L,
                    rollingShutterSkewNanos = 0L,
                    normalizedRow = 0.5,
                    decodeConfidence =
                        if (index < 4) 0.20 else 0.95,
                )
            }

        val result = engine.calibrate(observations).perDevice.single()

        assertTrue(result.accepted)
        assertEquals(8, result.acceptedObservationCount)
        assertTrue(result.warnings.none { it == "BEACON_DECODE_CONFIDENCE_LOW" })
    }

    @Test
    fun temporalOutliersAreReportedWithoutCorruptingAcceptedFit() {
        val engine = VisualTimingBeaconCalibrationEngine(
            VisualTimingBeaconCalibrationEngine.Config(
                maximumAcceptedResidualNanos = 4_000_000L,
            ),
        )
        val device = DeviceId("cam-c")

        val observations =
            (0 until 40).map { index ->
                val base =
                    3_000_000_000L +
                        index * 25_000_000L
                val outlier =
                    index == 9 || index == 27
                VisualTimingBeaconCalibrationEngine.Observation(
                    deviceId = device,
                    deviceTimestampNanos = base,
                    beaconTimestampNanos =
                        base +
                            if (outlier) 60_000_000L else 1_750_000L,
                    exposureTimeNanos = 5_000_000L,
                    rollingShutterSkewNanos = 0L,
                    normalizedRow = 0.5,
                    decodeConfidence = 0.97,
                )
            }

        val result = engine.calibrate(observations).perDevice.single()

        assertTrue(result.accepted)
        assertTrue(result.rejectedObservationCount >= 1)
        assertTrue(
            abs(result.fit.clockOffsetNanos - 1_750_000L) <
                1_000_000L,
        )
    }

    @Test
    fun insufficientEvidenceFailsClosed() {
        val engine = VisualTimingBeaconCalibrationEngine()
        val result =
            engine.calibrate(
                listOf(
                    VisualTimingBeaconCalibrationEngine.Observation(
                        deviceId = DeviceId("cam-d"),
                        deviceTimestampNanos = 10_000_000L,
                        beaconTimestampNanos = 12_000_000L,
                        exposureTimeNanos = 4_000_000L,
                        rollingShutterSkewNanos = 0L,
                        normalizedRow = 0.5,
                        decodeConfidence = 0.99,
                    ),
                ),
            ).perDevice.single()

        assertFalse(result.accepted)
        assertTrue(
            result.warnings.contains(
                "CLOCK_BEACON_SAMPLE_COUNT_INSUFFICIENT",
            ),
        )
    }
}
