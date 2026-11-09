package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SpatiotemporalCalibrationEngineTest {

    @Test
    fun robustFitRecoversOffsetDriftAndRollingShutterTerm() {
        val engine = SpatiotemporalCalibrationEngine()

        repeat(24) { index ->
            val local = 1_000_000_000L + index * 33_333_333L
            val drift = index * 1_200L
            val row = (index % 10) / 9.0
            val rowBias = (row * 2_000_000.0).toLong()
            val reference = local + 3_000_000L + drift + rowBias

            assertTrue(
                engine.add(
                    SpatiotemporalCalibrationEngine.Observation(
                        cameraId = "cam-a",
                        sequence = index.toLong(),
                        localTimestampNanos = local,
                        referenceTimestampNanos = reference,
                        normalizedRow = row,
                    ),
                ),
            )
        }

        val result = engine.fit("cam-a")
        assertTrue(result.valid)
        assertTrue(result.sampleCount == 24)
        assertTrue(result.residualP95Milliseconds < 1.0)
        assertTrue(result.outlierRatio == 0.0)
        assertTrue(result.temporalConfidence > 0.90)
    }

    @Test
    fun staleEvidenceIsRejectedWithoutMutatingFitHistory() {
        val engine = SpatiotemporalCalibrationEngine()

        repeat(8) { index ->
            val timestamp = 2_000_000_000L + index * 10_000_000L
            assertTrue(
                engine.add(
                    SpatiotemporalCalibrationEngine.Observation(
                        cameraId = "cam-b",
                        sequence = index.toLong(),
                        localTimestampNanos = timestamp,
                        referenceTimestampNanos = timestamp + 1_000_000L,
                    ),
                ),
            )
        }

        assertFalse(
            engine.add(
                SpatiotemporalCalibrationEngine.Observation(
                    cameraId = "cam-b",
                    sequence = 3L,
                    localTimestampNanos = 2_010_000_000L,
                    referenceTimestampNanos = 2_011_000_000L,
                ),
            ),
        )

        val result = engine.fit("cam-b")
        assertTrue(result.sampleCount == 8)
        assertTrue(result.staleEvidenceRejected == 1)
        assertTrue(result.warningCodes.contains("stale-temporal-evidence-rejected"))
    }

    @Test
    fun timestampCorrectionIncludesRowAndCalibratedClockTerms() {
        val engine = SpatiotemporalCalibrationEngine()

        repeat(20) { index ->
            val local = 7_000_000_000L + index * 50_000_000L
            val reference = local + 4_000_000L + index * 2_000L
            assertTrue(
                engine.add(
                    SpatiotemporalCalibrationEngine.Observation(
                        cameraId = "cam-warp",
                        sequence = index.toLong(),
                        localTimestampNanos = local,
                        referenceTimestampNanos = reference,
                        normalizedRow = 1.0,
                    ),
                ),
            )
        }

        val result = engine.fit("cam-warp")
        val correction = result.correctTimestamp(
            localTimestampNanos = 7_500_000_000L,
            normalizedRow = 1.0,
            exposureDurationNanos = 20_000_000L,
        )

        assertTrue(
            correction.correctedReferenceTimestampNanos >
                correction.localTimestampNanos,
        )
        assertTrue(
            correction.rowOffsetMilliseconds.isFinite(),
        )
        assertTrue(
            correction.confidence in 0.0..1.0,
        )
    }


    @Test
    fun residualAndOutlierContractsCanHardFail() {
        val engine = SpatiotemporalCalibrationEngine(
            SpatiotemporalCalibrationEngine.Config(
                warningResidualP95Milliseconds = 1.0,
                hardResidualP95Milliseconds = 2.0,
                hardOutlierRatio = 0.10,
            ),
        )

        repeat(7) { index ->
            val local = 3_000_000_000L + index * 20_000_000L
            val largeError = if (index == 6) 30_000_000L else 0L
            assertTrue(
                engine.add(
                    SpatiotemporalCalibrationEngine.Observation(
                        cameraId = "cam-c",
                        sequence = index.toLong(),
                        localTimestampNanos = local,
                        referenceTimestampNanos = local + 1_000_000L + largeError,
                    ),
                ),
            )
        }

        val result = engine.fit("cam-c")
        assertFalse(result.valid)
        assertTrue(result.hardFailure)
        assertTrue(result.warningCodes.any { it.contains("hard-failure") })
    }
}
