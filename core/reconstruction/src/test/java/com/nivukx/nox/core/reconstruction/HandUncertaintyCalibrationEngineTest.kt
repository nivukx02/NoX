package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.Hand3DLandmark
import com.nivukx.nox.core.model.Hand3DResult
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HandUncertaintyCalibrationEngineTest {

    @Test
    fun calibrationWarmsUpWithoutRejectingHealthyHands() {
        val engine =
            HandUncertaintyCalibrationEngine(
                HandUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 4,
                ),
            )

        var result =
            engine.evaluate(
                frame(
                    timestampNanos = 1_000_000_000L,
                    uncertainty = 0.010,
                    reprojectionErrorPx = 0.8,
                ),
            )

        assertTrue(result.policy != HandUncertaintyCalibrationEngine.Policy.REJECT)

        repeat(6) { index ->
            result =
                engine.evaluate(
                    frame(
                        timestampNanos =
                            1_033_000_000L +
                                index * 33_000_000L,
                        uncertainty = 0.011,
                        reprojectionErrorPx = 0.9,
                    ),
                )
        }

        assertTrue(result.calibrationReady)
        assertTrue(result.uncertaintyConfidence > 0.55)
        assertTrue(result.meanRadiusMeters < 0.10)
    }

    @Test
    fun currentExtremeOutlierDoesNotWidenItsOwnHistory() {
        val engine =
            HandUncertaintyCalibrationEngine(
                HandUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 4,
                    maximumRadiusMeters = 0.20,
                ),
            )

        repeat(8) { index ->
            engine.evaluate(
                frame(
                    timestampNanos =
                        2_000_000_000L +
                            index * 33_000_000L,
                    uncertainty = 0.012,
                    reprojectionErrorPx = 0.9,
                ),
            )
        }

        val before =
            engine.snapshot()?.meanRadiusMeters ?: 0.0

        val outlier =
            engine.evaluate(
                frame(
                    timestampNanos = 2_300_000_000L,
                    uncertainty = 0.70,
                    reprojectionErrorPx = 48.0,
                ),
            )

        assertTrue(
            outlier.outOfDistributionRisk > 0.60,
        )
        assertTrue(
            outlier.meanRadiusMeters >
                before,
        )

        val recovery =
            engine.evaluate(
                frame(
                    timestampNanos = 2_333_000_000L,
                    uncertainty = 0.012,
                    reprojectionErrorPx = 0.9,
                ),
            )

        assertTrue(
            recovery.meanRadiusMeters <
                outlier.meanRadiusMeters,
        )
    }

    @Test
    fun repeatedOutliersReachProductionReject() {
        val engine =
            HandUncertaintyCalibrationEngine(
                HandUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 3,
                    oodThreshold = 0.70,
                    holdOodThreshold = 0.40,
                ),
            )

        repeat(7) { index ->
            engine.evaluate(
                frame(
                    timestampNanos =
                        3_000_000_000L +
                            index * 33_000_000L,
                    uncertainty = 0.010,
                    reprojectionErrorPx = 0.8,
                ),
            )
        }

        var result =
            engine.evaluate(
                frame(
                    timestampNanos = 3_300_000_000L,
                    uncertainty = 0.70,
                    reprojectionErrorPx = 50.0,
                ),
            )

        repeat(4) { index ->
            result =
                engine.evaluate(
                    frame(
                        timestampNanos =
                            3_333_000_000L +
                                index * 33_000_000L,
                        uncertainty = 0.72,
                        reprojectionErrorPx = 52.0,
                    ),
                )
        }

        assertTrue(
            result.outOfDistributionRisk >=
                0.70 ||
                result.policy ==
                HandUncertaintyCalibrationEngine.Policy.REJECT,
        )
    }

    @Test
    fun dynamicAndOcclusionInflationAreBounded() {
        val engine = HandUncertaintyCalibrationEngine()

        val result =
            engine.evaluate(
                frame(
                    timestampNanos = 4_000_000_000L,
                    uncertainty = 0.020,
                    reprojectionErrorPx = 1.1,
                ),
                dynamicStress = 1.0,
                occlusionRisk = 1.0,
                synchronizationConfidence = 0.0,
            )

        assertTrue(result.dynamicInflation <= 0.40)
        assertTrue(result.occlusionInflation <= 0.45)
        assertTrue(result.synchronizationInflation <= 0.30)
        assertTrue(result.meanRadiusMeters <= 0.22)
    }

    @Test
    fun resetClearsHandCalibration() {
        val engine =
            HandUncertaintyCalibrationEngine(
                HandUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 3,
                ),
            )

        repeat(4) { index ->
            engine.evaluate(
                frame(
                    timestampNanos =
                        5_000_000_000L +
                            index * 33_000_000L,
                    uncertainty = 0.012,
                    reprojectionErrorPx = 0.8,
                ),
            )
        }

        assertTrue(
            engine.snapshot()?.calibrationReady == true,
        )

        engine.reset()

        assertTrue(engine.snapshot() == null)

        val fresh =
            engine.evaluate(
                frame(
                    timestampNanos = 5_500_000_000L,
                    uncertainty = 0.012,
                    reprojectionErrorPx = 0.8,
                ),
            )

        assertTrue(!fresh.calibrationReady)
    }


    @Test
    fun stale_hand_frame_does_not_contaminate_empirical_uncertainty_history() {
        val engine =
            HandUncertaintyCalibrationEngine(
                HandUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 3,
                ),
            )

        repeat(4) { index ->
            engine.evaluate(
                frame(
                    timestampNanos =
                        6_000_000_000L +
                            index * 33_000_000L,
                    uncertainty = 0.012,
                    reprojectionErrorPx = 0.8,
                ),
            )
        }

        val before = engine.snapshot()
        val stale = engine.evaluate(
            frame(
                timestampNanos = 5_800_000_000L,
                uncertainty = 0.70,
                reprojectionErrorPx = 50.0,
            ),
        )

        assertTrue(before != null)
        assertTrue(
            stale.warnings.any {
                it.contains("stale")
            },
        )
        assertTrue(
            stale.meanRadiusMeters ==
                before!!.meanRadiusMeters,
        )
    }

    private fun frame(
        timestampNanos: Long,
        uncertainty: Double,
        reprojectionErrorPx: Double,
    ): Hand3DFrame =
        Hand3DFrame(
            timestampNanos = timestampNanos,
            hands = listOf(
                Hand3DResult(
                    handId = 11L,
                    side = HandSide.RIGHT,
                    timestampNanos = timestampNanos,
                    landmarks =
                        (0 until 21).map {
                            Hand3DLandmark(
                                index = it,
                                position =
                                    Vec3(
                                        it * 0.006,
                                        0.01 * it,
                                        0.0,
                                    ),
                                confidence = 0.94,
                                uncertaintyMeters = uncertainty,
                                observationCount = 5,
                            )
                        },
                    confidence = 0.94,
                    meanUncertaintyMeters = uncertainty,
                    reprojectionErrorPx = reprojectionErrorPx,
                ),
            ),
            confidence = 0.94,
        )
}
