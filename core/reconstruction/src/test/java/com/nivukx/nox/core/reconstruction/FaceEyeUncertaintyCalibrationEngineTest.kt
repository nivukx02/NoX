package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertTrue
import org.junit.Test

class FaceEyeUncertaintyCalibrationEngineTest {

    private fun healthyInput(
        timestamp: Long,
        faceUncertainty: Double = 0.02,
        eyeUncertainty: Double = 0.012,
    ) =
        FaceEyeUncertaintyCalibrationEngine.Input(
            timestampNanos = timestamp,
            faceConfidence = 0.94,
            faceTemporalConfidence = 0.92,
            faceReprojectionScore = 0.94,
            faceShapeConfidence = 0.93,
            faceFreshness = 0.96,
            faceUncertaintyMeters = faceUncertainty,
            leftEyeConfidence = 0.93,
            rightEyeConfidence = 0.92,
            binocularConfidence = 0.94,
            eyeTemporalConfidence = 0.93,
            gazeAgreement = 0.95,
            angularSpreadDegrees = 1.6,
            angularSpreadRejectDegrees = 12.0,
            eyeUncertaintyMeters = eyeUncertainty,
            ipdConsistencyScore = 0.96,
            faceOodRisk = 0.04,
            eyeOodRisk = 0.03,
            faceAccepted = true,
            eyeAccepted = true,
        )

    @Test
    fun cleanEvidenceReachesCalibrationReadiness() {
        val engine =
            FaceEyeUncertaintyCalibrationEngine(
                FaceEyeUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 6,
                    maximumCalibrationSamples = 24,
                    targetCoverage = 0.95,
                ),
            )

        var result =
            engine.evaluate(
                healthyInput(
                    timestamp = 1_000_000_000L,
                ),
            )

        repeat(14) { index ->
            result =
                engine.evaluate(
                    healthyInput(
                        timestamp =
                            1_033_000_000L +
                                index * 33_000_000L,
                    ),
                )
        }

        assertTrue(result.face.calibrationReady)
        assertTrue(result.eye.calibrationReady)
        assertTrue(result.calibrationConfidence > 0.40)
        assertTrue(
            result.face.calibratedUncertaintyMeters > 0.0,
        )
        assertTrue(
            result.eye.calibratedUncertaintyMeters > 0.0,
        )
    }

    @Test
    fun rejectedOodSamplesCannotTeachCalibration() {
        val engine =
            FaceEyeUncertaintyCalibrationEngine(
                FaceEyeUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 4,
                    maximumCalibrationSamples = 16,
                ),
            )

        repeat(8) { index ->
            engine.evaluate(
                healthyInput(
                    timestamp =
                        2_000_000_000L +
                            index * 33_000_000L,
                ),
            )
        }

        val before = engine.snapshot()

        repeat(8) { index ->
            engine.evaluate(
                healthyInput(
                    timestamp =
                        2_500_000_000L +
                            index * 33_000_000L,
                ).copy(
                    faceOodRisk = 0.98,
                    eyeOodRisk = 0.98,
                    faceAccepted = false,
                    eyeAccepted = false,
                    predictionOnlyFace = true,
                    predictionOnlyEyes = true,
                ),
            )
        }

        val after = engine.snapshot()

        assertTrue(
            after.faceSamples == before.faceSamples,
        )
        assertTrue(
            after.eyeSamples == before.eyeSamples,
        )
        assertTrue(
            after.faceMultiplier <=
                before.faceMultiplier * 1.05,
        )
        assertTrue(
            after.eyeMultiplier <=
                before.eyeMultiplier * 1.05,
        )
    }

    @Test
    fun pathologicalScoresAreRejectedInsteadOfInflatingEnvelope() {
        val engine =
            FaceEyeUncertaintyCalibrationEngine(
                FaceEyeUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 4,
                    maximumCalibrationSamples = 24,
                    maximumAcceptedNonconformity = 6.0,
                ),
            )

        repeat(8) { index ->
            engine.evaluate(
                healthyInput(
                    timestamp =
                        3_000_000_000L +
                            index * 33_000_000L,
                    faceUncertainty = 0.03,
                    eyeUncertainty = 0.016,
                ),
            )
        }

        val before = engine.snapshot()

        val pathological =
            engine.evaluate(
                healthyInput(
                    timestamp = 3_500_000_000L,
                    faceUncertainty = 0.004,
                    eyeUncertainty = 0.004,
                ).copy(
                    faceReprojectionScore = 0.05,
                    faceConfidence = 0.20,
                    eyeUncertaintyMeters = 0.004,
                    angularSpreadDegrees = 11.0,
                    binocularConfidence = 0.24,
                    gazeAgreement = 0.22,
                ),
            )

        val after = engine.snapshot()

        assertTrue(
            pathological.face.usedForCalibration.not(),
        )
        assertTrue(
            pathological.eye.usedForCalibration.not(),
        )
        assertTrue(
            after.faceMultiplier <=
                before.faceMultiplier * 1.15,
        )
        assertTrue(
            after.eyeMultiplier <=
                before.eyeMultiplier * 1.15,
        )
    }

    @Test
    fun equivalentConfigurationPreservesLearnedCalibration() {
        val config =
            FaceEyeUncertaintyCalibrationEngine.Config(
                minimumCalibrationSamples = 4,
            )
        val engine =
            FaceEyeUncertaintyCalibrationEngine(config)

        repeat(10) { index ->
            engine.evaluate(
                healthyInput(
                    timestamp =
                        4_000_000_000L +
                            index * 33_000_000L,
                ),
            )
        }

        val before = engine.snapshot()

        engine.configure(config)

        val after = engine.snapshot()

        assertTrue(
            after.faceSamples == before.faceSamples,
        )
        assertTrue(
            after.eyeSamples == before.eyeSamples,
        )
        assertTrue(
            after.faceMultiplier ==
                before.faceMultiplier,
        )
        assertTrue(
            after.eyeMultiplier ==
                before.eyeMultiplier,
        )
    }

    @Test
    fun coverageIsComputedFromEmpiricalCalibrationScores() {
        val engine =
            FaceEyeUncertaintyCalibrationEngine(
                FaceEyeUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 5,
                    maximumCalibrationSamples = 32,
                    targetCoverage = 0.90,
                    learningAlpha = 1.0,
                ),
            )

        repeat(20) { index ->
            engine.evaluate(
                healthyInput(
                    timestamp =
                        5_000_000_000L +
                            index * 33_000_000L,
                    faceUncertainty =
                        if (index < 18) {
                            0.025
                        } else {
                            0.045
                        },
                    eyeUncertainty =
                        if (index < 18) {
                            0.014
                        } else {
                            0.022
                        },
                ),
            )
        }

        val result =
            engine.evaluate(
                healthyInput(
                    timestamp = 5_800_000_000L,
                    faceUncertainty = 0.03,
                    eyeUncertainty = 0.017,
                ),
            )

        assertTrue(
            result.face.empiricalCoverage in 0.0..1.0,
        )
        assertTrue(
            result.eye.empiricalCoverage in 0.0..1.0,
        )
        assertTrue(
            result.warnings.none {
                it.contains("calibration-warmup")
            },
        )
    }

    @Test
    fun resetRemovesCalibrationEvidence() {
        val engine =
            FaceEyeUncertaintyCalibrationEngine(
                FaceEyeUncertaintyCalibrationEngine.Config(
                    minimumCalibrationSamples = 4,
                ),
            )

        repeat(7) { index ->
            engine.evaluate(
                healthyInput(
                    timestamp =
                        6_000_000_000L +
                            index * 33_000_000L,
                ),
            )
        }

        check(engine.snapshot().faceSamples > 0)
        check(engine.snapshot().eyeSamples > 0)

        engine.reset()

        val snapshot = engine.snapshot()

        assertTrue(snapshot.faceSamples == 0)
        assertTrue(snapshot.eyeSamples == 0)
        assertTrue(snapshot.faceMultiplier == 1.0)
        assertTrue(snapshot.eyeMultiplier == 1.0)
    }
}

fun main() {
    FaceEyeUncertaintyCalibrationEngineTest().cleanEvidenceReachesCalibrationReadiness()
    FaceEyeUncertaintyCalibrationEngineTest().rejectedOodSamplesCannotTeachCalibration()
    FaceEyeUncertaintyCalibrationEngineTest().pathologicalScoresAreRejectedInsteadOfInflatingEnvelope()
    FaceEyeUncertaintyCalibrationEngineTest().equivalentConfigurationPreservesLearnedCalibration()
    FaceEyeUncertaintyCalibrationEngineTest().coverageIsComputedFromEmpiricalCalibrationScores()
    FaceEyeUncertaintyCalibrationEngineTest().resetRemovesCalibrationEvidence()
}
