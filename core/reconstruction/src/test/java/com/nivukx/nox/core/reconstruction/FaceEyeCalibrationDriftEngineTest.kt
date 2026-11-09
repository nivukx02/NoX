package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertTrue
import org.junit.Test

class FaceEyeCalibrationDriftEngineTest {

    @Test
    fun stableHealthyEvidenceDoesNotTriggerDrift() {
        val engine = FaceEyeCalibrationDriftEngine()

        repeat(12) { index ->
            val result = engine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = 1_000_000_000L + index * 33_000_000L,
                    interpupillaryDistanceMeters = 0.064,
                    angularDisagreementDegrees = 1.8,
                    reprojectionErrorPx = 1.6,
                    binocularConfidence = 0.94,
                    faceEyeFusionConfidence = 0.93,
                ),
            )

            assertTrue(result.driftScore < 0.40)
            assertTrue(!result.recalibrationRecommended)
        }
    }

    @Test
    fun gradualIpdAndAngularDriftBecomesPersistentCalibrationWarning() {
        val engine = FaceEyeCalibrationDriftEngine(
            FaceEyeCalibrationDriftEngine.Config(
                warmupSamples = 4,
                persistentWarningFrames = 3,
                recoveryFrames = 4,
            ),
        )

        repeat(4) { index ->
            engine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = 2_000_000_000L + index * 33_000_000L,
                    interpupillaryDistanceMeters = 0.064,
                    angularDisagreementDegrees = 1.8,
                    reprojectionErrorPx = 1.6,
                    binocularConfidence = 0.94,
                    faceEyeFusionConfidence = 0.93,
                ),
            )
        }

        var final = engine.evaluate(
            FaceEyeCalibrationDriftEngine.Input(
                timestampNanos = 2_500_000_000L,
                interpupillaryDistanceMeters = 0.079,
                angularDisagreementDegrees = 11.0,
                reprojectionErrorPx = 5.8,
                binocularConfidence = 0.48,
                faceEyeFusionConfidence = 0.52,
            ),
        )

        repeat(4) { index ->
            final = engine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = 2_533_000_000L + index * 33_000_000L,
                    interpupillaryDistanceMeters = 0.080,
                    angularDisagreementDegrees = 11.2,
                    reprojectionErrorPx = 6.0,
                    binocularConfidence = 0.46,
                    faceEyeFusionConfidence = 0.50,
                ),
            )
        }

        assertTrue(final.driftScore > 0.55)
        assertTrue(final.persistent)
        assertTrue(final.recalibrationRecommended)
        assertTrue(
            final.warnings.any {
                it.contains("recalibration") ||
                    it.contains("calibration-drift")
            },
        )
    }

    @Test
    fun noisySingleFrameCannotPoisonTheDriftBaseline() {
        val engine = FaceEyeCalibrationDriftEngine(
            FaceEyeCalibrationDriftEngine.Config(
                warmupSamples = 5,
                persistentWarningFrames = 3,
            ),
        )

        repeat(5) { index ->
            engine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = 3_000_000_000L + index * 33_000_000L,
                    interpupillaryDistanceMeters = 0.064,
                    angularDisagreementDegrees = 1.6,
                    reprojectionErrorPx = 1.5,
                    binocularConfidence = 0.95,
                    faceEyeFusionConfidence = 0.94,
                ),
            )
        }

        val bad = engine.evaluate(
            FaceEyeCalibrationDriftEngine.Input(
                timestampNanos = 3_200_000_000L,
                interpupillaryDistanceMeters = 0.110,
                angularDisagreementDegrees = 30.0,
                reprojectionErrorPx = 18.0,
                binocularConfidence = 0.05,
                faceEyeFusionConfidence = 0.08,
            ),
        )

        val recovered = engine.evaluate(
            FaceEyeCalibrationDriftEngine.Input(
                timestampNanos = 3_233_000_000L,
                interpupillaryDistanceMeters = 0.064,
                angularDisagreementDegrees = 1.7,
                reprojectionErrorPx = 1.6,
                binocularConfidence = 0.94,
                faceEyeFusionConfidence = 0.93,
            ),
        )

        assertTrue(bad.driftScore > recovered.driftScore)
        assertTrue(recovered.driftScore < 0.55)
    }

    @Test
    fun repeatedHealthyRecoveryClearsPersistentRecommendation() {
        val engine = FaceEyeCalibrationDriftEngine(
            FaceEyeCalibrationDriftEngine.Config(
                warmupSamples = 3,
                persistentWarningFrames = 2,
                recoveryFrames = 3,
            ),
        )

        repeat(3) { index ->
            engine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = 4_000_000_000L + index * 33_000_000L,
                    interpupillaryDistanceMeters = 0.064,
                    angularDisagreementDegrees = 1.7,
                    reprojectionErrorPx = 1.5,
                    binocularConfidence = 0.95,
                    faceEyeFusionConfidence = 0.94,
                ),
            )
        }

        repeat(5) { index ->
            engine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = 4_200_000_000L + index * 33_000_000L,
                    interpupillaryDistanceMeters = 0.081,
                    angularDisagreementDegrees = 12.0,
                    reprojectionErrorPx = 6.0,
                    binocularConfidence = 0.45,
                    faceEyeFusionConfidence = 0.48,
                ),
            )
        }

        repeat(5) { index ->
            val result =
                engine.evaluate(
                    FaceEyeCalibrationDriftEngine.Input(
                        timestampNanos = 4_500_000_000L + index * 33_000_000L,
                        interpupillaryDistanceMeters = 0.064,
                        angularDisagreementDegrees = 1.8,
                        reprojectionErrorPx = 1.6,
                        binocularConfidence = 0.95,
                        faceEyeFusionConfidence = 0.94,
                    ),
                )

            if (index == 4) {
                assertTrue(!result.recalibrationRecommended)
            }
        }
    }
}
