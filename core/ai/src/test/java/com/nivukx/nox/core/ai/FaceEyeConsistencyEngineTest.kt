package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeState
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.FaceEyeFusionState
import com.nivukx.nox.core.model.PerceptionSource
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.GazeConfidenceBand
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceEyeConsistencyEngineTest {

    @Test
    fun coherentFaceEyeStateRemainsHighConfidence() {
        val engine = FaceEyeConsistencyEngine()

        val result = engine.evaluate(
            FaceEyeConsistencyEngine.Input(
                face = face(
                    gaze = Vec3(0.01, 0.01, 0.999),
                    confidence = 0.94,
                ),
                eyes = eye(
                    gaze = Vec3(0.01, 0.01, 0.999),
                    confidence = 0.92,
                ),
                face3DConfidence = 0.95,
                face3DErrorPx = 1.4,
                timestampNanos = 1_000_000_000L,
            ),
        )

        assertTrue(result.reliable)
        assertTrue(result.confidence > 0.70)
        assertTrue(result.headGazeConsistency > 0.90)
        assertTrue(result.temporalStability > 0.60)
    }

    @Test
    fun severeHeadGazeDisagreementIsFlagged() {
        val engine = FaceEyeConsistencyEngine()

        val result = engine.evaluate(
            FaceEyeConsistencyEngine.Input(
                face = face(
                    gaze = Vec3(1.0, 0.0, 0.0),
                    confidence = 0.95,
                ),
                eyes = eye(
                    gaze = Vec3(1.0, 0.0, 0.0),
                    confidence = 0.94,
                ),
                face3DConfidence = 0.92,
                face3DErrorPx = 1.8,
                timestampNanos = 2_000_000_000L,
            ),
        )

        assertTrue(result.confidence < 0.65)
        assertTrue(
            result.warnings.any {
                it.contains("gaze/head") || it.startsWith("fatal:")
            },
        )
    }

    @Test
    fun blinkDisagreementReducesConsistencyWithoutDestroyingHeadState() {
        val engine = FaceEyeConsistencyEngine()

        val result = engine.evaluate(
            FaceEyeConsistencyEngine.Input(
                face = face(
                    gaze = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.90,
                    leftOpenness = 0.90,
                    rightOpenness = 0.90,
                ),
                eyes = eye(
                    gaze = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.88,
                    leftBlink = 0.98,
                    rightBlink = 0.96,
                ),
                face3DConfidence = 0.90,
                face3DErrorPx = 2.0,
                timestampNanos = 3_000_000_000L,
            ),
        )

        assertTrue(result.blinkConsistency < 0.80)
        assertTrue(result.faceConfidence > 0.50)
        assertTrue(result.warnings.any { it.contains("blink") })
    }

    @Test
    fun prolongedFaceGapResetsTemporalHistory() {
        val engine =
            FaceEyeConsistencyEngine(
                FaceEyeConsistencyEngine.Config(
                    maximumTrackGapSeconds = 0.25,
                ),
            )

        val first =
            engine.evaluate(
                FaceEyeConsistencyEngine.Input(
                    face = face(
                        gaze = Vec3(0.0, 0.0, 1.0),
                        confidence = 0.94,
                    ),
                    eyes = eye(
                        gaze = Vec3(0.0, 0.0, 1.0),
                        confidence = 0.92,
                    ),
                    face3DConfidence = 0.95,
                    face3DErrorPx = 1.5,
                    timestampNanos = 1_000_000_000L,
                ),
            )

        val second =
            engine.evaluate(
                FaceEyeConsistencyEngine.Input(
                    face = face(
                        gaze = Vec3(0.30, 0.0, 0.954),
                        confidence = 0.94,
                    ),
                    eyes = eye(
                        gaze = Vec3(0.30, 0.0, 0.954),
                        confidence = 0.92,
                    ),
                    face3DConfidence = 0.95,
                    face3DErrorPx = 1.5,
                    timestampNanos = 1_600_000_000L,
                ),
            )

        assertTrue(first.temporalHistoryReset)
        assertTrue(second.temporalHistoryReset)
        assertTrue(second.gazeAngularVelocityDegreesPerSecond < 200.0)
    }

    @Test
    fun highFusionUncertaintyReducesReliabilityAndIsSurfaced() {
        val engine = FaceEyeConsistencyEngine()

        val uncertainty =
            FaceEyeFusionEngine.UncertaintyDecomposition(
                aleatoric = 0.72,
                temporal = 0.68,
                binocular = 0.74,
                innovation = 0.81,
                staleEvidence = 0.10,
                modelMismatch = 0.76,
                outOfDistributionRisk = 0.82,
                total = 0.76,
                trustworthy = false,
            )

        val result =
            engine.evaluate(
                FaceEyeConsistencyEngine.Input(
                    face = face(
                        gaze = Vec3(0.0, 0.0, 1.0),
                        confidence = 0.92,
                    ),
                    eyes = eye(
                        gaze = Vec3(0.0, 0.0, 1.0),
                        confidence = 0.90,
                    ),
                    face3DConfidence = 0.94,
                    face3DErrorPx = 1.5,
                    timestampNanos = 5_000_000_000L,
                    fusionUncertainty = uncertainty,
                ),
            )

        assertTrue(result.uncertaintyTotal == 0.76)
        assertTrue(result.uncertaintyOodRisk == 0.82)
        assertTrue(!result.uncertaintyTrustworthy)
        assertTrue(
            result.warnings.any {
                it.contains("uncertainty")
            },
        )
        assertTrue(
            result.warnings.any {
                it.contains("out-of-distribution")
            },
        )
        assertTrue(result.confidence < 0.90)
    }


    @Test
    fun poor3DFaceEvidenceReducesFinalConfidence() {
        val engine = FaceEyeConsistencyEngine()

        val result = engine.evaluate(
            FaceEyeConsistencyEngine.Input(
                face = face(
                    gaze = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.92,
                ),
                eyes = eye(
                    gaze = Vec3(0.0, 0.0, 1.0),
                    confidence = 0.90,
                ),
                face3DConfidence = 0.25,
                face3DErrorPx = 12.0,
                timestampNanos = 4_000_000_000L,
            ),
        )

        assertTrue(result.face3DConsistency < 0.50)
        assertTrue(result.confidence < 0.75)
        assertTrue(
            result.warnings.any {
                it.contains("3D face")
            },
        )
    }

    private fun face(
        gaze: Vec3,
        confidence: Double,
        leftOpenness: Double = 0.90,
        rightOpenness: Double = 0.90,
    ) =
        FaceEyeFusionState(
            timestampNanos = 1_000_000_000L,
            trackId = 42L,
            headRotation = Quat.IDENTITY,
            headYawDegrees = 0.0,
            headRollDegrees = 0.0,
            combinedGazeCamera = gaze,
            leftGazeCamera = gaze,
            rightGazeCamera = gaze,
            leftOpenness = leftOpenness,
            rightOpenness = rightOpenness,
            blinkProbability = maxOf(
                1.0 - leftOpenness,
                1.0 - rightOpenness,
            ),
            gazeStability = 0.90,
            saccadeVelocityDegreesPerSecond = 15.0,
            vergenceMeters = 0.60,
            confidence = confidence,
            fresh = true,
            source = PerceptionSource.MULTI_VIEW,
        )

    private fun eye(
        gaze: Vec3,
        confidence: Double,
        leftBlink: Double = 0.05,
        rightBlink: Double = 0.05,
    ) =
        EyeState(
            timestampNanos = 1_000_000_000L,
            trackId = 42L,
            leftGazeCamera = gaze,
            rightGazeCamera = gaze,
            fusedGazeCamera = gaze,
            leftOpenness = 1.0 - leftBlink,
            rightOpenness = 1.0 - rightBlink,
            leftBlinkProbability = leftBlink,
            rightBlinkProbability = rightBlink,
            vergenceMeters = 0.60,
            saccadeVelocityDegPerSecond = 18.0,
            gazeStability = 0.92,
            confidence = confidence,
            band = GazeConfidenceBand.EXCELLENT,
            source = PerceptionSource.MULTI_VIEW,
            fresh = true,
        )
}
