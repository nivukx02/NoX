package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveFaceEyeEvidenceFusionEngineTest {

    private val engine = AdaptiveFaceEyeEvidenceFusionEngine()

    @Test
    fun strongFaceWeakEyesKeepsHeadConfidenceWhileReducingEyeInfluence() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_000_000_000L,
                    faceGeometryConfidence = 0.94,
                    faceTemporalConfidence = 0.91,
                    denseFaceMeshConfidence = 0.92,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.94,
                    leftEyeConfidence = 0.24,
                    rightEyeConfidence = 0.18,
                    eyeTemporalConfidence = 0.20,
                    binocularConfidence = 0.18,
                    gazeStability = 0.22,
                    gazeAgreement = 0.30,
                    saccadeVelocityDegreesPerSecond = 1100.0,
                    eyeUncertaintyMeters = 0.090,
                    blinkProbability = 0.0,
                ),
            )

        assertTrue(result.faceWeight > result.eyeWeight)
        assertTrue(result.confidence > 0.60)
        assertTrue(
            result.state ==
                AdaptiveFaceEyeEvidenceFusionEngine.State.EYE_DEGRADED ||
                result.state ==
                AdaptiveFaceEyeEvidenceFusionEngine.State.RAPID_GAZE,
        )
    }

    @Test
    fun strongBinocularGeometryRaisesEyeContribution() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_033_000_000L,
                    faceGeometryConfidence = 0.88,
                    faceTemporalConfidence = 0.87,
                    denseFaceMeshConfidence = 0.84,
                    faceFreshness = 0.98,
                    faceReprojectionScore = 0.90,
                    leftEyeConfidence = 0.93,
                    rightEyeConfidence = 0.92,
                    eyeTemporalConfidence = 0.91,
                    binocularConfidence = 0.96,
                    gazeStability = 0.94,
                    gazeAgreement = 0.98,
                    saccadeVelocityDegreesPerSecond = 120.0,
                    eyeUncertaintyMeters = 0.008,
                    blinkProbability = 0.06,
                ),
            )

        assertTrue(result.eyeWeight > result.faceWeight * 0.45)
        assertTrue(result.bilateralEyeSupport > 0.85)
        assertTrue(result.confidence > 0.75)
        assertEquals(
            AdaptiveFaceEyeEvidenceFusionEngine.State.STABLE,
            result.state,
        )
    }

    @Test
    fun staleFaceEvidenceIsDownweightedInsteadOfDeletingEyeEvidence() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_066_000_000L,
                    faceGeometryConfidence = 0.92,
                    faceTemporalConfidence = 0.90,
                    denseFaceMeshConfidence = 0.90,
                    faceFreshness = 0.18,
                    faceReprojectionScore = 0.94,
                    leftEyeConfidence = 0.91,
                    rightEyeConfidence = 0.90,
                    eyeTemporalConfidence = 0.90,
                    binocularConfidence = 0.92,
                    gazeStability = 0.91,
                    gazeAgreement = 0.94,
                    saccadeVelocityDegreesPerSecond = 150.0,
                    eyeUncertaintyMeters = 0.010,
                    blinkProbability = 0.02,
                ),
            )

        assertTrue(result.faceWeight < result.eyeWeight)
        assertTrue(result.eyeWeight > 0.30)
        assertTrue(
            result.warnings.any {
                it.contains("face-stale")
            },
        )
    }

    @Test
    fun rapidGazeDoesNotDiscardValidFaceGeometry() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_099_000_000L,
                    faceGeometryConfidence = 0.90,
                    faceTemporalConfidence = 0.92,
                    denseFaceMeshConfidence = 0.88,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.92,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.89,
                    eyeTemporalConfidence = 0.62,
                    binocularConfidence = 0.88,
                    gazeStability = 0.58,
                    gazeAgreement = 0.91,
                    saccadeVelocityDegreesPerSecond = 820.0,
                    eyeUncertaintyMeters = 0.014,
                    blinkProbability = 0.02,
                ),
            )

        assertTrue(
            result.warnings.any {
                it.contains("rapid-gaze")
            },
        )
        assertTrue(result.faceWeight > result.eyeWeight)
        assertTrue(result.confidence > 0.62)
    }

    @Test
    fun rejectedFaceShapeLowersFaceContributionWithoutDeletingEyes() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_360_000_000L,
                    faceGeometryConfidence = 0.90,
                    faceTemporalConfidence = 0.90,
                    denseFaceMeshConfidence = 0.82,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.92,
                    leftEyeConfidence = 0.92,
                    rightEyeConfidence = 0.91,
                    eyeTemporalConfidence = 0.91,
                    binocularConfidence = 0.94,
                    gazeStability = 0.90,
                    gazeAgreement = 0.96,
                    saccadeVelocityDegreesPerSecond = 180.0,
                    eyeUncertaintyMeters = 0.010,
                    blinkProbability = 0.02,
                    faceShapeConfidence = 0.18,
                    faceShapeAccepted = false,
                    denseFaceMeshObserved = true,
                ),
            )

        assertTrue(result.faceHardFailure)
        assertTrue(result.eyeHardFailure.not())
        assertTrue(result.eyeWeight > result.faceWeight)
        assertTrue(
            result.warnings.any {
                it.contains("face-shape-gate")
            },
        )
    }

    @Test
    fun rejectedEyeGatesPreserveFaceContribution() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_393_000_000L,
                    faceGeometryConfidence = 0.94,
                    faceTemporalConfidence = 0.93,
                    denseFaceMeshConfidence = 0.92,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.94,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.89,
                    eyeTemporalConfidence = 0.90,
                    binocularConfidence = 0.16,
                    gazeStability = 0.82,
                    gazeAgreement = 0.22,
                    saccadeVelocityDegreesPerSecond = 220.0,
                    eyeUncertaintyMeters = 0.080,
                    blinkProbability = 0.02,
                    eyeTemporalAccepted = false,
                    binocularAccepted = false,
                ),
            )

        assertTrue(result.eyeHardFailure)
        assertTrue(result.faceHardFailure.not())
        assertTrue(result.faceWeight > result.eyeWeight)
        assertTrue(
            result.warnings.any {
                it.contains("eye-temporal-gate-rejected") ||
                    it.contains("binocular-gate-rejected")
            },
        )
    }

    @Test
    fun rapidGazeAccelerationIsSoftUntilHardThreshold() {
        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_426_000_000L,
                    faceGeometryConfidence = 0.91,
                    faceTemporalConfidence = 0.90,
                    denseFaceMeshConfidence = 0.89,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.93,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.89,
                    eyeTemporalConfidence = 0.86,
                    binocularConfidence = 0.91,
                    gazeStability = 0.84,
                    gazeAgreement = 0.94,
                    saccadeVelocityDegreesPerSecond = 760.0,
                    eyeUncertaintyMeters = 0.014,
                    blinkProbability = 0.01,
                    gazeAccelerationDegreesPerSecond2 = 5200.0,
                ),
            )

        assertTrue(result.eyeDynamicConsistency < 1.0)
        assertTrue(result.eyeDynamicConsistency > 0.30)
        assertTrue(result.eyeHardFailure.not())
    }

    @Test
    fun repeatedEyeFailuresEnterHystereticDegradedState() {
        repeat(4) { index ->
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos =
                        1_132_000_000L +
                            index * 33_000_000L,
                    faceGeometryConfidence = 0.91,
                    faceTemporalConfidence = 0.90,
                    denseFaceMeshConfidence = 0.88,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.93,
                    leftEyeConfidence = 0.18,
                    rightEyeConfidence = 0.17,
                    eyeTemporalConfidence = 0.16,
                    binocularConfidence = 0.12,
                    gazeStability = 0.14,
                    gazeAgreement = 0.20,
                    saccadeVelocityDegreesPerSecond = 900.0,
                    eyeUncertaintyMeters = 0.110,
                    blinkProbability = 0.0,
                ),
            )
        }

        val result =
            engine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_300_000_000L,
                    faceGeometryConfidence = 0.91,
                    faceTemporalConfidence = 0.90,
                    denseFaceMeshConfidence = 0.88,
                    faceFreshness = 1.0,
                    faceReprojectionScore = 0.93,
                    leftEyeConfidence = 0.18,
                    rightEyeConfidence = 0.17,
                    eyeTemporalConfidence = 0.16,
                    binocularConfidence = 0.12,
                    gazeStability = 0.14,
                    gazeAgreement = 0.20,
                    saccadeVelocityDegreesPerSecond = 900.0,
                    eyeUncertaintyMeters = 0.110,
                    blinkProbability = 0.0,
                ),
            )

        assertTrue(result.eyeDegradedPersistently)
        assertEquals(
            AdaptiveFaceEyeEvidenceFusionEngine.State.EYE_DEGRADED,
            result.state,
        )
    }
}
