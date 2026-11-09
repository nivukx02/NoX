package com.nivukx.nox.core.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class MotionPolicyEvidenceDeriverTest {

    private fun quality(
        confidence: Double = 0.95,
        occlusionRatio: Double = 0.05,
        temporalStability: Double = 0.92,
    ) = MotionCaptureEngine.Quality(
        timestampNanos = 1L,
        motionScore = 0.20,
        motionClass = MotionCaptureEngine.MotionClass.NORMAL,
        tuningResponsiveness = 0.80,
        predictionHorizonSeconds = 0.10,
        reacquisitionRadiusNormalized = 0.08,
        meanSpeedNormalizedPerSecond = 0.30,
        maxSpeedNormalizedPerSecond = 0.90,
        meanAccelerationNormalizedPerSecond2 = 0.40,
        maxAccelerationNormalizedPerSecond2 = 1.20,
        meanJerkNormalizedPerSecond3 = 0.20,
        jitterNormalized = 0.04,
        trackedLandmarks = 17,
        predictedLandmarks = 0,
        rejectedLandmarks = 0,
        meanConfidence = confidence,
        occlusionRatio = occlusionRatio,
        temporalStability = temporalStability,
        spatialSpread = 0.40,
        stationaryRatio = 0.10,
        dynamicsConfidence = confidence,
        meanInnovationNormalized = 0.02,
        rejectedMeasurementRatio = 0.01,
    )

    private fun assessment(
        semanticCoverage: Double = 0.96,
        occlusionRisk: Double = 0.05,
        attentionConfidence: Double = 0.95,
        cameraMotionRisk: Double = 0.04,
    ) = AdvancedMotionFusionEngine.Assessment(
        timestampNanos = 1L,
        dominantSubject = null,
        semanticCoverage = semanticCoverage,
        humanConfidence = 0.95,
        animalConfidence = 0.0,
        objectConfidence = 0.0,
        faceCoverage = 0.90,
        eyeCoverage = 0.90,
        handCoverage = 0.90,
        motionComplexity = 0.20,
        occlusionRisk = occlusionRisk,
        attentionConfidence = attentionConfidence,
        semanticMotion = SemanticMotionEstimator.GlobalMotion(
            timestampNanos = 1L,
            confidence = 0.95,
            globalEnergy = 0.20,
            cameraMotionRisk = cameraMotionRisk,
            commonModeVelocityX = 0.0,
            commonModeVelocityY = 0.0,
            meanAccelerationNormalizedPerSecond2 = 0.20,
            temporalStability = 0.92,
            subjectCount = 1,
            activeSubjects = 1,
            movingSubjects = 1,
            relativeMotionEnergy = 0.20,
            commonModeEnergy = 0.02,
            meanSpeedNormalizedPerSecond = 0.30,
            maxSpeedNormalizedPerSecond = 0.90,
            maxAccelerationNormalizedPerSecond2 = 1.20,
            meanJerkNormalizedPerSecond3 = 0.20,
            subjects = emptyList(),
        ),
        subjectCount = 1,
        activeSubjects = 1,
        movingSubjects = 1,
        relativeMotionEnergy = 0.20,
        commonModeEnergy = 0.02,
        meanSpeedNormalizedPerSecond = 0.30,
        maxSpeedNormalizedPerSecond = 0.90,
        maxAccelerationNormalizedPerSecond2 = 1.20,
        meanJerkNormalizedPerSecond3 = 0.20,
        subjects = emptyList(),
    )

    @Test
    fun clearStableEvidence_staysLowRisk() {
        val evidence = MotionPolicyEvidenceDeriver.derive(
            assessment = assessment(),
            motionQuality = quality(),
        )

        assertTrue(evidence.viewpointChallenge < 0.20)
        assertTrue(evidence.leftRightAmbiguity < 0.20)
    }

    @Test
    fun occlusionAndCameraMotion_raiseViewpointChallenge() {
        val evidence = MotionPolicyEvidenceDeriver.derive(
            assessment = assessment(
                semanticCoverage = 0.62,
                occlusionRisk = 0.88,
                attentionConfidence = 0.65,
                cameraMotionRisk = 0.84,
            ),
            motionQuality = quality(
                confidence = 0.70,
                occlusionRatio = 0.72,
                temporalStability = 0.55,
            ),
        )

        assertTrue(evidence.viewpointChallenge > 0.55)
        assertTrue(evidence.leftRightAmbiguity > 0.50)
    }
}
