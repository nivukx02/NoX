package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import org.junit.Assert.assertTrue
import org.junit.Test

class UniversalMotionInferenceOrchestratorTest {

    private fun input(
        frame: Long,
        confidence: Double,
        subjects: Int = 1,
    ): UniversalMotionInferenceOrchestrator.FrameInput =
        UniversalMotionInferenceOrchestrator.FrameInput(
            timestampNanos = frame * 33_333_333L,
            frameIndex = frame,
            motionQuality = MotionCaptureEngine.Quality(
                timestampNanos = frame,
                motionScore = 0.20,
                motionClass = MotionCaptureEngine.MotionClass.NORMAL,
                tuningResponsiveness = 0.7,
                predictionHorizonSeconds = 0.10,
                reacquisitionRadiusNormalized = 0.08,
                meanSpeedNormalizedPerSecond = 0.4,
                maxSpeedNormalizedPerSecond = 1.2,
                meanAccelerationNormalizedPerSecond2 = 0.5,
                maxAccelerationNormalizedPerSecond2 = 1.5,
                meanJerkNormalizedPerSecond3 = 0.2,
                jitterNormalized = 0.05,
                trackedLandmarks = 17,
                predictedLandmarks = 0,
                rejectedLandmarks = 0,
                meanConfidence = confidence,
                occlusionRatio = 0.05,
                temporalStability = 0.82,
                spatialSpread = 0.35,
                stationaryRatio = 0.1,
                dynamicsConfidence = confidence,
                meanInnovationNormalized = 0.02,
                rejectedMeasurementRatio = 0.01,
            ),
            assessment = AdvancedMotionFusionEngine.Assessment(
                timestampNanos = frame,
                dominantSubject = null,
                semanticCoverage = 1.0,
                humanConfidence = confidence,
                animalConfidence = 0.0,
                objectConfidence = 0.0,
                faceCoverage = confidence,
                eyeCoverage = confidence,
                handCoverage = confidence,
                motionComplexity = 0.2,
                occlusionRisk = 0.05,
                attentionConfidence = confidence,
                semanticMotion = SemanticMotionEstimator.GlobalMotion(
                    timestampNanos = frame,
                    confidence = confidence,
                    globalEnergy = 0.12,
                    cameraMotionRisk = 0.02,
                    commonModeVelocityX = 0.0,
                    commonModeVelocityY = 0.0,
                    meanAccelerationNormalizedPerSecond2 = 0.2,
                    temporalStability = 0.85,
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
                subjectCount = subjects,
                activeSubjects = subjects,
                movingSubjects = subjects,
                relativeMotionEnergy = 0.12,
                commonModeEnergy = 0.02,
                meanSpeedNormalizedPerSecond = 0.4,
                maxSpeedNormalizedPerSecond = 1.2,
                maxAccelerationNormalizedPerSecond2 = 1.5,
                meanJerkNormalizedPerSecond3 = 0.2,
                subjects = emptyList(),
            ),
            subjectCount = subjects,
            dominantTrackId = 1L,
            dominantSubjectType = SubjectType.HUMAN,
            lockConfidence = confidence,
        )

    @Test
    fun detectorIsRequiredDuringSearch() {
        val decision = UniversalMotionInferenceOrchestrator().update(input(1, 0.7))
        assertTrue(decision.detectorRequired)
    }

    @Test
    fun extremeMotionForcesDenseInferenceSchedule() {
        val orchestrator = UniversalMotionInferenceOrchestrator()
        val d = orchestrator.update(
            input(1, 0.9).copy(
                motionQuality = input(1, 0.9).motionQuality.copy(
                    motionClass = MotionCaptureEngine.MotionClass.EXTREME,
                ),
            ),
        )
        assertTrue(d.detectorRequired)
        assertTrue(d.depthRequired)
        assertTrue(d.topologyRequired)
        assertTrue(d.reconstructionRequired)
    }
    @Test
    fun ambiguousTrackCompetition_holdsStableIdentity() {
        val orchestrator = UniversalMotionInferenceOrchestrator()
        val first = orchestrator.update(input(1, 0.92))
        assertTrue(first.activeTrackId == 1L)

        val competing = orchestrator.update(
            input(2, 0.92, subjects = 2).copy(
                dominantTrackId = 2L,
                viewpointChallenge = 0.82,
                leftRightAmbiguity = 0.94,
                motionQuality = input(2, 0.92, subjects = 2).motionQuality.copy(
                    temporalStability = 0.78,
                ),
            ),
        )

        assertTrue(competing.activeTrackId == 1L)
        assertTrue(competing.identityAction == "HOLD")
        assertTrue(competing.identitySwapRisk > 0.60)
    }

    @Test
    fun shortIdentityGap_keepsTrackRecoverable() {
        val orchestrator = UniversalMotionInferenceOrchestrator()
        orchestrator.update(input(1, 0.92))

        val recovered = orchestrator.update(
            input(2, 0.40, subjects = 0).copy(
                dominantTrackId = null,
                lockConfidence = 0.08,
            ),
        )

        assertTrue(recovered.activeTrackId == 1L)
        assertTrue(recovered.identityAction == "RECOVER")
    }

    @Test
    fun difficultViewTriggersSelectiveRefinement() {
        val orchestrator = UniversalMotionInferenceOrchestrator()
        val base = input(1, 0.75)
        val decision = orchestrator.update(
            base.copy(
                viewpointChallenge = 0.92,
                leftRightAmbiguity = 0.82,
            ),
        )
        assertTrue(decision.viewRefinementRequired)
        assertTrue(decision.viewRefinementScore >= 0.62)
    }

}
