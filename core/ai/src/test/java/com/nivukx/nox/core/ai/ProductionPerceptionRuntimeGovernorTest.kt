package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertTrue

class ProductionPerceptionRuntimeGovernorTest {

    @Test
    fun highMotionReceivesAdditionalMotionPriority() {
        val governor = ProductionPerceptionRuntimeGovernor()
        governor.updateSystemLoad(
            thermalScale = 1.0,
            memoryScale = 1.0,
            gpuScale = 1.0,
            inferenceScale = 1.0,
        )

        val low = SemanticMotionEstimator.GlobalMotion(
            timestampNanos = 1L,
            subjectCount = 1,
            activeSubjects = 1,
            movingSubjects = 0,
            globalEnergy = 0.02,
            relativeMotionEnergy = 0.01,
            commonModeEnergy = 0.0,
            commonModeVelocityX = 0.0,
            commonModeVelocityY = 0.0,
            cameraMotionRisk = 0.02,
            meanSpeedNormalizedPerSecond = 0.01,
            maxSpeedNormalizedPerSecond = 0.02,
            meanAccelerationNormalizedPerSecond2 = 0.01,
            maxAccelerationNormalizedPerSecond2 = 0.02,
            meanJerkNormalizedPerSecond3 = 0.02,
            temporalStability = 0.95,
            confidence = 0.95,
            subjects = emptyList(),
        )

        val high = low.copy(
            movingSubjects = 1,
            globalEnergy = 0.92,
            relativeMotionEnergy = 0.86,
            meanSpeedNormalizedPerSecond = 1.2,
            maxSpeedNormalizedPerSecond = 3.8,
            meanAccelerationNormalizedPerSecond2 = 14.0,
            maxAccelerationNormalizedPerSecond2 = 26.0,
        )

        val lowDecision = governor.decide(1L, low)
        val highDecision = governor.decide(2L, high)

        assertTrue(
            highDecision.motionAmplification >=
                lowDecision.motionAmplification,
        )
        assertTrue(
            highDecision.handPriority >=
                lowDecision.handPriority,
        )
        assertTrue(
            highDecision.reasons.contains("high-subject-motion") ||
                highDecision.reasons.contains("moderate-subject-motion"),
        )
    }

    @Test
    fun thermalProtectionDropsRuntimeBand() {
        val governor = ProductionPerceptionRuntimeGovernor()
        governor.updateSystemLoad(
            thermalScale = 0.10,
            memoryScale = 0.70,
            gpuScale = 0.80,
            inferenceScale = 0.90,
        )

        val motion = SemanticMotionEstimator.GlobalMotion(
            timestampNanos = 10L,
            subjectCount = 2,
            activeSubjects = 2,
            movingSubjects = 2,
            globalEnergy = 0.90,
            relativeMotionEnergy = 0.50,
            commonModeEnergy = 0.40,
            commonModeVelocityX = 0.2,
            commonModeVelocityY = 0.1,
            cameraMotionRisk = 0.3,
            meanSpeedNormalizedPerSecond = 0.8,
            maxSpeedNormalizedPerSecond = 2.4,
            meanAccelerationNormalizedPerSecond2 = 12.0,
            maxAccelerationNormalizedPerSecond2 = 25.0,
            meanJerkNormalizedPerSecond3 = 80.0,
            temporalStability = 0.72,
            confidence = 0.81,
            subjects = emptyList(),
        )

        val decision = governor.decide(10L, motion)

        assertTrue(decision.emergencyThermalProtection)
        assertTrue(
            decision.band ==
                ProductionPerceptionRuntimeGovernor.RuntimeBand.PROTECTIVE,
        )
        assertTrue(decision.qualityScale < 0.8)
    }
}
