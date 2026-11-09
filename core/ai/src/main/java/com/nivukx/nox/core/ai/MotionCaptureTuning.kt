package com.nivukx.nox.core.ai

/**
 * User-adjustable motion-capture policy.
 *
 * These settings alter tracker behavior without replacing the estimator or weakening its safety
 * limits. Values are intentionally normalized so a UI can expose them as human-readable controls.
 */
data class MotionCaptureTuning(
    val responsiveness: Double = 0.82,
    val motionSensitivity: Double = 1.0,
    val predictionHorizonSeconds: Double = 0.18,
    val occlusionPersistence: Double = 1.0,
    val reacquisitionRadius: Double = 0.11,
    val minimumConfidence: Double = 0.08,
    val rejectAggressiveness: Double = 0.72,
    val jitterSuppression: Double = 0.55,
    val velocityResponsiveness: Double = 1.00,
    val accelerationResponsiveness: Double = 1.00,
    val innovationGateElasticity: Double = 1.00,
    val motionThresholdScale: Double = 1.00,
    val adaptiveRegimeStrength: Double = 1.00,
) {
    init {
        require(responsiveness in 0.20..1.20)
        require(motionSensitivity in 0.50..1.80)
        require(predictionHorizonSeconds in 0.04..0.35)
        require(occlusionPersistence in 0.50..2.50)
        require(reacquisitionRadius in 0.03..0.22)
        require(minimumConfidence in 0.02..0.30)
        require(rejectAggressiveness in 0.20..1.00)
        require(jitterSuppression in 0.0..1.0)
        require(velocityResponsiveness in 0.55..1.60)
        require(accelerationResponsiveness in 0.55..1.60)
        require(innovationGateElasticity in 0.65..1.60)
        require(motionThresholdScale in 0.70..1.40)
        require(adaptiveRegimeStrength in 0.0..1.0)
    }

    companion object {
        fun cinematic(): MotionCaptureTuning =
            MotionCaptureTuning(
                responsiveness = 0.62,
                motionSensitivity = 0.84,
                predictionHorizonSeconds = 0.12,
                occlusionPersistence = 0.85,
                reacquisitionRadius = 0.08,
                minimumConfidence = 0.10,
                rejectAggressiveness = 0.82,
                jitterSuppression = 0.72,
                velocityResponsiveness = 0.82,
                accelerationResponsiveness = 0.86,
                innovationGateElasticity = 0.88,
                motionThresholdScale = 1.12,
                adaptiveRegimeStrength = 0.72,
            )

        fun action(): MotionCaptureTuning =
            MotionCaptureTuning(
                responsiveness = 1.02,
                motionSensitivity = 1.42,
                predictionHorizonSeconds = 0.24,
                occlusionPersistence = 1.35,
                reacquisitionRadius = 0.15,
                minimumConfidence = 0.06,
                rejectAggressiveness = 0.56,
                jitterSuppression = 0.38,
                velocityResponsiveness = 1.28,
                accelerationResponsiveness = 1.30,
                innovationGateElasticity = 1.24,
                motionThresholdScale = 0.86,
                adaptiveRegimeStrength = 0.96,
            )

        fun precision(): MotionCaptureTuning =
            MotionCaptureTuning(
                responsiveness = 0.76,
                motionSensitivity = 0.92,
                predictionHorizonSeconds = 0.10,
                occlusionPersistence = 0.68,
                reacquisitionRadius = 0.06,
                minimumConfidence = 0.12,
                rejectAggressiveness = 0.90,
                jitterSuppression = 0.82,
                velocityResponsiveness = 0.92,
                accelerationResponsiveness = 0.88,
                innovationGateElasticity = 0.76,
                motionThresholdScale = 1.16,
                adaptiveRegimeStrength = 0.82,
            )
        fun creature(): MotionCaptureTuning =
            MotionCaptureTuning(
                responsiveness = 0.88,
                motionSensitivity = 1.22,
                predictionHorizonSeconds = 0.20,
                occlusionPersistence = 1.42,
                reacquisitionRadius = 0.16,
                minimumConfidence = 0.08,
                rejectAggressiveness = 0.62,
                jitterSuppression = 0.44,
                velocityResponsiveness = 1.16,
                accelerationResponsiveness = 1.22,
                innovationGateElasticity = 1.20,
                motionThresholdScale = 0.92,
                adaptiveRegimeStrength = 0.94,
            )

        fun facial(): MotionCaptureTuning =
            MotionCaptureTuning(
                responsiveness = 0.72,
                motionSensitivity = 0.88,
                predictionHorizonSeconds = 0.10,
                occlusionPersistence = 0.72,
                reacquisitionRadius = 0.07,
                minimumConfidence = 0.12,
                rejectAggressiveness = 0.90,
                jitterSuppression = 0.70,
                velocityResponsiveness = 0.84,
                accelerationResponsiveness = 0.82,
                innovationGateElasticity = 0.82,
                motionThresholdScale = 1.08,
                adaptiveRegimeStrength = 0.78,
            )

        fun handDetail(): MotionCaptureTuning =
            MotionCaptureTuning(
                responsiveness = 0.94,
                motionSensitivity = 1.18,
                predictionHorizonSeconds = 0.08,
                occlusionPersistence = 1.18,
                reacquisitionRadius = 0.11,
                minimumConfidence = 0.07,
                rejectAggressiveness = 0.66,
                jitterSuppression = 0.50,
                velocityResponsiveness = 1.18,
                accelerationResponsiveness = 1.14,
                innovationGateElasticity = 1.10,
                motionThresholdScale = 0.94,
                adaptiveRegimeStrength = 0.90,
            )

    }
}
