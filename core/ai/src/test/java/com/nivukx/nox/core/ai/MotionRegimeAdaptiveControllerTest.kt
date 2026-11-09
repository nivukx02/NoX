package com.nivukx.nox.core.ai

import kotlin.test.assertTrue
import org.junit.Test

class MotionRegimeAdaptiveControllerTest {

    private val controller = MotionRegimeAdaptiveController()

    @Test
    fun extremeMotionReducesPredictionReliance() {
        val policy = controller.resolve(
            quality = quality(
                motionClass = MotionCaptureEngine.MotionClass.EXTREME,
                occlusion = 0.20,
                jitter = 0.18,
                vibration = 0.20,
                spectralConfidence = 0.80,
            ),
            context = MotionCaptureEngine.MotionContext(),
        )

        assertTrue(
            policy.regime == MotionRegimeAdaptiveController.Regime.FAST_ACTION ||
                policy.regime == MotionRegimeAdaptiveController.Regime.EXTREME_ACTION,
        )
        assertTrue(policy.predictionHorizonScale <= 1.0)
        assertTrue(policy.innovationGateScale >= 1.0)
    }

    @Test
    fun dominantCameraMotionSuppressesAggressiveSubjectAdaptation() {
        val policy = controller.resolve(
            quality = quality(
                motionClass = MotionCaptureEngine.MotionClass.FAST,
                occlusion = 0.10,
                jitter = 0.15,
                vibration = 0.10,
                spectralConfidence = 0.70,
            ),
            context = MotionCaptureEngine.MotionContext(
                cameraMotionRisk = 0.90,
                sensorConfidence = 0.95,
                rollingShutterRisk = 0.40,
                globalMotionConfidence = 0.90,
            ),
        )

        assertTrue(policy.regime == MotionRegimeAdaptiveController.Regime.CAMERA_DOMINANT)
        assertTrue(policy.velocityResponsivenessScale < 1.0)
        assertTrue(policy.innovationGateScale > 1.0)
    }

    @Test
    fun occlusionEntersRecoveryMode() {
        val policy = controller.resolve(
            quality = quality(
                motionClass = MotionCaptureEngine.MotionClass.NORMAL,
                occlusion = 0.72,
                jitter = 0.20,
                vibration = 0.16,
                spectralConfidence = 0.70,
            ),
            context = MotionCaptureEngine.MotionContext(),
        )

        assertTrue(policy.regime == MotionRegimeAdaptiveController.Regime.OCCLUDED_RECOVERY)
        assertTrue(policy.reacquisitionScale > 1.0)
        assertTrue(policy.predictionHorizonScale > 1.0)
    }

    private fun quality(
        motionClass: MotionCaptureEngine.MotionClass,
        occlusion: Double,
        jitter: Double,
        vibration: Double,
        spectralConfidence: Double,
    ) = MotionCaptureEngine.Quality(
        timestampNanos = 1_000_000L,
        motionScore = 0.8,
        motionClass = motionClass,
        tuningResponsiveness = 0.82,
        predictionHorizonSeconds = 0.18,
        reacquisitionRadiusNormalized = 0.11,
        meanSpeedNormalizedPerSecond = 1.0,
        maxSpeedNormalizedPerSecond = 2.5,
        meanAccelerationNormalizedPerSecond2 = 4.0,
        maxAccelerationNormalizedPerSecond2 = 12.0,
        meanJerkNormalizedPerSecond3 = 10.0,
        jitterNormalized = jitter,
        trackedLandmarks = 24,
        predictedLandmarks = 2,
        rejectedLandmarks = 1,
        meanConfidence = 0.90,
        occlusionRatio = occlusion,
        temporalStability = 0.84,
        spatialSpread = 0.42,
        stationaryRatio = 0.10,
        dynamicsConfidence = 0.86,
        meanInnovationNormalized = 0.08,
        rejectedMeasurementRatio = 0.04,
        directionChangeRatio = 0.12,
        microMotionScore = 0.55,
        meanMeasurementUncertaintyScore = 0.18,
        commonModeMotionNormalizedPerSecond = 0.20,
        differentialMotionScore = 0.80,
        cameraCompensatedMotionScore = 0.72,
        dominantFrequencyHz = 2.2,
        periodicMotionScore = 0.30,
        vibrationRisk = vibration,
        spectralConfidence = spectralConfidence,
    )
}
