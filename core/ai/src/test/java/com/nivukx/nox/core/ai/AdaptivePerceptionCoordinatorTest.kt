package com.nivukx.nox.core.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptivePerceptionCoordinatorTest {

    @Test
    fun staticSceneReducesExpensiveStages() {
        val coordinator = AdaptivePerceptionCoordinator()
        val plan = coordinator.beginFrame(
            timestampNanos = 1_000_000_000L,
            motionQuality = quality(
                motionClass = MotionCaptureEngine.MotionClass.STATIC,
                readiness = 0.42,
            ),
        )

        assertEquals(1, plan.decisions.getValue(
            AdaptivePerceptionCoordinator.Stage.POSE,
        ).effectiveIntervalFrames)

        assertTrue(
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.OBJECT,
            ).effectiveIntervalFrames >= 2,
        )

        assertTrue(
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.CONTOUR,
            ).effectiveIntervalFrames >= 3,
        )
    }

    @Test
    fun extremeMotionRestoresHighRatePerception() {
        val coordinator = AdaptivePerceptionCoordinator()
        val plan = coordinator.beginFrame(
            timestampNanos = 2_000_000_000L,
            motionQuality = quality(
                motionClass = MotionCaptureEngine.MotionClass.EXTREME,
                readiness = 0.91,
                occlusionRatio = 0.08,
            ),
        )

        assertEquals(
            1,
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.HAND,
            ).effectiveIntervalFrames,
        )

        assertEquals(
            1,
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.OBJECT,
            ).effectiveIntervalFrames,
        )

        assertTrue(plan.overallInferenceScale > 1.0)
    }

    @Test
    fun highStageLatencyCausesStageSpecificRefreshWithoutGlobalThrottling() {
        val coordinator = AdaptivePerceptionCoordinator()
        repeat(6) {
            coordinator.recordStageDuration(
                AdaptivePerceptionCoordinator.Stage.CONTOUR,
                36.0,
            )
        }

        val plan = coordinator.beginFrame(
            timestampNanos = 3_500_000_000L,
            motionQuality = quality(
                motionClass = MotionCaptureEngine.MotionClass.NORMAL,
                readiness = 0.62,
            ),
        )

        val contour =
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.CONTOUR,
            )
        val pose =
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.POSE,
            )

        assertTrue(contour.effectiveIntervalFrames >= 3)
        assertTrue(
            contour.reason.contains("stage-latency") ||
                contour.reason.contains("cached"),
        )
        assertEquals(1, pose.effectiveIntervalFrames)
    }

    @Test
    fun thermalPressureNeverChangesPoseToAZeroRate() {
        val coordinator = AdaptivePerceptionCoordinator()
        coordinator.updateSystemLoad(
            thermalScale = 0.25,
            memoryScale = 0.80,
            gpuScale = 0.70,
            inferenceScale = 0.80,
        )

        val plan = coordinator.beginFrame(
            timestampNanos = 3_000_000_000L,
            motionQuality = quality(
                motionClass = MotionCaptureEngine.MotionClass.NORMAL,
                readiness = 0.60,
            ),
        )

        assertTrue(
            plan.decisions.getValue(
                AdaptivePerceptionCoordinator.Stage.POSE,
            ).effectiveIntervalFrames >= 1,
        )
        assertEquals("thermal-budget", plan.reason)
    }

    @Test
    fun stageTelemetryUsesEwma() {
        val coordinator = AdaptivePerceptionCoordinator()
        coordinator.recordStageDuration(
            AdaptivePerceptionCoordinator.Stage.POSE,
            10.0,
        )
        coordinator.recordStageDuration(
            AdaptivePerceptionCoordinator.Stage.POSE,
            30.0,
        )

        val ewma = coordinator.telemetry()
            .ewmaStageMillis
            .getValue(AdaptivePerceptionCoordinator.Stage.POSE)

        assertEquals(13.0, ewma, 0.0001)
    }

    private fun quality(
        motionClass: MotionCaptureEngine.MotionClass,
        readiness: Double,
        occlusionRatio: Double = 0.15,
    ): MotionCaptureEngine.Quality =
        MotionCaptureEngine.Quality(
            timestampNanos = 0L,
            motionScore = if (motionClass == MotionCaptureEngine.MotionClass.EXTREME) 0.95 else 0.02,
            motionClass = motionClass,
            meanSpeedNormalizedPerSecond = 0.1,
            maxSpeedNormalizedPerSecond = 1.2,
            meanAccelerationNormalizedPerSecond2 = 1.0,
            maxAccelerationNormalizedPerSecond2 = 4.0,
            meanJerkNormalizedPerSecond3 = 0.4,
            jitterNormalized = 0.05,
            trackedLandmarks = 17,
            predictedLandmarks = 1,
            rejectedLandmarks = 0,
            meanConfidence = 0.88,
            occlusionRatio = occlusionRatio,
            temporalStability = 0.85,
            spatialSpread = 0.22,
            stationaryRatio = 0.15,
            dynamicsConfidence = 0.90,
            meanInnovationNormalized = 0.01,
            rejectedMeasurementRatio = 0.0,
        )
}
