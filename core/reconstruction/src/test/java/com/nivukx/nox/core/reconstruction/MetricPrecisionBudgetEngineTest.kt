package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MetricPrecisionBudgetEngineTest {

    @Test
    fun wellConditionedRigProducesFiniteBudget() {
        val engine = MetricPrecisionBudgetEngine()
        val result = engine.evaluate(
            MetricPrecisionBudgetEngine.Input(
                targetP95Millimeters = 5.0,
                workingDistanceMeters = 2.5,
                cameraBaselineMeters = 1.4,
                effectiveFocalLengthPixels = 2200.0,
                reprojectionErrorPixels = 0.12,
                triangulationParallaxDegrees = 18.0,
                depthUncertaintyMeters = 0.002,
                poseTranslationUncertaintyMeters = 0.001,
                poseAngularUncertaintyDegrees = 0.02,
                calibrationConfidence = 0.97,
                synchronizationConfidence = 0.97,
                viewDiversity = 0.86,
                crossViewAgreement = 0.90,
                temporalSkewMilliseconds = 0.50,
                rollingShutterFraction = 0.05,
                metricScaleUncertaintyRatio = 0.0005,
                reconstructionPriorUncertaintyMeters = 0.0005,
                observationCount = 6,
                identityContinuity = 0.99,
                thermalRisk = 0.10,
                memoryPressure = 0.10,
                surfaceTemporalStability = 0.97,
            ),
        )

        assertTrue(result.modeledP95ErrorMillimeters.isFinite())
        assertTrue(result.modeledP99ErrorMillimeters.isFinite())
        assertTrue(result.feasibilityScore > 0.60)
        assertFalse(result.hardBlock)
    }

    @Test
    fun poorTriangulationIsPenalizedAndBlocksReferenceGrade() {
        val engine = MetricPrecisionBudgetEngine()
        val result = engine.evaluate(
            MetricPrecisionBudgetEngine.Input(
                targetP95Millimeters = 2.0,
                referenceGradeEnabled = true,
                workingDistanceMeters = 8.0,
                cameraBaselineMeters = 0.20,
                effectiveFocalLengthPixels = 1000.0,
                reprojectionErrorPixels = 2.0,
                triangulationParallaxDegrees = 0.6,
                depthUncertaintyMeters = 0.30,
                poseTranslationUncertaintyMeters = 0.02,
                poseAngularUncertaintyDegrees = 2.0,
                calibrationConfidence = 0.65,
                synchronizationConfidence = 0.70,
                viewDiversity = 0.25,
                crossViewAgreement = 0.20,
                temporalSkewMilliseconds = 8.0,
                rollingShutterFraction = 0.40,
                metricScaleUncertaintyRatio = 0.04,
                reconstructionPriorUncertaintyMeters = 0.10,
                observationCount = 2,
                occlusionRatio = 0.40,
                identityContinuity = 0.70,
                thermalRisk = 0.80,
                memoryPressure = 0.85,
                surfaceTemporalStability = 0.55,
            ),
        )

        assertTrue(result.modeledP95ErrorMillimeters > 2.0)
        assertTrue(result.hardBlock)
        assertFalse(result.referenceGradeFeasible)
        assertTrue(
            result.warnings.any {
                it.contains("triangulation")
            },
        )
    }

    @Test
    fun subMillimeterTargetRemainsMeasurementGated() {
        val engine = MetricPrecisionBudgetEngine()
        val result = engine.evaluate(
            MetricPrecisionBudgetEngine.Input(
                targetP95Millimeters = 0.75,
                referenceGradeEnabled = true,
                workingDistanceMeters = 1.5,
                cameraBaselineMeters = 1.0,
                effectiveFocalLengthPixels = 3200.0,
                reprojectionErrorPixels = 0.03,
                triangulationParallaxDegrees = 25.0,
                depthUncertaintyMeters = 0.0002,
                poseTranslationUncertaintyMeters = 0.0002,
                poseAngularUncertaintyDegrees = 0.01,
                calibrationConfidence = 0.995,
                synchronizationConfidence = 0.995,
                viewDiversity = 0.95,
                crossViewAgreement = 0.96,
                temporalSkewMilliseconds = 0.20,
                rollingShutterFraction = 0.02,
                metricScaleUncertaintyRatio = 0.0001,
                reconstructionPriorUncertaintyMeters = 0.0001,
                observationCount = 8,
                identityContinuity = 0.995,
                thermalRisk = 0.05,
                memoryPressure = 0.05,
                surfaceTemporalStability = 0.99,
            ),
        )

        assertTrue(result.modeledP95ErrorMillimeters.isFinite())
        assertFalse(
            result.referenceGradeFeasible &&
                result.modeledP95ErrorMillimeters < 0.75 &&
                result.hardBlock,
        )
    }

    @Test
    fun timingAndRollingShutterContributionsRemainVisible() {
        val engine = MetricPrecisionBudgetEngine()
        val clean = engine.evaluate(
            MetricPrecisionBudgetEngine.Input(
                temporalSkewMilliseconds = 0.25,
                rollingShutterFraction = 0.02,
                motionSpeedMetersPerSecond = 2.5,
                motionAccelerationMetersPerSecondSquared = 5.0,
            ),
        )
        val degraded = engine.evaluate(
            MetricPrecisionBudgetEngine.Input(
                temporalSkewMilliseconds = 8.0,
                rollingShutterFraction = 0.40,
                motionSpeedMetersPerSecond = 2.5,
                motionAccelerationMetersPerSecondSquared = 5.0,
            ),
        )

        assertTrue(
            degraded.modeledP95ErrorMillimeters >
                clean.modeledP95ErrorMillimeters,
        )
    }
}
