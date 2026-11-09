package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.reconstruction.MetricPrecisionBudgetEngine
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ReferenceGradeTimingAuthorityTest {

    private fun metrology(): MetricPrecisionBudgetEngine.Decision =
        MetricPrecisionBudgetEngine().evaluate(
            MetricPrecisionBudgetEngine.Input(
                referenceGradeEnabled = true,
                calibrationConfidence = 0.99,
                synchronizationConfidence = 0.99,
                viewDiversity = 0.95,
                crossViewAgreement = 0.96,
                triangulationParallaxDegrees = 18.0,
                reprojectionErrorPixels = 0.25,
                observationCount = 8,
                temporalSkewMilliseconds = 0.25,
                rollingShutterFraction = 0.03,
                cameraBaselineMeters = 1.4,
                effectiveFocalLengthPixels = 3000.0,
                depthUncertaintyMeters = 0.0005,
                poseTranslationUncertaintyMeters = 0.0003,
                poseAngularUncertaintyDegrees = 0.01,
                metricScaleUncertaintyRatio = 0.0002,
                surfaceTemporalStability = 0.98,
            ),
        )

    @Test
    fun referenceProtocolBlocksWeakTimingAuthority() {
        val result =
            ReferenceGradeCalibrationProtocolEngine().evaluate(
                ReferenceGradeCalibrationProtocolEngine.Input(
                    workflowReady = true,
                    rigScore = 0.95,
                    coverage = 0.96,
                    reprojectionErrorPx = 0.40,
                    clockSkewMilliseconds = 0.40,
                    temporalAlignmentConfidence = 0.96,
                    timingAuthorityConfidence = 0.61,
                    timingAuthorityResidualP95Milliseconds = 4.0,
                    timingAuthorityReferenceGrade = false,
                    driftStability = 0.97,
                    recalibrationRequired = false,
                    scaleDriftPercentPerMinute = 0.05,
                    temperatureDeltaCelsius = 0.5,
                    thermalStability = 0.97,
                    observabilityScore = 0.95,
                    metrology = metrology(),
                ),
            )

        assertFalse(result.ready)
        assertTrue(
            result.blockingReasons.contains(
                "timing-authority-below-reference-threshold",
            ),
        )
    }

    @Test
    fun referenceProtocolAcceptsIndependentMeasuredTimingAuthority() {
        val result =
            ReferenceGradeCalibrationProtocolEngine().evaluate(
                ReferenceGradeCalibrationProtocolEngine.Input(
                    workflowReady = true,
                    rigScore = 0.95,
                    coverage = 0.96,
                    reprojectionErrorPx = 0.40,
                    clockSkewMilliseconds = 0.40,
                    temporalAlignmentConfidence = 0.96,
                    timingAuthorityConfidence = 0.98,
                    timingAuthorityResidualP95Milliseconds = 0.7,
                    timingAuthorityReferenceGrade = true,
                    driftStability = 0.97,
                    recalibrationRequired = false,
                    scaleDriftPercentPerMinute = 0.05,
                    temperatureDeltaCelsius = 0.5,
                    thermalStability = 0.97,
                    observabilityScore = 0.95,
                    metrology = metrology(),
                ),
            )

        assertTrue(result.referenceGrade)
        assertTrue(result.ready)
    }
}
