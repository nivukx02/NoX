package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UltraPrecisionEvidenceEngineTest {

    private fun input(
        feasibility: Double,
        firstOrderDepthErrorMm: Double,
        motionStability: Double,
        kinematicInnovationMm: Double = 0.8,
        tailErrorMm: Double = firstOrderDepthErrorMm,
        tailFeasibility: Double = feasibility,
    ) =
        UltraPrecisionEvidenceEngine.Input(
            workingDistanceMeters = 2.0,
            cameraBaselineMeters = 0.8,
            effectiveFocalLengthPixels = 2_400.0,
            medianParallaxDegrees = 12.0,
            reprojectionErrorPixels = 0.35,
            subpixelObservationSigmaPixels = 0.18,
            cameraTranslationSigmaMeters = 0.0007,
            cameraAngularSigmaDegrees = 0.025,
            synchronizationSkewMilliseconds = 0.25,
            rollingShutterEquivalentMilliseconds = 0.25,
            motionSpeedMetersPerSecond = 0.8,
            metricScaleUncertaintyRatio = 0.0004,
            calibrationConfidence = 0.98,
            synchronizationConfidence = 0.98,
            viewDiversity = 0.86,
            crossViewAgreement = 0.95,
            occlusionRatio = 0.06,
            identityContinuity = 0.97,
            surfaceTemporalStability = 0.97,
            thermalRisk = 0.20,
            memoryPressure = 0.30,
            subMillimeterGeometryFeasibility = feasibility,
            firstOrderDepthErrorMillimeters = firstOrderDepthErrorMm,
            tailAdjustedFirstOrderDepthErrorMillimeters = tailErrorMm,
            tailAdjustedSubMillimeterFeasibilityScore = tailFeasibility,
            denseMotionStability = motionStability,
            denseKinematicInnovationMillimeters = kinematicInnovationMm,
            observationCount = 2_000,
            empiricalP95Millimeters = 0.62,
            empiricalP99Millimeters = 1.32,
            empiricalSampleCount = 1_200,
            matchedReference = true,
            referenceKind =
                UltraPrecisionEvidenceEngine.ReferenceKind
                    .OPTICAL_INFRARED_MARKER_SYSTEM,
            referenceP95Millimeters = 0.72,
            referenceUncertaintyMillimeters = 0.10,
        )

    @Test
    fun longExposureAddsPhysicalBlurToModeledErrorBudget() {
        val engine = UltraPrecisionEvidenceEngine()
        val shortExposure =
            engine.evaluate(
                UltraPrecisionEvidenceEngine.Policy.highEndQualification(),
                input(
                    feasibility = 0.91,
                    firstOrderDepthErrorMm = 0.56,
                    motionStability = 0.93,
                ).copy(
                    exposureIntegrationMilliseconds = 1.0,
                ),
            )
        val longExposure =
            engine.evaluate(
                UltraPrecisionEvidenceEngine.Policy.highEndQualification(),
                input(
                    feasibility = 0.91,
                    firstOrderDepthErrorMm = 0.56,
                    motionStability = 0.93,
                ).copy(
                    exposureIntegrationMilliseconds = 12.0,
                ),
            )

        assertTrue(
            longExposure.errorBudget.exposureBlurMillimeters >
                shortExposure.errorBudget.exposureBlurMillimeters,
        )
        assertTrue(
            longExposure.errorBudget.modeledP95Millimeters >=
                shortExposure.errorBudget.modeledP95Millimeters,
        )
    }

    @Test
    fun weakDensePrecisionEvidenceBlocksHighEndQualification() {
        val engine = UltraPrecisionEvidenceEngine()
        val result =
            engine.evaluate(
                UltraPrecisionEvidenceEngine.Policy
                    .highEndQualification(),
                input(
                    feasibility = 0.18,
                    firstOrderDepthErrorMm = 2.40,
                    motionStability = 0.48,
                ),
            )

        assertFalse(result.qualificationEligible)
        assertTrue(
            result.blockers.any {
                it.contains("dense-geometry-submillimeter")
            },
        )
        assertTrue(
            result.blockers.any {
                it.contains("dense-motion-stability")
            },
        )
    }

    @Test
    fun degradedReprojectionTailIsVisibleWithoutInventingAQualificationBlocker() {
        val engine = UltraPrecisionEvidenceEngine()
        val result =
            engine.evaluate(
                UltraPrecisionEvidenceEngine.Policy
                    .highEndQualification(),
                input(
                    feasibility = 0.91,
                    firstOrderDepthErrorMm = 0.56,
                    motionStability = 0.93,
                    tailErrorMm = 1.42,
                    tailFeasibility = 0.44,
                ),
            )

        assertTrue(
            result.warnings.any {
                it.contains("tail-adjusted-depth-error")
            },
        )
    }

    @Test
    fun strongDensePrecisionEvidenceRemainsEligibleForMatchedReferenceEvaluation() {
        val engine = UltraPrecisionEvidenceEngine()
        val result =
            engine.evaluate(
                UltraPrecisionEvidenceEngine.Policy
                    .highEndQualification(),
                input(
                    feasibility = 0.91,
                    firstOrderDepthErrorMm = 0.56,
                    motionStability = 0.93,
                ),
            )

        assertTrue(result.empiricalEvidenceAvailable)
        assertTrue(result.matchedReferenceEvidenceAvailable)
        assertTrue(
            result.blockers.none {
                it.contains("dense-geometry-submillimeter")
            },
        )
        assertTrue(
            result.warnings.none {
                it.contains("dense-motion-stability")
            },
        )
    }
}
