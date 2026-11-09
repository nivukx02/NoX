package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PrecisionCertificationEngineTest {
    private fun metrology() =
        MetricPrecisionBudgetEngine().evaluate(
            MetricPrecisionBudgetEngine.Input(
                targetP95Millimeters = 2.0,
                referenceGradeEnabled = true,
                workingDistanceMeters = 2.0,
                cameraBaselineMeters = 1.5,
                effectiveFocalLengthPixels = 2400.0,
                reprojectionErrorPixels = 0.5,
                triangulationParallaxDegrees = 18.0,
                depthUncertaintyMeters = 0.0008,
                poseTranslationUncertaintyMeters = 0.0005,
                poseAngularUncertaintyDegrees = 0.02,
                calibrationConfidence = 0.98,
                synchronizationConfidence = 0.98,
                viewDiversity = 0.90,
                crossViewAgreement = 0.92,
                temporalSkewMilliseconds = 0.3,
                rollingShutterFraction = 0.02,
                metricScaleUncertaintyRatio = 0.0003,
                reconstructionPriorUncertaintyMeters = 0.0005,
                observationCount = 8,
            ),
        )

    @Test
    fun modeledBudgetDoesNotBecomeCertification() {
        val decision = PrecisionCertificationEngine().evaluate(
            PrecisionCertificationEngine.Input(
                metrology = metrology(),
                empiricalEvidenceEnabled = false,
            ),
        )
        assertTrue(decision.modeledOnly)
        assertFalse(decision.certificationReady)
    }

    @Test
    fun subMillimeterQualificationRequiresStricterEmpiricalEvidence() {
        val decision = PrecisionCertificationEngine().evaluate(
            PrecisionCertificationEngine.Input(
                metrology = metrology(),
                empiricalEvidenceEnabled = true,
                empiricalSampleCount = 260,
                empiricalP95Millimeters = 0.82,
                empiricalP99Millimeters = 1.90,
                empiricalRepeatabilityMillimeters = 0.32,
                empiricalOccludedP95Millimeters = 0.92,
                empiricalLowVisibilityP95Millimeters = 0.88,
                calibrationConfidence = 0.98,
                synchronizationConfidence = 0.98,
                viewDiversity = 0.94,
                crossViewAgreement = 0.94,
                identityContinuity = 0.96,
                surfaceTemporalStability = 0.95,
                referenceUncertaintyMillimeters = 0.20,
                referenceConditionsMatched = true,
            ),
        )

        assertTrue(decision.subMillimeterEvidenceSufficient)
    }

    @Test
    fun modeledOrSmallSampleEvidenceCannotQualifySubMillimeter() {
        val decision = PrecisionCertificationEngine().evaluate(
            PrecisionCertificationEngine.Input(
                metrology = metrology(),
                empiricalEvidenceEnabled = false,
            ),
        )

        assertFalse(decision.subMillimeterEvidenceSufficient)
    }

    @Test
    fun unmatchedReferenceConditionsCannotCertify() {
        val decision = PrecisionCertificationEngine().evaluate(
            PrecisionCertificationEngine.Input(
                metrology = metrology(),
                empiricalEvidenceEnabled = true,
                empiricalSampleCount = 500,
                empiricalP95Millimeters = 1.5,
                empiricalP99Millimeters = 3.8,
                empiricalRepeatabilityMillimeters = 0.5,
                referenceUncertaintyMillimeters = 0.2,
                referenceConditionsMatched = false,
            ),
        )
        assertFalse(decision.certificationReady)
        assertTrue("reference-conditions-not-matched" in decision.warnings)
    }
}
