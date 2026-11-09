package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MeasuredPrecisionQualificationEngineTest {

    @Test
    fun strict_profile_requires_independent_sessions_and_surface_evidence() {
        val engine =
            MeasuredPrecisionQualificationEngine(
                MeasuredPrecisionQualificationEngine.Config.strictSubMillimeter(),
            )

        val blocked =
            engine.evaluateSubMillimeter(
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 512,
                measuredP95Millimeters = 0.70,
                measuredP99Millimeters = 1.40,
                referenceUncertaintyMillimeters = 0.05,
                referenceProvenanceDocumented = true,
                independentReferenceSessionCount = 1,
            )

        assertFalse(blocked.qualified)
        assertTrue(
            blocked.blockingReasons.any {
                it.contains("independent reference sessions")
            },
        )

        val qualified =
            engine.evaluate(
                MeasuredPrecisionQualificationEngine.Input(
                    requestedTier =
                        MeasuredPrecisionQualificationEngine.Tier.SUB_MILLIMETER,
                    matchedCaptureConditions = true,
                    empiricalBenchmarkAccepted = true,
                    sampleCount = 512,
                    independentReferenceSessionCount = 3,
                    measuredP95Millimeters = 0.70,
                    measuredP99Millimeters = 1.40,
                    referenceUncertaintyMillimeters = 0.05,
                    referenceProvenanceDocumented = true,
                    surfaceEvidenceAvailable = true,
                    measuredSurfaceP95Millimeters = 1.10,
                    measuredSurfaceP99Millimeters = 2.10,
                ),
            )

        assertTrue(qualified.qualified)
        assertTrue(qualified.surfaceEvidenceRequired)
        assertTrue(qualified.independentReferenceSessionCount == 3)
    }

    @Test
    fun subMillimeterQualificationRequiresAllEvidenceContracts() {
        val engine = MeasuredPrecisionQualificationEngine()

        val blocked = engine.evaluateSubMillimeter(
            matchedCaptureConditions = true,
            empiricalBenchmarkAccepted = true,
            sampleCount = 300,
            independentReferenceSessionCount = 1,
            measuredP95Millimeters = 0.82,
            measuredP99Millimeters = 1.64,
            referenceUncertaintyMillimeters = 0.20,
            referenceProvenanceDocumented = true,
        )

        assertTrue(blocked.qualified)

        val missingEvidence = engine.evaluateSubMillimeter(
            matchedCaptureConditions = false,
            empiricalBenchmarkAccepted = false,
            sampleCount = 64,
            measuredP95Millimeters = null,
            measuredP99Millimeters = null,
            referenceUncertaintyMillimeters = null,
            referenceProvenanceDocumented = false,
        )

        assertFalse(missingEvidence.qualified)
        assertTrue(
            missingEvidence.blockingReasons.isNotEmpty(),
        )
    }

    @Test
    fun subMillimeterQualificationRejectsMeasuredTailAboveTarget() {
        val result =
            MeasuredPrecisionQualificationEngine().evaluateSubMillimeter(
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 512,
                measuredP95Millimeters = 1.04,
                measuredP99Millimeters = 1.80,
                referenceUncertaintyMillimeters = 0.10,
                referenceProvenanceDocumented = true,
            )

        assertFalse(result.qualified)
        assertTrue(
            result.blockingReasons.any {
                it.contains("P95")
            },
        )
    }


    @Test
    fun p99QualificationConsumesReferenceUncertaintyBudget() {
        val result =
            MeasuredPrecisionQualificationEngine().evaluateSubMillimeter(
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 512,
                measuredP95Millimeters = 0.70,
                measuredP99Millimeters = 2.10,
                referenceUncertaintyMillimeters = 0.50,
                referenceProvenanceDocumented = true,
            )

        assertFalse(result.qualified)
        assertTrue(
            result.uncertaintyAdjustedP99TargetMillimeters <
                result.targetP99Millimeters,
        )
        assertTrue(
            result.blockingReasons.any {
                it.contains("P99")
            },
        )
    }

    @Test
    fun strictSubMillimeterSurfaceQualificationRequiresSurfaceEvidenceAndTails() {
        val engine = MeasuredPrecisionQualificationEngine(
            MeasuredPrecisionQualificationEngine.Config(
                requireSurfaceEvidenceForSubMillimeter = true,
            ),
        )

        val missingSurface = engine.evaluateSubMillimeter(
            matchedCaptureConditions = true,
            empiricalBenchmarkAccepted = true,
            sampleCount = 512,
            measuredP95Millimeters = 0.70,
            measuredP99Millimeters = 1.40,
            referenceUncertaintyMillimeters = 0.05,
            referenceProvenanceDocumented = true,
        )

        assertFalse(missingSurface.qualified)
        assertTrue(
            missingSurface.blockingReasons.any {
                it.contains("dense surface")
            },
        )

        val qualified = engine.evaluateSubMillimeterWithSurfaceEvidence(
            matchedCaptureConditions = true,
            empiricalBenchmarkAccepted = true,
            sampleCount = 512,
            measuredP95Millimeters = 0.70,
            measuredP99Millimeters = 1.40,
            referenceUncertaintyMillimeters = 0.05,
            referenceProvenanceDocumented = true,
            measuredSurfaceP95Millimeters = 1.10,
            measuredSurfaceP99Millimeters = 2.10,
        )

        assertTrue(qualified.qualified)
        assertTrue(qualified.surfaceEvidenceRequired)
        assertTrue(qualified.surfaceEvidenceAvailable)
    }

    @Test
    fun strictSubMillimeterSurfaceQualificationRejectsSurfaceTail() {
        val engine = MeasuredPrecisionQualificationEngine(
            MeasuredPrecisionQualificationEngine.Config(
                requireSurfaceEvidenceForSubMillimeter = true,
            ),
        )

        val result = engine.evaluateSubMillimeterWithSurfaceEvidence(
            matchedCaptureConditions = true,
            empiricalBenchmarkAccepted = true,
            sampleCount = 512,
            measuredP95Millimeters = 0.70,
            measuredP99Millimeters = 1.40,
            referenceUncertaintyMillimeters = 0.05,
            referenceProvenanceDocumented = true,
            measuredSurfaceP95Millimeters = 1.70,
            measuredSurfaceP99Millimeters = 2.80,
        )

        assertFalse(result.qualified)
        assertTrue(
            result.blockingReasons.any {
                it.contains("dense surface P95")
            },
        )
    }

    @Test
    fun noRequestDoesNotClaimQualification() {
        val result =
            MeasuredPrecisionQualificationEngine().evaluate(
                MeasuredPrecisionQualificationEngine.Input(
                    requestedTier = null,
                    matchedCaptureConditions = true,
                    empiricalBenchmarkAccepted = true,
                    sampleCount = 512,
                    measuredP95Millimeters = 0.70,
                    measuredP99Millimeters = 1.40,
                    referenceUncertaintyMillimeters = 0.05,
                    referenceProvenanceDocumented = true,
                ),
            )

        assertFalse(result.qualified)
        assertTrue(
            result.status ==
                MeasuredPrecisionQualificationEngine.Status
                    .NOT_REQUESTED,
        )
    }
}
