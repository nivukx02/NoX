package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ReferenceGradeEvidenceGateEngineTest {
    @Test
    fun disabled_mode_is_not_reported_as_ready() {
        val result = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = false,
            ),
        )
        assertEquals(
            ReferenceGradeEvidenceGateEngine.Status.NOT_REQUESTED,
            result.status,
        )
        assertFalse(result.ready)
    }

    @Test
    fun missing_empirical_benchmark_blocks_reference_mode() {
        val result = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = true,
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = false,
                sampleCount = 500,
                measuredP95Millimeters = 1.2,
                measuredP99Millimeters = 2.4,
                referenceUncertaintyMillimeters = 0.3,
                externalReferenceDocumented = true,
            ),
        )
        assertEquals(
            ReferenceGradeEvidenceGateEngine.Status.BLOCKED,
            result.status,
        )
        assertTrue(result.blockingReasons.any { it.contains("benchmark") })
    }

    @Test
    fun target_excess_blocks_even_when_benchmark_exists() {
        val result = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = true,
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 500,
                measuredP95Millimeters = 2.10,
                measuredP99Millimeters = 3.8,
                referenceUncertaintyMillimeters = 0.25,
                externalReferenceDocumented = true,
            ),
        )
        assertFalse(result.ready)
        assertTrue(result.blockingReasons.any { it.contains("P95") })
    }

    @Test
    fun undocumented_or_too_uncertain_reference_blocks() {
        val undocumented = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = true,
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 500,
                measuredP95Millimeters = 1.4,
                measuredP99Millimeters = 3.2,
                referenceUncertaintyMillimeters = null,
                externalReferenceDocumented = false,
            ),
        )
        assertFalse(undocumented.ready)

        val uncertain = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = true,
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 500,
                measuredP95Millimeters = 1.1,
                measuredP99Millimeters = 2.8,
                referenceUncertaintyMillimeters = 2.2,
                externalReferenceDocumented = true,
            ),
        )
        assertFalse(uncertain.ready)
        assertTrue(uncertain.blockingReasons.any { it.contains("uncertainty") })
    }

    @Test
    fun sufficient_qualified_evidence_is_ready() {
        val result = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = true,
                matchedCaptureConditions = true,
                empiricalBenchmarkAccepted = true,
                sampleCount = 800,
                measuredP95Millimeters = 1.80,
                measuredP99Millimeters = 3.45,
                referenceUncertaintyMillimeters = 0.40,
                externalReferenceDocumented = true,
            ),
        )
        assertEquals(
            ReferenceGradeEvidenceGateEngine.Status.READY,
            result.status,
        )
        assertTrue(result.ready)
        assertTrue(result.effectiveP95TargetMillimeters < 2.0)
        assertTrue(result.effectiveP95TargetMillimeters > 1.8)
    }

    @Test
    fun matched_conditions_are_required() {
        val result = ReferenceGradeEvidenceGateEngine().evaluate(
            ReferenceGradeEvidenceGateEngine.Input(
                enabled = true,
                matchedCaptureConditions = false,
                empiricalBenchmarkAccepted = true,
                sampleCount = 900,
                measuredP95Millimeters = 1.4,
                measuredP99Millimeters = 2.8,
                referenceUncertaintyMillimeters = 0.25,
                externalReferenceDocumented = true,
            ),
        )
        assertFalse(result.ready)
        assertTrue(result.blockingReasons.any { it.contains("matched") })
    }

    @Test
    fun benchmark_report_can_be_consumed_without_repeating_measurement_plumbing() {
        val samples = buildList {
            repeat(128) { frame ->
                add(
                    CapturePrecisionBenchmarkEngine.JointSample(
                        timestampNanos = frame * 16_666_667L,
                        jointId = 0,
                        estimate = Vec3(0.0005, 0.0, 1.0),
                        reference = Vec3(0.0, 0.0, 1.0),
                        confidence = 0.99,
                    ),
                )
            }
        }
        val report = CapturePrecisionBenchmarkEngine(
            CapturePrecisionBenchmarkEngine.Config(
                minimumSamples = 64,
            ),
        ).evaluate(
            CapturePrecisionBenchmarkEngine.Input(
                captureId = "reference-adapter",
                referenceType =
                    CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE,
                samples = samples,
                referenceUncertaintyMillimeters = 0.20,
            ),
        )
        val result = ReferenceGradeEvidenceGateEngine().evaluate(
            report = report,
            matchedCaptureConditions = true,
        )
        assertTrue(result.status != ReferenceGradeEvidenceGateEngine.Status.NOT_REQUESTED)
        assertTrue(result.blockingReasons.none { it.contains("reference uncertainty") })
    }

}
