package com.nivukx.nox.core.reconstruction

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class ReferenceParityCohortEngineTest {

    private fun report(
        captureId: String,
        referenceType: CapturePrecisionBenchmarkEngine.ReferenceType,
        p95: Double,
        p99: Double,
        accepted: Boolean = true,
    ) = CapturePrecisionBenchmarkEngine.Report(
        captureId = captureId,
        referenceType = referenceType,
        referenceUncertaintyMillimeters = 0.30,
        aggregate = CapturePrecisionBenchmarkEngine.Aggregate(
            sampleCount = 800,
            validCount = 780,
            jointCount = 24,
            dropoutRatio = 0.025,
            meanErrorMm = p95 * 0.58,
            rmsErrorMm = p95 * 0.72,
            p50ErrorMm = p95 * 0.55,
            p95ErrorMm = p95,
            p99ErrorMm = p99,
            maxErrorMm = p99 * 1.35,
            jitterRmsMm = 0.9,
            temporalLagMs = 8.0,
            scaleRatio = 1.0,
            scaleDriftPercentPerMinute = 0.12,
            surface = null,
        ),
        jointMetrics = emptyList(),
        gates = emptyList(),
        confidence = 0.92,
        accepted = accepted,
        warnings = emptyList(),
    )

    @Test
    fun repeated_pairs_produce_cohort_metrics_and_confidence() {
        val engine = ReferenceParityCohortEngine(
            ReferenceParityCohortEngine.Config(
                minimumPairs = 3,
            ),
        )

        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "rigA|60fps|8cams|1.0m|24C",
                noxReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.6, 2.8),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.8, 3.1),
                    report("take-3", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.7, 3.0),
                ),
                referenceReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 0.7, 1.2),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 0.8, 1.3),
                    report("take-3", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 0.7, 1.1),
                ),
            ),
        )

        assertTrue(result.matchedPairs == 3)
        assertTrue(result.aggregateNoxP95Millimeters < 2.0)
        assertTrue(result.aggregateNoxP99Millimeters < 3.2)
        assertTrue(result.confidence > 0.80)
        assertTrue(result.evidenceQualified)
    }

    @Test
    fun mismatched_condition_fingerprint_blocks_qualification() {
        val engine = ReferenceParityCohortEngine()
        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "rigA|60fps|8cams",
                noxReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 1.4, 2.6),
                ),
                referenceReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 0.6, 1.0),
                ),
                referenceConditionFingerprint = "rigA|30fps|4cams",
            ),
        )

        assertTrue(!result.evidenceQualified)
        assertTrue(result.blockingReasons.any { it.contains("condition-fingerprint") })
    }

    @Test
    fun insufficient_pairs_stay_provisional() {
        val engine = ReferenceParityCohortEngine(
            ReferenceParityCohortEngine.Config(minimumPairs = 4),
        )
        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "same",
                noxReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.4, 2.4),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.3, 2.3),
                ),
                referenceReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 0.6, 0.9),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 0.6, 0.9),
                ),
            ),
        )

        assertEquals(2, result.matchedPairs)
        assertTrue(!result.evidenceQualified)
        assertTrue(result.blockingReasons.any { it.contains("minimum-pairs") })
    }

    @Test
    fun rejected_runs_are_not_silently_counted_as_reference_quality() {
        val engine = ReferenceParityCohortEngine(
            ReferenceParityCohortEngine.Config(minimumPairs = 2),
        )
        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "same",
                noxReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 1.1, 1.8),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 1.2, 1.9, accepted = false),
                    report("take-3", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 1.3, 2.0),
                ),
                referenceReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 0.5, 0.8),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 0.5, 0.8),
                    report("take-3", CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE, 0.5, 0.8),
                ),
            ),
        )

        assertTrue(result.acceptedNoxPairs == 2)
        assertTrue(result.rejectedNoxPairs == 1)
        assertTrue(result.evidenceQualified)
    }

    @Test
    fun sparse_cohort_quantiles_use_conservative_tail_value() {
        val engine = ReferenceParityCohortEngine(
            ReferenceParityCohortEngine.Config(minimumPairs = 2),
        )

        fun make(
            id: String,
            p95: Double,
            p99: Double,
        ) = report(
            id,
            CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE,
            p95,
            p99,
        )

        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "same",
                noxReports = listOf(
                    make("a", 1.0, 2.0),
                    make("b", 1.1, 2.1),
                ),
                referenceReports = listOf(
                    make("a", 0.4, 0.8),
                    make("b", 0.5, 0.9),
                ),
            ),
        )

        assertEquals(1.1, result.aggregateNoxP95Millimeters, 1e-9)
        assertEquals(2.1, result.aggregateNoxP99Millimeters, 1e-9)
    }


    @Test
    fun rejected_reference_run_cannot_qualify_a_pair() {
        val engine = ReferenceParityCohortEngine(
            ReferenceParityCohortEngine.Config(minimumPairs = 2),
        )

        val acceptedReference =
            report(
                "take-1",
                CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE,
                0.6,
                0.9,
                accepted = true,
            )
        val rejectedReference =
            report(
                "take-2",
                CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE,
                0.6,
                0.9,
                accepted = false,
            )

        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "same",
                noxReports = listOf(
                    report("take-1", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.1, 2.1),
                    report("take-2", CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE, 1.0, 2.0),
                ),
                referenceReports = listOf(
                    acceptedReference,
                    rejectedReference,
                ),
            ),
        )

        assertTrue(!result.evidenceQualified)
        assertTrue(
            result.blockingReasons.any {
                it.contains("accepted-reference-pair-ratio")
            },
        )
    }

    @Test
    fun mismatched_reference_types_block_pairwise_qualification() {
        val engine = ReferenceParityCohortEngine(
            ReferenceParityCohortEngine.Config(minimumPairs = 1),
        )

        val result = engine.evaluate(
            ReferenceParityCohortEngine.Input(
                matchedConditionFingerprint = "same",
                noxReports = listOf(
                    report(
                        "take-1",
                        CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE,
                        1.1,
                        2.0,
                    ),
                ),
                referenceReports = listOf(
                    report(
                        "take-1",
                        CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE,
                        0.5,
                        0.9,
                    ),
                ),
            ),
        )

        assertTrue(!result.evidenceQualified)
        assertTrue(
            result.blockingReasons.contains("reference-type-mismatch"),
        )
    }

}
