package com.nivukx.nox.core.reconstruction

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class CapturePrecisionBenchmarkReferenceComparisonTest {

    private val engine =
        CapturePrecisionBenchmarkEngine(
            CapturePrecisionBenchmarkEngine.Config(
                minimumSamples = 24,
            ),
        )

    private fun report(
        type: CapturePrecisionBenchmarkEngine.ReferenceType,
        uncertaintyMm: Double?,
        p95: Double,
        p99: Double,
        jitter: Double,
        dropout: Double,
        lag: Double,
        sampleCount: Int = 100,
    ): CapturePrecisionBenchmarkEngine.Report =
        CapturePrecisionBenchmarkEngine.Report(
            captureId = "capture-$type",
            referenceType = type,
            referenceUncertaintyMillimeters = uncertaintyMm,
            aggregate =
                CapturePrecisionBenchmarkEngine.Aggregate(
                    sampleCount = sampleCount,
                    validCount =
                        (
                            sampleCount *
                                (1.0 - dropout)
                            ).toInt(),
                    jointCount = 1,
                    dropoutRatio = dropout,
                    meanErrorMm = p95 * 0.40,
                    rmsErrorMm = p95 * 0.70,
                    p50ErrorMm = p95 * 0.45,
                    p95ErrorMm = p95,
                    p99ErrorMm = p99,
                    maxErrorMm = p99 * 1.10,
                    jitterRmsMm = jitter,
                    temporalLagMs = lag,
                    scaleRatio = 1.0,
                    scaleDriftPercentPerMinute = 0.08,
                    surface = null,
                ),
            jointMetrics = emptyList(),
            gates = emptyList(),
            confidence = 0.95,
            accepted = true,
            warnings = emptyList(),
        )

    @Test
    fun matchedLowerErrorReferenceCanEarnBetterVerdict() {
        val reference =
            report(
                type =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .OPTICAL_REFERENCE,
                uncertaintyMm = 0.35,
                p95 = 2.0,
                p99 = 3.5,
                jitter = 1.2,
                dropout = 0.020,
                lag = 8.0,
            )
        val nox =
            report(
                type =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .OPTICAL_REFERENCE,
                uncertaintyMm = 0.35,
                p95 = 1.0,
                p99 = 2.0,
                jitter = 0.5,
                dropout = 0.005,
                lag = 2.0,
            )

        val comparison =
            engine.compareAgainstReference(
                nox = nox,
                reference = reference,
            )

        assertEquals(
            CapturePrecisionBenchmarkEngine.ReferenceVerdict.BETTER,
            comparison.verdict,
        )
        assertTrue(comparison.p95DeltaMm < 0.0)
        assertTrue(comparison.uncertaintyAware)
        assertTrue(comparison.sampleComparable)
    }

    @Test
    fun missingReferenceUncertaintyPreventsBetterClaim() {
        val reference =
            report(
                type =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .OPTICAL_REFERENCE,
                uncertaintyMm = null,
                p95 = 2.0,
                p99 = 3.0,
                jitter = 1.0,
                dropout = 0.01,
                lag = 5.0,
            )
        val nox =
            report(
                type =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .OPTICAL_REFERENCE,
                uncertaintyMm = 0.25,
                p95 = 0.8,
                p99 = 1.4,
                jitter = 0.4,
                dropout = 0.002,
                lag = 2.0,
            )

        val comparison =
            engine.compareAgainstReference(
                nox = nox,
                reference = reference,
            )

        assertEquals(
            CapturePrecisionBenchmarkEngine.ReferenceVerdict.INCONCLUSIVE,
            comparison.verdict,
        )
        assertTrue(
            comparison.rationale.contains(
                "reference-uncertainty-not-documented",
            ),
        )
    }

    @Test
    fun differentReferenceFamiliesPreventCrossFamilyClaim() {
        val marker =
            report(
                type =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .MARKER_REFERENCE,
                uncertaintyMm = 0.25,
                p95 = 2.0,
                p99 = 3.0,
                jitter = 1.0,
                dropout = 0.01,
                lag = 5.0,
            )
        val optical =
            report(
                type =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .OPTICAL_REFERENCE,
                uncertaintyMm = 0.25,
                p95 = 0.8,
                p99 = 1.4,
                jitter = 0.4,
                dropout = 0.002,
                lag = 2.0,
            )

        val comparison =
            engine.compareAgainstReference(
                nox = optical,
                reference = marker,
            )

        assertEquals(
            CapturePrecisionBenchmarkEngine.ReferenceVerdict.INCONCLUSIVE,
            comparison.verdict,
        )
        assertTrue(
            comparison.rationale.contains(
                "reference-family-mismatch",
            ),
        )
    }
}
