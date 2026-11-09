package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MatchedReferencePrecisionBenchmarkEngineTest {

    @Test
    fun matchedEvidenceProducesDescriptiveStatistics() {
        val engine =
            MatchedReferencePrecisionBenchmarkEngine(
                MatchedReferencePrecisionBenchmarkEngine.Config(
                    minimumSamples = 20,
                ),
            )

        val samples =
            (0 until 40).map { index ->
                MatchedReferencePrecisionBenchmarkEngine.Sample(
                    sampleId = index.toLong(),
                    timestampNanos = index * 10_000_000L,
                    systemErrorMillimeters =
                        0.55 +
                            (index % 5) * 0.03,
                    referenceErrorMillimeters =
                        0.65 +
                            (index % 4) * 0.03,
                    systemUncertaintyMillimeters = 0.08,
                    referenceUncertaintyMillimeters = 0.08,
                    matched = true,
                    confidence = 0.98,
                )
            }

        val result = engine.evaluate(samples)

        assertTrue(result.comparable)
        assertTrue(result.matchedSampleCount == 40)
        assertTrue(result.system.p95Millimeters.isFinite())
        assertTrue(result.reference.p95Millimeters.isFinite())
        assertTrue(result.paired.p95AbsoluteDeltaMillimeters.isFinite())
    }

    @Test
    fun highReferenceUncertaintyPreventsQualifiedComparison() {
        val engine =
            MatchedReferencePrecisionBenchmarkEngine(
                MatchedReferencePrecisionBenchmarkEngine.Config(
                    minimumSamples = 20,
                    maximumReferenceUncertaintyMillimeters = 0.50,
                ),
            )

        val samples =
            (0 until 40).map { index ->
                MatchedReferencePrecisionBenchmarkEngine.Sample(
                    sampleId = index.toLong(),
                    timestampNanos = index * 10_000_000L,
                    systemErrorMillimeters = 0.60,
                    referenceErrorMillimeters = 0.65,
                    referenceUncertaintyMillimeters = 0.90,
                    confidence = 0.98,
                )
            }

        val result = engine.evaluate(samples)

        assertFalse(result.referenceUncertaintyPass)
        assertTrue(
            result.warnings.any {
                it.contains("reference-uncertainty", ignoreCase = true)
            },
        )
    }

    @Test
    fun unmatchedEvidenceCannotBecomeQualified() {
        val engine =
            MatchedReferencePrecisionBenchmarkEngine()

        val result =
            engine.evaluate(
                listOf(
                    MatchedReferencePrecisionBenchmarkEngine.Sample(
                        sampleId = 1L,
                        timestampNanos = 1L,
                        systemErrorMillimeters = 0.5,
                        referenceErrorMillimeters = 0.6,
                        matched = false,
                    ),
                ),
            )

        assertFalse(result.qualified)
        assertFalse(result.comparable)
    }
}
