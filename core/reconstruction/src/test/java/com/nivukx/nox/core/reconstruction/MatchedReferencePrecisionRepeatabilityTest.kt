package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MatchedReferencePrecisionRepeatabilityTest {

    @Test
    fun multipleRunsProduceRepeatabilityStatistics() {
        val engine =
            MatchedReferencePrecisionBenchmarkEngine(
                MatchedReferencePrecisionBenchmarkEngine.Config(
                    minimumSamples = 20,
                    minimumRuns = 3,
                ),
            )

        val samples =
            (0 until 60).map { index ->
                val run = index / 20
                MatchedReferencePrecisionBenchmarkEngine.Sample(
                    sampleId = index.toLong(),
                    captureRunId = "run-$run",
                    timestampNanos =
                        index * 10_000_000L,
                    systemErrorMillimeters =
                        0.55 +
                            run * 0.02 +
                            (index % 5) * 0.01,
                    referenceErrorMillimeters = 0.90,
                    systemUncertaintyMillimeters = 0.08,
                    referenceUncertaintyMillimeters = 0.08,
                    confidence = 0.99,
                )
            }

        val result = engine.evaluate(samples)

        assertTrue(result.repeatability.runCount == 3)
        assertTrue(result.repeatability.score > 0.70)
        assertTrue(result.paired.meanSignedDeltaMillimeters < 0.0)
    }

    @Test
    fun unstableRunToRunVariationCannotQualify() {
        val engine =
            MatchedReferencePrecisionBenchmarkEngine(
                MatchedReferencePrecisionBenchmarkEngine.Config(
                    minimumSamples = 20,
                    minimumRuns = 3,
                    maximumRunToRunSpreadMillimeters = 0.20,
                ),
            )

        val samples =
            (0 until 60).map { index ->
                val run = index / 20
                MatchedReferencePrecisionBenchmarkEngine.Sample(
                    sampleId = index.toLong(),
                    captureRunId = "run-$run",
                    timestampNanos =
                        index * 10_000_000L,
                    systemErrorMillimeters =
                        if (run == 1) 0.50 else 2.80,
                    referenceErrorMillimeters = 0.70,
                    referenceUncertaintyMillimeters = 0.08,
                    confidence = 0.99,
                )
            }

        val result = engine.evaluate(samples)

        assertTrue(
            result.repeatability.runToRunSpreadMillimeters >
                0.20,
        )
        assertFalse(result.qualified)
        assertTrue(
            result.blockers.contains(
                "repeatability-exceeds-contract",
            ),
        )
    }
}
