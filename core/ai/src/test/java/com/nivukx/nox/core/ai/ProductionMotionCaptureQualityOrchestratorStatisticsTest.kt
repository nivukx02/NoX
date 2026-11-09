package com.nivukx.nox.core.ai

import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertTrue

class ProductionMotionCaptureQualityOrchestratorStatisticsTest {

    @Test
    fun referenceComparisonExposesQuantileConfidenceBounds() {
        val engine =
            ProductionMotionCaptureQualityOrchestrator()

        repeat(600) { index ->
            val error =
                1.8 +
                    0.25 *
                    kotlin.math.abs(
                        cos(index.toDouble()),
                    )

            engine.addBenchmarkSample(
                ProductionMotionCaptureQualityOrchestrator.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region =
                        ProductionMotionCaptureQualityOrchestrator.MotionRegion.FULL_BODY,
                    errorMillimeters = error,
                    temporalLagMilliseconds = 3.0,
                    confidence = 0.98,
                ),
            )
        }

        val report =
            engine.completeBenchmark(
                referenceClass =
                    ProductionMotionCaptureQualityOrchestrator.ReferenceClass
                        .INFRARED_OPTICAL,
                referenceP95ErrorMillimeters = 2.8,
                referenceUncertaintyMillimeters = 0.20,
                conditionsMatched = true,
            )

        assertTrue(
            report.overall.p95ConfidenceLowerMillimeters <=
                report.overall.p95ErrorMillimeters,
        )
        assertTrue(
            report.overall.p95ConfidenceUpperMillimeters >=
                report.overall.p95ErrorMillimeters,
        )
        assertTrue(
            report.referenceComparison != null,
        )
        assertTrue(
            report.referenceComparison!!.systemP95ConfidenceUpperMillimeters >=
                report.referenceComparison!!.systemP95ErrorMillimeters,
        )

        engine.close()
    }

    @Test
    fun smallBenchmarkProducesWideQuantileUncertainty() {
        val engine =
            ProductionMotionCaptureQualityOrchestrator()

        repeat(120) { index ->
            engine.addBenchmarkSample(
                ProductionMotionCaptureQualityOrchestrator.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region =
                        ProductionMotionCaptureQualityOrchestrator.MotionRegion.TORSO,
                    errorMillimeters = 2.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }

        val report =
            engine.completeBenchmark()

        assertTrue(
            report.overall.p95ConfidenceUpperMillimeters >=
                report.overall.p95ErrorMillimeters,
        )
        assertTrue(
            report.overall.p99ConfidenceUpperMillimeters >=
                report.overall.p99ErrorMillimeters,
        )

        engine.close()
    }
}
