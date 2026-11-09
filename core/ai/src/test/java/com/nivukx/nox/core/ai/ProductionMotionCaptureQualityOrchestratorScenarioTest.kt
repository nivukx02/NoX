package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ProductionMotionCaptureQualityOrchestratorScenarioTest {

    @Test
    fun benchmarkReportSeparatesFastMotionAndOcclusionScenes() {
        val engine =
            ProductionMotionCaptureQualityOrchestrator()

        repeat(140) { index ->
            engine.addBenchmarkSample(
                ProductionMotionCaptureQualityOrchestrator.BenchmarkSample(
                    timestampNanos = index * 16_666_667L,
                    region =
                        ProductionMotionCaptureQualityOrchestrator.MotionRegion.FULL_BODY,
                    errorMillimeters =
                        if (index < 70) 2.0 else 4.0,
                    temporalLagMilliseconds = 5.0,
                    visibility =
                        if (index < 70) 0.95 else 0.42,
                    occluded = index >= 70,
                    fastMotion = index < 35,
                    multiPersonOcclusion = index >= 70,
                    identityExchangeRisk =
                        if (index >= 70) 0.35 else 0.05,
                    identityStable = true,
                    confidence = 0.95,
                    referenceErrorMillimeters = 2.5,
                ),
            )
        }

        val report =
            engine.completeBenchmark(
                referenceClass =
                    ProductionMotionCaptureQualityOrchestrator.ReferenceClass.INFRARED_OPTICAL,
                referenceP95ErrorMillimeters = 2.5,
                referenceUncertaintyMillimeters = 0.20,
                conditionsMatched = true,
            )

        assertTrue(report.scenarioStatistics.isNotEmpty())
        assertEquals(
            3,
            report.scenarioStatistics.keys.size,
        )
        assertTrue(
            report.scenarioStatistics.values.any {
                it.scenario ==
                    ProductionMotionCaptureQualityOrchestrator.Scenario.FAST_MOTION
            },
        )
        assertTrue(
            report.scenarioStatistics.values.any {
                it.scenario ==
                    ProductionMotionCaptureQualityOrchestrator.Scenario.OCCLUSION_HEAVY
            },
        )
        assertTrue(
            report.scenarioReferenceComparisons[
                ProductionMotionCaptureQualityOrchestrator.Scenario.FAST_MOTION
            ] != null,
        )
        assertTrue(
            report.scenarioReferenceComparisons[
                ProductionMotionCaptureQualityOrchestrator.Scenario.OCCLUSION_HEAVY
            ] != null,
        )

        engine.close()
    }
}
