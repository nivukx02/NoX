package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProductionMotionCaptureQualityOrchestratorTest {

    @Test
    fun cleanEvidenceCanReachReferenceBand() {
        val engine = ProductionMotionCaptureQualityOrchestrator()
        val decision = engine.evaluate(
            ProductionMotionCaptureQualityOrchestrator.Evidence(
                timestampNanos = 1_000_000_000L,
                p95ErrorMillimeters = 2.2,
                p99ErrorMillimeters = 5.0,
                temporalLagMilliseconds = 7.0,
                dropoutRatio = 0.004,
                synchronizationConfidence = 0.98,
                calibrationConfidence = 0.97,
                trackingConfidence = 0.96,
                crossViewAgreement = 0.92,
                viewDiversity = 0.88,
                identityContinuity = 0.97,
                surfaceTemporalStability = 0.94,
                faceEyeConfidence = 0.95,
                hand3dConfidence = 0.94,
                gpuHeadroom = 0.86,
                npuHeadroom = 0.88,
                cpuHeadroom = 0.82,
            ),
        )
        assertTrue(decision.captureAllowed)
        assertTrue(decision.contractSatisfied)
        assertTrue(decision.isReferenceGrade)
        engine.close()
    }

    @Test
    fun criticalRuntimePressureBlocksUnsafeCapture() {
        val engine = ProductionMotionCaptureQualityOrchestrator()
        val decision = engine.evaluate(
            ProductionMotionCaptureQualityOrchestrator.Evidence(
                timestampNanos = 2L,
                memoryPressure = 0.995,
                thermalRisk = 0.98,
            ),
        )
        assertFalse(decision.captureAllowed)
        assertTrue(decision.hardSafetyBlock)
        assertTrue("memory-pressure-critical" in decision.blockingReasons)
        assertTrue("thermal-risk-critical" in decision.blockingReasons)
        engine.close()
    }

    @Test
    fun infraredComparisonStaysInconclusiveWithoutMatchedConditions() {
        val engine = ProductionMotionCaptureQualityOrchestrator()
        repeat(160) { index ->
            engine.addBenchmarkSample(
                ProductionMotionCaptureQualityOrchestrator.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region =
                        ProductionMotionCaptureQualityOrchestrator.MotionRegion.FULL_BODY,
                    errorMillimeters = 1.0,
                    temporalLagMilliseconds = 5.0,
                ),
            )
        }
        val report = engine.completeBenchmark(
            referenceClass =
                ProductionMotionCaptureQualityOrchestrator.ReferenceClass
                    .INFRARED_OPTICAL,
            referenceP95ErrorMillimeters = 2.0,
            referenceUncertaintyMillimeters = 0.25,
            conditionsMatched = false,
        )
        assertTrue(report.referenceComparison != null)
        assertFalse(report.referenceComparison!!.statisticallyLowerErrorThanReference)
        assertTrue(report.referenceComparison!!.inconclusiveReason != null)
        engine.close()
    }

    @Test
    fun catastrophicTailErrorsFailTheBenchmarkGate() {
        val engine = ProductionMotionCaptureQualityOrchestrator()
        repeat(500) { i ->
            engine.addBenchmarkSample(
                ProductionMotionCaptureQualityOrchestrator.BenchmarkSample(
                    timestampNanos = i.toLong() + 1L,
                    region = ProductionMotionCaptureQualityOrchestrator.MotionRegion.TORSO,
                    errorMillimeters = if (i == 17) 16.0 else 2.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }
        val report = engine.completeBenchmark(
            contract =
                ProductionMotionCaptureQualityOrchestrator.QualityContract(
                    maximumCatastrophicErrorRatio = 0.0,
                ),
        )
        assertFalse(report.qualityGate)
        assertTrue(report.catastrophicErrorCount == 1)
        assertTrue(report.catastrophicErrorRatio > 0.0)
        engine.close()
    }

    @Test
    fun benchmarkWindowIsBounded() {
        val engine =
            ProductionMotionCaptureQualityOrchestrator(
                maximumBenchmarkSamples = 32,
            )
        repeat(100) { index ->
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
        val report = engine.completeBenchmark()
        assertTrue(report.sampleCount == 32)
        engine.close()
    }
}
