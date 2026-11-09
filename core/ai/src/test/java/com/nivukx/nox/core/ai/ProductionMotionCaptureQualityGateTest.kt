
package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProductionMotionCaptureQualityGateTest {

    @Test(expected = IllegalArgumentException::class)
    fun rejectsZeroBenchmarkRetention() {
        ProductionMotionCaptureQualityGate(maxBenchmarkSamples = 0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsZeroHealthEventRetention() {
        ProductionMotionCaptureQualityGate(maxHealthEvents = 0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteScaleDriftLimit() {
        ProductionMotionCaptureQualityGate.Contract(
            metricScaleDriftMaximumPercentPerMinute = Double.NaN,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInfiniteCatastrophicErrorLimit() {
        ProductionMotionCaptureQualityGate.Contract(
            catastrophicErrorMaximumMillimeters = Double.POSITIVE_INFINITY,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInvalidCatastrophicTailRatio() {
        ProductionMotionCaptureQualityGate.Contract(
            catastrophicTailMaximumRatio = 1.01,
        )
    }

    @Test
    fun cleanEvidenceCanReachReferenceBand() {
        val gate = ProductionMotionCaptureQualityGate()
        val decision = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 1_000_000L,
                p95ErrorMillimeters = 2.0,
                p99ErrorMillimeters = 4.5,
                temporalLagMilliseconds = 7.0,
                dropoutRatio = 0.003,
                synchronizationConfidence = 0.98,
                calibrationConfidence = 0.97,
                trackingConfidence = 0.96,
                crossViewAgreement = 0.94,
                viewDiversity = 0.90,
                identityContinuity = 0.97,
                surfaceTemporalStability = 0.95,
                faceEyeConfidence = 0.96,
                handConfidence = 0.94,
                meshTopologyIntegrity = 0.96,
                gpuHeadroom = 0.85,
                npuHeadroom = 0.88,
                cpuHeadroom = 0.82,
                metricScaleDriftPercentPerMinute = 0.20,
                global4DTemporalConsistency = 0.96,
                contactConsistency = 0.94,
            ),
        )
        assertTrue(decision.allowed)
        assertTrue(decision.contractSatisfied)
        assertEquals(
            ProductionMotionCaptureQualityGate.Band.REFERENCE,
            decision.band,
        )
        assertEquals(1, decision.recommendedFrameStride)
        assertTrue(decision.runtimeBudget.detectorScale > 0.0)
        assertTrue(decision.runtimeBudget.temporalFusionScale > 0.0)
        assertTrue(decision.runtimeBudget.memoryReserve > 0.0)
        gate.close()
    }

    @Test
    fun criticalRuntimePressureFailsClosed() {
        val gate = ProductionMotionCaptureQualityGate()
        val decision = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 2L,
                memoryPressure = 0.995,
                thermalRisk = 0.98,
            ),
        )
        assertFalse(decision.allowed)
        assertTrue(decision.hardBlock)
        assertTrue("memory-pressure-critical" in decision.blockers)
        assertTrue("thermal-risk-critical" in decision.blockers)
        gate.close()
    }

    @Test
    fun benchmarkBufferRemainsBounded() {
        val gate = ProductionMotionCaptureQualityGate(maxBenchmarkSamples = 32)
        repeat(100) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1,
                    region = ProductionMotionCaptureQualityGate.Region.TORSO,
                    systemErrorMillimeters = 2.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }
        assertEquals(32, gate.benchmarkSize())
        val report = gate.completeBenchmark()
        assertEquals(32, report.sampleCount)
        gate.close()
    }

    @Test
    fun unmatchedInfraredComparisonRemainsInconclusive() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(160) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1,
                    region = ProductionMotionCaptureQualityGate.Region.FULL_BODY,
                    systemErrorMillimeters = 1.2,
                    temporalLagMilliseconds = 5.0,
                ),
            )
        }
        val report = gate.completeBenchmark(
            referenceSystem =
                ProductionMotionCaptureQualityGate.ReferenceSystem.INFRARED_OPTICAL,
            referenceP95Millimeters = 2.0,
            referenceUncertaintyMillimeters = 0.25,
            matchedConditions = false,
        )
        assertTrue(report.referenceComparison != null)
        assertFalse(
            report.referenceComparison!!.statisticallySeparated,
        )
        assertEquals(
            "reference-conditions-not-matched",
            report.referenceComparison!!.inconclusiveReason,
        )
        gate.close()
    }

    @Test
    fun matchedReferenceComparisonExposesStatisticalSeparation() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(256) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1,
                    region = ProductionMotionCaptureQualityGate.Region.FULL_BODY,
                    systemErrorMillimeters = 1.0,
                    temporalLagMilliseconds = 4.0,
                    visibility = 0.95,
                    identityStable = true,
                ),
            )
        }
        val report = gate.completeBenchmark(
            referenceSystem =
                ProductionMotionCaptureQualityGate.ReferenceSystem.INFRARED_OPTICAL,
            referenceP95Millimeters = 3.0,
            referenceUncertaintyMillimeters = 0.10,
            matchedConditions = true,
        )
        val comparison = report.referenceComparison
        assertTrue(comparison != null)
        assertTrue(comparison!!.statisticallySeparated)
        assertTrue(comparison.separationRatio >= 1.96)
        assertTrue(comparison.effectMarginMillimeters > 0.0)
        assertTrue(report.identityFailureRatio == 0.0)
        gate.close()
    }

    @Test
    fun referenceQualificationUsesUncertaintyOfTheP95Tail() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(256) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region = ProductionMotionCaptureQualityGate.Region.FULL_BODY,
                    systemErrorMillimeters =
                        0.80 + (index % 32) * 0.02,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }

        val report = gate.completeBenchmark(
            referenceSystem =
                ProductionMotionCaptureQualityGate.ReferenceSystem.INFRARED_OPTICAL,
            referenceP95Millimeters = 1.50,
            referenceUncertaintyMillimeters = 0.02,
            matchedConditions = true,
        )
        val comparison = report.referenceComparison

        assertTrue(comparison != null)
        assertTrue(
            comparison!!.systemUncertaintyMillimeters.isFinite(),
        )
        assertTrue(
            comparison.systemUncertaintyMillimeters > 0.0,
        )
        assertTrue(
            comparison.combinedUncertaintyMillimeters >=
                comparison.systemUncertaintyMillimeters,
        )
        gate.close()
    }

    @Test
    fun benchmarkReportsInteractionFastMotionAndMultiPersonCohorts() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(180) { index ->
            val interaction = index % 3 == 0
            val fast = index % 4 == 0
            val multiPerson = index % 5 == 0
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region =
                        if (interaction) {
                            ProductionMotionCaptureQualityGate.Region.LEFT_HAND
                        } else {
                            ProductionMotionCaptureQualityGate.Region.TORSO
                        },
                    systemErrorMillimeters =
                        if (interaction || fast) 4.0 else 2.0,
                    temporalLagMilliseconds = 6.0,
                    visibility = if (multiPerson) 0.42 else 0.92,
                    occluded = multiPerson,
                    interaction = interaction,
                    fastMotion = fast,
                    multiPersonOcclusion = multiPerson,
                    identityExchangeRisk =
                        if (multiPerson) 0.70 else 0.10,
                    identityStable = index % 17 != 0,
                ),
            )
        }
        val report = gate.completeBenchmark()
        assertTrue(report.interactionSampleRatio > 0.30)
        assertTrue(report.fastMotionSampleRatio > 0.20)
        assertTrue(report.multiPersonOcclusionSampleRatio > 0.15)
        assertTrue(report.interactionP95Millimeters >= 4.0)
        assertTrue(report.fastMotionP95Millimeters >= 4.0)
        assertTrue(report.multiPersonOcclusionP95Millimeters >= 4.0)
        assertTrue(report.identityExchangeFailureRatio >= 0.0)
        gate.close()
    }

    @Test
    fun scaleAndGlobalTemporalEvidenceArePartOfTheContract() {
        val gate = ProductionMotionCaptureQualityGate()
        val decision = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 3L,
                p95ErrorMillimeters = 2.0,
                p99ErrorMillimeters = 4.0,
                temporalLagMilliseconds = 6.0,
                synchronizationConfidence = 0.95,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.95,
                crossViewAgreement = 0.92,
                viewDiversity = 0.90,
                identityContinuity = 0.94,
                metricScaleDriftPercentPerMinute = 3.0,
                global4DTemporalConsistency = 0.60,
                contactConsistency = 0.60,
            ),
        )
        assertFalse(decision.contractSatisfied)
        assertTrue("metric-scale-drift-high" in decision.warnings)
        assertTrue("global-4d-temporal-consistency-low" in decision.warnings)
        assertTrue("contact-consistency-low" in decision.warnings)
        gate.close()
    }


    @Test
    fun catastrophicTailPreventsBenchmarkPass() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(1000) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region = ProductionMotionCaptureQualityGate.Region.TORSO,
                    systemErrorMillimeters = if (index == 999) 150.0 else 2.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }
        val report = gate.completeBenchmark()
        assertFalse(report.gatePassed)
        assertTrue(report.catastrophicTailRatio > 0.0)
        assertTrue(report.catastrophicErrorMaximumMillimeters >= 150.0)
        gate.close()
    }

    @Test
    fun benchmarkBelowEmpiricalMinimumCannotPassQualification() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(599) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region = ProductionMotionCaptureQualityGate.Region.TORSO,
                    systemErrorMillimeters = 1.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }

        val report = gate.completeBenchmark()
        assertEquals(599, report.sampleCount)
        assertFalse(report.gatePassed)
        gate.close()
    }

    @Test
    fun cleanBenchmarkPassesCatastrophicTailGate() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(1000) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region = ProductionMotionCaptureQualityGate.Region.TORSO,
                    systemErrorMillimeters = 2.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }
        val report = gate.completeBenchmark()
        assertTrue(report.gatePassed)
        assertEquals(0.0, report.catastrophicTailRatio)
        assertEquals(2.0, report.catastrophicErrorMaximumMillimeters)
        gate.close()
    }

    @Test
    fun matchedButInconclusiveReferenceCannotPassBenchmarkGate() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(160) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region = ProductionMotionCaptureQualityGate.Region.FULL_BODY,
                    systemErrorMillimeters = 2.0,
                    temporalLagMilliseconds = 4.0,
                ),
            )
        }

        val report =
            gate.completeBenchmark(
                referenceSystem =
                    ProductionMotionCaptureQualityGate.ReferenceSystem
                        .INFRARED_OPTICAL,
                referenceP95Millimeters = 2.1,
                referenceUncertaintyMillimeters = 2.0,
                matchedConditions = true,
            )

        assertTrue(report.referenceComparison != null)
        assertTrue(
            report.referenceComparison!!.matchedConditions,
        )
        assertFalse(
            report.referenceComparison!!.statisticallySeparated,
        )
        assertFalse(report.gatePassed)
        gate.close()
    }

    @Test
    fun pairedReferenceEvidenceIsReportedWithoutChangingInconclusiveRules() {
        val gate = ProductionMotionCaptureQualityGate()
        repeat(160) { index ->
            gate.addBenchmarkSample(
                ProductionMotionCaptureQualityGate.BenchmarkSample(
                    timestampNanos = index.toLong() + 1L,
                    region = ProductionMotionCaptureQualityGate.Region.FULL_BODY,
                    systemErrorMillimeters = 1.0,
                    temporalLagMilliseconds = 4.0,
                    referenceErrorMillimeters = 2.0,
                ),
            )
        }
        val report = gate.completeBenchmark(
            referenceSystem =
                ProductionMotionCaptureQualityGate.ReferenceSystem.INFRARED_OPTICAL,
            referenceP95Millimeters = 2.0,
            referenceUncertaintyMillimeters = 0.25,
            matchedConditions = true,
        )
        val comparison = report.referenceComparison
        assertTrue(comparison != null)
        assertEquals(160, comparison!!.pairedSampleCount)
        assertTrue(comparison.pairedSystemWinRate == 1.0)
        assertTrue(comparison.pairedMedianDeltaMillimeters < 0.0)
        gate.close()
    }
    @Test
    fun denseLandmarkEvidenceIsAuthoritativeWhenHealthy() {
        val gate = ProductionMotionCaptureQualityGate()
        val decision = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 10L,
                p95ErrorMillimeters = 2.0,
                p99ErrorMillimeters = 4.0,
                synchronizationConfidence = 0.96,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.94,
                crossViewAgreement = 0.92,
                viewDiversity = 0.90,
                identityContinuity = 0.94,
                denseLandmarkCoverage = 0.92,
                denseLandmarkVisibleCoverage = 0.90,
                denseLandmarkUncertaintyP95 = 0.14,
                denseLandmarkCrossViewAgreement = 0.88,
                denseLandmarkTemporalStability = 0.91,
                denseLandmarkContactCoverage = 0.65,
                denseLandmarkPredictionBridgeRatio = 0.12,
                denseLandmarkCrossViewEvidenceAvailable = true,
                denseLandmarkHardFailure = false,
            ),
        )
        assertTrue(decision.contractSatisfied)
        assertTrue(decision.allowed)
        gate.close()
    }

    @Test
    fun denseLandmarkUncertaintyCanHardBlockAuthoritativeCapture() {
        val gate = ProductionMotionCaptureQualityGate()
        val decision = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 11L,
                synchronizationConfidence = 0.96,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.94,
                crossViewAgreement = 0.90,
                viewDiversity = 0.90,
                identityContinuity = 0.94,
                denseLandmarkCoverage = 0.85,
                denseLandmarkVisibleCoverage = 0.84,
                denseLandmarkUncertaintyP95 = 0.72,
            ),
        )
        assertTrue(decision.hardBlock)
        assertTrue(
            decision.blockers.contains(
                "dense-landmark-uncertainty-critical",
            ),
        )
        gate.close()
    }

    @Test
    fun denseLandmarkCrossViewCanHardBlockWhenEvidenceExists() {
        val gate = ProductionMotionCaptureQualityGate()
        val decision = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 12L,
                synchronizationConfidence = 0.96,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.94,
                crossViewAgreement = 0.90,
                viewDiversity = 0.90,
                identityContinuity = 0.94,
                denseLandmarkCoverage = 0.85,
                denseLandmarkVisibleCoverage = 0.84,
                denseLandmarkUncertaintyP95 = 0.18,
                denseLandmarkCrossViewAgreement = 0.20,
                denseLandmarkCrossViewEvidenceAvailable = true,
            ),
        )
        assertTrue(decision.hardBlock)
        assertTrue(
            decision.blockers.contains(
                "dense-landmark-cross-view-critical",
            ),
        )
        gate.close()
    }


    @Test
    fun measuredSynchronizationDiagnosticsArePartOfProductionContract() {
        val gate = ProductionMotionCaptureQualityGate()

        val clean = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 4L,
                synchronizationConfidence = 0.98,
                synchronizationResidualP95Milliseconds = 1.5,
                synchronizationOutlierRatio = 0.01,
            ),
        )

        assertTrue(clean.allowed)
        assertFalse(
            clean.warnings.any {
                it.contains("sync-residual")
            },
        )

        val degraded = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 5L,
                synchronizationConfidence = 0.98,
                synchronizationResidualP95Milliseconds = 13.0,
                synchronizationOutlierRatio = 0.21,
            ),
        )

        assertTrue(
            degraded.warnings.contains(
                "sync-residual-p95-above-contract",
            ),
        )
        assertTrue(
            degraded.warnings.contains(
                "sync-outlier-ratio-above-contract",
            ),
        )
        assertFalse(degraded.contractSatisfied)

        val critical = gate.evaluate(
            ProductionMotionCaptureQualityGate.Evidence(
                timestampNanos = 6L,
                synchronizationConfidence = 0.98,
                synchronizationResidualP95Milliseconds = 30.0,
                synchronizationOutlierRatio = 0.45,
            ),
        )

        assertTrue(critical.hardBlock)
        assertTrue(
            critical.blockers.contains(
                "sync-evidence-critical",
            ),
        )

        gate.close()
    }

}
