package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PrecisionCriticalPathEngineTest {

    private fun input(
        target: Double = 5.0,
        reprojection: Double = 1.0,
        baseline: Double = 1.0,
        parallax: Double = 12.0,
    ): PrecisionCriticalPathEngine.Input {
        val metrology = MetricPrecisionBudgetEngine().evaluate(
            MetricPrecisionBudgetEngine.Input(
                targetP95Millimeters = target,
                workingDistanceMeters = 2.0,
                cameraBaselineMeters = baseline,
                effectiveFocalLengthPixels = 2200.0,
                reprojectionErrorPixels = reprojection,
                subpixelLocalizationSigmaPixels = 0.08,
                photonShotNoiseSigmaPixels = 0.03,
                sensorReadNoiseSigmaPixels = 0.02,
                quantizationSigmaPixels = 0.01,
                triangulationParallaxDegrees = parallax,
                depthUncertaintyMeters = 0.002,
                poseTranslationUncertaintyMeters = 0.001,
                poseAngularUncertaintyDegrees = 0.03,
                calibrationConfidence = 0.97,
                synchronizationConfidence = 0.97,
                viewDiversity = 0.90,
                crossViewAgreement = 0.92,
                temporalSkewMilliseconds = 1.0,
                rollingShutterFraction = 0.05,
                metricScaleUncertaintyRatio = 0.0005,
                reconstructionPriorUncertaintyMeters = 0.0008,
            ),
        )
        return PrecisionCriticalPathEngine.Input(
            metrology = metrology,
            targetP95Millimeters = target,
            targetP99Millimeters = target * 2.4,
            calibrationConfidence = 0.97,
            synchronizationConfidence = 0.97,
            viewDiversity = 0.90,
            crossViewAgreement = 0.92,
            identityContinuity = 0.98,
            temporalLagMilliseconds = 2.0,
            rollingShutterRisk = 0.05,
            occlusionRisk = 0.02,
            metricScaleDriftPercentPerMinute = 0.01,
            thermalRisk = 0.10,
            memoryPressure = 0.10,
            surfaceTemporalStability = 0.96,
        )
    }

    @Test
    fun producesConcreteDominantTerm() {
        val decision =
            PrecisionCriticalPathEngine().evaluate(input())
        assertTrue(decision.terms.isNotEmpty())
        assertTrue(decision.dominantTerm != null)
        assertTrue(decision.mitigations.isNotEmpty())
    }

    @Test
    fun observationSamplingBecomesAnAttributedPrecisionTerm() {
        val decision =
            PrecisionCriticalPathEngine().evaluate(input())
        assertTrue(
            decision.terms.any {
                it.term ==
                    PrecisionCriticalPathEngine.Term.OBSERVATION_SAMPLING
            },
        )
    }

    @Test
    fun severeRollingShutterBecomesVisibleAsRuntimeRisk() {
        val base = input()
        val decision = PrecisionCriticalPathEngine().evaluate(
            base.copy(
                rollingShutterRisk = 0.95,
                temporalLagMilliseconds = 14.0,
            ),
        )
        assertTrue(
            decision.reasons.contains(
                "rolling-shutter-regime-dominant",
            ),
        )
        assertTrue(
            decision.terms.any {
                it.term ==
                    PrecisionCriticalPathEngine.Term.ROLLING_SHUTTER
            },
        )
    }

    @Test
    fun subTwoMillimeterRequiresEmpiricalEvidence() {
        val base = input(target = 2.0)
        val decision = PrecisionCriticalPathEngine().evaluate(base)
        assertTrue(decision.requiresMeasuredCertification)
        assertTrue(
            decision.reasons.contains(
                "sub-2mm-target-requires-empirical-certification",
            ),
        )
    }

    @Test
    fun lowCrossViewAgreementCreatesCorrelatedRisk() {
        val base = input()
        val decision = PrecisionCriticalPathEngine().evaluate(
            base.copy(crossViewAgreement = 0.20),
        )
        assertTrue(
            decision.terms.any {
                it.term ==
                    PrecisionCriticalPathEngine.Term.CORRELATED_RUNTIME
            },
        )
    }

    @Test
    fun decisionIsBoundedAndDeterministic() {
        val base = input()
        val engine = PrecisionCriticalPathEngine()
        val a = engine.evaluate(base)
        val b = engine.evaluate(base)
        assertEquals(a.dominantTerm, b.dominantTerm)
        assertEquals(a.modeledP95Millimeters, b.modeledP95Millimeters)
        assertEquals(a.terms, b.terms)
    }
}
