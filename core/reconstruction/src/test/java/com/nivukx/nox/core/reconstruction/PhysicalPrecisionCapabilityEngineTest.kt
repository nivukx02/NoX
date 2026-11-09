package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhysicalPrecisionCapabilityEngineTest {
    @Test
    fun liveEvidenceRequiresReferenceForCertification() {
        val decision = PhysicalPrecisionCapabilityEngine().evaluate(strongInput(false, null, null))
        assertTrue(decision.liveSubMillimeterCapable)
        assertFalse(decision.measuredSubMillimeterQualified)
        assertEquals(PhysicalPrecisionCapabilityEngine.State.HIGH_PRECISION, decision.state)
    }

    @Test
    fun measuredReferenceCanOpenSubMillimeterGate() {
        val decision = PhysicalPrecisionCapabilityEngine().evaluate(strongInput(true, 0.72, 0.18))
        assertTrue(decision.liveSubMillimeterCapable)
        assertTrue(decision.measuredSubMillimeterQualified)
        assertEquals(PhysicalPrecisionCapabilityEngine.State.SUB_MILLIMETER_QUALIFIED, decision.state)
    }

    @Test
    fun referenceBenchmarkIsMappedIntoLiveGate() {
        val benchmark =
            ReferenceMetrologyBenchmarkEngine.Result.empty().copy(
                totalSamples = 640,
                validSamples = 640,
                invalidSamples = 0,
                dropoutFraction = 0.0,
                independentSessions = 4,
                subjects = 1,
                joints = 24,
                global =
                    ReferenceMetrologyBenchmarkEngine.Result
                        .empty()
                        .global
                        .copy(
                            count = 640,
                            meanMillimeters = 0.32,
                            meanAbsoluteMillimeters = 0.34,
                            rmseMillimeters = 0.41,
                            p50Millimeters = 0.29,
                            p95Millimeters = 0.72,
                            p99Millimeters = 0.86,
                            maximumMillimeters = 0.95,
                            standardDeviationMillimeters = 0.20,
                            truenessMillimeters = 0.32,
                        ),
                referenceUncertaintyP95Millimeters = 0.18,
                combinedP95Millimeters = 0.74,
                repeatabilityP95Millimeters = 0.68,
                dynamicP95Millimeters = 0.81,
                staticP95Millimeters = 0.55,
                dynamicSampleCount = 320,
                staticSampleCount = 320,
                qualificationReady = true,
                qualificationReasons = emptyList(),
                warnings = emptyList(),
            )

        val decision =
            PhysicalPrecisionCapabilityEngine()
                .evaluateWithReferenceBenchmark(
                    liveInput =
                        strongInput(
                            benchmarkArmed = false,
                            empiricalP95 = null,
                            referenceUncertainty = null,
                        ),
                    benchmark = benchmark,
                )

        assertTrue(decision.measuredSubMillimeterQualified)
        assertEquals(
            PhysicalPrecisionCapabilityEngine.EvidenceLevel.EMPIRICAL_AND_LIVE,
            decision.evidenceLevel,
        )
    }

    @Test
    fun calibrationFailureIsFailClosed() {
        val decision = PhysicalPrecisionCapabilityEngine().evaluate(strongInput().copy(calibrationConfidence = 0.71))
        assertEquals(PhysicalPrecisionCapabilityEngine.State.EVIDENCE_ONLY, decision.state)
        assertEquals(PhysicalPrecisionCapabilityEngine.Bottleneck.CALIBRATION, decision.bottleneck)
        assertFalse(decision.captureAdmission)
    }

    @Test
    fun historyIsBounded() {
        val engine = PhysicalPrecisionCapabilityEngine(
            PhysicalPrecisionCapabilityEngine.Config(maximumHistory = 8),
        )
        repeat(32) { engine.evaluate(strongInput(true, 0.72, 0.18, it + 1L)) }
        assertEquals(8, engine.recent(100).size)
        engine.reset()
        assertEquals(PhysicalPrecisionCapabilityEngine.State.NO_DATA, engine.latest().state)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFinitePrecisionBounds() {
        PhysicalPrecisionCapabilityEngine.Config(
            highPrecisionBoundMillimeters = Double.NaN,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInvertedPrecisionBounds() {
        PhysicalPrecisionCapabilityEngine.Config(
            highPrecisionBoundMillimeters = 2.0,
            subMillimeterBoundMillimeters = 3.0,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsEvidenceThresholdsOutsideUnitInterval() {
        PhysicalPrecisionCapabilityEngine.Config(
            calibrationMinimum = 1.01,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsUnboundedHistoryConfiguration() {
        PhysicalPrecisionCapabilityEngine.Config(
            maximumHistory = 0,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInvalidEmpiricalQualificationRequirements() {
        PhysicalPrecisionCapabilityEngine.Config(
            minimumEmpiricalSamples = 0,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteGlobalRigEvidence() {
        strongInput().copy(globalRigTrust = Double.NaN)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNegativeSpatialUncertainty() {
        strongInput().copy(reconstructionUncertaintyP95Meters = -0.001)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOutOfRangeGlobalTemporalEvidence() {
        strongInput().copy(global4DScore = 1.2)
    }

    private fun strongInput(
        benchmarkArmed: Boolean = true,
        empiricalP95: Double? = 0.72,
        referenceUncertainty: Double? = 0.18,
        timestampNanos: Long = 1L,
    ) = PhysicalPrecisionCapabilityEngine.Input(
        timestampNanos = timestampNanos,
        reconstructionUncertaintyP95Meters = 0.000065,
        directionalMajor95Meters = 0.000080,
        directionalDepth95Meters = 0.000090,
        geometricInformationP95MajorMeters = 0.000075,
        geometricInformationP95DepthMeters = 0.000085,
        denseSurfaceUncertaintyP95Meters = 0.000100,
        calibrationConfidence = 0.985,
        synchronizationConfidence = 0.987,
        temporalAlignmentConfidence = 0.972,
        temporalAlignmentSpreadMillis = 0.70,
        crossViewAgreement = 0.94,
        viewDiversity = 0.91,
        cameraInformationGain = 0.90,
        cameraCount = 6,
        motionUncertaintyConfidence = 0.96,
        motionUncertaintyRadiusMeters = 0.000055,
        captureTimingConfidence = 0.97,
        captureTimingJitterMillis = 0.45,
        temporalStability = 0.95,
        occlusionRisk = 0.04,
        runtimeHealthy = true,
        benchmarkArmed = benchmarkArmed,
        calibrationUncertaintyCoverage = 0.96,
        calibrationUncertaintyEvidenceConfidence = 0.94,
        empiricalP95Millimeters = empiricalP95,
        empiricalReferenceUncertaintyMillimeters = referenceUncertainty,
        empiricalSampleCount = 640,
        empiricalIndependentSessions = 4,
    )
}
