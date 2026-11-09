package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndCaptureAuthorityEngineTest {

    private fun healthy(
        connectedCameras: Int = 4,
        authoritativeCameras: Int = 4,
        epipolar: Double = 0.96,
        blur: Double = 0.08,
        timing: Double = 0.96,
    ) =
        HighEndCaptureAuthorityEngine.Input(
            timestampNanos = 1_000_000_000L,
            connectedCameras = connectedCameras,
            authoritativeCameras = authoritativeCameras,
            calibrationConfidence = 0.97,
            synchronizationConfidence = timing,
            timingAuthorityConfidence = timing,
            temporalConsistency = 0.96,
            viewDiversity = 0.94,
            crossViewAgreement = 0.95,
            epipolarConsistency = epipolar,
            bodyMetricEvidenceAuthority = 0.95,
            faceMetricEvidenceAuthority = 0.95,
            handMetricEvidenceAuthority = 0.95,
            faceEpipolarConsistency = epipolar,
            handEpipolarConsistency = epipolar,
            reconstructionUncertaintyMillimeters = 0.40,
            motionP95ErrorMillimeters = 1.4,
            motionP99ErrorMillimeters = 2.8,
            temporalLagMilliseconds = 3.0,
            rollingShutterRisk = 0.08,
            motionBlurRisk = blur,
            cameraMotionRisk = 0.04,
            identityContinuity = 0.97,
            occlusionRisk = 0.08,
            contactConsistency = 0.94,
            thermalHeadroom = 0.88,
            memoryHeadroom = 0.86,
            gpuHeadroom = 0.90,
            npuHeadroom = 0.90,
            cpuHeadroom = 0.88,
            transportHeadroom = 0.92,
            observedGeometryRatio = 0.94,
            inferredGeometryRatio = 0.06,
            metricScaleUncertaintyRatio = 0.01,
            targetP95Millimeters = 5.0,
            targetRequiresIndependentReference = false,
        )

    @Test
    fun strongIndependentEvidenceBecomesHighConfidence() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(healthy())
        assertTrue(result.authoritative)
        assertFalse(result.hardBlocked)
        assertTrue(
            result.band == HighEndCaptureAuthorityEngine.Band.HIGH_CONFIDENCE ||
                result.band == HighEndCaptureAuthorityEngine.Band.AUTHORITATIVE,
        )
        assertTrue(result.score >= 0.78)
    }

    @Test
    fun severeEpipolarContradictionHardBlocksAuthority() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy(epipolar = 0.10),
            )
        assertTrue(result.hardBlocked)
        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.EPIPOLAR_CONSISTENCY_TOO_WEAK,
            ),
        )
        assertFalse(result.authoritative)
    }

    @Test
    fun weakFaceOrHandEpipolarEvidenceBlocksCrossEngineAuthority() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy().copy(
                    faceEpipolarConsistency = 0.12,
                ),
            )

        assertFalse(result.authoritative)
        assertTrue(result.hardBlocked)
        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.EPIPOLAR_CONSISTENCY_TOO_WEAK,
            ),
        )
    }

    @Test
    fun weakTwoViewTimingHardBlocksMultiViewAuthority() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy(
                    connectedCameras = 2,
                    authoritativeCameras = 2,
                    timing = 0.20,
                ),
            )
        assertTrue(result.hardBlocked)
        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.TIMING_TOO_WEAK_FOR_MULTI_VIEW,
            ),
        )
    }

    @Test
    fun excessiveBlurIsAHardRuntimeBlock() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy(blur = 0.80),
            )

        assertTrue(result.hardBlocked)
        assertTrue(!result.authoritative)
    }

    @Test
    fun weakCrossModalAuthorityBlocksAuthoritativeAdmission() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy().copy(
                    crossModalMetricAuthority = 0.22,
                    crossModalEpipolarAuthority = 0.24,
                ),
            )

        assertTrue(result.hardBlocked)
        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.METRIC_EVIDENCE_TOO_WEAK,
            ) ||
                result.band ==
                    HighEndCaptureAuthorityEngine.Band.GUARDED ||
                result.band ==
                    HighEndCaptureAuthorityEngine.Band.OBSERVATION_ONLY,
        )
        assertFalse(result.authoritative)
    }

    @Test
    fun excessiveBlurIsNeverHiddenByStrongGeometry() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy(blur = 0.80),
            )
        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.EXCESSIVE_OPTICAL_BLUR,
            ),
        )
        assertTrue(result.warnings.contains("optical-quality-limited"))
    }

    @Test
    fun ordinaryLiveCaptureDoesNotLoseAuthorityBecauseBenchmarkIsAbsent() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy().copy(
                    targetP95Millimeters = 5.0,
                    targetRequiresIndependentReference = false,
                    empiricalBenchmarkAccepted = false,
                ),
            )

        assertTrue(result.metricScore > 0.80)
        assertTrue(result.authoritative)
    }

    @Test
    fun referenceUncertaintyMustBeSmallRelativeToPrecisionTarget() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy().copy(
                    targetP95Millimeters = 0.75,
                    targetRequiresIndependentReference = true,
                    empiricalBenchmarkAccepted = true,
                    empiricalBenchmarkSampleCount = 1_200,
                    empiricalP95Millimeters = 0.52,
                    referenceUncertaintyMillimeters = 0.40,
                    externalReferenceDocumented = true,
                    referenceConditionsMatched = true,
                    motionP95ErrorMillimeters = 0.52,
                    motionP99ErrorMillimeters = 1.10,
                ),
            )

        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.REFERENCE_UNCERTAINTY_TOO_HIGH,
            ),
        )
        assertTrue(!result.precisionQualified)
    }

    @Test
    fun subMillimeterPrecisionNeedsMatchedEmpiricalEvidence() {
        val result =
            HighEndCaptureAuthorityEngine().evaluate(
                healthy().copy(
                    targetP95Millimeters = 0.75,
                    targetRequiresIndependentReference = true,
                    motionP95ErrorMillimeters = 0.50,
                    motionP99ErrorMillimeters = 1.00,
                ),
            )
        assertFalse(result.precisionQualified)
        assertTrue(
            result.blockCodes.contains(
                HighEndCaptureAuthorityEngine.BlockCode.EMPIRICAL_PRECISION_EVIDENCE_MISSING,
            ),
        )
        assertFalse(result.empiricalPrecisionSupported)
    }

    @Test(expected = IllegalArgumentException::class)
    fun authoritativeCameraCountCannotExceedConnectedCount() {
        HighEndCaptureAuthorityEngine.Input(
            timestampNanos = 0L,
            connectedCameras = 2,
            authoritativeCameras = 3,
        )
    }
}
