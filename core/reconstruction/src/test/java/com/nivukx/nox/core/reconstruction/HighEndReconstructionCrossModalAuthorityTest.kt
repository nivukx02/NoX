package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ProductionCaptureProfile
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndReconstructionCrossModalAuthorityTest {

    private fun cleanInput() =
        HighEndReconstructionQualityEngine.Input(
            timestampNanos = 1_000_000_000L,
            profile = ProductionCaptureProfile.LIVE_ACTION,
            reconstructionConfidence = 0.97,
            reprojectionErrorPx = 0.8,
            meanUncertaintyMeters = 0.012,
            geometricObservability = 0.96,
            temporalConfidence = 0.96,
            synchronizationConfidence = 0.98,
            calibrationConfidence = 0.98,
            trackingConfidence = 0.97,
            identitySwapRisk = 0.01,
            occlusionRisk = 0.02,
            cameraMotionRisk = 0.02,
            faceEyeConfidence = 0.96,
            handConfidence = 0.94,
            surfaceConfidence = 0.92,
            thermalScale = 0.92,
            memoryScale = 0.94,
            runtimeQualityScale = 0.94,
            runtimeInferenceScale = 0.96,
            gpuDispatchMillis = 5.0,
            connectedCameraCount = 3,
            observedSubjectCount = 1,
        )

    @Test
    fun weakCrossModalEpipolarAuthorityHardFails() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    timestampNanos = 2_000_000_000L,
                    faceEpipolarConsistency = 0.12,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertTrue(
            result.warnings.contains(
                "cross-modal-epipolar-authority-reduced",
            ),
        )
    }

    @Test
    fun weakCrossModalMetricAuthorityHardFails() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    timestampNanos = 3_000_000_000L,
                    handMetricEvidenceAuthority = 0.12,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertTrue(
            result.warnings.contains(
                "cross-modal-metric-authority-reduced",
            ),
        )
    }
}
