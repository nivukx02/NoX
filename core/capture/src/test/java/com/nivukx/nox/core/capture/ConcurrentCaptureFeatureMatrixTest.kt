package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CameraFeatureMatrix
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class ConcurrentCaptureFeatureMatrixTest {

    @Test
    fun platformDeclaredCameraSetsBecomeRuntimeCandidates() {
        val profile = CameraCaptureProfile.adaptive(
            requestedWidth = 1920,
            requestedHeight = 1080,
            requestedFps = 60,
        )
        val matrices = listOf(
            matrix(
                id = "0",
                profile = profile,
                concurrent = setOf("0", "1"),
            ),
            matrix(
                id = "1",
                profile = profile,
                concurrent = setOf("0", "1"),
            ),
        )

        val plan =
            ConcurrentCaptureCapabilityEngine().planFromFeatureMatrices(
                matrices = matrices,
                synchronizationConfidence =
                    mapOf("0" to 0.98, "1" to 0.98),
                calibrationConfidence =
                    mapOf("0" to 0.97, "1" to 0.97),
                viewDiversity =
                    mapOf("0" to 0.86, "1" to 0.86),
                pairSkewMillis =
                    mapOf("0" to 1.0, "1" to 1.0),
                requestedCameraCount = 2,
            )

        assertEquals(listOf("0", "1"), plan.selectedCameraIds.sorted())
        assertTrue(plan.concurrentSet != null)
        assertTrue(plan.rigScore > 0.0)
    }

    @Test
    fun missingSynchronizationAndCalibrationEvidenceIsNotPromoted() {
        val profile = CameraCaptureProfile.adaptive(
            requestedWidth = 1280,
            requestedHeight = 720,
            requestedFps = 30,
        )
        val plan =
            ConcurrentCaptureCapabilityEngine().planFromFeatureMatrices(
                matrices = listOf(
                    matrix("0", profile, setOf("0", "1")),
                    matrix("1", profile, setOf("0", "1")),
                ),
                requestedCameraCount = 2,
            )

        assertTrue(!plan.admissible)
        assertTrue(
            plan.blockers.contains("SYNC_EVIDENCE_LOW") ||
                plan.blockers.contains("CALIBRATION_EVIDENCE_LOW"),
        )
    }

    private fun matrix(
        id: String,
        profile: CameraCaptureProfile,
        concurrent: Set<String>,
    ): CameraFeatureMatrix =
        CameraFeatureMatrix(
            cameraId = id,
            supports10BitDynamicRange = false,
            supportsRaw = false,
            rawWidth = 0,
            rawHeight = 0,
            supportsDepth16 = true,
            depthWidth = 640,
            depthHeight = 480,
            supportsManualSensor = true,
            supportsOis = true,
            supportsPdaf = true,
            supportsConcurrentCamera = true,
            maxAnalysisWidth = profile.analysisWidth,
            maxAnalysisHeight = profile.analysisHeight,
            maxFps = profile.fps,
            recommendedProfile = profile,
            concurrentCameraSets = listOf(concurrent),
        )
}
