package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CameraFeatureMatrix
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class CameraTopologyEvidenceEngineTest {

    @Test
    fun calibratedLogicalCameraIsDistinguishedFromApproximateSync() {
        val base = matrix("0")
        val calibrated = base.copy(
            cameraId = "1",
            supportsLogicalMultiCamera = true,
            logicalSensorSyncType =
                android.hardware.camera2.CameraMetadata
                    .LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED,
            physicalCameraIds = setOf("0", "1"),
        )
        val approximate = base.copy(
            cameraId = "2",
            supportsLogicalMultiCamera = true,
            logicalSensorSyncType =
                android.hardware.camera2.CameraMetadata
                    .LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_APPROXIMATE,
        )

        val engine = CameraTopologyEvidenceEngine()

        val hardware =
            engine.evaluate(
                matrices = listOf(base, calibrated),
            )
        val software =
            engine.evaluate(
                matrices = listOf(base, approximate),
            )

        assertEquals(
            CameraTopologyEvidenceEngine.Mode.LOGICAL_CALIBRATED,
            hardware.mode,
        )
        assertEquals(
            CameraTopologyEvidenceEngine.Mode.LOGICAL_APPROXIMATE,
            software.mode,
        )
        assertTrue(hardware.score > software.score)
    }

    @Test
    fun independentCamerasRemainExplicitlyIndependent() {
        val result =
            CameraTopologyEvidenceEngine().evaluate(
                matrices = listOf(matrix("0"), matrix("1")),
            )

        assertEquals(
            CameraTopologyEvidenceEngine.Mode.INDEPENDENT_MULTI_CAMERA,
            result.mode,
        )
        assertTrue(result.score > 0.50)
    }

    private fun matrix(id: String) =
        CameraFeatureMatrix(
            cameraId = id,
            supportsRaw = false,
            rawWidth = 0,
            rawHeight = 0,
            supportsDepth16 = false,
            depthWidth = 0,
            depthHeight = 0,
            supportsManualSensor = false,
            supportsOis = false,
            supportsPdaf = false,
            supportsConcurrentCamera = true,
            maxAnalysisWidth = 1920,
            maxAnalysisHeight = 1080,
            maxFps = 60,
            recommendedProfile = CameraCaptureProfile(),
        )
}
