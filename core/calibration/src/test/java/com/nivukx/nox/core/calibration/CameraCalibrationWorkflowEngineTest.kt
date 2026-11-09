package com.nivukx.nox.core.calibration

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraCalibrationWorkflowEngineTest {
    @Test
    fun insufficientCameraCountBlocksWorkflow() {
        val status = CameraCalibrationWorkflowEngine().evaluate(
            CameraCalibrationWorkflowEngine.Input(
                cameraCount = 1,
                coverage = 1.0,
                synchronized = true,
                clockSkewMillis = 0.5,
                intrinsicsSolved = true,
                extrinsicsSolved = true,
                quality = null,
            ),
        )
        assertEquals(
            CameraCalibrationWorkflowEngine.Phase.CONNECT,
            status.phase,
        )
        assertTrue(!status.ready)
    }

    @Test
    fun lowSubframeTemporalConfidenceBlocksReadyRig() {
        val quality =
            MultiCameraCalibrationQualityEngine.Result(
                ready = true,
                cameraCount = 3,
                pairCount = 3,
                coverageScore = 0.95,
                reprojectionScore = 0.94,
                timingScore = 0.96,
                baselineScore = 0.90,
                triangulationScore = 0.92,
                rigScore = 0.93,
                recommendations = emptyList(),
                diagnostics = emptyList(),
            )

        val status =
            CameraCalibrationWorkflowEngine().evaluate(
                CameraCalibrationWorkflowEngine.Input(
                    cameraCount = 3,
                    coverage = 0.95,
                    synchronized = true,
                    clockSkewMillis = 0.5,
                    intrinsicsSolved = true,
                    extrinsicsSolved = true,
                    quality = quality,
                    temporalAlignmentConfidence = 0.42,
                    subframeOffsetMillis = 7.0,
                ),
            )

        assertEquals(
            CameraCalibrationWorkflowEngine.Phase.SYNC,
            status.phase,
        )
        assertTrue(!status.ready)
        assertTrue(
            status.blocking.any {
                it.contains("sub-frame", ignoreCase = true)
            },
        )
    }

    @Test
    fun fullyVerifiedRigBecomesReady() {
        val quality = MultiCameraCalibrationQualityEngine.Result(
            ready = true,
            cameraCount = 3,
            pairCount = 3,
            coverageScore = 0.95,
            reprojectionScore = 0.94,
            timingScore = 0.96,
            baselineScore = 0.90,
            triangulationScore = 0.92,
            rigScore = 0.93,
            recommendations = emptyList(),
            diagnostics = emptyList(),
        )
        val status = CameraCalibrationWorkflowEngine().evaluate(
            CameraCalibrationWorkflowEngine.Input(
                cameraCount = 3,
                coverage = 0.95,
                synchronized = true,
                clockSkewMillis = 0.5,
                intrinsicsSolved = true,
                extrinsicsSolved = true,
                quality = quality,
            ),
        )
        assertEquals(
            CameraCalibrationWorkflowEngine.Phase.READY,
            status.phase,
        )
        assertEquals(1.0, status.progress, 0.0001)
    }
}
