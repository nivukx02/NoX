package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiCameraCalibrationQualityEngineTest {

    @Test
    fun wellSeparatedSynchronizedRigReceivesUsableScore() {
        val engine = MultiCameraCalibrationQualityEngine()
        val result = engine.evaluate(
            cameras = listOf(
                camera("A", Vec3(-0.9, 0.0, 2.5), Vec3(0.20, 0.0, -1.0)),
                camera("B", Vec3(0.9, 0.0, 2.5), Vec3(-0.20, 0.0, -1.0)),
                camera("C", Vec3(0.0, 1.2, 2.8), Vec3(0.0, -0.20, -1.0)),
            ),
            observedCoverage = 0.92,
        )

        assertTrue(result.cameraCount == 3)
        assertTrue(result.pairCount == 3)
        assertTrue(result.rigScore > 0.50)
        assertTrue(result.diagnostics.isNotEmpty())
    }

    @Test
    fun singleCameraRigIsNotMultiViewReady() {
        val engine = MultiCameraCalibrationQualityEngine()
        val result = engine.evaluate(
            cameras = listOf(
                camera("A", Vec3(0.0, 0.0, 2.0), Vec3(0.0, 0.0, -1.0)),
            ),
            observedCoverage = 0.90,
        )

        assertTrue(!result.ready)
        assertTrue(result.recommendations.isNotEmpty())
    }

    private fun camera(id: String, position: Vec3, forward: Vec3) =
        MultiCameraCalibrationQualityEngine.Camera(
            id = id,
            positionMeters = position,
            forward = forward,
            reprojectionErrorPx = 1.2,
            clockOffsetNanos = 500_000L,
            confidence = 0.92,
        )
}
