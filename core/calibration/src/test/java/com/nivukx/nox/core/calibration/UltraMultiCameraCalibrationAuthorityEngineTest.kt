package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UltraMultiCameraCalibrationAuthorityEngineTest {

    private fun camera(
        id: String,
        sector: Int,
        position: Vec3,
        forward: Vec3,
        reprojection: Double = 0.6,
        timingResidual: Double = 0.5,
    ) =
        UltraMultiCameraCalibrationAuthorityEngine.Camera(
            id = id,
            sector = sector,
            positionMeters = position,
            forward = forward.normalized(),
            reprojectionErrorPixels = reprojection,
            synchronizationConfidence = 0.98,
            calibrationConfidence = 0.98,
            coverage = 0.92,
            uncertainty =
                CalibrationUncertaintyProfile(
                    translationSigmaMeters = 0.0008,
                    rotationSigmaRadians = 0.0003,
                    evidenceConfidence = 0.98,
                ),
            timingResidualMillis = timingResidual,
        )

    @Test
    fun derivesBidirectionalGraphAndSelectsAuthority() {
        val engine =
            UltraMultiCameraCalibrationAuthorityEngine()

        val cameras =
            listOf(
                camera(
                    "a",
                    0,
                    Vec3(-2.0, 0.0, 2.0),
                    Vec3(1.0, 0.0, -1.0),
                ),
                camera(
                    "b",
                    1,
                    Vec3(2.0, 0.0, 2.0),
                    Vec3(-1.0, 0.0, -1.0),
                ),
                camera(
                    "c",
                    2,
                    Vec3(0.0, -2.0, 2.0),
                    Vec3(0.0, 1.0, -1.0),
                ),
                camera(
                    "d",
                    3,
                    Vec3(0.0, 2.0, 2.0),
                    Vec3(0.0, -1.0, -1.0),
                ),
            )

        val result = engine.evaluate(cameras)

        assertEquals(4, result.edges.size)
        assertEquals(4, result.authoritativeCameras.size)
        assertTrue(result.graphConnectivity > 0.90)
        assertTrue(result.globalConsistency > 0.50)
    }

    @Test
    fun identifiesActiveCalibrationTargetFromUncertainty() {
        val engine =
            UltraMultiCameraCalibrationAuthorityEngine()

        val stable =
            camera(
                "stable",
                0,
                Vec3(-2.0, 0.0, 2.0),
                Vec3(1.0, 0.0, -1.0),
            )
        val unstable =
            camera(
                id = "unstable",
                sector = 1,
                position = Vec3(2.0, 0.0, 2.0),
                forward = Vec3(-1.0, 0.0, -1.0),
                reprojection = 4.0,
                timingResidual = 7.0,
            ).copy(
                uncertainty =
                    CalibrationUncertaintyProfile(
                        translationSigmaMeters = 0.030,
                        rotationSigmaRadians = 0.008,
                        evidenceConfidence = 1.0,
                    ),
            )

        val result =
            engine.evaluate(
                listOf(
                    stable,
                    unstable,
                ),
            )

        assertTrue(
            result.activeCalibrationTargets.any {
                it.cameraId == "unstable" &&
                    it.priority >= 0.35
            },
        )
        assertTrue(result.warnings.isNotEmpty())
    }
}
