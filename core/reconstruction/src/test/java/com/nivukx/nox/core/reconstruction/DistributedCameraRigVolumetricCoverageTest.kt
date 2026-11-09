package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertTrue

class DistributedCameraRigVolumetricCoverageTest {

    @Test
    fun tetrahedralCameraPlacementHasHigherVolumetricCoverageThanPlanarWall() {
        val planar =
            listOf(
                camera("p0", Vec3(-2.0, 1.5, 0.0)),
                camera("p1", Vec3(-0.7, 1.5, 1.5)),
                camera("p2", Vec3(0.7, 1.5, 1.5)),
                camera("p3", Vec3(2.0, 1.5, 0.0)),
                camera("p4", Vec3(0.0, 1.5, -2.0)),
            )

        val volumetric =
            planar.mapIndexed { index, value ->
                when (index) {
                    0 ->
                        value.copy(
                            positionMeters =
                                Vec3(-2.0, 0.5, 0.0),
                        )
                    1 ->
                        value.copy(
                            positionMeters =
                                Vec3(-0.7, 2.6, 1.5),
                        )
                    2 ->
                        value.copy(
                            positionMeters =
                                Vec3(0.7, 0.4, 1.5),
                        )
                    3 ->
                        value.copy(
                            positionMeters =
                                Vec3(2.0, 2.4, 0.0),
                        )
                    else ->
                        value.copy(
                            positionMeters =
                                Vec3(0.0, 1.5, -2.0),
                        )
                }
            }

        val coordinator =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    targetSelectedCameras = 5,
                    maximumSelectedCameras = 5,
                ),
            )

        val planarPlan =
            coordinator.plan(
                timestampNanos = 1L,
                cameras = planar,
            )
        val volumetricPlan =
            coordinator.plan(
                timestampNanos = 2L,
                cameras = volumetric,
            )

        assertTrue(
            volumetricPlan.globalGeometryScore >=
                planarPlan.globalGeometryScore,
        )
    }

    @Test
    fun coplanarRigReportsVolumetricGeometryWarning() {
        val cameras =
            (0 until 6).map { index ->
                camera(
                    id = "wall-$index",
                    position =
                        Vec3(
                            index.toDouble() - 2.5,
                            1.5,
                            0.0,
                        ),
                )
            }

        val plan =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    targetSelectedCameras = 6,
                    maximumSelectedCameras = 6,
                ),
            ).plan(
                timestampNanos = 3L,
                cameras = cameras,
            )

        assertTrue(
            plan.warnings.any {
                it == "camera-volumetric-coverage-low"
            },
        )
    }

    private fun camera(
        id: String,
        position: Vec3,
    ) =
        CaptureRigObservabilityEngine.CameraEvidence(
            cameraId = id,
            role = DeviceRole.MASTER,
            positionMeters = position,
            forwardVector = Vec3(0.0, 0.0, -1.0),
            calibrationConfidence = 0.96,
            synchronizationConfidence = 0.96,
            trackingConfidence = 0.96,
            thermalHeadroom = 0.96,
            transportHeadroom = 0.96,
            active = true,
            imageErrorPx = 0.25,
            centroidConnectivity = 0.96,
            calibrationDriftStability = 0.96,
            trackedObjectCoverage = 0.96,
            surfaceInformationGain = 0.96,
            occlusionCoverage = 0.96,
            dynamicParallaxScore = 0.96,
            temporalResidualP95Milliseconds = 0.5,
            temporalOutlierRatio = 0.0,
            hardwareReadoutTimingConfidence = 0.98,
            hardwareReadoutP95JitterMilliseconds = 0.6,
            logicalSensorSyncCalibrated = true,
        )
}
