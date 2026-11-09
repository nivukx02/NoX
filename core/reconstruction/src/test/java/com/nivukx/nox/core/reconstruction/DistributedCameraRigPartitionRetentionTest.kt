package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedCameraRigPartitionRetentionTest {

    @Test
    fun allOneHundredTwentyEightBoundedCamerasRemainRepresented() {
        val coordinator =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    maximumGlobalCameras = 128,
                    maximumShards = 16,
                    localShardCapacity = 8,
                    targetSelectedCameras = 32,
                    maximumSelectedCameras = 32,
                ),
            )

        val cameras =
            (0 until 128).map { index ->
                CaptureRigObservabilityEngine.CameraEvidence(
                    cameraId = "cam-$index",
                    role = DeviceRole.CAMERA,
                    positionMeters =
                        Vec3(
                            x = (index % 32) * 1.7,
                            y = if (index % 3 == 0) 3.2 else 1.5,
                            z = (index / 32) * 1.9,
                        ),
                    forwardVector =
                        Vec3(0.0, 0.0, -1.0),
                    calibrationConfidence = 0.96,
                    synchronizationConfidence = 0.95,
                    trackingConfidence = 0.94,
                    thermalHeadroom = 0.92,
                    transportHeadroom = 0.94,
                    sector =
                        when (index % 8) {
                            0 -> CaptureRigObservabilityEngine.Sector.FRONT
                            1 -> CaptureRigObservabilityEngine.Sector.FRONT_RIGHT
                            2 -> CaptureRigObservabilityEngine.Sector.RIGHT
                            3 -> CaptureRigObservabilityEngine.Sector.REAR_RIGHT
                            4 -> CaptureRigObservabilityEngine.Sector.REAR
                            5 -> CaptureRigObservabilityEngine.Sector.REAR_LEFT
                            6 -> CaptureRigObservabilityEngine.Sector.LEFT
                            else -> CaptureRigObservabilityEngine.Sector.FRONT_LEFT
                        },
                )
            }

        val plan =
            coordinator.plan(
                timestampNanos = 1_000_000_000L,
                cameras = cameras,
            )

        assertEquals(128, plan.acceptedCameraCount)
        assertEquals(16, plan.shardCount)
        assertEquals(
            128,
            plan.shards.sumOf { it.cameraIds.size },
        )
        assertTrue(
            plan.shards.all {
                it.cameraIds.size <= 8
            },
        )

        assertEquals(32, plan.selectedCameraCount)
        assertTrue(
            plan.shards.count {
                it.selectedCameraIds.isNotEmpty()
            } >= 16,
        )
    }
}
