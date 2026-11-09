package com.nivukx.nox.core.model

import kotlin.test.assertEquals
import org.junit.Test

class ReconstructionContractCompatibilityTest {
    @Test
    fun camera_observation_exposes_legacy_pose_alias_without_copying_state() {
        val pose = Pose(
            position = Vec3(1.0, 2.0, 3.0),
            rotation = Quat(0.96, 0.10, 0.20, 0.21).normalized(),
        )
        val observation = CameraObservation(
            deviceId = DeviceId("camera-a"),
            point = Vec2(320.0, 240.0),
            confidence = 0.94,
            intrinsics = CameraIntrinsics(
                fx = 1200.0,
                fy = 1200.0,
                cx = 320.0,
                cy = 240.0,
                imageWidth = 640,
                imageHeight = 480,
            ),
            pose = pose,
        )

        assertEquals(pose, observation.cameraPose)
    }

    @Test
    fun reconstruction_joint_keeps_uncertainty_alias_and_acceleration_default() {
        val joint = ReconstructionJoint(
            landmarkId = 17,
            position = Vec3(0.1, 0.2, 0.3),
            velocity = Vec3(0.01, 0.02, 0.03),
            confidence = 0.88,
            reprojectionErrorPx = 1.2,
            observationCount = 3,
            positionUncertaintyMeters = 0.012,
        )

        assertEquals(0.012, joint.uncertaintyMeters, 0.0)
        assertEquals(
            Vec3(0.0, 0.0, 0.0),
            joint.acceleration,
        )
    }

    @Test
    fun reconstruction_joint_preserves_explicit_acceleration() {
        val acceleration = Vec3(0.7, -0.2, 0.4)
        val joint = ReconstructionJoint(
            landmarkId = 3,
            position = Vec3(0.0, 0.0, 1.0),
            velocity = Vec3(0.1, 0.0, 0.0),
            confidence = 0.91,
            reprojectionErrorPx = 0.8,
            observationCount = 4,
            positionUncertaintyMeters = 0.006,
            acceleration = acceleration,
        )

        assertEquals(acceleration, joint.acceleration)
        assertEquals(0.006, joint.uncertaintyMeters, 0.0)
    }
}
