package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class BundleAdjustmentRefinerTest {

    private val intrinsics = CameraIntrinsics(
        fx = 900.0,
        fy = 900.0,
        cx = 640.0,
        cy = 360.0,
    )

    @Test
    fun refinerReducesConsistentMultiViewResidual() {
        val cameras = listOf(
            Pose(Vec3(0.0, 0.0, 0.0), Quat.IDENTITY),
            Pose(Vec3(1.8, 0.0, 0.0), Quat.IDENTITY),
            Pose(Vec3(-1.4, 0.1, 0.0), Quat.IDENTITY),
        )
        val truth = Vec3(0.15, -0.08, 4.2)

        val observations = cameras.mapIndexed { index, pose ->
            CameraObservation(
                deviceId = DeviceId("cam-$index"),
                point = project(truth, pose),
                confidence = 0.98,
                intrinsics = intrinsics,
                pose = pose,
            )
        }

        val initial = ReconstructionJoint(
            landmarkId = 7,
            position = truth + Vec3(0.16, -0.10, 0.22),
            velocity = Vec3(0.0, 0.0, 0.0),
            confidence = 0.82,
            reprojectionErrorPx = 15.0,
            observationCount = observations.size,
            positionUncertaintyMeters = 0.20,
        )

        val result = BundleAdjustmentRefiner().refine(
            joint = initial,
            observations = observations,
            priorPosition = initial.position,
        )

        assertTrue(result.accepted)
        assertTrue(result.finalErrorPx < result.initialErrorPx)
        assertTrue(
            result.joint.position.distanceTo(truth) <
                initial.position.distanceTo(truth),
        )
    }

    @Test
    fun robustWeightingKeepsSolutionNearConsistentViews() {
        val cameras = listOf(
            Pose(Vec3(0.0, 0.0, 0.0), Quat.IDENTITY),
            Pose(Vec3(2.0, 0.0, 0.0), Quat.IDENTITY),
            Pose(Vec3(-2.0, 0.0, 0.0), Quat.IDENTITY),
        )
        val truth = Vec3(0.1, 0.1, 4.0)

        val observations = cameras.mapIndexed { index, pose ->
            val point = project(truth, pose)
            CameraObservation(
                deviceId = DeviceId("cam-$index"),
                point = if (index == 2) {
                    point + Vec2(110.0, -80.0)
                } else {
                    point
                },
                confidence = 0.97,
                intrinsics = intrinsics,
                pose = pose,
            )
        }

        val initial = ReconstructionJoint(
            landmarkId = 8,
            position = truth + Vec3(0.10, -0.06, 0.20),
            velocity = Vec3(0.0, 0.0, 0.0),
            confidence = 0.80,
            reprojectionErrorPx = 10.0,
            observationCount = observations.size,
        )

        val result = BundleAdjustmentRefiner().refine(
            joint = initial,
            observations = observations,
            priorPosition = initial.position,
        )

        assertTrue(result.accepted)
        assertTrue(result.inliers >= 2)
        assertTrue(result.joint.position.z > 3.0)
    }

    @Test
    fun insufficientViewsAreRejectedWithoutMutatingPosition() {
        val truth = Vec3(0.0, 0.0, 5.0)
        val observation = CameraObservation(
            deviceId = DeviceId("single"),
            point = project(truth, Pose()),
            confidence = 0.99,
            intrinsics = intrinsics,
            pose = Pose(),
        )
        val joint = ReconstructionJoint(
            landmarkId = 9,
            position = Vec3(0.2, 0.0, 4.8),
            velocity = Vec3(0.0, 0.0, 0.0),
            confidence = 0.70,
            reprojectionErrorPx = 4.0,
            observationCount = 1,
        )

        val result = BundleAdjustmentRefiner().refine(
            joint = joint,
            observations = listOf(observation),
        )

        assertTrue(!result.accepted)
        assertTrue(result.reason == "insufficient-independent-views")
        assertTrue(result.joint.position == joint.position)
    }

    private fun project(
        point: Vec3,
        pose: Pose,
    ): Vec2 {
        val cameraPoint = pose.rotation
            .normalized()
            .conjugate()
            .rotate(point - pose.position)
        return Vec2(
            intrinsics.fx * cameraPoint.x / cameraPoint.z + intrinsics.cx,
            intrinsics.fy * cameraPoint.y / cameraPoint.z + intrinsics.cy,
        )
    }
    @Test
    fun distortedProjectionRemainsRefinableWithAnalyticJacobian() {
        val distortedIntrinsics =
            intrinsics.copy(
                k1 = -0.12,
                k2 = 0.018,
                p1 = 0.0012,
                p2 = -0.0008,
                k3 = 0.002,
            )
        val poses =
            listOf(
                Pose(
                    Vec3(0.0, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                Pose(
                    Vec3(1.6, 0.05, 0.0),
                    Quat.IDENTITY,
                ),
                Pose(
                    Vec3(-1.2, -0.04, 0.0),
                    Quat.IDENTITY,
                ),
            )
        val truth =
            Vec3(
                0.35,
                -0.22,
                3.7,
            )

        val observations =
            poses.mapIndexed { index, pose ->
                CameraObservation(
                    deviceId = DeviceId("distorted-$index"),
                    point =
                        projectWith(
                            truth,
                            pose,
                            distortedIntrinsics,
                        ),
                    confidence = 0.99,
                    intrinsics = distortedIntrinsics,
                    pose = pose,
                )
            }

        val initial =
            ReconstructionJoint(
                landmarkId = 77,
                position =
                    truth +
                        Vec3(
                            0.10,
                            -0.07,
                            0.18,
                        ),
                velocity =
                    Vec3(
                        0.0,
                        0.0,
                        0.0,
                    ),
                confidence = 0.90,
                reprojectionErrorPx = 12.0,
                observationCount = 3,
                positionUncertaintyMeters = 0.10,
            )

        val result =
            BundleAdjustmentRefiner().refine(
                joint = initial,
                observations = observations,
                priorPosition = initial.position,
            )

        assertTrue(result.accepted)
        assertTrue(
            result.finalErrorPx <
                result.initialErrorPx,
        )
        assertTrue(
            result.joint.position.distanceTo(truth) <
                initial.position.distanceTo(truth),
        )
    }

    private fun projectWith(
        point: Vec3,
        pose: Pose,
        intrinsics: CameraIntrinsics,
    ): Vec2 {
        val cameraPoint =
            pose.rotation
                .normalized()
                .conjugate()
                .rotate(
                    point -
                        pose.position,
                )
        val x =
            cameraPoint.x /
                cameraPoint.z
        val y =
            cameraPoint.y /
                cameraPoint.z
        val r2 =
            x * x +
                y * y
        val radial =
            1.0 +
                intrinsics.k1 * r2 +
                intrinsics.k2 * r2 * r2 +
                intrinsics.k3 * r2 * r2 * r2
        val xd =
            x * radial +
                2.0 * intrinsics.p1 * x * y +
                intrinsics.p2 *
                    (
                        r2 +
                            2.0 * x * x
                    )
        val yd =
            y * radial +
                intrinsics.p1 *
                    (
                        r2 +
                            2.0 * y * y
                    ) +
                2.0 * intrinsics.p2 * x * y
        return Vec2(
            intrinsics.fx * xd +
                intrinsics.cx,
            intrinsics.fy * yd +
                intrinsics.cy,
        )
    }

}
