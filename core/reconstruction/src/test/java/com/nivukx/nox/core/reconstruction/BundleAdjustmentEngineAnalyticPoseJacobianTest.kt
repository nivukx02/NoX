package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BundleAdjustmentEngineAnalyticPoseJacobianTest {

    @Test
    fun distortedCalibrationUsesAnalyticPoseDerivative() {
        val intrinsics =
            CameraIntrinsics(
                fx = 1450.0,
                fy = 1430.0,
                cx = 960.0,
                cy = 540.0,
                k1 = -0.08,
                k2 = 0.012,
                k3 = -0.001,
                p1 = 0.001,
                p2 = -0.0007,
            )

        val truthPoints =
            listOf(
                Vec3(0.25, -0.14, 4.2),
                Vec3(-0.30, 0.20, 4.0),
                Vec3(0.42, 0.12, 4.6),
                Vec3(-0.15, -0.28, 4.4),
            )

        val reference =
            BundleAdjustmentEngine.Camera(
                id = "reference",
                intrinsics = intrinsics,
                pose =
                    Pose(
                        Vec3(0.0, 0.0, 0.0),
                        Quat.IDENTITY,
                    ),
                fixed = true,
            )

        val trueMovingPose =
            Pose(
                Vec3(1.1, 0.02, 0.03),
                Quat.IDENTITY,
            )

        val moving =
            BundleAdjustmentEngine.Camera(
                id = "moving",
                intrinsics = intrinsics,
                pose =
                    Pose(
                        Vec3(1.0, 0.02, 0.03),
                        Quat.IDENTITY,
                    ),
            )

        fun project(
            point: Vec3,
            pose: Pose,
        ): Vec2 {
            val cameraPoint =
                pose.rotation
                    .conjugate()
                    .normalized()
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
            val dx =
                2.0 * intrinsics.p1 * x * y +
                    intrinsics.p2 *
                    (r2 + 2.0 * x * x)
            val dy =
                intrinsics.p1 *
                    (r2 + 2.0 * y * y) +
                    2.0 * intrinsics.p2 * x * y
            return Vec2(
                intrinsics.fx *
                    (x * radial + dx) +
                    intrinsics.cx,
                intrinsics.fy *
                    (y * radial + dy) +
                    intrinsics.cy,
            )
        }

        val observations =
            truthPoints.flatMapIndexed { index, point ->
                listOf(
                    BundleAdjustmentEngine.Observation(
                        cameraId = "reference",
                        pointId = index + 1,
                        pixel = project(point, reference.pose),
                    ),
                    BundleAdjustmentEngine.Observation(
                        cameraId = "moving",
                        pointId = index + 1,
                        pixel = project(
                            point,
                            trueMovingPose,
                        ),
                    ),
                )
            }

        val result =
            BundleAdjustmentEngine(
                maxIterations = 8,
            ).optimize(
                inputCameras =
                    listOf(
                        reference,
                        moving,
                    ),
                inputPoints =
                    truthPoints.mapIndexed { index, point ->
                        BundleAdjustmentEngine.Point(
                            id = index + 1,
                            position =
                                point +
                                    Vec3(
                                        0.04,
                                        -0.03,
                                        0.08,
                                    ),
                        )
                    },
                observations = observations,
            )

        assertTrue(result.converged)
        assertTrue(result.finalRmsPx <= result.initialRmsPx)
        assertEquals(
            0,
            result.numericalJacobianFallbackCount,
        )
    }
}
