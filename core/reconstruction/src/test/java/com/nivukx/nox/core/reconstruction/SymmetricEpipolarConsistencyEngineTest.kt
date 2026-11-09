package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertTrue
import org.junit.Test

class SymmetricEpipolarConsistencyEngineTest {

    private val intrinsics =
        CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 500.0,
            cy = 500.0,
        )

    private val engine =
        SymmetricEpipolarConsistencyEngine()

    private fun pixel(
        point: Vec3,
        cameraX: Double,
    ): Pair<Double, Double> =
        (
            intrinsics.fx *
                (point.x - cameraX) /
                point.z +
                intrinsics.cx
            ) to (
            intrinsics.fy *
                point.y /
                point.z +
                intrinsics.cy
            )

    @Test
    fun consistentRaysProduceHighSymmetricEpipolarScore() {
        val point = Vec3(0.2, 0.1, 4.0)
        val left = pixel(point, 0.0)
        val right = pixel(point, 1.0)

        val result =
            engine.evaluate(
                left =
                    SymmetricEpipolarConsistencyEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = left.first,
                        v = left.second,
                    ),
                right =
                    SymmetricEpipolarConsistencyEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = right.first,
                        v = right.second,
                    ),
            )

        assertTrue(result.accepted)
        assertTrue(result.score > 0.90)
        assertTrue(
            result.symmetricAngularResidualRadians <
                Math.toRadians(0.10),
        )
        assertTrue(result.confidence > 0.70)
    }

    @Test
    fun contradictoryCorrespondenceIsPenalized() {
        val point = Vec3(0.2, 0.1, 4.0)
        val left = pixel(point, 0.0)
        val right = pixel(point, 1.0)

        val result =
            engine.evaluate(
                left =
                    SymmetricEpipolarConsistencyEngine.Observation(
                        cameraPose = Pose(
                            Vec3(0.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = left.first,
                        v = left.second,
                    ),
                right =
                    SymmetricEpipolarConsistencyEngine.Observation(
                        cameraPose = Pose(
                            Vec3(1.0, 0.0, 0.0),
                            Quat.IDENTITY,
                        ),
                        intrinsics = intrinsics,
                        u = right.first + 180.0,
                        v = right.second + 130.0,
                    ),
            )

        assertTrue(result.score < 0.80)
        assertTrue(
            result.warnings.isNotEmpty(),
        )
    }

    @Test
    fun degenerateBaselineFailsClosed() {
        val left =
            SymmetricEpipolarConsistencyEngine.Observation(
                cameraPose = Pose(
                    Vec3(0.0, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
                intrinsics = intrinsics,
                u = 500.0,
                v = 500.0,
            )
        val right =
            left.copy(
                cameraPose = Pose(
                    Vec3(0.001, 0.0, 0.0),
                    Quat.IDENTITY,
                ),
            )

        val result =
            engine.evaluate(
                left = left,
                right = right,
            )

        assertTrue(!result.accepted)
        assertTrue(result.score == 0.0)
        assertTrue(
            result.warnings.contains(
                "EPIPOLAR_BASELINE_DEGENERATE",
            ),
        )
    }
}
