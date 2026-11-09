package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class TriangulationEngineQualityTest {
    @Test
    fun analyticJacobianRemainsStableWithBrownConradyDistortion() {
        val intrinsics = CameraIntrinsics(
            fx = 1100.0,
            fy = 1080.0,
            cx = 640.0,
            cy = 360.0,
            k1 = -0.08,
            k2 = 0.012,
            k3 = -0.001,
            p1 = 0.0012,
            p2 = -0.0008,
        )
        val point = Vec3(0.42, -0.16, 3.2)

        fun distortedPixel(cameraX: Double): Pair<Double, Double> {
            val x = (point.x - cameraX) / point.z
            val y = point.y / point.z
            val r2 = x * x + y * y
            val radial =
                1.0 +
                    intrinsics.k1 * r2 +
                    intrinsics.k2 * r2 * r2 +
                    intrinsics.k3 * r2 * r2 * r2
            val dx =
                2.0 * intrinsics.p1 * x * y +
                    intrinsics.p2 * (r2 + 2.0 * x * x)
            val dy =
                intrinsics.p1 * (r2 + 2.0 * y * y) +
                    2.0 * intrinsics.p2 * x * y
            return (
                intrinsics.fx * (x * radial + dx) + intrinsics.cx
            ) to (
                intrinsics.fy * (y * radial + dy) + intrinsics.cy
            )
        }

        val left = distortedPixel(0.0)
        val right = distortedPixel(0.9)
        val result =
            TriangulationEngine(minParallaxDegrees = 0.1).robust(
                listOf(
                    TriangulationEngine.Observation(
                        cameraPose = Pose(Vec3(0.0, 0.0, 0.0), Quat.IDENTITY),
                        intrinsics = intrinsics,
                        cameraId = "left",
                        u = left.first,
                        v = left.second,
                    ),
                    TriangulationEngine.Observation(
                        cameraPose = Pose(Vec3(0.9, 0.0, 0.0), Quat.IDENTITY),
                        intrinsics = intrinsics,
                        cameraId = "right",
                        u = right.first,
                        v = right.second,
                    ),
                ),
            )

        assertTrue(result != null)
        assertTrue(result!!.point.isFinite())
        assertTrue(result.rmsReprojectionPx.isFinite())
        assertTrue(result.uncertaintyMeters.isFinite())
    }

    @Test
    fun reportsFiniteDepthUncertaintyWhenGeometryIsHealthy() {
        val intrinsics = CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 640.0,
            cy = 360.0,
        )
        val point = Vec3(0.2, 0.1, 3.5)

        fun pixel(cameraX: Double): Pair<Double, Double> =
            (
                intrinsics.fx * (point.x - cameraX) / point.z + intrinsics.cx
            ) to (
                intrinsics.fy * point.y / point.z + intrinsics.cy
            )

        val p0 = pixel(0.0)
        val p1 = pixel(0.8)

        val result = TriangulationEngine(minParallaxDegrees = 0.1).robust(
            listOf(
                TriangulationEngine.Observation(
                    cameraPose = Pose(Vec3(0.0, 0.0, 0.0), Quat.IDENTITY),
                    intrinsics = intrinsics,
                    cameraId = "left",
                    u = p0.first,
                    v = p0.second,
                ),
                TriangulationEngine.Observation(
                    cameraPose = Pose(Vec3(0.8, 0.0, 0.0), Quat.IDENTITY),
                    intrinsics = intrinsics,
                    cameraId = "right",
                    u = p1.first,
                    v = p1.second,
                ),
            ),
        )

        assertTrue(result != null)
        assertTrue(result!!.uncertaintyMeters.isFinite())
        assertTrue(result.conditionScore > 0.01)
        assertTrue(result.cheiralityRatio >= 1.0)
        assertTrue(result.effectiveBaselineMeters >= 0.8)
        assertTrue(result.firstOrderDepthSensitivityMetersPerPixel.isFinite())
        assertTrue(result.firstOrderDepthErrorMillimeters.isFinite())
        assertTrue(result.subMillimeterFeasibilityScore in 0.0..1.0)
    }
}
