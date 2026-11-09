package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TriangulationUncertaintyEngineAnalyticJacobianTest {

    @Test
    fun healthyBrownConradyGeometryUsesAnalyticJacobian() {
        val intrinsics =
            CameraIntrinsics(
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

        val point =
            Vec3(
                0.42,
                -0.16,
                3.2,
            )

        fun project(
            cameraX: Double,
            cameraY: Double,
        ): Pair<Double, Double> {
            val cameraPoint =
                Vec3(
                    point.x - cameraX,
                    point.y - cameraY,
                    point.z,
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
            return (
                intrinsics.fx *
                    (x * radial + dx) +
                    intrinsics.cx
            ) to (
                intrinsics.fy *
                    (y * radial + dy) +
                    intrinsics.cy
            )
        }

        val leftPixel = project(0.0, 0.0)
        val rightPixel = project(0.9, 0.0)

        val result =
            TriangulationUncertaintyEngine().estimate(
                TriangulationUncertaintyEngine.Input(
                    point = point,
                    observations =
                        listOf(
                            observation(
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        Vec3(0.0, 0.0, 0.0),
                                        Quat.IDENTITY,
                                    ),
                                u = leftPixel.first,
                                v = leftPixel.second,
                            ),
                            observation(
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        Vec3(0.9, 0.0, 0.0),
                                        Quat.IDENTITY,
                                    ),
                                u = rightPixel.first,
                                v = rightPixel.second,
                            ),
                        ),
                    robustResidualScalePx = 1.0,
                ),
            )

        assertTrue(result.valid)
        assertTrue(
            result.covariance
                .flatMap { row -> row.asList() }
                .all { it.isFinite() },
        )
        assertEquals(
            0,
            result.numericJacobianFallbackCount,
        )
        assertTrue(
            result.warnings.none {
                it ==
                    "analytic-jacobian-numerical-fallback-used"
            },
        )
    }

    @Test
    fun analyticPathPreservesAnisotropicCovarianceOutput() {
        val intrinsics =
            CameraIntrinsics(
                fx = 1500.0,
                fy = 1480.0,
                cx = 960.0,
                cy = 540.0,
                k1 = -0.11,
                k2 = 0.021,
                k3 = -0.002,
                p1 = 0.0008,
                p2 = -0.0005,
            )

        val result =
            TriangulationUncertaintyEngine().estimate(
                TriangulationUncertaintyEngine.Input(
                    point =
                        Vec3(
                            0.25,
                            0.08,
                            4.0,
                        ),
                    observations =
                        listOf(
                            observation(
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        Vec3(0.0, 0.0, 0.0),
                                        Quat.IDENTITY,
                                    ),
                            ),
                            observation(
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        Vec3(1.2, 0.0, 0.0),
                                        Quat.IDENTITY,
                                    ),
                            ),
                            observation(
                                intrinsics = intrinsics,
                                pose =
                                    Pose(
                                        Vec3(0.0, 0.8, 0.0),
                                        Quat.IDENTITY,
                                    ),
                            ),
                        ),
                    robustResidualScalePx = 0.75,
                ),
            )

        assertTrue(result.sigmaMajorMeters.isFinite())
        assertTrue(result.sigmaMiddleMeters.isFinite())
        assertTrue(result.sigmaMinorMeters.isFinite())
        assertTrue(result.sigmaDepthMeters.isFinite())
        assertTrue(result.anisotropyRatio >= 1.0)
        assertTrue(result.geometricConditionScore in 0.0..1.0)
        assertEquals(
            0,
            result.numericJacobianFallbackCount,
        )
    }

    private fun observation(
        intrinsics: CameraIntrinsics,
        pose: Pose,
        u: Double = 500.0,
        v: Double = 400.0,
    ) =
        TriangulationUncertaintyEngine.Observation(
            cameraPose = pose,
            intrinsics = intrinsics,
            u = u,
            v = v,
            weight = 1.0,
            confidence = 0.98,
            temporalAlignmentConfidence = 0.97,
        )
}
