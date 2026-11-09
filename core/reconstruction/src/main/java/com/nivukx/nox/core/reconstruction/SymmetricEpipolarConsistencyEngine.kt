package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Calibrated symmetric epipolar consistency evaluator.
 *
 * The metric is computed in normalized-ray space instead of assuming a pinhole-only pixel model:
 * Brown-Conrady distortion is iteratively removed before ray construction. For two calibrated
 * observations, each viewing ray must lie in the epipolar plane defined by both camera centers.
 *
 * The score is deliberately complementary to reprojection error. Reprojection asks whether a
 * hypothesized 3D point explains both images; epipolar consistency asks whether the two observations
 * are mutually compatible with a calibrated two-view geometry before a particular depth is trusted.
 *
 * This is an evidence metric, not an accuracy guarantee. Degenerate baselines, malformed intrinsics,
 * near-parallel rays and non-finite values fail closed.
 */
class SymmetricEpipolarConsistencyEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumBaselineMeters: Double = 0.005,
        val softAngularResidualRadians: Double = Math.toRadians(0.35),
        val hardAngularResidualRadians: Double = Math.toRadians(3.0),
        val minimumRayNorm: Double = 1e-9,
        val maximumUndistortionIterations: Int = 12,
    ) {
        init {
            require(minimumBaselineMeters > 0.0)
            require(
                softAngularResidualRadians > 0.0 &&
                    softAngularResidualRadians <
                    hardAngularResidualRadians,
            )
            require(hardAngularResidualRadians < Math.PI / 2.0)
            require(minimumRayNorm > 0.0)
            require(maximumUndistortionIterations in 2..32)
        }
    }

    data class Observation(
        val cameraPose: Pose,
        val intrinsics: CameraIntrinsics,
        val u: Double,
        val v: Double,
        val confidence: Double = 1.0,
    ) {
        init {
            require(u.isFinite() && v.isFinite())
            require(confidence in 0.0..1.0)
            require(intrinsics.fx.isFinite() && intrinsics.fx > 1e-6)
            require(intrinsics.fy.isFinite() && intrinsics.fy > 1e-6)
        }
    }

    data class Result(
        val accepted: Boolean,
        val confidence: Double,
        val symmetricAngularResidualRadians: Double,
        val leftRayPlaneResidualRadians: Double,
        val rightRayPlaneResidualRadians: Double,
        val baselineMeters: Double,
        val score: Double,
        val warnings: List<String>,
    )

    fun evaluate(
        left: Observation,
        right: Observation,
    ): Result {
        val leftRay = ray(left)
        val rightRay = ray(right)
        val baselineVector =
            rightRay.origin - leftRay.origin
        val baseline =
            baselineVector.norm()

        if (
            !baseline.isFinite() ||
            baseline < config.minimumBaselineMeters
        ) {
            return rejected(
                confidence = minOf(
                    left.confidence,
                    right.confidence,
                ),
                baselineMeters = baseline.coerceAtLeast(0.0),
                warning = "EPIPOLAR_BASELINE_DEGENERATE",
            )
        }

        val baselineDirection =
            baselineVector / baseline

        val leftNormal =
            leftRay.direction
                .cross(baselineDirection)
        val rightNormal =
            rightRay.direction
                .cross(baselineDirection)

        val leftNormalNorm = leftNormal.norm()
        val rightNormalNorm = rightNormal.norm()

        if (
            !leftNormalNorm.isFinite() ||
            !rightNormalNorm.isFinite() ||
            leftNormalNorm < config.minimumRayNorm ||
            rightNormalNorm < config.minimumRayNorm
        ) {
            return rejected(
                confidence = minOf(
                    left.confidence,
                    right.confidence,
                ) * 0.50,
                baselineMeters = baseline,
                warning = "EPIPOLAR_RAY_PLANE_DEGENERATE",
            )
        }

        val leftResidual =
            abs(
                leftRay.direction
                    .dot(rightNormal) /
                    rightNormalNorm,
            ).coerceIn(0.0, 1.0)

        val rightResidual =
            abs(
                rightRay.direction
                    .dot(leftNormal) /
                    leftNormalNorm,
            ).coerceIn(0.0, 1.0)

        val symmetricResidual =
            (
                leftResidual +
                    rightResidual
                ) * 0.5

        val angularResidual =
            max(
                leftResidual,
                rightResidual,
            )
                .let { asinSafe(it) }

        val score =
            when {
                !angularResidual.isFinite() ->
                    0.0

                angularResidual <=
                    config.softAngularResidualRadians ->
                    1.0

                angularResidual >=
                    config.hardAngularResidualRadians ->
                    0.0

                else ->
                    exp(
                        -(
                            angularResidual -
                                config.softAngularResidualRadians
                            ) /
                            (
                                config.hardAngularResidualRadians -
                                    config.softAngularResidualRadians
                                ).coerceAtLeast(1e-9),
                    ).coerceIn(0.0, 1.0)
            }

        val directionalAgreement =
            1.0 -
                (
                    acos(
                        leftRay.direction
                            .dot(rightRay.direction)
                            .coerceIn(-1.0, 1.0),
                    ) /
                        (Math.PI / 2.0)
                    ).coerceIn(0.0, 1.0)

        val confidence =
            (
                minOf(
                    left.confidence,
                    right.confidence,
                ) * 0.70 +
                    score * 0.20 +
                    directionalAgreement * 0.10
                ).coerceIn(0.0, 1.0)

        val accepted =
            confidence >= 0.35 &&
                angularResidual <=
                    config.hardAngularResidualRadians

        val warnings =
            buildList {
                if (
                    angularResidual >
                        config.softAngularResidualRadians
                ) {
                    add("EPIPOLAR_RESIDUAL_ELEVATED")
                }
                if (
                    angularResidual >
                        config.hardAngularResidualRadians
                ) {
                    add("EPIPOLAR_RESIDUAL_HARD_FAILURE")
                }
                if (
                    baseline <
                        config.minimumBaselineMeters * 2.0
                ) {
                    add("EPIPOLAR_BASELINE_LOW")
                }
                if (directionalAgreement < 0.20) {
                    add("EPIPOLAR_RAY_DIRECTION_DISAGREEMENT")
                }
            }.distinct()

        return Result(
            accepted = accepted,
            confidence = confidence,
            symmetricAngularResidualRadians = angularResidual,
            leftRayPlaneResidualRadians = asinSafe(
                leftResidual,
            ),
            rightRayPlaneResidualRadians = asinSafe(
                rightResidual,
            ),
            baselineMeters = baseline,
            score = score,
            warnings = warnings,
        )
    }

    private data class Ray(
        val origin: Vec3,
        val direction: Vec3,
    )

    private fun ray(
        observation: Observation,
    ): Ray {
        val normalized =
            undistort(
                observation.u,
                observation.v,
                observation.intrinsics,
            )
        val cameraRay =
            Vec3(
                normalized.x,
                normalized.y,
                1.0,
            ).normalized()
        return Ray(
            origin = observation.cameraPose.position,
            direction =
                observation.cameraPose.rotation
                    .rotate(cameraRay)
                    .normalized(),
        )
    }

    private fun undistort(
        u: Double,
        v: Double,
        intrinsics: CameraIntrinsics,
    ): Vec2 {
        val xd =
            (u - intrinsics.cx) /
                intrinsics.fx
        val yd =
            (v - intrinsics.cy) /
                intrinsics.fy

        var x = xd
        var y = yd

        repeat(config.maximumUndistortionIterations) {
            val r2 =
                x * x +
                    y * y
            val r4 = r2 * r2
            val r6 = r4 * r2
            val radial =
                1.0 +
                    intrinsics.k1 * r2 +
                    intrinsics.k2 * r4 +
                    intrinsics.k3 * r6

            if (
                !radial.isFinite() ||
                abs(radial) < 1e-9
            ) {
                return@repeat
            }

            val tx =
                2.0 * intrinsics.p1 * x * y +
                    intrinsics.p2 *
                    (r2 + 2.0 * x * x)
            val ty =
                intrinsics.p1 *
                    (r2 + 2.0 * y * y) +
                    2.0 * intrinsics.p2 * x * y

            x =
                (xd - tx) /
                    radial
            y =
                (yd - ty) /
                    radial
        }

        return Vec2(x, y)
    }

    private fun asinSafe(
        value: Double,
    ): Double =
        kotlin.math.asin(
            value.coerceIn(0.0, 1.0),
        )

    private fun rejected(
        confidence: Double,
        baselineMeters: Double,
        warning: String,
    ): Result =
        Result(
            accepted = false,
            confidence = confidence.coerceIn(0.0, 1.0),
            symmetricAngularResidualRadians =
                Math.PI / 2.0,
            leftRayPlaneResidualRadians =
                Math.PI / 2.0,
            rightRayPlaneResidualRadians =
                Math.PI / 2.0,
            baselineMeters = baselineMeters,
            score = 0.0,
            warnings = listOf(warning),
        )
}
