package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EyeSide
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max

/**
 * Production binocular gaze geometry validator.
 *
 * Eye3DReconstructionEngine provides per-eye world-space centers and directions. This stage checks
 * whether the pair is geometrically self-consistent without assuming that a particular gaze target
 * exists. It evaluates interpupillary scale, ray convergence, ray miss distance and directional
 * agreement, then returns diagnostics that can be fused into capture integrity.
 */
class HighEndBinocularGazeGeometryEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumConfidence: Double = 0.20,
        val minimumInterpupillaryDistanceMeters: Double = 0.025,
        val preferredInterpupillaryDistanceMeters: Double = 0.065,
        val maximumInterpupillaryDistanceMeters: Double = 0.110,
        val softRayMissDistanceMeters: Double = 0.045,
        val hardRayMissDistanceMeters: Double = 0.16,
        val softAngularDisagreementDegrees: Double = 6.0,
        val hardAngularDisagreementDegrees: Double = 18.0,
        val preferredConvergenceMeters: Double = 1.2,
        val minimumConvergenceMeters: Double = 0.12,
        val maximumConvergenceMeters: Double = 12.0,
        val strictQualificationMaxGazeUncertaintyMeters: Double = 0.0010,
    ) {
        init {
            require(minimumConfidence in 0.0..1.0)
            require(minimumInterpupillaryDistanceMeters > 0.0)
            require(preferredInterpupillaryDistanceMeters >= minimumInterpupillaryDistanceMeters)
            require(maximumInterpupillaryDistanceMeters > preferredInterpupillaryDistanceMeters)
            require(softRayMissDistanceMeters > 0.0)
            require(hardRayMissDistanceMeters > softRayMissDistanceMeters)
            require(softAngularDisagreementDegrees > 0.0)
            require(hardAngularDisagreementDegrees > softAngularDisagreementDegrees)
            require(minimumConvergenceMeters > 0.0)
            require(maximumConvergenceMeters > minimumConvergenceMeters)
            require(
                strictQualificationMaxGazeUncertaintyMeters in
                    0.0001..0.0100
            )
        }
    }

    data class Result(
        val confidence: Double,
        val interpupillaryDistanceMeters: Double,
        val angularDisagreementDegrees: Double,
        val rayMissDistanceMeters: Double,
        val rayGeometryCondition: Double = 0.0,
        val epipolarEvidenceScore: Double = 1.0,
        val convergenceDistanceMeters: Double?,
        val convergenceDefined: Boolean,
        val gazePointUncertaintyMeters: Double = Double.POSITIVE_INFINITY,
        val precisionQualified: Boolean = false,
        val strictSubMillimeterQualified: Boolean = false,
        val warnings: List<String>,
    ) {
        val accepted: Boolean
            get() =
                confidence >= 0.45 &&
                    rayMissDistanceMeters <= 0.16

        val converged: Boolean
            get() =
                convergenceDefined &&
                    convergenceDistanceMeters != null &&
                    convergenceDistanceMeters >= 0.0
    }

    fun reset() = Unit

    fun evaluate(
        left: Eye3DReconstructionEngine.Result?,
        right: Eye3DReconstructionEngine.Result?,
    ): Result {
        if (left == null || right == null) {
            return Result(
                confidence = 0.0,
                interpupillaryDistanceMeters = 0.0,
                angularDisagreementDegrees = 180.0,
                rayMissDistanceMeters = config.hardRayMissDistanceMeters,
                rayGeometryCondition = 0.0,
                epipolarEvidenceScore = 0.0,
                convergenceDistanceMeters = null,
                convergenceDefined = false,
                gazePointUncertaintyMeters = Double.POSITIVE_INFINITY,
                precisionQualified = false,
                warnings = listOf("binocular-gaze-pair-incomplete"),
            )
        }

        val leftWeight = left.confidence.coerceIn(0.0, 1.0)
        val rightWeight = right.confidence.coerceIn(0.0, 1.0)

        if (
            leftWeight < config.minimumConfidence ||
            rightWeight < config.minimumConfidence
        ) {
            return Result(
                confidence = minOf(leftWeight, rightWeight) * 0.55,
                interpupillaryDistanceMeters =
                    left.centerWorld.distanceTo(right.centerWorld),
                angularDisagreementDegrees =
                    angularDisagreement(left, right),
                rayMissDistanceMeters = config.hardRayMissDistanceMeters,
                rayGeometryCondition = 0.0,
                epipolarEvidenceScore =
                    minOf(
                        left.symmetricEpipolarConsistencyScore,
                        right.symmetricEpipolarConsistencyScore,
                    ).coerceIn(0.0, 1.0),
                convergenceDistanceMeters = null,
                convergenceDefined = false,
                gazePointUncertaintyMeters =
                    config.hardRayMissDistanceMeters,
                precisionQualified = false,
                warnings = listOf(
                    "binocular-gaze-confidence-limited",
                ),
            )
        }

        val ipd =
            left.centerWorld.distanceTo(right.centerWorld)

        val angle =
            angularDisagreement(left, right)

        val ray = closestRayPoints(
            leftCenter = left.centerWorld,
            leftDirection = left.gazeDirectionWorld,
            rightCenter = right.centerWorld,
            rightDirection = right.gazeDirectionWorld,
        )

        val convergence =
            if (ray.parallel) null else ray.distanceAlongAverageAxis

        val ipdScore =
            when {
                ipd < config.minimumInterpupillaryDistanceMeters ->
                    0.0

                ipd > config.maximumInterpupillaryDistanceMeters ->
                    0.0

                ipd <= config.preferredInterpupillaryDistanceMeters ->
                    (
                        ipd -
                            config.minimumInterpupillaryDistanceMeters
                    ) /
                        (
                            config.preferredInterpupillaryDistanceMeters -
                                config.minimumInterpupillaryDistanceMeters
                            )

                else ->
                    (
                        config.maximumInterpupillaryDistanceMeters -
                            ipd
                    ) /
                        (
                            config.maximumInterpupillaryDistanceMeters -
                                config.preferredInterpupillaryDistanceMeters
                            )
            }.coerceIn(0.0, 1.0)

        val angleScore =
            (
                1.0 -
                    angle / config.hardAngularDisagreementDegrees
            ).coerceIn(0.0, 1.0)

        val rayScore =
            (
                1.0 -
                    ray.missDistance /
                    config.hardRayMissDistanceMeters
            ).coerceIn(0.0, 1.0)

        val epipolarEvidenceScore =
            minOf(
                left.symmetricEpipolarConsistencyScore,
                right.symmetricEpipolarConsistencyScore,
            ).coerceIn(0.0, 1.0)

        val convergenceScore =
            if (convergence == null) {
                0.50
            } else {
                when {
                    convergence < config.minimumConvergenceMeters ->
                        0.35

                    convergence > config.maximumConvergenceMeters ->
                        0.72

                    else -> {
                        val ratio =
                            abs(
                                convergence -
                                    config.preferredConvergenceMeters,
                            ) /
                                config.preferredConvergenceMeters
                        (
                            1.0 -
                                ratio * 0.35
                        ).coerceIn(0.55, 1.0)
                    }
                }
            }

        val confidence = (
            minOf(leftWeight, rightWeight) * 0.30 +
                ipdScore * 0.13 +
                angleScore * 0.17 +
                rayScore * 0.20 +
                ray.conditionScore * 0.10 +
                epipolarEvidenceScore * 0.10 +
                convergenceScore * 0.06 +
                uncertaintyConfidence(
                    left.uncertainty95MajorMeters,
                    right.uncertainty95MajorMeters,
                ) * 0.08
            ).coerceIn(0.0, 1.0)

        val centerUncertainty =
            (
                max(0.0, left.uncertainty95MajorMeters) +
                    max(0.0, right.uncertainty95MajorMeters)
                ) * 0.5

        val gazeUncertainty =
            (
                ray.missDistance * 0.62 +
                    centerUncertainty * 0.80 +
                    (
                        1.0 -
                            ray.conditionScore
                    ) *
                    max(
                        config.softRayMissDistanceMeters,
                        0.005,
                    ) *
                    1.40 +
                    (
                        angle /
                            config.hardAngularDisagreementDegrees
                    ).coerceIn(0.0, 1.0) *
                    0.012
                ).coerceIn(
                    0.0005,
                    2.0,
                )

        val precisionQualified =
            confidence >= 0.88 &&
                ray.conditionScore >= 0.45 &&
                epipolarEvidenceScore >= 0.55 &&
                gazeUncertainty <= 0.006 &&
                angle <=
                    config.softAngularDisagreementDegrees &&
                ray.missDistance <=
                    config.softRayMissDistanceMeters

        val strictSubMillimeterQualified =
            precisionQualified &&
                gazeUncertainty <=
                    config.strictQualificationMaxGazeUncertaintyMeters

        val warnings = buildList {
            if (ipdScore < 0.45) {
                add("binocular-gaze-ipd-out-of-range")
            }
            if (angle > config.softAngularDisagreementDegrees) {
                add("binocular-gaze-angular-disagreement")
            }
            if (angle > config.hardAngularDisagreementDegrees) {
                add("binocular-gaze-angular-hard-failure")
            }
            if (ray.missDistance > config.softRayMissDistanceMeters) {
                add("binocular-gaze-ray-miss-distance-high")
            }
            if (ray.conditionScore < 0.08) {
                add("binocular-gaze-ray-geometry-near-parallel")
            }
            if (ray.missDistance > config.hardRayMissDistanceMeters) {
                add("binocular-gaze-ray-miss-hard-failure")
            }
            if (convergence == null) {
                add("binocular-gaze-convergence-undefined")
            }
            if (epipolarEvidenceScore < 0.45) {
                add("binocular-gaze-epipolar-evidence-degraded")
            }
            if (epipolarEvidenceScore < 0.30) {
                add("binocular-gaze-epipolar-evidence-hard-limited")
            }
        }.distinct()

        return Result(
            confidence = confidence,
            interpupillaryDistanceMeters = ipd,
            angularDisagreementDegrees = angle,
            rayMissDistanceMeters = ray.missDistance,
            rayGeometryCondition = ray.conditionScore,
            epipolarEvidenceScore = epipolarEvidenceScore,
            convergenceDistanceMeters = convergence,
            convergenceDefined = convergence != null,
            gazePointUncertaintyMeters =
                gazeUncertainty,
            precisionQualified =
                precisionQualified,
            strictSubMillimeterQualified =
                strictSubMillimeterQualified,
            warnings = warnings +
                if (
                    gazeUncertainty > 0.010
                ) {
                    listOf("binocular-gaze-point-uncertainty-high")
                } else {
                    emptyList()
                } +
                if (
                    precisionQualified &&
                        !strictSubMillimeterQualified
                ) {
                    listOf(
                        "binocular-gaze-submillimeter-not-qualified",
                    )
                } else {
                    emptyList()
                } +
                if (
                    precisionQualified
                ) {
                    emptyList()
                } else {
                    listOf("binocular-gaze-precision-not-qualified")
                },
        )
    }

    private fun uncertaintyConfidence(
        leftMeters: Double,
        rightMeters: Double,
    ): Double {
        val mean =
            (
                max(0.0, leftMeters) +
                    max(0.0, rightMeters)
                ) * 0.5
        if (!mean.isFinite()) return 0.0
        return (
            1.0 -
                mean / 0.020
            ).coerceIn(0.0, 1.0)
    }

    private fun angularDisagreement(
        left: Eye3DReconstructionEngine.Result,
        right: Eye3DReconstructionEngine.Result,
    ): Double =
        Math.toDegrees(
            acos(
                left.gazeDirectionWorld
                    .dot(right.gazeDirectionWorld)
                    .coerceIn(-1.0, 1.0),
            ),
        )

    private data class RayGeometry(
        val missDistance: Double,
        val distanceAlongAverageAxis: Double?,
        val parallel: Boolean,
        val conditionScore: Double,
    )

    /**
     * Computes the closest points between two rays using a stable line-line least-squares solve.
     * Negative forward parameters are clipped for convergence scoring because a point behind an eye
     * cannot be a plausible gaze target.
     */
    private fun closestRayPoints(
        leftCenter: com.nivukx.nox.core.model.Vec3,
        leftDirection: com.nivukx.nox.core.model.Vec3,
        rightCenter: com.nivukx.nox.core.model.Vec3,
        rightDirection: com.nivukx.nox.core.model.Vec3,
    ): RayGeometry {
        val u = leftDirection.normalized()
        val v = rightDirection.normalized()
        val w0 = leftCenter - rightCenter

        val a = u.dot(u)
        val b = u.dot(v)
        val c = v.dot(v)
        val d = u.dot(w0)
        val e = v.dot(w0)

        val denominator = a * c - b * b

        if (!denominator.isFinite() || abs(denominator) < 1e-7) {
            val midpointDelta = rightCenter - leftCenter
            val axis = (u + v).normalized()
            val projected = midpointDelta.dot(axis)

            return RayGeometry(
                missDistance = midpointDelta.norm(),
                distanceAlongAverageAxis =
                    if (projected.isFinite()) max(0.0, projected) else null,
                parallel = true,
                conditionScore = 0.0,
            )
        }

        var s = (b * e - c * d) / denominator
        var t = (a * e - b * d) / denominator

        s = max(0.0, s)
        t = max(0.0, t)

        val leftPoint = leftCenter + u * s
        val rightPoint = rightCenter + v * t
        val separation = leftPoint.distanceTo(rightPoint)

        val midpoint =
            (leftPoint + rightPoint) * 0.5
        val axis =
            (u + v).normalized()
        val averageOrigin =
            (leftCenter + rightCenter) * 0.5

        val distanceAlongAxis =
            (midpoint - averageOrigin).dot(axis)

        val conditionScore =
            denominator.coerceAtLeast(0.0)
                .coerceIn(0.0, 1.0)

        return RayGeometry(
            missDistance = separation,
            distanceAlongAverageAxis =
                max(0.0, distanceAlongAxis),
            parallel = false,
            conditionScore = conditionScore,
        )
    }
}
