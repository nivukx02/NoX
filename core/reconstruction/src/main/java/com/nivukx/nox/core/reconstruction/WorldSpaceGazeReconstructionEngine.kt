package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Robust world-space ocular direction reconstruction.
 *
 * The solver treats eyelid state, observation uncertainty, occlusion and cross-eye agreement as
 * evidence weights. It does not invent a gaze target and it never converts a confidence target into
 * a physical precision claim.
 */
class WorldSpaceGazeReconstructionEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumConfidence: Double = 0.15,
        val cauchyScaleDegrees: Double = 8.0,
        val maximumRays: Int = 32,
        val minimumInlierRatio: Double = 0.45,
        val maximumRejectAngleDegrees: Double = 38.0,
        val binocularAgreementScaleDegrees: Double = 10.0,
        val uncertaintyFloorDegrees: Double = 0.25,
        val robustIterations: Int = 6,
    ) {
        init {
            require(minimumConfidence in 0.0..1.0)
            require(cauchyScaleDegrees > 0.0)
            require(maximumRays > 0)
            require(minimumInlierRatio in 0.0..1.0)
            require(maximumRejectAngleDegrees > 0.0)
            require(binocularAgreementScaleDegrees > 0.0)
            require(uncertaintyFloorDegrees > 0.0)
            require(robustIterations in 1..16)
        }
    }

    data class EyeRay(
        val eye: Eye,
        val originMeters: Vec3,
        val directionHeadSpace: Vec3,
        val eyelidOpenness: Double,
        val confidence: Double,
        val blink: Boolean = false,
        val measurementUncertaintyDegrees: Double = 2.0,
        val occlusionRisk: Double = 0.0,
        val viewWeight: Double = 1.0,
        val freshness: Double = 1.0,
    ) {
        init {
            require(finite(originMeters))
            require(finite(directionHeadSpace))
            require(eyelidOpenness in 0.0..1.0)
            require(confidence in 0.0..1.0)
            require(measurementUncertaintyDegrees.isFinite())
            require(measurementUncertaintyDegrees >= 0.0)
            require(occlusionRisk in 0.0..1.0)
            require(viewWeight.isFinite() && viewWeight >= 0.0)
            require(freshness in 0.0..1.0)
        }
    }

    enum class Eye {
        LEFT,
        RIGHT,
    }

    data class Result(
        val directionWorld: Vec3,
        val angularUncertaintyDegrees: Double,
        val inlierRatio: Double,
        val binocularAgreementDegrees: Double,
        val valid: Boolean,
        val warnings: List<String>,
        val confidence: Double = 0.0,
        val leftRightAgreementScore: Double = 0.0,
        val effectiveSampleCount: Double = 0.0,
        val outlierCount: Int = 0,
        val gazeTargetWorld: Vec3? = null,
        val gazeTargetUncertaintyMeters: Double = Double.POSITIVE_INFINITY,
        val vergenceAngleDegrees: Double = 180.0,
    )

    private data class WeightedRay(
        val source: EyeRay,
        val directionWorld: Vec3,
        val baseWeight: Double,
    )

    fun reconstruct(
        headRotation: Quat,
        rays: List<EyeRay>,
    ): Result {
        val bounded = rays
            .asSequence()
            .filter { !it.blink }
            .filter { it.confidence >= config.minimumConfidence }
            .filter { it.eyelidOpenness > 0.10 }
            .filter { finite(it.directionHeadSpace) }
            .sortedWith(
                compareByDescending<EyeRay> { it.confidence }
                    .thenBy { it.eye.name }
            )
            .take(config.maximumRays)
            .toList()

        if (bounded.isEmpty()) {
            return invalid("no-valid-eye-evidence")
        }

        val worldRays = bounded.mapNotNull { ray ->
            val direction = headRotation
                .rotate(ray.directionHeadSpace)
                .normalized()
            if (!finite(direction) || direction.norm() < 0.99) {
                null
            } else {
                val uncertaintyWeight = 1.0 /
                    (
                        1.0 +
                            ray.measurementUncertaintyDegrees /
                                10.0
                    )
                val baseWeight = (
                    ray.confidence *
                        ray.eyelidOpenness *
                        (1.0 - ray.occlusionRisk) *
                        ray.viewWeight.coerceAtMost(1.0) *
                        ray.freshness *
                        uncertaintyWeight
                    ).coerceAtLeast(1e-6)
                WeightedRay(
                    source = ray,
                    directionWorld = direction,
                    baseWeight = baseWeight,
                )
            }
        }

        if (worldRays.isEmpty()) {
            return invalid("no-finite-world-gaze-evidence")
        }

        var consensus =
            worldRays.maxBy { it.baseWeight }.directionWorld

        repeat(config.robustIterations) {
            var weightedDirection = Vec3(0.0, 0.0, 0.0)
            var totalWeight = 0.0

            worldRays.forEach { ray ->
                val angle = angularDegrees(
                    ray.directionWorld,
                    consensus,
                )
                val robust = robustWeight(angle)
                val crossEye = crossEyeWeight(
                    ray = ray,
                    all = worldRays,
                )
                val weight =
                    ray.baseWeight *
                        robust *
                        crossEye
                if (weight.isFinite() && weight > 1e-9) {
                    weightedDirection +=
                        ray.directionWorld * weight
                    totalWeight += weight
                }
            }

            if (totalWeight > 1e-9) {
                consensus =
                    (
                        weightedDirection /
                            totalWeight
                        ).normalized()
            }
        }

        val angles = worldRays.map {
            angularDegrees(
                it.directionWorld,
                consensus,
            )
        }
        val weightedP95 =
            weightedQuantile(
                values = worldRays.zip(angles),
                quantile = 0.95,
            )
        val inliers = worldRays.indices.count { index ->
            val angle = angles[index]
            robustWeight(angle) > 0.35 &&
                angle <= config.maximumRejectAngleDegrees
        }
        val outliers = (
            worldRays.size - inliers
        ).coerceAtLeast(0)

        val left =
            robustEyeConsensus(
                worldRays.filter { it.source.eye == Eye.LEFT },
            )
        val right =
            robustEyeConsensus(
                worldRays.filter { it.source.eye == Eye.RIGHT },
            )

        val binocularAgreement =
            if (left != null && right != null) {
                angularDegrees(left, right)
            } else {
                weightedP95
            }

        val agreementScore =
            (
                1.0 -
                    binocularAgreement /
                        config.binocularAgreementScaleDegrees
                ).coerceIn(0.0, 1.0)

        val inlierRatio =
            inliers.toDouble() /
                worldRays.size.coerceAtLeast(1).toDouble()

        val uncertainty =
            max(
                config.uncertaintyFloorDegrees,
                (
                    weightedP95 *
                        (
                            1.0 +
                                0.35 * (
                                    1.0 -
                                        agreementScore
                                    )
                            )
                    ).coerceAtLeast(
                        worldRays
                            .map {
                                it.source
                                    .measurementUncertaintyDegrees
                            }
                            .medianFinite()
                            ?: config.uncertaintyFloorDegrees,
                    ),
            )

        val effectiveSampleCount =
            worldRays.sumOf {
                it.baseWeight.coerceAtLeast(0.0)
            }

        val binocularGeometry =
            estimateBinocularTarget(
                left = left,
                right = right,
                leftRays = worldRays.filter {
                    it.source.eye == Eye.LEFT
                },
                rightRays = worldRays.filter {
                    it.source.eye == Eye.RIGHT
                },
                angularUncertaintyDegrees = uncertainty,
            )

        val confidence = (
            0.38 * inlierRatio +
                0.20 * agreementScore +
                0.16 * effectiveSampleConfidence(
                    effectiveSampleCount,
                ) +
                0.14 * uncertaintyConfidence(uncertainty) +
                0.12 * (
                    worldRays
                        .map { it.source.freshness }
                        .average()
                )
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (inlierRatio < config.minimumInlierRatio) {
                add("gaze-outlier-consensus-weak")
            }
            if (
                left != null &&
                    right != null &&
                    binocularAgreement >
                    config.binocularAgreementScaleDegrees
            ) {
                add("gaze-binocular-disagreement")
            }
            if (outliers > 0) {
                add("gaze-outliers-rejected")
            }
            if (effectiveSampleCount < 0.75) {
                add("gaze-low-effective-sample-count")
            }
        }

        return Result(
            directionWorld = consensus,
            angularUncertaintyDegrees = uncertainty,
            inlierRatio = inlierRatio,
            binocularAgreementDegrees = binocularAgreement,
            valid =
                consensus.norm() > 0.99 &&
                    inlierRatio >= config.minimumInlierRatio,
            warnings = warnings.distinct(),
            confidence = confidence,
            leftRightAgreementScore = agreementScore,
            effectiveSampleCount = effectiveSampleCount,
            outlierCount = outliers,
            gazeTargetWorld = binocularGeometry?.targetWorld,
            gazeTargetUncertaintyMeters =
                binocularGeometry?.uncertaintyMeters
                    ?: Double.POSITIVE_INFINITY,
            vergenceAngleDegrees =
                binocularGeometry?.vergenceAngleDegrees
                    ?: 180.0,
        )
    }

    private fun robustEyeConsensus(
        rays: List<WeightedRay>,
    ): Vec3? {
        if (rays.isEmpty()) return null
        var estimate =
            rays.maxBy { it.baseWeight }.directionWorld

        repeat(4) {
            var sum = Vec3(0.0, 0.0, 0.0)
            var total = 0.0
            rays.forEach { ray ->
                val angle =
                    angularDegrees(
                        ray.directionWorld,
                        estimate,
                    )
                val weight =
                    ray.baseWeight *
                        robustWeight(angle)
                if (weight > 1e-9) {
                    sum += ray.directionWorld * weight
                    total += weight
                }
            }
            if (total > 1e-9) {
                estimate = (sum / total).normalized()
            }
        }
        return estimate
    }

    private fun crossEyeWeight(
        ray: WeightedRay,
        all: List<WeightedRay>,
    ): Double {
        val peer = all
            .asSequence()
            .filter {
                it.source.eye != ray.source.eye
            }
            .maxByOrNull { it.baseWeight }
            ?: return 1.0

        val disagreement =
            angularDegrees(
                ray.directionWorld,
                peer.directionWorld,
            )
        return (
            0.50 +
                0.50 * robustWeight(
                    disagreement,
                )
            ).coerceIn(0.5, 1.0)
    }

    private fun robustWeight(
        angleDegrees: Double,
    ): Double {
        if (!angleDegrees.isFinite()) return 0.0
        val ratio =
            angleDegrees /
                max(
                    0.1,
                    config.cauchyScaleDegrees,
                )
        return 1.0 /
            (
                1.0 +
                    ratio * ratio
            )
    }

    private fun weightedQuantile(
        values: List<Pair<WeightedRay, Double>>,
        quantile: Double,
    ): Double {
        if (values.isEmpty()) {
            return 180.0
        }

        val ordered = values
            .sortedBy { it.second }
        val totalWeight =
            ordered.sumOf {
                it.first.baseWeight
            }.coerceAtLeast(1e-9)
        val target =
            totalWeight *
                quantile.coerceIn(
                    0.0,
                    1.0,
                )

        var cumulative = 0.0
        for ((ray, value) in ordered) {
            cumulative += ray.baseWeight
            if (cumulative >= target) {
                return value
            }
        }

        return ordered.last().second
    }

    private data class BinocularGeometry(
        val targetWorld: Vec3,
        val uncertaintyMeters: Double,
        val vergenceAngleDegrees: Double,
    )

    private fun estimateBinocularTarget(
        left: Vec3?,
        right: Vec3?,
        leftRays: List<WeightedRay>,
        rightRays: List<WeightedRay>,
        angularUncertaintyDegrees: Double,
    ): BinocularGeometry? {
        val leftRay = leftRays.maxByOrNull { it.baseWeight }
            ?: return null
        val rightRay = rightRays.maxByOrNull { it.baseWeight }
            ?: return null
        if (left == null || right == null) return null

        val p1 = leftRay.source.originMeters
        val p2 = rightRay.source.originMeters
        val d1 = leftRay.directionWorld.normalized()
        val d2 = rightRay.directionWorld.normalized()

        val w0 = p1 - p2
        val b = d1.dot(d2)
        val d = d1.dot(w0)
        val e = d2.dot(w0)
        val denominator = 1.0 - b * b

        if (!denominator.isFinite() || denominator < 1e-7) {
            return null
        }

        val s = ((b * e - d) / denominator).coerceAtLeast(0.0)
        val t = ((e - b * d) / denominator).coerceAtLeast(0.0)

        if (
            !s.isFinite() ||
            !t.isFinite() ||
            s > 100.0 ||
            t > 100.0
        ) {
            return null
        }

        val closestLeft = p1 + d1 * s
        val closestRight = p2 + d2 * t
        val separation = closestLeft.distanceTo(closestRight)

        if (!separation.isFinite()) return null

        val target = (
            closestLeft + closestRight
        ) * 0.5

        val range = (
            s + t
        ) * 0.5

        val angularTerm = range * tan(
            Math.toRadians(
                angularUncertaintyDegrees.coerceIn(0.0, 89.0),
            ),
        )

        return BinocularGeometry(
            targetWorld = target,
            uncertaintyMeters = max(
                0.001,
                0.5 * separation + angularTerm,
            ),
            vergenceAngleDegrees = angularDegrees(
                d1,
                d2,
            ),
        )
    }

    private fun effectiveSampleConfidence(
        count: Double,
    ): Double =
        (
            1.0 -
                1.0 /
                    (
                        1.0 +
                            count
                    )
            ).coerceIn(0.0, 1.0)

    private fun uncertaintyConfidence(
        degrees: Double,
    ): Double =
        (
            1.0 -
                degrees /
                    45.0
            ).coerceIn(0.0, 1.0)

    private fun angularDegrees(
        a: Vec3,
        b: Vec3,
    ): Double =
        Math.toDegrees(
            acos(
                a.normalized()
                    .dot(
                        b.normalized(),
                    )
                    .coerceIn(
                        -1.0,
                        1.0,
                    ),
            ),
        )

    private fun invalid(
        warning: String,
    ): Result =
        Result(
            directionWorld = Vec3(0.0, 0.0, 1.0),
            angularUncertaintyDegrees = 180.0,
            inlierRatio = 0.0,
            binocularAgreementDegrees = 180.0,
            valid = false,
            warnings = listOf(warning),
        )
}

private fun List<Double>.medianFinite(): Double? {
    val finite = filter { it.isFinite() }.sorted()
    if (finite.isEmpty()) return null
    val middle = finite.size / 2
    return if (finite.size % 2 == 0) {
        (finite[middle - 1] + finite[middle]) * 0.5
    } else {
        finite[middle]
    }
}

private fun finite(value: Vec3): Boolean =
    value.x.isFinite() &&
        value.y.isFinite() &&
        value.z.isFinite()
