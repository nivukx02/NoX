package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Computes per-camera geometric conditioning before dense triangulation.
 *
 * High-confidence observations are not automatically good metric observations. This layer scores
 * ray separation, camera-center diversity and peer agreement, then returns a bounded multiplier that
 * the reconstruction authority can apply to the original observation weight.
 */
class PerObservationGeometricConditioningEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumParallaxDegrees: Double = 1.5,
        val targetParallaxDegrees: Double = 12.0,
        val strongParallaxDegrees: Double = 28.0,
        val contradictionDegrees: Double = 18.0,
        val minimumConditioning: Double = 0.20,
    ) {
        init {
            require(minimumParallaxDegrees > 0.0)
            require(targetParallaxDegrees >= minimumParallaxDegrees)
            require(strongParallaxDegrees >= targetParallaxDegrees)
            require(contradictionDegrees > minimumParallaxDegrees)
            require(minimumConditioning in 0.0..1.0)
        }
    }

    data class ObservationScore(
        val cameraId: String,
        val parallaxScore: Double,
        val baselineScore: Double,
        val peerConsistencyScore: Double,
        val confidenceScore: Double,
        val conditioningScore: Double,
        val weightMultiplier: Double,
        val sourceDiversityScore: Double,
        val weakGeometry: Boolean,
        val contradictoryGeometry: Boolean,
        val diagnostics: List<String>,
    )

    data class Result(
        val scores: List<ObservationScore>,
        val globalConditioningScore: Double,
        val medianParallaxDegrees: Double,
        val minimumParallaxDegrees: Double,
        val cameraDiversityScore: Double,
        val minimumObservationConditioningScore: Double,
        val authoritative: Boolean,
        val warnings: List<String>,
    ) {
        fun scoreFor(cameraId: String): ObservationScore? =
            scores.firstOrNull { it.cameraId == cameraId }
    }

    @Synchronized
    fun evaluate(
        observations: List<TriangulationEngine.Observation>,
    ): Result {
        val unique = observations.distinctBy { it.cameraId }.take(32)
        if (unique.isEmpty()) {
            return Result(
                scores = emptyList(),
                globalConditioningScore = 0.0,
                medianParallaxDegrees = 0.0,
                minimumParallaxDegrees = 0.0,
                cameraDiversityScore = 0.0,
                minimumObservationConditioningScore = 0.0,
                authoritative = false,
                warnings = listOf("NO_GEOMETRIC_OBSERVATIONS"),
            )
        }

        if (unique.size == 1) {
            val observation = unique.single()
            val confidence =
                (
                    observation.sourceQuality.coerceIn(0.0, 1.0) *
                        observation.visibility.coerceIn(0.0, 1.0)
                ).coerceIn(0.05, 1.0)
            return Result(
                scores = listOf(
                    ObservationScore(
                        cameraId = observation.cameraId,
                        parallaxScore = 0.0,
                        baselineScore = 0.0,
                        peerConsistencyScore = 0.0,
                        confidenceScore = confidence,
                        conditioningScore = 0.0,
                        weightMultiplier = 0.35,
                        sourceDiversityScore = 0.0,
                        weakGeometry = true,
                        contradictoryGeometry = false,
                        diagnostics = listOf(
                            "SINGLE_VIEW_DEPTH_UNDERCONSTRAINED",
                        ),
                    ),
                ),
                globalConditioningScore = 0.0,
                medianParallaxDegrees = 0.0,
                minimumParallaxDegrees = 0.0,
                cameraDiversityScore = 0.0,
                minimumObservationConditioningScore = 0.0,
                authoritative = false,
                warnings = listOf("SINGLE_VIEW_DEPTH_UNDERCONSTRAINED"),
            )
        }

        val rays = unique.associateWith(::rayFor)
        val pairs = ArrayList<PairMetric>()
        for (i in 0 until unique.lastIndex) {
            for (j in (i + 1 until unique.size)) {
                val a = unique[i]
                val b = unique[j]
                val rayA = rays[a] ?: continue
                val rayB = rays[b] ?: continue
                val cross = rayA.cross(rayB)
                val angle = Math.toDegrees(
                    atan2(
                        cross.norm(),
                        rayA.dot(rayB).coerceIn(-1.0, 1.0),
                    ),
                ).coerceIn(0.0, 179.999)
                val baseline = distance(
                    a.cameraPose.position,
                    b.cameraPose.position,
                )
                pairs += PairMetric(
                    first = a.cameraId,
                    second = b.cameraId,
                    angleDegrees = angle,
                    baseline = baseline,
                )
            }
        }

        val angles = pairs.map { it.angleDegrees }.filter(Double::isFinite).sorted()
        val baselines = pairs.map { it.baseline }
            .filter { it.isFinite() && it > 1e-6 }
            .sorted()
        val medianParallax = median(angles)
        val minimumParallax = angles.minOrNull() ?: 0.0
        val medianBaseline = median(baselines)
        val cameraDiversity =
            cameraDiversity(unique, pairs, medianBaseline)

        val scores = unique.map { observation ->
            val related = pairs.filter {
                it.first == observation.cameraId ||
                    it.second == observation.cameraId
            }
            val bestParallax =
                related.maxOfOrNull { it.angleDegrees } ?: 0.0
            val parallaxScore = parallaxScore(bestParallax)
            val baselineScore =
                baselineScore(related, medianBaseline)
            val peerConsistency =
                peerConsistency(related)

            val confidence =
                (
                    observation.sourceQuality.coerceIn(0.0, 1.0) *
                        observation.visibility.coerceIn(0.0, 1.0)
                ).coerceIn(0.05, 1.0)

            val uncertaintyPenalty =
                observation.observationUncertainty
                    .coerceIn(0.0, 1.0) * 0.70

            val raw =
                (
                    parallaxScore * 0.48 +
                        baselineScore * 0.18 +
                        peerConsistency * 0.18 +
                        confidence * 0.16 -
                        uncertaintyPenalty * 0.12
                ).coerceIn(0.0, 1.0)

            val conditioning =
                (
                    raw * 0.82 +
                        cameraDiversity * 0.18
                ).coerceIn(0.0, 1.0)

            val contradictory =
                related.size >= 2 &&
                    peerConsistency < 0.20

            val weak =
                bestParallax <
                    config.minimumParallaxDegrees ||
                    conditioning <
                    config.minimumConditioning

            val multiplier =
                when {
                    contradictory -> 0.20
                    weak ->
                        min(
                            0.68,
                            0.35 + conditioning * 0.55,
                        )
                    else ->
                        (
                            0.78 +
                                conditioning * 0.32
                            ).coerceIn(0.78, 1.10)
                }

            val diagnostics = buildList {
                if (bestParallax < config.minimumParallaxDegrees) {
                    add("LOW_PARALLAX")
                }
                if (bestParallax >= config.strongParallaxDegrees) {
                    add("STRONG_PARALLAX")
                }
                if (baselineScore < 0.25) {
                    add("LOW_BASELINE_DIVERSITY")
                }
                if (peerConsistency < 0.35) {
                    add("PEER_GEOMETRY_DISAGREEMENT")
                }
                if (observation.observationUncertainty >= 0.70) {
                    add("HIGH_OBSERVATION_UNCERTAINTY")
                }
                if (contradictory) {
                    add("GEOMETRIC_CONTRADICTION")
                }
            }

            ObservationScore(
                cameraId = observation.cameraId,
                parallaxScore = parallaxScore,
                baselineScore = baselineScore,
                peerConsistencyScore = peerConsistency,
                confidenceScore = confidence,
                conditioningScore = conditioning,
                weightMultiplier = multiplier,
                sourceDiversityScore =
                    (
                        conditioning * 0.72 +
                            cameraDiversity * 0.28
                    ).coerceIn(0.0, 1.0),
                weakGeometry = weak,
                contradictoryGeometry = contradictory,
                diagnostics = diagnostics,
            )
        }

        val orderedConditioning =
            scores.map { it.conditioningScore }.sorted()
        val global =
            (
                median(orderedConditioning) * 0.72 +
                    lowerTail(orderedConditioning) * 0.28
            ).coerceIn(0.0, 1.0)

        val minimumObservationConditioningScore =
            scores
                .map { it.conditioningScore }
                .minOrNull()
                ?: 0.0

        val authoritative =
            unique.size >= 2 &&
                global >= 0.45 &&
                minimumObservationConditioningScore >=
                    config.minimumConditioning &&
                medianParallax >= config.minimumParallaxDegrees

        val warnings = buildList {
            if (medianParallax < config.minimumParallaxDegrees) {
                add("RIG_PARALLAX_TOO_SMALL")
            }
            if (global < 0.45) {
                add("GLOBAL_GEOMETRIC_CONDITIONING_DEGRADED")
            }
            if (scores.count { it.contradictoryGeometry } >
                max(1, scores.size / 3)
            ) {
                add("MULTI_CAMERA_GEOMETRY_CONTRADICTION")
            }
            if (cameraDiversity < 0.40) {
                add("CAMERA_DIVERSITY_LOW")
            }
            if (
                minimumObservationConditioningScore <
                    config.minimumConditioning
            ) {
                add("LOWEST_CAMERA_CONDITIONING_BELOW_TRUST")
            }
        }.distinct()

        return Result(
            scores = scores,
            globalConditioningScore = global,
            medianParallaxDegrees = medianParallax,
            minimumParallaxDegrees = minimumParallax,
            cameraDiversityScore = cameraDiversity,
            minimumObservationConditioningScore =
                minimumObservationConditioningScore,
            authoritative = authoritative,
            warnings = warnings,
        )
    }

    private fun parallaxScore(degrees: Double): Double {
        if (degrees < config.minimumParallaxDegrees) return 0.0
        val normalized =
            (
                (degrees - config.minimumParallaxDegrees) /
                    (
                        config.strongParallaxDegrees -
                            config.minimumParallaxDegrees
                    ).coerceAtLeast(1e-6)
            ).coerceIn(0.0, 1.0)

        return if (degrees < config.targetParallaxDegrees) {
            (normalized * 0.85).coerceIn(0.0, 0.85)
        } else {
            (
                0.15 +
                    0.85 *
                        sin(
                            (
                                (
                                    min(
                                        degrees,
                                        config.strongParallaxDegrees,
                                    ) -
                                        config.targetParallaxDegrees
                                ) /
                                    (
                                        config.strongParallaxDegrees -
                                            config.targetParallaxDegrees
                                    ).coerceAtLeast(1e-6)
                            ).coerceIn(0.0, 1.0) *
                                Math.PI * 0.5,
                        )
            ).coerceIn(0.0, 1.0)
        }
    }

    private fun baselineScore(
        related: List<PairMetric>,
        medianBaseline: Double,
    ): Double {
        if (related.isEmpty() || medianBaseline <= 1e-6) return 0.0
        val ratios = related.map { pair ->
            (pair.baseline / medianBaseline).coerceIn(0.0, 3.0)
        }
        return (ratios.average() / 1.75).coerceIn(0.0, 1.0)
    }

    private fun peerConsistency(
        related: List<PairMetric>,
    ): Double {
        if (related.size < 2) return 0.58
        val values = related.map { it.angleDegrees }
            .filter(Double::isFinite)
            .sorted()
        if (values.size < 2) return 0.58
        val center = median(values)
        if (center <= 1e-6) return 0.10
        val deviation = median(values.map { abs(it - center) })
        val score =
            (
                1.0 -
                    (
                        deviation /
                            center
                    ).coerceIn(0.0, 2.0) * 0.82
            ).coerceIn(0.0, 1.0)
        return if (
            values.any {
                abs(it - center) >= config.contradictionDegrees
            }
        ) {
            score * 0.48
        } else {
            score
        }
    }

    private fun cameraDiversity(
        observations: List<TriangulationEngine.Observation>,
        pairs: List<PairMetric>,
        medianBaseline: Double,
    ): Double {
        if (observations.size < 2 || pairs.isEmpty()) return 0.0
        val cameraFactor =
            (
                observations.map { it.cameraId }.distinct().size /
                    4.0
            ).coerceIn(0.0, 1.0)
        val baselineFactor =
            if (medianBaseline <= 1e-6) {
                0.0
            } else {
                (
                    medianBaseline /
                        (
                            pairs.maxOfOrNull { it.baseline } ?: medianBaseline
                        ).coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)
            }
        val angularCoverage =
            (
                pairs.count {
                    it.angleDegrees >=
                        config.minimumParallaxDegrees
                }.toDouble() /
                    pairs.size.coerceAtLeast(1).toDouble()
            ).coerceIn(0.0, 1.0)
        return (
            cameraFactor * 0.40 +
                baselineFactor * 0.25 +
                angularCoverage * 0.35
        ).coerceIn(0.0, 1.0)
    }

    private fun rayFor(
        observation: TriangulationEngine.Observation,
    ): Vec3? {
        val fx = observation.intrinsics.fx
            .takeIf { it.isFinite() && it > 1e-6 } ?: return null
        val fy = observation.intrinsics.fy
            .takeIf { it.isFinite() && it > 1e-6 } ?: return null
        val x = (observation.u - observation.intrinsics.cx) / fx
        val y = (observation.v - observation.intrinsics.cy) / fy
        val length =
            sqrt(x * x + y * y + 1.0).coerceAtLeast(1e-12)
        val local =
            Vec3(
                x / length,
                y / length,
                1.0 / length,
            )
        val world = observation.cameraPose.rotation.rotate(local)
        val norm = world.norm()
            .takeIf { it.isFinite() && it > 1e-9 } ?: return null
        return Vec3(
            world.x / norm,
            world.y / norm,
            world.z / norm,
        )
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val middle = values.size / 2
        return if (values.size % 2 == 0) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    }

    private fun lowerTail(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        return values[
            (values.size * 0.20).toInt()
                .coerceIn(0, values.lastIndex)
        ].coerceIn(0.0, 1.0)
    }

    private fun distance(a: Vec3, b: Vec3): Double =
        sqrt(
            (a.x - b.x) * (a.x - b.x) +
                (a.y - b.y) * (a.y - b.y) +
                (a.z - b.z) * (a.z - b.z),
        )

    private data class PairMetric(
        val first: String,
        val second: String,
        val angleDegrees: Double,
        val baseline: Double,
    )
}
