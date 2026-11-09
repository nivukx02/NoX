package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Anisotropic camera-contribution selector.
 *
 * Dense reconstruction becomes fragile when the remaining camera rays are nearly parallel.
 * In that situation the weakest eigen-direction of the accumulated ray information matrix is
 * the direction in which 3D position is least constrained. A camera that observes from a
 * complementary direction can contribute more useful information than a camera with slightly
 * higher detector confidence.
 *
 * The selector computes a bounded ray-information matrix:
 *
 *   H = sum_i w_i (I - r_i r_i^T)
 *
 * and extracts its weakest principal axis. Each candidate view is then rewarded for providing
 * information perpendicular to that weak axis. This is an information-direction correction to
 * ordinary confidence/diversity ranking, not a claim of global Bayesian optimality.
 */
class AnisotropicViewContributionEngine(
    private val config: Config = Config.production(),
) {

    data class Config(
        val minimumObservationWeight: Double = 0.05,
        val confidenceWeight: Double = 0.48,
        val visibilityWeight: Double = 0.16,
        val uncertaintyWeight: Double = 0.16,
        val temporalWeight: Double = 0.12,
        val sourceDiversityWeight: Double = 0.08,
        val principalAxisWeight: Double = 0.42,
        val noveltyWeight: Double = 0.28,
        val baselineWeight: Double = 0.18,
        val appearanceWeight: Double = 0.12,
        val damping: Double = 1e-9,
    ) {
        init {
            require(minimumObservationWeight in 0.0..1.0)
            listOf(
                confidenceWeight,
                visibilityWeight,
                uncertaintyWeight,
                temporalWeight,
                sourceDiversityWeight,
                principalAxisWeight,
                noveltyWeight,
                baselineWeight,
                appearanceWeight,
            ).forEach {
                require(it >= 0.0)
            }
            require(damping > 0.0)
        }

        companion object {
            fun production(): Config = Config()

            fun precision(): Config = Config(
                confidenceWeight = 0.42,
                visibilityWeight = 0.14,
                uncertaintyWeight = 0.14,
                temporalWeight = 0.10,
                sourceDiversityWeight = 0.06,
                principalAxisWeight = 0.54,
                noveltyWeight = 0.30,
                baselineWeight = 0.10,
                appearanceWeight = 0.10,
            )
        }
    }

    data class Candidate(
        val observation: TriangulationEngine.Observation,
        val semanticConfidence: Double = 1.0,
        val visibility: Double = 1.0,
        val sourceDiversity: Double = 1.0,
        val appearanceEvidence: Double = 1.0,
    ) {
        init {
            require(semanticConfidence in 0.0..1.0)
            require(visibility in 0.0..1.0)
            require(sourceDiversity in 0.0..1.0)
            require(appearanceEvidence in 0.0..1.0)
        }
    }

    data class ViewScore(
        val cameraId: String,
        val evidenceScore: Double,
        val principalAxisConstraint: Double,
        val noveltyScore: Double,
        val baselineScore: Double,
        val totalScore: Double,
        val weakDirectionAngleDegrees: Double,
        val weakestAxis: Vec3,
    )

    data class Result(
        val weakestInformationAxis: Vec3,
        val weakestEigenvalue: Double,
        val strongestEigenvalue: Double,
        val conditionScore: Double,
        val selected: List<ViewScore>,
        val warnings: List<String>,
    )

    fun rank(
        candidates: List<Candidate>,
        alreadySelected: List<Candidate> = emptyList(),
        limit: Int = candidates.size,
    ): Result {
        if (candidates.isEmpty()) {
            return Result(
                weakestInformationAxis = Vec3(1.0, 0.0, 0.0),
                weakestEigenvalue = 0.0,
                strongestEigenvalue = 0.0,
                conditionScore = 0.0,
                selected = emptyList(),
                warnings = listOf("no-camera-candidates"),
            )
        }

        val rays =
            candidates.map {
                worldRay(
                    observation = it.observation,
                )
            }

        val matrix = informationMatrix(
            candidates = candidates,
            rays = rays,
        )

        val eig = jacobiEigen3(matrix)

        val weakestIndex =
            eig.values.indices.minByOrNull {
                eig.values[it]
            } ?: 0
        val strongestIndex =
            eig.values.indices.maxByOrNull {
                eig.values[it]
            } ?: 0

        val weakestAxis =
            eig.vectors[weakestIndex]
                .normalized()
                .takeIf { it.norm() > 1e-9 }
                ?: Vec3(1.0, 0.0, 0.0)

        val weakestEigenvalue =
            eig.values[weakestIndex].coerceAtLeast(0.0)
        val strongestEigenvalue =
            eig.values[strongestIndex].coerceAtLeast(config.damping)
        val conditionScore =
            (
                weakestEigenvalue /
                    strongestEigenvalue
            ).coerceIn(0.0, 1.0)

        val scored =
            candidates.mapIndexed { index, candidate ->
                val ray = rays[index]
                val baseEvidence = baseEvidence(candidate)
                val principalConstraint =
                    (
                        1.0 -
                            abs(
                                ray.dot(weakestAxis)
                            ).coerceIn(0.0, 1.0)
                    ).coerceIn(0.0, 1.0)

                val novelty =
                    if (alreadySelected.isEmpty()) {
                        1.0
                    } else {
                        alreadySelected.maxOfOrNull { selected ->
                            pairwiseNovelty(
                                candidate.observation,
                                selected.observation,
                            )
                        } ?: 0.0
                    }

                val baseline =
                    if (alreadySelected.isEmpty()) {
                        1.0
                    } else {
                        alreadySelected
                            .map {
                                baselineScore(
                                    candidate.observation,
                                    it.observation,
                                )
                            }
                            .maxOrNull()
                            ?: 0.0
                    }

                val total =
                    (
                        baseEvidence *
                            (
                                1.0 -
                                    config.principalAxisWeight
                            ) +
                            principalConstraint *
                                config.principalAxisWeight
                    ) *
                    (
                        0.70 +
                            0.30 *
                                (
                                    novelty *
                                        config.noveltyWeight +
                                        baseline *
                                            config.baselineWeight +
                                        candidate.appearanceEvidence *
                                            config.appearanceWeight
                                ).coerceIn(0.0, 1.0)
                    )

                ViewScore(
                    cameraId = candidate.observation.cameraId,
                    evidenceScore = baseEvidence,
                    principalAxisConstraint = principalConstraint,
                    noveltyScore = novelty,
                    baselineScore = baseline,
                    totalScore = total.coerceIn(0.0, 1.0),
                    weakDirectionAngleDegrees =
                        Math.toDegrees(
                            acos(
                                abs(ray.dot(weakestAxis))
                                    .coerceIn(0.0, 1.0),
                            ),
                        ),
                    weakestAxis = weakestAxis,
                )
            }
                .sortedByDescending { it.totalScore }
                .take(limit.coerceAtLeast(0))

        val warnings = buildList {
            if (conditionScore < 0.10) {
                add("3d-information-weak-axis-dominant")
            }
            if (conditionScore < 0.20) {
                add("camera-selection-needs-axis-complement")
            }
            if (weakestEigenvalue <= config.damping * 10.0) {
                add("ray-information-rank-deficient")
            }
            if (scored.any { it.principalAxisConstraint < 0.35 }) {
                add("some-views-do-not-constrain-weak-axis")
            }
        }.distinct()

        return Result(
            weakestInformationAxis = weakestAxis,
            weakestEigenvalue = weakestEigenvalue,
            strongestEigenvalue = strongestEigenvalue,
            conditionScore = conditionScore,
            selected = scored,
            warnings = warnings,
        )
    }

    private fun baseEvidence(
        candidate: Candidate,
    ): Double =
        (
            candidate.semanticConfidence *
                config.confidenceWeight +
                candidate.visibility *
                config.visibilityWeight +
                (
                    1.0 -
                        candidate.observation.observationUncertainty
                            .coerceIn(0.0, 1.0)
                    ) *
                config.uncertaintyWeight +
                candidate.observation.temporalAlignmentConfidence *
                config.temporalWeight +
                candidate.sourceDiversity *
                config.sourceDiversityWeight
        ).coerceIn(0.0, 1.0)

    private fun worldRay(
        observation: TriangulationEngine.Observation,
    ): Vec3 {
        val k = observation.intrinsics
        val x =
            (
                observation.u -
                    k.cx
            ) / k.fx.coerceAtLeast(1e-9)
        val y =
            (
                observation.v -
                    k.cy
            ) / k.fy.coerceAtLeast(1e-9)

        return observation.cameraPose.rotation
            .rotate(
                Vec3(
                    x,
                    y,
                    1.0,
                ).normalized(),
            )
            .normalized()
    }

    private fun informationMatrix(
        candidates: List<Candidate>,
        rays: List<Vec3>,
    ): Array<DoubleArray> {
        val h =
            Array(3) {
                DoubleArray(3)
            }

        candidates.forEachIndexed { index, candidate ->
            val ray = rays[index]
            val weight =
                (
                    baseEvidence(candidate)
                ).coerceAtLeast(
                    config.minimumObservationWeight,
                )

            val ix = 1.0 - ray.x * ray.x
            val iy = 1.0 - ray.y * ray.y
            val iz = 1.0 - ray.z * ray.z

            h[0][0] += weight * ix
            h[1][1] += weight * iy
            h[2][2] += weight * iz
            h[0][1] += -weight * ray.x * ray.y
            h[1][0] = h[0][1]
            h[0][2] += -weight * ray.x * ray.z
            h[2][0] = h[0][2]
            h[1][2] += -weight * ray.y * ray.z
            h[2][1] = h[1][2]
        }

        h[0][0] += config.damping
        h[1][1] += config.damping
        h[2][2] += config.damping
        return h
    }

    private data class Eigen3(
        val values: DoubleArray,
        val vectors: Array<Vec3>,
    )

    private fun jacobiEigen3(
        original: Array<DoubleArray>,
    ): Eigen3 {
        val a =
            Array(3) { row ->
                original[row].clone()
            }

        val v =
            Array(3) { row ->
                DoubleArray(3)
            }

        for (i in 0..2) {
            v[i][i] = 1.0
        }

        repeat(18) {
            var p = 0
            var q = 1
            var maximum = abs(a[0][1])

            fun consider(
                row: Int,
                col: Int,
            ) {
                val value =
                    abs(a[row][col])
                if (value > maximum) {
                    maximum = value
                    p = row
                    q = col
                }
            }

            consider(0, 2)
            consider(1, 2)

            if (maximum <= 1e-12) return@repeat

            val angle =
                0.5 *
                    kotlin.math.atan2(
                        2.0 * a[p][q],
                        a[q][q] - a[p][p],
                    )

            val c = kotlin.math.cos(angle)
            val s = kotlin.math.sin(angle)

            for (k in 0..2) {
                val apk = a[p][k]
                val aqk = a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            }

            for (k in 0..2) {
                val akp = a[k][p]
                val akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            }

            for (k in 0..2) {
                val vkp = v[k][p]
                val vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }

        val values =
            doubleArrayOf(
                a[0][0],
                a[1][1],
                a[2][2],
            )

        val vectors =
            arrayOf(
                Vec3(v[0][0], v[1][0], v[2][0]),
                Vec3(v[0][1], v[1][1], v[2][1]),
                Vec3(v[0][2], v[1][2], v[2][2]),
            )

        return Eigen3(
            values = values,
            vectors = vectors,
        )
    }

    private fun pairwiseNovelty(
        a: TriangulationEngine.Observation,
        b: TriangulationEngine.Observation,
    ): Double =
        (
            baselineScore(a, b) * 0.58 +
                orientationNovelty(a, b) * 0.42
        ).coerceIn(0.0, 1.0)

    private fun baselineScore(
        a: TriangulationEngine.Observation,
        b: TriangulationEngine.Observation,
    ): Double {
        val baseline =
            a.cameraPose.position.distanceTo(
                b.cameraPose.position,
            )
        return (
            baseline /
                (baseline + 0.50)
        ).coerceIn(0.0, 1.0)
    }

    private fun orientationNovelty(
        a: TriangulationEngine.Observation,
        b: TriangulationEngine.Observation,
    ): Double {
        val forwardA =
            a.cameraPose.rotation.rotate(
                Vec3(0.0, 0.0, 1.0),
            ).normalized()
        val forwardB =
            b.cameraPose.rotation.rotate(
                Vec3(0.0, 0.0, 1.0),
            ).normalized()

        return (
            acos(
                forwardA.dot(forwardB)
                    .coerceIn(-1.0, 1.0),
            ) / Math.PI
        ).coerceIn(0.0, 1.0)
    }
}
