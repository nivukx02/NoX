package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Rotation-invariant geometric shape validator for reconstructed facial point clouds.
 *
 * Temporal face tracking already checks landmark edge and radial ratios. This engine adds an
 * independent global-shape signal using the eigenvalue signature of the centered 3D covariance
 * matrix. A rigid head rotation changes the basis vectors but preserves the spectrum, while
 * incorrect depth, stretched geometry, collapsed regions, and unstable triangulation alter the
 * spectrum.
 *
 * The implementation deliberately avoids a heavyweight numerical dependency. A bounded Jacobi
 * iteration solves the symmetric 3x3 covariance eigenproblem, which is sufficient for the compact
 * facial landmark cloud used here.
 */
class HighEndFaceShapeAlignmentEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumPoints: Int = 14,
        val minimumOverlapRatio: Double = 0.72,
        val minimumScaleMeters: Double = 0.01,
        val maximumScaleRatioDeviation: Double = 0.28,
        val maximumAnisotropyRatioDeviation: Double = 0.32,
        val covarianceIterations: Int = 18,
        val covarianceTolerance: Double = 1e-9,
        val smoothingAlpha: Double = 0.24,
        val warningScore: Double = 0.72,
        val rejectScore: Double = 0.46,
        val persistentWarningFrames: Int = 4,
        val persistentRejectFrames: Int = 7,
    ) {
        init {
            require(minimumPoints >= 3)
            require(minimumOverlapRatio in 0.0..1.0)
            require(minimumScaleMeters > 0.0)
            require(maximumScaleRatioDeviation in 0.0..1.0)
            require(maximumAnisotropyRatioDeviation in 0.0..1.0)
            require(covarianceIterations >= 4)
            require(covarianceTolerance > 0.0)
            require(smoothingAlpha in 0.0..1.0)
            require(warningScore in 0.0..1.0)
            require(rejectScore in 0.0..warningScore)
            require(persistentWarningFrames >= 1)
            require(persistentRejectFrames >= persistentWarningFrames)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val pointsBySemanticId: Map<Int, Vec3>,
        val baseConfidence: Double,
    )

    data class Result(
        val timestampNanos: Long,
        val accepted: Boolean,
        val confidence: Double,
        val overlapRatio: Double,
        val scaleRatio: Double,
        val shapeScaleMeters: Double,
        val anisotropyDeviation: Double,
        val eigenvalues: List<Double>,
        val persistentWarningFrames: Int,
        val persistentRejectFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class Signature(
        val eigenvalues: List<Double>,
        val scaleMeters: Double,
        val anisotropy: List<Double>,
    )

    private data class History(
        var timestampNanos: Long,
        var ids: Set<Int>,
        var signature: Signature,
        var confidenceEma: Double,
        var warningFrames: Int,
        var rejectFrames: Int,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val clean = input.pointsBySemanticId
            .filterValues {
                it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.z.isFinite()
            }

        if (clean.size < config.minimumPoints) {
            return Result(
                timestampNanos = input.timestampNanos,
                accepted = false,
                confidence = 0.0,
                overlapRatio = 0.0,
                scaleRatio = 1.0,
                shapeScaleMeters = 0.0,
                anisotropyDeviation = 1.0,
                eigenvalues = emptyList(),
                persistentWarningFrames = history?.warningFrames ?: 0,
                persistentRejectFrames = history?.rejectFrames ?: 0,
                warnings = listOf(
                    "face-shape-point-count-below-production-minimum",
                ),
                recommendations = listOf(
                    "Recover more facial geometry before trusting 3D head shape.",
                ),
            )
        }

        val signature = signature(clean.values.toList())
            ?: return Result(
                timestampNanos = input.timestampNanos,
                accepted = false,
                confidence = 0.0,
                overlapRatio = 0.0,
                scaleRatio = 1.0,
                shapeScaleMeters = 0.0,
                anisotropyDeviation = 1.0,
                eigenvalues = emptyList(),
                persistentWarningFrames = history?.warningFrames ?: 0,
                persistentRejectFrames = history?.rejectFrames ?: 0,
                warnings = listOf("face-shape-covariance-invalid"),
                recommendations = listOf(
                    "Hold facial geometry when the 3D covariance becomes degenerate.",
                ),
            )

        val previous = history
        if (previous == null) {
            history = History(
                timestampNanos = input.timestampNanos,
                ids = clean.keys,
                signature = signature,
                confidenceEma = input.baseConfidence.coerceIn(0.0, 1.0),
                warningFrames = 0,
                rejectFrames = 0,
            )
            val initialConfidence = (
                input.baseConfidence.coerceIn(0.0, 1.0) * 0.72 +
                    scaleAvailability(signature.scaleMeters) * 0.28
            ).coerceIn(0.0, 1.0)

            return Result(
                timestampNanos = input.timestampNanos,
                accepted =
                    initialConfidence >= config.rejectScore &&
                        signature.scaleMeters >= config.minimumScaleMeters,
                confidence = initialConfidence,
                overlapRatio = 1.0,
                scaleRatio = 1.0,
                shapeScaleMeters = signature.scaleMeters,
                anisotropyDeviation = 0.0,
                eigenvalues = signature.eigenvalues,
                persistentWarningFrames = 0,
                persistentRejectFrames = 0,
                warnings = emptyList(),
                recommendations = emptyList(),
            )
        }

        val overlapRatio =
            clean.keys.intersect(previous.ids).size.toDouble() /
                max(1, min(clean.size, previous.ids.size)).toDouble()

        val scaleRatio =
            if (previous.signature.scaleMeters <= 1e-9) {
                1.0
            } else {
                signature.scaleMeters /
                    previous.signature.scaleMeters
            }

        val scaleDeviation = abs(scaleRatio - 1.0)

        val anisotropyDeviation =
            anisotropyDeviation(
                current = signature.eigenvalues,
                previous = previous.signature.eigenvalues,
            )

        val overlapScore =
            overlapRatio.coerceIn(0.0, 1.0)

        val scaleScore =
            (
                1.0 -
                    scaleDeviation /
                        config.maximumScaleRatioDeviation
            ).coerceIn(0.0, 1.0)

        val anisotropyScore =
            (
                1.0 -
                    anisotropyDeviation /
                        config.maximumAnisotropyRatioDeviation
            ).coerceIn(0.0, 1.0)

        val scaleAvailability =
            scaleAvailability(signature.scaleMeters)

        val confidence = (
            overlapScore * 0.24 +
                scaleScore * 0.28 +
                anisotropyScore * 0.34 +
                scaleAvailability * 0.14
        ).coerceIn(0.0, 1.0)

        val warning =
            confidence < config.warningScore ||
                overlapRatio < config.minimumOverlapRatio

        val reject =
            confidence < config.rejectScore ||
                overlapRatio < config.minimumOverlapRatio * 0.62 ||
                signature.scaleMeters < config.minimumScaleMeters

        if (warning) {
            previous.warningFrames += 1
        } else {
            previous.warningFrames =
                max(0, previous.warningFrames - 1)
        }

        if (reject) {
            previous.rejectFrames += 1
        } else {
            previous.rejectFrames =
                max(0, previous.rejectFrames - 1)
        }

        previous.confidenceEma =
            previous.confidenceEma * (1.0 - config.smoothingAlpha) +
                confidence * config.smoothingAlpha
        previous.timestampNanos = input.timestampNanos
        previous.ids = clean.keys
        previous.signature = signature

        val persistentWarning =
            previous.warningFrames >= config.persistentWarningFrames
        val persistentReject =
            previous.rejectFrames >= config.persistentRejectFrames

        val warnings = buildList {
            if (overlapRatio < config.minimumOverlapRatio) {
                add("face-shape-semantic-overlap-low")
            }
            if (scaleDeviation >
                config.maximumScaleRatioDeviation
            ) {
                add("face-shape-scale-jump")
            }
            if (anisotropyDeviation >
                config.maximumAnisotropyRatioDeviation
            ) {
                add("face-shape-anisotropy-jump")
            }
            if (signature.scaleMeters < config.minimumScaleMeters) {
                add("face-shape-scale-degenerate")
            }
            if (persistentWarning) {
                add("face-shape-geometric-consistency-persistent-warning")
            }
            if (persistentReject) {
                add("face-shape-geometric-consistency-persistent-reject")
            }
        }.distinct()

        val recommendations = buildList {
            if (overlapRatio < config.minimumOverlapRatio) {
                add("Recover a stable facial landmark set before accepting shape changes.")
            }
            if (scaleDeviation >
                config.maximumScaleRatioDeviation
            ) {
                add("Check depth scale and face-camera distance before trusting this frame.")
            }
            if (anisotropyDeviation >
                config.maximumAnisotropyRatioDeviation
            ) {
                add("Hold facial geometry while reconstructed depth stabilizes.")
            }
            if (persistentReject) {
                add("Keep the last trusted facial shape until geometric evidence recovers.")
            }
        }.distinct()

        return Result(
            timestampNanos = input.timestampNanos,
            accepted =
                !persistentReject &&
                    confidence >= config.rejectScore &&
                    input.baseConfidence >= 0.20,
            confidence = (
                previous.confidenceEma * 0.42 +
                    confidence * 0.58
            ).coerceIn(0.0, 1.0),
            overlapRatio = overlapRatio,
            scaleRatio = scaleRatio,
            shapeScaleMeters = signature.scaleMeters,
            anisotropyDeviation = anisotropyDeviation,
            eigenvalues = signature.eigenvalues,
            persistentWarningFrames = previous.warningFrames,
            persistentRejectFrames = previous.rejectFrames,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    private fun signature(
        points: List<Vec3>,
    ): Signature? {
        if (points.size < 3) return null

        val center = Vec3(
            points.sumOf { it.x } / points.size,
            points.sumOf { it.y } / points.size,
            points.sumOf { it.z } / points.size,
        )

        var xx = 0.0
        var xy = 0.0
        var xz = 0.0
        var yy = 0.0
        var yz = 0.0
        var zz = 0.0

        points.forEach { point ->
            val d = point - center
            xx += d.x * d.x
            xy += d.x * d.y
            xz += d.x * d.z
            yy += d.y * d.y
            yz += d.y * d.z
            zz += d.z * d.z
        }

        val scale = points.size.toDouble()
        val matrix = arrayOf(
            doubleArrayOf(xx / scale, xy / scale, xz / scale),
            doubleArrayOf(xy / scale, yy / scale, yz / scale),
            doubleArrayOf(xz / scale, yz / scale, zz / scale),
        )

        val eigenvalues =
            jacobiEigenvalues(matrix)
                .sortedDescending()
                .map { it.coerceAtLeast(0.0) }

        if (eigenvalues.size != 3 ||
            eigenvalues.any { !it.isFinite() }
        ) {
            return null
        }

        val rmsRadius = sqrt(
            (
                eigenvalues.sum()
            ).coerceAtLeast(0.0),
        )

        val anisotropy = listOf(
            ratio(
                eigenvalues[0],
                eigenvalues[1],
            ),
            ratio(
                eigenvalues[1],
                eigenvalues[2],
            ),
        )

        return Signature(
            eigenvalues = eigenvalues,
            scaleMeters = rmsRadius,
            anisotropy = anisotropy,
        )
    }

    private fun jacobiEigenvalues(
        source: Array<DoubleArray>,
    ): List<Double> {
        val matrix = Array(3) { row ->
            source[row].clone()
        }

        repeat(config.covarianceIterations) {
            var p = 0
            var q = 1
            var maximum = abs(matrix[0][1])

            if (abs(matrix[0][2]) > maximum) {
                p = 0
                q = 2
                maximum = abs(matrix[0][2])
            }

            if (abs(matrix[1][2]) > maximum) {
                p = 1
                q = 2
                maximum = abs(matrix[1][2])
            }

            if (maximum <= config.covarianceTolerance) {
                return@repeat
            }

            val app = matrix[p][p]
            val aqq = matrix[q][q]
            val apq = matrix[p][q]

            val phi =
                0.5 *
                    kotlin.math.atan2(
                        2.0 * apq,
                        aqq - app,
                    )

            val cosine = kotlin.math.cos(phi)
            val sine = kotlin.math.sin(phi)

            for (index in 0..2) {
                if (index == p || index == q) continue
                val mip = matrix[index][p]
                val miq = matrix[index][q]

                matrix[index][p] =
                    cosine * mip - sine * miq
                matrix[p][index] =
                    matrix[index][p]

                matrix[index][q] =
                    sine * mip + cosine * miq
                matrix[q][index] =
                    matrix[index][q]
            }

            matrix[p][p] =
                cosine * cosine * app -
                    2.0 * cosine * sine * apq +
                    sine * sine * aqq

            matrix[q][q] =
                sine * sine * app +
                    2.0 * cosine * sine * apq +
                    cosine * cosine * aqq

            matrix[p][q] = 0.0
            matrix[q][p] = 0.0
        }

        return listOf(
            matrix[0][0],
            matrix[1][1],
            matrix[2][2],
        )
    }

    private fun anisotropyDeviation(
        current: List<Double>,
        previous: List<Double>,
    ): Double {
        if (current.size < 3 || previous.size < 3) return 1.0

        val currentA = ratio(current[0], current[1])
        val currentB = ratio(current[1], current[2])
        val previousA = ratio(previous[0], previous[1])
        val previousB = ratio(previous[1], previous[2])

        return (
            abs(currentA / previousA.coerceAtLeast(1e-6) - 1.0) +
                abs(currentB / previousB.coerceAtLeast(1e-6) - 1.0)
        ) * 0.5
    }

    private fun ratio(
        numerator: Double,
        denominator: Double,
    ): Double =
        if (denominator <= 1e-9) {
            if (numerator <= 1e-9) 1.0 else 1000.0
        } else {
            (numerator / denominator)
                .coerceIn(0.001, 1000.0)
        }

    private fun scaleAvailability(
        scaleMeters: Double,
    ): Double =
        (
            scaleMeters /
                (config.minimumScaleMeters * 5.0)
        ).coerceIn(0.0, 1.0)
}
