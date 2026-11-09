package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EyeGazeCalibrationProfile
import com.nivukx.nox.core.model.EyeGazeCalibrationSample
import com.nivukx.nox.core.model.Vec2
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Per-subject eye-camera calibration.
 *
 * A short calibration sequence samples known fixation targets. A weighted affine model maps
 * normalized iris displacement to eye-relative yaw/pitch. This compensates for camera placement,
 * eye geometry, crop, and per-device optics without changing the runtime tracker.
 */
class EyeGazeCalibrationEngine {
    fun calibrate(
        samples: List<EyeGazeCalibrationSample>,
        nowNanos: Long = System.nanoTime(),
    ): EyeGazeCalibrationProfile? {
        val valid = samples.filter {
            it.weight > 1e-6 &&
                it.offset.x.isFinite() &&
                it.offset.y.isFinite() &&
                it.targetYawRadians.isFinite() &&
                it.targetPitchRadians.isFinite()
        }

        if (valid.size < 5) return null

        val yaw = solveRobustAffine(
            samples = valid,
            target = { sample -> sample.targetYawRadians },
        ) ?: return null

        val pitch = solveRobustAffine(
            samples = valid,
            target = { sample -> sample.targetPitchRadians },
        ) ?: return null

        val rms = sqrt(
            valid.map { sample ->
                val predicted = Vec2(
                    yaw[0] * sample.offset.x +
                        yaw[1] * sample.offset.y +
                        yaw[2],
                    pitch[0] * sample.offset.x +
                        pitch[1] * sample.offset.y +
                        pitch[2],
                )

                val dx = predicted.x - sample.targetYawRadians
                val dy = predicted.y - sample.targetPitchRadians
                dx * dx + dy * dy
            }.average(),
        )

        val robustCoverageConfidence =
            (
                valid.size.toDouble() /
                    12.0
            ).coerceIn(0.0, 1.0)

        if (!rms.isFinite() || robustCoverageConfidence < 0.35) {
            return null
        }

        return EyeGazeCalibrationProfile(
            yawCoefficients = yaw,
            pitchCoefficients = pitch,
            samples = valid.size,
            rmsAngularErrorRadians = rms,
            calibratedAtNanos = nowNanos,
        )
    }

    private fun solveRobustAffine(
        samples: List<EyeGazeCalibrationSample>,
        target: (EyeGazeCalibrationSample) -> Double,
    ): DoubleArray? {
        var coefficients =
            solveWeightedAffine(
                samples = samples,
                target = target,
                weights = samples.map {
                    it.weight.coerceAtLeast(1e-6)
                },
            ) ?: return null

        repeat(6) {
            val residuals =
                samples.map { sample ->
                    val predicted =
                        coefficients[0] * sample.offset.x +
                            coefficients[1] * sample.offset.y +
                            coefficients[2]
                    target(sample) - predicted
                }

            val finiteResiduals =
                residuals.filter(Double::isFinite)

            if (finiteResiduals.size < 5) return@repeat

            val sorted = finiteResiduals.sorted()
            val median =
                sorted[sorted.size / 2]
            val deviations =
                finiteResiduals
                    .map { abs(it - median) }
                    .sorted()
            val mad =
                deviations[deviations.size / 2]
            val scale =
                max(
                    1e-4,
                    1.4826 * mad,
                )

            val robustWeights =
                samples.mapIndexed { index, sample ->
                    val residual =
                        residuals[index]
                    val z =
                        abs(residual - median) /
                            scale
                    val robust =
                        if (z <= 2.5) {
                            1.0
                        } else {
                            2.5 /
                                max(
                                    2.5,
                                    z,
                                )
                        }

                    sample.weight
                        .coerceAtLeast(1e-6) *
                        robust.coerceIn(0.08, 1.0)
                }

            val refined =
                solveWeightedAffine(
                    samples = samples,
                    target = target,
                    weights = robustWeights,
                ) ?: return@repeat

            val coefficientDelta =
                sqrt(
                    refined.indices.sumOf { index ->
                        val delta =
                            refined[index] -
                                coefficients[index]
                        delta * delta
                    },
                )

            coefficients = refined

            if (coefficientDelta < 1e-6) {
                return@repeat
            }
        }

        return coefficients
    }

    private fun solveWeightedAffine(
        samples: List<EyeGazeCalibrationSample>,
        target: (EyeGazeCalibrationSample) -> Double,
        weights: List<Double>,
    ): DoubleArray? {
        if (samples.size != weights.size) return null

        val normal = Array(3) { DoubleArray(4) }

        samples.forEachIndexed { index, sample ->
            val x = sample.offset.x
            val y = sample.offset.y
            val w =
                weights[index]
                    .coerceIn(1e-6, 1.0)
            val features =
                doubleArrayOf(x, y, 1.0)
            val t = target(sample)

            for (row in 0 until 3) {
                for (col in 0 until 3) {
                    normal[row][col] +=
                        w *
                            features[row] *
                            features[col]
                }
                normal[row][3] +=
                    w *
                        features[row] *
                        t
            }
        }

        return solve3(normal)
    }

    private fun solve3(matrix: Array<DoubleArray>): DoubleArray? {
        val a = Array(3) { row -> matrix[row].copyOf() }

        for (column in 0 until 3) {
            var pivot = column
            for (row in column + 1 until 3) {
                if (kotlin.math.abs(a[row][column]) >
                    kotlin.math.abs(a[pivot][column])
                ) {
                    pivot = row
                }
            }

            if (kotlin.math.abs(a[pivot][column]) < 1e-10) {
                return null
            }

            val temp = a[column]
            a[column] = a[pivot]
            a[pivot] = temp

            val divisor = a[column][column]
            for (c in column until 4) {
                a[column][c] /= divisor
            }

            for (row in 0 until 3) {
                if (row == column) continue

                val factor = a[row][column]
                for (c in column until 4) {
                    a[row][c] -= factor * a[column][c]
                }
            }
        }

        return doubleArrayOf(
            a[0][3],
            a[1][3],
            a[2][3],
        )
    }
}
