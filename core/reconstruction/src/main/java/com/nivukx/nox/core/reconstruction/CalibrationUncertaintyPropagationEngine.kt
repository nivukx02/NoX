package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.math.sqrt

/**
 * Propagates per-camera calibration uncertainty into a world-space 3D covariance envelope.
 *
 * Translation error contributes isotropically. Small rotational error is propagated through the
 * skew-symmetric Jacobian of the point relative to the camera center, so distant points correctly
 * become more sensitive to angular calibration error. Intrinsic uncertainty is represented as a
 * conservative range-scaled isotropic term because the model does not currently expose a full
 * intrinsic-parameter covariance matrix.
 *
 * The engine intentionally returns an envelope for one camera at a time. A caller that lacks a
 * cross-camera calibration covariance model can then select the dominant envelope without
 * incorrectly reducing shared/systematic calibration error by sqrt(N).
 */
class CalibrationUncertaintyPropagationEngine {

    data class Result(
        val covariance: Array<DoubleArray>,
        val equivalentSigmaMeters: Double,
        val radialSigmaMeters: Double,
        val rotationalSigmaMeters: Double,
        val intrinsicSigmaMeters: Double,
        val confidence: Double,
        val applied: Boolean,
    )

    fun propagate(
        pointWorld: Vec3,
        cameraPose: Pose,
        profile: CalibrationUncertaintyProfile,
    ): Result {
        if (!profile.hasEvidence) {
            return Result(
                covariance = zeroMatrix(),
                equivalentSigmaMeters = 0.0,
                radialSigmaMeters = 0.0,
                rotationalSigmaMeters = 0.0,
                intrinsicSigmaMeters = 0.0,
                confidence = 0.0,
                applied = false,
            )
        }

        val evidenceConfidence =
            profile.evidenceConfidence
                .coerceIn(0.05, 1.0)

        /*
         * A low-confidence uncertainty estimate should not become artificially optimistic. We
         * inflate the reported sigmas gently as evidence confidence falls, capped to avoid an
         * explosive covariance when a calibration report is only partially qualified.
         */
        val confidenceInflation =
            1.0 / sqrt(
                evidenceConfidence
                    .coerceAtLeast(0.25),
            )

        val translationSigma =
            profile.translationSigmaMeters *
                confidenceInflation

        val rotationSigma =
            profile.rotationSigmaRadians *
                confidenceInflation

        val intrinsicSigmaFraction =
            profile.intrinsicRelativeSigma *
                confidenceInflation

        val relative =
            pointWorld - cameraPose.position

        val range =
            relative.norm()
                .coerceAtLeast(0.0)

        val radialSigma =
            translationSigma

        val rotationalSigma =
            rotationSigma * range

        /*
         * Intrinsic uncertainty is conservative and range-scaled. It represents an image-to-ray
         * scale uncertainty mapped into world-space distance; it is not a claim that every
         * intrinsic parameter perturbs the point isotropically.
         */
        val intrinsicSigma =
            intrinsicSigmaFraction * range

        val translationCovariance =
            isotropic(
                radialSigma * radialSigma,
            )

        val rotationCovariance =
            rotationCovariance(
                relative = relative,
                sigmaRadians = rotationSigma,
            )

        val intrinsicCovariance =
            isotropic(
                intrinsicSigma * intrinsicSigma,
            )

        val covariance =
            add(
                add(
                    translationCovariance,
                    rotationCovariance,
                ),
                intrinsicCovariance,
            )

        return Result(
            covariance = covariance,
            equivalentSigmaMeters =
                sqrt(
                    trace(covariance) /
                        3.0,
                ).coerceAtLeast(0.0),
            radialSigmaMeters = radialSigma,
            rotationalSigmaMeters = rotationalSigma,
            intrinsicSigmaMeters = intrinsicSigma,
            confidence = profile.evidenceConfidence
                .coerceIn(0.0, 1.0),
            applied = true,
        )
    }

    private fun rotationCovariance(
        relative: Vec3,
        sigmaRadians: Double,
    ): Array<DoubleArray> {
        val x = relative.x
        val y = relative.y
        val z = relative.z

        /*
         * J = -skew(relative). For isotropic angular covariance sigma^2 I:
         *   C = J (sigma^2 I) J^T
         */
        val variance = sigmaRadians * sigmaRadians

        return arrayOf(
            doubleArrayOf(
                (y * y + z * z) * variance,
                (-x * y) * variance,
                (-x * z) * variance,
            ),
            doubleArrayOf(
                (-x * y) * variance,
                (x * x + z * z) * variance,
                (-y * z) * variance,
            ),
            doubleArrayOf(
                (-x * z) * variance,
                (-y * z) * variance,
                (x * x + y * y) * variance,
            ),
        )
    }

    private fun isotropic(
        variance: Double,
    ): Array<DoubleArray> =
        arrayOf(
            doubleArrayOf(variance, 0.0, 0.0),
            doubleArrayOf(0.0, variance, 0.0),
            doubleArrayOf(0.0, 0.0, variance),
        )

    private fun add(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> =
        Array(3) { row ->
            DoubleArray(3) { column ->
                a[row][column] + b[row][column]
            }
        }

    private fun trace(
        matrix: Array<DoubleArray>,
    ): Double =
        matrix[0][0] +
            matrix[1][1] +
            matrix[2][2]

    private fun zeroMatrix(): Array<DoubleArray> =
        Array(3) {
            DoubleArray(3)
        }
}
