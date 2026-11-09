package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile

/**
 * Conservative policy defaults for calibrated rays whose uncertainty metadata is incomplete.
 *
 * These are safety priors used to reduce authority, not measured properties of any camera.
 * A zero sigma in [CalibrationUncertaintyProfile] explicitly means that an estimate was not
 * supplied; it must never be interpreted as zero physical uncertainty.
 */
internal const val UNKNOWN_CALIBRATION_TRANSLATION_SIGMA_METERS = 0.05
internal const val UNKNOWN_CALIBRATION_ROTATION_SIGMA_RADIANS = 0.035
internal const val UNKNOWN_CALIBRATION_INTRINSIC_RELATIVE_SIGMA = 0.05
internal const val UNKNOWN_CALIBRATION_EVIDENCE_CONFIDENCE = 0.35

internal data class CalibratedRayUncertaintyResolution(
    val translationSigmaMeters: Double,
    val rotationPenalty: Double,
    val intrinsicPenalty: Double,
    val evidenceConfidence: Double,
)

/**
 * Resolves missing uncertainty fields to conservative fallback priors and caps penalties.
 *
 * The original measured field is retained whenever it is positive. Missing fields are resolved
 * independently because a calibration may have a translation estimate but no rotation estimate,
 * or vice versa.
 */
internal fun resolveCalibratedRayUncertainty(
    profile: CalibrationUncertaintyProfile,
): CalibratedRayUncertaintyResolution {
    val translationSigma =
        profile.translationSigmaMeters
            .takeIf { it > 0.0 }
            ?: UNKNOWN_CALIBRATION_TRANSLATION_SIGMA_METERS
    val rotationSigma =
        profile.rotationSigmaRadians
            .takeIf { it > 0.0 }
            ?: UNKNOWN_CALIBRATION_ROTATION_SIGMA_RADIANS
    val intrinsicSigma =
        profile.intrinsicRelativeSigma
            .takeIf { it > 0.0 }
            ?: UNKNOWN_CALIBRATION_INTRINSIC_RELATIVE_SIGMA
    val confidence =
        profile.evidenceConfidence
            .takeIf { it > 0.0 }
            ?: UNKNOWN_CALIBRATION_EVIDENCE_CONFIDENCE

    return CalibratedRayUncertaintyResolution(
        translationSigmaMeters = translationSigma,
        rotationPenalty = (rotationSigma / 0.010).coerceIn(0.0, 2.0),
        intrinsicPenalty = (intrinsicSigma / 0.010).coerceIn(0.0, 2.0),
        evidenceConfidence = confidence.coerceIn(0.0, 1.0),
    )
}
