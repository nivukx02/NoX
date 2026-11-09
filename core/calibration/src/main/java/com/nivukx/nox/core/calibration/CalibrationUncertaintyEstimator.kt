package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Conservative calibration uncertainty envelope estimator.
 *
 * This is not a replacement for a full joint camera covariance matrix. It converts observable
 * calibration-stage quantities into a bounded per-camera envelope that downstream reconstruction
 * can combine with image measurement uncertainty. The estimator intentionally inflates rather than
 * invents precision when calibration evidence is sparse.
 */
class CalibrationUncertaintyEstimator(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumTranslationSigmaMeters: Double = 0.00005,
        val maximumTranslationSigmaMeters: Double = 0.050,
        val minimumRotationSigmaRadians: Double = 1e-5,
        val maximumRotationSigmaRadians: Double = 0.10,
        val minimumIntrinsicRelativeSigma: Double = 1e-5,
        val maximumIntrinsicRelativeSigma: Double = 0.05,
        val residualToPixelSigmaMultiplier: Double = 1.35,
        val landmarkSupportFloor: Int = 12,
        val baselineFloorMeters: Double = 0.25,
        val evidenceFloor: Double = 0.15,
    ) {
        init {
            require(minimumTranslationSigmaMeters > 0.0)
            require(maximumTranslationSigmaMeters > minimumTranslationSigmaMeters)
            require(minimumRotationSigmaRadians > 0.0)
            require(maximumRotationSigmaRadians > minimumRotationSigmaRadians)
            require(minimumIntrinsicRelativeSigma > 0.0)
            require(maximumIntrinsicRelativeSigma > minimumIntrinsicRelativeSigma)
            require(residualToPixelSigmaMultiplier >= 1.0)
            require(landmarkSupportFloor >= 4)
            require(baselineFloorMeters > 0.0)
            require(evidenceFloor in 0.0..1.0)
        }
    }

    data class CameraEvidence(
        val cameraId: String,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val landmarkCount: Int,
        val finalReprojectionRmsPx: Double,
        val medianSceneDepthMeters: Double,
        val maximumBaselineMeters: Double,
        val cameraParticipationRatio: Double,
        val calibrated: Boolean = true,
    )

    fun estimate(evidence: CameraEvidence): CalibrationUncertaintyProfile {
        require(evidence.cameraId.isNotBlank())
        require(evidence.landmarkCount >= 0)
        require(evidence.finalReprojectionRmsPx.isFinite() &&
            evidence.finalReprojectionRmsPx >= 0.0)
        require(evidence.medianSceneDepthMeters.isFinite() &&
            evidence.medianSceneDepthMeters >= 0.0)
        require(evidence.maximumBaselineMeters.isFinite() &&
            evidence.maximumBaselineMeters >= 0.0)
        require(evidence.cameraParticipationRatio in 0.0..1.0)

        if (!evidence.calibrated || evidence.landmarkCount < 2) {
            return CalibrationUncertaintyProfile()
        }

        val focal =
            sqrt(
                (
                    evidence.intrinsics.fx *
                        evidence.intrinsics.fy
                ).coerceAtLeast(1.0),
            )

        val baseline =
            evidence.maximumBaselineMeters
                .coerceAtLeast(config.baselineFloorMeters)
        val depth =
            evidence.medianSceneDepthMeters
                .coerceAtLeast(0.5)
        val residualSigmaPx =
            (
                max(
                    evidence.finalReprojectionRmsPx,
                    0.25,
                ) *
                    config.residualToPixelSigmaMultiplier
                ).coerceAtLeast(0.25)

        /*
         * First-order triangulation sensitivity approximates depth error as Z²/(fB) * pixel error.
         * Translation is assigned a conservative fraction of that scale and then inflated when a
         * camera contributed to fewer landmarks.
         */
        val depthSigmaMeters =
            depth * depth /
                (focal * baseline) *
                residualSigmaPx
        val supportInflation =
            (
                config.landmarkSupportFloor.toDouble() /
                    max(
                        config.landmarkSupportFloor,
                        evidence.landmarkCount,
                    ).toDouble()
                ).coerceIn(0.35, 1.0)
        val participationInflation =
            (
                1.0 +
                    (
                        1.0 -
                            evidence.cameraParticipationRatio
                    ) * 1.5
                ).coerceIn(1.0, 2.5)

        val translationSigma =
            (
                max(
                    depthSigmaMeters * 0.45,
                    config.minimumTranslationSigmaMeters,
                ) *
                    (1.0 + supportInflation) *
                    participationInflation
                ).coerceIn(
                config.minimumTranslationSigmaMeters,
                config.maximumTranslationSigmaMeters,
            )

        val rotationSigma =
            (
                translationSigma /
                    baseline
                ).coerceIn(
                config.minimumRotationSigmaRadians,
                config.maximumRotationSigmaRadians,
            )

        /*
         * Intrinsic uncertainty is deliberately weaker than pose uncertainty because this
         * optimizer keeps sensor intrinsics fixed. A modest residual-derived proxy tells the
         * downstream solver not to treat those intrinsics as exact without manufacturer uncertainty
         * metadata, while never claiming a calibrated covariance was measured.
         */
        val intrinsicRelativeSigma =
            (
                residualSigmaPx /
                    focal *
                    (0.75 + 0.75 * supportInflation)
                ).coerceIn(
                config.minimumIntrinsicRelativeSigma,
                config.maximumIntrinsicRelativeSigma,
            )

        val residualConfidence =
            (
                1.0 -
                    evidence.finalReprojectionRmsPx / 4.0
                ).coerceIn(0.0, 1.0)
        val supportConfidence =
            (
                evidence.landmarkCount.toDouble() /
                    96.0
                ).coerceIn(0.0, 1.0)
        val geometryConfidence =
            (
                evidence.maximumBaselineMeters /
                    2.0
                ).coerceIn(0.0, 1.0)
        val participationConfidence =
            evidence.cameraParticipationRatio

        val evidenceConfidence =
            (
                residualConfidence * 0.36 +
                    supportConfidence * 0.22 +
                    geometryConfidence * 0.20 +
                    participationConfidence * 0.22
                ).coerceIn(
                config.evidenceFloor,
                1.0,
            )

        return CalibrationUncertaintyProfile(
            translationSigmaMeters = translationSigma,
            rotationSigmaRadians = rotationSigma,
            intrinsicRelativeSigma = intrinsicRelativeSigma,
            evidenceConfidence = evidenceConfidence,
        )
    }

    fun maximumBaseline(
        referencePose: Pose,
        poses: Collection<Pose>,
    ): Double =
        poses.maxOfOrNull {
            referencePose.position.distanceTo(
                it.position,
            )
        } ?: 0.0

    fun medianDepth(
        points: Collection<Vec3>,
    ): Double {
        val values = points
            .map { it.z }
            .filter { it.isFinite() && it > 0.0 }
            .sorted()
        if (values.isEmpty()) return 0.0
        val position = (values.size - 1) * 0.50
        val lower = position.toInt()
        val upper = min(
            lower + 1,
            values.lastIndex,
        )
        val fraction = position - lower
        return values[lower] * (1.0 - fraction) +
            values[upper] * fraction
    }
}
