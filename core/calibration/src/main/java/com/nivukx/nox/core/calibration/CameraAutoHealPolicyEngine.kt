package com.nivukx.nox.core.calibration

import kotlin.math.min

/**
 * Camera-health recovery policy.
 *
 * This engine never mutates calibration on its own. It determines whether a camera is safe to
 * keep, eligible for bounded auto-healing, or requires operator/calibration intervention.
 */
class CameraAutoHealPolicyEngine(
    private val config: Config = Config(),
) {

    enum class Action {
        KEEP,
        HEAL_CANDIDATE,
        RECALIBRATION_REQUIRED,
        HOLD,
    }

    data class Config(
        val imageErrorHealThresholdPx: Double = 2.5,
        val imageErrorRecalibrationThresholdPx: Double = 6.0,
        val minimumConnectivityForHeal: Double = 0.50,
        val minimumCoverageForHeal: Double = 0.55,
        val minimumCalibrationConfidence: Double = 0.70,
        val criticalCalibrationConfidence: Double = 0.45,
        val temperatureDeltaHealCelsius: Double = 3.0,
        val temperatureDeltaHoldCelsius: Double = 7.0,
        val minimumDriftStability: Double = 0.55,
        val criticalDriftStability: Double = 0.35,
        val maximumCalibrationAgeSeconds: Double = 21_600.0,
        val maximumAutomaticHealingAgeSeconds: Double = 3_600.0,
        val minimumConfidenceToKeep: Double = 0.72,
    ) {
        init {
            require(imageErrorHealThresholdPx > 0.0)
            require(
                imageErrorRecalibrationThresholdPx >
                    imageErrorHealThresholdPx,
            )
            require(minimumConnectivityForHeal in 0.0..1.0)
            require(minimumCoverageForHeal in 0.0..1.0)
            require(minimumCalibrationConfidence in 0.0..1.0)
            require(
                criticalCalibrationConfidence <
                    minimumCalibrationConfidence,
            )
            require(temperatureDeltaHealCelsius >= 0.0)
            require(
                temperatureDeltaHoldCelsius >
                    temperatureDeltaHealCelsius,
            )
            require(minimumDriftStability in 0.0..1.0)
            require(
                criticalDriftStability <
                    minimumDriftStability,
            )
            require(maximumCalibrationAgeSeconds > 0.0)
            require(
                maximumAutomaticHealingAgeSeconds <=
                    maximumCalibrationAgeSeconds,
            )
            require(minimumConfidenceToKeep in 0.0..1.0)
        }
    }

    data class Input(
        val cameraId: String,
        val calibrationConfidence: Double,
        val imageErrorPx: Double,
        val centroidConnectivity: Double,
        val temperatureDeltaCelsius: Double,
        val calibrationDriftStability: Double,
        val trackedObjectCoverage: Double,
        val calibrationAgeSeconds: Double,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(
                calibrationConfidence.isFinite() &&
                    calibrationConfidence in 0.0..1.0,
            )
            require(
                imageErrorPx.isFinite() &&
                    imageErrorPx >= 0.0,
            )
            require(
                centroidConnectivity.isFinite() &&
                    centroidConnectivity in 0.0..1.0,
            )
            require(
                temperatureDeltaCelsius.isFinite() &&
                    temperatureDeltaCelsius >= 0.0,
            )
            require(
                calibrationDriftStability.isFinite() &&
                    calibrationDriftStability in 0.0..1.0,
            )
            require(
                trackedObjectCoverage.isFinite() &&
                    trackedObjectCoverage in 0.0..1.0,
            )
            require(
                calibrationAgeSeconds.isFinite() &&
                    calibrationAgeSeconds >= 0.0,
            )
        }
    }

    data class Result(
        val cameraId: String,
        val action: Action,
        val confidence: Double,
        val reasons: List<String>,
        val recommendedCooldownSeconds: Double,
        val allowSilentHealing: Boolean,
        val requiresOperatorAttention: Boolean,
    )

    fun evaluate(input: Input): Result {
        val critical =
            input.calibrationConfidence <
                config.criticalCalibrationConfidence ||
                input.calibrationDriftStability <
                config.criticalDriftStability ||
                input.temperatureDeltaCelsius >=
                config.temperatureDeltaHoldCelsius ||
                input.trackedObjectCoverage <
                config.minimumCoverageForHeal * 0.65

        if (critical) {
            return Result(
                cameraId = input.cameraId,
                action =
                    if (
                        input.temperatureDeltaCelsius >=
                            config.temperatureDeltaHoldCelsius
                    ) {
                        Action.HOLD
                    } else {
                        Action.RECALIBRATION_REQUIRED
                    },
                confidence = 0.0,
                reasons = buildList {
                    if (
                        input.calibrationConfidence <
                        config.criticalCalibrationConfidence
                    ) {
                        add("calibration-confidence-critical")
                    }
                    if (
                        input.calibrationDriftStability <
                        config.criticalDriftStability
                    ) {
                        add("calibration-drift-critical")
                    }
                    if (
                        input.temperatureDeltaCelsius >=
                        config.temperatureDeltaHoldCelsius
                    ) {
                        add("temperature-drift-critical")
                    }
                    if (
                        input.trackedObjectCoverage <
                        config.minimumCoverageForHeal * 0.65
                    ) {
                        add("tracking-coverage-critical")
                    }
                },
                recommendedCooldownSeconds = 60.0,
                allowSilentHealing = false,
                requiresOperatorAttention = true,
            )
        }

        val reliability =
            (
                input.calibrationConfidence * 0.28 +
                    (
                        1.0 -
                            min(
                                1.0,
                                input.imageErrorPx /
                                    config.imageErrorRecalibrationThresholdPx,
                            )
                        ) * 0.24 +
                    input.centroidConnectivity * 0.18 +
                    input.calibrationDriftStability * 0.12 +
                    input.trackedObjectCoverage * 0.10 +
                    (
                        1.0 -
                            min(
                                1.0,
                                input.temperatureDeltaCelsius /
                                    config.temperatureDeltaHoldCelsius,
                            )
                        ) * 0.08
                ).coerceIn(0.0, 1.0)

        val agePenalty =
            if (
                input.calibrationAgeSeconds >
                config.maximumAutomaticHealingAgeSeconds
            ) {
                min(
                    0.20,
                    (
                        input.calibrationAgeSeconds -
                            config.maximumAutomaticHealingAgeSeconds
                        ) /
                        config.maximumCalibrationAgeSeconds *
                        0.20,
                )
            } else {
                0.0
            }

        val confidence =
            (reliability - agePenalty)
                .coerceIn(0.0, 1.0)

        val healSignal =
            input.imageErrorPx >=
                config.imageErrorHealThresholdPx ||
                input.centroidConnectivity <
                config.minimumConnectivityForHeal ||
                input.trackedObjectCoverage <
                config.minimumCoverageForHeal ||
                input.temperatureDeltaCelsius >
                config.temperatureDeltaHealCelsius ||
                input.calibrationDriftStability <
                config.minimumDriftStability

        val severeSignal =
            input.imageErrorPx >=
                config.imageErrorRecalibrationThresholdPx ||
                input.calibrationConfidence <
                config.minimumCalibrationConfidence

        val staleSignal =
            input.calibrationAgeSeconds >
                config.maximumCalibrationAgeSeconds

        val action =
            when {
                staleSignal || severeSignal ->
                    Action.RECALIBRATION_REQUIRED
                healSignal ->
                    Action.HEAL_CANDIDATE
                confidence >= config.minimumConfidenceToKeep ->
                    Action.KEEP
                else ->
                    Action.HEAL_CANDIDATE
            }

        val reasons = mutableListOf<String>()

        if (
            input.imageErrorPx >=
            config.imageErrorHealThresholdPx
        ) {
            reasons += "image-error-elevated"
        }
        if (
            input.imageErrorPx >=
            config.imageErrorRecalibrationThresholdPx
        ) {
            reasons += "image-error-recalibration"
        }
        if (
            input.centroidConnectivity <
            config.minimumConnectivityForHeal
        ) {
            reasons += "centroid-connectivity-low"
        }
        if (
            input.trackedObjectCoverage <
            config.minimumCoverageForHeal
        ) {
            reasons += "tracking-coverage-low"
        }
        if (
            input.temperatureDeltaCelsius >
            config.temperatureDeltaHealCelsius
        ) {
            reasons += "temperature-drift"
        }
        if (
            input.calibrationDriftStability <
            config.minimumDriftStability
        ) {
            reasons += "calibration-drift"
        }
        if (
            input.calibrationAgeSeconds >
            config.maximumAutomaticHealingAgeSeconds
        ) {
            reasons += "calibration-age-high"
        }
        if (reasons.isEmpty()) {
            reasons += "camera-health-stable"
        }

        return Result(
            cameraId = input.cameraId,
            action = action,
            confidence = confidence,
            reasons = reasons.distinct(),
            recommendedCooldownSeconds =
                if (action == Action.HEAL_CANDIDATE) {
                    15.0
                } else {
                    60.0
                },
            allowSilentHealing =
                action == Action.HEAL_CANDIDATE &&
                    confidence >= 0.72 &&
                    input.trackedObjectCoverage >=
                    config.minimumCoverageForHeal &&
                    input.centroidConnectivity >=
                    config.minimumConnectivityForHeal &&
                    input.temperatureDeltaCelsius <=
                    config.temperatureDeltaHealCelsius &&
                    input.imageErrorPx <
                    config.imageErrorHealThresholdPx &&
                    input.calibrationDriftStability >=
                    config.minimumDriftStability &&
                    input.calibrationAgeSeconds <=
                    config.maximumAutomaticHealingAgeSeconds,
            requiresOperatorAttention =
                action == Action.RECALIBRATION_REQUIRED ||
                    action == Action.HOLD,
        )
    }
}
