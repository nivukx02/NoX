package com.nivukx.nox.core.ai

import kotlin.math.max

/**
 * Converts measured motion quality into bounded temporal-policy multipliers.
 *
 * The controller does not replace user tuning. It only adjusts how strongly the tracker trusts
 * prediction, how elastic innovation gates are, and how aggressively stale identities are
 * reacquired for the next frame.
 */
class MotionRegimeAdaptiveController {

    enum class Regime {
        STEADY,
        MICRO_DETAIL,
        PRECISION,
        NORMAL,
        FAST_ACTION,
        EXTREME_ACTION,
        CAMERA_DOMINANT,
        OCCLUDED_RECOVERY,
    }

    data class Policy(
        val regime: Regime,
        val innovationGateScale: Double = 1.0,
        val velocityResponsivenessScale: Double = 1.0,
        val accelerationResponsivenessScale: Double = 1.0,
        val predictionHorizonScale: Double = 1.0,
        val reacquisitionScale: Double = 1.0,
        val jitterSuppressionScale: Double = 1.0,
        val reason: String,
    ) {
        init {
            require(innovationGateScale in 0.60..1.80)
            require(velocityResponsivenessScale in 0.55..1.45)
            require(accelerationResponsivenessScale in 0.55..1.45)
            require(predictionHorizonScale in 0.50..1.35)
            require(reacquisitionScale in 0.60..1.55)
            require(jitterSuppressionScale in 0.55..1.40)
        }
    }

    fun resolve(
        quality: MotionCaptureEngine.Quality,
        context: MotionCaptureEngine.MotionContext,
    ): Policy {
        val occlusion = quality.occlusionRatio.coerceIn(0.0, 1.0)
        val jitter = quality.jitterNormalized.coerceIn(0.0, 1.0)
        val vibration = quality.vibrationRisk.coerceIn(0.0, 1.0)
        val spectralConfidence =
            quality.spectralConfidence.coerceIn(0.0, 1.0)
        val cameraDominance =
            context.cameraDominance.coerceIn(0.0, 1.0)

        if (occlusion >= 0.62) {
            return Policy(
                regime = Regime.OCCLUDED_RECOVERY,
                innovationGateScale = 1.18,
                velocityResponsivenessScale = 0.88,
                accelerationResponsivenessScale = 0.86,
                predictionHorizonScale = 1.18,
                reacquisitionScale = 1.30,
                jitterSuppressionScale = 0.96,
                reason = "occlusion recovery",
            )
        }

        if (cameraDominance >= 0.68) {
            return Policy(
                regime = Regime.CAMERA_DOMINANT,
                innovationGateScale =
                    (1.06 + 0.28 * cameraDominance)
                        .coerceIn(1.06, 1.36),
                velocityResponsivenessScale =
                    (0.92 - 0.22 * cameraDominance)
                        .coerceIn(0.70, 0.92),
                accelerationResponsivenessScale =
                    (0.94 - 0.24 * cameraDominance)
                        .coerceIn(0.70, 0.94),
                predictionHorizonScale =
                    (0.92 - 0.18 * cameraDominance)
                        .coerceIn(0.74, 0.92),
                reacquisitionScale =
                    (1.00 + 0.18 * cameraDominance)
                        .coerceIn(1.00, 1.18),
                jitterSuppressionScale =
                    (1.04 + 0.16 * cameraDominance)
                        .coerceIn(1.04, 1.20),
                reason = "camera-dominant motion",
            )
        }

        if (
            quality.motionClass == MotionCaptureEngine.MotionClass.EXTREME
        ) {
            return Policy(
                regime = Regime.EXTREME_ACTION,
                innovationGateScale = 1.20,
                velocityResponsivenessScale = 1.28,
                accelerationResponsivenessScale = 1.32,
                predictionHorizonScale = 0.72,
                reacquisitionScale = 1.16,
                jitterSuppressionScale = 0.82,
                reason = "extreme subject motion",
            )
        }

        if (
            quality.motionClass == MotionCaptureEngine.MotionClass.FAST
        ) {
            return Policy(
                regime = Regime.FAST_ACTION,
                innovationGateScale = 1.12,
                velocityResponsivenessScale = 1.20,
                accelerationResponsivenessScale = 1.24,
                predictionHorizonScale = 0.82,
                reacquisitionScale = 1.12,
                jitterSuppressionScale = 0.86,
                reason = "fast subject motion",
            )
        }

        if (
            quality.motionClass == MotionCaptureEngine.MotionClass.MICRO_MOTION &&
            quality.microMotionScore >= 0.35
        ) {
            return Policy(
                regime = Regime.MICRO_DETAIL,
                innovationGateScale = 0.90,
                velocityResponsivenessScale = 0.88,
                accelerationResponsivenessScale = 0.84,
                predictionHorizonScale = 0.84,
                reacquisitionScale = 0.82,
                jitterSuppressionScale = 1.16,
                reason = "micro-motion preservation",
            )
        }

        if (
            quality.motionClass == MotionCaptureEngine.MotionClass.STATIC &&
            jitter < 0.18 &&
            (
                spectralConfidence < 0.45 ||
                    vibration < 0.30
                )
        ) {
            return Policy(
                regime = Regime.STEADY,
                innovationGateScale = 0.86,
                velocityResponsivenessScale = 0.72,
                accelerationResponsivenessScale = 0.68,
                predictionHorizonScale = 0.72,
                reacquisitionScale = 0.76,
                jitterSuppressionScale = 1.20,
                reason = "stable subject",
            )
        }

        val precisionPressure =
            max(
                jitter * 0.55,
                vibration * 0.45,
            )

        return Policy(
            regime =
                if (precisionPressure >= 0.42) {
                    Regime.PRECISION
                } else {
                    Regime.NORMAL
                },
            innovationGateScale =
                (
                    0.96 +
                        precisionPressure * 0.16
                ).coerceIn(0.90, 1.16),
            velocityResponsivenessScale =
                (
                    1.00 -
                        precisionPressure * 0.14
                ).coerceIn(0.82, 1.0),
            accelerationResponsivenessScale =
                (
                    1.00 -
                        precisionPressure * 0.16
                ).coerceIn(0.80, 1.0),
            predictionHorizonScale =
                (
                    1.0 -
                        precisionPressure * 0.10
                ).coerceIn(0.86, 1.0),
            reacquisitionScale =
                (
                    1.0 +
                        precisionPressure * 0.10
                ).coerceIn(1.0, 1.10),
            jitterSuppressionScale =
                (
                    1.0 +
                        precisionPressure * 0.12
                ).coerceIn(1.0, 1.12),
            reason =
                if (precisionPressure >= 0.42) {
                    "precision protection"
                } else {
                    "balanced motion"
                },
        )
    }

    fun identityPolicy(
        reason: String = "adaptive disabled",
    ): Policy =
        Policy(
            regime = Regime.NORMAL,
            reason = reason,
        )

    fun blend(
        current: Policy,
        target: Policy,
        alpha: Double,
    ): Policy {
        val t = alpha.coerceIn(0.0, 1.0)

        fun mix(a: Double, b: Double) =
            a * (1.0 - t) + b * t

        return Policy(
            regime = if (t < 0.5) current.regime else target.regime,
            innovationGateScale =
                mix(
                    current.innovationGateScale,
                    target.innovationGateScale,
                ).coerceIn(0.60, 1.80),
            velocityResponsivenessScale =
                mix(
                    current.velocityResponsivenessScale,
                    target.velocityResponsivenessScale,
                ).coerceIn(0.55, 1.45),
            accelerationResponsivenessScale =
                mix(
                    current.accelerationResponsivenessScale,
                    target.accelerationResponsivenessScale,
                ).coerceIn(0.55, 1.45),
            predictionHorizonScale =
                mix(
                    current.predictionHorizonScale,
                    target.predictionHorizonScale,
                ).coerceIn(0.50, 1.35),
            reacquisitionScale =
                mix(
                    current.reacquisitionScale,
                    target.reacquisitionScale,
                ).coerceIn(0.60, 1.55),
            jitterSuppressionScale =
                mix(
                    current.jitterSuppressionScale,
                    target.jitterSuppressionScale,
                ).coerceIn(0.55, 1.40),
            reason = target.reason,
        )
    }
}
