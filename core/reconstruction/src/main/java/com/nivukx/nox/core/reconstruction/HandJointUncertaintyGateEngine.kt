package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Joint-wise, view-dependent reliability model for multi-view hand reconstruction.
 *
 * Each 2D hand observation is treated as an independent view expert. The model does not assume
 * that every camera contributes equally to every finger joint. It combines detector confidence,
 * hand confidence, temporal stability, prediction state, image-boundary proximity and landmark
 * consistency into an explicit uncertainty estimate.
 *
 * The output is intentionally interpretable:
 * - reliability is the fusion weight contribution;
 * - uncertainty is a bounded meters-equivalent proxy used to penalize weak evidence;
 * - predicted/unstable/border observations are suppressed rather than allowed to contaminate
 *   otherwise strong cross-view geometry.
 *
 * This is a policy and evidence layer. It does not replace the geometric triangulator.
 */
class HandJointUncertaintyGateEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val minimumInputConfidence: Double = 0.05,
        val uncertaintyFloorMeters: Double = 0.0015,
        val uncertaintyCeilingMeters: Double = 0.25,
        val confidenceExponent: Double = 1.55,
        val temporalExponent: Double = 0.85,
        val handExponent: Double = 0.55,
        val borderPenaltyStrength: Double = 0.45,
        val predictedPenaltyStrength: Double = 0.55,
        val unstablePenaltyStrength: Double = 0.35,
        val minimumFusionWeight: Double = 0.01,
        val robustOutlierWeightFloor: Double = 0.06,
        val fingerTipExtraSensitivity: Double = 0.10,
        val wristExtraSensitivity: Double = 0.05,
        val expectedImageMargin: Double = 0.035,
        val hardBorderMargin: Double = 0.008,
        val maximumVelocityMetersPerSecond: Double = 12.0,
    ) {
        init {
            require(minimumInputConfidence in 0.0..1.0)
            require(uncertaintyFloorMeters > 0.0)
            require(uncertaintyCeilingMeters >= uncertaintyFloorMeters)
            require(confidenceExponent > 0.0)
            require(temporalExponent > 0.0)
            require(handExponent > 0.0)
            require(borderPenaltyStrength in 0.0..1.0)
            require(predictedPenaltyStrength in 0.0..1.0)
            require(unstablePenaltyStrength in 0.0..1.0)
            require(minimumFusionWeight in 0.0..1.0)
            require(robustOutlierWeightFloor in 0.0..1.0)
            require(fingerTipExtraSensitivity >= 0.0)
            require(wristExtraSensitivity >= 0.0)
            require(expectedImageMargin in 0.0..0.25)
            require(hardBorderMargin in 0.0..expectedImageMargin)
            require(maximumVelocityMetersPerSecond > 0.0)
        }
    }

    data class Input(
        val jointIndex: Int,
        val normalizedX: Double,
        val normalizedY: Double,
        val landmarkConfidence: Double,
        val handConfidence: Double,
        val handednessConfidence: Double,
        val temporalStability: Double,
        val predicted: Boolean,
        val wristVelocityMetersPerSecond: Double,
        val observationFreshness: Double = 1.0,
    ) {
        init {
            require(jointIndex in 0..20)
            require(normalizedX.isFinite())
            require(normalizedY.isFinite())
            listOf(
                landmarkConfidence,
                handConfidence,
                handednessConfidence,
                temporalStability,
                observationFreshness,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(wristVelocityMetersPerSecond.isFinite())
        }
    }

    data class Result(
        val reliability: Double,
        val uncertaintyMeters: Double,
        val borderConfidence: Double,
        val motionConfidence: Double,
        val temporalConfidence: Double,
        val viewExpertWeight: Double,
        val predictedSuppression: Double,
        val hardRejected: Boolean,
        val reasons: List<String>,
    ) {
        val accepted: Boolean
            get() = !hardRejected && viewExpertWeight > 0.0
    }

    @Synchronized
    fun configure(newConfig: Config) {
        config = newConfig
    }

    fun configuration(): Config = config

    fun evaluate(input: Input): Result {
        val cfg = config
        val reasons = mutableListOf<String>()

        if (input.landmarkConfidence < cfg.minimumInputConfidence) {
            reasons += "landmark-confidence-below-floor"
        }

        val borderConfidence =
            borderConfidence(
                x = input.normalizedX,
                y = input.normalizedY,
                expectedMargin = cfg.expectedImageMargin,
                hardMargin = cfg.hardBorderMargin,
            )

        if (borderConfidence < 0.30) {
            reasons += "image-boundary-risk"
        }

        val temporalConfidence =
            (
                input.temporalStability.coerceIn(0.0, 1.0) *
                    input.observationFreshness.coerceIn(0.0, 1.0)
            ).powSafe(cfg.temporalExponent)

        if (temporalConfidence < 0.35) {
            reasons += "temporal-instability"
        }

        val motionMagnitude =
            abs(input.wristVelocityMetersPerSecond)
                .coerceIn(0.0, cfg.maximumVelocityMetersPerSecond)
        val motionConfidence =
            (1.0 - motionMagnitude /
                cfg.maximumVelocityMetersPerSecond)
                .coerceIn(0.0, 1.0)

        if (motionConfidence < 0.20) {
            reasons += "high-motion-risk"
        }

        val detectorConfidence =
            input.landmarkConfidence
                .coerceIn(0.0, 1.0)
                .powSafe(cfg.confidenceExponent)

        val handConfidence =
            (
                input.handConfidence *
                    (0.55 + 0.45 * input.handednessConfidence)
            ).coerceIn(0.0, 1.0)
                .powSafe(cfg.handExponent)

        val specialSensitivity =
            when (input.jointIndex) {
                0 -> 1.0 - cfg.wristExtraSensitivity
                4, 8, 12, 16, 20 ->
                    1.0 - cfg.fingerTipExtraSensitivity
                else -> 1.0
            }.coerceIn(0.70, 1.0)

        val predictedSuppression =
            if (input.predicted) {
                (1.0 - cfg.predictedPenaltyStrength)
            } else {
                1.0
            }

        if (input.predicted) {
            reasons += "prediction-only-evidence"
        }

        val instabilityPenalty =
            1.0 -
                cfg.unstablePenaltyStrength *
                (1.0 - input.temporalStability.coerceIn(0.0, 1.0))

        val velocityPenalty =
            0.62 +
                0.38 * motionConfidence

        val reliability = (
            detectorConfidence *
                handConfidence *
                temporalConfidence *
                borderConfidence.powSafe(
                    1.0 +
                        cfg.borderPenaltyStrength * 1.35,
                ) *
                motionConfidence.powSafe(
                    0.55,
                ) *
                input.observationFreshness *
                predictedSuppression *
                instabilityPenalty *
                specialSensitivity
            ).coerceIn(0.0, 1.0)

        val uncertaintyInflation =
            (
                1.0 +
                    (1.0 - borderConfidence) *
                        cfg.borderPenaltyStrength * 2.2 +
                    (1.0 - temporalConfidence) *
                        cfg.unstablePenaltyStrength * 1.8 +
                    if (input.predicted) {
                        cfg.predictedPenaltyStrength * 2.4
                    } else {
                        0.0
                    } +
                    (1.0 - motionConfidence) *
                        0.80
            ).coerceAtLeast(1.0)

        val uncertaintyMeters =
            (
                cfg.uncertaintyFloorMeters +
                    (
                        1.0 -
                            reliability
                        ) *
                        (
                            cfg.uncertaintyCeilingMeters -
                                cfg.uncertaintyFloorMeters
                        )
            ) *
                uncertaintyInflation *
                velocityPenalty
                .let { value ->
                    if (value.isFinite()) {
                        value
                    } else {
                        cfg.uncertaintyCeilingMeters
                    }
                }
                .coerceIn(
                    cfg.uncertaintyFloorMeters,
                    cfg.uncertaintyCeilingMeters,
                )

        val hardRejected =
            reliability < cfg.robustOutlierWeightFloor ||
                borderConfidence <= 0.02 ||
                !uncertaintyMeters.isFinite()

        if (hardRejected) {
            reasons += "joint-expert-rejected"
        }

        val viewExpertWeight =
            if (hardRejected) {
                0.0
            } else {
                max(
                    cfg.minimumFusionWeight,
                    reliability *
                        exp(
                            -uncertaintyMeters /
                                cfg.uncertaintyCeilingMeters
                        ),
                ).coerceIn(0.0, 1.0)
            }

        return Result(
            reliability = reliability,
            uncertaintyMeters = uncertaintyMeters,
            borderConfidence = borderConfidence,
            motionConfidence = motionConfidence,
            temporalConfidence = temporalConfidence,
            viewExpertWeight = viewExpertWeight,
            predictedSuppression = predictedSuppression,
            hardRejected = hardRejected,
            reasons = reasons.distinct(),
        )
    }

    private fun borderConfidence(
        x: Double,
        y: Double,
        expectedMargin: Double,
        hardMargin: Double,
    ): Double {
        val distanceToBorder =
            min(
                min(x, 1.0 - x),
                min(y, 1.0 - y),
            )

        if (!distanceToBorder.isFinite()) {
            return 0.0
        }

        if (distanceToBorder <= hardMargin) {
            return 0.02
        }

        if (distanceToBorder >= expectedMargin) {
            return 1.0
        }

        val normalized =
            (
                distanceToBorder - hardMargin
            ) /
                (
                    expectedMargin - hardMargin
                ).coerceAtLeast(1e-9)

        return (
            0.10 +
                normalized.coerceIn(0.0, 1.0) * 0.90
        ).coerceIn(0.0, 1.0)
    }

    private fun Double.powSafe(exponent: Double): Double =
        if (!isFinite() || this <= 0.0) {
            if (exponent == 0.0) 1.0 else 0.0
        } else {
            this.pow(exponent)
                .coerceIn(0.0, 1.0)
        }
}
