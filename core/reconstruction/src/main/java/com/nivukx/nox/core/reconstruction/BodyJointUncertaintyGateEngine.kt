package com.nivukx.nox.core.reconstruction

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Joint-level observation reliability gate for body reconstruction.
 *
 * Each camera/joint observation is treated as an evidence expert. The gate combines detector
 * confidence, temporal timing quality, rolling-shutter risk, exposure/readout ratio, image-border
 * risk, and prediction status into a bounded uncertainty proxy and fusion weight.
 *
 * This layer deliberately does not triangulate or rewrite geometry. It only determines how much a
 * single observation should be allowed to influence the existing calibrated reconstruction stack.
 */
class BodyJointUncertaintyGateEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val minimumObservationConfidence: Double = 0.04,
        val confidenceExponent: Double = 1.35,
        val timingExponent: Double = 0.90,
        val rollingShutterExponent: Double = 0.82,
        val borderPenaltyStrength: Double = 0.42,
        val exposurePenaltyStrength: Double = 0.22,
        val predictionPenaltyStrength: Double = 0.56,
        val minimumFusionWeight: Double = 0.006,
        val robustWeightFloor: Double = 0.025,
        val uncertaintyFloorMeters: Double = 0.0015,
        val uncertaintyCeilingMeters: Double = 0.75,
        val expectedBorderMargin: Double = 0.03,
        val hardBorderMargin: Double = 0.006,
        val maximumExposureFraction: Double = 0.95,
        val maximumTimingUncertaintyMillis: Double = 40.0,
    ) {
        init {
            require(minimumObservationConfidence in 0.0..1.0)
            require(confidenceExponent > 0.0)
            require(timingExponent > 0.0)
            require(rollingShutterExponent > 0.0)
            require(borderPenaltyStrength in 0.0..1.0)
            require(exposurePenaltyStrength in 0.0..1.0)
            require(predictionPenaltyStrength in 0.0..1.0)
            require(minimumFusionWeight in 0.0..1.0)
            require(robustWeightFloor in 0.0..1.0)
            require(uncertaintyFloorMeters > 0.0)
            require(uncertaintyCeilingMeters >= uncertaintyFloorMeters)
            require(expectedBorderMargin in 0.0..0.25)
            require(hardBorderMargin in 0.0..expectedBorderMargin)
            require(maximumExposureFraction > 0.0)
            require(maximumTimingUncertaintyMillis > 0.0)
        }
    }

    data class Input(
        val normalizedX: Double,
        val normalizedY: Double,
        val observationConfidence: Double,
        val captureTimingConfidence: Double,
        val temporalAlignmentConfidence: Double,
        val temporalUncertaintyNanos: Long,
        val rollingShutterFraction: Double,
        val rowTimeOffsetNanos: Long,
        val frameDurationNanos: Long,
        val exposureTimeNanos: Long,
        val predicted: Boolean = false,
        val baseUncertaintyMeters: Double = 0.0,
        val anatomicalStability: Double = 1.0,
        val temporalSpanFrames: Int = 1,
    ) {
        init {
            require(normalizedX.isFinite())
            require(normalizedY.isFinite())
            listOf(
                observationConfidence,
                captureTimingConfidence,
                temporalAlignmentConfidence,
                rollingShutterFraction,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(temporalUncertaintyNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(baseUncertaintyMeters.isFinite())
            require(baseUncertaintyMeters >= 0.0)
            require(anatomicalStability.isFinite() && anatomicalStability in 0.0..1.0)
            require(temporalSpanFrames >= 1)
        }
    }

    data class Result(
        val viewExpertWeight: Double,
        val uncertaintyMeters: Double,
        val borderConfidence: Double,
        val timingConfidence: Double,
        val rollingShutterConfidence: Double,
        val exposureConfidence: Double,
        val hardRejected: Boolean,
        val reasons: List<String>,
    ) {
        val accepted: Boolean
            get() = !hardRejected && viewExpertWeight > 0.0
    }

    @Synchronized
    fun configure(value: Config) {
        config = value
    }

    @Synchronized
    fun configureFromMotionCaptureCustomization(
        customization: com.nivukx.nox.core.model.MotionCaptureCustomization,
    ) {
        configure(
            Config(
                minimumObservationConfidence =
                    (
                        0.02 +
                            (1.0 -
                                customization.landmarkConsistencyStrictness) *
                                0.08
                    ).coerceIn(0.01, 0.12),
                confidenceExponent =
                    1.05 +
                        customization.reconstructionEvidenceFusionStrength *
                            0.75,
                timingExponent =
                    0.62 +
                        customization.subframeAlignmentStrictness *
                            0.72,
                rollingShutterExponent =
                    0.58 +
                        customization.rollingShutterCompensation *
                            0.66,
                borderPenaltyStrength =
                    0.20 +
                        customization.landmarkConsistencyStrictness *
                            0.60,
                exposurePenaltyStrength =
                    0.12 +
                        customization.actorMicroMotionRetention *
                            0.24,
                predictionPenaltyStrength =
                    0.25 +
                        customization.occlusionRecoveryConfidence *
                            0.62,
                minimumFusionWeight =
                    (
                        0.004 +
                            customization.multiViewEvidenceAdaptivity *
                                0.014
                    ).coerceIn(0.003, 0.025),
                robustWeightFloor =
                    (
                        0.018 +
                            customization.landmarkConsistencyStrictness *
                                0.048
                    ).coerceIn(0.015, 0.08),
                uncertaintyFloorMeters =
                    (
                        0.001 +
                            (
                                1.0 -
                                    customization.scalePreservation
                            ) *
                                0.005
                    ).coerceIn(0.001, 0.006),
                uncertaintyCeilingMeters =
                    customization.maximumReconstructionUncertaintyMeters
                        .coerceIn(0.05, 1.0),
                expectedBorderMargin =
                    (
                        0.02 +
                            customization.landmarkConsistencyStrictness *
                                0.03
                    ).coerceIn(0.015, 0.08),
                maximumExposureFraction = 0.95,
                maximumTimingUncertaintyMillis =
                    (
                        8.0 +
                            customization.networkJitterToleranceMillis
                        ).coerceIn(8.0, 80.0),
            ),
        )
    }

    fun configuration(): Config = config

    fun evaluate(input: Input): Result {
        val cfg = config
        val reasons = mutableListOf<String>()

        if (input.observationConfidence < cfg.minimumObservationConfidence) {
            reasons += "observation-confidence-below-floor"
        }

        val borderConfidence =
            borderConfidence(
                input.normalizedX,
                input.normalizedY,
                cfg.expectedBorderMargin,
                cfg.hardBorderMargin,
            )

        if (borderConfidence < 0.30) {
            reasons += "image-boundary-risk"
        }

        val timingUncertaintyMillis =
            input.temporalUncertaintyNanos.toDouble() / 1_000_000.0

        val timingConfidence = (
            input.captureTimingConfidence *
                input.temporalAlignmentConfidence *
                exp(
                    -timingUncertaintyMillis /
                        cfg.maximumTimingUncertaintyMillis,
                )
        ).coerceIn(0.0, 1.0)

        if (timingConfidence < 0.35) {
            reasons += "temporal-alignment-risk"
        }

        val frameDuration =
            max(1L, input.frameDurationNanos).toDouble()
        val exposureFraction =
            (
                input.exposureTimeNanos.toDouble() /
                    frameDuration
            ).coerceIn(0.0, 1.0)

        val exposureConfidence =
            (
                1.0 -
                    (
                        exposureFraction /
                            cfg.maximumExposureFraction
                    ).coerceIn(0.0, 1.0) *
                        cfg.exposurePenaltyStrength
            ).coerceIn(0.0, 1.0)

        if (exposureFraction > 0.85) {
            reasons += "long-exposure-motion-risk"
        }

        val rowOffsetRatio =
            (
                kotlin.math.abs(input.rowTimeOffsetNanos).toDouble() /
                    frameDuration
            ).coerceIn(0.0, 1.0)

        val rollingConfidence =
            (
                input.rollingShutterFraction *
                    (1.0 - rowOffsetRatio * 0.72)
            )
                .coerceIn(0.0, 1.0)
                .let { measured ->
                    (
                        1.0 -
                            (
                                1.0 - measured
                            ) *
                            cfg.rollingShutterExponent
                    ).coerceIn(0.0, 1.0)
                }

        if (rollingConfidence < 0.55) {
            reasons += "rolling-shutter-risk"
        }

        val detectorConfidence =
            input.observationConfidence
                .coerceIn(0.0, 1.0)
                .powSafe(cfg.confidenceExponent)

        val timingTerm =
            timingConfidence
                .powSafe(cfg.timingExponent)

        val rollingTerm =
            rollingConfidence
                .powSafe(cfg.rollingShutterExponent)

        val predictionTerm =
            if (input.predicted) {
                1.0 - cfg.predictionPenaltyStrength
            } else {
                1.0
            }

        if (input.predicted) {
            reasons += "prediction-only-observation"
        }

        val anatomicalTerm =
            input.anatomicalStability
                .coerceIn(0.0, 1.0)
                .powSafe(0.65)
        if (input.anatomicalStability < 0.55) {
            reasons += "anatomically-unstable-landmark"
        }

        val temporalSpanTerm =
            (
                1.0 -
                    (input.temporalSpanFrames - 1).coerceAtLeast(0)
                        .toDouble() * 0.012
            ).coerceIn(0.55, 1.0)
        if (input.temporalSpanFrames > 24) {
            reasons += "long-temporal-span-reliability-reduced"
        }

        val reliability = (
            detectorConfidence *
                timingTerm *
                rollingTerm *
                borderConfidence.powSafe(
                    1.0 +
                        cfg.borderPenaltyStrength,
                ) *
                exposureConfidence *
                predictionTerm *
                anatomicalTerm *
                temporalSpanTerm
        ).coerceIn(0.0, 1.0)

        val uncertaintyInflation =
            (
                1.0 +
                    (1.0 - borderConfidence) *
                        cfg.borderPenaltyStrength *
                        2.0 +
                    (1.0 - timingConfidence) *
                        1.8 +
                    (1.0 - rollingConfidence) *
                        1.4 +
                    (1.0 - exposureConfidence) *
                        cfg.exposurePenaltyStrength *
                        2.5 +
                    if (input.predicted) {
                        cfg.predictionPenaltyStrength * 2.5
                    } else {
                        0.0
                    }
            ).coerceAtLeast(1.0)

        val baseUncertainty =
            input.baseUncertaintyMeters
                .coerceIn(
                    cfg.uncertaintyFloorMeters,
                    cfg.uncertaintyCeilingMeters,
                )

        val uncertaintyMeters = (
            baseUncertainty *
                uncertaintyInflation +
                (
                    1.0 - reliability
                ) *
                    (
                        cfg.uncertaintyCeilingMeters -
                            cfg.uncertaintyFloorMeters
                    ) *
                    0.16
        ).coerceIn(
            cfg.uncertaintyFloorMeters,
            cfg.uncertaintyCeilingMeters,
        )

        val hardRejected =
            reliability < cfg.robustWeightFloor ||
                borderConfidence <= 0.02 ||
                !uncertaintyMeters.isFinite()

        if (hardRejected) {
            reasons += "body-joint-evidence-rejected"
        }

        val weight =
            if (hardRejected) {
                0.0
            } else {
                max(
                    cfg.minimumFusionWeight,
                    reliability *
                        exp(
                            -uncertaintyMeters /
                                cfg.uncertaintyCeilingMeters,
                        ),
                ).coerceIn(0.0, 1.0)
            }

        return Result(
            viewExpertWeight = weight,
            uncertaintyMeters = uncertaintyMeters,
            borderConfidence = borderConfidence,
            timingConfidence = timingConfidence,
            rollingShutterConfidence = rollingConfidence,
            exposureConfidence = exposureConfidence,
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
        val distance =
            min(
                min(x, 1.0 - x),
                min(y, 1.0 - y),
            )

        if (!distance.isFinite()) return 0.0
        if (distance <= hardMargin) return 0.02
        if (distance >= expectedMargin) return 1.0

        val normalized =
            (
                distance - hardMargin
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
