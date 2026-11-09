package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Post-hoc uncertainty calibration for face and binocular-eye reconstruction.
 *
 * This engine does not retrain a model and it does not require ground-truth labels during capture.
 * Instead it treats internal geometric disagreement as a bounded nonconformity signal:
 * reprojection/shape disagreement for the face and binocular/angular/distribution disagreement for
 * the eyes. Clean samples produce a calibration score:
 *
 *     nonconformity = observed_quality_error / predicted_uncertainty
 *
 * A finite-sample empirical quantile of these scores becomes the session-specific scale factor.
 * Downstream systems can therefore widen or tighten raw uncertainty under a domain shift instead of
 * trusting a fixed training-domain uncertainty estimate.
 *
 * Important contract:
 * - this is calibration of an internal consistency envelope, not a claim of ground-truth metric
 *   accuracy;
 * - rejected, OOD, stale, or prediction-only samples never teach calibration;
 * - calibration history is bounded;
 * - robust trimming prevents one pathological frame from rewriting the scale;
 * - the engine is deterministic and replayable from the same input sequence.
 */
class FaceEyeUncertaintyCalibrationEngine(
    private var config: Config = Config(),
) {

    enum class Channel {
        FACE,
        EYE,
    }

    data class Config(
        val targetCoverage: Double = 0.95,
        val minimumCalibrationSamples: Int = 8,
        val maximumCalibrationSamples: Int = 96,
        val minimumPredictedUncertainty: Double = 0.004,
        val maximumAcceptedNonconformity: Double = 12.0,
        val minimumQualityForCalibration: Double = 0.58,
        val minimumEyeGeometryForCalibration: Double = 0.62,
        val maximumOodRiskForCalibration: Double = 0.30,
        val robustTrimFraction: Double = 0.08,
        val multiplierFloor: Double = 0.70,
        val multiplierCeiling: Double = 4.00,
        val learningAlpha: Double = 0.18,
        val coverageTolerance: Double = 0.06,
        val faceProxyScaleMeters: Double = 0.035,
        val eyeProxyScaleMeters: Double = 0.018,
        val faceResidualWeight: Double = 0.46,
        val faceTemporalWeight: Double = 0.20,
        val faceReprojectionWeight: Double = 0.22,
        val faceShapeWeight: Double = 0.12,
        val eyeAngularWeight: Double = 0.26,
        val eyeBinocularWeight: Double = 0.24,
        val eyeTemporalWeight: Double = 0.16,
        val eyeAgreementWeight: Double = 0.14,
        val eyeUncertaintyWeight: Double = 0.12,
        val eyeIpdWeight: Double = 0.08,
    ) {
        init {
            require(targetCoverage in 0.50..0.999)
            require(minimumCalibrationSamples >= 2)
            require(maximumCalibrationSamples >= minimumCalibrationSamples)
            require(minimumPredictedUncertainty > 0.0)
            require(maximumAcceptedNonconformity > 0.0)
            require(minimumQualityForCalibration in 0.0..1.0)
            require(minimumEyeGeometryForCalibration in 0.0..1.0)
            require(maximumOodRiskForCalibration in 0.0..1.0)
            require(robustTrimFraction in 0.0..0.30)
            require(multiplierFloor > 0.0)
            require(multiplierCeiling >= multiplierFloor)
            require(learningAlpha in 0.0..1.0)
            require(coverageTolerance in 0.0..0.30)
            listOf(
                faceProxyScaleMeters,
                eyeProxyScaleMeters,
            ).forEach { require(it > 0.0) }
            listOf(
                faceResidualWeight,
                faceTemporalWeight,
                faceReprojectionWeight,
                faceShapeWeight,
                eyeAngularWeight,
                eyeBinocularWeight,
                eyeTemporalWeight,
                eyeAgreementWeight,
                eyeUncertaintyWeight,
                eyeIpdWeight,
            ).forEach { require(it >= 0.0) }
        }
    }

    data class Input(
        val timestampNanos: Long,
        val faceConfidence: Double,
        val faceTemporalConfidence: Double,
        val faceReprojectionScore: Double,
        val faceShapeConfidence: Double,
        val faceFreshness: Double,
        val faceUncertaintyMeters: Double,
        val leftEyeConfidence: Double,
        val rightEyeConfidence: Double,
        val binocularConfidence: Double,
        val eyeTemporalConfidence: Double,
        val gazeAgreement: Double,
        val angularSpreadDegrees: Double,
        val angularSpreadRejectDegrees: Double,
        val eyeUncertaintyMeters: Double,
        val ipdConsistencyScore: Double,
        val faceOodRisk: Double,
        val eyeOodRisk: Double,
        val faceAccepted: Boolean,
        val eyeAccepted: Boolean,
        val predictionOnlyFace: Boolean = false,
        val predictionOnlyEyes: Boolean = false,
        val faceFresh: Boolean = true,
        val eyesFresh: Boolean = true,
    ) {
        init {
            require(timestampNanos > 0L)
            listOf(
                faceConfidence,
                faceTemporalConfidence,
                faceReprojectionScore,
                faceShapeConfidence,
                faceFreshness,
                leftEyeConfidence,
                rightEyeConfidence,
                binocularConfidence,
                eyeTemporalConfidence,
                gazeAgreement,
                ipdConsistencyScore,
                faceOodRisk,
                eyeOodRisk,
            ).forEach { require(it in 0.0..1.0) }
            require(faceUncertaintyMeters.isFinite() && faceUncertaintyMeters >= 0.0)
            require(eyeUncertaintyMeters.isFinite() && eyeUncertaintyMeters >= 0.0)
            require(angularSpreadDegrees.isFinite() && angularSpreadDegrees >= 0.0)
            require(angularSpreadRejectDegrees > 0.0)
        }
    }

    data class ChannelResult(
        val channel: Channel,
        val calibratedUncertaintyMeters: Double,
        val rawUncertaintyMeters: Double,
        val multiplier: Double,
        val empiricalCoverage: Double,
        val calibrationSamples: Int,
        val calibrationReady: Boolean,
        val qualityForCalibration: Double,
        val nonconformity: Double?,
        val usedForCalibration: Boolean,
        val warning: String?,
    )

    data class Result(
        val timestampNanos: Long,
        val face: ChannelResult,
        val eye: ChannelResult,
        val calibrationConfidence: Double,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val ready: Boolean
            get() =
                face.calibrationReady &&
                    eye.calibrationReady

        val calibratedFaceUncertaintyMeters: Double
            get() = face.calibratedUncertaintyMeters

        val calibratedEyeUncertaintyMeters: Double
            get() = eye.calibratedUncertaintyMeters
    }

    private data class ChannelState(
        val scores: ArrayDeque<Double> = ArrayDeque(),
        var multiplier: Double = 1.0,
        var empiricalCoverage: Double = 0.0,
        var acceptedSamples: Int = 0,
        var rejectedSamples: Int = 0,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
    )

    private val faceState = ChannelState()
    private val eyeState = ChannelState()

    @Synchronized
    fun configuration(): Config = config

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig

        clampState(faceState)
        clampState(eyeState)
    }

    @Synchronized
    fun reset() {
        clearState(faceState)
        clearState(eyeState)
    }

    @Synchronized
    fun snapshot(): Snapshot =
        Snapshot(
            faceMultiplier = faceState.multiplier,
            eyeMultiplier = eyeState.multiplier,
            faceCoverage = faceState.empiricalCoverage,
            eyeCoverage = eyeState.empiricalCoverage,
            faceSamples = faceState.acceptedSamples,
            eyeSamples = eyeState.acceptedSamples,
            faceRejectedSamples = faceState.rejectedSamples,
            eyeRejectedSamples = eyeState.rejectedSamples,
        )

    data class Snapshot(
        val faceMultiplier: Double,
        val eyeMultiplier: Double,
        val faceCoverage: Double,
        val eyeCoverage: Double,
        val faceSamples: Int,
        val eyeSamples: Int,
        val faceRejectedSamples: Int,
        val eyeRejectedSamples: Int,
    )

    @Synchronized
    fun evaluate(input: Input): Result {
        val faceTimestampFresh =
            input.timestampNanos >
                faceState.lastTimestampNanos
        val eyeTimestampFresh =
            input.timestampNanos >
                eyeState.lastTimestampNanos

        val faceProxy =
            faceObservedProxyMeters(input)

        val eyeProxy =
            eyeObservedProxyMeters(input)

        val faceCalibrationQuality =
            faceCalibrationQuality(input)

        val eyeCalibrationQuality =
            eyeCalibrationQuality(input)

        val faceUsed =
            faceTimestampFresh &&
                shouldTeachFace(
                    input,
                    faceCalibrationQuality,
                )

        val eyeUsed =
            eyeTimestampFresh &&
                shouldTeachEyes(
                    input,
                    eyeCalibrationQuality,
                )

        val faceNonconformity =
            nonconformity(
                observedMeters = faceProxy,
                predictedMeters = input.faceUncertaintyMeters,
                state = faceState,
            )

        val eyeNonconformity =
            nonconformity(
                observedMeters = eyeProxy,
                predictedMeters = input.eyeUncertaintyMeters,
                state = eyeState,
            )

        if (faceUsed) {
            faceNonconformity?.let {
                addCalibrationScore(
                    state = faceState,
                    score = it,
                )
            } ?: run {
                faceState.rejectedSamples += 1
            }
        } else {
            faceState.rejectedSamples += 1
        }

        if (eyeUsed) {
            eyeNonconformity?.let {
                addCalibrationScore(
                    state = eyeState,
                    score = it,
                )
            } ?: run {
                eyeState.rejectedSamples += 1
            }
        } else {
            eyeState.rejectedSamples += 1
        }

        updateCalibrationScale(faceState)
        updateCalibrationScale(eyeState)

        val faceReady =
            faceState.scores.size >=
                config.minimumCalibrationSamples

        val eyeReady =
            eyeState.scores.size >=
                config.minimumCalibrationSamples

        val faceCalibrationConfidence =
            calibrationConfidence(faceState)

        val eyeCalibrationConfidence =
            calibrationConfidence(eyeState)

        val warnings = buildList {
            if (!faceTimestampFresh) {
                add("face-uncertainty-calibration-timestamp-reversal")
            }
            if (!eyeTimestampFresh) {
                add("eye-uncertainty-calibration-timestamp-reversal")
            }
            if (!faceReady) {
                add(
                    "face-uncertainty-calibration-warmup=" +
                        faceState.scores.size +
                        "/" +
                        config.minimumCalibrationSamples,
                )
            }

            if (!eyeReady) {
                add(
                    "eye-uncertainty-calibration-warmup=" +
                        eyeState.scores.size +
                        "/" +
                        config.minimumCalibrationSamples,
                )
            }

            if (
                faceState.empiricalCoverage > 0.0 &&
                abs(
                    faceState.empiricalCoverage -
                        config.targetCoverage,
                ) >
                    config.coverageTolerance
            ) {
                add(
                    "face-uncertainty-coverage-out-of-envelope",
                )
            }

            if (
                eyeState.empiricalCoverage > 0.0 &&
                abs(
                    eyeState.empiricalCoverage -
                        config.targetCoverage,
                ) >
                    config.coverageTolerance
            ) {
                add(
                    "eye-uncertainty-coverage-out-of-envelope",
                )
            }

            if (
                faceState.multiplier >=
                    config.multiplierCeiling * 0.92
            ) {
                add(
                    "face-uncertainty-calibration-near-ceiling",
                )
            }

            if (
                eyeState.multiplier >=
                    config.multiplierCeiling * 0.92
            ) {
                add(
                    "eye-uncertainty-calibration-near-ceiling",
                )
            }
        }

        val calibrationConfidence =
            (
                faceCalibrationConfidence * 0.46 +
                    eyeCalibrationConfidence * 0.54
            ).coerceIn(0.0, 1.0)

        val result = Result(
            timestampNanos = input.timestampNanos,
            face =
                channelResult(
                    channel = Channel.FACE,
                    state = faceState,
                    rawUncertaintyMeters =
                        input.faceUncertaintyMeters,
                    qualityForCalibration =
                        faceCalibrationQuality,
                    nonconformity =
                        if (faceUsed) {
                            faceNonconformity
                        } else {
                            null
                        },
                    calibrationReady = faceReady,
                    usedForCalibration = faceUsed,
                ),
            eye =
                channelResult(
                    channel = Channel.EYE,
                    state = eyeState,
                    rawUncertaintyMeters =
                        input.eyeUncertaintyMeters,
                    qualityForCalibration =
                        eyeCalibrationQuality,
                    nonconformity =
                        if (eyeUsed) {
                            eyeNonconformity
                        } else {
                            null
                        },
                    calibrationReady = eyeReady,
                    usedForCalibration = eyeUsed,
                ),
            calibrationConfidence = calibrationConfidence,
            warnings = warnings.distinct(),
            recommendations = buildRecommendations(
                faceReady = faceReady,
                eyeReady = eyeReady,
                faceMultiplier = faceState.multiplier,
                eyeMultiplier = eyeState.multiplier,
                faceCoverage = faceState.empiricalCoverage,
                eyeCoverage = eyeState.empiricalCoverage,
                faceQuality = faceCalibrationQuality,
                eyeQuality = eyeCalibrationQuality,
            ),
        )

        if (faceTimestampFresh) {
            faceState.lastTimestampNanos =
                input.timestampNanos
        }
        if (eyeTimestampFresh) {
            eyeState.lastTimestampNanos =
                input.timestampNanos
        }

        return result
    }

    private fun shouldTeachFace(
        input: Input,
        quality: Double,
    ): Boolean =
        input.faceAccepted &&
            input.faceFresh &&
            !input.predictionOnlyFace &&
            input.faceOodRisk <= config.maximumOodRiskForCalibration &&
            input.faceConfidence >= config.minimumQualityForCalibration &&
            quality >= config.minimumQualityForCalibration

    private fun shouldTeachEyes(
        input: Input,
        quality: Double,
    ): Boolean =
        input.eyeAccepted &&
            input.eyesFresh &&
            !input.predictionOnlyEyes &&
            input.eyeOodRisk <= config.maximumOodRiskForCalibration &&
            min(
                input.leftEyeConfidence,
                input.rightEyeConfidence,
            ) >= config.minimumEyeGeometryForCalibration &&
            input.binocularConfidence >=
                config.minimumEyeGeometryForCalibration &&
            quality >= config.minimumEyeGeometryForCalibration

    private fun faceCalibrationQuality(
        input: Input,
    ): Double =
        (
            input.faceConfidence * 0.38 +
                input.faceTemporalConfidence * 0.20 +
                input.faceReprojectionScore * 0.18 +
                input.faceShapeConfidence * 0.14 +
                input.faceFreshness * 0.10
            ).coerceIn(0.0, 1.0)

    private fun eyeCalibrationQuality(
        input: Input,
    ): Double =
        (
            min(
                input.leftEyeConfidence,
                input.rightEyeConfidence,
            ) * 0.22 +
                input.binocularConfidence * 0.24 +
                input.eyeTemporalConfidence * 0.18 +
                input.gazeAgreement * 0.16 +
                input.ipdConsistencyScore * 0.10 +
                (
                    1.0 -
                        input.angularSpreadDegrees /
                            input.angularSpreadRejectDegrees
                ).coerceIn(0.0, 1.0) * 0.10
            ).coerceIn(0.0, 1.0)

    private fun faceObservedProxyMeters(
        input: Input,
    ): Double {
        val reprojectionError =
            (
                1.0 -
                    input.faceReprojectionScore
                ).coerceIn(0.0, 1.0)

        val geometryError =
            (
                1.0 -
                    input.faceConfidence
                ).coerceIn(0.0, 1.0)

        val temporalError =
            (
                1.0 -
                    input.faceTemporalConfidence
                ).coerceIn(0.0, 1.0)

        val shapeError =
            (
                1.0 -
                    input.faceShapeConfidence
                ).coerceIn(0.0, 1.0)

        val normalized =
            (
                reprojectionError * config.faceReprojectionWeight +
                    geometryError * config.faceResidualWeight +
                    temporalError * config.faceTemporalWeight +
                    shapeError * config.faceShapeWeight
                ).coerceIn(0.0, 1.0)

        return normalized *
            config.faceProxyScaleMeters
    }

    private fun eyeObservedProxyMeters(
        input: Input,
    ): Double {
        val angularError =
            (
                input.angularSpreadDegrees /
                    input.angularSpreadRejectDegrees
            ).coerceIn(0.0, 1.0)

        val binocularError =
            (
                1.0 -
                    input.binocularConfidence
                ).coerceIn(0.0, 1.0)

        val temporalError =
            (
                1.0 -
                    input.eyeTemporalConfidence
                ).coerceIn(0.0, 1.0)

        val agreementError =
            (
                1.0 -
                    input.gazeAgreement
                ).coerceIn(0.0, 1.0)

        val uncertaintyError =
            (
                input.eyeUncertaintyMeters /
                    config.eyeProxyScaleMeters
            ).coerceIn(0.0, 1.0)

        val ipdError =
            (
                1.0 -
                    input.ipdConsistencyScore
                ).coerceIn(0.0, 1.0)

        val normalized =
            (
                angularError * config.eyeAngularWeight +
                    binocularError * config.eyeBinocularWeight +
                    temporalError * config.eyeTemporalWeight +
                    agreementError * config.eyeAgreementWeight +
                    uncertaintyError * config.eyeUncertaintyWeight +
                    ipdError * config.eyeIpdWeight
                ).coerceIn(0.0, 1.0)

        return normalized *
            config.eyeProxyScaleMeters
    }

    private fun nonconformity(
        observedMeters: Double,
        predictedMeters: Double,
        state: ChannelState,
    ): Double? {
        if (!observedMeters.isFinite() || observedMeters < 0.0) {
            return null
        }

        val baseline =
            max(
                predictedMeters,
                config.minimumPredictedUncertainty,
            )

        /*
         * The first few observations are intentionally allowed to populate the empirical
         * distribution. Extreme scores are rejected rather than allowed to permanently widen the
         * calibration envelope.
         */
        val score =
            observedMeters /
                baseline

        if (!score.isFinite() ||
            score < 0.0 ||
            score > config.maximumAcceptedNonconformity
        ) {
            return null
        }

        return score
    }

    private fun addCalibrationScore(
        state: ChannelState,
        score: Double,
    ) {
        state.scores.addLast(
            score.coerceIn(
                0.0,
                config.maximumAcceptedNonconformity,
            ),
        )

        while (
            state.scores.size >
                config.maximumCalibrationSamples
        ) {
            state.scores.removeFirst()
        }

        state.acceptedSamples += 1
    }

    private fun updateCalibrationScale(
        state: ChannelState,
    ) {
        if (
            state.scores.size <
                config.minimumCalibrationSamples
        ) {
            return
        }

        val sorted =
            state.scores
                .toList()
                .sorted()

        val trimCount =
            (
                sorted.size *
                    config.robustTrimFraction
                ).toInt()
                .coerceAtMost(
                    (sorted.size - 2) / 2,
                )

        val trimmed =
            if (trimCount > 0) {
                sorted.drop(trimCount).dropLast(trimCount)
            } else {
                sorted
            }

        val quantile =
            empiricalQuantile(
                trimmed,
                config.targetCoverage,
            )

        state.multiplier =
            (
                state.multiplier *
                    (1.0 - config.learningAlpha) +
                    quantile *
                    config.learningAlpha
                ).coerceIn(
                    config.multiplierFloor,
                    config.multiplierCeiling,
                )

        state.empiricalCoverage =
            trimmed.count {
                it <= state.multiplier
            }.toDouble() /
                trimmed.size.coerceAtLeast(1).toDouble()
    }

    private fun calibrationConfidence(
        state: ChannelState,
    ): Double {
        if (state.scores.isEmpty()) return 0.0

        val readiness =
            (
                state.scores.size.toDouble() /
                    config.minimumCalibrationSamples
            ).coerceIn(0.0, 1.0)

        val coverage =
            if (state.empiricalCoverage <= 0.0) {
                0.42
            } else {
                (
                    1.0 -
                        abs(
                            state.empiricalCoverage -
                                config.targetCoverage,
                        ) /
                        config.coverageTolerance.coerceAtLeast(1e-6)
                    ).coerceIn(0.0, 1.0)
            }

        val stability =
            (
                1.0 -
                    abs(
                        state.multiplier - 1.0,
                    ) / config.multiplierCeiling
                ).coerceIn(0.0, 1.0)

        return (
            readiness * 0.56 +
                coverage * 0.30 +
                stability * 0.14
            ).coerceIn(0.0, 1.0)
    }

    private fun channelResult(
        channel: Channel,
        state: ChannelState,
        rawUncertaintyMeters: Double,
        qualityForCalibration: Double,
        nonconformity: Double?,
        calibrationReady: Boolean,
        usedForCalibration: Boolean,
    ): ChannelResult {
        val multiplier =
            state.multiplier.coerceIn(
                config.multiplierFloor,
                config.multiplierCeiling,
            )

        val calibrated =
            (
                rawUncertaintyMeters.coerceAtLeast(
                    config.minimumPredictedUncertainty,
                ) * multiplier
            ).coerceAtMost(
                rawUncertaintyMeters.coerceAtLeast(
                    config.minimumPredictedUncertainty,
                ) * config.multiplierCeiling,
            )

        return ChannelResult(
            channel = channel,
            calibratedUncertaintyMeters = calibrated,
            rawUncertaintyMeters = rawUncertaintyMeters,
            multiplier = multiplier,
            empiricalCoverage = state.empiricalCoverage,
            calibrationSamples = state.scores.size,
            calibrationReady = calibrationReady,
            qualityForCalibration = qualityForCalibration,
            nonconformity = nonconformity,
            usedForCalibration = usedForCalibration,
            warning =
                when {
                    !usedForCalibration ->
                        "calibration-evidence-rejected"

                    !calibrationReady ->
                        "calibration-warmup"

                    abs(
                        state.empiricalCoverage -
                            config.targetCoverage,
                    ) > config.coverageTolerance ->
                        "empirical-coverage-out-of-envelope"

                    else -> null
                },
        )
    }

    private fun buildRecommendations(
        faceReady: Boolean,
        eyeReady: Boolean,
        faceMultiplier: Double,
        eyeMultiplier: Double,
        faceCoverage: Double,
        eyeCoverage: Double,
        faceQuality: Double,
        eyeQuality: Double,
    ): List<String> = buildList {
        if (!faceReady) {
            add(
                "Collect stable face frames before trusting calibrated face uncertainty.",
            )
        }

        if (!eyeReady) {
            add(
                "Collect stable bilateral gaze frames before trusting calibrated eye uncertainty.",
            )
        }

        if (faceMultiplier > 2.0) {
            add(
                "Face uncertainty has widened materially; improve framing, lighting, or reprojection quality.",
            )
        }

        if (eyeMultiplier > 2.0) {
            add(
                "Eye uncertainty has widened materially; improve binocular parallax and eye visibility.",
            )
        }

        if (
            abs(
                faceCoverage - config.targetCoverage,
            ) > config.coverageTolerance &&
            faceReady
        ) {
            add(
                "Refresh face calibration under the current capture domain before long metric segments.",
            )
        }

        if (
            abs(
                eyeCoverage - config.targetCoverage,
            ) > config.coverageTolerance &&
            eyeReady
        ) {
            add(
                "Refresh eye calibration under the current lighting/head-pose domain before trusting gaze-heavy segments.",
            )
        }

        if (faceQuality < config.minimumQualityForCalibration) {
            add(
                "Keep low-quality face evidence out of the calibration set.",
            )
        }

        if (eyeQuality < config.minimumEyeGeometryForCalibration) {
            add(
                "Keep weak binocular geometry out of the calibration set.",
            )
        }
    }.distinct().take(8)

    private fun empiricalQuantile(
        sorted: List<Double>,
        probability: Double,
    ): Double {
        if (sorted.isEmpty()) return 1.0
        if (sorted.size == 1) return sorted.first()

        val p =
            probability.coerceIn(0.0, 1.0) *
                (sorted.size - 1)

        val lower = p.toInt()
        val upper =
            min(
                lower + 1,
                sorted.lastIndex,
            )

        val fraction = p - lower

        return sorted[lower] * (1.0 - fraction) +
            sorted[upper] * fraction
    }

    private fun clearState(
        state: ChannelState,
    ) {
        state.scores.clear()
        state.multiplier = 1.0
        state.empiricalCoverage = 0.0
        state.acceptedSamples = 0
        state.rejectedSamples = 0
        state.lastTimestampNanos = Long.MIN_VALUE
    }

    private fun clampState(
        state: ChannelState,
    ) {
        state.multiplier =
            state.multiplier.coerceIn(
                config.multiplierFloor,
                config.multiplierCeiling,
            )

        while (
            state.scores.size >
                config.maximumCalibrationSamples
        ) {
            state.scores.removeFirst()
        }
    }
}
