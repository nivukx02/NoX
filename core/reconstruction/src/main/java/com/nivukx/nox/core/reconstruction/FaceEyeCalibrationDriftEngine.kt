package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Slow-drift detector for face/eye calibration.
 *
 * Frame-level fusion answers "is this frame trustworthy?" This engine answers a different question:
 * "is the calibration slowly becoming wrong even though individual frames still look plausible?"
 *
 * It tracks robust, bounded trends for:
 * - interpupillary distance,
 * - binocular angular disagreement,
 * - face reprojection error,
 * - binocular confidence,
 * - fused face/eye confidence.
 *
 * A single pathological frame is treated as an incident, not as a new calibration baseline.
 * Persistent drift produces a recalibration recommendation without inventing corrected geometry.
 */
class FaceEyeCalibrationDriftEngine(
    private var config: Config = Config(),
) {

    enum class State {
        WARMING,
        STABLE,
        DRIFTING,
        PERSISTENT_DRIFT,
        RECOVERING,
    }

    data class Config(
        val warmupSamples: Int = 8,
        val historyAlpha: Double = 0.16,
        val residualAlpha: Double = 0.24,
        val stableIpdMeters: Double = 0.064,
        val expectedIpdFloorMeters: Double = 0.045,
        val expectedIpdCeilingMeters: Double = 0.095,
        val warningIpdDriftMeters: Double = 0.007,
        val hardIpdDriftMeters: Double = 0.016,
        val warningAngularDisagreementDegrees: Double = 6.0,
        val hardAngularDisagreementDegrees: Double = 14.0,
        val warningReprojectionErrorPx: Double = 4.0,
        val hardReprojectionErrorPx: Double = 9.0,
        val warningBinocularConfidence: Double = 0.62,
        val hardBinocularConfidence: Double = 0.30,
        val warningFusionConfidence: Double = 0.64,
        val hardFusionConfidence: Double = 0.32,
        val warningDriftScore: Double = 0.42,
        val persistentDriftScore: Double = 0.62,
        val persistentWarningFrames: Int = 4,
        val recoveryFrames: Int = 5,
        val minimumHealthyConfidence: Double = 0.64,
        val singleFramePoisonResistance: Double = 0.34,
    ) {
        init {
            require(warmupSamples >= 2)
            require(historyAlpha in 0.0..1.0)
            require(residualAlpha in 0.0..1.0)
            require(stableIpdMeters > 0.0)
            require(expectedIpdFloorMeters > 0.0)
            require(expectedIpdCeilingMeters > expectedIpdFloorMeters)
            require(
                warningIpdDriftMeters > 0.0 &&
                    hardIpdDriftMeters > warningIpdDriftMeters,
            )
            require(
                warningAngularDisagreementDegrees > 0.0 &&
                    hardAngularDisagreementDegrees >
                    warningAngularDisagreementDegrees,
            )
            require(
                warningReprojectionErrorPx > 0.0 &&
                    hardReprojectionErrorPx > warningReprojectionErrorPx,
            )
            require(
                warningBinocularConfidence in 0.0..1.0 &&
                    hardBinocularConfidence in 0.0..warningBinocularConfidence,
            )
            require(
                warningFusionConfidence in 0.0..1.0 &&
                    hardFusionConfidence in 0.0..warningFusionConfidence,
            )
            require(
                warningDriftScore in 0.0..1.0 &&
                    persistentDriftScore >= warningDriftScore &&
                    persistentDriftScore <= 1.0,
            )
            require(persistentWarningFrames >= 1)
            require(recoveryFrames >= 1)
            require(minimumHealthyConfidence in 0.0..1.0)
            require(singleFramePoisonResistance in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val interpupillaryDistanceMeters: Double,
        val angularDisagreementDegrees: Double,
        val reprojectionErrorPx: Double,
        val binocularConfidence: Double,
        val faceEyeFusionConfidence: Double,
    ) {
        init {
            require(timestampNanos > 0L)
            require(
                interpupillaryDistanceMeters.isFinite() &&
                    interpupillaryDistanceMeters >= 0.0,
            )
            require(
                angularDisagreementDegrees.isFinite() &&
                    angularDisagreementDegrees >= 0.0,
            )
            require(
                reprojectionErrorPx.isFinite() &&
                    reprojectionErrorPx >= 0.0,
            )
            require(binocularConfidence in 0.0..1.0)
            require(faceEyeFusionConfidence in 0.0..1.0)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val driftScore: Double,
        val ipdDriftMeters: Double,
        val angularDisagreementDegrees: Double,
        val reprojectionErrorPx: Double,
        val binocularConfidence: Double,
        val faceEyeFusionConfidence: Double,
        val baselineIpdMeters: Double,
        val warmupProgress: Double,
        val persistent: Boolean,
        val recalibrationRecommended: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class StateData(
        var timestampNanos: Long = Long.MIN_VALUE,
        var baselineIpdMeters: Double = 0.0,
        var ipdSamples: Int = 0,
        var driftEma: Double = 0.0,
        var angularEma: Double = 0.0,
        var reprojectionEma: Double = 0.0,
        var binocularConfidenceEma: Double = 1.0,
        var fusionConfidenceEma: Double = 1.0,
        var persistentFrames: Int = 0,
        var recoveryFrames: Int = 0,
        var state: State = State.WARMING,
    )

    private val state = StateData()

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig

        state.baselineIpdMeters =
            state.baselineIpdMeters.takeIf { it > 0.0 }
                ?: newConfig.stableIpdMeters
    }

    fun configuration(): Config = config

    @Synchronized
    fun evaluate(input: Input): Result {
        if (
            state.timestampNanos != Long.MIN_VALUE &&
            input.timestampNanos <= state.timestampNanos
        ) {
            return duplicateTimestampResult()
        }

        val previousBaseline =
            state.baselineIpdMeters.takeIf { it > 0.0 }
                ?: config.stableIpdMeters

        val calibrationQuality =
            calibrationQuality(input)

        if (
            state.ipdSamples < config.warmupSamples &&
            calibrationQuality >= config.minimumHealthyConfidence &&
            input.interpupillaryDistanceMeters in
            config.expectedIpdFloorMeters..config.expectedIpdCeilingMeters
        ) {
            val alpha =
                1.0 /
                    (state.ipdSamples + 1).toDouble()

            state.baselineIpdMeters =
                if (state.ipdSamples == 0) {
                    input.interpupillaryDistanceMeters
                } else {
                    state.baselineIpdMeters * (1.0 - alpha) +
                        input.interpupillaryDistanceMeters * alpha
                }

            state.ipdSamples += 1
        }

        val baseline =
            state.baselineIpdMeters.takeIf { it > 0.0 }
                ?: previousBaseline

        val ipdDrift =
            if (
                input.interpupillaryDistanceMeters <= 0.0 ||
                state.ipdSamples == 0
            ) {
                0.0
            } else {
                abs(
                    input.interpupillaryDistanceMeters -
                        baseline,
                )
            }

        val ipdScore =
            thresholdScore(
                value = ipdDrift,
                warning = config.warningIpdDriftMeters,
                hard = config.hardIpdDriftMeters,
                inverse = true,
            )

        val angularScore =
            thresholdScore(
                value = input.angularDisagreementDegrees,
                warning = config.warningAngularDisagreementDegrees,
                hard = config.hardAngularDisagreementDegrees,
                inverse = true,
            )

        val reprojectionScore =
            thresholdScore(
                value = input.reprojectionErrorPx,
                warning = config.warningReprojectionErrorPx,
                hard = config.hardReprojectionErrorPx,
                inverse = true,
            )

        val binocularScore =
            thresholdScore(
                value = input.binocularConfidence,
                warning = config.warningBinocularConfidence,
                hard = config.hardBinocularConfidence,
                inverse = false,
            )

        val fusionScore =
            thresholdScore(
                value = input.faceEyeFusionConfidence,
                warning = config.warningFusionConfidence,
                hard = config.hardFusionConfidence,
                inverse = false,
            )

        val frameDrift =
            (
                (1.0 - ipdScore) * 0.30 +
                    (1.0 - angularScore) * 0.22 +
                    (1.0 - reprojectionScore) * 0.18 +
                    (1.0 - binocularScore) * 0.14 +
                    (1.0 - fusionScore) * 0.16
                ).coerceIn(0.0, 1.0)

        val hardIncident =
            ipdScore < 0.15 ||
                angularScore < 0.15 ||
                reprojectionScore < 0.15 ||
                binocularScore < 0.15 ||
                fusionScore < 0.15

        val updateScale =
            if (hardIncident) {
                config.singleFramePoisonResistance
            } else {
                1.0
            }

        val baseAlpha =
            config.historyAlpha * updateScale
        val effectiveAlpha =
            if (frameDrift >= config.warningDriftScore) {
                max(baseAlpha, 0.62)
            } else {
                baseAlpha
            }

        state.driftEma =
            state.driftEma * (1.0 - effectiveAlpha) +
                frameDrift * effectiveAlpha

        state.angularEma =
            state.angularEma * (1.0 - config.residualAlpha) +
                input.angularDisagreementDegrees *
                config.residualAlpha

        state.reprojectionEma =
            state.reprojectionEma * (1.0 - config.residualAlpha) +
                input.reprojectionErrorPx *
                config.residualAlpha

        state.binocularConfidenceEma =
            state.binocularConfidenceEma *
                (1.0 - config.residualAlpha) +
                input.binocularConfidence *
                config.residualAlpha

        state.fusionConfidenceEma =
            state.fusionConfidenceEma *
                (1.0 - config.residualAlpha) +
                input.faceEyeFusionConfidence *
                config.residualAlpha

        val drifting =
            frameDrift >= config.warningDriftScore ||
                state.driftEma >= config.warningDriftScore

        val persistentNow =
            state.driftEma >= config.persistentDriftScore ||
                (
                    drifting &&
                        state.persistentFrames >=
                        config.persistentWarningFrames
                    )

        if (drifting) {
            state.persistentFrames += 1
            state.recoveryFrames = 0
        } else {
            state.persistentFrames =
                max(0, state.persistentFrames - 1)
            if (
                state.driftEma <
                    config.warningDriftScore * 0.68
            ) {
                state.recoveryFrames += 1
            } else {
                state.recoveryFrames = 0
            }
        }

        if (
            state.recoveryFrames >= config.recoveryFrames &&
            state.driftEma <
                config.warningDriftScore * 0.68
        ) {
            state.persistentFrames = 0
        }

        val nextState =
            when {
                state.ipdSamples <
                    config.warmupSamples ->
                    State.WARMING

                persistentNow ->
                    State.PERSISTENT_DRIFT

                drifting ->
                    if (state.recoveryFrames > 0) {
                        State.RECOVERING
                    } else {
                        State.DRIFTING
                    }

                else ->
                    State.STABLE
            }

        state.state = nextState
        state.timestampNanos = input.timestampNanos

        val recommended =
            nextState == State.PERSISTENT_DRIFT ||
                state.driftEma >=
                    config.persistentDriftScore ||
                (
                    ipdDrift >= config.hardIpdDriftMeters &&
                        input.binocularConfidence <
                        config.warningBinocularConfidence
                    )

        return Result(
            timestampNanos = input.timestampNanos,
            state = nextState,
            driftScore = state.driftEma.coerceIn(0.0, 1.0),
            ipdDriftMeters = ipdDrift,
            angularDisagreementDegrees =
                state.angularEma.coerceAtLeast(0.0),
            reprojectionErrorPx =
                state.reprojectionEma.coerceAtLeast(0.0),
            binocularConfidence =
                state.binocularConfidenceEma.coerceIn(0.0, 1.0),
            faceEyeFusionConfidence =
                state.fusionConfidenceEma.coerceIn(0.0, 1.0),
            baselineIpdMeters = baseline,
            warmupProgress =
                (
                    state.ipdSamples.toDouble() /
                        config.warmupSamples.toDouble()
                    ).coerceIn(0.0, 1.0),
            persistent = persistentNow,
            recalibrationRecommended = recommended,
            warnings = buildWarnings(
                nextState = nextState,
                driftScore = state.driftEma,
                ipdDriftMeters = ipdDrift,
                angularDisagreementDegrees =
                    state.angularEma,
                reprojectionErrorPx =
                    state.reprojectionEma,
                binocularConfidence =
                    state.binocularConfidenceEma,
            ),
            recommendations = buildRecommendations(
                nextState = nextState,
                driftScore = state.driftEma,
                ipdDriftMeters = ipdDrift,
                angularDisagreementDegrees =
                    state.angularEma,
                reprojectionErrorPx =
                    state.reprojectionEma,
                binocularConfidence =
                    state.binocularConfidenceEma,
            ),
        )
    }

    @Synchronized
    fun reset() {
        state.timestampNanos = Long.MIN_VALUE
        state.baselineIpdMeters = 0.0
        state.ipdSamples = 0
        state.driftEma = 0.0
        state.angularEma = 0.0
        state.reprojectionEma = 0.0
        state.binocularConfidenceEma = 1.0
        state.fusionConfidenceEma = 1.0
        state.persistentFrames = 0
        state.recoveryFrames = 0
        state.state = State.WARMING
    }

    private fun calibrationQuality(
        input: Input,
    ): Double =
        (
            input.binocularConfidence * 0.34 +
                input.faceEyeFusionConfidence * 0.36 +
                thresholdScore(
                    value = input.angularDisagreementDegrees,
                    warning = config.warningAngularDisagreementDegrees,
                    hard = config.hardAngularDisagreementDegrees,
                    inverse = true,
                ) * 0.16 +
                thresholdScore(
                    value = input.reprojectionErrorPx,
                    warning = config.warningReprojectionErrorPx,
                    hard = config.hardReprojectionErrorPx,
                    inverse = true,
                ) * 0.14
            ).coerceIn(0.0, 1.0)

    private fun thresholdScore(
        value: Double,
        warning: Double,
        hard: Double,
        inverse: Boolean,
    ): Double {
        if (!value.isFinite()) return 0.0

        if (inverse) {
            if (value <= warning) return 1.0
            if (value >= hard) return 0.0
            return (
                1.0 -
                    (
                        value - warning
                    ) /
                    (hard - warning).coerceAtLeast(1e-9)
                ).coerceIn(0.0, 1.0)
        }

        if (value >= warning) return 1.0
        if (value <= hard) return 0.0

        return (
            (
                value - hard
            ) /
                (warning - hard).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun buildWarnings(
        nextState: State,
        driftScore: Double,
        ipdDriftMeters: Double,
        angularDisagreementDegrees: Double,
        reprojectionErrorPx: Double,
        binocularConfidence: Double,
    ): List<String> = buildList {
        if (driftScore >= config.warningDriftScore) {
            add("face-eye-calibration-drift-elevated")
        }
        if (driftScore >= config.persistentDriftScore) {
            add("face-eye-calibration-drift-persistent")
        }
        if (ipdDriftMeters >= config.warningIpdDriftMeters) {
            add("face-eye-ipd-drift-elevated")
        }
        if (
            ipdDriftMeters >= config.hardIpdDriftMeters
        ) {
            add("face-eye-ipd-drift-hard-failure")
        }
        if (
            angularDisagreementDegrees >=
            config.warningAngularDisagreementDegrees
        ) {
            add("face-eye-angular-drift-elevated")
        }
        if (
            reprojectionErrorPx >=
            config.warningReprojectionErrorPx
        ) {
            add("face-eye-reprojection-drift-elevated")
        }
        if (
            binocularConfidence <=
            config.warningBinocularConfidence
        ) {
            add("face-eye-binocular-confidence-degraded")
        }
        if (
            nextState == State.PERSISTENT_DRIFT
        ) {
            add("face-eye-recalibration-recommended")
        }
        if (nextState == State.RECOVERING) {
            add("face-eye-calibration-recovering")
        }
    }.distinct()

    private fun buildRecommendations(
        nextState: State,
        driftScore: Double,
        ipdDriftMeters: Double,
        angularDisagreementDegrees: Double,
        reprojectionErrorPx: Double,
        binocularConfidence: Double,
    ): List<String> = buildList {
        if (
            nextState == State.PERSISTENT_DRIFT ||
            driftScore >= config.persistentDriftScore
        ) {
            add(
                "re-run face/eye geometric calibration before committing a long metric segment",
            )
        }
        if (ipdDriftMeters >= config.warningIpdDriftMeters) {
            add(
                "verify camera baseline, face scale and inter-camera extrinsics",
            )
        }
        if (
            angularDisagreementDegrees >=
            config.warningAngularDisagreementDegrees
        ) {
            add(
                "inspect binocular ray geometry and cross-camera timestamps",
            )
        }
        if (
            reprojectionErrorPx >=
            config.warningReprojectionErrorPx
        ) {
            add(
                "revalidate face-camera intrinsics and distortion parameters",
            )
        }
        if (
            binocularConfidence <=
            config.warningBinocularConfidence
        ) {
            add(
                "increase eye-camera coverage or lower eye influence until geometry recovers",
            )
        }
        if (nextState == State.RECOVERING) {
            add(
                "retain current calibration but require repeated healthy evidence before clearing the warning",
            )
        }
    }.distinct().take(8)

    private fun duplicateTimestampResult(): Result =
        Result(
            timestampNanos = state.timestampNanos,
            state = state.state,
            driftScore = state.driftEma.coerceIn(0.0, 1.0),
            ipdDriftMeters = 0.0,
            angularDisagreementDegrees =
                state.angularEma.coerceAtLeast(0.0),
            reprojectionErrorPx =
                state.reprojectionEma.coerceAtLeast(0.0),
            binocularConfidence =
                state.binocularConfidenceEma.coerceIn(0.0, 1.0),
            faceEyeFusionConfidence =
                state.fusionConfidenceEma.coerceIn(0.0, 1.0),
            baselineIpdMeters = state.baselineIpdMeters,
            warmupProgress =
                (
                    state.ipdSamples.toDouble() /
                        config.warmupSamples.toDouble()
                    ).coerceIn(0.0, 1.0),
            persistent =
                state.state == State.PERSISTENT_DRIFT,
            recalibrationRecommended =
                state.state == State.PERSISTENT_DRIFT,
            warnings = listOf(
                "face-eye-calibration-drift-duplicate-timestamp",
            ),
            recommendations = listOf(
                "discard duplicate face/eye calibration telemetry and retain the latest monotonic sample",
            ),
        )
}
