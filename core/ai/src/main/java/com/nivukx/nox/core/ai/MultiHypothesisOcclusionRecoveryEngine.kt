package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Bounded multi-hypothesis recovery for temporarily occluded actors.
 *
 * The engine keeps a small deterministic hypothesis set instead of committing immediately to one
 * extrapolation. Each hypothesis is physically interpretable: constant velocity, damped velocity,
 * constant acceleration and contact-biased vertical damping. New observations collapse the set using
 * robust residual likelihood. Hypotheses are bounded and never outrank directly observed evidence.
 */
class MultiHypothesisOcclusionRecoveryEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumHypotheses: Int = 4,
        val maximumGapSeconds: Double = 0.60,
        val maximumSpeedMetersPerSecond: Double = 12.0,
        val maximumAccelerationMetersPerSecond2: Double = 80.0,
        val positionalSigmaMeters: Double = 0.025,
        val robustCutoffSigma: Double = 3.0,
        val confidenceDecayPerSecond: Double = 0.86,
        val minimumConfidenceToRecover: Double = 0.12,
        val contactVerticalDamping: Double = 0.20,
        val accelerationDamping: Double = 0.75,
    ) {
        init {
            require(maximumHypotheses in 2..8)
            require(maximumGapSeconds > 0.0)
            require(maximumSpeedMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(positionalSigmaMeters > 0.0)
            require(robustCutoffSigma > 0.0)
            require(confidenceDecayPerSecond in 0.0..1.0)
            require(minimumConfidenceToRecover in 0.0..1.0)
            require(contactVerticalDamping in 0.0..1.0)
            require(accelerationDamping in 0.0..1.0)
        }
    }

    enum class MotionModel {
        CONSTANT_VELOCITY,
        DAMPED_VELOCITY,
        CONSTANT_ACCELERATION,
        CONTACT_BIASED,
    }

    data class TrackState(
        val actorId: Long,
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val accelerationMetersPerSecond2: Vec3 = Vec3(0.0, 0.0, 0.0),
        val confidence: Double,
        val uncertaintyMeters: Double,
        val contactProbability: Double = 0.0,
    ) {
        init {
            require(actorId >= 0L)
            require(timestampNanos > 0L)
            require(positionMeters.finite())
            require(velocityMetersPerSecond.finite())
            require(accelerationMetersPerSecond2.finite())
            require(confidence in 0.0..1.0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
            require(contactProbability in 0.0..1.0)
        }
    }

    data class Observation(
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val confidence: Double,
        val uncertaintyMeters: Double,
    ) {
        init {
            require(timestampNanos > 0L)
            require(positionMeters.finite())
            require(confidence in 0.0..1.0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
        }
    }

    data class Hypothesis(
        val model: MotionModel,
        val state: TrackState,
        val likelihood: Double,
        val residualMeters: Double,
        val predictionAgeSeconds: Double,
    )

    data class Result(
        val actorId: Long,
        val timestampNanos: Long,
        val state: TrackState?,
        val hypotheses: List<Hypothesis>,
        val recovered: Boolean,
        val ambiguous: Boolean,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val warnings: List<String>,
    )

    fun predict(
        track: TrackState,
        timestampNanos: Long,
        observation: Observation? = null,
    ): Result {
        require(timestampNanos >= track.timestampNanos)

        val gap =
            (
                timestampNanos -
                    track.timestampNanos
                ).toDouble() /
                1_000_000_000.0

        val boundedGap =
            gap.coerceIn(
                0.0,
                config.maximumGapSeconds,
            )

        val generated =
            listOf(
                generate(
                    track,
                    boundedGap,
                    MotionModel.CONSTANT_VELOCITY,
                ),
                generate(
                    track,
                    boundedGap,
                    MotionModel.DAMPED_VELOCITY,
                ),
                generate(
                    track,
                    boundedGap,
                    MotionModel.CONSTANT_ACCELERATION,
                ),
                generate(
                    track,
                    boundedGap,
                    MotionModel.CONTACT_BIASED,
                ),
            )

        val scored =
            generated
                .map { hypothesis ->
                    val residual =
                        observation?.let {
                            hypothesis.state.positionMeters
                                .distanceTo(it.positionMeters)
                        }
                            ?: 0.0

                    val observationVariance =
                        observation
                            ?.uncertaintyMeters
                            ?.let { it * it }
                            ?: 0.0

                    val combinedSigma =
                        sqrt(
                            config.positionalSigmaMeters *
                                config.positionalSigmaMeters +
                                hypothesis.state.uncertaintyMeters *
                                    hypothesis.state.uncertaintyMeters +
                                observationVariance,
                        )

                    val normalized =
                        residual /
                            max(
                                combinedSigma,
                                1e-6,
                            )

                    val robust =
                        if (normalized <= config.robustCutoffSigma) {
                            1.0 -
                                0.5 *
                                (
                                    normalized /
                                        config.robustCutoffSigma
                                )
                                .let { value -> value * value }
                        } else {
                            (
                                0.5 *
                                    config.robustCutoffSigma /
                                    max(
                                        normalized,
                                        1e-6,
                                    )
                            )
                                .coerceAtLeast(0.02)
                        }

                    val modelPrior =
                        when (hypothesis.model) {
                            MotionModel.CONSTANT_VELOCITY -> 1.00
                            MotionModel.DAMPED_VELOCITY -> 0.96
                            MotionModel.CONSTANT_ACCELERATION -> 0.88
                            MotionModel.CONTACT_BIASED -> 0.90
                        }

                    val likelihood =
                        (
                            robust *
                                modelPrior *
                                track.confidence *
                                kotlin.math.exp(
                                    -boundedGap /
                                        max(
                                            config.maximumGapSeconds,
                                            1e-6,
                                        ),
                                )
                            )
                            .coerceIn(0.0, 1.0)

                    hypothesis.copy(
                        likelihood = likelihood,
                        residualMeters = residual,
                    )
                }
                .sortedWith(
                    compareByDescending<Hypothesis> {
                        it.likelihood
                    }.thenBy {
                        it.model.ordinal
                    },
                )
                .take(config.maximumHypotheses)

        val top = scored.firstOrNull()
        val second = scored.getOrNull(1)

        val ambiguous =
            top != null &&
                second != null &&
                abs(
                    top.likelihood -
                        second.likelihood,
                ) < 0.05

        val recovered =
            gap <= config.maximumGapSeconds &&
                top != null &&
                top.likelihood >=
                    config.minimumConfidenceToRecover &&
                track.confidence >=
                    config.minimumConfidenceToRecover &&
                !(
                    observation != null &&
                        top.residualMeters >
                            config.positionalSigmaMeters *
                                config.robustCutoffSigma *
                                2.0
                    )

        val finalState =
            when {
                observation != null &&
                    recovered &&
                    top != null ->
                    top.state.copy(
                        positionMeters =
                            top.state.positionMeters.lerp(
                                observation.positionMeters,
                                blendFactor(
                                    top.likelihood,
                                    observation.confidence,
                                ),
                            ),
                        confidence =
                            (
                                top.state.confidence *
                                    0.55 +
                                    observation.confidence *
                                        0.45
                                )
                                .coerceIn(0.0, 1.0),
                        uncertaintyMeters =
                            max(
                                1e-6,
                                (
                                    top.state.uncertaintyMeters +
                                        observation.uncertaintyMeters
                                    ) * 0.5,
                            ),
                        timestampNanos =
                            observation.timestampNanos,
                    )
                recovered && top != null ->
                    top.state
                else ->
                    null
            }

        val warnings =
            buildList {
                if (gap > config.maximumGapSeconds) {
                    add("occlusion-gap-exceeds-recovery-window")
                }
                if (ambiguous) {
                    add("occlusion-motion-hypotheses-ambiguous")
                }
                if (!recovered) {
                    add("occlusion-recovery-not-authoritative")
                }
                if (observation == null && top != null) {
                    add("occlusion-state-is-predicted")
                }
            }

        return Result(
            actorId = track.actorId,
            timestampNanos = timestampNanos,
            state = finalState,
            hypotheses = scored,
            recovered = recovered,
            ambiguous = ambiguous,
            confidence =
                finalState?.confidence ?: 0.0,
            uncertaintyMeters =
                finalState?.uncertaintyMeters
                    ?: Double.POSITIVE_INFINITY,
            warnings = warnings.distinct(),
        )
    }

    private fun generate(
        track: TrackState,
        dt: Double,
        model: MotionModel,
    ): Hypothesis {
        val velocity =
            when (model) {
                MotionModel.CONSTANT_VELOCITY ->
                    track.velocityMetersPerSecond
                MotionModel.DAMPED_VELOCITY ->
                    track.velocityMetersPerSecond *
                        kotlin.math.exp(
                            -1.8 * dt,
                        )
                MotionModel.CONSTANT_ACCELERATION ->
                    (
                        track.velocityMetersPerSecond +
                            track.accelerationMetersPerSecond2 * dt
                        )
                MotionModel.CONTACT_BIASED ->
                    Vec3(
                        track.velocityMetersPerSecond.x,
                        track.velocityMetersPerSecond.y *
                            (
                                1.0 -
                                    config.contactVerticalDamping *
                                    track.contactProbability
                            ),
                        track.velocityMetersPerSecond.z,
                    )
            }
                .clampedMagnitude(
                    config.maximumSpeedMetersPerSecond,
                )

        val acceleration =
            when (model) {
                MotionModel.CONSTANT_ACCELERATION ->
                    track.accelerationMetersPerSecond2
                        .clampedMagnitude(
                            config.maximumAccelerationMetersPerSecond2,
                        )
                else ->
                    track.accelerationMetersPerSecond2 *
                        config.accelerationDamping
            }

        val position =
            track.positionMeters +
                velocity * dt +
                acceleration *
                    (0.5 * dt * dt)

        val uncertaintyGrowth =
            config.positionalSigmaMeters * (
                1.0 +
                    dt *
                        (
                            1.0 +
                                track.velocityMetersPerSecond.norm()
                        )
                )

        return Hypothesis(
            model = model,
            state =
                track.copy(
                    positionMeters = position,
                    velocityMetersPerSecond = velocity,
                    accelerationMetersPerSecond2 = acceleration,
                    confidence =
                        (
                            track.confidence *
                                kotlin.math.exp(
                                    kotlin.math.ln(
                                        max(
                                            config.confidenceDecayPerSecond,
                                            1e-6,
                                        ),
                                    ) *
                                        dt,
                                )
                            )
                            .coerceIn(0.0, 1.0),
                    uncertaintyMeters =
                        max(
                            track.uncertaintyMeters,
                            uncertaintyGrowth,
                        ),
                    timestampNanos =
                        (
                            track.timestampNanos +
                                (dt *
                                    1_000_000_000.0)
                                    .toLong()
                            ),
                ),
            likelihood = 0.0,
            residualMeters = 0.0,
            predictionAgeSeconds = dt,
        )
    }

    private fun blendFactor(
        hypothesisLikelihood: Double,
        observationConfidence: Double,
    ): Double =
        (
            observationConfidence /
                max(
                    observationConfidence +
                        hypothesisLikelihood,
                    1e-6,
                )
            )
            .coerceIn(0.05, 0.85)

    private fun Vec3.lerp(
        target: Vec3,
        alpha: Double,
    ): Vec3 =
        this * (1.0 - alpha) +
            target * alpha

    private fun Vec3.clampedMagnitude(
        maximum: Double,
    ): Vec3 {
        val magnitude = norm()
        if (!magnitude.isFinite() || magnitude <= maximum) return this
        return this *
            (
                maximum /
                    max(
                        magnitude,
                        1e-9,
                    )
            )
    }

    private fun Vec3.finite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()
}
