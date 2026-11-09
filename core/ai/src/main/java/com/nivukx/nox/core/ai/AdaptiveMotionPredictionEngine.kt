package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Adaptive constant-acceleration predictor for image-space human tracking.
 *
 * The predictor is deliberately deterministic and bounded. It models position, velocity and
 * acceleration and adapts its correction gains from innovation magnitude and observation quality.
 * Large innovations reduce trust in the prior state instead of allowing stale predictions to drag an
 * identity through a crowded crossing.
 */
class AdaptiveMotionPredictionEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val positionGain: Double = 0.72,
        val velocityGain: Double = 0.48,
        val accelerationGain: Double = 0.24,
        val minimumDtSeconds: Double = 1.0 / 240.0,
        val maximumDtSeconds: Double = 0.20,
        val maximumPredictionSeconds: Double = 0.30,
        val maximumSpeedNormalizedPerSecond: Double = 5.0,
        val maximumAccelerationNormalizedPerSecond2: Double = 30.0,
        val maximumJerkNormalizedPerSecond3: Double = 140.0,
        val innovationScaleNormalized: Double = 0.12,
        val minimumObservationConfidence: Double = 0.08,
        val initialPositionSigma: Double = 0.018,
        val processAccelerationSigma: Double = 1.8,
        val maximumPositionSigma: Double = 0.35,
    ) {
        init {
            require(positionGain in 0.0..1.0)
            require(velocityGain in 0.0..1.0)
            require(accelerationGain in 0.0..1.0)
            require(minimumDtSeconds > 0.0)
            require(maximumDtSeconds >= minimumDtSeconds)
            require(maximumPredictionSeconds >= maximumDtSeconds)
            require(maximumSpeedNormalizedPerSecond > 0.0)
            require(maximumAccelerationNormalizedPerSecond2 > 0.0)
            require(maximumJerkNormalizedPerSecond3 > 0.0)
            require(innovationScaleNormalized > 0.0)
            require(minimumObservationConfidence in 0.0..1.0)
            require(initialPositionSigma > 0.0)
            require(processAccelerationSigma > 0.0)
            require(maximumPositionSigma > initialPositionSigma)
        }
    }

    data class Prediction(
        val position: Vec2,
        val velocity: Vec2,
        val acceleration: Vec2,
        val positionSigma: Double,
        val confidence: Double,
    )

    data class UpdateResult(
        val predictionBeforeUpdate: Prediction,
        val state: Prediction,
        val innovation: Double,
        val correctionWeight: Double,
        val acceleratedMotion: Boolean,
    )

    private var initialized = false
    private var position = Vec2(0.0, 0.0)
    private var velocity = Vec2(0.0, 0.0)
    private var acceleration = Vec2(0.0, 0.0)
    private var positionSigma = config.initialPositionSigma
    private var confidence = 0.0
    private var lastTimestampNanos = Long.MIN_VALUE
    private var lastObservedVelocity = Vec2(0.0, 0.0)
    private var lastJerkNorm = 0.0

    @Synchronized
    fun reset() {
        initialized = false
        position = Vec2(0.0, 0.0)
        velocity = Vec2(0.0, 0.0)
        acceleration = Vec2(0.0, 0.0)
        positionSigma = config.initialPositionSigma
        confidence = 0.0
        lastTimestampNanos = Long.MIN_VALUE
        lastObservedVelocity = Vec2(0.0, 0.0)
        lastJerkNorm = 0.0
    }

    @Synchronized
    fun initialize(
        measurement: Vec2,
        timestampNanos: Long,
        observationConfidence: Double,
    ): Prediction {
        require(measurement.isFinite())
        require(timestampNanos >= 0L)
        val quality = observationConfidence.coerceIn(0.0, 1.0)

        initialized = true
        position = measurement
        velocity = Vec2(0.0, 0.0)
        acceleration = Vec2(0.0, 0.0)
        positionSigma = lerp(
            config.maximumPositionSigma,
            config.initialPositionSigma,
            quality,
        )
        confidence = quality
        lastTimestampNanos = timestampNanos
        lastObservedVelocity = Vec2(0.0, 0.0)
        lastJerkNorm = 0.0

        return snapshot()
    }

    @Synchronized
    fun predict(timestampNanos: Long): Prediction {
        require(timestampNanos >= 0L)

        if (!initialized) {
            return Prediction(
                position = position,
                velocity = velocity,
                acceleration = acceleration,
                positionSigma = config.maximumPositionSigma,
                confidence = 0.0,
            )
        }

        val dt = dtSeconds(timestampNanos)
        val boundedDt = dt.coerceIn(
            0.0,
            config.maximumPredictionSeconds,
        )

        val predictedPosition =
            position +
                velocity * boundedDt +
                acceleration * (0.5 * boundedDt * boundedDt)

        val predictedVelocity =
            velocity +
                acceleration * boundedDt

        val positionGrowth =
            config.processAccelerationSigma *
                0.5 *
                boundedDt *
                boundedDt

        return Prediction(
            position = predictedPosition,
            velocity = clampNorm(
                predictedVelocity,
                config.maximumSpeedNormalizedPerSecond,
            ),
            acceleration = clampNorm(
                acceleration,
                config.maximumAccelerationNormalizedPerSecond2,
            ),
            positionSigma = (
                positionSigma +
                    positionGrowth
            ).coerceAtMost(config.maximumPositionSigma),
            confidence = (
                confidence *
                    exp(
                        -boundedDt /
                            0.32
                    )
                ).coerceIn(0.0, 1.0),
        )
    }

    @Synchronized
    fun update(
        measurement: Vec2,
        timestampNanos: Long,
        observationConfidence: Double,
    ): UpdateResult {
        require(measurement.isFinite())
        require(timestampNanos >= 0L)
        val quality = observationConfidence.coerceIn(0.0, 1.0)

        if (!initialized) {
            val state = initialize(
                measurement = measurement,
                timestampNanos = timestampNanos,
                observationConfidence = quality,
            )
            return UpdateResult(
                predictionBeforeUpdate = state,
                state = state,
                innovation = 0.0,
                correctionWeight = 1.0,
                acceleratedMotion = false,
            )
        }

        val prediction = predict(timestampNanos)
        val dt = dtSeconds(timestampNanos)
            .coerceIn(
                config.minimumDtSeconds,
                config.maximumDtSeconds,
            )

        val innovationVector =
            measurement -
                prediction.position
        val innovation =
            innovationVector.norm()

        /*
         * Huber-like adaptive weighting. Strong observations with small innovations get normal
         * gains. A large innovation automatically trusts the measurement more and the prior less.
         */
        val innovationWeight =
            (
                1.0 -
                    (
                        innovation /
                            config.innovationScaleNormalized
                    ).coerceIn(0.0, 1.0) * 0.58
                ) *
                (
                    0.35 +
                        0.65 * quality
                    )
        val correctionWeight =
            innovationWeight.coerceIn(0.12, 0.94)

        val correctedPosition =
            prediction.position +
                innovationVector *
                (
                    config.positionGain *
                        correctionWeight
                ).coerceIn(0.06, 0.92)

        val measuredVelocity =
            (
                measurement -
                    position
                ) / dt

        val correctedVelocity =
            prediction.velocity.lerp(
                clampNorm(
                    measuredVelocity,
                    config.maximumSpeedNormalizedPerSecond,
                ),
                (
                    config.velocityGain *
                        correctionWeight
                    ).coerceIn(0.08, 0.85),
            )

        val velocityChange =
            correctedVelocity -
                lastObservedVelocity
        val inferredAcceleration =
            velocityChange / dt

        val correctedAcceleration =
            prediction.acceleration.lerp(
                clampNorm(
                    inferredAcceleration,
                    config.maximumAccelerationNormalizedPerSecond2,
                ),
                (
                    config.accelerationGain *
                        correctionWeight
                    ).coerceIn(0.05, 0.62),
            )

        val jerk =
            (
                correctedAcceleration -
                    acceleration
            ).norm() /
                dt
        lastJerkNorm =
            jerk.coerceAtMost(
                config.maximumJerkNormalizedPerSecond3,
            )

        position = correctedPosition
        velocity = clampNorm(
            correctedVelocity,
            config.maximumSpeedNormalizedPerSecond,
        )
        acceleration = clampNorm(
            correctedAcceleration,
            config.maximumAccelerationNormalizedPerSecond2,
        )
        lastObservedVelocity = velocity
        lastTimestampNanos = timestampNanos

        val measurementSigma =
            (
                0.002 +
                    (
                        1.0 -
                            quality
                    ) * 0.055 +
                    innovation * 0.12
                ).coerceIn(
                0.002,
                config.maximumPositionSigma,
            )

        positionSigma =
            (
                prediction.positionSigma * 0.55 +
                    measurementSigma * 0.45
                ).coerceIn(
                0.001,
                config.maximumPositionSigma,
            )

        confidence =
            (
                prediction.confidence * 0.24 +
                    quality * 0.54 +
                    correctionWeight * 0.22
                ).coerceIn(0.0, 1.0)

        return UpdateResult(
            predictionBeforeUpdate = prediction,
            state = snapshot(),
            innovation = innovation,
            correctionWeight = correctionWeight,
            acceleratedMotion =
                lastJerkNorm >
                    config.maximumJerkNormalizedPerSecond3 * 0.20,
        )
    }

    @Synchronized
    fun propagateOcclusion(timestampNanos: Long): Prediction {
        val prediction = predict(timestampNanos)
        if (!initialized) return prediction

        val bounded = timestampNanos.coerceAtLeast(
            lastTimestampNanos,
        )
        val dt = (
            bounded -
                lastTimestampNanos
            ).toDouble() * 1e-9
        val safeDt = dt.coerceIn(
            0.0,
            config.maximumPredictionSeconds,
        )

        position =
            prediction.position
        velocity =
            prediction.velocity
        acceleration =
            prediction.acceleration * 0.92
        positionSigma =
            (
                prediction.positionSigma +
                    0.012 * safeDt
                ).coerceAtMost(
                config.maximumPositionSigma,
            )
        confidence =
            (
                prediction.confidence * 0.93
            ).coerceAtLeast(0.0)
        lastTimestampNanos = timestampNanos.coerceAtLeast(
            lastTimestampNanos,
        )

        return snapshot()
    }

    @Synchronized
    fun snapshot(): Prediction =
        Prediction(
            position = position,
            velocity = velocity,
            acceleration = acceleration,
            positionSigma = positionSigma,
            confidence = confidence,
        )

    @Synchronized
    fun isInitialized(): Boolean = initialized

    private fun dtSeconds(timestampNanos: Long): Double {
        if (lastTimestampNanos == Long.MIN_VALUE) {
            return config.minimumDtSeconds
        }
        return (
            timestampNanos -
                lastTimestampNanos
            ).coerceAtLeast(0L) *
            1e-9
    }

    private fun clampNorm(
        value: Vec2,
        maximum: Double,
    ): Vec2 {
        val norm = value.norm()
        if (!norm.isFinite() || norm <= maximum) return value
        return value * (
            maximum /
                norm.coerceAtLeast(1e-12)
        )
    }

    private fun Vec2.lerp(
        target: Vec2,
        alpha: Double,
    ): Vec2 =
        this * (1.0 - alpha) +
            target * alpha

    private fun Vec2.isFinite(): Boolean =
        x.isFinite() && y.isFinite()

    private fun lerp(
        start: Double,
        end: Double,
        alpha: Double,
    ): Double =
        start * (1.0 - alpha) +
            end * alpha
}
