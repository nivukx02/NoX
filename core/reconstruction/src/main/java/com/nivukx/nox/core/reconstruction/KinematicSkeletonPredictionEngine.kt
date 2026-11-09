package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Bounded per-actor 3D skeleton predictor for high-density real-time capture.
 *
 * Prediction is a continuity mechanism, not a substitute for new observations. It extrapolates
 * only the most recent trusted skeleton for a short configurable horizon, inflates joint
 * uncertainty as the prediction horizon grows, and decays confidence monotonically.
 *
 * No neural hallucination is performed here. The predictor uses the velocity already supplied by
 * the reconstruction stage and a bounded acceleration estimate from two consecutive skeletons.
 */
class KinematicSkeletonPredictionEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumHistoryActors: Int = 64,
        val maximumPredictionHorizonMillis: Long = 180L,
        val minimumPredictionIntervalMillis: Long = 1L,
        val confidenceHalfLifeMillis: Double = 90.0,
        val accelerationSmoothing: Double = 0.28,
        val uncertaintyAccelerationScale: Double = 0.035,
        val uncertaintyTimeScale: Double = 0.008,
        val maximumConfidenceLoss: Double = 0.82,
        val maximumUncertaintyMeters: Double = 1.50,
    ) {
        init {
            require(maximumHistoryActors in 1..128)
            require(maximumPredictionHorizonMillis >= 1L)
            require(minimumPredictionIntervalMillis >= 1L)
            require(confidenceHalfLifeMillis > 0.0)
            require(accelerationSmoothing in 0.0..1.0)
            require(uncertaintyAccelerationScale >= 0.0)
            require(uncertaintyTimeScale >= 0.0)
            require(maximumConfidenceLoss in 0.0..1.0)
            require(maximumUncertaintyMeters > 0.0)
        }
    }

    data class Result(
        val skeleton: SkeletonFrame,
        val predicted: Boolean,
        val predictionAgeMillis: Long,
        val confidenceScale: Double,
        val meanUncertaintyInflationMeters: Double,
        val warnings: List<String>,
    )

    private data class State(
        var previous: SkeletonFrame,
        var latest: SkeletonFrame,
        var acceleration: Map<Int, Vec3>,
        var lastPredictionTimestampNanos: Long,
    )

    private val states = LinkedHashMap<Long, State>()

    @Synchronized
    fun observe(skeleton: SkeletonFrame) {
        val current = states[skeleton.trackId]

        if (
            current == null ||
            skeleton.timestampNanos > current.latest.timestampNanos
        ) {
            val acceleration =
                if (
                    current != null &&
                    skeleton.timestampNanos > current.latest.timestampNanos
                ) {
                    estimateAcceleration(
                        previous = current.latest,
                        current = skeleton,
                    )
                } else {
                    emptyMap()
                }

            val blendedAcceleration =
                when {
                    current == null -> acceleration
                    acceleration.isEmpty() -> current.acceleration
                    else ->
                        blendAcceleration(
                            previous = current.acceleration,
                            current = acceleration,
                        )
                }

            states[skeleton.trackId] =
                State(
                    previous = current?.latest ?: skeleton,
                    latest = skeleton,
                    acceleration = blendedAcceleration,
                    lastPredictionTimestampNanos =
                        skeleton.timestampNanos,
                )
        }

        trim()
    }

    @Synchronized
    fun predict(
        trackId: Long,
        targetTimestampNanos: Long,
        maximumPredictionHorizonMillis: Long =
            config.maximumPredictionHorizonMillis,
    ): Result? {
        require(trackId >= 0L)
        require(targetTimestampNanos >= 0L)
        require(maximumPredictionHorizonMillis >= 1L)

        val state = states[trackId] ?: return null
        val base = state.latest

        if (targetTimestampNanos <= base.timestampNanos) {
            return Result(
                skeleton = base,
                predicted = false,
                predictionAgeMillis = 0L,
                confidenceScale = 1.0,
                meanUncertaintyInflationMeters = 0.0,
                warnings = emptyList(),
            )
        }

        val ageMillis =
            (targetTimestampNanos - base.timestampNanos) / 1_000_000L

        if (ageMillis < config.minimumPredictionIntervalMillis) {
            return null
        }

        if (ageMillis > maximumPredictionHorizonMillis) {
            return Result(
                skeleton = base,
                predicted = true,
                predictionAgeMillis = ageMillis,
                confidenceScale = 0.0,
                meanUncertaintyInflationMeters =
                    config.maximumUncertaintyMeters,
                warnings = listOf(
                    "ACTOR_3D_PREDICTION_HORIZON_EXCEEDED",
                    "ACTOR_3D_PREDICTION_OUTPUT_HELD_WITH_ZERO_CONFIDENCE",
                ),
            )
        }

        val dt = ageMillis / 1000.0
        val confidenceScale =
            exp(
                -kotlin.math.ln(2.0) *
                    ageMillis /
                    config.confidenceHalfLifeMillis,
            ).coerceIn(
                1.0 - config.maximumConfidenceLoss,
                1.0,
            )

        val predictedJoints =
            base.joints.map { joint ->
                val acceleration =
                    state.acceleration[joint.landmarkId]
                        ?: Vec3(0.0, 0.0, 0.0)

                val displacement =
                    joint.velocity * dt +
                        acceleration *
                        (0.5 * dt * dt)

                val baseUncertainty =
                    joint.positionUncertaintyMeters
                        .takeIf { it.isFinite() && it > 0.0 }
                        ?: 0.006

                val kinematicGrowth =
                    acceleration.norm() *
                        dt *
                        config.uncertaintyAccelerationScale

                val temporalGrowth =
                    dt * config.uncertaintyTimeScale

                val uncertainty =
                    sqrt(
                        baseUncertainty * baseUncertainty +
                            kinematicGrowth * kinematicGrowth +
                            temporalGrowth * temporalGrowth,
                    ).coerceAtMost(
                        config.maximumUncertaintyMeters,
                    )

                joint.copy(
                    position = joint.position + displacement,
                    velocity = joint.velocity + acceleration * dt,
                    acceleration = acceleration,
                    confidence =
                        (joint.confidence * confidenceScale)
                            .coerceIn(0.0, 1.0),
                    positionUncertaintyMeters = uncertainty,
                )
            }

        val skeletonConfidence =
            (base.globalConfidence * confidenceScale)
                .coerceIn(0.0, 1.0)

        state.lastPredictionTimestampNanos = targetTimestampNanos

        val meanInflation =
            predictedJoints
                .map { predicted ->
                    val originalUncertainty =
                        base.joints
                            .firstOrNull { original ->
                                original.landmarkId == predicted.landmarkId
                            }
                            ?.positionUncertaintyMeters
                            ?: 0.0
                    (predicted.positionUncertaintyMeters -
                        originalUncertainty)
                        .coerceAtLeast(0.0)
                }
                .averageOrZero()

        return Result(
            skeleton =
                base.copy(
                    timestampNanos = targetTimestampNanos,
                    joints = predictedJoints,
                    globalConfidence = skeletonConfidence,
                ),
            predicted = true,
            predictionAgeMillis = ageMillis,
            confidenceScale = confidenceScale,
            meanUncertaintyInflationMeters = meanInflation,
            warnings = listOf("ACTOR_3D_SKELETON_PREDICTED"),
        )
    }

    @Synchronized
    fun forget(trackId: Long) {
        states.remove(trackId)
    }

    @Synchronized
    fun reset() {
        states.clear()
    }

    @Synchronized
    fun trackedActorIds(): List<Long> =
        states.keys.toList()

    private fun estimateAcceleration(
        previous: SkeletonFrame,
        current: SkeletonFrame,
    ): Map<Int, Vec3> {
        val dt =
            (
                current.timestampNanos -
                    previous.timestampNanos
            ) / 1e9

        if (dt <= 1e-6) return emptyMap()

        val previousById =
            previous.joints.associateBy {
                it.landmarkId
            }

        return current.joints.mapNotNull { joint ->
            val previousJoint =
                previousById[joint.landmarkId]
                    ?: return@mapNotNull null

            val delta =
                (
                    joint.velocity -
                        previousJoint.velocity
                    ) / dt

            if (!delta.x.isFinite() || !delta.y.isFinite() || !delta.z.isFinite()) {
                null
            } else {
                joint.landmarkId to delta
            }
        }.toMap()
    }

    private fun blendAcceleration(
        previous: Map<Int, Vec3>,
        current: Map<Int, Vec3>,
    ): Map<Int, Vec3> =
        current.mapValues { (landmarkId, value) ->
            val old = previous[landmarkId]
                ?: value
            old * (1.0 - config.accelerationSmoothing) +
                value * config.accelerationSmoothing
        }

    private fun trim() {
        while (states.size > config.maximumHistoryActors) {
            val victim =
                states.entries
                    .minByOrNull {
                        it.value.latest.timestampNanos
                    }?.key
                    ?: break
            states.remove(victim)
        }
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

