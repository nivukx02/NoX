package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Temporal filter for calibrated dense facial surfaces.
 *
 * The multi-view solver decides whether a point is geometrically valid. This engine decides how
 * that valid point should evolve over time. It uses a bounded constant-acceleration state per mesh
 * index, robust residual weighting, uncertainty propagation, and deformation-preserving confidence
 * blending.
 *
 * The filter is deliberately applied after metric triangulation. It never manufactures topology,
 * never fills an absent face, and never turns detector Z into metric depth. Missing points can be
 * carried for a bounded number of frames, after which they are dropped and reacquisition starts
 * from the measured geometry.
 */
class FaceMeshSurfaceTemporalEngine(
    private var config: Config = Config(),
) {

    data class Config(
        val minimumConfidence: Double = 0.16,
        val predictionHorizonSeconds: Double = 0.10,
        val maximumGapSeconds: Double = 0.18,
        val maximumOcclusionFrames: Int = 4,
        val softInnovationMeters: Double = 0.012,
        val hardInnovationMeters: Double = 0.055,
        val velocityLimitMetersPerSecond: Double = 4.0,
        val accelerationLimitMetersPerSecond2: Double = 20.0,
        val jerkLimitMetersPerSecond3: Double = 120.0,
        val positionBlendMin: Double = 0.18,
        val positionBlendMax: Double = 0.92,
        val velocityGain: Double = 0.62,
        val accelerationGain: Double = 0.24,
        val microMotionPreservation: Double = 0.92,
        val uncertaintyGrowthPerFrame: Double = 0.003,
        val uncertaintyGrowthPerSecond: Double = 0.010,
        val uncertaintyFloorMeters: Double = 0.0008,
        val uncertaintyCeilingMeters: Double = 0.080,
    ) {
        init {
            require(minimumConfidence in 0.0..1.0)
            require(predictionHorizonSeconds in 0.02..0.30)
            require(maximumGapSeconds > 0.0)
            require(maximumOcclusionFrames >= 0)
            require(softInnovationMeters > 0.0)
            require(hardInnovationMeters > softInnovationMeters)
            require(velocityLimitMetersPerSecond > 0.0)
            require(accelerationLimitMetersPerSecond2 > 0.0)
            require(jerkLimitMetersPerSecond3 > 0.0)
            require(positionBlendMin in 0.0..1.0)
            require(positionBlendMax in positionBlendMin..1.0)
            require(velocityGain in 0.0..1.0)
            require(accelerationGain in 0.0..1.0)
            require(microMotionPreservation in 0.0..1.0)
            require(uncertaintyGrowthPerFrame >= 0.0)
            require(uncertaintyGrowthPerSecond >= 0.0)
            require(uncertaintyFloorMeters > 0.0)
            require(uncertaintyCeilingMeters >= uncertaintyFloorMeters)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val surface: FaceMeshMultiViewFusionEngine.Result,
        val preserveMicroMotion: Boolean = true,
    ) {
        init {
            require(timestampNanos > 0L)
        }
    }

    data class PointDiagnostics(
        val index: Int,
        val innovationMeters: Double,
        val velocityMetersPerSecond: Double,
        val accelerationMetersPerSecond2: Double,
        val uncertaintyMeters: Double,
        val confidence: Double,
        val predictionOnly: Boolean,
        val acceptedMeasurement: Boolean,
        val occlusionFrames: Int,
        val microMotionMeters: Double,
    )

    data class Diagnostics(
        val timestampNanos: Long,
        val acceptedPoints: Int,
        val rejectedPoints: Int,
        val predictionOnlyPoints: Int,
        val meanInnovationMeters: Double,
        val p95InnovationMeters: Double,
        val meanUncertaintyMeters: Double,
        val meanVelocityMetersPerSecond: Double,
        val surfaceConfidence: Double,
        val temporalContinuity: Double,
        val microMotionRetention: Double,
        val finalScore: Double,
        val warnings: List<String>,
        val points: List<PointDiagnostics>,
    )

    data class Result(
        val surface: FaceMeshMultiViewFusionEngine.Result,
        val diagnostics: Diagnostics,
    )

    private data class State(
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var uncertaintyMeters: Double,
        var confidence: Double,
        var lastTimestampNanos: Long,
        var occlusionFrames: Int,
        var residualMeters: Double,
    )

    private val states =
        LinkedHashMap<Int, State>()
    private var lastTimestampNanos =
        Long.MIN_VALUE

    @Synchronized
    fun process(input: Input): Result {
        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            input.timestampNanos <= lastTimestampNanos
        ) {
            reset()
        }

        val dt =
            deltaSeconds(
                lastTimestampNanos,
                input.timestampNanos,
            )

        val pointDiagnostics =
            ArrayList<PointDiagnostics>(
                input.surface.points.size,
            )
        val outputPoints =
            ArrayList<FaceMeshMultiViewFusionEngine.SurfacePoint>(
                input.surface.points.size,
            )

        input.surface.points.forEach { measured ->
            if (
                measured.confidence <
                    config.minimumConfidence
            ) {
                return@forEach
            }

            val state =
                states.getOrPut(measured.index) {
                    State(
                        position = measured.positionWorld,
                        velocity = Vec3(0.0, 0.0, 0.0),
                        acceleration = Vec3(0.0, 0.0, 0.0),
                        uncertaintyMeters =
                            measured.uncertaintyMeters.coerceIn(
                                config.uncertaintyFloorMeters,
                                config.uncertaintyCeilingMeters,
                            ),
                        confidence = measured.confidence,
                        lastTimestampNanos =
                            input.timestampNanos,
                        occlusionFrames = 0,
                        residualMeters = 0.0,
                    )
                }

            val prediction =
                predict(
                    state = state,
                    dtSeconds = dt,
                )

            val residualVector =
                measured.positionWorld -
                    prediction.position
            val innovation =
                residualVector.norm()

            val gate =
                config.softInnovationMeters *
                    (
                        1.0 +
                            state.uncertaintyMeters /
                                config.uncertaintyFloorMeters *
                                0.025
                        )
                    .coerceIn(
                        1.0,
                        2.8,
                    )

            val robustWeight =
                innovationWeight(
                    residualMeters = innovation,
                    softGate = gate,
                    hardGate =
                        config.hardInnovationMeters,
                )

            val accepted =
                innovation <= config.hardInnovationMeters ||
                    measured.confidence >= 0.88

            val measurementTrust =
                if (accepted) {
                    (
                        robustWeight *
                            measured.confidence *
                            (
                                0.58 +
                                    0.42 *
                                        measured.conditionScore
                                )
                        ).coerceIn(
                            1.0 - config.positionBlendMax,
                            1.0 - config.positionBlendMin,
                        )
                } else {
                    0.0
                }

            val predictionTrust =
                1.0 - measurementTrust

            val microMotion =
                residualVector -
                    state.velocity * dt

            val microMotionGain =
                if (input.preserveMicroMotion) {
                    config.microMotionPreservation *
                        measured.confidence *
                        measured.conditionScore
                } else {
                    0.0
                }

            val candidatePosition =
                prediction.position *
                    predictionTrust +
                    measured.positionWorld *
                    measurementTrust +
                    microMotion *
                    (microMotionGain * 0.18)

            val boundedPosition =
                boundStep(
                    previous = state.position,
                    candidate = candidatePosition,
                    dtSeconds = dt,
                )

            val observedVelocity =
                (
                    boundedPosition -
                        state.position
                    ) /
                    dt.coerceAtLeast(1e-4)

            val velocityBlend =
                (
                    config.velocityGain *
                        measured.confidence *
                        measured.conditionScore
                    ).coerceIn(0.06, 0.92)

            var velocity =
                lerp(
                    prediction.velocity,
                    observedVelocity,
                    velocityBlend,
                )

            val observedAcceleration =
                (
                    velocity -
                        state.velocity
                    ) /
                    dt.coerceAtLeast(1e-4)

            var acceleration =
                lerp(
                    state.acceleration,
                    observedAcceleration,
                    config.accelerationGain *
                        measured.confidence *
                        measured.conditionScore,
                )

            val jerk =
                (
                    acceleration -
                        state.acceleration
                    ) /
                    dt.coerceAtLeast(1e-4)

            val boundedJerk =
                clampMagnitude(
                    jerk,
                    config.jerkLimitMetersPerSecond3,
                )

            acceleration =
                state.acceleration +
                    boundedJerk *
                    dt.coerceIn(1e-4, 0.10) *
                    0.35

            acceleration =
                clampMagnitude(
                    acceleration,
                    config.accelerationLimitMetersPerSecond2,
                )

            velocity =
                clampMagnitude(
                    velocity,
                    config.velocityLimitMetersPerSecond,
                )

            val nextUncertainty =
                if (accepted) {
                    (
                        state.uncertaintyMeters *
                            0.76 +
                            measured.uncertaintyMeters *
                            0.24 +
                            config.uncertaintyGrowthPerSecond *
                                dt
                        ).coerceIn(
                            config.uncertaintyFloorMeters,
                            config.uncertaintyCeilingMeters,
                        )
                } else {
                    (
                        state.uncertaintyMeters +
                            config.uncertaintyGrowthPerFrame +
                            config.uncertaintyGrowthPerSecond *
                                dt
                        ).coerceIn(
                            config.uncertaintyFloorMeters,
                            config.uncertaintyCeilingMeters,
                        )
                }

            state.position =
                boundedPosition
            state.velocity =
                velocity
            state.acceleration =
                acceleration
            state.uncertaintyMeters =
                nextUncertainty
            state.confidence =
                if (accepted) {
                    (
                        state.confidence * 0.24 +
                            measured.confidence * 0.46 +
                            robustWeight * 0.18 +
                            measured.conditionScore * 0.12
                        ).coerceIn(0.0, 1.0)
                } else {
                    (
                        state.confidence * 0.82
                    ).coerceIn(0.0, 1.0)
                }
            state.occlusionFrames =
                if (accepted) {
                    0
                } else {
                    state.occlusionFrames + 1
                }
            state.lastTimestampNanos =
                input.timestampNanos
            state.residualMeters =
                innovation

            if (
                state.occlusionFrames <=
                    config.maximumOcclusionFrames
            ) {
                outputPoints +=
                    measured.copy(
                        positionWorld = state.position,
                        confidence = state.confidence,
                        uncertaintyMeters =
                            state.uncertaintyMeters,
                    )
            }

            pointDiagnostics +=
                PointDiagnostics(
                    index = measured.index,
                    innovationMeters = innovation,
                    velocityMetersPerSecond =
                        velocity.norm(),
                    accelerationMetersPerSecond2 =
                        acceleration.norm(),
                    uncertaintyMeters =
                        state.uncertaintyMeters,
                    confidence =
                        state.confidence,
                    predictionOnly = !accepted,
                    acceptedMeasurement = accepted,
                    occlusionFrames =
                        state.occlusionFrames,
                    microMotionMeters =
                        (microMotion * microMotionGain).norm(),
                )
        }

        carryMissingPoints(
            timestampNanos = input.timestampNanos,
            dtSeconds = dt,
            output = outputPoints,
            diagnostics = pointDiagnostics,
        )

        val innovations =
            pointDiagnostics.map {
                it.innovationMeters
            }.sorted()

        val uncertainties =
            pointDiagnostics.map {
                it.uncertaintyMeters
            }

        val velocities =
            pointDiagnostics.map {
                it.velocityMetersPerSecond
            }

        val meanInnovation =
            innovations.averageOrZero()
        val p95Innovation =
            percentile(innovations, 0.95)
        val meanUncertainty =
            uncertainties.averageOrZero()
        val meanVelocity =
            velocities.averageOrZero()

        val continuity =
            if (pointDiagnostics.isEmpty()) {
                0.0
            } else {
                pointDiagnostics.count {
                    it.acceptedMeasurement
                }.toDouble() /
                    pointDiagnostics.size.toDouble()
            }

        val confidence =
            if (outputPoints.isEmpty()) {
                0.0
            } else {
                outputPoints.map {
                    it.confidence
                }.average()
            }

        val microMotionRetention =
            if (pointDiagnostics.isEmpty()) {
                0.0
            } else {
                pointDiagnostics.map {
                    min(
                        1.0,
                        it.microMotionMeters /
                            0.0001.coerceAtLeast(
                                it.innovationMeters,
                            ),
                    )
                }.average()
            }

        val finalScore =
            (
                confidence * 0.42 +
                    continuity * 0.26 +
                    expDecay(
                        meanInnovation,
                        config.softInnovationMeters,
                    ) * 0.14 +
                    expDecay(
                        meanUncertainty,
                        config.uncertaintyFloorMeters * 8.0,
                    ) * 0.10 +
                    microMotionRetention * 0.08
                ).coerceIn(0.0, 1.0)

        val warnings =
            buildWarnings(
                points = pointDiagnostics,
                meanInnovation = meanInnovation,
                meanUncertainty = meanUncertainty,
                finalScore = finalScore,
            )

        lastTimestampNanos =
            input.timestampNanos

        pruneExpiredStates()

        val outputSurface =
            input.surface.copy(
                points = outputPoints.sortedBy {
                    it.index
                },
                confidence = (
                    input.surface.confidence * 0.60 +
                        finalScore * 0.40
                    ).coerceIn(0.0, 1.0),
                warnings = (
                    input.surface.warnings +
                        warnings
                    ).distinct().take(16),
            )

        return Result(
            surface = outputSurface,
            diagnostics = Diagnostics(
                timestampNanos = input.timestampNanos,
                acceptedPoints =
                    pointDiagnostics.count {
                        it.acceptedMeasurement
                    },
                rejectedPoints =
                    pointDiagnostics.count {
                        !it.acceptedMeasurement
                    },
                predictionOnlyPoints =
                    pointDiagnostics.count {
                        it.predictionOnly
                    },
                meanInnovationMeters =
                    meanInnovation,
                p95InnovationMeters =
                    p95Innovation,
                meanUncertaintyMeters =
                    meanUncertainty,
                meanVelocityMetersPerSecond =
                    meanVelocity,
                surfaceConfidence =
                    confidence.coerceIn(0.0, 1.0),
                temporalContinuity =
                    continuity.coerceIn(0.0, 1.0),
                microMotionRetention =
                    microMotionRetention.coerceIn(0.0, 1.0),
                finalScore = finalScore,
                warnings = warnings,
                points = pointDiagnostics.sortedBy {
                    it.index
                },
            ),
        )
    }

    fun updateConfig(newConfig: Config) {
        config = newConfig
    }

    fun reset() {
        states.clear()
        lastTimestampNanos = Long.MIN_VALUE
    }

    private data class Prediction(
        val position: Vec3,
        val velocity: Vec3,
    )

    private fun predict(
        state: State,
        dtSeconds: Double,
    ): Prediction {
        val dt =
            dtSeconds.coerceIn(
                1.0 / 480.0,
                config.predictionHorizonSeconds,
            )

        return Prediction(
            position =
                state.position +
                    state.velocity * dt +
                    state.acceleration *
                        (0.5 * dt * dt),
            velocity =
                state.velocity +
                    state.acceleration * dt,
        )
    }

    private fun carryMissingPoints(
        timestampNanos: Long,
        dtSeconds: Double,
        output: MutableList<FaceMeshMultiViewFusionEngine.SurfacePoint>,
        diagnostics: MutableList<PointDiagnostics>,
    ) {
        val activeIds =
            output.map { it.index }.toHashSet()

        states.forEach { (index, state) ->
            if (index in activeIds) return@forEach
            if (
                state.occlusionFrames >=
                    config.maximumOcclusionFrames
            ) {
                return@forEach
            }

            val prediction =
                predict(
                    state = state,
                    dtSeconds = dtSeconds,
                )

            state.position =
                prediction.position
            state.velocity =
                clampMagnitude(
                    prediction.velocity,
                    config.velocityLimitMetersPerSecond,
                )
            state.acceleration =
                clampMagnitude(
                    state.acceleration,
                    config.accelerationLimitMetersPerSecond2,
                )
            state.uncertaintyMeters =
                (
                    state.uncertaintyMeters +
                        config.uncertaintyGrowthPerFrame
                ).coerceIn(
                    config.uncertaintyFloorMeters,
                    config.uncertaintyCeilingMeters,
                )
            state.confidence =
                (
                    state.confidence * 0.82
                ).coerceIn(0.0, 1.0)
            state.occlusionFrames += 1
            state.lastTimestampNanos =
                timestampNanos

            output +=
                FaceMeshMultiViewFusionEngine.SurfacePoint(
                    index = index,
                    positionWorld = state.position,
                    confidence = state.confidence,
                    uncertaintyMeters =
                        state.uncertaintyMeters,
                    reprojectionErrorPx =
                        state.residualMeters,
                    parallaxDegrees = 0.0,
                    sourceCount = 1,
                    conditionScore =
                        (
                            state.confidence *
                                0.78
                        ).coerceIn(0.0, 1.0),
                )

            diagnostics +=
                PointDiagnostics(
                    index = index,
                    innovationMeters =
                        state.residualMeters,
                    velocityMetersPerSecond =
                        state.velocity.norm(),
                    accelerationMetersPerSecond2 =
                        state.acceleration.norm(),
                    uncertaintyMeters =
                        state.uncertaintyMeters,
                    confidence =
                        state.confidence,
                    predictionOnly = true,
                    acceptedMeasurement = false,
                    occlusionFrames =
                        state.occlusionFrames,
                    microMotionMeters = 0.0,
                )
        }
    }

    private fun boundStep(
        previous: Vec3,
        candidate: Vec3,
        dtSeconds: Double,
    ): Vec3 {
        val delta =
            candidate - previous
        val distance =
            delta.norm()

        if (
            !distance.isFinite()
        ) {
            return previous
        }

        val maximumDistance =
            config.velocityLimitMetersPerSecond *
                dtSeconds.coerceIn(
                    1.0 / 480.0,
                    0.20,
                )

        if (
            distance <= maximumDistance
        ) {
            return candidate
        }

        return previous +
            delta *
            (
                maximumDistance /
                    distance.coerceAtLeast(1e-9)
                )
    }

    private fun innovationWeight(
        residualMeters: Double,
        softGate: Double,
        hardGate: Double,
    ): Double {
        if (!residualMeters.isFinite()) {
            return 0.0
        }
        if (residualMeters <= softGate) {
            return 1.0
        }
        if (residualMeters >= hardGate) {
            return 0.025
        }

        val normalized =
            (
                residualMeters - softGate
                ) /
                (hardGate - softGate)
                    .coerceAtLeast(1e-8)

        return (
            1.0 -
                0.975 * normalized *
                    normalized
            ).coerceIn(0.025, 1.0)
    }

    private fun buildWarnings(
        points: List<PointDiagnostics>,
        meanInnovation: Double,
        meanUncertainty: Double,
        finalScore: Double,
    ): List<String> {
        val warnings = ArrayList<String>()

        if (points.isEmpty()) {
            warnings += "dense-face-temporal-no-points"
            return warnings
        }

        val predictionRatio =
            points.count {
                it.predictionOnly
            }.toDouble() /
                points.size.toDouble()

        if (predictionRatio > 0.25) {
            warnings += "dense-face-temporal-prediction-heavy"
        }

        if (
            meanInnovation >
                config.softInnovationMeters
        ) {
            warnings +=
                "dense-face-temporal-residual-high"
        }

        if (
            meanUncertainty >
                config.uncertaintyFloorMeters * 8.0
        ) {
            warnings +=
                "dense-face-temporal-uncertainty-high"
        }

        if (finalScore < 0.48) {
            warnings +=
                "dense-face-temporal-quality-degraded"
        }

        return warnings.distinct().take(10)
    }

    private fun pruneExpiredStates() {
        val iterator =
            states.entries.iterator()

        while (iterator.hasNext()) {
            val state =
                iterator.next().value

            if (
                state.occlusionFrames >
                    config.maximumOcclusionFrames +
                    2
            ) {
                iterator.remove()
            }
        }
    }

    private fun deltaSeconds(
        previous: Long,
        current: Long,
    ): Double {
        if (previous == Long.MIN_VALUE) {
            return 1.0 / 60.0
        }

        return (
            (
                current - previous
            ).toDouble() / 1e9
        ).coerceIn(
            1.0 / 480.0,
            config.maximumGapSeconds,
        )
    }

    private fun lerp(
        a: Vec3,
        b: Vec3,
        factor: Double,
    ): Vec3 {
        val t =
            factor.coerceIn(0.0, 1.0)
        return a +
            (b - a) * t
    }

    private fun clampMagnitude(
        value: Vec3,
        maximum: Double,
    ): Vec3 {
        val norm = value.norm()
        if (!norm.isFinite()) {
            return Vec3(0.0, 0.0, 0.0)
        }
        if (norm <= maximum) {
            return value
        }
        return value *
            (
                maximum /
                    norm.coerceAtLeast(1e-9)
                )
    }

    private fun expDecay(
        value: Double,
        scale: Double,
    ): Double {
        if (!value.isFinite()) {
            return 0.0
        }
        return exp(
            -value.coerceAtLeast(0.0) /
                scale.coerceAtLeast(1e-7),
        ).coerceIn(0.0, 1.0)
    }

    private fun List<Double>.averageOrZero(): Double =
        filter(Double::isFinite)
            .let {
                if (it.isEmpty()) {
                    0.0
                } else {
                    it.average()
                }
            }

    private fun percentile(
        values: List<Double>,
        percentile: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        if (values.size == 1) return values.first()

        val sorted =
            values.filter(Double::isFinite).sorted()
        if (sorted.isEmpty()) return 0.0

        val position =
            percentile.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper =
            min(
                sorted.lastIndex,
                lower + 1,
            )
        val factor = position - lower

        return sorted[lower] * (1.0 - factor) +
            sorted[upper] * factor
    }
}
