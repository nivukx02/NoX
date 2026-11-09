package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Bounded, robust temporal dynamics solver for triangulated 3D motion.
 *
 * It estimates position, velocity, acceleration and jerk from a short history using confidence
 * weighted cubic regression with Huber reweighting. It also provides a strictly bounded prediction
 * path for brief occlusions. The engine never invents geometry and never turns a rejected
 * observation into an authoritative observation.
 */
class KinematicStabilityEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumHistorySamples: Int = 12,
        val minimumRegressionSamples: Int = 4,
        val maximumRegressionWindowSeconds: Double = 0.16,
        val maximumPredictionHorizonSeconds: Double = 0.10,
        val minimumObservationConfidence: Double = 0.20,
        val measurementSigmaMeters: Double = 0.0025,
        val huberScaleMeters: Double = 0.0060,
        val maximumSpeedMetersPerSecond: Double = 10.0,
        val maximumAccelerationMetersPerSecond2: Double = 120.0,
        val maximumJerkMetersPerSecond3: Double = 1_200.0,
        val predictionConfidenceFloor: Double = 0.15,
        val predictionConfidenceHalfLifeSeconds: Double = 0.050,
        val stabilityInnovationScaleMeters: Double = 0.010,
        val stabilityHalfLifeSeconds: Double = 0.12,
        val outlierWeightFloor: Double = 0.08,
        val robustIterations: Int = 3,
    ) {
        init {
            require(maximumHistorySamples >= 4)
            require(minimumRegressionSamples in 3..maximumHistorySamples)
            require(maximumRegressionWindowSeconds > 0.0)
            require(maximumPredictionHorizonSeconds >= 0.0)
            require(minimumObservationConfidence in 0.0..1.0)
            require(measurementSigmaMeters > 0.0)
            require(huberScaleMeters > 0.0)
            require(maximumSpeedMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(predictionConfidenceFloor in 0.0..1.0)
            require(predictionConfidenceHalfLifeSeconds > 0.0)
            require(stabilityInnovationScaleMeters > 0.0)
            require(stabilityHalfLifeSeconds > 0.0)
            require(outlierWeightFloor in 0.0..1.0)
            require(robustIterations in 1..8)
        }
    }

    data class Observation(
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val confidence: Double,
        val observed: Boolean = true,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    enum class State {
        EMPTY,
        WARMING,
        OBSERVED,
        REJECTED,
        PREDICTED,
        STALE,
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val accelerationMetersPerSecond2: Vec3,
        val jerkMetersPerSecond3: Vec3,
        val uncertaintyMeters: Double,
        val innovationMeters: Double,
        val innovationConfidence: Double,
        val motionStability: Double,
        val modelConfidence: Double,
        val predictionAgeSeconds: Double,
        val regressionSamples: Int,
        val rejectedObservation: Boolean,
        val warnings: List<String>,
    ) {
        val finite: Boolean
            get() =
                positionMeters.x.isFinite() &&
                    positionMeters.y.isFinite() &&
                    positionMeters.z.isFinite() &&
                    velocityMetersPerSecond.x.isFinite() &&
                    velocityMetersPerSecond.y.isFinite() &&
                    velocityMetersPerSecond.z.isFinite() &&
                    accelerationMetersPerSecond2.x.isFinite() &&
                    accelerationMetersPerSecond2.y.isFinite() &&
                    accelerationMetersPerSecond2.z.isFinite() &&
                    jerkMetersPerSecond3.x.isFinite() &&
                    jerkMetersPerSecond3.y.isFinite() &&
                    jerkMetersPerSecond3.z.isFinite() &&
                    uncertaintyMeters.isFinite() &&
                    innovationMeters.isFinite() &&
                    innovationConfidence.isFinite() &&
                    motionStability.isFinite() &&
                    modelConfidence.isFinite()
    }

    private data class Sample(
        val t: Double,
        val position: Vec3,
        val confidence: Double,
    )

    private data class ScalarPoly(
        val p: Double,
        val v: Double,
        val a: Double,
        val j: Double,
    )

    private val history = ArrayDeque<Sample>()
    private var latest = emptyResult()
    private var closed = false

    @Synchronized
    fun observe(observation: Observation): Result {
        check(!closed) { "KinematicStabilityEngine is closed" }

        val t = observation.timestampNanos.toDouble() / 1_000_000_000.0
        val previous = history.lastOrNull()
        if (previous != null) {
            val dt = t - previous.t
            if (dt <= 0.0) {
                return reject(
                    observation.timestampNanos,
                    "kinematic-timestamp-reversal",
                )
            }
            if (dt > config.maximumRegressionWindowSeconds * 4.0) {
                history.clear()
                latest = emptyResult().copy(
                    timestampNanos = observation.timestampNanos,
                    state = State.WARMING,
                    warnings = listOf("kinematic-history-gap-reset"),
                )
            }
        }

        if (
            !observation.observed ||
            observation.confidence < config.minimumObservationConfidence
        ) {
            latest = latest.copy(
                timestampNanos = observation.timestampNanos,
                state = if (latest.state == State.PREDICTED) {
                    State.PREDICTED
                } else {
                    State.WARMING
                },
                rejectedObservation = true,
                warnings = (
                    latest.warnings +
                        "kinematic-observation-below-confidence-threshold"
                    ).distinct(),
            )
            return latest
        }

        history.addLast(
            Sample(
                t = t,
                position = observation.positionMeters,
                confidence = observation.confidence,
            ),
        )
        prune(t)

        if (history.size < config.minimumRegressionSamples) {
            latest = warmup(observation.timestampNanos)
            return latest
        }

        val fit = robustFit() ?: return reject(
            observation.timestampNanos,
            "kinematic-regression-failed",
        )

        val modelPosition = Vec3(fit[0].p, fit[1].p, fit[2].p)
        val velocity = Vec3(fit[0].v, fit[1].v, fit[2].v)
            .clamp(config.maximumSpeedMetersPerSecond)
        val acceleration = Vec3(fit[0].a, fit[1].a, fit[2].a)
            .clamp(config.maximumAccelerationMetersPerSecond2)
        val jerk = Vec3(fit[0].j, fit[1].j, fit[2].j)
            .clamp(config.maximumJerkMetersPerSecond3)

        val innovation =
            history.last().position.distanceTo(modelPosition)
        val innovationConfidence =
            exp(
                -innovation /
                    config.stabilityInnovationScaleMeters
                        .coerceAtLeast(1e-9),
            ).coerceIn(0.0, 1.0)

        val sampleConfidence =
            history.map { it.confidence }.average().coerceIn(0.0, 1.0)

        val windowSeconds =
            (
                history.last().t -
                    history.first().t
                ).coerceAtLeast(1e-6)

        val windowConfidence =
            min(
                1.0,
                windowSeconds /
                    config.maximumRegressionWindowSeconds
                        .coerceAtLeast(1e-6),
            )

        val densityConfidence =
            min(
                1.0,
                history.size.toDouble() /
                    config.maximumHistorySamples.toDouble(),
            )

        val modelConfidence =
            (
                sampleConfidence * 0.34 +
                    innovationConfidence * 0.36 +
                    windowConfidence * 0.20 +
                    densityConfidence * 0.10
                ).coerceIn(0.0, 1.0)

        val previousTimestamp = latest.timestampNanos
        val dt =
            if (previousTimestamp <= 0L) {
                1.0 / 60.0
            } else {
                (
                    observation.timestampNanos -
                        previousTimestamp
                    ).coerceAtLeast(1L)
                    .toDouble() /
                    1_000_000_000.0
            }

        val decay =
            exp(
                -ln(2.0) *
                    dt /
                    config.stabilityHalfLifeSeconds,
            ).coerceIn(0.0, 1.0)

        val stability =
            (
                latest.motionStability * decay * 0.40 +
                    innovationConfidence * 0.32 +
                    modelConfidence * 0.28
                ).coerceIn(0.0, 1.0)

        val uncertainty =
            (
                config.measurementSigmaMeters *
                    (
                        1.0 +
                            innovation /
                            config.stabilityInnovationScaleMeters
                                .coerceAtLeast(1e-9)
                        ) +
                    acceleration.norm() * dt * dt * 0.12 +
                    jerk.norm() * dt * dt * dt * 0.06
                ).coerceIn(
                    config.measurementSigmaMeters,
                    2.0,
                )

        val warnings = buildList {
            if (innovation > config.stabilityInnovationScaleMeters * 2.5) {
                add("kinematic-innovation-elevated")
            }
            if (
                velocity.norm() >=
                    config.maximumSpeedMetersPerSecond * 0.90
            ) {
                add("kinematic-speed-envelope-near-limit")
            }
            if (
                acceleration.norm() >=
                    config.maximumAccelerationMetersPerSecond2 * 0.90
            ) {
                add("kinematic-acceleration-envelope-near-limit")
            }
            if (
                jerk.norm() >=
                    config.maximumJerkMetersPerSecond3 * 0.90
            ) {
                add("kinematic-jerk-envelope-near-limit")
            }
            if (history.size < config.maximumHistorySamples / 2) {
                add("kinematic-regression-window-partial")
            }
        }

        latest = Result(
            timestampNanos = observation.timestampNanos,
            state = State.OBSERVED,
            positionMeters = modelPosition,
            velocityMetersPerSecond = velocity,
            accelerationMetersPerSecond2 = acceleration,
            jerkMetersPerSecond3 = jerk,
            uncertaintyMeters = uncertainty,
            innovationMeters = innovation,
            innovationConfidence = innovationConfidence,
            motionStability = stability,
            modelConfidence = modelConfidence,
            predictionAgeSeconds = 0.0,
            regressionSamples = history.size,
            rejectedObservation = false,
            warnings = warnings.distinct(),
        )
        return latest
    }

    @Synchronized
    fun predict(timestampNanos: Long): Result {
        check(!closed) { "KinematicStabilityEngine is closed" }
        val base = latest
        if (base.timestampNanos <= 0L) return base

        val age =
            (
                timestampNanos -
                    base.timestampNanos
                ).coerceAtLeast(0L)
                .toDouble() / 1_000_000_000.0

        if (age <= 0.0) return base

        if (age > config.maximumPredictionHorizonSeconds) {
            latest = base.copy(
                timestampNanos = timestampNanos,
                state = State.STALE,
                predictionAgeSeconds = age,
                modelConfidence = (
                    base.modelConfidence * 0.25
                ).coerceIn(0.0, 1.0),
                motionStability = (
                    base.motionStability * 0.40
                ).coerceIn(0.0, 1.0),
                warnings = (
                    base.warnings +
                        "kinematic-prediction-horizon-exceeded"
                    ).distinct(),
            )
            return latest
        }

        val t = age
        val predicted =
            base.positionMeters +
                base.velocityMetersPerSecond * t +
                base.accelerationMetersPerSecond2 * (0.5 * t * t) +
                base.jerkMetersPerSecond3 * (t * t * t / 6.0)

        val decay =
            exp(
                -t /
                    config.predictionConfidenceHalfLifeSeconds
                        .coerceAtLeast(1e-9),
            ).coerceIn(0.0, 1.0)

        val uncertainty =
            (
                base.uncertaintyMeters +
                    base.accelerationMetersPerSecond2.norm() *
                        t * t * 0.20 +
                    base.jerkMetersPerSecond3.norm() *
                        t * t * t * 0.10
                ).coerceAtMost(2.0)

        latest = base.copy(
            timestampNanos = timestampNanos,
            state = State.PREDICTED,
            positionMeters = predicted,
            uncertaintyMeters = uncertainty,
            innovationMeters = 0.0,
            innovationConfidence = decay,
            motionStability = (
                base.motionStability * decay
            ).coerceIn(0.0, 1.0),
            modelConfidence = max(
                config.predictionConfidenceFloor,
                base.modelConfidence * decay,
            ),
            predictionAgeSeconds = t,
            rejectedObservation = false,
            warnings = (
                base.warnings +
                    "kinematic-short-horizon-prediction"
                ).distinct(),
        )
        return latest
    }

    @Synchronized
    fun latest(): Result = latest

    @Synchronized
    fun reset() {
        history.clear()
        latest = emptyResult()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        history.clear()
        latest = emptyResult()
        closed = true
    }

    private fun prune(newest: Double) {
        while (
            history.size > config.maximumHistorySamples ||
            (
                history.size > config.minimumRegressionSamples &&
                    newest - history.first().t >
                    config.maximumRegressionWindowSeconds
            )
        ) {
            history.removeFirst()
        }
    }

    private fun robustFit(): Array<ScalarPoly>? {
        if (history.size < config.minimumRegressionSamples) return null

        val anchor = history.last().t
        val points = history.map {
            it.t - anchor to it.position
        }

        var weights = DoubleArray(history.size) {
            history.elementAt(it).confidence.coerceAtLeast(0.05)
        }

        var result = fitWeighted(points, weights) ?: return null

        repeat(config.robustIterations) {
            val residuals = points.mapIndexed { index, sample ->
                val prediction = evaluate(result, sample.first)
                sample.second.distanceTo(prediction)
            }

            weights = DoubleArray(points.size) { index ->
                val residual = residuals[index]
                val huber =
                    if (residual <= config.huberScaleMeters) {
                        1.0
                    } else {
                        config.huberScaleMeters /
                            residual.coerceAtLeast(config.huberScaleMeters)
                    }
                (
                    history.elementAt(index).confidence *
                        max(config.outlierWeightFloor, huber)
                    ).coerceIn(0.01, 1.0)
            }

            val refined = fitWeighted(points, weights)
            if (refined != null) result = refined
        }

        return result
    }

    private fun fitWeighted(
        points: List<Pair<Double, Vec3>>,
        weights: DoubleArray,
    ): Array<ScalarPoly>? {
        val normal = Array(4) { DoubleArray(4) }
        val rhs = Array(3) { DoubleArray(4) }

        for (i in points.indices) {
            val (t, p) = points[i]
            val w = weights.getOrNull(i)?.coerceAtLeast(0.01) ?: 0.01
            val basis = doubleArrayOf(
                1.0,
                t,
                t * t,
                t * t * t,
            )

            for (r in 0..3) {
                for (c in 0..3) {
                    normal[r][c] += w * basis[r] * basis[c]
                }
                rhs[0][r] += w * basis[r] * p.x
                rhs[1][r] += w * basis[r] * p.y
                rhs[2][r] += w * basis[r] * p.z
            }
        }

        for (i in 0..3) normal[i][i] += 1e-9

        val x = solve4(normal, rhs[0]) ?: return null
        val y = solve4(normal, rhs[1]) ?: return null
        val z = solve4(normal, rhs[2]) ?: return null

        return arrayOf(
            toPoly(x),
            toPoly(y),
            toPoly(z),
        )
    }

    private fun toPoly(c: DoubleArray): ScalarPoly =
        ScalarPoly(
            p = c[0],
            v = c[1],
            a = 2.0 * c[2],
            j = 6.0 * c[3],
        )

    private fun evaluate(
        p: Array<ScalarPoly>,
        t: Double,
    ): Vec3 =
        Vec3(
            p[0].p + p[0].v * t +
                0.5 * p[0].a * t * t +
                p[0].j * t * t * t / 6.0,
            p[1].p + p[1].v * t +
                0.5 * p[1].a * t * t +
                p[1].j * t * t * t / 6.0,
            p[2].p + p[2].v * t +
                0.5 * p[2].a * t * t +
                p[2].j * t * t * t / 6.0,
        )

    private fun solve4(
        a: Array<DoubleArray>,
        b: DoubleArray,
    ): DoubleArray? {
        val m = Array(4) { r ->
            DoubleArray(5) { c ->
                if (c < 4) a[r][c] else b[r]
            }
        }

        for (pivot in 0..3) {
            var row = pivot
            var best = abs(m[pivot][pivot])
            for (candidate in (pivot + 1)..3) {
                val magnitude = abs(m[candidate][pivot])
                if (magnitude > best) {
                    best = magnitude
                    row = candidate
                }
            }
            if (!best.isFinite() || best < 1e-12) return null

            if (row != pivot) {
                val tmp = m[pivot]
                m[pivot] = m[row]
                m[row] = tmp
            }

            val divisor = m[pivot][pivot]
            for (col in pivot..4) m[pivot][col] /= divisor

            for (other in 0..3) {
                if (other == pivot) continue
                val factor = m[other][pivot]
                if (abs(factor) < 1e-15) continue
                for (col in pivot..4) {
                    m[other][col] -= factor * m[pivot][col]
                }
            }
        }

        val result = DoubleArray(4)
        for (i in 0..3) {
            result[i] = m[i][4]
            if (!result[i].isFinite()) return null
        }
        return result
    }

    private fun reject(
        timestampNanos: Long,
        warning: String,
    ): Result {
        latest = latest.copy(
            timestampNanos = timestampNanos,
            state = State.REJECTED,
            innovationConfidence = 0.0,
            modelConfidence = (
                latest.modelConfidence * 0.30
            ).coerceIn(0.0, 1.0),
            motionStability = (
                latest.motionStability * 0.45
            ).coerceIn(0.0, 1.0),
            rejectedObservation = true,
            warnings = (latest.warnings + warning).distinct(),
        )
        return latest
    }

    private fun warmup(timestampNanos: Long): Result {
        val position =
            history.lastOrNull()?.position
                ?: Vec3(0.0, 0.0, 0.0)
        return Result(
            timestampNanos = timestampNanos,
            state = State.WARMING,
            positionMeters = position,
            velocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
            accelerationMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
            jerkMetersPerSecond3 = Vec3(0.0, 0.0, 0.0),
            uncertaintyMeters = 0.010,
            innovationMeters = 0.0,
            innovationConfidence = 0.0,
            motionStability = 0.20,
            modelConfidence = history.map { it.confidence }.average(),
            predictionAgeSeconds = 0.0,
            regressionSamples = history.size,
            rejectedObservation = false,
            warnings = listOf("kinematic-model-warming"),
        )
    }

    private fun emptyResult(): Result =
        Result(
            timestampNanos = 0L,
            state = State.EMPTY,
            positionMeters = Vec3(0.0, 0.0, 0.0),
            velocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
            accelerationMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
            jerkMetersPerSecond3 = Vec3(0.0, 0.0, 0.0),
            uncertaintyMeters = 1.0,
            innovationMeters = 0.0,
            innovationConfidence = 0.0,
            motionStability = 0.0,
            modelConfidence = 0.0,
            predictionAgeSeconds = 0.0,
            regressionSamples = 0,
            rejectedObservation = false,
            warnings = listOf("kinematic-no-evidence"),
        )

    private fun Vec3.clamp(maximum: Double): Vec3 {
        val magnitude = norm()
        if (!magnitude.isFinite() || magnitude <= maximum) return this
        return this * (maximum / magnitude.coerceAtLeast(1e-9))
    }
}
