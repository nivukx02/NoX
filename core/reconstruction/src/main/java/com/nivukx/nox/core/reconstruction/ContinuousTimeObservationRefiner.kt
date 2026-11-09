package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Continuous-time temporal prior for row-dependent rolling-shutter observations.
 *
 * Known states on either side of an observation are interpolated with cubic Hermite motion.
 * The result exposes temporal uncertainty and robust agreement with the measured observation,
 * allowing geometric reconstruction to down-weight unstable temporal estimates instead of hiding
 * them inside a frame-level confidence.
 */
class ContinuousTimeObservationRefiner(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumIntervalNanos: Long = 250_000L,
        val maximumIntervalNanos: Long = 250_000_000L,
        val maximumExtrapolationNanos: Long = 12_000_000L,
        val minimumConfidence: Double = 0.12,
        val velocityUncertaintyScale: Double = 0.0025,
        val accelerationUncertaintyScale: Double = 0.0008,
        val robustResidualScaleMeters: Double = 0.012,
        val minimumRobustWeight: Double = 0.03,
    )

    data class State(
        val timestampNanos: Long,
        val xMeters: Double,
        val yMeters: Double,
        val zMeters: Double,
        val vxMetersPerSecond: Double = 0.0,
        val vyMetersPerSecond: Double = 0.0,
        val vzMetersPerSecond: Double = 0.0,
        val confidence: Double = 1.0,
    )

    data class RowTime(
        val frameTimestampNanos: Long,
        val normalizedRow: Double,
        val readoutTimeNanos: Long,
        val directionSign: Int = 1,
    ) {
        init {
            require(normalizedRow in 0.0..1.0)
            require(readoutTimeNanos >= 0L)
            require(directionSign == 1 || directionSign == -1)
        }

        fun timestampNanos(): Long =
            frameTimestampNanos +
                (
                    (normalizedRow - 0.5) *
                        readoutTimeNanos *
                        directionSign
                    ).toLong()
    }

    data class Result(
        val targetTimestampNanos: Long,
        val xMeters: Double,
        val yMeters: Double,
        val zMeters: Double,
        val vxMetersPerSecond: Double,
        val vyMetersPerSecond: Double,
        val vzMetersPerSecond: Double,
        val temporalResidualNanos: Long,
        val interpolationAlpha: Double,
        val temporalConfidence: Double,
        val positionUncertaintyMeters: Double,
        val robustWeight: Double,
        val extrapolated: Boolean,
        val warnings: List<String>,
    )

    fun refine(
        previous: State,
        next: State,
        target: RowTime,
        measured: Triple<Double, Double, Double>? = null,
    ): Result {
        require(next.timestampNanos > previous.timestampNanos)
        val rawTarget = target.timestampNanos()
        val support = next.timestampNanos - previous.timestampNanos
        val warnings = mutableListOf<String>()

        if (support < config.minimumIntervalNanos) warnings += "SUPPORT_INTERVAL_TOO_SHORT"
        if (support > config.maximumIntervalNanos) warnings += "SUPPORT_INTERVAL_TOO_LONG"

        val clamped = rawTarget.coerceIn(previous.timestampNanos, next.timestampNanos)
        val extrapolated = clamped != rawTarget
        val extrapolation = abs(rawTarget - clamped)

        if (extrapolated && extrapolation > config.maximumExtrapolationNanos) {
            return fallback(
                if (rawTarget < previous.timestampNanos) previous else next,
                rawTarget,
                "EXTRAPOLATION_EXCEEDS_HARD_BOUND",
            )
        }

        val dt = support.toDouble() / 1e9
        val t = (clamped - previous.timestampNanos).toDouble() / support.toDouble()

        val x = hermite(
            previous.xMeters,
            next.xMeters,
            previous.vxMetersPerSecond,
            next.vxMetersPerSecond,
            t,
            dt,
        )
        val y = hermite(
            previous.yMeters,
            next.yMeters,
            previous.vyMetersPerSecond,
            next.vyMetersPerSecond,
            t,
            dt,
        )
        val z = hermite(
            previous.zMeters,
            next.zMeters,
            previous.vzMetersPerSecond,
            next.vzMetersPerSecond,
            t,
            dt,
        )

        val vx = derivative(
            previous.xMeters,
            next.xMeters,
            previous.vxMetersPerSecond,
            next.vxMetersPerSecond,
            t,
            dt,
        )
        val vy = derivative(
            previous.yMeters,
            next.yMeters,
            previous.vyMetersPerSecond,
            next.vyMetersPerSecond,
            t,
            dt,
        )
        val vz = derivative(
            previous.zMeters,
            next.zMeters,
            previous.vzMetersPerSecond,
            next.vzMetersPerSecond,
            t,
            dt,
        )

        val supportConfidence =
            previous.confidence * (1.0 - t) +
                next.confidence * t

        val intervalQuality = (
            1.0 -
                (
                    support.toDouble() -
                        config.minimumIntervalNanos
                ) /
                (
                    config.maximumIntervalNanos.toDouble() -
                        config.minimumIntervalNanos.toDouble()
                ).coerceAtLeast(1.0)
            ).coerceIn(0.0, 1.0)
        val intervalScore =
            (0.55 + 0.45 * intervalQuality).coerceIn(0.55, 1.0)

        val extrapolationScore = if (!extrapolated) {
            1.0
        } else {
            exp(
                -extrapolation /
                    max(config.maximumExtrapolationNanos.toDouble(), 1.0),
            )
        }

        val temporalConfidence = (
            supportConfidence *
                intervalScore *
                extrapolationScore
            ).coerceIn(config.minimumConfidence, 1.0)

        val acceleration = acceleration(previous, next)
        val uncertainty = (
            config.velocityUncertaintyScale *
                norm(vx, vy, vz) +
                config.accelerationUncertaintyScale *
                acceleration
            ) * max(dt, 0.001)

        val robustWeight = measured?.let {
            robust(
                norm(
                    x - it.first,
                    y - it.second,
                    z - it.third,
                ),
            )
        } ?: 1.0

        if (robustWeight < 0.25) {
            warnings += "TEMPORAL_MODEL_AND_MEASUREMENT_DISAGREE"
        }
        if (temporalConfidence < 0.50) {
            warnings += "TEMPORAL_CONFIDENCE_DEGRADED"
        }
        if (uncertainty > 0.010) {
            warnings += "TEMPORAL_MOTION_UNCERTAINTY_HIGH"
        }
        if (extrapolated) {
            warnings += "TARGET_CLAMPED_TO_SUPPORT_INTERVAL"
        }

        return Result(
            targetTimestampNanos = rawTarget,
            xMeters = x,
            yMeters = y,
            zMeters = z,
            vxMetersPerSecond = vx,
            vyMetersPerSecond = vy,
            vzMetersPerSecond = vz,
            temporalResidualNanos = rawTarget - clamped,
            interpolationAlpha = t,
            temporalConfidence = temporalConfidence,
            positionUncertaintyMeters = uncertainty.coerceAtLeast(0.0),
            robustWeight = max(
                robustWeight,
                config.minimumRobustWeight,
            ),
            extrapolated = extrapolated,
            warnings = warnings.distinct(),
        )
    }

    fun refineBatch(
        supports: List<Pair<State, State>>,
        targets: List<RowTime>,
    ): List<Result> {
        val count = min(supports.size, targets.size)
        return (0 until count).map { index ->
            val support = supports[index]
            refine(support.first, support.second, targets[index])
        }
    }

    private fun fallback(
        state: State,
        targetTimestampNanos: Long,
        warning: String,
    ) = Result(
        targetTimestampNanos = targetTimestampNanos,
        xMeters = state.xMeters,
        yMeters = state.yMeters,
        zMeters = state.zMeters,
        vxMetersPerSecond = state.vxMetersPerSecond,
        vyMetersPerSecond = state.vyMetersPerSecond,
        vzMetersPerSecond = state.vzMetersPerSecond,
        temporalResidualNanos = targetTimestampNanos - state.timestampNanos,
        interpolationAlpha = 0.0,
        temporalConfidence = config.minimumConfidence,
        positionUncertaintyMeters =
            config.velocityUncertaintyScale *
                norm(
                    state.vxMetersPerSecond,
                    state.vyMetersPerSecond,
                    state.vzMetersPerSecond,
                ),
        robustWeight = config.minimumRobustWeight,
        extrapolated = targetTimestampNanos != state.timestampNanos,
        warnings = listOf(warning),
    )

    private fun hermite(
        p0: Double,
        p1: Double,
        v0: Double,
        v1: Double,
        t: Double,
        dt: Double,
    ): Double {
        val h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0
        val h10 = t * t * t - 2.0 * t * t + t
        val h01 = -2.0 * t * t * t + 3.0 * t * t
        val h11 = t * t * t - t * t
        return h00 * p0 +
            h10 * dt * v0 +
            h01 * p1 +
            h11 * dt * v1
    }

    private fun derivative(
        p0: Double,
        p1: Double,
        v0: Double,
        v1: Double,
        t: Double,
        dt: Double,
    ): Double {
        val dh00 = 6.0 * t * t - 6.0 * t
        val dh10 = 3.0 * t * t - 4.0 * t + 1.0
        val dh01 = -6.0 * t * t + 6.0 * t
        val dh11 = 3.0 * t * t - 2.0 * t
        return (
            dh00 * p0 +
                dh10 * dt * v0 +
                dh01 * p1 +
                dh11 * dt * v1
            ) / dt.coerceAtLeast(1e-9)
    }

    private fun acceleration(
        previous: State,
        next: State,
    ): Double {
        val dt = (next.timestampNanos - previous.timestampNanos).toDouble() / 1e9
        return norm(
            (next.vxMetersPerSecond - previous.vxMetersPerSecond) / dt,
            (next.vyMetersPerSecond - previous.vyMetersPerSecond) / dt,
            (next.vzMetersPerSecond - previous.vzMetersPerSecond) / dt,
        )
    }

    private fun robust(residual: Double): Double {
        val u = residual / max(config.robustResidualScaleMeters, 1e-9)
        if (!u.isFinite()) return 0.0
        return if (u <= 1.0) {
            val t = 1.0 - u * u
            t * t
        } else {
            1.0 / (1.0 + u * u)
        }
    }

    private fun norm(x: Double, y: Double, z: Double) =
        sqrt(x * x + y * y + z * z)
}
