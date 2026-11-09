package com.nivukx.nox.core.capture

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Hardware readout timing evidence authority.
 *
 * Exposure-start and sensor-readout-start are distinct timestamps. This engine keeps both
 * semantics separate and evaluates whether the observed readout interval is stable enough to be
 * used by temporal reconstruction. It does not declare a camera "accurate" from a timestamp alone.
 */
class HardwareReadoutTimingEvidenceEngine(
    private val config: Config = Config(),
) {
    enum class State {
        UNKNOWN,
        STABLE,
        WATCH,
        DEGRADED,
        BLOCKED,
    }

    data class Config(
        val historySize: Int = 128,
        val minimumSamples: Int = 8,
        val maximumValidDelayMillis: Double = 250.0,
        val stableP95JitterMillis: Double = 0.75,
        val watchP95JitterMillis: Double = 2.5,
        val degradedP95JitterMillis: Double = 6.0,
        val stableBaselineDriftMillis: Double = 1.5,
        val watchBaselineDriftMillis: Double = 5.0,
    ) {
        init {
            require(historySize >= minimumSamples)
            require(minimumSamples >= 2)
            require(maximumValidDelayMillis > 0.0)
            require(stableP95JitterMillis > 0.0)
            require(watchP95JitterMillis >= stableP95JitterMillis)
            require(degradedP95JitterMillis >= watchP95JitterMillis)
            require(stableBaselineDriftMillis > 0.0)
            require(watchBaselineDriftMillis >= stableBaselineDriftMillis)
        }
    }

    data class Observation(
        val sequence: Long,
        val sensorTimestampNanos: Long,
        val readoutTimestampNanos: Long,
    ) {
        init {
            require(sequence >= 0L)
            require(sensorTimestampNanos >= 0L)
            // Negative readout timestamps are retained as evidence so the observer
            // can explicitly classify them as invalid samples instead of crashing the capture audit.
        }
    }

    data class Result(
        val state: State,
        val confidence: Double,
        val sampleCount: Int,
        val validSampleCount: Int,
        val invalidSampleCount: Int,
        val outOfOrderCount: Int,
        val coverage: Double,
        val medianDelayMillis: Double,
        val p95DelayMillis: Double,
        val p95JitterMillis: Double,
        val baselineDriftMillis: Double,
        val monotonicReadout: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private val observations = ArrayDeque<Observation>()
    private var lastReadoutTimestampNanos = Long.MIN_VALUE
    private var invalidSampleCount = 0
    private var outOfOrderCount = 0

    @Synchronized
    fun observe(observation: Observation): Result {
        if (lastReadoutTimestampNanos != Long.MIN_VALUE &&
            observation.readoutTimestampNanos < lastReadoutTimestampNanos
        ) {
            outOfOrderCount += 1
        }

        if (
            observation.readoutTimestampNanos < observation.sensorTimestampNanos ||
            observation.readoutTimestampNanos - observation.sensorTimestampNanos >
                (
                    config.maximumValidDelayMillis * 1_000_000.0
                ).toLong()
        ) {
            invalidSampleCount += 1
        } else {
            observations.addLast(observation)
            while (observations.size > config.historySize) {
                observations.removeFirst()
            }
        }

        lastReadoutTimestampNanos =
            maxOf(
                lastReadoutTimestampNanos,
                observation.readoutTimestampNanos,
            )

        return evaluate()
    }

    @Synchronized
    fun evaluate(): Result {
        val valid = observations.toList()
        if (valid.size < config.minimumSamples) {
            return Result(
                state = State.UNKNOWN,
                confidence = (
                    valid.size.toDouble() /
                        config.minimumSamples.toDouble()
                    ).coerceIn(0.0, 0.90),
                sampleCount = valid.size + invalidSampleCount,
                validSampleCount = valid.size,
                invalidSampleCount = invalidSampleCount,
                outOfOrderCount = outOfOrderCount,
                coverage = coverage(),
                medianDelayMillis = medianDelayMillis(valid),
                p95DelayMillis = p95DelayMillis(valid),
                p95JitterMillis = Double.POSITIVE_INFINITY,
                baselineDriftMillis = Double.POSITIVE_INFINITY,
                monotonicReadout = outOfOrderCount == 0,
                warnings = buildList {
                    add("INSUFFICIENT_HARDWARE_READOUT_SAMPLES")
                    if (invalidSampleCount > 0) {
                        add("HARDWARE_READOUT_INVALID_SAMPLES_REJECTED")
                    }
                    if (outOfOrderCount > 0) {
                        add("HARDWARE_READOUT_OUT_OF_ORDER")
                    }
                }.distinct(),
                recommendations = listOf(
                    "Collect more hardware readout callbacks before changing temporal reconstruction timing policy.",
                ),
            )
        }

        val delaysMillis = valid
            .map {
                (
                    it.readoutTimestampNanos -
                        it.sensorTimestampNanos
                    ) / 1_000_000.0
            }
            .sorted()

        val median = quantile(delaysMillis, 0.50)
        val p95 = quantile(delaysMillis, 0.95)
        val deviations = delaysMillis
            .map { abs(it - median) }
            .sorted()
        val p95Jitter = quantile(deviations, 0.95)

        val firstWindow = delaysMillis.take(
            maxOf(1, delaysMillis.size / 4),
        ).average()
        val lastWindow = delaysMillis.takeLast(
            maxOf(1, delaysMillis.size / 4),
        ).average()
        val baselineDrift = abs(lastWindow - firstWindow)

        val coverage = coverage()
        val monotonic = outOfOrderCount == 0

        val jitterScore =
            inverse(
                p95Jitter,
                config.stableP95JitterMillis,
                config.degradedP95JitterMillis,
            )
        val baselineScore =
            inverse(
                baselineDrift,
                config.stableBaselineDriftMillis,
                config.watchBaselineDriftMillis,
            )
        val orderingScore =
            if (monotonic) {
                1.0
            } else {
                (
                    1.0 -
                        outOfOrderCount.toDouble() /
                            maxOf(
                                1,
                                valid.size + outOfOrderCount,
                            ).toDouble()
                    ).coerceIn(0.0, 1.0)
            }
        val confidence = (
            coverage * 0.30 +
                jitterScore * 0.36 +
                baselineScore * 0.20 +
                orderingScore * 0.14
            ).coerceIn(0.0, 1.0)

        val state = when {
            invalidSampleCount > valid.size / 3 ->
                State.BLOCKED
            p95Jitter <= config.stableP95JitterMillis &&
                baselineDrift <= config.stableBaselineDriftMillis &&
                monotonic ->
                State.STABLE
            p95Jitter <= config.watchP95JitterMillis &&
                baselineDrift <= config.watchBaselineDriftMillis ->
                State.WATCH
            p95Jitter <= config.degradedP95JitterMillis ->
                State.DEGRADED
            else ->
                State.BLOCKED
        }

        val warnings = buildList {
            if (p95Jitter > config.stableP95JitterMillis) {
                add("HARDWARE_READOUT_JITTER_ELEVATED")
            }
            if (baselineDrift > config.stableBaselineDriftMillis) {
                add("HARDWARE_READOUT_BASELINE_DRIFT")
            }
            if (!monotonic) {
                add("HARDWARE_READOUT_OUT_OF_ORDER")
            }
            if (invalidSampleCount > 0) {
                add("HARDWARE_READOUT_INVALID_SAMPLES_REJECTED")
            }
            if (coverage < 0.95) {
                add("HARDWARE_READOUT_COVERAGE_INCOMPLETE")
            }
        }.distinct()

        val recommendations = buildList {
            when (state) {
                State.STABLE,
                State.UNKNOWN -> Unit
                State.WATCH ->
                    add("Use readout timestamps as secondary temporal evidence while preserving exposure-start timestamps as the canonical source clock.")
                State.DEGRADED ->
                    add("Increase temporal uncertainty and avoid treating hardware readout timing as deterministic at sub-frame precision.")
                State.BLOCKED ->
                    add("Reject hardware-readout timing as authoritative until callback ordering and delay stability recover.")
            }
            if (baselineDrift > config.stableBaselineDriftMillis) {
                add("Investigate thermal or driver-state changes before changing camera synchronization parameters.")
            }
        }.distinct()

        return Result(
            state = state,
            confidence = confidence,
            sampleCount = valid.size + invalidSampleCount,
            validSampleCount = valid.size,
            invalidSampleCount = invalidSampleCount,
            outOfOrderCount = outOfOrderCount,
            coverage = coverage,
            medianDelayMillis = median,
            p95DelayMillis = p95,
            p95JitterMillis = p95Jitter,
            baselineDriftMillis = baselineDrift,
            monotonicReadout = monotonic,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        observations.clear()
        lastReadoutTimestampNanos = Long.MIN_VALUE
        invalidSampleCount = 0
        outOfOrderCount = 0
    }

    private fun coverage(): Double =
        if (observations.isEmpty() && invalidSampleCount == 0) {
            0.0
        } else {
            observations.size.toDouble() /
                (
                    observations.size +
                        invalidSampleCount
                    ).coerceAtLeast(1).toDouble()
        }

    private fun medianDelayMillis(
        valid: List<Observation>,
    ): Double {
        if (valid.isEmpty()) return Double.NaN
        return quantile(
            valid.map {
                (
                    it.readoutTimestampNanos -
                        it.sensorTimestampNanos
                    ) / 1_000_000.0
            }.sorted(),
            0.50,
        )
    }

    private fun p95DelayMillis(
        valid: List<Observation>,
    ): Double {
        if (valid.isEmpty()) return Double.NaN
        return quantile(
            valid.map {
                (
                    it.readoutTimestampNanos -
                        it.sensorTimestampNanos
                    ) / 1_000_000.0
            }.sorted(),
            0.95,
        )
    }

    private fun inverse(
        value: Double,
        stable: Double,
        hard: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= stable) return 1.0
        if (value >= hard) return 0.0
        return (
            1.0 -
                (value - stable) /
                    (hard - stable).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun quantile(
        sorted: List<Double>,
        fraction: Double,
    ): Double {
        if (sorted.isEmpty()) return Double.NaN
        if (sorted.size == 1) return sorted.first()
        val position =
            fraction.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        val blend = position - lower
        return sorted[lower] * (1.0 - blend) +
            sorted[upper] * blend
    }

    @Suppress("UNUSED_PARAMETER")
    private fun unusedSqrtMarker(value: Double): Double = sqrt(
        value.coerceAtLeast(0.0),
    )
}
