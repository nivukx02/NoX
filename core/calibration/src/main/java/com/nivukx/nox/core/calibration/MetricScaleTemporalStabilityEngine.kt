package com.nivukx.nox.core.calibration

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Temporal quality gate for metric scale.
 *
 * A metric scale estimate that is correct for one frame but drifts over a long capture should not
 * be treated as authoritative. This engine keeps a small bounded history of accepted scale estimates,
 * computes robust frame-to-frame drift and jitter, and exposes a qualification state.
 *
 * The engine never invents a scale. It only evaluates the stability of a scale supplied by an
 * upstream calibrated or scale-anchored estimator.
 */
class MetricScaleTemporalStabilityEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumHistorySamples: Int = 180,
        val minimumSamplesForStability: Int = 12,
        val maximumDriftPercentPerMinute: Double = 0.25,
        val maximumJitterPercent: Double = 0.20,
        val minimumConfidenceForObservation: Double = 0.55,
        val warningDriftFraction: Double = 0.60,
        val warningJitterFraction: Double = 0.65,
        val robustTrendIterations: Int = 3,
        val maximumStepJumpPercent: Double = 0.35,
        val minimumTrendConfidence: Double = 0.70,
    ) {
        init {
            require(maximumHistorySamples >= 8)
            require(minimumSamplesForStability in 3..maximumHistorySamples)
            require(maximumDriftPercentPerMinute > 0.0)
            require(maximumJitterPercent > 0.0)
            require(minimumConfidenceForObservation in 0.0..1.0)
            require(warningDriftFraction in 0.0..1.0)
            require(warningJitterFraction in 0.0..1.0)
            require(robustTrendIterations in 1..8)
            require(maximumStepJumpPercent > 0.0)
            require(minimumTrendConfidence in 0.0..1.0)
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val scaleMetersPerUnit: Double,
        val confidence: Double,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(scaleMetersPerUnit.isFinite() && scaleMetersPerUnit > 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    enum class Status {
        WARMING_UP,
        STABLE,
        DRIFTING,
        UNSTABLE,
        REJECTED,
    }

    data class Result(
        val status: Status,
        val sampleCount: Int,
        val medianScaleMetersPerUnit: Double,
        val scaleJitterPercent: Double,
        val driftPercentPerMinute: Double,
        val maxStepJumpPercent: Double = 0.0,
        val trendConfidence: Double = 0.0,
        val outlierCount: Int = 0,
        val confidence: Double,
        val authoritative: Boolean,
        val warnings: List<String>,
    ) {
        val stable: Boolean
            get() =
                status == Status.STABLE &&
                    authoritative
    }

    private val samples = ArrayDeque<Sample>()
    private var latest: Result = emptyResult()

    @Synchronized
    fun observe(
        timestampNanos: Long,
        scaleMetersPerUnit: Double,
        confidence: Double,
    ): Result {
        if (
            timestampNanos < 0L ||
            !scaleMetersPerUnit.isFinite() ||
            scaleMetersPerUnit <= 0.0 ||
            !confidence.isFinite() ||
            confidence < config.minimumConfidenceForObservation
        ) {
            latest =
                latest.copy(
                    status = Status.REJECTED,
                    authoritative = false,
                    warnings =
                        (
                            latest.warnings +
                                "metric-scale-observation-rejected"
                            ).distinct(),
                )
            return latest
        }

        val previous =
            samples.lastOrNull()

        if (
            previous != null &&
                timestampNanos <= previous.timestampNanos
        ) {
            latest =
                latest.copy(
                    status = Status.REJECTED,
                    authoritative = false,
                    warnings =
                        (
                            latest.warnings +
                                "metric-scale-timestamp-reversal"
                            ).distinct(),
                )
            return latest
        }

        samples.addLast(
            Sample(
                timestampNanos = timestampNanos,
                scaleMetersPerUnit = scaleMetersPerUnit,
                confidence = confidence,
            ),
        )

        while (samples.size > config.maximumHistorySamples) {
            samples.removeFirst()
        }

        latest = evaluate()
        return latest
    }

    @Synchronized
    fun latest(): Result = latest

    @Synchronized
    fun reset() {
        samples.clear()
        latest = emptyResult()
    }

    @Synchronized
    override fun close() {
        reset()
    }

    private fun evaluate(): Result {
        val accepted = samples.toList()
        if (accepted.isEmpty()) {
            return emptyResult()
        }

        val scales =
            accepted
                .map { it.scaleMetersPerUnit }
                .sorted()

        val medianScale = median(scales).coerceAtLeast(1e-9)

        val deviations =
            scales
                .map {
                    abs(it - medianScale) /
                        medianScale
                }

        val jitter =
            median(deviations)
                .times(100.0)
                .coerceIn(0.0, 100.0)

        val trend =
            robustTrend(
                accepted = accepted,
                centerScale = medianScale,
            )

        val drift = trend.driftPercentPerMinute
        val maxStepJump = trend.maxStepJumpPercent
        val trendConfidence = trend.confidence
        val outlierCount = trend.outlierCount

        val meanConfidence =
            accepted
                .map { it.confidence }
                .average()
                .coerceIn(0.0, 1.0)

        val sampleConfidence =
            min(
                1.0,
                accepted.size.toDouble() /
                    config.minimumSamplesForStability
                        .coerceAtLeast(1)
                        .toDouble(),
            )

        val driftConfidence =
            (
                1.0 -
                    drift /
                        config.maximumDriftPercentPerMinute
            ).coerceIn(0.0, 1.0)

        val jitterConfidence =
            (
                1.0 -
                    jitter /
                        config.maximumJitterPercent
            ).coerceIn(0.0, 1.0)

        val confidence =
            (
                meanConfidence * 0.40 +
                    sampleConfidence * 0.20 +
                    driftConfidence * 0.22 +
                    jitterConfidence * 0.18
                ).coerceIn(0.0, 1.0)

        val warmed =
            accepted.size >=
                config.minimumSamplesForStability

        val status =
            when {
                !warmed -> Status.WARMING_UP
                maxStepJump > config.maximumStepJumpPercent ->
                    Status.UNSTABLE
                drift >
                    config.maximumDriftPercentPerMinute ->
                    Status.DRIFTING
                jitter >
                    config.maximumJitterPercent ->
                    Status.UNSTABLE
                trendConfidence < config.minimumTrendConfidence ->
                    Status.UNSTABLE
                else -> Status.STABLE
            }

        val warnings = buildList {
            if (!warmed) {
                add(
                    "metric-scale-stability-warmup=" +
                        accepted.size +
                        "/" +
                        config.minimumSamplesForStability,
                )
            }
            if (
                drift >
                    config.maximumDriftPercentPerMinute *
                        config.warningDriftFraction
            ) {
                add("metric-scale-drift-elevated")
            }
            if (
                jitter >
                    config.maximumJitterPercent *
                        config.warningJitterFraction
            ) {
                add("metric-scale-jitter-elevated")
            }
            if (
                meanConfidence <
                    config.minimumConfidenceForObservation
            ) {
                add("metric-scale-observation-confidence-low")
            }
            if (maxStepJump > config.maximumStepJumpPercent) {
                add("metric-scale-step-jump-too-large")
            } else if (
                maxStepJump >
                    config.maximumStepJumpPercent * 0.70
            ) {
                add("metric-scale-step-jump-elevated")
            }
            if (trendConfidence < config.minimumTrendConfidence) {
                add("metric-scale-trend-confidence-low")
            }
            if (outlierCount > accepted.size / 4) {
                add("metric-scale-outlier-density-elevated")
            }
        }

        val authoritative =
            warmed &&
                status == Status.STABLE &&
                confidence >= 0.70 &&
                trendConfidence >= config.minimumTrendConfidence &&
                maxStepJump <= config.maximumStepJumpPercent

        return Result(
            status = status,
            sampleCount = accepted.size,
            medianScaleMetersPerUnit = medianScale,
            scaleJitterPercent = jitter,
            driftPercentPerMinute = drift,
            maxStepJumpPercent = maxStepJump,
            trendConfidence = trendConfidence,
            outlierCount = outlierCount,
            confidence = confidence,
            authoritative = authoritative,
            warnings = warnings.distinct(),
        )
    }

    private data class TrendEstimate(
        val driftPercentPerMinute: Double,
        val maxStepJumpPercent: Double,
        val confidence: Double,
        val outlierCount: Int,
    )

    private fun robustTrend(
        accepted: List<Sample>,
        centerScale: Double,
    ): TrendEstimate {
        if (accepted.size < 2 || !centerScale.isFinite() || centerScale <= 0.0) {
            return TrendEstimate(0.0, 0.0, 0.0, 0)
        }

        val baseTime = accepted.first().timestampNanos.toDouble()
        val observations = accepted.map {
            val seconds =
                (it.timestampNanos.toDouble() - baseTime) /
                    1_000_000_000.0
            seconds to (it.scaleMetersPerUnit / centerScale)
        }

        var weights = DoubleArray(observations.size) { index ->
            accepted[index].confidence.coerceAtLeast(0.05)
        }
        var slopePerSecond = 0.0
        var intercept = 1.0

        repeat(config.robustTrendIterations) {
            var sumW = 0.0
            var sumX = 0.0
            var sumY = 0.0
            var sumXX = 0.0
            var sumXY = 0.0

            observations.forEachIndexed { index, (x, y) ->
                val w = weights[index]
                sumW += w
                sumX += w * x
                sumY += w * y
                sumXX += w * x * x
                sumXY += w * x * y
            }

            val denominator =
                sumW * sumXX - sumX * sumX

            if (denominator > 1e-12 && denominator.isFinite()) {
                slopePerSecond =
                    (
                        sumW * sumXY -
                            sumX * sumY
                        ) / denominator
                intercept =
                    (
                        sumY -
                            slopePerSecond * sumX
                        ) / sumW.coerceAtLeast(1e-9)
            }

            weights =
                DoubleArray(observations.size) { index ->
                    val (x, y) = observations[index]
                    val residual =
                        abs(
                            y -
                                (intercept + slopePerSecond * x),
                        )
                    val scale = config.maximumJitterPercent / 100.0
                    val huber =
                        if (residual <= scale) {
                            1.0
                        } else {
                            scale /
                                residual.coerceAtLeast(scale)
                        }
                    (
                        accepted[index].confidence.coerceAtLeast(0.05) *
                            huber.coerceAtLeast(0.08)
                        ).coerceIn(0.01, 1.0)
                }
        }

        val spanSeconds =
            (
                observations.last().first -
                    observations.first().first
                ).coerceAtLeast(1e-6)

        val driftPercentPerMinute =
            abs(slopePerSecond) *
                60.0 *
                100.0

        val stepJumps =
            observations.zipWithNext().mapNotNull { (a, b) ->
                val dt =
                    (b.first - a.first).coerceAtLeast(1e-6)
                val jump =
                    abs(b.second - a.second)
                (
                    jump *
                        100.0
                ).takeIf(Double::isFinite)
                    ?.div(dt)
                    ?.let {
                        it * min(1.0, dt / 0.25)
                    }
            }

        val maxStepJumpPercent =
            stepJumps.maxOrNull()?.coerceAtLeast(0.0) ?: 0.0

        val residuals =
            observations.map { (x, y) ->
                abs(
                    y -
                        (intercept + slopePerSecond * x),
                )
            }.sorted()

        val residualMedian =
            median(residuals)

        val residualMad =
            residuals
                .map { abs(it - residualMedian) }
                .let(::median)
                .coerceAtLeast(1e-9)

        val outlierThreshold =
            residualMedian +
                3.5 *
                residualMad

        val outlierCount =
            residuals.count {
                it > outlierThreshold
            }

        val spanConfidence =
            min(
                1.0,
                spanSeconds /
                    10.0,
            )

        val residualConfidence =
            (
                1.0 -
                    residualMedian /
                        (config.maximumJitterPercent / 100.0)
            ).coerceIn(0.0, 1.0)

        val sampleConfidence =
            min(
                1.0,
                accepted.size.toDouble() /
                    config.minimumSamplesForStability
                        .coerceAtLeast(1)
                        .toDouble(),
            )

        return TrendEstimate(
            driftPercentPerMinute = driftPercentPerMinute,
            maxStepJumpPercent = maxStepJumpPercent,
            confidence = (
                spanConfidence * 0.30 +
                    residualConfidence * 0.40 +
                    sampleConfidence * 0.30
                ).coerceIn(0.0, 1.0),
            outlierCount = outlierCount,
        )
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val middle = values.size / 2
        return if (values.size % 2 == 0) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    }

    private fun emptyResult(): Result =
        Result(
            status = Status.WARMING_UP,
            sampleCount = 0,
            medianScaleMetersPerUnit = 0.0,
            scaleJitterPercent = 0.0,
            driftPercentPerMinute = 0.0,
            confidence = 0.0,
            authoritative = false,
            warnings =
                listOf(
                    "metric-scale-stability-no-evidence",
                ),
        )
}
