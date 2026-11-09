package com.nivukx.nox.core.sync

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Estimates signed sub-frame delay between two motion trajectories.
 *
 * The returned offset uses this convention:
 * positive offset means the second trajectory's event time lags the first trajectory by that amount.
 * The second trajectory is therefore sampled at (firstTimestamp + offset) during correlation.
 *
 * This is a calibration-stage estimator. It produces evidence; it never mutates authoritative
 * sensor timestamps.
 */
class SubframeMotionAlignmentEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumSamples: Int = 16,
        val maximumOffsetNanos: Long = 80_000_000L,
        val minimumCorrelation: Double = 0.45,
        val minimumMargin: Double = 0.015,
        val maximumSamplingJitterRatio: Double = 0.08,
    ) {
        init {
            require(minimumSamples >= 6)
            require(maximumOffsetNanos > 0L)
            require(minimumCorrelation in -1.0..1.0)
            require(minimumMargin >= 0.0)
            require(maximumSamplingJitterRatio >= 0.0)
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val value: Double,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(value.isFinite())
        }
    }

    data class Result(
        val offsetNanos: Double,
        val confidence: Double,
        val correlation: Double,
        val overlapSamples: Int,
        val samplePeriodNanos: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    fun estimate(
        first: List<Sample>,
        second: List<Sample>,
    ): Result {
        val a = sanitize(first)
        val b = sanitize(second)

        if (a.size < config.minimumSamples ||
            b.size < config.minimumSamples
        ) {
            return rejected(
                warning = "insufficient motion-alignment samples",
            )
        }

        val periodNanos = medianPeriodNanos(a)
        if (!periodNanos.isFinite() || periodNanos <= 0.0) {
            return rejected(
                warning = "unable to estimate a stable sample period",
            )
        }

        val firstJitterRatio = samplingJitterRatio(a, periodNanos)
        val secondJitterRatio = samplingJitterRatio(b, periodNanos)
        val jitterRatio =
            max(
                firstJitterRatio,
                secondJitterRatio,
            )

        val regularityPenalty =
            (
                1.0 -
                    jitterRatio /
                        config.maximumSamplingJitterRatio
                            .coerceAtLeast(1e-9)
                ).coerceIn(0.0, 1.0)

        val stepNanos =
            periodNanos.coerceAtLeast(1.0)
        val maxLagIndex =
            (
                config.maximumOffsetNanos /
                    stepNanos
            ).toInt().coerceAtLeast(1)

        val scores = ArrayList<Pair<Double, Double>>(
            maxLagIndex * 2 + 1,
        )

        for (index in -maxLagIndex..maxLagIndex) {
            val offset =
                index * stepNanos
            val score =
                correlationAtOffset(
                    first = a,
                    second = b,
                    offsetNanos = offset,
                )
            if (score.isFinite()) {
                scores += offset to score
            }
        }

        if (scores.isEmpty()) {
            return rejected(
                warning = "no overlapping trajectory window",
            )
        }

        val sorted =
            scores.sortedByDescending { it.second }
        val best =
            sorted.first()
        val secondBest =
            sorted.getOrNull(1)?.second
                ?: best.second

        val bestIndex =
            scores.indexOfFirst {
                abs(it.first - best.first) < 0.5
            }

        val refinedFraction =
            if (bestIndex > 0 &&
                bestIndex < scores.lastIndex
            ) {
                val left =
                    scores[bestIndex - 1].second
                val center =
                    scores[bestIndex].second
                val right =
                    scores[bestIndex + 1].second
                val denominator =
                    left -
                        2.0 * center +
                        right

                if (abs(denominator) > 1e-9) {
                    (
                        0.5 *
                            (left - right) /
                            denominator
                        ).coerceIn(
                            -0.5,
                            0.5,
                        )
                } else {
                    0.0
                }
            } else {
                0.0
            }

        val refinedOffset =
            best.first +
                refinedFraction *
                stepNanos

        val overlap =
            overlapCount(
                first = a,
                second = b,
                offsetNanos = refinedOffset,
            )

        val correlation =
            correlationAtOffset(
                first = a,
                second = b,
                offsetNanos = refinedOffset,
            )

        val margin =
            (
                best.second -
                    secondBest
                ).coerceIn(0.0, 1.0)

        val overlapConfidence =
            (
                overlap.toDouble() /
                    min(
                        a.size,
                        b.size,
                    ).coerceAtLeast(1)
                ).coerceIn(0.0, 1.0)

        val correlationConfidence =
            ((correlation + 1.0) * 0.5)
                .coerceIn(0.0, 1.0)

        val marginConfidence =
            (
                margin /
                    0.20
                ).coerceIn(0.0, 1.0)

        val confidence =
            (
                0.48 * correlationConfidence +
                    0.22 * overlapConfidence +
                    0.18 * marginConfidence +
                    0.12 * regularityPenalty
                ).coerceIn(0.0, 1.0)

        val strongPeakOverride =
            correlation >=
                0.92 &&
                overlap >=
                config.minimumSamples

        val accepted =
            correlation >=
                config.minimumCorrelation &&
                (margin >= config.minimumMargin || strongPeakOverride) &&
                overlap >=
                config.minimumSamples &&
                confidence >=
                config.minimumCorrelation.coerceAtLeast(0.0)

        val warnings =
            buildList {
                if (jitterRatio >
                    config.maximumSamplingJitterRatio
                ) {
                    add("sampling jitter is high")
                }
                if (overlap <
                    config.minimumSamples
                ) {
                    add("trajectory overlap is too short")
                }
                if (margin <
                    config.minimumMargin
                ) {
                    add("correlation peak is ambiguous")
                }
                if (correlation <
                    config.minimumCorrelation
                ) {
                    add("correlation is below the acceptance threshold")
                }
            }

        return Result(
            offsetNanos = refinedOffset,
            confidence = confidence,
            correlation = correlation,
            overlapSamples = overlap,
            samplePeriodNanos = periodNanos,
            accepted = accepted,
            warnings = warnings,
        )
    }

    private fun sanitize(
        samples: List<Sample>,
    ): List<Sample> =
        samples
            .filter {
                it.value.isFinite()
            }
            .sortedBy { it.timestampNanos }
            .distinctBy { it.timestampNanos }

    private fun medianPeriodNanos(
        samples: List<Sample>,
    ): Double {
        if (samples.size < 2) {
            return Double.NaN
        }

        val periods =
            samples
                .zipWithNext { a, b ->
                    (b.timestampNanos -
                        a.timestampNanos)
                        .toDouble()
                }
                .filter { it > 0.0 }

        if (periods.isEmpty()) {
            return Double.NaN
        }

        return periods.sorted()[periods.size / 2]
    }

    private fun samplingJitterRatio(
        samples: List<Sample>,
        medianPeriodNanos: Double,
    ): Double {
        val periods =
            samples
                .zipWithNext { a, b ->
                    (b.timestampNanos -
                        a.timestampNanos)
                        .toDouble()
                }
                .filter { it > 0.0 }

        if (periods.isEmpty()) {
            return 1.0
        }

        val deviations =
            periods.map {
                abs(it - medianPeriodNanos)
            }

        return deviations
            .sorted()[deviations.size / 2] /
            medianPeriodNanos.coerceAtLeast(
                1.0,
            )
    }

    private fun correlationAtOffset(
        first: List<Sample>,
        second: List<Sample>,
        offsetNanos: Double,
    ): Double {
        val pairs =
            ArrayList<Pair<Double, Double>>(
                min(first.size, second.size),
            )

        first.forEach { source ->
            val target =
                source.timestampNanos.toDouble() +
                    offsetNanos
            val value =
                interpolate(
                    second,
                    target,
                )
                    ?: return@forEach
            pairs += source.value to value
        }

        if (pairs.size < config.minimumSamples) {
            return Double.NaN
        }

        val meanA =
            pairs.sumOf { it.first } /
                pairs.size
        val meanB =
            pairs.sumOf { it.second } /
                pairs.size

        var numerator = 0.0
        var energyA = 0.0
        var energyB = 0.0

        pairs.forEach { (a, b) ->
            val da = a - meanA
            val db = b - meanB
            numerator += da * db
            energyA += da * da
            energyB += db * db
        }

        val denominator =
            sqrt(
                energyA *
                    energyB,
            )

        if (denominator <= 1e-12) {
            return -1.0
        }

        return (
            numerator /
                denominator
            ).coerceIn(-1.0, 1.0)
    }

    private fun overlapCount(
        first: List<Sample>,
        second: List<Sample>,
        offsetNanos: Double,
    ): Int =
        first.count { source ->
            interpolate(
                second,
                source.timestampNanos.toDouble() +
                    offsetNanos,
            ) != null
        }

    private fun interpolate(
        samples: List<Sample>,
        timestampNanos: Double,
    ): Double? {
        if (samples.isEmpty()) return null
        if (timestampNanos <
            samples.first().timestampNanos
        ) {
            return null
        }
        if (timestampNanos >
            samples.last().timestampNanos
        ) {
            return null
        }

        var low = 0
        var high = samples.lastIndex

        while (low <= high) {
            val mid =
                (low + high) ushr 1
            val timestamp =
                samples[mid]
                    .timestampNanos
                    .toDouble()

            when {
                timestamp <
                    timestampNanos -> {
                    low = mid + 1
                }
                timestamp >
                    timestampNanos -> {
                    high = mid - 1
                }
                else -> {
                    return samples[mid].value
                }
            }
        }

        val rightIndex = low
        val leftIndex = low - 1
        if (leftIndex < 0 ||
            rightIndex > samples.lastIndex
        ) {
            return null
        }

        val left =
            samples[leftIndex]
        val right =
            samples[rightIndex]
        val span =
            (
                right.timestampNanos -
                    left.timestampNanos
                ).toDouble()

        if (span <= 0.0) {
            return left.value
        }

        val alpha =
            (
                timestampNanos -
                    left.timestampNanos
                ) / span

        return left.value +
            (
                right.value -
                    left.value
                ) * alpha
    }

    private fun rejected(
        warning: String,
    ): Result =
        Result(
            offsetNanos = 0.0,
            confidence = 0.0,
            correlation = 0.0,
            overlapSamples = 0,
            samplePeriodNanos = 0.0,
            accepted = false,
            warnings = listOf(warning),
        )
}
