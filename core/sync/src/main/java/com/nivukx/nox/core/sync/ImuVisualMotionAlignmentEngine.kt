package com.nivukx.nox.core.sync

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Estimates the relative time offset between two motion streams.
 *
 * The estimator is deliberately modality-neutral. One stream normally comes from IMU angular
 * velocity, while the other comes from visual/VIO angular velocity. The result is evidence for
 * sensor-fusion scheduling and calibration. It does not rewrite either sensor clock.
 */
class ImuVisualMotionAlignmentEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumSamples: Int = 20,
        val maximumOffsetNanos: Long = 50_000_000L,
        val searchStepNanos: Long = 1_000_000L,
        val minimumCorrelation: Double = 0.50,
        val minimumEnergy: Double = 1e-5,
        val minimumConfidence: Double = 0.42,
        val minimumPeakMargin: Double = 0.01,
    ) {
        init {
            require(minimumSamples >= 8)
            require(maximumOffsetNanos > 0L)
            require(searchStepNanos > 0L)
            require(searchStepNanos <= maximumOffsetNanos)
            require(minimumCorrelation in -1.0..1.0)
            require(minimumEnergy > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(minimumPeakMargin >= 0.0)
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val angularSpeedRadiansPerSecond: Double,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(angularSpeedRadiansPerSecond.isFinite())
        }
    }

    data class Result(
        val offsetNanos: Double,
        val confidence: Double,
        val correlation: Double,
        val peakMargin: Double,
        val overlapSamples: Int,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    fun estimate(
        imu: List<Sample>,
        visual: List<Sample>,
    ): Result {
        val a = sanitize(imu)
        val b = sanitize(visual)

        if (
            a.size <
                config.minimumSamples ||
            b.size <
                config.minimumSamples
        ) {
            return rejected(
                "insufficient IMU/visual motion samples",
            )
        }

        val energyA = signalEnergy(a)
        val energyB = signalEnergy(b)

        if (
            energyA <
                config.minimumEnergy ||
            energyB <
                config.minimumEnergy
        ) {
            return rejected(
                "motion signal energy is too low",
            )
        }

        val periodNanos =
            max(
                1.0,
                min(
                    medianPeriodNanos(a),
                    medianPeriodNanos(b),
                ),
            )

        val maxLagSteps =
            (
                config.maximumOffsetNanos /
                    config.searchStepNanos.toDouble()
                ).toInt()
                .coerceAtLeast(1)

        val scores =
            ArrayList<Pair<Double, Double>>(
                maxLagSteps * 2 + 1,
            )

        for (step in -maxLagSteps..maxLagSteps) {
            val offset =
                step *
                    config.searchStepNanos.toDouble()

            val correlation =
                correlationAtOffset(
                    source = a,
                    target = b,
                    offsetNanos = offset,
                )

            if (correlation.isFinite()) {
                scores += offset to correlation
            }
        }

        if (scores.isEmpty()) {
            return rejected(
                "motion streams have no temporal overlap",
            )
        }

        val sorted =
            scores.sortedByDescending {
                it.second
            }

        val best = sorted.first()
        val second =
            sorted.getOrNull(1)
                ?: best

        val peakMargin =
            (
                best.second -
                    second.second
                ).coerceAtLeast(0.0)

        val bestIndex =
            scores.indexOfFirst {
                abs(
                    it.first -
                        best.first,
                ) < 0.5
            }

        val refinedOffset =
            refinePeak(
                scores = scores,
                bestIndex = bestIndex,
                stepNanos = config.searchStepNanos.toDouble(),
            )

        val refinedCorrelation =
            correlationAtOffset(
                source = a,
                target = b,
                offsetNanos = refinedOffset,
            )

        val overlap =
            overlapCount(
                source = a,
                target = b,
                offsetNanos = refinedOffset,
            )

        val overlapScore =
            (
                overlap.toDouble() /
                    min(
                        a.size,
                        b.size,
                    ).coerceAtLeast(1)
                ).coerceIn(
                0.0,
                1.0,
            )

        val correlationScore =
            (
                refinedCorrelation +
                    1.0
                ) * 0.5

        val sharpnessScore =
            (
                peakMargin /
                    0.12
                ).coerceIn(
                0.0,
                1.0,
            )

        val energyScore =
            (
                sqrt(
                    energyA *
                        energyB,
                ) /
                    10.0
                ).coerceIn(
                0.0,
                1.0,
            )

        val confidence =
            (
                0.46 * correlationScore +
                    0.20 * overlapScore +
                    0.22 * sharpnessScore +
                    0.12 * energyScore
                ).coerceIn(
                0.0,
                1.0,
            )

        val strongAlignment =
            refinedCorrelation >=
                (config.minimumCorrelation + 0.18).coerceAtMost(0.95) &&
                overlapScore >= 0.85

        val accepted =
            refinedCorrelation >=
                config.minimumCorrelation &&
                overlap >=
                config.minimumSamples &&
                (
                    peakMargin >=
                        config.minimumPeakMargin ||
                        strongAlignment
                ) &&
                confidence >=
                config.minimumConfidence

        val warnings =
            buildList {
                if (
                    refinedCorrelation <
                        config.minimumCorrelation
                ) {
                    add(
                        "motion correlation is below the acceptance threshold",
                    )
                }
                if (
                    peakMargin <
                        config.minimumPeakMargin
                ) {
                    add(
                        "offset peak is ambiguous",
                    )
                }
                if (
                    overlap <
                        config.minimumSamples
                ) {
                    add(
                        "temporal overlap is too short",
                    )
                }
            }

        return Result(
            offsetNanos = refinedOffset,
            confidence = confidence,
            correlation =
                refinedCorrelation.coerceIn(
                    -1.0,
                    1.0,
                ),
            peakMargin = peakMargin,
            overlapSamples = overlap,
            accepted = accepted,
            warnings = warnings,
        )
    }

    private fun sanitize(
        input: List<Sample>,
    ): List<Sample> =
        input
            .filter {
                it.angularSpeedRadiansPerSecond
                    .isFinite()
            }
            .sortedBy {
                it.timestampNanos
            }
            .distinctBy {
                it.timestampNanos
            }

    private fun signalEnergy(
        samples: List<Sample>,
    ): Double {
        if (samples.isEmpty()) return 0.0

        val mean =
            samples
                .map {
                    it.angularSpeedRadiansPerSecond
                }
                .average()

        return samples.sumOf {
            val d =
                it.angularSpeedRadiansPerSecond -
                    mean
            d * d
        } / samples.size
    }

    private fun medianPeriodNanos(
        samples: List<Sample>,
    ): Double {
        val periods =
            samples
                .zipWithNext()
                .map {
                    (
                        it.second.timestampNanos -
                            it.first.timestampNanos
                        ).toDouble()
                }
                .filter {
                    it > 0.0
                }

        return if (
            periods.isEmpty()
        ) {
            1.0
        } else {
            periods.sorted()[
                periods.size / 2
            ]
        }
    }

    private fun correlationAtOffset(
        source: List<Sample>,
        target: List<Sample>,
        offsetNanos: Double,
    ): Double {
        val pairs =
            ArrayList<Pair<Double, Double>>()

        source.forEach { sample ->
            val targetTime =
                sample.timestampNanos
                    .toDouble() +
                    offsetNanos

            val targetValue =
                interpolate(
                    target,
                    targetTime,
                ) ?: return@forEach

            pairs +=
                sample.angularSpeedRadiansPerSecond to
                    targetValue
        }

        if (
            pairs.size <
                config.minimumSamples
        ) {
            return Double.NaN
        }

        val meanA =
            pairs
                .sumOf { it.first } /
                pairs.size
        val meanB =
            pairs
                .sumOf { it.second } /
                pairs.size

        var numerator = 0.0
        var energyA = 0.0
        var energyB = 0.0

        pairs.forEach { pair ->
            val a =
                pair.first -
                    meanA
            val b =
                pair.second -
                    meanB

            numerator += a * b
            energyA += a * a
            energyB += b * b
        }

        val denominator =
            sqrt(
                energyA *
                    energyB,
            )

        if (
            denominator <=
                config.minimumEnergy
        ) {
            return Double.NaN
        }

        return (
            numerator /
                denominator
            ).coerceIn(
            -1.0,
            1.0,
        )
    }

    private fun refinePeak(
        scores: List<Pair<Double, Double>>,
        bestIndex: Int,
        stepNanos: Double,
    ): Double {
        if (
            bestIndex <= 0 ||
            bestIndex >=
            scores.lastIndex
        ) {
            return scores[
                bestIndex.coerceIn(
                    0,
                    scores.lastIndex,
                )
            ].first
        }

        val left =
            scores[
                bestIndex - 1
            ].second
        val center =
            scores[bestIndex].second
        val right =
            scores[
                bestIndex + 1
            ].second

        val denominator =
            left -
                2.0 * center +
                right

        if (
            abs(denominator) <=
                1e-9
        ) {
            return scores[
                bestIndex
            ].first
        }

        val fraction =
            (
                0.5 *
                    (left - right) /
                    denominator
                ).coerceIn(
                -0.5,
                0.5,
            )

        return scores[
            bestIndex
        ].first +
            fraction *
            stepNanos
    }

    private fun overlapCount(
        source: List<Sample>,
        target: List<Sample>,
        offsetNanos: Double,
    ): Int =
        source.count {
            interpolate(
                target,
                it.timestampNanos.toDouble() +
                    offsetNanos,
            ) != null
        }

    private fun interpolate(
        samples: List<Sample>,
        timestampNanos: Double,
    ): Double? {
        if (samples.isEmpty()) return null

        if (
            timestampNanos <
                samples.first()
                    .timestampNanos ||
            timestampNanos >
                samples.last()
                    .timestampNanos
        ) {
            return null
        }

        var low = 0
        var high = samples.lastIndex

        while (low <= high) {
            val mid =
                (low + high) ushr 1
            val time =
                samples[mid]
                    .timestampNanos
                    .toDouble()

            when {
                time < timestampNanos -> {
                    low = mid + 1
                }

                time > timestampNanos -> {
                    high = mid - 1
                }

                else -> {
                    return samples[mid]
                        .angularSpeedRadiansPerSecond
                }
            }
        }

        val rightIndex = low
        val leftIndex = low - 1

        if (
            leftIndex < 0 ||
            rightIndex >
            samples.lastIndex
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

        if (
            span <= 0.0
        ) {
            return left
                .angularSpeedRadiansPerSecond
        }

        val alpha =
            (
                timestampNanos -
                    left.timestampNanos
                ) / span

        return left.angularSpeedRadiansPerSecond +
            (
                right.angularSpeedRadiansPerSecond -
                    left.angularSpeedRadiansPerSecond
                ) * alpha
    }

    private fun rejected(
        warning: String,
    ): Result =
        Result(
            offsetNanos = 0.0,
            confidence = 0.0,
            correlation = 0.0,
            peakMargin = 0.0,
            overlapSamples = 0,
            accepted = false,
            warnings = listOf(warning),
        )
}
