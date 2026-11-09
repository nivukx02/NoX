package com.nivukx.nox.core.ai

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sin

/**
 * Bounded spectral diagnostics for temporal motion traces.
 *
 * This is an auxiliary evidence channel, not a replacement for the kinematic state estimator.
 * It uses a small fixed-frequency bank so the capture path remains deterministic and allocation
 * bounded on mobile hardware.
 */
class TemporalMotionSpectrumAnalyzer(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumSamples: Int = 24,
        val maximumSamples: Int = 96,
        val minimumFrequencyHz: Double = 0.5,
        val maximumFrequencyHz: Double = 12.0,
        val frequencyBins: Int = 24,
        val highFrequencyBoundaryHz: Double = 7.0,
        val minimumVariance: Double = 1e-5,
    ) {
        init {
            require(minimumSamples >= 8)
            require(maximumSamples >= minimumSamples)
            require(minimumFrequencyHz > 0.0)
            require(maximumFrequencyHz > minimumFrequencyHz)
            require(frequencyBins >= 6)
            require(
                highFrequencyBoundaryHz >
                    minimumFrequencyHz &&
                    highFrequencyBoundaryHz <
                    maximumFrequencyHz,
            )
            require(minimumVariance > 0.0)
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val magnitude: Double,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(magnitude.isFinite())
        }
    }

    data class Result(
        val valid: Boolean,
        val dominantFrequencyHz: Double,
        val periodicMotionScore: Double,
        val vibrationRisk: Double,
        val spectralConfidence: Double,
        val dominantBandEnergy: Double,
        val highFrequencyEnergyRatio: Double,
    )

    fun analyze(
        source: List<Sample>,
    ): Result {
        val samples =
            source
                .filter {
                    it.magnitude.isFinite()
                }
                .sortedBy {
                    it.timestampNanos
                }
                .distinctBy {
                    it.timestampNanos
                }
                .takeLast(
                    config.maximumSamples,
                )

        if (
            samples.size <
                config.minimumSamples
        ) {
            return invalid()
        }

        val mean =
            samples
                .map { it.magnitude }
                .average()

        val centered =
            samples.map {
                it.magnitude - mean
            }

        val variance =
            centered
                .map { it * it }
                .average()

        if (
            variance <
                config.minimumVariance
        ) {
            return invalid()
        }

        val dtSeconds =
            samples
                .zipWithNext()
                .map {
                    (
                        it.second.timestampNanos -
                            it.first.timestampNanos
                        ) / 1e9
                }
                .filter {
                    it > 0.0
                }
                .sorted()
                .let { periods ->
                    if (periods.isEmpty()) {
                        return invalid()
                    }
                    periods[
                        periods.size / 2
                    ]
                }

        val sampleRateHz =
            1.0 /
                dtSeconds
                    .coerceAtLeast(
                        1e-6,
                    )

        val frequencyStep =
            (
                config.maximumFrequencyHz -
                    config.minimumFrequencyHz
                ) /
                (
                    config.frequencyBins -
                        1
                    )

        var totalEnergy = 0.0
        var dominantEnergy = 0.0
        var dominantFrequency =
            config.minimumFrequencyHz
        var highFrequencyEnergy = 0.0

        for (
            bin in 0 until config.frequencyBins
        ) {
            val frequency =
                config.minimumFrequencyHz +
                    bin *
                    frequencyStep

            if (
                frequency >=
                    sampleRateHz * 0.5
            ) {
                continue
            }

            val omega =
                2.0 *
                    PI *
                    frequency *
                    dtSeconds

            var real = 0.0
            var imag = 0.0

            centered.forEachIndexed { index, value ->
                val phase =
                    omega * index
                val window =
                    0.5 *
                        (
                            1.0 -
                                cos(
                                    2.0 *
                                        PI *
                                        index /
                                        (
                                            samples.size -
                                                1
                                        )
                                )
                            )

                real +=
                    value *
                    window *
                    cos(phase)
                imag +=
                    value *
                    window *
                    sin(phase)
            }

            val energy =
                (real * real +
                    imag * imag) /
                    samples.size
                    .toDouble()
                    .coerceAtLeast(1.0)

            totalEnergy += energy

            if (
                energy >
                    dominantEnergy
            ) {
                dominantEnergy = energy
                dominantFrequency = frequency
            }

            if (
                frequency >=
                    config.highFrequencyBoundaryHz
            ) {
                highFrequencyEnergy +=
                    energy
            }
        }

        if (totalEnergy <= 1e-12) {
            return invalid()
        }

        val normalizedDominantEnergy =
            (
                dominantEnergy /
                    totalEnergy
                ).coerceIn(
                0.0,
                1.0,
            )

        val highFrequencyRatio =
            (
                highFrequencyEnergy /
                    totalEnergy
                ).coerceIn(
                0.0,
                1.0,
            )

        val periodicity =
            (
                normalizedDominantEnergy *
                    (
                        0.70 +
                            0.30 *
                                autocorrelationPeak(
                                    centered,
                                )
                    )
                ).coerceIn(
                0.0,
                1.0,
            )

        val frequencyReliability =
            (
                1.0 -
                    abs(
                        sampleRateHz -
                            30.0,
                    ) /
                    30.0
                ).coerceIn(
                0.25,
                1.0,
            )

        val confidence =
            (
                0.55 *
                    normalizedDominantEnergy +
                    0.25 *
                    frequencyReliability +
                    0.20 *
                    (
                        1.0 -
                            highFrequencyRatio *
                            0.35
                        )
                ).coerceIn(
                0.0,
                1.0,
            )

        val vibrationRisk =
            (
                highFrequencyRatio *
                    (
                        0.55 +
                            0.45 *
                                normalizedDominantEnergy
                        )
                ).coerceIn(
                0.0,
                1.0,
            )

        return Result(
            valid = true,
            dominantFrequencyHz =
                dominantFrequency,
            periodicMotionScore =
                periodicity,
            vibrationRisk =
                vibrationRisk,
            spectralConfidence =
                confidence,
            dominantBandEnergy =
                normalizedDominantEnergy,
            highFrequencyEnergyRatio =
                highFrequencyRatio,
        )
    }

    private fun autocorrelationPeak(
        centered: List<Double>,
    ): Double {
        val energy =
            centered.sumOf {
                it * it
            }

        if (energy <= 1e-12) {
            return 0.0
        }

        val maxLag =
            minOf(
                centered.size / 2,
                18,
            )

        var best = 0.0
        for (lag in 1..maxLag) {
            var correlation = 0.0
            for (
                index in 0 until
                    centered.size - lag
            ) {
                correlation +=
                    centered[index] *
                        centered[index + lag]
            }

            val normalized =
                (
                    correlation /
                        energy
                    ).coerceIn(
                    -1.0,
                    1.0,
                )

            best =
                max(
                    best,
                    normalized,
                )
        }

        return (
            best + 1.0
            ) * 0.5
    }

    private fun invalid(): Result =
        Result(
            valid = false,
            dominantFrequencyHz = 0.0,
            periodicMotionScore = 0.0,
            vibrationRisk = 0.0,
            spectralConfidence = 0.0,
            dominantBandEnergy = 0.0,
            highFrequencyEnergyRatio = 0.0,
        )
}
