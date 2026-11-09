package com.nivukx.nox.core.ai

class TemporalMotionSpectrumAnalyzerTest {

    private val analyzer =
        TemporalMotionSpectrumAnalyzer(
            TemporalMotionSpectrumAnalyzer.Config(
                minimumSamples = 24,
                maximumSamples = 96,
                minimumFrequencyHz = 0.5,
                maximumFrequencyHz = 12.0,
                highFrequencyBoundaryHz = 7.0,
            ),
        )

    fun detects_periodic_motion_band() {
        val samples =
            (0 until 60).map { index ->
                TemporalMotionSpectrumAnalyzer.Sample(
                    timestampNanos =
                        1_000_000_000L +
                            index * 33_333_333L,
                    magnitude =
                        0.20 +
                            0.35 *
                            kotlin.math.sin(
                                2.0 *
                                    Math.PI *
                                    2.0 *
                                    index /
                                    30.0,
                            ),
                )
            }

        val result = analyzer.analyze(samples)

        check(result.valid)
        check(result.periodicMotionScore > 0.40)
        check(
            kotlin.math.abs(
                result.dominantFrequencyHz - 2.0,
            ) < 0.45,
        )
        check(result.spectralConfidence > 0.45)
    }

    fun identifies_high_frequency_vibration() {
        val samples =
            (0 until 72).map { index ->
                TemporalMotionSpectrumAnalyzer.Sample(
                    timestampNanos =
                        2_000_000_000L +
                            index * 33_333_333L,
                    magnitude =
                        0.12 +
                            0.25 *
                            kotlin.math.sin(
                                2.0 *
                                    Math.PI *
                                    9.0 *
                                    index /
                                    30.0,
                            ),
                )
            }

        val result = analyzer.analyze(samples)

        check(result.valid)
        check(result.vibrationRisk > 0.45)
    }

    fun rejects_insufficient_history() {
        val result =
            analyzer.analyze(
                listOf(
                    TemporalMotionSpectrumAnalyzer.Sample(
                        timestampNanos = 1_000_000_000L,
                        magnitude = 0.1,
                    ),
                ),
            )

        check(!result.valid)
        check(result.spectralConfidence == 0.0)
    }
}

fun main() {
    val test = TemporalMotionSpectrumAnalyzerTest()
    test.detects_periodic_motion_band()
    test.identifies_high_frequency_vibration()
    test.rejects_insufficient_history()
}
