package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.BenchmarkMetricSample
import com.nivukx.nox.core.model.MotionCaptureBenchmarkMetrics
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Deterministic evaluator for matched-reference motion capture experiments.
 *
 * It consumes measured samples. It never synthesizes a reference trajectory and never converts
 * confidence into a physical accuracy claim.
 */
class MotionCaptureBenchmarkEvaluator {

    fun evaluate(
        samples: List<BenchmarkMetricSample>,
        totalExpectedSamples: Int,
        frameLatenciesMilliseconds: List<Double>,
        sustainedFps: Double,
        thermalRiseCelsius: Double,
        droppedFrameRatio: Double,
    ): MotionCaptureBenchmarkMetrics {
        require(totalExpectedSamples >= 0)
        require(frameLatenciesMilliseconds.all { it.isFinite() && it >= 0.0 })
        require(sustainedFps.isFinite() && sustainedFps >= 0.0)
        require(thermalRiseCelsius.isFinite() && thermalRiseCelsius >= 0.0)
        require(droppedFrameRatio.isFinite() && droppedFrameRatio in 0.0..1.0)

        val errors =
            samples.map {
                it.errorMillimeters
            }.sorted()

        val mean =
            errors.averageOrZero()

        val median =
            percentile(
                errors,
                0.50,
            )

        val p95 =
            percentile(
                errors,
                0.95,
            )

        val p99 =
            percentile(
                errors,
                0.99,
            )

        val maximum =
            errors.lastOrNull() ?: 0.0

        val actorCount =
            samples
                .map { it.actorId }
                .distinct()
                .size

        val identityAccuracy =
            if (samples.isEmpty()) {
                0.0
            } else {
                samples.count {
                    it.identityCorrect
                }.toDouble() /
                    samples.size.toDouble()
            }

        val temporalJitter =
            temporalJitter(
                samples,
            )

        val predictedRatio =
            if (samples.isEmpty()) {
                0.0
            } else {
                samples.count {
                    it.predicted
                }.toDouble() /
                    samples.size.toDouble()
            }

        val visibleRatio =
            if (samples.isEmpty()) {
                0.0
            } else {
                samples.count {
                    it.visible
                }.toDouble() /
                    samples.size.toDouble()
            }

        val coverage =
            if (totalExpectedSamples <= 0) {
                0.0
            } else {
                (
                    samples.size.toDouble() /
                        totalExpectedSamples.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val latency =
            frameLatenciesMilliseconds.averageOrZero()

        return MotionCaptureBenchmarkMetrics(
            sampleCount = samples.size,
            actorCount = actorCount,
            meanErrorMillimeters = mean,
            medianErrorMillimeters = median,
            p95ErrorMillimeters = p95,
            p99ErrorMillimeters = p99,
            maximumErrorMillimeters = maximum,
            identityAccuracy =
                identityAccuracy.coerceIn(
                    0.0,
                    1.0,
                ),
            temporalJitterMillimeters =
                temporalJitter,
            predictedSampleRatio =
                predictedRatio.coerceIn(
                    0.0,
                    1.0,
                ),
            visibleSampleRatio =
                visibleRatio.coerceIn(
                    0.0,
                    1.0,
                ),
            reconstructionCoverage =
                coverage,
            frameLatencyMilliseconds =
                latency,
            sustainedCaptureFps =
                sustainedFps,
            thermalRiseCelsius =
                thermalRiseCelsius,
            droppedFrameRatio =
                droppedFrameRatio,
        )
    }

    private fun temporalJitter(
        samples: List<BenchmarkMetricSample>,
    ): Double {
        if (samples.size < 3) return 0.0

        val grouped =
            samples
                .groupBy {
                    it.actorId to it.jointId
                }

        val deviations = ArrayList<Double>()

        grouped.values.forEach { trajectory ->
            val ordered =
                trajectory.sortedBy {
                    it.timestampNanos
                }

            if (ordered.size < 3) return@forEach

            val velocities =
                ordered
                    .zipWithNext()
                    .map { (a, b) ->
                        val dt =
                            (
                                b.timestampNanos -
                                    a.timestampNanos
                                ).coerceAtLeast(1L) /
                                1_000_000_000.0

                        val dx =
                            b.estimatedX -
                                a.estimatedX
                        val dy =
                            b.estimatedY -
                                a.estimatedY
                        val dz =
                            b.estimatedZ -
                                a.estimatedZ

                        sqrt(
                            dx * dx +
                                dy * dy +
                                dz * dz,
                        ) / dt
                    }

            if (velocities.size < 2) return@forEach

            val mean =
                velocities.average()

            deviations +=
                velocities.map {
                    abs(
                        it - mean,
                    ) * 1000.0
                }
        }

        return deviations.averageOrZero()
    }

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return 0.0

        val p =
            probability.coerceIn(
                0.0,
                1.0,
            )

        val position =
            p *
                (
                    values.size - 1
                    ).toDouble()

        val lower =
            position.toInt()

        val upper =
            minOf(
                values.lastIndex,
                lower + 1,
            )

        val fraction =
            position -
                lower.toDouble()

        return values[lower] *
            (1.0 - fraction) +
            values[upper] *
            fraction
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) {
            0.0
        } else {
            average()
        }
}
