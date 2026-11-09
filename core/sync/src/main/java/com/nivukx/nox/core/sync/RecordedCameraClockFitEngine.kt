package com.nivukx.nox.core.sync

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Robust affine clock fitting for recorded multi-camera evidence.
 *
 * The fitted model is:
 *
 *   masterTime = offset + deviceTime * (1 + driftPpm * 1e-6)
 *
 * A plain offset is not sufficient for long captures because commodity device oscillators drift.
 * The fitter performs a weighted least-squares pass followed by Huber-style residual reweighting.
 */
class RecordedCameraClockFitEngine {

    data class Sample(
        val deviceTimestampNanos: Long,
        val masterTimestampNanos: Long,
        val weight: Double = 1.0,
    ) {
        init {
            require(deviceTimestampNanos >= 0L)
            require(masterTimestampNanos >= 0L)
            require(weight.isFinite() && weight > 0.0)
        }
    }

    data class FitResult(
        val accepted: Boolean,
        val sampleCount: Int,
        val clockOffsetNanos: Long,
        val clockDriftPpm: Double,
        val residualRmsNanos: Double,
        val residualP95Nanos: Double,
        val residualMaxNanos: Double,
        val confidence: Double,
        val warnings: List<String>,
    ) {
        val synchronizationConfidence: Double
            get() = confidence.coerceIn(0.0, 1.0)
    }

    fun fit(
        samples: List<Sample>,
        minimumSamples: Int = 8,
        huberThresholdNanos: Double = 1_500_000.0,
        maximumAcceptedRmsNanos: Double = 8_000_000.0,
    ): FitResult {
        require(minimumSamples >= 3)
        require(huberThresholdNanos > 0.0)
        require(maximumAcceptedRmsNanos > huberThresholdNanos)

        val ordered =
            samples
                .distinctBy {
                    it.deviceTimestampNanos
                }
                .sortedBy {
                    it.deviceTimestampNanos
                }

        if (ordered.size < minimumSamples) {
            return emptyResult(
                sampleCount = ordered.size,
                warning = "CLOCK_FIT_SAMPLE_COUNT_INSUFFICIENT",
            )
        }

        val referenceDeviceNanos =
            ordered[ordered.size / 2]
                .deviceTimestampNanos

        val normalized =
            ordered.map {
                RegressionSample(
                    x =
                        (
                            it.deviceTimestampNanos -
                                referenceDeviceNanos
                            ).toDouble(),
                    y =
                        it.masterTimestampNanos.toDouble(),
                    baseWeight =
                        it.weight,
                )
            }

        var weights =
            normalized.map {
                it.baseWeight
            }

        var fit =
            weightedLeastSquares(
                normalized,
                weights,
            )

        repeat(4) {
            val residuals =
                normalized.map { sample ->
                    sample.y -
                        (
                            fit.intercept +
                                fit.slope * sample.x
                            )
                }

            weights =
                normalized.mapIndexed { index, sample ->
                    val residual =
                        abs(
                            residuals[index],
                        )

                    sample.baseWeight *
                        huberWeight(
                            residual,
                            huberThresholdNanos,
                        )
                }

            fit =
                weightedLeastSquares(
                    normalized,
                    weights,
                )
        }

        val residuals =
            normalized.map { sample ->
                abs(
                    sample.y -
                        (
                            fit.intercept +
                                fit.slope * sample.x
                            ),
                )
            }

        val rms =
            sqrt(
                residuals.mapIndexed { index, residual ->
                    val weight =
                        weights[index]
                            .coerceAtLeast(1e-9)
                    residual * residual * weight
                }.sum() /
                    weights.sum().coerceAtLeast(1e-9),
            )

        val sortedResiduals =
            residuals.sorted()

        val p95 =
            percentile(
                sortedResiduals,
                0.95,
            )
        val maxResidual =
            sortedResiduals.lastOrNull() ?: 0.0

        val driftPpm =
            (
                fit.slope - 1.0
            ) * 1_000_000.0

        val offsetAtReference =
            fit.intercept -
                fit.slope *
                referenceDeviceNanos.toDouble()

        val fitConfidence =
            (
                coverageScore(ordered.size) *
                    residualScore(
                        rms = rms,
                        maximum = maximumAcceptedRmsNanos,
                    ) *
                    driftPlausibilityScore(driftPpm)
                ).coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (rms > maximumAcceptedRmsNanos) {
                    add("CLOCK_FIT_RESIDUAL_RMS_HIGH")
                }
                if (p95 > maximumAcceptedRmsNanos * 1.5) {
                    add("CLOCK_FIT_P95_RESIDUAL_HIGH")
                }
                if (abs(driftPpm) > 100.0) {
                    add("CLOCK_FIT_CLOCK_DRIFT_HIGH")
                }
                if (ordered.size < 16) {
                    add("CLOCK_FIT_LOW_SAMPLE_DENSITY")
                }
            }

        return FitResult(
            accepted =
                rms <= maximumAcceptedRmsNanos &&
                    fitConfidence >= 0.55,
            sampleCount = ordered.size,
            clockOffsetNanos =
                offsetAtReference
                    .roundToLongSafe(),
            clockDriftPpm =
                driftPpm
                    .coerceIn(-2_000.0, 2_000.0),
            residualRmsNanos =
                rms,
            residualP95Nanos =
                p95,
            residualMaxNanos =
                maxResidual,
            confidence =
                fitConfidence,
            warnings =
                warnings.distinct(),
        )
    }

    fun convertDeviceToMaster(
        deviceTimestampNanos: Long,
        fit: FitResult,
    ): Long {
        val slope =
            1.0 +
                fit.clockDriftPpm * 1e-6

        val corrected =
            fit.clockOffsetNanos.toDouble() +
                deviceTimestampNanos.toDouble() * slope

        return corrected
            .coerceIn(
                0.0,
                Long.MAX_VALUE.toDouble(),
            )
            .toLong()
    }

    private data class RegressionSample(
        val x: Double,
        val y: Double,
        val baseWeight: Double,
    )

    private data class RegressionFit(
        val intercept: Double,
        val slope: Double,
    )

    private fun weightedLeastSquares(
        samples: List<RegressionSample>,
        weights: List<Double>,
    ): RegressionFit {
        var sumW = 0.0
        var sumX = 0.0
        var sumY = 0.0
        var sumXX = 0.0
        var sumXY = 0.0

        samples.indices.forEach { index ->
            val weight =
                weights[index]
                    .coerceAtLeast(1e-9)
            val x = samples[index].x
            val y = samples[index].y

            sumW += weight
            sumX += weight * x
            sumY += weight * y
            sumXX += weight * x * x
            sumXY += weight * x * y
        }

        val denominator =
            (
                sumW * sumXX -
                    sumX * sumX
                ).coerceAtLeast(1e-9)

        val slope =
            (
                sumW * sumXY -
                    sumX * sumY
                ) /
                denominator

        val intercept =
            (
                sumY -
                    slope * sumX
                ) /
                sumW.coerceAtLeast(1e-9)

        return RegressionFit(
            intercept = intercept,
            slope = slope,
        )
    }

    private fun huberWeight(
        residualNanos: Double,
        thresholdNanos: Double,
    ): Double =
        if (residualNanos <= thresholdNanos) {
            1.0
        } else {
            thresholdNanos /
                residualNanos.coerceAtLeast(
                    thresholdNanos,
                )
        }

    private fun coverageScore(
        sampleCount: Int,
    ): Double =
        (
            1.0 -
                1.0 /
                (
                    1.0 +
                        sampleCount.toDouble() /
                            12.0
                    )
            ).coerceIn(0.0, 1.0)

    private fun residualScore(
        rms: Double,
        maximum: Double,
    ): Double =
        (
            1.0 -
                rms /
                maximum.coerceAtLeast(1.0)
            ).coerceIn(0.0, 1.0)

    private fun driftPlausibilityScore(
        driftPpm: Double,
    ): Double =
        (
            1.0 -
                abs(driftPpm) /
                    250.0
            ).coerceIn(0.0, 1.0)

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val p =
            probability.coerceIn(0.0, 1.0)
        val position =
            p * (values.size - 1).toDouble()
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

        return values[lower] * (1.0 - fraction) +
            values[upper] * fraction
    }

    private fun emptyResult(
        sampleCount: Int,
        warning: String,
    ): FitResult =
        FitResult(
            accepted = false,
            sampleCount = sampleCount,
            clockOffsetNanos = 0L,
            clockDriftPpm = 0.0,
            residualRmsNanos =
                Double.POSITIVE_INFINITY,
            residualP95Nanos =
                Double.POSITIVE_INFINITY,
            residualMaxNanos =
                Double.POSITIVE_INFINITY,
            confidence = 0.0,
            warnings = listOf(warning),
        )

    private fun Double.roundToLongSafe(): Long =
        when {
            this.isNaN() -> 0L
            this >= Long.MAX_VALUE.toDouble() -> Long.MAX_VALUE
            this <= Long.MIN_VALUE.toDouble() -> Long.MIN_VALUE
            else -> kotlin.math.round(this).toLong()
        }
}
