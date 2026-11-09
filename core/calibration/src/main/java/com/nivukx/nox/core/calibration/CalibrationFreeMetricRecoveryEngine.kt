package com.nivukx.nox.core.calibration

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Conservative metric-scale recovery for calibration-light or calibration-free capture.
 *
 * Relative reconstruction is never mislabeled as metric. An explicit metric anchor can qualify
 * calibrated metric mode; otherwise robust anthropometric segment priors can provide a
 * scale-anchored metric estimate with lower confidence and visible provenance.
 */
class CalibrationFreeMetricRecoveryEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class Mode {
        CALIBRATED_METRIC,
        SCALE_ANCHORED_METRIC,
        RELATIVE_ONLY,
        BLOCKED,
    }

    data class Config(
        val minimumConfidence: Double = 0.20,
        val minimumAnchorConfidence: Double = 0.70,
        val maximumMetricAnchors: Int = 128,
        val minimumSegmentObservations: Int = 3,
        val maximumSegmentObservations: Int = 256,
        val robustTrimFraction: Double = 0.15,
        val maxScaleDeviationFromMedian: Double = 0.30,
        val maximumAbsoluteScaleMetersPerUnit: Double = 20.0,
        val minimumMetricScaleMetersPerUnit: Double = 1e-4,
    ) {
        init {
            require(minimumConfidence in 0.0..1.0)
            require(minimumAnchorConfidence in 0.0..1.0)
            require(maximumMetricAnchors in 2..128)
            require(minimumSegmentObservations > 0)
            require(maximumSegmentObservations >= minimumSegmentObservations)
            require(robustTrimFraction in 0.0..0.45)
            require(maxScaleDeviationFromMedian in 0.05..1.0)
            require(maximumAbsoluteScaleMetersPerUnit > 0.0)
            require(minimumMetricScaleMetersPerUnit > 0.0)
        }
    }

    data class MetricAnchor(
        val metricDistanceMeters: Double,
        val observedDistanceUnits: Double,
        val confidence: Double,
    ) {
        init {
            require(metricDistanceMeters.isFinite() && metricDistanceMeters > 0.0)
            require(observedDistanceUnits.isFinite() && observedDistanceUnits > 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }

        val scaleMetersPerUnit: Double
            get() = metricDistanceMeters / observedDistanceUnits
    }

    data class SegmentObservation(
        val observedDistanceUnits: Double,
        val expectedMetricDistanceMeters: Double,
        val confidence: Double,
    ) {
        init {
            require(observedDistanceUnits.isFinite() && observedDistanceUnits > 0.0)
            require(
                expectedMetricDistanceMeters.isFinite() &&
                    expectedMetricDistanceMeters > 0.0,
            )
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }

        val impliedScaleMetersPerUnit: Double
            get() = expectedMetricDistanceMeters / observedDistanceUnits
    }

    data class Input(
        val timestampNanos: Long,
        val calibrated: Boolean,
        val metricAnchors: List<MetricAnchor>,
        val segmentObservations: List<SegmentObservation>,
    ) {
        init {
            require(timestampNanos >= 0L)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val mode: Mode,
        val scaleMetersPerUnit: Double?,
        val confidence: Double,
        val acceptedAnchors: Int,
        val acceptedSegments: Int,
        val rejectedSegments: Int,
        val robustScaleSpread: Double,
        val metricProvenance: String,
        val warnings: List<String>,
        val temporalScaleStability:
            MetricScaleTemporalStabilityEngine.Result =
            MetricScaleTemporalStabilityEngine.Result(
                status =
                    MetricScaleTemporalStabilityEngine.Status.WARMING_UP,
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
            ),
    )

    private var latest: Result? = null
    private var closed = false

    private val temporalScaleStabilityEngine =
        MetricScaleTemporalStabilityEngine()

    @Synchronized
    fun recover(input: Input): Result {
        check(!closed) { "CalibrationFreeMetricRecoveryEngine is closed" }

        fun publish(candidate: Result): Result {
            val scale =
                candidate.scaleMetersPerUnit

            if (
                scale == null ||
                    !scale.isFinite() ||
                    scale <= 0.0
            ) {
                latest = candidate
                return candidate
            }

            val temporal =
                temporalScaleStabilityEngine.observe(
                    timestampNanos = candidate.timestampNanos,
                    scaleMetersPerUnit = scale,
                    confidence = candidate.confidence,
                )

            val warnings =
                (
                    candidate.warnings +
                        temporal.warnings
                    ).distinct()

            val published =
                candidate.copy(
                    warnings = warnings,
                    temporalScaleStability = temporal,
                )

            latest = published
            return published
        }

        latest?.let { previous ->
            if (input.timestampNanos <= previous.timestampNanos) {
                return previous.copy(
                    warnings =
                        (previous.warnings +
                            "timestamp-reversal" +
                            "stale-frame-rejected")
                            .distinct(),
                )
            }
        }

        val boundedAnchors =
            input.metricAnchors
                .asSequence()
                .filter {
                    it.confidence >=
                        config.minimumAnchorConfidence
                }
                .take(config.maximumMetricAnchors)
                .toList()

        if (input.calibrated && boundedAnchors.isNotEmpty()) {
            val scale = robustWeightedScale(
                boundedAnchors.map { it.scaleMetersPerUnit to it.confidence },
            )
            if (scale != null) {
                val spread =
                    robustSpread(
                        boundedAnchors.map {
                            it.scaleMetersPerUnit
                        },
                        scale,
                    )
                val confidence =
                    max(
                        0.92,
                        (
                            boundedAnchors.map { it.confidence }.average() * 0.74 +
                                (1.0 - spread).coerceIn(0.0, 1.0) * 0.26
                            ).coerceIn(0.0, 1.0),
                    ).coerceAtMost(1.0)

                val result = Result(
                    timestampNanos = input.timestampNanos,
                    mode = Mode.CALIBRATED_METRIC,
                    scaleMetersPerUnit = scale,
                    confidence = confidence,
                    acceptedAnchors = boundedAnchors.size,
                    acceptedSegments = 0,
                    rejectedSegments = 0,
                    robustScaleSpread = spread,
                    metricProvenance = "explicit-metric-anchor",
                    warnings = buildList {
                        if (spread > 0.10) {
                            add("metric-anchor-spread-elevated")
                        }
                    },
                )
                return publish(result)
            }
        }

        val boundedSegments =
            input.segmentObservations
                .asSequence()
                .filter {
                    it.confidence >=
                        config.minimumConfidence
                }
                .take(config.maximumSegmentObservations)
                .toList()

        if (boundedSegments.size < config.minimumSegmentObservations) {
            val result = Result(
                timestampNanos = input.timestampNanos,
                mode = if (input.calibrated) Mode.BLOCKED else Mode.RELATIVE_ONLY,
                scaleMetersPerUnit = null,
                confidence = 0.0,
                acceptedAnchors = boundedAnchors.size,
                acceptedSegments = 0,
                rejectedSegments = boundedSegments.size,
                robustScaleSpread = 1.0,
                metricProvenance = "no-sufficient-scale-evidence",
                warnings = listOf(
                    "metric-anchor-required",
                    "insufficient-segment-scale-evidence",
                ),
            )
            return publish(result)
        }

        val implied =
            boundedSegments.map {
                it.impliedScaleMetersPerUnit
            }

        val median =
            median(implied)
        val accepted =
            boundedSegments.filter {
                val ratio =
                    abs(
                        it.impliedScaleMetersPerUnit -
                            median,
                    ) /
                        median.coerceAtLeast(
                            config.minimumMetricScaleMetersPerUnit,
                        )
                ratio <= config.maxScaleDeviationFromMedian
            }

        if (accepted.size < config.minimumSegmentObservations) {
            val result = Result(
                timestampNanos = input.timestampNanos,
                mode = if (input.calibrated) Mode.BLOCKED else Mode.RELATIVE_ONLY,
                scaleMetersPerUnit = null,
                confidence = 0.0,
                acceptedAnchors = boundedAnchors.size,
                acceptedSegments = accepted.size,
                rejectedSegments = boundedSegments.size - accepted.size,
                robustScaleSpread = 1.0,
                metricProvenance = "segment-scale-inconsistent",
                warnings = listOf(
                    "segment-scale-inconsistent",
                    "metric-anchor-required-for-fallback",
                ),
            )
            latest = result
            return result
        }

        val scale =
            robustWeightedScale(
                accepted.map {
                    it.impliedScaleMetersPerUnit to
                        it.confidence
                },
            )

        if (
            scale == null ||
            scale < config.minimumMetricScaleMetersPerUnit ||
            scale > config.maximumAbsoluteScaleMetersPerUnit
        ) {
            val result = Result(
                timestampNanos = input.timestampNanos,
                mode = Mode.RELATIVE_ONLY,
                scaleMetersPerUnit = null,
                confidence = 0.0,
                acceptedAnchors = boundedAnchors.size,
                acceptedSegments = accepted.size,
                rejectedSegments = boundedSegments.size - accepted.size,
                robustScaleSpread = 1.0,
                metricProvenance = "scale-out-of-bounds",
                warnings = listOf(
                    "metric-scale-out-of-bounds",
                    "relative-geometry-only",
                ),
            )
            latest = result
            return result
        }

        val spread =
            robustSpread(
                accepted.map {
                    it.impliedScaleMetersPerUnit
                },
                scale,
            )

        val segmentConfidence =
            accepted.map { it.confidence }.average()

        val consistencyConfidence =
            (1.0 - spread).coerceIn(0.0, 1.0)

        val quantityConfidence =
            min(
                1.0,
                accepted.size.toDouble() /
                    max(
                        config.minimumSegmentObservations,
                        8,
                    ).toDouble(),
            )

        val confidence =
            (
                segmentConfidence * 0.46 +
                    consistencyConfidence * 0.34 +
                    quantityConfidence * 0.20
                ).coerceIn(0.0, 1.0)

        val result = Result(
            timestampNanos = input.timestampNanos,
            mode = Mode.SCALE_ANCHORED_METRIC,
            scaleMetersPerUnit = scale,
            confidence = confidence,
            acceptedAnchors = boundedAnchors.size,
            acceptedSegments = accepted.size,
            rejectedSegments =
                boundedSegments.size - accepted.size,
            robustScaleSpread = spread,
            metricProvenance = "anthropometric-segment-prior",
            warnings = buildList {
                add("metric-scale-is-prior-derived")
                if (confidence < 0.70) {
                    add("metric-scale-confidence-moderate")
                }
                if (spread > 0.12) {
                    add("segment-scale-spread-elevated")
                }
            },
        )
        return publish(result)
    }

    @Synchronized
    fun latest(): Result? = latest

    @Synchronized
    fun reset() {
        latest = null
        temporalScaleStabilityEngine.reset()
    }

    @Synchronized
    override fun close() {
        latest = null
        temporalScaleStabilityEngine.close()
        closed = true
    }

    private fun robustWeightedScale(
        values: List<Pair<Double, Double>>,
    ): Double? {
        if (values.isEmpty()) return null

        val sorted = values
            .filter { it.first.isFinite() && it.second > 0.0 }
            .sortedBy { it.first }

        if (sorted.isEmpty()) return null

        val trim =
            (sorted.size * config.robustTrimFraction)
                .toInt()
                .coerceAtMost((sorted.size - 1) / 2)

        val trimmed =
            if (trim == 0) {
                sorted
            } else {
                sorted.drop(trim).dropLast(trim)
            }

        val denominator =
            trimmed.sumOf { it.second }
                .coerceAtLeast(1e-9)

        return (
            trimmed.sumOf {
                it.first * it.second
            } / denominator
        ).takeIf { it.isFinite() && it > 0.0 }
    }

    private fun robustSpread(
        values: List<Double>,
        center: Double,
    ): Double {
        if (values.size < 2 || !center.isFinite() || center <= 0.0) {
            return 1.0
        }

        val meanAbsoluteDeviation =
            values
                .map {
                    abs(it - center)
                }
                .average()

        return (
            meanAbsoluteDeviation /
                center
            ).coerceIn(0.0, 1.0)
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }
}
