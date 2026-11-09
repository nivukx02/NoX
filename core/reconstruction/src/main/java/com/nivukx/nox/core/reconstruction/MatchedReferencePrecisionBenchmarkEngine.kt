package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Matched-condition precision comparison for markerless RGB capture against an external reference.
 *
 * This engine is intentionally descriptive rather than promotional. It requires paired observations
 * from the same time/joint/sample identity and keeps the uncertainty of both systems visible.
 *
 * It does not declare a universal winner. It answers narrower engineering questions:
 * - what are the P50/P95/P99 errors for each system?
 * - how large are the paired differences?
 * - are the tails materially different?
 * - is the observed separation larger than the declared measurement uncertainty?
 * - does the comparison have enough matched samples to be credible?
 *
 * Recommended references include calibrated marker-based optical systems, robotic references, or
 * high-precision calibration rigs with explicitly documented uncertainty.
 */
class MatchedReferencePrecisionBenchmarkEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumSamples: Int = 120,
        val minimumMatchedFraction: Double = 0.90,
        val maximumReferenceGapMilliseconds: Double = 8.0,
        val maximumReferenceUncertaintyMillimeters: Double = 1.50,
        val minimumTailSeparationMillimeters: Double = 0.10,
        val minimumMedianSeparationMillimeters: Double = 0.05,
        val minimumRuns: Int = 3,
        val maximumRunToRunSpreadMillimeters: Double = 0.50,
        val madMultiplierForSeparation: Double = 3.0,
        val maximumOutlierFraction: Double = 0.05,
        val referenceRequired: Boolean = true,
    ) {
        init {
            require(minimumSamples >= 20)
            require(minimumMatchedFraction in 0.5..1.0)
            require(maximumReferenceGapMilliseconds > 0.0)
            require(maximumReferenceUncertaintyMillimeters > 0.0)
            require(minimumTailSeparationMillimeters >= 0.0)
            require(minimumMedianSeparationMillimeters >= 0.0)
            require(minimumRuns in 1..32)
            require(maximumRunToRunSpreadMillimeters > 0.0)
            require(madMultiplierForSeparation > 0.0)
            require(maximumOutlierFraction in 0.0..0.5)
        }
    }

    data class Sample(
        val sampleId: Long,
        val captureRunId: String = "default",
        val timestampNanos: Long,
        val systemErrorMillimeters: Double,
        val referenceErrorMillimeters: Double,
        val systemUncertaintyMillimeters: Double = 0.0,
        val referenceUncertaintyMillimeters: Double = 0.0,
        val matched: Boolean = true,
        val confidence: Double = 1.0,
    ) {
        init {
            require(captureRunId.isNotBlank())
            require(captureRunId.length <= 96)
            require(timestampNanos >= 0L)
            require(systemErrorMillimeters.isFinite() && systemErrorMillimeters >= 0.0)
            require(referenceErrorMillimeters.isFinite() && referenceErrorMillimeters >= 0.0)
            require(systemUncertaintyMillimeters.isFinite() && systemUncertaintyMillimeters >= 0.0)
            require(referenceUncertaintyMillimeters.isFinite() && referenceUncertaintyMillimeters >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    enum class Status {
        INSUFFICIENT_EVIDENCE,
        INCOMPARABLE,
        QUALIFIED_COMPARISON,
        REFERENCE_UNCERTAINTY_TOO_HIGH,
        OUTLIER_DOMINATED,
    }

    data class Statistics(
        val count: Int,
        val p50Millimeters: Double,
        val p95Millimeters: Double,
        val p99Millimeters: Double,
        val meanMillimeters: Double,
        val rmsMillimeters: Double,
        val maximumMillimeters: Double,
    )

    data class PairedStatistics(
        val count: Int,
        val medianSignedDeltaMillimeters: Double,
        val meanSignedDeltaMillimeters: Double,
        val p95AbsoluteDeltaMillimeters: Double,
        val madMillimeters: Double,
        val maximumAbsoluteDeltaMillimeters: Double,
        val confidenceWeightedMedianDeltaMillimeters: Double,
    )

    data class RepeatabilityStatistics(
        val runCount: Int,
        val runToRunSpreadMillimeters: Double,
        val runMedianP95Millimeters: Double,
        val score: Double,
    )

    data class Result(
        val status: Status,
        val matchedSampleCount: Int,
        val totalInputSampleCount: Int,
        val matchedFraction: Double,
        val system: Statistics,
        val reference: Statistics,
        val paired: PairedStatistics,
        val repeatability: RepeatabilityStatistics,
        val meanReferenceUncertaintyMillimeters: Double,
        val maximumReferenceUncertaintyMillimeters: Double,
        val referenceUncertaintyPass: Boolean,
        val tailSeparationMillimeters: Double,
        val medianSeparationMillimeters: Double,
        val separationScore: Double,
        val outlierFraction: Double,
        val comparable: Boolean,
        val warnings: List<String>,
        val blockers: List<String>,
    ) {
        val qualified: Boolean
            get() =
                status == Status.QUALIFIED_COMPARISON &&
                    comparable &&
                    referenceUncertaintyPass &&
                    separationScore >= 0.75 &&
                    repeatability.runCount >= config.minimumRuns &&
                    repeatability.runToRunSpreadMillimeters <=
                        config.maximumRunToRunSpreadMillimeters &&
                    repeatability.score >= 0.70
    }

    fun evaluate(samples: List<Sample>): Result {
        val total = samples.size

        val matched = samples
            .asSequence()
            .filter { it.matched }
            .filter { it.confidence >= 0.20 }
            .distinctBy { it.sampleId }
            .toList()

        val matchedFraction =
            if (total <= 0) {
                0.0
            } else {
                matched.size.toDouble() /
                    total.toDouble()
            }

        if (matched.isEmpty()) {
            return emptyResult(
                total = total,
                warning = "No matched reference samples survived evidence gating.",
            )
        }

        val systemValues =
            matched.map { it.systemErrorMillimeters }
        val referenceValues =
            matched.map { it.referenceErrorMillimeters }

        val systemStats = statistics(systemValues)
        val referenceStats = statistics(referenceValues)

        val deltas =
            matched.map {
                it.systemErrorMillimeters -
                    it.referenceErrorMillimeters
            }

        val paired = pairedStatistics(
            samples = matched,
            deltas = deltas,
        )

        val repeatability =
            repeatabilityStatistics(
                samples = matched,
            )

        val referenceUncertainty =
            matched.map {
                it.referenceUncertaintyMillimeters
            }

        val meanReferenceUncertainty =
            referenceUncertainty.average()

        val maximumReferenceUncertainty =
            referenceUncertainty.maxOrNull() ?: Double.POSITIVE_INFINITY

        val referenceUncertaintyPass =
            maximumReferenceUncertainty <=
                config.maximumReferenceUncertaintyMillimeters

        val tailSeparation =
            abs(
                systemStats.p95Millimeters -
                    referenceStats.p95Millimeters,
            )

        val medianSeparation =
            abs(
                systemStats.p50Millimeters -
                    referenceStats.p50Millimeters,
            )

        val separationScore =
            calculateSeparationScore(
                paired = paired,
                tailSeparationMillimeters = tailSeparation,
                medianSeparationMillimeters = medianSeparation,
                referenceUncertaintyMillimeters =
                    meanReferenceUncertainty,
            )

        val outlierFraction =
            matched.count {
                val combined =
                    it.systemErrorMillimeters +
                        it.referenceErrorMillimeters
                combined >
                    (systemStats.p95Millimeters +
                        referenceStats.p95Millimeters)
            }.toDouble() /
                matched.size.toDouble().coerceAtLeast(1.0)

        val warnings = buildList {
            if (matched.size < config.minimumSamples) {
                add("matched-sample-count-below-precision-benchmark-minimum")
            }
            if (matchedFraction < config.minimumMatchedFraction) {
                add("matched-fraction-below-precision-benchmark-minimum")
            }
            if (!referenceUncertaintyPass) {
                add("reference-uncertainty-too-high-for-qualified-comparison")
            }
            if (tailSeparation < config.minimumTailSeparationMillimeters) {
                add("tail-separation-is-small")
            }
            if (medianSeparation < config.minimumMedianSeparationMillimeters) {
                add("median-separation-is-small")
            }
            if (outlierFraction > config.maximumOutlierFraction) {
                add("comparison-tail-outlier-fraction-is-elevated")
            }
            if (repeatability.runCount < config.minimumRuns) {
                add("repeatability-run-count-below-contract")
            }
            if (
                repeatability.runToRunSpreadMillimeters >
                    config.maximumRunToRunSpreadMillimeters
            ) {
                add("repeatability-run-to-run-spread-high")
            }
            if (paired.p95AbsoluteDeltaMillimeters >
                max(
                    config.minimumTailSeparationMillimeters * 8.0,
                    1.0,
                )
            ) {
                add("paired-error-tail-is-wide")
            }
        }

        val blockers = buildList {
            if (matched.size < config.minimumSamples) {
                add("insufficient-matched-samples")
            }
            if (matchedFraction < config.minimumMatchedFraction) {
                add("insufficient-matched-fraction")
            }
            if (config.referenceRequired && !referenceUncertaintyPass) {
                add("reference-uncertainty-exceeds-contract")
            }
            if (outlierFraction > config.maximumOutlierFraction) {
                add("outlier-fraction-exceeds-contract")
            }
            if (repeatability.runCount < config.minimumRuns) {
                add("repeatability-run-count-below-contract")
            }
            if (
                repeatability.runToRunSpreadMillimeters >
                    config.maximumRunToRunSpreadMillimeters
            ) {
                add("repeatability-exceeds-contract")
            }
        }

        val status =
            when {
                matched.size < config.minimumSamples ||
                    matchedFraction < config.minimumMatchedFraction ->
                    Status.INSUFFICIENT_EVIDENCE

                !referenceUncertaintyPass ->
                    Status.REFERENCE_UNCERTAINTY_TOO_HIGH

                outlierFraction > config.maximumOutlierFraction ->
                    Status.OUTLIER_DOMINATED

                repeatability.runCount < config.minimumRuns ->
                    Status.INSUFFICIENT_EVIDENCE

                repeatability.runToRunSpreadMillimeters >
                    config.maximumRunToRunSpreadMillimeters ->
                    Status.INCOMPARABLE

                tailSeparation <
                    config.minimumTailSeparationMillimeters &&
                    medianSeparation <
                    config.minimumMedianSeparationMillimeters ->
                    Status.INCOMPARABLE

                else ->
                    Status.QUALIFIED_COMPARISON
            }

        return Result(
            status = status,
            matchedSampleCount = matched.size,
            totalInputSampleCount = total,
            matchedFraction = matchedFraction.coerceIn(0.0, 1.0),
            system = systemStats,
            reference = referenceStats,
            paired = paired,
            repeatability = repeatability,
            meanReferenceUncertaintyMillimeters =
                meanReferenceUncertainty,
            maximumReferenceUncertaintyMillimeters =
                maximumReferenceUncertainty,
            referenceUncertaintyPass =
                referenceUncertaintyPass,
            tailSeparationMillimeters = tailSeparation,
            medianSeparationMillimeters = medianSeparation,
            separationScore = separationScore,
            outlierFraction = outlierFraction,
            comparable = status != Status.INCOMPARABLE &&
                blockers.isEmpty(),
            warnings = warnings.distinct(),
            blockers = blockers.distinct(),
        )
    }

    private fun statistics(
        values: List<Double>,
    ): Statistics {
        if (values.isEmpty()) {
            return Statistics(
                count = 0,
                p50Millimeters = Double.POSITIVE_INFINITY,
                p95Millimeters = Double.POSITIVE_INFINITY,
                p99Millimeters = Double.POSITIVE_INFINITY,
                meanMillimeters = Double.POSITIVE_INFINITY,
                rmsMillimeters = Double.POSITIVE_INFINITY,
                maximumMillimeters = Double.POSITIVE_INFINITY,
            )
        }

        val sorted = values.sorted()
        val mean =
            sorted.average()
        val rms =
            sqrt(
                sorted.map {
                    it * it
                }.average(),
            )

        return Statistics(
            count = sorted.size,
            p50Millimeters =
                percentile(
                    sorted,
                    0.50,
                ),
            p95Millimeters =
                percentile(
                    sorted,
                    0.95,
                ),
            p99Millimeters =
                percentile(
                    sorted,
                    0.99,
                ),
            meanMillimeters = mean,
            rmsMillimeters = rms,
            maximumMillimeters = sorted.last(),
        )
    }

    private fun repeatabilityStatistics(
        samples: List<Sample>,
    ): RepeatabilityStatistics {
        val runs =
            samples
                .groupBy { it.captureRunId }
                .toSortedMap()
                .values

        if (runs.isEmpty()) {
            return RepeatabilityStatistics(
                runCount = 0,
                runToRunSpreadMillimeters = Double.POSITIVE_INFINITY,
                runMedianP95Millimeters = Double.POSITIVE_INFINITY,
                score = 0.0,
            )
        }

        val runP95s =
            runs.map { run ->
                percentile(
                    run.map { it.systemErrorMillimeters }.sorted(),
                    0.95,
                )
            }

        val spread =
            (runP95s.maxOrNull() ?: Double.POSITIVE_INFINITY) -
                (runP95s.minOrNull() ?: 0.0)

        val runCountScore =
            (
                runs.size.toDouble() /
                    config.minimumRuns.coerceAtLeast(1).toDouble()
            ).coerceAtMost(1.0)

        val spreadScore =
            if (!spread.isFinite()) {
                0.0
            } else {
                kotlin.math.exp(
                    -spread /
                        config.maximumRunToRunSpreadMillimeters
                            .coerceAtLeast(1e-6),
                ).coerceIn(0.0, 1.0)
            }

        val score =
            (
                runCountScore * 0.35 +
                    spreadScore * 0.65
            ).coerceIn(0.0, 1.0)

        return RepeatabilityStatistics(
            runCount = runs.size,
            runToRunSpreadMillimeters = spread,
            runMedianP95Millimeters =
                runP95s.average(),
            score = score,
        )
    }

    private fun pairedStatistics(
        samples: List<Sample>,
        deltas: List<Double>,
    ): PairedStatistics {
        val sorted = deltas.sorted()
        val absoluteSorted =
            deltas.map(::abs).sorted()

        val medianDelta =
            percentile(sorted, 0.50)

        val mad =
            medianAbsoluteDeviation(
                sorted,
                medianDelta,
            )

        val weightedPairs =
            samples.sortedWith(
                compareByDescending<Sample> {
                    it.confidence
                }.thenBy {
                    it.timestampNanos
                },
            )

        var cumulativeWeight = 0.0
        val totalWeight =
            weightedPairs.sumOf {
                it.confidence.coerceAtLeast(0.05)
            }

        var weightedMedian = 0.0
        for (sample in weightedPairs) {
            cumulativeWeight +=
                sample.confidence.coerceAtLeast(0.05)
            if (cumulativeWeight >= totalWeight * 0.50) {
                weightedMedian =
                    sample.systemErrorMillimeters -
                        sample.referenceErrorMillimeters
                break
            }
        }

        return PairedStatistics(
            count = deltas.size,
            medianSignedDeltaMillimeters = medianDelta,
            meanSignedDeltaMillimeters =
                deltas.average(),
            p95AbsoluteDeltaMillimeters =
                percentile(
                    absoluteSorted,
                    0.95,
                ),
            madMillimeters = mad,
            maximumAbsoluteDeltaMillimeters =
                absoluteSorted.maxOrNull()
                    ?: Double.POSITIVE_INFINITY,
            confidenceWeightedMedianDeltaMillimeters =
                weightedMedian,
        )
    }

    private fun medianAbsoluteDeviation(
        sortedDeltas: List<Double>,
        medianDelta: Double,
    ): Double {
        if (sortedDeltas.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }
        val deviations =
            sortedDeltas.map {
                abs(it - medianDelta)
            }.sorted()
        return percentile(
            deviations,
            0.50,
        )
    }

    private fun calculateSeparationScore(
        paired: PairedStatistics,
        tailSeparationMillimeters: Double,
        medianSeparationMillimeters: Double,
        referenceUncertaintyMillimeters: Double,
    ): Double {
        val uncertaintyFloor =
            max(
                referenceUncertaintyMillimeters,
                0.05,
            )

        val medianSignal =
            (
                medianSeparationMillimeters /
                    (
                        medianSeparationMillimeters +
                            uncertaintyFloor
                    )
                ).coerceIn(0.0, 1.0)

        val tailSignal =
            (
                tailSeparationMillimeters /
                    (
                        tailSeparationMillimeters +
                            uncertaintyFloor
                    )
                ).coerceIn(0.0, 1.0)

        val robustSignal =
            (
                abs(paired.medianSignedDeltaMillimeters) /
                    (
                        abs(paired.medianSignedDeltaMillimeters) +
                            max(
                                paired.madMillimeters,
                                0.05,
                            )
                    )
                ).coerceIn(0.0, 1.0)

        val tailConsistency =
            (
                1.0 -
                    paired.p95AbsoluteDeltaMillimeters /
                        max(
                            tailSeparationMillimeters * 4.0,
                            2.0,
                        )
                ).coerceIn(0.0, 1.0)

        return (
            medianSignal * 0.25 +
                tailSignal * 0.30 +
                robustSignal * 0.20 +
                tailConsistency * 0.25
            ).coerceIn(0.0, 1.0)
    }

    private fun percentile(
        sortedValues: List<Double>,
        quantile: Double,
    ): Double {
        if (sortedValues.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }
        if (sortedValues.size == 1) {
            return sortedValues.first()
        }

        val position =
            quantile.coerceIn(0.0, 1.0) *
                (sortedValues.size - 1)

        val lower = position.toInt()
        val upper =
            minOf(
                sortedValues.lastIndex,
                lower + 1,
            )
        val fraction =
            position - lower

        return sortedValues[lower] +
            (
                sortedValues[upper] -
                    sortedValues[lower]
                ) *
                fraction
    }

    private fun emptyResult(
        total: Int,
        warning: String,
    ): Result =
        Result(
            status = Status.INSUFFICIENT_EVIDENCE,
            matchedSampleCount = 0,
            totalInputSampleCount = total,
            matchedFraction = 0.0,
            system = statistics(emptyList()),
            reference = statistics(emptyList()),
            paired = PairedStatistics(
                count = 0,
                medianSignedDeltaMillimeters = Double.POSITIVE_INFINITY,
                meanSignedDeltaMillimeters = Double.POSITIVE_INFINITY,
                p95AbsoluteDeltaMillimeters = Double.POSITIVE_INFINITY,
                madMillimeters = Double.POSITIVE_INFINITY,
                maximumAbsoluteDeltaMillimeters = Double.POSITIVE_INFINITY,
                confidenceWeightedMedianDeltaMillimeters =
                    Double.POSITIVE_INFINITY,
            ),
            repeatability = RepeatabilityStatistics(
                runCount = 0,
                runToRunSpreadMillimeters = Double.POSITIVE_INFINITY,
                runMedianP95Millimeters = Double.POSITIVE_INFINITY,
                score = 0.0,
            ),
            meanReferenceUncertaintyMillimeters =
                Double.POSITIVE_INFINITY,
            maximumReferenceUncertaintyMillimeters =
                Double.POSITIVE_INFINITY,
            referenceUncertaintyPass = false,
            tailSeparationMillimeters = 0.0,
            medianSeparationMillimeters = 0.0,
            separationScore = 0.0,
            outlierFraction = 0.0,
            comparable = false,
            warnings = listOf(warning),
            blockers = listOf("no-matched-evidence"),
        )
}
