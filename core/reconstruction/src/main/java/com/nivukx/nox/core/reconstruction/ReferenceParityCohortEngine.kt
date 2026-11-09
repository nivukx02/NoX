package com.nivukx.nox.core.reconstruction

import kotlin.math.sqrt

/**
 * Cohort-level paired-reference evaluator for motion-capture metrology.
 *
 * A single successful take is not enough to qualify a precision target. This engine requires
 * repeated matched-condition NoX/reference pairs, rejects mismatched condition fingerprints,
 * separates accepted from rejected runs, and computes conservative cohort-level P95/P99 metrics.
 */
class ReferenceParityCohortEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumPairs: Int = 5,
        val maximumPairs: Int = 256,
        val maximumCohortP95Millimeters: Double = 2.0,
        val maximumCohortP99Millimeters: Double = 4.0,
        val maximumCohortReferenceUncertaintyMillimeters: Double = 0.75,
        val minimumConfidence: Double = 0.75,
        val minimumAcceptedPairRatio: Double = 0.80,
        val minimumAcceptedReferencePairRatio: Double = 0.80,
    ) {
        init {
            require(minimumPairs > 0)
            require(maximumPairs >= minimumPairs)
            require(maximumCohortP95Millimeters > 0.0)
            require(maximumCohortP99Millimeters >= maximumCohortP95Millimeters)
            require(maximumCohortReferenceUncertaintyMillimeters > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(minimumAcceptedPairRatio in 0.0..1.0)
            require(minimumAcceptedReferencePairRatio in 0.0..1.0)
        }
    }

    data class Input(
        val matchedConditionFingerprint: String,
        val referenceConditionFingerprint: String = matchedConditionFingerprint,
        val noxReports: List<CapturePrecisionBenchmarkEngine.Report>,
        val referenceReports: List<CapturePrecisionBenchmarkEngine.Report>,
    ) {
        init {
            require(matchedConditionFingerprint.isNotBlank())
            require(referenceConditionFingerprint.isNotBlank())
        }
    }

    data class PairDelta(
        val captureId: String,
        val deltaP50Millimeters: Double,
        val deltaP95Millimeters: Double,
        val deltaP99Millimeters: Double,
        val deltaJitterMillimeters: Double,
        val deltaDropoutRatio: Double,
        val deltaLagMilliseconds: Double,
        val noxReferenceUncertaintyMillimeters: Double?,
    )

    data class Result(
        val matchedPairs: Int,
        val acceptedNoxPairs: Int,
        val rejectedNoxPairs: Int,
        val aggregateNoxP95Millimeters: Double,
        val aggregateNoxP99Millimeters: Double,
        val aggregateReferenceP95Millimeters: Double,
        val aggregateReferenceP99Millimeters: Double,
        val aggregateDeltaP95Millimeters: Double,
        val aggregateDeltaP99Millimeters: Double,
        val maximumReferenceUncertaintyMillimeters: Double,
        val acceptedPairRatio: Double,
        val confidence: Double,
        val evidenceQualified: Boolean,
        val pairDeltas: List<PairDelta>,
        val blockingReasons: List<String>,
        val warnings: List<String>,
    )

    fun evaluate(
        noxConditions: CaptureConditionFingerprintEngine.Result,
        referenceConditions: CaptureConditionFingerprintEngine.Result,
        noxReports: List<CapturePrecisionBenchmarkEngine.Report>,
        referenceReports: List<CapturePrecisionBenchmarkEngine.Report>,
    ): Result =
        evaluate(
            Input(
                matchedConditionFingerprint =
                    noxConditions.fingerprint,
                referenceConditionFingerprint =
                    referenceConditions.fingerprint,
                noxReports = noxReports,
                referenceReports = referenceReports,
            ),
        )

    fun evaluate(input: Input): Result {
        if (
            input.matchedConditionFingerprint !=
                input.referenceConditionFingerprint
        ) {
            return blocked(
                reason =
                    "condition-fingerprint-mismatch",
            )
        }

        val noxById =
            input.noxReports
                .associateBy { it.captureId }

        val referenceById =
            input.referenceReports
                .associateBy { it.captureId }

        val commonIds =
            noxById.keys
                .intersect(referenceById.keys)
                .toList()
                .sorted()
                .take(config.maximumPairs)

        if (commonIds.isEmpty()) {
            return blocked(
                reason = "no-matched-reference-pairs",
            )
        }

        val pairDeltas = ArrayList<PairDelta>(commonIds.size)
        var acceptedNox = 0
        var rejectedNox = 0
        var acceptedReference = 0
        var rejectedReference = 0
        var jointlyAcceptedPairs = 0

        val noxP95 = ArrayList<Double>()
        val noxP99 = ArrayList<Double>()
        val refP95 = ArrayList<Double>()
        val refP99 = ArrayList<Double>()
        val referenceUncertainties = ArrayList<Double>()
        val confidenceSamples = ArrayList<Double>()

        for (captureId in commonIds) {
            val nox = noxById.getValue(captureId)
            val reference =
                referenceById.getValue(captureId)

            if (nox.accepted) {
                acceptedNox++
            } else {
                rejectedNox++
            }

            if (reference.accepted) {
                acceptedReference++
            } else {
                rejectedReference++
            }

            if (nox.accepted && reference.accepted) {
                noxP95 +=
                    nox.aggregate.p95ErrorMm
                noxP99 +=
                    nox.aggregate.p99ErrorMm
            }
            if (nox.accepted && reference.accepted) {
                refP95 +=
                    reference.aggregate.p95ErrorMm
                refP99 +=
                    reference.aggregate.p99ErrorMm
            }

            reference.referenceUncertaintyMillimeters?.let {
                referenceUncertainties += it
            }

            if (nox.accepted && reference.accepted) {
                jointlyAcceptedPairs += 1
                confidenceSamples +=
                    minOf(
                        nox.confidence,
                        reference.confidence,
                    )
            }

            pairDeltas +=
                PairDelta(
                    captureId = captureId,
                    deltaP50Millimeters =
                        nox.aggregate.p50ErrorMm -
                            reference.aggregate.p50ErrorMm,
                    deltaP95Millimeters =
                        nox.aggregate.p95ErrorMm -
                            reference.aggregate.p95ErrorMm,
                    deltaP99Millimeters =
                        nox.aggregate.p99ErrorMm -
                            reference.aggregate.p99ErrorMm,
                    deltaJitterMillimeters =
                        nox.aggregate.jitterRmsMm -
                            reference.aggregate.jitterRmsMm,
                    deltaDropoutRatio =
                        nox.aggregate.dropoutRatio -
                            reference.aggregate.dropoutRatio,
                    deltaLagMilliseconds =
                        nox.aggregate.temporalLagMs -
                            reference.aggregate.temporalLagMs,
                    noxReferenceUncertaintyMillimeters =
                        reference.referenceUncertaintyMillimeters,
                )
        }

        if (noxP95.isEmpty() || refP95.isEmpty()) {
            return Result(
                matchedPairs = commonIds.size,
                acceptedNoxPairs = acceptedNox,
                rejectedNoxPairs = rejectedNox,
                aggregateNoxP95Millimeters = Double.POSITIVE_INFINITY,
                aggregateNoxP99Millimeters = Double.POSITIVE_INFINITY,
                aggregateReferenceP95Millimeters = Double.POSITIVE_INFINITY,
                aggregateReferenceP99Millimeters = Double.POSITIVE_INFINITY,
                aggregateDeltaP95Millimeters = Double.POSITIVE_INFINITY,
                aggregateDeltaP99Millimeters = Double.POSITIVE_INFINITY,
                maximumReferenceUncertaintyMillimeters =
                    referenceUncertainties.maxOrNull()
                        ?: Double.POSITIVE_INFINITY,
                acceptedPairRatio = jointlyAcceptedPairs.toDouble() /
                    commonIds.size.toDouble(),
                confidence = 0.0,
                evidenceQualified = false,
                pairDeltas = pairDeltas,
                blockingReasons =
                    listOf("accepted-reference-pairs-required"),
                warnings = emptyList(),
            )
        }

        val aggregateNoxP95 =
            conservativeQuantile(
                noxP95,
                0.95,
            )
        val aggregateNoxP99 =
            conservativeQuantile(
                noxP99,
                0.99,
            )
        val aggregateReferenceP95 =
            conservativeQuantile(
                refP95,
                0.95,
            )
        val aggregateReferenceP99 =
            conservativeQuantile(
                refP99,
                0.99,
            )
        val deltaP95 =
            aggregateNoxP95 -
                aggregateReferenceP95
        val deltaP99 =
            aggregateNoxP99 -
                aggregateReferenceP99
        val maxReferenceUncertainty =
            referenceUncertainties.maxOrNull()
                ?: 0.0

        val acceptedPairRatio =
            jointlyAcceptedPairs.toDouble() /
                commonIds.size
                    .coerceAtLeast(1)
                    .toDouble()

        val acceptedReferencePairRatio =
            acceptedReference.toDouble() /
                commonIds.size
                    .coerceAtLeast(1)
                    .toDouble()

        val referenceTypeMismatch =
            commonIds.any { id ->
                noxById.getValue(id).referenceType !=
                    referenceById.getValue(id).referenceType
            }

        if (referenceTypeMismatch) {
            return Result(
                matchedPairs = commonIds.size,
                acceptedNoxPairs = acceptedNox,
                rejectedNoxPairs = rejectedNox,
                aggregateNoxP95Millimeters = Double.POSITIVE_INFINITY,
                aggregateNoxP99Millimeters = Double.POSITIVE_INFINITY,
                aggregateReferenceP95Millimeters = Double.POSITIVE_INFINITY,
                aggregateReferenceP99Millimeters = Double.POSITIVE_INFINITY,
                aggregateDeltaP95Millimeters = Double.POSITIVE_INFINITY,
                aggregateDeltaP99Millimeters = Double.POSITIVE_INFINITY,
                maximumReferenceUncertaintyMillimeters =
                    maxReferenceUncertainty,
                acceptedPairRatio = acceptedPairRatio,
                confidence = 0.0,
                evidenceQualified = false,
                pairDeltas = pairDeltas,
                blockingReasons = listOf("reference-type-mismatch"),
                warnings = emptyList(),
            )
        }

        val confidenceMean =
            confidenceSamples
                .takeIf { it.isNotEmpty() }
                ?.average()
                ?: 0.0

        val pairCountConfidence =
            (
                commonIds.size.toDouble() /
                    config.minimumPairs
                        .toDouble()
                ).coerceIn(0.0, 1.0)

        val consistencyConfidence =
            (
                1.0 -
                    robustSpread(
                        pairDeltas.map {
                            it.deltaP95Millimeters
                        },
                    )
                ).coerceIn(0.0, 1.0)

        val confidence =
            (
                confidenceMean * 0.38 +
                    pairCountConfidence * 0.24 +
                    acceptedPairRatio * 0.14 +
                    acceptedReferencePairRatio * 0.06 +
                    consistencyConfidence * 0.18
                ).coerceIn(0.0, 1.0)

        val blockers = buildList {
            if (commonIds.size < config.minimumPairs) {
                add("minimum-pairs-not-reached")
            }
            if (acceptedNox < config.minimumPairs) {
                add("accepted-pair-count-below-minimum")
            }
            if (acceptedReference < config.minimumPairs) {
                add("accepted-reference-pair-count-below-minimum")
            }
            if (
                acceptedReferencePairRatio <
                    config.minimumAcceptedReferencePairRatio
            ) {
                add("accepted-reference-pair-ratio-below-preferred-minimum")
            }

            if (
                aggregateNoxP95 >
                    config.maximumCohortP95Millimeters
            ) {
                add("cohort-P95-exceeds-2mm-target")
            }
            if (
                aggregateNoxP99 >
                    config.maximumCohortP99Millimeters
            ) {
                add("cohort-P99-exceeds-target")
            }
            if (
                maxReferenceUncertainty >
                    config.maximumCohortReferenceUncertaintyMillimeters
            ) {
                add("reference-uncertainty-too-high")
            }
            if (confidence < config.minimumConfidence) {
                add("cohort-confidence-too-low")
            }
        }

        val warnings = buildList {
            if (acceptedPairRatio < config.minimumAcceptedPairRatio) {
                add("accepted-pair-ratio-below-preferred-minimum")
            }
            if (acceptedReferencePairRatio < config.minimumAcceptedReferencePairRatio) {
                add("accepted-reference-pair-ratio-below-preferred-minimum")
            }
            if (rejectedNox > 0) {
                add("rejected-nox-runs-excluded-from-qualified-cohort")
            }
            if (rejectedReference > 0) {
                add("rejected-reference-runs-excluded-from-qualified-cohort")
            }
            if (consistencyConfidence < 0.70) {
                add("paired-error-variance-elevated")
            }
            if (deltaP95 > 0.0) {
                add("nox-p95-above-reference-p95")
            } else {
                add("nox-p95-below-reference-p95")
            }
            if (deltaP99 > 0.0) {
                add("nox-p99-above-reference-p99")
            }
        }

        return Result(
            matchedPairs = commonIds.size,
            acceptedNoxPairs = acceptedNox,
            rejectedNoxPairs = rejectedNox,
            aggregateNoxP95Millimeters = aggregateNoxP95,
            aggregateNoxP99Millimeters = aggregateNoxP99,
            aggregateReferenceP95Millimeters =
                aggregateReferenceP95,
            aggregateReferenceP99Millimeters =
                aggregateReferenceP99,
            aggregateDeltaP95Millimeters = deltaP95,
            aggregateDeltaP99Millimeters = deltaP99,
            maximumReferenceUncertaintyMillimeters =
                maxReferenceUncertainty,
            acceptedPairRatio = acceptedPairRatio,
            confidence = confidence,
            evidenceQualified = blockers.isEmpty(),
            pairDeltas = pairDeltas,
            blockingReasons = blockers,
            warnings = warnings,
        )
    }

    private fun blocked(
        reason: String,
    ): Result =
        Result(
            matchedPairs = 0,
            acceptedNoxPairs = 0,
            rejectedNoxPairs = 0,
            aggregateNoxP95Millimeters = Double.POSITIVE_INFINITY,
            aggregateNoxP99Millimeters = Double.POSITIVE_INFINITY,
            aggregateReferenceP95Millimeters =
                Double.POSITIVE_INFINITY,
            aggregateReferenceP99Millimeters =
                Double.POSITIVE_INFINITY,
            aggregateDeltaP95Millimeters =
                Double.POSITIVE_INFINITY,
            aggregateDeltaP99Millimeters =
                Double.POSITIVE_INFINITY,
            maximumReferenceUncertaintyMillimeters =
                Double.POSITIVE_INFINITY,
            acceptedPairRatio = 0.0,
            confidence = 0.0,
            evidenceQualified = false,
            pairDeltas = emptyList(),
            blockingReasons = listOf(reason),
            warnings = emptyList(),
        )

    private fun conservativeQuantile(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val sorted = values
            .filter(Double::isFinite)
            .sorted()
        if (sorted.isEmpty()) return Double.POSITIVE_INFINITY
        val rawIndex =
            probability *
                sorted.size.toDouble()
        val index =
            kotlin.math.ceil(rawIndex)
                .toInt()
                .minus(1)
                .coerceIn(
                    0,
                    sorted.lastIndex,
                )
        return sorted[index]
    }

    private fun robustSpread(values: List<Double>): Double {
        if (values.size < 2) return 0.0
        val mean =
            values.average()
        val variance =
            values.map {
                val delta = it - mean
                delta * delta
            }.average()
        val standardDeviation =
            sqrt(variance)
        val scale =
            values.map { kotlin.math.abs(it) }
                .average()
                .coerceAtLeast(1e-6)
        return (
            standardDeviation / scale
            ).coerceIn(0.0, 1.0)
    }
}
