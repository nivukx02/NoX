package com.nivukx.nox.core.reconstruction

/**
 * Infrared/optical reference interpretation layer.
 *
 * Reuses NoX's matched-condition cohort benchmark and adds an explicit classification for an
 * infrared/optical reference. No universal winner is assumed.
 */
class InfraredReferenceQualificationEngine(
    private val config: Config = Config(),
    private val cohort: ReferenceParityCohortEngine = ReferenceParityCohortEngine(),
) {
    enum class Outcome {
        INCONCLUSIVE,
        STATISTICAL_PARITY,
        NOX_LOWER_ERROR,
        REFERENCE_LOWER_ERROR,
    }

    data class Config(
        val p95TieFraction: Double = 0.05,
        val p99TieFraction: Double = 0.05,
        val minimumP95TieMillimeters: Double = 0.05,
        val minimumP99TieMillimeters: Double = 0.10,
        val minimumConfidence: Double = 0.80,
        val minimumMatchedPairs: Int = 20,
        val requireOpticalReferenceType: Boolean = true,
    ) {
        init {
            require(p95TieFraction in 0.0..0.5)
            require(p99TieFraction in 0.0..0.5)
            require(minimumP95TieMillimeters >= 0.0)
            require(minimumP99TieMillimeters >= 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(minimumMatchedPairs >= 1)
        }
    }

    data class Comparison(
        val outcome: Outcome,
        val qualified: Boolean,
        val matchedPairs: Int,
        val acceptedPairRatio: Double,
        val confidence: Double,
        val noxP95Millimeters: Double,
        val referenceP95Millimeters: Double,
        val noxP99Millimeters: Double,
        val referenceP99Millimeters: Double,
        val p95DeltaMillimeters: Double,
        val p99DeltaMillimeters: Double,
        val p95TieMarginMillimeters: Double,
        val p99TieMarginMillimeters: Double,
        val referenceUncertaintyMillimeters: Double,
        val blockers: List<String>,
        val warnings: List<String>,
    ) {
        val noxPhysicallyFavored: Boolean
            get() = qualified && outcome == Outcome.NOX_LOWER_ERROR

        val referencePhysicallyFavored: Boolean
            get() = qualified && outcome == Outcome.REFERENCE_LOWER_ERROR

        val statisticalParity: Boolean
            get() = qualified && outcome == Outcome.STATISTICAL_PARITY
    }

    fun evaluate(
        noxConditions: CaptureConditionFingerprintEngine.Result,
        infraredConditions: CaptureConditionFingerprintEngine.Result,
        noxReports: List<CapturePrecisionBenchmarkEngine.Report>,
        infraredReports: List<CapturePrecisionBenchmarkEngine.Report>,
    ): Comparison {
        if (
            config.requireOpticalReferenceType &&
            infraredReports.any {
                it.referenceType !=
                    CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE &&
                    it.referenceType !=
                    CapturePrecisionBenchmarkEngine.ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE
            }
        ) {
            return blocked("infrared-optical-reference-type-required")
        }
        return evaluate(
            cohort.evaluate(
                noxConditions = noxConditions,
                referenceConditions = infraredConditions,
                noxReports = noxReports,
                referenceReports = infraredReports,
            ),
        )
    }

    fun evaluate(cohortResult: ReferenceParityCohortEngine.Result): Comparison {
        val p95Reference = cohortResult.aggregateReferenceP95Millimeters
        val p99Reference = cohortResult.aggregateReferenceP99Millimeters
        val p95Margin = if (p95Reference.isFinite()) {
            maxOf(config.minimumP95TieMillimeters, p95Reference * config.p95TieFraction)
        } else {
            Double.POSITIVE_INFINITY
        }
        val p99Margin = if (p99Reference.isFinite()) {
            maxOf(config.minimumP99TieMillimeters, p99Reference * config.p99TieFraction)
        } else {
            Double.POSITIVE_INFINITY
        }
        val finiteEvidence =
            cohortResult.aggregateNoxP95Millimeters.isFinite() &&
                cohortResult.aggregateReferenceP95Millimeters.isFinite() &&
                cohortResult.aggregateNoxP99Millimeters.isFinite() &&
                cohortResult.aggregateReferenceP99Millimeters.isFinite()

        val qualified =
            cohortResult.evidenceQualified &&
                cohortResult.matchedPairs >= config.minimumMatchedPairs &&
                cohortResult.confidence >= config.minimumConfidence &&
                cohortResult.maximumReferenceUncertaintyMillimeters.isFinite() &&
                finiteEvidence

        val outcome = when {
            !qualified -> Outcome.INCONCLUSIVE
            kotlin.math.abs(cohortResult.aggregateDeltaP95Millimeters) <= p95Margin &&
                kotlin.math.abs(cohortResult.aggregateDeltaP99Millimeters) <= p99Margin ->
                Outcome.STATISTICAL_PARITY
            cohortResult.aggregateDeltaP95Millimeters < -p95Margin &&
                cohortResult.aggregateDeltaP99Millimeters < -p99Margin ->
                Outcome.NOX_LOWER_ERROR
            cohortResult.aggregateDeltaP95Millimeters > p95Margin &&
                cohortResult.aggregateDeltaP99Millimeters > p99Margin ->
                Outcome.REFERENCE_LOWER_ERROR
            else -> Outcome.INCONCLUSIVE
        }

        return Comparison(
            outcome = outcome,
            qualified = qualified,
            matchedPairs = cohortResult.matchedPairs,
            acceptedPairRatio = cohortResult.acceptedPairRatio,
            confidence = cohortResult.confidence,
            noxP95Millimeters = cohortResult.aggregateNoxP95Millimeters,
            referenceP95Millimeters = cohortResult.aggregateReferenceP95Millimeters,
            noxP99Millimeters = cohortResult.aggregateNoxP99Millimeters,
            referenceP99Millimeters = cohortResult.aggregateReferenceP99Millimeters,
            p95DeltaMillimeters = cohortResult.aggregateDeltaP95Millimeters,
            p99DeltaMillimeters = cohortResult.aggregateDeltaP99Millimeters,
            p95TieMarginMillimeters = p95Margin,
            p99TieMarginMillimeters = p99Margin,
            referenceUncertaintyMillimeters = cohortResult.maximumReferenceUncertaintyMillimeters,
            blockers = buildList {
                addAll(cohortResult.blockingReasons)
                if (cohortResult.matchedPairs < config.minimumMatchedPairs) add("infrared-minimum-matched-pairs-not-reached")
                if (cohortResult.confidence < config.minimumConfidence) add("infrared-comparison-confidence-below-contract")
                if (!cohortResult.maximumReferenceUncertaintyMillimeters.isFinite()) add("infrared-reference-uncertainty-unreported")
                if (!finiteEvidence) add("infrared-comparison-nonfinite-evidence")
            }.distinct(),
            warnings = buildList {
                addAll(cohortResult.warnings)
                when (outcome) {
                    Outcome.NOX_LOWER_ERROR -> add("nox-lower-error-under-matched-infrared-reference")
                    Outcome.REFERENCE_LOWER_ERROR -> add("infrared-reference-lower-error")
                    Outcome.STATISTICAL_PARITY -> add("nox-and-infrared-within-statistical-tie-band")
                    Outcome.INCONCLUSIVE -> add("infrared-comparison-inconclusive")
                }
            }.distinct(),
        )
    }

    /**
     * Runs the stricter active-optical marker qualification protocol.
     *
     * This supplements the legacy cohort comparison; it does not remove or reinterpret its
     * statistical parity result.
     */
    fun evaluateActiveOpticalParity(
        input: ActiveOpticalReferenceParityEngine.Input,
    ): ActiveOpticalReferenceParityEngine.Result =
        ActiveOpticalReferenceParityEngine().evaluate(
            input.copy(
                referenceType =
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .ACTIVE_OPTICAL_MARKER_REFERENCE,
            ),
        )

    private fun blocked(reason: String) = Comparison(
        outcome = Outcome.INCONCLUSIVE,
        qualified = false,
        matchedPairs = 0,
        acceptedPairRatio = 0.0,
        confidence = 0.0,
        noxP95Millimeters = Double.POSITIVE_INFINITY,
        referenceP95Millimeters = Double.POSITIVE_INFINITY,
        noxP99Millimeters = Double.POSITIVE_INFINITY,
        referenceP99Millimeters = Double.POSITIVE_INFINITY,
        p95DeltaMillimeters = Double.POSITIVE_INFINITY,
        p99DeltaMillimeters = Double.POSITIVE_INFINITY,
        p95TieMarginMillimeters = Double.POSITIVE_INFINITY,
        p99TieMarginMillimeters = Double.POSITIVE_INFINITY,
        referenceUncertaintyMillimeters = Double.POSITIVE_INFINITY,
        blockers = listOf(reason),
        warnings = listOf("infrared-comparison-inconclusive"),
    )
}
