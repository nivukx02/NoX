package com.nivukx.nox.core.reconstruction

import kotlin.math.sqrt

/**
 * Evidence-only precision qualification for high-end motion capture.
 *
 * This class never estimates or predicts physical accuracy from software architecture alone.
 * It only decides whether an externally measured benchmark supports a requested precision tier.
 *
 * The sub-millimeter tier is deliberately strict:
 * - matched capture conditions are required;
 * - an accepted empirical benchmark is required;
 * - an external reference and its uncertainty must be documented;
 * - enough samples are required;
 * - measured P95/P99 must satisfy the target after accounting for reference uncertainty.
 *
 * A READY result means "the supplied benchmark evidence satisfies the contract", not
 * "NoX universally achieves this accuracy".
 */
class MeasuredPrecisionQualificationEngine(
    private val config: Config = Config(),
) {

    enum class Tier {
        REFERENCE_TWO_MM,
        SUB_MILLIMETER,
    }

    enum class Status {
        NOT_REQUESTED,
        BLOCKED,
        QUALIFIED,
    }

    data class Config(
        val minimumReferenceSamples: Int = 128,
        val minimumSubMillimeterSamples: Int = 256,
        val minimumIndependentReferenceSessions: Int = 1,
        val minimumReferenceUncertaintyMarginMillimeters: Double = 0.001,
        val defaultReferenceP95TargetMillimeters: Double = 2.0,
        val defaultReferenceP99TargetMillimeters: Double = 4.0,
        val defaultSubMillimeterP95TargetMillimeters: Double = 1.0,
        val defaultSubMillimeterP99TargetMillimeters: Double = 2.0,
        val requireSurfaceEvidenceForSubMillimeter: Boolean = false,
        val defaultSubMillimeterSurfaceP95TargetMillimeters: Double = 1.5,
        val defaultSubMillimeterSurfaceP99TargetMillimeters: Double = 3.0,
    ) {
        init {
            require(minimumReferenceSamples > 0)
            require(minimumSubMillimeterSamples >= minimumReferenceSamples)
            require(minimumIndependentReferenceSessions >= 1)
            require(minimumReferenceUncertaintyMarginMillimeters > 0.0)
            require(defaultReferenceP95TargetMillimeters > 0.0)
            require(defaultReferenceP99TargetMillimeters >= defaultReferenceP95TargetMillimeters)
            require(defaultSubMillimeterP95TargetMillimeters > 0.0)
            require(defaultSubMillimeterP99TargetMillimeters >= defaultSubMillimeterP95TargetMillimeters)
            require(defaultSubMillimeterSurfaceP95TargetMillimeters > 0.0)
            require(
                defaultSubMillimeterSurfaceP99TargetMillimeters >=
                    defaultSubMillimeterSurfaceP95TargetMillimeters,
            )
        }

        companion object {
            fun strictSubMillimeter(): Config =
                Config(
                    minimumIndependentReferenceSessions = 3,
                    requireSurfaceEvidenceForSubMillimeter = true,
                )
        }
    }

    data class Input(
        val requestedTier: Tier?,
        val matchedCaptureConditions: Boolean,
        val empiricalBenchmarkAccepted: Boolean,
        val sampleCount: Int,
        val independentReferenceSessionCount: Int = 1,
        val measuredP95Millimeters: Double?,
        val measuredP99Millimeters: Double?,
        val referenceUncertaintyMillimeters: Double?,
        val referenceProvenanceDocumented: Boolean,
        val targetP95Millimeters: Double? = null,
        val targetP99Millimeters: Double? = null,
        val surfaceEvidenceAvailable: Boolean = false,
        val measuredSurfaceP95Millimeters: Double? = null,
        val measuredSurfaceP99Millimeters: Double? = null,
        val targetSurfaceP95Millimeters: Double? = null,
        val targetSurfaceP99Millimeters: Double? = null,
    ) {
        init {
            require(sampleCount >= 0)
            require(independentReferenceSessionCount >= 0)
            measuredP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            measuredP99Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            referenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            targetP95Millimeters?.let {
                require(it.isFinite() && it > 0.0)
            }
            targetP99Millimeters?.let {
                require(it.isFinite() && it > 0.0)
            }
            if (
                targetP95Millimeters != null &&
                targetP99Millimeters != null
            ) {
                require(targetP99Millimeters >= targetP95Millimeters)
            }
            measuredSurfaceP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            measuredSurfaceP99Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            targetSurfaceP95Millimeters?.let {
                require(it.isFinite() && it > 0.0)
            }
            targetSurfaceP99Millimeters?.let {
                require(it.isFinite() && it > 0.0)
            }
            if (
                targetSurfaceP95Millimeters != null &&
                targetSurfaceP99Millimeters != null
            ) {
                require(
                    targetSurfaceP99Millimeters >=
                        targetSurfaceP95Millimeters,
                )
            }
        }
    }

    data class Result(
        val tier: Tier?,
        val status: Status,
        val qualified: Boolean,
        val targetP95Millimeters: Double,
        val targetP99Millimeters: Double,
        val uncertaintyAdjustedP95TargetMillimeters: Double,
        val uncertaintyAdjustedP99TargetMillimeters: Double = Double.POSITIVE_INFINITY,
        val sampleCount: Int,
        val measuredP95Millimeters: Double?,
        val measuredP99Millimeters: Double?,
        val referenceUncertaintyMillimeters: Double?,
        val surfaceEvidenceRequired: Boolean = false,
        val surfaceEvidenceAvailable: Boolean = false,
        val measuredSurfaceP95Millimeters: Double? = null,
        val measuredSurfaceP99Millimeters: Double? = null,
        val targetSurfaceP95Millimeters: Double? = null,
        val targetSurfaceP99Millimeters: Double? = null,
        val blockingReasons: List<String>,
        val warnings: List<String>,
    )

    fun evaluate(input: Input): Result {
        val tier = input.requestedTier
        if (tier == null) {
            return Result(
                tier = null,
                status = Status.NOT_REQUESTED,
                qualified = false,
                targetP95Millimeters =
                    input.targetP95Millimeters
                        ?: config.defaultReferenceP95TargetMillimeters,
                targetP99Millimeters =
                    input.targetP99Millimeters
                        ?: config.defaultReferenceP99TargetMillimeters,
                uncertaintyAdjustedP95TargetMillimeters =
                    input.targetP95Millimeters
                        ?: config.defaultReferenceP95TargetMillimeters,
                sampleCount = input.sampleCount,
                measuredP95Millimeters = input.measuredP95Millimeters,
                measuredP99Millimeters = input.measuredP99Millimeters,
                referenceUncertaintyMillimeters =
                    input.referenceUncertaintyMillimeters,
                blockingReasons = emptyList(),
                warnings = listOf(
                    "Precision qualification was not requested.",
                ),
            )
        }

        val defaultP95 =
            when (tier) {
                Tier.REFERENCE_TWO_MM ->
                    config.defaultReferenceP95TargetMillimeters

                Tier.SUB_MILLIMETER ->
                    config.defaultSubMillimeterP95TargetMillimeters
            }

        val defaultP99 =
            when (tier) {
                Tier.REFERENCE_TWO_MM ->
                    config.defaultReferenceP99TargetMillimeters

                Tier.SUB_MILLIMETER ->
                    config.defaultSubMillimeterP99TargetMillimeters
            }

        val targetP95 =
            input.targetP95Millimeters ?: defaultP95
        val targetP99 =
            input.targetP99Millimeters ?: defaultP99

        val minimumSamples =
            when (tier) {
                Tier.REFERENCE_TWO_MM ->
                    config.minimumReferenceSamples

                Tier.SUB_MILLIMETER ->
                    config.minimumSubMillimeterSamples
            }

        val blockers = buildList {
            if (!input.matchedCaptureConditions) {
                add("matched capture conditions are not established")
            }
            if (!input.empiricalBenchmarkAccepted) {
                add("empirical benchmark evidence is not accepted")
            }
            if (input.sampleCount < minimumSamples) {
                add(
                    "insufficient benchmark samples: " +
                        input.sampleCount +
                        " < " +
                        minimumSamples,
                )
            }
            if (!input.referenceProvenanceDocumented) {
                add("reference provenance and uncertainty are not documented")
            }
            if (
                tier == Tier.SUB_MILLIMETER &&
                    input.independentReferenceSessionCount <
                    config.minimumIndependentReferenceSessions
            ) {
                add("insufficient independent reference sessions")
            }

            val referenceUncertainty =
                input.referenceUncertaintyMillimeters

            if (referenceUncertainty == null) {
                add("reference uncertainty is missing")
            } else if (
                referenceUncertainty >=
                    targetP95 -
                        config.minimumReferenceUncertaintyMarginMillimeters
            ) {
                add("reference uncertainty consumes the P95 precision budget")
            }

            if (input.measuredP95Millimeters == null) {
                add("measured P95 is missing")
            }

            if (input.measuredP99Millimeters == null) {
                add("measured P99 is missing")
            }

            if (
                tier == Tier.SUB_MILLIMETER &&
                config.requireSurfaceEvidenceForSubMillimeter &&
                !input.surfaceEvidenceAvailable
            ) {
                add("dense surface benchmark evidence is missing")
            }

            if (
                tier == Tier.SUB_MILLIMETER &&
                config.requireSurfaceEvidenceForSubMillimeter &&
                input.measuredSurfaceP95Millimeters == null
            ) {
                add("dense surface P95 is missing for sub-millimeter qualification")
            }

            if (
                tier == Tier.SUB_MILLIMETER &&
                config.requireSurfaceEvidenceForSubMillimeter &&
                input.measuredSurfaceP99Millimeters == null
            ) {
                add("dense surface P99 is missing for sub-millimeter qualification")
            }
        }.toMutableList()

        val uncertaintyAdjustedP95Target =
            input.referenceUncertaintyMillimeters?.let { uncertainty ->
                sqrt(
                    (
                        targetP95 * targetP95 -
                            uncertainty * uncertainty
                        ).coerceAtLeast(0.0),
                )
            } ?: targetP95

        val uncertaintyAdjustedP99Target =
            input.referenceUncertaintyMillimeters?.let { uncertainty ->
                sqrt(
                    (
                        targetP99 * targetP99 -
                            uncertainty * uncertainty
                        ).coerceAtLeast(0.0),
                )
            } ?: targetP99

        input.measuredP95Millimeters?.let { measured ->
            if (measured > uncertaintyAdjustedP95Target) {
                blockers +=
                    "measured P95 exceeds the uncertainty-adjusted target"
            }
        }

        input.measuredP99Millimeters?.let { measured ->
            if (measured > uncertaintyAdjustedP99Target) {
                blockers +=
                    "measured P99 exceeds the uncertainty-adjusted target"
            }
        }

        if (
            tier == Tier.SUB_MILLIMETER &&
            config.requireSurfaceEvidenceForSubMillimeter
        ) {
            val surfaceP95Target =
                input.targetSurfaceP95Millimeters
                    ?: config.defaultSubMillimeterSurfaceP95TargetMillimeters
            val surfaceP99Target =
                input.targetSurfaceP99Millimeters
                    ?: config.defaultSubMillimeterSurfaceP99TargetMillimeters

            input.measuredSurfaceP95Millimeters?.let { measured ->
                if (
                    input.surfaceEvidenceAvailable &&
                    measured > surfaceP95Target
                ) {
                    blockers +=
                        "dense surface P95 exceeds the configured target"
                }
            }

            input.measuredSurfaceP99Millimeters?.let { measured ->
                if (
                    input.surfaceEvidenceAvailable &&
                    measured > surfaceP99Target
                ) {
                    blockers +=
                        "dense surface P99 exceeds the configured target"
                }
            }
        }

        if (
            tier == Tier.SUB_MILLIMETER &&
            targetP95 > config.defaultSubMillimeterP95TargetMillimeters
        ) {
            blockers +=
                "sub-millimeter tier cannot use a P95 target above 1.0 mm"
        }

        val distinctBlockers =
            blockers.distinct()

        val qualified =
            distinctBlockers.isEmpty()

        val warnings = buildList {
            if (tier == Tier.SUB_MILLIMETER) {
                add(
                    "Sub-millimeter qualification is a benchmark contract, not a universal accuracy claim.",
                )
            }
            if (qualified) {
                add(
                    "Qualification is valid only for the supplied benchmark conditions and reference setup.",
                )
            }
        }

        return Result(
            tier = tier,
            status =
                if (qualified) {
                    Status.QUALIFIED
                } else {
                    Status.BLOCKED
                },
            qualified = qualified,
            targetP95Millimeters = targetP95,
            targetP99Millimeters = targetP99,
            uncertaintyAdjustedP95TargetMillimeters =
                uncertaintyAdjustedP95Target,
            uncertaintyAdjustedP99TargetMillimeters =
                uncertaintyAdjustedP99Target,
            sampleCount = input.sampleCount,
            measuredP95Millimeters = input.measuredP95Millimeters,
            measuredP99Millimeters = input.measuredP99Millimeters,
            referenceUncertaintyMillimeters =
                input.referenceUncertaintyMillimeters,
            surfaceEvidenceRequired =
                tier == Tier.SUB_MILLIMETER &&
                    config.requireSurfaceEvidenceForSubMillimeter,
            surfaceEvidenceAvailable =
                input.surfaceEvidenceAvailable,
            measuredSurfaceP95Millimeters =
                input.measuredSurfaceP95Millimeters,
            measuredSurfaceP99Millimeters =
                input.measuredSurfaceP99Millimeters,
            targetSurfaceP95Millimeters =
                input.targetSurfaceP95Millimeters,
            targetSurfaceP99Millimeters =
                input.targetSurfaceP99Millimeters,
            blockingReasons = distinctBlockers,
            warnings = warnings,
        )
    }

    fun evaluateSubMillimeter(
        matchedCaptureConditions: Boolean,
        empiricalBenchmarkAccepted: Boolean,
        sampleCount: Int,
        independentReferenceSessionCount: Int = 0,
        measuredP95Millimeters: Double?,
        measuredP99Millimeters: Double?,
        referenceUncertaintyMillimeters: Double?,
        referenceProvenanceDocumented: Boolean,
    ): Result =
        evaluate(
            Input(
                requestedTier = Tier.SUB_MILLIMETER,
                matchedCaptureConditions = matchedCaptureConditions,
                empiricalBenchmarkAccepted = empiricalBenchmarkAccepted,
                sampleCount = sampleCount,
                independentReferenceSessionCount =
                    independentReferenceSessionCount,
                measuredP95Millimeters = measuredP95Millimeters,
                measuredP99Millimeters = measuredP99Millimeters,
                referenceUncertaintyMillimeters =
                    referenceUncertaintyMillimeters,
                referenceProvenanceDocumented =
                    referenceProvenanceDocumented,
                targetP95Millimeters =
                    config.defaultSubMillimeterP95TargetMillimeters,
                targetP99Millimeters =
                    config.defaultSubMillimeterP99TargetMillimeters,
                surfaceEvidenceAvailable = false,
                measuredSurfaceP95Millimeters = null,
                measuredSurfaceP99Millimeters = null,
                targetSurfaceP95Millimeters =
                    config.defaultSubMillimeterSurfaceP95TargetMillimeters,
                targetSurfaceP99Millimeters =
                    config.defaultSubMillimeterSurfaceP99TargetMillimeters,
            ),
        )

    fun evaluateSubMillimeterWithSurfaceEvidence(
        matchedCaptureConditions: Boolean,
        empiricalBenchmarkAccepted: Boolean,
        sampleCount: Int,
        measuredP95Millimeters: Double?,
        measuredP99Millimeters: Double?,
        referenceUncertaintyMillimeters: Double?,
        referenceProvenanceDocumented: Boolean,
        measuredSurfaceP95Millimeters: Double?,
        measuredSurfaceP99Millimeters: Double?,
    ): Result =
        evaluate(
            Input(
                requestedTier = Tier.SUB_MILLIMETER,
                matchedCaptureConditions = matchedCaptureConditions,
                empiricalBenchmarkAccepted = empiricalBenchmarkAccepted,
                sampleCount = sampleCount,
                measuredP95Millimeters = measuredP95Millimeters,
                measuredP99Millimeters = measuredP99Millimeters,
                referenceUncertaintyMillimeters =
                    referenceUncertaintyMillimeters,
                referenceProvenanceDocumented =
                    referenceProvenanceDocumented,
                targetP95Millimeters =
                    config.defaultSubMillimeterP95TargetMillimeters,
                targetP99Millimeters =
                    config.defaultSubMillimeterP99TargetMillimeters,
                surfaceEvidenceAvailable = true,
                measuredSurfaceP95Millimeters =
                    measuredSurfaceP95Millimeters,
                measuredSurfaceP99Millimeters =
                    measuredSurfaceP99Millimeters,
                targetSurfaceP95Millimeters =
                    config.defaultSubMillimeterSurfaceP95TargetMillimeters,
                targetSurfaceP99Millimeters =
                    config.defaultSubMillimeterSurfaceP99TargetMillimeters,
            ),
        )
}
