package com.nivukx.nox.core.reconstruction

import kotlin.math.sqrt

/**
 * Converts modeled feasibility plus measured benchmark evidence into an explicit reference-grade
 * contract. A two-millimeter target stays blocked until real measurements, matched conditions,
 * and documented reference uncertainty support it.
 */
class ReferenceGradeEvidenceGateEngine(
    private val config: Config = Config(),
) {
    enum class Status { NOT_REQUESTED, BLOCKED, READY }

    data class Config(
        val minimumSamples: Int = 128,
        val minimumReferenceUncertaintyMargin: Double = 1e-3,
    ) {
        init {
            require(minimumSamples > 0)
            require(minimumReferenceUncertaintyMargin > 0.0)
        }
    }

    data class Input(
        val enabled: Boolean,
        val matchedCaptureConditions: Boolean = false,
        val empiricalBenchmarkAccepted: Boolean = false,
        val sampleCount: Int = 0,
        val measuredP95Millimeters: Double? = null,
        val measuredP99Millimeters: Double? = null,
        val targetP95Millimeters: Double = 2.0,
        val targetP99Millimeters: Double = 4.0,
        val referenceUncertaintyMillimeters: Double? = null,
        val externalReferenceDocumented: Boolean = false,
    ) {
        init {
            require(sampleCount >= 0)
            require(targetP95Millimeters > 0.0)
            require(targetP99Millimeters >= targetP95Millimeters)
            referenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            measuredP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            measuredP99Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Result(
        val status: Status,
        val ready: Boolean,
        val effectiveP95TargetMillimeters: Double,
        val blockingReasons: List<String>,
        val warnings: List<String>,
    )

    fun evaluate(
        report: CapturePrecisionBenchmarkEngine.Report,
        matchedCaptureConditions: Boolean,
    ): Result =
        evaluate(
            Input(
                enabled = true,
                matchedCaptureConditions = matchedCaptureConditions,
                empiricalBenchmarkAccepted = report.accepted,
                sampleCount = report.aggregate.sampleCount,
                measuredP95Millimeters = report.aggregate.p95ErrorMm,
                measuredP99Millimeters = report.aggregate.p99ErrorMm,
                referenceUncertaintyMillimeters =
                    report.referenceUncertaintyMillimeters,
                externalReferenceDocumented =
                    report.referenceType ==
                        CapturePrecisionBenchmarkEngine.ReferenceType.OPTICAL_REFERENCE ||
                        report.referenceType ==
                        CapturePrecisionBenchmarkEngine.ReferenceType.MARKER_REFERENCE,
            ),
        )

    fun evaluate(input: Input): Result {
        if (!input.enabled) {
            return Result(
                status = Status.NOT_REQUESTED,
                ready = false,
                effectiveP95TargetMillimeters = input.targetP95Millimeters,
                blockingReasons = emptyList(),
                warnings = listOf("Reference-grade mode is not requested."),
            )
        }

        val blockers = buildList {
            if (!input.matchedCaptureConditions) {
                add("matched capture conditions are not established")
            }
            if (!input.empiricalBenchmarkAccepted) {
                add("empirical benchmark evidence is missing or failed")
            }
            if (input.sampleCount < config.minimumSamples) {
                add("insufficient benchmark samples")
            }
            if (!input.externalReferenceDocumented) {
                add("external reference provenance and uncertainty are not documented")
            }
            val uncertainty = input.referenceUncertaintyMillimeters
            if (uncertainty == null) {
                add("reference uncertainty is missing")
            } else if (
                uncertainty >=
                    input.targetP95Millimeters -
                        config.minimumReferenceUncertaintyMargin
            ) {
                add("reference uncertainty consumes the target P95 budget")
            }
            if (
                input.measuredP95Millimeters == null ||
                input.measuredP99Millimeters == null
            ) {
                add("measured P95/P99 evidence is missing")
            }
        }

        val uncertainty = input.referenceUncertaintyMillimeters
        val effectiveTarget = if (uncertainty != null) {
            sqrt(
                (
                    input.targetP95Millimeters *
                        input.targetP95Millimeters -
                        uncertainty * uncertainty
                    ).coerceAtLeast(0.0),
            )
        } else {
            input.targetP95Millimeters
        }

        input.measuredP95Millimeters?.let {
            if (it > effectiveTarget) {
                return Result(
                    status = Status.BLOCKED,
                    ready = false,
                    effectiveP95TargetMillimeters = effectiveTarget,
                    blockingReasons = (
                        blockers +
                            "measured P95 exceeds the uncertainty-adjusted P95 target"
                        ).distinct(),
                    warnings = emptyList(),
                )
            }
        }
        input.measuredP99Millimeters?.let {
            if (it > input.targetP99Millimeters) {
                return Result(
                    status = Status.BLOCKED,
                    ready = false,
                    effectiveP95TargetMillimeters = effectiveTarget,
                    blockingReasons = (
                        blockers +
                            "measured P99 exceeds the configured P99 target"
                        ).distinct(),
                    warnings = emptyList(),
                )
            }
        }

        val ready = blockers.isEmpty()
        return Result(
            status = if (ready) Status.READY else Status.BLOCKED,
            ready = ready,
            effectiveP95TargetMillimeters = effectiveTarget,
            blockingReasons = blockers,
            warnings = if (ready) {
                listOf(
                    "Reference-grade status is evidence-qualified for the supplied benchmark conditions.",
                )
            } else {
                emptyList()
            },
        )
    }
}
