package com.nivukx.nox.core.calibration

import kotlin.math.max

/**
 * Cross-subsystem metrology admission gate.
 *
 * This coordinator deliberately separates three concepts:
 * 1. NoX internal precision metrics.
 * 2. Calibration stability required to trust those metrics in world space.
 * 3. External-reference comparability required for claims about another capture system.
 *
 * A METROLOGY_READY result is not a proof of physical sub-millimeter accuracy. It means the
 * measured calibration and precision envelopes satisfy the declared qualification protocol and
 * the capture can proceed to independent metrology verification.
 */
class MetrologyPrecisionQualificationCoordinator(
    private val config: Config = Config(),
) {

    enum class Status {
        BLOCKED,
        QUALIFICATION_READY,
        REFERENCE_COMPARABLE,
        REFERENCE_BETTER,
        REFERENCE_PARITY,
        REFERENCE_WORSE,
        INCONCLUSIVE,
    }

    data class Config(
        val maximumP95ErrorMillimeters: Double = 0.75,
        val maximumP99ErrorMillimeters: Double = 1.75,
        val maximumJitterMillimeters: Double = 0.50,
        val maximumLagMilliseconds: Double = 5.0,
        val maximumScaleDriftPercentPerMinute: Double = 0.25,
        val maximumSurfaceP95ErrorMillimeters: Double = 1.0,
        val maximumSurfaceNormalErrorDegrees: Double = 2.0,
        val requireSurfaceMetrologyEvidence: Boolean = true,
        val minimumCalibrationTrust: Double = 0.96,
        val maximumReferenceUncertaintyMillimeters: Double = 0.50,
        val minimumReferenceConfidence: Double = 0.90,
        val minimumMatchedPairs: Int = 40,
    ) {
        init {
            require(maximumP95ErrorMillimeters > 0.0)
            require(maximumP99ErrorMillimeters >= maximumP95ErrorMillimeters)
            require(maximumJitterMillimeters > 0.0)
            require(maximumLagMilliseconds > 0.0)
            require(maximumScaleDriftPercentPerMinute > 0.0)
            require(maximumSurfaceP95ErrorMillimeters > 0.0)
            require(maximumSurfaceNormalErrorDegrees > 0.0)
            require(minimumCalibrationTrust in 0.0..1.0)
            require(maximumReferenceUncertaintyMillimeters > 0.0)
            require(minimumReferenceConfidence in 0.0..1.0)
            require(minimumMatchedPairs >= 1)
        }

        companion object {
            fun productionMetrology(): Config = Config()
        }
    }

    data class Qualification(
        val status: Status,
        val qualified: Boolean,
        val calibrationQualified: Boolean,
        val precisionQualified: Boolean,
        val referenceComparable: Boolean,
        val referenceComparison:
            CapturePrecisionBenchmarkEngine.ReferenceComparison?,
        val blockingReasons: List<String>,
        val warnings: List<String>,
    )

    fun qualify(
        precisionReport: CapturePrecisionBenchmarkEngine.Report,
        calibrationSnapshot:
            MultiCameraCalibrationDriftEngine.Snapshot,
        referenceReport:
            CapturePrecisionBenchmarkEngine.Report? = null,
    ): Qualification {
        val aggregate = precisionReport.aggregate

        val calibrationQualified =
            !calibrationSnapshot.hardFailure &&
                !calibrationSnapshot.recalibrationRecommended &&
                calibrationSnapshot.rigTrust >=
                    config.minimumCalibrationTrust &&
                calibrationSnapshot.geometryTrust >=
                    config.minimumCalibrationTrust

        val surface =
            aggregate.surface

        val surfaceQualified =
            if (!config.requireSurfaceMetrologyEvidence) {
                true
            } else {
                surface != null &&
                    surface.p95PointErrorMm <=
                        config.maximumSurfaceP95ErrorMillimeters &&
                    (
                        surface.meanNormalErrorDegrees
                            ?: Double.POSITIVE_INFINITY
                    ) <=
                        config.maximumSurfaceNormalErrorDegrees
            }

        val precisionQualified =
            precisionReport.accepted &&
                aggregate.p95ErrorMm <=
                    config.maximumP95ErrorMillimeters &&
                aggregate.p99ErrorMm <=
                    config.maximumP99ErrorMillimeters &&
                aggregate.jitterRmsMm <=
                    config.maximumJitterMillimeters &&
                aggregate.temporalLagMs <=
                    config.maximumLagMilliseconds &&
                (
                    aggregate.scaleDriftPercentPerMinute
                        ?: Double.POSITIVE_INFINITY
                ) <=
                    config.maximumScaleDriftPercentPerMinute &&
                surfaceQualified

        val referenceComparable =
            referenceReport != null &&
                referenceReport.referenceType ==
                    CapturePrecisionBenchmarkEngine.ReferenceType
                        .ACTIVE_OPTICAL_MARKER_REFERENCE &&
                referenceReport.referenceUncertaintyMillimeters
                    ?.let {
                        it.isFinite() &&
                            it <=
                                config.maximumReferenceUncertaintyMillimeters
                    } == true &&
                referenceReport.confidence >=
                    config.minimumReferenceConfidence &&
                referenceReport.aggregate.sampleCount >=
                    precisionReport.aggregate.sampleCount.coerceAtLeast(
                        config.minimumMatchedPairs,
                    )

        val comparison =
            if (
                referenceComparable &&
                    referenceReport != null
            ) {
                CapturePrecisionBenchmarkEngine(
                    CapturePrecisionBenchmarkEngine.Config(
                        minimumSamples =
                            minOf(
                                precisionReport.aggregate.sampleCount,
                                referenceReport.aggregate.sampleCount,
                            ).coerceAtLeast(1),
                    ),
                ).compareAgainstReference(
                    nox = precisionReport,
                    reference = referenceReport,
                )
            } else {
                null
            }

        val blockingReasons =
            buildList {
                if (!calibrationQualified) {
                    add("metrology-calibration-not-qualified")
                }
                if (!precisionQualified) {
                    add("metrology-precision-envelope-not-qualified")
                }
                if (
                    config.requireSurfaceMetrologyEvidence &&
                    surface == null
                ) {
                    add("metrology-surface-evidence-missing")
                }
                if (
                    surface != null &&
                    surface.p95PointErrorMm >
                        config.maximumSurfaceP95ErrorMillimeters
                ) {
                    add("metrology-surface-p95-error-too-large")
                }
                if (
                    surface?.meanNormalErrorDegrees != null &&
                    surface.meanNormalErrorDegrees >
                        config.maximumSurfaceNormalErrorDegrees
                ) {
                    add("metrology-surface-normal-error-too-large")
                }
                if (referenceReport != null &&
                    referenceReport.referenceType !=
                        CapturePrecisionBenchmarkEngine.ReferenceType
                            .ACTIVE_OPTICAL_MARKER_REFERENCE
                ) {
                    add("metrology-reference-family-not-active-optical")
                }
                if (
                    referenceReport?.referenceUncertaintyMillimeters == null
                ) {
                    if (referenceReport != null) {
                        add("metrology-reference-uncertainty-missing")
                    }
                } else if (
                    referenceReport.referenceUncertaintyMillimeters >
                        config.maximumReferenceUncertaintyMillimeters
                ) {
                    add("metrology-reference-uncertainty-too-large")
                }
                if (
                    referenceReport != null &&
                    referenceReport.confidence <
                        config.minimumReferenceConfidence
                ) {
                    add("metrology-reference-confidence-too-low")
                }
                if (
                    referenceReport != null &&
                    referenceReport.aggregate.sampleCount <
                        config.minimumMatchedPairs
                ) {
                    add("metrology-reference-sample-count-too-low")
                }
            }.distinct()

        val warnings =
            buildList {
                if (aggregate.p95ErrorMm >
                    config.maximumP95ErrorMillimeters * 0.80
                ) {
                    add("p95-error-near-metrology-limit")
                }
                if (aggregate.p99ErrorMm >
                    config.maximumP99ErrorMillimeters * 0.80
                ) {
                    add("p99-error-near-metrology-limit")
                }
                if (
                    surface != null &&
                    surface.p95PointErrorMm >
                        config.maximumSurfaceP95ErrorMillimeters * 0.80
                ) {
                    add("surface-p95-error-near-metrology-limit")
                }
                if (calibrationSnapshot.temporalTrust < 0.95) {
                    add("temporal-calibration-trust-below-reference-grade")
                }
                comparison?.rationale?.forEach { rationale ->
                    add("reference:" + rationale)
                }
            }.distinct()

        val status =
            when {
                !calibrationQualified ||
                    !precisionQualified ->
                    Status.BLOCKED
                comparison == null ->
                    Status.QUALIFICATION_READY
                comparison.verdict ==
                    CapturePrecisionBenchmarkEngine.ReferenceVerdict.BETTER ->
                    Status.REFERENCE_BETTER
                comparison.verdict ==
                    CapturePrecisionBenchmarkEngine.ReferenceVerdict.EQUIVALENT ->
                    Status.REFERENCE_PARITY
                comparison.verdict ==
                    CapturePrecisionBenchmarkEngine.ReferenceVerdict.WORSE ->
                    Status.REFERENCE_WORSE
                else ->
                    Status.INCONCLUSIVE
            }

        return Qualification(
            status = status,
            qualified =
                calibrationQualified &&
                    precisionQualified,
            calibrationQualified = calibrationQualified,
            precisionQualified = precisionQualified,
            referenceComparable = referenceComparable,
            referenceComparison = comparison,
            blockingReasons = blockingReasons,
            warnings = warnings,
        )
    }
}
