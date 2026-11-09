package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Empirical precision qualification authority.
 *
 * A camera pipeline can be architecturally sophisticated and still fail to achieve a physical
 * precision target because sensor noise, lens calibration, synchronization, baseline, motion blur,
 * thermal drift and detector uncertainty are coupled. This authority therefore separates:
 *
 * 1. evidence quality;
 * 2. eligibility to attempt a sub-millimeter target;
 * 3. measured target attainment;
 * 4. comparison qualification against an external reference.
 *
 * No state in this class turns a requested target into a claimed result. A target is only marked
 * attained when measured reference-aligned error evidence is supplied.
 */
class FrontierPrecisionQualificationEngine(
    private var config: Config = Config(),
) {
    data class Config(
        val targetP95Millimeters: Double = 0.75,
        val targetP99Millimeters: Double = 1.75,
        val minimumEvidenceSamples: Int = 600,
        val minimumBaselineMeters: Double = 0.60,
        val minimumParallaxDegrees: Double = 10.0,
        val minimumCalibrationConfidence: Double = 0.95,
        val minimumSynchronizationConfidence: Double = 0.95,
        val minimumViewDiversity: Double = 0.75,
        val minimumCrossViewAgreement: Double = 0.90,
        val maximumOcclusionRatio: Double = 0.12,
        val minimumIdentityContinuity: Double = 0.95,
        val minimumSurfaceTemporalStability: Double = 0.94,
        val maximumThermalRisk: Double = 0.60,
        val maximumMemoryPressure: Double = 0.75,
        val maximumTimingUncertaintyMicros: Double = 2500.0,
        val maximumReprojectionP95Pixels: Double = 2.0,
        val requiredIndependentViews: Int = 4,
    ) {
        init {
            require(targetP95Millimeters > 0.0)
            require(targetP99Millimeters >= targetP95Millimeters)
            require(minimumEvidenceSamples >= 1)
            require(minimumBaselineMeters > 0.0)
            require(minimumParallaxDegrees > 0.0 && minimumParallaxDegrees < 90.0)
            require(minimumCalibrationConfidence in 0.0..1.0)
            require(minimumSynchronizationConfidence in 0.0..1.0)
            require(minimumViewDiversity in 0.0..1.0)
            require(minimumCrossViewAgreement in 0.0..1.0)
            require(maximumOcclusionRatio in 0.0..1.0)
            require(minimumIdentityContinuity in 0.0..1.0)
            require(minimumSurfaceTemporalStability in 0.0..1.0)
            require(maximumThermalRisk in 0.0..1.0)
            require(maximumMemoryPressure in 0.0..1.0)
            require(maximumTimingUncertaintyMicros > 0.0)
            require(maximumReprojectionP95Pixels > 0.0)
            require(requiredIndependentViews >= 2)
        }
    }

    data class Input(
        val evidenceSamples: Int,
        val measuredP95Millimeters: Double? = null,
        val measuredP99Millimeters: Double? = null,
        val externalReferenceP95Millimeters: Double? = null,
        val externalReferenceUncertaintyMillimeters: Double? = null,
        val systemMeasurementUncertaintyMillimeters: Double? = null,
        val matchedReference: Boolean = false,
        val independentViews: Int,
        val baselineMeters: Double,
        val parallaxDegrees: Double,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val viewDiversity: Double,
        val crossViewAgreement: Double,
        val occlusionRatio: Double,
        val identityContinuity: Double,
        val surfaceTemporalStability: Double,
        val thermalRisk: Double,
        val memoryPressure: Double,
        val timingUncertaintyMicros: Double,
        val reprojectionP95Pixels: Double,
        val hardFailure: Boolean = false,
    ) {
        init {
            require(evidenceSamples >= 0)
            listOfNotNull(
                measuredP95Millimeters,
                measuredP99Millimeters,
                externalReferenceP95Millimeters,
                externalReferenceUncertaintyMillimeters,
                systemMeasurementUncertaintyMillimeters,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
            require(independentViews >= 0)
            require(baselineMeters.isFinite() && baselineMeters >= 0.0)
            require(parallaxDegrees.isFinite() && parallaxDegrees >= 0.0)
            require(calibrationConfidence in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(viewDiversity in 0.0..1.0)
            require(crossViewAgreement in 0.0..1.0)
            require(occlusionRatio in 0.0..1.0)
            require(identityContinuity in 0.0..1.0)
            require(surfaceTemporalStability in 0.0..1.0)
            require(thermalRisk in 0.0..1.0)
            require(memoryPressure in 0.0..1.0)
            require(timingUncertaintyMicros.isFinite() && timingUncertaintyMicros >= 0.0)
            require(reprojectionP95Pixels.isFinite() && reprojectionP95Pixels >= 0.0)
        }
    }

    enum class State {
        DISABLED,
        INELIGIBLE,
        EVIDENCE_READY,
        TARGET_ATTAINED,
        REFERENCE_QUALIFIED,
        BLOCKED,
    }

    enum class Blocker {
        NONE,
        HARD_FAILURE,
        INSUFFICIENT_SAMPLES,
        INSUFFICIENT_VIEWS,
        BASELINE,
        PARALLAX,
        CALIBRATION,
        SYNCHRONIZATION,
        VIEW_DIVERSITY,
        CROSS_VIEW,
        OCCLUSION,
        IDENTITY,
        SURFACE_TEMPORAL,
        THERMAL,
        MEMORY,
        TIMING,
        REPROJECTION,
        MEASUREMENT_MISSING,
        MEASUREMENT_EXCEEDS_TARGET,
        REFERENCE_MISSING,
        REFERENCE_WORSE_THAN_TARGET,
        EXTERNAL_REFERENCE_UNCERTAIN,
    }

    data class Result(
        val state: State,
        val readinessScore: Double,
        val evidenceQualityScore: Double,
        val targetP95Millimeters: Double,
        val targetP99Millimeters: Double,
        val measuredP95Millimeters: Double?,
        val measuredP99Millimeters: Double?,
        val externalReferenceP95Millimeters: Double?,
        val measuredWithinTarget: Boolean,
        val matchedReference: Boolean,
        val physicallyQualified: Boolean,
        val referenceQualified: Boolean,
        val blocker: Blocker,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val claimable: Boolean
            get() = physicallyQualified || referenceQualified
    }

    fun updateConfig(next: Config) {
        config = next
    }

    fun currentConfig(): Config = config

    fun evaluate(
        input: Input,
        enabled: Boolean = true,
    ): Result {
        if (!enabled) {
            return Result(
                state = State.DISABLED,
                readinessScore = 0.0,
                evidenceQualityScore = 0.0,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = input.measuredP95Millimeters,
                measuredP99Millimeters = input.measuredP99Millimeters,
                externalReferenceP95Millimeters =
                    input.externalReferenceP95Millimeters,
                measuredWithinTarget = false,
                matchedReference = input.matchedReference,
                physicallyQualified = false,
                referenceQualified = false,
                blocker = Blocker.NONE,
                warnings = listOf("PRECISION_QUALIFICATION_DISABLED"),
                recommendations =
                    listOf(
                        "Enable qualification only when reference-grade evidence collection is available.",
                    ),
            )
        }

        if (input.hardFailure) {
            return blocked(
                input = input,
                blocker = Blocker.HARD_FAILURE,
                warning = "Upstream hard failure prevents precision qualification.",
            )
        }

        val checks =
            listOf(
                qualityLowerBound(
                    input.calibrationConfidence,
                    config.minimumCalibrationConfidence,
                ) to 0.15,
                qualityLowerBound(
                    input.synchronizationConfidence,
                    config.minimumSynchronizationConfidence,
                ) to 0.15,
                qualityLowerBound(
                    input.viewDiversity,
                    config.minimumViewDiversity,
                ) to 0.10,
                qualityLowerBound(
                    input.crossViewAgreement,
                    config.minimumCrossViewAgreement,
                ) to 0.12,
                qualityUpperBound(
                    input.occlusionRatio,
                    config.maximumOcclusionRatio,
                ) to 0.07,
                qualityLowerBound(
                    input.identityContinuity,
                    config.minimumIdentityContinuity,
                ) to 0.08,
                qualityLowerBound(
                    input.surfaceTemporalStability,
                    config.minimumSurfaceTemporalStability,
                ) to 0.07,
                qualityUpperBound(
                    input.thermalRisk,
                    config.maximumThermalRisk,
                ) to 0.04,
                qualityUpperBound(
                    input.memoryPressure,
                    config.maximumMemoryPressure,
                ) to 0.04,
                qualityUpperBound(
                    input.timingUncertaintyMicros,
                    config.maximumTimingUncertaintyMicros,
                ) to 0.04,
                qualityUpperBound(
                    input.reprojectionP95Pixels,
                    config.maximumReprojectionP95Pixels,
                ) to 0.04,
            )

        val weighted =
            checks.sumOf { (value, weight) ->
                value.coerceIn(0.0, 1.0) * weight
            } /
                checks.sumOf { it.second }.coerceAtLeast(1e-9)

        val geometryReadiness =
            (
                min(
                    1.0,
                    input.baselineMeters /
                        config.minimumBaselineMeters,
                ) * 0.50 +
                    min(
                        1.0,
                        input.parallaxDegrees /
                            config.minimumParallaxDegrees,
                    ) * 0.30 +
                    min(
                        1.0,
                        input.independentViews.toDouble() /
                            config.requiredIndependentViews.toDouble(),
                    ) * 0.20
                ).coerceIn(0.0, 1.0)

        val sampleReadiness =
            min(
                1.0,
                input.evidenceSamples.toDouble() /
                    config.minimumEvidenceSamples.toDouble(),
            )

        val readiness =
            (
                weighted * 0.72 +
                    geometryReadiness * 0.18 +
                    sampleReadiness * 0.10
                ).coerceIn(0.0, 1.0)

        val blocker =
            firstBlocker(input)

        if (blocker != Blocker.NONE) {
            return Result(
                state = State.INELIGIBLE,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = input.measuredP95Millimeters,
                measuredP99Millimeters = input.measuredP99Millimeters,
                externalReferenceP95Millimeters =
                    input.externalReferenceP95Millimeters,
                measuredWithinTarget = false,
                matchedReference = input.matchedReference,
                physicallyQualified = false,
                referenceQualified = false,
                blocker = blocker,
                warnings =
                    listOf(
                        "PRECISION_QUALIFICATION_INCOMPLETE",
                        blocker.name,
                    ),
                recommendations = recommendationsFor(blocker),
            )
        }

        val p95 = input.measuredP95Millimeters
        val p99 = input.measuredP99Millimeters

        if (p95 == null || p99 == null) {
            return Result(
                state = State.EVIDENCE_READY,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = p95,
                measuredP99Millimeters = p99,
                externalReferenceP95Millimeters =
                    input.externalReferenceP95Millimeters,
                measuredWithinTarget = false,
                matchedReference = input.matchedReference,
                physicallyQualified = false,
                referenceQualified = false,
                blocker = Blocker.MEASUREMENT_MISSING,
                warnings =
                    listOf(
                        "PHYSICAL_PRECISION_TARGET_NOT_MEASURED",
                    ),
                recommendations =
                    listOf(
                        "Supply reference-aligned P95 and P99 positional error measurements before claiming target attainment.",
                    ),
            )
        }

        val measuredWithin =
            p95 <= config.targetP95Millimeters &&
                p99 <= config.targetP99Millimeters

        if (!measuredWithin) {
            return Result(
                state = State.INELIGIBLE,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = p95,
                measuredP99Millimeters = p99,
                externalReferenceP95Millimeters =
                    input.externalReferenceP95Millimeters,
                measuredWithinTarget = false,
                matchedReference = input.matchedReference,
                physicallyQualified = false,
                referenceQualified = false,
                blocker = Blocker.MEASUREMENT_EXCEEDS_TARGET,
                warnings =
                    listOf(
                        "MEASURED_ERROR_EXCEEDS_REQUESTED_TARGET",
                    ),
                recommendations =
                    listOf(
                        "Reduce reprojection, synchronization and calibration error or increase baseline and independent-view diversity.",
                    ),
            )
        }

        val reference =
            input.externalReferenceP95Millimeters

        if (!input.matchedReference || reference == null) {
            return Result(
                state = State.TARGET_ATTAINED,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = p95,
                measuredP99Millimeters = p99,
                externalReferenceP95Millimeters = reference,
                measuredWithinTarget = true,
                matchedReference = false,
                physicallyQualified = true,
                referenceQualified = false,
                blocker = Blocker.REFERENCE_MISSING,
                warnings =
                    listOf(
                        "TARGET_ATTAINED_WITHOUT_EXTERNAL_REFERENCE_COMPARISON",
                    ),
                recommendations =
                    listOf(
                        "Run matched reference trials before declaring parity or superiority against another motion-capture technology.",
                    ),
            )
        }

        val refUncertainty =
            input.externalReferenceUncertaintyMillimeters

        if (refUncertainty != null &&
            refUncertainty >
                config.targetP95Millimeters
        ) {
            return Result(
                state = State.TARGET_ATTAINED,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = p95,
                measuredP99Millimeters = p99,
                externalReferenceP95Millimeters = reference,
                measuredWithinTarget = true,
                matchedReference = true,
                physicallyQualified = true,
                referenceQualified = false,
                blocker = Blocker.EXTERNAL_REFERENCE_UNCERTAIN,
                warnings =
                    listOf(
                        "REFERENCE_UNCERTAINTY_TOO_LARGE_FOR_SUPERIORITY_QUALIFICATION",
                    ),
                recommendations =
                    listOf(
                        "Use a traceable reference with uncertainty materially smaller than the requested performance bound.",
                    ),
            )
        }

        val systemUncertainty =
            input.systemMeasurementUncertaintyMillimeters
                ?.coerceAtLeast(0.0)
                ?: 0.0
        val referenceUncertainty =
            refUncertainty
                .coerceAtLeast(0.0)

        val systemUpperBound =
            p95 + systemUncertainty
        val referenceUpperBound =
            reference + referenceUncertainty

        val separation =
            referenceUpperBound -
                systemUpperBound

        val pooledUncertainty =
            sqrt(
                systemUncertainty * systemUncertainty +
                    referenceUncertainty * referenceUncertainty,
            )

        val statisticallySeparated =
            separation >
                max(
                    0.05,
                    1.96 * pooledUncertainty,
                )

        val better =
            statisticallySeparated &&
                p95 < reference &&
                p99 <= reference * 1.15

        return if (better) {
            Result(
                state = State.REFERENCE_QUALIFIED,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = p95,
                measuredP99Millimeters = p99,
                externalReferenceP95Millimeters = reference,
                measuredWithinTarget = true,
                matchedReference = true,
                physicallyQualified = true,
                referenceQualified = true,
                blocker = Blocker.NONE,
                warnings =
                    buildList {
                        add("EMPIRICAL_REFERENCE_COMPARISON_FAVOURS_NOX")
                        if (
                            input.systemMeasurementUncertaintyMillimeters ==
                                null
                        ) {
                            add(
                                "SYSTEM_MEASUREMENT_UNCERTAINTY_NOT_SUPPLIED",
                            )
                        }
                    },
                recommendations =
                    listOf(
                        "Repeat across motion, occlusion, lighting, camera-layout and thermal conditions before generalizing the result.",
                    ),
            )
        } else {
            Result(
                state = State.TARGET_ATTAINED,
                readinessScore = readiness,
                evidenceQualityScore = weighted,
                targetP95Millimeters = config.targetP95Millimeters,
                targetP99Millimeters = config.targetP99Millimeters,
                measuredP95Millimeters = p95,
                measuredP99Millimeters = p99,
                externalReferenceP95Millimeters = reference,
                measuredWithinTarget = true,
                matchedReference = true,
                physicallyQualified = true,
                referenceQualified = false,
                blocker = Blocker.REFERENCE_WORSE_THAN_TARGET,
                warnings =
                    listOf(
                        "NOX_MEETS_TARGET_BUT_REFERENCE_SUPERIORITY_NOT_ESTABLISHED",
                    ),
                recommendations =
                    listOf(
                        "Compare matched trajectories over the same workspace, motion sequence and acquisition settings.",
                    ),
            )
        }
    }

    private fun firstBlocker(input: Input): Blocker {
        if (input.evidenceSamples < config.minimumEvidenceSamples) {
            return Blocker.INSUFFICIENT_SAMPLES
        }
        if (input.independentViews < config.requiredIndependentViews) {
            return Blocker.INSUFFICIENT_VIEWS
        }
        if (input.baselineMeters < config.minimumBaselineMeters) {
            return Blocker.BASELINE
        }
        if (input.parallaxDegrees < config.minimumParallaxDegrees) {
            return Blocker.PARALLAX
        }
        if (input.calibrationConfidence < config.minimumCalibrationConfidence) {
            return Blocker.CALIBRATION
        }
        if (input.synchronizationConfidence < config.minimumSynchronizationConfidence) {
            return Blocker.SYNCHRONIZATION
        }
        if (input.viewDiversity < config.minimumViewDiversity) {
            return Blocker.VIEW_DIVERSITY
        }
        if (input.crossViewAgreement < config.minimumCrossViewAgreement) {
            return Blocker.CROSS_VIEW
        }
        if (input.occlusionRatio > config.maximumOcclusionRatio) {
            return Blocker.OCCLUSION
        }
        if (input.identityContinuity < config.minimumIdentityContinuity) {
            return Blocker.IDENTITY
        }
        if (input.surfaceTemporalStability < config.minimumSurfaceTemporalStability) {
            return Blocker.SURFACE_TEMPORAL
        }
        if (input.thermalRisk > config.maximumThermalRisk) {
            return Blocker.THERMAL
        }
        if (input.memoryPressure > config.maximumMemoryPressure) {
            return Blocker.MEMORY
        }
        if (input.timingUncertaintyMicros > config.maximumTimingUncertaintyMicros) {
            return Blocker.TIMING
        }
        if (input.reprojectionP95Pixels > config.maximumReprojectionP95Pixels) {
            return Blocker.REPROJECTION
        }
        return Blocker.NONE
    }

    private fun blocked(
        input: Input,
        blocker: Blocker,
        warning: String,
    ): Result =
        Result(
            state = State.BLOCKED,
            readinessScore = 0.0,
            evidenceQualityScore = 0.0,
            targetP95Millimeters = config.targetP95Millimeters,
            targetP99Millimeters = config.targetP99Millimeters,
            measuredP95Millimeters = input.measuredP95Millimeters,
            measuredP99Millimeters = input.measuredP99Millimeters,
            externalReferenceP95Millimeters =
                input.externalReferenceP95Millimeters,
            measuredWithinTarget = false,
            matchedReference = input.matchedReference,
            physicallyQualified = false,
            referenceQualified = false,
            blocker = blocker,
            warnings = listOf(warning),
            recommendations = recommendationsFor(blocker),
        )

    private fun recommendationsFor(
        blocker: Blocker,
    ): List<String> =
        when (blocker) {
            Blocker.INSUFFICIENT_SAMPLES ->
                listOf("Increase reference-trial sample count.")
            Blocker.INSUFFICIENT_VIEWS ->
                listOf("Add independent camera viewpoints with useful parallax.")
            Blocker.BASELINE ->
                listOf("Increase physical baseline within the calibrated workspace.")
            Blocker.PARALLAX ->
                listOf("Reposition cameras to improve triangulation parallax.")
            Blocker.CALIBRATION ->
                listOf("Recalibrate intrinsics and extrinsics and verify thermal stability.")
            Blocker.SYNCHRONIZATION ->
                listOf("Tighten inter-camera temporal alignment and verify sensor timestamps.")
            Blocker.VIEW_DIVERSITY ->
                listOf("Increase angular coverage rather than repeating nearly co-linear views.")
            Blocker.CROSS_VIEW ->
                listOf("Reject identity-inconsistent views and improve cross-view correspondence.")
            Blocker.OCCLUSION ->
                listOf("Increase camera coverage around occluded regions.")
            Blocker.IDENTITY ->
                listOf("Increase cross-camera identity continuity evidence.")
            Blocker.SURFACE_TEMPORAL ->
                listOf("Stabilize dense surface bindings before qualification.")
            Blocker.THERMAL ->
                listOf("Allow thermal stabilization or recalibrate thermally drifting cameras.")
            Blocker.MEMORY ->
                listOf("Reduce retained buffers and shed non-critical detail before overflow.")
            Blocker.TIMING ->
                listOf("Reduce timing uncertainty and rolling-shutter exposure burden.")
            Blocker.REPROJECTION ->
                listOf("Improve calibration, correspondence and robust residual rejection.")
            Blocker.REFERENCE_MISSING ->
                listOf("Collect a matched external reference trial.")
            Blocker.MEASUREMENT_MISSING ->
                listOf("Collect reference-aligned P95 and P99 positional error.")
            Blocker.MEASUREMENT_EXCEEDS_TARGET ->
                listOf(
                    "Reduce measured P95/P99 error through calibration, synchronization, correspondence or view-geometry improvements before claiming target attainment.",
                )
            Blocker.REFERENCE_WORSE_THAN_TARGET ->
                listOf("Repeat matched reference trials across the complete operating envelope.")
            Blocker.EXTERNAL_REFERENCE_UNCERTAIN ->
                listOf("Use a higher-confidence measurement reference.")
            else ->
                emptyList()
        }

    private fun qualityLowerBound(
        measured: Double,
        minimum: Double,
    ): Double =
        if (measured >= minimum) {
            1.0
        } else {
            (measured / max(minimum, 1e-9))
                .coerceIn(0.0, 1.0)
        }

    private fun qualityUpperBound(
        measured: Double,
        maximum: Double,
    ): Double =
        if (measured <= maximum) {
            1.0
        } else {
            (maximum / max(measured, 1e-9))
                .coerceIn(0.0, 1.0)
        }
}
