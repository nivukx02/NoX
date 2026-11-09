package com.nivukx.nox.core.reconstruction

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Evidence-driven ultra-precision qualification engine for markerless RGB multi-view capture.
 *
 * This engine deliberately separates:
 *  - geometric/modelled precision,
 *  - empirical measurement evidence,
 *  - matched-reference comparison,
 *  - and production capture readiness.
 *
 * A sub-millimeter target is therefore a qualification condition, never a hard-coded claim
 * that arbitrary consumer cameras will achieve sub-millimeter world-space accuracy.
 */
class UltraPrecisionEvidenceEngine {

    enum class ReferenceKind {
        NONE,
        OPTICAL_INFRARED_MARKER_SYSTEM,
        MARKER_BASED_OPTICAL_SYSTEM,
        HIGH_PRECISION_CALIBRATION_RIG,
        ROBOTIC_REFERENCE,
    }

    enum class Status {
        DISABLED,
        MODELED_ONLY,
        EVIDENCE_INSUFFICIENT,
        QUALIFIED,
        BLOCKED,
    }

    data class Policy(
        val enabled: Boolean = false,
        val qualificationMode: Boolean = false,
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
        val matchedReferenceRequired: Boolean = true,
        val referenceKind: ReferenceKind = ReferenceKind.NONE,
    ) {
        init {
            require(targetP95Millimeters in 0.05..5.0)
            require(targetP99Millimeters >= targetP95Millimeters)
            require(minimumEvidenceSamples >= 60)
            require(minimumBaselineMeters > 0.0)
            require(minimumParallaxDegrees in 1.0..89.0)
            listOf(
                minimumCalibrationConfidence,
                minimumSynchronizationConfidence,
                minimumViewDiversity,
                minimumCrossViewAgreement,
                maximumOcclusionRatio,
                minimumIdentityContinuity,
                minimumSurfaceTemporalStability,
                maximumThermalRisk,
                maximumMemoryPressure,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
        }

        companion object {
            fun disabled(): Policy = Policy()

            fun highEndQualification(): Policy =
                Policy(
                    enabled = true,
                    qualificationMode = true,
                    targetP95Millimeters = 0.75,
                    targetP99Millimeters = 1.75,
                    minimumEvidenceSamples = 600,
                    minimumBaselineMeters = 0.60,
                    minimumParallaxDegrees = 10.0,
                    minimumCalibrationConfidence = 0.95,
                    minimumSynchronizationConfidence = 0.95,
                    minimumViewDiversity = 0.75,
                    minimumCrossViewAgreement = 0.90,
                    maximumOcclusionRatio = 0.12,
                    minimumIdentityContinuity = 0.95,
                    minimumSurfaceTemporalStability = 0.94,
                    maximumThermalRisk = 0.60,
                    maximumMemoryPressure = 0.75,
                    matchedReferenceRequired = true,
                )
        }
    }

    data class Input(
        val workingDistanceMeters: Double,
        val cameraBaselineMeters: Double,
        val effectiveFocalLengthPixels: Double,
        val medianParallaxDegrees: Double,
        val reprojectionErrorPixels: Double,
        val subpixelObservationSigmaPixels: Double = 0.25,
        val cameraTranslationSigmaMeters: Double = 0.0015,
        val cameraAngularSigmaDegrees: Double = 0.05,
        val synchronizationSkewMilliseconds: Double = 1.0,
        val rollingShutterEquivalentMilliseconds: Double = 0.0,
        val motionSpeedMetersPerSecond: Double = 1.0,
        /** Robust exposure integration duration across the contributing camera set. */
        val exposureIntegrationMilliseconds: Double = 0.0,
        val metricScaleUncertaintyRatio: Double = 0.001,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val viewDiversity: Double,
        val crossViewAgreement: Double,
        val occlusionRatio: Double,
        val identityContinuity: Double,
        val surfaceTemporalStability: Double,
        val thermalRisk: Double,
        val memoryPressure: Double,
        val subMillimeterGeometryFeasibility: Double = 0.0,
        val firstOrderDepthErrorMillimeters: Double = Double.POSITIVE_INFINITY,
        val tailAdjustedFirstOrderDepthErrorMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val tailAdjustedSubMillimeterFeasibilityScore: Double = 0.0,
        val denseMotionStability: Double = 0.0,
        val denseKinematicInnovationMillimeters: Double = 0.0,
        val observationCount: Int,
        val empiricalP95Millimeters: Double? = null,
        val empiricalP99Millimeters: Double? = null,
        val empiricalSampleCount: Int = 0,
        val matchedReference: Boolean = false,
        val referenceKind: ReferenceKind = ReferenceKind.NONE,
        val referenceP95Millimeters: Double? = null,
        val referenceUncertaintyMillimeters: Double? = null,
        val systemMeasurementUncertaintyMillimeters: Double? = null,
    ) {
        init {
            listOf(
                workingDistanceMeters,
                cameraBaselineMeters,
                effectiveFocalLengthPixels,
                medianParallaxDegrees,
                reprojectionErrorPixels,
                subpixelObservationSigmaPixels,
                cameraTranslationSigmaMeters,
                cameraAngularSigmaDegrees,
                synchronizationSkewMilliseconds,
                rollingShutterEquivalentMilliseconds,
                motionSpeedMetersPerSecond,
                exposureIntegrationMilliseconds,
                metricScaleUncertaintyRatio,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
            listOf(
                calibrationConfidence,
                synchronizationConfidence,
                viewDiversity,
                crossViewAgreement,
                occlusionRatio,
                identityContinuity,
                surfaceTemporalStability,
                thermalRisk,
                memoryPressure,
                subMillimeterGeometryFeasibility,
                denseMotionStability,
                tailAdjustedSubMillimeterFeasibilityScore,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(medianParallaxDegrees in 0.0..179.0)
            require(observationCount >= 0)
            require(empiricalSampleCount >= 0)
            empiricalP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            empiricalP99Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            referenceP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            referenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            systemMeasurementUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class ErrorBudget(
        val triangulationMillimeters: Double,
        val reprojectionMillimeters: Double,
        val cameraPoseMillimeters: Double,
        val synchronizationMillimeters: Double,
        val rollingShutterMillimeters: Double,
        val exposureBlurMillimeters: Double = 0.0,
        val scaleMillimeters: Double,
        val combinedSigmaMillimeters: Double,
        val modeledP95Millimeters: Double,
        val modeledP99Millimeters: Double,
    )

    data class ReferenceComparison(
        val kind: ReferenceKind,
        val comparable: Boolean,
        val statisticallySeparated: Boolean,
        val measuredLowerError: Boolean,
        val systemP95PlusUncertaintyMillimeters: Double?,
        val referenceP95PlusUncertaintyMillimeters: Double?,
        val marginMillimeters: Double?,
    )

    data class Result(
        val status: Status,
        val policy: Policy,
        val errorBudget: ErrorBudget,
        val evidenceScore: Double,
        val empiricalEvidenceAvailable: Boolean,
        val matchedReferenceEvidenceAvailable: Boolean,
        val qualificationEligible: Boolean,
        val referenceComparison: ReferenceComparison,
        val requiredBaselineMetersForTarget: Double,
        val recommendedBaselineMeters: Double,
        val warnings: List<String>,
        val blockers: List<String>,
    )

    fun evaluate(
        policy: Policy,
        input: Input,
    ): Result {
        if (!policy.enabled) {
            return Result(
                status = Status.DISABLED,
                policy = policy,
                errorBudget = emptyBudget(),
                evidenceScore = 1.0,
                empiricalEvidenceAvailable = false,
                matchedReferenceEvidenceAvailable = false,
                qualificationEligible = false,
                referenceComparison = ReferenceComparison(
                    kind = ReferenceKind.NONE,
                    comparable = false,
                    statisticallySeparated = false,
                    measuredLowerError = false,
                    systemP95PlusUncertaintyMillimeters = null,
                    referenceP95PlusUncertaintyMillimeters = null,
                    marginMillimeters = null,
                ),
                requiredBaselineMetersForTarget = input.cameraBaselineMeters,
                recommendedBaselineMeters = input.cameraBaselineMeters,
                warnings = emptyList(),
                blockers = emptyList(),
            )
        }

        val warnings = mutableListOf<String>()
        val blockers = mutableListOf<String>()

        val budget = modelErrorBudget(
            policy = policy,
            input = input,
        )

        if (input.cameraBaselineMeters < policy.minimumBaselineMeters) {
            blockers += "baseline-below-ultra-precision-minimum"
        }
        if (input.medianParallaxDegrees < policy.minimumParallaxDegrees) {
            blockers += "parallax-below-ultra-precision-minimum"
        }
        if (input.calibrationConfidence < policy.minimumCalibrationConfidence) {
            blockers += "calibration-confidence-below-ultra-precision-minimum"
        }
        if (
            input.synchronizationConfidence <
                policy.minimumSynchronizationConfidence
        ) {
            blockers += "synchronization-confidence-below-ultra-precision-minimum"
        }
        if (input.viewDiversity < policy.minimumViewDiversity) {
            blockers += "view-diversity-below-ultra-precision-minimum"
        }
        if (input.crossViewAgreement < policy.minimumCrossViewAgreement) {
            blockers += "cross-view-agreement-below-ultra-precision-minimum"
        }
        if (input.occlusionRatio > policy.maximumOcclusionRatio) {
            blockers += "occlusion-above-ultra-precision-limit"
        }
        if (input.identityContinuity < policy.minimumIdentityContinuity) {
            blockers += "identity-continuity-below-ultra-precision-minimum"
        }
        if (
            input.surfaceTemporalStability <
                policy.minimumSurfaceTemporalStability
        ) {
            blockers += "surface-temporal-stability-below-ultra-precision-minimum"
        }
        if (input.subMillimeterGeometryFeasibility < 0.70) {
            blockers += "dense-geometry-submillimeter-evidence-below-threshold"
        }
        if (input.denseMotionStability < 0.70) {
            blockers += "dense-motion-stability-below-threshold"
        }
        if (input.thermalRisk > policy.maximumThermalRisk) {
            blockers += "thermal-risk-above-ultra-precision-limit"
        }
        if (input.memoryPressure > policy.maximumMemoryPressure) {
            blockers += "memory-pressure-above-ultra-precision-limit"
        }
        if (budget.modeledP95Millimeters > policy.targetP95Millimeters) {
            warnings += "modeled-p95-above-ultra-precision-target"
        }
        if (budget.modeledP99Millimeters > policy.targetP99Millimeters) {
            warnings += "modeled-p99-above-ultra-precision-target"
        }
        if (input.reprojectionErrorPixels > 1.0) {
            warnings += "reprojection-error-above-one-pixel"
        }
        if (
            input.tailAdjustedFirstOrderDepthErrorMillimeters.isFinite() &&
            input.tailAdjustedFirstOrderDepthErrorMillimeters >
                policy.targetP95Millimeters
        ) {
            warnings += "tail-adjusted-depth-error-above-ultra-precision-target"
        }
        if (
            input.tailAdjustedSubMillimeterFeasibilityScore <
                policy.minimumViewDiversity * 0.70
        ) {
            warnings += "tail-adjusted-geometry-feasibility-is-degraded"
        }
        if (input.synchronizationSkewMilliseconds > 2.0) {
            warnings += "temporal-skew-above-two-milliseconds"
        }
        if (input.rollingShutterEquivalentMilliseconds > 2.0) {
            warnings += "rolling-shutter-contribution-is-material"
        }
        if (input.exposureIntegrationMilliseconds > 4.0 &&
            input.motionSpeedMetersPerSecond > 0.5
        ) {
            warnings += "exposure-motion-blur-contribution-is-material"
        }
        if (policy.qualificationMode &&
            input.exposureIntegrationMilliseconds > 8.0 &&
            input.motionSpeedMetersPerSecond > 1.0
        ) {
            blockers += "exposure-motion-blur-too-large-for-precision-qualification"
        }

        val empiricalAvailable =
            input.empiricalP95Millimeters != null &&
                input.empiricalSampleCount >= policy.minimumEvidenceSamples

        val referenceAvailable =
            input.matchedReference &&
                input.referenceP95Millimeters != null &&
                input.referenceUncertaintyMillimeters != null &&
                input.referenceKind != ReferenceKind.NONE

        if (policy.qualificationMode && !empiricalAvailable) {
            blockers += "empirical-evidence-sample-count-insufficient"
        }

        if (
            policy.qualificationMode &&
            policy.matchedReferenceRequired &&
            !referenceAvailable
        ) {
            blockers += "matched-reference-evidence-required"
        }

        val referenceComparison = compareReference(
            input = input,
            policy = policy,
            empiricalAvailable = empiricalAvailable,
        )

        if (
            policy.qualificationMode &&
            referenceAvailable &&
            !referenceComparison.statisticallySeparated
        ) {
            blockers += "reference-comparison-not-statistically-separated"
        }

        val qualificationEligible =
            policy.qualificationMode &&
                empiricalAvailable &&
                budgetInvariantPasses(policy, input, budget) &&
                blockers.isEmpty() &&
                input.empiricalP95Millimeters
                    ?.let { it <= policy.targetP95Millimeters }
                    ?: false &&
                input.empiricalP99Millimeters
                    ?.let { it <= policy.targetP99Millimeters }
                    ?: true &&
                (!policy.matchedReferenceRequired ||
                    referenceComparison.statisticallySeparated)

        val status =
            when {
                blockers.isNotEmpty() && policy.qualificationMode ->
                    Status.BLOCKED
                qualificationEligible ->
                    Status.QUALIFIED
                !empiricalAvailable ->
                    Status.MODELED_ONLY
                blockers.isNotEmpty() ->
                    Status.BLOCKED
                else ->
                    Status.EVIDENCE_INSUFFICIENT
            }

        val evidenceScore = (
            calculateEvidenceScore(
                input = input,
                policy = policy,
                budget = budget,
                empiricalAvailable = empiricalAvailable,
            ) * (
                0.78 +
                    0.07 *
                        input.subMillimeterGeometryFeasibility
                            .coerceIn(0.0, 1.0) +
                    0.06 *
                        input.tailAdjustedSubMillimeterFeasibilityScore
                            .coerceIn(0.0, 1.0) +
                    0.05 *
                        input.denseMotionStability
                            .coerceIn(0.0, 1.0) +
                    0.03 *
                        (
                            1.0 -
                                (
                                    input.exposureIntegrationMilliseconds *
                                        0.025 *
                                        input.motionSpeedMetersPerSecond
                                ).coerceIn(0.0, 1.0)
                            ) +
                    0.04 *
                        (
                            1.0 -
                                (
                                    input.denseKinematicInnovationMillimeters /
                                        4.0
                                    ).coerceIn(0.0, 1.0)
                            )
                    )
            ).coerceIn(0.0, 1.0)

        val requiredBaseline = requiredBaselineForTarget(
            targetMillimeters = policy.targetP95Millimeters,
            input = input,
            scaleFactor = 1.96,
        )

        return Result(
            status = status,
            policy = policy,
            errorBudget = budget,
            evidenceScore = evidenceScore,
            empiricalEvidenceAvailable = empiricalAvailable,
            matchedReferenceEvidenceAvailable = referenceAvailable,
            qualificationEligible = qualificationEligible,
            referenceComparison = referenceComparison,
            requiredBaselineMetersForTarget = requiredBaseline,
            recommendedBaselineMeters =
                max(
                    policy.minimumBaselineMeters,
                    requiredBaseline * 1.15,
                ),
            warnings = warnings.distinct(),
            blockers = blockers.distinct(),
        )
    }

    private fun modelErrorBudget(
        policy: Policy,
        input: Input,
    ): ErrorBudget {
        val parallaxRadians =
            (input.medianParallaxDegrees.coerceIn(0.05, 179.0) * PI / 180.0)
        val sinParallax =
            sin(parallaxRadians).coerceIn(0.06, 1.0)

        val triangulationMeters =
            input.workingDistanceMeters.pow(2.0) /
                max(
                    1e-6,
                    input.effectiveFocalLengthPixels *
                        input.cameraBaselineMeters,
                ) *
                input.subpixelObservationSigmaPixels /
                sinParallax

        val reprojectionMeters =
            input.workingDistanceMeters /
                max(1.0, input.effectiveFocalLengthPixels) *
                input.reprojectionErrorPixels /
                sinParallax

        val poseAngularMeters =
            input.workingDistanceMeters *
                (
                    input.cameraAngularSigmaDegrees *
                        PI /
                        180.0
                    )

        val cameraPoseMeters =
            sqrt(
                input.cameraTranslationSigmaMeters.pow(2.0) +
                    poseAngularMeters.pow(2.0),
            )

        val synchronizationMeters =
            input.motionSpeedMetersPerSecond *
                (input.synchronizationSkewMilliseconds / 1000.0)

        val rollingShutterMeters =
            input.motionSpeedMetersPerSecond *
                (input.rollingShutterEquivalentMilliseconds / 1000.0) *
                0.50

        val exposureBlurMeters =
            input.motionSpeedMetersPerSecond *
                (input.exposureIntegrationMilliseconds / 1000.0)

        val scaleMeters =
            input.workingDistanceMeters *
                input.metricScaleUncertaintyRatio

        val sigmaMillimeters =
            sqrt(
                triangulationMeters.pow(2.0) +
                    reprojectionMeters.pow(2.0) +
                    cameraPoseMeters.pow(2.0) +
                    synchronizationMeters.pow(2.0) +
                    rollingShutterMeters.pow(2.0) +
                    exposureBlurMeters.pow(2.0) +
                    scaleMeters.pow(2.0),
            ) * 1000.0

        val regimeInflation =
            (
                1.0 +
                    input.occlusionRatio * 0.40 +
                    (1.0 - input.crossViewAgreement) * 0.35 +
                    (1.0 - input.surfaceTemporalStability) * 0.25 +
                    input.thermalRisk * 0.10 +
                    input.memoryPressure * 0.10
                ).coerceIn(1.0, 2.75)

        val sigma = sigmaMillimeters * regimeInflation

        return ErrorBudget(
            triangulationMillimeters = triangulationMeters * 1000.0,
            reprojectionMillimeters = reprojectionMeters * 1000.0,
            cameraPoseMillimeters = cameraPoseMeters * 1000.0,
            synchronizationMillimeters = synchronizationMeters * 1000.0,
            rollingShutterMillimeters = rollingShutterMeters * 1000.0,
            exposureBlurMillimeters =
                exposureBlurMeters * 1000.0,
            scaleMillimeters = scaleMeters * 1000.0,
            combinedSigmaMillimeters = sigma,
            modeledP95Millimeters = sigma * 1.96,
            modeledP99Millimeters = sigma * 2.576,
        )
    }

    private fun budgetInvariantPasses(
        policy: Policy,
        input: Input,
        budget: ErrorBudget,
    ): Boolean {
        return budget.modeledP95Millimeters <=
            max(policy.targetP95Millimeters * 2.50, 1.50) &&
            input.cameraBaselineMeters >= policy.minimumBaselineMeters &&
            input.medianParallaxDegrees >= policy.minimumParallaxDegrees &&
            input.calibrationConfidence >= policy.minimumCalibrationConfidence &&
            input.synchronizationConfidence >=
                policy.minimumSynchronizationConfidence &&
            input.viewDiversity >= policy.minimumViewDiversity &&
            input.crossViewAgreement >= policy.minimumCrossViewAgreement
    }

    private fun compareReference(
        input: Input,
        policy: Policy,
        empiricalAvailable: Boolean,
    ): ReferenceComparison {
        if (
            !empiricalAvailable ||
            !input.matchedReference ||
            input.referenceP95Millimeters == null ||
            input.referenceUncertaintyMillimeters == null ||
            input.empiricalP95Millimeters == null ||
            input.referenceKind == ReferenceKind.NONE
        ) {
            return ReferenceComparison(
                kind = input.referenceKind,
                comparable = false,
                statisticallySeparated = false,
                measuredLowerError = false,
                systemP95PlusUncertaintyMillimeters =
                    input.empiricalP95Millimeters,
                referenceP95PlusUncertaintyMillimeters =
                    input.referenceP95Millimeters,
                marginMillimeters = null,
            )
        }

        val systemUncertainty =
            input.systemMeasurementUncertaintyMillimeters
                ?.coerceAtLeast(0.0)
                ?: 0.0
        val referenceUncertainty =
            input.referenceUncertaintyMillimeters
                .coerceAtLeast(0.0)

        val system =
            input.empiricalP95Millimeters +
                systemUncertainty
        val reference =
            input.referenceP95Millimeters +
                referenceUncertainty

        val separation =
            reference - system
        val pooledUncertainty =
            sqrt(
                systemUncertainty.pow(2.0) +
                    referenceUncertainty.pow(2.0),
            )
        val statisticallySeparated =
            abs(separation) >
                max(
                    0.05,
                    1.96 * pooledUncertainty,
                )

        return ReferenceComparison(
            kind = input.referenceKind,
            comparable = true,
            statisticallySeparated = statisticallySeparated,
            measuredLowerError =
                statisticallySeparated &&
                    separation > 0.0,
            systemP95PlusUncertaintyMillimeters = system,
            referenceP95PlusUncertaintyMillimeters = reference,
            marginMillimeters = separation,
        )
    }

    private fun calculateEvidenceScore(
        input: Input,
        policy: Policy,
        budget: ErrorBudget,
        empiricalAvailable: Boolean,
    ): Double {
        val geometry =
            (
                0.50 *
                    (input.cameraBaselineMeters /
                        max(
                            input.cameraBaselineMeters,
                            policy.minimumBaselineMeters,
                        )).coerceIn(0.0, 1.0) +
                    0.50 *
                        (input.medianParallaxDegrees /
                            max(
                                input.medianParallaxDegrees,
                                policy.minimumParallaxDegrees,
                            )).coerceIn(0.0, 1.0)
                ).coerceIn(0.0, 1.0)

        val evidence =
            if (empiricalAvailable) 1.0 else 0.35

        return (
            0.16 * geometry +
                0.18 * input.calibrationConfidence +
                0.18 * input.synchronizationConfidence +
                0.12 * input.viewDiversity +
                0.14 * input.crossViewAgreement +
                0.08 * input.identityContinuity +
                0.06 * input.surfaceTemporalStability +
                0.08 * evidence -
                0.06 *
                    (budget.modeledP95Millimeters /
                        max(0.1, policy.targetP95Millimeters))
                    .coerceAtMost(3.0) /
                    3.0
            ).coerceIn(0.0, 1.0)
    }

    private fun requiredBaselineForTarget(
        targetMillimeters: Double,
        input: Input,
        scaleFactor: Double,
    ): Double {
        val targetMeters =
            (targetMillimeters / scaleFactor) / 1000.0
        val parallaxFactor =
            sin(
                input.medianParallaxDegrees.coerceIn(0.05, 89.0) *
                    PI /
                    180.0,
            ).coerceAtLeast(0.06)
        val usableSigma =
            max(0.05, input.subpixelObservationSigmaPixels)
        val baseline =
            input.workingDistanceMeters.pow(2.0) *
                usableSigma /
                max(
                    1e-6,
                    input.effectiveFocalLengthPixels *
                        targetMeters *
                        parallaxFactor,
                )
        return baseline.coerceAtLeast(0.05)
    }

    private fun emptyBudget() = ErrorBudget(
        triangulationMillimeters = 0.0,
        reprojectionMillimeters = 0.0,
        cameraPoseMillimeters = 0.0,
        synchronizationMillimeters = 0.0,
        rollingShutterMillimeters = 0.0,
        scaleMillimeters = 0.0,
        combinedSigmaMillimeters = 0.0,
        modeledP95Millimeters = 0.0,
        modeledP99Millimeters = 0.0,
    )
}
