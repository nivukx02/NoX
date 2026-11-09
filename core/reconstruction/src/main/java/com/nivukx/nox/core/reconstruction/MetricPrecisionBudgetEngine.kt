package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Conservative metrology model for production markerless 3D capture.
 *
 * The result is a modeled uncertainty budget, not ground-truth accuracy. It combines measurable
 * contributions from image reprojection, stereo geometry, depth sensing, camera pose, timing,
 * rolling shutter, calibration, and metric scale.
 */
class MetricPrecisionBudgetEngine {

    enum class Grade {
        REFERENCE_2MM,
        HIGH_PRECISION,
        PRODUCTION,
        DEGRADED,
        BLOCKED,
    }

    data class Input(
        val targetP95Millimeters: Double = 5.0,
        val referenceGradeEnabled: Boolean = false,
        val referenceMinimumCalibrationConfidence: Double = 0.92,
        val referenceMinimumSynchronizationConfidence: Double = 0.92,
        val referenceMinimumViewDiversity: Double = 0.65,
        val referenceMinimumCrossViewAgreement: Double = 0.75,
        val workingDistanceMeters: Double = 2.5,
        val cameraBaselineMeters: Double = 1.0,
        val effectiveFocalLengthPixels: Double = 1800.0,
        val reprojectionErrorPixels: Double = 1.5,
        /**
         * Optional independent observation-noise terms. They default to zero because many
         * pipelines already include these effects in reprojection residuals.
         */
        val subpixelLocalizationSigmaPixels: Double = 0.0,
        val photonShotNoiseSigmaPixels: Double = 0.0,
        val sensorReadNoiseSigmaPixels: Double = 0.0,
        val quantizationSigmaPixels: Double = 0.0,
        val triangulationParallaxDegrees: Double = 12.0,
        val depthUncertaintyMeters: Double = 0.004,
        val poseTranslationUncertaintyMeters: Double = 0.002,
        val poseAngularUncertaintyDegrees: Double = 0.05,
        val calibrationConfidence: Double = 0.95,
        val synchronizationConfidence: Double = 0.95,
        val viewDiversity: Double = 0.80,
        val crossViewAgreement: Double = 0.85,
        val temporalSkewMilliseconds: Double = 1.5,
        val timingAuthorityConfidence: Double = 0.0,
        val timingAuthorityResidualP95Milliseconds: Double = 0.0,
        val frameDurationMilliseconds: Double = 16.6667,
        val rollingShutterFraction: Double = 0.10,
        val motionSpeedMetersPerSecond: Double = 1.0,
        val motionAccelerationMetersPerSecondSquared: Double = 2.0,
        val metricScaleUncertaintyRatio: Double = 0.001,
        val reconstructionPriorUncertaintyMeters: Double = 0.001,
        val observationCount: Int = 4,
        val occlusionRatio: Double = 0.0,
        val identityContinuity: Double = 1.0,
        val thermalRisk: Double = 0.0,
        val memoryPressure: Double = 0.0,
        val surfaceTemporalStability: Double = 1.0,
    ) {
        init {
            require(targetP95Millimeters > 0.0 && targetP95Millimeters.isFinite())
            require(
                referenceMinimumCalibrationConfidence in 0.0..1.0 &&
                    referenceMinimumSynchronizationConfidence in 0.0..1.0 &&
                    referenceMinimumViewDiversity in 0.0..1.0 &&
                    referenceMinimumCrossViewAgreement in 0.0..1.0,
            )
            require(workingDistanceMeters > 0.0 && workingDistanceMeters.isFinite())
            require(cameraBaselineMeters > 0.0 && cameraBaselineMeters.isFinite())
            require(effectiveFocalLengthPixels > 0.0 && effectiveFocalLengthPixels.isFinite())
            require(reprojectionErrorPixels >= 0.0 && reprojectionErrorPixels.isFinite())
            require(
                triangulationParallaxDegrees > 0.0 &&
                    triangulationParallaxDegrees < 179.0,
            )
            listOf(
                subpixelLocalizationSigmaPixels,
                photonShotNoiseSigmaPixels,
                sensorReadNoiseSigmaPixels,
                quantizationSigmaPixels,
                depthUncertaintyMeters,
                poseTranslationUncertaintyMeters,
                poseAngularUncertaintyDegrees,
                temporalSkewMilliseconds,
                timingAuthorityResidualP95Milliseconds,
                frameDurationMilliseconds,
                rollingShutterFraction,
                motionSpeedMetersPerSecond,
                motionAccelerationMetersPerSecondSquared,
                metricScaleUncertaintyRatio,
                reconstructionPriorUncertaintyMeters,
            ).forEach { require(it.isFinite() && it >= 0.0) }
            listOf(
                calibrationConfidence,
                synchronizationConfidence,
                viewDiversity,
                crossViewAgreement,
                timingAuthorityConfidence,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            require(observationCount >= 1)
            listOf(
                occlusionRatio,
                identityContinuity,
                thermalRisk,
                memoryPressure,
                surfaceTemporalStability,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
        }
    }

    data class Contribution(
        val imageReprojectionMillimeters: Double,
        val observationSamplingMillimeters: Double,
        val stereoDepthMillimeters: Double,
        val depthSensorMillimeters: Double,
        val cameraPoseMillimeters: Double,
        val timingMillimeters: Double,
        val rollingShutterMillimeters: Double,
        val calibrationMillimeters: Double,
        val scaleMillimeters: Double,
        val reconstructionPriorMillimeters: Double,
    ) {
        val rootSumSquareMillimeters: Double
            get() = sqrt(
                imageReprojectionMillimeters * imageReprojectionMillimeters +
                    observationSamplingMillimeters * observationSamplingMillimeters +
                    stereoDepthMillimeters * stereoDepthMillimeters +
                    depthSensorMillimeters * depthSensorMillimeters +
                    cameraPoseMillimeters * cameraPoseMillimeters +
                    timingMillimeters * timingMillimeters +
                    rollingShutterMillimeters * rollingShutterMillimeters +
                    calibrationMillimeters * calibrationMillimeters +
                    scaleMillimeters * scaleMillimeters +
                    reconstructionPriorMillimeters * reconstructionPriorMillimeters,
            )
    }
    data class Decision(
        val grade: Grade,
        val modeledP95ErrorMillimeters: Double,
        val modeledP99ErrorMillimeters: Double,
        val contribution: Contribution,
        val feasibilityScore: Double,
        val referenceGradeFeasible: Boolean,
        val physicalEvidenceSufficient: Boolean,
        val hardBlock: Boolean,
        val warnings: List<String>,
        val blockers: List<String>,
        val requiredBaselineMetersForTarget: Double,
        val recommendedBaselineMeters: Double,
        val requiredFocalLengthPixelsForTarget: Double,
        val recommendedFocalLengthPixels: Double,
        val recommendedMinimumParallaxDegrees: Double,
    )

    fun evaluate(input: Input): Decision {
        val warnings = mutableListOf<String>()
        val blockers = mutableListOf<String>()

        if (input.cameraBaselineMeters < 0.25) warnings += "camera-baseline-small"
        if (input.triangulationParallaxDegrees < 5.0) warnings += "triangulation-parallax-low"
        if (input.reprojectionErrorPixels > 2.0) warnings += "reprojection-error-high"
        if (
            input.calibrationConfidence <
                input.referenceMinimumCalibrationConfidence
        ) {
            warnings += "calibration-below-reference-target"
        }
        if (
            input.synchronizationConfidence <
                input.referenceMinimumSynchronizationConfidence
        ) {
            warnings += "synchronization-below-reference-target"
        }
        if (input.viewDiversity < input.referenceMinimumViewDiversity) {
            warnings += "view-diversity-below-reference-target"
        }
        if (
            input.crossViewAgreement <
                input.referenceMinimumCrossViewAgreement
        ) {
            warnings += "cross-view-agreement-below-reference-target"
        }
        if (input.rollingShutterFraction > 0.20) {
            warnings += "rolling-shutter-contribution-high"
        }
        if (
            input.timingAuthorityConfidence < 0.80 &&
            input.referenceGradeEnabled
        ) {
            warnings += "timing-authority-quality-low"
        }
        if (
            input.timingAuthorityResidualP95Milliseconds > 2.0 &&
            input.referenceGradeEnabled
        ) {
            warnings += "timing-authority-residual-high"
        }

        val contribution = contribution(input)
        val sigmaMillimeters = contribution.rootSumSquareMillimeters

        val operationalInflation = (
            1.0 +
                input.occlusionRatio.coerceIn(0.0, 1.0) * 0.30 +
                (1.0 - input.identityContinuity).coerceIn(0.0, 1.0) * 0.35 +
                input.thermalRisk.coerceIn(0.0, 1.0) * 0.15 +
                input.memoryPressure.coerceIn(0.0, 1.0) * 0.10 +
                (1.0 - input.surfaceTemporalStability)
                    .coerceIn(0.0, 1.0) * 0.10
            ).coerceIn(1.0, 2.50)

        val modeledP95 =
            sigmaMillimeters * 1.96 * operationalInflation
        val modeledP99 =
            sigmaMillimeters * 2.576 * operationalInflation

        val physicalEvidenceSufficient =
            input.calibrationConfidence >=
                input.referenceMinimumCalibrationConfidence &&
                input.synchronizationConfidence >=
                input.referenceMinimumSynchronizationConfidence &&
                input.viewDiversity >=
                input.referenceMinimumViewDiversity &&
                input.crossViewAgreement >=
                input.referenceMinimumCrossViewAgreement &&
                input.identityContinuity >= 0.92 &&
                input.occlusionRatio <= 0.15 &&
                input.thermalRisk <= 0.65 &&
                input.memoryPressure <= 0.82 &&
                input.surfaceTemporalStability >= 0.90 &&
                input.observationCount >= 3

        val referenceGradeFeasible =
            input.referenceGradeEnabled &&
                modeledP95 <= input.targetP95Millimeters &&
                modeledP95 <= 2.0 &&
                physicalEvidenceSufficient

        if (input.referenceGradeEnabled && modeledP95 > 2.0) {
            blockers += "modeled-p95-exceeds-2mm"
        }
        if (input.referenceGradeEnabled && !physicalEvidenceSufficient) {
            blockers += "reference-grade-evidence-insufficient"
        }
        if (
            input.referenceGradeEnabled &&
            input.occlusionRatio > 0.25
        ) {
            blockers += "reference-grade-occlusion-too-high"
        }
        if (
            input.referenceGradeEnabled &&
            input.identityContinuity < 0.90
        ) {
            blockers += "reference-grade-identity-instability"
        }
        if (
            input.referenceGradeEnabled &&
            input.thermalRisk > 0.75
        ) {
            blockers += "reference-grade-thermal-risk"
        }
        if (
            input.referenceGradeEnabled &&
            input.memoryPressure > 0.90
        ) {
            blockers += "reference-grade-memory-pressure"
        }
        if (modeledP99 > max(input.targetP95Millimeters * 2.4, 3.0)) {
            warnings += "modeled-p99-tail-wide"
        }
        if (input.metricScaleUncertaintyRatio > 0.01) {
            warnings += "metric-scale-uncertainty-high"
        }
        if (input.occlusionRatio > 0.15) {
            warnings += "occlusion-inflation-high"
        }
        if (input.identityContinuity < 0.92) {
            warnings += "identity-continuity-below-reference-target"
        }
        if (input.thermalRisk > 0.65) {
            warnings += "thermal-risk-below-reference-headroom"
        }
        if (input.memoryPressure > 0.82) {
            warnings += "memory-pressure-below-reference-headroom"
        }
        if (input.surfaceTemporalStability < 0.90) {
            warnings += "surface-temporal-stability-below-reference-target"
        }

        val feasibilityScore = (
            geometricScore(input) * 0.34 +
                timingScore(input) * 0.18 +
                calibrationScore(input) * 0.18 +
                viewScore(input) * 0.16 +
                uncertaintyScore(modeledP95, input.targetP95Millimeters) * 0.14
            ).coerceIn(0.0, 1.0)

        val requiredBaseline = requiredBaselineMeters(input)
        val requiredFocal = requiredFocalLengthPixels(input)

        val grade = when {
            blockers.isNotEmpty() -> Grade.BLOCKED
            referenceGradeFeasible -> Grade.REFERENCE_2MM
            modeledP95 <= 5.0 && feasibilityScore >= 0.75 ->
                Grade.HIGH_PRECISION
            modeledP95 <= 12.0 && feasibilityScore >= 0.55 ->
                Grade.PRODUCTION
            else -> Grade.DEGRADED
        }

        return Decision(
            grade = grade,
            modeledP95ErrorMillimeters = modeledP95,
            modeledP99ErrorMillimeters = modeledP99,
            contribution = contribution,
            feasibilityScore = feasibilityScore,
            referenceGradeFeasible = referenceGradeFeasible,
            physicalEvidenceSufficient = physicalEvidenceSufficient,
            hardBlock = blockers.isNotEmpty(),
            warnings = warnings.distinct(),
            blockers = blockers.distinct(),
            requiredBaselineMetersForTarget = requiredBaseline,
            recommendedBaselineMeters =
                max(input.cameraBaselineMeters, requiredBaseline),
            requiredFocalLengthPixelsForTarget = requiredFocal,
            recommendedFocalLengthPixels =
                max(input.effectiveFocalLengthPixels, requiredFocal),
            recommendedMinimumParallaxDegrees =
                max(input.triangulationParallaxDegrees, 8.0),
        )
    }

    private fun contribution(input: Input): Contribution {
        val pixelAngularSigma =
            input.reprojectionErrorPixels /
                input.effectiveFocalLengthPixels
        val imageReprojectionMeters =
            input.workingDistanceMeters * pixelAngularSigma

        val observationSamplingSigmaPixels =
            sqrt(
                input.subpixelLocalizationSigmaPixels.squared() +
                    input.photonShotNoiseSigmaPixels.squared() +
                    input.sensorReadNoiseSigmaPixels.squared() +
                    input.quantizationSigmaPixels.squared(),
            )
        val observationSamplingMeters =
            input.workingDistanceMeters *
                observationSamplingSigmaPixels /
                input.effectiveFocalLengthPixels

        val parallaxRadians =
            Math.toRadians(
                input.triangulationParallaxDegrees
                    .coerceIn(0.1, 178.9),
            )

        val baselineDepthSigma =
            input.workingDistanceMeters *
                input.workingDistanceMeters *
                sqrt(2.0) *
                input.reprojectionErrorPixels /
                (
                    input.effectiveFocalLengthPixels *
                        input.cameraBaselineMeters
                )

        val angularDepthSigma =
            input.workingDistanceMeters *
                pixelAngularSigma /
                max(sin(parallaxRadians), 1e-4)

        val stereoDepthMeters =
            max(baselineDepthSigma, angularDepthSigma)

        val poseAngularMeters =
            input.workingDistanceMeters *
                Math.toRadians(input.poseAngularUncertaintyDegrees)

        val cameraPoseMeters = sqrt(
            input.poseTranslationUncertaintyMeters.squared() +
                poseAngularMeters.squared(),
        )

        val measuredResidualSeconds =
            input.timingAuthorityResidualP95Milliseconds
                .coerceAtLeast(0.0) /
                1000.0

        val authorityConfidence =
            input.timingAuthorityConfidence
                .coerceIn(0.0, 1.0)

        val temporalSeconds =
            max(
                input.temporalSkewMilliseconds / 1000.0,
                measuredResidualSeconds,
            )

        /*
         * When the temporal authority is weak, preserve a conservative fraction of the nominal
         * frame-skew model as an uncertainty floor. When measured authority is strong, the measured
         * residual dominates instead of a generic skew heuristic.
         */
        val timingSeconds =
            if (authorityConfidence >= 0.80) {
                temporalSeconds
            } else {
                max(
                    temporalSeconds,
                    (
                        input.frameDurationMilliseconds *
                            (1.0 -
                                authorityConfidence)
                    ) /
                        1000.0 *
                        0.20,
                )
            }

        val timingMeters =
            input.motionSpeedMetersPerSecond *
                timingSeconds +
                0.5 *
                input.motionAccelerationMetersPerSecondSquared *
                timingSeconds.squared()

        val rollingSeconds =
            input.frameDurationMilliseconds / 1000.0 *
                input.rollingShutterFraction

        val rollingMeters =
            input.motionSpeedMetersPerSecond *
                rollingSeconds +
                0.5 *
                input.motionAccelerationMetersPerSecondSquared *
                rollingSeconds.squared()

        val calibrationPrior =
            input.workingDistanceMeters *
                (1.0 - input.calibrationConfidence) *
                0.002

        val scaleMeters =
            input.workingDistanceMeters *
                input.metricScaleUncertaintyRatio

        return Contribution(
            imageReprojectionMillimeters =
                imageReprojectionMeters.toMillimeters(),
            observationSamplingMillimeters =
                observationSamplingMeters.toMillimeters(),
            stereoDepthMillimeters =
                stereoDepthMeters.toMillimeters(),
            depthSensorMillimeters =
                input.depthUncertaintyMeters.toMillimeters(),
            cameraPoseMillimeters =
                cameraPoseMeters.toMillimeters(),
            timingMillimeters =
                timingMeters.toMillimeters(),
            rollingShutterMillimeters =
                rollingMeters.toMillimeters(),
            calibrationMillimeters =
                max(
                    calibrationPrior,
                    input.reconstructionPriorUncertaintyMeters * 0.5,
                ).toMillimeters(),
            scaleMillimeters =
                scaleMeters.toMillimeters(),
            reconstructionPriorMillimeters =
                input.reconstructionPriorUncertaintyMeters.toMillimeters(),
        )
    }

    private fun requiredBaselineMeters(input: Input): Double {
        val operationalInflation = (
            1.0 +
                input.occlusionRatio * 0.30 +
                (1.0 - input.identityContinuity) * 0.35 +
                input.thermalRisk * 0.15 +
                input.memoryPressure * 0.10 +
                (1.0 - input.surfaceTemporalStability) * 0.10
            ).coerceIn(1.0, 2.50)
        val targetSigmaMeters =
            (2.0 / 1.96 / 1000.0) /
                operationalInflation
        val current = contribution(input)
        val fixedSigmaMeters = sqrt(
            current.imageReprojectionMillimeters * current.imageReprojectionMillimeters +
                current.observationSamplingMillimeters * current.observationSamplingMillimeters +
                current.depthSensorMillimeters * current.depthSensorMillimeters +
                current.cameraPoseMillimeters * current.cameraPoseMillimeters +
                current.timingMillimeters * current.timingMillimeters +
                current.rollingShutterMillimeters * current.rollingShutterMillimeters +
                current.calibrationMillimeters * current.calibrationMillimeters +
                current.scaleMillimeters * current.scaleMillimeters +
                current.reconstructionPriorMillimeters * current.reconstructionPriorMillimeters,
        ) / 1000.0
        val varianceBudget =
            targetSigmaMeters.squared() -
                fixedSigmaMeters.squared()

        if (varianceBudget <= 0.0) return Double.POSITIVE_INFINITY

        val numerator =
            input.workingDistanceMeters.squared() *
                sqrt(2.0) *
                input.reprojectionErrorPixels /
                input.effectiveFocalLengthPixels

        return numerator / sqrt(varianceBudget)
    }

    private fun requiredFocalLengthPixels(input: Input): Double {
        val operationalInflation = (
            1.0 +
                input.occlusionRatio * 0.30 +
                (1.0 - input.identityContinuity) * 0.35 +
                input.thermalRisk * 0.15 +
                input.memoryPressure * 0.10 +
                (1.0 - input.surfaceTemporalStability) * 0.10
            ).coerceIn(1.0, 2.50)
        val targetSigmaMeters =
            (2.0 / 1.96 / 1000.0) /
                operationalInflation
        val current = contribution(input)
        val fixedMeters = sqrt(
            current.stereoDepthMillimeters * current.stereoDepthMillimeters +
                current.observationSamplingMillimeters * current.observationSamplingMillimeters +
                current.depthSensorMillimeters * current.depthSensorMillimeters +
                current.cameraPoseMillimeters * current.cameraPoseMillimeters +
                current.timingMillimeters * current.timingMillimeters +
                current.rollingShutterMillimeters * current.rollingShutterMillimeters +
                current.calibrationMillimeters * current.calibrationMillimeters +
                current.scaleMillimeters * current.scaleMillimeters +
                current.reconstructionPriorMillimeters * current.reconstructionPriorMillimeters,
        ) / 1000.0
        val budget = (
            targetSigmaMeters.squared() -
                fixedMeters.squared()
        ).coerceAtLeast(1e-12)
        val angularBudget =
            sqrt(budget) /
                input.workingDistanceMeters
        return input.reprojectionErrorPixels /
            angularBudget.coerceAtLeast(1e-12)
    }

    private fun geometricScore(input: Input): Double =
        (
            input.triangulationParallaxDegrees / 20.0
        ).coerceIn(0.0, 1.0) * 0.45 +
            (
                input.cameraBaselineMeters / 2.0
            ).coerceIn(0.0, 1.0) * 0.35 +
            (
                input.effectiveFocalLengthPixels / 2500.0
            ).coerceIn(0.0, 1.0) * 0.20

    private fun timingScore(input: Input): Double =
        (
            1.0 -
                (
                    input.temporalSkewMilliseconds / 10.0
                ).coerceIn(0.0, 1.0) * 0.65 -
                input.rollingShutterFraction * 0.35
        ).coerceIn(0.0, 1.0)

    private fun calibrationScore(input: Input): Double =
        (
            input.calibrationConfidence * 0.55 +
                input.synchronizationConfidence * 0.45
        ).coerceIn(0.0, 1.0)

    private fun viewScore(input: Input): Double =
        (
            input.viewDiversity * 0.55 +
                input.crossViewAgreement * 0.45
        ).coerceIn(0.0, 1.0)

    private fun uncertaintyScore(
        modeledP95: Double,
        target: Double,
    ): Double =
        (
            1.0 -
                modeledP95 / max(target, 0.1)
        ).coerceIn(0.0, 1.0)

    private fun Double.squared(): Double = this * this
    private fun Double.toMillimeters(): Double = this * 1000.0
}

object ReferencePrecisionTargets {
    const val TWO_MILLIMETER_P95 = 2.0
    const val FIVE_MILLIMETER_P95 = 5.0
    const val MIN_REFERENCE_PARALLAX_DEGREES = 8.0
    const val MIN_REFERENCE_CALIBRATION_CONFIDENCE = 0.92
    const val MIN_REFERENCE_SYNC_CONFIDENCE = 0.92
}
