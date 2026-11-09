package com.nivukx.nox.core.model

import kotlin.math.roundToInt

enum class CaptureWorkflowPreset(
    val label: String,
    val description: String,
) {
    CALIBRATION(
        label = "Calibration",
        description = "Maximize rig observability, synchronization and metric scale evidence.",
    ),
    PRECISION_CAPTURE(
        label = "Precision Capture",
        description = "Prioritize dense geometry, micro-motion and strict temporal continuity.",
    ),
    INTERACTION_CAPTURE(
        label = "Interaction",
        description = "Prioritize occlusion recovery, identity persistence and contact constraints.",
    ),
    RECONSTRUCTION_QA(
        label = "Reconstruction QA",
        description = "Prioritize uncertainty visibility, topology integrity and evidence gating.",
    ),
    REFERENCE_QUALIFICATION(
        label = "Reference Qualification",
        description = "Require the strictest synchronization, geometry and measurement evidence before qualification.",
    ),
    FRONTIER_50_ACTOR(
        label = "Frontier 50-Actor",
        description = "Maximum-fidelity crowd capture with strict timing, identity, dense geometry and empirical precision gates.",
    ),
}


data class HighEndMotionCaptureSettings(
    val profileId: String = ProductionCaptureProfile.default.id,
    val cameraResolutionPreset: CameraResolutionPreset = CameraResolutionPreset.AUTO,
    val cameraDynamicRangeMode: CameraDynamicRangeMode =
        CameraDynamicRangeMode.STANDARD,
    val cameraExposureMode: CameraExposureMode = CameraExposureMode.AUTO,
    val cameraAntibandingMode: CameraAntibandingMode =
        CameraAntibandingMode.AUTO,
    val cameraFocusMode: CameraFocusMode = CameraFocusMode.TRACKED_SUBJECT,
    val cameraManualExposureMicros: Long = 8333L,
    val cameraManualIso: Int = 800,
    val cameraExposureCompensationEv: Double = 0.0,
    val cameraAutoExposureLock: Boolean = false,
    val cameraAutoWhiteBalanceLock: Boolean = false,
    /** Android 16+ precise CCT white-balance controls. */
    val cameraPreciseColorTemperatureEnabled: Boolean = false,
    val cameraColorTemperatureKelvin: Int = 5000,
    val cameraColorTint: Int = 0,
    val cameraOisEnabled: Boolean = false,
    val cameraElectronicStabilizationEnabled: Boolean = false,
    val cameraPreserveSensorTimestamp: Boolean = true,
    val cameraRawSidecarEnabled: Boolean = false,
    val cameraDepthSidecarEnabled: Boolean = true,
    val cameraManualFocusDistanceDiopters: Double? = null,
    val targetFps: Int = 60,
    val inferenceFps: Int = 60,
    val motionSensitivity: Double = 0.65,
    val predictionHorizonMs: Double = 45.0,
    val occlusionPersistenceMs: Double = 180.0,
    val jitterSuppression: Double = 0.55,
    val reprojectionGatePx: Double = 3.0,
    val reconstructionUncertaintyMeters: Double = 0.08,
    val minimumParallaxDegrees: Double = 4.0,
    val preferredTemporalSkewMs: Double = 2.0,
    val maximumTemporalSkewMs: Double = 12.0,
    val maximumSynchronizationResidualP95Ms: Double = 12.0,
    val maximumSynchronizationOutlierRatio: Double = 0.20,
    val maximumRollingShutterReadoutMilliseconds: Double = 40.0,
    val minimumHandObjectContactConfidence: Double = 0.72,
    val maximumContactPenetrationMillimeters: Double = 15.0,
    val maximumSurfaceStrainP95Ratio: Double = 0.35,
    val minimumSurfaceTemporalStability: Double = 0.78,
    val minimumSurfaceTopologyContinuity: Double = 0.72,
    val maximumRuntimeFaultsPerWindow: Int = 4,
    val cameraMotionSuppression: Double = 0.7,
    val microMotionPreservation: Double = 0.8,
    val contactConstraintStrength: Double = 0.75,
    val faceEyeFusionWeight: Double = 0.75,
    val handGeometryStrictness: Double = 0.8,
    val denseMeshDetail: Double = 0.72,
    val personSegmentationEnabled: Boolean = true,
    val personSegmentationCadenceFrames: Int = 4,
    val personSegmentationPriority: Double = 0.72,
    val lightingTelemetryEnabled: Boolean = true,
    val diagnosticsOverlayEnabled: Boolean = true,
    val uncertaintyOverlayEnabled: Boolean = true,
    val cameraFrustaEnabled: Boolean = true,
    val preserveMicroMotion: Double = 0.88,
    val temporalContinuityPriority: Double = 0.86,
    val surfaceDetailPriority: Double = 0.78,
    val faceEyeDetailPriority: Double = 0.90,
    val handDetailPriority: Double = 0.82,
    val occlusionRecoveryPriority: Double = 0.88,
    val interactionContactPriority: Double = 0.84,
    val cameraDiversityPriority: Double = 0.86,
    val faceExpressionPriority: Double = 0.84,
    val handContactPriority: Double = 0.86,
    val identityPersistencePriority: Double = 0.92,
    val captureIntegrityStrictness: Double = 0.90,
    val interleavedTemporalFusionEnabled: Boolean = true,
    val interleavedTemporalWindowMillis: Double = 45.0,
    val interleavedPredictionHorizonMillis: Double = 18.0,
    val interleavedMaximumSampleSkewMillis: Double = 25.0,
    val interleavedOutlierThresholdPixels: Double = 24.0,
    val photometricExposureShiftSoftLimit: Double = 0.12,
    val photometricExposureShiftHardLimit: Double = 0.40,
    val photometricExposureReliabilityFloor: Double = 0.35,
    val temporalSupersamplingPriority: Double = 0.90,
    val temporalFusionPredictionBudgetMillis: Double = 80.0,
    val temporalFusionTrackRetentionMillis: Double = 2000.0,
    val temporalFusionHardOutlierGate: Double = 1.75,
    val temporalFusionHuberDelta: Double = 1.25,
    val temporalFusionConfidenceHalfLifeMillis: Double = 160.0,
    val temporalFusionMaxTracks: Int = 128,
    val temporalFusionMaxObservationsPerFrame: Int = 2048,
    val adaptiveQualityEnabled: Boolean = true,
    val qualityLockEnabled: Boolean = false,
    val targetMotionP95ErrorMillimeters: Double = 5.0,
    val maximumTemporalLagMilliseconds: Double = 25.0,
    val maximumDropoutRatio: Double = 0.02,
    val minimumCalibrationConfidence: Double = 0.75,
    val minimumViewDiversity: Double = 0.40,
    val minimumCrossViewAgreement: Double = 0.50,
    val referenceGradePrecisionEnabled: Boolean = false,
    val referenceGradeTargetP95ErrorMillimeters: Double = 2.0,
    val referenceGradeMinimumParallaxDegrees: Double = 8.0,
    val referenceGradeMinimumCalibrationConfidence: Double = 0.92,
    val referenceGradeMinimumSynchronizationConfidence: Double = 0.92,
    val referenceGradeMinimumViewDiversity: Double = 0.65,
    val referenceGradeMinimumCrossViewAgreement: Double = 0.75,
    val ultraPrecisionEnabled: Boolean = false,
    val ultraPrecisionQualificationMode: Boolean = false,
    val ultraPrecisionTargetP95ErrorMillimeters: Double = 0.75,
    val ultraPrecisionTargetP99ErrorMillimeters: Double = 1.75,
    val ultraPrecisionMinimumEvidenceSamples: Int = 600,
    val ultraPrecisionMinimumBaselineMeters: Double = 0.60,
    val ultraPrecisionMinimumParallaxDegrees: Double = 10.0,
    val ultraPrecisionMinimumCalibrationConfidence: Double = 0.95,
    val ultraPrecisionMinimumSynchronizationConfidence: Double = 0.95,
    val ultraPrecisionMinimumViewDiversity: Double = 0.75,
    val ultraPrecisionMinimumCrossViewAgreement: Double = 0.90,
    val ultraPrecisionMaximumOcclusionRatio: Double = 0.12,
    val ultraPrecisionMinimumIdentityContinuity: Double = 0.95,
    val ultraPrecisionMinimumSurfaceTemporalStability: Double = 0.94,
    val ultraPrecisionMaximumThermalRisk: Double = 0.60,
    val ultraPrecisionMaximumMemoryPressure: Double = 0.75,
    val ultraPrecisionMatchedReferenceRequired: Boolean = true,
    val denseLandmarkCoverageStrictness: Double = 0.84,
    val denseLandmarkUncertaintyStrictness: Double = 0.86,
    val denseLandmarkCrossViewStrictness: Double = 0.84,
    val denseLandmarkTemporalStrictness: Double = 0.82,
    val correlatedEvidenceEnabled: Boolean = true,
    val correlatedEvidenceQualificationMode: Boolean = false,
    val correlatedEvidenceMinimumCoverage: Double = 0.62,
    val correlatedEvidenceMaximumCommonModeRisk: Double = 0.55,
    val identityExchangeRiskEnabled: Boolean = true,
    val identityExchangeQualificationMode: Boolean = false,
    val identityExchangeHighRiskThreshold: Double = 0.58,
    val identityExchangeHoldThreshold: Double = 0.76,
    val identityExchangeHistoryFrames: Int = 10,

    // High-end crowd identity and adaptive authority controls.
    val adaptiveAuthorityDepthEnabled: Boolean = true,
    val maximumAdaptiveAuthorityBonus: Int = 4,
    val adaptiveAuthorityDifficultyThreshold: Double = 0.45,
    val adaptiveAuthorityPriorityThreshold: Double = 0.85,
    val maximumAdaptiveAuthorityCamerasPerActor: Int = 16,
    val crowdedSceneOverlapThreshold: Double = 0.35,
    val crowdedSceneMatchScoreBoost: Double = 0.10,

    // Distributed rig controls are deliberately explicit so the operator can tune the global
    // federation separately from per-device Camera2 concurrency.
    val distributedGlobalCameraEnvelope: Int = 128,
    val distributedAuthoritativeCameraBudget: Int = 32,
    val distributedMaximumShards: Int = 16,
    val distributedMinimumAuthorityHealth: Double = 0.35,
    val distributedMinimumAuthoritySynchronization: Double = 0.55,
    val distributedMaximumTemporalResidualMs: Double = 12.0,
    val distributedMaximumTemporalOutlierRatio: Double = 0.20,
    val distributedMinimumTransportHeadroom: Double = 0.45,
    val frontierWorldSpaceRobustIterations: Int = 5,
    val frontierWorldSpaceHuberResidualMillimeters: Double = 15.0,
    val frontierWorldSpaceMaximumConditionEstimate: Double = 1.0e10,
    val frontierSurfaceFusionEnabled: Boolean = true,
    val frontierSurfacePatchCapacity: Int = 750_000,
    val frontierSurfaceHistoryFrames: Int = 12,
    val frontierCalibrationGraphEnabled: Boolean = true,
    val frontierCalibrationMaximumCycleResidualMillimeters: Double = 20.0,
    val frontierCameraAuthorityEnabled: Boolean = true,
    val frontierAuthorityCameraBudget: Int = 32,
    val frontierWitnessCameraBudget: Int = 96,
    val frontierMaximumCameraFederation: Int = 128,
    val frontierMinimumObservedViews: Int = 3,
    val frontierMinimumObservedParallaxDegrees: Double = 8.0,
    val frontierPrecisionQualificationRequiresReference: Boolean = true,
    val maximumMotionBlurRiskForAuthority: Double = 0.48,
    val minimumEpipolarConsistencyForAuthority: Double = 0.50,
    val minimumMetricEvidenceAuthorityForAuthority: Double = 0.48,
    val minimumCaptureAuthorityScore: Double = 0.78,
    val precisionAuthorityMinimumScore: Double = 0.86,
    val precisionAuthorityMinimumSamples: Int = 600,
) {
    init {
        require(profileId.isNotBlank())
        require(cameraManualExposureMicros in 1L..500_000L)
        require(cameraManualIso in 25..12_800)
        require(cameraExposureCompensationEv in -8.0..8.0)
        require(cameraColorTemperatureKelvin in 2000..10000)
        require(cameraColorTint in -100..100)
        cameraManualFocusDistanceDiopters?.let {
            require(it.isFinite() && it >= 0.0)
        }
        require(targetFps in 24..240)
        require(inferenceFps in 15..240)
        require(motionSensitivity in 0.0..1.0)
        require(predictionHorizonMs in 0.0..250.0)
        require(occlusionPersistenceMs in 0.0..1000.0)
        require(jitterSuppression in 0.0..1.0)
        require(reprojectionGatePx in 0.25..20.0)
        require(reconstructionUncertaintyMeters in 0.001..1.0)
        require(minimumParallaxDegrees in 0.0..45.0)
        require(preferredTemporalSkewMs in 0.0..50.0)
        require(maximumTemporalSkewMs >= preferredTemporalSkewMs)
        require(maximumSynchronizationResidualP95Ms in 1.0..50.0)
        require(maximumSynchronizationOutlierRatio in 0.0..0.50)
        require(maximumRollingShutterReadoutMilliseconds in 1.0..80.0)
        require(minimumHandObjectContactConfidence in 0.0..1.0)
        require(maximumContactPenetrationMillimeters in 0.5..50.0)
        require(maximumSurfaceStrainP95Ratio in 0.05..1.0)
        require(minimumSurfaceTemporalStability in 0.0..1.0)
        require(minimumSurfaceTopologyContinuity in 0.0..1.0)
        require(maximumRuntimeFaultsPerWindow in 1..32)
        require(cameraMotionSuppression in 0.0..1.0)
        require(microMotionPreservation in 0.0..1.0)
        require(contactConstraintStrength in 0.0..1.0)
        require(faceEyeFusionWeight in 0.0..1.0)
        require(handGeometryStrictness in 0.0..1.0)
        require(denseMeshDetail in 0.0..1.0)
        require(personSegmentationCadenceFrames in 1..120)
        require(personSegmentationPriority in 0.0..1.0)
        require(preserveMicroMotion in 0.0..1.0)
        require(temporalContinuityPriority in 0.0..1.0)
        require(surfaceDetailPriority in 0.0..1.0)
        require(faceEyeDetailPriority in 0.0..1.0)
        require(handDetailPriority in 0.0..1.0)
        require(occlusionRecoveryPriority in 0.0..1.0)
        require(interactionContactPriority in 0.0..1.0)
        require(cameraDiversityPriority in 0.0..1.0)
        require(faceExpressionPriority in 0.0..1.0)
        require(handContactPriority in 0.0..1.0)
        require(identityPersistencePriority in 0.0..1.0)
        require(captureIntegrityStrictness in 0.0..1.0)
        require(interleavedTemporalWindowMillis in 8.0..250.0)
        require(interleavedPredictionHorizonMillis in 0.0..100.0)
        require(
            interleavedPredictionHorizonMillis <=
                interleavedTemporalWindowMillis,
        )
        require(interleavedMaximumSampleSkewMillis in 1.0..100.0)
        require(interleavedOutlierThresholdPixels in 2.0..128.0)
        require(photometricExposureShiftSoftLimit >= 0.0)
        require(
            photometricExposureShiftHardLimit >
                photometricExposureShiftSoftLimit,
        )
        require(
            photometricExposureReliabilityFloor in 0.0..1.0,
        )
        require(temporalSupersamplingPriority in 0.0..1.0)
        require(temporalFusionPredictionBudgetMillis in 8.0..250.0)
        require(
            temporalFusionTrackRetentionMillis >=
                temporalFusionPredictionBudgetMillis &&
                temporalFusionTrackRetentionMillis <= 10_000.0
        )
        require(temporalFusionHardOutlierGate in 1.0..8.0)
        require(temporalFusionHuberDelta in 0.25..4.0)
        require(temporalFusionConfidenceHalfLifeMillis in 40.0..1000.0)
        require(temporalFusionMaxTracks in 32..512)
        require(temporalFusionMaxObservationsPerFrame in 128..8192)
        require(targetMotionP95ErrorMillimeters in 0.1..50.0)
        require(maximumTemporalLagMilliseconds in 1.0..250.0)
        require(maximumDropoutRatio in 0.0..0.50)
        require(minimumCalibrationConfidence in 0.0..1.0)
        require(minimumViewDiversity in 0.0..1.0)
        require(minimumCrossViewAgreement in 0.0..1.0)
        require(referenceGradeTargetP95ErrorMillimeters in 0.1..2.0)
        require(referenceGradeMinimumParallaxDegrees in 2.0..45.0)
        require(referenceGradeMinimumCalibrationConfidence in 0.0..1.0)
        require(referenceGradeMinimumSynchronizationConfidence in 0.0..1.0)
        require(referenceGradeMinimumViewDiversity in 0.0..1.0)
        require(referenceGradeMinimumCrossViewAgreement in 0.0..1.0)
        require(ultraPrecisionTargetP95ErrorMillimeters in 0.05..5.0)
        require(ultraPrecisionTargetP99ErrorMillimeters >= ultraPrecisionTargetP95ErrorMillimeters)
        require(ultraPrecisionMinimumEvidenceSamples in 60..1_000_000)
        require(ultraPrecisionMinimumBaselineMeters > 0.0)
        require(ultraPrecisionMinimumParallaxDegrees in 1.0..89.0)
        require(ultraPrecisionMinimumCalibrationConfidence in 0.0..1.0)
        require(ultraPrecisionMinimumSynchronizationConfidence in 0.0..1.0)
        require(ultraPrecisionMinimumViewDiversity in 0.0..1.0)
        require(ultraPrecisionMinimumCrossViewAgreement in 0.0..1.0)
        require(ultraPrecisionMaximumOcclusionRatio in 0.0..1.0)
        require(ultraPrecisionMinimumIdentityContinuity in 0.0..1.0)
        require(ultraPrecisionMinimumSurfaceTemporalStability in 0.0..1.0)
        require(ultraPrecisionMaximumThermalRisk in 0.0..1.0)
        require(ultraPrecisionMaximumMemoryPressure in 0.0..1.0)
        require(denseLandmarkCoverageStrictness in 0.0..1.0)
        require(denseLandmarkUncertaintyStrictness in 0.0..1.0)
        require(denseLandmarkCrossViewStrictness in 0.0..1.0)
        require(denseLandmarkTemporalStrictness in 0.0..1.0)
        require(correlatedEvidenceMinimumCoverage in 0.0..1.0)
        require(correlatedEvidenceMaximumCommonModeRisk in 0.0..1.0)
        require(identityExchangeHighRiskThreshold in 0.0..1.0)
        require(identityExchangeHoldThreshold in identityExchangeHighRiskThreshold..1.0)
        require(identityExchangeHistoryFrames in 2..120)
        require(maximumAdaptiveAuthorityBonus in 0..8)
        require(adaptiveAuthorityDifficultyThreshold in 0.0..1.0)
        require(adaptiveAuthorityPriorityThreshold in 0.0..1.0)
        require(
            maximumAdaptiveAuthorityCamerasPerActor in 1..32,
        )
        require(crowdedSceneOverlapThreshold in 0.0..1.0)
        require(crowdedSceneMatchScoreBoost in 0.0..0.30)
        require(distributedGlobalCameraEnvelope in 2..128)
        require(distributedAuthoritativeCameraBudget in 2..32)
        require(
            distributedAuthoritativeCameraBudget <=
                distributedGlobalCameraEnvelope,
        )
        require(distributedMaximumShards in 1..16)
        require(distributedMinimumAuthorityHealth in 0.0..1.0)
        require(distributedMinimumAuthoritySynchronization in 0.0..1.0)
        require(distributedMaximumTemporalResidualMs in 1.0..50.0)
        require(distributedMaximumTemporalOutlierRatio in 0.0..1.0)
        require(distributedMinimumTransportHeadroom in 0.0..1.0)
        require(maximumMotionBlurRiskForAuthority in 0.0..1.0)
        require(minimumEpipolarConsistencyForAuthority in 0.0..1.0)
        require(minimumMetricEvidenceAuthorityForAuthority in 0.0..1.0)
        require(minimumCaptureAuthorityScore in 0.0..1.0)
        require(precisionAuthorityMinimumScore in 0.0..1.0)
        require(precisionAuthorityMinimumSamples in 1..1_000_000)
        require(frontierWorldSpaceRobustIterations in 1..8)
        require(frontierWorldSpaceHuberResidualMillimeters in 0.5..100.0)
        require(frontierWorldSpaceMaximumConditionEstimate > 1.0)
        require(frontierSurfacePatchCapacity in 1_024..750_000)
        require(frontierSurfaceHistoryFrames in 2..120)
        require(frontierCalibrationMaximumCycleResidualMillimeters in 1.0..100.0)
        require(frontierAuthorityCameraBudget in 2..32)
        require(frontierWitnessCameraBudget in 0..96)
        require(frontierMaximumCameraFederation in 2..128)
        require(frontierAuthorityCameraBudget <= frontierMaximumCameraFederation)
        require(frontierWitnessCameraBudget <= frontierMaximumCameraFederation - frontierAuthorityCameraBudget)
        require(frontierMinimumObservedViews in 2..16)
        require(frontierMinimumObservedParallaxDegrees in 1.0..89.0)
        require(!(adaptiveQualityEnabled && qualityLockEnabled)) {
            "Adaptive quality and quality lock are mutually exclusive"
        }
    }

    fun profile(): ProductionCaptureProfile =
        ProductionCaptureProfile.byId(profileId)

    fun applyWorkflowPreset(
        preset: CaptureWorkflowPreset,
    ): HighEndMotionCaptureSettings =
        when (preset) {
            CaptureWorkflowPreset.CALIBRATION ->
                copy(
                    cameraResolutionPreset =
                        CameraResolutionPreset.MAX_SUPPORTED,
                    cameraDynamicRangeMode =
                        CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                    cameraExposureMode =
                        CameraExposureMode.MANUAL,
                    cameraAntibandingMode =
                        CameraAntibandingMode.AUTO,
                    cameraManualExposureMicros = 2_000L,
                    cameraManualIso = 400,
                    cameraOisEnabled = false,
                    cameraElectronicStabilizationEnabled = false,
                    cameraPreserveSensorTimestamp = true,
                    targetFps = 60,
                    inferenceFps = 60,
                    motionSensitivity = 0.72,
                    predictionHorizonMs = 25.0,
                    jitterSuppression = 0.42,
                    minimumParallaxDegrees = 8.0,
                    preferredTemporalSkewMs = 1.0,
                    maximumTemporalSkewMs = 6.0,
                    maximumSynchronizationResidualP95Ms = 6.0,
                    maximumSynchronizationOutlierRatio = 0.08,
                    minimumCalibrationConfidence = 0.92,
                    minimumViewDiversity = 0.65,
                    minimumCrossViewAgreement = 0.78,
                    referenceGradePrecisionEnabled = true,
                    adaptiveQualityEnabled = false,
                    qualityLockEnabled = true,
                    cameraDiversityPriority = 0.96,
                    captureIntegrityStrictness = 0.98,
                    distributedGlobalCameraEnvelope = 128,
                    distributedAuthoritativeCameraBudget = 32,
                    distributedMaximumShards = 16,
                    distributedMinimumAuthorityHealth = 0.78,
                    distributedMinimumAuthoritySynchronization = 0.90,
                    distributedMaximumTemporalResidualMs = 6.0,
                    distributedMaximumTemporalOutlierRatio = 0.08,
                    distributedMinimumTransportHeadroom = 0.65,
                )

            CaptureWorkflowPreset.PRECISION_CAPTURE ->
                copy(
                    cameraResolutionPreset =
                        CameraResolutionPreset.MAX_SUPPORTED,
                    cameraDynamicRangeMode =
                        CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                    cameraExposureMode =
                        CameraExposureMode.MANUAL,
                    cameraAntibandingMode =
                        CameraAntibandingMode.AUTO,
                    cameraManualExposureMicros = 1_500L,
                    cameraManualIso = 400,
                    cameraOisEnabled = false,
                    cameraElectronicStabilizationEnabled = false,
                    cameraPreserveSensorTimestamp = true,
                    targetFps = 120,
                    inferenceFps = 120,
                    motionSensitivity = 0.90,
                    predictionHorizonMs = 18.0,
                    occlusionPersistenceMs = 120.0,
                    jitterSuppression = 0.28,
                    reprojectionGatePx = 1.2,
                    reconstructionUncertaintyMeters = 0.003,
                    minimumParallaxDegrees = 10.0,
                    preferredTemporalSkewMs = 0.75,
                    maximumTemporalSkewMs = 4.0,
                    maximumSynchronizationResidualP95Ms = 4.0,
                    maximumSynchronizationOutlierRatio = 0.05,
                    microMotionPreservation = 0.98,
                    preserveMicroMotion = 0.98,
                    temporalContinuityPriority = 0.97,
                    surfaceDetailPriority = 0.96,
                    faceEyeDetailPriority = 0.96,
                    handDetailPriority = 0.95,
                    occlusionRecoveryPriority = 0.96,
                    cameraDiversityPriority = 0.95,
                    captureIntegrityStrictness = 0.99,
                    denseMeshDetail = 0.96,
                    denseLandmarkCoverageStrictness = 0.95,
                    denseLandmarkUncertaintyStrictness = 0.95,
                    denseLandmarkCrossViewStrictness = 0.95,
                    denseLandmarkTemporalStrictness = 0.95,
                    distributedGlobalCameraEnvelope = 128,
                    distributedAuthoritativeCameraBudget = 32,
                    distributedMaximumShards = 16,
                    distributedMinimumAuthorityHealth = 0.70,
                    distributedMinimumAuthoritySynchronization = 0.85,
                    distributedMaximumTemporalResidualMs = 4.0,
                    distributedMaximumTemporalOutlierRatio = 0.05,
                    adaptiveQualityEnabled = false,
                    qualityLockEnabled = true,
                )

            CaptureWorkflowPreset.INTERACTION_CAPTURE ->
                copy(
                    targetFps = 90,
                    inferenceFps = 90,
                    motionSensitivity = 0.88,
                    predictionHorizonMs = 24.0,
                    occlusionPersistenceMs = 260.0,
                    minimumParallaxDegrees = 9.0,
                    minimumHandObjectContactConfidence = 0.86,
                    maximumContactPenetrationMillimeters = 5.0,
                    minimumSurfaceTemporalStability = 0.88,
                    minimumSurfaceTopologyContinuity = 0.86,
                    occlusionRecoveryPriority = 0.98,
                    interactionContactPriority = 0.98,
                    handContactPriority = 0.98,
                    identityPersistencePriority = 0.99,
                    cameraDiversityPriority = 0.96,
                    faceExpressionPriority = 0.92,
                    captureIntegrityStrictness = 0.98,
                    denseLandmarkCoverageStrictness = 0.94,
                    denseLandmarkUncertaintyStrictness = 0.92,
                    denseLandmarkCrossViewStrictness = 0.96,
                    denseLandmarkTemporalStrictness = 0.95,
                    adaptiveQualityEnabled = false,
                    qualityLockEnabled = true,
                )

            CaptureWorkflowPreset.FRONTIER_50_ACTOR ->
                copy(
                    cameraResolutionPreset =
                        CameraResolutionPreset.MAX_SUPPORTED,
                    cameraDynamicRangeMode =
                        CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                    cameraExposureMode = CameraExposureMode.MANUAL,
                    cameraManualExposureMicros = 1_500L,
                    cameraManualIso = 400,
                    cameraOisEnabled = false,
                    cameraElectronicStabilizationEnabled = false,
                    cameraPreserveSensorTimestamp = true,
                    cameraRawSidecarEnabled = true,
                    cameraDepthSidecarEnabled = true,
                    targetFps = 60,
                    inferenceFps = 60,
                    motionSensitivity = 0.98,
                    predictionHorizonMs = 18.0,
                    occlusionPersistenceMs = 120.0,
                    jitterSuppression = 0.22,
                    reprojectionGatePx = 1.0,
                    reconstructionUncertaintyMeters = 0.003,
                    minimumParallaxDegrees = 10.0,
                    preferredTemporalSkewMs = 0.75,
                    maximumTemporalSkewMs = 4.0,
                    maximumSynchronizationResidualP95Ms = 4.0,
                    maximumSynchronizationOutlierRatio = 0.05,
                    maximumRollingShutterReadoutMilliseconds = 20.0,
                    microMotionPreservation = 1.0,
                    preserveMicroMotion = 1.0,
                    temporalContinuityPriority = 0.98,
                    surfaceDetailPriority = 0.98,
                    faceEyeDetailPriority = 1.0,
                    handDetailPriority = 0.98,
                    occlusionRecoveryPriority = 0.99,
                    interactionContactPriority = 0.96,
                    cameraDiversityPriority = 0.99,
                    faceExpressionPriority = 0.96,
                    handContactPriority = 0.96,
                    identityPersistencePriority = 1.0,
                    captureIntegrityStrictness = 1.0,
                    denseMeshDetail = 0.99,
                    denseLandmarkCoverageStrictness = 0.99,
                    denseLandmarkUncertaintyStrictness = 0.99,
                    denseLandmarkCrossViewStrictness = 0.99,
                    denseLandmarkTemporalStrictness = 0.98,
                    personSegmentationCadenceFrames = 2,
                    personSegmentationPriority = 0.98,
                    correlatedEvidenceEnabled = true,
                    correlatedEvidenceQualificationMode = true,
                    correlatedEvidenceMinimumCoverage = 0.80,
                    identityExchangeRiskEnabled = true,
                    identityExchangeQualificationMode = true,
                    identityExchangeHighRiskThreshold = 0.40,
                    identityExchangeHoldThreshold = 0.70,
                    identityExchangeHistoryFrames = 24,
                    referenceGradePrecisionEnabled = true,
                    referenceGradeTargetP95ErrorMillimeters = 0.75,
                    referenceGradeMinimumParallaxDegrees = 10.0,
                    referenceGradeMinimumCalibrationConfidence = 0.97,
                    referenceGradeMinimumSynchronizationConfidence = 0.97,
                    referenceGradeMinimumViewDiversity = 0.85,
                    referenceGradeMinimumCrossViewAgreement = 0.95,
                    ultraPrecisionEnabled = true,
                    ultraPrecisionQualificationMode = true,
                    ultraPrecisionTargetP95ErrorMillimeters = 0.75,
                    ultraPrecisionTargetP99ErrorMillimeters = 1.50,
                    ultraPrecisionMinimumEvidenceSamples = 2000,
                    ultraPrecisionMinimumBaselineMeters = 0.75,
                    ultraPrecisionMinimumParallaxDegrees = 12.0,
                    ultraPrecisionMinimumCalibrationConfidence = 0.98,
                    ultraPrecisionMinimumSynchronizationConfidence = 0.98,
                    ultraPrecisionMinimumViewDiversity = 0.85,
                    ultraPrecisionMinimumCrossViewAgreement = 0.95,
                    ultraPrecisionMaximumOcclusionRatio = 0.10,
                    ultraPrecisionMinimumIdentityContinuity = 0.98,
                    ultraPrecisionMinimumSurfaceTemporalStability = 0.97,
                    ultraPrecisionMaximumThermalRisk = 0.50,
                    ultraPrecisionMaximumMemoryPressure = 0.65,
                    ultraPrecisionMatchedReferenceRequired = true,
                    distributedGlobalCameraEnvelope = 128,
                    distributedAuthoritativeCameraBudget = 32,
                    distributedMaximumShards = 16,
                    distributedMinimumAuthorityHealth = 0.70,
                    distributedMinimumAuthoritySynchronization = 0.90,
                    distributedMaximumTemporalResidualMs = 4.0,
                    distributedMaximumTemporalOutlierRatio = 0.05,
                    distributedMinimumTransportHeadroom = 0.65,
                    frontierWorldSpaceRobustIterations = 6,
                    frontierWorldSpaceHuberResidualMillimeters = 8.0,
                    frontierWorldSpaceMaximumConditionEstimate = 1.0e9,
                    frontierSurfaceFusionEnabled = true,
                    frontierSurfacePatchCapacity = 750_000,
                    frontierSurfaceHistoryFrames = 16,
                    frontierCalibrationGraphEnabled = true,
                    frontierCalibrationMaximumCycleResidualMillimeters = 12.0,
                    frontierCameraAuthorityEnabled = true,
                    frontierAuthorityCameraBudget = 32,
                    frontierWitnessCameraBudget = 96,
                    frontierMaximumCameraFederation = 128,
                    frontierMinimumObservedViews = 4,
                    frontierMinimumObservedParallaxDegrees = 12.0,
                    frontierPrecisionQualificationRequiresReference = true,
                    adaptiveQualityEnabled = false,
                    qualityLockEnabled = true,
                )

            CaptureWorkflowPreset.RECONSTRUCTION_QA ->
                copy(
                    targetFps = 60,
                    inferenceFps = 60,
                    motionSensitivity = 0.70,
                    predictionHorizonMs = 30.0,
                    jitterSuppression = 0.35,
                    reconstructionUncertaintyMeters = 0.005,
                    minimumParallaxDegrees = 7.5,
                    minimumSurfaceTemporalStability = 0.92,
                    minimumSurfaceTopologyContinuity = 0.90,
                    maximumSurfaceStrainP95Ratio = 0.20,
                    denseMeshDetail = 0.94,
                    surfaceDetailPriority = 0.96,
                    temporalContinuityPriority = 0.94,
                    captureIntegrityStrictness = 0.99,
                    lightingTelemetryEnabled = true,
                    diagnosticsOverlayEnabled = true,
                    uncertaintyOverlayEnabled = true,
                    cameraFrustaEnabled = true,
                    adaptiveQualityEnabled = false,
                    qualityLockEnabled = true,
                )

            CaptureWorkflowPreset.REFERENCE_QUALIFICATION ->
                copy(
                    cameraResolutionPreset =
                        CameraResolutionPreset.MAX_SUPPORTED,
                    cameraDynamicRangeMode =
                        CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                    cameraExposureMode =
                        CameraExposureMode.MANUAL,
                    cameraAntibandingMode =
                        CameraAntibandingMode.AUTO,
                    cameraManualExposureMicros = 1_000L,
                    cameraManualIso = 320,
                    cameraOisEnabled = false,
                    cameraElectronicStabilizationEnabled = false,
                    cameraPreserveSensorTimestamp = true,
                    targetFps = 120,
                    inferenceFps = 120,
                    motionSensitivity = 0.92,
                    predictionHorizonMs = 16.0,
                    occlusionPersistenceMs = 80.0,
                    jitterSuppression = 0.20,
                    reprojectionGatePx = 0.80,
                    reconstructionUncertaintyMeters = 0.001,
                    minimumParallaxDegrees = 10.0,
                    preferredTemporalSkewMs = 0.50,
                    maximumTemporalSkewMs = 2.50,
                    maximumSynchronizationResidualP95Ms = 2.50,
                    maximumSynchronizationOutlierRatio = 0.03,
                    maximumRollingShutterReadoutMilliseconds = 20.0,
                    minimumCalibrationConfidence = 0.95,
                    minimumViewDiversity = 0.75,
                    minimumCrossViewAgreement = 0.90,
                    referenceGradePrecisionEnabled = true,
                    referenceGradeTargetP95ErrorMillimeters = 1.0,
                    referenceGradeMinimumParallaxDegrees = 10.0,
                    referenceGradeMinimumCalibrationConfidence = 0.95,
                    referenceGradeMinimumSynchronizationConfidence = 0.95,
                    referenceGradeMinimumViewDiversity = 0.75,
                    referenceGradeMinimumCrossViewAgreement = 0.90,
                    ultraPrecisionEnabled = true,
                    ultraPrecisionQualificationMode = true,
                    ultraPrecisionTargetP95ErrorMillimeters = 0.75,
                    ultraPrecisionTargetP99ErrorMillimeters = 1.75,
                    ultraPrecisionMinimumEvidenceSamples = 600,
                    ultraPrecisionMinimumBaselineMeters = 0.60,
                    ultraPrecisionMinimumParallaxDegrees = 10.0,
                    ultraPrecisionMinimumCalibrationConfidence = 0.95,
                    ultraPrecisionMinimumSynchronizationConfidence = 0.95,
                    ultraPrecisionMinimumViewDiversity = 0.75,
                    ultraPrecisionMinimumCrossViewAgreement = 0.90,
                    ultraPrecisionMaximumOcclusionRatio = 0.12,
                    ultraPrecisionMinimumIdentityContinuity = 0.95,
                    ultraPrecisionMinimumSurfaceTemporalStability = 0.94,
                    ultraPrecisionMaximumThermalRisk = 0.60,
                    ultraPrecisionMaximumMemoryPressure = 0.75,
                    ultraPrecisionMatchedReferenceRequired = true,
                    microMotionPreservation = 1.0,
                    preserveMicroMotion = 1.0,
                    temporalContinuityPriority = 1.0,
                    surfaceDetailPriority = 1.0,
                    faceEyeDetailPriority = 1.0,
                    handDetailPriority = 1.0,
                    occlusionRecoveryPriority = 1.0,
                    interactionContactPriority = 0.98,
                    cameraDiversityPriority = 1.0,
                    faceExpressionPriority = 0.98,
                    handContactPriority = 0.98,
                    identityPersistencePriority = 1.0,
                    captureIntegrityStrictness = 1.0,
                    distributedGlobalCameraEnvelope = 128,
                    distributedAuthoritativeCameraBudget = 32,
                    distributedMaximumShards = 16,
                    distributedMinimumAuthorityHealth = 0.90,
                    distributedMinimumAuthoritySynchronization = 0.95,
                    distributedMaximumTemporalResidualMs = 2.5,
                    distributedMaximumTemporalOutlierRatio = 0.03,
                    frontierWorldSpaceRobustIterations = 6,
                    frontierWorldSpaceHuberResidualMillimeters = 6.0,
                    frontierWorldSpaceMaximumConditionEstimate = 5.0e8,
                    frontierSurfaceFusionEnabled = true,
                    frontierSurfacePatchCapacity = 750_000,
                    frontierSurfaceHistoryFrames = 20,
                    frontierCalibrationGraphEnabled = true,
                    frontierCalibrationMaximumCycleResidualMillimeters = 8.0,
                    frontierCameraAuthorityEnabled = true,
                    frontierAuthorityCameraBudget = 32,
                    frontierWitnessCameraBudget = 96,
                    frontierMaximumCameraFederation = 128,
                    frontierMinimumObservedViews = 4,
                    frontierMinimumObservedParallaxDegrees = 12.0,
                    frontierPrecisionQualificationRequiresReference = true,
                    denseMeshDetail = 1.0,
                    denseLandmarkCoverageStrictness = 1.0,
                    denseLandmarkUncertaintyStrictness = 1.0,
                    denseLandmarkCrossViewStrictness = 1.0,
                    denseLandmarkTemporalStrictness = 1.0,
                    adaptiveQualityEnabled = false,
                    qualityLockEnabled = true,
                )
        }


    fun withProfile(value: String): HighEndMotionCaptureSettings =
        copy(profileId = ProductionCaptureProfile.byId(value).id)

    fun withMicroMotionPreservation(value: Double): HighEndMotionCaptureSettings =
        copy(preserveMicroMotion = value.coerceIn(0.0, 1.0))

    fun withTemporalContinuity(value: Double): HighEndMotionCaptureSettings =
        copy(temporalContinuityPriority = value.coerceIn(0.0, 1.0))

    fun withSurfaceDetail(value: Double): HighEndMotionCaptureSettings =
        copy(surfaceDetailPriority = value.coerceIn(0.0, 1.0))

    fun withFaceEyeDetail(value: Double): HighEndMotionCaptureSettings =
        copy(faceEyeDetailPriority = value.coerceIn(0.0, 1.0))

    fun withHandDetail(value: Double): HighEndMotionCaptureSettings =
        copy(handDetailPriority = value.coerceIn(0.0, 1.0))

    fun withOcclusionRecovery(value: Double): HighEndMotionCaptureSettings =
        copy(occlusionRecoveryPriority = value.coerceIn(0.0, 1.0))

    fun withInteractionContact(value: Double): HighEndMotionCaptureSettings =
        copy(interactionContactPriority = value.coerceIn(0.0, 1.0))

    fun withCameraDiversity(value: Double): HighEndMotionCaptureSettings =
        copy(cameraDiversityPriority = value.coerceIn(0.0, 1.0))

    fun withFaceExpression(value: Double): HighEndMotionCaptureSettings =
        copy(faceExpressionPriority = value.coerceIn(0.0, 1.0))

    fun withHandContact(value: Double): HighEndMotionCaptureSettings =
        copy(handContactPriority = value.coerceIn(0.0, 1.0))

    fun withIdentityPersistence(value: Double): HighEndMotionCaptureSettings =
        copy(identityPersistencePriority = value.coerceIn(0.0, 1.0))

    fun withCaptureIntegrity(value: Double): HighEndMotionCaptureSettings =
        copy(captureIntegrityStrictness = value.coerceIn(0.0, 1.0))

    fun withTemporalSupersamplingEnabled(
        value: Boolean,
    ): HighEndMotionCaptureSettings =
        copy(interleavedTemporalFusionEnabled = value)

    fun withTemporalSupersamplingPriority(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(temporalSupersamplingPriority = value.coerceIn(0.0, 1.0))

    fun withTemporalFusionWindow(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(interleavedTemporalWindowMillis = value.coerceIn(8.0, 250.0))

    fun withPredictionHorizon(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(interleavedPredictionHorizonMillis = value.coerceIn(0.0, 100.0))

    fun withMaximumSampleSkew(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(interleavedMaximumSampleSkewMillis = value.coerceIn(1.0, 100.0))

    fun withOutlierEnvelopePixels(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(interleavedOutlierThresholdPixels = value.coerceIn(2.0, 128.0))

    fun withTemporalFusionPredictionBudget(
        value: Double,
    ): HighEndMotionCaptureSettings {
        val nextBudget = value.coerceIn(8.0, 250.0)
        return copy(
            temporalFusionPredictionBudgetMillis = nextBudget,
            temporalFusionTrackRetentionMillis =
                maxOf(
                    temporalFusionTrackRetentionMillis,
                    nextBudget,
                ),
        )
    }

    fun withTemporalFusionTrackRetention(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            temporalFusionTrackRetentionMillis =
                value.coerceIn(
                    temporalFusionPredictionBudgetMillis,
                    10_000.0,
                ),
        )

    fun withTemporalFusionOutlierGate(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(temporalFusionHardOutlierGate = value.coerceIn(1.0, 8.0))

    fun withTemporalFusionHuberDelta(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(temporalFusionHuberDelta = value.coerceIn(0.25, 4.0))

    fun withTemporalFusionConfidenceHalfLife(
        value: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            temporalFusionConfidenceHalfLifeMillis =
                value.coerceIn(40.0, 1000.0),
        )

    fun withTemporalFusionCapacity(
        maxTracks: Int,
        maxObservationsPerFrame: Int,
    ): HighEndMotionCaptureSettings =
        copy(
            temporalFusionMaxTracks =
                maxTracks.coerceIn(32, 512),
            temporalFusionMaxObservationsPerFrame =
                maxObservationsPerFrame.coerceIn(128, 8192),
        )

    fun withSynchronizationContract(
        residualP95Milliseconds: Double,
        outlierRatio: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            maximumSynchronizationResidualP95Ms =
                residualP95Milliseconds.coerceIn(1.0, 50.0),
            maximumSynchronizationOutlierRatio =
                outlierRatio.coerceIn(0.0, 0.50),
        )

    fun withTemporalInteractionContract(
        maximumRollingShutterReadoutMilliseconds: Double,
        minimumHandObjectContactConfidence: Double,
        maximumContactPenetrationMillimeters: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            maximumRollingShutterReadoutMilliseconds =
                maximumRollingShutterReadoutMilliseconds.coerceIn(1.0, 80.0),
            minimumHandObjectContactConfidence =
                minimumHandObjectContactConfidence.coerceIn(0.0, 1.0),
            maximumContactPenetrationMillimeters =
                maximumContactPenetrationMillimeters.coerceIn(0.5, 50.0),
        )

    fun withSurfaceContract(
        maximumStrainP95Ratio: Double,
        minimumTemporalStability: Double,
        minimumTopologyContinuity: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            maximumSurfaceStrainP95Ratio =
                maximumStrainP95Ratio.coerceIn(0.05, 1.0),
            minimumSurfaceTemporalStability =
                minimumTemporalStability.coerceIn(0.0, 1.0),
            minimumSurfaceTopologyContinuity =
                minimumTopologyContinuity.coerceIn(0.0, 1.0),
        )

    fun withRuntimeFaultBudget(
        maximumFaultsPerWindow: Int,
    ): HighEndMotionCaptureSettings =
        copy(
            maximumRuntimeFaultsPerWindow =
                maximumFaultsPerWindow.coerceIn(1, 32),
        )

    fun withPrecisionContract(
        p95Millimeters: Double,
        lagMilliseconds: Double,
        dropoutRatio: Double,
        calibrationConfidence: Double,
        viewDiversity: Double,
        crossViewAgreement: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            targetMotionP95ErrorMillimeters =
                p95Millimeters.coerceIn(0.1, 50.0),
            maximumTemporalLagMilliseconds =
                lagMilliseconds.coerceIn(1.0, 250.0),
            maximumDropoutRatio =
                dropoutRatio.coerceIn(0.0, 0.50),
            minimumCalibrationConfidence =
                calibrationConfidence.coerceIn(0.0, 1.0),
            minimumViewDiversity =
                viewDiversity.coerceIn(0.0, 1.0),
            minimumCrossViewAgreement =
                crossViewAgreement.coerceIn(0.0, 1.0),
        )

    fun withReferenceGradePrecision(
        enabled: Boolean,
    ): HighEndMotionCaptureSettings =
        copy(referenceGradePrecisionEnabled = enabled)

    fun withUltraPrecisionQualification(
        enabled: Boolean,
        qualificationMode: Boolean = ultraPrecisionQualificationMode,
    ): HighEndMotionCaptureSettings =
        copy(
            ultraPrecisionEnabled = enabled,
            ultraPrecisionQualificationMode =
                if (enabled) qualificationMode else false,
        )

    fun withUltraPrecisionContract(
        targetP95Millimeters: Double,
        minimumBaselineMeters: Double,
        minimumParallaxDegrees: Double,
        minimumCalibrationConfidence: Double,
        minimumSynchronizationConfidence: Double,
        minimumViewDiversity: Double,
        minimumCrossViewAgreement: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            ultraPrecisionTargetP95ErrorMillimeters =
                targetP95Millimeters.coerceIn(0.05, 5.0),
            ultraPrecisionMinimumBaselineMeters =
                minimumBaselineMeters.coerceAtLeast(0.05),
            ultraPrecisionMinimumParallaxDegrees =
                minimumParallaxDegrees.coerceIn(1.0, 89.0),
            ultraPrecisionMinimumCalibrationConfidence =
                minimumCalibrationConfidence.coerceIn(0.0, 1.0),
            ultraPrecisionMinimumSynchronizationConfidence =
                minimumSynchronizationConfidence.coerceIn(0.0, 1.0),
            ultraPrecisionMinimumViewDiversity =
                minimumViewDiversity.coerceIn(0.0, 1.0),
            ultraPrecisionMinimumCrossViewAgreement =
                minimumCrossViewAgreement.coerceIn(0.0, 1.0),
        )

    fun withDenseLandmarkContractStrictness(
        coverage: Double,
        uncertainty: Double,
        crossView: Double,
        temporal: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            denseLandmarkCoverageStrictness =
                coverage.coerceIn(0.0, 1.0),
            denseLandmarkUncertaintyStrictness =
                uncertainty.coerceIn(0.0, 1.0),
            denseLandmarkCrossViewStrictness =
                crossView.coerceIn(0.0, 1.0),
            denseLandmarkTemporalStrictness =
                temporal.coerceIn(0.0, 1.0),
        )

    fun withReferenceGradeContract(
        targetP95Millimeters: Double,
        minimumParallaxDegrees: Double,
        minimumCalibrationConfidence: Double,
        minimumSynchronizationConfidence: Double,
        minimumViewDiversity: Double,
        minimumCrossViewAgreement: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            referenceGradeTargetP95ErrorMillimeters =
                targetP95Millimeters.coerceIn(0.1, 2.0),
            referenceGradeMinimumParallaxDegrees =
                minimumParallaxDegrees.coerceIn(2.0, 45.0),
            referenceGradeMinimumCalibrationConfidence =
                minimumCalibrationConfidence.coerceIn(0.0, 1.0),
            referenceGradeMinimumSynchronizationConfidence =
                minimumSynchronizationConfidence.coerceIn(0.0, 1.0),
            referenceGradeMinimumViewDiversity =
                minimumViewDiversity.coerceIn(0.0, 1.0),
            referenceGradeMinimumCrossViewAgreement =
                minimumCrossViewAgreement.coerceIn(0.0, 1.0),
        )

    fun withFrontierWorldSpaceRobustness(
        iterations: Int,
        huberResidualMillimeters: Double,
        maximumConditionEstimate: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            frontierWorldSpaceRobustIterations = iterations.coerceIn(1, 8),
            frontierWorldSpaceHuberResidualMillimeters =
                huberResidualMillimeters.coerceIn(0.5, 100.0),
            frontierWorldSpaceMaximumConditionEstimate =
                maximumConditionEstimate.coerceAtLeast(1.01),
        )

    fun withFrontierSurfaceCapacity(
        enabled: Boolean,
        patchCapacity: Int,
        historyFrames: Int,
    ): HighEndMotionCaptureSettings =
        copy(
            frontierSurfaceFusionEnabled = enabled,
            frontierSurfacePatchCapacity =
                patchCapacity.coerceIn(1_024, 750_000),
            frontierSurfaceHistoryFrames =
                historyFrames.coerceIn(2, 120),
        )

    fun withFrontierCalibrationPolicy(
        graphEnabled: Boolean,
        maximumCycleResidualMillimeters: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            frontierCalibrationGraphEnabled = graphEnabled,
            frontierCalibrationMaximumCycleResidualMillimeters =
                maximumCycleResidualMillimeters.coerceIn(1.0, 100.0),
        )

    fun withFrontierCameraFederationPolicy(
        authorityBudget: Int,
        witnessBudget: Int,
        globalEnvelope: Int,
    ): HighEndMotionCaptureSettings {
        val envelope = globalEnvelope.coerceIn(2, 128)
        val authority = authorityBudget.coerceIn(2, minOf(32, envelope))
        val witness = witnessBudget.coerceIn(
            0,
            minOf(96, envelope - authority),
        )
        return copy(
            frontierCameraAuthorityEnabled = true,
            frontierAuthorityCameraBudget = authority,
            frontierWitnessCameraBudget = witness,
            frontierMaximumCameraFederation = envelope,
        )
    }

    fun withFrontierPrecisionQualification(
        requiresIndependentReference: Boolean,
        minimumViews: Int,
        minimumParallaxDegrees: Double,
    ): HighEndMotionCaptureSettings =
        copy(
            frontierPrecisionQualificationRequiresReference =
                requiresIndependentReference,
            frontierMinimumObservedViews =
                minimumViews.coerceIn(2, 16),
            frontierMinimumObservedParallaxDegrees =
                minimumParallaxDegrees.coerceIn(1.0, 89.0),
        )

    fun withQualityPolicy(
        adaptive: Boolean,
        locked: Boolean,
    ): HighEndMotionCaptureSettings =
        copy(
            adaptiveQualityEnabled = adaptive,
            qualityLockEnabled = locked,
        )


    fun withEvidenceAuthorityPolicy(
        maximumMotionBlurRisk: Double,
        minimumEpipolarConsistency: Double,
        minimumMetricEvidenceAuthority: Double,
        minimumAuthorityScore: Double,
        precisionMinimumScore: Double = precisionAuthorityMinimumScore,
        precisionMinimumSamples: Int = precisionAuthorityMinimumSamples,
    ): HighEndMotionCaptureSettings =
        copy(
            maximumMotionBlurRiskForAuthority =
                maximumMotionBlurRisk.coerceIn(0.0, 1.0),
            minimumEpipolarConsistencyForAuthority =
                minimumEpipolarConsistency.coerceIn(0.0, 1.0),
            minimumMetricEvidenceAuthorityForAuthority =
                minimumMetricEvidenceAuthority.coerceIn(0.0, 1.0),
            minimumCaptureAuthorityScore =
                minimumAuthorityScore.coerceIn(0.0, 1.0),
            precisionAuthorityMinimumScore =
                precisionMinimumScore.coerceIn(0.0, 1.0),
            precisionAuthorityMinimumSamples =
                precisionMinimumSamples.coerceIn(1, 1_000_000),
        )

    fun toMotionCaptureCustomization(): MotionCaptureCustomization =
        MotionCaptureCustomization(
            cameraResolutionPreset =
                cameraResolutionPreset,
            cameraDynamicRangeMode =
                cameraDynamicRangeMode,
            cameraExposureMode =
                cameraExposureMode,
            cameraAntibandingMode =
                cameraAntibandingMode,
            cameraFocusMode =
                cameraFocusMode,
            cameraManualExposureMicros =
                cameraManualExposureMicros,
            cameraManualIso =
                cameraManualIso,
            cameraExposureCompensationEv =
                cameraExposureCompensationEv,
            cameraAutoExposureLock =
                cameraAutoExposureLock,
            cameraAutoWhiteBalanceLock =
                cameraAutoWhiteBalanceLock,
            cameraPreciseColorTemperatureEnabled =
                cameraPreciseColorTemperatureEnabled,
            cameraColorTemperatureKelvin =
                cameraColorTemperatureKelvin,
            cameraColorTint =
                cameraColorTint,
            cameraOisEnabled =
                cameraOisEnabled,
            cameraElectronicStabilizationEnabled =
                cameraElectronicStabilizationEnabled,
            cameraPreserveSensorTimestamp =
                cameraPreserveSensorTimestamp,
            cameraRawSidecarEnabled =
                cameraRawSidecarEnabled,
            cameraDepthSidecarEnabled =
                cameraDepthSidecarEnabled,
            cameraManualFocusDistanceDiopters =
                cameraManualFocusDistanceDiopters,
            targetFps = targetFps,
            inferenceFps = inferenceFps,
            actorMicroMotionRetention = preserveMicroMotion,
            actorSubframeTimingStrictness =
                temporalContinuityPriority,
            actorOcclusionRecovery = occlusionRecoveryPriority,
            actorIdentityStrictness = identityPersistencePriority,
            crowdedSceneOverlapThreshold =
                crowdedSceneOverlapThreshold,
            crowdedSceneMatchScoreBoost =
                crowdedSceneMatchScoreBoost,
            actorCrossCameraAssociation = cameraDiversityPriority,
            bodyDetail = motionSensitivity,
            faceDetail = maxOf(
                faceEyeDetailPriority,
                faceExpressionPriority,
            ),
            eyeDetail = faceEyeDetailPriority,
            handDetail = maxOf(
                handDetailPriority,
                handContactPriority,
            ),
            contactDetail = interactionContactPriority,
            handLongitudinalAnatomyStrictness =
                handContactPriority,
            denseMeshDetail = surfaceDetailPriority,
            personSegmentationEnabled =
                personSegmentationEnabled,
            personSegmentationCadenceFrames =
                personSegmentationCadenceFrames,
            personSegmentationPriority =
                personSegmentationPriority,
            denseLandmarkCoverageStrictness =
                denseLandmarkCoverageStrictness,
            denseLandmarkUncertaintyStrictness =
                denseLandmarkUncertaintyStrictness,
            denseLandmarkCrossViewStrictness =
                denseLandmarkCrossViewStrictness,
            denseLandmarkTemporalStrictness =
                denseLandmarkTemporalStrictness,
            outputIntegrityStrictness = captureIntegrityStrictness,
            temporalSmoothing = jitterSuppression,
            microMotionPreservation = microMotionPreservation,
            interleavedTemporalFusionEnabled =
                interleavedTemporalFusionEnabled,
            interleavedTemporalWindowMillis =
                interleavedTemporalWindowMillis,
            interleavedPredictionHorizonMillis =
                interleavedPredictionHorizonMillis,
            interleavedMaximumSampleSkewMillis =
                interleavedMaximumSampleSkewMillis,
            interleavedOutlierThresholdPixels =
                interleavedOutlierThresholdPixels,
            photometricExposureShiftSoftLimit =
                photometricExposureShiftSoftLimit,
            photometricExposureShiftHardLimit =
                photometricExposureShiftHardLimit,
            photometricExposureReliabilityFloor =
                photometricExposureReliabilityFloor,
            temporalSupersamplingPriority =
                temporalSupersamplingPriority,
            maximumReprojectionErrorPx =
                reprojectionGatePx,
            maximumReconstructionUncertaintyMeters =
                reconstructionUncertaintyMeters,
            minimumParallaxDegrees =
                minimumParallaxDegrees,
            preferredTemporalSkewMillis =
                preferredTemporalSkewMs,
            maximumTemporalSkewMillis =
                maximumTemporalSkewMs,
            faceEyeFusionWeight =
                faceEyeFusionWeight,
            handGeometryStrictness =
                handGeometryStrictness,
            adaptiveDetectorIntervalFrames =
                maxOf(
                    1,
                    (targetFps.toDouble() /
                        inferenceFps.coerceAtLeast(1))
                        .roundToInt(),
                ),
            dynamicTopology = true,
            preserveMicroMotion = true,
            ultraPrecisionEnabled = ultraPrecisionEnabled,
            ultraPrecisionQualificationMode = ultraPrecisionQualificationMode,
            ultraPrecisionTargetP95ErrorMillimeters =
                ultraPrecisionTargetP95ErrorMillimeters,
            ultraPrecisionTargetP99ErrorMillimeters =
                ultraPrecisionTargetP99ErrorMillimeters,
            ultraPrecisionMinimumEvidenceSamples =
                ultraPrecisionMinimumEvidenceSamples,
            ultraPrecisionMinimumBaselineMeters =
                ultraPrecisionMinimumBaselineMeters,
            ultraPrecisionMinimumParallaxDegrees =
                ultraPrecisionMinimumParallaxDegrees,
            ultraPrecisionMinimumCalibrationConfidence =
                ultraPrecisionMinimumCalibrationConfidence,
            ultraPrecisionMinimumSynchronizationConfidence =
                ultraPrecisionMinimumSynchronizationConfidence,
            ultraPrecisionMinimumViewDiversity =
                ultraPrecisionMinimumViewDiversity,
            ultraPrecisionMinimumCrossViewAgreement =
                ultraPrecisionMinimumCrossViewAgreement,
            ultraPrecisionMaximumOcclusionRatio =
                ultraPrecisionMaximumOcclusionRatio,
            ultraPrecisionMinimumIdentityContinuity =
                ultraPrecisionMinimumIdentityContinuity,
            ultraPrecisionMinimumSurfaceTemporalStability =
                ultraPrecisionMinimumSurfaceTemporalStability,
            ultraPrecisionMaximumThermalRisk =
                ultraPrecisionMaximumThermalRisk,
            ultraPrecisionMaximumMemoryPressure =
                ultraPrecisionMaximumMemoryPressure,
            ultraPrecisionMatchedReferenceRequired =
                ultraPrecisionMatchedReferenceRequired,
            distributedGlobalCameraEnvelope =
                distributedGlobalCameraEnvelope,
            distributedAuthoritativeCameraBudget =
                distributedAuthoritativeCameraBudget,
            distributedMaximumShards =
                distributedMaximumShards,
            distributedMinimumAuthorityHealth =
                distributedMinimumAuthorityHealth,
            distributedMinimumAuthoritySynchronization =
                distributedMinimumAuthoritySynchronization,
            distributedMaximumTemporalResidualMs =
                distributedMaximumTemporalResidualMs,
            distributedMaximumTemporalOutlierRatio =
                distributedMaximumTemporalOutlierRatio,
            distributedMinimumTransportHeadroom =
                distributedMinimumTransportHeadroom,
            recordLightingState = lightingTelemetryEnabled,
            recordImuState = true,
            recordCameraTelemetry = diagnosticsOverlayEnabled,
            recordConfidence = true,
        )

    fun withTargetFps(value: Int): HighEndMotionCaptureSettings =
        copy(targetFps = value.coerceIn(24, 240))

    fun withInferenceFps(value: Int): HighEndMotionCaptureSettings =
        copy(inferenceFps = value.coerceIn(15, 240))
}
