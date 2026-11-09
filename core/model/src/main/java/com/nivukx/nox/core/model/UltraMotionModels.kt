package com.nivukx.nox.core.model

/**
 * High-end motion-capture contracts for the RGB-only NoX pipeline.
 *
 * The types in this file are intentionally provider-neutral. A detector can be MediaPipe/ML Kit,
 * TFLite, ONNX, vendor NPU, or a future model without forcing the reconstruction/session layers
 * to know how the observations were produced.
 */

enum class MotionCaptureMode {
    MONOCULAR_RGB,
    MULTI_VIEW_RGB,
    RGB_DEPTH_HYBRID,
    MULTI_VIEW_RGB_DEPTH,
}

enum class SubjectLockPhase {
    SEARCH,
    DETECT,
    LOCK,
    TRACK,
    RECHECK,
    LOST,
}

enum class InferenceStage {
    DETECTOR,
    TRACKER,
    FACE,
    EYES,
    HANDS,
    OBJECTS,
    DEPTH,
    TOPOLOGY,
    RECONSTRUCTION,
    TEMPORAL,
    QUALITY,
}

enum class ScaleSource {
    CALIBRATED_METRIC,
    DEPTH_SENSOR,
    MULTI_VIEW_TRIANGULATION,
    ANTHROPOMETRIC_PRIOR,
    ANTHROPOMETRIC_REFINED,
    SCREEN_SIZE_PRIOR,
    TEMPORAL_PROPAGATION,
    UNKNOWN,
}

enum class TopologyKind {
    HUMAN,
    QUADRUPED,
    BIPED,
    AVIAN,
    INSECT,
    ARACHNID,
    DEFORMABLE,
    RIGID_OBJECT,
    UNKNOWN,
}

enum class DepthRepresentation {
    METRIC_METERS,
    RELATIVE_ORDERED_DEPTH,
}

enum class ReconstructionConfidenceBand {
    EXCELLENT,
    GOOD,
    USABLE,
    DEGRADED,
    UNTRUSTED,
}

data class InferenceBudget(
    val detectorIntervalFrames: Int = 8,
    val depthIntervalFrames: Int = 2,
    val faceIntervalFrames: Int = 3,
    val eyeIntervalFrames: Int = 2,
    val handIntervalFrames: Int = 2,
    val objectIntervalFrames: Int = 4,
    val denseMeshIntervalFrames: Int = 6,
    val contactIntervalFrames: Int = 1,
    val topologyIntervalFrames: Int = 1,
    val reconstructionIntervalFrames: Int = 1,
    val maximumLatencyMillis: Double = 30.0,
    val maximumMemoryMegabytes: Double = 384.0,
    val minimumTrackingConfidence: Double = 0.30,
) {
    init {
        require(detectorIntervalFrames >= 1)
        require(depthIntervalFrames >= 1)
        require(faceIntervalFrames >= 1)
        require(eyeIntervalFrames >= 1)
        require(handIntervalFrames >= 1)
        require(objectIntervalFrames >= 1)
        require(denseMeshIntervalFrames >= 1)
        require(contactIntervalFrames >= 1)
        require(topologyIntervalFrames >= 1)
        require(reconstructionIntervalFrames >= 1)
        require(maximumLatencyMillis > 0.0)
        require(maximumMemoryMegabytes > 64.0)
        require(minimumTrackingConfidence in 0.0..1.0)
    }

    fun degraded(factor: Double): InferenceBudget {
        val f = factor.coerceIn(0.25, 1.0)
        return copy(
            detectorIntervalFrames = (detectorIntervalFrames / f).toInt().coerceAtLeast(1),
            depthIntervalFrames = (depthIntervalFrames / f).toInt().coerceAtLeast(1),
            faceIntervalFrames = (faceIntervalFrames / f).toInt().coerceAtLeast(1),
            eyeIntervalFrames = (eyeIntervalFrames / f).toInt().coerceAtLeast(1),
            handIntervalFrames = (handIntervalFrames / f).toInt().coerceAtLeast(1),
            objectIntervalFrames = (objectIntervalFrames / f).toInt().coerceAtLeast(1),
            denseMeshIntervalFrames = (denseMeshIntervalFrames / f).toInt().coerceAtLeast(1),
            contactIntervalFrames = (contactIntervalFrames / f).toInt().coerceAtLeast(1),
            maximumLatencyMillis = maximumLatencyMillis / f,
            minimumTrackingConfidence =
                (minimumTrackingConfidence + (1.0 - f) * 0.10).coerceAtMost(0.90),
        )
    }
}

data class ConfidenceBreakdown(
    val detector: Double = 0.0,
    val geometry: Double = 0.0,
    val temporal: Double = 0.0,
    val topology: Double = 0.0,
    val occlusion: Double = 0.0,
    val depth: Double = 0.0,
    val scale: Double = 0.0,
    val final: Double = 0.0,
) {
    init {
        require(detector in 0.0..1.0)
        require(geometry in 0.0..1.0)
        require(temporal in 0.0..1.0)
        require(topology in 0.0..1.0)
        require(occlusion in 0.0..1.0)
        require(depth in 0.0..1.0)
        require(scale in 0.0..1.0)
        require(final in 0.0..1.0)
    }

    companion object {
        fun weighted(
            detector: Double,
            geometry: Double,
            temporal: Double,
            topology: Double,
            occlusion: Double,
            depth: Double,
            scale: Double,
        ): ConfidenceBreakdown {
            val d = detector.coerceIn(0.0, 1.0)
            val g = geometry.coerceIn(0.0, 1.0)
            val t = temporal.coerceIn(0.0, 1.0)
            val top = topology.coerceIn(0.0, 1.0)
            val o = occlusion.coerceIn(0.0, 1.0)
            val dep = depth.coerceIn(0.0, 1.0)
            val s = scale.coerceIn(0.0, 1.0)
            val result =
                0.20 * d +
                    0.18 * g +
                    0.17 * t +
                    0.15 * top +
                    0.10 * o +
                    0.12 * dep +
                    0.08 * s
            return ConfidenceBreakdown(
                detector = d,
                geometry = g,
                temporal = t,
                topology = top,
                occlusion = o,
                depth = dep,
                scale = s,
                final = result.coerceIn(0.0, 1.0),
            )
        }
    }
}

data class MotionNode(
    val id: Int,
    val semanticName: String,
    val position2D: Vec2,
    val position3D: Vec3?,
    val velocity3D: Vec3 = Vec3(0.0, 0.0, 0.0),
    val acceleration3D: Vec3 = Vec3(0.0, 0.0, 0.0),
    val confidence: Double,
    val uncertaintyMeters: Double,
    val predicted: Boolean,
    val visible: Boolean,
)

data class MotionEdge(
    val a: Int,
    val b: Int,
    val preferredLengthMeters: Double,
    val minimumLengthMeters: Double,
    val maximumLengthMeters: Double,
    val stiffness: Double,
    val semantic: String = "constraint",
)

data class DynamicTopology(
    val topologyId: String,
    val kind: TopologyKind,
    val nodes: List<MotionNode>,
    val edges: List<MotionEdge>,
    val rootNodeId: Int?,
    val scaleSource: ScaleSource,
    val globalConfidence: Double,
    val temporalStability: Double,
    val topologyStability: Double,
    val timestampNanos: Long,
) {
    val nodeCount: Int
        get() = nodes.size

    val edgeCount: Int
        get() = edges.size

    val visibleNodeCount: Int
        get() = nodes.count { it.visible && it.confidence > 0.10 }

    val predictedNodeCount: Int
        get() = nodes.count { it.predicted }
}

data class MonocularDepthSample(
    val x: Double,
    val y: Double,
    val depthMeters: Double,
    val confidence: Double,
    val source: ScaleSource,
)

data class MonocularDepthFrame(
    val timestampNanos: Long,
    val width: Int,
    val height: Int,
    val samples: List<MonocularDepthSample>,
    val meanDepthMeters: Double,
    val medianDepthMeters: Double,
    val scaleSource: ScaleSource,
    val confidence: Double,
    val temporalStability: Double,
    val metric: Boolean,
    val representation: DepthRepresentation =
        if (metric) DepthRepresentation.METRIC_METERS
        else DepthRepresentation.RELATIVE_ORDERED_DEPTH,
) {
    init {
        require(width > 0)
        require(height > 0)
        require(meanDepthMeters.isFinite())
        require(medianDepthMeters.isFinite())
    }
}

data class MonocularReconstructionJoint(
    val landmarkId: Int,
    val semanticName: String,
    val positionCameraMeters: Vec3,
    val velocityCameraMetersPerSecond: Vec3,
    val accelerationCameraMetersPerSecond2: Vec3,
    val confidence: Double,
    val uncertaintyMeters: Double,
    val depthMeters: Double,
    val depthConfidence: Double,
    val predicted: Boolean,
)

data class MonocularReconstructionFrame(
    val timestampNanos: Long,
    val subjectType: SubjectType,
    val trackId: Long,
    val joints: List<MonocularReconstructionJoint>,
    val rootPositionMeters: Vec3,
    val globalScaleMeters: Double,
    val scaleSource: ScaleSource,
    val scaleConfidence: Double,
    val globalConfidence: Double,
    val reprojectionErrorPx: Double,
    val meanUncertaintyMeters: Double,
    val temporalStability: Double,
    val confidenceBand: ReconstructionConfidenceBand,
) {
    val jointCount: Int
        get() = joints.size

    val predictedJointCount: Int
        get() = joints.count { it.predicted }
}

enum class ActorDetailPriorityMode {
    BALANCED,
    MOTION_FIRST,
    VISIBILITY_FIRST,
    PROXIMITY_FIRST,
    IDENTITY_FIRST,
    UNIFORM,
}

data class MotionInferenceDecision(
    val timestampNanos: Long,
    val subjectPhase: SubjectLockPhase,
    val activeTrackId: Long?,
    val dominantSubjectType: SubjectType,
    val detectorRequired: Boolean,
    val depthRequired: Boolean,
    val topologyRequired: Boolean,
    val reconstructionRequired: Boolean,
    val reason: String,
    val confidence: Double,
    val budget: InferenceBudget,
    val viewRefinementRequired: Boolean = false,
    val viewRefinementScore: Double = 0.0,
    val viewRefinementReason: String = "not_evaluated",
    val viewRefinementForced: Boolean = false,
    val identityContinuityScore: Double = 0.0,
    val identitySwapRisk: Double = 0.0,
    val identityAction: String = "SEARCH",
    val identityReason: String = "not_evaluated",
    val identitySwitchEvidenceFrames: Int = 0,
)

data class MotionPipelineTelemetry(
    val timestampNanos: Long,
    val captureFps: Double,
    val inferenceFps: Double,
    val detectorLatencyMs: Double,
    val depthLatencyMs: Double,
    val reconstructionLatencyMs: Double,
    val totalLatencyMs: Double,
    val droppedFrames: Long,
    val trackedSubjects: Int,
    val visibleJoints: Int,
    val predictedJoints: Int,
    val meanConfidence: Double,
    val temporalStability: Double,
    val thermalScale: Double,
    val memoryScale: Double,
    val gpuScale: Double,
    val npuScale: Double,
    val currentMode: MotionCaptureMode,
    val warnings: List<String>,
)

class MotionCaptureCustomization(
    val mode: MotionCaptureMode = MotionCaptureMode.MONOCULAR_RGB,
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
    val cameraManualFocusDistanceDiopters: Double? = null,
    val cameraPreciseColorTemperatureEnabled: Boolean = false,
    val cameraColorTemperatureKelvin: Int = 5000,
    val cameraColorTint: Int = 0,

    val cameraOisEnabled: Boolean = false,
    val cameraElectronicStabilizationEnabled: Boolean = false,
    val liveMotionPreviewEnabled: Boolean = false,
    val liveMotionPreviewGpuMode: MotionFieldGpuMode = MotionFieldGpuMode.AUTO,
    val liveMotionOverlayBody: Boolean = true,
    val liveMotionOverlayFace: Boolean = true,
    val liveMotionOverlayEyes: Boolean = true,
    val liveMotionOverlayDepth: Boolean = false,
    val liveMotionOverlayMicroMotion: Boolean = true,
    val liveMotionOverlayVelocity: Boolean = true,
    val liveMotionOverlayUncertainty: Boolean = true,
    val liveMotionOverlayActorIds: Boolean = true,
    val liveMotionPreviewPredictionHorizonMillis: Double = 18.0,
    val liveMotionPreviewMaxStaleMillis: Double = 120.0,
    val liveMotionPreviewDetail: Double = 0.92,
    val captureEvidenceMode: CaptureEvidenceMode = CaptureEvidenceMode.LIVE_MOTION_ONLY,
    val recordedVideoEvidenceProfile: RecordedVideoEvidenceProfile? = null,
    val postCaptureReconstructionPolicy: PostCaptureReconstructionPolicy =
        PostCaptureReconstructionPolicy(),
    val postCaptureAutoReconstructionEnabled: Boolean = true,
    val multiPersonCapacityPolicy: MultiPersonCapacityPolicy =
        MultiPersonCapacityPolicy.fiftyPersonProduction(),
    val cameraRawSidecarEnabled: Boolean = false,
    val cameraDepthSidecarEnabled: Boolean = true,
    val cameraPreserveSensorTimestamp: Boolean = true,
    val targetFps: Int = 60,
    val inferenceFps: Int = 30,
    val maxSubjects: Int = 50,
    val actorDetailPriorityMode: ActorDetailPriorityMode =
        ActorDetailPriorityMode.BALANCED,
    val fullDetailActorBudget: Int = 50,
    val faceDetailActorBudget: Int = 50,
    val eyeDetailActorBudget: Int = 50,
    val handDetailActorBudget: Int = 50,
    val denseSurfaceActorBudget: Int = 50,
    val priorityDetailRetention: Double = 0.94,
    val actorIdentityStrictness: Double = 0.90,
    val actorOcclusionRecovery: Double = 0.86,
    val crowdedSceneOverlapThreshold: Double = 0.35,
    val crowdedSceneMatchScoreBoost: Double = 0.10,
    val actorCrossCameraAssociation: Double = 0.88,
    val actorMicroMotionSensitivity: Double = 0.92,
    val motionFrequencyBandSensitivity: Double = 0.82,
    val actorMicroMotionRetention: Double = 0.94,
    val actorSubframeTimingStrictness: Double = 0.92,
    val bodyDetail: Double = 0.90,
    val faceDetail: Double = 0.95,
    val eyeDetail: Double = 0.92,
    val handDetail: Double = 0.92,
    val depthDetail: Double = 0.88,
    val objectDetail: Double = 0.80,
    val denseMeshDetail: Double = 0.74,
    val personSegmentationEnabled: Boolean = true,
    val personSegmentationCadenceFrames: Int = 4,
    val personSegmentationPriority: Double = 0.72,
    val contactDetail: Double = 0.76,
    val worldConsistencyDetail: Double = 0.86,
    val topologyConstraintStrength: Double = 0.84,
    val multiViewFusionWeight: Double = 0.82,
    val multiViewEvidenceAdaptivity: Double = 0.84,
    val reconstructionEvidenceFusionStrength: Double = 0.92,
    val reconstructionEvidenceDisagreementToleranceMeters: Double = 0.12,
    val reconstructionEvidenceMinimumSourceDiversity: Double = 0.20,
    val adaptiveDetectorIntervalFrames: Int = 12,
    val maximumOcclusionPredictionFrames: Int = 8,
    val temporalSmoothing: Double = 0.64,
    val occlusionPersistence: Double = 0.78,
    val microMotionPreservation: Double = 0.88,
    val cameraMotionSuppression: Double = 0.72,
    val faceEyeFusionWeight: Double = 0.62,
    val faceEyeFusionStrictness: Double = 0.84,
    val faceEyeIpdDriftToleranceMillimeters: Double = 12.0,
    val eyeTemporalStrictness: Double = 0.78,
    val handTemporalStrictness: Double = 0.78,
    val handGeometryStrictness: Double = 0.84,
    val handLongitudinalAnatomyStrictness: Double = 0.84,
    val temporalAlignmentStrictness: Double = 0.78,
    val subframeAlignmentStrictness: Double = 0.82,
    val interleavedTemporalFusionEnabled: Boolean = true,
    val interleavedTemporalWindowMillis: Double = 45.0,
    val interleavedPredictionHorizonMillis: Double = 18.0,
    val interleavedMaximumSampleSkewMillis: Double = 25.0,
    val interleavedOutlierThresholdPixels: Double = 24.0,
    val photometricExposureShiftSoftLimit: Double = 0.12,
    val photometricExposureShiftHardLimit: Double = 0.40,
    val photometricExposureReliabilityFloor: Double = 0.35,
    val temporalSupersamplingPriority: Double = 0.90,
    val rollingShutterCompensation: Double = 0.88,
    val outputIntegrityStrictness: Double = 0.78,
    val physicsConstraintStrictness: Double = 0.84,
    val global4DConsistencyStrictness: Double = 0.86,
    val surfaceEvidenceStrictness: Double = 0.84,
    val denseLandmarkCoverageStrictness: Double = 0.84,
    val denseLandmarkUncertaintyStrictness: Double = 0.86,
    val denseLandmarkCrossViewStrictness: Double = 0.84,
    val denseLandmarkTemporalStrictness: Double = 0.82,
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
    val minimumParallaxDegrees: Double = 0.55,
    val preferredTemporalSkewMillis: Double = 5.0,
    val maximumTemporalSkewMillis: Double = 20.0,
    val maximumReprojectionErrorPx: Double = 8.0,
    val maximumReconstructionUncertaintyMeters: Double = 0.75,
    val posePriorStrength: Double = 0.84,
    val footContactStrictness: Double = 0.80,
    val headPoseStrictness: Double = 0.86,
    val gazePriority: Double = 0.72,
    val depthTemporalFusion: Double = 0.86,
    val cameraDriftResponse: Double = 0.82,
    val thermalQualityFloor: Double = 0.55,
    val networkJitterToleranceMillis: Double = 18.0,
    val meshAdaptiveDetail: Double = 0.82,
    val retargetingStrictness: Double = 0.86,
    val scalePreservation: Double = 0.94,
    val occlusionRecoveryConfidence: Double = 0.82,
    val landmarkConsistencyStrictness: Double = 0.86,
    val semanticMotionSensitivity: Double = 0.82,
    val motionThresholdScale: Double = 1.0,
    val rootMotionLock: Double = 0.78,
    val dynamicTopology: Boolean = true,
    val preserveMicroMotion: Boolean = true,
    val recordLightingState: Boolean = true,
    val recordImuState: Boolean = true,
    val recordCameraTelemetry: Boolean = true,
    val recordDepthMetadata: Boolean = true,
    val recordFaceEyeFusion: Boolean = true,
    val recordHand3D: Boolean = true,
    val recordDenseMesh: Boolean = true,
    val recordConfidence: Boolean = true,
    val preferNpuForDensePerception: Boolean = true,
    val allowRelativeDepthWhenMetricUnavailable: Boolean = true,
    val compactSession: Boolean = true,
    val correlatedEvidenceEnabled: Boolean = true,
    val correlatedEvidenceQualificationMode: Boolean = false,
    val correlatedEvidenceMinimumCoverage: Double = 0.60,
    val identityExchangeRiskEnabled: Boolean = true,
    val identityExchangeHistoryFrames: Int = 12,
    val lightingConfidence: Double = 1.0,
    val distributedGlobalCameraEnvelope: Int = 128,
    val distributedAuthoritativeCameraBudget: Int = 32,
    val distributedMaximumShards: Int = 16,
    val distributedMinimumAuthorityHealth: Double = 0.35,
    val distributedMinimumAuthoritySynchronization: Double = 0.55,
    val distributedMaximumTemporalResidualMs: Double = 12.0,
    val distributedMaximumTemporalOutlierRatio: Double = 0.20,
    val distributedMinimumTransportHeadroom: Double = 0.45,
) {
    val correlatedEvidenceMaximumCommonModeRisk: Double = 0.70
    val identityExchangeHighRiskThreshold: Double = 0.72
    val identityExchangeHoldThreshold: Double = 0.48

    init {
        require(cameraManualExposureMicros in 1L..500_000L)
        require(liveMotionPreviewPredictionHorizonMillis in 0.0..100.0)
        require(liveMotionPreviewMaxStaleMillis in 16.0..1000.0)
        require(liveMotionPreviewDetail in 0.0..1.0)
        require(cameraManualIso in 25..12_800)
        require(cameraExposureCompensationEv in -8.0..8.0)
        require(cameraColorTemperatureKelvin in 2000..10000)
        require(cameraColorTint in -100..100)

        cameraManualFocusDistanceDiopters?.let {
            require(it.isFinite() && it >= 0.0)
        }
        require(targetFps in 15..240)
        require(inferenceFps in 5..targetFps)
        require(maxSubjects in 1..64)
        require(fullDetailActorBudget in 1..64)
        require(faceDetailActorBudget in 1..64)
        require(eyeDetailActorBudget in 1..64)
        require(handDetailActorBudget in 1..64)
        require(denseSurfaceActorBudget in 1..64)
        require(priorityDetailRetention in 0.0..1.0)
        if (captureEvidenceMode != CaptureEvidenceMode.LIVE_MOTION_ONLY) {
            require(recordedVideoEvidenceProfile != null) {
                "Recorded video evidence profile is required for $captureEvidenceMode"
            }
        }
        require(multiPersonCapacityPolicy.targetSubjects <= maxSubjects) {
            "Target multi-person capacity cannot exceed maxSubjects"
        }
        require(actorIdentityStrictness in 0.0..1.0)
        require(actorOcclusionRecovery in 0.0..1.0)
        require(crowdedSceneOverlapThreshold in 0.0..1.0)
        require(crowdedSceneMatchScoreBoost in 0.0..0.30)
        require(actorCrossCameraAssociation in 0.0..1.0)
        require(actorMicroMotionSensitivity in 0.0..1.0)
        require(motionFrequencyBandSensitivity in 0.0..1.0)
        require(actorMicroMotionRetention in 0.0..1.0)
        require(actorSubframeTimingStrictness in 0.0..1.0)
        require(bodyDetail in 0.0..1.0)
        require(faceDetail in 0.0..1.0)
        require(eyeDetail in 0.0..1.0)
        require(handDetail in 0.0..1.0)
        require(depthDetail in 0.0..1.0)
        require(objectDetail in 0.0..1.0)
        require(denseMeshDetail in 0.0..1.0)
        require(personSegmentationCadenceFrames in 1..120)
        require(personSegmentationPriority in 0.0..1.0)
        require(contactDetail in 0.0..1.0)
        require(worldConsistencyDetail in 0.0..1.0)
        require(topologyConstraintStrength in 0.0..1.0)
        require(multiViewFusionWeight in 0.0..1.0)
        require(multiViewEvidenceAdaptivity in 0.0..1.0)
        require(reconstructionEvidenceFusionStrength in 0.0..1.0)
        require(reconstructionEvidenceDisagreementToleranceMeters in 0.02..1.0)
        require(reconstructionEvidenceMinimumSourceDiversity in 0.0..1.0)
        require(adaptiveDetectorIntervalFrames in 1..120)
        require(maximumOcclusionPredictionFrames in 0..60)
        require(temporalSmoothing in 0.0..1.0)
        require(occlusionPersistence in 0.0..1.0)
        require(microMotionPreservation in 0.0..1.0)
        require(cameraMotionSuppression in 0.0..1.0)
        require(faceEyeFusionWeight in 0.0..1.0)
        require(faceEyeFusionStrictness in 0.0..1.0)
        require(faceEyeIpdDriftToleranceMillimeters in 4.0..30.0)
        require(eyeTemporalStrictness in 0.0..1.0)
        require(handTemporalStrictness in 0.0..1.0)
        require(handGeometryStrictness in 0.0..1.0)
        require(handLongitudinalAnatomyStrictness in 0.0..1.0)
        require(temporalAlignmentStrictness in 0.0..1.0)
        require(subframeAlignmentStrictness in 0.0..1.0)
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
        require(rollingShutterCompensation in 0.0..1.0)
        require(outputIntegrityStrictness in 0.0..1.0)
        require(physicsConstraintStrictness in 0.0..1.0)
        require(global4DConsistencyStrictness in 0.0..1.0)
        require(surfaceEvidenceStrictness in 0.0..1.0)
        require(denseLandmarkCoverageStrictness in 0.0..1.0)
        require(denseLandmarkUncertaintyStrictness in 0.0..1.0)
        require(denseLandmarkCrossViewStrictness in 0.0..1.0)
        require(denseLandmarkTemporalStrictness in 0.0..1.0)
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
        require(minimumParallaxDegrees > 0.0 && minimumParallaxDegrees <= 10.0)
        require(preferredTemporalSkewMillis > 0.0)
        require(maximumTemporalSkewMillis > preferredTemporalSkewMillis)
        require(maximumTemporalSkewMillis <= 250.0)
        require(maximumReprojectionErrorPx > 0.0 && maximumReprojectionErrorPx <= 100.0)
        require(
            maximumReconstructionUncertaintyMeters > 0.0 &&
                maximumReconstructionUncertaintyMeters <= 10.0,
        )
        require(distributedGlobalCameraEnvelope in 2..128)
        require(
            distributedAuthoritativeCameraBudget in
                2..distributedGlobalCameraEnvelope.coerceAtMost(32),
        )
        require(
            distributedMaximumShards in
                1..distributedGlobalCameraEnvelope.coerceAtMost(16),
        )
        require(distributedMinimumAuthorityHealth in 0.0..1.0)
        require(
            distributedMinimumAuthoritySynchronization in
                0.0..1.0,
        )
        require(distributedMaximumTemporalResidualMs > 0.0)
        require(distributedMaximumTemporalOutlierRatio in 0.0..1.0)
        require(distributedMinimumTransportHeadroom in 0.0..1.0)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MotionCaptureCustomization) return false
        return (
            mode == other.mode &&
            cameraResolutionPreset == other.cameraResolutionPreset &&
            cameraDynamicRangeMode == other.cameraDynamicRangeMode &&
            cameraExposureMode == other.cameraExposureMode &&
            cameraAntibandingMode == other.cameraAntibandingMode &&
            cameraFocusMode == other.cameraFocusMode &&
            cameraManualExposureMicros == other.cameraManualExposureMicros &&
            cameraManualIso == other.cameraManualIso &&
            cameraExposureCompensationEv == other.cameraExposureCompensationEv &&
            cameraAutoExposureLock == other.cameraAutoExposureLock &&
            cameraAutoWhiteBalanceLock == other.cameraAutoWhiteBalanceLock &&
            cameraManualFocusDistanceDiopters == other.cameraManualFocusDistanceDiopters &&
            cameraPreciseColorTemperatureEnabled ==
                other.cameraPreciseColorTemperatureEnabled &&
            cameraColorTemperatureKelvin ==
                other.cameraColorTemperatureKelvin &&
            cameraColorTint ==
                other.cameraColorTint &&

            cameraOisEnabled == other.cameraOisEnabled &&
            cameraElectronicStabilizationEnabled == other.cameraElectronicStabilizationEnabled &&
            liveMotionPreviewEnabled == other.liveMotionPreviewEnabled &&
            liveMotionPreviewGpuMode == other.liveMotionPreviewGpuMode &&
            liveMotionOverlayBody == other.liveMotionOverlayBody &&
            liveMotionOverlayFace == other.liveMotionOverlayFace &&
            liveMotionOverlayEyes == other.liveMotionOverlayEyes &&
            liveMotionOverlayDepth == other.liveMotionOverlayDepth &&
            liveMotionOverlayMicroMotion == other.liveMotionOverlayMicroMotion &&
            liveMotionOverlayVelocity == other.liveMotionOverlayVelocity &&
            liveMotionOverlayUncertainty == other.liveMotionOverlayUncertainty &&
            liveMotionOverlayActorIds == other.liveMotionOverlayActorIds &&
            liveMotionPreviewPredictionHorizonMillis == other.liveMotionPreviewPredictionHorizonMillis &&
            liveMotionPreviewMaxStaleMillis == other.liveMotionPreviewMaxStaleMillis &&
            liveMotionPreviewDetail == other.liveMotionPreviewDetail &&
            captureEvidenceMode == other.captureEvidenceMode &&
            recordedVideoEvidenceProfile == other.recordedVideoEvidenceProfile &&
            postCaptureReconstructionPolicy == other.postCaptureReconstructionPolicy &&
            postCaptureAutoReconstructionEnabled == other.postCaptureAutoReconstructionEnabled &&
            multiPersonCapacityPolicy == other.multiPersonCapacityPolicy &&
            cameraRawSidecarEnabled == other.cameraRawSidecarEnabled &&
            cameraDepthSidecarEnabled == other.cameraDepthSidecarEnabled &&
            cameraPreserveSensorTimestamp == other.cameraPreserveSensorTimestamp &&
            targetFps == other.targetFps &&
            inferenceFps == other.inferenceFps &&
            maxSubjects == other.maxSubjects &&
            actorDetailPriorityMode == other.actorDetailPriorityMode &&
            fullDetailActorBudget == other.fullDetailActorBudget &&
            faceDetailActorBudget == other.faceDetailActorBudget &&
            eyeDetailActorBudget == other.eyeDetailActorBudget &&
            handDetailActorBudget == other.handDetailActorBudget &&
            denseSurfaceActorBudget == other.denseSurfaceActorBudget &&
            priorityDetailRetention == other.priorityDetailRetention &&
            actorIdentityStrictness == other.actorIdentityStrictness &&
            actorOcclusionRecovery == other.actorOcclusionRecovery &&
            crowdedSceneOverlapThreshold == other.crowdedSceneOverlapThreshold &&
            crowdedSceneMatchScoreBoost == other.crowdedSceneMatchScoreBoost &&
            actorCrossCameraAssociation == other.actorCrossCameraAssociation &&
            actorMicroMotionSensitivity == other.actorMicroMotionSensitivity &&
            motionFrequencyBandSensitivity == other.motionFrequencyBandSensitivity &&
            actorMicroMotionRetention == other.actorMicroMotionRetention &&
            actorSubframeTimingStrictness == other.actorSubframeTimingStrictness &&
            bodyDetail == other.bodyDetail &&
            faceDetail == other.faceDetail &&
            eyeDetail == other.eyeDetail &&
            handDetail == other.handDetail &&
            depthDetail == other.depthDetail &&
            objectDetail == other.objectDetail &&
            denseMeshDetail == other.denseMeshDetail &&
            personSegmentationEnabled == other.personSegmentationEnabled &&
            personSegmentationCadenceFrames == other.personSegmentationCadenceFrames &&
            personSegmentationPriority == other.personSegmentationPriority &&
            contactDetail == other.contactDetail &&
            worldConsistencyDetail == other.worldConsistencyDetail &&
            topologyConstraintStrength == other.topologyConstraintStrength &&
            multiViewFusionWeight == other.multiViewFusionWeight &&
            multiViewEvidenceAdaptivity == other.multiViewEvidenceAdaptivity &&
            reconstructionEvidenceFusionStrength == other.reconstructionEvidenceFusionStrength &&
            reconstructionEvidenceDisagreementToleranceMeters == other.reconstructionEvidenceDisagreementToleranceMeters &&
            reconstructionEvidenceMinimumSourceDiversity == other.reconstructionEvidenceMinimumSourceDiversity &&
            adaptiveDetectorIntervalFrames == other.adaptiveDetectorIntervalFrames &&
            maximumOcclusionPredictionFrames == other.maximumOcclusionPredictionFrames &&
            temporalSmoothing == other.temporalSmoothing &&
            occlusionPersistence == other.occlusionPersistence &&
            microMotionPreservation == other.microMotionPreservation &&
            cameraMotionSuppression == other.cameraMotionSuppression &&
            faceEyeFusionWeight == other.faceEyeFusionWeight &&
            faceEyeFusionStrictness == other.faceEyeFusionStrictness &&
            faceEyeIpdDriftToleranceMillimeters == other.faceEyeIpdDriftToleranceMillimeters &&
            eyeTemporalStrictness == other.eyeTemporalStrictness &&
            handTemporalStrictness == other.handTemporalStrictness &&
            handGeometryStrictness == other.handGeometryStrictness &&
            handLongitudinalAnatomyStrictness == other.handLongitudinalAnatomyStrictness &&
            temporalAlignmentStrictness == other.temporalAlignmentStrictness &&
            subframeAlignmentStrictness == other.subframeAlignmentStrictness &&
            interleavedTemporalFusionEnabled == other.interleavedTemporalFusionEnabled &&
            interleavedTemporalWindowMillis == other.interleavedTemporalWindowMillis &&
            interleavedPredictionHorizonMillis == other.interleavedPredictionHorizonMillis &&
            interleavedMaximumSampleSkewMillis == other.interleavedMaximumSampleSkewMillis &&
            interleavedOutlierThresholdPixels == other.interleavedOutlierThresholdPixels &&
            temporalSupersamplingPriority == other.temporalSupersamplingPriority &&
            rollingShutterCompensation == other.rollingShutterCompensation &&
            outputIntegrityStrictness == other.outputIntegrityStrictness &&
            physicsConstraintStrictness == other.physicsConstraintStrictness &&
            global4DConsistencyStrictness == other.global4DConsistencyStrictness &&
            surfaceEvidenceStrictness == other.surfaceEvidenceStrictness &&
            denseLandmarkCoverageStrictness == other.denseLandmarkCoverageStrictness &&
            denseLandmarkUncertaintyStrictness == other.denseLandmarkUncertaintyStrictness &&
            denseLandmarkCrossViewStrictness == other.denseLandmarkCrossViewStrictness &&
            denseLandmarkTemporalStrictness == other.denseLandmarkTemporalStrictness &&
            ultraPrecisionEnabled == other.ultraPrecisionEnabled &&
            ultraPrecisionQualificationMode == other.ultraPrecisionQualificationMode &&
            ultraPrecisionTargetP95ErrorMillimeters == other.ultraPrecisionTargetP95ErrorMillimeters &&
            ultraPrecisionTargetP99ErrorMillimeters == other.ultraPrecisionTargetP99ErrorMillimeters &&
            ultraPrecisionMinimumEvidenceSamples == other.ultraPrecisionMinimumEvidenceSamples &&
            ultraPrecisionMinimumBaselineMeters == other.ultraPrecisionMinimumBaselineMeters &&
            ultraPrecisionMinimumParallaxDegrees == other.ultraPrecisionMinimumParallaxDegrees &&
            ultraPrecisionMinimumCalibrationConfidence == other.ultraPrecisionMinimumCalibrationConfidence &&
            ultraPrecisionMinimumSynchronizationConfidence == other.ultraPrecisionMinimumSynchronizationConfidence &&
            ultraPrecisionMinimumViewDiversity == other.ultraPrecisionMinimumViewDiversity &&
            ultraPrecisionMinimumCrossViewAgreement == other.ultraPrecisionMinimumCrossViewAgreement &&
            ultraPrecisionMaximumOcclusionRatio == other.ultraPrecisionMaximumOcclusionRatio &&
            ultraPrecisionMinimumIdentityContinuity == other.ultraPrecisionMinimumIdentityContinuity &&
            ultraPrecisionMinimumSurfaceTemporalStability == other.ultraPrecisionMinimumSurfaceTemporalStability &&
            ultraPrecisionMaximumThermalRisk == other.ultraPrecisionMaximumThermalRisk &&
            ultraPrecisionMaximumMemoryPressure == other.ultraPrecisionMaximumMemoryPressure &&
            ultraPrecisionMatchedReferenceRequired == other.ultraPrecisionMatchedReferenceRequired &&
            minimumParallaxDegrees == other.minimumParallaxDegrees &&
            preferredTemporalSkewMillis == other.preferredTemporalSkewMillis &&
            maximumTemporalSkewMillis == other.maximumTemporalSkewMillis &&
            maximumReprojectionErrorPx == other.maximumReprojectionErrorPx &&
            maximumReconstructionUncertaintyMeters == other.maximumReconstructionUncertaintyMeters &&
            posePriorStrength == other.posePriorStrength &&
            footContactStrictness == other.footContactStrictness &&
            headPoseStrictness == other.headPoseStrictness &&
            gazePriority == other.gazePriority &&
            depthTemporalFusion == other.depthTemporalFusion &&
            cameraDriftResponse == other.cameraDriftResponse &&
            thermalQualityFloor == other.thermalQualityFloor &&
            networkJitterToleranceMillis == other.networkJitterToleranceMillis &&
            meshAdaptiveDetail == other.meshAdaptiveDetail &&
            retargetingStrictness == other.retargetingStrictness &&
            scalePreservation == other.scalePreservation &&
            occlusionRecoveryConfidence == other.occlusionRecoveryConfidence &&
            landmarkConsistencyStrictness == other.landmarkConsistencyStrictness &&
            semanticMotionSensitivity == other.semanticMotionSensitivity &&
            motionThresholdScale == other.motionThresholdScale &&
            rootMotionLock == other.rootMotionLock &&
            dynamicTopology == other.dynamicTopology &&
            preserveMicroMotion == other.preserveMicroMotion &&
            recordLightingState == other.recordLightingState &&
            recordImuState == other.recordImuState &&
            recordCameraTelemetry == other.recordCameraTelemetry &&
            recordDepthMetadata == other.recordDepthMetadata &&
            recordFaceEyeFusion == other.recordFaceEyeFusion &&
            recordHand3D == other.recordHand3D &&
            recordDenseMesh == other.recordDenseMesh &&
            recordConfidence == other.recordConfidence &&
            preferNpuForDensePerception == other.preferNpuForDensePerception &&
            allowRelativeDepthWhenMetricUnavailable == other.allowRelativeDepthWhenMetricUnavailable &&
            compactSession == other.compactSession &&
            correlatedEvidenceEnabled == other.correlatedEvidenceEnabled &&
            correlatedEvidenceQualificationMode == other.correlatedEvidenceQualificationMode &&
            correlatedEvidenceMinimumCoverage == other.correlatedEvidenceMinimumCoverage &&
            identityExchangeRiskEnabled == other.identityExchangeRiskEnabled &&
            identityExchangeHistoryFrames == other.identityExchangeHistoryFrames &&
            lightingConfidence == other.lightingConfidence &&
            distributedGlobalCameraEnvelope == other.distributedGlobalCameraEnvelope &&
            distributedAuthoritativeCameraBudget == other.distributedAuthoritativeCameraBudget &&
            distributedMaximumShards == other.distributedMaximumShards &&
            distributedMinimumAuthorityHealth == other.distributedMinimumAuthorityHealth &&
            distributedMinimumAuthoritySynchronization == other.distributedMinimumAuthoritySynchronization &&
            distributedMaximumTemporalResidualMs == other.distributedMaximumTemporalResidualMs &&
            distributedMaximumTemporalOutlierRatio == other.distributedMaximumTemporalOutlierRatio &&
            distributedMinimumTransportHeadroom == other.distributedMinimumTransportHeadroom
        )
    }

    override fun hashCode(): Int {
        var result = 1
        result = 31 * result + mode.hashCode()
        result = 31 * result + cameraResolutionPreset.hashCode()
        result = 31 * result + cameraDynamicRangeMode.hashCode()
        result = 31 * result + cameraExposureMode.hashCode()
        result = 31 * result + cameraAntibandingMode.hashCode()
        result = 31 * result + cameraFocusMode.hashCode()
        result = 31 * result + cameraManualExposureMicros.hashCode()
        result = 31 * result + cameraManualIso.hashCode()
        result = 31 * result + cameraExposureCompensationEv.hashCode()
        result = 31 * result + cameraAutoExposureLock.hashCode()
        result = 31 * result + cameraAutoWhiteBalanceLock.hashCode()
        result = 31 * result + cameraManualFocusDistanceDiopters.hashCode()
        result = 31 * result + cameraPreciseColorTemperatureEnabled.hashCode()
        result = 31 * result + cameraColorTemperatureKelvin.hashCode()
        result = 31 * result + cameraColorTint.hashCode()
        result = 31 * result + cameraOisEnabled.hashCode()
        result = 31 * result + cameraElectronicStabilizationEnabled.hashCode()
        result = 31 * result + liveMotionPreviewEnabled.hashCode()
        result = 31 * result + liveMotionPreviewGpuMode.hashCode()
        result = 31 * result + liveMotionOverlayBody.hashCode()
        result = 31 * result + liveMotionOverlayFace.hashCode()
        result = 31 * result + liveMotionOverlayEyes.hashCode()
        result = 31 * result + liveMotionOverlayDepth.hashCode()
        result = 31 * result + liveMotionOverlayMicroMotion.hashCode()
        result = 31 * result + liveMotionOverlayVelocity.hashCode()
        result = 31 * result + liveMotionOverlayUncertainty.hashCode()
        result = 31 * result + liveMotionOverlayActorIds.hashCode()
        result = 31 * result + liveMotionPreviewPredictionHorizonMillis.hashCode()
        result = 31 * result + liveMotionPreviewMaxStaleMillis.hashCode()
        result = 31 * result + liveMotionPreviewDetail.hashCode()
        result = 31 * result + captureEvidenceMode.hashCode()
        result = 31 * result + recordedVideoEvidenceProfile.hashCode()
        result = 31 * result + postCaptureReconstructionPolicy.hashCode()
        result = 31 * result + postCaptureAutoReconstructionEnabled.hashCode()
        result = 31 * result + multiPersonCapacityPolicy.hashCode()
        result = 31 * result + cameraRawSidecarEnabled.hashCode()
        result = 31 * result + cameraDepthSidecarEnabled.hashCode()
        result = 31 * result + cameraPreserveSensorTimestamp.hashCode()
        result = 31 * result + targetFps.hashCode()
        result = 31 * result + inferenceFps.hashCode()
        result = 31 * result + maxSubjects.hashCode()
        result = 31 * result + actorDetailPriorityMode.hashCode()
        result = 31 * result + fullDetailActorBudget.hashCode()
        result = 31 * result + faceDetailActorBudget.hashCode()
        result = 31 * result + eyeDetailActorBudget.hashCode()
        result = 31 * result + handDetailActorBudget.hashCode()
        result = 31 * result + denseSurfaceActorBudget.hashCode()
        result = 31 * result + priorityDetailRetention.hashCode()
        result = 31 * result + actorIdentityStrictness.hashCode()
        result = 31 * result + actorOcclusionRecovery.hashCode()
        result = 31 * result + crowdedSceneOverlapThreshold.hashCode()
        result = 31 * result + crowdedSceneMatchScoreBoost.hashCode()
        result = 31 * result + actorCrossCameraAssociation.hashCode()
        result = 31 * result + actorMicroMotionSensitivity.hashCode()
        result = 31 * result + motionFrequencyBandSensitivity.hashCode()
        result = 31 * result + actorMicroMotionRetention.hashCode()
        result = 31 * result + actorSubframeTimingStrictness.hashCode()
        result = 31 * result + bodyDetail.hashCode()
        result = 31 * result + faceDetail.hashCode()
        result = 31 * result + eyeDetail.hashCode()
        result = 31 * result + handDetail.hashCode()
        result = 31 * result + depthDetail.hashCode()
        result = 31 * result + objectDetail.hashCode()
        result = 31 * result + denseMeshDetail.hashCode()
        result = 31 * result + personSegmentationEnabled.hashCode()
        result = 31 * result + personSegmentationCadenceFrames.hashCode()
        result = 31 * result + personSegmentationPriority.hashCode()
        result = 31 * result + contactDetail.hashCode()
        result = 31 * result + worldConsistencyDetail.hashCode()
        result = 31 * result + topologyConstraintStrength.hashCode()
        result = 31 * result + multiViewFusionWeight.hashCode()
        result = 31 * result + multiViewEvidenceAdaptivity.hashCode()
        result = 31 * result + reconstructionEvidenceFusionStrength.hashCode()
        result = 31 * result + reconstructionEvidenceDisagreementToleranceMeters.hashCode()
        result = 31 * result + reconstructionEvidenceMinimumSourceDiversity.hashCode()
        result = 31 * result + adaptiveDetectorIntervalFrames.hashCode()
        result = 31 * result + maximumOcclusionPredictionFrames.hashCode()
        result = 31 * result + temporalSmoothing.hashCode()
        result = 31 * result + occlusionPersistence.hashCode()
        result = 31 * result + microMotionPreservation.hashCode()
        result = 31 * result + cameraMotionSuppression.hashCode()
        result = 31 * result + faceEyeFusionWeight.hashCode()
        result = 31 * result + faceEyeFusionStrictness.hashCode()
        result = 31 * result + faceEyeIpdDriftToleranceMillimeters.hashCode()
        result = 31 * result + eyeTemporalStrictness.hashCode()
        result = 31 * result + handTemporalStrictness.hashCode()
        result = 31 * result + handGeometryStrictness.hashCode()
        result = 31 * result + handLongitudinalAnatomyStrictness.hashCode()
        result = 31 * result + temporalAlignmentStrictness.hashCode()
        result = 31 * result + subframeAlignmentStrictness.hashCode()
        result = 31 * result + interleavedTemporalFusionEnabled.hashCode()
        result = 31 * result + interleavedTemporalWindowMillis.hashCode()
        result = 31 * result + interleavedPredictionHorizonMillis.hashCode()
        result = 31 * result + interleavedMaximumSampleSkewMillis.hashCode()
        result = 31 * result + interleavedOutlierThresholdPixels.hashCode()
        result = 31 * result + temporalSupersamplingPriority.hashCode()
        result = 31 * result + rollingShutterCompensation.hashCode()
        result = 31 * result + outputIntegrityStrictness.hashCode()
        result = 31 * result + physicsConstraintStrictness.hashCode()
        result = 31 * result + global4DConsistencyStrictness.hashCode()
        result = 31 * result + surfaceEvidenceStrictness.hashCode()
        result = 31 * result + denseLandmarkCoverageStrictness.hashCode()
        result = 31 * result + denseLandmarkUncertaintyStrictness.hashCode()
        result = 31 * result + denseLandmarkCrossViewStrictness.hashCode()
        result = 31 * result + denseLandmarkTemporalStrictness.hashCode()
        result = 31 * result + ultraPrecisionEnabled.hashCode()
        result = 31 * result + ultraPrecisionQualificationMode.hashCode()
        result = 31 * result + ultraPrecisionTargetP95ErrorMillimeters.hashCode()
        result = 31 * result + ultraPrecisionTargetP99ErrorMillimeters.hashCode()
        result = 31 * result + ultraPrecisionMinimumEvidenceSamples.hashCode()
        result = 31 * result + ultraPrecisionMinimumBaselineMeters.hashCode()
        result = 31 * result + ultraPrecisionMinimumParallaxDegrees.hashCode()
        result = 31 * result + ultraPrecisionMinimumCalibrationConfidence.hashCode()
        result = 31 * result + ultraPrecisionMinimumSynchronizationConfidence.hashCode()
        result = 31 * result + ultraPrecisionMinimumViewDiversity.hashCode()
        result = 31 * result + ultraPrecisionMinimumCrossViewAgreement.hashCode()
        result = 31 * result + ultraPrecisionMaximumOcclusionRatio.hashCode()
        result = 31 * result + ultraPrecisionMinimumIdentityContinuity.hashCode()
        result = 31 * result + ultraPrecisionMinimumSurfaceTemporalStability.hashCode()
        result = 31 * result + ultraPrecisionMaximumThermalRisk.hashCode()
        result = 31 * result + ultraPrecisionMaximumMemoryPressure.hashCode()
        result = 31 * result + ultraPrecisionMatchedReferenceRequired.hashCode()
        result = 31 * result + minimumParallaxDegrees.hashCode()
        result = 31 * result + preferredTemporalSkewMillis.hashCode()
        result = 31 * result + maximumTemporalSkewMillis.hashCode()
        result = 31 * result + maximumReprojectionErrorPx.hashCode()
        result = 31 * result + maximumReconstructionUncertaintyMeters.hashCode()
        result = 31 * result + posePriorStrength.hashCode()
        result = 31 * result + footContactStrictness.hashCode()
        result = 31 * result + headPoseStrictness.hashCode()
        result = 31 * result + gazePriority.hashCode()
        result = 31 * result + depthTemporalFusion.hashCode()
        result = 31 * result + cameraDriftResponse.hashCode()
        result = 31 * result + thermalQualityFloor.hashCode()
        result = 31 * result + networkJitterToleranceMillis.hashCode()
        result = 31 * result + meshAdaptiveDetail.hashCode()
        result = 31 * result + retargetingStrictness.hashCode()
        result = 31 * result + scalePreservation.hashCode()
        result = 31 * result + occlusionRecoveryConfidence.hashCode()
        result = 31 * result + landmarkConsistencyStrictness.hashCode()
        result = 31 * result + semanticMotionSensitivity.hashCode()
        result = 31 * result + motionThresholdScale.hashCode()
        result = 31 * result + rootMotionLock.hashCode()
        result = 31 * result + dynamicTopology.hashCode()
        result = 31 * result + preserveMicroMotion.hashCode()
        result = 31 * result + recordLightingState.hashCode()
        result = 31 * result + recordImuState.hashCode()
        result = 31 * result + recordCameraTelemetry.hashCode()
        result = 31 * result + recordDepthMetadata.hashCode()
        result = 31 * result + recordFaceEyeFusion.hashCode()
        result = 31 * result + recordHand3D.hashCode()
        result = 31 * result + recordDenseMesh.hashCode()
        result = 31 * result + recordConfidence.hashCode()
        result = 31 * result + preferNpuForDensePerception.hashCode()
        result = 31 * result + allowRelativeDepthWhenMetricUnavailable.hashCode()
        result = 31 * result + compactSession.hashCode()
        result = 31 * result + correlatedEvidenceEnabled.hashCode()
        result = 31 * result + correlatedEvidenceQualificationMode.hashCode()
        result = 31 * result + correlatedEvidenceMinimumCoverage.hashCode()
        result = 31 * result + identityExchangeRiskEnabled.hashCode()
        result = 31 * result + identityExchangeHistoryFrames.hashCode()
        result = 31 * result + lightingConfidence.hashCode()
        result = 31 * result + distributedGlobalCameraEnvelope.hashCode()
        result = 31 * result + distributedAuthoritativeCameraBudget.hashCode()
        result = 31 * result + distributedMaximumShards.hashCode()
        result = 31 * result + distributedMinimumAuthorityHealth.hashCode()
        result = 31 * result + distributedMinimumAuthoritySynchronization.hashCode()
        result = 31 * result + distributedMaximumTemporalResidualMs.hashCode()
        result = 31 * result + distributedMaximumTemporalOutlierRatio.hashCode()
        result = 31 * result + distributedMinimumTransportHeadroom.hashCode()
        return result
    }

    fun copy(
        recordFaceEyeFusion: Boolean = this.recordFaceEyeFusion,
        recordHand3D: Boolean = this.recordHand3D,
        outputIntegrityStrictness: Double = this.outputIntegrityStrictness,
    ): MotionCaptureCustomization =
        copyWith {
            this.recordFaceEyeFusion = recordFaceEyeFusion
            this.recordHand3D = recordHand3D
            this.outputIntegrityStrictness = outputIntegrityStrictness
        }

    fun copyWith(
        block: MotionCaptureCustomizationCopyBuilder.() -> Unit,
    ): MotionCaptureCustomization =
        MotionCaptureCustomizationCopyBuilder(this)
            .apply(block)
            .build()

    companion object {
        fun production(): MotionCaptureCustomization =
            MotionCaptureCustomization(
                cameraResolutionPreset = CameraResolutionPreset.AUTO,
                cameraExposureMode = CameraExposureMode.AUTO,
                cameraFocusMode = CameraFocusMode.TRACKED_SUBJECT,
                cameraDepthSidecarEnabled = true,
                cameraPreserveSensorTimestamp = true,
            )

        fun qualityFirst(): MotionCaptureCustomization =
            MotionCaptureCustomization(
                targetFps = 60,
                inferenceFps = 60,
                cameraResolutionPreset = CameraResolutionPreset.AUTO,
                cameraExposureMode = CameraExposureMode.AUTO,
                cameraFocusMode = CameraFocusMode.TRACKED_SUBJECT,
                cameraDepthSidecarEnabled = true,
                bodyDetail = 1.0,
                faceDetail = 1.0,
                eyeDetail = 1.0,
                handDetail = 1.0,
                depthDetail = 1.0,
                worldConsistencyDetail = 1.0,
                topologyConstraintStrength = 0.96,
                multiViewEvidenceAdaptivity = 0.98,
                temporalSmoothing = 0.72,
                occlusionPersistence = 0.88,
                microMotionPreservation = 1.0,
                cameraMotionSuppression = 0.88,
                faceEyeFusionWeight = 0.68,
                faceEyeFusionStrictness = 0.96,
                faceEyeIpdDriftToleranceMillimeters = 8.0,
                eyeTemporalStrictness = 0.92,
                handTemporalStrictness = 0.92,
                handGeometryStrictness = 0.98,
                temporalAlignmentStrictness = 0.92,
                subframeAlignmentStrictness = 0.96,
                rollingShutterCompensation = 0.98,
                outputIntegrityStrictness = 0.94,
                physicsConstraintStrictness = 0.96,
                surfaceEvidenceStrictness = 0.96,
                denseLandmarkCoverageStrictness = 0.96,
                denseLandmarkUncertaintyStrictness = 0.98,
                denseLandmarkCrossViewStrictness = 0.96,
                denseLandmarkTemporalStrictness = 0.94,
                ultraPrecisionEnabled = true,
                ultraPrecisionQualificationMode = false,
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
                minimumParallaxDegrees = 0.85,
                preferredTemporalSkewMillis = 2.0,
                maximumTemporalSkewMillis = 10.0,
                maximumReprojectionErrorPx = 4.0,
                maximumReconstructionUncertaintyMeters = 0.30,
                posePriorStrength = 0.96,
                footContactStrictness = 0.94,
                headPoseStrictness = 0.96,
                gazePriority = 0.82,
                depthTemporalFusion = 0.95,
                cameraDriftResponse = 0.96,
                thermalQualityFloor = 0.70,
                networkJitterToleranceMillis = 8.0,
                meshAdaptiveDetail = 0.94,
                retargetingStrictness = 0.96,
                scalePreservation = 0.99,
                occlusionRecoveryConfidence = 0.92,
                landmarkConsistencyStrictness = 0.96,
                semanticMotionSensitivity = 0.92,
                rootMotionLock = 0.90,
                preserveMicroMotion = true,
                compactSession = false,
            )

            fun frontierPrecision50Actor(): MotionCaptureCustomization =
            qualityFirst().copyWith {
                mode = MotionCaptureMode.MULTI_VIEW_RGB
                targetFps = 60
                inferenceFps = 60
                cameraExposureMode = CameraExposureMode.MANUAL
                cameraManualExposureMicros = 5000L
                cameraManualIso = 800
                cameraAutoExposureLock = true
                cameraAutoWhiteBalanceLock = true
                cameraOisEnabled = false
                cameraElectronicStabilizationEnabled = false
                cameraDepthSidecarEnabled = true
                cameraRawSidecarEnabled = true
                cameraPreserveSensorTimestamp = true

                maxSubjects = 50
                fullDetailActorBudget = 16
                faceDetailActorBudget = 24
                eyeDetailActorBudget = 24
                handDetailActorBudget = 24
                denseSurfaceActorBudget = 20
                actorDetailPriorityMode = ActorDetailPriorityMode.MOTION_FIRST
                priorityDetailRetention = 0.98
                actorIdentityStrictness = 0.98
                actorOcclusionRecovery = 0.94
                actorCrossCameraAssociation = 0.98
                actorMicroMotionSensitivity = 0.98
                actorMicroMotionRetention = 0.99
                actorSubframeTimingStrictness = 0.99

                bodyDetail = 1.0
                faceDetail = 1.0
                eyeDetail = 1.0
                handDetail = 1.0
                depthDetail = 1.0
                denseMeshDetail = 0.98
                worldConsistencyDetail = 0.99
                topologyConstraintStrength = 0.99
                multiViewFusionWeight = 0.99
                multiViewEvidenceAdaptivity = 0.99
                reconstructionEvidenceFusionStrength = 0.99
                temporalSmoothing = 0.80
                occlusionPersistence = 0.96
                microMotionPreservation = 1.0
                cameraMotionSuppression = 0.95

                faceEyeFusionWeight = 0.78
                faceEyeFusionStrictness = 0.99
                eyeTemporalStrictness = 0.98
                handTemporalStrictness = 0.97
                handGeometryStrictness = 0.99
                handLongitudinalAnatomyStrictness = 0.99
                temporalAlignmentStrictness = 0.99
                subframeAlignmentStrictness = 0.99
                temporalSupersamplingPriority = 0.98
                rollingShutterCompensation = 0.99

                physicsConstraintStrictness = 0.99
                global4DConsistencyStrictness = 0.99
                surfaceEvidenceStrictness = 0.99
                denseLandmarkCoverageStrictness = 0.99
                denseLandmarkUncertaintyStrictness = 0.99
                denseLandmarkCrossViewStrictness = 0.99
                denseLandmarkTemporalStrictness = 0.98

                ultraPrecisionEnabled = true
                ultraPrecisionQualificationMode = true
                ultraPrecisionTargetP95ErrorMillimeters = 0.75
                ultraPrecisionTargetP99ErrorMillimeters = 1.50
                ultraPrecisionMinimumEvidenceSamples = 2000
                ultraPrecisionMinimumBaselineMeters = 0.75
                ultraPrecisionMinimumParallaxDegrees = 12.0
                ultraPrecisionMinimumCalibrationConfidence = 0.98
                ultraPrecisionMinimumSynchronizationConfidence = 0.98
                ultraPrecisionMinimumViewDiversity = 0.85
                ultraPrecisionMinimumCrossViewAgreement = 0.95
                ultraPrecisionMaximumOcclusionRatio = 0.10
                ultraPrecisionMinimumIdentityContinuity = 0.98
                ultraPrecisionMinimumSurfaceTemporalStability = 0.97
                ultraPrecisionMaximumThermalRisk = 0.50
                ultraPrecisionMaximumMemoryPressure = 0.65
                ultraPrecisionMatchedReferenceRequired = true

                minimumParallaxDegrees = 1.0
                preferredTemporalSkewMillis = 1.0
                maximumTemporalSkewMillis = 4.0
                maximumReprojectionErrorPx = 2.5
                maximumReconstructionUncertaintyMeters = 0.12
                posePriorStrength = 0.98
                footContactStrictness = 0.98
                headPoseStrictness = 0.99
                gazePriority = 0.92
                depthTemporalFusion = 0.99
                cameraDriftResponse = 0.99
                thermalQualityFloor = 0.75
                networkJitterToleranceMillis = 4.0
                meshAdaptiveDetail = 0.99
                retargetingStrictness = 0.99
                scalePreservation = 0.998
                occlusionRecoveryConfidence = 0.97
                landmarkConsistencyStrictness = 0.99
                semanticMotionSensitivity = 0.98
                rootMotionLock = 0.95

                personSegmentationEnabled = true
                personSegmentationCadenceFrames = 2
                personSegmentationPriority = 0.96
                contactDetail = 0.96
                dynamicTopology = true
                preferNpuForDensePerception = true
                allowRelativeDepthWhenMetricUnavailable = false

                correlatedEvidenceEnabled = true
                correlatedEvidenceQualificationMode = true
                correlatedEvidenceMinimumCoverage = 0.80
                identityExchangeRiskEnabled = true
                identityExchangeHistoryFrames = 24

                compactSession = false
                recordLightingState = true
                recordImuState = true
                recordCameraTelemetry = true
                recordDepthMetadata = true
                recordFaceEyeFusion = true
                recordHand3D = true
                recordDenseMesh = true
                recordConfidence = true
            }

             fun multiViewPrecision(): MotionCaptureCustomization =
            multiViewPrecisionFrom(qualityFirst())

         fun faceEyePrecision(): MotionCaptureCustomization =
            faceEyePrecisionFrom(qualityFirst())

    fun lowLatency(): MotionCaptureCustomization =
            MotionCaptureCustomization(
                targetFps = 120,
                inferenceFps = 60,
                cameraResolutionPreset = CameraResolutionPreset.FULL_HD_1080P,
                cameraExposureMode = CameraExposureMode.AUTO,
                cameraFocusMode = CameraFocusMode.CONTINUOUS,
                cameraDepthSidecarEnabled = false,
                bodyDetail = 0.82,
                faceDetail = 0.82,
                eyeDetail = 0.78,
                handDetail = 0.84,
                depthDetail = 0.82,
                worldConsistencyDetail = 0.68,
                topologyConstraintStrength = 0.58,
                multiViewEvidenceAdaptivity = 0.60,
                temporalSmoothing = 0.42,
                occlusionPersistence = 0.62,
                microMotionPreservation = 0.62,
                cameraMotionSuppression = 0.42,
                faceEyeFusionWeight = 0.56,
                faceEyeFusionStrictness = 0.58,
                eyeTemporalStrictness = 0.54,
                handTemporalStrictness = 0.54,
                handGeometryStrictness = 0.60,
                temporalAlignmentStrictness = 0.54,
                subframeAlignmentStrictness = 0.52,
                rollingShutterCompensation = 0.56,
                outputIntegrityStrictness = 0.48,
                physicsConstraintStrictness = 0.58,
                surfaceEvidenceStrictness = 0.56,
                minimumParallaxDegrees = 0.40,
                preferredTemporalSkewMillis = 8.0,
                maximumTemporalSkewMillis = 35.0,
                maximumReprojectionErrorPx = 12.0,
                maximumReconstructionUncertaintyMeters = 1.20,
                posePriorStrength = 0.66,
                footContactStrictness = 0.54,
                headPoseStrictness = 0.62,
                gazePriority = 0.55,
                depthTemporalFusion = 0.68,
                cameraDriftResponse = 0.60,
                thermalQualityFloor = 0.40,
                networkJitterToleranceMillis = 35.0,
                meshAdaptiveDetail = 0.64,
                retargetingStrictness = 0.62,
                scalePreservation = 0.86,
                occlusionRecoveryConfidence = 0.68,
                landmarkConsistencyStrictness = 0.62,
                semanticMotionSensitivity = 0.70,
                rootMotionLock = 0.58,
            )

        private fun multiViewPrecisionFrom(
            base: MotionCaptureCustomization,
        ): MotionCaptureCustomization =
            MotionCaptureCustomization(
                mode = base.mode,
                cameraResolutionPreset = base.cameraResolutionPreset,
                cameraDynamicRangeMode = base.cameraDynamicRangeMode,
                cameraExposureMode = base.cameraExposureMode,
                cameraAntibandingMode = base.cameraAntibandingMode,
                cameraFocusMode = base.cameraFocusMode,
                cameraManualExposureMicros = base.cameraManualExposureMicros,
                cameraManualIso = base.cameraManualIso,
                cameraExposureCompensationEv = base.cameraExposureCompensationEv,
                cameraAutoExposureLock = base.cameraAutoExposureLock,
                cameraAutoWhiteBalanceLock = base.cameraAutoWhiteBalanceLock,
                cameraManualFocusDistanceDiopters = base.cameraManualFocusDistanceDiopters,
                cameraOisEnabled = base.cameraOisEnabled,
                cameraElectronicStabilizationEnabled = base.cameraElectronicStabilizationEnabled,
                liveMotionPreviewEnabled = base.liveMotionPreviewEnabled,
                liveMotionPreviewGpuMode = base.liveMotionPreviewGpuMode,
                liveMotionOverlayBody = base.liveMotionOverlayBody,
                liveMotionOverlayFace = base.liveMotionOverlayFace,
                liveMotionOverlayEyes = base.liveMotionOverlayEyes,
                liveMotionOverlayDepth = base.liveMotionOverlayDepth,
                liveMotionOverlayMicroMotion = base.liveMotionOverlayMicroMotion,
                liveMotionOverlayVelocity = base.liveMotionOverlayVelocity,
                liveMotionOverlayUncertainty = base.liveMotionOverlayUncertainty,
                liveMotionOverlayActorIds = base.liveMotionOverlayActorIds,
                liveMotionPreviewPredictionHorizonMillis = base.liveMotionPreviewPredictionHorizonMillis,
                liveMotionPreviewMaxStaleMillis = base.liveMotionPreviewMaxStaleMillis,
                liveMotionPreviewDetail = base.liveMotionPreviewDetail,
                captureEvidenceMode = base.captureEvidenceMode,
                recordedVideoEvidenceProfile = base.recordedVideoEvidenceProfile,
                postCaptureReconstructionPolicy = base.postCaptureReconstructionPolicy,
                postCaptureAutoReconstructionEnabled = base.postCaptureAutoReconstructionEnabled,
                multiPersonCapacityPolicy = base.multiPersonCapacityPolicy,
                cameraRawSidecarEnabled = base.cameraRawSidecarEnabled,
                cameraDepthSidecarEnabled = base.cameraDepthSidecarEnabled,
                cameraPreserveSensorTimestamp = base.cameraPreserveSensorTimestamp,
                targetFps = 60,
                inferenceFps = 30,
                maxSubjects = base.maxSubjects,
                actorDetailPriorityMode = base.actorDetailPriorityMode,
                fullDetailActorBudget = base.fullDetailActorBudget,
                faceDetailActorBudget = base.faceDetailActorBudget,
                eyeDetailActorBudget = base.eyeDetailActorBudget,
                handDetailActorBudget = base.handDetailActorBudget,
                denseSurfaceActorBudget = base.denseSurfaceActorBudget,
                priorityDetailRetention = base.priorityDetailRetention,
                actorIdentityStrictness = base.actorIdentityStrictness,
                actorOcclusionRecovery = base.actorOcclusionRecovery,
                actorCrossCameraAssociation = base.actorCrossCameraAssociation,
                actorMicroMotionSensitivity = base.actorMicroMotionSensitivity,
                motionFrequencyBandSensitivity = base.motionFrequencyBandSensitivity,
                actorMicroMotionRetention = base.actorMicroMotionRetention,
                actorSubframeTimingStrictness = base.actorSubframeTimingStrictness,
                bodyDetail = base.bodyDetail,
                faceDetail = base.faceDetail,
                eyeDetail = base.eyeDetail,
                handDetail = base.handDetail,
                depthDetail = base.depthDetail,
                objectDetail = base.objectDetail,
                denseMeshDetail = base.denseMeshDetail,
                personSegmentationEnabled = base.personSegmentationEnabled,
                personSegmentationCadenceFrames = base.personSegmentationCadenceFrames,
                personSegmentationPriority = base.personSegmentationPriority,
                contactDetail = base.contactDetail,
                worldConsistencyDetail = base.worldConsistencyDetail,
                topologyConstraintStrength = 1.0,
                multiViewFusionWeight = 0.96,
                multiViewEvidenceAdaptivity = 1.0,
                reconstructionEvidenceFusionStrength = base.reconstructionEvidenceFusionStrength,
                reconstructionEvidenceDisagreementToleranceMeters = base.reconstructionEvidenceDisagreementToleranceMeters,
                reconstructionEvidenceMinimumSourceDiversity = base.reconstructionEvidenceMinimumSourceDiversity,
                adaptiveDetectorIntervalFrames = base.adaptiveDetectorIntervalFrames,
                maximumOcclusionPredictionFrames = base.maximumOcclusionPredictionFrames,
                temporalSmoothing = base.temporalSmoothing,
                occlusionPersistence = base.occlusionPersistence,
                microMotionPreservation = base.microMotionPreservation,
                cameraMotionSuppression = base.cameraMotionSuppression,
                faceEyeFusionWeight = base.faceEyeFusionWeight,
                faceEyeFusionStrictness = base.faceEyeFusionStrictness,
                faceEyeIpdDriftToleranceMillimeters = base.faceEyeIpdDriftToleranceMillimeters,
                eyeTemporalStrictness = base.eyeTemporalStrictness,
                handTemporalStrictness = base.handTemporalStrictness,
                handGeometryStrictness = base.handGeometryStrictness,
                handLongitudinalAnatomyStrictness = base.handLongitudinalAnatomyStrictness,
                temporalAlignmentStrictness = 0.98,
                subframeAlignmentStrictness = 1.0,
                interleavedTemporalFusionEnabled = base.interleavedTemporalFusionEnabled,
                interleavedTemporalWindowMillis = base.interleavedTemporalWindowMillis,
                interleavedPredictionHorizonMillis = base.interleavedPredictionHorizonMillis,
                interleavedMaximumSampleSkewMillis = base.interleavedMaximumSampleSkewMillis,
                interleavedOutlierThresholdPixels = base.interleavedOutlierThresholdPixels,
                photometricExposureShiftSoftLimit =
                    base.photometricExposureShiftSoftLimit,
                photometricExposureShiftHardLimit =
                    base.photometricExposureShiftHardLimit,
                photometricExposureReliabilityFloor =
                    base.photometricExposureReliabilityFloor,
                temporalSupersamplingPriority = base.temporalSupersamplingPriority,
                rollingShutterCompensation = 1.0,
                outputIntegrityStrictness = base.outputIntegrityStrictness,
                physicsConstraintStrictness = 1.0,
                global4DConsistencyStrictness = base.global4DConsistencyStrictness,
                surfaceEvidenceStrictness = 1.0,
                denseLandmarkCoverageStrictness = 1.0,
                denseLandmarkUncertaintyStrictness = 1.0,
                denseLandmarkCrossViewStrictness = 1.0,
                denseLandmarkTemporalStrictness = 0.98,
                ultraPrecisionEnabled = base.ultraPrecisionEnabled,
                ultraPrecisionQualificationMode = base.ultraPrecisionQualificationMode,
                ultraPrecisionTargetP95ErrorMillimeters = base.ultraPrecisionTargetP95ErrorMillimeters,
                ultraPrecisionTargetP99ErrorMillimeters = base.ultraPrecisionTargetP99ErrorMillimeters,
                ultraPrecisionMinimumEvidenceSamples = base.ultraPrecisionMinimumEvidenceSamples,
                ultraPrecisionMinimumBaselineMeters = base.ultraPrecisionMinimumBaselineMeters,
                ultraPrecisionMinimumParallaxDegrees = base.ultraPrecisionMinimumParallaxDegrees,
                ultraPrecisionMinimumCalibrationConfidence = base.ultraPrecisionMinimumCalibrationConfidence,
                ultraPrecisionMinimumSynchronizationConfidence = base.ultraPrecisionMinimumSynchronizationConfidence,
                ultraPrecisionMinimumViewDiversity = base.ultraPrecisionMinimumViewDiversity,
                ultraPrecisionMinimumCrossViewAgreement = base.ultraPrecisionMinimumCrossViewAgreement,
                ultraPrecisionMaximumOcclusionRatio = base.ultraPrecisionMaximumOcclusionRatio,
                ultraPrecisionMinimumIdentityContinuity = base.ultraPrecisionMinimumIdentityContinuity,
                ultraPrecisionMinimumSurfaceTemporalStability = base.ultraPrecisionMinimumSurfaceTemporalStability,
                ultraPrecisionMaximumThermalRisk = base.ultraPrecisionMaximumThermalRisk,
                ultraPrecisionMaximumMemoryPressure = base.ultraPrecisionMaximumMemoryPressure,
                ultraPrecisionMatchedReferenceRequired = base.ultraPrecisionMatchedReferenceRequired,
                minimumParallaxDegrees = 0.70,
                preferredTemporalSkewMillis = 2.0,
                maximumTemporalSkewMillis = 8.0,
                maximumReprojectionErrorPx = 3.5,
                maximumReconstructionUncertaintyMeters = 0.22,
                posePriorStrength = base.posePriorStrength,
                footContactStrictness = base.footContactStrictness,
                headPoseStrictness = base.headPoseStrictness,
                gazePriority = base.gazePriority,
                depthTemporalFusion = 0.98,
                cameraDriftResponse = 1.0,
                thermalQualityFloor = base.thermalQualityFloor,
                networkJitterToleranceMillis = base.networkJitterToleranceMillis,
                meshAdaptiveDetail = 1.0,
                retargetingStrictness = base.retargetingStrictness,
                scalePreservation = 1.0,
                occlusionRecoveryConfidence = base.occlusionRecoveryConfidence,
                landmarkConsistencyStrictness = base.landmarkConsistencyStrictness,
                semanticMotionSensitivity = base.semanticMotionSensitivity,
                motionThresholdScale = base.motionThresholdScale,
                rootMotionLock = base.rootMotionLock,
                dynamicTopology = base.dynamicTopology,
                preserveMicroMotion = base.preserveMicroMotion,
                recordLightingState = base.recordLightingState,
                recordImuState = base.recordImuState,
                recordCameraTelemetry = base.recordCameraTelemetry,
                recordDepthMetadata = base.recordDepthMetadata,
                recordFaceEyeFusion = base.recordFaceEyeFusion,
                recordHand3D = base.recordHand3D,
                recordDenseMesh = base.recordDenseMesh,
                recordConfidence = base.recordConfidence,
                preferNpuForDensePerception = base.preferNpuForDensePerception,
                allowRelativeDepthWhenMetricUnavailable = base.allowRelativeDepthWhenMetricUnavailable,
                compactSession = false,
                correlatedEvidenceEnabled = base.correlatedEvidenceEnabled,
                correlatedEvidenceQualificationMode = base.correlatedEvidenceQualificationMode,
                correlatedEvidenceMinimumCoverage = base.correlatedEvidenceMinimumCoverage,
                identityExchangeRiskEnabled = base.identityExchangeRiskEnabled,
                identityExchangeHistoryFrames = base.identityExchangeHistoryFrames,
                lightingConfidence = base.lightingConfidence,
                distributedGlobalCameraEnvelope =
                    base.distributedGlobalCameraEnvelope,
                distributedAuthoritativeCameraBudget =
                    base.distributedAuthoritativeCameraBudget,
                distributedMaximumShards =
                    base.distributedMaximumShards,
                distributedMinimumAuthorityHealth =
                    base.distributedMinimumAuthorityHealth,
                distributedMinimumAuthoritySynchronization =
                    base.distributedMinimumAuthoritySynchronization,
                distributedMaximumTemporalResidualMs =
                    base.distributedMaximumTemporalResidualMs,
                distributedMaximumTemporalOutlierRatio =
                    base.distributedMaximumTemporalOutlierRatio,
                distributedMinimumTransportHeadroom =
                    base.distributedMinimumTransportHeadroom,
            )

        private fun faceEyePrecisionFrom(
            base: MotionCaptureCustomization,
        ): MotionCaptureCustomization =
            MotionCaptureCustomization(
                mode = base.mode,
                cameraResolutionPreset = base.cameraResolutionPreset,
                cameraExposureMode = base.cameraExposureMode,
                cameraFocusMode = base.cameraFocusMode,
                cameraManualExposureMicros = base.cameraManualExposureMicros,
                cameraManualIso = base.cameraManualIso,
                cameraExposureCompensationEv = base.cameraExposureCompensationEv,
                cameraAutoExposureLock = base.cameraAutoExposureLock,
                cameraAutoWhiteBalanceLock = base.cameraAutoWhiteBalanceLock,
                cameraManualFocusDistanceDiopters = base.cameraManualFocusDistanceDiopters,
                cameraOisEnabled = base.cameraOisEnabled,
                cameraElectronicStabilizationEnabled = base.cameraElectronicStabilizationEnabled,
                liveMotionPreviewEnabled = base.liveMotionPreviewEnabled,
                liveMotionPreviewGpuMode = base.liveMotionPreviewGpuMode,
                liveMotionOverlayBody = base.liveMotionOverlayBody,
                liveMotionOverlayFace = base.liveMotionOverlayFace,
                liveMotionOverlayEyes = base.liveMotionOverlayEyes,
                liveMotionOverlayDepth = base.liveMotionOverlayDepth,
                liveMotionOverlayMicroMotion = base.liveMotionOverlayMicroMotion,
                liveMotionOverlayVelocity = base.liveMotionOverlayVelocity,
                liveMotionOverlayUncertainty = base.liveMotionOverlayUncertainty,
                liveMotionOverlayActorIds = base.liveMotionOverlayActorIds,
                liveMotionPreviewPredictionHorizonMillis = base.liveMotionPreviewPredictionHorizonMillis,
                liveMotionPreviewMaxStaleMillis = base.liveMotionPreviewMaxStaleMillis,
                liveMotionPreviewDetail = base.liveMotionPreviewDetail,
                captureEvidenceMode = base.captureEvidenceMode,
                recordedVideoEvidenceProfile = base.recordedVideoEvidenceProfile,
                postCaptureReconstructionPolicy = base.postCaptureReconstructionPolicy,
                postCaptureAutoReconstructionEnabled = base.postCaptureAutoReconstructionEnabled,
                multiPersonCapacityPolicy = base.multiPersonCapacityPolicy,
                cameraRawSidecarEnabled = base.cameraRawSidecarEnabled,
                cameraDepthSidecarEnabled = base.cameraDepthSidecarEnabled,
                cameraPreserveSensorTimestamp = base.cameraPreserveSensorTimestamp,
                targetFps = 60,
                inferenceFps = 60,
                maxSubjects = base.maxSubjects,
                actorDetailPriorityMode = base.actorDetailPriorityMode,
                fullDetailActorBudget = base.fullDetailActorBudget,
                faceDetailActorBudget = base.faceDetailActorBudget,
                eyeDetailActorBudget = base.eyeDetailActorBudget,
                handDetailActorBudget = base.handDetailActorBudget,
                denseSurfaceActorBudget = base.denseSurfaceActorBudget,
                priorityDetailRetention = base.priorityDetailRetention,
                actorIdentityStrictness = base.actorIdentityStrictness,
                actorOcclusionRecovery = base.actorOcclusionRecovery,
                actorCrossCameraAssociation = base.actorCrossCameraAssociation,
                actorMicroMotionSensitivity = base.actorMicroMotionSensitivity,
                motionFrequencyBandSensitivity = base.motionFrequencyBandSensitivity,
                actorMicroMotionRetention = base.actorMicroMotionRetention,
                actorSubframeTimingStrictness = base.actorSubframeTimingStrictness,
                bodyDetail = base.bodyDetail,
                faceDetail = 1.0,
                eyeDetail = 1.0,
                handDetail = base.handDetail,
                depthDetail = base.depthDetail,
                objectDetail = base.objectDetail,
                denseMeshDetail = base.denseMeshDetail,
                personSegmentationEnabled = base.personSegmentationEnabled,
                personSegmentationCadenceFrames = base.personSegmentationCadenceFrames,
                personSegmentationPriority = base.personSegmentationPriority,
                contactDetail = base.contactDetail,
                worldConsistencyDetail = base.worldConsistencyDetail,
                topologyConstraintStrength = base.topologyConstraintStrength,
                multiViewFusionWeight = base.multiViewFusionWeight,
                multiViewEvidenceAdaptivity = base.multiViewEvidenceAdaptivity,
                reconstructionEvidenceFusionStrength = base.reconstructionEvidenceFusionStrength,
                reconstructionEvidenceDisagreementToleranceMeters = base.reconstructionEvidenceDisagreementToleranceMeters,
                reconstructionEvidenceMinimumSourceDiversity = base.reconstructionEvidenceMinimumSourceDiversity,
                adaptiveDetectorIntervalFrames = base.adaptiveDetectorIntervalFrames,
                maximumOcclusionPredictionFrames = base.maximumOcclusionPredictionFrames,
                temporalSmoothing = 0.66,
                occlusionPersistence = 0.90,
                microMotionPreservation = 1.0,
                cameraMotionSuppression = base.cameraMotionSuppression,
                faceEyeFusionWeight = 0.92,
                faceEyeFusionStrictness = 0.99,
                faceEyeIpdDriftToleranceMillimeters = base.faceEyeIpdDriftToleranceMillimeters,
                eyeTemporalStrictness = 1.0,
                handTemporalStrictness = base.handTemporalStrictness,
                handGeometryStrictness = base.handGeometryStrictness,
                handLongitudinalAnatomyStrictness = base.handLongitudinalAnatomyStrictness,
                temporalAlignmentStrictness = base.temporalAlignmentStrictness,
                subframeAlignmentStrictness = base.subframeAlignmentStrictness,
                interleavedTemporalFusionEnabled = base.interleavedTemporalFusionEnabled,
                interleavedTemporalWindowMillis = base.interleavedTemporalWindowMillis,
                interleavedPredictionHorizonMillis = base.interleavedPredictionHorizonMillis,
                interleavedMaximumSampleSkewMillis = base.interleavedMaximumSampleSkewMillis,
                interleavedOutlierThresholdPixels = base.interleavedOutlierThresholdPixels,
                temporalSupersamplingPriority = base.temporalSupersamplingPriority,
                rollingShutterCompensation = base.rollingShutterCompensation,
                outputIntegrityStrictness = base.outputIntegrityStrictness,
                physicsConstraintStrictness = 0.98,
                global4DConsistencyStrictness = base.global4DConsistencyStrictness,
                surfaceEvidenceStrictness = 0.98,
                denseLandmarkCoverageStrictness = base.denseLandmarkCoverageStrictness,
                denseLandmarkUncertaintyStrictness = base.denseLandmarkUncertaintyStrictness,
                denseLandmarkCrossViewStrictness = base.denseLandmarkCrossViewStrictness,
                denseLandmarkTemporalStrictness = base.denseLandmarkTemporalStrictness,
                ultraPrecisionEnabled = base.ultraPrecisionEnabled,
                ultraPrecisionQualificationMode = base.ultraPrecisionQualificationMode,
                ultraPrecisionTargetP95ErrorMillimeters = base.ultraPrecisionTargetP95ErrorMillimeters,
                ultraPrecisionTargetP99ErrorMillimeters = base.ultraPrecisionTargetP99ErrorMillimeters,
                ultraPrecisionMinimumEvidenceSamples = base.ultraPrecisionMinimumEvidenceSamples,
                ultraPrecisionMinimumBaselineMeters = base.ultraPrecisionMinimumBaselineMeters,
                ultraPrecisionMinimumParallaxDegrees = base.ultraPrecisionMinimumParallaxDegrees,
                ultraPrecisionMinimumCalibrationConfidence = base.ultraPrecisionMinimumCalibrationConfidence,
                ultraPrecisionMinimumSynchronizationConfidence = base.ultraPrecisionMinimumSynchronizationConfidence,
                ultraPrecisionMinimumViewDiversity = base.ultraPrecisionMinimumViewDiversity,
                ultraPrecisionMinimumCrossViewAgreement = base.ultraPrecisionMinimumCrossViewAgreement,
                ultraPrecisionMaximumOcclusionRatio = base.ultraPrecisionMaximumOcclusionRatio,
                ultraPrecisionMinimumIdentityContinuity = base.ultraPrecisionMinimumIdentityContinuity,
                ultraPrecisionMinimumSurfaceTemporalStability = base.ultraPrecisionMinimumSurfaceTemporalStability,
                ultraPrecisionMaximumThermalRisk = base.ultraPrecisionMaximumThermalRisk,
                ultraPrecisionMaximumMemoryPressure = base.ultraPrecisionMaximumMemoryPressure,
                ultraPrecisionMatchedReferenceRequired = base.ultraPrecisionMatchedReferenceRequired,
                minimumParallaxDegrees = base.minimumParallaxDegrees,
                preferredTemporalSkewMillis = base.preferredTemporalSkewMillis,
                maximumTemporalSkewMillis = base.maximumTemporalSkewMillis,
                maximumReprojectionErrorPx = 2.5,
                maximumReconstructionUncertaintyMeters = 0.18,
                posePriorStrength = base.posePriorStrength,
                footContactStrictness = base.footContactStrictness,
                headPoseStrictness = 1.0,
                gazePriority = 1.0,
                depthTemporalFusion = base.depthTemporalFusion,
                cameraDriftResponse = base.cameraDriftResponse,
                thermalQualityFloor = base.thermalQualityFloor,
                networkJitterToleranceMillis = base.networkJitterToleranceMillis,
                meshAdaptiveDetail = base.meshAdaptiveDetail,
                retargetingStrictness = base.retargetingStrictness,
                scalePreservation = base.scalePreservation,
                occlusionRecoveryConfidence = base.occlusionRecoveryConfidence,
                landmarkConsistencyStrictness = 1.0,
                semanticMotionSensitivity = base.semanticMotionSensitivity,
                motionThresholdScale = base.motionThresholdScale,
                rootMotionLock = base.rootMotionLock,
                dynamicTopology = base.dynamicTopology,
                preserveMicroMotion = base.preserveMicroMotion,
                recordLightingState = base.recordLightingState,
                recordImuState = base.recordImuState,
                recordCameraTelemetry = base.recordCameraTelemetry,
                recordDepthMetadata = base.recordDepthMetadata,
                recordFaceEyeFusion = base.recordFaceEyeFusion,
                recordHand3D = base.recordHand3D,
                recordDenseMesh = base.recordDenseMesh,
                recordConfidence = base.recordConfidence,
                preferNpuForDensePerception = base.preferNpuForDensePerception,
                allowRelativeDepthWhenMetricUnavailable = base.allowRelativeDepthWhenMetricUnavailable,
                compactSession = false,
                correlatedEvidenceEnabled = base.correlatedEvidenceEnabled,
                correlatedEvidenceQualificationMode = base.correlatedEvidenceQualificationMode,
                correlatedEvidenceMinimumCoverage = base.correlatedEvidenceMinimumCoverage,
                identityExchangeRiskEnabled = base.identityExchangeRiskEnabled,
                identityExchangeHistoryFrames = base.identityExchangeHistoryFrames,
                lightingConfidence = base.lightingConfidence,
                distributedGlobalCameraEnvelope =
                    base.distributedGlobalCameraEnvelope,
                distributedAuthoritativeCameraBudget =
                    base.distributedAuthoritativeCameraBudget,
                distributedMaximumShards =
                    base.distributedMaximumShards,
                distributedMinimumAuthorityHealth =
                    base.distributedMinimumAuthorityHealth,
                distributedMinimumAuthoritySynchronization =
                    base.distributedMinimumAuthoritySynchronization,
                distributedMaximumTemporalResidualMs =
                    base.distributedMaximumTemporalResidualMs,
                distributedMaximumTemporalOutlierRatio =
                    base.distributedMaximumTemporalOutlierRatio,
                distributedMinimumTransportHeadroom =
                    base.distributedMinimumTransportHeadroom,
            )
    }
}


/**
 * Mutable copy builder for [MotionCaptureCustomization].
 *
 * This keeps named-property update ergonomics without relying on the JVM-generated
 * data-class copy/default machinery, which becomes unsafe for this large model.
 */
class MotionCaptureCustomizationCopyBuilder internal constructor(
    base: MotionCaptureCustomization,
) {
    var mode: MotionCaptureMode = base.mode
    var cameraResolutionPreset: CameraResolutionPreset = base.cameraResolutionPreset
    var cameraDynamicRangeMode: CameraDynamicRangeMode =
        base.cameraDynamicRangeMode
    var cameraExposureMode: CameraExposureMode = base.cameraExposureMode
    var cameraAntibandingMode: CameraAntibandingMode =
        base.cameraAntibandingMode
    var cameraFocusMode: CameraFocusMode = base.cameraFocusMode
    var cameraManualExposureMicros: Long = base.cameraManualExposureMicros
    var cameraManualIso: Int = base.cameraManualIso
    var cameraExposureCompensationEv: Double = base.cameraExposureCompensationEv
    var cameraAutoExposureLock: Boolean = base.cameraAutoExposureLock
    var cameraAutoWhiteBalanceLock: Boolean = base.cameraAutoWhiteBalanceLock
    var cameraManualFocusDistanceDiopters: Double? = base.cameraManualFocusDistanceDiopters
    var cameraPreciseColorTemperatureEnabled: Boolean =
        base.cameraPreciseColorTemperatureEnabled
    var cameraColorTemperatureKelvin: Int =
        base.cameraColorTemperatureKelvin
    var cameraColorTint: Int =
        base.cameraColorTint

    var cameraOisEnabled: Boolean = base.cameraOisEnabled
    var cameraElectronicStabilizationEnabled: Boolean = base.cameraElectronicStabilizationEnabled
    var liveMotionPreviewEnabled: Boolean = base.liveMotionPreviewEnabled
    var liveMotionPreviewGpuMode: MotionFieldGpuMode = base.liveMotionPreviewGpuMode
    var liveMotionOverlayBody: Boolean = base.liveMotionOverlayBody
    var liveMotionOverlayFace: Boolean = base.liveMotionOverlayFace
    var liveMotionOverlayEyes: Boolean = base.liveMotionOverlayEyes
    var liveMotionOverlayDepth: Boolean = base.liveMotionOverlayDepth
    var liveMotionOverlayMicroMotion: Boolean = base.liveMotionOverlayMicroMotion
    var liveMotionOverlayVelocity: Boolean = base.liveMotionOverlayVelocity
    var liveMotionOverlayUncertainty: Boolean = base.liveMotionOverlayUncertainty
    var liveMotionOverlayActorIds: Boolean = base.liveMotionOverlayActorIds
    var liveMotionPreviewPredictionHorizonMillis: Double = base.liveMotionPreviewPredictionHorizonMillis
    var liveMotionPreviewMaxStaleMillis: Double = base.liveMotionPreviewMaxStaleMillis
    var liveMotionPreviewDetail: Double = base.liveMotionPreviewDetail
    var captureEvidenceMode: CaptureEvidenceMode = base.captureEvidenceMode
    var recordedVideoEvidenceProfile: RecordedVideoEvidenceProfile? = base.recordedVideoEvidenceProfile
    var postCaptureReconstructionPolicy: PostCaptureReconstructionPolicy = base.postCaptureReconstructionPolicy
    var postCaptureAutoReconstructionEnabled: Boolean = base.postCaptureAutoReconstructionEnabled
    var multiPersonCapacityPolicy: MultiPersonCapacityPolicy = base.multiPersonCapacityPolicy
    var cameraRawSidecarEnabled: Boolean = base.cameraRawSidecarEnabled
    var cameraDepthSidecarEnabled: Boolean = base.cameraDepthSidecarEnabled
    var cameraPreserveSensorTimestamp: Boolean = base.cameraPreserveSensorTimestamp
    var targetFps: Int = base.targetFps
    var inferenceFps: Int = base.inferenceFps
    var maxSubjects: Int = base.maxSubjects
    var actorDetailPriorityMode: ActorDetailPriorityMode = base.actorDetailPriorityMode
    var fullDetailActorBudget: Int = base.fullDetailActorBudget
    var faceDetailActorBudget: Int = base.faceDetailActorBudget
    var eyeDetailActorBudget: Int = base.eyeDetailActorBudget
    var handDetailActorBudget: Int = base.handDetailActorBudget
    var denseSurfaceActorBudget: Int = base.denseSurfaceActorBudget
    var priorityDetailRetention: Double = base.priorityDetailRetention
    var actorIdentityStrictness: Double = base.actorIdentityStrictness
    var actorOcclusionRecovery: Double = base.actorOcclusionRecovery
    var crowdedSceneOverlapThreshold: Double = base.crowdedSceneOverlapThreshold
    var crowdedSceneMatchScoreBoost: Double = base.crowdedSceneMatchScoreBoost
    var actorCrossCameraAssociation: Double = base.actorCrossCameraAssociation
    var actorMicroMotionSensitivity: Double = base.actorMicroMotionSensitivity
    var motionFrequencyBandSensitivity: Double = base.motionFrequencyBandSensitivity
    var actorMicroMotionRetention: Double = base.actorMicroMotionRetention
    var actorSubframeTimingStrictness: Double = base.actorSubframeTimingStrictness
    var bodyDetail: Double = base.bodyDetail
    var faceDetail: Double = base.faceDetail
    var eyeDetail: Double = base.eyeDetail
    var handDetail: Double = base.handDetail
    var depthDetail: Double = base.depthDetail
    var objectDetail: Double = base.objectDetail
    var denseMeshDetail: Double = base.denseMeshDetail
    var personSegmentationEnabled: Boolean = base.personSegmentationEnabled
    var personSegmentationCadenceFrames: Int = base.personSegmentationCadenceFrames
    var personSegmentationPriority: Double = base.personSegmentationPriority
    var contactDetail: Double = base.contactDetail
    var worldConsistencyDetail: Double = base.worldConsistencyDetail
    var topologyConstraintStrength: Double = base.topologyConstraintStrength
    var multiViewFusionWeight: Double = base.multiViewFusionWeight
    var multiViewEvidenceAdaptivity: Double = base.multiViewEvidenceAdaptivity
    var reconstructionEvidenceFusionStrength: Double = base.reconstructionEvidenceFusionStrength
    var reconstructionEvidenceDisagreementToleranceMeters: Double = base.reconstructionEvidenceDisagreementToleranceMeters
    var reconstructionEvidenceMinimumSourceDiversity: Double = base.reconstructionEvidenceMinimumSourceDiversity
    var adaptiveDetectorIntervalFrames: Int = base.adaptiveDetectorIntervalFrames
    var maximumOcclusionPredictionFrames: Int = base.maximumOcclusionPredictionFrames
    var temporalSmoothing: Double = base.temporalSmoothing
    var occlusionPersistence: Double = base.occlusionPersistence
    var microMotionPreservation: Double = base.microMotionPreservation
    var cameraMotionSuppression: Double = base.cameraMotionSuppression
    var faceEyeFusionWeight: Double = base.faceEyeFusionWeight
    var faceEyeFusionStrictness: Double = base.faceEyeFusionStrictness
    var faceEyeIpdDriftToleranceMillimeters: Double = base.faceEyeIpdDriftToleranceMillimeters
    var eyeTemporalStrictness: Double = base.eyeTemporalStrictness
    var handTemporalStrictness: Double = base.handTemporalStrictness
    var handGeometryStrictness: Double = base.handGeometryStrictness
    var handLongitudinalAnatomyStrictness: Double = base.handLongitudinalAnatomyStrictness
    var temporalAlignmentStrictness: Double = base.temporalAlignmentStrictness
    var subframeAlignmentStrictness: Double = base.subframeAlignmentStrictness
    var interleavedTemporalFusionEnabled: Boolean = base.interleavedTemporalFusionEnabled
    var interleavedTemporalWindowMillis: Double = base.interleavedTemporalWindowMillis
    var interleavedPredictionHorizonMillis: Double = base.interleavedPredictionHorizonMillis
    var interleavedMaximumSampleSkewMillis: Double = base.interleavedMaximumSampleSkewMillis
    var interleavedOutlierThresholdPixels: Double = base.interleavedOutlierThresholdPixels
    var photometricExposureShiftSoftLimit: Double =
        base.photometricExposureShiftSoftLimit
    var photometricExposureShiftHardLimit: Double =
        base.photometricExposureShiftHardLimit
    var photometricExposureReliabilityFloor: Double =
        base.photometricExposureReliabilityFloor
    var temporalSupersamplingPriority: Double = base.temporalSupersamplingPriority
    var rollingShutterCompensation: Double = base.rollingShutterCompensation
    var outputIntegrityStrictness: Double = base.outputIntegrityStrictness
    var physicsConstraintStrictness: Double = base.physicsConstraintStrictness
    var global4DConsistencyStrictness: Double = base.global4DConsistencyStrictness
    var surfaceEvidenceStrictness: Double = base.surfaceEvidenceStrictness
    var denseLandmarkCoverageStrictness: Double = base.denseLandmarkCoverageStrictness
    var denseLandmarkUncertaintyStrictness: Double = base.denseLandmarkUncertaintyStrictness
    var denseLandmarkCrossViewStrictness: Double = base.denseLandmarkCrossViewStrictness
    var denseLandmarkTemporalStrictness: Double = base.denseLandmarkTemporalStrictness
    var ultraPrecisionEnabled: Boolean = base.ultraPrecisionEnabled
    var ultraPrecisionQualificationMode: Boolean = base.ultraPrecisionQualificationMode
    var ultraPrecisionTargetP95ErrorMillimeters: Double = base.ultraPrecisionTargetP95ErrorMillimeters
    var ultraPrecisionTargetP99ErrorMillimeters: Double = base.ultraPrecisionTargetP99ErrorMillimeters
    var ultraPrecisionMinimumEvidenceSamples: Int = base.ultraPrecisionMinimumEvidenceSamples
    var ultraPrecisionMinimumBaselineMeters: Double = base.ultraPrecisionMinimumBaselineMeters
    var ultraPrecisionMinimumParallaxDegrees: Double = base.ultraPrecisionMinimumParallaxDegrees
    var ultraPrecisionMinimumCalibrationConfidence: Double = base.ultraPrecisionMinimumCalibrationConfidence
    var ultraPrecisionMinimumSynchronizationConfidence: Double = base.ultraPrecisionMinimumSynchronizationConfidence
    var ultraPrecisionMinimumViewDiversity: Double = base.ultraPrecisionMinimumViewDiversity
    var ultraPrecisionMinimumCrossViewAgreement: Double = base.ultraPrecisionMinimumCrossViewAgreement
    var ultraPrecisionMaximumOcclusionRatio: Double = base.ultraPrecisionMaximumOcclusionRatio
    var ultraPrecisionMinimumIdentityContinuity: Double = base.ultraPrecisionMinimumIdentityContinuity
    var ultraPrecisionMinimumSurfaceTemporalStability: Double = base.ultraPrecisionMinimumSurfaceTemporalStability
    var ultraPrecisionMaximumThermalRisk: Double = base.ultraPrecisionMaximumThermalRisk
    var ultraPrecisionMaximumMemoryPressure: Double = base.ultraPrecisionMaximumMemoryPressure
    var ultraPrecisionMatchedReferenceRequired: Boolean = base.ultraPrecisionMatchedReferenceRequired
    var minimumParallaxDegrees: Double = base.minimumParallaxDegrees
    var preferredTemporalSkewMillis: Double = base.preferredTemporalSkewMillis
    var maximumTemporalSkewMillis: Double = base.maximumTemporalSkewMillis
    var maximumReprojectionErrorPx: Double = base.maximumReprojectionErrorPx
    var maximumReconstructionUncertaintyMeters: Double = base.maximumReconstructionUncertaintyMeters
    var posePriorStrength: Double = base.posePriorStrength
    var footContactStrictness: Double = base.footContactStrictness
    var headPoseStrictness: Double = base.headPoseStrictness
    var gazePriority: Double = base.gazePriority
    var depthTemporalFusion: Double = base.depthTemporalFusion
    var cameraDriftResponse: Double = base.cameraDriftResponse
    var thermalQualityFloor: Double = base.thermalQualityFloor
    var networkJitterToleranceMillis: Double = base.networkJitterToleranceMillis
    var meshAdaptiveDetail: Double = base.meshAdaptiveDetail
    var retargetingStrictness: Double = base.retargetingStrictness
    var scalePreservation: Double = base.scalePreservation
    var occlusionRecoveryConfidence: Double = base.occlusionRecoveryConfidence
    var landmarkConsistencyStrictness: Double = base.landmarkConsistencyStrictness
    var semanticMotionSensitivity: Double = base.semanticMotionSensitivity
    var motionThresholdScale: Double = base.motionThresholdScale
    var rootMotionLock: Double = base.rootMotionLock
    var dynamicTopology: Boolean = base.dynamicTopology
    var preserveMicroMotion: Boolean = base.preserveMicroMotion
    var recordLightingState: Boolean = base.recordLightingState
    var recordImuState: Boolean = base.recordImuState
    var recordCameraTelemetry: Boolean = base.recordCameraTelemetry
    var recordDepthMetadata: Boolean = base.recordDepthMetadata
    var recordFaceEyeFusion: Boolean = base.recordFaceEyeFusion
    var recordHand3D: Boolean = base.recordHand3D
    var recordDenseMesh: Boolean = base.recordDenseMesh
    var recordConfidence: Boolean = base.recordConfidence
    var preferNpuForDensePerception: Boolean = base.preferNpuForDensePerception
    var allowRelativeDepthWhenMetricUnavailable: Boolean = base.allowRelativeDepthWhenMetricUnavailable
    var compactSession: Boolean = base.compactSession
    var correlatedEvidenceEnabled: Boolean = base.correlatedEvidenceEnabled
    var correlatedEvidenceQualificationMode: Boolean = base.correlatedEvidenceQualificationMode
    var correlatedEvidenceMinimumCoverage: Double = base.correlatedEvidenceMinimumCoverage
    var identityExchangeRiskEnabled: Boolean = base.identityExchangeRiskEnabled
    var identityExchangeHistoryFrames: Int = base.identityExchangeHistoryFrames
    var lightingConfidence: Double = base.lightingConfidence
    var distributedGlobalCameraEnvelope: Int = base.distributedGlobalCameraEnvelope
    var distributedAuthoritativeCameraBudget: Int = base.distributedAuthoritativeCameraBudget
    var distributedMaximumShards: Int = base.distributedMaximumShards
    var distributedMinimumAuthorityHealth: Double = base.distributedMinimumAuthorityHealth
    var distributedMinimumAuthoritySynchronization: Double = base.distributedMinimumAuthoritySynchronization
    var distributedMaximumTemporalResidualMs: Double = base.distributedMaximumTemporalResidualMs
    var distributedMaximumTemporalOutlierRatio: Double = base.distributedMaximumTemporalOutlierRatio
    var distributedMinimumTransportHeadroom: Double = base.distributedMinimumTransportHeadroom

    fun build(): MotionCaptureCustomization =
        MotionCaptureCustomization(
            mode = mode,
            cameraResolutionPreset = cameraResolutionPreset,
            cameraDynamicRangeMode = cameraDynamicRangeMode,
            cameraExposureMode = cameraExposureMode,
            cameraAntibandingMode = cameraAntibandingMode,
            cameraFocusMode = cameraFocusMode,
            cameraManualExposureMicros = cameraManualExposureMicros,
            cameraManualIso = cameraManualIso,
            cameraExposureCompensationEv = cameraExposureCompensationEv,
            cameraAutoExposureLock = cameraAutoExposureLock,
            cameraAutoWhiteBalanceLock = cameraAutoWhiteBalanceLock,
            cameraPreciseColorTemperatureEnabled =
                cameraPreciseColorTemperatureEnabled,
            cameraColorTemperatureKelvin =
                cameraColorTemperatureKelvin,
            cameraColorTint =
                cameraColorTint,
            cameraManualFocusDistanceDiopters = cameraManualFocusDistanceDiopters,
            cameraOisEnabled = cameraOisEnabled,
            cameraElectronicStabilizationEnabled = cameraElectronicStabilizationEnabled,
            liveMotionPreviewEnabled = liveMotionPreviewEnabled,
            liveMotionPreviewGpuMode = liveMotionPreviewGpuMode,
            liveMotionOverlayBody = liveMotionOverlayBody,
            liveMotionOverlayFace = liveMotionOverlayFace,
            liveMotionOverlayEyes = liveMotionOverlayEyes,
            liveMotionOverlayDepth = liveMotionOverlayDepth,
            liveMotionOverlayMicroMotion = liveMotionOverlayMicroMotion,
            liveMotionOverlayVelocity = liveMotionOverlayVelocity,
            liveMotionOverlayUncertainty = liveMotionOverlayUncertainty,
            liveMotionOverlayActorIds = liveMotionOverlayActorIds,
            liveMotionPreviewPredictionHorizonMillis = liveMotionPreviewPredictionHorizonMillis,
            liveMotionPreviewMaxStaleMillis = liveMotionPreviewMaxStaleMillis,
            liveMotionPreviewDetail = liveMotionPreviewDetail,
            captureEvidenceMode = captureEvidenceMode,
            recordedVideoEvidenceProfile = recordedVideoEvidenceProfile,
            postCaptureReconstructionPolicy = postCaptureReconstructionPolicy,
            postCaptureAutoReconstructionEnabled = postCaptureAutoReconstructionEnabled,
            multiPersonCapacityPolicy = multiPersonCapacityPolicy,
            cameraRawSidecarEnabled = cameraRawSidecarEnabled,
            cameraDepthSidecarEnabled = cameraDepthSidecarEnabled,
            cameraPreserveSensorTimestamp = cameraPreserveSensorTimestamp,
            targetFps = targetFps,
            inferenceFps = inferenceFps,
            maxSubjects = maxSubjects,
            actorDetailPriorityMode = actorDetailPriorityMode,
            fullDetailActorBudget = fullDetailActorBudget,
            faceDetailActorBudget = faceDetailActorBudget,
            eyeDetailActorBudget = eyeDetailActorBudget,
            handDetailActorBudget = handDetailActorBudget,
            denseSurfaceActorBudget = denseSurfaceActorBudget,
            priorityDetailRetention = priorityDetailRetention,
            actorIdentityStrictness = actorIdentityStrictness,
            actorOcclusionRecovery = actorOcclusionRecovery,
            crowdedSceneOverlapThreshold = crowdedSceneOverlapThreshold,
            crowdedSceneMatchScoreBoost = crowdedSceneMatchScoreBoost,
            actorCrossCameraAssociation = actorCrossCameraAssociation,
            actorMicroMotionSensitivity = actorMicroMotionSensitivity,
            motionFrequencyBandSensitivity = motionFrequencyBandSensitivity,
            actorMicroMotionRetention = actorMicroMotionRetention,
            actorSubframeTimingStrictness = actorSubframeTimingStrictness,
            bodyDetail = bodyDetail,
            faceDetail = faceDetail,
            eyeDetail = eyeDetail,
            handDetail = handDetail,
            depthDetail = depthDetail,
            objectDetail = objectDetail,
            denseMeshDetail = denseMeshDetail,
            personSegmentationEnabled = personSegmentationEnabled,
            personSegmentationCadenceFrames = personSegmentationCadenceFrames,
            personSegmentationPriority = personSegmentationPriority,
            contactDetail = contactDetail,
            worldConsistencyDetail = worldConsistencyDetail,
            topologyConstraintStrength = topologyConstraintStrength,
            multiViewFusionWeight = multiViewFusionWeight,
            multiViewEvidenceAdaptivity = multiViewEvidenceAdaptivity,
            reconstructionEvidenceFusionStrength = reconstructionEvidenceFusionStrength,
            reconstructionEvidenceDisagreementToleranceMeters = reconstructionEvidenceDisagreementToleranceMeters,
            reconstructionEvidenceMinimumSourceDiversity = reconstructionEvidenceMinimumSourceDiversity,
            adaptiveDetectorIntervalFrames = adaptiveDetectorIntervalFrames,
            maximumOcclusionPredictionFrames = maximumOcclusionPredictionFrames,
            temporalSmoothing = temporalSmoothing,
            occlusionPersistence = occlusionPersistence,
            microMotionPreservation = microMotionPreservation,
            cameraMotionSuppression = cameraMotionSuppression,
            faceEyeFusionWeight = faceEyeFusionWeight,
            faceEyeFusionStrictness = faceEyeFusionStrictness,
            faceEyeIpdDriftToleranceMillimeters = faceEyeIpdDriftToleranceMillimeters,
            eyeTemporalStrictness = eyeTemporalStrictness,
            handTemporalStrictness = handTemporalStrictness,
            handGeometryStrictness = handGeometryStrictness,
            handLongitudinalAnatomyStrictness = handLongitudinalAnatomyStrictness,
            temporalAlignmentStrictness = temporalAlignmentStrictness,
            subframeAlignmentStrictness = subframeAlignmentStrictness,
            interleavedTemporalFusionEnabled = interleavedTemporalFusionEnabled,
            interleavedTemporalWindowMillis = interleavedTemporalWindowMillis,
            interleavedPredictionHorizonMillis = interleavedPredictionHorizonMillis,
            interleavedMaximumSampleSkewMillis = interleavedMaximumSampleSkewMillis,
            interleavedOutlierThresholdPixels = interleavedOutlierThresholdPixels,
            photometricExposureShiftSoftLimit =
                photometricExposureShiftSoftLimit,
            photometricExposureShiftHardLimit =
                photometricExposureShiftHardLimit,
            photometricExposureReliabilityFloor =
                photometricExposureReliabilityFloor,
            temporalSupersamplingPriority = temporalSupersamplingPriority,
            rollingShutterCompensation = rollingShutterCompensation,
            outputIntegrityStrictness = outputIntegrityStrictness,
            physicsConstraintStrictness = physicsConstraintStrictness,
            global4DConsistencyStrictness = global4DConsistencyStrictness,
            surfaceEvidenceStrictness = surfaceEvidenceStrictness,
            denseLandmarkCoverageStrictness = denseLandmarkCoverageStrictness,
            denseLandmarkUncertaintyStrictness = denseLandmarkUncertaintyStrictness,
            denseLandmarkCrossViewStrictness = denseLandmarkCrossViewStrictness,
            denseLandmarkTemporalStrictness = denseLandmarkTemporalStrictness,
            ultraPrecisionEnabled = ultraPrecisionEnabled,
            ultraPrecisionQualificationMode = ultraPrecisionQualificationMode,
            ultraPrecisionTargetP95ErrorMillimeters = ultraPrecisionTargetP95ErrorMillimeters,
            ultraPrecisionTargetP99ErrorMillimeters = ultraPrecisionTargetP99ErrorMillimeters,
            ultraPrecisionMinimumEvidenceSamples = ultraPrecisionMinimumEvidenceSamples,
            ultraPrecisionMinimumBaselineMeters = ultraPrecisionMinimumBaselineMeters,
            ultraPrecisionMinimumParallaxDegrees = ultraPrecisionMinimumParallaxDegrees,
            ultraPrecisionMinimumCalibrationConfidence = ultraPrecisionMinimumCalibrationConfidence,
            ultraPrecisionMinimumSynchronizationConfidence = ultraPrecisionMinimumSynchronizationConfidence,
            ultraPrecisionMinimumViewDiversity = ultraPrecisionMinimumViewDiversity,
            ultraPrecisionMinimumCrossViewAgreement = ultraPrecisionMinimumCrossViewAgreement,
            ultraPrecisionMaximumOcclusionRatio = ultraPrecisionMaximumOcclusionRatio,
            ultraPrecisionMinimumIdentityContinuity = ultraPrecisionMinimumIdentityContinuity,
            ultraPrecisionMinimumSurfaceTemporalStability = ultraPrecisionMinimumSurfaceTemporalStability,
            ultraPrecisionMaximumThermalRisk = ultraPrecisionMaximumThermalRisk,
            ultraPrecisionMaximumMemoryPressure = ultraPrecisionMaximumMemoryPressure,
            ultraPrecisionMatchedReferenceRequired = ultraPrecisionMatchedReferenceRequired,
            minimumParallaxDegrees = minimumParallaxDegrees,
            preferredTemporalSkewMillis = preferredTemporalSkewMillis,
            maximumTemporalSkewMillis = maximumTemporalSkewMillis,
            maximumReprojectionErrorPx = maximumReprojectionErrorPx,
            maximumReconstructionUncertaintyMeters = maximumReconstructionUncertaintyMeters,
            posePriorStrength = posePriorStrength,
            footContactStrictness = footContactStrictness,
            headPoseStrictness = headPoseStrictness,
            gazePriority = gazePriority,
            depthTemporalFusion = depthTemporalFusion,
            cameraDriftResponse = cameraDriftResponse,
            thermalQualityFloor = thermalQualityFloor,
            networkJitterToleranceMillis = networkJitterToleranceMillis,
            meshAdaptiveDetail = meshAdaptiveDetail,
            retargetingStrictness = retargetingStrictness,
            scalePreservation = scalePreservation,
            occlusionRecoveryConfidence = occlusionRecoveryConfidence,
            landmarkConsistencyStrictness = landmarkConsistencyStrictness,
            semanticMotionSensitivity = semanticMotionSensitivity,
            motionThresholdScale = motionThresholdScale,
            rootMotionLock = rootMotionLock,
            dynamicTopology = dynamicTopology,
            preserveMicroMotion = preserveMicroMotion,
            recordLightingState = recordLightingState,
            recordImuState = recordImuState,
            recordCameraTelemetry = recordCameraTelemetry,
            recordDepthMetadata = recordDepthMetadata,
            recordFaceEyeFusion = recordFaceEyeFusion,
            recordHand3D = recordHand3D,
            recordDenseMesh = recordDenseMesh,
            recordConfidence = recordConfidence,
            preferNpuForDensePerception = preferNpuForDensePerception,
            allowRelativeDepthWhenMetricUnavailable = allowRelativeDepthWhenMetricUnavailable,
            compactSession = compactSession,
            correlatedEvidenceEnabled = correlatedEvidenceEnabled,
            correlatedEvidenceQualificationMode = correlatedEvidenceQualificationMode,
            correlatedEvidenceMinimumCoverage = correlatedEvidenceMinimumCoverage,
            identityExchangeRiskEnabled = identityExchangeRiskEnabled,
            identityExchangeHistoryFrames = identityExchangeHistoryFrames,
            lightingConfidence = lightingConfidence,
            distributedGlobalCameraEnvelope = distributedGlobalCameraEnvelope,
            distributedAuthoritativeCameraBudget = distributedAuthoritativeCameraBudget,
            distributedMaximumShards = distributedMaximumShards,
            distributedMinimumAuthorityHealth = distributedMinimumAuthorityHealth,
            distributedMinimumAuthoritySynchronization = distributedMinimumAuthoritySynchronization,
            distributedMaximumTemporalResidualMs = distributedMaximumTemporalResidualMs,
            distributedMaximumTemporalOutlierRatio = distributedMaximumTemporalOutlierRatio,
            distributedMinimumTransportHeadroom = distributedMinimumTransportHeadroom,
        )
}

data class MotionSessionHeaderV2(
    val schema: Int = 2,
    val sessionId: SessionId,
    val mode: MotionCaptureMode,
    val subjectType: SubjectType,
    val topologyKind: TopologyKind,
    val createdAtNanos: Long,
    val targetFps: Int,
    val calibrationVersion: String,
    val modelProfile: String,
    val motionOnly: Boolean = true,
)

data class MotionSampleV2(
    val timestampNanos: Long,
    val trackId: Long,
    val nodeId: Int,
    val position: Vec3,
    val rotation: Quat,
    val linearVelocity: Vec3,
    val angularVelocity: Vec3,
    val confidence: Float,
    val uncertaintyMeters: Float,
    val predicted: Boolean,
)

data class MotionQualityGate(
    val accepted: Boolean,
    val confidenceBand: ReconstructionConfidenceBand,
    val reasons: List<String>,
    val confidence: Double,
    val maxUncertaintyMeters: Double,
    val reprojectionErrorPx: Double,
    val temporalStability: Double,
) {
    companion object {
        fun evaluate(
            confidence: Double,
            uncertaintyMeters: Double,
            reprojectionErrorPx: Double,
            temporalStability: Double,
            hardFailureReasons: List<String> = emptyList(),
        ): MotionQualityGate {
            val reasons = hardFailureReasons.toMutableList()
            if (confidence < 0.35) reasons += "global confidence below capture gate"
            if (uncertaintyMeters > 0.35) reasons += "metric uncertainty exceeds 35 cm"
            if (reprojectionErrorPx > 8.0) reasons += "reprojection residual exceeds 8 px"
            if (temporalStability < 0.35) reasons += "temporal stability below gate"

            val band = when {
                confidence >= 0.90 && uncertaintyMeters <= 0.10 &&
                    reprojectionErrorPx <= 2.0 && temporalStability >= 0.88 ->
                    ReconstructionConfidenceBand.EXCELLENT

                confidence >= 0.75 && uncertaintyMeters <= 0.18 &&
                    reprojectionErrorPx <= 4.0 && temporalStability >= 0.72 ->
                    ReconstructionConfidenceBand.GOOD

                confidence >= 0.55 && uncertaintyMeters <= 0.30 &&
                    reprojectionErrorPx <= 6.0 && temporalStability >= 0.52 ->
                    ReconstructionConfidenceBand.USABLE

                confidence >= 0.35 ->
                    ReconstructionConfidenceBand.DEGRADED

                else ->
                    ReconstructionConfidenceBand.UNTRUSTED
            }

            return MotionQualityGate(
                accepted = reasons.isEmpty() && band != ReconstructionConfidenceBand.UNTRUSTED,
                confidenceBand = band,
                reasons = reasons.distinct(),
                confidence = confidence.coerceIn(0.0, 1.0),
                maxUncertaintyMeters = uncertaintyMeters.coerceAtLeast(0.0),
                reprojectionErrorPx = reprojectionErrorPx.coerceAtLeast(0.0),
                temporalStability = temporalStability.coerceIn(0.0, 1.0),
            )
        }
    }
}


/**
 * High-end perception contracts.
 *
 * These records keep body, face, eye, hand, lighting, and camera evidence independent while
 * allowing the fusion layer to reason about freshness, uncertainty, and provenance.
 */

enum class PerceptionSource {
    ML_KIT,
    MEDIAPIPE,
    ONNX,
    TFLITE,
    NPU,
    GPU,
    SENSOR,
    MULTI_VIEW,
    TEMPORAL_PREDICTION,
    USER_CALIBRATION,
    UNKNOWN,
}

enum class GazeConfidenceBand {
    EXCELLENT,
    GOOD,
    LIMITED,
    LOST,
}

enum class LightingStabilityBand {
    STABLE,
    CHANGING,
    CHALLENGING,
    UNKNOWN,
}

data class FaceExpressionState(
    val timestampNanos: Long,
    val trackId: Long,
    val landmarkCount: Int,
    val headRotation: Quat,
    val headTranslationMeters: Vec3,
    val mouthOpen: Double,
    val leftBrowRaise: Double,
    val rightBrowRaise: Double,
    val leftEyeClosure: Double,
    val rightEyeClosure: Double,
    val smile: Double,
    val expressionEnergy: Double,
    val confidence: Double,
    val source: PerceptionSource,
    val fresh: Boolean = true,
) {
    init {
        require(landmarkCount >= 0)
        require(mouthOpen in 0.0..1.0)
        require(leftBrowRaise in 0.0..1.0)
        require(rightBrowRaise in 0.0..1.0)
        require(leftEyeClosure in 0.0..1.0)
        require(rightEyeClosure in 0.0..1.0)
        require(smile in 0.0..1.0)
        require(expressionEnergy in 0.0..1.0)
        require(confidence in 0.0..1.0)
    }
}

data class EyeState(
    val timestampNanos: Long,
    val trackId: Long,
    val leftGazeCamera: Vec3,
    val rightGazeCamera: Vec3,
    val fusedGazeCamera: Vec3,
    val leftOpenness: Double,
    val rightOpenness: Double,
    val leftBlinkProbability: Double,
    val rightBlinkProbability: Double,
    val vergenceMeters: Double?,
    val saccadeVelocityDegPerSecond: Double,
    val gazeStability: Double,
    val confidence: Double,
    val band: GazeConfidenceBand,
    val source: PerceptionSource,
    val fresh: Boolean = true,
) {
    init {
        require(leftOpenness in 0.0..1.0)
        require(rightOpenness in 0.0..1.0)
        require(leftBlinkProbability in 0.0..1.0)
        require(rightBlinkProbability in 0.0..1.0)
        require(saccadeVelocityDegPerSecond >= 0.0)
        require(gazeStability in 0.0..1.0)
        require(confidence in 0.0..1.0)
        if (vergenceMeters != null) require(vergenceMeters > 0.0)
    }

    val blinkProbability: Double
        get() = maxOf(leftBlinkProbability, rightBlinkProbability)

    val bothEyesOpen: Boolean
        get() = leftOpenness > 0.35 && rightOpenness > 0.35
}

data class HandState(
    val timestampNanos: Long,
    val trackId: Long,
    val side: String,
    val wrist: Vec3,
    val palmNormal: Vec3,
    val fingerTips: Map<String, Vec3>,
    val pinchStrength: Double,
    val gripStrength: Double,
    val articulationEnergy: Double,
    val confidence: Double,
    val source: PerceptionSource,
    val fresh: Boolean = true,
) {
    init {
        require(side == "LEFT" || side == "RIGHT" || side == "UNKNOWN")
        require(pinchStrength in 0.0..1.0)
        require(gripStrength in 0.0..1.0)
        require(articulationEnergy in 0.0..1.0)
        require(confidence in 0.0..1.0)
    }
}

data class LightingState(
    val timestampNanos: Long,
    val lux: Double,
    val correlatedColorTemperatureKelvin: Double,
    val exposureCompensation: Double,
    val highlightRisk: Double,
    val shadowRisk: Double,
    val flickerRisk: Double,
    val dynamicRangePressure: Double,
    val stability: LightingStabilityBand,
    val confidence: Double,
    val source: PerceptionSource = PerceptionSource.SENSOR,
) {
    init {
        require(lux >= 0.0)
        require(correlatedColorTemperatureKelvin > 0.0)
        require(highlightRisk in 0.0..1.0)
        require(shadowRisk in 0.0..1.0)
        require(flickerRisk in 0.0..1.0)
        require(dynamicRangePressure in 0.0..1.0)
        require(confidence in 0.0..1.0)
    }
}

data class CameraTelemetryState(
    val timestampNanos: Long,
    val sensorTimestampNanos: Long,
    val exposureTimeNanos: Long,
    val sensitivityIso: Int,
    val focusDistanceDiopters: Double,
    val focusConfidence: Double,
    val aperture: Double?,
    val focalLengthMillimeters: Double?,
    val rollingShutterReadoutNanos: Long?,
    val frameDurationNanos: Long,
    val zoomRatio: Double,
    val stabilizationActive: Boolean,
    val cameraMotionRisk: Double,
) {
    init {
        require(exposureTimeNanos >= 0L)
        require(sensitivityIso >= 0)
        require(focusDistanceDiopters >= 0.0)
        require(focusConfidence in 0.0..1.0)
        require(frameDurationNanos > 0L)
        require(zoomRatio > 0.0)
        require(cameraMotionRisk in 0.0..1.0)
        if (aperture != null) require(aperture > 0.0)
        if (focalLengthMillimeters != null) require(focalLengthMillimeters > 0.0)
        if (rollingShutterReadoutNanos != null) require(rollingShutterReadoutNanos >= 0L)
    }
}

data class MultiSubjectPolicy(
    val enabled: Boolean = true,
    val maximumSubjects: Int = 50,
    val hardSubjectCeiling: Int = 64,
    val retainLostTrackFrames: Int = 24,
    val relocalizationIntervalFrames: Int = 12,
    val minimumIdentityConfidence: Double = 0.42,
    val minimumAssociationIoU: Double = 0.10,
    val allowUnknownObjects: Boolean = true,
    val lockToDominantSubject: Boolean = false,
) {
    init {
        require(maximumSubjects in 1..64)
        require(hardSubjectCeiling in maximumSubjects..64)
        require(retainLostTrackFrames >= 1)
        require(relocalizationIntervalFrames >= 1)
        require(minimumIdentityConfidence in 0.0..1.0)
        require(minimumAssociationIoU in 0.0..1.0)
    }
}

data class ReconstructionTuning(
    val solverIterations: Int = 12,
    val temporalWindowFrames: Int = 8,
    val maxPredictionFrames: Int = 6,
    val boneLengthStiffness: Double = 0.82,
    val jointLimitStiffness: Double = 0.74,
    val depthPriorWeight: Double = 0.60,
    val cameraMotionCompensationWeight: Double = 0.72,
    val occlusionRecoveryWeight: Double = 0.68,
    val microMotionPreservation: Double = 0.88,
    val jitterSuppression: Double = 0.54,
    val outlierSigma: Double = 3.5,
) {
    init {
        require(solverIterations in 1..128)
        require(temporalWindowFrames in 1..64)
        require(maxPredictionFrames in 0..30)
        require(boneLengthStiffness in 0.0..1.0)
        require(jointLimitStiffness in 0.0..1.0)
        require(depthPriorWeight in 0.0..1.0)
        require(cameraMotionCompensationWeight in 0.0..1.0)
        require(occlusionRecoveryWeight in 0.0..1.0)
        require(microMotionPreservation in 0.0..1.0)
        require(jitterSuppression in 0.0..1.0)
        require(outlierSigma > 0.0)
    }

    companion object {
        fun cinematic() = ReconstructionTuning(
            solverIterations = 18,
            temporalWindowFrames = 12,
            maxPredictionFrames = 5,
            boneLengthStiffness = 0.88,
            jointLimitStiffness = 0.82,
            depthPriorWeight = 0.68,
            cameraMotionCompensationWeight = 0.82,
            occlusionRecoveryWeight = 0.76,
            microMotionPreservation = 0.94,
            jitterSuppression = 0.42,
        )

        fun action() = ReconstructionTuning(
            solverIterations = 10,
            temporalWindowFrames = 6,
            maxPredictionFrames = 4,
            boneLengthStiffness = 0.76,
            jointLimitStiffness = 0.72,
            depthPriorWeight = 0.54,
            cameraMotionCompensationWeight = 0.86,
            occlusionRecoveryWeight = 0.84,
            microMotionPreservation = 0.84,
            jitterSuppression = 0.60,
        )

        fun precision() = ReconstructionTuning(
            solverIterations = 28,
            temporalWindowFrames = 16,
            maxPredictionFrames = 8,
            boneLengthStiffness = 0.94,
            jointLimitStiffness = 0.92,
            depthPriorWeight = 0.82,
            cameraMotionCompensationWeight = 0.74,
            occlusionRecoveryWeight = 0.72,
            microMotionPreservation = 0.90,
            jitterSuppression = 0.68,
        )
    }
}

data class MotionExportPolicy(
    val motionOnly: Boolean = true,
    val persistRgbFrames: Boolean = false,
    val persistVideo: Boolean = false,
    val persistRawSensorFrames: Boolean = false,
    val persistDepthFrames: Boolean = false,
    val persistFaceLandmarks: Boolean = true,
    val persistEyeState: Boolean = true,
    val persistHandState: Boolean = true,
    val persistLightingState: Boolean = true,
    val persistCameraTelemetry: Boolean = true,
    val persistQualityEvidence: Boolean = true,
    val maxChunkFrames: Int = 120,
    val checksumAlgorithm: String = "CRC32C",
) {
    init {
        require(motionOnly)
        require(!persistRgbFrames)
        require(!persistVideo)
        require(!persistRawSensorFrames)
        require(!persistDepthFrames)
        require(maxChunkFrames in 1..10_000)
        require(checksumAlgorithm.isNotBlank())
    }
}

data class CaptureIntegrityEvidence(
    val timestampNanos: Long,
    val rawScore: Double,
    val smoothedScore: Double,
    val band: String,
    val outputAllowed: Boolean,
    val bottleneck: String,
    val reconstruction: Double,
    val geometry: Double,
    val rigCoverage: Double = 1.0,
    val temporal: Double,
    val synchronization: Double,
    val calibration: Double,
    val tracking: Double,
    val occlusion: Double,
    val cameraMotion: Double,
    val performance: Double,
    val microMotion: Double,
    val faceGeometry: Double = 1.0,
    val eyeTemporal: Double = 1.0,
    val faceEye: Double,
    val faceEyeFusionAccepted: Boolean = true,
    val faceEyeFusionMode: String = "HOLD",
    val faceEyeIndependentEvidenceSupport: Double = 0.0,
    val faceEyeReliabilityTrend: Double = 0.0,
    val faceEyeFusionConfidence: Double = 0.0,
    val faceEyeOutOfDistributionRisk: Double = 0.0,
    val faceEyeCalibrationDriftPersistent: Boolean = false,
    val hand: Double,
    val kinematic: Double = 1.0,
    val surface: Double = 1.0,
    val physics: Double = 1.0,
    val physicsAccepted: Boolean = true,
    val physicsMeanBoneResidualMeters: Double = 0.0,
    val physicsP95BoneResidualMeters: Double = 0.0,
    val physicsMaxBoneResidualMeters: Double = 0.0,
    val physicsSymmetryScore: Double = 1.0,
    val physicsFloorContactScore: Double = 1.0,
    val physicsCorrectedJointCount: Int = 0,
    val physicsConstrainedEdgeCount: Int = 0,
    val directionalUncertaintyMajor95Meters: Double = 0.0,
    val directionalUncertaintyDepthMeters: Double = 0.0,
    val directionalUncertaintyAnisotropy: Double = 1.0,
    val directionalUncertaintyConfidence: Double = 1.0,
    val directionalUncertaintyDegradedJointCount: Int = 0,
    val global4DScore: Double = 1.0,
    val global4DAccepted: Boolean = true,
    val global4DPersistentFailure: Boolean = false,
    val global4DRootMotionConsistency: Double = 1.0,
    val global4DRootVelocityAgreement: Double = 1.0,
    val global4DSceneConsistency: Double = 0.72,
    val global4DDirectionalUncertaintyConsistency: Double = 1.0,
    val global4DFaceEyeConsistency: Double = 1.0,
    val globalRigTrust: Double = 1.0,
    val hardFailures: List<String> = emptyList(),
    val warnings: List<String> = emptyList(),
    val recommendations: List<String> = emptyList(),
) {
    init {
        require(timestampNanos >= 0L)
        listOf(
            rawScore,
            smoothedScore,
            reconstruction,
            geometry,
            rigCoverage,
            temporal,
            synchronization,
            calibration,
            tracking,
            occlusion,
            cameraMotion,
            performance,
            microMotion,
            faceGeometry,
            eyeTemporal,
            faceEye,
            faceEyeIndependentEvidenceSupport,
            faceEyeFusionConfidence,
            faceEyeOutOfDistributionRisk,
            hand,
            kinematic,
            surface,
            physics,
            physicsSymmetryScore,
            physicsFloorContactScore,
        ).forEach { require(it in 0.0..1.0) }
        require(physicsMeanBoneResidualMeters.isFinite() && physicsMeanBoneResidualMeters >= 0.0)
        require(physicsP95BoneResidualMeters.isFinite() && physicsP95BoneResidualMeters >= 0.0)
        require(physicsMaxBoneResidualMeters.isFinite() && physicsMaxBoneResidualMeters >= 0.0)
        require(physicsCorrectedJointCount >= 0)
        require(physicsConstrainedEdgeCount >= 0)
        require(
            directionalUncertaintyMajor95Meters.isFinite() &&
                directionalUncertaintyMajor95Meters >= 0.0,
        )
        require(
            directionalUncertaintyDepthMeters.isFinite() &&
                directionalUncertaintyDepthMeters >= 0.0,
        )
        require(directionalUncertaintyAnisotropy >= 1.0)
        require(directionalUncertaintyConfidence in 0.0..1.0)
        require(directionalUncertaintyDegradedJointCount >= 0)
        require(global4DScore in 0.0..1.0)
        require(global4DRootMotionConsistency in 0.0..1.0)
        require(global4DSceneConsistency in 0.0..1.0)
        require(band.isNotBlank())
        require(bottleneck.isNotBlank())
    }
}

data class CaptureReadiness(
    val cameraReady: Boolean,
    val modelReady: Boolean,
    val calibrationReady: Boolean,
    val thermalReady: Boolean,
    val storageReady: Boolean,
    val sessionPolicyReady: Boolean,
    val subjectVisible: Boolean,
    val motionQuality: Double,
    val reasons: List<String>,
    val captureTimingConfidence: Double = 1.0,
    val captureTimingFps: Double = 0.0,
    val captureTimingJitterMillis: Double = 0.0,
    val captureTimingHealthy: Boolean = true,
    val multiCameraSyncConfidence: Double = 1.0,
    val multiCameraSyncHealthy: Boolean = true,
    val captureIntegrityScore: Double = 1.0,
    val captureIntegrityHealthy: Boolean = true,
    val captureIntegrityBottleneck: String = "NONE",
) {
    val ready: Boolean
        get() =
            cameraReady &&
                modelReady &&
                thermalReady &&
                storageReady &&
                sessionPolicyReady &&
                subjectVisible &&
                motionQuality >= 0.50 &&
                captureTimingConfidence >= 0.60 &&
                (captureTimingFps <= 0.0 || captureTimingFps >= 15.0) &&
                captureTimingJitterMillis <= 6.0 &&
                captureTimingHealthy &&
                multiCameraSyncConfidence >= 0.55 &&
                multiCameraSyncHealthy &&
                captureIntegrityHealthy &&
                captureIntegrityScore >= 0.55 &&
                reasons.isEmpty()
}


/**
 * Fused face/eye state used by the motion graph and avatar output.
 *
 * The contract distinguishes measured gaze from derived head pose and carries freshness/error
 * metadata so downstream systems can reject stale eye evidence without discarding valid body motion.
 */
data class FaceEyeFusionState(
    val timestampNanos: Long,
    val trackId: Long,
    val headRotation: Quat,
    val headYawDegrees: Double,
    val headRollDegrees: Double,
    val combinedGazeCamera: Vec3,
    val leftGazeCamera: Vec3?,
    val rightGazeCamera: Vec3?,
    val leftOpenness: Double,
    val rightOpenness: Double,
    val blinkProbability: Double,
    val gazeStability: Double,
    val saccadeVelocityDegreesPerSecond: Double,
    val vergenceMeters: Double?,
    val confidence: Double,
    val fresh: Boolean,
    val source: PerceptionSource,
    val diagnostics: List<String> = emptyList(),
) {
    init {
        require(leftOpenness in 0.0..1.0)
        require(rightOpenness in 0.0..1.0)
        require(blinkProbability in 0.0..1.0)
        require(gazeStability in 0.0..1.0)
        require(saccadeVelocityDegreesPerSecond >= 0.0)
        require(confidence in 0.0..1.0)
        if (vergenceMeters != null) require(vergenceMeters > 0.0)
    }
}

data class SessionPresentationSettings(
    val clipName: String = "Take",
    val countdownSeconds: Int = 3,
    val requireCalibrationBeforeRecord: Boolean = false,
    val automaticallyIndexAfterStop: Boolean = true,
    val showPreflightWarnings: Boolean = true,
) {
    init {
        require(clipName.length in 1..80)
        require(countdownSeconds in 0..10)
    }

    fun sanitizedClipName(): String =
        clipName
            .trim()
            .replace(Regex("[^A-Za-z0-9 _-]"), "_")
            .replace(Regex("\\s+"), " ")
            .take(80)
            .ifBlank { "Take" }
}

data class MotionStudioState(
    val phase: SubjectLockPhase = SubjectLockPhase.SEARCH,
    val selectedTrackId: Long? = null,
    val mode: MotionCaptureMode = MotionCaptureMode.MONOCULAR_RGB,
    val motionProfile: String = "CINEMATIC",
    val reconstructionProfile: String = "QUALITY",
    val captureReadiness: CaptureReadiness? = null,
    val bodyConfidence: Double = 0.0,
    val faceConfidence: Double = 0.0,
    val eyeConfidence: Double = 0.0,
    val handConfidence: Double = 0.0,
    val depthConfidence: Double = 0.0,
    val calibrationConfidence: Double = 0.0,
    val kinematicConfidence: Double = 1.0,
    val surfaceConfidence: Double = 1.0,
    val lighting: LightingState? = null,
    val camera: CameraTelemetryState? = null,
    val face: FaceExpressionState? = null,
    val eyes: EyeState? = null,
    val hands: List<HandState> = emptyList(),
    val warnings: List<String> = emptyList(),
    val errors: List<String> = emptyList(),
)

data class FusionEvidence(
    val timestampNanos: Long,
    val body: Double,
    val face: Double,
    val eyes: Double,
    val hands: Double,
    val depth: Double,
    val topology: Double,
    val temporal: Double,
    val calibration: Double,
    val lighting: Double,
    val camera: Double,
) {
    init {
        listOf(
            body, face, eyes, hands, depth, topology,
            temporal, calibration, lighting, camera,
        ).forEach { require(it in 0.0..1.0) }
    }

    fun aggregate(): Double =
        (
            body * 0.24 +
                face * 0.10 +
                eyes * 0.08 +
                hands * 0.08 +
                depth * 0.14 +
                topology * 0.10 +
                temporal * 0.10 +
                calibration * 0.06 +
                lighting * 0.05 +
                camera * 0.05
            ).coerceIn(0.0, 1.0)
}

fun Vec3.isFinite(): Boolean =
    x.isFinite() && y.isFinite() && z.isFinite()

fun Vec2.isFinite(): Boolean =
    x.isFinite() && y.isFinite()
