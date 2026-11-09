package com.nivukx.nox.core.capture

import android.content.Context
import com.nivukx.nox.core.model.AvatarPose
import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CaptureIntegrityEvidence
import com.nivukx.nox.core.model.DeviceDescriptor
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FrameTelemetry
import com.nivukx.nox.core.model.FocusTarget
import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.LightingEstimate
import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.reconstruction.MotionEventSegmentationEngine
import com.nivukx.nox.core.reconstruction.MotionEvidenceFusionEngine
import com.nivukx.nox.core.reconstruction.LightingEnvironmentTemporalConsistencyEngine
import com.nivukx.nox.core.model.PipelineTelemetry
import com.nivukx.nox.core.model.ProductionPolicyRecord
import com.nivukx.nox.core.reconstruction.CrossModalIdentityFusionEngine
import com.nivukx.nox.core.reconstruction.CalibratedMotionUncertaintyEngine
import com.nivukx.nox.core.reconstruction.SparseTsdfFusionEngine
import com.nivukx.nox.core.reconstruction.ProductionMotionDynamicsEngine
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.DenseMeshQuality
import com.nivukx.nox.core.model.SessionId
import com.nivukx.nox.core.model.SkeletonFrame
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter

/**
 * Structured synchronized-session writer.
 *
 * Motion capture is stored as an append-only structured session. Computational signals remain
 * independently inspectable so an interrupted take can be repaired or reprocessed without
 * relying on a rendered movie asset.
 */
class SessionRecorder(
    private val context: Context,
) : AutoCloseable {

    private val artifactIntegrity =
        SessionArtifactIntegrityEngine()

    private var root: File? = null
    private var framesWriter: BufferedWriter? = null
    private var posesWriter: BufferedWriter? = null
    private var facesWriter: BufferedWriter? = null
    private var faceMeshIndexWriter: BufferedWriter? = null
    private var faceMeshStream: DataOutputStream? = null
    private var faceMeshBytesWritten: Long = 0L
    private var eyesWriter: BufferedWriter? = null
    private var faceEyeFusionWriter: BufferedWriter? = null
    private var faceEyeEvidenceWriter: BufferedWriter? = null
    private var faceEyeCalibrationDriftWriter: BufferedWriter? = null
    private var handsWriter: BufferedWriter? = null
    private var hand3dWriter: BufferedWriter? = null
    private var avatarWriter: BufferedWriter? = null
    private var reconstructionWriter: BufferedWriter? = null
    private var predictedReconstructionWriter: BufferedWriter? = null
    private var reconstructionQualityWriter: BufferedWriter? = null
    private var denseMeshIndexWriter: BufferedWriter? = null
    private var denseMeshStream: DataOutputStream? = null
    private var denseMeshBytesWritten: Long = 0L
    private var telemetryWriter: BufferedWriter? = null
    private var captureTimingWriter: BufferedWriter? = null
    private var hardwareReadoutTimingWriter: BufferedWriter? = null
    private var sessionHealthWriter: BufferedWriter? = null
    private var captureIntegrityWriter: BufferedWriter? = null
    private var captureIntegritySummaryFile: File? = null
    private var captureIntegrityFrameCount = 0L
    private var captureIntegrityOutputFrames = 0L
    private var captureIntegrityScoreSum = 0.0
    private var captureIntegrityScoreMin = Double.POSITIVE_INFINITY
    private var captureIntegrityScoreMax = Double.NEGATIVE_INFINITY
    private var captureIntegrityCurrentHoldStreak = 0L
    private var captureIntegrityMaxHoldStreak = 0L
    private var captureIntegrityPreviousOutputAllowed: Boolean? = null
    private val captureIntegrityBandCounts = linkedMapOf<String, Long>()
    private val captureIntegrityBottleneckCounts = linkedMapOf<String, Long>()
    private val captureIntegrityWarningCounts = linkedMapOf<String, Long>()
    private val captureIntegrityHardFailureCounts = linkedMapOf<String, Long>()
    private var pipelineWriter: BufferedWriter? = null
    private var lightingWriter: BufferedWriter? = null
    private var imuWriter: BufferedWriter? = null
    private var focusWriter: BufferedWriter? = null
    private var motionEventsWriter: BufferedWriter? = null
    private var motionEvidenceWriter: BufferedWriter? = null
    private var lightingConsistencyWriter: BufferedWriter? = null
    private var trackLifecycleWriter: BufferedWriter? = null
    private var crossModalIdentityWriter: BufferedWriter? = null
    private var motionUncertaintyWriter: BufferedWriter? = null
    private var tsdfEvidenceWriter: BufferedWriter? = null
    private var motionDynamicsWriter: BufferedWriter? = null
    private var productionPolicyWriter: BufferedWriter? = null

    fun start(
        sessionId: SessionId,
        targetFps: Int,
        width: Int,
        height: Int,
        devices: List<DeviceDescriptor>,
        captureProfile: CameraCaptureProfile? = null,
        mode: MotionCaptureMode = MotionCaptureMode.MONOCULAR_RGB,
        clipName: String = sessionId.value,
        configurationFingerprint: String? = null,
    ): File {
        check(root == null) { "Session already active" }

        val dir = File(
            File(context.filesDir, "sessions"),
            sessionId.value,
        ).apply {
            mkdirs()
        }

        root = dir

        framesWriter = writer(dir, "frames.ndjson")
        posesWriter = writer(dir, "poses.ndjson")
        facesWriter = writer(dir, "faces.ndjson")
        faceMeshIndexWriter = writer(dir, "face_mesh_index.ndjson")
        faceMeshStream = DataOutputStream(
            BufferedOutputStream(
                FileOutputStream(
                    File(dir, "face_mesh.bin"),
                    false,
                ),
                256 * 1024,
            ),
        )
        faceMeshBytesWritten = 0L
        eyesWriter = writer(dir, "eyes.ndjson")
        faceEyeFusionWriter = writer(dir, "face_eye_fusion.ndjson")
        faceEyeEvidenceWriter =
            writer(dir, "face_eye_evidence.ndjson")
        faceEyeCalibrationDriftWriter =
            writer(dir, "face_eye_calibration_drift.ndjson")
        handsWriter = writer(dir, "hands.ndjson")
        hand3dWriter = writer(dir, "hands_3d.ndjson")
        reconstructionWriter = writer(dir, "reconstruction.ndjson")
        predictedReconstructionWriter = writer(dir, "predicted_reconstruction.ndjson")
        reconstructionQualityWriter =
            writer(dir, "reconstruction_quality.ndjson")
        denseMeshIndexWriter = writer(dir, "dense_mesh_index.ndjson")
        denseMeshStream = DataOutputStream(
            BufferedOutputStream(
                FileOutputStream(
                    File(dir, "dense_mesh.bin"),
                    false,
                ),
                256 * 1024,
            ),
        )
        denseMeshBytesWritten = 0L
        avatarWriter = writer(dir, "avatar_motion.ndjson")
        telemetryWriter = writer(dir, "telemetry.ndjson")
        captureTimingWriter = writer(dir, "capture_timing.ndjson")
        hardwareReadoutTimingWriter = writer(dir, "hardware_readout_timing.ndjson")
        sessionHealthWriter = writer(dir, "session_health.ndjson")
        captureIntegrityWriter = writer(dir, "capture_integrity.ndjson")
        captureIntegritySummaryFile = File(dir, "capture_integrity_summary.json")
        captureIntegrityFrameCount = 0L
        captureIntegrityOutputFrames = 0L
        captureIntegrityScoreSum = 0.0
        captureIntegrityScoreMin = Double.POSITIVE_INFINITY
        captureIntegrityScoreMax = Double.NEGATIVE_INFINITY
        captureIntegrityCurrentHoldStreak = 0L
        captureIntegrityMaxHoldStreak = 0L
        captureIntegrityPreviousOutputAllowed = null
        captureIntegrityBandCounts.clear()
        captureIntegrityBottleneckCounts.clear()
        captureIntegrityWarningCounts.clear()
        captureIntegrityHardFailureCounts.clear()
        pipelineWriter = writer(dir, "pipeline_telemetry.ndjson")
        lightingWriter = writer(dir, "lighting.ndjson")
        imuWriter = writer(dir, "imu.ndjson")
        focusWriter = writer(dir, "focus.ndjson")
        motionEventsWriter = writer(dir, "motion_events.ndjson")
        motionEvidenceWriter = writer(dir, "motion_evidence.ndjson")
        lightingConsistencyWriter = writer(
            dir,
            "lighting_consistency.ndjson",
        )
        trackLifecycleWriter =
            writer(dir, "track_lifecycle.ndjson")
        crossModalIdentityWriter =
            writer(dir, "cross_modal_identity.ndjson")
        motionUncertaintyWriter =
            writer(dir, "motion_uncertainty.ndjson")
        tsdfEvidenceWriter =
            writer(dir, "tsdf_evidence.ndjson")
        motionDynamicsWriter =
            writer(dir, "motion_dynamics.ndjson")
        productionPolicyWriter =
            writer(dir, "production_policy.ndjson")

        val deviceArray = JSONArray().apply {
            devices.forEach { device ->
                put(
                    JSONObject().apply {
                        put("id", device.id.value)
                        put("name", device.displayName)
                        put("role", device.role.name)
                        put("connected", device.connected)
                        put("clockOffsetNanos", device.clockOffsetNanos)
                        put("roundTripNanos", device.roundTripNanos)
                        put("packetLoss", device.packetLoss)
                        put("supportsRaw", device.capabilities.supportsRaw)
                        put(
                            "supportsConcurrentCamera",
                            device.capabilities.supportsConcurrentCamera,
                        )
                        put(
                            "supportsHardwareReadoutTimestamp",
                            device.capabilities.supportsHardwareReadoutTimestamp,
                        )
                        put(
                            "maximumVerifiedConcurrentCameras",
                            device.capabilities.maximumVerifiedConcurrentCameras,
                        )
                        put(
                            "minimumExposureTimeNanos",
                            device.capabilities.minimumExposureTimeNanos,
                        )
                        put(
                            "maximumExposureTimeNanos",
                            device.capabilities.maximumExposureTimeNanos,
                        )
                        put(
                            "maxAnalysis",
                            device.capabilities.maxAnalysisWidth.toString() +
                                "x" +
                                device.capabilities.maxAnalysisHeight,
                        )
                        put("maxFps", device.capabilities.maxFps)
                    },
                )
            }
        }

        val manifest = JSONObject().apply {
            put("schema", 18)
            put("motionOnly", true)
            put("mode", mode.name)
            put("mediaFramesPersisted", false)
            put("videoEncoder", "none")
            configurationFingerprint?.let { fingerprint ->
                put("configurationFingerprintSchema", SessionConfigurationFingerprint.SCHEMA)
                put("configurationFingerprint", fingerprint)
            }
            put(
                "faceMesh",
                JSONObject().apply {
                    put("file", "face_mesh.bin")
                    put("index", "face_mesh_index.ndjson")
                    put("format", "NXFM1")
                    put("pointType", "float32_index_xyz")
                    put("triangleType", "int32_abc")
                    put("maxPoints", 468)
                    put("maxTriangles", 960)
                },
            )
            put(
                "denseMesh",
                JSONObject().apply {
                    put("file", "dense_mesh.bin")
                    put("index", "dense_mesh_index.ndjson")
                    put("format", "NXM1")
                    put("vertexType", "float32_xyz")
                    put("triangleType", "int32_abc")
                },
            )
            put("sessionId", sessionId.value)
            put("createdAtMillis", System.currentTimeMillis())
            put("createdAtNanos", System.nanoTime())
            put("targetFps", targetFps)
            put("width", width)
            put("height", height)
            put("timebase", "camera_sensor")
            put(
                "telemetry",
                JSONObject().apply {
                    put("frameTiming", "capture_timing.ndjson")
                    put("hardwareReadoutTiming", "hardware_readout_timing.ndjson")
                    put("hardwareReadoutClock", "readoutTimestampNanos")
                    put("sessionHealth", "session_health.ndjson")
                    put("pipeline", "pipeline_telemetry.ndjson")
                    put("motionEvidence", "motion_evidence.ndjson")
                    put(
                        "lightingConsistency",
                        "lighting_consistency.ndjson",
                    )
                    put(
                        "crossModalIdentity",
                        "cross_modal_identity.ndjson",
                    )
                    put(
                        "faceEyeEvidence",
                        "face_eye_evidence.ndjson",
                    )
                    put(
                        "faceEyeCalibrationDrift",
                        "face_eye_calibration_drift.ndjson",
                    )
                    put(
                        "motionUncertainty",
                        "motion_uncertainty.ndjson",
                    )
                    put(
                        "tsdfEvidence",
                        "tsdf_evidence.ndjson",
                    )
                    put(
                        "motionDynamics",
                        "motion_dynamics.ndjson",
                    )
                },
            )
            put("devices", deviceArray)
            put(
                "productionPolicy",
                JSONObject().apply {
                    put("stream", "production_policy.ndjson")
                    put("recordSchema", 2)
                    put("runtimeQualityGate", "HighEndReconstructionQualityEngine")
                    put("nativeQualityGate", "ProductionQualityGate")
                    put("provenanceRequired", true)
                },
            )
            put(
                "artifactIntegrity",
                JSONObject().apply {
                    put("file", "checksums.json")
                    put("algorithm", "SHA-256")
                    put("generatedOnFinalize", true)
                    put("verificationSupported", true)
                },
            )
            put(
                "offlineRefinement",
                JSONObject().apply {
                    put("planner", "HighEndOfflineRefinementPlanner")
                    put("deterministicStages", true)
                    put("crossModalIdentity", "CrossModalIdentityFusionEngine")
                    put("dynamicStaticDecomposition", "DynamicStaticSceneDecompositionEngine")
                    put("neuralStagesRequireExternalWeights", true)
                },
            )

            captureProfile?.let { profile ->
                put(
                    "motionCaptureProfile",
                    JSONObject().apply {
                        put("analysisWidth", profile.analysisWidth)
                        put("analysisHeight", profile.analysisHeight)
                        put("fps", profile.fps)
                        put("rawSidecar", profile.enableRawSidecar)
                        put("depthSidecar", profile.enableDepthSidecar)
                        put("gyroscope", profile.enableGyroscope)
                        put("accelerometer", profile.enableAccelerometer)
                        put("gpu", profile.preferredGpu.name)
                        put("preserveSensorTimestamp", profile.preserveSensorTimestamp)
                        profile.manualExposureNanos?.let {
                            put("manualExposureNanos", it)
                        }
                        profile.manualIso?.let {
                            put("manualIso", it)
                        }
                        put("oisEnabled", profile.oisEnabled)
                        put(
                            "electronicStabilizationEnabled",
                            profile.electronicStabilizationEnabled,
                        )
                    },
                )
            }
        }

        File(dir, "manifest.json").writeText(
            manifest.toString(2),
        )

        return dir
    }

    fun appendProductionPolicy(
        record: ProductionPolicyRecord,
    ) {
        val writer = productionPolicyWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", record.timestampNanos)
                put("profileId", record.profileId)
                put("policy", record.policy)
                put("bottleneck", record.bottleneck)
                put("score", record.score)
                put("rawScore", record.rawScore)
                put("outputAllowed", record.outputAllowed)
                put("hardFailure", record.hardFailure)
                put(
                    "budget",
                    JSONObject().apply {
                        put(
                            "detectorIntervalFrames",
                            record.detectorIntervalFrames,
                        )
                        put(
                            "targetInferenceFps",
                            record.targetInferenceFps,
                        )
                        put(
                            "targetCaptureFps",
                            record.targetCaptureFps,
                        )
                        put("denseMeshHz", record.denseMeshHz)
                        put("faceMeshHz", record.faceMeshHz)
                        put("eyeTrackingHz", record.eyeTrackingHz)
                        put("handTrackingHz", record.handTrackingHz)
                        put(
                            "bundleAdjustmentIterations",
                            record.bundleAdjustmentIterations,
                        )
                        put("tsdfStride", record.tsdfStride)
                        put("meshVertexBudget", record.meshVertexBudget)
                        put(
                            "allowDenseSurface",
                            record.allowDenseSurface,
                        )
                        put(
                            "allowReferenceOptimization",
                            record.allowReferenceOptimization,
                        )
                        put(
                            "calibratedUncertaintyConfidence",
                            record.calibratedUncertaintyConfidence,
                        )
                        put(
                            "motionOutOfDistributionRisk",
                            record.motionOutOfDistributionRisk,
                        )
                        put(
                            "motionUncertaintyCoverage",
                            record.motionUncertaintyCoverage,
                        )
                        put(
                            "globalRigTrust",
                            record.globalRigTrust,
                        )
                        put(
                            "motionDynamicStress",
                            record.motionDynamicStress,
                        )
                        put(
                            "motionDynamicsScore",
                            record.motionDynamicsScore,
                        )
                        put(
                            "motionLowFrequencyScore",
                            record.motionLowFrequencyScore,
                        )
                        put(
                            "motionHighFrequencyScore",
                            record.motionHighFrequencyScore,
                        )
                        put(
                            "motionFrequencyBandSensitivity",
                            record.motionFrequencyBandSensitivity,
                        )
                        put(
                            "physicsConstraintScore",
                            record.physicsConstraintScore,
                        )
                        put(
                            "physicsConstraintAccepted",
                            record.physicsConstraintAccepted,
                        )
                        put(
                            "handUncertaintyConfidence",
                            record.handUncertaintyConfidence,
                        )
                        put(
                            "handUncertaintyRadiusMeters",
                            record.handUncertaintyRadiusMeters,
                        )
                        put(
                            "handUncertaintyCoverage",
                            record.handUncertaintyCoverage,
                        )
                        put(
                            "handUncertaintyOodRisk",
                            record.handUncertaintyOodRisk,
                        )
                        put(
                            "handUncertaintyPolicy",
                            record.handUncertaintyPolicy,
                        )
                        put(
                            "worldSpaceConsistencyScore",
                            record.worldSpaceConsistencyScore,
                        )
                        put(
                            "worldSpaceRobustOutlierRatio",
                            record.worldSpaceRobustOutlierRatio,
                        )
                        put(
                            "surfaceNormalObservability",
                            record.surfaceNormalObservability,
                        )
                        put(
                            "surfaceHighUncertaintyRatio",
                            record.surfaceHighUncertaintyRatio,
                        )
                        put(
                            "surfaceCameraInformationGain",
                            record.surfaceCameraInformationGain,
                        )
                        put(
                            "surfaceEvidenceStrictness",
                            record.surfaceEvidenceStrictness,
                        )
                        put(
                            "faceEyeFusionAccepted",
                            record.faceEyeFusionAccepted,
                        )
                        put(
                            "faceEyeFusionMode",
                            record.faceEyeFusionMode,
                        )
                        put(
                            "faceEyeIndependentEvidenceSupport",
                            record.faceEyeIndependentEvidenceSupport,
                        )
                        put(
                            "faceEyeReliabilityTrend",
                            record.faceEyeReliabilityTrend,
                        )
                        put(
                            "faceEyeHardFailure",
                            record.faceEyeHardFailure,
                        )
                        put(
                            "faceEyeShapeConfidence",
                            record.faceEyeShapeConfidence,
                        )
                        put(
                            "faceEyeEyeDynamicConsistency",
                            record.faceEyeEyeDynamicConsistency,
                        )
                        put(
                            "faceEyeHardFailureReasonCount",
                            record.faceEyeHardFailureReasonCount,
                        )
                        put(
                            "faceEyeFusionConfidence",
                            record.faceEyeFusionConfidence,
                        )
                        put(
                            "faceEyeOutOfDistributionRisk",
                            record.faceEyeOutOfDistributionRisk,
                        )
                        put(
                            "faceEyeFusionStrictness",
                            record.faceEyeFusionStrictness,
                        )
                        put(
                            "faceEyeSaccadeVelocityDegreesPerSecond",
                            record.faceEyeSaccadeVelocityDegreesPerSecond,
                        )
                        put(
                            "faceEyeDominantFailure",
                            record.faceEyeDominantFailure,
                        )
                        put(
                            "faceEyeCalibrationDriftScore",
                            record.faceEyeCalibrationDriftScore,
                        )
                        put(
                            "faceEyeCalibrationDriftPersistent",
                            record.faceEyeCalibrationDriftPersistent,
                        )
                        put(
                            "faceEyeIpdDriftToleranceMillimeters",
                            record.faceEyeIpdDriftToleranceMillimeters,
                        )
                        put(
                            "physicsConstraintStrictness",
                            record.physicsConstraintStrictness,
                        )
                        put(
                            "global4DConsistencyStrictness",
                            record.global4DConsistencyStrictness,
                        )
                        put(
                            "physicsMeanBoneResidualMeters",
                            record.physicsMeanBoneResidualMeters,
                        )
                        put(
                            "physicsP95BoneResidualMeters",
                            record.physicsP95BoneResidualMeters,
                        )
                        put(
                            "physicsMaxBoneResidualMeters",
                            record.physicsMaxBoneResidualMeters,
                        )
                        put(
                            "physicsSymmetryScore",
                            record.physicsSymmetryScore,
                        )
                        put(
                            "physicsFloorContactScore",
                            record.physicsFloorContactScore,
                        )
                        put(
                            "physicsCorrectedJointCount",
                            record.physicsCorrectedJointCount,
                        )
                        put(
                            "physicsConstrainedEdgeCount",
                            record.physicsConstrainedEdgeCount,
                        )
                        put(
                            "directionalUncertaintyMajor95Meters",
                            record.directionalUncertaintyMajor95Meters,
                        )
                        put(
                            "directionalUncertaintyDepthMeters",
                            record.directionalUncertaintyDepthMeters,
                        )
                        put(
                            "directionalUncertaintyAnisotropy",
                            record.directionalUncertaintyAnisotropy,
                        )
                        put(
                            "directionalUncertaintyConfidence",
                            record.directionalUncertaintyConfidence,
                        )
                        put(
                            "directionalUncertaintyDegradedJointCount",
                            record.directionalUncertaintyDegradedJointCount,
                        )
                        put("global4DScore", record.global4DScore)
                        put(
                            "global4DAccepted",
                            record.global4DAccepted,
                        )
                        put(
                            "global4DPersistentFailure",
                            record.global4DPersistentFailure,
                        )
                        put(
                            "global4DRootMotionConsistency",
                            record.global4DRootMotionConsistency,
                        )
                        put(
                            "global4DRootVelocityAgreement",
                            record.global4DRootVelocityAgreement,
                        )
                        put(
                            "global4DSceneConsistency",
                            record.global4DSceneConsistency,
                        )
                        put(
                            "global4DDirectionalUncertaintyConsistency",
                            record.global4DDirectionalUncertaintyConsistency,
                        )
                        put(
                            "global4DFaceEyeConsistency",
                            record.global4DFaceEyeConsistency,
                        )
                    },
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        record.warnings
                            .distinct()
                            .take(16)
                            .forEach(::put)
                    },
                )
            },
        )
    }

    fun updateCalibration(snapshot: CalibrationSnapshot?) {
        val directory = root ?: return
        val manifestFile = File(
            directory,
            "manifest.json",
        )
        if (!manifestFile.isFile) return

        val manifest = runCatching {
            JSONObject(manifestFile.readText())
        }.getOrElse {
            return
        }

        if (snapshot == null) {
            manifest.remove("calibration")
        } else {
            manifest.put(
                "calibration",
                JSONObject().apply {
                    put(
                        "reprojectionErrorPx",
                        snapshot.reprojectionErrorPx,
                    )
                    put(
                        "calibratedAtNanos",
                        snapshot.calibratedAtNanos,
                    )
                    put(
                        "cameraCount",
                        snapshot.intrinsics.size,
                    )
                    put(
                        "devices",
                        JSONArray().apply {
                            snapshot.intrinsics.keys
                                .sortedBy { it.value }
                                .forEach { id ->
                                    put(
                                        JSONObject().apply {
                                            put("id", id.value)

                                            val intrinsics =
                                                snapshot.intrinsics.getValue(id)

                                            val extrinsics =
                                                snapshot.extrinsics.getValue(id)

                                            put("fx", intrinsics.fx)
                                            put("fy", intrinsics.fy)
                                            put("cx", intrinsics.cx)
                                            put("cy", intrinsics.cy)

                                            put(
                                                "positionX",
                                                extrinsics.pose.position.x,
                                            )
                                            put(
                                                "positionY",
                                                extrinsics.pose.position.y,
                                            )
                                            put(
                                                "positionZ",
                                                extrinsics.pose.position.z,
                                            )

                                            put(
                                                "rotationW",
                                                extrinsics.pose.rotation.w,
                                            )
                                            put(
                                                "rotationX",
                                                extrinsics.pose.rotation.x,
                                            )
                                            put(
                                                "rotationY",
                                                extrinsics.pose.rotation.y,
                                            )
                                            put(
                                                "rotationZ",
                                                extrinsics.pose.rotation.z,
                                            )

                                            val uncertainty =
                                                snapshot.uncertaintyProfiles[id]
                                            uncertainty?.let { profile ->
                                                put(
                                                    "translationSigmaMeters",
                                                    profile.translationSigmaMeters,
                                                )
                                                put(
                                                    "rotationSigmaRadians",
                                                    profile.rotationSigmaRadians,
                                                )
                                                put(
                                                    "intrinsicRelativeSigma",
                                                    profile.intrinsicRelativeSigma,
                                                )
                                                put(
                                                    "uncertaintyEvidenceConfidence",
                                                    profile.evidenceConfidence,
                                                )
                                            }
                                        },
                                    )
                                }
                        },
                    )
                },
            )
        }

        manifest.put(
            "schema",
            maxOf(
                17,
                manifest.optInt("schema", 17),
            ),
        )
        manifest.put("motionOnly", true)
        manifest.put("mediaFramesPersisted", false)
        manifestFile.writeText(
            manifest.toString(2),
        )
    }

    fun appendFrame(
        sequence: Long,
        sensorTimestampNanos: Long,
        monotonicTimestampNanos: Long,
        poseLandmarks: Int = 0,
    ) {
        val writer = framesWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("sequence", sequence)
                put("sensorTimestampNanos", sensorTimestampNanos)
                put(
                    "monotonicTimestampNanos",
                    monotonicTimestampNanos,
                )
                put("poseLandmarks", poseLandmarks)
            },
        )
    }

    fun appendFrameTelemetry(
        sequence: Long,
        telemetry: FrameTelemetry,
        cameraQuality: CameraCaptureQualityEngine.Result? = null,
    ) {
        val writer = telemetryWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("sequence", sequence)
                put(
                    "sensorTimestampNanos",
                    telemetry.sensorTimestampNanos,
                )
                put(
                    "monotonicTimestampNanos",
                    telemetry.monotonicTimestampNanos,
                )
                put(
                    "exposureTimeNanos",
                    telemetry.exposureTimeNanos,
                )
                put(
                    "iso",
                    telemetry.sensorSensitivityIso,
                )
                put(
                    "frameDurationNanos",
                    telemetry.frameDurationNanos,
                )
                put(
                    "rollingShutterSkewNanos",
                    telemetry.rollingShutterSkewNanos,
                )
                put(
                    "aeState",
                    telemetry.aeState ?: -1,
                )
                put(
                    "afState",
                    telemetry.afState ?: -1,
                )
                put(
                    "awbState",
                    telemetry.awbState ?: -1,
                )
                telemetry.lensFocusDistanceDiopters?.let {
                    put("focusDiopters", it)
                }
                telemetry.focusDistanceMeters?.let {
                    put("focusDistanceMeters", it)
                }
                telemetry.readoutTimestampNanos?.let {
                    put(
                        "readoutTimestampNanos",
                        it,
                    )
                    put(
                        "readoutDeltaFromSensorNanos",
                        it - telemetry.sensorTimestampNanos,
                    )
                }
                cameraQuality?.let { quality ->
                    put("cameraQualityScore", quality.score)
                    put("cameraCadenceFps", quality.cadenceFps)
                    put("cameraExposureFraction", quality.exposureFraction)
                    put("cameraRollingShutterFraction", quality.rollingShutterFraction)
                    put("cameraOpticalBlurRisk", quality.opticalBlurRisk)
                    put(
                        "cameraHardwareReadoutTimingConfidence",
                        quality.hardwareReadoutTimingConfidence,
                    )
                    quality.readoutDelayNanos?.let {
                        put("cameraReadoutDelayNanos", it)
                    }
                    put("cameraAnalysisDropRatio", quality.analysisDropRatio)
                    put("cameraFocusConfidence", quality.focusConfidence)
                    put("cameraMetadataCompleteness", quality.metadataCompleteness)
                    put("cameraQualityHardFailure", quality.hardFailure)
                    put(
                        "cameraQualityWarnings",
                        JSONArray(quality.warnings),
                    )
                }
                telemetry.sensorTemperatureCelsius?.let {
                    put(
                        "sensorTemperatureCelsius",
                        it,
                    )
                }
            },
        )
    }

    fun appendHardwareReadoutTiming(
        sequence: Long,
        result: HardwareReadoutTimingEvidenceEngine.Result,
    ) {
        val writer = hardwareReadoutTimingWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("sequence", sequence)
                put("state", result.state.name)
                put("confidence", result.confidence)
                put("sampleCount", result.sampleCount)
                put("validSampleCount", result.validSampleCount)
                put("invalidSampleCount", result.invalidSampleCount)
                put("outOfOrderCount", result.outOfOrderCount)
                put("coverage", result.coverage)
                put("medianDelayMillis", result.medianDelayMillis)
                put("p95DelayMillis", result.p95DelayMillis)
                put("p95JitterMillis", result.p95JitterMillis)
                put("baselineDriftMillis", result.baselineDriftMillis)
                put("monotonicReadout", result.monotonicReadout)
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendCaptureTiming(
        sequence: Long,
        quality: CaptureTimestampIntegrityEngine.Quality,
        assessment: CaptureTimestampIntegrityEngine.Assessment? = null,
    ) {
        val writer = captureTimingWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("sequence", sequence)
                put("sampleCount", quality.sampleCount)
                put("validSampleCount", quality.validSampleCount)
                put("duplicateCount", quality.duplicateCount)
                put("outOfOrderCount", quality.outOfOrderCount)
                put("invalidIntervalCount", quality.invalidIntervalCount)
                put("gapCount", quality.gapCount)
                put("largestGapNanos", quality.largestGapNanos)
                put("meanIntervalNanos", quality.meanIntervalNanos)
                put("medianIntervalNanos", quality.medianIntervalNanos)
                put("p95AbsoluteJitterNanos", quality.p95AbsoluteJitterNanos)
                put("meanAbsoluteJitterNanos", quality.meanAbsoluteJitterNanos)
                put("cadenceFps", quality.cadenceFps)
                put("hostCadenceFps", quality.hostCadenceFps)
                put("driftPpm", quality.driftPpm)
                put("rollingShutterFraction", quality.rollingShutterFraction)
                put("exposureDutyCycle", quality.exposureDutyCycle)
                put("confidence", quality.confidence)
                put("healthy", quality.healthy)
                assessment?.let {
                    put("assessmentVerdict", it.verdict.name)
                    put("assessmentSensorTimestampNanos", it.sensorTimestampNanos)
                    put("assessmentDeltaNanos", it.deltaNanos ?: -1L)
                    put("assessmentHostDeltaNanos", it.hostDeltaNanos ?: -1L)
                    put("assessmentIntervalErrorFraction", it.intervalErrorFraction)
                    put("assessmentJitterFraction", it.jitterFraction)
                    put("assessmentRollingShutterFraction", it.rollingShutterFraction)
                    put("assessmentExposureDutyCycle", it.exposureDutyCycle)
                    put("assessmentDriftPpm", it.driftPpm)
                    put("assessmentCadenceFps", it.cadenceFps)
                    put("assessmentConfidence", it.confidence)
                    put(
                        "assessmentWarnings",
                        JSONArray().apply {
                            it.warnings.forEach(::put)
                        },
                    )
                }
                put(
                    "warnings",
                    JSONArray().apply {
                        quality.warnings.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendCaptureIntegrity(
        evidence: CaptureIntegrityEvidence,
    ) {
        val writer = captureIntegrityWriter ?: return

        captureIntegrityFrameCount += 1L
        captureIntegrityScoreSum += evidence.smoothedScore
        captureIntegrityScoreMin =
            minOf(captureIntegrityScoreMin, evidence.smoothedScore)
        captureIntegrityScoreMax =
            maxOf(captureIntegrityScoreMax, evidence.smoothedScore)

        if (evidence.outputAllowed) {
            captureIntegrityOutputFrames += 1L
            captureIntegrityCurrentHoldStreak = 0L
        } else {
            captureIntegrityCurrentHoldStreak += 1L
            captureIntegrityMaxHoldStreak =
                maxOf(
                    captureIntegrityMaxHoldStreak,
                    captureIntegrityCurrentHoldStreak,
                )
        }
        captureIntegrityPreviousOutputAllowed = evidence.outputAllowed

        incrementCount(captureIntegrityBandCounts, evidence.band)
        incrementCount(
            captureIntegrityBottleneckCounts,
            evidence.bottleneck,
        )
        evidence.warnings.forEach {
            incrementCount(captureIntegrityWarningCounts, it)
        }
        evidence.hardFailures.forEach {
            incrementCount(captureIntegrityHardFailureCounts, it)
        }

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", evidence.timestampNanos)
                put("rawScore", evidence.rawScore)
                put("smoothedScore", evidence.smoothedScore)
                put("band", evidence.band)
                put("outputAllowed", evidence.outputAllowed)
                put("bottleneck", evidence.bottleneck)
                put("reconstruction", evidence.reconstruction)
                put("geometry", evidence.geometry)
                put("rigCoverage", evidence.rigCoverage)
                put("temporal", evidence.temporal)
                put("synchronization", evidence.synchronization)
                put("calibration", evidence.calibration)
                put("tracking", evidence.tracking)
                put("occlusion", evidence.occlusion)
                put("cameraMotion", evidence.cameraMotion)
                put("performance", evidence.performance)
                put("microMotion", evidence.microMotion)
                put("faceGeometry", evidence.faceGeometry)
                put("eyeTemporal", evidence.eyeTemporal)
                put("faceEye", evidence.faceEye)
                put(
                    "faceEyeFusionAccepted",
                    evidence.faceEyeFusionAccepted,
                )
                put(
                    "faceEyeFusionMode",
                    evidence.faceEyeFusionMode,
                )
                put(
                    "faceEyeIndependentEvidenceSupport",
                    evidence.faceEyeIndependentEvidenceSupport,
                )
                put(
                    "faceEyeReliabilityTrend",
                    evidence.faceEyeReliabilityTrend,
                )
                put(
                    "faceEyeFusionConfidence",
                    evidence.faceEyeFusionConfidence,
                )
                put(
                    "faceEyeOutOfDistributionRisk",
                    evidence.faceEyeOutOfDistributionRisk,
                )
                put(
                    "faceEyeCalibrationDriftPersistent",
                    evidence.faceEyeCalibrationDriftPersistent,
                )
                put("global4DScore", evidence.global4DScore)
                put(
                    "global4DAccepted",
                    evidence.global4DAccepted,
                )
                put(
                    "global4DPersistentFailure",
                    evidence.global4DPersistentFailure,
                )
                put(
                    "global4DRootMotionConsistency",
                    evidence.global4DRootMotionConsistency,
                )
                put(
                    "global4DSceneConsistency",
                    evidence.global4DSceneConsistency,
                )
                put("hand", evidence.hand)
                put("physics", evidence.physics)
                put(
                    "physicsAccepted",
                    evidence.physicsAccepted,
                )
                put(
                    "physicsMeanBoneResidualMeters",
                    evidence.physicsMeanBoneResidualMeters,
                )
                put(
                    "physicsP95BoneResidualMeters",
                    evidence.physicsP95BoneResidualMeters,
                )
                put(
                    "physicsMaxBoneResidualMeters",
                    evidence.physicsMaxBoneResidualMeters,
                )
                put(
                    "physicsSymmetryScore",
                    evidence.physicsSymmetryScore,
                )
                put(
                    "physicsFloorContactScore",
                    evidence.physicsFloorContactScore,
                )
                put(
                    "physicsCorrectedJointCount",
                    evidence.physicsCorrectedJointCount,
                )
                put(
                    "physicsConstrainedEdgeCount",
                    evidence.physicsConstrainedEdgeCount,
                )
                put(
                    "directionalUncertaintyMajor95Meters",
                    evidence.directionalUncertaintyMajor95Meters,
                )
                put(
                    "directionalUncertaintyDepthMeters",
                    evidence.directionalUncertaintyDepthMeters,
                )
                put(
                    "directionalUncertaintyAnisotropy",
                    evidence.directionalUncertaintyAnisotropy,
                )
                put(
                    "directionalUncertaintyConfidence",
                    evidence.directionalUncertaintyConfidence,
                )
                put(
                    "directionalUncertaintyDegradedJointCount",
                    evidence.directionalUncertaintyDegradedJointCount,
                )
                put(
                    "hardFailures",
                    JSONArray().apply {
                        evidence.hardFailures.forEach(::put)
                    },
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        evidence.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        evidence.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendSessionHealth(
        health: SessionHealthEngine.Result,
    ) {
        val writer = sessionHealthWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", health.timestampNanos)
                put("state", health.state.name)
                put("score", health.score)
                put("captureStability", health.captureStability)
                put("perceptionStability", health.perceptionStability)
                put("reconstructionStability", health.reconstructionStability)
                put("runtimeStability", health.runtimeStability)
                put("operatorConfidence", health.operatorConfidence)
                put(
                    "hardFailures",
                    JSONArray().apply {
                        health.hardFailures.forEach(::put)
                    },
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        health.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        health.recommendations.forEach(::put)
                    },
                )
                health.bottleneck?.let {
                    put("bottleneck", it.label)
                    put("bottleneckScore", it.score)
                }
                put(
                    "signals",
                    JSONArray().apply {
                        health.signals.forEach { signal ->
                            put(
                                JSONObject().apply {
                                    put("type", signal.type.name)
                                    put("score", signal.score)
                                    put("weight", signal.weight)
                                    put("hardFailure", signal.hardFailure)
                                    put("label", signal.label)
                                    put("detail", signal.detail)
                                },
                            )
                        }
                    },
                )
            },
        )
    }

    fun appendPipelineTelemetry(
        telemetry: PipelineTelemetry,
    ) {
        val writer = pipelineWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", telemetry.timestampNanos)
                put("captureFps", telemetry.captureFps)
                put("inferenceFps", telemetry.inferenceFps)
                put("inferenceLatencyMillis", telemetry.inferenceLatencyMillis)
                put(
                    "droppedInferenceFrames",
                    telemetry.droppedInferenceFrames,
                )
                put("trackedSubjects", telemetry.trackedSubjects)
                put("faceTracks", telemetry.faceTracks)
                put("eyeTracks", telemetry.eyeTracks)
                put("triangulatedJoints", telemetry.triangulatedJoints)
                put("reconstructedVertices", telemetry.reconstructedVertices)
                put(
                    "reprojectionErrorPx",
                    telemetry.reprojectionErrorPx,
                )
                put("clockSkewNanos", telemetry.clockSkewNanos)
                put("thermalStatus", telemetry.thermalStatus)
                put("memoryPressure", telemetry.memoryPressure)
                put("gpuBackend", telemetry.gpuBackend.name)
                put("healthy", telemetry.healthy)
                put(
                    "warnings",
                    JSONArray().apply {
                        telemetry.warnings.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendLighting(
        estimate: LightingEstimate,
    ) {
        val writer = lightingWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", estimate.timestampNanos)
                put(
                    "illuminanceLux",
                    estimate.illuminanceLux,
                )
                put(
                    "colorTemperatureKelvin",
                    estimate.colorTemperatureKelvin,
                )
                put(
                    "exposureCompensationEv",
                    estimate.exposureCompensationEv,
                )
                put(
                    "ambientLuminance",
                    estimate.ambientLuminance,
                )
                put(
                    "directionalX",
                    estimate.directionalLight.x,
                )
                put(
                    "directionalY",
                    estimate.directionalLight.y,
                )
                put(
                    "directionalZ",
                    estimate.directionalLight.z,
                )
                put(
                    "directionalIntensity",
                    estimate.directionalIntensity,
                )
                put("confidence", estimate.confidence)
                put(
                    "dynamicRangeStops",
                    estimate.dynamicRangeStops,
                )
            },
        )
    }

    fun appendImu(
        sample: ImuSample,
    ) {
        val writer = imuWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", sample.timestampNanos)
                put("sensorType", sample.sensorType)
                put("x", sample.x)
                put("y", sample.y)
                put("z", sample.z)
                put("accuracy", sample.accuracy)
            },
        )
    }

    fun appendAvatarPose(
        pose: AvatarPose,
    ) {
        val writer = avatarWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", pose.timestampNanos)
                put("rootX", pose.rootPositionMeters.x)
                put("rootY", pose.rootPositionMeters.y)
                put("rootZ", pose.rootPositionMeters.z)
                put("scale", pose.scale)
                put("confidence", pose.confidence)
                put(
                    "bones",
                    JSONObject().apply {
                        pose.bones
                            .toSortedMap()
                            .forEach { (name, value) ->
                                put(
                                    name,
                                    JSONObject().apply {
                                        put(
                                            "x",
                                            value.position.x,
                                        )
                                        put(
                                            "y",
                                            value.position.y,
                                        )
                                        put(
                                            "z",
                                            value.position.z,
                                        )
                                        put(
                                            "qw",
                                            value.rotation.w,
                                        )
                                        put(
                                            "qx",
                                            value.rotation.x,
                                        )
                                        put(
                                            "qy",
                                            value.rotation.y,
                                        )
                                        put(
                                            "qz",
                                            value.rotation.z,
                                        )
                                    },
                                )
                            }
                    },
                )
            },
        )
    }

    fun appendReconstructionQuality(
        result: com.nivukx.nox.core.reconstruction.MultiViewFusionEngine.DetailedResult,
    ) {
        val writer = reconstructionQualityWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.skeleton.timestampNanos)
                put("trackId", result.skeleton.trackId)
                put("accepted", result.accepted)
                put("qualityScore", result.qualityScore)
                put("temporalQualityScore", result.temporalQualityScore)
                put("continuityScore", result.continuityScore)
                put("reprojectionQualityScore", result.reprojectionQualityScore)
                put(
                    "reprojectionBottleneck",
                    result.reprojectionBottleneck,
                )
                put(
                    "rigGeometryDriftScore",
                    result.rigGeometryDriftScore,
                )
                put(
                    "rigGeometryDriftState",
                    result.rigGeometryDriftState.name,
                )
                put(
                    "geometricObservabilityScore",
                    result.geometricObservabilityScore,
                )
                put(
                    "cameraContributionSelectedCount",
                    result.cameraContributionSelectedCount,
                )
                put(
                    "cameraContributionExcludedCount",
                    result.cameraContributionExcludedCount,
                )
                put(
                    "cameraContributionDiversitySatisfied",
                    result.cameraContributionDiversitySatisfied,
                )
                put(
                    "cameraContributionInformationGainScore",
                    result.cameraContributionInformationGainScore,
                )
                put(
                    "supportEvidenceScore",
                    result.supportEvidenceScore,
                )
                put(
                    "supportEvidenceSupportingCameraCount",
                    result.supportEvidenceSupportingCameraCount,
                )
                put(
                    "supportEvidenceRejectedCameraCount",
                    result.supportEvidenceRejectedCameraCount,
                )
                put(
                    "supportEvidenceContradictionRisk",
                    result.supportEvidenceContradictionRisk,
                )
                put(
                    "supportEvidenceSupportingJointCount",
                    result.supportEvidenceSupportingJointCount,
                )
                put(
                    "supportEvidenceContradictedJointCount",
                    result.supportEvidenceContradictedJointCount,
                )
                put(
                    "supportEvidenceWarnings",
                    JSONArray().apply {
                        result.supportEvidenceWarnings
                            .distinct()
                            .take(16)
                            .forEach(::put)
                    },
                )
                put(
                    "cameraContributionReason",
                    result.cameraContributionReason,
                )
                put(
                    "jointViewSelectionScore",
                    result.jointViewSelectionScore,
                )
                put(
                    "jointViewDiversitySatisfiedRatio",
                    result.jointViewDiversitySatisfiedRatio,
                )
                put(
                    "jointSelectedObservationCount",
                    result.jointSelectedObservationCount,
                )
                put(
                    "jointExcludedObservationCount",
                    result.jointExcludedObservationCount,
                )
                put(
                    "jointViewSelectionReasons",
                    JSONArray().apply {
                        result.jointViewSelectionReasons.forEach(::put)
                    },
                )
                put(
                    "geometricObservabilityBottleneck",
                    result.geometricObservabilityBottleneck,
                )
                put("uncertaintyScore", result.uncertaintyScore)
                put(
                    "uncertaintyConfidence",
                    result.uncertaintyConfidence,
                )
                put(
                    "meanUncertaintyMeters",
                    result.meanUncertaintyMeters,
                )
                put(
                    "p95UncertaintyMeters",
                    result.p95UncertaintyMeters,
                )
                put(
                    "trustedJointRatio",
                    result.trustedJointRatio,
                )
                put(
                    "uncertaintyBottleneck",
                    result.uncertaintyBottleneck,
                )
                put(
                    "global4DScore",
                    result.global4DScore,
                )
                put(
                    "global4DAccepted",
                    result.global4DAccepted,
                )
                put(
                    "global4DState",
                    result.global4DState.name,
                )
                put(
                    "global4DPersistentFailure",
                    result.global4DPersistentFailure,
                )
                put(
                    "global4DRootMotionConsistency",
                    result.global4DRootMotionConsistency,
                )
                put(
                    "global4DRootVelocityAgreement",
                    result.global4DRootVelocityAgreement,
                )
                put(
                    "global4DSceneConsistency",
                    result.global4DSceneConsistency,
                )
                put(
                    "global4DDirectionalUncertaintyConsistency",
                    result.global4DDirectionalUncertaintyConsistency,
                )
                put(
                    "global4DFaceEyeConsistency",
                    result.global4DFaceEyeConsistency,
                )
                put(
                    "bodyMetricEvidenceAuthority",
                    result.bodyMetricEvidenceAuthority,
                )
                put(
                    "bodySymmetricEpipolarConsistencyScore",
                    result.bodySymmetricEpipolarConsistencyScore,
                )
                put(
                    "faceMetricEvidenceAuthority",
                    result.faceMetricEvidenceAuthority,
                )
                put(
                    "handMetricEvidenceAuthority",
                    result.handMetricEvidenceAuthority,
                )
                put(
                    "faceEpipolarConsistency",
                    result.faceEpipolarConsistency,
                )
                put(
                    "handEpipolarConsistency",
                    result.handEpipolarConsistency,
                )
                put(
                    "crossModalMetricAuthority",
                    result.crossModalMetricAuthority,
                )
                put(
                    "crossModalEpipolarAuthority",
                    result.crossModalEpipolarAuthority,
                )
                put(
                    "captureAuthorityScore",
                    result.captureAuthorityScore,
                )
                put(
                    "captureAuthorityBand",
                    result.captureAuthorityBand.name,
                )
                put(
                    "captureAuthorityHardBlocked",
                    result.captureAuthorityHardBlocked,
                )
                put(
                    "captureAuthorityWarnings",
                    JSONArray().apply {
                        result.captureAuthorityWarnings
                            .distinct()
                            .take(16)
                            .forEach(::put)
                    },
                )
                put(
                    "rejectionReasons",
                    JSONArray().apply {
                        result.rejectionReasons.forEach(::put)
                    },
                )
                put(
                    "temporalRejectionReasons",
                    JSONArray().apply {
                        result.temporalRejectionReasons.forEach(::put)
                    },
                )
                put(
                    "continuityReasons",
                    JSONArray().apply {
                        result.continuityReasons.forEach(::put)
                    },
                )
                put(
                    "reprojectionWarnings",
                    JSONArray().apply {
                        result.reprojectionRejectionReasons.forEach(::put)
                    },
                )
                put(
                    "rigGeometryWarnings",
                    JSONArray().apply {
                        result.rigGeometryDriftWarnings.forEach(::put)
                    },
                )
                put(
                    "observabilityWarnings",
                    JSONArray().apply {
                        result.geometricObservabilityWarnings.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendReconstruction(
        frame: SkeletonFrame,
    ) {
        val writer = reconstructionWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", frame.timestampNanos)
                put("subjectType", frame.subjectType.name)
                put("trackId", frame.trackId)
                put("globalConfidence", frame.globalConfidence)
                put(
                    "joints",
                    JSONArray().apply {
                        frame.joints
                            .sortedBy { it.landmarkId }
                            .forEach { joint ->
                                put(
                                    JSONObject().apply {
                                        put("landmarkId", joint.landmarkId)
                                        put("x", joint.position.x)
                                        put("y", joint.position.y)
                                        put("z", joint.position.z)
                                        put("vx", joint.velocity.x)
                                        put("vy", joint.velocity.y)
                                        put("vz", joint.velocity.z)
                                        put("confidence", joint.confidence)
                                        put("reprojectionErrorPx", joint.reprojectionErrorPx)
                                        put("observationCount", joint.observationCount)
                                        put(
                                            "positionUncertaintyMeters",
                                            joint.positionUncertaintyMeters,
                                        )
                                    },
                                )
                            }
                    },
                )
            },
        )
    }

    fun appendPredictedReconstruction(
        frame: SkeletonFrame,
        predictionAgeMillis: Long,
        predictionConfidence: Double,
        warnings: List<String>,
    ) {
        val writer = predictedReconstructionWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", frame.timestampNanos)
                put("trackId", frame.trackId)
                put("subjectType", frame.subjectType.name)
                put("globalConfidence", frame.globalConfidence)
                put("predictionAgeMillis", predictionAgeMillis)
                put("predictionConfidence", predictionConfidence)
                put(
                    "joints",
                    JSONArray().apply {
                        frame.joints
                            .sortedBy { it.landmarkId }
                            .forEach { joint ->
                                put(
                                    JSONObject().apply {
                                        put("landmarkId", joint.landmarkId)
                                        put("x", joint.position.x)
                                        put("y", joint.position.y)
                                        put("z", joint.position.z)
                                        put("vx", joint.velocity.x)
                                        put("vy", joint.velocity.y)
                                        put("vz", joint.velocity.z)
                                        put("confidence", joint.confidence)
                                        put(
                                            "positionUncertaintyMeters",
                                            joint.positionUncertaintyMeters,
                                        )
                                    },
                                )
                            }
                    },
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        warnings.distinct().forEach(::put)
                    },
                )
            },
        )
    }

    fun appendDenseMesh(
        mesh: ReconstructionMesh,
        quality: DenseMeshQuality,
    ) {
        val stream = denseMeshStream ?: return
        val indexWriter = denseMeshIndexWriter ?: return

        val vertexCount = mesh.vertices.size
        val triangleCount = mesh.triangles.size
        if (vertexCount <= 0 || triangleCount <= 0) return

        val byteOffset = denseMeshBytesWritten
        val byteLength =
            4L + // magic
                4L + // format version
                8L + // timestamp
                4L + // confidence
                4L + // vertices
                4L + // triangles
                12L * vertexCount +
                12L * triangleCount

        synchronized(stream) {
            stream.writeInt(0x4E584D31)
            stream.writeInt(1)
            stream.writeLong(mesh.timestampNanos)
            stream.writeFloat(mesh.confidence.toFloat())
            stream.writeInt(vertexCount)
            stream.writeInt(triangleCount)

            mesh.vertices.forEach { vertex ->
                stream.writeFloat(vertex.position.x.toFloat())
                stream.writeFloat(vertex.position.y.toFloat())
                stream.writeFloat(vertex.position.z.toFloat())
            }

            mesh.triangles.forEach { triangle ->
                stream.writeInt(triangle.a)
                stream.writeInt(triangle.b)
                stream.writeInt(triangle.c)
            }

            stream.flush()
        }

        denseMeshBytesWritten += byteLength

        append(
            indexWriter,
            JSONObject().apply {
                put("timestampNanos", mesh.timestampNanos)
                put("offsetBytes", byteOffset)
                put("lengthBytes", byteLength)
                put("vertices", vertexCount)
                put("triangles", triangleCount)
                put("meshConfidence", mesh.confidence)
                put("geometricConfidence", quality.geometricConfidence)
                if (quality.meanReprojectionErrorPx.isFinite()) {
                    put(
                        "reprojectionErrorPx",
                        quality.meanReprojectionErrorPx,
                    )
                } else {
                    put(
                        "reprojectionErrorPx",
                        JSONObject.NULL,
                    )
                }
                put("watertightness", quality.watertightnessScore)
                put(
                    "topologyIntegrity",
                    quality.topologyIntegrityScore,
                )
                put(
                    "boundaryEdges",
                    quality.boundaryEdgeCount,
                )
                put(
                    "nonManifoldEdges",
                    quality.nonManifoldEdgeCount,
                )
                put(
                    "duplicateTrianglesRemoved",
                    quality.duplicateTrianglesRemoved,
                )
                put("normalCoverage", quality.normalCoverage)
                put(
                    "meanUncertaintyMeters",
                    quality.meanUncertaintyMeters,
                )
                put("temporalStability", quality.temporalStability)
            },
        )
    }

    fun appendTrackLifecycle(
        timestampNanos: Long,
        state: String,
        score: Double,
        predictedCount: Int,
        recoveringCount: Int,
        warnings: List<String>,
    ) {
        val writer = trackLifecycleWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", timestampNanos)
                put("state", state)
                put("score", score.coerceIn(0.0, 1.0))
                put("predictedCount", predictedCount.coerceAtLeast(0))
                put("recoveringCount", recoveringCount.coerceAtLeast(0))
                put(
                    "warnings",
                    JSONArray().apply {
                        warnings.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendMotionDynamics(
        result: ProductionMotionDynamicsEngine.Result,
    ) {
        val writer = motionDynamicsWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("phase", result.phase.name)
                put("phaseConfidence", result.phaseConfidence)
                put("globalScore", result.globalScore)
                put("dynamicStressScore", result.dynamicStressScore)
                put("uncertaintyScore", result.uncertaintyScore)
                put(
                    "lowFrequencyMotionScore",
                    result.lowFrequencyMotionScore,
                )
                put(
                    "highFrequencyMotionScore",
                    result.highFrequencyMotionScore,
                )
                put(
                    "microMotionRetention",
                    result.microMotionRetention,
                )
                put(
                    "trustedJointRatio",
                    result.trustedJointRatio,
                )
                put(
                    "joints",
                    JSONArray().apply {
                        result.joints.forEach { joint ->
                            put(
                                JSONObject().apply {
                                    put(
                                        "landmarkId",
                                        joint.landmarkId,
                                    )
                                    put(
                                        "speedMetersPerSecond",
                                        joint.speedMetersPerSecond,
                                    )
                                    put(
                                        "accelerationMetersPerSecond2",
                                        joint.accelerationMetersPerSecond2,
                                    )
                                    put(
                                        "jerkMetersPerSecond3",
                                        joint.jerkMetersPerSecond3,
                                    )
                                    put(
                                        "motionEnergy",
                                        joint.motionEnergy,
                                    )
                                    put(
                                        "lowFrequencyMotionEnergy",
                                        joint.lowFrequencyMotionEnergy,
                                    )
                                    put(
                                        "highFrequencyMotionEnergy",
                                        joint.highFrequencyMotionEnergy,
                                    )
                                    put(
                                        "uncertaintyMeters",
                                        joint.uncertaintyMeters,
                                    )
                                    put(
                                        "confidence",
                                        joint.confidence,
                                    )
                                    put(
                                        "highDynamics",
                                        joint.highDynamics,
                                    )
                                },
                            )
                        }
                    },
                )
            },
        )
    }

    fun appendTsdfEvidence(
        timestampNanos: Long,
        decision: SparseTsdfFusionEngine.AdaptiveDecision,
        temporalDepthConsistency: Double = 0.0,
    ) {
        val writer = tsdfEvidenceWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", timestampNanos)
                put("stride", decision.stride)
                put("confidenceScale", decision.confidenceScale)
                put("detailLevel", decision.detailLevel)
                put("computePressure", decision.computePressure)
                put(
                    "temporalDepthConsistency",
                    temporalDepthConsistency.coerceIn(0.0, 1.0),
                )
                put("reason", decision.reason)
                put("acceptedSamples", decision.acceptedSamples)
                put("rejectedSamples", decision.rejectedSamples)
                put("meanEvidenceWeight", decision.meanEvidenceWeight)
                put(
                    "meanSurfaceNormalWeight",
                    decision.meanSurfaceNormalWeight,
                )
            },
        )
    }

    fun appendMotionUncertainty(
        result: CalibratedMotionUncertaintyEngine.Result,
    ) {
        val writer = motionUncertaintyWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("targetCoverage", result.targetCoverage)
                put("calibrationReady", result.calibrationReady)
                put("measuredJointCount", result.measuredJointCount)
                put("envelopeJointCount", result.envelopeJointCount)
                put("predictionOnlyJointCount", result.predictionOnlyJointCount)
                put("meanRadiusMeters", result.meanRadiusMeters)
                put("p95RadiusMeters", result.p95RadiusMeters)
                put("empiricalCoverage", result.empiricalCoverage)
                put("uncertaintyConfidence", result.uncertaintyConfidence)
                put("outOfDistributionRisk", result.outOfDistributionRisk)
                put("dynamicInflation", result.dynamicInflation)
                put("calibrationProgress", result.calibrationProgress)
                put("policy", result.policy.name)
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
                put(
                    "envelopes",
                    JSONArray().apply {
                        result.envelopes.forEach { envelope ->
                            put(
                                JSONObject().apply {
                                    put("landmarkId", envelope.landmarkId)
                                    put("radiusMeters", envelope.radiusMeters)
                                    put("empiricalCoverage", envelope.empiricalCoverage)
                                    put("calibrationSamples", envelope.calibrationSamples)
                                    put("innovationMeters", envelope.innovationMeters)
                                    put(
                                        "normalizedInnovation",
                                        envelope.normalizedInnovation,
                                    )
                                    put(
                                        "outOfDistributionRisk",
                                        envelope.outOfDistributionRisk,
                                    )
                                    put(
                                        "measurementUncertaintyMeters",
                                        envelope.measurementUncertaintyMeters,
                                    )
                                    put(
                                        "predictionUncertaintyMeters",
                                        envelope.predictionUncertaintyMeters,
                                    )
                                    put("confidence", envelope.confidence)
                                    put("predictionOnly", envelope.predictionOnly)
                                    put("calibrationReady", envelope.calibrationReady)
                                    envelope.warning?.let {
                                        put("warning", it)
                                    }
                                },
                            )
                        }
                    },
                )
            },
        )
    }

    fun appendCrossModalIdentity(
        result: CrossModalIdentityFusionEngine.Result,
    ) {
        val writer = crossModalIdentityWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("action", result.action.name)
                put("reason", result.reason)
                put("continuityScore", result.continuityScore.coerceIn(0.0, 1.0))
                put("swapRisk", result.swapRisk.coerceIn(0.0, 1.0))
                put("assignedTrackCount", result.assignedTrackCount)
                put("recoveredTrackCount", result.recoveredTrackCount)
                put("switchedTrackCount", result.switchedTrackCount)
                put("evidenceFusionScore", result.evidenceFusionScore)
                put(
                    "selectedEvidenceModalityCount",
                    result.selectedEvidenceModalityCount,
                )
                put(
                    "selectedEvidenceModalities",
                    JSONArray().apply {
                        result.assignments
                            .flatMap {
                                it.selectedEvidenceModalities
                            }
                            .distinct()
                            .sorted()
                            .forEach(::put)
                    },
                )
                put(
                    "evidenceWarnings",
                    JSONArray().apply {
                        result.evidenceWarnings.forEach(::put)
                    },
                )
                put(
                    "unmatchedObservationIds",
                    JSONArray().apply {
                        result.unmatchedObservationIds.forEach(::put)
                    },
                )
                put(
                    "orphanTrackIds",
                    JSONArray().apply {
                        result.orphanTrackIds.forEach(::put)
                    },
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendPose(
        timestampNanos: Long,
        confidence: Float,
        landmarks: List<Triple<String, Float, Float>>,
    ) {
        val writer = posesWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", timestampNanos)
                put("confidence", confidence)
                put(
                    "landmarks",
                    JSONArray().apply {
                        landmarks.forEach { (name, x, y) ->
                            put(
                                JSONObject().apply {
                                    put("name", name)
                                    put("x", x)
                                    put("y", y)
                                },
                            )
                        }
                    },
                )
            },
        )
    }

    fun appendFace(
        frame: FaceFrame,
    ) {
        val writer = facesWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", frame.timestampNanos)
                put("trackingId", frame.trackingId ?: -1)
                put(
                    "bounds",
                    JSONArray().apply {
                        put(frame.boundsLeft)
                        put(frame.boundsTop)
                        put(frame.boundsRight)
                        put(frame.boundsBottom)
                    },
                )
                put("yawDegrees", frame.yawDegrees)
                put("rollDegrees", frame.rollDegrees)
                put(
                    "leftEyeOpenProbability",
                    frame.leftEyeOpenProbability,
                )
                put(
                    "rightEyeOpenProbability",
                    frame.rightEyeOpenProbability,
                )
                put(
                    "points",
                    JSONArray().apply {
                        frame.points.forEach { point ->
                            put(
                                JSONObject().apply {
                                    put("index", point.index)
                                    put("x", point.point.x)
                                    put("y", point.point.y)
                                    put(
                                        "confidence",
                                        point.confidence,
                                    )
                                },
                            )
                        }
                    },
                )
            },
        )
    }


    fun appendFaceMesh(
        frame: FaceMeshFrame,
    ) {
        val stream = faceMeshStream ?: return
        val indexWriter = faceMeshIndexWriter ?: return

        val points = frame.points
            .distinctBy { it.index }
            .filter { it.index in 0..467 }
            .take(468)
        val pointIds = points.map { it.index }.toHashSet()
        val triangles = frame.triangles
            .filter {
                it.a in pointIds &&
                    it.b in pointIds &&
                    it.c in pointIds
            }
            .distinct()
            .take(960)

        if (points.isEmpty()) return

        val byteOffset = faceMeshBytesWritten
        val byteLength =
            4L + // magic
                4L + // format
                8L + // timestamp
                4L + // tracking id
                4L * 7L + // width + height + rotation + bounds
                4L + // confidence
                4L + // point coverage
                4L + // triangle count
                4L + // point count
                16L * points.size +
                12L * triangles.size

        synchronized(stream) {
            stream.writeInt(0x4E58464D)
            stream.writeInt(1)
            stream.writeLong(frame.timestampNanos)
            stream.writeInt(frame.trackingId ?: -1)
            stream.writeInt(frame.imageWidth)
            stream.writeInt(frame.imageHeight)
            stream.writeInt(frame.rotationDegrees)
            stream.writeFloat(frame.boundsLeft)
            stream.writeFloat(frame.boundsTop)
            stream.writeFloat(frame.boundsRight)
            stream.writeFloat(frame.boundsBottom)
            stream.writeFloat(frame.confidence.toFloat())
            stream.writeFloat(frame.pointCoverage.toFloat())
            stream.writeInt(points.size)
            stream.writeInt(triangles.size)

            points.forEach { point ->
                stream.writeInt(point.index)
                stream.writeFloat(point.position.x.toFloat())
                stream.writeFloat(point.position.y.toFloat())
                stream.writeFloat(point.position.z.toFloat())
            }

            triangles.forEach { triangle ->
                stream.writeInt(triangle.a)
                stream.writeInt(triangle.b)
                stream.writeInt(triangle.c)
            }

            stream.flush()
        }

        faceMeshBytesWritten += byteLength

        append(
            indexWriter,
            JSONObject().apply {
                put("timestampNanos", frame.timestampNanos)
                put("offsetBytes", byteOffset)
                put("lengthBytes", byteLength)
                put("trackingId", frame.trackingId ?: -1)
                put("points", points.size)
                put("triangles", triangles.size)
                put("confidence", frame.confidence)
                put("pointCoverage", frame.pointCoverage)
                put("triangleCoverage", frame.triangleCoverage)
            },
        )
    }

    fun appendFaceEyeFusion(
        state: com.nivukx.nox.core.model.FaceEyeFusionState,
    ) {
        val writer = faceEyeFusionWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", state.timestampNanos)
                put("trackId", state.trackId)
                put("headYawDegrees", state.headYawDegrees)
                put("headRollDegrees", state.headRollDegrees)
                put(
                    "headRotation",
                    JSONArray().apply {
                        put(state.headRotation.w)
                        put(state.headRotation.x)
                        put(state.headRotation.y)
                        put(state.headRotation.z)
                    },
                )
                put(
                    "combinedGazeCamera",
                    JSONArray().apply {
                        put(state.combinedGazeCamera.x)
                        put(state.combinedGazeCamera.y)
                        put(state.combinedGazeCamera.z)
                    },
                )
                state.leftGazeCamera?.let {
                    put(
                        "leftGazeCamera",
                        JSONArray().apply {
                            put(it.x)
                            put(it.y)
                            put(it.z)
                        },
                    )
                }
                state.rightGazeCamera?.let {
                    put(
                        "rightGazeCamera",
                        JSONArray().apply {
                            put(it.x)
                            put(it.y)
                            put(it.z)
                        },
                    )
                }
                put("leftOpenness", state.leftOpenness)
                put("rightOpenness", state.rightOpenness)
                put("blinkProbability", state.blinkProbability)
                put("gazeStability", state.gazeStability)
                put(
                    "saccadeVelocityDegreesPerSecond",
                    state.saccadeVelocityDegreesPerSecond,
                )
                state.vergenceMeters?.let {
                    put("vergenceMeters", it)
                }
                put("confidence", state.confidence)
                put("fresh", state.fresh)
                put("source", state.source.name)
                put(
                    "diagnostics",
                    JSONArray().apply {
                        state.diagnostics.forEach(::put)
                    },
                )
            },
        )
    }


    fun appendFaceEyeCalibrationDrift(
        result:
            com.nivukx.nox.core.reconstruction.FaceEyeCalibrationDriftEngine.Result,
    ) {
        val writer = faceEyeCalibrationDriftWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("state", result.state.name)
                put("driftScore", result.driftScore)
                put("ipdDriftMeters", result.ipdDriftMeters)
                put(
                    "angularDisagreementDegrees",
                    result.angularDisagreementDegrees,
                )
                put("reprojectionErrorPx", result.reprojectionErrorPx)
                put("binocularConfidence", result.binocularConfidence)
                put(
                    "faceEyeFusionConfidence",
                    result.faceEyeFusionConfidence,
                )
                put("baselineIpdMeters", result.baselineIpdMeters)
                put("warmupProgress", result.warmupProgress)
                put("persistent", result.persistent)
                put(
                    "recalibrationRecommended",
                    result.recalibrationRecommended,
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendFaceEyeEvidenceFusion(
        result:
            com.nivukx.nox.core.reconstruction.FaceEyeEvidenceFusionEngine.Result,
    ) {
        val writer = faceEyeEvidenceWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("faceConfidence", result.faceConfidence)
                put("eyeConfidence", result.eyeConfidence)
                put("fusedConfidence", result.fusedConfidence)
                put("faceReliability", result.faceReliability)
                put("eyeReliability", result.eyeReliability)
                put("faceEyeAgreement", result.faceEyeAgreement)
                put("faceFreshnessScore", result.faceFreshnessScore)
                put(
                    "faceReprojectionScore",
                    result.faceReprojectionScore,
                )
                put("gazeStabilityScore", result.gazeStabilityScore)
                put("gazeAgreementScore", result.gazeAgreementScore)
                put(
                    "geometricIntegrityScore",
                    result.geometricIntegrityScore,
                )
                put(
                    "ipdConsistencyScore",
                    result.ipdConsistencyScore,
                )
                put(
                    "faceOutOfDistributionRisk",
                    result.faceOutOfDistributionRisk,
                )
                put(
                    "eyeOutOfDistributionRisk",
                    result.eyeOutOfDistributionRisk,
                )
                put("eyeQuarantined", result.eyeQuarantined)
                put("faceQuarantined", result.faceQuarantined)
                put("accepted", result.accepted)
                put("fusionMode", result.fusionMode.name)
                put(
                    "independentEvidenceSupport",
                    result.independentEvidenceSupport,
                )
                put(
                    "faceReliabilityTrend",
                    result.faceReliabilityTrend,
                )
                put(
                    "eyeReliabilityTrend",
                    result.eyeReliabilityTrend,
                )
                put(
                    "faceShapeConfidence",
                    result.faceShapeConfidence,
                )
                put(
                    "eyeDynamicConsistency",
                    result.eyeDynamicConsistency,
                )
                put(
                    "faceUncertaintyCalibrationMultiplier",
                    result.faceUncertaintyCalibrationMultiplier,
                )
                put(
                    "eyeUncertaintyCalibrationMultiplier",
                    result.eyeUncertaintyCalibrationMultiplier,
                )
                put(
                    "faceUncertaintyCoverage",
                    result.faceUncertaintyCoverage,
                )
                put(
                    "eyeUncertaintyCoverage",
                    result.eyeUncertaintyCoverage,
                )
                put(
                    "uncertaintyCalibrationConfidence",
                    result.uncertaintyCalibrationConfidence,
                )
                put(
                    "uncertaintyCalibrationReady",
                    result.uncertaintyCalibrationReady,
                )
                put(
                    "captureDomainConfidence",
                    result.captureDomainConfidence,
                )
                put(
                    "lightingConfidence",
                    result.lightingConfidence,
                )
                put(
                    "faceScaleConfidence",
                    result.faceScaleConfidence,
                )
                put("faceHardFailure", result.faceHardFailure)
                put("eyeHardFailure", result.eyeHardFailure)
                put(
                    "hardFailureReasons",
                    JSONArray().apply {
                        result.hardFailureReasons.forEach(::put)
                    },
                )
                put(
                    "failureAttribution",
                    JSONArray().apply {
                        result.failureAttribution.forEach { failure ->
                            put(
                                JSONObject().apply {
                                    put(
                                        "domain",
                                        failure.domain.name,
                                    )
                                    put(
                                        "severity",
                                        failure.severity,
                                    )
                                    put(
                                        "evidence",
                                        failure.evidence,
                                    )
                                    put(
                                        "blocking",
                                        failure.blocking,
                                    )
                                    put(
                                        "rationale",
                                        failure.rationale,
                                    )
                                },
                            )
                        }
                    },
                )
                put(
                    "selectedSources",
                    JSONArray().apply {
                        result.selectedSources.forEach(::put)
                    },
                )
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendMotionEvidence(
        result: MotionEvidenceFusionEngine.Result,
    ) {
        val writer = motionEvidenceWriter ?: return
        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("state", result.state.name)
                put("score", result.score)
                put("confidence", result.confidence)
                put("subjectEvidence", result.subjectEvidence)
                put("cameraInterference", result.cameraInterference)
                put("occlusionInterference", result.occlusionInterference)
                put("temporalEvidence", result.temporalEvidence)
                put("degraded", result.degraded)
                put("enterActiveFrames", result.enterActiveFrames)
                put("exitActiveFrames", result.exitActiveFrames)
                put("degradedFrames", result.degradedFrames)
                put("recoveryFrames", result.recoveryFrames)
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendLightingConsistency(
        result: LightingEnvironmentTemporalConsistencyEngine.Result,
    ) {
        val writer = lightingConsistencyWriter ?: return
        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", result.timestampNanos)
                put("state", result.state.name)
                put("confidence", result.confidence)
                put("continuityScore", result.continuityScore)
                put(
                    "illuminanceChangeRatio",
                    result.illuminanceChangeRatio,
                )
                put(
                    "colorTemperatureChangeRatio",
                    result.colorTemperatureChangeRatio,
                )
                put(
                    "directionChangeDegrees",
                    result.directionChangeDegrees,
                )
                put(
                    "persistentFlickerFrames",
                    result.persistentFlickerFrames,
                )
                put(
                    "persistentDegradedFrames",
                    result.persistentDegradedFrames,
                )
                put("recoveryFrames", result.recoveryFrames)
                put(
                    "warnings",
                    JSONArray().apply {
                        result.warnings.forEach(::put)
                    },
                )
                put(
                    "recommendations",
                    JSONArray().apply {
                        result.recommendations.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendMotionEvent(
        event: MotionEventSegmentationEngine.Event,
    ) {
        val writer = motionEventsWriter ?: return
        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", event.timestampNanos)
                put("trackId", event.trackId)
                put("type", event.type.name)
                put("phase", event.phase.name)
                put("confidence", event.confidence)
                put("intensity", event.intensity)
                put("cadenceHz", event.cadenceHz)
                put("directionDegrees", event.directionDegrees)
                put("transitionScore", event.transitionScore)
                put("evidenceQuality", event.evidenceQuality)
                put("uncertaintyScore", event.uncertaintyScore)
                put("footSkateRisk", event.footSkateRisk)
                put("cameraMotionRisk", event.cameraMotionRisk)
                put(
                    "diagnostics",
                    JSONArray().apply {
                        event.diagnostics.forEach(::put)
                    },
                )
            },
        )
    }

    fun appendFocusTarget(
        target: FocusTarget,
    ) {
        val writer = focusWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", target.timestampNanos)
                put("x", target.normalizedX)
                put("y", target.normalizedY)
                put("confidence", target.confidence)
                put("mode", target.mode.name)
                put("trackingId", target.trackingId)
                put("subjectType", target.subjectType.name)
            },
        )
    }

    fun appendHands(
        deviceId: com.nivukx.nox.core.model.DeviceId,
        frame: HandTrackingFrame,
    ) {
        val writer = handsWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("deviceId", deviceId.value)
                put("timestampNanos", frame.timestampNanos)
                put("imageWidth", frame.imageWidth)
                put("imageHeight", frame.imageHeight)
                put("inferenceLatencyMillis", frame.inferenceLatencyMillis)
                put("modelReady", frame.modelReady)
                put("droppedByBackpressure", frame.droppedByBackpressure)
                put(
                    "hands",
                    JSONArray().apply {
                        frame.hands.forEach { hand ->
                            put(
                                JSONObject().apply {
                                    put("handId", hand.handId)
                                    put("side", hand.side.name)
                                    put("handednessConfidence", hand.handednessConfidence)
                                    put("confidence", hand.confidence)
                                    put("temporalStability", hand.temporalStability)
                                    put("predicted", hand.predicted)
                                    put(
                                        "wristVelocity",
                                        JSONArray().apply {
                                            put(hand.wristVelocityMetersPerSecond.x)
                                            put(hand.wristVelocityMetersPerSecond.y)
                                            put(hand.wristVelocityMetersPerSecond.z)
                                        },
                                    )
                                    put(
                                        "landmarks",
                                        JSONArray().apply {
                                            hand.landmarks.forEach { landmark ->
                                                put(
                                                    JSONObject().apply {
                                                        put("index", landmark.index)
                                                        put("x", landmark.normalizedX)
                                                        put("y", landmark.normalizedY)
                                                        put("z", landmark.normalizedZ)
                                                        put("confidence", landmark.confidence)
                                                        landmark.worldPositionMeters?.let {
                                                            put("worldX", it.x)
                                                            put("worldY", it.y)
                                                            put("worldZ", it.z)
                                                        }
                                                    },
                                                )
                                            }
                                        },
                                    )
                                },
                            )
                        }
                    },
                )
            },
        )
    }

    fun appendHand3D(
        frame: Hand3DFrame,
    ) {
        val writer = hand3dWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", frame.timestampNanos)
                put("confidence", frame.confidence)
                put(
                    "hands",
                    JSONArray().apply {
                        frame.hands.forEach { hand ->
                            put(
                                JSONObject().apply {
                                    put("handId", hand.handId)
                                    put("side", hand.side.name)
                                    put("confidence", hand.confidence)
                                    put("meanUncertaintyMeters", hand.meanUncertaintyMeters)
                                    put("reprojectionErrorPx", hand.reprojectionErrorPx)
                                    put(
                                        "landmarks",
                                        JSONArray().apply {
                                            hand.landmarks.forEach { landmark ->
                                                put(
                                                    JSONObject().apply {
                                                        put("index", landmark.index)
                                                        put("x", landmark.position.x)
                                                        put("y", landmark.position.y)
                                                        put("z", landmark.position.z)
                                                        put("confidence", landmark.confidence)
                                                        put("uncertaintyMeters", landmark.uncertaintyMeters)
                                                        put("observationCount", landmark.observationCount)
                                                    },
                                                )
                                            }
                                        },
                                    )
                                },
                            )
                        }
                    },
                )
            },
        )
    }

    fun appendEye(
        frame: EyeTrackingFrame,
    ) {
        val writer = eyesWriter ?: return

        append(
            writer,
            JSONObject().apply {
                put("timestampNanos", frame.timestampNanos)
                put("trackingId", frame.trackingId ?: -1)
                put(
                    "combinedGaze",
                    JSONArray().apply {
                        put(frame.combinedGaze.x)
                        put(frame.combinedGaze.y)
                        put(frame.combinedGaze.z)
                    },
                )
                put(
                    "fixationConfidence",
                    frame.fixationConfidence,
                )

                fun writeEye(
                    eye: com.nivukx.nox.core.model.EyeObservation,
                ): JSONObject =
                    JSONObject().apply {
                        put("x", eye.center.x)
                        put("y", eye.center.y)
                        put("irisX", eye.irisCenter?.x)
                        put("irisY", eye.irisCenter?.y)
                        put(
                            "offsetX",
                            eye.normalizedOffset.x,
                        )
                        put(
                            "offsetY",
                            eye.normalizedOffset.y,
                        )
                        put("openness", eye.openness)
                        put("blink", eye.blink)
                        put("confidence", eye.confidence)
                        put(
                            "gaze",
                            JSONArray().apply {
                                put(eye.gazeDirection.x)
                                put(eye.gazeDirection.y)
                                put(eye.gazeDirection.z)
                            },
                        )
                    }

                frame.left?.let { put("left", writeEye(it)) }
                frame.right?.let { put("right", writeEye(it)) }
            },
        )
    }

    private fun incrementCount(
        counts: MutableMap<String, Long>,
        key: String,
    ) {
        counts[key] = (counts[key] ?: 0L) + 1L
    }

    private fun writeCaptureIntegritySummary() {
        val file = captureIntegritySummaryFile ?: return
        val frameCount = captureIntegrityFrameCount
        val meanScore = if (frameCount > 0L) {
            captureIntegrityScoreSum / frameCount
        } else {
            0.0
        }
        val outputRatio = if (frameCount > 0L) {
            captureIntegrityOutputFrames.toDouble() / frameCount
        } else {
            0.0
        }

        val summary = JSONObject().apply {
            put("frameCount", frameCount)
            put("outputFrames", captureIntegrityOutputFrames)
            put("outputRatio", outputRatio)
            put("meanSmoothedScore", meanScore)
            put("minSmoothedScore", if (captureIntegrityScoreMin.isFinite()) captureIntegrityScoreMin else 0.0)
            put("maxSmoothedScore", if (captureIntegrityScoreMax.isFinite()) captureIntegrityScoreMax else 0.0)
            put("maxHoldStreakFrames", captureIntegrityMaxHoldStreak)
            put("finalOutputAllowed", captureIntegrityPreviousOutputAllowed ?: false)
            put("bandCounts", mapToJson(captureIntegrityBandCounts))
            put("bottleneckCounts", mapToJson(captureIntegrityBottleneckCounts))
            put("warningCounts", mapToJson(captureIntegrityWarningCounts))
            put("hardFailureCounts", mapToJson(captureIntegrityHardFailureCounts))
        }
        runCatching {
            file.writeText(summary.toString(2), Charsets.UTF_8)
        }
    }

    private fun mapToJson(
        values: Map<String, Long>,
    ): JSONObject =
        JSONObject().apply {
            values.toSortedMap().forEach { (key, value) ->
                put(key, value)
            }
        }

    private fun append(
        writer: BufferedWriter,
        record: JSONObject,
    ) {
        synchronized(writer) {
            writer.write(record.toString())
            writer.newLine()
        }
    }

    private fun writer(
        directory: File,
        name: String,
    ): BufferedWriter =
        BufferedWriter(
            FileWriter(
                File(directory, name),
                false,
            ),
            128 * 1024,
        )

    override fun close() {
        writeCaptureIntegritySummary()
        closeWriter(framesWriter)
        closeWriter(posesWriter)
        closeWriter(facesWriter)
        closeWriter(faceMeshIndexWriter)
        runCatching {
            faceMeshStream?.flush()
            faceMeshStream?.close()
        }
        closeWriter(eyesWriter)
        closeWriter(faceEyeFusionWriter)
        closeWriter(faceEyeEvidenceWriter)
        closeWriter(faceEyeCalibrationDriftWriter)
        closeWriter(handsWriter)
        closeWriter(hand3dWriter)
        closeWriter(reconstructionWriter)
        closeWriter(predictedReconstructionWriter)
        closeWriter(reconstructionQualityWriter)
        closeWriter(denseMeshIndexWriter)
        runCatching {
            denseMeshStream?.flush()
            denseMeshStream?.close()
        }
        closeWriter(avatarWriter)
        closeWriter(telemetryWriter)
        closeWriter(captureTimingWriter)
        closeWriter(hardwareReadoutTimingWriter)
        closeWriter(sessionHealthWriter)
        closeWriter(captureIntegrityWriter)
        closeWriter(pipelineWriter)
        closeWriter(lightingWriter)
        closeWriter(imuWriter)
        closeWriter(focusWriter)
        closeWriter(motionEventsWriter)
        closeWriter(motionEvidenceWriter)
        closeWriter(lightingConsistencyWriter)
        closeWriter(trackLifecycleWriter)
        closeWriter(crossModalIdentityWriter)
        closeWriter(motionUncertaintyWriter)
        closeWriter(tsdfEvidenceWriter)
        closeWriter(motionDynamicsWriter)
        closeWriter(productionPolicyWriter)

        framesWriter = null
        posesWriter = null
        facesWriter = null
        faceMeshIndexWriter = null
        faceMeshStream = null
        faceMeshBytesWritten = 0L
        eyesWriter = null
        faceEyeFusionWriter = null
        faceEyeEvidenceWriter = null
        faceEyeCalibrationDriftWriter = null
        handsWriter = null
        hand3dWriter = null
        reconstructionWriter = null
        predictedReconstructionWriter = null
        reconstructionQualityWriter = null
        denseMeshIndexWriter = null
        denseMeshStream = null
        denseMeshBytesWritten = 0L
        avatarWriter = null
        telemetryWriter = null
        captureTimingWriter = null
        hardwareReadoutTimingWriter = null
        sessionHealthWriter = null
        captureIntegrityWriter = null
        captureIntegritySummaryFile = null
        captureIntegrityFrameCount = 0L
        captureIntegrityOutputFrames = 0L
        captureIntegrityScoreSum = 0.0
        captureIntegrityScoreMin = Double.POSITIVE_INFINITY
        captureIntegrityScoreMax = Double.NEGATIVE_INFINITY
        captureIntegrityCurrentHoldStreak = 0L
        captureIntegrityMaxHoldStreak = 0L
        captureIntegrityPreviousOutputAllowed = null
        captureIntegrityBandCounts.clear()
        captureIntegrityBottleneckCounts.clear()
        captureIntegrityWarningCounts.clear()
        captureIntegrityHardFailureCounts.clear()
        pipelineWriter = null
        lightingWriter = null
        imuWriter = null
        focusWriter = null
        motionEventsWriter = null
        motionEvidenceWriter = null
        lightingConsistencyWriter = null
        trackLifecycleWriter = null
        crossModalIdentityWriter = null
        motionUncertaintyWriter = null
        tsdfEvidenceWriter = null
        motionDynamicsWriter = null
        productionPolicyWriter = null

        root?.let { directory ->
            runCatching {
                artifactIntegrity.writeManifest(directory)
            }
        }

        root = null
    }

    private fun closeWriter(
        writer: BufferedWriter?,
    ) {
        writer?.let {
            synchronized(it) {
                runCatching {
                    it.flush()
                    it.close()
                }
            }
        }
    }
}
