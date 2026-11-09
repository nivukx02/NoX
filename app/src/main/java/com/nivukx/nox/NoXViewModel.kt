package com.nivukx.nox

import com.nivukx.nox.core.model.SessionPresentationSettings
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.reconstruction.HighEndFaceShapeAlignmentEngine
import com.nivukx.nox.core.reconstruction.FaceEyeCalibrationDriftEngine
import com.nivukx.nox.core.reconstruction.CrossModalIdentityFusionEngine
import com.nivukx.nox.core.reconstruction.AdaptiveDenseMeshExtractionController
import com.nivukx.nox.core.reconstruction.DepthUncertaintyModelEngine
import com.nivukx.nox.core.reconstruction.LightingEnvironmentTemporalConsistencyEngine
import com.nivukx.nox.core.capture.CameraTopologyEvidenceEngine
import android.os.PowerManager

import android.app.Application
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.Settings
import android.os.SystemClock
import android.view.Surface
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nivukx.nox.core.ai.InferenceScheduler
import com.nivukx.nox.core.ai.CaptureIntegrityEngine
import com.nivukx.nox.core.ai.PerceptionPipeline
import com.nivukx.nox.core.ai.RecordedVideoPostProcessor
import com.nivukx.nox.core.ai.DenseLandmarkEvidenceFusionEngine
import com.nivukx.nox.core.ai.MotionCaptureTuning
import com.nivukx.nox.core.ai.FocusTargetEngine
import com.nivukx.nox.core.ai.FaceEyeConsistencyEngine
import com.nivukx.nox.core.ai.FaceMeshGeometryEngine
import com.nivukx.nox.core.ai.HandTemporalConsistencyEngine
import com.nivukx.nox.core.calibration.CalibrationEngine
import com.nivukx.nox.core.calibration.DeviceRoleAssignmentEngine
import com.nivukx.nox.core.calibration.RigidTransform
import com.nivukx.nox.core.calibration.RigOrientationEngine
import com.nivukx.nox.core.calibration.SelfCalibrationEngine
import com.nivukx.nox.core.calibration.RigQualityGate
import com.nivukx.nox.core.calibration.RigCoverageAnalyzer
import com.nivukx.nox.core.calibration.AdvancedRigCoveragePlanner
import com.nivukx.nox.core.capture.Camera2CaptureController
import com.nivukx.nox.core.capture.CameraCapturePolicyEngine
import com.nivukx.nox.core.capture.CameraCaptureQualityEngine
import com.nivukx.nox.core.capture.CaptureOrchestrator
import com.nivukx.nox.core.capture.ImuCaptureController
import com.nivukx.nox.core.capture.SessionRecorder
import com.nivukx.nox.core.capture.SessionRepository
import com.nivukx.nox.core.capture.ProductionCapturePreflightEngine
import com.nivukx.nox.core.capture.CameraSessionNegotiationEngine
import com.nivukx.nox.core.capture.ConcurrentCaptureCapabilityEngine
import com.nivukx.nox.core.capture.HardwareReadoutTimingEvidenceEngine
import com.nivukx.nox.core.capture.SessionHealthEngine
import com.nivukx.nox.core.capture.HybridCaptureCoordinator
import com.nivukx.nox.core.model.CaptureEvidenceMode
import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.RecordedVideoEvidenceProfile
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CaptureState
import com.nivukx.nox.core.model.CaptureIntegrityEvidence
import com.nivukx.nox.core.model.GpuBackend
import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceDescriptor
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.FrameStamp
import com.nivukx.nox.core.model.SessionId
import com.nivukx.nox.core.model.PipelineTelemetry
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.ProductionCaptureProfile
import com.nivukx.nox.core.model.ProductionPolicyRecord
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.MotionFieldActor
import com.nivukx.nox.core.model.MotionFieldGpuMode
import com.nivukx.nox.core.model.MotionFieldGpuState
import com.nivukx.nox.core.model.MotionFieldPreviewFrame
import com.nivukx.nox.core.model.MotionFieldWorldJoint
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.EyeSide
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.MonocularReconstructionFrame
import com.nivukx.nox.core.network.NearbyPeerController
import com.nivukx.nox.core.network.CalibrationSnapshotWireCodec
import com.nivukx.nox.core.network.CalibrationWireCodec
import com.nivukx.nox.core.network.DepthWireCodec
import com.nivukx.nox.core.network.DenseLandmarkWireCodec
import com.nivukx.nox.core.network.EyeWireCodec
import com.nivukx.nox.core.network.HandWireCodec
import com.nivukx.nox.core.network.FaceContourWireCodec
import com.nivukx.nox.core.network.FaceWireCodec
import com.nivukx.nox.core.network.FaceMeshWireCodec
import com.nivukx.nox.core.network.FaceMeshTransportPolicyEngine
import com.nivukx.nox.core.network.NoXWireProtocol
import com.nivukx.nox.core.network.PoseWireCodec
import com.nivukx.nox.core.network.PeerCapabilitiesWireCodec
import com.nivukx.nox.core.network.MultiActorPoseWireCodec
import com.nivukx.nox.core.sync.ClockSynchronizer
import com.nivukx.nox.core.sync.MultiCameraSynchronizationQualityEngine
import com.nivukx.nox.core.sync.MultiCameraTemporalAlignmentEngine
import com.nivukx.nox.core.reconstruction.Eye3DReconstructionEngine
import com.nivukx.nox.core.reconstruction.Eye3DTemporalConsistencyEngine
import com.nivukx.nox.core.reconstruction.FaceMultiViewFusionEngine
import com.nivukx.nox.core.reconstruction.GpuAccelerationEngine
import com.nivukx.nox.core.reconstruction.InertialFusionEngine
import com.nivukx.nox.core.reconstruction.LightingEnvironmentEngine
import com.nivukx.nox.core.reconstruction.MultiViewFusionEngine
import com.nivukx.nox.core.reconstruction.CrossCameraActorIdentityFusionEngine
import com.nivukx.nox.core.reconstruction.NativeSparseTsdfReconstructionBridge
import com.nivukx.nox.core.reconstruction.Global4DConsistencyEngine
import com.nivukx.nox.core.reconstruction.MultiCameraRigHealthEngine
import com.nivukx.nox.core.reconstruction.ReconstructionContinuityTuning
import com.nivukx.nox.core.reconstruction.TemporalReconstructionTuning
import com.nivukx.nox.core.reconstruction.SkeletonRetargetingEngine
import com.nivukx.nox.core.reconstruction.SparseTsdfFusionEngine
import com.nivukx.nox.core.reconstruction.AvatarMotionOutputGuard
import com.nivukx.nox.core.reconstruction.HighEndMeshQualityEngine
import com.nivukx.nox.core.reconstruction.HighEndKinematicConstraintEngine
import com.nivukx.nox.core.reconstruction.HighEndHandGeometryQualityEngine
import com.nivukx.nox.core.reconstruction.KinematicTrajectoryOptimizer
import com.nivukx.nox.core.reconstruction.HighEndBinocularGazeGeometryEngine
import com.nivukx.nox.core.reconstruction.DenseMeshTemporalConsistencyEngine
import com.nivukx.nox.core.reconstruction.MotionEventSegmentationEngine
import com.nivukx.nox.core.reconstruction.MotionFieldFrameSynchronizer
import com.nivukx.nox.core.reconstruction.MotionFieldGpuPolicyEngine
import com.nivukx.nox.core.reconstruction.MotionFieldDisplayPredictor
import com.nivukx.nox.core.reconstruction.MotionEvidenceFusionEngine
import com.nivukx.nox.core.reconstruction.MonocularMotionReconstructionPipeline
import com.nivukx.nox.core.reconstruction.JointRotationSolver
import com.nivukx.nox.core.reconstruction.HighEndMotionStateEstimator
import com.nivukx.nox.core.reconstruction.ProductionMotionDynamicsEngine
import com.nivukx.nox.core.reconstruction.CalibratedMotionUncertaintyEngine
import com.nivukx.nox.core.reconstruction.HighEndReconstructionQualityEngine
import com.nivukx.nox.core.reconstruction.PhysicalPrecisionCapabilityEngine
import com.nivukx.nox.core.reconstruction.DenseLandmarkPrecisionContractEngine
import com.nivukx.nox.core.reconstruction.PhysicsConstrainedSkeletonEngine
import com.nivukx.nox.core.reconstruction.WorldSpaceMotionConsistencyEngine
import com.nivukx.nox.core.reconstruction.Hand3DReconstructionEngine
import com.nivukx.nox.core.reconstruction.HandUncertaintyCalibrationEngine
import com.nivukx.nox.core.reconstruction.FaceMeshTemporalConsistencyEngine
import com.nivukx.nox.core.reconstruction.FaceEyeEvidenceFusionEngine
import com.nivukx.nox.core.reconstruction.FaceEyeMetricEvidenceEngine
import com.nivukx.nox.core.reconstruction.AdaptiveFaceEyeEvidenceFusionEngine
import com.nivukx.nox.core.reconstruction.TemporalDepthConsistencyEngine
import com.nivukx.nox.core.reconstruction.FaceMeshMultiViewFusionEngine
import com.nivukx.nox.core.reconstruction.FaceMeshSurfaceTemporalEngine
import com.nivukx.nox.core.reconstruction.AdaptiveReconstructionDetailEngine
import com.nivukx.nox.core.reconstruction.ThermalExtrinsicDriftEngine
import com.nivukx.nox.core.nativeengine.NativeEngineConfig
import com.nivukx.nox.core.nativeengine.NativeMotionEngine
import com.nivukx.nox.service.CaptureForegroundService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class NoXViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val CAPTURE_WIDTH = 1920
        private const val CAPTURE_HEIGHT = 1080
        private const val CAPTURE_FPS = 30
        private const val CALIBRATION_BUCKET_NANOS = 20_000_000L
        private const val CALIBRATION_WARMUP_MS = 700L
        private const val CALIBRATION_DURATION_MS = 2_500L
    }

    private val camera = Camera2CaptureController(application)
    private val hybridCaptureCoordinator =
        HybridCaptureCoordinator(application, camera)
    private val cameraCapturePolicyEngine = CameraCapturePolicyEngine()
    private val cameraCaptureQualityEngine = CameraCaptureQualityEngine()
    private val hardwareReadoutTimingEvidenceEngine = HardwareReadoutTimingEvidenceEngine()
    private val concurrentCaptureCapabilityEngine = ConcurrentCaptureCapabilityEngine()
    private val cameraTopologyEvidenceEngine = CameraTopologyEvidenceEngine()
    private var lastConcurrentRigRefreshNanos = Long.MIN_VALUE
    private val thermalExtrinsicDriftEngine = ThermalExtrinsicDriftEngine()
    private val peers = NearbyPeerController(application)
    private val clock = ClockSynchronizer()
    private val multiCameraSyncQualityEngine =
        MultiCameraSynchronizationQualityEngine()
    private val multiCameraTemporalAlignmentEngine =
        MultiCameraTemporalAlignmentEngine()
    private val orchestrator = CaptureOrchestrator(clock)
    private val perception = PerceptionPipeline(application)
    private val postCaptureReconstructionBridge =
        com.nivukx.nox.core.reconstruction.PostCaptureReconstructionBridge(
            calibrationProvider = { calibrationSnapshot },
        )
    private val crossCameraActorIdentityFusionEngine =
        CrossCameraActorIdentityFusionEngine {
            calibrationSnapshot
        }
    private val recordedVideoPostProcessor =
        RecordedVideoPostProcessor(
            perception = perception,
            reconstructionCallback = { result ->
                postCaptureReconstructionBridge
                    .pushAndReconstruct(
                        result = result,
                        cameraId =
                            DeviceId(
                                result.multiActorPose
                                    ?.let { _ ->
                                        "offline-camera"
                                    }
                                    ?: "offline-camera",
                            ),
                        customization =
                            _motionCustomization.value,
                    )
                    .acceptedActorCount
            },
        )

    private val nativeMotionEngine: NativeMotionEngine? by lazy {
        runCatching {
            NativeMotionEngine.create(
                NativeEngineConfig(
                    workerCount =
                        Runtime.getRuntime().availableProcessors()
                            .coerceAtLeast(2)
                            .coerceAtMost(8),
                    observationCapacity = 8192,
                    deterministicSeed = 0x4E4F5846,
                    frameBudgetNs = 16_666_667L,
                    validationLevel = 1,
                ),
            )
        }.getOrNull()
    }

    private var nativeMotionEngineStarted = false
    private var lastNativeThermalStatus = Int.MIN_VALUE
    private var lastNativeMemoryPressure = Double.NaN
    private var lastNativeGpuHeadroom = Double.NaN
    private var lastNativeNpuHeadroom = Double.NaN
    private var lastNativeBatterySaver = false
    private val focusTargetEngine = FocusTargetEngine()
    private val inferenceScheduler = InferenceScheduler(application)
    private val calibration = CalibrationEngine()
    private val roleAssignment = DeviceRoleAssignmentEngine()
    private val orientation = RigOrientationEngine(application)
    private val selfCalibration = SelfCalibrationEngine()
    private val rigQualityGate = RigQualityGate(maxReprojectionErrorPx = 4.0)
    private val rigCoverage = RigCoverageAnalyzer()
    private val advancedRigCoveragePlanner = AdvancedRigCoveragePlanner()
    private val recorder = SessionRecorder(application)
    private val sessions = SessionRepository(application)
    private val preflightEngine = ProductionCapturePreflightEngine()
    private val sessionHealthEngine = SessionHealthEngine()
    private val captureIntegrityEngine = CaptureIntegrityEngine()
    private val captureEvidenceLedger =
        com.nivukx.nox.core.ai.ProductionCaptureEvidenceLedger()
    private val controlPreferences = application.getSharedPreferences(
        "nox_capture_controls",
        Application.MODE_PRIVATE,
    )

    private val localId: DeviceId by lazy {
        val stable = Settings.Secure.getString(getApplication<Application>().contentResolver, Settings.Secure.ANDROID_ID)
            ?.takeIf { it.isNotBlank() }
            ?: android.os.Build.MODEL
        DeviceId("nox-" + stable)
    }
    private val packetSequence = AtomicLong(10)
    private val pendingPings = ConcurrentHashMap<Long, Long>()
    private val pendingPingCapacity = 4096
    private val pendingPingTtlNanos = 5_000_000_000L
    private val latestMasterPoseTimestamps =
        ConcurrentHashMap<DeviceId, Long>()
    private var calibrationSnapshot: CalibrationSnapshot? = null
    private var nativeDenseTsdfBridge:
        NativeSparseTsdfReconstructionBridge? = null
    private var nativeDenseTsdfFaultReported = false

    private val fusion =
        MultiViewFusionEngine(
            calibrationProvider = { calibrationSnapshot },
        ).apply {
            configureGaussian4DEvidenceProvider {
                nativeDenseTsdfBridge
                ?.gaussianSurfaceTelemetry()
                ?.takeIf { telemetry ->
                    telemetry.nativeTelemetryAvailable
                }
                ?.let { telemetry ->
                    Global4DConsistencyEngine.Gaussian4DEvidence(
                        temporalStability =
                            telemetry.deformationTemporalStability,
                        trustedRatio =
                            telemetry.deformationTrustedBindingRatio,
                        meanNodeConfidence =
                            telemetry.deformationMeanNodeConfidence,
                        meanBindingConfidence =
                            telemetry.deformationMeanBindingConfidence,
                        p95StrainRatio =
                            telemetry.deformationP95StrainRatio,
                        maximumStrainRatio =
                            telemetry.deformationMaximumStrainRatio,
                    )
                }
            }
        }
    private val multiActor3DReconstructionCoordinator =
        com.nivukx.nox.core.reconstruction.MultiActor3DReconstructionCoordinator(
            fusion,
        )
    private val rigHealthEngine = MultiCameraRigHealthEngine()
    private val retargeter = SkeletonRetargetingEngine()
    private val avatarOutputGuard = AvatarMotionOutputGuard()
    private val motionEventEngine = MotionEventSegmentationEngine()
    private val faceFusion = FaceMultiViewFusionEngine()
    private val eyeFusion = Eye3DReconstructionEngine()
    private val eye3DTemporalConsistencyEngine =
        Eye3DTemporalConsistencyEngine()
    private val binocularGazeGeometryEngine =
        HighEndBinocularGazeGeometryEngine()
    private val handFusion = Hand3DReconstructionEngine()
    private val faceEyeConsistencyEngine = FaceEyeConsistencyEngine()
    private val handTemporalConsistencyEngine =
        HandTemporalConsistencyEngine()
    private val handGeometryQualityEngine =
        HighEndHandGeometryQualityEngine()
    private val handUncertaintyCalibrationEngine =
        HandUncertaintyCalibrationEngine()
    private val faceMeshTemporalConsistencyEngine =
        FaceMeshTemporalConsistencyEngine()
    private val faceMeshFusion =
        FaceMeshMultiViewFusionEngine()
    private val faceMeshSurfaceTemporalEngine =
        FaceMeshSurfaceTemporalEngine()
    private val faceMeshTransportPolicy =
        FaceMeshTransportPolicyEngine()
    private val lightingEngine = LightingEnvironmentEngine()
    private val lightingTemporalConsistencyEngine =
        LightingEnvironmentTemporalConsistencyEngine()
    private val imuCapture = ImuCaptureController(application)
    private val inertialFusion = InertialFusionEngine()
    private val gpuAcceleration = GpuAccelerationEngine(application)
    private val handFrames =
        ConcurrentHashMap<DeviceId, Hand3DReconstructionEngine.CameraFrame>()
    private val denseFusion = SparseTsdfFusionEngine(
        voxelSizeMeters = 0.0125,
        truncationDistanceMeters = 0.05,
        maxVoxels = 1_000_000,
    )
    private val depthUncertaintyModelEngine =
        DepthUncertaintyModelEngine(
            DepthUncertaintyModelEngine.Config(
                baseSigmaMeters = 0.006,
                rangeLinearCoefficient = 0.0035,
                rangeQuadraticCoefficient = 0.0010,
            ),
        )
    private val denseMeshQualityEngine = HighEndMeshQualityEngine()
    private val denseMeshTemporalConsistencyEngine =
        DenseMeshTemporalConsistencyEngine()
    private val adaptiveDenseMeshExtractionController =
        AdaptiveDenseMeshExtractionController()
    private val temporalDepthConsistencyEngine =
        TemporalDepthConsistencyEngine()
    private var latestTemporalDepthConsistency = 0.0
    private val motionEvidenceEngine =
        MotionEvidenceFusionEngine()
    private val monocularPipeline =
        MonocularMotionReconstructionPipeline()
    private val monocularRotationSolver =
        JointRotationSolver()
    private val monocularKinematicConstraintEngine =
        HighEndKinematicConstraintEngine()
    private val monocularTrajectoryOptimizer =
        KinematicTrajectoryOptimizer()
    private val monocularPhysicsConstraintEngine =
        PhysicsConstrainedSkeletonEngine()
    private val motionStateEstimator =
        HighEndMotionStateEstimator()
    private val productionQualityEngine =
        HighEndReconstructionQualityEngine()
    private val adaptiveReconstructionDetailEngine =
        AdaptiveReconstructionDetailEngine()
    private val physicalPrecisionCapabilityEngine =
        PhysicalPrecisionCapabilityEngine()
    private val denseLandmarkPrecisionContractEngine =
        DenseLandmarkPrecisionContractEngine()
    private val globalDenseLandmarkEvidenceEngine =
        DenseLandmarkEvidenceFusionEngine(
            DenseLandmarkEvidenceFusionEngine.Config(
                maximumCameras = 32,
                maximumSubjects = 50,
                maximumLandmarksPerSubject = 1024,
                maximumStoredLandmarks = 64_000,
                maximumObservationsPerFrame = 65_536,
                maximumHistoryFrames = 18,
                staleObservationMillis = 140.0,
                staleCameraMillis = 1_200.0,
                occlusionPredictionHorizonMillis = 96.0,
            ),
        )
    private val denseEvidenceCameraIds =
        ConcurrentHashMap<DeviceId, Int>()
    private val productionMotionDynamicsEngine =
        ProductionMotionDynamicsEngine()
    private val calibratedMotionUncertaintyEngine =
        CalibratedMotionUncertaintyEngine()
    private val crossModalIdentityFusionEngine =
        CrossModalIdentityFusionEngine()
    private val faceEyeEvidenceFusionEngine =
        FaceEyeEvidenceFusionEngine()
    private val adaptiveFaceEyeEvidenceFusionEngine =
        AdaptiveFaceEyeEvidenceFusionEngine()
    private val faceEyeMetricEvidenceEngine =
        FaceEyeMetricEvidenceEngine()
    private val faceEyeCalibrationDriftEngine =
        FaceEyeCalibrationDriftEngine()
    private val faceShapeAlignmentEngine =
        HighEndFaceShapeAlignmentEngine()
    private val motionFieldSynchronizer =
        MotionFieldFrameSynchronizer()
    private val motionFieldGpuPolicyEngine =
        MotionFieldGpuPolicyEngine()
    private val motionFieldDisplayPredictor =
        MotionFieldDisplayPredictor()


    private var activeCaptureProfile =
        CameraCaptureProfile.adaptive(1920, 1080, 30)

    @Volatile
    private var previewSurfaceReady = false

    private var lastDenseMeshExtractNanos = Long.MIN_VALUE
    private var previousMonocularFrame:
        MonocularReconstructionFrame? = null
    private var previousMonocularAvatarPose:
        com.nivukx.nox.core.model.AvatarPose? = null

    private val crossModalObservationSequence = AtomicLong(1L)
    private var latestFaceCenter3D: Vec3? = null
    private var latestFaceCenterTimestampNanos = Long.MIN_VALUE
    private var latestLeftHand3D: Vec3? = null
    private var latestLeftHandTimestampNanos = Long.MIN_VALUE
    private var latestRightHand3D: Vec3? = null
    private var latestRightHandTimestampNanos = Long.MIN_VALUE
    private var latestLeftEye3D: Vec3? = null
    private var latestLeftEyeTimestampNanos = Long.MIN_VALUE
    private var latestRightEye3D: Vec3? = null
    private var latestRightEyeTimestampNanos = Long.MIN_VALUE
    private var latestFaceShapeConfidence = 0.0
    private var latestFaceShapeAccepted = false
    private var latestFacePointCount = 0
    private var latestDenseFaceObserved = false


    private var frameJob: Job? = null
    private var captureGeneration: Long = 0L
    private var imuJob: Job? = null
    private var depthJob: Job? = null
    private var scheduledStart: Job? = null
    private var activeHybridSessionId: String? = null
    private var calibrationJob: Job? = null
    private var calibrationReferenceDevice: DeviceId? = null

    private val calibrationDescriptors =
        ConcurrentHashMap<DeviceId, CalibrationWireCodec.Frame>()
    private val peerCapabilities =
        ConcurrentHashMap<DeviceId, PeerCapabilitiesWireCodec.Capabilities>()
    private val calibrationViews =
        ConcurrentHashMap<String, SelfCalibrationEngine.LandmarkView>()
    private val calibrationRawTimestamps =
        ConcurrentHashMap<String, Long>()
    private val assignedRoles =
        ConcurrentHashMap<DeviceId, DeviceRole>()

    private val _state = MutableStateFlow(CaptureState.IDLE)
    val state: StateFlow<CaptureState> = _state.asStateFlow()

    private val _devices = MutableStateFlow<List<DeviceDescriptor>>(emptyList())
    val devices: StateFlow<List<DeviceDescriptor>> = _devices.asStateFlow()

    private val _frameCount = MutableStateFlow(0L)
    val frameCount: StateFlow<Long> = _frameCount.asStateFlow()

    private val _poseCount = MutableStateFlow(0)
    val poseCount: StateFlow<Int> = _poseCount.asStateFlow()

    private val _poseConfidence = MutableStateFlow(0f)
    val poseConfidence: StateFlow<Float> = _poseConfidence.asStateFlow()

    private val _multiPersonObservedCount = MutableStateFlow(0)
    val multiPersonObservedCount: StateFlow<Int> =
        _multiPersonObservedCount.asStateFlow()

    private val _multiPersonAdmittedCount = MutableStateFlow(0)
    val multiPersonAdmittedCount: StateFlow<Int> =
        _multiPersonAdmittedCount.asStateFlow()

    private val _multiPersonHeldCount = MutableStateFlow(0)
    val multiPersonHeldCount: StateFlow<Int> =
        _multiPersonHeldCount.asStateFlow()

    private val _multiPersonStarvationRisk = MutableStateFlow(0.0)
    val multiPersonStarvationRisk: StateFlow<Double> =
        _multiPersonStarvationRisk.asStateFlow()

    private val _multiPersonFairnessBoostedCount = MutableStateFlow(0)
    private val _multiPersonIdentityContinuity = MutableStateFlow(0.0)
    val multiPersonIdentityContinuity: StateFlow<Double> =
        _multiPersonIdentityContinuity.asStateFlow()

    private val _multiPersonIdentityReidentifications =
        MutableStateFlow(0)
    val multiPersonIdentityReidentifications: StateFlow<Int> =
        _multiPersonIdentityReidentifications.asStateFlow()

    private val _multiPersonIdentityPredictions =
        MutableStateFlow(0)
    val multiPersonIdentityPredictions: StateFlow<Int> =
        _multiPersonIdentityPredictions.asStateFlow()

    val multiPersonFairnessBoostedCount: StateFlow<Int> =
        _multiPersonFairnessBoostedCount.asStateFlow()

    private val _crowdMotionMeanIdentitySafety =
        MutableStateFlow(0.0)
    val crowdMotionMeanIdentitySafety: StateFlow<Double> =
        _crowdMotionMeanIdentitySafety.asStateFlow()

    private val _crowdMotionInteractionPressure =
        MutableStateFlow(0.0)
    val crowdMotionInteractionPressure: StateFlow<Double> =
        _crowdMotionInteractionPressure.asStateFlow()

    private val _crowdMotionDifferentialScore =
        MutableStateFlow(0.0)
    val crowdMotionDifferentialScore: StateFlow<Double> =
        _crowdMotionDifferentialScore.asStateFlow()

    private val _crowdMotionSchedulingCapacity =
        MutableStateFlow(0.0)
    val crowdMotionSchedulingCapacity: StateFlow<Double> =
        _crowdMotionSchedulingCapacity.asStateFlow()

    private val _crowdWorldLandmarkCoverage =
        MutableStateFlow(0.0)
    val crowdWorldLandmarkCoverage: StateFlow<Double> =
        _crowdWorldLandmarkCoverage.asStateFlow()

    private val _crowdWorldBoneConsistency =
        MutableStateFlow(0.0)
    val crowdWorldBoneConsistency: StateFlow<Double> =
        _crowdWorldBoneConsistency.asStateFlow()

    private val _crowdWorldGeometryReliability =
        MutableStateFlow(0.0)
    val crowdWorldGeometryReliability: StateFlow<Double> =
        _crowdWorldGeometryReliability.asStateFlow()

    private val _crowdAppearanceQuality =
        MutableStateFlow(0.0)
    val crowdAppearanceQuality: StateFlow<Double> =
        _crowdAppearanceQuality.asStateFlow()

    private val _crowdSegmentationEvidence =
        MutableStateFlow(0.0)
    val crowdSegmentationEvidence: StateFlow<Double> =
        _crowdSegmentationEvidence.asStateFlow()

    private val _crowdSegmentationCoverage =
        MutableStateFlow(0.0)
    val crowdSegmentationCoverage: StateFlow<Double> =
        _crowdSegmentationCoverage.asStateFlow()

    private val _crowdMotionBottleneck =
        MutableStateFlow(
            com.nivukx.nox.core.ai.ProductionCrowdMotionEvidenceEngine.Bottleneck.NONE.name,
        )
    val crowdMotionBottleneck: StateFlow<String> =
        _crowdMotionBottleneck.asStateFlow()

    private val _crowdMotionHeldCount =
        MutableStateFlow(0)
    val crowdMotionHeldCount: StateFlow<Int> =
        _crowdMotionHeldCount.asStateFlow()

    private val _crowdMotionPredictedCount =
        MutableStateFlow(0)
    val crowdMotionPredictedCount: StateFlow<Int> =
        _crowdMotionPredictedCount.asStateFlow()

    private val _crowdMotionWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val crowdMotionWarnings: StateFlow<List<String>> =
        _crowdMotionWarnings.asStateFlow()


    private val _multiPerson3DReconstructedCount =
        MutableStateFlow(0)
    val multiPerson3DReconstructedCount: StateFlow<Int> =
        _multiPerson3DReconstructedCount.asStateFlow()

    private val _multiPerson3DDetailedCount =
        MutableStateFlow(0)
    val multiPerson3DDetailedCount: StateFlow<Int> =
        _multiPerson3DDetailedCount.asStateFlow()

    private val _multiPerson3DPredictedCount =
        MutableStateFlow(0)
    val multiPerson3DPredictedCount: StateFlow<Int> =
        _multiPerson3DPredictedCount.asStateFlow()

    private val _multiPerson3DPredictionMaxAgeMillis =
        MutableStateFlow(0L)
    val multiPerson3DPredictionMaxAgeMillis: StateFlow<Long> =
        _multiPerson3DPredictionMaxAgeMillis.asStateFlow()

    private val _denseLandmarkCoverage = MutableStateFlow(0.0)
    val denseLandmarkCoverage: StateFlow<Double> =
        _denseLandmarkCoverage.asStateFlow()

    private val _denseLandmarkVisibleCoverage = MutableStateFlow(0.0)
    val denseLandmarkVisibleCoverage: StateFlow<Double> =
        _denseLandmarkVisibleCoverage.asStateFlow()

    private val _denseLandmarkUncertaintyP95 = MutableStateFlow(1.0)
    val denseLandmarkUncertaintyP95: StateFlow<Double> =
        _denseLandmarkUncertaintyP95.asStateFlow()

    private val _denseLandmarkCrossViewAgreement = MutableStateFlow(0.0)
    val denseLandmarkCrossViewAgreement: StateFlow<Double> =
        _denseLandmarkCrossViewAgreement.asStateFlow()

    private val _denseLandmarkCrossViewEvidenceAvailable =
        MutableStateFlow(false)
    val denseLandmarkCrossViewEvidenceAvailable: StateFlow<Boolean> =
        _denseLandmarkCrossViewEvidenceAvailable.asStateFlow()

    private val _denseLandmarkTemporalStability = MutableStateFlow(0.0)
    val denseLandmarkTemporalStability: StateFlow<Double> =
        _denseLandmarkTemporalStability.asStateFlow()

    private val _denseLandmarkContactCoverage = MutableStateFlow(0.0)
    val denseLandmarkContactCoverage: StateFlow<Double> =
        _denseLandmarkContactCoverage.asStateFlow()

    private val _denseLandmarkPredictionBridgeRatio = MutableStateFlow(0.0)
    val denseLandmarkPredictionBridgeRatio: StateFlow<Double> =
        _denseLandmarkPredictionBridgeRatio.asStateFlow()

    private val _denseLandmarkRejectedObservations =
        MutableStateFlow(0L)
    val denseLandmarkRejectedObservations: StateFlow<Long> =
        _denseLandmarkRejectedObservations.asStateFlow()

    private val _denseLandmarkStaleObservations =
        MutableStateFlow(0L)
    val denseLandmarkStaleObservations: StateFlow<Long> =
        _denseLandmarkStaleObservations.asStateFlow()

    private val _denseLandmarkNonFiniteObservations =
        MutableStateFlow(0L)
    val denseLandmarkNonFiniteObservations: StateFlow<Long> =
        _denseLandmarkNonFiniteObservations.asStateFlow()

    private val _denseLandmarkHardFailure =
        MutableStateFlow(false)
    val denseLandmarkHardFailure: StateFlow<Boolean> =
        _denseLandmarkHardFailure.asStateFlow()

    private val _denseLandmarkContractScore =
        MutableStateFlow(0.0)
    val denseLandmarkContractScore: StateFlow<Double> =
        _denseLandmarkContractScore.asStateFlow()

    private val _denseLandmarkContractState =
        MutableStateFlow(
            DenseLandmarkPrecisionContractEngine.State.UNKNOWN.name,
        )
    val denseLandmarkContractState: StateFlow<String> =
        _denseLandmarkContractState.asStateFlow()

    private val _denseLandmarkContractAuthoritative =
        MutableStateFlow(false)
    val denseLandmarkContractAuthoritative: StateFlow<Boolean> =
        _denseLandmarkContractAuthoritative.asStateFlow()

    private val _denseLandmarkContractFailure =
        MutableStateFlow(
            DenseLandmarkPrecisionContractEngine.FailureClass.NONE.name,
        )
    val denseLandmarkContractFailure: StateFlow<String> =
        _denseLandmarkContractFailure.asStateFlow()

    private val _denseGeometricConfidence = MutableStateFlow(0.0)
    val denseGeometricConfidence: StateFlow<Double> =
        _denseGeometricConfidence.asStateFlow()

    private val _denseViewDiversity = MutableStateFlow(0.0)
    val denseViewDiversity: StateFlow<Double> =
        _denseViewDiversity.asStateFlow()

    private val _denseMotionBoundaryConfidence = MutableStateFlow(0.0)
    val denseMotionBoundaryConfidence: StateFlow<Double> =
        _denseMotionBoundaryConfidence.asStateFlow()

    private val _denseAppearanceConsistency = MutableStateFlow(0.0)
    val denseAppearanceConsistency: StateFlow<Double> =
        _denseAppearanceConsistency.asStateFlow()

    private val _denseDepthConfidence = MutableStateFlow(0.0)
    val denseDepthConfidence: StateFlow<Double> =
        _denseDepthConfidence.asStateFlow()

    private val _denseTemporalInnovationP95 = MutableStateFlow(0.0)
    val denseTemporalInnovationP95: StateFlow<Double> =
        _denseTemporalInnovationP95.asStateFlow()

    private val _gazeBlinkProbability = MutableStateFlow(0.0)
    val gazeBlinkProbability: StateFlow<Double> =
        _gazeBlinkProbability.asStateFlow()

    private val _crowdActorSilhouetteSupport =
        MutableStateFlow(0.0)
    val crowdActorSilhouetteSupport: StateFlow<Double> =
        _crowdActorSilhouetteSupport.asStateFlow()

    private val _crowdActorSilhouetteAmbiguity =
        MutableStateFlow(0.0)
    val crowdActorSilhouetteAmbiguity: StateFlow<Double> =
        _crowdActorSilhouetteAmbiguity.asStateFlow()

    private val _personSegmentationCoverage =
        MutableStateFlow(0.0)
    val personSegmentationCoverage: StateFlow<Double> =
        _personSegmentationCoverage.asStateFlow()

    private val _personSegmentationConfidence =
        MutableStateFlow(0.0)
    val personSegmentationConfidence: StateFlow<Double> =
        _personSegmentationConfidence.asStateFlow()

    private val _personSegmentationBoundaryStability =
        MutableStateFlow(0.0)
    val personSegmentationBoundaryStability: StateFlow<Double> =
        _personSegmentationBoundaryStability.asStateFlow()

    private val _personSegmentationLatencyMs =
        MutableStateFlow(0L)
    val personSegmentationLatencyMs: StateFlow<Long> =
        _personSegmentationLatencyMs.asStateFlow()

    private val _personSegmentationState =
        MutableStateFlow("DEGRADED")
    val personSegmentationState: StateFlow<String> =
        _personSegmentationState.asStateFlow()

    private val _personSegmentationDelegate =
        MutableStateFlow("NONE")
    val personSegmentationDelegate: StateFlow<String> =
        _personSegmentationDelegate.asStateFlow()

    private val _personSegmentationFallbackCount =
        MutableStateFlow(0L)
    val personSegmentationFallbackCount: StateFlow<Long> =
        _personSegmentationFallbackCount.asStateFlow()

    private val _denseSurfaceTrustedPointRatio =
        MutableStateFlow(0.0)
    val denseSurfaceTrustedPointRatio: StateFlow<Double> =
        _denseSurfaceTrustedPointRatio.asStateFlow()

    private val _denseSurfaceUncertaintyP95Meters =
        MutableStateFlow(Double.POSITIVE_INFINITY)
    val denseSurfaceUncertaintyP95Meters: StateFlow<Double> =
        _denseSurfaceUncertaintyP95Meters.asStateFlow()

    private val _denseSurfaceTopologyStability =
        MutableStateFlow(0.0)
    val denseSurfaceTopologyStability: StateFlow<Double> =
        _denseSurfaceTopologyStability.asStateFlow()

    private val _denseSurfaceViewDiversity =
        MutableStateFlow(0.0)
    val denseSurfaceViewDiversity: StateFlow<Double> =
        _denseSurfaceViewDiversity.asStateFlow()

    private val _denseSurfaceProbabilisticEvidenceScore =
        MutableStateFlow(1.0)
    val denseSurfaceProbabilisticEvidenceScore: StateFlow<Double> =
        _denseSurfaceProbabilisticEvidenceScore.asStateFlow()

    private val _denseSurfaceAdaptiveViewSelectionScore =
        MutableStateFlow(0.0)
    val denseSurfaceAdaptiveViewSelectionScore: StateFlow<Double> =
        _denseSurfaceAdaptiveViewSelectionScore.asStateFlow()

    private val _denseSurfaceAdaptiveViewDiversityScore =
        MutableStateFlow(0.0)
    val denseSurfaceAdaptiveViewDiversityScore: StateFlow<Double> =
        _denseSurfaceAdaptiveViewDiversityScore.asStateFlow()

    private val _denseSurfaceExcludedWeakViewCount =
        MutableStateFlow(0)
    val denseSurfaceExcludedWeakViewCount: StateFlow<Int> =
        _denseSurfaceExcludedWeakViewCount.asStateFlow()

    private val _denseSurfaceMeanReprojectionErrorPx =
        MutableStateFlow(Double.POSITIVE_INFINITY)
    val denseSurfaceMeanReprojectionErrorPx: StateFlow<Double> =
        _denseSurfaceMeanReprojectionErrorPx.asStateFlow()

    private val _denseSurfaceMeanAppearanceConsistency =
        MutableStateFlow(0.0)
    val denseSurfaceMeanAppearanceConsistency: StateFlow<Double> =
        _denseSurfaceMeanAppearanceConsistency.asStateFlow()

    private val _denseSurfaceMeanSilhouetteConsistency =
        MutableStateFlow(0.0)
    val denseSurfaceMeanSilhouetteConsistency: StateFlow<Double> =
        _denseSurfaceMeanSilhouetteConsistency.asStateFlow()

    private val _denseSurfaceMeanMotionBoundaryConfidence =
        MutableStateFlow(0.0)
    val denseSurfaceMeanMotionBoundaryConfidence: StateFlow<Double> =
        _denseSurfaceMeanMotionBoundaryConfidence.asStateFlow()

    private val _denseSurfaceMeanLightingStability =
        MutableStateFlow(0.0)
    val denseSurfaceMeanLightingStability: StateFlow<Double> =
        _denseSurfaceMeanLightingStability.asStateFlow()

    private val _denseSurfaceHardFailure =
        MutableStateFlow(false)
    val denseSurfaceHardFailure: StateFlow<Boolean> =
        _denseSurfaceHardFailure.asStateFlow()

    private val _denseGeometricConditioningScore =
        MutableStateFlow(0.0)
    val denseGeometricConditioningScore: StateFlow<Double> =
        _denseGeometricConditioningScore.asStateFlow()

    private val _denseGeometricConditioningMedianParallaxDegrees =
        MutableStateFlow(0.0)
    val denseGeometricConditioningMedianParallaxDegrees:
        StateFlow<Double> =
        _denseGeometricConditioningMedianParallaxDegrees
            .asStateFlow()

    private val _denseGeometricConditioningMinimumObservationScore =
        MutableStateFlow(1.0)
    val denseGeometricConditioningMinimumObservationScore:
        StateFlow<Double> =
        _denseGeometricConditioningMinimumObservationScore
            .asStateFlow()

    private val _denseGeometricConditioningWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val denseGeometricConditioningWarnings:
        StateFlow<List<String>> =
        _denseGeometricConditioningWarnings.asStateFlow()

    private val _denseIdentityGateQuality =
        MutableStateFlow(1.0)
    val denseIdentityGateQuality: StateFlow<Double> =
        _denseIdentityGateQuality.asStateFlow()

    private val _denseIdentityGateRejectedCount =
        MutableStateFlow(0)
    val denseIdentityGateRejectedCount: StateFlow<Int> =
        _denseIdentityGateRejectedCount.asStateFlow()

    private val _denseIdentityGateEvaluatedPoints =
        MutableStateFlow(0)
    val denseIdentityGateEvaluatedPoints: StateFlow<Int> =
        _denseIdentityGateEvaluatedPoints.asStateFlow()

    private val _denseIdentityGateWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val denseIdentityGateWarnings: StateFlow<List<String>> =
        _denseIdentityGateWarnings.asStateFlow()

    private val _denseContactConfidence =
        MutableStateFlow(0.0)
    val denseContactConfidence: StateFlow<Double> =
        _denseContactConfidence.asStateFlow()

    private val _denseContactProtectedLandmarkCount =
        MutableStateFlow(0)
    val denseContactProtectedLandmarkCount: StateFlow<Int> =
        _denseContactProtectedLandmarkCount.asStateFlow()

    private val _denseContactWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val denseContactWarnings: StateFlow<List<String>> =
        _denseContactWarnings.asStateFlow()

    private val _handCount = MutableStateFlow(0)
    val handCount: StateFlow<Int> = _handCount.asStateFlow()

    private val _handConfidence = MutableStateFlow(0.0)
    val handConfidence: StateFlow<Double> = _handConfidence.asStateFlow()

    private val _hand3DConfidence = MutableStateFlow(0.0)
    val hand3DConfidence: StateFlow<Double> = _hand3DConfidence.asStateFlow()

    private val _handTemporalConsistencyConfidence =
        MutableStateFlow(0.0)
    val handTemporalConsistencyConfidence: StateFlow<Double> =
        _handTemporalConsistencyConfidence.asStateFlow()

    private val _handLongitudinalAnatomyConsistency =
        MutableStateFlow(0.78)
    val handLongitudinalAnatomyConsistency: StateFlow<Double> =
        _handLongitudinalAnatomyConsistency.asStateFlow()

    private val _handUncertaintyConfidence =
        MutableStateFlow(1.0)
    val handUncertaintyConfidence: StateFlow<Double> =
        _handUncertaintyConfidence.asStateFlow()

    private val _handUncertaintyRadiusMeters =
        MutableStateFlow(0.0)
    val handUncertaintyRadiusMeters: StateFlow<Double> =
        _handUncertaintyRadiusMeters.asStateFlow()

    private val _handUncertaintyCoverage =
        MutableStateFlow(0.0)
    val handUncertaintyCoverage: StateFlow<Double> =
        _handUncertaintyCoverage.asStateFlow()

    private val _handUncertaintyOodRisk =
        MutableStateFlow(0.0)
    val handUncertaintyOodRisk: StateFlow<Double> =
        _handUncertaintyOodRisk.asStateFlow()

    private val _handUncertaintyPolicy =
        MutableStateFlow("NORMAL")
    val handUncertaintyPolicy: StateFlow<String> =
        _handUncertaintyPolicy.asStateFlow()

    private val _handUncertaintyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val handUncertaintyWarnings: StateFlow<List<String>> =
        _handUncertaintyWarnings.asStateFlow()

    private val _handTemporalConsistencyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val handTemporalConsistencyWarnings: StateFlow<List<String>> =
        _handTemporalConsistencyWarnings.asStateFlow()

    private val _focusMode = MutableStateFlow("NONE")
    val focusMode: StateFlow<String> = _focusMode.asStateFlow()

    private val _focusConfidence = MutableStateFlow(0.0)
    val focusConfidence: StateFlow<Double> = _focusConfidence.asStateFlow()

    private val _subjectLockState = MutableStateFlow("SEARCHING")
    val subjectLockState: StateFlow<String> = _subjectLockState.asStateFlow()

    private val _subjectLockConfidence = MutableStateFlow(0.0)
    val subjectLockConfidence: StateFlow<Double> = _subjectLockConfidence.asStateFlow()

    private val _subjectLockTrackId = MutableStateFlow<Long?>(null)
    val subjectLockTrackId: StateFlow<Long?> = _subjectLockTrackId.asStateFlow()

    private val _subjectFocusX = MutableStateFlow(0.5)
    val subjectFocusX: StateFlow<Double> = _subjectFocusX.asStateFlow()

    private val _subjectFocusY = MutableStateFlow(0.5)
    val subjectFocusY: StateFlow<Double> = _subjectFocusY.asStateFlow()
    private val _identityContinuityScore = MutableStateFlow(0.0)
    val identityContinuityScore: StateFlow<Double> =
        _identityContinuityScore.asStateFlow()

    private val _identitySwapRisk = MutableStateFlow(0.0)
    val identitySwapRisk: StateFlow<Double> =
        _identitySwapRisk.asStateFlow()

    private val _identityAction = MutableStateFlow("SEARCH")
    val identityAction: StateFlow<String> =
        _identityAction.asStateFlow()

    private val _identityReason = MutableStateFlow("not_evaluated")
    val identityReason: StateFlow<String> =
        _identityReason.asStateFlow()

    private val _identityEvidenceFusionScore =
        MutableStateFlow(0.0)
    val identityEvidenceFusionScore: StateFlow<Double> =
        _identityEvidenceFusionScore.asStateFlow()

    private val _identitySelectedModalityCount =
        MutableStateFlow(0)
    val identitySelectedModalityCount: StateFlow<Int> =
        _identitySelectedModalityCount.asStateFlow()

    private val _identityEvidenceWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val identityEvidenceWarnings: StateFlow<List<String>> =
        _identityEvidenceWarnings.asStateFlow()

        private val _faceEyeMetricState =
        MutableStateFlow(
            FaceEyeMetricEvidenceEngine.State.NO_DATA.name,
        )
    val faceEyeMetricState: StateFlow<String> =
        _faceEyeMetricState.asStateFlow()

    private val _faceEyeMetricConfidence =
        MutableStateFlow(0.0)
    val faceEyeMetricConfidence: StateFlow<Double> =
        _faceEyeMetricConfidence.asStateFlow()

    private val _faceEyeMetricEyeUncertaintyMm =
        MutableStateFlow(Double.POSITIVE_INFINITY)
    val faceEyeMetricEyeUncertaintyMm: StateFlow<Double> =
        _faceEyeMetricEyeUncertaintyMm.asStateFlow()

    private val _faceEyeMetricWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeMetricWarnings: StateFlow<List<String>> =
        _faceEyeMetricWarnings.asStateFlow()

    private val _faceEyeMetricRecommendations =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeMetricRecommendations: StateFlow<List<String>> =
        _faceEyeMetricRecommendations.asStateFlow()

private val _faceEyeFusionConfidence =
        MutableStateFlow(0.0)
    val faceEyeFusionConfidence: StateFlow<Double> =
        _faceEyeFusionConfidence.asStateFlow()

    private val _faceEyeFusionAccepted =
        MutableStateFlow(false)
    val faceEyeFusionAccepted: StateFlow<Boolean> =
        _faceEyeFusionAccepted.asStateFlow()

    private val _faceEyeFusionMode =
        MutableStateFlow(
            FaceEyeEvidenceFusionEngine.FusionMode.HOLD.name,
        )
    val faceEyeFusionMode: StateFlow<String> =
        _faceEyeFusionMode.asStateFlow()

    private val _faceEyeIndependentEvidenceSupport =
        MutableStateFlow(0.0)
    val faceEyeIndependentEvidenceSupport: StateFlow<Double> =
        _faceEyeIndependentEvidenceSupport.asStateFlow()

    private val _faceEyeReliabilityTrend =
        MutableStateFlow(0.0)
    val faceEyeReliabilityTrend: StateFlow<Double> =
        _faceEyeReliabilityTrend.asStateFlow()

    private val _faceEyeFusionWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeFusionWarnings: StateFlow<List<String>> =
        _faceEyeFusionWarnings.asStateFlow()

    private val _faceEyeCaptureDomainConfidence =
        MutableStateFlow(0.0)
    val faceEyeCaptureDomainConfidence: StateFlow<Double> =
        _faceEyeCaptureDomainConfidence.asStateFlow()

    private val _faceEyeLightingConfidence =
        MutableStateFlow(0.0)
    val faceEyeLightingConfidence: StateFlow<Double> =
        _faceEyeLightingConfidence.asStateFlow()

    private val _faceEyeScaleConfidence =
        MutableStateFlow(0.0)
    val faceEyeScaleConfidence: StateFlow<Double> =
        _faceEyeScaleConfidence.asStateFlow()

    private val _faceEyeUncertaintyCalibrationConfidence =
        MutableStateFlow(0.0)
    val faceEyeUncertaintyCalibrationConfidence: StateFlow<Double> =
        _faceEyeUncertaintyCalibrationConfidence.asStateFlow()

    private val _faceEyeUncertaintyCalibrationReady =
        MutableStateFlow(false)
    val faceEyeUncertaintyCalibrationReady: StateFlow<Boolean> =
        _faceEyeUncertaintyCalibrationReady.asStateFlow()

    private val _faceUncertaintyCalibrationMultiplier =
        MutableStateFlow(1.0)
    val faceUncertaintyCalibrationMultiplier: StateFlow<Double> =
        _faceUncertaintyCalibrationMultiplier.asStateFlow()

    private val _eyeUncertaintyCalibrationMultiplier =
        MutableStateFlow(1.0)
    val eyeUncertaintyCalibrationMultiplier: StateFlow<Double> =
        _eyeUncertaintyCalibrationMultiplier.asStateFlow()

    private val _faceUncertaintyCoverage =
        MutableStateFlow(0.0)
    val faceUncertaintyCoverage: StateFlow<Double> =
        _faceUncertaintyCoverage.asStateFlow()

    private val _eyeUncertaintyCoverage =
        MutableStateFlow(0.0)
    val eyeUncertaintyCoverage: StateFlow<Double> =
        _eyeUncertaintyCoverage.asStateFlow()

    private val _faceEyeFailureAttribution =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeFailureAttribution: StateFlow<List<String>> =
        _faceEyeFailureAttribution.asStateFlow()
    private val _faceEyeCalibrationDriftScore =
        MutableStateFlow(0.0)
    val faceEyeCalibrationDriftScore: StateFlow<Double> =
        _faceEyeCalibrationDriftScore.asStateFlow()
    private val _faceEyeCalibrationDriftPersistent =
        MutableStateFlow(false)
    val faceEyeCalibrationDriftPersistent: StateFlow<Boolean> =
        _faceEyeCalibrationDriftPersistent.asStateFlow()


    private val _faceEyeRecalibrationRecommended =
        MutableStateFlow(false)
    val faceEyeRecalibrationRecommended: StateFlow<Boolean> =
        _faceEyeRecalibrationRecommended.asStateFlow()

    private val _faceEyeCalibrationDriftWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeCalibrationDriftWarnings: StateFlow<List<String>> =
        _faceEyeCalibrationDriftWarnings.asStateFlow()
    private val _faceEyeSaccadeVelocityDegreesPerSecond =
        MutableStateFlow(0.0)
    val faceEyeSaccadeVelocityDegreesPerSecond:
        StateFlow<Double> =
        _faceEyeSaccadeVelocityDegreesPerSecond.asStateFlow()
    private val _trackLifecycleState =
        MutableStateFlow(
            com.nivukx.nox.core.ai.ProductionTrackLifecycleController.State.NEW.name,
        )
    val trackLifecycleState: StateFlow<String> =
        _trackLifecycleState.asStateFlow()

    private val _trackLifecycleScore =
        MutableStateFlow(0.0)
    val trackLifecycleScore: StateFlow<Double> =
        _trackLifecycleScore.asStateFlow()

    private val _trackLifecyclePredictedCount =
        MutableStateFlow(0)
    val trackLifecyclePredictedCount: StateFlow<Int> =
        _trackLifecyclePredictedCount.asStateFlow()

    private val _trackLifecycleRecoveringCount =
        MutableStateFlow(0)
    val trackLifecycleRecoveringCount: StateFlow<Int> =
        _trackLifecycleRecoveringCount.asStateFlow()

    private val _trackLifecycleWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val trackLifecycleWarnings: StateFlow<List<String>> =
        _trackLifecycleWarnings.asStateFlow()

    private val _inferenceFps = MutableStateFlow(0)
    val inferenceFps: StateFlow<Int> = _inferenceFps.asStateFlow()

    private val _inferenceReason = MutableStateFlow("IDLE")
    val inferenceReason: StateFlow<String> = _inferenceReason.asStateFlow()

    private val _lastError = MutableStateFlow<String?>(null)
    val lastError: StateFlow<String?> = _lastError.asStateFlow()

    private val _postCaptureState = MutableStateFlow("IDLE")
    val postCaptureState: StateFlow<String> =
        _postCaptureState.asStateFlow()

    private val _postCaptureProgress = MutableStateFlow(0.0)
    val postCaptureProgress: StateFlow<Double> =
        _postCaptureProgress.asStateFlow()

    private val _postCaptureProcessedFrames = MutableStateFlow(0L)
    val postCaptureProcessedFrames: StateFlow<Long> =
        _postCaptureProcessedFrames.asStateFlow()

    private val _postCaptureMaxActors = MutableStateFlow(0)
    val postCaptureMaxActors: StateFlow<Int> =
        _postCaptureMaxActors.asStateFlow()

    private val _postCaptureVideoPath = MutableStateFlow<String?>(null)
    val postCaptureVideoPath: StateFlow<String?> =
        _postCaptureVideoPath.asStateFlow()

    private val _skewNanos = MutableStateFlow(0L)
    val skewNanos: StateFlow<Long> = _skewNanos.asStateFlow()

    private val _captureTimingConfidence = MutableStateFlow(0.0)
    val captureTimingConfidence: StateFlow<Double> =
        _captureTimingConfidence.asStateFlow()

    private val _captureTimingFps = MutableStateFlow(0.0)
    val captureTimingFps: StateFlow<Double> =
        _captureTimingFps.asStateFlow()

    private val _captureTimingJitterMillis = MutableStateFlow(0.0)
    val captureTimingJitterMillis: StateFlow<Double> =
        _captureTimingJitterMillis.asStateFlow()

    private val _captureTimingHealthy = MutableStateFlow(false)
    val captureTimingHealthy: StateFlow<Boolean> =
        _captureTimingHealthy.asStateFlow()

    private val _cameraSessionVerification =
        MutableStateFlow(
            CameraSessionNegotiationEngine.Verification.UNKNOWN.name,
        )
    val cameraSessionVerification: StateFlow<String> =
        _cameraSessionVerification.asStateFlow()

    private val _cameraSessionSupported =
        MutableStateFlow<Boolean?>(null)
    val cameraSessionSupported: StateFlow<Boolean?> =
        _cameraSessionSupported.asStateFlow()

    private val _highSpeedSessionVerification =
        MutableStateFlow(
            CameraSessionNegotiationEngine.Verification.UNKNOWN.name,
        )
    val highSpeedSessionVerification: StateFlow<String> =
        _highSpeedSessionVerification.asStateFlow()

    private val _highSpeedSessionSupported =
        MutableStateFlow<Boolean?>(null)
    val highSpeedSessionSupported: StateFlow<Boolean?> =
        _highSpeedSessionSupported.asStateFlow()

    private val _highSpeedSessionActive =
        MutableStateFlow(false)
    val highSpeedSessionActive: StateFlow<Boolean> =
        _highSpeedSessionActive.asStateFlow()

    private val _highSpeedSelectedFps =
        MutableStateFlow<Int?>(null)
    val highSpeedSelectedFps: StateFlow<Int?> =
        _highSpeedSelectedFps.asStateFlow()

    private val _highSpeedSessionReason =
        MutableStateFlow<String?>(null)
    val highSpeedSessionReason: StateFlow<String?> =
        _highSpeedSessionReason.asStateFlow()

    private val _hardwareReadoutTimestampSupported =
        MutableStateFlow(false)
    val hardwareReadoutTimestampSupported: StateFlow<Boolean> =
        _hardwareReadoutTimestampSupported.asStateFlow()

    private val _latestReadoutTimestampNanos =
        MutableStateFlow<Long?>(null)
    val latestReadoutTimestampNanos: StateFlow<Long?> =
        _latestReadoutTimestampNanos.asStateFlow()

    private val _hardwareReadoutTimingState =
        MutableStateFlow(
            HardwareReadoutTimingEvidenceEngine.State.UNKNOWN.name,
        )
    val hardwareReadoutTimingState: StateFlow<String> =
        _hardwareReadoutTimingState.asStateFlow()

    private val _hardwareReadoutTimingConfidence =
        MutableStateFlow(0.0)
    val hardwareReadoutTimingConfidence: StateFlow<Double> =
        _hardwareReadoutTimingConfidence.asStateFlow()

    private val _hardwareReadoutP95JitterMillis =
        MutableStateFlow(Double.POSITIVE_INFINITY)
    val hardwareReadoutP95JitterMillis: StateFlow<Double> =
        _hardwareReadoutP95JitterMillis.asStateFlow()

    private val _hardwareReadoutMedianDelayMillis =
        MutableStateFlow(Double.NaN)
    val hardwareReadoutMedianDelayMillis: StateFlow<Double> =
        _hardwareReadoutMedianDelayMillis.asStateFlow()

    private val _thermalExtrinsicDriftState =
        MutableStateFlow(
            ThermalExtrinsicDriftEngine.State.UNKNOWN.name,
        )
    val thermalExtrinsicDriftState: StateFlow<String> =
        _thermalExtrinsicDriftState.asStateFlow()

    private val _thermalExtrinsicDriftConfidence =
        MutableStateFlow(0.0)
    val thermalExtrinsicDriftConfidence: StateFlow<Double> =
        _thermalExtrinsicDriftConfidence.asStateFlow()

    private val _thermalExtrinsicTranslationDriftMm =
        MutableStateFlow(0.0)
    val thermalExtrinsicTranslationDriftMm: StateFlow<Double> =
        _thermalExtrinsicTranslationDriftMm.asStateFlow()

    private val _thermalExtrinsicRotationDriftDegrees =
        MutableStateFlow(0.0)
    val thermalExtrinsicRotationDriftDegrees: StateFlow<Double> =
        _thermalExtrinsicRotationDriftDegrees.asStateFlow()

    private val _thermalExtrinsicTemperatureDeltaCelsius =
        MutableStateFlow(0.0)
    val thermalExtrinsicTemperatureDeltaCelsius: StateFlow<Double> =
        _thermalExtrinsicTemperatureDeltaCelsius.asStateFlow()

    private val _thermalExtrinsicHardFailure =
        MutableStateFlow(false)
    val thermalExtrinsicHardFailure: StateFlow<Boolean> =
        _thermalExtrinsicHardFailure.asStateFlow()

    private val _thermalExtrinsicWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val thermalExtrinsicWarnings: StateFlow<List<String>> =
        _thermalExtrinsicWarnings.asStateFlow()

    private val _concurrentRigAdmissionState =
        MutableStateFlow("NOT_APPLICABLE")
    val concurrentRigAdmissionState: StateFlow<String> =
        _concurrentRigAdmissionState.asStateFlow()

    private val _concurrentRigAdmissionScore =
        MutableStateFlow(1.0)
    val concurrentRigAdmissionScore: StateFlow<Double> =
        _concurrentRigAdmissionScore.asStateFlow()

    private val _concurrentRigTopologyScore =
        MutableStateFlow(0.0)
    val concurrentRigTopologyScore: StateFlow<Double> =
        _concurrentRigTopologyScore.asStateFlow()

    private val _concurrentRigTopologyMode =
        MutableStateFlow(
            CameraTopologyEvidenceEngine.Mode.UNKNOWN.name,
        )
    val concurrentRigTopologyMode: StateFlow<String> =
        _concurrentRigTopologyMode.asStateFlow()

    private val _concurrentRigSelectedCameraIds =
        MutableStateFlow<List<String>>(emptyList())
    val concurrentRigSelectedCameraIds: StateFlow<List<String>> =
        _concurrentRigSelectedCameraIds.asStateFlow()

    private val _concurrentRigBlockers =
        MutableStateFlow<List<String>>(emptyList())
    val concurrentRigBlockers: StateFlow<List<String>> =
        _concurrentRigBlockers.asStateFlow()

    private val _concurrentRigWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val concurrentRigWarnings: StateFlow<List<String>> =
        _concurrentRigWarnings.asStateFlow()

    private val _cameraCaptureQualityScore = MutableStateFlow(0.0)
    val cameraCaptureQualityScore: StateFlow<Double> =
        _cameraCaptureQualityScore.asStateFlow()

    private val _cameraOpticalBlurRisk =
        MutableStateFlow(0.0)
    val cameraOpticalBlurRisk: StateFlow<Double> =
        _cameraOpticalBlurRisk.asStateFlow()

    private val _cameraCaptureQualityWarnings =
        MutableStateFlow<List<String>>(emptyList())

    private val _cameraAnalysisDropRatio =
        MutableStateFlow(0.0)
    val cameraAnalysisDropRatio: StateFlow<Double> =
        _cameraAnalysisDropRatio.asStateFlow()
    val cameraCaptureQualityWarnings: StateFlow<List<String>> =
        _cameraCaptureQualityWarnings.asStateFlow()

    private val _sessionHealthState =
        MutableStateFlow(SessionHealthEngine.State.NO_DATA.name)
    val sessionHealthState: StateFlow<String> =
        _sessionHealthState.asStateFlow()

    private val _sessionHealthScore = MutableStateFlow(0.0)
    val sessionHealthScore: StateFlow<Double> =
        _sessionHealthScore.asStateFlow()

    private val _sessionHealthBottleneck = MutableStateFlow("NONE")
    val sessionHealthBottleneck: StateFlow<String> =
        _sessionHealthBottleneck.asStateFlow()

    private val _captureIntegrityScore = MutableStateFlow(0.0)
    val captureIntegrityScore: StateFlow<Double> =
        _captureIntegrityScore.asStateFlow()

    private val _captureIntegrityBand = MutableStateFlow(
        CaptureIntegrityEngine.Band.BLOCKED.name,
    )
    val captureIntegrityBand: StateFlow<String> =
        _captureIntegrityBand.asStateFlow()

    private val _captureIntegrityOutputAllowed = MutableStateFlow(false)
    val captureIntegrityOutputAllowed: StateFlow<Boolean> =
        _captureIntegrityOutputAllowed.asStateFlow()

    private val _captureEvidenceTrend =
        MutableStateFlow("UNKNOWN")
    val captureEvidenceTrend: StateFlow<String> =
        _captureEvidenceTrend.asStateFlow()

    private val _captureEvidenceSeverity =
        MutableStateFlow("CRITICAL")
    val captureEvidenceSeverity: StateFlow<String> =
        _captureEvidenceSeverity.asStateFlow()

    private val _captureEvidenceCurrentScore =
        MutableStateFlow(0.0)
    val captureEvidenceCurrentScore: StateFlow<Double> =
        _captureEvidenceCurrentScore.asStateFlow()

    private val _captureEvidenceRecentScore =
        MutableStateFlow(0.0)
    val captureEvidenceRecentScore: StateFlow<Double> =
        _captureEvidenceRecentScore.asStateFlow()

    private val _captureEvidenceSlope =
        MutableStateFlow(0.0)
    val captureEvidenceSlope: StateFlow<Double> =
        _captureEvidenceSlope.asStateFlow()

    private val _captureEvidenceBottleneck =
        MutableStateFlow("NONE")
    val captureEvidenceBottleneck: StateFlow<String> =
        _captureEvidenceBottleneck.asStateFlow()

    private val _captureEvidenceWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val captureEvidenceWarnings: StateFlow<List<String>> =
        _captureEvidenceWarnings.asStateFlow()

    private val _captureIntegrityBottleneck = MutableStateFlow("NONE")
    val captureIntegrityBottleneck: StateFlow<String> =
        _captureIntegrityBottleneck.asStateFlow()

    private val _captureIntegrityWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val captureIntegrityWarnings: StateFlow<List<String>> =
        _captureIntegrityWarnings.asStateFlow()

    private val _captureIntegrityHardFailures =
        MutableStateFlow<List<String>>(emptyList())
    val captureIntegrityHardFailures: StateFlow<List<String>> =
        _captureIntegrityHardFailures.asStateFlow()

    private val _productionProfile =
        MutableStateFlow(ProductionCaptureProfile.default)
    val productionProfile: StateFlow<ProductionCaptureProfile> =
        _productionProfile.asStateFlow()

    private val _productionQualityScore = MutableStateFlow(0.0)
    val productionQualityScore: StateFlow<Double> =
        _productionQualityScore.asStateFlow()

    private val _captureAuthorityScore =
        MutableStateFlow(0.0)
    val captureAuthorityScore: StateFlow<Double> =
        _captureAuthorityScore.asStateFlow()

    private val _captureAuthorityBand =
        MutableStateFlow("BLOCKED")
    val captureAuthorityBand: StateFlow<String> =
        _captureAuthorityBand.asStateFlow()

    private val _captureAuthorityHardBlocked =
        MutableStateFlow(true)
    val captureAuthorityHardBlocked: StateFlow<Boolean> =
        _captureAuthorityHardBlocked.asStateFlow()

    private val _captureAuthorityPrecisionQualified =
        MutableStateFlow(false)
    val captureAuthorityPrecisionQualified: StateFlow<Boolean> =
        _captureAuthorityPrecisionQualified.asStateFlow()

    private val _captureAuthorityWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val captureAuthorityWarnings: StateFlow<List<String>> =
        _captureAuthorityWarnings.asStateFlow()

    private val _productionPolicy =
        MutableStateFlow(HighEndReconstructionQualityEngine.Policy.HOLD.name)
    val productionPolicy: StateFlow<String> =
        _productionPolicy.asStateFlow()

    private val _productionBottleneck =
        MutableStateFlow(HighEndReconstructionQualityEngine.Bottleneck.NONE.name)
    val productionBottleneck: StateFlow<String> =
        _productionBottleneck.asStateFlow()

    private val _productionBudgetLevel =
        MutableStateFlow("STARTUP")
    val productionBudgetLevel: StateFlow<String> =
        _productionBudgetLevel.asStateFlow()

    private val _productionOutputAllowed =
        MutableStateFlow(false)
    val productionOutputAllowed: StateFlow<Boolean> =
        _productionOutputAllowed.asStateFlow()

    private val _physicalPrecisionCapability =
        MutableStateFlow(PhysicalPrecisionCapabilityEngine.Decision.idle())
    val physicalPrecisionCapability:
        StateFlow<PhysicalPrecisionCapabilityEngine.Decision> =
        _physicalPrecisionCapability.asStateFlow()

    private val _productionWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val productionWarnings: StateFlow<List<String>> =
        _productionWarnings.asStateFlow()

    private val _productionDetectorInterval = MutableStateFlow(3)
    val productionDetectorInterval: StateFlow<Int> =
        _productionDetectorInterval.asStateFlow()

    private val _productionMeshVertexBudget = MutableStateFlow(0)
    val productionMeshVertexBudget: StateFlow<Int> =
        _productionMeshVertexBudget.asStateFlow()

    private val _productionTsdfStride = MutableStateFlow(1)
    val productionTsdfStride: StateFlow<Int> =
        _productionTsdfStride.asStateFlow()

    private val _productionBundleAdjustmentIterations = MutableStateFlow(0)
    val productionBundleAdjustmentIterations: StateFlow<Int> =
        _productionBundleAdjustmentIterations.asStateFlow()

    private val _sessionHealthWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val sessionHealthWarnings: StateFlow<List<String>> =
        _sessionHealthWarnings.asStateFlow()

    private var semanticEventRootPosition:
        com.nivukx.nox.core.model.Vec3? = null
    private var semanticEventRootVelocity =
        com.nivukx.nox.core.model.Vec3(0.0, 0.0, 0.0)
    private var semanticEventTimestampNanos = Long.MIN_VALUE

    private val _motionDynamicsPhase =
        MutableStateFlow(ProductionMotionDynamicsEngine.MotionPhase.UNKNOWN.name)
    val motionDynamicsPhase: StateFlow<String> =
        _motionDynamicsPhase.asStateFlow()

    private val _motionDynamicsScore =
        MutableStateFlow(0.0)
    val motionDynamicsScore: StateFlow<Double> =
        _motionDynamicsScore.asStateFlow()

    private val _motionContactStability =
        MutableStateFlow(0.0)
    val motionContactStability: StateFlow<Double> =
        _motionContactStability.asStateFlow()

    private val _motionSymmetryScore =
        MutableStateFlow(0.0)
    val motionSymmetryScore: StateFlow<Double> =
        _motionSymmetryScore.asStateFlow()

    private val _motionDynamicStress =
        MutableStateFlow(0.0)
    val motionDynamicStress: StateFlow<Double> =
        _motionDynamicStress.asStateFlow()

    private val _motionDynamicsUncertainty =
        MutableStateFlow(0.0)
    val motionDynamicsUncertainty: StateFlow<Double> =
        _motionDynamicsUncertainty.asStateFlow()

    private val _motionTrustedJointRatio =
        MutableStateFlow(0.0)
    val motionTrustedJointRatio: StateFlow<Double> =
        _motionTrustedJointRatio.asStateFlow()

    private val _motionMicroMotionRetention =
        MutableStateFlow(0.0)
    val motionMicroMotionRetention: StateFlow<Double> =
        _motionMicroMotionRetention.asStateFlow()
    private val _motionLowFrequencyScore =
        MutableStateFlow(0.0)
    val motionLowFrequencyScore: StateFlow<Double> =
        _motionLowFrequencyScore.asStateFlow()

    private val _motionHighFrequencyScore =
        MutableStateFlow(0.0)
    val motionHighFrequencyScore: StateFlow<Double> =
        _motionHighFrequencyScore.asStateFlow()

    private val _motionTemporalMemoryScore =
        MutableStateFlow(0.0)
    val motionTemporalMemoryScore: StateFlow<Double> =
        _motionTemporalMemoryScore.asStateFlow()
    private val _motionUncertaintyCoverage =
        MutableStateFlow(0.0)
    val motionUncertaintyCoverage: StateFlow<Double> =
        _motionUncertaintyCoverage.asStateFlow()

    private val _motionUncertaintyConfidence =
        MutableStateFlow(0.0)
    val motionUncertaintyConfidence: StateFlow<Double> =
        _motionUncertaintyConfidence.asStateFlow()

    private val _motionUncertaintyOodRisk =
        MutableStateFlow(0.0)
    val motionUncertaintyOodRisk: StateFlow<Double> =
        _motionUncertaintyOodRisk.asStateFlow()

    private val _motionUncertaintyRadiusMeters =
        MutableStateFlow(0.0)
    val motionUncertaintyRadiusMeters: StateFlow<Double> =
        _motionUncertaintyRadiusMeters.asStateFlow()

    private val _motionUncertaintyCalibrationProgress =
        MutableStateFlow(0.0)
    val motionUncertaintyCalibrationProgress: StateFlow<Double> =
        _motionUncertaintyCalibrationProgress.asStateFlow()

    private val _motionUncertaintyPolicy =
        MutableStateFlow(
            CalibratedMotionUncertaintyEngine.Result.Policy.CAUTION.name,
        )
    val motionUncertaintyPolicy: StateFlow<String> =
        _motionUncertaintyPolicy.asStateFlow()

    private val _motionUncertaintyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val motionUncertaintyWarnings: StateFlow<List<String>> =
        _motionUncertaintyWarnings.asStateFlow()

    private val _motionEventType = MutableStateFlow("UNKNOWN")
    val motionEventType: StateFlow<String> = _motionEventType.asStateFlow()

    private val _motionEventConfidence = MutableStateFlow(0.0)
    val motionEventConfidence: StateFlow<Double> = _motionEventConfidence.asStateFlow()

    private val _motionScore = MutableStateFlow(0.0)
    val motionScore: StateFlow<Double> = _motionScore.asStateFlow()

    private val _commonModeMotionNormalizedPerSecond =
        MutableStateFlow(0.0)
    val commonModeMotionNormalizedPerSecond: StateFlow<Double> =
        _commonModeMotionNormalizedPerSecond.asStateFlow()

    private val _differentialMotionScore =
        MutableStateFlow(0.0)
    val differentialMotionScore: StateFlow<Double> =
        _differentialMotionScore.asStateFlow()

    private val _cameraCompensatedMotionScore =
        MutableStateFlow(0.0)
    val cameraCompensatedMotionScore: StateFlow<Double> =
        _cameraCompensatedMotionScore.asStateFlow()

    private val _microMotionScore = MutableStateFlow(0.0)
    val microMotionScore: StateFlow<Double> =
        _microMotionScore.asStateFlow()

    private val _directionChangeRatio = MutableStateFlow(0.0)
    val directionChangeRatio: StateFlow<Double> =
        _directionChangeRatio.asStateFlow()

    private val _motionMeasurementUncertainty =
        MutableStateFlow(0.0)
    val motionMeasurementUncertainty: StateFlow<Double> =
        _motionMeasurementUncertainty.asStateFlow()

    private val _motionEvidenceScore =
        MutableStateFlow(0.0)
    val motionEvidenceScore: StateFlow<Double> =
        _motionEvidenceScore.asStateFlow()

    private val _motionEvidenceConfidence =
        MutableStateFlow(0.0)
    val motionEvidenceConfidence: StateFlow<Double> =
        _motionEvidenceConfidence.asStateFlow()

    private val _motionEvidenceWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val motionEvidenceWarnings: StateFlow<List<String>> =
        _motionEvidenceWarnings.asStateFlow()

    private val _temporalStability = MutableStateFlow(0.0)
    val temporalStability: StateFlow<Double> = _temporalStability.asStateFlow()

    private val _reconstructionJoints = MutableStateFlow(0)
    val reconstructionJoints: StateFlow<Int> = _reconstructionJoints.asStateFlow()

    private val _semanticCoverage = MutableStateFlow(0.0)
    val semanticCoverage: StateFlow<Double> = _semanticCoverage.asStateFlow()

    private val _attentionConfidence = MutableStateFlow(0.0)
    val attentionConfidence: StateFlow<Double> = _attentionConfidence.asStateFlow()

    private val _occlusionRisk = MutableStateFlow(0.0)
    val occlusionRisk: StateFlow<Double> = _occlusionRisk.asStateFlow()

    private val _dominantSubject = MutableStateFlow("NONE")
    val dominantSubject: StateFlow<String> = _dominantSubject.asStateFlow()

    private val _motionClass = MutableStateFlow("STATIC")
    val motionClass: StateFlow<String> = _motionClass.asStateFlow()

    private val _semanticMotionScore = MutableStateFlow(0.0)
    val semanticMotionScore: StateFlow<Double> =
        _semanticMotionScore.asStateFlow()

    private val _cameraMotionRisk = MutableStateFlow(0.0)
    val cameraMotionRisk: StateFlow<Double> =
        _cameraMotionRisk.asStateFlow()

    private val _runtimeBand = MutableStateFlow("HIGH")
    val runtimeBand: StateFlow<String> = _runtimeBand.asStateFlow()

    private val _runtimeQualityScale = MutableStateFlow(0.88)
    val runtimeQualityScale: StateFlow<Double> =
        _runtimeQualityScale.asStateFlow()

    private val _runtimeInferenceScale = MutableStateFlow(0.88)
    val runtimeInferenceScale: StateFlow<Double> =
        _runtimeInferenceScale.asStateFlow()

    private val _runtimeProtection = MutableStateFlow(false)
    val runtimeProtection: StateFlow<Boolean> =
        _runtimeProtection.asStateFlow()

    private val _runtimeReasons =
        MutableStateFlow<List<String>>(emptyList())
    val runtimeReasons: StateFlow<List<String>> =
        _runtimeReasons.asStateFlow()

    private val _gpuDispatchMillis = MutableStateFlow(0.0)
    val gpuDispatchMillis: StateFlow<Double> = _gpuDispatchMillis.asStateFlow()

    private val _liveMotionPreviewFrame =
        MutableStateFlow<MotionFieldPreviewFrame?>(null)
    val liveMotionPreviewFrame: StateFlow<MotionFieldPreviewFrame?> =
        _liveMotionPreviewFrame.asStateFlow()

    private val _liveMotionPreviewGpuState =
        MutableStateFlow(MotionFieldGpuState())
    val liveMotionPreviewGpuState: StateFlow<MotionFieldGpuState> =
        _liveMotionPreviewGpuState.asStateFlow()


    private val _reconstructionErrorPx = MutableStateFlow(Double.POSITIVE_INFINITY)
    val reconstructionErrorPx: StateFlow<Double> = _reconstructionErrorPx.asStateFlow()

    private val _reconstructionQualityScore = MutableStateFlow(0.0)
    val reconstructionQualityScore: StateFlow<Double> =
        _reconstructionQualityScore.asStateFlow()

    private val _bodyMetricEvidenceAuthority =
        MutableStateFlow(0.0)
    private val _bodySymmetricEpipolarConsistency =
        MutableStateFlow(1.0)

    private val _worldConsistencyScore = MutableStateFlow(0.0)
    val worldConsistencyScore: StateFlow<Double> =
        _worldConsistencyScore.asStateFlow()

    private val _worldConsistencyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val worldConsistencyWarnings: StateFlow<List<String>> =
        _worldConsistencyWarnings.asStateFlow()

    private val _reconstructionGateReasons =
        MutableStateFlow<List<String>>(emptyList())
    val reconstructionGateReasons: StateFlow<List<String>> =
        _reconstructionGateReasons.asStateFlow()

    private val _denseMeshVertices = MutableStateFlow(0)

    private val _denseMeshTriangles = MutableStateFlow(0)

    private val _denseMeshConfidence = MutableStateFlow(0.0)
    val denseMeshVertices: StateFlow<Int> = _denseMeshVertices.asStateFlow()
    val denseMeshTriangles: StateFlow<Int> = _denseMeshTriangles.asStateFlow()
    val denseMeshConfidence: StateFlow<Double> = _denseMeshConfidence.asStateFlow()


    private val _denseMeshNormalCoverage =
        MutableStateFlow(0.0)
    val denseMeshNormalCoverage: StateFlow<Double> =
        _denseMeshNormalCoverage.asStateFlow()

    private val _denseMeshSurfaceConfidence =
        MutableStateFlow(0.0)
    val denseMeshSurfaceConfidence: StateFlow<Double> =
        _denseMeshSurfaceConfidence.asStateFlow()

    private val _denseMeshSurfaceObserved =
        MutableStateFlow(false)
    private val _denseMeshTopologyIntegrity =
        MutableStateFlow(0.0)
    val denseMeshTopologyIntegrity: StateFlow<Double> =
        _denseMeshTopologyIntegrity.asStateFlow()

    private val _denseMeshBoundaryEdges =
        MutableStateFlow(0)
    val denseMeshBoundaryEdges: StateFlow<Int> =
        _denseMeshBoundaryEdges.asStateFlow()

    private val _denseMeshNonManifoldEdges =
        MutableStateFlow(0)
    val denseMeshNonManifoldEdges: StateFlow<Int> =
        _denseMeshNonManifoldEdges.asStateFlow()

    private val _denseMeshDuplicateTrianglesRemoved =
        MutableStateFlow(0)
    val denseMeshDuplicateTrianglesRemoved: StateFlow<Int> =
        _denseMeshDuplicateTrianglesRemoved.asStateFlow()

    private val _denseTsdfDetailLevel =
        MutableStateFlow(1.0)
    val denseTsdfDetailLevel: StateFlow<Double> =
        _denseTsdfDetailLevel.asStateFlow()

    private val _denseTsdfComputePressure =
        MutableStateFlow(0.0)
    val denseTsdfComputePressure: StateFlow<Double> =
        _denseTsdfComputePressure.asStateFlow()

    private val _denseTsdfSurfaceNormalObservability =
        MutableStateFlow(1.0)
    val denseTsdfSurfaceNormalObservability: StateFlow<Double> =
        _denseTsdfSurfaceNormalObservability.asStateFlow()

    private val _denseTsdfMeanEvidenceWeight =
        MutableStateFlow(0.0)
    val denseTsdfMeanEvidenceWeight: StateFlow<Double> =
        _denseTsdfMeanEvidenceWeight.asStateFlow()

    private val _denseTsdfMeanUncertainty =
        MutableStateFlow(0.0)
    val denseTsdfMeanUncertainty: StateFlow<Double> =
        _denseTsdfMeanUncertainty.asStateFlow()

    private val _denseTsdfHighUncertaintyRatio =
        MutableStateFlow(0.0)
    val denseTsdfHighUncertaintyRatio: StateFlow<Double> =
        _denseTsdfHighUncertaintyRatio.asStateFlow()

    private val _denseTsdfConfidenceFactor =
        MutableStateFlow(1.0)
    val denseTsdfConfidenceFactor: StateFlow<Double> =
        _denseTsdfConfidenceFactor.asStateFlow()

    private val _denseTsdfDecisionReason =
        MutableStateFlow("balanced")
    val denseTsdfDecisionReason: StateFlow<String> =
        _denseTsdfDecisionReason.asStateFlow()

    private val _denseSurfaceMeanDynamicProbability =
        MutableStateFlow(0.0)
    val denseSurfaceMeanDynamicProbability: StateFlow<Double> =
        _denseSurfaceMeanDynamicProbability.asStateFlow()

    private val _denseSurfaceMeanTemporalStability =
        MutableStateFlow(1.0)
    val denseSurfaceMeanTemporalStability: StateFlow<Double> =
        _denseSurfaceMeanTemporalStability.asStateFlow()

    private val _denseSurfaceTrustedGaussianRatio =
        MutableStateFlow(0.0)
    val denseSurfaceTrustedGaussianRatio: StateFlow<Double> =
        _denseSurfaceTrustedGaussianRatio.asStateFlow()

    private val _temporalDepthConsistency =
        MutableStateFlow(0.0)
    val temporalDepthConsistency: StateFlow<Double> =
        _temporalDepthConsistency.asStateFlow()

    private val _denseMeshExtractionDetail =
        MutableStateFlow(0.0)
    val denseMeshExtractionDetail: StateFlow<Double> =
        _denseMeshExtractionDetail.asStateFlow()

    private val _denseMeshExtractionPressure =
        MutableStateFlow(0.0)
    val denseMeshExtractionPressure: StateFlow<Double> =
        _denseMeshExtractionPressure.asStateFlow()

    private val _denseMeshExtractionReason =
        MutableStateFlow("not_ready")
    val denseMeshExtractionReason: StateFlow<String> =
        _denseMeshExtractionReason.asStateFlow()

    private val _kinematicConstraintScore =
        MutableStateFlow(1.0)
    val kinematicConstraintScore: StateFlow<Double> =
        _kinematicConstraintScore.asStateFlow()

    private val _kinematicConstraintAccepted =
        MutableStateFlow(true)
    val kinematicConstraintAccepted: StateFlow<Boolean> =
        _kinematicConstraintAccepted.asStateFlow()

    private val _kinematicConstraintBottleneck =
        MutableStateFlow("NONE")
    val kinematicConstraintBottleneck: StateFlow<String> =
        _kinematicConstraintBottleneck.asStateFlow()

    private val _kinematicConstraintWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val kinematicConstraintWarnings: StateFlow<List<String>> =
        _kinematicConstraintWarnings.asStateFlow()

    private val _physicsConstraintScore =
        MutableStateFlow(1.0)
    val physicsConstraintScore: StateFlow<Double> =
        _physicsConstraintScore.asStateFlow()

    private val _physicsConstraintAccepted =
        MutableStateFlow(true)
    val physicsConstraintAccepted: StateFlow<Boolean> =
        _physicsConstraintAccepted.asStateFlow()

    private val _physicsConstraintMeanBoneResidualMeters =
        MutableStateFlow(0.0)
    val physicsConstraintMeanBoneResidualMeters: StateFlow<Double> =
        _physicsConstraintMeanBoneResidualMeters.asStateFlow()

    private val _physicsConstraintP95BoneResidualMeters =
        MutableStateFlow(0.0)
    val physicsConstraintP95BoneResidualMeters: StateFlow<Double> =
        _physicsConstraintP95BoneResidualMeters.asStateFlow()

    private val _physicsConstraintMaxBoneResidualMeters =
        MutableStateFlow(0.0)
    val physicsConstraintMaxBoneResidualMeters: StateFlow<Double> =
        _physicsConstraintMaxBoneResidualMeters.asStateFlow()

    private val _physicsConstraintSymmetryScore =
        MutableStateFlow(1.0)
    val physicsConstraintSymmetryScore: StateFlow<Double> =
        _physicsConstraintSymmetryScore.asStateFlow()

    private val _physicsConstraintFloorContactScore =
        MutableStateFlow(1.0)
    val physicsConstraintFloorContactScore: StateFlow<Double> =
        _physicsConstraintFloorContactScore.asStateFlow()

    private val _physicsConstraintCorrectedJointCount =
        MutableStateFlow(0)
    val physicsConstraintCorrectedJointCount: StateFlow<Int> =
        _physicsConstraintCorrectedJointCount.asStateFlow()

    private val _physicsConstraintConstrainedEdgeCount =
        MutableStateFlow(0)
    val physicsConstraintConstrainedEdgeCount: StateFlow<Int> =
        _physicsConstraintConstrainedEdgeCount.asStateFlow()

    private val _physicsConstraintWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val physicsConstraintWarnings: StateFlow<List<String>> =
        _physicsConstraintWarnings.asStateFlow()


    private val _global4DScore =
        MutableStateFlow(1.0)
    val global4DScore: StateFlow<Double> =
        _global4DScore.asStateFlow()

    private val _global4DAccepted =
        MutableStateFlow(true)
    val global4DAccepted: StateFlow<Boolean> =
        _global4DAccepted.asStateFlow()

    private val _global4DState =
        MutableStateFlow(
            com.nivukx.nox.core.reconstruction.Global4DConsistencyEngine.State.WARMING.name,
        )
    val global4DState: StateFlow<String> =
        _global4DState.asStateFlow()

    private val _global4DPersistentFailure =
        MutableStateFlow(false)
    val global4DPersistentFailure: StateFlow<Boolean> =
        _global4DPersistentFailure.asStateFlow()

    private val _global4DRootMotionConsistency =
        MutableStateFlow(1.0)
    val global4DRootMotionConsistency: StateFlow<Double> =
        _global4DRootMotionConsistency.asStateFlow()

    private val _global4DRootVelocityAgreement =
        MutableStateFlow(1.0)
    val global4DRootVelocityAgreement: StateFlow<Double> =
        _global4DRootVelocityAgreement.asStateFlow()

    private val _global4DSceneConsistency =
        MutableStateFlow(0.72)
    val global4DSceneConsistency: StateFlow<Double> =
        _global4DSceneConsistency.asStateFlow()

    private val _global4DDirectionalUncertaintyConsistency =
        MutableStateFlow(1.0)
    val global4DDirectionalUncertaintyConsistency: StateFlow<Double> =
        _global4DDirectionalUncertaintyConsistency.asStateFlow()

    private val _global4DFaceEyeConsistency =
        MutableStateFlow(1.0)
    val global4DFaceEyeConsistency: StateFlow<Double> =
        _global4DFaceEyeConsistency.asStateFlow()

    private val _global4DExternalContactConsistency =
        MutableStateFlow(1.0)
    val global4DExternalContactConsistency: StateFlow<Double> =
        _global4DExternalContactConsistency.asStateFlow()

    private val _global4DGaussianConsistency =
        MutableStateFlow(1.0)
    val global4DGaussianConsistency: StateFlow<Double> =
        _global4DGaussianConsistency.asStateFlow()

    private val _global4DGaussianStrainConsistency =
        MutableStateFlow(1.0)
    val global4DGaussianStrainConsistency: StateFlow<Double> =
        _global4DGaussianStrainConsistency.asStateFlow()

    private val _global4DGaussianEvidenceAvailable =
        MutableStateFlow(false)
    val global4DGaussianEvidenceAvailable: StateFlow<Boolean> =
        _global4DGaussianEvidenceAvailable.asStateFlow()

    private val _global4DWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val global4DWarnings: StateFlow<List<String>> =
        _global4DWarnings.asStateFlow()

    private val _worldSpaceConsistencyScore =
        MutableStateFlow(1.0)
    val worldSpaceConsistencyScore: StateFlow<Double> =
        _worldSpaceConsistencyScore.asStateFlow()

    private val _worldSpaceAccepted =
        MutableStateFlow(true)
    val worldSpaceAccepted: StateFlow<Boolean> =
        _worldSpaceAccepted.asStateFlow()

    private val _worldSpaceTemporalStability =
        MutableStateFlow(1.0)
    val worldSpaceTemporalStability: StateFlow<Double> =
        _worldSpaceTemporalStability.asStateFlow()

    private val _worldSpaceMotionContinuity =
        MutableStateFlow(1.0)
    val worldSpaceMotionContinuity: StateFlow<Double> =
        _worldSpaceMotionContinuity.asStateFlow()

    private val _worldSpaceRejectedJointCount =
        MutableStateFlow(0)
    val worldSpaceRejectedJointCount: StateFlow<Int> =
        _worldSpaceRejectedJointCount.asStateFlow()

    private val _worldSpaceRobustOutlierRatio =
        MutableStateFlow(0.0)
    val worldSpaceRobustOutlierRatio: StateFlow<Double> =
        _worldSpaceRobustOutlierRatio.asStateFlow()

    private val _worldSpaceWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val worldSpaceWarnings: StateFlow<List<String>> =
        _worldSpaceWarnings.asStateFlow()

    private val _denseMeshTemporalConsistencyConfidence =
        MutableStateFlow(0.0)
    val denseMeshTemporalConsistencyConfidence: StateFlow<Double> =
        _denseMeshTemporalConsistencyConfidence.asStateFlow()

    private val _denseMeshTemporalConsistencyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val denseMeshTemporalConsistencyWarnings: StateFlow<List<String>> =
        _denseMeshTemporalConsistencyWarnings.asStateFlow()

    private val _face3DPoints = MutableStateFlow(0)
    val face3DPoints: StateFlow<Int> = _face3DPoints.asStateFlow()

    private val _geometricObservabilityScore =
        MutableStateFlow(0.0)
    val geometricObservabilityScore: StateFlow<Double> =
        _geometricObservabilityScore.asStateFlow()

    private val _geometricObservabilityWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val geometricObservabilityWarnings: StateFlow<List<String>> =
        _geometricObservabilityWarnings.asStateFlow()

    private val _geometricInformationConditionScore =
        MutableStateFlow(0.0)
    val geometricInformationConditionScore: StateFlow<Double> =
        _geometricInformationConditionScore.asStateFlow()

    private val _geometricInformationP95MajorMeters =
        MutableStateFlow(0.0)
    val geometricInformationP95MajorMeters: StateFlow<Double> =
        _geometricInformationP95MajorMeters.asStateFlow()

    private val _geometricInformationP95DepthMeters =
        MutableStateFlow(0.0)
    val geometricInformationP95DepthMeters: StateFlow<Double> =
        _geometricInformationP95DepthMeters.asStateFlow()

    private val _reconstructionUncertaintyConfidence =
        MutableStateFlow(1.0)
    val reconstructionUncertaintyConfidence: StateFlow<Double> =
        _reconstructionUncertaintyConfidence.asStateFlow()

    private val _reconstructionUncertaintyMeanMeters =
        MutableStateFlow(0.0)
    val reconstructionUncertaintyMeanMeters: StateFlow<Double> =
        _reconstructionUncertaintyMeanMeters.asStateFlow()

    private val _reconstructionUncertaintyP95Meters =
        MutableStateFlow(0.0)
    val reconstructionUncertaintyP95Meters: StateFlow<Double> =
        _reconstructionUncertaintyP95Meters.asStateFlow()

    private val _reconstructionTrustedJointRatio =
        MutableStateFlow(1.0)
    val reconstructionTrustedJointRatio: StateFlow<Double> =
        _reconstructionTrustedJointRatio.asStateFlow()

    private val _reconstructionUncertaintyBottleneck =
        MutableStateFlow("NONE")
    val reconstructionUncertaintyBottleneck: StateFlow<String> =
        _reconstructionUncertaintyBottleneck.asStateFlow()

    private val _reconstructionUncertaintyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val reconstructionUncertaintyWarnings: StateFlow<List<String>> =
        _reconstructionUncertaintyWarnings.asStateFlow()

    private val _directionalUncertaintyMajor95Meters =
        MutableStateFlow(0.0)
    val directionalUncertaintyMajor95Meters: StateFlow<Double> =
        _directionalUncertaintyMajor95Meters.asStateFlow()

    private val _directionalUncertaintyDepthMeters =
        MutableStateFlow(0.0)
    val directionalUncertaintyDepthMeters: StateFlow<Double> =
        _directionalUncertaintyDepthMeters.asStateFlow()

    private val _directionalUncertaintyAnisotropy =
        MutableStateFlow(1.0)
    val directionalUncertaintyAnisotropy: StateFlow<Double> =
        _directionalUncertaintyAnisotropy.asStateFlow()

    private val _directionalUncertaintyConfidence =
        MutableStateFlow(1.0)
    val directionalUncertaintyConfidence: StateFlow<Double> =
        _directionalUncertaintyConfidence.asStateFlow()

    private val _directionalUncertaintyDegradedJointCount =
        MutableStateFlow(0)
    val directionalUncertaintyDegradedJointCount: StateFlow<Int> =
        _directionalUncertaintyDegradedJointCount.asStateFlow()

    private val _directionalUncertaintyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val directionalUncertaintyWarnings: StateFlow<List<String>> =
        _directionalUncertaintyWarnings.asStateFlow()

    private val _adaptiveViewEvidenceScore =
        MutableStateFlow(1.0)
    val adaptiveViewEvidenceScore: StateFlow<Double> =
        _adaptiveViewEvidenceScore.asStateFlow()

    private val _adaptiveViewEvidenceGlobalRigTrust =
        MutableStateFlow(1.0)
    val adaptiveViewEvidenceGlobalRigTrust: StateFlow<Double> =
        _adaptiveViewEvidenceGlobalRigTrust.asStateFlow()

    private val _adaptiveViewEvidenceDegradedCameraCount =
        MutableStateFlow(0)
    val adaptiveViewEvidenceDegradedCameraCount: StateFlow<Int> =
        _adaptiveViewEvidenceDegradedCameraCount.asStateFlow()

    private val _adaptiveViewEvidenceHoldCameraCount =
        MutableStateFlow(0)
    val adaptiveViewEvidenceHoldCameraCount: StateFlow<Int> =
        _adaptiveViewEvidenceHoldCameraCount.asStateFlow()

    private val _adaptiveViewEvidenceDominantCamera =
        MutableStateFlow("NONE")
    val adaptiveViewEvidenceDominantCamera: StateFlow<String> =
        _adaptiveViewEvidenceDominantCamera.asStateFlow()

    private val _adaptiveViewEvidenceWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val adaptiveViewEvidenceWarnings: StateFlow<List<String>> =
        _adaptiveViewEvidenceWarnings.asStateFlow()
    private val _cameraContributionSelectedCount =
        MutableStateFlow(0)
    val cameraContributionSelectedCount: StateFlow<Int> =
        _cameraContributionSelectedCount.asStateFlow()

    private val _cameraContributionExcludedCount =
        MutableStateFlow(0)
    val cameraContributionExcludedCount: StateFlow<Int> =
        _cameraContributionExcludedCount.asStateFlow()

    private val _cameraContributionDiversitySatisfied =
        MutableStateFlow(false)
    val cameraContributionDiversitySatisfied: StateFlow<Boolean> =
        _cameraContributionDiversitySatisfied.asStateFlow()

    private val _cameraContributionReason =
        MutableStateFlow("UNINITIALIZED")
    val cameraContributionReason: StateFlow<String> =
        _cameraContributionReason.asStateFlow()

    private val _cameraContributionInformationGainScore =
        MutableStateFlow(0.0)
    val cameraContributionInformationGainScore: StateFlow<Double> =
        _cameraContributionInformationGainScore.asStateFlow()
    private val _jointViewSelectionScore =
        MutableStateFlow(0.0)
    val jointViewSelectionScore: StateFlow<Double> =
        _jointViewSelectionScore.asStateFlow()

    private val _jointViewDiversitySatisfiedRatio =
        MutableStateFlow(0.0)
    val jointViewDiversitySatisfiedRatio: StateFlow<Double> =
        _jointViewDiversitySatisfiedRatio.asStateFlow()

    private val _jointSelectedObservationCount =
        MutableStateFlow(0)
    val jointSelectedObservationCount: StateFlow<Int> =
        _jointSelectedObservationCount.asStateFlow()

    private val _jointExcludedObservationCount =
        MutableStateFlow(0)
    val jointExcludedObservationCount: StateFlow<Int> =
        _jointExcludedObservationCount.asStateFlow()

    private val _jointViewSelectionReasons =
        MutableStateFlow<List<String>>(emptyList())
    val jointViewSelectionReasons: StateFlow<List<String>> =
        _jointViewSelectionReasons.asStateFlow()
    private val _rigHealthState =
        MutableStateFlow(MultiCameraRigHealthEngine.State.UNKNOWN.name)
    val rigHealthState: StateFlow<String> =
        _rigHealthState.asStateFlow()

    private val _rigHealthScore =
        MutableStateFlow(0.0)
    val rigHealthScore: StateFlow<Double> =
        _rigHealthScore.asStateFlow()

    private val _rigQuarantinedCameraCount =
        MutableStateFlow(0)
    val rigQuarantinedCameraCount: StateFlow<Int> =
        _rigQuarantinedCameraCount.asStateFlow()

    private val _rigHealthWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val rigHealthWarnings: StateFlow<List<String>> =
        _rigHealthWarnings.asStateFlow()

    private val _rigCameraHealth =
        MutableStateFlow<List<MultiCameraRigHealthEngine.CameraHealth>>(emptyList())
    val rigCameraHealth:
        StateFlow<List<MultiCameraRigHealthEngine.CameraHealth>> =
        _rigCameraHealth.asStateFlow()

    private val _spatiotemporalConfidence =
        MutableStateFlow(0.0)
    val spatiotemporalConfidence: StateFlow<Double> =
        _spatiotemporalConfidence.asStateFlow()

    private val _spatiotemporalInitialRmsPx =
        MutableStateFlow(0.0)
    val spatiotemporalInitialRmsPx: StateFlow<Double> =
        _spatiotemporalInitialRmsPx.asStateFlow()

    private val _spatiotemporalFinalRmsPx =
        MutableStateFlow(0.0)
    val spatiotemporalFinalRmsPx: StateFlow<Double> =
        _spatiotemporalFinalRmsPx.asStateFlow()

    private val _spatiotemporalOffsetsNanos =
        MutableStateFlow<Map<String, Double>>(emptyMap())
    val spatiotemporalOffsetsNanos:
        StateFlow<Map<String, Double>> =
        _spatiotemporalOffsetsNanos.asStateFlow()

    private val _spatiotemporalWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val spatiotemporalWarnings: StateFlow<List<String>> =
        _spatiotemporalWarnings.asStateFlow()

    private val _face3DErrorPx = MutableStateFlow(Double.POSITIVE_INFINITY)
    val face3DErrorPx: StateFlow<Double> = _face3DErrorPx.asStateFlow()

    private val _face3DTemporalConsistencyConfidence = MutableStateFlow(0.0)
    val face3DTemporalConsistencyConfidence: StateFlow<Double> =
        _face3DTemporalConsistencyConfidence.asStateFlow()

    private val _face3DTemporalConsistencyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val face3DTemporalConsistencyWarnings: StateFlow<List<String>> =
        _face3DTemporalConsistencyWarnings.asStateFlow()

    private val _eye3DConfidence = MutableStateFlow(0.0)
    val eye3DConfidence: StateFlow<Double> = _eye3DConfidence.asStateFlow()

    private val _eye3DTemporalConsistencyConfidence =
        MutableStateFlow(0.0)
    val eye3DTemporalConsistencyConfidence: StateFlow<Double> =
        _eye3DTemporalConsistencyConfidence.asStateFlow()

    private val _eye3DTemporalConsistencyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val eye3DTemporalConsistencyWarnings: StateFlow<List<String>> =
        _eye3DTemporalConsistencyWarnings.asStateFlow()

    private val _gazeConfidence = MutableStateFlow(0.0)
    val gazeConfidence: StateFlow<Double> = _gazeConfidence.asStateFlow()

    private val _faceEyeConsistencyConfidence = MutableStateFlow(0.0)
    val faceEyeConsistencyConfidence: StateFlow<Double> =
        _faceEyeConsistencyConfidence.asStateFlow()

    private val _faceEyeConsistencyWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeConsistencyWarnings: StateFlow<List<String>> =
        _faceEyeConsistencyWarnings.asStateFlow()

    private val _faceEyeGeometricIntegrity =
        MutableStateFlow(0.0)
    val faceEyeGeometricIntegrity: StateFlow<Double> =
        _faceEyeGeometricIntegrity.asStateFlow()

    private val _faceEyeFaceFreshnessScore =
        MutableStateFlow(0.0)
    val faceEyeFaceFreshnessScore: StateFlow<Double> =
        _faceEyeFaceFreshnessScore.asStateFlow()

    private val _faceEyeGazeAgreementScore =
        MutableStateFlow(0.0)
    val faceEyeGazeAgreementScore: StateFlow<Double> =
        _faceEyeGazeAgreementScore.asStateFlow()

    private val _faceEyeRayGeometryCondition =
        MutableStateFlow(0.0)
    val faceEyeRayGeometryCondition: StateFlow<Double> =
        _faceEyeRayGeometryCondition.asStateFlow()

    private val _faceEyeIpdConsistencyScore =
        MutableStateFlow(1.0)
    val faceEyeIpdConsistencyScore: StateFlow<Double> =
        _faceEyeIpdConsistencyScore.asStateFlow()

    private val _faceEyeOutOfDistributionRisk =
        MutableStateFlow(0.0)
    val faceEyeOutOfDistributionRisk: StateFlow<Double> =
        _faceEyeOutOfDistributionRisk.asStateFlow()

    private val _faceEyeShapeConfidence =
        MutableStateFlow(0.0)
    val faceEyeShapeConfidence: StateFlow<Double> =
        _faceEyeShapeConfidence.asStateFlow()

    private val _faceEyeEyeDynamicConsistency =
        MutableStateFlow(0.0)
    val faceEyeEyeDynamicConsistency: StateFlow<Double> =
        _faceEyeEyeDynamicConsistency.asStateFlow()

    private val _faceEyeHardFailureReasons =
        MutableStateFlow<List<String>>(emptyList())
    val faceEyeHardFailureReasons: StateFlow<List<String>> =
        _faceEyeHardFailureReasons.asStateFlow()

    private val _gazeStability = MutableStateFlow(0.0)
    val gazeStability: StateFlow<Double> = _gazeStability.asStateFlow()

    private val _faceHeadYaw = MutableStateFlow(0.0)
    val faceHeadYaw: StateFlow<Double> = _faceHeadYaw.asStateFlow()

    private val _faceHeadRoll = MutableStateFlow(0.0)
    val faceHeadRoll: StateFlow<Double> = _faceHeadRoll.asStateFlow()

    private val _lightingLux = MutableStateFlow(0.0)
    val lightingLux: StateFlow<Double> = _lightingLux.asStateFlow()

    private val _colorTemperature = MutableStateFlow(0.0)
    val colorTemperature: StateFlow<Double> = _colorTemperature.asStateFlow()

    private val _lightingConfidence = MutableStateFlow(0.0)
    val lightingConfidence: StateFlow<Double> =
        _lightingConfidence.asStateFlow()

    private val _gpuBackend = MutableStateFlow(gpuAcceleration.backend)
    val gpuBackend: StateFlow<GpuBackend> = _gpuBackend.asStateFlow()

    private val _rawSidecarEnabled = MutableStateFlow(false)
    val rawSidecarEnabled: StateFlow<Boolean> = _rawSidecarEnabled.asStateFlow()

    private val _depthSidecarEnabled = MutableStateFlow(false)
    val depthSidecarEnabled: StateFlow<Boolean> = _depthSidecarEnabled.asStateFlow()


    private val _motionCustomization =
        MutableStateFlow(MotionCaptureCustomization.production())
    val motionCustomization: StateFlow<MotionCaptureCustomization> =
        _motionCustomization.asStateFlow()

    private val _sessionPresentationSettings =
        MutableStateFlow(SessionPresentationSettings())
    val sessionPresentationSettings: StateFlow<SessionPresentationSettings> =
        _sessionPresentationSettings.asStateFlow()

    private val _recordingCountdown = MutableStateFlow(0)
    val recordingCountdown: StateFlow<Int> =
        _recordingCountdown.asStateFlow()

    private val _captureReadiness =
        MutableStateFlow<com.nivukx.nox.core.model.CaptureReadiness?>(null)
    val captureReadiness:
        StateFlow<com.nivukx.nox.core.model.CaptureReadiness?> =
        _captureReadiness.asStateFlow()

    private val _calibrationReady = MutableStateFlow(false)
    val calibrationReady: StateFlow<Boolean> = _calibrationReady.asStateFlow()

    private val _rigScore = MutableStateFlow(0.0)
    val rigScore: StateFlow<Double> = _rigScore.asStateFlow()

    private val _rigReasons = MutableStateFlow<List<String>>(emptyList())
    val rigReasons: StateFlow<List<String>> = _rigReasons.asStateFlow()

    private val _rigCoverageScore = MutableStateFlow(0.0)
    val rigCoverageScore: StateFlow<Double> = _rigCoverageScore.asStateFlow()

    private val _rigRecommendations = MutableStateFlow<List<String>>(emptyList())
    val rigRecommendations: StateFlow<List<String>> = _rigRecommendations.asStateFlow()

    private val _multiCameraSyncConfidence = MutableStateFlow(1.0)
    val multiCameraSyncConfidence: StateFlow<Double> =
        _multiCameraSyncConfidence.asStateFlow()

    private val _multiCameraSyncState =
        MutableStateFlow(
            MultiCameraSynchronizationQualityEngine.State.NO_PEERS.name,
        )
    val multiCameraSyncState: StateFlow<String> =
        _multiCameraSyncState.asStateFlow()

    private val _temporalAlignmentConfidence =
        MutableStateFlow(0.0)
    val temporalAlignmentConfidence: StateFlow<Double> =
        _temporalAlignmentConfidence.asStateFlow()

    private val _temporalAlignmentSpreadMillis =
        MutableStateFlow(0.0)
    val temporalAlignmentSpreadMillis: StateFlow<Double> =
        _temporalAlignmentSpreadMillis.asStateFlow()

    private val _temporalAlignmentState =
        MutableStateFlow("NO_DATA")
    val temporalAlignmentState: StateFlow<String> =
        _temporalAlignmentState.asStateFlow()

    private val _temporalAlignmentWarnings =
        MutableStateFlow<List<String>>(emptyList())
    val temporalAlignmentWarnings: StateFlow<List<String>> =
        _temporalAlignmentWarnings.asStateFlow()

    private val _sessions = MutableStateFlow(sessions.list())
    val sessionList: StateFlow<List<SessionRepository.SessionSummary>> = _sessions.asStateFlow()

    private val initialMotionTuning = loadMotionTuning()
    private val initialReconstructionTuning = loadReconstructionTuning()
    private val initialTemporalReconstructionTuning =
        loadTemporalReconstructionTuning()

    private val _motionProfile = MutableStateFlow(
        controlPreferences.getString("motion_profile", "CUSTOM") ?: "CUSTOM",
    )
    val motionProfile: StateFlow<String> = _motionProfile.asStateFlow()

    private val _motionResponsiveness = MutableStateFlow(
        initialMotionTuning.responsiveness,
    )
    val motionResponsiveness: StateFlow<Double> =
        _motionResponsiveness.asStateFlow()

    private val _motionSensitivity = MutableStateFlow(
        initialMotionTuning.motionSensitivity,
    )
    val motionSensitivity: StateFlow<Double> =
        _motionSensitivity.asStateFlow()

    private val _predictionHorizon = MutableStateFlow(
        initialMotionTuning.predictionHorizonSeconds,
    )
    val predictionHorizon: StateFlow<Double> =
        _predictionHorizon.asStateFlow()

    private val _occlusionPersistence = MutableStateFlow(
        initialMotionTuning.occlusionPersistence,
    )
    val occlusionPersistence: StateFlow<Double> =
        _occlusionPersistence.asStateFlow()

    private val _reacquisitionRadius = MutableStateFlow(
        initialMotionTuning.reacquisitionRadius,
    )
    val reacquisitionRadius: StateFlow<Double> =
        _reacquisitionRadius.asStateFlow()

    private val _rejectAggressiveness = MutableStateFlow(
        initialMotionTuning.rejectAggressiveness,
    )
    val rejectAggressiveness: StateFlow<Double> =
        _rejectAggressiveness.asStateFlow()

    private val _jitterSuppression = MutableStateFlow(
        initialMotionTuning.jitterSuppression,
    )
    val jitterSuppression: StateFlow<Double> =
        _jitterSuppression.asStateFlow()

    private val _velocityResponsiveness = MutableStateFlow(
        initialMotionTuning.velocityResponsiveness,
    )
    val velocityResponsiveness: StateFlow<Double> =
        _velocityResponsiveness.asStateFlow()

    private val _accelerationResponsiveness = MutableStateFlow(
        initialMotionTuning.accelerationResponsiveness,
    )
    val accelerationResponsiveness: StateFlow<Double> =
        _accelerationResponsiveness.asStateFlow()

    private val _innovationGateElasticity = MutableStateFlow(
        initialMotionTuning.innovationGateElasticity,
    )
    val innovationGateElasticity: StateFlow<Double> =
        _innovationGateElasticity.asStateFlow()

    private val _motionThresholdScale = MutableStateFlow(
        initialMotionTuning.motionThresholdScale,
    )
    val motionThresholdScale: StateFlow<Double> =
        _motionThresholdScale.asStateFlow()

    private val _adaptiveRegimeStrength = MutableStateFlow(
        initialMotionTuning.adaptiveRegimeStrength,
    )
    val adaptiveRegimeStrength: StateFlow<Double> =
        _adaptiveRegimeStrength.asStateFlow()

    private val _reconstructionProfile = MutableStateFlow(
        controlPreferences.getString("reconstruction_profile", "CUSTOM") ?: "CUSTOM",
    )
    val reconstructionProfile: StateFlow<String> =
        _reconstructionProfile.asStateFlow()

    private val _reconstructionSoftAccept = MutableStateFlow(
        initialReconstructionTuning.softAcceptScore,
    )
    val reconstructionSoftAccept: StateFlow<Double> =
        _reconstructionSoftAccept.asStateFlow()

    private val _reconstructionHoldFrames = MutableStateFlow(
        initialReconstructionTuning.holdFrames,
    )
    val reconstructionHoldFrames: StateFlow<Int> =
        _reconstructionHoldFrames.asStateFlow()

    private val _reconstructionHoldAgeMillis = MutableStateFlow(
        initialReconstructionTuning.holdAgeMillis,
    )
    val reconstructionHoldAgeMillis: StateFlow<Long> =
        _reconstructionHoldAgeMillis.asStateFlow()

    private val _reconstructionStabilitySensitivity = MutableStateFlow(
        initialReconstructionTuning.stabilitySensitivity,
    )
    val reconstructionStabilitySensitivity: StateFlow<Double> =
        _reconstructionStabilitySensitivity.asStateFlow()

    private val _temporalPredictionHorizon =
        MutableStateFlow(initialTemporalReconstructionTuning.predictionHorizonSeconds)
    val temporalPredictionHorizon: StateFlow<Double> =
        _temporalPredictionHorizon.asStateFlow()

    private val _temporalInnovationGateScale =
        MutableStateFlow(initialTemporalReconstructionTuning.innovationGateScale)
    val temporalInnovationGateScale: StateFlow<Double> =
        _temporalInnovationGateScale.asStateFlow()

    private val _temporalUncertaintyGateScale =
        MutableStateFlow(initialTemporalReconstructionTuning.uncertaintyGateScale)
    val temporalUncertaintyGateScale: StateFlow<Double> =
        _temporalUncertaintyGateScale.asStateFlow()

    private val _temporalReprojectionGateScale =
        MutableStateFlow(initialTemporalReconstructionTuning.reprojectionGateScale)
    val temporalReprojectionGateScale: StateFlow<Double> =
        _temporalReprojectionGateScale.asStateFlow()

    private val _temporalSmoothingResponsiveness =
        MutableStateFlow(initialTemporalReconstructionTuning.smoothingResponsiveness)
    val temporalSmoothingResponsiveness: StateFlow<Double> =
        _temporalSmoothingResponsiveness.asStateFlow()

    private val _temporalConfidenceDecayScale =
        MutableStateFlow(initialTemporalReconstructionTuning.confidenceDecayScale)
    val temporalConfidenceDecayScale: StateFlow<Double> =
        _temporalConfidenceDecayScale.asStateFlow()

    private val _temporalOutlierRetention =
        MutableStateFlow(initialTemporalReconstructionTuning.outlierRetention)
    val temporalOutlierRetention: StateFlow<Double> =
        _temporalOutlierRetention.asStateFlow()

    private val _temporalTopologyToleranceScale =
        MutableStateFlow(initialTemporalReconstructionTuning.topologyToleranceScale)
    val temporalTopologyToleranceScale: StateFlow<Double> =
        _temporalTopologyToleranceScale.asStateFlow()

    init {
        _sessionPresentationSettings.value =
            loadSessionPresentationSettings()
        val initialCustomization = loadMotionCustomization()
        _motionCustomization.value = initialCustomization
        perception.configureCustomization(initialCustomization)
        monocularPipeline.configureCapturePolicy(
            maximumOcclusionPredictionFrames =
                initialCustomization.maximumOcclusionPredictionFrames,
            captureFps = initialCustomization.targetFps.coerceIn(15, 240),
        )
        assignedRoles[localId] = DeviceRole.MASTER
        applyMotionTuning(initialMotionTuning, persist = false)
        applyReconstructionTuning(initialReconstructionTuning, persist = false)
        applyTemporalReconstructionTuning(
            initialTemporalReconstructionTuning,
            persist = false,
        )
        refreshDevices(emptySet())

        viewModelScope.launch {
            peers.connected.collect { connected ->
                refreshDevices(connected)
                connected.forEach {
                    sendClockProbes(it)
                    sendLocalCalibrationFrame(it)
                }
            }
        }

        viewModelScope.launch {
            peers.packets.collect { event -> handlePeerPacket(event.endpoint, event.payload) }
        }

        // Fragmented dense evidence is reassembled and CRC-checked by NearbyPeerController.
        // Consume the completed frames as a first-class production input, not as telemetry only.
        viewModelScope.launch {
            peers.denseLandmarkFrames.collect { event ->
                handleDenseLandmarkFrame(event.endpoint, event.frame)
            }
        }
    }

    fun applyMotionPreset(name: String) {
        val preset = when (name.uppercase(Locale.US)) {
            "CINEMATIC" -> MotionCaptureTuning.cinematic()
            "ACTION" -> MotionCaptureTuning.action()
            "PRECISION" -> MotionCaptureTuning.precision()
            "CREATURE" -> MotionCaptureTuning.creature()
            "FACIAL" -> MotionCaptureTuning.facial()
            "HAND_DETAIL" -> MotionCaptureTuning.handDetail()
            else -> tuningFromState()
        }
        _motionProfile.value = name.uppercase(Locale.US)
        applyMotionTuning(preset)
    }

    fun applyCustomizationPreset(name: String) {
        val preset = when (name.uppercase(Locale.US)) {
            "QUALITY" -> MotionCaptureCustomization.qualityFirst()
            "LOW_LATENCY" -> MotionCaptureCustomization.lowLatency()
            "BALANCED" -> MotionCaptureCustomization.production()
            "MULTIVIEW_PRECISION" -> MotionCaptureCustomization.multiViewPrecision()
            "FACE_EYE_PRECISION" -> MotionCaptureCustomization.faceEyePrecision()
            else -> _motionCustomization.value
        }
        applyMotionCustomization(preset)
    }

    fun updateAdvancedMotionCustomization(
        transform: (MotionCaptureCustomization) -> MotionCaptureCustomization,
    ) {
        applyMotionCustomization(
            transform(
                _motionCustomization.value,
            ),
        )
    }

    fun setLiveMotionPreviewEnabled(enabled: Boolean) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                liveMotionPreviewEnabled = enabled
            },
        )
        if (!enabled) {
            motionFieldSynchronizer.clear()
            _liveMotionPreviewFrame.value = null
        }
    }

    fun setLiveMotionPreviewGpuMode(mode: MotionFieldGpuMode) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                liveMotionPreviewGpuMode = mode
            },
        )
    }

    fun setCustomizationBodyDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                bodyDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationFaceDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                faceDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationEyeDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                eyeDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationHandDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                handDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationDepthDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                depthDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationDenseMeshDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                denseMeshDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationWorldConsistencyDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                worldConsistencyDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationMultiViewEvidenceAdaptivity(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                multiViewEvidenceAdaptivity = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationTopologyConstraintStrength(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                topologyConstraintStrength = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationContactDetail(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                contactDetail = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationMaxSubjects(value: Int) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                maxSubjects = value.coerceIn(1, 64)
            },
        )
    }

    fun setCaptureEvidenceMode(mode: CaptureEvidenceMode) {
        updateAdvancedMotionCustomization { current ->
            current.copyWith {
                captureEvidenceMode = mode
                recordedVideoEvidenceProfile =
                    if (mode == CaptureEvidenceMode.LIVE_MOTION_ONLY) {
                        current.recordedVideoEvidenceProfile
                    } else {
                        current.recordedVideoEvidenceProfile
                            ?: RecordedVideoEvidenceProfile.liveUltra()
                    }
            }
        }
    }

    fun setRecordedVideoEvidenceProfile(
        profile: RecordedVideoEvidenceProfile,
    ) {
        updateAdvancedMotionCustomization { current ->
            current.copyWith {
                recordedVideoEvidenceProfile = profile
            }
        }
    }

    fun setCustomizationOcclusionPredictionFrames(value: Int) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                maximumOcclusionPredictionFrames =
                    value.coerceIn(0, 60)
            },
        )
    }

    fun setCustomizationTargetFps(value: Int) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                targetFps = value.coerceIn(15, 240)
            },
        )
    }

    fun setCustomizationInferenceFps(value: Int) {
        val target = _motionCustomization.value.targetFps
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                inferenceFps = value.coerceIn(
                    5,
                    target,
                )
            },
        )
    }

    fun setCustomizationMicroMotionPreservation(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                microMotionPreservation = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationCameraMotionSuppression(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                cameraMotionSuppression = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationFaceEyeFusionWeight(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                faceEyeFusionWeight = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationEyeTemporalStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                eyeTemporalStrictness = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationHandTemporalStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                handTemporalStrictness = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationHandGeometryStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                handGeometryStrictness = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationHandLongitudinalAnatomyStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                handLongitudinalAnatomyStrictness =
                    value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationTemporalAlignmentStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                temporalAlignmentStrictness = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationOutputIntegrityStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                outputIntegrityStrictness = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationPhysicsConstraintStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                physicsConstraintStrictness = value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationGlobal4DConsistencyStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                global4DConsistencyStrictness =
                    value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationSurfaceEvidenceStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                surfaceEvidenceStrictness =
                    value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationFaceEyeFusionStrictness(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                faceEyeFusionStrictness =
                    value.coerceIn(0.0, 1.0)
            },
        )
    }

    fun setCustomizationFaceEyeIpdDriftToleranceMillimeters(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                faceEyeIpdDriftToleranceMillimeters =
                    value.coerceIn(4.0, 30.0)
            },
        )
    }

    fun setCustomizationMinimumParallaxDegrees(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                minimumParallaxDegrees = value.coerceIn(0.20, 2.50)
            },
        )
    }

    fun setCustomizationPreferredTemporalSkewMillis(value: Double) {
        val current = _motionCustomization.value
        val safeMaximum =
            (current.maximumTemporalSkewMillis - 0.50).coerceAtLeast(1.0)
        applyMotionCustomization(
            current.copyWith {
                preferredTemporalSkewMillis =
                    value.coerceIn(0.50, safeMaximum)
            },
        )
    }

    fun setCustomizationMaximumTemporalSkewMillis(value: Double) {
        val current = _motionCustomization.value
        val safeMinimum =
            current.preferredTemporalSkewMillis + 0.50
        applyMotionCustomization(
            current.copyWith {
                maximumTemporalSkewMillis =
                    value.coerceAtLeast(safeMinimum).coerceAtMost(250.0)
            },
        )
    }

    fun setCustomizationMaximumReprojectionErrorPx(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                maximumReprojectionErrorPx =
                    value.coerceIn(1.0, 30.0)
            },
        )
    }

    fun setCustomizationMaximumReconstructionUncertaintyMeters(value: Double) {
        applyMotionCustomization(
            _motionCustomization.value.copyWith {
                maximumReconstructionUncertaintyMeters =
                    value.coerceIn(0.05, 3.0)
            },
        )
    }

    fun setSessionClipName(value: String) {
        val sanitized = value
            .trim()
            .replace(Regex("[^A-Za-z0-9._ -]"), "_")
            .take(80)
            .ifBlank { "Take" }

        applySessionPresentationSettings(
            _sessionPresentationSettings.value.copy(
                clipName = sanitized,
            ),
        )
    }

    fun setSessionCountdownSeconds(value: Int) {
        applySessionPresentationSettings(
            _sessionPresentationSettings.value.copy(
                countdownSeconds = value.coerceIn(0, 10),
            ),
        )
    }

    fun setRequireCalibrationBeforeRecord(value: Boolean) {
        applySessionPresentationSettings(
            _sessionPresentationSettings.value.copy(
                requireCalibrationBeforeRecord = value,
            ),
        )
    }

    fun setMotionResponsiveness(value: Double) {
        applyMotionTuning(tuningFromState().copy(responsiveness = value))
    }

    fun setMotionSensitivity(value: Double) {
        applyMotionTuning(tuningFromState().copy(motionSensitivity = value))
    }

    fun setPredictionHorizon(value: Double) {
        applyMotionTuning(tuningFromState().copy(predictionHorizonSeconds = value))
    }

    fun setOcclusionPersistence(value: Double) {
        applyMotionTuning(tuningFromState().copy(occlusionPersistence = value))
    }

    fun setReacquisitionRadius(value: Double) {
        applyMotionTuning(tuningFromState().copy(reacquisitionRadius = value))
    }

    fun setRejectAggressiveness(value: Double) {
        applyMotionTuning(tuningFromState().copy(rejectAggressiveness = value))
    }

    fun setJitterSuppression(value: Double) {
        applyMotionTuning(tuningFromState().copy(jitterSuppression = value))
    }

    fun setVelocityResponsiveness(value: Double) {
        applyMotionTuning(
            tuningFromState().copy(velocityResponsiveness = value),
        )
    }

    fun setAccelerationResponsiveness(value: Double) {
        applyMotionTuning(
            tuningFromState().copy(accelerationResponsiveness = value),
        )
    }

    fun setInnovationGateElasticity(value: Double) {
        applyMotionTuning(
            tuningFromState().copy(innovationGateElasticity = value),
        )
    }

    fun setMotionThresholdScale(value: Double) {
        applyMotionTuning(
            tuningFromState().copy(motionThresholdScale = value),
        )
    }

    fun setAdaptiveRegimeStrength(value: Double) {
        applyMotionTuning(
            tuningFromState().copy(
                adaptiveRegimeStrength = value.coerceIn(0.0, 1.0),
            ),
        )
    }

    fun applyReconstructionPreset(name: String) {
        val preset = when (name.uppercase(Locale.US)) {
            "CINEMATIC" -> ReconstructionContinuityTuning.cinematic()
            "ACTION" -> ReconstructionContinuityTuning.liveAction()
            "PRECISION" -> ReconstructionContinuityTuning.precision()
            "CREATURE" -> ReconstructionContinuityTuning.creature()
            "FACIAL" -> ReconstructionContinuityTuning.facial()
            "HAND_DETAIL" -> ReconstructionContinuityTuning.handDetail()
            else -> reconstructionTuningFromState()
        }
        val temporalPreset = when (name.uppercase(Locale.US)) {
            "CINEMATIC" -> TemporalReconstructionTuning.cinematic()
            "ACTION" -> TemporalReconstructionTuning.liveAction()
            "PRECISION" -> TemporalReconstructionTuning.precision()
            "CREATURE" -> TemporalReconstructionTuning.creature()
            "FACIAL" -> TemporalReconstructionTuning.facial()
            "HAND_DETAIL" -> TemporalReconstructionTuning.handDetail()
            else -> temporalReconstructionTuningFromState()
        }
        _reconstructionProfile.value = name.uppercase(Locale.US)
        applyReconstructionTuning(preset)
        applyTemporalReconstructionTuning(temporalPreset)
    }

    fun setReconstructionSoftAccept(value: Double) {
        applyReconstructionTuning(
            reconstructionTuningFromState().copy(softAcceptScore = value),
        )
    }

    fun setReconstructionHoldFrames(value: Int) {
        applyReconstructionTuning(
            reconstructionTuningFromState().copy(holdFrames = value),
        )
    }

    fun setReconstructionHoldAgeMillis(value: Long) {
        applyReconstructionTuning(
            reconstructionTuningFromState().copy(holdAgeMillis = value),
        )
    }

    fun setReconstructionStabilitySensitivity(value: Double) {
        applyReconstructionTuning(
            reconstructionTuningFromState().copy(
                stabilitySensitivity = value,
            ),
        )
    }

    fun setTemporalPredictionHorizon(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                predictionHorizonSeconds = value,
            ),
        )
    }

    fun setTemporalInnovationGateScale(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                innovationGateScale = value,
            ),
        )
    }

    fun setTemporalUncertaintyGateScale(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                uncertaintyGateScale = value,
            ),
        )
    }

    fun setTemporalReprojectionGateScale(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                reprojectionGateScale = value,
            ),
        )
    }

    fun setTemporalSmoothingResponsiveness(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                smoothingResponsiveness = value,
            ),
        )
    }

    fun setTemporalConfidenceDecayScale(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                confidenceDecayScale = value,
            ),
        )
    }

    fun setTemporalOutlierRetention(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                outlierRetention = value,
            ),
        )
    }

    fun setTemporalTopologyToleranceScale(value: Double) {
        applyTemporalReconstructionTuning(
            temporalReconstructionTuningFromState().copy(
                topologyToleranceScale = value,
            ),
        )
    }

    private fun tuningFromState(): MotionCaptureTuning {
        val current = perception.motionTuning()
        return MotionCaptureTuning(
            responsiveness = _motionResponsiveness.value,
            motionSensitivity = _motionSensitivity.value,
            predictionHorizonSeconds = _predictionHorizon.value,
            occlusionPersistence = _occlusionPersistence.value,
            reacquisitionRadius = _reacquisitionRadius.value,
            minimumConfidence = current.minimumConfidence,
            rejectAggressiveness = _rejectAggressiveness.value,
            jitterSuppression = _jitterSuppression.value,
            velocityResponsiveness = _velocityResponsiveness.value,
            accelerationResponsiveness = _accelerationResponsiveness.value,
            innovationGateElasticity = _innovationGateElasticity.value,
            motionThresholdScale = _motionThresholdScale.value,
            adaptiveRegimeStrength = _adaptiveRegimeStrength.value,
        )
    }

    private fun reconstructionTuningFromState(): ReconstructionContinuityTuning {
        val current = fusion.continuityTuning()
        return ReconstructionContinuityTuning(
            softAcceptScore = _reconstructionSoftAccept.value,
            hardRejectScore = current.hardRejectScore
                .coerceAtMost(_reconstructionSoftAccept.value - 0.01)
                .coerceAtLeast(0.10),
            maxJointJumpMeters = current.maxJointJumpMeters,
            maxJointSpeedMetersPerSecond =
                current.maxJointSpeedMetersPerSecond,
            holdFrames = _reconstructionHoldFrames.value,
            holdAgeMillis = _reconstructionHoldAgeMillis.value,
            stabilitySensitivity = _reconstructionStabilitySensitivity.value,
        )
    }

    private fun applyMotionTuning(
        tuning: MotionCaptureTuning,
        persist: Boolean = true,
    ) {
        perception.configureMotionTuning(tuning)
        _motionResponsiveness.value = tuning.responsiveness
        _motionSensitivity.value = tuning.motionSensitivity
        _predictionHorizon.value = tuning.predictionHorizonSeconds
        _occlusionPersistence.value = tuning.occlusionPersistence
        _reacquisitionRadius.value = tuning.reacquisitionRadius
        _rejectAggressiveness.value = tuning.rejectAggressiveness
        _jitterSuppression.value = tuning.jitterSuppression
        _velocityResponsiveness.value = tuning.velocityResponsiveness
        _accelerationResponsiveness.value = tuning.accelerationResponsiveness
        _innovationGateElasticity.value = tuning.innovationGateElasticity
        _motionThresholdScale.value = tuning.motionThresholdScale
        _adaptiveRegimeStrength.value = tuning.adaptiveRegimeStrength
        if (persist) {
            controlPreferences.edit()
                .putFloat("motion_responsiveness", tuning.responsiveness.toFloat())
                .putFloat("motion_sensitivity", tuning.motionSensitivity.toFloat())
                .putFloat("prediction_horizon", tuning.predictionHorizonSeconds.toFloat())
                .putFloat("occlusion_persistence", tuning.occlusionPersistence.toFloat())
                .putFloat("reacquisition_radius", tuning.reacquisitionRadius.toFloat())
                .putFloat("reject_aggressiveness", tuning.rejectAggressiveness.toFloat())
                .putFloat("jitter_suppression", tuning.jitterSuppression.toFloat())
                .putFloat("velocity_responsiveness", tuning.velocityResponsiveness.toFloat())
                .putFloat("acceleration_responsiveness", tuning.accelerationResponsiveness.toFloat())
                .putFloat("innovation_gate_elasticity", tuning.innovationGateElasticity.toFloat())
                .putFloat("motion_threshold_scale", tuning.motionThresholdScale.toFloat())
                .putFloat("adaptive_regime_strength", tuning.adaptiveRegimeStrength.toFloat())
                .apply()
        }
    }

    private fun applyReconstructionTuning(
        tuning: ReconstructionContinuityTuning,
        persist: Boolean = true,
    ) {
        fusion.configureContinuity(tuning)
        _reconstructionSoftAccept.value = tuning.softAcceptScore
        _reconstructionHoldFrames.value = tuning.holdFrames
        _reconstructionHoldAgeMillis.value = tuning.holdAgeMillis
        _reconstructionStabilitySensitivity.value = tuning.stabilitySensitivity
        if (persist) {
            controlPreferences.edit()
                .putFloat("reconstruction_soft_accept", tuning.softAcceptScore.toFloat())
                .putInt("reconstruction_hold_frames", tuning.holdFrames)
                .putLong("reconstruction_hold_age", tuning.holdAgeMillis)
                .putFloat("reconstruction_stability", tuning.stabilitySensitivity.toFloat())
                .apply()
        }
    }

    private fun temporalReconstructionTuningFromState(): TemporalReconstructionTuning =
        fusion.temporalTuning().copy(
            predictionHorizonSeconds = _temporalPredictionHorizon.value,
            innovationGateScale = _temporalInnovationGateScale.value,
            uncertaintyGateScale = _temporalUncertaintyGateScale.value,
            reprojectionGateScale = _temporalReprojectionGateScale.value,
            smoothingResponsiveness = _temporalSmoothingResponsiveness.value,
            confidenceDecayScale = _temporalConfidenceDecayScale.value,
            outlierRetention = _temporalOutlierRetention.value,
            topologyToleranceScale = _temporalTopologyToleranceScale.value,
        )

    private fun applyTemporalReconstructionTuning(
        tuning: TemporalReconstructionTuning,
        persist: Boolean = true,
    ) {
        fusion.configureTemporalTuning(tuning)
        _temporalPredictionHorizon.value = tuning.predictionHorizonSeconds
        _temporalInnovationGateScale.value = tuning.innovationGateScale
        _temporalUncertaintyGateScale.value = tuning.uncertaintyGateScale
        _temporalReprojectionGateScale.value = tuning.reprojectionGateScale
        _temporalSmoothingResponsiveness.value = tuning.smoothingResponsiveness
        _temporalConfidenceDecayScale.value = tuning.confidenceDecayScale
        _temporalOutlierRetention.value = tuning.outlierRetention
        _temporalTopologyToleranceScale.value = tuning.topologyToleranceScale

        if (persist) {
            controlPreferences.edit()
                .putFloat("temporal_prediction_horizon", tuning.predictionHorizonSeconds.toFloat())
                .putFloat("temporal_innovation_gate", tuning.innovationGateScale.toFloat())
                .putFloat("temporal_uncertainty_gate", tuning.uncertaintyGateScale.toFloat())
                .putFloat("temporal_reprojection_gate", tuning.reprojectionGateScale.toFloat())
                .putFloat("temporal_smoothing", tuning.smoothingResponsiveness.toFloat())
                .putFloat("temporal_confidence_decay", tuning.confidenceDecayScale.toFloat())
                .putFloat("temporal_outlier_retention", tuning.outlierRetention.toFloat())
                .putFloat("temporal_topology_tolerance", tuning.topologyToleranceScale.toFloat())
                .apply()
        }
    }

    private fun loadTemporalReconstructionTuning(): TemporalReconstructionTuning =
        TemporalReconstructionTuning(
            predictionHorizonSeconds = controlPreferences.getFloat(
                "temporal_prediction_horizon",
                TemporalReconstructionTuning().predictionHorizonSeconds.toFloat(),
            ).toDouble(),
            innovationGateScale = controlPreferences.getFloat(
                "temporal_innovation_gate",
                TemporalReconstructionTuning().innovationGateScale.toFloat(),
            ).toDouble(),
            uncertaintyGateScale = controlPreferences.getFloat(
                "temporal_uncertainty_gate",
                TemporalReconstructionTuning().uncertaintyGateScale.toFloat(),
            ).toDouble(),
            reprojectionGateScale = controlPreferences.getFloat(
                "temporal_reprojection_gate",
                TemporalReconstructionTuning().reprojectionGateScale.toFloat(),
            ).toDouble(),
            smoothingResponsiveness = controlPreferences.getFloat(
                "temporal_smoothing",
                TemporalReconstructionTuning().smoothingResponsiveness.toFloat(),
            ).toDouble(),
            confidenceDecayScale = controlPreferences.getFloat(
                "temporal_confidence_decay",
                TemporalReconstructionTuning().confidenceDecayScale.toFloat(),
            ).toDouble(),
            outlierRetention = controlPreferences.getFloat(
                "temporal_outlier_retention",
                TemporalReconstructionTuning().outlierRetention.toFloat(),
            ).toDouble(),
            topologyToleranceScale = controlPreferences.getFloat(
                "temporal_topology_tolerance",
                TemporalReconstructionTuning().topologyToleranceScale.toFloat(),
            ).toDouble(),
        )


    private fun updateGlobalDenseLandmarkEvidence(
        deviceId: DeviceId,
        timestampNanos: Long,
        dense:
            DenseLandmarkEvidenceFusionEngine.FrameResult,
    ) {
        val observations =
            dense.landmarks
                .filter {
                    it.state !=
                        DenseLandmarkEvidenceFusionEngine
                            .EvidenceState.REJECTED
                }
                .map { landmark ->
                    DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                        subjectId = landmark.key.subjectId,
                        landmarkId = landmark.key.landmarkId,
                    ) to
                        DenseLandmarkEvidenceFusionEngine.CameraObservation(
                            cameraId = denseEvidenceCameraId(deviceId),
                            timestampNanos = timestampNanos,
                            normalizedX = landmark.normalizedX,
                            normalizedY = landmark.normalizedY,
                            depthMeters = landmark.depthMeters,
                            confidence = landmark.confidence,
                            uncertainty = landmark.uncertainty,
                            visibility = landmark.visibility,
                            contactProbability = landmark.contactProbability,
                            cameraQuality = landmark.temporalStability,
                            reprojectionResidualNormalized =
                                landmark.weightedReprojectionResidualNormalized,
                            depthResidualMeters =
                                landmark.weightedDepthResidualMeters,
                            sourceTag = "local-dense",
                            rollingShutterCompensated = true,
                        )
                }

        if (observations.isEmpty()) return

        val global =
            globalDenseLandmarkEvidenceEngine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = timestampNanos,
                    observations = observations,
                ),
            )

        updateDenseLandmarkUiFromGlobal(
            dense = global,
            connectedCameraCount =
                _devices.value.count { it.connected }
                    .coerceAtLeast(1),
        )
    }

    private fun updateDenseLandmarkUiFromGlobal(
        dense:
            DenseLandmarkEvidenceFusionEngine.FrameResult,
        connectedCameraCount: Int,
    ) {
        _denseLandmarkCoverage.value =
            dense.coverage.coerceIn(0.0, 1.0)
        _denseLandmarkVisibleCoverage.value =
            dense.visibleCoverage.coerceIn(0.0, 1.0)
        _denseLandmarkUncertaintyP95.value =
            dense.uncertaintyP95.coerceIn(0.0, 1.0)
        _denseLandmarkCrossViewAgreement.value =
            dense.crossViewAgreement.coerceIn(0.0, 1.0)
        _denseLandmarkCrossViewEvidenceAvailable.value =
            dense.crossViewEvidenceAvailable
        _denseLandmarkTemporalStability.value =
            dense.temporalStability.coerceIn(0.0, 1.0)

        _denseGeometricConfidence.value =
            dense.geometricConfidence.coerceIn(0.0, 1.0)
        _denseViewDiversity.value =
            dense.viewDiversity.coerceIn(0.0, 1.0)
        _denseMotionBoundaryConfidence.value =
            dense.motionBoundaryConfidence.coerceIn(0.0, 1.0)
        _denseAppearanceConsistency.value =
            dense.appearanceConsistency.coerceIn(0.0, 1.0)
        _denseDepthConfidence.value =
            dense.depthConfidence.coerceIn(0.0, 1.0)
        _denseTemporalInnovationP95.value =
            dense.temporalInnovationP95.coerceIn(0.0, 1.0)
        _denseLandmarkContactCoverage.value =
            dense.contactCoverage.coerceIn(0.0, 1.0)
        _denseLandmarkPredictionBridgeRatio.value =
            dense.predictionBridgeRatio.coerceIn(0.0, 1.0)
        _denseLandmarkRejectedObservations.value =
            dense.rejectedObservations
        _denseLandmarkStaleObservations.value =
            dense.staleObservations
        _denseLandmarkNonFiniteObservations.value =
            dense.nonFiniteObservations
        _denseLandmarkHardFailure.value =
            dense.hardFailure
        _denseGeometricConfidence.value =
            dense.geometricConfidence.coerceIn(0.0, 1.0)
        _denseViewDiversity.value =
            dense.viewDiversity.coerceIn(0.0, 1.0)
        _denseMotionBoundaryConfidence.value =
            dense.motionBoundaryConfidence.coerceIn(0.0, 1.0)
        _denseAppearanceConsistency.value =
            dense.appearanceConsistency.coerceIn(0.0, 1.0)
        _denseDepthConfidence.value =
            dense.depthConfidence.coerceIn(0.0, 1.0)
        _denseTemporalInnovationP95.value =
            dense.temporalInnovationP95.coerceIn(0.0, 1.0)

        val customization = _motionCustomization.value
        denseLandmarkPrecisionContractEngine.configure(
            denseLandmarkContractConfig(
                customization = customization,
                requireCrossView =
                    connectedCameraCount >= 2,
            ),
        )

        val contract =
            denseLandmarkPrecisionContractEngine.evaluate(
                DenseLandmarkPrecisionContractEngine.Input(
                    coverage = dense.coverage,
                    visibleCoverage = dense.visibleCoverage,
                    uncertaintyP95 = dense.uncertaintyP95,
                    crossViewAgreement = dense.crossViewAgreement,
                    crossViewEvidenceAvailable =
                        dense.crossViewEvidenceAvailable,
                    supportingCameraCount =
                        dense.supportingCameraCount,
                    temporalStability = dense.temporalStability,
                    contactCoverage = dense.contactCoverage,
                    contactEvidenceActive =
                        _denseContactProtectedLandmarkCount.value > 0,
                    predictionBridgeRatio =
                        dense.predictionBridgeRatio,
                    acceptedObservations =
                        dense.acceptedObservations,
                    rejectedObservations =
                        dense.rejectedObservations,
                    staleObservations =
                        dense.staleObservations,
                    nonFiniteObservations =
                        dense.nonFiniteObservations,
                    hardFailure = dense.hardFailure,
                ),
            )

        _denseLandmarkContractScore.value =
            contract.score
        _denseLandmarkContractState.value =
            contract.state.name
        _denseLandmarkContractAuthoritative.value =
            contract.authoritative
        _denseLandmarkContractFailure.value =
            contract.failureClass.name
    }

    private fun denseEvidenceCameraId(
        deviceId: DeviceId,
    ): Int {
        if (deviceId == localId) return 0

        denseEvidenceCameraIds[deviceId]?.let { return it }

        synchronized(denseEvidenceCameraIds) {
            denseEvidenceCameraIds[deviceId]?.let { return it }

            val used =
                denseEvidenceCameraIds.values.toHashSet()

            for (candidate in 1..31) {
                if (candidate !in used) {
                    denseEvidenceCameraIds[deviceId] =
                        candidate
                    return candidate
                }
            }

            val fallback =
                1 + (abs(deviceId.value.hashCode()) % 31)

            denseEvidenceCameraIds[deviceId] =
                fallback
            return fallback
        }
    }

    private fun denseLandmarkContractConfig(
        customization: com.nivukx.nox.core.model.MotionCaptureCustomization,
        requireCrossView: Boolean,
    ): DenseLandmarkPrecisionContractEngine.Config {
        val coverage =
            customization.denseLandmarkCoverageStrictness.coerceIn(0.0, 1.0)
        val uncertainty =
            customization.denseLandmarkUncertaintyStrictness
                .coerceIn(0.0, 1.0)
        val crossView =
            customization.denseLandmarkCrossViewStrictness
                .coerceIn(0.0, 1.0)
        val temporal =
            customization.denseLandmarkTemporalStrictness
                .coerceIn(0.0, 1.0)

        val minimumCoverage = 0.55 + coverage * 0.30
        val minimumVisibleCoverage = 0.42 + coverage * 0.28
        val maximumUncertainty =
            (0.42 - uncertainty * 0.22).coerceIn(0.14, 0.42)
        val minimumCrossView =
            0.42 + crossView * 0.38
        val minimumTemporal =
            0.48 + temporal * 0.34
        val maximumRejected =
            (0.20 - uncertainty * 0.14).coerceIn(0.04, 0.20)
        val maximumPrediction =
            (0.55 - temporal * 0.30).coerceIn(0.14, 0.55)
        val evidenceScore =
            0.60 + minOf(
                coverage,
                uncertainty,
                crossView,
                temporal,
            ) * 0.20
        val authoritativeScore =
            0.72 + minOf(
                coverage,
                uncertainty,
                crossView,
                temporal,
            ) * 0.20

        return DenseLandmarkPrecisionContractEngine.Config(
            minimumCoverage = minimumCoverage,
            minimumVisibleCoverage = minimumVisibleCoverage,
            maximumUncertaintyP95 = maximumUncertainty,
            minimumCrossViewAgreement = minimumCrossView,
            minimumTemporalStability = minimumTemporal,
            maximumRejectedObservationRatio = maximumRejected,
            maximumPredictionBridgeRatio = maximumPrediction,
            minimumContactCoverageWhenActive =
                0.35 + customization.contactDetail * 0.25,
            minimumEvidenceScore =
                evidenceScore.coerceIn(0.60, 0.80),
            minimumAuthoritativeScore =
                authoritativeScore.coerceIn(0.72, 0.92),
            hardUncertaintyP95 =
                (maximumUncertainty + 0.44).coerceAtMost(0.94),
            hardCoverageFloor =
                (minimumCoverage - 0.48).coerceAtLeast(0.12),
            hardVisibleCoverageFloor =
                (minimumVisibleCoverage - 0.40).coerceAtLeast(0.10),
            hardCrossViewAgreement =
                (minimumCrossView * 0.36).coerceAtLeast(0.12),
            hardTemporalStability =
                (minimumTemporal * 0.36).coerceAtLeast(0.12),
            hardRejectedObservationRatio =
                (maximumRejected + 0.40).coerceAtMost(0.70),
            hardPredictionBridgeRatio =
                (maximumPrediction + 0.42).coerceAtMost(0.96),
            minimumCrossViewCameras =
                if (requireCrossView) 2 else 1,
        )
    }

    private fun applyMotionCustomization(
        customization: MotionCaptureCustomization,
        persist: Boolean = true,
    ) {
        val safe = customization.copyWith {
            targetFps =
                customization.targetFps.coerceIn(15, 240)
            inferenceFps =
                customization.inferenceFps.coerceIn(
                    5,
                    customization.targetFps.coerceIn(15, 240),
                )
            maxSubjects =
                customization.maxSubjects.coerceIn(1, 64)
            maximumOcclusionPredictionFrames =
                customization.maximumOcclusionPredictionFrames.coerceIn(
                    0,
                    60,
                )
        }

        _motionCustomization.value = safe

        fusion.configureUltraPrecisionPolicy(
            com.nivukx.nox.core.reconstruction.UltraPrecisionEvidenceEngine.Policy(
                enabled = safe.ultraPrecisionEnabled,
                qualificationMode =
                    safe.ultraPrecisionQualificationMode,
                targetP95Millimeters =
                    safe.ultraPrecisionTargetP95ErrorMillimeters,
                targetP99Millimeters =
                    safe.ultraPrecisionTargetP99ErrorMillimeters,
                minimumEvidenceSamples =
                    safe.ultraPrecisionMinimumEvidenceSamples,
                minimumBaselineMeters =
                    safe.ultraPrecisionMinimumBaselineMeters,
                minimumParallaxDegrees =
                    safe.ultraPrecisionMinimumParallaxDegrees,
                minimumCalibrationConfidence =
                    safe.ultraPrecisionMinimumCalibrationConfidence,
                minimumSynchronizationConfidence =
                    safe.ultraPrecisionMinimumSynchronizationConfidence,
                minimumViewDiversity =
                    safe.ultraPrecisionMinimumViewDiversity,
                minimumCrossViewAgreement =
                    safe.ultraPrecisionMinimumCrossViewAgreement,
                maximumOcclusionRatio =
                    safe.ultraPrecisionMaximumOcclusionRatio,
                minimumIdentityContinuity =
                    safe.ultraPrecisionMinimumIdentityContinuity,
                minimumSurfaceTemporalStability =
                    safe.ultraPrecisionMinimumSurfaceTemporalStability,
                maximumThermalRisk =
                    safe.ultraPrecisionMaximumThermalRisk,
                maximumMemoryPressure =
                    safe.ultraPrecisionMaximumMemoryPressure,
                matchedReferenceRequired =
                    safe.ultraPrecisionMatchedReferenceRequired,
            ),
        )

        motionFieldSynchronizer.configure(
            maxStaleNanos =
                (safe.liveMotionPreviewMaxStaleMillis * 1_000_000.0)
                    .toLong()
                    .coerceAtLeast(16_000_000L),
            maxPredictionNanos =
                (safe.liveMotionPreviewPredictionHorizonMillis * 1_000_000.0)
                    .toLong()
                    .coerceAtLeast(0L),
        )
        perception.configureCustomization(safe)
        eye3DTemporalConsistencyEngine.configureStrictness(
            safe.eyeTemporalStrictness,
        )
        handTemporalConsistencyEngine.configureStrictness(
            safe.handTemporalStrictness,
        )
        handGeometryQualityEngine.configureStrictness(
            (
                safe.handGeometryStrictness * 0.62 +
                    safe.handLongitudinalAnatomyStrictness * 0.38
                ).coerceIn(0.0, 1.0),
        )

        handUncertaintyCalibrationEngine.configure(
            HandUncertaintyCalibrationEngine.Config(
                targetCoverage =
                    (
                        0.91 +
                            safe.handLongitudinalAnatomyStrictness * 0.07
                        ).coerceIn(0.91, 0.98),
                minimumCalibrationSamples =
                    (5 + (safe.handGeometryStrictness * 4.0).toInt())
                        .coerceIn(5, 9),
                maximumResidualSamples =
                    (64 + (safe.temporalSmoothing * 64.0).toInt())
                        .coerceIn(64, 128),
                maximumRadiusMeters =
                    (
                        0.20 -
                            safe.outputIntegrityStrictness * 0.05
                        ).coerceIn(0.15, 0.20),
                warningRadiusMeters =
                    (
                        0.075 -
                            safe.handLongitudinalAnatomyStrictness * 0.018
                        ).coerceIn(0.055, 0.075),
                oodThreshold =
                    (
                        0.74 +
                            safe.outputIntegrityStrictness * 0.14
                        ).coerceIn(0.74, 0.88),
                holdOodThreshold =
                    (
                        0.54 +
                            safe.outputIntegrityStrictness * 0.08
                        ).coerceIn(0.54, 0.66),
                historyAlpha =
                    (
                        0.16 +
                            safe.temporalSmoothing * 0.08
                        ).coerceIn(0.16, 0.24),
                dynamicInflationWeight =
                    (
                        0.12 +
                            safe.semanticMotionSensitivity * 0.08
                        ).coerceIn(0.12, 0.24),
                occlusionInflationWeight =
                    (
                        0.18 +
                            safe.maximumOcclusionPredictionFrames
                                .coerceIn(0, 60) * 0.006
                        ).coerceIn(0.18, 0.42),
            ),
        )
        fusion.configureAdaptiveEvidenceTuning(
            com.nivukx.nox.core.reconstruction.AdaptiveMultiViewEvidenceWeightEngine.Tuning(
                adaptivity =
                    safe.multiViewEvidenceAdaptivity.coerceIn(0.0, 1.0),
                minimumViewTrust =
                    (0.10 + safe.multiViewEvidenceAdaptivity * 0.25)
                        .coerceIn(0.10, 0.40),
                staleRecovery =
                    (0.55 + safe.occlusionRecoveryConfidence * 0.35)
                        .coerceIn(0.55, 0.90),
                disagreementSensitivity =
                    (0.45 + safe.landmarkConsistencyStrictness * 0.50)
                        .coerceIn(0.45, 0.95),
            ),
        )
        fusion.configureSubframeAlignmentStrictness(
            safe.subframeAlignmentStrictness,
        )
        fusion.configureCameraContributionTuning(
            com.nivukx.nox.core.reconstruction.MultiViewCameraContributionPlanner.Config(
                minimumCameras = 2,
                maximumCameras =
                    (4 + (safe.bodyDetail * 4.0).toInt())
                        .coerceIn(4, 8),
                reliabilityWeight =
                    (0.46 + safe.multiViewEvidenceAdaptivity * 0.10)
                        .coerceIn(0.46, 0.58),
                diversityWeight =
                    (0.24 + safe.depthDetail * 0.14)
                        .coerceIn(0.24, 0.38),
                coverageWeight =
                    (
                        1.0 -
                            (0.46 + safe.multiViewEvidenceAdaptivity * 0.10)
                                .coerceIn(0.46, 0.58) -
                            (0.24 + safe.depthDetail * 0.14)
                                .coerceIn(0.24, 0.38)
                        ).coerceIn(0.08, 0.30),
                holdStateBonus =
                    -(
                        0.18 +
                            safe.outputIntegrityStrictness * 0.18
                        ).coerceIn(0.18, 0.36),
                degradedStateBonus =
                    -(
                        0.06 +
                            safe.multiViewEvidenceAdaptivity * 0.10
                        ).coerceIn(0.06, 0.16),
                minimumSelectionScore =
                    (0.20 + safe.outputIntegrityStrictness * 0.14)
                        .coerceIn(0.20, 0.34),
            ),
        )
        fusion.configureJointObservationTuning(
            com.nivukx.nox.core.reconstruction.JointObservationSubsetPlanner.Config(
                minimumObservations = 2,
                maximumObservations =
                    (4 + (safe.bodyDetail * 2.0).toInt())
                        .coerceIn(4, 6),
                minimumConfidence =
                    (0.05 + safe.outputIntegrityStrictness * 0.08)
                        .coerceIn(0.05, 0.18),
                reliabilityWeight =
                    (0.42 + safe.multiViewEvidenceAdaptivity * 0.12)
                        .coerceIn(0.42, 0.54),
                diversityWeight =
                    (0.28 + safe.depthDetail * 0.12)
                        .coerceIn(0.28, 0.40),
                timingWeight =
                    (
                        1.0 -
                            (0.42 + safe.multiViewEvidenceAdaptivity * 0.12)
                                .coerceIn(0.42, 0.54) -
                            (0.28 + safe.depthDetail * 0.12)
                                .coerceIn(0.28, 0.40)
                        ).coerceIn(0.06, 0.30),
                minimumSelectionScore =
                    (0.22 + safe.outputIntegrityStrictness * 0.14)
                        .coerceIn(0.22, 0.36),
                minimumDiversityScore =
                    (0.12 + safe.topologyConstraintStrength * 0.16)
                        .coerceIn(0.12, 0.28),
            ),
        )
        fusion.configureObservability(
            com.nivukx.nox.core.reconstruction.SkeletonReconstructionEngine.ObservabilityTuning(
                maximumTemporalSkewMillis = safe.maximumTemporalSkewMillis,
                preferredTemporalSkewMillis = safe.preferredTemporalSkewMillis,
                minimumParallaxDegrees = safe.minimumParallaxDegrees,
                maximumReprojectionErrorPx = safe.maximumReprojectionErrorPx,
                maximumUncertaintyMeters =
                    safe.maximumReconstructionUncertaintyMeters,
                rollingShutterCompensationTrust =
                    safe.rollingShutterCompensation,
            ),
        )

        monocularTrajectoryOptimizer.configure(
            KinematicTrajectoryOptimizer.Tuning(
                predictionGateMeters =
                    (0.06 + safe.maximumReconstructionUncertaintyMeters * 0.22)
                        .coerceIn(0.06, 0.42),
                microMotionPreservation = safe.microMotionPreservation,
                temporalSmoothing = safe.temporalSmoothing,
                semanticMotionSensitivity = safe.semanticMotionSensitivity,
                maximumVelocityMetersPerSecond =
                    (4.0 + safe.semanticMotionSensitivity * 14.0)
                        .coerceIn(4.0, 18.0),
                maximumAccelerationMetersPerSecond2 =
                    (22.0 + safe.semanticMotionSensitivity * 70.0)
                        .coerceIn(22.0, 92.0),
                maximumJerkMetersPerSecond3 =
                    (120.0 + safe.semanticMotionSensitivity * 420.0)
                        .coerceIn(120.0, 540.0),
                innovationHardRatio =
                    (2.0 + safe.microMotionPreservation * 1.2)
                        .coerceIn(2.0, 3.2),
                confidenceRetention =
                    (0.58 + safe.outputIntegrityStrictness * 0.30)
                        .coerceIn(0.58, 0.90),
            ),
        )
        fusion.configureKinematicTuning(
            HighEndKinematicConstraintEngine.Config(
                correctionStrength =
                    safe.landmarkConsistencyStrictness.coerceIn(0.0, 1.0),
                rootMotionLock =
                    safe.rootMotionLock.coerceIn(0.0, 1.0),
                scalePreservation =
                    safe.scalePreservation.coerceIn(0.0, 1.0),
                footContactStrictness =
                    safe.footContactStrictness.coerceIn(0.0, 1.0),
                posePriorStrength =
                    safe.posePriorStrength.coerceIn(0.0, 1.0),
                microMotionPreservation =
                    safe.microMotionPreservation.coerceIn(0.0, 1.0),
                maximumVelocityMetersPerSecond =
                    (4.0 + safe.semanticMotionSensitivity * 14.0)
                        .coerceIn(4.0, 18.0),
                maximumAccelerationMetersPerSecond2 =
                    (22.0 + safe.semanticMotionSensitivity * 70.0)
                        .coerceIn(22.0, 92.0),
                maximumJerkMetersPerSecond3 =
                    (120.0 + safe.semanticMotionSensitivity * 420.0)
                        .coerceIn(120.0, 540.0),
                rejectionFloor =
                    (0.20 + safe.outputIntegrityStrictness * 0.16)
                        .coerceIn(0.20, 0.36),
            ),
        )

        val physicsStrictness =
            (
                safe.physicsConstraintStrictness * 0.68 +
                    safe.landmarkConsistencyStrictness * 0.32
                ).coerceIn(0.0, 1.0)

        val global4DStrictness =
            (
                safe.global4DConsistencyStrictness * 0.72 +
                    safe.outputIntegrityStrictness * 0.28
                ).coerceIn(0.0, 1.0)

        val physicsTuning =
            PhysicsConstrainedSkeletonEngine.Config(
                iterations =
                    (4 + safe.reconstructionEvidenceFusionStrength * 4.0)
                        .toInt()
                        .coerceIn(4, 8),
                learningRate =
                    (
                        0.08 +
                            safe.landmarkConsistencyStrictness * 0.14
                        ).coerceIn(0.08, 0.22),
                projectionStrength =
                    (
                        0.46 +
                            physicsStrictness * 0.44
                        ).coerceIn(0.46, 0.90),
                maxCorrectionFraction =
                    (
                        0.18 +
                            safe.microMotionPreservation * 0.12
                        ).coerceIn(0.18, 0.30),
                minimumJointConfidence =
                    (
                        0.24 +
                            (
                                physicsStrictness * 0.10 +
                                    safe.outputIntegrityStrictness * 0.06
                                )
                        ).coerceIn(0.24, 0.40),
                maximumBoneResidualMeters =
                    (
                        0.050 +
                            safe.maximumReconstructionUncertaintyMeters * 0.08
                        ).coerceIn(0.050, 0.80),
                warningBoneResidualMeters =
                    (
                        0.028 +
                            safe.maximumReconstructionUncertaintyMeters * 0.04
                        ).coerceIn(0.028, 0.35),
                baselineLearningRate =
                    (
                        0.085 -
                            physicsStrictness * 0.055 +
                            safe.scalePreservation * 0.025
                        ).coerceIn(0.025, 0.11),
                maximumVelocityMetersPerSecond =
                    (4.0 + safe.semanticMotionSensitivity * 14.0)
                        .coerceIn(4.0, 18.0),
                warningVelocityMetersPerSecond =
                    (2.5 + safe.semanticMotionSensitivity * 7.0)
                        .coerceIn(2.5, 9.5),
                maximumAccelerationMetersPerSecond2 =
                    (22.0 + safe.semanticMotionSensitivity * 70.0)
                        .coerceIn(22.0, 92.0),
                warningAccelerationMetersPerSecond2 =
                    (14.0 + safe.semanticMotionSensitivity * 42.0)
                        .coerceIn(14.0, 56.0),
                maximumJerkMetersPerSecond3 =
                    (120.0 + safe.semanticMotionSensitivity * 420.0)
                        .coerceIn(120.0, 540.0),
                warningJerkMetersPerSecond3 =
                    (80.0 + safe.semanticMotionSensitivity * 260.0)
                        .coerceIn(80.0, 340.0),
                symmetryToleranceRatio =
                    (
                        0.32 -
                            physicsStrictness * 0.11
                        ).coerceIn(0.18, 0.32),
                blockedPhysicsScore =
                    (
                        0.30 +
                            physicsStrictness * 0.24 +
                            safe.outputIntegrityStrictness * 0.06
                        ).coerceIn(0.30, 0.58),
                warningPhysicsScore =
                    (
                        0.56 +
                            physicsStrictness * 0.22 +
                            safe.outputIntegrityStrictness * 0.06
                        ).coerceIn(0.56, 0.86),
                historyAlpha =
                    (
                        0.14 +
                            safe.temporalSmoothing * 0.12
                        ).coerceIn(0.14, 0.26),
                recoveryFrames =
                    (4 + (safe.outputIntegrityStrictness * 4.0).toInt())
                        .coerceIn(4, 8),
            )

        fusion.configurePhysicsConstraints(
            tuning = physicsTuning,
        )
        fusion.configureGlobal4DConsistency(
            Global4DConsistencyEngine.Config(
                preferredRootConfidence =
                    (
                        0.62 +
                            global4DStrictness * 0.22
                        ).coerceIn(0.62, 0.84),
                maximumRootSpeedMetersPerSecond =
                    (
                        6.0 +
                            safe.semanticMotionSensitivity * 5.0
                        ).coerceIn(6.0, 14.0),
                maximumRootJerkMetersPerSecond3 =
                    (
                        280.0 +
                            safe.semanticMotionSensitivity * 220.0
                        ).coerceIn(280.0, 500.0),
                rootVelocityAgreementWeight =
                    (
                        0.05 +
                            global4DStrictness * 0.11
                        ).coerceIn(0.05, 0.16),
                warningScore =
                    (
                        0.58 +
                            global4DStrictness * 0.22
                        ).coerceIn(0.58, 0.80),
                failureScore =
                    (
                        0.34 +
                            global4DStrictness * 0.18
                        ).coerceIn(0.34, 0.52),
                persistentFailureFrames =
                    (
                        3 +
                            global4DStrictness * 2.0
                        ).toInt().coerceIn(3, 5),
                recoveryFrames =
                    (
                        4 +
                            global4DStrictness * 3.0
                        ).toInt().coerceIn(4, 7),
                historyAlpha =
                    (
                        0.16 +
                            safe.temporalSmoothing * 0.10
                        ).coerceIn(0.16, 0.26),
            ),
        )
        fusion.configureWorldSpaceMotionConsistency(
            WorldSpaceMotionConsistencyEngine.Config(
                baseInnovationGateMeters =
                    (
                        0.060 -
                            safe.global4DConsistencyStrictness * 0.018
                        ).coerceIn(0.042, 0.060),
                maximumInnovationGateMeters =
                    (
                        0.36 -
                            safe.outputIntegrityStrictness * 0.08
                        ).coerceIn(0.24, 0.36),
                velocityBlend =
                    (
                        0.60 +
                            safe.temporalSmoothing * 0.18
                        ).coerceIn(0.60, 0.78),
                accelerationBlend =
                    (
                        0.46 +
                            safe.global4DConsistencyStrictness * 0.18
                        ).coerceIn(0.46, 0.64),
                cameraRiskPenaltyWeight =
                    (
                        0.20 +
                            safe.cameraMotionSuppression * 0.16
                        ).coerceIn(0.20, 0.36),
                uncertaintyPenaltyWeight =
                    (
                        0.24 +
                            safe.outputIntegrityStrictness * 0.18
                        ).coerceIn(0.24, 0.42),
                topologyCorrectionGain =
                    (
                        0.38 +
                            safe.topologyConstraintStrength * 0.34
                        ).coerceIn(0.38, 0.72),
                topologyIterations =
                    (5 + (safe.topologyConstraintStrength * 6.0).toInt())
                        .coerceIn(5, 11),
                maximumAccelerationMetersPerSecond2 =
                    (
                        46.0 +
                            safe.semanticMotionSensitivity * 48.0
                        ).coerceIn(46.0, 94.0),
                maximumJerkMetersPerSecond3 =
                    (
                        300.0 +
                            safe.semanticMotionSensitivity * 300.0
                        ).coerceIn(300.0, 600.0),
                microMotionPreservation =
                    safe.microMotionPreservation.coerceIn(0.0, 1.0),
                hardOutlierInnovationMeters =
                    (
                        0.28 -
                            safe.outputIntegrityStrictness * 0.04
                        ).coerceIn(0.20, 0.28),
                robustOutlierSigma =
                    (
                        3.8 -
                            safe.outputIntegrityStrictness * 0.8
                        ).coerceIn(2.6, 3.8),
                minimumJointSupportRatio =
                    (
                        0.30 +
                            safe.landmarkConsistencyStrictness * 0.22
                        ).coerceIn(0.30, 0.52),
            ),
        )
        monocularPhysicsConstraintEngine.configure(
            newConfig = physicsTuning,
        )

        monocularPipeline.configureCapturePolicy(
            maximumOcclusionPredictionFrames =
                safe.maximumOcclusionPredictionFrames,
            captureFps =
                safe.targetFps.coerceIn(15, 240),
        )

        if (persist) {
            controlPreferences.edit()
                .putString(
                    "custom_camera_resolution_preset",
                    safe.cameraResolutionPreset.name,
                )
                .putString(
                    "custom_camera_exposure_mode",
                    safe.cameraExposureMode.name,
                )
                .putString(
                    "custom_camera_focus_mode",
                    safe.cameraFocusMode.name,
                )
                .putLong(
                    "custom_camera_manual_exposure_micros",
                    safe.cameraManualExposureMicros,
                )
                .putInt(
                    "custom_camera_manual_iso",
                    safe.cameraManualIso,
                )
                .putFloat(
                    "custom_camera_exposure_compensation_ev",
                    safe.cameraExposureCompensationEv.toFloat(),
                )
                .putBoolean(
                    "custom_camera_ae_lock",
                    safe.cameraAutoExposureLock,
                )
                .putBoolean(
                    "custom_camera_awb_lock",
                    safe.cameraAutoWhiteBalanceLock,
                )
                .putString(
                    "custom_camera_manual_focus_diopters",
                    safe.cameraManualFocusDistanceDiopters?.toString(),
                )
                .putBoolean(
                    "custom_camera_ois",
                    safe.cameraOisEnabled,
                )
                .putBoolean(
                    "custom_camera_eis",
                    safe.cameraElectronicStabilizationEnabled,
                )
                .putBoolean(
                    "custom_live_motion_preview_enabled",
                    safe.liveMotionPreviewEnabled,
                )
                .putString(
                    "custom_live_motion_preview_gpu_mode",
                    safe.liveMotionPreviewGpuMode.name,
                )
                .putBoolean(
                    "custom_camera_raw_sidecar",
                    safe.cameraRawSidecarEnabled,
                )
                .putBoolean(
                    "custom_camera_depth_sidecar",
                    safe.cameraDepthSidecarEnabled,
                )
                .putBoolean(
                    "custom_camera_sensor_timestamp",
                    safe.cameraPreserveSensorTimestamp,
                )
                .putInt("custom_target_fps", safe.targetFps)
                .putInt("custom_inference_fps", safe.inferenceFps)
                .putInt("custom_max_subjects", safe.maxSubjects)
                .putString("custom_actor_detail_priority_mode", safe.actorDetailPriorityMode.name)
                .putInt("custom_full_detail_actor_budget", safe.fullDetailActorBudget)
                .putInt("custom_face_detail_actor_budget", safe.faceDetailActorBudget)
                .putInt("custom_eye_detail_actor_budget", safe.eyeDetailActorBudget)
                .putInt("custom_hand_detail_actor_budget", safe.handDetailActorBudget)
                .putInt("custom_dense_surface_actor_budget", safe.denseSurfaceActorBudget)
                .putFloat("custom_priority_detail_retention", safe.priorityDetailRetention.toFloat())
                .putString("custom_capture_evidence_mode", safe.captureEvidenceMode.name)
                .putBoolean(
                    "custom_post_capture_auto_reconstruction",
                    safe.postCaptureAutoReconstructionEnabled,
                )
                .putFloat("custom_body_detail", safe.bodyDetail.toFloat())
                .putFloat("custom_face_detail", safe.faceDetail.toFloat())
                .putFloat("custom_eye_detail", safe.eyeDetail.toFloat())
                .putFloat("custom_hand_detail", safe.handDetail.toFloat())
                .putFloat("custom_depth_detail", safe.depthDetail.toFloat())
                .putFloat(
                    "custom_dense_mesh_detail",
                    safe.denseMeshDetail.toFloat(),
                )
                .putFloat(
                    "custom_contact_detail",
                    safe.contactDetail.toFloat(),
                )
                .putFloat(
                    "custom_world_consistency_detail",
                    safe.worldConsistencyDetail.toFloat(),
                )
                .putFloat(
                    "custom_multi_view_evidence_adaptivity",
                    safe.multiViewEvidenceAdaptivity.toFloat(),
                )
                .putFloat(
                    "custom_reconstruction_evidence_fusion_strength",
                    safe.reconstructionEvidenceFusionStrength.toFloat(),
                )
                .putFloat(
                    "custom_reconstruction_evidence_disagreement_tolerance_meters",
                    safe.reconstructionEvidenceDisagreementToleranceMeters.toFloat(),
                )
                .putFloat(
                    "custom_reconstruction_evidence_source_diversity",
                    safe.reconstructionEvidenceMinimumSourceDiversity.toFloat(),
                )
                .putFloat(
                    "custom_topology_constraint_strength",
                    safe.topologyConstraintStrength.toFloat(),
                )
                .putFloat(
                    "custom_micro_motion_preservation",
                    safe.microMotionPreservation.toFloat(),
                )
                .putFloat(
                    "custom_camera_motion_suppression",
                    safe.cameraMotionSuppression.toFloat(),
                )
                .putFloat(
                    "custom_face_eye_fusion_weight",
                    safe.faceEyeFusionWeight.toFloat(),
                )
                .putFloat(
                    "custom_eye_temporal_strictness",
                    safe.eyeTemporalStrictness.toFloat(),
                )
                .putFloat(
                    "custom_hand_temporal_strictness",
                    safe.handTemporalStrictness.toFloat(),
                )
                .putFloat(
                    "custom_hand_geometry_strictness",
                    safe.handGeometryStrictness.toFloat(),
                )
                .putFloat(
                    "custom_temporal_alignment_strictness",
                    safe.temporalAlignmentStrictness.toFloat(),
                )
                .putFloat(
                    "custom_subframe_alignment_strictness",
                    safe.subframeAlignmentStrictness.toFloat(),
                )
                .putFloat(
                    "custom_rolling_shutter_compensation",
                    safe.rollingShutterCompensation.toFloat(),
                )
                .putFloat(
                    "custom_output_integrity_strictness",
                    safe.outputIntegrityStrictness.toFloat(),
                )
                .putBoolean(
                    "custom_ultra_precision_enabled",
                    safe.ultraPrecisionEnabled,
                )
                .putBoolean(
                    "custom_ultra_precision_qualification_mode",
                    safe.ultraPrecisionQualificationMode,
                )
                .putFloat(
                    "custom_ultra_precision_target_p95_mm",
                    safe.ultraPrecisionTargetP95ErrorMillimeters.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_target_p99_mm",
                    safe.ultraPrecisionTargetP99ErrorMillimeters.toFloat(),
                )
                .putInt(
                    "custom_ultra_precision_min_samples",
                    safe.ultraPrecisionMinimumEvidenceSamples,
                )
                .putFloat(
                    "custom_ultra_precision_min_baseline_m",
                    safe.ultraPrecisionMinimumBaselineMeters.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_parallax_deg",
                    safe.ultraPrecisionMinimumParallaxDegrees.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_calibration_confidence",
                    safe.ultraPrecisionMinimumCalibrationConfidence.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_sync_confidence",
                    safe.ultraPrecisionMinimumSynchronizationConfidence.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_view_diversity",
                    safe.ultraPrecisionMinimumViewDiversity.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_cross_view_agreement",
                    safe.ultraPrecisionMinimumCrossViewAgreement.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_max_occlusion_ratio",
                    safe.ultraPrecisionMaximumOcclusionRatio.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_identity",
                    safe.ultraPrecisionMinimumIdentityContinuity.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_min_surface_stability",
                    safe.ultraPrecisionMinimumSurfaceTemporalStability.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_max_thermal_risk",
                    safe.ultraPrecisionMaximumThermalRisk.toFloat(),
                )
                .putFloat(
                    "custom_ultra_precision_max_memory_pressure",
                    safe.ultraPrecisionMaximumMemoryPressure.toFloat(),
                )
                .putBoolean(
                    "custom_ultra_precision_reference_required",
                    safe.ultraPrecisionMatchedReferenceRequired,
                )
                .putFloat(
                    "custom_minimum_parallax_degrees",
                    safe.minimumParallaxDegrees.toFloat(),
                )
                .putFloat(
                    "custom_preferred_temporal_skew_millis",
                    safe.preferredTemporalSkewMillis.toFloat(),
                )
                .putFloat(
                    "custom_maximum_temporal_skew_millis",
                    safe.maximumTemporalSkewMillis.toFloat(),
                )
                .putFloat(
                    "custom_maximum_reprojection_error_px",
                    safe.maximumReprojectionErrorPx.toFloat(),
                )
                .putFloat(
                    "custom_maximum_reconstruction_uncertainty_meters",
                    safe.maximumReconstructionUncertaintyMeters.toFloat(),
                )
                .putFloat(
                    "custom_pose_prior_strength",
                    safe.posePriorStrength.toFloat(),
                )
                .putFloat(
                    "custom_foot_contact_strictness",
                    safe.footContactStrictness.toFloat(),
                )
                .putFloat(
                    "custom_head_pose_strictness",
                    safe.headPoseStrictness.toFloat(),
                )
                .putFloat(
                    "custom_gaze_priority",
                    safe.gazePriority.toFloat(),
                )
                .putFloat(
                    "custom_depth_temporal_fusion",
                    safe.depthTemporalFusion.toFloat(),
                )
                .putFloat(
                    "custom_camera_drift_response",
                    safe.cameraDriftResponse.toFloat(),
                )
                .putFloat(
                    "custom_thermal_quality_floor",
                    safe.thermalQualityFloor.toFloat(),
                )
                .putFloat(
                    "custom_network_jitter_tolerance_ms",
                    safe.networkJitterToleranceMillis.toFloat(),
                )
                .putFloat(
                    "custom_mesh_adaptive_detail",
                    safe.meshAdaptiveDetail.toFloat(),
                )
                .putFloat(
                    "custom_retargeting_strictness",
                    safe.retargetingStrictness.toFloat(),
                )
                .putFloat(
                    "custom_scale_preservation",
                    safe.scalePreservation.toFloat(),
                )
                .putFloat(
                    "custom_occlusion_recovery_confidence",
                    safe.occlusionRecoveryConfidence.toFloat(),
                )
                .putFloat(
                    "custom_landmark_consistency_strictness",
                    safe.landmarkConsistencyStrictness.toFloat(),
                )
                .putFloat(
                    "custom_semantic_motion_sensitivity",
                    safe.semanticMotionSensitivity.toFloat(),
                )
                .putFloat(
                    "custom_root_motion_lock",
                    safe.rootMotionLock.toFloat(),
                )
                .putInt(
                    "custom_occlusion_prediction_frames",
                    safe.maximumOcclusionPredictionFrames,
                )
                .apply()
        }
    }

    private fun applySessionPresentationSettings(
        settings: SessionPresentationSettings,
        persist: Boolean = true,
    ) {
        val safe = settings.copy(
            clipName =
                settings.clipName
                    .trim()
                    .replace(Regex("[^A-Za-z0-9._ -]"), "_")
                    .take(80)
                    .ifBlank { "Take" },
            countdownSeconds =
                settings.countdownSeconds.coerceIn(0, 10),
        )

        _sessionPresentationSettings.value = safe

        if (persist) {
            controlPreferences.edit()
                .putString("session_clip_name", safe.clipName)
                .putInt(
                    "session_countdown_seconds",
                    safe.countdownSeconds,
                )
                .putBoolean(
                    "session_require_calibration",
                    safe.requireCalibrationBeforeRecord,
                )
                .putBoolean(
                    "session_auto_index",
                    safe.automaticallyIndexAfterStop,
                )
                .putBoolean(
                    "session_show_preflight_warnings",
                    safe.showPreflightWarnings,
                )
                .apply()
        }
    }

    private fun loadSessionPresentationSettings():
        SessionPresentationSettings =
        SessionPresentationSettings(
            clipName =
                controlPreferences.getString(
                    "session_clip_name",
                    "Take",
                ) ?: "Take",
            countdownSeconds =
                controlPreferences.getInt(
                    "session_countdown_seconds",
                    3,
                ).coerceIn(0, 10),
            requireCalibrationBeforeRecord =
                controlPreferences.getBoolean(
                    "session_require_calibration",
                    false,
                ),
            automaticallyIndexAfterStop =
                controlPreferences.getBoolean(
                    "session_auto_index",
                    true,
                ),
            showPreflightWarnings =
                controlPreferences.getBoolean(
                    "session_show_preflight_warnings",
                    true,
                ),
        )

    private fun loadMotionCustomization(): MotionCaptureCustomization =
        MotionCaptureCustomization(
            cameraResolutionPreset =
                runCatching {
                    com.nivukx.nox.core.model.CameraResolutionPreset.valueOf(
                        controlPreferences.getString(
                            "custom_camera_resolution_preset",
                            "AUTO",
                        ) ?: "AUTO",
                    )
                }.getOrDefault(
                    com.nivukx.nox.core.model.CameraResolutionPreset.AUTO,
                ),
            cameraExposureMode =
                runCatching {
                    com.nivukx.nox.core.model.CameraExposureMode.valueOf(
                        controlPreferences.getString(
                            "custom_camera_exposure_mode",
                            "AUTO",
                        ) ?: "AUTO",
                    )
                }.getOrDefault(
                    com.nivukx.nox.core.model.CameraExposureMode.AUTO,
                ),
            cameraFocusMode =
                runCatching {
                    com.nivukx.nox.core.model.CameraFocusMode.valueOf(
                        controlPreferences.getString(
                            "custom_camera_focus_mode",
                            "TRACKED_SUBJECT",
                        ) ?: "TRACKED_SUBJECT",
                    )
                }.getOrDefault(
                    com.nivukx.nox.core.model.CameraFocusMode.TRACKED_SUBJECT,
                ),
            cameraManualExposureMicros =
                controlPreferences.getLong(
                    "custom_camera_manual_exposure_micros",
                    8333L,
                ).coerceIn(1L, 500_000L),
            cameraManualIso =
                controlPreferences.getInt(
                    "custom_camera_manual_iso",
                    800,
                ).coerceIn(25, 12_800),
            cameraExposureCompensationEv =
                controlPreferences.getFloat(
                    "custom_camera_exposure_compensation_ev",
                    0.0f,
                ).toDouble().coerceIn(-8.0, 8.0),
            cameraAutoExposureLock =
                controlPreferences.getBoolean(
                    "custom_camera_ae_lock",
                    false,
                ),
            cameraAutoWhiteBalanceLock =
                controlPreferences.getBoolean(
                    "custom_camera_awb_lock",
                    false,
                ),
            cameraManualFocusDistanceDiopters =
                controlPreferences.getString(
                    "custom_camera_manual_focus_diopters",
                    null,
                )?.toDoubleOrNull()?.takeIf {
                    it.isFinite() && it >= 0.0
                },
            cameraOisEnabled =
                controlPreferences.getBoolean(
                    "custom_camera_ois",
                    false,
                ),
            cameraElectronicStabilizationEnabled =
                controlPreferences.getBoolean(
                    "custom_camera_eis",
                    false,
                ),
            liveMotionPreviewEnabled =
                controlPreferences.getBoolean(
                    "custom_live_motion_preview_enabled",
                    false,
                ),
            liveMotionPreviewGpuMode =
                runCatching {
                    MotionFieldGpuMode.valueOf(
                        controlPreferences.getString(
                            "custom_live_motion_preview_gpu_mode",
                            "AUTO",
                        ) ?: "AUTO",
                    )
                }.getOrDefault(MotionFieldGpuMode.AUTO),
            cameraRawSidecarEnabled =
                controlPreferences.getBoolean(
                    "custom_camera_raw_sidecar",
                    false,
                ),
            cameraDepthSidecarEnabled =
                controlPreferences.getBoolean(
                    "custom_camera_depth_sidecar",
                    true,
                ),
            cameraPreserveSensorTimestamp =
                controlPreferences.getBoolean(
                    "custom_camera_sensor_timestamp",
                    true,
                ),
            targetFps = controlPreferences.getInt("custom_target_fps", 60),
            inferenceFps = controlPreferences.getInt("custom_inference_fps", 30),
            maxSubjects =
                controlPreferences.getInt("custom_max_subjects", 50)
                    .coerceIn(1, 64),
            actorDetailPriorityMode = runCatching {
                com.nivukx.nox.core.model.ActorDetailPriorityMode.valueOf(
                    controlPreferences.getString(
                        "custom_actor_detail_priority_mode",
                        com.nivukx.nox.core.model.ActorDetailPriorityMode.BALANCED.name,
                    ) ?: com.nivukx.nox.core.model.ActorDetailPriorityMode.BALANCED.name,
                )
            }.getOrDefault(
                com.nivukx.nox.core.model.ActorDetailPriorityMode.BALANCED,
            ),
            fullDetailActorBudget =
                controlPreferences.getInt("custom_full_detail_actor_budget", 50)
                    .coerceIn(1, 64),
            faceDetailActorBudget =
                controlPreferences.getInt("custom_face_detail_actor_budget", 50)
                    .coerceIn(1, 64),
            eyeDetailActorBudget =
                controlPreferences.getInt("custom_eye_detail_actor_budget", 50)
                    .coerceIn(1, 64),
            handDetailActorBudget =
                controlPreferences.getInt("custom_hand_detail_actor_budget", 50)
                    .coerceIn(1, 64),
            denseSurfaceActorBudget =
                controlPreferences.getInt("custom_dense_surface_actor_budget", 50)
                    .coerceIn(1, 64),
            priorityDetailRetention =
                controlPreferences.getFloat(
                    "custom_priority_detail_retention",
                    0.94f,
                ).toDouble().coerceIn(0.0, 1.0),
            captureEvidenceMode = runCatching {
                com.nivukx.nox.core.model.CaptureEvidenceMode.valueOf(
                    controlPreferences.getString(
                        "custom_capture_evidence_mode",
                        com.nivukx.nox.core.model.CaptureEvidenceMode.LIVE_MOTION_ONLY.name,
                    ) ?: com.nivukx.nox.core.model.CaptureEvidenceMode.LIVE_MOTION_ONLY.name,
                )
            }.getOrDefault(
                com.nivukx.nox.core.model.CaptureEvidenceMode.LIVE_MOTION_ONLY,
            ),
            recordedVideoEvidenceProfile =
                com.nivukx.nox.core.model.RecordedVideoEvidenceProfile.liveUltra(),
            postCaptureAutoReconstructionEnabled = controlPreferences.getBoolean(
                "custom_post_capture_auto_reconstruction",
                true,
            ),
            bodyDetail = controlPreferences.getFloat(
                "custom_body_detail", 0.90f,
            ).toDouble(),
            faceDetail = controlPreferences.getFloat(
                "custom_face_detail", 0.95f,
            ).toDouble(),
            eyeDetail = controlPreferences.getFloat(
                "custom_eye_detail", 0.92f,
            ).toDouble(),
            handDetail = controlPreferences.getFloat(
                "custom_hand_detail", 0.92f,
            ).toDouble(),
            depthDetail = controlPreferences.getFloat(
                "custom_depth_detail", 0.88f,
            ).toDouble(),
            denseMeshDetail = controlPreferences.getFloat(
                "custom_dense_mesh_detail", 0.74f,
            ).toDouble(),
            contactDetail = controlPreferences.getFloat(
                "custom_contact_detail", 0.76f,
            ).toDouble(),
            maximumOcclusionPredictionFrames =
                controlPreferences.getInt(
                    "custom_occlusion_prediction_frames",
                    8,
                ).coerceIn(0, 60),
            worldConsistencyDetail =
                controlPreferences.getFloat(
                    "custom_world_consistency_detail",
                    0.86f,
                ).toDouble(),
            multiViewEvidenceAdaptivity =
                controlPreferences.getFloat(
                    "custom_multi_view_evidence_adaptivity",
                    0.84f,
                ).toDouble(),
            reconstructionEvidenceFusionStrength =
                controlPreferences.getFloat(
                    "custom_reconstruction_evidence_fusion_strength",
                    0.92f,
                ).toDouble(),
            reconstructionEvidenceDisagreementToleranceMeters =
                controlPreferences.getFloat(
                    "custom_reconstruction_evidence_disagreement_tolerance_meters",
                    0.12f,
                ).toDouble(),
            reconstructionEvidenceMinimumSourceDiversity =
                controlPreferences.getFloat(
                    "custom_reconstruction_evidence_source_diversity",
                    0.20f,
                ).toDouble(),
            topologyConstraintStrength =
                controlPreferences.getFloat(
                    "custom_topology_constraint_strength",
                    0.84f,
                ).toDouble(),
            microMotionPreservation =
                controlPreferences.getFloat(
                    "custom_micro_motion_preservation",
                    0.88f,
                ).toDouble(),
            cameraMotionSuppression =
                controlPreferences.getFloat(
                    "custom_camera_motion_suppression",
                    0.72f,
                ).toDouble(),
            faceEyeFusionWeight =
                controlPreferences.getFloat(
                    "custom_face_eye_fusion_weight",
                    0.62f,
                ).toDouble(),
            eyeTemporalStrictness =
                controlPreferences.getFloat(
                    "custom_eye_temporal_strictness",
                    0.78f,
                ).toDouble(),
            handTemporalStrictness =
                controlPreferences.getFloat(
                    "custom_hand_temporal_strictness",
                    0.78f,
                ).toDouble(),
            handGeometryStrictness =
                controlPreferences.getFloat(
                    "custom_hand_geometry_strictness",
                    0.84f,
                ).toDouble(),
            temporalAlignmentStrictness =
                controlPreferences.getFloat(
                    "custom_temporal_alignment_strictness",
                    0.78f,
                ).toDouble(),
            subframeAlignmentStrictness =
                controlPreferences.getFloat(
                    "custom_subframe_alignment_strictness",
                    0.82f,
                ).toDouble(),
            rollingShutterCompensation =
                controlPreferences.getFloat(
                    "custom_rolling_shutter_compensation",
                    0.88f,
                ).toDouble(),
            outputIntegrityStrictness =
                controlPreferences.getFloat(
                    "custom_output_integrity_strictness",
                    0.78f,
                ).toDouble(),
            minimumParallaxDegrees =
                controlPreferences.getFloat(
                    "custom_minimum_parallax_degrees",
                    0.55f,
                ).toDouble(),
            preferredTemporalSkewMillis =
                controlPreferences.getFloat(
                    "custom_preferred_temporal_skew_millis",
                    5.0f,
                ).toDouble(),
            maximumTemporalSkewMillis =
                controlPreferences.getFloat(
                    "custom_maximum_temporal_skew_millis",
                    20.0f,
                ).toDouble(),
            maximumReprojectionErrorPx =
                controlPreferences.getFloat(
                    "custom_maximum_reprojection_error_px",
                    8.0f,
                ).toDouble(),
            maximumReconstructionUncertaintyMeters =
                controlPreferences.getFloat(
                    "custom_maximum_reconstruction_uncertainty_meters",
                    0.75f,
                ).toDouble(),
            posePriorStrength =
                controlPreferences.getFloat(
                    "custom_pose_prior_strength",
                    0.84f,
                ).toDouble(),
            footContactStrictness =
                controlPreferences.getFloat(
                    "custom_foot_contact_strictness",
                    0.80f,
                ).toDouble(),
            headPoseStrictness =
                controlPreferences.getFloat(
                    "custom_head_pose_strictness",
                    0.86f,
                ).toDouble(),
            gazePriority =
                controlPreferences.getFloat(
                    "custom_gaze_priority",
                    0.72f,
                ).toDouble(),
            depthTemporalFusion =
                controlPreferences.getFloat(
                    "custom_depth_temporal_fusion",
                    0.86f,
                ).toDouble(),
            cameraDriftResponse =
                controlPreferences.getFloat(
                    "custom_camera_drift_response",
                    0.82f,
                ).toDouble(),
            thermalQualityFloor =
                controlPreferences.getFloat(
                    "custom_thermal_quality_floor",
                    0.55f,
                ).toDouble(),
            networkJitterToleranceMillis =
                controlPreferences.getFloat(
                    "custom_network_jitter_tolerance_ms",
                    18.0f,
                ).toDouble(),
            meshAdaptiveDetail =
                controlPreferences.getFloat(
                    "custom_mesh_adaptive_detail",
                    0.82f,
                ).toDouble(),
            retargetingStrictness =
                controlPreferences.getFloat(
                    "custom_retargeting_strictness",
                    0.86f,
                ).toDouble(),
            scalePreservation =
                controlPreferences.getFloat(
                    "custom_scale_preservation",
                    0.94f,
                ).toDouble(),
            occlusionRecoveryConfidence =
                controlPreferences.getFloat(
                    "custom_occlusion_recovery_confidence",
                    0.82f,
                ).toDouble(),
            landmarkConsistencyStrictness =
                controlPreferences.getFloat(
                    "custom_landmark_consistency_strictness",
                    0.86f,
                ).toDouble(),
            semanticMotionSensitivity =
                controlPreferences.getFloat(
                    "custom_semantic_motion_sensitivity",
                    0.82f,
                ).toDouble(),
            rootMotionLock =
                controlPreferences.getFloat(
                    "custom_root_motion_lock",
                    0.78f,
                ).toDouble(),
            ultraPrecisionEnabled =
                controlPreferences.getBoolean(
                    "custom_ultra_precision_enabled",
                    false,
                ),
            ultraPrecisionQualificationMode =
                controlPreferences.getBoolean(
                    "custom_ultra_precision_qualification_mode",
                    false,
                ),
            ultraPrecisionTargetP95ErrorMillimeters =
                controlPreferences.getFloat(
                    "custom_ultra_precision_target_p95_mm",
                    0.75f,
                ).toDouble(),
            ultraPrecisionTargetP99ErrorMillimeters =
                controlPreferences.getFloat(
                    "custom_ultra_precision_target_p99_mm",
                    1.75f,
                ).toDouble(),
            ultraPrecisionMinimumEvidenceSamples =
                controlPreferences.getInt(
                    "custom_ultra_precision_min_samples",
                    600,
                ),
            ultraPrecisionMinimumBaselineMeters =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_baseline_m",
                    0.60f,
                ).toDouble(),
            ultraPrecisionMinimumParallaxDegrees =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_parallax_deg",
                    10.0f,
                ).toDouble(),
            ultraPrecisionMinimumCalibrationConfidence =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_calibration_confidence",
                    0.95f,
                ).toDouble(),
            ultraPrecisionMinimumSynchronizationConfidence =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_sync_confidence",
                    0.95f,
                ).toDouble(),
            ultraPrecisionMinimumViewDiversity =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_view_diversity",
                    0.75f,
                ).toDouble(),
            ultraPrecisionMinimumCrossViewAgreement =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_cross_view_agreement",
                    0.90f,
                ).toDouble(),
            ultraPrecisionMaximumOcclusionRatio =
                controlPreferences.getFloat(
                    "custom_ultra_precision_max_occlusion_ratio",
                    0.12f,
                ).toDouble(),
            ultraPrecisionMinimumIdentityContinuity =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_identity",
                    0.95f,
                ).toDouble(),
            ultraPrecisionMinimumSurfaceTemporalStability =
                controlPreferences.getFloat(
                    "custom_ultra_precision_min_surface_stability",
                    0.94f,
                ).toDouble(),
            ultraPrecisionMaximumThermalRisk =
                controlPreferences.getFloat(
                    "custom_ultra_precision_max_thermal_risk",
                    0.60f,
                ).toDouble(),
            ultraPrecisionMaximumMemoryPressure =
                controlPreferences.getFloat(
                    "custom_ultra_precision_max_memory_pressure",
                    0.75f,
                ).toDouble(),
            ultraPrecisionMatchedReferenceRequired =
                controlPreferences.getBoolean(
                    "custom_ultra_precision_reference_required",
                    true,
                ),
        )

    private fun loadMotionTuning(): MotionCaptureTuning =
        MotionCaptureTuning(
            responsiveness = controlPreferences.getFloat(
                "motion_responsiveness",
                MotionCaptureTuning().responsiveness.toFloat(),
            ).toDouble(),
            motionSensitivity = controlPreferences.getFloat(
                "motion_sensitivity",
                MotionCaptureTuning().motionSensitivity.toFloat(),
            ).toDouble(),
            predictionHorizonSeconds = controlPreferences.getFloat(
                "prediction_horizon",
                MotionCaptureTuning().predictionHorizonSeconds.toFloat(),
            ).toDouble(),
            occlusionPersistence = controlPreferences.getFloat(
                "occlusion_persistence",
                MotionCaptureTuning().occlusionPersistence.toFloat(),
            ).toDouble(),
            reacquisitionRadius = controlPreferences.getFloat(
                "reacquisition_radius",
                MotionCaptureTuning().reacquisitionRadius.toFloat(),
            ).toDouble(),
            rejectAggressiveness = controlPreferences.getFloat(
                "reject_aggressiveness",
                MotionCaptureTuning().rejectAggressiveness.toFloat(),
            ).toDouble(),
            jitterSuppression = controlPreferences.getFloat(
                "jitter_suppression",
                MotionCaptureTuning().jitterSuppression.toFloat(),
            ).toDouble(),
            velocityResponsiveness = controlPreferences.getFloat(
                "velocity_responsiveness",
                MotionCaptureTuning().velocityResponsiveness.toFloat(),
            ).toDouble(),
            accelerationResponsiveness = controlPreferences.getFloat(
                "acceleration_responsiveness",
                MotionCaptureTuning().accelerationResponsiveness.toFloat(),
            ).toDouble(),
            innovationGateElasticity = controlPreferences.getFloat(
                "innovation_gate_elasticity",
                MotionCaptureTuning().innovationGateElasticity.toFloat(),
            ).toDouble(),
            motionThresholdScale = controlPreferences.getFloat(
                "motion_threshold_scale",
                MotionCaptureTuning().motionThresholdScale.toFloat(),
            ).toDouble(),
            adaptiveRegimeStrength = controlPreferences.getFloat(
                "adaptive_regime_strength",
                MotionCaptureTuning().adaptiveRegimeStrength.toFloat(),
            ).toDouble(),
        )

    private fun loadReconstructionTuning(): ReconstructionContinuityTuning =
        ReconstructionContinuityTuning(
            softAcceptScore = controlPreferences.getFloat(
                "reconstruction_soft_accept",
                ReconstructionContinuityTuning().softAcceptScore.toFloat(),
            ).toDouble(),
            holdFrames = controlPreferences.getInt(
                "reconstruction_hold_frames",
                ReconstructionContinuityTuning().holdFrames,
            ),
            holdAgeMillis = controlPreferences.getLong(
                "reconstruction_hold_age",
                ReconstructionContinuityTuning().holdAgeMillis,
            ),
            stabilitySensitivity = controlPreferences.getFloat(
                "reconstruction_stability",
                ReconstructionContinuityTuning().stabilitySensitivity.toFloat(),
            ).toDouble(),
        )

    fun setPreviewSurface(surface: Surface?) {
        previewSurfaceReady = surface != null
        camera.setPreviewSurface(surface)
        if (surface != null && _state.value == CaptureState.IDLE) {
            startPreview()
        }
    }

    fun onCameraPermissionResult(granted: Boolean) {
        if (!granted) {
            _lastError.value =
                "Camera permission is required to start NoX capture."
            _state.value = CaptureState.ERROR
            return
        }
        startPreview()
    }

    fun startPreview() {
        if (!previewSurfaceReady) return
        if (
            ContextCompat.checkSelfPermission(
                getApplication(),
                Manifest.permission.CAMERA,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        val id = camera.capabilities().cameraIds.firstOrNull() ?: run {
            fail("No Camera2 camera is available")
            return
        }

        val customization = _motionCustomization.value
        val decision =
            cameraCapturePolicyEngine.resolve(
                capabilities = camera.cameraFeatureMatrix(id),
                customization = customization,
            )

        val previewProfile =
            decision.profile.copy(
                enableRawSidecar = false,
                enableDepthSidecar = false,
                exposureMode =
                    if (customization.cameraExposureMode.name == "MANUAL") {
                        decision.profile.exposureMode
                    } else {
                        com.nivukx.nox.core.model.CameraExposureMode.AUTO
                    },
            )

        camera.open(
            cameraId = id,
            profile = previewProfile,
            allowPersistentSensorSidecars = false,
        ) { error ->
            if (error != null) {
                fail(error.message ?: error.javaClass.simpleName)
            }
        }
    }

    fun discover() {
        _lastError.value = null
        _state.value = CaptureState.DISCOVERING
        runCatching {
            peers.startAdvertising("NoX-" + android.os.Build.MODEL)
            peers.startDiscovery()
        }.onFailure { fail(it.message ?: it.javaClass.simpleName) }
    }

    fun calibrate() {
        _lastError.value = null
        _calibrationReady.value = false
        calibrationSnapshot = null
        calibrationDescriptors.clear()
        calibrationViews.clear()
        peerCapabilities.clear()
        calibrationRawTimestamps.clear()
        multiCameraTemporalAlignmentEngine.reset()
        latestMasterPoseTimestamps.clear()
        _temporalAlignmentConfidence.value = 0.0
        _temporalAlignmentSpreadMillis.value = 0.0
        _temporalAlignmentState.value = "NO_DATA"
        _temporalAlignmentWarnings.value = emptyList()
        calibrationReferenceDevice = localId
        assignedRoles.clear()
        assignedRoles[localId] = DeviceRole.MASTER

        val localFrame = buildLocalCalibrationFrame() ?: return
        calibrationDescriptors[localId] = localFrame
        _state.value = CaptureState.CALIBRATING

        peers.broadcast(
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.CALIBRATION,
                sequence = packetSequence.incrementAndGet(),
                timestampNanos = System.nanoTime(),
                payload = localId.value.toByteArray(Charsets.UTF_8),
            ),
        )
        peers.broadcast(
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.CALIBRATION_FRAME,
                sequence = packetSequence.incrementAndGet(),
                timestampNanos = System.nanoTime(),
                payload = CalibrationWireCodec.encode(localFrame),
            ),
        )
        beginCalibrationCollection(localId)
    }

    fun arm() {
        val snapshot = calibrationSnapshot
        if (peers.connected.value.isNotEmpty() && snapshot == null) {
            fail("Rig calibration is required before distributed capture")
            return
        }

        if (snapshot != null && peers.connected.value.isNotEmpty()) {
            val report = rigQualityGate.evaluate(
                calibration = snapshot,
                cameraPositionsMeters = snapshot.extrinsics.mapValues { it.value.pose.position }
                    .mapKeys { it.key.value },
                clockSkewNanosByCamera = peers.connected.value.associateWith { id ->
                    kotlin.math.abs(clock.estimate(id).offsetNanos)
                }.mapKeys { it.key.value },
            )
            _rigScore.value = report.score
            _rigReasons.value = report.reasons
            if (!report.accepted) {
                fail(
                    "Rig quality gate rejected capture: " +
                        report.reasons.joinToString("; "),
                )
                return
            }
        }

        _lastError.value = null
        _rigScore.value = 1.0
        _rigReasons.value = emptyList()
        _state.value = CaptureState.ARMED
        peers.broadcast(
            NoXWireProtocol.encode(
                NoXWireProtocol.Type.ARM,
                packetSequence.incrementAndGet(),
                System.nanoTime(),
            ),
        )
    }

    fun toggleCapture() {
        when (_state.value) {
            CaptureState.ARMED -> startCapture()
            CaptureState.RECORDING -> stopCapture()
            else -> Unit
        }
    }

    private fun startCapture(
        remote: Boolean = false,
        endpoint: DeviceId? = null,
        masterStartNanos: Long? = null,
    ) {
        scheduledStart?.cancel()

        if (
            _postCaptureState.value == "QUEUED" ||
            _postCaptureState.value == "PROCESSING"
        ) {
            _lastError.value =
                "Post-capture reconstruction is still processing; start a new take after it completes."
            return
        }

        if (
            !remote &&
            _sessionPresentationSettings.value.requireCalibrationBeforeRecord &&
            !_calibrationReady.value
        ) {
            _lastError.value =
                "Calibration is required before recording this take"
            return
        }

        if (!remote) {
            val countdown =
                _sessionPresentationSettings.value.countdownSeconds

            if (countdown > 0) {
                scheduledStart = viewModelScope.launch {
                    for (remaining in countdown downTo 1) {
                        _recordingCountdown.value = remaining
                        delay(1_000L)
                    }
                    _recordingCountdown.value = 0
                    scheduleCaptureStart(
                        endpoint = null,
                        masterStartNanos = null,
                    )
                }
                return
            }
        }

        scheduleCaptureStart(
            endpoint = endpoint,
            masterStartNanos = masterStartNanos,
        )
    }

    private fun scheduleCaptureStart(
        endpoint: DeviceId?,
        masterStartNanos: Long?,
    ) {
        val remote =
            endpoint != null &&
                masterStartNanos != null

        val targetLocalTime =
            if (remote) {
                clock.convertMasterToPeer(
                    endpoint!!,
                    masterStartNanos!!,
                )
            } else {
                val schedule = orchestrator.schedule(
                    peers = _devices.value
                        .filter { it.connected }
                        .map { it.id },
                    nowMasterNanos = System.nanoTime(),
                )

                peers.broadcast(
                    NoXWireProtocol.encode(
                        NoXWireProtocol.Type.START,
                        packetSequence.incrementAndGet(),
                        schedule.masterStartNanos,
                    ),
                )

                schedule.masterStartNanos
            }

        scheduledStart = viewModelScope.launch {
            val remaining = targetLocalTime - System.nanoTime()
            if (remaining > 0L) {
                delay(
                    remaining / 1_000_000L,
                )
            }
            beginRecording()
        }
    }

    private fun beginRecording() {
        scheduledStart = null
        captureGeneration += 1L
        val activeGeneration = captureGeneration
        _lastError.value = null

        val cameraId = camera.capabilities().cameraIds.firstOrNull() ?: run {
            fail("No capture camera available")
            return
        }

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val safeModel = android.os.Build.MODEL.replace(Regex("[^A-Za-z0-9_-]"), "_")
        val sessionId = SessionId(stamp + "_" + safeModel)

        ContextCompat.startForegroundService(
            getApplication(),
            Intent(getApplication(), CaptureForegroundService::class.java),
        )

        val featureMatrix = camera.cameraFeatureMatrix(cameraId)
        val policyDecision =
            cameraCapturePolicyEngine.resolve(
                capabilities = featureMatrix,
                customization = _motionCustomization.value,
            )
        activeCaptureProfile =
            policyDecision.profile.copy(
                preferredGpu = gpuAcceleration.backend,
                fps = minOf(
                    policyDecision.profile.fps,
                    _productionProfile.value.targetCaptureFps,
                ),
            )

        if (policyDecision.warnings.isNotEmpty()) {
            _runtimeReasons.value =
                (
                    _runtimeReasons.value +
                        policyDecision.warnings
                    )
                    .distinct()
                    .takeLast(12)
        }

        inferenceScheduler.configure(
            captureFps = activeCaptureProfile.fps,
            preferredInferenceFps = minOf(
                activeCaptureProfile.fps,
                _motionCustomization.value.inferenceFps,
                _productionProfile.value.inferenceFps,
            ).coerceIn(5, 60),
        )

        val captureMode =
            if (calibrationSnapshot != null && _devices.value.count { it.connected } > 1) {
                MotionCaptureMode.MULTI_VIEW_RGB
            } else {
                MotionCaptureMode.MONOCULAR_RGB
            }

        val configurationFingerprint =
            com.nivukx.nox.core.capture.SessionConfigurationFingerprint.compute(
                customization = _motionCustomization.value,
                captureProfile = activeCaptureProfile,
                mode = captureMode,
                calibration = calibrationSnapshot,
                devices = _devices.value,
            )

        val dir = recorder.start(
            sessionId = sessionId,
            targetFps = activeCaptureProfile.fps,
            width = activeCaptureProfile.analysisWidth,
            height = activeCaptureProfile.analysisHeight,
            devices = _devices.value,
            captureProfile = activeCaptureProfile,
            mode = captureMode,
            clipName = _sessionPresentationSettings.value.sanitizedClipName(),
            configurationFingerprint = configurationFingerprint,
        )

        activeHybridSessionId = sessionId.value
        runCatching {
            hybridCaptureCoordinator.prepare(
                sessionId = sessionId.value,
                mode = _motionCustomization.value.captureEvidenceMode,
                videoProfile = _motionCustomization.value.recordedVideoEvidenceProfile,
                configurationFingerprint = configurationFingerprint,
                reconstructionPolicy = _motionCustomization.value.postCaptureReconstructionPolicy,
                capacityPolicy = _motionCustomization.value.multiPersonCapacityPolicy.copy(
                    targetSubjects = _motionCustomization.value.maxSubjects.coerceAtMost(50),
                ),
                motionArtifactPath = dir.absolutePath,
                depthEvidencePath = null,
            )
            hybridCaptureCoordinator.start()
        }.onFailure { error ->
            activeHybridSessionId = null
            hybridCaptureCoordinator.fail(
                "Hybrid capture preparation failed: " +
                    (error.message ?: error.javaClass.simpleName),
            )
            _lastError.value =
                error.message ?: error.javaClass.simpleName
            _state.value = CaptureState.ERROR
            return
        }

        _frameCount.value = 0L
        _poseCount.value = 0
        _poseConfidence.value = 0f
        _multiPersonObservedCount.value = 0
        _multiPerson3DReconstructedCount.value = 0
        _multiPerson3DDetailedCount.value = 0
        _multiPerson3DPredictedCount.value = 0
        _multiPerson3DPredictionMaxAgeMillis.value = 0L
        _multiPersonAdmittedCount.value = 0
        _multiPersonHeldCount.value = 0
        _multiPersonStarvationRisk.value = 0.0
        _multiPersonFairnessBoostedCount.value = 0
        _multiPersonIdentityContinuity.value = 0.0
        _multiPersonIdentityReidentifications.value = 0
        _multiPersonIdentityPredictions.value = 0
        _crowdMotionMeanIdentitySafety.value = 0.0
        _crowdMotionInteractionPressure.value = 0.0
        _crowdMotionDifferentialScore.value = 0.0
        _crowdMotionSchedulingCapacity.value = 0.0
        _crowdWorldLandmarkCoverage.value = 0.0
        _crowdWorldBoneConsistency.value = 0.0
        _crowdWorldGeometryReliability.value = 0.0
        _crowdAppearanceQuality.value = 0.0
        _crowdSegmentationEvidence.value = 0.0
        _crowdSegmentationCoverage.value = 0.0
        _crowdActorSilhouetteSupport.value = 0.0
        _crowdActorSilhouetteAmbiguity.value = 0.0
        _crowdMotionBottleneck.value =
            com.nivukx.nox.core.ai.ProductionCrowdMotionEvidenceEngine.Bottleneck.NONE.name
        _crowdMotionHeldCount.value = 0
        _crowdMotionPredictedCount.value = 0
        _crowdMotionWarnings.value = emptyList()
        _denseLandmarkCoverage.value = 0.0
        _denseLandmarkVisibleCoverage.value = 0.0
        _denseLandmarkUncertaintyP95.value = 1.0
        _denseLandmarkCrossViewAgreement.value = 0.0
        _denseLandmarkCrossViewEvidenceAvailable.value = false
        _denseLandmarkTemporalStability.value = 0.0
        _denseLandmarkContactCoverage.value = 0.0
        _denseLandmarkPredictionBridgeRatio.value = 0.0
        _denseLandmarkRejectedObservations.value = 0L
        _denseLandmarkStaleObservations.value = 0L
        _denseLandmarkNonFiniteObservations.value = 0L
        _denseLandmarkHardFailure.value = false
        _denseLandmarkContractScore.value = 0.0
        _denseLandmarkContractState.value =
            DenseLandmarkPrecisionContractEngine.State.UNKNOWN.name
        _denseLandmarkContractAuthoritative.value = false
        _denseLandmarkContractFailure.value =
            DenseLandmarkPrecisionContractEngine.FailureClass.NONE.name
        globalDenseLandmarkEvidenceEngine.reset()
        denseEvidenceCameraIds.clear()
        _crowdActorSilhouetteSupport.value = 0.0
        _crowdActorSilhouetteAmbiguity.value = 0.0
        _personSegmentationCoverage.value = 0.0
        _personSegmentationConfidence.value = 0.0
        _personSegmentationBoundaryStability.value = 0.0
        _personSegmentationLatencyMs.value = 0L
        _personSegmentationState.value = "DEGRADED"
        _personSegmentationDelegate.value = "NONE"
        _personSegmentationFallbackCount.value = 0L
        _denseSurfaceTrustedPointRatio.value = 0.0
        _denseSurfaceUncertaintyP95Meters.value =
            Double.POSITIVE_INFINITY
        _denseSurfaceTopologyStability.value = 0.0
        _denseSurfaceViewDiversity.value = 0.0
        _denseSurfaceProbabilisticEvidenceScore.value = 1.0
        _denseSurfaceMeanReprojectionErrorPx.value =
            Double.POSITIVE_INFINITY
        _denseSurfaceMeanAppearanceConsistency.value = 0.0
        _denseSurfaceMeanSilhouetteConsistency.value = 0.0
        _denseSurfaceMeanMotionBoundaryConfidence.value = 0.0
        _denseSurfaceMeanLightingStability.value = 0.0
        _denseSurfaceHardFailure.value = false
        _denseGeometricConditioningScore.value = 0.0
        _denseGeometricConditioningMedianParallaxDegrees.value = 0.0
        _denseGeometricConditioningMinimumObservationScore.value = 1.0
        _denseGeometricConditioningWarnings.value = emptyList()
        _denseIdentityGateQuality.value = 1.0
        _denseIdentityGateRejectedCount.value = 0
        _denseIdentityGateEvaluatedPoints.value = 0
        _denseIdentityGateWarnings.value = emptyList()
        _denseContactConfidence.value = 0.0
        _denseContactProtectedLandmarkCount.value = 0
        _denseContactWarnings.value = emptyList()
        _motionScore.value = 0.0
        _commonModeMotionNormalizedPerSecond.value = 0.0
        _differentialMotionScore.value = 0.0
        _cameraCompensatedMotionScore.value = 0.0
        _motionMeasurementUncertainty.value = 0.0
        _microMotionScore.value = 0.0
        _directionChangeRatio.value = 0.0
        _temporalStability.value = 0.0
        lightingTemporalConsistencyEngine.reset()
        motionEvidenceEngine.reset()
        faceMeshTemporalConsistencyEngine.reset()
        faceMeshFusion.reset()
        faceMeshSurfaceTemporalEngine.reset()
        faceMeshTransportPolicy.reset()
        faceShapeAlignmentEngine.reset()
        _motionEvidenceScore.value = 0.0
        _motionEvidenceConfidence.value = 0.0
        _motionEvidenceWarnings.value = emptyList()
        _reconstructionJoints.value = 0
        _reconstructionErrorPx.value = Double.POSITIVE_INFINITY
        _reconstructionQualityScore.value = 0.0
        _bodyMetricEvidenceAuthority.value = 0.0
        _bodySymmetricEpipolarConsistency.value = 1.0
        _reconstructionGateReasons.value = emptyList()
        _kinematicConstraintScore.value = 1.0
        _kinematicConstraintAccepted.value = true
        _kinematicConstraintBottleneck.value = "NONE"
        _kinematicConstraintWarnings.value = emptyList()
        _physicsConstraintScore.value = 1.0
        _physicsConstraintAccepted.value = true
        _physicsConstraintMeanBoneResidualMeters.value = 0.0
        _physicsConstraintP95BoneResidualMeters.value = 0.0
        _physicsConstraintMaxBoneResidualMeters.value = 0.0
        _physicsConstraintSymmetryScore.value = 1.0
        _physicsConstraintFloorContactScore.value = 1.0
        _physicsConstraintCorrectedJointCount.value = 0
        _physicsConstraintConstrainedEdgeCount.value = 0
        _physicsConstraintWarnings.value = emptyList()
        _global4DScore.value = 1.0
        _global4DAccepted.value = true
        _global4DState.value =
            com.nivukx.nox.core.reconstruction.Global4DConsistencyEngine.State.WARMING.name
        _global4DPersistentFailure.value = false
        _global4DRootMotionConsistency.value = 1.0
        _global4DRootVelocityAgreement.value = 1.0
        _global4DSceneConsistency.value = 0.72
        _global4DDirectionalUncertaintyConsistency.value = 1.0
        _global4DFaceEyeConsistency.value = 1.0
        _global4DExternalContactConsistency.value = 1.0
        _global4DGaussianConsistency.value = 1.0
        _global4DGaussianStrainConsistency.value = 1.0
        _global4DGaussianEvidenceAvailable.value = false
        _global4DWarnings.value = emptyList()
        _worldSpaceConsistencyScore.value = 1.0
        _worldSpaceAccepted.value = true
        _worldSpaceTemporalStability.value = 1.0
        _worldSpaceMotionContinuity.value = 1.0
        _worldSpaceRejectedJointCount.value = 0
        _worldSpaceRobustOutlierRatio.value = 0.0
        _worldSpaceWarnings.value = emptyList()
        _denseMeshSurfaceConfidence.value = 0.0
        _denseMeshSurfaceObserved.value = false
        _denseMeshTopologyIntegrity.value = 0.0
        _denseMeshBoundaryEdges.value = 0
        _denseMeshNonManifoldEdges.value = 0
        _denseMeshDuplicateTrianglesRemoved.value = 0
        _denseTsdfDetailLevel.value = 1.0
        _denseTsdfComputePressure.value = 0.0
        _denseTsdfSurfaceNormalObservability.value = 1.0
        _denseTsdfMeanEvidenceWeight.value = 0.0
        _denseTsdfMeanUncertainty.value = 0.0
        _denseTsdfHighUncertaintyRatio.value = 0.0
        _denseTsdfConfidenceFactor.value = 1.0
        _denseTsdfDecisionReason.value = "RESET"
        _denseSurfaceMeanDynamicProbability.value = 0.0
        _denseSurfaceMeanTemporalStability.value = 1.0
        _denseSurfaceTrustedGaussianRatio.value = 0.0
        _worldConsistencyScore.value = 0.0
        _worldConsistencyWarnings.value = emptyList()
        _captureTimingConfidence.value = 0.0
        _captureTimingFps.value = 0.0
        _captureTimingJitterMillis.value = 0.0
        _captureTimingHealthy.value = false
        _cameraSessionVerification.value =
            CameraSessionNegotiationEngine.Verification.UNKNOWN.name
        _cameraSessionSupported.value = null
        _highSpeedSessionVerification.value =
            CameraSessionNegotiationEngine.Verification.UNKNOWN.name
        _highSpeedSessionSupported.value = null
        _highSpeedSessionActive.value = false
        _highSpeedSelectedFps.value = null
        _highSpeedSessionReason.value = null
        _hardwareReadoutTimestampSupported.value = false
        _latestReadoutTimestampNanos.value = null
        _hardwareReadoutTimingState.value =
            HardwareReadoutTimingEvidenceEngine.State.UNKNOWN.name
        _hardwareReadoutTimingConfidence.value = 0.0
        _hardwareReadoutP95JitterMillis.value = Double.POSITIVE_INFINITY
        _hardwareReadoutMedianDelayMillis.value = Double.NaN
        hardwareReadoutTimingEvidenceEngine.reset()
        _thermalExtrinsicDriftState.value =
            ThermalExtrinsicDriftEngine.State.UNKNOWN.name
        _thermalExtrinsicDriftConfidence.value = 0.0
        _thermalExtrinsicTranslationDriftMm.value = 0.0
        _thermalExtrinsicRotationDriftDegrees.value = 0.0
        _thermalExtrinsicTemperatureDeltaCelsius.value = 0.0
        _thermalExtrinsicHardFailure.value = false
        _thermalExtrinsicWarnings.value = emptyList()
        thermalExtrinsicDriftEngine.clear()
        rigHealthEngine.reset()
        _concurrentRigAdmissionState.value = "NOT_APPLICABLE"
        _concurrentRigAdmissionScore.value = 1.0
        _concurrentRigTopologyScore.value = 0.0
        _concurrentRigTopologyMode.value =
            CameraTopologyEvidenceEngine.Mode.UNKNOWN.name
        _concurrentRigSelectedCameraIds.value = emptyList()
        _concurrentRigBlockers.value = emptyList()
        _concurrentRigWarnings.value = emptyList()
        lastConcurrentRigRefreshNanos = Long.MIN_VALUE
        _cameraCaptureQualityScore.value = 0.0
        _cameraOpticalBlurRisk.value = 0.0
        _cameraCaptureQualityWarnings.value = emptyList()
        _cameraAnalysisDropRatio.value = 0.0
        cameraCaptureQualityEngine.reset()
        camera.resetAnalysisBackpressureTelemetry()
        _sessionHealthState.value = SessionHealthEngine.State.NO_DATA.name
        _sessionHealthScore.value = 0.0
        _sessionHealthBottleneck.value = "NONE"
        _sessionHealthWarnings.value = emptyList()
        _multiCameraSyncConfidence.value = 1.0
        _multiCameraSyncState.value =
            MultiCameraSynchronizationQualityEngine.State.NO_PEERS.name
        _temporalAlignmentConfidence.value = 0.0
        _temporalAlignmentSpreadMillis.value = 0.0
        _temporalAlignmentState.value = "NO_DATA"
        _temporalAlignmentWarnings.value = emptyList()
        multiCameraTemporalAlignmentEngine.reset()
        latestMasterPoseTimestamps.clear()
        lightingTemporalConsistencyEngine.reset()
        _lightingLux.value = 0.0
        _colorTemperature.value = 0.0
        _lightingConfidence.value = 0.0
        _gpuBackend.value = gpuAcceleration.backend
        _rawSidecarEnabled.value = activeCaptureProfile.enableRawSidecar
        _postCaptureState.value = "IDLE"
        _postCaptureProgress.value = 0.0
        _postCaptureProcessedFrames.value = 0L
        _postCaptureMaxActors.value = 0
        _postCaptureVideoPath.value = null
        _depthSidecarEnabled.value = activeCaptureProfile.enableDepthSidecar

        _calibrationReady.value = calibrationSnapshot != null
        calibrationSnapshot?.let(::configurePerceptionCalibration)
            ?: perception.configureCalibratedCameraGeometry(emptyMap())

        recorder.updateCalibration(calibrationSnapshot)

        // Reserve the capture plan once; the actual Camera2 open happens after all frame-flow
        // collectors are installed at the end of beginRecording(). A second open here would
        // invalidate this controller generation and discard its optional output topology.
        val enableLiveDepthStream =
            calibrationSnapshot?.let { snapshot ->
                localId in snapshot.intrinsics && localId in snapshot.extrinsics
            } == true

        inferenceScheduler.reset()
        inertialFusion.reset()
        motionStateEstimator.reset()
        monocularPhysicsConstraintEngine.reset()
        productionQualityEngine.reset()
        physicalPrecisionCapabilityEngine.reset()
        _physicalPrecisionCapability.value =
            PhysicalPrecisionCapabilityEngine.Decision.idle()
        productionMotionDynamicsEngine.reset()
        _motionLowFrequencyScore.value = 0.0
        _motionHighFrequencyScore.value = 0.0
        _motionTemporalMemoryScore.value = 0.0
        _productionQualityScore.value = 0.0
        _captureAuthorityScore.value = 0.0
        _captureAuthorityBand.value = "BLOCKED"
        _captureAuthorityHardBlocked.value = true
        _captureAuthorityPrecisionQualified.value = false
        _captureAuthorityWarnings.value = emptyList()
        _productionPolicy.value =
            HighEndReconstructionQualityEngine.Policy.HOLD.name
        _productionBottleneck.value =
            HighEndReconstructionQualityEngine.Bottleneck.NONE.name
        _productionOutputAllowed.value = false
        _productionWarnings.value = emptyList()
        previousMonocularFrame = null
        previousMonocularAvatarPose = null
        imuCapture.start()

        imuJob?.cancel()
        imuJob = viewModelScope.launch {
            var lastGyro = Long.MIN_VALUE
            var lastAccel = Long.MIN_VALUE

            imuCapture.samples.collect { pair ->
                if (activeGeneration != captureGeneration) {
                    return@collect
                }
                pair.gyroscope?.let { sample ->
                    if (sample.timestampNanos != lastGyro) {
                        recorder.appendImu(sample)
                        lastGyro = sample.timestampNanos
                    }
                }

                pair.accelerometer?.let { sample ->
                    if (sample.timestampNanos != lastAccel) {
                        recorder.appendImu(sample)
                        lastAccel = sample.timestampNanos
                    }
                }

                val inertialPose = inertialFusion.update(
                    gyroscope = pair.gyroscope,
                    accelerometer = pair.accelerometer,
                )

                val inertialCameraRisk =
                    inertialFusion.cameraMotionRisk()

                if (inertialCameraRisk > _cameraMotionRisk.value) {
                    _cameraMotionRisk.value =
                        inertialCameraRisk
                }

                if (inertialPose.confidence < 0.22) {
                    _runtimeReasons.value =
                        (
                            _runtimeReasons.value +
                                "inertial-confidence-low"
                            ).distinct().takeLast(12)
                }
            }
        }

        handFrames.clear()
        handFusion.reset()
        crossModalIdentityFusionEngine.reset()
        latestFaceCenter3D = null
        latestFaceCenterTimestampNanos = Long.MIN_VALUE
        latestLeftHand3D = null
        latestLeftHandTimestampNanos = Long.MIN_VALUE
        latestRightHand3D = null
        latestRightHandTimestampNanos = Long.MIN_VALUE
        handTemporalConsistencyEngine.reset()
        monocularTrajectoryOptimizer.reset()
        handGeometryQualityEngine.reset()
        handUncertaintyCalibrationEngine.reset()
        _handCount.value = 0
        _handConfidence.value = 0.0
        _hand3DConfidence.value = 0.0
        _handTemporalConsistencyConfidence.value = 0.0
        _handLongitudinalAnatomyConsistency.value = 0.78
        _handUncertaintyConfidence.value = 1.0
        _handUncertaintyRadiusMeters.value = 0.0
        _handUncertaintyCoverage.value = 0.0
        _handUncertaintyOodRisk.value = 0.0
        _handUncertaintyPolicy.value = "NORMAL"
        _handUncertaintyWarnings.value = emptyList()
        _handTemporalConsistencyWarnings.value = emptyList()
        _focusMode.value = "NONE"
        _gpuDispatchMillis.value = 0.0
        _semanticCoverage.value = 0.0
        _attentionConfidence.value = 0.0
        _occlusionRisk.value = 0.0
        _dominantSubject.value = "NONE"
        _motionClass.value = "STATIC"
        _focusConfidence.value = 0.0
        _inferenceFps.value = 0
        _inferenceReason.value = "RESET"
        focusTargetEngine.reset()
        camera.setFocusTarget(null)
        denseFusion.reset()
        nativeDenseTsdfBridge?.reset()
        nativeDenseTsdfFaultReported = false
        temporalDepthConsistencyEngine.reset()
        latestTemporalDepthConsistency = 0.0
        _temporalDepthConsistency.value = 0.0
        denseMeshTemporalConsistencyEngine.reset()
        lastDenseMeshExtractNanos =
            Long.MIN_VALUE
        _denseMeshVertices.value = 0
        _denseMeshTriangles.value = 0
        _denseMeshConfidence.value = 0.0
        _denseMeshNormalCoverage.value = 0.0
        _denseMeshTemporalConsistencyConfidence.value = 0.0
        _denseMeshTemporalConsistencyWarnings.value = emptyList()

        depthJob?.cancel()
        depthJob = viewModelScope.launch {
            var depthFrameCounter = 0L

            camera.depthFlow.collect { depthFrame ->
                if (activeGeneration != captureGeneration) {
                    return@collect
                }
                depthFrameCounter += 1L
                val snapshot = calibrationSnapshot
                    ?: return@collect

                val cameraId = camera.capabilities()
                    .cameraIds
                    .firstOrNull()
                    ?: return@collect

                val intrinsics = camera.intrinsicsModel(
                    cameraId = cameraId,
                    imageWidth = depthFrame.width,
                    imageHeight = depthFrame.height,
                )
                    ?: return@collect

                val extrinsics =
                    snapshot.extrinsics[localId]
                        ?: return@collect

                val depthConfidence =
                    depthFrame.descriptor()
                        .meanConfidence
                        .coerceIn(0.05, 1.0)

                val temporalDepth =
                    temporalDepthConsistencyEngine.evaluate(
                        depthMeters = depthFrame.depthMeters,
                        width = depthFrame.width,
                        height = depthFrame.height,
                        intrinsics = intrinsics,
                        cameraPose = extrinsics.pose,
                        timestampNanos = depthFrame.timestampNanos,
                        sourceConfidence = depthConfidence,
                        depthConfidence = depthFrame.confidence,
                    )
                latestTemporalDepthConsistency =
                    temporalDepth.confidence.coerceIn(0.0, 1.0)
                _temporalDepthConsistency.value =
                    latestTemporalDepthConsistency

                val adaptiveDepthDecision = denseFusion.integrateDepthAdaptive(
                    depthMeters = depthFrame.depthMeters,
                    width = depthFrame.width,
                    height = depthFrame.height,
                    intrinsics = intrinsics,
                    cameraPose = extrinsics.pose,
                    timestampNanos = depthFrame.timestampNanos,
                    baseStride =
                        _productionTsdfStride.value.coerceIn(1, 4),
                    confidence = depthConfidence,
                    depthConfidence = depthFrame.confidence,
                    evidence = adaptiveTsdfEvidence(),
                    uncertaintyModel =
                        depthUncertaintyModelEngine,
                )
                _denseTsdfDetailLevel.value = adaptiveDepthDecision.detailLevel
                _denseTsdfComputePressure.value = adaptiveDepthDecision.computePressure
                _denseTsdfSurfaceNormalObservability.value =
                    adaptiveDepthDecision.meanSurfaceNormalWeight
                        .coerceIn(0.0, 1.0)
                _denseTsdfMeanEvidenceWeight.value =
                    adaptiveDepthDecision.meanEvidenceWeight
                        .coerceIn(0.0, 1.0)

                if (depthFrameCounter % 2L == 0L) {
                    runCatching {
                        val nativeBridge =
                            nativeDenseTsdfBridge
                                ?: NativeSparseTsdfReconstructionBridge(
                                    voxelSizeMeters = 0.00625,
                                    truncationDistanceMeters = 0.03,
                                    maxTriangles = 300_000,
                                ).also {
                                    nativeDenseTsdfBridge = it
                                }

                        val nativeEvidence =
                            com.nivukx.nox.core.nativeengine.TsdfDetailEvidence(
                                motionEnergy =
                                    _motionScore.value.coerceIn(0.0, 1.0),
                                uncertainty =
                                    (
                                        1.0 -
                                            _reconstructionUncertaintyConfidence.value
                                    ).coerceIn(0.0, 1.0),
                                cameraCoverage =
                                    (
                                        _devices.value.count { it.connected }
                                            .toDouble() /
                                            _devices.value.size
                                                .coerceAtLeast(1)
                                                .toDouble()
                                    ).coerceIn(0.0, 1.0),
                                reconstructionConfidence =
                                    _reconstructionQualityScore.value
                                        .coerceIn(0.0, 1.0),
                                thermalHeadroom =
                                    runtimeThermalHeadroom(),
                                gpuHeadroom =
                                    (
                                        1.0 -
                                            _gpuDispatchMillis.value /
                                                16.67
                                    ).coerceIn(0.20, 1.0),
                            )

                        nativeBridge.integrateDepthFrameAdaptive(
                            depthFrame = depthFrame,
                            intrinsics = intrinsics,
                            pose = extrinsics.pose,
                            stride =
                                adaptiveDepthDecision.stride
                                    .coerceIn(1, 4),
                            confidence =
                                NativeDepthConfidencePolicy.resolve(
                                    frameConfidence = depthConfidence,
                                    adaptiveScale = adaptiveDepthDecision.confidenceScale,
                                ),
                            evidence = nativeEvidence,
                        )

                        val surfels =
                            denseFusion.extractSurfaceDetail(
                                timestampNanos =
                                    depthFrame.timestampNanos,
                                maxPoints = 4_096,
                            )

                        if (surfels.isNotEmpty()) {
                            nativeBridge.integrateGaussianEvidence(
                                surfels.map { surfel ->
                                    com.nivukx.nox.core.nativeengine.GaussianSurfaceObservation(
                                        positionX = surfel.position.x,
                                        positionY = surfel.position.y,
                                        positionZ = surfel.position.z,
                                        normalX = surfel.normal.x,
                                        normalY = surfel.normal.y,
                                        normalZ = surfel.normal.z,
                                        velocityX = 0.0,
                                        velocityY = 0.0,
                                        velocityZ = 0.0,
                                        varianceX =
                                            surfel.varianceX.coerceAtLeast(1e-7),
                                        varianceY =
                                            surfel.varianceY.coerceAtLeast(1e-7),
                                        varianceZ =
                                            surfel.varianceZ.coerceAtLeast(1e-7),
                                        confidence =
                                            surfel.confidence.coerceIn(0.0, 1.0),
                                        dynamicProbability =
                                            surfel.dynamicProbability
                                                .coerceIn(0.0, 1.0),
                                        temporalConsistency =
                                            surfel.temporalStability
                                                .coerceIn(0.0, 1.0),
                                        timestampNanos =
                                            depthFrame.timestampNanos,
                                    )
                                },
                            )
                        }
                    }.onFailure { error ->
                        if (!nativeDenseTsdfFaultReported) {
                            nativeDenseTsdfFaultReported = true
                            _runtimeReasons.value =
                                (
                                    _runtimeReasons.value +
                                        "native-reconstruction-path-" +
                                        (error.message
                                            ?: error.javaClass.simpleName)
                                    )
                                    .distinct()
                                    .takeLast(12)
                        }
                    }
                }

                if (
                    temporalDepth.warnings.isNotEmpty() &&
                    _state.value == CaptureState.RECORDING
                ) {
                    _runtimeReasons.value =
                        (
                            _runtimeReasons.value +
                                temporalDepth.warnings.map {
                                    "temporal-depth-$it"
                                }
                            )
                            .distinct()
                            .takeLast(12)
                }
                _denseTsdfDecisionReason.value = adaptiveDepthDecision.reason
                if (_state.value == CaptureState.RECORDING) {
                    recorder.appendTsdfEvidence(
                        timestampNanos = depthFrame.timestampNanos,
                        decision = adaptiveDepthDecision,
                        temporalDepthConsistency =
                            temporalDepth.confidence,
                    )
                }
                refreshTsdfUncertaintyTelemetry()

                if (depthFrameCounter % 3L == 0L) {
                    DepthWireCodec.encode(
                        deviceId = localId,
                        frame = depthFrame,
                    )?.let { payload ->
                        peers.broadcast(
                            NoXWireProtocol.encode(
                                type = NoXWireProtocol.Type.DEPTH_FRAME,
                                sequence = packetSequence.incrementAndGet(),
                                timestampNanos = depthFrame.timestampNanos,
                                payload = payload,
                            ),
                        )
                    }
                }

                refreshDenseMesh(
                    depthFrame.timestampNanos,
                )
            }
        }

        _state.value = CaptureState.RECORDING

        frameJob?.cancel()
        frameJob = viewModelScope.launch {
            camera.frameFlow.collect { (frame, image) ->
                if (activeGeneration != captureGeneration) {
                    image.close()
                    return@collect
                }
                try {
                    _frameCount.value += 1L

                    val telemetry = camera.latestTelemetry
                    telemetry?.let {
                        _cameraAnalysisDropRatio.value =
                            camera.analysisDropRatio.coerceIn(0.0, 1.0)

                        val cameraQuality =
                            cameraCaptureQualityEngine.evaluate(
                                telemetry = it,
                                targetFps = activeCaptureProfile.fps,
                                focusMode = activeCaptureProfile.focusMode,
                                analysisDropRatio = camera.analysisDropRatio,
                            )

                        _cameraCaptureQualityScore.value =
                            cameraQuality.score
                        _cameraOpticalBlurRisk.value =
                            cameraQuality.opticalBlurRisk
                                .coerceIn(0.0, 1.0)
                        _cameraCaptureQualityWarnings.value =
                            cameraQuality.warnings

                        if (cameraQuality.warnings.isNotEmpty()) {
                            _runtimeReasons.value =
                                (
                                    _runtimeReasons.value +
                                        cameraQuality.warnings
                                    )
                                    .distinct()
                                    .takeLast(12)
                        }

                        recorder.appendFrameTelemetry(
                            sequence = frame.sequence,
                            telemetry = it,
                            cameraQuality = cameraQuality,
                        )
                    }

                    camera.latestSessionPlan?.let { plan ->
                        _cameraSessionVerification.value =
                            plan.verification.name
                        _cameraSessionSupported.value =
                            when (plan.verification) {
                                CameraSessionNegotiationEngine.Verification.HARDWARE_VERIFIED,
                                CameraSessionNegotiationEngine.Verification.REJECTED ->
                                    plan.supported
                                CameraSessionNegotiationEngine.Verification.PLATFORM_DECLARED_ONLY,
                                CameraSessionNegotiationEngine.Verification.UNKNOWN ->
                                    null
                            }
                    }
                    camera.latestHighSpeedSessionPlan?.let { plan ->
                        _highSpeedSessionVerification.value =
                            plan.verification.name
                        _highSpeedSessionSupported.value =
                            when (plan.verification) {
                                CameraSessionNegotiationEngine.Verification.HARDWARE_VERIFIED,
                                CameraSessionNegotiationEngine.Verification.REJECTED ->
                                    plan.supported
                                CameraSessionNegotiationEngine.Verification.PLATFORM_DECLARED_ONLY,
                                CameraSessionNegotiationEngine.Verification.UNKNOWN ->
                                    null
                            }
                        _highSpeedSelectedFps.value =
                            plan.selectedFps
                        _highSpeedSessionReason.value =
                            plan.reason
                    }
                    _highSpeedSessionActive.value =
                        camera.isHighSpeedSessionActive

                    _hardwareReadoutTimestampSupported.value =
                        camera.hardwareReadoutTimestampSupported
                    _latestReadoutTimestampNanos.value =
                        camera.latestTelemetry?.readoutTimestampNanos

                    camera.latestTelemetry?.let { telemetry ->
                        telemetry.readoutTimestampNanos?.let { readoutNanos ->
                            val readoutEvidence =
                                hardwareReadoutTimingEvidenceEngine.observe(
                                    HardwareReadoutTimingEvidenceEngine.Observation(
                                        sequence = frame.sequence,
                                        sensorTimestampNanos =
                                            telemetry.sensorTimestampNanos,
                                        readoutTimestampNanos = readoutNanos,
                                    ),
                                )
                            _hardwareReadoutTimingState.value =
                                readoutEvidence.state.name
                            _hardwareReadoutTimingConfidence.value =
                                readoutEvidence.confidence
                            _hardwareReadoutP95JitterMillis.value =
                                readoutEvidence.p95JitterMillis
                            _hardwareReadoutMedianDelayMillis.value =
                                readoutEvidence.medianDelayMillis
                            recorder.appendHardwareReadoutTiming(
                                sequence = frame.sequence,
                                result = readoutEvidence,
                            )
                        }
                    }

                    updateThermalExtrinsicEvidence(
                        telemetry = telemetry,
                        timestampNanos = frame.sensorTimestampNanos,
                    )

                    camera.latestTimingQuality?.let { timing ->
                        _captureTimingConfidence.value = timing.confidence
                        _captureTimingFps.value = timing.cadenceFps
                        _captureTimingJitterMillis.value =
                            timing.p95AbsoluteJitterNanos / 1_000_000.0
                        _captureTimingHealthy.value = timing.healthy
                        recorder.appendCaptureTiming(
                            sequence = frame.sequence,
                            quality = timing,
                            assessment = camera.latestTimingAssessment,
                        )
                    }

                    val decision =
                        inferenceScheduler.begin(
                            frame.sensorTimestampNanos,
                        )

                    _inferenceFps.value =
                        decision.effectiveInferenceFps
                    _inferenceReason.value =
                        decision.reason

                    perception.updateSystemLoad(
                        thermalScale = when {
                            decision.thermalStatus >= android.os.PowerManager.THERMAL_STATUS_CRITICAL -> 0.22
                            decision.thermalStatus >= android.os.PowerManager.THERMAL_STATUS_SEVERE -> 0.32
                            decision.thermalStatus >= android.os.PowerManager.THERMAL_STATUS_MODERATE -> 0.55
                            decision.thermalStatus >= android.os.PowerManager.THERMAL_STATUS_LIGHT -> 0.80
                            else -> 1.0
                        },
                        memoryScale = (1.0 - decision.heapPressure)
                            .coerceIn(0.25, 1.0),
                    )

                    if (!decision.process) {
                        if (frame.sequence % 6L == 0L) {
                            val lighting =
                                lightingEngine.analyze(
                                    image = image,
                                    telemetry = telemetry,
                                    timestampNanos = frame.sensorTimestampNanos,
                                )
                            applyLightingEstimate(lighting)
                        }

                        image.close()
                        recorder.appendFrame(
                            sequence = frame.sequence,
                            sensorTimestampNanos = frame.sensorTimestampNanos,
                            monotonicTimestampNanos = frame.monotonicTimestampNanos,
                            poseLandmarks = 0,
                        )
                        return@collect
                    }

                    val inferenceStartNanos =
                        SystemClock.elapsedRealtimeNanos()
                    val frameTimingConfidence = resolvedCaptureTimingConfidence()

                    perception.process(
                        image = image,
                        rotationDegrees = camera.sensorOrientation(cameraId),
                        timestampNanos = frame.sensorTimestampNanos,
                        cameraId = cameraId,
                        calibrationCameraId = localId.value,
                        motionContext =
                            motionContextForPerception(),
                        rollingShutterSkewNanos =
                            telemetry?.rollingShutterSkewNanos ?: 0L,
                        frameDurationNanos =
                            telemetry?.frameDurationNanos ?: 0L,
                        exposureTimeNanos =
                            telemetry?.exposureTimeNanos ?: 0L,
                        captureTimingConfidence = frameTimingConfidence,
                        lightingConfidence =
                            _lightingConfidence.value
                                .coerceIn(0.0, 1.0),
                    ).addOnSuccessListener { result ->
                        val lighting =
                            if (frame.sequence % 3L == 0L) {
                                lightingEngine.analyze(
                                    image = image,
                                    telemetry = telemetry,
                                    timestampNanos = frame.sensorTimestampNanos,
                                )
                            } else {
                                null
                            }

                        lighting?.let {
                            applyLightingEstimate(it)
                        }

                        val pose = result.pose

                        ingestNativePerceptionResult(
                            cameraId =
                                localId.value.hashCode() and
                                    0x7fffffff,
                            result = result,
                            imageWidth = image.width,
                            imageHeight = image.height,
                            thermalStatus = decision.thermalStatus,
                            memoryPressure = decision.heapPressure,
                        )

                        publishLiveMotionPreview(
                            result = result,
                            skeleton = null,
                            cameraWidth = image.width,
                            cameraHeight = image.height,
                            timestampNanos = pose.timestampNanos,
                        )
                        latestMasterPoseTimestamps[localId] =
                            pose.timestampNanos
                        _poseCount.value = pose.landmarks.size
                        _poseConfidence.value = pose.confidence
                        result.personSegmentation?.let { segmentation ->
                            _personSegmentationCoverage.value =
                                segmentation.foregroundCoverage
                                    .coerceIn(0.0, 1.0)
                            _personSegmentationConfidence.value =
                                segmentation.meanForegroundConfidence
                                    .coerceIn(0.0, 1.0)
                            _personSegmentationBoundaryStability.value =
                                segmentation.boundaryStability
                                    .coerceIn(0.0, 1.0)
                            _personSegmentationLatencyMs.value =
                                segmentation.inferenceLatencyMs
                                    .coerceAtLeast(0L)
                            _personSegmentationState.value =
                                segmentation.state.name
                            _personSegmentationDelegate.value =
                                segmentation.activeDelegate?.name
                                    ?: "NONE"
                            _personSegmentationFallbackCount.value =
                                segmentation.gpuFallbackCount
                        }

                        result.crowdMotionEvidence?.let { crowd ->
                            _crowdActorSilhouetteSupport.value =
                                crowd.meanActorSilhouetteSupport
                                    .coerceIn(0.0, 1.0)
                            _crowdActorSilhouetteAmbiguity.value =
                                crowd.meanActorSilhouetteAmbiguity
                                    .coerceIn(0.0, 1.0)
                            _crowdMotionMeanIdentitySafety.value =
                                crowd.meanIdentitySafety.coerceIn(0.0, 1.0)
                            _crowdMotionInteractionPressure.value =
                                crowd.interactionPressure.coerceIn(0.0, 1.0)
                            _crowdMotionDifferentialScore.value =
                                crowd.differentialMotionScore.coerceIn(0.0, 1.0)
                            _crowdMotionSchedulingCapacity.value =
                                crowd.globalSchedulingCapacity.coerceIn(0.0, 1.0)
                            _crowdWorldLandmarkCoverage.value =
                                crowd.meanWorldLandmarkCoverage.coerceIn(0.0, 1.0)
                            _crowdWorldBoneConsistency.value =
                                crowd.meanWorldBoneConsistency.coerceIn(0.0, 1.0)
                            _crowdWorldGeometryReliability.value =
                                crowd.meanWorldGeometryReliability.coerceIn(0.0, 1.0)
                            _crowdAppearanceQuality.value =
                                crowd.meanAppearanceQuality.coerceIn(0.0, 1.0)
                            _crowdSegmentationEvidence.value =
                                crowd.segmentationEvidence.coerceIn(0.0, 1.0)
                            _crowdSegmentationCoverage.value =
                                crowd.segmentationCoverage.coerceIn(0.0, 1.0)
                            _crowdActorSilhouetteSupport.value =
                                crowd.meanActorSilhouetteSupport.coerceIn(0.0, 1.0)
                            _crowdActorSilhouetteAmbiguity.value =
                                crowd.meanActorSilhouetteAmbiguity.coerceIn(0.0, 1.0)
                            _crowdMotionBottleneck.value =
                                crowd.bottleneck.name
                            _crowdMotionHeldCount.value =
                                crowd.heldActorCount
                            _crowdMotionPredictedCount.value =
                                crowd.predictedActorCount
                            _crowdMotionWarnings.value =
                                crowd.warnings.takeLast(8)
                        }

                        result.multiActorPose?.let { multiActor ->
                            _multiPersonIdentityContinuity.value =
                                multiActor.identityContinuityScore
                                    .coerceIn(0.0, 1.0)
                            _multiPersonIdentityReidentifications.value =
                                multiActor.identityReidentifiedTrackCount
                            _multiPersonIdentityPredictions.value =
                                multiActor.identityPredictedTrackCount
                        }

                        result.multiPersonCapacityAllocation?.let { allocation ->
                            _multiPersonObservedCount.value =
                                allocation.runtime.subjectCount
                            _multiPersonAdmittedCount.value =
                                allocation.admittedSubjectIds.size
                            _multiPersonHeldCount.value =
                                allocation.heldSubjectIds.size
                            _multiPersonStarvationRisk.value =
                                allocation.starvationRisk.coerceIn(0.0, 1.0)
                            _multiPersonFairnessBoostedCount.value =
                                allocation.fairnessBoostedSubjectIds.size
                        }
                        result.denseContactEvidence?.let { contact ->
                            _denseContactConfidence.value =
                                contact.contactConfidence.coerceIn(
                                    0.0,
                                    1.0,
                                )
                            _denseContactProtectedLandmarkCount.value =
                                contact.protectedLandmarks.size
                            _denseContactWarnings.value =
                                contact.warningFlags
                        }

                        result.denseLandmarkEvidence?.let { dense ->
                            updateGlobalDenseLandmarkEvidence(
                                deviceId = localId,
                                timestampNanos = pose.timestampNanos,
                                dense = dense,
                            )
                            _denseLandmarkCoverage.value =
                                dense.coverage.coerceIn(0.0, 1.0)
                            _denseLandmarkVisibleCoverage.value =
                                dense.visibleCoverage.coerceIn(0.0, 1.0)
                            _denseLandmarkUncertaintyP95.value =
                                dense.uncertaintyP95.coerceIn(0.0, 1.0)
                            _denseLandmarkCrossViewAgreement.value =
                                dense.crossViewAgreement.coerceIn(0.0, 1.0)
                            _denseLandmarkCrossViewEvidenceAvailable.value =
                                dense.crossViewEvidenceAvailable
                            _denseLandmarkTemporalStability.value =
                                dense.temporalStability.coerceIn(0.0, 1.0)
                            _denseLandmarkContactCoverage.value =
                                dense.contactCoverage.coerceIn(0.0, 1.0)
                            _denseLandmarkPredictionBridgeRatio.value =
                                dense.predictionBridgeRatio.coerceIn(0.0, 1.0)
                            _denseLandmarkRejectedObservations.value =
                                dense.rejectedObservations
                            _denseLandmarkStaleObservations.value =
                                dense.staleObservations
                            _denseLandmarkNonFiniteObservations.value =
                                dense.nonFiniteObservations
                            _denseLandmarkHardFailure.value =
                                dense.hardFailure

                            val connectedCameraCount =
                                _devices.value.count { it.connected }
                                    .coerceAtLeast(1)
                            val customization =
                                _motionCustomization.value
                            denseLandmarkPrecisionContractEngine.configure(
                                denseLandmarkContractConfig(
                                    customization = customization,
                                    requireCrossView =
                                        connectedCameraCount >= 2,
                                ),
                            )
                            val contract =
                                denseLandmarkPrecisionContractEngine.evaluate(
                                    DenseLandmarkPrecisionContractEngine.Input(
                                        coverage = dense.coverage,
                                        visibleCoverage = dense.visibleCoverage,
                                        uncertaintyP95 =
                                            dense.uncertaintyP95,
                                        crossViewAgreement =
                                            dense.crossViewAgreement,
                                        crossViewEvidenceAvailable =
                                            dense.crossViewEvidenceAvailable,
                                        supportingCameraCount =
                                            connectedCameraCount,
                                        temporalStability =
                                            dense.temporalStability,
                                        contactCoverage =
                                            dense.contactCoverage,
                                        contactEvidenceActive =
                                            _denseContactProtectedLandmarkCount.value > 0,
                                        predictionBridgeRatio =
                                            dense.predictionBridgeRatio,
                                        acceptedObservations =
                                            dense.acceptedObservations,
                                        rejectedObservations =
                                            dense.rejectedObservations,
                                        staleObservations =
                                            dense.staleObservations,
                                        nonFiniteObservations =
                                            dense.nonFiniteObservations,
                                        hardFailure = dense.hardFailure,
                                    ),
                                )
                            _denseLandmarkContractScore.value =
                                contract.score
                            _denseLandmarkContractState.value =
                                contract.state.name
                            _denseLandmarkContractAuthoritative.value =
                                contract.authoritative
                            _denseLandmarkContractFailure.value =
                                contract.failureClass.name
                        }

                        globalDenseLandmarkEvidenceEngine.latest()?.let {
                            updateDenseLandmarkUiFromGlobal(
                                dense = it,
                                connectedCameraCount =
                                    _devices.value.count { device -> device.connected }
                                        .coerceAtLeast(1),
                            )
                        }

                        _motionScore.value = result.motion.motionScore
                        _commonModeMotionNormalizedPerSecond.value =
                            result.motion.commonModeMotionNormalizedPerSecond
                        _differentialMotionScore.value =
                            result.motion.differentialMotionScore
                        _cameraCompensatedMotionScore.value =
                            result.motion.cameraCompensatedMotionScore
                        _microMotionScore.value =
                            result.motion.microMotionScore
                        _directionChangeRatio.value =
                            result.motion.directionChangeRatio
                        _motionMeasurementUncertainty.value =
                            result.motion.meanMeasurementUncertaintyScore
                        _temporalStability.value = result.motion.temporalStability
                        _motionClass.value = result.motion.motionClass.name
                        _semanticMotionScore.value =
                            result.motionFusion.semanticMotion.globalEnergy
                        _cameraMotionRisk.value =
                            result.motionFusion.semanticMotion.cameraMotionRisk

                        val motionEvidence =
                            motionEvidenceEngine.evaluate(
                                MotionEvidenceFusionEngine.Input(
                                    timestampNanos = pose.timestampNanos,
                                    modelMotionScore =
                                        result.motion.motionScore.toDouble()
                                            .coerceIn(0.0, 1.0),
                                    modelTemporalStability =
                                        result.motion.temporalStability.toDouble()
                                            .coerceIn(0.0, 1.0),
                                    semanticMotionEnergy =
                                        result.motionFusion.semanticMotion
                                            .globalEnergy
                                            .coerceIn(0.0, 1.0),
                                    cameraMotionRisk =
                                        result.motionFusion.semanticMotion
                                            .cameraMotionRisk
                                            .coerceIn(0.0, 1.0),
                                    occlusionRisk =
                                        result.motionFusion.occlusionRisk
                                            .coerceIn(0.0, 1.0),
                                    subjectLockConfidence =
                                        result.subjectLock.globalConfidence
                                            .coerceIn(0.0, 1.0),
                                    poseConfidence =
                                        pose.confidence.toDouble()
                                            .coerceIn(0.0, 1.0),
                                    reconstructionConfidence =
                                        _reconstructionQualityScore.value
                                            .coerceIn(0.0, 1.0),
                                    captureTimingConfidence =
                                        _captureTimingConfidence.value
                                            .coerceIn(0.0, 1.0),
                                    synchronizationConfidence =
                                        _multiCameraSyncConfidence.value
                                            .coerceIn(0.0, 1.0),
                                    lightingConfidence =
                                        _lightingConfidence.value
                                            .coerceIn(0.0, 1.0),
                                ),
                            )

                        _motionEvidenceScore.value =
                            motionEvidence.score
                        _motionEvidenceConfidence.value =
                            motionEvidence.confidence
                        _motionEvidenceWarnings.value =
                            motionEvidence.warnings

                        recorder.appendMotionEvidence(motionEvidence)

                        _runtimeBand.value = result.runtimeDecision.band.name
                        _runtimeQualityScale.value = result.runtimeDecision.qualityScale
                        _runtimeInferenceScale.value = result.runtimeDecision.inferenceScale
                        _runtimeProtection.value = result.runtimeDecision.emergencyThermalProtection
                        _runtimeReasons.value = result.runtimeDecision.reasons

                        result.motionOrchestration?.let { orchestration ->
                            _identityContinuityScore.value =
                                orchestration.identityContinuityScore
                            _identitySwapRisk.value =
                                orchestration.identitySwapRisk
                            _identityAction.value =
                                orchestration.identityAction
                            _identityReason.value =
                                orchestration.identityReason
                        }

                        _trackLifecycleState.value =
                            result.trackLifecycle.dominantState.name
                        _trackLifecycleScore.value =
                            result.trackLifecycle.continuityScore
                        _trackLifecyclePredictedCount.value =
                            result.trackLifecycle.predictedTrackIds.size
                        _trackLifecycleRecoveringCount.value =
                            result.trackLifecycle.recoveringTrackIds.size
                        _trackLifecycleWarnings.value =
                            result.trackLifecycle.warnings

                        _subjectLockState.value =
                            result.subjectLock.lockState.name
                        _subjectLockConfidence.value =
                            result.subjectLock.globalConfidence
                        _subjectLockTrackId.value =
                            result.subjectLock.lockedTrackId
                        result.subjectLock.focus?.let {
                            _subjectFocusX.value = it.x
                            _subjectFocusY.value = it.y
                        }

                        result.faceEyeFusion?.let { fusion ->
                            recorder.appendFaceEyeFusion(fusion.state)

                            val bestEyeFrame =
                                result.eyes.maxByOrNull { eye ->
                                    eye.fixationConfidence
                                }
                            val bestEyeState =
                                bestEyeFrame?.let { eyeFrame ->
                                    val left = eyeFrame.left
                                    val right = eyeFrame.right
                                    val fixation = eyeFrame.fixationConfidence
                                        .coerceIn(0.0, 1.0)
                                    com.nivukx.nox.core.model.EyeState(
                                        timestampNanos = eyeFrame.timestampNanos,
                                        trackId =
                                            eyeFrame.trackingId?.toLong()
                                                ?: fusion.state.trackId,
                                        leftGazeCamera =
                                            left?.gazeDirection
                                                ?: eyeFrame.combinedGaze,
                                        rightGazeCamera =
                                            right?.gazeDirection
                                                ?: eyeFrame.combinedGaze,
                                        fusedGazeCamera =
                                            eyeFrame.combinedGaze,
                                        leftOpenness =
                                            left?.openness?.coerceIn(0.0, 1.0)
                                                ?: 0.0,
                                        rightOpenness =
                                            right?.openness?.coerceIn(0.0, 1.0)
                                                ?: 0.0,
                                        leftBlinkProbability =
                                            if (left?.blink == true) 1.0 else 0.0,
                                        rightBlinkProbability =
                                            if (right?.blink == true) 1.0 else 0.0,
                                        vergenceMeters = null,
                                        saccadeVelocityDegPerSecond = 0.0,
                                        gazeStability = fixation,
                                        confidence = fixation,
                                        band =
                                            when {
                                                fixation >= 0.90 ->
                                                    com.nivukx.nox.core.model.GazeConfidenceBand.EXCELLENT
                                                fixation >= 0.70 ->
                                                    com.nivukx.nox.core.model.GazeConfidenceBand.GOOD
                                                fixation >= 0.40 ->
                                                    com.nivukx.nox.core.model.GazeConfidenceBand.LIMITED
                                                else ->
                                                    com.nivukx.nox.core.model.GazeConfidenceBand.LOST
                                            },
                                        source =
                                            com.nivukx.nox.core.model.PerceptionSource.UNKNOWN,
                                        fresh = true,
                                    )
                                }

                            val face3dConfidence =
                                if (
                                    _face3DPoints.value > 0 &&
                                    _face3DErrorPx.value.isFinite()
                                ) {
                                    (
                                        kotlin.math.exp(
                                            -_face3DErrorPx.value /
                                                5.0,
                                        ) *
                                            (
                                                _face3DPoints.value /
                                                    68.0
                                            ).coerceIn(0.25, 1.0)
                                        ).coerceIn(0.0, 1.0)
                                } else {
                                    0.72
                                }

                            val faceEyeConsistency =
                                faceEyeConsistencyEngine.evaluate(
                                    FaceEyeConsistencyEngine.Input(
                                        face = fusion.state,
                                        eyes = bestEyeState,
                                        face3DConfidence =
                                            face3dConfidence,
                                        face3DErrorPx =
                                            if (_face3DErrorPx.value.isFinite()) {
                                                _face3DErrorPx.value
                                            } else {
                                                12.0
                                            },
                                        timestampNanos =
                                            fusion.state.timestampNanos,
                                    ),
                                )

                            val faceEyeWeight =
                                _motionCustomization.value.faceEyeFusionWeight
                                    .coerceIn(0.0, 1.0)

                            val combinedGazeConfidence =
                                (
                                    fusion.state.confidence * faceEyeWeight +
                                        faceEyeConsistency.confidence *
                                            (1.0 - faceEyeWeight)
                                    ).coerceIn(0.0, 1.0)

                            _faceEyeConsistencyConfidence.value =
                                faceEyeConsistency.confidence
                            _faceEyeConsistencyWarnings.value =
                                faceEyeConsistency.warnings
                            _gazeConfidence.value =
                                combinedGazeConfidence
                            _gazeStability.value =
                                minOf(
                                    fusion.state.gazeStability,
                                    faceEyeConsistency.temporalStability,
                                )
                            _faceHeadYaw.value =
                                fusion.state.headYawDegrees
                            _faceHeadRoll.value =
                                fusion.state.headRollDegrees
                        }

                        _semanticCoverage.value =
                            result.motionFusion.semanticCoverage
                        _attentionConfidence.value =
                            result.motionFusion.attentionConfidence
                        _occlusionRisk.value =
                            result.motionFusion.occlusionRisk
                        _dominantSubject.value =
                            result.motionFusion.dominantSubject
                                ?.type
                                ?.name
                                ?: "NONE"

                        _gpuDispatchMillis.value =
                            gpuAcceleration.telemetry().lastDispatchMillis

                        val currentMode =
                            if (
                                _calibrationReady.value &&
                                    _devices.value.count { it.connected } > 1
                            ) {
                                com.nivukx.nox.core.model.MotionCaptureMode.MULTI_VIEW_RGB
                            } else {
                                com.nivukx.nox.core.model.MotionCaptureMode.MONOCULAR_RGB
                            }

                        val syncSamples =
                            _devices.value
                                .filter { it.connected }
                                .map {
                                    val temporalTelemetry =
                                        multiCameraTemporalAlignmentEngine
                                            .telemetry(it.id)

                                    MultiCameraSynchronizationQualityEngine.PeerSample(
                                        deviceId = it.id,
                                        offsetNanos = it.clockOffsetNanos,
                                        roundTripNanos =
                                            it.roundTripNanos.coerceAtLeast(0L),
                                        packetLoss =
                                            it.packetLoss.coerceIn(0.0, 1.0),
                                        sampleAgeMillis = 0.0,
                                        subframeOffsetNanos =
                                            temporalTelemetry
                                                .subframeOffsetEwmaNanos,
                                        subframeConfidence =
                                            temporalTelemetry
                                                .subframeConfidence
                                                .takeIf { confidence ->
                                                    confidence > 0.0
                                                }
                                                ?: 1.0,
                                    )
                                }

                        val synchronization =
                            multiCameraSyncQualityEngine.evaluate(
                                timestampNanos = pose.timestampNanos,
                                samples = syncSamples,
                            )

                        val temporalAlignment =
                            evaluateTemporalAlignment(
                                connectedDevices =
                                    _devices.value
                                        .filter { it.connected }
                                        .map { it.id },
                            )

                        val temporalConfidence =
                            temporalAlignment?.confidence
                                ?.coerceIn(0.0, 1.0)
                                ?: 1.0

                        _multiCameraSyncConfidence.value =
                            minOf(
                                synchronization.confidence,
                                temporalConfidence,
                            )
                        _multiCameraSyncState.value =
                            synchronization.state.name

                        val schedulerTelemetry =
                            inferenceScheduler.telemetry()

                        val thermalScale =
                            when {
                                decision.thermalStatus >=
                                    android.os.PowerManager.THERMAL_STATUS_CRITICAL -> 0.25

                                decision.thermalStatus >=
                                    android.os.PowerManager.THERMAL_STATUS_SEVERE -> 0.25

                                decision.thermalStatus >=
                                    android.os.PowerManager.THERMAL_STATUS_MODERATE -> 0.55

                                decision.thermalStatus >=
                                    android.os.PowerManager.THERMAL_STATUS_LIGHT -> 0.80

                                else -> 1.0
                            }

                        val sessionHealth =
                            sessionHealthEngine.evaluate(
                                SessionHealthEngine.Input(
                                    timestampNanos = pose.timestampNanos,
                                    targetCaptureFps =
                                        activeCaptureProfile.fps.toDouble(),
                                    captureTimingConfidence =
                                        _captureTimingConfidence.value,
                                    captureTimingFps =
                                        _captureTimingFps.value,
                                    captureTimingJitterMillis =
                                        _captureTimingJitterMillis.value,
                                    captureTimingHealthy =
                                        _captureTimingHealthy.value,
                                    inferenceFps =
                                        decision.effectiveInferenceFps.toDouble(),
                                    inferenceLatencyMillis =
                                        schedulerTelemetry.inferenceEwmaMillis,
                                    droppedInferenceFrames =
                                        schedulerTelemetry.throttledFrames,
                                    capturedFrames =
                                        schedulerTelemetry.frameCount,
                                    subjectLockConfidence =
                                        result.subjectLock.globalConfidence,
                                    perceptionConfidence =
                                        result.motionFusion.attentionConfidence,
                                    reconstructionConfidence =
                                        _reconstructionQualityScore.value
                                            .coerceIn(0.0, 1.0),
                                    reconstructionReprojectionErrorPx =
                                        if (_reconstructionErrorPx.value.isFinite()) {
                                            _reconstructionErrorPx.value.coerceAtLeast(0.0)
                                        } else {
                                            999.0
                                        },
                                    temporalStability =
                                        result.motion.temporalStability,
                                    motionEvidenceConfidence =
                                        _motionEvidenceConfidence.value
                                            .coerceIn(0.0, 1.0),
                                    motionEvidenceDegraded =
                                        _motionEvidenceWarnings.value.isNotEmpty(),
                                    geometricObservabilityConfidence =
                                        _geometricObservabilityScore.value
                                            .coerceIn(0.0, 1.0),
                                    geometricObservabilityDegraded =
                                        _geometricObservabilityWarnings.value.isNotEmpty(),
                                    calibrationConfidence =
                                        _rigScore.value.coerceIn(0.0, 1.0),
                                    multiCameraSyncConfidence =
                                        synchronization.confidence,
                                    cameraMotionRisk =
                                        result.motionFusion.semanticMotion.cameraMotionRisk,
                                    occlusionRisk =
                                        result.motionFusion.occlusionRisk,
                                    meshConfidence =
                                        _denseMeshConfidence.value.coerceIn(0.0, 1.0),
                                    faceConfidence =
                                        _gazeConfidence.value.coerceIn(0.0, 1.0),
                                    eyeConfidence =
                                        _gazeConfidence.value.coerceIn(0.0, 1.0),
                                    eyeTemporalConfidence =
                                        _eye3DTemporalConsistencyConfidence.value
                                            .coerceIn(0.0, 1.0),
                                    eyeTemporalDegraded =
                                        _eye3DTemporalConsistencyWarnings.value.isNotEmpty(),
                                    handConfidence =
                                        _hand3DConfidence.value
                                            .coerceIn(0.0, 1.0),
                                    thermalScale = thermalScale,
                                    memoryScale =
                                        (
                                            1.0 -
                                                decision.heapPressure
                                        ).coerceIn(0.0, 1.0),
                                ),
                            )

                        _sessionHealthState.value =
                            sessionHealth.state.name
                        _sessionHealthScore.value =
                            sessionHealth.score
                        _sessionHealthBottleneck.value =
                            sessionHealth.bottleneck?.label ?: "NONE"
                        _sessionHealthWarnings.value =
                            (
                                sessionHealth.hardFailures +
                                    sessionHealth.warnings
                                )
                                .distinct()
                                .take(6)

                        recorder.appendSessionHealth(
                            sessionHealth,
                        )

                        val preflight = preflightEngine.evaluate(
                            ProductionCapturePreflightEngine.Input(
                                mode = currentMode,
                                cameraAvailable = true,
                                modelPipelineReady = true,
                                calibrationReady = _calibrationReady.value,
                                calibrationConfidence = _rigScore.value,
                                multiCameraSyncConfidence =
                                    synchronization.confidence,
                                rigHealthConfidence =
                                    if (
                                        _rigHealthState.value ==
                                            MultiCameraRigHealthEngine.State.UNKNOWN.name
                                    ) {
                                        1.0
                                    } else {
                                        _rigHealthScore.value
                                            .coerceIn(0.0, 1.0)
                                    },
                                rigHealthHealthy =
                                    _rigHealthState.value !=
                                        MultiCameraRigHealthEngine.State.HOLD.name &&
                                        _rigHealthState.value !=
                                        MultiCameraRigHealthEngine.State.BLOCKED.name,
                                geometricObservabilityConfidence =
                                    _geometricObservabilityScore.value
                                        .coerceIn(0.0, 1.0),
                                geometricObservabilityHealthy =
                                    _geometricObservabilityWarnings.value.isEmpty() &&
                                        _geometricObservabilityScore.value >= 0.52,
                                reconstructionUncertaintyConfidence =
                                    _reconstructionUncertaintyConfidence.value
                                        .coerceIn(0.0, 1.0),
                                reconstructionUncertaintyHealthy =
                                    _reconstructionUncertaintyWarnings.value.isEmpty() &&
                                        _reconstructionUncertaintyConfidence.value >= 0.55,
                                thermalScale =
                                    result.runtimeDecision.qualityScale
                                        .coerceIn(0.0, 1.0),
                                memoryScale =
                                    (
                                        1.0 -
                                            decision.heapPressure
                                    ).coerceIn(0.0, 1.0),
                                storageAvailable =
                                    getApplication<Application>()
                                        .filesDir
                                        .usableSpace > 128L * 1024L * 1024L,
                                sessionPolicyReady = true,
                                subjectVisible = result.subjectCount > 0,
                                motionQuality = result.motion.motionScore
                                    .coerceIn(0.0, 1.0),
                                subjectLockConfidence =
                                    result.subjectLock.globalConfidence,
                                cameraMotionRisk =
                                    result.motionFusion.semanticMotion.cameraMotionRisk,
                                captureTimingConfidence =
                                    _captureTimingConfidence.value,
                                captureTimingFps =
                                    _captureTimingFps.value,
                                captureTimingJitterMillis =
                                    _captureTimingJitterMillis.value,
                                captureTimingHealthy =
                                    _captureTimingHealthy.value,
                                pipelineError = null,
                                rawSidecarEnabled = false,
                                depthSidecarEnabled = false,
                                faceEyeMetricReady =
                                    _faceEyeMetricState.value ==
                                        FaceEyeMetricEvidenceEngine.State.METRIC_READY.name,
                                faceEyeMetricConfidence =
                                    _faceEyeMetricConfidence.value
                                        .coerceIn(0.0, 1.0),
                                hardwareReadoutTimingAvailable =
                                    camera.hardwareReadoutTimestampSupported,
                                hardwareReadoutTimingConfidence =
                                    _hardwareReadoutTimingConfidence.value
                                        .coerceIn(0.0, 1.0),
                                hardwareReadoutTimingHealthy =
                                    _hardwareReadoutTimingState.value ==
                                        HardwareReadoutTimingEvidenceEngine.State.STABLE.name ||
                                        !camera.hardwareReadoutTimestampSupported,
                            ),
                        )
                        _captureReadiness.value = preflight.readiness

                        inferenceScheduler.updateMotionContext(
                            result.motion,
                        )

                        recorder.appendPose(
                            timestampNanos = pose.timestampNanos,
                            confidence = pose.confidence,
                            landmarks = pose.landmarks.map { Triple(it.name, it.x, it.y) },
                        )

                        val fusionObservations =
                            if (result.actorTracks.isNotEmpty()) {
                                val actorWidth =
                                    result.multiActorPose
                                        ?.inputWidth
                                        ?.coerceAtLeast(1)
                                        ?: pose.imageWidth.coerceAtLeast(1)
                                val actorHeight =
                                    result.multiActorPose
                                        ?.inputHeight
                                        ?.coerceAtLeast(1)
                                        ?: pose.imageHeight.coerceAtLeast(1)

                                result.actorTracks
                                    .take(
                                        _motionCustomization.value.maxSubjects
                                            .coerceIn(1, 64),
                                    )
                                    .flatMap { actor ->
                                        actor.joints
                                            .take(64)
                                            .map { joint ->
                                                com.nivukx.nox.core.model.LandmarkObservation(
                                                    landmarkId = joint.landmarkId,
                                                    x =
                                                        joint.x *
                                                            actorWidth,
                                                    y =
                                                        joint.y *
                                                            actorHeight,
                                                    confidence =
                                                        joint.confidence
                                                            .coerceIn(0.0, 1.0),
                                                    timestampNanos =
                                                        pose.timestampNanos,
                                                    frameWidth = actorWidth,
                                                    frameHeight = actorHeight,
                                                    space =
                                                        com.nivukx.nox.core.model.LandmarkSpace.PIXELS,
                                                    trackingId =
                                                        actor.actorId
                                                            .coerceAtMost(
                                                                Int.MAX_VALUE.toLong(),
                                                            )
                                                            .toInt(),
                                                )
                                            }
                                    }
                            } else {
                                pose.observations()
                            }

                        fusion.push(
                            localId,
                            pose.timestampNanos,
                            fusionObservations,
                        )

                        val multiActor3D =
                            if (
                                result.actorTracks.isNotEmpty() &&
                                (
                                    currentMode ==
                                        MotionCaptureMode.MULTI_VIEW_RGB ||
                                        currentMode ==
                                            MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
                                    )
                            ) {
                                multiActor3DReconstructionCoordinator.process(
                                    timestampNanos = pose.timestampNanos,
                                    actors = result.actorTracks,
                                    allocation =
                                        result.detailAllocation,
                                    customization =
                                        _motionCustomization.value,
                                    primaryActorId =
                                        result.subjectLock.lockedTrackId
                                            ?.takeIf { lockedId ->
                                                result.actorTracks.any {
                                                    it.actorId == lockedId
                                                }
                                            }
                                            ?: result.actorTracks
                                                .maxWithOrNull(
                                                    compareBy<
                                                        com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine.Result
                                                    > {
                                                        it.confidence
                                                    }.thenByDescending {
                                                        it.joints.size
                                                    }.thenBy {
                                                        it.actorId
                                                    },
                                                )
                                                ?.actorId,
                                    minimumActorIntervalMillis =
                                        (
                                            1_000.0 /
                                                _motionCustomization.value.targetFps
                                                    .coerceIn(15, 240)
                                            )
                                            .toLong()
                                            .coerceAtLeast(8L),
                                )
                            } else {
                                null
                            }

                        multiActor3D?.let { actorResult ->
                            _multiPerson3DReconstructedCount.value =
                                actorResult.reconstructedActorCount
                            _multiPerson3DDetailedCount.value =
                                actorResult.detailedActorCount
                            _multiPerson3DPredictedCount.value =
                                actorResult.actors.count { it.predicted }
                            _multiPerson3DPredictionMaxAgeMillis.value =
                                actorResult.actors
                                    .filter { it.predicted }
                                    .maxOfOrNull { it.predictionAgeMillis }
                                    ?: 0L

                            actorResult.actors
                                .filter { it.predicted && it.skeleton != null }
                                .forEach { actor ->
                                    recorder.appendPredictedReconstruction(
                                        frame = actor.skeleton!!,
                                        predictionAgeMillis =
                                            actor.predictionAgeMillis,
                                        predictionConfidence =
                                            actor.predictionConfidence,
                                        warnings = actor.warnings,
                                    )
                                }
                        }

                        multiActor3D?.actors
                            ?.filter {
                                it.actorId !=
                                    multiActor3D.primaryActorId
                            }
                            ?.mapNotNull {
                                it.detailed?.skeleton ?: it.skeleton
                            }
                            ?.forEach {
                                recorder.appendReconstruction(it)
                            }

                        val detailed =
                            multiActor3D?.primaryDetailed
                                ?: fusion.reconstructDetailed(
                                    timestampNanos = pose.timestampNanos,
                                    trackId =
                                        result.actorTracks
                                            .firstOrNull()
                                            ?.actorId
                                            ?: result.faceTracks
                                                .firstOrNull()
                                                ?.id
                                            ?: 0L,
                            clockSkewNanos = _skewNanos.value,
                            cameraMotionRisk =
                                result.motionFusion.semanticMotion.cameraMotionRisk,
                            synchronizationConfidence =
                                _multiCameraSyncConfidence.value,
                            captureTimingConfidence =
                                _captureTimingConfidence.value,
                            faceEyeFusionConfidence =
                                _faceEyeFusionConfidence.value
                                    .coerceIn(0.0, 1.0),
                            faceEyeFusionAccepted =
                                _faceEyeFusionAccepted.value,
                            faceEyeOutOfDistributionRisk =
                                _faceEyeOutOfDistributionRisk.value
                                    .coerceIn(0.0, 1.0),
                            denseContactEvidenceConfidence =
                                result.denseContactEvidence
                                    ?.contactConfidence
                                    ?.coerceIn(0.0, 1.0)
                                    ?: 1.0,
                            denseContactEvidenceAvailable =
                                result.denseContactEvidence
                                    ?.protectedLandmarks
                                    ?.isNotEmpty() == true,
                        )

                        if (detailed != null) {
                            _reconstructionJoints.value = detailed.skeleton.joints.size
                            _reconstructionErrorPx.value =
                                detailed.diagnostics.meanReprojectionErrorPx
                            _reconstructionQualityScore.value =
                                detailed.qualityScore
                            _bodyMetricEvidenceAuthority.value =
                                detailed.bodyMetricEvidenceAuthority
                                    .coerceIn(0.0, 1.0)
                            _bodySymmetricEpipolarConsistency.value =
                                detailed.bodySymmetricEpipolarConsistencyScore
                                    .coerceIn(0.0, 1.0)
                            _reconstructionGateReasons.value =
                                detailed.rejectionReasons
                            _geometricObservabilityScore.value =
                                detailed.geometricObservabilityScore
                            _geometricObservabilityWarnings.value =
                                detailed.geometricObservabilityWarnings
                            _geometricInformationConditionScore.value =
                                detailed.geometricInformationConditionScore
                            _geometricInformationP95MajorMeters.value =
                                detailed.geometricInformationP95MajorMeters
                            _geometricInformationP95DepthMeters.value =
                                detailed.geometricInformationP95DepthMeters
                            _denseIdentityGateQuality.value =
                                detailed.denseIdentityGateQuality
                                    .coerceIn(0.0, 1.0)
                            _denseIdentityGateRejectedCount.value =
                                detailed.denseIdentityGateRejectedCount
                            _denseIdentityGateEvaluatedPoints.value =
                                detailed.denseIdentityGateEvaluatedPoints
                            _denseIdentityGateWarnings.value =
                                detailed.denseIdentityGateWarnings
                            _denseGeometricConditioningScore.value =
                                detailed.denseGeometricConditioningScore
                                    .coerceIn(0.0, 1.0)
                            _denseGeometricConditioningMedianParallaxDegrees.value =
                                detailed.denseGeometricConditioningMedianParallaxDegrees
                            _denseGeometricConditioningMinimumObservationScore.value =
                                detailed.denseGeometricConditioningMinimumObservationScore
                                    .coerceIn(0.0, 1.0)
                                    .coerceAtLeast(0.0)
                            _denseGeometricConditioningWarnings.value =
                                detailed.denseGeometricConditioningWarnings
                            _reconstructionUncertaintyConfidence.value =
                                detailed.uncertaintyConfidence
                            _reconstructionUncertaintyMeanMeters.value =
                                detailed.meanUncertaintyMeters
                            _reconstructionUncertaintyP95Meters.value =
                                detailed.p95UncertaintyMeters
                            _reconstructionTrustedJointRatio.value =
                                detailed.trustedJointRatio
                            _reconstructionUncertaintyBottleneck.value =
                                detailed.uncertaintyBottleneck
                            _reconstructionUncertaintyWarnings.value =
                                detailed.uncertaintyWarnings
                            _directionalUncertaintyMajor95Meters.value =
                                detailed.directionalUncertaintyMajor95Meters
                            _directionalUncertaintyDepthMeters.value =
                                detailed.directionalUncertaintyDepthMeters
                            _directionalUncertaintyAnisotropy.value =
                                detailed.directionalUncertaintyAnisotropy
                                    .coerceAtLeast(1.0)
                            _directionalUncertaintyConfidence.value =
                                detailed.directionalUncertaintyConfidence
                                    .coerceIn(0.0, 1.0)
                            _directionalUncertaintyDegradedJointCount.value =
                                detailed.directionalUncertaintyDegradedJointCount
                            _directionalUncertaintyWarnings.value =
                                detailed.directionalUncertaintyWarnings
                            _adaptiveViewEvidenceScore.value =
                                detailed.adaptiveViewEvidenceScore
                            _adaptiveViewEvidenceGlobalRigTrust.value =
                                detailed.adaptiveViewEvidenceGlobalRigTrust
                            _adaptiveViewEvidenceDegradedCameraCount.value =
                                detailed.adaptiveViewEvidenceDegradedCameraCount
                            _adaptiveViewEvidenceHoldCameraCount.value =
                                detailed.adaptiveViewEvidenceHoldCameraCount
                            _adaptiveViewEvidenceDominantCamera.value =
                                detailed.adaptiveViewEvidenceDominantCamera
                            _adaptiveViewEvidenceWarnings.value =
                                detailed.adaptiveViewEvidenceWarnings
                            _cameraContributionSelectedCount.value =
                                detailed.cameraContributionSelectedCount
                            _cameraContributionExcludedCount.value =
                                detailed.cameraContributionExcludedCount
                            _cameraContributionDiversitySatisfied.value =
                                detailed.cameraContributionDiversitySatisfied
                            _cameraContributionReason.value =
                                detailed.cameraContributionReason
                            _cameraContributionInformationGainScore.value =
                                detailed.cameraContributionInformationGainScore
                                    .coerceIn(0.0, 1.0)
                            _jointViewSelectionScore.value =
                                detailed.jointViewSelectionScore
                            _jointViewDiversitySatisfiedRatio.value =
                                detailed.jointViewDiversitySatisfiedRatio
                            _jointSelectedObservationCount.value =
                                detailed.jointSelectedObservationCount
                            _jointExcludedObservationCount.value =
                                detailed.jointExcludedObservationCount
                            _jointViewSelectionReasons.value =
                                detailed.jointViewSelectionReasons
                            val rigHealth =
                                rigHealthEngine.evaluate(
                                    MultiCameraRigHealthEngine.Input(
                                        devices = _devices.value,
                                        calibrationReady =
                                            _calibrationReady.value,
                                        calibrationConfidence =
                                            _rigScore.value.coerceIn(0.0, 1.0),
                                        synchronizationConfidence =
                                            _multiCameraSyncConfidence.value
                                                .coerceIn(0.0, 1.0),
                                        captureTimingConfidence =
                                            _captureTimingConfidence.value
                                                .coerceIn(0.0, 1.0),
                                        cameraStatuses =
                                            detailed.adaptiveViewEvidenceCameraStatuses,
                                    ),
                                )

                            _rigHealthState.value =
                                rigHealth.state.name
                            _rigHealthScore.value =
                                rigHealth.overallScore
                            _rigQuarantinedCameraCount.value =
                                rigHealth.quarantinedCameraCount
                            _adaptiveViewEvidenceGlobalRigTrust.value =
                                minOf(
                                    _adaptiveViewEvidenceGlobalRigTrust.value
                                        .coerceIn(0.0, 1.0),
                                    rigHealth.overallScore
                                        .coerceIn(0.0, 1.0),
                                )
                            _rigHealthWarnings.value =
                                rigHealth.warnings
                            _rigCameraHealth.value =
                                rigHealth.cameras
                            _spatiotemporalConfidence.value =
                                detailed.spatiotemporalConfidence
                            _spatiotemporalInitialRmsPx.value =
                                detailed.spatiotemporalInitialRmsPx
                            _spatiotemporalFinalRmsPx.value =
                                detailed.spatiotemporalFinalRmsPx
                            _spatiotemporalOffsetsNanos.value =
                                detailed.spatiotemporalOffsetsNanos
                            _spatiotemporalWarnings.value =
                                detailed.spatiotemporalWarnings

                            val bundleTemporalConfidence =
                                detailed.spatiotemporalConfidence
                                    .coerceIn(0.0, 1.0)

                            detailed.spatiotemporalOffsetsNanos
                                .forEach { (cameraIdValue, offsetNanos) ->
                                    if (
                                        offsetNanos.isFinite() &&
                                        cameraIdValue.isNotBlank()
                                    ) {
                                        runCatching {
                                            multiCameraTemporalAlignmentEngine
                                                .applySubframeOffsetEvidence(
                                                    deviceId =
                                                        DeviceId(
                                                            cameraIdValue,
                                                        ),
                                                    offsetNanos =
                                                        offsetNanos,
                                                    confidence =
                                                        bundleTemporalConfidence,
                                                )
                                        }
                                    }
                                }

                            if (
                                _devices.value.count {
                                    it.connected
                                } > 1
                            ) {
                                _multiCameraSyncConfidence.value =
                                    minOf(
                                        _multiCameraSyncConfidence.value,
                                        bundleTemporalConfidence,
                                    )

                                if (
                                    bundleTemporalConfidence <
                                        0.45
                                ) {
                                    _multiCameraSyncState.value =
                                        "DEGRADED"
                                }
                            }

                            _kinematicConstraintScore.value =
                                detailed.kinematicScore.coerceIn(0.0, 1.0)
                            _kinematicConstraintAccepted.value =
                                detailed.kinematicAccepted
                            _kinematicConstraintBottleneck.value =
                                detailed.kinematicBottleneck
                            _kinematicConstraintWarnings.value =
                                detailed.kinematicWarnings
                            _physicsConstraintScore.value =
                                detailed.physicsScore.coerceIn(0.0, 1.0)
                            _physicsConstraintAccepted.value =
                                detailed.physicsAccepted
                            _physicsConstraintMeanBoneResidualMeters.value =
                                detailed.physicsMeanBoneResidualMeters
                                    .takeIf(Double::isFinite)
                                    ?: 0.0
                            _physicsConstraintP95BoneResidualMeters.value =
                                detailed.physicsP95BoneResidualMeters
                                    .takeIf(Double::isFinite)
                                    ?: 0.0
                            _physicsConstraintMaxBoneResidualMeters.value =
                                detailed.physicsMaxBoneResidualMeters
                                    .takeIf(Double::isFinite)
                                    ?: 0.0
                            _physicsConstraintSymmetryScore.value =
                                detailed.physicsSymmetryScore.coerceIn(0.0, 1.0)
                            _physicsConstraintFloorContactScore.value =
                                detailed.physicsFloorContactScore.coerceIn(0.0, 1.0)
                            _physicsConstraintCorrectedJointCount.value =
                                detailed.physicsCorrectedJointCount
                            _physicsConstraintConstrainedEdgeCount.value =
                                detailed.physicsConstrainedEdgeCount
                            _physicsConstraintWarnings.value =
                                detailed.physicsWarnings
                            _worldSpaceConsistencyScore.value =
                                detailed.worldSpaceConsistencyScore
                                    .coerceIn(0.0, 1.0)
                            _worldSpaceAccepted.value =
                                detailed.worldSpaceAccepted
                            _worldSpaceTemporalStability.value =
                                detailed.worldSpaceTemporalStability
                                    .coerceIn(0.0, 1.0)
                            _worldSpaceMotionContinuity.value =
                                detailed.worldSpaceMotionContinuity
                                    .coerceIn(0.0, 1.0)
                            _worldSpaceRejectedJointCount.value =
                                detailed.worldSpaceRejectedJointCount
                            _worldSpaceRobustOutlierRatio.value =
                                detailed.worldSpaceRobustOutlierRatio
                                    .coerceIn(0.0, 1.0)
                            _worldSpaceWarnings.value =
                                detailed.worldSpaceWarnings
                            _global4DScore.value =
                                detailed.global4DScore.coerceIn(0.0, 1.0)
                            _global4DAccepted.value =
                                detailed.global4DAccepted
                            _global4DState.value =
                                detailed.global4DState.name
                            _global4DPersistentFailure.value =
                                detailed.global4DPersistentFailure
                            _global4DRootMotionConsistency.value =
                                detailed.global4DRootMotionConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DSceneConsistency.value =
                                detailed.global4DSceneConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DDirectionalUncertaintyConsistency.value =
                                detailed.global4DDirectionalUncertaintyConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DFaceEyeConsistency.value =
                                detailed.global4DFaceEyeConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DExternalContactConsistency.value =
                                detailed.global4DExternalContactConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DGaussianConsistency.value =
                                detailed.global4DGaussianConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DGaussianStrainConsistency.value =
                                detailed.global4DGaussianStrainConsistency
                                    .coerceIn(0.0, 1.0)
                            _global4DGaussianEvidenceAvailable.value =
                                detailed.global4DGaussianEvidenceAvailable
                            _global4DWarnings.value =
                                detailed.global4DWarnings

                            configureTemporalMotionEstimator()
                            val temporalMotion =
                                motionStateEstimator.estimate(
                                    HighEndMotionStateEstimator.Input(
                                        timestampNanos = detailed.skeleton.timestampNanos,
                                        skeleton = detailed.skeleton,
                                        cameraMotionRisk =
                                            _cameraMotionRisk.value.coerceIn(0.0, 1.0),
                                        occlusionRisk =
                                            _occlusionRisk.value.coerceIn(0.0, 1.0),
                                        synchronizationConfidence =
                                            _multiCameraSyncConfidence.value.coerceIn(0.0, 1.0),
                                        preserveMicroMotion =
                                            _motionCustomization.value.preserveMicroMotion,
                                    ),
                                )
                            configureProductionMotionDynamics()
                            val motionSkeleton = temporalMotion.skeleton
                            val motionDynamics =
                                productionMotionDynamicsEngine.analyze(
                                    motionSkeleton,
                                )
                            _motionDynamicsPhase.value =
                                motionDynamics.phase.name
                            _motionDynamicsScore.value =
                                motionDynamics.globalScore
                            _motionContactStability.value =
                                motionDynamics.lowerBodyContactStability
                            _motionSymmetryScore.value =
                                motionDynamics.symmetryScore
                            _motionDynamicStress.value =
                                motionDynamics.dynamicStressScore
                            _motionDynamicsUncertainty.value =
                                1.0 - motionDynamics.uncertaintyScore
                            _motionTrustedJointRatio.value =
                                motionDynamics.trustedJointRatio
                            _motionMicroMotionRetention.value =
                                motionDynamics.microMotionRetention
                            _motionLowFrequencyScore.value =
                                motionDynamics.lowFrequencyMotionScore
                                    .coerceIn(0.0, 1.0)
                            _motionHighFrequencyScore.value =
                                motionDynamics.highFrequencyMotionScore
                                    .coerceIn(0.0, 1.0)
                            _motionTemporalMemoryScore.value =
                                temporalMotion.diagnostics
                                    .meanTemporalMemoryScore
                                    .coerceIn(0.0, 1.0)
                            if (_state.value == CaptureState.RECORDING) {
                                recorder.appendMotionDynamics(
                                    motionDynamics,
                                )
                            }
                            applyCalibratedMotionUncertainty(
                                skeleton = motionSkeleton,
                            )
                            updateCrossModalIdentity(
                                skeleton = motionSkeleton,
                                timestampNanos = motionSkeleton.timestampNanos,
                            )
                            publishLiveMotionPreview(
                                result = result,
                                skeleton = motionSkeleton,
                                cameraWidth = image.width,
                                cameraHeight = image.height,
                                timestampNanos = motionSkeleton.timestampNanos,
                            )
                            val temporalQuality =
                                (
                                    detailed.qualityScore * 0.70 +
                                        temporalMotion.diagnostics.finalScore * 0.20 +
                                        temporalMotion.diagnostics.meanTemporalMemoryScore * 0.10
                                    ).coerceIn(0.0, 1.0)

                            _reconstructionQualityScore.value =
                                (_reconstructionQualityScore.value * 0.82 + temporalQuality * 0.18)
                                    .coerceIn(0.0, 1.0)
                            _reconstructionGateReasons.value =
                                (
                                    _reconstructionGateReasons.value +
                                        temporalMotion.diagnostics.warnings
                                    )
                                    .distinct()
                                    .take(12)

                            recorder.appendReconstruction(motionSkeleton)
                            recorder.appendReconstructionQuality(detailed)
                            recordSemanticMotionEvent(
                                skeleton = motionSkeleton,
                                confidence = temporalQuality,
                            )
                            retargeter.retarget(motionSkeleton)?.let { avatar ->
                                appendAvatarIfSafe(
                                    skeleton = motionSkeleton,
                                    avatar = avatar,
                                    confidence = detailed.qualityScore,
                                    reprojectionErrorPx =
                                        detailed.diagnostics.meanReprojectionErrorPx,
                                    uncertaintyConfidence =
                                        detailed.uncertaintyConfidence,
                                    p95UncertaintyMeters =
                                        detailed.p95UncertaintyMeters,
                                    trustedJointRatio =
                                        detailed.trustedJointRatio,
                                    uncertaintyBottleneck =
                                        detailed.uncertaintyBottleneck,
                                     thermalScale = thermalScale,
                                     memoryScale =
                                         (
                                             1.0 -
                                                 decision.heapPressure
                                         ).coerceIn(0.0, 1.0),
                                     runtimeQualityScale =
                                         result.runtimeDecision.qualityScale
                                             .coerceIn(0.0, 1.0),
                                )
                            }
                        } else {
                            if (
                                currentMode == MotionCaptureMode.MULTI_VIEW_RGB ||
                                    currentMode == MotionCaptureMode.MULTI_VIEW_RGB_DEPTH
                            ) {
                                _reconstructionUncertaintyConfidence.value = 0.0
                                _reconstructionUncertaintyMeanMeters.value =
                                    Double.POSITIVE_INFINITY
                                _reconstructionUncertaintyP95Meters.value =
                                    Double.POSITIVE_INFINITY
                                _reconstructionTrustedJointRatio.value = 0.0
                                _reconstructionUncertaintyBottleneck.value =
                                    "RECONSTRUCTION_UNAVAILABLE"
                                _reconstructionUncertaintyWarnings.value =
                                    listOf(
                                        "multi-view-uncertainty-unavailable",
                                    )
                            }
                            val monoIntrinsics = runCatching {
                                camera.intrinsicsModel(
                                    cameraId = cameraId,
                                    imageWidth = pose.imageWidth,
                                    imageHeight = pose.imageHeight,
                                )
                            }.getOrNull()

                            if (monoIntrinsics != null) {
                                val dominantType =
                                    result.motionFusion.dominantSubject?.type
                                        ?: com.nivukx.nox.core.model.SubjectType.HUMAN
                                val mono = runCatching {
                                    monocularPipeline.reconstruct(
                                        MonocularMotionReconstructionPipeline.Input(
                                            timestampNanos = pose.timestampNanos,
                                            subjectType = dominantType,
                                            trackId =
                                                result.subjectLock.lockedTrackId
                                                    ?: result.motionFusion.dominantSubject?.trackingId
                                                    ?: 0L,
                                            pose = pose,
                                            intrinsics = monoIntrinsics,
                                            denseDepth = null,
                                            previous = previousMonocularFrame,
                                            calibratedScaleMeters = null,
                                            captureTimingConfidence =
                                                _captureTimingConfidence.value,
                                            lightingConfidence =
                                                _lightingConfidence.value,
                                            cameraMotionRisk =
                                                result.motionFusion.semanticMotion.cameraMotionRisk,
                                            cameraMotionSuppression =
                                                _motionCustomization.value.cameraMotionSuppression,
                                            microMotionPreservation =
                                                _motionCustomization.value.microMotionPreservation,
                                            preserveMicroMotion =
                                                _motionCustomization.value.preserveMicroMotion,
                                            temporalEvidenceScale =
                                                (
                                                    0.45 +
                                                        _motionCustomization.value.worldConsistencyDetail *
                                                        0.80
                                                    ).coerceIn(0.25, 1.50),
                                            topologyConstraintScale =
                                                (
                                                    0.25 +
                                                        _motionCustomization.value.topologyConstraintStrength *
                                                        1.25
                                                    ).coerceIn(0.25, 1.50),
                                        ),
                                    )
                                }.getOrNull()

                                if (mono != null) {
                                    previousMonocularFrame = mono.monocular

                                    val kinematic =
                                        monocularKinematicConstraintEngine.solve(
                                            HighEndKinematicConstraintEngine.Input(
                                                skeleton = mono.skeleton,
                                                preserveMicroMotion = true,
                                                confidenceHint =
                                                    mono.quality.confidence
                                                        .coerceIn(0.0, 1.0),
                                                cameraMotionRisk =
                                                    result.motionFusion.semanticMotion
                                                        .cameraMotionRisk
                                                        .coerceIn(0.0, 1.0),
                                                occlusionRisk =
                                                    result.motionFusion.occlusionRisk
                                                        .coerceIn(0.0, 1.0),
                                                synchronizationConfidence = 1.0,
                                                rootJointId = mono.skeleton.joints
                                                    .firstOrNull { it.landmarkId == 23 }
                                                    ?.landmarkId,
                                            ),
                                        )
                                    val constrainedMonoSkeleton =
                                        kinematic.skeleton
                                    val refinedMonoSkeleton =
                                        monocularTrajectoryOptimizer.optimize(
                                            constrainedMonoSkeleton,
                                        )

                                    val physicsMono =
                                        monocularPhysicsConstraintEngine.refine(
                                            refinedMonoSkeleton,
                                        )
                                    val physicsMonoSkeleton =
                                        physicsMono.frame
                                    _physicsConstraintScore.value =
                                        physicsMono.physicsScore.coerceIn(0.0, 1.0)
                                    _physicsConstraintAccepted.value =
                                        physicsMono.accepted
                                    _physicsConstraintMeanBoneResidualMeters.value =
                                        physicsMono.meanBoneResidualMeters
                                            .takeIf(Double::isFinite)
                                            ?: 0.0
                                    _physicsConstraintP95BoneResidualMeters.value =
                                        physicsMono.p95BoneResidualMeters
                                            .takeIf(Double::isFinite)
                                            ?: 0.0
                                    _physicsConstraintMaxBoneResidualMeters.value =
                                        physicsMono.maxBoneResidualMeters
                                            .takeIf(Double::isFinite)
                                            ?: 0.0
                                    _physicsConstraintSymmetryScore.value =
                                        physicsMono.symmetryScore.coerceIn(0.0, 1.0)
                                    _physicsConstraintFloorContactScore.value =
                                        physicsMono.floorContactScore.coerceIn(0.0, 1.0)
                                    _physicsConstraintCorrectedJointCount.value =
                                        physicsMono.correctedJointCount
                                    _physicsConstraintConstrainedEdgeCount.value =
                                        physicsMono.constrainedEdgeCount
                                    _physicsConstraintWarnings.value =
                                        physicsMono.warnings

                                    configureTemporalMotionEstimator()
                            val temporalMotion =
                                        motionStateEstimator.estimate(
                                            HighEndMotionStateEstimator.Input(
                                                timestampNanos = physicsMonoSkeleton.timestampNanos,
                                                skeleton = physicsMonoSkeleton,
                                                cameraMotionRisk =
                                                    result.motionFusion.semanticMotion.cameraMotionRisk
                                                        .coerceIn(0.0, 1.0),
                                                occlusionRisk =
                                                    result.motionFusion.occlusionRisk
                                                        .coerceIn(0.0, 1.0),
                                                synchronizationConfidence = 1.0,
                                                preserveMicroMotion =
                                                    _motionCustomization.value.preserveMicroMotion,
                                            ),
                                        )
                                    configureProductionMotionDynamics()
                                    val finalMotionSkeleton = temporalMotion.skeleton
                                    val motionDynamics =
                                        productionMotionDynamicsEngine.analyze(
                                            finalMotionSkeleton,
                                        )
                                    _motionDynamicsPhase.value =
                                        motionDynamics.phase.name
                                    _motionDynamicsScore.value =
                                        motionDynamics.globalScore
                                    _motionContactStability.value =
                                        motionDynamics.lowerBodyContactStability
                                    _motionSymmetryScore.value =
                                        motionDynamics.symmetryScore
                                    _motionDynamicStress.value =
                                        motionDynamics.dynamicStressScore
                                    _motionDynamicsUncertainty.value =
                                        1.0 - motionDynamics.uncertaintyScore
                                    _motionTrustedJointRatio.value =
                                        motionDynamics.trustedJointRatio
                                    _motionMicroMotionRetention.value =
                                        motionDynamics.microMotionRetention
                                    _motionLowFrequencyScore.value =
                                        motionDynamics.lowFrequencyMotionScore
                                            .coerceIn(0.0, 1.0)
                                    _motionHighFrequencyScore.value =
                                        motionDynamics.highFrequencyMotionScore
                                            .coerceIn(0.0, 1.0)
                                    _motionTemporalMemoryScore.value =
                                        temporalMotion.diagnostics
                                            .meanTemporalMemoryScore
                                            .coerceIn(0.0, 1.0)
                                    if (_state.value == CaptureState.RECORDING) {
                                        recorder.appendMotionDynamics(
                                            motionDynamics,
                                        )
                                    }
                                    applyCalibratedMotionUncertainty(
                                        skeleton = finalMotionSkeleton,
                                    )
                                    updateCrossModalIdentity(
                                        skeleton = finalMotionSkeleton,
                                        timestampNanos = finalMotionSkeleton.timestampNanos,
                                    )

                                    _kinematicConstraintScore.value =
                                        kinematic.score.coerceIn(0.0, 1.0)
                                    _kinematicConstraintAccepted.value =
                                        kinematic.accepted
                                    _kinematicConstraintBottleneck.value =
                                        kinematic.diagnostics.bottleneck
                                    _kinematicConstraintWarnings.value =
                                        kinematic.diagnostics.warnings

                                    _reconstructionJoints.value =
                                        finalMotionSkeleton.joints.size
                                    _reconstructionErrorPx.value =
                                        mono.monocular.reprojectionErrorPx
                                    val worldConsistencyScore =
                                        mono.worldConsistency.score
                                    val finalMonocularQuality =
                                        (
                                            mono.quality.confidence * 0.56 +
                                                worldConsistencyScore * 0.22 +
                                                kinematic.score * 0.22
                                            ).coerceIn(0.0, 1.0)

                                    _worldConsistencyScore.value =
                                        worldConsistencyScore
                                    _worldConsistencyWarnings.value =
                                        mono.worldConsistency.warnings
                                    _reconstructionQualityScore.value =
                                        finalMonocularQuality
                                    _reconstructionGateReasons.value =
                                        (
                                            mono.quality.rejectionReasons +
                                                mono.worldConsistency.warnings
                                        )
                                        .distinct()
                                        .take(8)

                                    val finalMonocularQualityWithTemporal =
                                        (
                                            finalMonocularQuality * 0.68 +
                                                temporalMotion.diagnostics.finalScore * 0.22 +
                                                temporalMotion.diagnostics.meanTemporalMemoryScore * 0.10
                                            ).coerceIn(0.0, 1.0)
                                    _reconstructionQualityScore.value =
                                        finalMonocularQualityWithTemporal
                                    _reconstructionGateReasons.value =
                                        (
                                            _reconstructionGateReasons.value +
                                                temporalMotion.diagnostics.warnings
                                            )
                                            .distinct()
                                            .take(12)

                                    recorder.appendReconstruction(
                                        finalMotionSkeleton,
                                    )
                                    recordSemanticMotionEvent(
                                        skeleton = finalMotionSkeleton,
                                        confidence = finalMonocularQualityWithTemporal,
                                    )

                                    val avatar =
                                        monocularRotationSolver.solve(
                                            frame = finalMotionSkeleton,
                                            previous = previousMonocularAvatarPose,
                                        ).pose

                                    previousMonocularAvatarPose = avatar
                                    appendAvatarIfSafe(
                                        skeleton = finalMotionSkeleton,
                                        avatar = avatar,
                                        confidence = finalMonocularQualityWithTemporal,
                                        reprojectionErrorPx =
                                            mono.monocular.reprojectionErrorPx,
                                         thermalScale = thermalScale,
                                         memoryScale =
                                             (
                                                 1.0 -
                                                     decision.heapPressure
                                             ).coerceIn(0.0, 1.0),
                                         runtimeQualityScale =
                                             result.runtimeDecision.qualityScale
                                                 .coerceIn(0.0, 1.0),
                                    )
                                }
                            }
                        }

                        val posePayload = PoseWireCodec.encode(
                            PoseWireCodec.Frame(
                                timestampNanos = pose.timestampNanos,
                                landmarks = pose.observations(),
                            ),
                        )
                        peers.broadcast(
                            NoXWireProtocol.encode(
                                type = NoXWireProtocol.Type.POSE_FRAME,
                                sequence = packetSequence.incrementAndGet(),
                                timestampNanos = pose.timestampNanos,
                                payload = posePayload,
                            ),
                        )

                        if (result.actorTracks.isNotEmpty()) {
                            val actorIdentityObservations =
                                result.actorTracks.map { actor ->
                                    CrossCameraActorIdentityFusionEngine.ActorObservation(
                                        localActorId = actor.actorId,
                                        confidence = actor.confidence,
                                        inputWidth =
                                            result.multiActorPose
                                                ?.inputWidth
                                                ?.coerceAtLeast(1)
                                                ?: pose.imageWidth.coerceAtLeast(1),
                                        inputHeight =
                                            result.multiActorPose
                                                ?.inputHeight
                                                ?.coerceAtLeast(1)
                                                ?: pose.imageHeight.coerceAtLeast(1),
                                        joints = actor.joints,
                                        worldJoints = actor.worldJoints,
                                        appearanceSignature =
                                            emptyList(),
                                        appearanceQuality =
                                            actor.appearanceQuality
                                                .coerceIn(0.0, 1.0),
                                        timestampNanos = pose.timestampNanos,
                                        cameraId = localId,
                                    )
                                }
                            crossCameraActorIdentityFusionEngine.rememberReference(
                                observations = actorIdentityObservations,
                            )

                            val multiActorPayload =
                                MultiActorPoseWireCodec.encode(
                                    MultiActorPoseWireCodec.Frame(
                                        timestampNanos = pose.timestampNanos,
                                        sourceDeviceId = localId.value,
                                        inputWidth =
                                            result.multiActorPose
                                                ?.inputWidth
                                                ?.coerceAtLeast(1)
                                                ?: pose.imageWidth,
                                        inputHeight =
                                            result.multiActorPose
                                                ?.inputHeight
                                                ?.coerceAtLeast(1)
                                                ?: pose.imageHeight,
                                        actors =
                                            result.actorTracks
                                                .take(64)
                                                .map { actor ->
                                                    MultiActorPoseWireCodec.Actor(
                                                        actorId = actor.actorId,
                                                        confidence =
                                                            actor.confidence
                                                                .toFloat()
                                                                .coerceIn(0f, 1f),
                                                        predicted = actor.predicted,
                                                        missedFrames = actor.missedFrames,
                                                        centerX =
                                                            actor.center.x
                                                                .toFloat(),
                                                        centerY =
                                                            actor.center.y
                                                                .toFloat(),
                                                        boundsLeft =
                                                            actor.bounds.left
                                                                .toFloat(),
                                                        boundsTop =
                                                            actor.bounds.top
                                                                .toFloat(),
                                                        boundsRight =
                                                            actor.bounds.right
                                                                .toFloat(),
                                                        boundsBottom =
                                                            actor.bounds.bottom
                                                                .toFloat(),
                                                        velocityX =
                                                            actor.velocityXNormalizedPerSecond
                                                                .toFloat(),
                                                        velocityY =
                                                            actor.velocityYNormalizedPerSecond
                                                                .toFloat(),
                                                        joints =
                                                            actor.joints
                                                                .take(33)
                                                                .map { joint ->
                                                                    MultiActorPoseWireCodec.Joint(
                                                                        landmarkId = joint.landmarkId,
                                                                        x = joint.x.toFloat(),
                                                                        y = joint.y.toFloat(),
                                                                        confidence =
                                                                            joint.confidence
                                                                                .toFloat()
                                                                                .coerceIn(0f, 1f),
                                                                    )
                                                                },
                                                        worldJoints =
                                                            actor.worldJoints
                                                                .take(33)
                                                                .map { joint ->
                                                                    MultiActorPoseWireCodec.WorldJoint(
                                                                        landmarkId = joint.landmarkId,
                                                                        xMeters = joint.xMeters.toFloat(),
                                                                        yMeters = joint.yMeters.toFloat(),
                                                                        zMeters = joint.zMeters.toFloat(),
                                                                        confidence =
                                                                            joint.confidence
                                                                                .toFloat()
                                                                                .coerceIn(0f, 1f),
                                                                    )
                                                                },
                                                        predictionConfidence =
                                                            actor.predictionConfidence
                                                                .toFloat()
                                                                .coerceIn(0f, 1f),
                                                        meanMotionInnovationNormalized =
                                                            actor.meanMotionInnovationNormalized
                                                                .toFloat()
                                                                .coerceAtLeast(0f),
                                                        acceleratedTrack =
                                                            actor.acceleratedTrackCount > 0,
                                                        globalAssignmentBounded =
                                                            actor.globalAssignmentBounded,
                                                        appearanceFingerprint =
                                                            actor.appearanceFingerprint
                                                                .take(16)
                                                                .map { it.toFloat().coerceIn(0f, 1f) },
                                                        appearanceQuality =
                                                            actor.appearanceQuality
                                                                .toFloat()
                                                                .coerceIn(0f, 1f),
                                                    )
                                                },
                                    ),
                                )

                            peers.broadcast(
                                NoXWireProtocol.encode(
                                    type =
                                        NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAME,
                                    sequence =
                                        packetSequence.incrementAndGet(),
                                    timestampNanos = pose.timestampNanos,
                                    payload = multiActorPayload,
                                ),
                            )
                        }

                        result.denseLandmarkEvidence?.let { dense ->
                            val denseInput =
                                dense.landmarks
                                    .filter {
                                        it.state !=
                                            com.nivukx.nox.core.ai.DenseLandmarkEvidenceFusionEngine
                                                .EvidenceState.REJECTED
                                    }
                                    .map { landmark ->
                                        DenseLandmarkWireCodec.Landmark(
                                            subjectId = landmark.key.subjectId,
                                            landmarkId = landmark.key.landmarkId,
                                            normalizedX =
                                                landmark.normalizedX.toFloat(),
                                            normalizedY =
                                                landmark.normalizedY.toFloat(),
                                            confidence =
                                                landmark.confidence
                                                    .toFloat(),
                                            uncertainty =
                                                landmark.uncertainty
                                                    .toFloat(),
                                            visibility =
                                                landmark.visibility
                                                    .toFloat(),
                                            contactProbability =
                                                landmark.contactProbability
                                                    .toFloat(),
                                            temporalStability =
                                                landmark.temporalStability
                                                    .toFloat(),
                                            crossViewAgreement =
                                                landmark.crossViewAgreement
                                                    .toFloat(),
                                            reprojectionResidualNormalized =
                                                landmark
                                                    .weightedReprojectionResidualNormalized
                                                    .toFloat(),
                                            probabilitySigmaXNormalized =
                                                landmark
                                                    .probabilitySigmaXNormalized
                                                    .toFloat(),
                                            probabilitySigmaYNormalized =
                                                landmark
                                                    .probabilitySigmaYNormalized
                                                    .toFloat(),
                                            probabilityCalibrationScore =
                                                landmark
                                                    .probabilityCalibrationScore
                                                    .toFloat(),
                                            segmentationConsistency =
                                                landmark.segmentationConsistency
                                                    .toFloat()
                                                    .coerceIn(0f, 1f),
                                            segmentationEvidenceAvailable =
                                                landmark.segmentationEvidenceAvailable,
                                            predicted =
                                                landmark.state ==
                                                    com.nivukx.nox.core.ai
                                                        .DenseLandmarkEvidenceFusionEngine
                                                        .EvidenceState
                                                        .TEMPORALLY_PREDICTED,
                                        )
                                    }

                            if (denseInput.isNotEmpty()) {
                                val wireFrame =
                                    DenseLandmarkWireCodec.Frame(
                                        timestampNanos =
                                            pose.timestampNanos,
                                        landmarks = denseInput,
                                        imageWidth = image.width,
                                        imageHeight = image.height,
                                        rollingShutterSkewNanos =
                                            telemetry?.rollingShutterSkewNanos ?: 0L,
                                        frameDurationNanos =
                                            telemetry?.frameDurationNanos ?: 0L,
                                        exposureTimeNanos =
                                            telemetry?.exposureTimeNanos ?: 0L,
                                        captureTimingConfidence =
                                            frameTimingConfidence.toFloat().coerceIn(0f, 1f),
                                    )

                                fusion.pushDenseLandmarks(
                                    deviceId = localId,
                                    timestampNanos =
                                        pose.timestampNanos,
                                    landmarks =
                                        denseInput.map { landmark ->
                                            com.nivukx.nox.core.reconstruction
                                                .MultiViewFusionEngine
                                                .DenseLandmarkInput(
                                                    subjectId =
                                                        landmark.subjectId,
                                                    landmarkId =
                                                        landmark.landmarkId,
                                                    normalizedX =
                                                        landmark.normalizedX
                                                            .toDouble(),
                                                    normalizedY =
                                                        landmark.normalizedY
                                                            .toDouble(),
                                                    confidence =
                                                        landmark.confidence
                                                            .toDouble(),
                                                    uncertainty =
                                                        landmark.uncertainty
                                                            .toDouble(),
                                                    visibility =
                                                        landmark.visibility
                                                            .toDouble(),
                                                    contactProbability =
                                                        landmark.contactProbability
                                                            .toDouble(),
                                                    temporalStability =
                                                        landmark.temporalStability
                                                            .toDouble(),
                                                    crossViewAgreement =
                                                        landmark.crossViewAgreement
                                                            .toDouble(),
                                                    reprojectionResidualNormalized =
                                                        landmark
                                                            .reprojectionResidualNormalized
                                                            .toDouble(),
                                                    probabilitySigmaXNormalized =
                                                        landmark
                                                            .probabilitySigmaXNormalized
                                                            .toDouble(),
                                                    probabilitySigmaYNormalized =
                                                        landmark
                                                            .probabilitySigmaYNormalized
                                                            .toDouble(),
                                                    probabilityCalibrationScore =
                                                        landmark
                                                            .probabilityCalibrationScore
                                                            .toDouble(),
                                                    silhouetteConsistency =
                                                        landmark.segmentationConsistency.toDouble(),
                                                    frameHeightPx = image.height,
                                                    rollingShutterSkewNanos =
                                                        telemetry?.rollingShutterSkewNanos ?: 0L,
                                                    frameDurationNanos =
                                                        telemetry?.frameDurationNanos ?: 0L,
                                                    exposureTimeNanos =
                                                        telemetry?.exposureTimeNanos ?: 0L,
                                                    captureTimingConfidence =
                                                        frameTimingConfidence,
                                                    predicted =
                                                        landmark.predicted,
                                                )
                                        },
                                )

                                fusion
                                    .reconstructBufferedDenseSurface(
                                        timestampNanos =
                                            pose.timestampNanos,
                                    )
                                    ?.let(::applyDenseSurfaceResult)

                                val connectedPeers =
                                    peers.connected.value.sortedBy { it.value }
                                connectedPeers.forEach { endpoint ->
                                    val capability = peers.peerCapabilities(endpoint)
                                    val negotiatedWireVersion =
                                        if (capability.supportsDenseLandmarkTimingMetadata) {
                                            DenseLandmarkChunkedWireCodec.FrameWireVersion
                                                .V4_TIMING_METADATA
                                        } else {
                                            DenseLandmarkChunkedWireCodec.FrameWireVersion.LEGACY_V3
                                        }
                                    val directPacket =
                                        if (
                                            wireFrame.landmarks.size <=
                                                DenseLandmarkWireCodec.MAX_LANDMARKS_PER_FRAME
                                        ) {
                                            runCatching {
                                                NoXWireProtocol.encode(
                                                    type =
                                                        NoXWireProtocol.Type
                                                            .DENSE_LANDMARK_FRAME,
                                                    sequence =
                                                        packetSequence.incrementAndGet(),
                                                    timestampNanos =
                                                        wireFrame.timestampNanos,
                                                    payload =
                                                        if (
                                                            negotiatedWireVersion ==
                                                                DenseLandmarkChunkedWireCodec
                                                                    .FrameWireVersion.V4_TIMING_METADATA
                                                        ) {
                                                            DenseLandmarkWireCodec.encode(wireFrame)
                                                        } else {
                                                            DenseLandmarkWireCodec
                                                                .encodeV3Compatibility(wireFrame)
                                                        },
                                                )
                                            }.getOrNull()
                                        } else {
                                            null
                                        }

                                    when {
                                        directPacket != null &&
                                            directPacket.size <=
                                                NearbyPeerController.MAX_NEARBY_BYTES_PAYLOAD -> {
                                            if (
                                                !peers.send(
                                                    endpoint = endpoint,
                                                    packet = directPacket,
                                                    reconstructionCritical = true,
                                                )
                                            ) {
                                                _runtimeReasons.value =
                                                    (
                                                        _runtimeReasons.value +
                                                            "dense-landmark-direct-send-rejected"
                                                        ).distinct().takeLast(12)
                                            }
                                        }

                                        capability.supportsDenseLandmarkFragmentation -> {
                                            runCatching {
                                                val sequenceStart =
                                                    packetSequence.getAndAdd(
                                                        DenseLandmarkChunkedWireCodec.MAX_CHUNKS
                                                            .toLong(),
                                                    ) + 1L
                                                peers.sendDenseLandmarkFrame(
                                                    endpoint = endpoint,
                                                    frame = wireFrame,
                                                    streamId = localId.value.take(96),
                                                    frameSequence = wireFrame.timestampNanos,
                                                    packetSequenceStart = sequenceStart,
                                                    reconstructionCritical = true,
                                                    wireVersion = negotiatedWireVersion,
                                                )
                                            }.onFailure {
                                                _runtimeReasons.value =
                                                    (
                                                        _runtimeReasons.value +
                                                            "dense-landmark-fragment-send-failed"
                                                            ).distinct().takeLast(12)
                                            }
                                        }

                                        else -> {
                                            // Never truncate high-detail evidence to fit a legacy packet.
                                            _runtimeReasons.value =
                                                (
                                                    _runtimeReasons.value +
                                                        "dense-landmark-peer-lacks-fragmentation"
                                                    ).distinct().takeLast(12)
                                        }
                                    }
                                }                            }
                        }

                        result.faceContour?.let { contour ->
                            faceFusion.push(localId, contour)
                            peers.broadcast(
                                NoXWireProtocol.encode(
                                    type = NoXWireProtocol.Type.FACE_CONTOUR_FRAME,
                                    sequence = packetSequence.incrementAndGet(),
                                    timestampNanos = contour.timestampNanos,
                                    payload = FaceContourWireCodec.encode(contour),
                                ),
                            )
                        }

                        result.eyes.forEach { eye ->
                            eyeFusion.push(localId, eye)
                        }

                        result.faceMeshes.forEach { mesh ->
                            val frame = mesh.toModelFrame()
                            faceMeshFusion.push(
                                deviceId = localId,
                                frame = frame,
                            )
                            recorder.appendFaceMesh(frame)

                            val connectedPeerTopologyKey =
                                peers.connected.value
                                    .map { it.value }
                                    .sorted()
                                    .joinToString("|")
                                    .ifEmpty { "local-only" }

                            val transportDecision =
                                faceMeshTransportPolicy.evaluate(
                                    frame = frame,
                                    peerCount = peers.connected.value.size,
                                    peerTopologyKey =
                                        connectedPeerTopologyKey,
                                )

                            if (transportDecision.send) {
                                peers.broadcast(
                                    NoXWireProtocol.encode(
                                        type =
                                            NoXWireProtocol.Type.FACE_MESH_FRAME,
                                        sequence =
                                            packetSequence.incrementAndGet(),
                                        timestampNanos =
                                            frame.timestampNanos,
                                        payload =
                                            FaceMeshWireCodec.encode(frame),
                                    ),
                                )
                            }
                        }

                        updateSecondaryFusion(
                            timestampNanos = pose.timestampNanos,
                            faceMeshes = result.faceMeshes,
                        )

                        result.faces.forEach { face ->
                            recorder.appendFace(face)
                            peers.broadcast(
                                NoXWireProtocol.encode(
                                    type = NoXWireProtocol.Type.FACE_FRAME,
                                    sequence = packetSequence.incrementAndGet(),
                                    timestampNanos = face.timestampNanos,
                                    payload = FaceWireCodec.encode(face),
                                ),
                            )
                        }

                        val stageWarnings =
                            result.stageFreshness
                                .filterValues { !it }
                                .keys
                                .map { it.name.lowercase(Locale.US) }
                                .toMutableList()
                        if (_inferenceReason.value != "full-rate") {
                            stageWarnings += _inferenceReason.value
                        }

                        recorder.appendPipelineTelemetry(
                            PipelineTelemetry(
                                timestampNanos = pose.timestampNanos,
                                captureFps = activeCaptureProfile.fps.toDouble(),
                                inferenceFps = decision.effectiveInferenceFps.toDouble(),
                                inferenceLatencyMillis =
                                    (
                                        SystemClock.elapsedRealtimeNanos() -
                                            inferenceStartNanos
                                        ).coerceAtLeast(0L) / 1e6,
                                droppedInferenceFrames =
                                    perception.diagnostics().skippedFrames,
                                trackedSubjects = result.subjectCount,
                                faceTracks = result.faceTracks.size,
                                eyeTracks = result.eyes.size,
                                triangulatedJoints =
                                    detailed?.skeleton?.joints?.size ?: 0,
                                reconstructedVertices =
                                    _denseMeshVertices.value,
                                reprojectionErrorPx =
                                    detailed?.diagnostics
                                        ?.meanReprojectionErrorPx
                                        ?: 0.050,
                                clockSkewNanos = _skewNanos.value,
                                thermalStatus = decision.thermalStatus,
                                memoryPressure = decision.heapPressure,
                                gpuBackend = gpuAcceleration.backend,
                                healthy =
                                    result.motion.healthy &&
                                        result.motionFusion.attentionConfidence >= 0.20,
                                warnings = stageWarnings,
                            ),
                        )

                        recorder.appendTrackLifecycle(
                            timestampNanos = pose.timestampNanos,
                            state = result.trackLifecycle.dominantState.name,
                            score = result.trackLifecycle.continuityScore,
                            predictedCount =
                                result.trackLifecycle.predictedTrackIds.size,
                            recoveringCount =
                                result.trackLifecycle.recoveringTrackIds.size,
                            warnings =
                                result.trackLifecycle.warnings,
                        )

                        result.eyes.forEach { eye ->
                            recorder.appendEye(eye)
                            peers.broadcast(
                                NoXWireProtocol.encode(
                                    type = NoXWireProtocol.Type.EYE_FRAME,
                                    sequence = packetSequence.incrementAndGet(),
                                    timestampNanos = eye.timestampNanos,
                                    payload = EyeWireCodec.encode(eye),
                                ),
                            )
                        }

                        val detectorFocusTarget =
                            focusTargetEngine.select(
                                timestampNanos =
                                    frame.sensorTimestampNanos,
                                imageWidth = image.width,
                                imageHeight = image.height,
                                rotationDegrees =
                                    camera.sensorOrientation(cameraId),
                                faces = result.faces,
                                eyes = result.eyes,
                                objects = result.objects,
                            )

                        val subjectLockFocusTarget =
                            result.subjectLock.focus?.let { lock ->
                                com.nivukx.nox.core.model.FocusTarget(
                                    timestampNanos = frame.sensorTimestampNanos,
                                    normalizedX = lock.x.coerceIn(0.0, 1.0),
                                    normalizedY = lock.y.coerceIn(0.0, 1.0),
                                    confidence = lock.confidence.coerceIn(0.0, 1.0),
                                    mode = when (lock.source) {
                                        com.nivukx.nox.core.ai.ProductionSubjectLockEngine.FocusSource.LEFT_EYE,
                                        com.nivukx.nox.core.ai.ProductionSubjectLockEngine.FocusSource.RIGHT_EYE,
                                        com.nivukx.nox.core.ai.ProductionSubjectLockEngine.FocusSource.GAZE_FUSION ->
                                            com.nivukx.nox.core.model.FocusTargetMode.EYE
                                        com.nivukx.nox.core.ai.ProductionSubjectLockEngine.FocusSource.FACE_CENTER ->
                                            com.nivukx.nox.core.model.FocusTargetMode.FACE
                                        else ->
                                            com.nivukx.nox.core.model.FocusTargetMode.SUBJECT
                                    },
                                    trackingId = lock.trackId?.coerceIn(0L, Int.MAX_VALUE.toLong())?.toInt(),
                                    subjectType = com.nivukx.nox.core.model.SubjectType.UNKNOWN,
                                )
                            }

                        val focusTarget = when {
                            detectorFocusTarget == null -> subjectLockFocusTarget
                            subjectLockFocusTarget == null -> detectorFocusTarget
                            result.subjectLock.lockState ==
                                com.nivukx.nox.core.ai.ProductionSubjectLockEngine.LockState.LOCKED &&
                                subjectLockFocusTarget.mode !=
                                    com.nivukx.nox.core.model.FocusTargetMode.EYE &&
                                subjectLockFocusTarget.confidence >=
                                    detectorFocusTarget.confidence + 0.20 ->
                                subjectLockFocusTarget
                            else -> detectorFocusTarget
                        }

                        camera.setFocusTarget(focusTarget)

                        focusTarget?.let {
                            recorder.appendFocusTarget(it)
                        }

                        _focusMode.value =
                            focusTarget
                                ?.mode
                                ?.name
                                ?: "NONE"
                        _focusConfidence.value =
                            focusTarget
                                ?.confidence
                                ?: 0.0

                        val hands = result.hands
                        _handCount.value = hands.trackedHands
                        _handConfidence.value = hands.meanConfidence

                        recorder.appendHands(
                            deviceId = localId,
                            frame = hands,
                        )
                        peers.broadcast(
                            NoXWireProtocol.encode(
                                type = NoXWireProtocol.Type.HAND_FRAME,
                                sequence = packetSequence.incrementAndGet(),
                                timestampNanos = hands.timestampNanos,
                                payload = HandWireCodec.encode(hands),
                            ),
                        )
                        pushHandFrame(
                            deviceId = localId,
                            frame = hands,
                        )
                    }.addOnCompleteListener {
                        inferenceScheduler.recordInferenceDuration(
                            inferenceStartNanos,
                            SystemClock.elapsedRealtimeNanos(),
                        )
                        image.close()
                    }

                    recorder.appendFrame(
                        sequence = frame.sequence,
                        sensorTimestampNanos = frame.sensorTimestampNanos,
                        monotonicTimestampNanos = frame.monotonicTimestampNanos,
                    )
                } catch (error: Throwable) {
                    image.close()
                    fail(error.message ?: error.javaClass.simpleName)
                }
            }
        }

        openCaptureCamera(
            cameraId = cameraId,
            sidecarDirectory = dir,
            enableLiveDepthStream = enableLiveDepthStream,
            onFailure = { error ->
                recorder.close()
                fail(error.message ?: error.javaClass.simpleName)
            },
        )
    }

    private fun openCaptureCamera(
        cameraId: String,
        sidecarDirectory: java.io.File,
        enableLiveDepthStream: Boolean,
        onFailure: (Throwable) -> Unit,
    ) {
        camera.open(
            cameraId = cameraId,
            profile = activeCaptureProfile,
            sidecarDirectory = sidecarDirectory,
            allowPersistentSensorSidecars =
                activeCaptureProfile.enableRawSidecar ||
                    activeCaptureProfile.enableDepthSidecar,
            enableLiveDepthStream = enableLiveDepthStream,
        ) { error ->
            if (error != null) {
                val message = error.message ?: error.javaClass.simpleName
                _runtimeReasons.value =
                    (
                        _runtimeReasons.value +
                            "camera-recording-open-failed:$message"
                    )
                        .distinct()
                        .takeLast(12)
                onFailure(error)
            } else if (
                enableLiveDepthStream &&
                !camera.isDepthReaderActive
            ) {
                _runtimeReasons.value =
                    (
                        _runtimeReasons.value +
                            "live-depth-output-unavailable-after-session-negotiation"
                    )
                        .distinct()
                        .takeLast(12)
            }
        }
    }

    private fun stopCapture(remote: Boolean = false) {
        captureGeneration += 1L
        scheduledStart?.cancel()
        scheduledStart = null
        _state.value = CaptureState.FINALIZING
        _motionScore.value = 0.0
        _motionTemporalMemoryScore.value = 0.0
        _motionMeasurementUncertainty.value = 0.0
        _temporalStability.value = 0.0
        motionEvidenceEngine.reset()
        _motionEvidenceScore.value = 0.0
        _motionEvidenceConfidence.value = 0.0
        _motionEvidenceWarnings.value = emptyList()

        frameJob?.cancel()
        frameJob = null

        imuJob?.cancel()
        imuJob = null
        depthJob?.cancel()
        depthJob = null
        val rawEvidencePath =
            camera.activeRawOutput?.absolutePath
        val depthEvidencePath =
            camera.activeDepthOutput?.absolutePath

        imuCapture.stop()

        camera.stopCapture()

        val hybridCompletion =
            activeHybridSessionId?.let { sessionId ->
                runCatching {
                    hybridCaptureCoordinator.finish(
                        sessionId = sessionId,
                        maxSubjectsObserved = _motionCustomization.value.maxSubjects,
                        frameCount = _frameCount.value,
                        rawEvidencePath = rawEvidencePath,
                        depthEvidencePath = depthEvidencePath,
                    )
                }.onFailure { error ->
                    _runtimeReasons.value =
                        (
                            _runtimeReasons.value +
                                "hybrid-finish-failed:" +
                                (error.message ?: error.javaClass.simpleName)
                            )
                            .distinct()
                            .takeLast(12)
                }.getOrNull()
            }

        activeHybridSessionId = null

        val postVideoPath =
            hybridCompletion
                ?.manifest
                ?.visualEvidencePath
                ?.takeIf { it.isNotBlank() && java.io.File(it).isFile }

        if (
            _motionCustomization.value.postCaptureAutoReconstructionEnabled &&
            postVideoPath != null
        ) {
            _postCaptureState.value = "QUEUED"
            _postCaptureProgress.value = 0.0
            _postCaptureProcessedFrames.value = 0L
            _postCaptureMaxActors.value = 0
            _postCaptureVideoPath.value = postVideoPath

            val offlineCustomization = _motionCustomization.value
            val offlinePath = postVideoPath

            viewModelScope.launch(Dispatchers.Default) {
                _postCaptureState.value = "PROCESSING"

                runCatching {
                    perception.resetForNewSequence()
                    postCaptureReconstructionBridge.reset()
                }.onFailure { error ->
                    _runtimeReasons.value =
                        (
                            _runtimeReasons.value +
                                "post-capture-sequence-reset-failed:" +
                                (error.message ?: error.javaClass.simpleName)
                            )
                            .distinct()
                            .takeLast(12)
                }

                val result =
                    runCatching {
                        recordedVideoPostProcessor.process(
                            videoFile = java.io.File(offlinePath),
                            configuration =
                                RecordedVideoPostProcessor.Configuration(
                                    cameraId = "offline-0",
                                    customization = offlineCustomization,
                                ),
                            onFrame = { summary, _ ->
                                _postCaptureProcessedFrames.value =
                                    summary.sequence + 1L
                                _postCaptureMaxActors.value =
                                    maxOf(
                                        _postCaptureMaxActors.value,
                                        summary.actorCount,
                                    )
                            },
                        )
                    }

                result
                    .onSuccess { processed ->
                        _postCaptureProcessedFrames.value =
                            processed.processedFrames
                        _postCaptureMaxActors.value =
                            maxOf(
                                processed.maxObservedActors,
                                processed.acceptedReconstructedActors.toInt(),
                            )
                        _postCaptureProgress.value =
                            processed.processingCoverage.coerceIn(0.0, 1.0)
                        _postCaptureState.value =
                            when {
                                processed.completed &&
                                    processed.processingCoverage >= 0.98 ->
                                    "COMPLETE"
                                processed.cancelled ->
                                    "CANCELLED"
                                else ->
                                    "DEGRADED"
                            }

                        if (processed.warnings.isNotEmpty()) {
                            _runtimeReasons.value =
                                (
                                    _runtimeReasons.value +
                                        processed.warnings
                                    )
                                    .distinct()
                                    .takeLast(12)
                        }
                    }
                    .onFailure { error ->
                        _postCaptureState.value = "FAILED"
                        _runtimeReasons.value =
                            (
                                _runtimeReasons.value +
                                    "post-capture-reconstruction-failed:" +
                                    (
                                        error.message
                                            ?: error.javaClass.simpleName
                                        )
                                )
                                .distinct()
                                .takeLast(12)
                    }
            }
        } else {
            _postCaptureState.value =
                if (postVideoPath == null) {
                    "NOT_AVAILABLE"
                } else {
                    "DISABLED"
                }
            _postCaptureProgress.value = 0.0
        }

        focusTargetEngine.reset()
        camera.setFocusTarget(null)
        _focusMode.value = "NONE"
        _focusConfidence.value = 0.0
        _subjectLockState.value = "SEARCHING"
        _subjectLockConfidence.value = 0.0
        _subjectLockTrackId.value = null
        _subjectFocusX.value = 0.5
        _subjectFocusY.value = 0.5
        _identityContinuityScore.value = 0.0
        _identitySwapRisk.value = 0.0
        _identityAction.value = "SEARCH"
        _identityReason.value = "stopped"
        _identityEvidenceFusionScore.value = 0.0
        _identitySelectedModalityCount.value = 0
        _identityEvidenceWarnings.value = emptyList()
        _faceEyeGeometricIntegrity.value = 0.0
        _faceEyeFaceFreshnessScore.value = 0.0
        _faceEyeGazeAgreementScore.value = 0.0
        _faceEyeRayGeometryCondition.value = 0.0
        _faceEyeIpdConsistencyScore.value = 1.0
        _faceEyeOutOfDistributionRisk.value = 0.0
        _faceEyeShapeConfidence.value = 0.0
        _faceEyeEyeDynamicConsistency.value = 0.0
        _faceEyeHardFailureReasons.value = emptyList()
        faceEyeEvidenceFusionEngine.reset()
        adaptiveFaceEyeEvidenceFusionEngine.reset()
        _faceEyeMetricState.value =
            FaceEyeMetricEvidenceEngine.State.NO_DATA.name
        _faceEyeMetricConfidence.value = 0.0
        _faceEyeMetricEyeUncertaintyMm.value =
            Double.POSITIVE_INFINITY
        _faceEyeMetricWarnings.value = emptyList()
        _faceEyeMetricRecommendations.value = emptyList()
        _faceEyeFusionConfidence.value = 0.0
        _faceEyeFusionAccepted.value = false
        _faceEyeFusionMode.value =
            FaceEyeEvidenceFusionEngine.FusionMode.HOLD.name
        _faceEyeIndependentEvidenceSupport.value = 0.0
        _faceEyeReliabilityTrend.value = 0.0
        _faceEyeFusionWarnings.value = emptyList()
        _faceEyeCaptureDomainConfidence.value = 0.0
        _faceEyeLightingConfidence.value = 0.0
        _faceEyeScaleConfidence.value = 0.0
        _faceEyeUncertaintyCalibrationConfidence.value = 0.0
        _faceEyeUncertaintyCalibrationReady.value = false
        _faceUncertaintyCalibrationMultiplier.value = 1.0
        _eyeUncertaintyCalibrationMultiplier.value = 1.0
        _faceUncertaintyCoverage.value = 0.0
        _eyeUncertaintyCoverage.value = 0.0
        _faceEyeFailureAttribution.value = emptyList()
        _faceEyeSaccadeVelocityDegreesPerSecond.value = 0.0
        faceEyeCalibrationDriftEngine.reset()
        _faceEyeCalibrationDriftScore.value = 0.0
        _faceEyeCalibrationDriftPersistent.value = false
        _faceEyeRecalibrationRecommended.value = false
        _faceEyeCalibrationDriftWarnings.value = emptyList()
        latestLeftEye3D = null
        latestLeftEyeTimestampNanos = Long.MIN_VALUE
        latestRightEye3D = null
        latestRightEyeTimestampNanos = Long.MIN_VALUE
        latestFaceShapeConfidence = 0.0
        latestFaceShapeAccepted = false
        latestFacePointCount = 0
        latestDenseFaceObserved = false
        _trackLifecycleState.value =
            com.nivukx.nox.core.ai.ProductionTrackLifecycleController.State.NEW.name
        _trackLifecycleScore.value = 0.0
        _trackLifecyclePredictedCount.value = 0
        _trackLifecycleRecoveringCount.value = 0
        _trackLifecycleWarnings.value = emptyList()
        _inferenceFps.value = 0
        _inferenceReason.value = "STOPPED"
        retargeter.reset()
        recorder.close()
        getApplication<Application>().stopService(
            Intent(getApplication(), CaptureForegroundService::class.java),
        )

        _sessions.value = sessions.list()

        if (!remote) {
            peers.broadcast(
                NoXWireProtocol.encode(
                    NoXWireProtocol.Type.STOP,
                    packetSequence.incrementAndGet(),
                    System.nanoTime(),
                ),
            )
        }
        _state.value = CaptureState.ARMED
    }

    fun refreshSessions() {
        _sessions.value = sessions.list()
    }

    private fun ingestNativePerceptionResult(
        cameraId: Int,
        result: PerceptionPipeline.Result,
        imageWidth: Int,
        imageHeight: Int,
        thermalStatus: Int = 0,
        memoryPressure: Double = 0.0,
    ) {
        val width = imageWidth.coerceAtLeast(1)
        val height = imageHeight.coerceAtLeast(1)
        val observations =
            ArrayList<com.nivukx.nox.core.model.LandmarkObservation>(4096)

        observations += result.pose.observations()

        val actorWidth =
            result.multiActorPose?.inputWidth?.coerceAtLeast(1)
                ?: imageWidth.coerceAtLeast(1)
        val actorHeight =
            result.multiActorPose?.inputHeight?.coerceAtLeast(1)
                ?: imageHeight.coerceAtLeast(1)

        result.actorTracks
            .asSequence()
            .take(64)
            .forEach { actor ->
                actor.joints
                    .asSequence()
                    .take(64)
                    .forEach { joint ->
                        observations +=
                            com.nivukx.nox.core.model.LandmarkObservation(
                                landmarkId = joint.landmarkId,
                                x = joint.x * actorWidth,
                                y = joint.y * actorHeight,
                                confidence =
                                    joint.confidence
                                        .coerceIn(0.0, 1.0),
                                timestampNanos =
                                    result.timestampNanos,
                                frameWidth = actorWidth,
                                frameHeight = actorHeight,
                                space =
                                    com.nivukx.nox.core.model.LandmarkSpace.PIXELS,
                                trackingId =
                                    actor.actorId
                                        .coerceAtMost(
                                            Int.MAX_VALUE.toLong(),
                                        )
                                        .toInt(),
                            )
                    }
            }

        result.eyes.firstOrNull()?.let { eyes ->
            eyes.left?.let { eye ->
                observations +=
                    com.nivukx.nox.core.model.LandmarkObservation(
                        landmarkId = 2000,
                        x = eye.center.x,
                        y = eye.center.y,
                        confidence = eye.confidence,
                        timestampNanos = eyes.timestampNanos,
                        frameWidth = width,
                        frameHeight = height,
                    )
                eye.irisCenter?.let { iris ->
                    observations +=
                        com.nivukx.nox.core.model.LandmarkObservation(
                            landmarkId = 2001,
                            x = iris.x,
                            y = iris.y,
                            confidence = eye.confidence,
                            timestampNanos = eyes.timestampNanos,
                            frameWidth = width,
                            frameHeight = height,
                        )
                }
            }
            eyes.right?.let { eye ->
                observations +=
                    com.nivukx.nox.core.model.LandmarkObservation(
                        landmarkId = 2002,
                        x = eye.center.x,
                        y = eye.center.y,
                        confidence = eye.confidence,
                        timestampNanos = eyes.timestampNanos,
                        frameWidth = width,
                        frameHeight = height,
                    )
                eye.irisCenter?.let { iris ->
                    observations +=
                        com.nivukx.nox.core.model.LandmarkObservation(
                            landmarkId = 2003,
                            x = iris.x,
                            y = iris.y,
                            confidence = eye.confidence,
                            timestampNanos = eyes.timestampNanos,
                            frameWidth = width,
                            frameHeight = height,
                        )
                }
            }
        }

        result.faces.firstOrNull()?.let { face ->
            face.points
                .asSequence()
                .take(49)
                .forEach { point ->
                    observations +=
                        com.nivukx.nox.core.model.LandmarkObservation(
                            landmarkId = 1000 + point.index,
                            x = point.point.x,
                            y = point.point.y,
                            confidence = point.confidence,
                            timestampNanos = face.timestampNanos,
                            frameWidth = width,
                            frameHeight = height,
                        )
                }
        }

        result.hands.hands
            .asSequence()
            .take(2)
            .forEachIndexed { handSlot, hand ->
                hand.landmarks
                    .asSequence()
                    .take(21)
                    .forEach { landmark ->
                        observations +=
                            com.nivukx.nox.core.model.LandmarkObservation(
                                landmarkId =
                                    3000 +
                                        handSlot * 32 +
                                        landmark.index,
                                x = landmark.normalizedX * width,
                                y = landmark.normalizedY * height,
                                confidence =
                                    (
                                        hand.confidence *
                                            landmark.confidence
                                        ).coerceIn(0.0, 1.0),
                                timestampNanos = hand.timestampNanos,
                                frameWidth = width,
                                frameHeight = height,
                            )
                    }
            }

        ingestNativeMotionObservations(
            cameraId = cameraId,
            observations =
                observations
                    .take(8192),
        )
    }

    private fun ingestNativeMotionObservations(
        cameraId: Int,
        observations: List<com.nivukx.nox.core.model.LandmarkObservation>,
    ) {
        val native = nativeMotionEngine ?: return

        if (!nativeMotionEngineStarted) {
            nativeMotionEngineStarted =
                runCatching { native.start() }.getOrDefault(false)

            if (!nativeMotionEngineStarted) {
                _runtimeReasons.value =
                    (
                        _runtimeReasons.value +
                            "native-engine-start-rejected"
                        )
                        .distinct()
                        .takeLast(12)
                return
            }
        }

        runCatching {
            val memoryPressure =
                runCatching {
                    val manager =
                        getApplication<Application>()
                            .getSystemService(
                                android.app.ActivityManager::class.java,
                            )
                    val info = android.app.ActivityManager.MemoryInfo()
                    manager?.getMemoryInfo(info)
                    if (info.totalMem > 0L) {
                        (
                            1.0 -
                                info.availMem.toDouble() /
                                    info.totalMem.toDouble()
                        ).coerceIn(0.0, 1.0)
                    } else {
                        0.0
                    }
                }.getOrDefault(0.0)
            val thermalStatus =
                runCatching {
                    getApplication<Application>()
                        .getSystemService(
                            android.os.PowerManager::class.java,
                        )?.currentThermalStatus
                            ?: android.os.PowerManager.THERMAL_STATUS_NONE
                }.getOrDefault(
                    android.os.PowerManager.THERMAL_STATUS_NONE,
                )
            val inferenceFps = _inferenceFps.value
            val safeMemoryPressure =
                memoryPressure.coerceIn(0.0, 1.0)
            val safeGpuHeadroom =
                (
                    1.0 -
                        _gpuDispatchMillis.value / 16.67
                    ).coerceIn(0.10, 1.0)
            val safeNpuHeadroom =
                (
                    if (_inferenceFps.value <= 0) {
                        0.85
                    } else {
                        (
                            _inferenceFps.value.toDouble() /
                                _motionCustomization.value.inferenceFps
                                    .coerceAtLeast(1)
                        ).coerceIn(0.10, 1.0)
                    }
                )
            val safeBatterySaver =
                runCatching {
                    (
                        getApplication<Application>().getSystemService(
                            android.os.PowerManager::class.java,
                        )?.isPowerSaveMode == true
                    )
                }.getOrDefault(false)

            val environmentChanged =
                lastNativeThermalStatus != thermalStatus ||
                    lastNativeMemoryPressure != safeMemoryPressure ||
                    lastNativeGpuHeadroom != safeGpuHeadroom ||
                    lastNativeNpuHeadroom != safeNpuHeadroom ||
                    lastNativeBatterySaver != safeBatterySaver

            if (environmentChanged) {
                native.setEnvironment(
                    thermalStatus = thermalStatus,
                    memoryPressure = safeMemoryPressure,
                    gpuHeadroom = safeGpuHeadroom,
                    npuHeadroom = safeNpuHeadroom,
                    batterySaver = safeBatterySaver,
                )
                lastNativeThermalStatus = thermalStatus
                lastNativeMemoryPressure = safeMemoryPressure
                lastNativeGpuHeadroom = safeGpuHeadroom
                lastNativeNpuHeadroom = safeNpuHeadroom
                lastNativeBatterySaver = safeBatterySaver
            }

            native.ingestLandmarks(
                cameraId = cameraId,
                observations = observations,
            )
        }.onFailure {
            _runtimeReasons.value =
                (
                    _runtimeReasons.value +
                        "native-ingest-failed"
                    )
                    .distinct()
                    .takeLast(12)
        }
    }

    private fun handlePeerPacket(endpoint: DeviceId, payload: ByteArray) {
        runCatching { NoXWireProtocol.decode(payload) }
            .onSuccess { packet ->
                when (packet.type) {
                    NoXWireProtocol.Type.HELLO -> {
                        peerCapabilities[endpoint] =
                            PeerCapabilitiesWireCodec.decode(
                                packet.payload,
                            ) ?: PeerCapabilitiesWireCodec.legacyFallback()

                        sendClockProbes(endpoint)
                        sendLocalCalibrationFrame(endpoint)
                        peers.send(
                            endpoint,
                            NoXWireProtocol.encode(
                                type = NoXWireProtocol.Type.HELLO,
                                sequence = packet.sequence,
                                timestampNanos = System.nanoTime(),
                                payload =
                                    PeerCapabilitiesWireCodec.encode(
                                        PeerCapabilitiesWireCodec.localDefaults(),
                                    ),
                            ),
                        )
                    }
                    NoXWireProtocol.Type.PING -> peers.send(
                        endpoint,
                        NoXWireProtocol.encode(
                            NoXWireProtocol.Type.PONG,
                            packet.sequence,
                            System.nanoTime(),
                        ),
                    )
                    NoXWireProtocol.Type.PONG -> {
                        pendingPings.remove(packet.sequence)?.let { t1 ->
                            val estimate = clock.observe(
                                endpoint,
                                ClockSynchronizer.Sample(
                                    t1 = t1,
                                    t2 = packet.timestampNanos,
                                    t3 = packet.timestampNanos,
                                    t4 = System.nanoTime(),
                                ),
                            )
                            _skewNanos.value = maxOf(
                                _skewNanos.value,
                                kotlin.math.abs(estimate.offsetNanos),
                            )
                            refreshDevices(peers.connected.value)
                        }
                    }
                    NoXWireProtocol.Type.ARM -> {
                        if (peers.connected.value.isNotEmpty() &&
                            calibrationSnapshot == null
                        ) {
                            _lastError.value = "Waiting for optimized rig calibration"
                        } else {
                            _state.value = CaptureState.ARMED
                        }
                    }
                    NoXWireProtocol.Type.START -> {
                        if (_state.value == CaptureState.ARMED) {
                            startCapture(
                                remote = true,
                                endpoint = endpoint,
                                masterStartNanos = packet.timestampNanos,
                            )
                        }
                    }
                    NoXWireProtocol.Type.STOP -> {
                        if (_state.value == CaptureState.RECORDING) stopCapture(remote = true)
                    }
                    NoXWireProtocol.Type.CALIBRATION -> {
                        calibrationReferenceDevice = endpoint
                        _state.value = CaptureState.CALIBRATING
                        sendLocalCalibrationFrame(endpoint)
                        beginCalibrationCollection(endpoint)
                    }

                    NoXWireProtocol.Type.CALIBRATION_FRAME -> {
                        runCatching { CalibrationWireCodec.decode(packet.payload) }
                            .onSuccess { frame ->
                                calibrationDescriptors[endpoint] = frame
                                assignedRoles.putIfAbsent(endpoint, DeviceRole.CAMERA)
                                refreshDevices(peers.connected.value)
                            }
                    }

                    NoXWireProtocol.Type.CALIBRATION_RESULT -> {
                        runCatching {
                            CalibrationSnapshotWireCodec.decode(packet.payload)
                        }.onSuccess { snapshot ->
                            if (snapshot.reprojectionErrorPx <= 4.0) {
                                calibrationSnapshot = snapshot
                                configurePerceptionCalibration(snapshot)
                                _calibrationReady.value = true
                                updateRolesFromSnapshot(snapshot)
                                refreshDevices(peers.connected.value)
                                if (_state.value == CaptureState.CALIBRATING) {
                                    _state.value = CaptureState.ARMED
                                }
                            } else {
                                fail(
                                    "Peer calibration result rejected at " +
                                        "%.2f px".format(snapshot.reprojectionErrorPx),
                                )
                            }
                        }.onFailure {
                            _lastError.value = "Invalid optimized calibration result"
                        }
                    }

                    NoXWireProtocol.Type.FACE_CONTOUR_FRAME -> {
                        runCatching { FaceContourWireCodec.decode(packet.payload) }
                            .onSuccess { contour ->
                                val masterTimestamp = clock.convertPeerToMaster(
                                    endpoint,
                                    contour.timestampNanos,
                                )
                                faceFusion.push(
                                    endpoint,
                                    contour.copy(timestampNanos = masterTimestamp),
                                )
                                updateSecondaryFusion(masterTimestamp)
                            }
                    }
                    NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAME -> {
                        runCatching {
                            MultiActorPoseWireCodec.decode(packet.payload)
                        }.onSuccess { frame ->
                            val masterTimestamp =
                                clock.convertPeerToMaster(
                                    endpoint,
                                    frame.timestampNanos,
                                )

                            val actorObservations =
                                frame.actors
                                    .take(64)
                                    .map { actor ->
                                        CrossCameraActorIdentityFusionEngine.ActorObservation(
                                            localActorId = actor.actorId,
                                            confidence = actor.confidence.toDouble(),
                                            inputWidth = frame.inputWidth.coerceAtLeast(1),
                                            inputHeight = frame.inputHeight.coerceAtLeast(1),
                                            joints =
                                                actor.joints.map { joint ->
                                                    com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine.Joint(
                                                        landmarkId = joint.landmarkId,
                                                        x = joint.x.toDouble(),
                                                        y = joint.y.toDouble(),
                                                        confidence = joint.confidence.toDouble(),
                                                    )
                                                },
                                            worldJoints =
                                                actor.worldJoints.map { joint ->
                                                    com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine.WorldJoint(
                                                        landmarkId = joint.landmarkId,
                                                        xMeters = joint.xMeters.toDouble(),
                                                        yMeters = joint.yMeters.toDouble(),
                                                        zMeters = joint.zMeters.toDouble(),
                                                        confidence = joint.confidence.toDouble(),
                                                    )
                                                },
                                            appearanceFingerprint =
                                                actor.appearanceFingerprint.map {
                                                    it.toDouble().coerceIn(0.0, 1.0)
                                                },
                                            appearanceQuality =
                                                actor.appearanceQuality
                                                    .toDouble()
                                                    .coerceIn(0.0, 1.0),
                                            timestampNanos = masterTimestamp,
                                            cameraId = endpoint,
                                        )
                                    }

                            val identityResult =
                                crossCameraActorIdentityFusionEngine.associate(
                                    observations = actorObservations,
                                    maximumAgeNanos = 250_000_000L,
                                    acceptanceThreshold = 0.56,
                                )

                            val globalIds =
                                identityResult.assignments.associate {
                                    it.localActorId to it.globalActorId
                                }

                            frame.actors
                                .take(64)
                                .forEach { actor ->
                                    val observations =
                                        actor.joints
                                            .take(64)
                                            .map { joint ->
                                                com.nivukx.nox.core.model.LandmarkObservation(
                                                    landmarkId = joint.landmarkId,
                                                    x =
                                                        joint.x.toDouble() *
                                                            frame.inputWidth
                                                                .coerceAtLeast(1),
                                                    y =
                                                        joint.y.toDouble() *
                                                            frame.inputHeight
                                                                .coerceAtLeast(1),
                                                    confidence =
                                                        joint.confidence
                                                            .toDouble()
                                                            .coerceIn(0.0, 1.0),
                                                    timestampNanos =
                                                        masterTimestamp,
                                                    frameWidth =
                                                        frame.inputWidth
                                                            .coerceAtLeast(1),
                                                    frameHeight =
                                                        frame.inputHeight
                                                            .coerceAtLeast(1),
                                                    space =
                                                        com.nivukx.nox.core.model.LandmarkSpace.PIXELS,
                                                    trackingId =
                                                        (
                                                            globalIds[actor.actorId]
                                                                ?: actor.actorId
                                                            )
                                                            .coerceAtMost(
                                                                Int.MAX_VALUE.toLong(),
                                                            )
                                                            .toInt(),
                                                )
                                            }

                                    if (observations.size >= 4) {
                                        fusion.push(
                                            endpoint,
                                            masterTimestamp,
                                            observations,
                                        )
                                    }
                                }
                        }.onFailure {
                            _lastError.value =
                                "Multi-actor pose packet rejected: " +
                                    (
                                        it.message
                                            ?: "invalid payload"
                                        )
                        }
                    }

                    NoXWireProtocol.Type.POSE_FRAME -> {
                        runCatching { PoseWireCodec.decode(packet.payload) }
                            .onSuccess { poseFrame ->
                                val masterTimestamp = clock.convertPeerToMaster(
                                    endpoint,
                                    poseFrame.timestampNanos,
                                )
                                val peerLandmarks =
                                    poseFrame.landmarks.map { landmark ->
                                        landmark.copy(
                                            timestampNanos = masterTimestamp,
                                            frameWidth =
                                                activeCaptureProfile.analysisWidth,
                                            frameHeight =
                                                activeCaptureProfile.analysisHeight,
                                        )
                                    }

                                ingestNativeMotionObservations(
                                    cameraId =
                                        endpoint.value.hashCode() and
                                            0x7fffffff,
                                    observations = peerLandmarks,
                                )

                                latestMasterPoseTimestamps[endpoint] =
                                    masterTimestamp
                                fusion.push(
                                    endpoint,
                                    masterTimestamp,
                                    poseFrame.landmarks,
                                )
                                if (_state.value == CaptureState.CALIBRATING) {
                                    recordCalibrationView(
                                        endpoint,
                                        masterTimestamp,
                                        poseFrame.landmarks,
                                    )
                                }
                                updateSecondaryFusion(masterTimestamp)
                            }
                    }

                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAME -> {
                        runCatching {
                            DenseLandmarkWireCodec.decode(
                                packet.payload,
                                packet.timestampNanos,
                            )
                        }.onSuccess { frame ->
                            handleDenseLandmarkFrame(endpoint, frame)
                        }.onFailure {
                            _lastError.value =
                                "Dense landmark packet rejected: " +
                                    (it.message ?: "invalid dense landmark frame")
                        }
                    }

                    NoXWireProtocol.Type.FACE_MESH_FRAME -> {
                        runCatching {
                            FaceMeshWireCodec.decode(
                                packet.payload,
                            )
                        }.onSuccess { remoteMesh ->
                            val masterTimestamp =
                                clock.convertPeerToMaster(
                                    endpoint,
                                    remoteMesh.timestampNanos,
                                )
                            val synchronizedMesh =
                                remoteMesh.copy(
                                    timestampNanos = masterTimestamp,
                                )

                            faceMeshFusion.push(
                                deviceId = endpoint,
                                frame = synchronizedMesh,
                            )

                            if (_state.value == CaptureState.RECORDING) {
                                recorder.appendFaceMesh(
                                    synchronizedMesh,
                                )
                            }

                            updateSecondaryFusion(
                                timestampNanos = masterTimestamp,
                                faceMeshes = emptyList(),
                            )
                        }.onFailure {
                            _lastError.value =
                                "Face mesh packet rejected: " +
                                    (it.message ?: "invalid face mesh")
                        }
                    }

                    NoXWireProtocol.Type.FACE_FRAME -> {
                        runCatching { FaceWireCodec.decode(packet.payload) }
                            .onSuccess { face ->
                                val masterTimestamp =
                                    clock.convertPeerToMaster(
                                        endpoint,
                                        face.timestampNanos,
                                    )
                                val synchronizedFace =
                                    face.copy(
                                        timestampNanos = masterTimestamp,
                                    )

                                val faceObservations =
                                    synchronizedFace.points
                                        .asSequence()
                                        .take(49)
                                        .map { point ->
                                            com.nivukx.nox.core.model.LandmarkObservation(
                                                landmarkId = 1000 + point.index,
                                                x = point.point.x,
                                                y = point.point.y,
                                                confidence =
                                                    point.confidence
                                                        .coerceIn(0.0, 1.0),
                                                timestampNanos = masterTimestamp,
                                                frameWidth =
                                                    activeCaptureProfile
                                                        .analysisWidth,
                                                frameHeight =
                                                    activeCaptureProfile
                                                        .analysisHeight,
                                            )
                                        }
                                        .toList()

                                ingestNativeMotionObservations(
                                    cameraId =
                                        endpoint.value.hashCode() and
                                            0x7fffffff,
                                    observations = faceObservations,
                                )

                                recorder.appendFace(synchronizedFace)
                            }
                    }
                    NoXWireProtocol.Type.HAND_FRAME -> {
                        runCatching {
                            HandWireCodec.decode(packet.payload)
                        }.onSuccess { hand ->
                            val masterTimestamp =
                                clock.convertPeerToMaster(
                                    endpoint,
                                    hand.timestampNanos,
                                )
                            val synchronized =
                                hand.copy(timestampNanos = masterTimestamp)

                            val handObservations =
                                synchronized.hands
                                    .take(2)
                                    .flatMapIndexed { handSlot, observation ->
                                        observation.landmarks
                                            .take(21)
                                            .map { landmark ->
                                                com.nivukx.nox.core.model.LandmarkObservation(
                                                    landmarkId =
                                                        3000 +
                                                            handSlot * 32 +
                                                            landmark.index,
                                                    x =
                                                        landmark.normalizedX *
                                                            synchronized.imageWidth,
                                                    y =
                                                        landmark.normalizedY *
                                                            synchronized.imageHeight,
                                                    confidence =
                                                        (
                                                            observation.confidence *
                                                                landmark.confidence
                                                            ).coerceIn(
                                                                0.0,
                                                                1.0,
                                                            ),
                                                    timestampNanos =
                                                        masterTimestamp,
                                                    frameWidth =
                                                        synchronized.imageWidth
                                                            .coerceAtLeast(1),
                                                    frameHeight =
                                                        synchronized.imageHeight
                                                            .coerceAtLeast(1),
                                                )
                                            }
                                    }

                            ingestNativeMotionObservations(
                                cameraId =
                                    endpoint.value.hashCode() and
                                        0x7fffffff,
                                observations = handObservations,
                            )

                            if (_state.value == CaptureState.RECORDING) {
                                recorder.appendHands(
                                    deviceId = endpoint,
                                    frame = HandTrackingFrame(
                                        timestampNanos = masterTimestamp,
                                        imageWidth = hand.imageWidth,
                                        imageHeight = hand.imageHeight,
                                        hands = hand.hands,
                                        inferenceLatencyMillis = 0.0,
                                        modelReady = true,
                                    ),
                                )
                            }

                            pushHandFrame(
                                deviceId = endpoint,
                                frame = HandTrackingFrame(
                                    timestampNanos = masterTimestamp,
                                    imageWidth = hand.imageWidth,
                                    imageHeight = hand.imageHeight,
                                    hands = hand.hands,
                                    inferenceLatencyMillis = 0.0,
                                    modelReady = true,
                                ),
                            )
                        }.onFailure {
                            _lastError.value =
                                "Hand fusion packet rejected: " +
                                    (it.message ?: "invalid hand frame")
                        }
                    }

                    NoXWireProtocol.Type.EYE_FRAME -> {
                        runCatching { EyeWireCodec.decode(packet.payload) }
                            .onSuccess { eye ->
                                val masterTimestamp =
                                    clock.convertPeerToMaster(
                                        endpoint,
                                        eye.timestampNanos,
                                    )
                                val synchronizedEye =
                                    eye.copy(
                                        timestampNanos = masterTimestamp,
                                    )

                                val eyeObservations =
                                    buildList {
                                        synchronizedEye.left?.let { left ->
                                            add(
                                                com.nivukx.nox.core.model.LandmarkObservation(
                                                    landmarkId = 2000,
                                                    x = left.center.x,
                                                    y = left.center.y,
                                                    confidence = left.confidence,
                                                    timestampNanos = masterTimestamp,
                                                    frameWidth =
                                                        activeCaptureProfile
                                                            .analysisWidth,
                                                    frameHeight =
                                                        activeCaptureProfile
                                                            .analysisHeight,
                                                ),
                                            )
                                            left.irisCenter?.let { iris ->
                                                add(
                                                    com.nivukx.nox.core.model.LandmarkObservation(
                                                        landmarkId = 2001,
                                                        x = iris.x,
                                                        y = iris.y,
                                                        confidence = left.confidence,
                                                        timestampNanos =
                                                            masterTimestamp,
                                                        frameWidth =
                                                            activeCaptureProfile
                                                                .analysisWidth,
                                                        frameHeight =
                                                            activeCaptureProfile
                                                                .analysisHeight,
                                                    ),
                                                )
                                            }
                                        }
                                        synchronizedEye.right?.let { right ->
                                            add(
                                                com.nivukx.nox.core.model.LandmarkObservation(
                                                    landmarkId = 2002,
                                                    x = right.center.x,
                                                    y = right.center.y,
                                                    confidence = right.confidence,
                                                    timestampNanos = masterTimestamp,
                                                    frameWidth =
                                                        activeCaptureProfile
                                                            .analysisWidth,
                                                    frameHeight =
                                                        activeCaptureProfile
                                                            .analysisHeight,
                                                ),
                                            )
                                            right.irisCenter?.let { iris ->
                                                add(
                                                    com.nivukx.nox.core.model.LandmarkObservation(
                                                        landmarkId = 2003,
                                                        x = iris.x,
                                                        y = iris.y,
                                                        confidence = right.confidence,
                                                        timestampNanos =
                                                            masterTimestamp,
                                                        frameWidth =
                                                            activeCaptureProfile
                                                                .analysisWidth,
                                                        frameHeight =
                                                            activeCaptureProfile
                                                               .analysisHeight,
                                                    ),
                                                )
                                            }
                                        }
                                    }

                                ingestNativeMotionObservations(
                                    cameraId =
                                        endpoint.value.hashCode() and
                                            0x7fffffff,
                                    observations = eyeObservations,
                                )

                                if (_state.value == CaptureState.RECORDING) {
                                    recorder.appendEye(synchronizedEye)
                                }
                                eyeFusion.push(endpoint, synchronizedEye)
                                updateSecondaryFusion(masterTimestamp)
                            }
                    }

                    NoXWireProtocol.Type.DEPTH_FRAME -> {
                        runCatching {
                            DepthWireCodec.decode(packet.payload)
                        }.onSuccess { depth ->
                            if (depth.deviceId != endpoint) {
                                _lastError.value =
                                    "Rejected depth identity mismatch from " +
                                        endpoint.value
                                return@onSuccess
                            }

                            val snapshot =
                                calibrationSnapshot
                                    ?: return@onSuccess

                            val intrinsicsBase =
                                snapshot.intrinsics[endpoint]
                                    ?: return@onSuccess

                            val extrinsics =
                                snapshot.extrinsics[endpoint]
                                    ?: return@onSuccess

                            val scaledIntrinsics =
                                intrinsicsBase.copy(
                                    fx =
                                        intrinsicsBase.fx *
                                            depth.width.toDouble() /
                                            depth.sourceWidth.toDouble(),
                                    fy =
                                        intrinsicsBase.fy *
                                            depth.height.toDouble() /
                                            depth.sourceHeight.toDouble(),
                                    cx =
                                        intrinsicsBase.cx *
                                            depth.width.toDouble() /
                                            depth.sourceWidth.toDouble(),
                                    cy =
                                        intrinsicsBase.cy *
                                            depth.height.toDouble() /
                                            depth.sourceHeight.toDouble(),
                                    imageWidth = depth.width,
                                    imageHeight = depth.height,
                                )

                            val masterTimestamp =
                                clock.convertPeerToMaster(
                                    endpoint,
                                    depth.timestampNanos,
                                )

                            val adaptiveDepthDecision = denseFusion.integrateDepthAdaptive(
                                depthMeters = depth.depthMeters,
                                width = depth.width,
                                height = depth.height,
                                intrinsics = scaledIntrinsics,
                                cameraPose = extrinsics.pose,
                                timestampNanos = masterTimestamp,
                                baseStride = 1,
                                confidence = 0.78,
                                evidence = adaptiveTsdfEvidence(),
                                uncertaintyModel =
                                    depthUncertaintyModelEngine,
                            )
                            _denseTsdfDetailLevel.value = adaptiveDepthDecision.detailLevel
                            _denseTsdfComputePressure.value = adaptiveDepthDecision.computePressure
                            _denseTsdfDecisionReason.value = adaptiveDepthDecision.reason
                            refreshTsdfUncertaintyTelemetry()

                            refreshDenseMesh(
                                masterTimestamp,
                            )
                        }.onFailure {
                            _lastError.value =
                                "Depth fusion packet rejected: " +
                                    (it.message ?: "invalid depth")
                        }
                    }

                    NoXWireProtocol.Type.HEARTBEAT,
                    NoXWireProtocol.Type.DEVICE_DESCRIPTOR,
                    NoXWireProtocol.Type.CAPTURE_ACK,
                    NoXWireProtocol.Type.ERROR -> Unit
                }
            }
    }

    private fun refreshConcurrentRigAdmission(
        force: Boolean = false,
        nowNanos: Long = System.nanoTime(),
        memoryScale: Double = 1.0,
    ) {
        if (
            !force &&
            lastConcurrentRigRefreshNanos != Long.MIN_VALUE &&
            nowNanos - lastConcurrentRigRefreshNanos < 500_000_000L
        ) {
            return
        }

        lastConcurrentRigRefreshNanos = nowNanos

        val cameraIds =
            runCatching {
                camera.capabilities().cameraIds
            }.getOrDefault(emptyList())
                .distinct()

        if (cameraIds.size < 2) {
            _concurrentRigAdmissionState.value = "NOT_APPLICABLE"
            _concurrentRigAdmissionScore.value = 1.0
            _concurrentRigSelectedCameraIds.value = cameraIds
            _concurrentRigBlockers.value = emptyList()
            _concurrentRigWarnings.value = emptyList()
            return
        }

        val matrices =
            cameraIds.mapNotNull { id ->
                runCatching {
                    camera.cameraFeatureMatrix(id)
                }.getOrNull()
            }

        val calibration =
            calibrationSnapshot
        val baselineDiversity =
            buildMap {
                cameraIds.forEach { id ->
                    val center =
                        calibration
                            ?.extrinsics
                            ?.get(DeviceId(id))
                            ?.pose
                            ?.position
                    val nearest =
                        cameraIds
                            .filter { it != id }
                            .mapNotNull { other ->
                                calibration
                                    ?.extrinsics
                                    ?.get(DeviceId(other))
                                    ?.pose
                                    ?.position
                                    ?.let { point ->
                                        center?.distanceTo(point)
                                    }
                            }
                            .minOrNull()
                    put(
                        id,
                        (
                            (nearest ?: 0.0) / 0.60
                        ).coerceIn(0.0, 1.0),
                    )
                }
            }

        val sync =
            _multiCameraSyncConfidence.value
                .coerceIn(0.0, 1.0)
        val calibrationConfidence =
            _rigScore.value.coerceIn(0.0, 1.0)
        val thermalLoad =
            (
                1.0 -
                    runtimeThermalHeadroom()
            ).coerceIn(0.0, 1.0)
        val memoryPressure =
            (1.0 - memoryScale.coerceIn(0.0, 1.0))
                .coerceIn(0.0, 1.0)
        val pairSkew =
            _captureTimingJitterMillis.value.coerceAtLeast(0.0)

        val plan =
            concurrentCaptureCapabilityEngine.planFromFeatureMatrices(
                matrices = matrices,
                synchronizationConfidence =
                    cameraIds.associateWith { sync },
                calibrationConfidence =
                    cameraIds.associateWith { calibrationConfidence },
                thermalLoad =
                    cameraIds.associateWith { thermalLoad },
                memoryPressure =
                    cameraIds.associateWith { memoryPressure },
                viewDiversity =
                    baselineDiversity,
                pairSkewMillis =
                    cameraIds.associateWith { pairSkew },
                requestedCameraCount =
                    minOf(
                        8,
                        matrices.size,
                        matrices.maxOfOrNull { matrix ->
                            matrix.concurrentCameraSets
                                .maxOfOrNull { it.size }
                                ?: 1
                        }?.coerceAtLeast(2)
                            ?: 2,
                    ),
            )

        _concurrentRigAdmissionState.value =
            when {
                plan.admissible -> "ADMISSIBLE"
                plan.blockers.isEmpty() -> "UNKNOWN"
                else -> "REJECTED"
            }
        _concurrentRigAdmissionScore.value =
            plan.rigScore.coerceIn(0.0, 1.0)
        val topologyEvidence =
            cameraTopologyEvidenceEngine.evaluate(
                matrices = matrices,
                selectedCameraIds = plan.selectedCameraIds,
            )

        _concurrentRigTopologyScore.value =
            topologyEvidence.score
        _concurrentRigTopologyMode.value =
            topologyEvidence.mode.name
        _concurrentRigSelectedCameraIds.value =
            plan.selectedCameraIds
        _concurrentRigBlockers.value =
            plan.blockers
        _concurrentRigWarnings.value =
            (
                plan.warnings +
                    topologyEvidence.warnings
            ).distinct().take(16)
    }

    private fun updateThermalExtrinsicEvidence(
        telemetry: com.nivukx.nox.core.model.FrameTelemetry?,
        timestampNanos: Long,
    ) {
        val snapshot = calibrationSnapshot
            ?: return
        val temperature =
            telemetry?.sensorTemperatureCelsius
                ?.toDouble()
                ?.takeIf(Double::isFinite)
                ?: return
        val calibratedPose =
            snapshot.extrinsics[localId]?.pose
                ?: return
        val orientationSample =
            orientation.latest()
                ?: return

        val observedRotation =
            RigidTransform.alignAxes(
                orientationSample.forward,
                orientationSample.up,
            )

        val accuracyWeight =
            when (orientationSample.accuracy) {
                android.hardware.SensorManager.SENSOR_STATUS_ACCURACY_HIGH -> 1.0
                android.hardware.SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> 0.85
                android.hardware.SensorManager.SENSOR_STATUS_ACCURACY_LOW -> 0.65
                else -> 0.35
            }

        val result =
            thermalExtrinsicDriftEngine.observe(
                ThermalExtrinsicDriftEngine.Sample(
                    cameraId = localId,
                    timestampNanos = timestampNanos,
                    temperatureCelsius = temperature,
                    calibratedPose = calibratedPose,
                    observedPose = Pose(
                        position = calibratedPose.position,
                        rotation = observedRotation,
                    ),
                    calibrationConfidence =
                        _rigScore.value.coerceIn(0.0, 1.0),
                    mountIntegrity = accuracyWeight,
                    translationEvidenceAvailable = false,
                    rotationEvidenceAvailable = false,
                ),
            )

        _thermalExtrinsicDriftState.value =
            result.state.name
        _thermalExtrinsicDriftConfidence.value =
            result.confidence.coerceIn(0.0, 1.0)
        _thermalExtrinsicTranslationDriftMm.value =
            result.translationDriftMm
        _thermalExtrinsicRotationDriftDegrees.value =
            result.rotationDriftDegrees
        _thermalExtrinsicTemperatureDeltaCelsius.value =
            result.temperatureDeltaCelsius
        _thermalExtrinsicHardFailure.value =
            result.state == ThermalExtrinsicDriftEngine.State.BLOCKED
        _thermalExtrinsicWarnings.value =
            result.warnings
    }

    private fun beginCalibrationCollection(referenceDevice: DeviceId) {
        if (calibrationJob?.isActive == true) return

        calibrationJob = viewModelScope.launch {
            delay(CALIBRATION_WARMUP_MS)

            val cameraId = camera.capabilities().cameraIds.firstOrNull()
                ?: run {
                    fail("Calibration camera is unavailable")
                    return@launch
                }

            withTimeoutOrNull(CALIBRATION_DURATION_MS) {
                camera.frameFlow.collect { (frame, image) ->
                    val task = runCatching {
                        perception.process(
                            image = image,
                            rotationDegrees = camera.sensorOrientation(cameraId),
                            timestampNanos = frame.sensorTimestampNanos,
                            cameraId = cameraId,
                            calibrationCameraId = localId.value,
                            motionContext =
                                motionContextForPerception(),
                            rollingShutterSkewNanos =
                                camera.latestTelemetry?.rollingShutterSkewNanos ?: 0L,
                            frameDurationNanos =
                                camera.latestTelemetry?.frameDurationNanos ?: 0L,
                            exposureTimeNanos =
                                camera.latestTelemetry?.exposureTimeNanos ?: 0L,
                            captureTimingConfidence = resolvedCaptureTimingConfidence(),
                        )
                    }.getOrElse {
                        image.close()
                        return@collect
                    }

                    task.addOnSuccessListener { result ->
                        val pose = result.pose
                        val observations = pose.observations()
                        recordCalibrationView(
                            localId,
                            pose.timestampNanos,
                            observations,
                        )
                        peers.broadcast(
                            NoXWireProtocol.encode(
                                type = NoXWireProtocol.Type.POSE_FRAME,
                                sequence = packetSequence.incrementAndGet(),
                                timestampNanos = pose.timestampNanos,
                                payload = PoseWireCodec.encode(
                                    PoseWireCodec.Frame(
                                        timestampNanos = pose.timestampNanos,
                                        landmarks = observations,
                                    ),
                                ),
                            ),
                        )
                    }.addOnCompleteListener {
                        image.close()
                    }
                }
            }

            calibrationJob = null
            completeCalibration(referenceDevice)
        }
    }

    private fun completeCalibration(referenceDevice: DeviceId) {
        if (_state.value != CaptureState.CALIBRATING) return

        val descriptors = calibrationDescriptors.toMap()
        if (descriptors.size < 2) {
            fail("Calibration needs at least two calibrated cameras")
            return
        }

        val reference = descriptors[referenceDevice]
            ?: descriptors[localId]
            ?: run {
                fail("Calibration reference camera missing")
                return
            }

        val referenceForward = reference.pose.rotation
            .rotate(Vec3(0.0, 0.0, 1.0))
            .normalized()
        val referenceUp = reference.pose.rotation
            .rotate(Vec3(0.0, 1.0, 0.0))
            .normalized()
        val referenceEast = referenceUp.cross(referenceForward).normalized()
        val referenceNorth = referenceEast.cross(referenceUp).normalized()

        val candidates = descriptors.map { (id, descriptor) ->
            DeviceRoleAssignmentEngine.Candidate(
                id = id,
                forward = descriptor.pose.rotation
                    .rotate(Vec3(0.0, 0.0, 1.0))
                    .normalized(),
                up = descriptor.pose.rotation
                    .rotate(Vec3(0.0, 1.0, 0.0))
                    .normalized(),
            )
        }

        roleAssignment.assign(
            candidates = candidates,
            north = referenceNorth,
            east = referenceEast,
            up = referenceUp,
        ).forEach {
            assignedRoles[it.id] = it.role
        }
        assignedRoles[referenceDevice] = DeviceRole.MASTER
        updateCoverageReport()

        val seeds = descriptors.map { (id, descriptor) ->
            SelfCalibrationEngine.CameraSeed(
                id = id,
                intrinsics = descriptor.intrinsics,
                pose = Pose(
                    position = seedCenter(
                        assignedRoles[id] ?: DeviceRole.AUXILIARY,
                        id,
                    ),
                    rotation = descriptor.pose.rotation,
                ),
            )
        }

        val bestViews = selectBestCalibrationViews()
        if (bestViews.size < 2) {
            fail("Calibration sweep did not produce enough synchronized views")
            return
        }

        val result = selfCalibration.refine(
            seeds = seeds,
            views = bestViews,
            referenceDevice = referenceDevice,
        )

        if (result == null ||
            !result.reprojectionErrorPx.isFinite() ||
            result.landmarksUsed < 6
        ) {
            fail("Calibration optimizer rejected the rig geometry")
            return
        }

        calibrationSnapshot = result.snapshot
        configurePerceptionCalibration(result.snapshot)
        _calibrationReady.value =
            result.converged && result.reprojectionErrorPx <= 6.0

        if (!_calibrationReady.value) {
            fail(
                "Calibration rejected at " +
                    "%.2f px reprojection error".format(result.reprojectionErrorPx),
            )
            return
        }

        refreshDevices(peers.connected.value)
        peers.connected.value
            .sortedBy { it.value }
            .forEach { endpoint ->
                val capabilities =
                    peerCapabilities[endpoint]
                        ?: PeerCapabilitiesWireCodec.legacyFallback()
                val schema =
                    capabilities.calibrationSchemaVersion
                        .coerceIn(1, 2)
                peers.send(
                    endpoint,
                    NoXWireProtocol.encode(
                        type = NoXWireProtocol.Type.CALIBRATION_RESULT,
                        sequence = packetSequence.incrementAndGet(),
                        timestampNanos = result.snapshot.calibratedAtNanos,
                        payload =
                            CalibrationSnapshotWireCodec.encode(
                                snapshot = result.snapshot,
                                requestedVersion = schema,
                            ),
                    ),
                )
            }
        _state.value = CaptureState.ARMED
        peers.broadcast(
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.ARM,
                sequence = packetSequence.incrementAndGet(),
                timestampNanos = System.nanoTime(),
            ),
        )
    }

    private fun buildLocalCalibrationFrame(
        required: Boolean = true,
    ): CalibrationWireCodec.Frame? {
        val cameraId = camera.capabilities().cameraIds.firstOrNull()
            ?: run {
                if (required) fail("Calibration camera is unavailable")
                return null
            }

        val intrinsics = camera.intrinsicsModel(
            cameraId,
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
        ) ?: run {
            if (required) fail("Camera calibration intrinsics are unavailable")
            return null
        }

        val latest = orientation.latest() ?: run {
            if (required) fail("Orientation sensor has no sample yet")
            return null
        }

        return CalibrationWireCodec.Frame(
            deviceId = localId.value,
            role = DeviceRole.MASTER,
            intrinsics = intrinsics,
            pose = Pose(
                position = Vec3(0.0, 0.0, 0.0),
                rotation = RigidTransform.alignAxes(
                    latest.forward,
                    latest.up,
                ),
            ),
            accuracy = latest.accuracy,
        )
    }

    private fun sendLocalCalibrationFrame(endpoint: DeviceId) {
        val frame = buildLocalCalibrationFrame(required = false) ?: return
        calibrationDescriptors[localId] = frame
        peers.send(
            endpoint,
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.CALIBRATION_FRAME,
                sequence = packetSequence.incrementAndGet(),
                timestampNanos = System.nanoTime(),
                payload = CalibrationWireCodec.encode(frame),
            ),
        )
    }

    private fun recordCalibrationView(
        deviceId: DeviceId,
        timestampNanos: Long,
        landmarks: List<com.nivukx.nox.core.model.LandmarkObservation>,
    ) {
        if (landmarks.size < 6) return
        val bucket = (timestampNanos / CALIBRATION_BUCKET_NANOS) *
            CALIBRATION_BUCKET_NANOS
        val key = deviceId.value + ":" + bucket
        val view = SelfCalibrationEngine.LandmarkView(
            cameraId = deviceId,
            timestampNanos = bucket,
            landmarks = landmarks.associate {
                it.landmarkId to Vec2(it.x, it.y)
            },
        )
        calibrationViews[key] = view
        calibrationRawTimestamps[key] = timestampNanos
        latestMasterPoseTimestamps[deviceId] = timestampNanos

        updateCalibrationTemporalQuality(bucket)
    }

    private fun updateCalibrationTemporalQuality(bucket: Long) {
        val group = calibrationViews.values
            .filter { it.timestampNanos == bucket }
            .sortedBy { it.cameraId.value }

        val result = temporalAlignmentForViews(group)
            ?: return

        val strictnessFloor =
            (
                0.45 +
                    0.45 *
                        _motionCustomization.value.temporalAlignmentStrictness
                ).coerceIn(0.45, 0.90)
        val strictAccepted =
            result.accepted &&
                result.confidence >= strictnessFloor

        _temporalAlignmentConfidence.value =
            result.confidence.coerceIn(0.0, 1.0)
        _temporalAlignmentSpreadMillis.value =
            if (result.spreadNanos == Long.MAX_VALUE) {
                0.0
            } else {
                result.spreadNanos / 1_000_000.0
            }
        _temporalAlignmentState.value =
            if (strictAccepted) "ALIGNED" else "REJECTED"
        _temporalAlignmentWarnings.value =
            (
                result.warnings +
                    if (!strictAccepted && result.accepted) {
                        listOf("operator-temporal-strictness-gate")
                    } else {
                        emptyList()
                    }
                ).distinct().take(4)
    }

    private fun temporalAlignmentForViews(
        views: List<SelfCalibrationEngine.LandmarkView>,
    ): MultiCameraTemporalAlignmentEngine.GroupResult? {
        if (views.size < 2) return null

        val frames = views.mapIndexed { index, view ->
            val key =
                view.cameraId.value + ":" + view.timestampNanos
            val timestamp =
                calibrationRawTimestamps[key]
                    ?: view.timestampNanos

            MultiCameraTemporalAlignmentEngine.FrameInput(
                deviceId = view.cameraId,
                stamp = FrameStamp(
                    sequence = index.toLong(),
                    sensorTimestampNanos = timestamp,
                    monotonicTimestampNanos = timestamp,
                    exposureTimeNanos = 0L,
                ),
                domain =
                    MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC,
                hardwareSynchronized = false,
                sourceIsComparableAcrossDevices = true,
            )
        }

        return multiCameraTemporalAlignmentEngine.alignGroup(frames)
    }

    private fun selectBestCalibrationViews(): List<SelfCalibrationEngine.LandmarkView> =
        calibrationViews.values
            .groupBy { it.timestampNanos }
            .values
            .map { group ->
                val cameras = group.map { it.cameraId }.distinct().size
                val landmarks = group
                    .flatMap { it.landmarks.keys }
                    .groupingBy { it }
                    .eachCount()
                    .count { it.value >= 2 }

                val temporal =
                    temporalAlignmentForViews(group)

                val temporalScore =
                    if (temporal?.accepted == true) {
                        (temporal.confidence * 50.0).toInt()
                    } else {
                        -200
                    }

                group to (
                    cameras * 100 +
                        landmarks +
                        temporalScore
                    )
            }
            .filter { it.second >= 202 }
            .maxByOrNull { it.second }
            ?.first
            .orEmpty()

    private fun evaluateTemporalAlignment(
        connectedDevices: List<DeviceId>,
    ): MultiCameraTemporalAlignmentEngine.GroupResult? {
        val ids = connectedDevices.distinct()
        if (ids.size < 2) {
            _temporalAlignmentConfidence.value = 0.0
            _temporalAlignmentSpreadMillis.value = 0.0
            _temporalAlignmentState.value = "SINGLE_CAMERA"
            _temporalAlignmentWarnings.value = emptyList()
            return null
        }

        val frames = ids.mapIndexedNotNull { index, id ->
            val timestamp =
                latestMasterPoseTimestamps[id]
                    ?: return@mapIndexedNotNull null

            MultiCameraTemporalAlignmentEngine.FrameInput(
                deviceId = id,
                stamp = FrameStamp(
                    sequence = index.toLong(),
                    sensorTimestampNanos = timestamp,
                    monotonicTimestampNanos = timestamp,
                    exposureTimeNanos = 0L,
                ),
                domain =
                    MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC,
                hardwareSynchronized = false,
                sourceIsComparableAcrossDevices = true,
            )
        }

        if (frames.size < 2) {
            _temporalAlignmentConfidence.value = 0.0
            _temporalAlignmentSpreadMillis.value = 0.0
            _temporalAlignmentState.value = "WAITING_FRAMES"
            _temporalAlignmentWarnings.value =
                listOf("Waiting for pose timestamps from all rig cameras.")
            return null
        }

        val result =
            multiCameraTemporalAlignmentEngine.alignGroup(frames)

        val strictnessFloor =
            (
                0.45 +
                    0.45 *
                        _motionCustomization.value.temporalAlignmentStrictness
                ).coerceIn(0.45, 0.90)
        val strictAccepted =
            result.accepted &&
                result.confidence >= strictnessFloor

        _temporalAlignmentConfidence.value =
            result.confidence.coerceIn(0.0, 1.0)
        _temporalAlignmentSpreadMillis.value =
            if (result.spreadNanos == Long.MAX_VALUE) {
                0.0
            } else {
                result.spreadNanos / 1_000_000.0
            }
        _temporalAlignmentState.value =
            if (strictAccepted) "ALIGNED" else "REJECTED"
        _temporalAlignmentWarnings.value =
            (
                result.warnings +
                    if (!strictAccepted && result.accepted) {
                        listOf("operator-temporal-strictness-gate")
                    } else {
                        emptyList()
                    }
                ).distinct().take(4)

        return result
    }

    private fun seedCenter(role: DeviceRole, id: DeviceId): Vec3 {
        val radius = 2.0
        return when (role) {
            DeviceRole.MASTER -> Vec3(0.0, 0.0, 0.0)
            DeviceRole.FRONT -> Vec3(0.0, 0.0, radius)
            DeviceRole.REAR -> Vec3(0.0, 0.0, -radius)
            DeviceRole.RIGHT -> Vec3(radius, 0.0, 0.0)
            DeviceRole.LEFT -> Vec3(-radius, 0.0, 0.0)
            DeviceRole.TOP -> Vec3(0.0, radius, 0.0)
            DeviceRole.BOTTOM -> Vec3(0.0, -radius, 0.0)
            else -> {
                val angle = abs(id.value.hashCode()) % 360 * PI / 180.0
                Vec3(
                    cos(angle) * radius,
                    0.35,
                    sin(angle) * radius,
                )
            }
        }
    }

    private fun pushHandFrame(
        deviceId: DeviceId,
        frame: HandTrackingFrame,
    ) {
        val snapshot = calibrationSnapshot ?: return
        val intrinsics = snapshot.intrinsics[deviceId] ?: return
        val extrinsics = snapshot.extrinsics[deviceId] ?: return

        handFrames[deviceId] =
            Hand3DReconstructionEngine.CameraFrame(
                deviceId = deviceId,
                timestampNanos = frame.timestampNanos,
                imageWidth = frame.imageWidth,
                imageHeight = frame.imageHeight,
                hands = frame.hands,
                intrinsics = intrinsics.copy(
                    imageWidth = frame.imageWidth,
                    imageHeight = frame.imageHeight,
                ),
                pose = extrinsics.pose,
            )

        val staleBefore =
            frame.timestampNanos - 80_000_000L
        handFrames.entries.removeIf {
            it.value.timestampNanos < staleBefore
        }

        val reconstructed =
            handFusion.reconstruct(
                timestampNanos = frame.timestampNanos,
                frames = handFrames.values.toList(),
            )

        if (reconstructed != null) {
            val leftWrist =
                reconstructed.hands
                    .firstOrNull {
                        it.side.name.equals("LEFT", ignoreCase = true)
                    }
                    ?.landmarks
                    ?.firstOrNull { it.index == 0 }
                    ?.position
            val rightWrist =
                reconstructed.hands
                    .firstOrNull {
                        it.side.name.equals("RIGHT", ignoreCase = true)
                    }
                    ?.landmarks
                    ?.firstOrNull { it.index == 0 }
                    ?.position

            if (leftWrist != null) {
                latestLeftHand3D = leftWrist
                latestLeftHandTimestampNanos = frame.timestampNanos
            }
            if (rightWrist != null) {
                latestRightHand3D = rightWrist
                latestRightHandTimestampNanos = frame.timestampNanos
            }

            val consistency =
                handTemporalConsistencyEngine.evaluate(
                    reconstructed,
                )
            val geometry =
                handGeometryQualityEngine.evaluate(reconstructed)
            val handUncertainty =
                handUncertaintyCalibrationEngine.evaluate(
                    frame = reconstructed,
                    dynamicStress = _motionDynamicStress.value
                        .coerceIn(0.0, 1.0),
                    occlusionRisk = _occlusionRisk.value
                        .coerceIn(0.0, 1.0),
                    synchronizationConfidence =
                        _multiCameraSyncConfidence.value
                            .coerceIn(0.0, 1.0),
                )

            _handUncertaintyConfidence.value =
                handUncertainty.uncertaintyConfidence
                    .coerceIn(0.0, 1.0)
            _handUncertaintyRadiusMeters.value =
                handUncertainty.meanRadiusMeters
            _handUncertaintyCoverage.value =
                handUncertainty.empiricalCoverage
            _handUncertaintyOodRisk.value =
                handUncertainty.outOfDistributionRisk
            _handUncertaintyPolicy.value =
                handUncertainty.policy.name
            _handUncertaintyWarnings.value =
                handUncertainty.warnings

            val geometryWeight =
                (0.16 + 0.24 * _motionCustomization.value.handGeometryStrictness)
                    .coerceIn(0.16, 0.40)
            val temporalWeight =
                (0.14 + 0.12 * _motionCustomization.value.handTemporalStrictness)
                    .coerceIn(0.14, 0.26)
            val baseWeight =
                (1.0 - geometryWeight - temporalWeight)
                    .coerceIn(0.34, 0.70)

            _handTemporalConsistencyConfidence.value =
                (
                    consistency.confidence * 0.70 +
                        geometry.temporalStability * 0.30
                    ).coerceIn(0.0, 1.0)
            _handLongitudinalAnatomyConsistency.value =
                geometry.meanLongitudinalSegmentConsistency
                    .coerceIn(0.0, 1.0)
            _handTemporalConsistencyWarnings.value =
                (
                    consistency.warnings +
                        geometry.warnings.map { "geometry-$it" }
                ).distinct().take(10)

            _hand3DConfidence.value =
                (
                    reconstructed.confidence * baseWeight +
                        consistency.confidence * temporalWeight +
                        geometry.confidence * geometryWeight +
                        handUncertainty.uncertaintyConfidence * 0.08
                    ).coerceIn(0.0, 1.0)

            if (_state.value == CaptureState.RECORDING) {
                recorder.appendHand3D(reconstructed)
            }
        }
    }

    private fun FaceMeshGeometryEngine.Result.toModelFrame(): FaceMeshFrame =
        FaceMeshFrame(
            timestampNanos = timestampNanos,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = rotationDegrees,
            trackingId = trackingId,
            boundsLeft = boundsLeft,
            boundsTop = boundsTop,
            boundsRight = boundsRight,
            boundsBottom = boundsBottom,
            confidence = confidence,
            pointCoverage = pointCoverage,
            triangleCoverage = triangleCoverage,
            points = points.map {
                FaceMeshPoint3D(
                    index = it.index,
                    position = it.positionImageSpace,
                )
            },
            triangles = triangles.map {
                FaceMeshTriangle(
                    a = it.a,
                    b = it.b,
                    c = it.c,
                )
            },
        )

    private fun applyDenseSurfaceResult(
        result:
            com.nivukx.nox.core.reconstruction
                .DenseMultiViewSurfaceReconstructionEngine.Result,
    ) {
        _denseSurfaceTrustedPointRatio.value =
            result.trustedPointRatio.coerceIn(0.0, 1.0)
        _denseSurfaceUncertaintyP95Meters.value =
            result.uncertaintyP95Meters
                .takeIf(Double::isFinite)
                ?: Double.POSITIVE_INFINITY
        _denseSurfaceTopologyStability.value =
            result.topologyStability.coerceIn(0.0, 1.0)
        _denseSurfaceViewDiversity.value =
            result.viewDiversity.coerceIn(0.0, 1.0)
        _denseSurfaceProbabilisticEvidenceScore.value =
            result.meanProbabilisticEvidenceScore
                .coerceIn(0.0, 1.0)
        _denseSurfaceAdaptiveViewSelectionScore.value =
            result.adaptiveViewSelectionScore.coerceIn(0.0, 1.0)
        _denseSurfaceAdaptiveViewDiversityScore.value =
            result.adaptiveViewDiversityScore.coerceIn(0.0, 1.0)
        _denseSurfaceExcludedWeakViewCount.value =
            result.excludedWeakViewCount.coerceAtLeast(0)
        _denseSurfaceMeanReprojectionErrorPx.value =
            result.meanReprojectionErrorPx
                .takeIf(Double::isFinite)
                ?: Double.POSITIVE_INFINITY
        _denseSurfaceMeanAppearanceConsistency.value =
            result.meanAppearanceConsistency.coerceIn(0.0, 1.0)
        _denseSurfaceMeanSilhouetteConsistency.value =
            result.meanSilhouetteConsistency.coerceIn(0.0, 1.0)
        _denseSurfaceMeanMotionBoundaryConfidence.value =
            result.meanMotionBoundaryConfidence.coerceIn(0.0, 1.0)
        _denseSurfaceMeanLightingStability.value =
            result.meanLightingStability.coerceIn(0.0, 1.0)
        _denseSurfaceHardFailure.value =
            result.hardFailure
    }

    private fun updateSecondaryFusion(
        timestampNanos: Long,
        faceMeshes: List<FaceMeshGeometryEngine.Result> = emptyList(),
    ) {
        val snapshot = calibrationSnapshot ?: return

        var faceGeometryFusionConfidence = 0.0
        var denseFaceFusionConfidence = 0.0

        faceFusion.reconstruct(
            timestampNanos,
            snapshot,
        )?.let {
            val center = averageFinitePoint(it.pointsBySemanticId.values)
            if (center != null) {
                latestFaceCenter3D = center
                latestFaceCenterTimestampNanos = timestampNanos
            }

            val temporal = faceMeshTemporalConsistencyEngine.evaluate(
                FaceMeshTemporalConsistencyEngine.Input(
                    timestampNanos = timestampNanos,
                    pointsBySemanticId = it.pointsBySemanticId,
                    baseConfidence = it.confidence,
                    meanReprojectionErrorPx = it.meanReprojectionErrorPx,
                ),
            )

            val shape = faceShapeAlignmentEngine.evaluate(
                HighEndFaceShapeAlignmentEngine.Input(
                    timestampNanos = timestampNanos,
                    pointsBySemanticId = it.pointsBySemanticId,
                    baseConfidence = it.confidence,
                ),
            )

            latestFaceShapeConfidence =
                shape.confidence.coerceIn(0.0, 1.0)
            latestFaceShapeAccepted =
                shape.accepted
            latestFacePointCount =
                it.pointsBySemanticId.size

            val faceGeometryConfidence = (
                temporal.confidence * 0.62 +
                    shape.confidence * 0.38
            ).coerceIn(0.0, 1.0)

            faceGeometryFusionConfidence =
                faceGeometryConfidence

            _face3DPoints.value = it.points
            _face3DErrorPx.value =
                it.meanReprojectionErrorPx *
                    (
                        1.0 +
                            (1.0 - faceGeometryConfidence) * 0.35
                        )
            _face3DTemporalConsistencyConfidence.value =
                faceGeometryConfidence
            _face3DTemporalConsistencyWarnings.value =
                (
                    temporal.warnings +
                        shape.warnings.map { warning ->
                            "shape-$warning"
                        }
                ).distinct()
        }

        val denseFaceMesh = faceMeshes
            .maxByOrNull { mesh ->
                (
                    mesh.confidence * 0.52 +
                        mesh.pointCoverage * 0.28 +
                        mesh.triangleCoverage * 0.20
                    ).coerceIn(0.0, 1.0)
            }

        denseFaceMesh?.let { mesh ->
            latestDenseFaceObserved = true
            val meshCoverageScore = (
                mesh.pointCoverage * 0.46 +
                    mesh.triangleCoverage * 0.18 +
                    mesh.confidence * 0.24 +
                    mesh.faceAreaRatio.coerceIn(0.0, 1.0) * 0.12
                ).coerceIn(0.0, 1.0)

            val denseGeometryConfidence = (
                meshCoverageScore * 0.76 +
                    _face3DTemporalConsistencyConfidence.value * 0.14 +
                    _faceEyeConsistencyConfidence.value * 0.10
                ).coerceIn(0.0, 1.0)

            denseFaceFusionConfidence =
                maxOf(
                    denseFaceFusionConfidence,
                    denseGeometryConfidence,
                )

            _face3DPoints.value =
                maxOf(
                    _face3DPoints.value,
                    mesh.pointCount,
                )

            val meshErrorEquivalentPx = (
                (
                    1.0 - meshCoverageScore
                    ) * 8.0 +
                    (1.0 - mesh.confidence) * 3.0
                ).coerceAtLeast(0.25)

            _face3DErrorPx.value =
                if (_face3DErrorPx.value.isFinite()) {
                    minOf(
                        _face3DErrorPx.value,
                        meshErrorEquivalentPx,
                    )
                } else {
                    meshErrorEquivalentPx
                }

            _face3DTemporalConsistencyConfidence.value =
                (
                    _face3DTemporalConsistencyConfidence.value * 0.78 +
                        denseGeometryConfidence * 0.22
                    ).coerceIn(0.0, 1.0)

            _face3DTemporalConsistencyWarnings.value =
                (
                    _face3DTemporalConsistencyWarnings.value +
                        buildList {
                            if (mesh.faceAreaRatio < 0.025) {
                                add("dense-face-mesh-subject-too-small")
                            }
                            if (mesh.pointCoverage < 0.82) {
                                add("dense-face-mesh-point-coverage-degraded")
                            }
                            if (mesh.triangleCoverage < 0.72) {
                                add("dense-face-mesh-triangle-coverage-degraded")
                            }
                            if (mesh.depthSpreadImageUnits < 0.50) {
                                add("dense-face-mesh-low-depth-variation")
                            }
                        }
                    )
                    .distinct()
                    .take(10)
        }

        if (denseFaceMesh == null) {
            latestDenseFaceObserved = false
        }

        faceMeshFusion.reconstruct(
            timestampNanos = timestampNanos,
            calibration = snapshot,
        )?.let { surface ->
            val temporalSurface =
                faceMeshSurfaceTemporalEngine.process(
                    FaceMeshSurfaceTemporalEngine.Input(
                        timestampNanos = timestampNanos,
                        surface = surface,
                        preserveMicroMotion =
                            _motionCustomization.value
                                .preserveMicroMotion,
                    ),
                )

            val stableSurface =
                temporalSurface.surface

            val metricMeshConfidence = (
                stableSurface.confidence * 0.40 +
                    stableSurface.acceptedPointRatio * 0.18 +
                    temporalSurface.diagnostics.finalScore * 0.20 +
                    (
                        if (stableSurface.sourceDevices.size >= 3) {
                            1.0
                        } else {
                            0.68
                        }
                    ) * 0.08 +
                    (
                        if (stableSurface.triangleCount >= 350) {
                            1.0
                        } else {
                            0.55
                        }
                    ) * 0.06 +
                    (
                        if (stableSurface.meanReprojectionErrorPx.isFinite()) {
                            kotlin.math.exp(
                                -stableSurface.meanReprojectionErrorPx / 4.5,
                            )
                        } else {
                            0.0
                        }
                    ) * 0.08
                ).coerceIn(0.0, 1.0)

            denseFaceFusionConfidence =
                maxOf(
                    denseFaceFusionConfidence,
                    metricMeshConfidence,
                )

            _face3DPoints.value =
                maxOf(
                    _face3DPoints.value,
                    stableSurface.pointCount,
                )

            _face3DErrorPx.value =
                if (stableSurface.meanReprojectionErrorPx.isFinite()) {
                    minOf(
                        _face3DErrorPx.value.takeIf { it.isFinite() }
                            ?: Double.POSITIVE_INFINITY,
                        stableSurface.meanReprojectionErrorPx,
                    )
                } else {
                    _face3DErrorPx.value
                }

            _face3DTemporalConsistencyConfidence.value =
                (
                    _face3DTemporalConsistencyConfidence.value * 0.62 +
                        metricMeshConfidence * 0.20 +
                        temporalSurface.diagnostics.finalScore * 0.18
                    ).coerceIn(0.0, 1.0)

            _face3DTemporalConsistencyWarnings.value =
                (
                    _face3DTemporalConsistencyWarnings.value +
                        stableSurface.warnings.map {
                            "metric-face-$it"
                        } +
                        temporalSurface.diagnostics.warnings.map {
                            "metric-face-temporal-$it"
                        }
                    )
                    .distinct()
                    .take(16)
        }

        val leftEye =
            eyeFusion.reconstruct(
                timestampNanos,
                snapshot,
                EyeSide.LEFT,
            )
        val rightEye =
            eyeFusion.reconstruct(
                timestampNanos,
                snapshot,
                EyeSide.RIGHT,
            )

        val eyeTemporal =
            eye3DTemporalConsistencyEngine.evaluate(
                Eye3DTemporalConsistencyEngine.Input(
                    timestampNanos = timestampNanos,
                    left = leftEye,
                    right = rightEye,
                ),
            )

        val binocular =
            binocularGazeGeometryEngine.evaluate(
                left = leftEye,
                right = rightEye,
            )

        _eye3DTemporalConsistencyConfidence.value =
            (
                eyeTemporal.confidence * 0.78 +
                    binocular.confidence * 0.22
                ).coerceIn(0.0, 1.0)
        _eye3DTemporalConsistencyWarnings.value =
            (
                eyeTemporal.warnings +
                    binocular.warnings.map { "binocular-$it" }
                ).distinct().take(10)

        val eyeSaccadeVelocity =
            maxOf(
                leftEye?.angularVelocityDegreesPerSecond ?: 0.0,
                rightEye?.angularVelocityDegreesPerSecond ?: 0.0,
            ).coerceAtLeast(0.0)

        _faceEyeSaccadeVelocityDegreesPerSecond.value =
            eyeSaccadeVelocity

        val eyeBaseConfidence = listOfNotNull(
            leftEye?.confidence,
            rightEye?.confidence,
        ).let { averageOrZero(it) }

        _eye3DConfidence.value = (
            eyeBaseConfidence * 0.58 +
                eyeTemporal.confidence * 0.24 +
                binocular.confidence * 0.18
            ).coerceIn(0.0, 1.0)

        val faceEyeFusionConfig =
            faceEyeEvidenceFusionEngine.configuration()
        faceEyeEvidenceFusionEngine.configure(
            faceEyeFusionConfig.copy(
                angularSpreadWarningDegrees =
                    (
                        4.0 +
                            (
                                1.0 -
                                    _motionCustomization.value
                                        .outputIntegrityStrictness
                                        .coerceIn(0.0, 1.0)
                                ) * 2.0
                        ).coerceIn(4.0, 6.0),
                angularSpreadRejectDegrees =
                    (
                        10.0 +
                            _motionCustomization.value
                                .outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 4.0
                        ).coerceIn(10.0, 14.0),
                minimumParallaxDegrees =
                    _motionCustomization.value.minimumParallaxDegrees
                        .coerceIn(0.20, 2.50),
                strongParallaxDegrees =
                    (
                        _motionCustomization.value.minimumParallaxDegrees
                            .coerceIn(0.20, 2.50) * 2.35
                        ).coerceIn(0.75, 5.50),
                eyeTemporalWeight =
                    (
                        0.16 +
                            _motionCustomization.value.eyeTemporalStrictness
                                .coerceIn(0.0, 1.0) * 0.18
                        ).coerceIn(0.16, 0.34),
                binocularWeight =
                    (
                        0.22 +
                            _motionCustomization.value.gazePriority
                                .coerceIn(0.0, 1.0) * 0.18
                        ).coerceIn(0.22, 0.40),
                captureLightingWeight =
                    (
                        0.04 +
                            _motionCustomization.value
                                .faceEyeFusionStrictness
                                .coerceIn(0.0, 1.0) * 0.06
                        ).coerceIn(0.04, 0.10),
                captureScaleWeight =
                    (
                        0.03 +
                            _motionCustomization.value
                                .faceEyeFusionStrictness
                                .coerceIn(0.0, 1.0) * 0.06
                        ).coerceIn(0.03, 0.09),
                rejectFusedConfidence =
                    (
                        0.40 +
                            _motionCustomization.value
                                .outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.08 +
                            _motionCustomization.value
                                .faceEyeFusionStrictness
                                .coerceIn(0.0, 1.0) * 0.14
                        ).coerceIn(0.40, 0.62),
                binocularRejectThreshold =
                    (
                        0.54 -
                            _motionCustomization.value
                                .faceEyeFusionStrictness
                                .coerceIn(0.0, 1.0) * 0.16
                        ).coerceIn(0.34, 0.54),
                recoveryEyeConfidence =
                    (
                        0.58 +
                            _motionCustomization.value
                                .faceEyeFusionStrictness
                                .coerceIn(0.0, 1.0) * 0.18
                        ).coerceIn(0.58, 0.78),
                recoveryBinocularConfidence =
                    (
                        0.52 +
                            _motionCustomization.value
                                .faceEyeFusionStrictness
                                .coerceIn(0.0, 1.0) * 0.20
                        ).coerceIn(0.52, 0.82),
                historyAlpha =
                    (
                        0.14 +
                            _motionCustomization.value
                                .temporalSmoothing
                                .coerceIn(0.0, 1.0) * 0.14
                        ).coerceIn(0.14, 0.28),
                quarantineAfterBadFrames =
                    (
                        3 +
                            _motionCustomization.value
                                .outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 2.0
                        ).toInt().coerceIn(3, 5),
                maximumIpdDeviationMeters =
                    (
                        _motionCustomization.value
                            .faceEyeIpdDriftToleranceMillimeters
                            .coerceIn(4.0, 30.0) / 1000.0
                        ).coerceIn(0.004, 0.030),
            ),
        )

        val leftEyeUncertainty =
            leftEye?.centerUncertaintyMeters
                ?: 0.08
        val rightEyeUncertainty =
            rightEye?.centerUncertaintyMeters
                ?: 0.08
        val eyeUncertainty =
            maxOf(
                leftEyeUncertainty,
                rightEyeUncertainty,
            ).coerceAtLeast(0.0)

        val faceConfidenceReference =
            minOf(
                faceGeometryFusionConfidence
                    .takeIf { it > 0.0 }
                    ?: _face3DTemporalConsistencyConfidence.value,
                _face3DTemporalConsistencyConfidence.value
                    .coerceIn(0.0, 1.0),
                denseFaceFusionConfidence
                    .takeIf { it > 0.0 }
                    ?: _face3DTemporalConsistencyConfidence.value,
            ).coerceIn(0.0, 1.0)

        val faceOodRisk =
            (
                (1.0 - faceConfidenceReference) * 0.62 +
                    (
                        if (_face3DErrorPx.value.isFinite()) {
                            (_face3DErrorPx.value / 8.0)
                                .coerceIn(0.0, 1.0)
                        } else {
                            1.0
                        }
                    ) * 0.18 +
                    if (
                        _face3DTemporalConsistencyWarnings.value.any {
                            it.contains("degraded") ||
                                it.contains("unstable")
                        }
                    ) 0.20 else 0.0
                ).coerceIn(0.0, 1.0)

        val eyeConfidenceReference =
            minOf(
                eyeBaseConfidence,
                eyeTemporal.confidence.coerceIn(0.0, 1.0),
                binocular.confidence.coerceIn(0.0, 1.0),
            )

        val eyeOodRisk =
            (
                (1.0 - eyeConfidenceReference) * 0.52 +
                    (binocular.angularDisagreementDegrees / 18.0)
                        .coerceIn(0.0, 1.0) * 0.24 +
                    (1.0 - eyeUncertaintyScore(
                        leftEye?.centerUncertaintyMeters,
                        rightEye?.centerUncertaintyMeters,
                    )) * 0.14 +
                    if (
                        eyeTemporal.warnings.any {
                            it.contains("reject") ||
                                it.contains("unstable") ||
                                it.contains("outlier")
                        }
                    ) 0.10 else 0.0
                ).coerceIn(0.0, 1.0)

        val faceFreshnessScore =
            evidenceFreshnessScore(
                timestampNanos = timestampNanos,
                sourceTimestampNanos = latestFaceCenterTimestampNanos,
                halfLifeSeconds = 0.16,
                hardStaleSeconds = 0.72,
            )
        val leftEyeFreshnessScore =
            evidenceFreshnessScore(
                timestampNanos = timestampNanos,
                sourceTimestampNanos = latestLeftEyeTimestampNanos,
                halfLifeSeconds = 0.08,
                hardStaleSeconds = 0.36,
            )
        val rightEyeFreshnessScore =
            evidenceFreshnessScore(
                timestampNanos = timestampNanos,
                sourceTimestampNanos = latestRightEyeTimestampNanos,
                halfLifeSeconds = 0.08,
                hardStaleSeconds = 0.36,
            )
        val eyeFreshnessScore =
            minOf(
                leftEyeFreshnessScore,
                rightEyeFreshnessScore,
            )

        val faceEyeFusion =
            faceEyeEvidenceFusionEngine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = timestampNanos,
                    faceGeometryConfidence =
                        faceGeometryFusionConfidence
                            .takeIf { it > 0.0 }
                            ?: _face3DTemporalConsistencyConfidence.value,
                    faceTemporalConfidence =
                        _face3DTemporalConsistencyConfidence.value
                            .coerceIn(0.0, 1.0),
                    denseMeshConfidence =
                        denseFaceFusionConfidence
                            .takeIf { it > 0.0 }
                            ?: _face3DTemporalConsistencyConfidence.value,
                    faceShapeConfidence =
                        latestFaceShapeConfidence
                            .coerceIn(0.0, 1.0),
                    faceShapeAccepted =
                        latestFaceShapeAccepted,
                    denseMeshObserved =
                        latestDenseFaceObserved,
                    facePointCount =
                        maxOf(
                            latestFacePointCount,
                            if (latestDenseFaceObserved) {
                                denseFaceMesh?.pointCount ?: 0
                            } else {
                                0
                            },
                        ),
                    faceUncertaintyMeters =
                        (
                            0.012 +
                                (
                                    1.0 -
                                        faceGeometryFusionConfidence
                                    ).coerceIn(0.0, 1.0) *
                                    0.090
                            ).coerceIn(0.012, 0.120),
                    leftEyeConfidence =
                        leftEye?.confidence
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                    rightEyeConfidence =
                        rightEye?.confidence
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                    faceOutOfDistributionRisk =
                        faceOodRisk,
                    eyeOutOfDistributionRisk =
                        eyeOodRisk,
                    binocularConfidence =
                        binocular.confidence.coerceIn(0.0, 1.0),
                    eyeTemporalConfidence =
                        eyeTemporal.confidence.coerceIn(0.0, 1.0),
                    eyeTemporalAccepted =
                        eyeTemporal.accepted,
                    binocularAccepted =
                        binocular.accepted,
                    eyeUncertaintyMeters =
                        eyeUncertainty.coerceIn(0.0, 0.50),

                    angularSpreadDegrees =
                        binocular.angularDisagreementDegrees
                            .coerceAtLeast(0.0),
                    parallaxDegrees =
                        maxOf(
                            leftEye?.parallaxDegrees ?: 0.0,
                            rightEye?.parallaxDegrees ?: 0.0,
                        ),
                    saccadeVelocityDegreesPerSecond =
                        maxOf(
                            leftEye?.angularVelocityDegreesPerSecond
                                ?: 0.0,
                            rightEye?.angularVelocityDegreesPerSecond
                                ?: 0.0,
                        ),
                    gazeAccelerationDegreesPerSecond2 =
                        eyeTemporal.gazeAccelerationDegreesPerSecond2
                            .coerceAtLeast(0.0),
                    blinkProbability =
                        _gazeBlinkProbability.value
                            .coerceIn(0.0, 1.0),
                    faceFresh =
                        faceFreshnessScore >= 0.36,
                    eyesFresh =
                        eyeFreshnessScore >= 0.36,
                    faceFreshness =
                        faceFreshnessScore,
                    faceReprojectionScore =
                        if (_face3DErrorPx.value.isFinite()) {
                            kotlin.math.exp(
                                -_face3DErrorPx.value / 4.0,
                            ).coerceIn(0.0, 1.0)
                        } else {
                            0.0
                        },
                    lightingConfidence =
                        _lightingConfidence.value
                            .coerceIn(0.0, 1.0),
                    faceScaleConfidence =
                        (
                            denseFaceMesh?.faceAreaRatio
                                ?.let { ratio ->
                                    (
                                        (ratio / 0.025)
                                            .coerceIn(0.0, 1.0)
                                    )
                                }
                                ?: if (latestDenseFaceObserved) {
                                    0.86
                                } else {
                                    0.68
                                }
                            ).coerceIn(0.0, 1.0),
                    gazeStability =
                        (
                            eyeTemporal.temporalConfidence * 0.56 +
                                eyeTemporal.bilateralConfidence * 0.20 +
                                binocular.confidence * 0.24
                            ).coerceIn(0.0, 1.0),
                    gazeAgreement =
                        eyeTemporal.gazeAgreementScore
                            .coerceIn(0.0, 1.0),
                    interpupillaryDistanceMeters =
                        binocular.interpupillaryDistanceMeters
                            .coerceAtLeast(0.0),
                ),
            )

        val adaptiveFaceEyeFusion =
            adaptiveFaceEyeEvidenceFusionEngine.evaluate(
                AdaptiveFaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = timestampNanos,
                    faceGeometryConfidence =
                        faceEyeFusion.faceConfidence
                            .coerceIn(0.0, 1.0),
                    faceTemporalConfidence =
                        _face3DTemporalConsistencyConfidence.value
                            .coerceIn(0.0, 1.0),
                    denseFaceMeshConfidence =
                        denseFaceFusionConfidence
                            .takeIf { it > 0.0 }
                            ?: faceEyeFusion.faceConfidence,
                    faceFreshness =
                        faceFreshnessScore.coerceIn(0.0, 1.0),
                    faceReprojectionScore =
                        faceEyeFusion.faceReprojectionScore
                            .coerceIn(0.0, 1.0),
                    leftEyeConfidence =
                        leftEye?.confidence
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                    rightEyeConfidence =
                        rightEye?.confidence
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                    eyeTemporalConfidence =
                        eyeTemporal.confidence
                            .coerceIn(0.0, 1.0),
                    binocularConfidence =
                        binocular.confidence
                            .coerceIn(0.0, 1.0),
                    eyeUncertaintyMeters =
                        maxOf(
                            leftEye?.centerUncertaintyMeters ?: 0.08,
                            rightEye?.centerUncertaintyMeters ?: 0.08,
                        ).coerceAtLeast(0.0),
                    gazeStability =
                        faceEyeFusion.gazeStabilityScore
                            .coerceIn(0.0, 1.0),
                    gazeAgreement =
                        faceEyeFusion.gazeAgreementScore
                            .coerceIn(0.0, 1.0),
                    saccadeVelocityDegreesPerSecond =
                        maxOf(
                            leftEye?.angularVelocityDegreesPerSecond ?: 0.0,
                            rightEye?.angularVelocityDegreesPerSecond ?: 0.0,
                        ).coerceAtLeast(0.0),
                    blinkProbability =
                        _gazeBlinkProbability.value
                            .coerceIn(0.0, 1.0),
                    faceShapeConfidence =
                        latestFaceShapeConfidence
                            .coerceIn(0.0, 1.0),
                    faceShapeAccepted =
                        latestFaceShapeAccepted,
                    denseFaceMeshObserved =
                        latestDenseFaceObserved,
                    eyeTemporalAccepted =
                        eyeTemporal.accepted,
                    binocularAccepted =
                        binocular.accepted,
                    gazeAccelerationDegreesPerSecond2 =
                        eyeTemporal.gazeAccelerationDegreesPerSecond2
                            .coerceAtLeast(0.0),
                ),
            )

        val corroboratedFaceEyeConfidence =
            (
                faceEyeFusion.fusedConfidence * 0.78 +
                    adaptiveFaceEyeFusion.confidence * 0.22
                ).coerceIn(0.0, 1.0)

        val adaptiveFaceEyeWarnings =
            adaptiveFaceEyeFusion.warnings.map {
                "adaptive-face-eye-$it"
            }

        _face3DTemporalConsistencyConfidence.value =
            faceEyeFusion.faceConfidence
                .coerceIn(0.0, 1.0)
        _eye3DTemporalConsistencyConfidence.value =
            faceEyeFusion.eyeConfidence
                .coerceIn(0.0, 1.0)
        _eye3DConfidence.value =
            (
                _eye3DConfidence.value * 0.28 +
                    corroboratedFaceEyeConfidence * 0.72
                ).coerceIn(0.0, 1.0)

        _faceEyeFusionConfidence.value =
            corroboratedFaceEyeConfidence
        _faceEyeFusionAccepted.value =
            faceEyeFusion.accepted
        _faceEyeFusionMode.value =
            faceEyeFusion.fusionMode.name
        _faceEyeIndependentEvidenceSupport.value =
            faceEyeFusion.independentEvidenceSupport
                .coerceIn(0.0, 1.0)
        _faceEyeReliabilityTrend.value =
            (
                faceEyeFusion.faceReliabilityTrend +
                    faceEyeFusion.eyeReliabilityTrend
                ) * 0.5
        _faceEyeFusionWarnings.value =
            (
                faceEyeFusion.warnings +
                    adaptiveFaceEyeWarnings
            ).distinct().take(20)

        _faceEyeCaptureDomainConfidence.value =
            faceEyeFusion.captureDomainConfidence
                .coerceIn(0.0, 1.0)
        _faceEyeLightingConfidence.value =
            faceEyeFusion.lightingConfidence
                .coerceIn(0.0, 1.0)
        _faceEyeScaleConfidence.value =
            faceEyeFusion.faceScaleConfidence
                .coerceIn(0.0, 1.0)

        _faceEyeUncertaintyCalibrationConfidence.value =
            faceEyeFusion.uncertaintyCalibrationConfidence
                .coerceIn(0.0, 1.0)
        _faceEyeUncertaintyCalibrationReady.value =
            faceEyeFusion.uncertaintyCalibrationReady
        _faceUncertaintyCalibrationMultiplier.value =
            faceEyeFusion.faceUncertaintyCalibrationMultiplier
                .coerceAtLeast(0.0)
        _eyeUncertaintyCalibrationMultiplier.value =
            faceEyeFusion.eyeUncertaintyCalibrationMultiplier
                .coerceAtLeast(0.0)
        _faceUncertaintyCoverage.value =
            faceEyeFusion.faceUncertaintyCoverage
                .coerceIn(0.0, 1.0)
        _eyeUncertaintyCoverage.value =
            faceEyeFusion.eyeUncertaintyCoverage
                .coerceIn(0.0, 1.0)

        _faceEyeFailureAttribution.value =
            faceEyeFusion.failureAttribution
                .take(5)
                .map {
                    "${it.domain.name}:%.0f%%".format(
                        it.severity * 100.0,
                    )
                }
        _faceEyeGeometricIntegrity.value =
            faceEyeFusion.geometricIntegrityScore
                .coerceIn(0.0, 1.0)
        _faceEyeFaceFreshnessScore.value =
            faceEyeFusion.faceFreshnessScore
                .coerceIn(0.0, 1.0)
        _faceEyeGazeAgreementScore.value =
            faceEyeFusion.gazeAgreementScore
                .coerceIn(0.0, 1.0)

        _faceEyeRayGeometryCondition.value =
            binocular.rayGeometryCondition
                .coerceIn(0.0, 1.0)

        _faceEyeIpdConsistencyScore.value =
            faceEyeFusion.ipdConsistencyScore
                .coerceIn(0.0, 1.0)

        val faceEyeCalibrationDrift =
            faceEyeCalibrationDriftEngine.evaluate(
                FaceEyeCalibrationDriftEngine.Input(
                    timestampNanos = timestampNanos,
                    interpupillaryDistanceMeters =
                        binocular.interpupillaryDistanceMeters
                            .coerceAtLeast(0.0),
                    angularDisagreementDegrees =
                        binocular.angularDisagreementDegrees
                            .coerceAtLeast(0.0),
                    reprojectionErrorPx =
                        _face3DErrorPx.value
                            .takeIf { it.isFinite() }
                            ?.coerceAtLeast(0.0)
                            ?: 0.0,
                    binocularConfidence =
                        binocular.confidence
                            .coerceIn(0.0, 1.0),
                    faceEyeFusionConfidence =
                        faceEyeFusion.fusedConfidence
                            .coerceIn(0.0, 1.0),
                ),
            )

        val metricFaceMesh =
            faceMeshes
                .maxByOrNull {
                    (
                        it.confidence * 0.35 +
                            it.pointCoverage * 0.35 +
                            it.triangleCoverage * 0.30
                    )
                }
                ?.toModelFrame()

        val metricEvidence =
            faceEyeMetricEvidenceEngine.evaluate(
                FaceEyeMetricEvidenceEngine.Input(
                    face = metricFaceMesh,
                    leftEye = leftEye,
                    rightEye = rightEye,
                    calibrationConfidence =
                        _rigScore.value.coerceIn(0.0, 1.0),
                    temporalStability =
                        (
                            _face3DTemporalConsistencyConfidence.value * 0.70 +
                                eyeTemporal.confidence * 0.30
                        ).coerceIn(0.0, 1.0),
                    gazeCalibrationConfidence =
                        binocular.confidence.coerceIn(0.0, 1.0),
                    faceCalibrationDrift =
                        (
                            faceEyeCalibrationDrift.driftScore *
                                FaceEyeMetricEvidenceEngine.Config()
                                    .maximumCalibrationDrift
                        ).coerceIn(0.0, 1.0),
                    headMotionConfidence =
                        (
                            1.0 -
                                _cameraMotionRisk.value
                        ).coerceIn(0.0, 1.0),
                ),
            )

        _faceEyeMetricState.value = metricEvidence.state.name
        _faceEyeMetricConfidence.value =
            metricEvidence.confidence.coerceIn(0.0, 1.0)
        _faceEyeMetricEyeUncertaintyMm.value =
            metricEvidence.eyeP95UncertaintyMm
        _faceEyeMetricWarnings.value =
            metricEvidence.warnings
        _faceEyeMetricRecommendations.value =
            metricEvidence.recommendations

        _faceEyeCalibrationDriftScore.value =
            faceEyeCalibrationDrift.driftScore
                .coerceIn(0.0, 1.0)
        _faceEyeCalibrationDriftPersistent.value =
            faceEyeCalibrationDrift.persistent
        _faceEyeRecalibrationRecommended.value =
            faceEyeCalibrationDrift.recalibrationRecommended
        _faceEyeCalibrationDriftWarnings.value =
            faceEyeCalibrationDrift.warnings

        _faceEyeOutOfDistributionRisk.value =
            maxOf(
                faceEyeFusion.faceOutOfDistributionRisk,
                faceEyeFusion.eyeOutOfDistributionRisk,
                faceEyeCalibrationDrift.driftScore * 0.42,
            ).coerceIn(0.0, 1.0)

        _faceEyeShapeConfidence.value =
            faceEyeFusion.faceShapeConfidence
                .coerceIn(0.0, 1.0)
        _faceEyeEyeDynamicConsistency.value =
            faceEyeFusion.eyeDynamicConsistency
                .coerceIn(0.0, 1.0)
        _faceEyeHardFailureReasons.value =
            faceEyeFusion.hardFailureReasons

        if (_state.value == CaptureState.RECORDING) {
            recorder.appendFaceEyeEvidenceFusion(
                faceEyeFusion,
            )
            recorder.appendFaceEyeCalibrationDrift(
                faceEyeCalibrationDrift,
            )
        }

        _face3DTemporalConsistencyWarnings.value =
            (
                _face3DTemporalConsistencyWarnings.value +
                    faceEyeFusion.warnings.map {
                        "face-eye-$it"
                    }
                ).distinct()
                .take(16)

        if (
            faceEyeCalibrationDrift.recalibrationRecommended
        ) {
            _reconstructionGateReasons.value =
                (
                    _reconstructionGateReasons.value +
                        faceEyeCalibrationDrift.warnings.map {
                            "face-eye-calibration-$it"
                        }
                    )
                    .distinct()
                    .takeLast(14)
        }

        if (!faceEyeFusion.accepted) {
            _reconstructionGateReasons.value =
                (
                    _reconstructionGateReasons.value +
                        faceEyeFusion.warnings.map {
                            "face-eye-gate-$it"
                        }
                    )
                    .distinct()
                    .takeLast(12)
        }

        leftEye?.let {
            latestLeftEye3D = it.centerWorld
            latestLeftEyeTimestampNanos = timestampNanos
        }
        rightEye?.let {
            latestRightEye3D = it.centerWorld
            latestRightEyeTimestampNanos = timestampNanos
        }
    }

    private fun averageOrZero(values: List<Double>): Double =
        if (values.isEmpty()) 0.0 else values.average()

    private fun eyeUncertaintyScore(
        leftMeters: Double?,
        rightMeters: Double?,
    ): Double {
        val uncertainty =
            maxOf(
                leftMeters ?: 0.08,
                rightMeters ?: 0.08,
            ).coerceAtLeast(0.0)
        return (
            1.0 -
                uncertainty / 0.10
        ).coerceIn(0.0, 1.0)
    }

    private fun evidenceFreshnessScore(
        timestampNanos: Long,
        sourceTimestampNanos: Long,
        halfLifeSeconds: Double,
        hardStaleSeconds: Double,
    ): Double {
        if (
            sourceTimestampNanos == Long.MIN_VALUE ||
            timestampNanos < sourceTimestampNanos
        ) {
            return 0.0
        }

        val ageSeconds =
            (
                timestampNanos -
                    sourceTimestampNanos
                ).toDouble() / 1e9

        if (!ageSeconds.isFinite() ||
            ageSeconds >= hardStaleSeconds
        ) {
            return 0.0
        }

        return kotlin.math.exp(
            -ageSeconds /
                halfLifeSeconds.coerceAtLeast(1e-3),
        ).coerceIn(0.0, 1.0)
    }

    private fun leftEyeUncertaintyMeters(): Double {
        val confidence =
            _eye3DConfidence.value.coerceIn(0.0, 1.0)
        return (
            0.008 +
                (1.0 - confidence) * 0.072
            ).coerceIn(0.008, 0.080)
    }

    private fun rightEyeUncertaintyMeters(): Double {
        val confidence =
            _eye3DConfidence.value.coerceIn(0.0, 1.0)
        return (
            0.008 +
                (1.0 - confidence) * 0.072
            ).coerceIn(0.008, 0.080)
    }

    private fun configurePerceptionCalibration(
        snapshot: CalibrationSnapshot,
    ) {
        val geometry =
            snapshot.intrinsics.keys
                .intersect(snapshot.extrinsics.keys)
                .associate { deviceId ->
                    deviceId.value to
                        PerceptionPipeline.CalibratedCameraGeometry(
                            intrinsics =
                                snapshot.intrinsics.getValue(deviceId),
                            pose =
                                snapshot.extrinsics.getValue(deviceId).pose,
                            uncertainty =
                                snapshot.uncertaintyProfiles[deviceId]
                                    ?: CalibrationUncertaintyProfile(),
                        )
                }
        perception.configureCalibratedCameraGeometry(geometry)
    }

    private fun updateRolesFromSnapshot(snapshot: CalibrationSnapshot) {
        val referenceDevice = snapshot.extrinsics.values
            .firstOrNull { it.referenceDevice != null }
            ?.referenceDevice
            ?: return
        val referencePose = snapshot.extrinsics[referenceDevice]?.pose ?: return

        val north = referencePose.rotation
            .rotate(Vec3(0.0, 0.0, 1.0))
            .normalized()
        val up = referencePose.rotation
            .rotate(Vec3(0.0, 1.0, 0.0))
            .normalized()
        val east = up.cross(north).normalized()

        val candidates = snapshot.extrinsics.map { (id, extrinsic) ->
            DeviceRoleAssignmentEngine.Candidate(
                id = id,
                forward = extrinsic.pose.rotation
                    .rotate(Vec3(0.0, 0.0, 1.0))
                    .normalized(),
                up = extrinsic.pose.rotation
                    .rotate(Vec3(0.0, 1.0, 0.0))
                    .normalized(),
            )
        }

        roleAssignment.assign(
            candidates = candidates,
            north = north,
            east = east,
            up = up,
        ).forEach { assignedRoles[it.id] = it.role }

        assignedRoles[referenceDevice] = DeviceRole.MASTER
        updateCoverageReport()
    }

    private fun updateCoverageReport() {
        val assignments = assignedRoles.map { (id, role) ->
            DeviceRoleAssignmentEngine.Assignment(
                id = id,
                role = role,
                score = if (role == DeviceRole.MASTER) 1.0 else 0.80,
                directionDegrees = null,
            )
        }

        val report = rigCoverage.evaluate(
            assignments = assignments,
            profile = RigCoverageAnalyzer.CaptureProfile.FULL_VOLUME,
        )
        val regionReport =
            advancedRigCoveragePlanner.evaluate(
                assignments = assignments,
            )

        _rigCoverageScore.value =
            (
                report.score * 0.58 +
                    regionReport.globalScore * 0.42
                ).coerceIn(0.0, 1.0)

        _rigRecommendations.value =
            (
                report.recommendations +
                    regionReport.recommendations +
                    listOf(
                        "face-coverage=%.0f%%".format(
                            regionReport.faceScore * 100.0,
                        ),
                        "upper-body-coverage=%.0f%%".format(
                            regionReport.upperBodyScore * 100.0,
                        ),
                        "lower-body-coverage=%.0f%%".format(
                            regionReport.lowerBodyScore * 100.0,
                        ),
                        "hand-coverage=%.0f%%".format(
                            regionReport.handScore * 100.0,
                        ),
                        "foot-coverage=%.0f%%".format(
                            regionReport.footScore * 100.0,
                        ),
                        "surface-coverage=%.0f%%".format(
                            regionReport.surfaceScore * 100.0,
                        ),
                    )
                )
                .distinct()
                .take(16)
    }

    private fun sendClockProbes(endpoint: DeviceId) {
        repeat(6) { index ->
            viewModelScope.launch {
                delay(index * 100L)
                val sequence = packetSequence.incrementAndGet()
                val t1 = System.nanoTime()
                prunePendingPings(t1)
                pendingPings[sequence] = t1
                if (pendingPings.size > pendingPingCapacity) {
                    var oldestSequence: Long? = null
                    var oldestTimestamp = Long.MAX_VALUE
                    pendingPings.forEach { (candidateSequence, timestamp) ->
                        if (timestamp < oldestTimestamp) {
                            oldestTimestamp = timestamp
                            oldestSequence = candidateSequence
                        }
                    }
                    oldestSequence?.let(pendingPings::remove)
                }
                peers.send(
                    endpoint,
                    NoXWireProtocol.encode(
                        NoXWireProtocol.Type.PING,
                        sequence,
                        t1,
                    ),
                )
            }
        }
    }

    private fun prunePendingPings(nowNanos: Long) {
        pendingPings.entries.removeIf { (_, sentAt) ->
            nowNanos - sentAt >
                pendingPingTtlNanos
        }
    }

    private fun refreshDevices(connected: Set<DeviceId>) {
        peerCapabilities.keys
            .filter { it !in connected }
            .forEach(peerCapabilities::remove)

        val localCapabilities = camera.capabilities()
        val local = DeviceDescriptor(
            id = localId,
            role = DeviceRole.MASTER,
            displayName = android.os.Build.MODEL,
            capabilities = localCapabilities,
            connected = true,
        )

        _devices.value = listOf(local) + connected.map { id ->
            val estimate = clock.estimate(id)
            DeviceDescriptor(
                id = id,
                role = assignedRoles[id] ?: DeviceRole.CAMERA,
                displayName = "Peer " + id.value.takeLast(8),
                capabilities = localCapabilities,
                connected = true,
                clockOffsetNanos = estimate.offsetNanos,
                roundTripNanos = estimate.roundTripNanos,
            )
        }
    }

    private fun geometricObservabilityScoreReset() {
        _geometricObservabilityScore.value = 0.0
        _geometricObservabilityWarnings.value = emptyList()
        _geometricInformationConditionScore.value = 0.0
        _geometricInformationP95MajorMeters.value = 0.0
        _geometricInformationP95DepthMeters.value = 0.0
        _denseIdentityGateQuality.value = 1.0
        _denseIdentityGateRejectedCount.value = 0
        _denseIdentityGateEvaluatedPoints.value = 0
        _denseIdentityGateWarnings.value = emptyList()
        _denseContactConfidence.value = 0.0
        _denseContactProtectedLandmarkCount.value = 0
        _denseContactWarnings.value = emptyList()
        _reconstructionUncertaintyConfidence.value = 1.0
        _reconstructionUncertaintyMeanMeters.value = 0.0
        _reconstructionUncertaintyP95Meters.value = 0.0
        _reconstructionTrustedJointRatio.value = 1.0
        _reconstructionUncertaintyBottleneck.value = "NONE"
        _reconstructionUncertaintyWarnings.value = emptyList()
        _directionalUncertaintyMajor95Meters.value = 0.0
        _directionalUncertaintyDepthMeters.value = 0.0
        _directionalUncertaintyAnisotropy.value = 1.0
        _directionalUncertaintyConfidence.value = 1.0
        _directionalUncertaintyDegradedJointCount.value = 0
        _directionalUncertaintyWarnings.value = emptyList()
        _adaptiveViewEvidenceScore.value = 1.0
        _adaptiveViewEvidenceGlobalRigTrust.value = 1.0
        _adaptiveViewEvidenceDegradedCameraCount.value = 0
        _adaptiveViewEvidenceHoldCameraCount.value = 0
        _adaptiveViewEvidenceDominantCamera.value = "NONE"
        _adaptiveViewEvidenceWarnings.value = emptyList()
        _cameraContributionSelectedCount.value = 0
        _cameraContributionExcludedCount.value = 0
        _cameraContributionDiversitySatisfied.value = false
        _cameraContributionReason.value = "RESET"
        _cameraContributionInformationGainScore.value = 0.0
        _jointViewSelectionScore.value = 0.0
        _jointViewDiversitySatisfiedRatio.value = 0.0
        _jointSelectedObservationCount.value = 0
        _jointExcludedObservationCount.value = 0
        _jointViewSelectionReasons.value = emptyList()
        _rigHealthState.value =
            MultiCameraRigHealthEngine.State.UNKNOWN.name
        _rigHealthScore.value = 0.0
        _rigQuarantinedCameraCount.value = 0
        _rigHealthWarnings.value = emptyList()
        _rigCameraHealth.value = emptyList()
    }

    private fun applyLightingEstimate(
        lighting: com.nivukx.nox.core.model.LightingEstimate,
    ) {
        val temporal =
            lightingTemporalConsistencyEngine.evaluate(lighting)

        recorder.appendLighting(lighting)
        recorder.appendLightingConsistency(temporal)
        _lightingLux.value = lighting.illuminanceLux
        _colorTemperature.value = lighting.colorTemperatureKelvin
        _lightingConfidence.value = (
            lighting.confidence * 0.68 +
                temporal.confidence * 0.32
            ).coerceIn(0.0, 1.0)
    }

    private fun runtimeThermalHeadroom(): Double {
        val powerManager =
            getApplication<Application>()
                .getSystemService(PowerManager::class.java)
                ?: return 1.0

        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> 1.0
            PowerManager.THERMAL_STATUS_LIGHT -> 0.85
            PowerManager.THERMAL_STATUS_MODERATE -> 0.65
            PowerManager.THERMAL_STATUS_SEVERE -> 0.45
            PowerManager.THERMAL_STATUS_CRITICAL -> 0.25
            PowerManager.THERMAL_STATUS_EMERGENCY -> 0.10
            PowerManager.THERMAL_STATUS_SHUTDOWN -> 0.0
            else -> 0.55
        }
    }

    private fun refreshTsdfUncertaintyTelemetry() {
        val metrics = denseFusion.uncertaintyMetrics()
        _denseTsdfMeanUncertainty.value =
            metrics.meanSignedDistanceStdDev
        _denseTsdfHighUncertaintyRatio.value =
            metrics.highUncertaintyVoxelRatio
        _denseTsdfConfidenceFactor.value =
            metrics.confidenceFactor

        val surfaceMetrics =
            denseFusion.surfaceDetailMetrics()
        _denseSurfaceMeanDynamicProbability.value =
            surfaceMetrics.meanDynamicProbability
                .coerceIn(0.0, 1.0)
        _denseSurfaceMeanTemporalStability.value =
            surfaceMetrics.meanTemporalStability
                .coerceIn(0.0, 1.0)
        _denseSurfaceTrustedGaussianRatio.value =
            surfaceMetrics.trustedGaussianRatio
                .coerceIn(0.0, 1.0)
    }

    private fun adaptiveTsdfEvidence():
        SparseTsdfFusionEngine.AdaptiveEvidence {
        val connected = _devices.value.count { it.connected }
        val expected = _devices.value.size.coerceAtLeast(1)
        val cameraCoverage =
            (connected.toDouble() / expected.toDouble())
                .coerceIn(0.0, 1.0)
        val motionEnergy = _motionScore.value.coerceIn(0.0, 1.0)
        val uncertaintyConfidence =
            _reconstructionUncertaintyConfidence.value
                .coerceIn(0.0, 1.0)
        val reconstructionConfidence =
            maxOf(
                0.35,
                _reconstructionQualityScore.value.coerceIn(0.0, 1.0),
            )
        val gpuHeadroom =
            (
                1.0 -
                    (_gpuDispatchMillis.value / 16.67)
            ).coerceIn(0.20, 1.0)
        val requestedDetail =
            _motionCustomization.value.meshAdaptiveDetail
                .coerceIn(0.0, 1.0)

        return SparseTsdfFusionEngine.AdaptiveEvidence(
            motionEnergy = motionEnergy,
            uncertainty = 1.0 - uncertaintyConfidence,
            cameraCoverage = cameraCoverage,
            cameraInformationGain =
                _cameraContributionInformationGainScore.value
                    .coerceIn(0.0, 1.0),
            reconstructionConfidence = reconstructionConfidence,
            temporalDepthConsistency =
                latestTemporalDepthConsistency.coerceIn(0.0, 1.0),
            thermalHeadroom = runtimeThermalHeadroom(),
            gpuHeadroom = gpuHeadroom,
            requestedDetail = requestedDetail,
        )
    }

    private fun refreshDenseMesh(
        timestampNanos: Long,
    ) {
        val minimumMeshIntervalNanos =
            (
                1_000_000_000L /
                    _productionProfile.value.denseMeshHz.coerceAtLeast(1)
            )
        if (
            lastDenseMeshExtractNanos != Long.MIN_VALUE &&
            timestampNanos - lastDenseMeshExtractNanos <
                minimumMeshIntervalNanos
        ) {
            return
        }

        val extractionDecision =
            adaptiveDenseMeshExtractionController.evaluate(
                AdaptiveDenseMeshExtractionController.Evidence(
                    timestampNanos = timestampNanos,
                    motionEnergy = _motionScore.value
                        .coerceIn(0.0, 1.0),
                    uncertainty = (
                        1.0 -
                            _reconstructionUncertaintyConfidence.value
                    ).coerceIn(0.0, 1.0),
                    cameraCoverage = _rigCoverageScore.value
                        .coerceIn(0.0, 1.0),
                    cameraInformationGain =
                        _cameraContributionInformationGainScore.value
                            .coerceIn(0.0, 1.0),
                    reconstructionConfidence =
                        _reconstructionQualityScore.value
                            .coerceIn(0.0, 1.0),
                    temporalStability = _temporalStability.value
                        .coerceIn(0.0, 1.0),
                    thermalHeadroom = runtimeThermalHeadroom(),
                    gpuHeadroom = (
                        1.0 -
                            _gpuDispatchMillis.value / 16.67
                        ).coerceIn(0.20, 1.0),
                    requestedDetail =
                        _motionCustomization.value.meshAdaptiveDetail
                            .coerceIn(0.0, 1.0),
                    voxelCount = denseFusion.voxelCount(),
                    maxVoxels = 1_000_000,
                    geometryConditioning =
                        _geometricInformationConditionScore.value
                            .coerceIn(0.0, 1.0),
                    viewDiversity =
                        (
                            _denseViewDiversity.value
                                .coerceIn(0.0, 1.0) * 0.70 +
                                _denseSurfaceAdaptiveViewDiversityScore.value
                                    .coerceIn(0.0, 1.0) * 0.30
                        ).coerceIn(0.0, 1.0),
                    metricUncertaintyP95Meters =
                        _denseSurfaceUncertaintyP95Meters.value
                            .takeIf { it.isFinite() }
                            ?.coerceAtLeast(0.0)
                            ?: _reconstructionUncertaintyP95Meters.value
                                .takeIf { it.isFinite() }
                                ?.coerceAtLeast(0.0)
                            ?: Double.POSITIVE_INFINITY,
                    contactConfidence =
                        _denseContactConfidence.value
                            .coerceIn(0.0, 1.0),
                    identitySafety =
                        _denseIdentityGateQuality.value
                            .coerceIn(0.0, 1.0),
                    calibrationConfidence =
                        (
                            _rigScore.value.coerceIn(0.0, 1.0) * 0.70 +
                                _rigHealthScore.value.coerceIn(0.0, 1.0) * 0.30
                        ).coerceIn(0.0, 1.0),
                ),
            )

        _denseMeshExtractionDetail.value =
            extractionDecision.detailLevel
        _denseMeshExtractionPressure.value =
            extractionDecision.computePressure
        _denseMeshExtractionReason.value =
            extractionDecision.reason

        if (!extractionDecision.shouldExtract) {
            return
        }

        lastDenseMeshExtractNanos =
            timestampNanos

        val extractionStartedNanos = System.nanoTime()
        val extractedMesh =
            denseFusion.extractMesh(
                timestampNanos = timestampNanos,
                isoBand = extractionDecision.isoBand,
                cellBudget = extractionDecision.cellBudget,
                maxVertices = minOf(
                    extractionDecision.vertexBudget,
                    _productionMeshVertexBudget.value
                        .takeIf { it > 0 }
                        ?: extractionDecision.vertexBudget,
                ),
            )
        adaptiveDenseMeshExtractionController.recordExtractionLatency(
            (System.nanoTime() - extractionStartedNanos) / 1_000_000.0,
        )

        extractedMesh?.let { pair ->
            val postProcessed =
                denseMeshQualityEngine.process(
                    mesh = pair.first,
                    incomingQuality = pair.second,
                )

            val temporal =
                denseMeshTemporalConsistencyEngine.evaluate(
                    DenseMeshTemporalConsistencyEngine.Input(
                        timestampNanos = timestampNanos,
                        mesh = postProcessed.mesh,
                        upstreamConfidence =
                            postProcessed.quality.geometricConfidence,
                    ),
                )

            _denseMeshVertices.value =
                postProcessed.mesh.vertices.size
            _denseMeshTriangles.value =
                postProcessed.mesh.triangles.size
            _denseMeshNormalCoverage.value =
                postProcessed.quality.normalCoverage
            _denseMeshTemporalConsistencyConfidence.value =
                temporal.confidence
            _denseMeshTemporalConsistencyWarnings.value =
                temporal.warnings
            _denseMeshSurfaceObserved.value = true
            _denseMeshTopologyIntegrity.value =
                postProcessed.topologyIntegrityScore
                    .coerceIn(0.0, 1.0)
            _denseMeshBoundaryEdges.value =
                postProcessed.boundaryEdgeCount
            _denseMeshNonManifoldEdges.value =
                postProcessed.nonManifoldEdgeCount
            _denseMeshDuplicateTrianglesRemoved.value =
                postProcessed.duplicateTrianglesRemoved
            _denseMeshSurfaceConfidence.value =
                postProcessed.surfaceConfidence?.globalConfidence
                    ?.coerceIn(0.0, 1.0)
                    ?: postProcessed.quality.geometricConfidence
            val surfaceWeight =
                _motionCustomization.value.meshAdaptiveDetail.coerceIn(
                    0.0,
                    1.0,
                ) * 0.20 + 0.10
            _denseMeshConfidence.value =
                (
                    postProcessed.quality.geometricConfidence * 0.50 +
                        (
                            postProcessed.surfaceConfidence
                                ?.globalConfidence
                                ?: postProcessed.quality.geometricConfidence
                        ) * surfaceWeight +
                        temporal.confidence *
                            (0.40 - surfaceWeight)
                    ).coerceIn(0.0, 1.0)

            if (_state.value == CaptureState.RECORDING) {
                recorder.appendDenseMesh(
                    mesh = postProcessed.mesh,
                    quality = postProcessed.quality,
                )
            }
        }
    }



    private fun publishLiveMotionPreview(
        result: PerceptionPipeline.Result,
        skeleton: com.nivukx.nox.core.model.SkeletonFrame?,
        cameraWidth: Int,
        cameraHeight: Int,
        timestampNanos: Long,
    ) {
        if (!_motionCustomization.value.liveMotionPreviewEnabled) return

        val actors =
            result.actorTracks.map { actor ->
                MotionFieldActor(
                    actorId = actor.actorId,
                    confidence = actor.confidence,
                    predicted = actor.predicted,
                    missedFrames = actor.missedFrames,
                    microMotionScore = actor.microMotion.preservedSignal,
                    velocityNormalizedPerSecond =
                        actor.microMotion.velocityNormalizedPerSecond,
                    velocityXNormalizedPerSecond =
                        actor.velocityXNormalizedPerSecond,
                    velocityYNormalizedPerSecond =
                        actor.velocityYNormalizedPerSecond,
                    joints = actor.joints.map { joint ->
                        com.nivukx.nox.core.model.MotionFieldJoint(
                            landmarkId = joint.landmarkId,
                            x = joint.x,
                            y = joint.y,
                            confidence = joint.confidence,
                            predicted = actor.predicted,
                            microMotionScore =
                                actor.microMotion.preservedSignal,
                        )
                    },
                )
            }

        val worldJoints =
            skeleton?.joints?.map { joint ->
                MotionFieldWorldJoint(
                    landmarkId = joint.landmarkId,
                    xMeters = joint.position.x,
                    yMeters = joint.position.y,
                    zMeters = joint.position.z,
                    confidence = joint.confidence.coerceIn(0.0, 1.0),
                    predicted = false,
                )
            } ?: emptyList()

        val runtimeDecision =
            result.runtimeDecision
        val gpuDecision =
            motionFieldGpuPolicyEngine.decide(
                MotionFieldGpuPolicyEngine.Input(
                    vulkanAvailable = false,
                    vulkanExternalMemoryInterop = false,
                    vulkanRendererImplemented = false,
                    gles31Available =
                        gpuAcceleration.backend == GpuBackend.OPENGL_ES_COMPUTE,
                    gles32Available = false,
                    userMode =
                        _motionCustomization.value.liveMotionPreviewGpuMode,
                    qualityScale =
                        runtimeDecision.qualityScale.coerceIn(0.20, 1.0),
                    memoryScale =
                        runtimeDecision.inferenceScale.coerceIn(0.20, 1.0),
                    actorCount = actors.size,
                    detail =
                        _motionCustomization.value.liveMotionPreviewDetail,
                ),
            )

        val gpuState =
            MotionFieldGpuState(
                backend =
                    if (gpuDecision.backend ==
                        com.nivukx.nox.core.model.MotionFieldGpuBackend.UNAVAILABLE
                    ) {
                        com.nivukx.nox.core.model.MotionFieldGpuBackend.UNAVAILABLE
                    } else {
                        com.nivukx.nox.core.model.MotionFieldGpuBackend.OPENGL_ES_RENDER
                    },
                mode =
                    _motionCustomization.value.liveMotionPreviewGpuMode,
                renderLatencyMillis = _gpuDispatchMillis.value,
                gpuUtilizationEstimate = 0.0,
                warning = listOfNotNull(
                    gpuDecision.warnings.firstOrNull(),
                    if (gpuDecision.backend ==
                        com.nivukx.nox.core.model.MotionFieldGpuBackend.UNAVAILABLE
                    ) {
                        "GPU preview backend unavailable; renderer may fall back."
                    } else {
                        null
                    },
                ).firstOrNull(),
            )

        val preview =
            MotionFieldPreviewFrame(
                timestampNanos = timestampNanos,
                cameraWidth = cameraWidth.coerceAtLeast(1),
                cameraHeight = cameraHeight.coerceAtLeast(1),
                displayTimestampNanos = timestampNanos,
                actors = actors,
                reconstructionJoints3d = worldJoints,
                reconstructionConfidence =
                    result.motion.temporalStability.coerceIn(0.0, 1.0),
                reconstructionUncertaintyMeters =
                    skeleton?.joints
                        ?.map { it.positionUncertaintyMeters }
                        ?.average()
                        ?: 0.0,
                cameraMotionRisk =
                    result.motionFusion.semanticMotion.cameraMotionRisk
                        .coerceIn(0.0, 1.0),
                gpu = gpuState,
            )
        motionFieldSynchronizer.submit(preview)
        val selection =
            motionFieldSynchronizer.select(timestampNanos)
        _liveMotionPreviewGpuState.value = gpuState
        _liveMotionPreviewFrame.value =
            selection.frame
                ?.let(motionFieldDisplayPredictor::predict)
                ?: preview
    }

    /**
     * Resolve timing trust from measured camera quality or the session's measured timing state.
     * Missing or non-finite telemetry is unknown evidence and must never imply perfect sync.
     */
    private fun resolvedCaptureTimingConfidence(): Double {
        val cameraConfidence =
            camera.latestTimingQuality?.confidence
                ?.takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
        if (cameraConfidence != null) return cameraConfidence

        return _captureTimingConfidence.value
            .takeIf { it.isFinite() }
            ?.coerceIn(0.0, 1.0)
            ?: 0.0
    }

    private fun motionContextForPerception():
        com.nivukx.nox.core.ai.MotionCaptureEngine.MotionContext {
        val telemetry = camera.latestTelemetry
        // Keep motion and dense geometry on the same bounded timing-confidence policy.
        // A non-finite camera value is ignored rather than allowed to contaminate motion state.
        // When telemetry and measured session quality are both absent, confidence stays unknown.
        val timingConfidence =
            resolvedCaptureTimingConfidence()
        val rollingShutterRisk =
            com.nivukx.nox.core.ai.MotionCaptureEngine
                .rollingShutterRiskFromTelemetry(
                    rollingShutterSkewNanos =
                        telemetry?.rollingShutterSkewNanos ?: 0L,
                    frameDurationNanos =
                        telemetry?.frameDurationNanos ?: 0L,
                    captureTimingConfidence = timingConfidence,
                )

        return com.nivukx.nox.core.ai.MotionCaptureEngine.MotionContext(
            cameraMotionRisk =
                _cameraMotionRisk.value.coerceIn(0.0, 1.0),
            sensorConfidence =
                (
                    timingConfidence * 0.62 +
                        _multiCameraSyncConfidence.value * 0.38
                    ).coerceIn(0.0, 1.0),
            rollingShutterRisk = rollingShutterRisk,
            globalMotionConfidence =
                (
                    _cameraMotionRisk.value *
                        timingConfidence
                    ).coerceIn(0.0, 1.0),
        )
    }

    private fun configureTemporalMotionEstimator() {
        val customization = _motionCustomization.value
        val responsiveness = _motionResponsiveness.value.coerceIn(0.20, 1.20)
        val sensitivity = _motionSensitivity.value.coerceIn(0.50, 1.80)
        val prediction = _predictionHorizon.value.coerceIn(0.04, 0.35)
        val occlusionFrames =
            customization.maximumOcclusionPredictionFrames.coerceIn(0, 60)
        val gateElasticity = _innovationGateElasticity.value.coerceIn(0.55, 1.80)
        val temporalSmoothing = customization.temporalSmoothing.coerceIn(0.0, 1.0)
        val velocityGain = (0.42 + 0.42 * responsiveness).coerceIn(0.20, 0.94)
        val accelerationGain = (0.18 + 0.50 * responsiveness).coerceIn(0.10, 0.78)
        val jerkGain = (0.10 + 0.34 * responsiveness).coerceIn(0.06, 0.56)

        motionStateEstimator.updateConfig(
            HighEndMotionStateEstimator.Config(
                minimumConfidence =
                    (0.04 + (1.0 - sensitivity / 1.80) * 0.12)
                        .coerceIn(0.02, 0.18),
                velocityGain = velocityGain,
                accelerationGain = accelerationGain,
                jerkGain = jerkGain,
                predictionHorizonSeconds = prediction,
                maximumOcclusionFrames = occlusionFrames,
                softInnovationGateMeters =
                    (0.060 * gateElasticity * (0.88 + sensitivity * 0.10))
                        .coerceIn(0.040, 0.120),
                hardInnovationGateMeters =
                    (0.180 * gateElasticity * (0.92 + sensitivity * 0.08))
                        .coerceIn(0.100, 0.320),
                cameraMotionSuppression =
                    customization.cameraMotionSuppression.coerceIn(0.0, 1.0),
                occlusionSuppression =
                    (0.42 + 0.40 * _occlusionPersistence.value)
                        .coerceIn(0.20, 1.0),
                microMotionPreservation =
                    customization.microMotionPreservation.coerceIn(0.0, 1.0),
                temporalContinuityWeight = temporalSmoothing,
                highConfidenceResidualTrust =
                    (0.72 + 0.24 * responsiveness).coerceIn(0.70, 0.98),
                lowConfidencePredictionTrust =
                    (0.70 + 0.18 * _occlusionPersistence.value)
                        .coerceIn(0.65, 0.94),
                uncertaintyGrowthPerMissingFrame =
                    (0.10 + 0.18 * (1.0 - _occlusionPersistence.value))
                        .coerceIn(0.05, 0.30),
                temporalMemoryAlpha =
                    (0.12 + temporalSmoothing * 0.34)
                        .coerceIn(0.08, 0.50),
                temporalMemoryMinimumTrust =
                    (0.26 +
                        customization.outputIntegrityStrictness * 0.18)
                        .coerceIn(0.20, 0.50),
            ),
        )
    }

    private fun recordSemanticMotionEvent(
        skeleton: com.nivukx.nox.core.model.SkeletonFrame,
        confidence: Double,
    ) {
        val root = skeleton.joints
            .firstOrNull { it.landmarkId == 23 }
            ?: skeleton.joints.firstOrNull()
            ?: return

        val previous = semanticEventRootPosition
        val previousVelocity = semanticEventRootVelocity
        val dtSeconds =
            if (
                semanticEventTimestampNanos == Long.MIN_VALUE ||
                    skeleton.timestampNanos <= semanticEventTimestampNanos
            ) {
                1.0 / 60.0
            } else {
                (
                    skeleton.timestampNanos -
                        semanticEventTimestampNanos
                    ).toDouble() / 1e9
            }.coerceIn(1.0 / 240.0, 0.50)

        val velocity =
            if (previous == null) {
                root.velocity
            } else {
                (root.position - previous) /
                    dtSeconds
            }

        val acceleration =
            (velocity - previousVelocity) /
                dtSeconds

        val footLeft = skeleton.joints
            .firstOrNull { it.landmarkId == 31 }
            ?: skeleton.joints.firstOrNull { it.landmarkId == 29 }
        val footRight = skeleton.joints
            .firstOrNull { it.landmarkId == 32 }
            ?: skeleton.joints.firstOrNull { it.landmarkId == 30 }

        val leftStable =
            footLeft != null &&
                footLeft.velocity.norm() < 0.09
        val rightStable =
            footRight != null &&
                footRight.velocity.norm() < 0.09

        val event = motionEventEngine.classify(
            MotionEventSegmentationEngine.Input(
                frame = skeleton,
                rootVelocityMetersPerSecond = velocity.norm(),
                rootAccelerationMetersPerSecond2 = acceleration.norm(),
                leftFootStable = leftStable,
                rightFootStable = rightStable,
                footSkateRisk = run {
                    val left =
                        footLeft?.velocity?.norm() ?: 0.0
                    val right =
                        footRight?.velocity?.norm() ?: 0.0
                    val stableSpeed =
                        0.09
                    maxOf(
                        if (leftStable) {
                            (left / stableSpeed).coerceIn(0.0, 1.0)
                        } else {
                            0.0
                        },
                        if (rightStable) {
                            (right / stableSpeed).coerceIn(0.0, 1.0)
                        } else {
                            0.0
                        },
                    )
                },
                cameraMotionRisk =
                    _cameraMotionRisk.value
                        .coerceIn(0.0, 1.0),
                confidence = confidence,
            ),
        )

        _motionEventType.value = event.type.name
        _motionEventConfidence.value = event.confidence
        recorder.appendMotionEvent(event)

        semanticEventRootPosition = root.position
        semanticEventRootVelocity = velocity
        semanticEventTimestampNanos = skeleton.timestampNanos
    }

    private fun configureProductionMotionDynamics() {
        val tuning = _motionCustomization.value
        val base = productionMotionDynamicsEngine.config()

        productionMotionDynamicsEngine.updateConfig(
            base.copy(
                idleSpeedMetersPerSecond =
                    (
                        base.idleSpeedMetersPerSecond /
                            _motionSensitivity.value
                        ).coerceIn(0.015, 0.18),
                gestureSpeedMetersPerSecond =
                    (
                        base.gestureSpeedMetersPerSecond /
                            _motionSensitivity.value
                        ).coerceIn(0.20, 1.60),
                locomotionSpeedMetersPerSecond =
                    (
                        base.locomotionSpeedMetersPerSecond /
                            _motionSensitivity.value
                        ).coerceIn(0.60, 3.50),
                runningSpeedMetersPerSecond =
                    (
                        base.runningSpeedMetersPerSecond /
                            _motionSensitivity.value
                        ).coerceIn(1.10, 6.00),
                airborneAccelerationMetersPerSecond2 =
                    (
                        base.airborneAccelerationMetersPerSecond2 /
                            _adaptiveRegimeStrength.value.coerceAtLeast(0.35)
                        ).coerceIn(2.0, 14.0),
                highDynamicsJerkMetersPerSecond3 =
                    (
                        base.highDynamicsJerkMetersPerSecond3 /
                            _motionSensitivity.value
                        ).coerceIn(30.0, 180.0),
                instabilityJerkMetersPerSecond3 =
                    (
                        base.instabilityJerkMetersPerSecond3 /
                            _motionSensitivity.value
                        ).coerceIn(80.0, 420.0),
                contactSpeedMetersPerSecond =
                    (
                        base.contactSpeedMetersPerSecond /
                            _velocityResponsiveness.value
                        ).coerceIn(0.02, 0.18),
                symmetryToleranceMeters =
                    (
                        base.symmetryToleranceMeters *
                            (
                                1.08 -
                                    _rejectAggressiveness.value * 0.28
                                )
                        ).coerceIn(0.06, 0.20),
                microMotionThresholdMeters =
                    (
                        base.microMotionThresholdMeters /
                            _jitterSuppression.value.coerceIn(0.25, 1.0)
                        ).coerceIn(0.004, 0.035),
                temporalBandAlpha =
                    (
                        0.08 +
                            tuning.motionFrequencyBandSensitivity
                                .coerceIn(0.0, 1.0) * 0.42
                        ).coerceIn(0.08, 0.50),
                temporalBandTimeConstantSeconds =
                    (
                        0.13 -
                            tuning.motionFrequencyBandSensitivity
                                .coerceIn(0.0, 1.0) * 0.08
                        ).coerceIn(0.05, 0.13),
                smoothingAlpha =
                    (
                        0.12 +
                            _motionResponsiveness.value * 0.18 +
                            _adaptiveRegimeStrength.value * 0.18
                        ).coerceIn(0.08, 0.62),
            ),
        )
    }

    private fun updateCrossModalIdentity(
        skeleton: com.nivukx.nox.core.model.SkeletonFrame,
        timestampNanos: Long,
    ) {
        val body = bodyRootEvidence(skeleton)
        if (body == null) {
            return
        }

        fun ageSeconds(lastTimestampNanos: Long): Double =
            if (lastTimestampNanos == Long.MIN_VALUE ||
                timestampNanos <= lastTimestampNanos
            ) {
                Double.POSITIVE_INFINITY
            } else {
                (
                    timestampNanos - lastTimestampNanos
                ).coerceAtLeast(0L) / 1e9
            }

        val faceAge = ageSeconds(latestFaceCenterTimestampNanos)
        val leftHandAge = ageSeconds(latestLeftHandTimestampNanos)
        val rightHandAge = ageSeconds(latestRightHandTimestampNanos)
        val leftEyeAge = ageSeconds(latestLeftEyeTimestampNanos)
        val rightEyeAge = ageSeconds(latestRightEyeTimestampNanos)
        val leftEye =
            latestLeftEye3D?.takeIf { leftEyeAge <= 0.30 }
        val rightEye =
            latestRightEye3D?.takeIf { rightEyeAge <= 0.30 }
        val freshness = minOf(
            faceAge,
            leftHandAge,
            rightHandAge,
            leftEyeAge,
            rightEyeAge,
        ).takeIf { it.isFinite() }
            ?: 0.0

        val face = latestFaceCenter3D
            ?.takeIf { faceAge <= 0.80 }
        val leftHand = latestLeftHand3D
            ?.takeIf { leftHandAge <= 0.45 }
        val rightHand = latestRightHand3D
            ?.takeIf { rightHandAge <= 0.45 }

        val bodyConfidence =
            body.second.coerceIn(0.0, 1.0)
        val faceConfidence =
            _face3DTemporalConsistencyConfidence.value
                .coerceIn(0.0, 1.0)
        val handConfidence =
            _hand3DConfidence.value
                .coerceIn(0.0, 1.0)

        val evidenceConfidence =
            listOfNotNull(
                bodyConfidence,
                face?.let { faceConfidence },
                leftHand?.let { handConfidence },
                rightHand?.let { handConfidence },
                leftEye?.let {
                    _eye3DConfidence.value
                        .coerceIn(0.0, 1.0)
                },
                rightEye?.let {
                    _eye3DConfidence.value
                        .coerceIn(0.0, 1.0)
                },
            ).average()

        val observation =
            CrossModalIdentityFusionEngine.Observation(
                observationId =
                    crossModalObservationSequence.incrementAndGet(),
                timestampNanos = timestampNanos,
                bodyRoot = body.first,
                faceCenter = face,
                leftHand = leftHand,
                rightHand = rightHand,
                leftEye = leftEye,
                rightEye = rightEye,
                confidence = evidenceConfidence.coerceIn(0.0, 1.0),
                freshnessSeconds = freshness.coerceIn(0.0, 2.0),
                evidenceProfile = buildMap {
                    put(
                        com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.BODY,
                        com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                            modality =
                                com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = bodyConfidence,
                            uncertaintyMeters =
                                _motionUncertaintyRadiusMeters.value
                                    .coerceAtLeast(0.01),
                        ),
                    )
                    face?.let {
                        put(
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.FACE,
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                                modality =
                                    com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.FACE,
                                confidence = faceConfidence,
                                uncertaintyMeters = 0.03,
                                freshnessSeconds =
                                    faceAge.coerceAtLeast(0.0),
                            ),
                        )
                    }
                    leftHand?.let {
                        put(
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND,
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                                modality =
                                    com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND,
                                confidence =
                                    minOf(
                                        handConfidence,
                                        _handLongitudinalAnatomyConsistency.value
                                            .coerceIn(0.0, 1.0),
                                        _handUncertaintyConfidence.value
                                            .coerceIn(0.0, 1.0),
                                    ),
                                uncertaintyMeters = (
                                    0.04 +
                                        (
                                            1.0 -
                                                _handLongitudinalAnatomyConsistency.value
                                                    .coerceIn(0.0, 1.0)
                                            ) * 0.06
                                    ).coerceAtMost(0.10),
                                freshnessSeconds =
                                    leftHandAge.coerceAtLeast(0.0),
                            ),
                        )
                    }
                    rightHand?.let {
                        put(
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.RIGHT_HAND,
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                                modality =
                                    com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.RIGHT_HAND,
                                confidence = handConfidence,
                                uncertaintyMeters = 0.04,
                                freshnessSeconds =
                                    rightHandAge.coerceAtLeast(0.0),
                            ),
                        )
                    }
                    leftEye?.let {
                        put(
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT,
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                                modality =
                                    com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT,
                                confidence =
                                    _eye3DConfidence.value
                                        .coerceIn(0.0, 1.0),
                                uncertaintyMeters =
                                    leftEyeUncertaintyMeters(),
                                freshnessSeconds = 0.0,
                                redundancy =
                                    _eye3DConfidence.value
                                        .coerceIn(0.0, 1.0),
                            ),
                        )
                    }
                    rightEye?.let {
                        put(
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.EYE_RIGHT,
                            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                                modality =
                                    com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Modality.EYE_RIGHT,
                                confidence =
                                    _eye3DConfidence.value
                                        .coerceIn(0.0, 1.0),
                                uncertaintyMeters =
                                    rightEyeUncertaintyMeters(),
                                freshnessSeconds = 0.0,
                                redundancy =
                                    _eye3DConfidence.value
                                        .coerceIn(0.0, 1.0),
                            ),
                        )
                    }
                },
                predicted = false,
                occluded =
                    face == null ||
                        (leftHand == null && rightHand == null),
            )

        configureCrossModalIdentity()

        val result =
            crossModalIdentityFusionEngine.update(
                timestampNanos = timestampNanos,
                observations = listOf(observation),
            )

        _identityContinuityScore.value =
            (
                _identityContinuityScore.value * 0.42 +
                    result.continuityScore * 0.58
                ).coerceIn(0.0, 1.0)

        _identitySwapRisk.value =
            maxOf(
                _identitySwapRisk.value * 0.72,
                result.swapRisk,
            ).coerceIn(0.0, 1.0)

        _identityAction.value =
            result.action.name
        _identityReason.value =
            result.reason
        _identityEvidenceFusionScore.value =
            result.evidenceFusionScore.coerceIn(0.0, 1.0)
        _identitySelectedModalityCount.value =
            result.selectedEvidenceModalityCount
        _identityEvidenceWarnings.value =
            result.evidenceWarnings

        if (_state.value == CaptureState.RECORDING) {
            recorder.appendCrossModalIdentity(result)
        }

        if (result.warnings.isNotEmpty()) {
            _reconstructionGateReasons.value =
                (
                    _reconstructionGateReasons.value +
                        result.warnings.map { "identity-$it" }
                    )
                    .distinct()
                    .takeLast(12)
        }
    }

    private fun configureMotionUncertainty() {
        val customization = _motionCustomization.value
        val current = calibratedMotionUncertaintyEngine.configuration()

        calibratedMotionUncertaintyEngine.configure(
            current.copy(
                targetCoverage = (
                    0.90 +
                        customization.outputIntegrityStrictness
                            .coerceIn(0.0, 1.0) * 0.09
                    ).coerceIn(0.90, 0.99),
                minimumCalibrationSamples = (
                    6 +
                        customization.worldConsistencyDetail
                            .coerceIn(0.0, 1.0) * 10.0
                    ).toInt().coerceIn(6, 16),
                maximumResidualSamples =
                    (
                        48 +
                            customization.temporalAlignmentStrictness
                                .coerceIn(0.0, 1.0) * 96.0
                        ).toInt().coerceIn(48, 144),
                maximumRadiusMeters =
                    (
                        0.18 +
                            (
                                1.0 -
                                    customization.outputIntegrityStrictness
                                        .coerceIn(0.0, 1.0)
                                ) * 0.30
                        ).coerceIn(0.18, 0.48),
                dynamicStressInflation =
                    (
                        0.10 +
                            _motionThresholdScale.value
                                .coerceIn(0.35, 1.65) * 0.08
                        ).coerceIn(0.10, 0.24),
                occlusionInflation =
                    (
                        0.16 +
                            customization.maximumOcclusionPredictionFrames
                                .coerceIn(0, 60) * 0.012
                        ).coerceIn(0.16, 0.36),
                synchronizationInflation =
                    (
                        0.12 +
                            customization.subframeAlignmentStrictness
                                .coerceIn(0.0, 1.0) * 0.14
                        ).coerceIn(0.12, 0.26),
                highRiskThreshold =
                    (
                        0.66 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.18
                        ).coerceIn(0.66, 0.88),
                holdRiskThreshold =
                    (
                        0.50 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.16
                        ).coerceIn(0.50, 0.68),
            ),
        )
    }

    private fun applyCalibratedMotionUncertainty(
        skeleton: com.nivukx.nox.core.model.SkeletonFrame,
    ) {
        configureMotionUncertainty()

        val result = calibratedMotionUncertaintyEngine.evaluate(
            skeleton = skeleton,
            motionScore = _motionScore.value.toDouble(),
            dynamicStress = _motionDynamicStress.value,
            occlusionRisk = _occlusionRisk.value,
            cameraMotionRisk = _cameraMotionRisk.value,
            synchronizationConfidence =
                _multiCameraSyncConfidence.value.coerceIn(0.0, 1.0),
        )

        _motionUncertaintyCoverage.value =
            result.empiricalCoverage
        _motionUncertaintyConfidence.value =
            result.uncertaintyConfidence
        _motionUncertaintyOodRisk.value =
            result.outOfDistributionRisk
        _motionUncertaintyRadiusMeters.value =
            result.meanRadiusMeters
        _motionUncertaintyCalibrationProgress.value =
            result.calibrationProgress
        _motionUncertaintyPolicy.value =
            result.policy.name
        _motionUncertaintyWarnings.value =
            result.warnings

        if (result.warnings.isNotEmpty()) {
            _reconstructionGateReasons.value =
                (
                    _reconstructionGateReasons.value +
                        result.warnings.map {
                            "uncertainty-$it"
                        }
                    )
                    .distinct()
                    .takeLast(14)
        }

        _reconstructionQualityScore.value =
            (
                _reconstructionQualityScore.value * 0.84 +
                    result.uncertaintyConfidence * 0.16
                ).coerceIn(0.0, 1.0)

        if (_state.value == CaptureState.RECORDING) {
            recorder.appendMotionUncertainty(result)
        }
    }

    private fun configureCrossModalIdentity() {
        val customization = _motionCustomization.value
        val body = 0.26 + customization.bodyDetail.coerceIn(0.0, 1.0) * 0.18
        val face = 0.18 + customization.faceDetail.coerceIn(0.0, 1.0) * 0.16
        val hand = 0.12 + customization.handDetail.coerceIn(0.0, 1.0) * 0.10
        val eye = 0.06 + customization.eyeDetail.coerceIn(0.0, 1.0) * 0.08
        val motion = 0.10
        val sum = body + face + hand + eye + motion
        val scale = 0.96 / sum.coerceAtLeast(1e-6)

        val targetFps =
            customization.targetFps.coerceIn(15, 120)
        val maximumAgeSeconds =
            (
                customization.maximumOcclusionPredictionFrames
                    .coerceIn(0, 60)
                    .toDouble() /
                    targetFps.toDouble()
            ).coerceIn(0.16, 1.20)

        val current =
            crossModalIdentityFusionEngine.configuration()

        crossModalIdentityFusionEngine.configureEvidenceArbitration(
            com.nivukx.nox.core.reconstruction.CrossModalEvidenceArbitrationEngine.Config(
                minimumSelectedModalities = 1,
                maximumSelectedModalities =
                    (
                        2 +
                            customization.maxSubjects.coerceIn(1, 4)
                    ).coerceAtMost(4),
                minimumConfidence =
                    (
                        0.10 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.10
                        ).coerceIn(0.10, 0.22),
                staleEvidenceSeconds =
                    (
                        0.10 +
                            customization.temporalAlignmentStrictness
                                .coerceIn(0.0, 1.0) * 0.10
                        ).coerceIn(0.10, 0.24),
                oodQuarantineThreshold =
                    (
                        0.74 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.14
                        ).coerceIn(0.74, 0.88),
                quarantineAfterBadFrames =
                    (
                        3 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 2.0
                        ).toInt().coerceIn(3, 5),
                recoveryFrames =
                    (
                        4 +
                            customization.temporalSmoothing
                                .coerceIn(0.0, 1.0) * 3.0
                        ).toInt().coerceIn(4, 7),
                historyAlpha =
                    (
                        0.14 +
                            customization.temporalSmoothing
                                .coerceIn(0.0, 1.0) * 0.14
                        ).coerceIn(0.14, 0.28),
                redundancyPenaltyWeight =
                    (
                        0.42 +
                            customization.multiViewEvidenceAdaptivity
                                .coerceIn(0.0, 1.0) * 0.34
                        ).coerceIn(0.42, 0.76),
                marginalSelectionThreshold =
                    (
                        0.07 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.08
                        ).coerceIn(0.07, 0.15),
            ),
        )

        crossModalIdentityFusionEngine.configure(
            current.copy(
                maximumTracks =
                    customization.maxSubjects.coerceIn(1, 64),
                maximumObservations =
                    customization.maxSubjects.coerceIn(1, 64),
                bodyWeight = body * scale,
                faceWeight = face * scale,
                handWeight = hand * scale,
                eyeWeight = eye * scale,
                motionWeight = motion * scale,
                maximumAgeSeconds = maximumAgeSeconds,
                missingModalityPenalty =
                    (
                        0.08 +
                            customization
                                .reconstructionEvidenceDisagreementToleranceMeters
                                .coerceIn(0.02, 0.30)
                        ).coerceIn(0.08, 0.15),
                predictionPenalty =
                    (
                        0.04 +
                            (
                                1.0 -
                                    customization.microMotionPreservation
                                        .coerceIn(0.0, 1.0)
                                ) * 0.12
                        ).coerceIn(0.04, 0.16),
                occlusionPenalty =
                    (
                        0.08 +
                            customization
                                .cameraMotionSuppression
                                .coerceIn(0.0, 1.0) * 0.10
                        ).coerceIn(0.08, 0.18),
                switchCost =
                    (
                        0.12 +
                            customization
                                .outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.24
                        ).coerceIn(0.12, 0.36),
                assignmentGate =
                    (
                        0.62 +
                            customization.faceEyeFusionWeight
                                .coerceIn(0.0, 1.0) * 0.18 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.08
                        ).coerceIn(0.58, 0.88),
                switchGate =
                    (
                        0.46 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.18
                        ).coerceIn(0.44, 0.72),
                recoveryGate =
                    (
                        0.56 +
                            customization.microMotionPreservation
                                .coerceIn(0.0, 1.0) * 0.12
                        ).coerceIn(0.52, 0.74),
                maximumVelocityMetersPerSecond =
                    (
                        4.0 +
                            customization.bodyDetail
                                .coerceIn(0.0, 1.0) * 6.0
                        ).coerceIn(4.0, 10.0),
                hysteresisMargin =
                    (
                        0.04 +
                            customization.outputIntegrityStrictness
                                .coerceIn(0.0, 1.0) * 0.08
                        ).coerceIn(0.04, 0.12),
            ),
        )
    }

    private fun bodyRootEvidence(
        skeleton: com.nivukx.nox.core.model.SkeletonFrame,
    ): Pair<Vec3, Double>? {
        val primary =
            skeleton.joints.filter {
                it.landmarkId == 23 || it.landmarkId == 24
            }

        val fallback =
            if (primary.isEmpty()) {
                skeleton.joints.filter {
                    it.landmarkId == 11 || it.landmarkId == 12
                }
            } else {
                primary
            }

        val valid =
            fallback.filter {
                it.confidence >= 0.10 &&
                    it.position.x.isFinite() &&
                    it.position.y.isFinite() &&
                    it.position.z.isFinite()
            }

        if (valid.isEmpty()) return null

        val center =
            Vec3(
                valid.map { it.position.x }.average(),
                valid.map { it.position.y }.average(),
                valid.map { it.position.z }.average(),
            )

        val confidence =
            valid.map { it.confidence.coerceIn(0.0, 1.0) }.average()

        return center to confidence
    }

    private fun averageFinitePoint(
        points: Collection<Vec3>,
    ): Vec3? {
        val valid =
            points.filter {
                it.x.isFinite() &&
                    it.y.isFinite() &&
                    it.z.isFinite()
            }

        if (valid.isEmpty()) return null

        return Vec3(
            valid.map { it.x }.average(),
            valid.map { it.y }.average(),
            valid.map { it.z }.average(),
        )
    }

    private fun evaluateCaptureIntegrity(
        timestampNanos: Long,
        confidence: Double,
        reprojectionErrorPx: Double,
        meanUncertaintyMeters: Double,
        thermalScale: Double = 1.0,
        memoryScale: Double = 1.0,
        runtimeQualityScale: Double = 1.0,
        multiCamera: Boolean,
    ): CaptureIntegrityEngine.Result {
        val result = captureIntegrityEngine.evaluate(
            CaptureIntegrityEngine.Input(
                timestampNanos = timestampNanos,
                reconstructionConfidence = confidence.coerceIn(0.0, 1.0),
                reprojectionErrorPx = reprojectionErrorPx,
                meanUncertaintyMeters = meanUncertaintyMeters,
                geometricObservability =
                    _geometricObservabilityScore.value.coerceIn(0.0, 1.0),
                rigCoverageScore =
                    _rigCoverageScore.value.coerceIn(0.0, 1.0),
                temporalConfidence =
                    if (multiCamera) {
                        _temporalAlignmentConfidence.value.coerceIn(0.0, 1.0)
                    } else {
                        1.0
                    },
                temporalStability = _temporalStability.value.coerceIn(0.0, 1.0),
                synchronizationConfidence =
                    _multiCameraSyncConfidence.value.coerceIn(0.0, 1.0),
                calibrationConfidence = _rigScore.value.coerceIn(0.0, 1.0),
                subjectConfidence =
                    _subjectLockConfidence.value.coerceIn(0.0, 1.0),
                motionScore = _motionScore.value.coerceIn(0.0, 1.0),
                motionEvidenceConfidence =
                    _motionEvidenceConfidence.value.coerceIn(0.0, 1.0),
                microMotionScore = _microMotionScore.value.coerceIn(0.0, 1.0),
                measurementUncertaintyScore =
                    _motionMeasurementUncertainty.value.coerceIn(0.0, 1.0),
                motionBlurRisk =
                    _cameraOpticalBlurRisk.value.coerceIn(0.0, 1.0),
                occlusionRisk = _occlusionRisk.value.coerceIn(0.0, 1.0),
                cameraMotionRisk = _cameraMotionRisk.value.coerceIn(0.0, 1.0),
                faceEyeConfidence =
                    _faceEyeFusionConfidence.value
                        .coerceIn(0.0, 1.0),
                faceGeometryConfidence =
                    _face3DTemporalConsistencyConfidence.value.coerceIn(0.0, 1.0),
                eyeTemporalConfidence =
                    _eye3DTemporalConsistencyConfidence.value.coerceIn(0.0, 1.0),
                handConfidence = _hand3DConfidence.value.coerceIn(0.0, 1.0),
                thermalScale = thermalScale.coerceIn(0.0, 1.0),
                memoryScale = memoryScale.coerceIn(0.0, 1.0),
                runtimeQualityScale = runtimeQualityScale.coerceIn(0.0, 1.0),
                multiCamera = multiCamera,
                customization = _motionCustomization.value,
                reconstructionWarnings = _reconstructionGateReasons.value,
                temporalWarnings = _temporalAlignmentWarnings.value,
                geometryWarnings = _geometricObservabilityWarnings.value,
                kinematicConstraintScore =
                    _kinematicConstraintScore.value.coerceIn(0.0, 1.0),
                surfaceConfidence =
                    if (_denseMeshSurfaceObserved.value) {
                        _denseMeshSurfaceConfidence.value
                            .coerceIn(0.0, 1.0)
                    } else {
                        1.0
                    },
                surfaceNormalObservability =
                    _denseTsdfSurfaceNormalObservability.value
                        .coerceIn(0.0, 1.0),
                surfaceHighUncertaintyRatio =
                    _denseTsdfHighUncertaintyRatio.value
                        .coerceIn(0.0, 1.0),
                surfaceCameraInformationGain =
                    _cameraContributionInformationGainScore.value
                        .coerceIn(0.0, 1.0),
                surfaceEvidenceStrictness =
                    _motionCustomization.value.surfaceEvidenceStrictness
                        .coerceIn(0.0, 1.0),
                denseLandmarkCoverage =
                    _denseLandmarkCoverage.value
                        .coerceIn(0.0, 1.0),
                denseLandmarkVisibleCoverage =
                    _denseLandmarkVisibleCoverage.value
                        .coerceIn(0.0, 1.0),
                denseLandmarkUncertaintyP95 =
                    _denseLandmarkUncertaintyP95.value
                        .coerceIn(0.0, 1.0),
                denseLandmarkCrossViewAgreement =
                    _denseLandmarkCrossViewAgreement.value
                        .coerceIn(0.0, 1.0),
                denseLandmarkCrossViewEvidenceAvailable =
                    _denseLandmarkCrossViewEvidenceAvailable.value,
                denseLandmarkTemporalStability =
                    _denseLandmarkTemporalStability.value
                        .coerceIn(0.0, 1.0),
                denseLandmarkHardFailure =
                    _denseLandmarkHardFailure.value,
                physicsConstraintScore =
                    _physicsConstraintScore.value.coerceIn(0.0, 1.0),
                physicsConstraintAccepted =
                    _physicsConstraintAccepted.value,
                directionalUncertaintyMajor95Meters =
                    _directionalUncertaintyMajor95Meters.value
                        .coerceAtLeast(0.0),
                directionalUncertaintyDepthMeters =
                    _directionalUncertaintyDepthMeters.value
                        .coerceAtLeast(0.0),
                directionalUncertaintyAnisotropy =
                    _directionalUncertaintyAnisotropy.value
                        .coerceAtLeast(1.0),
                directionalUncertaintyConfidence =
                    _directionalUncertaintyConfidence.value
                        .coerceIn(0.0, 1.0),
                directionalUncertaintyDegradedJointCount =
                    _directionalUncertaintyDegradedJointCount.value,
                faceEyeFusionAccepted =
                    _faceEyeFusionAccepted.value,
                faceEyeFusionMode =
                    _faceEyeFusionMode.value,
                faceEyeIndependentEvidenceSupport =
                    _faceEyeIndependentEvidenceSupport.value
                        .coerceIn(0.0, 1.0),
                faceEyeReliabilityTrend =
                    _faceEyeReliabilityTrend.value
                        .coerceIn(-1.0, 1.0),
                faceEyeHardFailure =
                    _faceEyeHardFailureReasons.value.any {
                        it.contains("hard") ||
                            it.contains("rejected") ||
                            it == "face-stale"
                    },
                faceEyeShapeConfidence =
                    _faceEyeShapeConfidence.value
                        .coerceIn(0.0, 1.0),
                faceEyeEyeDynamicConsistency =
                    _faceEyeEyeDynamicConsistency.value
                        .coerceIn(0.0, 1.0),
                faceEyeCalibrationDriftScore =
                    _faceEyeCalibrationDriftScore.value
                        .coerceIn(0.0, 1.0),
                faceEyeCalibrationDriftPersistent =
                    _faceEyeCalibrationDriftPersistent.value,
                faceEyeOutOfDistributionRisk =
                    _faceEyeOutOfDistributionRisk.value
                        .coerceIn(0.0, 1.0),
                global4DScore =
                    _global4DScore.value
                        .coerceIn(0.0, 1.0),
                global4DAccepted =
                    _global4DAccepted.value,
                global4DPersistentFailure =
                    _global4DPersistentFailure.value,
                global4DRootMotionConsistency =
                    _global4DRootMotionConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DRootVelocityAgreement =
                    _global4DRootVelocityAgreement.value
                        .coerceIn(0.0, 1.0),
                global4DSceneConsistency =
                    _global4DSceneConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DDirectionalUncertaintyConsistency =
                    _global4DDirectionalUncertaintyConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DFaceEyeConsistency =
                    _global4DFaceEyeConsistency.value
                        .coerceIn(0.0, 1.0),
                kinematicWarnings =
                    _kinematicConstraintWarnings.value,
                surfaceWarnings =
                    if (_denseMeshSurfaceObserved.value) {
                        _denseMeshTemporalConsistencyWarnings.value
                    } else {
                        emptyList()
                    },
            ),
        )

        _captureIntegrityScore.value = result.smoothedScore
        _captureIntegrityBand.value = result.band.name
        _captureIntegrityOutputAllowed.value = result.outputAllowed
        _captureIntegrityBottleneck.value = result.bottleneck.name
        _captureIntegrityWarnings.value = result.warnings
        _captureIntegrityHardFailures.value = result.hardFailures

        refreshConcurrentRigAdmission(
            nowNanos = result.timestampNanos,
            memoryScale = memoryScale,
        )

        val adaptiveDetailBudget =
            adaptiveReconstructionDetailEngine.evaluate(
                AdaptiveReconstructionDetailEngine.Input(
                    profile = _productionProfile.value,
                    p95UncertaintyMeters =
                        meanUncertaintyMeters
                            .takeIf { it.isFinite() }
                            ?.coerceAtLeast(0.0)
                            ?: 1.0,
                    geometricObservability =
                        _geometricObservabilityScore.value
                            .coerceIn(0.0, 1.0),
                    thermalHeadroom =
                        thermalScale.coerceIn(0.0, 1.0),
                    memoryHeadroom =
                        memoryScale.coerceIn(0.0, 1.0),
                    gpuLatencyMs =
                        _gpuDispatchMillis.value,
                    dynamicStress =
                        _motionDynamicStress.value
                            .coerceIn(0.0, 1.0),
                    rigConfidence =
                        _adaptiveViewEvidenceGlobalRigTrust.value
                            .coerceIn(0.0, 1.0),
                    viewDiversity =
                        (
                            _denseLandmarkCrossViewAgreement.value * 0.70 +
                                _cameraContributionInformationGainScore.value * 0.30
                            ).coerceIn(0.0, 1.0),
                    captureTimingConfidence =
                        _captureTimingConfidence.value
                            .coerceIn(0.0, 1.0),
                ),
            )

        val productionDecision =
            productionQualityEngine.evaluate(
                HighEndReconstructionQualityEngine.Input(
                    timestampNanos = result.timestampNanos,
                    profile = _productionProfile.value,
                    reconstructionConfidence =
                        result.components.reconstruction.coerceIn(0.0, 1.0),
                    reprojectionErrorPx = reprojectionErrorPx,
                    meanUncertaintyMeters = meanUncertaintyMeters,
                    geometricObservability =
                        _geometricObservabilityScore.value.coerceIn(0.0, 1.0),
                    temporalConfidence =
                        _temporalAlignmentConfidence.value.coerceIn(0.0, 1.0) *
                            _temporalStability.value.coerceIn(0.0, 1.0),
                    synchronizationConfidence =
                        _multiCameraSyncConfidence.value.coerceIn(0.0, 1.0),
                    calibrationConfidence =
                        _rigScore.value.coerceIn(0.0, 1.0),
                    trackingConfidence =
                        _subjectLockConfidence.value.coerceIn(0.0, 1.0),
                    identitySwapRisk =
                        _identitySwapRisk.value.coerceIn(0.0, 1.0),
                    occlusionRisk =
                        _occlusionRisk.value.coerceIn(0.0, 1.0),
                    cameraMotionRisk =
                        _cameraMotionRisk.value.coerceIn(0.0, 1.0),
                    faceEyeConfidence =
                        _faceEyeFusionConfidence.value
                            .coerceIn(0.0, 1.0),
                    handConfidence =
                        _hand3DConfidence.value.coerceIn(0.0, 1.0),
                    handUncertaintyConfidence =
                        _handUncertaintyConfidence.value
                            .coerceIn(0.0, 1.0),
                    handUncertaintyOodRisk =
                        _handUncertaintyOodRisk.value
                            .coerceIn(0.0, 1.0),
                    surfaceConfidence =
                        if (_denseMeshSurfaceObserved.value) {
                            _denseMeshSurfaceConfidence.value.coerceIn(0.0, 1.0)
                        } else {
                            1.0
                        },
                    surfaceNormalObservability =
                        _denseTsdfSurfaceNormalObservability.value
                            .coerceIn(0.0, 1.0),
                    surfaceHighUncertaintyRatio =
                        _denseTsdfHighUncertaintyRatio.value
                            .coerceIn(0.0, 1.0),
                    cameraInformationGain =
                        _cameraContributionInformationGainScore.value
                            .coerceIn(0.0, 1.0),
                    surfaceEvidenceStrictness =
                        _motionCustomization.value.surfaceEvidenceStrictness
                            .coerceIn(0.0, 1.0),
                    denseLandmarkCoverage =
                        _denseLandmarkCoverage.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkVisibleCoverage =
                        _denseLandmarkVisibleCoverage.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkUncertaintyP95 =
                        _denseLandmarkUncertaintyP95.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkCrossViewAgreement =
                        _denseLandmarkCrossViewAgreement.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkCrossViewEvidenceAvailable =
                        _denseLandmarkCrossViewEvidenceAvailable.value,
                    denseLandmarkTemporalStability =
                        _denseLandmarkTemporalStability.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkContactCoverage =
                        _denseLandmarkContactCoverage.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkPredictionBridgeRatio =
                        _denseLandmarkPredictionBridgeRatio.value
                            .coerceIn(0.0, 1.0),
                    denseLandmarkHardFailure =
                        _denseLandmarkUncertaintyP95.value >= 0.95,
                    thermalScale = thermalScale.coerceIn(0.0, 1.0),
                    memoryScale = memoryScale.coerceIn(0.0, 1.0),
                    runtimeQualityScale =
                        runtimeQualityScale.coerceIn(0.0, 1.0),
                    runtimeInferenceScale =
                        _runtimeInferenceScale.value.coerceIn(0.0, 1.0),
                    gpuDispatchMillis =
                        _gpuDispatchMillis.value,
                    connectedCameraCount =
                        _devices.value.count { it.connected },
                    observedSubjectCount =
                        _multiPersonObservedCount.value.coerceAtLeast(
                            if (_subjectLockState.value == "SEARCHING") {
                                0
                            } else {
                                1
                            },
                        ),
                    identityContinuity =
                        _multiPersonIdentityContinuity.value
                            .coerceIn(0.0, 1.0),
                    calibratedUncertaintyConfidence =
                        _motionUncertaintyConfidence.value
                            .coerceIn(0.0, 1.0),
                    motionOutOfDistributionRisk =
                        _motionUncertaintyOodRisk.value
                            .coerceIn(0.0, 1.0),
                    motionUncertaintyCoverage =
                        _motionUncertaintyCoverage.value
                            .coerceIn(0.0, 1.0),
                    motionDynamicStress =
                        _motionDynamicStress.value
                            .coerceIn(0.0, 1.0),
                    motionDynamicsScore =
                        _motionDynamicsScore.value
                            .coerceIn(0.0, 1.0),
                    motionLowFrequencyScore =
                        _motionLowFrequencyScore.value
                            .coerceIn(0.0, 1.0),
                    motionHighFrequencyScore =
                        _motionHighFrequencyScore.value
                            .coerceIn(0.0, 1.0),
                    physicsConstraintScore =
                        _physicsConstraintScore.value
                            .coerceIn(0.0, 1.0),
                    physicsConstraintAccepted =
                        _physicsConstraintAccepted.value,
                    directionalUncertaintyMajor95Meters =
                        _directionalUncertaintyMajor95Meters.value
                            .coerceAtLeast(0.0),
                    directionalUncertaintyDepthMeters =
                        _directionalUncertaintyDepthMeters.value
                            .coerceAtLeast(0.0),
                    directionalUncertaintyAnisotropy =
                        _directionalUncertaintyAnisotropy.value
                            .coerceAtLeast(1.0),
                    directionalUncertaintyConfidence =
                        _directionalUncertaintyConfidence.value
                            .coerceIn(0.0, 1.0),
                    directionalUncertaintyDegradedJointCount =
                        _directionalUncertaintyDegradedJointCount.value,
                    faceEyeFusionAccepted =
                        _faceEyeFusionAccepted.value,
                    faceEyeHardFailure =
                        _faceEyeHardFailureReasons.value.isNotEmpty() &&
                            _faceEyeFusionWarnings.value.any {
                                it.contains("hard-failure") ||
                                    it.contains("gate")
                            },
                    faceEyeShapeConfidence =
                        _faceEyeShapeConfidence.value
                            .coerceIn(0.0, 1.0),
                    faceEyeEyeDynamicConsistency =
                        _faceEyeEyeDynamicConsistency.value
                            .coerceIn(0.0, 1.0),
                    faceEyeOutOfDistributionRisk =
                        _faceEyeOutOfDistributionRisk.value
                            .coerceIn(0.0, 1.0),
                    faceEyeUncertaintyCalibrationConfidence =
                        _faceEyeUncertaintyCalibrationConfidence.value
                            .coerceIn(0.0, 1.0),
                    faceEyeUncertaintyCalibrationReady =
                        _faceEyeUncertaintyCalibrationReady.value,
                    faceUncertaintyCoverage =
                        _faceUncertaintyCoverage.value
                            .coerceIn(0.0, 1.0),
                    eyeUncertaintyCoverage =
                        _eyeUncertaintyCoverage.value
                            .coerceIn(0.0, 1.0),
                    faceUncertaintyCalibrationMultiplier =
                        _faceUncertaintyCalibrationMultiplier.value
                            .coerceAtLeast(0.0),
                    eyeUncertaintyCalibrationMultiplier =
                        _eyeUncertaintyCalibrationMultiplier.value
                            .coerceAtLeast(0.0),
                    faceEyeFusionStrictness =
                        _motionCustomization.value
                            .faceEyeFusionStrictness
                            .coerceIn(0.0, 1.0),
                    faceEyeReliabilityTrend =
                        _faceEyeReliabilityTrend.value
                            .coerceIn(-1.0, 1.0),
                    faceEyeCaptureDomainConfidence =
                        _faceEyeCaptureDomainConfidence.value
                            .coerceIn(0.0, 1.0),
                    faceEyeSaccadeVelocityDegreesPerSecond =
                        _faceEyeSaccadeVelocityDegreesPerSecond.value
                            .coerceAtLeast(0.0),
                    faceEyeDominantFailure =
                        _faceEyeFailureAttribution.value
                            .firstOrNull()
                            ?.substringBefore(':')
                            ?: "NONE",
                    globalRigTrust =
                        _adaptiveViewEvidenceGlobalRigTrust.value
                            .coerceIn(0.0, 1.0),
                    adaptiveDetailScore =
                        adaptiveDetailBudget.score
                            .coerceIn(0.0, 1.0),
                    adaptiveDetailHold =
                        adaptiveDetailBudget.holdOutput,
                    thermalDriftConfidence =
                        _thermalExtrinsicDriftConfidence.value
                            .coerceIn(0.0, 1.0),
                    thermalDriftHardFailure =
                        _thermalExtrinsicHardFailure.value,
                    concurrentRigScore =
                        if (
                            _concurrentRigAdmissionState.value ==
                                "NOT_APPLICABLE"
                        ) {
                            _adaptiveViewEvidenceGlobalRigTrust.value
                                .coerceIn(0.0, 1.0)
                        } else {
                            _concurrentRigAdmissionScore.value
                                .coerceIn(0.0, 1.0)
                        },
                    concurrentRigAdmissible =
                        if (
                            _concurrentRigAdmissionState.value ==
                                "NOT_APPLICABLE"
                        ) {
                            _rigHealthState.value !=
                                MultiCameraRigHealthEngine.State.BLOCKED.name
                        } else {
                            _concurrentRigAdmissionState.value ==
                                "ADMISSIBLE"
                        },
                    faceEyeMetricEvidence =
                        _faceEyeMetricConfidence.value
                            .coerceIn(0.0, 1.0),
                    hardwareReadoutTimingAvailable =
                        camera.hardwareReadoutTimestampSupported,
                    hardwareReadoutTimingConfidence =
                        _hardwareReadoutTimingConfidence.value
                            .coerceIn(0.0, 1.0),
                    hardwareReadoutTimingHardFailure =
                        camera.hardwareReadoutTimestampSupported &&
                            _hardwareReadoutTimingState.value ==
                                HardwareReadoutTimingEvidenceEngine.State.BLOCKED.name,
                    multiPersonPredictionRatio =
                        (
                            _multiPerson3DPredictedCount.value.toDouble() /
                                _multiPersonAdmittedCount.value.coerceAtLeast(1).toDouble()
                        ).coerceIn(0.0, 1.0),
                    global4DScore =
                        _global4DScore.value
                            .coerceIn(0.0, 1.0),
                    global4DAccepted =
                        _global4DAccepted.value,
                    global4DPersistentFailure =
                        _global4DPersistentFailure.value,
                    global4DRootMotionConsistency =
                        _global4DRootMotionConsistency.value
                            .coerceIn(0.0, 1.0),
                    global4DRootVelocityAgreement =
                        _global4DRootVelocityAgreement.value
                            .coerceIn(0.0, 1.0),
                    global4DSceneConsistency =
                        _global4DSceneConsistency.value
                            .coerceIn(0.0, 1.0),
                    global4DDirectionalUncertaintyConsistency =
                        _global4DDirectionalUncertaintyConsistency.value
                            .coerceIn(0.0, 1.0),
                    global4DFaceEyeConsistency =
                        _global4DFaceEyeConsistency.value
                            .coerceIn(0.0, 1.0),
                    hardCaptureFailure =
                        result.hardFailures.isNotEmpty(),
                    warnings =
                        result.warnings +
                            _captureEvidenceWarnings.value,
                ),
            )

        val connectedCamerasForAuthority =
            _devices.value.count { it.connected }
        val multiViewForAuthority =
            connectedCamerasForAuthority >= 2
        val authoritativeCamerasForAuthority =
            if (!multiViewForAuthority) {
                if (connectedCamerasForAuthority > 0) 1 else 0
            } else {
                _cameraContributionSelectedCount.value
                    .coerceIn(0, connectedCamerasForAuthority)
            }
        val captureAuthority =
            com.nivukx.nox.core.reconstruction.HighEndCaptureAuthorityEngine(
                com.nivukx.nox.core.reconstruction.HighEndCaptureAuthorityEngine.Config(
                    maximumMotionBlurRisk =
                        _motionCustomization.value
                            .maximumMotionBlurRiskForAuthority,
                    minimumEpipolar =
                        _motionCustomization.value
                            .minimumEpipolarConsistencyForAuthority,
                    minimumMetricEvidence =
                        _motionCustomization.value
                            .minimumMetricEvidenceAuthorityForAuthority,
                    minimumAuthorityScore =
                        _motionCustomization.value
                            .minimumCaptureAuthorityScore,
                    minimumPrecisionScore =
                        _motionCustomization.value
                            .precisionAuthorityMinimumScore,
                    minimumPrecisionEvidenceSamples =
                        _motionCustomization.value
                            .precisionAuthorityMinimumSamples,
                ),
            ).evaluate(
                com.nivukx.nox.core.reconstruction.HighEndCaptureAuthorityEngine.Input(
                    timestampNanos = result.timestampNanos,
                    connectedCameras = connectedCamerasForAuthority,
                    authoritativeCameras =
                        authoritativeCamerasForAuthority,
                    calibrationConfidence =
                        _rigScore.value.coerceIn(0.0, 1.0),
                    synchronizationConfidence =
                        _multiCameraSyncConfidence.value
                            .coerceIn(0.0, 1.0),
                    timingAuthorityConfidence =
                        _captureTimingConfidence.value
                            .coerceIn(0.0, 1.0),
                    temporalConsistency =
                        _temporalStability.value.coerceIn(0.0, 1.0),
                    viewDiversity =
                        _geometricObservabilityScore.value
                            .coerceIn(0.0, 1.0),
                    crossViewAgreement =
                        _denseLandmarkCrossViewAgreement.value
                            .coerceIn(0.0, 1.0),
                    epipolarConsistency =
                        if (multiViewForAuthority) {
                            _bodySymmetricEpipolarConsistency.value
                                .coerceIn(0.0, 1.0)
                        } else {
                            1.0
                        },
                    bodyMetricEvidenceAuthority =
                        if (multiViewForAuthority) {
                            _bodyMetricEvidenceAuthority.value
                                .coerceIn(0.0, 1.0)
                        } else {
                            1.0
                        },
                    faceMetricEvidenceAuthority =
                        _faceEyeMetricConfidence.value
                            .takeIf { it > 0.0 }
                            ?.coerceIn(0.0, 1.0),
                    handMetricEvidenceAuthority = null,
                    faceEpipolarConsistency = null,
                    handEpipolarConsistency = null,
                    reconstructionUncertaintyMillimeters =
                        (
                            _reconstructionUncertaintyP95Meters.value
                                .takeIf { it.isFinite() }
                                ?: 0.0
                        ) * 1000.0,
                    motionP95ErrorMillimeters = 0.0,
                    motionP99ErrorMillimeters = 0.0,
                    temporalLagMilliseconds =
                        _temporalLagMilliseconds.value
                            .takeIf { it.isFinite() }
                            ?.coerceAtLeast(0.0)
                            ?: 0.0,
                    rollingShutterRisk =
                        com.nivukx.nox.core.ai.MotionCaptureEngine
                            .rollingShutterRiskFromTelemetry(
                                rollingShutterSkewNanos =
                                    camera.latestTelemetry
                                        ?.rollingShutterSkewNanos
                                        ?: 0L,
                                frameDurationNanos =
                                    camera.latestTelemetry
                                        ?.frameDurationNanos
                                        ?: 0L,
                                captureTimingConfidence =
                                    _captureTimingConfidence.value
                                        .coerceIn(0.0, 1.0),
                            )
                            .coerceIn(0.0, 1.0),
                    motionBlurRisk =
                        _cameraOpticalBlurRisk.value
                            .coerceIn(0.0, 1.0),
                    cameraMotionRisk =
                        _cameraMotionRisk.value
                            .coerceIn(0.0, 1.0),
                    identityContinuity =
                        _multiPersonIdentityContinuity.value
                            .coerceIn(0.0, 1.0),
                    occlusionRisk =
                        _occlusionRisk.value.coerceIn(0.0, 1.0),
                    contactConsistency =
                        minOf(
                            _denseContactConfidence.value
                                .coerceIn(0.0, 1.0),
                            1.0 -
                                _contactPenetrationRisk.value
                                    .coerceIn(0.0, 1.0),
                        ),
                    thermalHeadroom =
                        thermalScale.coerceIn(0.0, 1.0),
                    memoryHeadroom =
                        memoryScale.coerceIn(0.0, 1.0),
                    gpuHeadroom =
                        (
                            1.0 -
                                _gpuDispatchMillis.value /
                                    16.67
                        ).coerceIn(0.20, 1.0),
                    transportHeadroom = 1.0,
                    observedGeometryRatio =
                        _denseLandmarkVisibleCoverage.value
                            .coerceIn(0.0, 1.0),
                    inferredGeometryRatio =
                        (
                            1.0 -
                                _denseLandmarkVisibleCoverage.value
                                    .coerceIn(0.0, 1.0)
                        ).coerceIn(0.0, 1.0),
                    metricScaleUncertaintyRatio = 0.0,
                    targetP95Millimeters =
                        _motionCustomization.value
                            .targetMotionP95ErrorMillimeters
                            .coerceAtLeast(0.10),
                    targetRequiresIndependentReference =
                        _motionCustomization.value
                            .ultraPrecisionEnabled &&
                            _motionCustomization.value
                                .ultraPrecisionMatchedReferenceRequired,
                ),
            )

        _captureAuthorityScore.value =
            captureAuthority.score
        _captureAuthorityBand.value =
            captureAuthority.band.name
        _captureAuthorityHardBlocked.value =
            captureAuthority.hardBlocked
        _captureAuthorityPrecisionQualified.value =
            captureAuthority.precisionQualified
        _captureAuthorityWarnings.value =
            (
                captureAuthority.warnings +
                    captureAuthority.recommendations
            ).distinct()

        _productionQualityScore.value =
            minOf(
                productionDecision.score,
                captureAuthority.score,
            )
        _productionPolicy.value =
            productionDecision.policy.name
        _productionBottleneck.value =
            productionDecision.bottleneck.name
        _productionOutputAllowed.value =
            productionDecision.outputAllowed &&
                !captureAuthority.hardBlocked
        _productionWarnings.value =
            productionDecision.warnings
        _productionDetectorInterval.value =
            productionDecision.budget.detectorIntervalFrames
        _productionMeshVertexBudget.value =
            productionDecision.budget.meshVertexBudget
        _productionTsdfStride.value =
            productionDecision.budget.tsdfStride
        _productionBundleAdjustmentIterations.value =
            productionDecision.budget.bundleAdjustmentIterations

        val connectedCalibrationDevices =
            _devices.value
                .filter { it.connected }
                .map { it.id }
                .distinct()

        val calibrationUncertaintyProfiles =
            calibrationSnapshot
                ?.uncertaintyProfiles
                .orEmpty()

        val calibrationUncertaintyEvidenceDevices =
            connectedCalibrationDevices.count { id ->
                calibrationUncertaintyProfiles[id]
                    ?.hasEvidence == true
            }

        val calibrationUncertaintyCoverage =
            if (connectedCalibrationDevices.isEmpty()) {
                0.0
            } else {
                calibrationUncertaintyEvidenceDevices.toDouble() /
                    connectedCalibrationDevices.size.toDouble()
            }

        val calibrationUncertaintyEvidenceConfidence =
            connectedCalibrationDevices
                .mapNotNull {
                    calibrationUncertaintyProfiles[it]
                        ?.takeIf { profile -> profile.hasEvidence }
                        ?.evidenceConfidence
                }
                .average()
                .takeIf(Double::isFinite)
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        val precisionCapability =
            physicalPrecisionCapabilityEngine.evaluate(
                PhysicalPrecisionCapabilityEngine.Input(
                    timestampNanos = result.timestampNanos,
                    reconstructionUncertaintyP95Meters =
                        _reconstructionUncertaintyP95Meters.value.coerceAtLeast(0.0),
                    directionalMajor95Meters =
                        _directionalUncertaintyMajor95Meters.value.coerceAtLeast(0.0),
                    directionalDepth95Meters =
                        _directionalUncertaintyDepthMeters.value.coerceAtLeast(0.0),
                    geometricInformationP95MajorMeters =
                        _geometricInformationP95MajorMeters.value.coerceAtLeast(0.0),
                    geometricInformationP95DepthMeters =
                        _geometricInformationP95DepthMeters.value.coerceAtLeast(0.0),
                    denseSurfaceUncertaintyP95Meters =
                        _denseSurfaceUncertaintyP95Meters.value
                            .takeIf(Double::isFinite)
                            ?.coerceAtLeast(0.0)
                            ?: _reconstructionUncertaintyP95Meters.value.coerceAtLeast(0.0),
                    calibrationConfidence = _rigScore.value.coerceIn(0.0, 1.0),
                    calibrationUncertaintyCoverage =
                        calibrationUncertaintyCoverage
                            .coerceIn(0.0, 1.0),
                    calibrationUncertaintyEvidenceConfidence =
                        calibrationUncertaintyEvidenceConfidence
                            .coerceIn(0.0, 1.0),
                    synchronizationConfidence = _multiCameraSyncConfidence.value.coerceIn(0.0, 1.0),
                    temporalAlignmentConfidence = _temporalAlignmentConfidence.value.coerceIn(0.0, 1.0),
                    temporalAlignmentSpreadMillis = _temporalAlignmentSpreadMillis.value.coerceAtLeast(0.0),
                    crossViewAgreement = _denseLandmarkCrossViewAgreement.value.coerceIn(0.0, 1.0),
                    viewDiversity = _denseSurfaceViewDiversity.value.coerceIn(0.0, 1.0),
                    cameraInformationGain = _cameraContributionInformationGainScore.value.coerceIn(0.0, 1.0),
                    cameraCount = _devices.value.count { it.connected },
                    motionUncertaintyConfidence = _motionUncertaintyConfidence.value.coerceIn(0.0, 1.0),
                    motionUncertaintyRadiusMeters = _motionUncertaintyRadiusMeters.value.coerceAtLeast(0.0),
                    captureTimingConfidence =
                        minOf(
                            _captureTimingConfidence.value
                                .coerceIn(0.0, 1.0),
                            _hardwareReadoutTimingConfidence.value
                                .takeIf { _hardwareReadoutTimestampSupported.value }
                                ?.coerceIn(0.0, 1.0)
                                ?: 1.0,
                        ),
                    captureTimingJitterMillis =
                        maxOf(
                            _captureTimingJitterMillis.value.coerceAtLeast(0.0),
                            _hardwareReadoutP95JitterMillis.value
                                .takeIf { it.isFinite() }
                                ?.coerceAtLeast(0.0)
                                ?: 0.0,
                        ),
                    opticalSignalQuality =
                        _lightingConfidence.value.coerceIn(0.0, 1.0),
                    temporalStability = _temporalStability.value.coerceIn(0.0, 1.0),
                    occlusionRisk = _occlusionRisk.value.coerceIn(0.0, 1.0),
                    runtimeHealthy =
                        _captureIntegrityOutputAllowed.value &&
                            _runtimeInferenceScale.value >= 0.50 &&
                            _sessionHealthScore.value >= 0.25,
                    benchmarkArmed = _motionCustomization.value.ultraPrecisionEnabled,
                ),
            )
        _physicalPrecisionCapability.value = precisionCapability
        _productionBudgetLevel.value =
            when (productionDecision.policy) {
                HighEndReconstructionQualityEngine.Policy.ACCEPT ->
                    "REFERENCE"
                HighEndReconstructionQualityEngine.Policy.HOLD ->
                    "HIGH"
                HighEndReconstructionQualityEngine.Policy.DEGRADE ->
                    "PROTECTED"
                HighEndReconstructionQualityEngine.Policy.REJECT ->
                    "SAFE"
            }

        recorder.appendProductionPolicy(
            ProductionPolicyRecord(
                timestampNanos = productionDecision.timestampNanos,
                profileId = _productionProfile.value.id,
                policy = productionDecision.policy.name,
                bottleneck = productionDecision.bottleneck.name,
                score = productionDecision.score,
                rawScore = productionDecision.rawScore,
                outputAllowed = productionDecision.outputAllowed,
                hardFailure = productionDecision.hardFailure,
                detectorIntervalFrames =
                    productionDecision.budget.detectorIntervalFrames,
                targetInferenceFps =
                    productionDecision.budget.targetInferenceFps,
                targetCaptureFps =
                    productionDecision.budget.targetCaptureFps,
                denseMeshHz =
                    productionDecision.budget.denseMeshHz,
                faceMeshHz =
                    productionDecision.budget.faceMeshHz,
                eyeTrackingHz =
                    productionDecision.budget.eyeTrackingHz,
                handTrackingHz =
                    productionDecision.budget.handTrackingHz,
                bundleAdjustmentIterations =
                    productionDecision.budget.bundleAdjustmentIterations,
                tsdfStride =
                    productionDecision.budget.tsdfStride,
                meshVertexBudget =
                    productionDecision.budget.meshVertexBudget,
                allowDenseSurface =
                    productionDecision.budget.allowDenseSurface,
                allowReferenceOptimization =
                    productionDecision.budget.allowReferenceOptimization,
                calibratedUncertaintyConfidence =
                    _motionUncertaintyConfidence.value
                        .coerceIn(0.0, 1.0),
                motionOutOfDistributionRisk =
                    _motionUncertaintyOodRisk.value
                        .coerceIn(0.0, 1.0),
                motionUncertaintyCoverage =
                    _motionUncertaintyCoverage.value
                        .coerceIn(0.0, 1.0),
                globalRigTrust =
                    _adaptiveViewEvidenceGlobalRigTrust.value
                        .coerceIn(0.0, 1.0),
                motionDynamicStress =
                    _motionDynamicStress.value
                        .coerceIn(0.0, 1.0),
                motionDynamicsScore =
                    _motionDynamicsScore.value
                        .coerceIn(0.0, 1.0),
                motionLowFrequencyScore =
                    _motionLowFrequencyScore.value
                        .coerceIn(0.0, 1.0),
                motionHighFrequencyScore =
                    _motionHighFrequencyScore.value
                        .coerceIn(0.0, 1.0),
                motionFrequencyBandSensitivity =
                    _motionCustomization.value
                        .motionFrequencyBandSensitivity
                        .coerceIn(0.0, 1.0),
                physicsConstraintScore =
                    _physicsConstraintScore.value
                        .coerceIn(0.0, 1.0),
                physicsConstraintAccepted =
                    _physicsConstraintAccepted.value,
                handUncertaintyConfidence =
                    _handUncertaintyConfidence.value
                        .coerceIn(0.0, 1.0),
                handUncertaintyRadiusMeters =
                    _handUncertaintyRadiusMeters.value
                        .coerceAtLeast(0.0),
                handUncertaintyCoverage =
                    _handUncertaintyCoverage.value
                        .coerceIn(0.0, 1.0),
                handUncertaintyOodRisk =
                    _handUncertaintyOodRisk.value
                        .coerceIn(0.0, 1.0),
                handUncertaintyPolicy =
                    _handUncertaintyPolicy.value,
                surfaceNormalObservability =
                    _denseTsdfSurfaceNormalObservability.value
                        .coerceIn(0.0, 1.0),
                surfaceHighUncertaintyRatio =
                    _denseTsdfHighUncertaintyRatio.value
                        .coerceIn(0.0, 1.0),
                surfaceCameraInformationGain =
                    _cameraContributionInformationGainScore.value
                        .coerceIn(0.0, 1.0),
                surfaceEvidenceStrictness =
                    _motionCustomization.value.surfaceEvidenceStrictness
                        .coerceIn(0.0, 1.0),
                physicsConstraintStrictness =
                    _motionCustomization.value.physicsConstraintStrictness
                        .coerceIn(0.0, 1.0),
                global4DConsistencyStrictness =
                    _motionCustomization.value.global4DConsistencyStrictness
                        .coerceIn(0.0, 1.0),
                faceEyeFusionAccepted =
                    _faceEyeFusionAccepted.value,
                faceEyeHardFailure =
                    _faceEyeHardFailureReasons.value.isNotEmpty() &&
                        _faceEyeHardFailureReasons.value.any {
                            it.contains("hard") ||
                                it.contains("rejected") ||
                                it == "face-stale"
                        },
                faceEyeFusionConfidence =
                    _faceEyeFusionConfidence.value
                        .coerceIn(0.0, 1.0),
                faceEyeShapeConfidence =
                    _faceEyeShapeConfidence.value
                        .coerceIn(0.0, 1.0),
                faceEyeEyeDynamicConsistency =
                    _faceEyeEyeDynamicConsistency.value
                        .coerceIn(0.0, 1.0),
                faceEyeHardFailureReasonCount =
                    _faceEyeHardFailureReasons.value.size,
                faceEyeOutOfDistributionRisk =
                    _faceEyeOutOfDistributionRisk.value
                        .coerceIn(0.0, 1.0),
                faceEyeFusionStrictness =
                    _motionCustomization.value.faceEyeFusionStrictness
                        .coerceIn(0.0, 1.0),
                faceEyeSaccadeVelocityDegreesPerSecond =
                    _faceEyeSaccadeVelocityDegreesPerSecond.value
                        .coerceAtLeast(0.0),
                faceEyeDominantFailure =
                    _faceEyeFailureAttribution.value
                        .firstOrNull()
                        ?.substringBefore(':')
                        ?: "NONE",
                faceEyeCalibrationDriftScore =
                    _faceEyeCalibrationDriftScore.value
                        .coerceIn(0.0, 1.0),
                faceEyeCalibrationDriftPersistent =
                    _faceEyeCalibrationDriftPersistent.value,
                faceEyeIpdDriftToleranceMillimeters =
                    _motionCustomization.value
                        .faceEyeIpdDriftToleranceMillimeters
                        .coerceIn(4.0, 30.0),
                physicsMeanBoneResidualMeters =
                    _physicsConstraintMeanBoneResidualMeters.value
                        .coerceAtLeast(0.0),
                physicsP95BoneResidualMeters =
                    _physicsConstraintP95BoneResidualMeters.value
                        .coerceAtLeast(0.0),
                physicsMaxBoneResidualMeters =
                    _physicsConstraintMaxBoneResidualMeters.value
                        .coerceAtLeast(0.0),
                physicsSymmetryScore =
                    _physicsConstraintSymmetryScore.value
                        .coerceIn(0.0, 1.0),
                physicsFloorContactScore =
                    _physicsConstraintFloorContactScore.value
                        .coerceIn(0.0, 1.0),
                physicsCorrectedJointCount =
                    _physicsConstraintCorrectedJointCount.value,
                physicsConstrainedEdgeCount =
                    _physicsConstraintConstrainedEdgeCount.value,
                directionalUncertaintyMajor95Meters =
                    _directionalUncertaintyMajor95Meters.value
                        .coerceAtLeast(0.0),
                directionalUncertaintyDepthMeters =
                    _directionalUncertaintyDepthMeters.value
                        .coerceAtLeast(0.0),
                directionalUncertaintyAnisotropy =
                    _directionalUncertaintyAnisotropy.value
                        .coerceAtLeast(1.0),
                directionalUncertaintyConfidence =
                    _directionalUncertaintyConfidence.value
                        .coerceIn(0.0, 1.0),
                directionalUncertaintyDegradedJointCount =
                    _directionalUncertaintyDegradedJointCount.value,
                global4DScore =
                    _global4DScore.value
                        .coerceIn(0.0, 1.0),
                global4DAccepted =
                    _global4DAccepted.value,
                global4DPersistentFailure =
                    _global4DPersistentFailure.value,
                global4DRootMotionConsistency =
                    _global4DRootMotionConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DRootVelocityAgreement =
                    _global4DRootVelocityAgreement.value
                        .coerceIn(0.0, 1.0),
                global4DSceneConsistency =
                    _global4DSceneConsistency.value
                        .coerceIn(0.0, 1.0),
                warnings =
                    (
                        productionDecision.warnings +
                            _physicsConstraintWarnings.value
                    ).distinct(),
            ),
        )

        val ledger =
            captureEvidenceLedger.append(
                com.nivukx.nox.core.ai.ProductionCaptureEvidenceLedger.Sample(
                    timestampNanos = result.timestampNanos,
                    integrity = result.smoothedScore,
                    reconstruction = result.components.reconstruction,
                    geometry = result.components.geometry,
                    temporal = result.components.temporal,
                    synchronization = result.components.synchronization,
                    calibration = result.components.calibration,
                    bodyTracking = result.components.tracking,
                    faceEye = result.components.faceEye,
                    faceEyeAccepted = _faceEyeFusionAccepted.value,
                    faceEyeReliabilityTrend =
                        _faceEyeReliabilityTrend.value
                            .coerceIn(-1.0, 1.0),
                    faceEyeCalibrationDriftScore =
                        _faceEyeCalibrationDriftScore.value
                            .coerceIn(0.0, 1.0),
                    faceEyeCalibrationDriftPersistent =
                        _faceEyeCalibrationDriftPersistent.value,
                    hand = result.components.hand,
                    kinematic = result.components.kinematic,
                    surface = result.components.surface,
                    physics = result.components.physics,
                    global4D = _global4DScore.value
                        .coerceIn(0.0, 1.0),
                    global4DAccepted = _global4DAccepted.value,
                    global4DDirectionalUncertaintyConsistency =
                        _global4DDirectionalUncertaintyConsistency.value
                            .coerceIn(0.0, 1.0),
                    global4DFaceEyeConsistency =
                        _global4DFaceEyeConsistency.value
                            .coerceIn(0.0, 1.0),
                    global4DPersistentFailure =
                        _global4DPersistentFailure.value,
                    outputAllowed = result.outputAllowed,
                    hardFailure = result.hardFailures.isNotEmpty(),
                ),
            )

        _captureEvidenceTrend.value = ledger.trend.name
        _captureEvidenceSeverity.value = ledger.severity.name
        _captureEvidenceCurrentScore.value = ledger.currentScore
        _captureEvidenceRecentScore.value = ledger.recentMeanScore
        _captureEvidenceSlope.value = ledger.scoreSlopePerSample
        _captureEvidenceBottleneck.value =
            ledger.dominantComponent?.name?.uppercase()
                ?: "NONE"
        _captureEvidenceWarnings.value =
            ledger.warnings
        _captureReadiness.value?.let { readiness ->
            _captureReadiness.value =
                readiness.copy(
                    captureIntegrityScore = result.smoothedScore,
                    captureIntegrityHealthy = result.outputAllowed,
                    captureIntegrityBottleneck = result.bottleneck.name,
                    reasons =
                        (
                            readiness.reasons +
                                result.hardFailures +
                                if (!result.outputAllowed) {
                                    listOf(
                                        "capture-integrity-" +
                                            result.band.name.lowercase(),
                                    )
                                } else {
                                    emptyList()
                                }
                            )
                        .distinct()
                        .take(12),
                )
        }

        recorder.appendCaptureIntegrity(            CaptureIntegrityEvidence(
                timestampNanos = result.timestampNanos,
                rawScore = result.rawScore.coerceIn(0.0, 1.0),
                smoothedScore = result.smoothedScore.coerceIn(0.0, 1.0),
                band = result.band.name,
                outputAllowed = result.outputAllowed,
                bottleneck = result.bottleneck.name,
                reconstruction = result.components.reconstruction,
                geometry = result.components.geometry,
                rigCoverage = result.components.rigCoverage,
                temporal = result.components.temporal,
                synchronization = result.components.synchronization,
                calibration = result.components.calibration,
                tracking = result.components.tracking,
                occlusion = result.components.occlusion,
                cameraMotion = result.components.cameraMotion,
                performance = result.components.performance,
                microMotion = result.components.microMotion,
                faceGeometry = result.components.faceGeometry,
                eyeTemporal = result.components.eyeTemporal,
                faceEye = result.components.faceEye,
                faceEyeFusionAccepted =
                    _faceEyeFusionAccepted.value,
                faceEyeFusionMode =
                    _faceEyeFusionMode.value,
                faceEyeIndependentEvidenceSupport =
                    _faceEyeIndependentEvidenceSupport.value
                        .coerceIn(0.0, 1.0),
                faceEyeReliabilityTrend =
                    _faceEyeReliabilityTrend.value
                        .coerceIn(-1.0, 1.0),
                globalRigTrust =
                    _adaptiveViewEvidenceGlobalRigTrust.value
                        .coerceIn(0.0, 1.0),
                faceEyeFusionConfidence =
                    _faceEyeFusionConfidence.value
                        .coerceIn(0.0, 1.0),
                faceEyeOutOfDistributionRisk =
                    _faceEyeOutOfDistributionRisk.value
                        .coerceIn(0.0, 1.0),
                faceEyeCalibrationDriftPersistent =
                    _faceEyeCalibrationDriftPersistent.value,
                hand = result.components.hand,
                kinematic = result.components.kinematic,
                surface = result.components.surface,
                physics = result.components.physics,
                physicsAccepted = _physicsConstraintAccepted.value,
                physicsMeanBoneResidualMeters =
                    _physicsConstraintMeanBoneResidualMeters.value
                        .coerceAtLeast(0.0),
                physicsP95BoneResidualMeters =
                    _physicsConstraintP95BoneResidualMeters.value
                        .coerceAtLeast(0.0),
                physicsMaxBoneResidualMeters =
                    _physicsConstraintMaxBoneResidualMeters.value
                        .coerceAtLeast(0.0),
                physicsSymmetryScore =
                    _physicsConstraintSymmetryScore.value
                        .coerceIn(0.0, 1.0),
                physicsFloorContactScore =
                    _physicsConstraintFloorContactScore.value
                        .coerceIn(0.0, 1.0),
                physicsCorrectedJointCount =
                    _physicsConstraintCorrectedJointCount.value,
                physicsConstrainedEdgeCount =
                    _physicsConstraintConstrainedEdgeCount.value,
                directionalUncertaintyMajor95Meters =
                    _directionalUncertaintyMajor95Meters.value
                        .coerceAtLeast(0.0),
                directionalUncertaintyDepthMeters =
                    _directionalUncertaintyDepthMeters.value
                        .coerceAtLeast(0.0),
                directionalUncertaintyAnisotropy =
                    _directionalUncertaintyAnisotropy.value
                        .coerceAtLeast(1.0),
                directionalUncertaintyConfidence =
                    _directionalUncertaintyConfidence.value
                        .coerceIn(0.0, 1.0),
                directionalUncertaintyDegradedJointCount =
                    _directionalUncertaintyDegradedJointCount.value,
                global4DScore =
                    _global4DScore.value
                        .coerceIn(0.0, 1.0),
                global4DAccepted =
                    _global4DAccepted.value,
                global4DPersistentFailure =
                    _global4DPersistentFailure.value,
                global4DRootMotionConsistency =
                    _global4DRootMotionConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DSceneConsistency =
                    _global4DSceneConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DDirectionalUncertaintyConsistency =
                    _global4DDirectionalUncertaintyConsistency.value
                        .coerceIn(0.0, 1.0),
                global4DFaceEyeConsistency =
                    _global4DFaceEyeConsistency.value
                        .coerceIn(0.0, 1.0),
                hardFailures = result.hardFailures,
                warnings = result.warnings,
                recommendations = result.recommendations,
            ),
        )
        return result
    }

    private fun appendAvatarIfSafe(
        skeleton: com.nivukx.nox.core.model.SkeletonFrame,
        avatar: com.nivukx.nox.core.model.AvatarPose,
        confidence: Double,
        reprojectionErrorPx: Double,
        uncertaintyConfidence: Double = 1.0,
        p95UncertaintyMeters: Double = 0.0,
        trustedJointRatio: Double = 1.0,
        thermalScale: Double = 1.0,
        memoryScale: Double = 1.0,
        runtimeQualityScale: Double = 1.0,
        uncertaintyBottleneck: String = "NONE",
    ) {
        val meanUncertainty =
            skeleton.joints
                .map { it.positionUncertaintyMeters }
                .filter { it.isFinite() }
                .average()
                .let { if (it.isNaN()) Double.POSITIVE_INFINITY else it }

        val band =
            when {
                confidence >= 0.90 ->
                    com.nivukx.nox.core.model.ReconstructionConfidenceBand.EXCELLENT
                confidence >= 0.75 ->
                    com.nivukx.nox.core.model.ReconstructionConfidenceBand.GOOD
                confidence >= 0.55 ->
                    com.nivukx.nox.core.model.ReconstructionConfidenceBand.USABLE
                confidence >= 0.35 ->
                    com.nivukx.nox.core.model.ReconstructionConfidenceBand.DEGRADED
                else ->
                    com.nivukx.nox.core.model.ReconstructionConfidenceBand.UNTRUSTED
            }

        val integrity = evaluateCaptureIntegrity(
            timestampNanos = skeleton.timestampNanos,
            confidence = confidence,
            reprojectionErrorPx = reprojectionErrorPx,
            meanUncertaintyMeters = meanUncertainty,
            thermalScale = thermalScale.coerceIn(0.0, 1.0),
            memoryScale = memoryScale.coerceIn(0.0, 1.0),
            runtimeQualityScale = runtimeQualityScale.coerceIn(0.0, 1.0),
            multiCamera = _devices.value.count { it.connected } >= 2,
        )

        val decision = avatarOutputGuard.evaluate(
            AvatarMotionOutputGuard.Input(
                skeleton = skeleton,
                avatarPose = avatar,
                confidenceBand = band,
                reprojectionErrorPx = reprojectionErrorPx,
                meanUncertaintyMeters = meanUncertainty,
                uncertaintyConfidence = uncertaintyConfidence,
                p95UncertaintyMeters = p95UncertaintyMeters,
                trustedJointRatio = trustedJointRatio,
                uncertaintyBottleneck = uncertaintyBottleneck,
            ),
        )

        if (
            integrity.outputAllowed &&
                _productionOutputAllowed.value &&
                decision.accepted &&
                decision.pose != null
        ) {
            val safePose = decision.pose ?: return
            recorder.appendAvatarPose(safePose)
        }
    }

    fun applyProductionProfile(profileId: String) {
        val selected = ProductionCaptureProfile.byId(profileId)
        _productionProfile.value = selected
        productionQualityEngine.reset()
        _productionQualityScore.value = 0.0
        _productionPolicy.value =
            HighEndReconstructionQualityEngine.Policy.HOLD.name
        _productionBottleneck.value =
            HighEndReconstructionQualityEngine.Bottleneck.NONE.name
        _productionOutputAllowed.value = false
        _productionWarnings.value = listOf(
            "production-profile-" + selected.id.lowercase(),
        )
    }

    fun productionProfiles(): List<ProductionCaptureProfile> =
        ProductionCaptureProfile.all()

    private fun fail(message: String) {
        _reconstructionErrorPx.value = Double.POSITIVE_INFINITY
        _face3DErrorPx.value = Double.POSITIVE_INFINITY
        _geometricObservabilityScore.value = 0.0
        _geometricObservabilityWarnings.value = emptyList()
        _geometricInformationConditionScore.value = 0.0
        _geometricInformationP95MajorMeters.value = 0.0
        _geometricInformationP95DepthMeters.value = 0.0
        _denseContactConfidence.value = 0.0
        _denseContactProtectedLandmarkCount.value = 0
        _denseContactWarnings.value = emptyList()
        _lastError.value = message
        _state.value = CaptureState.ERROR
    }

    override fun onCleared() {
        scheduledStart?.cancel()
        calibrationJob?.cancel()
        frameJob?.cancel()
        imuJob?.cancel()
        depthJob?.cancel()
        runCatching { peers.stop() }

        pendingPings.clear()

        runCatching {
            lightingTemporalConsistencyEngine.reset()
            motionEvidenceEngine.reset()
            fusion.reset()
            denseFusion.reset()
            nativeDenseTsdfBridge?.let {
                runCatching {
                    it.close()
                }
            }
            nativeDenseTsdfBridge = null
            nativeDenseTsdfFaultReported = false
            adaptiveDenseMeshExtractionController.reset()
            lastDenseMeshExtractNanos =
                Long.MIN_VALUE
            denseMeshTemporalConsistencyEngine.reset()
            retargeter.reset()
            faceFusion.clear()
            eyeFusion.clear()
            eye3DTemporalConsistencyEngine.reset()
            binocularGazeGeometryEngine.reset()
            faceEyeConsistencyEngine.reset()
            faceShapeAlignmentEngine.reset()
            captureIntegrityEngine.reset()
            productionQualityEngine.reset()
            physicalPrecisionCapabilityEngine.reset()
            _physicalPrecisionCapability.value =
                PhysicalPrecisionCapabilityEngine.Decision.idle()
            productionMotionDynamicsEngine.reset()
            captureEvidenceLedger.reset()
            cameraCaptureQualityEngine.reset()
            _cameraAnalysisDropRatio.value = 0.0
            handTemporalConsistencyEngine.reset()
            faceMeshTemporalConsistencyEngine.reset()
            faceMeshFusion.reset()
            handFusion.reset()
            monocularTrajectoryOptimizer.reset()
            handFrames.clear()
            denseEvidenceCameraIds.clear()
        }.onFailure {
            _lastError.value =
                "Session cleanup degraded: " +
                    (it.message ?: it::class.simpleName.orEmpty())
        }

        closeSafely("imu-capture") {
            imuCapture.close()
        }
        closeSafely("recorder") {
            recorder.close()
        }
        closeSafely("post-capture-reconstruction") {
            recordedVideoPostProcessor.close()
        }
        closeSafely("post-capture-reconstruction-bridge") {
            postCaptureReconstructionBridge.close()
        }
        closeSafely("cross-camera-actor-identity") {
            crossCameraActorIdentityFusionEngine.reset()
        }
        closeSafely("hybrid-capture") {
            hybridCaptureCoordinator.close()
        }
        closeSafely("perception") {
            perception.close()
        }
        closeSafely("gpu-acceleration") {
            gpuAcceleration.close()
        }
        closeSafely("dense-landmark-evidence") {
            globalDenseLandmarkEvidenceEngine.close()
        }
        closeSafely("orientation") {
            orientation.close()
        }
        closeSafely("camera") {
            camera.close()
        }
        closeSafely("physical-precision") {
            physicalPrecisionCapabilityEngine.close()
        }
        closeSafely("native-motion") {
            nativeMotionEngine?.close()
        }

        nativeMotionEngineStarted = false
        super.onCleared()
    }

    private fun closeSafely(
        resourceName: String,
        closeAction: () -> Unit,
    ) {
        runCatching {
            closeAction()
        }.onFailure { error ->
            _lastError.value =
                "Resource cleanup failed [$resourceName]: " +
                    (error.message ?: error::class.simpleName.orEmpty())
        }
    }
    private fun handleDenseLandmarkFrame(
        endpoint: DeviceId,
        frame: DenseLandmarkWireCodec.Frame,
    ) {
        runCatching {
            val masterTimestamp =
                clock.convertPeerToMaster(
                    endpoint,
                    frame.timestampNanos,
                )

            val denseObservations =
                frame.landmarks.map { landmark ->
                    DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                        subjectId = landmark.subjectId,
                        landmarkId = landmark.landmarkId,
                    ) to DenseLandmarkEvidenceFusionEngine.CameraObservation(
                        cameraId =
                            denseEvidenceCameraId(
                                endpoint,
                            ),
                        timestampNanos =
                            masterTimestamp,
                        normalizedX =
                            landmark.normalizedX.toDouble(),
                        normalizedY =
                            landmark.normalizedY.toDouble(),
                        confidence =
                            landmark.confidence.toDouble(),
                        uncertainty =
                            landmark.uncertainty.toDouble(),
                        visibility =
                            landmark.visibility.toDouble(),
                        contactProbability =
                            landmark.contactProbability.toDouble(),
                        cameraQuality =
                            landmark.temporalStability.toDouble(),
                        reprojectionResidualNormalized =
                            landmark.reprojectionResidualNormalized.toDouble(),
                        probabilitySigmaXNormalized =
                            landmark.probabilitySigmaXNormalized.toDouble(),
                        probabilitySigmaYNormalized =
                            landmark.probabilitySigmaYNormalized.toDouble(),
                        probabilityCalibrationScore =
                            landmark.probabilityCalibrationScore.toDouble(),
                        segmentationConsistency =
                            landmark.segmentationConsistency.toDouble(),
                        segmentationEvidenceAvailable =
                            landmark.segmentationEvidenceAvailable,
                        sourceTag = "peer-dense",
                    )
                }

            val globalDense =
                globalDenseLandmarkEvidenceEngine.process(
                    DenseLandmarkEvidenceFusionEngine.FrameInput(
                        timestampNanos = masterTimestamp,
                        observations = denseObservations,
                    ),
                )

            updateDenseLandmarkUiFromGlobal(
                dense = globalDense,
                connectedCameraCount =
                    _devices.value.count { it.connected }
                        .coerceAtLeast(1),
            )

            fusion.pushDenseLandmarks(
                deviceId = endpoint,
                timestampNanos = masterTimestamp,
                landmarks =
                    frame.landmarks.map { landmark ->
                        com.nivukx.nox.core.reconstruction
                            .MultiViewFusionEngine
                            .DenseLandmarkInput(
                                subjectId =
                                    landmark.subjectId,
                                landmarkId =
                                    landmark.landmarkId,
                                normalizedX =
                                    landmark.normalizedX
                                        .toDouble(),
                                normalizedY =
                                    landmark.normalizedY
                                        .toDouble(),
                                confidence =
                                    landmark.confidence
                                        .toDouble(),
                                uncertainty =
                                    landmark.uncertainty
                                        .toDouble(),
                                visibility =
                                    landmark.visibility
                                        .toDouble(),
                                contactProbability =
                                    landmark.contactProbability
                                        .toDouble(),
                                temporalStability =
                                    landmark.temporalStability
                                        .toDouble(),
                                crossViewAgreement =
                                    landmark.crossViewAgreement
                                        .toDouble(),
                                reprojectionResidualNormalized =
                                    landmark
                                        .reprojectionResidualNormalized
                                        .toDouble(),
                                probabilitySigmaXNormalized =
                                    landmark
                                        .probabilitySigmaXNormalized
                                        .toDouble(),
                                probabilitySigmaYNormalized =
                                    landmark
                                        .probabilitySigmaYNormalized
                                        .toDouble(),
                                probabilityCalibrationScore =
                                    landmark
                                        .probabilityCalibrationScore
                                        .toDouble(),
                                silhouetteConsistency =
                                    landmark.segmentationConsistency.toDouble(),
                                frameHeightPx = frame.imageHeight,
                                rollingShutterSkewNanos =
                                    frame.rollingShutterSkewNanos,
                                frameDurationNanos = frame.frameDurationNanos,
                                exposureTimeNanos = frame.exposureTimeNanos,
                                captureTimingConfidence =
                                    frame.captureTimingConfidence.toDouble(),
                                predicted =
                                    landmark.predicted,
                            )
                    },
            )

            fusion
                .reconstructBufferedDenseSurface(
                    timestampNanos = masterTimestamp,
                )
                ?.let(::applyDenseSurfaceResult)
        }.onFailure {
            _lastError.value =
                "Dense landmark reconstruction failed: " +
                    (it.message ?: "invalid dense landmark frame")
        }
    }


}
