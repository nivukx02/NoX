package com.nivukx.nox.core.ai

import android.content.Context
import android.media.Image
import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.mediapipe.framework.image.MediaImageBuilder
import com.nivukx.nox.core.model.EyeTrackingFrame
import com.nivukx.nox.core.model.FaceContourFrame
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.SubjectTrack
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.PerceptionSource
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.MotionInferenceDecision
import com.nivukx.nox.core.model.TrackedObject
import kotlin.math.max
import kotlin.math.min

/**
 * High-throughput multimodal perception graph.
 *
 * Capture frames are never dropped by this class. Instead, expensive perception stages are
 * independently scheduled and recent results are retained for a bounded interval. This preserves
 * temporal continuity while allowing the body, face, eye, hand, object and contour engines to run
 * at different rates.
 *
 * Important invariants:
 * - pose is always the temporal anchor for the motion estimator;
 * - cached detector output is never fed back into a tracker as a new observation;
 * - stale values are explicitly marked;
 * - all published result objects remain timestamped and coherent;
 * - closing the graph releases every underlying detector.
 */
class PerceptionPipeline(
    context: Context,
    private val pose: PoseDetectionEngine = PoseDetectionEngine(),
    private val face: FaceLandmarkEngine = FaceLandmarkEngine(),
    private val faceMesh: FaceMeshGeometryEngine = FaceMeshGeometryEngine(),
    private val contours: FaceContourEngine = FaceContourEngine(),
    private val eyes: EyeTrackingEngine = EyeTrackingEngine(),
    private val hands: HandTrackingEngine = HandTrackingEngine(context),
    private val objects: ObjectTrackingEngine = ObjectTrackingEngine(),
    private val subjectTracks: SubjectTrackManager = SubjectTrackManager(),
    private val faceTracks: FaceTrackManager = FaceTrackManager(),
    private val faceStabilizer: FaceTemporalStabilizer = FaceTemporalStabilizer(),
    private val motion: MotionCaptureEngine = MotionCaptureEngine(),
    private val coordinator: AdaptivePerceptionCoordinator =
        AdaptivePerceptionCoordinator(),
    private val motionFusion: AdvancedMotionFusionEngine =
        AdvancedMotionFusionEngine(),
    private val subjectLockEngine: ProductionSubjectLockEngine =
        ProductionSubjectLockEngine(),
    private val runtimeGovernor: ProductionPerceptionRuntimeGovernor =
        ProductionPerceptionRuntimeGovernor(),
    private val highEndFusionEngine: HighEndPerceptionFusionEngine =
        HighEndPerceptionFusionEngine(),
    private val denseLandmarkFusionEngine:
        DenseLandmarkEvidenceFusionEngine =
        DenseLandmarkEvidenceFusionEngine(),
    private val denseContactEvidenceAdapter:
        DenseContactEvidenceAdapter =
        DenseContactEvidenceAdapter(),
    private val subpixelLandmarkRefinementEngine:
        SubpixelLandmarkRefinementEngine =
        SubpixelLandmarkRefinementEngine(),
    private val highOrderMotionRefinementEngine:
        HighOrderMotionRefinementEngine =
        HighOrderMotionRefinementEngine(),
    private val faceEyeFusionEngine: FaceEyeFusionEngine =
        FaceEyeFusionEngine(),
    private val subjectReidentification: SubjectReidentificationEngine =
        SubjectReidentificationEngine(),
    private val multiActorPose: MultiActorPoseLandmarkerEngine =
        MultiActorPoseLandmarkerEngine(context),
    private val multiActorAssociation:
        MultiActorPoseAssociationEngine =
        MultiActorPoseAssociationEngine(),
    private val subframeTemporalCompensationEngine:
        SubframeTemporalCompensationEngine =
        SubframeTemporalCompensationEngine(),
    private val actorDetailAllocationEngine:
        ActorDetailAllocationEngine =
        ActorDetailAllocationEngine(),
    private val actorFaceAssociationEngine:
        MultiActorFaceAssociationEngine =
        MultiActorFaceAssociationEngine(),
    private val actorHandAssociationEngine:
        MultiActorHandAssociationEngine =
        MultiActorHandAssociationEngine(),
    private val opticalEvidenceQualityEngine:
        OpticalEvidenceQualityEngine =
        OpticalEvidenceQualityEngine(),
    private val photometricMotionFieldEngine:
        SparsePhotometricMotionFieldEngine =
        SparsePhotometricMotionFieldEngine(),
    private val temporalPhotometricConsensusEngine:
        TemporalPhotometricConsensusEngine =
        TemporalPhotometricConsensusEngine(),
    private val motionOrchestrator: UniversalMotionInferenceOrchestrator =
        UniversalMotionInferenceOrchestrator(),
    private val trackLifecycleController:
        ProductionTrackLifecycleController =
        ProductionTrackLifecycleController(),
    private val faceEyeInferencePolicy: AdaptiveFaceEyeInferencePolicy =
        AdaptiveFaceEyeInferencePolicy(),
    private val personSegmentationEngine:
        PersonSegmentationEngine =
        PersonSegmentationEngine(context),
    private val actorSilhouetteEvidenceEngine:
        ActorSilhouetteEvidenceEngine =
        ActorSilhouetteEvidenceEngine(),
    private val contourFrameInterval: Long = 3L,
) : AutoCloseable {

    data class CalibratedCameraGeometry(
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val uncertainty: CalibrationUncertaintyProfile =
            CalibrationUncertaintyProfile(),
    )

    init {
        require(contourFrameInterval >= 1L) {
            "contourFrameInterval must be positive"
        }
    }

    private var frameCounter = 0L
    private var lastContourFrame: FaceContourFrame? = null
    private var lastContourTimestampNanos = Long.MIN_VALUE

    private var lastFaces: List<FaceFrame> = emptyList()
    private var lastFacesTimestampNanos = Long.MIN_VALUE

    private var lastFaceMeshes: List<FaceMeshGeometryEngine.Result> = emptyList()
    private var lastFaceMeshesTimestampNanos = Long.MIN_VALUE

    private var lastObjects: List<TrackedObject> = emptyList()
    private var lastObjectsTimestampNanos = Long.MIN_VALUE

    private var lastHands: HandTrackingFrame? = null
    private var lastHandsTimestampNanos = Long.MIN_VALUE

    private var lastPose: PoseDetectionEngine.Result? = null
    private var lastMultiActorPoseResult:
        MultiActorPoseLandmarkerEngine.Result? = null
    private var lastMotionQuality: MotionCaptureEngine.Quality =
        emptyMotionQuality()

    private var npuScale: Double = 1.0
    private var thermalScale: Double = 1.0
    private var memoryScale: Double = 1.0
    private var gpuScale: Double = 1.0
    private var cpuScale: Double = 1.0
    private val multiPersonCapacityGovernor =
        MultiPersonRuntimeCapacityGovernor()
    private val crowdMotionEvidenceEngine =
        ProductionCrowdMotionEvidenceEngine()
    private val frontierActorDetailScheduler =
        FrontierActorDetailScheduler()
    private var customization =
        MotionCaptureCustomization.production()
    @Volatile
    private var calibratedCameraGeometry: Map<String, CalibratedCameraGeometry> = emptyMap()

    private val denseActorSubjectSlots =
        LinkedHashMap<Long, DenseActorSlot>()

    private data class DenseActorSlot(
        val slot: Int,
        var lastSeenTimestampNanos: Long,
    )

    data class Result(
        val pose: PoseDetectionEngine.Result,
        val faces: List<FaceFrame>,
        val eyes: List<EyeTrackingFrame>,
        val hands: HandTrackingFrame,
        val objects: List<TrackedObject>,
        val tracks: List<SubjectTrackManager.Track>,
        val faceTracks: List<FaceTrackManager.Track>,
        val faceContour: FaceContourFrame?,
        val faceMeshes: List<FaceMeshGeometryEngine.Result> = emptyList(),
        val motion: MotionCaptureEngine.Quality,
        val photometricMotion:
            SparsePhotometricMotionFieldEngine.Result? = null,
        val temporalPhotometricConsensus:
            TemporalPhotometricConsensusEngine.Result? = null,
        val subframeTiming:
            SubframeTemporalCompensationEngine.Result? = null,
        val motionFusion: AdvancedMotionFusionEngine.Assessment,
        val subjectLock: ProductionSubjectLockEngine.Decision,
        val runtimeDecision: ProductionPerceptionRuntimeGovernor.Decision,
        val stageFreshness: Map<AdaptivePerceptionCoordinator.Stage, Boolean>,
        val inferencePlan: AdaptivePerceptionCoordinator.Plan,
        val fusion: HighEndPerceptionFusionEngine.Result? = null,
        val denseLandmarkEvidence:
            DenseLandmarkEvidenceFusionEngine.FrameResult? = null,
        val denseContactEvidence:
            DenseContactEvidenceAdapter.Result? = null,
        val subpixelLandmarkRefinement:
            SubpixelLandmarkRefinementEngine.FrameResult? = null,
        val highOrderMotion:
            HighOrderMotionRefinementEngine.Assessment? = null,
        val faceEyeFusion: FaceEyeFusionEngine.Result? = null,
        val stableSubjectIdentities: SubjectReidentificationEngine.Assignment? = null,
        val motionOrchestration: MotionInferenceDecision? = null,
        val multiActorPose:
            MultiActorPoseLandmarkerEngine.Result? = null,
        val actorTracks:
            List<MultiActorPoseAssociationEngine.Result> = emptyList(),
        val crowdMotionEvidence:
            ProductionCrowdMotionEvidenceEngine.FrameResult? = null,
        val frontierActorAllocation:
            FrontierActorDetailScheduler.Result? = null,
        val detailAllocation:
            ActorDetailAllocationEngine.Result? = null,
        val multiPersonCapacityAllocation:
            MultiPersonRuntimeCapacityGovernor.Allocation? = null,
        val personSegmentation:
            PersonSegmentationEngine.Result? = null,
        val actorSilhouetteEvidence:
            Map<Long, ActorSilhouetteEvidenceEngine.Evidence> =
            emptyMap(),
        val opticalEvidence:
            OpticalEvidenceQualityEngine.Result? = null,
        val faceActorAssociation:
            MultiActorFaceAssociationEngine.Result? = null,
        val handActorAssociation:
            MultiActorHandAssociationEngine.Result? = null,
        val trackLifecycle:
            ProductionTrackLifecycleController.Result =
            ProductionTrackLifecycleController.Result(
                states = emptyMap(),
                continuityScores = emptyMap(),
                predictedTrackIds = emptySet(),
                recoveringTrackIds = emptySet(),
                confirmedTrackIds = emptySet(),
                dominantTrackId = null,
                dominantState =
                    ProductionTrackLifecycleController.State.NEW,
                continuityScore = 0.0,
                warnings = emptyList(),
            ),
    ) {
        val timestampNanos: Long
            get() = pose.timestampNanos

        val subjectCount: Int
            get() = maxOf(
                faces.size,
                hands.trackedHands,
                objects.count { it.subjectType.name == "HUMAN" },
                objects.count { it.subjectType.name == "ANIMAL" },
                actorTracks.size,
            )

        val handConfidence: Double
            get() = hands.meanConfidence
    }

    fun configureCustomization(
        customization: MotionCaptureCustomization,
    ) {
        this.customization = customization
        photometricMotionFieldEngine.configureExposureReliability(
            softLimit =
                customization.photometricExposureShiftSoftLimit,
            hardLimit =
                customization.photometricExposureShiftHardLimit,
            floor =
                customization.photometricExposureReliabilityFloor,
        )

        val targetActors =
            customization.multiPersonCapacityPolicy.targetSubjects
                .coerceIn(1, customization.maxSubjects.coerceIn(1, 64))
        val hardActorCeiling =
            customization.multiPersonCapacityPolicy.hardSubjectCeiling
                .coerceIn(targetActors, 64)
        multiPersonCapacityGovernor.configure(
            customization.multiPersonCapacityPolicy.copy(
                targetSubjects = targetActors,
                hardSubjectCeiling = hardActorCeiling,
            ),
        )
        frontierActorDetailScheduler.configure(
            FrontierActorDetailScheduler.Config(
                targetActors = targetActors,
                hardActorCeiling = hardActorCeiling,
                frameBudgetUnits =
                    (
                        customization.targetFps.coerceIn(15, 240) *
                            (
                                1.0 +
                                    customization.surfaceDetailPriority *
                                        0.75
                                )
                        ).coerceAtLeast(128.0),
                minimumBodyShare =
                    (
                        0.34 +
                            customization.captureIntegrityStrictness * 0.18
                        ).coerceIn(0.34, 0.60),
                starvationAgeFrames =
                    (
                        customization.multiPersonCapacityPolicy.identityRetentionFrames /
                            12
                    ).coerceIn(4, 32).toLong(),
                maximumDetailCadenceDivisor =
                    (
                        1 +
                            (
                                8 *
                                    (
                                        1.0 -
                                            customization.faceEyeFusionStrictness * 0.45
                                    )
                                ).toInt()
                    ).coerceIn(2, 8),
                minimumBodyAllocationFraction =
                    (
                        0.50 +
                            customization.captureIntegrityStrictness * 0.22
                        ).coerceIn(0.50, 0.82),
            ),
        )

        subjectLockEngine.configureMaximumSubjects(
            customization.maxSubjects,
        )
        multiActorAssociation.configure(
            maxActors = customization.maxSubjects,
            minimumMatchScore =
                (
                    0.22 -
                        customization.actorIdentityStrictness * 0.10
                    ).coerceIn(0.08, 0.22),
            microMotionRetention =
                customization.actorMicroMotionRetention.coerceIn(0.0, 1.0),
            maxMissedFrames =
                (
                    customization.multiPersonCapacityPolicy.identityRetentionFrames / 2
                )
                    .coerceIn(45, 180),
            crowdedSceneOverlapThreshold =
                customization.crowdedSceneOverlapThreshold,
            crowdedSceneMatchScoreBoost =
                customization.crowdedSceneMatchScoreBoost,
        )
        faceEyeFusionEngine.configureCustomization(
            headPoseStrictness = customization.headPoseStrictness,
            gazePriority = customization.gazePriority,
            eyeTemporalStrictness = customization.eyeTemporalStrictness,
            semanticMotionSensitivity =
                customization.semanticMotionSensitivity,
        )
    }

    fun configureMotionTuning(tuning: MotionCaptureTuning) {
        motion.configure(tuning)
    }

    fun motionTuning(): MotionCaptureTuning = motion.tuning()

    /**
     * Clears temporal state between independent capture sequences without closing model instances.
     * This is required before offline re-entry so the first decoded frame cannot inherit identities,
     * velocity priors, face caches or motion history from the preceding live take.
     */
    fun resetForNewSequence() {
        frameCounter = 0L
        lastContourFrame = null
        lastContourTimestampNanos = Long.MIN_VALUE
        lastFaces = emptyList()
        lastFacesTimestampNanos = Long.MIN_VALUE
        lastFaceMeshes = emptyList()
        lastFaceMeshesTimestampNanos = Long.MIN_VALUE
        lastObjects = emptyList()
        lastObjectsTimestampNanos = Long.MIN_VALUE
        lastHands = null
        lastHandsTimestampNanos = Long.MIN_VALUE
        lastPose = null
        lastMultiActorPoseResult = null
        lastEyeResults = emptyList()
        lastEyeResultsTimestampNanos = Long.MIN_VALUE
        denseActorSubjectSlots.clear()

        subjectTracks.clear()
        multiActorAssociation.clear()
        multiActorPose.resetForNewSequence()
        faceTracks.clear()
        faceStabilizer.clear()
        motion.reset()
        motionFusion.reset()
        faceEyeFusionEngine.reset()
        subjectReidentification.reset()
        motionOrchestrator.reset()
        trackLifecycleController.reset()
        subjectLockEngine.reset()
        handTrackingReset()
        coordinator.reset()
        personSegmentationEngine.reset()
        lastMotionQuality = emptyMotionQuality()
    }

    private fun emptyMultiActorPoseResult(
        timestampNanos: Long,
        rotationDegrees: Int,
        inputWidth: Int,
        inputHeight: Int,
        warning: String,
    ): MultiActorPoseLandmarkerEngine.Result =
        MultiActorPoseLandmarkerEngine.Result(
            timestampNanos = timestampNanos,
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            rotationDegrees = rotationDegrees,
            observations = emptyList(),
            inferenceTimeMillis = 0L,
            detectorPoseCount = 0,
            landmarkerTimestampMillis =
                timestampNanos / 1_000_000L,
            timestampAdjusted = false,
            temporalTrackingUsed = false,
            warnings = listOf(warning),
        )

    private fun handTrackingReset() {
        hands.reset()
    }

    @Synchronized
    fun configureCalibratedCameraGeometry(
        geometry: Map<String, CalibratedCameraGeometry>,
    ) {
        calibratedCameraGeometry =
            geometry
                .filterKeys { it.isNotBlank() }
                .toMap()
    }

    fun updateSystemLoad(
        thermalScale: Double,
        memoryScale: Double,
        gpuScale: Double = 1.0,
        inferenceScale: Double = 1.0,
        npuScale: Double = 1.0,
    ) {
        this.thermalScale = thermalScale.coerceIn(0.10, 1.0)
        this.memoryScale = memoryScale.coerceIn(0.10, 1.0)
        this.gpuScale = gpuScale.coerceIn(0.10, 1.0)
        this.npuScale = npuScale.coerceIn(0.10, 1.0)
        this.cpuScale = inferenceScale.coerceIn(0.10, 1.0)

        coordinator.updateSystemLoad(
            thermalScale = this.thermalScale,
            memoryScale = this.memoryScale,
            gpuScale = this.gpuScale,
            inferenceScale = inferenceScale,
        )
    }

    fun diagnostics(): AdaptivePerceptionCoordinator.Telemetry =
        coordinator.telemetry()

    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
        cameraId: String = "0",
        calibrationCameraId: String = cameraId,
        motionContext: MotionCaptureEngine.MotionContext =
            MotionCaptureEngine.MotionContext(),
        rollingShutterSkewNanos: Long = 0L,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
        // Missing timing metadata is unknown evidence, not perfect synchronization.
        captureTimingConfidence: Double = 0.0,
        lightingConfidence: Double = 1.0,
        rollingShutterDirectionSign: Int = 1,
    ): Task<Result> {
        require(
            rollingShutterDirectionSign == 1 ||
                rollingShutterDirectionSign == -1,
        )

        frameCounter += 1L

        coordinator.configureSegmentation(
            enabled =
                customization.personSegmentationEnabled,
            cadenceFrames =
                (
                    customization.personSegmentationCadenceFrames /
                        customization.personSegmentationPriority
                            .coerceIn(0.20, 1.0)
                    ).toDouble()
                    .coerceIn(1.0, 120.0)
                    .toInt(),
        )

        val plan = coordinator.beginFrame(
            timestampNanos = timestampNanos,
            motionQuality = lastMotionQuality,
        )

        val segmentationShouldRun =
            plan.shouldProcess(
                AdaptivePerceptionCoordinator.Stage.SEGMENTATION,
            ) ||
                personSegmentationEngine.latest()?.let {
                    timestampNanos -
                        it.timestampMs * 1_000_000L
                }?.let { age ->
                    age !in 0L..plan.maxAge(
                        AdaptivePerceptionCoordinator.Stage.SEGMENTATION,
                    )
                } != false

        if (segmentationShouldRun) {
            val segmentationStart = System.nanoTime()
            runCatching {
                personSegmentationEngine.submit(
                    MediaImageBuilder(image)
                        .build(),
                    timestampNanos / 1_000_000L,
                )
            }
            coordinator.recordStageDuration(
                AdaptivePerceptionCoordinator.Stage.SEGMENTATION,
                (
                    System.nanoTime() -
                        segmentationStart
                ).coerceAtLeast(0L) / 1e6,
            )
        }

        val subframeTiming =
            runCatching {
                subframeTemporalCompensationEngine.compensate(
                    SubframeTemporalCompensationEngine.Input(
                        frameTimestampNanos = timestampNanos,
                        frameDurationNanos = frameDurationNanos,
                        exposureTimeNanos = exposureTimeNanos,
                        rollingShutterSkewNanos = rollingShutterSkewNanos,
                        captureTimingConfidence = captureTimingConfidence,
                        sensorConfidence = motionContext.sensorConfidence,
                        cameraMotionRisk = motionContext.cameraMotionRisk,
                        globalMotionConfidence =
                            motionContext.globalMotionConfidence,
                    ),
                )
            }.getOrNull()

        val opticalEvidence =
            runCatching {
                opticalEvidenceQualityEngine.evaluate(
                    image = image,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                    cameraMotionRisk = motionContext.cameraMotionRisk,
                    captureTimingConfidence =
                        captureTimingConfidence,
                    rollingShutterFraction =
                        if (frameDurationNanos > 0L) {
                            rollingShutterSkewNanos.toDouble() /
                                frameDurationNanos.toDouble()
                        } else {
                            0.0
                        },
                )
            }.getOrNull()

        val photometricMotion =
            runCatching {
                photometricMotionFieldEngine.process(image)
            }.getOrNull()

        val temporalPhotometricConsensus =
            photometricMotion
                ?.let {
                    runCatching {
                        temporalPhotometricConsensusEngine.fuse(it)
                    }.getOrNull()
                }

        val effectivePhotometricVectors =
            temporalPhotometricConsensus
                ?.takeIf { it.valid }
                ?.vectors
                ?.map {
                    SparsePhotometricMotionFieldEngine.FlowVector(
                        normalizedX = it.normalizedX,
                        normalizedY = it.normalizedY,
                        dxPixels = it.dxPixels,
                        dyPixels = it.dyPixels,
                        confidence = it.confidence,
                        residualEnergy = 0.0,
                        photometricError =
                            it.photometricError,
                        gradientAgreement =
                            it.gradientAgreement,
                        textureStrength =
                            it.textureStrength,
                        forwardBackwardConsistency =
                            it.forwardBackwardConsistency,
                        subpixelRefined =
                            it.subpixelRefinedRatio >= 0.50,
                        sourceState =
                            SparsePhotometricMotionFieldEngine.SourceState.OBSERVED,
                    )
                }
                ?: photometricMotion?.vectors
                ?: emptyList()

        val timingAdjustedContext =
            if (subframeTiming == null) {
                motionContext
            } else {
                motionContext.copy(
                    sensorConfidence =
                        (
                            motionContext.sensorConfidence *
                                subframeTiming.timingConfidence
                            ).coerceIn(0.0, 1.0),
                    rollingShutterRisk =
                        maxOf(
                            motionContext.rollingShutterRisk,
                            subframeTiming.rollingShutterRisk,
                        ).coerceIn(0.0, 1.0),
                    motionBlurRisk =
                        maxOf(
                            motionContext.motionBlurRisk,
                            subframeTiming.exposureBlurRisk,
                            opticalEvidence?.motionBlurRisk ?: 0.0,
                        ).coerceIn(0.0, 1.0),
                    globalMotionConfidence =
                        maxOf(
                            motionContext.globalMotionConfidence,
                            1.0 -
                                subframeTiming.timestampJitterRisk,
                        ).coerceIn(0.0, 1.0),
                )
            }

        val opticalAdjustedContext =
            if (opticalEvidence == null) {
                timingAdjustedContext
            } else {
                timingAdjustedContext.copy(
                    motionBlurRisk =
                        maxOf(
                            timingAdjustedContext.motionBlurRisk,
                            opticalEvidence.motionBlurRisk,
                        ).coerceIn(0.0, 1.0),
                    globalMotionConfidence =
                        maxOf(
                            timingAdjustedContext.globalMotionConfidence,
                            opticalEvidence.signalQuality,
                        ).coerceIn(0.0, 1.0),
                )
            }

        val effectiveMotionContext =
            if (photometricMotion?.valid == true) {
                opticalAdjustedContext.copy(
                    cameraMotionRisk =
                        maxOf(
                            opticalAdjustedContext.cameraMotionRisk,
                            photometricMotion.cameraMotionRisk,
                        ).coerceIn(0.0, 1.0),
                    globalMotionConfidence =
                        maxOf(
                            opticalAdjustedContext.globalMotionConfidence,
                            photometricMotion.globalMotionConfidence,
                        ).coerceIn(0.0, 1.0),
                )
            } else {
                opticalAdjustedContext
            }

        val poseStartNanos = System.nanoTime()
        val poseTask = pose.process(
            image = image,
            rotationDegrees = rotationDegrees,
            timestampNanos = timestampNanos,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
            captureTimingConfidence = captureTimingConfidence,
        ).continueWith { task ->
            coordinator.recordStageDuration(
                AdaptivePerceptionCoordinator.Stage.POSE,
                (System.nanoTime() - poseStartNanos).coerceAtLeast(0L) / 1e6,
            )
            if (task.isSuccessful) {
                task.getResult().also {
                    lastPose = it
                }
            } else {
                lastPose?.copy(
                    timestampNanos = timestampNanos,
                    confidence =
                        (lastPose?.confidence ?: 0f) * 0.20f,
                    staleEvidence = true,
                ) ?: PoseDetectionEngine.Result(
                    timestampNanos = timestampNanos,
                    imageWidth = image.width,
                    imageHeight = image.height,
                    landmarks = emptyList(),
                    confidence = 0f,
                    frameIndex = 0L,
                    rotationDegrees = rotationDegrees,
                    rollingShutterSkewNanos =
                        rollingShutterSkewNanos.coerceAtLeast(0L),
                    frameDurationNanos =
                        frameDurationNanos.coerceAtLeast(0L),
                    exposureTimeNanos =
                        exposureTimeNanos.coerceAtLeast(0L),
                    captureTimingConfidence = 0.0,
                    staleEvidence = true,
                )
            }
        }

        val multiActorPoseTask =
            multiActorPose.process(
                image = image,
                rotationDegrees = rotationDegrees,
                timestampNanos = timestampNanos,
            )
            .continueWith { task ->
                if (!task.isSuccessful) {
                    lastMultiActorPoseResult
                        ?: emptyMultiActorPoseResult(
                            timestampNanos = timestampNanos,
                            rotationDegrees = rotationDegrees,
                            inputWidth = image.width,
                            inputHeight = image.height,
                            warning =
                                "Multi-actor pose inference failed; continuing with bounded stale/empty evidence.",
                        )
                } else {
                    task.result
                }
            }

        val faceShouldRun =
            plan.shouldProcess(AdaptivePerceptionCoordinator.Stage.FACE) ||
                !isFresh(
                    lastFacesTimestampNanos,
                    timestampNanos,
                    plan.maxAge(AdaptivePerceptionCoordinator.Stage.FACE),
                )

        var faceEvidenceFresh = false

        val faceTask = if (faceShouldRun) {
            val start = System.nanoTime()
            face.process(
                image = image,
                rotationDegrees = rotationDegrees,
                timestampNanos = timestampNanos,
                captureTimingConfidence = captureTimingConfidence,
                rollingShutterSkewNanos = rollingShutterSkewNanos,
                frameDurationNanos = frameDurationNanos,
                exposureTimeNanos = exposureTimeNanos,
            ).continueWith { task ->
                coordinator.recordStageDuration(
                    AdaptivePerceptionCoordinator.Stage.FACE,
                    (System.nanoTime() - start).coerceAtLeast(0L) / 1e6,
                )
                if (!task.isSuccessful) {
                    /*
                     * Face failure is isolated from the multimodal graph. The last successful
                     * timestamp remains authoritative, so a detector outage cannot manufacture
                     * fresh face evidence or abort pose/hand/body reconstruction for the frame.
                     */
                    faceEvidenceFresh = false
                    return@continueWith lastFaces
                }
                val stabilized = faceStabilizer.refine(task.result)
                lastFaces = stabilized
                lastFacesTimestampNanos = timestampNanos
                faceEvidenceFresh = true
                stabilized
            }
        } else {
            faceEvidenceFresh =
                isFresh(
                    lastFacesTimestampNanos,
                    timestampNanos,
                    plan.maxAge(
                        AdaptivePerceptionCoordinator.Stage.FACE,
                    ),
                )
            Tasks.forResult(lastFaces)
        }

        val objectShouldRun =
            plan.shouldProcess(AdaptivePerceptionCoordinator.Stage.OBJECT) ||
                !isFresh(
                    lastObjectsTimestampNanos,
                    timestampNanos,
                    plan.maxAge(AdaptivePerceptionCoordinator.Stage.OBJECT),
                )

        val objectTask = if (objectShouldRun) {
            val start = System.nanoTime()
            objects.process(
                image = image,
                rotationDegrees = rotationDegrees,
                timestampNanos = timestampNanos,
            ).continueWith { task ->
                coordinator.recordStageDuration(
                    AdaptivePerceptionCoordinator.Stage.OBJECT,
                    (System.nanoTime() - start).coerceAtLeast(0L) / 1e6,
                )
                if (!task.isSuccessful) {
                    return@continueWith lastObjects
                }
                lastObjects = task.result
                lastObjectsTimestampNanos = timestampNanos
                task.result
            }
        } else {
            Tasks.forResult(lastObjects)
        }

        val handShouldRun =
            plan.shouldProcess(AdaptivePerceptionCoordinator.Stage.HAND) ||
                !isFresh(
                    lastHandsTimestampNanos,
                    timestampNanos,
                    plan.maxAge(AdaptivePerceptionCoordinator.Stage.HAND),
                )

        val handTask = if (handShouldRun) {
            val start = System.nanoTime()
            hands.process(
                image = image,
                rotationDegrees = rotationDegrees,
                timestampNanos = timestampNanos,
                captureTimingConfidence =
                    captureTimingConfidence,
                rollingShutterSkewNanos =
                    rollingShutterSkewNanos,
                frameDurationNanos =
                    frameDurationNanos,
                exposureTimeNanos =
                    exposureTimeNanos,
            ).continueWith { task ->
                coordinator.recordStageDuration(
                    AdaptivePerceptionCoordinator.Stage.HAND,
                    (System.nanoTime() - start).coerceAtLeast(0L) / 1e6,
                )
                if (!task.isSuccessful) {
                    return@continueWith (
                        lastHands
                            ?: emptyHandFrame(timestampNanos)
                    )
                }
                lastHands = task.result
                lastHandsTimestampNanos = timestampNanos
                task.result
            }
        } else {
            Tasks.forResult(
                lastHands ?: emptyHandFrame(timestampNanos),
            )
        }

        val faceMeshTask =
            faceTask.continueWithTask { faceResultTask ->
                val facesForMesh =
                    if (faceResultTask.isSuccessful) {
                        faceResultTask.result
                    } else {
                        lastFaces
                    }

                val policyDecision =
                    faceEyeInferencePolicy.decide(
                        faceEyePolicyEvidence(
                            image = image,
                            faces = facesForMesh,
                            eyeResults = lastEyeResults,
                            timestampNanos = timestampNanos,
                            lightingConfidence = lightingConfidence,
                        ),
                    )

                val staleFaceMesh =
                    !isFresh(
                        lastFaceMeshesTimestampNanos,
                        timestampNanos,
                        plan.maxAge(
                            AdaptivePerceptionCoordinator.Stage.FACE,
                        ),
                    )

                val shouldRunFaceMesh =
                    facesForMesh.isNotEmpty() &&
                        (
                            policyDecision.runFaceMesh ||
                                staleFaceMesh &&
                                    policyDecision.faceMeshPriority >= 0.34
                            )

                if (!shouldRunFaceMesh) {
                    return@continueWithTask Tasks.forResult(
                        lastFaceMeshes,
                    )
                }

                val start = System.nanoTime()
                faceMesh.process(
                    image = image,
                    rotationDegrees = rotationDegrees,
                    timestampNanos = timestampNanos,
                    captureTimingConfidence = captureTimingConfidence,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                    rollingShutterDirectionSign = rollingShutterDirectionSign,
                ).continueWith { task ->
                    coordinator.recordStageDuration(
                        AdaptivePerceptionCoordinator.Stage.FACE,
                        (System.nanoTime() - start).coerceAtLeast(0L) / 1e6,
                    )
                    val result =
                        if (task.isSuccessful) {
                            task.result
                        } else {
                            lastFaceMeshes
                        }
                    if (result.isNotEmpty()) {
                        lastFaceMeshes = result
                        lastFaceMeshesTimestampNanos = timestampNanos
                    }
                    result
                }
            }

        val contourShouldRun =
            (
                plan.shouldProcess(
                    AdaptivePerceptionCoordinator.Stage.CONTOUR,
                ) &&
                    (
                        contourFrameInterval <= 1L ||
                            frameCounter % contourFrameInterval == 0L
                        )
                ) ||
                !isFresh(
                    lastContourTimestampNanos,
                    timestampNanos,
                    plan.maxAge(AdaptivePerceptionCoordinator.Stage.CONTOUR),
                )

        val contourTask =
            if (contourShouldRun) {
                val start = System.nanoTime()
                contours.process(
                    image = image,
                    rotationDegrees = rotationDegrees,
                    timestampNanos = timestampNanos,
                    captureTimingConfidence = captureTimingConfidence,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                ).continueWith { task ->
                    coordinator.recordStageDuration(
                        AdaptivePerceptionCoordinator.Stage.CONTOUR,
                        (System.nanoTime() - start).coerceAtLeast(0L) / 1e6,
                    )
                    if (!task.isSuccessful) {
                        return@continueWith lastContourFrame
                    }
                    task.result
                }
            } else {
                Tasks.forResult(lastContourFrame)
            }

        return Tasks.whenAllSuccess<Any>(
            poseTask,
            faceTask,
            objectTask,
            handTask,
            contourTask,
            faceMeshTask,
            multiActorPoseTask,
        ).continueWith { all ->
            if (!all.isSuccessful) {
                throw all.exception
                    ?: IllegalStateException("Perception graph failed")
            }

            val poseResult = all.result[0] as PoseDetectionEngine.Result
            val rawFaces = all.result[1] as List<FaceFrame>
            val trackedObjects = all.result[2] as List<TrackedObject>
            val handFrame = all.result[3] as HandTrackingFrame
            val freshContour = all.result[4] as FaceContourFrame?
            val effectiveFaceMeshes =
                associateFaceMeshes(
                    meshes = all.result[5] as List<FaceMeshGeometryEngine.Result>,
                    faces = rawFaces,
                )
            val rawMultiActorPoseResult =
                (all.result[6] as MultiActorPoseLandmarkerEngine.Result)
                    .also {
                        if (it.observations.isNotEmpty()) {
                            lastMultiActorPoseResult = it
                        }
                    }

            val personSegmentation =
                personSegmentationEngine.latest()
                    ?.takeIf { segmentation ->
                        val ageNanos =
                            timestampNanos -
                                segmentation.timestampMs *
                                1_000_000L
                        ageNanos in 0L..plan.maxAge(
                            AdaptivePerceptionCoordinator.Stage.SEGMENTATION,
                        )
                    }

            val subjectDemands =
                rawMultiActorPoseResult.observations.map { observation ->
                    MultiPersonRuntimeCapacityGovernor.SubjectDemand(
                        trackId =
                            observation.detectorTrackingId
                                .toLong()
                                .coerceAtLeast(0L),
                        confidence = observation.confidence,
                        visibleRatio =
                            observation.joints
                                .count { it.confidence >= 0.35 }
                                .toDouble() /
                                observation.joints.size.coerceAtLeast(1).toDouble(),
                        motionEnergy = 0.5,
                        faceRequested = customization.faceDetail >= 0.55,
                        eyeRequested = customization.eyeDetail >= 0.55,
                        handsRequested = customization.handDetail >= 0.55,
                        surfaceRequested = customization.denseMeshDetail >= 0.55,
                        interactionPriority =
                            if (observation.subjectType == SubjectType.HUMAN) 1.0 else 0.35,
                        identityPriority = customization.actorIdentityStrictness,
                    )
                }

            val capacityAllocation =
                multiPersonCapacityGovernor.allocate(
                    demands = subjectDemands,
                    headroom =
                        MultiPersonRuntimeCapacityGovernor.RuntimeHeadroom(
                            cpuHeadroom = cpuScale,
                            gpuHeadroom = gpuScale,
                            npuHeadroom = npuScale,
                            thermalHeadroom = thermalScale,
                            memoryHeadroom = memoryScale,
                        ),
                    captureFps =
                        customization.targetFps.coerceIn(15, 240),
                )

            val admittedIds =
                capacityAllocation.admittedSubjectIds.toHashSet()
            val admittedObservations =
                rawMultiActorPoseResult.observations.filter {
                    it.detectorTrackingId
                        .toLong()
                        .coerceAtLeast(0L) in admittedIds
                }

            val multiActorPoseResult =
                rawMultiActorPoseResult.copy(
                    observations = admittedObservations,
                    warnings =
                        (
                            rawMultiActorPoseResult.warnings +
                                capacityAllocation.warnings
                            )
                            .distinct()
                            .takeLast(24),
                )

            val actorTracks =
                multiActorAssociation.update(
                    timestampNanos = timestampNanos,
                    observations = multiActorPoseResult.observations,
                )

            val segmentationEvidence =
                personSegmentation?.let { segmentation ->
                    (
                        segmentation.modelQuality * 0.66 +
                            segmentation.boundaryStability * 0.34
                        ).coerceIn(0.0, 1.0)
                } ?: 0.0

            val segmentationCoverage =
                personSegmentation?.foregroundCoverage
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0

            val actorSilhouetteEvidence =
                actorSilhouetteEvidenceEngine.evaluate(
                    tracks = actorTracks,
                    segmentation = personSegmentation,
                )

            val crowdMotionEvidence =
                runCatching {
                    crowdMotionEvidenceEngine.evaluate(
                        ProductionCrowdMotionEvidenceEngine.Input(
                            timestampNanos = timestampNanos,
                            actors = actorTracks,
                            cameraMotionRisk =
                                effectiveMotionContext.cameraMotionRisk
                                    .coerceIn(0.0, 1.0),
                            timingConfidence =
                                effectiveMotionContext.sensorConfidence
                                    .coerceIn(0.0, 1.0),
                            runtimeHeadroom =
                                minOf(
                                    thermalScale,
                                    memoryScale,
                                    gpuScale,
                                    npuScale,
                                    cpuScale,
                                ).coerceIn(0.0, 1.0),
                            segmentationEvidence = segmentationEvidence,
                            segmentationCoverage = segmentationCoverage,
                            actorSilhouetteEvidence =
                                actorSilhouetteEvidence,
                        ),
                    )
                }.getOrNull()

            val refinedPoseBase =
                if (poseResult.staleEvidence) {
                    MotionCaptureEngine.Refined(
                        result = poseResult,
                        quality =
                            lastMotionQuality.copy(
                                timestampNanos = timestampNanos,
                                motionScore = 0.0,
                                trackedLandmarks = 0,
                                predictedLandmarks =
                                    poseResult.landmarks.size,
                                rejectedLandmarks =
                                    poseResult.landmarks.size,
                                meanConfidence =
                                    (
                                        lastMotionQuality.meanConfidence *
                                            0.20
                                        ).coerceIn(0.0, 1.0),
                                occlusionRatio = 1.0,
                                temporalStability =
                                    (
                                        lastMotionQuality.temporalStability *
                                            0.25
                                        ).coerceIn(0.0, 1.0),
                                dynamicsConfidence =
                                    (
                                        lastMotionQuality.dynamicsConfidence *
                                            0.25
                                        ).coerceIn(0.0, 1.0),
                                rejectedMeasurementRatio = 1.0,
                            ),
                        landmarkQuality = emptyList(),
                    )
                } else {
                    motion.refine(
                        input = poseResult,
                        context = effectiveMotionContext,
                    )
                }
            val highOrderMotion =
                highOrderMotionRefinementEngine.refine(
                    quality = refinedPoseBase.quality,
                    context = motionContext,
                )
            val refinedPose =
                refinedPoseBase.copy(
                    quality = highOrderMotion.refinedQuality,
                )
            lastPose = refinedPose.result
            lastMotionQuality = refinedPose.quality

            val facesFresh =
                faceEvidenceFresh &&
                    isFresh(
                        lastFacesTimestampNanos,
                        timestampNanos,
                        plan.maxAge(
                            AdaptivePerceptionCoordinator.Stage.FACE,
                        ),
                    )
            val objectsFresh = objectShouldRun
            val handsFresh = handShouldRun

            val effectiveFaces = if (facesFresh) {
                rawFaces
            } else {
                lastFaces
            }

            val effectiveObjects = if (objectsFresh) {
                trackedObjects
            } else {
                lastObjects
            }

            val effectiveHands = if (handsFresh) {
                handFrame
            } else {
                lastHands ?: handFrame
            }

            if (freshContour != null) {
                val accepted = effectiveFaces.any { faceFrame ->
                    val trackingMatch =
                        freshContour.trackingId != null &&
                            faceFrame.trackingId != null &&
                            freshContour.trackingId ==
                            faceFrame.trackingId
                    trackingMatch ||
                        contourIou(
                            contour = freshContour,
                            face = faceFrame,
                        ) >= 0.35
                }

                if (accepted || effectiveFaces.isEmpty()) {
                    lastContourFrame = freshContour
                    lastContourTimestampNanos = timestampNanos
                }
            }

            val contourForFrame = lastContourFrame?.takeIf {
                isFresh(
                    lastContourTimestampNanos,
                    timestampNanos,
                    plan.maxAge(AdaptivePerceptionCoordinator.Stage.CONTOUR),
                )
            }

            val faceEyeDecision =
                faceEyeInferencePolicy.decide(
                    faceEyePolicyEvidence(
                        image = image,
                        faces = effectiveFaces,
                        eyeResults = lastEyeResults,
                        timestampNanos = timestampNanos,
                        lightingConfidence = lightingConfidence,
                    ),
                )

            val eyeDecision =
                plan.shouldProcess(
                    AdaptivePerceptionCoordinator.Stage.EYE,
                ) ||
                    faceEyeDecision.runEyes ||
                    !isFresh(
                        lastEyeResultsTimestampNanos,
                        timestampNanos,
                        plan.maxAge(
                            AdaptivePerceptionCoordinator.Stage.EYE,
                        ),
                    )

            val eyeFrames = buildEyeFrames(
                image = image,
                rotationDegrees = rotationDegrees,
                faces = effectiveFaces,
                plan = plan,
                forceProcess = eyeDecision,
                timestampNanos = timestampNanos,
                captureTimingConfidence = captureTimingConfidence,
                rollingShutterSkewNanos = rollingShutterSkewNanos,
                frameDurationNanos = frameDurationNanos,
                exposureTimeNanos = exposureTimeNanos,
            )

            val tracks = if (objectsFresh) {
                subjectTracks.update(
                    effectiveObjects,
                    timestampNanos,
                )
            } else {
                subjectTracks.snapshot()
            }

            val trackLifecycle =
                trackLifecycleController.update(
                    timestampNanos = timestampNanos,
                    tracks = tracks.map { track ->
                        ProductionTrackLifecycleController.TrackEvidence(
                            trackId = track.id,
                            timestampNanos =
                                track.lastTimestampNanos
                                    .coerceAtMost(timestampNanos),
                            confidence = track.confidence,
                            missedFrames = track.missedFrames,
                            ageFrames = track.ageFrames,
                            temporalStability =
                                refinedPose.quality.temporalStability,
                            velocityMagnitude =
                                kotlin.math.hypot(
                                    track.velocityX.toDouble(),
                                    track.velocityY.toDouble(),
                                ),
                        )
                    },
                    evidenceFresh = objectsFresh,
                )

            val stableSubjectIdentities =
                subjectReidentification.update(
                    tracks = tracks,
                    timestampNanos = timestampNanos,
                )

            val detailAllocation =
                actorDetailAllocationEngine.allocate(
                    evidence =
                        actorTracks.map { actor ->
                            ActorDetailAllocationEngine.ActorEvidence(
                                actorId = actor.actorId,
                                confidence = actor.confidence.coerceIn(0.0, 1.0),
                                visibleRatio =
                                    actor.joints
                                        .count { it.confidence >= 0.35 }
                                        .toDouble() /
                                        actor.joints.size.coerceAtLeast(1).toDouble(),
                                motionEnergy =
                                    (
                                        actor.microMotion.preservedSignal * 0.42 +
                                            (
                                                crowdMotionEvidence
                                                    ?.actors
                                                    ?.firstOrNull {
                                                        it.actorId == actor.actorId
                                                    }
                                                    ?.cameraCompensatedMotionEnergy
                                                    ?: actor.microMotion.preservedSignal
                                            ) * 0.58
                                        )
                                        .coerceIn(0.0, 1.0),
                                interactionPriority =
                                    (
                                        (
                                            if (actor.missedFrames > 0) 0.85 else 0.55
                                        ) * 0.58 +
                                            (
                                                crowdMotionEvidence
                                                    ?.actors
                                                    ?.firstOrNull {
                                                        it.actorId == actor.actorId
                                                    }
                                                    ?.crowdPressure
                                                    ?: 0.0
                                            ) * 0.42
                                        )
                                        .coerceIn(0.0, 1.0),
                                identityContinuity =
                                    (
                                        trackLifecycle.continuityScores[
                                            actor.actorId
                                        ]
                                            ?.coerceIn(0.0, 1.0)
                                            ?: 0.70
                                    ) *
                                    (
                                        1.0 -
                                            (
                                                crowdMotionEvidence
                                                    ?.actors
                                                    ?.firstOrNull {
                                                        it.actorId == actor.actorId
                                                    }
                                                    ?.identityRisk
                                                    ?: 0.0
                                            ) * 0.32
                                    ).coerceIn(0.0, 1.0),
                                occlusionRisk =
                                    (
                                        actor.missedFrames /
                                            45.0
                                        )
                                        .coerceIn(0.0, 1.0),
                                screenAreaRatio =
                                    actor.bounds.area.coerceIn(0.0, 1.0),
                            )
                        },
                    customization = customization,
                )

            val faceActorAssociation =
                actorFaceAssociationEngine.associate(
                    actors =
                        actorTracks.map { actor ->
                            MultiActorFaceAssociationEngine.Actor(
                                actorId = actor.actorId,
                                center = actor.center,
                                boundsLeft = actor.bounds.left,
                                boundsTop = actor.bounds.top,
                                boundsRight = actor.bounds.right,
                                boundsBottom = actor.bounds.bottom,
                                confidence = actor.confidence,
                            )
                        },
                    faces = effectiveFaces,
                )

            val handActorAssociation =
                actorHandAssociationEngine.associate(
                    actors =
                        actorTracks.map { actor ->
                            MultiActorHandAssociationEngine.Actor(
                                actorId = actor.actorId,
                                center = actor.center,
                                boundsLeft = actor.bounds.left,
                                boundsTop = actor.bounds.top,
                                boundsRight = actor.bounds.right,
                                boundsBottom = actor.bounds.bottom,
                                velocityXNormalizedPerSecond =
                                    actor.velocityXNormalizedPerSecond,
                                velocityYNormalizedPerSecond =
                                    actor.velocityYNormalizedPerSecond,
                                confidence = actor.confidence,
                            )
                        },
                    hands = effectiveHands.hands,
                )
            val faceTrackSnapshot = if (facesFresh) {
                faceTracks.update(effectiveFaces)
            } else {
                faceTracks.snapshot()
            }

            val assessment = motionFusion.assess(
                timestampNanos = timestampNanos,
                faces = effectiveFaces,
                eyes = eyeFrames,
                hands = effectiveHands,
                objects = effectiveObjects,
                subjectTracks = tracks,
                motion = refinedPose.quality,
                frameWidth = image.width,
                frameHeight = image.height,
            )

            val subjectLock = subjectLockEngine.update(
                timestampNanos = timestampNanos,
                observations = buildSubjectLockObservations(
                    objects = effectiveObjects,
                    faces = effectiveFaces,
                    eyes = eyeFrames,
                ),
                cameraMotion =
                    ProductionSubjectLockEngine.CameraMotion(
                        vx = assessment.semanticMotion.commonModeVelocityX,
                    vy = assessment.semanticMotion.commonModeVelocityY,
                    confidence = assessment.semanticMotion.cameraMotionRisk,
                    ),
            )


            val faceEyeFusion = effectiveFaces
                .firstOrNull()
                ?.let { faceFrame ->
                    val eyeFrame = eyeFrames.firstOrNull {
                        it.trackingId == faceFrame.trackingId
                    } ?: eyeFrames.firstOrNull()

                    runCatching {
                        faceEyeFusionEngine.fuse(
                            FaceEyeFusionEngine.Input(
                                timestampNanos = timestampNanos,
                                face = faceFrame,
                                eyes = eyeFrame,
                                captureTimingConfidence =
                                    captureTimingConfidence,
                                rollingShutterSkewNanos =
                                    rollingShutterSkewNanos,
                                frameDurationNanos =
                                    frameDurationNanos,
                                exposureTimeNanos =
                                    exposureTimeNanos,
                            ),
                        )
                    }.getOrNull()
                }

            val faceEvidence = effectiveFaces
                .map { frame ->
                    if (frame.points.isEmpty()) 0.0
                    else frame.points.map {
                        it.confidence.coerceIn(0.0, 1.0)
                    }.average()
                }
                .ifEmpty { listOf(0.0) }
                .average()

            val eyeEvidence = eyeFrames
                .map {
                    it.fixationConfidence.coerceIn(0.0, 1.0)
                }
                .ifEmpty { listOf(0.0) }
                .average()

            val fusion = highEndFusionEngine.fuse(
                HighEndPerceptionFusionEngine.Input(
                    timestampNanos = timestampNanos,
                    body = HighEndPerceptionFusionEngine.BodyEvidence(
                        confidence = refinedPose.quality.meanConfidence,
                        visibleNodes = refinedPose.quality.trackedLandmarks,
                        totalNodes = (
                            refinedPose.quality.trackedLandmarks +
                                refinedPose.quality.predictedLandmarks
                        ).coerceAtLeast(1),
                        velocityMetersPerSecond = com.nivukx.nox.core.model.Vec3(
                            refinedPose.quality.meanSpeedNormalizedPerSecond,
                            0.0,
                            0.0,
                        ),
                        accelerationMetersPerSecond2 = com.nivukx.nox.core.model.Vec3(
                            refinedPose.quality.meanAccelerationNormalizedPerSecond2,
                            0.0,
                            0.0,
                        ),
                        source = PerceptionSource.ML_KIT,
                    ),
                    depth = null,
                    topology = HighEndPerceptionFusionEngine.TopologyEvidence(
                        confidence = assessment.semanticMotion.confidence,
                        stability = refinedPose.quality.temporalStability,
                        edgeConstraintResidual = refinedPose.quality.jitterNormalized,
                        nodeCount = refinedPose.quality.trackedLandmarks,
                    ),
                    temporal = HighEndPerceptionFusionEngine.TemporalEvidence(
                        stability = refinedPose.quality.temporalStability,
                        innovation = refinedPose.quality.meanInnovationNormalized,
                        velocityContinuity = (
                            1.0 - refinedPose.quality.meanInnovationNormalized
                        ).coerceIn(0.0, 1.0),
                    ),
                    calibration = HighEndPerceptionFusionEngine.CalibrationEvidence(
                        confidence = subjectLock.globalConfidence,
                        reprojectionErrorPx = 0.0,
                        scaleConfidence = subjectLock.globalConfidence,
                    ),
                    lighting = null,
                    camera = null,
                    face = null,
                    eyes = null,
                    hands = emptyList(),
                ),
            )

            val runtimeDecision = runtimeGovernor.decide(
                timestampNanos = timestampNanos,
                motion = assessment.semanticMotion,
                trackedConfidence = (
                    0.62 * assessment.semanticMotion.confidence +
                        0.38 * subjectLock.globalConfidence
                ).coerceIn(0.0, 1.0),
            )


            val motionPolicyEvidence =
                MotionPolicyEvidenceDeriver.derive(
                    assessment = assessment,
                    motionQuality = refinedPose.quality,
                )

            val densePoseLandmarks =
                pose.normalized(refinedPose.result)

            val photometricHints =
                effectivePhotometricVectors
                    .mapNotNull { vector ->
                        if (
                            !vector.normalizedX.isFinite() ||
                            !vector.normalizedY.isFinite() ||
                            !vector.dxPixels.isFinite() ||
                            !vector.dyPixels.isFinite()
                        ) {
                            null
                        } else {
                            SubpixelLandmarkRefinementEngine.PhotometricHint(
                                normalizedX =
                                    vector.normalizedX
                                        .coerceIn(0.0, 1.0),
                                normalizedY =
                                    vector.normalizedY
                                        .coerceIn(0.0, 1.0),
                                normalizedDx =
                                    (
                                        vector.dxPixels /
                                            image.width.coerceAtLeast(1)
                                                .toDouble()
                                    ).coerceIn(-0.25, 0.25),
                                normalizedDy =
                                    (
                                        vector.dyPixels /
                                            image.height.coerceAtLeast(1)
                                                .toDouble()
                                    ).coerceIn(-0.25, 0.25),
                                confidence =
                                    vector.confidence
                                        .coerceIn(0.0, 1.0),
                                gradientAgreement =
                                    vector.gradientAgreement
                                        .coerceIn(0.0, 1.0),
                                textureStrength =
                                    vector.textureStrength
                                        .coerceIn(0.0, 1.0),
                                forwardBackwardConsistency =
                                    vector.forwardBackwardConsistency
                                        .coerceIn(0.0, 1.0),
                                subpixelRefined =
                                    vector.subpixelRefined,
                            )
                        }
                    }
                    ?: emptyList()

            val subpixelLandmarkRefinement =
                runCatching {
                    subpixelLandmarkRefinementEngine.refine(
                        observations = densePoseLandmarks,
                        timestampNanos = timestampNanos,
                        photometricHints = photometricHints,
                    )
                }.getOrNull()

            val denseContactEvidence =
                runCatching {
                    denseContactEvidenceAdapter.evaluate(
                        DenseContactEvidenceAdapter.Input(
                            timestampNanos = timestampNanos,
                            actorId =
                                assessment.dominantSubject?.trackingId
                                    ?.toLong()
                                    ?.coerceAtLeast(0L)
                                    ?: 0L,
                            handLandmarks =
                                effectiveHands.hands.flatMap { hand ->
                                    val offset =
                                        when (hand.side) {
                                            com.nivukx.nox.core.model.HandSide.LEFT -> 760
                                            com.nivukx.nox.core.model.HandSide.RIGHT -> 790
                                            com.nivukx.nox.core.model.HandSide.UNKNOWN -> 820
                                        }
                                    hand.landmarks.map { landmark ->
                                        DenseContactEvidenceAdapter.Point(
                                            landmarkId =
                                                offset +
                                                    landmark.index.coerceIn(
                                                        0,
                                                        21,
                                                    ),
                                            x =
                                                landmark.normalizedX
                                                    .coerceIn(-1.0, 2.0),
                                            y =
                                                landmark.normalizedY
                                                    .coerceIn(-1.0, 2.0),
                                            confidence =
                                                (
                                                    landmark.confidence *
                                                        hand.confidence
                                                    ).coerceIn(0.0, 1.0),
                                            semanticGroup = "hand",
                                        )
                                    }
                                },
                            counterpartLandmarks =
                                densePoseLandmarks.map { landmark ->
                                    DenseContactEvidenceAdapter.Point(
                                        landmarkId = landmark.landmarkId,
                                        x = landmark.x,
                                        y = landmark.y,
                                        confidence = landmark.confidence,
                                        semanticGroup = "body",
                                    )
                                } +
                                    effectiveObjects.map { tracked ->
                                        DenseContactEvidenceAdapter.Point(
                                            landmarkId =
                                                10_000 +
                                                    tracked.trackingId
                                                        .coerceIn(0, 999),
                                            x =
                                                (
                                                    tracked.left +
                                                        tracked.right
                                                    ).toDouble() * 0.5,
                                            y =
                                                (
                                                    tracked.top +
                                                        tracked.bottom
                                                    ).toDouble() * 0.5,
                                            confidence =
                                                tracked.confidence
                                                    .coerceIn(0.0, 1.0),
                                            semanticGroup = "object",
                                        )
                                    },
                        ),
                    )
                }.getOrNull()

            val denseContactProbability =
                denseContactEvidence
                    ?.protectedLandmarkConfidence
                    .orEmpty()
            val denseContactVisibility =
                denseContactEvidence
                    ?.landmarkVisibility
                    .orEmpty()
            val denseContactUncertaintyMultiplier =
                denseContactEvidence
                    ?.landmarkUncertaintyMultiplier
                    .orEmpty()

            val densePhotometricConsistency =
                temporalPhotometricConsensus
                    ?.takeIf { it.valid }
                    ?.temporalConsensusScore
                    ?.coerceIn(0.25, 1.0)
                    ?: 1.0

            val denseLandmarkEvidence =
                runCatching {
                    val denseSurfaceActors =
                        detailAllocation.denseSurfaceActorIds

                    val denseObservations =
                        buildList<
                            Pair<
                                DenseLandmarkEvidenceFusionEngine.LandmarkKey,
                                DenseLandmarkEvidenceFusionEngine.CameraObservation,
                            >
                        > {
                            (
                                subpixelLandmarkRefinement
                                    ?.observations
                                    ?.filter {
                                        it.state !=
                                            SubpixelLandmarkRefinementEngine
                                                .RefinementState.REJECTED
                                    }
                                    ?.map {
                                        it.source.copy(
                                            x = it.x,
                                            y = it.y,
                                            confidence =
                                                it.confidence.coerceIn(
                                                    0.0,
                                                    1.0,
                                                ),
                                            space =
                                                com.nivukx.nox.core.model
                                                    .LandmarkSpace.NORMALIZED,
                                        )
                                    }
                                    ?.takeIf { it.isNotEmpty() }
                                    ?: densePoseLandmarks
                                ).forEach { landmark ->
                                add(
                                    DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                        subjectId = 0,
                                        landmarkId = landmark.landmarkId,
                                    ) to
                                        DenseLandmarkEvidenceFusionEngine.CameraObservation(
                                            cameraId =
                                                stableCameraIdentifier(cameraId),
                                            timestampNanos = timestampNanos,
                                            normalizedX = landmark.x,
                                            normalizedY = landmark.y,
                                            confidence =
                                                landmark.confidence
                                                    .coerceIn(0.0, 1.0),
                                            uncertainty =
                                                (
                                                    1.0 -
                                                        landmark.confidence
                                                            .coerceIn(0.0, 1.0)
                                                ).coerceIn(0.0, 1.0).let { base ->
                                                    val multiplier =
                                                        denseContactUncertaintyMultiplier[
                                                            landmark.landmarkId
                                                        ] ?: 1.0
                                                    (
                                                        base +
                                                            (
                                                                multiplier - 1.0
                                                            ).coerceIn(0.0, 3.0) *
                                                            0.12
                                                    ).coerceIn(
                                                        0.0,
                                                        1.0,
                                                    )
                                                },
                                            visibility =
                                                minOf(
                                                    landmark.confidence
                                                        .coerceIn(0.0, 1.0),
                                                    denseContactVisibility[
                                                        landmark.landmarkId
                                                    ] ?: 1.0,
                                                ),
                                            contactProbability =
                                                denseContactProbability[
                                                    landmark.landmarkId
                                                ]?.coerceIn(
                                                    0.0,
                                                    1.0,
                                                ) ?: 0.0,
                                            cameraQuality =
                                                (
                                                    captureTimingConfidence *
                                                        (
                                                            opticalEvidence
                                                                ?.signalQuality
                                                                ?: 1.0
                                                        )
                                                ).coerceIn(
                                                    0.0,
                                                    1.0,
                                                ),
                                            photometricConsistency =
                                                densePhotometricConsistency,
                                            segmentationConsistency =
                                                personSegmentation?.let { segmentation ->
                                                    (
                                                        segmentation.silhouetteSupport(
                                                            landmark.x,
                                                            landmark.y,
                                                        ) * 0.72 +
                                                            segmentation.modelQuality * 0.18 +
                                                            segmentation.boundaryStability * 0.10
                                                    ).coerceIn(0.0, 1.0)
                                                } ?: 0.0,

                                            segmentationEvidenceAvailable = personSegmentation != null,
                                            rollingShutterCompensated =
                                                rollingShutterSkewNanos == 0L,
                                            sourceTag = "pose-front-end",
                                            probabilitySigmaXNormalized =
                                                (
                                                    0.002 +
                                                        (1.0 -
                                                            landmark.confidence.coerceIn(
                                                                0.0,
                                                                1.0,
                                                            )
                                                        ) * 0.08
                                                ).coerceIn(
                                                    1e-4,
                                                    0.25,
                                                ),
                                            probabilitySigmaYNormalized =
                                                (
                                                    0.002 +
                                                        (1.0 -
                                                            landmark.confidence.coerceIn(
                                                                0.0,
                                                                1.0,
                                                            )
                                                        ) * 0.08
                                                ).coerceIn(
                                                    1e-4,
                                                    0.25,
                                                ),
                                            probabilityCalibrationScore =
                                                landmark.confidence
                                                    .coerceIn(0.0, 1.0),
                                        ),
                                )
                            }

                            effectiveFaces.forEach { faceFrame ->
                                val faceMeshSubjectSlot =
                                    faceActorAssociation.assignments
                                        .firstOrNull {
                                            it.accepted &&
                                                it.faceTrackingId != null &&
                                                it.faceTrackingId ==
                                                    faceFrame.trackingId
                                        }
                                        ?.actorId
                                        ?.let {
                                            stableDenseSubjectSlot(
                                                actorId = it,
                                                timestampNanos = timestampNanos,
                                            )
                                        }
                                        ?: 0

                                val faceQuality =
                                    (
                                        faceFrame.captureTimingConfidence *
                                            (
                                                if (
                                                    faceFrame.rollingShutterSkewNanos ==
                                                        0L
                                                ) {
                                                    1.0
                                                } else {
                                                    0.86
                                                }
                                            )
                                        ).coerceIn(0.0, 1.0)

                                val faceSubjectSlot =
                                    faceActorAssociation.assignments
                                        .firstOrNull {
                                            it.accepted &&
                                                it.faceTrackingId != null &&
                                                it.faceTrackingId ==
                                                    faceFrame.trackingId
                                        }
                                        ?.actorId
                                        ?.let {
                                            stableDenseSubjectSlot(
                                                actorId = it,
                                                timestampNanos = timestampNanos,
                                            )
                                        }
                                        ?: 0

                                faceFrame.points.forEach { point ->
                                    val x =
                                        (
                                            point.point.x /
                                                image.width
                                                    .coerceAtLeast(1)
                                                    .toDouble()
                                            ).coerceIn(-1.0, 2.0)
                                    val y =
                                        (
                                            point.point.y /
                                                image.height
                                                    .coerceAtLeast(1)
                                                    .toDouble()
                                            ).coerceIn(-1.0, 2.0)
                                    add(
                                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                            subjectId = faceSubjectSlot,
                                            landmarkId =
                                                256 +
                                                    point.index.coerceIn(
                                                        0,
                                                        511,
                                                    ),
                                        ) to
                                            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                                                cameraId =
                                                    stableCameraIdentifier(
                                                        cameraId,
                                                    ),
                                                timestampNanos =
                                                    timestampNanos,
                                                normalizedX = x,
                                                normalizedY = y,
                                                confidence =
                                                    (
                                                        point.confidence *
                                                            faceQuality
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                uncertainty =
                                                    (
                                                        1.0 -
                                                            (
                                                                point.confidence *
                                                                    faceQuality
                                                                ).coerceIn(
                                                                    0.0,
                                                                    1.0,
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                visibility =
                                                    point.confidence
                                                        .coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                cameraQuality =
                                                    (
                                                        captureTimingConfidence *
                                                            (
                                                                opticalEvidence
                                                                    ?.signalQuality
                                                                    ?: 1.0
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                segmentationConsistency =
                                                    personSegmentation?.let { segmentation ->
                                                        (
                                                            segmentation.silhouetteSupport(
                                                                x,
                                                                y,
                                                            ) * 0.68 +
                                                                faceQuality * 0.22 +
                                                                segmentation.boundaryStability * 0.10
                                                        ).coerceIn(0.0, 1.0)
                                                    } ?: 0.0,

                                                segmentationEvidenceAvailable = personSegmentation != null,
                                                rollingShutterCompensated =
                                                    faceFrame.rollingShutterSkewNanos ==
                                                        0L,
                                                sourceTag = "face-mesh",
                                            ),
                                    )
                                }
                            }

                            effectiveHands.hands.forEach { hand ->
                                val handSubjectSlot =
                                    handActorAssociation.assignments
                                        .firstOrNull {
                                            it.accepted &&
                                                it.handId ==
                                                    hand.handId
                                        }
                                        ?.actorId
                                        ?.let {
                                            stableDenseSubjectSlot(
                                                actorId = it,
                                                timestampNanos = timestampNanos,
                                            )
                                        }
                                        ?: 0

                                val offset =
                                    when (hand.side) {
                                        com.nivukx.nox.core.model.HandSide.LEFT -> 760
                                        com.nivukx.nox.core.model.HandSide.RIGHT -> 790
                                        com.nivukx.nox.core.model.HandSide.UNKNOWN -> 820
                                    }
                                hand.landmarks.forEach { landmark ->
                                    val confidence =
                                        (
                                            landmark.confidence *
                                                hand.confidence
                                            ).coerceIn(0.0, 1.0)
                                    add(
                                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                            subjectId = handSubjectSlot,
                                            landmarkId =
                                                offset +
                                                    landmark.index.coerceIn(
                                                        0,
                                                        21,
                                                    ),
                                        ) to
                                            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                                                cameraId =
                                                    stableCameraIdentifier(
                                                        cameraId,
                                                    ),
                                                timestampNanos =
                                                    timestampNanos,
                                                normalizedX =
                                                    landmark.normalizedX
                                                        .coerceIn(
                                                            -1.0,
                                                            2.0,
                                                        ),
                                                normalizedY =
                                                    landmark.normalizedY
                                                        .coerceIn(
                                                            -1.0,
                                                            2.0,
                                                        ),
                                                depthMeters =
                                                    landmark.normalizedZ
                                                        .takeIf {
                                                            it.isFinite()
                                                        },
                                                confidence = confidence,
                                                uncertainty =
                                                    (
                                                        1.0 -
                                                            confidence
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                    ).let { base ->
                                                        val multiplier =
                                                            denseContactUncertaintyMultiplier[
                                                                offset +
                                                                    landmark.index.coerceIn(
                                                                        0,
                                                                        21,
                                                                    )
                                                            ] ?: 1.0
                                                        (
                                                            base +
                                                                (
                                                                    multiplier - 1.0
                                                                ).coerceIn(
                                                                    0.0,
                                                                    3.0,
                                                                ) * 0.12
                                                            ).coerceIn(
                                                                0.0,
                                                                1.0,
                                                            )
                                                    },
                                                visibility =
                                                    minOf(
                                                        confidence,
                                                        denseContactVisibility[
                                                            offset +
                                                                landmark.index.coerceIn(
                                                                    0,
                                                                    21,
                                                                )
                                                        ] ?: 1.0,
                                                    ),
                                                contactProbability =
                                                    denseContactProbability[
                                                        offset +
                                                            landmark.index.coerceIn(
                                                                0,
                                                                21,
                                                            )
                                                    ]?.coerceIn(
                                                        0.0,
                                                        1.0,
                                                    ) ?: 0.0,
                                                cameraQuality =
                                                    (
                                                        captureTimingConfidence *
                                                            (
                                                                opticalEvidence
                                                                    ?.signalQuality
                                                                    ?: 1.0
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                rollingShutterCompensated =
                                                    rollingShutterSkewNanos ==
                                                        0L,
                                                sourceTag =
                                                    "hand-" +
                                                        hand.side.name
                                                            .lowercase(),
                                            ),
                                    )
                                }
                            }

                            eyeFrames.forEach { eyeFrame ->
                                val eyeSubjectSlot =
                                    faceActorAssociation.assignments
                                        .firstOrNull {
                                            it.accepted &&
                                                it.faceTrackingId != null &&
                                                it.faceTrackingId ==
                                                    eyeFrame.trackingId
                                        }
                                        ?.actorId
                                        ?.let {
                                            stableDenseSubjectSlot(
                                                actorId = it,
                                                timestampNanos = timestampNanos,
                                            )
                                        }
                                        ?: 0

                                val leftEye = eyeFrame.left
                                val rightEye = eyeFrame.right
                                if (leftEye != null) {
                                    add(
                                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                            subjectId = eyeSubjectSlot,
                                            landmarkId = 840,
                                        ) to
                                            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                                                cameraId =
                                                    stableCameraIdentifier(
                                                        cameraId,
                                                    ),
                                                timestampNanos =
                                                    timestampNanos,
                                                normalizedX =
                                                    (
                                                        leftEye.center.x /
                                                            image.width
                                                                .coerceAtLeast(1)
                                                                .toDouble()
                                                        ).coerceIn(
                                                            -1.0,
                                                            2.0,
                                                        ),
                                                normalizedY =
                                                    (
                                                        leftEye.center.y /
                                                            image.height
                                                                .coerceAtLeast(1)
                                                                .toDouble()
                                                        ).coerceIn(
                                                            -1.0,
                                                            2.0,
                                                        ),
                                                confidence =
                                                    leftEye.confidence
                                                        .coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                uncertainty =
                                                    (
                                                        1.0 -
                                                            leftEye.confidence
                                                                .coerceIn(
                                                                    0.0,
                                                                    1.0,
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                visibility =
                                                    leftEye.confidence
                                                        .coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                cameraQuality =
                                                    (
                                                        captureTimingConfidence *
                                                            (
                                                                opticalEvidence
                                                                    ?.signalQuality
                                                                    ?: 1.0
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                sourceTag = "eye-left",
                                            ),
                                    )
                                }

                                if (rightEye != null) {
                                    add(
                                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                            subjectId = eyeSubjectSlot,
                                            landmarkId = 841,
                                        ) to
                                            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                                                cameraId =
                                                    stableCameraIdentifier(
                                                        cameraId,
                                                    ),
                                                timestampNanos =
                                                    timestampNanos,
                                                normalizedX =
                                                    (
                                                        rightEye.center.x /
                                                            image.width
                                                                .coerceAtLeast(1)
                                                                .toDouble()
                                                        ).coerceIn(
                                                            -1.0,
                                                            2.0,
                                                        ),
                                                normalizedY =
                                                    (
                                                        rightEye.center.y /
                                                            image.height
                                                                .coerceAtLeast(1)
                                                                .toDouble()
                                                        ).coerceIn(
                                                            -1.0,
                                                            2.0,
                                                        ),
                                                confidence =
                                                    rightEye.confidence
                                                        .coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                uncertainty =
                                                    (
                                                        1.0 -
                                                            rightEye.confidence
                                                                .coerceIn(
                                                                    0.0,
                                                                    1.0,
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                visibility =
                                                    rightEye.confidence
                                                        .coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                cameraQuality =
                                                    (
                                                        captureTimingConfidence *
                                                            (
                                                                opticalEvidence
                                                                    ?.signalQuality
                                                                    ?: 1.0
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                sourceTag = "eye-right",
                                            ),
                                    )
                                }
                            }

                            actorTracks
                        .filter {
                            denseSurfaceActors.isEmpty() ||
                                it.actorId in denseSurfaceActors
                        }
                        .forEach { actor ->
                                val subjectSlot =
                                    stableDenseSubjectSlot(
                                        actorId = actor.actorId,
                                        timestampNanos = timestampNanos,
                                    )

                                actor.joints.forEach { joint ->
                                    val jointConfidence =
                                        (
                                            joint.confidence *
                                                actor.confidence
                                            ).coerceIn(0.0, 1.0)
                                    add(
                                        DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                            subjectId = subjectSlot,
                                            landmarkId = joint.landmarkId,
                                        ) to
                                            DenseLandmarkEvidenceFusionEngine.CameraObservation(
                                                cameraId =
                                                    stableCameraIdentifier(
                                                        cameraId,
                                                    ),
                                                timestampNanos =
                                                    timestampNanos,
                                                normalizedX =
                                                    joint.x.coerceIn(
                                                        -1.0,
                                                        2.0,
                                                    ),
                                                normalizedY =
                                                    joint.y.coerceIn(
                                                        -1.0,
                                                        2.0,
                                                    ),
                                                confidence =
                                                    jointConfidence,
                                                uncertainty =
                                                    (
                                                        1.0 -
                                                            jointConfidence
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                visibility =
                                                    minOf(
                                                        if (actor.predicted) {
                                                            jointConfidence * 0.72
                                                        } else {
                                                            jointConfidence
                                                        },
                                                        denseContactVisibility[
                                                            joint.landmarkId
                                                        ] ?: 1.0,
                                                    ),
                                                contactProbability =
                                                    denseContactProbability[
                                                        joint.landmarkId
                                                    ]?.coerceIn(
                                                        0.0,
                                                        1.0,
                                                    ) ?: 0.0,
                                                cameraQuality =
                                                    (
                                                        captureTimingConfidence *
                                                            (
                                                                opticalEvidence
                                                                    ?.signalQuality
                                                                    ?: 1.0
                                                                )
                                                        ).coerceIn(
                                                            0.0,
                                                            1.0,
                                                        ),
                                                segmentationConsistency =
                                                    personSegmentation?.let { segmentation ->
                                                        (
                                                            segmentation.silhouetteSupport(
                                                                joint.x,
                                                                joint.y,
                                                            ) * 0.70 +
                                                                jointConfidence * 0.20 +
                                                                segmentation.boundaryStability * 0.10
                                                        ).coerceIn(0.0, 1.0)
                                                    } ?: 0.0,

                                                segmentationEvidenceAvailable = personSegmentation != null,
                                                rollingShutterCompensated =
                                                    rollingShutterSkewNanos ==
                                                        0L,
                                                sourceTag =
                                                    "multi-actor-" +
                                                        actor.actorId,
                                            ),
                                    )
                                }
                            }
                        }

                    val calibratedDenseObservations =
                        denseObservations.map { (key, observation) ->
                            val ray =
                                worldRayForDenseObservation(
                                    cameraId = calibrationCameraId,
                                    normalizedX = observation.normalizedX,
                                    normalizedY = observation.normalizedY,
                                    imageWidth = image.width,
                                    imageHeight = image.height,
                                )
                            key to observation.copy(
                                worldRayOriginMeters = ray?.origin,
                                worldRayDirection = ray?.direction,
                                calibrationUncertaintyMeters =
                                    ray?.calibrationUncertaintyMeters ?: 0.0,
                                calibrationConfidence =
                                    ray?.calibrationConfidence ?: 0.0,
                                calibrationEvidenceAvailable = ray != null,
                                motionBlurRisk =
                                    opticalEvidence
                                        ?.motionBlurRisk
                                        ?.coerceIn(0.0, 1.0)
                                        ?: 0.0,
                            )
                        }

                    denseLandmarkFusionEngine.process(
                        DenseLandmarkEvidenceFusionEngine.FrameInput(
                            timestampNanos = timestampNanos,
                            observations = calibratedDenseObservations,
                        ),
                    )
                }.getOrNull()

            val frontierActorDemand =
                actorTracks
                    .take(64)
                    .map { actor ->
                        FrontierActorDetailScheduler.ActorDemand(
                            actorId = actor.actorId.toInt().coerceAtLeast(0),
                            uncertainty = (1.0 - actor.confidence).coerceIn(0.0, 1.0),
                            visibility = if (actor.predicted) 0.35 else 0.85,
                            motionComplexity = actor.microMotion.preservedSignal.coerceIn(0.0, 1.0),
                            interactionRisk = actor.identityAmbiguity.coerceIn(0.0, 1.0),
                            reconstructionValue =
                                (
                                    actor.confidence *
                                        (1.0 - actor.identityAmbiguity.coerceIn(0.0, 1.0))
                                ).coerceIn(0.0, 1.0),
                            ageSinceDetailFrames = actor.missedFrames.toLong().coerceAtLeast(0L),
                            opticalMotionBlurRisk =
                                opticalEvidence
                                    ?.motionBlurRisk
                                    ?.coerceIn(0.0, 1.0)
                                    ?: 0.0,
                        )
                    }
            val frontierActorAllocation =
                frontierActorDetailScheduler.allocate(
                    demands = frontierActorDemand,
                    pressure =
                        (
                            1.0 -
                                minOf(
                                    thermalScale,
                                    memoryScale,
                                    gpuScale,
                                    npuScale,
                                )
                        ).coerceIn(0.0, 1.0),
                )

            val motionOrchestration =
                motionOrchestrator.update(
                    UniversalMotionInferenceOrchestrator.FrameInput(
                        timestampNanos = timestampNanos,
                        frameIndex = frameCounter,
                        motionQuality = refinedPose.quality,
                        assessment = assessment,
                        subjectCount = maxOf(
                            effectiveFaces.size,
                            effectiveHands.trackedHands,
                            effectiveObjects.count {
                                it.subjectType == SubjectType.HUMAN ||
                                    it.subjectType == SubjectType.ANIMAL
                            },
                            actorTracks.size,
                        ),
                        dominantTrackId =
                            assessment.dominantSubject?.trackingId,
                        dominantSubjectType =
                            assessment.dominantSubject?.type
                                ?: SubjectType.UNKNOWN,
                        lockConfidence =
                            (
                                subjectLock.globalConfidence *
                                    (
                                        0.58 +
                                            0.42 *
                                                trackLifecycle.continuityScore
                                    )
                                ).coerceIn(0.0, 1.0),
                        thermalScale = thermalScale,
                        memoryScale = memoryScale,
                        gpuScale = gpuScale,
                        npuScale = npuScale,
                        viewpointChallenge =
                            motionPolicyEvidence.viewpointChallenge,
                        leftRightAmbiguity =
                            motionPolicyEvidence.leftRightAmbiguity,
                    ),
                )

            Result(
                pose = refinedPose.result,
                faces = effectiveFaces,
                eyes = eyeFrames,
                hands = effectiveHands.copy(
                    timestampNanos = if (handsFresh) {
                        timestampNanos
                    } else {
                        effectiveHands.timestampNanos
                    },
                ),
                objects = effectiveObjects,
                tracks = tracks,
                faceTracks = faceTrackSnapshot,
                faceContour = contourForFrame,
                faceMeshes = effectiveFaceMeshes,
                motion = refinedPose.quality,
                subframeTiming = subframeTiming,
                photometricMotion = photometricMotion,
                temporalPhotometricConsensus =
                    temporalPhotometricConsensus,
                highOrderMotion = highOrderMotion,
                motionFusion = assessment,
                subjectLock = subjectLock,
                runtimeDecision = runtimeDecision,
                stageFreshness = mapOf(
                    AdaptivePerceptionCoordinator.Stage.POSE to true,
                    AdaptivePerceptionCoordinator.Stage.FACE to
                        (facesFresh || faceMeshTask.isSuccessful),
                    AdaptivePerceptionCoordinator.Stage.EYE to eyeDecision,
                    AdaptivePerceptionCoordinator.Stage.HAND to handsFresh,
                    AdaptivePerceptionCoordinator.Stage.OBJECT to objectsFresh,
                    AdaptivePerceptionCoordinator.Stage.CONTOUR to contourShouldRun,
                    AdaptivePerceptionCoordinator.Stage.SEGMENTATION to
                        (personSegmentation != null),
                ),
                inferencePlan = plan,
                fusion = fusion,
                denseLandmarkEvidence = denseLandmarkEvidence,
                denseContactEvidence = denseContactEvidence,
                subpixelLandmarkRefinement = subpixelLandmarkRefinement,
                faceEyeFusion = faceEyeFusion,
                stableSubjectIdentities = stableSubjectIdentities,
                motionOrchestration = motionOrchestration,
                multiActorPose = multiActorPoseResult,
                actorTracks = actorTracks,
                crowdMotionEvidence = crowdMotionEvidence,
                frontierActorAllocation = frontierActorAllocation,
                personSegmentation = personSegmentation,
                actorSilhouetteEvidence = actorSilhouetteEvidence,
                detailAllocation = detailAllocation,
                multiPersonCapacityAllocation =
                    capacityAllocation,
                opticalEvidence = opticalEvidence,
                faceActorAssociation = faceActorAssociation,
                handActorAssociation = handActorAssociation,
                trackLifecycle = trackLifecycle,
            )
        }
    }

    private data class CalibratedWorldRay(
        val origin: com.nivukx.nox.core.model.Vec3,
        val direction: com.nivukx.nox.core.model.Vec3,
        val calibrationUncertaintyMeters: Double,
        val calibrationConfidence: Double,
    )

    private fun worldRayForDenseObservation(
        cameraId: String,
        normalizedX: Double,
        normalizedY: Double,
        imageWidth: Int,
        imageHeight: Int,
    ): CalibratedWorldRay? {
        val geometry = calibratedCameraGeometry[cameraId] ?: return null
        val width = imageWidth.coerceAtLeast(1).toDouble()
        val height = imageHeight.coerceAtLeast(1).toDouble()
        val fx = geometry.intrinsics.fx
        val fy = geometry.intrinsics.fy
        if (
            !fx.isFinite() ||
                !fy.isFinite() ||
                fx <= 1e-6 ||
                fy <= 1e-6
        ) {
            return null
        }

        val pixelX = normalizedX * width
        val pixelY = normalizedY * height
        val xd = (pixelX - geometry.intrinsics.cx) / fx
        val yd = (pixelY - geometry.intrinsics.cy) / fy
        var x = xd
        var y = yd

        repeat(8) {
            val radius2 = x * x + y * y
            val radial =
                1.0 +
                    geometry.intrinsics.k1 * radius2 +
                    geometry.intrinsics.k2 * radius2 * radius2 +
                    geometry.intrinsics.k3 * radius2 * radius2 * radius2
            if (
                !radial.isFinite() ||
                    kotlin.math.abs(radial) < 1e-9
            ) {
                return@repeat
            }
            val tx =
                2.0 * geometry.intrinsics.p1 * x * y +
                    geometry.intrinsics.p2 *
                        (radius2 + 2.0 * x * x)
            val ty =
                geometry.intrinsics.p1 *
                    (radius2 + 2.0 * y * y) +
                    2.0 * geometry.intrinsics.p2 * x * y
            x = (xd - tx) / radial
            y = (yd - ty) / radial
        }

        val directionCamera =
            com.nivukx.nox.core.model.Vec3(
                x,
                y,
                1.0,
            ).normalized()
        val directionWorld =
            geometry.pose.rotation
                .rotate(directionCamera)
                .normalized()

        if (
            !finiteVec(directionWorld) ||
                directionWorld.norm() < 0.99
        ) {
            return null
        }

        // Zero sigma fields mean the estimate is absent, not that calibration is exact.
        // The shared policy supplies conservative priors independently for missing fields.
        // A missing geometry record is handled separately at observation adaptation time.
        // That path emits no world ray and must not gain calibrated-ray agreement weight.
        // Geometry is always looked up by the stable identity attached to the calibration artifact.
        val resolvedUncertainty =
            resolveCalibratedRayUncertainty(geometry.uncertainty)
        val uncertainty =
            resolvedUncertainty.translationSigmaMeters
        val rotationPenalty =
            resolvedUncertainty.rotationPenalty
        val intrinsicPenalty =
            resolvedUncertainty.intrinsicPenalty
        val evidenceConfidence =
            resolvedUncertainty.evidenceConfidence

        // Preserve the established score composition while lowering authority for unknown
        // calibration; this score remains an evidence weight, not a measured accuracy claim.
        val confidence =
            (
                evidenceConfidence *
                    (1.0 - 0.20 * rotationPenalty) *
                    (1.0 - 0.15 * intrinsicPenalty)
                ).coerceIn(0.0, 1.0)

        return CalibratedWorldRay(
            origin = geometry.pose.position,
            direction = directionWorld,
            calibrationUncertaintyMeters = uncertainty,
            calibrationConfidence = confidence,
        )
    }

    private fun finiteVec(
        value: com.nivukx.nox.core.model.Vec3,
    ): Boolean =
        value.x.isFinite() &&
            value.y.isFinite() &&
            value.z.isFinite()

    private fun stableCameraIdentifier(cameraId: String): Int {
        val normalized = cameraId.trim()
        val numeric = normalized.toIntOrNull()
        if (numeric != null && numeric >= 0) return numeric

        var hash = 0x811C9DC5.toInt()
        for (character in normalized) {
            hash = hash xor character.code
            hash *= 0x01000193
        }

        return (hash and Int.MAX_VALUE)
            .coerceAtMost(2_000_000_000)
    }

    private fun stableDenseSubjectSlot(
        actorId: Long,
        timestampNanos: Long,
    ): Int {
        val retentionNanos = 5_000_000_000L

        val stale = denseActorSubjectSlots.entries
            .filter {
                timestampNanos -
                    it.value.lastSeenTimestampNanos >
                    retentionNanos
            }
            .map { it.key }

        stale.forEach(denseActorSubjectSlots::remove)

        denseActorSubjectSlots[actorId]?.let {
            it.lastSeenTimestampNanos = timestampNanos
            return it.slot
        }

        val used = denseActorSubjectSlots.values
            .map { it.slot }
            .toSet()

        val nextSlot = (1..50)
            .firstOrNull { it !in used }
            ?: run {
                denseActorSubjectSlots.minByOrNull {
                    it.value.lastSeenTimestampNanos
                }?.let { (oldId, oldState) ->
                    denseActorSubjectSlots.remove(oldId)
                    oldState.slot
                }
                    ?: 1
            }

        denseActorSubjectSlots[actorId] =
            DenseActorSlot(
                slot = nextSlot,
                lastSeenTimestampNanos = timestampNanos,
            )

        return nextSlot
    }

    private fun buildSubjectLockObservations(
        objects: List<TrackedObject>,
        faces: List<FaceFrame>,
        eyes: List<EyeTrackingFrame>,
    ): List<ProductionSubjectLockEngine.Observation> {
        val output = objects.map {
            ProductionSubjectLockEngine.Observation(
                detectorId = it.trackingId.toLong(),
                kind = when (it.subjectType) {
                    SubjectType.HUMAN -> ProductionSubjectLockEngine.SubjectKind.HUMAN
                    SubjectType.ANIMAL -> ProductionSubjectLockEngine.SubjectKind.ANIMAL
                    SubjectType.OBJECT -> ProductionSubjectLockEngine.SubjectKind.OBJECT
                    SubjectType.UNKNOWN -> ProductionSubjectLockEngine.SubjectKind.UNKNOWN
                },
                left = it.left.toDouble().coerceIn(0.0, 1.0),
                top = it.top.toDouble().coerceIn(0.0, 1.0),
                right = it.right.toDouble().coerceIn(0.0, 1.0),
                bottom = it.bottom.toDouble().coerceIn(0.0, 1.0),
                confidence = it.confidence,
                saliency = it.confidence,
            )
        }.toMutableList()

        faces.forEach { face ->
            val faceTrackingId = face.trackingId
            val faceConfidence = if (face.points.isEmpty()) {
                0.0
            } else {
                face.points
                    .map { it.confidence.coerceIn(0.0, 1.0) }
                    .average()
                    .coerceIn(0.0, 1.0)
            }
            val left = face.boundsLeft.toDouble().coerceIn(0.0, 1.0)
            val top = face.boundsTop.toDouble().coerceIn(0.0, 1.0)
            val right = face.boundsRight.toDouble().coerceIn(0.0, 1.0)
            val bottom = face.boundsBottom.toDouble().coerceIn(0.0, 1.0)

            val matchingIndex = output.indices
                .filter { index ->
                    output[index].kind == ProductionSubjectLockEngine.SubjectKind.HUMAN
                }
                .minByOrNull { index ->
                    val candidate = output[index]
                    centerDistance(
                        candidate.centerX,
                        candidate.centerY,
                        (left + right) * 0.5,
                        (top + bottom) * 0.5,
                    )
                }
                ?.takeIf { index ->
                    val candidate = output[index]
                    faceTrackingId != null &&
                        candidate.detectorId == faceTrackingId.toLong()
                }
                ?: output.indices
                    .filter { index ->
                        output[index].kind ==
                            ProductionSubjectLockEngine.SubjectKind.HUMAN
                    }
                    .minByOrNull { index ->
                        val candidate = output[index]
                        centerDistance(
                            candidate.centerX,
                            candidate.centerY,
                            (left + right) * 0.5,
                            (top + bottom) * 0.5,
                        )
                    }
                    ?.takeIf { index ->
                        val candidate = output[index]
                        centerDistance(
                            candidate.centerX,
                            candidate.centerY,
                            (left + right) * 0.5,
                            (top + bottom) * 0.5,
                        ) <= 0.18
                    }

            val eyeFrame = eyes
                .firstOrNull {
                    it.trackingId != null &&
                        it.trackingId == faceTrackingId
                }

            val leftEye = eyeFrame?.left?.let { observed ->
                ProductionSubjectLockEngine.EyeObservation(
                    x = observed.center.x,
                    y = observed.center.y,
                    confidence = observed.confidence
                        .coerceIn(0.0, 1.0),
                    openness = observed.openness
                        .coerceIn(0.0, 1.0),
                )
            } ?: face.leftEyeCenter?.let {
                ProductionSubjectLockEngine.EyeObservation(
                    x = it.x,
                    y = it.y,
                    confidence = face.leftEyeOpenProbability
                        .coerceIn(0.0, 1.0),
                    openness = face.leftEyeOpenProbability
                        .coerceIn(0.0, 1.0),
                )
            }

            val rightEye = eyeFrame?.right?.let { observed ->
                ProductionSubjectLockEngine.EyeObservation(
                    x = observed.center.x,
                    y = observed.center.y,
                    confidence = observed.confidence
                        .coerceIn(0.0, 1.0),
                    openness = observed.openness
                        .coerceIn(0.0, 1.0),
                )
            } ?: face.rightEyeCenter?.let {
                ProductionSubjectLockEngine.EyeObservation(
                    x = it.x,
                    y = it.y,
                    confidence = face.rightEyeOpenProbability
                        .coerceIn(0.0, 1.0),
                    openness = face.rightEyeOpenProbability
                        .coerceIn(0.0, 1.0),
                )
            }

            val gaze = eyeFrame?.let { frame ->
                val gazeConfidence = frame.fixationConfidence
                    .coerceIn(0.0, 1.0)
                if (gazeConfidence > 0.0) {
                    val observedLeft = frame.left
                    val observedRight = frame.right
                    val eyeCenterX = when {
                        observedLeft != null && observedRight != null ->
                            (observedLeft.center.x + observedRight.center.x) * 0.5
                        observedLeft != null -> observedLeft.center.x
                        observedRight != null -> observedRight.center.x
                        else -> (left + right) * 0.5
                    }
                    val eyeCenterY = when {
                        observedLeft != null && observedRight != null ->
                            (observedLeft.center.y + observedRight.center.y) * 0.5
                        observedLeft != null -> observedLeft.center.y
                        observedRight != null -> observedRight.center.y
                        else -> (top + bottom) * 0.5
                    }
                    ProductionSubjectLockEngine.GazeObservation(
                        x = eyeCenterX,
                        y = eyeCenterY,
                        confidence = gazeConfidence,
                    )
                } else {
                    null
                }
            }

            if (matchingIndex != null) {
                val existing = output[matchingIndex]
                output[matchingIndex] = existing.copy(
                    faceCenterX = (
                        face.boundsLeft + face.boundsRight
                    ).toDouble() * 0.5,
                    faceCenterY = (
                        face.boundsTop + face.boundsBottom
                    ).toDouble() * 0.5,
                    leftEye = leftEye,
                    rightEye = rightEye,
                    gaze = gaze,
                    confidence = maxOf(
                        existing.confidence,
                        faceConfidence,
                    ),
                    saliency = maxOf(
                        existing.saliency,
                        faceConfidence,
                    ),
                )
            } else {
                output += ProductionSubjectLockEngine.Observation(
                    detectorId = faceTrackingId?.toLong() ?: -1L,
                    kind = ProductionSubjectLockEngine.SubjectKind.HUMAN,
                    left = left,
                    top = top,
                    right = right,
                    bottom = bottom,
                    confidence = faceConfidence,
                    saliency = faceConfidence,
                    faceCenterX = (left + right) * 0.5,
                    faceCenterY = (top + bottom) * 0.5,
                    leftEye = leftEye,
                    rightEye = rightEye,
                    gaze = gaze,
                )
            }
        }

        return output
    }

    private fun centerDistance(
        x1: Double,
        y1: Double,
        x2: Double,
        y2: Double,
    ): Double =
        kotlin.math.hypot(x1 - x2, y1 - y2)

    private fun faceEyePolicyEvidence(
        image: Image,
        faces: List<FaceFrame>,
        eyeResults: List<EyeTrackingFrame>,
        timestampNanos: Long,
        lightingConfidence: Double,
    ): AdaptiveFaceEyeInferencePolicy.Evidence {
        val faceAreaRatio =
            faces.maxOfOrNull { face ->
                val widthRatio =
                    (
                        face.boundsRight -
                            face.boundsLeft
                        ).coerceAtLeast(0f).toDouble() /
                        image.width.coerceAtLeast(1).toDouble()
                val heightRatio =
                    (
                        face.boundsBottom -
                            face.boundsTop
                        ).coerceAtLeast(0f).toDouble() /
                        image.height.coerceAtLeast(1).toDouble()
                widthRatio.coerceAtLeast(0.0) *
                    heightRatio.coerceAtLeast(0.0)
            } ?: 0.0

        val faceConfidence =
            faces.mapNotNull { face ->
                face.points
                    .map { it.confidence.coerceIn(0.0, 1.0) }
                    .takeIf { it.isNotEmpty() }
                    ?.average()
            }.average().takeIf { it.isFinite() } ?: 0.0

        val eyeConfidence =
            eyeResults
                .map { it.fixationConfidence.coerceIn(0.0, 1.0) }
                .average()
                .takeIf { it.isFinite() } ?: 0.0

        val faceMeshAge =
            ageNanos(
                lastFaceMeshesTimestampNanos,
                timestampNanos,
            )
        val eyeAge =
            ageNanos(
                lastEyeResultsTimestampNanos,
                timestampNanos,
            )

        return AdaptiveFaceEyeInferencePolicy.Evidence(
            faceAvailable = faces.isNotEmpty(),
            eyeAvailable = eyeResults.isNotEmpty(),
            faceAreaRatio = faceAreaRatio.coerceIn(0.0, 1.0),
            faceConfidence = faceConfidence.coerceIn(0.0, 1.0),
            eyeConfidence = eyeConfidence,
            fixationStability = eyeConfidence,
            motionEnergy = lastMotionQuality.motionScore
                .coerceIn(0.0, 1.0),
            occlusionRisk = lastMotionQuality.occlusionRatio
                .coerceIn(0.0, 1.0),
            faceMeshAgeNanos = faceMeshAge,
            eyeAgeNanos = eyeAge,
            thermalHeadroom = thermalScale,
            gpuHeadroom = gpuScale,
            faceDetail = customization.faceDetail,
            eyeDetail = customization.eyeDetail,
            lightingConfidence =
                lightingConfidence.coerceIn(0.0, 1.0),
            faceUncertainty =
                (
                    1.0 -
                        faceConfidence.coerceIn(0.0, 1.0)
                    ) * 0.08,
            eyeUncertainty =
                (
                    1.0 -
                        eyeConfidence.coerceIn(0.0, 1.0)
                    ) * 0.05,
        )
    }

    private fun ageNanos(
        previousTimestampNanos: Long,
        timestampNanos: Long,
    ): Long =
        if (previousTimestampNanos == Long.MIN_VALUE) {
            Long.MAX_VALUE
        } else {
            (timestampNanos - previousTimestampNanos)
                .coerceAtLeast(0L)
        }

    private fun buildEyeFrames(
        image: Image,
        rotationDegrees: Int,
        faces: List<FaceFrame>,
        plan: AdaptivePerceptionCoordinator.Plan,
        forceProcess: Boolean,
        timestampNanos: Long,
        captureTimingConfidence: Double,
        rollingShutterSkewNanos: Long,
        frameDurationNanos: Long,
        exposureTimeNanos: Long,
    ): List<EyeTrackingFrame> {
        if (faces.isEmpty()) return emptyList()

        val shouldProcess = forceProcess || lastEyeResults.isEmpty()

        return if (shouldProcess) {
            val start = System.nanoTime()
            val results = faces.map { faceFrame ->
                eyes.process(
                    image = image,
                    rotationDegrees = rotationDegrees,
                    face = faceFrame,
                    captureTimingConfidence = captureTimingConfidence,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                )
            }
            coordinator.recordStageDuration(
                AdaptivePerceptionCoordinator.Stage.EYE,
                (System.nanoTime() - start).coerceAtLeast(0L) / 1e6,
            )
            lastEyeResults = results
            lastEyeResultsTimestampNanos = timestampNanos
            results
        } else {
            lastEyeResults
        }
    }

    private var lastEyeResults: List<EyeTrackingFrame> = emptyList()
    private var lastEyeResultsTimestampNanos: Long = Long.MIN_VALUE

    fun handReadiness(): Task<HandTrackingEngine.Readiness> =
        hands.readiness()

    private fun associateFaceMeshes(
        meshes: List<FaceMeshGeometryEngine.Result>,
        faces: List<FaceFrame>,
    ): List<FaceMeshGeometryEngine.Result> {
        if (meshes.isEmpty() || faces.isEmpty()) return meshes

        data class Candidate(
            val meshIndex: Int,
            val faceIndex: Int,
            val score: Double,
        )

        val candidates = ArrayList<Candidate>()

        meshes.forEachIndexed { meshIndex, mesh ->
            val meshCx =
                (mesh.boundsLeft + mesh.boundsRight) * 0.5
            val meshCy =
                (mesh.boundsTop + mesh.boundsBottom) * 0.5

            faces.forEachIndexed { faceIndex, face ->
                val faceCx =
                    (face.boundsLeft + face.boundsRight) * 0.5
                val faceCy =
                    (face.boundsTop + face.boundsBottom) * 0.5

                val faceDiagonal =
                    kotlin.math.hypot(
                        (face.boundsRight - face.boundsLeft).toDouble(),
                        (face.boundsBottom - face.boundsTop).toDouble(),
                    ).coerceAtLeast(1.0)

                val centerScore =
                    (
                        1.0 -
                            kotlin.math.hypot(
                                (meshCx - faceCx).toDouble(),
                                (meshCy - faceCy).toDouble(),
                            ) /
                            (faceDiagonal * 0.90)
                        ).coerceIn(0.0, 1.0)

                val intersectionLeft =
                    maxOf(mesh.boundsLeft, face.boundsLeft)
                val intersectionTop =
                    maxOf(mesh.boundsTop, face.boundsTop)
                val intersectionRight =
                    minOf(mesh.boundsRight, face.boundsRight)
                val intersectionBottom =
                    minOf(mesh.boundsBottom, face.boundsBottom)

                val intersectionArea =
                    (
                        (intersectionRight - intersectionLeft)
                            .coerceAtLeast(0f) *
                            (intersectionBottom - intersectionTop)
                                .coerceAtLeast(0f)
                        ).toDouble()

                val meshArea =
                    (
                        (mesh.boundsRight - mesh.boundsLeft)
                            .coerceAtLeast(0f) *
                            (mesh.boundsBottom - mesh.boundsTop)
                                .coerceAtLeast(0f)
                        ).toDouble()

                val faceArea =
                    (
                        (face.boundsRight - face.boundsLeft)
                            .coerceAtLeast(0f) *
                            (face.boundsBottom - face.boundsTop)
                                .coerceAtLeast(0f)
                        ).toDouble()

                val unionArea =
                    (
                        meshArea +
                            faceArea -
                            intersectionArea
                        ).coerceAtLeast(1.0)

                val iou =
                    (
                        intersectionArea / unionArea
                    ).coerceIn(0.0, 1.0)

                candidates += Candidate(
                    meshIndex = meshIndex,
                    faceIndex = faceIndex,
                    score = (
                        centerScore * 0.60 +
                            iou * 0.40
                        ).coerceIn(0.0, 1.0),
                )
            }
        }

        val assignments = HashMap<Int, Int>()
        val usedFaces = HashSet<Int>()

        candidates
            .sortedByDescending { it.score }
            .forEach { candidate ->
                if (
                    candidate.meshIndex !in assignments &&
                    candidate.faceIndex !in usedFaces &&
                    candidate.score >= 0.24
                ) {
                    assignments[candidate.meshIndex] =
                        candidate.faceIndex
                    usedFaces += candidate.faceIndex
                }
            }

        return meshes.mapIndexed { index, mesh ->
            assignments[index]
                ?.let { faceIndex ->
                    mesh.copy(
                        trackingId = faces[faceIndex].trackingId,
                    )
                }
                ?: mesh
        }
    }

    private fun isFresh(
        lastTimestampNanos: Long,
        nowTimestampNanos: Long,
        maximumAgeNanos: Long,
    ): Boolean {
        if (lastTimestampNanos == Long.MIN_VALUE) return false
        val age = nowTimestampNanos - lastTimestampNanos
        return age in 0L..maximumAgeNanos
    }

    private fun contourIou(
        contour: FaceContourFrame,
        face: FaceFrame,
    ): Double {
        val left = max(
            contour.boundsLeft,
            face.boundsLeft,
        )
        val top = max(
            contour.boundsTop,
            face.boundsTop,
        )
        val right = min(
            contour.boundsRight,
            face.boundsRight,
        )
        val bottom = min(
            contour.boundsBottom,
            face.boundsBottom,
        )

        val intersection = (
            (right - left).coerceAtLeast(0f) *
                (bottom - top).coerceAtLeast(0f)
            )

        val contourArea = (
            (contour.boundsRight - contour.boundsLeft).coerceAtLeast(0f) *
                (contour.boundsBottom - contour.boundsTop).coerceAtLeast(0f)
            )

        val faceArea = (
            (face.boundsRight - face.boundsLeft).coerceAtLeast(0f) *
                (face.boundsBottom - face.boundsTop).coerceAtLeast(0f)
            )

        val union = contourArea + faceArea - intersection

        return if (union <= 1e-6f) {
            0.0
        } else {
            (intersection / union).toDouble()
        }
    }

    private fun emptyHandFrame(
        timestampNanos: Long,
    ): HandTrackingFrame =
        HandTrackingFrame(
            timestampNanos = timestampNanos,
            imageWidth = 1,
            imageHeight = 1,
            hands = emptyList(),
            inferenceLatencyMillis = 0.0,
            modelReady = false,
            droppedByBackpressure = false,
        )

    private fun emptyMotionQuality(): MotionCaptureEngine.Quality =
        MotionCaptureEngine.Quality(
            timestampNanos = 0L,
            motionScore = 0.0,
            motionClass = MotionCaptureEngine.MotionClass.STATIC,
            tuningResponsiveness = 1.0,
            predictionHorizonSeconds = 0.12,
            reacquisitionRadiusNormalized = 0.11,
            meanSpeedNormalizedPerSecond = 0.0,
            maxSpeedNormalizedPerSecond = 0.0,
            meanAccelerationNormalizedPerSecond2 = 0.0,
            maxAccelerationNormalizedPerSecond2 = 0.0,
            meanJerkNormalizedPerSecond3 = 0.0,
            jitterNormalized = 0.0,
            trackedLandmarks = 0,
            predictedLandmarks = 0,
            rejectedLandmarks = 0,
            meanConfidence = 0.0,
            occlusionRatio = 1.0,
            temporalStability = 0.0,
            spatialSpread = 0.0,
            stationaryRatio = 1.0,
            dynamicsConfidence = 0.0,
            meanInnovationNormalized = 0.0,
            rejectedMeasurementRatio = 1.0,
        )

    override fun close() {
        pose.close()
        face.close()
        contours.close()
        faceMesh.close()
        eyes.close()
        hands.close()
        objects.close()
        multiActorPose.close()
        subjectTracks.clear()
        multiActorAssociation.clear()
        faceTracks.clear()
        faceStabilizer.close()
        motion.close()
        motionFusion.reset()
        faceEyeFusionEngine.reset()
        subjectReidentification.reset()
        motionOrchestrator.reset()
        trackLifecycleController.reset()
        subjectLockEngine.reset()
        runtimeGovernor.reset()
        crowdMotionEvidenceEngine.reset()
        photometricMotionFieldEngine.close()
        temporalPhotometricConsensusEngine.close()
        personSegmentationEngine.close()
        denseLandmarkFusionEngine.close()
        denseContactEvidenceAdapter.close()
        subpixelLandmarkRefinementEngine.close()
        coordinator.reset()
        npuScale = 1.0
        thermalScale = 1.0
        memoryScale = 1.0
        gpuScale = 1.0
        lastContourFrame = null
        lastContourTimestampNanos = Long.MIN_VALUE
        lastFaces = emptyList()
        lastFacesTimestampNanos = Long.MIN_VALUE
        lastFaceMeshes = emptyList()
        lastFaceMeshesTimestampNanos = Long.MIN_VALUE
        lastObjects = emptyList()
        lastObjectsTimestampNanos = Long.MIN_VALUE
        lastHands = null
        lastHandsTimestampNanos = Long.MIN_VALUE
        lastPose = null
        lastEyeResults = emptyList()
        lastEyeResultsTimestampNanos = Long.MIN_VALUE
        denseActorSubjectSlots.clear()
    }
}
