package com.nivukx.nox.core.ai

import android.content.Context
import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.mediapipe.framework.image.MediaImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.nivukx.nox.core.model.SubjectType
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Production multi-person body-pose front end backed by MediaPipe Pose Landmarker.
 *
 * Google documents `numPoses` as the maximum number of bodies returned by the task and exposes both
 * normalized image landmarks and metric world landmarks. This class keeps inference asynchronous
 * on a dedicated single-thread executor so the stateful VIDEO-mode task receives frames in strict
 * timestamp order without concurrent access to the same landmarker instance.
 */
class MultiActorPoseLandmarkerEngine(
    context: Context,
    private val config: Config = Config(),
    modelAssetPath: String = DEFAULT_MODEL_ASSET_PATH,
) : AutoCloseable {
    data class Config(
        val maxPoses: Int = 64,
        val minDetectionConfidence: Float = 0.45f,
        val minTrackingConfidence: Float = 0.45f,
        val minPresenceConfidence: Float = 0.45f,
        val executorThreads: Int = 1,
        val minimumTemporalGapMillis: Long = 1L,
        val outputSegmentationMasks: Boolean = true,
    ) {
        init {
            require(maxPoses in 1..64)
            require(minDetectionConfidence in 0f..1f)
            require(minTrackingConfidence in 0f..1f)
            require(minPresenceConfidence in 0f..1f)
            require(executorThreads == 1) {
                "Stateful VIDEO-mode pose inference requires exactly one worker."
            }
            require(minimumTemporalGapMillis >= 1L)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val inputWidth: Int,
        val inputHeight: Int,
        val rotationDegrees: Int,
        val observations: List<MultiActorPoseAssociationEngine.Observation>,
        val inferenceTimeMillis: Long,
        val detectorPoseCount: Int,
        val landmarkerTimestampMillis: Long,
        val timestampAdjusted: Boolean,
        val temporalTrackingUsed: Boolean = true,
        val identityContinuityScore: Double = 0.0,
        val identityCreatedTrackCount: Int = 0,
        val identityReidentifiedTrackCount: Int = 0,
        val identityPredictedTrackCount: Int = 0,
        val identityDroppedObservationCount: Int = 0,
        val identityPredictionConfidence: Double = 0.0,
        val identityMeanInnovationNormalized: Double = 0.0,
        val identityAcceleratedTrackCount: Int = 0,
        val identityAssignmentBounded: Boolean = false,
        val poseSegmentationEvidence:
            Map<Long, PoseMaskEvidenceEngine.Result> =
            emptyMap(),
        val warnings: List<String>,
    )

    private val closed = AtomicBoolean(false)
    private val resetRequested = AtomicBoolean(false)
    private val timestampSequencer = MonotonicTimestampSequencer(
        minimumGapMillis = config.minimumTemporalGapMillis,
    )
    private val appearanceDescriptorEngine =
        YuvTrackAppearanceDescriptorEngine()

    private val poseMaskEvidenceEngine =
        PoseMaskEvidenceEngine()

    private val identityTracker =
        TemporalPoseIdentityTracker(
            TemporalPoseIdentityTracker.Config(
                maximumTracks = config.maxPoses,
            ),
        )
    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val landmarkerContext = context.applicationContext
    private val modelPath = modelAssetPath
    private var landmarker: PoseLandmarker? = null
    private var landmarkerInitializationWarning: String? = null

    init {
        initializeLandmarker()
    }

    private fun initializeLandmarker() {
        if (landmarker != null) return
        val created =
            runCatching {
                createLandmarker()
            }.onFailure { error ->
                landmarkerInitializationWarning =
                    "Pose Landmarker unavailable: " +
                        (error.message
                            ?.takeIf { it.isNotBlank() }
                            ?: error.javaClass.simpleName)
            }.getOrNull()
        landmarker = created
    }

    private fun createLandmarker(): PoseLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(modelPath)
            .build()

        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setMinPoseDetectionConfidence(config.minDetectionConfidence)
            .setMinTrackingConfidence(config.minTrackingConfidence)
            .setMinPosePresenceConfidence(config.minPresenceConfidence)
            .setNumPoses(config.maxPoses)
            .setOutputSegmentationMasks(
                config.outputSegmentationMasks,
            )
            .setRunningMode(RunningMode.VIDEO)
            .build()

        return PoseLandmarker.createFromOptions(
            landmarkerContext,
            options,
        )
    }

    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
    ): Task<Result> {
        check(!closed.get()) { "MultiActorPoseLandmarkerEngine is closed" }
        require(timestampNanos >= 0L)

        val ticket = timestampSequencer.next(
            sourceTimestampNanos = timestampNanos,
        )
        val startNanos = System.nanoTime()
        val inputWidth = image.width
        val inputHeight = image.height
        val mpImage: MPImage =
            MediaImageBuilder(image).build()
        val processingOptions = ImageProcessingOptions.builder()
            .setRotationDegrees(rotationDegrees)
            .build()

        return Tasks.call(executor) {
            if (resetRequested.compareAndSet(true, false)) {
                runCatching {
                    landmarker?.close()
                }
                landmarker = null
                landmarkerInitializationWarning = null
                identityTracker.reset()
                initializeLandmarker()
            }

            val activeLandmarker =
                landmarker
            if (activeLandmarker == null) {
                return@call unavailableResult(
                    timestampNanos = timestampNanos,
                    inputWidth = inputWidth,
                    inputHeight = inputHeight,
                    rotationDegrees = rotationDegrees,
                    timestampMillis = ticket.timestampMillis,
                    timestampAdjusted = ticket.adjusted,
                )
            }

            try {
                val result = activeLandmarker.detectForVideo(
                    mpImage,
                    processingOptions,
                    ticket.timestampMillis,
                )

                mapResult(
                    result = result,
                    timestampNanos = timestampNanos,
                    inputWidth = inputWidth,
                    inputHeight = inputHeight,
                    rotationDegrees = rotationDegrees,
                    sourceImage = image,
                    landmarkerTimestampMillis = ticket.timestampMillis,
                    timestampAdjusted = ticket.adjusted,
                    inferenceTimeMillis =
                        ((System.nanoTime() - startNanos) / 1_000_000L)
                            .coerceAtLeast(0L),
                )
            } finally {
                runCatching {
                    mpImage.close()
                }
            }
        }
    }

    /**
     * Requests a temporal state reset at the next serialized inference boundary.
     *
     * The reset is intentionally deferred onto the same executor that owns the landmarker. This
     * avoids closing/replacing a MediaPipe task while another frame is being processed.
     */
    fun resetForNewSequence() {
        if (closed.get()) return
        timestampSequencer.reset()
        resetRequested.set(true)
    }

    private fun unavailableResult(
        timestampNanos: Long,
        inputWidth: Int,
        inputHeight: Int,
        rotationDegrees: Int,
        timestampMillis: Long,
        timestampAdjusted: Boolean,
    ): Result =
        Result(
            timestampNanos = timestampNanos,
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            rotationDegrees = rotationDegrees,
            observations = emptyList(),
            inferenceTimeMillis = 0L,
            detectorPoseCount = 0,
            landmarkerTimestampMillis = timestampMillis,
            timestampAdjusted = timestampAdjusted,
            temporalTrackingUsed = false,
            warnings = listOf(
                landmarkerInitializationWarning
                    ?: "Pose Landmarker unavailable; continuing with non-pose evidence.",
            ),
        )

    private fun mapResult(
        result: PoseLandmarkerResult,
        timestampNanos: Long,
        inputWidth: Int,
        inputHeight: Int,
        rotationDegrees: Int,
        sourceImage: Image,
        landmarkerTimestampMillis: Long,
        timestampAdjusted: Boolean,
        inferenceTimeMillis: Long,
    ): Result {
        val rawObservations =
            result.landmarks().mapIndexedNotNull { poseIndex, landmarks ->
                mapPose(
                    poseIndex = poseIndex,
                    landmarks = landmarks,
                    worldLandmarks =
                        result.worldLandmarks()
                            .getOrNull(poseIndex)
                            .orEmpty(),
                )
            }

        val poseSegmentationMasks =
            result
                .segmentationMasks()
                .orElse(emptyList())

        val poseMaskEvidenceByObservationIndex =
            try {
                rawObservations
                    .mapIndexedNotNull { index, observation ->
                        val evidence =
                            poseMaskEvidenceEngine.evaluate(
                                mask =
                                    poseSegmentationMasks
                                        .getOrNull(index),
                                bounds = observation.bounds,
                                joints = observation.joints,
                            )
                        if (evidence.validSamples <= 0) {
                            null
                        } else {
                            observation.sourceObservationId to evidence
                        }
                    }
                    .toMap()
            } finally {
                poseSegmentationMasks.forEach { mask ->
                    runCatching { mask.close() }
                }
            }

        val appearanceObservations =
            rawObservations.map { observation ->
                val descriptor =
                    runCatching {
                        appearanceDescriptorEngine.describe(
                            image = sourceImage,
                            bounds = observation.bounds,
                            rotationDegrees = rotationDegrees,
                        )
                    }.getOrNull()

                if (descriptor == null) {
                    observation
                } else {
                    observation.copy(
                        appearanceSignature =
                            descriptor.signature,
                        appearanceFingerprint =
                            descriptor.networkFingerprint,
                        appearanceQuality =
                            descriptor.quality,
                    )
                }
            }

        val identityResult =
            identityTracker.update(
                timestampNanos = timestampNanos,
                detections = appearanceObservations.map { observation ->
                    TemporalPoseIdentityTracker.Observation(
                        center = observation.center,
                        width = (
                            observation.bounds.right -
                                observation.bounds.left
                        ).coerceAtLeast(1e-6),
                        height = (
                            observation.bounds.bottom -
                                observation.bounds.top
                        ).coerceAtLeast(1e-6),
                        confidence =
                            observation.confidence.coerceIn(
                                0.0,
                                1.0,
                            ),
                        detectorIndex = observation.sourceObservationId,
                        shapeSignature =
                            observation.shapeSignature,
                        appearanceSignature =
                            observation.appearanceSignature,
                        appearanceQuality =
                            observation.appearanceQuality
                                .coerceIn(0.0, 1.0),
                    )
                },
            )

        val rawByDetectorIndex =
            appearanceObservations.associateBy {
                it.sourceObservationId
            }

        val poseSegmentationEvidence =
            identityResult.observations
                .mapNotNull { assigned ->
                    val evidence =
                        poseMaskEvidenceByObservationIndex[
                            assigned.observation.detectorIndex
                        ] ?: return@mapNotNull null
                    assigned.localTrackId to evidence
                }
                .toMap()

        val observations =
            identityResult.observations.mapNotNull { assigned ->
                val source =
                    rawByDetectorIndex[
                        assigned.observation.detectorIndex
                    ] ?: return@mapNotNull null

                source.copy(
                    detectorTrackingId =
                        assigned.localTrackId
                            .coerceIn(
                                Int.MIN_VALUE.toLong(),
                                Int.MAX_VALUE.toLong(),
                            )
                            .toInt(),
                )
            }

        val warnings = buildList {
            if (observations.size < result.landmarks().size) {
                add("MULTI_ACTOR_POSE_WITHOUT_USABLE_LANDMARKS")
            }
            if (observations.size >= config.maxPoses) {
                add("MULTI_ACTOR_POSE_CAP_REACHED")
            }
            if (observations.none { it.worldJoints.isNotEmpty() } &&
                observations.isNotEmpty()
            ) {
                add("MULTI_ACTOR_WORLD_LANDMARK_EVIDENCE_UNAVAILABLE")
            }
            addAll(identityResult.warnings)
            if (identityResult.droppedObservationCount > 0) {
                add(
                    "MULTI_ACTOR_POSE_IDENTITY_CAP_DROPPED_OBSERVATIONS",
                )
            }
        }

        return Result(
            timestampNanos = timestampNanos,
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            rotationDegrees = rotationDegrees,
            observations = observations,
            inferenceTimeMillis = inferenceTimeMillis,
            detectorPoseCount = result.landmarks().size,
            landmarkerTimestampMillis = landmarkerTimestampMillis,
            timestampAdjusted = timestampAdjusted,
            identityContinuityScore =
                identityResult.identityContinuityScore,
            identityCreatedTrackCount =
                identityResult.createdTrackCount,
            identityReidentifiedTrackCount =
                identityResult.reidentifiedTrackCount,
            identityPredictedTrackCount =
                identityResult.predictedTrackCount,
            identityDroppedObservationCount =
                identityResult.droppedObservationCount,
            identityPredictionConfidence =
                identityResult.predictionConfidence,
            identityMeanInnovationNormalized =
                identityResult.meanInnovationNormalized,
            identityAcceleratedTrackCount =
                identityResult.acceleratedTrackCount,
            identityAssignmentBounded =
                identityResult.globalAssignmentBounded,
            poseSegmentationEvidence =
                poseSegmentationEvidence,
            warnings = buildList {
                if (
                    observations.count {
                        it.appearanceQuality >= 0.20
                    } <
                    observations.size / 2
                ) {
                    add(
                        "MULTI_ACTOR_APPEARANCE_EVIDENCE_LOW",
                    )
                }
                if (timestampAdjusted) {
                    add("MULTI_ACTOR_POSE_TIMESTAMP_ADJUSTED_TO_MONOTONIC")
                }
                addAll(warnings)
            }.distinct(),
        )
    }

    private fun mapPose(
        poseIndex: Int,
        landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>,
        worldLandmarks:
            List<com.google.mediapipe.tasks.components.containers.Landmark> =
            emptyList(),
    ): MultiActorPoseAssociationEngine.Observation? {
        if (landmarks.isEmpty()) return null

        val joints = landmarks.mapIndexed { index, landmark ->
            val visibility = landmark.visibility().orElse(1.0f).coerceIn(0f, 1f)
            val presence = landmark.presence().orElse(1.0f).coerceIn(0f, 1f)
            MultiActorPoseAssociationEngine.Joint(
                landmarkId = index,
                x = landmark.x().toDouble().coerceIn(0.0, 1.0),
                y = landmark.y().toDouble().coerceIn(0.0, 1.0),
                confidence = (visibility * presence).toDouble().coerceIn(0.0, 1.0),
            )
        }

        val usable = joints.filter { it.confidence >= 0.15 }
        if (usable.size < 4) return null

        val left = usable.minOf { it.x }
        val top = usable.minOf { it.y }
        val right = usable.maxOf { it.x }
        val bottom = usable.maxOf { it.y }
        val center = com.nivukx.nox.core.model.Vec2(
            (left + right) * 0.5,
            (top + bottom) * 0.5,
        )
        val confidence = usable.map { it.confidence }.average()
        val shapeWidth =
            (right - left).coerceAtLeast(1e-6)
        val shapeHeight =
            (bottom - top).coerceAtLeast(1e-6)
        val shapeSignature =
            joints
                .take(33)
                .flatMap { joint ->
                    listOf(
                        (
                            (joint.x - center.x) /
                                shapeWidth
                            ).coerceIn(-2.0, 2.0),
                        (
                            (joint.y - center.y) /
                                shapeHeight
                            ).coerceIn(-2.0, 2.0),
                        joint.confidence.coerceIn(0.0, 1.0),
                    )
                }

        val mappedWorldJoints =
            worldLandmarks.mapIndexedNotNull { index, landmark ->
                val x = landmark.x()
                val y = landmark.y()
                val z = landmark.z()
                if (!x.isFinite() || !y.isFinite() || !z.isFinite()) {
                    null
                } else {
                    val visibility =
                        landmark.visibility()
                            .orElse(1.0f)
                            .coerceIn(0.0f, 1.0f)
                    val presence =
                        landmark.presence()
                            .orElse(1.0f)
                            .coerceIn(0.0f, 1.0f)
                    MultiActorPoseAssociationEngine.WorldJoint(
                        landmarkId = index,
                        xMeters = x.toDouble(),
                        yMeters = y.toDouble(),
                        zMeters = z.toDouble(),
                        confidence =
                            (visibility * presence)
                                .toDouble()
                                .coerceIn(0.0, 1.0),
                    )
                }
            }

        return MultiActorPoseAssociationEngine.Observation(
            sourceObservationId = poseIndex,
            detectorTrackingId = poseIndex,
            subjectType = SubjectType.HUMAN,
            center = center,
            bounds = MultiActorPoseAssociationEngine.Bounds(
                left = left,
                top = top,
                right = right,
                bottom = bottom,
            ),
            confidence = confidence,
            joints = joints,
            shapeSignature = shapeSignature,
            worldJoints = mappedWorldJoints,
        )
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return

        resetRequested.set(false)
        timestampSequencer.reset()

        executor.shutdown()

        val terminated =
            runCatching {
                executor.awaitTermination(
                    750L,
                    TimeUnit.MILLISECONDS,
                )
            }.getOrDefault(false)

        if (!terminated) {
            executor.shutdownNow()
            runCatching {
                executor.awaitTermination(
                    250L,
                    TimeUnit.MILLISECONDS,
                )
            }
        }

        identityTracker.reset()
        runCatching {
            landmarker?.close()
        }
        landmarker = null
    }

    companion object {
        const val DEFAULT_MODEL_ASSET_PATH = "pose_landmarker_full.task"
    }
}
