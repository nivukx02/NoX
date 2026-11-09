package com.nivukx.nox.core.ai

import android.content.Context
import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.nivukx.nox.core.model.HandLandmark
import com.nivukx.nox.core.model.HandObservation
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.HandTrackingFrame
import com.nivukx.nox.core.model.Vec3
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Production hand-landmark inference for NoX.
 *
 * The engine:
 *  - uses the MediaPipe Hand Landmarker task for 21-point hand geometry,
 *  - loads a verified hand model packaged by the Gradle build pipeline,
 *  - converts only a bounded YUV working set to ARGB_8888 for MediaPipe,
 *  - runs inference off the Camera2 thread,
 *  - applies temporal wrist-velocity and confidence stabilization,
 *  - preserves metric world landmarks exposed by the model,
 *  - never performs network I/O during capture.
 *
 * Keeping model acquisition at build time makes runtime capture deterministic and removes
 * connectivity from the critical path of a synchronized take.
 */
class HandTrackingEngine(
    context: Context,
    private val maxHands: Int = 4,
    private val inferenceMaxDimension: Int = 960,
) : AutoCloseable {

    companion object {
        private const val MODEL_ASSET = "hand_landmarker.task"
        private const val MIN_HAND_CONFIDENCE = 0.50f
        private const val MIN_PRESENCE_CONFIDENCE = 0.50f
        private const val MIN_TRACKING_CONFIDENCE = 0.55f
        private const val STALE_TRACK_NANOS = 400_000_000L
        private const val ACTIVE_TRACK_NANOS = 250_000_000L
    }

    data class Readiness(
        val ready: Boolean,
        val modelPackaged: Boolean,
        val error: String?,
    )

    private data class MutableTrack(
        var id: Long,
        var side: HandSide,
        var wrist: Vec3,
        var lastTimestampNanos: Long,
        var velocity: Vec3,
        var confidence: Double,
        var stability: Double,
        var misses: Int,
    )

    private val appContext = context.applicationContext
    private val converter =
        Yuv420BitmapConverter(inferenceMaxDimension)
    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "NoX-HandInference").apply {
                priority = Thread.NORM_PRIORITY
            }
        }
    private val closed = AtomicBoolean(false)
    private val sequence = AtomicLong(0L)

    @Volatile
    private var landmarker: HandLandmarker? = null

    private val tracks = LinkedHashMap<Long, MutableTrack>()
    private var nextTrackId = 1L
    private var lastTimestampNanos = Long.MIN_VALUE

    @Volatile
    private var lastSuccessfulFrame: HandTrackingFrame? = null

    @Volatile
    private var lastInferenceLatencyMillis: Double = 0.0

    fun readiness(): Task<Readiness> =
        Tasks.call(executor, Callable {
            if (closed.get()) {
                Readiness(
                    ready = false,
                    modelPackaged = false,
                    error = "HandTrackingEngine is closed",
                )
            } else {
                val packaged = isModelPackaged()
                val initialized = packaged && ensureLandmarker() != null
                Readiness(
                    ready = initialized,
                    modelPackaged = packaged,
                    error = when {
                        !packaged -> "hand_landmarker.task is missing from module assets"
                        !initialized -> "MediaPipe Hand Landmarker initialization failed"
                        else -> null
                    },
                )
            }
        })

    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
        captureTimingConfidence: Double = 1.0,
        rollingShutterSkewNanos: Long = 0L,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
    ): Task<HandTrackingFrame> =
        Tasks.call(executor, Callable {
            val started = System.nanoTime()
            require(captureTimingConfidence in 0.0..1.0)
            require(rollingShutterSkewNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            if (closed.get()) {
                return@Callable HandTrackingFrame(
                    timestampNanos = timestampNanos,
                    imageWidth = image.width,
                    imageHeight = image.height,
                    hands = emptyList(),
                    inferenceLatencyMillis = 0.0,
                    modelReady = false,
                    captureTimingConfidence =
                        captureTimingConfidence,
                    rollingShutterSkewNanos =
                        rollingShutterSkewNanos,
                    frameDurationNanos =
                        frameDurationNanos,
                    exposureTimeNanos =
                        exposureTimeNanos,
                )
            }

            /*
             * MediaPipe VIDEO mode requires strictly increasing timestamps. Capture pipelines can
             * legitimately restart, seek, or swap clock domains, so recover by recreating the
             * stateful landmarker rather than allowing a timestamp regression to poison the session.
             */
            if (
                lastTimestampNanos != Long.MIN_VALUE &&
                timestampNanos <= lastTimestampNanos
            ) {
                runCatching { landmarker?.close() }
                landmarker = null
                synchronized(this@HandTrackingEngine) {
                    tracks.clear()
                    lastTimestampNanos = Long.MIN_VALUE
                }
            }

            val task = ensureLandmarker()
            if (task == null) {
                return@Callable HandTrackingFrame(
                    timestampNanos = timestampNanos,
                    imageWidth = image.width,
                    imageHeight = image.height,
                    hands = emptyList(),
                    inferenceLatencyMillis =
                        (System.nanoTime() - started) / 1e6,
                    modelReady = false,
                    captureTimingConfidence =
                        captureTimingConfidence,
                    rollingShutterSkewNanos =
                        rollingShutterSkewNanos,
                    frameDurationNanos =
                        frameDurationNanos,
                    exposureTimeNanos =
                        exposureTimeNanos,
                )
            }

            try {
                val bitmap =
                    converter.convert(image)
                val mpImage =
                    BitmapImageBuilder(bitmap).build()
                try {
                    val rotation =
                        ImageProcessingOptions.builder()
                            .setRotationDegrees(
                                normalizeRotation(rotationDegrees),
                            )
                            .build()

                    val result = task.detectForVideo(
                        mpImage,
                        rotation,
                        (
                            timestampNanos / 1_000_000L
                        ).coerceAtLeast(0L),
                    )

                    val hands = mapResult(
                        result = result,
                        timestampNanos = timestampNanos,
                    )

                    val latencyMillis =
                        (System.nanoTime() - started)
                            .coerceAtLeast(0L) / 1e6
                    lastInferenceLatencyMillis = latencyMillis

                    HandTrackingFrame(
                        timestampNanos = timestampNanos,
                        imageWidth = image.width,
                        imageHeight = image.height,
                        hands = hands,
                        inferenceLatencyMillis = latencyMillis,
                        modelReady = true,
                        captureTimingConfidence =
                            captureTimingConfidence,
                        rollingShutterSkewNanos =
                            rollingShutterSkewNanos,
                        frameDurationNanos =
                            frameDurationNanos,
                        exposureTimeNanos =
                            exposureTimeNanos,
                    ).also {
                        lastSuccessfulFrame = it
                    }
                } finally {
                    mpImage.close()
                }
            } catch (failure: Exception) {
                /*
                 * A detector failure must not take the full perception graph down. Drop only this
                 * modality, reset stateful MediaPipe VIDEO state, and keep the last verified frame
                 * marked as backpressure/stale rather than inventing new hand geometry.
                 */
                runCatching { landmarker?.close() }
                landmarker = null
                synchronized(this@HandTrackingEngine) {
                    tracks.clear()
                    lastTimestampNanos = Long.MIN_VALUE
                }

                val latencyMillis =
                    (System.nanoTime() - started)
                        .coerceAtLeast(0L) / 1e6
                val fallback =
                    lastSuccessfulFrame?.copy(
                        timestampNanos = timestampNanos,
                        imageWidth = image.width,
                        imageHeight = image.height,
                        inferenceLatencyMillis = latencyMillis,
                        droppedByBackpressure = true,
                        staleEvidence = true,
                        captureTimingConfidence =
                            captureTimingConfidence,
                        rollingShutterSkewNanos =
                            rollingShutterSkewNanos,
                        frameDurationNanos =
                            frameDurationNanos,
                        exposureTimeNanos =
                            exposureTimeNanos,
                    )

                fallback ?: HandTrackingFrame(
                    timestampNanos = timestampNanos,
                    imageWidth = image.width,
                    imageHeight = image.height,
                    hands = emptyList(),
                    inferenceLatencyMillis = latencyMillis,
                    modelReady = false,
                    droppedByBackpressure = true,
                    staleEvidence = true,
                    captureTimingConfidence =
                        captureTimingConfidence,
                    rollingShutterSkewNanos =
                        rollingShutterSkewNanos,
                    frameDurationNanos =
                        frameDurationNanos,
                    exposureTimeNanos =
                        exposureTimeNanos,
                )
            }
        })

    @Synchronized
    private fun ensureLandmarker(): HandLandmarker? {
        landmarker?.let { return it }
        if (!isModelPackaged()) return null

        val options =
            HandLandmarker.HandLandmarkerOptions
                .builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(MODEL_ASSET)
                        .build(),
                )
                .setNumHands(
                    maxHands.coerceIn(1, 4),
                )
                .setMinHandDetectionConfidence(
                    MIN_HAND_CONFIDENCE,
                )
                .setMinHandPresenceConfidence(
                    MIN_PRESENCE_CONFIDENCE,
                )
                .setMinTrackingConfidence(
                    MIN_TRACKING_CONFIDENCE,
                )
                .setRunningMode(
                    RunningMode.VIDEO,
                )
                .build()

        return runCatching {
            HandLandmarker.createFromOptions(
                appContext,
                options,
            )
        }.getOrNull()
            .also {
                landmarker = it
            }
    }

    private fun isModelPackaged(): Boolean =
        runCatching {
            appContext.assets.open(MODEL_ASSET).use {
                it.available() > 0
            }
        }.getOrDefault(false)

    private fun mapResult(
        result: HandLandmarkerResult,
        timestampNanos: Long,
    ): List<HandObservation> {
        val occupiedTrackIds =
            HashSet<Long>()

        val normalized = result.landmarks()
        val world = result.worldLandmarks()
        val handedness = result.handedness()

        val observations = normalized.indices.mapNotNull { index ->
            val pixels = normalized.getOrNull(index)
                ?: return@mapNotNull null
            if (pixels.size < 21) return@mapNotNull null

            val metric =
                world.getOrNull(index)

            val sideCategory =
                handedness.getOrNull(index)
                    ?.maxByOrNull { it.score() }

            val side =
                when (
                    sideCategory
                        ?.categoryName()
                        ?.uppercase()
                ) {
                    "LEFT" -> HandSide.LEFT
                    "RIGHT" -> HandSide.RIGHT
                    else -> HandSide.UNKNOWN
                }

            val sideConfidence =
                sideCategory
                    ?.score()
                    ?.toDouble()
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0

            val landmarks =
                pixels.mapIndexed { landmarkIndex, point ->
                    val worldPoint =
                        metric
                            ?.getOrNull(landmarkIndex)
                            ?.let {
                                com.nivukx.nox.core.model.Vec3(
                                    it.x().toDouble(),
                                    it.y().toDouble(),
                                    it.z().toDouble(),
                                )
                            }

                    HandLandmark(
                        index = landmarkIndex,
                        normalizedX = point.x().toDouble(),
                        normalizedY = point.y().toDouble(),
                        normalizedZ = point.z().toDouble(),
                        worldPositionMeters = worldPoint,
                        confidence = (
                            0.75 * sideConfidence +
                                0.25
                        ).coerceIn(0.0, 1.0),
                    )
                }

            val confidence =
                landmarks.map { it.confidence }.average()

            val wrist =
                landmarks.firstOrNull()
                    ?.worldPositionMeters
                    ?: com.nivukx.nox.core.model.Vec3(
                        pixels[0].x().toDouble(),
                        pixels[0].y().toDouble(),
                        pixels[0].z().toDouble(),
                    )

            val id =
                associateTrack(
                    side = side,
                    wrist = wrist,
                    confidence = confidence,
                    timestampNanos = timestampNanos,
                    occupiedTrackIds = occupiedTrackIds,
                )

            occupiedTrackIds += id

            HandObservation(
                handId = id,
                side = side,
                handednessConfidence = sideConfidence,
                landmarks = landmarks,
                confidence = confidence,
                timestampNanos = timestampNanos,
                wristVelocityMetersPerSecond =
                    tracks[id]?.velocity
                        ?: com.nivukx.nox.core.model.Vec3(
                            0.0,
                            0.0,
                            0.0,
                        ),
                temporalStability =
                    tracks[id]?.stability ?: confidence,
            )
        }

        pruneTracks(timestampNanos)
        lastTimestampNanos = timestampNanos
        sequence.incrementAndGet()
        return observations
    }

    @Synchronized
    private fun associateTrack(
        side: HandSide,
        wrist: com.nivukx.nox.core.model.Vec3,
        confidence: Double,
        timestampNanos: Long,
        occupiedTrackIds: Set<Long>,
    ): Long {
        val dt =
            if (lastTimestampNanos == Long.MIN_VALUE) {
                1.0 / 30.0
            } else {
                (
                    (timestampNanos - lastTimestampNanos)
                        .coerceAtLeast(1L) / 1e9
                ).coerceIn(1.0 / 240.0, 0.25)
            }

        var best: MutableTrack? = null
        var bestCost = Double.POSITIVE_INFINITY

        tracks.values.forEach { track ->
            if (track.id in occupiedTrackIds) {
                return@forEach
            }

            if (track.side != HandSide.UNKNOWN &&
                side != HandSide.UNKNOWN &&
                track.side != side
            ) {
                return@forEach
            }

            val predicted =
                track.wrist + track.velocity * dt
            val distance =
                predicted.distanceTo(wrist)
            val speedTerm =
                track.velocity.norm() * 0.08
            val confidenceTerm =
                (1.0 - confidence) * 0.10
            val cost =
                distance + speedTerm + confidenceTerm

            if (cost < bestCost &&
                distance <= 0.45
            ) {
                best = track
                bestCost = cost
            }
        }

        val selected =
            best ?: MutableTrack(
                id = nextTrackId++,
                side = side,
                wrist = wrist,
                lastTimestampNanos = timestampNanos,
                velocity =
                    com.nivukx.nox.core.model.Vec3(
                        0.0,
                        0.0,
                        0.0,
                    ),
                confidence = confidence,
                stability = confidence,
                misses = 0,
            ).also {
                tracks[it.id] = it
            }

        val displacement =
            wrist - selected.wrist
        val measuredVelocity =
            displacement / dt
        val velocityGain =
            (
                0.20 +
                    0.60 * confidence
            ).coerceIn(0.20, 0.80)

        selected.velocity =
            selected.velocity.lerp(
                measuredVelocity,
                velocityGain,
            )
        selected.wrist = wrist
        selected.lastTimestampNanos = timestampNanos
        selected.side = side
            .takeIf { it != HandSide.UNKNOWN }
            ?: selected.side
        selected.confidence =
            (
                selected.confidence * 0.30 +
                    confidence * 0.70
            ).coerceIn(0.0, 1.0)

        val innovation =
            displacement.norm()

        val stability =
            (
                1.0 -
                    min(
                        1.0,
                        innovation /
                            0.20,
                    )
            ).coerceIn(0.0, 1.0)

        selected.stability =
            selected.stability * 0.78 +
                stability * 0.22
        selected.misses = 0

        return selected.id
    }

    private fun pruneTracks(timestampNanos: Long) {
        val stale =
            tracks.values.filter {
                timestampNanos -
                    it.lastTimestampNanos >
                    STALE_TRACK_NANOS
            }

        stale.forEach {
            tracks.remove(it.id)
        }
    }

    fun snapshotDiagnostics(
        timestampNanos: Long,
        droppedFrames: Long = 0L,
    ): com.nivukx.nox.core.model.HandTrackingDiagnostics =
        synchronized(this) {
            val active =
                tracks.values.filter {
                    timestampNanos -
                        it.lastTimestampNanos <=
                        ACTIVE_TRACK_NANOS
                }

            com.nivukx.nox.core.model.HandTrackingDiagnostics(
                timestampNanos = timestampNanos,
                modelReady = landmarker != null,
                activeHands = active.size,
                meanConfidence =
                    if (active.isEmpty()) 0.0
                    else active.map { it.confidence }.average(),
                meanTemporalStability =
                    if (active.isEmpty()) 0.0
                    else active.map { it.stability }.average(),
                maxWristSpeedMetersPerSecond =
                    active.maxOfOrNull {
                        it.velocity.norm()
                    } ?: 0.0,
                inferenceLatencyMillis =
                    lastInferenceLatencyMillis,
                droppedFrames = droppedFrames,
                provider = "MediaPipe Hand Landmarker",
            )
        }

    @Synchronized
    fun reset() {
        tracks.clear()
        lastTimestampNanos = Long.MIN_VALUE
        lastSuccessfulFrame = null
        lastInferenceLatencyMillis = 0.0
    }

    private fun normalizeRotation(rotationDegrees: Int): Int {
        val normalized =
            (rotationDegrees % 360 + 360) % 360
        return ((normalized / 90) * 90).coerceIn(0, 270)
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        runCatching { landmarker?.close() }
        landmarker = null
        converter.close()
        reset()
        executor.shutdownNow()
    }
}