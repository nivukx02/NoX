package com.nivukx.nox.core.ai

import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import com.nivukx.nox.core.model.ImageCoordinateTransform
import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.LandmarkSpace
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.hypot

/**
 * Hybrid body-motion front-end.
 *
 * The streaming detector handles temporal continuity and low latency. A separate accurate detector
 * periodically re-localizes the pose. Those measurements are blended on the keyframe rather than
 * forcing the expensive accurate model onto every captured frame.
 *
 * This separation is important for NoX because the same phone may also run face, iris and object
 * analysis while still recording a full-resolution video stream.
 */
class PoseDetectionEngine(
    private val accurateCorrectionInterval: Long = 10L,
) : AutoCloseable {

    data class Landmark(
        val id: Int,
        val name: String,
        val x: Float,
        val y: Float,
        val confidence: Float,
    )

    data class Result(
        val timestampNanos: Long,
        val imageWidth: Int,
        val imageHeight: Int,
        val landmarks: List<Landmark>,
        val confidence: Float,
        val boundingBoxLeft: Float = Float.NaN,
        val boundingBoxTop: Float = Float.NaN,
        val boundingBoxRight: Float = Float.NaN,
        val boundingBoxBottom: Float = Float.NaN,
        val accurateCorrectionApplied: Boolean = false,
        val frameIndex: Long = 0L,
        val rotationDegrees: Int = 0,
        val rollingShutterSkewNanos: Long = 0L,
        val frameDurationNanos: Long = 0L,
        val exposureTimeNanos: Long = 0L,
        val captureTimingConfidence: Double = 1.0,
        val staleEvidence: Boolean = false,
    ) {
        fun observations(): List<LandmarkObservation> = landmarks.map {
            LandmarkObservation(
                landmarkId = it.id,
                x = it.x.toDouble(),
                y = it.y.toDouble(),
                confidence = it.confidence.toDouble(),
                timestampNanos = timestampNanos,
                space = LandmarkSpace.PIXELS,
                frameWidth = imageWidth,
                frameHeight = imageHeight,
                rollingShutterSkewNanos = rollingShutterSkewNanos,
                frameDurationNanos = frameDurationNanos,
                exposureTimeNanos = exposureTimeNanos,
                captureTimingConfidence = captureTimingConfidence,
                sensorRowNormalized =
                    ImageCoordinateTransform
                        .detectorToNormalizedSensor(
                            point = Vec2(
                                x = it.x.toDouble(),
                                y = it.y.toDouble(),
                            ),
                            rawWidth = imageWidth,
                            rawHeight = imageHeight,
                            rotationDegrees = rotationDegrees,
                        ).y,
            )
        }
    }

    private val closed = AtomicBoolean(false)
    private val frameCounter = AtomicLong(0L)

    private val streamDetector = PoseDetection.getClient(
        PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
            .build(),
    )

    private val accurateDetector = PoseDetection.getClient(
        AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.SINGLE_IMAGE_MODE)
            .build(),
    )

    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
        rollingShutterSkewNanos: Long = 0L,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
        captureTimingConfidence: Double = 1.0,
    ): Task<Result> {
        check(!closed.get()) { "PoseDetectionEngine is closed" }

        val index = frameCounter.incrementAndGet()
        val width = image.width
        val height = image.height
        val input = InputImage.fromMediaImage(image, rotationDegrees)

        val streamTask = streamDetector.process(input)

        if (accurateCorrectionInterval <= 0L ||
            index % accurateCorrectionInterval != 0L
        ) {
            return streamTask.continueWith { task ->
                if (!task.isSuccessful) {
                    throw task.exception ?: IllegalStateException("Streaming pose detection failed")
                }
                mapResult(
                    pose = task.result,
                    width = width,
                    height = height,
                    timestampNanos = timestampNanos,
                    accurateCorrection = null,
                    frameIndex = index,
                    rotationDegrees = rotationDegrees,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                    captureTimingConfidence = captureTimingConfidence,
                )
            }
        }

        val accurateTask = accurateDetector.process(input)

        return Tasks.whenAllSuccess<Pose>(
            streamTask,
            accurateTask,
        ).continueWith { task ->
            if (!task.isSuccessful) {
                throw task.exception ?: IllegalStateException("Hybrid pose detection failed")
            }

            val streamPose = task.result[0]
            val accuratePose = task.result[1]

            mapResult(
                pose = streamPose,
                width = width,
                height = height,
                timestampNanos = timestampNanos,
                accurateCorrection = accuratePose,
                frameIndex = index,
                rotationDegrees = rotationDegrees,
                rollingShutterSkewNanos = rollingShutterSkewNanos,
                frameDurationNanos = frameDurationNanos,
                exposureTimeNanos = exposureTimeNanos,
                captureTimingConfidence = captureTimingConfidence,
            )
        }
    }

    fun normalized(result: Result): List<LandmarkObservation> {
        val sx = result.imageWidth.coerceAtLeast(1).toDouble()
        val sy = result.imageHeight.coerceAtLeast(1).toDouble()

        return result.landmarks.map {
            LandmarkObservation(
                landmarkId = it.id,
                x = (it.x.toDouble() / sx).coerceIn(0.0, 1.0),
                y = (it.y.toDouble() / sy).coerceIn(0.0, 1.0),
                confidence = it.confidence.toDouble(),
                timestampNanos = result.timestampNanos,
                space = LandmarkSpace.NORMALIZED,
                frameWidth = result.imageWidth,
                frameHeight = result.imageHeight,
                rollingShutterSkewNanos = result.rollingShutterSkewNanos,
                frameDurationNanos = result.frameDurationNanos,
                exposureTimeNanos = result.exposureTimeNanos,
                captureTimingConfidence = result.captureTimingConfidence,
                sensorRowNormalized =
                    ImageCoordinateTransform
                        .detectorToNormalizedSensor(
                            point = Vec2(
                                x = it.x.toDouble(),
                                y = it.y.toDouble(),
                            ),
                            rawWidth = result.imageWidth,
                            rawHeight = result.imageHeight,
                            rotationDegrees = result.rotationDegrees,
                        ).y,
            )
        }
    }

    fun motionMagnitude(
        previous: Result?,
        current: Result,
    ): Double {
        if (previous == null) return 0.0

        val previousMap = previous.landmarks.associateBy { it.id }
        val samples = current.landmarks.mapNotNull { now ->
            val old = previousMap[now.id]
                ?: return@mapNotNull null

            hypot(
                (now.x - old.x).toDouble(),
                (now.y - old.y).toDouble(),
            )
        }

        return if (samples.isEmpty()) 0.0 else samples.average()
    }

    private fun mapResult(
        pose: Pose,
        width: Int,
        height: Int,
        timestampNanos: Long,
        accurateCorrection: Pose?,
        frameIndex: Long,
        rotationDegrees: Int,
        rollingShutterSkewNanos: Long,
        frameDurationNanos: Long,
        exposureTimeNanos: Long,
        captureTimingConfidence: Double,
    ): Result {
        val stream = pose.allPoseLandmarks.associateBy { it.landmarkType }
        val accurate = accurateCorrection
            ?.allPoseLandmarks
            ?.associateBy { it.landmarkType }
            .orEmpty()

        val ids = (stream.keys + accurate.keys).distinct()
        val landmarks = ids.mapNotNull { id ->
            val s = stream[id]
            val a = accurate[id]

            if (s == null && a == null) {
                null
            } else {
                val useAccurate = a != null
                val blend =
                    if (useAccurate && s != null) 0.72f
                    else 1.0f

                val x = when {
                    s != null && a != null -> s.position.x * (1f - blend) + a.position.x * blend
                    a != null -> a.position.x
                    else -> s!!.position.x
                }

                val y = when {
                    s != null && a != null -> s.position.y * (1f - blend) + a.position.y * blend
                    a != null -> a.position.y
                    else -> s!!.position.y
                }

                val streamConfidence = s?.inFrameLikelihood ?: 0f
                val accurateConfidence = a?.inFrameLikelihood ?: 0f

                val confidence = when {
                    s != null && a != null ->
                        maxOf(streamConfidence, accurateConfidence * 0.98f)

                    a != null -> accurateConfidence
                    else -> streamConfidence
                }.coerceIn(0f, 1f)

                Landmark(
                    id = id,
                    name = id.toString(),
                    x = x,
                    y = y,
                    confidence = confidence,
                )
            }
        }

        val confident = landmarks.filter { it.confidence >= 0.20f }
        val averageConfidence =
            if (confident.isEmpty()) 0f
            else confident.map { it.confidence.toDouble() }.average().toFloat()

        val minX = confident.minOfOrNull { it.x } ?: Float.NaN
        val minY = confident.minOfOrNull { it.y } ?: Float.NaN
        val maxX = confident.maxOfOrNull { it.x } ?: Float.NaN
        val maxY = confident.maxOfOrNull { it.y } ?: Float.NaN

        return Result(
            timestampNanos = timestampNanos,
            imageWidth = width,
            imageHeight = height,
            landmarks = landmarks,
            confidence = averageConfidence,
            boundingBoxLeft = minX,
            boundingBoxTop = minY,
            boundingBoxRight = maxX,
            boundingBoxBottom = maxY,
            accurateCorrectionApplied = accurateCorrection != null,
            frameIndex = frameIndex,
            rotationDegrees = rotationDegrees,
            rollingShutterSkewNanos = rollingShutterSkewNanos.coerceAtLeast(0L),
            frameDurationNanos = frameDurationNanos.coerceAtLeast(0L),
            exposureTimeNanos = exposureTimeNanos.coerceAtLeast(0L),
            captureTimingConfidence = captureTimingConfidence.coerceIn(0.0, 1.0),
        )
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        streamDetector.close()
        accurateDetector.close()
    }
}
