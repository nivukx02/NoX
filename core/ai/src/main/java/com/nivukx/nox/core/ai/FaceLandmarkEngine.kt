package com.nivukx.nox.core.ai

import android.graphics.PointF
import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.FaceMeshPoint
import com.nivukx.nox.core.model.Vec2
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * High-quality face analysis channel for the distributed face-camera role.
 *
 * The detector uses accurate landmarks, expression classification and persistent tracking.
 * Contour detection is intentionally not mixed into this real-time tracked path because ML Kit
 * documents that contour mode is limited to the most prominent face and does not pair usefully
 * with tracking. The result is therefore stable under the dedicated-face-camera workflow.
 */
class FaceLandmarkEngine : AutoCloseable {
    companion object {
        fun temporalQualityScale(
            captureTimingConfidence: Double,
            rollingShutterSkewNanos: Long,
            frameDurationNanos: Long,
            exposureTimeNanos: Long,
        ): Double {
            require(captureTimingConfidence in 0.0..1.0)
            require(rollingShutterSkewNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)

            val frameDuration = maxOf(
                frameDurationNanos,
                1_000_000L,
            ).toDouble()
            val readoutRatio = (
                rollingShutterSkewNanos.toDouble() /
                    frameDuration
            ).coerceAtLeast(0.0)
            val exposureRatio = (
                exposureTimeNanos.toDouble() /
                    frameDuration
            ).coerceIn(0.0, 1.0)

            return (
                captureTimingConfidence *
                    exp(
                        -0.28 * readoutRatio -
                            0.18 * exposureRatio,
                    )
                ).coerceIn(0.05, 1.0)
        }
    }
    private val closed = AtomicBoolean(false)

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .setMinFaceSize(0.08f)
            .build(),
    )

    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
        captureTimingConfidence: Double = 1.0,
        rollingShutterSkewNanos: Long = 0L,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
    ): Task<List<FaceFrame>> {
        check(!closed.get()) { "FaceLandmarkEngine is closed" }
        val input = InputImage.fromMediaImage(image, rotationDegrees)
        return detector.process(input).continueWith { task ->
            if (!task.isSuccessful) {
                throw task.exception ?: IllegalStateException("Face detection failed")
            }
            val timingQuality = temporalQualityScale(
                captureTimingConfidence = captureTimingConfidence,
                rollingShutterSkewNanos = rollingShutterSkewNanos,
                frameDurationNanos = frameDurationNanos,
                exposureTimeNanos = exposureTimeNanos,
            )
            task.result.map {
                mapFace(
                    face = it,
                    timestampNanos = timestampNanos,
                    timingQuality = timingQuality,
                    captureTimingConfidence = captureTimingConfidence,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                )
            }
        }
    }

    private fun mapFace(
        face: Face,
        timestampNanos: Long,
        timingQuality: Double,
        captureTimingConfidence: Double,
        rollingShutterSkewNanos: Long,
        frameDurationNanos: Long,
        exposureTimeNanos: Long,
    ): FaceFrame {
        val points = mutableListOf<FaceMeshPoint>()
        val types = listOf(
            FaceLandmark.LEFT_EYE,
            FaceLandmark.RIGHT_EYE,
            FaceLandmark.NOSE_BASE,
            FaceLandmark.LEFT_EAR,
            FaceLandmark.RIGHT_EAR,
            FaceLandmark.LEFT_CHEEK,
            FaceLandmark.RIGHT_CHEEK,
            FaceLandmark.MOUTH_LEFT,
            FaceLandmark.MOUTH_RIGHT,
        )

        var index = 0
        var leftEyeCenter: Vec2? = null
        var rightEyeCenter: Vec2? = null

        types.forEach { type ->
            val landmark = face.getLandmark(type) ?: return@forEach
            val p: PointF = landmark.position
            val point = Vec2(p.x.toDouble(), p.y.toDouble())

            if (type == FaceLandmark.LEFT_EYE) leftEyeCenter = point
            if (type == FaceLandmark.RIGHT_EYE) rightEyeCenter = point

            points += FaceMeshPoint(
                index = index++,
                point = point,
                confidence = timingQuality,
            )
        }

        val leftOpen = (face.leftEyeOpenProbability ?: 1f)
            .toDouble()
            .coerceIn(0.0, 1.0)
        val rightOpen = (face.rightEyeOpenProbability ?: 1f)
            .toDouble()
            .coerceIn(0.0, 1.0)

        return FaceFrame(
            timestampNanos = timestampNanos,
            trackingId = face.trackingId,
            boundsLeft = face.boundingBox.left.toFloat(),
            boundsTop = face.boundingBox.top.toFloat(),
            boundsRight = face.boundingBox.right.toFloat(),
            boundsBottom = face.boundingBox.bottom.toFloat(),
            yawDegrees = face.headEulerAngleY,
            rollDegrees = face.headEulerAngleZ,
            points = points,
            leftEyeCenter = leftEyeCenter,
            rightEyeCenter = rightEyeCenter,
            leftEyeOpenProbability = leftOpen,
            rightEyeOpenProbability = rightOpen,
            captureTimingConfidence =
                captureTimingConfidence.coerceIn(0.0, 1.0),
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )
    }

    fun faceScale(frame: FaceFrame): Double =
        sqrt(
            ((frame.boundsRight - frame.boundsLeft).toDouble()).let { it * it } +
                ((frame.boundsBottom - frame.boundsTop).toDouble()).let { it * it },
        )

    override fun close() {
        if (closed.compareAndSet(false, true)) detector.close()
    }
}
