package com.nivukx.nox.core.ai

import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.nivukx.nox.core.model.FaceContourFrame
import com.nivukx.nox.core.model.FaceContourKind
import com.nivukx.nox.core.model.FaceContour as NoXFaceContour
import com.nivukx.nox.core.model.Vec2
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-detail facial geometry path.
 *
 * ML Kit contour mode provides ordered 2D points for the face oval, eyes, brows, nose and lips.
 * This engine intentionally stays separate from the tracked landmark/classification detector
 * because contour mode is documented as a prominent-face path rather than a multi-face tracker.
 */
class FaceContourEngine : AutoCloseable {
    private val closed = AtomicBoolean(false)

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setMinFaceSize(0.10f)
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
        rollingShutterDirectionSign: Int = 1,
    ): Task<FaceContourFrame?> {
        require(rollingShutterDirectionSign == 1 || rollingShutterDirectionSign == -1)
        require(captureTimingConfidence in 0.0..1.0)
        require(rollingShutterSkewNanos >= 0L)
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        check(!closed.get()) { "FaceContourEngine is closed" }
        val input = InputImage.fromMediaImage(image, rotationDegrees)

        return detector.process(input).continueWith { task ->
            if (!task.isSuccessful) {
                throw task.exception ?: IllegalStateException("Face contour detection failed")
            }

            task.result.firstOrNull()?.let { face ->
                mapFace(
                    face = face,
                    timestampNanos = timestampNanos,
                    imageWidth = image.width,
                    imageHeight = image.height,
                    rotationDegrees = rotationDegrees,
                    captureTimingConfidence = captureTimingConfidence,
                    rollingShutterSkewNanos = rollingShutterSkewNanos,
                    frameDurationNanos = frameDurationNanos,
                    exposureTimeNanos = exposureTimeNanos,
                    rollingShutterDirectionSign = rollingShutterDirectionSign,
                )
            }
        }
    }

    private fun mapFace(
        face: Face,
        timestampNanos: Long,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int,
        captureTimingConfidence: Double,
        rollingShutterSkewNanos: Long,
        frameDurationNanos: Long,
        exposureTimeNanos: Long,
        rollingShutterDirectionSign: Int,
    ): FaceContourFrame {
        val mappings = listOf(
            FaceContour.FACE to FaceContourKind.FACE_OVAL,
            FaceContour.LEFT_EYE to FaceContourKind.LEFT_EYE,
            FaceContour.RIGHT_EYE to FaceContourKind.RIGHT_EYE,
            FaceContour.LEFT_EYEBROW_TOP to FaceContourKind.LEFT_EYEBROW_TOP,
            FaceContour.RIGHT_EYEBROW_TOP to FaceContourKind.RIGHT_EYEBROW_TOP,
            FaceContour.NOSE_BRIDGE to FaceContourKind.NOSE_BRIDGE,
            FaceContour.NOSE_BOTTOM to FaceContourKind.NOSE_BOTTOM,
            FaceContour.UPPER_LIP_TOP to FaceContourKind.UPPER_LIP_TOP,
            FaceContour.UPPER_LIP_BOTTOM to FaceContourKind.UPPER_LIP_BOTTOM,
            FaceContour.LOWER_LIP_TOP to FaceContourKind.LOWER_LIP_TOP,
            FaceContour.LOWER_LIP_BOTTOM to FaceContourKind.LOWER_LIP_BOTTOM,
        )

        val contours = mappings.mapNotNull { (mlKind, noxKind) ->
            val points = face.getContour(mlKind)?.points.orEmpty()
            if (points.isEmpty()) return@mapNotNull null

            NoXFaceContour(
                kind = noxKind,
                points = points.map { Vec2(it.x.toDouble(), it.y.toDouble()) },
                confidence = 1.0,
            )
        }

        return FaceContourFrame(
            timestampNanos = timestampNanos,
            trackingId = face.trackingId,
            boundsLeft = face.boundingBox.left.toFloat(),
            boundsTop = face.boundingBox.top.toFloat(),
            boundsRight = face.boundingBox.right.toFloat(),
            boundsBottom = face.boundingBox.bottom.toFloat(),
            yawDegrees = face.headEulerAngleY,
            rollDegrees = face.headEulerAngleZ,
            contours = contours,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = rotationDegrees,
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
            rollingShutterDirectionSign = rollingShutterDirectionSign,
        )
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) detector.close()
    }
}
