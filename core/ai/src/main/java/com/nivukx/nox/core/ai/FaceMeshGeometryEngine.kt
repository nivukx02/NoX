package com.nivukx.nox.core.ai

import android.media.Image
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.common.Triangle as MlKitTriangle
import com.google.mlkit.vision.facemesh.FaceMesh
import com.google.mlkit.vision.facemesh.FaceMeshPoint
import com.google.mlkit.vision.facemesh.FaceMeshDetection
import com.google.mlkit.vision.facemesh.FaceMeshDetector
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Dense close-range facial geometry channel.
 *
 * ML Kit Face Mesh provides a stable 0..467 point index space and triangle topology. The output is
 * intentionally expressed in image-space units because its Z coordinate is scaled to image size;
 * it must not be interpreted as metric depth. Downstream reconstruction decides whether and how to
 * combine this geometry with calibrated camera intrinsics, triangulated face geometry, or temporal
 * constraints.
 *
 * The engine is auxiliary to FaceLandmarkEngine:
 *  - FaceLandmarkEngine supplies tracked identity, head orientation and eye-open classification.
 *  - FaceContourEngine supplies semantic contour geometry.
 *  - FaceMeshGeometryEngine supplies dense 3D surface evidence when the face is close enough.
 *
 * This separation prevents a close-range mesh detector from becoming a single point of failure for
 * multi-subject tracking or longer camera-to-subject distances.
 */
class FaceMeshGeometryEngine : AutoCloseable {

    data class Point(
        val index: Int,
        val positionImageSpace: Vec3,
        val confidence: Double,
    )

    data class Triangle(
        val a: Int,
        val b: Int,
        val c: Int,
    )

    data class Result(
        val timestampNanos: Long,
        val boundsLeft: Float,
        val boundsTop: Float,
        val boundsRight: Float,
        val boundsBottom: Float,
        val points: List<Point>,
        val triangles: List<Triangle>,
        val imageWidth: Int,
        val imageHeight: Int,
        val rotationDegrees: Int,
        val trackingId: Int? = null,
        val confidence: Double,
        val pointCoverage: Double,
        val triangleCoverage: Double,
        val faceAreaRatio: Double,
        val depthSpreadImageUnits: Double,
    ) {
        val pointCount: Int
            get() = points.size

        val triangleCount: Int
            get() = triangles.size

        val hasDenseGeometry: Boolean
            get() = points.size >= 400 && triangles.size >= 500

        val centerImageSpace: Vec3
            get() {
                if (points.isEmpty()) {
                    return Vec3(0.0, 0.0, 0.0)
                }
                return Vec3(
                    points.map { it.positionImageSpace.x }.average(),
                    points.map { it.positionImageSpace.y }.average(),
                    points.map { it.positionImageSpace.z }.average(),
                )
            }

        /**
         * Converts the detector output into NoX's detector-neutral face-mesh contract.
         *
         * ML Kit's Z value is image-space scaled rather than metric, so this conversion intentionally
         * preserves it as image-space geometry. Metric world-space reconstruction remains the job of
         * FaceMeshMultiViewFusionEngine and its calibrated triangulation stage.
         */
        fun toModelFrame(): FaceMeshFrame =
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

        val meanPointRadiusImageUnits: Double
            get() {
                val center = centerImageSpace
                if (points.isEmpty()) return 0.0
                return points
                    .map {
                        val dx = it.positionImageSpace.x - center.x
                        val dy = it.positionImageSpace.y - center.y
                        val dz = it.positionImageSpace.z - center.z
                        sqrt(dx * dx + dy * dy + dz * dz)
                    }
                    .average()
            }
    }

    private val closed = AtomicBoolean(false)

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
            val readoutRatio =
                rollingShutterSkewNanos.toDouble() / frameDuration
            val exposureRatio =
                (
                    exposureTimeNanos.toDouble() / frameDuration
                ).coerceIn(0.0, 1.0)

            return (
                captureTimingConfidence *
                    exp(
                        -0.22 * readoutRatio -
                            0.14 * exposureRatio,
                    )
                ).coerceIn(0.05, 1.0)
        }
    }

    private val detector: FaceMeshDetector =
        FaceMeshDetection.getClient()

    fun process(
        image: Image,
        rotationDegrees: Int,
        timestampNanos: Long,
        captureTimingConfidence: Double = 1.0,
        rollingShutterSkewNanos: Long = 0L,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
    ): Task<List<Result>> {
        check(!closed.get()) { "FaceMeshGeometryEngine is closed" }

        val input = InputImage.fromMediaImage(
            image,
            rotationDegrees,
        )

        return detector.process(input)
            .continueWith { task ->
                if (!task.isSuccessful) {
                    throw task.exception
                        ?: IllegalStateException(
                            "Face mesh detection failed",
                        )
                }

                task.result.map { face ->
                    mapFace(
                        face = face,
                        imageWidth = image.width,
                        imageHeight = image.height,
                        timestampNanos = timestampNanos,
                        rotationDegrees = rotationDegrees,
                        captureTimingConfidence = captureTimingConfidence,
                        rollingShutterSkewNanos = rollingShutterSkewNanos,
                        frameDurationNanos = frameDurationNanos,
                        exposureTimeNanos = exposureTimeNanos,
                    )
                }
            }
    }

    private fun mapFace(
        face: FaceMesh,
        imageWidth: Int,
        imageHeight: Int,
        timestampNanos: Long,
        rotationDegrees: Int,
        captureTimingConfidence: Double,
        rollingShutterSkewNanos: Long,
        frameDurationNanos: Long,
        exposureTimeNanos: Long,
    ): Result {
        val timingQuality = temporalQualityScale(
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )
        val rawPoints = face.allPoints
        val pointsByIndex = rawPoints.associateBy { it.index }

        val points = rawPoints.mapNotNull { point ->
            val position = point.position
            val x = position.x.toDouble()
            val y = position.y.toDouble()
            val z = position.z.toDouble()

            if (
                !x.isFinite() ||
                !y.isFinite() ||
                !z.isFinite()
            ) {
                null
            } else {
                Point(
                    index = point.index,
                    positionImageSpace = Vec3(
                        x = x,
                        y = y,
                        z = z,
                    ),
                    confidence = timingQuality,
                )
            }
        }

        val triangles =
            face.allTriangles
                .asSequence()
                .mapNotNull {
                    triangle: MlKitTriangle<FaceMeshPoint> ->

                    val connected =
                        triangle.getAllPoints()
            if (connected.size != 3) {
                return@mapNotNull null
            }

            val ids = connected.map { it.index }
            if (
                ids.any { it !in pointsByIndex } ||
                ids.distinct().size != 3
            ) {
                return@mapNotNull null
            }

            Triangle(
                a = ids[0],
                b = ids[1],
                c = ids[2],
            )
        }.toList()

        val pointCoverage =
            (points.size.toDouble() / 468.0)
                .coerceIn(0.0, 1.0)

        val triangleCoverage =
            (
                triangles.size.toDouble() /
                    900.0
                ).coerceIn(0.0, 1.0)

        val boundsWidth =
            (face.boundingBox.width().coerceAtLeast(1))
                .toDouble()
        val boundsHeight =
            (face.boundingBox.height().coerceAtLeast(1))
                .toDouble()
        val imageArea =
            (imageWidth.coerceAtLeast(1).toDouble() *
                imageHeight.coerceAtLeast(1).toDouble())
        val faceAreaRatio =
            (
                boundsWidth * boundsHeight /
                    imageArea.coerceAtLeast(1.0)
                ).coerceIn(0.0, 1.0)

        val zValues = points
            .map { it.positionImageSpace.z }
            .filter(Double::isFinite)

        val depthSpread =
            if (zValues.isEmpty()) {
                0.0
            } else {
                (
                    zValues.maxOrNull()!! -
                        zValues.minOrNull()!!
                    ).coerceAtLeast(0.0)
            }

        val faceSizeQuality =
            when {
                faceAreaRatio >= 0.16 -> 1.0
                faceAreaRatio >= 0.08 ->
                    0.70 + 0.30 * (
                        (faceAreaRatio - 0.08) / 0.08
                    )
                faceAreaRatio >= 0.025 ->
                    0.40 + 0.30 * (
                        (faceAreaRatio - 0.025) / 0.055
                    )
                else ->
                    (
                        faceAreaRatio / 0.025
                    ).coerceIn(0.0, 1.0)
            }

        val depthQuality =
            if (depthSpread.isFinite() && depthSpread > 0.5) {
                1.0
            } else if (depthSpread.isFinite()) {
                (
                    depthSpread / 0.5
                ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val geometricConfidence = (
            pointCoverage * 0.46 +
                triangleCoverage * 0.18 +
                faceSizeQuality * 0.24 +
                depthQuality * 0.12
            ).coerceIn(0.0, 1.0)

        val confidence = (
            geometricConfidence *
                timingQuality.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        return Result(
            timestampNanos = timestampNanos,
            boundsLeft = face.boundingBox.left.toFloat(),
            boundsTop = face.boundingBox.top.toFloat(),
            boundsRight = face.boundingBox.right.toFloat(),
            boundsBottom = face.boundingBox.bottom.toFloat(),
            points = points.sortedBy { it.index },
            triangles = triangles,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = rotationDegrees,
            confidence = confidence,
            pointCoverage = pointCoverage,
            triangleCoverage = triangleCoverage,
            faceAreaRatio = faceAreaRatio,
            depthSpreadImageUnits = depthSpread,
        )
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            detector.close()
        }
    }
}
