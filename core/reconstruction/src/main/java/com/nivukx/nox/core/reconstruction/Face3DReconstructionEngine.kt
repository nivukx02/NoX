package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceContourKind
import com.nivukx.nox.core.model.FaceContourFrame
import com.nivukx.nox.core.model.ImageCoordinateTransform
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * Metric facial point and patch reconstruction from synchronized phone cameras.
 *
 * The measurement stage triangulates stable ordered contour points. The topology stage only creates
 * deterministic surface patches where semantic correspondence is explicit: eye-to-brow strips and
 * upper/lower lip strips. It does not fabricate a full face surface from insufficient evidence.
 */
class Face3DReconstructionEngine(
    private val triangulator: TriangulationEngine = TriangulationEngine(
        minParallaxDegrees = 0.4,
        huberDeltaPx = 2.0,
    ),
    private val maxTimestampSkewNanos: Long = 30_000_000L,
    private val minimumAcceptedParallaxDegrees: Double = 0.75,
    private val preferredParallaxDegrees: Double = 4.0,
    private val maximumAcceptedReprojectionErrorPx: Double = 8.0,
    private val preferredReprojectionErrorPx: Double = 2.0,
    private val maximumUncertaintyMeters: Double = 0.12,
) {
    init {
        require(maxTimestampSkewNanos > 0L)
        require(minimumAcceptedParallaxDegrees > 0.0)
        require(preferredParallaxDegrees >= minimumAcceptedParallaxDegrees)
        require(maximumAcceptedReprojectionErrorPx > preferredReprojectionErrorPx)
        require(preferredReprojectionErrorPx > 0.0)
        require(maximumUncertaintyMeters > 0.0)
    }
    data class CameraFrame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val face: FaceContourFrame,
    )

    private data class SemanticPoint(
        val id: Int,
        val contour: FaceContourKind,
        val index: Int,
        val position: Vec3,
        val errorPx: Double,
        val confidence: Double,
        val parallaxDegrees: Double,
        val uncertaintyMeters: Double,
        val independentViews: Int,
        val metricEvidenceAuthority: Double = 0.0,
        val epipolarConsistencyScore: Double = 1.0,
    )

    data class Result(
        val mesh: ReconstructionMesh,
        val pointsBySemanticId: Map<Int, Vec3>,
        val meanReprojectionErrorPx: Double,
        val validPointCount: Int,
        val patchCount: Int = 0,
        val meanPointConfidence: Double = 0.0,
        val surfaceCoverage: Double = 0.0,
        val meanParallaxDegrees: Double = 0.0,
        val meanUncertaintyMeters: Double = 0.0,
        val independentViewCount: Int = 0,
        val precisionQualified: Boolean = false,
        val strictSubMillimeterQualified: Boolean = false,
        val criticalRegionCoverage: Double = 0.0,
        val criticalRegionConfidence: Double = 0.0,
        val weightedReprojectionErrorPx: Double = Double.POSITIVE_INFINITY,
        val meanMetricEvidenceAuthority: Double = 0.0,
        val meanEpipolarConsistencyScore: Double = 1.0,
        val epipolarWarnings: List<String> = emptyList(),
    )

    fun reconstruct(
        targetTimestampNanos: Long,
        frames: List<CameraFrame>,
        intrinsics: Map<DeviceId, CameraIntrinsics>,
        extrinsics: Map<DeviceId, Pose>,
        calibrationUncertaintyProfiles:
            Map<DeviceId, CalibrationUncertaintyProfile> = emptyMap(),
    ): Result? {
        if (frames.size < 2) return null

        val validFrames = frames.filter {
            abs(it.timestampNanos - targetTimestampNanos) <= maxTimestampSkewNanos &&
                intrinsics.containsKey(it.deviceId) &&
                extrinsics.containsKey(it.deviceId)
        }
        if (validFrames.size < 2) return null

        val points = mutableListOf<SemanticPoint>()

        FaceContourKind.entries.forEach { contourKind ->
            val maximumIndex = validFrames
                .mapNotNull { frame ->
                    frame.face.contours
                        .firstOrNull { it.kind == contourKind }
                        ?.points
                        ?.lastIndex
                }
                .maxOrNull()
                ?: return@forEach

            for (index in 0..maximumIndex) {
                val observations = validFrames.mapNotNull { frame ->
                    val contour = frame.face.contours.firstOrNull { it.kind == contourKind }
                        ?: return@mapNotNull null
                    val point = contour.points.getOrNull(index)
                        ?: return@mapNotNull null
                    val cameraIntrinsics = intrinsics[frame.deviceId]
                        ?: return@mapNotNull null
                    val pose = extrinsics[frame.deviceId]
                        ?: return@mapNotNull null

                    val sensorPoint =
                        if (
                            frame.face.imageWidth <= 0 ||
                            frame.face.imageHeight <= 0
                        ) {
                            null
                        } else {
                            ImageCoordinateTransform.detectorToSensor(
                                point = Vec2(point.x, point.y),
                                rawWidth = frame.face.imageWidth,
                                rawHeight = frame.face.imageHeight,
                                rotationDegrees = frame.face.rotationDegrees,
                            )
                        }

                    sensorPoint?.let {
                        CameraObservation(
                            deviceId = frame.deviceId,
                            point = it,
                            confidence =
                                (
                                    contour.confidence *
                                        frame.face.captureTimingConfidence
                                ).coerceIn(0.0, 1.0),
                            intrinsics = cameraIntrinsics,
                            pose = pose,
                            rollingShutterFraction =
                                if (frame.face.frameDurationNanos > 0L) {
                                    (
                                        frame.face.rollingShutterSkewNanos.toDouble() /
                                            frame.face.frameDurationNanos.toDouble()
                                    ).coerceIn(0.0, 1.0)
                                } else {
                                    0.0
                                },
                            rowTimeOffsetNanos =
                                if (frame.face.imageHeight > 0) {
                                    (
                                        (
                                            (
                                                it.y /
                                                    frame.face.imageHeight.toDouble()
                                            ) - 0.5
                                        ) *
                                            frame.face.rollingShutterSkewNanos.toDouble() *
                                            frame.face.rollingShutterDirectionSign
                                        ).toLong()
                                } else {
                                    0L
                                },
                            temporalAlignmentConfidence =
                                frame.face.captureTimingConfidence
                                    .coerceIn(0.0, 1.0),
                            temporalUncertaintyNanos =
                                frame.face.frameDurationNanos.coerceAtLeast(0L),
                            motionBlurRisk =
                                frame.face.estimatedMotionBlurRisk,
                            calibrationUncertainty =
                                calibrationUncertaintyProfiles[
                                    frame.deviceId
                                ]
                                    ?: CalibrationUncertaintyProfile(),
                        )
                    }
                }

                if (observations.size < 2) continue

                val result = triangulator.robust(
                    observations.map(TriangulationEngine::Observation),
                ) ?: continue

                if (
                    result.inliers < 2 ||
                    !result.rmsReprojectionPx.isFinite() ||
                    result.rmsReprojectionPx >
                        maximumAcceptedReprojectionErrorPx ||
                    result.parallaxDegrees <
                        minimumAcceptedParallaxDegrees ||
                    !result.uncertaintyMeters.isFinite() ||
                    result.uncertaintyMeters >
                        maximumUncertaintyMeters
                ) {
                    continue
                }

                val residualQuality =
                    exp(
                        -result.rmsReprojectionPx /
                            preferredReprojectionErrorPx,
                    ).coerceIn(0.0, 1.0)

                val parallaxQuality =
                    (
                        result.parallaxDegrees /
                            preferredParallaxDegrees
                    ).coerceIn(0.0, 1.0)

                val uncertaintyQuality =
                    (
                        1.0 -
                            result.uncertaintyMeters /
                                maximumUncertaintyMeters
                    ).coerceIn(0.0, 1.0)

                val viewQuality =
                    (
                        result.independentViewCount
                            .coerceAtMost(8)
                            .toDouble() /
                            8.0
                    ).coerceIn(0.0, 1.0)

                val observationQuality =
                    (
                        result.inliers.toDouble() /
                            result.observations.coerceAtLeast(1)
                    ).coerceIn(0.0, 1.0)

                val epipolarQuality =
                    result.symmetricEpipolarConsistencyScore
                        .coerceIn(0.0, 1.0)
                val metricEvidenceAuthority =
                    minOf(
                        result.metricEvidenceAuthority
                            .coerceIn(0.0, 1.0),
                        (
                            result.activeViewInformationGainScore * 0.55 +
                                result.activeViewRedundancyScore * 0.20 +
                                epipolarQuality * 0.25
                            ).coerceIn(0.0, 1.0),
                    )

                val pointConfidence =
                    (
                        observationQuality * 0.22 +
                            residualQuality * 0.24 +
                            parallaxQuality * 0.17 +
                            uncertaintyQuality * 0.19 +
                            viewQuality * 0.08 +
                            metricEvidenceAuthority * 0.10
                    ).coerceIn(0.0, 1.0)

                val semanticId = contourKind.ordinal * 1000 + index
                points += SemanticPoint(
                    id = semanticId,
                    contour = contourKind,
                    index = index,
                    position = result.point,
                    errorPx = result.rmsReprojectionPx,
                    confidence = pointConfidence,
                    parallaxDegrees =
                        result.parallaxDegrees,
                    uncertaintyMeters =
                        result.uncertaintyMeters,
                    independentViews =
                        result.independentViewCount,
                    metricEvidenceAuthority =
                        metricEvidenceAuthority,
                    epipolarConsistencyScore =
                        epipolarQuality,
                )
            }
        }

        if (points.size < 10) return null

        val vertices =
            points.map {
                MeshVertex(
                    position = it.position,
                    uncertaintyMeters =
                        it.uncertaintyMeters.coerceAtLeast(0.0),
                )
            }
        val vertexBySemantic = points.withIndex().associate { it.value.id to it.index }
        val triangles = buildSurfacePatches(points, vertexBySemantic)

        val pointConfidence =
            weightedPointConfidence(points)

        val criticalPoints = points.filter {
            it.contour == FaceContourKind.LEFT_EYE ||
                it.contour == FaceContourKind.RIGHT_EYE ||
                it.contour == FaceContourKind.LEFT_EYEBROW_TOP ||
                it.contour == FaceContourKind.RIGHT_EYEBROW_TOP ||
                it.contour == FaceContourKind.UPPER_LIP_TOP ||
                it.contour == FaceContourKind.UPPER_LIP_BOTTOM ||
                it.contour == FaceContourKind.LOWER_LIP_TOP ||
                it.contour == FaceContourKind.LOWER_LIP_BOTTOM ||
                it.contour == FaceContourKind.NOSE_BRIDGE ||
                it.contour == FaceContourKind.NOSE_BOTTOM
        ]
        val criticalCoverage = (
            criticalPoints.size.toDouble() /
                max(10.0, criticalContourPointTarget())
            ).coerceIn(0.0, 1.0)
        val criticalConfidence =
            weightedPointConfidence(criticalPoints)

        val weightedError =
            weightedReprojectionError(points)

        val meanMetricEvidenceAuthority =
            points
                .map { it.metricEvidenceAuthority }
                .average()
                .coerceIn(0.0, 1.0)
        val meanEpipolarConsistencyScore =
            points
                .map { it.epipolarConsistencyScore }
                .average()
                .coerceIn(0.0, 1.0)
        val epipolarWarnings =
            buildList {
                if (meanEpipolarConsistencyScore < 0.45) {
                    add("FACE_EPIPOLAR_EVIDENCE_DEGRADED")
                }
                if (meanEpipolarConsistencyScore < 0.30) {
                    add("FACE_EPIPOLAR_EVIDENCE_STRONGLY_DEGRADED")
                }
            }.distinct()

        val coverage = (
            triangles.size.toDouble() /
                max(points.size * 1.5, 1.0)
            ).coerceIn(0.0, 1.0)

        val meanParallax =
            points
                .map { it.parallaxDegrees }
                .average()
                .coerceAtLeast(0.0)

        val meanUncertainty =
            points
                .map { it.uncertaintyMeters }
                .average()
                .coerceAtLeast(0.0)

        val independentViews =
            points
                .maxOfOrNull { it.independentViews }
                ?: 0

        val precisionQualified =
            pointConfidence >= 0.80 &&
                criticalCoverage >= 0.70 &&
                criticalConfidence >= 0.78 &&
                meanParallax >= preferredParallaxDegrees &&
                meanUncertainty <= 0.025 &&
                meanMetricEvidenceAuthority >= 0.55 &&
                meanEpipolarConsistencyScore >= 0.55 &&
                weightedError <= preferredReprojectionErrorPx

        val strictSubMillimeterQualified =
            pointConfidence >= 0.92 &&
                criticalCoverage >= 0.95 &&
                criticalConfidence >= 0.92 &&
                meanParallax >= preferredParallaxDegrees &&
                independentViews >= 3 &&
                meanUncertainty <= 0.0010 &&
                meanMetricEvidenceAuthority >= 0.90 &&
                meanEpipolarConsistencyScore >= 0.90 &&
                weightedError <= 0.75 &&
                points.all { it.uncertaintyMeters <= 0.0010 }

        return Result(
            mesh = ReconstructionMesh(
                vertices = vertices,
                triangles = triangles,
                timestampNanos = targetTimestampNanos,
                confidence = (
                    0.70 * pointConfidence +
                        0.30 * coverage
                    ).coerceIn(0.0, 1.0),
                vertexUncertaintyMeters =
                    points.map {
                        it.uncertaintyMeters.coerceAtLeast(0.0)
                    },
            ),
            pointsBySemanticId = points.associate { it.id to it.position },
            meanReprojectionErrorPx = points.map { it.errorPx }.average(),
            validPointCount = points.size,
            patchCount = triangles.size,
            meanPointConfidence = pointConfidence,
            surfaceCoverage = coverage,
            meanParallaxDegrees = meanParallax,
            meanUncertaintyMeters = meanUncertainty,
            independentViewCount = independentViews,
            precisionQualified = precisionQualified,
            strictSubMillimeterQualified =
                strictSubMillimeterQualified,
            criticalRegionCoverage = criticalCoverage,
            criticalRegionConfidence = criticalConfidence,
            weightedReprojectionErrorPx = weightedError,
            meanMetricEvidenceAuthority =
                meanMetricEvidenceAuthority,
            meanEpipolarConsistencyScore =
                meanEpipolarConsistencyScore,
            epipolarWarnings =
                epipolarWarnings,
        )
    }

    private fun weightedPointConfidence(
        points: List<SemanticPoint>,
    ): Double {
        if (points.isEmpty()) return 0.0
        var numerator = 0.0
        var denominator = 0.0
        points.forEach { point ->
            val weight = contourImportance(point.contour)
            numerator += point.confidence * weight
            denominator += weight
        }
        return if (denominator <= 1e-9) {
            0.0
        } else {
            (numerator / denominator).coerceIn(0.0, 1.0)
        }
    }

    private fun weightedReprojectionError(
        points: List<SemanticPoint>,
    ): Double {
        if (points.isEmpty()) return Double.POSITIVE_INFINITY
        var numerator = 0.0
        var denominator = 0.0
        points.forEach { point ->
            val weight = contourImportance(point.contour)
            numerator += point.errorPx * weight
            denominator += weight
        }
        return if (denominator <= 1e-9) {
            Double.POSITIVE_INFINITY
        } else {
            numerator / denominator
        }
    }

    private fun contourImportance(
        contour: FaceContourKind,
    ): Double =
        when (contour) {
            FaceContourKind.LEFT_EYE,
            FaceContourKind.RIGHT_EYE ->
                1.40
            FaceContourKind.LEFT_EYEBROW_TOP,
            FaceContourKind.RIGHT_EYEBROW_TOP ->
                1.25
            FaceContourKind.NOSE_BRIDGE,
            FaceContourKind.NOSE_BOTTOM ->
                1.18
            FaceContourKind.UPPER_LIP_TOP,
            FaceContourKind.UPPER_LIP_BOTTOM,
            FaceContourKind.LOWER_LIP_TOP,
            FaceContourKind.LOWER_LIP_BOTTOM ->
                1.35
            else ->
                1.0
        }

    private fun criticalContourPointTarget(): Int =
        max(
            10,
            FaceContourKind.entries.count { it != FaceContourKind.FACE_OVAL },
        )
    }

    private fun meanReprojection(
        points: List<SemanticPoint>,
    ): Double =
        points
            .map { it.errorPx }
            .average()
            .takeIf(Double::isFinite)
            ?: Double.POSITIVE_INFINITY

    private fun buildSurfacePatches(
        points: List<SemanticPoint>,
        vertexBySemantic: Map<Int, Int>,
    ): List<MeshTriangle> {
        val triangles = mutableListOf<MeshTriangle>()

        fun connectStrip(
            a: FaceContourKind,
            b: FaceContourKind,
            reverseB: Boolean = false,
        ) {
            val aPoints = points.filter { it.contour == a }.sortedBy { it.index }
            var bPoints = points.filter { it.contour == b }.sortedBy { it.index }
            if (reverseB) bPoints = bPoints.reversed()

            val count = minOf(aPoints.size, bPoints.size)
            if (count < 2) return

            for (i in 0 until count - 1) {
                val a0 = vertexBySemantic[aPoints[i].id] ?: continue
                val a1 = vertexBySemantic[aPoints[i + 1].id] ?: continue
                val b0 = vertexBySemantic[bPoints[i].id] ?: continue
                val b1 = vertexBySemantic[bPoints[i + 1].id] ?: continue

                triangles += MeshTriangle(a0, b0, a1)
                triangles += MeshTriangle(a1, b0, b1)
            }
        }

        connectStrip(FaceContourKind.LEFT_EYE, FaceContourKind.LEFT_EYEBROW_TOP)
        connectStrip(FaceContourKind.RIGHT_EYE, FaceContourKind.RIGHT_EYEBROW_TOP, reverseB = true)
        connectStrip(FaceContourKind.NOSE_BRIDGE, FaceContourKind.NOSE_BOTTOM)
        connectStrip(FaceContourKind.UPPER_LIP_TOP, FaceContourKind.UPPER_LIP_BOTTOM)
        connectStrip(FaceContourKind.LOWER_LIP_TOP, FaceContourKind.LOWER_LIP_BOTTOM)

        return triangles
    }
}
