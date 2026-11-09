package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Temporal quality monitor for TSDF/dense-surface reconstruction.
 *
 * Dense extraction is not guaranteed to preserve vertex identity. Voxel marching and cleanup can
 * legitimately create or remove vertices even when the physical surface is stable. Therefore this
 * engine compares topology-independent geometric distributions instead of trusting vertex indices.
 */
class DenseMeshTemporalConsistencyEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumGapSeconds: Double = 0.50,
        val minimumVertices: Int = 24,
        val minimumTriangles: Int = 12,
        val relativeScaleWarning: Double = 0.16,
        val relativeScaleReject: Double = 0.32,
        val areaChangeWarning: Double = 0.25,
        val areaChangeReject: Double = 0.50,
        val edgeDistributionWarning: Double = 0.24,
        val edgeDistributionReject: Double = 0.48,
        val triangleShapeWarning: Double = 0.28,
        val triangleShapeReject: Double = 0.55,
        val vertexCountChangeWarning: Double = 0.45,
        val vertexCountChangeReject: Double = 0.70,
        val historyAlpha: Double = 0.18,
        val warningScore: Double = 0.68,
        val rejectScore: Double = 0.42,
        val persistentWarningFrames: Int = 4,
        val persistentRejectFrames: Int = 8,
        val evolutionVelocityAlpha: Double = 0.24,
        val maximumCentroidSpeedMetersPerSecond: Double = 6.0,
        val maximumAreaChangeRatePerSecond: Double = 2.5,
        val maximumScaleChangeRatePerSecond: Double = 2.0,
    ) {
        init {
            require(maximumGapSeconds > 0.0)
            require(minimumVertices >= 3)
            require(minimumTriangles >= 1)
            require(relativeScaleWarning > 0.0)
            require(relativeScaleReject > relativeScaleWarning)
            require(areaChangeWarning > 0.0)
            require(areaChangeReject > areaChangeWarning)
            require(edgeDistributionWarning > 0.0)
            require(edgeDistributionReject > edgeDistributionWarning)
            require(triangleShapeWarning > 0.0)
            require(triangleShapeReject > triangleShapeWarning)
            require(vertexCountChangeWarning > 0.0)
            require(vertexCountChangeReject > vertexCountChangeWarning)
            require(historyAlpha in 0.0..1.0)
            require(rejectScore in 0.0..warningScore)
            require(persistentWarningFrames >= 1)
            require(persistentRejectFrames >= persistentWarningFrames)
            require(evolutionVelocityAlpha in 0.01..1.0)
            require(maximumCentroidSpeedMetersPerSecond > 0.0)
            require(maximumAreaChangeRatePerSecond > 0.0)
            require(maximumScaleChangeRatePerSecond > 0.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val mesh: ReconstructionMesh,
        val upstreamConfidence: Double,
    )

    data class Result(
        val timestampNanos: Long,
        val accepted: Boolean,
        val confidence: Double,
        val centroidMotionMeters: Double,
        val scaleChangeRatio: Double,
        val surfaceAreaChangeRatio: Double,
        val edgeDistributionDelta: Double,
        val triangleShapeDelta: Double,
        val vertexCountChangeRatio: Double,
        val persistentWarningFrames: Int,
        val persistentRejectFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
        val evolutionPredictionError: Double = 0.0,
        val evolutionConsistency: Double = 1.0,
    )

    private data class Statistics(
        val centroid: Vec3,
        val radialMedianMeters: Double,
        val radialP90Meters: Double,
        val surfaceAreaMeters2: Double,
        val edgeP50Meters: Double,
        val edgeP90Meters: Double,
        val triangleAspectMedian: Double,
        val triangleAspectP90: Double,
        val vertexCount: Int,
        val triangleCount: Int,
    )

    private data class History(
        var timestampNanos: Long,
        var statistics: Statistics,
        var warningFrames: Int,
        var rejectFrames: Int,
        var confidenceEma: Double,
        var centroidVelocity: Vec3 = Vec3(0.0, 0.0, 0.0),
        var areaVelocity: Double = 0.0,
        var scaleVelocity: Double = 0.0,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val statistics = statistics(input.mesh)
            ?: return Result(
                timestampNanos = input.timestampNanos,
                accepted = false,
                confidence = 0.0,
                centroidMotionMeters = 0.0,
                scaleChangeRatio = 1.0,
                surfaceAreaChangeRatio = 1.0,
                edgeDistributionDelta = 1.0,
                triangleShapeDelta = 1.0,
                vertexCountChangeRatio = 1.0,
                persistentWarningFrames = history?.warningFrames ?: 0,
                persistentRejectFrames = history?.rejectFrames ?: 0,
                warnings = listOf(
                    "dense-mesh-observation-below-production-minimum",
                ),
                recommendations = listOf(
                    "Hold the surface output until the TSDF contains enough valid geometry.",
                ),
            )

        val previous = history
        if (previous == null) {
            history = History(
                timestampNanos = input.timestampNanos,
                statistics = statistics,
                warningFrames = 0,
                rejectFrames = 0,
                confidenceEma =
                    input.upstreamConfidence.coerceIn(0.0, 1.0),
                centroidVelocity = Vec3(0.0, 0.0, 0.0),
                areaVelocity = 0.0,
                scaleVelocity = 0.0,
            )
            return Result(
                timestampNanos = input.timestampNanos,
                accepted = input.upstreamConfidence >= config.rejectScore,
                confidence = input.upstreamConfidence.coerceIn(0.0, 1.0),
                centroidMotionMeters = 0.0,
                scaleChangeRatio = 1.0,
                surfaceAreaChangeRatio = 1.0,
                edgeDistributionDelta = 0.0,
                triangleShapeDelta = 0.0,
                vertexCountChangeRatio = 0.0,
                persistentWarningFrames = 0,
                persistentRejectFrames = 0,
                warnings = emptyList(),
                recommendations = emptyList(),
            )
        }

        val dtSeconds = (
            (input.timestampNanos - previous.timestampNanos)
                .coerceAtLeast(1L) / 1e9
        )

        if (dtSeconds > config.maximumGapSeconds) {
            history = History(
                timestampNanos = input.timestampNanos,
                statistics = statistics,
                warningFrames = 0,
                rejectFrames = 0,
                confidenceEma =
                    input.upstreamConfidence.coerceIn(0.0, 1.0),
                centroidVelocity = Vec3(0.0, 0.0, 0.0),
                areaVelocity = 0.0,
                scaleVelocity = 0.0,
            )
            return Result(
                timestampNanos = input.timestampNanos,
                accepted = input.upstreamConfidence >= config.rejectScore,
                confidence =
                    input.upstreamConfidence.coerceIn(0.0, 1.0) * 0.90,
                centroidMotionMeters =
                    statistics.centroid.distanceTo(previous.statistics.centroid),
                scaleChangeRatio = 1.0,
                surfaceAreaChangeRatio = 1.0,
                edgeDistributionDelta = 0.0,
                triangleShapeDelta = 0.0,
                vertexCountChangeRatio = 0.0,
                persistentWarningFrames = 0,
                persistentRejectFrames = 0,
                warnings = listOf(
                    "dense-mesh-temporal-history-reset-after-gap",
                ),
                recommendations = listOf(
                    "Keep dense-depth acquisition continuous for temporal surface validation.",
                ),
            )
        }

        val predictedCentroid =
            previous.statistics.centroid +
                previous.centroidVelocity * dtSeconds
        val predictedArea =
            (
                previous.statistics.surfaceAreaMeters2 +
                    previous.areaVelocity * dtSeconds
                ).coerceAtLeast(1e-9)
        val predictedScale =
            (
                previous.statistics.radialMedianMeters +
                    previous.scaleVelocity * dtSeconds
                ).coerceAtLeast(1e-9)

        val centroidMotion =
            statistics.centroid.distanceTo(previous.statistics.centroid)

        val scaleRatio = safeRatio(
            statistics.radialMedianMeters,
            previous.statistics.radialMedianMeters,
        )

        val areaRatio = safeRatio(
            statistics.surfaceAreaMeters2,
            previous.statistics.surfaceAreaMeters2,
        )

        val edgeDelta = (
            relativeDelta(
                statistics.edgeP50Meters,
                previous.statistics.edgeP50Meters,
            ) * 0.58 +
                relativeDelta(
                    statistics.edgeP90Meters,
                    previous.statistics.edgeP90Meters,
                ) * 0.42
            ).coerceIn(0.0, 4.0)

        val triangleDelta = (
            relativeDelta(
                statistics.triangleAspectMedian,
                previous.statistics.triangleAspectMedian,
            ) * 0.60 +
                relativeDelta(
                    statistics.triangleAspectP90,
                    previous.statistics.triangleAspectP90,
                ) * 0.40
            ).coerceIn(0.0, 4.0)

        val vertexRatio = relativeDelta(
            statistics.vertexCount.toDouble(),
            previous.statistics.vertexCount.toDouble(),
        )

        val scalePenalty = thresholdPenalty(
            abs(scaleRatio - 1.0),
            config.relativeScaleWarning,
            config.relativeScaleReject,
        )

        val areaPenalty = thresholdPenalty(
            abs(areaRatio - 1.0),
            config.areaChangeWarning,
            config.areaChangeReject,
        )

        val edgePenalty = thresholdPenalty(
            edgeDelta,
            config.edgeDistributionWarning,
            config.edgeDistributionReject,
        )

        val trianglePenalty = thresholdPenalty(
            triangleDelta,
            config.triangleShapeWarning,
            config.triangleShapeReject,
        )

        val vertexPenalty = thresholdPenalty(
            vertexRatio,
            config.vertexCountChangeWarning,
            config.vertexCountChangeReject,
        )

        val predictionCentroidError =
            statistics.centroid.distanceTo(
                predictedCentroid,
            ) /
                config.maximumCentroidSpeedMetersPerSecond
                    .coerceAtLeast(1e-6)

        val predictionAreaRatio =
            safeRatio(
                statistics.surfaceAreaMeters2,
                predictedArea,
            )

        val predictionScaleRatio =
            safeRatio(
                statistics.radialMedianMeters,
                predictedScale,
            )

        val evolutionPredictionError = (
            predictionCentroidError.coerceIn(0.0, 2.0) * 0.42 +
                abs(predictionAreaRatio - 1.0)
                    .coerceIn(0.0, 2.0) * 0.30 +
                abs(predictionScaleRatio - 1.0)
                    .coerceIn(0.0, 2.0) * 0.28
            ).coerceIn(0.0, 1.0)

        val evolutionConsistency =
            exp(-2.2 * evolutionPredictionError)
                .coerceIn(0.0, 1.0)

        val confidence = (
            (1.0 - scalePenalty) * 0.22 +
                (1.0 - areaPenalty) * 0.22 +
                (1.0 - edgePenalty) * 0.22 +
                (1.0 - trianglePenalty) * 0.18 +
                (1.0 - vertexPenalty) * 0.08 +
                evolutionConsistency * 0.10 +
                input.upstreamConfidence.coerceIn(0.0, 1.0) * 0.04
        ).coerceIn(0.0, 1.0)

        val warning =
            confidence < config.warningScore
        val reject =
            confidence < config.rejectScore ||
                scalePenalty >= 1.0 ||
                areaPenalty >= 1.0 ||
                edgePenalty >= 1.0 ||
                trianglePenalty >= 1.0

        val confidenceAlpha = config.historyAlpha
        previous.confidenceEma =
            previous.confidenceEma * (1.0 - confidenceAlpha) +
                confidence * confidenceAlpha

        previous.warningFrames = if (warning) {
            previous.warningFrames + 1
        } else {
            max(0, previous.warningFrames - 1)
        }

        previous.rejectFrames = if (reject) {
            previous.rejectFrames + 1
        } else {
            max(0, previous.rejectFrames - 1)
        }

        val evolutionAlpha = config.evolutionVelocityAlpha
        val measuredCentroidVelocity =
            (
                statistics.centroid -
                    previous.statistics.centroid
                ) / dtSeconds
        val measuredAreaVelocity =
            (
                statistics.surfaceAreaMeters2 -
                    previous.statistics.surfaceAreaMeters2
                ) / dtSeconds
        val measuredScaleVelocity =
            (
                statistics.radialMedianMeters -
                    previous.statistics.radialMedianMeters
                ) / dtSeconds

        previous.centroidVelocity =
            previous.centroidVelocity * (1.0 - evolutionAlpha) +
                measuredCentroidVelocity * evolutionAlpha
        previous.centroidVelocity =
            clampMagnitude(
                previous.centroidVelocity,
                config.maximumCentroidSpeedMetersPerSecond,
            )
        previous.areaVelocity =
            previous.areaVelocity * (1.0 - evolutionAlpha) +
                measuredAreaVelocity * evolutionAlpha
        previous.areaVelocity =
            previous.areaVelocity.coerceIn(
                -config.maximumAreaChangeRatePerSecond,
                config.maximumAreaChangeRatePerSecond,
            )
        previous.scaleVelocity =
            previous.scaleVelocity * (1.0 - evolutionAlpha) +
                measuredScaleVelocity * evolutionAlpha
        previous.scaleVelocity =
            previous.scaleVelocity.coerceIn(
                -config.maximumScaleChangeRatePerSecond,
                config.maximumScaleChangeRatePerSecond,
            )

        previous.timestampNanos = input.timestampNanos
        previous.statistics = statistics

        val warnings = buildWarnings(
            confidence = confidence,
            scalePenalty = scalePenalty,
            areaPenalty = areaPenalty,
            edgePenalty = edgePenalty,
            trianglePenalty = trianglePenalty,
            vertexPenalty = vertexPenalty,
            warningFrames = previous.warningFrames,
            rejectFrames = previous.rejectFrames,
            dtSeconds = dtSeconds,
        )

        val recommendations = buildRecommendations(
            scalePenalty = scalePenalty,
            areaPenalty = areaPenalty,
            edgePenalty = edgePenalty,
            trianglePenalty = trianglePenalty,
            vertexPenalty = vertexPenalty,
            confidence = confidence,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            accepted =
                previous.rejectFrames <
                    config.persistentRejectFrames &&
                    confidence >= config.rejectScore,
            confidence = (
                previous.confidenceEma * 0.42 +
                    confidence * 0.58
            ).coerceIn(0.0, 1.0),
            centroidMotionMeters = centroidMotion,
            scaleChangeRatio = scaleRatio,
            surfaceAreaChangeRatio = areaRatio,
            edgeDistributionDelta = edgeDelta,
            triangleShapeDelta = triangleDelta,
            vertexCountChangeRatio = vertexRatio,
            persistentWarningFrames = previous.warningFrames,
            persistentRejectFrames = previous.rejectFrames,
            warnings = warnings,
            recommendations = recommendations,
            evolutionPredictionError = evolutionPredictionError.coerceIn(0.0, 1.0),
            evolutionConsistency = evolutionConsistency,
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    private fun clampMagnitude(
        vector: Vec3,
        limit: Double,
    ): Vec3 {
        val magnitude = vector.norm()
        if (!magnitude.isFinite() || magnitude <= limit) {
            return vector
        }
        if (magnitude < 1e-9) {
            return Vec3(0.0, 0.0, 0.0)
        }
        return vector * (limit / magnitude)
    }

    private fun statistics(
        mesh: ReconstructionMesh,
    ): Statistics? {
        if (
            mesh.vertices.size < config.minimumVertices ||
            mesh.triangles.size < config.minimumTriangles
        ) {
            return null
        }

        val positions = mesh.vertices.map { it.position }
        if (positions.any {
                !it.x.isFinite() ||
                    !it.y.isFinite() ||
                    !it.z.isFinite()
            }
        ) {
            return null
        }

        val centroid = Vec3(
            positions.sumOf { it.x } / positions.size,
            positions.sumOf { it.y } / positions.size,
            positions.sumOf { it.z } / positions.size,
        )

        val radialDistances = positions.map {
            it.distanceTo(centroid)
        }

        val edges = edgeLengths(
            positions,
            mesh.triangles,
        )

        val aspectRatios = mesh.triangles.mapNotNull { triangle ->
            if (
                triangle.a !in positions.indices ||
                triangle.b !in positions.indices ||
                triangle.c !in positions.indices
            ) {
                null
            } else {
                val ab = positions[triangle.a]
                    .distanceTo(positions[triangle.b])
                val bc = positions[triangle.b]
                    .distanceTo(positions[triangle.c])
                val ca = positions[triangle.c]
                    .distanceTo(positions[triangle.a])
                if (
                    ab <= 1e-8 ||
                    bc <= 1e-8 ||
                    ca <= 1e-8
                ) {
                    null
                } else {
                    val longest = max(ab, max(bc, ca))
                    val shortest = min(ab, min(bc, ca))
                    longest / shortest
                }
            }
        }

        val area = mesh.triangles.sumOf { triangle ->
            if (
                triangle.a !in positions.indices ||
                triangle.b !in positions.indices ||
                triangle.c !in positions.indices
            ) {
                0.0
            } else {
                val cross =
                    (positions[triangle.b] - positions[triangle.a])
                        .cross(
                            positions[triangle.c] -
                                positions[triangle.a],
                        )
                0.5 * sqrt(
                    cross.x * cross.x +
                        cross.y * cross.y +
                        cross.z * cross.z,
                )
            }
        }

        return Statistics(
            centroid = centroid,
            radialMedianMeters = quantile(radialDistances, 0.50),
            radialP90Meters = quantile(radialDistances, 0.90),
            surfaceAreaMeters2 = area.coerceAtLeast(1e-9),
            edgeP50Meters = quantile(edges, 0.50),
            edgeP90Meters = quantile(edges, 0.90),
            triangleAspectMedian =
                quantile(aspectRatios, 0.50),
            triangleAspectP90 =
                quantile(aspectRatios, 0.90),
            vertexCount = positions.size,
            triangleCount = mesh.triangles.size,
        )
    }

    private fun edgeLengths(
        positions: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): List<Double> {
        val edges = HashSet<Pair<Int, Int>>()
        triangles.forEach { triangle ->
            addEdge(edges, triangle.a, triangle.b)
            addEdge(edges, triangle.b, triangle.c)
            addEdge(edges, triangle.c, triangle.a)
        }

        return edges.mapNotNull { (a, b) ->
            if (
                a !in positions.indices ||
                b !in positions.indices
            ) {
                null
            } else {
                positions[a].distanceTo(positions[b])
            }
        }
    }

    private fun addEdge(
        set: MutableSet<Pair<Int, Int>>,
        a: Int,
        b: Int,
    ) {
        if (a == b || a < 0 || b < 0) return
        set += if (a < b) {
            a to b
        } else {
            b to a
        }
    }

    private fun thresholdPenalty(
        value: Double,
        warning: Double,
        reject: Double,
    ): Double {
        if (!value.isFinite()) return 1.0
        if (value <= warning) return 0.0
        if (value >= reject) return 1.0
        return (
            (value - warning) /
                (reject - warning)
        ).coerceIn(0.0, 1.0)
    }

    private fun safeRatio(
        current: Double,
        previous: Double,
    ): Double =
        if (
            !current.isFinite() ||
            !previous.isFinite() ||
            previous <= 1e-9
        ) {
            1.0
        } else {
            (current / previous).coerceIn(0.05, 20.0)
        }

    private fun relativeDelta(
        current: Double,
        previous: Double,
    ): Double =
        if (
            !current.isFinite() ||
            !previous.isFinite() ||
            previous <= 1e-9
        ) {
            1.0
        } else {
            (
                abs(current - previous) /
                    previous
            ).coerceIn(0.0, 4.0)
        }

    private fun quantile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 0.0
        if (sorted.size == 1) return sorted.first()

        val position =
            fraction.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper = min(
            lower + 1,
            sorted.lastIndex,
        )
        val blend = position - lower

        return sorted[lower] * (1.0 - blend) +
            sorted[upper] * blend
    }

    private fun buildWarnings(
        confidence: Double,
        scalePenalty: Double,
        areaPenalty: Double,
        edgePenalty: Double,
        trianglePenalty: Double,
        vertexPenalty: Double,
        warningFrames: Int,
        rejectFrames: Int,
        dtSeconds: Double,
    ): List<String> = buildList {
        if (confidence < config.warningScore) {
            add("dense-mesh-temporal-consistency-degraded")
        }
        if (scalePenalty > 0.45) {
            add("dense-surface-scale-changed-abruptly")
        }
        if (areaPenalty > 0.45) {
            add("dense-surface-area-changed-abruptly")
        }
        if (edgePenalty > 0.45) {
            add("dense-mesh-edge-distribution-changed-abruptly")
        }
        if (trianglePenalty > 0.45) {
            add("dense-mesh-triangle-shape-distribution-changed")
        }
        if (vertexPenalty > 0.45) {
            add("dense-mesh-extraction-density-shifted")
        }
        if (dtSeconds > 0.20) {
            add("dense-mesh-temporal-sampling-gap")
        }
        if (warningFrames >= config.persistentWarningFrames) {
            add("persistent-dense-surface-instability")
        }
        if (rejectFrames >= config.persistentRejectFrames) {
            add("persistent-dense-mesh-instability")
        }
    }.distinct()

    private fun buildRecommendations(
        scalePenalty: Double,
        areaPenalty: Double,
        edgePenalty: Double,
        trianglePenalty: Double,
        vertexPenalty: Double,
        confidence: Double,
    ): List<String> = buildList {
        if (scalePenalty > 0.45) {
            add("verify depth calibration and camera pose before trusting apparent surface-scale changes")
        }
        if (areaPenalty > 0.45) {
            add("check TSDF coverage and remove views with unstable depth confidence")
        }
        if (edgePenalty > 0.45) {
            add("inspect voxel size, depth noise and motion blur for unstable surface sampling")
        }
        if (trianglePenalty > 0.45) {
            add("hold mesh output until triangulation returns a stable local surface")
        }
        if (vertexPenalty > 0.45) {
            add("treat large vertex-count changes as extraction instability rather than physical deformation")
        }
        if (confidence < config.rejectScore) {
            add("hold dense-surface output and preserve the last trusted mesh")
        }
    }.distinct()
}
