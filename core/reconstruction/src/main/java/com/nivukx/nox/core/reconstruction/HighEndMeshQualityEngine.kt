
package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DenseMeshQuality
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Post-processes a dense reconstruction mesh before it becomes session output.
 *
 * The engine is intentionally conservative and deterministic:
 * - invalid and degenerate triangles are removed;
 * - pathological edge lengths are rejected;
 * - isolated vertices disappear from the exported mesh;
 * - one bounded Laplacian pass reduces sensor-scale noise without erasing the silhouette;
 * - topology usage and triangle regularity become explicit quality evidence.
 *
 * This does not claim photogrammetric detail that the sensor did not observe.
 */
class HighEndMeshQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumTriangleAreaMeters2: Double = 1e-7,
        val minimumEdgeMeters: Double = 0.0008,
        val maximumEdgeMeters: Double = 0.45,
        val maximumEdgeToMedianRatio: Double = 5.0,
        val smoothingAlpha: Double = 0.10,
        val smoothingPasses: Int = 1,
        val minimumVertexDegree: Int = 1,
        val confidenceAreaToleranceRatio: Double = 0.35,
        val uncertaintySmoothingGain: Double = 0.50,
        val uncertaintyReferenceMeters: Double = 0.001,
    ) {
        init {
            require(minimumTriangleAreaMeters2 > 0.0)
            require(minimumEdgeMeters > 0.0)
            require(maximumEdgeMeters > minimumEdgeMeters)
            require(maximumEdgeToMedianRatio > 1.0)
            require(smoothingAlpha in 0.0..0.5)
            require(smoothingPasses in 0..4)
            require(minimumVertexDegree >= 1)
            require(confidenceAreaToleranceRatio in 0.05..1.0)
            require(uncertaintySmoothingGain in 0.0..1.0)
            require(uncertaintyReferenceMeters > 0.0)
        }
    }

    data class Result(
        val mesh: ReconstructionMesh,
        val quality: DenseMeshQuality,
        val removedTriangles: Int,
        val removedVertices: Int,
        val smoothedVertices: Int,
        val diagnostics: List<String>,
        val surfaceConfidence: HighEndSurfaceConfidenceEngine.Result? = null,
        val boundaryEdgeCount: Int = 0,
        val nonManifoldEdgeCount: Int = 0,
        val duplicateTrianglesRemoved: Int = 0,
        val topologyIntegrityScore: Double = 0.0,
    )

    fun process(
        mesh: ReconstructionMesh,
        incomingQuality: DenseMeshQuality,
    ): Result {
        if (mesh.vertices.isEmpty() || mesh.triangles.isEmpty()) {
            return Result(
                mesh = mesh,
                quality = incomingQuality,
                removedTriangles = 0,
                removedVertices = 0,
                smoothedVertices = 0,
                diagnostics = listOf("empty mesh"),
                surfaceConfidence = null,
            )
        }

        val inputVertexUncertainty =
            mesh.vertices.map {
                it.uncertaintyMeters.coerceAtLeast(0.0)
            }
        val positions = mesh.vertices.map { it.position }
        val medianEdge = medianEdgeLength(
            positions = positions,
            triangles = mesh.triangles,
        )

        val validTriangles = mesh.triangles.filter { triangle ->
            triangle.a in positions.indices &&
                triangle.b in positions.indices &&
                triangle.c in positions.indices &&
                triangle.a != triangle.b &&
                triangle.b != triangle.c &&
                triangle.a != triangle.c &&
                triangleIsValid(
                    triangle = triangle,
                    positions = positions,
                    medianEdge = medianEdge,
                )
        }

        val topologyReport =
            MeshTopologyQualityEngine().sanitize(
                vertices = mesh.vertices,
                triangles = validTriangles,
            )
        val topologyTriangles =
            topologyReport.sanitizedTriangles

        if (topologyTriangles.isEmpty()) {
            return Result(
                mesh = ReconstructionMesh(
                    vertices = emptyList(),
                    triangles = emptyList(),
                    timestampNanos = mesh.timestampNanos,
                    confidence = 0.0,
                ),
                quality = incomingQuality.copy(
                    vertexCount = 0,
                    triangleCount = 0,
                    geometricConfidence = 0.0,
                    watertightnessScore = 0.0,
                ),
                removedTriangles = mesh.triangles.size,
                removedVertices = mesh.vertices.size,
                smoothedVertices = 0,
                diagnostics = listOf(
                    "all triangles rejected",
                    "nonManifoldEdges=" +
                        topologyReport.nonManifoldEdgeCount,
                ),
                surfaceConfidence = null,
                boundaryEdgeCount =
                    topologyReport.boundaryEdgeCount,
                nonManifoldEdgeCount =
                    topologyReport.nonManifoldEdgeCount,
                duplicateTrianglesRemoved =
                    topologyReport.duplicateTriangleCount,
                topologyIntegrityScore =
                    topologyReport.topologyIntegrityScore,
            )
        }

        val used = BooleanArray(positions.size)
        topologyTriangles.forEach {
            used[it.a] = true
            used[it.b] = true
            used[it.c] = true
        }

        val remap = IntArray(positions.size) { -1 }
        val compactPositions = ArrayList<Vec3>()
        for (index in positions.indices) {
            if (used[index]) {
                remap[index] = compactPositions.size
                compactPositions += positions[index]
            }
        }

        var compactTriangles = topologyTriangles.map {
            MeshTriangle(
                a = remap[it.a],
                b = remap[it.b],
                c = remap[it.c],
            )
        }

        var compact = compactPositions
        val compactUncertainty =
            ArrayList<Double>(compact.size)

        for (index in positions.indices) {
            if (remap[index] >= 0) {
                compactUncertainty +=
                    inputVertexUncertainty[index]
            }
        }

        var smoothedCount = 0

        repeat(config.smoothingPasses) {
            val adjacency = buildAdjacency(
                vertexCount = compact.size,
                triangles = compactTriangles,
            )

            val next = ArrayList<Vec3>(compact.size)
            compact.forEachIndexed { index, position ->
                val neighbors = adjacency[index]
                if (neighbors.isEmpty()) {
                    next += position
                    return@forEachIndexed
                }

                val average = neighbors
                    .map { compact[it] }
                    .reduce { acc, value -> acc + value } /
                    neighbors.size.toDouble()

                val geometricConfidence =
                    smoothingConfidence(
                        position = position,
                        neighborAverage = average,
                    )
                val uncertainty =
                    compactUncertainty
                        .getOrElse(index) { 0.0 }
                        .coerceAtLeast(0.0)
                val uncertaintyRisk =
                    (
                        uncertainty /
                            (
                                uncertainty +
                                    config.uncertaintyReferenceMeters
                            )
                    ).coerceIn(0.0, 1.0)
                val alpha = (
                    config.smoothingAlpha *
                        (
                            1.0 +
                                config.uncertaintySmoothingGain *
                                    uncertaintyRisk
                        ) *
                        geometricConfidence
                    ).coerceIn(0.0, 0.5)

                next += position.lerp(
                    average,
                    alpha,
                )

                if (alpha > 1e-4) {
                    smoothedCount += 1
                }
            }

            compact = next
        }

        val compactNormals =
            computeVertexNormals(
                positions = compact,
                triangles = compactTriangles,
            )

        val compactMesh = ReconstructionMesh(
            vertices =
                compact.mapIndexed { index, position ->
                    MeshVertex(
                        position = position,
                        uncertaintyMeters =
                            compactUncertainty
                                .getOrElse(index) { 0.0 },
                    )
                },
            triangles = compactTriangles,
            timestampNanos = mesh.timestampNanos,
            confidence = confidence(
                mesh = compact,
                triangles = compactTriangles,
                incoming = incomingQuality,
                medianEdge = medianEdge,
            ),
            normals = compactNormals,
            vertexUncertaintyMeters =
                compactUncertainty.toList(),
        )

        val surfaceConfidence =
            HighEndSurfaceConfidenceEngine().analyze(compactMesh)

        val quality = quality(
            mesh = compactMesh,
            incoming = incomingQuality,
            medianEdge = medianEdge,
            surfaceConfidence = surfaceConfidence.globalConfidence,
            topologyIntegrityScore =
                topologyReport.topologyIntegrityScore,
            boundaryEdgeCount =
                topologyReport.boundaryEdgeCount,
            nonManifoldEdgeCount =
                topologyReport.nonManifoldEdgeCount,
            duplicateTrianglesRemoved =
                topologyReport.duplicateTriangleCount,
        )

        return Result(
            mesh = compactMesh,
            quality = quality,
            removedTriangles =
                mesh.triangles.size - compactTriangles.size,
            removedVertices =
                mesh.vertices.size - compact.size,
            smoothedVertices = smoothedCount,
            surfaceConfidence = surfaceConfidence,
            boundaryEdgeCount =
                topologyReport.boundaryEdgeCount,
            nonManifoldEdgeCount =
                topologyReport.nonManifoldEdgeCount,
            duplicateTrianglesRemoved =
                topologyReport.duplicateTriangleCount,
            topologyIntegrityScore =
                topologyReport.topologyIntegrityScore,
            diagnostics = diagnostics(
                original = mesh,
                result = compactMesh,
                quality = quality,
                removedTriangles =
                    mesh.triangles.size - compactTriangles.size,
                removedVertices =
                    mesh.vertices.size - compact.size,
                smoothedVertices = smoothedCount,
                boundaryEdgeCount =
                    topologyReport.boundaryEdgeCount,
                nonManifoldEdgeCount =
                    topologyReport.nonManifoldEdgeCount,
                topologyIntegrityScore =
                    topologyReport.topologyIntegrityScore,
            ),
        )
    }

    private fun triangleIsValid(
        triangle: MeshTriangle,
        positions: List<Vec3>,
        medianEdge: Double,
    ): Boolean {
        val a = positions[triangle.a]
        val b = positions[triangle.b]
        val c = positions[triangle.c]

        val ab = magnitude(b - a)
        val bc = magnitude(c - b)
        val ca = magnitude(a - c)

        if (
            !ab.isFinite() ||
            !bc.isFinite() ||
            !ca.isFinite()
        ) {
            return false
        }

        if (
            ab < config.minimumEdgeMeters ||
            bc < config.minimumEdgeMeters ||
            ca < config.minimumEdgeMeters
        ) {
            return false
        }

        val maxEdge = max(ab, max(bc, ca))
        if (maxEdge > config.maximumEdgeMeters) return false

        if (
            medianEdge > config.minimumEdgeMeters &&
            maxEdge >
            medianEdge * config.maximumEdgeToMedianRatio
        ) {
            return false
        }

        val cross = (b - a).cross(c - a)
        val area = magnitude(cross) * 0.5

        return area >= config.minimumTriangleAreaMeters2
    }

    private fun buildAdjacency(
        vertexCount: Int,
        triangles: List<MeshTriangle>,
    ): List<Set<Int>> {
        val adjacency = List(vertexCount) {
            LinkedHashSet<Int>()
        }

        triangles.forEach { triangle ->
            adjacency[triangle.a] += triangle.b
            adjacency[triangle.a] += triangle.c
            adjacency[triangle.b] += triangle.a
            adjacency[triangle.b] += triangle.c
            adjacency[triangle.c] += triangle.a
            adjacency[triangle.c] += triangle.b
        }

        return adjacency
    }

    private fun medianEdgeLength(
        positions: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): Double {
        val values = mutableListOf<Double>()

        triangles.forEach { triangle ->
            if (
                triangle.a !in positions.indices ||
                triangle.b !in positions.indices ||
                triangle.c !in positions.indices
            ) return@forEach

            values += magnitude(
                positions[triangle.a] -
                    positions[triangle.b],
            )
            values += magnitude(
                positions[triangle.b] -
                    positions[triangle.c],
            )
            values += magnitude(
                positions[triangle.c] -
                    positions[triangle.a],
            )
        }

        return median(values)
    }

    private fun smoothingConfidence(
        position: Vec3,
        neighborAverage: Vec3,
    ): Double {
        val delta = magnitude(neighborAverage - position)
        return (
            1.0 -
                delta /
                    config.maximumEdgeMeters
            ).coerceIn(0.10, 1.0)
    }

    private fun confidence(
        mesh: List<Vec3>,
        triangles: List<MeshTriangle>,
        incoming: DenseMeshQuality,
        medianEdge: Double,
    ): Double {
        if (mesh.isEmpty() || triangles.isEmpty()) return 0.0

        val regularity = triangles.mapNotNull { triangle ->
            triangleArea(
                mesh[triangle.a],
                mesh[triangle.b],
                mesh[triangle.c],
            )
        }.let { areas ->
            if (areas.isEmpty()) 0.0
            else {
                val medianArea = median(areas)
                val acceptable = areas.count {
                    medianArea <= 1e-12 ||
                        abs(it - medianArea) /
                            medianArea <=
                            config.confidenceAreaToleranceRatio
                }
                acceptable.toDouble() / areas.size.toDouble()
            }
        }

        val edgeLengths = meshEdgeLengths(
            mesh = mesh,
            triangles = triangles,
        )
        val edgeConsistency =
            edgeLengthConsistency(edgeLengths)

        return (
            incoming.geometricConfidence.coerceIn(0.0, 1.0) * 0.56 +
                regularity * 0.25 +
                edgeConsistency * 0.19
            ).coerceIn(0.0, 1.0)
    }

    private fun quality(
        mesh: ReconstructionMesh,
        incoming: DenseMeshQuality,
        medianEdge: Double,
        surfaceConfidence: Double,
        topologyIntegrityScore: Double,
        boundaryEdgeCount: Int,
        nonManifoldEdgeCount: Int,
        duplicateTrianglesRemoved: Int,
    ): DenseMeshQuality {
        val adjacency = buildAdjacency(
            vertexCount = mesh.vertices.size,
            triangles = mesh.triangles,
        )

        val degreeCoverage =
            if (adjacency.isEmpty()) {
                0.0
            } else {
                adjacency.count {
                    it.size >= config.minimumVertexDegree
                }.toDouble() / adjacency.size.toDouble()
            }

        val areas = mesh.triangles.map {
            triangleArea(
                mesh.vertices[it.a].position,
                mesh.vertices[it.b].position,
                mesh.vertices[it.c].position,
            )
        }

        val medianArea = median(areas)
        val areaRegularity = if (medianArea <= 1e-12) {
            0.0
        } else {
            areas.map {
                (
                    1.0 -
                        abs(it - medianArea) /
                            medianArea
                ).coerceIn(0.0, 1.0)
            }.average()
        }

        val geometric = (
            incoming.geometricConfidence.coerceIn(0.0, 1.0) * 0.38 +
                degreeCoverage * 0.16 +
                areaRegularity * 0.16 +
                surfaceConfidence.coerceIn(0.0, 1.0) * 0.20 +
                if (medianEdge <= config.maximumEdgeMeters) 0.10 else 0.0
            ).coerceIn(0.0, 1.0)

        return incoming.copy(
            vertexCount = mesh.vertices.size,
            triangleCount = mesh.triangles.size,
            geometricConfidence = geometric,
            watertightnessScore = (
                degreeCoverage * 0.72 +
                    areaRegularity * 0.28
                ).coerceIn(0.0, 1.0),
            topologyIntegrityScore =
                topologyIntegrityScore.coerceIn(0.0, 1.0),
            boundaryEdgeCount =
                boundaryEdgeCount.coerceAtLeast(0),
            nonManifoldEdgeCount =
                nonManifoldEdgeCount.coerceAtLeast(0),
            duplicateTrianglesRemoved =
                duplicateTrianglesRemoved.coerceAtLeast(0),
        )
    }

    private fun computeVertexNormals(
        positions: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): List<Vec3> {
        if (positions.isEmpty()) {
            return emptyList()
        }

        val accumulated =
            Array(positions.size) {
                Vec3(0.0, 0.0, 0.0)
            }

        triangles.forEach { triangle ->
            if (
                triangle.a !in positions.indices ||
                triangle.b !in positions.indices ||
                triangle.c !in positions.indices
            ) {
                return@forEach
            }

            val a = positions[triangle.a]
            val b = positions[triangle.b]
            val c = positions[triangle.c]
            val faceNormal =
                (b - a).cross(c - a)

            if (!finite(faceNormal)) {
                return@forEach
            }

            accumulated[triangle.a] =
                accumulated[triangle.a] +
                    faceNormal
            accumulated[triangle.b] =
                accumulated[triangle.b] +
                    faceNormal
            accumulated[triangle.c] =
                accumulated[triangle.c] +
                    faceNormal
        }

        return accumulated.map { normal ->
            val magnitude = normal.norm()
            if (
                !magnitude.isFinite() ||
                magnitude <= 1e-12
            ) {
                Vec3(0.0, 0.0, 1.0)
            } else {
                normal / magnitude
            }
        }
    }

    private fun meshEdgeLengths(
        mesh: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): List<Double> {
        val uniqueEdges = LinkedHashSet<Pair<Int, Int>>()

        triangles.forEach { triangle ->
            addUniqueEdge(
                uniqueEdges,
                triangle.a,
                triangle.b,
            )
            addUniqueEdge(
                uniqueEdges,
                triangle.b,
                triangle.c,
            )
            addUniqueEdge(
                uniqueEdges,
                triangle.c,
                triangle.a,
            )
        }

        return uniqueEdges.mapNotNull { (a, b) ->
            if (
                a !in mesh.indices ||
                b !in mesh.indices
            ) {
                null
            } else {
                magnitude(mesh[a] - mesh[b])
            }
        }.filter { it.isFinite() && it > 1e-8 }
    }

    private fun addUniqueEdge(
        edges: MutableSet<Pair<Int, Int>>,
        a: Int,
        b: Int,
    ) {
        if (a < 0 || b < 0 || a == b) return
        edges += if (a < b) a to b else b to a
    }

    private fun edgeLengthConsistency(
        values: List<Double>,
    ): Double {
        if (values.size < 3) return 0.0

        val sorted = values.sorted()
        val medianValue = median(sorted)
        if (medianValue <= 1e-8) return 0.0

        val deviations = sorted.map {
            abs(it - medianValue)
        }
        val medianAbsoluteDeviation =
            median(deviations)

        return (
            1.0 -
                medianAbsoluteDeviation /
                    medianValue.coerceAtLeast(1e-8)
            ).coerceIn(0.0, 1.0)
    }

    private fun triangleArea(
        a: Vec3,
        b: Vec3,
        c: Vec3,
    ): Double =
        magnitude(
            (b - a).cross(c - a),
        ) * 0.5

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 0.0

        return if (sorted.size % 2 == 0) {
            (
                sorted[sorted.size / 2 - 1] +
                    sorted[sorted.size / 2]
                ) * 0.5
        } else {
            sorted[sorted.size / 2]
        }
    }

    private fun diagnostics(
        original: ReconstructionMesh,
        result: ReconstructionMesh,
        quality: DenseMeshQuality,
        removedTriangles: Int,
        removedVertices: Int,
        smoothedVertices: Int,
        boundaryEdgeCount: Int,
        nonManifoldEdgeCount: Int,
        topologyIntegrityScore: Double,
    ): List<String> = listOf(
        "originalVertices=" + original.vertices.size,
        "originalTriangles=" + original.triangles.size,
        "vertices=" + result.vertices.size,
        "triangles=" + result.triangles.size,
        "removedVertices=" + removedVertices,
        "removedTriangles=" + removedTriangles,
        "smoothedVertices=" + smoothedVertices,
        "boundaryEdges=" + boundaryEdgeCount,
        "nonManifoldEdges=" + nonManifoldEdgeCount,
        "topologyIntegrity=" + format(topologyIntegrityScore),
        "geometricConfidence=" + format(quality.geometricConfidence),
        "watertightness=" + format(quality.watertightnessScore),
        "temporalStability=" + format(quality.temporalStability),
        "surfaceConfidence=" +
            format(
                (
                    quality.geometricConfidence * 0.72 +
                        (
                            result.vertices.size.toDouble() /
                                original.vertices.size.toDouble()
                                    .coerceAtLeast(1.0)
                        ).coerceIn(0.0, 1.0) * 0.28
                ).coerceIn(0.0, 1.0),
            ),
    )

    private fun format(value: Double): String =
        if (!value.isFinite()) "nan"
        else "%.4f".format(java.util.Locale.US, value)
}
