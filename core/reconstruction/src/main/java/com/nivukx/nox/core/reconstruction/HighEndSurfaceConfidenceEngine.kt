package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Dense-surface confidence analyzer for production motion reconstruction.
 *
 * A reconstructed mesh is only useful when its local geometry is trustworthy. This analyzer
 * scores every vertex neighborhood using topology degree, edge-length consistency, triangle
 * area regularity, normal agreement, local curvature proxy, and boundary exposure. It does not
 * fabricate surface detail and never mutates the mesh.
 *
 * The result is intended as a second-order quality signal for mesh export and temporal fusion.
 */
class HighEndSurfaceConfidenceEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumNeighborCount: Int = 2,
        val preferredNeighborCount: Int = 5,
        val normalConsistencyScale: Double = 0.35,
        val edgeConsistencyScale: Double = 0.26,
        val areaConsistencyScale: Double = 0.20,
        val curvatureScale: Double = 0.10,
        val boundaryScale: Double = 0.09,
        val isolatedPenalty: Double = 0.30,
        val boundaryPenalty: Double = 0.08,
        val maximumCurvatureProxy: Double = 2.5,
        val minimumTriangleAreaMeters2: Double = 1e-8,
        val maximumEdgeMeters: Double = 0.60,
    ) {
        init {
            require(minimumNeighborCount >= 1)
            require(preferredNeighborCount >= minimumNeighborCount)
            require(normalConsistencyScale >= 0.0)
            require(edgeConsistencyScale >= 0.0)
            require(areaConsistencyScale >= 0.0)
            require(curvatureScale >= 0.0)
            require(boundaryScale >= 0.0)
            require(isolatedPenalty in 0.0..1.0)
            require(boundaryPenalty in 0.0..1.0)
            require(maximumCurvatureProxy > 0.0)
            require(minimumTriangleAreaMeters2 > 0.0)
            require(maximumEdgeMeters > 0.0)
        }
    }

    data class VertexConfidence(
        val vertexIndex: Int,
        val confidence: Double,
        val neighborCount: Int,
        val normalConsistency: Double,
        val edgeConsistency: Double,
        val areaConsistency: Double,
        val curvatureStability: Double,
        val boundaryExposure: Double,
        val warnings: List<String>,
    )

    data class Result(
        val globalConfidence: Double,
        val p50Confidence: Double,
        val p05Confidence: Double,
        val lowConfidenceVertexRatio: Double,
        val boundaryVertexRatio: Double,
        val degenerateTriangleRatio: Double,
        val topologyCoverage: Double,
        val vertices: List<VertexConfidence>,
        val warnings: List<String>,
        val diagnostics: List<String>,
    )

    fun analyze(mesh: ReconstructionMesh): Result {
        if (mesh.vertices.isEmpty() || mesh.triangles.isEmpty()) {
            return Result(
                globalConfidence = 0.0,
                p50Confidence = 0.0,
                p05Confidence = 0.0,
                lowConfidenceVertexRatio = 1.0,
                boundaryVertexRatio = 0.0,
                degenerateTriangleRatio = 1.0,
                topologyCoverage = 0.0,
                vertices = emptyList(),
                warnings = listOf("surface-empty"),
                diagnostics = listOf("empty-mesh"),
            )
        }

        val positions = mesh.vertices.map { finite(it.position) }
        val adjacency = buildAdjacency(positions.size, mesh.triangles)
        val incidentTriangles = buildIncidentTriangles(
            vertexCount = positions.size,
            triangles = mesh.triangles,
        )
        val edgeCounts = countUndirectedEdges(mesh.triangles)

        val normals = mesh.triangles.map { triangle ->
            triangleNormal(
                triangle = triangle,
                positions = positions,
            )
        }

        val areas = mesh.triangles.map { triangle ->
            triangleArea(
                triangle = triangle,
                positions = positions,
            )
        }

        val validAreaMedian = median(
            areas.filter {
                it.isFinite() &&
                    it >= config.minimumTriangleAreaMeters2
            },
        )

        val edgeLengths = buildEdgeLengths(
            positions = positions,
            triangles = mesh.triangles,
        )
        val edgeMedian = median(edgeLengths.values.toList())

        val values = positions.indices.map { index ->
            val neighbors = adjacency[index]
            val incidents = incidentTriangles[index]

            val localEdgeConsistency =
                edgeConsistency(
                    index = index,
                    neighbors = neighbors,
                    positions = positions,
                    medianEdge = edgeMedian,
                )

            val localAreaConsistency =
                areaConsistency(
                    incidentTriangles = incidents,
                    areas = areas,
                    medianArea = validAreaMedian,
                )

            val localNormalConsistency =
                normalConsistency(
                    incidentTriangles = incidents,
                    normals = normals,
                )

            val curvatureStability =
                curvatureStability(
                    index = index,
                    neighbors = neighbors,
                    positions = positions,
                )

            val boundaryExposure =
                boundaryExposure(
                    index = index,
                    incidentTriangles = incidents,
                    edgeCounts = edgeCounts,
                )

            val topologyConfidence = topologyConfidence(
                neighborCount = neighbors.size,
            )

            val warnings = buildList {
                if (neighbors.size < config.minimumNeighborCount) {
                    add("low-neighbor-support")
                }
                if (localNormalConsistency < 0.35) {
                    add("normal-disagreement")
                }
                if (localEdgeConsistency < 0.40) {
                    add("edge-irregularity")
                }
                if (localAreaConsistency < 0.40) {
                    add("area-irregularity")
                }
                if (curvatureStability < 0.35) {
                    add("surface-variation")
                }
                if (boundaryExposure > 0.75) {
                    add("boundary-exposed")
                }
            }

            val confidence = (
                topologyConfidence * 0.15 +
                    localNormalConsistency * config.normalConsistencyScale +
                    localEdgeConsistency * config.edgeConsistencyScale +
                    localAreaConsistency * config.areaConsistencyScale +
                    curvatureStability * config.curvatureScale +
                    (1.0 - boundaryExposure) * config.boundaryScale
            ).coerceIn(0.0, 1.0)

            VertexConfidence(
                vertexIndex = index,
                confidence = confidence,
                neighborCount = neighbors.size,
                normalConsistency = localNormalConsistency,
                edgeConsistency = localEdgeConsistency,
                areaConsistency = localAreaConsistency,
                curvatureStability = curvatureStability,
                boundaryExposure = boundaryExposure,
                warnings = warnings,
            )
        }

        val confidences = values.map { it.confidence }
        val lowConfidenceRatio =
            confidences.count { it < 0.45 }.toDouble() /
                max(1, confidences.size).toDouble()

        val boundaryRatio =
            values.count { it.boundaryExposure > 0.50 }.toDouble() /
                max(1, values.size).toDouble()

        val degenerateRatio =
            areas.count {
                !it.isFinite() ||
                    it < config.minimumTriangleAreaMeters2
            }.toDouble() /
                max(1, areas.size).toDouble()

        val topologyCoverage =
            values.count {
                it.neighborCount >= config.minimumNeighborCount
            }.toDouble() /
                max(1, values.size).toDouble()

        val sorted = confidences.sorted()
        val p50 = percentile(sorted, 0.50)
        val p05 = percentile(sorted, 0.05)

        val global = (
            p50 * 0.46 +
                p05 * 0.24 +
                topologyCoverage * 0.16 +
                (1.0 - lowConfidenceRatio) * 0.08 +
                (1.0 - degenerateRatio) * 0.06
        ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (global < 0.45) add("surface-confidence-low")
            if (p05 < 0.25) add("surface-confidence-tail-low")
            if (lowConfidenceRatio > 0.30) add("too-many-low-confidence-vertices")
            if (boundaryRatio > 0.45) add("large-boundary-exposure")
            if (degenerateRatio > 0.05) add("degenerate-triangle-ratio-high")
            if (topologyCoverage < 0.75) add("topology-coverage-low")
        }

        return Result(
            globalConfidence = global,
            p50Confidence = p50,
            p05Confidence = p05,
            lowConfidenceVertexRatio = lowConfidenceRatio,
            boundaryVertexRatio = boundaryRatio,
            degenerateTriangleRatio = degenerateRatio,
            topologyCoverage = topologyCoverage,
            vertices = values,
            warnings = warnings.distinct(),
            diagnostics = buildList {
                add("globalConfidence=" + format(global))
                add("p50Confidence=" + format(p50))
                add("p05Confidence=" + format(p05))
                add("lowConfidenceVertexRatio=" + format(lowConfidenceRatio))
                add("boundaryVertexRatio=" + format(boundaryRatio))
                add("degenerateTriangleRatio=" + format(degenerateRatio))
                add("topologyCoverage=" + format(topologyCoverage))
                add("medianEdge=" + format(edgeMedian))
                add("medianArea=" + format(validAreaMedian))
            },
        )
    }

    private fun buildAdjacency(
        vertexCount: Int,
        triangles: List<MeshTriangle>,
    ): List<Set<Int>> {
        val result = List(vertexCount) { LinkedHashSet<Int>() }
        triangles.forEach { triangle ->
            if (triangle.a !in 0 until vertexCount ||
                triangle.b !in 0 until vertexCount ||
                triangle.c !in 0 until vertexCount
            ) {
                return@forEach
            }
            result[triangle.a] += triangle.b
            result[triangle.a] += triangle.c
            result[triangle.b] += triangle.a
            result[triangle.b] += triangle.c
            result[triangle.c] += triangle.a
            result[triangle.c] += triangle.b
        }
        return result
    }

    private fun buildIncidentTriangles(
        vertexCount: Int,
        triangles: List<MeshTriangle>,
    ): List<List<Int>> {
        val result = List(vertexCount) { ArrayList<Int>() }
        triangles.forEachIndexed { index, triangle ->
            if (triangle.a in 0 until vertexCount) result[triangle.a] += index
            if (triangle.b in 0 until vertexCount) result[triangle.b] += index
            if (triangle.c in 0 until vertexCount) result[triangle.c] += index
        }
        return result
    }

    private fun countUndirectedEdges(
        triangles: List<MeshTriangle>,
    ): Map<Pair<Int, Int>, Int> {
        val counts = HashMap<Pair<Int, Int>, Int>()
        triangles.forEach { triangle ->
            addEdge(counts, triangle.a, triangle.b)
            addEdge(counts, triangle.b, triangle.c)
            addEdge(counts, triangle.c, triangle.a)
        }
        return counts
    }

    private fun addEdge(
        counts: MutableMap<Pair<Int, Int>, Int>,
        a: Int,
        b: Int,
    ) {
        if (a < 0 || b < 0 || a == b) return
        val key = if (a < b) a to b else b to a
        counts[key] = (counts[key] ?: 0) + 1
    }

    private fun buildEdgeLengths(
        positions: List<Vec3>,
        triangles: List<MeshTriangle>,
    ): Map<Pair<Int, Int>, Double> {
        val result = LinkedHashMap<Pair<Int, Int>, Double>()
        triangles.forEach { triangle ->
            addLength(result, triangle.a, triangle.b, positions)
            addLength(result, triangle.b, triangle.c, positions)
            addLength(result, triangle.c, triangle.a, positions)
        }
        return result
    }

    private fun addLength(
        output: MutableMap<Pair<Int, Int>, Double>,
        a: Int,
        b: Int,
        positions: List<Vec3>,
    ) {
        if (a !in positions.indices || b !in positions.indices || a == b) return
        val key = if (a < b) a to b else b to a
        if (key !in output) {
            val length = distance(positions[a], positions[b])
            if (length.isFinite() && length <= config.maximumEdgeMeters) {
                output[key] = length
            }
        }
    }

    private fun normalConsistency(
        incidentTriangles: List<Int>,
        normals: List<Vec3>,
    ): Double {
        val valid = incidentTriangles
            .mapNotNull { normals.getOrNull(it) }
            .filter { magnitude(it) > 1e-8 }

        if (valid.size < 2) return if (valid.size == 1) 0.85 else 0.0

        val reference = normalize(valid.first())
        val similarities = valid.drop(1).map { normal ->
            (
                1.0 +
                    dot(
                        reference,
                        normalize(normal),
                    )
            ) * 0.5
        }

        return similarities.average().coerceIn(0.0, 1.0)
    }

    private fun edgeConsistency(
        index: Int,
        neighbors: Set<Int>,
        positions: List<Vec3>,
        medianEdge: Double,
    ): Double {
        if (neighbors.isEmpty() || medianEdge <= 1e-8) return 0.0

        val values = neighbors.mapNotNull { neighbor ->
            val length = distance(
                positions[index],
                positions[neighbor],
            )
            length.takeIf {
                it.isFinite() &&
                    it > 1e-8 &&
                    it <= config.maximumEdgeMeters
            }
        }

        if (values.isEmpty()) return 0.0
        val medianLocal = median(values)
        val deviation = median(
            values.map { abs(it - medianLocal) },
        )

        val globalDeviation = abs(medianLocal - medianEdge)
        val localScore = (
            1.0 -
                deviation /
                    medianLocal.coerceAtLeast(1e-8)
        ).coerceIn(0.0, 1.0)

        val globalScore = (
            1.0 -
                globalDeviation /
                    medianEdge.coerceAtLeast(1e-8)
        ).coerceIn(0.0, 1.0)

        return (
            localScore * 0.68 +
                globalScore * 0.32
        ).coerceIn(0.0, 1.0)
    }

    private fun areaConsistency(
        incidentTriangles: List<Int>,
        areas: List<Double>,
        medianArea: Double,
    ): Double {
        if (incidentTriangles.isEmpty() || medianArea <= 1e-12) return 0.0

        val scores =
            incidentTriangles.mapNotNull { areas.getOrNull(it) }
                .filter { it.isFinite() && it >= config.minimumTriangleAreaMeters2 }
                .map { area ->
                    (1.0 - abs(area - medianArea) / medianArea)
                        .coerceIn(0.0, 1.0)
                }
        return averageOrZero(scores)
    }

    private fun curvatureStability(
        index: Int,
        neighbors: Set<Int>,
        positions: List<Vec3>,
    ): Double {
        if (neighbors.size < 2) return 0.25

        val center = positions[index]
        val vectors = neighbors.map {
            sub(
                positions[it],
                center,
            )
        }

        val distances = vectors.map(::magnitude)
        val medianDistance = median(distances)
        if (!medianDistance.isFinite() || medianDistance <= 1e-8) return 0.0

        val spread = median(
            distances.map { abs(it - medianDistance) },
        )

        val localCurvatureProxy =
            spread / medianDistance

        return (
            1.0 -
                localCurvatureProxy /
                    config.maximumCurvatureProxy
        ).coerceIn(0.0, 1.0)
    }

    private fun topologyConfidence(
        neighborCount: Int,
    ): Double {
        if (neighborCount <= 0) return 0.0
        if (neighborCount >= config.preferredNeighborCount) return 1.0
        return (
            neighborCount.toDouble() /
                config.preferredNeighborCount.toDouble()
        ).coerceIn(0.0, 1.0)
    }

    private fun boundaryExposure(
        index: Int,
        incidentTriangles: List<Int>,
        edgeCounts: Map<Pair<Int, Int>, Int>,
    ): Double {
        if (incidentTriangles.isEmpty()) return 1.0
        var boundaryEdges = 0
        var incidentEdges = 0

        // Count only edges touching this vertex. Edges appearing once are boundary edges.
        edgeCounts.forEach { (edge, count) ->
            if (edge.first == index || edge.second == index) {
                incidentEdges += 1
                if (count == 1) boundaryEdges += 1
            }
        }

        if (incidentEdges == 0) return 1.0
        return (
            boundaryEdges.toDouble() /
                incidentEdges.toDouble()
        ).coerceIn(0.0, 1.0)
    }

    private fun triangleNormal(
        triangle: MeshTriangle,
        positions: List<Vec3>,
    ): Vec3 {
        if (triangle.a !in positions.indices ||
            triangle.b !in positions.indices ||
            triangle.c !in positions.indices
        ) {
            return Vec3(0.0, 0.0, 0.0)
        }
        return normalize(
            (positions[triangle.b] - positions[triangle.a])
                .cross(
                    positions[triangle.c] - positions[triangle.a],
                ),
        )
    }

    private fun triangleArea(
        triangle: MeshTriangle,
        positions: List<Vec3>,
    ): Double {
        if (triangle.a !in positions.indices ||
            triangle.b !in positions.indices ||
            triangle.c !in positions.indices
        ) {
            return 0.0
        }
        return magnitude(
            (positions[triangle.b] - positions[triangle.a])
                .cross(
                    positions[triangle.c] - positions[triangle.a],
                ),
        ) * 0.5
    }

    private fun dot(a: Vec3, b: Vec3): Double =
        a.x * b.x + a.y * b.y + a.z * b.z

    private fun sub(a: Vec3, b: Vec3): Vec3 =
        Vec3(
            a.x - b.x,
            a.y - b.y,
            a.z - b.z,
        )

    private fun normalize(value: Vec3): Vec3 {
        val m = magnitude(value)
        if (!m.isFinite() || m < 1e-10) return Vec3(0.0, 0.0, 0.0)
        return Vec3(value.x / m, value.y / m, value.z / m)
    }

    private fun finite(value: Vec3): Vec3 =
        Vec3(
            if (value.x.isFinite()) value.x else 0.0,
            if (value.y.isFinite()) value.y else 0.0,
            if (value.z.isFinite()) value.z else 0.0,
        )

    private fun magnitude(value: Vec3): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun distance(a: Vec3, b: Vec3): Double =
        magnitude(sub(a, b))

    private fun median(values: List<Double>): Double {
        val sorted = values.filter(Double::isFinite).sorted()
        if (sorted.isEmpty()) return 0.0
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun percentile(
        sortedValues: List<Double>,
        percentile: Double,
    ): Double {
        if (sortedValues.isEmpty()) return 0.0
        val position =
            percentile.coerceIn(0.0, 1.0) *
                (sortedValues.size - 1)
        val lower = position.toInt()
        val upper = min(sortedValues.lastIndex, lower + 1)
        val fraction = position - lower
        return sortedValues[lower] * (1.0 - fraction) +
            sortedValues[upper] * fraction
    }

    private fun averageOrZero(values: List<Double>): Double =
        if (values.isEmpty()) 0.0 else values.average()

    private fun format(value: Double): String =
        if (!value.isFinite()) "nan"
        else "%.5f".format(java.util.Locale.US, value)
}
