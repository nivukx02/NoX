package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max

/**
 * Deterministic topology-quality gate for reconstructed triangle meshes.
 *
 * It distinguishes legitimate open boundaries from invalid topology. Duplicate faces and
 * non-manifold edges are removed from the export candidate, while boundary edges remain measurable
 * evidence rather than being silently filled with invented geometry.
 */
class MeshTopologyQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumTrianglesPerEdge: Int = 2,
        val boundaryPenaltyScale: Double = 0.18,
        val nonManifoldPenaltyScale: Double = 0.35,
    ) {
        init {
            require(maximumTrianglesPerEdge >= 2)
            require(boundaryPenaltyScale > 0.0)
            require(nonManifoldPenaltyScale > 0.0)
        }
    }

    data class Report(
        val sanitizedTriangles: List<MeshTriangle>,
        val boundaryEdgeCount: Int,
        val nonManifoldEdgeCount: Int,
        val duplicateTriangleCount: Int,
        val invalidTriangleCount: Int,
        val isolatedVertexCount: Int,
        val topologyIntegrityScore: Double,
    )

    private data class Edge(
        val a: Int,
        val b: Int,
    ) {
        companion object {
            fun of(a: Int, b: Int): Edge =
                if (a < b) {
                    Edge(a, b)
                } else {
                    Edge(b, a)
                }
        }
    }

    private data class TriangleKey(
        val a: Int,
        val b: Int,
        val c: Int,
    ) {
        companion object {
            fun of(triangle: MeshTriangle): TriangleKey {
                val sorted =
                    listOf(
                        triangle.a,
                        triangle.b,
                        triangle.c,
                    ).sorted()
                return TriangleKey(
                    a = sorted[0],
                    b = sorted[1],
                    c = sorted[2],
                )
            }
        }
    }

    fun sanitize(
        vertices: List<MeshVertex>,
        triangles: List<MeshTriangle>,
    ): Report {
        val valid = ArrayList<MeshTriangle>()
        var invalidCount = 0
        var duplicateCount = 0
        val seenTriangles = HashSet<TriangleKey>()
        val rawEdgeOwners = HashMap<Edge, Int>()

        triangles.forEach { triangle ->
            if (
                triangle.a !in vertices.indices ||
                triangle.b !in vertices.indices ||
                triangle.c !in vertices.indices ||
                triangle.a == triangle.b ||
                triangle.b == triangle.c ||
                triangle.a == triangle.c
            ) {
                invalidCount += 1
                return@forEach
            }

            listOf(
                Edge.of(triangle.a, triangle.b),
                Edge.of(triangle.b, triangle.c),
                Edge.of(triangle.c, triangle.a),
            ).forEach { edge ->
                rawEdgeOwners[edge] =
                    (rawEdgeOwners[edge] ?: 0) + 1
            }

            val key = TriangleKey.of(triangle)
            if (!seenTriangles.add(key)) {
                duplicateCount += 1
                return@forEach
            }

            valid += triangle
        }

        val edgeOwners =
            HashMap<Edge, MutableList<Int>>()

        valid.forEachIndexed { index, triangle ->
            listOf(
                Edge.of(triangle.a, triangle.b),
                Edge.of(triangle.b, triangle.c),
                Edge.of(triangle.c, triangle.a),
            ).forEach { edge ->
                edgeOwners
                    .getOrPut(edge) { mutableListOf() }
                    .add(index)
            }
        }

        var nonManifoldEdges =
            rawEdgeOwners.values.count {
                it > config.maximumTrianglesPerEdge
            }
        val rejectedTriangleIndexes = HashSet<Int>()

        edgeOwners.forEach { (_, owners) ->
            if (owners.size > config.maximumTrianglesPerEdge) {
                owners
                    .drop(config.maximumTrianglesPerEdge)
                    .forEach {
                        rejectedTriangleIndexes += it
                    }
            }
        }

        val sanitized =
            valid.filterIndexed { index, _ ->
                index !in rejectedTriangleIndexes
            }

        val sanitizedEdgeOwners =
            HashMap<Edge, Int>()

        sanitized.forEach { triangle ->
            listOf(
                Edge.of(triangle.a, triangle.b),
                Edge.of(triangle.b, triangle.c),
                Edge.of(triangle.c, triangle.a),
            ).forEach { edge ->
                sanitizedEdgeOwners[edge] =
                    (sanitizedEdgeOwners[edge] ?: 0) + 1
            }
        }

        val boundaryEdges =
            sanitizedEdgeOwners.values.count { it == 1 }

        val usedVertices =
            BooleanArray(vertices.size)
        sanitized.forEach { triangle ->
            usedVertices[triangle.a] = true
            usedVertices[triangle.b] = true
            usedVertices[triangle.c] = true
        }

        val isolatedVertices =
            usedVertices.count { !it }

        val boundaryRatio =
            boundaryEdges.toDouble() /
                max(
                    1.0,
                    sanitizedEdgeOwners.size.toDouble(),
                )

        val nonManifoldPenalty =
            exp(
                -nonManifoldEdges *
                    config.nonManifoldPenaltyScale,
            ).coerceIn(0.0, 1.0)

        val boundaryPenalty =
            exp(
                -boundaryRatio /
                    config.boundaryPenaltyScale,
            ).coerceIn(0.0, 1.0)

        val invalidPenalty =
            if (triangles.isEmpty()) {
                1.0
            } else {
                (
                    1.0 -
                        invalidCount.toDouble() /
                            triangles.size.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val duplicatePenalty =
            if (triangles.isEmpty()) {
                1.0
            } else {
                (
                    1.0 -
                        duplicateCount.toDouble() /
                            triangles.size.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val integrity = (
            nonManifoldPenalty * 0.42 +
                boundaryPenalty * 0.18 +
                invalidPenalty * 0.20 +
                duplicatePenalty * 0.15 +
                if (isolatedVertices == 0) 0.05 else 0.0
            ).coerceIn(0.0, 1.0)

        return Report(
            sanitizedTriangles = sanitized,
            boundaryEdgeCount = boundaryEdges,
            nonManifoldEdgeCount = nonManifoldEdges,
            duplicateTriangleCount = duplicateCount,
            invalidTriangleCount = invalidCount,
            isolatedVertexCount = isolatedVertices,
            topologyIntegrityScore = integrity,
        )
    }
}
