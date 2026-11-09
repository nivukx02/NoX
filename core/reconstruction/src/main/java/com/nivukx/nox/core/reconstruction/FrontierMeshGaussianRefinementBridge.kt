package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3

/**
 * Converts a metric reconstruction mesh into bounded Gaussian surface evidence.
 * Normals come from the existing topology-safe area-weighted estimator. No synthetic normal is
 * emitted for isolated or degenerate vertices.
 */
class FrontierMeshGaussianRefinementBridge(
    private val gaussian: DynamicGaussianSurfaceRefinementEngine = DynamicGaussianSurfaceRefinementEngine(),
    private val maximumVertices: Int = 16_384,
) {
    init {
        require(maximumVertices > 0)
    }

    data class Result(
        val gaussians: List<DynamicGaussianSurfaceRefinementEngine.Gaussian>,
        val usableVertexCount: Int,
        val rejectedVertexCount: Int,
        val normalCoverage: Double,
    )

    fun refine(
        mesh: ReconstructionMesh,
        surfaceConfidence: Double = 1.0,
        origin: EvidenceOrigin = EvidenceOrigin.OBSERVED,
        flowByVertex: Map<Int, Vec3> = emptyMap(),
    ): Result {
        require(surfaceConfidence in 0.0..1.0)
        require(flowByVertex.values.all { it.x.isFinite() && it.y.isFinite() && it.z.isFinite() })
        if (mesh.vertices.isEmpty() || mesh.triangles.isEmpty()) {
            return Result(emptyList(), 0, 0, 0.0)
        }

        val boundedVertices = mesh.vertices.take(maximumVertices)
        val boundedTriangles = mesh.triangles.filter {
            it.a in boundedVertices.indices &&
                it.b in boundedVertices.indices &&
                it.c in boundedVertices.indices
        }
        val normals = MeshNormalEstimator.estimate(boundedVertices, boundedTriangles)
        val points = boundedVertices.mapIndexedNotNull { index, vertex ->
            val normal = normals.getOrNull(index) ?: return@mapIndexedNotNull null
            if (normal.norm() < 1e-9) return@mapIndexedNotNull null
            val uncertainty = vertex.uncertaintyMeters.coerceAtLeast(1e-6)
            DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                id = meshGaussianId(index),
                positionMeters = vertex.position,
                normal = normal,
                covariance = UncertaintyEnvelope(
                    xxMeters2 = uncertainty * uncertainty,
                    yyMeters2 = uncertainty * uncertainty,
                    zzMeters2 = uncertainty * uncertainty,
                    confidence = surfaceConfidence,
                ),
                confidence = (mesh.confidence * surfaceConfidence).coerceIn(0.0,1.0),
                origin = origin,
                flowDeltaMeters = flowByVertex[index] ?: Vec3(0.0,0.0,0.0),
            )
        }
        val gaussians = gaussian.refine(points)
        val usable = points.size
        return Result(
            gaussians = gaussians,
            usableVertexCount = usable,
            rejectedVertexCount = boundedVertices.size - usable,
            normalCoverage = usable.toDouble() / boundedVertices.size.coerceAtLeast(1).toDouble(),
        )
    }
}