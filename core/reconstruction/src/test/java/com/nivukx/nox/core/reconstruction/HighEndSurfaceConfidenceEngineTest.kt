package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndSurfaceConfidenceEngineTest {

    @Test
    fun wellSupportedSurfaceProducesFiniteConfidence() {
        val mesh = mesh()
        val result = HighEndSurfaceConfidenceEngine().analyze(mesh)

        assertTrue(result.globalConfidence >= 0.0)
        assertTrue(result.globalConfidence <= 1.0)
        assertTrue(result.vertices.size == mesh.vertices.size)
        assertTrue(result.topologyCoverage > 0.0)
        assertTrue(result.diagnostics.all { !it.contains("NaN") })
    }

    @Test
    fun degenerateTriangleIsReported() {
        val mesh = ReconstructionMesh(
            vertices = listOf(
                MeshVertex(Vec3(0.0, 0.0, 0.0)),
                MeshVertex(Vec3(1.0, 0.0, 0.0)),
                MeshVertex(Vec3(2.0, 0.0, 0.0)),
            ),
            triangles = listOf(
                MeshTriangle(0, 1, 2),
            ),
            timestampNanos = 1_000_000_000L,
            confidence = 0.50,
        )

        val result = HighEndSurfaceConfidenceEngine().analyze(mesh)

        assertTrue(result.degenerateTriangleRatio > 0.0)
        assertTrue(
            result.warnings.any {
                it.contains("degenerate") ||
                    it.contains("low")
            },
        )
    }

    private fun mesh(): ReconstructionMesh =
        ReconstructionMesh(
            vertices = listOf(
                MeshVertex(Vec3(-0.5, 0.0, 0.0)),
                MeshVertex(Vec3(0.5, 0.0, 0.0)),
                MeshVertex(Vec3(0.5, 1.0, 0.0)),
                MeshVertex(Vec3(-0.5, 1.0, 0.0)),
                MeshVertex(Vec3(0.0, 0.5, 0.35)),
            ),
            triangles = listOf(
                MeshTriangle(0, 1, 4),
                MeshTriangle(1, 2, 4),
                MeshTriangle(2, 3, 4),
                MeshTriangle(3, 0, 4),
            ),
            timestampNanos = 1_000_000_000L,
            confidence = 0.92,
        )
}
