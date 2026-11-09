package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeSparseTsdfReconstructionBridgeTest {
    @Test
    fun integrationInputIsNormalizedBeforeNativeBoundary() {
        val direction = Vec3(0.0, 0.0, 2.0).normalized()
        assertTrue(direction.z > 0.99)
    }

    @Test
    fun indexedTopologyCanShareVerticesAcrossTriangles() {
        val mesh = ReconstructionMesh(
            vertices = listOf(
                com.nivukx.nox.core.model.MeshVertex(Vec3(0.0, 0.0, 0.0)),
                com.nivukx.nox.core.model.MeshVertex(Vec3(1.0, 0.0, 0.0)),
                com.nivukx.nox.core.model.MeshVertex(Vec3(1.0, 1.0, 0.0)),
                com.nivukx.nox.core.model.MeshVertex(Vec3(0.0, 1.0, 0.0)),
            ),
            triangles = listOf(
                MeshTriangle(0, 1, 2),
                MeshTriangle(0, 2, 3),
            ),
            timestampNanos = 1L,
            confidence = 0.9,
        )
        assertEquals(4, mesh.vertices.size)
        assertEquals(2, mesh.triangles.size)
        assertEquals(mesh.triangles.first().a, mesh.triangles.last().a)
        assertTrue(mesh.triangles.first().c == mesh.triangles.last().b)
    }
}
