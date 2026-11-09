package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.Vec3
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MeshNormalEstimatorTest {

    @Test
    fun planarMeshProducesConsistentUnitNormals() {
        val vertices = listOf(
            MeshVertex(Vec3(0.0, 0.0, 0.0)),
            MeshVertex(Vec3(1.0, 0.0, 0.0)),
            MeshVertex(Vec3(1.0, 1.0, 0.0)),
            MeshVertex(Vec3(0.0, 1.0, 0.0)),
        )
        val triangles = listOf(
            MeshTriangle(0, 1, 2),
            MeshTriangle(0, 2, 3),
        )

        val normals = MeshNormalEstimator.estimate(vertices, triangles)

        assertEquals(vertices.size, normals.size)
        normals.forEach { normal ->
            assertEquals(1.0, normal.norm(), 1e-6)
            assertTrue(abs(normal.z) > 0.99)
        }
    }

    @Test
    fun isolatedAndInvalidTrianglesDoNotCreateNanNormals() {
        val vertices = listOf(
            MeshVertex(Vec3(0.0, 0.0, 0.0)),
            MeshVertex(Vec3(1.0, 0.0, 0.0)),
        )
        val triangles = listOf(
            MeshTriangle(0, 0, 1),
            MeshTriangle(-1, 0, 1),
        )

        val normals = MeshNormalEstimator.estimate(vertices, triangles)

        assertEquals(vertices.size, normals.size)
        normals.forEach { normal ->
            assertTrue(normal.x.isFinite())
            assertTrue(normal.y.isFinite())
            assertTrue(normal.z.isFinite())
            assertEquals(0.0, normal.norm(), 1e-9)
        }
    }

    private fun abs(value: Double): Double =
        kotlin.math.abs(value)
}
