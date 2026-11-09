package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MeshTopologyQualityEngineTest {

    private val vertices = listOf(
        MeshVertex(Vec3(0.0, 0.0, 0.0)),
        MeshVertex(Vec3(1.0, 0.0, 0.0)),
        MeshVertex(Vec3(0.0, 1.0, 0.0)),
        MeshVertex(Vec3(0.0, 0.0, 1.0)),
    )

    @Test
    fun nonManifoldEdge_isDetectedAndExtraFaceIsRemoved() {
        val engine = MeshTopologyQualityEngine()

        val report = engine.sanitize(
            vertices = vertices,
            triangles = listOf(
                MeshTriangle(0, 1, 2),
                MeshTriangle(0, 1, 3),
                MeshTriangle(0, 1, 2),
            ),
        )

        assertEquals(1, report.nonManifoldEdgeCount)
        assertEquals(1, report.duplicateTriangleCount)
        assertTrue(report.sanitizedTriangles.size <= 2)
        assertTrue(report.topologyIntegrityScore < 1.0)
    }

    @Test
    fun boundaryEdges_areMeasuredWithoutBeingTreatedAsInvalid() {
        val engine = MeshTopologyQualityEngine()

        val report = engine.sanitize(
            vertices = vertices,
            triangles = listOf(
                MeshTriangle(0, 1, 2),
            ),
        )

        assertTrue(report.boundaryEdgeCount > 0)
        assertEquals(0, report.nonManifoldEdgeCount)
        assertEquals(
            1,
            report.sanitizedTriangles.size,
        )
    }

    @Test
    fun invalidTriangleIndices_areRejectedDeterministically() {
        val engine = MeshTopologyQualityEngine()

        val report = engine.sanitize(
            vertices = vertices,
            triangles = listOf(
                MeshTriangle(0, 1, 99),
                MeshTriangle(0, 1, 2),
            ),
        )

        assertEquals(1, report.invalidTriangleCount)
        assertEquals(1, report.sanitizedTriangles.size)
    }
}
