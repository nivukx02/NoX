package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierMeshGaussianRefinementBridgeTest {
    @Test fun metricTriangleProducesGaussianSurfaceEvidence() {
        val mesh=ReconstructionMesh(
            vertices=listOf(
                MeshVertex(Vec3(0.0,0.0,0.0),0.001),
                MeshVertex(Vec3(1.0,0.0,0.0),0.001),
                MeshVertex(Vec3(0.0,1.0,0.0),0.001),
            ),
            triangles=listOf(MeshTriangle(0,1,2)),
            timestampNanos=10L,
            confidence=0.95,
        )
        val result=FrontierMeshGaussianRefinementBridge().refine(mesh,origin=EvidenceOrigin.OBSERVED)
        assertEquals(3,result.usableVertexCount)
        assertEquals(1.0,result.normalCoverage,1e-9)
        assertEquals(3,result.gaussians.size)
        assertTrue(result.gaussians.all { it.covariance.xxMeters2 > 0.0 })
        assertTrue(result.gaussians.all { it.id >= (1L shl 60) })
    }
}