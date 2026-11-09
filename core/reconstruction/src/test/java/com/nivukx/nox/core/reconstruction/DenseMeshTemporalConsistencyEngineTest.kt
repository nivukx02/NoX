package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseMeshTemporalConsistencyEngineTest {

    @Test
    fun stableTranslatedMeshRemainsStable() {
        val engine = DenseMeshTemporalConsistencyEngine()
        val first = mesh(0.0)
        engine.evaluate(
            DenseMeshTemporalConsistencyEngine.Input(
                timestampNanos = 1_000_000_000L,
                mesh = first,
                upstreamConfidence = 0.95,
            ),
        )

        val moved = mesh(0.20)
        val result = engine.evaluate(
            DenseMeshTemporalConsistencyEngine.Input(
                timestampNanos = 1_016_000_000L,
                mesh = moved,
                upstreamConfidence = 0.95,
            ),
        )

        assertTrue(result.accepted)
        assertTrue(result.confidence > 0.65)
    }

    @Test
    fun scaleJumpIsDetected() {
        val engine = DenseMeshTemporalConsistencyEngine()
        val first = mesh(0.0, scale = 1.0)
        engine.evaluate(
            DenseMeshTemporalConsistencyEngine.Input(
                timestampNanos = 2_000_000_000L,
                mesh = first,
                upstreamConfidence = 0.95,
            ),
        )

        val result = engine.evaluate(
            DenseMeshTemporalConsistencyEngine.Input(
                timestampNanos = 2_016_000_000L,
                mesh = mesh(0.0, scale = 1.5),
                upstreamConfidence = 0.95,
            ),
        )

        assertTrue(result.warnings.isNotEmpty())
        assertTrue(result.scaleChangeRatio > 1.15)
    }

    private fun mesh(
        offset: Double,
        scale: Double = 1.0,
    ): ReconstructionMesh {
        val vertices = (0 until 30).map { index ->
            val x = (index % 5) * 0.02 * scale + offset
            val y = (index / 5) * 0.02 * scale
            val z = ((index * 7) % 5) * 0.012 * scale
            MeshVertex(Vec3(x, y, z))
        }

        val triangles = buildList {
            for (row in 0 until 5) {
                for (column in 0 until 4) {
                    val a = row * 5 + column
                    val b = a + 1
                    val c = a + 5
                    val d = c + 1
                    add(MeshTriangle(a, b, c))
                    add(MeshTriangle(b, d, c))
                }
            }
        }

        return ReconstructionMesh(
            vertices = vertices,
            triangles = triangles,
            timestampNanos = 0L,
            confidence = 0.95,
        )
    }
}
