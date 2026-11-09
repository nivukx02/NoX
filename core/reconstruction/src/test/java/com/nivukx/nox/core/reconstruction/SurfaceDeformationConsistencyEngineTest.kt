package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SurfaceDeformationConsistencyEngineTest {

    private fun frame(
        timestamp: Long,
        offset: Double = 0.0,
        strain: Boolean = false,
    ): SurfaceDeformationConsistencyEngine.Frame {
        val vertices = (0L until 12L).map { id ->
            val base = id.toDouble() * 0.01
            SurfaceDeformationConsistencyEngine.Vertex(
                id = id,
                position =
                    SurfaceDeformationConsistencyEngine.Vec3(
                        x = base + offset + if (strain && id % 2L == 0L) 0.08 else 0.0,
                        y = id * 0.005,
                        z = 1.0 + id * 0.002,
                    ),
                normal =
                    SurfaceDeformationConsistencyEngine.Vec3(
                        0.0,
                        0.0,
                        1.0,
                    ),
                curvature = 0.1,
                confidence = 0.98,
                visible = true,
            )
        }

        val edges = (0L until 11L).map {
            SurfaceDeformationConsistencyEngine.Edge(
                firstVertexId = it,
                secondVertexId = it + 1L,
            )
        }

        return SurfaceDeformationConsistencyEngine.Frame(
            timestampNanos = timestamp,
            vertices = vertices,
            edges = edges,
            sourceConfidence = 0.98,
        )
    }

    @Test
    fun stableSurfaceBuildsTemporalConfidence() {
        val engine = SurfaceDeformationConsistencyEngine()

        val first = engine.evaluate(frame(1_000_000_000L))
        assertFalse(first.warmedUp)

        val second = engine.evaluate(frame(1_016_666_667L))
        assertTrue(second.warmedUp)
        assertTrue(second.accepted)
        assertTrue(second.topologyContinuity > 0.90)
        assertTrue(second.temporalStability > 0.60)
    }

    @Test
    fun largeStrainIsDetectedAndBlocksAcceptance() {
        val engine = SurfaceDeformationConsistencyEngine()

        engine.evaluate(frame(2_000_000_000L))
        val result = engine.evaluate(
            frame(
                timestamp = 2_016_666_667L,
                strain = true,
            ),
        )

        assertTrue(result.strainP95Ratio > 0.20)
        assertTrue(
            result.warnings.any {
                it.contains("surface-strain")
            },
        )
    }

    @Test
    fun staleFrameDoesNotProduceFalseTemporalConfidence() {
        val engine = SurfaceDeformationConsistencyEngine()
        engine.evaluate(frame(3_000_000_000L))

        val stale = engine.evaluate(
            frame(2_900_000_000L),
        )

        assertTrue(stale.staleFrameRejected)
        assertFalse(stale.accepted)
        assertTrue(
            stale.warnings.contains(
                "surface-temporal-history-reset",
            ),
        )
    }

    @Test
    fun disappearingVerticesReduceVisibleCoverage() {
        val engine = SurfaceDeformationConsistencyEngine()
        engine.evaluate(frame(4_000_000_000L))

        val base = frame(4_016_666_667L)
        val partiallyVisible =
            base.copy(
                vertices =
                    base.vertices.mapIndexed { index, vertex ->
                        vertex.copy(
                            visible = index < 4,
                        )
                    },
            )

        val result = engine.evaluate(partiallyVisible)
        assertTrue(result.visibleCoverage < 0.50)
        assertTrue(
            result.warnings.contains(
                "surface-visible-coverage-low",
            ),
        )
    }
}
