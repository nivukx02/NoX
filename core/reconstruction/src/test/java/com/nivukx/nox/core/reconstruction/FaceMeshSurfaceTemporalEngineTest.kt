package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.FaceMeshTriangle
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMeshSurfaceTemporalEngineTest {

    private fun surface(
        timestampNanos: Long,
        xOffset: Double = 0.0,
        confidence: Double = 0.92,
    ): FaceMeshMultiViewFusionEngine.Result {
        val points =
            (0 until 6).map { index ->
                FaceMeshMultiViewFusionEngine.SurfacePoint(
                    index = index,
                    positionWorld = Vec3(
                        x = index * 0.01 + xOffset,
                        y = (index % 2) * 0.01,
                        z = 0.45 + index * 0.002,
                    ),
                    confidence = confidence,
                    uncertaintyMeters = 0.003,
                    reprojectionErrorPx = 0.8,
                    parallaxDegrees = 2.2,
                    sourceCount = 3,
                    conditionScore = 0.91,
                )
            }

        return FaceMeshMultiViewFusionEngine.Result(
            timestampNanos = timestampNanos,
            points = points,
            triangles = listOf(
                FaceMeshTriangle(0, 1, 2),
                FaceMeshTriangle(2, 3, 4),
                FaceMeshTriangle(1, 4, 5),
            ),
            confidence = confidence,
            meanReprojectionErrorPx = 0.8,
            p95ReprojectionErrorPx = 1.5,
            meanUncertaintyMeters = 0.003,
            acceptedPointRatio = 1.0,
            sourceDevices = emptyList(),
            sourceFrameCount = 3,
            warnings = emptyList(),
        )
    }

    @Test
    fun firstSurfacePassIsStableAndFinite() {
        val engine = FaceMeshSurfaceTemporalEngine()

        val result =
            engine.process(
                FaceMeshSurfaceTemporalEngine.Input(
                    timestampNanos = 1_000_000_000L,
                    surface = surface(1_000_000_000L),
                ),
            )

        assertTrue(result.surface.points.isNotEmpty())
        assertTrue(result.diagnostics.finalScore > 0.5)
        assertTrue(
            result.surface.points.all {
                it.positionWorld.x.isFinite() &&
                    it.positionWorld.y.isFinite() &&
                    it.positionWorld.z.isFinite()
            },
        )
    }

    @Test
    fun smallMetricMotionIsNotCollapsedToZero() {
        val engine = FaceMeshSurfaceTemporalEngine()

        engine.process(
            FaceMeshSurfaceTemporalEngine.Input(
                timestampNanos = 1_000_000_000L,
                surface = surface(1_000_000_000L),
            ),
        )

        val result =
            engine.process(
                FaceMeshSurfaceTemporalEngine.Input(
                    timestampNanos = 1_016_666_666L,
                    surface = surface(
                        timestampNanos = 1_016_666_666L,
                        xOffset = 0.003,
                    ),
                    preserveMicroMotion = true,
                ),
            )

        val first =
            result.surface.points.first {
                it.index == 0
            }

        assertTrue(first.positionWorld.x > 0.0005)
        assertTrue(result.diagnostics.microMotionRetention >= 0.0)
    }

    @Test
    fun largeOutlierDoesNotTeleportWholeSurface() {
        val engine = FaceMeshSurfaceTemporalEngine()

        engine.process(
            FaceMeshSurfaceTemporalEngine.Input(
                timestampNanos = 1_000_000_000L,
                surface = surface(1_000_000_000L),
            ),
        )

        val outlier =
            surface(
                timestampNanos = 1_016_666_666L,
                xOffset = 1.0,
            )

        val result =
            engine.process(
                FaceMeshSurfaceTemporalEngine.Input(
                    timestampNanos = 1_016_666_666L,
                    surface = outlier,
                ),
            )

        val first =
            result.surface.points.first {
                it.index == 0
            }

        assertTrue(first.positionWorld.x < 0.20)
        assertTrue(
            result.diagnostics.rejectedPoints >= 0,
        )
    }

    @Test
    fun timestampRegressionResetsState() {
        val engine = FaceMeshSurfaceTemporalEngine()

        engine.process(
            FaceMeshSurfaceTemporalEngine.Input(
                timestampNanos = 2_000_000_000L,
                surface = surface(2_000_000_000L, xOffset = 0.3),
            ),
        )

        val result =
            engine.process(
                FaceMeshSurfaceTemporalEngine.Input(
                    timestampNanos = 1_000_000_000L,
                    surface = surface(1_000_000_000L, xOffset = 0.0),
                ),
            )

        assertTrue(
            kotlin.math.abs(
                result.surface.points.first().positionWorld.x,
            ) < 0.05,
        )
    }
}
