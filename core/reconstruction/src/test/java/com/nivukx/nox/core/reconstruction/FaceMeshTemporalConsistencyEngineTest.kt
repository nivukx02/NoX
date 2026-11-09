package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMeshTemporalConsistencyEngineTest {

    @Test
    fun rigidTranslationDoesNotLookLikeShapeInstability() {
        val engine = FaceMeshTemporalConsistencyEngine()
        val base = points(offset = Vec3(0.0, 0.0, 0.0))
        engine.evaluate(
            FaceMeshTemporalConsistencyEngine.Input(
                timestampNanos = 1_000_000_000L,
                pointsBySemanticId = base,
                baseConfidence = 0.95,
                meanReprojectionErrorPx = 1.5,
            ),
        )

        val translated = base.mapValues { (_, point) ->
            point + Vec3(0.30, -0.12, 0.18)
        }

        val result = engine.evaluate(
            FaceMeshTemporalConsistencyEngine.Input(
                timestampNanos = 1_016_000_000L,
                pointsBySemanticId = translated,
                baseConfidence = 0.95,
                meanReprojectionErrorPx = 1.5,
            ),
        )

        assertTrue(result.accepted)
        assertTrue(result.confidence > 0.70)
    }

    @Test
    fun suddenContourScaleJumpIsDetected() {
        val engine = FaceMeshTemporalConsistencyEngine()
        val base = points(offset = Vec3(0.0, 0.0, 0.0))
        engine.evaluate(
            FaceMeshTemporalConsistencyEngine.Input(
                timestampNanos = 2_000_000_000L,
                pointsBySemanticId = base,
                baseConfidence = 0.95,
                meanReprojectionErrorPx = 1.5,
            ),
        )

        val enlarged = base.mapValues { (_, point) ->
            point * 1.9
        }

        val result = engine.evaluate(
            FaceMeshTemporalConsistencyEngine.Input(
                timestampNanos = 2_016_000_000L,
                pointsBySemanticId = enlarged,
                baseConfidence = 0.95,
                meanReprojectionErrorPx = 1.5,
            ),
        )

        assertTrue(result.warnings.isNotEmpty())
        assertTrue(result.medianEdgeRatio > 1.5)
    }

    @Test
    fun persistentInstabilityAccumulates() {
        val engine = FaceMeshTemporalConsistencyEngine(
            FaceMeshTemporalConsistencyEngine.Config(
                persistentWarningFrames = 2,
                persistentRejectFrames = 4,
            ),
        )

        var result =
            engine.evaluate(
                input(3_000_000_000L, points()),
            )

        repeat(5) { index ->
            result =
                engine.evaluate(
                    input(
                        3_016_000_000L + index * 16_000_000L,
                        points(scale = 2.0),
                    ),
                )
        }

        assertTrue(result.persistentWarningFrames >= 2)
        assertTrue(result.persistentRejectFrames >= 1)
    }

    private fun input(
        timestamp: Long,
        points: Map<Int, Vec3>,
    ) =
        FaceMeshTemporalConsistencyEngine.Input(
            timestampNanos = timestamp,
            pointsBySemanticId = points,
            baseConfidence = 0.90,
            meanReprojectionErrorPx = 2.0,
        )

    private fun points(
        offset: Vec3 = Vec3(0.0, 0.0, 0.0),
        scale: Double = 1.0,
    ): Map<Int, Vec3> =
        (0 until 12).associate { index ->
            val contour = index / 4
            val local = index % 4
            val point = Vec3(
                (local - 1.5) * 0.012 * scale,
                (contour - 1.5) * 0.010 * scale,
                (local - 1.5) * 0.006 * scale,
            )
            contour * 1000 + local to point + offset
        }
}
