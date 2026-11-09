package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndFaceShapeAlignmentEngineTest {

    @Test
    fun rigidRotationPreservesShapeSignature() {
        val engine = HighEndFaceShapeAlignmentEngine()
        val first = points()
        val rotation = listOf(
            Vec3(0.80, 0.0, 0.60),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-0.60, 0.0, 0.80),
        )
        val rotated = first.mapValues { (_, value) ->
            Vec3(
                rotation[0].x * value.x +
                    rotation[0].z * value.z,
                value.y,
                rotation[2].x * value.x +
                    rotation[2].z * value.z,
            )
        }

        engine.evaluate(
            HighEndFaceShapeAlignmentEngine.Input(
                timestampNanos = 1_000_000_000L,
                pointsBySemanticId = first,
                baseConfidence = 0.95,
            ),
        )

        val result = engine.evaluate(
            HighEndFaceShapeAlignmentEngine.Input(
                timestampNanos = 1_016_666_667L,
                pointsBySemanticId = rotated,
                baseConfidence = 0.95,
            ),
        )

        assertTrue(result.confidence > 0.55)
        assertTrue(result.anisotropyDeviation < 0.25)
    }

    @Test
    fun severeDepthStretchReducesShapeConfidence() {
        val engine = HighEndFaceShapeAlignmentEngine()
        val first = points()

        engine.evaluate(
            HighEndFaceShapeAlignmentEngine.Input(
                timestampNanos = 2_000_000_000L,
                pointsBySemanticId = first,
                baseConfidence = 0.95,
            ),
        )

        val stretched = first.mapValues { (_, value) ->
            Vec3(
                value.x,
                value.y,
                value.z * 3.8,
            )
        }

        val result = engine.evaluate(
            HighEndFaceShapeAlignmentEngine.Input(
                timestampNanos = 2_016_666_667L,
                pointsBySemanticId = stretched,
                baseConfidence = 0.95,
            ),
        )

        assertTrue(result.confidence < 0.72)
        assertTrue(
            result.warnings.any {
                it.contains("anisotropy") ||
                    it.contains("scale")
            },
        )
    }

    private fun points(): Map<Int, Vec3> = mapOf(
        0 to Vec3(-0.08, 0.04, 0.00),
        1 to Vec3(-0.05, 0.10, 0.03),
        2 to Vec3(0.00, 0.12, 0.04),
        3 to Vec3(0.05, 0.10, 0.03),
        4 to Vec3(0.08, 0.04, 0.00),
        5 to Vec3(-0.10, 0.00, 0.01),
        6 to Vec3(-0.05, -0.02, 0.02),
        7 to Vec3(0.00, -0.03, 0.03),
        8 to Vec3(0.05, -0.02, 0.02),
        9 to Vec3(0.10, 0.00, 0.01),
        10 to Vec3(-0.05, 0.05, -0.03),
        11 to Vec3(0.05, 0.05, -0.03),
        12 to Vec3(-0.035, -0.05, 0.00),
        13 to Vec3(0.035, -0.05, 0.00),
        14 to Vec3(0.00, -0.08, 0.01),
        15 to Vec3(0.00, 0.02, 0.05),
    )
}
