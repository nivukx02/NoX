package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class DynamicStaticSceneDecompositionEngineTest {

    @Test
    fun persistentStableSampleBecomesStatic() {
        val engine = DynamicStaticSceneDecompositionEngine()
        val point = Vec3(1.0, 2.0, 3.0)

        var result = engine.evaluate(
            timestampNanos = 1_000_000_000L,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 1L,
                    position = point,
                    timestampNanos = 1_000_000_000L,
                    confidence = 0.98,
                ),
            ),
        )

        repeat(6) { index ->
            result = engine.evaluate(
                timestampNanos = 1_033_000_000L + index * 33_000_000L,
                samples = listOf(
                    DynamicStaticSceneDecompositionEngine.Sample(
                        id = 1L,
                        position = point,
                        timestampNanos = 1_033_000_000L + index * 33_000_000L,
                        confidence = 0.98,
                    ),
                ),
            )
        }

        assertTrue(result.staticCoverage > 0.0)
        assertTrue(
            result.states.single().label ==
                DynamicStaticSceneDecompositionEngine.Label.STATIC,
        )
    }

    @Test
    fun persistentMovingSampleBecomesDynamic() {
        val engine = DynamicStaticSceneDecompositionEngine()
        var timestamp = 2_000_000_000L
        var x = 0.0
        var result = engine.evaluate(
            timestampNanos = timestamp,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 5L,
                    position = Vec3(x, 0.0, 0.0),
                    timestampNanos = timestamp,
                    confidence = 0.95,
                ),
            ),
        )

        repeat(8) {
            timestamp += 33_000_000L
            x += 0.035
            result = engine.evaluate(
                timestampNanos = timestamp,
                samples = listOf(
                    DynamicStaticSceneDecompositionEngine.Sample(
                        id = 5L,
                        position = Vec3(x, 0.0, 0.0),
                        timestampNanos = timestamp,
                        confidence = 0.95,
                    ),
                ),
            )
        }

        assertTrue(result.dynamicCoverage > 0.0)
        assertTrue(
            result.states.single().label ==
                DynamicStaticSceneDecompositionEngine.Label.DYNAMIC,
        )
    }

    @Test
    fun normalChangeCanRevealDeformation() {
        val engine = DynamicStaticSceneDecompositionEngine(
            DynamicStaticSceneDecompositionEngine.Config(
                dynamicPersistence = 2,
                staticNormalChangeDegrees = 3.0,
                dynamicNormalChangeDegrees = 12.0,
            ),
        )
        val t0 = 4_000_000_000L

        engine.evaluate(
            timestampNanos = t0,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 12L,
                    position = Vec3(0.0, 0.0, 1.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    timestampNanos = t0,
                    confidence = 0.98,
                ),
            ),
        )

        val result = engine.evaluate(
            timestampNanos = t0 + 99_000_000L,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 12L,
                    position = Vec3(0.001, 0.0, 1.0),
                    normal = Vec3(0.82, 0.0, 0.57),
                    timestampNanos = t0 + 99_000_000L,
                    confidence = 0.98,
                ),
            ),
        )

        assertTrue(
            result.states.single().dynamicProbability >
                result.states.single().staticProbability,
        )
    }

    @Test
    fun resultExposesNormalChangeEvidence() {
        val engine = DynamicStaticSceneDecompositionEngine(
            DynamicStaticSceneDecompositionEngine.Config(
                staticNormalChangeDegrees = 2.0,
                dynamicNormalChangeDegrees = 10.0,
            ),
        )
        val t0 = 3_500_000_000L

        engine.evaluate(
            timestampNanos = t0,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 20L,
                    position = Vec3(0.0, 0.0, 1.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    timestampNanos = t0,
                    confidence = 0.95,
                ),
            ),
        )

        val result = engine.evaluate(
            timestampNanos = t0 + 33_000_000L,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 20L,
                    position = Vec3(0.0, 0.0, 1.0),
                    normal = Vec3(0.50, 0.0, 0.866),
                    timestampNanos = t0 + 33_000_000L,
                    confidence = 0.95,
                ),
            ),
        )

        assertTrue(
            result.states.single().normalChangeDegrees > 0.0,
        )
        assertTrue(result.meanNormalChangeDegrees > 0.0)
    }

    fun cameraMotionCompensationReducesFalseDynamicClassification() {
        val engine = DynamicStaticSceneDecompositionEngine()
        val timestamp0 = 3_000_000_000L
        val timestamp1 = timestamp0 + 33_000_000L

        engine.evaluate(
            timestampNanos = timestamp0,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 8L,
                    position = Vec3(0.0, 0.0, 0.0),
                    timestampNanos = timestamp0,
                    confidence = 0.97,
                ),
            ),
        )

        val result = engine.evaluate(
            timestampNanos = timestamp1,
            samples = listOf(
                DynamicStaticSceneDecompositionEngine.Sample(
                    id = 8L,
                    position = Vec3(0.029, 0.0, 0.0),
                    timestampNanos = timestamp1,
                    confidence = 0.97,
                ),
            ),
            cameraMotionMetersPerSecond = Vec3(0.88, 0.0, 0.0),
        )

        assertTrue(
            result.states.single().dynamicProbability <
                result.states.single().staticProbability + 0.40,
        )
    }
}
