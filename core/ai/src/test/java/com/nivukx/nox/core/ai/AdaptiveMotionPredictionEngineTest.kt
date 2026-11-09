package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import kotlin.test.assertTrue
import org.junit.Test

class AdaptiveMotionPredictionEngineTest {

    @Test
    fun predictionCarriesVelocityThroughShortOcclusion() {
        val engine = AdaptiveMotionPredictionEngine()
        engine.initialize(
            measurement = Vec2(0.40, 0.50),
            timestampNanos = 1_000_000_000L,
            observationConfidence = 0.95,
        )
        engine.update(
            measurement = Vec2(0.45, 0.50),
            timestampNanos = 1_016_666_667L,
            observationConfidence = 0.95,
        )

        val predicted =
            engine.predict(1_050_000_000L)

        assertTrue(predicted.position.x > 0.45)
        assertTrue(predicted.confidence > 0.0)
    }

    @Test
    fun largeInnovationReducesCorrectionWeight() {
        val engine = AdaptiveMotionPredictionEngine()
        engine.initialize(
            measurement = Vec2(0.50, 0.50),
            timestampNanos = 1_000_000_000L,
            observationConfidence = 0.95,
        )
        engine.update(
            measurement = Vec2(0.501, 0.50),
            timestampNanos = 1_016_666_667L,
            observationConfidence = 0.95,
        )
        val result =
            engine.update(
                measurement = Vec2(0.95, 0.50),
                timestampNanos = 1_033_333_334L,
                observationConfidence = 0.95,
            )
        assertTrue(result.innovation > 0.20)
        assertTrue(result.correctionWeight < 0.60)
    }

    @Test
    fun resetClearsState() {
        val engine = AdaptiveMotionPredictionEngine()
        engine.initialize(
            measurement = Vec2(0.50, 0.50),
            timestampNanos = 1_000_000_000L,
            observationConfidence = 0.95,
        )
        engine.reset()
        assertTrue(!engine.isInitialized())
    }
}
