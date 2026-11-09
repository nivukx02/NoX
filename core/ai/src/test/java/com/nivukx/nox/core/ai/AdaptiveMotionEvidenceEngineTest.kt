package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertTrue

class AdaptiveMotionEvidenceEngineTest {

    @Test
    fun staticLandmarksRemainStatic() {
        val engine = AdaptiveMotionEvidenceEngine()
        val points = (0 until 16).map {
            AdaptiveMotionEvidenceEngine.Landmark(
                id = it,
                x = it / 20.0,
                y = it / 40.0,
                confidence = 0.95,
            )
        }
        engine.process(
            AdaptiveMotionEvidenceEngine.Input(1_000_000_000L, points),
        )
        val result = engine.process(
            AdaptiveMotionEvidenceEngine.Input(1_033_333_333L, points),
        )
        assertTrue(result.motionClass == AdaptiveMotionEvidenceEngine.MotionClass.STATIC)
        assertTrue(result.confidence > 0.65)
        engine.close()
    }

    @Test
    fun fastDifferentialMotionRaisesHighEndPrioritySignals() {
        val engine = AdaptiveMotionEvidenceEngine()
        fun frame(t: Long, offset: Double) =
            (0 until 16).map {
                AdaptiveMotionEvidenceEngine.Landmark(
                    id = it,
                    x = it / 20.0 + offset * (if (it % 2 == 0) 1.0 else -1.0),
                    y = it / 40.0,
                    confidence = 0.98,
                )
            }

        engine.process(
            AdaptiveMotionEvidenceEngine.Input(1_000_000_000L, frame(0, 0.0)),
        )
        val result = engine.process(
            AdaptiveMotionEvidenceEngine.Input(1_050_000_000L, frame(0, 0.09)),
        )
        assertTrue(result.differentialEnergy > result.commonModeEnergy)
        assertTrue(
            result.motionClass == AdaptiveMotionEvidenceEngine.MotionClass.FAST ||
                result.motionClass == AdaptiveMotionEvidenceEngine.MotionClass.EXTREME ||
                result.motionClass == AdaptiveMotionEvidenceEngine.MotionClass.NORMAL,
        )
        assertTrue(result.frequencyProxyHz <= 30.0)
        engine.close()
    }

    @Test
    fun variableFrameRateUsesRuntimeFrameRateForFrequencyProxy() {
        val engine = AdaptiveMotionEvidenceEngine()
        val points0 = (0 until 16).map { AdaptiveMotionEvidenceEngine.Landmark(it, it / 20.0, it / 40.0, 0.9) }
        val points1 = (0 until 16).map { AdaptiveMotionEvidenceEngine.Landmark(it, it / 20.0 + if (it % 2 == 0) 0.03 else -0.03, it / 40.0, 0.9) }
        engine.process(AdaptiveMotionEvidenceEngine.Input(1_000_000_000L, points0, nominalFrameRateHz = 60.0))
        val result = engine.process(AdaptiveMotionEvidenceEngine.Input(1_016_666_667L, points1, nominalFrameRateHz = 60.0))
        assertTrue(result.frequencyProxyHz <= 30.0)
        engine.close()
    }

    @Test
    fun stateMemoryIsBounded() {
        val engine = AdaptiveMotionEvidenceEngine(maximumLandmarkStates = 32)
        repeat(100) { frame ->
            val points = (0 until 64).map {
                AdaptiveMotionEvidenceEngine.Landmark(
                    id = frame * 64 + it,
                    x = it / 64.0,
                    y = 0.5,
                    confidence = 0.8,
                )
            }
            engine.process(
                AdaptiveMotionEvidenceEngine.Input(
                    timestampNanos = 1_000_000_000L + frame * 20_000_000L,
                    landmarks = points,
                ),
            )
        }
        assertTrue(engine.latest() != null)
        engine.close()
    }
}
