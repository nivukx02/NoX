package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.BenchmarkMetricSample
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MotionCaptureBenchmarkEvaluatorTest {

    @Test
    fun computesTailErrorAndIdentityMetrics() {
        val samples =
            (0 until 100).map { index ->
                BenchmarkMetricSample(
                    timestampNanos = index * 16_666_667L,
                    actorId = 1L,
                    jointId = 0,
                    referenceX = 0.0,
                    referenceY = 0.0,
                    referenceZ = 1.0,
                    estimatedX =
                        if (index == 99) {
                            0.001
                        } else {
                            0.0001
                        },
                    estimatedY = 0.0,
                    estimatedZ = 1.0,
                    confidence = 0.98,
                    visible = true,
                    predicted = false,
                    identityCorrect = index != 98,
                )
            }

        val metrics =
            MotionCaptureBenchmarkEvaluator().evaluate(
                samples = samples,
                totalExpectedSamples = 100,
                frameLatenciesMilliseconds =
                    List(100) { 12.0 },
                sustainedFps = 60.0,
                thermalRiseCelsius = 5.0,
                droppedFrameRatio = 0.0,
            )

        assertEquals(100, metrics.sampleCount)
        assertEquals(1, metrics.actorCount)
        assertTrue(metrics.p99ErrorMillimeters >= metrics.medianErrorMillimeters)
        assertTrue(metrics.identityAccuracy < 1.0)
        assertEquals(1.0, metrics.reconstructionCoverage)
    }
}
