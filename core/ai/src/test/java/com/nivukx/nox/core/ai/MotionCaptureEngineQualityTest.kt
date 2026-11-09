package com.nivukx.nox.core.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class MotionCaptureEngineQualityTest {
    @Test
    fun exposesMicroMotionAndDirectionReversalMetrics() {
        val engine = MotionCaptureEngine()
        val landmarks = List(16) { id ->
            PoseDetectionEngine.Landmark(
                id = id,
                name = "L$id",
                x = 500f + id,
                y = 400f + id,
                confidence = 0.96f,
            )
        }

        fun result(
            timestamp: Long,
            dx: Float,
        ) = PoseDetectionEngine.Result(
            timestampNanos = timestamp,
            imageWidth = 1000,
            imageHeight = 1000,
            landmarks = landmarks.map {
                it.copy(
                    x = it.x + dx,
                    y = it.y,
                )
            },
            confidence = 0.96f,
        )

        engine.refine(result(0L, 0f))
        engine.refine(result(33_333_333L, 0.35f))
        engine.refine(result(66_666_666L, 0.70f))
        val reversal =
            engine.refine(
                result(99_999_999L, 0.20f),
            )

        assertTrue(
            "micro-motion should remain measurable",
            reversal.quality.microMotionScore > 0.0,
        )
        assertTrue(
            "direction reversal should be measurable",
            reversal.quality.directionChangeRatio > 0.0,
        )

        engine.close()
    }

    @Test
    fun reacquiresRenumberedLandmarkAgainstPredictedTrackPosition() {
        val engine = MotionCaptureEngine()

        val initial =
            PoseDetectionEngine.Result(
                timestampNanos = 0L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 7,
                        name = "wrist",
                        x = 200f,
                        y = 300f,
                        confidence = 0.98f,
                    ),
                ),
                confidence = 0.98f,
            )

        engine.refine(initial)

        val moving =
            initial.copy(
                timestampNanos = 33_333_333L,
                landmarks = listOf(
                    initial.landmarks.first().copy(
                        x = 250f,
                        y = 300f,
                    ),
                ),
            )
        engine.refine(moving)

        val renumbered =
            PoseDetectionEngine.Result(
                timestampNanos = 66_666_666L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 91,
                        name = "wrist",
                        x = 300f,
                        y = 300f,
                        confidence = 0.96f,
                    ),
                ),
                confidence = 0.96f,
            )

        val result = engine.refine(renumbered)

        assertTrue(
            "renumbered measurement should attach to the existing track",
            result.result.landmarks.any { it.id == 91 },
        )
        assertTrue(
            "track should not be duplicated after reassociation",
            result.result.landmarks.count { it.id == 91 } == 1,
        )

        engine.close()
    }

    @Test
    fun noisyMeasurementRaisesNormalizedMeasurementUncertainty() {
        val engine = MotionCaptureEngine()

        val base =
            PoseDetectionEngine.Result(
                timestampNanos = 0L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = List(8) { id ->
                    PoseDetectionEngine.Landmark(
                        id = id,
                        name = "L$id",
                        x = 300f + id * 4f,
                        y = 400f + id * 3f,
                        confidence = 0.97f,
                    )
                },
                confidence = 0.97f,
            )

        val stable = engine.refine(base)
        val clean = engine.refine(
            base.copy(
                timestampNanos = 33_333_333L,
                landmarks = base.landmarks.map {
                    it.copy(
                        x = it.x + 2f,
                        y = it.y + 1f,
                    )
                },
            ),
        )
        val noisy = engine.refine(
            base.copy(
                timestampNanos = 66_666_666L,
                landmarks = base.landmarks.map {
                    it.copy(
                        x = it.x + 140f,
                        y = it.y - 95f,
                        confidence = 0.96f,
                    )
                },
            ),
        )

        assertTrue(stable.quality.meanMeasurementUncertaintyScore >= 0.0)
        assertTrue(clean.quality.meanMeasurementUncertaintyScore >= 0.0)
        assertTrue(
            noisy.quality.meanMeasurementUncertaintyScore >
                clean.quality.meanMeasurementUncertaintyScore,
        )

        engine.close()
    }

    @Test
    fun predictsBriefOcclusionWithoutDestroyingTrackState() {
        val engine = MotionCaptureEngine()
        var current = PoseDetectionEngine.Result(
            timestampNanos = 0L,
            imageWidth = 1000,
            imageHeight = 1000,
            landmarks = List(12) { id ->
                PoseDetectionEngine.Landmark(
                    id = id,
                    name = "L$id",
                    x = 200f + id * 5f,
                    y = 300f + id * 3f,
                    confidence = 0.95f,
                )
            },
            confidence = 0.95f,
        )

        engine.refine(current)
        repeat(4) { index ->
            current = current.copy(
                timestampNanos = (index + 1L) * 33_333_333L,
                landmarks = current.landmarks.map {
                    it.copy(
                        x = it.x + 4f,
                        y = it.y + 2f,
                    )
                },
            )
            engine.refine(current)
        }
        val occluded = current.copy(
            timestampNanos = current.timestampNanos + 33_333_333L,
            landmarks = current.landmarks.take(8),
        )
        val result = engine.refine(occluded)

        assertTrue(result.quality.predictedLandmarks >= 4)
        assertTrue(result.quality.temporalStability > 0.30)
        assertTrue(result.result.landmarks.size >= 12)

        engine.close()
    }
}
