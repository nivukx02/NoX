package com.nivukx.nox.core.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class MotionCaptureEngineTest {

    @Test
    fun followsSmoothMotionWithoutDestroyingConfidence() {
        val engine = MotionCaptureEngine()
        var refined: MotionCaptureEngine.Refined? = null

        repeat(12) { index ->
            val result = PoseDetectionEngine.Result(
                timestampNanos = index * 33_333_333L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = List(10) { id ->
                    PoseDetectionEngine.Landmark(
                        id = id,
                        name = id.toString(),
                        x = 200f + index * 5f + id,
                        y = 300f + index * 3f,
                        confidence = 0.94f,
                    )
                },
                confidence = 0.94f,
            )
            refined = engine.refine(result)
        }

        assertTrue(refined != null)
        assertTrue(refined!!.quality.temporalStability > 0.70)
        assertTrue(refined!!.result.landmarks.all { it.confidence > 0.70f })
        engine.close()
    }

    @Test
    fun detectorIdChangeCanReacquireARecentLandmark() {
        val engine = MotionCaptureEngine(
            reacquisitionDistanceNormalized = 0.14,
        )

        engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 1_000_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 11,
                        name = "nose",
                        x = 500f,
                        y = 500f,
                        confidence = 0.92f,
                    ),
                ),
                confidence = 0.92f,
            ),
        )

        val refined = engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 1_033_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 91,
                        name = "nose",
                        x = 507f,
                        y = 503f,
                        confidence = 0.92f,
                    ),
                ),
                confidence = 0.92f,
            ),
        )

        assertTrue(refined.landmarkQuality.any { it.id == 91 })
        assertTrue(refined.quality.trackedLandmarks >= 1)
        assertTrue(refined.quality.temporalStability > 0.0)
        engine.close()
    }

    @Test
    fun duplicateTimestampIsIdempotentAndDoesNotReinjectMotion() {
        val engine = MotionCaptureEngine()

        val firstInput =
            PoseDetectionEngine.Result(
                timestampNanos = 4_000_000_000L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 1,
                        name = "root",
                        x = 300f,
                        y = 300f,
                        confidence = 0.96f,
                    ),
                ),
                confidence = 0.96f,
            )

        val first = engine.refine(firstInput)

        val duplicate =
            engine.refine(
                firstInput.copy(
                    landmarks =
                        listOf(
                            PoseDetectionEngine.Landmark(
                                id = 1,
                                name = "root",
                                x = 950f,
                                y = 40f,
                                confidence = 0.20f,
                            ),
                        ),
                    confidence = 0.20f,
                ),
            )

        assertTrue(
            duplicate.result.landmarks.single().x ==
                first.result.landmarks.single().x,
        )
        assertTrue(
            duplicate.quality.meanSpeedNormalizedPerSecond ==
                first.quality.meanSpeedNormalizedPerSecond,
        )
        engine.close()
    }

    @Test
    fun timestampRegressionResetsTemporalState() {
        val engine = MotionCaptureEngine()

        val first = PoseDetectionEngine.Result(
            timestampNanos = 3_000_000_000L,
            imageWidth = 1000,
            imageHeight = 1000,
            landmarks = listOf(
                PoseDetectionEngine.Landmark(
                    id = 1,
                    name = "root",
                    x = 200f,
                    y = 200f,
                    confidence = 0.95f,
                ),
                PoseDetectionEngine.Landmark(
                    id = 2,
                    name = "head",
                    x = 500f,
                    y = 200f,
                    confidence = 0.95f,
                ),
            ),
            confidence = 0.95f,
        )
        engine.refine(first)

        val regressed = engine.refine(
            first.copy(
                timestampNanos = 2_000_000_000L,
                landmarks = first.landmarks.map {
                    it.copy(
                        x = it.x + 400f,
                        y = it.y + 200f,
                    )
                },
            ),
        )

        assertTrue(
            regressed.quality.meanSpeedNormalizedPerSecond <
                2.0,
        )
        assertTrue(
            regressed.quality.temporalStability >= 0.0 &&
                regressed.quality.temporalStability <= 1.0,
        )

        engine.close()
    }

    @Test
    fun exposesCommonModeAndDifferentialMotionDiagnostics() {
        val engine = MotionCaptureEngine()

        val base = listOf(
            PoseDetectionEngine.Landmark(
                id = 1,
                name = "root",
                x = 400f,
                y = 500f,
                confidence = 0.95f,
            ),
            PoseDetectionEngine.Landmark(
                id = 2,
                name = "head",
                x = 500f,
                y = 300f,
                confidence = 0.95f,
            ),
            PoseDetectionEngine.Landmark(
                id = 3,
                name = "hand",
                x = 700f,
                y = 520f,
                confidence = 0.95f,
            ),
            PoseDetectionEngine.Landmark(
                id = 4,
                name = "foot",
                x = 460f,
                y = 900f,
                confidence = 0.95f,
            ),
        )

        engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 2_000_000_000L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = base,
                confidence = 0.95f,
            ),
        )

        val moved = base.map { landmark ->
            landmark.copy(
                x = landmark.x + 18f +
                    if (landmark.id == 3) 26f else 0f,
                y = landmark.y + 8f,
            )
        }

        val result = engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 2_016_666_667L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = moved,
                confidence = 0.95f,
            ),
        )

        assertTrue(
            result.quality.commonModeMotionNormalizedPerSecond >= 0.0,
        )
        assertTrue(
            result.quality.differentialMotionScore >= 0.0 &&
                result.quality.differentialMotionScore <= 1.0,
        )
        assertTrue(
            result.quality.cameraCompensatedMotionScore >= 0.0 &&
                result.quality.cameraCompensatedMotionScore <= 1.0,
        )

        engine.close()
    }

    @Test
    fun cameraMotionContextReducesFalseSubjectMotionScore() {
        val engine = MotionCaptureEngine()
        val landmarks = listOf(
            PoseDetectionEngine.Landmark(
                id = 1,
                name = "root",
                x = 480f,
                y = 520f,
                confidence = 0.96f,
            ),
            PoseDetectionEngine.Landmark(
                id = 2,
                name = "head",
                x = 510f,
                y = 300f,
                confidence = 0.96f,
            ),
            PoseDetectionEngine.Landmark(
                id = 3,
                name = "hand",
                x = 720f,
                y = 470f,
                confidence = 0.96f,
            ),
        )

        engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 1_000_000_000L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = landmarks,
                confidence = 0.96f,
            ),
        )

        val moved = landmarks.map {
            it.copy(
                x = it.x + 22f,
                y = it.y + 12f,
            )
        }

        val neutral =
            engine.refine(
                PoseDetectionEngine.Result(
                    timestampNanos = 1_016_666_667L,
                    imageWidth = 1000,
                    imageHeight = 1000,
                    landmarks = moved,
                    confidence = 0.96f,
                ),
                MotionCaptureEngine.MotionContext(),
            )

        engine.reset()

        val cameraDominant =
            engine.refine(
                PoseDetectionEngine.Result(
                    timestampNanos = 1_000_000_000L,
                    imageWidth = 1000,
                    imageHeight = 1000,
                    landmarks = landmarks,
                    confidence = 0.96f,
                ),
                MotionCaptureEngine.MotionContext(
                    cameraMotionRisk = 0.92,
                    sensorConfidence = 0.95,
                    globalMotionConfidence = 0.92,
                ),
            )

        val cameraResult =
            engine.refine(
                PoseDetectionEngine.Result(
                    timestampNanos = 1_016_666_667L,
                    imageWidth = 1000,
                    imageHeight = 1000,
                    landmarks = moved,
                    confidence = 0.96f,
                ),
                MotionCaptureEngine.MotionContext(
                    cameraMotionRisk = 0.92,
                    sensorConfidence = 0.95,
                    globalMotionConfidence = 0.92,
                ),
            )

        assertTrue(neutral.quality.motionScore > cameraResult.quality.motionScore)
        assertTrue(
            cameraResult.quality.dynamicsConfidence <=
                neutral.quality.dynamicsConfidence + 1e-6,
        )
        assertTrue(cameraDominant.quality.temporalStability >= 0.0)

        engine.close()
    }

    @Test
    fun motionQualityExposesSpectralDiagnosticsAfterTemporalHistoryBuilds() {
        val engine = MotionCaptureEngine()

        repeat(42) { index ->
            val phase = index * 0.16
            val dx =
                (
                    500.0 +
                        32.0 *
                        kotlin.math.sin(
                            phase * 2.4,
                        )
                ).toFloat()
            val dy =
                (
                    420.0 +
                        18.0 *
                        kotlin.math.sin(
                            phase * 2.4 +
                                0.9,
                        )
                ).toFloat()

            engine.refine(
                PoseDetectionEngine.Result(
                    timestampNanos =
                        1_000_000_000L +
                            index * 33_333_333L,
                    imageWidth = 1000,
                    imageHeight = 1000,
                    landmarks = listOf(
                        PoseDetectionEngine.Landmark(
                            id = 1,
                            name = "root",
                            x = dx,
                            y = dy,
                            confidence = 0.96f,
                        ),
                        PoseDetectionEngine.Landmark(
                            id = 2,
                            name = "head",
                            x = dx + 80f,
                            y = dy - 140f,
                            confidence = 0.96f,
                        ),
                        PoseDetectionEngine.Landmark(
                            id = 3,
                            name = "hand",
                            x = dx + 180f,
                            y = dy + 30f,
                            confidence = 0.96f,
                        ),
                    ),
                    confidence = 0.96f,
                ),
            )
        }

        val result =
            engine.refine(
                PoseDetectionEngine.Result(
                    timestampNanos =
                        2_400_000_000L,
                    imageWidth = 1000,
                    imageHeight = 1000,
                    landmarks = listOf(
                        PoseDetectionEngine.Landmark(
                            id = 1,
                            name = "root",
                            x = 500f,
                            y = 420f,
                            confidence = 0.96f,
                        ),
                        PoseDetectionEngine.Landmark(
                            id = 2,
                            name = "head",
                            x = 580f,
                            y = 280f,
                            confidence = 0.96f,
                        ),
                        PoseDetectionEngine.Landmark(
                            id = 3,
                            name = "hand",
                            x = 680f,
                            y = 450f,
                            confidence = 0.96f,
                        ),
                    ),
                    confidence = 0.96f,
                ),
            )

        assertTrue(
            result.quality.spectralConfidence >= 0.0,
        )
        assertTrue(
            result.quality.spectralConfidence <= 1.0,
        )
        assertTrue(
            result.quality.periodicMotionScore >= 0.0,
        )
        assertTrue(
            result.quality.periodicMotionScore <= 1.0,
        )
        assertTrue(
            result.quality.vibrationRisk >= 0.0,
        )
        assertTrue(
            result.quality.vibrationRisk <= 1.0,
        )

        engine.close()
    }

    @Test
    fun largeLowConfidenceInnovationIsRejected() {
        val engine = MotionCaptureEngine()

        engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 2_000_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 1,
                        name = "nose",
                        x = 100f,
                        y = 100f,
                        confidence = 0.92f,
                    ),
                ),
                confidence = 0.92f,
            ),
        )

        val refined = engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 2_033_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                landmarks = listOf(
                    PoseDetectionEngine.Landmark(
                        id = 1,
                        name = "nose",
                        x = 1800f,
                        y = 1700f,
                        confidence = 0.30f,
                    ),
                ),
                confidence = 0.30f,
            ),
        )

        assertTrue(
            refined.landmarkQuality.any { it.rejectedMeasurement },
        )
        engine.close()
    }
    @Test
    fun landmarkStateCapacityIsHardBoundedAndObservable() {
        val engine = MotionCaptureEngine(maximumTrackedLandmarks = 64)
        val result = engine.refine(
            PoseDetectionEngine.Result(
                timestampNanos = 8_000_000_000L,
                imageWidth = 1000,
                imageHeight = 1000,
                landmarks = List(256) { id ->
                    PoseDetectionEngine.Landmark(
                        id = id,
                        name = "p$id",
                        x = (id % 16) * 50f,
                        y = (id / 16) * 60f,
                        confidence = 0.90f,
                    )
                },
                confidence = 0.90f,
            ),
        )
        assertTrue(result.quality.stateCapacityEvictions >= 192L)
        engine.close()
    }
}