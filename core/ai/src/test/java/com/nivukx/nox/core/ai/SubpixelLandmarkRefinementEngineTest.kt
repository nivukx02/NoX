package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.LandmarkSpace
import kotlin.test.Test
import kotlin.test.assertTrue

class SubpixelLandmarkRefinementEngineTest {
    private fun observation(
        id: Int,
        time: Long,
        x: Double,
        y: Double,
        confidence: Double = 0.96,
        row: Double? = null,
        skew: Long = 0L,
    ) = LandmarkObservation(
        landmarkId = id,
        x = x,
        y = y,
        confidence = confidence,
        timestampNanos = time,
        space = LandmarkSpace.NORMALIZED,
        captureTimingConfidence = 0.96,
        sensorRowNormalized = row,
        rollingShutterSkewNanos = skew,
    )

    @Test
    fun stableLandmarkRemainsStableWithLowUncertainty() {
        val engine = SubpixelLandmarkRefinementEngine()
        val p0 = engine.refine(listOf(observation(1, 1_000_000_000L, 0.5, 0.4)))
        val p1 = engine.refine(listOf(observation(1, 1_016_666_667L, 0.5002, 0.4001)))
        assertTrue(p0.acceptedCount == 1)
        assertTrue(p1.acceptedCount == 1)
        assertTrue(p1.observations.single().uncertaintyNormalized < 0.40)
        assertTrue(p1.observations.single().temporalStability > 0.40)
        engine.close()
    }

    @Test
    fun hardInnovationIsRejectedInsteadOfTeleportingTheTrack() {
        val engine = SubpixelLandmarkRefinementEngine(
            SubpixelLandmarkRefinementEngine.Config(
                hardInnovationNormalized = 0.10,
            ),
        )
        engine.refine(listOf(observation(2, 1_000_000_000L, 0.2, 0.2)))
        val result = engine.refine(listOf(observation(2, 1_033_333_333L, 0.8, 0.8)))
        assertTrue(result.observations.single().state == SubpixelLandmarkRefinementEngine.RefinementState.REJECTED)
        assertTrue(result.observations.single().x < 0.5)
        engine.close()
    }

    @Test
    fun refinedLandmarkExportsFiniteProbabilityCovariance() {
        val engine = SubpixelLandmarkRefinementEngine()
        engine.refine(
            listOf(
                observation(
                    31,
                    1_000_000_000L,
                    0.42,
                    0.38,
                ),
            ),
        )
        val result =
            engine.refine(
                listOf(
                    observation(
                        31,
                        1_016_666_667L,
                        0.4204,
                        0.3801,
                    ),
                ),
            )

        val point = result.observations.single()
        assertTrue(
            point.probabilitySigmaXNormalized.isFinite() &&
                point.probabilitySigmaXNormalized > 0.0,
        )
        assertTrue(
            point.probabilitySigmaYNormalized.isFinite() &&
                point.probabilitySigmaYNormalized > 0.0,
        )
        assertTrue(
            point.probabilityCalibrationScore in 0.0..1.0,
        )
        engine.close()
    }

    @Test
    fun rollingShutterMetadataProducesBoundedTimingCorrection() {
        val engine = SubpixelLandmarkRefinementEngine(
            SubpixelLandmarkRefinementEngine.Config(
                rollingShutterScale = 1.0,
            ),
        )
        engine.refine(listOf(observation(3, 1_000_000_000L, 0.4, 0.4, row = 0.1, skew = 8_000_000L)))
        val result = engine.refine(listOf(observation(3, 1_016_666_667L, 0.402, 0.402, row = 0.9, skew = 8_000_000L)))
        assertTrue(result.observations.single().rollingShutterCorrectionNormalized >= 0.0)
        assertTrue(result.maxCorrectionNormalized <= 0.028 + 1e-9)
        engine.close()
    }

    @Test
    fun shortObservationGapUsesBoundedPrediction() {
        val engine = SubpixelLandmarkRefinementEngine(
            SubpixelLandmarkRefinementEngine.Config(predictionHorizonMillis = 70.0),
        )
        engine.refine(listOf(observation(4, 1_000_000_000L, 0.30, 0.30)))
        engine.refine(listOf(observation(4, 1_016_666_667L, 0.305, 0.30)))
        val result = engine.refine(emptyList(), 1_050_000_000L)
        assertTrue(result.predictedCount == 1)
        assertTrue(result.observations.single().state == SubpixelLandmarkRefinementEngine.RefinementState.PREDICTED)
        engine.close()
    }

    @Test
    fun nonFiniteObservationFailsClosed() {
        val engine = SubpixelLandmarkRefinementEngine()
        val result = engine.refine(
            listOf(observation(5, 1_000_000_000L, Double.NaN, 0.4)),
            1_000_000_000L,
        )
        assertTrue(result.nonFiniteCount == 1)
        assertTrue(result.hardFailure)
        engine.close()
    }

    @Test
    fun stateMemoryIsBounded() {
        val engine = SubpixelLandmarkRefinementEngine(
            SubpixelLandmarkRefinementEngine.Config(maximumLandmarks = 32),
        )
        repeat(100) { frame ->
            val time = 1_000_000_000L + frame * 16_666_667L
            val landmarks = (0 until 64).map { id ->
                observation(id, time, id / 100.0, 0.5)
            }
            engine.refine(landmarks, time)
        }
        assertTrue(engine.telemetry().activeStates <= 32)
        engine.close()
    }
    @Test
    fun photometricAgreementStrengthensMatchingEvidence() {
        val engine =
            SubpixelLandmarkRefinementEngine()

        val t0 = 2_000_000_000L
        val t1 = 2_016_666_667L
        engine.refine(
            listOf(
                observation(
                    id = 40,
                    time = t0,
                    x = 0.50,
                    y = 0.40,
                ),
            ),
        )

        val result =
            engine.refine(
                observations =
                    listOf(
                        observation(
                            id = 40,
                            time = t1,
                            x = 0.51,
                            y = 0.40,
                        ),
                    ),
                timestampNanos = t1,
                photometricHints =
                    listOf(
                        SubpixelLandmarkRefinementEngine.PhotometricHint(
                            normalizedX = 0.51,
                            normalizedY = 0.40,
                            normalizedDx = 0.010,
                            normalizedDy = 0.0,
                            confidence = 0.98,
                            gradientAgreement = 0.95,
                            textureStrength = 0.92,
                            forwardBackwardConsistency = 0.96,
                            subpixelRefined = true,
                        ),
                    ),
            )

        assertTrue(
            result.observations.single()
                .diagnostics
                .any { it.contains("photometric") },
        )
        assertTrue(
            result.observations.single()
                .confidence > 0.70,
        )
        engine.close()
    }

    @Test
    fun photometricDisagreementRaisesUncertaintyInsteadOfTeleporting() {
        val good =
            SubpixelLandmarkRefinementEngine()
        val bad =
            SubpixelLandmarkRefinementEngine()

        val t0 = 3_000_000_000L
        val t1 = 3_016_666_667L

        listOf(good, bad).forEach { engine ->
            engine.refine(
                listOf(
                    observation(
                        id = 41,
                        time = t0,
                        x = 0.50,
                        y = 0.40,
                    ),
                ),
            )
        }

        val goodResult =
            good.refine(
                listOf(
                    observation(
                        id = 41,
                        time = t1,
                        x = 0.51,
                        y = 0.40,
                    ),
                ),
                t1,
                listOf(
                    SubpixelLandmarkRefinementEngine.PhotometricHint(
                        normalizedX = 0.51,
                        normalizedY = 0.40,
                        normalizedDx = 0.010,
                        normalizedDy = 0.0,
                        confidence = 0.98,
                        gradientAgreement = 0.96,
                        textureStrength = 0.95,
                        forwardBackwardConsistency = 0.97,
                        subpixelRefined = true,
                    ),
                ),
            )

        val badResult =
            bad.refine(
                listOf(
                    observation(
                        id = 41,
                        time = t1,
                        x = 0.51,
                        y = 0.40,
                    ),
                ),
                t1,
                listOf(
                    SubpixelLandmarkRefinementEngine.PhotometricHint(
                        normalizedX = 0.51,
                        normalizedY = 0.40,
                        normalizedDx = -0.08,
                        normalizedDy = 0.02,
                        confidence = 0.98,
                        gradientAgreement = 0.96,
                        textureStrength = 0.95,
                        forwardBackwardConsistency = 0.97,
                        subpixelRefined = true,
                    ),
                ),
            )

        assertTrue(
            badResult.observations.single()
                .uncertaintyNormalized >=
                goodResult.observations.single()
                    .uncertaintyNormalized,
        )
        assertTrue(
            badResult.observations.single()
                .x < 0.51,
        )

        good.close()
        bad.close()
    }

}