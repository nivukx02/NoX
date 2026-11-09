package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertTrue

class DenseLandmarkEvidenceFusionEnginePhotometricTest {

    @Test
    fun lowPhotometricConsistencyReducesObservationInfluence() {
        val engine =
            DenseLandmarkEvidenceFusionEngine(
                DenseLandmarkEvidenceFusionEngine.Config(
                    appearanceConsistencyWeight = 0.04,
                ),
            )

        val strong =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    observations =
                        listOf(
                            DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                1,
                                0,
                            ) to
                                observation(
                                    cameraId = 1,
                                    x = 0.40,
                                    photometric = 1.0,
                                ),
                            DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                1,
                                0,
                            ) to
                                observation(
                                    cameraId = 2,
                                    x = 0.40,
                                    photometric = 1.0,
                                ),
                        ),
                ),
            )

        engine.reset()

        val weak =
            engine.process(
                DenseLandmarkEvidenceFusionEngine.FrameInput(
                    timestampNanos = 1_000_000_000L,
                    observations =
                        listOf(
                            DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                1,
                                0,
                            ) to
                                observation(
                                    cameraId = 1,
                                    x = 0.40,
                                    photometric = 0.20,
                                ),
                            DenseLandmarkEvidenceFusionEngine.LandmarkKey(
                                1,
                                0,
                            ) to
                                observation(
                                    cameraId = 2,
                                    x = 0.40,
                                    photometric = 1.0,
                                ),
                        ),
                ),
            )

        assertTrue(
            weak.meanConfidence <
                strong.meanConfidence,
        )
    }

    private fun observation(
        cameraId: Int,
        x: Double,
        photometric: Double,
    ): DenseLandmarkEvidenceFusionEngine.CameraObservation =
        DenseLandmarkEvidenceFusionEngine.CameraObservation(
            cameraId = cameraId,
            timestampNanos = 1_000_000_000L,
            normalizedX = x,
            normalizedY = 0.50,
            confidence = 0.95,
            uncertainty = 0.03,
            visibility = 0.98,
            cameraQuality = 0.95,
            reprojectionResidualNormalized = 0.01,
            depthConfidence = 0.95,
            appearanceConsistency = 0.95,
            viewSelectionScore = 0.95,
            photometricConsistency = photometric,
        )
}
