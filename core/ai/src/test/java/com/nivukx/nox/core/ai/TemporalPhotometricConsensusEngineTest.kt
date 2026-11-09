package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertTrue

class TemporalPhotometricConsensusEngineTest {

    @Test
    fun agreesWithStableMultiFrameFlowAndSuppressesOneFrameSpike() {
        val engine =
            TemporalPhotometricConsensusEngine(
                TemporalPhotometricConsensusEngine.Config(
                    maximumHistoryFrames = 4,
                    minimumFramesForConsensus = 3,
                ),
            )

        val results =
            listOf(
                result(
                    timestamp = 1_000L,
                    dx = 2.0,
                    dy = 0.5,
                ),
                result(
                    timestamp = 2_000L,
                    dx = 2.2,
                    dy = 0.6,
                ),
                result(
                    timestamp = 3_000L,
                    dx = 11.0,
                    dy = -8.0,
                ),
                result(
                    timestamp = 4_000L,
                    dx = 2.1,
                    dy = 0.55,
                ),
            )

        var fused =
            TemporalPhotometricConsensusEngine.Result.empty(4_000L)

        results.forEach {
            fused = engine.fuse(it)
        }

        assertTrue(fused.valid)
        assertTrue(fused.vectors.isNotEmpty())
        val vector = fused.vectors.single()
        assertTrue(vector.dxPixels < 4.0)
        assertTrue(vector.dyPixels in 0.0..2.0)
        assertTrue(vector.confidence > 0.70)
        assertTrue(fused.temporalConsensusScore > 0.70)
    }

    @Test
    fun insufficientTemporalHistoryRemainsExplicitlyWarmup() {
        val engine =
            TemporalPhotometricConsensusEngine()

        val fused =
            engine.fuse(
                result(
                    timestamp = 5_000L,
                    dx = 1.0,
                    dy = 1.0,
                ),
            )

        assertTrue(!fused.valid)
        assertTrue(
            fused.warnings.contains(
                "PHOTOMETRIC_TEMPORAL_CONSENSUS_WARMUP",
            ),
        )
    }

    private fun result(
        timestamp: Long,
        dx: Double,
        dy: Double,
    ): SparsePhotometricMotionFieldEngine.Result =
        SparsePhotometricMotionFieldEngine.Result(
            timestampNanos = timestamp,
            valid = true,
            sampleCount = 1,
            meanConfidence = 0.96,
            globalDxPixels = dx,
            globalDyPixels = dy,
            globalMotionMagnitudeNormalizedPerFrame =
                0.01,
            cameraMotionRisk = 0.12,
            globalMotionConfidence = 0.94,
            differentialMotionScore = 0.20,
            textureCoverage = 0.92,
            vectors =
                listOf(
                    SparsePhotometricMotionFieldEngine.FlowVector(
                        normalizedX = 0.5,
                        normalizedY = 0.5,
                        dxPixels = dx,
                        dyPixels = dy,
                        confidence = 0.96,
                        residualEnergy = 0.04,
                        photometricError = 0.03,
                        gradientAgreement = 0.94,
                        textureStrength = 0.90,
                        forwardBackwardConsistency = 0.95,
                        subpixelRefined = true,
                    ),
                ),
            warnings = emptyList(),
            medianPhotometricError = 0.03,
            p95PhotometricError = 0.05,
            meanGradientAgreement = 0.94,
            meanForwardBackwardConsistency = 0.95,
            subpixelRefinedRatio = 1.0,
            exposureShiftNormalized = 0.0,
            timestampDeltaNanos = 1_000L,
        )
}
