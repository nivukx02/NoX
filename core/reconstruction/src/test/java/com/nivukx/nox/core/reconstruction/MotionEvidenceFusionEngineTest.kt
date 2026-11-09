package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionEvidenceFusionEngineTest {

    @Test
    fun strongSubjectMotionBecomesActive() {
        val engine = MotionEvidenceFusionEngine(
            MotionEvidenceFusionEngine.Config(
                enterActiveFrames = 2,
            ),
        )
        var result = evaluate(engine, 1_000_000_000L)
        result = evaluate(engine, 1_016_000_000L)

        assertEquals(MotionEvidenceFusionEngine.State.ACTIVE, result.state)
        assertTrue(result.score > 0.55)
        assertTrue(result.confidence > 0.55)
    }

    @Test
    fun cameraMotionDominantDoesNotBecomeSubjectMotion() {
        val engine = MotionEvidenceFusionEngine()
        val result = engine.evaluate(
            MotionEvidenceFusionEngine.Input(
                timestampNanos = 2_000_000_000L,
                modelMotionScore = 0.80,
                modelTemporalStability = 0.90,
                semanticMotionEnergy = 0.18,
                cameraMotionRisk = 0.92,
                occlusionRisk = 0.10,
                subjectLockConfidence = 0.88,
                poseConfidence = 0.92,
                reconstructionConfidence = 0.90,
                captureTimingConfidence = 0.96,
                synchronizationConfidence = 0.95,
                lightingConfidence = 0.90,
            ),
        )

        assertEquals(
            MotionEvidenceFusionEngine.State.CAMERA_DOMINANT,
            result.state,
        )
        assertTrue(result.cameraInterference > 0.45)
    }

    @Test
    fun staticSceneRemainsStatic() {
        val engine = MotionEvidenceFusionEngine()
        var result = evaluateStatic(engine, 3_000_000_000L)
        repeat(4) { index ->
            result = evaluateStatic(
                engine,
                3_016_000_000L + index * 16_000_000L,
            )
        }

        assertEquals(MotionEvidenceFusionEngine.State.STATIC, result.state)
        assertTrue(result.score < 0.30)
    }

    @Test
    fun poorTimingDegradesHighFrequencyTrust() {
        val engine = MotionEvidenceFusionEngine()
        val result = engine.evaluate(
            MotionEvidenceFusionEngine.Input(
                timestampNanos = 4_000_000_000L,
                modelMotionScore = 0.75,
                modelTemporalStability = 0.18,
                semanticMotionEnergy = 0.80,
                cameraMotionRisk = 0.10,
                occlusionRisk = 0.10,
                subjectLockConfidence = 0.90,
                poseConfidence = 0.90,
                reconstructionConfidence = 0.84,
                captureTimingConfidence = 0.32,
                synchronizationConfidence = 0.92,
                lightingConfidence = 0.90,
            ),
        )

        assertTrue(result.degraded)
        assertTrue(result.warnings.any {
            it.contains("capture-timing")
        })
    }

    @Test
    fun denseGeometryAndContinuousTimeEvidenceReduceTrustWhenDegraded() {
        val engine = MotionEvidenceFusionEngine()
        val healthy = engine.evaluate(
            MotionEvidenceFusionEngine.Input(
                timestampNanos = 5_000_000_000L,
                modelMotionScore = 0.82,
                modelTemporalStability = 0.88,
                semanticMotionEnergy = 0.84,
                cameraMotionRisk = 0.06,
                occlusionRisk = 0.06,
                subjectLockConfidence = 0.94,
                poseConfidence = 0.94,
                reconstructionConfidence = 0.92,
                captureTimingConfidence = 0.95,
                synchronizationConfidence = 0.95,
                lightingConfidence = 0.92,
                denseConsensus = 0.96,
                viewDiversity = 0.94,
                continuousTimeConfidence = 0.97,
                reconstructionUncertaintyMeters = 0.001,
            ),
        )

        engine.reset()

        val degradedGeometry = engine.evaluate(
            MotionEvidenceFusionEngine.Input(
                timestampNanos = 5_000_000_000L,
                modelMotionScore = 0.82,
                modelTemporalStability = 0.88,
                semanticMotionEnergy = 0.84,
                cameraMotionRisk = 0.06,
                occlusionRisk = 0.06,
                subjectLockConfidence = 0.94,
                poseConfidence = 0.94,
                reconstructionConfidence = 0.92,
                captureTimingConfidence = 0.95,
                synchronizationConfidence = 0.95,
                lightingConfidence = 0.92,
                denseConsensus = 0.32,
                viewDiversity = 0.28,
                continuousTimeConfidence = 0.35,
                reconstructionUncertaintyMeters = 0.120,
            ),
        )

        assertTrue(healthy.confidence > degradedGeometry.confidence)
        assertTrue(degradedGeometry.denseGeometryEvidence < 0.20)
        assertTrue(degradedGeometry.continuousTimeEvidence < 0.50)
        assertTrue(degradedGeometry.uncertaintyPenalty > 0.10)
        assertTrue(
            degradedGeometry.warnings.any {
                it.contains("dense-cross-view")
            },
        )
    }

    private fun evaluate(
        engine: MotionEvidenceFusionEngine,
        timestamp: Long,
    ): MotionEvidenceFusionEngine.Result =
        engine.evaluate(
            MotionEvidenceFusionEngine.Input(
                timestampNanos = timestamp,
                modelMotionScore = 0.84,
                modelTemporalStability = 0.82,
                semanticMotionEnergy = 0.86,
                cameraMotionRisk = 0.08,
                occlusionRisk = 0.08,
                subjectLockConfidence = 0.94,
                poseConfidence = 0.95,
                reconstructionConfidence = 0.90,
                captureTimingConfidence = 0.96,
                synchronizationConfidence = 0.96,
                lightingConfidence = 0.92,
            ),
        )

    private fun evaluateStatic(
        engine: MotionEvidenceFusionEngine,
        timestamp: Long,
    ): MotionEvidenceFusionEngine.Result =
        engine.evaluate(
            MotionEvidenceFusionEngine.Input(
                timestampNanos = timestamp,
                modelMotionScore = 0.04,
                modelTemporalStability = 0.94,
                semanticMotionEnergy = 0.03,
                cameraMotionRisk = 0.03,
                occlusionRisk = 0.03,
                subjectLockConfidence = 0.92,
                poseConfidence = 0.94,
                reconstructionConfidence = 0.90,
                captureTimingConfidence = 0.96,
                synchronizationConfidence = 0.96,
                lightingConfidence = 0.92,
            ),
        )
}
