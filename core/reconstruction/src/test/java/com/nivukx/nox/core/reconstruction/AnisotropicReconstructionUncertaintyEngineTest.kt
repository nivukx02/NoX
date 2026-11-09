package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertTrue

class AnisotropicReconstructionUncertaintyEngineTest {

    @Test
    fun shallowParallaxProducesHighDepthAnisotropy() {
        val engine = AnisotropicReconstructionUncertaintyEngine()
        val result = engine.evaluate(
            AnisotropicReconstructionUncertaintyEngine.JointInput(
                jointId = 1,
                reprojectionRmsPx = 0.8,
                triangulationUncertaintyMeters = 0.002,
                parallaxDegrees = 1.0,
                angularDiversity = 0.14,
                inlierRatio = 0.92,
                cheiralityConfidence = 0.98,
                temporalUncertaintyMeters = 0.004,
                calibrationDriftRisk = 0.04,
                rollingShutterRisk = 0.08,
            ),
        )
        assertTrue(result.anisotropyRatio > 5.0)
        assertTrue(result.warnings.contains("low-parallax-depth-amplification"))
        engine.close()
    }

    @Test
    fun strongGeometryCanPassTightUncertaintyBudgets() {
        val engine = AnisotropicReconstructionUncertaintyEngine()
        val result = engine.evaluate(
            AnisotropicReconstructionUncertaintyEngine.JointInput(
                jointId = 3,
                reprojectionRmsPx = 0.25,
                triangulationUncertaintyMeters = 0.0008,
                parallaxDegrees = 24.0,
                angularDiversity = 0.92,
                inlierRatio = 0.98,
                cheiralityConfidence = 0.995,
                temporalUncertaintyMeters = 0.0005,
                calibrationDriftRisk = 0.01,
                rollingShutterRisk = 0.01,
                depthConfidence = 0.98,
                supportingViews = 6,
                sceneDepthMeters = 1.5,
            ),
            lateralBudgetMeters = 0.010,
            depthBudgetMeters = 0.025,
        )
        assertTrue(result.confidence > 0.65)
        assertTrue(result.gatePassed)
        engine.close()
    }
}
