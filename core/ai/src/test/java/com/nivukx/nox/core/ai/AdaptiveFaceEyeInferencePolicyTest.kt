package com.nivukx.nox.core.ai

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveFaceEyeInferencePolicyTest {

    @Test
    fun closeHighMotionFace_prioritizesFaceMeshAndEyes() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val decision = policy.decide(
            AdaptiveFaceEyeInferencePolicy.Evidence(
                faceAvailable = true,
                eyeAvailable = true,
                faceAreaRatio = 0.18,
                faceConfidence = 0.72,
                eyeConfidence = 0.48,
                fixationStability = 0.42,
                motionEnergy = 0.88,
                occlusionRisk = 0.32,
                faceMeshAgeNanos = 120_000_000L,
                eyeAgeNanos = 70_000_000L,
                thermalHeadroom = 0.90,
                gpuHeadroom = 0.88,
                faceDetail = 0.98,
                eyeDetail = 0.98,
            ),
        )

        assertTrue(decision.runFaceMesh)
        assertTrue(decision.runEyes)
        assertTrue(decision.faceMeshPriority > 0.65)
        assertTrue(decision.eyePriority > 0.65)
    }

    @Test
    fun stableSmallFace_deprioritizesExpensiveFaceMeshButKeepsEyesFresh() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val decision = policy.decide(
            AdaptiveFaceEyeInferencePolicy.Evidence(
                faceAvailable = true,
                eyeAvailable = true,
                faceAreaRatio = 0.012,
                faceConfidence = 0.94,
                eyeConfidence = 0.91,
                fixationStability = 0.95,
                motionEnergy = 0.06,
                occlusionRisk = 0.02,
                faceMeshAgeNanos = 50_000_000L,
                eyeAgeNanos = 42_000_000L,
                thermalHeadroom = 0.96,
                gpuHeadroom = 0.94,
                faceDetail = 0.60,
                eyeDetail = 0.90,
            ),
        )

        assertFalse(decision.runFaceMesh)
        assertFalse(decision.runEyes)
        assertTrue(decision.faceMeshPriority < 0.40)
        assertTrue(decision.eyePriority < 0.50)
    }

    @Test
    fun thermalPressure_reducesCadenceWithoutAllowingCriticalStaleness() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val decision = policy.decide(
            AdaptiveFaceEyeInferencePolicy.Evidence(
                faceAvailable = true,
                eyeAvailable = true,
                faceAreaRatio = 0.10,
                faceConfidence = 0.86,
                eyeConfidence = 0.58,
                fixationStability = 0.62,
                motionEnergy = 0.38,
                occlusionRisk = 0.18,
                faceMeshAgeNanos = 280_000_000L,
                eyeAgeNanos = 160_000_000L,
                thermalHeadroom = 0.28,
                gpuHeadroom = 0.42,
                faceDetail = 0.92,
                eyeDetail = 0.92,
            ),
        )

        assertTrue(decision.runEyes)
        assertTrue(decision.runFaceMesh)
        assertTrue(decision.computePressure > 0.55)
        assertTrue(decision.reason.contains("thermal"))
    }

    @Test
    fun lowLightingRaisesRecoveryPriority() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val bright =
            policy.decide(
                AdaptiveFaceEyeInferencePolicy.Evidence(
                    faceAvailable = true,
                    eyeAvailable = true,
                    faceAreaRatio = 0.10,
                    faceConfidence = 0.90,
                    eyeConfidence = 0.88,
                    fixationStability = 0.90,
                    motionEnergy = 0.18,
                    occlusionRisk = 0.08,
                    faceMeshAgeNanos = 80_000_000L,
                    eyeAgeNanos = 70_000_000L,
                    thermalHeadroom = 0.95,
                    gpuHeadroom = 0.94,
                    faceDetail = 0.82,
                    eyeDetail = 0.82,
                    lightingConfidence = 0.96,
                ),
            )

        val dark =
            policy.decide(
                AdaptiveFaceEyeInferencePolicy.Evidence(
                    faceAvailable = true,
                    eyeAvailable = true,
                    faceAreaRatio = 0.10,
                    faceConfidence = 0.90,
                    eyeConfidence = 0.88,
                    fixationStability = 0.90,
                    motionEnergy = 0.18,
                    occlusionRisk = 0.08,
                    faceMeshAgeNanos = 80_000_000L,
                    eyeAgeNanos = 70_000_000L,
                    thermalHeadroom = 0.95,
                    gpuHeadroom = 0.94,
                    faceDetail = 0.82,
                    eyeDetail = 0.82,
                    lightingConfidence = 0.28,
                ),
            )

        assertTrue(dark.faceMeshPriority > bright.faceMeshPriority)
        assertTrue(dark.eyePriority > bright.eyePriority)
        assertTrue(
            dark.reason.contains("lighting") ||
                dark.reason.contains("adaptive"),
        )
    }

    @Test
    fun highEyeUncertaintyForcesMoreFrequentEyeInference() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val lowUncertainty =
            policy.decide(
                AdaptiveFaceEyeInferencePolicy.Evidence(
                    faceAvailable = true,
                    eyeAvailable = true,
                    faceAreaRatio = 0.12,
                    faceConfidence = 0.90,
                    eyeConfidence = 0.90,
                    fixationStability = 0.92,
                    motionEnergy = 0.12,
                    occlusionRisk = 0.04,
                    faceMeshAgeNanos = 60_000_000L,
                    eyeAgeNanos = 60_000_000L,
                    thermalHeadroom = 0.96,
                    gpuHeadroom = 0.96,
                    faceDetail = 0.84,
                    eyeDetail = 0.90,
                    lightingConfidence = 0.96,
                    eyeUncertainty = 0.004,
                ),
            )

        val highUncertainty =
            policy.decide(
                AdaptiveFaceEyeInferencePolicy.Evidence(
                    faceAvailable = true,
                    eyeAvailable = true,
                    faceAreaRatio = 0.12,
                    faceConfidence = 0.90,
                    eyeConfidence = 0.90,
                    fixationStability = 0.92,
                    motionEnergy = 0.12,
                    occlusionRisk = 0.04,
                    faceMeshAgeNanos = 60_000_000L,
                    eyeAgeNanos = 60_000_000L,
                    thermalHeadroom = 0.96,
                    gpuHeadroom = 0.96,
                    faceDetail = 0.84,
                    eyeDetail = 0.90,
                    lightingConfidence = 0.96,
                    eyeUncertainty = 0.05,
                ),
            )

        assertTrue(
            highUncertainty.eyePriority >
                lowUncertainty.eyePriority,
        )
    }

    @Test
    fun unavailableEyeState_forcesAcquisitionWhenFaceExists() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val decision = policy.decide(
            AdaptiveFaceEyeInferencePolicy.Evidence(
                faceAvailable = true,
                eyeAvailable = false,
                faceAreaRatio = 0.08,
                faceConfidence = 0.84,
                eyeConfidence = 0.0,
                fixationStability = 0.0,
                motionEnergy = 0.22,
                occlusionRisk = 0.12,
                faceMeshAgeNanos = 90_000_000L,
                eyeAgeNanos = Long.MAX_VALUE,
                thermalHeadroom = 0.90,
                gpuHeadroom = 0.90,
                faceDetail = 0.82,
                eyeDetail = 0.94,
            ),
        )

        assertTrue(decision.runEyes)
        assertTrue(decision.reason.contains("eye-acquisition"))
    }

    @Test
    fun poorEyeSharpnessAndVisibility_raiseEyeRecoveryPriority() {
        val policy = AdaptiveFaceEyeInferencePolicy()

        val clean =
            policy.decide(
                AdaptiveFaceEyeInferencePolicy.Evidence(
                    faceAvailable = true,
                    eyeAvailable = true,
                    faceAreaRatio = 0.12,
                    faceConfidence = 0.92,
                    eyeConfidence = 0.90,
                    fixationStability = 0.90,
                    motionEnergy = 0.12,
                    occlusionRisk = 0.06,
                    faceMeshAgeNanos = 70_000_000L,
                    eyeAgeNanos = 70_000_000L,
                    thermalHeadroom = 0.95,
                    gpuHeadroom = 0.95,
                    faceDetail = 0.90,
                    eyeDetail = 0.90,
                    eyeSharpness = 0.95,
                    eyeVisibility = 0.96,
                    faceSharpness = 0.92,
                ),
            )

        val degraded =
            policy.decide(
                AdaptiveFaceEyeInferencePolicy.Evidence(
                    faceAvailable = true,
                    eyeAvailable = true,
                    faceAreaRatio = 0.12,
                    faceConfidence = 0.92,
                    eyeConfidence = 0.90,
                    fixationStability = 0.90,
                    motionEnergy = 0.12,
                    occlusionRisk = 0.06,
                    faceMeshAgeNanos = 70_000_000L,
                    eyeAgeNanos = 70_000_000L,
                    thermalHeadroom = 0.95,
                    gpuHeadroom = 0.95,
                    faceDetail = 0.90,
                    eyeDetail = 0.90,
                    eyeSharpness = 0.22,
                    eyeVisibility = 0.28,
                    faceSharpness = 0.34,
                ),
            )

        assertTrue(
            degraded.eyePriority > clean.eyePriority,
        )
        assertTrue(
            degraded.faceMeshPriority > clean.faceMeshPriority,
        )
    }

}
