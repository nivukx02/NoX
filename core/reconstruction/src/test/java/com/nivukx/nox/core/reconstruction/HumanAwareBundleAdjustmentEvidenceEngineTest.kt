package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class HumanAwareBundleAdjustmentEvidenceEngineTest {
    @Test
    fun motionConsistentEvidenceRetainsHighReliability() {
        val result = HumanAwareBundleAdjustmentEvidenceEngine().evaluate(
            observations = listOf(
                BundleAdjustmentEngine.Observation(
                    cameraId = "c0",
                    pointId = 1,
                    pixel = Vec2(100.0, 100.0),
                    confidence = 0.90,
                ),
            ),
            evidence = listOf(
                HumanAwareBundleAdjustmentEvidenceEngine.Evidence(
                    cameraId = "c0",
                    pointId = 1,
                    predictedMotionMeters = Vec3(0.10,0.0,0.0),
                    observedMotionMeters = Vec3(0.105,0.0,0.0),
                    motionUncertaintyMeters = 0.02,
                    confidence = 0.95,
                    dynamicProbability = 0.10,
                    cameraMotionCompensationConfidence = 0.95,
                    landmarkStability = 0.95,
                    temporalSpanFrames = 1,
                ),
            ),
        )
        assertTrue(result.weights.single().motionReliability > 0.70)
    }

    @Test
    fun longSpanAndDynamicEvidenceIsDownweighted() {
        val result = HumanAwareBundleAdjustmentEvidenceEngine().evaluate(
            observations = listOf(
                BundleAdjustmentEngine.Observation(
                    cameraId = "c0",
                    pointId = 1,
                    pixel = Vec2(100.0, 100.0),
                    confidence = 0.95,
                ),
            ),
            evidence = listOf(
                HumanAwareBundleAdjustmentEvidenceEngine.Evidence(
                    cameraId = "c0",
                    pointId = 1,
                    predictedMotionMeters = Vec3(0.0,0.0,0.0),
                    observedMotionMeters = Vec3(0.20,0.0,0.0),
                    motionUncertaintyMeters = 0.02,
                    confidence = 0.95,
                    dynamicProbability = 0.95,
                    cameraMotionCompensationConfidence = 0.20,
                    landmarkStability = 0.40,
                    temporalSpanFrames = 30,
                ),
            ),
        )
        assertTrue(result.weights.single().weight < 0.25)
    }
}
