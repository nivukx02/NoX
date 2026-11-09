package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionMotionDynamicsEngineTest {

    private fun joint(
        id: Int,
        x: Double,
        y: Double,
        z: Double,
        vx: Double = 0.0,
        vy: Double = 0.0,
        vz: Double = 0.0,
        confidence: Double = 0.95,
        uncertainty: Double = 0.01,
    ) =
        ReconstructionJoint(
            landmarkId = id,
            position = Vec3(x, y, z),
            velocity = Vec3(vx, vy, vz),
            acceleration = Vec3(0.0, 0.0, 0.0),
            confidence = confidence,
            positionUncertaintyMeters = uncertainty,
        )

    private fun skeleton(
        timestamp: Long,
        rootSpeed: Double,
    ) =
        SkeletonFrame(
            timestampNanos = timestamp,
            joints = listOf(
                joint(23, 0.0, 1.0, 0.0, rootSpeed, 0.0, 0.0),
                joint(24, 0.0, 1.0, 0.0),
                joint(11, -0.2, 1.4, 0.0, rootSpeed * 0.4),
                joint(12, 0.2, 1.4, 0.0, rootSpeed * 0.4),
                joint(13, -0.25, 0.9, 0.0, rootSpeed * 0.5),
                joint(14, 0.25, 0.9, 0.0, rootSpeed * 0.5),
                joint(15, -0.28, 0.55, 0.0),
                joint(16, 0.28, 0.55, 0.0),
                joint(31, -0.15, 0.0, 0.0, 0.01),
                joint(32, 0.15, 0.0, 0.0, 0.01),
            ),
            globalConfidence = 0.95,
        )

    @Test
    fun idleFrameIsStable() {
        val engine = ProductionMotionDynamicsEngine()

        val result = engine.analyze(skeleton(1_000_000_000L, 0.0))

        assertTrue(result.globalScore > 0.40)
        assertTrue(result.lowerBodyContactStability > 0.50)
    }

    @Test
    fun suddenVelocityChangeRaisesHighFrequencyMotionScore() {
        val engine = ProductionMotionDynamicsEngine()

        engine.analyze(
            skeleton(
                timestamp = 2_000_000_000L,
                rootSpeed = 0.12,
            ),
        )

        val result =
            engine.analyze(
                skeleton(
                    timestamp = 2_016_666_666L,
                    rootSpeed = 2.40,
                ),
            )

        assertTrue(
            result.highFrequencyMotionScore > 0.05,
        )
        assertTrue(
            result.joints.any {
                it.highFrequencyMotionEnergy > 0.05
            },
        )
    }

    @Test
    fun irregularFrameSpacingDoesNotInventHighFrequencyMotion() {
        val engine = ProductionMotionDynamicsEngine()

        engine.analyze(
            skeleton(
                timestamp = 5_000_000_000L,
                rootSpeed = 0.80,
            ),
        )

        val result =
            engine.analyze(
                skeleton(
                    timestamp = 5_240_000_000L,
                    rootSpeed = 0.80,
                ),
            )

        assertTrue(
            result.highFrequencyMotionScore < 0.20,
        )
    }

    @Test
    fun normalizedDynamicsWeightsDoNotArtificiallySaturateScore() {
        val engine = ProductionMotionDynamicsEngine()

        val result =
            engine.analyze(
                skeleton(
                    timestamp = 3_000_000_000L,
                    rootSpeed = 0.72,
                ),
            )

        assertTrue(result.globalScore < 0.99)
        assertTrue(result.globalScore >= 0.0)
    }

    @Test
    fun runningFrameRaisesDynamicEnergy() {
        val engine = ProductionMotionDynamicsEngine()

        engine.analyze(skeleton(1_000_000_000L, 0.1))
        val result = engine.analyze(
            skeleton(1_016_666_666L, 2.6),
        )

        assertTrue(result.rootSpeedMetersPerSecond > 2.0)
        assertTrue(result.phase != ProductionMotionDynamicsEngine.MotionPhase.UNKNOWN)
        assertTrue(result.meanSpeedMetersPerSecond > 0.05)
    }
}
