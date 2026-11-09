package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class KinematicTrajectoryOptimizerTest {

    @Test
    fun microMotionPreservationKeepsSmallMeasuredMotionResponsive() {
        val optimizer = KinematicTrajectoryOptimizer()
        optimizer.configure(
            KinematicTrajectoryOptimizer.Tuning(
                microMotionPreservation = 1.0,
                temporalSmoothing = 0.20,
                semanticMotionSensitivity = 0.90,
            ),
        )

        optimizer.optimize(frame(0L, 0.0))
        val result = optimizer.optimize(
            frame(16_666_667L, 0.004),
        )

        val x = result.joints.single().position.x
        assertTrue(x > 0.001)
        assertTrue(x < 0.006)
    }

    @Test
    fun extremeInnovationDoesNotCreateUnboundedVelocity() {
        val optimizer = KinematicTrajectoryOptimizer()
        optimizer.configure(
            KinematicTrajectoryOptimizer.Tuning(
                microMotionPreservation = 0.92,
                temporalSmoothing = 0.52,
                maximumVelocityMetersPerSecond = 5.0,
                maximumAccelerationMetersPerSecond2 = 20.0,
                maximumJerkMetersPerSecond3 = 120.0,
            ),
        )

        optimizer.optimize(frame(0L, 0.0))
        val result = optimizer.optimize(
            frame(16_666_667L, 4.0),
        )

        val velocity = result.joints.single().velocity.norm()
        assertTrue(velocity <= 5.0 + 1e-6)
        assertTrue(result.globalConfidence < 1.0)
    }

    @Test
    fun resetRemovesTemporalHistory() {
        val optimizer = KinematicTrajectoryOptimizer()

        optimizer.optimize(frame(0L, 0.0))
        optimizer.optimize(frame(16_666_667L, 0.05))
        optimizer.reset()

        val result = optimizer.optimize(
            frame(33_333_334L, 0.05),
        )

        assertTrue(result.joints.single().velocity.norm() < 0.5)
    }

    private fun frame(
        timestampNanos: Long,
        x: Double,
    ) =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 23,
                    position = Vec3(x, 0.0, 1.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.95,
                    reprojectionErrorPx = 1.0,
                    observationCount = 2,
                    positionUncertaintyMeters = 0.01,
                ),
            ),
            globalConfidence = 0.95,
        )
}
