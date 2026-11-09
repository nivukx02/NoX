package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class GroundContactAndRootMotionEngineTest {

    @Test
    fun stableFeetGenerateSupportEvidence() {
        val engine = GroundContactAndRootMotionEngine()

        val result = engine.process(frame(1_000_000_000L))
        engine.process(frame(1_016_666_666L))

        assertTrue(result.contacts.size == 2)
        assertTrue(result.confidence >= 0.0)
        assertTrue(result.plane.normal.y > 0.9)
    }

    @Test
    fun highFootUncertaintyWeakensGroundContact() {
        val engine = GroundContactAndRootMotionEngine(
            GroundContactAndRootMotionEngine.Config(
                minimumContactUncertaintyScore = 0.80,
            ),
        )

        val uncertain = frame(
            timestamp = 1_000_000_000L,
            footUncertainty = 0.24,
        )

        val result = engine.process(uncertain)

        assertTrue(result.contacts.all {
            it.uncertaintyScore < 0.80
        })
        assertTrue(result.stableContactCount == 0)
    }

    @Test
    fun resetClearsLearnedGround() {
        val engine = GroundContactAndRootMotionEngine()
        engine.process(frame(1_000_000_000L))
        engine.reset()

        val result = engine.process(frame(2_000_000_000L))
        assertTrue(result.plane.confidence <= 0.5)
    }

    private fun frame(
        timestamp: Long,
        footUncertainty: Double = 0.02,
    ) =
        SkeletonFrame(
            timestampNanos = timestamp,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            joints = listOf(
                joint(23, Vec3(0.0, 1.0, 0.0), 0.92),
                joint(24, Vec3(0.18, 1.0, 0.0), 0.92),
                joint(
                    31,
                    Vec3(-0.12, 0.0, 0.15),
                    0.96,
                    footUncertainty,
                ),
                joint(
                    32,
                    Vec3(0.12, 0.0, 0.15),
                    0.96,
                    footUncertainty,
                ),
            ),
            globalConfidence = 0.90,
        )

    private fun joint(
        id: Int,
        position: Vec3,
        confidence: Double,
        uncertainty: Double = 0.02,
    ) = ReconstructionJoint(
        landmarkId = id,
        position = position,
        velocity = Vec3(0.0, 0.0, 0.0),
        confidence = confidence,
        reprojectionErrorPx = 1.0,
        observationCount = 2,
        positionUncertaintyMeters = uncertainty,
    )
}
