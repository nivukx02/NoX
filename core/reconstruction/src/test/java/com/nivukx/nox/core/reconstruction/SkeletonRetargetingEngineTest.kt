package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class SkeletonRetargetingEngineTest {

    @Test
    fun highUncertaintyJointIsNotUsedForBoneRetargeting() {
        val engine = SkeletonRetargetingEngine()
        val result = engine.retarget(
            skeleton(
                uncertainty = 0.32,
            ),
        )

        assertTrue(result != null)
        assertTrue(
            result!!.bones.size <
                13,
        )
        assertTrue(result.confidence < 0.90)
    }

    private fun skeleton(
        uncertainty: Double,
    ) = SkeletonFrame(
        timestampNanos = 1_000_000_000L,
        subjectType = SubjectType.HUMAN,
        trackId = 1L,
        joints = listOf(
            joint(0, Vec3(0.0, 1.75, 0.0), 0.92, 0.02),
            joint(24, Vec3(0.0, 0.0, 0.0), 0.92, 0.02),
            joint(23, Vec3(-0.01, 0.0, 0.0), 0.92, 0.02),
            joint(11, Vec3(0.0, 1.2, 0.0), 0.92, uncertainty),
            joint(12, Vec3(0.0, 1.35, 0.0), 0.92, 0.02),
            joint(13, Vec3(-0.35, 1.2, 0.0), 0.92, 0.02),
            joint(14, Vec3(0.35, 1.2, 0.0), 0.92, 0.02),
            joint(15, Vec3(-0.55, 1.1, 0.0), 0.92, 0.02),
            joint(16, Vec3(0.55, 1.1, 0.0), 0.92, 0.02),
            joint(25, Vec3(-0.12, -0.45, 0.0), 0.92, 0.02),
            joint(26, Vec3(0.12, -0.45, 0.0), 0.92, 0.02),
            joint(27, Vec3(-0.12, -0.95, 0.0), 0.92, 0.02),
            joint(28, Vec3(0.12, -0.95, 0.0), 0.92, 0.02),
            joint(31, Vec3(-0.12, -1.0, 0.15), 0.92, 0.02),
            joint(32, Vec3(0.12, -1.0, 0.15), 0.92, 0.02),
        ),
        globalConfidence = 0.92,
    )

    private fun joint(
        id: Int,
        position: Vec3,
        confidence: Double,
        uncertainty: Double,
    ) = ReconstructionJoint(
        landmarkId = id,
        position = position,
        velocity = Vec3(0.0, 0.0, 0.0),
        confidence = confidence,
        reprojectionErrorPx = 1.0,
        observationCount = 3,
        positionUncertaintyMeters = uncertainty,
    )
}
