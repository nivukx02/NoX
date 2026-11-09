package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class JointRotationSolverTest {

    @Test
    fun solverProducesNormalizedBoneRotations() {
        val skeleton = SkeletonFrame(
            timestampNanos = 1L,
            subjectType = SubjectType.HUMAN,
            trackId = 1L,
            joints = listOf(
                joint(11, Vec3(-0.2, 0.3, 2.0)),
                joint(13, Vec3(-0.4, 0.15, 2.0)),
                joint(15, Vec3(-0.55, -0.05, 2.0)),
                joint(23, Vec3(-0.12, -0.30, 2.0)),
                joint(25, Vec3(-0.12, -0.75, 2.0)),
                joint(27, Vec3(-0.12, -1.20, 2.0)),
                joint(12, Vec3(0.2, 0.3, 2.0)),
                joint(14, Vec3(0.4, 0.15, 2.0)),
                joint(16, Vec3(0.55, -0.05, 2.0)),
                joint(24, Vec3(0.12, -0.30, 2.0)),
                joint(26, Vec3(0.12, -0.75, 2.0)),
                joint(28, Vec3(0.12, -1.20, 2.0)),
                joint(0, Vec3(0.0, 0.55, 2.0)),
            ),
            globalConfidence = 0.92,
        )

        val result = JointRotationSolver().solve(skeleton)
        assertTrue(result.pose.bones.isNotEmpty())

        result.pose.bones.values.forEach { bone ->
            val q = bone.rotation
            val norm = kotlin.math.sqrt(
                q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z,
            )
            assertTrue(kotlin.math.abs(norm - 1.0) < 1e-6)
        }
    }

    private fun joint(id: Int, position: Vec3) =
        ReconstructionJoint(
            landmarkId = id,
            position = position,
            velocity = Vec3(0.0, 0.0, 0.0),
            confidence = 0.95,
            reprojectionErrorPx = 1.0,
            observationCount = 1,
        )
}
