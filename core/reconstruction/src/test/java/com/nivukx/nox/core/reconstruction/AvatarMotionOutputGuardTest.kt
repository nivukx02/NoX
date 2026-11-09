package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.AvatarPose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.ReconstructionConfidenceBand
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AvatarMotionOutputGuardTest {

    @Test
    fun acceptedPosePasses() {
        val guard = AvatarMotionOutputGuard()
        val pose = pose(0.92)
        val decision = guard.evaluate(
            AvatarMotionOutputGuard.Input(
                skeleton = skeleton(0.92),
                avatarPose = pose,
                confidenceBand = ReconstructionConfidenceBand.EXCELLENT,
                reprojectionErrorPx = 1.2,
                meanUncertaintyMeters = 0.04,
            ),
        )

        assertTrue(decision.accepted)
        assertTrue(decision.pose != null)
    }

    @Test
    fun lowGlobalUncertaintyConfidenceRejectsLiveAvatarOutput() {
        val guard = AvatarMotionOutputGuard()
        val decision = guard.evaluate(
            AvatarMotionOutputGuard.Input(
                skeleton = skeleton(0.90),
                avatarPose = pose(0.90),
                confidenceBand = ReconstructionConfidenceBand.EXCELLENT,
                reprojectionErrorPx = 1.2,
                meanUncertaintyMeters = 0.04,
                uncertaintyConfidence = 0.32,
                p95UncertaintyMeters = 0.22,
                trustedJointRatio = 0.48,
                uncertaintyBottleneck = "OBSERVABILITY",
            ),
        )

        assertFalse(decision.accepted)
        assertTrue(
            decision.reasons.any {
                it.contains("uncertainty")
            },
        )
    }

    @Test
    fun badFrameIsHeldBrieflyThenRejected() {
        val guard = AvatarMotionOutputGuard(
            AvatarMotionOutputGuard.Config(
                holdFrames = 1,
            ),
        )

        val good = guard.evaluate(
            AvatarMotionOutputGuard.Input(
                skeleton = skeleton(0.90),
                avatarPose = pose(0.90),
                confidenceBand = ReconstructionConfidenceBand.GOOD,
                reprojectionErrorPx = 1.2,
                meanUncertaintyMeters = 0.04,
            ),
        )
        assertTrue(good.accepted)

        val held = guard.evaluate(
            AvatarMotionOutputGuard.Input(
                skeleton = skeleton(0.20),
                avatarPose = pose(0.20),
                confidenceBand = ReconstructionConfidenceBand.UNTRUSTED,
                reprojectionErrorPx = 12.0,
                meanUncertaintyMeters = 0.80,
            ),
        )
        assertTrue(held.accepted)
        assertTrue(held.usedHold)

        val rejected = guard.evaluate(
            AvatarMotionOutputGuard.Input(
                skeleton = skeleton(0.10),
                avatarPose = pose(0.10),
                confidenceBand = ReconstructionConfidenceBand.UNTRUSTED,
                reprojectionErrorPx = 20.0,
                meanUncertaintyMeters = 1.0,
            ),
        )
        assertFalse(rejected.accepted)
    }

    private fun skeleton(confidence: Double) = SkeletonFrame(
        timestampNanos = 1L,
        subjectType = SubjectType.HUMAN,
        trackId = 1L,
        joints = listOf(
            ReconstructionJoint(
                landmarkId = 23,
                position = Vec3(0.0, 1.0, 0.0),
                velocity = Vec3(0.0, 0.0, 0.0),
                confidence = confidence,
                reprojectionErrorPx = 1.0,
                observationCount = 2,
                positionUncertaintyMeters = 0.02,
            ),
        ),
        globalConfidence = confidence,
    )

    private fun pose(confidence: Double) =
        AvatarPose(
            timestampNanos = 1L,
            rootPositionMeters = Vec3(0.0, 1.0, 0.0),
            scale = 1.0,
            bones = mapOf(),
            confidence = confidence,
        )
}
