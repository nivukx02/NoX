package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ReconstructionContinuityControllerTest {

    @Test
    fun cleanFrameCommitsAndBadFrameCanHold() {
        val controller = ReconstructionContinuityController()

        val first = frame(
            timestamp = 1_000_000_000L,
            x = 0.0,
            confidence = 0.92,
        )

        val firstDecision = controller.evaluate(
            candidate = first,
            qualityAccepted = true,
            qualityScore = 0.90,
        )

        assertEquals(
            ReconstructionContinuityController.Action.COMMIT,
            firstDecision.action,
        )
        assertTrue(firstDecision.accepted)

        val unstable = frame(
            timestamp = 1_033_000_000L,
            x = 0.20,
            confidence = 0.20,
        )

        val secondDecision = controller.evaluate(
            candidate = unstable,
            qualityAccepted = false,
            qualityScore = 0.42,
        )

        assertEquals(
            ReconstructionContinuityController.Action.HOLD_LAST,
            secondDecision.action,
        )
        assertTrue(!secondDecision.accepted)
        assertEquals(
            first.joints.first().position,
            secondDecision.displayFrame.joints.first().position,
        )
    }

    @Test
    fun extremeDiscontinuityIsRejectedWhenHoldBudgetIsConsumed() {
        val controller = ReconstructionContinuityController(
            holdMaxFrames = 1,
        )

        controller.evaluate(
            candidate = frame(1_000_000_000L, 0.0, 0.95),
            qualityAccepted = true,
            qualityScore = 0.95,
        )

        val jump = frame(1_050_000_000L, 4.0, 0.10)
        val decision = controller.evaluate(
            candidate = jump,
            qualityAccepted = false,
            qualityScore = 0.10,
        )

        assertEquals(
            ReconstructionContinuityController.Action.REJECT,
            decision.action,
        )
        assertTrue(
            decision.reasons.any {
                it == "hard-quality-threshold" ||
                    it == "joint-speed-discontinuity" ||
                    it == "mean-joint-discontinuity"
            },
        )
    }

    private fun frame(
        timestamp: Long,
        x: Double,
        confidence: Double,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestamp,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            joints = listOf(
                ReconstructionJoint(
                    landmarkId = 0,
                    position = Vec3(x, 0.0, 1.0),
                    velocity = Vec3(0.0, 0.0, 0.0),
                    confidence = confidence,
                    reprojectionErrorPx = 1.0,
                    observationCount = 2,
                    positionUncertaintyMeters = 0.02,
                ),
            ),
            globalConfidence = confidence,
        )
}
