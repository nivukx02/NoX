package com.nivukx.nox.core.ai

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveViewRefinementControllerTest {
    private fun evidence(
        frame: Long,
        challenge: Double,
        ambiguity: Double,
        occlusion: Double,
        confidence: Double,
        stability: Double = 0.85,
    ) = AdaptiveViewRefinementController.FrameEvidence(
        frameIndex = frame,
        timestampNanos = frame * 16_666_667L,
        viewpointChallenge = challenge,
        leftRightAmbiguity = ambiguity,
        occlusionRisk = occlusion,
        confidence = confidence,
        temporalStability = stability,
        motionEnergy = 0.65,
    )

    @Test
    fun difficultViewRequestsSelectiveRefinement() {
        val controller = AdaptiveViewRefinementController()
        controller.evaluate(evidence(1, 0.95, 0.92, 0.60, 0.40))
        val decision = controller.evaluate(
            evidence(3, 0.92, 0.88, 0.72, 0.32),
        )
        assertTrue(decision.requestRefinement)
        assertTrue(decision.score >= 0.62)
    }

    @Test
    fun stableViewDoesNotWasteRefinementBudget() {
        val controller = AdaptiveViewRefinementController()
        val decision = controller.evaluate(
            evidence(1, 0.10, 0.05, 0.06, 0.94),
        )
        assertFalse(decision.requestRefinement)
        assertTrue(decision.temporalContinuity > 0.80)
    }

    @Test
    fun recoveryCanBeTriggeredByOcclusion() {
        val controller = AdaptiveViewRefinementController()
        controller.evaluate(evidence(1, 0.20, 0.10, 0.08, 0.92))
        val decision = controller.evaluate(
            evidence(3, 0.78, 0.46, 0.90, 0.36),
        )
        assertTrue(decision.recovery)
        assertTrue(decision.requestRefinement)
        assertTrue(decision.efficiencyRatio >= 0.82)
    }

    @Test
    fun lowExpectedValueDoesNotConsumeRefinementBudget() {
        val controller = AdaptiveViewRefinementController(
            AdaptiveViewRefinementController.Config(
                minimumExpectedGainToCostRatio = 1.50,
            ),
        )
        val decision = controller.evaluate(
            evidence(
                frame = 1,
                challenge = 0.67,
                ambiguity = 0.42,
                occlusion = 0.44,
                confidence = 0.62,
                stability = 0.78,
            ),
        )
        assertFalse(decision.requestRefinement)
        assertTrue(
            decision.reason ==
                "refinement_value_below_compute_cost" ||
                decision.forced,
        )
    }

}
