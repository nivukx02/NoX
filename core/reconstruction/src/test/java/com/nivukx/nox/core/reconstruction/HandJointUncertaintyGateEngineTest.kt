package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertTrue
import org.junit.Test

class HandJointUncertaintyGateEngineTest {

    private val engine = HandJointUncertaintyGateEngine()

    private fun input(
        joint: Int = 8,
        x: Double = 0.5,
        y: Double = 0.5,
        confidence: Double = 0.95,
        temporal: Double = 0.95,
        predicted: Boolean = false,
        velocity: Double = 0.2,
    ) =
        HandJointUncertaintyGateEngine.Input(
            jointIndex = joint,
            normalizedX = x,
            normalizedY = y,
            landmarkConfidence = confidence,
            handConfidence = 0.96,
            handednessConfidence = 0.98,
            temporalStability = temporal,
            predicted = predicted,
            wristVelocityMetersPerSecond = velocity,
            observationFreshness = 1.0,
        )

    @Test
    fun strongVisibleJointReceivesHighExpertWeight() {
        val result = engine.evaluate(input())

        assertTrue(result.accepted)
        assertTrue(result.reliability > 0.75)
        assertTrue(result.viewExpertWeight > 0.55)
        assertTrue(result.uncertaintyMeters < 0.08)
    }

    @Test
    fun imageBoundaryEvidenceIsSuppressed() {
        val result = engine.evaluate(
            input(
                x = 0.004,
                y = 0.500,
            ),
        )

        assertTrue(result.borderConfidence < 0.1)
        assertTrue(result.hardRejected)
        assertTrue(
            "image-boundary-risk" in result.reasons,
        )
    }

    @Test
    fun predictedEvidenceIsDownweightedButCanRemainUseful() {
        val observed = engine.evaluate(input())
        val predicted = engine.evaluate(
            input(predicted = true),
        )

        assertTrue(
            predicted.viewExpertWeight <
                observed.viewExpertWeight,
        )
        assertTrue(
            predicted.uncertaintyMeters >=
                observed.uncertaintyMeters,
        )
        assertTrue(
            "prediction-only-evidence" in
                predicted.reasons,
        )
    }

    @Test
    fun unstableFastMotionRaisesUncertainty() {
        val stable = engine.evaluate(
            input(
                velocity = 0.5,
                temporal = 0.95,
            ),
        )
        val unstable = engine.evaluate(
            input(
                velocity = 11.5,
                temporal = 0.20,
            ),
        )

        assertTrue(
            unstable.uncertaintyMeters >
                stable.uncertaintyMeters,
        )
        assertTrue(
            unstable.viewExpertWeight <
                stable.viewExpertWeight,
        )
        assertTrue(
            "high-motion-risk" in unstable.reasons,
        )
    }

    @Test
    fun fingertipGetsSlightlyStricterEvidenceWeighting() {
        val palm = engine.evaluate(
            input(joint = 6),
        )
        val fingertip = engine.evaluate(
            input(joint = 8),
        )

        assertTrue(
            fingertip.viewExpertWeight <=
                palm.viewExpertWeight,
        )
    }

    @Test
    fun outlierFloorRejectsVeryWeakEvidence() {
        val result = engine.evaluate(
            input(
                confidence = 0.01,
                temporal = 0.02,
                velocity = 12.0,
            ),
        )

        assertTrue(result.hardRejected)
        assertTrue(
            result.reasons.contains(
                "joint-expert-rejected",
            ),
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidJointIndexIsRejected() {
        engine.evaluate(
            input(joint = 21),
        )
    }
}
