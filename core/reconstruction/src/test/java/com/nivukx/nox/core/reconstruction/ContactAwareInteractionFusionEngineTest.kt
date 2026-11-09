package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContactAwareInteractionFusionEngineTest {
    private val engine = ContactAwareInteractionFusionEngine(
        ContactAwareInteractionFusionEngine.Config(
            contactEnterFrames = 2,
            contactStableFrames = 2,
            contactExitFrames = 2,
        ),
    )

    private fun input(
        time: Long,
        probability: Double,
        distance: Double,
        relativeSpeed: Double,
    ) = ContactAwareInteractionFusionEngine.ContactObservation(
        hand = ContactAwareInteractionFusionEngine.HandState(
            handId = "hand-left",
            timestampNanos = time,
            positionMeters = Vec3(0.0, 0.0, 0.0),
            velocityMetersPerSecond =
                Vec3(relativeSpeed, 0.0, 0.0),
            confidence = 0.98,
        ),
        trackedObject = ContactAwareInteractionFusionEngine.ObjectState(
            objectId = "prop-1",
            timestampNanos = time,
            positionMeters =
                Vec3(distance, 0.0, 0.0),
            velocityMetersPerSecond =
                Vec3(0.0, 0.0, 0.0),
            confidence = 0.97,
        ),
        contactProbability = probability,
        inertialConfidence = 0.96,
        temporalConsistency = 0.95,
        detectorAgreement = 0.94,
    )

    @Test
    fun persistentContactBecomesStable() {
        var result = engine.evaluate(
            input(1L, 0.90, 0.03, 0.05),
        )
        result = engine.evaluate(
            input(2L, 0.92, 0.032, 0.05),
        )
        result = engine.evaluate(
            input(3L, 0.93, 0.031, 0.05),
        )

        assertEquals(
            ContactAwareInteractionFusionEngine.ContactState.STABLE_CONTACT,
            result.state,
        )
        assertEquals(
            ContactAwareInteractionFusionEngine.ReasoningMode.FUSED_CONTACT,
            result.reasoningMode,
        )
        assertTrue(result.fusedConfidence > 0.70)
    }

    @Test
    fun contactCandidateDoesNotImmediatelyBecomeConstraint() {
        val result = engine.evaluate(
            input(10L, 0.82, 0.05, 0.12),
        )
        assertTrue(
            result.warnings.contains("CONTACT_CANDIDATE_REQUIRES_PERSISTENCE"),
        )
    }

    @Test
    fun separationExitsStableContact() {
        repeat(3) {
            engine.evaluate(
                input(20L + it, 0.94, 0.03, 0.04),
            )
        }

        var result = engine.evaluate(
            input(30L, 0.20, 0.18, 1.1),
        )
        result = engine.evaluate(
            input(31L, 0.20, 0.19, 1.2),
        )

        assertTrue(
            result.state ==
                ContactAwareInteractionFusionEngine.ContactState.SEPARATING ||
                result.state ==
                ContactAwareInteractionFusionEngine.ContactState.LOST,
        )
    }

    @Test
    fun lowInertialConfidenceFallsBackToKinematicMode() {
        val result = engine.evaluate(
            ContactAwareInteractionFusionEngine.ContactObservation(
                hand = ContactAwareInteractionFusionEngine.HandState(
                    handId = "hand-right",
                    timestampNanos = 100L,
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    velocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.96,
                ),
                trackedObject = ContactAwareInteractionFusionEngine.ObjectState(
                    objectId = "prop-2",
                    timestampNanos = 100L,
                    positionMeters = Vec3(0.03, 0.0, 0.0),
                    velocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                    confidence = 0.92,
                ),
                contactProbability = 0.90,
                inertialConfidence = 0.22,
                temporalConsistency = 0.90,
                detectorAgreement = 0.90,
            ),
        )

        assertEquals(
            ContactAwareInteractionFusionEngine.ReasoningMode.KINEMATIC,
            result.reasoningMode,
        )
    }
}
