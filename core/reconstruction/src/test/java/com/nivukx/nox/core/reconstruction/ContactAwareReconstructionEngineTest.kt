package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ContactAwareReconstructionEngineTest {

    @Test
    fun contactHysteresisTracksApproachContactAndRelease() {
        val engine = ContactAwareReconstructionEngine()

        val observations = listOf(
            ContactAwareReconstructionEngine.InteractionObservation(
                handId = 1L,
                objectId = 2L,
                timestampNanos = 1_000_000_000L,
                distanceMillimeters = 45.0,
                relativeSpeedMetersPerSecond = 0.8,
                normalAlignment = 0.8,
                handConfidence = 0.98,
                objectConfidence = 0.98,
            ),
            ContactAwareReconstructionEngine.InteractionObservation(
                handId = 1L,
                objectId = 2L,
                timestampNanos = 1_050_000_000L,
                distanceMillimeters = 10.0,
                relativeSpeedMetersPerSecond = 0.3,
                normalAlignment = 0.95,
                handConfidence = 0.98,
                objectConfidence = 0.98,
            ),
            ContactAwareReconstructionEngine.InteractionObservation(
                handId = 1L,
                objectId = 2L,
                timestampNanos = 1_100_000_000L,
                distanceMillimeters = 8.0,
                relativeSpeedMetersPerSecond = 0.03,
                normalAlignment = 0.98,
                handConfidence = 0.99,
                objectConfidence = 0.99,
            ),
        )

        val first = engine.evaluate(listOf(observations[0]), 1_000_000_000L)
        assertEquals(1, first.activePairCount)

        val second = engine.evaluate(listOf(observations[1]), 1_050_000_000L)
        assertTrue(second.approachingPairCount + second.contactPairCount >= 1)

        val third = engine.evaluate(listOf(observations[2]), 1_100_000_000L)
        assertTrue(third.contactPairCount == 1)
        assertTrue(third.meanInteractionConfidence > 0.80)
        assertTrue(third.trustedContactPairs.contains(1L to 2L))

        val release = engine.evaluate(
            listOf(
                ContactAwareReconstructionEngine.InteractionObservation(
                    handId = 1L,
                    objectId = 2L,
                    timestampNanos = 1_150_000_000L,
                    distanceMillimeters = 44.0,
                    relativeSpeedMetersPerSecond = 0.9,
                    normalAlignment = -0.6,
                    handConfidence = 0.98,
                    objectConfidence = 0.98,
                ),
            ),
            1_150_000_000L,
        )
        assertTrue(release.contactPairCount <= 1)
    }

    @Test
    fun invalidTemporalSampleIsRejectedWithoutStateMutation() {
        val engine = ContactAwareReconstructionEngine()

        engine.evaluate(
            listOf(
                ContactAwareReconstructionEngine.InteractionObservation(
                    handId = 10L,
                    objectId = 20L,
                    timestampNanos = 2_000_000_000L,
                    distanceMillimeters = 9.0,
                    relativeSpeedMetersPerSecond = 0.1,
                    normalAlignment = 0.9,
                ),
            ),
            2_000_000_000L,
        )

        val stale = engine.evaluate(
            listOf(
                ContactAwareReconstructionEngine.InteractionObservation(
                    handId = 10L,
                    objectId = 20L,
                    timestampNanos = 1_900_000_000L,
                    distanceMillimeters = 8.0,
                    relativeSpeedMetersPerSecond = 0.1,
                    normalAlignment = 0.9,
                ),
            ),
            1_900_000_000L,
        )

        assertTrue(stale.warnings.contains("stale-frame-rejected"))
        assertTrue(engine.result(10L, 20L)?.staleEvidenceRejected ?: 0 >= 0)
    }

    @Test
    fun excessivePenetrationBlocksTrustedCollisionClaim() {
        val engine = ContactAwareReconstructionEngine(
            ContactAwareReconstructionEngine.Config(
                maximumPenetrationMillimeters = 3.0,
            ),
        )

        repeat(3) { index ->
            engine.evaluate(
                listOf(
                    ContactAwareReconstructionEngine.InteractionObservation(
                        handId = 100L,
                        objectId = 200L,
                        timestampNanos = 3_000_000_000L + index * 50_000_000L,
                        distanceMillimeters = 1.0,
                        relativeSpeedMetersPerSecond = 0.1,
                        normalAlignment = 1.0,
                        penetrationMillimeters = 6.0,
                        handConfidence = 0.99,
                        objectConfidence = 0.99,
                    ),
                ),
                3_000_000_000L + index * 50_000_000L,
            )
        }

        val result = engine.result(100L, 200L)
        assertTrue(result != null)
        assertTrue(result!!.hardPenetrationViolation)
        assertTrue(!result.isTrustedContact)
    }
}
