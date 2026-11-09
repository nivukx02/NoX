package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MultiHypothesisOcclusionRecoveryEngineTest {

    private val engine =
        MultiHypothesisOcclusionRecoveryEngine(
            MultiHypothesisOcclusionRecoveryEngine.Config(
                maximumHypotheses = 4,
                maximumGapSeconds = 0.50,
                minimumConfidenceToRecover = 0.10,
            ),
        )

    private val track =
        MultiHypothesisOcclusionRecoveryEngine.TrackState(
            actorId = 17L,
            timestampNanos = 1_000_000_000L,
            positionMeters = Vec3(0.0, 1.0, 2.0),
            velocityMetersPerSecond = Vec3(1.0, 0.0, 0.0),
            accelerationMetersPerSecond2 = Vec3(0.2, 0.0, 0.0),
            confidence = 0.92,
            uncertaintyMeters = 0.004,
            contactProbability = 0.85,
        )

    @Test
    fun producesBoundedMotionHypotheses() {
        val result =
            engine.predict(
                track = track,
                timestampNanos = 1_100_000_000L,
            )

        assertEquals(4, result.hypotheses.size)
        assertTrue(
            result.hypotheses.all {
                it.likelihood in 0.0..1.0 &&
                    it.state.positionMeters.x.isFinite()
            },
        )
        assertTrue(result.recovered)
        assertTrue(result.state != null)
    }

    @Test
    fun observedPositionCollapsesHypothesisSet() {
        val observation =
            MultiHypothesisOcclusionRecoveryEngine.Observation(
                timestampNanos = 1_100_000_000L,
                positionMeters = Vec3(0.11, 1.0, 2.0),
                confidence = 0.96,
                uncertaintyMeters = 0.002,
            )

        val result =
            engine.predict(
                track = track,
                timestampNanos = observation.timestampNanos,
                observation = observation,
            )

        assertTrue(result.recovered)
        assertTrue(result.state != null)
        assertTrue(
            result.state!!
                .positionMeters
                .distanceTo(observation.positionMeters) <
                0.05,
        )
    }

    @Test
    fun marksLongGapAsNonAuthoritative() {
        val result =
            engine.predict(
                track = track,
                timestampNanos = 2_000_000_000L,
            )

        assertTrue(!result.recovered)
        assertTrue(
            result.warnings.any {
                it == "occlusion-gap-exceeds-recovery-window"
            },
        )
    }

    @Test
    fun contactBiasedModelRetainsVerticalContactPrior() {
        val contactTrack =
            track.copy(
                velocityMetersPerSecond =
                    Vec3(0.2, 3.0, 0.0),
                contactProbability = 1.0,
            )

        val result =
            engine.predict(
                track = contactTrack,
                timestampNanos = 1_100_000_000L,
            )

        val contact =
            result.hypotheses
                .first {
                    it.model ==
                        MultiHypothesisOcclusionRecoveryEngine.MotionModel.CONTACT_BIASED
                }

        assertTrue(
            kotlin.math.abs(
                contact.state.velocityMetersPerSecond.y,
            ) < 3.0,
        )
    }
}
