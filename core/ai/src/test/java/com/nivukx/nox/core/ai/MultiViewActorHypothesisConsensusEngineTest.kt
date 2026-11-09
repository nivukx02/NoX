package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import kotlin.test.Test
import kotlin.test.assertTrue

class MultiViewActorHypothesisConsensusEngineTest {

    @Test
    fun consistentDepthStrengthensThreeViewConsensus() {
        val engine =
            MultiViewActorHypothesisConsensusEngine()

        val members =
            listOf(
                observation("a", Vec2(0.00, 0.00), 2.00),
                observation("b", Vec2(0.02, 0.01), 2.01),
                observation("c", Vec2(-0.01, 0.02), 1.99),
            )

        val result = engine.evaluate(members)

        assertTrue(result.acceptedTripletCount > 0)
        assertTrue(result.depthConsistency > 0.95)
        assertTrue(result.consensusScore > 0.70)
    }

    @Test
    fun incompatibleDepthProducesExplicitConsensusDegradation() {
        val engine =
            MultiViewActorHypothesisConsensusEngine(
                MultiViewActorHypothesisConsensusEngine.Config(
                    maximumDepthDisagreementMeters = 0.08,
                ),
            )

        val members =
            listOf(
                observation("a", Vec2(0.00, 0.00), 1.80),
                observation("b", Vec2(0.01, 0.01), 2.40),
                observation("c", Vec2(-0.01, 0.02), 1.79),
            )

        val result = engine.evaluate(members)

        assertTrue(result.depthConsistency < 0.50)
        assertTrue(
            result.warnings.contains(
                "higher-order-depth-consistency-low",
            ),
        )
        assertTrue(result.consensusScore < 0.85)
    }

    private fun observation(
        device: String,
        center: Vec2,
        depth: Double,
    ): MultiCameraSubjectAssociationEngine.Observation =
        MultiCameraSubjectAssociationEngine.Observation(
            deviceId = DeviceId(device),
            localTrackId = 7L,
            subjectType = SubjectType.HUMAN,
            bounds =
                MultiCameraSubjectAssociationEngine.Bounds(
                    left = center.x - 0.05,
                    top = center.y - 0.10,
                    right = center.x + 0.05,
                    bottom = center.y + 0.10,
                ),
            velocity = Vec2(0.01, 0.01),
            confidence = 0.96,
            timestampNanos = 1_000_000_000L,
            calibratedCenterMeters = center,
            calibratedDepthMeters = depth,
        )
}
