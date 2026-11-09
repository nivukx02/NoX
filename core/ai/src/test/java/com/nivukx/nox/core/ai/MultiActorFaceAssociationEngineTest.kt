package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.FaceFrame
import com.nivukx.nox.core.model.Vec2
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiActorFaceAssociationEngineTest {
    private fun actor(
        id: Long,
        center: Vec2,
    ) =
        MultiActorFaceAssociationEngine.Actor(
            actorId = id,
            center = center,
            boundsLeft = center.x - 0.20,
            boundsTop = center.y - 0.30,
            boundsRight = center.x + 0.20,
            boundsBottom = center.y + 0.30,
            confidence = 0.98,
        )

    private fun face(
        trackingId: Int,
        center: Vec2,
        timestamp: Long,
    ) =
        FaceFrame(
            timestampNanos = timestamp,
            trackingId = trackingId,
            boundsLeft = center.x.toFloat() - 0.08f,
            boundsTop = center.y.toFloat() - 0.12f,
            boundsRight = center.x.toFloat() + 0.08f,
            boundsBottom = center.y.toFloat() + 0.12f,
            yawDegrees = 0f,
            rollDegrees = 0f,
            points = emptyList(),
        )

    @Test
    fun globalAssignmentProducesUniqueActorMappingForAmbiguousFaces() {
        val engine = MultiActorFaceAssociationEngine()

        val result =
            engine.associate(
                actors = listOf(
                    actor(10L, Vec2(0.42, 0.50)),
                    actor(11L, Vec2(0.58, 0.50)),
                ),
                faces = listOf(
                    face(100, Vec2(0.50, 0.50), 1_000L),
                    face(101, Vec2(0.51, 0.50), 1_000L),
                ),
                minimumScore = 0.20,
            )

        val accepted =
            result.assignments.filter { it.accepted }

        assertEquals(2, accepted.size)
        assertEquals(
            2,
            accepted.mapNotNull { it.actorId }.distinct().size,
        )
    }

    @Test
    fun trackingContinuitySurvivesCloseCrossing() {
        val engine =
            MultiActorFaceAssociationEngine(
                MultiActorFaceAssociationEngine.Config(
                    trackingContinuityWeight = 0.22,
                    maximumTrackingContinuityAgeNanos = 500_000_000L,
                ),
            )

        val actors = listOf(
            actor(21L, Vec2(0.44, 0.50)),
            actor(22L, Vec2(0.56, 0.50)),
        )

        val first =
            engine.associate(
                actors = actors,
                faces = listOf(
                    face(201, Vec2(0.44, 0.50), 1_000_000_000L),
                    face(202, Vec2(0.56, 0.50), 1_000_000_000L),
                ),
                minimumScore = 0.20,
            )

        val firstMapping =
            first.assignments
                .filter { it.accepted }
                .associateBy { it.faceTrackingId }

        assertEquals(21L, firstMapping[201]?.actorId)
        assertEquals(22L, firstMapping[202]?.actorId)

        val second =
            engine.associate(
                actors = actors,
                faces = listOf(
                    face(201, Vec2(0.55, 0.50), 1_050_000_000L),
                    face(202, Vec2(0.45, 0.50), 1_050_000_000L),
                ),
                minimumScore = 0.20,
            )

        val secondMapping =
            second.assignments
                .filter { it.accepted }
                .associateBy { it.faceTrackingId }

        assertEquals(21L, secondMapping[201]?.actorId)
        assertEquals(22L, secondMapping[202]?.actorId)
        assertTrue(
            second.assignments.any {
                it.reason.contains("TRACKING_CONTINUITY")
            },
        )
    }
}
