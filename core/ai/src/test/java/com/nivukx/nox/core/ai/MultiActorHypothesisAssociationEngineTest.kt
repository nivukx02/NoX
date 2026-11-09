package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiActorHypothesisAssociationEngineTest {
    @Test
    fun hardDepthGeometryContradictionCanRejectIdentity() {
        val engine = MultiActorHypothesisAssociationEngine()
        val result = engine.associate(
            listOf(
                MultiActorHypothesisAssociationEngine.ActorObservation(1L,1,100L,Vec3(10.0,0.0,0.0),mapOf(1 to 0.9),confidence=0.9),
            ),
            mapOf(1 to Vec3(0.0,0.0,0.0)),
        )
        assertTrue(result.accepted.isEmpty())
        assertTrue(result.rejectedDetectionIds.contains(1L) || result.ambiguousDetectionIds.contains(1L))
    }
}


class MultiActorHypothesisAssociationEngineCapacityTest {
    @Test
    fun observationBurstIsBoundedInsteadOfSilentlyTruncated() {
        val engine = MultiActorHypothesisAssociationEngine(
            MultiActorHypothesisAssociationEngine.Config(
                maximumActors = 4,
                maximumObservations = 4,
            ),
        )
        val observations = (0L until 12L).map { id ->
            MultiActorHypothesisAssociationEngine.ActorObservation(
                detectionId = id,
                actorHint = 0,
                timestampNanos = id + 1L,
                centerMeters = Vec3(0.0,0.0,1.0),
                confidence = 0.9,
            )
        }
        val result = engine.associate(observations, mapOf(0 to Vec3(0.0,0.0,1.0)))
        assertTrue(result.rejectedDetectionIds.isNotEmpty())
        assertTrue(
            result.accepted.size +
                result.rejectedDetectionIds.size +
                result.ambiguousDetectionIds.size ==
                12,
        )
    }
    @Test
    fun preservesHighQualityAlternateHypothesesForUnclaimedActors() {
        val engine = MultiActorHypothesisAssociationEngine()
        val result = engine.associate(
            observations = listOf(
                MultiActorHypothesisAssociationEngine.ActorObservation(
                    detectionId = 10L,
                    actorHint = null,
                    timestampNanos = 1_000L,
                    centerMeters = Vec3(0.0,0.0,1.0),
                    appearanceScoreByActor = mapOf(1 to 0.99, 2 to 0.92),
                    confidence = 0.99,
                ),
            ),
            predictions = mapOf(
                1 to Vec3(0.0,0.0,1.0),
                2 to Vec3(0.02,0.0,1.0),
            ),
        )
        assertTrue(result.accepted.count { !it.alternate } == 1)
        assertTrue(result.accepted.count { it.alternate } == 1)
        assertTrue(result.accepted.map { it.actorId }.toSet() == setOf(1, 2))
        assertTrue(result.ambiguousDetectionIds.contains(10L))
    )
}

class MultiActorHypothesisAssociationEngineGlobalAssignmentTest {
    @Test
    fun globalAssignmentMaximizesJointIdentityScoreInsteadOfGreedyOrder() {
        val engine = MultiActorHypothesisAssociationEngine(
            MultiActorHypothesisAssociationEngine.Config(
                geometryWeight = 0.0,
                appearanceWeight = 1.0,
                confidenceWeight = 0.0,
                depthWeight = 0.0,
                minimumAcceptanceScore = 0.10,
                unmatchedCost = 0.95,
            ),
        )
        val result = engine.associate(
            observations = listOf(
                MultiActorHypothesisAssociationEngine.ActorObservation(
                    detectionId = 100L,
                    actorHint = null,
                    timestampNanos = 1_000L,
                    centerMeters = Vec3(0.0, 0.0, 1.0),
                    appearanceScoreByActor = mapOf(0 to 0.90, 1 to 0.85),
                ),
                MultiActorHypothesisAssociationEngine.ActorObservation(
                    detectionId = 101L,
                    actorHint = null,
                    timestampNanos = 1_000L,
                    centerMeters = Vec3(0.0, 0.0, 1.0),
                    appearanceScoreByActor = mapOf(0 to 0.88, 1 to 0.10),
                ),
            ),
            predictions = mapOf(
                0 to Vec3(0.0, 0.0, 1.0),
                1 to Vec3(0.0, 0.0, 1.0),
            ),
        )

        val primary = result.accepted
            .filter { !it.alternate }
            .associate { it.detectionId to it.actorId }

        assertTrue(primary[100L] == 1)
        assertTrue(primary[101L] == 0)
        assertTrue(result.globallyOptimized)
        assertTrue(result.primaryAssignmentCount == 2)
    }
}


class MultiActorHypothesisAssociationEngineVelocityTest {
    @Test
    fun ambiguousCrossingUsesVelocityContinuity() {
        val engine =
            MultiActorHypothesisAssociationEngine(
                MultiActorHypothesisAssociationEngine.Config(
                    geometryWeight = 0.30,
                    appearanceWeight = 0.10,
                    confidenceWeight = 0.10,
                    depthWeight = 0.10,
                    velocityWeight = 0.40,
                    minimumAcceptanceScore = 0.20,
                ),
            )

        val result = engine.associate(
            observations = listOf(
                MultiActorHypothesisAssociationEngine.ActorObservation(
                    detectionId = 200L,
                    actorHint = null,
                    timestampNanos = 10L,
                    centerMeters = Vec3(0.50, 0.0, 2.0),
                    confidence = 0.98,
                    velocityMetersPerSecond = Vec3(2.0, 0.0, 0.0),
                    appearanceScoreByActor = mapOf(0 to 0.52, 1 to 0.51),
                ),
                MultiActorHypothesisAssociationEngine.ActorObservation(
                    detectionId = 201L,
                    actorHint = null,
                    timestampNanos = 10L,
                    centerMeters = Vec3(0.50, 0.0, 2.0),
                    confidence = 0.98,
                    velocityMetersPerSecond = Vec3(-2.0, 0.0, 0.0),
                    appearanceScoreByActor = mapOf(0 to 0.51, 1 to 0.52),
                ),
            ),
            predictions = mapOf(
                0 to Vec3(0.50, 0.0, 2.0),
                1 to Vec3(0.50, 0.0, 2.0),
            ),
            predictedVelocityByActor = mapOf(
                0 to Vec3(2.0, 0.0, 0.0),
                1 to Vec3(-2.0, 0.0, 0.0),
            ),
        )

        val primary =
            result.accepted
                .filter { !it.alternate }
                .associate { it.detectionId to it.actorId }

        assertTrue(primary[200L] == 0)
        assertTrue(primary[201L] == 1)
        assertTrue(result.primaryAssignmentCount == 2)
    }
}
