package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.MultiActorPoseAssociationEngine
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ActorWorkloadSchedulerTest {

    private fun actor(
        id: Long,
        confidence: Double = 0.88,
        identityRisk: Double = 0.20,
        occlusionRisk: Int = 0,
        motionSignal: Double = 0.55,
        continuity: Double = 0.90,
        predicted: Boolean = false,
    ) =
        MultiActorPoseAssociationEngine.Result(
            actorId = id,
            sourceObservationId = id.toInt(),
            detectorTrackingId = id.toInt(),
            subjectType = SubjectType.HUMAN,
            center = Vec2(
                (id % 10) / 10.0,
                (id / 10) / 10.0,
            ),
            bounds =
                MultiActorPoseAssociationEngine.Bounds(
                    left = 0.10,
                    top = 0.10,
                    right = 0.20,
                    bottom = 0.30,
                ),
            confidence = confidence,
            predicted = predicted,
            missedFrames = occlusionRisk,
            microMotion =
                MultiActorPoseAssociationEngine.MicroMotion(
                    timestampDeltaNanos = 33_000_000L,
                    maxDisplacementNormalized = 0.01,
                    rmsDisplacementNormalized = 0.005,
                    velocityNormalizedPerSecond = motionSignal,
                    accelerationNormalizedPerSecond2 = 0.5,
                    jerkNormalizedPerSecond3 = 0.2,
                    preservedSignal = motionSignal,
                    historyDepth = 8,
                ),
            velocityXNormalizedPerSecond = motionSignal,
            velocityYNormalizedPerSecond = 0.0,
            joints = emptyList(),
            predictionConfidence = if (predicted) 0.62 else 0.90,
            meanMotionInnovationNormalized = 0.02,
            acceleratedTrackCount = 0,
            globalAssignmentBounded = true,
            bodySpaceContinuity = continuity,
            identityAmbiguity = identityRisk,
            appearanceFingerprint = emptyList(),
            appearanceQuality = 0.0,
            worldJoints = emptyList(),
        )

    @Test
    fun fiftyActorSceneRemainsBoundedAndFair() {
        val scheduler = ActorWorkloadScheduler()

        val actors =
            (1L..50L).map { id ->
                actor(id = id)
            }

        var last =
            scheduler.schedule(
                timestampNanos = 1_000_000_000L,
                actors = actors,
                allocation = null,
                customization = MotionCaptureCustomization.production(),
                primaryActorId = 1L,
                maximumDetailedActorsPerFrame = 8,
            )

        repeat(12) { frame ->
            last =
                scheduler.schedule(
                    timestampNanos =
                        1_033_000_000L +
                            frame * 33_000_000L,
                    actors = actors,
                    allocation = null,
                    customization =
                        MotionCaptureCustomization.production(),
                    primaryActorId = 1L,
                    maximumDetailedActorsPerFrame = 8,
                )
        }

        assertEquals(50, last.admittedActorIds.size)
        assertEquals(8, last.selectedActorIds.size)
        assertTrue(last.fairnessScore > 0.55)
        assertTrue(last.maximumStarvationFrames < 8)
        assertTrue(
            last.warnings.contains(
                "CROWD_50_ACTOR_ENVELOPE_ACTIVE",
            ),
        )
    }

    @Test
    fun identityAndOcclusionRiskCanOutrankNominalConfidence() {
        val scheduler = ActorWorkloadScheduler()

        val baseline =
            listOf(
                actor(
                    id = 1L,
                    confidence = 0.96,
                    identityRisk = 0.02,
                ),
                actor(
                    id = 2L,
                    confidence = 0.72,
                    identityRisk = 0.92,
                    occlusionRisk = 6,
                ),
            )

        val result =
            scheduler.schedule(
                timestampNanos = 2_000_000_000L,
                actors = baseline,
                allocation = null,
                customization = MotionCaptureCustomization.production(),
                primaryActorId = null,
                maximumDetailedActorsPerFrame = 1,
            )

        assertEquals(setOf(2L), result.selectedActorIds)
        assertTrue(
            result.decisions.first { it.actorId == 2L }.reason
                .contains("IDENTITY_RISK"),
        )
    }
}
