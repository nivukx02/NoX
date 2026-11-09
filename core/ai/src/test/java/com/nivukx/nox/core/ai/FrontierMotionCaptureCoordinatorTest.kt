package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierMotionCaptureCoordinatorTest {
    @Test fun composesWorldProposalMotionAndActorScheduling() {
        val coordinator=FrontierMotionCaptureCoordinator()
        val result=coordinator.process(
            FrontierMotionCaptureCoordinator.Input(
                timestampNanos=1_000L,
                rayObservations=listOf(
                    WorldSpaceMotionProposalEngine.RayObservation(1,2,"a",1_000L,Vec3(0.0,0.0,0.0),Vec3(1.0,0.0,1.0),0.95,0.95,0.01),
                    WorldSpaceMotionProposalEngine.RayObservation(1,2,"b",1_000L,Vec3(1.0,0.0,0.0),Vec3(-1.0,0.0,1.0),0.95,0.95,0.01),
                ),
                denseObservations=listOf(DenseWorldPointCorrespondenceEngine.Observation(1,2,1_000L,"a",Vec2(0.5,0.5),Vec3(0.5,0.0,0.5),null,0.9)),
                actorDemands=listOf(FrontierActorDetailScheduler.ActorDemand(1,0.3,0.9,0.4,0.2,0.8,0L)),
            ),
        )
        assertEquals(1,result.proposals.size)
        assertEquals(1,result.motion.activeTrackCount)
        assertEquals(1,result.denseTracks.tracks.size)
        assertTrue(result.actorAllocation.admittedActorIds.contains(1))
        assertTrue(result.higherOrderIdentity.isEmpty())
        coordinator.close()
    }
}

class FrontierMotionCaptureCoordinatorCrowdEvidenceTest {
    @Test
    fun processPublishesMeasuredCrowdInteractionsForCloseActors() {
        val coordinator = FrontierMotionCaptureCoordinator()
        val result = coordinator.process(
            FrontierMotionCaptureCoordinator.Input(
                timestampNanos = 1_000L,
                rayObservations = emptyList(),
                actorObservations = listOf(
                    MultiActorHypothesisAssociationEngine.ActorObservation(
                        detectionId = 1L,
                        actorHint = 0,
                        timestampNanos = 1_000L,
                        centerMeters = Vec3(0.10, 0.10, 2.0),
                        confidence = 0.99,
                        appearanceQuality = 0.90,
                        appearanceScoreByActor = mapOf(0 to 0.99),
                    ),
                    MultiActorHypothesisAssociationEngine.ActorObservation(
                        detectionId = 2L,
                        actorHint = 1,
                        timestampNanos = 1_000L,
                        centerMeters = Vec3(0.12, 0.10, 2.0),
                        confidence = 0.99,
                        appearanceQuality = 0.90,
                        appearanceScoreByActor = mapOf(1 to 0.99),
                    ),
                ),
                actorPredictions = mapOf(
                    0 to Vec3(0.10, 0.10, 2.0),
                    1 to Vec3(0.12, 0.10, 2.0),
                ),
            ),
        )

        assertNotNull(result.crowdEvidence)
        assertTrue(
            result.crowdEvidence!!.interactions.any {
                setOf(it.actorA, it.actorB) == setOf(0L, 1L)
            },
        )
        coordinator.close()
    }
}


class FrontierMotionCaptureCoordinatorOcclusionTest {
    @Test
    fun exposesSecondaryOcclusionHypothesesAfterGap() {
        val coordinator =
            FrontierMotionCaptureCoordinator(
                proposalEngine =
                    WorldSpaceMotionProposalEngine(),
                motionEngine =
                    HierarchicalMotionInferenceEngine(
                        HierarchicalMotionInferenceEngine.Config(
                            maximumPredictionGapNanos = 100_000_000L,
                        ),
                    ),
                correspondenceEngine =
                    DenseWorldPointCorrespondenceEngine(),
                identityEngine =
                    MultiActorHypothesisAssociationEngine(),
                scheduler =
                    FrontierActorDetailScheduler(),
                evidenceAuthority =
                    FrontierEvidenceAuthorityEngine(),
                crowdEvidence =
                    ProductionCrowdMotionEvidenceEngine(),
            )

        val first =
            coordinator.process(
                FrontierMotionCaptureCoordinator.Input(
                    timestampNanos = 1_000_000L,
                    rayObservations =
                        listOf(
                            WorldSpaceMotionProposalEngine.RayObservation(
                                2,
                                3,
                                "a",
                                1_000_000L,
                                Vec3(0.0, 0.0, 0.0),
                                Vec3(0.2, 0.0, 1.0),
                                0.98,
                                0.98,
                                0.001,
                            ),
                            WorldSpaceMotionProposalEngine.RayObservation(
                                2,
                                3,
                                "b",
                                1_000_000L,
                                Vec3(1.0, 0.0, 0.0),
                                Vec3(-0.2, 0.0, 1.0),
                                0.98,
                                0.98,
                                0.001,
                            ),
                        ),
                ),
            )

        assertTrue(first.proposals.isNotEmpty())

        val second =
            coordinator.process(
                FrontierMotionCaptureCoordinator.Input(
                    timestampNanos = 50_000_000L,
                    rayObservations = emptyList(),
                ),
            )

        assertTrue(
            second.motion.recoveredTrackCount >= 1,
        )
        assertTrue(
            second.occlusionRecovery.isNotEmpty(),
        )
        assertTrue(
            second.occlusionRecovery.all {
                it.hypotheses.size <= 4
            },
        )

        coordinator.close()
    }
}
