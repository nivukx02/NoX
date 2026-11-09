package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MultiActorPoseAssociationEngineTest {
    @Test
    fun severe_world_joint_disagreement_blocks_assignment_instead_of_swapping_identity() {
        val engine =
            MultiActorPoseAssociationEngine(
                MultiActorPoseAssociationEngine.Config(
                    minimumMatchScore = 0.01,
                    minimumWorldJointContinuityForHardGate = 0.30,
                ),
            )

        fun worldJoint(
            id: Int,
            x: Double,
        ) =
            MultiActorPoseAssociationEngine.WorldJoint(
                landmarkId = id,
                xMeters = x,
                yMeters = 0.0,
                zMeters = 1.0,
                confidence = 0.95,
            )

        val baselineWorld =
            (0 until 8).map { id ->
                worldJoint(id, id * 0.05)
            }

        fun obs(
            sourceId: Int,
            world: List<MultiActorPoseAssociationEngine.WorldJoint>,
        ) =
            MultiActorPoseAssociationEngine.Observation(
                sourceObservationId = sourceId,
                detectorTrackingId = 7,
                center =
                    MultiActorPoseAssociationEngine.Vec2(
                        0.50,
                        0.50,
                    ),
                bounds =
                    MultiActorPoseAssociationEngine.Bounds(
                        0.40,
                        0.40,
                        0.60,
                        0.60,
                    ),
                confidence = 0.95,
                joints = emptyList(),
                worldJoints = world,
            )

        val first =
            engine.update(
                timestampNanos = 1_000_000_000L,
                observations = listOf(
                    obs(
                        sourceId = 1,
                        world = baselineWorld,
                    ),
                ),
            )
        val actorId =
            first.single().actorId

        val contradictoryWorld =
            baselineWorld.map {
                it.copy(
                    xMeters =
                        it.xMeters +
                            if (it.landmarkId % 2 == 0) {
                                2.0
                            } else {
                                -2.0
                            },
                )
            }

        val second =
            engine.update(
                timestampNanos = 1_016_000_000L,
                observations = listOf(
                    obs(
                        sourceId = 2,
                        world = contradictoryWorld,
                    ),
                ),
            )

        assertTrue(second.size == 1)
        assertTrue(second.single().actorId == actorId)
        assertTrue(second.single().predicted)
        assertTrue(second.single().worldJoints == baselineWorld)
    }

    @Test
    fun stable_actor_ids_follow_crossing_subjects_using_global_assignment() {
        val engine = MultiActorPoseAssociationEngine(
            MultiActorPoseAssociationEngine.Config(
                maxActors = 4,
                minimumMatchScore = 0.18,
            ),
        )

        val first = engine.update(
            timestampNanos = 1_000_000_000L,
            observations = listOf(
                observation(0, 0.20, 0.50, 0.90, 10),
                observation(1, 0.80, 0.50, 0.90, 20),
            ),
        )
        val actorA = first.single { it.sourceObservationId == 0 }.actorId
        val actorB = first.single { it.sourceObservationId == 1 }.actorId

        val second = engine.update(
            timestampNanos = 1_016_666_667L,
            observations = listOf(
                observation(0, 0.62, 0.50, 0.90, 10),
                observation(1, 0.38, 0.50, 0.90, 20),
            ),
        )

        assertEquals(actorA, second.single { it.sourceObservationId == 0 }.actorId)
        assertEquals(actorB, second.single { it.sourceObservationId == 1 }.actorId)
    }

    @Test
    fun appearance_continuity_strengthens_crossing_identity_when_geometry_is_ambiguous() {
        val engine = MultiActorPoseAssociationEngine(
            MultiActorPoseAssociationEngine.Config(
                maxActors = 4,
                minimumMatchScore = 0.12,
            ),
        )

        val first = engine.update(
            timestampNanos = 5_000_000_000L,
            observations = listOf(
                observation(
                    0,
                    0.44,
                    0.50,
                    0.92,
                    10,
                ).copy(
                    appearanceSignature =
                        List(48) { index ->
                            if (index % 3 == 0) 0.90 else 0.15
                        },
                    appearanceQuality = 0.94,
                ),
                observation(
                    1,
                    0.56,
                    0.50,
                    0.92,
                    20,
                ).copy(
                    appearanceSignature =
                        List(48) { index ->
                            if (index % 3 == 1) 0.90 else 0.15
                        },
                    appearanceQuality = 0.94,
                ),
            ),
        )

        val actorA =
            first.single {
                it.sourceObservationId == 0
            }.actorId
        val actorB =
            first.single {
                it.sourceObservationId == 1
            }.actorId

        val second = engine.update(
            timestampNanos = 5_016_666_667L,
            observations = listOf(
                observation(
                    0,
                    0.53,
                    0.50,
                    0.92,
                    101,
                ).copy(
                    appearanceSignature =
                        List(48) { index ->
                            if (index % 3 == 0) 0.90 else 0.15
                        },
                    appearanceQuality = 0.94,
                ),
                observation(
                    1,
                    0.47,
                    0.50,
                    0.92,
                    202,
                ).copy(
                    appearanceSignature =
                        List(48) { index ->
                            if (index % 3 == 1) 0.90 else 0.15
                        },
                    appearanceQuality = 0.94,
                ),
            ),
        )

        assertEquals(
            actorA,
            second.single {
                it.sourceObservationId == 0
            }.actorId,
        )
        assertEquals(
            actorB,
            second.single {
                it.sourceObservationId == 1
            }.actorId,
        )
        assertTrue(
            second.all {
                it.appearanceQuality >= 0.20
            },
        )
    }

    @Test
    fun strongAppearanceContradictionBlocksIdentitySwap() {
        val engine =
            MultiActorPoseAssociationEngine(
                MultiActorPoseAssociationEngine.Config(
                    maxActors = 2,
                    minimumMatchScore = 0.05,
                    appearanceHardMismatchThreshold = 0.24,
                    appearanceHardGateQuality = 0.72,
                ),
            )

        val actor =
            engine.update(
                timestampNanos = 6_000_000_000L,
                observations =
                    listOf(
                        observation(
                            0,
                            0.50,
                            0.50,
                            0.95,
                            1,
                        ).copy(
                            appearanceSignature =
                                listOf(
                                    1.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                ),
                            appearanceQuality = 0.95,
                        ),
                    ),
            ).single()

        val contradictory =
            engine.update(
                timestampNanos = 6_016_666_667L,
                observations =
                    listOf(
                        observation(
                            0,
                            0.50,
                            0.50,
                            0.95,
                            101,
                        ).copy(
                            appearanceSignature =
                                listOf(
                                    0.0,
                                    1.0,
                                    1.0,
                                    0.0,
                                ),
                            appearanceQuality = 0.95,
                        ),
                    ),
            )

        assertTrue(
            contradictory.single().actorId != actor.actorId,
        )
    }

    @Test
    fun micro_motion_preserves_subpixel_displacement_with_nanosecond_timing() {
        val engine = MultiActorPoseAssociationEngine()
        val first = engine.update(
            timestampNanos = 2_000_000_000L,
            observations = listOf(
                observation(0, 0.500000, 0.500000, 0.98, 7),
            ),
        ).single()
        val second = engine.update(
            timestampNanos = 2_004_000_000L,
            observations = listOf(
                observation(0, 0.500125, 0.500050, 0.98, 7),
            ),
        ).single()

        assertEquals(first.actorId, second.actorId)
        assertTrue(second.microMotion.maxDisplacementNormalized > 0.00010)
        assertTrue(second.microMotion.timestampDeltaNanos == 4_000_000L)
        assertTrue(second.microMotion.velocityNormalizedPerSecond > 0.0)
    }

    @Test
    fun runtime_configuration_can_raise_actor_capacity_without_recreating_engine() {
        val engine = MultiActorPoseAssociationEngine(
            MultiActorPoseAssociationEngine.Config(maxActors = 2),
        )

        engine.configure(
            maxActors = 8,
            minimumMatchScore = 0.10,
            microMotionRetention = 0.98,
        )

        assertEquals(8, engine.configuration().maxActors)
        assertEquals(0.98, engine.configuration().microMotionRetention)
    }

    @Test
    fun severe_actor_overlap_tightens_identity_gate_without_dropping_existing_tracks() {
        val engine =
            MultiActorPoseAssociationEngine(
                MultiActorPoseAssociationEngine.Config(
                    maxActors = 2,
                    minimumMatchScore = 0.05,
                    crowdedSceneOverlapThreshold = 0.20,
                    crowdedSceneMatchScoreBoost = 0.30,
                    maximumCenterDistanceNormalized = 0.80,
                ),
            )

        val first =
            engine.update(
                timestampNanos = 7_000_000_000L,
                observations =
                    listOf(
                        observation(0, 0.35, 0.50, 0.95, -1),
                        observation(1, 0.65, 0.50, 0.95, -1),
                    ),
            )

        val actorIds =
            first.map { it.actorId }.sorted()

        val collided =
            engine.update(
                timestampNanos = 7_016_666_667L,
                observations =
                    listOf(
                        observation(0, 0.50, 0.50, 0.95, -1),
                        observation(1, 0.50, 0.50, 0.95, -1),
                    ),
            )

        assertEquals(2, collided.size)
        assertEquals(
            actorIds,
            collided.map { it.actorId }.sorted(),
        )
        assertTrue(
            collided.count { it.predicted } >= 1,
        )
    }

    @Test
    fun occlusion_does_not_create_a_new_actor_immediately() {
        val engine = MultiActorPoseAssociationEngine(
            MultiActorPoseAssociationEngine.Config(
                maxMissedFrames = 3,
            ),
        )
        val first = engine.update(
            timestampNanos = 3_000_000_000L,
            observations = listOf(
                observation(0, 0.40, 0.40, 0.92, 4),
            ),
        ).single()

        val predicted = engine.update(
            timestampNanos = 3_016_666_667L,
            observations = emptyList(),
        )

        assertEquals(first.actorId, predicted.single().actorId)
        assertTrue(predicted.single().predicted)
        assertTrue(predicted.single().confidence < first.confidence)
    }


    @Test
    fun stale_frame_is_rejected_without_advancing_identity_state() {
        val engine = MultiActorPoseAssociationEngine()
        val first = engine.update(
            timestampNanos = 4_000_000_000L,
            observations = listOf(
                observation(0, 0.30, 0.40, 0.95, 11),
            ),
        ).single()

        val before = engine.current().single()

        val stale = engine.update(
            timestampNanos = 3_900_000_000L,
            observations = listOf(
                observation(0, 0.80, 0.80, 0.99, 99),
            ),
        ).single()

        assertEquals(first.actorId, stale.actorId)
        assertEquals(before.actorId, stale.actorId)
        assertEquals(before.sourceObservationId, stale.sourceObservationId)
        assertEquals(
            before.center,
            stale.center,
        )
        assertEquals(
            before.detectorTrackingId,
            stale.detectorTrackingId,
        )
    }

    private fun observation(
        id: Int,
        x: Double,
        y: Double,
        confidence: Double,
        detectorTrackingId: Int,
    ) = MultiActorPoseAssociationEngine.Observation(
        sourceObservationId = id,
        detectorTrackingId = detectorTrackingId,
        subjectType = com.nivukx.nox.core.model.SubjectType.HUMAN,
        center = com.nivukx.nox.core.model.Vec2(x, y),
        bounds = MultiActorPoseAssociationEngine.Bounds(
            left = x - 0.10,
            top = y - 0.22,
            right = x + 0.10,
            bottom = y + 0.22,
        ),
        confidence = confidence,
        joints = listOf(
            MultiActorPoseAssociationEngine.Joint(
                landmarkId = 0,
                x = x,
                y = y,
                confidence = confidence,
            ),
            MultiActorPoseAssociationEngine.Joint(
                landmarkId = 11,
                x = x - 0.03,
                y = y + 0.08,
                confidence = confidence,
            ),
        ),
    )
}
