package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiCameraSubjectAssociationEngineTest {

    @Test
    fun calibratedCrossCameraObservationsBecomeOneRigSubject() {
        val engine = MultiCameraSubjectAssociationEngine()

        val result = engine.associate(
            observations = listOf(
                observation(
                    device = "A",
                    track = 7,
                    center = Vec2(0.10, 0.25),
                    confidence = 0.94,
                    world = Vec2(0.0, 0.0),
                    timestamp = 1_000_000_000L,
                ),
                observation(
                    device = "B",
                    track = 11,
                    center = Vec2(0.30, 0.27),
                    confidence = 0.91,
                    world = Vec2(0.08, 0.01),
                    timestamp = 1_000_200_000L,
                ),
            ),
            timestampNanos = 1_000_200_000L,
        )

        assertEquals(1, result.subjects.size)
        assertEquals(2, result.subjects.first().observations.size)
        assertTrue(result.subjects.first().fusedConfidence > 0.70)
    }

    @Test
    fun differentSemanticTypesDoNotAssociate() {
        val engine = MultiCameraSubjectAssociationEngine()

        val human = observation(
            device = "A",
            track = 1,
            center = Vec2(0.2, 0.2),
            confidence = 0.95,
            world = Vec2(0.0, 0.0),
            timestamp = 1L,
            type = SubjectType.HUMAN,
        )

        val animal = observation(
            device = "B",
            track = 2,
            center = Vec2(0.2, 0.2),
            confidence = 0.95,
            world = Vec2(0.01, 0.01),
            timestamp = 1L,
            type = SubjectType.ANIMAL,
        )

        val result = engine.associate(
            observations = listOf(human, animal),
            timestampNanos = 1L,
        )

        assertTrue(result.subjects.size >= 2)
        assertTrue(result.unassignedObservations.isEmpty())
    }


    @Test
    fun calibratedDepthSeparatesOverlappingImagePositionsAcrossTheRig() {
        val engine =
            MultiCameraSubjectAssociationEngine(
                MultiCameraSubjectAssociationEngine.Config(
                    minimumScore = 0.40,
                    maximumGeometryDistanceMeters = 2.5,
                    maximumSubjects = 4,
                ),
            )

        val result =
            engine.associate(
                observations =
                    listOf(
                        observationWithDepth(
                            device = "camera-a",
                            track = 1L,
                            world = Vec2(0.25, 0.20),
                            depth = 2.0,
                        ),
                        observationWithDepth(
                            device = "camera-b",
                            track = 11L,
                            world = Vec2(0.25, 0.20),
                            depth = 6.0,
                        ),
                    ),
                timestampNanos = 7_000_000_000L,
            )

        assertEquals(2, result.subjects.size)
        assertTrue(
            result.subjects.all {
                it.observations.size == 1
            },
        )
    }

    @Test
    fun stale_rig_timestamp_does_not_rewrite_cross_camera_identity_state() {
        val engine = MultiCameraSubjectAssociationEngine()

        val firstObservation = observation(
            device = "camera-a",
            track = 1L,
            center = Vec2(0.20, 0.25),
            confidence = 0.95,
            world = Vec2(0.0, 0.0),
            timestamp = 5_000_000_000L,
        )
        val first = engine.associate(
            observations = listOf(firstObservation),
            timestampNanos = 5_000_000_000L,
        )
        assertTrue(first.subjects.isNotEmpty())

        val stale = engine.associate(
            observations = listOf(
                observation(
                    device = "camera-a",
                    track = 99L,
                    center = Vec2(0.20, 0.25),
                    confidence = 0.99,
                    world = Vec2(0.0, 0.0),
                    timestamp = 4_000_000_000L,
                ),
            ),
            timestampNanos = 4_000_000_000L,
        )

        assertEquals(
            first.subjects.map { it.rigTrackId },
            stale.subjects.map { it.rigTrackId },
        )
        assertTrue(
            stale.diagnostics.contains(
                "stale-rig-association-frame-rejected",
            ),
        )
    }


    @Test
    fun graph_assignment_is_less_order_sensitive_when_three_views_interact() {
        val engine = MultiCameraSubjectAssociationEngine(
            MultiCameraSubjectAssociationEngine.Config(
                minimumScore = 0.40,
                maximumSubjects = 4,
            ),
        )

        val observations = listOf(
            observation("camera-a", 1L, 5_000_000_000L, 0.25),
            observation("camera-a", 2L, 5_000_000_000L, 0.75),
            observation("camera-b", 11L, 5_000_100_000L, 0.27),
            observation("camera-b", 12L, 5_000_100_000L, 0.77),
            observation("camera-c", 21L, 5_000_050_000L, 0.26),
            observation("camera-c", 22L, 5_000_050_000L, 0.76),
        ).shuffled(java.util.Random(7))

        val result = engine.associate(
            observations = observations,
            timestampNanos = 5_000_100_000L,
        )

        assertEquals(2, result.subjects.size)
        assertTrue(
            result.subjects.all {
                it.observations.map { observation ->
                    observation.localTrackId
                }.size >= 2
            },
        )
        assertTrue(result.associationCount >= 4)
    }

    private fun observation(
        device: String,
        track: Long,
        timestamp: Long,
        centerX: Double,
    ) =
        observation(
            device = device,
            track = track,
            center = Vec2(centerX, 0.25),
            confidence = 0.94,
            world = Vec2(centerX, 0.0),
            timestamp = timestamp,
        )

    private fun observation(
        device: String,
        track: Long,
        center: Vec2,
        confidence: Double,
        world: Vec2,
        timestamp: Long,
        type: SubjectType = SubjectType.HUMAN,
    ) =
        MultiCameraSubjectAssociationEngine.Observation(
            deviceId = DeviceId(device),
            localTrackId = track,
            subjectType = type,
            bounds = MultiCameraSubjectAssociationEngine.Bounds(
                left = center.x - 0.10,
                top = center.y - 0.15,
                right = center.x + 0.10,
                bottom = center.y + 0.15,
            ),
            velocity = Vec2(0.5, 0.0),
            confidence = confidence,
            timestampNanos = timestamp,
            calibratedCenterMeters = world,
            calibratedDepthMeters = 2.0,
        )
    @Test
    fun globalRigAssignmentPreservesIdentityAcrossCloseCrossingWithChangedLocalIds() {
        val engine =
            MultiCameraSubjectAssociationEngine(
                MultiCameraSubjectAssociationEngine.Config(
                    minimumScore = 0.40,
                    maximumSubjects = 4,
                ),
            )

        val first =
            engine.associate(
                observations =
                    listOf(
                        observationWithVelocity(
                            device = "camera-a",
                            track = 1L,
                            center = Vec2(0.40, 0.25),
                            world = Vec2(0.40, 0.0),
                            velocity = Vec2(0.55, 0.0),
                            timestamp = 6_000_000_000L,
                        ),
                        observationWithVelocity(
                            device = "camera-a",
                            track = 2L,
                            center = Vec2(0.60, 0.25),
                            world = Vec2(0.60, 0.0),
                            velocity = Vec2(-0.55, 0.0),
                            timestamp = 6_000_000_000L,
                        ),
                        observationWithVelocity(
                            device = "camera-b",
                            track = 11L,
                            center = Vec2(0.41, 0.25),
                            world = Vec2(0.41, 0.0),
                            velocity = Vec2(0.55, 0.0),
                            timestamp = 6_000_100_000L,
                        ),
                        observationWithVelocity(
                            device = "camera-b",
                            track = 12L,
                            center = Vec2(0.61, 0.25),
                            world = Vec2(0.61, 0.0),
                            velocity = Vec2(-0.55, 0.0),
                            timestamp = 6_000_100_000L,
                        ),
                    ),
                timestampNanos = 6_000_100_000L,
            )

        assertEquals(2, first.subjects.size)

        val leftMovingId =
            first.subjects
                .first {
                    it.observations.any {
                        observation ->
                        observation.localTrackId == 1L
                    }
                }
                .rigTrackId

        val rightMovingId =
            first.subjects
                .first {
                    it.observations.any {
                        observation ->
                        observation.localTrackId == 2L
                    }
                }
                .rigTrackId

        val second =
            engine.associate(
                observations =
                    listOf(
                        observationWithVelocity(
                            device = "camera-a",
                            track = 101L,
                            center = Vec2(0.46, 0.25),
                            world = Vec2(0.46, 0.0),
                            velocity = Vec2(0.55, 0.0),
                            timestamp = 6_016_000_000L,
                        ),
                        observationWithVelocity(
                            device = "camera-a",
                            track = 102L,
                            center = Vec2(0.54, 0.25),
                            world = Vec2(0.54, 0.0),
                            velocity = Vec2(-0.55, 0.0),
                            timestamp = 6_016_000_000L,
                        ),
                        observationWithVelocity(
                            device = "camera-b",
                            track = 201L,
                            center = Vec2(0.47, 0.25),
                            world = Vec2(0.47, 0.0),
                            velocity = Vec2(0.55, 0.0),
                            timestamp = 6_016_100_000L,
                        ),
                        observationWithVelocity(
                            device = "camera-b",
                            track = 202L,
                            center = Vec2(0.55, 0.25),
                            world = Vec2(0.55, 0.0),
                            velocity = Vec2(-0.55, 0.0),
                            timestamp = 6_016_100_000L,
                        ),
                    ),
                timestampNanos = 6_016_100_000L,
            )

        assertEquals(2, second.subjects.size)

        val secondForwardId =
            second.subjects
                .first {
                    it.observations.any {
                        observation ->
                        observation.velocity.x > 0.0
                    }
                }
                .rigTrackId

        val secondBackwardId =
            second.subjects
                .first {
                    it.observations.any {
                        observation ->
                        observation.velocity.x < 0.0
                    }
                }
                .rigTrackId

        assertEquals(leftMovingId, secondForwardId)
        assertEquals(rightMovingId, secondBackwardId)

        assertTrue(
            second.subjects.all {
                it.identityAssignmentAccepted
            },
        )
        assertTrue(
            second.diagnostics.any {
                it.startsWith("globalIdentityConfidence=")
            },
        )
    }

    private fun observationWithDepth(
        device: String,
        track: Long,
        world: Vec2,
        depth: Double,
    ) =
        MultiCameraSubjectAssociationEngine.Observation(
            deviceId = DeviceId(device),
            localTrackId = track,
            subjectType = SubjectType.HUMAN,
            bounds =
                MultiCameraSubjectAssociationEngine.Bounds(
                    left = 0.15,
                    top = 0.13,
                    right = 0.35,
                    bottom = 0.43,
                ),
            velocity = Vec2(0.5, 0.0),
            confidence = 0.95,
            timestampNanos = 7_000_000_000L,
            calibratedCenterMeters = world,
            calibratedDepthMeters = depth,
        )

    private fun observationWithVelocity(
        device: String,
        track: Long,
        center: Vec2,
        world: Vec2,
        velocity: Vec2,
        timestamp: Long,
    ) =
        MultiCameraSubjectAssociationEngine.Observation(
            deviceId = DeviceId(device),
            localTrackId = track,
            subjectType = SubjectType.HUMAN,
            bounds =
                MultiCameraSubjectAssociationEngine.Bounds(
                    left = center.x - 0.08,
                    top = center.y - 0.12,
                    right = center.x + 0.08,
                    bottom = center.y + 0.12,
                ),
            velocity = velocity,
            confidence = 0.95,
            timestampNanos = timestamp,
            calibratedCenterMeters = world,
            calibratedDepthMeters = 2.0,
        )

}
