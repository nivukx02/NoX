package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TemporalReconstructionStabilizerTest {

    @Test
    fun stableGeometryIsAccepted() {
        val engine = TemporalReconstructionStabilizer()

        val frame = engine.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = 1_000_000_000L,
                camerasUsed = 3,
                clockSkewNanos = 2_000_000L,
                globalConfidence = 0.90,
                joints = (0 until 8).map { index ->
                    TemporalReconstructionStabilizer.JointObservation(
                        jointId = index,
                        positionMeters =
                            TemporalReconstructionStabilizer.Vec3(
                                x = index * 0.04,
                                y = index * 0.01,
                                z = 1.0 + index * 0.02,
                            ),
                        confidence = 0.92,
                        uncertaintyMeters = 0.04,
                        reprojectionErrorPx = 1.4,
                        viewCount = 3,
                    )
                },
            ),
        )

        assertTrue(frame.accepted)
        assertTrue(frame.score > 0.55)
        assertEquals(8, frame.diagnostics.jointsAccepted)
    }

    @Test
    fun implausibleJumpIsRejectedButPredictionRemainsFinite() {
        val engine = TemporalReconstructionStabilizer()

        engine.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = 0L,
                camerasUsed = 3,
                clockSkewNanos = 1_000_000L,
                globalConfidence = 0.90,
                joints = listOf(
                    TemporalReconstructionStabilizer.JointObservation(
                        jointId = 1,
                        positionMeters =
                            TemporalReconstructionStabilizer.Vec3(
                                0.0,
                                0.0,
                                1.0,
                            ),
                        confidence = 0.92,
                        uncertaintyMeters = 0.03,
                        reprojectionErrorPx = 1.0,
                        viewCount = 3,
                    ),
                ),
            ),
        )

        val frame = engine.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = 50_000_000L,
                camerasUsed = 3,
                clockSkewNanos = 1_000_000L,
                globalConfidence = 0.85,
                joints = listOf(
                    TemporalReconstructionStabilizer.JointObservation(
                        jointId = 1,
                        positionMeters =
                            TemporalReconstructionStabilizer.Vec3(
                                25.0,
                                0.0,
                                1.0,
                            ),
                        confidence = 0.99,
                        uncertaintyMeters = 0.02,
                        reprojectionErrorPx = 1.0,
                        viewCount = 3,
                    ),
                ),
            ),
        )

        assertFalse(frame.joints.isEmpty())
        assertTrue(
            frame.joints.single().positionMeters.finite(),
        )
        assertTrue(
            frame.diagnostics.jointsRejected >= 1,
        )
    }

    @Test
    fun persistentStateCannotExceedConfiguredTrackCap() {
        val engine =
            TemporalReconstructionStabilizer(
                TemporalReconstructionStabilizer.Config(
                    maxJointTracks = 8,
                ),
            )

        repeat(32) { index ->
            engine.update(
                TemporalReconstructionStabilizer.FrameInput(
                    timestampNanos = (index + 1L) * 10_000_000L,
                    camerasUsed = 2,
                    clockSkewNanos = 1_000_000L,
                    globalConfidence = 0.8,
                    joints = listOf(
                        observation(
                            id = index,
                            x = index * 0.001,
                            y = 0.0,
                            z = 1.0,
                        ),
                    ),
                ),
            )
        }

        assertTrue(engine.snapshot(1_000_000_000L).size <= 8)
    }

    @Test
    fun short_occlusion_prediction_remains_finite_with_higher_order_state() {
        val engine =
            TemporalReconstructionStabilizer(
                TemporalReconstructionStabilizer.Config(
                    maxPredictionSeconds = 0.16,
                    maxSpeedMetersPerSecond = 12.0,
                    maxAccelerationMetersPerSecond2 = 90.0,
                    maxJerkMetersPerSecond3 = 600.0,
                    minimumAcceptedJoints = 1,
                ),
            )

        engine.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = 0L,
                camerasUsed = 3,
                clockSkewNanos = 1_000_000L,
                globalConfidence = 0.95,
                joints = listOf(
                    observation(
                        id = 7,
                        x = 0.10,
                        y = 0.10,
                        z = 1.00,
                    ),
                ),
            ),
        )

        engine.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = 20_000_000L,
                camerasUsed = 3,
                clockSkewNanos = 1_000_000L,
                globalConfidence = 0.95,
                joints = listOf(
                    observation(
                        id = 7,
                        x = 0.104,
                        y = 0.10,
                        z = 1.00,
                    ),
                ),
            ),
        )

        val predicted =
            engine.update(
                TemporalReconstructionStabilizer.FrameInput(
                    timestampNanos = 40_000_000L,
                    camerasUsed = 3,
                    clockSkewNanos = 1_000_000L,
                    globalConfidence = 0.90,
                    joints = emptyList(),
                ),
            )

        val joint = predicted.joints.single()
        assertTrue(joint.predicted)
        assertTrue(joint.positionMeters.finite())
        assertTrue(joint.velocityMetersPerSecond.finite())
        assertTrue(joint.accelerationMetersPerSecond2.finite())
        assertTrue(joint.jerkMetersPerSecond3.finite())
        assertTrue(
            joint.jerkMetersPerSecond3.norm() <=
                600.0 + 1e-6,
        )
    }

    @Test
    fun topologyConstraintDoesNotCreateNonFiniteGeometry() {
        val engine = TemporalReconstructionStabilizer()

        val result = engine.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = 10_000_000L,
                camerasUsed = 2,
                clockSkewNanos = 2_000_000L,
                globalConfidence = 0.80,
                joints = listOf(
                    observation(0, 0.0, 0.0, 1.0),
                    observation(1, 1.0, 0.0, 1.0),
                    observation(2, 0.0, 1.0, 1.0),
                    observation(3, 0.0, 0.0, 2.0),
                    observation(4, 0.0, 0.0, 3.0),
                    observation(5, 0.0, 0.0, 4.0),
                ),
                links = listOf(
                    TemporalReconstructionStabilizer.Link(
                        parentJointId = 0,
                        childJointId = 1,
                        nominalLengthMeters = 1.0,
                    ),
                    TemporalReconstructionStabilizer.Link(
                        parentJointId = 0,
                        childJointId = 2,
                        nominalLengthMeters = 1.0,
                    ),
                ),
            ),
        )

        assertTrue(result.joints.all {
            it.positionMeters.finite() &&
                it.velocityMetersPerSecond.finite()
        })
    }

    private fun observation(
        id: Int,
        x: Double,
        y: Double,
        z: Double,
    ): TemporalReconstructionStabilizer.JointObservation =
        TemporalReconstructionStabilizer.JointObservation(
            jointId = id,
            positionMeters =
                TemporalReconstructionStabilizer.Vec3(x, y, z),
            confidence = 0.85,
            uncertaintyMeters = 0.06,
            reprojectionErrorPx = 2.2,
            viewCount = 2,
        )
}
