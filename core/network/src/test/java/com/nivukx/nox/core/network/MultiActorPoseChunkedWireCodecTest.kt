package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MultiActorPoseChunkedWireCodecTest {

    @Test
    fun sixtyFourActorsWithWorldJointsAreSplitWithoutLoss() {
        val actors =
            (0 until 64).map { actorId ->
                MultiActorPoseWireCodec.Actor(
                    actorId = actorId.toLong(),
                    confidence = 0.95f,
                    predicted = false,
                    missedFrames = 0,
                    centerX = 0.5f,
                    centerY = 0.5f,
                    boundsLeft = 0.2f,
                    boundsTop = 0.1f,
                    boundsRight = 0.8f,
                    boundsBottom = 0.9f,
                    velocityX = 0.01f,
                    velocityY = 0.02f,
                    joints =
                        (0 until 33).map { jointId ->
                            MultiActorPoseWireCodec.Joint(
                                landmarkId = jointId,
                                x = 0.2f,
                                y = 0.3f,
                                confidence = 0.95f,
                            )
                        },
                    worldJoints =
                        (0 until 33).map { jointId ->
                            MultiActorPoseWireCodec.WorldJoint(
                                landmarkId = jointId,
                                xMeters = 0.1f,
                                yMeters = 0.2f,
                                zMeters = 2.0f,
                                confidence = 0.9f,
                            )
                        },
                    appearanceFingerprint =
                        List(16) { 0.5f },
                    appearanceQuality = 0.92f,
                )
            }

        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 11_000_000_000L,
                sourceDeviceId = "actor-rig",
                inputWidth = 3840,
                inputHeight = 2160,
                actors = actors,
            )

        val chunks =
            MultiActorPoseChunkedWireCodec.encode(
                frame = frame,
                streamId = "rig-a",
                frameSequence = 100L,
            )

        assertTrue(chunks.size >= 2)
        assertTrue(
            chunks.all {
                it.payload.size < NearbyPeerController.MAX_NEARBY_BYTES_PAYLOAD
            },
        )

        val decoded =
            MultiActorPoseChunkedWireCodec.decode(
                chunks.reversed(),
            )

        assertEquals(
            64,
            decoded.actors.size,
        )
    }
}
