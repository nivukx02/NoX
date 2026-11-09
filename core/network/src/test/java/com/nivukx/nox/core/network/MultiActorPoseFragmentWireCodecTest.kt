package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MultiActorPoseFragmentWireCodecTest {

    @Test
    fun actorFragmentRoundTripsThroughNoXProtocol() {
        val actor =
            MultiActorPoseWireCodec.Actor(
                actorId = 1L,
                confidence = 0.95f,
                predicted = false,
                missedFrames = 0,
                centerX = 0.5f,
                centerY = 0.5f,
                boundsLeft = 0.2f,
                boundsTop = 0.1f,
                boundsRight = 0.8f,
                boundsBottom = 0.9f,
                velocityX = 0.0f,
                velocityY = 0.0f,
                joints = emptyList(),
            )

        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 12_000_000_000L,
                sourceDeviceId = "rig-a",
                inputWidth = 1920,
                inputHeight = 1080,
                actors = listOf(actor),
            )

        val chunk =
            MultiActorPoseChunkedWireCodec
                .encode(
                    frame,
                    "stream-a",
                    1L,
                )
                .first()

        val packet =
            MultiActorPoseFragmentWireCodec.encode(
                chunk,
                2L,
            )

        assertTrue(
            packet.size <=
                NearbyPeerController.MAX_NEARBY_BYTES_PAYLOAD,
        )

        val decoded =
            MultiActorPoseFragmentWireCodec.decode(
                NoXWireProtocol.decode(packet),
            )

        assertEquals(chunk.streamId, decoded.streamId)
        assertEquals(
            chunk.frameSequence,
            decoded.frameSequence,
        )
        assertEquals(
            chunk.chunkIndex,
            decoded.chunkIndex,
        )
        assertEquals(
            chunk.chunkCount,
            decoded.chunkCount,
        )
        assertEquals(
            chunk.payload.size,
            decoded.payload.size,
        )
    }
}
