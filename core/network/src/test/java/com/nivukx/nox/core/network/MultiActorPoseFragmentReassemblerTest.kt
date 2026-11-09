package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals

class MultiActorPoseFragmentReassemblerTest {

    @Test
    fun outOfOrderActorFragmentsReassembleLosslessly() {
        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 13_000_000_000L,
                sourceDeviceId = "rig",
                inputWidth = 3840,
                inputHeight = 2160,
                actors =
                    (0 until 20).map { id ->
                        MultiActorPoseWireCodec.Actor(
                            actorId = id.toLong(),
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
                    },
            )

        val chunks =
            MultiActorPoseChunkedWireCodec.encode(
                frame,
                "rig-stream",
                50L,
            )

        val reassembler =
            MultiActorPoseFragmentReassembler()

        var output: MultiActorPoseWireCodec.Frame? = null

        chunks.reversed().forEach { chunk ->
            val packet =
                MultiActorPoseFragmentWireCodec.encode(
                    chunk,
                    1_000L + chunk.chunkIndex,
                )
            output =
                reassembler.accept(
                    endpoint = "endpoint",
                    chunk =
                        MultiActorPoseFragmentWireCodec.decode(
                            NoXWireProtocol.decode(packet),
                        ),
                    nowNanos = 100L,
                ) ?: output
        }

        assertEquals(20, output?.actors?.size)
    }

    @Test
    fun timestampMismatchDropsPartialFrameAndReleasesPayloadBudget() {
        val frame =
            MultiActorPoseWireCodec.Frame(
                timestampNanos = 13_500_000_000L,
                sourceDeviceId = "rig",
                inputWidth = 3840,
                inputHeight = 2160,
                actors = (0 until 64).map { id ->
                    MultiActorPoseWireCodec.Actor(
                        actorId = id.toLong(),
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
                },
            )
        val chunks = MultiActorPoseChunkedWireCodec.encode(
            frame = frame,
            streamId = "rig-stream",
            frameSequence = 51L,
            maximumPayloadBytes = 1_024,
        )
        kotlin.test.assertTrue(chunks.size >= 2)
        val reassembler = MultiActorPoseFragmentReassembler()
        reassembler.accept("peer", chunks[0], 1L)
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept("peer", chunks[1].copy(timestampNanos = frame.timestampNanos + 1L), 2L)
        }
        assertEquals(0, reassembler.inFlightAssemblyCount())
        assertEquals(0, reassembler.inFlightBytes())
    }

}
