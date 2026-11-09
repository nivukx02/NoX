package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals

class FaceMeshFragmentReassemblerTest {

    @Test
    fun outOfOrderFaceMeshFragmentsReassemble() {
        val frame =
            FaceMeshWireCodec.decode(
                FaceMeshWireCodec.encode(
                    FaceMeshWireCodecTestFrames.full(),
                ),
            )

        val chunks =
            FaceMeshChunkedWireCodec.encode(
                frame = frame,
                streamId = "face",
                frameSequence = 77L,
                maximumBytesPerChunk = 4096,
            )

        val reassembler =
            FaceMeshFragmentReassembler()

        var output:
            com.nivukx.nox.core.model.FaceMeshFrame? = null

        chunks.reversed().forEach { chunk ->
            output =
                reassembler.accept(
                    endpoint = "endpoint",
                    chunk = chunk,
                    nowNanos = 100L,
                ) ?: output
        }

        assertEquals(
            frame.points.size,
            output?.points?.size,
        )
        assertEquals(
            frame.triangles.size,
            output?.triangles?.size,
        )
    }

    private object FaceMeshWireCodecTestFrames {
        fun full() =
            com.nivukx.nox.core.model.FaceMeshFrame(
                timestampNanos = 15_000_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                rotationDegrees = 0,
                trackingId = 1,
                boundsLeft = 0f,
                boundsTop = 0f,
                boundsRight = 1f,
                boundsBottom = 1f,
                confidence = 0.97,
                pointCoverage = 1.0,
                triangleCoverage = 1.0,
                points =
                    (0 until 468).map {
                        com.nivukx.nox.core.model.FaceMeshPoint3D(
                            index = it,
                            position =
                                com.nivukx.nox.core.model.Vec3(
                                    it * 0.001,
                                    it * 0.001,
                                    it * 0.002,
                                ),
                        )
                    },
                triangles =
                    (0 until 466).map {
                        com.nivukx.nox.core.model.FaceMeshTriangle(
                            a = it,
                            b = it + 1,
                            c = it + 2,
                        )
                    },
            )
    }

    @Test
    fun timestampMismatchDropsPartialFrameAndReleasesPayloadBudget() {
        val frame = FaceMeshWireCodec.decode(
            FaceMeshWireCodec.encode(FaceMeshWireCodecTestFrames.full()),
        )
        val chunks = FaceMeshChunkedWireCodec.encode(
            frame = frame,
            streamId = "face",
            frameSequence = 78L,
            maximumBytesPerChunk = 4_096,
        )
        kotlin.test.assertTrue(chunks.size >= 2)
        val reassembler = FaceMeshFragmentReassembler()
        reassembler.accept("peer", chunks[0], 1L)
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept(
                "peer",
                chunks[1].copy(timestampNanos = frame.timestampNanos + 1L),
                2L,
            )
        }
        assertEquals(0, reassembler.inFlightAssemblyCount())
        assertEquals(0, reassembler.inFlightBytes())
    }

}
