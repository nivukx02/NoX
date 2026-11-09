package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DenseLandmarkFragmentWireCodecTest {

    @Test
    fun fragmentPacketRoundTripsThroughProtocol() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 7_000_000_000L,
                landmarks =
                    (0 until 100).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.2f,
                            normalizedY = 0.4f,
                            confidence = 0.9f,
                            uncertainty = 0.1f,
                            visibility = 1.0f,
                            contactProbability = 0.0f,
                            temporalStability = 0.9f,
                            crossViewAgreement = 0.9f,
                            reprojectionResidualNormalized = 0.02f,
                            predicted = false,
                        )
                    },
            )

        val chunk =
            DenseLandmarkChunkedWireCodec.encode(
                frame = frame,
                streamId = "studio-a",
                frameSequence = 11L,
                maximumLandmarksPerChunk = 64,
            ).first()

        val packet =
            DenseLandmarkFragmentWireCodec.encode(
                chunk = chunk,
                packetSequence = 99L,
            )

        assertTrue(
            packet.size <=
                NearbyPeerController.MAX_NEARBY_BYTES_PAYLOAD,
        )

        val decodedPacket =
            NoXWireProtocol.decode(packet)

        assertEquals(
            NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            decodedPacket.type,
        )

        val decodedChunk =
            DenseLandmarkFragmentWireCodec.decode(
                packet = decodedPacket,
            )

        assertEquals(chunk.streamId, decodedChunk.streamId)
        assertEquals(chunk.frameSequence, decodedChunk.frameSequence)
        assertEquals(chunk.chunkIndex, decodedChunk.chunkIndex)
        assertEquals(chunk.chunkCount, decodedChunk.chunkCount)
        assertTrue(chunk.payload.contentEquals(decodedChunk.payload))
        assertEquals(chunk.crc32, decodedChunk.crc32)
    }
}
