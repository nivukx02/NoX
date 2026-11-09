package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals

class EvidenceFragmentFeedbackWireCodecTest {

    @Test
    fun nackRoundTripsWithMissingChunkMask() {
        val feedback =
            EvidenceFragmentFeedbackWireCodec.Feedback(
                direction =
                    EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                evidenceType =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                streamId = "rig-a",
                frameSequence = 42L,
                chunkCount = 8,
                receivedMask = 0b1011,
            )

        val decoded =
            EvidenceFragmentFeedbackWireCodec.decode(
                NoXWireProtocol.decode(
                    EvidenceFragmentFeedbackWireCodec.encode(
                        feedback = feedback,
                        packetSequence = 7L,
                        timestampNanos = 100L,
                    ),
                ),
            )

        assertEquals(feedback, decoded)
        assertEquals(
            setOf(2, 4, 5, 6, 7),
            decoded.missingChunkIndices(),
        )
    }
}
