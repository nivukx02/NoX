package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EvidenceFragmentRetransmissionCacheTest {
    private data class EvidenceFragmentRetransmissionCacheTestArgs(
        val endpoint: String,
        val streamId: String,
        val frameSequence: Long,
        val chunkIndex: Int,
        val chunkCount: Int,
    )


    @Test
    fun nackReturnsOnlyMissingFragmentsWithFreshPacketSequences() {
        val cache =
            EvidenceFragmentRetransmissionCache(
                EvidenceFragmentRetransmissionCache.Config(
                    maximumRetriesPerFrame = 2,
                ),
            )

        val packets =
            (0 until 3).map { index ->
                NoXWireProtocol.encode(
                    type =
                        NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                    sequence = (100 + index).toLong(),
                    timestampNanos = 1_000L,
                    payload =
                        byteArrayOf(
                            index.toByte(),
                            9,
                        ),
                )
            }

        packets.forEachIndexed { index, packet ->
            cache.remember(
                endpoint = "cam-a",
                evidenceType =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                streamId = "stream-a",
                frameSequence = 7L,
                chunkIndex = index,
                chunkCount = 3,
                packet = packet,
                reconstructionCritical = true,
                nowNanos = 10L,
            )
        }

        val resent =
            cache.handle(
                endpoint = "cam-a",
                feedback =
                    EvidenceFragmentFeedbackWireCodec.Feedback(
                        direction =
                            EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                        evidenceType =
                            NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                        streamId = "stream-a",
                        frameSequence = 7L,
                        chunkCount = 3,
                        receivedMask = 0b101,
                    ),
                nowNanos = 20L,
                nextPacketSequence = { 500L },
            )

        assertEquals(1, resent.size)
        assertEquals(
            NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            NoXWireProtocol.decode(resent.first().packet).type,
        )
        assertEquals(
            500L,
            NoXWireProtocol.decode(resent.first().packet).sequence,
        )
        assertTrue(resent.first().reconstructionCritical)
    }

    @Test
    fun critical_missing_chunks_are_retransmitted_before_noncritical_chunks() {
        val cache =
            EvidenceFragmentRetransmissionCache(
                EvidenceFragmentRetransmissionCache.Config(
                    maximumRetriesPerFrame = 2,
                ),
            )

        val packets =
            (0 until 4).map { index ->
                NoXWireProtocol.encode(
                    type =
                        NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                    sequence = (200 + index).toLong(),
                    timestampNanos = 1_000L,
                    payload = byteArrayOf(index.toByte()),
                )
            }

        packets.forEachIndexed { index, packet ->
            cache.remember(
                endpoint = "cam-priority",
                evidenceType =
                    NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                streamId = "face",
                frameSequence = 12L,
                chunkIndex = index,
                chunkCount = 4,
                packet = packet,
                reconstructionCritical = index == 2,
                nowNanos = 10L,
            )
        }

        val resent =
            cache.handle(
                endpoint = "cam-priority",
                feedback =
                    EvidenceFragmentFeedbackWireCodec.Feedback(
                        direction =
                            EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                        evidenceType =
                            NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                        streamId = "face",
                        frameSequence = 12L,
                        chunkCount = 4,
                        receivedMask = 0,
                    ),
                nowNanos = 20L,
                nextPacketSequence =
                    object {
                        var next = 700L
                        fun allocate(): Long = next++
                    }::allocate,
            )

        assertEquals(4, resent.size)
        assertEquals(2, resent.first().chunkIndex)
        assertTrue(resent.first().reconstructionCritical)
        assertEquals(0, resent[1].chunkIndex)
    }

    @Test
    fun acknowledgementEvictsReceivedChunks() {
        val cache =
            EvidenceFragmentRetransmissionCache()

        val packet =
            NoXWireProtocol.encode(
                type =
                    NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                sequence = 1L,
                timestampNanos = 1_000L,
                payload = byteArrayOf(1, 2, 3),
            )

        cache.remember(
            endpoint = "cam-b",
            evidenceType =
                NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
            streamId = "stream-b",
            frameSequence = 9L,
            chunkIndex = 0,
            chunkCount = 1,
            packet = packet,
            reconstructionCritical = false,
            nowNanos = 10L,
        )

        cache.handle(
            endpoint = "cam-b",
            feedback =
                EvidenceFragmentFeedbackWireCodec.Feedback(
                    direction =
                        EvidenceFragmentFeedbackWireCodec.Direction.ACK,
                    evidenceType =
                        NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                    streamId = "stream-b",
                    frameSequence = 9L,
                    chunkCount = 1,
                    receivedMask = 1,
                ),
            nowNanos = 20L,
            nextPacketSequence = { 2L },
        )

        assertEquals(0, cache.inFlightFrameCount())
    }

    @Test
    fun identicalDuplicateChunkIsIdempotentAndDoesNotDoubleCountBytes() {
        val cache = EvidenceFragmentRetransmissionCache()
        val packet = NoXWireProtocol.encode(
            type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            sequence = 100L,
            timestampNanos = 1_000L,
            payload = byteArrayOf(1, 2, 3, 4),
        )
        val args = EvidenceFragmentRetransmissionCacheTestArgs(
            endpoint = "camera-idempotent",
            streamId = "dense",
            frameSequence = 21L,
            chunkIndex = 0,
            chunkCount = 2,
        )
        cache.remember(
            endpoint = args.endpoint,
            evidenceType = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            streamId = args.streamId,
            frameSequence = args.frameSequence,
            chunkIndex = args.chunkIndex,
            chunkCount = args.chunkCount,
            packet = packet,
            reconstructionCritical = false,
            nowNanos = 10L,
        )
        val retained = cache.retainedBytes()
        cache.remember(
            endpoint = args.endpoint,
            evidenceType = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
            streamId = args.streamId,
            frameSequence = args.frameSequence,
            chunkIndex = args.chunkIndex,
            chunkCount = args.chunkCount,
            packet = NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                sequence = 101L,
                timestampNanos = 1_000L,
                payload = byteArrayOf(1, 2, 3, 4),
            ),
            reconstructionCritical = true,
            nowNanos = 11L,
        )

        assertEquals(retained, cache.retainedBytes())
        assertEquals(1, cache.inFlightFrameCount())
        val retransmission = cache.handle(
            endpoint = args.endpoint,
            feedback = EvidenceFragmentFeedbackWireCodec.Feedback(
                direction = EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                evidenceType = NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                streamId = args.streamId,
                frameSequence = args.frameSequence,
                chunkCount = args.chunkCount,
                receivedMask = 0,
            ),
            nowNanos = 12L,
            nextPacketSequence = { 500L },
        )
        assertTrue(retransmission.single().reconstructionCritical)
    }

    @Test
    fun conflictingDuplicateChunkDropsTheFrameAndReleasesRetainedBytes() {
        val cache = EvidenceFragmentRetransmissionCache()
        fun packet(sequence: Long, payload: ByteArray) =
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                sequence = sequence,
                timestampNanos = 2_000L,
                payload = payload,
            )
        cache.remember(
            endpoint = "camera-conflict",
            evidenceType = NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
            streamId = "face",
            frameSequence = 22L,
            chunkIndex = 0,
            chunkCount = 2,
            packet = packet(1L, byteArrayOf(1, 2, 3)),
            reconstructionCritical = true,
            nowNanos = 10L,
        )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            cache.remember(
                endpoint = "camera-conflict",
                evidenceType = NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                streamId = "face",
                frameSequence = 22L,
                chunkIndex = 0,
                chunkCount = 2,
                packet = packet(2L, byteArrayOf(9, 8, 7)),
                reconstructionCritical = true,
                nowNanos = 11L,
            )
        }
        assertEquals(0, cache.inFlightFrameCount())
        assertEquals(0, cache.retainedBytes())
    }

    @Test
    fun failedFrameByteBudgetAdmissionRemovesItsExistingChunks() {
        val cache = EvidenceFragmentRetransmissionCache(
            EvidenceFragmentRetransmissionCache.Config(
                maximumRetainedBytes = 64 * 1024,
            ),
        )
        fun maxPacket(sequence: Long) =
            NoXWireProtocol.encode(
                type = NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                sequence = sequence,
                timestampNanos = 3_000L,
                payload = ByteArray(24_000) { 7 },
            )

        cache.remember(
            endpoint = "camera-budget",
            evidenceType = NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
            streamId = "pose",
            frameSequence = 23L,
            chunkIndex = 0,
            chunkCount = 3,
            packet = maxPacket(1L),
            reconstructionCritical = true,
            nowNanos = 10L,
        )
        cache.remember(
            endpoint = "camera-budget",
            evidenceType = NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
            streamId = "pose",
            frameSequence = 23L,
            chunkIndex = 1,
            chunkCount = 3,
            packet = maxPacket(2L),
            reconstructionCritical = true,
            nowNanos = 11L,
        )
        assertEquals(2 * maxPacket(1L).size, cache.retainedBytes())

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            cache.remember(
                endpoint = "camera-budget",
                evidenceType = NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                streamId = "pose",
                frameSequence = 23L,
                chunkIndex = 2,
                chunkCount = 3,
                packet = maxPacket(3L),
                reconstructionCritical = true,
                nowNanos = 12L,
            )
        }

        assertEquals(0, cache.inFlightFrameCount())
        assertEquals(0, cache.retainedBytes())
    }

}
