package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class DenseLandmarkFragmentReassemblerTest {

    @Test
    fun outOfOrderFragmentsProduceOneFrame() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 8_000_000_000L,
                landmarks =
                    (0 until 1_200).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.2f,
                            normalizedY = 0.3f,
                            confidence = 0.95f,
                            uncertainty = 0.05f,
                            visibility = 0.98f,
                            contactProbability = 0.0f,
                            temporalStability = 0.95f,
                            crossViewAgreement = 0.92f,
                            reprojectionResidualNormalized = 0.02f,
                            predicted = false,
                        )
                    },
            )

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame = frame,
                streamId = "studio-a",
                frameSequence = 42L,
            )

        val reassembler =
            DenseLandmarkFragmentReassembler()

        var result:
            DenseLandmarkWireCodec.Frame? = null

        chunks.reversed().forEach { chunk ->
            result =
                reassembler.accept(
                    endpoint = "endpoint-a",
                    chunk = chunk,
                    nowNanos = 10_000L,
                ) ?: result
        }

        assertNotNull(result)
        assertEquals(1_200, result?.landmarks?.size)
    }

    @Test
    fun expiredPartialFrameIsDropped() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 9_000_000_000L,
                landmarks =
                    (0 until 1_000).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.2f,
                            normalizedY = 0.3f,
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

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame,
                "studio-b",
                2L,
            )

        val reassembler =
            DenseLandmarkFragmentReassembler(
                DenseLandmarkFragmentReassembler.Config(
                    maximumAssemblyAgeNanos = 100L,
                ),
            )

        assertNull(
            reassembler.accept(
                endpoint = "endpoint-b",
                chunk = chunks.first(),
                nowNanos = 1_000L,
            ),
        )

        reassembler.expire(nowNanos = 1_101L)

        assertNull(
            reassembler.accept(
                endpoint = "endpoint-b",
                chunk = chunks.last(),
                nowNanos = 1_200L,
            ),
        )
    }

    @Test
    fun globalAssemblyLimitPreventsUnboundedGrowth() {
        val reassembler =
            DenseLandmarkFragmentReassembler(
                DenseLandmarkFragmentReassembler.Config(
                    maximumInFlightAssemblies = 2,
                ),
            )

        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 10_000_000_000L,
                landmarks =
                    (0 until 1_000).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.1f,
                            normalizedY = 0.2f,
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

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame,
                "studio-c",
                3L,
                maximumLandmarksPerChunk = 64,
            )

        reassembler.accept("e1", chunks[0], 100L)
        reassembler.accept("e2", chunks[0], 101L)
        reassembler.accept("e3", chunks[0], 102L)

        assertEquals(
            2,
            reassembler.inFlightAssemblyCount(),
        )
    }
    @Test
    fun expiredPartialFrameReportsMissingChunkMask() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 9_500_000_000L,
                landmarks =
                    (0 until 300).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.2f,
                            normalizedY = 0.3f,
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

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame,
                "feedback-stream",
                88L,
                maximumLandmarksPerChunk = 64,
            )

        val reassembler =
            DenseLandmarkFragmentReassembler(
                DenseLandmarkFragmentReassembler.Config(
                    maximumAssemblyAgeNanos = 100L,
                ),
            )

        reassembler.accept(
            endpoint = "cam-feedback",
            chunk = chunks[0],
            nowNanos = 1_000L,
        )

        val expired =
            reassembler.expireWithFeedback(
                nowNanos = 1_101L,
            )

        assertEquals(1, expired.size)
        assertEquals(
            0b1,
            expired.first().receivedMask,
        )
        assertEquals(
            (1 shl chunks.size) - 2,
            expired.first().missingMask(),
        )
    }



    @Test
    fun timestampMismatchDropsPartialFrameAndReleasesPayloadBudget() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 9_600_000_000L,
                landmarks = (0 until 1_200).map { index ->
                    DenseLandmarkWireCodec.Landmark(
                        subjectId = index / 33,
                        landmarkId = index % 33,
                        normalizedX = 0.2f,
                        normalizedY = 0.3f,
                        confidence = 0.95f,
                        uncertainty = 0.05f,
                        visibility = 0.98f,
                        contactProbability = 0.0f,
                        temporalStability = 0.95f,
                        crossViewAgreement = 0.92f,
                        reprojectionResidualNormalized = 0.02f,
                        predicted = false,
                    )
                },
            )
        val chunks = DenseLandmarkChunkedWireCodec.encode(frame, "stream", 90L)
        assertEquals(2, chunks.size)
        val reassembler = DenseLandmarkFragmentReassembler()
        reassembler.accept("peer", chunks[0], 1L)
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            reassembler.accept("peer", chunks[1].copy(timestampNanos = frame.timestampNanos + 1L), 2L)
        }
        assertEquals(0, reassembler.inFlightAssemblyCount())
        assertEquals(0, reassembler.inFlightBytes())
    }

}
