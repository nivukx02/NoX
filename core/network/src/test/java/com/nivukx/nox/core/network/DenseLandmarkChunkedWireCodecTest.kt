package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DenseLandmarkChunkedWireCodecTest {

    @Test
    fun fiftyActorDenseEvidenceRoundTripsThroughBoundedChunks() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 4_000_000_000L,
                imageWidth = 2560,
                imageHeight = 1440,
                rollingShutterSkewNanos = 1_500_000L,
                frameDurationNanos = 8_333_333L,
                exposureTimeNanos = 2_000_000L,
                captureTimingConfidence = 0.87f,
                landmarks =
                    (0 until 50).flatMap { actor ->
                        (0 until 33).map { landmark ->
                            DenseLandmarkWireCodec.Landmark(
                                subjectId = actor,
                                landmarkId = landmark,
                                normalizedX = 0.2f,
                                normalizedY = 0.3f,
                                confidence = 0.97f,
                                uncertainty = 0.03f,
                                visibility = 0.98f,
                                contactProbability = 0.10f,
                                temporalStability = 0.95f,
                                crossViewAgreement = 0.92f,
                                reprojectionResidualNormalized = 0.02f,
                                probabilitySigmaXNormalized = 0.01f,
                                probabilitySigmaYNormalized = 0.01f,
                                probabilityCalibrationScore = 0.98f,
                                segmentationConsistency = 0.96f,
                                predicted = false,
                            )
                        }
                    },
            )

        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame = frame,
                streamId = "scene-a",
                frameSequence = 42L,
            )

        assertTrue(chunks.size >= 2)
        assertTrue(
            chunks.all {
                it.payload.size < NearbyPeerController.MAX_NEARBY_BYTES_PAYLOAD
            },
        )

        val decoded =
            DenseLandmarkChunkedWireCodec.decode(
                chunks = chunks.reversed(),
            )

        assertEquals(
            frame.timestampNanos,
            decoded.timestampNanos,
        )
        assertEquals(
            50 * 33,
            decoded.landmarks.size,
        )
        assertEquals(2560, decoded.imageWidth)
        assertEquals(1440, decoded.imageHeight)
        assertEquals(1_500_000L, decoded.rollingShutterSkewNanos)
        assertEquals(8_333_333L, decoded.frameDurationNanos)
        assertEquals(2_000_000L, decoded.exposureTimeNanos)
        assertEquals(0.87f, decoded.captureTimingConfidence, 1e-6f)
    }

    @Test
    fun missingChunkFailsClosed() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 5_000_000_000L,
                landmarks =
                    (0 until 2_000).map { index ->
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
                "scene-b",
                7L,
            )

        assertFailsWith<IllegalArgumentException> {
            DenseLandmarkChunkedWireCodec.decode(
                chunks.drop(1),
            )
        }
    }

    @Test
    fun duplicateChunkFailsClosed() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 6_000_000_000L,
                landmarks =
                    (0 until 1_200).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 33,
                            landmarkId = index % 33,
                            normalizedX = 0.4f,
                            normalizedY = 0.5f,
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
                "scene-c",
                9L,
            )

        assertFailsWith<IllegalArgumentException> {
            DenseLandmarkChunkedWireCodec.decode(
                chunks + chunks.first(),
            )
        }
    }
}


class DenseLandmarkChunkedWireCodecVersionNegotiationTest {
    @Test
    fun legacyV3FragmentsPreservePointsWithoutInventingTimingMetadata() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 12_000_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                rollingShutterSkewNanos = 1_200_000L,
                frameDurationNanos = 16_666_667L,
                exposureTimeNanos = 2_000_000L,
                captureTimingConfidence = 0.93f,
                landmarks =
                    (0 until 40).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 20,
                            landmarkId = index % 20,
                            normalizedX = 0.1f + index * 0.01f,
                            normalizedY = 0.2f + index * 0.01f,
                            confidence = 0.92f,
                            uncertainty = 0.08f,
                            visibility = 0.90f,
                            contactProbability = 0.1f,
                            temporalStability = 0.88f,
                            crossViewAgreement = 0.85f,
                            reprojectionResidualNormalized = 0.02f,
                            probabilitySigmaXNormalized = 0.003f,
                            probabilitySigmaYNormalized = 0.004f,
                            probabilityCalibrationScore = 0.90f,
                            segmentationConsistency = 0.94f,
                            predicted = false,
                        )
                    },
            )
        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame = frame,
                streamId = "legacy-peer",
                frameSequence = 12L,
                maximumLandmarksPerChunk = 32,
                maximumPayloadBytes = 2_500,
                wireVersion =
                    DenseLandmarkChunkedWireCodec.FrameWireVersion.LEGACY_V3,
            )

        assertTrue(chunks.size > 1)
        assertTrue(
            chunks.all {
                java.nio.ByteBuffer.wrap(it.payload)
                    .order(java.nio.ByteOrder.BIG_ENDIAN)
                    .int == 3
            },
        )
        val decoded = DenseLandmarkChunkedWireCodec.decode(chunks)
        assertEquals(40, decoded.landmarks.size)
        assertEquals(0, decoded.imageWidth)
        assertEquals(0, decoded.imageHeight)
        assertEquals(0.0f, decoded.captureTimingConfidence, 1e-6f)
    }
}


class DenseLandmarkChunkedSegmentationMetadataTest {
    @Test
    fun chunkedV4FramePreservesSegmentationAvailability() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 13_000_000_000L,
                landmarks =
                    (0 until 40).map { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index / 20,
                            landmarkId = index % 20,
                            normalizedX = 0.3f,
                            normalizedY = 0.6f,
                            confidence = 0.91f,
                            uncertainty = 0.09f,
                            visibility = 0.93f,
                            contactProbability = 0.1f,
                            temporalStability = 0.9f,
                            crossViewAgreement = 0.88f,
                            reprojectionResidualNormalized = 0.015f,
                            segmentationConsistency = 0.74f,
                            predicted = false,
                            segmentationEvidenceAvailable = true,
                        )
                    },
            )
        val chunks =
            DenseLandmarkChunkedWireCodec.encode(
                frame = frame,
                streamId = "segmentation-scene",
                frameSequence = 13L,
                maximumLandmarksPerChunk = 32,
                maximumPayloadBytes = 2_500,
            )
        val decoded = DenseLandmarkChunkedWireCodec.decode(chunks)

        assertTrue(decoded.landmarks.all { it.segmentationEvidenceAvailable })
        assertEquals(0.74f, decoded.landmarks.first().segmentationConsistency, 1e-6f)
    }
}
