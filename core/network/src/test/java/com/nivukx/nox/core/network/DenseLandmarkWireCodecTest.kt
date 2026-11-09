package com.nivukx.nox.core.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseLandmarkWireCodecTest {

    @Test
    fun roundTripPreservesBoundedDenseEvidenceAndV4TimingProvenance() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 1_000_000_000L,
                imageWidth = 1920,
                imageHeight = 1080,
                rollingShutterSkewNanos = 3_250_000L,
                frameDurationNanos = 16_666_667L,
                exposureTimeNanos = 4_000_000L,
                captureTimingConfidence = 0.91f,
                landmarks =
                    listOf(
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = 2,
                            landmarkId = 17,
                            normalizedX = 0.42f,
                            normalizedY = 0.63f,
                            confidence = 0.94f,
                            uncertainty = 0.08f,
                            visibility = 0.91f,
                            contactProbability = 0.12f,
                            temporalStability = 0.88f,
                            crossViewAgreement = 0.76f,
                            reprojectionResidualNormalized = 0.03f,
                            probabilitySigmaXNormalized = 0.0012f,
                            probabilitySigmaYNormalized = 0.0018f,
                            probabilityCalibrationScore = 0.93f,
                            segmentationConsistency = 0.91f,
                            predicted = false,
                            segmentationEvidenceAvailable = true,
                        ),
                    ),
            )

        val decoded =
            DenseLandmarkWireCodec.decode(
                DenseLandmarkWireCodec.encode(frame),
                frame.timestampNanos,
            )

        assertEquals(1, decoded.landmarks.size)
        val point = decoded.landmarks.single()
        assertEquals(2, point.subjectId)
        assertEquals(17, point.landmarkId)
        assertEquals(0.42f, point.normalizedX, 1e-5f)
        assertEquals(0.94f, point.confidence, 1e-5f)
        assertEquals(0.0012f, point.probabilitySigmaXNormalized, 1e-6f)
        assertEquals(0.0018f, point.probabilitySigmaYNormalized, 1e-6f)
        assertEquals(0.93f, point.probabilityCalibrationScore, 1e-6f)
        assertEquals(0.91f, point.segmentationConsistency, 1e-6f)
        assertTrue(point.segmentationEvidenceAvailable)
        assertEquals(1920, decoded.imageWidth)
        assertEquals(1080, decoded.imageHeight)
        assertEquals(3_250_000L, decoded.rollingShutterSkewNanos)
        assertEquals(16_666_667L, decoded.frameDurationNanos)
        assertEquals(4_000_000L, decoded.exposureTimeNanos)
        assertEquals(0.91f, decoded.captureTimingConfidence, 1e-6f)
        assertTrue(!point.predicted)
    }

    @Test
    fun roundTripPreservesSilhouetteConsistencyInV3() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 6_000_000_000L,
                landmarks =
                    listOf(
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = 4,
                            landmarkId = 11,
                            normalizedX = 0.41f,
                            normalizedY = 0.58f,
                            confidence = 0.95f,
                            uncertainty = 0.06f,
                            visibility = 0.94f,
                            contactProbability = 0.08f,
                            temporalStability = 0.92f,
                            crossViewAgreement = 0.87f,
                            reprojectionResidualNormalized = 0.01f,
                            probabilitySigmaXNormalized = 0.001f,
                            probabilitySigmaYNormalized = 0.001f,
                            probabilityCalibrationScore = 0.96f,
                            segmentationConsistency = 0.91f,
                            predicted = false,
                        ),
                    ),
            )

        val decoded =
            DenseLandmarkWireCodec.decode(
                DenseLandmarkWireCodec.encode(frame),
                frame.timestampNanos,
            )

        assertEquals(
            0.91f,
            decoded.landmarks.single().segmentationConsistency,
            1e-6f,
        )
    }

    @Test
    fun nonFiniteValuesAreSanitizedBeforeTransport() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 2_000_000_000L,
                landmarks =
                    listOf(
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = -10,
                            landmarkId = 99_999,
                            normalizedX = Float.NaN,
                            normalizedY = Float.POSITIVE_INFINITY,
                            confidence = Float.NaN,
                            uncertainty = Float.NaN,
                            visibility = 4f,
                            contactProbability = -2f,
                            temporalStability = Float.NaN,
                            crossViewAgreement = Float.NaN,
                            reprojectionResidualNormalized = Float.NaN,
                            predicted = true,
                        ),
                    ),
            )

        val decoded =
            DenseLandmarkWireCodec.decode(
                DenseLandmarkWireCodec.encode(frame),
                frame.timestampNanos,
            )

        val point = decoded.landmarks.single()
        assertEquals(0, point.subjectId)
        assertEquals(4095, point.landmarkId)
        assertEquals(0f, point.normalizedX, 1e-6f)
        assertEquals(0f, point.confidence, 1e-6f)
        assertEquals(1f, point.uncertainty, 1e-6f)
        assertEquals(1f, point.visibility, 1e-6f)
        assertTrue(point.predicted)
    }

    @Test
    fun pointCountIsCappedToPacketBudget() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 3_000_000_000L,
                landmarks =
                    List(2000) { index ->
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = index % 8,
                            landmarkId = index,
                            normalizedX = 0.5f,
                            normalizedY = 0.5f,
                            confidence = (1.0f - index / 4000.0f)
                                .coerceAtLeast(0.01f),
                            uncertainty = 0.1f,
                            visibility = 1.0f,
                            contactProbability = 0.0f,
                            temporalStability = 0.9f,
                            crossViewAgreement = 0.5f,
                            reprojectionResidualNormalized = 0.01f,
                            predicted = false,
                        )
                    },
            )

        val bytes = DenseLandmarkWireCodec.encode(frame)
        assertTrue(bytes.size < NoXWireProtocol.MAX_PACKET_BYTES)

        val decoded =
            DenseLandmarkWireCodec.decode(
                bytes,
                frame.timestampNanos,
            )
        assertTrue(decoded.landmarks.size <= 1024)
    }

    @Test
    fun legacyVersionOneEnvelopeWithPaddingRemainsDecodable() {
        val timestamp = 5_000_000_000L
        val source =
            DenseLandmarkWireCodec.Landmark(
                subjectId = 1,
                landmarkId = 2,
                normalizedX = 0.25f,
                normalizedY = 0.35f,
                confidence = 0.90f,
                uncertainty = 0.12f,
                visibility = 0.91f,
                contactProbability = 0.10f,
                temporalStability = 0.88f,
                crossViewAgreement = 0.80f,
                reprojectionResidualNormalized = 0.02f,
                predicted = true,
            )

        val bytes =
            java.nio.ByteBuffer
                .allocate(12 + 45)
                .order(java.nio.ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(1)
                    putInt(1)
                    putInt(45)
                    putShort(source.subjectId.toShort())
                    putShort(source.landmarkId.toShort())
                    putFloat(source.normalizedX)
                    putFloat(source.normalizedY)
                    putFloat(source.confidence)
                    putFloat(source.uncertainty)
                    putFloat(source.visibility)
                    putFloat(source.contactProbability)
                    putFloat(source.temporalStability)
                    putFloat(source.crossViewAgreement)
                    putFloat(source.reprojectionResidualNormalized)
                    put(
                        if (source.predicted) {
                            1.toByte()
                        } else {
                            0.toByte()
                        },
                    )
                    putInt(0)
                }
                .array()

        val decoded =
            DenseLandmarkWireCodec.decode(
                bytes,
                timestamp,
            )

        assertEquals(1, decoded.landmarks.size)
        assertEquals(
            0f,
            decoded.landmarks.single().probabilityCalibrationScore,
            1e-6f,
        )
        assertEquals(
            0f,
            decoded.landmarks.single().probabilitySigmaXNormalized,
            1e-6f,
        )
    }

    @Test
    fun malformedPayloadIsRejected() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 4_000_000_000L,
                landmarks = emptyList(),
            )
        val bytes =
            DenseLandmarkWireCodec.encode(frame)
                .copyOf()

        bytes[7] = 1

        try {
            DenseLandmarkWireCodec.decode(
                bytes,
                frame.timestampNanos,
            )
            throw AssertionError(
                "Malformed payload must be rejected",
            )
        } catch (_: IllegalArgumentException) {
            // expected
        }
    }
}


class DenseLandmarkWireCodecVersionThreeCompatibilityTest {
    @Test
    fun versionThreeFrameDecodesWithoutInventingTimingProvenance() {
        val timestamp = 9_000_000_000L
        val bytes =
            java.nio.ByteBuffer
                .allocate(12 + 57)
                .order(java.nio.ByteOrder.BIG_ENDIAN)
                .apply {
                    putInt(3)
                    putInt(1)
                    putInt(57)
                    putShort(2.toShort())
                    putShort(21.toShort())
                    putFloat(0.25f)
                    putFloat(0.35f)
                    putFloat(0.90f)
                    putFloat(0.12f)
                    putFloat(0.91f)
                    putFloat(0.10f)
                    putFloat(0.88f)
                    putFloat(0.80f)
                    putFloat(0.02f)
                    putFloat(0.001f)
                    putFloat(0.002f)
                    putFloat(0.95f)
                    putFloat(0.90f)
                    put(0.toByte())
                }
                .array()

        val decoded =
            DenseLandmarkWireCodec.decode(
                bytes = bytes,
                timestampNanos = timestamp,
            )

        assertEquals(1, decoded.landmarks.size)
        assertEquals(2, decoded.landmarks.single().subjectId)
        assertEquals(0, decoded.imageWidth)
        assertEquals(0, decoded.imageHeight)
        assertEquals(0L, decoded.rollingShutterSkewNanos)
        assertEquals(0L, decoded.frameDurationNanos)
        assertEquals(0L, decoded.exposureTimeNanos)
        assertEquals(0.0f, decoded.captureTimingConfidence, 1e-6f)
        assertTrue(!decoded.landmarks.single().segmentationEvidenceAvailable)
    }
}


class DenseLandmarkWireCodecLegacyEncoderTest {
    @Test
    fun v3CompatibilityEncodingPreservesLandmarksButDoesNotInventFrameTiming() {
        val frame =
            DenseLandmarkWireCodec.Frame(
                timestampNanos = 10_000_000_000L,
                imageWidth = 1280,
                imageHeight = 720,
                rollingShutterSkewNanos = 900_000L,
                frameDurationNanos = 16_666_667L,
                exposureTimeNanos = 3_000_000L,
                captureTimingConfidence = 0.94f,
                landmarks =
                    listOf(
                        DenseLandmarkWireCodec.Landmark(
                            subjectId = 3,
                            landmarkId = 27,
                            normalizedX = 0.42f,
                            normalizedY = 0.61f,
                            confidence = 0.93f,
                            uncertainty = 0.07f,
                            visibility = 0.92f,
                            contactProbability = 0.12f,
                            temporalStability = 0.91f,
                            crossViewAgreement = 0.88f,
                            reprojectionResidualNormalized = 0.01f,
                            probabilitySigmaXNormalized = 0.002f,
                            probabilitySigmaYNormalized = 0.003f,
                            probabilityCalibrationScore = 0.90f,
                            segmentationConsistency = 0.95f,
                            predicted = false,
                        ),
                    ),
            )

        val encoded = DenseLandmarkWireCodec.encodeV3Compatibility(frame)
        assertEquals(
            3,
            java.nio.ByteBuffer.wrap(encoded).order(java.nio.ByteOrder.BIG_ENDIAN).int,
        )
        assertEquals(12 + 57, encoded.size)

        val decoded =
            DenseLandmarkWireCodec.decode(
                bytes = encoded,
                timestampNanos = frame.timestampNanos,
            )

        assertEquals(1, decoded.landmarks.size)
        assertEquals(3, decoded.landmarks.single().subjectId)
        assertEquals(27, decoded.landmarks.single().landmarkId)
        assertEquals(0, decoded.imageWidth)
        assertEquals(0, decoded.imageHeight)
        assertEquals(0.0f, decoded.captureTimingConfidence, 1e-6f)
    }
}
