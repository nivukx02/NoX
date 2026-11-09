package com.nivukx.nox.core.network

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Compact versioned transport for dense 2D landmark evidence.
 *
 * The payload intentionally carries evidence, not rendered meshes or metric 3D coordinates.
 * Metric reconstruction remains on the calibrated reconstruction side after peer timestamps
 * have been normalized.
 */
object DenseLandmarkWireCodec {
    private const val VERSION = 4
    private const val VERSION_3 = 3
    private const val VERSION_2 = 2
    private const val LEGACY_VERSION = 1
    const val MAX_LANDMARKS_PER_FRAME = 1024
    private const val MAX_POINTS = MAX_LANDMARKS_PER_FRAME
    private const val HEADER_BYTES = 12
    private const val HEADER_BYTES_V4 = 48
    private const val LEGACY_POINT_BYTES = 45
    private const val POINT_BYTES_V2 = 53
    private const val POINT_BYTES = 57

    data class Landmark(
        val subjectId: Int,
        val landmarkId: Int,
        val normalizedX: Float,
        val normalizedY: Float,
        val confidence: Float,
        val uncertainty: Float,
        val visibility: Float,
        val contactProbability: Float,
        val temporalStability: Float,
        val crossViewAgreement: Float,
        val reprojectionResidualNormalized: Float,
        val probabilitySigmaXNormalized: Float = 0f,
        val probabilitySigmaYNormalized: Float = 0f,
        val probabilityCalibrationScore: Float = 1f,
        val segmentationConsistency: Float = 0f,
        val predicted: Boolean,
        val segmentationEvidenceAvailable: Boolean = false,
    ) {
        fun sanitized(): Landmark =
            copy(
                subjectId = subjectId.coerceIn(0, 1023),
                landmarkId = landmarkId.coerceIn(0, 4095),
                normalizedX = normalizedX.finiteOrZero().coerceIn(-4f, 4f),
                normalizedY = normalizedY.finiteOrZero().coerceIn(-4f, 4f),
                confidence = confidence.finiteOrZero().coerceIn(0f, 1f),
                uncertainty = uncertainty.finiteOrOne().coerceIn(0f, 1f),
                visibility = visibility.finiteOrZero().coerceIn(0f, 1f),
                contactProbability =
                    contactProbability.finiteOrZero().coerceIn(0f, 1f),
                temporalStability =
                    temporalStability.finiteOrZero().coerceIn(0f, 1f),
                crossViewAgreement =
                    crossViewAgreement.finiteOrZero().coerceIn(0f, 1f),
                reprojectionResidualNormalized =
                    abs(
                        reprojectionResidualNormalized.finiteOrZero(),
                    ).coerceAtMost(4f),
                probabilitySigmaXNormalized =
                    probabilitySigmaXNormalized.finiteOrZero()
                        .coerceIn(0f, 0.25f),
                probabilitySigmaYNormalized =
                    probabilitySigmaYNormalized.finiteOrZero()
                        .coerceIn(0f, 0.25f),
                probabilityCalibrationScore =
                    probabilityCalibrationScore.finiteOrOne()
                        .coerceIn(0f, 1f),
                segmentationConsistency =
                    segmentationConsistency.finiteOrZero()
                        .coerceIn(0f, 1f),
            )
    }

    data class Frame(
        val timestampNanos: Long,
        val landmarks: List<Landmark>,
        // Zero dimensions mean that an older peer did not transmit this provenance.
        val imageWidth: Int = 0,
        val imageHeight: Int = 0,
        val rollingShutterSkewNanos: Long = 0L,
        val frameDurationNanos: Long = 0L,
        val exposureTimeNanos: Long = 0L,
        val captureTimingConfidence: Float = 0.0f,
    ) {
        init {
            require(imageWidth in 0..16_384)
            require(imageHeight in 0..16_384)
            require(
                (imageWidth == 0 && imageHeight == 0) ||
                    (imageWidth > 0 && imageHeight > 0),
            )
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(captureTimingConfidence.isFinite())
            require(captureTimingConfidence in 0.0f..1.0f)
        }
    }

    fun encodedSizeForLandmarkCount(
        count: Int,
    ): Int {
        require(count in 0..MAX_POINTS)
        return HEADER_BYTES_V4 +
            count * POINT_BYTES
    }

    fun encodedSizeForV3LandmarkCount(
        count: Int,
    ): Int {
        require(count in 0..MAX_POINTS)
        return HEADER_BYTES +
            count * POINT_BYTES
    }

    /**
     * Encodes the original v3 dense-frame layout for peers that did not advertise fragment support.
     *
     * Frame-level timing provenance is intentionally omitted for compatibility. Callers must only
     * choose this path when the whole legacy packet fits the transport limit and the peer cannot
     * receive v4 fragments.
     */
    fun encodeV3Compatibility(frame: Frame): ByteArray {
        require(frame.timestampNanos > 0L)
        require(frame.landmarks.size in 0..MAX_POINTS) {
            "Legacy dense frame exceeds single-packet landmark capacity"
        }
        val encodedV4 = encode(frame)
        val legacySize = HEADER_BYTES + frame.landmarks.size * POINT_BYTES
        val legacy = ByteBuffer.allocate(legacySize)
            .order(ByteOrder.BIG_ENDIAN)
            .apply {
                putInt(VERSION_3)
                putInt(frame.landmarks.size)
                putInt(POINT_BYTES)
                put(
                    encodedV4,
                    HEADER_BYTES_V4,
                    encodedV4.size - HEADER_BYTES_V4,
                )
            }
            .array()
        // V1-V3 readers treat any non-zero flags byte as predicted. Strip the v4
        // segmentation-availability bit to avoid turning confirmed points into predictions.
        repeat(frame.landmarks.size) { index ->
            val flagsOffset = HEADER_BYTES + index * POINT_BYTES + POINT_BYTES - 1
            legacy[flagsOffset] = (legacy[flagsOffset].toInt() and 0x01).toByte()
        }
        return legacy
    }

    fun encode(frame: Frame): ByteArray {
        require(frame.timestampNanos > 0L)
        val selected =
            frame.landmarks
                .map(Landmark::sanitized)

        require(
            selected.distinctBy {
                it.subjectId to it.landmarkId
            }.size == selected.size,
        ) {
            "Duplicate dense-landmark keys cannot be silently merged"
        }

        require(selected.size <= MAX_POINTS) {
            "Dense landmark payload exceeds single-packet capacity; use DenseLandmarkChunkedWireCodec"
        }

        val ordered =
            selected.sortedByDescending {
                it.confidence * it.visibility
            }

        val buffer = ByteBuffer
            .allocate(
                encodedSizeForLandmarkCount(
                    selected.size,
                ),
            )
            .order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(VERSION)
        buffer.putInt(selected.size)
        buffer.putInt(POINT_BYTES)
        buffer.putInt(frame.imageWidth)
        buffer.putInt(frame.imageHeight)
        buffer.putLong(frame.rollingShutterSkewNanos)
        buffer.putLong(frame.frameDurationNanos)
        buffer.putLong(frame.exposureTimeNanos)
        buffer.putFloat(frame.captureTimingConfidence)

        ordered.forEach { point ->
            buffer.putShort(point.subjectId.toShort())
            buffer.putShort(point.landmarkId.toShort())
            buffer.putFloat(point.normalizedX)
            buffer.putFloat(point.normalizedY)
            buffer.putFloat(point.confidence)
            buffer.putFloat(point.uncertainty)
            buffer.putFloat(point.visibility)
            buffer.putFloat(point.contactProbability)
            buffer.putFloat(point.temporalStability)
            buffer.putFloat(point.crossViewAgreement)
            buffer.putFloat(point.reprojectionResidualNormalized)
            buffer.putFloat(point.probabilitySigmaXNormalized)
            buffer.putFloat(point.probabilitySigmaYNormalized)
            buffer.putFloat(point.probabilityCalibrationScore)
            buffer.putFloat(point.segmentationConsistency)
            val flags =
                (if (point.predicted) 0x01 else 0) or
                    (if (point.segmentationEvidenceAvailable) 0x02 else 0)
            buffer.put(flags.toByte())
        }

        return buffer.array()
    }

    fun decode(
        bytes: ByteArray,
        timestampNanos: Long,
    ): Frame {
        require(timestampNanos > 0L)
        require(bytes.size >= HEADER_BYTES)

        val buffer = ByteBuffer.wrap(bytes)
            .order(ByteOrder.BIG_ENDIAN)

        val version = buffer.int
        require(
            version == VERSION ||
                version == VERSION_3 ||
                version == VERSION_2 ||
                version == LEGACY_VERSION,
        ) {
            "Unsupported dense landmark wire version"
        }

        val count = buffer.int
        require(count in 0..MAX_POINTS) {
            "Invalid dense landmark count"
        }

        val pointBytes = buffer.int
        val expectedPointBytes =
            when (version) {
                LEGACY_VERSION -> LEGACY_POINT_BYTES
                VERSION_2 -> POINT_BYTES_V2
                VERSION_3, VERSION -> POINT_BYTES
                else -> error("Unsupported dense landmark wire version")
            }

        require(pointBytes == expectedPointBytes) {
            "Invalid dense landmark point size"
        }

        val expectedHeaderBytes =
            if (version == VERSION) HEADER_BYTES_V4 else HEADER_BYTES
        require(
            bytes.size ==
                expectedHeaderBytes + count * expectedPointBytes,
        ) {
            "Invalid dense landmark payload size"
        }

        val imageWidth: Int
        val imageHeight: Int
        val rollingShutterSkewNanos: Long
        val frameDurationNanos: Long
        val exposureTimeNanos: Long
        val captureTimingConfidence: Float
        if (version == VERSION) {
            imageWidth = buffer.int
            imageHeight = buffer.int
            rollingShutterSkewNanos = buffer.long
            frameDurationNanos = buffer.long
            exposureTimeNanos = buffer.long
            captureTimingConfidence = buffer.float
        } else {
            // Older peers did not send frame dimensions or timing provenance. Unknown stays unknown.
            imageWidth = 0
            imageHeight = 0
            rollingShutterSkewNanos = 0L
            frameDurationNanos = 0L
            exposureTimeNanos = 0L
            captureTimingConfidence = 0.0f
        }

        val output = ArrayList<Landmark>(count)

        repeat(count) {
            val subjectId = buffer.short.toInt() and 0xFFFF
            val landmarkId = buffer.short.toInt() and 0xFFFF
            val normalizedX = buffer.float
            val normalizedY = buffer.float
            val confidence = buffer.float
            val uncertainty = buffer.float
            val visibility = buffer.float
            val contactProbability = buffer.float
            val temporalStability = buffer.float
            val crossViewAgreement = buffer.float
            val reprojectionResidualNormalized = buffer.float
            val probabilitySigmaXNormalized =
                if (version == LEGACY_VERSION) 0f else buffer.float
            val probabilitySigmaYNormalized =
                if (version == LEGACY_VERSION) 0f else buffer.float
            val probabilityCalibrationScore =
                if (version == LEGACY_VERSION) 0.0f else buffer.float
            // Earlier payload versions did not declare whether segmentation was measured.
            val segmentationConsistency =
                if (version >= VERSION_3) buffer.float else 0.0f
            val pointFlags = buffer.get().toInt() and 0xFF
            val point =
                Landmark(
                    subjectId = subjectId,
                    landmarkId = landmarkId,
                    normalizedX = normalizedX,
                    normalizedY = normalizedY,
                    confidence = confidence,
                    uncertainty = uncertainty,
                    visibility = visibility,
                    contactProbability = contactProbability,
                    temporalStability = temporalStability,
                    crossViewAgreement = crossViewAgreement,
                    reprojectionResidualNormalized = reprojectionResidualNormalized,
                    probabilitySigmaXNormalized = probabilitySigmaXNormalized,
                    probabilitySigmaYNormalized = probabilitySigmaYNormalized,
                    probabilityCalibrationScore = probabilityCalibrationScore,
                    segmentationConsistency = segmentationConsistency,
                    predicted =
                        if (version == VERSION) {
                            (pointFlags and 0x01) != 0
                        } else {
                            pointFlags != 0
                        },
                    segmentationEvidenceAvailable =
                        version == VERSION && (pointFlags and 0x02) != 0,
                ).sanitized()

            output += point
        }

        if (version == LEGACY_VERSION) {
            val legacyPaddingBytes =
                LEGACY_POINT_BYTES - 41
            if (legacyPaddingBytes > 0) {
                repeat(legacyPaddingBytes) {
                    if (buffer.hasRemaining()) {
                        buffer.get()
                    }
                }
            }
        }

        require(!buffer.hasRemaining()) {
            "Trailing dense landmark bytes"
        }

        return Frame(
            timestampNanos = timestampNanos,
            landmarks = output,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
            captureTimingConfidence = captureTimingConfidence,
        )
    }

    private fun Float.finiteOrZero(): Float =
        takeIf { it.isFinite() } ?: 0f

    private fun Float.finiteOrOne(): Float =
        takeIf { it.isFinite() } ?: 1f
}
