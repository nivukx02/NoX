package com.nivukx.nox.core.network

import java.util.zip.CRC32

/**
 * Bounded transport envelope for dense landmark frames that exceed one NoX control packet.
 *
 * The base DenseLandmarkWireCodec deliberately remains single-packet and now fails closed rather
 * than silently dropping lower-ranked landmarks. This layer preserves the full frame by splitting
 * it into deterministic sub-packets small enough for the existing transport safety limit.
 *
 * Each chunk is independently decodable and protected by CRC32. Reassembly requires:
 * - a single stream and frame sequence;
 * - exact chunk-count agreement;
 * - every chunk index exactly once;
 * - matching timestamps;
 * - successful per-chunk CRC validation;
 * - total-landmark uniqueness after reassembly.
 */
object DenseLandmarkChunkedWireCodec {

    enum class FrameWireVersion {
        LEGACY_V3,
        V4_TIMING_METADATA,
    }

    private const val DEFAULT_MAX_LANDMARKS_PER_CHUNK = 900
    /**
     * Maximum per-frame fragment count reserved by the sender's packet-sequence allocator.
     */
    const val MAX_CHUNKS = 16
    private const val MAX_STREAM_ID_BYTES = 96
    private const val NOX_HEADER_BYTES = 36
    private const val FRAGMENT_HEADER_BYTES = 28
    private const val MAX_PACKET_PAYLOAD =
        NoXWireProtocol.MAX_PACKET_BYTES -
            NOX_HEADER_BYTES -
            FRAGMENT_HEADER_BYTES -
            MAX_STREAM_ID_BYTES

    data class Chunk(
        val streamId: String,
        val frameSequence: Long,
        val timestampNanos: Long,
        val chunkIndex: Int,
        val chunkCount: Int,
        val payload: ByteArray,
        val crc32: Long,
    ) {
        init {
            require(streamId.isNotBlank())
            require(streamId.toByteArray(Charsets.UTF_8).size <= MAX_STREAM_ID_BYTES)
            require(frameSequence >= 0L)
            require(timestampNanos > 0L)
            require(chunkCount in 1..MAX_CHUNKS)
            require(chunkIndex in 0 until chunkCount)
            require(payload.isNotEmpty())
            require(payload.size <= MAX_PACKET_PAYLOAD)
            require(crc32 in 0L..0xFFFF_FFFFL)
        }
    }

    fun encode(
        frame: DenseLandmarkWireCodec.Frame,
        streamId: String,
        frameSequence: Long,
        maximumLandmarksPerChunk: Int =
            DEFAULT_MAX_LANDMARKS_PER_CHUNK,
        maximumPayloadBytes: Int =
            MAX_PACKET_PAYLOAD,
        wireVersion: FrameWireVersion = FrameWireVersion.V4_TIMING_METADATA,
    ): List<Chunk> {
        require(streamId.isNotBlank())
        require(
            streamId.toByteArray(Charsets.UTF_8).size <=
                MAX_STREAM_ID_BYTES,
        )
        require(frame.timestampNanos > 0L)
        require(frameSequence >= 0L)
        require(maximumLandmarksPerChunk in 32..900)
        require(
            maximumPayloadBytes in
                1024..MAX_PACKET_PAYLOAD,
        )

        val landmarks =
            frame.landmarks
                .map(DenseLandmarkWireCodec.Landmark::sanitized)

        require(
            landmarks.distinctBy {
                it.subjectId to it.landmarkId
            }.size == landmarks.size,
        ) {
            "Duplicate dense-landmark keys cannot be fragmented"
        }

        if (landmarks.isEmpty()) {
            val payload =
                when (wireVersion) {
                    FrameWireVersion.LEGACY_V3 ->
                        DenseLandmarkWireCodec.encodeV3Compatibility(frame)
                    FrameWireVersion.V4_TIMING_METADATA ->
                        DenseLandmarkWireCodec.encode(frame)
                }
            return listOf(
                Chunk(
                    streamId = streamId,
                    frameSequence = frameSequence,
                    timestampNanos = frame.timestampNanos,
                    chunkIndex = 0,
                    chunkCount = 1,
                    payload = payload,
                    crc32 = crc(payload),
                ),
            )
        }

        val maximumCountByBytes =
            (
                maximumPayloadBytes -
                    DenseLandmarkWireCodec.encodedSizeForLandmarkCount(0)
            ) /
                57

        require(maximumCountByBytes >= 1) {
            "Dense landmark transport budget cannot fit one landmark"
        }

        val countLimit =
            minOf(
                maximumLandmarksPerChunk,
                maximumCountByBytes,
            )

        val groups =
            ArrayList<List<DenseLandmarkWireCodec.Landmark>>(
                (
                    landmarks.size +
                        countLimit -
                        1
                    ) /
                    countLimit
            )

        var start = 0
        while (start < landmarks.size) {
            val end =
                minOf(
                    landmarks.size,
                    start + countLimit,
                )
            groups +=
                landmarks.subList(
                    start,
                    end,
                ).toList()
            start = end
        }

        groups.forEach { group ->
            require(
                DenseLandmarkWireCodec
                    .encodedSizeForLandmarkCount(
                        group.size,
                    ) <=
                    maximumPayloadBytes,
            ) {
                "Dense landmark chunk exceeds transport payload budget"
            }
        }

        require(groups.isNotEmpty())
        require(groups.size <= MAX_CHUNKS) {
            "Dense landmark frame requires too many transport chunks"
        }

        return groups.mapIndexed { index, group ->
            val chunkFrame = frame.copy(landmarks = group)
            val payload =
                when (wireVersion) {
                    FrameWireVersion.LEGACY_V3 ->
                        DenseLandmarkWireCodec.encodeV3Compatibility(chunkFrame)
                    FrameWireVersion.V4_TIMING_METADATA ->
                        DenseLandmarkWireCodec.encode(chunkFrame)
                }
            Chunk(
                streamId = streamId,
                frameSequence = frameSequence,
                timestampNanos = frame.timestampNanos,
                chunkIndex = index,
                chunkCount = groups.size,
                payload = payload,
                crc32 = crc(payload),
            )
        }
    }

    fun decode(
        chunks: List<Chunk>,
    ): DenseLandmarkWireCodec.Frame {
        require(chunks.isNotEmpty())

        val first = chunks.first()

        require(
            chunks.size == first.chunkCount,
        ) {
            "Incomplete dense landmark chunk set"
        }

        val expectedIndices =
            (0 until first.chunkCount).toSet()
        val actualIndices =
            chunks.map { it.chunkIndex }.toSet()

        require(actualIndices == expectedIndices) {
            "Dense landmark chunk indices are incomplete or duplicated"
        }

        chunks.forEach { chunk ->
            require(chunk.streamId == first.streamId)
            require(chunk.frameSequence == first.frameSequence)
            require(chunk.timestampNanos == first.timestampNanos)
            require(chunk.chunkCount == first.chunkCount)
            require(
                crc(chunk.payload) ==
                    chunk.crc32,
            )
        }

        val decodedFrames =
            chunks
                .sortedBy { it.chunkIndex }
                .map { chunk ->
                    DenseLandmarkWireCodec.decode(
                        bytes = chunk.payload,
                        timestampNanos = chunk.timestampNanos,
                    )
                }
        val metadata = decodedFrames.first()
        require(
            decodedFrames.all {
                it.imageWidth == metadata.imageWidth &&
                    it.imageHeight == metadata.imageHeight &&
                    it.rollingShutterSkewNanos == metadata.rollingShutterSkewNanos &&
                    it.frameDurationNanos == metadata.frameDurationNanos &&
                    it.exposureTimeNanos == metadata.exposureTimeNanos &&
                    it.captureTimingConfidence == metadata.captureTimingConfidence
            },
        ) {
            "Dense landmark chunks contain inconsistent frame timing metadata"
        }
        val landmarks = decodedFrames.flatMap { it.landmarks }

        require(
            landmarks.distinctBy {
                it.subjectId to it.landmarkId
            }.size == landmarks.size,
        ) {
            "Dense landmark reassembly contains duplicate keys"
        }

        return metadata.copy(
            timestampNanos = first.timestampNanos,
            landmarks = landmarks,
        )
    }

    private fun crc(payload: ByteArray): Long {
        val crc = CRC32()
        crc.update(payload)
        return crc.value
    }
}
