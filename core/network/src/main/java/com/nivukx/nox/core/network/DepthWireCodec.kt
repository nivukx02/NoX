package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.DepthSource
import com.nivukx.nox.core.model.DeviceId
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Deflater
import java.util.zip.Inflater

/**
 * Compact metric-depth transport for synchronized multi-phone dense fusion.
 *
 * Depth is decimated to a bounded raster before transmission. Values remain metric millimetres,
 * are compressed with DEFLATE, and carry the original sensor dimensions so the receiver can
 * correctly rescale camera intrinsics.
 */
object DepthWireCodec {
    private const val VERSION = 1
    private const val HEADER_BYTES = 40
    private const val MAX_WIDTH = 160
    private const val MAX_HEIGHT = 90
    private const val MAX_COMPRESSED_BYTES = 64 * 1024

    data class Frame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val width: Int,
        val height: Int,
        val depthMeters: FloatArray,
    ) {
        init {
            require(sourceWidth > 0)
            require(sourceHeight > 0)
            require(width in 1..MAX_WIDTH)
            require(height in 1..MAX_HEIGHT)
            require(depthMeters.size == width * height)
        }
    }

    fun encode(
        deviceId: DeviceId,
        frame: DepthFrame,
    ): ByteArray? {
        val stride = maxOf(
            1,
            kotlin.math.ceil(
                maxOf(
                    frame.width.toDouble() / MAX_WIDTH.toDouble(),
                    frame.height.toDouble() / MAX_HEIGHT.toDouble(),
                ),
            ).toInt(),
        )

        val wireWidth =
            ((frame.width + stride - 1) / stride)
                .coerceAtMost(MAX_WIDTH)

        val wireHeight =
            ((frame.height + stride - 1) / stride)
                .coerceAtMost(MAX_HEIGHT)

        val raw = ByteBuffer
            .allocate(wireWidth * wireHeight * 2)
            .order(ByteOrder.BIG_ENDIAN)

        for (y in 0 until wireHeight) {
            val sourceY =
                (y * stride)
                    .coerceAtMost(frame.height - 1)

            for (x in 0 until wireWidth) {
                val sourceX =
                    (x * stride)
                        .coerceAtMost(frame.width - 1)

                val depth =
                    frame.depthMeters[
                        sourceY * frame.width + sourceX
                    ]

                val millimetres =
                    if (
                        depth.isFinite() &&
                        depth > 0f &&
                        depth < 65.535f
                    ) {
                        (depth * 1000f)
                            .toInt()
                            .coerceIn(1, 65_535)
                    } else {
                        0
                    }

                raw.putShort(
                    millimetres.toShort(),
                )
            }
        }

        val compressed =
            deflate(raw.array())

        if (
            compressed.size >
            MAX_COMPRESSED_BYTES
        ) {
            return null
        }

        val deviceBytes =
            deviceId.value
                .toByteArray(Charsets.UTF_8)
                .take(64)
                .toByteArray()

        val buffer =
            ByteBuffer.allocate(
                HEADER_BYTES +
                    deviceBytes.size +
                    compressed.size,
            ).order(ByteOrder.BIG_ENDIAN)

        buffer.putInt(VERSION)
        buffer.putInt(deviceBytes.size)
        buffer.putLong(frame.timestampNanos)
        buffer.putInt(frame.width)
        buffer.putInt(frame.height)
        buffer.putInt(wireWidth)
        buffer.putInt(wireHeight)
        buffer.putInt(stride)
        buffer.putInt(compressed.size)
        buffer.put(deviceBytes)
        buffer.put(compressed)

        return buffer.array()
    }

    fun decode(
        bytes: ByteArray,
    ): Frame {
        require(bytes.size >= HEADER_BYTES)

        val buffer =
            ByteBuffer.wrap(bytes)
                .order(ByteOrder.BIG_ENDIAN)

        require(
            buffer.int == VERSION,
        ) {
            "Unsupported depth payload version"
        }

        val deviceLength = buffer.int
        require(deviceLength in 1..64)

        val timestamp = buffer.long
        val sourceWidth = buffer.int
        val sourceHeight = buffer.int
        val width = buffer.int
        val height = buffer.int
        val stride = buffer.int
        val compressedLength = buffer.int

        require(sourceWidth > 0)
        require(sourceHeight > 0)
        require(width in 1..MAX_WIDTH)
        require(height in 1..MAX_HEIGHT)
        require(stride >= 1)
        require(compressedLength > 0)

        val remainingAfterHeader =
            buffer.remaining()

        require(
            deviceLength +
                compressedLength ==
                remainingAfterHeader,
        )

        val deviceBytes =
            ByteArray(deviceLength)

        buffer.get(deviceBytes)

        val compressed =
            ByteArray(compressedLength)

        buffer.get(compressed)

        val expectedBytes =
            width * height * 2

        val decompressed =
            inflate(
                compressed,
                expectedBytes,
            )

        require(
            decompressed.size ==
                expectedBytes,
        )

        val depthBuffer =
            ByteBuffer.wrap(decompressed)
                .order(ByteOrder.BIG_ENDIAN)

        val depth =
            FloatArray(width * height)

        repeat(depth.size) { index ->
            val millimetres =
                depthBuffer.short.toInt() and 0xFFFF

            depth[index] =
                if (millimetres == 0) {
                    0f
                } else {
                    millimetres / 1000f
                }
        }

        return Frame(
            deviceId = DeviceId(
                deviceBytes.toString(
                    Charsets.UTF_8,
                ),
            ),
            timestampNanos = timestamp,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            width = width,
            height = height,
            depthMeters = depth,
        )
    }

    fun toDepthFrame(
        frame: Frame,
    ): DepthFrame =
        DepthFrame(
            timestampNanos = frame.timestampNanos,
            source = DepthSource.NETWORK_COMPRESSED,
            width = frame.width,
            height = frame.height,
            depthMeters = frame.depthMeters,
        )

    private fun deflate(
        input: ByteArray,
    ): ByteArray {
        val compressor =
            Deflater(
                Deflater.BEST_SPEED,
                false,
            )

        return try {
            compressor.setInput(input)
            compressor.finish()

            val output =
                ByteArrayOutputStream(
                    input.size / 2,
                )

            val buffer =
                ByteArray(16 * 1024)

            while (!compressor.finished()) {
                val count =
                    compressor.deflate(buffer)

                if (count <= 0) break

                output.write(
                    buffer,
                    0,
                    count,
                )
            }

            output.toByteArray()
        } finally {
            compressor.end()
        }
    }

    private fun inflate(
        input: ByteArray,
        expectedBytes: Int,
    ): ByteArray {
        val inflater = Inflater(false)

        return try {
            inflater.setInput(input)

            val output =
                ByteArrayOutputStream(
                    expectedBytes,
                )

            val buffer =
                ByteArray(16 * 1024)

            while (!inflater.finished()) {
                val count =
                    inflater.inflate(buffer)

                if (count <= 0) {
                    if (
                        inflater.needsDictionary() ||
                        inflater.needsInput()
                    ) {
                        break
                    }
                } else {
                    output.write(
                        buffer,
                        0,
                        count,
                    )
                }
            }

            output.toByteArray()
        } catch (error: Throwable) {
            throw IllegalArgumentException(
                "Depth payload decompression failed",
                error,
            )
        } finally {
            inflater.end()
        }
    }
}
