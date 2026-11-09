package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.MotionSampleV2
import com.nivukx.nox.core.model.MotionSessionHeaderV2
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.SessionId
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.TopologyKind
import com.nivukx.nox.core.model.Vec3
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.CRC32
import kotlin.math.max

/**
 * Versioned binary motion-only export.
 *
 * The payload contains timestamps, transforms, velocities, confidence and uncertainty. It never
 * stores RGB, depth images, video frames or raw sensor buffers.
 */
object MotionExportCodecV2 {

    private const val MAGIC = 0x4E4D5832
    private const val INDEX_MAGIC = 0x4E494458
    private const val VERSION = 2
    private const val MAX_FRAME_BYTES = 16 * 1024 * 1024
    private const val MAX_JOINTS_PER_FRAME = 1024
    private const val MAX_FRAME_COUNT = 2_000_000

    data class Frame(
        val timestampNanos: Long,
        val samples: List<MotionSampleV2>,
    )

    data class ChunkIndexEntry(
        val frameIndex: Int,
        val fileOffset: Long,
        val payloadBytes: Int,
        val timestampNanos: Long,
    )

    data class Document(
        val header: MotionSessionHeaderV2,
        val frames: List<Frame>,
        val index: List<ChunkIndexEntry>,
    )

    data class Summary(
        val frameCount: Int,
        val jointSamples: Long,
        val firstTimestampNanos: Long,
        val lastTimestampNanos: Long,
        val durationNanos: Long,
    )

    fun write(
        output: OutputStream,
        header: MotionSessionHeaderV2,
        frames: Sequence<Frame>,
    ): Summary {
        require(header.motionOnly) {
            "MotionExportCodecV2 only accepts motionOnly sessions"
        }

        val counter =
            output as? CountingOutputStream
                ?: CountingOutputStream(output)

        DataOutputStream(
            BufferedOutputStream(counter, 256 * 1024),
        ).use { data ->
            data.writeInt(MAGIC)
            data.writeInt(VERSION)
            writeHeader(data, header)

            val index = ArrayList<ChunkIndexEntry>()
            var frameCount = 0
            var jointSamples = 0L
            var firstTimestamp = Long.MIN_VALUE
            var lastTimestamp = Long.MIN_VALUE

            frames.forEach { frame ->
                require(frameCount < MAX_FRAME_COUNT) {
                    "Frame count exceeds export limit"
                }
                require(frame.samples.size <= MAX_JOINTS_PER_FRAME) {
                    "Frame contains too many joint samples"
                }

                val payload = encodeFrame(frame)
                require(payload.size <= MAX_FRAME_BYTES) {
                    "Frame chunk exceeds configured maximum size"
                }

                data.flush()
                val chunkOffset = counter.bytesWritten()
                val crc = CRC32().apply { update(payload) }

                data.writeInt(payload.size)
                data.writeInt(crc.value.toInt())
                data.write(payload)

                index += ChunkIndexEntry(
                    frameIndex = frameCount,
                    fileOffset = chunkOffset,
                    payloadBytes = payload.size,
                    timestampNanos = frame.timestampNanos,
                )

                if (firstTimestamp == Long.MIN_VALUE) {
                    firstTimestamp = frame.timestampNanos
                }

                lastTimestamp = frame.timestampNanos
                jointSamples += frame.samples.size.toLong()
                frameCount += 1
            }

            data.writeInt(INDEX_MAGIC)
            data.writeInt(index.size)

            index.forEach { entry ->
                data.writeInt(entry.frameIndex)
                data.writeLong(entry.fileOffset)
                data.writeInt(entry.payloadBytes)
                data.writeLong(entry.timestampNanos)
            }

            data.flush()

            return Summary(
                frameCount = frameCount,
                jointSamples = jointSamples,
                firstTimestampNanos =
                    if (firstTimestamp == Long.MIN_VALUE) 0L
                    else firstTimestamp,
                lastTimestampNanos =
                    if (lastTimestamp == Long.MIN_VALUE) 0L
                    else lastTimestamp,
                durationNanos =
                    if (
                        firstTimestamp == Long.MIN_VALUE ||
                        lastTimestamp == Long.MIN_VALUE
                    ) {
                        0L
                    } else {
                        (lastTimestamp - firstTimestamp)
                            .coerceAtLeast(0L)
                    },
            )
        }
    }

    fun writeFile(
        file: File,
        header: MotionSessionHeaderV2,
        frames: Sequence<Frame>,
    ): Summary =
        file.outputStream().use { output ->
            write(
                output = output,
                header = header,
                frames = frames,
            )
        }

    fun read(
        input: InputStream,
    ): Document {
        DataInputStream(
            BufferedInputStream(input, 256 * 1024),
        ).use { data ->
            require(data.readInt() == MAGIC) {
                "Invalid NMX2 magic"
            }
            require(data.readInt() == VERSION) {
                "Unsupported NMX2 version"
            }

            val header = readHeader(data)
            val frames = ArrayList<Frame>()
            val index = ArrayList<ChunkIndexEntry>()

            while (true) {
                val marker = try {
                    data.readInt()
                } catch (_: EOFException) {
                    break
                }

                if (marker == INDEX_MAGIC) {
                    val count = data.readInt()
                    require(count in 0..MAX_FRAME_COUNT)

                    repeat(count) {
                        index += ChunkIndexEntry(
                            frameIndex = data.readInt(),
                            fileOffset = data.readLong(),
                            payloadBytes = data.readInt(),
                            timestampNanos = data.readLong(),
                        )
                    }
                    break
                }

                require(marker in 0..MAX_FRAME_BYTES) {
                    "Invalid NMX2 payload length"
                }

                val expectedCrc = data.readInt()
                val payload = ByteArray(marker)
                data.readFully(payload)

                val actualCrc = CRC32().apply {
                    update(payload)
                }.value.toInt()

                require(actualCrc == expectedCrc) {
                    "NMX2 frame CRC mismatch"
                }

                frames += decodeFrame(payload)
            }

            return Document(
                header = header,
                frames = frames,
                index = index,
            )
        }
    }

    fun readFile(file: File): Document =
        file.inputStream().use(::read)

    fun openStreamingReader(
        file: File,
    ): StreamingReader =
        StreamingReader(
            DataInputStream(
                BufferedInputStream(
                    file.inputStream(),
                    256 * 1024,
                ),
            ),
        )

    class StreamingReader internal constructor(
        private val input: DataInputStream,
    ) : AutoCloseable {

        val header: MotionSessionHeaderV2

        private var finished = false
        private var frameIndex = 0
        private val discoveredIndex = ArrayList<ChunkIndexEntry>()

        init {
            require(input.readInt() == MAGIC) {
                "Invalid NMX2 magic"
            }
            require(input.readInt() == VERSION) {
                "Unsupported NMX2 version"
            }
            header = readHeader(input)
        }

        fun next(): Frame? {
            if (finished) return null

            val marker = runCatching {
                input.readInt()
            }.getOrElse {
                finished = true
                return null
            }

            if (marker == INDEX_MAGIC) {
                val count = input.readInt()
                require(count in 0..MAX_FRAME_COUNT)

                repeat(count) {
                    discoveredIndex += ChunkIndexEntry(
                        frameIndex = input.readInt(),
                        fileOffset = input.readLong(),
                        payloadBytes = input.readInt(),
                        timestampNanos = input.readLong(),
                    )
                }

                finished = true
                return null
            }

            require(marker in 0..MAX_FRAME_BYTES) {
                "Invalid NMX2 payload length"
            }

            val expectedCrc = input.readInt()
            val payload = ByteArray(marker)
            input.readFully(payload)

            val actualCrc =
                CRC32().apply {
                    update(payload)
                }.value.toInt()

            require(actualCrc == expectedCrc) {
                "NMX2 frame CRC mismatch"
            }

            val frame = decodeFrame(payload)
            frameIndex += 1
            return frame
        }

        fun discoveredIndex(): List<ChunkIndexEntry> =
            discoveredIndex.toList()

        fun drain(
            consumer: (Frame) -> Unit,
        ): Int {
            var count = 0
            while (true) {
                val frame = next() ?: break
                consumer(frame)
                count += 1
            }
            return count
        }

        override fun close() {
            input.close()
            finished = true
        }
    }

    private fun writeHeader(
        data: DataOutputStream,
        header: MotionSessionHeaderV2,
    ) {
        data.writeInt(header.schema)
        data.writeUTF(header.sessionId.value)
        data.writeUTF(header.mode.name)
        data.writeUTF(header.subjectType.name)
        data.writeUTF(header.topologyKind.name)
        data.writeLong(header.createdAtNanos)
        data.writeInt(header.targetFps)
        data.writeUTF(header.calibrationVersion)
        data.writeUTF(header.modelProfile)
        data.writeBoolean(header.motionOnly)
    }

    private fun readHeader(
        data: DataInputStream,
    ): MotionSessionHeaderV2 =
        MotionSessionHeaderV2(
            schema = data.readInt(),
            sessionId = SessionId(data.readUTF()),
            mode = MotionCaptureMode.valueOf(data.readUTF()),
            subjectType =
                SubjectType.valueOf(data.readUTF()),
            topologyKind =
                TopologyKind.valueOf(data.readUTF()),
            createdAtNanos = data.readLong(),
            targetFps = data.readInt(),
            calibrationVersion = data.readUTF(),
            modelProfile = data.readUTF(),
            motionOnly = data.readBoolean(),
        )

    private fun encodeFrame(
        frame: Frame,
    ): ByteArray {
        val output = java.io.ByteArrayOutputStream(
            max(128, frame.samples.size * 64),
        )

        DataOutputStream(output).use { data ->
            data.writeLong(frame.timestampNanos)
            data.writeInt(frame.samples.size)

            frame.samples.forEach { sample ->
                data.writeLong(sample.timestampNanos)
                data.writeLong(sample.trackId)
                data.writeInt(sample.nodeId)
                writeVec3(data, sample.position)
                writeQuat(data, sample.rotation)
                writeVec3(data, sample.linearVelocity)
                writeVec3(data, sample.angularVelocity)
                data.writeShort(quantizeUnit(sample.confidence))
                data.writeInt(
                    quantizeUncertainty(
                        sample.uncertaintyMeters,
                    ),
                )
                data.writeBoolean(sample.predicted)
            }
        }

        return output.toByteArray()
    }

    private fun decodeFrame(
        payload: ByteArray,
    ): Frame {
        DataInputStream(
            payload.inputStream(),
        ).use { data ->
            val timestamp = data.readLong()
            val count = data.readInt()

            require(count in 0..MAX_JOINTS_PER_FRAME)

            val samples = ArrayList<MotionSampleV2>(count)

            repeat(count) {
                samples += MotionSampleV2(
                    timestampNanos = data.readLong(),
                    trackId = data.readLong(),
                    nodeId = data.readInt(),
                    position = readVec3(data),
                    rotation = readQuat(data),
                    linearVelocity = readVec3(data),
                    angularVelocity = readVec3(data),
                    confidence =
                        dequantizeUnit(
                            data.readUnsignedShort(),
                        ),
                    uncertaintyMeters =
                        data.readInt() / 10000f,
                    predicted = data.readBoolean(),
                )
            }

            return Frame(
                timestampNanos = timestamp,
                samples = samples,
            )
        }
    }

    private fun writeVec3(
        data: DataOutputStream,
        value: Vec3,
    ) {
        data.writeFloat(value.x.toFloat())
        data.writeFloat(value.y.toFloat())
        data.writeFloat(value.z.toFloat())
    }

    private fun readVec3(
        data: DataInputStream,
    ): Vec3 =
        Vec3(
            data.readFloat().toDouble(),
            data.readFloat().toDouble(),
            data.readFloat().toDouble(),
        )

    private fun writeQuat(
        data: DataOutputStream,
        value: Quat,
    ) {
        val q = value.normalized()
        data.writeShort(quantizeSigned(q.w).toInt())
        data.writeShort(quantizeSigned(q.x).toInt())
        data.writeShort(quantizeSigned(q.y).toInt())
        data.writeShort(quantizeSigned(q.z).toInt())
    }

    private fun readQuat(
        data: DataInputStream,
    ): Quat =
        Quat(
            dequantizeSigned(data.readShort()),
            dequantizeSigned(data.readShort()),
            dequantizeSigned(data.readShort()),
            dequantizeSigned(data.readShort()),
        ).normalized()

    private fun quantizeUnit(value: Float): Int =
        (
            value.coerceIn(0f, 1f) * 65535f
        ).toInt()

    private fun dequantizeUnit(value: Int): Float =
        value.coerceIn(0, 65535) / 65535f

    private fun quantizeSigned(value: Double): Short =
        (
            value.coerceIn(-1.0, 1.0) * 32767.0
        ).toInt().toShort()

    private fun dequantizeSigned(value: Short): Double =
        value.toDouble() / 32767.0

    private fun quantizeUncertainty(meters: Float): Int =
        (
            meters.coerceIn(0f, 6.5535f) * 10000f
        ).toInt()

    private class CountingOutputStream(
        private val delegate: OutputStream,
    ) : OutputStream() {

        private var count = 0L

        fun bytesWritten(): Long = count

        override fun write(value: Int) {
            delegate.write(value)
            count += 1L
        }

        override fun write(
            buffer: ByteArray,
            offset: Int,
            length: Int,
        ) {
            delegate.write(buffer, offset, length)
            count += length
        }

        override fun flush() = delegate.flush()

        override fun close() = delegate.close()
    }
}
