package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.FrameTelemetry
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * Lossless Camera2 RAW_SENSOR sidecar.
 *
 * The container is intentionally simple and self-describing. It stores the sensor buffer exactly
 * as delivered by the HAL together with the timestamp, exposure, ISO and row/pixel stride. No
 * demosaicing or tone mapping occurs here, so an offline renderer can perform its own sensor model.
 */
class RawSensorSidecarWriter(file: File) : AutoCloseable {
    private val output = DataOutputStream(
        BufferedOutputStream(FileOutputStream(file), 256 * 1024),
    )

    private var frames = 0L

    init {
        output.writeBytes("NOXRAW1")
        output.writeInt(1)
    }

    @Synchronized
    fun append(
        width: Int,
        height: Int,
        timestampNanos: Long,
        telemetry: FrameTelemetry?,
        rowStride: Int,
        pixelStride: Int,
        buffer: ByteBuffer,
    ) {
        val source = buffer.duplicate()
        val remaining = source.remaining()

        output.writeInt(width)
        output.writeInt(height)
        output.writeLong(timestampNanos)
        output.writeLong(telemetry?.exposureTimeNanos ?: 0L)
        output.writeInt(telemetry?.sensorSensitivityIso ?: 0)
        output.writeInt(rowStride)
        output.writeInt(pixelStride)
        output.writeInt(remaining)

        val chunk = ByteArray(256 * 1024)
        while (source.hasRemaining()) {
            val count = minOf(source.remaining(), chunk.size)
            source.get(chunk, 0, count)
            output.write(chunk, 0, count)
        }

        frames += 1L
        if ((frames and 0x0FL) == 0L) output.flush()
    }

    @Synchronized
    override fun close() {
        output.writeLong(frames)
        output.flush()
        output.close()
    }
}
