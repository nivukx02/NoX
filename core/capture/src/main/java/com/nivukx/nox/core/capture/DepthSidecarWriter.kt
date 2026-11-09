package com.nivukx.nox.core.capture

import android.media.Image
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * Lossless Camera2 depth sidecar.
 *
 * DEPTH16 is retained as a raw plane instead of being converted into an 8-bit visualization.
 * This preserves millimeter precision and leaves confidence/depth fusion to the reconstruction
 * layer, where calibrated camera geometry is available.
 */
class DepthSidecarWriter(file: File) : AutoCloseable {
    private val output = DataOutputStream(
        BufferedOutputStream(FileOutputStream(file), 256 * 1024),
    )

    private var frames = 0L

    init {
        output.writeBytes("NOXDEP1")
        output.writeInt(1)
    }

    @Synchronized
    fun append(image: Image) {
        val plane = image.planes.firstOrNull() ?: return
        val buffer = plane.buffer.duplicate()
        val remaining = buffer.remaining()

        output.writeInt(image.width)
        output.writeInt(image.height)
        output.writeLong(image.timestamp)
        output.writeInt(plane.rowStride)
        output.writeInt(plane.pixelStride)
        output.writeInt(remaining)

        val chunk = ByteArray(128 * 1024)
        while (buffer.hasRemaining()) {
            val count = minOf(buffer.remaining(), chunk.size)
            buffer.get(chunk, 0, count)
            output.write(chunk, 0, count)
        }

        frames += 1L
        if ((frames and 0x1FL) == 0L) output.flush()
    }

    @Synchronized
    override fun close() {
        output.writeLong(frames)
        output.flush()
        output.close()
    }
}
