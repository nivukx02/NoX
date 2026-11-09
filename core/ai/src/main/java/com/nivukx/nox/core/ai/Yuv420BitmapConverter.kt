package com.nivukx.nox.core.ai

import android.graphics.Bitmap
import android.media.Image
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Bounded Camera2 YUV_420_888 -> ARGB_8888 adapter for MediaPipe Tasks.
 *
 * MediaPipe's Android vision tasks expose ARGB_8888 as the supported video input format. NoX keeps
 * Camera2 on YUV_420_888 for capture efficiency, then converts only the inference working-set
 * resolution. The full-resolution camera frame is never copied just to feed a hand detector.
 *
 * The bitmap is reused across calls at a stable inference resolution, which keeps allocator and
 * GC pressure bounded during long takes.
 */
class Yuv420BitmapConverter(
    private val maxDimension: Int = 960,
) {
    private var bitmap: Bitmap? = null
    private var width = 0
    private var height = 0

    @Synchronized
    fun convert(image: Image): Bitmap {
        val scale = min(
            1.0,
            maxDimension.toDouble() /
                max(image.width, image.height).coerceAtLeast(1),
        )

        val targetWidth =
            max(1, (image.width * scale).roundToInt())
        val targetHeight =
            max(1, (image.height * scale).roundToInt())

        if (
            bitmap == null ||
            width != targetWidth ||
            height != targetHeight ||
            bitmap?.isRecycled == true
        ) {
            bitmap?.recycle()
            bitmap = Bitmap.createBitmap(
                targetWidth,
                targetHeight,
                Bitmap.Config.ARGB_8888,
            )
            width = targetWidth
            height = targetHeight
        }

        val output = IntArray(
            targetWidth * targetHeight,
        )

        val planes = image.planes
        require(planes.size >= 3) {
            "Expected YUV_420_888 with three planes"
        }

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        for (dy in 0 until targetHeight) {
            val sourceY = min(
                image.height - 1,
                dy * image.height / targetHeight,
            )
            val chromaY = sourceY / 2

            for (dx in 0 until targetWidth) {
                val sourceX = min(
                    image.width - 1,
                    dx * image.width / targetWidth,
                )
                val chromaX = sourceX / 2

                val yIndex =
                    sourceY * yRowStride +
                        sourceX * yPixelStride
                val uIndex =
                    chromaY * uRowStride +
                        chromaX * uPixelStride
                val vIndex =
                    chromaY * vRowStride +
                        chromaX * vPixelStride

                if (
                    yIndex >= yBuffer.limit() ||
                    uIndex >= uBuffer.limit() ||
                    vIndex >= vBuffer.limit()
                ) {
                    continue
                }

                val y = yBuffer.get(yIndex).toInt() and 0xFF
                val u = (uBuffer.get(uIndex).toInt() and 0xFF) - 128
                val v = (vBuffer.get(vIndex).toInt() and 0xFF) - 128

                val c = (y - 16).coerceAtLeast(0)

                val red =
                    ((1.164 * c) + (1.596 * v))
                        .roundToInt()
                        .coerceIn(0, 255)
                val green =
                    ((1.164 * c) - (0.392 * u) - (0.813 * v))
                        .roundToInt()
                        .coerceIn(0, 255)
                val blue =
                    ((1.164 * c) + (2.017 * u))
                        .roundToInt()
                        .coerceIn(0, 255)

                output[
                    dy * targetWidth + dx
                ] =
                    (0xFF shl 24) or
                        (red shl 16) or
                        (green shl 8) or
                        blue
            }
        }

        bitmap!!.setPixels(
            output,
            0,
            targetWidth,
            0,
            0,
            targetWidth,
            targetHeight,
        )

        return bitmap!!
    }

    @Synchronized
    fun close() {
        bitmap?.recycle()
        bitmap = null
        width = 0
        height = 0
    }
}
