package com.nivukx.nox.core.ai

import android.graphics.ImageFormat
import android.media.Image
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Sampled optical evidence evaluator for camera frames.
 *
 * It computes inexpensive, deterministic proxies from the luminance plane:
 *  - normalized luminance mean and variance,
 *  - highlight/shadow clipping,
 *  - gradient energy as a focus/sharpness proxy,
 *  - local contrast,
 *  - sensor readout completeness.
 *
 * The engine intentionally uses proxies rather than pretending to measure laboratory MTF from a
 * moving human scene. It is suitable for real-time gating and evidence weighting, not optical-bench
 * certification.
 */
class OpticalEvidenceQualityEngine(
    private val sampleGridWidth: Int = 48,
    private val sampleGridHeight: Int = 27,
) {

    data class Result(
        val timestampNanos: Long,
        val sampledPixels: Int,
        val luminanceMean: Double,
        val luminanceStdDev: Double,
        val shadowClippingRatio: Double,
        val highlightClippingRatio: Double,
        val gradientEnergy: Double,
        val localContrast: Double,
        val exposureQuality: Double,
        val sharpnessQuality: Double,
        /** Timing/scene-motion-derived optical integration blur risk, separate from static softness. */
        val motionBlurRisk: Double = 0.0,
        val contrastQuality: Double,
        val dynamicRangeQuality: Double,
        val signalQuality: Double,
        val warnings: List<String>,
    )

    fun evaluate(
        image: Image,
        frameDurationNanos: Long = 0L,
        exposureTimeNanos: Long = 0L,
        cameraMotionRisk: Double = 0.0,
        captureTimingConfidence: Double = 1.0,
        rollingShutterFraction: Double = 0.0,
    ): Result {
        require(frameDurationNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        require(cameraMotionRisk in 0.0..1.0)
        require(captureTimingConfidence in 0.0..1.0)
        require(rollingShutterFraction.isFinite())

        require(image.format == ImageFormat.YUV_420_888) {
            "OpticalEvidenceQualityEngine requires YUV_420_888"
        }

        val plane = image.planes.firstOrNull()
            ?: return empty(image.timestamp)

        val width = image.width.coerceAtLeast(1)
        val height = image.height.coerceAtLeast(1)
        val stepX = max(1, width / sampleGridWidth)
        val stepY = max(1, height / sampleGridHeight)

        val values = FloatArray(
            ((width + stepX - 1) / stepX) *
                ((height + stepY - 1) / stepY),
        )

        val buffer = plane.buffer.duplicate()
        val rowStride = plane.rowStride.coerceAtLeast(1)
        val pixelStride = plane.pixelStride.coerceAtLeast(1)

        var count = 0
        var y = 0
        while (y < height) {
            val rowBase = y * rowStride
            var x = 0
            while (x < width) {
                val index = rowBase + x * pixelStride
                if (index in 0 until buffer.limit()) {
                    values[count++] =
                        readLuma(buffer, index) / 255.0f
                }
                x += stepX
            }
            y += stepY
        }

        if (count == 0) {
            return empty(image.timestamp)
        }

        val mean =
            values.copyOf(count)
                .average()

        var squared = 0.0
        var below = 0
        var above = 0
        var gradientSum = 0.0
        var contrastSum = 0.0
        var comparisons = 0

        for (index in 0 until count) {
            val value = values[index].toDouble()
            val delta = value - mean
            squared += delta * delta
            if (value <= 0.02) below++
            if (value >= 0.98) above++
        }

        val cols = max(
            1,
            (width + stepX - 1) / stepX,
        )
        val rows = max(
            1,
            (height + stepY - 1) / stepY,
        )

        fun sampleAt(
            sx: Int,
            sy: Int,
        ): Double? {
            if (sx !in 0 until cols || sy !in 0 until rows) return null
            val i = sy * cols + sx
            return if (i < count) values[i].toDouble() else null
        }

        for (sy in 0 until rows) {
            for (sx in 0 until cols) {
                val center =
                    sampleAt(sx, sy) ?: continue
                val right =
                    sampleAt(sx + 1, sy)
                val down =
                    sampleAt(sx, sy + 1)

                if (right != null) {
                    val delta = abs(center - right)
                    gradientSum += delta * delta
                    comparisons++
                    contrastSum += delta
                }
                if (down != null) {
                    val delta = abs(center - down)
                    gradientSum += delta * delta
                    comparisons++
                    contrastSum += delta
                }
            }
        }

        val stdDev = sqrt(
            squared /
                count.coerceAtLeast(1),
        )

        val gradientEnergy =
            if (comparisons == 0) {
                0.0
            } else {
                sqrt(
                    gradientSum /
                        comparisons.toDouble(),
                )
            }

        val localContrast =
            if (comparisons == 0) {
                0.0
            } else {
                (
                    contrastSum /
                        comparisons.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        val shadowRatio =
            below.toDouble() /
                count.toDouble()

        val highlightRatio =
            above.toDouble() /
                count.toDouble()

        val exposureQuality =
            (
                1.0 -
                    (
                        abs(mean - 0.50) / 0.50 * 0.65 +
                            shadowRatio * 0.20 +
                            highlightRatio * 0.15
                        )
                ).coerceIn(0.0, 1.0)

        val sharpnessQuality =
            sigmoid(
                (gradientEnergy - 0.026) * 95.0,
            )

        val contrastQuality =
            sigmoid(
                (localContrast - 0.020) * 120.0,
            )

        val dynamicRangeQuality =
            (
                1.0 -
                    (shadowRatio + highlightRatio)
                    ).coerceIn(0.0, 1.0)

        val signalQuality =
            (
                exposureQuality * 0.30 +
                    sharpnessQuality * 0.32 +
                    contrastQuality * 0.22 +
                    dynamicRangeQuality * 0.16
                ).coerceIn(0.0, 1.0)

        val exposureRatio =
            if (frameDurationNanos > 0L) {
                (
                    exposureTimeNanos.toDouble() /
                        frameDurationNanos.toDouble()
                    ).coerceIn(0.0, 1.0)
            } else {
                0.0
            }

        val exposureMotionRisk =
            when {
                frameDurationNanos <= 0L ||
                    exposureTimeNanos <= 0L -> 0.0
                exposureRatio <= 0.20 -> 0.0
                exposureRatio >= 0.85 -> 1.0
                else ->
                    (
                        exposureRatio - 0.20
                    ) / 0.65
            }

        val readoutMotionRisk =
            (
                abs(rollingShutterFraction) *
                    0.72
            ).coerceIn(0.0, 1.0)

        val timingRisk =
            (
                1.0 -
                    captureTimingConfidence
            ).coerceIn(0.0, 1.0)

        val motionBlurRisk =
            (
                exposureMotionRisk * 0.52 +
                    cameraMotionRisk * 0.28 +
                    readoutMotionRisk * 0.14 +
                    timingRisk * 0.06
                ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (shadowRatio > 0.20) add("OPTICAL_SHADOW_CLIPPING_HIGH")
            if (highlightRatio > 0.12) add("OPTICAL_HIGHLIGHT_CLIPPING_HIGH")
            if (sharpnessQuality < 0.35) add("OPTICAL_SHARPNESS_LOW")
            if (contrastQuality < 0.35) add("OPTICAL_LOCAL_CONTRAST_LOW")
            if (signalQuality < 0.30) add("OPTICAL_SIGNAL_QUALITY_LOW")
            if (motionBlurRisk > 0.72) add("OPTICAL_MOTION_BLUR_RISK_HIGH")
            if (motionBlurRisk > 0.45) add("OPTICAL_MOTION_BLUR_RISK_ELEVATED")
        }

        return Result(
            timestampNanos = image.timestamp,
            sampledPixels = count,
            luminanceMean = mean,
            luminanceStdDev = stdDev,
            shadowClippingRatio = shadowRatio,
            highlightClippingRatio = highlightRatio,
            gradientEnergy = gradientEnergy,
            localContrast = localContrast,
            exposureQuality = exposureQuality,
            sharpnessQuality = sharpnessQuality,
            motionBlurRisk = motionBlurRisk,
            contrastQuality = contrastQuality,
            dynamicRangeQuality = dynamicRangeQuality,
            signalQuality = signalQuality,
            warnings = warnings,
        )
    }

    private fun readLuma(
        buffer: ByteBuffer,
        index: Int,
    ): Int =
        buffer.get(index).toInt() and 0xFF

    private fun sigmoid(value: Double): Double =
        1.0 /
            (
                1.0 +
                    kotlin.math.exp(
                        -value.coerceIn(-20.0, 20.0),
                    )
                )

    private fun empty(timestampNanos: Long): Result =
        Result(
            timestampNanos = timestampNanos,
            sampledPixels = 0,
            luminanceMean = 0.0,
            luminanceStdDev = 0.0,
            shadowClippingRatio = 1.0,
            highlightClippingRatio = 1.0,
            gradientEnergy = 0.0,
            localContrast = 0.0,
            exposureQuality = 0.0,
            sharpnessQuality = 0.0,
            motionBlurRisk = 1.0,
            contrastQuality = 0.0,
            dynamicRangeQuality = 0.0,
            signalQuality = 0.0,
            warnings = listOf("OPTICAL_SIGNAL_SAMPLE_UNAVAILABLE"),
        )
}
