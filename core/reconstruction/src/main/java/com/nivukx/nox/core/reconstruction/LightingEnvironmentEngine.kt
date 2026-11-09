package com.nivukx.nox.core.reconstruction

import android.media.Image
import com.nivukx.nox.core.model.EnvironmentLighting
import com.nivukx.nox.core.model.FrameTelemetry
import com.nivukx.nox.core.model.LightingEstimate
import com.nivukx.nox.core.model.Vec3
import java.nio.ByteBuffer
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Camera-derived illumination estimator.
 *
 * The estimator uses the luminance/chrominance planes and actual exposure/ISO telemetry. It does
 * not claim physical lux accuracy without a device-specific photometric calibration factor. The
 * output is therefore an estimated illumination signal with explicit confidence rather than a
 * fabricated "AI light score".
 *
 * The spatial gradient is a useful directional-light prior for relighting and shadow-aware
 * reconstruction. A production capture can replace the default photometric scale with a measured
 * calibration constant without changing the downstream model.
 */
class LightingEnvironmentEngine(
    private val photometricScaleLux: Double = 1000.0,
) {

    data class Config(
        val sampleStride: Int = 8,
        val minimumConfidenceSamples: Int = 32,
    )

    private val config = Config()

    fun analyze(
        image: Image,
        telemetry: FrameTelemetry?,
        timestampNanos: Long,
        calibration:
            com.nivukx.nox.core.model.PhotometricCalibration? = null,
    ): LightingEstimate {
        val stats = sample(image)
        if (stats.samples < config.minimumConfidenceSamples) {
            return LightingEstimate(
                timestampNanos = timestampNanos,
                illuminanceLux = 0.0,
                colorTemperatureKelvin = 0.0,
                exposureCompensationEv = 0.0,
                ambientLuminance = 0.0,
                directionalLight = Vec3(0.0, 0.0, 1.0),
                directionalIntensity = 0.0,
                confidence = 0.0,
                dynamicRangeStops = 0.0,
            )
        }

        val y =
            (stats.meanY / 255.0)
                .coerceIn(0.0, 1.0)

        val calibratedRadiance =
            normalizeRadiance(
                normalizedLuminance = y,
                telemetry = telemetry,
                calibration = calibration,
            )

        val exposureSeconds = (
            telemetry?.exposureTimeNanos ?: 10_000_000L
        ).coerceAtLeast(1L) / 1e9
        val iso = (telemetry?.sensorSensitivityIso ?: 100)
            .coerceAtLeast(1)

        // Relative photometric normalization. Device-specific calibration should replace
        // photometricScaleLux for physically measured lux.
        val exposureNormalized =
            calibratedRadiance *
                (iso / 100.0) *
                (0.010 / exposureSeconds)
                    .coerceIn(0.01, 100.0)
        val lux = (
            exposureNormalized.pow(0.72) *
                photometricScaleLux
        ).coerceIn(0.01, 1_000_000.0)

        val rgb = yuvToRgb(
            stats.meanY,
            stats.meanU,
            stats.meanV,
        )
        val cct = estimateColorTemperature(rgb)

        val directional = Vec3(
            -stats.gradientX,
            -stats.gradientY,
            1.0,
        ).normalized()

        val gradientMagnitude = sqrt(
            stats.gradientX * stats.gradientX +
                stats.gradientY * stats.gradientY,
        ).coerceIn(0.0, 1.0)

        val dynamicRange = estimateDynamicRange(stats.minY, stats.maxY)

        val exposureEv = if (exposureSeconds > 0.0) {
            kotlin.math.log2((iso / 100.0).coerceAtLeast(1e-6) / exposureSeconds)
        } else {
            0.0
        }

        val confidence = (
            0.35 * (1.0 - stats.varianceY.coerceIn(0.0, 1.0)) +
                0.25 * (stats.samples / 4096.0).coerceIn(0.0, 1.0) +
                0.20 * (1.0 - stats.clippedRatio) +
                0.20 * (1.0 - stats.crushedRatio)
            ).coerceIn(0.0, 1.0)

        return LightingEstimate(
            timestampNanos = timestampNanos,
            illuminanceLux = lux,
            colorTemperatureKelvin = cct,
            exposureCompensationEv = exposureEv,
            ambientLuminance = y,
            directionalLight = directional,
            directionalIntensity = gradientMagnitude,
            confidence = confidence,
            dynamicRangeStops = dynamicRange,
        )
    }

    /**
     * Converts the observed normalized luminance into a device-calibrated relative radiance signal.
     *
     * When a measured calibration is available, black/white response, exposure/ISO, and sensor
     * temperature are compensated. The result remains a relative reconstruction signal unless the
     * calibration scale was established against a traceable photometric reference.
     */
    fun normalizeRadiance(
        normalizedLuminance: Double,
        telemetry: FrameTelemetry?,
        calibration: com.nivukx.nox.core.model.PhotometricCalibration?,
    ): Double {
        val y =
            normalizedLuminance
                .coerceIn(0.0, 1.0)

        if (calibration == null) {
            return y
        }

        val black =
            calibration.blackLevel
                .coerceIn(0.0, 1.0)
        val white =
            calibration.whiteLevel
                .coerceIn(
                    black + 1e-6,
                    1.0,
                )

        val response =
            (
                (y - black) /
                    (white - black)
                ).coerceIn(0.0, 1.0)

        val exposureSeconds =
            (
                telemetry?.exposureTimeNanos ?: 10_000_000L
            )
                .coerceAtLeast(1L)
                .toDouble() /
                1e9

        val iso =
            (
                telemetry?.sensorSensitivityIso ?: 100
            )
                .coerceAtLeast(1)

        val exposureGain =
            (
                exposureSeconds *
                    (iso / 100.0)
            ).coerceAtLeast(1e-6)

        val temperature =
            telemetry
                ?.sensorTemperatureCelsius
                ?.toDouble()
                ?: calibration.referenceTemperatureCelsius

        val temperatureGain =
            (
                1.0 +
                    calibration.temperatureCoefficient *
                    (
                        temperature -
                            calibration.referenceTemperatureCelsius
                    )
            ).coerceIn(
                0.50,
                1.50,
            )

        return (
            response /
                exposureGain /
                temperatureGain
        )
            .coerceIn(0.0, 4.0)
    }

    fun environment(estimate: LightingEstimate): EnvironmentLighting =
        EnvironmentLighting(
            timestampNanos = estimate.timestampNanos,
            ambient = Vec3(
                estimate.ambientLuminance,
                estimate.ambientLuminance,
                estimate.ambientLuminance,
            ),
            dominantDirection = estimate.directionalLight,
            dominantIntensity = estimate.directionalIntensity,
            colorTemperatureKelvin = estimate.colorTemperatureKelvin,
            exposureValue = estimate.exposureCompensationEv,
            confidence = estimate.confidence,
        )

    private data class Statistics(
        var samples: Int = 0,
        var meanY: Double = 0.0,
        var meanU: Double = 128.0,
        var meanV: Double = 128.0,
        var minY: Double = 255.0,
        var maxY: Double = 0.0,
        var varianceY: Double = 0.0,
        var clippedRatio: Double = 0.0,
        var crushedRatio: Double = 0.0,
        var gradientX: Double = 0.0,
        var gradientY: Double = 0.0,
    )

    private fun sample(image: Image): Statistics {
        val yPlane = image.planes.firstOrNull() ?: return Statistics()
        val uPlane = image.planes.getOrNull(1)
        val vPlane = image.planes.getOrNull(2)

        val y = yPlane.buffer.duplicate()
        val u = uPlane?.buffer?.duplicate()
        val v = vPlane?.buffer?.duplicate()

        val stride = config.sampleStride.coerceAtLeast(1)
        var sumY = 0.0
        var sumU = 0.0
        var sumV = 0.0
        var sumY2 = 0.0
        var count = 0
        var clipped = 0
        var crushed = 0
        var minY = 255.0
        var maxY = 0.0
        var gx = 0.0
        var gy = 0.0

        for (row in stride until image.height step stride) {
            var previous = -1.0
            for (col in stride until image.width step stride) {
                val luma = samplePlane(
                    y,
                    yPlane.rowStride,
                    yPlane.pixelStride,
                    col,
                    row,
                    image.width,
                    image.height,
                ).toDouble()

                val chromaX = (col / 2).coerceAtLeast(0)
                val chromaY = (row / 2).coerceAtLeast(0)
                val chromaWidth = (image.width / 2).coerceAtLeast(1)
                val chromaHeight = (image.height / 2).coerceAtLeast(1)

                val chromaU = if (u != null && uPlane != null) {
                    samplePlane(
                        u,
                        uPlane.rowStride,
                        uPlane.pixelStride,
                        chromaX,
                        chromaY,
                        chromaWidth,
                        chromaHeight,
                    ).toDouble()
                } else {
                    128.0
                }

                val chromaV = if (v != null && vPlane != null) {
                    samplePlane(
                        v,
                        vPlane.rowStride,
                        vPlane.pixelStride,
                        chromaX,
                        chromaY,
                        chromaWidth,
                        chromaHeight,
                    ).toDouble()
                } else {
                    128.0
                }

                sumY += luma
                sumY2 += luma * luma
                sumU += chromaU
                sumV += chromaV
                count += 1

                if (luma >= 250.0) clipped += 1
                if (luma <= 5.0) crushed += 1
                minY = minOf(minY, luma)
                maxY = maxOf(maxY, luma)

                if (previous >= 0.0) {
                    gx += (luma - previous) / 255.0
                }
                previous = luma
            }
        }

        if (count == 0) return Statistics()

        val mean = sumY / count
        val variance = (
            sumY2 / count - mean * mean
        ).coerceAtLeast(0.0) / (255.0 * 255.0)

        val normalizedGx = gx / count.toDouble()

        // Vertical gradient is sampled in a second low-cost pass to avoid holding an image-sized
        // intermediate buffer just for illumination estimation.
        for (col in stride until image.width step stride) {
            var previous = -1.0
            for (row in stride until image.height step stride) {
                val current = samplePlane(
                    y,
                    yPlane.rowStride,
                    yPlane.pixelStride,
                    col,
                    row,
                    image.width,
                    image.height,
                ).toDouble()

                if (previous >= 0.0) {
                    gy += (current - previous) / 255.0
                }
                previous = current
            }
        }

        return Statistics(
            samples = count,
            meanY = mean,
            meanU = sumU / count,
            meanV = sumV / count,
            minY = minY,
            maxY = maxY,
            varianceY = variance.coerceAtMost(1.0),
            clippedRatio = clipped.toDouble() / count,
            crushedRatio = crushed.toDouble() / count,
            gradientX = normalizedGx.coerceIn(-1.0, 1.0),
            gradientY = (gy / count.toDouble()).coerceIn(-1.0, 1.0),
        )
    }

    private fun samplePlane(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
    ): Int {
        if (x !in 0 until width || y !in 0 until height) return 128

        val index = y * rowStride + x * pixelStride
        if (index < 0 || index >= buffer.limit()) return 128

        return buffer.get(index).toInt() and 0xFF
    }

    private fun yuvToRgb(
        y: Double,
        u: Double,
        v: Double,
    ): Vec3 {
        val yy = (y - 16.0).coerceAtLeast(0.0) * 1.16438356
        val uu = u - 128.0
        val vv = v - 128.0

        return Vec3(
            (yy + 1.59602678 * vv).coerceIn(0.0, 255.0) / 255.0,
            (yy - 0.39176272 * uu - 0.81296784 * vv)
                .coerceIn(0.0, 255.0) / 255.0,
            (yy + 2.01723214 * uu).coerceIn(0.0, 255.0) / 255.0,
        )
    }

    private fun estimateColorTemperature(rgb: Vec3): Double {
        val r = rgb.x
        val g = rgb.y
        val b = rgb.z
        val sum = (r + g + b).coerceAtLeast(1e-6)

        val x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / sum
        val y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / sum
        val n = (x - 0.3320) /
            (0.1858 - y).coerceAtLeast(1e-6)

        return (
            449.0 * n * n * n +
                3525.0 * n * n +
                6823.3 * n +
                5520.33
            ).coerceIn(1800.0, 20_000.0)
    }

    private fun estimateDynamicRange(
        minY: Double,
        maxY: Double,
    ): Double {
        val low = (minY / 255.0).coerceAtLeast(1e-5)
        val high = (maxY / 255.0).coerceAtLeast(low)
        return kotlin.math.log2(
            (high / low).coerceAtLeast(1.0),
        ).coerceIn(0.0, 20.0)
    }
}
