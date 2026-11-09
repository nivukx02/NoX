package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Robust cross-camera radiometric alignment for heterogeneous RGB rigs.
 *
 * The fit models each target channel as a constrained affine function of a normalized source
 * channel:
 *
 *     target = gain * source + offset
 *
 * The input is first normalized by the reported black/white levels and relative exposure*ISO
 * product. This separates capture-dependent exposure from the camera-to-camera residual response.
 *
 * This is intentionally not a neural color-transfer model. It is deterministic, bounded, auditable,
 * and safe to run online during a calibration sequence. It is an evidence layer for cross-view
 * consistency, not a physical-lux guarantee.
 */
class CrossCameraPhotometricCalibrationEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumSamples: Int = 24,
        val minimumDistinctLumaBins: Int = 8,
        val minimumSourceVariance: Double = 0.002,
        val maximumRejectRatio: Double = 0.45,
        val huberThreshold: Double = 0.035,
        val maximumIterations: Int = 6,
        val minimumConsistency: Double = 0.62,
        val minimumChannelConfidence: Double = 0.55,
        val minimumGain: Double = 0.25,
        val maximumGain: Double = 4.0,
        val minimumOffset: Double = -0.25,
        val maximumOffset: Double = 0.25,
    ) {
        init {
            require(minimumSamples >= 8)
            require(minimumDistinctLumaBins in 2..64)
            require(minimumSourceVariance > 0.0)
            require(maximumRejectRatio in 0.0..0.80)
            require(huberThreshold > 0.0)
            require(maximumIterations in 1..12)
            require(minimumConsistency in 0.0..1.0)
            require(minimumChannelConfidence in 0.0..1.0)
            require(minimumGain > 0.0)
            require(maximumGain > minimumGain)
            require(minimumOffset < maximumOffset)
        }
    }

    data class Rgb(
        val red: Double,
        val green: Double,
        val blue: Double,
    ) {
        init {
            require(red.isFinite() && green.isFinite() && blue.isFinite())
            require(red in 0.0..1.0)
            require(green in 0.0..1.0)
            require(blue in 0.0..1.0)
        }

        fun luminance(): Double =
            (
                red * 0.2126 +
                    green * 0.7152 +
                    blue * 0.0722
                ).coerceIn(0.0, 1.0)
    }

    data class CaptureMetadata(
        val exposureNanos: Long = 1L,
        val iso: Int = 1,
        val blackLevel: Double = 0.0,
        val whiteLevel: Double = 1.0,
    ) {
        init {
            require(exposureNanos > 0L)
            require(iso > 0)
            require(blackLevel.isFinite() && blackLevel in 0.0..0.99)
            require(
                whiteLevel.isFinite() &&
                    whiteLevel > blackLevel &&
                    whiteLevel <= 1.0,
            )
        }
    }

    data class Sample(
        val source: Rgb,
        val target: Rgb,
        val confidence: Double,
        val sourceSaturation: Boolean = false,
        val targetSaturation: Boolean = false,
        val sourceMetadata: CaptureMetadata =
            CaptureMetadata(),
        val targetMetadata: CaptureMetadata =
            CaptureMetadata(),
    ) {
        init {
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class ChannelTransform(
        val gain: Double,
        val offset: Double,
        val confidence: Double,
    ) {
        init {
            require(gain.isFinite() && gain > 0.0)
            require(offset.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }

        fun apply(value: Double): Double =
            (
                gain * value +
                    offset
                ).coerceIn(0.0, 1.0)
    }

    data class Transform(
        val red: ChannelTransform,
        val green: ChannelTransform,
        val blue: ChannelTransform,
        val calibratedSampleCount: Int,
        val rejectedSampleCount: Int,
    ) {
        fun apply(rgb: Rgb): Rgb =
            Rgb(
                red = red.apply(rgb.red),
                green = green.apply(rgb.green),
                blue = blue.apply(rgb.blue),
            )

        val confidence: Double
            get() =
                (
                    red.confidence * 0.3334 +
                        green.confidence * 0.3333 +
                        blue.confidence * 0.3333
                ).coerceIn(0.0, 1.0)
    }

    data class Result(
        val sourceDeviceId: DeviceId,
        val referenceDeviceId: DeviceId,
        val transform: Transform,
        val consistency: Double,
        val whiteBalanceShift: Double,
        val rmse: Double,
        val acceptedSampleCount: Int,
        val rejectedSampleCount: Int,
        val rejectionRatio: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    private data class NormalizedSample(
        val source: Rgb,
        val target: Rgb,
        val confidence: Double,
    )

    private data class ChannelFit(
        val transform: ChannelTransform,
        val rmse: Double,
        val acceptedCount: Int,
        val rejectedCount: Int,
    )

    fun fit(
        sourceDeviceId: DeviceId,
        referenceDeviceId: DeviceId,
        samples: List<Sample>,
    ): Result {
        require(sourceDeviceId != referenceDeviceId)

        val usable =
            samples
                .asSequence()
                .filterNot {
                    it.sourceSaturation ||
                        it.targetSaturation
                }
                .map(::normalize)
                .filter {
                    it.confidence >=
                        config.minimumChannelConfidence
                }
                .toList()

        if (usable.size < config.minimumSamples) {
            return rejected(
                sourceDeviceId = sourceDeviceId,
                referenceDeviceId = referenceDeviceId,
                inputCount = samples.size,
                warning = "PHOTOMETRIC_SAMPLE_COUNT_INSUFFICIENT",
            )
        }

        val distinctBins =
            usable
                .map {
                    (
                        it.source.luminance() *
                            32.0
                        ).toInt()
                    }
                    .distinct()
                    .size

        if (
            distinctBins <
                config.minimumDistinctLumaBins
        ) {
            return rejected(
                sourceDeviceId = sourceDeviceId,
                referenceDeviceId = referenceDeviceId,
                inputCount = samples.size,
                warning =
                    "PHOTOMETRIC_SOURCE_VARIATION_INSUFFICIENT",
            )
        }

        val sourceVariance =
            weightedVariance(
                values =
                    usable.map {
                        it.source.luminance()
                    },
                weights =
                    usable.map {
                        it.confidence
                    },
            )

        if (
            sourceVariance <
                config.minimumSourceVariance
        ) {
            return rejected(
                sourceDeviceId = sourceDeviceId,
                referenceDeviceId = referenceDeviceId,
                inputCount = samples.size,
                warning =
                    "PHOTOMETRIC_SOURCE_VARIATION_INSUFFICIENT",
            )
        }

        val red =
            fitChannel(
                source =
                    usable.map { it.source.red },
                target =
                    usable.map { it.target.red },
                weights =
                    usable.map { it.confidence },
            )
        val green =
            fitChannel(
                source =
                    usable.map { it.source.green },
                target =
                    usable.map { it.target.green },
                weights =
                    usable.map { it.confidence },
            )
        val blue =
            fitChannel(
                source =
                    usable.map { it.source.blue },
                target =
                    usable.map { it.target.blue },
                weights =
                    usable.map { it.confidence },
            )

        val accepted =
            min(
                red.transform.confidence,
                min(
                    green.transform.confidence,
                    blue.transform.confidence,
                ),
            ) >= config.minimumChannelConfidence

        val totalAccepted =
            minOf(
                red.acceptedCount,
                minOf(
                    green.acceptedCount,
                    blue.acceptedCount,
                ),
            )
        val totalRejected =
            maxOf(
                red.rejectedCount,
                maxOf(
                    green.rejectedCount,
                    blue.rejectedCount,
                ),
            )

        val totalUsed =
            (
                totalAccepted +
                    totalRejected
                ).coerceAtLeast(1)
        val rejectionRatio =
            totalRejected.toDouble() /
                totalUsed.toDouble()

        val rmse =
            (
                red.rmse * 0.3334 +
                    green.rmse * 0.3333 +
                    blue.rmse * 0.3333
                )

        val consistency =
            exp(
                -rmse /
                    0.035,
            ).coerceIn(0.0, 1.0)

        val whiteBalanceShift =
            whiteBalanceShift(
                red = red.transform.gain,
                green = green.transform.gain,
                blue = blue.transform.gain,
            )

        val warnings =
            buildList {
                if (rejectionRatio > 0.25) {
                    add(
                        "PHOTOMETRIC_OUTLIER_RATIO_HIGH",
                    )
                }
                if (consistency < 0.80) {
                    add(
                        "PHOTOMETRIC_CROSS_CAMERA_CONSISTENCY_LOW",
                    )
                }
                if (
                    whiteBalanceShift >
                        0.12
                ) {
                    add(
                        "PHOTOMETRIC_WHITE_BALANCE_SHIFT_HIGH",
                    )
                }
                if (
                    listOf(
                        red.transform,
                        green.transform,
                        blue.transform,
                    ).any {
                        it.gain <=
                            config.minimumGain + 1e-9 ||
                            it.gain >=
                            config.maximumGain - 1e-9
                    }
                ) {
                    add(
                        "PHOTOMETRIC_GAIN_CLAMPED",
                    )
                }
                if (!accepted) {
                    add("PHOTOMETRIC_CALIBRATION_NOT_ACCEPTED")
                }
            }.distinct()

        return Result(
            sourceDeviceId = sourceDeviceId,
            referenceDeviceId = referenceDeviceId,
            transform =
                Transform(
                    red = red.transform,
                    green = green.transform,
                    blue = blue.transform,
                    calibratedSampleCount = totalAccepted,
                    rejectedSampleCount = totalRejected,
                ),
            consistency = consistency,
            whiteBalanceShift = whiteBalanceShift,
            rmse = rmse,
            acceptedSampleCount = totalAccepted,
            rejectedSampleCount = totalRejected,
            rejectionRatio = rejectionRatio,
            accepted =
                accepted &&
                    consistency >=
                        config.minimumConsistency &&
                    rejectionRatio <=
                        config.maximumRejectRatio,
            warnings = warnings,
        )
    }

    private fun normalize(
        sample: Sample,
    ): NormalizedSample {
        val source =
            normalizeRgb(
                rgb = sample.source,
                metadata = sample.sourceMetadata,
            )
        val target =
            normalizeRgb(
                rgb = sample.target,
                metadata = sample.targetMetadata,
            )

        /*
         * Exposure*ISO is treated only as a relative capture factor. It is not claimed to be an
         * absolute radiometric measurement. The ratio is intentionally bounded to avoid exploding
         * noisy low-light samples. The source channel is scaled toward the target exposure domain,
         * so longer source exposures are normalized downward rather than amplified.
         */
        val sourceCaptureFactor =
            (
                sample.sourceMetadata.exposureNanos
                    .toDouble() *
                    sample.sourceMetadata.iso
            ).coerceIn(
                1.0,
                1.0e12,
            )

        val targetCaptureFactor =
            (
                sample.targetMetadata.exposureNanos
                    .toDouble() *
                    sample.targetMetadata.iso
            ).coerceIn(
                1.0,
                1.0e12,
            )

        val exposureRatio =
            (
                targetCaptureFactor /
                    sourceCaptureFactor
                ).coerceIn(
                    0.25,
                    4.0,
                )

        return NormalizedSample(
            source =
                Rgb(
                    red =
                        (source.red * exposureRatio)
                            .coerceIn(0.0, 1.0),
                    green =
                        (source.green * exposureRatio)
                            .coerceIn(0.0, 1.0),
                    blue =
                        (source.blue * exposureRatio)
                            .coerceIn(0.0, 1.0),
                ),
            target = target,
            confidence =
                sample.confidence *
                    lowSignalConfidence(
                        sample.source,
                    ) *
                    lowSignalConfidence(
                        sample.target,
                    ),
        )
    }

    private fun normalizeRgb(
        rgb: Rgb,
        metadata: CaptureMetadata,
    ): Rgb {
        val scale =
            (
                metadata.whiteLevel -
                    metadata.blackLevel
                ).coerceAtLeast(1e-6)

        return Rgb(
            red =
                (
                    rgb.red -
                        metadata.blackLevel
                ) / scale,
            green =
                (
                    rgb.green -
                        metadata.blackLevel
                ) / scale,
            blue =
                (
                    rgb.blue -
                        metadata.blackLevel
                ) / scale,
        )
    }

    private fun fitChannel(
        source: List<Double>,
        target: List<Double>,
        weights: List<Double>,
    ): ChannelFit {
        var activeWeights =
            weights.map {
                it.coerceIn(1e-4, 1.0)
            }

        var gain = 1.0
        var offset = 0.0

        repeat(config.maximumIterations) {
            val fit =
                weightedAffineFit(
                    source = source,
                    target = target,
                    weights = activeWeights,
                )

            gain =
                fit.first.coerceIn(
                    config.minimumGain,
                    config.maximumGain,
                )
            offset =
                fit.second.coerceIn(
                    config.minimumOffset,
                    config.maximumOffset,
                )

            activeWeights =
                source.indices.map { index ->
                    val residual =
                        abs(
                            target[index] -
                                (
                                    gain *
                                        source[index] +
                                        offset
                                    )
                        )
                    val huberScale =
                        config.huberThreshold /
                            max(
                                residual,
                                config.huberThreshold,
                            )
                    (
                        weights[index] *
                            huberScale
                        ).coerceIn(
                            1e-5,
                            1.0,
                        )
                }
        }

        val residuals =
            source.indices.map { index ->
                abs(
                    target[index] -
                        (
                            gain *
                                source[index] +
                                offset
                            )
                )
            }

        val acceptedFlags =
            residuals.map {
                it <=
                    config.huberThreshold *
                        2.5
            }

        val acceptedCount =
            acceptedFlags.count { it }
        val rejectedCount =
            acceptedFlags.size -
                acceptedCount

        val weightedRmse =
            sqrt(
                (
                    residuals.indices.sumOf { index ->
                        val weight =
                            if (
                                acceptedFlags[index]
                            ) {
                                weights[index]
                            } else {
                                weights[index] * 0.10
                            }
                        weight *
                            residuals[index] *
                            residuals[index]
                    }
                ) /
                    weights.sum()
                        .coerceAtLeast(1e-9)
            )

        val variance =
            weightedVariance(
                values = source,
                weights = weights,
            )

        val fitConfidence =
            (
                min(
                    acceptedCount.toDouble() /
                        source.size.coerceAtLeast(1),
                    1.0,
                ) * 0.55 +
                    exp(
                        -weightedRmse /
                            config.huberThreshold,
                    ) * 0.45
                ).coerceIn(0.0, 1.0) *
                (
                    (
                        variance /
                            config.minimumSourceVariance
                    ).coerceIn(0.0, 1.0)
                )

        return ChannelFit(
            transform =
                ChannelTransform(
                    gain = gain,
                    offset = offset,
                    confidence =
                        fitConfidence
                            .coerceIn(0.0, 1.0),
                ),
            rmse = weightedRmse,
            acceptedCount = acceptedCount,
            rejectedCount = rejectedCount,
        )
    }

    private fun weightedAffineFit(
        source: List<Double>,
        target: List<Double>,
        weights: List<Double>,
    ): Pair<Double, Double> {
        val weightSum =
            weights.sum()
                .coerceAtLeast(1e-9)
        val weightedMeanSource =
            source.indices.sumOf {
                source[it] *
                    weights[it]
            } / weightSum
        val weightedMeanTarget =
            target.indices.sumOf {
                target[it] *
                    weights[it]
            } / weightSum

        var covariance = 0.0
        var variance = 0.0

        source.indices.forEach { index ->
            val sourceCentered =
                source[index] -
                    weightedMeanSource
            val targetCentered =
                target[index] -
                    weightedMeanTarget

            covariance +=
                weights[index] *
                    sourceCentered *
                    targetCentered
            variance +=
                weights[index] *
                    sourceCentered *
                    sourceCentered
        }

        if (variance <= 1e-12) {
            return 1.0 to 0.0
        }

        val gain =
            covariance / variance
        val offset =
            weightedMeanTarget -
                gain *
                weightedMeanSource

        return gain to offset
    }

    private fun weightedVariance(
        values: List<Double>,
        weights: List<Double>,
    ): Double {
        val weightSum =
            weights.sum()
                .coerceAtLeast(1e-9)
        val mean =
            values.indices.sumOf {
                values[it] *
                    weights[it]
            } / weightSum

        return (
            values.indices.sumOf {
                weights[it] *
                    (
                        values[it] -
                            mean
                        ) *
                    (
                        values[it] -
                            mean
                        )
            } /
                weightSum
            ).coerceAtLeast(0.0)
    }

    private fun lowSignalConfidence(
        rgb: Rgb,
    ): Double {
        val luminance =
            rgb.luminance()

        return when {
            luminance < 0.01 -> 0.25
            luminance < 0.03 -> 0.55
            luminance > 0.99 -> 0.35
            luminance > 0.97 -> 0.65
            else -> 1.0
        }
    }

    private fun whiteBalanceShift(
        red: Double,
        green: Double,
        blue: Double,
    ): Double {
        val mean =
            (red + green + blue) / 3.0

        return (
            abs(red - mean) +
                abs(green - mean) +
                abs(blue - mean)
            ) /
            mean.coerceAtLeast(1e-9)
    }

    private fun rejected(
        sourceDeviceId: DeviceId,
        referenceDeviceId: DeviceId,
        inputCount: Int,
        warning: String,
    ): Result =
        Result(
            sourceDeviceId = sourceDeviceId,
            referenceDeviceId = referenceDeviceId,
            transform =
                Transform(
                    red =
                        ChannelTransform(
                            gain = 1.0,
                            offset = 0.0,
                            confidence = 0.0,
                        ),
                    green =
                        ChannelTransform(
                            gain = 1.0,
                            offset = 0.0,
                            confidence = 0.0,
                        ),
                    blue =
                        ChannelTransform(
                            gain = 1.0,
                            offset = 0.0,
                            confidence = 0.0,
                        ),
                    calibratedSampleCount = 0,
                    rejectedSampleCount = inputCount,
                ),
            consistency = 0.0,
            whiteBalanceShift = 0.0,
            rmse = Double.POSITIVE_INFINITY,
            acceptedSampleCount = 0,
            rejectedSampleCount = inputCount,
            rejectionRatio = 1.0,
            accepted = false,
            warnings =
                listOf(
                    warning,
                    "PHOTOMETRIC_CALIBRATION_NOT_ACCEPTED",
                ),
        )
}
