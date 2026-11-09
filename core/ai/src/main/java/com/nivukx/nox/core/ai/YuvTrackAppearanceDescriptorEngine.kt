package com.nivukx.nox.core.ai

import android.graphics.RectF
import android.media.Image
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Low-copy appearance continuity extractor for person association.
 *
 * The descriptor is intentionally designed for mobile capture:
 * - samples Y/U/V planes directly from Image without creating a Bitmap;
 * - normalizes luminance so exposure shifts do not dominate identity;
 * - retains coarse spatial chroma/luma layout that survives moderate pose changes;
 * - emits an explicit quality score so identity association can ignore unusable appearance data;
 * - is bounded to a fixed number of samples per actor for predictable 50-actor operation.
 *
 * This is appearance continuity evidence, not facial recognition. It is never sufficient to create or
 * permanently establish an actor identity by itself.
 */
class YuvTrackAppearanceDescriptorEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val columns: Int = 8,
        val rows: Int = 12,
        val marginFraction: Double = 0.06,
        val minimumSamples: Int = 24,
        val maximumSamples: Int = 96,
        val minimumPlaneCoverage: Double = 0.84,
        val minimumDynamicRange: Double = 0.025,
        val qualitySampleWeight: Double = 0.70,
        val qualityDynamicRangeWeight: Double = 0.18,
        val qualityPlaneCoverageWeight: Double = 0.12,
    ) {
        init {
            require(columns in 4..16)
            require(rows in 4..20)
            require(marginFraction in 0.0..0.25)
            require(minimumSamples >= 4)
            require(maximumSamples >= minimumSamples)
            require(minimumPlaneCoverage in 0.0..1.0)
            require(minimumDynamicRange >= 0.0)
            require(qualitySampleWeight >= 0.0)
            require(qualityDynamicRangeWeight >= 0.0)
            require(qualityPlaneCoverageWeight >= 0.0)
        }
    }

    data class Descriptor(
        val signature: List<Double>,
        val networkFingerprint: List<Double> = emptyList(),
        val quality: Double,
        val validSamples: Int,
        val bounds: RectF,
        val luminanceMean: Double,
        val luminanceStdDev: Double,
        val warnings: List<String>,
    )

    fun describe(
        image: Image,
        bounds: MultiActorPoseAssociationEngine.Bounds,
        rotationDegrees: Int = 0,
    ): Descriptor {
        require(image.format == android.graphics.ImageFormat.YUV_420_888)

        val yPlane = image.planes.getOrNull(0)
            ?: return emptyDescriptor(bounds)
        val uPlane = image.planes.getOrNull(1)
            ?: return emptyDescriptor(bounds)
        val vPlane = image.planes.getOrNull(2)
            ?: return emptyDescriptor(bounds)

        val width = image.width.coerceAtLeast(1)
        val height = image.height.coerceAtLeast(1)

        val rotatedBounds =
            normalizedBoundsForSource(
                bounds = bounds,
                rotationDegrees = rotationDegrees,
            )

        val left =
            (
                rotatedBounds.left *
                    width
            ).coerceIn(
                0f,
                width.toFloat(),
            )
        val top =
            (
                rotatedBounds.top *
                    height
            ).coerceIn(
                0f,
                height.toFloat(),
            )
        val right =
            (
                rotatedBounds.right *
                    width
            ).coerceIn(
                0f,
                width.toFloat(),
            )
        val bottom =
            (
                rotatedBounds.bottom *
                    height
            ).coerceIn(
                0f,
                height.toFloat(),
            )

        if (
            right - left < 4f ||
            bottom - top < 4f
        ) {
            return emptyDescriptor(
                bounds,
                warning = "APPEARANCE_BOUNDS_TOO_SMALL",
            )
        }

        val marginX =
            (right - left) *
                config.marginFraction.toFloat()
        val marginY =
            (bottom - top) *
                config.marginFraction.toFloat()

        val sampleLeft =
            (
                left + marginX
            ).coerceAtLeast(0f)
        val sampleTop =
            (
                top + marginY
            ).coerceAtLeast(0f)
        val sampleRight =
            (
                right - marginX
            ).coerceAtMost(width.toFloat())
        val sampleBottom =
            (
                bottom - marginY
            ).coerceAtMost(height.toFloat())

        val totalTargetSamples =
            (
                config.columns *
                    config.rows
                ).coerceAtMost(
                config.maximumSamples,
            )

        val lumaValues =
            DoubleArray(
                totalTargetSamples,
            )
        val uValues =
            DoubleArray(
                totalTargetSamples,
            )
        val vValues =
            DoubleArray(
                totalTargetSamples,
            )

        var valid = 0
        var sumY = 0.0
        var sumY2 = 0.0
        var sumU = 0.0
        var sumU2 = 0.0
        var sumV = 0.0
        var sumV2 = 0.0
        var chromaCoverage = 0

        for (row in 0 until config.rows) {
            for (column in 0 until config.columns) {
                if (
                    valid >=
                        totalTargetSamples
                ) break

                val nx =
                    (
                        column + 0.5
                    ) /
                        config.columns.toDouble()
                val ny =
                    (
                        row + 0.5
                    ) /
                        config.rows.toDouble()

                val x =
                    (
                        sampleLeft +
                            (
                                sampleRight -
                                    sampleLeft
                            ) *
                            nx.toFloat()
                    ).toInt().coerceIn(
                        0,
                        width - 1,
                    )
                val y =
                    (
                        sampleTop +
                            (
                                sampleBottom -
                                    sampleTop
                            ) *
                            ny.toFloat()
                    ).toInt().coerceIn(
                        0,
                        height - 1,
                    )

                val yValue =
                    samplePlane(
                        plane = yPlane,
                        x = x,
                        y = y,
                        fullWidth = width,
                        fullHeight = height,
                    ) ?: continue

                val uValue =
                    samplePlane(
                        plane = uPlane,
                        x = x,
                        y = y,
                        fullWidth = width,
                        fullHeight = height,
                        chroma = true,
                    ) ?: continue

                val vValue =
                    samplePlane(
                        plane = vPlane,
                        x = x,
                        y = y,
                        fullWidth = width,
                        fullHeight = height,
                        chroma = true,
                    ) ?: continue

                lumaValues[valid] =
                    yValue
                uValues[valid] =
                    uValue
                vValues[valid] =
                    vValue

                sumY += yValue
                sumY2 +=
                    yValue *
                        yValue
                sumU += uValue
                sumU2 +=
                    uValue *
                        uValue
                sumV += vValue
                sumV2 +=
                    vValue *
                        vValue

                valid += 1
                chromaCoverage += 1
            }
        }

        if (valid < config.minimumSamples) {
            return Descriptor(
                signature = emptyList(),
                networkFingerprint = emptyList(),
                quality = 0.0,
                validSamples = valid,
                bounds =
                    RectF(
                        left,
                        top,
                        right,
                        bottom,
                    ),
                luminanceMean =
                    if (valid > 0) {
                        sumY / valid
                    } else {
                        0.0
                    },
                luminanceStdDev = 0.0,
                warnings = listOf(
                    "APPEARANCE_VALID_SAMPLE_COUNT_LOW",
                ),
            )
        }

        val meanY =
            sumY /
                valid.toDouble()
        val variance =
            (
                sumY2 /
                    valid.toDouble() -
                    meanY *
                    meanY
            ).coerceAtLeast(0.0)
        val stdY =
            sqrt(variance)

        val meanU =
            sumU /
                valid.toDouble()
        val meanV =
            sumV /
                valid.toDouble()
        val varianceU =
            (
                sumU2 /
                    valid.toDouble() -
                    meanU *
                    meanU
            ).coerceAtLeast(0.0)
        val varianceV =
            (
                sumV2 /
                    valid.toDouble() -
                    meanV *
                    meanV
            ).coerceAtLeast(0.0)
        val normalizedChromaStdU =
            sqrt(varianceU).coerceAtLeast(8.0)
        val normalizedChromaStdV =
            sqrt(varianceV).coerceAtLeast(8.0)

        val dynamicRange =
            lumaValues
                .take(valid)
                .let { values ->
                    (
                        values.maxOrNull()
                            ?: meanY
                    ) -
                        (
                            values.minOrNull()
                                ?: meanY
                        )
                }

        val sampleCoverage =
            (
                valid.toDouble() /
                    totalTargetSamples.coerceAtLeast(1)
                        .toDouble()
            ).coerceIn(
                0.0,
                1.0,
            )

        val planeCoverage =
            (
                chromaCoverage.toDouble() /
                    valid.coerceAtLeast(1).toDouble()
            ).coerceIn(
                0.0,
                1.0,
            )

        val dynamicQuality =
            (
                dynamicRange /
                    config.minimumDynamicRange
                        .coerceAtLeast(1e-6)
            ).coerceIn(
                0.0,
                1.0,
            )

        val quality =
            (
                sampleCoverage *
                    config.qualitySampleWeight +
                    dynamicQuality *
                    config.qualityDynamicRangeWeight +
                    planeCoverage *
                    config.qualityPlaneCoverageWeight
                ).coerceIn(
                0.0,
                1.0,
            )

        val signature =
            ArrayList<Double>(
                valid * 3 + 12,
            )

        val normalizedStd =
            stdY.coerceAtLeast(
                8.0,
            )

        for (index in 0 until valid) {
            val normalizedY =
                (
                    (
                        lumaValues[index] -
                            meanY
                    ) /
                        normalizedStd
                    ).coerceIn(
                    -3.0,
                    3.0,
                ) / 3.0

            /*
             * Chroma is centered and scale-normalized per actor crop. This keeps the continuity
             * descriptor useful across cameras with different white-balance gains/offsets while
             * preserving the spatial chroma pattern itself.
             */
            val normalizedU =
                (
                    (
                        uValues[index] -
                            meanU
                    ) /
                        normalizedChromaStdU
                ).coerceIn(-3.0, 3.0) / 3.0

            val normalizedV =
                (
                    (
                        vValues[index] -
                            meanV
                    ) /
                        normalizedChromaStdV
                ).coerceIn(-3.0, 3.0) / 3.0

            signature +=
                normalizedY
            signature +=
                normalizedU
            signature +=
                normalizedV
        }

        val histogramBins = 8
        val yHistogram =
            DoubleArray(histogramBins)
        val uHistogram =
            DoubleArray(histogramBins)
        val vHistogram =
            DoubleArray(histogramBins)

        for (index in 0 until valid) {
            yHistogram[
                quantize(
                    lumaValues[index] /
                        255.0,
                    histogramBins,
                )
            ] += 1.0

            uHistogram[
                quantize(
                    uValues[index] /
                        255.0,
                    histogramBins,
                )
            ] += 1.0

            vHistogram[
                quantize(
                    vValues[index] /
                        255.0,
                    histogramBins,
                )
            ] += 1.0
        }

        val inverseCount =
            1.0 /
                valid.toDouble()

        yHistogram.forEach {
            signature +=
                it *
                    inverseCount
        }
        uHistogram.forEach {
            signature +=
                it *
                    inverseCount
        }
        vHistogram.forEach {
            signature +=
                it *
                    inverseCount
        }

        val networkFingerprint =
            yHistogram
                .take(8) +
                uHistogram
                    .take(4) +
                vHistogram
                    .take(4)

        val warnings =
            buildList {
                if (
                    quality < 0.25
                ) {
                    add(
                        "APPEARANCE_QUALITY_LOW",
                    )
                }
                if (
                    normalizedChromaStdU <= 8.0001 ||
                    normalizedChromaStdV <= 8.0001
                ) {
                    add(
                        "APPEARANCE_CHROMA_VARIATION_LOW",
                    )
                }
                if (
                    dynamicRange <
                        config.minimumDynamicRange
                ) {
                    add(
                        "APPEARANCE_LOW_LUMA_DYNAMIC_RANGE",
                    )
                }
                if (
                    planeCoverage <
                        config.minimumPlaneCoverage
                ) {
                    add(
                        "APPEARANCE_CHROMA_COVERAGE_LOW",
                    )
                }
            }.distinct()

        return Descriptor(
            signature = signature,
            networkFingerprint =
                networkFingerprint
                    .map { it.coerceIn(0.0, 1.0) },
            quality =
                quality.coerceIn(
                    0.0,
                    1.0,
                ),
            validSamples = valid,
            bounds =
                RectF(
                    left,
                    top,
                    right,
                    bottom,
                ),
            luminanceMean = meanY,
            luminanceStdDev = stdY,
            warnings = warnings,
        )
    }

    private fun normalizedBoundsForSource(
        bounds: MultiActorPoseAssociationEngine.Bounds,
        rotationDegrees: Int,
    ): RectF {
        val points =
            listOf(
                floatArrayOf(
                    bounds.left.toFloat(),
                    bounds.top.toFloat(),
                ),
                floatArrayOf(
                    bounds.right.toFloat(),
                    bounds.top.toFloat(),
                ),
                floatArrayOf(
                    bounds.left.toFloat(),
                    bounds.bottom.toFloat(),
                ),
                floatArrayOf(
                    bounds.right.toFloat(),
                    bounds.bottom.toFloat(),
                ),
            ).map {
                rotateNormalized(
                    x = it[0].coerceIn(0f, 1f),
                    y = it[1].coerceIn(0f, 1f),
                    degrees = rotationDegrees,
                )
            }

        return RectF(
            points.minOf { it[0] },
            points.minOf { it[1] },
            points.maxOf { it[0] },
            points.maxOf { it[1] },
        )
    }

    private fun rotateNormalized(
        x: Float,
        y: Float,
        degrees: Int,
    ): FloatArray =
        when (((degrees % 360) + 360) % 360) {
            90 ->
                floatArrayOf(
                    1f - y,
                    x,
                )
            180 ->
                floatArrayOf(
                    1f - x,
                    1f - y,
                )
            270 ->
                floatArrayOf(
                    y,
                    1f - x,
                )
            else ->
                floatArrayOf(
                    x,
                    y,
                )
        }

    private fun samplePlane(
        plane: Image.Plane,
        x: Int,
        y: Int,
        fullWidth: Int,
        fullHeight: Int,
        chroma: Boolean = false,
    ): Double? {
        val planeWidth =
            if (chroma) {
                (
                    (
                        fullWidth + 1
                    ) / 2
                ).coerceAtLeast(1)
            } else {
                fullWidth
            }

        val planeHeight =
            if (chroma) {
                (
                    (
                        fullHeight + 1
                    ) / 2
                ).coerceAtLeast(1)
            } else {
                fullHeight
            }

        val px =
            (
                x.toDouble() /
                    fullWidth.coerceAtLeast(1)
                        .toDouble() *
                    planeWidth
            ).toInt().coerceIn(
                0,
                planeWidth - 1,
            )

        val py =
            (
                y.toDouble() /
                    fullHeight.coerceAtLeast(1)
                        .toDouble() *
                    planeHeight
            ).toInt().coerceIn(
                0,
                planeHeight - 1,
            )

        val index =
            py *
                plane.rowStride +
                px *
                plane.pixelStride

        val buffer =
            plane.buffer

        if (
            index < 0 ||
            index >=
                buffer.limit()
        ) {
            return null
        }

        return (
            buffer.get(
                index,
            ).toInt() and
                0xff
            ).toDouble()
    }

    private fun quantize(
        value: Double,
        bins: Int,
    ): Int =
        (
            value.coerceIn(
                0.0,
                0.999999,
            ) *
                bins
        ).toInt().coerceIn(
            0,
            bins - 1,
        )

    private fun emptyDescriptor(
        bounds: MultiActorPoseAssociationEngine.Bounds,
        warning: String = "APPEARANCE_PLANES_UNAVAILABLE",
    ): Descriptor =
        Descriptor(
            signature = emptyList(),
            networkFingerprint = emptyList(),
            quality = 0.0,
            validSamples = 0,
            bounds =
                RectF(
                    bounds.left.toFloat(),
                    bounds.top.toFloat(),
                    bounds.right.toFloat(),
                    bounds.bottom.toFloat(),
                ),
            luminanceMean = 0.0,
            luminanceStdDev = 0.0,
            warnings = listOf(warning),
        )
}
