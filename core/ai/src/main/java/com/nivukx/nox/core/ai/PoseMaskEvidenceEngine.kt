package com.nivukx.nox.core.ai

import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.MPImage
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max

/**
 * Extracts conservative silhouette evidence from a Pose Landmarker per-pose segmentation mask.
 *
 * The mask is aligned to one detected pose, making it fundamentally different from a global semantic
 * foreground mask. This engine never turns the mask into metric geometry. It reports only support,
 * edge agreement and numerical quality for downstream identity/detail/reconstruction weighting.
 */
class PoseMaskEvidenceEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val foregroundThreshold: Double = 0.42,
        val strongThreshold: Double = 0.72,
        val maximumSamples: Int = 64,
        val minimumValidSamples: Int = 8,
        val minimumDynamicRange: Double = 0.08,
    ) {
        init {
            require(foregroundThreshold in 0.0..1.0)
            require(strongThreshold >= foregroundThreshold)
            require(maximumSamples in 16..128)
            require(minimumValidSamples in 4..maximumSamples)
            require(minimumDynamicRange in 0.0..1.0)
        }
    }

    data class Result(
        val support: Double,
        val strongSupport: Double,
        val boundaryAgreement: Double,
        val quality: Double,
        val validSamples: Int,
        val warnings: List<String>,
    ) {
        val usable: Boolean
            get() =
                validSamples >= 1 &&
                    quality >= 0.20
    }

    fun evaluate(
        mask: MPImage?,
        bounds: MultiActorPoseAssociationEngine.Bounds,
        joints: List<MultiActorPoseAssociationEngine.Joint>,
    ): Result {
        if (mask == null) {
            return empty("POSE_SEGMENTATION_MASK_UNAVAILABLE")
        }

        val buffer =
            runCatching {
                ByteBufferExtractor
                    .extract(mask)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
            }.getOrNull()
                ?: return empty("POSE_SEGMENTATION_MASK_BUFFER_UNAVAILABLE")

        val width =
            mask.width.coerceAtLeast(1)
        val height =
            mask.height.coerceAtLeast(1)

        val samplePoints =
            buildSamples(
                bounds = bounds,
                joints = joints,
            )

        if (samplePoints.isEmpty()) {
            return empty("POSE_SEGMENTATION_NO_SAMPLES")
        }

        val values =
            ArrayList<Double>(
                minOf(
                    samplePoints.size,
                    config.maximumSamples,
                ),
            )

        samplePoints
            .take(config.maximumSamples)
            .forEach { sample ->
                val value =
                    sampleMask(
                        buffer = buffer,
                        width = width,
                        height = height,
                        normalizedX = sample.x,
                        normalizedY = sample.y,
                    ) ?: return@forEach

                values += value
            }

        if (values.size < config.minimumValidSamples) {
            return Result(
                support = 0.0,
                strongSupport = 0.0,
                boundaryAgreement = 0.0,
                quality = 0.0,
                validSamples = values.size,
                warnings =
                    listOf(
                        "POSE_SEGMENTATION_VALID_SAMPLE_COUNT_LOW",
                    ),
            )
        }

        val support =
            values
                .average()
                .coerceIn(0.0, 1.0)

        val strongSupport =
            (
                values.count {
                    it >= config.strongThreshold
                }.toDouble() /
                    values.size.toDouble()
            ).coerceIn(0.0, 1.0)

        val dynamicRange =
            (
                (values.maxOrNull() ?: 0.0) -
                    (values.minOrNull() ?: 0.0)
                )
                .coerceIn(0.0, 1.0)

        val dynamicQuality =
            (
                dynamicRange /
                    config.minimumDynamicRange
                        .coerceAtLeast(1e-6)
                )
                .coerceIn(0.0, 1.0)

        val boundaryAgreement =
            sampleBoundaryAgreement(
                mask = mask,
                bounds = bounds,
            )

        val quality =
            (
                support * 0.42 +
                    strongSupport * 0.22 +
                    dynamicQuality * 0.16 +
                    boundaryAgreement * 0.20
                )
                .coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (support < 0.28) {
                    add("POSE_SEGMENTATION_SUPPORT_LOW")
                }
                if (strongSupport < 0.20) {
                    add("POSE_SEGMENTATION_STRONG_SUPPORT_LOW")
                }
                if (boundaryAgreement < 0.45) {
                    add("POSE_SEGMENTATION_BOUNDARY_WEAK")
                }
                if (
                    dynamicRange <
                        config.minimumDynamicRange
                ) {
                    add("POSE_SEGMENTATION_LOW_DYNAMIC_RANGE")
                }
            }.distinct()

        return Result(
            support = support,
            strongSupport = strongSupport,
            boundaryAgreement =
                boundaryAgreement.coerceIn(
                    0.0,
                    1.0,
                ),
            quality = quality,
            validSamples = values.size,
            warnings = warnings,
        )
    }

    private fun buildSamples(
        bounds: MultiActorPoseAssociationEngine.Bounds,
        joints: List<MultiActorPoseAssociationEngine.Joint>,
    ): List<Sample> {
        val points =
            ArrayList<Sample>(
                config.maximumSamples,
            )

        fun add(
            x: Double,
            y: Double,
            weight: Double,
        ) {
            if (
                x.isFinite() &&
                y.isFinite()
            ) {
                points +=
                    Sample(
                        x = x.coerceIn(0.0, 1.0),
                        y = y.coerceIn(0.0, 1.0),
                        weight = weight.coerceAtLeast(0.05),
                    )
            }
        }

        add(bounds.left, bounds.top, 0.5)
        add(bounds.right, bounds.top, 0.5)
        add(bounds.left, bounds.bottom, 0.5)
        add(bounds.right, bounds.bottom, 0.5)
        add(
            (bounds.left + bounds.right) * 0.5,
            bounds.top,
            0.7,
        )
        add(
            (bounds.left + bounds.right) * 0.5,
            bounds.bottom,
            0.7,
        )
        add(
            bounds.left,
            (bounds.top + bounds.bottom) * 0.5,
            0.55,
        )
        add(
            bounds.right,
            (bounds.top + bounds.bottom) * 0.5,
            0.55,
        )
        add(
            bounds.center.x,
            bounds.center.y,
            0.95,
        )

        joints
            .asSequence()
            .filter {
                it.confidence.isFinite() &&
                    it.confidence > 0.12 &&
                    it.x.isFinite() &&
                    it.y.isFinite()
            }
            .sortedByDescending {
                it.confidence
            }
            .take(
                config.maximumSamples -
                    points.size,
            )
            .forEach { joint ->
                add(
                    joint.x,
                    joint.y,
                    0.55 +
                        joint.confidence
                            .coerceIn(0.0, 1.0) *
                        0.45,
                )
            }

        return points
            .distinctBy {
                (
                    it.x * 2048.0
                ).toInt() to
                    (
                        it.y * 2048.0
                    ).toInt()
            }
    }

    private fun sampleMask(
        buffer: java.nio.FloatBuffer,
        width: Int,
        height: Int,
        normalizedX: Double,
        normalizedY: Double,
    ): Double? {
        val x =
            (
                normalizedX.coerceIn(0.0, 1.0) *
                    (width - 1)
                )
                .coerceAtLeast(0.0)
        val y =
            (
                normalizedY.coerceIn(0.0, 1.0) *
                    (height - 1)
                )
                .coerceAtLeast(0.0)

        val x0 =
            x.toInt()
                .coerceIn(0, width - 1)
        val y0 =
            y.toInt()
                .coerceIn(0, height - 1)
        val x1 =
            minOf(
                x0 + 1,
                width - 1,
            )
        val y1 =
            minOf(
                y0 + 1,
                height - 1,
            )

        val tx = x - x0
        val ty = y - y0

        fun value(px: Int, py: Int): Double {
            val index =
                py * width + px
            if (
                index < 0 ||
                index >= buffer.limit()
            ) {
                return 0.0
            }
            return buffer.get(index)
                .toDouble()
                .coerceIn(0.0, 1.0)
        }

        val v00 = value(x0, y0)
        val v10 = value(x1, y0)
        val v01 = value(x0, y1)
        val v11 = value(x1, y1)

        val top =
            v00 +
                (v10 - v00) *
                tx
        val bottom =
            v01 +
                (v11 - v01) *
                tx

        return (
            top +
                (bottom - top) *
                ty
            ).coerceIn(0.0, 1.0)
    }

    private fun sampleBoundaryAgreement(
        mask: MPImage,
        bounds: MultiActorPoseAssociationEngine.Bounds,
    ): Double {
        val buffer =
            runCatching {
                ByteBufferExtractor
                    .extract(mask)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
            }.getOrNull()
                ?: return 0.0

        val width =
            mask.width.coerceAtLeast(1)
        val height =
            mask.height.coerceAtLeast(1)

        val insetX =
            (
                bounds.right -
                    bounds.left
                ).coerceIn(0.01, 1.0) *
                0.08
        val insetY =
            (
                bounds.bottom -
                    bounds.top
                ).coerceIn(0.01, 1.0) *
                0.08

        val checks =
            listOf(
                bounds.left + insetX to
                    bounds.center.y,
                bounds.right - insetX to
                    bounds.center.y,
                bounds.center.x to
                    bounds.top + insetY,
                bounds.center.x to
                    bounds.bottom - insetY,
            )

        var agreement = 0.0
        var count = 0

        checks.forEach { (x, y) ->
            val center =
                sampleMask(
                    buffer,
                    width,
                    height,
                    x,
                    y,
                ) ?: return@forEach

            val left =
                sampleMask(
                    buffer,
                    width,
                    height,
                    x - 0.018,
                    y,
                ) ?: return@forEach
            val right =
                sampleMask(
                    buffer,
                    width,
                    height,
                    x + 0.018,
                    y,
                ) ?: return@forEach
            val up =
                sampleMask(
                    buffer,
                    width,
                    height,
                    x,
                    y - 0.018,
                ) ?: return@forEach
            val down =
                sampleMask(
                    buffer,
                    width,
                    height,
                    x,
                    y + 0.018,
                ) ?: return@forEach

            agreement +=
                (
                    1.0 -
                        listOf(
                            abs(center - left),
                            abs(center - right),
                            abs(center - up),
                            abs(center - down),
                        ).average()
                ).coerceIn(0.0, 1.0)
            count++
        }

        return if (count <= 0) {
            0.0
        } else {
            (
                agreement /
                    count.toDouble()
            ).coerceIn(0.0, 1.0)
        }
    }

    private fun empty(
        warning: String,
    ): Result =
        Result(
            support = 0.0,
            strongSupport = 0.0,
            boundaryAgreement = 0.0,
            quality = 0.0,
            validSamples = 0,
            warnings = listOf(warning),
        )

    private data class Sample(
        val x: Double,
        val y: Double,
        val weight: Double,
    )
}
