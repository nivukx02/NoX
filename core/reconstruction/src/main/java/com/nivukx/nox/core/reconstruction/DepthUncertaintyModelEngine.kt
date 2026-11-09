package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * On-demand depth uncertainty model for dense metric reconstruction.
 *
 * This is a conservative sensor-agnostic envelope. It is deliberately not presented as a vendor
 * depth-noise calibration. The model combines measurable terms at each sampled pixel:
 *
 *  - range-dependent noise,
 *  - local depth discontinuity,
 *  - ray obliqueness,
 *  - surface-normal observability,
 *  - temporal disagreement,
 *  - source confidence.
 *
 * It is evaluated only at TSDF sample points, avoiding a second full-frame float buffer.
 */
class DepthUncertaintyModelEngine(
    private val config: Config = Config(),
) {
    enum class SourceClass {
        DEPTH_SENSOR,
        STEREO,
        LEARNED_DEPTH,
        MONOCULAR_PRIOR,
        UNKNOWN,
    }

    data class Config(
        val minimumSigmaMeters: Double = 0.0015,
        val maximumSigmaMeters: Double = 0.75,
        val baseSigmaMeters: Double = 0.008,
        val rangeLinearCoefficient: Double = 0.004,
        val rangeQuadraticCoefficient: Double = 0.0012,
        val edgeSigmaMultiplier: Double = 2.8,
        val obliqueSigmaMultiplier: Double = 1.6,
        val temporalSigmaMultiplier: Double = 1.7,
        val confidenceSigmaMultiplier: Double = 1.9,
        val depthGradientReferenceMeters: Double = 0.015,
        val sourceScaleDepthSensor: Double = 1.0,
        val sourceScaleStereo: Double = 1.12,
        val sourceScaleLearned: Double = 1.65,
        val sourceScaleMonocular: Double = 2.40,
        val sourceScaleUnknown: Double = 2.0,
    ) {
        init {
            require(minimumSigmaMeters > 0.0)
            require(maximumSigmaMeters > minimumSigmaMeters)
            require(baseSigmaMeters > 0.0)
            require(rangeLinearCoefficient >= 0.0)
            require(rangeQuadraticCoefficient >= 0.0)
            require(edgeSigmaMultiplier >= 1.0)
            require(obliqueSigmaMultiplier >= 1.0)
            require(temporalSigmaMultiplier >= 1.0)
            require(confidenceSigmaMultiplier >= 1.0)
            require(depthGradientReferenceMeters > 0.0)
            listOf(
                sourceScaleDepthSensor,
                sourceScaleStereo,
                sourceScaleLearned,
                sourceScaleMonocular,
                sourceScaleUnknown,
            ).forEach {
                require(it.isFinite() && it >= 0.5)
            }
        }
    }

    data class Input(
        val depthMeters: Double,
        val neighborDepthX: Double?,
        val neighborDepthY: Double?,
        val rayZ: Double,
        val surfaceNormalObservability: Double,
        val temporalConsistency: Double = 1.0,
        val sourceConfidence: Double = 1.0,
        val sourceClass: SourceClass = SourceClass.UNKNOWN,
    ) {
        init {
            require(depthMeters.isFinite() && depthMeters > 0.0)
            require(
                neighborDepthX == null ||
                    (neighborDepthX.isFinite() && neighborDepthX > 0.0),
            )
            require(
                neighborDepthY == null ||
                    (neighborDepthY.isFinite() && neighborDepthY > 0.0),
            )
            require(rayZ.isFinite() && rayZ in -1.0..1.0)
            require(surfaceNormalObservability.isFinite() && surfaceNormalObservability in 0.0..1.0)
            require(temporalConsistency.isFinite() && temporalConsistency in 0.0..1.0)
            require(sourceConfidence.isFinite() && sourceConfidence in 0.0..1.0)
        }
    }

    data class Result(
        val sigmaMeters: Double,
        val confidence: Double,
        val rangeNoiseMeters: Double,
        val edgePenalty: Double,
        val obliquePenalty: Double,
        val temporalPenalty: Double,
        val confidencePenalty: Double,
        val sourceScale: Double,
        val warnings: List<String>,
    )

    fun estimate(input: Input): Result {
        val depth = input.depthMeters

        val rangeNoise =
            (
                config.baseSigmaMeters +
                    config.rangeLinearCoefficient * depth +
                    config.rangeQuadraticCoefficient * depth * depth
                ).coerceAtLeast(config.minimumSigmaMeters)

        val gradients =
            listOfNotNull(
                input.neighborDepthX?.let {
                    abs(it - depth)
                },
                input.neighborDepthY?.let {
                    abs(it - depth)
                },
            )

        val meanGradient =
            gradients
                .ifEmpty { listOf(0.0) }
                .average()

        val edgePenalty =
            (
                1.0 +
                    (
                        meanGradient /
                            config.depthGradientReferenceMeters
                    ).coerceIn(0.0, 4.0) *
                        (config.edgeSigmaMultiplier - 1.0)
                ).coerceAtLeast(1.0)

        val obliquePenalty =
            (
                1.0 +
                    (
                        1.0 -
                            abs(input.rayZ).coerceIn(0.0, 1.0)
                        ) *
                        (config.obliqueSigmaMultiplier - 1.0)
                ).coerceAtLeast(1.0)

        val temporalPenalty =
            (
                1.0 +
                    (
                        1.0 -
                            input.temporalConsistency
                    ) *
                        (config.temporalSigmaMultiplier - 1.0)
                ).coerceAtLeast(1.0)

        val confidencePenalty =
            (
                1.0 +
                    (
                        1.0 -
                            input.sourceConfidence
                    ) *
                        (config.confidenceSigmaMultiplier - 1.0)
                ).coerceAtLeast(1.0)

        val sourceScale =
            when (input.sourceClass) {
                SourceClass.DEPTH_SENSOR ->
                    config.sourceScaleDepthSensor
                SourceClass.STEREO ->
                    config.sourceScaleStereo
                SourceClass.LEARNED_DEPTH ->
                    config.sourceScaleLearned
                SourceClass.MONOCULAR_PRIOR ->
                    config.sourceScaleMonocular
                SourceClass.UNKNOWN ->
                    config.sourceScaleUnknown
            }

        val sigma =
            (
                rangeNoise *
                    edgePenalty *
                    obliquePenalty *
                    temporalPenalty *
                    confidencePenalty *
                    sourceScale /
                    sqrt(
                        input.surfaceNormalObservability
                            .coerceAtLeast(0.20)
                    )
                ).coerceIn(
                    config.minimumSigmaMeters,
                    config.maximumSigmaMeters,
                )

        val confidence =
            exp(
                -sigma /
                    0.065,
            ).coerceIn(0.0, 1.0) *
                input.surfaceNormalObservability *
                input.temporalConsistency *
                input.sourceConfidence

        val warnings = buildList {
            if (sigma >= 0.05) {
                add("DEPTH_UNCERTAINTY_HIGH")
            }
            if (meanGradient >= config.depthGradientReferenceMeters * 2.0) {
                add("DEPTH_DISCONTINUITY_UNCERTAINTY_HIGH")
            }
            if (input.temporalConsistency < 0.60) {
                add("DEPTH_TEMPORAL_CONSISTENCY_LIMITING")
            }
            if (input.surfaceNormalObservability < 0.45) {
                add("DEPTH_SURFACE_NORMAL_OBSERVABILITY_LOW")
            }
            if (input.sourceConfidence < 0.55) {
                add("DEPTH_SOURCE_CONFIDENCE_LOW")
            }
        }.distinct()

        return Result(
            sigmaMeters = sigma,
            confidence = confidence.coerceIn(0.0, 1.0),
            rangeNoiseMeters = rangeNoise,
            edgePenalty = edgePenalty,
            obliquePenalty = obliquePenalty,
            temporalPenalty = temporalPenalty,
            confidencePenalty = confidencePenalty,
            sourceScale = sourceScale,
            warnings = warnings,
        )
    }
}
