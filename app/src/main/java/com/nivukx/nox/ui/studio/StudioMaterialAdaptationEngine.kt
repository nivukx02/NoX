package com.nivukx.nox.ui.studio

import kotlin.math.abs

/**
 * Semantic material policy for the operator console.
 *
 * Glass is treated as a functional overlay whose opacity, contrast and motion can adapt to the
 * underlying reconstruction content and device accessibility preferences. The policy deliberately
 * protects the live 3D viewport from decorative rendering pressure.
 */
class StudioMaterialAdaptationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumContrast: Double = 0.78,
        val maximumTransparency: Double = 0.58,
        val minimumTransparency: Double = 0.18,
        val highLuminanceThreshold: Double = 0.72,
        val lowLuminanceThreshold: Double = 0.24,
        val maximumMotionScale: Double = 1.0,
        val reducedMotionScale: Double = 0.35,
    ) {
        init {
            require(minimumContrast.isFinite() && minimumContrast in 0.0..1.0)
            require(minimumTransparency.isFinite() && minimumTransparency in 0.0..1.0)
            require(maximumTransparency.isFinite() && maximumTransparency in minimumTransparency..1.0)
            require(lowLuminanceThreshold.isFinite() && lowLuminanceThreshold in 0.0..1.0)
            require(highLuminanceThreshold.isFinite() && highLuminanceThreshold in 0.0..1.0)
            require(lowLuminanceThreshold < highLuminanceThreshold)
            require(maximumMotionScale.isFinite() && maximumMotionScale in 0.0..1.0)
            require(reducedMotionScale.isFinite() && reducedMotionScale in 0.0..maximumMotionScale)
        }
    }

    enum class MaterialRole {
        PRIMARY_NAVIGATION,
        SECONDARY_TELEMETRY,
        ACTION_CLUSTER,
        TIMELINE,
        DIAGNOSTICS,
        MODAL,
    }

    data class Input(
        val underlyingLuminance: Double,
        val contentContrast: Double,
        val capturePriority: Double = 1.0,
        val reduceTransparency: Boolean = false,
        val reduceMotion: Boolean = false,
        val operatorFocus: Double = 1.0,
    )

    data class Policy(
        val materialAlpha: Double,
        val borderAlpha: Double,
        val specularAlpha: Double,
        val effectScale: Double,
        val animationScale: Double,
        val contentContrastBoost: Double,
        val preserveViewportPriority: Boolean,
        val warnings: List<String>,
    )

    fun evaluate(
        role: MaterialRole,
        input: Input,
    ): Policy {
        require(input.underlyingLuminance in 0.0..1.0)
        require(input.contentContrast in 0.0..1.0)
        require(input.capturePriority in 0.0..1.0)
        require(input.operatorFocus in 0.0..1.0)

        val luminanceRisk = when {
            input.underlyingLuminance >= config.highLuminanceThreshold -> 0.85
            input.underlyingLuminance <= config.lowLuminanceThreshold -> 0.65
            else -> abs(input.underlyingLuminance - 0.5) * 0.30 + 0.10
        }

        val roleWeight = when (role) {
            MaterialRole.PRIMARY_NAVIGATION -> 1.00
            MaterialRole.SECONDARY_TELEMETRY -> 0.92
            MaterialRole.ACTION_CLUSTER -> 0.96
            MaterialRole.TIMELINE -> 0.86
            MaterialRole.DIAGNOSTICS -> 0.78
            MaterialRole.MODAL -> 1.05
        }

        val contrastDeficit = (config.minimumContrast - input.contentContrast)
            .coerceAtLeast(0.0)
        val accessibilityBoost = if (input.reduceTransparency) 0.18 else 0.0
        val focusBoost = 0.10 * input.operatorFocus
        val priorityDamp = 0.20 * (1.0 - input.capturePriority)

        val requestedAlpha = (
            0.42 +
                luminanceRisk * 0.12 +
                contrastDeficit * 0.40 +
                accessibilityBoost +
                focusBoost +
                priorityDamp
            ).coerceIn(
                config.minimumTransparency,
                config.maximumTransparency,
            )

        // Reduced-transparency accessibility mode increases opacity without violating
        // the configured ceiling, including on compact or deliberately lightweight surfaces.
        val alpha =
            if (input.reduceTransparency) {
                maxOf(requestedAlpha, minOf(0.52, config.maximumTransparency))
            } else {
                requestedAlpha
            }

        val effectScale = (
            roleWeight *
                (0.82 + 0.18 * input.operatorFocus) *
                (1.0 - 0.12 * luminanceRisk) *
                (0.86 + 0.14 * input.capturePriority)
            ).coerceIn(0.35, 1.0)

        val animationScale = if (input.reduceMotion) {
            config.reducedMotionScale
        } else {
            config.maximumMotionScale * (
                0.78 + 0.22 * input.operatorFocus
            )
        }

        val warnings = buildList {
            if (contrastDeficit > 0.08) add("UNDERLAY_CONTRAST_LOW")
            if (luminanceRisk > 0.75) add("HIGH_LUMINANCE_UNDERLAY")
            if (input.reduceTransparency) add("REDUCED_TRANSPARENCY_ACTIVE")
            if (input.reduceMotion) add("REDUCED_MOTION_ACTIVE")
            if (input.capturePriority < 0.50) add("CAPTURE_PRIORITY_SUPPRESSES_DECORATION")
        }

        return Policy(
            materialAlpha = alpha,
            borderAlpha = (alpha * 0.48 + 0.10).coerceIn(0.10, 0.42),
            specularAlpha = (alpha * 0.22).coerceIn(0.025, 0.16),
            effectScale = effectScale,
            animationScale = animationScale,
            contentContrastBoost = contrastDeficit.coerceIn(0.0, 0.40),
            preserveViewportPriority = true,
            warnings = warnings.distinct(),
        )
    }
}
