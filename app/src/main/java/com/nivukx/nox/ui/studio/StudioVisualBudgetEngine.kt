package com.nivukx.nox.ui.studio

/**
 * Runtime visual-budget policy for the motion-capture operator surface.
 *
 * The studio is a real-time control surface, not a wallpaper engine. Material effects,
 * diagnostics density and decorative rendering must yield GPU time to the capture pipeline
 * when runtime pressure rises.
 */
class StudioVisualBudgetEngine(
    private val config: Config = Config.production(),
) {
    enum class Tier {
        FULL,
        ADAPTIVE,
        REDUCED,
        MINIMAL,
    }

    enum class TelemetryDensity {
        DENSE,
        NORMAL,
        COMPACT,
        CRITICAL,
    }

    data class Config(
        val fullGpuLatencyMs: Double = 10.0,
        val adaptiveGpuLatencyMs: Double = 16.0,
        val reducedGpuLatencyMs: Double = 24.0,
        val fullHealth: Double = 0.80,
        val adaptiveHealth: Double = 0.65,
        val reducedHealth: Double = 0.45,
        val minimumEffectScale: Float = 0.38f,
        val maximumEffectScale: Float = 1.0f,
    ) {
        init {
            require(fullGpuLatencyMs > 0.0)
            require(adaptiveGpuLatencyMs >= fullGpuLatencyMs)
            require(reducedGpuLatencyMs >= adaptiveGpuLatencyMs)
            require(fullHealth in 0.0..1.0)
            require(adaptiveHealth in 0.0..1.0)
            require(reducedHealth in 0.0..1.0)
            require(minimumEffectScale in 0.0f..1.0f)
            require(maximumEffectScale in 0.0f..1.0f)
            require(maximumEffectScale >= minimumEffectScale)
        }

        companion object {
            fun production(): Config = Config()
        }
    }

    data class Input(
        val gpuLatencyMs: Double,
        val sessionHealth: Double,
        val captureIntegrity: Double,
        val operatorPriority: Double = 1.0,
    ) {
        init {
            require(gpuLatencyMs >= 0.0 || gpuLatencyMs.isNaN())
            require(sessionHealth in 0.0..1.0)
            require(captureIntegrity in 0.0..1.0)
            require(operatorPriority in 0.0..1.0)
        }
    }

    data class Budget(
        val tier: Tier,
        val telemetryDensity: TelemetryDensity,
        val effectScale: Float,
        val shadowScale: Float,
        val animationScale: Float,
        val diagnosticRefreshHz: Double,
        val preserveViewportDetail: Boolean,
        val warnings: List<String>,
    )

    fun evaluate(input: Input): Budget {
        val latency =
            if (input.gpuLatencyMs.isFinite()) {
                input.gpuLatencyMs
            } else {
                config.reducedGpuLatencyMs
            }

        val health =
            minOf(
                input.sessionHealth,
                input.captureIntegrity,
            )

        val stress =
            (
                maxOf(
                    0.0,
                    (latency - config.fullGpuLatencyMs) /
                        (
                            config.reducedGpuLatencyMs -
                                config.fullGpuLatencyMs
                        ).coerceAtLeast(1e-6),
                ) * 0.62 +
                    (1.0 - health) * 0.38
                ).coerceIn(0.0, 1.0)

        val tier =
            when {
                latency > config.reducedGpuLatencyMs ||
                    health < config.reducedHealth ->
                    Tier.MINIMAL
                latency > config.adaptiveGpuLatencyMs ||
                    health < config.adaptiveHealth ->
                    Tier.REDUCED
                latency > config.fullGpuLatencyMs ||
                    health < config.fullHealth ->
                    Tier.ADAPTIVE
                else ->
                    Tier.FULL
            }

        val targetScale =
            when (tier) {
                Tier.FULL -> 1.0f
                Tier.ADAPTIVE -> 0.84f
                Tier.REDUCED -> 0.66f
                Tier.MINIMAL -> 0.44f
            }

        val effectScale =
            (
                targetScale -
                    stress.toFloat() * 0.08f +
                    input.operatorPriority.toFloat() * 0.02f
                ).coerceIn(
                    config.minimumEffectScale,
                    config.maximumEffectScale,
                )

        val telemetryDensity =
            when (tier) {
                Tier.FULL -> TelemetryDensity.DENSE
                Tier.ADAPTIVE -> TelemetryDensity.NORMAL
                Tier.REDUCED -> TelemetryDensity.COMPACT
                Tier.MINIMAL -> TelemetryDensity.CRITICAL
            }

        val refreshHz =
            when (tier) {
                Tier.FULL -> 30.0
                Tier.ADAPTIVE -> 20.0
                Tier.REDUCED -> 12.0
                Tier.MINIMAL -> 6.0
            }

        val warnings = buildList {
            if (tier != Tier.FULL) {
                add("visual-effects-adapted-to-capture-pressure")
            }
            if (tier == Tier.MINIMAL) {
                add("operator-ui-entered-minimal-rendering")
            }
            if (latency > config.reducedGpuLatencyMs) {
                add("gpu-latency-critical")
            }
            if (health < config.reducedHealth) {
                add("capture-health-critical")
            }
        }.distinct()

        return Budget(
            tier = tier,
            telemetryDensity = telemetryDensity,
            effectScale = effectScale,
            shadowScale =
                (
                    0.35f + 0.65f * effectScale
                ).coerceIn(0.25f, 1.0f),
            animationScale =
                (
                    0.45f + 0.55f * effectScale
                ).coerceIn(0.30f, 1.0f),
            diagnosticRefreshHz = refreshHz,
            preserveViewportDetail = true,
            warnings = warnings,
        )
    }
}
