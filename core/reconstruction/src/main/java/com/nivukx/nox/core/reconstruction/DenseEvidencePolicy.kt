package com.nivukx.nox.core.reconstruction

import kotlin.math.max

/**
 * Evidence weighting rules shared by dense multi-view reconstruction.
 *
 * This policy deliberately distinguishes local detector confidence from cross-view agreement.
 * A single camera can report confidence, visibility and temporal stability, but it cannot create
 * cross-view corroboration by itself.
 */
object DenseEvidencePolicy {

    fun crossViewAgreement(
        rawAgreement: Double,
        acceptedViewCount: Int,
    ): Double =
        if (acceptedViewCount < 2) {
            0.0
        } else {
            rawAgreement
                .takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0
        }

    fun temporalAlignmentConfidence(
        temporalStability: Double,
        predicted: Boolean,
    ): Double {
        val stability =
            temporalStability
                .takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        return (
            stability *
                if (predicted) 0.72 else 1.0
            ).coerceIn(0.0, 1.0)
    }

    fun observationWeight(
        confidence: Double,
        visibility: Double,
        uncertainty: Double,
        temporalStability: Double,
        crossViewAgreement: Double,
    ): Double {
        val c = confidence.finiteClamped()
        val v = visibility.finiteClamped()
        val u = uncertainty.finiteClamped()
        val t = temporalStability.finiteClamped()
        val x = crossViewAgreement.finiteClamped()

        val uncertaintyQuality = 1.0 - u

        return (
            c * 0.25 +
                v * 0.20 +
                uncertaintyQuality * 0.25 +
                t * 0.15 +
                x * 0.15
            ).coerceIn(0.0, 1.0)
    }

    fun combineViewSupport(
        acceptedViewCount: Int,
        distinctSourceCount: Int,
        independentViewCount: Int,
    ): Double {
        val accepted =
            max(0, acceptedViewCount)
        val distinct =
            max(0, distinctSourceCount)
        val independent =
            max(0, independentViewCount)

        if (accepted <= 0) return 0.0

        val acceptedScore =
            (accepted.toDouble() / 4.0).coerceIn(0.0, 1.0)
        val sourceScore =
            (distinct.toDouble() / 4.0).coerceIn(0.0, 1.0)
        val independentScore =
            (independent.toDouble() / 4.0).coerceIn(0.0, 1.0)

        return (
            acceptedScore * 0.35 +
                sourceScore * 0.25 +
                independentScore * 0.40
            ).coerceIn(0.0, 1.0)
    }

    private fun Double.finiteClamped(): Double =
        takeIf { it.isFinite() }
            ?.coerceIn(0.0, 1.0)
            ?: 0.0
}
