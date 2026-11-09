package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Correlation-aware uncertainty propagation for live motion-capture precision admission.
 *
 * The engine treats each supplied component as an already-calibrated one-sigma or
 * radial uncertainty quantity. It does not convert confidence into distance. Correlation
 * terms prevent the common mistake of linearly adding dependent error sources while still
 * preserving a conservative penalty for poor observability and temporal integrity.
 */
class CorrelatedUncertaintyBudgetEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val radialP95Multiplier: Double = 2.7954834829,
        val scalarP95Multiplier: Double = 1.9599639845,
        val minimumVariance: Double = 1e-12,
        val maximumCorrelationMagnitude: Double = 0.98,
        val poorEvidenceInflationLimit: Double = 4.0,
    ) {
        init {
            require(radialP95Multiplier > 0.0)
            require(scalarP95Multiplier > 0.0)
            require(minimumVariance > 0.0)
            require(maximumCorrelationMagnitude in 0.5..0.999)
            require(poorEvidenceInflationLimit >= 1.0)
        }
    }

    enum class Component {
        RECONSTRUCTION,
        DIRECTIONAL,
        GEOMETRIC,
        SURFACE,
        MOTION,
        TEMPORAL,
    }

    data class Correlation(
        val first: Component,
        val second: Component,
        val coefficient: Double,
    ) {
        init {
            require(first != second)
            require(coefficient.isFinite())
        }
    }

    data class Input(
        val reconstructionMm: Double,
        val directionalMm: Double,
        val geometricMm: Double,
        val surfaceMm: Double,
        val motionMm: Double,
        val temporalMm: Double,
        val correlations: List<Correlation> = defaultCorrelations(),
        val observability: Double = 1.0,
        val calibration: Double = 1.0,
        val synchronization: Double = 1.0,
        val temporalAlignment: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val captureTiming: Double = 1.0,
        val occlusionRisk: Double = 0.0,
    ) {
        init {
            listOf(
                reconstructionMm, directionalMm, geometricMm, surfaceMm,
                motionMm, temporalMm, observability, calibration, synchronization,
                temporalAlignment, crossViewAgreement, viewDiversity, captureTiming, occlusionRisk,
            ).forEach { require(it.isFinite()) }
            listOf(
                observability, calibration, synchronization, temporalAlignment,
                crossViewAgreement, viewDiversity, captureTiming, occlusionRisk,
            ).forEach { require(it in 0.0..1.0) }
            listOf(reconstructionMm, directionalMm, geometricMm, surfaceMm, motionMm, temporalMm)
                .forEach { require(it >= 0.0) }
        }

        companion object {
            fun defaultCorrelations(): List<Correlation> = listOf(
                Correlation(Component.RECONSTRUCTION, Component.GEOMETRIC, 0.35),
                Correlation(Component.RECONSTRUCTION, Component.SURFACE, 0.30),
                Correlation(Component.DIRECTIONAL, Component.GEOMETRIC, 0.45),
                Correlation(Component.GEOMETRIC, Component.SURFACE, 0.40),
                Correlation(Component.MOTION, Component.TEMPORAL, 0.35),
                Correlation(Component.RECONSTRUCTION, Component.TEMPORAL, 0.20),
            )
        }
    }

    data class Result(
        val oneSigmaMillimeters: Double,
        val scalarP95Millimeters: Double,
        val radialP95Millimeters: Double,
        val evidenceInflation: Double,
        val finalP95Millimeters: Double,
        val effectiveObservability: Double,
        val dominantComponent: Component,
        val componentOneSigmaMillimeters: Map<Component, Double>,
        val correlationsApplied: Int,
        val warnings: List<String>,
    )

    fun combine(input: Input): Result {
        val components = linkedMapOf(
            Component.RECONSTRUCTION to input.reconstructionMm,
            Component.DIRECTIONAL to input.directionalMm,
            Component.GEOMETRIC to input.geometricMm,
            Component.SURFACE to input.surfaceMm,
            Component.MOTION to input.motionMm,
            Component.TEMPORAL to input.temporalMm,
        )
        val variances = components.mapValues { (_, value) ->
            max(config.minimumVariance, value * value)
        }.toMutableMap()

        var covarianceContribution = 0.0
        var appliedCorrelations = 0
        val warnings = mutableListOf<String>()

        val seenCorrelationPairs =
            HashSet<String>()
        for (correlation in input.correlations) {
            val pairKey =
                canonicalCorrelationKey(
                    correlation.first,
                    correlation.second,
                )
            if (!seenCorrelationPairs.add(pairKey)) {
                warnings +=
                    "DUPLICATE_CORRELATION_PAIR_IGNORED:" + pairKey
                continue
            }

            val unclampedCoefficient =
                correlation.coefficient
            val coefficient =
                unclampedCoefficient.coerceIn(
                    -config.maximumCorrelationMagnitude,
                    config.maximumCorrelationMagnitude,
                )
            if (
                coefficient !=
                    unclampedCoefficient
            ) {
                warnings +=
                    "CORRELATION_COEFFICIENT_CLAMPED:" +
                        pairKey
            }

            val a =
                components.getValue(
                    correlation.first,
                )
            val b =
                components.getValue(
                    correlation.second,
                )
            covarianceContribution +=
                2.0 * coefficient * a * b
            appliedCorrelations += 1
        }

        val independentVariance = variances.values.sum()
        val rawVariance = max(
            config.minimumVariance,
            independentVariance + covarianceContribution,
        )
        val effectiveObservability = listOf(
            input.observability, input.calibration, input.synchronization,
            input.temporalAlignment, input.crossViewAgreement, input.viewDiversity,
            input.captureTiming, 1.0 - input.occlusionRisk,
        ).minOrNull()?.coerceIn(0.01, 1.0) ?: 0.01

        val evidenceInflation = (
            1.0 +
                (1.0 - input.observability) * 0.75 +
                (1.0 - input.calibration) * 0.55 +
                (1.0 - input.synchronization) * 0.55 +
                (1.0 - input.temporalAlignment) * 0.45 +
                (1.0 - input.crossViewAgreement) * 0.40 +
                (1.0 - input.viewDiversity) * 0.40 +
                (1.0 - input.captureTiming) * 0.25 +
                input.occlusionRisk * 0.55
            ).coerceIn(1.0, config.poorEvidenceInflationLimit)

        if (covarianceContribution < 0.0) {
            val conservativeFloor = independentVariance * 0.20
            if (rawVariance < conservativeFloor) {
                warnings += "NEGATIVE_CORRELATION_FLOOR_APPLIED"
            }
        }
        if (effectiveObservability < 0.60) warnings += "LOW_EFFECTIVE_OBSERVABILITY"
        if (input.occlusionRisk > 0.50) warnings += "HIGH_OCCLUSION_RISK_INFLATES_PRECISION_BUDGET"

        val oneSigma = sqrt(rawVariance)
        val scalarP95 = oneSigma * config.scalarP95Multiplier
        val radialP95 = oneSigma * config.radialP95Multiplier
        val finalP95 = radialP95 * evidenceInflation
        val dominant = components.maxByOrNull { it.value }?.key ?: Component.RECONSTRUCTION

        return Result(
            oneSigmaMillimeters = oneSigma,
            scalarP95Millimeters = scalarP95,
            radialP95Millimeters = radialP95,
            evidenceInflation = evidenceInflation,
            finalP95Millimeters = finalP95,
            effectiveObservability = effectiveObservability,
            dominantComponent = dominant,
            componentOneSigmaMillimeters = components.toMap(),
            correlationsApplied = appliedCorrelations,
            warnings = warnings.distinct(),
        )
    }
    
    private fun canonicalCorrelationKey(
        first: Component,
        second: Component,
    ): String =
        if (first.ordinal <= second.ordinal) {
            first.name + "|" + second.name
        } else {
            second.name + "|" + first.name
        }

    data class Contributor(
        val source: Source,
        val sigmaMeters: Double,
        val confidence: Double,
    ) {
        init {
            require(sigmaMeters.isFinite() && sigmaMeters >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    enum class Source {
        TRIANGULATION,
        REPROJECTION,
        TEMPORAL,
        CALIBRATION,
        ROLLING_SHUTTER,
        OCCLUSION,
        SCALE,
    }

    data class FusedResult(
        val fusedSigmaMeters: Double,
        val independenceScore: Double,
        val correlationPenalty: Double,
    )

    fun fuse(contributors: List<Contributor>): FusedResult {
        if (contributors.isEmpty()) {
            return FusedResult(
                fusedSigmaMeters = 0.0,
                independenceScore = 1.0,
                correlationPenalty = 1.0,
            )
        }

        val distinctSourceCount = contributors.map { it.source }.distinct().size
        val independenceScore =
            distinctSourceCount.toDouble() /
                contributors.size.toDouble()
        val weightedVariance =
            contributors.sumOf { contributor ->
                val confidence =
                    contributor.confidence.coerceAtLeast(0.05)
                contributor.sigmaMeters * contributor.sigmaMeters /
                    confidence
            }.coerceAtLeast(config.minimumVariance)
        val averageConfidence =
            contributors.map { it.confidence }.average()
                .coerceIn(0.05, 1.0)
        val correlationPenalty = (
            1.0 +
                (1.0 - independenceScore) * 0.35 +
                (1.0 - averageConfidence) * 0.15
            ).coerceAtMost(config.poorEvidenceInflationLimit)

        return FusedResult(
            fusedSigmaMeters =
                sqrt(weightedVariance) * correlationPenalty,
            independenceScore =
                independenceScore.coerceIn(0.0, 1.0),
            correlationPenalty = correlationPenalty,
        )
    }

}