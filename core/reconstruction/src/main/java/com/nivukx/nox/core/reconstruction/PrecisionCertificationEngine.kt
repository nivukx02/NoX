package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Practical precision certification layer.
 *
 * MetricPrecisionBudgetEngine models independent error contributions. Real capture also contains
 * correlated and regime-dependent error: clock skew can correlate across cameras, rolling shutter
 * can correlate along a body, and occlusion can make the visible subset systematically different.
 *
 * This engine therefore converts the metrology budget into a conservative certification decision.
 * It never converts a modelled 2 mm budget into a claim of 2 mm real-world accuracy. Certification
 * requires empirical evidence collected under the same capture regime.
 */
class PrecisionCertificationEngine {

    enum class Grade {
        CERTIFIED_2MM,
        EMPIRICALLY_READY,
        HIGH_PRECISION,
        PRODUCTION,
        BLOCKED,
    }

    data class Input(
        val metrology: MetricPrecisionBudgetEngine.Decision,
        val empiricalEvidenceEnabled: Boolean = false,
        val empiricalSampleCount: Int = 0,
        val empiricalP95Millimeters: Double? = null,
        val empiricalP99Millimeters: Double? = null,
        val empiricalRepeatabilityMillimeters: Double = 0.0,
        val empiricalOccludedP95Millimeters: Double = 0.0,
        val empiricalLowVisibilityP95Millimeters: Double = 0.0,
        val calibrationConfidence: Double = 1.0,
        val synchronizationConfidence: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val identityContinuity: Double = 1.0,
        val temporalLagMilliseconds: Double = 0.0,
        val dropoutRatio: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val metricScaleDriftPercentPerMinute: Double = 0.0,
        val thermalRisk: Double = 0.0,
        val memoryPressure: Double = 0.0,
        val surfaceTemporalStability: Double = 1.0,
        val referenceUncertaintyMillimeters: Double? = null,
        val referenceConditionsMatched: Boolean = false,
        val metrologyEvidence:
            HighEndMetrologyBenchmarkEngine.Report? = null,
    ) {
        init {
            require(empiricalSampleCount >= 0)
            listOf(
                calibrationConfidence,
                synchronizationConfidence,
                viewDiversity,
                crossViewAgreement,
                identityContinuity,
                dropoutRatio,
                rollingShutterRisk,
                occlusionRisk,
                thermalRisk,
                memoryPressure,
                surfaceTemporalStability,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            listOf(
                empiricalRepeatabilityMillimeters,
                empiricalOccludedP95Millimeters,
                empiricalLowVisibilityP95Millimeters,
                temporalLagMilliseconds,
                metricScaleDriftPercentPerMinute,
            ).forEach { require(it.isFinite() && it >= 0.0) }
            empiricalP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            empiricalP99Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            referenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class ErrorBudget(
        val modeledP95Millimeters: Double,
        val modeledP99Millimeters: Double,
        val correlationInflationMillimeters: Double,
        val repeatabilityMillimeters: Double,
        val occlusionInflationMillimeters: Double,
        val visibilityInflationMillimeters: Double,
        val driftInflationMillimeters: Double,
        val conservativeP95Millimeters: Double,
        val conservativeP99Millimeters: Double,
    )

    data class Decision(
        val grade: Grade,
        val modeledOnly: Boolean,
        val empiricalEvidenceSufficient: Boolean,
        val subMillimeterEvidenceSufficient: Boolean,
        val referenceGradeFeasible: Boolean,
        val hardBlock: Boolean,
        val certificationReady: Boolean,
        val score: Double,
        val errorBudget: ErrorBudget,
        val blockers: List<String>,
        val warnings: List<String>,
    )

    fun evaluate(input: Input): Decision {
        val blockers = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val m = input.metrology
        val metrology = input.metrologyEvidence

        val empiricalEnabled =
            input.empiricalEvidenceEnabled ||
                metrology != null
        val empiricalCount =
            if (input.empiricalSampleCount > 0) {
                input.empiricalSampleCount
            } else {
                metrology?.aggregate?.sampleCount ?: 0
            }
        val empiricalP95 =
            input.empiricalP95Millimeters
                ?: metrology?.aggregate?.p95ErrorMm
        val empiricalP99 =
            input.empiricalP99Millimeters
                ?: metrology?.aggregate?.p99ErrorMm
        val empiricalRepeatability =
            if (input.empiricalRepeatabilityMillimeters > 0.0) {
                input.empiricalRepeatabilityMillimeters
            } else {
                metrology?.aggregate?.repeatabilityMm ?: 0.0
            }
        val referenceUncertainty =
            input.referenceUncertaintyMillimeters
                ?: metrology?.aggregate?.referenceUncertaintyMm

        val referenceConditionsMatched =
            input.referenceConditionsMatched ||
                metrology?.certification?.matchedConditions == true

        if (m.hardBlock) blockers += m.blockers.map { "metrology:$it" }
        metrology?.certification?.warnings
            ?.take(8)
            ?.forEach { warnings += "metrology:" + it }
        if (input.calibrationConfidence < 0.92) warnings += "calibration-below-reference"
        if (input.synchronizationConfidence < 0.92) warnings += "synchronization-below-reference"
        if (input.viewDiversity < 0.65) warnings += "view-diversity-below-reference"
        if (input.crossViewAgreement < 0.75) warnings += "cross-view-agreement-below-reference"
        if (input.identityContinuity < 0.90) warnings += "identity-continuity-below-reference"
        if (input.temporalLagMilliseconds > 16.7) warnings += "temporal-lag-above-60fps-frame"
        if (input.dropoutRatio > 0.02) warnings += "dropout-above-reference"
        if (input.rollingShutterRisk > 0.35) warnings += "rolling-shutter-risk-high"
        if (input.occlusionRisk > 0.45) warnings += "occlusion-regime-high"
        if (input.metricScaleDriftPercentPerMinute > 0.10) {
            warnings += "metric-scale-drift-above-reference"
        }
        if (input.thermalRisk > 0.65) {
            warnings += "thermal-risk-above-reference"
        }
        if (input.memoryPressure > 0.82) {
            warnings += "memory-pressure-above-reference"
        }
        if (input.surfaceTemporalStability < 0.90) {
            warnings += "surface-temporal-stability-below-reference"
        }

        val correlation =
            correlatedInflation(
                input = input,
                empiricalP95 = empiricalP95,
            )
        val occlusion = max(
            0.0,
            (
                if (input.empiricalOccludedP95Millimeters > 0.0) {
                    input.empiricalOccludedP95Millimeters
                } else {
                    metrology?.regimeMetrics
                        ?.filter { it.key.startsWith("occlusion:") }
                        ?.maxOfOrNull { it.p95Mm }
                        ?: 0.0
                }
            ) -
                empiricalP95.orZero(),
        ) * 0.50
        val visibility = max(
            0.0,
            (
                if (input.empiricalLowVisibilityP95Millimeters > 0.0) {
                    input.empiricalLowVisibilityP95Millimeters
                } else {
                    metrology?.regimeMetrics
                        ?.filter { it.key.startsWith("lighting:") }
                        ?.maxOfOrNull { it.p95Mm }
                        ?: 0.0
                }
            ) -
                empiricalP95.orZero(),
        ) * 0.35
        val drift = input.metricScaleDriftPercentPerMinute *
            max(1.0, m.contribution.scaleMillimeters) * 0.50

        val conservativeP95 = sqrt(
            m.modeledP95ErrorMillimeters.squared() +
                correlation.squared() +
                empiricalRepeatability.squared() +
                occlusion.squared() +
                visibility.squared() +
                drift.squared(),
        )

        val conservativeP99 = sqrt(
            m.modeledP99ErrorMillimeters.squared() +
                (correlation * 1.35).squared() +
                (empiricalRepeatability * 1.35).squared() +
                (occlusion * 1.20).squared() +
                (visibility * 1.20).squared() +
                (drift * 1.25).squared(),
        )

        val budget = ErrorBudget(
            modeledP95Millimeters = m.modeledP95ErrorMillimeters,
            modeledP99Millimeters = m.modeledP99ErrorMillimeters,
            correlationInflationMillimeters = correlation,
            repeatabilityMillimeters = empiricalRepeatability,
            occlusionInflationMillimeters = occlusion,
            visibilityInflationMillimeters = visibility,
            driftInflationMillimeters = drift,
            conservativeP95Millimeters = conservativeP95,
            conservativeP99Millimeters = conservativeP99,
        )

        val empiricalSufficient =
            empiricalEnabled &&
                empiricalCount >= 120 &&
                empiricalP95 != null &&
                empiricalP99 != null &&
                referenceConditionsMatched &&
                referenceUncertainty != null &&
                (
                    metrology == null ||
                        metrology.reproducibility.sessionCount >= 3
                    )

        val subMillimeterEvidenceSufficient =
            (
                if (metrology != null) {
                    metrology.certification.oneMillimeterEvidence
                } else {
                    empiricalEnabled &&
                        empiricalCount >= 240 &&
                        empiricalP95?.let { it < 1.0 } == true &&
                        empiricalP99?.let { it <= 2.5 } == true &&
                        empiricalRepeatability <= 0.50 &&
                        input.empiricalOccludedP95Millimeters <= 1.25 &&
                        input.empiricalLowVisibilityP95Millimeters <= 1.25 &&
                        referenceConditionsMatched &&
                        referenceUncertainty?.let { it <= 0.25 } == true
                }
            )

        if (empiricalEnabled && empiricalCount < 120) {
            warnings += "empirical-sample-count-below-certification-minimum"
        }
        if (empiricalEnabled &&
            empiricalCount in 120 until 240
        ) {
            warnings += "sub-millimeter-sample-count-below-240"
        }
        if (empiricalEnabled && !input.referenceConditionsMatched) {
            warnings += "reference-conditions-not-matched"
        }
        if (empiricalEnabled &&
            referenceUncertainty == null
        ) {
            warnings += "reference-uncertainty-missing"
        }

        if (empiricalSufficient &&
            (empiricalP95!! > 2.0 || empiricalP99!! > 5.0)
        ) {
            blockers += "empirical-error-exceeds-2mm-reference-grade-budget"
        }

        val referenceGradeFeasible =
            m.referenceGradeFeasible &&
                conservativeP95 <= 2.0 &&
                conservativeP99 <= 5.0

        if (subMillimeterEvidenceSufficient) {
            warnings += "sub-millimeter-evidence-qualified-under-matched-reference"
        } else if (
            empiricalEnabled &&
            empiricalCount >= 240 &&
            empiricalP95 != null &&
            empiricalP95 >= 1.0
        ) {
            warnings += "sub-millimeter-p95-target-not-met"
        }

        if (m.referenceGradeFeasible && conservativeP95 > 2.0) {
            warnings += "correlated-error-erodes-2mm-budget"
        }

        val score = (
            m.feasibilityScore * 0.28 +
                inversePenalty(conservativeP95, 5.0) * 0.22 +
                inversePenalty(conservativeP99, 12.0) * 0.10 +
                input.calibrationConfidence * 0.10 +
                input.synchronizationConfidence * 0.10 +
                input.viewDiversity * 0.06 +
                input.crossViewAgreement * 0.06 +
                input.identityContinuity * 0.04 +
                (1.0 - input.dropoutRatio) * 0.04 +
                (1.0 - input.thermalRisk) * 0.03 +
                (1.0 - input.memoryPressure) * 0.03 +
                input.surfaceTemporalStability * 0.03
        ).coerceIn(0.0, 1.0)

        val certificationReady =
            empiricalSufficient &&
                empiricalP95!! <= 2.0 &&
                empiricalP99!! <= 5.0 &&
                empiricalRepeatability <= 1.0 &&
                referenceConditionsMatched &&
                input.thermalRisk <= 0.65 &&
                input.memoryPressure <= 0.82 &&
                input.surfaceTemporalStability >= 0.90 &&
                (referenceUncertainty ?: Double.POSITIVE_INFINITY) <= 0.5

        val grade = when {
            blockers.isNotEmpty() -> Grade.BLOCKED
            certificationReady && referenceGradeFeasible -> Grade.CERTIFIED_2MM
            empiricalSufficient && referenceGradeFeasible -> Grade.EMPIRICALLY_READY
            conservativeP95 <= 5.0 && score >= 0.75 -> Grade.HIGH_PRECISION
            conservativeP95 <= 12.0 && score >= 0.55 -> Grade.PRODUCTION
            else -> Grade.BLOCKED
        }

        return Decision(
            grade = grade,
            modeledOnly = !empiricalSufficient,
            empiricalEvidenceSufficient = empiricalSufficient,
            subMillimeterEvidenceSufficient =
                subMillimeterEvidenceSufficient,
            referenceGradeFeasible = referenceGradeFeasible,
            hardBlock = blockers.isNotEmpty(),
            certificationReady = certificationReady,
            score = score,
            errorBudget = budget,
            blockers = blockers.distinct(),
            warnings = warnings.distinct(),
        )
    }

    private fun correlatedInflation(
        input: Input,
        empiricalP95: Double?,
    ): Double {
        val timingCorrelation =
            input.temporalLagMilliseconds *
                input.rollingShutterRisk * 0.12
        val occlusionCorrelation =
            input.occlusionRisk *
                empiricalP95.orZero() * 0.18
        val rigCorrelation =
            (1.0 - input.crossViewAgreement) *
                max(1.0, empiricalP95.orZero()) * 0.22
        val driftCorrelation =
            input.metricScaleDriftPercentPerMinute * 0.40
        return sqrt(
            timingCorrelation.squared() +
                occlusionCorrelation.squared() +
                rigCorrelation.squared() +
                driftCorrelation.squared(),
        )
    }

    private fun inversePenalty(value: Double, limit: Double): Double =
        (1.0 - value / max(limit, 0.001)).coerceIn(0.0, 1.0)

    private fun Double.squared(): Double = this * this
    private fun Double?.orZero(): Double = this ?: 0.0
}
