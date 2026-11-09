package com.nivukx.nox.core.reconstruction

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Correlation-aware evidence fusion for production motion reconstruction.
 *
 * This engine prevents correlated confidence stacking across geometry, temporal,
 * uncertainty, rig, physics and global-consistency signals. Multiple quality
 * metrics can share the same pixels, clocks, calibration, or detector outputs,
 * so the engine groups dependent signals, limits redundant influence, and
 * exposes effective evidence coverage and common-mode risk.
 */
class CorrelatedEvidenceFusionEngine(
    private var config: Config = Config.production(),
) {
    enum class State { STRONG, GUARDED, DEGRADED, REJECTED }

    enum class DependencyGroup {
        GEOMETRY,
        TEMPORAL,
        UNCERTAINTY,
        RIG,
        PHYSICS,
        GLOBAL,
        IDENTITY,
        DENSE_SURFACE,
        OTHER,
    }

    data class Config(
        val minimumSignalReliability: Double = 0.20,
        val minimumSignalScore: Double = 0.0,
        val maximumSignalsPerGroup: Int = 4,
        val groupRedundancyPenalty: Double = 0.28,
        val commonModePenalty: Double = 0.24,
        val weakSignalPenalty: Double = 0.10,
        val minimumEffectiveEvidence: Double = 3.0,
        val guardedScoreThreshold: Double = 0.72,
        val degradedScoreThreshold: Double = 0.48,
        val rejectScoreThreshold: Double = 0.24,
        val maximumFailureRisk: Double = 0.78,
        val minimumAcceptanceCoverage: Double = 0.50,
        val maximumAcceptanceCommonModeRisk: Double = 0.70,
        val weightFloor: Double = 0.05,
    ) {
        init {
            require(minimumSignalReliability in 0.0..1.0)
            require(minimumSignalScore in 0.0..1.0)
            require(maximumSignalsPerGroup in 1..32)
            require(groupRedundancyPenalty in 0.0..1.0)
            require(commonModePenalty in 0.0..1.0)
            require(weakSignalPenalty in 0.0..1.0)
            require(minimumEffectiveEvidence >= 0.0)
            require(guardedScoreThreshold in 0.0..1.0)
            require(degradedScoreThreshold in 0.0..guardedScoreThreshold)
            require(rejectScoreThreshold in 0.0..degradedScoreThreshold)
            require(maximumFailureRisk in 0.0..1.0)
            require(minimumAcceptanceCoverage in 0.0..1.0)
            require(maximumAcceptanceCommonModeRisk in 0.0..1.0)
            require(weightFloor in 0.0..1.0)
        }

        companion object {
            fun production(): Config = Config(
                minimumSignalReliability = 0.18,
                maximumSignalsPerGroup = 4,
                groupRedundancyPenalty = 0.28,
                commonModePenalty = 0.24,
                weakSignalPenalty = 0.10,
                minimumEffectiveEvidence = 3.0,
                guardedScoreThreshold = 0.74,
                degradedScoreThreshold = 0.50,
                rejectScoreThreshold = 0.25,
                maximumFailureRisk = 0.78,
                weightFloor = 0.05,
            )

            fun qualification(): Config = Config(
                minimumSignalReliability = 0.30,
                maximumSignalsPerGroup = 3,
                groupRedundancyPenalty = 0.36,
                commonModePenalty = 0.30,
                weakSignalPenalty = 0.14,
                minimumEffectiveEvidence = 4.0,
                guardedScoreThreshold = 0.86,
                degradedScoreThreshold = 0.62,
                rejectScoreThreshold = 0.35,
                maximumFailureRisk = 0.58,
                weightFloor = 0.08,
            )
        }
    }

    data class Signal(
        val name: String,
        val score: Double,
        val reliability: Double = 1.0,
        val group: DependencyGroup = DependencyGroup.OTHER,
        val weight: Double = 1.0,
        val hardFailure: Boolean = false,
    ) {
        init { require(name.isNotBlank()) }
    }

    data class GroupEvidence(
        val group: DependencyGroup,
        val score: Double,
        val reliability: Double,
        val contributionWeight: Double,
        val signalCount: Int,
        val retainedSignalCount: Int,
        val redundancyPenalty: Double,
        val commonModePenalty: Double,
        val weakestSignal: String?,
    )

    data class Result(
        val fusedScore: Double,
        val independentEvidenceScore: Double,
        val effectiveEvidenceCount: Double,
        val evidenceCoverage: Double,
        val failureRisk: Double,
        val state: State,
        val hardFailure: Boolean,
        val weakestSignal: String?,
        val strongestFailureFamily: DependencyGroup?,
        val redundantSignalCount: Int,
        val correlatedGroupCount: Int,
        val commonModeRisk: Double,
        val groups: List<GroupEvidence>,
        val warnings: List<String>,
    )

    fun configure(newConfig: Config) { config = newConfig }

    fun configuration(): Config = config

    fun evaluate(signals: List<Signal>): Result {
        if (signals.isEmpty()) {
            return rejected("no-evidence", hardFailure = true)
        }

        val usable = signals.asSequence()
            .map {
                it.copy(
                    score = sanitize(it.score),
                    reliability = sanitize(it.reliability),
                    weight = it.weight.coerceIn(0.0, 8.0),
                )
            }
            .filter { it.reliability >= config.minimumSignalReliability }
            .filter { it.score >= config.minimumSignalScore }
            .toList()

        if (usable.isEmpty()) {
            return Result(
                fusedScore = 0.0,
                independentEvidenceScore = 0.0,
                effectiveEvidenceCount = 0.0,
                evidenceCoverage = 0.0,
                failureRisk = 1.0,
                state = State.REJECTED,
                hardFailure = signals.any { it.hardFailure },
                weakestSignal = signals.minByOrNull { sanitize(it.score) }?.name,
                strongestFailureFamily =
                    signals.firstOrNull { it.hardFailure }?.group,
                redundantSignalCount = 0,
                correlatedGroupCount = 0,
                commonModeRisk = 1.0,
                groups = emptyList(),
                warnings = listOf("insufficient-reliable-evidence"),
            )
        }

        val groups = usable.groupBy { it.group }
            .map { (group, familySignals) ->
                evaluateGroup(group, familySignals)
            }

        val retainedCount = groups.sumOf { it.retainedSignalCount }
        val rawCount = usable.size
        val redundantSignals =
            (rawCount - retainedCount).coerceAtLeast(0)

        val reliabilityMass =
            groups.sumOf { it.contributionWeight * it.reliability }
        val groupWeightSum =
            groups.sumOf { it.contributionWeight }.coerceAtLeast(1e-9)

        val independentScore =
            groups.fold(0.0) { accumulator, group ->
                accumulator + group.score * group.contributionWeight
            } / groupWeightSum

        val commonModeRisk =
            groups.map { it.commonModePenalty * it.reliability }
                .average()
                .coerceIn(0.0, 1.0)

        val redundancyScale =
            (1.0 -
                config.groupRedundancyPenalty *
                    redundantSignals.toDouble() /
                    max(1, rawCount).toDouble())
                .coerceIn(0.55, 1.0)

        val weakSignalScale =
            (1.0 -
                config.weakSignalPenalty *
                    groups.count { it.score < 0.50 }.toDouble() /
                    max(1, groups.size).toDouble())
                .coerceIn(0.65, 1.0)

        val commonModeScale =
            (1.0 - config.commonModePenalty * commonModeRisk)
                .coerceIn(0.55, 1.0)

        val effectiveEvidenceCount = groups.sumOf { group ->
            val diversity =
                1.0 /
                    (1.0 +
                        group.redundancyPenalty *
                            (group.retainedSignalCount - 1)
                                .coerceAtLeast(0))
            group.reliability *
                diversity *
                min(1.0, group.contributionWeight) *
                group.retainedSignalCount.toDouble()
        }

        val coverage =
            (effectiveEvidenceCount /
                max(
                    config.minimumEffectiveEvidence,
                    groups.size.toDouble(),
                ))
                .coerceIn(0.0, 1.0)

        val reliabilityScale =
            (reliabilityMass / groupWeightSum)
                .coerceIn(0.0, 1.0)

        val fused =
            (independentScore *
                (0.72 + 0.28 * reliabilityScale) *
                redundancyScale *
                weakSignalScale *
                commonModeScale)
                .coerceIn(0.0, 1.0)

        val hardFailure =
            signals.any { it.hardFailure } ||
                groups.any {
                    it.score <= config.rejectScoreThreshold &&
                        it.reliability >= 0.70
                }

        val failureRisk = (1.0 - fused).coerceIn(0.0, 1.0)

        val contractFailure =
            coverage < config.minimumAcceptanceCoverage ||
                commonModeRisk >
                config.maximumAcceptanceCommonModeRisk

        val state = when {
            hardFailure ||
                contractFailure ||
                fused < config.rejectScoreThreshold ||
                failureRisk >= config.maximumFailureRisk ->
                State.REJECTED
            fused < config.degradedScoreThreshold ->
                State.DEGRADED
            fused < config.guardedScoreThreshold ->
                State.GUARDED
            else ->
                State.STRONG
        }

        val weakest =
            usable.minByOrNull {
                it.score * it.reliability * max(it.weight, config.weightFloor)
            }

        val strongestFailureFamily =
            groups.filter {
                it.score < config.degradedScoreThreshold
            }.maxByOrNull {
                (1.0 - it.score) * it.reliability
            }?.group

        val warnings = buildList {
            if (
                redundantSignals > 0 ||
                groups.any { it.signalCount > 1 }
            ) {
                add("correlated-signal-redundancy")
            }
            if (commonModeRisk > 0.30) add("common-mode-risk")
            if (coverage < 0.60) add("low-independent-evidence-coverage")
            if (effectiveEvidenceCount < config.minimumEffectiveEvidence) {
                add("effective-evidence-below-contract")
            }
            if (coverage < config.minimumAcceptanceCoverage) {
                add("evidence-coverage-contract-breached")
            }
            if (commonModeRisk > config.maximumAcceptanceCommonModeRisk) {
                add("common-mode-contract-breached")
            }
            when (state) {
                State.GUARDED -> add("evidence-guarded")
                State.DEGRADED -> add("evidence-degraded")
                State.REJECTED -> add("evidence-rejected")
                State.STRONG -> Unit
            }
            strongestFailureFamily?.let {
                add("weak-evidence-family-${it.name.lowercase()}")
            }
            weakest?.let {
                add("weakest-signal-${it.name}")
            }
        }.distinct()

        return Result(
            fusedScore = fused,
            independentEvidenceScore = independentScore,
            effectiveEvidenceCount = effectiveEvidenceCount,
            evidenceCoverage = coverage,
            failureRisk = failureRisk,
            state = state,
            hardFailure = hardFailure,
            weakestSignal = weakest?.name,
            strongestFailureFamily = strongestFailureFamily,
            redundantSignalCount = redundantSignals,
            correlatedGroupCount = groups.count { it.signalCount > 1 },
            commonModeRisk = commonModeRisk,
            groups = groups,
            warnings = warnings,
        )
    }

    private fun evaluateGroup(
        group: DependencyGroup,
        signals: List<Signal>,
    ): GroupEvidence {
        val ordered =
            signals.sortedWith(
                compareByDescending<Signal> {
                    it.score * it.reliability * max(it.weight, config.weightFloor)
                }.thenBy { it.name }
            )

        val retained = ordered.take(config.maximumSignalsPerGroup)
        val discarded = (ordered.size - retained.size).coerceAtLeast(0)

        var weightedLogScore = 0.0
        var totalWeight = 0.0
        retained.forEach { signal ->
            val effectiveWeight =
                max(config.weightFloor, signal.weight * signal.reliability)
            weightedLogScore +=
                ln(signal.score.coerceAtLeast(1e-6)) * effectiveWeight
            totalWeight += effectiveWeight
        }

        val geometricScore =
            if (totalWeight <= 1e-9) {
                0.0
            } else {
                exp(weightedLogScore / totalWeight)
                    .coerceIn(0.0, 1.0)
            }

        val reliability =
            retained.map { it.reliability }
                .average()
                .coerceIn(0.0, 1.0)

        val withinGroupSpread =
            if (retained.size < 2) {
                0.0
            } else {
                val values = retained.map { it.score }
                val mean = values.average()
                sqrt(
                    values.sumOf { (it - mean) * (it - mean) } /
                        values.size.toDouble()
                ).coerceIn(0.0, 1.0)
            }

        val commonMode =
            (withinGroupSpread * 0.25 +
                discarded.toDouble() /
                    max(1, signals.size).toDouble() * 0.35 +
                (1.0 - reliability) * 0.40)
                .coerceIn(0.0, 1.0)

        val redundancyPenalty =
            if (retained.size <= 1) {
                0.0
            } else {
                (1.0 -
                    1.0 /
                        (1.0 +
                            config.groupRedundancyPenalty *
                                (retained.size - 1)))
                    .coerceIn(0.0, 1.0)
            }

        return GroupEvidence(
            group = group,
            score = geometricScore,
            reliability = reliability,
            contributionWeight =
                (0.55 + 0.45 * reliability) *
                    (1.0 - redundancyPenalty * 0.45)
                        .coerceAtLeast(0.05),
            signalCount = signals.size,
            retainedSignalCount = retained.size,
            redundancyPenalty = redundancyPenalty,
            commonModePenalty = commonMode,
            weakestSignal =
                retained.minByOrNull {
                    it.score * it.reliability
                }?.name,
        )
    }

    private fun rejected(
        warning: String,
        hardFailure: Boolean,
    ): Result =
        Result(
            fusedScore = 0.0,
            independentEvidenceScore = 0.0,
            effectiveEvidenceCount = 0.0,
            evidenceCoverage = 0.0,
            failureRisk = 1.0,
            state = State.REJECTED,
            hardFailure = hardFailure,
            weakestSignal = null,
            strongestFailureFamily = null,
            redundantSignalCount = 0,
            correlatedGroupCount = 0,
            commonModeRisk = 1.0,
            groups = emptyList(),
            warnings = listOf(warning),
        )

    private fun sanitize(value: Double): Double =
        if (value.isFinite()) value.coerceIn(0.0, 1.0) else 0.0
}
