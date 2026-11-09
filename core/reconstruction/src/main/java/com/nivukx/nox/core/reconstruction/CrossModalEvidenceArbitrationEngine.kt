package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production cross-modal evidence arbiter.
 *
 * Multimodal capture must not assume that every sensor-derived stream is beneficial on every
 * frame. This engine evaluates each body/face/hand/eye modality independently, maintains a
 * bounded temporal reliability prior, quarantines persistently pathological streams, penalizes
 * redundant evidence, and greedily selects the subset with the highest marginal utility.
 *
 * The arbiter never fabricates observations. It only decides how much influence already measured
 * evidence should receive downstream.
 *
 * Design goals:
 * - reliability is estimated before selection using only evidence that existed previously;
 * - prediction-only evidence cannot train reliability state;
 * - OOD, uncertainty, staleness, occlusion and prediction reduce trust;
 * - redundancy reduces marginal utility without automatically deleting a modality;
 * - a small set of independent strong modalities can produce a stronger fused signal than any one;
 * - all decisions remain deterministic and replayable from the same input sequence;
 * - state is bounded by the fixed modality enum and contains no unbounded per-frame history.
 */
class CrossModalEvidenceArbitrationEngine(
    private var config: Config = Config(),
) {

    enum class Modality(
        val defaultImportance: Double,
    ) {
        BODY(0.34),
        FACE(0.20),
        LEFT_HAND(0.14),
        RIGHT_HAND(0.14),
        EYE_LEFT(0.09),
        EYE_RIGHT(0.09),
    }

    data class Config(
        val minimumSelectedModalities: Int = 1,
        val maximumSelectedModalities: Int = 4,
        val minimumConfidence: Double = 0.12,
        val minimumAdjustedReliability: Double = 0.18,
        val staleEvidenceSeconds: Double = 0.16,
        val maximumEvidenceAgeSeconds: Double = 0.60,
        val oodQuarantineThreshold: Double = 0.82,
        val lowReliabilityQuarantineThreshold: Double = 0.22,
        val quarantineAfterBadFrames: Int = 3,
        val recoveryFrames: Int = 4,
        val recoveryOodThreshold: Double = 0.34,
        val historyAlpha: Double = 0.22,
        val qualityAlpha: Double = 0.62,
        val temporalPriorAlpha: Double = 0.38,
        val uncertaintyScaleMeters: Double = 0.06,
        val uncertaintyWeight: Double = 0.34,
        val freshnessWeight: Double = 0.16,
        val oodWeight: Double = 0.28,
        val predictionPenalty: Double = 0.10,
        val occlusionPenalty: Double = 0.08,
        val redundancyPenaltyWeight: Double = 0.58,
        val marginalSelectionThreshold: Double = 0.10,
        val noisyOrWeightFloor: Double = 0.04,
        val quarantinedWeight: Double = 0.025,
        val minimumFusionConfidence: Double = 0.20,
    ) {
        init {
            require(minimumSelectedModalities in 1..Modality.entries.size)
            require(maximumSelectedModalities in minimumSelectedModalities..Modality.entries.size)
            require(minimumConfidence in 0.0..1.0)
            require(minimumAdjustedReliability in 0.0..1.0)
            require(staleEvidenceSeconds > 0.0)
            require(maximumEvidenceAgeSeconds >= staleEvidenceSeconds)
            require(oodQuarantineThreshold in 0.0..1.0)
            require(lowReliabilityQuarantineThreshold in 0.0..1.0)
            require(quarantineAfterBadFrames >= 1)
            require(recoveryFrames >= 1)
            require(recoveryOodThreshold in 0.0..1.0)
            require(historyAlpha in 0.0..1.0)
            require(qualityAlpha in 0.0..1.0)
            require(temporalPriorAlpha in 0.0..1.0)
            require(uncertaintyScaleMeters > 0.0)
            require(uncertaintyWeight >= 0.0)
            require(freshnessWeight >= 0.0)
            require(oodWeight >= 0.0)
            require(predictionPenalty in 0.0..1.0)
            require(occlusionPenalty in 0.0..1.0)
            require(redundancyPenaltyWeight >= 0.0)
            require(marginalSelectionThreshold >= 0.0)
            require(noisyOrWeightFloor > 0.0)
            require(quarantinedWeight in 0.0..1.0)
            require(minimumFusionConfidence in 0.0..1.0)
        }
    }

    data class ModalityEvidence(
        val modality: Modality,
        val confidence: Double,
        val uncertaintyMeters: Double,
        val redundancy: Double = 0.0,
        val freshnessSeconds: Double = 0.0,
        val oodRisk: Double = 0.0,
        val predicted: Boolean = false,
        val predictionOnly: Boolean = false,
        val occluded: Boolean = false,
    ) {
        init {
            require(confidence in 0.0..1.0)
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
            require(redundancy in 0.0..1.0)
            require(freshnessSeconds >= 0.0)
            require(oodRisk in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val modalities: List<ModalityEvidence>,
    ) {
        init {
            require(timestampNanos > 0L)
            require(
                modalities.map { it.modality }.distinct().size ==
                    modalities.size,
            )
        }
    }

    data class ModalityDecision(
        val modality: Modality,
        val measured: Boolean,
        val rawQuality: Double,
        val temporalPrior: Double,
        val adjustedReliability: Double,
        val marginalUtility: Double,
        val contribution: Double,
        val selected: Boolean,
        val quarantined: Boolean,
        val stateSamples: Long,
        val diagnostics: List<String>,
    )

    data class Result(
        val timestampNanos: Long,
        val modalities: List<ModalityDecision>,
        val selectedModalityCount: Int,
        val effectiveModalityCount: Double,
        val noisyOrSupport: Double,
        val fusionConfidence: Double,
        val bestSingleModality: Modality?,
        val dominantModality: Modality?,
        val excludedReliability: Double,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class State(
        var reliability: Double = 0.72,
        var samples: Long = 0L,
        var goodFrames: Int = 0,
        var badFrames: Int = 0,
        var quarantined: Boolean = false,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
    )

    private val states =
        linkedMapOf<Modality, State>()

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) {
            return
        }

        config = newConfig

        states.values.forEach { state ->
            state.reliability =
                state.reliability.coerceIn(0.02, 1.0)
            if (
                state.quarantined &&
                state.reliability >
                    newConfig.lowReliabilityQuarantineThreshold &&
                state.badFrames <=
                    newConfig.quarantineAfterBadFrames / 2
            ) {
                state.goodFrames = 0
            }
        }
    }

    fun configuration(): Config = config

    @Synchronized
    fun arbitrate(input: Input): Result {
        val normalizedEvidence = input.modalities
            .filter { it.confidence >= 0.0 }
            .associateBy { it.modality }

        val decisions = Modality.entries.map { modality ->
            val evidence = normalizedEvidence[modality]
            evaluateModality(
                timestampNanos = input.timestampNanos,
                modality = modality,
                evidence = evidence,
            )
        }

        val selectedIds = selectModalities(decisions)
        val selected =
            decisions
                .distinctBy { it.modality }
                .map { decision ->
                if (decision.modality !in selectedIds) {
                    decision.copy(
                        selected = false,
                        contribution = 0.0,
                    )
                } else {
                    decision.copy(
                        selected = true,
                        contribution = decision.marginalUtility,
                    )
                }
            }

        val selectedDecisions =
            selected.filter { it.selected }

        val weightedContributions =
            selectedDecisions
                .map {
                    (
                        it.contribution *
                            modalityImportance(it.modality)
                        ).coerceAtLeast(0.0)
                }

        val noisyOrSupport =
            noisyOr(
                weightedContributions,
            )

        val effectiveModalityCount =
            selectedDecisions
                .sumOf {
                    it.contribution.coerceIn(0.0, 1.0)
                }

        val bestSingle =
            decisions
                .filter { it.measured }
                .maxByOrNull { it.adjustedReliability }
                ?.modality

        val dominant =
            selected
                .filter { it.selected }
                .maxByOrNull {
                    it.contribution *
                        modalityImportance(it.modality)
                }
                ?.modality

        val excludedReliability =
            decisions
                .filter { !it.selected && it.measured }
                .sumOf { it.adjustedReliability * 0.17 }
                .coerceIn(0.0, 1.0)

        val strongIndependentCount =
            selectedDecisions.count {
                it.rawQuality >= 0.65 &&
                    !it.quarantined &&
                    !it.diagnostics.contains(
                        "modality-out-of-distribution-risk-high",
                    )
            }
        val independentEvidenceBonus =
            if (strongIndependentCount >= 2) {
                min(
                    0.06,
                    (strongIndependentCount - 1) * 0.03,
                )
            } else {
                0.0
            }

        val fusionConfidence =
            (
                noisyOrSupport * 0.42 +
                    weightedAverage(
                        selectedDecisions,
                    ) * 0.40 +
                    complementarityScore(
                        selected,
                    ) * 0.18 +
                    independentEvidenceBonus
                ).coerceIn(0.0, 1.0)

        updateStates(
            timestampNanos = input.timestampNanos,
            evidence = normalizedEvidence,
            decisions = decisions,
            selectedModalityCount = selectedIds.size,
        )

        val warnings = (
            buildWarnings(
                decisions = selected,
                noisyOrSupport = noisyOrSupport,
                fusionConfidence = fusionConfidence,
                selectedModalityCount = selectedIds.size,
            ) +
                decisions
                    .filter { decision ->
                        decision.measured &&
                            decision.diagnostics.any { diagnostic ->
                                diagnostic.contains("out-of-distribution")
                            }
                    }
                    .flatMap { it.diagnostics }
        ).distinct()

        val recommendations = buildRecommendations(
            decisions = selected,
            fusionConfidence = fusionConfidence,
            selectedModalityCount = selectedIds.size,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            modalities = selected,
            selectedModalityCount = selectedIds.size,
            effectiveModalityCount = effectiveModalityCount,
            noisyOrSupport = noisyOrSupport,
            fusionConfidence = fusionConfidence,
            bestSingleModality = bestSingle,
            dominantModality = dominant,
            excludedReliability = excludedReliability,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        states.clear()
    }

    fun currentReliability(modality: Modality): Double =
        states[modality]?.reliability
            ?: 0.72

    private fun evaluateModality(
        timestampNanos: Long,
        modality: Modality,
        evidence: ModalityEvidence?,
    ): ModalityDecision {
        val state =
            states.getOrPut(modality) {
                State()
            }

        if (evidence == null) {
            return ModalityDecision(
                modality = modality,
                measured = false,
                rawQuality = 0.0,
                temporalPrior = state.reliability,
                adjustedReliability = 0.0,
                marginalUtility = 0.0,
                contribution = 0.0,
                selected = false,
                quarantined = state.quarantined,
                stateSamples = state.samples,
                diagnostics = listOf(
                    "modality-not-observed",
                ),
            )
        }

        val quality =
            rawQuality(evidence)

        /*
         * The current frame is scored against the state that existed before this frame. This
         * prevents a pathological observation from first teaching itself into the temporal prior.
         */
        val prior =
            state.reliability.coerceIn(0.0, 1.0)

        val temporalBlend =
            (
                quality * config.qualityAlpha +
                    prior * config.temporalPriorAlpha
                ).coerceIn(0.0, 1.0)

        val adjusted =
            when {
                evidence.predictionOnly -> prior
                state.quarantined ->
                    max(
                        config.quarantinedWeight,
                        temporalBlend * config.quarantinedWeight,
                    )
                else -> temporalBlend
            }

        val diagnostics = buildList {
            if (evidence.confidence < config.minimumConfidence) {
                add("modality-confidence-below-floor")
            }
            if (evidence.uncertaintyMeters > config.uncertaintyScaleMeters) {
                add("modality-uncertainty-elevated")
            }
            if (evidence.freshnessSeconds > config.staleEvidenceSeconds) {
                add("modality-stale")
            }
            if (evidence.oodRisk >= config.oodQuarantineThreshold) {
                add("modality-out-of-distribution-risk-high")
            }
            if (evidence.predicted) {
                add("prediction-derived-modality")
            }
            if (evidence.predictionOnly) {
                add("prediction-only-modality")
            }
            if (evidence.occluded) {
                add("occlusion-flagged-modality")
            }
            if (evidence.redundancy >= 0.75) {
                add("modality-redundancy-high")
            }
            if (state.quarantined) {
                add("modality-quarantined")
            }
        }

        return ModalityDecision(
            modality = modality,
            measured = true,
            rawQuality = quality,
            temporalPrior = prior,
            adjustedReliability = adjusted,
            marginalUtility = marginalUtility(
                reliability = adjusted,
                redundancy = evidence.redundancy,
                predictionOnly = evidence.predictionOnly,
            ),
            contribution = 0.0,
            selected = false,
            quarantined = state.quarantined,
            stateSamples = state.samples,
            diagnostics = diagnostics.distinct(),
        )
    }

    private fun rawQuality(
        evidence: ModalityEvidence,
    ): Double {
        val confidence =
            evidence.confidence.coerceIn(0.0, 1.0)

        val uncertainty =
            uncertaintyScore(
                uncertaintyMeters = evidence.uncertaintyMeters,
            )

        val freshness =
            freshnessScore(
                freshnessSeconds = evidence.freshnessSeconds,
            )

        val ood =
            1.0 -
                evidence.oodRisk.coerceIn(0.0, 1.0)

        val prediction =
            if (evidence.predicted) {
                1.0 - config.predictionPenalty
            } else {
                1.0
            }

        val occlusion =
            if (evidence.occluded) {
                1.0 - config.occlusionPenalty
            } else {
                1.0
            }

        val baseQuality = (
            confidence * 0.38 +
                uncertainty * 0.22 +
                freshness * config.freshnessWeight +
                ood * config.oodWeight +
                prediction * 0.06 +
                occlusion * 0.04
            ).coerceIn(0.0, 1.0)

        return (
            baseQuality *
                (0.20 + 0.80 * confidence) *
                (0.30 + 0.70 * ood) *
                (0.45 + 0.55 * freshness)
            ).coerceIn(0.0, 1.0)
    }

    private fun uncertaintyScore(
        uncertaintyMeters: Double,
    ): Double {
        if (uncertaintyMeters <= 0.0) return 1.0

        return exp(
            -(
                uncertaintyMeters /
                    config.uncertaintyScaleMeters.coerceAtLeast(1e-6)
                ),
        ).coerceIn(0.0, 1.0)
    }

    private fun freshnessScore(
        freshnessSeconds: Double,
    ): Double {
        if (freshnessSeconds <= 0.0) return 1.0
        if (freshnessSeconds >= config.maximumEvidenceAgeSeconds) {
            return 0.0
        }

        return (
            1.0 -
                freshnessSeconds /
                    config.maximumEvidenceAgeSeconds
            ).coerceIn(0.0, 1.0)
    }

    private fun marginalUtility(
        reliability: Double,
        redundancy: Double,
        predictionOnly: Boolean = false,
    ): Double {
        val redundancyScale =
            (
                1.0 -
                    redundancy.coerceIn(0.0, 1.0) *
                    config.redundancyPenaltyWeight
                ).coerceIn(0.0, 1.0)
        val predictionScale =
            if (predictionOnly) {
                (1.0 - config.predictionPenalty * 0.75)
                    .coerceIn(0.20, 1.0)
            } else {
                1.0
            }
        return (
            reliability *
                redundancyScale *
                predictionScale
            ).coerceIn(0.0, 1.0)
    }

    private fun selectModalities(
        decisions: List<ModalityDecision>,
    ): Set<Modality> {
        val available =
            decisions
                .filter {
                    it.measured &&
                        !it.quarantined &&
                        it.adjustedReliability >=
                            config.minimumAdjustedReliability &&
                        it.rawQuality >= 0.05 &&
                        !it.diagnostics.contains(
                            "modality-out-of-distribution-risk-high",
                        )
                }
                .sortedWith(
                    compareByDescending<ModalityDecision> {
                        it.marginalUtility *
                            modalityImportance(it.modality)
                    }.thenBy { it.modality.ordinal },
                )

        if (available.isEmpty()) {
            val emergency =
                decisions
                    .filter {
                        it.measured &&
                            !it.quarantined &&
                            it.adjustedReliability >= 0.05 &&
                            it.rawQuality >= 0.05 &&
                            !it.diagnostics.contains(
                                "modality-out-of-distribution-risk-high",
                            )
                    }
                    .maxByOrNull { it.adjustedReliability }

            return emergency?.let {
                setOf(it.modality)
            } ?: emptySet()
        }

        val selected = linkedSetOf<Modality>()

        for (decision in available) {
            if (selected.size >= config.maximumSelectedModalities) {
                break
            }

            val marginal =
                marginalGain(
                    candidate = decision,
                    selected = selected,
                    decisions = available,
                )

            if (
                selected.size < config.minimumSelectedModalities ||
                marginal >= config.marginalSelectionThreshold
            ) {
                selected += decision.modality
            }
        }

        if (selected.isEmpty()) {
            selected += available.first().modality
        }

        return selected
    }

    private fun marginalGain(
        candidate: ModalityDecision,
        selected: Set<Modality>,
        decisions: List<ModalityDecision>,
    ): Double {
        if (selected.isEmpty()) {
            return candidate.marginalUtility
        }

        val meanSelected =
            selected.mapNotNull { selectedModality ->
                decisions.firstOrNull {
                    it.modality == selectedModality
                }?.marginalUtility
            }.averageOrZero()

        val complementarity =
            (
                1.0 -
                    meanSelected.coerceIn(0.0, 1.0)
            ).coerceIn(0.0, 1.0)

        return (
            candidate.marginalUtility * 0.74 +
                complementarity * 0.26
            ).coerceIn(0.0, 1.0)
    }

    private fun complementarityScore(
        decisions: List<ModalityDecision>,
    ): Double {
        val selected =
            decisions.filter { it.selected }
        if (selected.size <= 1) return 0.55

        val utilities =
            selected.map { it.marginalUtility }
        val mean =
            utilities.average()
        val spread =
            utilities.maxOrNull()!! -
                utilities.minOrNull()!!

        return (
            mean * 0.72 +
                (1.0 - spread) * 0.28
            ).coerceIn(0.0, 1.0)
    }

    private fun weightedAverage(
        decisions: List<ModalityDecision>,
    ): Double {
        if (decisions.isEmpty()) return 0.0

        var numerator = 0.0
        var denominator = 0.0

        decisions.forEach { decision ->
            val weight =
                (
                    decision.contribution *
                        modalityImportance(decision.modality)
                    ).coerceAtLeast(1e-9)
            numerator +=
                decision.adjustedReliability *
                    weight
            denominator += weight
        }

        return (
            numerator /
                denominator.coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun noisyOr(
        contributions: List<Double>,
    ): Double {
        if (contributions.isEmpty()) return 0.0

        var complement = 1.0

        contributions.forEach { value ->
            val effective =
                value.coerceIn(0.0, 1.0)

            complement *=
                1.0 -
                    effective
        }

        return (
            1.0 -
                complement
            ).coerceIn(0.0, 1.0)
    }

    private fun updateStates(
        timestampNanos: Long,
        evidence: Map<Modality, ModalityEvidence>,
        decisions: List<ModalityDecision>,
        selectedModalityCount: Int,
    ) {
        decisions.forEach { decision ->
            val sample =
                evidence[decision.modality]
                    ?: return@forEach

            val state =
                states.getOrPut(
                    decision.modality,
                ) {
                    State()
                }

            val predictionOnly =
                sample.predictionOnly

            if (predictionOnly) {
                state.lastTimestampNanos =
                    max(
                        state.lastTimestampNanos,
                        timestampNanos,
                    )
                return@forEach
            }

            val bad =
                sample.confidence <
                    config.minimumConfidence ||
                    sample.oodRisk >=
                        config.oodQuarantineThreshold ||
                    decision.rawQuality <
                        config.lowReliabilityQuarantineThreshold

            val clean =
                !bad &&
                    decision.adjustedReliability >=
                        config.minimumAdjustedReliability &&
                    sample.oodRisk <
                        config.recoveryOodThreshold

            if (bad) {
                state.badFrames += 1
                state.goodFrames = 0
                state.reliability =
                    (
                        state.reliability *
                            (1.0 - config.historyAlpha * 0.72)
                        ).coerceAtLeast(0.02)
            } else {
                state.badFrames =
                    max(
                        0,
                        state.badFrames - 1,
                    )

                if (clean) {
                    state.goodFrames += 1
                } else {
                    state.goodFrames = 0
                }

                state.reliability =
                    (
                        state.reliability *
                            (1.0 - config.historyAlpha) +
                            decision.rawQuality *
                            config.historyAlpha
                        ).coerceIn(0.02, 1.0)
            }

            if (
                state.badFrames >=
                    config.quarantineAfterBadFrames
            ) {
                state.quarantined = true
                state.goodFrames = 0
            }

            if (
                state.quarantined &&
                state.goodFrames >=
                    config.recoveryFrames &&
                sample.oodRisk <
                    config.recoveryOodThreshold
            ) {
                state.quarantined = false
                state.badFrames = 0
                state.goodFrames = 0
            }

            state.lastTimestampNanos =
                max(
                    state.lastTimestampNanos,
                    timestampNanos,
                )
            state.samples += 1L
        }

        /*
         * A frame with no useful selected evidence is itself a reliability event. It does not
         * mutate modality state aggressively, but it is retained as a diagnostic condition.
         */
        if (selectedModalityCount == 0) {
            // Intentionally no modality mutation.
        }
    }

    private fun buildWarnings(
        decisions: List<ModalityDecision>,
        noisyOrSupport: Double,
        fusionConfidence: Double,
        selectedModalityCount: Int,
    ): List<String> = buildList {
        decisions
            .filter { it.quarantined && it.measured }
            .forEach {
                add(
                    "cross-modal-modality-quarantined=" +
                        it.modality.name.lowercase(),
                )
            }

        decisions
            .filter {
                it.measured &&
                    !it.selected &&
                    it.adjustedReliability >=
                        config.minimumAdjustedReliability
            }
            .forEach {
                add(
                    "cross-modal-modality-deprioritized=" +
                        it.modality.name.lowercase(),
                )
            }

        if (selectedModalityCount == 1) {
            add("cross-modal-single-modality-fallback")
        }

        if (noisyOrSupport < 0.45) {
            add("cross-modal-noisy-or-support-low")
        }

        if (fusionConfidence <
            config.minimumFusionConfidence
        ) {
            add("cross-modal-fusion-confidence-low")
        }

        val staleCount =
            decisions.count {
                it.diagnostics.contains("modality-stale")
            }

        if (staleCount > 0) {
            add(
                "cross-modal-stale-modality-count=" +
                    staleCount,
            )
        }

        val highOod =
            decisions.count {
                it.diagnostics.contains(
                    "modality-out-of-distribution-risk-high",
                )
            }

        if (highOod > 0) {
            add(
                "cross-modal-high-ood-modality-count=" +
                    highOod,
            )
        }
    }.distinct()

    private fun buildRecommendations(
        decisions: List<ModalityDecision>,
        fusionConfidence: Double,
        selectedModalityCount: Int,
    ): List<String> = buildList {
        if (selectedModalityCount == 1) {
            add(
                "Treat the current solve as single-modality fallback until independent evidence recovers.",
            )
        }

        decisions
            .filter { it.quarantined }
            .forEach {
                add(
                    "Hold " +
                        it.modality.name.lowercase() +
                        " out of fusion until repeated clean evidence clears quarantine.",
                )
            }

        decisions
            .filter {
                it.selected &&
                    it.rawQuality < 0.45
            }
            .forEach {
                add(
                    "Preserve " +
                        it.modality.name.lowercase() +
                        " but lower downstream influence until quality recovers.",
                )
            }

        if (fusionConfidence < 0.55) {
            add(
                "Increase independent camera coverage or preserve adjacent frames before trusting an identity switch.",
            )
        }

        if (
            decisions.count { it.modality == Modality.BODY && it.selected } == 0
        ) {
            add(
                "Recover body/root evidence before committing metric actor continuity.",
            )
        }
    }.distinct().take(8)

    private fun modalityImportance(
        modality: Modality,
    ): Double =
        when (modality) {
            Modality.BODY -> 0.34
            Modality.FACE -> 0.20
            Modality.LEFT_HAND -> 0.14
            Modality.RIGHT_HAND -> 0.14
            Modality.EYE_LEFT -> 0.09
            Modality.EYE_RIGHT -> 0.09
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
