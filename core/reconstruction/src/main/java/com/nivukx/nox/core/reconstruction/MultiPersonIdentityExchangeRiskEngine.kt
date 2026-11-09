package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Detects identity-exchange risk during heavy multi-person occlusion.
 *
 * Research on realistic 2026 markerless capture benchmarks shows that close
 * interaction, rapid crossings, severe occlusion and temporal identity swaps are
 * substantially harder than isolated single-person poses. This engine does not
 * invent identity labels. It estimates whether the current evidence is strong
 * enough to trust the identity continuity of the reconstruction.
 *
 * The design is intentionally conservative:
 *  - visible coverage drops increase risk;
 *  - pair overlap and similar confidence profiles increase ambiguity;
 *  - cross-view agreement and temporal identity confidence reduce risk;
 *  - bounded history prevents long takes from retaining unlimited subject state.
 */
class MultiPersonIdentityExchangeRiskEngine(
    private var config: Config = Config.production(),
) {
    enum class State {
        CLEAR,
        GUARDED,
        HIGH_RISK,
        HOLD,
    }

    data class Config(
        val enabled: Boolean = true,
        val maximumSubjects: Int = 64,
        val historyFrames: Int = 10,
        val minimumObservationConfidence: Double = 0.10,
        val highRiskThreshold: Double = 0.58,
        val holdThreshold: Double = 0.76,
        val temporalRiskWeight: Double = 0.24,
        val visibilityRiskWeight: Double = 0.23,
        val overlapRiskWeight: Double = 0.22,
        val crossViewRiskWeight: Double = 0.16,
        val ambiguityWeight: Double = 0.15,
        val continuityRecovery: Double = 0.12,
    ) {
        init {
            require(maximumSubjects in 1..256)
            require(historyFrames in 2..120)
            require(minimumObservationConfidence in 0.0..1.0)
            require(highRiskThreshold in 0.0..1.0)
            require(holdThreshold >= highRiskThreshold)
            require(temporalRiskWeight >= 0.0)
            require(visibilityRiskWeight >= 0.0)
            require(overlapRiskWeight >= 0.0)
            require(crossViewRiskWeight >= 0.0)
            require(ambiguityWeight >= 0.0)
            require(continuityRecovery >= 0.0)
        }

        companion object {
            fun production(): Config = Config()

            fun qualification(): Config = Config(
                maximumSubjects = 24,
                historyFrames = 16,
                minimumObservationConfidence = 0.18,
                highRiskThreshold = 0.46,
                holdThreshold = 0.68,
                temporalRiskWeight = 0.27,
                visibilityRiskWeight = 0.25,
                overlapRiskWeight = 0.20,
                crossViewRiskWeight = 0.16,
                ambiguityWeight = 0.12,
                continuityRecovery = 0.16,
            )
        }
    }

    data class SubjectEvidence(
        val subjectId: Int,
        val visibleLandmarks: Int,
        val totalLandmarks: Int,
        val meanConfidence: Double,
        val crossViewAgreement: Double,
        val temporalIdentityConfidence: Double,
        val contactOverlap: Double = 0.0,
        val cameraCount: Int = 1,
    ) {
        init {
            require(subjectId >= 0)
            require(visibleLandmarks >= 0)
            require(totalLandmarks >= 0)
            require(cameraCount >= 1)
        }

        val visibilityRatio: Double
            get() =
                if (totalLandmarks <= 0) 0.0
                else visibleLandmarks
                    .toDouble()
                    .div(totalLandmarks.toDouble())
                    .coerceIn(0.0, 1.0)
    }

    data class SubjectRisk(
        val subjectId: Int,
        val exchangeRisk: Double,
        val continuityScore: Double,
        val occlusionRisk: Double,
        val ambiguityRisk: Double,
        val warnings: List<String>,
    )

    data class Result(
        val state: State,
        val productionIdentityConfidence: Double,
        val exchangeRisk: Double,
        val temporalContinuity: Double,
        val occlusionBurden: Double,
        val ambiguity: Double,
        val evaluatedSubjects: Int,
        val heldSubjects: Int,
        val subjectRisks: List<SubjectRisk>,
        val warnings: List<String>,
    )

    private data class HistorySample(
        val timestampNanos: Long,
        val confidence: Double,
        val visibility: Double,
        val crossView: Double,
        val contactOverlap: Double,
    )

    private val history =
        LinkedHashMap<Int, ArrayDeque<HistorySample>>()
    private var subjectCapApplied = false

    fun configure(newConfig: Config) {
        config = newConfig
        trimState()
    }

    fun configuration(): Config = config

    @Synchronized
    fun reset() {
        history.clear()
        subjectCapApplied = false
    }

    @Synchronized
    fun evaluate(
        timestampNanos: Long,
        subjects: List<SubjectEvidence>,
    ): Result {
        require(timestampNanos >= 0L)
        if (!config.enabled) {
            return Result(
                state = State.CLEAR,
                productionIdentityConfidence = 1.0,
                exchangeRisk = 0.0,
                temporalContinuity = 1.0,
                occlusionBurden = 0.0,
                ambiguity = 0.0,
                evaluatedSubjects = 0,
                heldSubjects = 0,
                subjectRisks = emptyList(),
                warnings = listOf("identity-exchange-disabled"),
            )
        }
        if (subjects.isEmpty()) {
            return Result(
                state = State.HOLD,
                productionIdentityConfidence = 0.0,
                exchangeRisk = 1.0,
                temporalContinuity = 0.0,
                occlusionBurden = 1.0,
                ambiguity = 1.0,
                evaluatedSubjects = 0,
                heldSubjects = 0,
                subjectRisks = emptyList(),
                warnings = listOf("no-subject-identity-evidence"),
            )
        }

        val clipped =
            subjects
                .asSequence()
                .filter {
                    it.meanConfidence >=
                        config.minimumObservationConfidence
                }
                .take(config.maximumSubjects)
                .toList()

        if (subjects.size > config.maximumSubjects) {
            subjectCapApplied = true
        }

        val pairAmbiguity =
            pairAmbiguity(clipped)

        clipped.forEach { subject ->
            val visibility = subject.visibilityRatio
            val confidence =
                subject.meanConfidence.coerceIn(0.0, 1.0)
            val crossView =
                subject.crossViewAgreement.coerceIn(0.0, 1.0)
            val temporal =
                subject.temporalIdentityConfidence.coerceIn(0.0, 1.0)
            val overlap =
                subject.contactOverlap.coerceIn(0.0, 1.0)

            val previous =
                history[subject.subjectId]
                    ?.lastOrNull()

            val temporalJumpRisk =
                previous?.let {
                    (
                        abs(confidence - it.confidence) * 0.35 +
                            abs(visibility - it.visibility) * 0.25 +
                            abs(crossView - it.crossView) * 0.25 +
                            abs(overlap - it.contactOverlap) * 0.15
                    ).coerceIn(0.0, 1.0)
                } ?: 0.0

            val visibilityRisk =
                (1.0 - visibility).coerceIn(0.0, 1.0)

            val crossViewRisk =
                (1.0 - crossView).coerceIn(0.0, 1.0)

            val temporalRisk =
                (
                    (1.0 - temporal) * 0.72 +
                        temporalJumpRisk * 0.28
                ).coerceIn(0.0, 1.0)

            val ambiguityRisk =
                (
                    pairAmbiguity[subject.subjectId] ?: 0.0
                )

            val risk =
                (
                    config.temporalRiskWeight * temporalRisk +
                        config.visibilityRiskWeight * visibilityRisk +
                        config.overlapRiskWeight * overlap +
                        config.crossViewRiskWeight * crossViewRisk +
                        config.ambiguityWeight * ambiguityRisk
                ).coerceIn(0.0, 1.0)

            val continuity =
                (
                    1.0 - risk
                ).coerceIn(0.0, 1.0)

            val warnings = buildList {
                if (visibilityRisk > 0.55) add("heavy-occlusion")
                if (crossViewRisk > 0.35) add("weak-cross-view-identity")
                if (temporalRisk > 0.40) add("temporal-identity-instability")
                if (ambiguityRisk > 0.45) add("pair-ambiguity")
                if (risk >= config.highRiskThreshold) {
                    add("identity-exchange-risk")
                }
            }

            history.getOrPut(subject.subjectId) {
                ArrayDeque()
            }.addLast(
                HistorySample(
                    timestampNanos = timestampNanos,
                    confidence = confidence,
                    visibility = visibility,
                    crossView = crossView,
                    contactOverlap = overlap,
                ),
            )

            while (
                history[subject.subjectId]?.size ?: 0 >
                    config.historyFrames
            ) {
                history[subject.subjectId]?.removeFirst()
            }

            history[subject.subjectId]?.let { samples ->
                while (
                    samples.isNotEmpty() &&
                        timestampNanos -
                        samples.first().timestampNanos >
                        2_000_000_000L
                ) {
                    samples.removeFirst()
                }
            }
        }

        trimState()

        val risks =
            clipped.map { subject ->
                val subjectRisk = historyAwareRisk(subject, pairAmbiguity)
                subjectRisk
            }

        val meanRisk =
            risks.map { it.exchangeRisk }
                .average()
                .coerceIn(0.0, 1.0)

        val maxRisk =
            risks.maxOfOrNull { it.exchangeRisk }
                ?: 1.0

        val continuity =
            risks.map { it.continuityScore }
                .average()
                .coerceIn(0.0, 1.0)

        val occlusion =
            risks.map { it.occlusionRisk }
                .average()
                .coerceIn(0.0, 1.0)

        val ambiguity =
            risks.map { it.ambiguityRisk }
                .average()
                .coerceIn(0.0, 1.0)

        val held =
            risks.count {
                it.exchangeRisk >= config.holdThreshold
            }

        val state =
            when {
                risks.isEmpty() ->
                    State.HOLD
                held > 0 ->
                    State.HOLD
                maxRisk >= config.holdThreshold ->
                    State.HOLD
                meanRisk >= config.highRiskThreshold ->
                    State.HIGH_RISK
                meanRisk >= config.highRiskThreshold * 0.68 ->
                    State.GUARDED
                else ->
                    State.CLEAR
            }

        val recoveryFactor =
            (
                config.continuityRecovery *
                    continuity
            ).coerceIn(0.0, 0.20)

        val productionConfidence =
            (
                1.0 -
                    meanRisk +
                    recoveryFactor -
                    max(
                        0.0,
                        (maxRisk - meanRisk) * 0.22
                    )
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (subjectCapApplied) {
                add("subject-cap-applied")
            }
            if (occlusion > 0.40) {
                add("interaction-occlusion-burden")
            }
            if (ambiguity > 0.32) {
                add("identity-ambiguity-elevated")
            }
            if (state == State.GUARDED) add("identity-guarded")
            if (state == State.HIGH_RISK) add("identity-high-risk")
            if (state == State.HOLD) add("identity-hold-required")
            risks.flatMap { it.warnings }
                .distinct()
                .take(16)
                .forEach { add(it) }
        }.distinct()

        return Result(
            state = state,
            productionIdentityConfidence = productionConfidence,
            exchangeRisk = maxRisk,
            temporalContinuity = continuity,
            occlusionBurden = occlusion,
            ambiguity = ambiguity,
            evaluatedSubjects = clipped.size,
            heldSubjects = held,
            subjectRisks = risks,
            warnings = warnings,
        )
    }

    private fun historyAwareRisk(
        subject: SubjectEvidence,
        pairAmbiguity: Map<Int, Double>,
    ): SubjectRisk {
        val visibilityRisk =
            (1.0 - subject.visibilityRatio)
                .coerceIn(0.0, 1.0)
        val crossViewRisk =
            (1.0 - subject.crossViewAgreement.coerceIn(0.0, 1.0))
                .coerceIn(0.0, 1.0)
        val ambiguityRisk =
            pairAmbiguity[subject.subjectId] ?: 0.0

        val historySamples =
            history[subject.subjectId].orEmpty()

        val temporalVolatility =
            if (historySamples.size < 2) {
                0.0
            } else {
                val ordered = historySamples.toList()
                val deltas =
                    ordered.zipWithNext()
                        .map { (a, b) ->
                            abs(b.confidence - a.confidence) * 0.45 +
                                abs(b.visibility - a.visibility) * 0.30 +
                                abs(b.crossView - a.crossView) * 0.25
                        }
                (deltas.average() * 1.8).coerceIn(0.0, 1.0)
            }

        val temporalRisk =
            (
                (1.0 - subject.temporalIdentityConfidence) * 0.70 +
                    temporalVolatility * 0.30
            ).coerceIn(0.0, 1.0)

        val overlapRisk =
            subject.contactOverlap.coerceIn(0.0, 1.0)

        val risk =
            (
                config.temporalRiskWeight * temporalRisk +
                    config.visibilityRiskWeight * visibilityRisk +
                    config.overlapRiskWeight * overlapRisk +
                    config.crossViewRiskWeight * crossViewRisk +
                    config.ambiguityWeight * ambiguityRisk
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (visibilityRisk > 0.55) add("heavy-occlusion")
            if (crossViewRisk > 0.35) add("weak-cross-view-identity")
            if (temporalRisk > 0.40) add("temporal-identity-instability")
            if (ambiguityRisk > 0.45) add("pair-ambiguity")
            if (risk >= config.highRiskThreshold) add("identity-exchange-risk")
        }.distinct()

        return SubjectRisk(
            subjectId = subject.subjectId,
            exchangeRisk = risk,
            continuityScore = (1.0 - risk).coerceIn(0.0, 1.0),
            occlusionRisk = visibilityRisk,
            ambiguityRisk = ambiguityRisk,
            warnings = warnings,
        )
    }

    private fun pairAmbiguity(
        subjects: List<SubjectEvidence>,
    ): Map<Int, Double> {
        if (subjects.size < 2) {
            return subjects.associate { it.subjectId to 0.0 }
        }

        val result =
            subjects.associateWith { subject ->
                subjects.asSequence()
                    .filter { other ->
                        other.subjectId != subject.subjectId
                    }
                    .map { other ->
                        val confidenceSimilarity =
                            1.0 -
                                abs(
                                    subject.meanConfidence -
                                        other.meanConfidence,
                                ).coerceIn(0.0, 1.0)

                        val visibilitySimilarity =
                            1.0 -
                                abs(
                                    subject.visibilityRatio -
                                        other.visibilityRatio,
                                ).coerceIn(0.0, 1.0)

                        val crossViewSimilarity =
                            1.0 -
                                abs(
                                    subject.crossViewAgreement -
                                        other.crossViewAgreement,
                                ).coerceIn(0.0, 1.0)

                        val interactionOverlap =
                            min(
                                subject.contactOverlap,
                                other.contactOverlap,
                            ).coerceIn(0.0, 1.0)

                        (
                            confidenceSimilarity * 0.25 +
                                visibilitySimilarity * 0.20 +
                                crossViewSimilarity * 0.20 +
                                interactionOverlap * 0.35
                        ).coerceIn(0.0, 1.0)
                    }
                    .maxOrNull()
                    ?: 0.0
            }

        return result.mapKeys { it.key.subjectId }
    }

    private fun trimState() {
        while (history.size > config.maximumSubjects) {
            val victim =
                history.entries.minByOrNull {
                    it.value.lastOrNull()?.timestampNanos ?: Long.MIN_VALUE
                }?.key ?: break
            history.remove(victim)
        }
    }
}
