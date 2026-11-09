package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.PerceptionSource
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Adaptive reliability estimator for heterogeneous motion-capture evidence.
 *
 * The estimator is deliberately model-agnostic. It learns a bounded reliability value for each
 * evidence provider from confidence, freshness, innovation/residual behaviour, and agreement with
 * the recent consensus. A detector is never promoted merely because it reports a high confidence;
 * repeated inconsistency causes its contribution to decay until it demonstrates recovery.
 *
 * This is intended for live pipelines where body, face, eye, hand, depth, NPU, GPU and temporal
 * predictors arrive asynchronously and may have different failure modes.
 */
class AdaptiveEvidenceReliability(
    private val config: Config = Config(),
) {

    data class Config(
        val baseReliability: Double = 0.72,
        val minimumReliability: Double = 0.16,
        val maximumReliability: Double = 1.0,
        val learningRate: Double = 0.12,
        val recoveryRate: Double = 0.045,
        val freshnessHalfLifeMillis: Double = 240.0,
        val residualScale: Double = 1.75,
        val disagreementScale: Double = 2.20,
        val consensusWeight: Double = 0.30,
        val residualWeight: Double = 0.20,
        val freshnessWeight: Double = 0.18,
        val reportedConfidenceWeight: Double = 0.20,
        val stabilityWeight: Double = 0.12,
        val historyLength: Int = 24,
        val outlierSigma: Double = 3.0,
        val recoveryFrames: Int = 8,
        val staleCutoffMillis: Double = 1500.0,
        val minimumUpdateConfidence: Double = 0.10,
        val outOfDistributionPenaltyWeight: Double = 0.30,
        val outOfDistributionQuarantineThreshold: Double = 0.82,
        val quarantineAfterBadFrames: Int = 3,
        val quarantineWeight: Double = 0.02,
    ) {
        init {
            require(baseReliability in 0.0..1.0)
            require(minimumReliability in 0.0..1.0)
            require(maximumReliability in 0.0..1.0)
            require(minimumReliability <= baseReliability)
            require(baseReliability <= maximumReliability)
            require(learningRate in 0.0..1.0)
            require(recoveryRate in 0.0..1.0)
            require(freshnessHalfLifeMillis > 0.0)
            require(residualScale > 0.0)
            require(disagreementScale > 0.0)
            require(consensusWeight >= 0.0)
            require(residualWeight >= 0.0)
            require(freshnessWeight >= 0.0)
            require(reportedConfidenceWeight >= 0.0)
            require(stabilityWeight >= 0.0)
            require(historyLength >= 4)
            require(outlierSigma > 0.0)
            require(recoveryFrames >= 1)
            require(staleCutoffMillis >= freshnessHalfLifeMillis)
            require(minimumUpdateConfidence in 0.0..1.0)
            require(outOfDistributionPenaltyWeight >= 0.0)
            require(outOfDistributionQuarantineThreshold in 0.0..1.0)
            require(quarantineAfterBadFrames >= 1)
            require(quarantineWeight in 0.0..1.0)
            val weightSum =
                consensusWeight +
                    residualWeight +
                    freshnessWeight +
                    reportedConfidenceWeight +
                    stabilityWeight
            require(weightSum > 0.0)
        }
    }

    data class Observation(
        val source: PerceptionSource,
        val timestampNanos: Long,
        val reportedConfidence: Double,
        val residual: Double = 0.0,
        val disagreement: Double = 0.0,
        val stability: Double = 1.0,
        val ageMillis: Double = 0.0,
        val accepted: Boolean = true,
        val outOfDistributionRisk: Double = 0.0,
        val predictionOnly: Boolean = false,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(outOfDistributionRisk.isNaN() || outOfDistributionRisk.isInfinite() || outOfDistributionRisk in 0.0..1.0)
        }
    }

    data class Reliability(
        val source: PerceptionSource,
        val value: Double,
        val sampleCount: Long,
        val recentMeanResidual: Double,
        val recentMeanDisagreement: Double,
        val recentStability: Double,
        val freshness: Double,
        val recoveryProgress: Double,
        val degraded: Boolean,
        val quarantined: Boolean = false,
        val quarantineProgress: Double = 0.0,
        val outOfDistributionRisk: Double = 0.0,
        val robustResidualScale: Double = 0.0,
        val robustDisagreementScale: Double = 0.0,
        val warnings: List<String>,
    )

    data class Snapshot(
        val timestampNanos: Long,
        val values: Map<PerceptionSource, Reliability>,
        val effectiveConsensus: Double,
        val highestReliabilitySource: PerceptionSource?,
        val lowestReliabilitySource: PerceptionSource?,
        val warnings: List<String>,
    )

    private data class State(
        var reliability: Double,
        var samples: Long,
        var consecutiveGoodSamples: Int,
        var consecutiveBadSamples: Int,
        var consecutiveOodSamples: Int,
        var quarantined: Boolean,
        var lastTimestampNanos: Long,
        val residuals: ArrayDeque<Double> = ArrayDeque(),
        val disagreements: ArrayDeque<Double> = ArrayDeque(),
        val stabilities: ArrayDeque<Double> = ArrayDeque(),
    )

    private val states = LinkedHashMap<PerceptionSource, State>()

    @Synchronized
    fun observe(
        observation: Observation,
        consensusConfidence: Double,
    ): Reliability {
        val source = observation.source
        val state = states.getOrPut(source) {
            State(
                reliability = config.baseReliability,
                samples = 0L,
                consecutiveGoodSamples = 0,
                consecutiveBadSamples = 0,
                consecutiveOodSamples = 0,
                quarantined = false,
                lastTimestampNanos = observation.timestampNanos,
            )
        }

        val freshness = freshness(observation.ageMillis)
        val confidence = sanitize(observation.reportedConfidence)
        val stability = sanitize(observation.stability)
        val consensus = sanitize(consensusConfidence)

        /*
         * Score the current observation against history that existed before this sample.
         * A pathological sample therefore cannot first enter the history and then make itself
         * look normal through self-contamination.
         */
        val residualScore = robustResidualScore(
            residual = observation.residual,
            history = state.residuals,
        )
        val disagreementScore = robustDisagreementScore(
            disagreement = observation.disagreement,
            history = state.disagreements,
        )
        val consensusScore =
            agreementWithConsensus(
                confidence = confidence,
                consensus = consensus,
            )
        val oodPenalty = observation.outOfDistributionRisk *
            config.outOfDistributionPenaltyWeight
        val sampleScore =
            (
                weightedScore(
                    consensusScore = consensusScore,
                    residualScore = residualScore,
                    freshness = freshness,
                    confidence = confidence,
                    stability = stability,
                ) -
                    oodPenalty
                ).coerceIn(0.0, 1.0)

        /*
         * Prediction-only evidence is useful for downstream continuity but must not teach the
         * reliability estimator that predicted output is equivalent to observed evidence.
         */
        if (observation.predictionOnly) {
            state.lastTimestampNanos =
                max(
                    state.lastTimestampNanos,
                    observation.timestampNanos,
                )
            return reliability(
                source = source,
                state = state,
                currentFreshness = freshness,
                transientOutOfDistributionRisk = observation.outOfDistributionRisk,
            ).copy(
                warnings = (
                    reliability(
                        source = source,
                        state = state,
                        currentFreshness = freshness,
                        transientOutOfDistributionRisk =
                            observation.outOfDistributionRisk,
                    ).warnings +
                        "prediction-only evidence excluded from reliability adaptation"
                    ).distinct(),
            )
        }

        val hardFailure =
            !observation.accepted ||
                freshness <= 0.04 ||
                confidence < config.minimumUpdateConfidence ||
                observation.outOfDistributionRisk >=
                    config.outOfDistributionQuarantineThreshold

        val badSample =
            hardFailure ||
                sampleScore < 0.35 ||
                observation.outOfDistributionRisk >=
                    config.outOfDistributionQuarantineThreshold * 0.72

        if (hardFailure) {
            state.reliability =
                (
                    state.reliability *
                        (1.0 - config.learningRate * 0.72)
                ).coerceAtLeast(config.minimumReliability)
            state.consecutiveGoodSamples = 0
        } else {
            val rate =
                if (sampleScore >= 0.68) {
                    config.recoveryRate
                } else {
                    config.learningRate
                }

            state.reliability =
                (
                    state.reliability +
                        (sampleScore - state.reliability) * rate
                ).coerceIn(
                    config.minimumReliability,
                    config.maximumReliability,
                )

            if (sampleScore >= 0.72) {
                state.consecutiveGoodSamples += 1
            } else {
                state.consecutiveGoodSamples = 0
            }
        }

        if (badSample) {
            state.consecutiveBadSamples += 1
        } else {
            state.consecutiveBadSamples = max(
                0,
                state.consecutiveBadSamples - 1,
            )
        }

        if (
            observation.outOfDistributionRisk >=
                config.outOfDistributionQuarantineThreshold * 0.72
        ) {
            state.consecutiveOodSamples += 1
        } else {
            state.consecutiveOodSamples = max(
                0,
                state.consecutiveOodSamples - 1,
            )
        }

        if (
            state.consecutiveBadSamples >= config.quarantineAfterBadFrames ||
            state.consecutiveOodSamples >= config.quarantineAfterBadFrames
        ) {
            state.quarantined = true
            state.consecutiveGoodSamples = 0
        }

        if (
            state.quarantined &&
            state.consecutiveGoodSamples >= config.recoveryFrames &&
            observation.outOfDistributionRisk <
                config.outOfDistributionQuarantineThreshold * 0.45 &&
            sampleScore >= 0.72
        ) {
            state.quarantined = false
            state.consecutiveBadSamples = 0
            state.consecutiveOodSamples = 0
        }

        addHistory(
            state.residuals,
            abs(observation.residual),
        )
        addHistory(
            state.disagreements,
            observation.disagreement.coerceIn(0.0, 1.0),
        )
        addHistory(
            state.stabilities,
            stability,
        )

        state.samples += 1L
        state.lastTimestampNanos =
            max(
                state.lastTimestampNanos,
                observation.timestampNanos,
            )

        return reliability(
            source = source,
            state = state,
            currentFreshness = freshness,
            transientOutOfDistributionRisk = observation.outOfDistributionRisk,
        )
    }

    @Synchronized
    fun weight(
        source: PerceptionSource,
        ageMillis: Double = 0.0,
        reportedConfidence: Double = 1.0,
        outOfDistributionRisk: Double = 0.0,
    ): Double {
        val state = states[source]
        val reliability = state?.reliability ?: config.baseReliability
        val quarantineScale =
            if (state?.quarantined == true) {
                config.quarantineWeight
            } else {
                1.0
            }
        val oodScale = (
            1.0 -
                sanitize(outOfDistributionRisk) *
                    config.outOfDistributionPenaltyWeight
            ).coerceIn(0.0, 1.0)

        return (
            reliability *
                quarantineScale *
                oodScale *
                freshness(ageMillis) *
                sanitize(reportedConfidence)
            ).coerceIn(
                0.0,
                1.0,
            )
    }

    @Synchronized
    fun reliability(
        source: PerceptionSource,
    ): Reliability {
        val state = states[source]
            ?: return Reliability(
                source = source,
                value = config.baseReliability,
                sampleCount = 0L,
                recentMeanResidual = 0.0,
                recentMeanDisagreement = 0.0,
                recentStability = 1.0,
                freshness = 1.0,
                recoveryProgress = 0.0,
                degraded = false,
                warnings = emptyList(),
            )

        return reliability(
            source = source,
            state = state,
            currentFreshness = freshness(
                ageMillis = 0.0,
            ),
        )
    }

    @Synchronized
    fun snapshot(
        timestampNanos: Long,
    ): Snapshot {
        val values =
            states.mapValues { (source, state) ->
                reliability(
                    source = source,
                    state = state,
                    currentFreshness =
                        freshness(
                            ageMillis = (
                                (
                                    timestampNanos -
                                        state.lastTimestampNanos
                                    ).coerceAtLeast(0L) / 1e6
                                ),
                        ),
                    transientOutOfDistributionRisk = 0.0,
                )
            }

        val active = values
            .values
            .filterNot { it.quarantined }

        val highest =
            active.maxByOrNull { it.value }?.source
        val lowest =
            active.minByOrNull { it.value }?.source

        val weightedActive = active.sumOf {
            it.value * (
                1.0 -
                    it.outOfDistributionRisk *
                    config.outOfDistributionPenaltyWeight
                )
        }

        val effectiveConsensus =
            if (active.isEmpty()) {
                0.0
            } else {
                active.sumOf {
                    it.value *
                        it.recentStability *
                        it.freshness
                } /
                    weightedActive.coerceAtLeast(1e-9)
            }

        val warnings = buildList {
            values.values
                .filter { it.degraded }
                .forEach {
                    add(
                        "degraded source: " +
                            it.source.name.lowercase(),
                    )
                }

            values.values
                .filter { it.quarantined }
                .forEach {
                    add(
                        "quarantined source: " +
                            it.source.name.lowercase(),
                    )
                }

            if (effectiveConsensus < 0.45) {
                add("cross-source reliability consensus is weak")
            }

            if (active.isEmpty() && values.isNotEmpty()) {
                add("all evidence sources are quarantined")
            }
        }.distinct()

        return Snapshot(
            timestampNanos = timestampNanos,
            values = values,
            effectiveConsensus = effectiveConsensus.coerceIn(
                0.0,
                1.0,
            ),
            highestReliabilitySource = highest,
            lowestReliabilitySource = lowest,
            warnings = warnings,
        )
    }

    @Synchronized
    fun prune(
        timestampNanos: Long,
    ) {
        val cutoff =
            (
                timestampNanos -
                    (config.staleCutoffMillis * 1e6)
                        .toLong()
                ).coerceAtLeast(0L)

        states.entries.removeIf { (_, state) ->
            state.lastTimestampNanos < cutoff &&
                state.samples > 0L &&
                !state.quarantined
        }
    }

    @Synchronized
    fun reset() {
        states.clear()
    }

    private fun reliability(
        source: PerceptionSource,
        state: State,
        currentFreshness: Double,
        transientOutOfDistributionRisk: Double = 0.0,
    ): Reliability {
        val residualScale = robustScale(
            state.residuals,
            config.residualScale,
        )
        val disagreementScale = robustScale(
            state.disagreements,
            config.disagreementScale,
        )

        val recoveryProgress =
            (
                state.consecutiveGoodSamples.toDouble() /
                    config.recoveryFrames.toDouble()
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val quarantineProgress =
            (
                state.consecutiveBadSamples.toDouble() /
                    config.quarantineAfterBadFrames.toDouble()
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val degraded =
            state.quarantined ||
                state.reliability <=
                    config.baseReliability * 0.62 ||
                currentFreshness < 0.25

        val warnings = buildList {
            if (degraded) {
                add("source reliability degraded")
            }
            if (state.quarantined) {
                add("source quarantined pending clean recovery")
            }
            if (state.residuals.isNotEmpty() &&
                residualScale > config.residualScale * 1.8
            ) {
                add("robust residual scale is elevated")
            }
            if (state.disagreements.isNotEmpty() &&
                disagreementScale > config.disagreementScale * 1.8
            ) {
                add("robust disagreement scale is elevated")
            }
            if (transientOutOfDistributionRisk >=
                config.outOfDistributionQuarantineThreshold * 0.72
            ) {
                add("out-of-distribution risk is elevated")
            }
            if (currentFreshness < 0.25) {
                add("source is becoming stale")
            }
            if (
                state.samples >= 4 &&
                recoveryProgress > 0.0 &&
                state.reliability >=
                    config.baseReliability * 0.95
            ) {
                add("source is recovering")
            }
        }.distinct()

        return Reliability(
            source = source,
            value = (
                if (state.quarantined) {
                    min(
                        state.reliability,
                        config.minimumReliability +
                            (1.0 - config.minimumReliability) *
                            config.quarantineWeight,
                    )
                } else {
                    state.reliability
                }
            ).coerceIn(
                config.minimumReliability,
                config.maximumReliability,
            ),
            sampleCount = state.samples,
            recentMeanResidual = mean(state.residuals),
            recentMeanDisagreement = mean(state.disagreements),
            recentStability = mean(state.stabilities),
            freshness = currentFreshness,
            recoveryProgress = recoveryProgress,
            degraded = degraded,
            quarantined = state.quarantined,
            quarantineProgress = quarantineProgress,
            outOfDistributionRisk =
                transientOutOfDistributionRisk.coerceIn(0.0, 1.0),
            robustResidualScale = residualScale,
            robustDisagreementScale = disagreementScale,
            warnings = warnings,
        )
    }

    private fun robustResidualScore(
        residual: Double,
        history: Collection<Double>,
    ): Double =
        robustDeviationScore(
            value = abs(residual),
            history = history,
            fallbackScale = config.residualScale,
        )

    private fun robustDisagreementScore(
        disagreement: Double,
        history: Collection<Double>,
    ): Double =
        robustDeviationScore(
            value = disagreement.coerceAtLeast(0.0),
            history = history,
            fallbackScale = config.disagreementScale.coerceAtLeast(1.0),
        )

    private fun robustDeviationScore(
        value: Double,
        history: Collection<Double>,
        fallbackScale: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (history.size < 4) {
            return (
                1.0 /
                    (
                        1.0 +
                            (value / fallbackScale.coerceAtLeast(1e-6))
                                .pow(2.0)
                    )
            ).coerceIn(0.0, 1.0)
        }

        val scale = robustScale(history, fallbackScale)
        val median = median(history)
        val deviation =
            abs(value - median) /
                scale.coerceAtLeast(1e-6)

        return (
            1.0 /
                (
                    1.0 +
                        (deviation / 2.5).pow(2.0)
                )
            ).coerceIn(0.0, 1.0)
    }

    private fun robustScale(
        history: Collection<Double>,
        fallbackScale: Double,
    ): Double {
        if (history.isEmpty()) return fallbackScale.coerceAtLeast(1e-6)
        val median = median(history)
        val deviations = history.map { abs(it - median) }
        val mad = median(deviations)
        return max(
            fallbackScale * 0.20,
            median + 1.4826 * mad,
        ).coerceAtLeast(1e-6)
    }

    private fun median(values: Collection<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 0.0
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun weightedScore(
        consensusScore: Double,
        residualScore: Double,
        freshness: Double,
        confidence: Double,
        stability: Double,
    ): Double {
        val total =
            config.consensusWeight +
                config.residualWeight +
                config.freshnessWeight +
                config.reportedConfidenceWeight +
                config.stabilityWeight

        return (
            consensusScore * config.consensusWeight +
                residualScore * config.residualWeight +
                freshness * config.freshnessWeight +
                confidence * config.reportedConfidenceWeight +
                stability * config.stabilityWeight
            ) / total
    }

    private fun freshness(
        ageMillis: Double,
    ): Double {
        if (!ageMillis.isFinite()) return 0.0
        if (ageMillis <= 0.0) return 1.0
        if (ageMillis >= config.staleCutoffMillis) return 0.0

        return exp(
            -ln(2.0) *
                ageMillis /
                config.freshnessHalfLifeMillis,
        ).coerceIn(
            0.0,
            1.0,
        )
    }

    private fun residualScore(
        residual: Double,
    ): Double {
        val value =
            abs(residual)

        if (!value.isFinite()) return 0.0

        val normalized =
            value /
                config.residualScale
                    .coerceAtLeast(1e-9)

        return (
            1.0 /
                (
                    1.0 +
                        normalized.pow(2.0)
                    )
            ).coerceIn(
                0.0,
                1.0,
            )
    }

    private fun disagreementScore(
        disagreement: Double,
    ): Double {
        val value =
            disagreement
                .coerceAtLeast(0.0)

        return exp(
            -value *
                config.disagreementScale,
        ).coerceIn(
            0.0,
            1.0,
        )
    }

    private fun agreementWithConsensus(
        confidence: Double,
        consensus: Double,
    ): Double {
        val gap =
            abs(confidence - consensus)

        return exp(
            -gap *
                config.disagreementScale,
        ).coerceIn(
            0.0,
            1.0,
        )
    }

    private fun addHistory(
        history: ArrayDeque<Double>,
        value: Double,
    ) {
        history.addLast(
            value.coerceIn(
                0.0,
                max(
                    config.residualScale,
                    1.0,
                ) * 8.0,
            ),
        )

        while (
            history.size >
            config.historyLength
        ) {
            history.removeFirst()
        }
    }

    private fun mean(
        values: Collection<Double>,
    ): Double =
        if (values.isEmpty()) {
            0.0
        } else {
            values.average().coerceIn(
                0.0,
                max(
                    config.residualScale,
                    1.0,
                ) * 8.0,
            )
        }

    private fun sanitize(
        value: Double,
    ): Double =
        if (value.isFinite()) {
            value.coerceIn(
                0.0,
                1.0,
            )
        } else {
            0.0
        }

    private fun Double.pow(
        exponent: Double,
    ): Double {
        if (!isFinite()) return 0.0
        return Math.pow(
            this,
            exponent,
        )
    }
}
