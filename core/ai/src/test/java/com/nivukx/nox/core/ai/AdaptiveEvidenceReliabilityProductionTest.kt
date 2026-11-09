package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.PerceptionSource

/**
 * Behavioral tests for adaptive source reliability.
 *
 * The tests focus on failure isolation and recovery semantics rather than exact numeric tuning.
 */
class AdaptiveEvidenceReliabilityProductionTest {

    fun predictionOnlyEvidenceDoesNotTrainSource() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                baseReliability = 0.74,
                minimumReliability = 0.16,
                historyLength = 8,
            ),
        )

        val timestamp = 1_000_000_000L

        val baseline = engine.observe(
            observation = observation(
                timestampNanos = timestamp,
                confidence = 0.95,
                residual = 0.01,
                predictionOnly = false,
            ),
            consensusConfidence = 0.95,
        )

        val beforeSamples = baseline.sampleCount
        val beforeReliability = baseline.value

        val predicted = engine.observe(
            observation = observation(
                timestampNanos = timestamp + 33_000_000L,
                confidence = 0.20,
                residual = 100.0,
                predictionOnly = true,
                outOfDistributionRisk = 1.0,
            ),
            consensusConfidence = 0.10,
        )

        check(predicted.sampleCount == beforeSamples)
        check(predicted.value >= beforeReliability - 1e-9)
    }

    fun extremeResidualIsScoredAgainstPriorHistory() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                historyLength = 16,
                quarantineAfterBadFrames = 5,
            ),
        )

        repeat(8) { index ->
            engine.observe(
                observation = observation(
                    timestampNanos = 1_000_000_000L + index * 33_000_000L,
                    confidence = 0.94,
                    residual = 0.04 + index * 0.001,
                    predictionOnly = false,
                ),
                consensusConfidence = 0.94,
            )
        }

        val result = engine.observe(
            observation = observation(
                timestampNanos = 1_300_000_000L,
                confidence = 0.55,
                residual = 40.0,
                predictionOnly = false,
            ),
            consensusConfidence = 0.90,
        )

        check(result.degraded || result.value < 0.90)
        check(result.robustResidualScale < 10.0)
    }

    fun persistentOodQuarantinesSource() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                quarantineAfterBadFrames = 3,
                outOfDistributionQuarantineThreshold = 0.80,
            ),
        )

        repeat(5) { index ->
            engine.observe(
                observation = observation(
                    timestampNanos = 1_000_000_000L + index * 33_000_000L,
                    confidence = 0.90,
                    residual = 0.10,
                    predictionOnly = false,
                    outOfDistributionRisk = 0.95,
                ),
                consensusConfidence = 0.90,
            )
        }

        val snapshot = engine.snapshot(1_200_000_000L)
        val reliability = snapshot.values.values.single()

        check(reliability.quarantined)
        check(
            snapshot.warnings.any {
                it.contains("quarantined")
            },
        )
        check(snapshot.effectiveConsensus <= 1.0)
    }

    fun quarantinedSourceReceivesMinimalWeight() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                quarantineAfterBadFrames = 2,
                quarantineWeight = 0.02,
            ),
        )

        repeat(4) { index ->
            engine.observe(
                observation = observation(
                    timestampNanos = 1_000_000_000L + index * 33_000_000L,
                    confidence = 0.85,
                    residual = 2.0,
                    predictionOnly = false,
                    outOfDistributionRisk = 0.95,
                ),
                consensusConfidence = 0.90,
            )
        }

        val reliability = engine.snapshot(1_200_000_000L).values.values.single()
        check(reliability.quarantined)

        val weight = engine.weight(
            source = PerceptionSource.ML_KIT,
            ageMillis = 0.0,
            reportedConfidence = 1.0,
            outOfDistributionRisk = 0.95,
        )

        check(weight < 0.05)
    }

    fun recoveryRequiresRepeatedCleanEvidence() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                quarantineAfterBadFrames = 2,
                recoveryFrames = 4,
                outOfDistributionQuarantineThreshold = 0.80,
            ),
        )

        repeat(4) { index ->
            engine.observe(
                observation = observation(
                    timestampNanos = 1_000_000_000L + index * 33_000_000L,
                    confidence = 0.82,
                    residual = 2.0,
                    predictionOnly = false,
                    outOfDistributionRisk = 0.95,
                ),
                consensusConfidence = 0.90,
            )
        }

        check(
            engine.snapshot(1_200_000_000L)
                .values.values.single()
                .quarantined,
        )

        repeat(3) { index ->
            engine.observe(
                observation = observation(
                    timestampNanos = 1_400_000_000L + index * 33_000_000L,
                    confidence = 0.98,
                    residual = 0.03,
                    predictionOnly = false,
                    outOfDistributionRisk = 0.02,
                ),
                consensusConfidence = 0.97,
            )
        }

        check(
            engine.snapshot(1_500_000_000L)
                .values.values.single()
                .quarantined,
        )

        engine.observe(
            observation = observation(
                timestampNanos = 1_700_000_000L,
                confidence = 0.98,
                residual = 0.03,
                predictionOnly = false,
                outOfDistributionRisk = 0.01,
            ),
            consensusConfidence = 0.97,
        )

        check(
            !engine.snapshot(1_700_000_000L)
                .values.values.single()
                .quarantined,
        )
    }

    fun stalePruningKeepsQuarantinedStateForRecovery() {
        val engine = AdaptiveEvidenceReliability(
            AdaptiveEvidenceReliability.Config(
                quarantineAfterBadFrames = 1,
                staleCutoffMillis = 500.0,
            ),
        )

        engine.observe(
            observation = observation(
                timestampNanos = 1_000_000_000L,
                confidence = 0.40,
                residual = 4.0,
                predictionOnly = false,
                outOfDistributionRisk = 0.95,
            ),
            consensusConfidence = 0.50,
        )

        engine.snapshot(1_000_000_000L)
        engine.prune(3_000_000_000L)

        check(
            engine.snapshot(3_000_000_000L)
                .values
                .containsKey(PerceptionSource.ML_KIT),
        )
    }

    private fun observation(
        timestampNanos: Long,
        confidence: Double,
        residual: Double,
        predictionOnly: Boolean,
        outOfDistributionRisk: Double = 0.0,
        source: PerceptionSource = PerceptionSource.ML_KIT,
    ): AdaptiveEvidenceReliability.Observation =
        AdaptiveEvidenceReliability.Observation(
            source = source,
            timestampNanos = timestampNanos,
            reportedConfidence = confidence,
            residual = residual,
            disagreement = residual.coerceIn(0.0, 1.0),
            stability = confidence,
            ageMillis = 0.0,
            accepted = confidence >= 0.10,
            outOfDistributionRisk = outOfDistributionRisk,
            predictionOnly = predictionOnly,
        )
}

fun main() {
    val test = AdaptiveEvidenceReliabilityProductionTest()
    test.predictionOnlyEvidenceDoesNotTrainSource()
    test.extremeResidualIsScoredAgainstPriorHistory()
    test.persistentOodQuarantinesSource()
    test.quarantinedSourceReceivesMinimalWeight()
    test.recoveryRequiresRepeatedCleanEvidence()
    test.stalePruningKeepsQuarantinedStateForRecovery()
}
