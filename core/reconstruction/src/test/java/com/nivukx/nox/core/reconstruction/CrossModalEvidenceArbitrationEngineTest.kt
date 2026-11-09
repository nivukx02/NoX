package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertTrue
import org.junit.Test

class CrossModalEvidenceArbitrationEngineTest {

    private val timestamp = 1_000_000_000L

    @Test
    fun unreliableFaceIsExcludedWhileBodyAndHandRemainUseful() {
        val engine = CrossModalEvidenceArbitrationEngine(
            CrossModalEvidenceArbitrationEngine.Config(
                minimumSelectedModalities = 1,
                maximumSelectedModalities = 4,
                oodQuarantineThreshold = 0.82,
            ),
        )

        val result =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.94,
                            oodRisk = 0.02,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.FACE,
                            confidence = 0.93,
                            oodRisk = 0.97,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND,
                            confidence = 0.84,
                            oodRisk = 0.04,
                        ),
                    ),
                ),
            )

        checkSelected(
            result,
            CrossModalEvidenceArbitrationEngine.Modality.BODY,
        )
        checkSelected(
            result,
            CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND,
        )
        checkNotSelected(
            result,
            CrossModalEvidenceArbitrationEngine.Modality.FACE,
        )

        assertTrue(result.fusionConfidence > 0.60)
        assertTrue(
            result.warnings.any {
                it.contains("out-of-distribution")
            },
        )
    }

    @Test
    fun highRedundancyCanReduceContributionWithoutDeletingTheModality() {
        val engine = CrossModalEvidenceArbitrationEngine(
            CrossModalEvidenceArbitrationEngine.Config(
                redundancyPenaltyWeight = 0.78,
                minimumSelectedModalities = 2,
                maximumSelectedModalities = 4,
            ),
        )

        val result =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.94,
                            redundancy = 0.05,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.FACE,
                            confidence = 0.91,
                            redundancy = 0.93,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND,
                            confidence = 0.87,
                            redundancy = 0.08,
                        ),
                    ),
                ),
            )

        val face =
            result.modalities.single {
                it.modality ==
                    CrossModalEvidenceArbitrationEngine.Modality.FACE
            }
        val hand =
            result.modalities.single {
                it.modality ==
                    CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND
            }

        assertTrue(hand.marginalUtility > face.marginalUtility)
        assertTrue(result.selectedModalityCount >= 2)
    }

    @Test
    fun noisyOrRetainsStrongIndependentEvidence() {
        val engine = CrossModalEvidenceArbitrationEngine()

        val single =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.72,
                            uncertaintyMeters = 0.06,
                        ),
                    ),
                ),
            )

        engine.reset()

        val fused =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.72,
                            uncertaintyMeters = 0.06,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.FACE,
                            confidence = 0.71,
                            uncertaintyMeters = 0.05,
                            redundancy = 0.10,
                        ),
                    ),
                ),
            )

        assertTrue(
            fused.noisyOrSupport > single.noisyOrSupport,
        )
        assertTrue(
            fused.fusionConfidence >= single.fusionConfidence,
        )
    }

    @Test
    fun nearZeroEvidenceDoesNotCreateNoisyOrSupport() {
        val engine =
            CrossModalEvidenceArbitrationEngine(
                CrossModalEvidenceArbitrationEngine.Config(
                    minimumSelectedModalities = 1,
                    minimumAdjustedReliability = 0.0,
                    noisyOrWeightFloor = 0.20,
                ),
            )

        val result =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = 8_000_000_000L,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.001,
                            uncertaintyMeters = 1.0,
                            oodRisk = 0.99,
                        ),
                    ),
                ),
            )

        assertTrue(result.noisyOrSupport < 0.05)
    }

    @Test
    fun repeatedOutOfDistributionEvidenceQuarantinesOneModality() {
        val engine = CrossModalEvidenceArbitrationEngine(
            CrossModalEvidenceArbitrationEngine.Config(
                quarantineAfterBadFrames = 3,
                recoveryFrames = 4,
                oodQuarantineThreshold = 0.80,
            ),
        )

        repeat(4) { index ->
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp + index * 33_000_000L,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT,
                            confidence = 0.92,
                            oodRisk = 0.96,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.88,
                            oodRisk = 0.02,
                        ),
                    ),
                ),
            )
        }

        val result =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp + 5 * 33_000_000L,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT,
                            confidence = 0.92,
                            oodRisk = 0.96,
                        ),
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.88,
                            oodRisk = 0.02,
                        ),
                    ),
                ),
            )

        val eye =
            result.modalities.single {
                it.modality ==
                    CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT
            }

        assertTrue(eye.quarantined)
        assertTrue(
            result.warnings.any {
                it.contains("quarantined")
            },
        )
    }

    @Test
    fun reconfiguringWithEquivalentPolicyPreservesTemporalReliability() {
        val config =
            CrossModalEvidenceArbitrationEngine.Config()

        val engine =
            CrossModalEvidenceArbitrationEngine(config)

        engine.arbitrate(
            CrossModalEvidenceArbitrationEngine.Input(
                timestampNanos = timestamp,
                modalities = listOf(
                    evidence(
                        CrossModalEvidenceArbitrationEngine.Modality.BODY,
                        confidence = 0.91,
                    ),
                ),
            ),
        )

        val learnedBefore =
            engine.currentReliability(
                CrossModalEvidenceArbitrationEngine.Modality.BODY,
            )

        engine.configure(config)

        val learnedAfter =
            engine.currentReliability(
                CrossModalEvidenceArbitrationEngine.Modality.BODY,
            )

        assertTrue(learnedAfter >= learnedBefore * 0.98)
    }

    @Test
    fun predictionOnlyEvidenceCannotTeachReliability() {
        val engine = CrossModalEvidenceArbitrationEngine()

        val measured =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.90,
                            predictionOnly = false,
                        ),
                    ),
                ),
            )

        val predicted =
            engine.arbitrate(
                CrossModalEvidenceArbitrationEngine.Input(
                    timestampNanos = timestamp + 33_000_000L,
                    modalities = listOf(
                        evidence(
                            CrossModalEvidenceArbitrationEngine.Modality.BODY,
                            confidence = 0.10,
                            oodRisk = 0.99,
                            predictionOnly = true,
                        ),
                    ),
                ),
            )

        val measuredReliability =
            measured.modalities.single {
                it.modality ==
                    CrossModalEvidenceArbitrationEngine.Modality.BODY
            }.adjustedReliability
        val predictedReliability =
            predicted.modalities.single {
                it.modality ==
                    CrossModalEvidenceArbitrationEngine.Modality.BODY
            }.adjustedReliability

        assertTrue(
            predictedReliability >= measuredReliability * 0.90,
        )
    }

    private fun evidence(
        modality: CrossModalEvidenceArbitrationEngine.Modality,
        confidence: Double,
        uncertaintyMeters: Double = 0.03,
        redundancy: Double = 0.10,
        oodRisk: Double = 0.0,
        predicted: Boolean = false,
        predictionOnly: Boolean = false,
    ) =
        CrossModalEvidenceArbitrationEngine.ModalityEvidence(
            modality = modality,
            confidence = confidence,
            uncertaintyMeters = uncertaintyMeters,
            redundancy = redundancy,
            freshnessSeconds = 0.0,
            oodRisk = oodRisk,
            predicted = predicted,
            predictionOnly = predictionOnly,
        )

    private fun checkSelected(
        result: CrossModalEvidenceArbitrationEngine.Result,
        modality: CrossModalEvidenceArbitrationEngine.Modality,
    ) {
        assertTrue(
            result.modalities.single {
                it.modality == modality
            }.selected,
        )
    }

    private fun checkNotSelected(
        result: CrossModalEvidenceArbitrationEngine.Result,
        modality: CrossModalEvidenceArbitrationEngine.Modality,
    ) {
        assertTrue(
            !result.modalities.single {
                it.modality == modality
            }.selected,
        )
    }
}
