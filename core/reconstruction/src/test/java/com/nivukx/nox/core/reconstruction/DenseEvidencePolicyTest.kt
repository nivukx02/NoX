package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DenseEvidencePolicyTest {

    @Test
    fun singleViewEvidenceCannotClaimCrossViewAgreement() {
        assertEquals(
            0.0,
            DenseEvidencePolicy.crossViewAgreement(
                rawAgreement = 0.92,
                acceptedViewCount = 1,
            ),
        )
    }

    @Test
    fun multiViewAgreementIsPreservedAndClamped() {
        assertEquals(
            0.84,
            DenseEvidencePolicy.crossViewAgreement(
                rawAgreement = 0.84,
                acceptedViewCount = 3,
            ),
        )
        assertEquals(
            0.0,
            DenseEvidencePolicy.crossViewAgreement(
                rawAgreement = -1.0,
                acceptedViewCount = 2,
            ),
        )
        assertEquals(
            1.0,
            DenseEvidencePolicy.crossViewAgreement(
                rawAgreement = 9.0,
                acceptedViewCount = 2,
            ),
        )
    }

    @Test
    fun predictedEvidenceCannotClaimFullTemporalAlignment() {
        val observed =
            DenseEvidencePolicy.temporalAlignmentConfidence(
                temporalStability = 0.90,
                predicted = false,
            )
        val predicted =
            DenseEvidencePolicy.temporalAlignmentConfidence(
                temporalStability = 0.90,
                predicted = true,
            )

        assertTrue(observed > predicted)
        assertEquals(0.648, predicted, absoluteTolerance = 0.000001)
    }

    @Test
    fun observationWeightIsBoundedAndDropsWithUncertainty() {
        val clean =
            DenseEvidencePolicy.observationWeight(
                confidence = 0.95,
                visibility = 0.95,
                uncertainty = 0.05,
                temporalStability = 0.95,
                crossViewAgreement = 0.90,
            )
        val uncertain =
            DenseEvidencePolicy.observationWeight(
                confidence = 0.95,
                visibility = 0.95,
                uncertainty = 0.85,
                temporalStability = 0.95,
                crossViewAgreement = 0.90,
            )

        assertTrue(clean in 0.0..1.0)
        assertTrue(uncertain in 0.0..1.0)
        assertTrue(clean > uncertain)
    }
}
