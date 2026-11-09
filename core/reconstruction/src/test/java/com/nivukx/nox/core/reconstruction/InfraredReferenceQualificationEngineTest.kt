package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InfraredReferenceQualificationEngineTest {
    private fun qualified(deltaP95: Double, deltaP99: Double) =
        ReferenceParityCohortEngine.Result(
            matchedPairs = 40,
            acceptedNoxPairs = 40,
            rejectedNoxPairs = 0,
            aggregateNoxP95Millimeters = 0.70 + deltaP95,
            aggregateNoxP99Millimeters = 1.40 + deltaP99,
            aggregateReferenceP95Millimeters = 0.70,
            aggregateReferenceP99Millimeters = 1.40,
            aggregateDeltaP95Millimeters = deltaP95,
            aggregateDeltaP99Millimeters = deltaP99,
            maximumReferenceUncertaintyMillimeters = 0.30,
            acceptedPairRatio = 1.0,
            confidence = 0.94,
            evidenceQualified = true,
            pairDeltas = emptyList(),
            blockingReasons = emptyList(),
            warnings = emptyList(),
        )

    @Test
    fun classifiesNoxLowerErrorOnlyBeyondTieBand() {
        val result = InfraredReferenceQualificationEngine().evaluate(
            qualified(-0.20, -0.35),
        )
        assertEquals(
            InfraredReferenceQualificationEngine.Outcome.NOX_LOWER_ERROR,
            result.outcome,
        )
        assertTrue(result.noxPhysicallyFavored)
    }

    @Test
    fun classifiesStatisticalParityInsideTieBand() {
        val result = InfraredReferenceQualificationEngine().evaluate(
            qualified(0.01, -0.02),
        )
        assertEquals(
            InfraredReferenceQualificationEngine.Outcome.STATISTICAL_PARITY,
            result.outcome,
        )
        assertTrue(result.statisticalParity)
    }

    @Test
    fun classifiesReferenceLowerError() {
        val result = InfraredReferenceQualificationEngine().evaluate(
            qualified(0.20, 0.35),
        )
        assertEquals(
            InfraredReferenceQualificationEngine.Outcome.REFERENCE_LOWER_ERROR,
            result.outcome,
        )
        assertTrue(result.referencePhysicallyFavored)
    }

    @Test
    fun accepts_active_optical_marker_reference_family() {
        val cohort =
            qualified(-0.30, -0.45)

        val result =
            InfraredReferenceQualificationEngine().evaluate(
                cohort.copy(
                    warnings =
                        cohort.warnings +
                            "active-optical-marker-reference",
                ),
            )

        assertTrue(result.qualified)
        assertEquals(
            InfraredReferenceQualificationEngine.Outcome.NOX_LOWER_ERROR,
            result.outcome,
        )
    }

    @Test
    fun refusesSuperiorityWithoutQualifiedEvidence() {
        val result = InfraredReferenceQualificationEngine().evaluate(
            qualified(-0.30, -0.40).copy(evidenceQualified = false),
        )
        assertEquals(
            InfraredReferenceQualificationEngine.Outcome.INCONCLUSIVE,
            result.outcome,
        )
        assertFalse(result.noxPhysicallyFavored)
    }
}
