package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DenseLandmarkPrecisionContractEngineTest {

    private fun input(
        coverage: Double = 0.84,
        visibleCoverage: Double = 0.76,
        uncertaintyP95: Double = 0.20,
        crossViewAgreement: Double = 0.82,
        crossViewEvidenceAvailable: Boolean = true,
        supportingCameraCount: Int = 4,
        temporalStability: Double = 0.86,
        contactCoverage: Double = 0.62,
        contactEvidenceActive: Boolean = true,
        predictionBridgeRatio: Double = 0.08,
        acceptedObservations: Long = 128,
        rejectedObservations: Long = 4,
        staleObservations: Long = 0,
        nonFiniteObservations: Long = 0,
        hardFailure: Boolean = false,
    ) = DenseLandmarkPrecisionContractEngine.Input(
        coverage = coverage,
        visibleCoverage = visibleCoverage,
        uncertaintyP95 = uncertaintyP95,
        crossViewAgreement = crossViewAgreement,
        crossViewEvidenceAvailable = crossViewEvidenceAvailable,
        supportingCameraCount = supportingCameraCount,
        temporalStability = temporalStability,
        contactCoverage = contactCoverage,
        contactEvidenceActive = contactEvidenceActive,
        predictionBridgeRatio = predictionBridgeRatio,
        acceptedObservations = acceptedObservations,
        rejectedObservations = rejectedObservations,
        staleObservations = staleObservations,
        nonFiniteObservations = nonFiniteObservations,
        hardFailure = hardFailure,
    )

    @Test
    fun strongDenseEvidenceReachesAuthoritativeState() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val result = engine.evaluate(input())

        assertEquals(
            DenseLandmarkPrecisionContractEngine.State.READY,
            result.state,
        )
        assertTrue(result.authoritative)
        assertTrue(result.accepted)
        assertTrue(result.score >= 0.76)
        assertEquals(
            DenseLandmarkPrecisionContractEngine.FailureClass.NONE,
            result.failureClass,
        )
    }

    @Test
    fun weakCrossViewEvidenceTriggersNonAuthoritativeDegradation() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val result = engine.evaluate(
            input(
                crossViewAgreement = 0.42,
                supportingCameraCount = 2,
            ),
        )

        assertFalse(result.authoritative)
        assertTrue(
            result.state ==
                DenseLandmarkPrecisionContractEngine.State.DEGRADED ||
                result.state ==
                DenseLandmarkPrecisionContractEngine.State.HOLD,
        )
        assertTrue(
            result.failureClass ==
                DenseLandmarkPrecisionContractEngine.FailureClass.CROSS_VIEW,
        )
    }

    @Test
    fun severeUncertaintyHardBlocksEvidence() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val result = engine.evaluate(
            input(
                uncertaintyP95 = 0.86,
                coverage = 0.76,
                visibleCoverage = 0.70,
                temporalStability = 0.82,
            ),
        )

        assertTrue(result.hardFailure)
        assertEquals(
            DenseLandmarkPrecisionContractEngine.State.BLOCKED,
            result.state,
        )
        assertEquals(
            DenseLandmarkPrecisionContractEngine.FailureClass.UNCERTAINTY,
            result.failureClass,
        )
    }

    @Test
    fun excessivePredictionBridgeCannotBecomeAuthoritative() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val result = engine.evaluate(
            input(
                predictionBridgeRatio = 0.64,
                coverage = 0.84,
                visibleCoverage = 0.78,
                uncertaintyP95 = 0.18,
                crossViewAgreement = 0.84,
                temporalStability = 0.84,
            ),
        )

        assertFalse(result.authoritative)
        assertTrue(
            result.failureClass ==
                DenseLandmarkPrecisionContractEngine.FailureClass.PREDICTION ||
            result.failureClass ==
                DenseLandmarkPrecisionContractEngine.FailureClass.TEMPORAL,
        )
    }

    @Test
    fun missingCrossViewEvidenceIsDegradedWhenRedundancyIsRequired() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val result = engine.evaluate(
            input(
                crossViewEvidenceAvailable = false,
                supportingCameraCount = 1,
            ),
        )

        assertFalse(result.authoritative)
        assertTrue(
            result.warnings.any {
                it.contains("Cross-view dense evidence")
            },
        )
    }

    @Test
    fun singleCameraModeCanBeConfiguredWithoutCrossViewRequirement() {
        val engine = DenseLandmarkPrecisionContractEngine(
            DenseLandmarkPrecisionContractEngine.Config(
                minimumCrossViewCameras = 1,
                minimumCrossViewAgreement = 0.45,
            ),
        )

        val result = engine.evaluate(
            input(
                crossViewEvidenceAvailable = false,
                supportingCameraCount = 1,
                crossViewAgreement = 0.0,
            ),
        )

        assertTrue(result.accepted)
        assertTrue(
            result.state ==
                DenseLandmarkPrecisionContractEngine.State.READY ||
            result.state ==
                DenseLandmarkPrecisionContractEngine.State.DEGRADED,
        )
    }

    @Test
    fun runtimeConfigurationChangesTheContract() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val before = engine.evaluate(
            input(
                coverage = 0.72,
                visibleCoverage = 0.68,
                uncertaintyP95 = 0.27,
            ),
        )

        engine.configure(
            DenseLandmarkPrecisionContractEngine.Config(
                minimumCoverage = 0.82,
                minimumVisibleCoverage = 0.75,
                maximumUncertaintyP95 = 0.18,
                minimumCrossViewAgreement = 0.78,
                minimumTemporalStability = 0.82,
                maximumRejectedObservationRatio = 0.05,
                maximumPredictionBridgeRatio = 0.20,
                minimumContactCoverageWhenActive = 0.55,
                minimumEvidenceScore = 0.76,
                minimumAuthoritativeScore = 0.84,
            ),
        )

        val after = engine.evaluate(
            input(
                coverage = 0.72,
                visibleCoverage = 0.68,
                uncertaintyP95 = 0.27,
            ),
        )

        assertTrue(before.score > after.score)
        assertFalse(after.authoritative)
        assertTrue(
            after.failureClass !=
                DenseLandmarkPrecisionContractEngine.FailureClass.NONE,
        )
    }

    @Test
    fun nonFiniteEvidenceAndHardFailureArePreservedAsBlockers() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val result = engine.evaluate(
            input(
                nonFiniteObservations = 3,
                hardFailure = true,
            ),
        )

        assertTrue(result.hardFailure)
        assertEquals(
            DenseLandmarkPrecisionContractEngine.FailureClass.UPSTREAM_HARD_FAILURE,
            result.failureClass,
        )
        assertEquals(
            DenseLandmarkPrecisionContractEngine.State.BLOCKED,
            result.state,
        )
    }

    @Test
    fun rejectionRatioReducesContractScoreEvenWhenGeometryIsGood() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val clean = engine.evaluate(
            input(
                acceptedObservations = 80,
                rejectedObservations = 2,
            ),
        )

        val dirty = engine.evaluate(
            input(
                acceptedObservations = 40,
                rejectedObservations = 24,
            ),
        )

        assertTrue(clean.score > dirty.score)
        assertTrue(dirty.warnings.any { it.contains("rejection ratio") })
    }

    @Test
    fun contactEvidenceIsOptionalButStrictWhenActive() {
        val engine = DenseLandmarkPrecisionContractEngine()

        val inactive = engine.evaluate(
            input(
                contactEvidenceActive = false,
                contactCoverage = 0.0,
            ),
        )

        val active = engine.evaluate(
            input(
                contactEvidenceActive = true,
                contactCoverage = 0.10,
            ),
        )

        assertTrue(inactive.score > active.score)
        assertTrue(
            active.failureClass ==
                DenseLandmarkPrecisionContractEngine.FailureClass.CONTACT ||
            active.warnings.any { it.contains("Contact-aware") },
        )
    }
    @Test
    fun insufficient_evidence_cannot_reach_ready_even_with_strong_quality_scores() {
        val engine = DenseLandmarkPrecisionContractEngine(
            DenseLandmarkPrecisionContractEngine.Config(
                minimumCoverage = 0.20,
                minimumVisibleCoverage = 0.20,
                maximumUncertaintyP95 = 0.10,
                minimumCrossViewAgreement = 0.90,
                minimumTemporalStability = 0.90,
                minimumEvidenceScore = 0.50,
                minimumAuthoritativeScore = 0.60,
                fullEvidenceSampleCount = 256,
            ),
        )

        val result = engine.evaluate(
            DenseLandmarkPrecisionContractEngine.Input(
                coverage = 1.0,
                visibleCoverage = 1.0,
                uncertaintyP95 = 0.02,
                crossViewAgreement = 1.0,
                crossViewEvidenceAvailable = true,
                supportingCameraCount = 4,
                temporalStability = 1.0,
                contactCoverage = 1.0,
                contactEvidenceActive = true,
                predictionBridgeRatio = 0.0,
                acceptedObservations = 3L,
                rejectedObservations = 0L,
                staleObservations = 0L,
                nonFiniteObservations = 0L,
                hardFailure = false,
            ),
        )

        assertEquals(
            DenseLandmarkPrecisionContractEngine.State.HOLD,
            result.state,
        )
        assertTrue(!result.authoritative)
    }
}
