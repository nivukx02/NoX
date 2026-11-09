package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierEvidenceModelsTest {
    @Test
    fun midpointAndRowTimestampAreDeterministic() {
        val timing = CaptureTimingEvidence(900L, 800L, 900L, 1_000L, 20L, 20.0, 50.0, true, "camera2")
        assertEquals(200L, timing.exposureDurationNanos)
        assertEquals(890.0, timing.rowTimestamp(0, 10), 1e-9)
        assertEquals(910.0, timing.rowTimestamp(9, 10), 1e-9)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsTimingWithWrongMidpoint() {
        CaptureTimingEvidence(100L, 0L, 70L, 100L, 0L, 0.0, 1.0, false, "test")
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteCovariance() {
        UncertaintyEnvelope(Double.NaN, 1.0, 1.0)
    }

    @Test
    fun fullCovarianceRetainsCrossAxisInformation() {
        val envelope = UncertaintyEnvelope(4e-6, 9e-6, 16e-6, 1e-6, 2e-6, -2e-6, 0.97)
        assertTrue(envelope.determinant() >= -1e-12)
        assertTrue(envelope.principalVarianceUpperBound() >= 16e-6)
    }

    @Test
    fun observedEvidenceHasHigherAuthorityThanLearnedCompletion() {
        val provenance = ObservationProvenance("a", "cal", "tim", "model", EvidenceOrigin.OBSERVED, 3_000L, 1L, 0L)
        val observed = WorldSpaceProposal(
            actorId = 1, pointId = 2, timestampNanos = 3_000L,
            positionMeters = Vec3(1.0, 2.0, 3.0),
            uncertainty = UncertaintyEnvelope(1e-6, 1e-6, 1e-6),
            confidence = 0.9, geometricSupport = 0.9, temporalSupport = 0.9, learnedSupport = 0.9,
            independentCameraCount = 4, viewDiversity = 0.8, origin = EvidenceOrigin.OBSERVED,
            provenance = listOf(provenance),
        )
        assertTrue(observed.authorityScore > observed.copy(origin = EvidenceOrigin.LEARNED_PRIOR).authorityScore)
    }

    @Test(expected = IllegalArgumentException::class)
    fun actorProposalRejectsDuplicateCameraProvenance() {
        val p = ObservationProvenance("same", "cal", "tim", "model", EvidenceOrigin.OBSERVED, 1L, 1L, 0L)
        WorldSpaceProposal(
            actorId = 1, pointId = 1, timestampNanos = 1L, positionMeters = Vec3(0.0, 0.0, 0.0),
            uncertainty = UncertaintyEnvelope(1.0, 1.0, 1.0), confidence = 0.5, geometricSupport = 0.5,
            temporalSupport = 0.5, learnedSupport = 0.5, independentCameraCount = 2, viewDiversity = 0.5,
            origin = EvidenceOrigin.OBSERVED, provenance = listOf(p, p),
        )
    }

    @Test
    fun regionProvenanceSeparatesMeasuredAndInferredFractions() {
        val region = ReconstructionRegionProvenance(4, 20, 0.45, 0.25, 0.10, 0.10, 0.05, 0.92)
        assertEquals(0.70, region.measuredRatio, 1e-9)
        assertEquals(0.25, region.inferredRatio, 1e-9)
        assertTrue(region.authorityWeightedRatio > 0.55)
    }

    @Test
    fun precisionSnapshotDoesNotConfuseModeledFeasibilityWithPhysicalQualification() {
        val snapshot = PrecisionAuthoritySnapshot(
            0.0002, 0.0006, 0.0014, 0.0001, 8, 0.8, 1.2, 200_000.0, 0.9, 1_200, true, false, "abc",
        )
        assertFalse(snapshot.physicallyQualified(0.001, 0.002, 4, 600))
    }
}