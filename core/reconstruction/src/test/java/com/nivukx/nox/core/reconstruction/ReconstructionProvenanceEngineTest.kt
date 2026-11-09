package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.PrecisionAuthoritySnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReconstructionProvenanceEngineTest {
    @Test
    fun separatesObservedMeasuredAndInferredRegions() {
        val result = ReconstructionProvenanceEngine().aggregate(listOf(
            ReconstructionProvenanceEngine.RegionInput(1, 2, mapOf(EvidenceOrigin.OBSERVED to 60, EvidenceOrigin.GEOMETRICALLY_CONSTRAINED to 20, EvidenceOrigin.LEARNED_PRIOR to 20), 0.9),
        ))
        assertEquals(0.60, result.observedRatio, 1e-9)
        assertEquals(0.80, result.measuredRatio, 1e-9)
        assertEquals(0.20, result.inferredRatio, 1e-9)
        assertEquals(0.00, result.generationRatio, 1e-9)
    }

    @Test
    fun precisionNeedsReferenceQualificationAndEvidenceFloor() {
        val snapshot = PrecisionAuthoritySnapshot(0.0002,0.0008,0.0014,0.0001,8,0.8,1.2,100_000.0,0.8,600,true,false,"fp")
        assertFalse(ReconstructionProvenanceEngine().qualifies(snapshot))
        assertTrue(snapshot.modeledSubMillimeterFeasible)
    }
}