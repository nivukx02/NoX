package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.WorldSpaceProposal
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierEvidenceAuthorityEngineTest {
    @Test fun learnedEvidenceCannotDominateGeometryAuthority() {
        fun proposal(geometry:Double,learned:Double)=WorldSpaceProposal(
            1,2,1L,Vec3(0.0,0.0,1.0),uncertainty=UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=0.9),
            confidence=0.9,geometricSupport=geometry,temporalSupport=0.8,learnedSupport=learned,independentCameraCount=4,viewDiversity=0.9,
            origin=if(geometry>0.5) EvidenceOrigin.GEOMETRICALLY_CONSTRAINED else EvidenceOrigin.LEARNED_PRIOR,provenance=emptyList())
        val result=FrontierEvidenceAuthorityEngine().evaluate(
            listOf(proposal(0.95,0.10),proposal(0.90,1.0)),
            denseTrackCount=1,actorCount=2,
        )
        assertTrue(result.geometryAuthority >= result.learnedAuthority * 0.85)
        assertTrue(result.authorityScore <= 1.0)
        assertTrue(result.effectiveEvidenceCount > 0.0)
        assertTrue(result.correlationAdjustment in 0.0..1.0)
    }
    @Test
    fun weakMetricEvidenceIsSurfacedWithoutOverridingGeometry() {
        fun proposal(
            geometry: Double,
            metricConfidence: Double,
            diversity: Double,
        ) =
            WorldSpaceProposal(
                actorId = 7,
                pointId = 3,
                timestampNanos = 2L,
                positionMeters = Vec3(0.0, 0.0, 1.0),
                uncertainty =
                    UncertaintyEnvelope(
                        1e-4,
                        1e-4,
                        1e-4,
                        confidence = metricConfidence,
                    ),
                confidence = 0.92,
                geometricSupport = geometry,
                temporalSupport = 0.88,
                learnedSupport = 0.05,
                independentCameraCount = 4,
                viewDiversity = diversity,
                origin = EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,
                provenance = emptyList(),
            )

        val result =
            FrontierEvidenceAuthorityEngine().evaluate(
                listOf(
                    proposal(0.94, 0.40, 0.45),
                    proposal(0.92, 0.42, 0.48),
                ),
                denseTrackCount = 2,
                actorCount = 2,
            )

        assertTrue(result.metricAuthority < 0.50)
        assertTrue(
            result.warnings.contains(
                "metric-evidence-authority-low",
            ),
        )
        assertTrue(result.authorityScore in 0.0..1.0)
    }

    @Test fun identityAmbiguityMonotonicallyReducesAuthority() {
        fun proposal() = WorldSpaceProposal(
            1, 2, 1L, Vec3(0.0,0.0,1.0),
            uncertainty = UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=0.9),
            confidence=0.9, geometricSupport=0.95, temporalSupport=0.85, learnedSupport=0.10,
            independentCameraCount=4, viewDiversity=0.9,
            origin=EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,
            provenance=emptyList(),
        )
        val engine = FrontierEvidenceAuthorityEngine()
        val clear = engine.evaluate(listOf(proposal()), denseTrackCount=2, actorCount=2, identityAmbiguityRisk=0.0)
        val ambiguous = engine.evaluate(listOf(proposal()), denseTrackCount=2, actorCount=2, identityAmbiguityRisk=1.0)
        assertTrue(ambiguous.authorityScore < clear.authorityScore)
        assertTrue(ambiguous.identityAmbiguityRisk == 1.0)
        assertTrue(ambiguous.warnings.contains("identity-ambiguity-reduces-authority"))
    )
}