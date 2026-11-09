package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DynamicGaussianSurfaceRefinementEngineTest {
    @Test
    fun inferredSurfaceInflatesCovarianceAndKeepsBoundedDisplacement() {
        val result = DynamicGaussianSurfaceRefinementEngine().refine(listOf(
            DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                1L, Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,1.0), UncertaintyEnvelope(1e-4,1e-4,1e-4),
                0.9, EvidenceOrigin.LEARNED_PRIOR, Vec3(1.0,0.0,0.0),
            ),
        ))[0]
        assertTrue(result.positionMeters.x <= 0.20)
        assertTrue(result.covariance.xxMeters2 > 1e-4)
    }

    @Test
    fun observedSurfaceRetainsOriginalCovariance() {
        val covariance = UncertaintyEnvelope(1e-4,2e-4,3e-4, confidence=0.95)
        val result = DynamicGaussianSurfaceRefinementEngine().refine(listOf(
            DynamicGaussianSurfaceRefinementEngine.SurfacePoint(1L,Vec3(0.0,0.0,0.0),Vec3(0.0,0.0,1.0),covariance,0.95,EvidenceOrigin.OBSERVED),
        ))[0]
        assertEquals(covariance, result.covariance)
    }
    @Test
    fun sdfFlowConstraintRegularizesNormalDisplacement() {
        val result = DynamicGaussianSurfaceRefinementEngine().refine(
            listOf(
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 5L,
                    positionMeters = Vec3(0.0,0.0,0.0),
                    normal = Vec3(0.0,0.0,1.0),
                    covariance = UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=0.95),
                    confidence = 0.95,
                    origin = EvidenceOrigin.OBSERVED,
                    flowDeltaMeters = Vec3(0.0,0.0,0.10),
                    signedDistanceDeltaMeters = 0.0,
                ),
            ),
        ).single()
        assertTrue(result.positionMeters.z < 0.06)
    }
    @Test
    fun occludedSurfaceReceivesLowerAuthorityAndOpacity() {
        val covariance = UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=0.95)
        val result = DynamicGaussianSurfaceRefinementEngine().refine(
            listOf(
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 10L,
                    positionMeters = Vec3(0.0,0.0,0.0),
                    normal = Vec3(0.0,0.0,1.0),
                    covariance = covariance,
                    confidence = 0.95,
                    origin = EvidenceOrigin.OBSERVED,
                    visibility = 1.0,
                    occlusionRisk = 0.0,
                    photometricConsistency = 1.0,
                    normalConsistency = 1.0,
                    independentCameraCount = 8,
                ),
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 11L,
                    positionMeters = Vec3(0.01,0.0,0.0),
                    normal = Vec3(0.0,0.0,1.0),
                    covariance = covariance,
                    confidence = 0.95,
                    origin = EvidenceOrigin.OBSERVED,
                    visibility = 0.55,
                    occlusionRisk = 0.80,
                    photometricConsistency = 0.50,
                    normalConsistency = 0.60,
                    independentCameraCount = 1,
                ),
            ),
        )
        val strong = result.first { it.id == 10L }
        val weak = result.first { it.id == 11L }
        assertTrue(strong.authority > weak.authority)
        assertTrue(strong.opacity > weak.opacity)
    }
}

class DynamicGaussianSurfaceRefinementEngineFrontierTest {
    @Test
    fun nearbySurfaceEvidenceProducesNeighborhoodSupport() {
        val covariance = UncertaintyEnvelope(1e-4, 1e-4, 1e-4, confidence = 0.98)
        val result = DynamicGaussianSurfaceRefinementEngine().refine(
            listOf(
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 20L,
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    covariance = covariance,
                    confidence = 0.98,
                    origin = EvidenceOrigin.OBSERVED,
                    independentCameraCount = 6,
                    semanticRegionId = 4,
                ),
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 21L,
                    positionMeters = Vec3(0.01, 0.0, 0.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    covariance = covariance,
                    confidence = 0.95,
                    origin = EvidenceOrigin.OBSERVED,
                    independentCameraCount = 5,
                    semanticRegionId = 4,
                ),
            ),
        )
        val primary = result.first { it.id == 20L }
        assertTrue(primary.neighborhoodSupport >= 1)
        assertTrue(primary.cameraSupport >= 6)
    }

    @Test
    fun inferredGaussianRetainsLowerAuthorityThanEquivalentObservedEvidence() {
        val covariance = UncertaintyEnvelope(1e-4, 1e-4, 1e-4, confidence = 0.98)
        val result = DynamicGaussianSurfaceRefinementEngine().refine(
            listOf(
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 30L,
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    covariance = covariance,
                    confidence = 0.98,
                    origin = EvidenceOrigin.OBSERVED,
                    independentCameraCount = 5,
                ),
                DynamicGaussianSurfaceRefinementEngine.SurfacePoint(
                    id = 31L,
                    positionMeters = Vec3(0.02, 0.0, 0.0),
                    normal = Vec3(0.0, 0.0, 1.0),
                    covariance = covariance,
                    confidence = 0.98,
                    origin = EvidenceOrigin.LEARNED_PRIOR,
                    independentCameraCount = 5,
                ),
            ),
        )
        assertTrue(
            result.first { it.id == 30L }.authority >
                result.first { it.id == 31L }.authority,
        )
    }
}
