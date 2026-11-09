package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DenseMeshQuality
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReconstructionSurfaceReadinessEngineTest {

    private fun quality(
        geometric: Double,
        temporal: Double,
        reprojectionPx: Double,
        uncertaintyMeters: Double,
        normalCoverage: Double = 0.95,
        watertightness: Double = 0.92,
    ) = DenseMeshQuality(
        timestampNanos = 1_000_000_000L,
        vertexCount = 2400,
        triangleCount = 4200,
        meanReprojectionErrorPx = reprojectionPx,
        watertightnessScore = watertightness,
        temporalStability = temporal,
        geometricConfidence = geometric,
        normalCoverage = normalCoverage,
        meanUncertaintyMeters = uncertaintyMeters,
        reprojectionEvidenceAvailable = true,
        watertightnessEvidenceAvailable = true,
        temporalEvidenceAvailable = true,
    )

    @Test
    fun healthySurfaceBecomesReady() {
        val result = ReconstructionSurfaceReadinessEngine().evaluate(
            ReconstructionSurfaceReadinessEngine.Input(
                timestampNanos = 1_000_000_000L,
                meshQuality = quality(0.94, 0.93, 1.2, 0.018),
                surfaceConfidence = 0.95,
                temporalConfidence = 0.93,
                uncertaintyP95Meters = 0.028,
                uncertaintyBudgetMeters = 0.08,
                reprojectionErrorPx = 1.2,
                reprojectionBudgetPx = 3.0,
                supportingViewConfidence = 0.96,
                scaleConfidence = 0.98,
                lightingConfidence = 0.90,
                occlusionConfidence = 0.94,
            ),
        )

        assertTrue(result.accepted)
        assertEquals(ReconstructionSurfaceReadinessEngine.Band.READY, result.band)
        assertTrue(result.score >= 0.78)
        assertTrue(result.blockers.isEmpty())
    }

    @Test
    fun missingCriticalEvidenceNeverClaimsReady() {
        val result = ReconstructionSurfaceReadinessEngine().evaluate(
            ReconstructionSurfaceReadinessEngine.Input(
                timestampNanos = 2_000_000_000L,
                meshQuality = null,
                surfaceConfidence = null,
                temporalConfidence = null,
                uncertaintyP95Meters = null,
                uncertaintyBudgetMeters = 0.08,
                reprojectionErrorPx = null,
                reprojectionBudgetPx = 3.0,
                supportingViewConfidence = 0.50,
                scaleConfidence = null,
            ),
        )

        assertEquals(ReconstructionSurfaceReadinessEngine.Band.UNKNOWN, result.band)
        assertTrue(!result.accepted)
        assertTrue(result.blockers.isNotEmpty())
    }

    @Test
    fun severeGeometryAndUncertaintyBlockSurface() {
        val result = ReconstructionSurfaceReadinessEngine().evaluate(
            ReconstructionSurfaceReadinessEngine.Input(
                timestampNanos = 3_000_000_000L,
                meshQuality = quality(
                    geometric = 0.18,
                    temporal = 0.32,
                    reprojectionPx = 14.0,
                    uncertaintyMeters = 0.65,
                    normalCoverage = 0.22,
                    watertightness = 0.20,
                ),
                surfaceConfidence = 0.15,
                temporalConfidence = 0.25,
                uncertaintyP95Meters = 0.72,
                uncertaintyBudgetMeters = 0.08,
                reprojectionErrorPx = 16.0,
                reprojectionBudgetPx = 3.0,
                supportingViewConfidence = 0.18,
                scaleConfidence = 0.30,
                lightingConfidence = 0.25,
                occlusionConfidence = 0.20,
            ),
        )

        assertEquals(ReconstructionSurfaceReadinessEngine.Band.BLOCKED, result.band)
        assertTrue(!result.accepted)
        assertTrue(result.blockers.any { it.contains("uncertainty") })
        assertTrue(result.blockers.any { it.contains("geometry") })
    }

    @Test
    fun rejectedCanonicalSurfacePriorBlocksSurfaceReadiness() {
        val result =
            ReconstructionSurfaceReadinessEngine().evaluate(
                ReconstructionSurfaceReadinessEngine.Input(
                    timestampNanos = 5_000_000_000L,
                    meshQuality = quality(
                        geometric = 0.94,
                        temporal = 0.92,
                        reprojectionPx = 1.1,
                        uncertaintyMeters = 0.018,
                    ),
                    surfaceConfidence = 0.94,
                    temporalConfidence = 0.92,
                    uncertaintyP95Meters = 0.028,
                    uncertaintyBudgetMeters = 0.08,
                    reprojectionErrorPx = 1.1,
                    reprojectionBudgetPx = 3.0,
                    supportingViewConfidence = 0.95,
                    scaleConfidence = 0.96,
                    canonicalSurfaceConfidence = 0.92,
                    canonicalSurfaceAccepted = false,
                    canonicalCorrespondenceCoverage = 0.94,
                    canonicalEdgeStretchScore = 0.91,
                    lightingConfidence = 0.90,
                    occlusionConfidence = 0.94,
                ),
            )

        assertTrue(!result.accepted)
        assertTrue(
            result.blockers.any {
                it.contains("canonical")
            },
        )
    }

    @Test
    fun weakCanonicalSurfacePriorDegradesButDoesNotInventGeometry() {
        val result =
            ReconstructionSurfaceReadinessEngine().evaluate(
                ReconstructionSurfaceReadinessEngine.Input(
                    timestampNanos = 6_000_000_000L,
                    meshQuality = quality(
                        geometric = 0.88,
                        temporal = 0.84,
                        reprojectionPx = 1.7,
                        uncertaintyMeters = 0.024,
                    ),
                    surfaceConfidence = 0.86,
                    temporalConfidence = 0.84,
                    uncertaintyP95Meters = 0.052,
                    uncertaintyBudgetMeters = 0.10,
                    reprojectionErrorPx = 1.7,
                    reprojectionBudgetPx = 3.0,
                    supportingViewConfidence = 0.86,
                    scaleConfidence = 0.92,
                    canonicalSurfaceConfidence = 0.46,
                    canonicalSurfaceAccepted = true,
                    canonicalCorrespondenceCoverage = 0.55,
                    canonicalEdgeStretchScore = 0.48,
                    lightingConfidence = 0.88,
                    occlusionConfidence = 0.82,
                ),
            )

        assertTrue(
            result.warnings.any {
                it.contains("canonical")
            },
        )
        assertTrue(
            result.score < 0.86,
        )
    }

    @Test
    fun relativeDepthCanRemainUsableWithoutMetricScaleConfidence() {
        val result = ReconstructionSurfaceReadinessEngine().evaluate(
            ReconstructionSurfaceReadinessEngine.Input(
                timestampNanos = 4_000_000_000L,
                meshQuality = quality(0.84, 0.86, 1.8, 0.025),
                surfaceConfidence = 0.88,
                temporalConfidence = 0.87,
                uncertaintyP95Meters = 0.06,
                uncertaintyBudgetMeters = 0.12,
                reprojectionErrorPx = 1.8,
                reprojectionBudgetPx = 3.0,
                supportingViewConfidence = 0.82,
                scaleConfidence = 0.40,
                metricScaleAvailable = false,
                lightingConfidence = 0.84,
                occlusionConfidence = 0.80,
            ),
        )

        assertTrue(result.band != ReconstructionSurfaceReadinessEngine.Band.BLOCKED)
        assertTrue(result.warnings.any { it.contains("metric scale") })
    }
}
