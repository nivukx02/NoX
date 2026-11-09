package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SparseViewHumanReconstructionEngineTest {
    @Test
    fun sparseGeometryKeepsObservedRegionAuthoritative() {
        val result = SparseViewHumanReconstructionEngine().reconstruct(100L, listOf(
            SparseViewHumanReconstructionEngine.RegionEvidence(1,2,100L,Vec3(1.0,2.0,3.0),3,0.7,0.8,0.95),
        ))[0]
        assertEquals(SparseViewHumanReconstructionEngine.State.OBSERVED, result.state)
        assertTrue(result.measuredRatio > result.inferredRatio)
    }

    @Test
    fun oneViewRegionFailsClosed() {
        val result = SparseViewHumanReconstructionEngine().reconstruct(100L, listOf(
            SparseViewHumanReconstructionEngine.RegionEvidence(1,2,100L,Vec3(1.0,2.0,3.0),1,0.0,3.0,0.9),
        ))[0]
        assertEquals(SparseViewHumanReconstructionEngine.State.INSUFFICIENT_EVIDENCE, result.state)
        assertTrue(result.warnings.isNotEmpty())
    }
}

class SparseViewHumanReconstructionEngineFrontierTest {
    @Test
    fun robustViewEvidenceSuppressesAHighErrorOutlier() {
        val engine = SparseViewHumanReconstructionEngine()
        val result = engine.reconstructDetailed(
            timestampNanos = 500L,
            regions = listOf(
                SparseViewHumanReconstructionEngine.RegionEvidence(
                    actorId = 2,
                    regionId = 7,
                    timestampNanos = 500L,
                    worldPositionMeters = Vec3(0.10, 0.0, 0.0),
                    independentViews = 3,
                    viewDiversity = 0.85,
                    reprojectionResidualPixels = 1.0,
                    observedRatio = 0.90,
                    viewEvidence = listOf(
                        SparseViewHumanReconstructionEngine.ViewEvidence(
                            cameraId = "a",
                            positionMeters = Vec3(0.10, 0.0, 0.0),
                            confidence = 1.0,
                        ),
                        SparseViewHumanReconstructionEngine.ViewEvidence(
                            cameraId = "b",
                            positionMeters = Vec3(0.11, 0.0, 0.0),
                            confidence = 1.0,
                        ),
                        SparseViewHumanReconstructionEngine.ViewEvidence(
                            cameraId = "c",
                            positionMeters = Vec3(0.80, 0.0, 0.0),
                            confidence = 0.05,
                        ),
                    ),
                ),
            ),
        )

        val reconstructed = result.regions.single()
        assertTrue(reconstructed.positionMeters.x < 0.20)
        assertTrue(reconstructed.independentViews == 3)
        assertTrue(reconstructed.observedViewCount == 3)
    }

    @Test
    fun learnedPriorCannotOverrideInsufficientGeometry() {
        val engine = SparseViewHumanReconstructionEngine(
            SparseViewHumanReconstructionEngine.Config(
                maximumUncertaintyMeters = 0.10,
            ),
        )
        val result = engine.reconstruct(
            timestampNanos = 900L,
            regions = listOf(
                SparseViewHumanReconstructionEngine.RegionEvidence(
                    actorId = 4,
                    regionId = 9,
                    timestampNanos = 900L,
                    worldPositionMeters = Vec3(1.0, 0.0, 0.0),
                    independentViews = 1,
                    viewDiversity = 0.0,
                    reprojectionResidualPixels = 1.0,
                    observedRatio = 0.1,
                    learnedPrior = Vec3(0.0, 0.0, 0.0),
                ),
            ),
        ).single()

        assertTrue(result.state != SparseViewHumanReconstructionEngine.State.OBSERVED)
        assertTrue(result.warnings.contains("insufficient-independent-views"))
    }
}
