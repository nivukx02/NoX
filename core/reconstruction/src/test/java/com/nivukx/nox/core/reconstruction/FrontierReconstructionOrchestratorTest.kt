package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.MeshTriangle
import com.nivukx.nox.core.model.MeshVertex
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class FrontierReconstructionOrchestratorTest {
    @Test
    fun closeIsIdempotentAndRejectsFurtherFrames() {
        val orchestrator = FrontierReconstructionOrchestrator()

        orchestrator.close()
        orchestrator.close()

        assertThrows(IllegalStateException::class.java) {
            orchestrator.process(
                FrontierReconstructionOrchestrator.Input(
                    timestampNanos = 1L,
                    regions = emptyList(),
                    surfacePoints = emptyList(),
                    globalCameras = emptyList(),
                    globalPoints = emptyList(),
                ),
            )
        }
    }

    @Test fun reconstructsSparseRegionIntoBoundedGaussianAndGlobalState() {
        val input=FrontierReconstructionOrchestrator.Input(
            timestampNanos=10L,
            regions=listOf(SparseViewHumanReconstructionEngine.RegionEvidence(1,2,10L,Vec3(0.0,0.0,0.0),3,0.7,0.8,0.9)),
            surfacePoints=listOf(DynamicGaussianSurfaceRefinementEngine.SurfacePoint(1L,Vec3(0.0,0.0,0.0),Vec3(0.0,0.0,1.0),UncertaintyEnvelope(1e-4,1e-4,1e-4),0.9,EvidenceOrigin.OBSERVED)),
            globalCameras=listOf(GlobalHumanScene4DOptimizationEngine.CameraState("a",Vec3(0.0,0.0,0.0))),
            globalPoints=listOf(GlobalHumanScene4DOptimizationEngine.ActorPoint(1,2,Vec3(0.0,0.0,0.0),listOf(Vec3(0.1,0.0,0.0)),0.9)),
            surfaceMesh=ReconstructionMesh(
                vertices=listOf(MeshVertex(Vec3(0.0,0.0,0.0),0.001),MeshVertex(Vec3(1.0,0.0,0.0),0.001),MeshVertex(Vec3(0.0,1.0,0.0),0.001)),
                triangles=listOf(MeshTriangle(0,1,2)),
                timestampNanos=10L,
                confidence=0.95,
            ),
        )
        val result=FrontierReconstructionOrchestrator().process(input)
        assertTrue(result.sparseRegions.isNotEmpty())
        assertTrue(result.gaussians.size >= 3)
        assertTrue(result.gaussians.any { it.id >= (1L shl 60) })
        assertTrue(result.authoritativeRegionCount == 1)
        assertTrue(result.provenance.fingerprint.length == 64)
        assertTrue(result.humanAwareBundle == null)
    }
}

class FrontierReconstructionOrchestratorConstraintHandoffTest {
    @Test fun sparseObservedRegionFeedsMatchingGlobalPointConstraint() {
        val result = FrontierReconstructionOrchestrator().process(
            FrontierReconstructionOrchestrator.Input(
                timestampNanos = 20L,
                regions = listOf(
                    SparseViewHumanReconstructionEngine.RegionEvidence(
                        actorId = 3,
                        regionId = 5,
                        timestampNanos = 20L,
                        worldPositionMeters = Vec3(0.18, 0.0, 0.0),
                        independentViews = 4,
                        viewDiversity = 0.90,
                        reprojectionResidualPixels = 0.8,
                        observedRatio = 0.95,
                        uncertaintyMeters = 0.002,
                        surfaceConsistency = 0.92,
                    ),
                ),
                surfacePoints = emptyList(),
                globalCameras = listOf(
                    GlobalHumanScene4DOptimizationEngine.CameraState(
                        cameraId = "a",
                        positionMeters = Vec3(0.0, 0.0, 0.0),
                        fixed = true,
                    ),
                ),
                globalPoints = listOf(
                    GlobalHumanScene4DOptimizationEngine.ActorPoint(
                        actorId = 3,
                        pointId = 5,
                        positionMeters = Vec3(0.0, 0.0, 0.0),
                        cameraObservations = listOf(Vec3(0.20, 0.0, 0.0)),
                        confidence = 0.95,
                    ),
                ),
            ),
        )

        val reconstructed = result.globalRefinement.points.single()
        assertTrue(reconstructed.positionMeters.x > 0.02)
        assertTrue(result.globalRefinement.pointDiagnostics.single().residualAfterMeters.isFinite())
    }
}
