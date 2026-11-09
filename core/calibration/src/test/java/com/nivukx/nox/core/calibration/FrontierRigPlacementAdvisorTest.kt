package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierRigPlacementAdvisorTest {
    @Test
    fun selectsGeometricallyDiverseCandidates() {
        val advisor = FrontierRigPlacementAdvisor()
        val targets = listOf(
            FrontierRigPlacementAdvisor.SceneTarget(1, Vec3(0.0,0.0,1.0)),
            FrontierRigPlacementAdvisor.SceneTarget(2, Vec3(0.2,0.0,1.0)),
        )
        val candidates = listOf(
            FrontierRigPlacementAdvisor.Candidate("front", Vec3(0.0,0.0,-1.0), Vec3(0.0,0.0,1.0)),
            FrontierRigPlacementAdvisor.Candidate("right", Vec3(2.0,0.0,1.0), Vec3(-1.0,0.0,0.0)),
            FrontierRigPlacementAdvisor.Candidate("rear", Vec3(0.0,0.0,3.0), Vec3(0.0,0.0,-1.0)),
            FrontierRigPlacementAdvisor.Candidate("left", Vec3(-2.0,0.0,1.0), Vec3(1.0,0.0,0.0)),
            FrontierRigPlacementAdvisor.Candidate("duplicate", Vec3(0.0,0.0,-1.05), Vec3(0.0,0.0,1.0)),
        )
        val result = advisor.recommend(targets, candidates, requestedCount = 4)
        assertTrue(result.selected.size == 4)
        assertTrue(result.sectorCount >= 3)
    }

    @Test
    fun emptyTargetsFailClosed() {
        val result = FrontierRigPlacementAdvisor().recommend(
            targets = emptyList(),
            candidates = listOf(
                FrontierRigPlacementAdvisor.Candidate("camera", Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,1.0)),
            ),
            requestedCount = 1,
        )
        assertTrue(result.selected.isEmpty())
        assertTrue(result.warnings.isNotEmpty())
    }
