package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.ai.HierarchicalMotionInferenceEngine
import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.WorldSpaceProposal
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierMotionReconstructionAdapterTest {
    private fun proposal(actor: Int, point: Int, x: Double) = WorldSpaceProposal(
        actorId = actor, pointId = point, timestampNanos = 1_000L,
        positionMeters = Vec3(x, 0.0, 1.0),
        uncertainty = UncertaintyEnvelope(1e-4,1e-4,1e-4),
        confidence = 0.9, geometricSupport = 0.9, temporalSupport = 0.9, learnedSupport = 0.0,
        independentCameraCount = 3, viewDiversity = 0.8, origin = EvidenceOrigin.OBSERVED, provenance = emptyList(),
    )

    @Test
    fun preservesMultiActorSkeletonSeparation() {
        val engine=HierarchicalMotionInferenceEngine()
        val frame=engine.process(1_000L,listOf(proposal(1,2,1.0),proposal(2,2,2.0)))
        val result=FrontierMotionReconstructionAdapter.adapt(frame)
        assertEquals(2,result.skeletons.size)
        assertEquals(setOf(1L,2L),result.skeletons.map { it.trackId }.toSet())
        assertTrue(result.skeletons.all { it.joints.size == 1 })
        engine.close()
    }

    @Test
    fun exposesOriginAndPrimarySkeletonCompatibility() {
        val engine=HierarchicalMotionInferenceEngine()
        val frame=engine.process(1_000L,listOf(proposal(1,2,1.0)))
        val result=FrontierMotionReconstructionAdapter.adapt(frame)
        assertEquals(1,result.skeleton.joints.size)
        assertEquals(1,result.observedJointCount)
        assertEquals(EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,result.originsByJointKey.values.single())
        engine.close()
    }
}