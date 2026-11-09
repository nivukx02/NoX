package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WorldSpaceMotionProposalEngineTest {
    private fun obs(camera: String, origin: Vec3, direction: Vec3) =
        WorldSpaceMotionProposalEngine.RayObservation(
            actorId = 7, pointId = 11, cameraId = camera, timestampNanos = 1_000L,
            rayOriginMeters = origin, rayDirection = direction, pixelConfidence = 0.99,
            timingConfidence = 0.98, geometricUncertaintyMeters = 0.002,
            calibrationFingerprint = "cal-$camera", timingFingerprint = "tim-$camera", modelFingerprint = "model-v1",
        )

    @Test
    fun twoObliqueRaysRecoverTheirIntersection() {
        val engine = WorldSpaceMotionProposalEngine()
        val result = engine.propose(
            1_000L,
            listOf(
                obs("a", Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 1.0)),
                obs("b", Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 1.0)),
            ),
        )
        assertNotNull(result.proposal)
        assertEquals(0.5, result.proposal!!.positionMeters.x, 0.001)
        assertTrue(result.rejectedObservationCount == 0)
    }

    @Test
    fun duplicateCameraObservationCannotPretendToBeIndependentViews() {
        val engine = WorldSpaceMotionProposalEngine()
        val base = obs("same", Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,1.0))
        val result = engine.propose(1_000L, listOf(base, base.copy(rayOriginMeters = Vec3(1.0,0.0,0.0))))
        assertTrue(result.proposal == null)
        assertTrue(result.warnings.any { it.contains("insufficient") })
    }

    @Test
    fun learnedProposalIsBoundedInsteadOfOverridingGeometry() {
        val engine = WorldSpaceMotionProposalEngine(
            WorldSpaceMotionProposalEngine.Config(maximumLearnedDisplacementMeters = 0.10),
        )
        val result = engine.propose(
            1_000L,
            listOf(
                obs("a", Vec3(0.0,0.0,0.0), Vec3(1.0,0.0,1.0)),
                obs("b", Vec3(1.0,0.0,0.0), Vec3(-1.0,0.0,1.0)),
            ),
            WorldSpaceMotionProposalEngine.LearnedProposal(Vec3(10.0,0.0,10.0),1.0,0.001),
        )
        assertNotNull(result.proposal)
        assertTrue(result.proposal!!.positionMeters.distanceTo(Vec3(0.5,0.0,0.5)) <= 0.10 + 1e-6)
        assertTrue(result.warnings.any { it.contains("learned-proposal") })
    }

    @Test
    fun nonMatchingTimestampEvidenceIsRejected() {
        val engine = WorldSpaceMotionProposalEngine()
        val stale = obs("a", Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,1.0)).copy(timestampNanos = 999L)
        val current = obs("b", Vec3(1.0,0.0,0.0), Vec3(0.0,0.0,1.0))
        val result = engine.propose(1_000L, listOf(stale,current))
        assertTrue(result.proposal == null)
    }
}
