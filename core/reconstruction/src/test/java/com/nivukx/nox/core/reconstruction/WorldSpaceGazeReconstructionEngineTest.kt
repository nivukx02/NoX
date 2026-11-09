package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WorldSpaceGazeReconstructionEngineTest {
    @Test
    fun binocularRaysProduceStableWorldDirection() {
        val result = WorldSpaceGazeReconstructionEngine().reconstruct(Quat.IDENTITY, listOf(
            WorldSpaceGazeReconstructionEngine.EyeRay(WorldSpaceGazeReconstructionEngine.Eye.LEFT,Vec3(0.0,0.0,0.0),Vec3(0.01,0.0,1.0),1.0,0.95),
            WorldSpaceGazeReconstructionEngine.EyeRay(WorldSpaceGazeReconstructionEngine.Eye.RIGHT,Vec3(0.06,0.0,0.0),Vec3(-0.01,0.0,1.0),1.0,0.94),
        ))
        assertTrue(result.valid)
        assertEquals(0.0,result.directionWorld.x,0.03)
        assertTrue(result.inlierRatio >= 0.5)
    }

    @Test
    fun blinkEvidenceIsNotUsedAsGazeObservation() {
        val result = WorldSpaceGazeReconstructionEngine().reconstruct(Quat.IDENTITY, listOf(
            WorldSpaceGazeReconstructionEngine.EyeRay(WorldSpaceGazeReconstructionEngine.Eye.LEFT,Vec3(0.0,0.0,0.0),Vec3(1.0,0.0,0.0),1.0,0.99,true),
        ))
        assertTrue(!result.valid)
        assertEquals("no-valid-eye-evidence", result.warnings.first())
    }
}

class WorldSpaceGazeReconstructionEngineFrontierTest {
    @Test
    fun highAngleOutlierIsSuppressedBySphericalConsensus() {
        val result = WorldSpaceGazeReconstructionEngine().reconstruct(
            Quat.IDENTITY,
            listOf(
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                    originMeters = Vec3(0.0, 0.0, 0.0),
                    directionHeadSpace = Vec3(0.01, 0.0, 1.0),
                    eyelidOpenness = 1.0,
                    confidence = 0.98,
                ),
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                    originMeters = Vec3(0.06, 0.0, 0.0),
                    directionHeadSpace = Vec3(-0.01, 0.0, 1.0),
                    eyelidOpenness = 1.0,
                    confidence = 0.97,
                ),
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                    originMeters = Vec3(0.06, 0.0, 0.0),
                    directionHeadSpace = Vec3(0.95, 0.0, 0.20),
                    eyelidOpenness = 1.0,
                    confidence = 0.50,
                    measurementUncertaintyDegrees = 18.0,
                ),
            ),
        )

        assertTrue(result.valid)
        assertTrue(result.outlierCount >= 1)
        assertTrue(result.confidence > 0.50)
    }

    @Test
    fun binocularDisagreementIsExposedInsteadOfHidden() {
        val result = WorldSpaceGazeReconstructionEngine().reconstruct(
            Quat.IDENTITY,
            listOf(
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                    originMeters = Vec3(0.0, 0.0, 0.0),
                    directionHeadSpace = Vec3(0.0, 0.0, 1.0),
                    eyelidOpenness = 1.0,
                    confidence = 0.95,
                ),
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                    originMeters = Vec3(0.06, 0.0, 0.0),
                    directionHeadSpace = Vec3(0.5, 0.0, 0.8660254),
                    eyelidOpenness = 1.0,
                    confidence = 0.95,
                ),
            ),
        )

        assertTrue(result.binocularAgreementDegrees > 20.0)
        assertTrue(result.warnings.contains("gaze-binocular-disagreement"))
    }
}


class WorldSpaceGazeReconstructionEngineTargetTest {
    @Test
    fun convergingBinocularRaysProduceMetricGazeTarget() {
        val result = WorldSpaceGazeReconstructionEngine().reconstruct(
            Quat.IDENTITY,
            listOf(
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                    originMeters = Vec3(-0.032, 0.0, 0.0),
                    directionHeadSpace = Vec3(0.032, 0.0, 1.0),
                    eyelidOpenness = 1.0,
                    confidence = 0.99,
                ),
                WorldSpaceGazeReconstructionEngine.EyeRay(
                    eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                    originMeters = Vec3(0.032, 0.0, 0.0),
                    directionHeadSpace = Vec3(-0.032, 0.0, 1.0),
                    eyelidOpenness = 1.0,
                    confidence = 0.99,
                ),
            ),
        )

        assertTrue(result.valid)
        assertTrue(result.gazeTargetWorld != null)
        assertTrue(result.gazeTargetWorld!!.z > 0.5)
        assertTrue(result.gazeTargetUncertaintyMeters.isFinite())
        assertTrue(result.vergenceAngleDegrees > 0.0)
    }
}
