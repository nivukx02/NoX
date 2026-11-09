package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrontierFaceEyeAuthorityCoordinatorTest {
    private fun input(faceTime:Long, eyeTime:Long): FrontierFaceEyeAuthorityCoordinator.Input {
        val device=DeviceId("a")
        val calibration=CalibrationSnapshot(
            mapOf(device to CameraIntrinsics(500.0,500.0,500.0,500.0,imageWidth=1000,imageHeight=1000)),
            mapOf(device to CameraExtrinsics(Pose(),null)),0.0,1L,
        )
        val frame=FaceMeshFrame(faceTime,1000,1000,0,1,100f,100f,900f,900f,0.9,0.95,0.9,listOf(FaceMeshPoint3D(0,Vec3(100.0,100.0,1.0))),emptyList())
        return FrontierFaceEyeAuthorityCoordinator.Input(100L,listOf(device to frame),calibration,Quat.IDENTITY,emptyList(),eyeTime)
    }

    @Test fun blinkedEyesCannotMakeFaceGeometryAuthoritative() {
        val result=FrontierFaceEyeAuthorityCoordinator().process(input(100L,100L).copy(
            eyeRays=listOf(WorldSpaceGazeReconstructionEngine.EyeRay(WorldSpaceGazeReconstructionEngine.Eye.LEFT,Vec3(0.0,0.0,0.0),Vec3(1.0,0.0,0.0),1.0,0.99,true)),
        ))
        assertFalse(result.authoritative)
    }

    @Test fun staleEyeEvidenceIsRejected() {
        val result=FrontierFaceEyeAuthorityCoordinator().process(input(100L,200_000_000L))
        assertFalse(result.gaze.valid)
        assertFalse(result.authoritative)
    }
    @Test
    fun futureEyeEvidenceIsRejected() {
        val result = FrontierFaceEyeAuthorityCoordinator().process(
            input(100L, 200L).copy(
                eyeRays = listOf(
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                        originMeters = Vec3(0.0,0.0,0.0),
                        directionHeadSpace = Vec3(0.0,0.0,1.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                        originMeters = Vec3(0.064,0.0,0.0),
                        directionHeadSpace = Vec3(0.0,0.0,1.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                ),
            ),
        )
        assertFalse(result.gaze.valid)
        assertFalse(result.authoritative)
    }

    @Test
    fun monocularEvidenceCannotBecomePrecisionAuthoritative() {
        val result = FrontierFaceEyeAuthorityCoordinator().process(
            input(100L,100L).copy(
                eyeRays = listOf(
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                        originMeters = Vec3(0.0,0.0,0.0),
                        directionHeadSpace = Vec3(0.0,0.0,1.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                ),
            ),
        )
        assertFalse(result.authoritative)
        assertTrue(result.warnings.any { it == "binocular-eye-support-insufficient" })
    }

    @Test
    fun excessiveBinocularDisagreementBlocksAuthority() {
        val result = FrontierFaceEyeAuthorityCoordinator().process(
            input(100L,100L).copy(
                eyeRays = listOf(
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                        originMeters = Vec3(0.0,0.0,0.0),
                        directionHeadSpace = Vec3(0.0,0.0,1.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                        originMeters = Vec3(0.064,0.0,0.0),
                        directionHeadSpace = Vec3(1.0,0.0,0.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                ),
            ),
        )
        assertFalse(result.authoritative)
        assertTrue(result.warnings.any { it == "binocular-gaze-disagreement-high" })
    )
    @Test
    fun faceEyeTemporalSkewCannotBecomeAuthoritative() {
        val result = FrontierFaceEyeAuthorityCoordinator().process(
            input(100L, 100L).copy(
                faceFrames = input(100L, 100L).faceFrames.map { (device, frame) ->
                    device to frame.copy(timestampNanos = 100L)
                },
                eyeRays = listOf(
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.LEFT,
                        originMeters = Vec3(0.0,0.0,0.0),
                        directionHeadSpace = Vec3(0.0,0.0,1.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                    WorldSpaceGazeReconstructionEngine.EyeRay(
                        eye = WorldSpaceGazeReconstructionEngine.Eye.RIGHT,
                        originMeters = Vec3(0.064,0.0,0.0),
                        directionHeadSpace = Vec3(0.0,0.0,1.0),
                        eyelidOpenness = 1.0,
                        confidence = 0.99,
                    ),
                ),
                eyeTimestampNanos = 40_000_100L,
            ),
        )
        assertTrue(result.gaze.valid)
        assertTrue(result.warnings.any { it == "face-eye-temporal-skew-high" })
        assertTrue(!result.authoritative)
    }
}