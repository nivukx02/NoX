package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.FaceMeshPoint3D
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertFalse
import org.junit.Test

class FrontierFaceSurfaceAuthorityEngineTest {
    @Test
    fun insufficientPhysicalEvidenceCannotBecomeAuthoritative() {
        val engine=FrontierFaceSurfaceAuthorityEngine()
        val frame=FaceMeshFrame(100L,1000,1000,0,1,100f,100f,900f,900f,0.9,0.95,0.9,listOf(FaceMeshPoint3D(0,Vec3(100.0,100.0,1.0))),emptyList())
        engine.push(DeviceId("a"),frame)
        val result=engine.reconstruct(100L,CalibrationSnapshot(
            intrinsics=mapOf(DeviceId("a") to com.nivukx.nox.core.model.CameraIntrinsics(500.0,500.0,500.0,500.0,imageWidth=1000,imageHeight=1000)),
            extrinsics=mapOf(DeviceId("a") to com.nivukx.nox.core.model.CameraExtrinsics(com.nivukx.nox.core.model.Pose(),null)),
            reprojectionErrorPx=0.0,calibratedAtNanos=1L,
        ))
        assertFalse(result.authoritative)
        engine.close()
    }
}