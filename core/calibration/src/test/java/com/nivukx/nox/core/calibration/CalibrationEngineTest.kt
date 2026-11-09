package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationEngineTest {

    @Test
    fun referenceCalibrationBuildsRelativeCameraExtrinsics() {
        val reference = DeviceId("camera-a")
        val secondary = DeviceId("camera-b")
        val intrinsics = CameraIntrinsics(
            fx = 1200.0,
            fy = 1200.0,
            cx = 960.0,
            cy = 540.0,
            imageWidth = 1920,
            imageHeight = 1080,
        )

        val snapshot =
            CalibrationEngine().fromReference(
                reference = reference,
                intrinsics = mapOf(
                    reference to intrinsics,
                    secondary to intrinsics,
                ),
                cameraCenters = mapOf(
                    reference to Vec3(10.0, 5.0, 2.0),
                    secondary to Vec3(11.5, 5.2, 2.4),
                ),
                forwardVectors = mapOf(
                    reference to Vec3(0.0, 0.0, 1.0),
                    secondary to Vec3(0.0, 0.0, 1.0),
                ),
                nowNanos = 123L,
            )

        val referencePose =
            snapshot.extrinsics.getValue(reference).pose
        val secondaryPose =
            snapshot.extrinsics.getValue(secondary).pose

        assertEquals(0.0, referencePose.position.x, 1e-9)
        assertEquals(0.0, referencePose.position.y, 1e-9)
        assertEquals(0.0, referencePose.position.z, 1e-9)

        assertEquals(1.5, secondaryPose.position.x, 1e-9)
        assertEquals(0.2, secondaryPose.position.y, 1e-9)
        assertEquals(0.4, secondaryPose.position.z, 1e-9)

        assertTrue(snapshot.extrinsics.keys.containsAll(listOf(reference, secondary)))
        assertEquals(123L, snapshot.calibratedAtNanos)
    }
}
