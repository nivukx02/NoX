package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationWireCodecTest {
    @Test
    fun roundTripsCalibrationFrame() {
        val frame = CalibrationWireCodec.Frame(
            deviceId = "nox-test",
            role = DeviceRole.RIGHT,
            intrinsics = CameraIntrinsics(
                fx = 1200.0,
                fy = 1198.0,
                cx = 960.0,
                cy = 540.0,
                k1 = 0.01,
                k2 = -0.002,
                p1 = 0.0003,
                p2 = -0.0002,
                k3 = 0.0001,
                imageWidth = 1920,
                imageHeight = 1080,
            ),
            pose = Pose(
                position = Vec3(1.2, 0.3, -0.7),
                rotation = Quat(0.98, 0.02, -0.03, 0.04).normalized(),
            ),
            accuracy = 3,
        )

        val decoded = CalibrationWireCodec.decode(
            CalibrationWireCodec.encode(frame),
        )

        assertEquals(frame.deviceId, decoded.deviceId)
        assertEquals(frame.role, decoded.role)
        assertEquals(frame.intrinsics.fx, decoded.intrinsics.fx, 1e-9)
        assertEquals(frame.pose.position.z, decoded.pose.position.z, 1e-9)
        assertEquals(frame.accuracy, decoded.accuracy)
        assertTrue(decoded.pose.rotation.normalized().w.isFinite())
    }
}
