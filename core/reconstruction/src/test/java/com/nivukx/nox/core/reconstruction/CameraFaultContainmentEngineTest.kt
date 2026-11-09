package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class CameraFaultContainmentEngineTest {
    private val camera = DeviceId("camera-1")

    @Test
    fun persistentFailureQuarantinesCamera() {
        val engine = CameraFaultContainmentEngine(
            CameraFaultContainmentEngine.Config(enterFailureFrames = 3),
        )
        repeat(2) {
            val result = engine.observe(
                CameraFaultContainmentEngine.Observation(
                    cameraId = camera,
                    score = 0.20,
                ),
            )
            assertTrue(result.state != CameraFaultContainmentEngine.State.QUARANTINED)
        }
        val result = engine.observe(
            CameraFaultContainmentEngine.Observation(
                cameraId = camera,
                score = 0.20,
            ),
        )
        assertEquals(
            CameraFaultContainmentEngine.State.QUARANTINED,
            result.state,
        )
        assertFalse(result.admitted)
    }

    @Test
    fun oneBadFrameDoesNotQuarantineCamera() {
        val engine = CameraFaultContainmentEngine()
        val first = engine.observe(
            CameraFaultContainmentEngine.Observation(
                cameraId = camera,
                score = 0.20,
            ),
        )
        val second = engine.observe(
            CameraFaultContainmentEngine.Observation(
                cameraId = camera,
                score = 0.90,
            ),
        )
        assertTrue(first.state != CameraFaultContainmentEngine.State.QUARANTINED)
        assertEquals(
            CameraFaultContainmentEngine.State.ACTIVE,
            second.state,
        )
    }

    @Test
    fun quarantineRequiresSustainedRecovery() {
        val engine = CameraFaultContainmentEngine(
            CameraFaultContainmentEngine.Config(
                enterFailureFrames = 1,
                exitRecoveryFrames = 3,
            ),
        )
        engine.observe(
            CameraFaultContainmentEngine.Observation(
                cameraId = camera,
                score = 0.10,
            ),
        )
        repeat(2) {
            val result = engine.observe(
                CameraFaultContainmentEngine.Observation(
                    cameraId = camera,
                    score = 0.95,
                ),
            )
            assertEquals(
                CameraFaultContainmentEngine.State.QUARANTINED,
                result.state,
            )
        }
        val recovered = engine.observe(
            CameraFaultContainmentEngine.Observation(
                cameraId = camera,
                score = 0.95,
            ),
        )
        assertEquals(
            CameraFaultContainmentEngine.State.ACTIVE,
            recovered.state,
        )
        assertTrue(recovered.admitted)
    }

    @Test
    fun disconnectedCameraIsQuarantined() {
        val engine = CameraFaultContainmentEngine()
        val result = engine.observe(
            CameraFaultContainmentEngine.Observation(
                cameraId = camera,
                score = 1.0,
                connected = false,
            ),
        )
        assertEquals(
            CameraFaultContainmentEngine.State.QUARANTINED,
            result.state,
        )
        assertTrue(engine.quarantinedCameraIds().contains(camera))
    }
}