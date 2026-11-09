package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceCapabilities
import com.nivukx.nox.core.model.DeviceDescriptor
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiCameraRigHealthEngineTest {

    private fun device(
        id: String,
        connected: Boolean = true,
        packetLoss: Double = 0.01,
        clockOffsetNanos: Long = 400_000L,
        roundTripNanos: Long = 12_000_000L,
    ) = DeviceDescriptor(
        id = DeviceId(id),
        role = DeviceRole.AUXILIARY,
        displayName = id,
        capabilities = DeviceCapabilities(
            model = "test",
            manufacturer = "test",
            androidApi = 37,
            cameraIds = listOf("0"),
            supportsRaw = true,
            supportsConcurrentCamera = true,
            supportsGyroscope = true,
            supportsAccelerometer = true,
            maxAnalysisWidth = 1920,
            maxAnalysisHeight = 1080,
            maxFps = 60,
        ),
        connected = connected,
        clockOffsetNanos = clockOffsetNanos,
        roundTripNanos = roundTripNanos,
        packetLoss = packetLoss,
    )

    @Test
    fun stableRig_marksAllConnectedCamerasHealthy() {
        val engine = MultiCameraRigHealthEngine()

        val result = engine.evaluate(
            MultiCameraRigHealthEngine.Input(
                devices = listOf(device("a"), device("b")),
                calibrationReady = true,
                calibrationConfidence = 0.96,
                synchronizationConfidence = 0.95,
                captureTimingConfidence = 0.96,
                cameraStatuses = listOf(
                    healthyStatus("a"),
                    healthyStatus("b"),
                ),
            ),
        )

        assertEquals(
            MultiCameraRigHealthEngine.State.HEALTHY,
            result.state,
        )
        assertTrue(result.overallScore > 0.85)
        assertTrue(result.cameras.all {
            it.state == MultiCameraRigHealthEngine.CameraState.HEALTHY
        })
    }

    @Test
    fun packetLossAndClockDrift_lowerCameraHealthAndTriggerWatch() {
        val engine = MultiCameraRigHealthEngine()

        val result = engine.evaluate(
            MultiCameraRigHealthEngine.Input(
                devices = listOf(
                    device(
                        "a",
                        packetLoss = 0.02,
                        clockOffsetNanos = 500_000L,
                    ),
                    device(
                        "b",
                        packetLoss = 0.22,
                        clockOffsetNanos = 9_000_000L,
                        roundTripNanos = 180_000_000L,
                    ),
                ),
                calibrationReady = true,
                calibrationConfidence = 0.90,
                synchronizationConfidence = 0.72,
                captureTimingConfidence = 0.88,
                cameraStatuses = listOf(
                    healthyStatus("a"),
                    degradedStatus("b"),
                ),
            ),
        )

        val degraded = result.cameras.first { it.deviceId == DeviceId("b") }
        assertTrue(
            degraded.state ==
                MultiCameraRigHealthEngine.CameraState.WATCH ||
                degraded.state ==
                MultiCameraRigHealthEngine.CameraState.HOLD,
        )
        assertTrue(result.warnings.any {
            it.contains("b")
        })
    }

    @Test
    fun disconnectedCamera_isOfflineAndExcludedFromHealthyAverage() {
        val engine = MultiCameraRigHealthEngine()

        val result = engine.evaluate(
            MultiCameraRigHealthEngine.Input(
                devices = listOf(
                    device("a"),
                    device("b", connected = false),
                ),
                calibrationReady = true,
                calibrationConfidence = 0.94,
                synchronizationConfidence = 0.94,
                captureTimingConfidence = 0.94,
                cameraStatuses = listOf(healthyStatus("a")),
            ),
        )

        assertEquals(
            MultiCameraRigHealthEngine.CameraState.OFFLINE,
            result.cameras.first {
                it.deviceId == DeviceId("b")
            }.state,
        )
        assertTrue(result.onlineCameraCount == 1)
    }

    @Test
    fun criticalWeakestCameraCannotBeHiddenByHealthyMajority() {
        val engine = MultiCameraRigHealthEngine()

        val result = engine.evaluate(
            MultiCameraRigHealthEngine.Input(
                devices = listOf(
                    device("a"),
                    device("b"),
                    device("c"),
                    device(
                        "d",
                        packetLoss = 0.30,
                        clockOffsetNanos = 21_000_000L,
                        roundTripNanos = 480_000_000L,
                    ),
                ),
                calibrationReady = true,
                calibrationConfidence = 0.96,
                synchronizationConfidence = 0.95,
                captureTimingConfidence = 0.96,
                cameraStatuses = listOf(
                    healthyStatus("a"),
                    healthyStatus("b"),
                    healthyStatus("c"),
                    degradedStatus("d"),
                ),
            ),
        )

        assertTrue(
            result.state ==
                MultiCameraRigHealthEngine.State.HOLD ||
                result.state ==
                MultiCameraRigHealthEngine.State.BLOCKED,
        )
        assertTrue(result.overallScore < 0.82)
        assertTrue(result.warnings.any {
            it.contains("d")
        })
    }

    private fun healthyStatus(id: String) =
        AdaptiveMultiViewEvidenceWeightEngine.CameraStatus(
            deviceId = DeviceId(id),
            trust = 0.94,
            observationCount = 24,
            degradedJointCount = 1,
            hardFailureJointCount = 0,
            meanResidualPx = 1.8,
            state = AdaptiveMultiViewEvidenceWeightEngine.State.HEALTHY,
        )

    private fun degradedStatus(id: String) =
        AdaptiveMultiViewEvidenceWeightEngine.CameraStatus(
            deviceId = DeviceId(id),
            trust = 0.31,
            observationCount = 24,
            degradedJointCount = 14,
            hardFailureJointCount = 7,
            meanResidualPx = 9.4,
            state = AdaptiveMultiViewEvidenceWeightEngine.State.DEGRADED,
        )
}
