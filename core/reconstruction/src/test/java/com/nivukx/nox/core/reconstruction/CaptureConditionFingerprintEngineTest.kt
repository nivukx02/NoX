package com.nivukx.nox.core.reconstruction

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class CaptureConditionFingerprintEngineTest {

    @Test
    fun equivalent_capture_conditions_have_identical_fingerprints() {
        val engine = CaptureConditionFingerprintEngine()

        val a = engine.create(
            CaptureConditionFingerprintEngine.Input(
                rigId = "rig-alpha",
                cameraCount = 8,
                captureFps = 60,
                inferenceFps = 60,
                baselineMeters = 1.20,
                workingDistanceMeters = 2.80,
                calibrationConfidence = 0.97,
                synchronizationConfidence = 0.98,
                lightingLux = 450.0,
                colorTemperatureKelvin = 5600.0,
                roomTemperatureCelsius = 24.0,
            ),
        )

        val b = engine.create(
            CaptureConditionFingerprintEngine.Input(
                rigId = "rig-alpha",
                cameraCount = 8,
                captureFps = 60,
                inferenceFps = 60,
                baselineMeters = 1.2000001,
                workingDistanceMeters = 2.8000001,
                calibrationConfidence = 0.9700001,
                synchronizationConfidence = 0.9800001,
                lightingLux = 450.2,
                colorTemperatureKelvin = 5600.2,
                roomTemperatureCelsius = 24.02,
            ),
        )

        assertEquals(a.fingerprint, b.fingerprint)
        assertTrue(a.quantizedConfigurationHash.isNotBlank())
    }

    @Test
    fun materially_different_conditions_do_not_share_fingerprint() {
        val engine = CaptureConditionFingerprintEngine()

        val base = CaptureConditionFingerprintEngine.Input(
            rigId = "rig-alpha",
            cameraCount = 8,
            captureFps = 60,
            inferenceFps = 60,
            baselineMeters = 1.20,
            workingDistanceMeters = 2.80,
            calibrationConfidence = 0.97,
            synchronizationConfidence = 0.98,
            lightingLux = 450.0,
            colorTemperatureKelvin = 5600.0,
            roomTemperatureCelsius = 24.0,
        )

        val changed = base.copy(
            cameraCount = 4,
            captureFps = 30,
            inferenceFps = 30,
            baselineMeters = 0.55,
            roomTemperatureCelsius = 31.0,
        )

        assertTrue(
            engine.create(base).fingerprint !=
                engine.create(changed).fingerprint,
        )
    }

    @Test
    fun achieved_session_state_changes_fingerprint() {
        val engine = CaptureConditionFingerprintEngine()

        val base =
            CaptureConditionFingerprintEngine.Input(
                rigId = "rig-high-speed",
                cameraCount = 8,
                captureFps = 240,
                inferenceFps = 60,
                baselineMeters = 1.20,
                workingDistanceMeters = 2.80,
                calibrationConfidence = 0.99,
                synchronizationConfidence = 0.99,
                lightingLux = 500.0,
                colorTemperatureKelvin = 5600.0,
                roomTemperatureCelsius = 24.0,
                achievedCaptureFps = 239.6,
                highSpeedSessionActive = true,
                sessionVerification = "HARDWARE_VERIFIED",
            )

        val slower =
            base.copy(
                achievedCaptureFps = 119.8,
                highSpeedSessionActive = false,
                sessionVerification = "REJECTED",
            )

        assertTrue(
            engine.create(base).fingerprint !=
                engine.create(slower).fingerprint,
        )
    }

    @Test
    fun invalid_capture_conditions_are_rejected() {
        val engine = CaptureConditionFingerprintEngine()

        var rejected = false
        try {
            engine.create(
                CaptureConditionFingerprintEngine.Input(
                    rigId = "rig",
                    cameraCount = 0,
                    captureFps = 60,
                    inferenceFps = 60,
                    baselineMeters = 1.0,
                    workingDistanceMeters = 2.0,
                    calibrationConfidence = 0.9,
                    synchronizationConfidence = 0.9,
                    lightingLux = 400.0,
                    colorTemperatureKelvin = 5600.0,
                    roomTemperatureCelsius = 24.0,
                ),
            )
        } catch (_: IllegalArgumentException) {
            rejected = true
        }

        assertTrue(rejected)
    }
}
