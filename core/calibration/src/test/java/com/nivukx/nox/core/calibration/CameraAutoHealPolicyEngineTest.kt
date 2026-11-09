package com.nivukx.nox.core.calibration

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class CameraAutoHealPolicyEngineTest {

    private fun input(
        cameraId: String = "cam-1",
        calibrationConfidence: Double = 0.95,
        imageErrorPx: Double = 0.8,
        connectivity: Double = 0.95,
        temperatureDeltaCelsius: Double = 0.5,
        driftScore: Double = 0.95,
        trackedCoverage: Double = 0.90,
        calibrationAgeSeconds: Double = 30.0,
    ) = CameraAutoHealPolicyEngine.Input(
        cameraId = cameraId,
        calibrationConfidence = calibrationConfidence,
        imageErrorPx = imageErrorPx,
        centroidConnectivity = connectivity,
        temperatureDeltaCelsius = temperatureDeltaCelsius,
        calibrationDriftStability = driftScore,
        trackedObjectCoverage = trackedCoverage,
        calibrationAgeSeconds = calibrationAgeSeconds,
    )

    @Test
    fun healthy_camera_is_kept() {
        val result = CameraAutoHealPolicyEngine().evaluate(
            input(),
        )

        assertEquals(
            CameraAutoHealPolicyEngine.Action.KEEP,
            result.action,
        )
        assertTrue(result.confidence > 0.80)
    }

    @Test
    fun high_image_error_with_good_coverage_becomes_heal_candidate() {
        val result = CameraAutoHealPolicyEngine().evaluate(
            input(
                imageErrorPx = 4.5,
                connectivity = 0.80,
                temperatureDeltaCelsius = 2.5,
                driftScore = 0.55,
            ),
        )

        assertEquals(
            CameraAutoHealPolicyEngine.Action.HEAL_CANDIDATE,
            result.action,
        )
        assertTrue(result.reasons.any { it.contains("image-error") })
    }

    @Test
    fun low_tracking_coverage_blocks_auto_heal_and_requests_recalibration() {
        val result = CameraAutoHealPolicyEngine().evaluate(
            input(
                imageErrorPx = 5.0,
                connectivity = 0.30,
                trackedCoverage = 0.35,
                driftScore = 0.40,
            ),
        )

        assertEquals(
            CameraAutoHealPolicyEngine.Action.RECALIBRATION_REQUIRED,
            result.action,
        )
        assertTrue(result.reasons.any { it.contains("coverage") })
    }

    @Test
    fun stale_calibration_and_thermal_drift_can_force_hold() {
        val result = CameraAutoHealPolicyEngine().evaluate(
            input(
                calibrationConfidence = 0.50,
                imageErrorPx = 7.0,
                connectivity = 0.40,
                temperatureDeltaCelsius = 8.0,
                driftScore = 0.20,
                trackedCoverage = 0.20,
                calibrationAgeSeconds = 86_400.0,
            ),
        )

        assertTrue(
            result.action ==
                CameraAutoHealPolicyEngine.Action.HOLD ||
                result.action ==
                CameraAutoHealPolicyEngine.Action.RECALIBRATION_REQUIRED,
        )
        assertTrue(result.reasons.any { it.contains("temperature") })
    }

    @Test
    fun invalid_measurements_are_rejected() {
        var rejected = false
        try {
            CameraAutoHealPolicyEngine.Input(
                cameraId = "cam-1",
                calibrationConfidence = 2.0,
                imageErrorPx = 1.0,
                centroidConnectivity = 1.0,
                temperatureDeltaCelsius = 0.0,
                calibrationDriftStability = 1.0,
                trackedObjectCoverage = 1.0,
                calibrationAgeSeconds = 0.0,
            )
        } catch (_: IllegalArgumentException) {
            rejected = true
        }
        assertTrue(rejected)
    }

    @Test
    fun degraded_candidate_never_gets_silent_healing_when_thresholds_are_breached() {
        val result = CameraAutoHealPolicyEngine().evaluate(
            input(
                imageErrorPx = 3.2,
                connectivity = 0.65,
                temperatureDeltaCelsius = 3.8,
                driftScore = 0.60,
                trackedCoverage = 0.75,
                calibrationAgeSeconds = 2_000.0,
            ),
        )

        assertEquals(
            CameraAutoHealPolicyEngine.Action.HEAL_CANDIDATE,
            result.action,
        )
        assertTrue(!result.allowSilentHealing)
    }

}
