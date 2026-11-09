package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertTrue

class CaptureRigTimingEvidenceTest {

    @Test
    fun hardwareReadoutEvidenceImprovesCameraHealth() {
        val engine = CaptureRigObservabilityEngine()

        val software =
            camera(
                id = "software",
                hardwareTimingConfidence = 0.0,
                hardwareP95JitterMillis = 8.0,
                logicalCalibratedSync = false,
            )
        val hardware =
            camera(
                id = "hardware",
                hardwareTimingConfidence = 0.98,
                hardwareP95JitterMillis = 0.6,
                logicalCalibratedSync = true,
            )

        assertTrue(hardware.health > software.health)
    }

    @Test
    fun poorHardwareTimingProducesExplicitSelectionWarning() {
        val engine = CaptureRigObservabilityEngine()

        val poor =
            engine.analyze(
                listOf(
                    camera(
                        id = "cam-a",
                        hardwareTimingConfidence = 0.30,
                        hardwareP95JitterMillis = 8.0,
                        logicalCalibratedSync = false,
                    ),
                ),
            )

        assertTrue(
            poor.warnings.contains(
                "hardware-timing-authority-low",
            ),
        )
    }

    private fun camera(
        id: String,
        hardwareTimingConfidence: Double,
        hardwareP95JitterMillis: Double,
        logicalCalibratedSync: Boolean,
    ) =
        CaptureRigObservabilityEngine.CameraEvidence(
            cameraId = id,
            role = DeviceRole.MASTER,
            positionMeters = Vec3(0.0, 1.5, 2.0),
            forwardVector = Vec3(0.0, 0.0, -1.0),
            calibrationConfidence = 0.95,
            synchronizationConfidence = 0.95,
            hardwareReadoutTimingConfidence =
                hardwareTimingConfidence,
            hardwareReadoutP95JitterMilliseconds =
                hardwareP95JitterMillis,
            logicalSensorSyncCalibrated =
                logicalCalibratedSync,
        )
}
