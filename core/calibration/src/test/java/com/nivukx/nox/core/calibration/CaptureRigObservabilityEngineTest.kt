package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CaptureRigObservabilityEngineTest {
    private fun camera(
        id: String,
        role: DeviceRole,
        x: Double,
        z: Double,
        calibrationConfidence: Double = 0.95,
        imageErrorPx: Double = 0.0,
        centroidConnectivity: Double = 1.0,
        trackedObjectCoverage: Double = 1.0,
        calibrationDriftStability: Double = 1.0,
        surfaceInformationGain: Double = 1.0,
        occlusionCoverage: Double = 1.0,
        dynamicParallaxScore: Double = 1.0,
        temporalResidualP95Milliseconds: Double = 0.0,
        temporalOutlierRatio: Double = 0.0,
        motionBlurRisk: Double? = null,
    ) = CaptureRigObservabilityEngine.CameraEvidence(
        cameraId = id,
        role = role,
        positionMeters = Vec3(x, 1.4, z),
        forwardVector = Vec3(-x, 0.0, -z),
        calibrationConfidence = calibrationConfidence,
        synchronizationConfidence = 0.94,
        trackingConfidence = 0.93,
        thermalHeadroom = 0.90,
        transportHeadroom = 0.96,
        imageErrorPx = imageErrorPx,
        centroidConnectivity = centroidConnectivity,
        trackedObjectCoverage = trackedObjectCoverage,
        calibrationDriftStability = calibrationDriftStability,
        surfaceInformationGain = surfaceInformationGain,
        occlusionCoverage = occlusionCoverage,
        dynamicParallaxScore = dynamicParallaxScore,
        temporalResidualP95Milliseconds = temporalResidualP95Milliseconds,
        temporalOutlierRatio = temporalOutlierRatio,
        motionBlurRisk = motionBlurRisk,
    )

    @Test
    fun distributedRigHasUsefulObservability() {
        val health = CaptureRigObservabilityEngine().analyze(
            listOf(
                camera("front", DeviceRole.FRONT, 0.0, -2.0),
                camera("right", DeviceRole.RIGHT, 2.0, 0.0),
                camera("rear", DeviceRole.REAR, 0.0, 2.0),
                camera("left", DeviceRole.LEFT, -2.0, 0.0),
            ),
        )
        assertEquals(4, health.activeCameras)
        assertTrue(health.observabilityScore > 0.45)
        assertTrue(health.sectorCoverage >= 0.50)
    }

    @Test
    fun selectionAvoidsRedundantOnlyViews() {
        val selection = CaptureRigObservabilityEngine().select(
            listOf(
                camera("front-a", DeviceRole.FRONT, 0.0, -2.0),
                camera("front-b", DeviceRole.FRONT, 0.05, -2.02, 0.55),
                camera("right", DeviceRole.RIGHT, 2.0, 0.0),
                camera("left", DeviceRole.LEFT, -2.0, 0.0),
                camera("rear", DeviceRole.REAR, 0.0, 2.0),
            ),
            requestedCount = 3,
        )
        assertEquals(3, selection.selected.size)
        assertTrue(selection.selected.any { it.cameraId == "right" })
        assertTrue(selection.selected.any { it.cameraId == "left" })
    }

    @Test
    fun opticalBlurLowersHealthWithoutBreakingLegacyUnknownState() {
        val clean =
            camera(
                id = "clean",
                role = DeviceRole.FRONT,
                x = 0.0,
                z = -2.0,
                motionBlurRisk = 0.05,
            )
        val blurred =
            camera(
                id = "blurred",
                role = DeviceRole.RIGHT,
                x = 2.0,
                z = 0.0,
                motionBlurRisk = 0.90,
            )
        val legacy =
            camera(
                id = "legacy",
                role = DeviceRole.LEFT,
                x = -2.0,
                z = 0.0,
            )

        assertTrue(clean.health > blurred.health)
        assertTrue(legacy.health.isFinite())
        assertTrue(legacy.health in 0.0..1.0)
        assertTrue(blurred.health in 0.0..1.0)
    }

    @Test
    fun emptyRigFailsClosed() {
        val health = CaptureRigObservabilityEngine().analyze(emptyList())
        assertEquals(0, health.activeCameras)
        assertEquals(0.0, health.observabilityScore)
        assertTrue("no-active-cameras" in health.warnings)
    }

    @Test
    fun recovery_required_camera_is_excluded_from_active_observability() {
        val engine = CaptureRigObservabilityEngine()
        val result = engine.analyze(
            listOf(
                camera(
                    id = "good",
                    role = DeviceRole.RIGHT,
                    x = 2.0,
                    z = 0.0,
                ),
                camera(
                    id = "bad",
                    role = DeviceRole.LEFT,
                    x = -2.0,
                    z = 0.0,
                    calibrationConfidence = 0.25,
                    imageErrorPx = 8.0,
                    centroidConnectivity = 0.25,
                    trackedObjectCoverage = 0.20,
                    calibrationDriftStability = 0.20,
                ),
            ),
        )

        assertEquals(1, result.activeCameras)
        assertTrue(
            result.recoveryActions["bad"] ==
                CameraAutoHealPolicyEngine.Action.RECALIBRATION_REQUIRED,
        )
    }


    @Test
    fun temporalContractAndInformationGainAffectSelection() {
        val highInformation = camera(
            id = "information-rich",
            role = DeviceRole.FRONT,
            x = 1.8,
            z = -1.8,
            surfaceInformationGain = 1.0,
            occlusionCoverage = 1.0,
            dynamicParallaxScore = 1.0,
        )
        val lowInformation = camera(
            id = "redundant",
            role = DeviceRole.FRONT,
            x = 0.10,
            z = -2.02,
            surfaceInformationGain = 0.20,
            occlusionCoverage = 0.25,
            dynamicParallaxScore = 0.20,
            temporalResidualP95Milliseconds = 18.0,
            temporalOutlierRatio = 0.30,
        )
        val healthyPair = camera(
            id = "right",
            role = DeviceRole.RIGHT,
            x = 2.0,
            z = 0.0,
        )

        val result =
            CaptureRigObservabilityEngine().select(
                cameras = listOf(
                    highInformation,
                    lowInformation,
                    healthyPair,
                ),
                requestedCount = 2,
            )

        assertTrue(
            result.selected.any {
                it.cameraId == "information-rich"
            },
        )
        assertTrue(
            result.warnings.any {
                it == "camera-temporal-contract-breach" ||
                    it == "selected-rig-contains-low-health-camera"
            } || result.rejected.any {
                it.cameraId == "redundant"
            },
        )
    }

}
