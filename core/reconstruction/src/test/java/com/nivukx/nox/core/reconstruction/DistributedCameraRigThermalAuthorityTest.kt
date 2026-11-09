package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedCameraRigThermalAuthorityTest {

    @Test
    fun thermallyUnstableCameraIsKeptAsEvidenceButNotAuthoritative() {
        val coordinator =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    minimumAuthoritativeThermalHeadroom = 0.50,
                    minimumAuthoritativeDriftStability = 0.60,
                ),
            )

        val cameras =
            listOf(
                evidence(
                    id = "stable",
                    thermal = 0.95,
                    drift = 0.95,
                    x = -0.8,
                ),
                evidence(
                    id = "hot",
                    thermal = 0.20,
                    drift = 0.92,
                    x = 0.8,
                ),
            )

        val plan =
            coordinator.plan(
                timestampNanos = 1_000_000_000L,
                cameras = cameras,
            )

        assertTrue(
            plan.selectedCameraAuthorities["stable"] ==
                DistributedCameraRigCoordinator.Authority.AUTHORITATIVE,
        )
        assertTrue(
            plan.selectedCameraAuthorities["hot"] !=
                DistributedCameraRigCoordinator.Authority.AUTHORITATIVE,
        )
        assertEquals(2, plan.acceptedCameraCount)
    }

    @Test
    fun calibrationDriftCanDowngradeAuthorityWithoutDeletingCamera() {
        val coordinator =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    minimumAuthoritativeThermalHeadroom = 0.35,
                    minimumAuthoritativeDriftStability = 0.70,
                ),
            )

        val cameras =
            listOf(
                evidence(
                    id = "drifting",
                    thermal = 0.95,
                    drift = 0.40,
                    x = -0.7,
                ),
                evidence(
                    id = "stable",
                    thermal = 0.95,
                    drift = 0.95,
                    x = 0.7,
                ),
            )

        val plan =
            coordinator.plan(
                timestampNanos = 2_000_000_000L,
                cameras = cameras,
            )

        assertTrue(
            plan.selectedCameraAuthorities["drifting"] !=
                DistributedCameraRigCoordinator.Authority.AUTHORITATIVE,
        )
        assertEquals(2, plan.acceptedCameraCount)
    }

    private fun evidence(
        id: String,
        thermal: Double,
        drift: Double,
        x: Double,
    ) =
        CaptureRigObservabilityEngine.CameraEvidence(
            cameraId = id,
            role = DeviceRole.CAMERA,
            positionMeters = Vec3(x, 1.5, 2.0),
            forwardVector = Vec3(0.0, 0.0, -1.0),
            calibrationConfidence = 0.96,
            synchronizationConfidence = 0.96,
            trackingConfidence = 0.96,
            thermalHeadroom = thermal,
            transportHeadroom = 0.96,
            calibrationDriftStability = drift,
            dynamicParallaxScore = 0.96,
            surfaceInformationGain = 0.96,
            occlusionCoverage = 0.96,
            trackedObjectCoverage = 0.96,
            photometricConsistency = 0.96,
        )
}
