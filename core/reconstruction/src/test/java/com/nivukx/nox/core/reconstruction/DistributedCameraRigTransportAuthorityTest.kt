package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.test.Test
import kotlin.test.assertTrue

class DistributedCameraRigTransportAuthorityTest {

    @Test
    fun severeOpticalBlurQuarantinesCameraInsteadOfContaminatingAuthority() {
        val coordinator =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    maximumGlobalCameras = 4,
                    maximumShards = 1,
                    targetSelectedCameras = 4,
                    maximumSelectedCameras = 4,
                ),
            )

        val cameras =
            (0 until 4).map { index ->
                CaptureRigObservabilityEngine.CameraEvidence(
                    cameraId = "blur-$index",
                    role = DeviceRole.CAMERA,
                    positionMeters =
                        Vec3(
                            index.toDouble(),
                            1.5,
                            2.5,
                        ),
                    forwardVector = Vec3(0.0, 0.0, -1.0),
                    calibrationConfidence = 0.99,
                    synchronizationConfidence = 0.99,
                    trackingConfidence = 0.99,
                    thermalHeadroom = 0.99,
                    transportHeadroom = 0.99,
                    imageErrorPx = 0.10,
                    centroidConnectivity = 0.99,
                    calibrationDriftStability = 0.99,
                    trackedObjectCoverage = 0.99,
                    surfaceInformationGain = 0.99,
                    occlusionCoverage = 0.99,
                    dynamicParallaxScore = 0.99,
                    temporalResidualP95Milliseconds = 0.5,
                    temporalOutlierRatio = 0.01,
                    hardwareReadoutTimingConfidence = 0.99,
                    hardwareReadoutP95JitterMilliseconds = 0.5,
                    logicalSensorSyncCalibrated = true,
                    motionBlurRisk =
                        if (index == 0) 0.97 else 0.04,
                )
            }

        val result =
            coordinator.plan(
                timestampNanos = 11_000_000_000L,
                cameras = cameras,
            )

        assertTrue(
            result.selectedCameraAuthorities["blur-0"] ==
                DistributedCameraRigCoordinator.Authority.QUARANTINED ||
                result.selectedCameraAuthorities["blur-0"] !=
                    DistributedCameraRigCoordinator.Authority.AUTHORITATIVE,
        )
        assertTrue(
            result.warnings.contains(
                "authoritative-optical-blur-limited",
            ) ||
                result.warnings.contains(
                    "camera-quarantine-active",
                ),
        )
    }

    @Test
    fun lowTransportHeadroomPreventsAuthoritativeAdmission() {
        val coordinator =
            DistributedCameraRigCoordinator(
                DistributedCameraRigCoordinator.Config(
                    maximumGlobalCameras = 8,
                    maximumShards = 2,
                    targetSelectedCameras = 4,
                    maximumSelectedCameras = 4,
                    minimumAuthoritativeTransportHeadroom = 0.75,
                ),
            )

        val cameras =
            (0 until 4).map { index ->
                CaptureRigObservabilityEngine.CameraEvidence(
                    cameraId = "cam-$index",
                    role = DeviceRole.CAMERA,
                    positionMeters =
                        Vec3(
                            index.toDouble() * 1.0,
                            1.5,
                            2.5,
                        ),
                    forwardVector = Vec3(0.0, 0.0, -1.0),
                    calibrationConfidence = 0.99,
                    synchronizationConfidence = 0.99,
                    trackingConfidence = 0.99,
                    thermalHeadroom = 0.99,
                    transportHeadroom =
                        if (index == 0) 0.20 else 0.98,
                    imageErrorPx = 0.10,
                    centroidConnectivity = 0.99,
                    calibrationDriftStability = 0.99,
                    trackedObjectCoverage = 0.99,
                    surfaceInformationGain = 0.99,
                    occlusionCoverage = 0.99,
                    dynamicParallaxScore = 0.99,
                    temporalResidualP95Milliseconds = 0.5,
                    temporalOutlierRatio = 0.01,
                    hardwareReadoutTimingConfidence = 0.99,
                    hardwareReadoutP95JitterMilliseconds = 0.5,
                    logicalSensorSyncCalibrated = true,
                )
            }

        val result =
            coordinator.plan(
                timestampNanos = 10_000_000_000L,
                cameras = cameras,
            )

        assertTrue(
            result.selectedCameraAuthorities["cam-0"] !=
                DistributedCameraRigCoordinator.Authority.AUTHORITATIVE,
        )
        assertTrue(
            result.warnings.contains(
                "authoritative-transport-headroom-limited",
            ),
        )
    }
}
