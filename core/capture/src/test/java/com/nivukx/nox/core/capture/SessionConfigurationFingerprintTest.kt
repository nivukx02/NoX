package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraExtrinsics
import com.nivukx.nox.core.model.DeviceCapabilities
import com.nivukx.nox.core.model.DeviceDescriptor
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.MotionCaptureMode
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SessionConfigurationFingerprintTest {

    private fun capabilities(model: String) =
        DeviceCapabilities(
            model = model,
            manufacturer = "Test",
            androidApi = 37,
            cameraIds = listOf("0"),
            supportsRaw = true,
            supportsConcurrentCamera = true,
            supportsGyroscope = true,
            supportsAccelerometer = true,
            maxAnalysisWidth = 1920,
            maxAnalysisHeight = 1080,
            maxFps = 60,
        )

    private fun device(id: String, model: String = "Phone-$id") =
        DeviceDescriptor(
            id = DeviceId(id),
            role = DeviceRole.CAMERA,
            displayName = id,
            capabilities = capabilities(model),
            connected = true,
            clockOffsetNanos = 200_000L,
            roundTripNanos = 8_000_000L,
            packetLoss = 0.01,
        )

    private fun calibration(scale: Double = 1.0) =
        CalibrationSnapshot(
            intrinsics = mapOf(
                DeviceId("b") to CameraIntrinsics(
                    fx = 1200.0 * scale,
                    fy = 1200.0 * scale,
                    cx = 960.0,
                    cy = 540.0,
                    imageWidth = 1920,
                    imageHeight = 1080,
                ),
                DeviceId("a") to CameraIntrinsics(
                    fx = 1180.0 * scale,
                    fy = 1180.0 * scale,
                    cx = 960.0,
                    cy = 540.0,
                    imageWidth = 1920,
                    imageHeight = 1080,
                ),
            ),
            extrinsics = mapOf(
                DeviceId("b") to CameraExtrinsics(
                    pose = Pose(
                        position = Vec3(1.5, 0.0, 0.0),
                    ),
                    referenceDevice = DeviceId("a"),
                ),
                DeviceId("a") to CameraExtrinsics(
                    pose = Pose(),
                    referenceDevice = DeviceId("a"),
                ),
            ),
            reprojectionErrorPx = 1.8,
            calibratedAtNanos = 123_456_789L,
        )

    @Test
    fun deviceOrderingDoesNotChangeFingerprint() {
        val customization = MotionCaptureCustomization.multiViewPrecision()
        val profile = CameraCaptureProfile.adaptive(1920, 1080, 60)
        val first = SessionConfigurationFingerprint.compute(
            customization = customization,
            captureProfile = profile,
            mode = MotionCaptureMode.MULTI_VIEW_RGB,
            calibration = calibration(),
            devices = listOf(device("a"), device("b")),
        )
        val second = SessionConfigurationFingerprint.compute(
            customization = customization,
            captureProfile = profile,
            mode = MotionCaptureMode.MULTI_VIEW_RGB,
            calibration = calibration(),
            devices = listOf(device("b"), device("a")),
        )

        assertEquals(first, second)
        assertTrue(first.startsWith("NXCFG5-"))
    }

    @Test
    fun faceEyeFusionStrictnessIsExplicitlyFingerprintVersioned() {
        val profile =
            CameraCaptureProfile.adaptive(1920, 1080, 60)
        val base =
            MotionCaptureCustomization.production()
        val strict =
            base.copyWith {
                faceEyeFusionStrictness = 0.99
            }

        val baseFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = base,
                captureProfile = profile,
                mode = MotionCaptureMode.MONOCULAR_RGB,
                calibration = null,
                devices = listOf(device("a")),
            )
        val strictFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = strict,
                captureProfile = profile,
                mode = MotionCaptureMode.MONOCULAR_RGB,
                calibration = null,
                devices = listOf(device("a")),
            )

        assertTrue(
            baseFingerprint.startsWith("NXCFG5-"),
        )
        assertTrue(
            strictFingerprint.startsWith("NXCFG5-"),
        )
        assertNotEquals(
            baseFingerprint,
            strictFingerprint,
        )
    }

    @Test
    fun physicsConstraintStrictnessChangesFingerprint() {
        val profile =
            CameraCaptureProfile.adaptive(1920, 1080, 60)
        val base =
            MotionCaptureCustomization.production()
        val strict =
            base.copyWith {
                physicsConstraintStrictness = 0.99
            }

        val baseFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = base,
                captureProfile = profile,
                mode = MotionCaptureMode.MONOCULAR_RGB,
                calibration = null,
                devices = listOf(device("a")),
            )
        val strictFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = strict,
                captureProfile = profile,
                mode = MotionCaptureMode.MONOCULAR_RGB,
                calibration = null,
                devices = listOf(device("a")),
            )

        assertTrue(baseFingerprint.startsWith("NXCFG5-"))
        assertTrue(strictFingerprint.startsWith("NXCFG5-"))
        assertNotEquals(baseFingerprint, strictFingerprint)
    }

    @Test
    fun changingMotionCustomizationChangesFingerprint() {
        val profile = CameraCaptureProfile.adaptive(1920, 1080, 60)
        val base = SessionConfigurationFingerprint.compute(
            customization = MotionCaptureCustomization.production(),
            captureProfile = profile,
            mode = MotionCaptureMode.MONOCULAR_RGB,
            calibration = null,
            devices = listOf(device("a")),
        )
        val changed = SessionConfigurationFingerprint.compute(
            customization =
                MotionCaptureCustomization.production().copyWith {
                    faceDetail = 0.73
                },
            captureProfile = profile,
            mode = MotionCaptureMode.MONOCULAR_RGB,
            calibration = null,
            devices = listOf(device("a")),
        )

        assertNotEquals(base, changed)
    }

    @Test
    fun changingCalibrationGeometryChangesFingerprint() {
        val customization = MotionCaptureCustomization.multiViewPrecision()
        val profile = CameraCaptureProfile.adaptive(1920, 1080, 60)
        val base = SessionConfigurationFingerprint.compute(
            customization = customization,
            captureProfile = profile,
            mode = MotionCaptureMode.MULTI_VIEW_RGB,
            calibration = calibration(1.0),
            devices = listOf(device("a"), device("b")),
        )
        val changed = SessionConfigurationFingerprint.compute(
            customization = customization,
            captureProfile = profile,
            mode = MotionCaptureMode.MULTI_VIEW_RGB,
            calibration = calibration(1.001),
            devices = listOf(device("a"), device("b")),
        )

        assertNotEquals(base, changed)
    }
    @Test
    fun distributedCameraPolicyChangesFingerprint() {
        val profile =
            CameraCaptureProfile.adaptive(1920, 1080, 60)
        val base =
            MotionCaptureCustomization.production()
        val changed =
            base.copyWith {
                distributedAuthoritativeCameraBudget =
                    24
            }

        val baseFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = base,
                captureProfile = profile,
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibration = null,
                devices = listOf(device("a")),
            )
        val changedFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = changed,
                captureProfile = profile,
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibration = null,
                devices = listOf(device("a")),
            )

        assertNotEquals(baseFingerprint, changedFingerprint)
    }

    @Test
    fun hardwareTimingCapabilityChangesFingerprint() {
        val profile =
            CameraCaptureProfile.adaptive(1920, 1080, 60)
        val base =
            device("a").copy(
                capabilities =
                    capabilities("baseline").copy(
                        supportsHardwareReadoutTimestamp = false,
                    ),
            )
        val hardware =
            device("a").copy(
                capabilities =
                    capabilities("hardware").copy(
                        supportsHardwareReadoutTimestamp = true,
                        maximumVerifiedConcurrentCameras = 4,
                        minimumExposureTimeNanos = 100_000L,
                        maximumExposureTimeNanos = 20_000_000L,
                    ),
            )

        val baseFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = MotionCaptureCustomization.production(),
                captureProfile = profile,
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibration = null,
                devices = listOf(base),
            )
        val hardwareFingerprint =
            SessionConfigurationFingerprint.compute(
                customization = MotionCaptureCustomization.production(),
                captureProfile = profile,
                mode = MotionCaptureMode.MULTI_VIEW_RGB,
                calibration = null,
                devices = listOf(hardware),
            )

        assertNotEquals(baseFingerprint, hardwareFingerprint)
        assertTrue(hardwareFingerprint.startsWith("NXCFG5-"))
    }

}
