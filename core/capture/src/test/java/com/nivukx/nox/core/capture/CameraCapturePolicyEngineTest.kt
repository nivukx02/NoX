package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CameraExposureMode
import com.nivukx.nox.core.model.CameraFeatureMatrix
import com.nivukx.nox.core.model.CameraFocusMode
import com.nivukx.nox.core.model.CameraDynamicRangeMode
import com.nivukx.nox.core.model.CameraResolutionPreset
import com.nivukx.nox.core.model.MotionCaptureCustomization
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CameraCapturePolicyEngineTest {

    private fun capabilities(
        raw: Boolean = true,
        depth: Boolean = true,
        manual: Boolean = true,
        ois: Boolean = true,
        maxFps: Int = 120,
        width: Int = 3840,
        height: Int = 2160,
        tenBit: Boolean = false,
        recommendedDynamicRangeProfile: Long? = null,
        aePriorityModes: Set<Int> = emptySet(),
    ): CameraFeatureMatrix =
        CameraFeatureMatrix(
            cameraId = "test",
            supportedDynamicRangeProfiles =
                recommendedDynamicRangeProfile
                    ?.let(::setOf)
                    ?: emptySet(),
            recommendedDynamicRangeProfile =
                recommendedDynamicRangeProfile,
            supports10BitDynamicRange = tenBit,
            supportedAePriorityModes = aePriorityModes,
            supportsRaw = raw,
            rawWidth = if (raw) 4000 else 0,
            rawHeight = if (raw) 3000 else 0,
            supportsDepth16 = depth,
            depthWidth = if (depth) 640 else 0,
            depthHeight = if (depth) 480 else 0,
            supportsManualSensor = manual,
            supportsOis = ois,
            supportsPdaf = true,
            supportsConcurrentCamera = true,
            maxAnalysisWidth = width,
            maxAnalysisHeight = height,
            maxFps = maxFps,
            recommendedProfile = CameraCaptureProfile(
                analysisWidth = 1920,
                analysisHeight = 1080,
                fps = 60,
            ),
        )

    @Test
    fun autoSelects4KWhenCapabilityAndRatePermit() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(),
            MotionCaptureCustomization(
                targetFps = 60,
                cameraResolutionPreset = CameraResolutionPreset.AUTO,
            ),
        )

        assertEquals(3840, decision.profile.analysisWidth)
        assertEquals(2160, decision.profile.analysisHeight)
        assertEquals(60, decision.profile.fps)
        assertTrue(decision.warnings.isEmpty())
    }

    @Test
    fun fpsIsClampedToAdvertisedCameraMaximum() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(maxFps = 60),
            MotionCaptureCustomization(targetFps = 120),
        )

        assertEquals(60, decision.profile.fps)
        assertTrue(
            decision.warnings.any {
                it.contains("frame rate", ignoreCase = true)
            },
        )
    }

    @Test
    fun resolution_specific_fps_limit_is_applied_before_camera_session_creation() {
        val engine = CameraCapturePolicyEngine()
        val decision =
            engine.resolve(
                capabilities().copy(
                    maxFps = 120,
                    maximumFpsByResolution =
                        mapOf(
                            "1920x1080" to 30,
                            "3840x2160" to 24,
                        ),
                ),
                MotionCaptureCustomization(
                    targetFps = 120,
                    cameraResolutionPreset =
                        CameraResolutionPreset.FULL_HD_1080P,
                ),
            )

        assertEquals(30, decision.resolvedFps)
        assertTrue(
            decision.warnings.any {
                it.contains(
                    "selected resolution",
                    ignoreCase = true,
                )
            },
        )
    }

    @Test
    fun unavailableSidecarsAreRefusedWithoutLying() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(raw = false, depth = false),
            MotionCaptureCustomization(
                cameraRawSidecarEnabled = true,
                cameraDepthSidecarEnabled = true,
            ),
        )

        assertFalse(decision.profile.enableRawSidecar)
        assertFalse(decision.profile.enableDepthSidecar)
        assertTrue(decision.warnings.size >= 2)
    }

    @Test
    fun manualExposureFallsBackToAutoWithoutManualSensor() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(manual = false),
            MotionCaptureCustomization(
                cameraExposureMode = CameraExposureMode.MANUAL,
                cameraManualIso = 1600,
                cameraManualExposureMicros = 8000,
            ),
        )

        assertEquals(
            CameraExposureMode.AUTO,
            decision.profile.exposureMode,
        )
        assertFalse(decision.manualExposureEnabled)
        assertTrue(
            decision.warnings.any {
                it.contains("MANUAL_SENSOR", ignoreCase = true)
            },
        )
    }

    @Test
    fun manualFocusCarriesFiniteDistance() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(),
            MotionCaptureCustomization(
                cameraFocusMode = CameraFocusMode.MANUAL,
                cameraManualFocusDistanceDiopters = 2.5,
            ),
        )

        assertEquals(
            CameraFocusMode.MANUAL,
            decision.profile.focusMode,
        )
        assertEquals(
            2.5,
            decision.profile.manualFocusDistanceDiopters,
        )
        assertTrue(decision.manualFocusEnabled)
    }

    @Test
    fun unsupportedOisProducesWarningAndDisablesOis() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(ois = false),
            MotionCaptureCustomization(cameraOisEnabled = true),
        )

        assertFalse(decision.profile.oisEnabled)
        assertTrue(
            decision.warnings.any {
                it.contains("stabilization", ignoreCase = true)
            },
        )
    }

    @Test
    fun tenBitPreferredSelectsAdvertisedDynamicRangeProfile() {
        val engine = CameraCapturePolicyEngine()
        val decision =
            engine.resolve(
                capabilities(
                    tenBit = true,
                    recommendedDynamicRangeProfile = 7L,
                ),
                MotionCaptureCustomization(
                    cameraDynamicRangeMode =
                        CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                ),
            )

        assertEquals(
            7L,
            decision.profile.dynamicRangeProfile,
        )
        assertTrue(
            decision.tenBitDynamicRangeEnabled,
        )
        assertTrue(
            decision.warnings.isEmpty(),
        )
    }

    @Test
    fun tenBitRequestFallsBackWithoutCapability() {
        val engine = CameraCapturePolicyEngine()
        val decision =
            engine.resolve(
                capabilities(
                    tenBit = false,
                    recommendedDynamicRangeProfile = null,
                ),
                MotionCaptureCustomization(
                    cameraDynamicRangeMode =
                        CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                ),
            )

        assertEquals(
            null,
            decision.profile.dynamicRangeProfile,
        )
        assertFalse(
            decision.tenBitDynamicRangeEnabled,
        )
        assertTrue(
            decision.warnings.any {
                it.contains("10-bit", ignoreCase = true)
            },
        )
    }

    @Test
    fun explicitUhdRequestClampsToDeviceEnvelope() {
        val engine = CameraCapturePolicyEngine()
        val decision = engine.resolve(
            capabilities(width = 1920, height = 1080),
            MotionCaptureCustomization(
                cameraResolutionPreset = CameraResolutionPreset.UHD_4K,
            ),
        )

        assertEquals(1920, decision.profile.analysisWidth)
        assertEquals(1080, decision.profile.analysisHeight)
        assertTrue(
            decision.warnings.any {
                it.contains("4K", ignoreCase = true)
            },
        )
    }
    @Test
    fun manualExposureIsClampedToSensorRangeAndReported() {
        val engine = CameraCapturePolicyEngine()
        val capabilities =
            capabilities().copy(
                minimumExposureTimeNanos = 2_000_000L,
                maximumExposureTimeNanos = 8_000_000L,
            )

        val decision =
            engine.resolve(
                capabilities,
                MotionCaptureCustomization(
                    cameraExposureMode =
                        CameraExposureMode.MANUAL,
                    cameraManualExposureMicros = 500L,
                    cameraManualIso = 400,
                    targetFps = 120,
                ),
            )

        assertEquals(
            2_000_000L,
            decision.resolvedExposureTimeNanos,
        )
        assertTrue(
            decision.warnings.any {
                it.contains("clamped")
            },
        )
    }

    @Test
    fun longManualExposureIsFlaggedForFastMotionCapture() {
        val engine = CameraCapturePolicyEngine()
        val capabilities =
            capabilities().copy(
                minimumExposureTimeNanos = 100_000L,
                maximumExposureTimeNanos = 100_000_000L,
            )

        val decision =
            engine.resolve(
                capabilities,
                MotionCaptureCustomization(
                    cameraExposureMode =
                        CameraExposureMode.MANUAL,
                    cameraManualExposureMicros = 8_000L,
                    cameraManualIso = 800,
                    targetFps = 120,
                ),
            )

        assertTrue(
            decision.motionBlurExposureBudgetNanos ?: 0L <
                decision.resolvedExposureTimeNanos ?: 0L,
        )
        assertTrue(
            decision.warnings.any {
                it.contains("motion-blur budget")
            },
        )
    }

    @Test
    fun antiBandingModeIsPreservedInResolvedProfile() {
        val engine = CameraCapturePolicyEngine()
        val decision =
            engine.resolve(
                capabilities(),
                MotionCaptureCustomization(
                    cameraAntibandingMode =
                        CameraAntibandingMode.HZ_50,
                ),
            )

        assertEquals(
            CameraAntibandingMode.HZ_50,
            decision.profile.antiBandingMode,
        )
    }

    @Test
    fun sensitivityPriorityProfile_requires_iso_but_not_manual_shutter() {
        val profile =
            CameraCaptureProfile(
                exposureMode =
                    CameraExposureMode.SENSITIVITY_PRIORITY,
                manualIso = 640,
                manualExposureNanos = null,
            )

        assertEquals(
            CameraExposureMode.SENSITIVITY_PRIORITY,
            profile.exposureMode,
        )
        assertEquals(640, profile.manualIso)
        assertEquals(null, profile.manualExposureNanos)
    }

    @Test
    fun sensitivityPriority_falls_back_when_camera_does_not_advertise_mode() {
        val engine = CameraCapturePolicyEngine()
        val decision =
            engine.resolve(
                capabilities(aePriorityModes = emptySet()),
                MotionCaptureCustomization(
                    cameraExposureMode =
                        CameraExposureMode.SENSITIVITY_PRIORITY,
                    cameraManualIso = 800,
                ),
            )

        assertTrue(
            decision.profile.exposureMode !=
                CameraExposureMode.SENSITIVITY_PRIORITY,
        )
        assertTrue(
            decision.warnings.any {
                it.contains(
                    "does not advertise Android 16 hybrid AE priority",
                    ignoreCase = true,
                )
            },
        )
    }

    @Test
    fun advertised_sensitivity_priority_mode_is_retained_by_capability_matrix() {
        val priority =
            android.hardware.camera2.CameraMetadata
                .CONTROL_AE_PRIORITY_MODE_SENSOR_SENSITIVITY_PRIORITY

        val matrix =
            capabilities(
                aePriorityModes = setOf(priority),
            )

        assertTrue(
            priority in matrix.supportedAePriorityModes,
        )
    }

}
