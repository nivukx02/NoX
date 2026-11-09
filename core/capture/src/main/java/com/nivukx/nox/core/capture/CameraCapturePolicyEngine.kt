package com.nivukx.nox.core.capture

import android.hardware.camera2.params.DynamicRangeProfiles
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CameraDynamicRangeMode
import com.nivukx.nox.core.model.CameraExposureMode
import com.nivukx.nox.core.model.CameraFeatureMatrix
import com.nivukx.nox.core.model.CameraFocusMode
import com.nivukx.nox.core.model.CameraResolutionPreset
import com.nivukx.nox.core.model.MotionCaptureCustomization

/**
 * Resolves operator camera preferences against real Camera2 capabilities.
 *
 * The resolver never promises a capability the device did not advertise. Requested resolution,
 * frame rate, RAW/depth sidecars, manual sensor controls, stabilization and timing preservation are
 * all reduced to an executable CameraCaptureProfile before Camera2 session creation.
 */
class CameraCapturePolicyEngine {

    data class Decision(
        val profile: CameraCaptureProfile,
        val requestedResolution: CameraResolutionPreset,
        val dynamicRangeProfile: Long? = null,
        val tenBitDynamicRangeEnabled: Boolean = false,
        val resolvedWidth: Int,
        val resolvedHeight: Int,
        val resolvedFps: Int,
        val rawEnabled: Boolean,
        val depthEnabled: Boolean,
        val manualExposureEnabled: Boolean,
        val manualFocusEnabled: Boolean,
        val resolvedExposureTimeNanos: Long? = null,
        val motionBlurExposureBudgetNanos: Long? = null,
        val hardwareReadoutTimingPreferred: Boolean = false,
        val warnings: List<String>,
    )

    fun resolve(
        capabilities: CameraFeatureMatrix,
        customization: MotionCaptureCustomization,
    ): Decision {
        val warnings = mutableListOf<String>()

        if (
            customization.cameraExposureMode ==
                CameraExposureMode.SENSITIVITY_PRIORITY
        ) {
            return resolveSensitivityPriority(
                capabilities = capabilities,
                customization = customization,
            )
        }

        val resolvedDynamicRangeProfile =
            when (customization.cameraDynamicRangeMode) {
                CameraDynamicRangeMode.STANDARD ->
                    null
                CameraDynamicRangeMode.TEN_BIT_PREFERRED ->
                    if (capabilities.supports10BitDynamicRange) {
                        when {
                            capabilities.supportedDynamicRangeProfiles
                                .contains(
                                    DynamicRangeProfiles.HLG10,
                                ) ->
                                DynamicRangeProfiles.HLG10
                            capabilities.recommendedDynamicRangeProfile != null ->
                                capabilities.recommendedDynamicRangeProfile
                            else -> {
                                warnings +=
                                    "10-bit dynamic range requested but no advertised 10-bit profile is available"
                                null
                            }
                        }
                    } else {
                        warnings +=
                            "10-bit dynamic range requested but the camera does not advertise 10-bit dynamic range"
                        null
                    }
            }

        val target = resolutionFor(
            preset = customization.cameraResolutionPreset,
            maxWidth = capabilities.maxAnalysisWidth,
            maxHeight = capabilities.maxAnalysisHeight,
            targetFps = customization.targetFps,
        )

        if (
            customization.cameraResolutionPreset == CameraResolutionPreset.UHD_4K &&
            (capabilities.maxAnalysisWidth < 3840 || capabilities.maxAnalysisHeight < 2160)
        ) {
            warnings += "4K requested but the camera does not expose a 4K analysis stream"
        }

        val selectedModeKey =
            target.first.toString() + "x" + target.second.toString()
        val modeMaximumFps =
            capabilities.maximumFpsByResolution[
                selectedModeKey
            ]
                ?: capabilities.maxFps

        val fps = customization.targetFps
            .coerceIn(15, 240)
            .coerceAtMost(modeMaximumFps.coerceAtLeast(15))

        if (fps < customization.targetFps) {
            warnings +=
                "requested frame rate exceeds camera capability at the selected resolution and was clamped"
        }

        val rawEnabled =
            customization.cameraRawSidecarEnabled && capabilities.supportsRaw
        if (
            customization.cameraRawSidecarEnabled &&
            !capabilities.supportsRaw
        ) {
            warnings += "RAW sidecar requested but RAW_SENSOR is unavailable"
        }

        val depthEnabled =
            customization.cameraDepthSidecarEnabled && capabilities.supportsDepth16
        if (
            customization.cameraDepthSidecarEnabled &&
            !capabilities.supportsDepth16
        ) {
            warnings += "depth sidecar requested but DEPTH16 is unavailable"
        }

        val manualExposure =
            customization.cameraExposureMode == CameraExposureMode.MANUAL &&
                capabilities.supportsManualSensor

        val requestedManualExposureNanos =
            if (manualExposure) {
                customization.cameraManualExposureMicros
                    .coerceIn(1L, 500_000L) * 1_000L
            } else {
                null
            }

        val capabilityMinimumExposure =
            capabilities.minimumExposureTimeNanos
                .takeIf { it > 0L }
        val capabilityMaximumExposure =
            capabilities.maximumExposureTimeNanos
                .takeIf { it > 0L }

        val resolvedManualExposureNanos =
            requestedManualExposureNanos?.let { requested ->
                requested
                    .coerceAtLeast(
                        capabilityMinimumExposure ?: 1L,
                    )
                    .let { value ->
                        capabilityMaximumExposure
                            ?.let { value.coerceAtMost(it) }
                            ?: value
                    }
            }

        val motionBlurExposureBudgetNanos =
            (
                1_000_000_000L /
                    fps.coerceAtLeast(1).toLong() *
                    0.25
            ).toLong().coerceAtLeast(1L)

        if (
            customization.cameraExposureMode == CameraExposureMode.MANUAL &&
            !capabilities.supportsManualSensor
        ) {
            warnings += "manual exposure requested but MANUAL_SENSOR is unavailable"
        }

        val manualFocus =
            customization.cameraFocusMode == CameraFocusMode.MANUAL

        val profile = CameraCaptureProfile(
            analysisWidth = target.first,
            dynamicRangeMode =
                customization.cameraDynamicRangeMode,
            dynamicRangeProfile =
                resolvedDynamicRangeProfile,
            analysisHeight = target.second,
            fps = fps,
            exposureMode =
                if (manualExposure) {
                    CameraExposureMode.MANUAL
                } else {
                    CameraExposureMode.AUTO
                },
            antiBandingMode =
                customization.cameraAntibandingMode,
            focusMode = customization.cameraFocusMode,
            enableRawSidecar = rawEnabled,
            enableDepthSidecar = depthEnabled,
            enableGyroscope = true,
            enableAccelerometer = true,
            preferredGpu = com.nivukx.nox.core.model.GpuBackend.OPENGL_ES_COMPUTE,
            manualExposureNanos =
                resolvedManualExposureNanos,
            manualIso =
                if (manualExposure) {
                    customization.cameraManualIso
                        .coerceIn(25, 12_800)
                } else {
                    null
                },
            oisEnabled =
                customization.cameraOisEnabled &&
                    capabilities.supportsOis,
            electronicStabilizationEnabled =
                customization.cameraElectronicStabilizationEnabled,
            preserveSensorTimestamp =
                customization.cameraPreserveSensorTimestamp,
            exposureCompensationEv =
                customization.cameraExposureCompensationEv
                    .coerceIn(-8.0, 8.0),
            autoExposureLock =
                customization.cameraAutoExposureLock &&
                    !manualExposure,
            autoWhiteBalanceLock =
                customization.cameraAutoWhiteBalanceLock,
            preciseColorTemperatureEnabled =
                customization.cameraPreciseColorTemperatureEnabled &&
                    capabilities.supportsPreciseColorTemperature,
            colorTemperatureKelvin =
                customization.cameraColorTemperatureKelvin
                    .coerceIn(
                        capabilities.minimumColorTemperatureKelvin,
                        capabilities.maximumColorTemperatureKelvin,
                    ),
            colorTint =
                customization.cameraColorTint
                    .coerceIn(
                        capabilities.minimumColorTint,
                        capabilities.maximumColorTint,
                    ),
            manualFocusDistanceDiopters =
                if (manualFocus) {
                    customization.cameraManualFocusDistanceDiopters
                        ?.coerceIn(0.0, 1000.0)
                } else {
                    null
                },
        )

        if (
            customization.cameraPreciseColorTemperatureEnabled &&
            !capabilities.supportsPreciseColorTemperature
        ) {
            warnings +=
                "precise CCT white balance requested but unsupported by this camera"
        }

        if (
            customization.cameraOisEnabled &&
            !capabilities.supportsOis
        ) {
            warnings += "optical stabilization requested but unavailable"
        }

        return Decision(
            profile = profile,
            requestedResolution = customization.cameraResolutionPreset,
            dynamicRangeProfile = resolvedDynamicRangeProfile,
            tenBitDynamicRangeEnabled =
                resolvedDynamicRangeProfile != null,
            resolvedWidth = target.first,
            resolvedHeight = target.second,
            resolvedFps = fps,
            rawEnabled = rawEnabled,
            depthEnabled = depthEnabled,
            manualExposureEnabled = manualExposure,
            manualFocusEnabled = manualFocus,
            resolvedExposureTimeNanos =
                resolvedManualExposureNanos,
            motionBlurExposureBudgetNanos =
                motionBlurExposureBudgetNanos,
            hardwareReadoutTimingPreferred =
                capabilities.supportsHardwareReadoutTimestamp,
            warnings = warnings.distinct(),
        )
    }

    private fun resolveSensitivityPriority(
        capabilities: CameraFeatureMatrix,
        customization: MotionCaptureCustomization,
    ): Decision {
        val supported =
            android.os.Build.VERSION.SDK_INT >= 36 &&
                capabilities.supportedAePriorityModes.contains(
                    android.hardware.camera2.CameraMetadata
                        .CONTROL_AE_PRIORITY_MODE_SENSOR_SENSITIVITY_PRIORITY,
                )

        if (!supported) {
            val fallbackMode =
                if (capabilities.supportsManualSensor) {
                    CameraExposureMode.MANUAL
                } else {
                    CameraExposureMode.AUTO
                }
            val fallback =
                resolve(
                    capabilities = capabilities,
                    customization =
                        customization.copyWith {
                            cameraExposureMode = fallbackMode
                        },
                )
            return fallback.copy(
                warnings =
                    fallback.warnings +
                        "sensitivity-priority AE requested but this camera does not advertise Android 16 hybrid AE priority; using $fallbackMode fallback",
            )
        }

        val base =
            resolve(
                capabilities = capabilities,
                customization =
                    customization.copyWith {
                        cameraExposureMode =
                            CameraExposureMode.AUTO
                    },
            )

        val profile =
            base.profile.copy(
                exposureMode =
                    CameraExposureMode.SENSITIVITY_PRIORITY,
                manualIso =
                    customization.cameraManualIso
                        .coerceIn(25, 12_800),
                manualExposureNanos = null,
                autoExposureLock = false,
            )

        return base.copy(
            profile = profile,
            manualExposureEnabled = false,
            resolvedExposureTimeNanos = null,
            warnings =
                base.warnings +
                    "Android 16 hybrid AE sensitivity priority enabled",
        )
    }

    private fun resolutionFor(
        preset: CameraResolutionPreset,
        maxWidth: Int,
        maxHeight: Int,
        targetFps: Int,
    ): Pair<Int, Int> {
        val width = maxWidth.coerceAtLeast(640)
        val height = maxHeight.coerceAtLeast(480)

        val requested = when (preset) {
            CameraResolutionPreset.HD_720P ->
                1280 to 720
            CameraResolutionPreset.FULL_HD_1080P ->
                1920 to 1080
            CameraResolutionPreset.UHD_4K ->
                3840 to 2160
            CameraResolutionPreset.MAX_SUPPORTED ->
                width to height
            CameraResolutionPreset.AUTO -> {
                when {
                    width >= 3840 &&
                        height >= 2160 &&
                        targetFps <= 60 ->
                        3840 to 2160

                    width >= 1920 &&
                        height >= 1080 ->
                        1920 to 1080

                    else ->
                        1280 to 720
                }
            }
        }

        return requested.first
            .coerceAtMost(width)
            .coerceAtLeast(640) to
            requested.second
                .coerceAtMost(height)
                .coerceAtLeast(480)
    }
}
