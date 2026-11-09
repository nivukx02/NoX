package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.DeviceDescriptor
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.MotionCaptureMode
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Locale

/**
 * Deterministic provenance fingerprint for a NoX capture configuration.
 *
 * The fingerprint captures the effective configuration, not transient runtime telemetry. It is
 * intentionally order-independent for device/calibration maps so a reordered collection cannot
 * change the identity of an otherwise identical capture setup.
 */
object SessionConfigurationFingerprint {

    const val SCHEMA: Int = 6
    private const val ENGINE_CONTRACT = "nox-motion-hybrid-production-v6"

    fun compute(
        customization: MotionCaptureCustomization,
        captureProfile: CameraCaptureProfile?,
        mode: MotionCaptureMode,
        calibration: CalibrationSnapshot?,
        devices: List<DeviceDescriptor>,
    ): String {
        val canonical = buildString {
            field("schema", SCHEMA.toString())
            field("engineContract", ENGINE_CONTRACT)
            field("mode", mode.name)
            field(
                "motionCustomizationHashCode",
                customization.hashCode(),
            )
            field(
                "motionCustomization.captureEvidenceMode",
                customization.captureEvidenceMode.name,
            )
            field(
                "motionCustomization.cameraDynamicRangeMode",
                customization.cameraDynamicRangeMode.name,
            )
            field(
                "motionCustomization.postCaptureAutoReconstructionEnabled",
                customization.postCaptureAutoReconstructionEnabled,
            )
            field(
                "motionCustomization.recordedVideoEvidenceProfile",
                customization.recordedVideoEvidenceProfile?.toString() ?: "null",
            )
            field(
                "motionCustomization.multiPersonCapacityPolicy",
                customization.multiPersonCapacityPolicy.toString(),
            )
            field(
                "motionCustomization.faceEyeFusionStrictness",
                customization.faceEyeFusionStrictness,
            )
            field(
                "motionCustomization.faceEyeIpdDriftToleranceMillimeters",
                customization.faceEyeIpdDriftToleranceMillimeters,
            )
            field(
                "motionCustomization.motionFrequencyBandSensitivity",
                customization.motionFrequencyBandSensitivity,
            )
            field(
                "motionCustomization.distributedGlobalCameraEnvelope",
                customization.distributedGlobalCameraEnvelope,
            )
            field(
                "motionCustomization.distributedAuthoritativeCameraBudget",
                customization.distributedAuthoritativeCameraBudget,
            )
            field(
                "motionCustomization.distributedMaximumShards",
                customization.distributedMaximumShards,
            )
            field(
                "motionCustomization.distributedMinimumAuthorityHealth",
                customization.distributedMinimumAuthorityHealth,
            )
            field(
                "motionCustomization.distributedMinimumAuthoritySynchronization",
                customization.distributedMinimumAuthoritySynchronization,
            )
            field(
                "motionCustomization.distributedMaximumTemporalResidualMs",
                customization.distributedMaximumTemporalResidualMs,
            )
            field(
                "motionCustomization.distributedMaximumTemporalOutlierRatio",
                customization.distributedMaximumTemporalOutlierRatio,
            )
            field("captureProfile", captureProfile?.toString() ?: "null")

            if (calibration == null) {
                field("calibration", "null")
            } else {
                field("calibration.reprojectionErrorPx", calibration.reprojectionErrorPx)
                field(
                    "calibration.method",
                    calibration.calibrationMethod,
                )
                field(
                    "calibration.nativeIntrinsicRefinementApplied",
                    calibration.nativeIntrinsicRefinementApplied,
                )
                field(
                    "calibration.nativeIntrinsicRefinementConfidence",
                    calibration.nativeIntrinsicRefinementConfidence,
                )
                field(
                    "calibration.nativeIntrinsicRefinementImprovementPx",
                    calibration.nativeIntrinsicRefinementImprovementPx,
                )
                field(
                    "calibration.timingAuthorityConfidence",
                    calibration.timingAuthorityConfidence,
                )
                field(
                    "calibration.timingAuthorityResidualP95Milliseconds",
                    calibration.timingAuthorityResidualP95Milliseconds,
                )
                field(
                    "calibration.calibratedAtNanos",
                    calibration.calibratedAtNanos,
                )

                calibration.calibrationFocusDistanceDiopters.entries
                    .sortedBy { it.key.value }
                    .forEach { (deviceId, focusDiopters) ->
                        field(
                            "calibration.device.focusDiopters",
                            deviceId.value +
                                ":" +
                                stable(focusDiopters),
                        )
                    }

                calibration.intrinsics.keys
                    .sortedBy { it.value }
                    .forEach { deviceId ->
                        val intrinsics = calibration.intrinsics[deviceId]
                            ?: return@forEach
                        val extrinsics = calibration.extrinsics[deviceId]

                        field("calibration.device.id", deviceId.value)
                        field("calibration.device.fx", intrinsics.fx)
                        field("calibration.device.fy", intrinsics.fy)
                        field("calibration.device.cx", intrinsics.cx)
                        field("calibration.device.cy", intrinsics.cy)
                        field("calibration.device.k1", intrinsics.k1)
                        field("calibration.device.k2", intrinsics.k2)
                        field("calibration.device.p1", intrinsics.p1)
                        field("calibration.device.p2", intrinsics.p2)
                        field("calibration.device.k3", intrinsics.k3)
                        field("calibration.device.imageWidth", intrinsics.imageWidth)
                        field("calibration.device.imageHeight", intrinsics.imageHeight)

                        if (extrinsics == null) {
                            field("calibration.device.extrinsics", "null")
                        } else {
                            field(
                                "calibration.device.position",
                                listOf(
                                    extrinsics.pose.position.x,
                                    extrinsics.pose.position.y,
                                    extrinsics.pose.position.z,
                                ).joinToString(",") { stable(it) },
                            )
                            field(
                                "calibration.device.rotation",
                                listOf(
                                    extrinsics.pose.rotation.w,
                                    extrinsics.pose.rotation.x,
                                    extrinsics.pose.rotation.y,
                                    extrinsics.pose.rotation.z,
                                ).joinToString(",") { stable(it) },
                            )
                            field(
                                "calibration.device.reference",
                                extrinsics.referenceDevice?.value ?: "null",
                            )
                        }
                    }
            }

            devices
                .sortedBy { it.id.value }
                .forEach { device ->
                    field("device.id", device.id.value)
                    field("device.role", device.role.name)
                    field("device.name", device.displayName)
                    field("device.model", device.capabilities.model)
                    field("device.manufacturer", device.capabilities.manufacturer)
                    field("device.androidApi", device.capabilities.androidApi)
                    field(
                        "device.cameraIds",
                        device.capabilities.cameraIds.sorted().joinToString(","),
                    )
                    field("device.supportsRaw", device.capabilities.supportsRaw)
                    field(
                        "device.supportsConcurrentCamera",
                        device.capabilities.supportsConcurrentCamera,
                    )
                    field(
                        "device.supportsGyroscope",
                        device.capabilities.supportsGyroscope,
                    )
                    field(
                        "device.supportsAccelerometer",
                        device.capabilities.supportsAccelerometer,
                    )
                    field(
                        "device.maxAnalysis",
                        device.capabilities.maxAnalysisWidth.toString() +
                            "x" +
                            device.capabilities.maxAnalysisHeight,
                    )
                    field("device.maxFps", device.capabilities.maxFps)
                    field(
                        "device.supportsHardwareReadoutTimestamp",
                        device.capabilities.supportsHardwareReadoutTimestamp,
                    )
                    field(
                        "device.maximumVerifiedConcurrentCameras",
                        device.capabilities.maximumVerifiedConcurrentCameras,
                    )
                    field(
                        "device.minimumExposureTimeNanos",
                        device.capabilities.minimumExposureTimeNanos,
                    )
                    field(
                        "device.maximumExposureTimeNanos",
                        device.capabilities.maximumExposureTimeNanos,
                    )
                }
        }

        return "NXCFG6-" + sha256(canonical)
    }

    private fun StringBuilder.field(
        name: String,
        value: Any?,
    ) {
        val encoded = value?.toString() ?: "null"
        append(name.length)
            .append(':')
            .append(name)
            .append('=')
            .append(encoded.length)
            .append(':')
            .append(encoded)
            .append('|')
    }

    private fun sha256(value: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(StandardCharsets.UTF_8))
        return digest.joinToString("") {
            "%02x".format(Locale.US, it.toInt() and 0xFF)
        }
    }

    private fun stable(value: Double): String =
        if (value == 0.0) {
            "0.0"
        } else {
            "%.12f".format(Locale.US, value)
        }
}
