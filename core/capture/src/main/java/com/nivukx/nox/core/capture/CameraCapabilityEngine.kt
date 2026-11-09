package com.nivukx.nox.core.capture

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.util.Size
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.CameraFeatureMatrix
import kotlin.math.abs

/**
 * Resolves the camera capabilities required by the production motion pipeline.
 *
 * Dynamic-range support is reported separately from general RAW/depth capability because Android
 * exposes 10-bit/HDR negotiation as an explicit camera/session contract. The engine does not claim
 * 10-bit support merely because a device advertises RAW or a large YUV stream.
 */
class CameraCapabilityEngine(context: Context) {
    private val manager = context.applicationContext
        .getSystemService(Context.CAMERA_SERVICE) as CameraManager

    fun inspect(cameraId: String): CameraFeatureMatrix {
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val streamMap = characteristics.get(
            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP,
        )

        val yuvSizes = streamMap
            ?.getOutputSizes(ImageFormat.YUV_420_888)
            ?.toList()
            .orEmpty()

        val rawSizes = if (
            supportsCapability(
                characteristics,
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW,
            )
        ) {
            streamMap?.getOutputSizes(ImageFormat.RAW_SENSOR)
                ?.toList()
                .orEmpty()
        } else {
            emptyList()
        }

        val depthSizes = if (
            supportsCapability(
                characteristics,
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT,
            )
        ) {
            if (Build.VERSION.SDK_INT >= 23) {
                streamMap?.getOutputSizes(ImageFormat.DEPTH16)
                    ?.toList()
                    .orEmpty()
            } else {
                emptyList()
            }
        } else {
            emptyList()
        }

        val fpsRanges = characteristics
            .get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?.toList()
            .orEmpty()

        val maxFps = fpsRanges.maxOfOrNull { it.upper } ?: 30
        val supportedAePriorityModes =
            if (Build.VERSION.SDK_INT >= 36) {
                characteristics.get(
                    CameraCharacteristics.CONTROL_AE_AVAILABLE_PRIORITY_MODES,
                )?.toSet().orEmpty()
            } else {
                emptySet()
            }

        val colorTemperatureRange =
            if (Build.VERSION.SDK_INT >= 36) {
                characteristics.get(
                    CameraCharacteristics.COLOR_CORRECTION_COLOR_TEMPERATURE_RANGE,
                )
            } else {
                null
            }

        val analysisCandidates =
            yuvSizes.mapNotNull { size ->
                val durationNanos =
                    runCatching {
                        streamMap?.getOutputMinFrameDuration(
                            ImageFormat.YUV_420_888,
                            size,
                        ) ?: 0L
                    }.getOrDefault(0L)

                val durationBoundFps =
                    if (durationNanos > 0L) {
                        (
                            1_000_000_000.0 /
                                durationNanos.toDouble()
                        ).toInt()
                    } else {
                        maxFps
                    }

                CameraMotionModeOptimizer.Candidate(
                    width = size.width,
                    height = size.height,
                    maximumFps = durationBoundFps.coerceAtLeast(1),
                )
            }

        val analysisMode =
            CameraMotionModeOptimizer().select(
                candidates = analysisCandidates,
                advertisedMaximumFps = maxFps,
            )

        val analysisWidth =
            analysisMode?.width
                ?: minSize(yuvSizes, 1920, 1080).width
        val analysisHeight =
            analysisMode?.height
                ?: minSize(yuvSizes, 1920, 1080).height
        val analysisFps =
            analysisMode?.fps
                ?: maxFps.coerceIn(15, 240)

        val rawSize = rawSizes.maxByOrNull(::pixels)
        val depthSize = depthSizes.maxByOrNull(::pixels)
        val supportsManual = supportsCapability(
            characteristics,
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR,
        )

        val exposureRange =
            characteristics.get(
                CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE,
            )
        val sensorPhysicalSize =
            characteristics.get(
                CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE,
            )
        val sensorTimestampSource =
            characteristics.get(
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE,
            )

        val oisModes = characteristics.get(
            CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION,
        ) ?: intArrayOf()

        val concurrentSets = runCatching {
            if (Build.VERSION.SDK_INT >= 30) {
                manager.getConcurrentCameraIds()
                    .filter { it.size >= 2 }
                    .map { it.toSet() }
                    .filter { cameraId in it }
            } else {
                emptyList()
            }
        }.getOrDefault(emptyList())

        val concurrent = concurrentSets.isNotEmpty()

        val supportsLogicalMultiCamera =
            Build.VERSION.SDK_INT >= 28 &&
                supportsCapability(
                    characteristics,
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA,
                )

        val physicalCameraIds =
            if (Build.VERSION.SDK_INT >= 28 && supportsLogicalMultiCamera) {
                runCatching {
                    characteristics.physicalCameraIds
                }.getOrDefault(emptySet())
            } else {
                emptySet()
            }

        val logicalSensorSyncType =
            if (Build.VERSION.SDK_INT >= 28 && supportsLogicalMultiCamera) {
                characteristics.get(
                    CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE,
                )
            } else {
                null
            }

        val supportsHardwareReadoutTimestamp =
            if (Build.VERSION.SDK_INT >= 34) {
                characteristics.get(
                    CameraCharacteristics.SENSOR_READOUT_TIMESTAMP,
                ) ==
                    CameraCharacteristics.SENSOR_READOUT_TIMESTAMP_HARDWARE
            } else {
                false
            }

        val rollingShutterSkewNanos = 0L

        val dynamicRangeProfiles =
            if (Build.VERSION.SDK_INT >= 33) {
                characteristics.get(
                    CameraCharacteristics.REQUEST_AVAILABLE_DYNAMIC_RANGE_PROFILES,
                )?.supportedProfiles.orEmpty()
            } else {
                emptySet()
            }

        val recommendedDynamicRangeProfile =
            if (Build.VERSION.SDK_INT >= 33) {
                characteristics.get(
                    CameraCharacteristics.REQUEST_RECOMMENDED_TEN_BIT_DYNAMIC_RANGE_PROFILE,
                )
            } else {
                null
            }

        val supportsTenBit =
            Build.VERSION.SDK_INT >= 33 &&
                supportsCapability(
                    characteristics,
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DYNAMIC_RANGE_TEN_BIT,
                ) &&
                dynamicRangeProfiles.any {
                    it != android.hardware.camera2.params.DynamicRangeProfiles.STANDARD
                }

        val recommended = CameraCaptureProfile(
            analysisWidth = analysisWidth,
            analysisHeight = analysisHeight,
            fps = analysisFps,
            enableRawSidecar = rawSize != null,
            enableDepthSidecar = depthSize != null,
            enableGyroscope = true,
            enableAccelerometer = true,
        )

        return CameraFeatureMatrix(
            cameraId = cameraId,
            supportsRaw = rawSize != null,
            rawWidth = rawSize?.width ?: 0,
            rawHeight = rawSize?.height ?: 0,
            supportsDepth16 = depthSize != null,
            depthWidth = depthSize?.width ?: 0,
            depthHeight = depthSize?.height ?: 0,
            supportsManualSensor = supportsManual,
            supportsOis = oisModes.contains(
                CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_ON,
            ),
            supportsPdaf = false,
            supportsConcurrentCamera = concurrent,
            maxAnalysisWidth = yuvSizes.maxByOrNull(::pixels)?.width ?: 1920,
            maxAnalysisHeight = yuvSizes.maxByOrNull(::pixels)?.height ?: 1080,
            maxFps = maxFps,
            maximumFpsByResolution = maximumFpsByResolution,
            supportedDynamicRangeProfiles = dynamicRangeProfiles,
            recommendedDynamicRangeProfile = recommendedDynamicRangeProfile,
            supports10BitDynamicRange = supportsTenBit,
            supportedAePriorityModes = supportedAePriorityModes,
            supportsPreciseColorTemperature =
                colorTemperatureRange != null,
            minimumColorTemperatureKelvin =
                colorTemperatureRange?.lower ?: 2000,
            maximumColorTemperatureKelvin =
                colorTemperatureRange?.upper ?: 10000,
            minimumColorTint = -100,
            maximumColorTint = 100,
            recommendedProfile = recommended,
            supportsHardwareReadoutTimestamp = supportsHardwareReadoutTimestamp,
            minimumExposureTimeNanos =
                exposureRange?.lower ?: 0L,
            maximumExposureTimeNanos =
                exposureRange?.upper ?: 0L,
            sensorPhysicalWidthMillimeters =
                sensorPhysicalSize?.width?.toDouble() ?: 0.0,
            sensorPhysicalHeightMillimeters =
                sensorPhysicalSize?.height?.toDouble() ?: 0.0,
            sensorTimestampSource = sensorTimestampSource,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            concurrentCameraSets = concurrentSets,
            supportsLogicalMultiCamera = supportsLogicalMultiCamera,
            physicalCameraIds = physicalCameraIds,
            logicalSensorSyncType = logicalSensorSyncType,
        )
    }

    private fun supportsCapability(
        c: CameraCharacteristics,
        value: Int,
    ): Boolean =
        c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            ?.contains(value) == true

    private fun pixels(size: Size): Long =
        size.width.toLong() * size.height.toLong()

    private fun minSize(
        sizes: List<Size>,
        requestedWidth: Int,
        requestedHeight: Int,
    ): Size {
        if (sizes.isEmpty()) {
            return Size(requestedWidth, requestedHeight)
        }

        val target =
            requestedWidth.toLong() * requestedHeight.toLong()

        return sizes
            .filter {
                it.width <= requestedWidth &&
                    it.height <= requestedHeight
            }
            .maxByOrNull(::pixels)
            ?: sizes.minByOrNull {
                abs(pixels(it) - target)
            }
            ?: sizes.maxBy(::pixels)
    }
}
