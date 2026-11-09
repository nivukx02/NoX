package com.nivukx.nox.core.model

/**
 * Production capture and reconstruction contract models.
 *
 * These models deliberately contain capability/fallback information instead of pretending that
 * every Android camera exposes the same sensor pipeline. A profile is a request; the capture
 * controller resolves it against real Camera2 capabilities before a session is armed.
 */

enum class DepthSource {
    NONE,
    DEPTH16,
    DEPTH_JPEG,
    NETWORK_COMPRESSED,
    STEREO_TRIANGULATION,
    MULTI_VIEW,
}

enum class GpuBackend {
    CPU,
    OPENGL_ES_COMPUTE,
    VULKAN_COMPUTE,
}

data class CameraCaptureProfile(
    val analysisWidth: Int = 1920,
    val analysisHeight: Int = 1080,
    val fps: Int = 30,
    val dynamicRangeMode: CameraDynamicRangeMode =
        CameraDynamicRangeMode.STANDARD,
    val dynamicRangeProfile: Long? = null,
    val exposureMode: CameraExposureMode = CameraExposureMode.AUTO,
    val antiBandingMode: CameraAntibandingMode =
        CameraAntibandingMode.AUTO,
    val focusMode: CameraFocusMode = CameraFocusMode.TRACKED_SUBJECT,
    val enableRawSidecar: Boolean = false,
    val enableDepthSidecar: Boolean = false,
    val enableGyroscope: Boolean = true,
    val enableAccelerometer: Boolean = true,
    val preferredGpu: GpuBackend = GpuBackend.OPENGL_ES_COMPUTE,
    val manualExposureNanos: Long? = null,
    val manualIso: Int? = null,
    val oisEnabled: Boolean = false,
    val electronicStabilizationEnabled: Boolean = false,
    val preserveSensorTimestamp: Boolean = true,
    val exposureCompensationEv: Double = 0.0,
    val autoExposureLock: Boolean = false,
    val autoWhiteBalanceLock: Boolean = false,
    val preciseColorTemperatureEnabled: Boolean = false,
    val colorTemperatureKelvin: Int = 5000,
    val colorTint: Int = 0,
    val manualFocusDistanceDiopters: Double? = null,
) {
    init {
        require(analysisWidth >= 320)
        require(analysisHeight >= 240)
        require(fps in 15..240)
        dynamicRangeProfile?.let {
            require(it > 0L)
        }
        manualExposureNanos?.let { require(it > 0L) }
        manualIso?.let { require(it > 0) }
        if (exposureMode == CameraExposureMode.MANUAL) {
            require(manualExposureNanos != null)
            require(manualIso != null)
        }
        if (exposureMode == CameraExposureMode.SENSITIVITY_PRIORITY) {
            require(manualIso != null)
        }

        require(exposureCompensationEv in -8.0..8.0)
        require(colorTemperatureKelvin in 2000..10000)
        require(colorTint in -100..100)
        manualFocusDistanceDiopters?.let { require(it.isFinite() && it >= 0.0) }
    }

    companion object {
        fun production(): CameraCaptureProfile = CameraCaptureProfile()

        fun adaptive(
            requestedWidth: Int,
            requestedHeight: Int,
            requestedFps: Int,
        ): CameraCaptureProfile =
            CameraCaptureProfile(
                analysisWidth = requestedWidth.coerceAtLeast(320),
                analysisHeight = requestedHeight.coerceAtLeast(240),
                fps = requestedFps.coerceIn(15, 120),
            )
    }
}

data class CameraFeatureMatrix(
    val cameraId: String,
    val supportedDynamicRangeProfiles: Set<Long> = emptySet(),
    val recommendedDynamicRangeProfile: Long? = null,
    val supports10BitDynamicRange: Boolean = false,
    /** Android 16+ Camera2 hybrid AE priority modes advertised by the camera. */
    val supportedAePriorityModes: Set<Int> = emptySet(),
    /** Android 16+ CCT white-balance support and range. */
    val supportsPreciseColorTemperature: Boolean = false,
    val minimumColorTemperatureKelvin: Int = 2000,
    val maximumColorTemperatureKelvin: Int = 10000,
    val minimumColorTint: Int = -100,
    val maximumColorTint: Int = 100,
    val minimumExposureTimeNanos: Long = 0L,
    val maximumExposureTimeNanos: Long = 0L,
    val sensorPhysicalWidthMillimeters: Double = 0.0,
    val sensorPhysicalHeightMillimeters: Double = 0.0,
    val sensorTimestampSource: Int? = null,
    val supportsRaw: Boolean,
    val rawWidth: Int,
    val rawHeight: Int,
    val supportsDepth16: Boolean,
    val depthWidth: Int,
    val depthHeight: Int,
    val supportsManualSensor: Boolean,
    val supportsOis: Boolean,
    val supportsPdaf: Boolean,
    val supportsConcurrentCamera: Boolean,
    val maxAnalysisWidth: Int,
    val maxAnalysisHeight: Int,
    val maxFps: Int,
    /** Measured maximum FPS keyed by exact YUV analysis resolution (for example 1920x1080). */
    val maximumFpsByResolution: Map<String, Int> = emptyMap(),
    val recommendedProfile: CameraCaptureProfile,
    /**
     * API 34+ hardware readout timestamp support. This is distinct from rolling-shutter skew.
     */
    val supportsHardwareReadoutTimestamp: Boolean = false,
    /**
     * Static capability placeholder for rolling-shutter skew.
     *
     * Camera2 does not expose a reliable per-mode static skew value here. The authoritative value
     * is captured per frame from CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW and propagated through
     * FrameTelemetry/LandmarkObservation. A zero value therefore means "no static evidence", not
     * "this sensor is proven global-shutter".
     */
    val rollingShutterSkewNanos: Long = 0L,
    /**
     * Platform-declared concurrent camera sets containing this camera.
     */
    val concurrentCameraSets: List<Set<String>> = emptyList(),
    /**
     * True when the camera advertises the Camera2 logical-multi-camera capability.
     */
    val supportsLogicalMultiCamera: Boolean = false,
    /**
     * Physical child camera IDs exposed by a logical multi-camera device.
     */
    val physicalCameraIds: Set<String> = emptySet(),
    /**
     * Camera2 logical-camera sensor synchronization type when available.
     */
    val logicalSensorSyncType: Int? = null,
) {
    init {
        require(minimumExposureTimeNanos >= 0L)
        require(maximumExposureTimeNanos >= minimumExposureTimeNanos)
        require(
            sensorPhysicalWidthMillimeters.isFinite() &&
                sensorPhysicalWidthMillimeters >= 0.0
        )
        require(
            sensorPhysicalHeightMillimeters.isFinite() &&
                sensorPhysicalHeightMillimeters >= 0.0
        )
    }
}

data class FrameTelemetry(
    val sensorTimestampNanos: Long,
    val monotonicTimestampNanos: Long,
    val exposureTimeNanos: Long,
    val sensorSensitivityIso: Int,
    val frameDurationNanos: Long,
    val rollingShutterSkewNanos: Long,
    val aeState: Int?,
    val afState: Int?,
    val awbState: Int?,
    val lensFocusDistanceDiopters: Float?,
    val focusDistanceMeters: Float?,
    /** API 34+ hardware start-of-readout timestamp for the same capture frame, when available. */
    val readoutTimestampNanos: Long? = null,
    val sensorTemperatureCelsius: Float? = null,
)

data class LightingEstimate(
    val timestampNanos: Long,
    val illuminanceLux: Double,
    val colorTemperatureKelvin: Double,
    val exposureCompensationEv: Double,
    val ambientLuminance: Double,
    val directionalLight: Vec3,
    val directionalIntensity: Double,
    val confidence: Double,
    val dynamicRangeStops: Double,
)

data class EnvironmentLighting(
    val timestampNanos: Long,
    val ambient: Vec3,
    val dominantDirection: Vec3,
    val dominantIntensity: Double,
    val colorTemperatureKelvin: Double,
    val exposureValue: Double,
    val confidence: Double,
)

data class DepthObservation(
    val x: Int,
    val y: Int,
    val depthMeters: Double,
    val confidence: Double,
)

data class DepthFrameDescriptor(
    val timestampNanos: Long,
    val source: DepthSource,
    val width: Int,
    val height: Int,
    val minDepthMeters: Double,
    val maxDepthMeters: Double,
    val validSamples: Int,
    val meanConfidence: Double,
)

data class DepthFrame(
    val timestampNanos: Long,
    val source: DepthSource,
    val width: Int,
    val height: Int,
    val depthMeters: FloatArray,
    val confidence: FloatArray? = null,
) {
    init {
        require(width > 0)
        require(height > 0)
        // Int multiplication can wrap (for example, 65,536 * 65,536 becomes zero).
        // Validate sample count in Long before comparing against JVM array capacity.
        val expectedSamples = width.toLong() * height.toLong()
        require(expectedSamples <= Int.MAX_VALUE.toLong()) {
            "Depth frame exceeds JVM array sample capacity"
        }
        require(depthMeters.size.toLong() == expectedSamples)
        confidence?.let {
            require(it.size.toLong() == expectedSamples)
        }
    }

    fun descriptor(): DepthFrameDescriptor {
        var valid = 0
        var minDepth = Double.POSITIVE_INFINITY
        var maxDepth = 0.0
        var confidenceSum = 0.0

        depthMeters.forEachIndexed { index, value ->
            if (value.isFinite() && value > 0f) {
                valid += 1
                minDepth = minOf(minDepth, value.toDouble())
                maxDepth = maxOf(maxDepth, value.toDouble())
                val sampleConfidence =
                    confidence?.get(index)?.toDouble()
                        ?.takeIf { it.isFinite() }
                        ?.coerceIn(0.0, 1.0)
                        ?: if (confidence == null) 1.0 else 0.0
                confidenceSum += sampleConfidence
            }
        }

        return DepthFrameDescriptor(
            timestampNanos = timestampNanos,
            source = source,
            width = width,
            height = height,
            minDepthMeters =
                if (minDepth.isFinite()) minDepth else 0.0,
            maxDepthMeters = maxDepth,
            validSamples = valid,
            meanConfidence =
                if (valid == 0) 0.0
                else confidenceSum / valid,
        )
    }
}

data class ImuSample(
    val timestampNanos: Long,
    val sensorType: Int = 0,
    val x: Double,
    val y: Double,
    val z: Double,
    val accuracy: Int = 0,
)

data class VioPose(
    val timestampNanos: Long,
    val positionMeters: Vec3,
    val rotation: Quat,
    val linearVelocityMetersPerSecond: Vec3,
    val angularVelocityRadiansPerSecond: Vec3,
    val covarianceTrace: Double,
    val confidence: Double,
)

data class ReconstructionSettings(
    val voxelSizeMeters: Double = 0.0125,
    val truncationDistanceMeters: Double = 0.05,
    val maxVoxelCount: Int = 1_000_000,
    val maxCameras: Int = 128,
    val maxFramesInWindow: Int = 12,
    val maxReprojectionErrorPx: Double = 3.5,
    val minimumParallaxDegrees: Double = 1.0,
    val temporalHalfLifeSeconds: Double = 0.09,
)



enum class CameraResolutionPreset {
    AUTO,
    HD_720P,
    FULL_HD_1080P,
    UHD_4K,
    MAX_SUPPORTED,
}

enum class CameraExposureMode {
    AUTO,
    MANUAL,
    SENSITIVITY_PRIORITY,
}

enum class CameraDynamicRangeMode {
    STANDARD,
    TEN_BIT_PREFERRED,
}

enum class CameraAntibandingMode {
    AUTO,
    HZ_50,
    HZ_60,
    OFF,
}

enum class CameraFocusMode {
    CONTINUOUS,
    TRACKED_SUBJECT,
    MANUAL,
}

enum class ThermalState {
    NOMINAL,
    LIGHT,
    MODERATE,
    SEVERE,
    CRITICAL,
    UNKNOWN,
}

data class SyncQuality(
    val offsetNanos: Long,
    val roundTripNanos: Long,
    val jitterNanos: Long,
    val driftPpm: Double,
    val sampleCount: Int,
    val confidence: Double,
)

data class MotionTelemetry(
    val timestampNanos: Long,
    val motionScore: Double,
    val speed: Double,
    val acceleration: Double,
    val jerk: Double,
    val jitter: Double,
    val occlusionRatio: Double,
    val trackCount: Int,
    val confidence: Double,
)

data class ReconstructionTelemetry(
    val timestampNanos: Long,
    val joints: Int,
    val cameras: Int,
    val meanReprojectionPx: Double,
    val medianReprojectionPx: Double,
    val meanUncertaintyMeters: Double,
    val confidence: Double,
    val meshVertices: Int,
    val meshTriangles: Int,
    val meshConfidence: Double,
)

data class PhotometricCalibration(
    val deviceId: DeviceId,
    val scaleLux: Double,
    val blackLevel: Double = 0.0,
    val whiteLevel: Double = 1.0,
    val referenceTemperatureCelsius: Double = 25.0,
    val temperatureCoefficient: Double = 0.0,
    val calibratedAtNanos: Long,
) {
    init {
        require(scaleLux > 0.0)
        require(blackLevel.isFinite())
        require(whiteLevel.isFinite() && whiteLevel > blackLevel)
        require(referenceTemperatureCelsius.isFinite())
        require(temperatureCoefficient.isFinite())
        require(calibratedAtNanos >= 0L)
    }
}

data class PipelineTelemetry(
    val timestampNanos: Long,
    val captureFps: Double,
    val inferenceFps: Double,
    val inferenceLatencyMillis: Double,
    val droppedInferenceFrames: Long,
    val trackedSubjects: Int,
    val faceTracks: Int,
    val eyeTracks: Int,
    val triangulatedJoints: Int,
    val reconstructedVertices: Int,
    val reprojectionErrorPx: Double,
    val clockSkewNanos: Long,
    val thermalStatus: Int,
    val memoryPressure: Double,
    val gpuBackend: GpuBackend,
    val healthy: Boolean,
    val warnings: List<String> = emptyList(),
)

data class DenseMeshQuality(
    val timestampNanos: Long,
    val vertexCount: Int,
    val triangleCount: Int,
    val meanReprojectionErrorPx: Double,
    val watertightnessScore: Double,
    val temporalStability: Double,
    val geometricConfidence: Double,
    val normalCoverage: Double = 0.0,
    val topologyIntegrityScore: Double = 0.0,
    val boundaryEdgeCount: Int = 0,
    val nonManifoldEdgeCount: Int = 0,
    val duplicateTrianglesRemoved: Int = 0,
    val meanUncertaintyMeters: Double = 0.0,
    val reprojectionEvidenceAvailable: Boolean = false,
    val watertightnessEvidenceAvailable: Boolean = false,
    val temporalEvidenceAvailable: Boolean = false,
) {
    init {
        require(vertexCount >= 0)
        require(triangleCount >= 0)
        require(meanReprojectionErrorPx.isFinite() && meanReprojectionErrorPx >= 0.0)
        require(watertightnessScore.isFinite() && watertightnessScore in 0.0..1.0)
        require(temporalStability.isFinite() && temporalStability in 0.0..1.0)
        require(geometricConfidence.isFinite() && geometricConfidence in 0.0..1.0)
        require(normalCoverage.isFinite() && normalCoverage in 0.0..1.0)
        require(topologyIntegrityScore.isFinite() && topologyIntegrityScore in 0.0..1.0)
        require(boundaryEdgeCount >= 0)
        require(nonManifoldEdgeCount >= 0)
        require(duplicateTrianglesRemoved >= 0)
        require(meanUncertaintyMeters.isFinite() && meanUncertaintyMeters >= 0.0)
    }
}

data class RollingShutterModel(
    val readoutTimeNanos: Long,
    val exposureTimeNanos: Long,
    val directionSign: Int = 1,
) {
    init {
        require(readoutTimeNanos >= 0L)
        require(exposureTimeNanos >= 0L)
        require(directionSign == 1 || directionSign == -1)
    }

    fun rowTimestamp(
        frameTimestampNanos: Long,
        normalizedRow: Double,
    ): Long {
        val row = normalizedRow.coerceIn(0.0, 1.0)
        val centerOffset = (row - 0.5) * readoutTimeNanos * directionSign
        return frameTimestampNanos + centerOffset.toLong()
    }
}
