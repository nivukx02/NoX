package com.nivukx.nox.core.model

/**
 * A single camera's recorded visual evidence within a multi-camera session.
 *
 * Paths are local or materialized artifacts. They are never assumed to be network-accessible.
 * Timestamp bounds are source-clock timestamps and are compared against the session configuration
 * fingerprint before a bundle is admitted to reconstruction.
 */
data class HybridCameraEvidenceRef(
    val sessionId: String,
    val deviceId: DeviceId,
    val cameraId: String,
    val videoPath: String,
    val width: Int,
    val height: Int,
    val fps: Double,
    val firstTimestampNanos: Long,
    val lastTimestampNanos: Long,
    val frameCount: Long,
    val configurationFingerprint: String,
    val calibrationAvailable: Boolean,
    val clockOffsetNanos: Long = 0L,
    val clockDriftPpm: Double = 0.0,
    val synchronizationConfidence: Double = 1.0,
    val rollingShutterSkewNanos: Long = 0L,
    val sensorTimestampDomain: String = "ANDROID_SENSOR",
    val notes: List<String> = emptyList(),
) {
    init {
        require(sessionId.isNotBlank())
        require(cameraId.isNotBlank())
        require(videoPath.isNotBlank())
        require(width > 0)
        require(height > 0)
        require(fps > 0.0 && fps <= 240.0)
        require(firstTimestampNanos >= 0L)
        require(lastTimestampNanos >= firstTimestampNanos)
        require(frameCount >= 0L)
        require(configurationFingerprint.isNotBlank())
        require(clockDriftPpm.isFinite())
        require(synchronizationConfidence in 0.0..1.0)
        require(rollingShutterSkewNanos >= 0L)
        require(sensorTimestampDomain.isNotBlank())
    }

    val durationNanos: Long
        get() = lastTimestampNanos - firstTimestampNanos

    val averageFrameIntervalNanos: Long
        get() =
            if (frameCount <= 1L) {
                0L
            } else {
                (durationNanos / (frameCount - 1L).coerceAtLeast(1L))
            }
}

/**
 * A deterministic container for multiple recorded cameras participating in the same reconstruction
 * solve. The bundle is intentionally stricter than a list of video paths.
 */
data class HybridEvidenceBundleManifest(
    val sessionId: String,
    val createdAtEpochMillis: Long,
    val cameras: List<HybridCameraEvidenceRef>,
    val captureEvidenceMode: CaptureEvidenceMode,
    val reconstructionPolicy: PostCaptureReconstructionPolicy,
    val capacityPolicy: MultiPersonCapacityPolicy,
    val sourceMotionArtifactPath: String?,
    val bundleConfigurationFingerprint: String,
    val depthEvidencePaths: List<String> = emptyList(),
    val synchronizationConfidence: Double =
        cameras.minOf { it.synchronizationConfidence },
    val warnings: List<String> = emptyList(),
) {
    init {
        require(sessionId.isNotBlank())
        require(createdAtEpochMillis > 0L)
        require(cameras.isNotEmpty())
        require(bundleConfigurationFingerprint.isNotBlank())
        require(
            cameras.distinctBy { it.deviceId }.size == cameras.size,
        )
        require(
            cameras.all { it.sessionId == sessionId },
        )
        require(
            cameras.all {
                it.configurationFingerprint ==
                    cameras.first().configurationFingerprint
            },
        )
        // A bundle may represent a distributed capture fabric. Keep the model envelope aligned
        // with the 128-camera rig authority, while downstream reconstruction remains independently bounded.
        require(
            cameras.size <= 128,
        )
        require(synchronizationConfidence in 0.0..1.0)
    }

    val cameraCount: Int
        get() = cameras.size

    val firstTimestampNanos: Long
        get() = cameras.minOf { it.firstTimestampNanos }

    val lastTimestampNanos: Long
        get() = cameras.maxOf { it.lastTimestampNanos }

    val durationNanos: Long
        get() = lastTimestampNanos - firstTimestampNanos

    val calibratedCameraCount: Int
        get() = cameras.count { it.calibrationAvailable }

    val calibrationCoverage: Double
        get() =
            if (cameraCount == 0) 0.0
            else calibratedCameraCount.toDouble() / cameraCount.toDouble()

    val uniqueConfigurationFingerprint: Boolean
        get() =
            cameras.map { it.configurationFingerprint }.distinct().size == 1
}

/**
 * Bundle admission result.
 *
 * A bundle can be structurally valid and still be unsuitable for high-precision reconstruction.
 * Those concerns remain separate so operators can see whether failure came from missing evidence,
 * mismatched configuration, timing, calibration or capacity.
 */
data class HybridEvidenceBundleAdmission(
    val accepted: Boolean,
    val sessionId: String,
    val cameraCount: Int,
    val calibratedCameraCount: Int,
    val commonConfigurationFingerprint: String?,
    val temporalOverlapRatio: Double,
    val calibrationCoverage: Double,
    val reasons: List<String>,
    val warnings: List<String>,
) {
    init {
        require(sessionId.isNotBlank())
        require(cameraCount >= 0)
        require(calibratedCameraCount in 0..cameraCount)
        require(temporalOverlapRatio in 0.0..1.0)
        require(calibrationCoverage in 0.0..1.0)
    }
}

/**
 * Pure admission validator for a recorded multi-camera bundle.
 *
 * The validator intentionally does not read video bytes or perform reconstruction. It checks the
 * contract that can be proven from metadata before expensive reconstruction begins.
 */
object HybridEvidenceBundleValidator {

    private const val MAX_BUNDLE_CAMERAS = 128

    fun validate(
        sessionId: String,
        cameras: List<HybridCameraEvidenceRef>,
        capacityPolicy: MultiPersonCapacityPolicy =
            MultiPersonCapacityPolicy.fiftyPersonProduction(),
        minimumTemporalOverlapRatio: Double = 0.72,
        minimumCalibrationCoverage: Double = 0.50,
        minimumSynchronizationConfidence: Double = 0.85,
    ): HybridEvidenceBundleAdmission {
        require(sessionId.isNotBlank())
        require(minimumTemporalOverlapRatio in 0.0..1.0)
        require(minimumCalibrationCoverage in 0.0..1.0)
        require(minimumSynchronizationConfidence in 0.0..1.0)

        val reasons = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        val sessionMatches =
            cameras.all { it.sessionId == sessionId }

        if (!sessionMatches) {
            reasons += "SESSION_ID_MISMATCH"
        }

        val duplicateDevices =
            cameras.groupBy { it.deviceId }
                .filterValues { it.size > 1 }
                .keys

        if (duplicateDevices.isNotEmpty()) {
            reasons +=
                "DUPLICATE_DEVICE_EVIDENCE:" +
                    duplicateDevices.joinToString(",") { it.value }
        }

        if (cameras.isEmpty()) {
            reasons += "NO_CAMERA_EVIDENCE"
        }

        if (cameras.size > MAX_BUNDLE_CAMERAS) {
            reasons +=
                "CAMERA_BUNDLE_EXCEEDS_128_CAMERA_ENVELOPE"
        }

        val distinctFingerprints =
            cameras
                .map { it.configurationFingerprint }
                .distinct()

        if (distinctFingerprints.size > 1) {
            reasons += "CONFIGURATION_FINGERPRINT_MISMATCH"
        }

        val first =
            cameras.minOfOrNull { it.firstTimestampNanos } ?: 0L
        val last =
            cameras.maxOfOrNull { it.lastTimestampNanos } ?: 0L

        val overlapStart =
            cameras.maxOfOrNull { it.firstTimestampNanos } ?: 0L
        val overlapEnd =
            cameras.minOfOrNull { it.lastTimestampNanos } ?: 0L

        val unionDuration =
            (last - first).coerceAtLeast(0L)
        val overlapDuration =
            (overlapEnd - overlapStart).coerceAtLeast(0L)

        val overlapRatio =
            if (unionDuration <= 0L) {
                if (cameras.size <= 1) 1.0 else 0.0
            } else {
                (
                    overlapDuration.toDouble() /
                        unionDuration.toDouble()
                    ).coerceIn(0.0, 1.0)
            }

        if (overlapRatio < minimumTemporalOverlapRatio) {
            reasons += "INSUFFICIENT_TEMPORAL_OVERLAP"
        }

        val calibratedCount =
            cameras.count { it.calibrationAvailable }
        val calibrationCoverage =
            if (cameras.isEmpty()) {
                0.0
            } else {
                calibratedCount.toDouble() / cameras.size.toDouble()
            }

        if (calibrationCoverage < minimumCalibrationCoverage) {
            reasons += "INSUFFICIENT_CALIBRATION_COVERAGE"
        }

        val synchronizationConfidence =
            cameras.minOfOrNull {
                it.synchronizationConfidence
            } ?: 0.0

        if (
            synchronizationConfidence <
                minimumSynchronizationConfidence
        ) {
            reasons += "INSUFFICIENT_SYNCHRONIZATION_CONFIDENCE"
        }

        val maxFrameCount =
            cameras.maxOfOrNull { it.frameCount } ?: 0L
        if (maxFrameCount == 0L) {
            warnings += "NO_RECORDED_FRAME_COUNT_AVAILABLE"
        }

        val accepted = reasons.isEmpty()

        if (
            accepted &&
            cameras.size < 2
        ) {
            warnings +=
                "SINGLE_CAMERA_BUNDLE: calibrated metric 3D evidence depends on the available camera model and geometry."
        }

        return HybridEvidenceBundleAdmission(
            accepted = accepted,
            sessionId = sessionId,
            cameraCount = cameras.size,
            calibratedCameraCount = calibratedCount,
            commonConfigurationFingerprint =
                distinctFingerprints.singleOrNull(),
            temporalOverlapRatio = overlapRatio,
            calibrationCoverage = calibrationCoverage,
            reasons = reasons.distinct(),
            warnings = warnings.distinct(),
        )
    }
}
