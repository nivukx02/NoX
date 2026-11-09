package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CaptureEvidenceMode
import com.nivukx.nox.core.model.HybridCameraEvidenceRef
import com.nivukx.nox.core.model.HybridEvidenceBundleAdmission
import com.nivukx.nox.core.model.HybridEvidenceBundleManifest
import com.nivukx.nox.core.model.HybridEvidenceBundleValidator
import com.nivukx.nox.core.model.MultiPersonCapacityPolicy
import com.nivukx.nox.core.model.PostCaptureReconstructionPolicy
import com.nivukx.nox.core.sync.RecordedCameraClockFitEngine
import com.nivukx.nox.core.sync.VisualTimingBeaconCalibrationEngine
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

/**
 * Thread-safe assembler for multi-device recorded capture evidence.
 *
 * Camera devices can finish their recordings independently. The assembler replaces the same-device
 * record atomically, keeps every artifact associated with one session ID, and produces a deterministic
 * bundle fingerprint that can be persisted beside the session manifest.
 */
class HybridEvidenceBundleAssembler(
    private val sessionId: String,
    private val capacityPolicy: MultiPersonCapacityPolicy =
        MultiPersonCapacityPolicy.fiftyPersonProduction(),
    private val maximumRegisteredCameras: Int = 128,
) : AutoCloseable {

    init {
        require(sessionId.isNotBlank())
        require(maximumRegisteredCameras in 2..128)
    }

    private val cameras =
        ConcurrentHashMap<com.nivukx.nox.core.model.DeviceId, HybridCameraEvidenceRef>()

    @Volatile
    private var closed = false

    @Synchronized
    fun register(reference: HybridCameraEvidenceRef) {
        check(!closed) {
            "HybridEvidenceBundleAssembler is closed"
        }
        require(reference.sessionId == sessionId) {
            "Evidence session mismatch"
        }
        require(
            reference.frameCount == 0L ||
                reference.lastTimestampNanos >= reference.firstTimestampNanos,
        )
        val existing =
            cameras[reference.deviceId]
        if (
            existing == null &&
            cameras.size >= maximumRegisteredCameras
        ) {
            error(
                "Hybrid evidence bundle reached its " +
                    "$maximumRegisteredCameras-camera registration ceiling",
            )
        }
        cameras[reference.deviceId] = reference
    }

    fun fitCameraClock(
        reference: HybridCameraEvidenceRef,
        samples: List<RecordedCameraClockFitEngine.Sample>,
        minimumSamples: Int = 8,
    ): RecordedCameraClockFitEngine.FitResult {
        check(!closed)
        require(reference.sessionId == sessionId)

        val fit =
            RecordedCameraClockFitEngine().fit(
                samples = samples,
                minimumSamples = minimumSamples,
            )

        register(
            reference.copy(
                clockOffsetNanos =
                    fit.clockOffsetNanos,
                clockDriftPpm =
                    fit.clockDriftPpm,
                synchronizationConfidence =
                    fit.synchronizationConfidence,
            ),
        )

        return fit
    }

    @Synchronized
    /**
     * Fits a recorded camera clock from a visual timing beacon and stores the resulting affine
     * synchronization contract in the camera reference. This is stronger than trusting arrival
     * timestamps because exposure-center and row-dependent timing are corrected before fitting.
     */
    fun fitCameraClockFromVisualBeacon(
        reference: HybridCameraEvidenceRef,
        observations: List<VisualTimingBeaconCalibrationEngine.Observation>,
    ): VisualTimingBeaconCalibrationEngine.DeviceCalibration {
        check(!closed)
        require(reference.sessionId == sessionId)

        val resolved =
            VisualTimingBeaconCalibrationEngine()
                .calibrate(
                    observations =
                        observations.filter {
                            it.deviceId == reference.deviceId
                        },
                )
                .perDevice
                .firstOrNull()
                ?: error(
                    "Unable to construct visual timing-beacon calibration for " +
                        reference.deviceId.value +
                        ": insufficient beacon evidence",
                )

        register(
            reference.copy(
                clockOffsetNanos =
                    resolved.fit.clockOffsetNanos,
                clockDriftPpm =
                    resolved.fit.clockDriftPpm,
                synchronizationConfidence =
                    resolved.synchronizationConfidence,
                notes =
                    (
                        reference.notes +
                            "VISUAL_TIMING_BEACON" +
                            resolved.warnings.map { "BEACON:$it" }
                        )
                        .distinct()
                        .takeLast(24),
            ),
        )

        return resolved
    }

    fun unregister(
        deviceId: com.nivukx.nox.core.model.DeviceId,
    ) {
        check(!closed)
        cameras.remove(deviceId)
    }

    fun references(): List<HybridCameraEvidenceRef> =
        cameras.values
            .sortedBy {
                it.deviceId.value
            }

    fun admission(
        minimumTemporalOverlapRatio: Double = 0.72,
        minimumCalibrationCoverage: Double = 0.50,
        minimumSynchronizationConfidence: Double = 0.85,
    ): HybridEvidenceBundleAdmission =
        HybridEvidenceBundleValidator.validate(
            sessionId = sessionId,
            cameras = references(),
            capacityPolicy = capacityPolicy,
            minimumTemporalOverlapRatio =
                minimumTemporalOverlapRatio,
            minimumCalibrationCoverage =
                minimumCalibrationCoverage,
            minimumSynchronizationConfidence =
                minimumSynchronizationConfidence,
        )

    fun buildManifest(
        evidenceMode: CaptureEvidenceMode,
        reconstructionPolicy: PostCaptureReconstructionPolicy =
            PostCaptureReconstructionPolicy(),
        sourceMotionArtifactPath: String? = null,
        depthEvidencePaths: List<String> = emptyList(),
        strict: Boolean = true,
    ): HybridEvidenceBundleManifest {
        check(!closed)
        val snapshot = references()
        require(snapshot.isNotEmpty()) {
            "Cannot build an evidence bundle without camera references"
        }

        val admission = admission()
        if (strict) {
            require(admission.accepted) {
                "Evidence bundle rejected: " +
                    admission.reasons.joinToString(",")
            }
        }

        val fingerprint =
            fingerprint(
                evidenceMode = evidenceMode,
                reconstructionPolicy = reconstructionPolicy,
                cameras = snapshot,
            )

        return HybridEvidenceBundleManifest(
            sessionId = sessionId,
            createdAtEpochMillis =
                System.currentTimeMillis(),
            cameras = snapshot,
            captureEvidenceMode = evidenceMode,
            reconstructionPolicy =
                reconstructionPolicy,
            capacityPolicy = capacityPolicy,
            sourceMotionArtifactPath =
                sourceMotionArtifactPath,
            depthEvidencePaths =
                depthEvidencePaths,
            synchronizationConfidence =
                snapshot.minOf {
                    it.synchronizationConfidence
                },
            bundleConfigurationFingerprint =
                fingerprint,
            warnings =
                admission.warnings +
                    if (!strict && !admission.accepted) {
                        admission.reasons.map {
                            "NON_STRICT_BUNDLE:" + it
                        }
                    } else {
                        emptyList()
                    },
        )
    }

    fun summary(): String {
        val refs = references()
        if (refs.isEmpty()) {
            return "No camera evidence registered"
        }

        val calibrated =
            refs.count {
                it.calibrationAvailable
            }
        val synchronization =
            refs.minOf {
                it.synchronizationConfidence
            }

        return buildString {
            append(refs.size)
            append(" cameras · ")
            append(calibrated)
            append(" calibrated · sync ")
            append(
                "%.3f".format(
                    java.util.Locale.US,
                    synchronization,
                ),
            )
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        cameras.clear()
    }

    private fun fingerprint(
        evidenceMode: CaptureEvidenceMode,
        reconstructionPolicy: PostCaptureReconstructionPolicy,
        cameras: List<HybridCameraEvidenceRef>,
    ): String {
        val canonical =
            buildString {
                append("nox-hybrid-bundle-v1|")
                append(sessionId)
                append('|')
                append(evidenceMode.name)
                append('|')
                append(reconstructionPolicy.toString())
                cameras
                    .sortedBy {
                        it.deviceId.value
                    }
                    .forEach { camera ->
                        append('|')
                        append(camera.deviceId.value)
                        append('|')
                        append(camera.cameraId)
                        append('|')
                        append(camera.width)
                        append('x')
                        append(camera.height)
                        append('|')
                        append(camera.fps)
                        append('|')
                        append(camera.firstTimestampNanos)
                        append('|')
                        append(camera.lastTimestampNanos)
                        append('|')
                        append(camera.frameCount)
                        append('|')
                        append(camera.configurationFingerprint)
                        append('|')
                        append(camera.clockOffsetNanos)
                        append('|')
                        append(camera.clockDriftPpm)
                        append('|')
                        append(camera.synchronizationConfidence)
                        append('|')
                        append(camera.rollingShutterSkewNanos)
                    }
            }

        val bytes =
            MessageDigest.getInstance("SHA-256")
                .digest(
                    canonical.toByteArray(
                        StandardCharsets.UTF_8,
                    ),
                )

        return "NXBND1-" +
            bytes.joinToString("") {
                "%02x".format(
                    java.util.Locale.US,
                    it.toInt() and 0xFF,
                )
            }
    }
}
