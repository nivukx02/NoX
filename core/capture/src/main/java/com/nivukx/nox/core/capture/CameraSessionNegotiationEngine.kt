package com.nivukx.nox.core.capture

import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.util.Range
import android.util.Size
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.os.Build
import android.view.Surface
import java.util.concurrent.Executor

/**
 * Production Camera2 session negotiation authority.
 *
 * Separates hardware-verified support from platform declaration and query-unavailable fallback.
 * An unavailable query is never converted into a positive capability claim.
 */
class CameraSessionNegotiationEngine(
    private val manager: CameraManager,
) {
    enum class Verification {
        HARDWARE_VERIFIED,
        PLATFORM_DECLARED_ONLY,
        UNKNOWN,
        REJECTED,
    }

    data class OutputSpec(
        val surface: Surface,
        val name: String,
        val enableReadoutTimestamp: Boolean = false,
        val hardwareReadoutSupported: Boolean = false,
        val timestampBase: Int? = null,
        val physicalCameraId: String? = null,
        val dynamicRangeProfile: Long? = null,
    ) {
        init {
            require(name.isNotBlank())
        }
    }

    data class SessionPlan(
        val cameraId: String,
        val outputs: List<OutputSpec>,
        val verification: Verification,
        val supported: Boolean,
        val reason: String,
        val readoutTimestampRequested: Boolean,
    ) {
        val admissible: Boolean
            get() = supported
    }

    data class HighSpeedPlan(
        val cameraId: String,
        val requestedFps: Int,
        val selectedRange: Range<Int>?,
        val supported: Boolean,
        val verification: Verification,
        val requiresConstrainedSession: Boolean,
        val reason: String,
    ) {
        val selectedFps: Int?
            get() =
                selectedRange?.upper

        val admissible: Boolean
            get() = supported &&
                requiresConstrainedSession &&
                selectedRange != null
    }

    data class ConcurrentPlan(
        val cameraIds: Set<String>,
        val verification: Verification,
        val supported: Boolean,
        val reasons: List<String>,
    ) {
        val admissible: Boolean
            get() = supported
    }

    private val noOpExecutor = Executor { }

    fun buildRegularSessionConfiguration(
        outputs: List<OutputSpec>,
        callback: CameraCaptureSession.StateCallback? = null,
        executor: Executor = noOpExecutor,
    ): SessionConfiguration {
        require(outputs.isNotEmpty()) {
            "At least one output is required for a Camera2 session"
        }

        val requestedTimestampBases =
            outputs.mapNotNull { it.timestampBase }.distinct()
        require(requestedTimestampBases.size <= 1) {
            "All NoX outputs in one session must use one explicit timestamp base"
        }

        require(
            outputs.none {
                it.enableReadoutTimestamp &&
                    !it.hardwareReadoutSupported
            },
        ) {
            "A readout-timestamp request cannot be silently downgraded on an unsupported output"
        }

        /*
         * Android 17 requires a consistent timestamp-base/readout policy across session outputs.
         * Normalize here so analysis, preview, depth and optional evidence surfaces cannot create a
         * mixed timing topology that later fails during session creation.
         */
        val uniformReadoutTimestamp =
            outputs.any {
                it.enableReadoutTimestamp &&
                    it.hardwareReadoutSupported
            }
        val uniformTimestampBase =
            outputs
                .mapNotNull { it.timestampBase }
                .firstOrNull()
                ?: if (Build.VERSION.SDK_INT >= 33) {
                    OutputConfiguration.TIMESTAMP_BASE_SENSOR
                } else {
                    null
                }

        val dynamicRangeProfiles =
            outputs
                .mapNotNull {
                    it.dynamicRangeProfile
                }
                .distinct()

        require(dynamicRangeProfiles.size <= 1) {
            "Camera2 outputs in one NoX session must use one negotiated dynamic-range profile"
        }

        val uniformDynamicRangeProfile =
            dynamicRangeProfiles.firstOrNull()

        val normalizedOutputs =
            outputs.map {
                it.copy(
                    enableReadoutTimestamp =
                        uniformReadoutTimestamp,
                    timestampBase =
                        uniformTimestampBase,
                    dynamicRangeProfile =
                        uniformDynamicRangeProfile,
                )
            }

        return SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR,
            normalizedOutputs.map(::buildOutputConfiguration),
            executor,
            callback ?: NO_OP_STATE_CALLBACK,
        )
    }

    /**
     * API 35+ pre-open verification for the concrete output topology.
     */
    fun verifySingleCameraSession(
        cameraId: String,
        outputs: List<OutputSpec>,
    ): SessionPlan {
        require(cameraId.isNotBlank())
        require(outputs.isNotEmpty())

        val readoutRequested = outputs.any { it.enableReadoutTimestamp }

        if (Build.VERSION.SDK_INT < 35) {
            return SessionPlan(
                cameraId = cameraId,
                outputs = outputs,
                verification = Verification.UNKNOWN,
                supported = false,
                reason = "pre-open session verification is unavailable below API 35; live-session fallback remains possible",
                readoutTimestampRequested = readoutRequested,
            )
        }

        return runCatching {
            if (!manager.isCameraDeviceSetupSupported(cameraId)) {
                SessionPlan(
                    cameraId = cameraId,
                    outputs = outputs,
                    verification = Verification.PLATFORM_DECLARED_ONLY,
                    supported = false,
                    reason = "CameraDeviceSetup query is unavailable for this camera",
                    readoutTimestampRequested = readoutRequested,
                )
            } else {
                val setup = manager.getCameraDeviceSetup(cameraId)
                val supported = setup.isSessionConfigurationSupported(
                    buildRegularSessionConfiguration(outputs),
                )

                SessionPlan(
                    cameraId = cameraId,
                    outputs = outputs,
                    verification = if (supported) {
                        Verification.HARDWARE_VERIFIED
                    } else {
                        Verification.REJECTED
                    },
                    supported = supported,
                    reason = if (supported) {
                        "CameraDeviceSetup accepted the requested stream topology"
                    } else {
                        "CameraDeviceSetup rejected the requested stream topology"
                    },
                    readoutTimestampRequested = readoutRequested,
                )
            }
        }.getOrElse { failure ->
            SessionPlan(
                cameraId = cameraId,
                outputs = outputs,
                verification = Verification.UNKNOWN,
                supported = false,
                reason =
                    "session support query failed safely: " +
                        failure.javaClass.simpleName,
                readoutTimestampRequested = readoutRequested,
            )
        }
    }

    /**
     * API 30+ runtime verification of multiple camera SessionConfiguration objects.
     */
    /**
     * Selects a true Camera2 constrained-high-speed FPS range from measured HAL ranges.
     *
     * This is deliberately separate from the regular-session FPS request. Android exposes a
     * dedicated constrained-high-speed session type for high-speed recording, so a regular
     * SessionConfiguration is never labeled high-speed merely because an FPS number is large.
     */
    fun selectHighSpeedPlan(
        cameraId: String,
        requestedFps: Int,
        availableRanges: List<Range<Int>>,
        outputCount: Int,
    ): HighSpeedPlan {
        require(cameraId.isNotBlank())
        require(requestedFps in 1..240)
        require(outputCount >= 1)

        if (Build.VERSION.SDK_INT < 23) {
            return HighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                selectedRange = null,
                supported = false,
                verification = Verification.UNKNOWN,
                requiresConstrainedSession = false,
                reason = "constrained high-speed capture requires API 23+",
            )
        }

        if (availableRanges.isEmpty()) {
            return HighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                selectedRange = null,
                supported = false,
                verification = Verification.REJECTED,
                requiresConstrainedSession = true,
                reason = "camera exposes no constrained high-speed FPS ranges",
            )
        }

        if (outputCount > 2) {
            return HighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                selectedRange = null,
                supported = false,
                verification = Verification.REJECTED,
                requiresConstrainedSession = true,
                reason = "NoX keeps high-speed mode disabled for multi-output topologies above two surfaces",
            )
        }

        val selected =
            availableRanges
                .filter {
                    it.lower <= requestedFps &&
                        it.upper >= requestedFps
                }
                .minWithOrNull(
                    compareBy<Range<Int>> {
                        it.upper - requestedFps
                    }.thenBy {
                        it.upper
                    }.thenBy {
                        it.lower
                    },
                )
                ?: availableRanges
                    .filter { it.upper >= requestedFps }
                    .minWithOrNull(
                        compareBy<Range<Int>> {
                            it.upper - requestedFps
                        }.thenBy { it.upper },
                    )

        return if (selected != null) {
            HighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                selectedRange = selected,
                supported = true,
                verification = Verification.PLATFORM_DECLARED_ONLY,
                requiresConstrainedSession = true,
                reason = "measured constrained-high-speed FPS range selected; live session creation remains the hardware authority",
            )
        } else {
            HighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                selectedRange = null,
                supported = false,
                verification = Verification.REJECTED,
                requiresConstrainedSession = true,
                reason = "no constrained-high-speed FPS range satisfies requested rate",
            )
        }
    }

    fun inspectHighSpeedCapability(
        cameraId: String,
        outputSize: Size,
        requestedFps: Int,
        outputCount: Int,
    ): HighSpeedPlan {
        require(cameraId.isNotBlank())
        require(outputSize.width > 0 && outputSize.height > 0)

        if (Build.VERSION.SDK_INT < 23) {
            return selectHighSpeedPlan(
                cameraId,
                requestedFps,
                emptyList(),
                outputCount,
            )
        }

        return runCatching {
            val characteristics =
                manager.getCameraCharacteristics(cameraId)
            val streamMap =
                characteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP,
                )
            val supportedSizes =
                streamMap
                    ?.highSpeedVideoSizes
                    ?.toList()
                    .orEmpty()
            if (outputSize !in supportedSizes) {
                return@runCatching HighSpeedPlan(
                    cameraId = cameraId,
                    requestedFps = requestedFps,
                    selectedRange = null,
                    supported = false,
                    verification = Verification.REJECTED,
                    requiresConstrainedSession = true,
                    reason = "requested output size is not listed as a high-speed video size",
                )
            }
            val ranges =
                streamMap
                    ?.getHighSpeedVideoFpsRangesFor(outputSize)
                    ?.toList()
                    .orEmpty()
            selectHighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                availableRanges = ranges,
                outputCount = outputCount,
            )
        }.getOrElse { failure ->
            HighSpeedPlan(
                cameraId = cameraId,
                requestedFps = requestedFps,
                selectedRange = null,
                supported = false,
                verification = Verification.UNKNOWN,
                requiresConstrainedSession = true,
                reason =
                    "high-speed capability query failed safely: " +
                        failure.javaClass.simpleName,
            )
        }
    }

    fun verifyConcurrentSessions(
        cameraIdToOutputs: Map<String, List<OutputSpec>>,
    ): ConcurrentPlan {
        val cameraIds = cameraIdToOutputs.keys.toSet()

        require(cameraIds.size >= 2) {
            "Concurrent verification requires at least two cameras"
        }
        require(cameraIdToOutputs.values.all { it.isNotEmpty() }) {
            "Every concurrent camera must have at least one output"
        }

        if (Build.VERSION.SDK_INT < 30) {
            return ConcurrentPlan(
                cameraIds = cameraIds,
                verification = Verification.UNKNOWN,
                supported = false,
                reasons = listOf("concurrent Camera2 verification requires API 30+"),
            )
        }

        val declaredSets = runCatching {
            manager.getConcurrentCameraIds().map { it.toSet() }
        }.getOrDefault(emptySet())

        if (!declaredSets.any { candidate ->
                cameraIds.all { it in candidate }
            }) {
            return ConcurrentPlan(
                cameraIds = cameraIds,
                verification = Verification.REJECTED,
                supported = false,
                reasons = listOf(
                    "camera set is not declared by CameraManager.getConcurrentCameraIds()",
                ),
            )
        }

        return runCatching {
            val configs = cameraIdToOutputs.mapValues { (_, outputs) ->
                buildRegularSessionConfiguration(outputs)
            }
            val supported =
                manager.isConcurrentSessionConfigurationSupported(configs)

            ConcurrentPlan(
                cameraIds = cameraIds,
                verification = if (supported) {
                    Verification.HARDWARE_VERIFIED
                } else {
                    Verification.REJECTED
                },
                supported = supported,
                reasons = if (supported) {
                    emptyList()
                } else {
                    listOf(
                        "CameraManager rejected the combined concurrent session topology",
                    )
                },
            )
        }.getOrElse { failure ->
            ConcurrentPlan(
                cameraIds = cameraIds,
                verification = Verification.UNKNOWN,
                supported = false,
                reasons = listOf(
                    "concurrent session query failed: " +
                        failure.javaClass.simpleName,
                ),
            )
        }
    }

    /**
     * Builds the OutputConfiguration used by both capability checks and live sessions.
     */
    fun buildOutputConfiguration(
        spec: OutputSpec,
    ): OutputConfiguration =
        OutputConfiguration(spec.surface).apply {
            if (Build.VERSION.SDK_INT >= 33) {
                (
                    spec.timestampBase
                        ?: OutputConfiguration.TIMESTAMP_BASE_SENSOR
                ).let(::setTimestampBase)
            }
            if (
                Build.VERSION.SDK_INT >= 34 &&
                spec.enableReadoutTimestamp &&
                spec.hardwareReadoutSupported
            ) {
                setReadoutTimestampEnabled(true)
            }
            if (Build.VERSION.SDK_INT >= 28) {
                spec.physicalCameraId?.let(::setPhysicalCameraId)
            }
            if (
                Build.VERSION.SDK_INT >= 33
            ) {
                spec.dynamicRangeProfile?.let(
                    ::setDynamicRangeProfile,
                )
            }
        }

    fun supportsHardwareReadoutTimestamp(
        cameraId: String,
    ): Boolean {
        if (Build.VERSION.SDK_INT < 34) return false

        return runCatching {
            manager.getCameraCharacteristics(cameraId).get(
                CameraCharacteristics.SENSOR_READOUT_TIMESTAMP,
            ) ==
                CameraCharacteristics.SENSOR_READOUT_TIMESTAMP_HARDWARE
        }.getOrDefault(false)
    }

    fun declaredConcurrentCameraSets(): List<Set<String>> {
        if (Build.VERSION.SDK_INT < 30) return emptyList()

        return runCatching {
            manager.getConcurrentCameraIds()
                .map { it.toSet() }
                .filter { it.size >= 2 }
                .sortedBy { it.sorted().joinToString(",") }
        }.getOrDefault(emptyList())
    }

    fun summarize(
        plan: SessionPlan,
    ): String =
        buildString {
            append("camera=")
            append(plan.cameraId)
            append(";verification=")
            append(plan.verification.name)
            append(";supported=")
            append(plan.supported)
            append(";outputs=")
            append(plan.outputs.size)
            append(";readout=")
            append(plan.readoutTimestampRequested)
            append(";reason=")
            append(plan.reason)
        }

    fun summarize(
        plan: ConcurrentPlan,
    ): String =
        buildString {
            append("cameras=")
            append(plan.cameraIds.sorted().joinToString(","))
            append(";verification=")
            append(plan.verification.name)
            append(";supported=")
            append(plan.supported)
            if (plan.reasons.isNotEmpty()) {
                append(";reasons=")
                append(plan.reasons.joinToString("|"))
            }
        }

    private companion object {
        val NO_OP_STATE_CALLBACK =
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(
                    session: CameraCaptureSession,
                ) = Unit

                override fun onConfigureFailed(
                    session: CameraCaptureSession,
                ) = Unit
            }
    }
}
