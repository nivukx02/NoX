package com.nivukx.nox.core.capture

import android.content.Context
import com.nivukx.nox.core.model.CameraCaptureProfile
import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import com.nivukx.nox.core.nativeengine.NativeCapturePacket
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

/**
 * Fleet-level lifecycle authority for large Camera2 rigs.
 *
 * This layer deliberately owns lifecycle, admission and diagnostics, while each
 * Camera2CaptureController remains responsible for the HAL/session contract of one camera.
 *
 * Production invariants:
 * - at most [maximumCameras] active controller objects;
 * - duplicate camera ids never create a second controller;
 * - every created controller has one deterministic owner;
 * - failed opens remain retryable but never multiply resources;
 * - closeAll is idempotent and deterministic;
 * - camera callback execution remains bounded by CameraCaptureHandlerPool.
 */
class MultiCameraCaptureFleetCoordinator(
    context: Context,
    private val maximumCameras: Int = 128,
    private val packetSinkFactory:
        (cameraId: String) -> ((NativeCapturePacket) -> Unit)? = { null },
    private val controllerFactory:
        (Context, ((NativeCapturePacket) -> Unit)?) -> Camera2CaptureController =
        { applicationContext, sink ->
            Camera2CaptureController(
                context = applicationContext,
                nativeObservationSink = sink,
            )
        },
    private val depthFrameSinkFactory:
        (cameraId: String) -> ((DepthFrame) -> Unit)? = { null },
) : AutoCloseable {

    enum class CameraState {
        ADMITTED,
        OPENING,
        READY,
        FAILED,
        CLOSING,
    }

    data class CameraStatus(
        val cameraId: String,
        val state: CameraState,
        val requestedProfile: CameraCaptureProfile?,
        val effectiveProfile: CameraCaptureProfile?,
        val lastError: String?,
        val openAttempts: Long,
        val generation: Long,
        val depthStreamActive: Boolean = false,
    )

    data class FleetPlan(
        val plan: ConcurrentCaptureCapabilityEngine.RigPlan,
        val profiles: Map<String, CameraCaptureProfile>,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    data class FleetDiagnostics(
        val maximumCameras: Int,
        val trackedCameras: Int,
        val openingCameras: Int,
        val readyCameras: Int,
        val failedCameras: Int,
        val activeControllerCount: Int,
        val pool: CameraCaptureHandlerPool.Diagnostics,
        val totalOpenAttempts: Long,
        val totalFailures: Long,
        val lastFailureCameraId: String?,
        val lastFailure: String?,
    )

    private data class Entry(
        val cameraId: String,
        val controller: Camera2CaptureController,
        var state: CameraState,
        var requestedProfile: CameraCaptureProfile? = null,
        var effectiveProfile: CameraCaptureProfile? = null,
        var lastError: String? = null,
        var openAttempts: Long = 0L,
        var generation: Long = 0L,
        var depthForwardingGeneration: Long = 0L,
        var depthForwardingFailedGeneration: Long = 0L,
        var depthFrameSink: ((DepthFrame) -> Unit)? = null,
        var depthForwardingJob: Job? = null,
    )

    private val applicationContext = context.applicationContext
    private val lock = Any()
    private val entries =
        LinkedHashMap<String, Entry>()

    private var totalOpenAttempts = 0L
    private var totalFailures = 0L
    private var lastFailureCameraId: String? = null
    private var lastFailure: String? = null
    private var generationCounter = 0L
    private val closed = AtomicBoolean(false)
    private val depthScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    init {
        require(maximumCameras in 1..128)
    }

    /**
     * Admits a camera into the fleet without opening it.
     *
     * Admission is useful when the operator builds a rig incrementally and wants a deterministic
     * capacity decision before invoking any Camera2 HAL work.
     */
    fun admit(cameraId: String): Boolean {
        require(cameraId.isNotBlank())
        synchronized(lock) {
            check(!closed.get()) {
                "Camera fleet coordinator is closed"
            }

            if (entries.containsKey(cameraId)) {
                return true
            }

            if (entries.size >= maximumCameras) {
                recordFailureLocked(
                    cameraId = cameraId,
                    reason = "camera-fleet-capacity-exhausted",
                )
                return false
            }

            val controller =
                controllerFactory(
                    applicationContext,
                    packetSinkFactory(cameraId),
                )

            generationCounter += 1L
            entries[cameraId] =
                Entry(
                    cameraId = cameraId,
                    controller = controller,
                    state = CameraState.ADMITTED,
                    generation = generationCounter,
                )

            return true
        }
    }

    /**
     * Opens one camera with an already resolved production profile.
     *
     * The callback is invoked by Camera2CaptureController on its ordered camera execution context.
     */
    fun open(
        cameraId: String,
        profile: CameraCaptureProfile,
        sidecarDirectory: File? = null,
        allowPersistentSensorSidecars: Boolean = false,
        onReady: (CameraStatus) -> Unit = {},
    ): Boolean {
        require(cameraId.isNotBlank())

        var alreadyReadyStatus: CameraStatus? = null
        val entry =
            synchronized(lock) {
                check(!closed.get()) {
                    "Camera fleet coordinator is closed"
                }

                val existing =
                    entries[cameraId]

                if (existing == null) {
                    if (!admitLocked(cameraId)) {
                        return false
                    }
                    entries[cameraId]!!
                } else {
                    when (existing.state) {
                        CameraState.READY -> {
                            alreadyReadyStatus = statusOf(existing)
                            existing
                        }
                        CameraState.OPENING,
                        CameraState.CLOSING -> {
                            recordFailureLocked(
                                cameraId = cameraId,
                                reason = "camera-open-already-in-progress",
                            )
                            return false
                        }
                        CameraState.ADMITTED,
                        CameraState.FAILED -> existing
                    }
                }
            }

        // Never invoke client-owned code under the fleet-wide monitor. Callbacks may dispatch UI
        // work, close another camera, or query status; none should hold up unrelated camera state.
        alreadyReadyStatus?.let { status ->
            runCatching { onReady(status) }
            if (status.state == CameraState.READY) {
                startDepthForwarding(entry, status.generation)
            }
            return true
        }

        val requestGeneration =
            synchronized(lock) {
                // The first admission lock was released before reaching this point.
                // A concurrent close can remove the Entry during that interval.
                // Revalidate ownership before scheduling asynchronous Camera2 work.
                if (closed.get() || entries[cameraId] !== entry) {
                    return false
                }
                // Multiple callers can both observe ADMITTED/FAILED in the first lock section.
                // Claim OPENING atomically here so only one of those callers dispatches a request.
                if (
                    entry.state != CameraState.ADMITTED &&
                    entry.state != CameraState.FAILED
                ) {
                    recordFailureLocked(
                        cameraId = cameraId,
                        reason = "camera-open-already-in-progress",
                    )
                    return false
                }
                entry.state = CameraState.OPENING
                entry.requestedProfile = profile
                entry.effectiveProfile = null
                entry.lastError = null
                entry.openAttempts += 1L
                totalOpenAttempts += 1L
                entry.generation = ++generationCounter
                entry.depthForwardingFailedGeneration = 0L
                entry.depthFrameSink = null
                entry.generation
            }

        // Resolve the frame consumer before opening Camera2 so the controller can request a live
        // DEPTH16 output even when persistent sidecar recording is disabled.
        val depthSink = resolveDepthSink(entry, requestGeneration)
        val requestStillOwned =
            synchronized(lock) {
                if (
                    closed.get() ||
                    entries[cameraId] !== entry ||
                    entry.generation != requestGeneration ||
                    entry.state != CameraState.OPENING
                ) {
                    false
                } else {
                    entry.depthFrameSink = depthSink
                    true
                }
            }
        if (!requestStillOwned) return false

        runCatching {
            entry.controller.open(
                cameraId = cameraId,
                profile = profile,
                sidecarDirectory = sidecarDirectory,
                allowPersistentSensorSidecars =
                    allowPersistentSensorSidecars,
                enableLiveDepthStream = depthSink != null,
            ) { error ->
                val callbackStatus =
                    synchronized(lock) {
                        // Open/configure callbacks can arrive after close, retry or re-admission.
                        // Only the exact live Entry and generation may mutate current fleet state.
                        if (
                            closed.get() ||
                            entries[cameraId] !== entry ||
                            entry.generation != requestGeneration ||
                            entry.state != CameraState.OPENING
                        ) {
                            return@synchronized null
                        }

                        if (error == null) {
                            entry.state = CameraState.READY
                            entry.effectiveProfile = entry.controller.captureProfile
                            if (
                                entry.depthFrameSink != null &&
                                !entry.controller.isDepthReaderActive
                            ) {
                                entry.lastError =
                                    "live-depth-output-unavailable-after-session-negotiation"
                                recordFailureLocked(
                                    cameraId,
                                    entry.lastError!!,
                                )
                            } else {
                                entry.lastError = null
                            }
                        } else {
                            entry.state = CameraState.FAILED
                            entry.lastError =
                                error.message
                                    ?.take(512)
                                    ?: error.javaClass.simpleName
                            recordFailureLocked(
                                cameraId = cameraId,
                                reason = entry.lastError ?: "camera-open-failed",
                            )
                        }
                        statusOf(entry)
                    }

                // A null status means the callback was stale and must not update the UI.
                callbackStatus?.let { status ->
                    runCatching { onReady(status) }
                    if (status.state == CameraState.READY) {
                        startDepthForwarding(entry, requestGeneration)
                    } else {
                        cancelDepthForwarding(entry)
                    }
                }
            }
        }.onFailure { error ->
            val failureStatus =
                synchronized(lock) {
                    // Dispatch failure also belongs to a particular live request generation.
                    if (
                        closed.get() ||
                        entries[cameraId] !== entry ||
                        entry.generation != requestGeneration ||
                        entry.state != CameraState.OPENING
                    ) {
                        return@synchronized null
                    }
                    entry.state = CameraState.FAILED
                    entry.lastError =
                        error.message
                            ?.take(512)
                            ?: error.javaClass.simpleName
                    recordFailureLocked(
                        cameraId = cameraId,
                        reason = entry.lastError ?: "camera-open-dispatch-failed",
                    )
                    statusOf(entry)
                }
            failureStatus?.let {
                cancelDepthForwarding(entry)
                runCatching { onReady(it) }
            }
        }

        return true
    }

    /**
     * Resolves a profile from measured Camera2 capabilities and opens the camera.
     */
    fun openBest(
        cameraId: String,
        customization: HighEndMotionCaptureSettings,
        sidecarDirectory: File? = null,
        allowPersistentSensorSidecars: Boolean = false,
        onReady: (CameraStatus) -> Unit = {},
    ): CameraCapturePolicyEngine.Decision? {
        require(cameraId.isNotBlank())

        val capabilities =
            runCatching {
                CameraCapabilityEngine(
                    applicationContext,
                ).inspect(cameraId)
            }.getOrElse { error ->
                synchronized(lock) {
                    recordFailureLocked(
                        cameraId = cameraId,
                        reason =
                            "camera-capability-inspection-failed:" +
                                error.javaClass.simpleName,
                    )
                }
                return null
            }

        val decision =
            CameraCapturePolicyEngine()
                .resolve(
                    capabilities = capabilities,
                    customization =
                        customization.toMotionCaptureCustomization(),
                )

        if (
            !open(
                cameraId = cameraId,
                profile = decision.profile,
                sidecarDirectory = sidecarDirectory,
                allowPersistentSensorSidecars =
                    allowPersistentSensorSidecars,
                onReady = onReady,
            )
        ) {
            return null
        }

        return decision
    }

    /**
     * Builds a hardware-aware multi-camera admission plan before any Camera2 session is opened.
     *
     * Capability inspection is based on Camera2's advertised feature matrix; runtime confidence
     * inputs remain explicit so a supported camera is not automatically treated as a good camera.
     */
    fun planAdmission(
        cameraIds: Collection<String>,
        customization: HighEndMotionCaptureSettings,
        synchronizationConfidence: Map<String, Double> = emptyMap(),
        calibrationConfidence: Map<String, Double> = emptyMap(),
        thermalLoad: Map<String, Double> = emptyMap(),
        memoryPressure: Map<String, Double> = emptyMap(),
        viewDiversity: Map<String, Double> = emptyMap(),
        pairSkewMillis: Map<String, Double> = emptyMap(),
        requestedCameraCount: Int = cameraIds.size.coerceAtLeast(2),
    ): FleetPlan? {
        val uniqueCameraIds =
            cameraIds
                .filter { it.isNotBlank() }
                .distinct()
                .take(maximumCameras)

        if (uniqueCameraIds.size < 2) {
            return null
        }

        val matrices =
            uniqueCameraIds.mapNotNull { cameraId ->
                runCatching {
                    CameraCapabilityEngine(
                        applicationContext,
                    ).inspect(cameraId)
                }.getOrNull()
            }

        if (matrices.size < 2) {
            return null
        }

        val capabilityPlan =
            ConcurrentCaptureCapabilityEngine(
                ConcurrentCaptureCapabilityEngine.Config(
                    targetFps =
                        customization.targetFps
                            .coerceIn(15, 240),
                ),
            ).planFromFeatureMatrices(
                matrices = matrices,
                synchronizationConfidence =
                    synchronizationConfidence,
                calibrationConfidence =
                    calibrationConfidence,
                thermalLoad = thermalLoad,
                memoryPressure = memoryPressure,
                viewDiversity = viewDiversity,
                pairSkewMillis = pairSkewMillis,
                requestedCameraCount =
                    requestedCameraCount.coerceIn(2, matrices.size),
            )

        val policy = CameraCapturePolicyEngine()
        val profileByCamera =
            matrices.associate { matrix ->
                matrix.cameraId to
                    policy.resolve(
                        capabilities = matrix,
                        customization =
                            customization.toMotionCaptureCustomization(),
                    ).profile
            }

        return FleetPlan(
            plan = capabilityPlan,
            profiles =
                capabilityPlan.selectedCameraIds
                    .mapNotNull { id ->
                        profileByCamera[id]?.let { id to it }
                    }
                    .toMap(),
            warnings = capabilityPlan.warnings,
            recommendations = capabilityPlan.recommendations,
        )
    }

    /**
     * Opens only the cameras selected by the hardware-aware admission plan.
     *
     * The fleet refuses to bypass a rejected plan. Manual per-camera open() remains available for
     * diagnostics and controlled recovery.
     */
    fun openAdmissible(
        fleetPlan: FleetPlan,
        sidecarDirectory: File? = null,
        allowPersistentSensorSidecars: Boolean = false,
        onCameraReady: (CameraStatus) -> Unit = {},
    ): Boolean {
        if (!fleetPlan.plan.admissible) {
            return false
        }

        var allAdmitted = true
        fleetPlan.plan.selectedCameraIds.forEach { cameraId ->
            val profile =
                fleetPlan.profiles[cameraId]
            if (profile == null) {
                allAdmitted = false
                return@forEach
            }

            val opened =
                open(
                    cameraId = cameraId,
                    profile = profile,
                    sidecarDirectory = sidecarDirectory,
                    allowPersistentSensorSidecars =
                        allowPersistentSensorSidecars,
                    onReady = onCameraReady,
                )
            if (!opened) {
                allAdmitted = false
            }
        }

        return allAdmitted
    }

    fun status(cameraId: String): CameraStatus {
        require(cameraId.isNotBlank())
        synchronized(lock) {
            val entry =
                entries[cameraId]
                    ?: return CameraStatus(
                        cameraId = cameraId,
                        state = CameraState.FAILED,
                        requestedProfile = null,
                        effectiveProfile = null,
                        lastError =
                            "camera-not-admitted",
                        openAttempts = 0L,
                        generation = 0L,
                    )

            return CameraStatus(
                cameraId = entry.cameraId,
                state = entry.state,
                requestedProfile = entry.requestedProfile,
                effectiveProfile = entry.effectiveProfile,
                lastError = entry.lastError,
                openAttempts = entry.openAttempts,
                generation = entry.generation,
                depthStreamActive = entry.controller.isDepthReaderActive,
            )
        }
    }

    fun statuses(): List<CameraStatus> =
        synchronized(lock) {
            entries.values.map { entry ->
                CameraStatus(
                    cameraId = entry.cameraId,
                    state = entry.state,
                    requestedProfile = entry.requestedProfile,
                    effectiveProfile = entry.effectiveProfile,
                    lastError = entry.lastError,
                    openAttempts = entry.openAttempts,
                    generation = entry.generation,
                    depthStreamActive = entry.controller.isDepthReaderActive,
                )
            }
        }

    fun closeCamera(cameraId: String): Boolean {
        require(cameraId.isNotBlank())

        val entry =
            synchronized(lock) {
                val found = entries.remove(cameraId)
                    ?: return false
                found.state = CameraState.CLOSING
                found
            }

        cancelDepthForwarding(entry)
        runCatching {
            entry.controller.close()
        }

        return true
    }

    fun retry(
        cameraId: String,
        profile: CameraCaptureProfile,
        sidecarDirectory: File? = null,
        allowPersistentSensorSidecars: Boolean = false,
        onReady: (CameraStatus) -> Unit = {},
    ): Boolean {
        require(cameraId.isNotBlank())

        val admitted =
            synchronized(lock) {
                val entry = entries[cameraId]
                entry != null &&
                    (
                        entry.state ==
                            CameraState.FAILED ||
                            entry.state ==
                            CameraState.ADMITTED
                    )
            }

        return if (admitted) {
            open(
                cameraId = cameraId,
                profile = profile,
                sidecarDirectory = sidecarDirectory,
                allowPersistentSensorSidecars =
                    allowPersistentSensorSidecars,
                onReady = onReady,
            )
        } else {
            false
        }
    }

    fun diagnostics(): FleetDiagnostics =
        synchronized(lock) {
            FleetDiagnostics(
                maximumCameras = maximumCameras,
                trackedCameras = entries.size,
                openingCameras =
                    entries.values.count {
                        it.state == CameraState.OPENING
                    },
                readyCameras =
                    entries.values.count {
                        it.state == CameraState.READY
                    },
                failedCameras =
                    entries.values.count {
                        it.state == CameraState.FAILED
                    },
                activeControllerCount = entries.size,
                pool =
                    CameraCaptureHandlerPool.shared
                        .diagnostics(),
                totalOpenAttempts =
                    totalOpenAttempts,
                totalFailures = totalFailures,
                lastFailureCameraId =
                    lastFailureCameraId,
                lastFailure = lastFailure,
            )
        }

    fun closeAll() {
        val snapshot =
            synchronized(lock) {
                if (closed.get()) {
                    return
                }
                entries.values
                    .toList()
                    .also {
                        it.forEach { entry ->
                            entry.state = CameraState.CLOSING
                        }
                        entries.clear()
                    }
            }

        snapshot.forEach { entry ->
            cancelDepthForwarding(entry)
            runCatching {
                entry.controller.close()
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) {
            return
        }

        val snapshot =
            synchronized(lock) {
                entries.values
                    .toList()
                    .also { entries.clear() }
                    .onEach {
                        it.state = CameraState.CLOSING
                    }
            }

        depthScope.cancel()
        snapshot.forEach { entry ->
            cancelDepthForwarding(entry)
            runCatching {
                entry.controller.close()
            }
        }
    }

    private fun startDepthForwarding(
        entry: Entry,
        requestGeneration: Long,
    ) {
        // Use the consumer reserved by the open generation. Never call client-owned factory
        // code from a camera callback or create a second sink for the same live reader.
        val sink =
            synchronized(lock) {
                if (
                    closed.get() ||
                    entries[entry.cameraId] !== entry ||
                    entry.generation != requestGeneration ||
                    entry.state != CameraState.READY ||
                    entry.depthForwardingFailedGeneration == requestGeneration ||
                    !entry.controller.isDepthReaderActive
                ) {
                    return
                }
                entry.depthFrameSink ?: return
            }

        val job =
            synchronized(lock) {
                if (
                    closed.get() ||
                    entries[entry.cameraId] !== entry ||
                    entry.generation != requestGeneration ||
                    entry.state != CameraState.READY ||
                    entry.depthForwardingFailedGeneration == requestGeneration
                ) {
                    return
                }
                val existing = entry.depthForwardingJob
                if (
                    existing?.isActive == true &&
                    entry.depthForwardingGeneration == requestGeneration
                ) {
                    return
                }
                existing?.cancel()
                entry.depthForwardingGeneration = requestGeneration
                depthScope.launch(start = CoroutineStart.LAZY) {
                    try {
                        entry.controller.depthFlow.collect { frame ->
                            val current =
                                synchronized(lock) {
                                    !closed.get() &&
                                        entries[entry.cameraId] === entry &&
                                        entry.generation == requestGeneration &&
                                        entry.state == CameraState.READY
                                }
                            if (!current) return@collect

                            runCatching { sink(frame) }
                                .onFailure { failure ->
                                    val currentFailure =
                                        synchronized(lock) {
                                            if (
                                                !closed.get() &&
                                                entries[entry.cameraId] === entry &&
                                                entry.generation == requestGeneration &&
                                                entry.state == CameraState.READY &&
                                                entry.depthForwardingFailedGeneration != requestGeneration
                                            ) {
                                                entry.depthForwardingFailedGeneration = requestGeneration
                                                recordFailureLocked(
                                                    entry.cameraId,
                                                    "depth-frame-sink-failed:${failure.javaClass.simpleName}",
                                                )
                                                true
                                            } else {
                                                false
                                            }
                                        }
                                    // Stop after one sink failure; retrying a failing consumer on every
                                    // frame only burns CPU and floods diagnostics with duplicate faults.
                                    if (currentFailure) cancelDepthForwarding(entry)
                                }
                        }
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (failure: Throwable) {
                        synchronized(lock) {
                            if (
                                !closed.get() &&
                                entries[entry.cameraId] === entry &&
                                entry.generation == requestGeneration &&
                                entry.state == CameraState.READY
                            ) {
                                recordFailureLocked(
                                    entry.cameraId,
                                    "depth-flow-failed:${failure.javaClass.simpleName}",
                                )
                            }
                        }
                    }
                }.also { entry.depthForwardingJob = it }
            }
        job.start()
    }

    private fun resolveDepthSink(
        entry: Entry,
        requestGeneration: Long,
    ): ((DepthFrame) -> Unit)? {
        return runCatching { depthFrameSinkFactory(entry.cameraId) }
            .getOrElse { failure ->
                synchronized(lock) {
                    if (
                        !closed.get() &&
                        entries[entry.cameraId] === entry &&
                        entry.generation == requestGeneration &&
                        entry.state == CameraState.OPENING
                    ) {
                        recordFailureLocked(
                            entry.cameraId,
                            "depth-sink-factory-failed:${failure.javaClass.simpleName}",
                        )
                    }
                }
                null
            }
    }

    private fun cancelDepthForwarding(entry: Entry) {
        val job =
            synchronized(lock) {
                entry.depthForwardingGeneration = 0L
                entry.depthFrameSink = null
                entry.depthForwardingJob.also { entry.depthForwardingJob = null }
            }
        job?.cancel()
    }

    private fun admitLocked(cameraId: String): Boolean {
        if (entries.containsKey(cameraId)) {
            return true
        }

        if (entries.size >= maximumCameras) {
            recordFailureLocked(
                cameraId = cameraId,
                reason = "camera-fleet-capacity-exhausted",
            )
            return false
        }

        val controller =
            controllerFactory(
                applicationContext,
                packetSinkFactory(cameraId),
            )

        entries[cameraId] =
            Entry(
                cameraId = cameraId,
                controller = controller,
                state = CameraState.ADMITTED,
                generation = ++generationCounter,
            )
        return true
    }

    /**
     * Creates a stable status snapshot from an Entry that has already passed generation and
     * ownership validation. Call only while holding the fleet lock.
     */
    private fun statusOf(entry: Entry): CameraStatus =
        CameraStatus(
            cameraId = entry.cameraId,
            state = entry.state,
            requestedProfile = entry.requestedProfile,
            effectiveProfile = entry.effectiveProfile,
            lastError = entry.lastError,
            openAttempts = entry.openAttempts,
            generation = entry.generation,
            depthStreamActive = entry.controller.isDepthReaderActive,
        )

    private fun recordFailureLocked(
        cameraId: String,
        reason: String,
    ) {
        totalFailures += 1L
        lastFailureCameraId = cameraId
        lastFailure = reason.take(512)
    }
}
