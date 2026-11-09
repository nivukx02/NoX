package com.nivukx.nox

import com.nivukx.nox.core.ai.ProductionCaptureStageScheduler
import com.nivukx.nox.core.reconstruction.GpuBufferLeaseCoordinator
import com.nivukx.nox.core.ai.RuntimeFailureContainmentEngine
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min

/**
 * Bounded lifecycle/resource governor for long-running motion-capture sessions.
 *
 * It tracks resource pressure and ownership without allocating unbounded diagnostic state.
 * Decisions are deterministic and safe to replay from telemetry.
 */
class MotionRuntimeResourceGovernor(
    private val maximumEvents: Int = 2048,
    private val maximumResources: Int = 512,
    private val maximumAllocatedBytes: Long = Long.MAX_VALUE,
    private val maximumRuntimeFaultsPerWindow: Int = 4,
) : AutoCloseable {

    init {
        require(maximumEvents > 0)
        require(maximumResources > 0)
        require(maximumAllocatedBytes > 0L)
        require(maximumRuntimeFaultsPerWindow in 1..32)
    }

    enum class ResourceKind {
        NATIVE_BUFFER,
        GPU_BUFFER,
        TSDF_BLOCK,
        TRACK_STATE,
        FACE_STATE,
        EYE_STATE,
        HAND_STATE,
        CALIBRATION_STATE,
        TEMPORAL_WINDOW,
        SESSION,
    }

    enum class PressureBand {
        NOMINAL,
        ELEVATED,
        HIGH,
        CRITICAL,
    }

    data class ResourceLease(
        val id: Long,
        val kind: ResourceKind,
        val bytes: Long,
        val createdAtNanos: Long,
        val owner: String,
    ) {
        init {
            require(id > 0L)
            require(bytes >= 0L)
            require(createdAtNanos >= 0L)
            require(owner.isNotBlank())
        }
    }

    data class RuntimeSnapshot(
        val timestampNanos: Long,
        val allocatedBytes: Long,
        val resourceCount: Int,
        val memoryPressure: Double,
        val cpuPressure: Double,
        val gpuPressure: Double,
        val thermalRisk: Double,
        val oldestLeaseAgeNanos: Long,
        val staleLeaseCount: Int,
        val pressureBand: PressureBand,
        val forcedSheddingRequired: Boolean,
        val safeToContinue: Boolean,
        val warnings: List<String>,
        val runtimeFaultDecision:
            RuntimeFailureContainmentEngine.Decision? = null,
        val gpuLeaseWatchdogAborts: Long = 0L,
        val gpuLeaseStaleGenerationReleases: Long = 0L,
    )

    data class Event(
        val timestampNanos: Long,
        val code: String,
        val resourceKind: ResourceKind?,
        val bytes: Long,
        val recoverable: Boolean,
    )

    private val leases = LinkedHashMap<Long, ResourceLease>()
    private val events = ArrayDeque<Event>()
    private val nextId = AtomicLong(1L)
    private val closed = AtomicBoolean(false)
    private val lastSnapshot = AtomicReference<RuntimeSnapshot?>(null)
    private val gpuBufferLeaseCoordinator =
        GpuBufferLeaseCoordinator(
            GpuBufferLeaseCoordinator.Config(
                maximumSlots = maximumResources.coerceAtMost(64),
                maximumInFlight = maximumResources.coerceAtMost(16).coerceAtLeast(1),
            ),
        )
    private val frontierSessionRecoveryCoordinator =
        com.nivukx.nox.core.ai.FrontierSessionRecoveryCoordinator()
    private val runtimeFailureContainmentEngine =
        RuntimeFailureContainmentEngine(
            RuntimeFailureContainmentEngine.Config(
                maximumRecoveryAttemptsPerWindow =
                    maximumRuntimeFaultsPerWindow,
            ),
        )
    private val captureStageScheduler =
        ProductionCaptureStageScheduler()

    @Synchronized
    fun reserve(
        kind: ResourceKind,
        bytes: Long,
        owner: String,
        nowNanos: Long,
    ): ResourceLease? {
        ensureOpen()
        if (bytes < 0L || owner.isBlank() || leases.size >= maximumResources) {
            record(nowNanos, "resource-reservation-rejected", kind, max(bytes, 0L), true)
            return null
        }

        val currentBytes =
            saturatingAllocatedBytes()
        if (
            bytes > maximumAllocatedBytes ||
            currentBytes >
                maximumAllocatedBytes - bytes
        ) {
            record(
                nowNanos,
                "memory-budget-reservation-rejected",
                kind,
                max(bytes, 0L),
                true,
            )
            return null
        }

        var id = nextId.getAndIncrement()
        if (id <= 0L) {
            id = 1L
            nextId.set(2L)
        }
        while (leases.containsKey(id)) {
            id = nextId.getAndIncrement()
            if (id <= 0L) {
                id = 1L
                nextId.set(2L)
            }
        }

        val lease = ResourceLease(
            id = id,
            kind = kind,
            bytes = bytes,
            createdAtNanos = nowNanos.coerceAtLeast(0L),
            owner = owner,
        )
        leases[id] = lease
        record(
            nowNanos = nowNanos,
            code = "resource-reserved",
            resourceKind = kind,
            bytes = bytes,
            recoverable = true,
        )
        return lease
    }

    @Synchronized
    fun release(
        leaseId: Long,
        nowNanos: Long,
    ): Boolean {
        ensureOpen()
        val removed = leases.remove(leaseId)
        if (removed == null) {
            record(nowNanos, "resource-release-miss", null, 0L, true)
            return false
        }
        return true
    }

    @Synchronized
    fun releaseOwner(
        owner: String,
        nowNanos: Long,
    ): Int {
        ensureOpen()
        if (owner.isBlank()) return 0
        val matching = leases.values
            .filter { it.owner == owner }
            .map { it.id }
        matching.forEach { leases.remove(it) }
        if (matching.isNotEmpty()) {
            record(
                nowNanos,
                "owner-release",
                null,
                0L,
                true,
            )
        }
        return matching.size
    }

    @Synchronized
    fun forceRelease(
        kind: ResourceKind,
        nowNanos: Long,
    ): Int {
        ensureOpen()
        val matching = leases.values
            .filter { it.kind == kind }
            .map { it.id }
        matching.forEach { leases.remove(it) }
        if (matching.isNotEmpty()) {
            record(
                nowNanos,
                "resource-kind-shed",
                kind,
                0L,
                true,
            )
        }
        return matching.size
    }

    @Synchronized
    fun snapshot(
        timestampNanos: Long,
        memoryPressure: Double,
        cpuPressure: Double,
        gpuPressure: Double,
        thermalRisk: Double,
        staleLeaseAgeNanos: Long = 2_000_000_000L,
    ): RuntimeSnapshot {
        ensureOpen()
        require(timestampNanos >= 0L)
        listOf(
            memoryPressure,
            cpuPressure,
            gpuPressure,
            thermalRisk,
        ).forEach {
            require(it.isFinite() && it in 0.0..1.0)
        }
        require(staleLeaseAgeNanos >= 0L)

        val allocated =
            saturatingAllocatedBytes()
        val allocatedBudgetPressure =
            if (maximumAllocatedBytes == Long.MAX_VALUE) {
                0.0
            } else {
                (
                    allocated.toDouble() /
                        maximumAllocatedBytes.toDouble().coerceAtLeast(1.0)
                    ).coerceIn(0.0, 1.0)
            }
        val effectiveMemoryPressure =
            max(
                memoryPressure,
                allocatedBudgetPressure,
            )
        val stale = leases.values.count {
            timestampNanos >= it.createdAtNanos &&
                timestampNanos - it.createdAtNanos > staleLeaseAgeNanos
        }
        val oldest = leases.values.minOfOrNull {
            timestampNanos - it.createdAtNanos
        }?.coerceAtLeast(0L) ?: 0L

        val combinedPressure = maxOf(
            effectiveMemoryPressure,
            cpuPressure,
            gpuPressure,
            thermalRisk,
        )
        val runtimeFaultDecision =
            runtimeFailureContainmentEngine.tick(timestampNanos)
        val gpuLeaseTelemetry =
            gpuBufferLeaseCoordinator.telemetry(timestampNanos)

        val band = when {
            combinedPressure >= 0.97 -> PressureBand.CRITICAL
            combinedPressure >= 0.85 -> PressureBand.HIGH
            combinedPressure >= 0.65 -> PressureBand.ELEVATED
            else -> PressureBand.NOMINAL
        }

        val warnings = buildList {
            if (stale > 0) add("stale-resource-leases")
            if (effectiveMemoryPressure >= 0.90) {
                add("memory-pressure-high")
            }
            if (
                maximumAllocatedBytes != Long.MAX_VALUE &&
                allocatedBudgetPressure >= 0.85
            ) {
                add("memory-budget-near-capacity")
            }
            if (allocated >= maximumAllocatedBytes) {
                add("memory-budget-capacity-reached")
            }
            if (gpuPressure >= 0.90) add("gpu-pressure-high")
            if (cpuPressure >= 0.90) add("cpu-pressure-high")
            if (thermalRisk >= 0.82) add("thermal-risk-high")
            if (leases.size >= maximumResources) {
                add("resource-table-capacity-reached")
            }
            if (gpuLeaseTelemetry.watchdogAborts > 0L) {
                add("gpu-buffer-watchdog-abort")
            }
            if (gpuLeaseTelemetry.staleGenerationReleases > 0L) {
                add("gpu-buffer-stale-generation-rejected")
            }
            runtimeFaultDecision.warnings.forEach {
                add(it)
            }
        }

        val snapshot = RuntimeSnapshot(
            timestampNanos = timestampNanos,
            allocatedBytes = allocated,
            resourceCount = leases.size,
            memoryPressure = memoryPressure,
            cpuPressure = cpuPressure,
            gpuPressure = gpuPressure,
            thermalRisk = thermalRisk,
            oldestLeaseAgeNanos = oldest,
            staleLeaseCount = stale,
            pressureBand = band,
            forcedSheddingRequired =
                band == PressureBand.CRITICAL ||
                    stale > maximum(1, leases.size / 5),
            safeToContinue =
                band != PressureBand.CRITICAL &&
                    effectiveMemoryPressure < 0.98 &&
                    allocated <= maximumAllocatedBytes &&
                    runtimeFaultDecision.safeToContinue &&
                    gpuLeaseTelemetry.oldestLeaseAgeMilliseconds <= 350.0,
            warnings = warnings,
            runtimeFaultDecision = runtimeFaultDecision,
            gpuLeaseWatchdogAborts = gpuLeaseTelemetry.watchdogAborts,
            gpuLeaseStaleGenerationReleases =
                gpuLeaseTelemetry.staleGenerationReleases,
        )
        lastSnapshot.set(snapshot)
        return snapshot
    }

    fun latestSnapshot(): RuntimeSnapshot? = lastSnapshot.get()

    @Synchronized
    fun recordCaptureStageLatency(
        stage: ProductionCaptureStageScheduler.Stage,
        latencyMilliseconds: Double,
    ) {
        ensureOpen()
        captureStageScheduler.recordStageLatency(
            stage = stage,
            latencyMilliseconds = latencyMilliseconds,
        )
    }

    @Synchronized
    fun decideCaptureStages(
        pressure: ProductionCaptureStageScheduler.RuntimePressure,
    ): ProductionCaptureStageScheduler.FrameDecision {
        ensureOpen()
        return captureStageScheduler.decide(pressure)
    }

    @Synchronized
    fun latestCaptureStageDecision():
        ProductionCaptureStageScheduler.FrameDecision? {
        ensureOpen()
        return captureStageScheduler.latestDecision()
    }

    @Synchronized
    fun recordRuntimeFault(
        timestampNanos: Long,
        kind: RuntimeFailureContainmentEngine.FaultKind,
        severity: RuntimeFailureContainmentEngine.Severity,
        signature: String = kind.name,
    ): RuntimeFailureContainmentEngine.Decision {
        ensureOpen()
        return runtimeFailureContainmentEngine.observeFault(
            timestampNanos = timestampNanos,
            kind = kind,
            severity = severity,
            signature = signature,
        )
    }

    @Synchronized
    fun recordRuntimeSuccess(
        timestampNanos: Long,
        frames: Int = 1,
    ): RuntimeFailureContainmentEngine.Decision {
        ensureOpen()
        return runtimeFailureContainmentEngine.observeSuccess(
            timestampNanos = timestampNanos,
            frames = frames,
        )
    }

    @Synchronized
    fun checkpointSession(
        timestampNanos: Long,
        actorCount: Int,
        cameraCount: Int,
        authoritativeCameraCount: Int,
        reconstructionFrame: Long,
        activeStages: Set<String>,
        stateMaterial: ByteArray,
    ): com.nivukx.nox.core.ai.FrontierSessionRecoveryCoordinator.SessionCheckpoint {
        ensureOpen()
        return frontierSessionRecoveryCoordinator.checkpoint(
            timestampNanos = timestampNanos,
            actorCount = actorCount,
            cameraCount = cameraCount,
            authoritativeCameraCount = authoritativeCameraCount,
            reconstructionFrame = reconstructionFrame,
            activeStages = activeStages,
            stateMaterial = stateMaterial,
        )
    }

    @Synchronized
    fun latestSessionCheckpoint():
        com.nivukx.nox.core.ai.FrontierSessionRecoveryCoordinator.SessionCheckpoint? {
        ensureOpen()
        return frontierSessionRecoveryCoordinator.latestCheckpoint()
    }

    @Synchronized
    fun sessionRecoveryPlan(
        timestampNanos: Long,
        kind: RuntimeFailureContainmentEngine.FaultKind,
        severity: RuntimeFailureContainmentEngine.Severity,
        signature: String = kind.name,
    ): com.nivukx.nox.core.ai.FrontierSessionRecoveryCoordinator.RecoveryPlan {
        ensureOpen()
        return frontierSessionRecoveryCoordinator.planRecovery(
            timestampNanos = timestampNanos,
            faultKind = kind,
            severity = severity,
            signature = signature,
        )
    }

    @Synchronized
    fun recordSessionRecoverySuccess(
        timestampNanos: Long,
        frames: Int = 1,
    ) {
        ensureOpen()
        frontierSessionRecoveryCoordinator.recordRecoverySuccess(
            timestampNanos = timestampNanos,
            frames = frames,
        )
    }

    @Synchronized
    fun runtimeFaultDecision(
        timestampNanos: Long,
    ): RuntimeFailureContainmentEngine.Decision {
        ensureOpen()
        return runtimeFailureContainmentEngine.tick(timestampNanos)
    }

    @Synchronized
    fun acquireGpuBuffer(
        timestampNanos: Long,
    ): GpuBufferLeaseCoordinator.LeaseToken? {
        ensureOpen()
        return gpuBufferLeaseCoordinator.acquire(timestampNanos)
    }

    @Synchronized
    fun markGpuBufferSubmitted(
        token: GpuBufferLeaseCoordinator.LeaseToken,
        timestampNanos: Long,
    ): Boolean {
        ensureOpen()
        return gpuBufferLeaseCoordinator.markGpuSubmitted(
            token,
            timestampNanos,
        )
    }

    @Synchronized
    fun markGpuBufferCompleted(
        token: GpuBufferLeaseCoordinator.LeaseToken,
        timestampNanos: Long,
    ): Boolean {
        ensureOpen()
        return gpuBufferLeaseCoordinator.markGpuCompleted(
            token,
            timestampNanos,
        )
    }

    @Synchronized
    fun releaseGpuBuffer(
        token: GpuBufferLeaseCoordinator.LeaseToken,
        timestampNanos: Long,
        force: Boolean = false,
    ): GpuBufferLeaseCoordinator.ReleaseResult {
        ensureOpen()
        return gpuBufferLeaseCoordinator.release(
            token,
            timestampNanos,
            force,
        )
    }

    @Synchronized
    fun sweepGpuBufferLeases(
        timestampNanos: Long,
    ): Int {
        ensureOpen()
        return gpuBufferLeaseCoordinator.watchdogSweep(
            timestampNanos,
        )
    }

    @Synchronized
    fun gpuBufferLeaseTelemetry(
        timestampNanos: Long,
    ): GpuBufferLeaseCoordinator.Telemetry {
        ensureOpen()
        return gpuBufferLeaseCoordinator.telemetry(
            timestampNanos,
        )
    }

    @Synchronized
    fun sweepStaleLeases(
        nowNanos: Long,
        maximumAgeNanos: Long,
        preserveKinds: Set<ResourceKind> = emptySet(),
    ): Int {
        ensureOpen()
        require(nowNanos >= 0L)
        require(maximumAgeNanos >= 0L)

        val staleIds = leases.values
            .asSequence()
            .filter { it.kind !in preserveKinds }
            .filter {
                nowNanos >= it.createdAtNanos &&
                    nowNanos - it.createdAtNanos > maximumAgeNanos
            }
            .map { it.id }
            .toList()

        staleIds.forEach { leases.remove(it) }
        if (staleIds.isNotEmpty()) {
            record(
                nowNanos = nowNanos,
                code = "stale-resource-sweep",
                resourceKind = null,
                bytes = 0L,
                recoverable = true,
            )
        }
        return staleIds.size
    }

    @Synchronized
    fun activeLeaseCount(): Int = leases.size

    @Synchronized
    fun allocatedBytes(): Long =
        saturatingAllocatedBytes()

    @Synchronized
    fun eventCount(): Int = events.size

    @Synchronized
    fun recentEvents(limit: Int = 16): List<Event> {
        require(limit >= 0)
        return events.toList().takeLast(limit)
    }

    @Synchronized
    fun clearEvents() {
        events.clear()
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(this) {
            leases.clear()
            events.clear()
            gpuBufferLeaseCoordinator.close()
            runtimeFailureContainmentEngine.close()
            frontierSessionRecoveryCoordinator.close()
            captureStageScheduler.close()
            lastSnapshot.set(null)
        }
    }

    private fun saturatingAllocatedBytes(): Long {
        var total = 0L

        leases.values.forEach { lease ->
            val bytes =
                lease.bytes.coerceAtLeast(0L)

            if (
                bytes >
                    Long.MAX_VALUE - total
            ) {
                total = Long.MAX_VALUE
                return@forEach
            }

            total += bytes
        }

        return total
    }

    private fun ensureOpen() {
        check(!closed.get()) { "runtime governor is closed" }
    }

    private fun record(
        nowNanos: Long,
        code: String,
        resourceKind: ResourceKind?,
        bytes: Long,
        recoverable: Boolean,
    ) {
        while (events.size >= maximumEvents) {
            events.removeFirst()
        }
        events.addLast(
            Event(
                timestampNanos = nowNanos.coerceAtLeast(0L),
                code = code,
                resourceKind = resourceKind,
                bytes = bytes.coerceAtLeast(0L),
                recoverable = recoverable,
            ),
        )
    }

    private fun maximum(
        floor: Int,
        value: Int,
    ): Int = min(
        Int.MAX_VALUE,
        max(floor, value),
    )
}
