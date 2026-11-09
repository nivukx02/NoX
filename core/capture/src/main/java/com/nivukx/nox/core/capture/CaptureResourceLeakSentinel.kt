package com.nivukx.nox.core.capture

import kotlin.math.max
import kotlin.math.min

/**
 * Production resource-retention sentinel for long-running motion-capture sessions.
 *
 * This is a logical ownership layer, not a substitute for Android/native profilers. Its purpose is
 * to make resource lifetime failures observable at the exact subsystem boundary where they happen.
 *
 * Hardened invariants:
 * - every acquire has exactly one release;
 * - duplicate ids never overwrite a live lease;
 * - shutdown accounts for every outstanding lease;
 * - per-type and per-owner residency remain bounded and auditable;
 * - all byte counters are overflow-safe;
 * - history is bounded to one minute;
 * - timestamps are monotonic;
 * - long leases are explicitly visible instead of being silently treated as healthy.
 */
class CaptureResourceLeakSentinel(
    private val maximumTrackedResources: Int = 65_536,
    private val staleLeaseThresholdMilliseconds: Long = 5_000L,
    private val maximumBytesPerOwner: Long = 512L * 1024L * 1024L,
) : AutoCloseable {

    enum class ResourceType {
        CAMERA_SESSION,
        NATIVE_BUFFER,
        GPU_SURFACE,
        TEMPORAL_TRACK,
        FACE_TRACK,
        EYE_TRACK,
        HAND_TRACK,
        MESH_BLOCK,
        TSDF_BLOCK,
        CALIBRATION_CONTEXT,
        NETWORK_PEER,
        EXPORT_WRITER,
        WORKER,
        OTHER,
    }

    data class ResourceId(
        val type: ResourceType,
        val value: String,
    ) {
        init {
            require(value.isNotBlank())
        }
    }

    data class Lease(
        val id: ResourceId,
        val acquiredAtNanos: Long,
        val bytes: Long,
        val owner: String,
        val sequence: Long = 0L,
    ) {
        init {
            require(acquiredAtNanos >= 0L)
            require(bytes >= 0L)
            require(owner.isNotBlank())
            require(sequence >= 0L)
        }
    }

    data class TypeSnapshot(
        val activeResources: Int,
        val residentBytes: Long,
        val peakResidentBytes: Long,
        val staleResources: Int,
    )

    data class OwnerSnapshot(
        val activeResources: Int,
        val residentBytes: Long,
        val oldestLeaseAgeMilliseconds: Long,
    )

    data class Snapshot(
        val timestampNanos: Long,
        val activeResources: Int,
        val residentBytes: Long,
        val peakResidentBytes: Long,
        val staleResources: Int,
        val oldestLeaseAgeMilliseconds: Long,
        val acquisitionsPerMinute: Double,
        val releasesPerMinute: Double,
        val failedReleases: Long,
        val duplicateAcquisitions: Long,
        val rejectedAcquisitions: Long,
        val shutdownLeaks: Int,
        val leakRisk: Double,
        val warnings: List<String>,
        val byType: Map<ResourceType, TypeSnapshot> = emptyMap(),
        val byOwner: Map<String, OwnerSnapshot> = emptyMap(),
    )

    private val leases = LinkedHashMap<ResourceId, Lease>()
    private val acquisitionTimestamps = ArrayDeque<Long>()
    private val releaseTimestamps = ArrayDeque<Long>()
    private val ownerBytes = HashMap<String, Long>()

    private var residentBytes = 0L
    private var peakResidentBytes = 0L
    private var failedReleases = 0L
    private var duplicateAcquisitions = 0L
    private var rejectedAcquisitions = 0L
    private var shutdownLeaks = 0
    private var acquisitionSequence = 0L
    private var lastTimestampNanos = 0L
    private var closed = false

    @Synchronized
    fun acquire(
        id: ResourceId,
        timestampNanos: Long,
        bytes: Long,
        owner: String,
    ): Boolean {
        check(!closed) { "Leak sentinel is closed" }
        require(timestampNanos >= 0L)
        require(bytes >= 0L)
        require(owner.isNotBlank())

        monotonic(timestampNanos)

        if (leases.containsKey(id)) {
            duplicateAcquisitions += 1L
            return false
        }

        if (leases.size >= maximumTrackedResources) {
            rejectedAcquisitions += 1L
            return false
        }

        val currentOwnerBytes = ownerBytes[owner] ?: 0L
        if (bytes > Long.MAX_VALUE - currentOwnerBytes) {
            rejectedAcquisitions = safeIncrement(rejectedAcquisitions)
            return false
        }
        val projectedOwnerBytes =
            currentOwnerBytes + bytes
        if (projectedOwnerBytes > maximumBytesPerOwner) {
            rejectedAcquisitions = safeIncrement(rejectedAcquisitions)
            return false
        }

        acquisitionSequence = safeIncrement(acquisitionSequence)
        leases[id] = Lease(
            id = id,
            acquiredAtNanos = timestampNanos,
            bytes = bytes,
            owner = owner,
            sequence = acquisitionSequence,
        )
        if (bytes > Long.MAX_VALUE - residentBytes) {
            rejectedAcquisitions = safeIncrement(rejectedAcquisitions)
            leases.remove(id)
            return false
        }
        residentBytes += bytes
        ownerBytes[owner] = projectedOwnerBytes
        peakResidentBytes = max(peakResidentBytes, residentBytes)
        acquisitionTimestamps.addLast(timestampNanos)
        pruneHistory(timestampNanos)

        return true
    }

    @Synchronized
    fun release(
        id: ResourceId,
        timestampNanos: Long,
    ): Boolean {
        check(!closed) { "Leak sentinel is closed" }
        require(timestampNanos >= 0L)

        monotonic(timestampNanos)

        val lease = leases.remove(id)
            ?: run {
                failedReleases = safeIncrement(failedReleases)
                releaseTimestamps.addLast(timestampNanos)
                pruneHistory(timestampNanos)
                return false
            }

        residentBytes = max(0L, residentBytes - lease.bytes)
        decrementOwnerBytes(lease.owner, lease.bytes)
        releaseTimestamps.addLast(timestampNanos)
        pruneHistory(timestampNanos)
        return true
    }

    @Synchronized
    fun snapshot(
        timestampNanos: Long,
        activeWorkUnits: Int = 0,
    ): Snapshot {
        check(!closed)
        require(timestampNanos >= 0L)
        require(activeWorkUnits >= 0)

        monotonic(timestampNanos)
        pruneHistory(timestampNanos)

        val byTypeMutable = linkedMapOf<ResourceType, MutableTypeState>()
        val byOwnerMutable = linkedMapOf<String, MutableOwnerState>()

        var stale = 0
        var oldestAge = 0L

        for (lease in leases.values) {
            val age = (timestampNanos - lease.acquiredAtNanos).coerceAtLeast(0L)
            val ageMs = age / 1_000_000L
            oldestAge = max(oldestAge, ageMs)

            if (ageMs >= staleLeaseThresholdMilliseconds) {
                stale += 1
            }

            val typeState = byTypeMutable.getOrPut(lease.id.type) {
                MutableTypeState()
            }
            typeState.activeResources += 1
            typeState.residentBytes = safeAdd(typeState.residentBytes, lease.bytes)
            typeState.peakResidentBytes = max(
                typeState.peakResidentBytes,
                typeState.residentBytes,
            )
            if (ageMs >= staleLeaseThresholdMilliseconds) {
                typeState.staleResources += 1
            }

            val ownerState = byOwnerMutable.getOrPut(lease.owner) {
                MutableOwnerState()
            }
            ownerState.activeResources += 1
            ownerState.residentBytes = safeAdd(
                ownerState.residentBytes,
                lease.bytes,
            )
            ownerState.oldestLeaseAgeMilliseconds = max(
                ownerState.oldestLeaseAgeMilliseconds,
                ageMs,
            )
        }

        val churnMinutes = historyWindowMinutes(timestampNanos)
        val acquireRate = ratePerMinute(
            acquisitionTimestamps.size,
            churnMinutes,
        )
        val releaseRate = ratePerMinute(
            releaseTimestamps.size,
            churnMinutes,
        )

        val imbalance = (
            acquisitionTimestamps.size -
                releaseTimestamps.size
            ).coerceAtLeast(0)
        val imbalanceRisk = if (acquisitionTimestamps.isEmpty()) {
            0.0
        } else {
            (
                imbalance.toDouble() /
                    acquisitionTimestamps.size.toDouble()
            ).coerceIn(0.0, 1.0)
        }

        val staleRisk = (
            stale.toDouble() /
                max(1, leases.size).toDouble()
        ).coerceIn(0.0, 1.0)

        val pressureRisk =
            if (activeWorkUnits <= 0) {
                if (residentBytes == 0L) 0.0 else 0.45
            } else {
                (
                    residentBytes.toDouble() /
                        max(
                            1.0,
                            activeWorkUnits.toDouble() *
                                4_194_304.0,
                        )
                    ).coerceIn(0.0, 1.0)
            }

        val typeConcentrationRisk = byTypeMutable.values
            .maxOfOrNull { state ->
                if (residentBytes <= 0L) {
                    0.0
                } else {
                    state.residentBytes.toDouble() /
                        residentBytes.toDouble()
                }
            } ?: 0.0

        val ownerConcentrationRisk = byOwnerMutable.values
            .maxOfOrNull { state ->
                if (residentBytes <= 0L) {
                    0.0
                } else {
                    state.residentBytes.toDouble() /
                        residentBytes.toDouble()
                }
            } ?: 0.0

        val risk = (
            imbalanceRisk * 0.28 +
                staleRisk * 0.27 +
                pressureRisk * 0.16 +
                typeConcentrationRisk * 0.06 +
                ownerConcentrationRisk * 0.06 +
                if (failedReleases > 0L) 0.07 else 0.0 +
                if (duplicateAcquisitions > 0L) 0.06 else 0.0 +
                if (rejectedAcquisitions > 0L) 0.04 else 0.0 +
                if (shutdownLeaks > 0) 0.04 else 0.0
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (failedReleases > 0L) add("release-without-active-lease")
            if (duplicateAcquisitions > 0L) add("duplicate-resource-acquisition")
            if (rejectedAcquisitions > 0L) add("resource-acquisition-rejected")
            if (stale > 0) add("stale-resource-leases")
            if (imbalance > max(8, leases.size / 4)) {
                add("acquire-release-imbalance")
            }
            if (residentBytes > 0L && activeWorkUnits == 0) {
                add("resident-bytes-without-active-work")
            }
            if (byOwnerMutable.any { it.value.residentBytes >= maximumBytesPerOwner }) {
                add("owner-memory-budget-saturated")
            }
            if (risk >= 0.75) add("resource-leak-risk-high")
            if (risk >= 0.90) add("resource-leak-risk-critical")
        }.distinct()

        val byType = byTypeMutable.mapValues { (_, state) ->
            TypeSnapshot(
                activeResources = state.activeResources,
                residentBytes = state.residentBytes,
                peakResidentBytes = state.peakResidentBytes,
                staleResources = state.staleResources,
            )
        }

        val byOwner = byOwnerMutable.mapValues { (_, state) ->
            OwnerSnapshot(
                activeResources = state.activeResources,
                residentBytes = state.residentBytes,
                oldestLeaseAgeMilliseconds =
                    state.oldestLeaseAgeMilliseconds,
            )
        }

        return Snapshot(
            timestampNanos = timestampNanos,
            activeResources = leases.size,
            residentBytes = residentBytes,
            peakResidentBytes = peakResidentBytes,
            staleResources = stale,
            oldestLeaseAgeMilliseconds = oldestAge,
            acquisitionsPerMinute = acquireRate,
            releasesPerMinute = releaseRate,
            failedReleases = failedReleases,
            duplicateAcquisitions = duplicateAcquisitions,
            rejectedAcquisitions = rejectedAcquisitions,
            shutdownLeaks = shutdownLeaks,
            leakRisk = risk,
            warnings = warnings,
            byType = byType,
            byOwner = byOwner,
        )
    }

    @Synchronized
    fun activeLeases(): List<Lease> {
        check(!closed)
        return leases.values.toList()
    }

    @Synchronized
    fun activeLeases(
        type: ResourceType,
    ): List<Lease> {
        check(!closed)
        return leases.values.filter { it.id.type == type }
    }

    @Synchronized
    fun activeLeasesForOwner(
        owner: String,
    ): List<Lease> {
        check(!closed)
        require(owner.isNotBlank())
        return leases.values.filter { it.owner == owner }
    }

    @Synchronized
    fun releaseAll(
        timestampNanos: Long,
    ): Int {
        check(!closed)
        require(timestampNanos >= 0L)
        monotonic(timestampNanos)

        val released = leases.size
        if (released > 0) {
            repeat(released) {
                releaseTimestamps.addLast(timestampNanos)
            }
        }
        leases.clear()
        ownerBytes.clear()
        residentBytes = 0L
        pruneHistory(timestampNanos)
        return released
    }

    @Synchronized
    fun counters(): Map<String, Long> = mapOf(
        "active_resources" to leases.size.toLong(),
        "resident_bytes" to residentBytes,
        "peak_resident_bytes" to peakResidentBytes,
        "failed_releases" to failedReleases,
        "duplicate_acquisitions" to duplicateAcquisitions,
        "rejected_acquisitions" to rejectedAcquisitions,
        "shutdown_leaks" to shutdownLeaks.toLong(),
    )

    override fun close() {
        synchronized(this) {
            if (closed) return
            shutdownLeaks = leases.size
            if (shutdownLeaks > 0) {
                repeat(shutdownLeaks) {
                    releaseTimestamps.addLast(lastTimestampNanos)
                }
            }
            leases.clear()
            ownerBytes.clear()
            acquisitionTimestamps.clear()
            releaseTimestamps.clear()
            residentBytes = 0L
            closed = true
        }
    }

    private fun decrementOwnerBytes(
        owner: String,
        bytes: Long,
    ) {
        val current = ownerBytes[owner] ?: return
        val next = max(0L, current - bytes)
        if (next == 0L) {
            ownerBytes.remove(owner)
        } else {
            ownerBytes[owner] = next
        }
    }

    private fun ratePerMinute(
        count: Int,
        minutes: Double,
    ): Double =
        if (minutes <= 0.0) 0.0 else count / minutes

    private fun monotonic(timestampNanos: Long) {
        require(timestampNanos >= lastTimestampNanos) {
            "Resource timestamps must be monotonic"
        }
        lastTimestampNanos = timestampNanos
    }

    private fun pruneHistory(now: Long) {
        val window = 60_000_000_000L
        while (
            acquisitionTimestamps.isNotEmpty() &&
            now - acquisitionTimestamps.first() > window
        ) {
            acquisitionTimestamps.removeFirst()
        }
        while (
            releaseTimestamps.isNotEmpty() &&
            now - releaseTimestamps.first() > window
        ) {
            releaseTimestamps.removeFirst()
        }
    }

    private fun historyWindowMinutes(now: Long): Double {
        val oldest = minOf(
            acquisitionTimestamps.firstOrNull() ?: now,
            releaseTimestamps.firstOrNull() ?: now,
        )
        return max(
            1.0,
            (now - oldest).coerceAtLeast(0L) / 60_000_000_000.0,
        )
    }

    private fun safeAdd(
        a: Long,
        b: Long,
    ): Long =
        if (Long.MAX_VALUE - a < b) Long.MAX_VALUE else a + b

    private fun safeIncrement(
        value: Long,
    ): Long =
        if (value == Long.MAX_VALUE) Long.MAX_VALUE else value + 1L

    private class MutableTypeState {
        var activeResources: Int = 0
        var residentBytes: Long = 0L
        var peakResidentBytes: Long = 0L
        var staleResources: Int = 0
    }

    private class MutableOwnerState {
        var activeResources: Int = 0
        var residentBytes: Long = 0L
        var oldestLeaseAgeMilliseconds: Long = 0L
    }
}
