package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min

/**
 * Bounded ownership coordinator for camera -> GPU buffer lifetimes.
 *
 * The coordinator is intentionally independent from Android framework classes. Native/JNI glue can
 * translate an AHardwareBuffer or Vulkan allocation into an integer slot and use this state machine
 * to enforce one rule that production pipelines cannot compromise: a buffer must never be released
 * to the camera pool while a newer GPU generation still owns it.
 */
class GpuBufferLeaseCoordinator(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class State {
        ACQUIRED,
        GPU_SUBMITTED,
        GPU_COMPLETED,
        RELEASED,
        ABORTED,
        STALE,
    }

    enum class ReleaseResult {
        RELEASED,
        WAITING_FOR_GPU,
        STALE_GENERATION,
        INVALID_TOKEN,
        ALREADY_RELEASED,
        UNKNOWN_SLOT,
    }

    data class Config(
        val maximumSlots: Int = 32,
        val maximumInFlight: Int = 12,
        val maximumLeaseAgeMilliseconds: Double = 250.0,
        val watchdogGraceMilliseconds: Double = 100.0,
        val requireGenerationMatch: Boolean = true,
        val requireGpuCompletionBeforeRelease: Boolean = true,
    ) {
        init {
            require(maximumSlots > 0)
            require(maximumInFlight in 1..maximumSlots)
            require(maximumLeaseAgeMilliseconds > 0.0)
            require(watchdogGraceMilliseconds >= 0.0)
        }
    }

    data class LeaseToken(
        val slot: Int,
        val generation: Long,
        val valid: Boolean,
    )

    data class Lease(
        val token: LeaseToken,
        val acquiredTimestampNanos: Long,
        val state: State,
        val gpuSubmissionSequence: Long = 0L,
    )

    data class Telemetry(
        val capacity: Int,
        val activeLeases: Int,
        val inFlightGpuLeases: Int,
        val completedWaitingRelease: Int,
        val releaseCount: Long,
        val staleGenerationReleases: Long,
        val invalidReleases: Long,
        val doubleReleaseAttempts: Long,
        val watchdogAborts: Long,
        val forcedStaleLeases: Long,
        val peakActiveLeases: Int,
        val peakGpuInFlight: Int,
        val admissionRejected: Long,
        val oldestLeaseAgeMilliseconds: Double,
        val averageLeaseAgeMilliseconds: Double,
    )

    private data class Slot(
        var generation: Long = 0L,
        var state: State = State.RELEASED,
        var acquiredTimestampNanos: Long = Long.MIN_VALUE,
        var gpuSubmissionSequence: Long = 0L,
        var gpuSubmittedTimestampNanos: Long = Long.MIN_VALUE,
    )

    private val slots = Array(config.maximumSlots) { Slot() }
    private var sequence: Long = 0L
    private var releaseCount = 0L
    private var staleGenerationReleases = 0L
    private var invalidReleases = 0L
    private var doubleReleaseAttempts = 0L
    private var watchdogAborts = 0L
    private var forcedStaleLeases = 0L
    private var peakActiveLeases = 0
    private var peakGpuInFlight = 0
    private var admissionRejected = 0L

    @Synchronized
    fun acquire(timestampNanos: Long): LeaseToken? {
        require(timestampNanos >= 0L)

        val active = slots.count {
            it.state != State.RELEASED &&
                it.state != State.STALE &&
                it.state != State.ABORTED
        }
        if (active >= config.maximumInFlight) {
            admissionRejected += 1
            return null
        }

        val index = slots.indexOfFirst {
            it.state == State.RELEASED ||
                it.state == State.STALE ||
                it.state == State.ABORTED
        }
        if (index < 0) {
            admissionRejected += 1
            return null
        }

        val slot = slots[index]
        slot.generation =
            nextGeneration(slot.generation)
        slot.state = State.ACQUIRED
        slot.acquiredTimestampNanos = timestampNanos
        slot.gpuSubmissionSequence = 0L
        slot.gpuSubmittedTimestampNanos = Long.MIN_VALUE
        sequence += 1

        val activeAfter = active + 1
        peakActiveLeases = max(peakActiveLeases, activeAfter)

        return LeaseToken(
            slot = index,
            generation = slot.generation,
            valid = true,
        )
    }

    @Synchronized
    fun markGpuSubmitted(
        token: LeaseToken,
        timestampNanos: Long,
    ): Boolean {
        require(timestampNanos >= 0L)
        val slot = resolve(token) ?: return false

        if (slot.state != State.ACQUIRED) {
            return false
        }

        slot.state = State.GPU_SUBMITTED
        slot.gpuSubmissionSequence = sequence + 1
        slot.gpuSubmittedTimestampNanos = timestampNanos
        sequence += 1

        val inflight = slots.count {
            it.state == State.GPU_SUBMITTED
        }
        peakGpuInFlight = max(peakGpuInFlight, inflight)
        return true
    }

    @Synchronized
    fun markGpuCompleted(
        token: LeaseToken,
        completionTimestampNanos: Long,
    ): Boolean {
        require(completionTimestampNanos >= 0L)
        val slot = resolve(token) ?: return false

        return when (slot.state) {
            State.GPU_SUBMITTED -> {
                slot.state = State.GPU_COMPLETED
                true
            }

            State.GPU_COMPLETED -> true
            else -> false
        }
    }

    @Synchronized
    fun release(
        token: LeaseToken,
        timestampNanos: Long,
        force: Boolean = false,
    ): ReleaseResult {
        require(timestampNanos >= 0L)

        val slotIndex = token.slot
        if (!token.valid || slotIndex !in slots.indices) {
            invalidReleases += 1
            return ReleaseResult.INVALID_TOKEN
        }

        val slot = slots[slotIndex]

        if (
            config.requireGenerationMatch &&
            token.generation != slot.generation
        ) {
            staleGenerationReleases += 1
            return ReleaseResult.STALE_GENERATION
        }

        when (slot.state) {
            State.RELEASED -> {
                doubleReleaseAttempts += 1
                return ReleaseResult.ALREADY_RELEASED
            }

            State.STALE,
            State.ABORTED,
            -> {
                doubleReleaseAttempts += 1
                return ReleaseResult.ALREADY_RELEASED
            }

            State.GPU_SUBMITTED -> {
                if (
                    config.requireGpuCompletionBeforeRelease &&
                    !force
                ) {
                    return ReleaseResult.WAITING_FOR_GPU
                }
            }

            State.ACQUIRED -> {
                // No GPU work was submitted. Direct release is safe.
            }

            State.GPU_COMPLETED -> Unit
        }

        slot.state = State.RELEASED
        slot.acquiredTimestampNanos = Long.MIN_VALUE
        slot.gpuSubmissionSequence = 0L
        slot.gpuSubmittedTimestampNanos = Long.MIN_VALUE
        releaseCount += 1
        return ReleaseResult.RELEASED
    }

    @Synchronized
    fun abort(
        token: LeaseToken,
        timestampNanos: Long,
    ): Boolean {
        require(timestampNanos >= 0L)
        val slot = resolve(token) ?: return false
        if (slot.state == State.RELEASED) {
            return false
        }

        slot.state = State.ABORTED
        forcedStaleLeases += 1
        return true
    }

    @Synchronized
    fun watchdogSweep(
        timestampNanos: Long,
    ): Int {
        require(timestampNanos >= 0L)
        var aborted = 0

        slots.forEach { slot ->
            if (
                slot.state == State.RELEASED ||
                slot.state == State.ABORTED ||
                slot.state == State.STALE ||
                slot.acquiredTimestampNanos == Long.MIN_VALUE
            ) {
                return@forEach
            }

            val ageMillis =
                (timestampNanos - slot.acquiredTimestampNanos)
                    .coerceAtLeast(0L) / 1_000_000.0

            val hardLimit =
                config.maximumLeaseAgeMilliseconds +
                    config.watchdogGraceMilliseconds

            if (ageMillis > hardLimit) {
                slot.state = State.ABORTED
                watchdogAborts += 1
                forcedStaleLeases += 1
                aborted += 1
            }
        }

        return aborted
    }

    @Synchronized
    fun markGenerationStale(
        slotIndex: Int,
        generation: Long,
    ): Boolean {
        if (slotIndex !in slots.indices) {
            invalidReleases += 1
            return false
        }

        val slot = slots[slotIndex]
        if (slot.generation != generation) {
            staleGenerationReleases += 1
            return false
        }

        if (slot.state == State.RELEASED) {
            return false
        }

        slot.state = State.STALE
        forcedStaleLeases += 1
        return true
    }

    @Synchronized
    fun getLease(token: LeaseToken): Lease? =
        resolve(token)?.let { slot ->
            Lease(
                token = token,
                acquiredTimestampNanos =
                    slot.acquiredTimestampNanos.coerceAtLeast(0L),
                state = slot.state,
                gpuSubmissionSequence =
                    slot.gpuSubmissionSequence,
            )
        }

    @Synchronized
    fun telemetry(timestampNanos: Long): Telemetry {
        require(timestampNanos >= 0L)

        val active = slots.filter {
            it.state != State.RELEASED &&
                it.state != State.ABORTED &&
                it.state != State.STALE
        }
        val inFlight = active.count {
            it.state == State.GPU_SUBMITTED
        }
        val completed = active.count {
            it.state == State.GPU_COMPLETED
        }

        val ages = active.map {
            (
                timestampNanos - it.acquiredTimestampNanos
                ).coerceAtLeast(0L) / 1_000_000.0
        }

        return Telemetry(
            capacity = slots.size,
            activeLeases = active.size,
            inFlightGpuLeases = inFlight,
            completedWaitingRelease = completed,
            releaseCount = releaseCount,
            staleGenerationReleases = staleGenerationReleases,
            invalidReleases = invalidReleases,
            doubleReleaseAttempts = doubleReleaseAttempts,
            watchdogAborts = watchdogAborts,
            forcedStaleLeases = forcedStaleLeases,
            peakActiveLeases = peakActiveLeases,
            peakGpuInFlight = peakGpuInFlight,
            admissionRejected = admissionRejected,
            oldestLeaseAgeMilliseconds =
                ages.maxOrNull() ?: 0.0,
            averageLeaseAgeMilliseconds =
                ages.averageOrZero(),
        )
    }

    @Synchronized
    fun reset() {
        slots.forEach {
            it.state = State.RELEASED
            it.acquiredTimestampNanos = Long.MIN_VALUE
            it.gpuSubmissionSequence = 0L
            it.gpuSubmittedTimestampNanos = Long.MIN_VALUE
        }
        sequence = 0L
        releaseCount = 0L
        staleGenerationReleases = 0L
        invalidReleases = 0L
        doubleReleaseAttempts = 0L
        watchdogAborts = 0L
        forcedStaleLeases = 0L
        peakActiveLeases = 0
        peakGpuInFlight = 0
        admissionRejected = 0L
    }

    @Synchronized
    override fun close() {
        reset()
    }

    private fun resolve(token: LeaseToken): Slot? {
        if (!token.valid || token.slot !in slots.indices) {
            invalidReleases += 1
            return null
        }

        val slot = slots[token.slot]

        if (
            config.requireGenerationMatch &&
            token.generation != slot.generation
        ) {
            staleGenerationReleases += 1
            return null
        }

        return slot
    }

    private fun nextGeneration(previous: Long): Long =
        if (previous == Long.MAX_VALUE) 1L else previous + 1L

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
