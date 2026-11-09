package com.nivukx.nox.core.capture

import kotlin.math.max
import kotlin.math.min

/**
 * Bounded runtime fault-management engine for long motion-capture sessions.
 *
 * Recovery is staged and rate-limited. The engine never responds to every transient warning with a
 * full native-session restart, which is a reliable way to turn one hiccup into twenty failures.
 */
class CaptureRuntimeResilienceEngine(
    private val heartbeatTimeoutNanos: Long = 450_000_000L,
    private val maximumRestartAttemptsPerWindow: Int = 3,
    private val restartWindowNanos: Long = 60_000_000_000L,
) : AutoCloseable {

    init {
        require(heartbeatTimeoutNanos in 1L..(Long.MAX_VALUE / 3L))
        require(maximumRestartAttemptsPerWindow in 1..32)
        require(restartWindowNanos > 0L)
    }

    enum class State { NOMINAL, DEGRADED, RECOVERING, PROTECTIVE, BLOCKED }

    enum class Action {
        NONE,
        REDUCE_OPTIONAL_STAGES,
        DRAIN_QUEUES,
        RECYCLE_WORKERS,
        REQUEST_RECALIBRATION,
        RELEASE_STALE_RESOURCES,
        RESTART_NATIVE_SESSION,
        PAUSE_CAPTURE,
        STOP_FOR_SAFETY,
    }

    data class Input(
        val timestampNanos: Long,
        val lastHeartbeatNanos: Long,
        val thermalRisk: Double,
        val memoryPressure: Double,
        val queueAgeMilliseconds: Double,
        val processingMilliseconds: Double,
        val dropRatio: Double,
        val nativeErrorCount: Int,
        val cameraDisconnectCount: Int,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val resourceLeakRisk: Double = 0.0,
        val staleResourceCount: Int = 0,
        val residentResourceBytes: Long = 0L,
        val activeResourceCount: Int = 0,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(lastHeartbeatNanos in 0L..timestampNanos) {
                "Heartbeat timestamp cannot be later than the evaluated frame"
            }
            listOf(
                thermalRisk,
                memoryPressure,
                dropRatio,
                calibrationConfidence,
                synchronizationConfidence,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            require(queueAgeMilliseconds.isFinite() && queueAgeMilliseconds >= 0.0)
            require(processingMilliseconds.isFinite() && processingMilliseconds >= 0.0)
            require(nativeErrorCount >= 0)
            require(cameraDisconnectCount >= 0)
            require(resourceLeakRisk.isFinite() && resourceLeakRisk in 0.0..1.0)
            require(staleResourceCount >= 0)
            require(residentResourceBytes >= 0L)
            require(activeResourceCount >= 0)
        }
    }

    data class Decision(
        val timestampNanos: Long,
        val state: State,
        val action: Action,
        val restartAttemptCount: Int,
        val recoveryBackoffNanos: Long,
        val captureAllowed: Boolean,
        val reason: String,
        val score: Double,
        val warnings: List<String>,
    )

    private val restartTimes = ArrayDeque<Long>()
    private var lastAction = Action.NONE
    private var lastRecoveryTimestampNanos = Long.MIN_VALUE
    private var lastEvaluationTimestampNanos = Long.MIN_VALUE
    private var closed = false

    @Synchronized
    fun evaluate(input: Input): Decision {
        check(!closed)
        require(
            lastEvaluationTimestampNanos == Long.MIN_VALUE ||
                input.timestampNanos >= lastEvaluationTimestampNanos,
        ) {
            "Runtime evidence timestamps must be monotonic"
        }
        lastEvaluationTimestampNanos = input.timestampNanos
        pruneRestartHistory(input.timestampNanos)

        val heartbeatAge = input.timestampNanos - input.lastHeartbeatNanos
        val heartbeatRisk = (
            heartbeatAge.toDouble() / heartbeatTimeoutNanos.toDouble()
        ).coerceIn(0.0, 3.0)

        val computeRisk = max(
            input.queueAgeMilliseconds / 25.0,
            input.processingMilliseconds / 16.0,
        ).coerceIn(0.0, 3.0)
        val thermalMemoryRisk =
            max(input.thermalRisk, input.memoryPressure)
        val resourceRisk =
            max(
                input.resourceLeakRisk,
                if (input.staleResourceCount >= 64) 1.0 else
                    input.staleResourceCount / 64.0,
            ).coerceIn(0.0, 1.0)

        val dataRisk = max(
            input.dropRatio / 0.02,
            max(
                (1.0 - input.calibrationConfidence) / 0.35,
                (1.0 - input.synchronizationConfidence) / 0.35,
            ),
        ).coerceIn(0.0, 3.0)

        val score = (
            1.0 -
                min(
                    1.0,
                    heartbeatRisk * 0.18 +
                        computeRisk * 0.18 +
                        thermalMemoryRisk * 0.30 +
                        dataRisk * 0.18 +
                        resourceRisk * 0.08 +
                        input.nativeErrorCount.coerceAtMost(20) / 20.0 * 0.08 +
                        input.cameraDisconnectCount.coerceAtMost(8) / 8.0 * 0.06,
                )
            ).coerceIn(0.0, 1.0)

        val hardSafety =
            input.thermalRisk >= 0.99 ||
                input.memoryPressure >= 0.995 ||
                heartbeatAge >= heartbeatTimeoutNanos * 3L ||
                resourceRisk >= 0.98

        val blocked =
            hardSafety ||
                (input.nativeErrorCount >= 8 && score < 0.35) ||
                restartTimes.size >= maximumRestartAttemptsPerWindow &&
                (input.nativeErrorCount >= 3 || heartbeatRisk >= 1.4)

        val action = when {
            blocked -> {
                Action.STOP_FOR_SAFETY
            }
            input.thermalRisk >= 0.90 || input.memoryPressure >= 0.92 -> {
                Action.PAUSE_CAPTURE
            }
            resourceRisk >= 0.70 -> {
                Action.RELEASE_STALE_RESOURCES
            }
            heartbeatRisk >= 1.0 || input.nativeErrorCount >= 4 -> {
                if (canRestart(input.timestampNanos)) {
                    restartTimes.addLast(input.timestampNanos)
                    lastRecoveryTimestampNanos = input.timestampNanos
                    Action.RESTART_NATIVE_SESSION
                } else {
                    Action.RECYCLE_WORKERS
                }
            }
            input.queueAgeMilliseconds > 25.0 || input.processingMilliseconds > 16.0 -> {
                Action.DRAIN_QUEUES
            }
            input.dropRatio > 0.02 -> {
                Action.REDUCE_OPTIONAL_STAGES
            }
            input.calibrationConfidence < 0.55 ||
                input.synchronizationConfidence < 0.55 -> {
                Action.REQUEST_RECALIBRATION
            }
            input.nativeErrorCount > 0 -> Action.RECYCLE_WORKERS
            else -> Action.NONE
        }

        val state = when {
            blocked -> State.BLOCKED
            action == Action.PAUSE_CAPTURE ||
                action == Action.STOP_FOR_SAFETY -> State.PROTECTIVE
            action == Action.RESTART_NATIVE_SESSION ||
                action == Action.RECYCLE_WORKERS ||
                action == Action.REQUEST_RECALIBRATION ||
                action == Action.RELEASE_STALE_RESOURCES -> State.RECOVERING
            action != Action.NONE -> State.DEGRADED
            else -> State.NOMINAL
        }

        val warnings = buildList {
            if (heartbeatRisk >= 0.60) add("heartbeat-stale")
            if (computeRisk >= 0.60) add("compute-backpressure")
            if (resourceRisk >= 0.80) add("resource-pressure")
            if (input.dropRatio > 0.02) add("capture-drop-rate-high")
            if (input.calibrationConfidence < 0.55) add("calibration-confidence-low")
            if (input.synchronizationConfidence < 0.55) add("synchronization-confidence-low")
            if (input.resourceLeakRisk >= 0.50) add("resource-leak-risk-rising")
            if (input.staleResourceCount > 0) add("stale-resources-present")
            if (input.activeResourceCount > 0 &&
                input.residentResourceBytes == 0L
            ) add("resource-accounting-mismatch")
            if (restartTimes.size >= maximumRestartAttemptsPerWindow) {
                add("restart-budget-exhausted")
            }
        }

        lastAction = action
        return Decision(
            timestampNanos = input.timestampNanos,
            state = state,
            action = action,
            restartAttemptCount = restartTimes.size,
            recoveryBackoffNanos = recoveryBackoffNanos(restartTimes.size),
            captureAllowed =
                state != State.BLOCKED &&
                    state != State.PROTECTIVE &&
                    action != Action.RESTART_NATIVE_SESSION &&
                    action != Action.REQUEST_RECALIBRATION &&
                    action != Action.RELEASE_STALE_RESOURCES,
            reason = when (action) {
                Action.NONE -> "runtime-within-bounds"
                else -> action.name.lowercase()
            },
            score = score,
            warnings = warnings.distinct(),
        )
    }

    @Synchronized
    fun evaluateWithResourceHealth(
        input: Input,
        sentinel: CaptureResourceLeakSentinel,
        activeWorkUnits: Int,
    ): Decision {
        val resource = sentinel.snapshot(
            timestampNanos = input.timestampNanos,
            activeWorkUnits = activeWorkUnits,
        )
        return evaluate(
            input.copy(
                resourceLeakRisk = resource.leakRisk,
                staleResourceCount = resource.staleResources,
                residentResourceBytes = resource.residentBytes,
                activeResourceCount = resource.activeResources,
            ),
        )
    }

    @Synchronized
    fun lastAction(): Action = lastAction

    @Synchronized
    fun reset() {
        restartTimes.clear()
        lastAction = Action.NONE
        lastRecoveryTimestampNanos = Long.MIN_VALUE
        lastEvaluationTimestampNanos = Long.MIN_VALUE
    }

    @Synchronized
    override fun close() {
        if (closed) return
        // Clear lifecycle state and mark closed under the same monitor so evaluate() cannot
        // slip between reset and shutdown and recreate restart history after shutdown begins.
        restartTimes.clear()
        lastAction = Action.NONE
        lastRecoveryTimestampNanos = Long.MIN_VALUE
        lastEvaluationTimestampNanos = Long.MIN_VALUE
        closed = true
    }

    private fun pruneRestartHistory(now: Long) {
        while (restartTimes.isNotEmpty() &&
            now - restartTimes.first() > restartWindowNanos
        ) {
            restartTimes.removeFirst()
        }
    }

    private fun canRestart(now: Long): Boolean {
        pruneRestartHistory(now)
        if (restartTimes.size >= maximumRestartAttemptsPerWindow) return false
        if (lastRecoveryTimestampNanos == Long.MIN_VALUE) return true
        return now - lastRecoveryTimestampNanos >=
            recoveryBackoffNanos(restartTimes.size)
    }

    private fun recoveryBackoffNanos(attempt: Int): Long =
        when (attempt.coerceIn(0, 8)) {
            0 -> 0L
            1 -> 250_000_000L
            2 -> 1_000_000_000L
            3 -> 3_000_000_000L
            4 -> 8_000_000_000L
            else -> 15_000_000_000L
        }
}
