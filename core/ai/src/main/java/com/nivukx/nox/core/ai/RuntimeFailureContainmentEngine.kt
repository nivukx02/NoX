package com.nivukx.nox.core.ai

import kotlin.math.min
import kotlin.math.pow

/**
 * Bounded runtime fault-containment policy for long-running capture sessions.
 *
 * This engine decides recovery intent; it does not perform restarts itself. The native/runtime
 * owner can map the decision to an actual worker, camera, or GPU lifecycle operation.
 *
 * Design goals:
 *  - distinguish transient faults from repeated faults;
 *  - bound the number of recovery attempts in a time window;
 *  - exponential backoff for repeated failures;
 *  - circuit-break repeated catastrophic failures;
 *  - avoid retaining unbounded fault history;
 *  - keep every decision deterministic from timestamped evidence.
 */
class RuntimeFailureContainmentEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class FaultKind {
        NATIVE_CRASH,
        GPU_DEVICE_LOST,
        CAMERA_STREAM_RESET,
        OUT_OF_MEMORY,
        STALE_RUNTIME_STATE,
        SYNCHRONIZATION_FAILURE,
        RECONSTRUCTION_FAILURE,
        FACE_EYE_FAILURE,
        HAND_TRACKING_FAILURE,
        UNKNOWN,
    }

    enum class Severity {
        INFO,
        WARNING,
        ERROR,
        CRITICAL,
        FATAL,
    }

    enum class CircuitState {
        CLOSED,
        RECOVERING,
        OPEN,
    }

    enum class RecoveryAction {
        NONE,
        RESTART_WORKER,
        REINITIALIZE_CAMERA,
        RESET_GPU_PIPELINE,
        SHED_TRANSIENT_STATE,
        RESET_RECONSTRUCTION_STATE,
        PAUSE_CAPTURE,
        REQUIRE_USER_INTERVENTION,
    }

    data class Config(
        val maximumFaultRecords: Int = 512,
        val faultWindowMilliseconds: Long = 60_000L,
        val criticalWindowMilliseconds: Long = 15_000L,
        val maximumRecoveryAttemptsPerWindow: Int = 4,
        val maximumConsecutiveRecoveryAttempts: Int = 3,
        val baseBackoffMilliseconds: Long = 250L,
        val maximumBackoffMilliseconds: Long = 8_000L,
        val circuitOpenMilliseconds: Long = 10_000L,
        val recoverySuccessDecay: Double = 0.25,
        val warningFaultThreshold: Int = 2,
        val criticalFaultThreshold: Int = 3,
        val sameSignatureCircuitThreshold: Int = 3,
        val sameSignatureCircuitOpenMilliseconds: Long = 4_000L,
    ) {
        init {
            require(maximumFaultRecords > 0)
            require(faultWindowMilliseconds > 0L)
            require(criticalWindowMilliseconds in 1L..faultWindowMilliseconds)
            require(maximumRecoveryAttemptsPerWindow > 0)
            require(maximumConsecutiveRecoveryAttempts > 0)
            require(baseBackoffMilliseconds > 0L)
            require(maximumBackoffMilliseconds >= baseBackoffMilliseconds)
            require(circuitOpenMilliseconds > 0L)
            require(recoverySuccessDecay in 0.0..1.0)
            require(warningFaultThreshold >= 1)
            require(criticalFaultThreshold >= warningFaultThreshold)
            require(sameSignatureCircuitThreshold >= 2)
            require(sameSignatureCircuitOpenMilliseconds > 0L)
        }
    }

    data class Fault(
        val timestampNanos: Long,
        val kind: FaultKind,
        val severity: Severity,
        val signature: String,
        val recoveryAttempt: Int,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(signature.isNotBlank())
            require(recoveryAttempt >= 0)
        }
    }

    data class Decision(
        val timestampNanos: Long,
        val circuitState: CircuitState,
        val action: RecoveryAction,
        val allowed: Boolean,
        val shouldPauseCapture: Boolean,
        val recoveryAttempt: Int,
        val backoffMilliseconds: Long,
        val recentFaultCount: Int,
        val criticalFaultCount: Int,
        val sameSignatureCount: Int,
        val recoveryBudgetRemaining: Int,
        val consecutiveRecoveryAttempts: Int,
        val warnings: List<String>,
    ) {
        val requiresOperatorAction: Boolean
            get() =
                action == RecoveryAction.REQUIRE_USER_INTERVENTION ||
                    circuitState == CircuitState.OPEN

        val safeToContinue: Boolean
            get() =
                allowed &&
                    !shouldPauseCapture &&
                    circuitState != CircuitState.OPEN
    }

    private val faults = ArrayDeque<Fault>()
    private var circuitState = CircuitState.CLOSED
    private var nextRecoveryAllowedTimestampNanos = 0L
    private var consecutiveRecoveryAttempts = 0
    private var recoveryAttemptsInWindow = 0
    private var lastFaultTimestampNanos = 0L
    private var successfulFramesSinceFault = 0
    private var lastDecision: Decision? = null

    @Synchronized
    fun observeFault(
        timestampNanos: Long,
        kind: FaultKind,
        severity: Severity,
        signature: String = kind.name,
    ): Decision {
        require(timestampNanos >= 0L)
        require(signature.isNotBlank())
        prune(timestampNanos)

        val sameSignatureCount =
            faults.count { it.signature == signature }

        val recentFaultCount = faults.size
        val criticalFaultCount =
            faults.count {
                it.severity >= Severity.CRITICAL
            }

        val currentAttempt =
            when (severity) {
                Severity.INFO,
                Severity.WARNING,
                -> 0

                Severity.ERROR,
                Severity.CRITICAL,
                Severity.FATAL,
                ->
                    consecutiveRecoveryAttempts + 1
            }

        faults.addLast(
            Fault(
                timestampNanos = timestampNanos,
                kind = kind,
                severity = severity,
                signature = signature,
                recoveryAttempt = currentAttempt,
            ),
        )
        trimFaults()

        lastFaultTimestampNanos = timestampNanos
        successfulFramesSinceFault = 0

        val projectedCriticalFaultCount =
            criticalFaultCount +
                if (severity >= Severity.CRITICAL) 1 else 0
        val projectedRecoveryAttempts =
            recoveryAttemptsInWindow +
                if (
                    severity == Severity.ERROR ||
                    severity >= Severity.CRITICAL
                ) {
                    1
                } else {
                    0
                }

        val projectedSameSignatureCount =
            sameSignatureCount + 1

        val shouldOpenCircuit =
            severity == Severity.FATAL ||
                projectedCriticalFaultCount >=
                    config.criticalFaultThreshold ||
                projectedRecoveryAttempts >=
                    config.maximumRecoveryAttemptsPerWindow ||
                consecutiveRecoveryAttempts >=
                    config.maximumConsecutiveRecoveryAttempts ||
                (
                    severity >= Severity.ERROR &&
                        projectedSameSignatureCount >=
                            config.sameSignatureCircuitThreshold
                    )

        val action =
            if (shouldOpenCircuit) {
                circuitState = CircuitState.OPEN
                val openWindowMilliseconds =
                    if (
                        projectedSameSignatureCount >=
                            config.sameSignatureCircuitThreshold
                    ) {
                        config.sameSignatureCircuitOpenMilliseconds
                    } else {
                        config.circuitOpenMilliseconds
                    }

                nextRecoveryAllowedTimestampNanos =
                    timestampNanos +
                        openWindowMilliseconds *
                        1_000_000L
                RecoveryAction.REQUIRE_USER_INTERVENTION
            } else {
                circuitState = CircuitState.RECOVERING
                recoveryAttemptsInWindow += 1
                consecutiveRecoveryAttempts += 1
                nextRecoveryAllowedTimestampNanos =
                    timestampNanos +
                        backoffMilliseconds(consecutiveRecoveryAttempts) *
                        1_000_000L
                actionFor(kind, severity)
            }

        val decision =
            buildDecision(
                timestampNanos = timestampNanos,
                action = action,
                sameSignatureCount = sameSignatureCount + 1,
                recentFaultCount = recentFaultCount + 1,
                criticalFaultCount = projectedCriticalFaultCount,
            )

        lastDecision = decision
        return decision
    }

    @Synchronized
    fun observeSuccess(
        timestampNanos: Long,
        frames: Int = 1,
    ): Decision {
        require(timestampNanos >= 0L)
        require(frames >= 1)
        prune(timestampNanos)

        successfulFramesSinceFault += frames

        if (circuitState == CircuitState.RECOVERING) {
            val decayThreshold =
                maxOf(
                    1,
                    (1.0 / config.recoverySuccessDecay)
                        .toInt()
                        .coerceAtMost(16),
                )

            if (successfulFramesSinceFault >= decayThreshold) {
                consecutiveRecoveryAttempts =
                    maxOf(
                        0,
                        consecutiveRecoveryAttempts - 1,
                    )
                successfulFramesSinceFault = 0
            }

            if (consecutiveRecoveryAttempts == 0) {
                circuitState = CircuitState.CLOSED
                recoveryAttemptsInWindow =
                    maxOf(
                        0,
                        recoveryAttemptsInWindow - 1,
                    )
            }
        }

        val decision =
            buildDecision(
                timestampNanos = timestampNanos,
                action = RecoveryAction.NONE,
                sameSignatureCount = 0,
                recentFaultCount = faults.size,
                criticalFaultCount =
                    faults.count {
                        it.severity >= Severity.CRITICAL
                    },
            )
        lastDecision = decision
        return decision
    }

    @Synchronized
    fun tick(timestampNanos: Long): Decision {
        require(timestampNanos >= 0L)
        prune(timestampNanos)

        if (
            circuitState == CircuitState.OPEN &&
            timestampNanos >= nextRecoveryAllowedTimestampNanos
        ) {
            circuitState = CircuitState.RECOVERING
            consecutiveRecoveryAttempts = 0
            successfulFramesSinceFault = 0
        }

        val action =
            if (circuitState == CircuitState.OPEN) {
                RecoveryAction.REQUIRE_USER_INTERVENTION
            } else if (
                circuitState == CircuitState.RECOVERING &&
                timestampNanos < nextRecoveryAllowedTimestampNanos
            ) {
                RecoveryAction.PAUSE_CAPTURE
            } else {
                RecoveryAction.NONE
            }

        val decision =
            buildDecision(
                timestampNanos = timestampNanos,
                action = action,
                sameSignatureCount = 0,
                recentFaultCount = faults.size,
                criticalFaultCount =
                    faults.count {
                        it.severity >= Severity.CRITICAL
                    },
            )
        lastDecision = decision
        return decision
    }

    @Synchronized
    fun latestDecision(): Decision? = lastDecision

    @Synchronized
    fun recentFaults(limit: Int = 32): List<Fault> {
        require(limit >= 0)
        return faults.takeLast(
            min(limit, faults.size),
        )
    }

    @Synchronized
    fun reset() {
        faults.clear()
        circuitState = CircuitState.CLOSED
        nextRecoveryAllowedTimestampNanos = 0L
        consecutiveRecoveryAttempts = 0
        recoveryAttemptsInWindow = 0
        lastFaultTimestampNanos = 0L
        successfulFramesSinceFault = 0
        lastDecision = null
    }

    @Synchronized
    override fun close() {
        reset()
    }

    private fun actionFor(
        kind: FaultKind,
        severity: Severity,
    ): RecoveryAction =
        when (kind) {
            FaultKind.NATIVE_CRASH ->
                RecoveryAction.RESTART_WORKER

            FaultKind.GPU_DEVICE_LOST ->
                RecoveryAction.RESET_GPU_PIPELINE

            FaultKind.CAMERA_STREAM_RESET ->
                RecoveryAction.REINITIALIZE_CAMERA

            FaultKind.OUT_OF_MEMORY ->
                RecoveryAction.SHED_TRANSIENT_STATE

            FaultKind.STALE_RUNTIME_STATE ->
                RecoveryAction.RESET_RECONSTRUCTION_STATE

            FaultKind.SYNCHRONIZATION_FAILURE ->
                RecoveryAction.RESET_RECONSTRUCTION_STATE

            FaultKind.RECONSTRUCTION_FAILURE ->
                RecoveryAction.RESET_RECONSTRUCTION_STATE

            FaultKind.FACE_EYE_FAILURE ->
                RecoveryAction.RESTART_WORKER

            FaultKind.HAND_TRACKING_FAILURE ->
                RecoveryAction.RESTART_WORKER

            FaultKind.UNKNOWN ->
                if (severity >= Severity.CRITICAL) {
                    RecoveryAction.PAUSE_CAPTURE
                } else {
                    RecoveryAction.NONE
                }
        }

    private fun buildDecision(
        timestampNanos: Long,
        action: RecoveryAction,
        sameSignatureCount: Int,
        recentFaultCount: Int,
        criticalFaultCount: Int,
    ): Decision {
        val warnings = buildList {
            if (recentFaultCount >= config.warningFaultThreshold) {
                add("runtime-fault-rate-elevated")
            }
            if (criticalFaultCount > 0) {
                add("runtime-critical-fault-present")
            }
            if (sameSignatureCount >= 2) {
                add("runtime-fault-signature-repeating")
            }
            if (consecutiveRecoveryAttempts > 0) {
                add("runtime-recovery-active")
            }
            if (circuitState == CircuitState.OPEN) {
                add("runtime-circuit-open")
            }
            if (recoveryAttemptsInWindow >=
                config.maximumRecoveryAttemptsPerWindow
            ) {
                add("runtime-recovery-budget-exhausted")
            }
        }.distinct()

        val budgetRemaining =
            (
                config.maximumRecoveryAttemptsPerWindow -
                    recoveryAttemptsInWindow
                ).coerceAtLeast(0)

        return Decision(
            timestampNanos = timestampNanos,
            circuitState = circuitState,
            action = action,
            allowed =
                circuitState != CircuitState.OPEN &&
                    timestampNanos >= nextRecoveryAllowedTimestampNanos,
            shouldPauseCapture =
                action == RecoveryAction.PAUSE_CAPTURE ||
                    action == RecoveryAction.REQUIRE_USER_INTERVENTION,
            recoveryAttempt = consecutiveRecoveryAttempts,
            backoffMilliseconds =
                if (timestampNanos <
                    nextRecoveryAllowedTimestampNanos
                ) {
                    (
                        nextRecoveryAllowedTimestampNanos -
                            timestampNanos
                        ) / 1_000_000L
                } else {
                    0L
                },
            recentFaultCount = recentFaultCount,
            criticalFaultCount = criticalFaultCount,
            sameSignatureCount = sameSignatureCount,
            recoveryBudgetRemaining = budgetRemaining,
            consecutiveRecoveryAttempts = consecutiveRecoveryAttempts,
            warnings = warnings,
        )
    }

    private fun backoffMilliseconds(
        attempt: Int,
    ): Long {
        val multiplier =
            2.0.pow(
                (attempt - 1).coerceAtLeast(0).coerceAtMost(8)
                    .toDouble(),
            )
        return (
            config.baseBackoffMilliseconds * multiplier
            ).toLong().coerceAtMost(
                config.maximumBackoffMilliseconds,
            )
    }

    private fun prune(timestampNanos: Long) {
        val cutoff =
            timestampNanos -
                config.faultWindowMilliseconds *
                1_000_000L

        while (
            faults.isNotEmpty() &&
            faults.first().timestampNanos < cutoff
        ) {
            faults.removeFirst()
        }

        if (
            lastFaultTimestampNanos > 0L &&
            timestampNanos -
            lastFaultTimestampNanos >
            config.criticalWindowMilliseconds *
            1_000_000L &&
            consecutiveRecoveryAttempts > 0
        ) {
            consecutiveRecoveryAttempts =
                maxOf(
                    0,
                    consecutiveRecoveryAttempts - 1,
                )
        }

        if (
            timestampNanos >
            lastFaultTimestampNanos +
            config.faultWindowMilliseconds *
            1_000_000L
        ) {
            recoveryAttemptsInWindow = 0
        }
    }

    private fun trimFaults() {
        while (faults.size > config.maximumFaultRecords) {
            faults.removeFirst()
        }
    }
}
