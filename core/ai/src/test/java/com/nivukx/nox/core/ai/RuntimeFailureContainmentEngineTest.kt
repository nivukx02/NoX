package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RuntimeFailureContainmentEngineTest {

    @Test
    fun firstNativeFailureRequestsBoundedWorkerRestart() {
        val engine = RuntimeFailureContainmentEngine()

        val decision = engine.observeFault(
            timestampNanos = 1_000_000_000L,
            kind = RuntimeFailureContainmentEngine.FaultKind.NATIVE_CRASH,
            severity = RuntimeFailureContainmentEngine.Severity.ERROR,
            signature = "native-worker-1",
        )

        assertEquals(
            RuntimeFailureContainmentEngine.CircuitState.RECOVERING,
            decision.circuitState,
        )
        assertEquals(
            RuntimeFailureContainmentEngine.RecoveryAction.RESTART_WORKER,
            decision.action,
        )
        assertTrue(decision.recoveryAttempt >= 1)
    }

    @Test
    fun repeatedFatalFaultsOpenCircuit() {
        val engine = RuntimeFailureContainmentEngine(
            RuntimeFailureContainmentEngine.Config(
                criticalFaultThreshold = 2,
                maximumConsecutiveRecoveryAttempts = 2,
            ),
        )

        engine.observeFault(
            timestampNanos = 2_000_000_000L,
            kind = RuntimeFailureContainmentEngine.FaultKind.GPU_DEVICE_LOST,
            severity = RuntimeFailureContainmentEngine.Severity.CRITICAL,
        )

        val decision = engine.observeFault(
            timestampNanos = 2_001_000_000L,
            kind = RuntimeFailureContainmentEngine.FaultKind.GPU_DEVICE_LOST,
            severity = RuntimeFailureContainmentEngine.Severity.CRITICAL,
        )

        assertEquals(
            RuntimeFailureContainmentEngine.CircuitState.OPEN,
            decision.circuitState,
        )
        assertEquals(
            RuntimeFailureContainmentEngine.RecoveryAction.REQUIRE_USER_INTERVENTION,
            decision.action,
        )
        assertTrue(decision.shouldPauseCapture)
        assertFalse(decision.safeToContinue)
    }

    @Test
    fun successfulFramesDecayRecoveryPressure() {
        val engine = RuntimeFailureContainmentEngine(
            RuntimeFailureContainmentEngine.Config(
                recoverySuccessDecay = 0.50,
            ),
        )

        engine.observeFault(
            timestampNanos = 3_000_000_000L,
            kind = RuntimeFailureContainmentEngine.FaultKind.STALE_RUNTIME_STATE,
            severity = RuntimeFailureContainmentEngine.Severity.ERROR,
        )

        val success = engine.observeSuccess(
            timestampNanos = 4_000_000_000L,
            frames = 4,
        )

        assertTrue(
            success.circuitState !=
                RuntimeFailureContainmentEngine.CircuitState.OPEN,
        )
        assertTrue(success.consecutiveRecoveryAttempts <= 1)
    }

    @Test
    fun faultHistoryIsBounded() {
        val engine = RuntimeFailureContainmentEngine(
            RuntimeFailureContainmentEngine.Config(
                maximumFaultRecords = 3,
            ),
        )

        repeat(9) { index ->
            engine.observeFault(
                timestampNanos = 5_000_000_000L + index * 1_000_000L,
                kind = RuntimeFailureContainmentEngine.FaultKind.RECONSTRUCTION_FAILURE,
                severity = RuntimeFailureContainmentEngine.Severity.WARNING,
            )
        }

        assertEquals(3, engine.recentFaults(99).size)
    }

    @Test
    fun backoffIsBoundedAndVisibleToOperator() {
        val engine = RuntimeFailureContainmentEngine(
            RuntimeFailureContainmentEngine.Config(
                baseBackoffMilliseconds = 100L,
                maximumBackoffMilliseconds = 500L,
            ),
        )

        val first = engine.observeFault(
            timestampNanos = 6_000_000_000L,
            kind = RuntimeFailureContainmentEngine.FaultKind.CAMERA_STREAM_RESET,
            severity = RuntimeFailureContainmentEngine.Severity.ERROR,
        )
        val second = engine.observeFault(
            timestampNanos = 6_001_000_000L,
            kind = RuntimeFailureContainmentEngine.FaultKind.CAMERA_STREAM_RESET,
            severity = RuntimeFailureContainmentEngine.Severity.ERROR,
        )

        assertTrue(second.backoffMilliseconds >= first.backoffMilliseconds)
        assertTrue(second.backoffMilliseconds <= 500L)
        assertTrue(
            second.warnings.contains("runtime-fault-signature-repeating"),
        )
    }
}
