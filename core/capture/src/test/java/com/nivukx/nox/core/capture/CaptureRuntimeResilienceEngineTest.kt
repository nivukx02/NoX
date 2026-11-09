package com.nivukx.nox.core.capture

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CaptureRuntimeResilienceEngineTest {

    @Test
    fun criticalPressureStopsForSafety() {
        val engine = CaptureRuntimeResilienceEngine()
        val decision = engine.evaluate(
            CaptureRuntimeResilienceEngine.Input(
                timestampNanos = 10_000_000_000L,
                lastHeartbeatNanos = 9_900_000_000L,
                thermalRisk = 0.995,
                memoryPressure = 0.996,
                queueAgeMilliseconds = 4.0,
                processingMilliseconds = 5.0,
                dropRatio = 0.0,
                nativeErrorCount = 0,
                cameraDisconnectCount = 0,
                calibrationConfidence = 0.95,
                synchronizationConfidence = 0.95,
            ),
        )
        assertTrue(decision.state == CaptureRuntimeResilienceEngine.State.BLOCKED)
        assertTrue(decision.action == CaptureRuntimeResilienceEngine.Action.STOP_FOR_SAFETY)
        assertFalse(decision.captureAllowed)
        engine.close()
    }

    @Test
    fun queueBackpressureDoesNotImmediatelyRestartNativeSession() {
        val engine = CaptureRuntimeResilienceEngine()
        val decision = engine.evaluate(
            CaptureRuntimeResilienceEngine.Input(
                timestampNanos = 10_000_000_000L,
                lastHeartbeatNanos = 9_800_000_000L,
                thermalRisk = 0.40,
                memoryPressure = 0.40,
                queueAgeMilliseconds = 35.0,
                processingMilliseconds = 8.0,
                dropRatio = 0.01,
                nativeErrorCount = 0,
                cameraDisconnectCount = 0,
                calibrationConfidence = 0.94,
                synchronizationConfidence = 0.94,
            ),
        )
        assertTrue(
            decision.action == CaptureRuntimeResilienceEngine.Action.DRAIN_QUEUES ||
                decision.action == CaptureRuntimeResilienceEngine.Action.REDUCE_OPTIONAL_STAGES,
        )
        assertTrue(decision.captureAllowed)
        engine.close()
    }

    @Test
    fun restartBudgetIsRateLimited() {
        val engine = CaptureRuntimeResilienceEngine()
        var restartCount = 0
        repeat(8) { index ->
            val d = engine.evaluate(
                CaptureRuntimeResilienceEngine.Input(
                    timestampNanos = 10_000_000_000L + index * 1_000_000_000L,
                    lastHeartbeatNanos = 9_000_000_000L,
                    thermalRisk = 0.40,
                    memoryPressure = 0.40,
                    queueAgeMilliseconds = 4.0,
                    processingMilliseconds = 5.0,
                    dropRatio = 0.01,
                    nativeErrorCount = 4,
                    cameraDisconnectCount = 0,
                    calibrationConfidence = 0.94,
                    synchronizationConfidence = 0.94,
                ),
            )
            if (d.action == CaptureRuntimeResilienceEngine.Action.RESTART_NATIVE_SESSION) {
                restartCount++
            }
        }
        assertTrue(restartCount <= 3)
        assertTrue(restartCount > 0)
        engine.close()
    }

    @Test
    fun futureHeartbeatTimestampsAreRejected() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            CaptureRuntimeResilienceEngine.Input(
                timestampNanos = 1_000L,
                lastHeartbeatNanos = 1_001L,
                thermalRisk = 0.0,
                memoryPressure = 0.0,
                queueAgeMilliseconds = 0.0,
                processingMilliseconds = 0.0,
                dropRatio = 0.0,
                nativeErrorCount = 0,
                cameraDisconnectCount = 0,
                calibrationConfidence = 1.0,
                synchronizationConfidence = 1.0,
            )
        }
    }

    @Test
    fun zeroAndOverflowingRecoveryConfigurationAreRejected() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            CaptureRuntimeResilienceEngine(heartbeatTimeoutNanos = 0L)
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            CaptureRuntimeResilienceEngine(heartbeatTimeoutNanos = Long.MAX_VALUE)
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            CaptureRuntimeResilienceEngine(maximumRestartAttemptsPerWindow = 0)
        }
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            CaptureRuntimeResilienceEngine(restartWindowNanos = 0L)
        }
    }

    @Test
    fun evaluationTimestampsMustRemainMonotonicUntilReset() {
        val engine = CaptureRuntimeResilienceEngine()
        engine.evaluate(
            CaptureRuntimeResilienceEngine.Input(
                timestampNanos = 2_000_000_000L,
                lastHeartbeatNanos = 1_900_000_000L,
                thermalRisk = 0.0,
                memoryPressure = 0.0,
                queueAgeMilliseconds = 0.0,
                processingMilliseconds = 0.0,
                dropRatio = 0.0,
                nativeErrorCount = 0,
                cameraDisconnectCount = 0,
                calibrationConfidence = 1.0,
                synchronizationConfidence = 1.0,
            ),
        )
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            engine.evaluate(
                CaptureRuntimeResilienceEngine.Input(
                    timestampNanos = 1_900_000_000L,
                    lastHeartbeatNanos = 1_800_000_000L,
                    thermalRisk = 0.0,
                    memoryPressure = 0.0,
                    queueAgeMilliseconds = 0.0,
                    processingMilliseconds = 0.0,
                    dropRatio = 0.0,
                    nativeErrorCount = 0,
                    cameraDisconnectCount = 0,
                    calibrationConfidence = 1.0,
                    synchronizationConfidence = 1.0,
                ),
            )
        }
        engine.reset()
        val result =
            engine.evaluate(
                CaptureRuntimeResilienceEngine.Input(
                    timestampNanos = 1_000L,
                    lastHeartbeatNanos = 900L,
                    thermalRisk = 0.0,
                    memoryPressure = 0.0,
                    queueAgeMilliseconds = 0.0,
                    processingMilliseconds = 0.0,
                    dropRatio = 0.0,
                    nativeErrorCount = 0,
                    cameraDisconnectCount = 0,
                    calibrationConfidence = 1.0,
                    synchronizationConfidence = 1.0,
                ),
            )
        assertTrue(result.captureAllowed)
    }

}
