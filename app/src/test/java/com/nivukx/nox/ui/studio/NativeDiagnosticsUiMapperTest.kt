package com.nivukx.nox.ui.studio

import com.nivukx.nox.core.nativeengine.NativeEngineDiagnostics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeDiagnosticsUiMapperTest {
    @Test
    fun currentAbiMatchesNativeEngineAbi() {
        assertEquals(12, NativeDiagnosticsUiMapper.CURRENT_ABI)
    }

    @Test
    fun mergeTelemetryPreservesExistingAuthoritativeSignals() {
        val telemetry = StudioTelemetryState(
            camerasConnected = 2,
            camerasExpected = 4,
            synchronizationConfidence = 0.91,
            calibrationReady = true,
            trackingConfidence = 0.88,
            reconstructionConfidence = 0.86,
            droppedObservations = null,
            thermalConstrained = false,
            exportReady = true,
            nativeEngineHealthy = true,
        )
        val diagnostics = NativeEngineDiagnostics(
            processedObservations = 100L,
            droppedObservations = 3L,
            captureOverflowEvents = 2L,
            queueDepth = 4,
            maximumQueueDepth = 8,
            activeCameraCount = 6,
            workerCount = 4,
            processingEpoch = 2L,
            activeBufferBlocks = 10,
            freeBufferBlocks = 18,
            workerQueueAgeMicroseconds = 2_000L,
            workerProcessingMicroseconds = 6_000L,
        )

        val merged =
            NativeDiagnosticsUiMapper.mergeTelemetry(
                telemetry,
                diagnostics,
            )

        assertEquals(2, merged.camerasConnected)
        assertEquals(3L, merged.droppedObservations)
        assertEquals(4, merged.queueDepth)
        assertEquals(2L, merged.captureOverflowCount)
        assertEquals(2_000L, merged.nativeQueueAgeMicroseconds)
        assertEquals(
            3.0 / 103.0,
            merged.nativeDropRatio!!,
            1e-9,
        )
        assertEquals(6_000L, merged.nativeProcessingMicroseconds)
        assertEquals(false, merged.thermalConstrained)
    }

    @Test
    fun mapperSuppliesInspectorEvidence() {
        val diagnostics = NativeEngineDiagnostics(
            processedObservations = 100L,
            droppedObservations = 3L,
            captureOverflowEvents = 2L,
            queueDepth = 4,
            maximumQueueDepth = 8,
            activeCameraCount = 6,
            workerCount = 4,
            processingEpoch = 2L,
            activeBufferBlocks = 10,
            freeBufferBlocks = 18,
            workerQueueAgeMicroseconds = 2_000L,
            workerProcessingMicroseconds = 6_000L,
        )

        val state = NativeDiagnosticsUiMapper.toInspectorState(diagnostics, true)
        assertEquals(6, state.activeCameraCount)
        assertEquals(8, state.maximumQueueDepth)
        assertEquals(2L, state.captureOverflowEvents)
        assertTrue(state.nativeHealthy == true)
    }
}
