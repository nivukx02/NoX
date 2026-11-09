package com.nivukx.nox.core.nativeengine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeEngineDiagnosticsTest {
    @Test
    fun v6PayloadDecodesWithoutFieldShift() {
        val fields = longArrayOf(
            1000L, // processed
            17L, // dropped
            4L, // overflow
            3L, // queue depth
            9L, // max queue
            6L, // active cameras
            4L, // workers
            42L, // processing epoch
            12L, // active buffers
            20L, // free buffers
            750_000L, // workload
            320_000L, // queue utilization
            1L, // shedding
            1L, // thermal constrained
            900L, // worker processed
            820_000L, // worker accepted
            710_000L, // worker quality
            90_000L, // temporal gap
            4_500L, // queue age us
            8_500L, // processing us
        )

        val diagnostics = NativeEngineDiagnostics.fromNative(fields)

        assertEquals(1000L, diagnostics.processedObservations)
        assertEquals(17L, diagnostics.droppedObservations)
        assertEquals(4L, diagnostics.captureOverflowEvents)
        assertEquals(3, diagnostics.queueDepth)
        assertEquals(9, diagnostics.maximumQueueDepth)
        assertEquals(6, diagnostics.activeCameraCount)
        assertEquals(4, diagnostics.workerCount)
        assertEquals(42L, diagnostics.processingEpoch)
        assertEquals(12, diagnostics.activeBufferBlocks)
        assertEquals(20, diagnostics.freeBufferBlocks)
        assertEquals(0.75, diagnostics.workloadScale, 1e-9)
        assertEquals(0.32, diagnostics.queueUtilization, 1e-9)
        assertTrue(diagnostics.shedding)
        assertTrue(diagnostics.thermalConstrained)
        assertEquals(900L, diagnostics.workerProcessedObservations)
        assertEquals(0.82, diagnostics.workerAcceptedRatio, 1e-9)
        assertEquals(0.71, diagnostics.workerQuality, 1e-9)
        assertEquals(0.09, diagnostics.workerTemporalGap, 1e-9)
        assertEquals(4_500L, diagnostics.workerQueueAgeMicroseconds)
        assertEquals(8_500L, diagnostics.workerProcessingMicroseconds)
    }

    @Test
    fun legacyV9PayloadKeepsNewExposureFieldNeutral() {
        val fields =
            LongArray(41) { 0L }.also {
                it[40] = 910_000L
            }

        val diagnostics =
            NativeEngineDiagnostics.fromNative(fields)

        assertEquals(
            0.91,
            diagnostics.workerDynamicStability,
            1e-9,
        )
        assertEquals(
            0.0,
            diagnostics.workerExposureMotionBlurRisk,
            1e-9,
        )
    }

    @Test
    fun legacyV10PayloadPreservesExposureBlur() {
        val fields =
            LongArray(42) { 0L }.also {
                it[40] = 910_000L
                it[41] = 240_000L
            }

        val diagnostics =
            NativeEngineDiagnostics.fromNative(fields)

        assertEquals(
            0.91,
            diagnostics.workerDynamicStability,
            1e-9,
        )
        assertEquals(
            0.24,
            diagnostics.workerExposureMotionBlurRisk,
            1e-9,
        )
        assertEquals(
            1.0,
            diagnostics.workerSubpixelMotionRetention,
            1e-9,
        )
    }

    @Test
    fun currentPayloadDecodesExposureBlurAndSubpixelEvidence() {
        val fields =
            LongArray(43) { 0L }.also {
                it[40] = 910_000L
                it[41] = 220_000L
                it[42] = 870_000L
            }

        val diagnostics =
            NativeEngineDiagnostics.fromNative(fields)

        assertEquals(
            0.91,
            diagnostics.workerDynamicStability,
            1e-9,
        )
        assertEquals(
            0.22,
            diagnostics.workerExposureMotionBlurRisk,
            1e-9,
        )
        assertEquals(
            0.87,
            diagnostics.workerSubpixelMotionRetention,
            1e-9,
        )
    }

    @Test
    fun wrongFieldCountIsRejected() {
        try {
            NativeEngineDiagnostics.fromNative(LongArray(19))
            assertTrue("expected exception", false)
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message?.contains("20") == true)
        }
    }

    @Test
    fun defaultsRemainSafeForNewOptionalFields() {
        val diagnostics = NativeEngineDiagnostics(
            processedObservations = 1L,
            droppedObservations = 0L,
            processingEpoch = 1L,
            activeBufferBlocks = 0,
            freeBufferBlocks = 0,
        )
        assertFalse(diagnostics.shedding)
        assertEquals(0L, diagnostics.captureOverflowEvents)
        assertEquals(0L, diagnostics.workerQueueAgeMicroseconds)
        assertEquals(0L, diagnostics.workerProcessingMicroseconds)
    }
}
