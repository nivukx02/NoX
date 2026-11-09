package com.nivukx.nox.core.nativeengine.synchronization

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeSyncResultTest {
    @Test
    fun parsesFullNativeSynchronizationEvidence() {
        val fields = doubleArrayOf(
            1_200_000.0,
            18.5,
            240_000.0,
            1_200_000.0,
            0.125,
            0.93,
            0.81,
            0.91,
            680_000.0,
            1_400_000.0,
            0.04,
            0.96,
            42_000.0,
            16.0,
            8.0,
            48.0,
            1.0,
        )

        val result =
            NativeSyncResult.fromNative(fields)

        assertEquals(1_200_000.0, result.offsetNanos, 0.0)
        assertEquals(18.5, result.driftPpm, 0.0)
        assertEquals(0.93, result.subframeConfidence, 0.0)
        assertEquals(0.81, result.subframeSharpness, 0.0)
        assertEquals(0.91, result.temporalConfidence, 0.0)
        assertEquals(0.04, result.outlierRatio, 0.0)
        assertEquals(0.96, result.clockSupportRatio, 0.0)
        assertEquals(42_000.0, result.offsetUncertaintyNanos, 0.0)
        assertEquals(16.0, result.driftUncertaintyPpm, 0.0)
        assertEquals(8, result.clockIterations)
        assertEquals(48, result.sampleCount)
        assertTrue(result.valid)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsWrongNativeFieldCount() {
        NativeSyncResult.fromNative(
            DoubleArray(7),
        )
    }
}
