package com.nivukx.nox.core.nativeengine

import kotlin.test.Test
import kotlin.test.assertEquals

class NativeMotionEngineHardwareBufferDiagnosticsTest {
    @Test
    fun appendedBudgetFieldsPreserveOriginalDiagnosticIndices() {
        val result =
            NativeMotionEngine.HardwareBufferDiagnostics.fromNative(
                longArrayOf(
                    3L,
                    4_096L,
                    5L,
                    6L,
                    256L,
                    768L * 1024L * 1024L,
                ),
            )

        assertEquals(3L, result.activeLeaseCount)
        assertEquals(4_096L, result.residentBytes)
        assertEquals(5L, result.acquireFailures)
        assertEquals(6L, result.invalidReleaseCount)
        assertEquals(256L, result.maximumActiveLeaseCount)
        assertEquals(768L * 1024L * 1024L, result.maximumResidentBytes)
    }

    @Test
    fun legacyFourFieldDiagnosticsRemainDecodable() {
        val result =
            NativeMotionEngine.HardwareBufferDiagnostics.fromNative(
                longArrayOf(
                    1L,
                    2L,
                    3L,
                    4L,
                ),
            )

        assertEquals(1L, result.activeLeaseCount)
        assertEquals(2L, result.residentBytes)
        assertEquals(3L, result.acquireFailures)
        assertEquals(4L, result.invalidReleaseCount)
        assertEquals(0L, result.maximumActiveLeaseCount)
        assertEquals(0L, result.maximumResidentBytes)
    }
}
