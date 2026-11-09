package com.nivukx.nox

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MotionRuntimeResourceGovernorTest {

    @Test
    fun reservationsAreBoundedAndReleasable() {
        val governor = MotionRuntimeResourceGovernor(maximumResources = 2)
        val first = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.TRACK_STATE,
            bytes = 100,
            owner = "body",
            nowNanos = 10L,
        )
        val second = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.FACE_STATE,
            bytes = 200,
            owner = "face",
            nowNanos = 20L,
        )
        val third = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.EYE_STATE,
            bytes = 300,
            owner = "eye",
            nowNanos = 30L,
        )
        assertTrue(first != null)
        assertTrue(second != null)
        assertEquals(null, third)
        assertEquals(300L, governor.allocatedBytes())
        assertTrue(governor.release(first!!.id, 40L))
        assertEquals(200L, governor.allocatedBytes())
        governor.close()
    }

    @Test
    fun criticalPressureRequiresShedding() {
        val governor = MotionRuntimeResourceGovernor()
        governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.TSDF_BLOCK,
            bytes = 1024,
            owner = "reconstruction",
            nowNanos = 100L,
        )
        val snapshot = governor.snapshot(
            timestampNanos = 4_000_000_000L,
            memoryPressure = 0.99,
            cpuPressure = 0.71,
            gpuPressure = 0.74,
            thermalRisk = 0.93,
        )
        assertEquals(
            MotionRuntimeResourceGovernor.PressureBand.CRITICAL,
            snapshot.pressureBand,
        )
        assertTrue(snapshot.forcedSheddingRequired)
        assertFalse(snapshot.safeToContinue)
        governor.close()
    }

    @Test
    fun wraparoundDoesNotOverwriteALiveLease() {
        val governor = MotionRuntimeResourceGovernor(maximumResources = 4)
        val first = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.TRACK_STATE,
            bytes = 64,
            owner = "body",
            nowNanos = 1L,
        )
        requireNotNull(first)

        val field =
            MotionRuntimeResourceGovernor::class.java
                .getDeclaredField("nextId")
        field.isAccessible = true
        val nextId = field.get(governor) as java.util.concurrent.atomic.AtomicLong
        nextId.set(1L)

        val second = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.FACE_STATE,
            bytes = 96,
            owner = "face",
            nowNanos = 2L,
        )
        requireNotNull(second)

        assertTrue(second.id != first.id)
        assertEquals(2, governor.activeLeaseCount())
        assertEquals(160L, governor.allocatedBytes())
        governor.close()
    }

    @Test
    fun ownerReleasePreventsLongSessionLeakAccumulation() {
        val governor = MotionRuntimeResourceGovernor()
        repeat(32) { index ->
            governor.reserve(
                kind = MotionRuntimeResourceGovernor.ResourceKind.TEMPORAL_WINDOW,
                bytes = 4096,
                owner = "temporal-$index",
                nowNanos = index.toLong(),
            )
        }
        val released = governor.releaseOwner("temporal-5", 100L)
        assertEquals(1, released)
        assertEquals(31, governor.activeLeaseCount())
        governor.close()
    }

    @Test
    fun stale_sweep_reclaims_expired_resources_without_touching_preserved_kinds() {
        val governor = MotionRuntimeResourceGovernor()
        governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.TEMPORAL_WINDOW,
            bytes = 1024,
            owner = "old",
            nowNanos = 0L,
        )
        governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.SESSION,
            bytes = 2048,
            owner = "session",
            nowNanos = 0L,
        )

        val swept = governor.sweepStaleLeases(
            nowNanos = 3_000_000_000L,
            maximumAgeNanos = 2_000_000_000L,
            preserveKinds = setOf(
                MotionRuntimeResourceGovernor.ResourceKind.SESSION,
            ),
        )

        assertEquals(1, swept)
        assertEquals(1, governor.activeLeaseCount())
        assertEquals(2048L, governor.allocatedBytes())
        governor.close()
    }


    @Test
    fun byte_budget_rejects_reservations_before_pressure_overflow() {
        val governor =
            MotionRuntimeResourceGovernor(
                maximumResources = 8,
                maximumAllocatedBytes = 1024L,
            )

        val first = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.GPU_BUFFER,
            bytes = 768L,
            owner = "gpu",
            nowNanos = 1_000L,
        )
        val rejected = governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.TSDF_BLOCK,
            bytes = 320L,
            owner = "tsdf",
            nowNanos = 2_000L,
        )

        assertTrue(first != null)
        assertEquals(null, rejected)
        assertEquals(768L, governor.allocatedBytes())

        val snapshot = governor.snapshot(
            timestampNanos = 3_000L,
            memoryPressure = 0.95,
            cpuPressure = 0.30,
            gpuPressure = 0.70,
            thermalRisk = 0.40,
        )
        assertTrue(
            snapshot.warnings.any {
                it.contains("memory-pressure-high")
            },
        )
        governor.close()
    }

    @Test
    fun leaseBudgetPressureCannotBeHiddenByLowReportedMemoryPressure() {
        val governor =
            MotionRuntimeResourceGovernor(
                maximumResources = 8,
                maximumAllocatedBytes = 1000L,
            )

        governor.reserve(
            kind = MotionRuntimeResourceGovernor.ResourceKind.TSDF_BLOCK,
            bytes = 900L,
            owner = "surface",
            nowNanos = 8_000L,
        )

        val snapshot =
            governor.snapshot(
                timestampNanos = 9_000L,
                memoryPressure = 0.10,
                cpuPressure = 0.10,
                gpuPressure = 0.10,
                thermalRisk = 0.10,
            )

        assertTrue(
            snapshot.pressureBand ==
                MotionRuntimeResourceGovernor.PressureBand.HIGH,
        )
        assertTrue(
            snapshot.warnings.any {
                it.contains("memory-budget-near-capacity")
            },
        )
        governor.close()
    }

    @Test
    fun gpuBufferLifetimeIsGenerationSafeAndCompletionGated() {
        val governor = MotionRuntimeResourceGovernor(
            maximumResources = 4,
        )

        val first = governor.acquireGpuBuffer(6_000_000_000L)
        requireNotNull(first)

        assertTrue(
            governor.markGpuBufferSubmitted(
                first,
                6_001_000_000L,
            ),
        )
        assertEquals(
            com.nivukx.nox.core.reconstruction.GpuBufferLeaseCoordinator.ReleaseResult.WAITING_FOR_GPU,
            governor.releaseGpuBuffer(
                first,
                6_002_000_000L,
            ),
        )

        assertTrue(
            governor.markGpuBufferCompleted(
                first,
                6_003_000_000L,
            ),
        )
        assertEquals(
            com.nivukx.nox.core.reconstruction.GpuBufferLeaseCoordinator.ReleaseResult.RELEASED,
            governor.releaseGpuBuffer(
                first,
                6_004_000_000L,
            ),
        )

        val reused = governor.acquireGpuBuffer(6_005_000_000L)
        requireNotNull(reused)
        assertTrue(reused.generation != first.generation)
        assertEquals(
            com.nivukx.nox.core.reconstruction.GpuBufferLeaseCoordinator.ReleaseResult.STALE_GENERATION,
            governor.releaseGpuBuffer(
                first,
                6_006_000_000L,
            ),
        )
        assertTrue(
            governor.gpuBufferLeaseTelemetry(
                6_007_000_000L,
            ).staleGenerationReleases >= 1L,
        )
        governor.close()
    }


    @Test
    fun recentEventsReturnsBoundedTailWithoutCorruptingHistory() {
        val governor = MotionRuntimeResourceGovernor(maximumEvents = 4)
        repeat(6) { index ->
            governor.reserve(
                kind = MotionRuntimeResourceGovernor.ResourceKind.TRACK_STATE,
                bytes = 16L,
                owner = "event-$index",
                nowNanos = index.toLong(),
            )
        }

        val recent = governor.recentEvents(2)
        assertEquals(2, recent.size)
        assertTrue(recent.all { it.resourceKind == MotionRuntimeResourceGovernor.ResourceKind.TRACK_STATE })
        assertEquals(5L, recent.last().timestampNanos)
        assertEquals("resource-reserved", recent.last().code)
        governor.close()
    }

}
