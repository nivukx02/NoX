package com.nivukx.nox.core.reconstruction

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GpuBufferLeaseCoordinatorTest {

    @Test
    fun staleGenerationCannotReleaseNewLease() {
        val coordinator = GpuBufferLeaseCoordinator(
            GpuBufferLeaseCoordinator.Config(
                maximumSlots = 2,
                maximumInFlight = 2,
            ),
        )

        val first = assertNotNull(coordinator.acquire(1_000_000_000L))
        assertEquals(
            GpuBufferLeaseCoordinator.ReleaseResult.RELEASED,
            coordinator.release(first, 1_001_000_000L),
        )

        val reused = assertNotNull(coordinator.acquire(1_002_000_000L))
        assertEquals(first.slot, reused.slot)
        assertTrue(first.generation != reused.generation)

        assertEquals(
            GpuBufferLeaseCoordinator.ReleaseResult.STALE_GENERATION,
            coordinator.release(first, 1_003_000_000L),
        )
        assertEquals(1, coordinator.telemetry(1_003_000_000L).activeLeases)

        assertEquals(
            GpuBufferLeaseCoordinator.ReleaseResult.RELEASED,
            coordinator.release(reused, 1_004_000_000L),
        )
        assertEquals(0, coordinator.telemetry(1_004_000_000L).activeLeases)
    }

    @Test
    fun gpuSubmissionMustCompleteBeforeRelease() {
        val coordinator = GpuBufferLeaseCoordinator()

        val token = assertNotNull(coordinator.acquire(2_000_000_000L))
        assertTrue(
            coordinator.markGpuSubmitted(
                token,
                2_001_000_000L,
            ),
        )

        assertEquals(
            GpuBufferLeaseCoordinator.ReleaseResult.WAITING_FOR_GPU,
            coordinator.release(token, 2_002_000_000L),
        )

        assertTrue(
            coordinator.markGpuCompleted(
                token,
                2_003_000_000L,
            ),
        )

        assertEquals(
            GpuBufferLeaseCoordinator.ReleaseResult.RELEASED,
            coordinator.release(token, 2_004_000_000L),
        )
        assertEquals(0, coordinator.telemetry(2_004_000_000L).activeLeases)
    }

    @Test
    fun watchdogAbortsHungLeaseAndExposesTelemetry() {
        val coordinator = GpuBufferLeaseCoordinator(
            GpuBufferLeaseCoordinator.Config(
                maximumLeaseAgeMilliseconds = 10.0,
                watchdogGraceMilliseconds = 5.0,
            ),
        )

        val token = assertNotNull(coordinator.acquire(3_000_000_000L))
        assertTrue(
            coordinator.markGpuSubmitted(
                token,
                3_001_000_000L,
            ),
        )

        assertEquals(
            1,
            coordinator.watchdogSweep(3_020_000_000L),
        )

        val telemetry = coordinator.telemetry(3_021_000_000L)
        assertEquals(0, telemetry.activeLeases)
        assertEquals(1L, telemetry.watchdogAborts)
        assertEquals(1L, telemetry.forcedStaleLeases)
    }

    @Test
    fun boundedAdmissionRejectsExcessInFlightWork() {
        val coordinator = GpuBufferLeaseCoordinator(
            GpuBufferLeaseCoordinator.Config(
                maximumSlots = 3,
                maximumInFlight = 2,
            ),
        )

        assertNotNull(coordinator.acquire(4_000_000_000L))
        assertNotNull(coordinator.acquire(4_001_000_000L))
        assertTrue(coordinator.acquire(4_002_000_000L) == null)
        assertEquals(
            1L,
            coordinator.telemetry(4_003_000_000L).admissionRejected,
        )
    }

    @Test
    fun invalidTokensAreRejectedWithoutTouchingActiveSlots() {
        val coordinator = GpuBufferLeaseCoordinator()
        val token = GpuBufferLeaseCoordinator.LeaseToken(
            slot = 999,
            generation = 4L,
            valid = true,
        )

        assertFalse(coordinator.markGpuSubmitted(token, 5_000_000_000L))
        assertEquals(
            GpuBufferLeaseCoordinator.ReleaseResult.INVALID_TOKEN,
            coordinator.release(token, 5_001_000_000L),
        )
        assertEquals(0, coordinator.telemetry(5_002_000_000L).activeLeases)
    }
}
