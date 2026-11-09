package com.nivukx.nox.core.capture

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CameraCaptureHandlerPoolTest {

    @Test
    fun worker_set_scales_with_live_leases_instead_of_starting_full_budget() {
        val pool =
            CameraCaptureHandlerPool(
                maximumWorkers = 4,
            )

        val first = pool.acquire()
        assertEquals(1, pool.diagnostics().workerCount)

        val more =
            (0 until 4).map {
                pool.acquire()
            }

        assertTrue(
            pool.diagnostics().workerCount in 2..4,
        )

        first.close()
        more.forEach { it.close() }

        assertEquals(
            0,
            pool.diagnostics().workerCount,
        )
    }

    @Test
    fun many_camera_leases_share_a_bounded_worker_set() {
        val pool =
            CameraCaptureHandlerPool(
                maximumWorkers = 2,
            )

        val leases =
            (0 until 32).map {
                pool.acquire()
            }

        val diagnostics = pool.diagnostics()

        assertEquals(2, diagnostics.workerCount)
        assertEquals(32, diagnostics.activeLeases)
        assertTrue(
            leases
                .map { it.worker.name }
                .distinct()
                .size <= 2,
        )

        leases.forEach { it.close() }

        val released = pool.diagnostics()
        assertEquals(0, released.activeLeases)
        assertEquals(0, released.workerCount)
    }

    @Test
    fun releasing_a_lease_twice_is_idempotent() {
        val pool =
            CameraCaptureHandlerPool(
                maximumWorkers = 1,
            )

        val lease = pool.acquire()

        lease.close()
        lease.close()

        val diagnostics = pool.diagnostics()

        assertEquals(0, diagnostics.activeLeases)
        assertEquals(0, diagnostics.workerCount)
    }

    @Test
    fun pool_restarts_with_a_new_generation_after_full_release() {
        val pool =
            CameraCaptureHandlerPool(
                maximumWorkers = 1,
            )

        val first = pool.acquire()
        val generation1 = pool.diagnostics().generation
        first.close()

        val second = pool.acquire()
        val generation2 = pool.diagnostics().generation
        second.close()

        assertTrue(generation2 > generation1)
    }
}
