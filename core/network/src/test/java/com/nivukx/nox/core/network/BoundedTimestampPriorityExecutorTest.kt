package com.nivukx.nox.core.network

import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BoundedTimestampPriorityExecutorTest {

    @Test
    fun newer_timestamped_work_is_preferred_when_queueing() {
        val executor =
            BoundedTimestampPriorityExecutor(
                workerCount = 1,
                maximumOutstandingTasks = 8,
                threadName = "priority-test",
            )

        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val order =
            Collections.synchronizedList(
                mutableListOf<Long>(),
            )

        executor.tryExecute(10L) {
            started.countDown()
            release.await(1, TimeUnit.SECONDS)
            order += 10L
        }

        assertTrue(started.await(1, TimeUnit.SECONDS))

        executor.tryExecute(1L) { order += 1L }
        executor.tryExecute(30L) { order += 30L }
        executor.tryExecute(20L) { order += 20L }

        release.countDown()

        while (order.size < 4) {
            Thread.yield()
        }

        assertEquals(
            listOf(10L, 30L, 20L, 1L),
            order.toList(),
        )

        executor.close()
    }

    @Test
    fun clear_queued_tasks_preserves_non_negative_outstanding_accounting() {
        val executor =
            BoundedTimestampPriorityExecutor(
                workerCount = 1,
                maximumOutstandingTasks = 8,
                threadName = "clear-test",
            )

        val started = CountDownLatch(1)
        val release = CountDownLatch(1)

        assertTrue(
            executor.tryExecute(1L) {
                started.countDown()
                release.await(1, TimeUnit.SECONDS)
            },
        )
        assertTrue(started.await(1, TimeUnit.SECONDS))

        repeat(4) {
            assertTrue(executor.tryExecute(10L + it) {})
        }

        val removed = executor.clearQueuedTasks()
        assertEquals(4, removed)
        assertEquals(1, executor.outstandingTasks())

        release.countDown()
        while (executor.outstandingTasks() != 0) {
            Thread.yield()
        }

        executor.close()
        assertEquals(0, executor.outstandingTasks())
    }

    @Test
    fun outstanding_work_is_hard_bounded() {
        val executor =
            BoundedTimestampPriorityExecutor(
                workerCount = 1,
                maximumOutstandingTasks = 2,
                threadName = "bound-test",
            )

        val release = CountDownLatch(1)
        val started = CountDownLatch(1)

        assertTrue(
            executor.tryExecute(1L) {
                started.countDown()
                release.await(1, TimeUnit.SECONDS)
            },
        )

        assertTrue(started.await(1, TimeUnit.SECONDS))

        assertTrue(executor.tryExecute(2L) {})
        assertTrue(!executor.tryExecute(3L) {})
        assertTrue(executor.outstandingTasks() <= 2)

        release.countDown()
        executor.close()
        assertEquals(0, executor.outstandingTasks())
    }
}
