package com.nivukx.nox.core.network

import java.util.concurrent.PriorityBlockingQueue
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadFactory
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Bounded executor for time-sensitive transport work.
 *
 * Newer timestamped work is preferred over older queued work because stale motion evidence has
 * sharply reduced value in a live capture federation. The executor is strictly bounded by an
 * admission counter covering both queued and running tasks.
 *
 * This is intentionally generic so other remote evidence stages can reuse the same backpressure
 * contract without creating another executor per peer or per packet class.
 */
class BoundedTimestampPriorityExecutor(
    workerCount: Int,
    maximumOutstandingTasks: Int,
    threadName: String,
) : AutoCloseable {

    init {
        require(workerCount in 1..16)
        require(maximumOutstandingTasks >= workerCount)
        require(threadName.isNotBlank())
    }

    private data class Task(
        val timestampNanos: Long,
        val sequence: Long,
        val action: () -> Unit,
        val onComplete: () -> Unit,
    ) : Runnable {
        override fun run() {
            try {
                action()
            } finally {
                onComplete()
            }
        }
    }

    private val closed = AtomicBoolean(false)
    private val outstanding = AtomicInteger(0)
    private val sequence = AtomicLong(0L)

    private val queue =
        PriorityBlockingQueue<Runnable>(
            32,
            Comparator { first, second ->
                val a = first as Task
                val b = second as Task

                when {
                    a.timestampNanos > b.timestampNanos -> -1
                    a.timestampNanos < b.timestampNanos -> 1
                    a.sequence > b.sequence -> -1
                    a.sequence < b.sequence -> 1
                    else -> 0
                }
            },
        )

    private val executor =
        ThreadPoolExecutor(
            workerCount,
            workerCount,
            30L,
            TimeUnit.SECONDS,
            queue,
            NamedDaemonThreadFactory(threadName),
            ThreadPoolExecutor.AbortPolicy(),
        )

    fun tryExecute(
        timestampNanos: Long,
        action: () -> Unit,
    ): Boolean {
        require(timestampNanos >= 0L)
        check(!closed.get()) {
            "BoundedTimestampPriorityExecutor is closed"
        }

        val reserved =
            outstanding.incrementAndGet()

        if (reserved > maximumOutstandingTasks) {
            outstanding.decrementAndGet()
            return false
        }

        val task =
            Task(
                timestampNanos = timestampNanos,
                sequence = sequence.incrementAndGet(),
                action = action,
                onComplete = {
                    outstanding.decrementAndGet()
                },
            )

        return try {
            executor.execute(task)
            true
        } catch (_: RejectedExecutionException) {
            outstanding.decrementAndGet()
            false
        }
    }

    fun outstandingTasks(): Int =
        outstanding.get()

    fun queuedTasks(): Int =
        queue.size

    fun isShutdown(): Boolean =
        executor.isShutdown

    fun clearQueuedTasks(): Int {
        if (closed.get()) {
            return 0
        }

        val removed = mutableListOf<Runnable>()
        queue.drainTo(removed)
        if (removed.isEmpty()) {
            return 0
        }

        outstanding.addAndGet(-removed.size)
        return removed.size
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) {
            return
        }

        val removed = mutableListOf<Runnable>()
        queue.drainTo(removed)
        outstanding.addAndGet(-removed.size)
        executor.shutdownNow()
    }

    private class NamedDaemonThreadFactory(
        private val prefix: String,
    ) : ThreadFactory {
        private val counter = AtomicInteger(0)

        override fun newThread(
            runnable: Runnable,
        ): Thread =
            Thread(
                runnable,
                prefix + "-" + counter.incrementAndGet(),
            ).apply {
                isDaemon = true
            }
    }
}
