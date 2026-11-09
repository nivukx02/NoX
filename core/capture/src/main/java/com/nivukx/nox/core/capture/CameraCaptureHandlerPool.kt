package com.nivukx.nox.core.capture

import android.os.HandlerThread
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Bounded process-local execution pool for Camera2 callbacks.
 *
 * A Camera2CaptureController does not need a dedicated thread for every physical camera. Sharing
 * a small number of ordered HandlerThreads keeps callback execution serialized per worker while
 * preventing a large federated rig from creating one thread per device.
 *
 * The pool is reference-counted:
 * - the first controller starts the bounded worker set;
 * - each controller acquires one lease;
 * - closing a controller releases its lease;
 * - the final release shuts every worker down safely;
 * - acquiring after a full release creates a fresh generation.
 */
class CameraCaptureHandlerPool(
    private val maximumWorkers: Int = defaultWorkerCount(),
) {

    data class Diagnostics(
        val workerCount: Int,
        val activeLeases: Int,
        val generation: Long,
        val roundRobinCursor: Int,
    )

    class Lease internal constructor(
        internal val worker: HandlerThread,
        private val owner: CameraCaptureHandlerPool,
        private val closed: AtomicBoolean = AtomicBoolean(false),
    ) : AutoCloseable {

        val looper
            get() = worker.looper

        override fun close() {
            if (closed.compareAndSet(false, true)) {
                owner.release()
            }
        }
    }

    private val lock = Any()
    private val nextWorker = AtomicInteger(0)
    private val generationCounter = AtomicLong(0L)

    private var workers: List<HandlerThread> = emptyList()
    private var activeLeases = 0
    private var generation = 0L

    init {
        require(maximumWorkers in 1..16)
    }

    fun acquire(): Lease {
        synchronized(lock) {
            ensureCapacityForLeases(
                requestedLeases =
                    activeLeases + 1,
            )
            activeLeases += 1
            val index =
                nextWorker.getAndIncrement()
                    .floorMod(workers.size)
            return Lease(
                worker = workers[index],
                owner = this,
            )
        }
    }

    fun diagnostics(): Diagnostics =
        synchronized(lock) {
            Diagnostics(
                workerCount = workers.size,
                activeLeases = activeLeases,
                generation = generation,
                roundRobinCursor = nextWorker.get(),
            )
        }

    private fun release() {
        synchronized(lock) {
            activeLeases =
                (activeLeases - 1)
                    .coerceAtLeast(0)
            if (activeLeases != 0) {
                return
            }

            val oldWorkers = workers
            workers = emptyList()
            nextWorker.set(0)

            oldWorkers.forEach { worker ->
                runCatching {
                    worker.quitSafely()
                }
            }
        }
    }

    private fun ensureCapacityForLeases(
        requestedLeases: Int,
    ) {
        if (requestedLeases <= 0) {
            return
        }

        if (workers.isEmpty()) {
            generation =
                generationCounter.incrementAndGet()
        }

        val targetWorkers =
            requestedLeases
                .coerceIn(1, maximumWorkers)

        while (workers.size < targetWorkers) {
            val index = workers.size
            workers =
                workers +
                    HandlerThread(
                        "NoX-CameraPool-$generation-$index",
                    ).apply {
                        start()
                    }
        }
    }

    companion object {
        private fun defaultWorkerCount(): Int =
            (
                Runtime.getRuntime()
                    .availableProcessors()
                    .coerceAtLeast(2) /
                    2
            ).coerceIn(2, 6)

        val shared =
            CameraCaptureHandlerPool()
    }

    private fun Int.floorMod(modulus: Int): Int {
        if (modulus <= 0) return 0
        return ((this % modulus) + modulus) % modulus
    }
}
