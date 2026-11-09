package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MotionFieldPreviewFrame
import java.util.TreeMap

/**
 * Sensor-time selector for live MotionField presentation.
 *
 * The synchronizer never rewrites the authoritative capture timestamp. It only chooses the
 * best-known frame for a display timestamp and explicitly reports when the display target is
 * ahead of the measured motion state or when the state is stale.
 */
class MotionFieldFrameSynchronizer(
    private val maxHistory: Int = 24,
    maxStaleNanos: Long = 120_000_000L,
    maxPredictionNanos: Long = 18_000_000L,
) {
    data class Selection(
        val frame: MotionFieldPreviewFrame?,
        val targetTimestampNanos: Long,
        val deltaToDisplayNanos: Long,
        val predictedToDisplay: Boolean,
        val stale: Boolean,
        val interpolationAlpha: Double,
    )

    private var maxStaleNanos = maxStaleNanos
    private var maxPredictionNanos = maxPredictionNanos

    private val frames = TreeMap<Long, MotionFieldPreviewFrame>()
    private var newestTimestampNanos: Long = Long.MIN_VALUE

    init {
        require(maxHistory >= 2)
        require(maxStaleNanos >= 0L)
        require(maxPredictionNanos >= 0L)
        require(maxPredictionNanos <= maxStaleNanos || maxStaleNanos == 0L)
    }

    @Synchronized
    fun configure(
        maxStaleNanos: Long = this.maxStaleNanos,
        maxPredictionNanos: Long = this.maxPredictionNanos,
    ) {
        require(maxStaleNanos >= 0L)
        require(maxPredictionNanos >= 0L)
        require(
            maxPredictionNanos <= maxStaleNanos ||
                maxStaleNanos == 0L,
        )
        this.maxStaleNanos = maxStaleNanos
        this.maxPredictionNanos = maxPredictionNanos
    }

    @Synchronized
    fun submit(frame: MotionFieldPreviewFrame) {
        if (frame.timestampNanos < newestTimestampNanos) {
            return
        }
        frames[frame.timestampNanos] = frame
        newestTimestampNanos = maxOf(newestTimestampNanos, frame.timestampNanos)
        while (frames.size > maxHistory) {
            frames.pollFirstEntry()
        }
    }

    @Synchronized
    fun select(targetTimestampNanos: Long): Selection {
        require(targetTimestampNanos >= 0L)
        if (frames.isEmpty()) {
            return Selection(
                frame = null,
                targetTimestampNanos = targetTimestampNanos,
                deltaToDisplayNanos = 0L,
                predictedToDisplay = false,
                stale = true,
                interpolationAlpha = 0.0,
            )
        }

        val floor = frames.floorEntry(targetTimestampNanos)?.value
        val ceil = frames.ceilingEntry(targetTimestampNanos)?.value

        val candidate = when {
            floor == null -> ceil
            ceil == null -> floor
            targetTimestampNanos - floor.timestampNanos <= ceil.timestampNanos - targetTimestampNanos -> floor
            else -> ceil
        } ?: return Selection(null, targetTimestampNanos, 0L, false, true, 0.0)

        val delta = targetTimestampNanos - candidate.timestampNanos
        val absoluteDelta = kotlin.math.abs(delta)
        val predicted =
            delta > 0L && delta <= maxPredictionNanos

        val stale =
            absoluteDelta > maxStaleNanos

        val alpha = when {
            floor == null || ceil == null || floor.timestampNanos == ceil.timestampNanos -> 0.0
            else -> (
                (targetTimestampNanos - floor.timestampNanos).toDouble() /
                    (ceil.timestampNanos - floor.timestampNanos).toDouble()
            ).coerceIn(0.0, 1.0)
        }

        return Selection(
            frame = candidate.copy(
                displayTimestampNanos = targetTimestampNanos,
                stale = stale,
                predictedToDisplay = predicted,
            ),
            targetTimestampNanos = targetTimestampNanos,
            deltaToDisplayNanos = delta,
            predictedToDisplay = predicted,
            stale = stale,
            interpolationAlpha = alpha,
        )
    }

    @Synchronized
    fun latestTimestampNanos(): Long = newestTimestampNanos

    @Synchronized
    fun clear() {
        frames.clear()
        newestTimestampNanos = Long.MIN_VALUE
    }
}
