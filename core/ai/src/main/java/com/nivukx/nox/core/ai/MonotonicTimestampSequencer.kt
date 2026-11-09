package com.nivukx.nox.core.ai

/**
 * Converts camera nanosecond timestamps to the millisecond timebase required by MediaPipe VIDEO
 * mode while preserving strict monotonicity.
 *
 * Camera timestamps can legitimately produce equal millisecond values at high frame rates because
 * integer division discards sub-millisecond precision. MediaPipe explicitly requires monotonically
 * increasing video timestamps, so this class performs the smallest possible forward adjustment and
 * records that adjustment as evidence instead of silently changing the source timestamp.
 */
class MonotonicTimestampSequencer(
    private val minimumGapMillis: Long = 1L,
) {
    data class Ticket(
        val sourceTimestampNanos: Long,
        val sourceTimestampMillis: Long,
        val timestampMillis: Long,
        val adjusted: Boolean,
        val deltaMillis: Long,
    )

    private var lastTimestampMillis = Long.MIN_VALUE

    init {
        require(minimumGapMillis >= 1L)
    }

    @Synchronized
    fun next(sourceTimestampNanos: Long): Ticket {
        require(sourceTimestampNanos >= 0L)

        val sourceMillis = sourceTimestampNanos / 1_000_000L
        val minimumNext =
            if (lastTimestampMillis == Long.MIN_VALUE) {
                sourceMillis
            } else {
                lastTimestampMillis + minimumGapMillis
            }

        val timestampMillis = maxOf(
            sourceMillis,
            minimumNext,
        )

        lastTimestampMillis = timestampMillis

        return Ticket(
            sourceTimestampNanos = sourceTimestampNanos,
            sourceTimestampMillis = sourceMillis,
            timestampMillis = timestampMillis,
            adjusted = timestampMillis != sourceMillis,
            deltaMillis = timestampMillis - sourceMillis,
        )
    }

    @Synchronized
    fun reset() {
        lastTimestampMillis = Long.MIN_VALUE
    }

    @Synchronized
    fun latestTimestampMillis(): Long? =
        lastTimestampMillis.takeIf {
            it != Long.MIN_VALUE
        }

    @Synchronized
    fun hasObservedTimestamp(): Boolean =
        lastTimestampMillis != Long.MIN_VALUE
}
