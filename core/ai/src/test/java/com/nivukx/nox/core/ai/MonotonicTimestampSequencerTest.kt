package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class MonotonicTimestampSequencerTest {

    @Test
    fun firstTimestampPreservesSourceMillisecond() {
        val sequencer = MonotonicTimestampSequencer()
        val ticket = sequencer.next(123_456_789L)

        assertEquals(123L, ticket.sourceTimestampMillis)
        assertEquals(123L, ticket.timestampMillis)
        assertFalse(ticket.adjusted)
        assertEquals(0L, ticket.deltaMillis)
    }

    @Test
    fun duplicateMillisecondIsAdvancedByMinimumGap() {
        val sequencer = MonotonicTimestampSequencer(minimumGapMillis = 1L)

        val first = sequencer.next(100_900_000L)
        val second = sequencer.next(100_950_000L)

        assertEquals(100L, first.timestampMillis)
        assertEquals(101L, second.timestampMillis)
        assertTrue(second.adjusted)
        assertEquals(1L, second.deltaMillis)
    }

    @Test
    fun backwardSourceTimestampNeverMovesTemporalModelBackward() {
        val sequencer = MonotonicTimestampSequencer()

        sequencer.next(500_000_000L)
        val second = sequencer.next(490_000_000L)

        assertEquals(501L, second.timestampMillis)
        assertTrue(second.adjusted)
        assertEquals(11L, second.deltaMillis)
    }

    @Test
    fun resetStartsAFreshTemporalSequence() {
        val sequencer = MonotonicTimestampSequencer()

        sequencer.next(500_000_000L)
        sequencer.reset()
        val ticket = sequencer.next(200_000_000L)

        assertEquals(200L, ticket.timestampMillis)
        assertFalse(ticket.adjusted)
        assertEquals(200L, sequencer.latestTimestampMillis())
        assertTrue(sequencer.hasObservedTimestamp())
    }
}
