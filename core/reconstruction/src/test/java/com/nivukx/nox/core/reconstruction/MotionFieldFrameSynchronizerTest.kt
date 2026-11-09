package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MotionFieldGpuState
import com.nivukx.nox.core.model.MotionFieldPreviewFrame
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class MotionFieldFrameSynchronizerTest {
    @Test
    fun runtime_configuration_updates_prediction_and_stale_windows() {
        val engine = MotionFieldFrameSynchronizer()
        engine.configure(
            maxStaleNanos = 40_000_000L,
            maxPredictionNanos = 5_000_000L,
        )
        engine.submit(frame(100_000_000L))

        val predicted = engine.select(104_000_000L)
        val stale = engine.select(150_000_000L)

        assertTrue(predicted.predictedToDisplay)
        assertTrue(stale.stale)
    }

    @Test
    fun selects_closest_sensor_timestamp() {
        val engine = MotionFieldFrameSynchronizer(
            maxHistory = 8,
            maxStaleNanos = 20_000_000L,
            maxPredictionNanos = 10_000_000L,
        )
        engine.submit(frame(100_000_000L))
        engine.submit(frame(116_000_000L))

        val selection = engine.select(111_000_000L)

        assertEquals(116_000_000L, selection.frame?.timestampNanos)
        assertFalse(selection.stale)
        assertFalse(selection.predictedToDisplay)
    }

    @Test
    fun allows_bounded_future_display_prediction() {
        val engine = MotionFieldFrameSynchronizer(
            maxHistory = 8,
            maxStaleNanos = 40_000_000L,
            maxPredictionNanos = 10_000_000L,
        )
        engine.submit(frame(100_000_000L))

        val selection = engine.select(108_000_000L)

        assertEquals(100_000_000L, selection.frame?.timestampNanos)
        assertTrue(selection.predictedToDisplay)
        assertEquals(8_000_000L, selection.deltaToDisplayNanos)
    }

    @Test
    fun marks_stale_when_motion_state_is_too_old() {
        val engine = MotionFieldFrameSynchronizer(
            maxHistory = 8,
            maxStaleNanos = 20_000_000L,
            maxPredictionNanos = 10_000_000L,
        )
        engine.submit(frame(100_000_000L))

        val selection = engine.select(150_000_000L)

        assertTrue(selection.stale)
        assertEquals(100_000_000L, selection.frame?.timestampNanos)
    }

    @Test
    fun out_of_order_submission_does_not_move_published_time_backward() {
        val engine = MotionFieldFrameSynchronizer()
        engine.submit(frame(200_000_000L))
        engine.submit(frame(180_000_000L))

        assertEquals(200_000_000L, engine.latestTimestampNanos())
    }

    private fun frame(timestampNanos: Long) = MotionFieldPreviewFrame(
        timestampNanos = timestampNanos,
        cameraWidth = 1920,
        cameraHeight = 1080,
        displayTimestampNanos = timestampNanos,
        actors = emptyList(),
        reconstructionConfidence = 0.9,
        gpu = MotionFieldGpuState(),
    )
}
