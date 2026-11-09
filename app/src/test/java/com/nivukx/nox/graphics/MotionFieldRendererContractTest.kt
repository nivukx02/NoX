package com.nivukx.nox.graphics

import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import org.junit.Test

class MotionFieldRendererContractTest {
    @Test
    fun renderer_health_starts_new_and_accepts_monotonic_counters() {
        val health = MotionFieldRendererHealth()
        assertEquals(MotionFieldRendererLifecycle.NEW, health.lifecycle)
        assertEquals(0L, health.frameCount)
    }

    @Test
    fun negative_renderer_counters_are_rejected() {
        assertFailsWith<IllegalArgumentException> {
            MotionFieldRendererHealth(frameCount = -1L)
        }
        assertFailsWith<IllegalArgumentException> {
            MotionFieldRendererHealth(droppedFrames = -1L)
        }
    }
}
