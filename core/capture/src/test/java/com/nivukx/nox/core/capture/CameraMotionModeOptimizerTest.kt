package com.nivukx.nox.core.capture

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CameraMotionModeOptimizerTest {

    @Test
    fun favors_real_high_frame_rate_mode_when_1080p_cannot_sustain_it() {
        val optimizer =
            CameraMotionModeOptimizer()

        val result =
            optimizer.select(
                candidates =
                    listOf(
                        CameraMotionModeOptimizer.Candidate(
                            width = 1920,
                            height = 1080,
                            maximumFps = 30,
                        ),
                        CameraMotionModeOptimizer.Candidate(
                            width = 1280,
                            height = 720,
                            maximumFps = 120,
                        ),
                    ),
                advertisedMaximumFps = 120,
            )

        assertNotNull(result)
        assertEquals(120, result.fps)
        assertEquals(1280, result.width)
        assertEquals(720, result.height)
        assertTrue(result.temporalScore > result.spatialScore)
    }

    @Test
    fun never_selects_below_minimum_fps_or_above_advertised_maximum() {
        val optimizer =
            CameraMotionModeOptimizer(
                CameraMotionModeOptimizer.Config(
                    minimumFps = 30,
                    maximumFps = 240,
                ),
            )

        val result =
            optimizer.select(
                candidates =
                    listOf(
                        CameraMotionModeOptimizer.Candidate(
                            width = 1920,
                            height = 1080,
                            maximumFps = 24,
                        ),
                        CameraMotionModeOptimizer.Candidate(
                            width = 1280,
                            height = 720,
                            maximumFps = 90,
                        ),
                    ),
                advertisedMaximumFps = 60,
            )

        assertNotNull(result)
        assertEquals(60, result.fps)
        assertTrue(result.fps >= 30)
    }
}
