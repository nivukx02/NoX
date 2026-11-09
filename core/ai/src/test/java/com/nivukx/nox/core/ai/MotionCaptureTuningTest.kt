package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MotionCaptureTuningTest {

    @Test
    fun presetsExposeAdvancedDynamicsControls() {
        listOf(
            MotionCaptureTuning.cinematic(),
            MotionCaptureTuning.action(),
            MotionCaptureTuning.precision(),
        ).forEach { tuning ->
            assertTrue(tuning.velocityResponsiveness in 0.55..1.60)
            assertTrue(tuning.accelerationResponsiveness in 0.55..1.60)
            assertTrue(tuning.innovationGateElasticity in 0.65..1.60)
            assertTrue(tuning.motionThresholdScale in 0.70..1.40)
            assertTrue(tuning.adaptiveRegimeStrength in 0.0..1.0)
        }
    }

    @Test
    fun estimatorRetainsConfiguredPolicy() {
        val engine = MotionCaptureEngine()
        val policy = MotionCaptureTuning.action()

        engine.configure(policy)

        assertEquals(
            policy,
            engine.tuning(),
        )
    }
}
