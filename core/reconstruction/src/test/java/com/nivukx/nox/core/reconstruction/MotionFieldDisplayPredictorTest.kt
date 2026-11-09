package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MotionFieldActor
import com.nivukx.nox.core.model.MotionFieldGpuState
import com.nivukx.nox.core.model.MotionFieldJoint
import com.nivukx.nox.core.model.MotionFieldPreviewFrame
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MotionFieldDisplayPredictorTest {
    @Test
    fun applies_bounded_actor_velocity_when_display_is_ahead() {
        val frame = MotionFieldPreviewFrame(
            timestampNanos = 1_000_000_000L,
            cameraWidth = 1920,
            cameraHeight = 1080,
            displayTimestampNanos = 1_010_000_000L,
            actors = listOf(
                MotionFieldActor(
                    actorId = 1L,
                    confidence = 0.95,
                    predicted = false,
                    velocityXNormalizedPerSecond = 0.10,
                    velocityYNormalizedPerSecond = -0.05,
                    joints = listOf(
                        MotionFieldJoint(
                            landmarkId = 0,
                            x = 0.50,
                            y = 0.40,
                            confidence = 0.95,
                            predicted = false,
                        ),
                    ),
                ),
            ),
            reconstructionConfidence = 0.90,
            gpu = MotionFieldGpuState(),
        )

        val synchronizer = MotionFieldFrameSynchronizer()
        synchronizer.submit(frame)
        val synchronizedFrame =
            synchronizer.select(1_010_000_000L).frame!!
        val predicted =
            MotionFieldDisplayPredictor().predict(synchronizedFrame)
        val joint = predicted.actors.single().joints.single()

        assertEquals(0.501, joint.x, 0.000001)
        assertEquals(0.3995, joint.y, 0.000001)
        assertTrue(predicted.predictedToDisplay)
        assertTrue(joint.predicted)
    }

    @Test
    fun leaves_measurement_unchanged_when_target_is_not_ahead() {
        val frame = MotionFieldPreviewFrame(
            timestampNanos = 2_000_000_000L,
            cameraWidth = 1920,
            cameraHeight = 1080,
            displayTimestampNanos = 2_000_000_000L,
            actors = listOf(
                MotionFieldActor(
                    actorId = 2L,
                    confidence = 0.90,
                    predicted = false,
                    joints = listOf(
                        MotionFieldJoint(0, 0.3, 0.4, 0.9, false),
                    ),
                ),
            ),
            reconstructionConfidence = 0.8,
            gpu = MotionFieldGpuState(),
        )

        val predicted = MotionFieldDisplayPredictor().predict(frame)

        assertEquals(frame, predicted)
    }
}
