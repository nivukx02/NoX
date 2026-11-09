package com.nivukx.nox.core.model

import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import org.junit.Test

class MotionFieldPreviewModelTest {
    @Test
    fun preview_frame_rejects_invalid_timestamp_and_confidence() {
        assertFailsWith<IllegalArgumentException> {
            MotionFieldPreviewFrame(
                timestampNanos = -1L,
                cameraWidth = 1920,
                cameraHeight = 1080,
                displayTimestampNanos = 0L,
                actors = emptyList(),
                reconstructionConfidence = 0.8,
                gpu = MotionFieldGpuState(),
            )
        }
    }

    @Test
    fun preview_frame_accepts_measurement_and_prediction_states() {
        val frame = MotionFieldPreviewFrame(
            timestampNanos = 1_000L,
            cameraWidth = 1920,
            cameraHeight = 1080,
            displayTimestampNanos = 2_000L,
            actors = listOf(
                MotionFieldActor(
                    actorId = 1L,
                    confidence = 0.94,
                    predicted = false,
                    joints = listOf(
                        MotionFieldJoint(
                            landmarkId = 0,
                            x = 0.5,
                            y = 0.4,
                            confidence = 0.92,
                            predicted = false,
                        ),
                    ),
                ),
            ),
            reconstructionJoints3d = listOf(
                MotionFieldWorldJoint(
                    landmarkId = 0,
                    xMeters = 0.0,
                    yMeters = 0.9,
                    zMeters = 1.2,
                    confidence = 0.88,
                ),
            ),
            reconstructionConfidence = 0.88,
            gpu = MotionFieldGpuState(),
        )

        assertEquals(1, frame.actorCount)
        assertEquals(1, frame.totalJointCount)
    }
}
