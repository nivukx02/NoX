package com.nivukx.nox.core.ai

import org.junit.Test
import kotlin.test.assertEquals

class PoseObservationTimingMetadataTest {
    @Test
    fun pose_observations_carry_source_frame_geometry_and_timing() {
        val result = PoseDetectionEngine.Result(
            timestampNanos = 100L,
            imageWidth = 1920,
            imageHeight = 1080,
            landmarks = listOf(
                PoseDetectionEngine.Landmark(
                    id = 11,
                    name = "right_wrist",
                    x = 1432f,
                    y = 812f,
                    confidence = 0.96f,
                ),
            ),
            confidence = 0.96f,
            rollingShutterSkewNanos = 12_000_000L,
            frameDurationNanos = 33_333_333L,
            exposureTimeNanos = 8_000_000L,
            captureTimingConfidence = 0.94,
        )

        val observation = result.observations().single()
        assertEquals(1920, observation.frameWidth)
        assertEquals(1080, observation.frameHeight)
        assertEquals(12_000_000L, observation.rollingShutterSkewNanos)
        assertEquals(33_333_333L, observation.frameDurationNanos)
        assertEquals(8_000_000L, observation.exposureTimeNanos)
        assertEquals(0.94, observation.captureTimingConfidence)
        assertEquals(812.0 / 1080.0, observation.sensorRowNormalized!!, 1e-9)
    }

    @Test
    fun rotated_detector_coordinates_store_sensor_plane_row() {
        val result = PoseDetectionEngine.Result(
            timestampNanos = 100L,
            imageWidth = 1920,
            imageHeight = 1080,
            rotationDegrees = 90,
            landmarks = listOf(
                PoseDetectionEngine.Landmark(
                    id = 13,
                    name = "left_wrist",
                    x = 240f,
                    y = 640f,
                    confidence = 0.95f,
                ),
            ),
            confidence = 0.95f,
        )

        val observation = result.observations().single()
        val expectedSensorRow =
            (1080.0 - 1.0 - 240.0) / 1080.0

        assertEquals(
            expectedSensorRow,
            observation.sensorRowNormalized!!,
            1e-9,
        )
    }
}
