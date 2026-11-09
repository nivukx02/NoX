package com.nivukx.nox.core.model

import kotlin.test.assertEquals
import org.junit.Test

class CameraObservationTimingTest {
    @Test
    fun row_time_offset_preserves_signed_sensor_row_timing() {
        val observation = CameraObservation(
            deviceId = DeviceId("cam-a"),
            point = Vec2(100.0, 200.0),
            confidence = 0.9,
            intrinsics = CameraIntrinsics(
                fx = 1000.0,
                fy = 1000.0,
                cx = 500.0,
                cy = 500.0,
                imageWidth = 1000,
                imageHeight = 1000,
            ),
            pose = Pose(),
            rowTimeOffsetNanos = -3_000_000L,
        )

        assertEquals(-3_000_000L, observation.rowTimeOffsetNanos)
    }

    @Test
    fun exposureAndReadoutTimingExposeConservativeOpticalRisk() {
        val clear =
            CameraObservation(
                deviceId = DeviceId("cam-a"),
                point = Vec2(100.0, 200.0),
                confidence = 0.9,
                intrinsics = CameraIntrinsics(1000.0, 1000.0, 500.0, 500.0),
                pose = Pose(),
                frameDurationNanos = 8_333_333L,
                exposureTimeNanos = 1_000_000L,
                rollingShutterFraction = 0.05,
                captureTimingConfidence = 0.99,
            )
        val degraded =
            clear.copy(
                exposureTimeNanos = 7_000_000L,
                rollingShutterFraction = 0.70,
                captureTimingConfidence = 0.72,
            )

        assert(clear.estimatedMotionBlurRisk < degraded.estimatedMotionBlurRisk)
        assert(degraded.estimatedMotionBlurRisk in 0.0..1.0)
    }
}
