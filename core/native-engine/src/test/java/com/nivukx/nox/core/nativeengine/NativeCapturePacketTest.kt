package com.nivukx.nox.core.nativeengine

import org.junit.Assert.assertEquals
import org.junit.Test

class NativeCapturePacketTest {
    @Test
    fun emptyObservationIsValid() {
        val packet = NativeCapturePacket(
            cameraId = 7,
            frameId = 42L,
            sensorTimestampNanos = 1_000_000L,
            monotonicTimestampNanos = 2_000_000L,
            exposureDurationNanos = 10_000_000L,
            rollingShutterSkewNanos = 100_000L,
            keypointXy = FloatArray(0),
            keypointConfidence = FloatArray(0),
        )
        assertEquals(0, packet.keypointCount)
        assertEquals(33_333_333L, packet.nominalFrameIntervalNanos)
    }

    @Test
    fun measured_frame_duration_overrides_requested_nominal_cadence() {
        val stamp =
            com.nivukx.nox.core.model.FrameStamp(
                sequence = 7L,
                sensorTimestampNanos = 1_000_000L,
                monotonicTimestampNanos = 2_000_000L,
                exposureTimeNanos = 1_000_000L,
                iso = 100,
            )
        val telemetry =
            com.nivukx.nox.core.model.FrameTelemetry(
                sensorTimestampNanos = stamp.sensorTimestampNanos,
                monotonicTimestampNanos = stamp.monotonicTimestampNanos,
                exposureTimeNanos = stamp.exposureTimeNanos,
                sensorSensitivityIso = 100,
                frameDurationNanos = 8_333_333L,
            )

        val packet =
            NativeCapturePacket.fromCapture(
                cameraId = "high-speed",
                stamp = stamp,
                telemetry = telemetry,
                nominalFrameIntervalNanos = 16_666_667L,
            )

        assertEquals(
            8_333_333L,
            packet.nominalFrameIntervalNanos,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOddKeypointArray() {
        NativeCapturePacket(
            cameraId = 7,
            frameId = 42L,
            sensorTimestampNanos = 1L,
            monotonicTimestampNanos = 2L,
            exposureDurationNanos = 1L,
            rollingShutterSkewNanos = 0L,
            keypointXy = floatArrayOf(1f),
            keypointConfidence = FloatArray(0),
        )
    }
}
