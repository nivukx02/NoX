package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.LandmarkSpace
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ObservationFusionBufferTest {

    @Test
    fun exposesRecentFramesWithoutMutatingHistory() {
        val buffer =
            ObservationFusionBuffer(
                historyPerDevice = 4,
                maxSkewNanos = 25_000_000L,
                maximumLandmarksPerFrame = 8,
                maximumStoredLandmarksPerDevice = 32,
            )

        val device =
            DeviceId("cam-a")

        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = device,
                timestampNanos = 1_000_000L,
                landmarks = emptyList(),
            ),
        )
        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = device,
                timestampNanos = 2_000_000L,
                landmarks = emptyList(),
            ),
        )

        val recent =
            buffer.recentFramesSince(
                minimumTimestampNanos = 1_500_000L,
            )

        assertEquals(1, recent.size)
        assertEquals(2_000_000L, recent.single().timestampNanos)
        assertEquals(2, buffer.snapshot().size)
    }

    @Test
    fun perFrameLandmarkPayloadIsBounded() {
        val buffer =
            ObservationFusionBuffer(
                maximumLandmarksPerFrame = 3,
                maximumStoredLandmarksPerDevice = 12,
            )
        val frame =
            ObservationFusionBuffer.DeviceFrame(
                deviceId = DeviceId("cam-a"),
                timestampNanos = 1L,
                landmarks = landmarks(8, 1L),
            )

        buffer.push(frame)

        assertEquals(
            3,
            buffer.match(DeviceId("cam-a"), 1L)?.landmarks?.size,
        )
        assertEquals(
            3,
            buffer.telemetry().storedLandmarkCount,
        )
        assertEquals(
            5L,
            buffer.telemetry().droppedLandmarkCount,
        )
    }

    @Test
    fun totalDevicePayloadIsTrimmedOldestFirst() {
        val buffer =
            ObservationFusionBuffer(
                historyPerDevice = 8,
                maximumLandmarksPerFrame = 4,
                maximumStoredLandmarksPerDevice = 5,
            )

        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = DeviceId("cam-a"),
                timestampNanos = 1L,
                landmarks = landmarks(4, 1L),
            ),
        )
        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = DeviceId("cam-a"),
                timestampNanos = 2L,
                landmarks = landmarks(4, 2L),
            ),
        )

        val telemetry = buffer.telemetry()
        assertTrue(telemetry.storedLandmarkCount <= 5)
        assertTrue(telemetry.droppedFrameCount > 0L)
        assertTrue(telemetry.peakStoredLandmarksPerDevice >= 4)
    }

    @Test
    fun timestampReversalReplacesNewestFrameInsteadOfGrowingHistory() {
        val buffer =
            ObservationFusionBuffer(
                historyPerDevice = 4,
                maximumLandmarksPerFrame = 8,
                maximumStoredLandmarksPerDevice = 32,
            )

        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = DeviceId("cam-a"),
                timestampNanos = 10L,
                landmarks = landmarks(2, 10L),
            ),
        )
        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = DeviceId("cam-a"),
                timestampNanos = 9L,
                landmarks = landmarks(2, 9L),
            ),
        )

        assertEquals(1, buffer.snapshot().size)
        assertEquals(9L, buffer.snapshot().single().timestampNanos)
    }

    private fun landmarks(
        count: Int,
        timestampNanos: Long,
    ): List<LandmarkObservation> =
        (0 until count).map { index ->
            LandmarkObservation(
                landmarkId = index,
                x = 100.0 + index,
                y = 200.0 + index,
                confidence = 0.90,
                timestampNanos = timestampNanos,
                space = LandmarkSpace.PIXELS,
                frameWidth = 1920,
                frameHeight = 1080,
            )
        }
}
