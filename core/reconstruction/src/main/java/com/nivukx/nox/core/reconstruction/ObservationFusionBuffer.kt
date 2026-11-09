package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.LandmarkObservation
import kotlin.math.abs
import java.util.ArrayDeque

/**
 * Timestamp-indexed multi-phone measurement buffer.
 *
 * Every device can arrive over Wi-Fi at a different wall-clock time. Frames are held briefly and
 * matched against a requested reconstruction timestamp using monotonic sensor timestamps.
 */
class ObservationFusionBuffer(
    private val historyPerDevice: Int = 8,
    private val maxSkewNanos: Long = 25_000_000L,
    private val maximumLandmarksPerFrame: Int = 4096,
    private val maximumStoredLandmarksPerDevice: Int = 32768,
) {
    init {
        require(historyPerDevice >= 1)
        require(maxSkewNanos >= 0L)
        require(maximumLandmarksPerFrame >= 1)
        require(maximumStoredLandmarksPerDevice >= maximumLandmarksPerFrame)
    }

    data class DeviceFrame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val landmarks: List<LandmarkObservation>,
    )

    data class Telemetry(
        val deviceCount: Int,
        val storedFrameCount: Int,
        val storedLandmarkCount: Int,
        val droppedFrameCount: Long,
        val droppedLandmarkCount: Long,
        val peakStoredLandmarksPerDevice: Int,
    )

    private val history = HashMap<DeviceId, ArrayDeque<DeviceFrame>>()
    private var droppedFrameCount = 0L
    private var droppedLandmarkCount = 0L
    private var peakStoredLandmarksPerDevice = 0

    @Synchronized
    fun push(frame: DeviceFrame) {
        require(frame.timestampNanos >= 0L)

        val boundedLandmarks =
            if (frame.landmarks.size <= maximumLandmarksPerFrame) {
                frame.landmarks
            } else {
                droppedLandmarkCount +=
                    (frame.landmarks.size - maximumLandmarksPerFrame)
                        .toLong()
                frame.landmarks.take(maximumLandmarksPerFrame)
            }

        val boundedFrame =
            if (boundedLandmarks === frame.landmarks) {
                frame
            } else {
                frame.copy(landmarks = boundedLandmarks)
            }

        val queue = history.getOrPut(boundedFrame.deviceId) { ArrayDeque() }

        if (
            queue.isNotEmpty() &&
            boundedFrame.timestampNanos <= queue.last().timestampNanos
        ) {
            queue.removeLast()
        }

        if (queue.size >= historyPerDevice) {
            queue.removeFirst()
            droppedFrameCount += 1L
        }

        queue.addLast(boundedFrame)

        while (
            queue.sumOf { it.landmarks.size } >
            maximumStoredLandmarksPerDevice
        ) {
            if (queue.isEmpty()) break
            val oldest = queue.removeFirst()
            droppedLandmarkCount += oldest.landmarks.size.toLong()
            droppedFrameCount += 1L
        }

        peakStoredLandmarksPerDevice =
            maxOf(
                peakStoredLandmarksPerDevice,
                queue.sumOf { it.landmarks.size },
            )
    }

    @Synchronized
    fun telemetry(): Telemetry =
        Telemetry(
            deviceCount = history.size,
            storedFrameCount = history.values.sumOf { it.size },
            storedLandmarkCount =
                history.values.sumOf { queue ->
                    queue.sumOf { it.landmarks.size }
                },
            droppedFrameCount = droppedFrameCount,
            droppedLandmarkCount = droppedLandmarkCount,
            peakStoredLandmarksPerDevice =
                peakStoredLandmarksPerDevice,
        )

    @Synchronized
    fun match(deviceId: DeviceId, timestampNanos: Long): DeviceFrame? =
        history[deviceId]
            ?.minByOrNull { abs(it.timestampNanos - timestampNanos) }
            ?.takeIf { abs(it.timestampNanos - timestampNanos) <= maxSkewNanos }

    @Synchronized
    fun matchAll(timestampNanos: Long): List<DeviceFrame> =
        history.keys.mapNotNull { match(it, timestampNanos) }

    @Synchronized
    fun snapshot(): List<DeviceFrame> =
        history.values
            .flatMap { it.toList() }
            .sortedBy { it.timestampNanos }

    @Synchronized
    fun recentFramesSince(
        minimumTimestampNanos: Long,
    ): List<DeviceFrame> =
        snapshot().filter {
            it.timestampNanos >=
                minimumTimestampNanos
        }

    @Synchronized
    fun clear() {
        history.clear()
        droppedFrameCount = 0L
        droppedLandmarkCount = 0L
        peakStoredLandmarksPerDevice = 0
    }

    @Synchronized
    fun clear(deviceId: DeviceId) {
        history.remove(deviceId)
    }
}
