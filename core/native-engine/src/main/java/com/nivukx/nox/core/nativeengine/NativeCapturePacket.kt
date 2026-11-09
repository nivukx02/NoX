package com.nivukx.nox.core.nativeengine

import com.nivukx.nox.core.model.FrameStamp
import com.nivukx.nox.core.model.FrameTelemetry

data class NativeCapturePacket(
    val cameraId: Int,
    val frameId: Long,
    val sensorTimestampNanos: Long,
    val monotonicTimestampNanos: Long,
    val exposureDurationNanos: Long,
    val rollingShutterSkewNanos: Long,
    val keypointXy: FloatArray,
    val keypointConfidence: FloatArray,
    val keypointIds: IntArray = IntArray(0),
    val nominalFrameIntervalNanos: Long = 33_333_333L,
) {
    val keypointCount: Int get() = keypointConfidence.size

    init {
        require(cameraId >= 0)
        require(frameId >= 0L)
        require(sensorTimestampNanos >= 0L)
        require(monotonicTimestampNanos >= 0L)
        require(exposureDurationNanos >= 0L)
        require(rollingShutterSkewNanos >= 0L)
        require(
            nominalFrameIntervalNanos in
                1_000_000L..1_000_000_000L
        )
        require(keypointXy.size == keypointConfidence.size * 2)
        require(keypointIds.isEmpty() || keypointIds.size == keypointCount)
        require(keypointCount <= 128)
        if (keypointIds.isNotEmpty()) {
            require(keypointIds.all { it >= 0 })
        }
    }

    companion object {
        fun cameraIdFromString(value: String?): Int {
            if (value.isNullOrEmpty()) return 0
            var hash = 0x811C9DC5.toInt()
            for (ch in value) hash = (hash xor ch.code) * 0x01000193
            return hash and Int.MAX_VALUE
        }

        fun fromCapture(
            cameraId: String?,
            stamp: FrameStamp,
            telemetry: FrameTelemetry?,
            nominalFrameIntervalNanos: Long = 33_333_333L,
        ): NativeCapturePacket(
            cameraId = cameraIdFromString(cameraId),
            frameId = stamp.sequence,
            sensorTimestampNanos = stamp.sensorTimestampNanos,
            monotonicTimestampNanos = stamp.monotonicTimestampNanos,
            exposureDurationNanos = telemetry?.exposureTimeNanos ?: stamp.exposureTimeNanos,
            rollingShutterSkewNanos = telemetry?.rollingShutterSkewNanos ?: 0L,
            keypointXy = FloatArray(0),
            keypointConfidence = FloatArray(0),
            nominalFrameIntervalNanos =
                (
                    telemetry?.frameDurationNanos
                        ?.takeIf {
                            it in
                                1_000_000L..1_000_000_000L
                        }
                        ?: nominalFrameIntervalNanos
                ).coerceIn(
                    1_000_000L,
                    1_000_000_000L,
                ),
        )
    }
}
