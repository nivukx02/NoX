package com.nivukx.nox.core.ai

import com.nivukx.nox.core.nativeengine.tracking.NativeBodyObservation

/**
 * Additive bridge from the existing PoseDetectionEngine output to the native deterministic tracker.
 * The MediaPipe/ML Kit detector remains the producer of measurements.
 */
object NativeBodyTrackingBridge {
    fun observations(result: PoseDetectionEngine.Result): List<NativeBodyObservation> =
        result.landmarks.map {
            NativeBodyObservation(
                id = it.id,
                x = (it.x / result.imageWidth.coerceAtLeast(1)).toDouble().coerceIn(0.0, 1.0),
                y = (it.y / result.imageHeight.coerceAtLeast(1)).toDouble().coerceIn(0.0, 1.0),
                confidence = it.confidence.toDouble().coerceIn(0.0, 1.0),
                timestampNanos = result.timestampNanos,
            )
        }
}
