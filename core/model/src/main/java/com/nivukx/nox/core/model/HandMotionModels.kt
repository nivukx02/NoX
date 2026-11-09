package com.nivukx.nox.core.model

/**
 * High-fidelity hand motion contract.
 *
 * A hand frame keeps image-space landmarks, metric model-space landmarks when the inference
 * provider exposes them, handedness confidence, and temporal diagnostics. This is intentionally
 * independent of a specific inference provider so NoX can replace MediaPipe without rewriting
 * the transport or reconstruction layers.
 */
enum class HandSide {
    LEFT,
    RIGHT,
    UNKNOWN,
}

data class HandLandmark(
    val index: Int,
    val normalizedX: Double,
    val normalizedY: Double,
    val normalizedZ: Double,
    val worldPositionMeters: Vec3?,
    val confidence: Double,
)

data class HandObservation(
    val handId: Long,
    val side: HandSide,
    val handednessConfidence: Double,
    val landmarks: List<HandLandmark>,
    val confidence: Double,
    val timestampNanos: Long,
    val wristVelocityMetersPerSecond: Vec3 = Vec3(0.0, 0.0, 0.0),
    val temporalStability: Double = 0.0,
    val predicted: Boolean = false,
) {
    init {
        require(landmarks.size <= 21)
        require(handId >= 0L)
        require(timestampNanos >= 0L)
    }

    val wrist: HandLandmark?
        get() = landmarks.firstOrNull { it.index == 0 }

    val palmCenter: Vec3?
        get() {
            val metric = landmarks.mapNotNull { it.worldPositionMeters }
            if (metric.isEmpty()) return null
            return metric.reduce(Vec3::plus) / metric.size.toDouble()
        }
}

data class HandTrackingFrame(
    val timestampNanos: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val hands: List<HandObservation>,
    val inferenceLatencyMillis: Double,
    val modelReady: Boolean,
    val droppedByBackpressure: Boolean = false,
    val staleEvidence: Boolean = false,
    val captureTimingConfidence: Double = 1.0,
    val rollingShutterSkewNanos: Long = 0L,
    val frameDurationNanos: Long = 0L,
    val exposureTimeNanos: Long = 0L,
) {
    val trackedHands: Int
        get() = hands.size

    /** Conservative optical risk derived from retained capture timing metadata. */
    val estimatedMotionBlurRisk: Double
        get() = estimateOpticalMotionBlurRisk(
            captureTimingConfidence = captureTimingConfidence,
            rollingShutterSkewNanos = rollingShutterSkewNanos,
            frameDurationNanos = frameDurationNanos,
            exposureTimeNanos = exposureTimeNanos,
        )

    val meanConfidence: Double
        get() = if (hands.isEmpty()) 0.0
        else hands.map { it.confidence }.average()
}

data class HandTrackingDiagnostics(
    val timestampNanos: Long,
    val modelReady: Boolean,
    val activeHands: Int,
    val meanConfidence: Double,
    val meanTemporalStability: Double,
    val maxWristSpeedMetersPerSecond: Double,
    val inferenceLatencyMillis: Double,
    val droppedFrames: Long,
    val provider: String,
)