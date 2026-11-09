package com.nivukx.nox.core.model

/**
 * Immutable observation contract that carries spatial evidence together with the time at which
 * that evidence should be interpreted.
 *
 * Positive [rowTimestampOffsetNanos] means the observed row was sampled after the nominal frame
 * timestamp. [temporalAlignmentOffsetNanos] is the correction that maps this camera's clock onto
 * the rig reference timeline.
 */
data class SpatiotemporalObservation(
    val cameraId: String,
    val landmarkId: Int,
    val x: Double,
    val y: Double,
    val confidence: Double,
    val frameTimestampNanos: Long,
    val rowTimestampOffsetNanos: Long,
    val temporalAlignmentOffsetNanos: Long,
    val rollingShutterConfidence: Double,
    val sensorConfidence: Double,
) {
    init {
        require(cameraId.isNotBlank()) {
            "cameraId must not be blank"
        }
        require(landmarkId >= 0) {
            "landmarkId must be non-negative"
        }
        require(x.isFinite() && y.isFinite()) {
            "observation coordinates must be finite"
        }
        require(confidence in 0.0..1.0) {
            "confidence must be in [0,1]"
        }
        require(rollingShutterConfidence in 0.0..1.0) {
            "rollingShutterConfidence must be in [0,1]"
        }
        require(sensorConfidence in 0.0..1.0) {
            "sensorConfidence must be in [0,1]"
        }
        require(frameTimestampNanos >= 0L) {
            "frameTimestampNanos must be non-negative"
        }
    }

    fun sampleTimestampNanos(): Long =
        Math.addExact(
            Math.addExact(
                frameTimestampNanos,
                rowTimestampOffsetNanos,
            ),
            temporalAlignmentOffsetNanos,
        )

    val effectiveConfidence: Double
        get() = (
            confidence *
                rollingShutterConfidence *
                sensorConfidence
            ).coerceIn(0.0, 1.0)

    val temporalUncertaintyNanos: Long
        get() = (
            kotlin.math.abs(rowTimestampOffsetNanos) +
                kotlin.math.abs(temporalAlignmentOffsetNanos)
            ).coerceAtMost(Long.MAX_VALUE)

    fun isTemporallyCoherent(
        maximumTemporalUncertaintyNanos: Long,
    ): Boolean {
        require(maximumTemporalUncertaintyNanos >= 0L)
        return temporalUncertaintyNanos <=
            maximumTemporalUncertaintyNanos &&
            effectiveConfidence > 0.0
    }
}
