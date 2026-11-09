package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.MotionFieldJoint
import com.nivukx.nox.core.model.MotionFieldPreviewFrame

/**
 * Applies bounded sensor-to-display prediction to the display copy of a MotionField frame.
 * Recorded sensor measurements remain unchanged.
 */
class MotionFieldDisplayPredictor(
    private val maximumPredictionNanos: Long = 18_000_000L,
) {
    init {
        require(maximumPredictionNanos >= 0L)
    }

    fun predict(frame: MotionFieldPreviewFrame): MotionFieldPreviewFrame {
        if (!frame.predictedToDisplay) return frame
        val deltaNanos =
            (frame.displayTimestampNanos - frame.timestampNanos)
                .coerceIn(0L, maximumPredictionNanos)
        if (deltaNanos <= 0L) return frame

        val dt = deltaNanos / 1e9
        val actors = frame.actors.map { actor ->
            val dx =
                actor.velocityXNormalizedPerSecond * dt
            val dy =
                actor.velocityYNormalizedPerSecond * dt
            actor.copy(
                predicted = true,
                confidence =
                    (actor.confidence * 0.992)
                        .coerceIn(0.0, 1.0),
                joints = actor.joints.map { joint ->
                    predictJoint(
                        joint = joint,
                        dx = dx,
                        dy = dy,
                    )
                },
            )
        }

        return frame.copy(actors = actors)
    }

    private fun predictJoint(
        joint: MotionFieldJoint,
        dx: Double,
        dy: Double,
    ): MotionFieldJoint =
        joint.copy(
            x = (joint.x + dx).coerceIn(0.0, 1.0),
            y = (joint.y + dy).coerceIn(0.0, 1.0),
            predicted = true,
            confidence = (joint.confidence * 0.990).coerceIn(0.0, 1.0),
        )
}
