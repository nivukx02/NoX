package com.nivukx.nox.core.reconstruction

import kotlin.math.max

import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.RollingShutterModel
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs

/**
 * Converts a rolling-shutter observation into a common-time observation using a measured
 * line-readout model and an estimated world-space velocity.
 *
 * The correction deliberately stays conservative. Large extrapolations reduce confidence rather
 * than inventing sub-frame motion that the sensor never observed.
 */
class RollingShutterCompensator(
    private val maxCorrectionPixels: Double = 24.0,
    private val correctionStrength: Double = 1.0,
) {
    init {
        require(maxCorrectionPixels > 0.0)
        require(correctionStrength in 0.0..1.0)
    }

    data class Corrected(
        val point: Vec2,
        val temporalOffsetNanos: Long,
        val confidenceScale: Double,
    )

    fun correct(
        observation: CameraObservation,
        frameHeight: Int,
        model: RollingShutterModel,
        velocityMetersPerSecond: Vec3,
        targetTimestampNanos: Long,
        frameTimestampNanos: Long,
        referenceDepthMeters: Double = 2.0,
        angularVelocityRadPerSecond: Vec3 =
            Vec3(0.0, 0.0, 0.0),
    ): Corrected {
        val height = frameHeight.coerceAtLeast(1)
        val row = (
            observation.point.y / height.toDouble()
        ).coerceIn(0.0, 1.0)

        val sampleTime = model.rowTimestamp(
            frameTimestampNanos,
            row,
        )

        val dt = (
            targetTimestampNanos - sampleTime
        ).coerceIn(-100_000_000L, 100_000_000L)

        if (dt == 0L) {
            return Corrected(
                point = observation.point,
                temporalOffsetNanos = 0L,
                confidenceScale =
                    temporalConfidenceScale(
                        observation,
                    ),
            )
        }

        val seconds = dt / 1e9
        val safeDepth =
            max(referenceDepthMeters, 0.15)

        val cameraVelocity =
            observation.pose.rotation
                .conjugate()
                .rotate(velocityMetersPerSecond)

        val cameraAngularVelocity =
            observation.pose.rotation
                .conjugate()
                .rotate(angularVelocityRadPerSecond)

        val normalizedX =
            (
                observation.point.x -
                    observation.intrinsics.cx
                ) / observation.intrinsics.fx
        val normalizedY =
            (
                observation.point.y -
                    observation.intrinsics.cy
                ) / observation.intrinsics.fy

        val vx = cameraVelocity.x
        val vy = cameraVelocity.y
        val vz = cameraVelocity.z
        val wx = cameraAngularVelocity.x
        val wy = cameraAngularVelocity.y
        val wz = cameraAngularVelocity.z

        val translationalFlowX =
            observation.intrinsics.fx *
                ((vx - normalizedX * vz) / safeDepth)

        val translationalFlowY =
            observation.intrinsics.fy *
                ((vy - normalizedY * vz) / safeDepth)

        val rotationalFlowX =
            observation.intrinsics.fx *
                (
                    normalizedX * normalizedY * wx -
                        (1.0 + normalizedX * normalizedX) * wy +
                        normalizedY * wz
                )

        val rotationalFlowY =
            observation.intrinsics.fy *
                (
                    (1.0 + normalizedY * normalizedY) * wx -
                        normalizedX * normalizedY * wy -
                        normalizedX * wz
                )

        val correctionX =
            (
                (translationalFlowX + rotationalFlowX) *
                    seconds *
                    correctionStrength
                ).coerceIn(
                    -maxCorrectionPixels,
                    maxCorrectionPixels,
                )

        val correctionY =
            (
                (translationalFlowY + rotationalFlowY) *
                    seconds *
                    correctionStrength
                ).coerceIn(
                    -maxCorrectionPixels,
                    maxCorrectionPixels,
                )

        val correctionMagnitude =
            abs(correctionX) + abs(correctionY)

        val depthReliability =
            (0.15 / safeDepth).coerceIn(0.0, 1.0)

        val angularMagnitude =
            angularVelocityRadPerSecond
                .norm()
                .coerceAtLeast(0.0)

        val timingConfidence =
            (
                1.0 -
                    (
                        abs(dt).toDouble() /
                            100_000_000.0
                        ).coerceIn(0.0, 1.0) *
                        0.25
            ).coerceIn(0.50, 1.0)

        val correctionConfidence = (
            1.0 -
                correctionMagnitude /
                    (maxCorrectionPixels * 2.0)
            ).coerceIn(0.10, 1.0) *
            timingConfidence *
            (
                0.85 +
                    0.15 * depthReliability
            ) *
            (
                1.0 -
                    (
                        angularMagnitude / 8.0
                    ).coerceIn(0.0, 0.25)
            )

        val confidence =
            (
                correctionConfidence *
                    temporalConfidenceScale(
                        observation,
                    )
                ).coerceIn(0.05, 1.0)

        return Corrected(
            point = Vec2(
                observation.point.x - correctionX,
                observation.point.y - correctionY,
            ),
            temporalOffsetNanos = dt,
            confidenceScale = confidence,
        )
    }

    private fun temporalConfidenceScale(
        observation: CameraObservation,
    ): Double {
        val uncertaintyPenalty =
            (
                1.0 -
                    0.55 *
                        (
                            observation.temporalUncertaintyNanos
                                .toDouble() /
                                40_000_000.0
                            ).coerceIn(0.0, 1.0)
                ).coerceIn(0.35, 1.0)

        return (
            observation.temporalAlignmentConfidence *
                uncertaintyPenalty
            ).coerceIn(0.05, 1.0)
    }
}
