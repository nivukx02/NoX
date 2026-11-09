package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.RollingShutterModel
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3

class RollingShutterCompensatorTest {

    private val observation =
        CameraObservation(
            deviceId = com.nivukx.nox.core.model.DeviceId("cam-a"),
            point = Vec2(420.0, 720.0),
            confidence = 0.95,
            intrinsics =
                CameraIntrinsics(
                    fx = 900.0,
                    fy = 900.0,
                    cx = 500.0,
                    cy = 500.0,
                ),
            pose =
                Pose(
                    position = Vec3(0.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                ),
            temporalAlignmentConfidence = 0.92,
            temporalUncertaintyNanos = 4_000_000L,
        )

    private val model =
        RollingShutterModel(
            readoutTimeNanos = 16_000_000L,
            exposureTimeNanos = 2_000_000L,
        )

    fun row_timing_changes_temporal_offset() {
        val result =
            RollingShutterCompensator().correct(
                observation = observation,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.25, 0.0, 0.0),
                targetTimestampNanos = 1_020_000_000L,
                frameTimestampNanos = 1_000_000_000L,
            )

        check(result.temporalOffsetNanos != 0L)
        check(result.confidenceScale in 0.0..1.0)
    }

    fun strong_temporal_uncertainty_reduces_confidence() {
        val lowConfidence =
            observation.copy(
                temporalAlignmentConfidence = 0.25,
                temporalUncertaintyNanos = 36_000_000L,
            )

        val highConfidence =
            observation.copy(
                temporalAlignmentConfidence = 0.98,
                temporalUncertaintyNanos = 1_000_000L,
            )

        val engine = RollingShutterCompensator()

        val weak =
            engine.correct(
                observation = lowConfidence,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.25, 0.0, 0.0),
                targetTimestampNanos = 1_020_000_000L,
                frameTimestampNanos = 1_000_000_000L,
            )

        val strong =
            engine.correct(
                observation = highConfidence,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.25, 0.0, 0.0),
                targetTimestampNanos = 1_020_000_000L,
                frameTimestampNanos = 1_000_000_000L,
            )

        check(weak.confidenceScale < strong.confidenceScale)
    }

    fun correction_strength_scales_row_time_extrapolation() {
        val strong =
            RollingShutterCompensator(
                correctionStrength = 1.0,
            ).correct(
                observation = observation,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.25, 0.0, 0.0),
                targetTimestampNanos = 1_020_000_000L,
                frameTimestampNanos = 1_000_000_000L,
            )

        val restrained =
            RollingShutterCompensator(
                correctionStrength = 0.50,
            ).correct(
                observation = observation,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.25, 0.0, 0.0),
                targetTimestampNanos = 1_020_000_000L,
                frameTimestampNanos = 1_000_000_000L,
            )

        check(
            kotlin.math.abs(
                strong.point.x -
                    observation.point.x,
            ) >
                kotlin.math.abs(
                    restrained.point.x -
                        observation.point.x,
                ),
        )
        check(
            restrained.confidenceScale >
                0.0,
        )
    }

    fun angular_velocity_and_depth_are_reflected_in_confidence() {
        val engine = RollingShutterCompensator()

        val rotationOnly =
            engine.correct(
                observation = observation,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.0, 0.0, 0.0),
                targetTimestampNanos = 1_018_000_000L,
                frameTimestampNanos = 1_000_000_000L,
                referenceDepthMeters = 1.0,
                angularVelocityRadPerSecond =
                    Vec3(0.0, 2.0, 0.0),
            )

        val deepSubject =
            engine.correct(
                observation = observation,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.20, 0.0, 0.0),
                targetTimestampNanos = 1_018_000_000L,
                frameTimestampNanos = 1_000_000_000L,
                referenceDepthMeters = 6.0,
                angularVelocityRadPerSecond =
                    Vec3(0.0, 0.1, 0.0),
            )

        check(rotationOnly.point != observation.point)
        check(rotationOnly.confidenceScale < 1.0)
        check(deepSubject.confidenceScale > 0.0)
    }

    fun zero_temporal_offset_preserves_original_point() {
        val result =
            RollingShutterCompensator().correct(
                observation = observation,
                frameHeight = 1000,
                model = model,
                velocityMetersPerSecond =
                    Vec3(0.0, 0.0, 0.0),
                targetTimestampNanos = 1_008_000_000L,
                frameTimestampNanos = 1_008_000_000L,
            )

        check(result.point == observation.point)
        check(result.temporalOffsetNanos == 0L)
    }
}

fun main() {
    val test = RollingShutterCompensatorTest()
    test.row_timing_changes_temporal_offset()
    test.strong_temporal_uncertainty_reduces_confidence()
    test.zero_temporal_offset_preserves_original_point()
}
