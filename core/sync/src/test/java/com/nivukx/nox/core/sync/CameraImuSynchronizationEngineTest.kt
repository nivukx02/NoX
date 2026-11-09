package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.VioPose

class CameraImuSynchronizationEngineTest {

    fun accepted_alignment_updates_rig_offset() {
        val engine =
            CameraImuSynchronizationEngine(
                ImuVisualMotionAlignmentEngine.Config(
                    minimumSamples = 12,
                    maximumOffsetNanos = 30_000_000L,
                    searchStepNanos = 1_000_000L,
                ),
            )

        val imu =
            (0 until 30).map { index ->
                ImuSample(
                    timestampNanos =
                        1_000_000_000L +
                            index * 10_000_000L,
                    sensorType = 4,
                    x = kotlin.math.sin(index * 0.17),
                    y = 0.0,
                    z = 0.35 + kotlin.math.cos(index * 0.11),
                    accuracy = 3,
                )
            }

        val visual =
            imu.map { sample ->
                VioPose(
                    timestampNanos =
                        sample.timestampNanos +
                            7_000_000L,
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                    linearVelocityMetersPerSecond =
                        Vec3(0.0, 0.0, 0.0),
                    angularVelocityRadiansPerSecond =
                        Vec3(
                            sample.x,
                            sample.y,
                            sample.z,
                        ),
                    covarianceTrace = 0.01,
                    confidence = 0.94,
                )
            }

        val result =
            engine.estimate(
                imu = imu,
                visual = visual,
            )

        check(result.accepted)
        check(result.confidence > 0.45)
        check(
            kotlin.math.abs(
                result.estimatedOffsetNanos,
            ) > 4_000_000.0,
        )
        check(
            kotlin.math.abs(
                result.estimatedOffsetNanos,
            ) < 10_000_000.0,
        )
    }

    fun poor_alignment_does_not_change_authoritative_offset() {
        val engine =
            CameraImuSynchronizationEngine()

        val flat =
            (0 until 24).map { index ->
                ImuSample(
                    timestampNanos =
                        1_000_000_000L +
                            index * 10_000_000L,
                    sensorType = 4,
                    x = 0.0,
                    y = 0.0,
                    z = 0.0,
                    accuracy = 3,
                )
            }

        val visual =
            flat.map {
                VioPose(
                    timestampNanos = it.timestampNanos,
                    positionMeters = Vec3(0.0, 0.0, 0.0),
                    rotation = Quat.IDENTITY,
                    linearVelocityMetersPerSecond =
                        Vec3(0.0, 0.0, 0.0),
                    angularVelocityRadiansPerSecond =
                        Vec3(0.0, 0.0, 0.0),
                    covarianceTrace = 0.01,
                    confidence = 0.94,
                )
            }

        val first =
            engine.estimate(
                imu = flat,
                visual = visual,
            )

        check(!first.accepted)
        check(first.estimatedOffsetNanos == 0.0)
        check(first.confidence == 0.0)
    }
}

fun main() {
    val test = CameraImuSynchronizationEngineTest()
    test.accepted_alignment_updates_rig_offset()
    test.poor_alignment_does_not_change_authoritative_offset()
}
