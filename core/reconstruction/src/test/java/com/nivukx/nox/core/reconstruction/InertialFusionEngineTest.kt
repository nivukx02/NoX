package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.VioPose
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class InertialFusionEngineTest {

    @Test
    fun snapshotExposesCorrectedAngularVelocity() {
        val engine = InertialFusionEngine()

        val pose = engine.update(
            gyroscope = ImuSample(
                timestampNanos = 1_000_000L,
                sensorType = 4,
                x = 0.12,
                y = -0.08,
                z = 1.10,
                accuracy = 3,
            ),
            accelerometer = null,
        )

        assertEquals(
            0.12,
            pose.angularVelocityRadiansPerSecond.x,
            1e-9,
        )
        assertEquals(
            -0.08,
            pose.angularVelocityRadiansPerSecond.y,
            1e-9,
        )
        assertEquals(
            1.10,
            pose.angularVelocityRadiansPerSecond.z,
            1e-9,
        )
    }

    @Test
    fun rapidAngularMotionProducesCameraMotionRisk() {
        val engine = InertialFusionEngine()

        engine.update(
            gyroscope = ImuSample(
                timestampNanos = 1_000_000L,
                sensorType = 4,
                x = 0.0,
                y = 0.0,
                z = 2.0,
                accuracy = 3,
            ),
            accelerometer = null,
        )

        assertTrue(engine.cameraMotionRisk() > 0.60)
    }

    @Test
    fun cameraImuAlignmentIsExposedAsFusionDiagnostics() {
        val engine = InertialFusionEngine()

        val imu =
            (0 until 28).map { index ->
                ImuSample(
                    timestampNanos =
                        1_000_000_000L +
                            index * 10_000_000L,
                    sensorType = 4,
                    x = kotlin.math.sin(index * 0.19),
                    y = 0.0,
                    z = 0.35 +
                        kotlin.math.cos(index * 0.13),
                    accuracy = 3,
                )
            }

        val visual =
            imu.map { sample ->
                VioPose(
                    timestampNanos =
                        sample.timestampNanos +
                            6_000_000L,
                    positionMeters =
                        Vec3(0.0, 0.0, 0.0),
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
                    confidence = 0.95,
                )
            }

        val result =
            engine.estimateCameraImuOffset(
                imu = imu,
                visual = visual,
            )

        assertTrue(result.accepted)
        assertTrue(result.confidence > 0.40)

        val diagnostics =
            engine.diagnostics()

        assertTrue(
            diagnostics.cameraImuAlignmentConfidence >
                0.30,
        )
    }

    @Test
    fun visualCorrectionMovesInertialStateWithoutTeleportingToFullMeasurement() {
        val engine = InertialFusionEngine()

        val visual = engine.correctWithVisualPose(
            visualPose = VioPose(
                timestampNanos = 20_000_000L,
                positionMeters = Vec3(4.0, 0.0, 0.0),
                rotation = Quat.IDENTITY,
                linearVelocityMetersPerSecond = Vec3(0.0, 0.0, 0.0),
                angularVelocityRadiansPerSecond = Vec3(0.0, 0.0, 0.0),
                covarianceTrace = 0.02,
                confidence = 0.95,
            ),
        )

        assertTrue(visual.positionMeters.x > 0.0)
        assertTrue(visual.positionMeters.x < 4.0)
        assertTrue(abs(visual.confidence) <= 1.0)
    }
}
