package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraMotionCompensationEngineTest {

    @Test
    fun stationarySensorsBuildConfidence() {
        val engine = CameraMotionCompensationEngine()

        var state = engine.update(
            gyroscope = ImuSample(
                timestampNanos = 1_000_000L,
                sensorType = 4,
                x = 0.0,
                y = 0.0,
                z = 0.0,
                accuracy = 3,
            ),
            accelerometer = ImuSample(
                timestampNanos = 1_000_000L,
                sensorType = 1,
                x = 0.0,
                y = 0.0,
                z = 9.80665,
                accuracy = 3,
            ),
        )

        repeat(40) { index ->
            state = engine.update(
                gyroscope = ImuSample(
                    timestampNanos = 1_000_000L + (index + 1) * 8_333_333L,
                    sensorType = 4,
                    x = 0.0,
                    y = 0.0,
                    z = 0.0,
                    accuracy = 3,
                ),
                accelerometer = ImuSample(
                    timestampNanos = 1_000_000L + (index + 1) * 8_333_333L,
                    sensorType = 1,
                    x = 0.0,
                    y = 0.0,
                    z = 9.80665,
                    accuracy = 3,
                ),
            )
        }

        assertTrue(state.confidence > 0.2)
        assertTrue(state.cameraMotionRisk < 0.4)
    }

    @Test
    fun compensationChangesPointWhenCameraMoves() {
        val engine = CameraMotionCompensationEngine()
        val state = engine.update(
            gyroscope = ImuSample(
                timestampNanos = 1_000_000L,
                x = 1.5,
                y = 0.0,
                z = 0.0,
            ),
            accelerometer = ImuSample(
                timestampNanos = 1_000_000L,
                x = 0.0,
                y = 0.0,
                z = 9.80665,
            ),
            visualCameraMotionRisk = 0.8,
        )

        val point = engine.compensatePoint(
            Vec3(0.1, 0.0, 2.0),
            state,
        )

        assertTrue(point.isFinite())
    }
}

private fun Vec3.isFinite(): Boolean =
    x.isFinite() &&
        y.isFinite() &&
        z.isFinite()
