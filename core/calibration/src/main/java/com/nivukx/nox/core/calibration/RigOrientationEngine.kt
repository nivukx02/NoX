package com.nivukx.nox.core.calibration

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.nivukx.nox.core.model.Vec3
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.atan2

/**
 * Sensor-backed phone orientation estimator.
 *
 * TYPE_ROTATION_VECTOR is Android's fused orientation sensor. The resulting basis vectors are
 * expressed in the phone coordinate system and can be sampled at capture time to associate camera
 * poses with a stable rig frame.
 *
 * Magnetic heading is intentionally treated as auxiliary information. A NoX rig defines its own
 * north/reference direction from the master device; raw magnetic north is never allowed to move the
 * rig after calibration.
 */
class RigOrientationEngine(context: Context) : AutoCloseable {
    data class Orientation(
        val forward: Vec3,
        val right: Vec3,
        val up: Vec3,
        val headingDegrees: Double?,
        val timestampNanos: Long,
        val accuracy: Int,
    )

    private val manager =
        context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val rotationSensor =
        manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private val orientationRef = AtomicReference<Orientation?>()
    private var rotationAccuracy = SensorManager.SENSOR_STATUS_UNRELIABLE

    private val listener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            when (event.sensor.type) {
                Sensor.TYPE_ROTATION_VECTOR -> updateRotation(event)
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
            val current = orientationRef.get()
            if (current != null && sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
                orientationRef.set(current.copy(accuracy = accuracy))
            }
        }
    }

    init {
        rotationSensor?.let {
            manager.registerListener(
                listener,
                it,
                SensorManager.SENSOR_DELAY_GAME,
                20_000,
            )
        }

    }

    fun latest(): Orientation? = orientationRef.get()

    private fun updateRotation(event: SensorEvent) {
        val matrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(matrix, event.values)

        val right = Vec3(
            matrix[0].toDouble(),
            matrix[3].toDouble(),
            matrix[6].toDouble(),
        ).normalized()

        val up = Vec3(
            matrix[1].toDouble(),
            matrix[4].toDouble(),
            matrix[7].toDouble(),
        ).normalized()

        val forward = Vec3(
            matrix[2].toDouble(),
            matrix[5].toDouble(),
            matrix[8].toDouble(),
        ).normalized()

        rotationAccuracy = event.accuracy

        orientationRef.set(
            Orientation(
                forward = forward,
                right = right,
                up = up,
                headingDegrees = Math.toDegrees(
                    kotlin.math.atan2(forward.x, forward.y),
                ).let { if (it < 0.0) it + 360.0 else it },
                timestampNanos = event.timestamp,
                accuracy = rotationAccuracy,
            ),
        )
    }

    override fun close() {
        manager.unregisterListener(listener)
        orientationRef.set(null)
    }
}
