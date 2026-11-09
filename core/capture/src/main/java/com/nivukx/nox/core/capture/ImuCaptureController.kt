package com.nivukx.nox.core.capture

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.nivukx.nox.core.model.ImuSample
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow

/**
 * High-rate inertial acquisition channel.
 *
 * Sensor timestamps use the same monotonic timebase used by Camera2 sensor timestamps on modern
 * Android devices, allowing offline or online visual-inertial alignment without wall-clock
 * conversion. The channel drops the oldest sample pair only under sustained consumer pressure,
 * preserving recent motion rather than stalling camera capture.
 */
class ImuCaptureController(context: Context) : AutoCloseable {
    data class SamplePair(
        val timestampNanos: Long,
        val gyroscope: ImuSample?,
        val accelerometer: ImuSample?,
    )

    private val manager = context.applicationContext
        .getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val gyro = manager.getDefaultSensor(
        Sensor.TYPE_GYROSCOPE_UNCALIBRATED,
    ) ?: manager.getDefaultSensor(
        Sensor.TYPE_GYROSCOPE,
    )

    private val accelerometer = manager.getDefaultSensor(
        Sensor.TYPE_ACCELEROMETER,
    )

    private val channel = Channel<SamplePair>(
        capacity = 256,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    val samples = channel.receiveAsFlow()

    @Volatile
    private var running = false

    private var latestGyro: ImuSample? = null
    private var latestAccelerometer: ImuSample? = null

    private val listener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val timestamp = event.timestamp

            when (event.sensor.type) {
                Sensor.TYPE_GYROSCOPE,
                Sensor.TYPE_GYROSCOPE_UNCALIBRATED -> {
                    latestGyro = ImuSample(
                        timestampNanos = timestamp,
                        sensorType = event.sensor.type,
                        x = event.values.getOrNull(0)?.toDouble() ?: 0.0,
                        y = event.values.getOrNull(1)?.toDouble() ?: 0.0,
                        z = event.values.getOrNull(2)?.toDouble() ?: 0.0,
                        accuracy = event.accuracy,
                    )
                }

                Sensor.TYPE_ACCELEROMETER -> {
                    latestAccelerometer = ImuSample(
                        timestampNanos = timestamp,
                        sensorType = event.sensor.type,
                        x = event.values.getOrNull(0)?.toDouble() ?: 0.0,
                        y = event.values.getOrNull(1)?.toDouble() ?: 0.0,
                        z = event.values.getOrNull(2)?.toDouble() ?: 0.0,
                        accuracy = event.accuracy,
                    )
                }

                else -> return
            }

            channel.trySend(
                SamplePair(
                    timestampNanos = timestamp,
                    gyroscope = latestGyro,
                    accelerometer = latestAccelerometer,
                ),
            )
        }

        override fun onAccuracyChanged(
            sensor: Sensor?,
            accuracy: Int,
        ) = Unit
    }

    fun start() {
        if (running) return
        running = true

        gyro?.let {
            manager.registerListener(
                listener,
                it,
                SensorManager.SENSOR_DELAY_FASTEST,
                5_000,
            )
        }

        accelerometer?.let {
            manager.registerListener(
                listener,
                it,
                SensorManager.SENSOR_DELAY_FASTEST,
                5_000,
            )
        }
    }

    fun stop() {
        if (!running) return
        running = false
        manager.unregisterListener(listener)
        latestGyro = null
        latestAccelerometer = null
    }

    fun supportsGyroscope(): Boolean = gyro != null

    fun supportsAccelerometer(): Boolean = accelerometer != null

    override fun close() {
        stop()
        channel.close()
    }
}
