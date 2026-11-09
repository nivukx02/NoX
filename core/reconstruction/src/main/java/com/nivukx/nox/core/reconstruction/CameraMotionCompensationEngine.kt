package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max

/**
 * IMU-assisted common-mode camera motion estimator.
 *
 * This bounded compensation layer estimates short-horizon camera movement so subject motion is not
 * confused with camera movement. It is intentionally complementary to the visual common-mode
 * estimator and does not claim to be a full VIO backend.
 */
class CameraMotionCompensationEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val gyroBiasLearningRate: Double = 0.012,
        val accelerationBiasLearningRate: Double = 0.006,
        val gravityMagnitude: Double = 9.80665,
        val maximumAngularRate: Double = 14.0,
        val maximumLinearAcceleration: Double = 35.0,
        val orientationDecay: Double = 0.985,
        val velocityDecay: Double = 0.965,
        val confidenceRise: Double = 0.08,
        val confidenceFall: Double = 0.18,
    )

    data class State(
        val timestampNanos: Long,
        val angularVelocityRadiansPerSecond: Vec3,
        val angularVelocityBias: Vec3,
        val orientationDeltaRadians: Vec3,
        val linearAccelerationMetersPerSecond2: Vec3,
        val linearVelocityMetersPerSecond: Vec3,
        val translationDeltaMeters: Vec3,
        val cameraMotionRisk: Double,
        val confidence: Double,
    ) {
        val isMoving: Boolean
            get() = cameraMotionRisk > 0.35
    }

    private var lastTimestampNanos = Long.MIN_VALUE
    private var gyroBias = Vec3(0.0, 0.0, 0.0)
    private var accelBias = Vec3(0.0, 0.0, 0.0)
    private var orientationDelta = Vec3(0.0, 0.0, 0.0)
    private var velocity = Vec3(0.0, 0.0, 0.0)
    private var translation = Vec3(0.0, 0.0, 0.0)

    private var lastState = State(
        timestampNanos = 0L,
        angularVelocityRadiansPerSecond = Vec3(0.0, 0.0, 0.0),
        angularVelocityBias = gyroBias,
        orientationDeltaRadians = orientationDelta,
        linearAccelerationMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
        linearVelocityMetersPerSecond = velocity,
        translationDeltaMeters = translation,
        cameraMotionRisk = 0.0,
        confidence = 0.0,
    )

    @Synchronized
    fun update(
        gyroscope: ImuSample?,
        accelerometer: ImuSample?,
        visualCameraMotionRisk: Double = 0.0,
        timestampNanos: Long =
            max(
                gyroscope?.timestampNanos ?: Long.MIN_VALUE,
                accelerometer?.timestampNanos ?: Long.MIN_VALUE,
            ),
    ): State {
        if (timestampNanos == Long.MIN_VALUE || timestampNanos <= 0L) {
            return lastState
        }

        val dt =
            if (lastTimestampNanos == Long.MIN_VALUE) {
                1.0 / 120.0
            } else {
                (
                    timestampNanos - lastTimestampNanos
                ).coerceAtLeast(1L) / 1e9
            }.coerceAtMost(0.05)

        val rawGyro = gyroscope?.let(::vectorFromSample)
            ?: lastState.angularVelocityRadiansPerSecond
        val rawAccel = accelerometer?.let(::vectorFromSample)
            ?: Vec3(0.0, 0.0, config.gravityMagnitude)

        val correctedGyro = rawGyro - gyroBias
        val correctedAccel = rawAccel - accelBias
        val boundedGyro = clampNorm(
            correctedGyro,
            config.maximumAngularRate,
        )

        val gravityCompensated =
            correctedAccel - Vec3(
                0.0,
                0.0,
                config.gravityMagnitude,
            )

        val boundedAcceleration = clampNorm(
            gravityCompensated,
            config.maximumLinearAcceleration,
        )

        val gyroNorm = boundedGyro.norm()
        val accelerationNorm = boundedAcceleration.norm()

        val stableSensorState =
            gyroNorm < 0.08 &&
                accelerationNorm < 0.35

        if (stableSensorState) {
            gyroBias = gyroBias.lerp(
                rawGyro,
                config.gyroBiasLearningRate.coerceIn(0.0, 1.0),
            )
            accelBias = accelBias.lerp(
                gravityCompensated,
                config.accelerationBiasLearningRate.coerceIn(0.0, 1.0),
            )
        }

        orientationDelta = (
            orientationDelta +
                boundedGyro * dt
        ) * config.orientationDecay

        velocity = (
            velocity +
                boundedAcceleration * dt
        ) * config.velocityDecay

        translation = (
            translation +
                velocity * dt
        ) * 0.995

        val visualRisk =
            visualCameraMotionRisk.coerceIn(0.0, 1.0)
        val gyroRisk = (
            gyroNorm / config.maximumAngularRate
        ).coerceIn(0.0, 1.0)
        val accelRisk = (
            accelerationNorm /
                config.maximumLinearAcceleration
        ).coerceIn(0.0, 1.0)

        val risk = (
            0.46 * visualRisk +
                0.34 * gyroRisk +
                0.20 * accelRisk
        ).coerceIn(0.0, 1.0)

        val targetConfidence = (
            1.0 -
                abs(
                    visualRisk -
                        (gyroRisk * 0.62 + accelRisk * 0.38),
                )
        ).coerceIn(0.0, 1.0)

        val nextConfidence =
            if (risk > 0.75) {
                lastState.confidence -
                    config.confidenceFall
            } else {
                lastState.confidence +
                    config.confidenceRise *
                    targetConfidence
            }.coerceIn(0.0, 1.0)

        lastTimestampNanos = timestampNanos

        lastState = State(
            timestampNanos = timestampNanos,
            angularVelocityRadiansPerSecond = boundedGyro,
            angularVelocityBias = gyroBias,
            orientationDeltaRadians = orientationDelta,
            linearAccelerationMetersPerSecond2 = boundedAcceleration,
            linearVelocityMetersPerSecond = velocity,
            translationDeltaMeters = translation,
            cameraMotionRisk = risk,
            confidence = nextConfidence,
        )

        return lastState
    }

    @Synchronized
    fun compensatePoint(
        pointCameraMeters: Vec3,
        state: State = lastState,
    ): Vec3 {
        val corrected =
            Vec3(
                pointCameraMeters.x -
                    state.orientationDeltaRadians.y *
                    pointCameraMeters.z,
                pointCameraMeters.y +
                    state.orientationDeltaRadians.x *
                    pointCameraMeters.z,
                pointCameraMeters.z,
            )

        return corrected -
            state.translationDeltaMeters *
            state.confidence
    }

    @Synchronized
    fun fuseVisualRisk(
        semanticCameraMotionRisk: Double,
        imuState: State = lastState,
    ): Double =
        (
            semanticCameraMotionRisk.coerceIn(0.0, 1.0) * 0.58 +
                imuState.cameraMotionRisk * 0.42
        ).coerceIn(0.0, 1.0)

    @Synchronized
    fun state(): State = lastState

    @Synchronized
    fun reset() {
        lastTimestampNanos = Long.MIN_VALUE
        gyroBias = Vec3(0.0, 0.0, 0.0)
        accelBias = Vec3(0.0, 0.0, 0.0)
        orientationDelta = Vec3(0.0, 0.0, 0.0)
        velocity = Vec3(0.0, 0.0, 0.0)
        translation = Vec3(0.0, 0.0, 0.0)
        lastState = State(
            timestampNanos = 0L,
            angularVelocityRadiansPerSecond = Vec3(0.0, 0.0, 0.0),
            angularVelocityBias = gyroBias,
            orientationDeltaRadians = orientationDelta,
            linearAccelerationMetersPerSecond2 = Vec3(0.0, 0.0, 0.0),
            linearVelocityMetersPerSecond = velocity,
            translationDeltaMeters = translation,
            cameraMotionRisk = 0.0,
            confidence = 0.0,
        )
    }

    private fun vectorFromSample(
        sample: ImuSample,
    ): Vec3 =
        Vec3(sample.x, sample.y, sample.z)

    private fun clampNorm(
        value: Vec3,
        maximum: Double,
    ): Vec3 {
        val norm = value.norm()
        if (norm <= maximum || norm < 1e-9) {
            return value
        }

        return value * (maximum / norm)
    }
}
