package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.VioPose
import com.nivukx.nox.core.sync.CameraImuSynchronizationEngine
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Error-corrected inertial fusion core.
 *
 * The engine maintains orientation, velocity, position and slowly adapting IMU biases. A
 * stationary detector enables zero-velocity updates and bias learning, while gravity provides
 * pitch/roll stabilization. A separate visual-pose correction API is provided for true VIO:
 * when a visual estimator supplies a measured pose, the inertial state is gently corrected rather
 * than allowing IMU integration to drift indefinitely.
 *
 * The class deliberately does not claim standalone VIO accuracy. Without a visual correction
 * source, yaw and position will still drift according to sensor bias/noise.
 */
class InertialFusionEngine(
    private val gravityCorrectionGain: Double = 0.035,
    private val zeroVelocityGain: Double = 0.18,
    private val gyroBiasLearningGain: Double = 0.0025,
    private val accelBiasLearningGain: Double = 0.004,
    private val stationaryAccelerationTolerance: Double = 0.18,
    private val stationaryGyroThreshold: Double = 0.035,
) {

    private var orientation = Quat.IDENTITY
    private var position = Vec3(0.0, 0.0, 0.0)
    private var velocity = Vec3(0.0, 0.0, 0.0)

    private var gyroBias = Vec3(0.0, 0.0, 0.0)
    private var accelBias = Vec3(0.0, 0.0, 0.0)
    private var lastRawGyro = Vec3(0.0, 0.0, 0.0)
    private var lastCorrectedAngularVelocity = Vec3(0.0, 0.0, 0.0)

    private var lastGyroTimestamp = Long.MIN_VALUE
    private var lastAccelTimestamp = Long.MIN_VALUE
    private var lastVisualTimestamp = Long.MIN_VALUE

    private var gyroEnergy = 0.0
    private var accelResidual = 0.0
    private var stationaryScore = 0.0
    private var correctionCount = 0L

    private val cameraImuSynchronization =
        CameraImuSynchronizationEngine()

    @Synchronized
    fun update(
        gyroscope: ImuSample?,
        accelerometer: ImuSample?,
    ): VioPose {
        if (gyroscope != null) {
            lastRawGyro = Vec3(gyroscope.x, gyroscope.y, gyroscope.z)
            integrateGyroscope(gyroscope)
        }

        if (accelerometer != null) {
            integrateAccelerometer(accelerometer)
        }

        val timestamp = maxOf(
            gyroscope?.timestampNanos ?: Long.MIN_VALUE,
            accelerometer?.timestampNanos ?: Long.MIN_VALUE,
        ).takeIf { it != Long.MIN_VALUE } ?: System.nanoTime()

        return snapshot(timestamp)
    }

    /**
     * Applies a visual pose correction from a calibrated camera/VIO estimator.
     *
     * Position and orientation are blended rather than replaced so an occasional poor visual
     * measurement cannot teleport the inertial state. Velocity is corrected from the measured
     * visual displacement whenever a valid visual interval exists.
     */
    @Synchronized
    fun correctWithVisualPose(
        visualPose: VioPose,
        positionGain: Double = 0.30,
        rotationGain: Double = 0.24,
    ): VioPose {
        val pg = positionGain.coerceIn(0.02, 0.90)
        val rg = rotationGain.coerceIn(0.02, 0.90)

        val previousPosition = position
        val previousTimestamp = lastVisualTimestamp

        position = position.lerp(
            visualPose.positionMeters,
            pg,
        )
        orientation = orientation.slerp(
            visualPose.rotation.normalized(),
            rg,
        )

        if (
            previousTimestamp != Long.MIN_VALUE &&
            visualPose.timestampNanos > previousTimestamp
        ) {
            val dt = (
                (visualPose.timestampNanos - previousTimestamp)
                    .coerceAtLeast(1L) / 1e9
            ).coerceIn(
                1.0 / 240.0,
                0.5,
            )
            val measuredVelocity =
                (visualPose.positionMeters - previousPosition) / dt

            velocity = velocity.lerp(
                measuredVelocity,
                (0.35 + 0.45 * visualPose.confidence)
                    .coerceIn(0.10, 0.85),
            )
        }

        lastVisualTimestamp = visualPose.timestampNanos
        correctionCount += 1L

        return snapshot(visualPose.timestampNanos)
    }

    /**
     * Estimates motion of the camera device itself, not the tracked subject.
     *
     * Angular velocity is the dominant signal for handheld camera movement. Linear velocity
     * contributes a lower-weight term because pure inertial position integration is inherently
     * drift-prone without visual correction. Stationary confidence suppresses false positives
     * during sensor noise around a locked device.
     */
    @Synchronized
    fun cameraMotionRisk(): Double {
        val angularRisk =
            (
                lastCorrectedAngularVelocity.norm() /
                    1.35
                ).coerceIn(0.0, 1.0)

        val linearRisk =
            (
                velocity.norm() /
                    1.75
                ).coerceIn(0.0, 1.0)

        val nonStationaryRisk =
            (
                1.0 -
                    stationaryScore.coerceIn(0.0, 1.0)
                )

        return (
            angularRisk * 0.62 +
                linearRisk * 0.23 +
                nonStationaryRisk * 0.15
            ).coerceIn(0.0, 1.0)
    }

    @Synchronized
    fun estimateCameraImuOffset(
        imu: List<ImuSample>,
        visual: List<VioPose>,
    ): CameraImuSynchronizationEngine.Result =
        cameraImuSynchronization.estimate(
            imu = imu,
            visual = visual,
        )

    @Synchronized
    fun cameraImuOffsetNanos(): Double =
        cameraImuSynchronization.currentOffsetNanos()

    @Synchronized
    fun reset() {
        orientation = Quat.IDENTITY
        position = Vec3(0.0, 0.0, 0.0)
        velocity = Vec3(0.0, 0.0, 0.0)
        gyroBias = Vec3(0.0, 0.0, 0.0)
        accelBias = Vec3(0.0, 0.0, 0.0)
        lastRawGyro = Vec3(0.0, 0.0, 0.0)
        lastCorrectedAngularVelocity =
            Vec3(0.0, 0.0, 0.0)
        lastGyroTimestamp = Long.MIN_VALUE
        lastAccelTimestamp = Long.MIN_VALUE
        lastVisualTimestamp = Long.MIN_VALUE
        gyroEnergy = 0.0
        accelResidual = 0.0
        stationaryScore = 0.0
        correctionCount = 0L
        cameraImuSynchronization.reset()
    }

    @Synchronized
    fun snapshot(timestampNanos: Long = System.nanoTime()): VioPose =
        VioPose(
            timestampNanos = timestampNanos,
            positionMeters = position,
            rotation = orientation,
            linearVelocityMetersPerSecond = velocity,
            angularVelocityRadiansPerSecond = lastCorrectedAngularVelocity,
            covarianceTrace = covarianceTrace(),
            confidence = confidence(),
        )

    @Synchronized
    fun diagnostics(): Diagnostics =
        Diagnostics(
            stationaryScore = stationaryScore,
            gyroBiasNorm = gyroBias.norm(),
            accelerometerBiasNorm = accelBias.norm(),
            angularVelocityNorm =
                lastCorrectedAngularVelocity.norm(),
            linearVelocityNorm = velocity.norm(),
            covarianceTrace = covarianceTrace(),
            visualCorrections = correctionCount,
            cameraMotionRisk = cameraMotionRisk(),
            cameraImuAlignmentOffsetNanos =
                cameraImuSynchronization
                    .currentOffsetNanos(),
            cameraImuAlignmentConfidence =
                cameraImuSynchronization
                    .currentConfidence(),
            confidence = confidence(),
        )

    data class Diagnostics(
        val stationaryScore: Double,
        val gyroBiasNorm: Double,
        val accelerometerBiasNorm: Double,
        val angularVelocityNorm: Double,
        val linearVelocityNorm: Double,
        val covarianceTrace: Double,
        val visualCorrections: Long,
        val cameraMotionRisk: Double,
        val cameraImuAlignmentOffsetNanos: Double,
        val cameraImuAlignmentConfidence: Double,
        val confidence: Double,
    )

    private fun integrateGyroscope(sample: ImuSample) {
        val corrected = Vec3(
            sample.x - gyroBias.x,
            sample.y - gyroBias.y,
            sample.z - gyroBias.z,
        )
        lastCorrectedAngularVelocity = corrected

        if (lastGyroTimestamp == Long.MIN_VALUE) {
            lastGyroTimestamp = sample.timestampNanos
            return
        }

        val dt = (
            (sample.timestampNanos - lastGyroTimestamp)
                .coerceAtLeast(1L) / 1e9
        ).coerceIn(1.0 / 1000.0, 0.1)

        val magnitude = corrected.norm()
        val angle = magnitude * dt

        if (angle > 1e-8) {
            val half = angle * 0.5
            val scale = sin(half) / magnitude
            val delta = Quat(
                w = cos(half),
                x = corrected.x * scale,
                y = corrected.y * scale,
                z = corrected.z * scale,
            )
            orientation = (
                orientation * delta
            ).normalized()
        }

        gyroEnergy =
            gyroEnergy * 0.96 +
                magnitude * magnitude * 0.04

        lastGyroTimestamp = sample.timestampNanos
    }

    private fun integrateAccelerometer(sample: ImuSample) {
        val correctedMeasurement = Vec3(
            sample.x - accelBias.x,
            sample.y - accelBias.y,
            sample.z - accelBias.z,
        )

        val magnitude = correctedMeasurement.norm()
        if (magnitude < 1e-4) return

        val dt = if (lastAccelTimestamp == Long.MIN_VALUE) {
            1.0 / 100.0
        } else {
            (
                (sample.timestampNanos - lastAccelTimestamp)
                    .coerceAtLeast(1L) / 1e9
            ).coerceIn(1.0 / 500.0, 0.1)
        }

        val correctedGyroMagnitude =
            sqrt(gyroEnergy.coerceAtLeast(0.0))

        val gravityError =
            abs(magnitude - 9.80665)

        val isStationary =
            gravityError <= stationaryAccelerationTolerance &&
                correctedGyroMagnitude <= stationaryGyroThreshold &&
                velocity.norm() <= 1.75

        stationaryScore =
            stationaryScore * 0.88 +
                (if (isStationary) 1.0 else 0.0) * 0.12

        correctOrientationFromGravity(correctedMeasurement)

        if (stationaryScore > 0.65) {
            updateBiases(
                rawMeasurement = Vec3(
                    sample.x,
                    sample.y,
                    sample.z,
                ),
                rawGyro = lastRawGyro,
            )

            velocity = velocity.lerp(
                Vec3(0.0, 0.0, 0.0),
                zeroVelocityGain * stationaryScore,
            )
        }

        val worldAcceleration =
            orientation.rotate(correctedMeasurement) + Vec3(
                0.0,
                0.0,
                -9.80665,
            )

        velocity = velocity +
            worldAcceleration * dt

        position = position +
            velocity * dt +
            worldAcceleration * (
                0.5 * dt * dt
            )

        accelResidual =
            accelResidual * 0.94 +
                gravityError * gravityError * 0.06

        lastAccelTimestamp = sample.timestampNanos
    }

    private fun correctOrientationFromGravity(
        measured: Vec3,
    ) {
        val normalized = measured / measured.norm()
        val expectedWorldGravity =
            Vec3(0.0, 0.0, -1.0)

        val expectedSensorGravity =
            orientation
                .conjugate()
                .rotate(expectedWorldGravity * -1.0)
                .normalized()

        val correctionAxis =
            normalized.cross(expectedSensorGravity)
        val correctionMagnitude =
            correctionAxis.norm()

        if (correctionMagnitude <= 1e-6) return

        val weight =
            gravityCorrectionGain.coerceIn(0.0, 0.25)
        val halfAngle =
            correctionMagnitude * weight

        val axis =
            correctionAxis.normalized()

        val delta = Quat(
            w = cos(halfAngle),
            x = axis.x * sin(halfAngle),
            y = axis.y * sin(halfAngle),
            z = axis.z * sin(halfAngle),
        )

        orientation = (
            orientation * delta
        ).normalized()
    }

    private fun updateBiases(
        rawMeasurement: Vec3,
        rawGyro: Vec3,
    ) {
        val expectedSensorGravity =
            orientation
                .conjugate()
                .rotate(
                    Vec3(0.0, 0.0, 9.80665),
                )

        val accelError =
            rawMeasurement - expectedSensorGravity

        accelBias = accelBias.lerp(
            accelBias + accelError,
            accelBiasLearningGain.coerceIn(0.0, 0.25),
        )

        if (
            rawGyro.x.isFinite() &&
            rawGyro.y.isFinite() &&
            rawGyro.z.isFinite()
        ) {
            gyroBias = gyroBias.lerp(
                rawGyro,
                gyroBiasLearningGain.coerceIn(0.0, 0.25),
            )
        }
    }

    private fun covarianceTrace(): Double =
        (
            0.002 +
                gyroEnergy * 0.002 +
                accelResidual * 0.0005 +
                gyroBias.norm() * 0.0005 +
                accelBias.norm() * 0.0001
            ).coerceIn(
                0.002,
                10.0,
            )

    private fun confidence(): Double {
        val inertial =
            1.0 /
                (1.0 + covarianceTrace())

        val stationaryContribution =
            0.15 + 0.85 * stationaryScore.coerceIn(
                0.0,
                1.0,
            )

        val biasPenalty =
            (
                1.0 -
                    0.10 * gyroBias.norm().coerceIn(
                        0.0,
                        1.0,
                    ) -
                    0.06 * accelBias.norm().coerceIn(
                        0.0,
                        5.0,
                    )
            ).coerceIn(0.20, 1.0)

        return (
            inertial *
                stationaryContribution *
                biasPenalty
            ).coerceIn(0.0, 1.0)
    }

    private fun Vec3.lerp(
        target: Vec3,
        alpha: Double,
    ): Vec3 =
        Vec3(
            x * (1.0 - alpha) + target.x * alpha,
            y * (1.0 - alpha) + target.y * alpha,
            z * (1.0 - alpha) + target.z * alpha,
        )

    private fun abs(value: Double): Double =
        kotlin.math.abs(value)
}
