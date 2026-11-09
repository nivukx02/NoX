package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Generic temporal motion model for semantic subjects.
 *
 * Unlike pose-only motion, this engine works for every tracked semantic target:
 * humans, animals and ordinary objects. It turns detector box motion into a physically meaningful
 * temporal signal with velocity, acceleration, jerk, area-rate and common-mode analysis.
 *
 * The model is intentionally conservative:
 * - detector confidence gates the update;
 * - missed tracks are predicted but their confidence decays;
 * - velocity and acceleration are bounded to prevent detector teleportation;
 * - a common-mode component is separated from relative motion so camera movement is not
 *   automatically reported as subject motion.
 *
 * This engine does not claim optical-flow accuracy. It is the semantic motion layer that sits above
 * object detection and below capture-policy decisions.
 */
class SemanticMotionEstimator(
    private val velocitySmoothing: Double = 0.38,
    private val accelerationSmoothing: Double = 0.26,
    private val jerkSmoothing: Double = 0.20,
    private val areaRateSmoothing: Double = 0.30,
    private val maxSpeedNormalizedPerSecond: Double = 8.0,
    private val maxAccelerationNormalizedPerSecond2: Double = 60.0,
    private val maxJerkNormalizedPerSecond3: Double = 600.0,
    private val staleAfterNanos: Long = 650_000_000L,
) {

    enum class MotionClass {
        STATIC,
        MICRO_MOTION,
        NORMAL,
        FAST,
        EXTREME,
        LOST,
    }

    data class SubjectMotion(
        val trackId: Long,
        val subjectType: SubjectType,
        val classType: MotionClass,
        val confidence: Double,
        val speedNormalizedPerSecond: Double,
        val accelerationNormalizedPerSecond2: Double,
        val jerkNormalizedPerSecond3: Double,
        val areaRatePerSecond: Double,
        val directionX: Double,
        val directionY: Double,
        val missedFrames: Int,
        val temporalStability: Double,
        val motionScore: Double,
        val predictionErrorNormalized: Double,
        val directionStability: Double,
        val detectionConsistency: Double,
        val predicted: Boolean,
        val timestampNanos: Long,
    )

    data class GlobalMotion(
        val timestampNanos: Long,
        val subjectCount: Int,
        val activeSubjects: Int,
        val movingSubjects: Int,
        val globalEnergy: Double,
        val relativeMotionEnergy: Double,
        val commonModeEnergy: Double,
        val commonModeVelocityX: Double,
        val commonModeVelocityY: Double,
        val cameraMotionRisk: Double,
        val meanSpeedNormalizedPerSecond: Double,
        val maxSpeedNormalizedPerSecond: Double,
        val meanAccelerationNormalizedPerSecond2: Double,
        val maxAccelerationNormalizedPerSecond2: Double,
        val meanJerkNormalizedPerSecond3: Double,
        val temporalStability: Double,
        val confidence: Double,
        val subjects: List<SubjectMotion>,
    )

    private data class State(
        val trackId: Long,
        var subjectType: SubjectType,
        var centerX: Double,
        var centerY: Double,
        var width: Double,
        var height: Double,
        var velocityX: Double,
        var velocityY: Double,
        var accelerationX: Double,
        var accelerationY: Double,
        var jerk: Double,
        var areaRate: Double,
        var predictionErrorNormalized: Double,
        var directionStability: Double,
        var detectionConsistency: Double,
        var confidence: Double,
        var lastObservedTimestampNanos: Long,
        var lastTimestampNanos: Long,
        var missedFrames: Int,
    )

    private val states = LinkedHashMap<Long, State>()

    @Synchronized
    fun update(
        tracks: List<SubjectTrackManager.Track>,
        timestampNanos: Long,
        frameWidth: Int = 1920,
        frameHeight: Int = 1080,
    ): GlobalMotion {
        require(timestampNanos >= 0L) {
            "timestampNanos must be non-negative"
        }

        val seen = HashSet<Long>()

        tracks.forEach { track ->
            seen += track.id

            val cx = (
                track.centerX.toDouble() /
                    frameWidth.coerceAtLeast(1).toDouble()
                ).coerceIn(-0.5, 1.5)
            val cy = (
                track.centerY.toDouble() /
                    frameHeight.coerceAtLeast(1).toDouble()
                ).coerceIn(-0.5, 1.5)
            val width = track.width.coerceAtLeast(1f) / 1000.0
            val height = track.height.coerceAtLeast(1f) / 1000.0

            val existing = states[track.id]
            if (existing == null) {
                states[track.id] = State(
                    trackId = track.id,
                    subjectType = track.subjectType,
                    centerX = cx,
                    centerY = cy,
                    width = width,
                    height = height,
                    velocityX = track.velocityX / 1000.0,
                    velocityY = track.velocityY / 1000.0,
                    accelerationX = 0.0,
                    accelerationY = 0.0,
                    jerk = 0.0,
                    areaRate = 0.0,
                    predictionErrorNormalized = 0.0,
                    directionStability = 0.65,
                    detectionConsistency = 0.72,
                    confidence = track.confidence,
                    lastObservedTimestampNanos = timestampNanos,
                    lastTimestampNanos = timestampNanos,
                    missedFrames = track.missedFrames,
                )
                return@forEach
            }

            val dt = (
                (timestampNanos - existing.lastTimestampNanos)
                    .coerceAtLeast(1L) / 1e9
                ).coerceIn(1.0 / 240.0, 0.5)

            val measuredVx = (
                (cx - existing.centerX) / dt
            ).coerceIn(
                -maxSpeedNormalizedPerSecond,
                maxSpeedNormalizedPerSecond,
            )

            val measuredVy = (
                (cy - existing.centerY) / dt
            ).coerceIn(
                -maxSpeedNormalizedPerSecond,
                maxSpeedNormalizedPerSecond,
            )

            val confidenceGate = motionConfidence(track)
            val innovation = hypot(
                measuredVx - existing.velocityX,
                measuredVy - existing.velocityY,
            )
            val innovationNormalized = (
                innovation /
                    (1.0 + existing.jerk * 0.018)
            ).coerceIn(0.0, maxSpeedNormalizedPerSecond * 2.0)

            val measuredSpeed = hypot(measuredVx, measuredVy)
            val previousSpeed = hypot(
                existing.velocityX,
                existing.velocityY,
            )
            val measuredDirectionX = if (measuredSpeed > 1e-6) {
                measuredVx / measuredSpeed
            } else {
                0.0
            }
            val measuredDirectionY = if (measuredSpeed > 1e-6) {
                measuredVy / measuredSpeed
            } else {
                0.0
            }
            val previousDirectionX = if (previousSpeed > 1e-6) {
                existing.velocityX / previousSpeed
            } else {
                0.0
            }
            val previousDirectionY = if (previousSpeed > 1e-6) {
                existing.velocityY / previousSpeed
            } else {
                0.0
            }
            val directionAgreement = if (
                measuredSpeed > 1e-6 &&
                    previousSpeed > 1e-6
            ) {
                (
                    (
                        measuredDirectionX * previousDirectionX +
                            measuredDirectionY * previousDirectionY +
                            1.0
                    ) * 0.5
                ).coerceIn(0.0, 1.0)
            } else {
                0.72
            }

            val innovationTrust = exp(
                -innovationNormalized * 0.42,
            ).coerceIn(0.22, 1.0)

            val nextVx = blend(
                existing.velocityX,
                measuredVx,
                velocitySmoothing *
                    confidenceGate *
                    (0.62 + 0.38 * innovationTrust),
            )
            val nextVy = blend(
                existing.velocityY,
                measuredVy,
                velocitySmoothing *
                    confidenceGate *
                    (0.62 + 0.38 * innovationTrust),
            )

            val measuredAx = (
                nextVx - existing.velocityX
            ) / dt

            val measuredAy = (
                nextVy - existing.velocityY
            ) / dt

            val nextAx = blend(
                existing.accelerationX,
                measuredAx,
                accelerationSmoothing * motionConfidence(track),
            ).coerceIn(
                -maxAccelerationNormalizedPerSecond2,
                maxAccelerationNormalizedPerSecond2,
            )

            val nextAy = blend(
                existing.accelerationY,
                measuredAy,
                accelerationSmoothing * motionConfidence(track),
            ).coerceIn(
                -maxAccelerationNormalizedPerSecond2,
                maxAccelerationNormalizedPerSecond2,
            )

            val measuredJerk = hypot(
                (nextAx - existing.accelerationX) / dt,
                (nextAy - existing.accelerationY) / dt,
            )

            val nextJerk = blend(
                existing.jerk,
                measuredJerk,
                jerkSmoothing,
            ).coerceIn(
                0.0,
                maxJerkNormalizedPerSecond3,
            )

            val oldArea = (
                existing.width * existing.height
            ).coerceAtLeast(1e-9)

            val newArea = (
                width * height
            ).coerceAtLeast(1e-9)

            val measuredAreaRate =
                (
                    (newArea - oldArea) /
                        dt /
                        oldArea
                    ).coerceIn(-20.0, 20.0)

            val nextAreaRate = blend(
                existing.areaRate,
                measuredAreaRate,
                areaRateSmoothing,
            )

            val nextDirectionStability = blend(
                existing.directionStability,
                directionAgreement,
                0.28 * confidenceGate,
            )

            val nextDetectionConsistency = blend(
                existing.detectionConsistency,
                innovationTrust,
                0.24 * confidenceGate,
            )

            val confidence = (
                0.30 * existing.confidence +
                    0.50 * track.confidence +
                    0.20 * motionConfidence(track)
                ).coerceIn(0.0, 1.0)

            existing.subjectType = track.subjectType
            existing.centerX = cx
            existing.centerY = cy
            existing.width = width
            existing.height = height
            existing.velocityX = nextVx
            existing.velocityY = nextVy
            existing.accelerationX = nextAx
            existing.accelerationY = nextAy
            existing.jerk = nextJerk
            existing.areaRate = nextAreaRate
            existing.predictionErrorNormalized = blend(
                existing.predictionErrorNormalized,
                innovationNormalized,
                0.30 * confidenceGate,
            )
            existing.directionStability = nextDirectionStability
            existing.detectionConsistency = nextDetectionConsistency
            existing.confidence = confidence
            existing.lastObservedTimestampNanos = timestampNanos
            existing.lastTimestampNanos = timestampNanos
            existing.missedFrames = track.missedFrames
        }

        states.values.forEach { state ->
            if (state.trackId !in seen) {
                state.missedFrames += 1
                val dt = (
                    (timestampNanos - state.lastTimestampNanos)
                        .coerceAtLeast(1L) / 1e9
                    ).coerceIn(1.0 / 240.0, 0.25)

                state.centerX = (
                    state.centerX + state.velocityX * dt
                ).coerceIn(-0.25, 1.25)

                state.centerY = (
                    state.centerY + state.velocityY * dt
                ).coerceIn(-0.25, 1.25)

                val decay = exp(-2.6 * dt)
                state.confidence *= decay
                state.predictionErrorNormalized = min(
                    max(
                        state.predictionErrorNormalized +
                            dt * 0.08,
                        0.0,
                    ),
                    maxSpeedNormalizedPerSecond * 2.0,
                )
                state.directionStability = blend(
                    state.directionStability,
                    0.45,
                    0.16,
                )
                state.detectionConsistency = blend(
                    state.detectionConsistency,
                    decay,
                    0.18,
                )
                state.lastTimestampNanos = timestampNanos
            }
        }

        states.entries.removeIf { (_, state) ->
            timestampNanos - state.lastObservedTimestampNanos >
                staleAfterNanos
        }

        val subjects = states.values.map { state ->
            snapshot(state, timestampNanos)
        }

        return global(
            timestampNanos = timestampNanos,
            subjects = subjects,
        )
    }

    @Synchronized
    fun snapshot(timestampNanos: Long): GlobalMotion =
        global(
            timestampNanos = timestampNanos,
            subjects = states.values.map {
                snapshot(it, timestampNanos)
            },
        )

    @Synchronized
    fun clear() {
        states.clear()
    }

    private fun snapshot(
        state: State,
        timestampNanos: Long,
    ): SubjectMotion {
        val speed = hypot(
            state.velocityX,
            state.velocityY,
        )
        val acceleration = hypot(
            state.accelerationX,
            state.accelerationY,
        )

        val stability = (
            state.confidence *
                exp(-0.045 * state.jerk) *
                exp(-0.45 * state.missedFrames)
            ).coerceIn(0.0, 1.0)

        val kinematicScore = (
            1.0 -
                exp(
                    -(
                        speed * 3.8 +
                            acceleration * 0.18 +
                            abs(state.areaRate) * 0.18
                        ),
                )
            ).coerceIn(0.0, 1.0)

        val score = (
            kinematicScore *
                (
                    0.60 +
                        0.18 * state.directionStability +
                        0.14 * state.detectionConsistency +
                        0.08 * exp(-state.predictionErrorNormalized * 0.32)
                )
        ).coerceIn(0.0, 1.0)

        return SubjectMotion(
            trackId = state.trackId,
            subjectType = state.subjectType,
            classType = classify(
                speed = speed,
                acceleration = acceleration,
                score = score,
                confidence = state.confidence,
            ),
            confidence = state.confidence,
            speedNormalizedPerSecond = speed,
            accelerationNormalizedPerSecond2 = acceleration,
            jerkNormalizedPerSecond3 = state.jerk,
            areaRatePerSecond = state.areaRate,
            directionX = if (speed > 1e-6) state.velocityX / speed else 0.0,
            directionY = if (speed > 1e-6) state.velocityY / speed else 0.0,
            missedFrames = state.missedFrames,
            temporalStability = stability,
            motionScore = score,
            predictionErrorNormalized =
                state.predictionErrorNormalized,
            directionStability =
                state.directionStability.coerceIn(0.0, 1.0),
            detectionConsistency =
                state.detectionConsistency.coerceIn(0.0, 1.0),
            predicted = state.missedFrames > 0,
            timestampNanos = timestampNanos,
        )
    }

    private fun global(
        timestampNanos: Long,
        subjects: List<SubjectMotion>,
    ): GlobalMotion {
        val active = subjects.filter {
            it.confidence >= 0.18 && it.missedFrames <= 4
        }

        val moving = active.filter {
            (
                it.classType == MotionClass.NORMAL ||
                    it.classType == MotionClass.FAST ||
                    it.classType == MotionClass.EXTREME
                ) &&
                it.motionScore >= 0.14
        }

        if (active.isEmpty()) {
            return GlobalMotion(
                timestampNanos = timestampNanos,
                subjectCount = subjects.size,
                activeSubjects = 0,
                movingSubjects = 0,
                globalEnergy = 0.0,
                relativeMotionEnergy = 0.0,
                commonModeEnergy = 0.0,
                commonModeVelocityX = 0.0,
                commonModeVelocityY = 0.0,
                cameraMotionRisk = 0.0,
                meanSpeedNormalizedPerSecond = 0.0,
                maxSpeedNormalizedPerSecond = 0.0,
                meanAccelerationNormalizedPerSecond2 = 0.0,
                maxAccelerationNormalizedPerSecond2 = 0.0,
                meanJerkNormalizedPerSecond3 = 0.0,
                temporalStability = 0.0,
                confidence = 0.0,
                subjects = subjects,
            )
        }

        val meanVx = active.map { subject ->
            subject.directionX *
                subject.speedNormalizedPerSecond
        }.average()

        val meanVy = active.map { subject ->
            subject.directionY *
                subject.speedNormalizedPerSecond
        }.average()

        val commonMode = active.map { subject ->
            hypot(
                subject.directionX *
                    subject.speedNormalizedPerSecond -
                    meanVx,
                subject.directionY *
                    subject.speedNormalizedPerSecond -
                    meanVy,
            )
        }.average()

        val totalSpeed = active.map {
            it.speedNormalizedPerSecond
        }

        val totalAcceleration = active.map {
            it.accelerationNormalizedPerSecond2
        }

        val totalJerk = active.map {
            it.jerkNormalizedPerSecond3
        }

        val energy = (
            0.56 * active.map { it.motionScore }.average() +
                0.20 * normalizedMean(totalSpeed, 3.0) +
                0.14 * normalizedMean(totalAcceleration, 18.0) +
                0.10 * normalizedMean(totalJerk, 120.0)
            ).coerceIn(0.0, 1.0)

        val relativeEnergy = (
            1.0 -
                exp(
                    -commonMode * 2.8
                )
            ).coerceIn(0.0, 1.0)

        val commonSpeed = hypot(
            meanVx,
            meanVy,
        )

        val cameraRisk = (
            normalizedMean(totalSpeed, 1.5) *
                (1.0 - relativeEnergy * 0.65) *
                (
                    active.size.coerceAtLeast(1) /
                        min(active.size, 4).toDouble()
                ).coerceIn(0.25, 1.0)
            ).coerceIn(0.0, 1.0)

        val stability = active.map {
            it.temporalStability
        }.average().coerceIn(0.0, 1.0)

        return GlobalMotion(
            timestampNanos = timestampNanos,
            subjectCount = subjects.size,
            activeSubjects = active.size,
            movingSubjects = moving.size,
            globalEnergy = energy,
            relativeMotionEnergy = relativeEnergy,
            commonModeEnergy = (
                1.0 -
                    exp(-commonSpeed * 2.2)
                ).coerceIn(0.0, 1.0),
            commonModeVelocityX = meanVx,
            commonModeVelocityY = meanVy,
            cameraMotionRisk = cameraRisk,
            meanSpeedNormalizedPerSecond = totalSpeed.average(),
            maxSpeedNormalizedPerSecond = totalSpeed.maxOrNull() ?: 0.0,
            meanAccelerationNormalizedPerSecond2 = totalAcceleration.average(),
            maxAccelerationNormalizedPerSecond2 =
                totalAcceleration.maxOrNull() ?: 0.0,
            meanJerkNormalizedPerSecond3 = totalJerk.average(),
            temporalStability = stability,
            confidence = (
                active.map { it.confidence }.average() *
                    (0.55 + 0.45 * stability)
                ).coerceIn(0.0, 1.0),
            subjects = subjects,
        )
    }

    private fun classify(
        speed: Double,
        acceleration: Double,
        score: Double,
        confidence: Double,
    ): MotionClass {
        if (confidence < 0.18) return MotionClass.LOST
        return when {
            score < 0.05 && speed < 0.025 -> MotionClass.STATIC
            score < 0.18 && speed < 0.10 -> MotionClass.MICRO_MOTION
            score < 0.55 && speed < 1.25 -> MotionClass.NORMAL
            score < 0.82 && acceleration < 8.0 -> MotionClass.FAST
            else -> MotionClass.EXTREME
        }
    }

    private fun normalizedMean(
        values: List<Double>,
        scale: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        return (
            values.average() / scale.coerceAtLeast(1e-6)
        ).coerceIn(0.0, 1.0)
    }

    private fun motionConfidence(
        track: SubjectTrackManager.Track,
    ): Double = (
        0.65 * track.confidence +
            0.25 * (1.0 - track.missedFrames / 8.0).coerceIn(0.0, 1.0) +
            0.10 * if (track.detectorTrackingId >= 0) 1.0 else 0.65
        ).coerceIn(0.0, 1.0)

    private fun blend(
        current: Double,
        measured: Double,
        alpha: Double,
    ): Double =
        current +
            (measured - current) *
            alpha.coerceIn(0.02, 0.95)
}
