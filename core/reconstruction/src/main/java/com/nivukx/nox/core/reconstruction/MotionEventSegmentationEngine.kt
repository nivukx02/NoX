
package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Segments short-form motion events from already reconstructed motion evidence.
 *
 * Events are not used as a replacement for geometry. They are an additional semantic signal useful
 * for cleanup filters, operator telemetry, export metadata and adaptive inference budgets.
 */
class MotionEventSegmentationEngine(
    private val config: Config = Config(),
) {

    enum class EventType {
        IDLE,
        WALK,
        RUN,
        JUMP,
        CROUCH,
        TURN,
        FAST_ACTION,
        UNKNOWN,
    }

    enum class Phase {
        ENTRY,
        SUSTAIN,
        EXIT,
        TRANSITION,
    }

    data class Input(
        val frame: SkeletonFrame,
        val rootVelocityMetersPerSecond: Double,
        val rootAccelerationMetersPerSecond2: Double,
        val leftFootStable: Boolean,
        val rightFootStable: Boolean,
        val footSkateRisk: Double,
        val cameraMotionRisk: Double = 0.0,
        val confidence: Double,
    )

    data class Event(
        val timestampNanos: Long,
        val trackId: Long,
        val type: EventType,
        val phase: Phase,
        val confidence: Double,
        val intensity: Double,
        val cadenceHz: Double,
        val directionDegrees: Double,
        val transitionScore: Double,
        val evidenceQuality: Double,
        val uncertaintyScore: Double,
        val footSkateRisk: Double,
        val cameraMotionRisk: Double,
        val diagnostics: List<String>,
    )

    data class Config(
        val idleSpeed: Double = 0.06,
        val walkSpeed: Double = 1.20,
        val runSpeed: Double = 4.60,
        val jumpAcceleration: Double = 8.0,
        val crouchHeightRatio: Double = 0.65,
        val turnDegreesPerSecond: Double = 90.0,
        val fastActionAcceleration: Double = 14.0,
        val historyFrames: Int = 18,
        val confidenceFloor: Double = 0.30,
    )

    private data class Sample(
        val timestampNanos: Long,
        val speed: Double,
        val acceleration: Double,
        val direction: Double,
        val footSupport: Int,
    )

    private val history = ArrayDeque<Sample>()
    private var previousType = EventType.UNKNOWN
    private var previousTimestamp = Long.MIN_VALUE
    private var previousDirection = 0.0
    private var previousIntensity = 0.0
    private var transitionAgeFrames = 0

    @Synchronized
    fun classify(input: Input): Event {
        val frame = input.frame
        val direction = rootDirection(frame)
        val support = (
            if (input.leftFootStable) 1 else 0
        ) + if (input.rightFootStable) 1 else 0

        val sample = Sample(
            timestampNanos = frame.timestampNanos,
            speed = input.rootVelocityMetersPerSecond.coerceAtLeast(0.0),
            acceleration = input.rootAccelerationMetersPerSecond2.coerceAtLeast(0.0),
            direction = direction,
            footSupport = support,
        )

        history.addLast(sample)
        while (history.size > config.historyFrames) {
            history.removeFirst()
        }

        val cadence = estimateCadence()
        val directionRate = angularRate(direction)
        val verticalHeightRatio = bodyHeightRatio(frame)

        val uncertaintyScore =
            if (frame.joints.isEmpty()) {
                0.0
            } else {
                frame.joints.map { joint ->
                    (
                        1.0 -
                            joint.positionUncertaintyMeters
                                .coerceAtLeast(0.0) /
                                0.20
                        ).coerceIn(0.0, 1.0)
                }.average()
            }

        val trustedJointRatio =
            if (frame.joints.isEmpty()) {
                0.0
            } else {
                frame.joints.count {
                    it.confidence >= config.confidenceFloor &&
                        it.positionUncertaintyMeters <= 0.20
                }.toDouble() /
                    frame.joints.size.toDouble()
            }

        val evidenceQuality = (
            input.confidence.coerceIn(0.0, 1.0) * 0.42 +
                frame.globalConfidence.coerceIn(0.0, 1.0) * 0.18 +
                uncertaintyScore * 0.22 +
                trustedJointRatio * 0.18
            ).coerceIn(0.0, 1.0)

        val semanticConfidenceFloor =
            config.confidenceFloor +
                (
                    1.0 -
                        evidenceQuality
                    ).coerceIn(0.0, 1.0) * 0.14

        val semanticConfidence =
            input.confidence.coerceIn(0.0, 1.0)
        val cameraRisk =
            input.cameraMotionRisk.coerceIn(0.0, 1.0)

        val baseType = when {
            semanticConfidence < semanticConfidenceFloor ||
                evidenceQuality < 0.34 ->
                EventType.UNKNOWN

            cameraRisk >= 0.82 &&
                evidenceQuality < 0.78 ->
                EventType.UNKNOWN

            input.footSkateRisk >= 0.78 &&
                input.leftFootStable &&
                input.rightFootStable ->
                EventType.UNKNOWN

            input.rootAccelerationMetersPerSecond2 >= config.jumpAcceleration &&
                verticalHeightRatio < 0.78 ->
                EventType.JUMP

            verticalHeightRatio < config.crouchHeightRatio &&
                input.rootVelocityMetersPerSecond < config.walkSpeed ->
                EventType.CROUCH

            directionRate >= config.turnDegreesPerSecond ->
                EventType.TURN

            input.rootAccelerationMetersPerSecond2 >= config.fastActionAcceleration ->
                EventType.FAST_ACTION

            input.rootVelocityMetersPerSecond >= config.runSpeed ->
                EventType.RUN

            input.rootVelocityMetersPerSecond >= config.idleSpeed ->
                EventType.WALK

            else ->
                EventType.IDLE
        }

        val transition = if (
            previousType == EventType.UNKNOWN ||
            previousType == baseType
        ) {
            0.0
        } else {
            1.0
        }

        val currentIntensity = (
            input.rootVelocityMetersPerSecond /
                config.runSpeed * 0.60 +
                input.rootAccelerationMetersPerSecond2 /
                    config.fastActionAcceleration * 0.40
            ).coerceIn(0.0, 1.0)

        transitionAgeFrames =
            if (transition > 0.0) 0 else transitionAgeFrames + 1

        val phase = when {
            previousType != EventType.UNKNOWN &&
                previousType != EventType.IDLE &&
                previousIntensity > 0.20 &&
                currentIntensity < 0.16 ->
                Phase.EXIT

            previousType != baseType ->
                Phase.ENTRY

            transitionAgeFrames <= 2 ->
                Phase.TRANSITION

            history.size < config.historyFrames / 2 ->
                Phase.ENTRY

            else ->
                Phase.SUSTAIN
        }

        val stability = history
            .map { it.speed }
            .windowed(3, 1, partialWindows = false)
            .map { values ->
                val mean = values.average()
                values.map { abs(it - mean) }.average()
            }
            .let { deviations ->
                if (deviations.isEmpty()) 0.50
                else (
                    1.0 -
                        deviations.average() /
                            max(0.05, input.rootVelocityMetersPerSecond + 0.05)
                ).coerceIn(0.0, 1.0)
            }

        val semanticPenalty =
            (
                input.footSkateRisk * 0.18 +
                    cameraRisk * 0.20 +
                    (1.0 - evidenceQuality) * 0.22
                ).coerceIn(0.0, 0.54)

        val confidence = (
            input.confidence * 0.44 +
                stability * 0.18 +
                if (history.size >= 4) 0.14 else 0.04 +
                if (support > 0) 0.10 else 0.0 +
                evidenceQuality * 0.14 -
                semanticPenalty
            ).coerceIn(0.0, 1.0)

        val intensity = currentIntensity *
            (
                1.0 -
                    input.footSkateRisk.coerceIn(0.0, 1.0) * 0.18 -
                    cameraRisk * 0.16
                ).coerceIn(0.62, 1.0)

        val event = Event(
            timestampNanos = frame.timestampNanos,
            trackId = frame.trackId,
            type = baseType,
            phase = phase,
            confidence = confidence,
            intensity = intensity,
            cadenceHz = cadence,
            directionDegrees = direction,
            transitionScore = transition,
            evidenceQuality = evidenceQuality,
            uncertaintyScore = uncertaintyScore,
            footSkateRisk = input.footSkateRisk.coerceIn(0.0, 1.0),
            cameraMotionRisk = cameraRisk,
            diagnostics = diagnostics(
                frame = frame,
                type = baseType,
                support = support,
                cadence = cadence,
                directionRate = directionRate,
                heightRatio = verticalHeightRatio,
                confidence = confidence,
                evidenceQuality = evidenceQuality,
                uncertaintyScore = uncertaintyScore,
                footSkateRisk = input.footSkateRisk.coerceIn(0.0, 1.0),
                cameraMotionRisk = cameraRisk,
            ),
        )

        previousType = baseType
        previousTimestamp = frame.timestampNanos
        previousDirection = direction
        previousIntensity = intensity

        return event
    }

    @Synchronized
    fun reset() {
        history.clear()
        previousType = EventType.UNKNOWN
        previousTimestamp = Long.MIN_VALUE
        previousDirection = 0.0
        previousIntensity = 0.0
        transitionAgeFrames = 0
    }

    private fun rootDirection(
        frame: SkeletonFrame,
    ): Double {
        val root = frame.joints
            .firstOrNull { it.landmarkId == 23 }
            ?: frame.joints.firstOrNull()
            ?: return previousDirection

        val velocity = root.velocity
        if (magnitude(velocity) < 0.01) {
            return previousDirection
        }

        return Math.toDegrees(
            atan2(
                velocity.z,
                velocity.x,
            ),
        )
    }

    private fun angularRate(
        currentDirection: Double,
    ): Double {
        if (previousTimestamp == Long.MIN_VALUE || history.size < 2) {
            return 0.0
        }

        val dt = (
            history.last().timestampNanos -
                previousTimestamp
            ).coerceAtLeast(1L) / 1e9

        val delta = normalizeAngle(
            currentDirection - previousDirection,
        )

        return abs(delta) / dt.coerceAtLeast(1e-4)
    }

    private fun estimateCadence(): Double {
        if (history.size < 5) return 0.0

        val alternating = history
            .zipWithNext()
            .count { (a, b) ->
                a.footSupport != b.footSupport
            }

        val duration = (
            history.last().timestampNanos -
                history.first().timestampNanos
            ).coerceAtLeast(1L) / 1e9

        if (duration <= 0.0) return 0.0

        return (
            alternating.toDouble() /
                duration /
                2.0
            ).coerceIn(0.0, 8.0)
    }

    private fun bodyHeightRatio(
        frame: SkeletonFrame,
    ): Double {
        val shoulders = frame.joints.filter {
            it.landmarkId == 11 || it.landmarkId == 12
        }
        val hips = frame.joints.filter {
            it.landmarkId == 23 || it.landmarkId == 24
        }
        val ankles = frame.joints.filter {
            it.landmarkId == 27 || it.landmarkId == 28
        }

        if (
            shoulders.isEmpty() ||
            hips.isEmpty() ||
            ankles.isEmpty()
        ) {
            return 1.0
        }

        val highest = shoulders.minOf {
            it.position.y
        }

        val lowest = ankles.maxOf {
            it.position.y
        }

        val hip = hips.map { it.position.y }.average()

        val full = abs(lowest - highest)
            .coerceAtLeast(1e-4)
        val current = abs(hip - highest)

        return (
            current / full
        ).coerceIn(0.0, 1.5)
    }

    private fun normalizeAngle(
        angle: Double,
    ): Double {
        var result = angle
        while (result > 180.0) result -= 360.0
        while (result < -180.0) result += 360.0
        return result
    }

    private fun magnitude(
        x: com.nivukx.nox.core.model.Vec3,
    ): Double =
        sqrt(
            x.x * x.x +
                x.y * x.y +
                x.z * x.z,
        )

    private fun diagnostics(
        frame: SkeletonFrame,
        type: EventType,
        support: Int,
        cadence: Double,
        directionRate: Double,
        heightRatio: Double,
        confidence: Double,
        evidenceQuality: Double,
        uncertaintyScore: Double,
        footSkateRisk: Double,
        cameraMotionRisk: Double,
    ): List<String> = listOf(
        "subject=" + frame.subjectType.name,
        "event=" + type.name,
        "supportFeet=" + support,
        "cadenceHz=" + format(cadence),
        "directionRateDps=" + format(directionRate),
        "heightRatio=" + format(heightRatio),
        "confidence=" + format(confidence),
        "evidenceQuality=" + format(evidenceQuality),
        "uncertaintyScore=" + format(uncertaintyScore),
        "foot-skate-risk=" + format(footSkateRisk),
        "camera-motion-risk=" + format(cameraMotionRisk),
    )

    private fun format(
        value: Double,
    ): String =
        if (!value.isFinite()) "nan"
        else "%.4f".format(java.util.Locale.US, value)
}
