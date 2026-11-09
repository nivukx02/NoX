package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Robust temporal articulated trajectory optimizer.
 *
 * The optimizer operates after geometric reconstruction and before retargeting. It does not replace
 * triangulation or the learned bone constraints; it fuses their uncertainty with a short temporal
 * prior so fast motion remains responsive while isolated 3D spikes are rejected.
 *
 * The state is intentionally bounded to a small rolling window. This keeps memory deterministic on
 * phones and makes latency predictable.
 */
class KinematicTrajectoryOptimizer(
    private val historyLength: Int = 8,
    private val predictionGateMeters: Double = 0.28,
    private val velocityLearningRate: Double = 0.32,
    private val accelerationLearningRate: Double = 0.20,
) {

    data class Tuning(
        val predictionGateMeters: Double = 0.28,
        val microMotionPreservation: Double = 0.88,
        val temporalSmoothing: Double = 0.64,
        val semanticMotionSensitivity: Double = 0.82,
        val maximumVelocityMetersPerSecond: Double = 18.0,
        val maximumAccelerationMetersPerSecond2: Double = 92.0,
        val maximumJerkMetersPerSecond3: Double = 540.0,
        val innovationHardRatio: Double = 2.6,
        val confidenceRetention: Double = 0.72,
    ) {
        init {
            require(predictionGateMeters > 0.0)
            require(microMotionPreservation in 0.0..1.0)
            require(temporalSmoothing in 0.0..1.0)
            require(semanticMotionSensitivity in 0.0..1.0)
            require(maximumVelocityMetersPerSecond > 0.0)
            require(maximumAccelerationMetersPerSecond2 > 0.0)
            require(maximumJerkMetersPerSecond3 > 0.0)
            require(innovationHardRatio > 1.0)
            require(confidenceRetention in 0.0..1.0)
        }
    }

    @Volatile
    private var tuning = Tuning()

    @Synchronized
    fun configure(
        value: Tuning,
    ) {
        tuning = value
    }

    fun currentTuning(): Tuning = tuning

    private data class State(
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var timestampNanos: Long,
        var confidence: Double,
        val history: ArrayDeque<Vec3> = ArrayDeque(),
    )

    private val stateByJoint = HashMap<Int, State>()

    fun optimize(frame: SkeletonFrame): SkeletonFrame {
        if (frame.joints.isEmpty()) return frame

        val optimized = frame.joints.mapNotNull { joint ->
            val optimizedJoint = optimizeJoint(
                joint = joint,
                timestampNanos = frame.timestampNanos,
            )
            optimizedJoint
        }

        val confidence = if (optimized.isEmpty()) {
            0.0
        } else {
            optimized.sumOf { it.confidence } /
                optimized.size.toDouble()
        }

        return frame.copy(
            joints = optimized,
            globalConfidence = (
                frame.globalConfidence * 0.45 +
                    confidence * 0.55
                ).coerceIn(0.0, 1.0),
        )
    }

    fun reset() {
        stateByJoint.clear()
    }

    private fun optimizeJoint(
        joint: ReconstructionJoint,
        timestampNanos: Long,
    ): ReconstructionJoint {
        val state = stateByJoint[joint.landmarkId]

        if (state == null) {
            val created = State(
                position = joint.position,
                velocity = joint.velocity,
                acceleration = Vec3(0.0, 0.0, 0.0),
                timestampNanos = timestampNanos,
                confidence = joint.confidence,
            )
            created.history.addLast(joint.position)
            stateByJoint[joint.landmarkId] = created
            return joint
        }

        val dt = (
            (timestampNanos - state.timestampNanos)
                .coerceAtLeast(1L) / 1e9
            ).coerceIn(1.0 / 240.0, 0.20)

        val active = tuning
        val predicted = state.position +
            state.velocity * dt +
            state.acceleration * (
                0.5 * dt * dt
                )

        val innovation = joint.position.distanceTo(predicted)
        val uncertaintyScale = joint.positionUncertaintyMeters
            .coerceIn(0.001, 1.5)

        val dynamicGate = (
            active.predictionGateMeters.coerceAtLeast(0.02) +
                state.velocity.norm() * dt * (
                    0.50 +
                        active.semanticMotionSensitivity * 0.55
                    ) +
                uncertaintyScale * (
                    1.20 +
                        (1.0 - active.temporalSmoothing) * 0.90
                    )
            ).coerceIn(0.025, 1.25)

        val microMotionGate =
            active.predictionGateMeters *
                (
                    0.35 +
                        active.microMotionPreservation * 0.65
                    ).coerceIn(0.35, 1.0)

        val gate = max(
            dynamicGate,
            microMotionGate * (
                0.35 +
                    0.65 * joint.confidence
                ),
        ).coerceIn(0.025, 1.25)

        val hardGate =
            (
                gate *
                    active.innovationHardRatio
                ).coerceAtMost(2.5)

        val robustResidualWeight = when {
            innovation <= gate * 0.35 -> 1.0
            innovation <= gate -> (
                1.0 -
                    (
                        innovation -
                            gate * 0.35
                        ) /
                        (gate * 0.65)
                ).coerceIn(0.0, 1.0)
            innovation <= hardGate -> (
                gate /
                    innovation.coerceAtLeast(1e-6)
                ).coerceIn(0.08, 0.58)
            else -> 0.04
        }

        val uncertaintyWeight = exp(
            -uncertaintyScale / 0.16,
        ).coerceIn(0.05, 1.0)

        val responsiveness = (
            0.32 +
                0.48 * active.microMotionPreservation +
                0.20 * active.semanticMotionSensitivity
            ).coerceIn(0.32, 1.0)

        val measurementWeight = (
            0.08 +
                0.48 * joint.confidence +
                0.20 * uncertaintyWeight +
                0.24 * responsiveness
            ).coerceIn(0.08, 0.94)

        val smoothingPenalty =
            active.temporalSmoothing * 0.28

        val correctionWeight = (
            measurementWeight *
                robustResidualWeight *
                (1.0 - smoothingPenalty)
                + active.microMotionPreservation * 0.10
            ).coerceIn(0.04, 0.94)

        val corrected = predicted.lerp(
            joint.position,
            correctionWeight,
        )

        val previousVelocity = state.velocity
        val previousAcceleration = state.acceleration

        val measuredVelocity =
            (corrected - state.position) / dt

        val velocityLearning =
            (
                velocityLearningRate *
                    (
                        0.48 +
                            0.30 * joint.confidence +
                            0.22 * active.microMotionPreservation
                        ) *
                    (
                        1.0 -
                            active.temporalSmoothing * 0.24
                        )
                ).coerceIn(0.10, 0.68)

        val boundedVelocity =
            measuredVelocity.clampNorm(
                active.maximumVelocityMetersPerSecond,
            )

        state.velocity = state.velocity.lerp(
            boundedVelocity,
            velocityLearning,
        )

        val measuredAcceleration =
            (state.velocity - previousVelocity) / dt

        val boundedAcceleration =
            measuredAcceleration.clampNorm(
                active.maximumAccelerationMetersPerSecond2,
            )

        state.acceleration = state.acceleration.lerp(
            boundedAcceleration,
            (
                accelerationLearningRate *
                    (
                        0.45 +
                            0.38 * joint.confidence +
                            0.17 * active.microMotionPreservation
                        )
                ).coerceIn(0.06, 0.46),
        )

        val accelerationBeforeClamp = state.acceleration
        val jerkProxy =
            (
                accelerationBeforeClamp -
                    (
                        previousAcceleration
                    )
            ).norm() / dt.coerceAtLeast(1e-4)

        if (
            jerkProxy.isFinite() &&
            jerkProxy > active.maximumJerkMetersPerSecond3
        ) {
            val jerkScale =
                (
                    active.maximumJerkMetersPerSecond3 /
                        jerkProxy
                ).coerceIn(0.10, 1.0)
            state.acceleration *= jerkScale
        }

        state.position = corrected
        state.timestampNanos = timestampNanos
        state.confidence = (
            state.confidence * 0.28 +
                joint.confidence * 0.72
            ).coerceIn(0.0, 1.0)

        state.history.addLast(corrected)
        while (state.history.size > historyLength) {
            state.history.removeFirst()
        }

        val localJitter =
            estimateJitter(state.history)

        val dynamicPenalty = (
            localJitter /
                0.025
            ).coerceIn(0.0, 1.0)

        val innovationPenalty =
            if (innovation <= gate) {
                1.0
            } else {
                (
                    1.0 -
                        (
                            innovation -
                                gate
                            ) /
                            max(
                                hardGate - gate,
                                1e-6,
                            )
                    ).coerceIn(0.16, 1.0)
            }

        val finalConfidence = (
            state.confidence *
                (
                    active.confidenceRetention +
                        (1.0 - active.confidenceRetention) *
                            robustResidualWeight
                    ) *
                (1.0 - 0.22 * dynamicPenalty) *
                innovationPenalty
            ).coerceIn(0.0, 1.0)

        val uncertainty = (
            joint.positionUncertaintyMeters +
                innovation * 0.18 *
                    (1.0 - correctionWeight)
            ).coerceAtMost(2.0)

        return joint.copy(
            position = corrected,
            velocity = state.velocity,
            confidence = finalConfidence,
            positionUncertaintyMeters = uncertainty,
        )
    }

    private fun Vec3.clampNorm(
        maximum: Double,
    ): Vec3 {
        val norm = norm()
        if (!norm.isFinite() || norm <= maximum) return this
        if (maximum <= 0.0) {
            return Vec3(0.0, 0.0, 0.0)
        }
        return this * (maximum / norm)
    }

    private fun estimateJitter(
        history: Collection<Vec3>,
    ): Double {
        if (history.size < 3) return 0.0
        val points = history.toList()
        val deltas = points.zipWithNext()
            .map { (a, b) -> b - a }
        if (deltas.size < 2) return 0.0

        val accelerations = deltas.zipWithNext()
            .map { (a, b) -> b - a }

        return accelerations.map { it.norm() }.average()
    }
}
