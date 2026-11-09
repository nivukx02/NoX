package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class TemporalMotionFilter(
    private val baseAlpha: Double = 0.72,
    private val beta: Double = 0.18,
    private val maxGapSeconds: Double = 0.20,
) {
    data class State(
        val position: Vec3,
        val velocity: Vec3,
        val acceleration: Vec3,
        val timestampNanos: Long,
        val confidence: Double,
    )

    private data class MutableState(
        var position: Vec3,
        var velocity: Vec3,
        var acceleration: Vec3,
        var timestampNanos: Long,
        var confidence: Double,
    )

    private val states = HashMap<Int, MutableState>()

    @Synchronized
    fun update(id: Int, measurement: Vec3, confidence: Double, timestampNanos: Long): State {
        val previous = states[id]
        if (previous == null) {
            val initial = MutableState(
                position = measurement,
                velocity = Vec3(0.0, 0.0, 0.0),
                acceleration = Vec3(0.0, 0.0, 0.0),
                timestampNanos = timestampNanos,
                confidence = confidence,
            )
            states[id] = initial
            return snapshot(initial)
        }

        val dt = ((timestampNanos - previous.timestampNanos).coerceAtLeast(1L) / 1e9)
            .coerceAtMost(1.0)

        if (dt > maxGapSeconds) {
            previous.position = measurement
            previous.velocity = Vec3(0.0, 0.0, 0.0)
            previous.acceleration = Vec3(0.0, 0.0, 0.0)
            previous.timestampNanos = timestampNanos
            previous.confidence = confidence
            return snapshot(previous)
        }

        val predicted = previous.position +
            previous.velocity * dt +
            previous.acceleration * (0.5 * dt * dt)

        val innovation = measurement - predicted
        val motionBoost = 1.0 - exp(-innovation.norm() / 0.035)
        val alpha = min(0.96, max(0.18, baseAlpha + 0.20 * motionBoost))
        val confidenceGate = confidence.coerceIn(0.05, 1.0)
        val position = predicted + innovation * (alpha * confidenceGate)

        val velocityMeasurement = (position - previous.position) / dt
        val velocityGain = min(0.8, beta + 0.35 * confidenceGate)
        val newVelocity = previous.velocity.lerp(velocityMeasurement, velocityGain)
        val newAcceleration = (newVelocity - previous.velocity) / dt

        previous.position = position
        previous.velocity = newVelocity
        previous.acceleration = newAcceleration
        previous.timestampNanos = timestampNanos
        previous.confidence = min(
            1.0,
            max(previous.confidence * 0.15, confidenceGate),
        )

        return snapshot(previous)
    }

    @Synchronized
    fun remove(id: Int) { states.remove(id) }

    @Synchronized
    fun clear() { states.clear() }

    @Synchronized
    fun snapshot(): Map<Int, State> = states.mapValues { snapshot(it.value) }

    private fun snapshot(state: MutableState): State =
        State(
            state.position,
            state.velocity,
            state.acceleration,
            state.timestampNanos,
            state.confidence,
        )
}
