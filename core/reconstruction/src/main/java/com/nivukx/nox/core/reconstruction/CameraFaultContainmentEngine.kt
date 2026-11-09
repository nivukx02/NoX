package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.max

/**
 * Stateful camera fault-containment authority.
 *
 * A distributed motion-capture rig must not oscillate between admission and rejection because of
 * one noisy frame. This authority adds bounded entry/exit hysteresis around per-camera evidence.
 * Quarantine is local to a camera, so healthy peers can continue reconstructing while the failed
 * camera accumulates recovery evidence.
 */
class CameraFaultContainmentEngine(
    private val config: Config = Config(),
) {
    enum class State {
        UNKNOWN,
        ACTIVE,
        WATCH,
        QUARANTINED,
    }

    data class Config(
        val quarantineEnterScore: Double = 0.42,
        val quarantineExitScore: Double = 0.76,
        val enterFailureFrames: Int = 3,
        val exitRecoveryFrames: Int = 8,
        val maximumStateAgeFrames: Int = 300,
    ) {
        init {
            require(quarantineEnterScore in 0.0..1.0)
            require(quarantineExitScore in quarantineEnterScore..1.0)
            require(enterFailureFrames >= 1)
            require(exitRecoveryFrames >= 1)
            require(maximumStateAgeFrames >= exitRecoveryFrames)
        }
    }

    data class Observation(
        val cameraId: DeviceId,
        val score: Double,
        val hardFailure: Boolean = false,
        val connected: Boolean = true,
    ) {
        init {
            require(score in 0.0..1.0)
        }
    }

    data class Result(
        val cameraId: DeviceId,
        val state: State,
        val admitted: Boolean,
        val score: Double,
        val failureStreak: Int,
        val recoveryStreak: Int,
        val ageFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class Memory(
        var state: State = State.UNKNOWN,
        var failureStreak: Int = 0,
        var recoveryStreak: Int = 0,
        var ageFrames: Int = 0,
    )

    private val memory = linkedMapOf<DeviceId, Memory>()

    @Synchronized
    fun observe(observation: Observation): Result {
        val state = memory.getOrPut(observation.cameraId) { Memory() }

        if (!observation.connected) {
            state.state = State.QUARANTINED
            state.failureStreak = max(
                state.failureStreak,
                config.enterFailureFrames,
            )
            state.recoveryStreak = 0
            state.ageFrames = 0
            return result(
                observation = observation,
                memory = state,
                warning = "CAMERA_DISCONNECTED_QUARANTINED",
            )
        }

        state.ageFrames =
            (state.ageFrames + 1)
                .coerceAtMost(config.maximumStateAgeFrames)

        val failing =
            observation.hardFailure ||
                observation.score < config.quarantineEnterScore

        val healthyForRecovery =
            !observation.hardFailure &&
                observation.score >= config.quarantineExitScore

        when {
            failing -> {
                state.failureStreak =
                    (state.failureStreak + 1)
                        .coerceAtMost(config.enterFailureFrames + 1)
                state.recoveryStreak = 0
            }
            healthyForRecovery -> {
                state.failureStreak = 0
                state.recoveryStreak =
                    (state.recoveryStreak + 1)
                        .coerceAtMost(config.exitRecoveryFrames + 1)
            }
            else -> {
                state.failureStreak = max(0, state.failureStreak - 1)
                state.recoveryStreak = 0
            }
        }

        state.state =
            when {
                state.state == State.QUARANTINED &&
                    state.recoveryStreak >= config.exitRecoveryFrames ->
                    State.ACTIVE
                state.state == State.QUARANTINED ->
                    State.QUARANTINED
                state.failureStreak >= config.enterFailureFrames ->
                    State.QUARANTINED
                observation.score < config.quarantineExitScore ->
                    State.WATCH
                else ->
                    State.ACTIVE
            }

        return result(
            observation = observation,
            memory = state,
        )
    }

    @Synchronized
    fun state(cameraId: DeviceId): State =
        memory[cameraId]?.state ?: State.UNKNOWN

    @Synchronized
    fun quarantinedCameraIds(): List<DeviceId> =
        memory
            .filterValues { it.state == State.QUARANTINED }
            .keys
            .sortedBy { it.value }

    @Synchronized
    fun clear(cameraId: DeviceId) {
        memory.remove(cameraId)
    }

    @Synchronized
    fun reset() {
        memory.clear()
    }

    private fun result(
        observation: Observation,
        memory: Memory,
        warning: String? = null,
    ): Result {
        val warnings = buildList {
            warning?.let(::add)
            when (memory.state) {
                State.UNKNOWN -> add("CAMERA_FAULT_STATE_INITIALIZING")
                State.ACTIVE -> Unit
                State.WATCH -> add("CAMERA_FAULT_WATCH")
                State.QUARANTINED -> {
                    add("CAMERA_QUARANTINED_FROM_PRIMARY_RECONSTRUCTION")
                    if (memory.recoveryStreak > 0) {
                        add("CAMERA_QUARANTINE_RECOVERY_IN_PROGRESS")
                    }
                }
            }
            if (memory.failureStreak > 0) {
                add("CAMERA_FAILURE_STREAK_" + memory.failureStreak)
            }
        }.distinct()

        val recommendations = buildList {
            when (memory.state) {
                State.UNKNOWN,
                State.ACTIVE -> Unit
                State.WATCH -> add(
                    "keep the camera active but reduce its reconstruction weight until evidence stabilizes",
                )
                State.QUARANTINED -> add(
                    "exclude the camera from primary metric reconstruction and recover it only after sustained healthy evidence",
                )
            }
            if (!observation.connected) {
                add("restore the camera connection before attempting recovery")
            }
        }.distinct()

        return Result(
            cameraId = observation.cameraId,
            state = memory.state,
            admitted = memory.state == State.ACTIVE ||
                memory.state == State.WATCH,
            score = observation.score,
            failureStreak = memory.failureStreak,
            recoveryStreak = memory.recoveryStreak,
            ageFrames = memory.ageFrames,
            warnings = warnings,
            recommendations = recommendations,
        )
    }
}