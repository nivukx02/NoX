package com.nivukx.nox.core.ai

import java.security.MessageDigest

/**
 * Deterministic long-session recovery coordinator.
 *
 * It records bounded checkpoints and converts RuntimeFailureContainmentEngine decisions into
 * ordered recovery plans. Actual hardware restart/reset operations remain owned by the caller.
 */
class FrontierSessionRecoveryCoordinator(
    private val maximumCheckpoints: Int = 64,
) : AutoCloseable {

    data class SessionCheckpoint(
        val checkpointId: Long,
        val timestampNanos: Long,
        val actorCount: Int,
        val cameraCount: Int,
        val authoritativeCameraCount: Int,
        val reconstructionFrame: Long,
        val stateDigest: String,
        val activeStages: Set<String>,
    ) {
        init {
            require(checkpointId > 0L)
            require(timestampNanos >= 0L)
            require(actorCount in 0..64)
            require(cameraCount in 0..128)
            require(authoritativeCameraCount in 0..32)
            require(authoritativeCameraCount <= cameraCount)
            require(reconstructionFrame >= 0L)
            require(stateDigest.length == 64)
        }
    }

    enum class RecoveryStep {
        STOP_OPTIONAL_WORK,
        FLUSH_TRANSIENT_BUFFERS,
        RELEASE_GPU_LEASES,
        SHED_TEMPORAL_STATE,
        RESET_RECONSTRUCTION,
        RESET_GPU,
        REINITIALIZE_CAMERA,
        RESTART_WORKER,
        RESTORE_LAST_CHECKPOINT,
        RESUME_CAPTURE,
        REQUIRE_OPERATOR,
    }

    data class RecoveryPlan(
        val checkpoint: SessionCheckpoint?,
        val steps: List<RecoveryStep>,
        val canAutoRecover: Boolean,
        val warnings: List<String>,
    )

    private val checkpoints = ArrayDeque<SessionCheckpoint>()
    private val containment = RuntimeFailureContainmentEngine()
    private var nextCheckpointId = 1L
    private var closed = false

    init {
        require(maximumCheckpoints in 1..1024)
    }

    @Synchronized
    fun checkpoint(
        timestampNanos: Long,
        actorCount: Int,
        cameraCount: Int,
        authoritativeCameraCount: Int,
        reconstructionFrame: Long,
        activeStages: Set<String>,
        stateMaterial: ByteArray,
    ): SessionCheckpoint {
        ensureOpen()
        require(timestampNanos >= 0L)
        require(actorCount in 0..64)
        require(cameraCount in 0..128)
        require(authoritativeCameraCount in 0..32)
        require(authoritativeCameraCount <= cameraCount)
        require(reconstructionFrame >= 0L)
        require(activeStages.size <= 64)
        check(nextCheckpointId > 0L)
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(stateMaterial.copyOf())
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        val checkpoint = SessionCheckpoint(
            checkpointId = nextCheckpointId++,
            timestampNanos = timestampNanos,
            actorCount = actorCount,
            cameraCount = cameraCount,
            authoritativeCameraCount = authoritativeCameraCount,
            reconstructionFrame = reconstructionFrame,
            stateDigest = digest,
            activeStages = activeStages.toSet(),
        )
        checkpoints.addLast(checkpoint)
        while (checkpoints.size > maximumCheckpoints) checkpoints.removeFirst()
        return checkpoint
    }

    @Synchronized
    fun latestCheckpoint(): SessionCheckpoint? = checkpoints.lastOrNull()

    @Synchronized
    fun checkpointCount(): Int = checkpoints.size

    @Synchronized
    fun checkpointMatches(
        checkpoint: SessionCheckpoint,
        stateMaterial: ByteArray,
    ): Boolean {
        ensureOpen()
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(stateMaterial.copyOf())
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        return checkpoints.any { it.checkpointId == checkpoint.checkpointId } &&
            digest == checkpoint.stateDigest
    }


    @Synchronized
    fun planRecovery(
        timestampNanos: Long,
        faultKind: RuntimeFailureContainmentEngine.FaultKind,
        severity: RuntimeFailureContainmentEngine.Severity,
        signature: String = faultKind.name,
    ): RecoveryPlan {
        ensureOpen()
        val decision = containment.observeFault(timestampNanos, faultKind, severity, signature)
        val checkpoint = latestCheckpoint()
        val steps = buildList {
            add(RecoveryStep.STOP_OPTIONAL_WORK)
            when (faultKind) {
                RuntimeFailureContainmentEngine.FaultKind.OUT_OF_MEMORY -> {
                    add(RecoveryStep.RELEASE_GPU_LEASES)
                    add(RecoveryStep.SHED_TEMPORAL_STATE)
                    add(RecoveryStep.FLUSH_TRANSIENT_BUFFERS)
                }
                RuntimeFailureContainmentEngine.FaultKind.GPU_DEVICE_LOST -> {
                    add(RecoveryStep.RELEASE_GPU_LEASES)
                    add(RecoveryStep.RESET_GPU)
                    add(RecoveryStep.RESTORE_LAST_CHECKPOINT)
                }
                RuntimeFailureContainmentEngine.FaultKind.CAMERA_STREAM_RESET -> {
                    add(RecoveryStep.REINITIALIZE_CAMERA)
                    add(RecoveryStep.RESTORE_LAST_CHECKPOINT)
                }
                RuntimeFailureContainmentEngine.FaultKind.RECONSTRUCTION_FAILURE,
                RuntimeFailureContainmentEngine.FaultKind.STALE_RUNTIME_STATE,
                RuntimeFailureContainmentEngine.FaultKind.SYNCHRONIZATION_FAILURE -> {
                    add(RecoveryStep.RESET_RECONSTRUCTION)
                    add(RecoveryStep.RESTORE_LAST_CHECKPOINT)
                }
                RuntimeFailureContainmentEngine.FaultKind.NATIVE_CRASH,
                RuntimeFailureContainmentEngine.FaultKind.FACE_EYE_FAILURE,
                RuntimeFailureContainmentEngine.FaultKind.HAND_TRACKING_FAILURE -> {
                    add(RecoveryStep.RESTART_WORKER)
                    add(RecoveryStep.RESTORE_LAST_CHECKPOINT)
                }
                RuntimeFailureContainmentEngine.FaultKind.UNKNOWN -> {
                    add(RecoveryStep.FLUSH_TRANSIENT_BUFFERS)
                }
            }
            if (decision.requiresOperatorAction) {
                add(RecoveryStep.REQUIRE_OPERATOR)
            } else if (decision.safeToContinue) {
                add(RecoveryStep.RESUME_CAPTURE)
            }
        }
        return RecoveryPlan(
            checkpoint = checkpoint,
            steps = steps.distinct(),
            canAutoRecover = decision.safeToContinue && checkpoint != null,
            warnings = decision.warnings,
        )
    }

    @Synchronized
    fun recordRecoverySuccess(timestampNanos: Long, frames: Int = 1) {
        ensureOpen()
        containment.observeSuccess(timestampNanos, frames)
    }

    @Synchronized
    fun reset() {
        ensureOpen()
        checkpoints.clear()
        containment.reset()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        checkpoints.clear()
        containment.close()
        closed = true
    }

    private fun ensureOpen() {
        check(!closed) { "recovery coordinator is closed" }
    }
}
