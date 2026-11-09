package com.nivukx.nox.runtime

import com.nivukx.nox.core.ai.FrontierEvidenceAuthorityEngine
import com.nivukx.nox.core.ai.FrontierMotionCaptureCoordinator
import com.nivukx.nox.core.ai.FrontierSessionRecoveryCoordinator
import com.nivukx.nox.core.ai.RuntimeFailureContainmentEngine
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DepthFrame
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.RuntimeDecisionTrace
import com.nivukx.nox.core.model.RuntimeDecisionTraceLedger
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.network.DistributedTransportCoordinator
import com.nivukx.nox.core.network.FrontierCameraTransportBudgetEngine
import com.nivukx.nox.core.network.FrontierDistributedTransportOrchestrator
import com.nivukx.nox.core.network.PeerTransportQualityEngine
import com.nivukx.nox.core.reconstruction.FrontierFaceEyeAuthorityCoordinator
import com.nivukx.nox.core.reconstruction.FrontierMotionReconstructionAdapter
import com.nivukx.nox.core.reconstruction.FrontierReconstructionOrchestrator
import com.nivukx.nox.core.reconstruction.NativeDepthFrameReconstructionSink
import com.nivukx.nox.core.sync.FrontierRigEvidenceCoordinator

/**
 * End-to-end frontier session boundary.
 *
 * Core modules remain decoupled. This app-layer coordinator is the authoritative convergence point
 * for motion, actor association, reconstruction, distributed transport, and deterministic recovery.
 */
class FrontierCaptureSessionCoordinator(
    private val motion: FrontierMotionCaptureCoordinator = FrontierMotionCaptureCoordinator(),
    private val reconstruction: FrontierReconstructionOrchestrator = FrontierReconstructionOrchestrator(),
    private val transport: FrontierDistributedTransportOrchestrator = FrontierDistributedTransportOrchestrator(),
    private val recovery: FrontierSessionRecoveryCoordinator = FrontierSessionRecoveryCoordinator(),
    private val faceEyeAuthority: FrontierFaceEyeAuthorityCoordinator = FrontierFaceEyeAuthorityCoordinator(),
    private val rigEvidenceAuthority: FrontierRigEvidenceCoordinator = FrontierRigEvidenceCoordinator(),
    private val checkpointIntervalFrames: Long = 120L,
    private val decisionLedger: RuntimeDecisionTraceLedger = RuntimeDecisionTraceLedger(),
    private val nativeDepthSink: NativeDepthFrameReconstructionSink? = null,
) : AutoCloseable {

    private var closed = false

    init {
        require(checkpointIntervalFrames > 0L)
    }

    enum class SessionState {
        LIVE,
        GUARDED,
        EVIDENCE_LIMITED,
        TRANSPORT_LIMITED,
        RECONSTRUCTION_LIMITED,
        RECOVERING,
        FAILED,
    }

    data class Input(
        val timestampNanos: Long,
        val motion: FrontierMotionCaptureCoordinator.Input,
        val transportChannels: List<FrontierCameraTransportBudgetEngine.CameraChannel>,
        val linkCapacityKbps: Double,
        val transportNodes: List<DistributedTransportCoordinator.Node> = emptyList(),
        val transportQualities: Map<DeviceId, PeerTransportQualityEngine.Quality> = emptyMap(),
        val reconstruction: FrontierReconstructionOrchestrator.Input? = null,
        val frontierReconstruction:
            FrontierReconstructionOrchestrator.FrontierInput? = null,
        val faceEye: FrontierFaceEyeAuthorityCoordinator.Input? = null,
        val rigEvidence: RigEvidenceInput? = null,
        val checkpointMaterial: ByteArray? = null,
        val captureFrameIndex: Long = 0L,
        val faultKind: RuntimeFailureContainmentEngine.FaultKind? = null,
        val faultSeverity: RuntimeFailureContainmentEngine.Severity =
            RuntimeFailureContainmentEngine.Severity.ERROR,
        val faultSignature: String = "frontier-session",
        val highEndSettings: HighEndMotionCaptureSettings =
            HighEndMotionCaptureSettings(),
        val depthFrames: List<DepthFrameEvidence> = emptyList(),
    ) {
        init {
            require(timestampNanos >= 0L)
            require(captureFrameIndex >= 0L)
            require(linkCapacityKbps.isFinite() && linkCapacityKbps > 0.0)
            require(depthFrames.size <= 128)
            require(depthFrames.map { it.cameraId }.distinct().size == depthFrames.size) {
                "A session step may contain at most one depth frame per camera"
            }
            require(depthFrames.all { it.frame.timestampNanos <= timestampNanos }) {
                "A depth frame cannot be newer than the session timestamp"
            }
            checkpointMaterial?.let { require(it.size <= 4 * 1024 * 1024) }
        }
    }

    data class DepthFrameEvidence(
        val cameraId: String,
        val frame: DepthFrame,
    ) {
        init {
            require(cameraId.isNotBlank() && cameraId.length <= 128)
            require(frame.timestampNanos > 0L)
        }
    }

    data class DepthFrameSubmission(
        val cameraId: String,
        val timestampNanos: Long,
        val result: NativeDepthFrameReconstructionSink.SubmissionResult?,
    )

    data class RigEvidenceInput(
        val cameras: List<FrontierRigEvidenceCoordinator.CameraInput>,
        val actorIds: Set<Int> = emptySet(),
        val actorPositionsMeters: Map<Int, Vec3> = emptyMap(),
        val actorPriorities: Map<Int, Double> = emptyMap(),
        val actorOcclusionRisk: Map<Int, Double> = emptyMap(),
    ) {
        init {
            require(actorIds.all { it >= 0 })
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: SessionState,
        val motion: FrontierMotionCaptureCoordinator.Result,
        val reconstruction: FrontierReconstructionOrchestrator.Result?,
        val faceEye: FrontierFaceEyeAuthorityCoordinator.Result?,
        val rigEvidence: FrontierRigEvidenceCoordinator.Result?,
        val transport: FrontierDistributedTransportOrchestrator.Result,
        val skeletons: List<SkeletonFrame>,
        val recovery: FrontierSessionRecoveryCoordinator.RecoveryPlan?,
        val checkpointId: Long?,
        val warnings: List<String>,
        val decisionTrace: RuntimeDecisionTrace,
        val depthFrameSubmissions: List<DepthFrameSubmission> = emptyList(),
    )

    @Synchronized
    fun process(input: Input): Result {
        check(!closed) {
            "FrontierCaptureSessionCoordinator is closed"
        }
        val motionResult = motion.process(
            input.motion.copy(timestampNanos = input.timestampNanos),
        )
        val transportResult = transport.evaluate(
            channels = input.transportChannels,
            linkCapacityKbps = input.linkCapacityKbps,
            timestampNanos = input.timestampNanos,
            expectedPeerCount = input.transportChannels.size.coerceAtLeast(1),
            nodes = input.transportNodes,
            qualities = input.transportQualities,
        )
        val skeletons =
            FrontierMotionReconstructionAdapter
                .adapt(motionResult.motion)
                .skeletons
        val reconstructionResult =
            input.reconstruction?.let { request ->
                reconstruction.process(
                    request.copy(
                        frontier =
                            input.frontierReconstruction
                                ?: request.frontier,
                    ),
                )
            } ?: input.frontierReconstruction?.let {
                reconstruction.process(
                    FrontierReconstructionOrchestrator.Input(
                        timestampNanos = input.timestampNanos,
                        regions = emptyList(),
                        surfacePoints = emptyList(),
                        globalCameras = emptyList(),
                        globalPoints = emptyList(),
                        frontier = it,
                    ),
                )
            }
        val faceEyeResult =
            input.faceEye?.let(faceEyeAuthority::process)
        input.rigEvidence?.let {
            rigEvidenceAuthority.configure(
                input.highEndSettings,
            )
        }

        val rigEvidenceResult = input.rigEvidence?.let { rig ->
            rigEvidenceAuthority.evaluate(
                timestampNanos = input.timestampNanos,
                cameras = rig.cameras,
                actorIds = rig.actorIds,
                actorPositionsMeters = rig.actorPositionsMeters,
                actorPriorities = rig.actorPriorities,
                actorOcclusionRisk = rig.actorOcclusionRisk,
            )
        }

        // Resolve depth submissions before checkpoint stage selection: the checkpoint must
        // describe the actual integration result for this frame, not a later/uninitialized value.
        val depthFrameSubmissions =
            input.depthFrames.map { evidence ->
                val submissionResult =
                    if (nativeDepthSink == null) {
                        null
                    } else {
                        runCatching {
                            nativeDepthSink.submit(
                                cameraId = evidence.cameraId,
                                frame = evidence.frame,
                            )
                        }.getOrElse {
                            NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATION_FAILED
                        }
                    }
                DepthFrameSubmission(
                    cameraId = evidence.cameraId,
                    timestampNanos = evidence.frame.timestampNanos,
                    result = submissionResult,
                )
            }
        
        val shouldCheckpoint =
            input.checkpointMaterial != null &&
                (
                    input.captureFrameIndex == 0L ||
                        input.captureFrameIndex % checkpointIntervalFrames == 0L ||
                        recovery.latestCheckpoint() == null
                )
        val checkpoint =
            input.checkpointMaterial
                ?.takeIf { shouldCheckpoint }
                ?.let { material ->
                    recovery.checkpoint(
                    timestampNanos = input.timestampNanos,
                    actorCount =
                        motionResult.identities.accepted
                            .count { !it.alternate }
                            .coerceIn(0, 64),
                    cameraCount = input.transportChannels.size.coerceIn(0, 128),
                    authoritativeCameraCount =
                        transportResult.budget.allocations
                            .count {
                                it.tier ==
                                    FrontierCameraTransportBudgetEngine.TransportTier.FULL_EVIDENCE
                            }
                            .coerceIn(0, 32),
                    reconstructionFrame = input.captureFrameIndex,
                    activeStages = buildSet {
                        add("motion")
                        add("identity")
                        add("transport")
                        if (reconstructionResult != null) add("reconstruction")
                        if (depthFrameSubmissions.any { it.result == NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED }) {
                            add("native-depth-tsdf")
                        }
                        if (faceEyeResult != null) add("face-eye")
                        if (rigEvidenceResult != null) add("rig-authority")
                    },
                    stateMaterial = material,
                )
            }

        val recoveryPlan =
            input.faultKind?.let { fault ->
                recovery.planRecovery(
                    timestampNanos = input.timestampNanos,
                    faultKind = fault,
                    severity = input.faultSeverity,
                    signature = input.faultSignature,
                )
            }

        // Depth integration is opt-in: the session never fabricates camera pose or intrinsics.
        // Exceptions from an adapter are converted into an explicit failed submission.

        val state =
            stateOf(
                motion = motionResult,
                reconstruction = reconstructionResult,
                transport = transportResult,
                recovery = recoveryPlan,
                rigEvidence = rigEvidenceResult,
                depthFrameSubmissions = depthFrameSubmissions,
            )


        val stateScore = when (state) {
            SessionState.LIVE -> 1.0
            SessionState.GUARDED -> 0.75
            SessionState.EVIDENCE_LIMITED -> 0.45
            SessionState.TRANSPORT_LIMITED -> 0.40
            SessionState.RECONSTRUCTION_LIMITED -> 0.38
            SessionState.RECOVERING -> 0.25
            SessionState.FAILED -> 0.0
        }
        val decisionTrace = decisionLedger.append(
            timestampNanos = input.timestampNanos,
            stage = "frontier-session",
            state = state.name,
            decision = when (state) {
                SessionState.LIVE -> "continue"
                SessionState.GUARDED -> "constrain"
                SessionState.EVIDENCE_LIMITED,
                SessionState.TRANSPORT_LIMITED,
                SessionState.RECONSTRUCTION_LIMITED -> "hold-quality"
                SessionState.RECOVERING -> "recover"
                SessionState.FAILED -> "operator-intervention"
            },
            severity = when (state) {
                SessionState.LIVE -> RuntimeDecisionTrace.Severity.INFO
                SessionState.GUARDED,
                SessionState.EVIDENCE_LIMITED,
                SessionState.TRANSPORT_LIMITED,
                SessionState.RECONSTRUCTION_LIMITED,
                SessionState.RECOVERING -> RuntimeDecisionTrace.Severity.WARNING
                SessionState.FAILED -> RuntimeDecisionTrace.Severity.CRITICAL
            },
            score = stateScore,
            causeCodes = buildList {
                addAll(transportResult.warnings.map { "transport:" + it })
                addAll(motionResult.evidenceAuthority.warnings.map { "evidence:" + it })
                motionResult.crowdEvidence?.warnings?.forEach {
                    add("crowd:" + it)
                }
                recoveryPlan?.warnings?.forEach { add("recovery:" + it) }
                faceEyeResult?.warnings?.forEach { add("face-eye:" + it) }
                rigEvidenceResult?.warnings?.forEach { add("rig:" + it) }
                depthFrameSubmissions.forEach { submission ->
                    submission.result?.let { add("native-depth:" + it.name.lowercase()) }
                        ?: add("native-depth-sink-not-configured")
                }
                if (motionResult.identities.ambiguousDetectionIds.isNotEmpty()) {
                    add("identity:ambiguity")
                }
            }.distinct(),
            evidenceMaterial = (
                input.timestampNanos.toString() +
                    "|" + state.name +
                    "|" + transportResult.budget.totalAllocatedKbps +
                    "|" + motionResult.evidenceAuthority.authorityScore +
                    "|" + (rigEvidenceResult?.cameraAuthority?.authorityScore ?: -1.0)
                ).toByteArray(),
        )


        val warnings = buildList {
            addAll(motionResult.evidenceAuthority.warnings)
            motionResult.crowdEvidence?.warnings?.let(::addAll)
            if (
                motionResult.crowdEvidence?.interactionPressure ?: 0.0 >= 0.72
            ) {
                add("crowd-interaction-pressure-high")
            }
            addAll(transportResult.warnings)
            reconstructionResult?.provenance?.let {
                if (it.inferredRatio > 0.40) add("reconstruction-inference-ratio-high")
            }
            reconstructionResult?.frontier?.warnings?.forEach {
                add("frontier-reconstruction-$it")
            }
            reconstructionResult?.frontier?.failure
                ?.takeIf { it != com.nivukx.nox.core.model.MeasurementFailure.NONE }
                ?.let {
                    add("frontier-reconstruction-failure-$it")
                }
            recoveryPlan?.warnings?.let(::addAll)
            if (motionResult.identities.ambiguousDetectionIds.isNotEmpty()) {
                add("identity-ambiguity-present")
            }
            faceEyeResult?.let { result ->
                if (!result.authoritative) addAll(result.warnings.map { "face-eye-$it" })
            }
            rigEvidenceResult?.let { result ->
                if (!result.authoritativeReady) addAll(result.warnings.map { "rig-$it" })
            }
            depthFrameSubmissions.forEach { submission ->
                when (submission.result) {
                    null -> add("native-depth-sink-not-configured")
                    NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATED -> Unit
                    NativeDepthFrameReconstructionSink.SubmissionResult.STALE_FRAME ->
                        add("native-depth-stale-frame-dropped")
                    NativeDepthFrameReconstructionSink.SubmissionResult.NO_CALIBRATION ->
                        add("native-depth-calibration-unavailable")
                    NativeDepthFrameReconstructionSink.SubmissionResult.INVALID_CALIBRATION ->
                        add("native-depth-calibration-invalid")
                    NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATION_FAILED ->
                        add("native-depth-integration-failed")
                    NativeDepthFrameReconstructionSink.SubmissionResult.CLOSED ->
                        add("native-depth-sink-closed")
                }
            }
        }.distinct()

        return Result(
            timestampNanos = input.timestampNanos,
            state = state,
            motion = motionResult,
            reconstruction = reconstructionResult,
            faceEye = faceEyeResult,
            rigEvidence = rigEvidenceResult,
            transport = transportResult,
            skeletons = skeletons,
            recovery = recoveryPlan,
            checkpointId = checkpoint?.checkpointId,
            warnings = warnings,
            decisionTrace = decisionTrace,
            depthFrameSubmissions = depthFrameSubmissions,
        )
    }

    @Synchronized
    fun recentDecisionTraces(limit: Int = 64): List<RuntimeDecisionTrace> =
        decisionLedger.recent(limit)

    @Synchronized
    fun decisionLedgerDigest(): String = decisionLedger.ledgerDigest()

    @Synchronized
    fun recordRecoverySuccess(timestampNanos: Long, frames: Int = 1) {
        recovery.recordRecoverySuccess(timestampNanos, frames)
    }

    private fun stateOf(
        motion: FrontierMotionCaptureCoordinator.Result,
        reconstruction: FrontierReconstructionOrchestrator.Result?,
        transport: FrontierDistributedTransportOrchestrator.Result,
        recovery: FrontierSessionRecoveryCoordinator.RecoveryPlan?,
        rigEvidence: FrontierRigEvidenceCoordinator.Result? = null,
        depthFrameSubmissions: List<DepthFrameSubmission> = emptyList(),
    ): SessionState =
        when {
            recovery?.steps?.contains(
                FrontierSessionRecoveryCoordinator.RecoveryStep.REQUIRE_OPERATOR,
            ) == true ->
                SessionState.FAILED
            recovery != null ->
                SessionState.RECOVERING
            depthFrameSubmissions.any { submission ->
                submission.result == null ||
                    submission.result == NativeDepthFrameReconstructionSink.SubmissionResult.NO_CALIBRATION ||
                    submission.result == NativeDepthFrameReconstructionSink.SubmissionResult.INVALID_CALIBRATION ||
                    submission.result == NativeDepthFrameReconstructionSink.SubmissionResult.INTEGRATION_FAILED ||
                    submission.result == NativeDepthFrameReconstructionSink.SubmissionResult.CLOSED
            } ->
                SessionState.RECONSTRUCTION_LIMITED
            transport.budget.authoritativeUnderfunded > 0 ->
                SessionState.TRANSPORT_LIMITED
            rigEvidence != null && !rigEvidence.authoritativeReady ->
                SessionState.EVIDENCE_LIMITED
            motion.evidenceAuthority.state ==
                FrontierEvidenceAuthorityEngine.State.REJECTED ->
                SessionState.EVIDENCE_LIMITED
            reconstruction != null &&
                reconstruction.frontier?.failure
                    ?.takeIf {
                        it != com.nivukx.nox.core.model.MeasurementFailure.NONE
                    } != null ->
                SessionState.RECONSTRUCTION_LIMITED
            reconstruction != null &&
                reconstruction.globalRefinement.finalResidualMeters >
                    reconstruction.globalRefinement.initialResidualMeters ->
                SessionState.RECONSTRUCTION_LIMITED
            motion.evidenceAuthority.state ==
                FrontierEvidenceAuthorityEngine.State.DEGRADED ->
                SessionState.GUARDED
            else ->
                SessionState.LIVE
        }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true

        var failure: Throwable? = null
        fun closeStage(action: () -> Unit) {
            runCatching(action).onFailure { error ->
                val prior = failure
                if (prior == null) {
                    failure = error
                } else {
                    prior.addSuppressed(error)
                }
            }
        }

        closeStage { motion.close() }
        closeStage { recovery.close() }
        closeStage { faceEyeAuthority.close() }
        closeStage { reconstruction.close() }
        closeStage { transport.close() }
        closeStage { rigEvidenceAuthority.close() }
        closeStage { nativeDepthSink?.close() }
        closeStage { decisionLedger.close() }

        failure?.let { throw it }
    }
}
