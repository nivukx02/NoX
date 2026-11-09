package com.nivukx.nox.core.capture

import android.content.Context
import com.nivukx.nox.core.model.CaptureEvidenceMode
import com.nivukx.nox.core.model.HybridCaptureSessionManifest
import com.nivukx.nox.core.model.HybridCaptureState
import com.nivukx.nox.core.model.MultiPersonCapacityPolicy
import com.nivukx.nox.core.model.PostCaptureReconstructionPolicy
import com.nivukx.nox.core.model.PostCaptureStage
import com.nivukx.nox.core.model.RecordedVideoEvidenceProfile
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

/**
 * Coordinates the live-motion and deferred visual-evidence paths.
 *
 * SessionRecorder remains the authoritative structured motion artifact. This coordinator adds
 * optional visual evidence and produces a reproducible post-capture manifest that can feed the
 * existing reconstruction/refinement engines.
 *
 * Camera2 ordering is deliberate: prepare() must happen before Camera2CaptureController.open(),
 * because the recorder input Surface must be part of the capture session when that session is built.
 */
class HybridCaptureCoordinator(
    private val context: Context,
    private val camera: Camera2CaptureController,
) : AutoCloseable {

    data class PreparedSession(
        val sessionId: String,
        val configurationFingerprint: String?,
        val evidenceMode: CaptureEvidenceMode,
        val visualEvidenceFile: File?,
        val videoProfile: RecordedVideoEvidenceProfile?,
        val state: HybridCaptureState,
    )

    data class Completion(
        val manifest: HybridCaptureSessionManifest,
        val videoResult: HighEndVideoEvidenceRecorder.Result?,
    )

    private val stateRef = AtomicReference(HybridCaptureState.IDLE)

    private var recorder: HighEndVideoEvidenceRecorder? = null
    private var preparedSession: PreparedSession? = null
    private var evidenceMode = CaptureEvidenceMode.LIVE_MOTION_ONLY
    private var reconstructionPolicy = PostCaptureReconstructionPolicy()
    private var capacityPolicy = MultiPersonCapacityPolicy.fiftyPersonProduction()

    private var motionArtifactPath: String? = null
    private var depthEvidencePath: String? = null
    private var observedFrames = 0L
    private var observedSubjects = 0
    private var warningBuffer = mutableListOf<String>()

    val state: HybridCaptureState
        get() = stateRef.get()

    fun prepare(
        sessionId: String,
        mode: CaptureEvidenceMode,
        videoProfile: RecordedVideoEvidenceProfile? = null,
        configurationFingerprint: String? = null,
        reconstructionPolicy: PostCaptureReconstructionPolicy =
            PostCaptureReconstructionPolicy(),
        capacityPolicy: MultiPersonCapacityPolicy =
            MultiPersonCapacityPolicy.fiftyPersonProduction(),
        motionArtifactPath: String? = null,
        depthEvidencePath: String? = null,
    ): PreparedSession {
        check(stateRef.get() == HybridCaptureState.IDLE) {
            "Hybrid capture coordinator must be idle before prepare"
        }
        require(sessionId.isNotBlank())

        stateRef.set(HybridCaptureState.PREPARING)
        this.evidenceMode = mode
        this.reconstructionPolicy = reconstructionPolicy
        this.capacityPolicy = capacityPolicy
        this.motionArtifactPath = motionArtifactPath
        this.depthEvidencePath = depthEvidencePath
        warningBuffer = mutableListOf()
        observedFrames = 0L
        observedSubjects = 0

        val visualFile =
            if (mode == CaptureEvidenceMode.LIVE_MOTION_ONLY) {
                null
            } else {
                require(videoProfile != null) {
                    "A video evidence profile is required for $mode"
                }

                val file = createEvidenceFile(sessionId)
                val newRecorder =
                    HighEndVideoEvidenceRecorder(
                        requested = videoProfile,
                        outputFile = file,
                    )

                try {
                    newRecorder.prepare()
                } catch (error: Throwable) {
                    stateRef.set(HybridCaptureState.FAILED)
                    throw error
                }

                recorder = newRecorder
                camera.setRecordingSurface(newRecorder.surface)
                file
            }

        val prepared =
            PreparedSession(
                sessionId = sessionId,
                configurationFingerprint = configurationFingerprint,
                evidenceMode = mode,
                visualEvidenceFile = visualFile,
                videoProfile = videoProfile,
                state = HybridCaptureState.ARMED,
            )

        preparedSession = prepared
        stateRef.set(HybridCaptureState.ARMED)
        return prepared
    }

    fun start() {
        val prepared =
            preparedSession
                ?: error("Hybrid capture has not been prepared")

        check(stateRef.get() == HybridCaptureState.ARMED) {
            "Hybrid capture must be armed before start"
        }

        if (prepared.evidenceMode != CaptureEvidenceMode.LIVE_MOTION_ONLY) {
            recorder?.start()
        }

        stateRef.set(
            when (prepared.evidenceMode) {
                CaptureEvidenceMode.LIVE_MOTION_ONLY ->
                    HybridCaptureState.LIVE
                CaptureEvidenceMode.RECORD_AND_POSTPROCESS ->
                    HybridCaptureState.RECORDING
                CaptureEvidenceMode.HYBRID_LIVE_AND_POST ->
                    HybridCaptureState.HYBRID
            },
        )
    }

    fun observeFrame(
        frameSequence: Long,
        subjectCount: Int,
    ) {
        val current = stateRef.get()
        if (
            current != HybridCaptureState.LIVE &&
            current != HybridCaptureState.RECORDING &&
            current != HybridCaptureState.HYBRID
        ) {
            return
        }

        observedFrames =
            maxOf(
                observedFrames,
                frameSequence + 1L,
            )

        observedSubjects =
            maxOf(
                observedSubjects,
                subjectCount.coerceIn(0, 64),
            )

        if (subjectCount > capacityPolicy.targetSubjects) {
            warningBuffer +=
                "Observed $subjectCount subjects above the configured " +
                    "${capacityPolicy.targetSubjects}-subject production target."
            warningBuffer =
                warningBuffer
                    .distinct()
                    .takeLast(12)
                    .toMutableList()
        }
    }

    fun finish(
        sessionId: String,
        maxSubjectsObserved: Int = observedSubjects,
        frameCount: Long = observedFrames,
        motionArtifactPath: String? = this.motionArtifactPath,
        rawEvidencePath: String? = null,
        depthEvidencePath: String? = this.depthEvidencePath,
    ): Completion {
        val current = stateRef.get()
        check(
            current == HybridCaptureState.LIVE ||
                current == HybridCaptureState.RECORDING ||
                current == HybridCaptureState.HYBRID
        ) {
            "Hybrid capture is not running"
        }

        stateRef.set(HybridCaptureState.DRAINING)
        camera.setRecordingSurface(null)

        val videoResult = recorder?.stop()
        recorder = null

        val finalState =
            if (
                videoResult?.state ==
                    HighEndVideoEvidenceRecorder.State.FAILED
            ) {
                warningBuffer +=
                    "Visual evidence recorder failed: " +
                        (videoResult.error ?: "unknown error")
                stateRef.set(HybridCaptureState.FAILED)
                HybridCaptureState.FAILED
            } else {
                stateRef.set(
                    if (evidenceMode == CaptureEvidenceMode.LIVE_MOTION_ONLY) {
                        HybridCaptureState.COMPLETE
                    } else {
                        HybridCaptureState.POSTPROCESS_QUEUED
                    },
                )
                stateRef.get()
            }

        val integrity =
            EvidenceIntegritySealer.seal(
                configurationFingerprint =
                    preparedSession?.configurationFingerprint
                        ?: sessionId,
                artifacts =
                    listOf(
                        "visual-evidence" to
                            preparedSession?.visualEvidenceFile,
                        "raw-sensor-evidence" to
                            rawEvidencePath?.let(::File),
                        "depth-evidence" to
                            depthEvidencePath?.let(::File),
                    ),
            )

        val manifest =
            HybridCaptureSessionManifest(
                sessionId = sessionId,
                createdAtEpochMillis = System.currentTimeMillis(),
                configurationFingerprint =
                    preparedSession?.configurationFingerprint,
                evidenceMode = evidenceMode,
                videoProfile = preparedSession?.videoProfile,
                reconstructionPolicy = reconstructionPolicy,
                capacityPolicy = capacityPolicy,
                motionArtifactPath = motionArtifactPath,
                visualEvidencePath =
                    preparedSession?.visualEvidenceFile?.absolutePath,
                rawEvidencePath = rawEvidencePath,
                depthEvidencePath = depthEvidencePath,
                frameCount = frameCount.coerceAtLeast(0L),
                maxObservedSubjects =
                    maxSubjectsObserved.coerceIn(0, 64),
                state = finalState,
                stages = buildStageList(evidenceMode),
                warnings =
                    (
                        warningBuffer +
                            integrity.warnings
                        )
                        .distinct()
                        .takeLast(24),
                integrityRootSha256 =
                    integrity.rootSha256,
                integritySealed =
                    integrity.sealed,
            )

        preparedSession = null

        return Completion(
            manifest = manifest,
            videoResult = videoResult,
        )
    }

    fun fail(reason: String) {
        if (reason.isNotBlank()) {
            warningBuffer += reason
            warningBuffer =
                warningBuffer.distinct().takeLast(24).toMutableList()
        }

        stateRef.set(HybridCaptureState.FAILED)
        camera.setRecordingSurface(null)

        runCatching {
            recorder?.close()
        }
        recorder = null
    }

    fun reset() {
        check(stateRef.get() != HybridCaptureState.CLOSED) {
            "Hybrid capture coordinator is closed"
        }

        camera.setRecordingSurface(null)
        recorder?.close()
        recorder = null
        preparedSession = null
        observedFrames = 0L
        observedSubjects = 0
        warningBuffer.clear()
        stateRef.set(HybridCaptureState.IDLE)
    }

    override fun close() {
        if (stateRef.get() == HybridCaptureState.CLOSED) return

        camera.setRecordingSurface(null)
        runCatching {
            recorder?.close()
        }
        recorder = null
        preparedSession = null
        stateRef.set(HybridCaptureState.CLOSED)
    }

    private fun buildStageList(
        mode: CaptureEvidenceMode,
    ): List<PostCaptureStage> {
        if (mode == CaptureEvidenceMode.LIVE_MOTION_ONLY) {
            return listOf(
                PostCaptureStage.SESSION_INTEGRITY,
                PostCaptureStage.PRECISION_QUALIFICATION,
                PostCaptureStage.EXPORT,
            )
        }

        return listOf(
            PostCaptureStage.SESSION_INTEGRITY,
            PostCaptureStage.MEDIA_EVIDENCE,
            PostCaptureStage.CAMERA_GRAPH_REFINEMENT,
            PostCaptureStage.CLOCK_AND_SUBFRAME_ALIGNMENT,
            PostCaptureStage.MULTI_VIEW_BUNDLE_ADJUSTMENT,
            PostCaptureStage.MULTI_PERSON_IDENTITY_RECONSTRUCTION,
            PostCaptureStage.BODY_POSE_REFINEMENT,
            PostCaptureStage.DEPTH_FUSION,
            PostCaptureStage.DYNAMIC_STATIC_DECOMPOSITION,
            PostCaptureStage.FACE_MESH_REFINEMENT,
            PostCaptureStage.EYE_GEOMETRY_REFINEMENT,
            PostCaptureStage.HAND_GEOMETRY_REFINEMENT,
            PostCaptureStage.DENSE_SURFACE_RECONSTRUCTION,
            PostCaptureStage.TEMPORAL_SURFACE_REFINEMENT,
            PostCaptureStage.CONTACT_AND_KINEMATIC_OPTIMIZATION,
            PostCaptureStage.METRIC_SCALE_REFINEMENT,
            PostCaptureStage.GLOBAL_4D_CONSISTENCY,
            PostCaptureStage.PRECISION_QUALIFICATION,
            PostCaptureStage.EXPORT,
        )
    }

    private fun createEvidenceFile(
        sessionId: String,
    ): File {
        val root =
            context.getExternalFilesDir("nox_evidence")
                ?: File(
                    context.filesDir,
                    "nox_evidence",
                )

        val date =
            SimpleDateFormat(
                "yyyyMMdd_HHmmss_SSS",
                Locale.US,
            ).format(Date())

        val sanitized =
            sessionId
                .replace(
                    Regex("[^A-Za-z0-9._-]"),
                    "_",
                )
                .take(80)

        root.mkdirs()

        return File(
            root,
            "nox_${sanitized}_${date}.mp4",
        )
    }
}
