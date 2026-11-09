package com.nivukx.nox.core.model

/**
 * Acquisition modes for high-end motion capture.
 *
 * LIVE_MOTION_ONLY keeps the canonical motion-only workflow.
 * RECORD_AND_POSTPROCESS records camera evidence for an offline reconstruction pass.
 * HYBRID_LIVE_AND_POST exposes the live solution while retaining the full recorded evidence path.
 */
enum class CaptureEvidenceMode {
    LIVE_MOTION_ONLY,
    RECORD_AND_POSTPROCESS,
    HYBRID_LIVE_AND_POST,
}

/** Video codec preference for the evidence recorder. Selection is capability-negotiated. */
enum class RecordedVideoCodec {
    HEVC,
    AVC,
    AV1,
}

/** Offline reconstruction depth/geometry policy. */
enum class PostCaptureReconstructionTier {
    HIGH,
    ULTRA,
    REFERENCE,
}

/** Backpressure policy for visual evidence while motion telemetry remains authoritative. */
enum class CaptureEvidenceBackpressurePolicy {
    PRESERVE_MOTION_DROP_VISUAL,
    PRESERVE_ALL_FAIL_CLOSED,
    BLOCK_CAPTURE_ON_OVERFLOW,
}

/**
 * A negotiated video evidence profile.
 *
 * The recorder never promises that a requested codec, bit depth or frame rate exists on the
 * current Android device. The runtime exposes the negotiated result instead.
 */
data class RecordedVideoEvidenceProfile(
    val width: Int = 3840,
    val height: Int = 2160,
    val fps: Int = 60,
    val codec: RecordedVideoCodec = RecordedVideoCodec.HEVC,
    val bitrateBitsPerSecond: Int = 80_000_000,
    val iFrameIntervalSeconds: Int = 1,
    val requestTenBit: Boolean = true,
    val requestHdrTransfer: Boolean = false,
    val maxFileSizeBytes: Long = 512L * 1024L * 1024L,
) {
    init {
        require(width in 640..8_192)
        require(height in 360..4_320)
        require(fps in 24..120)
        require(bitrateBitsPerSecond in 1_000_000..250_000_000)
        require(iFrameIntervalSeconds in 1..10)
        require(maxFileSizeBytes in 64L * 1024L * 1024L..8L * 1024L * 1024L * 1024L)
    }

    val pixelsPerFrame: Long
        get() = width.toLong() * height.toLong()

    val megapixelsPerFrame: Double
        get() = pixelsPerFrame / 1_000_000.0

    companion object {
        fun liveUltra(): RecordedVideoEvidenceProfile =
            RecordedVideoEvidenceProfile(
                width = 3840,
                height = 2160,
                fps = 60,
                codec = RecordedVideoCodec.HEVC,
                bitrateBitsPerSecond = 80_000_000,
                iFrameIntervalSeconds = 1,
                requestTenBit = true,
            )

        fun archiveReference(): RecordedVideoEvidenceProfile =
            RecordedVideoEvidenceProfile(
                width = 7680,
                height = 4320,
                fps = 60,
                codec = RecordedVideoCodec.HEVC,
                bitrateBitsPerSecond = 200_000_000,
                iFrameIntervalSeconds = 1,
                requestTenBit = true,
                maxFileSizeBytes = 8L * 1024L * 1024L * 1024L,
            )
    }
}

/**
 * Formal description of the offline reconstruction pass.
 *
 * The stages are intentionally explicit so a recorded take can be reproducibly reconstructed on
 * device, workstation, or a future server backend without changing the captured evidence schema.
 */
data class PostCaptureReconstructionPolicy(
    val tier: PostCaptureReconstructionTier = PostCaptureReconstructionTier.ULTRA,
    val refineCameraGraph: Boolean = true,
    val refineTemporalAlignment: Boolean = true,
    val runSpatiotemporalBundleAdjustment: Boolean = true,
    val runDenseSurfaceReconstruction: Boolean = true,
    val runFaceMeshRefinement: Boolean = true,
    val runEyeGeometryRefinement: Boolean = true,
    val runHandGeometryRefinement: Boolean = true,
    val runContactConstraintOptimization: Boolean = true,
    val runDynamicStaticSceneDecomposition: Boolean = true,
    val runMetricScaleRefinement: Boolean = true,
    val runGlobal4DConsistency: Boolean = true,
    val preserveMicroMotion: Boolean = true,
    val exportIntermediateEvidence: Boolean = true,
) {
    companion object {
        fun reference(): PostCaptureReconstructionPolicy =
            PostCaptureReconstructionPolicy(
                tier = PostCaptureReconstructionTier.REFERENCE,
                refineCameraGraph = true,
                refineTemporalAlignment = true,
                runSpatiotemporalBundleAdjustment = true,
                runDenseSurfaceReconstruction = true,
                runFaceMeshRefinement = true,
                runEyeGeometryRefinement = true,
                runHandGeometryRefinement = true,
                runContactConstraintOptimization = true,
                runDynamicStaticSceneDecomposition = true,
                runMetricScaleRefinement = true,
                runGlobal4DConsistency = true,
                preserveMicroMotion = true,
                exportIntermediateEvidence = true,
            )
    }
}

/**
 * Capacity policy for dense multi-person operation.
 *
 * Fifty is the requested production target. Sixty-four is retained as an engineering ceiling so
 * the runtime can reserve bounded state without turning an accidental scene explosion into an
 * unbounded allocation problem.
 */
data class MultiPersonCapacityPolicy(
    val targetSubjects: Int = 50,
    val hardSubjectCeiling: Int = 64,
    val identityRetentionFrames: Int = 90,
    val reIdentificationFrames: Int = 180,
    val enableOcclusionPrediction: Boolean = true,
    val enableCrossViewIdentityFusion: Boolean = true,
    val enableAppearanceEmbeddingFusion: Boolean = true,
    val enableGeometricIdentityFusion: Boolean = true,
    val enableInteractionAwareAssociation: Boolean = true,
) {
    init {
        require(targetSubjects in 1..64)
        require(hardSubjectCeiling in targetSubjects..64)
        require(identityRetentionFrames in 1..600)
        require(reIdentificationFrames >= identityRetentionFrames)
    }

    companion object {
        fun fiftyPersonProduction(): MultiPersonCapacityPolicy =
            MultiPersonCapacityPolicy(
                targetSubjects = 50,
                hardSubjectCeiling = 64,
                identityRetentionFrames = 120,
                reIdentificationFrames = 300,
            )
    }
}

/** Deterministic per-actor resource allocation produced by the capacity governor. */
data class ActorRuntimeBudget(
    val trackId: Long,
    val ordinal: Int,
    val bodyInferenceHz: Int,
    val faceInferenceHz: Int,
    val eyeInferenceHz: Int,
    val handInferenceHz: Int,
    val denseSurfaceHz: Int,
    val reconstructionPriority: Double,
    val reserveStateBytes: Int,
    val occlusionPredictionFrames: Int,
    val identityReserve: Boolean,
) {
    init {
        require(ordinal >= 0)
        require(bodyInferenceHz in 1..240)
        require(faceInferenceHz in 0..240)
        require(eyeInferenceHz in 0..240)
        require(handInferenceHz in 0..120)
        require(denseSurfaceHz in 0..120)
        require(reconstructionPriority in 0.0..1.0)
        require(reserveStateBytes in 32 * 1024..16 * 1024 * 1024)
        require(occlusionPredictionFrames in 0..120)
    }
}

/** Bounded runtime allocation summary used by the operator telemetry layer. */
data class MultiPersonRuntimeBudget(
    val subjectCount: Int,
    val targetSubjectCount: Int,
    val hardSubjectCeiling: Int,
    val totalReservedBytes: Long,
    val activeTrackCount: Int,
    val predictedTrackCount: Int,
    val highPriorityTrackCount: Int,
    val visualEvidenceAdmissionRatio: Double,
    val warning: String? = null,
) {
    init {
        require(subjectCount >= 0)
        require(targetSubjectCount in 1..64)
        require(hardSubjectCeiling in targetSubjectCount..64)
        require(totalReservedBytes >= 0L)
        require(activeTrackCount >= 0)
        require(predictedTrackCount >= 0)
        require(highPriorityTrackCount >= 0)
        require(visualEvidenceAdmissionRatio in 0.0..1.0)
    }
}

/** Session-level state for the live/record/post pipeline. */
enum class HybridCaptureState {
    IDLE,
    PREPARING,
    ARMED,
    LIVE,
    RECORDING,
    HYBRID,
    DRAINING,
    POSTPROCESS_QUEUED,
    COMPLETE,
    DEGRADED,
    FAILED,
    CLOSED,
}

/** Explicit stage names persisted in the post-process graph. */
enum class PostCaptureStage {
    SESSION_INTEGRITY,
    MEDIA_EVIDENCE,
    CAMERA_GRAPH_REFINEMENT,
    CLOCK_AND_SUBFRAME_ALIGNMENT,
    MULTI_VIEW_BUNDLE_ADJUSTMENT,
    MULTI_PERSON_IDENTITY_RECONSTRUCTION,
    BODY_POSE_REFINEMENT,
    FACE_MESH_REFINEMENT,
    EYE_GEOMETRY_REFINEMENT,
    HAND_GEOMETRY_REFINEMENT,
    DEPTH_FUSION,
    DYNAMIC_STATIC_DECOMPOSITION,
    DENSE_SURFACE_RECONSTRUCTION,
    TEMPORAL_SURFACE_REFINEMENT,
    CONTACT_AND_KINEMATIC_OPTIMIZATION,
    METRIC_SCALE_REFINEMENT,
    GLOBAL_4D_CONSISTENCY,
    PRECISION_QUALIFICATION,
    EXPORT,
}

/** Runtime evidence counters. */
data class HybridCaptureEvidenceStats(
    val framesObserved: Long = 0L,
    val framesRecorded: Long = 0L,
    val framesDroppedForBackpressure: Long = 0L,
    val motionFramesEmitted: Long = 0L,
    val motionFramesDropped: Long = 0L,
    val visualBytesWritten: Long = 0L,
    val maxObservedSubjects: Int = 0,
    val activeSubjects: Int = 0,
    val predictedSubjects: Int = 0,
    val meanLatencyMs: Double = 0.0,
    val p95LatencyMs: Double = 0.0,
    val identitySwapRisk: Double = 0.0,
    val visualEvidenceAdmissionRatio: Double = 1.0,
    val integrityFailures: Long = 0L,
    val warnings: List<String> = emptyList(),
) {
    val recordingCoverageRatio: Double
        get() = if (framesObserved <= 0L) 1.0 else framesRecorded.toDouble() / framesObserved

    val motionCoverageRatio: Double
        get() = if (framesObserved <= 0L) 1.0 else motionFramesEmitted.toDouble() / framesObserved

    val maxSubjectCapacityUtilization: Double
        get() = (maxObservedSubjects / 50.0).coerceIn(0.0, 1.28)
}

/** One immutable record connecting motion evidence to deferred visual evidence. */
data class HybridCaptureFrameManifest(
    val sequence: Long,
    val sensorTimestampNanos: Long,
    val monotonicTimestampNanos: Long,
    val subjectCount: Int,
    val motionEvidencePresent: Boolean,
    val visualEvidencePresent: Boolean,
    val visualEvidenceOffset: Long,
    val visualEvidenceBytes: Int,
    val integrityHash: String,
) {
    init {
        require(sequence >= 0L)
        require(sensorTimestampNanos >= 0L)
        require(monotonicTimestampNanos >= 0L)
        require(subjectCount in 0..64)
        require(visualEvidenceOffset >= -1L)
        require(visualEvidenceBytes >= 0)
        require(integrityHash.isNotBlank())
    }
}

/**
 * Reproducible session manifest.
 *
 * This is metadata only. Motion state remains the canonical structured artifact; video evidence is
 * an optional observation source for post-processing and never silently replaces motion evidence.
 */
data class HybridCaptureSessionManifest(
    val sessionId: String,
    val createdAtEpochMillis: Long,
    val configurationFingerprint: String? = null,
    val evidenceMode: CaptureEvidenceMode,
    val videoProfile: RecordedVideoEvidenceProfile?,
    val reconstructionPolicy: PostCaptureReconstructionPolicy,
    val capacityPolicy: MultiPersonCapacityPolicy,
    val motionArtifactPath: String?,
    val visualEvidencePath: String?,
    val rawEvidencePath: String? = null,
    val depthEvidencePath: String?,
    val frameCount: Long,
    val maxObservedSubjects: Int,
    val state: HybridCaptureState,
    val stages: List<PostCaptureStage>,
    val warnings: List<String>,
    val integrityRootSha256: String? = null,
    val integritySealed: Boolean = false,
) {
    init {
        require(sessionId.isNotBlank())
        require(createdAtEpochMillis > 0L)
        require(frameCount >= 0L)
        require(maxObservedSubjects in 0..64)
        require(stages.isNotEmpty())
    }

    val recordsVideoEvidence: Boolean
        get() = evidenceMode != CaptureEvidenceMode.LIVE_MOTION_ONLY

    val exposesLiveMotion: Boolean
        get() = evidenceMode != CaptureEvidenceMode.RECORD_AND_POSTPROCESS
}

/** Explicit reconstruction plan passed to the existing offline refinement infrastructure. */
data class HybridPostCapturePlan(
    val sessionId: String,
    val stages: List<PostCaptureStage>,
    val prerequisiteChecks: List<String>,
    val expectedOutputs: List<String>,
    val targetSubjectCount: Int,
    val hardSubjectCeiling: Int,
    val evidenceMode: CaptureEvidenceMode,
    val tier: PostCaptureReconstructionTier,
) {
    init {
        require(sessionId.isNotBlank())
        require(stages.isNotEmpty())
        require(targetSubjectCount in 1..64)
        require(hardSubjectCeiling in targetSubjectCount..64)
        require(prerequisiteChecks.isNotEmpty())
        require(expectedOutputs.isNotEmpty())
    }
}

/** Human-readable capability summary for the Liquid Glass operator surface. */
data class HybridCaptureCapabilitySummary(
    val evidenceMode: CaptureEvidenceMode,
    val liveMotionAvailable: Boolean,
    val recordedVideoAvailable: Boolean,
    val postProcessingAvailable: Boolean,
    val targetSubjects: Int,
    val hardSubjectCeiling: Int,
    val faceTrackingAvailable: Boolean,
    val eyeTrackingAvailable: Boolean,
    val handTrackingAvailable: Boolean,
    val denseSurfaceAvailable: Boolean,
    val confidenceMessage: String,
) {
    val subjectCapacityLabel: String
        get() = "$targetSubjects target / $hardSubjectCeiling hard"
}

/**
 * Conservative precision statement. This object intentionally distinguishes requested precision
 * from physically qualified precision. A smaller target is not evidence that the capture achieved
 * that error bound.
 */
data class PrecisionQualificationState(
    val targetP95Millimeters: Double,
    val measuredP95Millimeters: Double?,
    val measuredP99Millimeters: Double?,
    val referenceUncertaintyMillimeters: Double?,
    val qualified: Boolean,
    val reason: String,
) {
    init {
        require(targetP95Millimeters > 0.0)
        require(measuredP95Millimeters == null || measuredP95Millimeters >= 0.0)
        require(measuredP99Millimeters == null || measuredP99Millimeters >= 0.0)
        require(referenceUncertaintyMillimeters == null || referenceUncertaintyMillimeters >= 0.0)
        require(reason.isNotBlank())
    }
}
