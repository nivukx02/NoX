package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CaptureQualityTier
import com.nivukx.nox.core.model.ProductionCaptureProfile

/**
 * Deterministic post-capture reconstruction refinement planner.
 *
 * Real-time mobile capture should not be forced to perform every expensive
 * reconstruction operation. This planner creates an explicit offline graph
 * that can be executed later by a workstation, server, or a future on-device
 * high-performance target.
 *
 * The planner describes work and prerequisites. It does not pretend that
 * external neural models are bundled into NoX when they are not.
 */
class HighEndOfflineRefinementPlanner {

    enum class Stage {
        SESSION_INTEGRITY,
        CAMERA_GRAPH_REFINEMENT,
        TEMPORAL_ALIGNMENT_REFINEMENT,
        MULTI_VIEW_BUNDLE_ADJUSTMENT,
        KINEMATIC_TRAJECTORY_REFINEMENT,
        PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
        GLOBAL_4D_CONSISTENCY_REFINEMENT,
        FACE_EYE_CALIBRATION_DRIFT_REFINEMENT,
        CROSS_MODAL_IDENTITY_REFINEMENT,
        FACE_EYE_EVIDENCE_REFINEMENT,
        DYNAMIC_STATIC_SCENE_DECOMPOSITION,
        FACE_REGION_DENSIFICATION,
        HAND_REGION_DENSIFICATION,
        SURFACE_TSDF_REINTEGRATION,
        CANONICAL_SURFACE_TEMPORAL_PRIOR,
        DYNAMIC_SURFACE_EVOLUTION,
        REGION_AWARE_NEURAL_REFINE,
        NOVEL_VIEW_REFINEMENT,
        EXPORT_QUALITY_ASSURANCE,
    }

    enum class ExecutionClass {
        DETERMINISTIC,
        GEOMETRIC_OPTIMIZATION,
        NEURAL_OPTIONAL,
        RENDERING,
        VALIDATION,
    }

    data class Input(
        val profile: ProductionCaptureProfile,
        val connectedCameraCount: Int,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val reconstructionConfidence: Double,
        val temporalConfidence: Double,
        val denseSurfaceConfidence: Double,
        val surfaceNormalObservability: Double = 1.0,
        val surfaceHighUncertaintyRatio: Double = 0.0,
        val cameraInformationGain: Double = 1.0,
        val faceConfidence: Double,
        val handConfidence: Double,
        val runtimeQualityScore: Double,
        val faceEyeFusionConfidence: Double = 1.0,
        val faceEyeFusionAccepted: Boolean = true,
        val faceEyeCalibrationDriftScore: Double = 0.0,
        val physicsConsistencyScore: Double = 1.0,
        val physicsConstraintAccepted: Boolean = true,
        val physicsMeanBoneResidualMeters: Double = 0.0,
        val global4DScore: Double = 1.0,
        val global4DAccepted: Boolean = true,
        val global4DPersistentFailure: Boolean = false,
        val global4DRootMotionConsistency: Double = 1.0,
        val global4DSceneConsistency: Double = 0.72,
        val meanNormalChangeDegrees: Double = 0.0,
        val canonicalSurfaceConfidence: Double = 1.0,
        val canonicalSurfaceAccepted: Boolean = true,
        val canonicalCorrespondenceCoverage: Double = 1.0,
        val canonicalEdgeStretchScore: Double = 1.0,
        val canonicalTemporalConsistency: Double = 1.0,
        val sessionIntegrityValid: Boolean,
    ) {
        init {
            require(connectedCameraCount >= 0)
            require(calibrationConfidence in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(reconstructionConfidence in 0.0..1.0)
            require(temporalConfidence in 0.0..1.0)
            require(denseSurfaceConfidence in 0.0..1.0)
            require(surfaceNormalObservability in 0.0..1.0)
            require(surfaceHighUncertaintyRatio in 0.0..1.0)
            require(cameraInformationGain in 0.0..1.0)
            require(faceConfidence in 0.0..1.0)
            require(handConfidence in 0.0..1.0)
            require(runtimeQualityScore in 0.0..1.0)
            require(faceEyeFusionConfidence in 0.0..1.0)
            require(faceEyeCalibrationDriftScore in 0.0..1.0)
            require(physicsConsistencyScore in 0.0..1.0)
            require(physicsMeanBoneResidualMeters.isFinite() && physicsMeanBoneResidualMeters >= 0.0)
            require(global4DScore in 0.0..1.0)
            require(global4DRootMotionConsistency in 0.0..1.0)
            require(global4DSceneConsistency in 0.0..1.0)
            require(meanNormalChangeDegrees.isFinite() && meanNormalChangeDegrees >= 0.0)
            require(canonicalSurfaceConfidence in 0.0..1.0)
            require(canonicalCorrespondenceCoverage in 0.0..1.0)
            require(canonicalEdgeStretchScore in 0.0..1.0)
            require(canonicalTemporalConsistency in 0.0..1.0)
        }
    }

    data class PlannedStage(
        val stage: Stage,
        val executionClass: ExecutionClass,
        val enabled: Boolean,
        val priority: Int,
        val budget: Int,
        val expectedGain: Double,
        val prerequisites: List<Stage>,
        val rationale: String,
    )

    data class Plan(
        val profileId: String,
        val overallConfidence: Double,
        val stages: List<PlannedStage>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val executableStages: List<PlannedStage>
            get() = stages.filter { it.enabled }

        val deterministicStageCount: Int
            get() =
                executableStages.count {
                    it.executionClass == ExecutionClass.DETERMINISTIC ||
                        it.executionClass == ExecutionClass.GEOMETRIC_OPTIMIZATION ||
                        it.executionClass == ExecutionClass.VALIDATION
                }

        val neuralOptionalStageCount: Int
            get() =
                executableStages.count {
                    it.executionClass == ExecutionClass.NEURAL_OPTIONAL
                }
    }

    fun createPlan(
        input: Input,
    ): Plan {
        val profile =
            input.profile.validated()

        val multiViewReady =
            input.connectedCameraCount >=
                profile.multiViewConsensusThreshold &&
                input.calibrationConfidence >= 0.70 &&
                input.synchronizationConfidence >= 0.70

        val surfaceEvidenceOverall =
            input.denseSurfaceConfidence * 0.52 +
                input.surfaceNormalObservability * 0.20 +
                (
                    1.0 -
                        input.surfaceHighUncertaintyRatio
                ).coerceIn(0.0, 1.0) * 0.18 +
                input.cameraInformationGain * 0.10

        val baseOverall =
            (
                input.reconstructionConfidence * 0.23 +
                    input.calibrationConfidence * 0.17 +
                    input.synchronizationConfidence * 0.15 +
                    input.temporalConfidence * 0.15 +
                    input.denseSurfaceConfidence * 0.07 +
                    surfaceEvidenceOverall * 0.02 +
                    input.faceConfidence * 0.05 +
                    input.faceEyeFusionConfidence * 0.08 +
                    (
                        1.0 -
                            input.faceEyeCalibrationDriftScore
                        ).coerceIn(0.0, 1.0) * 0.03 +
                    input.handConfidence * 0.04 +
                    input.canonicalSurfaceConfidence * 0.03 +
                    input.canonicalTemporalConsistency * 0.02 +
                    input.runtimeQualityScore * 0.01
                ).coerceIn(0.0, 1.0)

        val overall =
            (
                baseOverall * 0.79 +
                    input.physicsConsistencyScore * 0.11 +
                    input.global4DScore * 0.10
                ).coerceIn(0.0, 1.0)

        val stages = mutableListOf<PlannedStage>()

        stages += PlannedStage(
            stage = Stage.SESSION_INTEGRITY,
            executionClass = ExecutionClass.VALIDATION,
            enabled = input.sessionIntegrityValid,
            priority = 100,
            budget = 1,
            expectedGain = 0.0,
            prerequisites = emptyList(),
            rationale =
                "Validate manifest, timestamps, binary streams, and cryptographic artifacts before geometry refinement.",
        )

        stages += PlannedStage(
            stage = Stage.CAMERA_GRAPH_REFINEMENT,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled = multiViewReady,
            priority = 95,
            budget = profile.bundleAdjustmentIterations,
            expectedGain =
                (1.0 - input.calibrationConfidence) * 0.80,
            prerequisites = listOf(Stage.SESSION_INTEGRITY),
            rationale =
                "Refine camera poses and intrinsics before using multi-view observations as metric evidence.",
        )

        stages += PlannedStage(
            stage = Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
            executionClass = ExecutionClass.DETERMINISTIC,
            enabled = input.temporalConfidence < 0.94 ||
                multiViewReady,
            priority = 94,
            budget =
                (profile.targetCaptureFps * 2)
                    .coerceAtLeast(60),
            expectedGain =
                (1.0 - input.temporalConfidence) * 0.70,
            prerequisites = listOf(Stage.SESSION_INTEGRITY),
            rationale =
                "Re-evaluate exposure-center timing, clock offsets, frame pairing, and rolling-shutter timing before fusion.",
        )

        stages += PlannedStage(
            stage = Stage.MULTI_VIEW_BUNDLE_ADJUSTMENT,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled = multiViewReady,
            priority = 92,
            budget = profile.bundleAdjustmentIterations,
            expectedGain =
                (
                    1.0 -
                        minOf(
                            input.reconstructionConfidence,
                            input.calibrationConfidence,
                        )
                    ) * 0.85,
            prerequisites = listOf(
                Stage.CAMERA_GRAPH_REFINEMENT,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
            ),
            rationale =
                "Optimize cross-view residuals jointly rather than trusting frame-local triangulation alone.",
        )

        stages += PlannedStage(
            stage = Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled = input.reconstructionConfidence >= 0.35,
            priority = 90,
            budget =
                (
                    profile.bundleAdjustmentIterations *
                        2
                    ).coerceIn(8, 200),
            expectedGain =
                (1.0 - input.temporalConfidence) * 0.52,
            prerequisites = listOf(
                Stage.SESSION_INTEGRITY,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
            ),
            rationale =
                "Refine joint trajectories with temporal continuity, anatomical constraints, and measured residual confidence.",
        )

        stages += PlannedStage(
            stage = Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                input.reconstructionConfidence >= 0.30 &&
                    (
                        !input.physicsConstraintAccepted ||
                            input.physicsConsistencyScore < 0.94
                        ),
            priority = 91,
            budget =
                (
                    profile.bundleAdjustmentIterations *
                        (
                            1.0 +
                                (
                                    1.0 -
                                        input.physicsConsistencyScore
                                    ).coerceIn(0.0, 1.0) * 1.4
                            )
                    ).toInt()
                    .coerceIn(8, 256),
            expectedGain =
                (
                    1.0 -
                        input.physicsConsistencyScore
                    ).coerceIn(0.0, 1.0) * 0.78,
            prerequisites = listOf(
                Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
            ),
            rationale =
                "Re-run constrained skeletal refinement with metric bone lengths, dynamics envelopes, bilateral consistency, and support-contact evidence before downstream identity and surface passes.",
        )

        stages += PlannedStage(
            stage = Stage.GLOBAL_4D_CONSISTENCY_REFINEMENT,
            executionClass = ExecutionClass.DETERMINISTIC,
            enabled =
                input.reconstructionConfidence >= 0.28 &&
                    (
                        !input.global4DAccepted ||
                            input.global4DScore < 0.94 ||
                            input.global4DPersistentFailure ||
                            input.global4DRootMotionConsistency < 0.82 ||
                            input.global4DSceneConsistency < 0.78
                        ),
            priority = 92,
            budget =
                (
                    profile.bundleAdjustmentIterations *
                        (
                            1.0 +
                                (1.0 - input.global4DScore)
                                    .coerceIn(0.0, 1.0) * 1.5 +
                                if (input.global4DPersistentFailure) {
                                    0.55
                                } else {
                                    0.0
                                }
                        )
                    ).toInt()
                    .coerceIn(12, 280),
            expectedGain =
                (
                    1.0 -
                        (
                            input.global4DScore * 0.58 +
                                input.global4DRootMotionConsistency * 0.24 +
                                input.global4DSceneConsistency * 0.18
                            )
                    ).coerceIn(0.0, 1.0) * 0.90,
            prerequisites = listOf(
                Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
                Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
            ),
            rationale =
                "Reconcile human root motion, camera alignment, rig trust, physics consistency, and scene support evidence before identity, surface, or novel-view refinement.",
        )

        stages += PlannedStage(
            stage = Stage.FACE_EYE_EVIDENCE_REFINEMENT,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                input.faceEyeFusionConfidence < 0.94 ||
                    !input.faceEyeFusionAccepted,
            priority = 88,
            budget =
                (
                    when (profile.qualityTier) {
                        CaptureQualityTier.BALANCED -> 12
                        CaptureQualityTier.HIGH -> 20
                        CaptureQualityTier.ULTRA -> 36
                        CaptureQualityTier.REFERENCE -> 64
                    } *
                        (
                            1.0 +
                                (
                                    1.0 -
                                        input.faceEyeFusionConfidence
                                    ).coerceIn(0.0, 1.0) *
                                    1.35
                            )
                    ).toInt()
                    .coerceIn(12, 96),
            expectedGain =
                (
                    1.0 -
                        input.faceEyeFusionConfidence
                    ).coerceIn(0.0, 1.0) * 0.82,
            prerequisites = listOf(
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
                Stage.CROSS_MODAL_IDENTITY_REFINEMENT,
            ),
            rationale =
                "Replay face geometry, binocular gaze triangulation, uncertainty calibration, and temporal consistency with the same reliability gates used during live capture.",
        )

        stages += PlannedStage(
            stage = Stage.FACE_EYE_CALIBRATION_DRIFT_REFINEMENT,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                input.faceEyeCalibrationDriftScore >= 0.42 ||
                    !input.faceEyeFusionAccepted,
            priority = 90,
            budget =
                (
                    when (profile.qualityTier) {
                        CaptureQualityTier.BALANCED -> 12
                        CaptureQualityTier.HIGH -> 20
                        CaptureQualityTier.ULTRA -> 36
                        CaptureQualityTier.REFERENCE -> 64
                    } *
                        (
                            1.0 +
                                input.faceEyeCalibrationDriftScore * 1.20
                            )
                    ).toInt()
                    .coerceIn(12, 128),
            expectedGain =
                (
                    input.faceEyeCalibrationDriftScore
                ).coerceIn(0.0, 1.0) * 0.84,
            prerequisites = listOf(
                Stage.SESSION_INTEGRITY,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
                Stage.FACE_EYE_EVIDENCE_REFINEMENT,
            ),
            rationale =
                "Re-estimate face/eye camera scale, binocular geometry, timestamp alignment, and IPD consistency when slow calibration drift is persistent, before downstream identity and surface refinement.",
        )

        stages += PlannedStage(
            stage = Stage.CROSS_MODAL_IDENTITY_REFINEMENT,
            executionClass = ExecutionClass.DETERMINISTIC,
            enabled =
                input.reconstructionConfidence >= 0.30 &&
                    input.temporalConfidence >= 0.30,
            priority = 89,
            budget =
                (
                    profile.maxSubjects *
                        profile.targetCaptureFps *
                        2
                    ).coerceIn(30, 512),
            expectedGain =
                (
                    1.0 -
                        (
                            input.temporalConfidence * 0.55 +
                                input.faceConfidence * 0.25 +
                                input.handConfidence * 0.20
                            )
                    ).coerceIn(0.0, 1.0) * 0.66,
            prerequisites = listOf(
                Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
                Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
                Stage.GLOBAL_4D_CONSISTENCY_REFINEMENT,
                Stage.FACE_EYE_CALIBRATION_DRIFT_REFINEMENT,
            ),
            rationale =
                "Fuse body, face, hand, and eye continuity after physical and calibration-drift refinement so identity switches do not fragment the recovered trajectory.",
        )

        stages += PlannedStage(
            stage = Stage.DYNAMIC_STATIC_SCENE_DECOMPOSITION,
            executionClass = ExecutionClass.DETERMINISTIC,
            enabled =
                input.connectedCameraCount >= 1 &&
                    input.reconstructionConfidence >= 0.28,
            priority = 88,
            budget =
                (
                    profile.targetCaptureFps *
                        profile.maxSubjects *
                        4 *
                        (
                            1.0 +
                                (
                                    input.meanNormalChangeDegrees / 45.0
                                ).coerceIn(0.0, 1.0) * 0.75
                            )
                    ).toInt().coerceIn(60, 1536),
            expectedGain =
                (
                    (
                        1.0 -
                            minOf(
                                input.reconstructionConfidence,
                                input.calibrationConfidence,
                            )
                        ) * 0.64 +
                        (
                            input.meanNormalChangeDegrees / 45.0
                        ).coerceIn(0.0, 1.0) * 0.24
                    ).coerceIn(0.0, 1.0),
            prerequisites = listOf(
                Stage.SESSION_INTEGRITY,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
                Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
            ),
            rationale =
                "Separate persistent static geometry from dynamic human observations using physically consistent trajectories and surface-normal deformation evidence before camera refinement and final TSDF reintegration.",
        )

        stages += PlannedStage(
            stage = Stage.FACE_REGION_DENSIFICATION,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                profile.faceDetailPolicy !=
                    com.nivukx.nox.core.model.FaceDetailPolicy.DISABLED &&
                    input.faceConfidence >= 0.35,
            priority = 87,
            budget =
                when (profile.qualityTier) {
                    CaptureQualityTier.BALANCED -> 12
                    CaptureQualityTier.HIGH -> 20
                    CaptureQualityTier.ULTRA -> 36
                    CaptureQualityTier.REFERENCE -> 64
                },
            expectedGain =
                (1.0 - input.faceConfidence) * 0.74,
            prerequisites = listOf(
                Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
                Stage.CROSS_MODAL_IDENTITY_REFINEMENT,
                Stage.FACE_EYE_EVIDENCE_REFINEMENT,
                Stage.FACE_EYE_CALIBRATION_DRIFT_REFINEMENT,
            ),
            rationale =
                "Allocate additional geometric capacity to face landmarks only after face/eye calibration drift has been bounded.",
        )

        stages += PlannedStage(
            stage = Stage.HAND_REGION_DENSIFICATION,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                profile.handTrackingPolicy !=
                    com.nivukx.nox.core.model.HandTrackingPolicy.NONE &&
                    input.handConfidence >= 0.30,
            priority = 86,
            budget =
                when (profile.qualityTier) {
                    CaptureQualityTier.BALANCED -> 10
                    CaptureQualityTier.HIGH -> 18
                    CaptureQualityTier.ULTRA -> 32
                    CaptureQualityTier.REFERENCE -> 56
                },
            expectedGain =
                (1.0 - input.handConfidence) * 0.80,
            prerequisites = listOf(
                Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
                Stage.CROSS_MODAL_IDENTITY_REFINEMENT,
            ),
            rationale =
                "Preserve high-frequency finger geometry after actor identity continuity has been stabilized across modalities.",
        )

        stages += PlannedStage(
            stage = Stage.SURFACE_TSDF_REINTEGRATION,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                input.denseSurfaceConfidence >= 0.25 &&
                    input.surfaceNormalObservability >= 0.32 &&
                    input.surfaceHighUncertaintyRatio <= 0.78,
            priority = 84,
            budget =
                when (profile.tsdfResolutionClass) {
                    "REFERENCE" -> 8
                    "ULTRA" -> 5
                    "HIGH" -> 3
                    else -> 2
                },
            expectedGain =
                (1.0 - input.denseSurfaceConfidence) * 0.72,
            prerequisites = listOf(
                Stage.SESSION_INTEGRITY,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
                Stage.DYNAMIC_STATIC_SCENE_DECOMPOSITION,
                Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
            ),
            rationale =
                "Reintegrate calibrated depth with dynamic/static weights, physically consistent subject trajectories, and the final camera trajectory instead of treating real-time TSDF as the final mesh.",
        )

        stages += PlannedStage(
            stage = Stage.CANONICAL_SURFACE_TEMPORAL_PRIOR,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                input.denseSurfaceConfidence >= 0.28 &&
                    (
                        !input.canonicalSurfaceAccepted ||
                            input.canonicalSurfaceConfidence < 0.94 ||
                            input.canonicalCorrespondenceCoverage < 0.90 ||
                            input.canonicalEdgeStretchScore < 0.86 ||
                            input.canonicalTemporalConsistency < 0.92
                        ),
            priority = 83,
            budget =
                (
                    when (profile.qualityTier) {
                        CaptureQualityTier.BALANCED -> 16
                        CaptureQualityTier.HIGH -> 28
                        CaptureQualityTier.ULTRA -> 48
                        CaptureQualityTier.REFERENCE -> 72
                    } *
                        (
                            1.0 +
                                (
                                    1.0 -
                                        input.canonicalSurfaceConfidence
                                    ).coerceIn(0.0, 1.0) * 1.25
                            )
                    ).toInt()
                    .coerceIn(16, 112),
            expectedGain =
                (
                    (
                        1.0 -
                            input.canonicalSurfaceConfidence
                        ).coerceIn(0.0, 1.0) * 0.62 +
                    (
                        1.0 -
                            input.canonicalTemporalConsistency
                        ).coerceIn(0.0, 1.0) * 0.70
                ).coerceIn(0.0, 1.0) * 0.86,
            prerequisites = listOf(
                Stage.SURFACE_TSDF_REINTEGRATION,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
            ),
            rationale =
                "Validate frame-to-frame canonical temporal deformation with bounded surface correspondence, local edge stretch, normal agreement, and topology support before propagating deformation into later frames.",
        )

        stages += PlannedStage(
            stage = Stage.DYNAMIC_SURFACE_EVOLUTION,
            executionClass = ExecutionClass.GEOMETRIC_OPTIMIZATION,
            enabled =
                input.denseSurfaceConfidence >= 0.35 &&
                    input.surfaceNormalObservability >= 0.38 &&
                    input.surfaceHighUncertaintyRatio <= 0.72,
            priority = 82,
            budget =
                (
                    profile.denseMeshHz *
                        4
                    ).coerceAtLeast(16),
            expectedGain =
                (1.0 - input.temporalConfidence) * 0.58,
            prerequisites = listOf(
                Stage.SURFACE_TSDF_REINTEGRATION,
                Stage.CANONICAL_SURFACE_TEMPORAL_PRIOR,
                Stage.TEMPORAL_ALIGNMENT_REFINEMENT,
            ),
            rationale =
                "Model persistent surface evolution only after canonical deformation consistency validates the previous refined surface.",
        )

        stages += PlannedStage(
            stage = Stage.REGION_AWARE_NEURAL_REFINE,
            executionClass = ExecutionClass.NEURAL_OPTIONAL,
            enabled =
                profile.qualityTier >= CaptureQualityTier.ULTRA &&
                    overall >= 0.58,
            priority = 74,
            budget =
                when (profile.qualityTier) {
                    CaptureQualityTier.ULTRA -> 32
                    CaptureQualityTier.REFERENCE -> 96
                    else -> 0
                },
            expectedGain =
                (
                    1.0 -
                        minOf(
                            input.faceConfidence,
                            input.handConfidence,
                            input.denseSurfaceConfidence,
                        )
                    ) * 0.62,
            prerequisites = listOf(
                Stage.FACE_REGION_DENSIFICATION,
                Stage.HAND_REGION_DENSIFICATION,
                Stage.CROSS_MODAL_IDENTITY_REFINEMENT,
                Stage.FACE_EYE_EVIDENCE_REFINEMENT,
                Stage.DYNAMIC_STATIC_SCENE_DECOMPOSITION,
                Stage.DYNAMIC_SURFACE_EVOLUTION,
                Stage.CANONICAL_SURFACE_TEMPORAL_PRIOR,
            ),
            rationale =
                "Optional region-aware neural refinement for high-frequency face, hand, and surface detail after canonical deformation consistency is established. External weights must be explicitly provisioned.",
        )

        stages += PlannedStage(
            stage = Stage.NOVEL_VIEW_REFINEMENT,
            executionClass = ExecutionClass.RENDERING,
            enabled =
                multiViewReady &&
                    profile.qualityTier >=
                        CaptureQualityTier.ULTRA,
            priority = 66,
            budget =
                (
                    profile.maxSubjects *
                        profile.multiViewConsensusThreshold *
                        8
                    ).coerceAtLeast(16),
            expectedGain =
                if (multiViewReady) {
                    0.28
                } else {
                    0.0
                },
            prerequisites = listOf(
                Stage.MULTI_VIEW_BUNDLE_ADJUSTMENT,
                Stage.DYNAMIC_SURFACE_EVOLUTION,
            ),
            rationale =
                "Generate controlled evaluation views from the refined metric reconstruction instead of using synthetic views as measurement evidence.",
        )

        stages += PlannedStage(
            stage = Stage.EXPORT_QUALITY_ASSURANCE,
            executionClass = ExecutionClass.VALIDATION,
            enabled = overall >= 0.42,
            priority = 100,
            budget = 1,
            expectedGain = 0.0,
            prerequisites = listOf(
                Stage.KINEMATIC_TRAJECTORY_REFINEMENT,
                Stage.SURFACE_TSDF_REINTEGRATION,
                Stage.CANONICAL_SURFACE_TEMPORAL_PRIOR,
                Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT,
            ),
            rationale =
                "Validate residuals, uncertainty, topology, physical consistency, provenance, and checksum integrity before export.",
        )

        val warnings =
            buildList {
                if (
                    input.surfaceNormalObservability < 0.55 ||
                    input.surfaceHighUncertaintyRatio > 0.45 ||
                    input.cameraInformationGain < 0.42
                ) {
                    add(
                        "offline-surface-evidence-limited",
                    )
                }
                if (!input.sessionIntegrityValid) {
                    add(
                        "Session integrity is invalid; refinement must remain blocked.",
                    )
                }

                if (!multiViewReady &&
                    profile.multiViewRequired
                ) {
                    add(
                        "Profile requests multi-view refinement but the captured rig does not meet the metric prerequisites.",
                    )
                }

                if (input.denseSurfaceConfidence < 0.35) {
                    add(
                        "Dense surface evidence is weak; prefer skeleton and camera-graph refinement before surface refinement.",
                    )
                }

                if (profile.qualityTier == CaptureQualityTier.REFERENCE &&
                    overall < 0.70
                ) {
                    add(
                        "Reference profile selected with insufficient aggregate evidence for reference export.",
                    )
                }
            }

        val recommendations =
            buildList {
                if (input.calibrationConfidence < 0.75) {
                    add(
                        "Refine calibration before spending budget on neural detail.",
                    )
                }

                if (input.synchronizationConfidence < 0.75) {
                    add(
                        "Recompute cross-device timing before bundle adjustment.",
                    )
                }

                if (input.faceConfidence < 0.55) {
                    add(
                        "Prioritize facial capture before final avatar bake.",
                    )
                }

                if (input.handConfidence < 0.55) {
                    add(
                        "Prioritize hand geometry before final surface bake.",
                    )
                }

                if (input.temporalConfidence < 0.72) {
                    add(
                        "Refine temporal alignment and trajectory optimization before mesh deformation fitting.",
                    )
                }

                if (input.temporalConfidence < 0.60) {
                    add(
                        "Run cross-modal identity refinement before trusting long occlusion gaps or actor reassociation.",
                    )
                }

                if (input.reconstructionConfidence < 0.62) {
                    add(
                        "Run dynamic/static decomposition before camera and surface refinement so moving subjects do not contaminate the static solution.",
                    )
                }

                if (
                    !input.physicsConstraintAccepted ||
                    input.physicsConsistencyScore < 0.62
                ) {
                    add(
                        "Refine physically constrained trajectories before trusting metric export, dense surfaces, or long occlusion gaps.",
                    )
                }
                if (
                    input.faceEyeCalibrationDriftScore >= 0.62
                ) {
                    add(
                        "Persistent face/eye calibration drift detected. Re-run binocular geometry and IPD calibration before trusting long metric facial or gaze segments.",
                    )
                }

                if (
                    !input.faceEyeFusionAccepted ||
                    input.faceEyeFusionConfidence < 0.62
                ) {
                    add(
                        "Refine face/eye evidence before trusting gaze trajectories, facial detail, or metric export.",
                    )
                }

                if (
                    !input.global4DAccepted ||
                    input.global4DScore < 0.94 ||
                    input.global4DPersistentFailure
                ) {
                    add(
                        "Refine global human-camera-scene consistency before final metric export or long temporal propagation.",
                    )
                }

                if (
                    !input.canonicalSurfaceAccepted ||
                    input.canonicalSurfaceConfidence < 0.62
                ) {
                    add(
                        "Validate canonical surface deformation before trusting long temporal mesh evolution or final surface export.",
                    )
                }

                if (overall >= 0.82) {
                    add(
                        "Capture supports the full configured refinement envelope.",
                    )
                }
            }

        return Plan(
            profileId = profile.id,
            overallConfidence = overall,
            stages = stages.sortedByDescending { it.priority },
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
        )
    }
}
