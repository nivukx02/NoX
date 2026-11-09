package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ProductionCaptureProfile
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndOfflineRefinementPlannerTest {

    @Test
    fun referenceProfileProducesFullRefinementGraph() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.96,
                    synchronizationConfidence = 0.94,
                    reconstructionConfidence = 0.91,
                    temporalConfidence = 0.92,
                    denseSurfaceConfidence = 0.88,
                    faceConfidence = 0.84,
                    handConfidence = 0.82,
                    runtimeQualityScore = 0.90,
                    sessionIntegrityValid = true,
                ),
            )

        assertTrue(plan.overallConfidence > 0.80)
        assertTrue(plan.executableStages.any {
            it.stage ==
                HighEndOfflineRefinementPlanner.Stage.MULTI_VIEW_BUNDLE_ADJUSTMENT
        })
        assertTrue(plan.executableStages.any {
            it.stage ==
                HighEndOfflineRefinementPlanner.Stage.DYNAMIC_SURFACE_EVOLUTION
        })
        assertTrue(plan.executableStages.any {
            it.stage ==
                HighEndOfflineRefinementPlanner.Stage.CROSS_MODAL_IDENTITY_REFINEMENT
        })
        assertTrue(plan.executableStages.any {
            it.stage ==
                HighEndOfflineRefinementPlanner.Stage.DYNAMIC_STATIC_SCENE_DECOMPOSITION
        })
        assertTrue(plan.executableStages.any {
            it.stage ==
                HighEndOfflineRefinementPlanner.Stage.REGION_AWARE_NEURAL_REFINE
        })
    }

    @Test
    fun persistentFaceEyeCalibrationDriftAddsExplicitRecalibrationStage() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.92,
                    synchronizationConfidence = 0.93,
                    reconstructionConfidence = 0.88,
                    temporalConfidence = 0.90,
                    denseSurfaceConfidence = 0.82,
                    faceConfidence = 0.84,
                    handConfidence = 0.80,
                    runtimeQualityScore = 0.90,
                    faceEyeFusionConfidence = 0.74,
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.78,
                    physicsConsistencyScore = 0.90,
                    physicsConstraintAccepted = true,
                    sessionIntegrityValid = true,
                ),
            )

        val driftStage =
            plan.executableStages.first {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.FACE_EYE_CALIBRATION_DRIFT_REFINEMENT
            }

        assertTrue(driftStage.budget >= 12)
        assertTrue(driftStage.expectedGain > 0.20)
        assertTrue(
            plan.recommendations.any {
                it.contains("calibration drift") ||
                    it.contains("recalibration")
            },
        )
    }

    @Test
    fun degradedGlobal4DAddsExplicitConsistencyStage() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.94,
                    synchronizationConfidence = 0.94,
                    reconstructionConfidence = 0.88,
                    temporalConfidence = 0.90,
                    denseSurfaceConfidence = 0.82,
                    faceConfidence = 0.82,
                    handConfidence = 0.80,
                    runtimeQualityScore = 0.90,
                    global4DScore = 0.34,
                    global4DAccepted = false,
                    global4DPersistentFailure = true,
                    global4DRootMotionConsistency = 0.28,
                    global4DSceneConsistency = 0.30,
                    physicsConsistencyScore = 0.92,
                    physicsConstraintAccepted = true,
                    sessionIntegrityValid = true,
                ),
            )

        val stage =
            plan.executableStages.first {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.GLOBAL_4D_CONSISTENCY_REFINEMENT
            }

        assertTrue(stage.budget >= 12)
        assertTrue(stage.expectedGain > 0.40)
        assertTrue(
            plan.recommendations.any {
                it.contains("global human-camera-scene consistency")
            },
        )
    }

    @Test
    fun degradedPhysicsAddsExplicitRecoveryStage() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.94,
                    synchronizationConfidence = 0.94,
                    reconstructionConfidence = 0.88,
                    temporalConfidence = 0.90,
                    denseSurfaceConfidence = 0.82,
                    faceConfidence = 0.82,
                    handConfidence = 0.80,
                    runtimeQualityScore = 0.90,
                    physicsConsistencyScore = 0.38,
                    physicsConstraintAccepted = false,
                    physicsMeanBoneResidualMeters = 0.072,
                    sessionIntegrityValid = true,
                ),
            )

        val physicsStage =
            plan.executableStages.first {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.PHYSICS_CONSTRAINED_TRAJECTORY_REFINEMENT
            }

        assertTrue(physicsStage.budget >= 8)
        assertTrue(physicsStage.expectedGain > 0.40)
        assertTrue(
            plan.recommendations.any {
                it.contains("physically constrained trajectories")
            },
        )
    }

    @Test
    fun rejectedFaceEyeEvidenceAddsOfflineRecoveryStage() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.94,
                    synchronizationConfidence = 0.94,
                    reconstructionConfidence = 0.90,
                    temporalConfidence = 0.91,
                    denseSurfaceConfidence = 0.84,
                    faceConfidence = 0.48,
                    handConfidence = 0.82,
                    runtimeQualityScore = 0.91,
                    faceEyeFusionConfidence = 0.34,
                    faceEyeFusionAccepted = false,
                    physicsConsistencyScore = 0.96,
                    physicsConstraintAccepted = true,
                    sessionIntegrityValid = true,
                ),
            )

        val stage =
            plan.executableStages.first {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.FACE_EYE_EVIDENCE_REFINEMENT
            }

        assertTrue(stage.budget >= 12)
        assertTrue(stage.expectedGain > 0.30)
        assertTrue(
            plan.recommendations.any {
                it.contains("face/eye")
            },
        )
    }

    @Test
    fun weakCanonicalSurfacePriorAddsDeformationConsistencyStage() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.96,
                    synchronizationConfidence = 0.95,
                    reconstructionConfidence = 0.91,
                    temporalConfidence = 0.93,
                    denseSurfaceConfidence = 0.88,
                    faceConfidence = 0.88,
                    handConfidence = 0.86,
                    runtimeQualityScore = 0.93,
                    faceEyeFusionConfidence = 0.94,
                    faceEyeFusionAccepted = true,
                    physicsConsistencyScore = 0.95,
                    physicsConstraintAccepted = true,
                    canonicalSurfaceConfidence = 0.42,
                    canonicalSurfaceAccepted = false,
                    canonicalCorrespondenceCoverage = 0.54,
                    canonicalEdgeStretchScore = 0.36,
                    sessionIntegrityValid = true,
                ),
            )

        val stage =
            plan.executableStages.first {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage
                        .CANONICAL_SURFACE_TEMPORAL_PRIOR
            }

        assertTrue(stage.budget >= 16)
        assertTrue(stage.expectedGain > 0.30)
        assertTrue(
            stage.prerequisites.contains(
                HighEndOfflineRefinementPlanner.Stage
                    .SURFACE_TSDF_REINTEGRATION,
            ),
        )
    }

    @Test
    fun poorCanonicalTemporalConsistencyEnablesTemporalPriorStage() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 8,
                    calibrationConfidence = 0.96,
                    synchronizationConfidence = 0.96,
                    reconstructionConfidence = 0.92,
                    temporalConfidence = 0.92,
                    denseSurfaceConfidence = 0.91,
                    faceConfidence = 0.90,
                    handConfidence = 0.88,
                    runtimeQualityScore = 0.92,
                    faceEyeFusionConfidence = 0.90,
                    physicsConsistencyScore = 0.94,
                    physicsConstraintAccepted = true,
                    canonicalSurfaceConfidence = 0.97,
                    canonicalSurfaceAccepted = true,
                    canonicalCorrespondenceCoverage = 0.96,
                    canonicalEdgeStretchScore = 0.94,
                    canonicalTemporalConsistency = 0.62,
                    sessionIntegrityValid = true,
                ),
            )

        val stage =
            plan.executableStages.first {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.CANONICAL_SURFACE_TEMPORAL_PRIOR
            }

        assertTrue(stage.enabled)
        assertTrue(stage.expectedGain > 0.20)
        assertTrue(
            stage.rationale.contains("frame-to-frame"),
        )
    }

    @Test
    fun weakSurfaceEvidenceSuppressesSurfaceEvolutionStages() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.REFERENCE_BAKE,
                    connectedCameraCount = 6,
                    calibrationConfidence = 0.94,
                    synchronizationConfidence = 0.94,
                    reconstructionConfidence = 0.88,
                    temporalConfidence = 0.90,
                    denseSurfaceConfidence = 0.82,
                    surfaceNormalObservability = 0.28,
                    surfaceHighUncertaintyRatio = 0.82,
                    cameraInformationGain = 0.26,
                    faceConfidence = 0.84,
                    handConfidence = 0.82,
                    runtimeQualityScore = 0.92,
                    sessionIntegrityValid = true,
                ),
            )

        val tsdf =
            plan.executableStages.any {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.SURFACE_TSDF_REINTEGRATION
            }
        val evolution =
            plan.executableStages.any {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.DYNAMIC_SURFACE_EVOLUTION
            }

        assertTrue(!tsdf || !evolution)
        assertTrue(
            plan.warnings.any {
                it.contains("surface")
            } ||
                plan.recommendations.any {
                    it.contains("surface")
                },
        )
    }

    @Test
    fun invalidIntegrityBlocksRefinementEntryPoint() {
        val plan =
            HighEndOfflineRefinementPlanner().createPlan(
                HighEndOfflineRefinementPlanner.Input(
                    profile = ProductionCaptureProfile.LIVE_ACTION,
                    connectedCameraCount = 3,
                    calibrationConfidence = 0.80,
                    synchronizationConfidence = 0.80,
                    reconstructionConfidence = 0.80,
                    temporalConfidence = 0.80,
                    denseSurfaceConfidence = 0.80,
                    faceConfidence = 0.80,
                    handConfidence = 0.80,
                    runtimeQualityScore = 0.80,
                    sessionIntegrityValid = false,
                ),
            )

        assertTrue(
            plan.warnings.any {
                it.contains("integrity", ignoreCase = true)
            },
        )
        assertTrue(
            !plan.executableStages.any {
                it.stage ==
                    HighEndOfflineRefinementPlanner.Stage.SESSION_INTEGRITY
            },
        )
    }
}
