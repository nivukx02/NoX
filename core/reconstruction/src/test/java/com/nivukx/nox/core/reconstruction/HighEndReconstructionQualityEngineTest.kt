package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ProductionCaptureProfile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndReconstructionQualityEngineTest {

    private fun cleanInput() =
        HighEndReconstructionQualityEngine.Input(
            timestampNanos = 1_000_000_000L,
            profile = ProductionCaptureProfile.LIVE_ACTION,
            reconstructionConfidence = 0.97,
            reprojectionErrorPx = 0.8,
            meanUncertaintyMeters = 0.012,
            geometricObservability = 0.96,
            temporalConfidence = 0.96,
            synchronizationConfidence = 0.98,
            calibrationConfidence = 0.98,
            trackingConfidence = 0.97,
            identitySwapRisk = 0.01,
            occlusionRisk = 0.02,
            cameraMotionRisk = 0.02,
            faceEyeConfidence = 0.96,
            handConfidence = 0.94,
            surfaceConfidence = 0.92,
            thermalScale = 0.92,
            memoryScale = 0.94,
            runtimeQualityScale = 0.94,
            runtimeInferenceScale = 0.96,
            gpuDispatchMillis = 5.0,
            connectedCameraCount = 3,
            observedSubjectCount = 1,
        )

    @Test
    fun elevatedOpticalBlurBecomesExplicitBottleneck() {
        val engine = HighEndReconstructionQualityEngine()
        val result =
            engine.evaluate(
                cleanInput().copy(
                    timestampNanos = 2L,
                    motionBlurRisk = 0.76,
                ),
            )

        assertTrue(
            result.warnings.contains(
                "optical-motion-blur-risk-elevated",
            ) ||
                result.warnings.contains(
                    "optical-motion-blur-risk-high",
                ),
        )
        assertTrue(
            result.componentScores["opticalCapture"] ?: 1.0 < 0.30,
        )
        assertTrue(
            result.bottleneck ==
                HighEndReconstructionQualityEngine.Bottleneck.OPTICAL_CAPTURE ||
                (result.componentScores["opticalCapture"] ?: 1.0) < 0.30,
        )
    }

    @Test
    fun severeOpticalBlurHardRejectsProductionOutput() {
        val engine = HighEndReconstructionQualityEngine()
        val result =
            engine.evaluate(
                cleanInput().copy(
                    timestampNanos = 4L,
                    motionBlurRisk = 0.96,
                ),
            )

        assertTrue(result.hardFailure)
        assertTrue(!result.outputAllowed)
        assertTrue(
            result.warnings.contains(
                "optical-motion-blur-hard-limit",
            ),
        )
    }

    @Test
    fun cleanReferenceSignalsReachAcceptPolicy() {
        val engine = HighEndReconstructionQualityEngine()

        val decision = engine.evaluate(cleanInput())

        assertEquals(
            HighEndReconstructionQualityEngine.Policy.ACCEPT,
            decision.policy,
        )
        assertTrue(decision.outputAllowed)
        assertTrue(decision.score > 0.70)
        assertTrue(decision.budget.tsdfStride <= 2)
        assertTrue(decision.budget.meshVertexBudget > 100_000)
    }

    @Test
    fun synchronizationFailureForcesHoldOrWorse() {
        val engine = HighEndReconstructionQualityEngine()

        val degraded = cleanInput().copy(
            synchronizationConfidence = 0.25,
            calibrationConfidence = 0.45,
            meanUncertaintyMeters = 0.09,
        )

        val decision = engine.evaluate(degraded)

        assertTrue(
            decision.policy ==
                HighEndReconstructionQualityEngine.Policy.HOLD ||
                decision.policy ==
                HighEndReconstructionQualityEngine.Policy.DEGRADE ||
                decision.policy ==
                HighEndReconstructionQualityEngine.Policy.REJECT,
        )
        assertTrue(
            decision.bottleneck ==
                HighEndReconstructionQualityEngine.Bottleneck.SYNCHRONIZATION ||
                decision.bottleneck ==
                HighEndReconstructionQualityEngine.Bottleneck.CALIBRATION,
        )
        assertFalse(
            decision.budget.allowReferenceOptimization,
        )
    }

    @Test
    fun hardFailureRejectsOutput() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    hardCaptureFailure = true,
                ),
            )

        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            decision.policy,
        )
        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertEquals(
            4,
            decision.budget.tsdfStride,
        )
    }



    @Test
    fun adaptiveDetailHoldBecomesHardProductionGate() {
        val engine = HighEndReconstructionQualityEngine()
        val decision = engine.evaluate(
            cleanInput().copy(
                adaptiveDetailScore = 0.22,
                adaptiveDetailHold = true,
            ),
        )
        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertTrue(
            decision.warnings.any {
                it.contains("adaptive-reconstruction-detail-hold")
            },
        )
        assertTrue(
            decision.componentScores["adaptiveDetail"]!! < 0.30,
        )
    }

    @Test
    fun captureAuthoritiesInfluenceQualityWithoutMandatoryFailure() {
        val engine = HighEndReconstructionQualityEngine()
        val healthy = engine.evaluate(
            cleanInput().copy(
                thermalDriftConfidence = 0.98,
                concurrentRigScore = 0.96,
                faceEyeMetricEvidence = 0.97,
            ),
        )

        engine.reset()

        val degraded = engine.evaluate(
            cleanInput().copy(
                thermalDriftConfidence = 0.44,
                concurrentRigScore = 0.48,
                faceEyeMetricEvidence = 0.42,
            ),
        )

        assertTrue(degraded.score < healthy.score)
        assertTrue(
            degraded.componentScores["thermalDrift"]!! <
                healthy.componentScores["thermalDrift"]!!,
        )
        assertTrue(
            degraded.componentScores["concurrentRig"]!! <
                healthy.componentScores["concurrentRig"]!!,
        )
        assertTrue(
            degraded.componentScores["faceEyeMetric"]!! <
                healthy.componentScores["faceEyeMetric"]!!,
        )
    }

    @Test
    fun highMultiPersonPredictionPressureRejectsFreshMetricOutput() {
        val decision =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    observedSubjectCount = 20,
                    multiPersonPredictionRatio = 0.96,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertEquals(
            0.0,
            decision.componentScores["multiPersonPrediction"] ?: Double.NaN,
            1e-9,
        )
        assertTrue(
            decision.warnings.any {
                it.contains("multi-person-3d-prediction-pressure")
            },
        )
    }

    @Test
    fun moderateMultiPersonPredictionPressureLowersQualityWithoutImmediateReject() {
        val engine = HighEndReconstructionQualityEngine()

        val fresh =
            engine.evaluate(
                cleanInput().copy(
                    observedSubjectCount = 20,
                    multiPersonPredictionRatio = 0.05,
                ),
            )

        engine.reset()

        val predicted =
            engine.evaluate(
                cleanInput().copy(
                    observedSubjectCount = 20,
                    multiPersonPredictionRatio = 0.55,
                ),
            )

        assertTrue(
            predicted.score < fresh.score,
        )
        assertTrue(
            predicted.componentScores["multiPersonPrediction"]!! <
                fresh.componentScores["multiPersonPrediction"]!!,
        )
        assertTrue(
            predicted.warnings.any {
                it.contains("multi-person-3d-prediction-pressure")
            },
        )
    }

    @Test
    fun lowMultiPersonIdentityContinuityTriggersProductionHold() {
        val decision =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    observedSubjectCount = 12,
                    identityContinuity = 0.12,
                    identitySwapRisk = 0.20,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertTrue(
            decision.componentScores["identityContinuity"]!! < 0.20,
        )
    }

    @Test
    fun moderateMultiPersonIdentityContinuityLowersQualityWithoutHardReject() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    observedSubjectCount = 12,
                    identityContinuity = 0.96,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    observedSubjectCount = 12,
                    identityContinuity = 0.42,
                ),
            )

        assertTrue(
            degraded.score < healthy.score,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("identity continuity")
            },
        )
    }

    @Test
    fun calibratedUncertaintyBecomesAQualityComponent() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    calibratedUncertaintyConfidence = 0.94,
                    motionOutOfDistributionRisk = 0.02,
                    motionUncertaintyCoverage = 0.96,
                ),
            )

        engine.reset()

        val uncertain =
            engine.evaluate(
                cleanInput().copy(
                    calibratedUncertaintyConfidence = 0.38,
                    motionOutOfDistributionRisk = 0.72,
                    motionUncertaintyCoverage = 0.62,
                ),
            )

        assertTrue(
            uncertain.score < healthy.score,
        )
        assertTrue(
            uncertain.warnings.any {
                it.contains("motion-out-of-distribution-risk")
            },
        )
        assertTrue(
            uncertain.componentScores["uncertaintyCalibration"]!! <
                healthy.componentScores["uncertaintyCalibration"]!!,
        )
    }

    @Test
    fun extremeMotionOodRiskIsAHardFailure() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    motionOutOfDistributionRisk = 1.0,
                ),
            )

        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            decision.policy,
        )
        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
    }


    @Test
    fun highDynamicsPrioritizesInferenceAndHands() {
        val engine = HighEndReconstructionQualityEngine()

        val baseline =
            engine.evaluate(
                cleanInput().copy(
                    motionDynamicStress = 0.05,
                    motionDynamicsScore = 0.40,
                ),
            )

        engine.reset()

        val dynamic =
            engine.evaluate(
                cleanInput().copy(
                    motionDynamicStress = 0.94,
                    motionDynamicsScore = 0.96,
                ),
            )

        assertTrue(
            dynamic.budget.targetInferenceFps >=
                baseline.budget.targetInferenceFps,
        )
        assertTrue(
            dynamic.budget.handTrackingHz >=
                baseline.budget.handTrackingHz,
        )
        assertTrue(
            dynamic.budget.detectorIntervalFrames <=
                baseline.budget.detectorIntervalFrames,
        )
    }

    @Test
    fun global4DConsistencyBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    global4DScore = 0.96,
                    global4DAccepted = true,
                    global4DPersistentFailure = false,
                    global4DRootMotionConsistency = 0.95,
                    global4DSceneConsistency = 0.92,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    global4DScore = 0.34,
                    global4DAccepted = false,
                    global4DPersistentFailure = true,
                    global4DRootMotionConsistency = 0.28,
                    global4DSceneConsistency = 0.32,
                ),
            )

        assertTrue(
            degraded.componentScores["global4DConsistency"]!! <
                healthy.componentScores["global4DConsistency"]!!,
        )
        assertTrue(
            degraded.hardFailure ||
                degraded.warnings.any {
                    it.contains("global-4d")
                },
        )
        assertTrue(
            degraded.budget.meshVertexBudget <=
                healthy.budget.meshVertexBudget,
        )
    }

    @Test
    fun global4DDirectionalIntegrityChangesProductionQuality() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    global4DScore = 0.94,
                    global4DAccepted = true,
                    global4DDirectionalUncertaintyConsistency = 0.96,
                    global4DFaceEyeConsistency = 0.94,
                ),
            )

        engine.reset()

        val depthRisk =
            engine.evaluate(
                cleanInput().copy(
                    global4DScore = 0.94,
                    global4DAccepted = true,
                    global4DDirectionalUncertaintyConsistency = 0.34,
                    global4DFaceEyeConsistency = 0.94,
                ),
            )

        assertTrue(depthRisk.score < healthy.score)
        assertTrue(
            depthRisk.componentScores["global4DConsistency"]!! <
                healthy.componentScores["global4DConsistency"]!!,
        )
    }

    @Test
    fun global4DFaceEyeIntegrityChangesProductionQuality() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    global4DScore = 0.94,
                    global4DAccepted = true,
                    global4DDirectionalUncertaintyConsistency = 0.94,
                    global4DFaceEyeConsistency = 0.95,
                ),
            )

        engine.reset()

        val faceEyeRisk =
            engine.evaluate(
                cleanInput().copy(
                    global4DScore = 0.94,
                    global4DAccepted = true,
                    global4DDirectionalUncertaintyConsistency = 0.94,
                    global4DFaceEyeConsistency = 0.28,
                ),
            )

        assertTrue(faceEyeRisk.score < healthy.score)
        assertTrue(
            faceEyeRisk.componentScores["global4DConsistency"]!! <
                healthy.componentScores["global4DConsistency"]!!,
        )
    }

    @Test
    fun decliningFaceEyeReliabilityLowersQualityComponent() {
        val engine = HighEndReconstructionQualityEngine()

        val stable =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeReliabilityTrend = 0.02,
                    faceEyeFusionAccepted = true,
                    faceEyeConfidence = 0.92,
                ),
            )

        engine.reset()

        val declining =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeReliabilityTrend = -0.42,
                    faceEyeFusionAccepted = true,
                    faceEyeConfidence = 0.92,
                ),
            )

        assertTrue(
            declining.componentScores["faceEyeTrend"]!! <
                stable.componentScores["faceEyeTrend"]!!,
        )
        assertTrue(
            declining.warnings.any {
                it.contains("face-eye-reliability")
            },
        )
    }

    @Test
    fun severeFaceEyeReliabilityCollapseTriggersHardPolicy() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeReliabilityTrend = -0.82,
                    faceEyeFusionAccepted = true,
                    faceEyeConfidence = 0.86,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
    }

    @Test
    fun lowPhysicsConsistencyBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.92,
                    physicsConstraintAccepted = true,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.42,
                    physicsConstraintAccepted = true,
                ),
            )

        assertTrue(
            degraded.componentScores["physicsConsistency"]!! <
                clean.componentScores["physicsConsistency"]!!,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("physics-consistency")
            },
        )
        assertTrue(
            degraded.bottleneck ==
                HighEndReconstructionQualityEngine.Bottleneck.PHYSICS_CONSISTENCY ||
                degraded.score < clean.score,
        )
    }

    @Test
    fun directionalUncertaintyBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    directionalUncertaintyMajor95Meters = 0.03,
                    directionalUncertaintyDepthMeters = 0.02,
                    directionalUncertaintyAnisotropy = 1.6,
                    directionalUncertaintyConfidence = 0.96,
                    directionalUncertaintyDegradedJointCount = 0,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    directionalUncertaintyMajor95Meters = 0.32,
                    directionalUncertaintyDepthMeters = 0.29,
                    directionalUncertaintyAnisotropy = 8.5,
                    directionalUncertaintyConfidence = 0.34,
                    directionalUncertaintyDegradedJointCount = 7,
                ),
            )

        assertTrue(
            degraded.componentScores["directionalUncertainty"]!! <
                healthy.componentScores["directionalUncertainty"]!!,
        )
        assertTrue(degraded.score < healthy.score)
        assertTrue(
            degraded.warnings.any {
                it.contains("directional-uncertainty")
            },
        )
        assertTrue(
            degraded.bottleneck ==
                HighEndReconstructionQualityEngine.Bottleneck.DIRECTIONAL_UNCERTAINTY ||
                degraded.componentScores["directionalUncertainty"]!! < 0.65,
        )
    }

    @Test
    fun directionalAnisotropyHardFailsExtremeDepthInstability() {
        val decision =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    directionalUncertaintyMajor95Meters = 1.20,
                    directionalUncertaintyDepthMeters = 1.04,
                    directionalUncertaintyAnisotropy = 24.0,
                    directionalUncertaintyConfidence = 0.04,
                    directionalUncertaintyDegradedJointCount = 18,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertTrue(
            decision.componentScores["directionalUncertainty"]!! < 0.30,
        )
    }

    @Test
    fun directionalDepthRiskRedirectsComputeBudget() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    directionalUncertaintyMajor95Meters = 0.04,
                    directionalUncertaintyDepthMeters = 0.03,
                    directionalUncertaintyAnisotropy = 1.8,
                    directionalUncertaintyConfidence = 0.95,
                ),
            )

        engine.reset()

        val depthRisk =
            engine.evaluate(
                cleanInput().copy(
                    directionalUncertaintyMajor95Meters = 0.30,
                    directionalUncertaintyDepthMeters = 0.30,
                    directionalUncertaintyAnisotropy = 8.0,
                    directionalUncertaintyConfidence = 0.38,
                    directionalUncertaintyDegradedJointCount = 6,
                ),
            )

        assertTrue(
            depthRisk.budget.bundleAdjustmentIterations >=
                healthy.budget.bundleAdjustmentIterations,
        )
        assertTrue(
            depthRisk.budget.meshVertexBudget <=
                healthy.budget.meshVertexBudget,
        )
        assertTrue(
            depthRisk.budget.detectorIntervalFrames <=
                healthy.budget.detectorIntervalFrames,
        )
    }

    @Test
    fun lowGlobalRigTrustRejectsHighPrecisionProductionOutput() {
        val decision =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    globalRigTrust = 0.12,
                ),
            )

        assertTrue(decision.hardFailure)
        assertTrue(!decision.outputAllowed)
    }

    @Test
    fun moderateFaceEyeRejectionPreservesBodyOutputAndTriggersRecoveryBudget() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.86,
                    faceEyeFusionAccepted = false,
                ),
            )

        assertFalse(decision.hardFailure)
        assertTrue(decision.outputAllowed)
        assertTrue(
            decision.warnings.any {
                it.contains("face-eye-recovery-only-body-output-preserved")
            },
        )
        assertTrue(
            decision.budget.faceMeshHz <=
                cleanInput().profile.faceMeshHz,
        )
    }

    @Test
    fun severeFaceEyeRejectionStillHardFailsFacialEvidencePath() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.05,
                    faceEyeFusionAccepted = false,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
    }

    @Test
    fun rejectedPhysicsConsistencyIsHardFailure() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.88,
                    physicsConstraintAccepted = false,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            decision.policy,
        )
    }

    @Test
    fun elevatedFaceEyeOodLowersQualityWithoutImmediateHardFailure() {
        val engine = HighEndReconstructionQualityEngine()

        val stable =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeOutOfDistributionRisk = 0.05,
                    faceEyeFusionAccepted = true,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeOutOfDistributionRisk = 0.78,
                    faceEyeFusionAccepted = true,
                ),
            )

        assertTrue(
            degraded.componentScores["faceEyeOod"]!! <
                stable.componentScores["faceEyeOod"]!!,
        )
        assertTrue(degraded.score < stable.score)
        assertFalse(degraded.hardFailure)
        assertTrue(
            degraded.warnings.any {
                it.contains("face-eye-out-of-distribution")
            },
        )
    }

    @Test
    fun persistentFaceEyeCalibrationDriftRejectsProductionOutput() {
        val decision =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeOutOfDistributionRisk = 0.10,
                    faceEyeCalibrationDriftPersistent = true,
                    faceEyeCalibrationDriftScore = 0.66,
                    faceEyeDominantFailure = "IPD_CALIBRATION",
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertTrue(
            decision.warnings.any {
                it.contains("face-eye-calibration-drift")
            },
        )
    }

    @Test
    fun extremeFaceEyeOodStillHardFails() {
        val decision =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    faceEyeOutOfDistributionRisk = 0.995,
                    faceEyeFusionAccepted = true,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
    }

    @Test
    fun degradedFaceEyeFusionRedirectsComputeTowardFaceEyeRecovery() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.96,
                    faceEyeFusionAccepted = true,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.38,
                    faceEyeFusionAccepted = false,
                ),
            )

        assertTrue(
            degraded.budget.faceMeshHz >=
                healthy.budget.faceMeshHz,
        )
        assertTrue(
            degraded.budget.eyeTrackingHz >=
                healthy.budget.eyeTrackingHz,
        )
        assertTrue(
            degraded.budget.meshVertexBudget <=
                healthy.budget.meshVertexBudget,
        )
    }

    @Test
    fun persistentFaceEyeCalibrationDriftRedirectsRecoveryCompute() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.96,
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.02,
                    faceEyeCalibrationDriftPersistent = false,
                ),
            )

        engine.reset()

        val drifted =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.82,
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.76,
                    faceEyeCalibrationDriftPersistent = true,
                ),
            )

        assertTrue(
            drifted.budget.faceMeshHz >=
                healthy.budget.faceMeshHz,
        )
        assertTrue(
            drifted.budget.eyeTrackingHz >=
                healthy.budget.eyeTrackingHz,
        )
        assertTrue(
            drifted.budget.meshVertexBudget <=
                healthy.budget.meshVertexBudget,
        )
        assertFalse(
            drifted.budget.allowReferenceOptimization,
        )
    }

    @Test
    fun physicsAndRigTrustInfluenceQualityScoreNotOnlyHardGate() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.96,
                    physicsConstraintAccepted = true,
                    globalRigTrust = 0.96,
                    faceEyeFusionAccepted = true,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.44,
                    physicsConstraintAccepted = true,
                    globalRigTrust = 0.42,
                    faceEyeFusionAccepted = true,
                ),
            )

        assertTrue(degraded.score < healthy.score)
        assertTrue(
            degraded.componentScores["physicsConsistency"]!! <
                healthy.componentScores["physicsConsistency"]!!,
        )
        assertTrue(
            degraded.componentScores["globalRigTrust"]!! <
                healthy.componentScores["globalRigTrust"]!!,
        )
    }

    @Test
    fun rejectedFaceEyeFusionRemainsVisibleAsQualityBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.34,
                    faceEyeFusionAccepted = false,
                ),
            )

        assertTrue(decision.hardFailure)
        assertTrue(
            decision.componentScores["faceEye"]!! < 0.60,
        )
        assertTrue(
            decision.warnings.any {
                it.contains("face-eye") ||
                    it.contains("FACE_EYE")
            },
        )
    }

    @Test
    fun highFrequencyMotionRaisesTargetInferenceAndTrackingPriority() {
        val engine = HighEndReconstructionQualityEngine()

        val stable =
            engine.evaluate(
                cleanInput().copy(
                    motionDynamicStress = 0.18,
                    motionDynamicsScore = 0.72,
                    motionLowFrequencyScore = 0.42,
                    motionHighFrequencyScore = 0.08,
                ),
            )

        engine.reset()

        val burst =
            engine.evaluate(
                cleanInput().copy(
                    motionDynamicStress = 0.52,
                    motionDynamicsScore = 0.84,
                    motionLowFrequencyScore = 0.44,
                    motionHighFrequencyScore = 0.82,
                ),
            )

        assertTrue(
            burst.budget.targetInferenceFps >=
                stable.budget.targetInferenceFps,
        )
        assertTrue(
            burst.budget.eyeTrackingHz >=
                stable.budget.eyeTrackingHz,
        )
        assertTrue(
            burst.budget.handTrackingHz >=
                stable.budget.handTrackingHz,
        )
        assertTrue(
            burst.budget.detectorIntervalFrames <=
                stable.budget.detectorIntervalFrames,
        )
    }

    @Test
    fun rootVelocityAgreementBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    global4DRootVelocityAgreement = 0.97,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    global4DRootVelocityAgreement = 0.44,
                ),
            )

        assertTrue(
            degraded.componentScores["global4DRootVelocity"]!! <
                clean.componentScores["global4DRootVelocity"]!!,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("root-velocity")
            },
        )
        assertTrue(degraded.score < clean.score)
    }

    @Test
    fun extremelyLowRootVelocityAgreementRejectsTrustedOutput() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    global4DRootVelocityAgreement = 0.08,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            result.policy,
        )
    }

    @Test
    fun lowFaceEyeCaptureDomainBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeCaptureDomainConfidence = 0.96,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeCaptureDomainConfidence = 0.42,
                ),
            )

        assertTrue(
            degraded.componentScores["faceEyeCaptureDomain"]!! <
                clean.componentScores["faceEyeCaptureDomain"]!!,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("capture-domain")
            },
        )
        assertTrue(
            degraded.score < clean.score,
        )
    }

    @Test
    fun extremelyLowFaceEyeCaptureDomainRejectsTrustedOutput() {
        val engine = HighEndReconstructionQualityEngine()

        val result =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeCaptureDomainConfidence = 0.08,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            result.policy,
        )
    }

    @Test
    fun handUncertaintyCalibrationBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    handUncertaintyConfidence = 0.96,
                    handUncertaintyOodRisk = 0.02,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    handUncertaintyConfidence = 0.38,
                    handUncertaintyOodRisk = 0.62,
                ),
            )

        assertTrue(
            degraded.componentScores["handUncertaintyCalibration"]!! <
                clean.componentScores["handUncertaintyCalibration"]!!,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("hand-uncertainty")
            },
        )
        assertTrue(degraded.score < clean.score)
    }

    @Test
    fun extremeHandUncertaintyRejectsTrustedOutput() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    handUncertaintyConfidence = 0.08,
                    handUncertaintyOodRisk = 0.98,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            result.policy,
        )
    }

    @Test
    fun worldSpaceConsistencyBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    worldSpaceConsistencyScore = 0.96,
                    worldSpaceRobustOutlierRatio = 0.02,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    worldSpaceConsistencyScore = 0.44,
                    worldSpaceRobustOutlierRatio = 0.56,
                ),
            )

        assertTrue(
            degraded.componentScores["worldSpaceConsistency"]!! <
                clean.componentScores["worldSpaceConsistency"]!!,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("world-space")
            },
        )
        assertTrue(degraded.score < clean.score)
    }

    @Test
    fun extremeWorldSpaceInconsistencyRejectsTrustedOutput() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    worldSpaceConsistencyScore = 0.08,
                    worldSpaceRobustOutlierRatio = 0.94,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            result.policy,
        )
    }

    @Test
    fun worldSpaceDegradationRedirectsComputeTowardRecovery() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    worldSpaceConsistencyScore = 0.98,
                    worldSpaceRobustOutlierRatio = 0.02,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    worldSpaceConsistencyScore = 0.34,
                    worldSpaceRobustOutlierRatio = 0.66,
                ),
            )

        assertTrue(
            degraded.budget.bundleAdjustmentIterations >=
                clean.budget.bundleAdjustmentIterations,
        )
        assertTrue(
            degraded.budget.meshVertexBudget <=
                clean.budget.meshVertexBudget,
        )
        assertTrue(
            degraded.budget.targetInferenceFps >=
                clean.budget.targetInferenceFps,
        )
        assertFalse(
            degraded.budget.allowDenseSurface,
        )
    }

    @Test
    fun rootVelocityRecoveryRedirectsComputeTowardInertialResolution() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    global4DRootVelocityAgreement = 0.98,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    global4DRootVelocityAgreement = 0.36,
                ),
            )

        assertTrue(
            degraded.budget.bundleAdjustmentIterations >=
                healthy.budget.bundleAdjustmentIterations,
        )
        assertTrue(
            degraded.budget.meshVertexBudget <=
                healthy.budget.meshVertexBudget,
        )
        assertTrue(
            degraded.budget.targetInferenceFps >=
                healthy.budget.targetInferenceFps,
        )
    }

    @Test
    fun physicsDegradationRedirectsComputeTowardRecovery() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.96,
                    physicsConstraintAccepted = true,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    physicsConstraintScore = 0.34,
                    physicsConstraintAccepted = true,
                ),
            )

        assertTrue(
            degraded.budget.bundleAdjustmentIterations >=
                healthy.budget.bundleAdjustmentIterations,
        )
        assertTrue(
            degraded.budget.meshVertexBudget <=
                healthy.budget.meshVertexBudget,
        )
        assertTrue(
            !degraded.budget.allowDenseSurface ||
                degraded.budget.meshVertexBudget <
                    healthy.budget.meshVertexBudget,
        )
        assertFalse(
            degraded.budget.allowReferenceOptimization,
        )
    }

    @Test
    fun lowRigTrustProtectsDenseSurfaceAndReferenceOptimization() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    globalRigTrust = 0.48,
                    physicsConstraintScore = 0.92,
                    physicsConstraintAccepted = true,
                    faceEyeFusionAccepted = true,
                ),
            )

        assertTrue(
            decision.budget.meshVertexBudget <
                ProductionCaptureProfile.REFERENCE_BAKE.meshVertexBudget,
        )
        assertTrue(
            !decision.budget.allowReferenceOptimization ||
                decision.score >= 0.88,
        )
        assertTrue(
            decision.budget.denseMeshHz <=
                ProductionCaptureProfile.REFERENCE_BAKE.denseMeshHz,
        )
    }

    @Test
    fun rapidGazeReceivesTemporalBurstWithoutChangingHealthyPolicy() {
        val engine = HighEndReconstructionQualityEngine()

        val steady =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.92,
                    faceEyeFusionAccepted = true,
                    faceEyeSaccadeVelocityDegreesPerSecond = 60.0,
                ),
            )

        engine.reset()

        val rapid =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.92,
                    faceEyeFusionAccepted = true,
                    faceEyeSaccadeVelocityDegreesPerSecond = 820.0,
                ),
            )

        assertTrue(
            rapid.budget.eyeTrackingHz >=
                steady.budget.eyeTrackingHz,
        )
        assertTrue(
            rapid.budget.targetInferenceFps >=
                steady.budget.targetInferenceFps,
        )
        assertTrue(
            rapid.outputAllowed,
        )
    }

    @Test
    fun explicitFaceEyeHardFailureCanRejectGlobalOutput() {
        val engine = HighEndReconstructionQualityEngine()

        val decision =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeConfidence = 0.86,
                    faceEyeFusionAccepted = true,
                    faceEyeHardFailure = true,
                    faceEyeShapeConfidence = 0.18,
                    faceEyeEyeDynamicConsistency = 0.24,
                ),
            )

        assertTrue(decision.hardFailure)
        assertFalse(decision.outputAllowed)
        assertTrue(
            decision.warnings.any {
                it.contains("face-eye-hard-failure")
            },
        )
    }

    @Test
    fun weakFaceShapeAndEyeDynamicsLowerGlobalQualityWithoutArtificialHardFailure() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeShapeConfidence = 0.96,
                    faceEyeEyeDynamicConsistency = 0.95,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeShapeConfidence = 0.46,
                    faceEyeEyeDynamicConsistency = 0.44,
                ),
            )

        assertTrue(
            degraded.score < healthy.score,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("face-eye-shape") ||
                    it.contains("face-eye-dynamic")
            },
        )
    }

    @Test
    fun lowSurfaceObservabilityAndHighUncertaintyReduceGlobalQuality() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    surfaceConfidence = 0.94,
                    surfaceNormalObservability = 0.95,
                    surfaceHighUncertaintyRatio = 0.08,
                    cameraInformationGain = 0.94,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    surfaceConfidence = 0.94,
                    surfaceNormalObservability = 0.44,
                    surfaceHighUncertaintyRatio = 0.58,
                    cameraInformationGain = 0.28,
                ),
            )

        assertTrue(degraded.score < healthy.score)
        assertTrue(
            degraded.warnings.any {
                it.contains("surface-normal") ||
                    it.contains("surface-high") ||
                    it.contains("surface-view")
            },
        )
        assertTrue(
            degraded.bottleneck ==
                HighEndReconstructionQualityEngine.Bottleneck.SURFACE ||
                degraded.componentScores["surface"]!! <
                    healthy.componentScores["surface"]!!,
        )
    }

    @Test
    fun higherSurfaceEvidenceStrictnessTightensGeometryAcceptance() {
        val engine = HighEndReconstructionQualityEngine()

        val permissive =
            engine.evaluate(
                cleanInput().copy(
                    surfaceEvidenceStrictness = 0.20,
                    surfaceNormalObservability = 0.50,
                    surfaceHighUncertaintyRatio = 0.42,
                    cameraInformationGain = 0.48,
                ),
            )

        engine.reset()

        val strict =
            engine.evaluate(
                cleanInput().copy(
                    surfaceEvidenceStrictness = 0.96,
                    surfaceNormalObservability = 0.50,
                    surfaceHighUncertaintyRatio = 0.42,
                    cameraInformationGain = 0.48,
                ),
            )

        assertTrue(strict.score <= permissive.score)
        assertTrue(
            strict.hardFailure ||
                strict.warnings.any {
                    it.contains("surface")
                },
        )
    }

    @Test
    fun degradedFaceEyeUncertaintyCalibrationLowersProductionQuality() {
        val engine = HighEndReconstructionQualityEngine()

        val healthy =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeUncertaintyCalibrationConfidence = 0.94,
                    faceEyeUncertaintyCalibrationReady = true,
                    faceUncertaintyCoverage = 0.95,
                    eyeUncertaintyCoverage = 0.95,
                    faceUncertaintyCalibrationMultiplier = 1.02,
                    eyeUncertaintyCalibrationMultiplier = 1.04,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    faceEyeUncertaintyCalibrationConfidence = 0.40,
                    faceEyeUncertaintyCalibrationReady = false,
                    faceUncertaintyCoverage = 0.68,
                    eyeUncertaintyCoverage = 0.71,
                    faceUncertaintyCalibrationMultiplier = 2.80,
                    eyeUncertaintyCalibrationMultiplier = 3.10,
                ),
            )

        assertTrue(
            degraded.componentScores["faceEyeUncertaintyCalibration"]!! <
                healthy.componentScores["faceEyeUncertaintyCalibration"]!!,
        )
        assertTrue(
            degraded.score < healthy.score,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("face-eye-uncertainty")
            },
        )
    }

    @Test
    fun thermalPressureReducesComputeBudget() {
        val engine = HighEndReconstructionQualityEngine()

        val baseline = engine.evaluate(cleanInput())
        engine.reset()
        val constrained =
            engine.evaluate(
                cleanInput().copy(
                    thermalScale = 0.30,
                    memoryScale = 0.45,
                    runtimeQualityScale = 0.48,
                    runtimeInferenceScale = 0.55,
                ),
            )

        assertTrue(
            constrained.budget.tsdfStride >= baseline.budget.tsdfStride,
        )
        assertTrue(
            constrained.budget.meshVertexBudget <=
                baseline.budget.meshVertexBudget,
        )
        assertTrue(
            constrained.budget.bundleAdjustmentIterations <=
                baseline.budget.bundleAdjustmentIterations,
        )
    }

    @Test
    fun stale_quality_frame_cannot_rewrite_hysteresis_state() {
        val engine = HighEndReconstructionQualityEngine()

        val current = engine.evaluate(
            cleanInput().copy(
                timestampNanos = 2_000_000_000L,
                synchronizationConfidence = 0.98,
            ),
        )

        val stale = engine.evaluate(
            cleanInput().copy(
                timestampNanos = 1_500_000_000L,
                synchronizationConfidence = 0.10,
                calibrationConfidence = 0.10,
                hardCaptureFailure = true,
            ),
        )

        assertEquals(current.timestampNanos, stale.timestampNanos)
        assertTrue(
            stale.warnings.contains("stale-quality-frame-rejected"),
        )
        assertEquals(current.policy, stale.policy)
    }


    @Test
    fun measuredSynchronizationResidualsLowerQualityAndCanHardFail() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    synchronizationResidualP95Milliseconds = 1.5,
                    synchronizationOutlierRatio = 0.01,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    synchronizationResidualP95Milliseconds = 8.0,
                    synchronizationOutlierRatio = 0.18,
                ),
            )

        assertTrue(
            degraded.componentScores["synchronization"]!! <
                clean.componentScores["synchronization"]!!,
        )
        assertTrue(degraded.score < clean.score)

        engine.reset()

        val hard =
            engine.evaluate(
                cleanInput().copy(
                    synchronizationResidualP95Milliseconds = 15.0,
                    synchronizationOutlierRatio = 0.24,
                ),
            )

        assertTrue(hard.hardFailure)
        assertFalse(hard.outputAllowed)
    }



    @Test
    fun denseLandmarkDegradationBecomesProductionBottleneck() {
        val engine = HighEndReconstructionQualityEngine()

        val clean =
            engine.evaluate(
                cleanInput().copy(
                    denseLandmarkCoverage = 0.96,
                    denseLandmarkVisibleCoverage = 0.94,
                    denseLandmarkUncertaintyP95 = 0.08,
                    denseLandmarkCrossViewAgreement = 0.94,
                    denseLandmarkCrossViewEvidenceAvailable = true,
                    denseLandmarkTemporalStability = 0.94,
                ),
            )

        engine.reset()

        val degraded =
            engine.evaluate(
                cleanInput().copy(
                    denseLandmarkCoverage = 0.42,
                    denseLandmarkVisibleCoverage = 0.38,
                    denseLandmarkUncertaintyP95 = 0.48,
                    denseLandmarkCrossViewAgreement = 0.34,
                    denseLandmarkCrossViewEvidenceAvailable = true,
                    denseLandmarkTemporalStability = 0.40,
                ),
            )

        assertTrue(
            degraded.componentScores["denseLandmarks"]!! <
                clean.componentScores["denseLandmarks"]!!,
        )
        assertTrue(degraded.score < clean.score)
        assertTrue(
            degraded.warnings.any {
                it.contains("dense-landmark")
            },
        )
    }

    @Test
    fun denseLandmarkHardFailureRejectsTrustedOutput() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    denseLandmarkHardFailure = true,
                    denseLandmarkCoverage = 0.10,
                    denseLandmarkUncertaintyP95 = 0.98,
                ),
            )

        assertTrue(result.hardFailure)
        assertFalse(result.outputAllowed)
        assertEquals(
            HighEndReconstructionQualityEngine.Policy.REJECT,
            result.policy,
        )
    }

    @Test
    fun unavailableDenseCrossViewEvidenceDoesNotBecomeFalseFailure() {
        val result =
            HighEndReconstructionQualityEngine().evaluate(
                cleanInput().copy(
                    denseLandmarkCoverage = 0.92,
                    denseLandmarkVisibleCoverage = 0.90,
                    denseLandmarkUncertaintyP95 = 0.12,
                    denseLandmarkCrossViewAgreement = 0.0,
                    denseLandmarkCrossViewEvidenceAvailable = false,
                    denseLandmarkTemporalStability = 0.92,
                ),
            )

        assertTrue(
            result.componentScores["denseLandmarks"]!! > 0.60,
        )
        assertTrue(
            result.warnings.none {
                it.contains("cross-view-agreement")
            },
        )
    }
}
