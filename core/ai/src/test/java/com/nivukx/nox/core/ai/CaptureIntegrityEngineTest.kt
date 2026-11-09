package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.MotionCaptureCustomization
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureIntegrityEngineTest {

    private fun input(
        reconstruction: Double = 0.92,
        reprojection: Double = 1.8,
        uncertainty: Double = 0.06,
        geometry: Double = 0.92,
        temporal: Double = 0.92,
        temporalStability: Double = 0.90,
        sync: Double = 0.94,
        calibration: Double = 0.95,
        subject: Double = 0.94,
        motion: Double = 0.72,
        evidence: Double = 0.90,
        micro: Double = 0.60,
        occlusion: Double = 0.12,
        cameraMotion: Double = 0.06,
        faceEye: Double = 0.88,
        faceGeometry: Double = 0.90,
        eyeTemporal: Double = 0.92,
        hand: Double = 0.88,
        thermal: Double = 0.95,
        memory: Double = 0.95,
        runtime: Double = 0.95,
        multiCamera: Boolean = true,
    ) = CaptureIntegrityEngine.Input(
        timestampNanos = 1_000_000L,
        reconstructionConfidence = reconstruction,
        reprojectionErrorPx = reprojection,
        meanUncertaintyMeters = uncertainty,
        geometricObservability = geometry,
        temporalConfidence = temporal,
        temporalStability = temporalStability,
        synchronizationConfidence = sync,
        calibrationConfidence = calibration,
        subjectConfidence = subject,
        motionScore = motion,
        motionEvidenceConfidence = evidence,
        microMotionScore = micro,
        occlusionRisk = occlusion,
        cameraMotionRisk = cameraMotion,
        faceEyeConfidence = faceEye,
        faceGeometryConfidence = faceGeometry,
        eyeTemporalConfidence = eyeTemporal,
        handConfidence = hand,
        thermalScale = thermal,
        memoryScale = memory,
        runtimeQualityScale = runtime,
        multiCamera = multiCamera,
        customization = MotionCaptureCustomization.qualityFirst(),
    )

        @Test
    fun severeOpticalBlurBlocksCaptureIntegrity() {
        val engine = CaptureIntegrityEngine()
        val base =
            baseInput().copy(
                motionBlurRisk = 0.96,
                timestampNanos = 10L,
            )
        val result = engine.evaluate(base)

        assertTrue(
            result.hardFailures.contains(
                "optical-motion-blur-hard-gate",
            ),
        )
        assertTrue(!result.outputAllowed)
    }

@Test
    fun cleanFrameNeedsConfirmationBeforeTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        val first = engine.evaluate(input())

        assertTrue(first.smoothedScore > 0.70)
        assertFalse(first.outputAllowed)

        repeat(4) { engine.evaluate(input()) }
        val trusted = engine.evaluate(input())

        assertTrue(trusted.outputAllowed)
        assertTrue(
            trusted.band == CaptureIntegrityEngine.Band.TRUSTED ||
                trusted.band == CaptureIntegrityEngine.Band.EXCELLENT,
        )
    }

    @Test
    fun hardGeometryFailureBlocksOutputImmediately() {
        val engine = CaptureIntegrityEngine()
        repeat(4) { engine.evaluate(input()) }

        val blocked =
            engine.evaluate(
                input(
                    geometry = 0.10,
                    reprojection = 18.0,
                ),
            )

        assertFalse(blocked.outputAllowed)
        assertTrue(blocked.band == CaptureIntegrityEngine.Band.BLOCKED)
        assertTrue(blocked.hardFailures.isNotEmpty())
    }

    
    @Test
    fun disabledOptionalSubsystemsDoNotBecomeHiddenIntegrityFailures() {
        val engine = CaptureIntegrityEngine()
        val customization =
            MotionCaptureCustomization.qualityFirst().copy(
                recordFaceEyeFusion = false,
                recordHand3D = false,
            )
        val result =
            engine.evaluate(
                input(
                    faceEye = 0.0,
                    hand = 0.0,
                ).copy(customization = customization),
            )

        assertTrue(result.components.faceEye >= 0.99)
        assertTrue(result.components.hand >= 0.99)
        assertTrue(
            result.warnings.none {
                it == "face-eye-evidence-limited" ||
                    it == "hand-evidence-limited"
            },
        )
    }

    @Test
    fun higherOutputStrictnessRaisesTrustedOutputFloor() {
        val relaxedEngine = CaptureIntegrityEngine()
        val strictEngine = CaptureIntegrityEngine()
        val relaxed =
            MotionCaptureCustomization.production().copy(
                outputIntegrityStrictness = 0.20,
            )
        val strict =
            MotionCaptureCustomization.production().copy(
                outputIntegrityStrictness = 1.0,
            )

        repeat(8) {
            relaxedEngine.evaluate(
                input().copy(customization = relaxed),
            )
            strictEngine.evaluate(
                input().copy(customization = strict),
            )
        }

        val relaxedResult =
            relaxedEngine.evaluate(
                input().copy(customization = relaxed),
            )
        val strictResult =
            strictEngine.evaluate(
                input().copy(customization = strict),
            )

        assertTrue(
            "strict integrity floor should be at least as hard to satisfy",
            !strictResult.outputAllowed ||
                relaxedResult.outputAllowed,
        )
    }

    @Test
    fun kinematicFailureBlocksTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        repeat(4) {
            engine.evaluate(
                input().copy(
                    kinematicConstraintScore = 0.82,
                    surfaceConfidence = 0.84,
                ),
            )
        }

        val result =
            engine.evaluate(
                input().copy(
                    kinematicConstraintScore = 0.08,
                    kinematicWarnings = listOf("hard-kinematic-violation"),
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "kinematic-constraint-hard-gate",
            ),
        )
        assertTrue(
            result.bottleneck ==
                CaptureIntegrityEngine.Bottleneck.KINEMATIC ||
                result.band == CaptureIntegrityEngine.Band.BLOCKED,
        )
    }

    @Test
    fun denseSurfaceConfidenceParticipatesInIntegrityScore() {
        val engine = CaptureIntegrityEngine()

        repeat(6) {
            engine.evaluate(
                input().copy(
                    surfaceConfidence = 0.92,
                    kinematicConstraintScore = 0.94,
                ),
            )
        }

        val strong =
            engine.evaluate(
                input().copy(
                    surfaceConfidence = 0.92,
                    kinematicConstraintScore = 0.94,
                ),
            )

        val weakEngine = CaptureIntegrityEngine()
        repeat(6) {
            weakEngine.evaluate(
                input().copy(
                    surfaceConfidence = 0.32,
                    kinematicConstraintScore = 0.94,
                    surfaceWarnings =
                        listOf("surface-confidence-tail-low"),
                ),
            )
        }

        val weak =
            weakEngine.evaluate(
                input().copy(
                    surfaceConfidence = 0.32,
                    kinematicConstraintScore = 0.94,
                    surfaceWarnings =
                        listOf("surface-confidence-tail-low"),
                ),
            )

        assertTrue(
            strong.components.surface >
                weak.components.surface,
        )
        assertTrue(strong.rawScore > weak.rawScore)
    }

    @Test
    fun lowGlobalRigTrustBlocksTrustedOutput() {
        val result =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    globalRigTrust = 0.11,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "global-rig-trust-hard-gate",
            ),
        )
    }

    @Test
    fun moderateFaceEyeRejectionDoesNotHardBlockBodyIntegrity() {
        val engine = CaptureIntegrityEngine()

        val result =
            engine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.92,
                    faceEyeFusionAccepted = false,
                ),
            )

        assertTrue(
            result.hardFailures.none {
                it == "face-eye-fusion-hard-gate"
            },
        )
        assertTrue(
            result.warnings.any {
                it.contains("face-eye-recovery-only-body-output-preserved")
            },
        )
    }

    @Test
    fun severeFaceEyeRejectionHardBlocksIntegrity() {
        val engine = CaptureIntegrityEngine()

        val result =
            engine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.04,
                    faceEyeFusionAccepted = false,
                ),
            )

        assertTrue(
            result.hardFailures.contains(
                "face-eye-fusion-hard-gate",
            ),
        )
        assertFalse(result.outputAllowed)
    }

    @Test
    fun faceEyeFusionUncertaintyReducesIntegrityAndCanHardBlock() {
        val clean =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    faceEyeFusionUncertaintyTotal = 0.08,
                    faceEyeFusionUncertaintyTrustworthy = true,
                ),
            )

        val degraded =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    faceEyeFusionUncertaintyTotal = 0.76,
                    faceEyeFusionUncertaintyTrustworthy = false,
                ),
            )

        assertTrue(
            degraded.components.faceEye <
                clean.components.faceEye,
        )
        assertTrue(
            degraded.warnings.any {
                it.contains("face-eye-fusion-uncertainty")
            },
        )

        val blocked =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    faceEyeFusionUncertaintyTotal = 0.90,
                    faceEyeFusionUncertaintyTrustworthy = false,
                ),
            )

        assertTrue(
            blocked.hardFailures.contains(
                "face-eye-fusion-uncertainty-hard-gate",
            ),
        )
        assertFalse(blocked.outputAllowed)
    }


    @Test
    fun faceEyeCalibrationDriftReducesIntegrityBeforeHardFailure() {
        val cleanEngine = CaptureIntegrityEngine()
        val driftEngine = CaptureIntegrityEngine()

        val clean =
            cleanEngine.evaluate(
                input().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.05,
                ),
            )

        val drifted =
            driftEngine.evaluate(
                input().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.82,
                ),
            )

        assertTrue(
            drifted.components.faceEye <
                clean.components.faceEye,
        )
        assertTrue(
            drifted.rawScore <
                clean.rawScore,
        )
        assertTrue(
            drifted.warnings.any {
                it.contains("recalibration")
            },
        )
    }

    @Test
    fun persistentFaceEyeCalibrationDriftHardBlocksTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        val result =
            engine.evaluate(
                input().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.68,
                    faceEyeCalibrationDriftPersistent = true,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "face-eye-calibration-drift-persistent-hard-gate",
            ),
        )
    }

    @Test
    fun extremeFaceEyeCalibrationDriftBlocksTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        val result =
            engine.evaluate(
                input().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeCalibrationDriftScore = 0.94,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "face-eye-calibration-drift-hard-gate",
            ),
        )
    }

    @Test
    fun global4DConsistencyBlocksTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        val healthy =
            engine.evaluate(
                input().copy(
                    global4DScore = 0.94,
                    global4DAccepted = true,
                    global4DPersistentFailure = false,
                    global4DRootMotionConsistency = 0.94,
                    global4DSceneConsistency = 0.90,
                ),
            )

        val broken =
            engine.evaluate(
                input().copy(
                    global4DScore = 0.12,
                    global4DAccepted = false,
                    global4DPersistentFailure = true,
                    global4DRootMotionConsistency = 0.22,
                    global4DSceneConsistency = 0.30,
                ),
            )

        assertTrue(
            broken.hardFailures.contains(
                "global-4d-consistency-hard-gate",
            ),
        )
        assertTrue(
            broken.components.global4D <
                healthy.components.global4D,
        )
        assertTrue(!broken.outputAllowed)
    }

    @Test
    fun faceEyeOutOfDistributionRiskBlocksTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        val healthy =
            engine.evaluate(
                input().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeConfidence = 0.94,
                    faceEyeOutOfDistributionRisk = 0.08,
                ),
            )

        val ood =
            engine.evaluate(
                input().copy(
                    faceEyeFusionAccepted = true,
                    faceEyeConfidence = 0.94,
                    faceEyeOutOfDistributionRisk = 0.97,
                ),
            )

        assertTrue(
            ood.hardFailures.contains(
                "face-eye-out-of-distribution-hard-gate",
            ),
        )
        assertTrue(
            ood.components.faceEyeOod <
                healthy.components.faceEyeOod,
        )
        assertTrue(!ood.outputAllowed)
    }

    @Test
    fun physicsFailureBlocksTrustedOutput() {
        val engine = CaptureIntegrityEngine()

        repeat(6) {
            engine.evaluate(
                input().copy(
                    physicsConstraintScore = 0.94,
                    physicsConstraintAccepted = true,
                ),
            )
        }

        val result =
            engine.evaluate(
                input().copy(
                    physicsConstraintScore = 0.10,
                    physicsConstraintAccepted = false,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "physics-constraint-hard-gate",
            ),
        )
        assertTrue(
            result.components.physics < 0.20,
        )
        assertTrue(
            result.bottleneck ==
                CaptureIntegrityEngine.Bottleneck.PHYSICS ||
                result.band == CaptureIntegrityEngine.Band.BLOCKED,
        )
    }

    @Test
    fun directionalUncertaintyBlocksTrustedIntegrity() {
        val engine = CaptureIntegrityEngine()

        repeat(6) {
            engine.evaluate(
                input().copy(
                    directionalUncertaintyMajor95Meters = 0.03,
                    directionalUncertaintyDepthMeters = 0.02,
                    directionalUncertaintyAnisotropy = 1.5,
                    directionalUncertaintyConfidence = 0.96,
                ),
            )
        }

        val result =
            engine.evaluate(
                input().copy(
                    directionalUncertaintyMajor95Meters = 1.10,
                    directionalUncertaintyDepthMeters = 1.04,
                    directionalUncertaintyAnisotropy = 22.0,
                    directionalUncertaintyConfidence = 0.05,
                    directionalUncertaintyDegradedJointCount = 15,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "directional-uncertainty-hard-gate",
            ),
        )
        assertTrue(
            result.components.directionalUncertainty < 0.25,
        )
        assertTrue(
            result.bottleneck ==
                CaptureIntegrityEngine.Bottleneck.DIRECTIONAL_UNCERTAINTY ||
                result.band ==
                    CaptureIntegrityEngine.Band.BLOCKED,
        )
    }

    @Test
    fun globalRigTrustInfluencesTrustedOutputScore() {
        val strongEngine = CaptureIntegrityEngine()
        val weakEngine = CaptureIntegrityEngine()

        repeat(6) {
            strongEngine.evaluate(
                input().copy(
                    globalRigTrust = 0.96,
                ),
            )
            weakEngine.evaluate(
                input().copy(
                    globalRigTrust = 0.42,
                ),
            )
        }

        val strong =
            strongEngine.evaluate(
                input().copy(
                    globalRigTrust = 0.96,
                ),
            )
        val weak =
            weakEngine.evaluate(
                input().copy(
                    globalRigTrust = 0.42,
                ),
            )

        assertTrue(
            strong.components.final >
                weak.components.final,
        )
        assertTrue(
            weak.warnings.any {
                it.contains("rig") ||
                    it.contains("coverage")
            } || weak.components.rigCoverage < strong.components.rigCoverage,
        )
    }

    @Test
    fun physicsEvidenceChangesIntegrityScoreWithoutBreakingHealthyDefaults() {
        val strongEngine = CaptureIntegrityEngine()
        val weakEngine = CaptureIntegrityEngine()

        repeat(7) {
            strongEngine.evaluate(
                input().copy(
                    physicsConstraintScore = 0.94,
                    physicsConstraintAccepted = true,
                ),
            )
            weakEngine.evaluate(
                input().copy(
                    physicsConstraintScore = 0.46,
                    physicsConstraintAccepted = true,
                ),
            )
        }

        val strong =
            strongEngine.evaluate(
                input().copy(
                    physicsConstraintScore = 0.94,
                    physicsConstraintAccepted = true,
                ),
            )
        val weak =
            weakEngine.evaluate(
                input().copy(
                    physicsConstraintScore = 0.46,
                    physicsConstraintAccepted = true,
                ),
            )

        assertTrue(strong.components.physics > weak.components.physics)
        assertTrue(strong.rawScore > weak.rawScore)
    }

    @Test
    fun decliningFaceEyeReliabilityLowersTrustedIntegrity() {
        val stableEngine = CaptureIntegrityEngine()
        val decliningEngine = CaptureIntegrityEngine()

        repeat(6) {
            stableEngine.evaluate(
                input().copy(
                    faceEyeReliabilityTrend = 0.02,
                ),
            )
            decliningEngine.evaluate(
                input().copy(
                    faceEyeReliabilityTrend = -0.42,
                ),
            )
        }

        val stable =
            stableEngine.evaluate(
                input().copy(
                    faceEyeReliabilityTrend = 0.02,
                ),
            )
        val declining =
            decliningEngine.evaluate(
                input().copy(
                    faceEyeReliabilityTrend = -0.42,
                ),
            )

        assertTrue(
            declining.components.faceEye <
                stable.components.faceEye,
        )
        assertTrue(
            declining.warnings.any {
                it.contains("face-eye-reliability")
            },
        )
        assertTrue(
            declining.rawScore < stable.rawScore,
        )
    }

    @Test
    fun catastrophicFaceEyeReliabilityCollapseBlocksTrustedIntegrity() {
        val engine = CaptureIntegrityEngine()

        val result =
            engine.evaluate(
                input().copy(
                    faceEyeReliabilityTrend = -0.84,
                    faceEyeFusionAccepted = true,
                    faceEyeConfidence = 0.94,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "face-eye-reliability-collapse-hard-gate",
            ),
        )
    }

    @Test
    fun explicitFaceEyeHardFailureBlocksTrustedIntegrity() {
        val engine = CaptureIntegrityEngine()

        val result =
            engine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.92,
                    faceEyeFusionAccepted = true,
                    faceEyeHardFailure = true,
                    faceEyeShapeConfidence = 0.90,
                    faceEyeEyeDynamicConsistency = 0.90,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "face-eye-hard-gate",
            ),
        )
    }

    @Test
    fun lowFaceShapeAndEyeDynamicConsistencyReduceFaceEyeIntegrity() {
        val strongEngine = CaptureIntegrityEngine()
        val weakEngine = CaptureIntegrityEngine()

        repeat(5) {
            strongEngine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.94,
                    faceEyeFusionAccepted = true,
                    faceEyeShapeConfidence = 0.94,
                    faceEyeEyeDynamicConsistency = 0.94,
                ),
            )
            weakEngine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.94,
                    faceEyeFusionAccepted = true,
                    faceEyeShapeConfidence = 0.44,
                    faceEyeEyeDynamicConsistency = 0.42,
                ),
            )
        }

        val strong =
            strongEngine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.94,
                    faceEyeFusionAccepted = true,
                    faceEyeShapeConfidence = 0.94,
                    faceEyeEyeDynamicConsistency = 0.94,
                ),
            )

        val weak =
            weakEngine.evaluate(
                input().copy(
                    faceEyeConfidence = 0.94,
                    faceEyeFusionAccepted = true,
                    faceEyeShapeConfidence = 0.44,
                    faceEyeEyeDynamicConsistency = 0.42,
                ),
            )

        assertTrue(
            strong.components.faceEye >
                weak.components.faceEye,
        )
        assertTrue(
            strong.rawScore >
                weak.rawScore,
        )
        assertTrue(
            weak.warnings.any {
                it.contains("face-eye-shape") ||
                    it.contains("face-eye-dynamic")
            },
        )
    }

    @Test
    fun poorSurfaceObservabilityBlocksTrustedSurfaceOutput() {
        val result =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    surfaceConfidence = 0.92,
                    surfaceNormalObservability = 0.10,
                    surfaceHighUncertaintyRatio = 0.90,
                    surfaceCameraInformationGain = 0.24,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.any {
                it.contains("surface-normal")
            },
        )
    }

    @Test
    fun lowSurfaceViewInformationReducesIntegrityWithoutImmediateHardFailure() {
        val strong =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    surfaceConfidence = 0.92,
                    surfaceNormalObservability = 0.90,
                    surfaceHighUncertaintyRatio = 0.10,
                    surfaceCameraInformationGain = 0.94,
                ),
            )

        val weak =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    surfaceConfidence = 0.92,
                    surfaceNormalObservability = 0.90,
                    surfaceHighUncertaintyRatio = 0.10,
                    surfaceCameraInformationGain = 0.32,
                ),
            )

        assertTrue(strong.rawScore > weak.rawScore)
        assertTrue(
            weak.warnings.any {
                it.contains("surface-view-information") ||
                    it.contains("surface")
            },
        )
    }

    @Test
    fun criticalRigCoverageBlocksMultiCameraOutput() {
        val engine = CaptureIntegrityEngine()

        repeat(5) {
            engine.evaluate(
                input().copy(
                    rigCoverageScore = 0.12,
                ),
            )
        }

        val result =
            engine.evaluate(
                input().copy(
                    rigCoverageScore = 0.12,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains("rig-coverage-critical"),
        )
        assertTrue(
            result.bottleneck ==
                CaptureIntegrityEngine.Bottleneck.RIG_COVERAGE ||
                result.band == CaptureIntegrityEngine.Band.BLOCKED,
        )
    }

    @Test
    fun degradedFaceEyeTemporalEvidenceReducesFaceEyeIntegrity() {
        val engine = CaptureIntegrityEngine()
        val result = engine.evaluate(
            input(
                faceEye = 0.95,
                faceGeometry = 0.20,
                eyeTemporal = 0.15,
            ),
        )

        assertTrue(
            "face-eye score should reflect 3D temporal evidence",
            result.components.faceEye < 0.85,
        )
        assertTrue(
            "face/eye temporal warnings should be surfaced",
            result.warnings.any {
                it.contains("face-3d-temporal") ||
                    it.contains("eye-3d-temporal")
            },
        )
    }

    fun monocularModeDoesNotRequireMultiCameraSync() {
        val engine = CaptureIntegrityEngine()

        repeat(5) {
            engine.evaluate(
                input(
                    sync = 0.0,
                    calibration = 0.0,
                    multiCamera = false,
                ),
            )
        }

        val result =
            engine.evaluate(
                input(
                    sync = 0.0,
                    calibration = 0.0,
                    multiCamera = false,
                ),
            )

        assertTrue(result.components.synchronization >= 0.99)
        assertTrue(result.components.calibration >= 0.55)
        assertFalse(
            result.hardFailures.any {
                it.contains("sync") || it.contains("calibration")
            },
        )
    }

    @Test
    fun weakCrossModalEpipolarEvidenceBlocksIntegrity() {
        val result =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    faceEpipolarConsistency = 0.10,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "cross-modal-epipolar-authority-hard-gate",
            ),
        )
        assertTrue(
            result.warnings.contains(
                "cross-modal-epipolar-authority-reduced",
            ),
        )
    }

    @Test
    fun weakCrossModalMetricEvidenceBlocksIntegrity() {
        val result =
            CaptureIntegrityEngine().evaluate(
                input().copy(
                    handMetricEvidenceAuthority = 0.10,
                ),
            )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.contains(
                "cross-modal-metric-authority-hard-gate",
            ),
        )
        assertTrue(
            result.warnings.contains(
                "cross-modal-metric-authority-reduced",
            ),
        )
    }

    @Test
    fun denseLandmarkDegradationReducesCaptureIntegrity() {
        val healthy = CaptureIntegrityEngine().evaluate(
            input().copy(
                denseLandmarkCoverage = 0.96,
                denseLandmarkVisibleCoverage = 0.94,
                denseLandmarkUncertaintyP95 = 0.08,
                denseLandmarkCrossViewAgreement = 0.94,
                denseLandmarkCrossViewEvidenceAvailable = true,
                denseLandmarkTemporalStability = 0.94,
            ),
        )

        val degraded = CaptureIntegrityEngine().evaluate(
            input().copy(
                denseLandmarkCoverage = 0.42,
                denseLandmarkVisibleCoverage = 0.38,
                denseLandmarkUncertaintyP95 = 0.48,
                denseLandmarkCrossViewAgreement = 0.34,
                denseLandmarkCrossViewEvidenceAvailable = true,
                denseLandmarkTemporalStability = 0.40,
            ),
        )

        assertTrue(
            degraded.components.denseLandmarks <
                healthy.components.denseLandmarks,
        )
        assertTrue(degraded.rawScore < healthy.rawScore)
        assertTrue(
            degraded.warnings.any {
                it.contains("dense-landmark")
            },
        )
    }

    @Test
    fun denseLandmarkHardFailureBlocksCaptureIntegrity() {
        val result = CaptureIntegrityEngine().evaluate(
            input().copy(
                denseLandmarkHardFailure = true,
                denseLandmarkCoverage = 0.12,
                denseLandmarkUncertaintyP95 = 0.98,
            ),
        )

        assertFalse(result.outputAllowed)
        assertTrue(
            result.hardFailures.any {
                it.startsWith("dense-landmark")
            },
        )
    }

    @Test
    fun unavailableDenseCrossViewEvidenceDoesNotBecomeFalseFailure() {
        val result = CaptureIntegrityEngine().evaluate(
            input().copy(
                denseLandmarkCoverage = 0.92,
                denseLandmarkVisibleCoverage = 0.90,
                denseLandmarkUncertaintyP95 = 0.12,
                denseLandmarkCrossViewAgreement = 0.0,
                denseLandmarkCrossViewEvidenceAvailable = false,
                denseLandmarkTemporalStability = 0.92,
            ),
        )

        assertTrue(result.components.denseLandmarks > 0.60)
        assertTrue(
            result.hardFailures.none {
                it.contains("dense-landmark-cross-view")
            },
        )

}
}
