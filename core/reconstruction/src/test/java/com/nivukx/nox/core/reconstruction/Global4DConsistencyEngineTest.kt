package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class Global4DConsistencyEngineTest {

    private fun skeleton(
        timestampNanos: Long,
        rootX: Double = 0.0,
        rootVelocityX: Double = 0.0,
    ): SkeletonFrame =
        SkeletonFrame(
            timestampNanos = timestampNanos,
            subjectType = SubjectType.HUMAN,
            trackId = 7L,
            joints = listOf(
                joint(23, -0.12 + rootX, 1.0, 0.0, rootVelocityX),
                joint(24, 0.12 + rootX, 1.0, 0.0, rootVelocityX),
                joint(11, -0.22 + rootX, 1.48, 0.0, rootVelocityX),
                joint(12, 0.22 + rootX, 1.48, 0.0, rootVelocityX),
            ),
            globalConfidence = 0.94,
        )

    private fun joint(
        id: Int,
        x: Double,
        y: Double,
        z: Double,
        velocityX: Double,
    ) =
        ReconstructionJoint(
            landmarkId = id,
            position = Vec3(x, y, z),
            velocity = Vec3(velocityX, 0.0, 0.0),
            confidence = 0.94,
            reprojectionErrorPx = 1.0,
            observationCount = 5,
            positionUncertaintyMeters = 0.015,
        )

    @Test
    fun healthyGlobalEvidenceProducesStableState() {
        val engine = Global4DConsistencyEngine()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(1_000_000_000L),
                    physicsScore = 0.94,
                    physicsAccepted = true,
                    reprojectionScore = 0.92,
                    temporalScore = 0.94,
                    synchronizationConfidence = 0.96,
                    rigTrust = 0.95,
                    cameraMotionRisk = 0.08,
                    sceneDecompositionConfidence = 0.90,
                    staticCoverage = 0.64,
                    dynamicCoverage = 0.22,
                    unknownCoverage = 0.14,
                ),
            )

        assertTrue(result.score > 0.78)
        assertTrue(
            result.state ==
                Global4DConsistencyEngine.State.STABLE ||
                result.state ==
                Global4DConsistencyEngine.State.WARMING,
        )
        assertTrue(result.accepted)
    }

    @Test
    fun cameraHumanInconsistencyBecomesPersistentFailure() {
        val engine = Global4DConsistencyEngine(
            Global4DConsistencyEngine.Config(
                persistentFailureFrames = 2,
                recoveryFrames = 3,
            ),
        )

        repeat(4) { index ->
            val result =
                engine.evaluate(
                    Global4DConsistencyEngine.Input(
                        skeleton =
                            skeleton(
                                2_000_000_000L +
                                    index * 33_000_000L,
                            ),
                        physicsScore = 0.62,
                        physicsAccepted = true,
                        reprojectionScore = 0.34,
                        temporalScore = 0.58,
                        synchronizationConfidence = 0.31,
                        rigTrust = 0.46,
                        cameraMotionRisk = 0.92,
                        sceneDecompositionConfidence = 0.36,
                        staticCoverage = 0.12,
                        dynamicCoverage = 0.70,
                        unknownCoverage = 0.18,
                    ),
                )

            if (index == 3) {
                assertTrue(result.persistentFailure)
                assertTrue(
                    result.warnings.any {
                        it.contains("global-4d")
                    },
                )
                assertTrue(!result.accepted)
            }
        }
    }

    @Test
    fun healthyRecoveryRequiresRepeatedConsistentFrames() {
        val engine = Global4DConsistencyEngine(
            Global4DConsistencyEngine.Config(
                persistentFailureFrames = 2,
                recoveryFrames = 4,
            ),
        )

        repeat(4) { index ->
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton =
                        skeleton(
                            3_000_000_000L +
                                index * 33_000_000L,
                        ),
                    physicsScore = 0.44,
                    physicsAccepted = false,
                    reprojectionScore = 0.28,
                    temporalScore = 0.38,
                    synchronizationConfidence = 0.25,
                    rigTrust = 0.35,
                    cameraMotionRisk = 0.95,
                    sceneDecompositionConfidence = 0.25,
                    staticCoverage = 0.08,
                    dynamicCoverage = 0.82,
                    unknownCoverage = 0.10,
                ),
            )
        }

        var recovering = engine.evaluate(
            Global4DConsistencyEngine.Input(
                skeleton = skeleton(3_200_000_000L),
                physicsScore = 0.92,
                physicsAccepted = true,
                reprojectionScore = 0.93,
                temporalScore = 0.94,
                synchronizationConfidence = 0.95,
                rigTrust = 0.94,
                cameraMotionRisk = 0.08,
                sceneDecompositionConfidence = 0.90,
                staticCoverage = 0.62,
                dynamicCoverage = 0.24,
                unknownCoverage = 0.14,
            ),
        )

        assertTrue(
            recovering.state ==
                Global4DConsistencyEngine.State.RECOVERING ||
                recovering.state ==
                Global4DConsistencyEngine.State.DEGRADED,
        )

        repeat(5) {
            recovering = engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton =
                        skeleton(
                            3_233_000_000L +
                                it * 33_000_000L,
                        ),
                    physicsScore = 0.95,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.96,
                    synchronizationConfidence = 0.97,
                    rigTrust = 0.96,
                    cameraMotionRisk = 0.04,
                    sceneDecompositionConfidence = 0.94,
                    staticCoverage = 0.66,
                    dynamicCoverage = 0.21,
                    unknownCoverage = 0.13,
                ),
            )
        }

        assertTrue(recovering.accepted)
        assertTrue(
            recovering.state ==
                Global4DConsistencyEngine.State.STABLE ||
                recovering.state ==
                Global4DConsistencyEngine.State.RECOVERING,
        )
    }

    @Test
    fun directionalUncertaintyCanDegradeGlobalAcceptance() {
        val engine = Global4DConsistencyEngine()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_500_000_000L),
                    physicsScore = 0.94,
                    physicsAccepted = true,
                    reprojectionScore = 0.93,
                    temporalScore = 0.94,
                    synchronizationConfidence = 0.95,
                    rigTrust = 0.95,
                    cameraMotionRisk = 0.08,
                    directionalUncertaintyConfidence = 0.12,
                    directionalUncertaintyAnisotropy = 15.5,
                    directionalUncertaintyDepthMeters = 0.19,
                    faceEyeConfidence = 0.92,
                    faceEyeAccepted = true,
                ),
            )

        assertTrue(
            result.warnings.any {
                it.contains("directional")
            },
        )
        assertTrue(
            result.directionalUncertaintyConsistency < 0.35,
        )
        assertTrue(!result.accepted)
    }

    @Test
    fun trustedExternalContactStrengthensContactConsistencyButWeakContactCanFail() {
        val healthyEngine = Global4DConsistencyEngine()

        val trusted =
            healthyEngine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_550_000_000L),
                    physicsScore = 0.92,
                    physicsAccepted = true,
                    reprojectionScore = 0.94,
                    temporalScore = 0.95,
                    synchronizationConfidence = 0.96,
                    rigTrust = 0.95,
                    cameraMotionRisk = 0.05,
                    externalContactConfidence = 0.96,
                    externalContactEvidenceAvailable = true,
                ),
            )

        assertTrue(trusted.externalContactConsistency > 0.90)
        assertTrue(trusted.contactConsistency > 0.70)

        val degradedEngine = Global4DConsistencyEngine()
        val weak =
            degradedEngine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_550_000_000L),
                    physicsScore = 0.92,
                    physicsAccepted = true,
                    reprojectionScore = 0.94,
                    temporalScore = 0.95,
                    synchronizationConfidence = 0.96,
                    rigTrust = 0.95,
                    cameraMotionRisk = 0.05,
                    externalContactConfidence = 0.12,
                    externalContactEvidenceAvailable = true,
                ),
            )

        assertTrue(weak.externalContactConsistency < 0.20)
        assertTrue(
            weak.warnings.any {
                it.contains("contact")
            },
        )
    }

    @Test
    fun degradedFaceEyeIntegrityDoesNotReplaceHealthyBodyState() {
        val engine = Global4DConsistencyEngine(
            Global4DConsistencyEngine.Config(
                minimumFaceEyeConfidence = 0.30,
            ),
        )

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_600_000_000L),
                    physicsScore = 0.94,
                    physicsAccepted = true,
                    reprojectionScore = 0.93,
                    temporalScore = 0.94,
                    synchronizationConfidence = 0.95,
                    rigTrust = 0.95,
                    cameraMotionRisk = 0.08,
                    directionalUncertaintyConfidence = 0.94,
                    directionalUncertaintyAnisotropy = 2.1,
                    directionalUncertaintyDepthMeters = 0.04,
                    faceEyeConfidence = 0.18,
                    faceEyeAccepted = false,
                    faceEyeOutOfDistributionRisk = 0.76,
                ),
            )

        assertTrue(
            result.faceEyeConsistency < 0.65,
        )
        assertTrue(
            result.warnings.any {
                it.contains("face-eye")
            },
        )
    }

    @Test
    fun crossModalEpipolarAuthorityCannotBeAveragedAway() {
        val engine = Global4DConsistencyEngine()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_650_000_000L),
                    physicsScore = 0.96,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.97,
                    synchronizationConfidence = 0.98,
                    rigTrust = 0.97,
                    cameraMotionRisk = 0.03,
                    bodyMetricEvidenceAuthority = 0.96,
                    faceMetricEvidenceAuthority = 0.96,
                    handMetricEvidenceAuthority = 0.96,
                    bodyEpipolarConsistency = 0.96,
                    faceEpipolarConsistency = 0.10,
                    handEpipolarConsistency = 0.96,
                ),
            )

        assertTrue(result.crossModalEpipolarAuthority < 0.18)
        assertTrue(
            result.warnings.any {
                it.contains("cross-modal-epipolar")
            },
        )
        assertTrue(!result.accepted)
    }

    @Test
    fun crossModalMetricAuthorityIsVisibleInGlobalResult() {
        val engine = Global4DConsistencyEngine()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_660_000_000L),
                    physicsScore = 0.96,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.97,
                    synchronizationConfidence = 0.98,
                    rigTrust = 0.97,
                    cameraMotionRisk = 0.03,
                    bodyMetricEvidenceAuthority = 0.92,
                    faceMetricEvidenceAuthority = 0.64,
                    handMetricEvidenceAuthority = 0.88,
                    bodyEpipolarConsistency = 0.96,
                    faceEpipolarConsistency = 0.94,
                    handEpipolarConsistency = 0.95,
                ),
            )

        assertTrue(result.crossModalMetricAuthority > 0.75)
        assertTrue(result.crossModalMetricAuthority < 0.90)
        assertTrue(result.crossModalEpipolarAuthority > 0.90)
    }

    @Test
    fun reportedRootVelocityMismatchDegradesGlobal4DConsistency() {
        val engine = Global4DConsistencyEngine()

        val first =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton =
                        skeleton(
                            timestampNanos = 4_700_000_000L,
                            rootVelocityX = 0.10,
                        ),
                    physicsScore = 0.94,
                    physicsAccepted = true,
                    reprojectionScore = 0.94,
                    temporalScore = 0.95,
                    synchronizationConfidence = 0.96,
                    rigTrust = 0.96,
                    cameraMotionRisk = 0.06,
                ),
            )

        val second =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton =
                        skeleton(
                            timestampNanos = 4_733_000_000L,
                            rootX = 0.60,
                            rootVelocityX = 5.8,
                        ),
                    physicsScore = 0.94,
                    physicsAccepted = true,
                    reprojectionScore = 0.94,
                    temporalScore = 0.95,
                    synchronizationConfidence = 0.96,
                    rigTrust = 0.96,
                    cameraMotionRisk = 0.06,
                ),
            )

        assertTrue(first.rootVelocityAgreement > 0.90)
        assertTrue(second.rootVelocityAgreement < first.rootVelocityAgreement)
        assertTrue(
            second.warnings.any {
                it.contains("root-velocity")
            },
        )
    }

    @Test
    fun cleanReportedRootVelocityPreservesGlobalAcceptance() {
        val engine = Global4DConsistencyEngine()

        var result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton =
                        skeleton(
                            timestampNanos = 4_800_000_000L,
                            rootVelocityX = 0.0,
                        ),
                    physicsScore = 0.95,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.97,
                    synchronizationConfidence = 0.98,
                    rigTrust = 0.97,
                    cameraMotionRisk = 0.04,
                ),
            )

        repeat(5) { index ->
            result =
                engine.evaluate(
                    Global4DConsistencyEngine.Input(
                        skeleton =
                            skeleton(
                                timestampNanos =
                                    4_833_000_000L +
                                        index * 33_000_000L,
                                rootX =
                                    (index + 1) * 0.10,
                                rootVelocityX = 3.03,
                            ),
                        physicsScore = 0.96,
                        physicsAccepted = true,
                        reprojectionScore = 0.97,
                        temporalScore = 0.97,
                        synchronizationConfidence = 0.98,
                        rigTrust = 0.98,
                        cameraMotionRisk = 0.03,
                    ),
                )
        }

        assertTrue(result.rootVelocityAgreement > 0.95)
        assertTrue(result.accepted)
    }

    @Test
    fun absentSceneDecompositionDoesNotInventSceneEvidence() {
        val engine = Global4DConsistencyEngine()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(4_000_000_000L),
                    physicsScore = 0.90,
                    physicsAccepted = true,
                    reprojectionScore = 0.92,
                    temporalScore = 0.93,
                    synchronizationConfidence = 0.95,
                    rigTrust = 0.94,
                    cameraMotionRisk = 0.08,
                    sceneDecompositionConfidence = null,
                    staticCoverage = null,
                    dynamicCoverage = null,
                    unknownCoverage = null,
                ),
            )

        assertTrue(result.sceneEvidenceAvailable.not())
        assertTrue(result.score > 0.65)
        assertTrue(
            result.warnings.any {
                it.contains("scene-decomposition-not-available")
            },
        )
    }

    @Test
    fun duplicateOrOlderTimestampDoesNotMutateGlobalKinematics() {
        val engine = Global4DConsistencyEngine()

        val first =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(
                        timestampNanos = 5_100_000_000L,
                        rootX = 0.0,
                        rootVelocityX = 0.0,
                    ),
                    physicsScore = 0.94,
                    physicsAccepted = true,
                    reprojectionScore = 0.95,
                    temporalScore = 0.96,
                    synchronizationConfidence = 0.97,
                    rigTrust = 0.96,
                    cameraMotionRisk = 0.04,
                ),
            )

        val duplicate =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(
                        timestampNanos = 5_100_000_000L,
                        rootX = 4.0,
                        rootVelocityX = 120.0,
                    ),
                    physicsScore = 0.20,
                    physicsAccepted = false,
                    reprojectionScore = 0.20,
                    temporalScore = 0.20,
                    synchronizationConfidence = 0.20,
                    rigTrust = 0.20,
                    cameraMotionRisk = 0.95,
                ),
            )

        val older =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(
                        timestampNanos = 5_099_500_000L,
                        rootX = -4.0,
                        rootVelocityX = -120.0,
                    ),
                    physicsScore = 0.10,
                    physicsAccepted = false,
                    reprojectionScore = 0.10,
                    temporalScore = 0.10,
                    synchronizationConfidence = 0.10,
                    rigTrust = 0.10,
                    cameraMotionRisk = 1.0,
                ),
            )

        assertTrue(duplicate.timestampNanos == first.timestampNanos)
        assertTrue(older.timestampNanos == first.timestampNanos)
        assertTrue(
            duplicate.rootSpeedMetersPerSecond ==
                first.rootSpeedMetersPerSecond,
        )
        assertTrue(
            older.rootSpeedMetersPerSecond ==
                first.rootSpeedMetersPerSecond,
        )
        assertTrue(
            duplicate.warnings.any {
                it.contains("timestamp") &&
                    it.contains("ignored")
            },
        )
        assertTrue(
            older.warnings.any {
                it.contains("timestamp") &&
                    it.contains("ignored")
            },
        )
    }

    @Test
    fun resetClearsGlobalTemporalState() {
        val engine = Global4DConsistencyEngine()
        engine.evaluate(
            Global4DConsistencyEngine.Input(
                skeleton = skeleton(5_000_000_000L),
                physicsScore = 0.92,
                physicsAccepted = true,
                reprojectionScore = 0.94,
                temporalScore = 0.95,
                synchronizationConfidence = 0.96,
                rigTrust = 0.96,
                cameraMotionRisk = 0.05,
            ),
        )

        engine.reset()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(5_033_000_000L),
                    physicsScore = 0.92,
                    physicsAccepted = true,
                    reprojectionScore = 0.94,
                    temporalScore = 0.95,
                    synchronizationConfidence = 0.96,
                    rigTrust = 0.96,
                    cameraMotionRisk = 0.05,
                ),
            )

        assertTrue(
            result.state ==
                Global4DConsistencyEngine.State.WARMING ||
                result.state ==
                Global4DConsistencyEngine.State.STABLE,
        )
    }

    @Test
    fun bodyScaleContinuityDetectsSuddenMetricDrift() {
        val engine = Global4DConsistencyEngine()

        engine.evaluate(
            Global4DConsistencyEngine.Input(
                skeleton = skeleton(5_400_000_000L),
                physicsScore = 0.96,
                physicsAccepted = true,
                reprojectionScore = 0.96,
                temporalScore = 0.96,
                synchronizationConfidence = 0.97,
                rigTrust = 0.97,
                cameraMotionRisk = 0.03,
            ),
        )

        val source = skeleton(5_433_000_000L)
        val scaled = source.copy(
            joints = source.joints.map { joint ->
                joint.copy(
                    position =
                        Vec3(
                            joint.position.x * 1.45,
                            joint.position.y * 1.45,
                            joint.position.z,
                        ),
                )
            },
        )

        val result = engine.evaluate(
            Global4DConsistencyEngine.Input(
                skeleton = scaled,
                physicsScore = 0.96,
                physicsAccepted = true,
                reprojectionScore = 0.96,
                temporalScore = 0.96,
                synchronizationConfidence = 0.97,
                rigTrust = 0.97,
                cameraMotionRisk = 0.03,
            ),
        )

        assertTrue(result.bodyScaleConsistency < 0.10)
        assertTrue(
            result.warnings.any {
                it.contains("body-scale")
            },
        )
    }

    @Test
    fun physicalRigAndDynamicMetrologyEvidenceCanBlockGlobalAcceptance() {
        val engine = Global4DConsistencyEngine()

        val result =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(5_600_000_000L),
                    physicsScore = 0.96,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.97,
                    synchronizationConfidence = 0.98,
                    rigTrust = 0.96,
                    cameraMotionRisk = 0.03,
                    rigStabilityConfidence = 0.38,
                    rigTranslationDriftMeters = 0.004,
                    rigRotationDriftRadians = 0.005,
                    rigUnstable = true,
                    dynamicMetrologyP95Millimeters = 1.8,
                    dynamicMetrologyRmseMillimeters = 1.3,
                    dynamicMetrologyMaxErrorMillimeters = 2.8,
                    dynamicMetrologyConfidence = 0.82,
                    dynamicMetrologyValid = true,
                ),
            )

        assertTrue(result.rigStabilityConsistency < 0.20)
        assertTrue(result.dynamicMetrologyConsistency < 0.45)
        assertTrue(!result.accepted)
        assertTrue(
            result.warnings.any {
                it.contains("rig-stability")
            },
        )
        assertTrue(
            result.warnings.any {
                it.contains("dynamic-metrology")
            },
        )
    }


    @Test
    fun gaussianDeformationEvidenceImprovesOrBlocksGlobalConsistency() {
        val engine = Global4DConsistencyEngine()

        val healthy =
            engine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(5_800_000_000L),
                    physicsScore = 0.96,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.97,
                    synchronizationConfidence = 0.98,
                    rigTrust = 0.97,
                    cameraMotionRisk = 0.03,
                    gaussianTemporalStability = 0.94,
                    gaussianTrustedRatio = 0.91,
                    gaussianMeanNodeConfidence = 0.95,
                    gaussianMeanBindingConfidence = 0.94,
                    gaussianP95StrainRatio = 0.08,
                    gaussianMaximumStrainRatio = 0.16,
                    gaussianEvidenceAvailable = true,
                ),
            )

        assertTrue(healthy.gaussianEvidenceAvailable)
        assertTrue(healthy.gaussian4DConsistency > 0.85)
        assertTrue(healthy.gaussianStrainConsistency > 0.85)
        assertTrue(healthy.accepted)

        val weak =
            Global4DConsistencyEngine().evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = skeleton(5_800_000_000L),
                    physicsScore = 0.96,
                    physicsAccepted = true,
                    reprojectionScore = 0.96,
                    temporalScore = 0.97,
                    synchronizationConfidence = 0.98,
                    rigTrust = 0.97,
                    cameraMotionRisk = 0.03,
                    gaussianTemporalStability = 0.34,
                    gaussianTrustedRatio = 0.22,
                    gaussianMeanNodeConfidence = 0.44,
                    gaussianMeanBindingConfidence = 0.38,
                    gaussianP95StrainRatio = 0.44,
                    gaussianMaximumStrainRatio = 0.88,
                    gaussianEvidenceAvailable = true,
                ),
            )

        assertTrue(weak.gaussian4DConsistency < 0.60)
        assertTrue(weak.gaussianStrainConsistency < 0.20)
        assertTrue(!weak.accepted)
        assertTrue(
            weak.warnings.any {
                it.contains("gaussian")
            },
        )
    }


}
