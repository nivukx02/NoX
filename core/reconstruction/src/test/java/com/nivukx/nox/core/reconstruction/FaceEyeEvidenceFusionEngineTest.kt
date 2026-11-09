package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertTrue
import org.junit.Test

class FaceEyeEvidenceFusionEngineTest {

    @Test
    fun healthyFaceAndBinocularEyeEvidenceProducesHighFusionConfidence() {
        val engine = FaceEyeEvidenceFusionEngine()

        val result =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 1_000_000_000L,
                    faceGeometryConfidence = 0.94,
                    faceTemporalConfidence = 0.92,
                    denseMeshConfidence = 0.90,
                    faceUncertaintyMeters = 0.018,
                    leftEyeConfidence = 0.92,
                    rightEyeConfidence = 0.91,
                    binocularConfidence = 0.94,
                    eyeTemporalConfidence = 0.93,
                    eyeUncertaintyMeters = 0.012,
                    angularSpreadDegrees = 1.4,
                    parallaxDegrees = 2.4,
                    saccadeVelocityDegreesPerSecond = 120.0,
                    blinkProbability = 0.08,
                ),
            )

        assertTrue(result.faceConfidence > 0.80)
        assertTrue(result.eyeConfidence > 0.80)
        assertTrue(result.fusedConfidence > 0.82)
        assertTrue(result.accepted)
        assertTrue(result.selectedSources.contains("FACE_TEMPORAL"))
        assertTrue(result.selectedSources.contains("BINOCULAR_EYE"))
    }

    @Test
    fun binocularFailureAttributionBecomesDominantWhenGeometryCollapses() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(
                    12_000_000_000L,
                ).copy(
                    binocularConfidence = 0.12,
                    angularSpreadDegrees = 15.0,
                    parallaxDegrees = 0.20,
                    eyeTemporalConfidence = 0.74,
                ),
            )

        val dominant = result.dominantFailure

        assertTrue(dominant != null)
        assertTrue(
            dominant!!.domain ==
                FaceEyeEvidenceFusionEngine.FailureDomain.EYE_BINOCULAR ||
                dominant.domain ==
                FaceEyeEvidenceFusionEngine.FailureDomain.EYE_UNCERTAINTY,
        )
        assertTrue(dominant.severity > 0.55)
        assertTrue(
            dominant.rationale.isNotBlank(),
        )
    }

    @Test
    fun faceOodAttributionDoesNotEraseHealthyEyeEvidence() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(
                    12_033_000_000L,
                ).copy(
                    faceOutOfDistributionRisk = 0.97,
                    eyeOutOfDistributionRisk = 0.03,
                    leftEyeConfidence = 0.95,
                    rightEyeConfidence = 0.95,
                    binocularConfidence = 0.96,
                    eyeTemporalConfidence = 0.95,
                ),
            )

        assertTrue(
            result.failureAttribution.any {
                it.domain ==
                    FaceEyeEvidenceFusionEngine.FailureDomain.FACE_OOD
            },
        )
        assertTrue(result.eyeConfidence > result.faceConfidence)
        assertTrue(result.eyeConfidence > 0.70)
    }

    @Test
    fun healthyEvidenceDoesNotCreateMeaningfulFailureAttribution() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(
                    12_066_000_000L,
                ),
            )

        assertTrue(
            result.failureAttribution.none {
                it.severity >= 0.70 &&
                    it.blocking
            },
        )
    }

    @Test
    fun lowQualityFaceCanDeprioritizeDenseFaceEvidenceWithoutKillingReliableEyes() {
        val engine = FaceEyeEvidenceFusionEngine()

        val result =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 2_000_000_000L,
                    faceGeometryConfidence = 0.31,
                    faceTemporalConfidence = 0.24,
                    denseMeshConfidence = 0.28,
                    faceUncertaintyMeters = 0.24,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.89,
                    binocularConfidence = 0.91,
                    eyeTemporalConfidence = 0.90,
                    eyeUncertaintyMeters = 0.018,
                    angularSpreadDegrees = 2.0,
                    parallaxDegrees = 1.7,
                    saccadeVelocityDegreesPerSecond = 180.0,
                    blinkProbability = 0.10,
                ),
            )

        assertTrue(result.faceConfidence < result.eyeConfidence)
        assertTrue(result.eyeConfidence > 0.70)
        assertTrue(
            result.warnings.any {
                it.contains("face") &&
                    it.contains("deprioritized")
            },
        )
        assertTrue(result.accepted)
    }

    @Test
    fun binocularDisagreementCanRejectEyeFusionEvenWhenIndividualEyesLookStrong() {
        val engine = FaceEyeEvidenceFusionEngine(
            FaceEyeEvidenceFusionEngine.Config(
                rejectFusedConfidence = 0.46,
                binocularRejectThreshold = 0.42,
            ),
        )

        val result =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 3_000_000_000L,
                    faceGeometryConfidence = 0.86,
                    faceTemporalConfidence = 0.82,
                    denseMeshConfidence = 0.80,
                    faceUncertaintyMeters = 0.03,
                    leftEyeConfidence = 0.93,
                    rightEyeConfidence = 0.92,
                    binocularConfidence = 0.18,
                    eyeTemporalConfidence = 0.72,
                    eyeUncertaintyMeters = 0.025,
                    angularSpreadDegrees = 11.5,
                    parallaxDegrees = 0.32,
                    saccadeVelocityDegreesPerSecond = 190.0,
                    blinkProbability = 0.08,
                ),
            )

        assertTrue(result.eyeConfidence < 0.55)
        assertTrue(!result.accepted)
        assertTrue(
            result.warnings.any {
                it.contains("binocular")
            },
        )
    }

    @Test
    fun rapidSaccadeIsPenalizedWithoutBeingTreatedAsPermanentFailure() {
        val engine = FaceEyeEvidenceFusionEngine()

        val baseline =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 4_000_000_000L,
                    faceGeometryConfidence = 0.86,
                    faceTemporalConfidence = 0.84,
                    denseMeshConfidence = 0.83,
                    faceUncertaintyMeters = 0.03,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.90,
                    binocularConfidence = 0.90,
                    eyeTemporalConfidence = 0.88,
                    eyeUncertaintyMeters = 0.025,
                    angularSpreadDegrees = 2.2,
                    parallaxDegrees = 1.8,
                    saccadeVelocityDegreesPerSecond = 70.0,
                    blinkProbability = 0.08,
                ),
            )

        val saccade =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 4_033_000_000L,
                    faceGeometryConfidence = 0.86,
                    faceTemporalConfidence = 0.84,
                    denseMeshConfidence = 0.83,
                    faceUncertaintyMeters = 0.03,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.90,
                    binocularConfidence = 0.90,
                    eyeTemporalConfidence = 0.79,
                    eyeUncertaintyMeters = 0.025,
                    angularSpreadDegrees = 2.4,
                    parallaxDegrees = 1.8,
                    saccadeVelocityDegreesPerSecond = 710.0,
                    blinkProbability = 0.08,
                ),
            )

        assertTrue(saccade.eyeConfidence < baseline.eyeConfidence)
        assertTrue(saccade.fusedConfidence > 0.45)
        assertTrue(!saccade.warnings.any { it == "eye-hard-failure" })
    }

    @Test
    fun equivalentReconfigurationPreservesFaceEyeReliability() {
        val config =
            FaceEyeEvidenceFusionEngine.Config()
        val engine =
            FaceEyeEvidenceFusionEngine(config)

        engine.fuse(
            healthyInput(7_100_000_000L),
        )

        val faceBefore =
            engine.configuration().faceGeometryWeight

        val stableResult =
            engine.fuse(
                healthyInput(7_133_000_000L),
            )
        val before =
            stableResult.faceReliability

        engine.configure(config)

        val after =
            engine.fuse(
                healthyInput(7_166_000_000L),
            ).faceReliability

        assertTrue(faceBefore > 0.0)
        assertTrue(after >= before * 0.96)
    }

    @Test
    fun eyeFailureProducesExplicitFaceOnlyFusionMode() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(6_000_000_000L).copy(
                    binocularConfidence = 0.12,
                    angularSpreadDegrees = 14.0,
                    eyeTemporalConfidence = 0.18,
                    eyeTemporalAccepted = false,
                    binocularAccepted = false,
                ),
            )

        assertTrue(result.fusionMode ==
            FaceEyeEvidenceFusionEngine.FusionMode.FACE_ONLY)
        assertTrue(result.faceConfidence > result.eyeConfidence)
        assertTrue(result.accepted)
    }

    @Test
    fun faceFailureProducesExplicitEyeOnlyFusionMode() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(6_033_000_000L).copy(
                    faceGeometryConfidence = 0.16,
                    faceTemporalConfidence = 0.14,
                    denseMeshConfidence = 0.18,
                    faceShapeConfidence = 0.14,
                    faceShapeAccepted = false,
                    faceFreshness = 0.12,
                    faceReprojectionScore = 0.10,
                    facePointCount = 12,
                ),
            )

        assertTrue(result.fusionMode ==
            FaceEyeEvidenceFusionEngine.FusionMode.EYE_ONLY)
        assertTrue(result.eyeConfidence > result.faceConfidence)
    }

    @Test
    fun reliabilityTrendRegistersRapidChannelDegradation() {
        val engine = FaceEyeEvidenceFusionEngine()

        engine.fuse(
            healthyInput(6_066_000_000L),
        )

        val degraded =
            engine.fuse(
                healthyInput(6_099_000_000L).copy(
                    binocularConfidence = 0.20,
                    eyeTemporalConfidence = 0.22,
                    eyeUncertaintyMeters = 0.11,
                    angularSpreadDegrees = 12.0,
                    faceOutOfDistributionRisk = 0.04,
                    eyeOutOfDistributionRisk = 0.72,
                ),
            )

        assertTrue(degraded.eyeReliabilityTrend < 0.0)
        assertTrue(degraded.independentEvidenceSupport in 0.0..1.0)
    }

    @Test
    fun duplicateTimestampDoesNotCrashFusion() {
        val engine = FaceEyeEvidenceFusionEngine()

        val first =
            engine.fuse(
                healthyInput(
                    timestampNanos = 7_000_000_000L,
                ),
            )

        val repeated =
            engine.fuse(
                healthyInput(
                    timestampNanos = 7_000_000_000L,
                ),
            )

        assertTrue(repeated.timestampNanos == first.timestampNanos)
        assertTrue(repeated.fusedConfidence >= 0.0)
        assertTrue(repeated.fusedConfidence <= 1.0)
    }

    private fun healthyInput(timestampNanos: Long) =
        FaceEyeEvidenceFusionEngine.Input(
            timestampNanos = timestampNanos,
            faceGeometryConfidence = 0.92,
            faceTemporalConfidence = 0.90,
            denseMeshConfidence = 0.88,
            faceUncertaintyMeters = 0.02,
            leftEyeConfidence = 0.90,
            rightEyeConfidence = 0.90,
            binocularConfidence = 0.92,
            eyeTemporalConfidence = 0.91,
            eyeUncertaintyMeters = 0.015,
            angularSpreadDegrees = 1.8,
            parallaxDegrees = 2.1,
            saccadeVelocityDegreesPerSecond = 90.0,
            blinkProbability = 0.05,
        )

    @Test
    fun rejectedFaceShapeDoesNotDestroyHealthyBinocularEyes() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(11_000_000_000L).copy(
                    faceShapeConfidence = 0.18,
                    faceShapeAccepted = false,
                    facePointCount = 20,
                    denseMeshObserved = false,
                    leftEyeConfidence = 0.94,
                    rightEyeConfidence = 0.94,
                    binocularConfidence = 0.95,
                    eyeTemporalConfidence = 0.94,
                ),
            )

        assertTrue(result.faceHardFailure)
        assertTrue(!result.eyeHardFailure)
        assertTrue(result.eyeConfidence > result.faceConfidence)
        assertTrue(
            result.hardFailureReasons.contains("face-shape-rejected"),
        )
        assertTrue(result.accepted)
    }

    @Test
    fun rejectedEyeTemporalAndBinocularEvidenceCanFallBackToFace() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(11_033_000_000L).copy(
                    eyeTemporalAccepted = false,
                    binocularAccepted = false,
                    binocularConfidence = 0.18,
                    angularSpreadDegrees = 15.0,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.90,
                ),
            )

        assertTrue(result.eyeHardFailure)
        assertTrue(!result.faceHardFailure)
        assertTrue(
            result.hardFailureReasons.any {
                it == "eye-temporal-rejected" ||
                    it == "binocular-geometry-rejected"
            },
        )
        assertTrue(result.faceConfidence > result.eyeConfidence)
    }

    @Test
    fun highGazeAccelerationReducesDynamicConsistencyWithoutAutomaticHardReject() {
        val engine = FaceEyeEvidenceFusionEngine()

        val baseline =
            engine.fuse(
                healthyInput(11_066_000_000L),
            )

        val accelerated =
            engine.fuse(
                healthyInput(11_099_000_000L).copy(
                    gazeAccelerationDegreesPerSecond2 = 5200.0,
                    saccadeVelocityDegreesPerSecond = 760.0,
                    binocularConfidence = 0.91,
                    angularSpreadDegrees = 3.5,
                    eyeTemporalConfidence = 0.82,
                ),
            )

        assertTrue(
            accelerated.eyeDynamicConsistency <
                baseline.eyeDynamicConsistency,
        )
        assertTrue(
            accelerated.eyeHardFailure.not(),
        )
    }

    @Test
    fun repeatedBadBinocularEvidenceIsQuarantined() {
        val engine =
            FaceEyeEvidenceFusionEngine(
                FaceEyeEvidenceFusionEngine.Config(
                    quarantineAfterBadFrames = 2,
                    recoveryFrames = 3,
                ),
            )

        repeat(3) { index ->
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos =
                        5_000_000_000L +
                            index * 33_000_000L,
                    faceGeometryConfidence = 0.84,
                    faceTemporalConfidence = 0.82,
                    denseMeshConfidence = 0.80,
                    faceUncertaintyMeters = 0.03,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.90,
                    binocularConfidence = 0.16,
                    eyeTemporalConfidence = 0.74,
                    eyeUncertaintyMeters = 0.04,
                    angularSpreadDegrees = 13.0,
                    parallaxDegrees = 0.25,
                    saccadeVelocityDegreesPerSecond = 130.0,
                    blinkProbability = 0.08,
                ),
            )
        }

        val result =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 5_099_000_000L,
                    faceGeometryConfidence = 0.84,
                    faceTemporalConfidence = 0.82,
                    denseMeshConfidence = 0.80,
                    faceUncertaintyMeters = 0.03,
                    leftEyeConfidence = 0.90,
                    rightEyeConfidence = 0.90,
                    binocularConfidence = 0.16,
                    eyeTemporalConfidence = 0.74,
                    eyeUncertaintyMeters = 0.04,
                    angularSpreadDegrees = 13.0,
                    parallaxDegrees = 0.25,
                    saccadeVelocityDegreesPerSecond = 130.0,
                    blinkProbability = 0.08,
                ),
            )

        assertTrue(result.eyeQuarantined)
        assertTrue(
            result.warnings.any {
                it.contains("eye") &&
                    it.contains("quarantined")
            },
        )
    }
    @Test
    fun empiricalUncertaintyCalibrationBecomesPartOfFusionResult() {
        val engine = FaceEyeEvidenceFusionEngine()
        engine.configureUncertaintyCalibration(
            FaceEyeUncertaintyCalibrationEngine.Config(
                minimumCalibrationSamples = 5,
                maximumCalibrationSamples = 24,
                targetCoverage = 0.90,
            ),
        )

        var result =
            engine.fuse(
                healthyInput(
                    8_000_000_000L,
                ),
            )

        repeat(12) { index ->
            result =
                engine.fuse(
                    healthyInput(
                        8_033_000_000L +
                            index * 33_000_000L,
                    ).copy(
                        faceUncertaintyMeters =
                            0.018 + index * 0.0002,
                        eyeUncertaintyMeters =
                            0.010 + index * 0.0001,
                    ),
                )
        }

        assertTrue(result.uncertaintyCalibrationReady)
        assertTrue(
            result.faceUncertaintyCalibrationMultiplier > 0.0,
        )
        assertTrue(
            result.eyeUncertaintyCalibrationMultiplier > 0.0,
        )
        assertTrue(
            result.faceUncertaintyCoverage in 0.0..1.0,
        )
        assertTrue(
            result.eyeUncertaintyCoverage in 0.0..1.0,
        )
        assertTrue(
            result.uncertaintyCalibrationConfidence >= 0.0,
        )
    }

    @Test
    fun lowQualityCalibrationFramesDoNotBecomeReliableFaceEyeEvidence() {
        val engine = FaceEyeEvidenceFusionEngine()
        engine.configureUncertaintyCalibration(
            FaceEyeUncertaintyCalibrationEngine.Config(
                minimumCalibrationSamples = 4,
                maximumCalibrationSamples = 24,
            ),
        )

        repeat(8) { index ->
            engine.fuse(
                healthyInput(
                    8_500_000_000L +
                        index * 33_000_000L,
                ),
            )
        }

        val result =
            engine.fuse(
                healthyInput(
                    8_900_000_000L,
                ).copy(
                    faceGeometryConfidence = 0.18,
                    faceTemporalConfidence = 0.20,
                    faceReprojectionScore = 0.16,
                    faceShapeConfidence = 0.19,
                    faceUncertaintyMeters = 0.004,
                    leftEyeConfidence = 0.20,
                    rightEyeConfidence = 0.18,
                    binocularConfidence = 0.19,
                    eyeTemporalConfidence = 0.21,
                    angularSpreadDegrees = 10.8,
                    eyeUncertaintyMeters = 0.004,
                ),
            )

        assertTrue(
            result.warnings.any {
                it.contains("calibration")
            } ||
                result.fusedConfidence < 0.70,
        )
        assertTrue(
            result.faceUncertaintyCalibrationMultiplier >= 0.70,
        )
        assertTrue(
            result.eyeUncertaintyCalibrationMultiplier >= 0.70,
        )
    }

    @Test
    fun elevatedOutOfDistributionRiskDeprioritizesFaceEvidence() {
        val engine = FaceEyeEvidenceFusionEngine()

        val healthy =
            engine.fuse(
                healthyInput(
                    8_900_000_000L,
                ),
            )

        engine.reset()

        val ood =
            engine.fuse(
                healthyInput(
                    8_933_000_000L,
                ).copy(
                    faceOutOfDistributionRisk = 0.92,
                    eyeOutOfDistributionRisk = 0.18,
                ),
            )

        assertTrue(
            ood.faceReliability < healthy.faceReliability,
        )
        assertTrue(
            ood.warnings.any {
                it.contains("out-of-distribution")
            },
        )
    }

    @Test
    fun reliableEyesCanSurviveFaceOutOfDistributionEvidence() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(
                    8_966_000_000L,
                ).copy(
                    faceGeometryConfidence = 0.92,
                    faceTemporalConfidence = 0.91,
                    denseMeshConfidence = 0.90,
                    faceOutOfDistributionRisk = 0.96,
                    leftEyeConfidence = 0.94,
                    rightEyeConfidence = 0.94,
                    binocularConfidence = 0.95,
                    eyeTemporalConfidence = 0.94,
                    eyeOutOfDistributionRisk = 0.05,
                ),
            )

        assertTrue(result.eyeConfidence > result.faceConfidence)
        assertTrue(result.eyeConfidence > 0.70)
    }

    @Test
    fun staleFaceAndPoorReprojectionAreHardEvidenceFailures() {
        val engine = FaceEyeEvidenceFusionEngine()

        val result =
            engine.fuse(
                FaceEyeEvidenceFusionEngine.Input(
                    timestampNanos = 9_000_000_000L,
                    faceGeometryConfidence = 0.92,
                    faceTemporalConfidence = 0.90,
                    denseMeshConfidence = 0.88,
                    faceUncertaintyMeters = 0.02,
                    leftEyeConfidence = 0.91,
                    rightEyeConfidence = 0.90,
                    binocularConfidence = 0.92,
                    eyeTemporalConfidence = 0.91,
                    eyeUncertaintyMeters = 0.012,
                    angularSpreadDegrees = 1.8,
                    parallaxDegrees = 2.1,
                    saccadeVelocityDegreesPerSecond = 110.0,
                    blinkProbability = 0.04,
                    faceFreshness = 0.10,
                    faceReprojectionScore = 0.05,
                    gazeStability = 0.94,
                    gazeAgreement = 0.95,
                ),
            )

        assertTrue(
            result.warnings.any {
                it.contains("face-geometry-hard-failure")
            },
        )
        assertTrue(
            result.faceFreshnessScore < 0.20,
        )
        assertTrue(
            !result.accepted,
        )
    }

    @Test
    fun lowQualityIpdEvidenceCannotSeedTheMetricBaseline() {
        val engine =
            FaceEyeEvidenceFusionEngine(
                FaceEyeEvidenceFusionEngine.Config(
                    ipdCalibrationWarmupSamples = 3,
                    maximumIpdDeviationMeters = 0.010,
                ),
            )

        repeat(3) { index ->
            engine.fuse(
                healthyInput(
                    10_000_000_000L +
                        index * 33_000_000L,
                ).copy(
                    leftEyeConfidence = 0.12,
                    rightEyeConfidence = 0.11,
                    binocularConfidence = 0.08,
                    angularSpreadDegrees = 18.0,
                    parallaxDegrees = 0.10,
                    interpupillaryDistanceMeters = 0.100,
                ),
            )
        }

        val recovered =
            engine.fuse(
                healthyInput(10_120_000_000L).copy(
                    interpupillaryDistanceMeters = 0.064,
                    binocularConfidence = 0.94,
                    leftEyeConfidence = 0.93,
                    rightEyeConfidence = 0.93,
                    angularSpreadDegrees = 1.5,
                    parallaxDegrees = 2.2,
                ),
            )

        assertTrue(
            recovered.ipdConsistencyScore > 0.70,
        )
        assertTrue(
            recovered.geometricIntegrityScore > 0.70,
        )
    }

    @Test
    fun learnedIpdConsistencyImprovesStableBinocularEvidence() {
        val engine = FaceEyeEvidenceFusionEngine(
            FaceEyeEvidenceFusionEngine.Config(
                ipdCalibrationWarmupSamples = 3,
                maximumIpdDeviationMeters = 0.010,
            ),
        )

        repeat(4) { index ->
            engine.fuse(
                healthyInput(
                    9_200_000_000L + index * 33_000_000L,
                ).copy(
                    interpupillaryDistanceMeters = 0.062,
                ),
            )
        }

        val stable =
            engine.fuse(
                healthyInput(9_350_000_000L).copy(
                    interpupillaryDistanceMeters = 0.063,
                ),
            )

        val drifted =
            engine.fuse(
                healthyInput(9_383_000_000L).copy(
                    interpupillaryDistanceMeters = 0.080,
                ),
            )

        assertTrue(
            stable.ipdConsistencyScore >
                drifted.ipdConsistencyScore,
        )
        assertTrue(
            stable.geometricIntegrityScore >
                drifted.geometricIntegrityScore,
        )
        assertTrue(
            drifted.warnings.any {
                it.contains("ipd")
            },
        )
    }

    @Test
    fun lowLightingIsExplicitlyAttributedToCaptureDomain() {
        val result =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(11_000_000_000L).copy(
                    lightingConfidence = 0.18,
                    faceScaleConfidence = 0.88,
                ),
            )

        assertTrue(
            result.failureAttribution.any {
                it.domain ==
                    FaceEyeEvidenceFusionEngine.FailureDomain.CAPTURE_DOMAIN
            },
        )
        assertTrue(
            result.warnings.any {
                it.contains("lighting")
            },
        )
    }

    @Test
    fun smallFaceDomainDeprioritizesFaceEvidenceWithoutDeletingEyes() {
        val bright =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(11_033_000_000L).copy(
                    faceScaleConfidence = 0.96,
                    lightingConfidence = 0.96,
                ),
            )

        val small =
            FaceEyeEvidenceFusionEngine().fuse(
                healthyInput(11_066_000_000L).copy(
                    faceScaleConfidence = 0.22,
                    lightingConfidence = 0.96,
                ),
            )

        assertTrue(
            small.faceConfidence <= bright.faceConfidence,
        )
        assertTrue(
            small.eyeConfidence > 0.65,
        )
        assertTrue(
            small.warnings.any {
                it.contains("scale")
            },
        )
    }

    @Test
    fun highGazeStabilityImprovesFusionEvidence() {
        val engine = FaceEyeEvidenceFusionEngine()

        val unstable =
            engine.fuse(
                healthyInput(9_100_000_000L).copy(
                    gazeStability = 0.30,
                    gazeAgreement = 0.35,
                ),
            )

        val stable =
            engine.fuse(
                healthyInput(9_133_000_000L).copy(
                    gazeStability = 0.96,
                    gazeAgreement = 0.97,
                ),
            )

        assertTrue(
            stable.fusedConfidence >=
                unstable.fusedConfidence,
        )
        assertTrue(
            stable.geometricIntegrityScore >
                unstable.geometricIntegrityScore,
        )
    }

}
