package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HighEndMotionCaptureSettingsTest {
    @Test
    fun photometricExposurePolicy_survivesSettingsToCustomizationConversion() {
        val settings =
            HighEndMotionCaptureSettings(
                photometricExposureShiftSoftLimit = 0.18,
                photometricExposureShiftHardLimit = 0.52,
                photometricExposureReliabilityFloor = 0.42,
            )

        val customization =
            settings.toMotionCaptureCustomization()

        assertEquals(
            0.18,
            customization.photometricExposureShiftSoftLimit,
            1e-9,
        )
        assertEquals(
            0.52,
            customization.photometricExposureShiftHardLimit,
            1e-9,
        )
        assertEquals(
            0.42,
            customization.photometricExposureReliabilityFloor,
            1e-9,
        )
    }

    @Test
    fun clampsSupportedFpsWithoutBreakingOtherSettings() {
        val base = HighEndMotionCaptureSettings()
        val updated = base.withTargetFps(240)
        assertEquals(120, updated.targetFps)
        assertEquals(base.motionSensitivity, updated.motionSensitivity, 0.0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsImpossibleTemporalSkew() {
        HighEndMotionCaptureSettings(
            preferredTemporalSkewMs = 10.0,
            maximumTemporalSkewMs = 4.0,
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejects_conflicting_quality_policies() {
        HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = true,
            qualityLockEnabled = true,
        )
    }

    @Test
    fun customization_helpers_are_bounded() {
        val settings = HighEndMotionCaptureSettings()
            .withProfile("not-a-real-profile")
            .withMicroMotionPreservation(5.0)
            .withSurfaceDetail(-1.0)
            .withFaceEyeDetail(5.0)
            .withHandDetail(-1.0)

        assert(settings.profileId == ProductionCaptureProfile.default.id)
        assert(settings.preserveMicroMotion == 1.0)
        assert(settings.surfaceDetailPriority == 0.0)
        assert(settings.faceEyeDetailPriority == 1.0)
        assert(settings.handDetailPriority == 0.0)
    }

    @Test
    fun precision_contract_helper_preserves_bounds() {
        val settings = HighEndMotionCaptureSettings().withPrecisionContract(
            p95Millimeters = 0.01,
            lagMilliseconds = 500.0,
            dropoutRatio = 0.90,
            calibrationConfidence = -1.0,
            viewDiversity = 2.0,
            crossViewAgreement = -1.0,
        )

        assertEquals(0.1, settings.targetMotionP95ErrorMillimeters, 0.0)
        assertEquals(250.0, settings.maximumTemporalLagMilliseconds, 0.0)
        assertEquals(0.50, settings.maximumDropoutRatio, 0.0)
        assertEquals(0.0, settings.minimumCalibrationConfidence, 0.0)
        assertEquals(1.0, settings.minimumViewDiversity, 0.0)
        assertEquals(0.0, settings.minimumCrossViewAgreement, 0.0)
    }

    @Test
    fun high_speed_settings_and_crowd_controls_remain_bounded() {
        val settings =
            HighEndMotionCaptureSettings(
                targetFps = 240,
                inferenceFps = 240,
                adaptiveAuthorityDepthEnabled = true,
                maximumAdaptiveAuthorityBonus = 8,
                adaptiveAuthorityDifficultyThreshold = 0.62,
                adaptiveAuthorityPriorityThreshold = 0.93,
                maximumAdaptiveAuthorityCamerasPerActor = 32,
                crowdedSceneOverlapThreshold = 0.55,
                crowdedSceneMatchScoreBoost = 0.28,
            )

        assertEquals(240, settings.targetFps)
        assertEquals(240, settings.inferenceFps)
        assertTrue(settings.adaptiveAuthorityDepthEnabled)
        assertEquals(8, settings.maximumAdaptiveAuthorityBonus)
        assertEquals(
            0.62,
            settings.adaptiveAuthorityDifficultyThreshold,
            1e-9,
        )
        assertEquals(
            0.93,
            settings.adaptiveAuthorityPriorityThreshold,
            1e-9,
        )
        assertEquals(
            32,
            settings.maximumAdaptiveAuthorityCamerasPerActor,
        )
        assertEquals(
            0.55,
            settings.crowdedSceneOverlapThreshold,
            1e-9,
        )
        assertEquals(
            0.28,
            settings.crowdedSceneMatchScoreBoost,
            1e-9,
        )
    }

    @Test
    fun runtimeCustomizationCarriesTemporalAndQualityIntent() {
        val settings = HighEndMotionCaptureSettings(
            targetFps = 90,
            inferenceFps = 45,
            motionSensitivity = 0.92,
            preserveMicroMotion = 0.96,
            temporalContinuityPriority = 0.94,
            faceEyeDetailPriority = 0.98,
            handDetailPriority = 0.93,
            surfaceDetailPriority = 0.91,
            interleavedTemporalFusionEnabled = true,
            interleavedTemporalWindowMillis = 70.0,
            interleavedPredictionHorizonMillis = 14.0,
            interleavedMaximumSampleSkewMillis = 18.0,
            interleavedOutlierThresholdPixels = 16.0,
            temporalSupersamplingPriority = 0.97,
        )

        val runtime = settings.toMotionCaptureCustomization()

        assertEquals(90, runtime.targetFps)
        assertEquals(45, runtime.inferenceFps)
        assertEquals(0.96, runtime.actorMicroMotionRetention, 1e-9)
        assertEquals(0.94, runtime.actorSubframeTimingStrictness, 1e-9)
        assertEquals(0.98, runtime.faceDetail, 1e-9)
        assertEquals(0.93, runtime.handDetail, 1e-9)
        assertEquals(0.91, runtime.denseMeshDetail, 1e-9)
        assertTrue(runtime.interleavedTemporalFusionEnabled)
        assertEquals(70.0, runtime.interleavedTemporalWindowMillis, 1e-9)
        assertEquals(14.0, runtime.interleavedPredictionHorizonMillis, 1e-9)
        assertEquals(18.0, runtime.interleavedMaximumSampleSkewMillis, 1e-9)
        assertEquals(16.0, runtime.interleavedOutlierThresholdPixels, 1e-9)
        assertEquals(0.97, runtime.temporalSupersamplingPriority, 1e-9)
    }

    @Test
    fun adaptive_actor_authority_controls_reach_runtime_customization() {
        val settings =
            HighEndMotionCaptureSettings(
                adaptiveAuthorityDepthEnabled = false,
                maximumAdaptiveAuthorityBonus = 6,
                adaptiveAuthorityDifficultyThreshold = 0.58,
                adaptiveAuthorityPriorityThreshold = 0.91,
                maximumAdaptiveAuthorityCamerasPerActor = 20,
                crowdedSceneOverlapThreshold = 0.42,
                crowdedSceneMatchScoreBoost = 0.22,
            )

        val runtime =
            settings.toMotionCaptureCustomization()

        assertTrue(!settings.adaptiveAuthorityDepthEnabled)
        assertEquals(
            6,
            settings.maximumAdaptiveAuthorityBonus,
        )
        assertEquals(
            0.58,
            settings.adaptiveAuthorityDifficultyThreshold,
            1e-9,
        )
        assertEquals(
            0.91,
            settings.adaptiveAuthorityPriorityThreshold,
            1e-9,
        )
        assertEquals(
            20,
            settings.maximumAdaptiveAuthorityCamerasPerActor,
        )
        assertEquals(
            0.42,
            runtime.crowdedSceneOverlapThreshold,
            1e-9,
        )
        assertEquals(
            0.22,
            runtime.crowdedSceneMatchScoreBoost,
            1e-9,
        )
    }

    @Test
    fun distributedTransportHeadroomReachesRuntimeCustomization() {
        val settings =
            HighEndMotionCaptureSettings(
                distributedMinimumTransportHeadroom = 0.88,
            )

        val runtime =
            settings.toMotionCaptureCustomization()

        assertEquals(
            0.88,
            runtime.distributedMinimumTransportHeadroom,
            1e-9,
        )
    }

    @Test
    fun extendedProductionControlsReachRuntimeCustomization() {
        val settings = HighEndMotionCaptureSettings(
            occlusionRecoveryPriority = 0.91,
            interactionContactPriority = 0.93,
            cameraDiversityPriority = 0.94,
            faceExpressionPriority = 0.95,
            handContactPriority = 0.96,
            identityPersistencePriority = 0.97,
            captureIntegrityStrictness = 0.98,
        )

        val runtime = settings.toMotionCaptureCustomization()

        assertEquals(
            0.91,
            runtime.actorOcclusionRecovery,
            1e-9,
        )
        assertEquals(
            0.97,
            runtime.actorIdentityStrictness,
            1e-9,
        )
        assertEquals(
            0.94,
            runtime.actorCrossCameraAssociation,
            1e-9,
        )
        assertEquals(
            0.95,
            runtime.faceDetail,
            1e-9,
        )
        assertEquals(
            0.96,
            runtime.handDetail,
            1e-9,
        )
        assertEquals(
            0.93,
            runtime.contactDetail,
            1e-9,
        )
        assertEquals(
            0.98,
            runtime.outputIntegrityStrictness,
            1e-9,
        )
    }

    @Test
    fun temporal_fusion_controls_remain_bounded_and_consistent() {
        val configured = HighEndMotionCaptureSettings()
            .withTemporalFusionPredictionBudget(220.0)
            .withTemporalFusionTrackRetention(180.0)
            .withTemporalFusionOutlierGate(4.5)
            .withTemporalFusionHuberDelta(2.2)
            .withTemporalFusionConfidenceHalfLife(600.0)
            .withTemporalFusionCapacity(
                maxTracks = 420,
                maxObservationsPerFrame = 7000,
            )

        assertEquals(220.0, configured.temporalFusionPredictionBudgetMillis, 1e-9)
        assertEquals(
            220.0,
            configured.temporalFusionTrackRetentionMillis,
            1e-9,
        )
        assertEquals(4.5, configured.temporalFusionHardOutlierGate, 1e-9)
        assertEquals(2.2, configured.temporalFusionHuberDelta, 1e-9)
        assertEquals(
            600.0,
            configured.temporalFusionConfidenceHalfLifeMillis,
            1e-9,
        )
        assertEquals(420, configured.temporalFusionMaxTracks)
        assertEquals(
            7000,
            configured.temporalFusionMaxObservationsPerFrame,
        )
    }


    @Test
    fun temporal_interaction_contract_helper_is_bounded() {
        val settings =
            HighEndMotionCaptureSettings()
                .withTemporalInteractionContract(
                    maximumRollingShutterReadoutMilliseconds = 120.0,
                    minimumHandObjectContactConfidence = 2.0,
                    maximumContactPenetrationMillimeters = 0.01,
                )

        assertEquals(
            80.0,
            settings.maximumRollingShutterReadoutMilliseconds,
            0.0,
        )
        assertEquals(
            1.0,
            settings.minimumHandObjectContactConfidence,
            0.0,
        )
        assertEquals(
            0.5,
            settings.maximumContactPenetrationMillimeters,
            0.0,
        )
    }

    @Test
    fun surface_and_runtime_fault_contract_helpers_are_bounded() {
        val settings =
            HighEndMotionCaptureSettings()
                .withSurfaceContract(
                    maximumStrainP95Ratio = 2.0,
                    minimumTemporalStability = -1.0,
                    minimumTopologyContinuity = 2.0,
                )
                .withRuntimeFaultBudget(100)

        assertEquals(
            1.0,
            settings.maximumSurfaceStrainP95Ratio,
            0.0,
        )
        assertEquals(
            0.0,
            settings.minimumSurfaceTemporalStability,
            0.0,
        )
        assertEquals(
            1.0,
            settings.minimumSurfaceTopologyContinuity,
            0.0,
        )
        assertEquals(32, settings.maximumRuntimeFaultsPerWindow)
    }


    @Test
    fun synchronization_contract_helper_is_bounded() {
        val settings =
            HighEndMotionCaptureSettings()
                .withSynchronizationContract(
                    residualP95Milliseconds = 80.0,
                    outlierRatio = 0.90,
                )

        assertEquals(
            50.0,
            settings.maximumSynchronizationResidualP95Ms,
            0.0,
        )
        assertEquals(
            0.50,
            settings.maximumSynchronizationOutlierRatio,
            0.0,
        )
    }

    @Test
    fun workflow_presets_enable_strict_high_end_contracts() {
        val base = HighEndMotionCaptureSettings()

        val calibration =
            base.applyWorkflowPreset(
                CaptureWorkflowPreset.CALIBRATION,
            )
        assertTrue(calibration.qualityLockEnabled)
        assertTrue(!calibration.adaptiveQualityEnabled)
        assertTrue(calibration.referenceGradePrecisionEnabled)
        assertTrue(calibration.minimumCalibrationConfidence >= 0.92)
        assertTrue(calibration.minimumViewDiversity >= 0.65)
        assertTrue(calibration.distributedMinimumTransportHeadroom >= 0.65)

        val precision =
            base.applyWorkflowPreset(
                CaptureWorkflowPreset.PRECISION_CAPTURE,
            )
        assertTrue(precision.targetFps >= 120)
        assertTrue(precision.inferenceFps >= 120)
        assertTrue(precision.preserveMicroMotion >= 0.98)
        assertTrue(precision.captureIntegrityStrictness >= 0.99)

        val interaction =
            base.applyWorkflowPreset(
                CaptureWorkflowPreset.INTERACTION_CAPTURE,
            )
        assertTrue(interaction.identityPersistencePriority >= 0.99)
        assertTrue(interaction.interactionContactPriority >= 0.98)
        assertTrue(interaction.minimumHandObjectContactConfidence >= 0.86)

        val qa =
            base.applyWorkflowPreset(
                CaptureWorkflowPreset.RECONSTRUCTION_QA,
            )
        assertTrue(qa.diagnosticsOverlayEnabled)
        assertTrue(qa.uncertaintyOverlayEnabled)
        assertTrue(qa.minimumSurfaceTopologyContinuity >= 0.90)

        val reference =
            base.applyWorkflowPreset(
                CaptureWorkflowPreset.REFERENCE_QUALIFICATION,
            )
        assertEquals(120, reference.targetFps)
        assertEquals(120, reference.inferenceFps)
        assertTrue(reference.reconstructionUncertaintyMeters <= 0.001)
        assertTrue(reference.referenceGradePrecisionEnabled)
        assertTrue(reference.qualityLockEnabled)
        assertTrue(!reference.adaptiveQualityEnabled)
        assertTrue(reference.minimumCalibrationConfidence >= 0.95)
        assertTrue(reference.minimumCrossViewAgreement >= 0.90)
        assertTrue(reference.distributedMinimumTransportHeadroom >= 0.90)
    }


    @Test
    fun precisionPresetUsesShortExposureWithoutForcingUnsupportedHardware() {
        val settings =
            HighEndMotionCaptureSettings()
                .applyWorkflowPreset(
                    CaptureWorkflowPreset.PRECISION_CAPTURE,
                )

        assertEquals(
            CameraExposureMode.MANUAL,
            settings.cameraExposureMode,
        )
        assertEquals(
            CameraDynamicRangeMode.TEN_BIT_PREFERRED,
            settings.cameraDynamicRangeMode,
        )
        assertTrue(
            settings.cameraManualExposureMicros <= 2_500L,
        )
        assertEquals(
            false,
            settings.cameraOisEnabled,
        )
        assertEquals(
            false,
            settings.cameraElectronicStabilizationEnabled,
        )
    }

    @Test
    fun referencePresetUsesEvenStricterAcquisitionTiming() {
        val settings =
            HighEndMotionCaptureSettings()
                .applyWorkflowPreset(
                    CaptureWorkflowPreset.REFERENCE_QUALIFICATION,
                )

        assertEquals(
            CameraExposureMode.MANUAL,
            settings.cameraExposureMode,
        )
        assertEquals(
            CameraDynamicRangeMode.TEN_BIT_PREFERRED,
            settings.cameraDynamicRangeMode,
        )
        assertTrue(
            settings.cameraManualExposureMicros <= 1_500L,
        )
        assertEquals(
            CameraResolutionPreset.MAX_SUPPORTED,
            settings.cameraResolutionPreset,
        )
    }

    @Test
    fun evidenceAuthorityPolicyPersistsAllProductionGates() {
        val updated =
            HighEndMotionCaptureSettings()
                .withEvidenceAuthorityPolicy(
                    maximumMotionBlurRisk = 0.31,
                    minimumEpipolarConsistency = 0.82,
                    minimumMetricEvidenceAuthority = 0.79,
                    minimumAuthorityScore = 0.88,
                    precisionMinimumScore = 0.93,
                    precisionMinimumSamples = 2_400,
                )

        assertEquals(0.31, updated.maximumMotionBlurRiskForAuthority, 1e-9)
        assertEquals(0.82, updated.minimumEpipolarConsistencyForAuthority, 1e-9)
        assertEquals(0.79, updated.minimumMetricEvidenceAuthorityForAuthority, 1e-9)
        assertEquals(0.88, updated.minimumCaptureAuthorityScore, 1e-9)
        assertEquals(0.93, updated.precisionAuthorityMinimumScore, 1e-9)
        assertEquals(2_400, updated.precisionAuthorityMinimumSamples)
    }

    @Test
    fun antiBandingModePersistsThroughSettingsToCustomization() {
        val settings =
            HighEndMotionCaptureSettings(
                cameraAntibandingMode =
                    CameraAntibandingMode.HZ_60,
            )

        val customization =
            settings.customization()

        assertEquals(
            CameraAntibandingMode.HZ_60,
            customization.cameraAntibandingMode,
        )
    }

    @Test
    fun preciseCctAndTintPersistIntoRuntimeCustomization() {
        val settings =
            HighEndMotionCaptureSettings(
                cameraPreciseColorTemperatureEnabled = true,
                cameraColorTemperatureKelvin = 3200,
                cameraColorTint = -18,
            )

        val customization = settings.toMotionCaptureCustomization()

        assertTrue(customization.cameraPreciseColorTemperatureEnabled)
        assertEquals(
            3200,
            customization.cameraColorTemperatureKelvin,
        )
        assertEquals(
            -18,
            customization.cameraColorTint,
        )
    }

}
