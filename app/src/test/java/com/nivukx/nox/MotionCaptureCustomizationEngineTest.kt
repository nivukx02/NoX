package com.nivukx.nox

import com.nivukx.nox.core.ai.AdaptiveMotionEvidenceEngine
import com.nivukx.nox.core.calibration.CalibrationFreeMetricRecoveryEngine
import com.nivukx.nox.core.reconstruction.ContactAwareReconstructionEngine
import com.nivukx.nox.core.reconstruction.SpatiotemporalCalibrationEngine
import com.nivukx.nox.core.reconstruction.SurfaceDeformationConsistencyEngine
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MotionCaptureCustomizationEngineTest {

    private val engine = MotionCaptureCustomizationEngine()

    @Test
    fun resolves_profile_and_preserves_named_intent() {
        val settings = HighEndMotionCaptureSettings()
            .withProfile("CINEMATIC_ULTRA")
            .withMicroMotionPreservation(1.0)
            .withTemporalContinuity(1.0)

        val plan = engine.resolve(settings)

        assertEquals("CINEMATIC_ULTRA", plan.profile.id)
        assertTrue(plan.tuning.motionSensitivity >= 1.0)
        assertTrue(plan.tuning.predictionHorizonSeconds >= 0.08)
        assertTrue(plan.featureAllocation.temporal >= 0.99)
    }

    @Test
    fun adaptive_policy_responds_to_pressure_without_breaking_safe_floor() {
        val plan = engine.resolve(
            HighEndMotionCaptureSettings(adaptiveQualityEnabled = true),
            MotionCaptureCustomizationEngine.RuntimeEvidence(
                thermalHeadroom = 0.30,
                gpuHeadroom = 0.34,
                memoryHeadroom = 0.60,
                synchronizationConfidence = 0.90,
                calibrationConfidence = 0.90,
                trackingConfidence = 0.90,
                connectedCameras = 4,
            ),
        )

        assertTrue(plan.adaptiveDegradationApplied)
        assertTrue(plan.adaptiveScale < 1.0)
        assertTrue(plan.adaptiveScale >= 0.42)
        assertTrue("adaptive-budget-active" in plan.reasonCodes)
    }

    @Test
    fun quality_lock_blocks_unsafe_adaptation_instead_of_silently_reducing_fidelity() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
        )

        val plan = engine.resolve(
            settings,
            MotionCaptureCustomizationEngine.RuntimeEvidence(
                thermalHeadroom = 0.25,
                gpuHeadroom = 0.40,
                memoryHeadroom = 0.55,
                synchronizationConfidence = 0.40,
                calibrationConfidence = 0.90,
                trackingConfidence = 0.90,
                connectedCameras = 4,
            ),
        )

        assertFalse(plan.captureAllowed)
        assertTrue("quality-lock-blocked" in plan.reasonCodes)
        assertEquals(1.0, plan.adaptiveScale)
    }

    @Test
    fun precision_contract_reports_and_blocks_only_when_quality_lock_is_enabled() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
        ).withPrecisionContract(
            p95Millimeters = 2.0,
            lagMilliseconds = 10.0,
            dropoutRatio = 0.01,
            calibrationConfidence = 0.85,
            viewDiversity = 0.60,
            crossViewAgreement = 0.75,
        )

        val evidence = MotionCaptureCustomizationEngine.RuntimeEvidence(
            thermalHeadroom = 0.95,
            gpuHeadroom = 0.95,
            memoryHeadroom = 0.95,
            synchronizationConfidence = 0.95,
            calibrationConfidence = 0.80,
            trackingConfidence = 0.95,
            motionP95ErrorMillimeters = 4.0,
            temporalLagMilliseconds = 20.0,
            dropoutRatio = 0.04,
            viewDiversity = 0.30,
            crossViewAgreement = 0.40,
            connectedCameras = 4,
        )

        val plan = engine.resolve(settings, evidence)

        assertFalse(plan.captureAllowed)
        assertTrue("precision-contract-breached" in plan.reasonCodes)
        assertTrue("quality-lock-blocked" in plan.reasonCodes)
    }

    @Test
    fun motion_evidence_is_promoted_into_runtime_capture_policy() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = true,
            qualityLockEnabled = false,
        )
        val points0 = (0 until 16).map {
            AdaptiveMotionEvidenceEngine.Landmark(
                id = it,
                x = it / 20.0,
                y = it / 40.0,
                confidence = 0.96,
            )
        }
        val points1 = (0 until 16).map {
            AdaptiveMotionEvidenceEngine.Landmark(
                id = it,
                x = it / 20.0 + if (it % 2 == 0) 0.055 else -0.055,
                y = it / 40.0,
                confidence = 0.96,
            )
        }
        val input0 = AdaptiveMotionEvidenceEngine.Input(
            timestampNanos = 1_000_000_000L,
            landmarks = points0,
            nominalFrameRateHz = 60.0,
        )
        val input1 = input0.copy(
            timestampNanos = 1_016_666_667L,
            landmarks = points1,
            cameraMotionRisk = 0.10,
            rollingShutterRisk = 0.06,
            occlusionRatio = 0.02,
        )

        val first = engine.resolveWithMotionEvidence(settings, input0)
        val second = engine.resolveWithMotionEvidence(settings, input1)

        assertTrue(first.evidence.trackedLandmarks == 16)
        assertTrue(second.evidence.differentialEnergy > 0.0)
        assertTrue(second.plan.featureAllocation.motion > 0.50)
        assertTrue(second.plan.reasonCodes.none { it.contains("quality-block:") })
    }

    @Test
    fun healthy_precision_contract_remains_authoritative() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
        )
        val evidence = MotionCaptureCustomizationEngine.RuntimeEvidence(
            thermalHeadroom = 0.95,
            gpuHeadroom = 0.95,
            memoryHeadroom = 0.95,
            synchronizationConfidence = 0.95,
            calibrationConfidence = 0.92,
            trackingConfidence = 0.95,
            motionP95ErrorMillimeters = 1.5,
            temporalLagMilliseconds = 8.0,
            dropoutRatio = 0.005,
            viewDiversity = 0.82,
            crossViewAgreement = 0.90,
            connectedCameras = 6,
        )
        val plan = engine.resolve(settings, evidence)

        assertTrue(plan.captureAllowed)
        assertTrue(plan.reasonCodes.none { it == "precision-contract-breached" })
    }

    @Test
    fun healthy_reference_plan_keeps_full_requested_rates() {
        val settings = HighEndMotionCaptureSettings(
            profileId = "REFERENCE_BAKE",
            adaptiveQualityEnabled = true,
        )

        val plan = engine.resolve(settings)

        assertEquals(120, plan.effectiveCaptureFps)
        assertEquals(90, plan.effectiveInferenceFps)
        assertTrue(plan.faceMeshHz >= 80)
        assertTrue(plan.eyeTrackingHz >= 100)
        assertTrue(plan.meshVertexBudget >= 900_000)
    }

    @Test
    fun temporal_supersampling_settings_survive_policy_resolution() {
        val settings = HighEndMotionCaptureSettings(
            interleavedTemporalFusionEnabled = true,
            interleavedTemporalWindowMillis = 64.0,
            interleavedPredictionHorizonMillis = 16.0,
            interleavedMaximumSampleSkewMillis = 20.0,
            interleavedOutlierThresholdPixels = 18.0,
            temporalSupersamplingPriority = 0.97,
        )
        val plan = engine.resolve(
            settings,
            MotionCaptureCustomizationEngine.RuntimeEvidence(
                synchronizationConfidence = 0.96,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.94,
                connectedCameras = 6,
            ),
        )

        assertTrue(plan.interleavedTemporalFusionEnabled)
        assertEquals(64.0, plan.interleavedTemporalWindowMillis)
        assertEquals(16.0, plan.interleavedPredictionHorizonMillis)
        assertEquals(20.0, plan.interleavedMaximumSampleSkewMillis)
        assertEquals(18.0, plan.interleavedOutlierThresholdPixels)
        assertEquals(0.97, plan.temporalSupersamplingPriority)
        assertTrue(
            "interleaved-temporal-priority-active" in
                plan.reasonCodes,
        )
    }

    @Test
    fun interleaved_runtime_stays_unqualified_until_measured_camera_samples_arrive() {
        val settings =
            HighEndMotionCaptureSettings(
                interleavedTemporalFusionEnabled = true,
            )

        val before =
            engine.resolve(
                settings,
                MotionCaptureCustomizationEngine.RuntimeEvidence(
                    synchronizationConfidence = 0.98,
                    calibrationConfidence = 0.98,
                    trackingConfidence = 0.98,
                    connectedCameras = 8,
                ),
            )

        assertTrue(before.interleavedWindowCoverage == 0.0)
        assertTrue(before.interleavedEffectiveSamplingMultiplier == 1.0)
        assertTrue(!before.interleavedWindowPublishable)

        val cameras =
            (0 until 4).map { index ->
                engine.ingestInterleavedTemporalSample(
                    com.nivukx.nox.core.reconstruction.InterleavedTemporalSamplingEngine.CameraSample(
                        cameraId = "cam-$index",
                        timestampNanos =
                            10_000_000_000L +
                                index * 1_000_000L,
                        sequence = index.toLong(),
                        sector = index,
                        confidence = 0.99,
                        synchronized = true,
                        timingResidualMillis = 0.20,
                        geometricQuality = 0.98,
                        outlierScore = 0.0,
                    ),
                )
            }

        assertTrue(cameras.last() != null)

        val after =
            engine.resolve(
                settings,
                MotionCaptureCustomizationEngine.RuntimeEvidence(
                    synchronizationConfidence = 0.98,
                    calibrationConfidence = 0.98,
                    trackingConfidence = 0.98,
                    connectedCameras = 8,
                ),
            )

        assertTrue(
            after.interleavedEffectiveSamplingMultiplier > 1.0,
        )
        assertTrue(after.interleavedWindowCoverage > 0.0)
        assertTrue(after.interleavedWindowPublishable)
    }

    @Test
    fun disabled_temporal_supersampling_is_explicit_in_policy_reasons() {
        val plan = engine.resolve(
            HighEndMotionCaptureSettings(
                interleavedTemporalFusionEnabled = false,
            ),
        )
        assertFalse(plan.interleavedTemporalFusionEnabled)
        assertTrue(
            "interleaved-temporal-disabled-by-policy" in
                plan.reasonCodes,
        )
    }


    @Test
    fun real_world_precision_units_are_accepted() {
        val evidence =
            MotionCaptureCustomizationEngine.RuntimeEvidence(
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.95,
                memoryHeadroom = 0.95,
                synchronizationConfidence = 0.95,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.95,
                motionP95ErrorMillimeters = 5.0,
                temporalLagMilliseconds = 20.0,
                dropoutRatio = 0.01,
                viewDiversity = 0.60,
                crossViewAgreement = 0.75,
                connectedCameras = 4,
            )
        val plan = engine.resolve(
            HighEndMotionCaptureSettings(),
            evidence,
        )
        assertTrue(plan.captureAllowed)
    }


    @Test
    fun quality_locked_reference_mode_requires_empirical_precision_evidence() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
            referenceGradePrecisionEnabled = true,
        )

        val plan = engine.resolve(
            settings,
            MotionCaptureCustomizationEngine.RuntimeEvidence(
                thermalHeadroom = 0.98,
                gpuHeadroom = 0.98,
                memoryHeadroom = 0.98,
                synchronizationConfidence = 0.98,
                calibrationConfidence = 0.98,
                trackingConfidence = 0.98,
                connectedCameras = 8,
                referenceCalibrationWorkflowReady = true,
                referenceCalibrationRigScore = 0.98,
                referenceCalibrationCoverage = 0.95,
                referenceCalibrationReprojectionErrorPx = 0.30,
                referenceCalibrationClockSkewMilliseconds = 0.20,
                referenceTemporalAlignmentConfidence = 0.99,
                referenceDriftStability = 0.99,
                referenceThermalTemperatureDeltaCelsius = 0.10,
                referenceThermalStability = 0.99,
                referenceObservabilityScore = 0.97,
            ),
        )

        assertFalse(plan.captureAllowed)
        assertTrue(
            "reference-grade-empirical-evidence-required" in
                plan.reasonCodes,
        )
        assertFalse(plan.referenceGradeEvidence.ready)
    }

    @Test
    fun motion_evidence_resolves_through_bounded_temporal_fusion() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = true,
            qualityLockEnabled = false,
        )
        val landmarks = (0 until 12).map {
            AdaptiveMotionEvidenceEngine.Landmark(
                id = it,
                x = 0.10 + it * 0.01,
                y = 0.15 + it * 0.005,
                confidence = 0.98,
            )
        }

        val first = engine.resolveWithMotionEvidence(
            settings = settings,
            input = AdaptiveMotionEvidenceEngine.Input(
                timestampNanos = 2_000_000_000L,
                landmarks = landmarks,
                nominalFrameRateHz = 60.0,
            ),
        )
        val second = engine.resolveWithMotionEvidence(
            settings = settings,
            input = AdaptiveMotionEvidenceEngine.Input(
                timestampNanos = 2_016_666_667L,
                landmarks = landmarks.map {
                    it.copy(x = it.x + 0.004)
                },
                nominalFrameRateHz = 60.0,
                occlusionRatio = 0.02,
            ),
        )

        assertTrue(first.plan.temporalFusionSummary != null)
        assertTrue(second.plan.temporalFusionSummary != null)
        assertEquals(12, second.plan.temporalFusionSummary!!.activeTracks)
        assertTrue(
            second.plan.temporalFusionSummary!!.temporalConsistency in 0.0..1.0,
        )
        assertTrue(
            second.plan.productionQualityScore in 0.0..1.0,
        )
    }


    @Test
    fun motion_evidence_is_reduced_when_visibility_contact_evidence_is_weak() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = true,
            qualityLockEnabled = false,
        )

        val strong = engine.resolveWithMotionEvidence(
            settings = settings,
            input = AdaptiveMotionEvidenceEngine.Input(
                timestampNanos = 8_000_000_000L,
                landmarks = (0 until 16).map {
                    AdaptiveMotionEvidenceEngine.Landmark(
                        id = it,
                        x = 0.2 + it * 0.01,
                        y = 0.3 + it * 0.005,
                        confidence = 0.98,
                    )
                },
                occlusionRatio = 0.0,
            ),
        )

        val weak = engine.resolveWithMotionEvidence(
            settings = settings,
            input = AdaptiveMotionEvidenceEngine.Input(
                timestampNanos = 8_016_666_667L,
                landmarks = (0 until 16).map {
                    AdaptiveMotionEvidenceEngine.Landmark(
                        id = it,
                        x = 0.2 + it * 0.012,
                        y = 0.3 + it * 0.006,
                        confidence = if (it < 4) 0.95 else 0.08,
                    )
                },
                occlusionRatio = 0.72,
            ),
        )

        assertTrue(
            weak.plan.occlusionContactEvidence.captureTrust <
                strong.plan.occlusionContactEvidence.captureTrust,
        )
        assertTrue(
            weak.plan.productionQualityScore <=
                strong.plan.productionQualityScore + 1e-9,
        )
        assertTrue(
            weak.plan.reasonCodes.any {
                it.contains("occlusion")
            },
        )
    }


    @Test
    fun temporal_calibration_contract_is_propagated_into_quality_policy() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
        )

        val observations = (0 until 12).map { index ->
            val local = 4_000_000_000L + index * 20_000_000L
            SpatiotemporalCalibrationEngine.Observation(
                cameraId = "cam-sync",
                sequence = index.toLong(),
                localTimestampNanos = local,
                referenceTimestampNanos =
                    local + 2_000_000L,
                normalizedRow = (index % 4) / 3.0,
            )
        }

        val plan = engine.resolveWithTemporalCalibration(
            settings = settings,
            observations = observations,
        )

        assertTrue(plan.temporalCalibrationEvidence != null)
        assertTrue(
            plan.temporalCalibrationEvidence!!.residualP95Milliseconds < 12.0,
        )
        assertTrue(
            plan.reasonCodes.none {
                it == "temporal-calibration-contract-breached"
            },
        )
    }

    @Test
    fun interaction_contract_blocks_excessive_penetration() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = false,
            maximumContactPenetrationMillimeters = 2.0,
        )

        val observation =
            ContactAwareReconstructionEngine.InteractionObservation(
                handId = 1L,
                objectId = 2L,
                timestampNanos = 5_000_000_000L,
                distanceMillimeters = 1.0,
                relativeSpeedMetersPerSecond = 0.05,
                normalAlignment = 1.0,
                penetrationMillimeters = 5.0,
                handConfidence = 0.99,
                objectConfidence = 0.99,
            )

        val isolatedEngine = MotionCaptureCustomizationEngine()
        try {
            val plan = isolatedEngine.resolveWithInteractionEvidence(
                settings = settings,
                observations = listOf(observation),
                timestampNanos = 5_000_000_000L,
            )

            assertTrue(
                "hand-object-contact-contract-breached" in
                    plan.reasonCodes,
            )
            assertFalse(plan.captureAllowed)
        } finally {
            isolatedEngine.close()
        }
    }


    @Test
    fun surface_deformation_evidence_reduces_quality_on_geometry_violation() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = false,
        )
        val firstFrame =
            SurfaceDeformationConsistencyEngine.Frame(
                timestampNanos = 11_000_000_000L,
                vertices = (0L until 12L).map {
                    SurfaceDeformationConsistencyEngine.Vertex(
                        id = it,
                        position =
                            SurfaceDeformationConsistencyEngine.Vec3(
                                x = it * 0.01,
                                y = 0.0,
                                z = 1.0,
                            ),
                        normal =
                            SurfaceDeformationConsistencyEngine.Vec3(
                                0.0, 0.0, 1.0,
                            ),
                        curvature = 0.1,
                        confidence = 0.98,
                    )
                },
                edges = (0L until 11L).map {
                    SurfaceDeformationConsistencyEngine.Edge(
                        firstVertexId = it,
                        secondVertexId = it + 1L,
                    )
                },
            )

        val shiftedFrame =
            firstFrame.copy(
                timestampNanos = 11_016_666_667L,
                vertices = firstFrame.vertices.map {
                    it.copy(
                        position =
                            SurfaceDeformationConsistencyEngine.Vec3(
                                x = it.position.x +
                                    if (it.id % 2L == 0L) 0.20 else 0.0,
                                y = it.position.y,
                                z = it.position.z,
                            ),
                    )
                },
            )

        val first =
            engine.resolveWithSurfaceEvidence(
                settings = settings,
                frame = firstFrame,
            )
        val second =
            engine.resolveWithSurfaceEvidence(
                settings = settings,
                frame = shiftedFrame,
            )

        assertTrue(first.surfaceDeformationEvidence != null)
        assertTrue(second.surfaceDeformationEvidence != null)
        assertTrue(
            second.surfaceDeformationEvidence!!.strainP95Ratio >=
                first.surfaceDeformationEvidence!!.strainP95Ratio,
        )
        assertTrue(
            second.productionQualityScore <=
                first.productionQualityScore + 1e-9,
        )
        assertTrue(
            second.reasonCodes.any {
                it.contains("surface:")
            },
        )
    }


    @Test
    fun reference_grade_mode_blocks_relative_only_metric_geometry() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
            referenceGradePrecisionEnabled = true,
        )

        val scaleRecovery =
            CalibrationFreeMetricRecoveryEngine.Result(
                timestampNanos = 9_000_000_000L,
                mode =
                    CalibrationFreeMetricRecoveryEngine.Mode.RELATIVE_ONLY,
                scaleMetersPerUnit = null,
                confidence = 0.0,
                acceptedAnchors = 0,
                acceptedSegments = 0,
                rejectedSegments = 0,
                robustScaleSpread = 1.0,
                metricProvenance = "relative-only",
                warnings = listOf("metric-anchor-required"),
            )

        val result = engine.resolve(
            settings = settings,
            evidence =
                MotionCaptureCustomizationEngine.RuntimeEvidence(
                    timestampNanos = 9_000_000_000L,
                    metricScaleRecovery = scaleRecovery,
                ),
        )

        assertFalse(result.captureAllowed)
        assertTrue(
            result.reasonCodes.contains(
                "reference-grade-metric-scale-insufficient",
            ),
        )
    }


    @Test
    fun reference_grade_blocks_low_confidence_prior_derived_metric_scale() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
            referenceGradePrecisionEnabled = true,
        )

        val recovery =
            CalibrationFreeMetricRecoveryEngine.Result(
                timestampNanos = 10_000_000_000L,
                mode =
                    CalibrationFreeMetricRecoveryEngine.Mode.SCALE_ANCHORED_METRIC,
                scaleMetersPerUnit = 1.82,
                confidence = 0.58,
                acceptedAnchors = 0,
                acceptedSegments = 6,
                rejectedSegments = 1,
                robustScaleSpread = 0.08,
                metricProvenance = "anthropometric-segment-prior",
                warnings = listOf("metric-scale-is-prior-derived"),
            )

        val result = engine.resolve(
            settings = settings,
            evidence =
                MotionCaptureCustomizationEngine.RuntimeEvidence(
                    timestampNanos = 10_000_000_000L,
                    metricScaleRecovery = recovery,
                ),
        )

        assertTrue(!result.captureAllowed)
        assertTrue(
            result.reasonCodes.any {
                it.contains("metric-scale")
            },
        )
    }

    @Test
    fun frontier_profile_bridges_ultra_precision_and_distributed_camera_policy() {
        val settings = HighEndMotionCaptureSettings(
            cameraResolutionPreset = com.nivukx.nox.core.model.CameraResolutionPreset.AUTO,
            targetFps = 120,
            inferenceFps = 60,
            cameraRawSidecarEnabled = true,
            cameraDepthSidecarEnabled = true,
            ultraPrecisionTargetP95ErrorMillimeters = 0.75,
            ultraPrecisionMinimumEvidenceSamples = 1200,
            distributedAuthoritativeCameraBudget = 32,
            distributedGlobalCameraEnvelope = 128,
        )
        val profile = engine.frontierProfile(settings)
        assertEquals(120, profile.targetFps)
        assertEquals(60, profile.inferenceFps)
        assertEquals(32, profile.authoritativeCameraLimit)
        assertEquals(128, profile.globalCameraLimit)
        assertEquals(0.75, profile.precisionTargetMillimeters)
        assertEquals(1200, profile.minimumEmpiricalSamples)
        assertTrue(profile.fingerprint().length == 64)
    }

    @Test
    fun quality_lock_blocks_underfunded_authoritative_camera_transport() {
        val settings = HighEndMotionCaptureSettings(
            adaptiveQualityEnabled = false,
            qualityLockEnabled = true,
            distributedGlobalCameraEnvelope = 128,
            distributedAuthoritativeCameraBudget = 32,
            distributedMinimumAuthorityHealth = 0.30,
            distributedMinimumAuthoritySynchronization = 0.50,
        )
        val cameras = (0 until 128).map { id ->
            com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine.CameraEvidence(
                cameraId = "cam-$id",
                role = com.nivukx.nox.core.model.DeviceRole.CAMERA,
                positionMeters = com.nivukx.nox.core.model.Vec3(
                    kotlin.math.cos(id.toDouble()),
                    1.5,
                    kotlin.math.sin(id.toDouble()) + 2.0,
                ),
                forwardVector = com.nivukx.nox.core.model.Vec3(0.0,0.0,1.0),
                calibrationConfidence = 0.95,
                synchronizationConfidence = 0.95,
                trackingConfidence = 0.95,
                thermalHeadroom = 0.95,
                transportHeadroom = 0.95,
                active = true,
                sector = com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine.Sector.entries[id % 8],
            )
        }
        val plan = engine.resolveWithDistributedRig(
            settings = settings,
            evidence = MotionCaptureCustomizationEngine.RuntimeEvidence(
                thermalHeadroom = 0.95,
                gpuHeadroom = 0.95,
                memoryHeadroom = 0.95,
                synchronizationConfidence = 0.95,
                calibrationConfidence = 0.95,
                trackingConfidence = 0.95,
                connectedCameras = 128,
                networkLinkCapacityKbps = 10_000.0,
            ),
            cameras = cameras,
            timestampNanos = 1_000L,
        )
        assertTrue(plan.transportBudget != null)
        assertTrue(plan.transportBudget!!.authoritativeUnderfunded > 0)
        assertFalse(plan.captureAllowed)
        assertTrue(plan.reasonCodes.any { it.startsWith("transport:") })
        assertTrue(plan.captureAuthority.distributedPlanApplied)
        assertEquals(
            plan.distributedRig?.selectedCameraCount ?: 0,
            plan.captureAuthority.distributedSelectedCameraCount,
        )
    )
}