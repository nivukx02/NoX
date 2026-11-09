package com.nivukx.nox

import com.nivukx.nox.core.model.CaptureQualityTier
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import com.nivukx.nox.core.model.ProductionCaptureProfile
import com.nivukx.nox.core.network.FrontierCameraTransportBudgetEngine
import com.nivukx.nox.core.ai.MotionCaptureTuning
import com.nivukx.nox.core.ai.ProductionMotionCaptureQualityGate
import com.nivukx.nox.core.ai.AdaptiveMotionEvidenceEngine
import com.nivukx.nox.core.ai.OcclusionContactEvidenceEngine
import com.nivukx.nox.core.reconstruction.PrecisionCriticalPathEngine
import com.nivukx.nox.core.calibration.CalibrationFreeMetricRecoveryEngine
import com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine
import com.nivukx.nox.core.capture.FocusConditioningEngine
import com.nivukx.nox.core.calibration.ReferenceGradeCalibrationProtocolEngine
import com.nivukx.nox.core.reconstruction.ContactAwareReconstructionEngine
import com.nivukx.nox.core.reconstruction.DistributedCameraRigCoordinator
import com.nivukx.nox.core.reconstruction.HighEndTemporalFusionEngine
import com.nivukx.nox.core.reconstruction.InterleavedTemporalSamplingEngine
import com.nivukx.nox.core.reconstruction.HighEndCaptureAuthorityEngine
import com.nivukx.nox.core.reconstruction.InfraredReferenceQualificationEngine
import com.nivukx.nox.core.reconstruction.MetricPrecisionBudgetEngine
import com.nivukx.nox.core.reconstruction.PrecisionCertificationEngine
import com.nivukx.nox.core.reconstruction.ReferenceGradeEvidenceGateEngine
import com.nivukx.nox.core.reconstruction.SpatiotemporalCalibrationEngine
import com.nivukx.nox.core.reconstruction.SurfaceDeformationConsistencyEngine
import com.nivukx.nox.core.sync.TimingAuthorityFusionEngine

/**
 * Resolves high-end operator customization into one bounded, replayable runtime plan.
 *
 * User intent controls relative allocation. Runtime evidence controls whether that intent can be
 * sustained. Quality-lock mode never silently reduces fidelity; it blocks unsafe capture instead.
 */
class MotionCaptureCustomizationEngine : AutoCloseable {

    private val qualityGate = ProductionMotionCaptureQualityGate()
    private val motionEvidenceEngine = AdaptiveMotionEvidenceEngine()
    private val metricPrecisionEngine = MetricPrecisionBudgetEngine()
    private val precisionCertificationEngine = PrecisionCertificationEngine()
    private val rigObservabilityEngine =
        CaptureRigObservabilityEngine()
    private val distributedCameraRigCoordinator =
        DistributedCameraRigCoordinator()
    private val referenceGradeProtocolEngine =
        ReferenceGradeCalibrationProtocolEngine()
    private val referenceGradeEvidenceGateEngine =
        ReferenceGradeEvidenceGateEngine()
    private val precisionCriticalPathEngine =
        PrecisionCriticalPathEngine()
    private val temporalFusionEngine =
        HighEndTemporalFusionEngine()
    private val interleavedTemporalSamplingEngine =
        InterleavedTemporalSamplingEngine()
    private val occlusionContactEvidenceEngine =
        OcclusionContactEvidenceEngine()
    private val contactAwareReconstructionEngine =
        ContactAwareReconstructionEngine()
    private val spatiotemporalCalibrationEngine =
        SpatiotemporalCalibrationEngine()
    private val surfaceDeformationConsistencyEngine =
        SurfaceDeformationConsistencyEngine()
    private val timingAuthorityFusionEngine =
        TimingAuthorityFusionEngine()
    private val focusConditioningEngine =
        FocusConditioningEngine()
    private val frontierTransportBudgetEngine =
        FrontierCameraTransportBudgetEngine()

    data class RuntimeEvidence(
        val thermalHeadroom: Double = 1.0,
        val gpuHeadroom: Double = 1.0,
        val memoryHeadroom: Double = 1.0,
        val synchronizationConfidence: Double = 1.0,
        val synchronizationResidualP95Milliseconds: Double = 0.0,
        val hardwareReadoutTimingAvailable: Boolean = false,
        val hardwareReadoutTimingConfidence: Double = 0.0,
        val hardwareReadoutP95JitterMilliseconds: Double = 0.0,
        val hardwareSynchronized: Boolean = false,
        val logicalCalibratedSynchronization: Boolean = false,
        val visualTimingBeaconAvailable: Boolean = false,
        val visualTimingBeaconConfidence: Double = 0.0,
        val visualTimingBeaconResidualP95Milliseconds: Double = 0.0,
        val synchronizationOutlierRatio: Double = 0.0,
        val calibrationFocusDistanceDiopters: Double? = null,
        val currentFocusDistanceDiopters: Double? = null,
        val focusConfidence: Double = 0.0,
        val manualFocusActive: Boolean = false,
        val calibrationConfidence: Double = 1.0,
        val trackingConfidence: Double = 1.0,
        val motionP95ErrorMillimeters: Double = 0.0,
        val motionP99ErrorMillimeters: Double =
            motionP95ErrorMillimeters * 2.0,
        val temporalLagMilliseconds: Double = 0.0,
        val dropoutRatio: Double = 0.0,
        val viewDiversity: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val connectedCameras: Int = 1,
        val timestampNanos: Long = 0L,
        val identityContinuity: Double = 1.0,
        val nativeQueueAgeMilliseconds: Double = 0.0,
        val nativeProcessingMilliseconds: Double = 0.0,
        val thermalRisk: Double = 0.0,
        val memoryPressure: Double = 0.0,
        val npuHeadroom: Double = 1.0,
        val cpuHeadroom: Double = 1.0,
        val meshTopologyRisk: Double = 0.0,
        val meshTopologyIntegrity: Double = 1.0,
        val surfaceTemporalStability: Double = 1.0,
        val faceEyeConfidence: Double = 1.0,
        val hand3dConfidence: Double = 1.0,
        val handConfidence: Double = 1.0,
        val calibrationDriftRisk: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val metricScaleDriftPercentPerMinute: Double = 0.0,
        val global4DTemporalConsistency: Double = 1.0,
        val contactConsistency: Double = 1.0,
        val handObjectContactConfidence: Double = 1.0,
        val interactionTemporalConsistency: Double = 1.0,
        val interactionCollisionConfidence: Double = 1.0,
        val contactPenetrationRisk: Double = 0.0,
        val surfaceDeformationStability: Double = 1.0,
        val surfaceTopologyContinuity: Double = 1.0,
        val surfaceStrainRisk: Double = 0.0,
        val surfaceNormalRisk: Double = 0.0,
        val surfaceCurvatureRisk: Double = 0.0,
        val crashRatePerMinute: Double = 0.0,
        val runtimeGovernorBlocked: Boolean = false,
        val empiricalBenchmarkAccepted: Boolean = false,
        val metricScaleRecovery:
            CalibrationFreeMetricRecoveryEngine.Result? = null,
        val empiricalBenchmarkSampleCount: Int = 0,
        val empiricalBenchmarkP95ErrorMillimeters: Double? = null,
        val empiricalBenchmarkP99ErrorMillimeters: Double? = null,
        val externalReferenceUncertaintyMillimeters: Double? = null,
        val matchedReferenceCaptureConditions: Boolean = false,
        val externalReferenceDocumented: Boolean = false,
        val referenceCalibrationWorkflowReady: Boolean = false,
        val referenceCalibrationRigScore: Double = 0.0,
        val referenceCalibrationCoverage: Double = 0.0,
        val referenceCalibrationReprojectionErrorPx: Double = 1.5,
        val referenceCalibrationClockSkewMilliseconds: Double = 1.5,
        val referenceTemporalAlignmentConfidence: Double = 0.95,
        val referenceDriftStability: Double = 0.95,
        val referenceRecalibrationRequired: Boolean = false,
        val referenceThermalTemperatureDeltaCelsius: Double = 0.5,
        val referenceThermalStability: Double = 0.95,
        val referenceObservabilityScore: Double = 0.0,
        val workingDistanceMeters: Double = 2.5,
        val cameraBaselineMeters: Double = 1.0,
        val effectiveFocalLengthPixels: Double = 1800.0,
        val reprojectionErrorPixels: Double = 1.5,
        val triangulationParallaxDegrees: Double = 12.0,
        val depthUncertaintyMeters: Double = 0.004,
        val poseTranslationUncertaintyMeters: Double = 0.002,
        val poseAngularUncertaintyDegrees: Double = 0.05,
        val temporalSkewMilliseconds: Double = 1.5,
        val frameDurationMilliseconds: Double = 16.6667,
        val rollingShutterFraction: Double = 0.10,
        val motionSpeedMetersPerSecond: Double = 1.0,
        val motionAccelerationMetersPerSecondSquared: Double = 2.0,
        val metricScaleUncertaintyRatio: Double = 0.001,
        val reconstructionPriorUncertaintyMeters: Double = 0.001,
        val networkLinkCapacityKbps: Double = 0.0,
        val networkPacketLossByCamera: Map<String, Double> = emptyMap(),
        val networkLatencyMillisByCamera: Map<String, Double> = emptyMap(),
        val authoritativeCameras: Int = 0,
        val epipolarConsistency: Double = 1.0,
        val bodyMetricEvidenceAuthority: Double = 1.0,
        val faceMetricEvidenceAuthority: Double = 1.0,
        val handMetricEvidenceAuthority: Double = 1.0,
        val faceEpipolarConsistency: Double = 1.0,
        val handEpipolarConsistency: Double = 1.0,
        val crossModalMetricAuthority: Double = 1.0,
        val crossModalEpipolarAuthority: Double = 1.0,
        val motionBlurRisk: Double = 0.0,
        val observedGeometryRatio: Double = 1.0,
        val inferredGeometryRatio: Double = 0.0,
        val cameraMotionRisk: Double = 0.0,
        val distributedRigPlanForAuthority:
            DistributedCameraRigCoordinator.Plan? = null,
    ) {
        init {
            require(
                synchronizationResidualP95Milliseconds.isFinite() &&
                    synchronizationResidualP95Milliseconds >= 0.0,
            )
            require(
                synchronizationOutlierRatio.isFinite() &&
                    synchronizationOutlierRatio in 0.0..1.0,
            )
            require(hardwareReadoutTimingConfidence in 0.0..1.0)
            require(visualTimingBeaconConfidence in 0.0..1.0)
            require(hardwareReadoutP95JitterMilliseconds >= 0.0)
            require(visualTimingBeaconResidualP95Milliseconds >= 0.0)
            require(focusConfidence in 0.0..1.0)
            listOf(
                thermalHeadroom,
                gpuHeadroom,
                memoryHeadroom,
                synchronizationConfidence,
                calibrationConfidence,
                trackingConfidence,
                hardwareReadoutTimingConfidence,
                visualTimingBeaconConfidence,
                dropoutRatio,
                viewDiversity,
                crossViewAgreement,
                epipolarConsistency,
                bodyMetricEvidenceAuthority,
                faceMetricEvidenceAuthority,
                handMetricEvidenceAuthority,
                faceEpipolarConsistency,
                handEpipolarConsistency,
                crossModalMetricAuthority,
                crossModalEpipolarAuthority,
                motionBlurRisk,
                observedGeometryRatio,
                inferredGeometryRatio,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(motionP95ErrorMillimeters.isFinite() && motionP95ErrorMillimeters >= 0.0)
            require(
                motionP99ErrorMillimeters.isFinite() &&
                    motionP99ErrorMillimeters >= motionP95ErrorMillimeters,
            )
            require(temporalLagMilliseconds.isFinite() && temporalLagMilliseconds >= 0.0)
            require(connectedCameras >= 0)
            require(authoritativeCameras in 0..connectedCameras)
            require(cameraMotionRisk.isFinite() && cameraMotionRisk in 0.0..1.0)
            require(timestampNanos >= 0L)
            networkPacketLossByCamera.values.forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            networkLatencyMillisByCamera.values.forEach {
                require(it.isFinite() && it >= 0.0)
            }
            listOf(
                referenceCalibrationRigScore,
                referenceCalibrationCoverage,
                referenceTemporalAlignmentConfidence,
                referenceDriftStability,
                referenceThermalStability,
                referenceObservabilityScore,
                identityContinuity,
                thermalRisk,
                memoryPressure,
                gpuHeadroom,
                npuHeadroom,
                cpuHeadroom,
                meshTopologyRisk,
                meshTopologyIntegrity,
                surfaceTemporalStability,
                faceEyeConfidence,
                hand3dConfidence,
                handConfidence,
                calibrationDriftRisk,
                rollingShutterRisk,
                occlusionRisk,
                global4DTemporalConsistency,
                contactConsistency,
                handObjectContactConfidence,
                interactionTemporalConsistency,
                interactionCollisionConfidence,
                contactPenetrationRisk,
                surfaceDeformationStability,
                surfaceTopologyContinuity,
                surfaceStrainRisk,
                surfaceNormalRisk,
                surfaceCurvatureRisk,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(nativeQueueAgeMilliseconds.isFinite() && nativeQueueAgeMilliseconds >= 0.0)
            require(nativeProcessingMilliseconds.isFinite() && nativeProcessingMilliseconds >= 0.0)
            require(metricScaleDriftPercentPerMinute.isFinite() && metricScaleDriftPercentPerMinute >= 0.0)
            require(
                referenceCalibrationRigScore in 0.0..1.0 &&
                    referenceTemporalAlignmentConfidence in 0.0..1.0 &&
                    referenceDriftStability in 0.0..1.0 &&
                    referenceThermalStability in 0.0..1.0 &&
                    referenceObservabilityScore in 0.0..1.0
            )
            require(referenceCalibrationReprojectionErrorPx >= 0.0)
            require(referenceCalibrationClockSkewMilliseconds >= 0.0)
            require(referenceThermalTemperatureDeltaCelsius >= 0.0)
            require(crashRatePerMinute.isFinite() && crashRatePerMinute >= 0.0)
            require(empiricalBenchmarkSampleCount >= 0)
            empiricalBenchmarkP95ErrorMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            empiricalBenchmarkP99ErrorMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            externalReferenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            listOf(
                workingDistanceMeters,
                cameraBaselineMeters,
                effectiveFocalLengthPixels,
                reprojectionErrorPixels,
                triangulationParallaxDegrees,
                depthUncertaintyMeters,
                poseTranslationUncertaintyMeters,
                poseAngularUncertaintyDegrees,
                temporalSkewMilliseconds,
                frameDurationMilliseconds,
                rollingShutterFraction,
                motionSpeedMetersPerSecond,
                motionAccelerationMetersPerSecondSquared,
                metricScaleUncertaintyRatio,
                reconstructionPriorUncertaintyMeters,
            ).forEach { require(it.isFinite() && it >= 0.0) }
            require(networkLinkCapacityKbps.isFinite() && networkLinkCapacityKbps >= 0.0)
            require(networkPacketLossByCamera.keys.all { it.isNotBlank() })
            require(networkPacketLossByCamera.values.all { it.isFinite() && it in 0.0..1.0 })
            require(networkLatencyMillisByCamera.keys.all { it.isNotBlank() })
            require(networkLatencyMillisByCamera.values.all { it.isFinite() && it >= 0.0 })
        }

        val computeHeadroom: Double
            get() = minOf(
                thermalHeadroom,
                gpuHeadroom,
                memoryHeadroom,
                npuHeadroom,
                cpuHeadroom,
            )
    }

    data class FeatureAllocation(
        val motion: Double,
        val temporal: Double,
        val surface: Double,
        val faceEye: Double,
        val hands: Double,
        val cameraCoverage: Double,
    )

    data class MotionEvidencePlan(
        val plan: Plan,
        val evidence: AdaptiveMotionEvidenceEngine.Evidence,
    )

    data class Plan(
        val profile: ProductionCaptureProfile,
        val tuning: MotionCaptureTuning,
        val effectiveCaptureFps: Int,
        val effectiveInferenceFps: Int,
        val denseMeshHz: Int,
        val faceMeshHz: Int,
        val eyeTrackingHz: Int,
        val handTrackingHz: Int,
        val meshVertexBudget: Int,
        val featureAllocation: FeatureAllocation,
        val interleavedTemporalFusionEnabled: Boolean,
        val interleavedTemporalWindowMillis: Double,
        val interleavedPredictionHorizonMillis: Double,
        val interleavedMaximumSampleSkewMillis: Double,
        val interleavedOutlierThresholdPixels: Double,
        val temporalSupersamplingPriority: Double,
        val interleavedEffectiveSamplingMultiplier: Double = 1.0,
        val interleavedTemporalHealth: Double = 0.0,
        val interleavedWindowCoverage: Double = 0.0,
        val interleavedWindowPublishable: Boolean = false,
        val interleavedWarnings: List<String> = emptyList(),
        val adaptiveScale: Double,
        val adaptiveDegradationApplied: Boolean,
        val captureAllowed: Boolean,
        val reasonCodes: List<String>,
        val productionQualityBand:
            ProductionMotionCaptureQualityGate.Band =
            ProductionMotionCaptureQualityGate.Band.HIGH,
        val productionQualityScore: Double = 0.0,
        val productionQualityContractSatisfied: Boolean = false,
        val productionRecommendedFrameStride: Int = 1,
        val productionRecommendedWorkers: Int = 1,
        val productionRuntimeBudget:
            ProductionMotionCaptureQualityGate.RuntimeBudget =
            ProductionMotionCaptureQualityGate.RuntimeBudget(
                detectorScale = 1.0,
                depthScale = 1.0,
                triangulationScale = 1.0,
                temporalFusionScale = 1.0,
                faceEyeScale = 1.0,
                handScale = 1.0,
                surfaceScale = 1.0,
                diagnosticsScale = 0.5,
                recommendedCameraSelectionCount = 1,
                thermalReserve = 1.0,
                memoryReserve = 1.0,
            ),
        val captureAuthority:
            HighEndCaptureAuthorityEngine.Result =
            HighEndCaptureAuthorityEngine().evaluate(
                HighEndCaptureAuthorityEngine.Input(
                    timestampNanos = 0L,
                    connectedCameras = 0,
                ),
            ),
        val metricPrecision:
            MetricPrecisionBudgetEngine.Decision =
            MetricPrecisionBudgetEngine().evaluate(
                MetricPrecisionBudgetEngine.Input(),
            ),
        val cameraRigObservabilityScore: Double = 0.0,
        val selectedCameraIds: List<String> = emptyList(),
        val cameraRigWarnings: List<String> = emptyList(),
        val distributedRig:
            DistributedCameraRigCoordinator.Plan? = null,
        val transportBudget:
            FrontierCameraTransportBudgetEngine.Result? = null,
        val infraredReferenceQualification:
            InfraredReferenceQualificationEngine.Comparison? = null,
        val precisionCriticalPath:
            PrecisionCriticalPathEngine.Decision? = null,
        val precisionCertification:
            PrecisionCertificationEngine.Decision =
            PrecisionCertificationEngine().evaluate(
                PrecisionCertificationEngine.Input(
                    metrology = MetricPrecisionBudgetEngine().evaluate(
                        MetricPrecisionBudgetEngine.Input(),
                    ),
                ),
            ),
        val metricScaleRecovery:
            CalibrationFreeMetricRecoveryEngine.Result? = null,
        val referenceGradeEvidence:
            ReferenceGradeEvidenceGateEngine.Result =
            ReferenceGradeEvidenceGateEngine().evaluate(
                ReferenceGradeEvidenceGateEngine.Input(enabled = false),
            ),
        val temporalFusionSummary:
            HighEndTemporalFusionEngine.FrameSummary? = null,
        val temporalCalibrationEvidence:
            SpatiotemporalCalibrationEngine.Result? = null,
        val occlusionContactEvidence:
            OcclusionContactEvidenceEngine.Result =
            OcclusionContactEvidenceEngine.Result(
                timestampNanos = 0L,
                actorId = 0L,
                processedLandmarkCount = 0,
                processedContactCount = 0,
                meanVisibility = 0.0,
                criticalOcclusionRatio = 1.0,
                visibilityUncertaintyMultiplier = 1.0,
                predictionBudgetRemainingMillis = 0.0,
                contactClusterCount = 0,
                contactPreservationScore = 0.0,
                identityMergeSuppression = 0.0,
                frequencyRecoveryPrior = 0.0,
                momentumRecoveryPrior = 0.0,
                captureTrust = 0.0,
                landmarkEvidence = emptyList(),
                warnings = listOf("occlusion-evidence-uninitialized"),
            ),
        val handObjectContactEvidence:
            ContactAwareReconstructionEngine.Aggregate? = null,
        val surfaceDeformationEvidence:
            SurfaceDeformationConsistencyEngine.Result? = null,
        val referenceGradeProtocol:
            ReferenceGradeCalibrationProtocolEngine.Status =
            ReferenceGradeCalibrationProtocolEngine(
                ).evaluate(
                    ReferenceGradeCalibrationProtocolEngine.Input(
                        workflowReady = false,
                        rigScore = 0.0,
                        coverage = 0.0,
                        reprojectionErrorPx = 99.0,
                        clockSkewMilliseconds = 99.0,
                        temporalAlignmentConfidence = 0.0,
                        driftStability = 0.0,
                        recalibrationRequired = true,
                        scaleDriftPercentPerMinute = 99.0,
                        temperatureDeltaCelsius = 99.0,
                        thermalStability = 0.0,
                        observabilityScore = 0.0,
                        metrology =
                            MetricPrecisionBudgetEngine()
                                .evaluate(
                                    MetricPrecisionBudgetEngine.Input(),
                                ),
                    ),
                ),
    ) {
        val hasHardBlock: Boolean
            get() = !captureAllowed

        val qualityTier: CaptureQualityTier
            get() = profile.qualityTier
    }

    fun resolveWithDistributedRig(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
        timestampNanos: Long = evidence.timestampNanos,
    ): Plan {
        distributedCameraRigCoordinator.configure(
            DistributedCameraRigCoordinator.Config(
                maximumGlobalCameras =
                    settings.distributedGlobalCameraEnvelope,
                maximumShards =
                    settings.distributedMaximumShards,
                targetSelectedCameras =
                    minOf(
                        settings.distributedAuthoritativeCameraBudget,
                        settings.distributedGlobalCameraEnvelope,
                    ),
                maximumSelectedCameras =
                    minOf(
                        settings.distributedAuthoritativeCameraBudget,
                        settings.distributedGlobalCameraEnvelope,
                    ),
                minimumAuthoritativeHealth =
                    settings.distributedMinimumAuthorityHealth,
                minimumAuthoritativeSync =
                    settings.distributedMinimumAuthoritySynchronization,
                maximumTemporalResidualMillis =
                    settings.distributedMaximumTemporalResidualMs,
                maximumTemporalOutlierRatio =
                    settings.distributedMaximumTemporalOutlierRatio,
                minimumAuthoritativeTransportHeadroom =
                    settings.distributedMinimumTransportHeadroom,
            ),
        )

        val distributed =
            distributedCameraRigCoordinator.plan(
                timestampNanos = timestampNanos,
                cameras = cameras,
            )

        val transportBudget =
            if (evidence.networkLinkCapacityKbps > 0.0) {
                val selectedIds = distributed.selectedCameraIds.toSet()
                frontierTransportBudgetEngine.allocate(
                    channels = cameras.map { camera ->
                        FrontierCameraTransportBudgetEngine.CameraChannel(
                            cameraId = camera.cameraId,
                            authoritative = camera.cameraId in selectedIds,
                            quality = camera.transportHeadroom,
                            packetLoss = evidence.networkPacketLossByCamera[camera.cameraId]
                                ?: (1.0 - camera.transportHeadroom).coerceIn(0.0,1.0) * 0.03,
                            latencyMillis = evidence.networkLatencyMillisByCamera[camera.cameraId]
                                ?: (1.0 - camera.transportHeadroom).coerceIn(0.0,1.0) * 30.0,
                            requestedKbps =
                                if (camera.cameraId in selectedIds) 12_000.0 else 1_500.0,
                        )
                    },
                    linkCapacityKbps = evidence.networkLinkCapacityKbps,
                    timestampNanos = timestampNanos,
                )
            } else {
                null
            }

        val distributedEvidence =
            evidence.copy(
                connectedCameras = distributed.acceptedCameraCount,
                authoritativeCameras = distributed.selectedCameraCount,
                synchronizationConfidence =
                    minOf(
                        evidence.synchronizationConfidence,
                        distributed.globalTemporalHealth,
                    ),
                viewDiversity =
                    minOf(
                        evidence.viewDiversity,
                        distributed.minimumPairwiseDiversity
                            .coerceAtLeast(
                                evidence.viewDiversity.takeIf {
                                    distributed.selectedCameraCount < 2
                                } ?: 0.0,
                            ),
                    ),
                crossViewAgreement =
                    minOf(
                        evidence.crossViewAgreement,
                        distributed.globalGeometryScore,
                    ),
                distributedRigPlanForAuthority = distributed,
                timestampNanos = timestampNanos,
            )

        val base =
            resolve(
                settings = settings,
                evidence = distributedEvidence,
            )

        return base.copy(
            selectedCameraIds = distributed.selectedCameraIds,
            cameraRigObservabilityScore = distributed.globalHealth,
            cameraRigWarnings = distributed.warnings,
            distributedRig = distributed,
            transportBudget = transportBudget,
            captureAllowed =
                base.captureAllowed &&
                    distributed.authoritative &&
                    (
                        transportBudget == null ||
                            !settings.qualityLockEnabled ||
                            (
                                transportBudget.authoritativeUnderfunded == 0 &&
                                    !transportBudget.overloaded
                            )
                    ),
            reasonCodes =
                (
                    base.reasonCodes +
                        distributed.warnings.map {
                            "distributed-rig:" + it
                        } +
                        if (distributed.authoritative) {
                            listOf("distributed-rig-authoritative")
                        } else {
                            listOf("distributed-rig-authority-blocked")
                        } +
                        (transportBudget?.warnings?.map { "transport:" + it }.orEmpty())
                    ).distinct(),
        )
    }

    fun resolveWithReconstructionEvidence(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        reconstruction:
            MultiViewFusionEngine.DetailedResult? = null,
    ): Plan =
        resolve(
            settings = settings,
            evidence =
                reconstruction?.let {
                    evidence.copy(
                        bodyMetricEvidenceAuthority =
                            it.bodyMetricEvidenceAuthority.coerceIn(
                                0.0,
                                1.0,
                            ),
                        epipolarConsistency =
                            it.bodySymmetricEpipolarConsistencyScore
                                .coerceIn(
                                    0.0,
                                    1.0,
                                ),
                        faceMetricEvidenceAuthority =
                            it.faceMetricEvidenceAuthority
                                .coerceIn(0.0, 1.0),
                        handMetricEvidenceAuthority =
                            it.handMetricEvidenceAuthority
                                .coerceIn(0.0, 1.0),
                        faceEpipolarConsistency =
                            it.faceEpipolarConsistency
                                .coerceIn(0.0, 1.0),
                        handEpipolarConsistency =
                            it.handEpipolarConsistency
                                .coerceIn(0.0, 1.0),
                        crossModalMetricAuthority =
                            it.crossModalMetricAuthority
                                .coerceIn(0.0, 1.0),
                        crossModalEpipolarAuthority =
                            it.crossModalEpipolarAuthority
                                .coerceIn(0.0, 1.0),
                        cameraMotionRisk =
                            evidence.cameraMotionRisk
                                .coerceIn(0.0, 1.0),
                        bodyEpipolarConsistency =
                            it.bodySymmetricEpipolarConsistencyScore
                                .coerceIn(
                                    0.0,
                                    1.0,
                                ),
                        depthUncertaintyMeters =
                            it.bodyEvidenceMeanUncertaintyMeters
                                .coerceAtLeast(0.0),
                        observedGeometryRatio =
                            it.bodyEvidenceAcceptedRatio
                                .coerceIn(0.0, 1.0),
                    )
                }
                    ?: evidence,
        )

    fun resolve(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
    ): Plan {
        val profile = settings.profile().validated()
        val baseTuning = tuningFor(profile)

        val interleavedRuntime =
            interleavedTemporalSamplingEngine.summary()

        val timingAuthority =
            timingAuthorityFusionEngine.evaluate(
                TimingAuthorityFusionEngine.Input(
                    hardwareReadoutAvailable =
                        evidence.hardwareReadoutTimingAvailable,
                    hardwareReadoutConfidence =
                        evidence.hardwareReadoutTimingConfidence,
                    hardwareReadoutP95JitterMillis =
                        evidence.hardwareReadoutP95JitterMilliseconds,
                    hardwareSynchronized =
                        evidence.hardwareSynchronized,
                    logicalCalibratedSynchronization =
                        evidence.logicalCalibratedSynchronization,
                    visualBeaconAvailable =
                        evidence.visualTimingBeaconAvailable,
                    visualBeaconConfidence =
                        evidence.visualTimingBeaconConfidence,
                    visualBeaconResidualP95Millis =
                        evidence.visualTimingBeaconResidualP95Milliseconds,
                    peerSynchronizationConfidence =
                        evidence.synchronizationConfidence,
                    peerResidualP95Millis =
                        evidence.synchronizationResidualP95Milliseconds,
                    peerSynchronizationEvidenceAvailable =
                        evidence.connectedCameras >= 2,
                    rollingShutterRisk =
                        evidence.rollingShutterRisk,
                ),
            )

        val focusConditioning =
            focusConditioningEngine.evaluate(
                FocusConditioningEngine.Input(
                    calibrationFocusDistanceDiopters =
                        evidence.calibrationFocusDistanceDiopters,
                    currentFocusDistanceDiopters =
                        evidence.currentFocusDistanceDiopters,
                    focusConfidence =
                        evidence.focusConfidence,
                    manualFocus =
                        evidence.manualFocusActive,
                ),
            )

        val qualityDecision =
            qualityGate.evaluate(
                ProductionMotionCaptureQualityGate.Evidence(
                    timestampNanos = evidence.timestampNanos,
                    synchronizationConfidence = evidence.synchronizationConfidence,
                    synchronizationResidualP95Milliseconds =
                        evidence.synchronizationResidualP95Milliseconds,
                    synchronizationOutlierRatio =
                        evidence.synchronizationOutlierRatio,
                    calibrationConfidence = evidence.calibrationConfidence,
                    trackingConfidence = evidence.trackingConfidence,
                    crossViewAgreement = evidence.crossViewAgreement,
                    viewDiversity = evidence.viewDiversity,
                    identityContinuity = evidence.identityContinuity,
                    p95ErrorMillimeters = evidence.motionP95ErrorMillimeters,
                    p99ErrorMillimeters = evidence.motionP99ErrorMillimeters,
                    temporalLagMilliseconds = evidence.temporalLagMilliseconds,
                    dropoutRatio = evidence.dropoutRatio,
                    queueAgeMilliseconds = evidence.nativeQueueAgeMilliseconds,
                    processingMilliseconds = evidence.nativeProcessingMilliseconds,
                    thermalRisk = evidence.thermalRisk,
                    memoryPressure = evidence.memoryPressure,
                    meshTopologyIntegrity =
                        minOf(
                            evidence.meshTopologyIntegrity,
                            evidence.surfaceTopologyContinuity,
                        ),
                    surfaceTemporalStability =
                        minOf(
                            evidence.surfaceTemporalStability,
                            evidence.surfaceDeformationStability,
                        ),
                    faceEyeConfidence = evidence.faceEyeConfidence,
                    handConfidence = evidence.handConfidence,
                    calibrationDriftRisk = evidence.calibrationDriftRisk,
                    rollingShutterRisk = evidence.rollingShutterRisk,
                    occlusionRisk = evidence.occlusionRisk,
                    metricScaleDriftPercentPerMinute =
                        evidence.metricScaleDriftPercentPerMinute,
                    global4DTemporalConsistency =
                        evidence.global4DTemporalConsistency,
                    contactConsistency =
                        minOf(
                            evidence.contactConsistency,
                            evidence.handObjectContactConfidence,
                            evidence.interactionTemporalConsistency,
                            evidence.interactionCollisionConfidence,
                            1.0 - evidence.contactPenetrationRisk,
                            evidence.surfaceDeformationStability,
                            evidence.surfaceTopologyContinuity,
                            1.0 - evidence.surfaceStrainRisk,
                            1.0 - evidence.surfaceNormalRisk,
                            1.0 - evidence.surfaceCurvatureRisk,
                        ),
                    gpuHeadroom = evidence.gpuHeadroom,
                    npuHeadroom = evidence.npuHeadroom,
                    cpuHeadroom = evidence.cpuHeadroom,
                    activeCameras = evidence.connectedCameras,
                    timingAuthorityConfidence =
                        timingAuthority.confidence,
                    timingAuthorityResidualP95Milliseconds =
                        timingAuthority.residualP95Millis
                            .coerceAtLeast(0.0),
                    focusConditioningScore =
                        focusConditioning.conditioningScore,
                    focusConditioningReferenceCompatible =
                        focusConditioning.referenceGradeCompatible,
                ),
                ProductionMotionCaptureQualityGate.Contract(
                    p95TargetMillimeters =
                        settings.targetMotionP95ErrorMillimeters,
                    synchronizationResidualP95MaximumMilliseconds =
                        settings.maximumSynchronizationResidualP95Ms,
                    synchronizationOutlierMaximum =
                        settings.maximumSynchronizationOutlierRatio,
                    p99MaximumMillimeters =
                        (settings.targetMotionP95ErrorMillimeters * 2.4)
                            .coerceAtLeast(8.0),
                    lagMaximumMilliseconds =
                        settings.maximumTemporalLagMilliseconds,
                    dropoutMaximum = settings.maximumDropoutRatio,
                    calibrationMinimum = settings.minimumCalibrationConfidence,
                    viewDiversityMinimum = settings.minimumViewDiversity,
                    crossViewMinimum = settings.minimumCrossViewAgreement,
                    minimumActiveCameras =
                        if (
                            settings.referenceGradePrecisionEnabled ||
                                settings.ultraPrecisionEnabled
                        ) 2 else 1,
                    timingAuthorityMinimum =
                        if (
                            settings.referenceGradePrecisionEnabled ||
                                settings.ultraPrecisionEnabled
                        ) 0.90 else 0.0,
                    focusConditioningMinimum =
                        if (
                            settings.referenceGradePrecisionEnabled ||
                                settings.ultraPrecisionEnabled
                        ) 0.65 else 0.0,
                    referenceFocusConditioningMinimum =
                        if (
                            settings.referenceGradePrecisionEnabled ||
                                settings.ultraPrecisionEnabled
                        ) 0.90 else 0.0,
                    timingAuthorityResidualP95MaximumMilliseconds =
                        if (
                            settings.referenceGradePrecisionEnabled ||
                                settings.ultraPrecisionEnabled
                        ) 2.0 else 50.0,
                ),
            )

        val tuning = baseTuning.copy(
            responsiveness = (
                0.58 +
                    settings.motionSensitivity * 0.50 +
                    settings.preserveMicroMotion * 0.14
                ).coerceIn(0.58, 1.18),
            motionSensitivity = (
                0.72 +
                    settings.motionSensitivity * 0.92
                ).coerceIn(0.72, 1.60),
            predictionHorizonSeconds = (
                settings.predictionHorizonMs / 1000.0 *
                    (0.78 + settings.temporalContinuityPriority * 0.42)
                ).coerceIn(0.04, 0.35)
                .coerceAtLeast(0.08),
            occlusionPersistence = (
                settings.occlusionPersistenceMs / 180.0 *
                    (0.82 + settings.temporalContinuityPriority * 0.30)
                ).coerceIn(0.50, 2.50),
            jitterSuppression = (
                settings.jitterSuppression *
                    (0.62 + settings.preserveMicroMotion * 0.38)
                ).coerceIn(0.0, 1.0),
            velocityResponsiveness = (
                baseTuning.velocityResponsiveness *
                    (0.76 + settings.preserveMicroMotion * 0.34)
                ).coerceIn(0.55, 1.60),
            accelerationResponsiveness = (
                baseTuning.accelerationResponsiveness *
                    (0.72 + settings.preserveMicroMotion * 0.42)
                ).coerceIn(0.55, 1.60),
            innovationGateElasticity = (
                baseTuning.innovationGateElasticity *
                    (0.78 + (1.0 - settings.motionSensitivity) * 0.22)
                ).coerceIn(0.65, 1.60),
            adaptiveRegimeStrength = (
                baseTuning.adaptiveRegimeStrength *
                    (0.70 + settings.temporalContinuityPriority * 0.30)
                ).coerceIn(0.0, 1.0),
        )

        val cameraCoverage =
            (
                evidence.connectedCameras.toDouble() /
                    profile.multiViewConsensusThreshold.coerceAtLeast(1)
            ).coerceIn(0.0, 1.0)

        val allocation = FeatureAllocation(
            motion = (
                0.60 +
                    settings.motionSensitivity * 0.22 +
                    settings.preserveMicroMotion * 0.18
                ).coerceIn(0.0, 1.0),
            temporal = settings.temporalContinuityPriority,
            surface = settings.surfaceDetailPriority,
            faceEye = settings.faceEyeDetailPriority,
            hands = settings.handDetailPriority,
            cameraCoverage = cameraCoverage,
        )

        val evidenceQuality = (
            evidence.synchronizationConfidence * 0.24 +
                evidence.calibrationConfidence * 0.24 +
                evidence.trackingConfidence * 0.26 +
                evidence.computeHeadroom * 0.16 +
                cameraCoverage * 0.10
            ).coerceIn(0.0, 1.0)

        val pressure = (1.0 - evidence.computeHeadroom)
            .coerceIn(0.0, 1.0)

        val rawScale = when {
            !settings.adaptiveQualityEnabled -> 1.0
            pressure <= 0.20 -> 1.0
            pressure >= 0.88 -> 0.42
            else -> (
                1.0 -
                    (pressure - 0.20) * 0.72
                ).coerceIn(0.42, 1.0)
        }

        val structuralFloor = when (profile.qualityTier) {
            CaptureQualityTier.BALANCED -> 0.55
            CaptureQualityTier.HIGH -> 0.62
            CaptureQualityTier.ULTRA -> 0.70
            CaptureQualityTier.REFERENCE -> 0.78
        }

        val adaptiveScale = rawScale.coerceAtLeast(
            structuralFloor * 0.72,
        ).coerceIn(0.42, 1.0)

        val metricPrecision =
            metricPrecisionEngine.evaluate(
                MetricPrecisionBudgetEngine.Input(
                    targetP95Millimeters =
                        settings.referenceGradeTargetP95ErrorMillimeters,
                    referenceGradeEnabled =
                        settings.referenceGradePrecisionEnabled,
                    referenceMinimumCalibrationConfidence =
                        settings.referenceGradeMinimumCalibrationConfidence,
                    referenceMinimumSynchronizationConfidence =
                        settings.referenceGradeMinimumSynchronizationConfidence,
                    referenceMinimumViewDiversity =
                        settings.referenceGradeMinimumViewDiversity,
                    referenceMinimumCrossViewAgreement =
                        settings.referenceGradeMinimumCrossViewAgreement,
                    workingDistanceMeters =
                        evidence.workingDistanceMeters,
                    cameraBaselineMeters =
                        evidence.cameraBaselineMeters,
                    effectiveFocalLengthPixels =
                        evidence.effectiveFocalLengthPixels,
                    reprojectionErrorPixels =
                        evidence.reprojectionErrorPixels,
                    triangulationParallaxDegrees =
                        maxOf(
                            evidence.triangulationParallaxDegrees,
                            settings.referenceGradeMinimumParallaxDegrees,
                        ),
                    depthUncertaintyMeters =
                        evidence.depthUncertaintyMeters,
                    poseTranslationUncertaintyMeters =
                        evidence.poseTranslationUncertaintyMeters,
                    poseAngularUncertaintyDegrees =
                        evidence.poseAngularUncertaintyDegrees,
                    calibrationConfidence =
                        evidence.calibrationConfidence,
                    synchronizationConfidence =
                        evidence.synchronizationConfidence,
                    viewDiversity =
                        evidence.viewDiversity,
                    crossViewAgreement =
                        evidence.crossViewAgreement,
                    temporalSkewMilliseconds =
                        evidence.temporalSkewMilliseconds,
                    timingAuthorityConfidence =
                        timingAuthority.confidence,
                    timingAuthorityResidualP95Milliseconds =
                        timingAuthority.residualP95Millis,
                    frameDurationMilliseconds =
                        evidence.frameDurationMilliseconds,
                    rollingShutterFraction =
                        evidence.rollingShutterFraction,
                    motionSpeedMetersPerSecond =
                        evidence.motionSpeedMetersPerSecond,
                    motionAccelerationMetersPerSecondSquared =
                        evidence.motionAccelerationMetersPerSecondSquared,
                    metricScaleUncertaintyRatio =
                        evidence.metricScaleUncertaintyRatio,
                    reconstructionPriorUncertaintyMeters =
                        evidence.reconstructionPriorUncertaintyMeters,
                    observationCount =
                        evidence.connectedCameras.coerceAtLeast(1),
                    occlusionRatio =
                        evidence.occlusionRisk,
                    identityContinuity =
                        evidence.identityContinuity,
                    thermalRisk =
                        evidence.thermalRisk,
                    memoryPressure =
                        evidence.memoryPressure,
                    surfaceTemporalStability =
                        evidence.surfaceTemporalStability,
                ),
            )

        val precisionCriticalPath =
            precisionCriticalPathEngine.evaluate(
                PrecisionCriticalPathEngine.Input(
                    metrology = metricPrecision,
                    targetP95Millimeters =
                        settings.targetMotionP95ErrorMillimeters,
                    targetP99Millimeters =
                        (settings.targetMotionP95ErrorMillimeters * 2.4)
                            .coerceAtLeast(8.0),
                    calibrationConfidence =
                        evidence.calibrationConfidence,
                    synchronizationConfidence =
                        evidence.synchronizationConfidence,
                    viewDiversity =
                        evidence.viewDiversity,
                    crossViewAgreement =
                        evidence.crossViewAgreement,
                    identityContinuity =
                        evidence.identityContinuity,
                    temporalLagMilliseconds =
                        evidence.temporalLagMilliseconds,
                    rollingShutterRisk =
                        evidence.rollingShutterRisk,
                    occlusionRisk =
                        evidence.occlusionRisk,
                    metricScaleDriftPercentPerMinute =
                        evidence.metricScaleDriftPercentPerMinute,
                    thermalRisk =
                        evidence.thermalRisk,
                    memoryPressure =
                        evidence.memoryPressure,
                    surfaceTemporalStability =
                        evidence.surfaceTemporalStability,
                    empiricalP95Millimeters =
                        evidence.empiricalBenchmarkP95ErrorMillimeters,
                    empiricalP99Millimeters =
                        evidence.empiricalBenchmarkP99ErrorMillimeters,
                ),
            )

        val precisionCertification =
            precisionCertificationEngine.evaluate(
                PrecisionCertificationEngine.Input(
                    metrology = metricPrecision,
                    empiricalEvidenceEnabled =
                        evidence.empiricalBenchmarkAccepted,
                    empiricalSampleCount =
                        evidence.empiricalBenchmarkSampleCount,
                    empiricalP95Millimeters =
                        evidence.empiricalBenchmarkP95ErrorMillimeters,
                    empiricalP99Millimeters =
                        evidence.empiricalBenchmarkP99ErrorMillimeters,
                    calibrationConfidence =
                        evidence.calibrationConfidence,
                    synchronizationConfidence =
                        evidence.synchronizationConfidence,
                    viewDiversity =
                        evidence.viewDiversity,
                    crossViewAgreement =
                        evidence.crossViewAgreement,
                    identityContinuity =
                        evidence.identityContinuity,
                    temporalLagMilliseconds =
                        evidence.temporalLagMilliseconds,
                    dropoutRatio =
                        evidence.dropoutRatio,
                    rollingShutterRisk =
                        evidence.rollingShutterRisk,
                    occlusionRisk =
                        evidence.occlusionRisk,
                    metricScaleDriftPercentPerMinute =
                        evidence.metricScaleDriftPercentPerMinute,
                    thermalRisk =
                        evidence.thermalRisk,
                    memoryPressure =
                        evidence.memoryPressure,
                    surfaceTemporalStability =
                        evidence.surfaceTemporalStability,
                    referenceUncertaintyMillimeters =
                        evidence.externalReferenceUncertaintyMillimeters,
                    referenceConditionsMatched =
                        evidence.matchedReferenceCaptureConditions,
                ),
            )

        val captureAuthorityEngineConfigured =
            HighEndCaptureAuthorityEngine(
                HighEndCaptureAuthorityEngine.Config(
                    maximumMotionBlurRisk =
                        settings.maximumMotionBlurRiskForAuthority,
                    minimumEpipolar =
                        settings.minimumEpipolarConsistencyForAuthority,
                    minimumMetricEvidence =
                        settings.minimumMetricEvidenceAuthorityForAuthority,
                    minimumAuthorityScore =
                        settings.minimumCaptureAuthorityScore,
                    minimumPrecisionScore =
                        settings.precisionAuthorityMinimumScore,
                    minimumPrecisionEvidenceSamples =
                        settings.precisionAuthorityMinimumSamples,
                ),
            )
        val captureAuthority =
            captureAuthorityEngineConfigured.evaluate(
                HighEndCaptureAuthorityEngine.Input(
                    timestampNanos = evidence.timestampNanos,
                    connectedCameras = evidence.connectedCameras,
                    authoritativeCameras = evidence.authoritativeCameras,
                    calibrationConfidence = evidence.calibrationConfidence,
                    synchronizationConfidence =
                        evidence.synchronizationConfidence,
                    timingAuthorityConfidence =
                        timingAuthority.confidence,
                    temporalConsistency =
                        evidence.global4DTemporalConsistency,
                    viewDiversity = evidence.viewDiversity,
                    crossViewAgreement = evidence.crossViewAgreement,
                    epipolarConsistency = evidence.epipolarConsistency,
                    bodyMetricEvidenceAuthority =
                        evidence.bodyMetricEvidenceAuthority,
                    faceMetricEvidenceAuthority =
                        evidence.faceMetricEvidenceAuthority,
                    handMetricEvidenceAuthority =
                        evidence.handMetricEvidenceAuthority,
                    faceEpipolarConsistency =
                        evidence.faceEpipolarConsistency,
                    handEpipolarConsistency =
                        evidence.handEpipolarConsistency,
                    crossModalMetricAuthority =
                        evidence.crossModalMetricAuthority,
                    crossModalEpipolarAuthority =
                        evidence.crossModalEpipolarAuthority,
                    reconstructionUncertaintyMillimeters =
                        evidence.depthUncertaintyMeters * 1000.0,
                    motionP95ErrorMillimeters =
                        evidence.motionP95ErrorMillimeters,
                    motionP99ErrorMillimeters =
                        evidence.motionP99ErrorMillimeters,
                    temporalLagMilliseconds =
                        evidence.temporalLagMilliseconds,
                    rollingShutterRisk =
                        evidence.rollingShutterRisk,
                    cameraMotionRisk =
                        evidence.cameraMotionRisk,
                    motionBlurRisk =
                        evidence.motionBlurRisk,
                    identityContinuity =
                        evidence.identityContinuity,
                    occlusionRisk =
                        evidence.occlusionRisk,
                    contactConsistency =
                        minOf(
                            evidence.contactConsistency,
                            evidence.handObjectContactConfidence,
                            evidence.interactionTemporalConsistency,
                            evidence.interactionCollisionConfidence,
                            1.0 - evidence.contactPenetrationRisk,
                        ),
                    thermalHeadroom = evidence.thermalHeadroom,
                    memoryHeadroom = evidence.memoryHeadroom,
                    gpuHeadroom = evidence.gpuHeadroom,
                    npuHeadroom = evidence.npuHeadroom,
                    cpuHeadroom = evidence.cpuHeadroom,
                    transportHeadroom =
                        transportBudget
                            ?.networkHealth
                            ?.coerceIn(0.0, 1.0)
                            ?: run {
                                val latencyRisk =
                                    evidence.networkLatencyMillisByCamera
                                        .values
                                        .takeIf { it.isNotEmpty() }
                                        ?.average()
                                        ?.div(200.0)
                                        ?.coerceIn(0.0, 1.0)
                                        ?: 0.0
                                val packetLossRisk =
                                    evidence.networkPacketLossByCamera
                                        .values
                                        .takeIf { it.isNotEmpty() }
                                        ?.average()
                                        ?.div(0.05)
                                        ?.coerceIn(0.0, 1.0)
                                        ?: 0.0
                                (
                                    1.0 -
                                        (
                                            0.48 * latencyRisk +
                                                0.42 * packetLossRisk
                                        )
                                    ).coerceIn(0.0, 1.0)
                            },
                    observedGeometryRatio =
                        evidence.observedGeometryRatio,
                    inferredGeometryRatio =
                        evidence.inferredGeometryRatio,
                    metricScaleUncertaintyRatio =
                        evidence.metricScaleUncertaintyRatio,
                    empiricalBenchmarkAccepted =
                        evidence.empiricalBenchmarkAccepted,
                    empiricalBenchmarkSampleCount =
                        evidence.empiricalBenchmarkSampleCount,
                    empiricalP95Millimeters =
                        evidence.empiricalBenchmarkP95ErrorMillimeters,
                    referenceUncertaintyMillimeters =
                        evidence.externalReferenceUncertaintyMillimeters,
                    referenceConditionsMatched =
                        evidence.matchedReferenceCaptureConditions,
                    externalReferenceDocumented =
                        evidence.externalReferenceDocumented,
                    targetP95Millimeters =
                        if (settings.ultraPrecisionEnabled) {
                            minOf(
                                settings.targetMotionP95ErrorMillimeters,
                                settings.ultraPrecisionTargetP95ErrorMillimeters,
                            )
                        } else {
                            settings.targetMotionP95ErrorMillimeters
                        }.coerceAtLeast(0.10),
                    targetRequiresIndependentReference =
                        (
                            settings.ultraPrecisionEnabled &&
                                settings.ultraPrecisionMatchedReferenceRequired
                        ) ||
                            (
                                settings.referenceGradePrecisionEnabled &&
                                    settings.frontierPrecisionQualificationRequiresReference
                            ),
                    distributedPlan =
                        evidence.distributedRigPlanForAuthority,
                ),
            )

        val referenceGradeProtocol =
            referenceGradeProtocolEngine.evaluate(
                ReferenceGradeCalibrationProtocolEngine.Input(
                    workflowReady =
                        evidence.referenceCalibrationWorkflowReady,
                    rigScore =
                        evidence.referenceCalibrationRigScore,
                    coverage =
                        evidence.referenceCalibrationCoverage,
                    reprojectionErrorPx =
                        evidence.referenceCalibrationReprojectionErrorPx,
                    clockSkewMilliseconds =
                        evidence.referenceCalibrationClockSkewMilliseconds,
                    temporalAlignmentConfidence =
                        evidence.referenceTemporalAlignmentConfidence,
                    timingAuthorityConfidence =
                        timingAuthority.confidence,
                    timingAuthorityResidualP95Milliseconds =
                        timingAuthority.residualP95Millis,
                    timingAuthorityReferenceGrade =
                        timingAuthority.referenceGrade,
                    driftStability =
                        evidence.referenceDriftStability,
                    recalibrationRequired =
                        evidence.referenceRecalibrationRequired,
                    scaleDriftPercentPerMinute =
                        evidence.metricScaleDriftPercentPerMinute,
                    temperatureDeltaCelsius =
                        evidence.referenceThermalTemperatureDeltaCelsius,
                    thermalStability =
                        evidence.referenceThermalStability,
                    observabilityScore =
                        evidence.referenceObservabilityScore,
                    metrology = metricPrecision,
                ),
            )

        val referenceGradeEvidence =
            referenceGradeEvidenceGateEngine.evaluate(
                ReferenceGradeEvidenceGateEngine.Input(
                    enabled = settings.referenceGradePrecisionEnabled,
                    matchedCaptureConditions =
                        evidence.matchedReferenceCaptureConditions,
                    empiricalBenchmarkAccepted =
                        evidence.empiricalBenchmarkAccepted,
                    sampleCount =
                        evidence.empiricalBenchmarkSampleCount,
                    measuredP95Millimeters =
                        evidence.empiricalBenchmarkP95ErrorMillimeters,
                    measuredP99Millimeters =
                        evidence.empiricalBenchmarkP99ErrorMillimeters,
                    targetP95Millimeters =
                        settings.referenceGradeTargetP95ErrorMillimeters,
                    targetP99Millimeters =
                        (settings.referenceGradeTargetP95ErrorMillimeters * 2.0)
                            .coerceAtLeast(4.0),
                    referenceUncertaintyMillimeters =
                        evidence.externalReferenceUncertaintyMillimeters,
                    externalReferenceDocumented =
                        evidence.externalReferenceDocumented,
                ),
            )

        val multiCameraTimingUnsafe =
            evidence.connectedCameras >= 2 &&
                timingAuthority.confidence < 0.35

        val precisionTimingUnsafe =
            (
                settings.referenceGradePrecisionEnabled ||
                    settings.ultraPrecisionEnabled
            ) &&
                timingAuthority.confidence < 0.35

        val unsafeEvidence =
            (
                multiCameraTimingUnsafe ||
                    precisionTimingUnsafe
            ) ||
                evidence.synchronizationConfidence < 0.45 ||
                evidence.calibrationConfidence < 0.45 ||
                evidence.trackingConfidence < 0.35 ||
                evidence.computeHeadroom < 0.12 ||
                evidence.runtimeGovernorBlocked ||
                qualityDecision.hardBlock

        val contractBreached =
            evidence.motionP95ErrorMillimeters >
                settings.targetMotionP95ErrorMillimeters ||
                evidence.temporalLagMilliseconds >
                settings.maximumTemporalLagMilliseconds ||
                evidence.dropoutRatio >
                settings.maximumDropoutRatio ||
                evidence.viewDiversity <
                settings.minimumViewDiversity ||
                evidence.crossViewAgreement <
                settings.minimumCrossViewAgreement ||
                evidence.calibrationConfidence <
                settings.minimumCalibrationConfidence

        val lockBlocked = settings.qualityLockEnabled &&
            (unsafeEvidence || pressure > 0.55 || contractBreached)
        val referenceGradeEvidenceBlocked =
            settings.referenceGradePrecisionEnabled &&
                settings.qualityLockEnabled &&
                !referenceGradeEvidence.ready

        val referenceGradeMetricScaleInsufficient =
            settings.referenceGradePrecisionEnabled &&
                (
                    evidence.metricScaleRecovery == null ||
                        evidence.metricScaleRecovery.mode ==
                            CalibrationFreeMetricRecoveryEngine.Mode.BLOCKED ||
                        evidence.metricScaleRecovery.mode ==
                            CalibrationFreeMetricRecoveryEngine.Mode.RELATIVE_ONLY ||
                        (
                            evidence.metricScaleRecovery.mode ==
                                CalibrationFreeMetricRecoveryEngine.Mode.SCALE_ANCHORED_METRIC &&
                                evidence.metricScaleRecovery.confidence < 0.90
                            )
                    )

        val adaptiveApplied =
            settings.adaptiveQualityEnabled &&
                adaptiveScale < 0.999

        val safeScale = if (settings.qualityLockEnabled) {
            1.0
        } else {
            minOf(
                adaptiveScale,
                qualityDecision.adaptiveScale,
            ).coerceIn(0.42, 1.0)
        }

        val reasonCodes = buildList {
            if (evidence.synchronizationConfidence < 0.70) {
                add("sync-evidence-degraded")
            }
            if (evidence.calibrationConfidence < 0.70) {
                add("calibration-evidence-degraded")
            }
            if (evidence.trackingConfidence < 0.70) {
                add("tracking-evidence-degraded")
            }
            if (evidence.computeHeadroom < 0.45) {
                add("compute-headroom-limited")
            }
            if (adaptiveApplied) {
                add("adaptive-budget-active")
            }
            if (!settings.interleavedTemporalFusionEnabled) {
                add("interleaved-temporal-disabled-by-policy")
            } else if (
                interleavedRuntime.latestPlan != null &&
                !interleavedRuntime.latestPlan.publishable
            ) {
                add("interleaved-window-not-publishable")
            } else if (
                settings.temporalSupersamplingPriority >= 0.80 &&
                evidence.connectedCameras >= 2
            ) {
                add("interleaved-temporal-priority-active")
            }
            if (contractBreached) {
                add("precision-contract-breached")
            }
            interleavedRuntime.warnings.forEach {
                add("interleaved:" + it.lowercase())
            }
            qualityDecision.warnings.forEach {
                add("quality:" + it)
            }
            qualityDecision.blockers.forEach {
                add("quality-block:" + it)
            }
            timingAuthority.warnings.forEach {
                add("timing-authority:" + it)
            }
            focusConditioning.warnings.forEach {
                add("focus-conditioning:" + it)
            }
            if (qualityDecision.contractSatisfied) {
                add("production-quality-contract-satisfied")
            }
            if (lockBlocked) {
                add("quality-lock-blocked")
            }
            if (
                cameraCoverage < 0.50 &&
                profile.reconstructionMode.name != "MONOCULAR_FIRST"
            ) {
                add("multi-view-coverage-limited")
            }
            if (evidenceQuality < 0.55) {
                add("capture-evidence-below-operational-target")
            }
            if (evidence.runtimeGovernorBlocked) {
                add("runtime-governor-blocked")
            }
            metricPrecision.warnings.forEach {
                add("metrology:" + it)
            }
            metricPrecision.blockers.forEach {
                add("metrology-block:" + it)
            }
            if (metricPrecision.referenceGradeFeasible) {
                add("reference-grade-2mm-feasible")
            }
            precisionCertification.warnings.forEach {
                add("precision-certification:" + it)
            }
            precisionCertification.blockers.forEach {
                add("precision-certification-block:" + it)
            }
            if (precisionCertification.referenceGradeFeasible) {
                add("conservative-2mm-budget-feasible")
            }
            precisionCriticalPath.reasons.forEach {
                add("critical-path:" + it)
            }
            precisionCriticalPath.dominantTerm?.let {
                add("critical-term:" + it.name.lowercase())
            }
            if (precisionCriticalPath.blocked) {
                add("critical-path-blocked")
            }
            if (
                precisionCertification.modeledOnly &&
                settings.referenceGradePrecisionEnabled
            ) {
                add("reference-grade-requires-empirical-certification")
            }
            if (referenceGradeMetricScaleInsufficient) {
                add("reference-grade-metric-scale-insufficient")
                evidence.metricScaleRecovery?.warnings?.forEach {
                    add("metric-scale:" + it)
                }
            }
            if (settings.referenceGradePrecisionEnabled &&
                !referenceGradeEvidence.ready
            ) {
                add("reference-grade-empirical-evidence-required")
                referenceGradeEvidence.blockingReasons.forEach {
                    add("reference-evidence-block:" + it)
                }
            }
            if (settings.referenceGradePrecisionEnabled) {
                referenceGradeProtocol.warnings.forEach {
                    add("reference-protocol:" + it)
                }
                referenceGradeProtocol.blockingReasons.forEach {
                    add("reference-protocol-block:" + it)
                }
                if (referenceGradeProtocol.ready) {
                    add("reference-grade-protocol-ready")
                }
            }
        }

        fun scaleInt(value: Int, floor: Int = 1): Int =
            (value.toDouble() * safeScale)
                .toInt()
                .coerceIn(floor, value)

        val captureFps = scaleInt(profile.targetCaptureFps, 15)
        val inferenceFps = scaleInt(
            minOf(profile.inferenceFps, captureFps),
            1,
        )

        val meshBudget = (
            profile.meshVertexBudget.toDouble() *
                (0.55 + 0.45 * settings.surfaceDetailPriority) *
                safeScale
            ).toInt().coerceIn(
                4_000,
                profile.meshVertexBudget,
            )

        return Plan(
            profile = profile,
            tuning = tuning,
            effectiveCaptureFps = captureFps,
            effectiveInferenceFps = inferenceFps,
            denseMeshHz = scaleInt(profile.denseMeshHz),
            faceMeshHz = scaleInt(profile.faceMeshHz),
            eyeTrackingHz = scaleInt(profile.eyeTrackingHz),
            handTrackingHz = scaleInt(profile.handTrackingHz),
            meshVertexBudget = meshBudget,
            interleavedTemporalFusionEnabled =
                settings.interleavedTemporalFusionEnabled,
            interleavedTemporalWindowMillis =
                settings.interleavedTemporalWindowMillis,
            interleavedPredictionHorizonMillis =
                settings.interleavedPredictionHorizonMillis,
            interleavedMaximumSampleSkewMillis =
                settings.interleavedMaximumSampleSkewMillis,
            interleavedOutlierThresholdPixels =
                settings.interleavedOutlierThresholdPixels,
            temporalSupersamplingPriority =
                settings.temporalSupersamplingPriority,
            interleavedEffectiveSamplingMultiplier =
                interleavedRuntime.effectiveSamplingMultiplier,
            interleavedTemporalHealth =
                interleavedRuntime.temporalHealth,
            interleavedWindowCoverage =
                interleavedRuntime.coverageRatio,
            interleavedWindowPublishable =
                interleavedRuntime.latestPlan?.publishable == true,
            interleavedWarnings =
                interleavedRuntime.warnings,
            featureAllocation = allocation.copy(
                motion = (
                    allocation.motion *
                        (0.82 + safeScale * 0.18)
                    ).coerceIn(0.0, 1.0),
                surface = (allocation.surface * safeScale)
                    .coerceIn(0.0, 1.0),
                faceEye = (allocation.faceEye * safeScale)
                    .coerceIn(0.0, 1.0),
                hands = (allocation.hands * safeScale)
                    .coerceIn(0.0, 1.0),
            ),
            adaptiveScale = safeScale,
            adaptiveDegradationApplied = adaptiveApplied,
            captureAllowed =
                !lockBlocked &&
                    !captureAuthority.hardBlocked &&
                    !precisionCertification.hardBlock &&
                    !referenceGradeEvidenceBlocked &&
                    !referenceGradeMetricScaleInsufficient &&
                    qualityDecision.allowed &&
                    (
                        !settings.referenceGradePrecisionEnabled ||
                            (
                                !settings.qualityLockEnabled ||
                                    referenceGradeProtocol.ready
                            )
                    ) &&
                    (
                        !settings.referenceGradePrecisionEnabled ||
                            !settings.qualityLockEnabled ||
                            !metricPrecision.hardBlock
                    ),
            reasonCodes =
                (
                    reasonCodes +
                        captureAuthority.blockCodes.map {
                            "capture-authority-block:" + it.name
                        } +
                        captureAuthority.warnings.map {
                            "capture-authority-warning:" + it
                        } +
                        listOf(
                            "capture-authority-band:" +
                                captureAuthority.band.name.lowercase(),
                        )
                ).distinct(),
            productionQualityBand = qualityDecision.band,
            productionQualityScore = qualityDecision.score,
            productionQualityContractSatisfied =
                qualityDecision.contractSatisfied,
            productionRecommendedFrameStride =
                qualityDecision.recommendedFrameStride,
            productionRecommendedWorkers =
                qualityDecision.recommendedWorkers,
            productionRuntimeBudget =
                qualityDecision.runtimeBudget,
            captureAuthority = captureAuthority,
            metricPrecision = metricPrecision,
            precisionCertification = precisionCertification,
            precisionCriticalPath = precisionCriticalPath,
            metricScaleRecovery = evidence.metricScaleRecovery,
            referenceGradeEvidence = referenceGradeEvidence,
            referenceGradeProtocol = referenceGradeProtocol,
        )
    }

    fun resolveWithInfraredReference(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        noxConditions:
            com.nivukx.nox.core.reconstruction.CaptureConditionFingerprintEngine.Result,
        infraredConditions:
            com.nivukx.nox.core.reconstruction.CaptureConditionFingerprintEngine.Result,
        noxReports:
            List<com.nivukx.nox.core.reconstruction.CapturePrecisionBenchmarkEngine.Report>,
        infraredReports:
            List<com.nivukx.nox.core.reconstruction.CapturePrecisionBenchmarkEngine.Report>,
    ): Plan {
        val comparison =
            InfraredReferenceQualificationEngine().evaluate(
                noxConditions = noxConditions,
                infraredConditions = infraredConditions,
                noxReports = noxReports,
                infraredReports = infraredReports,
            )
        val plan = resolve(settings, evidence)
        val required = settings.ultraPrecisionMatchedReferenceRequired
        return plan.copy(
            infraredReferenceQualification = comparison,
            captureAllowed =
                plan.captureAllowed &&
                    (!required || comparison.qualified),
            reasonCodes = buildList {
                addAll(plan.reasonCodes)
                comparison.warnings.forEach {
                    add("infrared:" + it)
                }
                comparison.blockers.forEach {
                    add("infrared-block:" + it)
                }
                if (comparison.noxPhysicallyFavored) {
                    add("infrared-reference:nox-lower-error")
                }
                if (comparison.referencePhysicallyFavored) {
                    add("infrared-reference:optical-reference-lower-error")
                }
                if (comparison.statisticalParity) {
                    add("infrared-reference:statistical-parity")
                }
                if (required && !comparison.qualified) {
                    add("infrared-reference:quality-lock-blocked")
                }
            }.distinct(),
        )
    }

    fun resolveWithTemporalCalibration(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        observations:
            List<SpatiotemporalCalibrationEngine.Observation>,
        cameraIds: List<String> = observations.map { it.cameraId }.distinct(),
    ): Plan {
        observations.forEach {
            spatiotemporalCalibrationEngine.add(it)
        }

        val results =
            cameraIds.map {
                spatiotemporalCalibrationEngine.fit(it)
            }

        val contractResults =
            results.filter {
                it.valid &&
                    kotlin.math.abs(it.rowReadoutMilliseconds) <=
                        settings.maximumRollingShutterReadoutMilliseconds
            }
        val synchronizationConfidence =
            if (contractResults.isEmpty()) {
                0.0
            } else {
                contractResults.map { it.subFrameConfidence }.average()
            }
        val residualP95 =
            contractResults.maxOfOrNull {
                it.residualP95Milliseconds.takeIf { value -> value.isFinite() }
                    ?: 50.0
            } ?: 50.0
        val outlierRatio =
            contractResults.maxOfOrNull {
                it.outlierRatio
            } ?: 1.0

        val plan = resolve(
            settings = settings,
            evidence = evidence.copy(
                synchronizationConfidence =
                    minOf(
                        evidence.synchronizationConfidence,
                        synchronizationConfidence,
                    ),
                synchronizationResidualP95Milliseconds =
                    maxOf(
                        evidence.synchronizationResidualP95Milliseconds,
                        residualP95.coerceAtLeast(0.0),
                    ),
                synchronizationOutlierRatio =
                    maxOf(
                        evidence.synchronizationOutlierRatio,
                        outlierRatio.coerceIn(0.0, 1.0),
                    ),
            ),
        )

        return plan.copy(
            temporalCalibrationEvidence =
                contractResults.minByOrNull {
                    it.residualP95Milliseconds
                } ?: results.firstOrNull(),
            reasonCodes = buildList {
                addAll(plan.reasonCodes)
                if (contractResults.size < results.size) {
                    add("temporal-calibration-contract-breached")
                }
                results.flatMap { it.warningCodes }
                    .distinct()
                    .forEach {
                        add("temporal-calibration:" + it)
                    }
                if (results.any { !it.valid }) {
                    add("temporal-calibration-validation-blocked")
                }
            }.distinct(),
        )
    }

    fun resolveWithSurfaceEvidence(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        frame: SurfaceDeformationConsistencyEngine.Frame,
    ): Plan {
        val currentSurfaceConfig =
            surfaceDeformationConsistencyEngine.configuration()
        surfaceDeformationConsistencyEngine.configure(
            currentSurfaceConfig.copy(
                hardStrainRatio =
                    settings.maximumSurfaceStrainP95Ratio,
                highQualityConfidence =
                    settings.minimumSurfaceTemporalStability,
            ),
        )

        val surfaceEvidence =
            surfaceDeformationConsistencyEngine.evaluate(frame)

        val strainRisk =
            (
                surfaceEvidence.strainP95Ratio / 0.35
                ).coerceIn(0.0, 1.0)
        val normalRisk =
            (
                surfaceEvidence.normalP95Radians / 1.75
                ).coerceIn(0.0, 1.0)
        val curvatureRisk =
            (
                surfaceEvidence.curvatureP95Delta / 2.5
                ).coerceIn(0.0, 1.0)

        val plan = resolve(
            settings = settings,
            evidence = evidence.copy(
                surfaceDeformationStability =
                    minOf(
                        evidence.surfaceDeformationStability,
                        surfaceEvidence.temporalStability,
                    ),
                surfaceTopologyContinuity =
                    minOf(
                        evidence.surfaceTopologyContinuity,
                        surfaceEvidence.topologyContinuity,
                    ),
                surfaceStrainRisk =
                    maxOf(
                        evidence.surfaceStrainRisk,
                        strainRisk,
                    ),
                surfaceNormalRisk =
                    maxOf(
                        evidence.surfaceNormalRisk,
                        normalRisk,
                    ),
                surfaceCurvatureRisk =
                    maxOf(
                        evidence.surfaceCurvatureRisk,
                        curvatureRisk,
                    ),
                surfaceTemporalStability =
                    minOf(
                        evidence.surfaceTemporalStability,
                        surfaceEvidence.temporalStability,
                    ),
                meshTopologyIntegrity =
                    minOf(
                        evidence.meshTopologyIntegrity,
                        surfaceEvidence.topologyContinuity,
                    ),
            ),
        )

        val surfaceContractBreached =
            surfaceEvidence.strainP95Ratio >
                settings.maximumSurfaceStrainP95Ratio ||
                surfaceEvidence.temporalStability <
                settings.minimumSurfaceTemporalStability ||
                surfaceEvidence.topologyContinuity <
                settings.minimumSurfaceTopologyContinuity

        return plan.copy(
            surfaceDeformationEvidence = surfaceEvidence,
            captureAllowed =
                plan.captureAllowed &&
                    !surfaceEvidence.hardViolation &&
                    !surfaceContractBreached,
            reasonCodes = buildList {
                addAll(plan.reasonCodes)
                surfaceEvidence.warnings.forEach {
                    add("surface:" + it)
                }
                if (surfaceEvidence.accepted) {
                    add("surface-deformation-evidence-accepted")
                }
                if (surfaceEvidence.hardViolation ||
                    surfaceContractBreached
                ) {
                    add("surface-deformation-contract-breached")
                }
            }.distinct(),
        )
    }

    fun resolveWithInteractionEvidence(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        observations:
            List<ContactAwareReconstructionEngine.InteractionObservation>,
        timestampNanos: Long,
    ): Plan {
        val interactionEvidence =
            contactAwareReconstructionEngine.evaluate(
                observations = observations,
                timestampNanos = timestampNanos,
            )

        val interactionConfidence =
            minOf(
                interactionEvidence.productionContactConfidence,
                interactionEvidence.meanInteractionConfidence,
            ).coerceIn(0.0, 1.0)

        val penetrationRisk =
            if (interactionEvidence.maximumPenetrationMillimeters <= 0.0) {
                0.0
            } else {
                (
                    interactionEvidence.maximumPenetrationMillimeters /
                        settings.maximumContactPenetrationMillimeters
                    ).coerceIn(0.0, 1.0)
            }

        val contactContractBreached =
            interactionEvidence.maximumPenetrationMillimeters >
                settings.maximumContactPenetrationMillimeters ||
                (
                    interactionEvidence.contactPairCount > 0 &&
                        interactionEvidence.meanInteractionConfidence <
                        settings.minimumHandObjectContactConfidence
                    )

        val interactionAwarePlan =
            resolve(
                settings = settings,
                evidence = evidence.copy(
                    contactConsistency =
                        minOf(
                            evidence.contactConsistency,
                            interactionConfidence,
                        ),
                    handObjectContactConfidence =
                        minOf(
                            evidence.handObjectContactConfidence,
                            interactionEvidence.meanInteractionConfidence,
                        ),
                    interactionTemporalConsistency =
                        minOf(
                            evidence.interactionTemporalConsistency,
                            interactionEvidence.temporalConsistency,
                        ),
                    interactionCollisionConfidence =
                        minOf(
                            evidence.interactionCollisionConfidence,
                            interactionEvidence.collisionConfidence,
                        ),
                    contactPenetrationRisk =
                        maxOf(
                            evidence.contactPenetrationRisk,
                            penetrationRisk,
                        ),
                ),
            )

        return interactionAwarePlan.copy(
            handObjectContactEvidence = interactionEvidence,
            captureAllowed =
                interactionAwarePlan.captureAllowed &&
                    !contactContractBreached,
            reasonCodes = buildList {
                addAll(interactionAwarePlan.reasonCodes)
                interactionEvidence.warnings.forEach {
                    add("interaction:" + it)
                }
                if (interactionEvidence.contactPairCount > 0) {
                    add("hand-object-contact-active")
                }
                if (contactContractBreached) {
                    add("hand-object-contact-contract-breached")
                }
                if (interactionEvidence.trustedContactPairs.isNotEmpty()) {
                    add("hand-object-contact-trusted")
                }
            }.distinct(),
        )
    }

    data class RigAwarePlan(
        val plan: Plan,
        val health: CaptureRigObservabilityEngine.Health,
        val selection: CaptureRigObservabilityEngine.Selection,
    )

    fun resolveWithRig(
        settings: HighEndMotionCaptureSettings,
        evidence: RuntimeEvidence = RuntimeEvidence(),
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
        requestedCameraCount: Int = cameras.size.coerceAtLeast(1),
    ): RigAwarePlan {
        val health = rigObservabilityEngine.analyze(cameras)
        val selection = rigObservabilityEngine.select(
            cameras = cameras,
            requestedCount = requestedCameraCount,
        )

        val selectedPairwise =
            rigObservabilityEngine.pairwise(selection.selected)

        val effectiveBaseline =
            selectedPairwise
                .maxByOrNull { it.utility }
                ?.baselineMeters
                ?: evidence.cameraBaselineMeters

        val rigViewDiversity = (
            health.angularDiversity * 0.55 +
                health.sectorCoverage * 0.30 +
                health.roleDiversity * 0.15
            ).coerceIn(0.0, 1.0)

        val rigCrossViewAgreement = minOf(
            evidence.crossViewAgreement,
            health.meanHealth,
            selection.objectiveScore,
        )

        val plan = resolve(
            settings = settings,
            evidence = evidence.copy(
                connectedCameras = selection.selected.size,
                cameraBaselineMeters =
                    maxOf(
                        evidence.cameraBaselineMeters,
                        effectiveBaseline,
                    ),
                viewDiversity = rigViewDiversity,
                crossViewAgreement = rigCrossViewAgreement,
            ),
        ).copy(
            cameraRigObservabilityScore =
                health.observabilityScore,
            selectedCameraIds =
                selection.selected.map { it.cameraId },
            cameraRigWarnings =
                (health.warnings + selection.warnings).distinct(),
        )

        return RigAwarePlan(
            plan = plan,
            health = health,
            selection = selection,
        )
    }

    fun resolveWithMotionEvidence(
        settings: HighEndMotionCaptureSettings,
        input: AdaptiveMotionEvidenceEngine.Input,
        nominalFrameRateHz: Double = input.nominalFrameRateHz,
    ): MotionEvidencePlan {
        val evidence =
            motionEvidenceEngine.process(
                input.copy(
                    nominalFrameRateHz = nominalFrameRateHz
                        .coerceIn(1.0, 240.0),
                ),
            )

        temporalFusionEngine.configure(
            HighEndTemporalFusionEngine.Config(
                maxTracks = settings.temporalFusionMaxTracks,
                maxObservationsPerFrame =
                    settings.temporalFusionMaxObservationsPerFrame,
                maxPredictionMilliseconds =
                    settings.temporalFusionPredictionBudgetMillis,
                trackRetentionMilliseconds =
                    settings.temporalFusionTrackRetentionMillis,
                hardOutlierDistance =
                    settings.temporalFusionHardOutlierGate,
                huberDelta =
                    settings.temporalFusionHuberDelta,
                confidenceHalfLifeMilliseconds =
                    settings.temporalFusionConfidenceHalfLifeMillis,
            ),
        )

        var temporalFusionStateReset = false
        val occlusionContactEvidence =
            occlusionContactEvidenceEngine.evaluate(
                OcclusionContactEvidenceEngine.Input(
                    timestampNanos = evidence.timestampNanos,
                    actorId = 0L,
                    landmarks = input.landmarks.map {
                        OcclusionContactEvidenceEngine.JointObservation(
                            landmarkId = it.id,
                            x = it.x,
                            y = it.y,
                            confidence = it.confidence,
                            visible = it.confidence >= 0.20,
                        )
                    },
                    periodicMotionScore =
                        evidence.periodicMotionScore,
                    motionFrequencyHz =
                        evidence.frequencyProxyHz,
                    momentumConfidence =
                        evidence.temporalConfidence,
                ),
            )

        val temporalFusionSummary =
            try {
                temporalFusionEngine.update(
                    timestampNanos = evidence.timestampNanos,
                    observations =
                        input.landmarks.map {
                            HighEndTemporalFusionEngine.Observation(
                                trackId = it.id,
                                timestampNanos = evidence.timestampNanos,
                                point =
                                    HighEndTemporalFusionEngine.Point3(
                                        x = it.x,
                                        y = it.y,
                                        z = 0.0,
                                    ),
                                confidence = it.confidence,
                                cameraId = "primary",
                                visible = it.confidence >= 0.10,
                                occluded = evidence.occlusionRisk > 0.35,
                                covarianceTrace =
                                    evidence.occlusionRisk.coerceIn(0.0, 1.0),
                            )
                        },
                )
            } catch (error: IllegalArgumentException) {
                if (!error.message.orEmpty().contains("strictly increasing")) {
                    throw error
                }
                temporalFusionStateReset = true
                temporalFusionEngine.reset()
                temporalFusionEngine.update(
                    timestampNanos = evidence.timestampNanos,
                    observations =
                        input.landmarks.map {
                            HighEndTemporalFusionEngine.Observation(
                                trackId = it.id,
                                timestampNanos = evidence.timestampNanos,
                                point =
                                    HighEndTemporalFusionEngine.Point3(
                                        x = it.x,
                                        y = it.y,
                                        z = 0.0,
                                    ),
                                confidence = it.confidence,
                                cameraId = "primary",
                                visible = it.confidence >= 0.10,
                                occluded = evidence.occlusionRisk > 0.35,
                                covarianceTrace =
                                    evidence.occlusionRisk.coerceIn(0.0, 1.0),
                            )
                        },
                )
            }

        val plan = resolve(
            settings = settings,
            evidence = RuntimeEvidence(
                synchronizationConfidence =
                    evidence.temporalConfidence,
                calibrationConfidence =
                    (1.0 - evidence.rollingShutterRisk)
                        .coerceIn(0.0, 1.0),
                trackingConfidence =
                    minOf(
                        evidence.confidence,
                        temporalFusionSummary.meanConfidence,
                        occlusionContactEvidence.captureTrust,
                    ),
                temporalLagMilliseconds =
                    evidence.rollingShutterRisk * 25.0 +
                        evidence.predictedRisk * 15.0,
                dropoutRatio =
                    evidence.occlusionRisk,
                viewDiversity =
                    (1.0 - evidence.cameraMotionRisk * 0.55)
                        .coerceIn(0.0, 1.0),
                crossViewAgreement =
                    (1.0 - evidence.commonModeEnergy * 0.75)
                        .coerceIn(0.0, 1.0),
                connectedCameras = 1,
                timestampNanos = evidence.timestampNanos,
                identityContinuity = evidence.identityContinuity,
                rollingShutterRisk = evidence.rollingShutterRisk,
                occlusionRisk = evidence.occlusionRisk,
                global4DTemporalConsistency =
                    minOf(
                        1.0,
                        temporalFusionSummary.temporalConsistency,
                        occlusionContactEvidence.captureTrust,
                    ),
                contactConsistency =
                    minOf(
                        evidence.identityContinuity,
                        occlusionContactEvidence.contactPreservationScore,
                    ),
            ),
        )
        return MotionEvidencePlan(
            plan = plan.copy(
                temporalFusionSummary = temporalFusionSummary,
                occlusionContactEvidence = occlusionContactEvidence,
                reasonCodes =
                    if (temporalFusionStateReset) {
                        (plan.reasonCodes + "temporal-fusion-state-reset").distinct()
                    } else {
                        plan.reasonCodes
                    } +
                    occlusionContactEvidence.warnings.map {
                        "occlusion:" + it
                    }.distinct(),
            ),
            evidence = evidence,
        )
    }

    /** Builds the canonical frontier profile without duplicating the underlying settings object. */
    fun frontierProfile(settings: HighEndMotionCaptureSettings): com.nivukx.nox.core.model.FrontierCaptureProfile =
        com.nivukx.nox.core.model.FrontierCaptureProfile(
            resolution = settings.cameraResolutionPreset,
            targetFps = settings.targetFps,
            inferenceFps = settings.inferenceFps,
            shutterMicros = settings.cameraManualExposureMicros,
            iso = settings.cameraManualIso,
            preciseColorTemperatureEnabled =
                settings.cameraPreciseColorTemperatureEnabled,
            colorTemperatureKelvin =
                settings.cameraColorTemperatureKelvin,
            colorTint =
                settings.cameraColorTint,
            stabilizationEnabled =
                settings.cameraOisEnabled ||
                    settings.cameraElectronicStabilizationEnabled,
            rawEnabled = settings.cameraRawSidecarEnabled,
            depthEnabled = settings.cameraDepthSidecarEnabled,
            denseSurfaceEnabled = settings.denseMeshDetail > 0.0,
            gaussianRefinementEnabled = settings.surfaceDetailPriority > 0.0,
            sparseViewRecoveryEnabled = settings.occlusionRecoveryPriority > 0.0,
            worldSpaceLearnedProposalsEnabled = true,
            targetActorCount = 50,
            hardActorCeiling = 64,
            authoritativeCameraLimit = settings.distributedAuthoritativeCameraBudget.coerceIn(2, 32),
            globalCameraLimit = settings.distributedGlobalCameraEnvelope.coerceIn(2, 128),
            precisionTargetMillimeters =
                settings.ultraPrecisionTargetP95ErrorMillimeters,
            minimumIndependentViews =
                settings.distributedAuthoritativeCameraBudget.coerceIn(2, 32),
            minimumEmpiricalSamples =
                settings.ultraPrecisionMinimumEvidenceSamples.coerceAtLeast(600),
            maximumMotionBlurRiskForAuthority =
                settings.maximumMotionBlurRiskForAuthority,
            minimumEpipolarConsistencyForAuthority =
                settings.minimumEpipolarConsistencyForAuthority,
            minimumMetricEvidenceAuthorityForAuthority =
                settings.minimumMetricEvidenceAuthorityForAuthority,
            minimumCaptureAuthorityScore =
                settings.minimumCaptureAuthorityScore,
            precisionAuthorityMinimumScore =
                settings.precisionAuthorityMinimumScore,
            precisionAuthorityMinimumSamples =
                settings.precisionAuthorityMinimumSamples,
            referenceQualificationRequired =
                settings.frontierPrecisionQualificationRequiresReference,
            reducedTransparency = false,
            reducedMotion = false,
        )

    @Synchronized
    fun ingestInterleavedTemporalSample(
        sample: InterleavedTemporalSamplingEngine.CameraSample,
    ): InterleavedTemporalSamplingEngine.WindowPlan? =
        interleavedTemporalSamplingEngine.ingest(sample)

    @Synchronized
    fun interleavedTemporalSummary():
        InterleavedTemporalSamplingEngine.RuntimeSummary =
        interleavedTemporalSamplingEngine.summary()

    override fun close() {
        interleavedTemporalSamplingEngine.close()
        temporalFusionEngine.close()
        motionEvidenceEngine.close()
        occlusionContactEvidenceEngine.close()
        surfaceDeformationConsistencyEngine.close()
        timingAuthorityFusionEngine.close()
        qualityGate.close()
        super.close()
    }

    private fun tuningFor(
        profile: ProductionCaptureProfile,
    ): MotionCaptureTuning =
        when (profile.id) {
            "CINEMATIC_ULTRA" -> MotionCaptureTuning.cinematic()
            "LIVE_ACTION" -> MotionCaptureTuning.action()
            "MULTI_VIEW_PRECISION" -> MotionCaptureTuning.precision()
            "FACE_PERFORMANCE" -> MotionCaptureTuning.facial()
            "SUSTAINED_MOBILE" -> MotionCaptureTuning(
                responsiveness = 0.88,
                motionSensitivity = 1.05,
                predictionHorizonSeconds = 0.16,
                occlusionPersistence = 1.22,
                reacquisitionRadius = 0.14,
                minimumConfidence = 0.08,
                rejectAggressiveness = 0.64,
                jitterSuppression = 0.48,
                velocityResponsiveness = 1.06,
                accelerationResponsiveness = 1.02,
                innovationGateElasticity = 1.28,
                motionThresholdScale = 0.94,
                adaptiveRegimeStrength = 0.94,
            )
            "REFERENCE_BAKE" -> MotionCaptureTuning.precision().copy(
                responsiveness = 0.84,
                motionSensitivity = 1.02,
                predictionHorizonSeconds = 0.08,
                jitterSuppression = 0.86,
                rejectAggressiveness = 0.94,
                adaptiveRegimeStrength = 0.92,
            )
            else -> MotionCaptureTuning.action()
        }
}
