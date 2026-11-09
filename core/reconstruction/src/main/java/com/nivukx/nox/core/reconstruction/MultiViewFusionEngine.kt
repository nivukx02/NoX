package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max

/**
 * Complete 2D -> calibrated multi-view -> 3D skeleton bridge.
 *
 * Calibration is explicit input. NoX never invents camera pose or metric depth. Once a rig snapshot
 * is available, each phone's 2D landmarks are converted into calibrated observations and fused.
 */
class MultiViewFusionEngine(
    private val calibrationProvider: () -> CalibrationSnapshot?,
    private val buffer: ObservationFusionBuffer = ObservationFusionBuffer(),
    private val skeleton: SkeletonReconstructionEngine = SkeletonReconstructionEngine(),
    private val quality: ReconstructionQualityAnalyzer = ReconstructionQualityAnalyzer(),
    private val temporal: TemporalReconstructionStabilizer =
        TemporalReconstructionStabilizer(),
    private val continuity: ReconstructionContinuityController =
        ReconstructionContinuityController(),
    private val reprojectionConsistency: MultiViewReprojectionConsistencyEngine =
        MultiViewReprojectionConsistencyEngine(),
    private val rigGeometryDrift: LiveRigGeometryDriftEngine =
        LiveRigGeometryDriftEngine(),
    private val geometricObservability: MultiViewGeometricObservabilityEngine =
        MultiViewGeometricObservabilityEngine(),
    private val uncertaintyPropagation: ReconstructionUncertaintyPropagationEngine =
        ReconstructionUncertaintyPropagationEngine(),
    private val kinematicConstraintEngine: HighEndKinematicConstraintEngine =
        HighEndKinematicConstraintEngine(),
    private val physicsConstraintEngine: PhysicsConstrainedSkeletonEngine =
        PhysicsConstrainedSkeletonEngine(),
    private val worldSpaceMotionConsistencyEngine: WorldSpaceMotionConsistencyEngine =
        WorldSpaceMotionConsistencyEngine(),
    private val global4DConsistencyEngine: Global4DConsistencyEngine =
        Global4DConsistencyEngine(),
    private val adaptiveEvidenceWeightEngine: AdaptiveMultiViewEvidenceWeightEngine =
        AdaptiveMultiViewEvidenceWeightEngine(),
) {

    private val correlatedEvidenceFusionEngine =
        CorrelatedEvidenceFusionEngine()

    @Volatile
    private var gaussian4DEvidenceProvider:
        (() -> Global4DConsistencyEngine.Gaussian4DEvidence?)? =
        null

    private val bodyJointUncertaintyGateEngine =
        BodyJointUncertaintyGateEngine()

    private val denseSurfaceReconstructionEngine =
        DenseMultiViewSurfaceReconstructionEngine()

    private val crossViewLandmarkIdentityGateEngine =
        CrossViewLandmarkIdentityGateEngine()

    private val perObservationGeometricConditioningEngine =
        PerObservationGeometricConditioningEngine()

    private val multiPersonIdentityExchangeRiskEngine =
        MultiPersonIdentityExchangeRiskEngine()

    private val ultraPrecisionEvidenceEngine =
        UltraPrecisionEvidenceEngine()

    private val distributedSupportEvidenceEngine =
        DistributedSupportEvidenceEngine()

    @Volatile
    private var ultraPrecisionPolicy =
        UltraPrecisionEvidenceEngine.Policy.disabled()

    @Volatile
    private var ultraPrecisionEmpiricalP95Millimeters: Double? = null

    @Volatile
    private var ultraPrecisionEmpiricalP99Millimeters: Double? = null

    @Volatile
    private var ultraPrecisionEmpiricalSampleCount: Int = 0

    @Volatile
    private var ultraPrecisionMatchedReference = false

    @Volatile
    private var ultraPrecisionReferenceKind =
        UltraPrecisionEvidenceEngine.ReferenceKind.NONE

    @Volatile
    private var ultraPrecisionReferenceP95Millimeters: Double? = null

    @Volatile
    private var ultraPrecisionReferenceUncertaintyMillimeters: Double? = null

    @Volatile
    private var latestDenseIdentityGateQuality = 1.0

    @Volatile
    private var latestDenseIdentityGateRejectedCount = 0

    @Volatile
    private var latestDenseIdentityGateWarnings: List<String> = emptyList()

    @Volatile
    private var latestDenseIdentityGateEvaluatedPoints = 0

    @Volatile
    private var latestDenseGeometricConditioningScore = 1.0

    @Volatile
    private var latestDenseGeometricConditioningMedianParallaxDegrees = 0.0

    @Volatile
    private var latestDenseGeometricConditioningMinimumObservationScore = 1.0

    @Volatile
    private var latestDenseGeometricConditioningWarnings: List<String> =
        emptyList()

    data class DenseLandmarkInput(
        val subjectId: Int,
        val landmarkId: Int,
        val normalizedX: Double,
        val normalizedY: Double,
        val confidence: Double,
        val uncertainty: Double,
        val visibility: Double,
        val contactProbability: Double,
        val temporalStability: Double,
        val crossViewAgreement: Double,
        val reprojectionResidualNormalized: Double,
        val predicted: Boolean,
        val frameHeightPx: Int = 0,
        val rollingShutterSkewNanos: Long = 0L,
        val frameDurationNanos: Long = 0L,
        val exposureTimeNanos: Long = 0L,
        val captureTimingConfidence: Double = 0.0,
        val probabilitySigmaXNormalized: Double = 0.0,
        val probabilitySigmaYNormalized: Double = 0.0,
        val probabilityCalibrationScore: Double = 1.0,
        val silhouetteConsistency: Double = 1.0,
    ) {
        init {
            require(silhouetteConsistency in 0.0..1.0)
        }
    }

    private data class DenseLandmarkObservation(
        val subjectId: Int,
        val landmarkId: Int,
        val normalizedX: Double,
        val normalizedY: Double,
        val confidence: Double,
        val uncertainty: Double,
        val visibility: Double,
        val contactProbability: Double,
        val temporalStability: Double,
        val crossViewAgreement: Double,
        val reprojectionResidualNormalized: Double,
        val predicted: Boolean,
        val frameHeightPx: Int = 0,
        val rollingShutterSkewNanos: Long = 0L,
        val frameDurationNanos: Long = 0L,
        val exposureTimeNanos: Long = 0L,
        val captureTimingConfidence: Double = 0.0,
        val probabilitySigmaXNormalized: Double = 0.0,
        val probabilitySigmaYNormalized: Double = 0.0,
        val probabilityCalibrationScore: Double = 1.0,
        val silhouetteConsistency: Double = 1.0,
    ) {
        init {
            require(silhouetteConsistency in 0.0..1.0)
        }
    }

    private data class DenseLandmarkDeviceFrame(
        val deviceId: DeviceId,
        val timestampNanos: Long,
        val landmarks:
            List<DenseLandmarkObservation>,
    )

    private val denseLandmarkHistory =
        LinkedHashMap<DeviceId, ArrayDeque<DenseLandmarkDeviceFrame>>()

    @Volatile
    private var latestDenseSurfaceResult:
        DenseMultiViewSurfaceReconstructionEngine.Result? =
        null

    @Volatile
    private var latestIdentityExchangeResult =
        MultiPersonIdentityExchangeRiskEngine.Result(
            state = MultiPersonIdentityExchangeRiskEngine.State.CLEAR,
            productionIdentityConfidence = 1.0,
            exchangeRisk = 0.0,
            temporalContinuity = 1.0,
            occlusionBurden = 0.0,
            ambiguity = 0.0,
            evaluatedSubjects = 0,
            heldSubjects = 0,
            subjectRisks = emptyList(),
            warnings = emptyList(),
        )

    private val interleavedSpatiotemporalFusionEngine =
        InterleavedSpatiotemporalFusionEngine()

    private val interleavedTemporalSamplingEngine =
        InterleavedTemporalSamplingEngine()

    @Volatile
    private var latestInterleavedTemporalSamplingSummary =
        interleavedTemporalSamplingEngine.summary()

    @Volatile
    private var cameraContributionPlanner =
        MultiViewCameraContributionPlanner()

    private val rollingShutterEvidenceEngine =
        RollingShutterEvidenceEngine()

    private val spatiotemporalBundleAdjustmentEngine =
        SpatiotemporalBundleAdjustmentEngine()

    private val reconstructionHistory =
        ArrayDeque<SkeletonFrame>()

    @Volatile
    private var latestSpatiotemporalBundleResult:
        SpatiotemporalBundleAdjustmentEngine.Result? = null

    @Volatile
    private var subframeAlignmentStrictness =
        0.82

    data class BodyEvidenceSnapshot(
        val evaluatedObservationCount: Int = 0,
        val acceptedObservationCount: Int = 0,
        val rejectedObservationCount: Int = 0,
        val meanFusionWeight: Double = 0.0,
        val meanUncertaintyMeters: Double = 0.0,
        val timingRiskObservationCount: Int = 0,
        val boundaryRiskObservationCount: Int = 0,
    ) {
        val acceptedRatio: Double
            get() =
                acceptedObservationCount.toDouble() /
                    evaluatedObservationCount.coerceAtLeast(1).toDouble()
    }

    @Volatile
    private var latestBodyEvidenceSnapshot =
        BodyEvidenceSnapshot()

    @Volatile
    private var latestCameraFrames:
        List<SkeletonReconstructionEngine.CameraFrame> = emptyList()

    @Volatile
    private var latestCalibrationSnapshot: CalibrationSnapshot? = null

    @Volatile
    private var latestCameraContributionPlan =
        MultiViewCameraContributionPlanner.Plan(
            selectedDeviceIds = emptySet(),
            excludedDeviceIds = emptySet(),
            rankedCameras = emptyList(),
            diversitySatisfied = false,
            reason = "uninitialized",
        )

    @Synchronized
    fun configureInterleavedTemporalFusion(
        config: InterleavedSpatiotemporalFusionEngine.Config,
    ) {
        interleavedSpatiotemporalFusionEngine.configure(config)
        interleavedTemporalSamplingEngine.configure(
            InterleavedTemporalSamplingEngine.Config(
                maximumCameras = 128,
                targetCameraCountPerWindow =
                    config.minimumSamplesForAuthoritativeOutput.coerceIn(2, 32),
                nominalCameraFrameIntervalMillis = 16.6667,
                maximumWindowMillis =
                    (config.maximumTemporalWindowNanos / 1_000_000.0).coerceAtLeast(16.6667),
                predictionHorizonMillis =
                    config.maximumPredictionNanos / 1_000_000.0,
                maximumSampleSkewMillis =
                    (config.maximumSampleSkewNanos / 1_000_000.0).coerceAtLeast(1.0),
                maximumTimingResidualMillis =
                    (config.maximumSampleSkewNanos / 2_000_000.0).coerceIn(0.5, 25.0),
                minimumTemporalHealth = if (config.enabled) 0.62 else 0.0,
            ),
        )
    }

    @Synchronized
    fun configureUltraPrecisionPolicy(
        policy: UltraPrecisionEvidenceEngine.Policy,
    ) {
        ultraPrecisionPolicy = policy
    }

    @Synchronized
    fun configureUltraPrecisionEmpiricalEvidence(
        empiricalP95Millimeters: Double?,
        empiricalP99Millimeters: Double?,
        empiricalSampleCount: Int,
        matchedReference: Boolean,
        referenceKind: UltraPrecisionEvidenceEngine.ReferenceKind =
            UltraPrecisionEvidenceEngine.ReferenceKind.NONE,
        referenceP95Millimeters: Double? = null,
        referenceUncertaintyMillimeters: Double? = null,
    ) {
        require(empiricalSampleCount >= 0)
        empiricalP95Millimeters?.let {
            require(it.isFinite() && it >= 0.0)
        }
        empiricalP99Millimeters?.let {
            require(it.isFinite() && it >= 0.0)
        }
        referenceP95Millimeters?.let {
            require(it.isFinite() && it >= 0.0)
        }
        referenceUncertaintyMillimeters?.let {
            require(it.isFinite() && it >= 0.0)
        }

        ultraPrecisionEmpiricalP95Millimeters =
            empiricalP95Millimeters
        ultraPrecisionEmpiricalP99Millimeters =
            empiricalP99Millimeters
        ultraPrecisionEmpiricalSampleCount =
            empiricalSampleCount
        ultraPrecisionMatchedReference =
            matchedReference
        ultraPrecisionReferenceKind =
            referenceKind
        ultraPrecisionReferenceP95Millimeters =
            referenceP95Millimeters
        ultraPrecisionReferenceUncertaintyMillimeters =
            referenceUncertaintyMillimeters
    }

    fun ultraPrecisionPolicy():
        UltraPrecisionEvidenceEngine.Policy =
        ultraPrecisionPolicy

    fun configureFromMotionCaptureCustomization(
        customization: MotionCaptureCustomization,
    ) {
        skeleton.configurePrecisionTuning(
            TriangulationEngine.PrecisionTuning(
                tailResidualQuantile =
                    (
                        0.90 +
                            0.09 *
                                customization.denseLandmarkUncertaintyStrictness
                        ).coerceIn(0.50, 0.999),
                residualInflation =
                    (
                        1.05 +
                            0.30 *
                                customization.surfaceEvidenceStrictness
                        ).coerceIn(1.0, 3.0),
                minimumPixelNoisePx =
                    (
                        0.40 -
                            0.20 *
                                customization.denseLandmarkCrossViewStrictness
                        ).coerceIn(0.05, 2.0),
            ),
        )

        configureUltraPrecisionPolicy(
            UltraPrecisionEvidenceEngine.Policy(
                enabled = customization.ultraPrecisionEnabled,
                qualificationMode =
                    customization.ultraPrecisionQualificationMode,
                targetP95Millimeters =
                    customization.ultraPrecisionTargetP95ErrorMillimeters,
                targetP99Millimeters =
                    customization.ultraPrecisionTargetP99ErrorMillimeters,
                minimumEvidenceSamples =
                    customization.ultraPrecisionMinimumEvidenceSamples,
                minimumBaselineMeters =
                    customization.ultraPrecisionMinimumBaselineMeters,
                minimumParallaxDegrees =
                    customization.ultraPrecisionMinimumParallaxDegrees,
                minimumCalibrationConfidence =
                    customization.ultraPrecisionMinimumCalibrationConfidence,
                minimumSynchronizationConfidence =
                    customization.ultraPrecisionMinimumSynchronizationConfidence,
                minimumViewDiversity =
                    customization.ultraPrecisionMinimumViewDiversity,
                minimumCrossViewAgreement =
                    customization.ultraPrecisionMinimumCrossViewAgreement,
                maximumOcclusionRatio =
                    customization.ultraPrecisionMaximumOcclusionRatio,
                minimumIdentityContinuity =
                    customization.ultraPrecisionMinimumIdentityContinuity,
                minimumSurfaceTemporalStability =
                    customization.ultraPrecisionMinimumSurfaceTemporalStability,
                maximumThermalRisk =
                    customization.ultraPrecisionMaximumThermalRisk,
                maximumMemoryPressure =
                    customization.ultraPrecisionMaximumMemoryPressure,
                matchedReferenceRequired =
                    customization.ultraPrecisionMatchedReferenceRequired,
            ),
        )

        bodyJointUncertaintyGateEngine
            .configureFromMotionCaptureCustomization(
                customization,
            )
        correlatedEvidenceFusionEngine.configure(
            when {
                !customization.correlatedEvidenceEnabled ->
                    CorrelatedEvidenceFusionEngine.Config.production().copy(
                        minimumAcceptanceCoverage = 0.0,
                        maximumAcceptanceCommonModeRisk = 1.0,
                        maximumFailureRisk = 1.0,
                    )
                customization.correlatedEvidenceQualificationMode ->
                    CorrelatedEvidenceFusionEngine.Config.qualification().copy(
                        minimumAcceptanceCoverage =
                            customization.correlatedEvidenceMinimumCoverage,
                        maximumAcceptanceCommonModeRisk =
                            customization.correlatedEvidenceMaximumCommonModeRisk,
                    )
                else ->
                    CorrelatedEvidenceFusionEngine.Config.production().copy(
                        minimumAcceptanceCoverage =
                            customization.correlatedEvidenceMinimumCoverage,
                        maximumAcceptanceCommonModeRisk =
                            customization.correlatedEvidenceMaximumCommonModeRisk,
                    )
            },
        )

        multiPersonIdentityExchangeRiskEngine.configure(
            MultiPersonIdentityExchangeRiskEngine.Config(
                enabled =
                    customization.identityExchangeRiskEnabled,
                maximumSubjects =
                    customization.maxSubjects.coerceIn(
                        1,
                        64,
                    ),
                historyFrames =
                    customization.identityExchangeHistoryFrames,
                highRiskThreshold =
                    customization.identityExchangeHighRiskThreshold,
                holdThreshold =
                    customization.identityExchangeHoldThreshold,
            ),
        )

        configureInterleavedTemporalFusion(
            InterleavedSpatiotemporalFusionEngine.Config(
                enabled = customization.interleavedTemporalFusionEnabled,
                maximumTemporalWindowNanos =
                    (customization.interleavedTemporalWindowMillis * 1_000_000.0)
                        .toLong()
                        .coerceAtLeast(1L),
                maximumPredictionNanos =
                    (customization.interleavedPredictionHorizonMillis * 1_000_000.0)
                        .toLong()
                        .coerceAtLeast(0L),
                maximumSampleSkewNanos =
                    (customization.interleavedMaximumSampleSkewMillis * 1_000_000.0)
                        .toLong()
                        .coerceAtLeast(1L),
                outlierDistancePixels =
                    customization.interleavedOutlierThresholdPixels,
            ),
        )
    }

    fun configureContinuity(tuning: ReconstructionContinuityTuning) {
        continuity.configure(tuning)
    }

    fun continuityTuning(): ReconstructionContinuityTuning =
        continuity.tuning()

    fun configureTemporalTuning(tuning: TemporalReconstructionTuning) {
        temporal.configure(tuning)
    }

    fun configureObservability(
        tuning: SkeletonReconstructionEngine.ObservabilityTuning,
    ) {
        skeleton.configureObservability(tuning)
    }

    fun observabilityTuning():
        SkeletonReconstructionEngine.ObservabilityTuning =
        skeleton.observabilityTuning()

    fun temporalTuning(): TemporalReconstructionTuning =
        temporal.tuning()

    fun configureKinematicTuning(
        tuning: HighEndKinematicConstraintEngine.Config,
    ) {
        kinematicConstraintEngine.configure(tuning)
    }

    fun kinematicTuning():
        HighEndKinematicConstraintEngine.Config =
        kinematicConstraintEngine.configuration()

    fun configurePhysicsConstraints(
        tuning: PhysicsConstrainedSkeletonEngine.Config,
        edges: List<PhysicsConstrainedSkeletonEngine.Edge> =
            PhysicsConstrainedSkeletonEngine.defaultEdges(),
    ) {
        physicsConstraintEngine.configure(
            newConfig = tuning,
            newEdges = edges,
        )
    }

    fun physicsConstraintTuning():
        PhysicsConstrainedSkeletonEngine.Config =
        physicsConstraintEngine.currentConfig()

    fun configureGlobal4DConsistency(
        tuning: Global4DConsistencyEngine.Config,
    ) {
        global4DConsistencyEngine.configure(tuning)
    }

    fun configureGaussian4DEvidenceProvider(
        provider:
            (() -> Global4DConsistencyEngine.Gaussian4DEvidence?)?,
    ) {
        gaussian4DEvidenceProvider =
            provider
    }

    fun configureWorldSpaceMotionConsistency(
        tuning: WorldSpaceMotionConsistencyEngine.Config,
    ) {
        worldSpaceMotionConsistencyEngine.configure(tuning)
    }

    fun worldSpaceMotionConsistencyTuning():
        WorldSpaceMotionConsistencyEngine.Config =
        worldSpaceMotionConsistencyEngine.configuration()

    fun global4DConsistencyTuning():
        Global4DConsistencyEngine.Config =
        global4DConsistencyEngine.configuration()

    fun configureAdaptiveEvidenceTuning(
        tuning: AdaptiveMultiViewEvidenceWeightEngine.Tuning,
    ) {
        adaptiveEvidenceWeightEngine.configure(tuning)
    }

    fun adaptiveEvidenceTuning():
        AdaptiveMultiViewEvidenceWeightEngine.Tuning =
        adaptiveEvidenceWeightEngine.currentTuning()

    @Synchronized
    fun configureCameraContributionTuning(
        tuning: MultiViewCameraContributionPlanner.Config,
    ) {
        cameraContributionPlanner =
            MultiViewCameraContributionPlanner(tuning)
    }

    fun cameraContributionTuning():
        MultiViewCameraContributionPlanner.Config =
        cameraContributionPlanner.configuration()

    @Synchronized
    fun configureJointObservationTuning(
        tuning: JointObservationSubsetPlanner.Config,
    ) {
        skeleton.configureJointObservationTuning(tuning)
    }

    data class DetailedResult(
        val skeleton: SkeletonFrame,
        val diagnostics: com.nivukx.nox.core.model.ReconstructionDiagnostics,
        val accepted: Boolean,
        val ultraPrecision: UltraPrecisionEvidenceEngine.Result? = null,
        val qualityScore: Double = 0.0,
        val rejectionReasons: List<String> = emptyList(),
        val temporalQualityScore: Double = 0.0,
        val temporalRejectionReasons: List<String> = emptyList(),
        val continuityScore: Double = 0.0,
        val continuityAction: ReconstructionContinuityController.Action =
            ReconstructionContinuityController.Action.REJECT,
        val continuityReasons: List<String> = emptyList(),
        val reprojectionQualityScore: Double = 0.0,
        val reprojectionRejectionReasons: List<String> = emptyList(),
        val reprojectionBottleneck: String = "NONE",
        val rigGeometryDriftScore: Double = 1.0,
        val rigGeometryDriftState: LiveRigGeometryDriftEngine.State =
            LiveRigGeometryDriftEngine.State.UNKNOWN,
        val rigGeometryDriftWarnings: List<String> = emptyList(),
        val geometricObservabilityScore: Double = 0.0,
        val geometricObservabilityWarnings: List<String> = emptyList(),
        val geometricInformationConditionScore: Double = 0.0,
        val geometricInformationP95MajorMeters: Double = 0.0,
        val geometricInformationP95DepthMeters: Double = 0.0,
        val geometricObservabilityBottleneck: String = "NONE",
        val uncertaintyScore: Double = 0.0,
        val uncertaintyConfidence: Double = 0.0,
        val meanUncertaintyMeters: Double = 0.0,
        val p95UncertaintyMeters: Double = 0.0,
        val trustedJointRatio: Double = 0.0,
        val uncertaintyBottleneck: String = "NONE",
        val uncertaintyWarnings: List<String> = emptyList(),
        val directionalUncertaintyMajor95Meters: Double = 0.0,
        val directionalUncertaintyDepthMeters: Double = 0.0,
        val directionalUncertaintyAnisotropy: Double = 1.0,
        val directionalUncertaintyConfidence: Double = 1.0,
        val directionalUncertaintyDegradedJointCount: Int = 0,
        val directionalUncertaintyWarnings: List<String> = emptyList(),
        val kinematicScore: Double = 1.0,
        val kinematicAccepted: Boolean = true,
        val kinematicBottleneck: String = "NONE",
        val kinematicWarnings: List<String> = emptyList(),
        val physicsScore: Double = 1.0,
        val physicsAccepted: Boolean = true,
        val correlatedEvidenceScore: Double = 0.0,
        val correlatedEvidenceState:
            CorrelatedEvidenceFusionEngine.State =
            CorrelatedEvidenceFusionEngine.State.GUARDED,
        val correlatedEvidenceCoverage: Double = 0.0,
        val correlatedEvidenceFailureRisk: Double = 1.0,
        val correlatedEvidenceCommonModeRisk: Double = 1.0,
        val correlatedEvidenceRedundantSignalCount: Int = 0,
        val correlatedEvidenceWarnings: List<String> = emptyList(),
        val global4DScore: Double = 0.0,
        val global4DAccepted: Boolean = true,
        val global4DState:
            Global4DConsistencyEngine.State =
            Global4DConsistencyEngine.State.WARMING,
        val global4DPersistentFailure: Boolean = false,
        val global4DRootMotionConsistency: Double = 1.0,
        val global4DRootVelocityAgreement: Double = 1.0,
        val global4DSceneConsistency: Double = 0.72,
        val global4DDirectionalUncertaintyConsistency: Double = 1.0,
        val global4DFaceEyeConsistency: Double = 1.0,
        val global4DExternalContactConsistency: Double = 1.0,
        val global4DGaussianConsistency: Double = 1.0,
        val global4DGaussianStrainConsistency: Double = 1.0,
        val global4DGaussianEvidenceAvailable: Boolean = false,
        val global4DWarnings: List<String> = emptyList(),
        val physicsMeanBoneResidualMeters: Double = 0.0,
        val physicsP95BoneResidualMeters: Double = 0.0,
        val physicsMaxBoneResidualMeters: Double = 0.0,
        val physicsSymmetryScore: Double = 1.0,
        val physicsFloorContactScore: Double = 1.0,
        val physicsCorrectedJointCount: Int = 0,
        val physicsConstrainedEdgeCount: Int = 0,
        val physicsWarnings: List<String> = emptyList(),
        val worldSpaceConsistencyScore: Double = 1.0,
        val worldSpaceAccepted: Boolean = true,
        val worldSpaceTemporalStability: Double = 1.0,
        val worldSpaceMotionContinuity: Double = 1.0,
        val worldSpaceRejectedJointCount: Int = 0,
        val worldSpaceRobustOutlierRatio: Double = 0.0,
        val worldSpaceWarnings: List<String> = emptyList(),
        val adaptiveViewEvidenceScore: Double = 1.0,
        val adaptiveViewEvidenceGlobalRigTrust: Double = 1.0,
        val adaptiveViewEvidenceWarnings: List<String> = emptyList(),
        val adaptiveViewEvidenceDegradedCameraCount: Int = 0,
        val adaptiveViewEvidenceHoldCameraCount: Int = 0,
        val adaptiveViewEvidenceDominantCamera: String = "NONE",
        val adaptiveViewEvidenceCameraStatuses:
            List<AdaptiveMultiViewEvidenceWeightEngine.CameraStatus> =
            emptyList(),
        val cameraContributionSelectedCount: Int = 0,
        val cameraContributionExcludedCount: Int = 0,
        val cameraContributionDiversitySatisfied: Boolean = false,
        val cameraContributionInformationGainScore: Double = 0.0,
        val cameraContributionMinimumPairwiseDiversity: Double = 0.0,
        val cameraContributionViewStarvationRisk: Double = 0.0,
        val cameraContributionReason: String = "UNINITIALIZED",
        val supportEvidenceScore: Double = 0.0,
        val supportEvidenceSupportingCameraCount: Int = 0,
        val supportEvidenceRejectedCameraCount: Int = 0,
        val supportEvidenceContradictionRisk: Double = 0.0,
        val supportEvidenceSupportingJointCount: Int = 0,
        val supportEvidenceContradictedJointCount: Int = 0,
        val supportEvidenceWarnings: List<String> = emptyList(),
        val jointViewSelectionScore: Double = 0.0,
        val jointViewDiversitySatisfiedRatio: Double = 0.0,
        val jointSelectedObservationCount: Int = 0,
        val jointExcludedObservationCount: Int = 0,
        val jointViewSelectionReasons: List<String> = emptyList(),
        val spatiotemporalConfidence: Double = 0.0,
        val spatiotemporalInitialRmsPx: Double = 0.0,
        val spatiotemporalFinalRmsPx: Double = 0.0,
        val spatiotemporalOffsetsNanos: Map<String, Double> = emptyMap(),
        val spatiotemporalWarnings: List<String> = emptyList(),
        val interleavedTemporalConfidence: Double = 0.0,
        val interleavedSpatialAgreement: Double = 0.0,
        val interleavedEffectiveInputRateHz: Double = 0.0,
        val interleavedSupportingCameraCount: Int = 0,
        val interleavedAuthoritative: Boolean = false,
        val interleavedWarnings: List<String> = emptyList(),
        val interleavedSchedulerCoverage: Double = 0.0,
        val interleavedSchedulerTemporalHealth: Double = 0.0,
        val interleavedSchedulerEffectiveSamplingMultiplier: Double = 1.0,
        val interleavedSchedulerPublishable: Boolean = false,
        val interleavedSchedulerWarnings: List<String> = emptyList(),
        val bodyEvidenceAcceptedRatio: Double = 0.0,
        val bodyEvidenceRejectedObservationCount: Int = 0,
        val bodyEvidenceMeanFusionWeight: Double = 0.0,
        val bodyEvidenceMeanUncertaintyMeters: Double = 0.0,
        val bodyEvidenceTimingRiskObservationCount: Int = 0,
        val bodyEvidenceBoundaryRiskObservationCount: Int = 0,
        val denseSurfaceResult:
            DenseMultiViewSurfaceReconstructionEngine.Result? = null,
        val denseSurfacePrecisionQualified: Boolean = false,
        val denseSurfaceTailAdjustedDepthErrorP95Millimeters: Double =
            Double.POSITIVE_INFINITY,
        val denseSurfaceTailAdjustedFeasibilityScore: Double = 0.0,
        val denseIdentityGateQuality: Double = 1.0,
        val denseIdentityGateRejectedCount: Int = 0,
        val denseIdentityGateEvaluatedPoints: Int = 0,
        val denseIdentityGateWarnings: List<String> = emptyList(),
        val denseGeometricConditioningScore: Double = 1.0,
        val denseGeometricConditioningMedianParallaxDegrees: Double = 0.0,
        val denseGeometricConditioningMinimumObservationScore: Double = 1.0,
        val denseGeometricConditioningWarnings: List<String> = emptyList(),
        val identityExchangeState:
            MultiPersonIdentityExchangeRiskEngine.State =
            MultiPersonIdentityExchangeRiskEngine.State.CLEAR,
        val identityExchangeRisk: Double = 0.0,
        val identityExchangeConfidence: Double = 1.0,
        val identityExchangeEvaluatedSubjects: Int = 0,
        val identityExchangeHeldSubjects: Int = 0,
        val identityExchangeWarnings: List<String> = emptyList(),
        val bodyMetricEvidenceAuthority: Double = 0.0,
        val bodySymmetricEpipolarConsistencyScore: Double = 0.0,
        val bodyEpipolarWarnings: List<String> = emptyList(),
        val faceMetricEvidenceAuthority: Double = 1.0,
        val handMetricEvidenceAuthority: Double = 1.0,
        val faceEpipolarConsistency: Double = 1.0,
        val handEpipolarConsistency: Double = 1.0,
        val crossModalMetricAuthority: Double = 1.0,
        val crossModalEpipolarAuthority: Double = 1.0,
        val captureAuthorityScore: Double = 0.0,
        val captureAuthorityBand:
            HighEndCaptureAuthorityEngine.Band =
            HighEndCaptureAuthorityEngine.Band.OBSERVATION_ONLY,
        val captureAuthorityHardBlocked: Boolean = false,
        val captureAuthorityWarnings: List<String> = emptyList(),
    )

    fun push(
        deviceId: DeviceId,
        timestampNanos: Long,
        landmarks: List<com.nivukx.nox.core.model.LandmarkObservation>,
    ) {
        buffer.push(
            ObservationFusionBuffer.DeviceFrame(
                deviceId = deviceId,
                timestampNanos = timestampNanos,
                landmarks = landmarks,
            ),
        )
    }

    fun reconstruct(
        timestampNanos: Long,
        trackId: Long = 0L,
        subjectType: SubjectType = SubjectType.HUMAN,
        bodySpacePriors:
            List<BodySpacePriorAlignmentEngine.Prior> = emptyList(),
    ): SkeletonFrame? {
        latestBodyEvidenceSnapshot =
            BodyEvidenceSnapshot()
        adaptiveEvidenceWeightEngine.beginFrame()
        val calibration = calibrationProvider() ?: return null
        val frames = buffer.matchAll(timestampNanos)
        val cameraPlan =
            cameraContributionPlanner.plan(
                frames = frames.map { frame ->
                    SkeletonReconstructionEngine.CameraFrame(
                        deviceId = frame.deviceId,
                        timestampNanos = frame.timestampNanos,
                        observations = frame.landmarks.mapNotNull { landmark ->
                            if (
                                landmark.space !=
                                    com.nivukx.nox.core.model.LandmarkSpace.PIXELS
                            ) {
                                null
                            } else {
                                CameraObservation(
                                    deviceId = frame.deviceId,
                                    point = Vec2(landmark.x, landmark.y),
                                    confidence = landmark.confidence,
                                    intrinsics =
                                        calibration.intrinsics[frame.deviceId]
                                            ?: return@mapNotNull null,
                                    pose =
                                        calibration.extrinsics[frame.deviceId]
                                            ?.pose
                                            ?: return@mapNotNull null,
                                    landmarkId = landmark.landmarkId,
                                    frameDurationNanos =
                                        landmark.frameDurationNanos,
                                    exposureTimeNanos =
                                        landmark.exposureTimeNanos,
                                    captureTimingConfidence =
                                        landmark.captureTimingConfidence,
                                    calibrationUncertainty =
                                        calibration.uncertaintyProfiles[
                                            frame.deviceId
                                        ] ?: CalibrationUncertaintyProfile(),
                                )
                            }
                        },
                        frameDurationNanos =
                            frame.landmarks
                                .map { it.frameDurationNanos }
                                .filter { it > 0L }
                                .maxOrNull()
                                ?: 0L,
                        exposureTimeNanos =
                            frame.landmarks
                                .map { it.exposureTimeNanos }
                                .filter { it > 0L }
                                .maxOrNull()
                                ?: 0L,
                        rollingShutterSkewNanos =
                            frame.landmarks
                                .maxOfOrNull { it.rollingShutterSkewNanos }
                                ?: 0L,
                        captureTimingConfidence =
                            frame.landmarks
                                .map { it.captureTimingConfidence }
                                .minOrNull()
                                ?.coerceIn(0.0, 1.0)
                                ?: 1.0,
                    )
                },
                cameraStatuses =
                    adaptiveEvidenceWeightEngine
                        .snapshot()
                        .cameraStatuses,
            )
        latestCameraContributionPlan = cameraPlan

        var evaluatedBodyObservations = 0
        var acceptedBodyObservations = 0
        var rejectedBodyObservations = 0
        var bodyWeightSum = 0.0
        var bodyUncertaintySum = 0.0
        var timingRiskObservations = 0
        var boundaryRiskObservations = 0

        val cameraFrames = frames
            .filter {
                cameraPlan.selectedDeviceIds.contains(it.deviceId)
            }
            .mapNotNull { frame ->
            val intrinsics = calibration.intrinsics[frame.deviceId] ?: return@mapNotNull null
            val extrinsics = calibration.extrinsics[frame.deviceId] ?: return@mapNotNull null

            val hasTrackMetadata =
                trackId > 0L &&
                    frame.landmarks.any {
                        it.trackingId != null
                    }

            val observations = frame.landmarks
                .asSequence()
                .filter { landmark ->
                    if (!hasTrackMetadata) {
                        true
                    } else {
                        landmark.trackingId?.toLong() ==
                            trackId
                    }
                }
                .mapNotNull { landmark ->
                if (landmark.space != com.nivukx.nox.core.model.LandmarkSpace.PIXELS) {
                    return@mapNotNull null
                }

                val rollingShutterEvidence =
                    rollingShutterEvidence(landmark)

                val bodyJointEvidence =
                    bodyJointUncertaintyGateEngine.evaluate(
                        BodyJointUncertaintyGateEngine.Input(
                            normalizedX =
                                if (landmark.frameWidth > 0) {
                                    landmark.x /
                                        landmark.frameWidth.toDouble()
                                } else {
                                    0.5
                                },
                            normalizedY =
                                if (landmark.frameHeight > 0) {
                                    landmark.y /
                                        landmark.frameHeight.toDouble()
                                } else {
                                    0.5
                                },
                            observationConfidence =
                                landmark.confidence,
                            captureTimingConfidence =
                                landmark.captureTimingConfidence,
                            temporalAlignmentConfidence =
                                (
                                    landmark.captureTimingConfidence *
                                        rollingShutterEvidence.confidenceScale
                                ).coerceIn(0.0, 1.0),
                            temporalUncertaintyNanos =
                                kotlin.math.abs(
                                    landmark.timestampNanos -
                                        frame.timestampNanos,
                                ),
                            rollingShutterFraction =
                                rollingShutterEvidence.rollingShutterFraction,
                            rowTimeOffsetNanos =
                                rollingShutterEvidence.rowOffsetNanos,
                            frameDurationNanos =
                                landmark.frameDurationNanos,
                            exposureTimeNanos =
                                landmark.exposureTimeNanos,
                            predicted = false,
                            baseUncertaintyMeters =
                                (
                                    0.01 +
                                        (
                                            1.0 -
                                                landmark.confidence
                                        ) *
                                        0.20
                                ).coerceIn(0.001, 0.75),
                        ),
                    )

                evaluatedBodyObservations += 1
                bodyWeightSum += bodyJointEvidence.viewExpertWeight
                bodyUncertaintySum += bodyJointEvidence.uncertaintyMeters
                if (
                    bodyJointEvidence.reasons.any {
                        it == "temporal-alignment-risk"
                    }
                ) {
                    timingRiskObservations += 1
                }
                if (
                    bodyJointEvidence.reasons.any {
                        it == "image-boundary-risk"
                    }
                ) {
                    boundaryRiskObservations += 1
                }

                if (bodyJointEvidence.hardRejected) {
                    rejectedBodyObservations += 1
                    return@mapNotNull null
                }

                acceptedBodyObservations += 1

                CameraObservation(
                    deviceId = frame.deviceId,
                    point = Vec2(landmark.x, landmark.y),
                    confidence =
                        (
                            adaptiveEvidenceWeightEngine.weight(
                                deviceId = frame.deviceId,
                                jointId = landmark.landmarkId,
                                observationConfidence = landmark.confidence,
                            ) *
                                rollingShutterEvidence.confidenceScale *
                                bodyJointEvidence.viewExpertWeight
                            ).coerceIn(1e-4, 1.0),
                    landmarkId = landmark.landmarkId,
                    intrinsics = intrinsics,
                    pose = extrinsics.pose,
                    frameDurationNanos =
                        landmark.frameDurationNanos,
                    exposureTimeNanos =
                        landmark.exposureTimeNanos,
                    captureTimingConfidence =
                        landmark.captureTimingConfidence,
                    calibrationUncertainty =
                        calibration.uncertaintyProfiles[frame.deviceId]
                            ?: CalibrationUncertaintyProfile(),
                    rollingShutterFraction =
                        rollingShutterEvidence.rollingShutterFraction,
                    temporalAlignmentConfidence =
                        (
                            landmark.captureTimingConfidence *
                                rollingShutterEvidence.confidenceScale
                            ).coerceIn(0.0, 1.0),
                    rowTimeOffsetNanos =
                        rollingShutterEvidence.rowOffsetNanos,
                )
            }

            val observationList = observations.toList()
            if (observationList.isEmpty()) null
            else SkeletonReconstructionEngine.CameraFrame(
                deviceId = frame.deviceId,
                timestampNanos = frame.timestampNanos,
                observations = observationList,
                frameDurationNanos =
                    frame.landmarks
                        .map { it.frameDurationNanos }
                        .filter { it > 0L }
                        .maxOrNull()
                        ?: 0L,
                exposureTimeNanos =
                    frame.landmarks
                        .map { it.exposureTimeNanos }
                        .filter { it > 0L }
                        .maxOrNull()
                        ?: 0L,
                rollingShutterSkewNanos =
                    frame.landmarks
                        .maxOfOrNull { it.rollingShutterSkewNanos }
                        ?: 0L,
                captureTimingConfidence =
                    frame.landmarks
                        .map { it.captureTimingConfidence }
                        .minOrNull()
                        ?.coerceIn(0.0, 1.0)
                        ?: 1.0,
            )
        }

        latestCameraFrames = cameraFrames
        latestCalibrationSnapshot = calibration

        val geometricConditioningMultipliers =
            HashMap<String, Double>()

        val observationsByLandmark =
            cameraFrames
                .flatMap { frame ->
                    frame.observations.map { observation ->
                        (
                            observation.landmarkId
                        ) to observation
                    }
                }
                .groupBy { it.first }

        observationsByLandmark.forEach { (landmarkId, entries) ->
            if (landmarkId < 0 || entries.size < 2) {
                return@forEach
            }

            val triObservations =
                entries.map { (_, observation) ->
                    TriangulationEngine.Observation(
                        cameraPose = observation.pose,
                        intrinsics = observation.intrinsics,
                        cameraId = observation.deviceId.value,
                        u = observation.point.x,
                        v = observation.point.y,
                        weight =
                            observation.confidence.coerceIn(
                                1e-4,
                                1.0,
                            ),
                        temporalAlignmentConfidence =
                            observation.temporalAlignmentConfidence,
                        temporalUncertaintyNanos =
                            observation.temporalUncertaintyNanos,
                        rowTimeOffsetNanos =
                            observation.rowTimeOffsetNanos,
                        visibility =
                            observation.confidence.coerceIn(
                                0.05,
                                1.0,
                            ),
                        sourceQuality =
                            observation.confidence.coerceIn(
                                0.05,
                                1.0,
                            ),
                    )
                }

            val conditioning =
                perObservationGeometricConditioningEngine.evaluate(
                    triObservations,
                )

            conditioning.scores.forEach { score ->
                geometricConditioningMultipliers[
                    score.cameraId + "|" + landmarkId
                ] =
                    score.weightMultiplier
            }

            if (conditioning.globalConditioningScore < 0.45) {
                latestDenseGeometricConditioningWarnings =
                    (
                        latestDenseGeometricConditioningWarnings +
                            ("SKELETON_" + landmarkId +
                                "_GEOMETRIC_CONDITIONING_DEGRADED")
                        )
                        .distinct()
                        .takeLast(12)
            }
        }

        val conditionedCameraFrames =
            cameraFrames.map { frame ->
                SkeletonReconstructionEngine.CameraFrame(
                    deviceId = frame.deviceId,
                    timestampNanos = frame.timestampNanos,
                    observations =
                        frame.observations.map { observation ->
                            val multiplier =
                                geometricConditioningMultipliers[
                                    observation.deviceId.value + "|" +
                                        observation.landmarkId,
                                ] ?: 1.0

                            observation.copy(
                                confidence =
                                    (
                                        observation.confidence *
                                            multiplier
                                    ).coerceIn(
                                        1e-4,
                                        1.0,
                                    ),
                            )
                        },
                    frameDurationNanos = frame.frameDurationNanos,
                    exposureTimeNanos = frame.exposureTimeNanos,
                    rollingShutterSkewNanos = frame.rollingShutterSkewNanos,
                    captureTimingConfidence = frame.captureTimingConfidence,
                )
            }
        latestBodyEvidenceSnapshot =
            BodyEvidenceSnapshot(
                evaluatedObservationCount =
                    evaluatedBodyObservations,
                acceptedObservationCount =
                    acceptedBodyObservations,
                rejectedObservationCount =
                    rejectedBodyObservations,
                meanFusionWeight =
                    if (evaluatedBodyObservations == 0) {
                        0.0
                    } else {
                        bodyWeightSum /
                            evaluatedBodyObservations
                    },
                meanUncertaintyMeters =
                    if (evaluatedBodyObservations == 0) {
                        0.0
                    } else {
                        bodyUncertaintySum /
                            evaluatedBodyObservations
                    },
                timingRiskObservationCount =
                    timingRiskObservations,
                boundaryRiskObservationCount =
                    boundaryRiskObservations,
            )

        if (cameraFrames.size < 2) return null

        val detailed = skeleton.reconstructDetailed(
            timestampNanos = timestampNanos,
            subjectType = subjectType,
            trackId = trackId,
            frames = conditionedCameraFrames,
            bodySpacePriors = bodySpacePriors,
        )

        if (detailed.frame.joints.size < 2 ||
            detailed.frame.globalConfidence < 0.18
        ) {
            return null
        }

        return detailed.frame
    }


    fun reconstructDetailed(
        timestampNanos: Long,
        trackId: Long = 0L,
        subjectType: SubjectType = SubjectType.HUMAN,
        bodySpacePriors:
            List<BodySpacePriorAlignmentEngine.Prior> = emptyList(),
        clockSkewNanos: Long = 0L,
        cameraMotionRisk: Double = 0.0,
        synchronizationConfidence: Double = 1.0,
        captureTimingConfidence: Double = 0.0,
        faceEyeFusionConfidence: Double = 1.0,
        faceEyeFusionAccepted: Boolean = true,
        faceEyeOutOfDistributionRisk: Double = 0.0,
        faceEyeMetricEvidenceAuthority: Double = 1.0,
        handMetricEvidenceAuthority: Double = 1.0,
        faceEpipolarConsistency: Double = 1.0,
        handEpipolarConsistency: Double = 1.0,
        denseContactEvidenceConfidence: Double = 1.0,
        denseContactEvidenceAvailable: Boolean = false,
        lightingConfidence: Double = 1.0,
    ): DetailedResult? {
        require(faceEyeMetricEvidenceAuthority in 0.0..1.0)
        require(handMetricEvidenceAuthority in 0.0..1.0)
        require(faceEpipolarConsistency in 0.0..1.0)
        require(handEpipolarConsistency in 0.0..1.0)

        val rawSkeleton = reconstruct(
            timestampNanos = timestampNanos,
            trackId = trackId,
            subjectType = subjectType,
            bodySpacePriors = bodySpacePriors,
        ) ?: return null

        val bodyMetricEvidenceAuthority =
            skeleton.metricEvidenceAuthoritySummary()
                .coerceIn(0.0, 1.0)
        val bodySymmetricEpipolarConsistencyScore =
            skeleton.symmetricEpipolarConsistencySummary()
                .coerceIn(0.0, 1.0)

        val directionalUncertainty =
            skeleton.directionalUncertaintySummary()

        val cameraCount = buffer.matchAll(timestampNanos).size
        val calibration = calibrationProvider() ?: return null

        val framesForTemporalScheduler =
            buffer.matchAll(timestampNanos)

        val measuredFrameIntervalsMillis =
            framesForTemporalScheduler
                .asSequence()
                .flatMap { frame ->
                    frame.landmarks.asSequence()
                        .map { it.frameDurationNanos }
                }
                .filter { it > 0L }
                .map { it.toDouble() / 1_000_000.0 }
                .filter { it.isFinite() && it > 0.0 }
                .sorted()
                .toList()

        val temporalSchedulerConfig =
            interleavedTemporalSamplingEngine.configuration()

        val measuredNominalFrameIntervalMillis =
            if (measuredFrameIntervalsMillis.isEmpty()) {
                temporalSchedulerConfig.nominalCameraFrameIntervalMillis
            } else {
                measuredFrameIntervalsMillis[
                    measuredFrameIntervalsMillis.size / 2
                ]
            }

        interleavedTemporalSamplingEngine.configure(
            temporalSchedulerConfig.copy(
                nominalCameraFrameIntervalMillis =
                    measuredNominalFrameIntervalMillis.coerceIn(
                        1.0,
                        1_000.0,
                    ),
            ),
        )

        framesForTemporalScheduler.forEach { frame ->
            val calibratedPose =
                calibration.extrinsics[frame.deviceId]?.pose
            val cameraPosition =
                calibratedPose?.position
            val cameraAngle =
                cameraPosition?.let {
                    atan2(
                        it.z,
                        it.x,
                    )
                } ?: 0.0
            val sector =
                (
                    (
                        (
                            cameraAngle +
                                Math.PI
                            ) /
                            (2.0 * Math.PI)
                        ) *
                        16.0
                    ).toInt().coerceIn(0, 15)

            val confidence =
                frame.landmarks
                    .asSequence()
                    .map { it.confidence }
                    .filter { it.isFinite() }
                    .average()
                    .takeIf { it.isFinite() }
                    ?: 0.0

            val timingConfidence =
                frame.landmarks
                    .asSequence()
                    .map { it.captureTimingConfidence }
                    .filter { it.isFinite() }
                    .minOrNull()
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0

            val geometricQuality =
                frame.landmarks
                    .asSequence()
                    .map { it.crossViewAgreement }
                    .filter { it.isFinite() }
                    .average()
                    .takeIf { it.isFinite() }
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0

            interleavedTemporalSamplingEngine.ingest(
                InterleavedTemporalSamplingEngine.CameraSample(
                    cameraId = frame.deviceId.value,
                    timestampNanos = frame.timestampNanos,
                    sequence = frame.timestampNanos,
                    sector = sector,
                    confidence =
                        confidence.coerceIn(0.0, 1.0),
                    synchronized =
                        timingConfidence >= 0.80,
                    timingResidualMillis = 0.0,
                    geometricQuality = geometricQuality,
                    outlierScore =
                        (1.0 - timingConfidence)
                            .coerceIn(0.0, 1.0),
                ),
            )
        }

        latestInterleavedTemporalSamplingSummary =
            interleavedTemporalSamplingEngine.summary()

        val cameraFrames = buildReprojectionFrames(
            timestampNanos = timestampNanos,
            clockSkewNanos = clockSkewNanos,
        )

        val spatiotemporalResult =
            estimateSpatiotemporalAlignment(
                timestampNanos = timestampNanos,
                skeleton = rawSkeleton,
            )

        val interleavedTemporalResult =
            interleavedSpatiotemporalFusionEngine.fuseDetailed(
                targetTimestampNanos = timestampNanos,
                frames = buffer.matchAll(timestampNanos).map { frame ->
                    InterleavedSpatiotemporalFusionEngine.CameraFrame(
                        cameraId = frame.deviceId.value,
                        timestampNanos = frame.timestampNanos,
                        sequence = frame.timestampNanos,
                        observations = frame.landmarks.mapNotNull { landmark ->
                            if (
                                landmark.space !=
                                    com.nivukx.nox.core.model.LandmarkSpace.PIXELS
                            ) {
                                null
                            } else {
                                InterleavedSpatiotemporalFusionEngine.JointObservation(
                                    jointId = landmark.landmarkId,
                                    position = Vec2(
                                        landmark.x,
                                        landmark.y,
                                    ),
                                    confidence = landmark.confidence,
                                    visibility = 1.0,
                                )
                            }
                        },
                    )
                }.filter { it.observations.isNotEmpty() },
            )

        val temporalEvaluation = temporal.update(
            TemporalReconstructionStabilizer.FrameInput(
                timestampNanos = timestampNanos,
                joints = rawSkeleton.joints.map { joint ->
                    TemporalReconstructionStabilizer.JointObservation(
                        jointId = joint.landmarkId,
                        positionMeters =
                            TemporalReconstructionStabilizer.Vec3(
                                x = joint.position.x,
                                y = joint.position.y,
                                z = joint.position.z,
                            ),
                        confidence = joint.confidence,
                        uncertaintyMeters = joint.positionUncertaintyMeters,
                        reprojectionErrorPx = joint.reprojectionErrorPx,
                        viewCount = joint.observationCount,
                        timestampNanos = timestampNanos,
                    )
                },
                camerasUsed = cameraCount,
                clockSkewNanos = clockSkewNanos,
                globalConfidence = rawSkeleton.globalConfidence,
            ),
        )

        val stabilizedById = temporalEvaluation.joints.associateBy {
            it.jointId
        }

        val stabilizedSkeleton = rawSkeleton.copy(
            joints = rawSkeleton.joints.map { joint ->
                val stable = stabilizedById[joint.landmarkId]
                if (stable == null) {
                    joint
                } else {
                    joint.copy(
                        position = com.nivukx.nox.core.model.Vec3(
                            stable.positionMeters.x,
                            stable.positionMeters.y,
                            stable.positionMeters.z,
                        ),
                        velocity = com.nivukx.nox.core.model.Vec3(
                            stable.velocityMetersPerSecond.x,
                            stable.velocityMetersPerSecond.y,
                            stable.velocityMetersPerSecond.z,
                        ),
                        confidence = stable.confidence,
                        positionUncertaintyMeters =
                            stable.uncertaintyMeters,
                    )
                }
            },
            globalConfidence = (
                0.58 * rawSkeleton.globalConfidence +
                    0.42 * temporalEvaluation.score
            ).coerceIn(0.0, 1.0),
        )

        val reprojectionFrames = buildReprojectionFrames(
            timestampNanos = timestampNanos,
            clockSkewNanos = clockSkewNanos,
        )

        val reprojectionEvaluation = reprojectionConsistency.evaluate(
            MultiViewReprojectionConsistencyEngine.Input(
                skeleton = stabilizedSkeleton,
                cameraFrames = reprojectionFrames,
                targetTimestampNanos = timestampNanos,
                calibrationReprojectionErrorPx =
                    calibrationProvider()?.reprojectionErrorPx ?: 0.0,
                synchronizationConfidence =
                    (
                        (
                            if (cameraCount > 1) {
                                (
                                    1.0 -
                                        kotlin.math.abs(clockSkewNanos) /
                                            4_000_000.0
                                    ).coerceIn(0.0, 1.0)
                            } else {
                                0.0
                            }
                        ) *
                            (
                                0.72 +
                                    0.28 *
                                        spatiotemporalResult.confidence
                                            .coerceIn(0.0, 1.0)
                            )
                        ).coerceIn(0.0, 1.0),
            ),
        )

        val adaptiveEvidenceSnapshot =
            adaptiveEvidenceWeightEngine.update(
                result = reprojectionEvaluation,
                calibrationConfidence =
                    (
                        kotlin.math.exp(
                            -(
                                calibrationProvider()
                                    ?.reprojectionErrorPx
                                    ?: 0.0
                                ) / 4.0,
                        )
                    ).coerceIn(0.0, 1.0),
                synchronizationConfidence =
                    synchronizationConfidence.coerceIn(0.0, 1.0),
            )

        val reprojectionSkeleton = reprojectionEvaluation.skeleton

        val observationsByJoint =
            reprojectionFrames
                .flatMap { frame ->
                    frame.observations.map { observation ->
                        observation.landmarkId to observation.camera
                    }
                }
                .groupBy({ it.first }, { it.second })

        val geometricObservabilityEvaluation =
            geometricObservability.evaluate(
                MultiViewGeometricObservabilityEngine.Input(
                    timestampNanos = timestampNanos,
                    skeleton = reprojectionSkeleton,
                    observationsByJoint = observationsByJoint,
                    synchronizedCameraCount = cameraCount,
                ),
            )

        val rigGeometryDriftEvaluation = rigGeometryDrift.update(
            LiveRigGeometryDriftEngine.Input(
                timestampNanos = timestampNanos,
                calibrationReprojectionErrorPx =
                    calibrationProvider()?.reprojectionErrorPx ?: 0.0,
                liveMedianResidualPx =
                    reprojectionEvaluation.medianResidualPx,
                liveP95ResidualPx =
                    reprojectionEvaluation.p95ResidualPx,
                liveMaximumResidualPx =
                    reprojectionEvaluation.maximumResidualPx,
                liveInlierRatio =
                    reprojectionEvaluation.inlierRatio,
                liveMultiViewScore =
                    reprojectionEvaluation.score,
                cameraCount = cameraCount,
                clockSkewMillis =
                    kotlin.math.abs(clockSkewNanos) / 1e6,
                cameraMotionRisk =
                    cameraMotionRisk.coerceIn(0.0, 1.0),
                synchronizationConfidence =
                    synchronizationConfidence.coerceIn(0.0, 1.0),
                captureTimingConfidence =
                    captureTimingConfidence.coerceIn(0.0, 1.0),
            ),
        )

        val uncertaintyEvaluation =
            uncertaintyPropagation.evaluate(
                ReconstructionUncertaintyPropagationEngine.Input(
                    timestampNanos = timestampNanos,
                    skeleton = reprojectionSkeleton,
                    reprojectionQuality =
                        reprojectionEvaluation.score.coerceIn(0.0, 1.0),
                    temporalQuality =
                        temporalEvaluation.score.coerceIn(0.0, 1.0),
                    geometricObservability =
                        geometricObservabilityEvaluation.score.coerceIn(0.0, 1.0),
                    rigGeometryDriftScore =
                        rigGeometryDriftEvaluation.driftScore.coerceIn(0.0, 1.0),
                    synchronizationConfidence =
                        synchronizationConfidence.coerceIn(0.0, 1.0),
                    captureTimingConfidence =
                        captureTimingConfidence.coerceIn(0.0, 1.0),
                    cameraMotionRisk =
                        cameraMotionRisk.coerceIn(0.0, 1.0),
                    occlusionRisk =
                        (1.0 - geometricObservabilityEvaluation.score)
                            .coerceIn(0.0, 1.0),
                ),
            )

        val kinematicEvaluation =
            kinematicConstraintEngine.solve(
                HighEndKinematicConstraintEngine.Input(
                    skeleton = reprojectionSkeleton,
                    preserveMicroMotion = true,
                    confidenceHint = reprojectionEvaluation.score
                        .coerceIn(0.0, 1.0),
                    cameraMotionRisk = cameraMotionRisk
                        .coerceIn(0.0, 1.0),
                    occlusionRisk = (
                        1.0 -
                            geometricObservabilityEvaluation.score
                    ).coerceIn(0.0, 1.0),
                    synchronizationConfidence =
                        synchronizationConfidence.coerceIn(0.0, 1.0),
                ),
            )

        val constrainedSkeleton = kinematicEvaluation.skeleton

        val physicsEvaluation =
            physicsConstraintEngine.refine(
                constrainedSkeleton,
            )

        val physicsSkeleton = physicsEvaluation.frame

        val worldSpaceEvaluation =
            worldSpaceMotionConsistencyEngine.optimize(
                WorldSpaceMotionConsistencyEngine.Input(
                    frame = physicsSkeleton,
                    topology = null,
                    captureTimingConfidence =
                        captureTimingConfidence.coerceIn(0.0, 1.0),
                    lightingConfidence =
                        lightingConfidence.coerceIn(0.0, 1.0),
                    cameraMotionRisk =
                        cameraMotionRisk.coerceIn(0.0, 1.0),
                    preserveMicroMotion = true,
                    temporalEvidenceScale =
                        (
                            0.72 +
                                synchronizationConfidence.coerceIn(0.0, 1.0) *
                                    0.28
                            ).coerceIn(0.25, 1.50),
                    topologyConstraintScale = 1.0,
                ),
            )

        val worldSpaceSkeleton =
            worldSpaceEvaluation.skeleton

        val gaussian4DEvidence =
            gaussian4DEvidenceProvider
                ?.invoke()

        val global4DEvaluation =
            global4DConsistencyEngine.evaluate(
                Global4DConsistencyEngine.Input(
                    skeleton = worldSpaceSkeleton,
                    physicsScore =
                        physicsEvaluation.physicsScore
                            .coerceIn(0.0, 1.0),
                    physicsAccepted =
                        physicsEvaluation.accepted,
                    reprojectionScore =
                        reprojectionEvaluation.score
                            .coerceIn(0.0, 1.0),
                    temporalScore =
                        temporalEvaluation.score
                            .coerceIn(0.0, 1.0),
                    synchronizationConfidence =
                        synchronizationConfidence
                            .coerceIn(0.0, 1.0),
                    rigTrust =
                        adaptiveEvidenceSnapshot.globalRigTrust
                            .coerceIn(0.0, 1.0),
                    cameraMotionRisk =
                        cameraMotionRisk
                            .coerceIn(0.0, 1.0),
                    directionalUncertaintyConfidence =
                        directionalUncertainty.confidence
                            .coerceIn(0.0, 1.0),
                    directionalUncertaintyAnisotropy =
                        directionalUncertainty.maxAnisotropyRatio
                            .coerceAtLeast(1.0),
                    directionalUncertaintyDepthMeters =
                        directionalUncertainty.depthMeters
                            .coerceAtLeast(0.0),
                    faceEyeConfidence =
                        faceEyeFusionConfidence
                            .coerceIn(0.0, 1.0),
                    faceEyeAccepted =
                        faceEyeFusionAccepted,
                    faceEyeOutOfDistributionRisk =
                        faceEyeOutOfDistributionRisk
                            .coerceIn(0.0, 1.0),
                    bodyMetricEvidenceAuthority =
                        bodyMetricEvidenceAuthority.coerceIn(0.0, 1.0),
                    faceMetricEvidenceAuthority =
                        faceEyeMetricEvidenceAuthority.coerceIn(0.0, 1.0),
                    handMetricEvidenceAuthority =
                        handMetricEvidenceAuthority.coerceIn(0.0, 1.0),
                    bodyEpipolarConsistency =
                        bodySymmetricEpipolarConsistencyScore.coerceIn(0.0, 1.0),
                    faceEpipolarConsistency =
                        faceEpipolarConsistency.coerceIn(0.0, 1.0),
                    handEpipolarConsistency =
                        handEpipolarConsistency.coerceIn(0.0, 1.0),
                    externalContactConfidence =
                        denseContactEvidenceConfidence
                            .coerceIn(0.0, 1.0),
                    externalContactEvidenceAvailable =
                        denseContactEvidenceAvailable,
                    gaussianTemporalStability =
                        gaussian4DEvidence
                            ?.temporalStability
                            ?.coerceIn(0.0, 1.0)
                            ?: 1.0,
                    gaussianTrustedRatio =
                        gaussian4DEvidence
                            ?.trustedRatio
                            ?.coerceIn(0.0, 1.0)
                            ?: 1.0,
                    gaussianMeanNodeConfidence =
                        gaussian4DEvidence
                            ?.meanNodeConfidence
                            ?.coerceIn(0.0, 1.0)
                            ?: 1.0,
                    gaussianMeanBindingConfidence =
                        gaussian4DEvidence
                            ?.meanBindingConfidence
                            ?.coerceIn(0.0, 1.0)
                            ?: 1.0,
                    gaussianP95StrainRatio =
                        gaussian4DEvidence
                            ?.p95StrainRatio
                            ?.coerceAtLeast(0.0)
                            ?: 0.0,
                    gaussianMaximumStrainRatio =
                        gaussian4DEvidence
                            ?.maximumStrainRatio
                            ?.coerceAtLeast(0.0)
                            ?: 0.0,
                    gaussianEvidenceAvailable =
                        gaussian4DEvidence != null,
                ),
            )

        val evaluation = quality.evaluate(
            frame = worldSpaceSkeleton,
            camerasUsed = cameraCount,
            clockSkewNanos = clockSkewNanos,
        )

        val reasons = (
            evaluation.reasons +
                interleavedTemporalResult.frame.warnings
                    .map { "interleaved-temporal-$it" } +
                temporalEvaluation.reasons
                    .map { "temporal-$it" } +
                reprojectionEvaluation.warnings
                    .map { "reprojection-$it" } +
                rigGeometryDriftEvaluation.warnings
                    .map { "geometry-drift-$it" } +
                geometricObservabilityEvaluation.warnings
                    .map { "observability-$it" } +
                uncertaintyEvaluation.warnings
                    .map { "uncertainty-$it" } +
                adaptiveEvidenceSnapshot.warnings
                    .map { "view-evidence-$it" } +
                spatiotemporalResult.warnings
                    .map { "spatiotemporal-$it" } +
                kinematicEvaluation.diagnostics.warnings
                    .map { "kinematic-$it" } +
                physicsEvaluation.warnings
                    .map { "physics-$it" } +
                worldSpaceEvaluation.warnings
                    .map { "world-space-$it" } +
                global4DEvaluation.warnings
                    .map { "global4d-$it" }
        ).distinct()

        val continuityDecision = continuity.evaluate(
            candidate = worldSpaceSkeleton,
            qualityAccepted =
                evaluation.accepted &&
                    temporalEvaluation.accepted &&
                    reprojectionEvaluation.accepted &&
                    rigGeometryDriftEvaluation.accepted &&
                    geometricObservabilityEvaluation.accepted &&
                    uncertaintyEvaluation.accepted &&
                    kinematicEvaluation.accepted &&
                    directionalUncertainty.confidence >= 0.18 &&
                    physicsEvaluation.accepted &&
                    global4DEvaluation.accepted,
            qualityScore = (
                0.45 * evaluation.score +
                    0.16 * temporalEvaluation.score +
                    0.14 * reprojectionEvaluation.score +
                    0.10 * geometricObservabilityEvaluation.score +
                    0.05 * kinematicEvaluation.score +
                    0.10 * worldSpaceEvaluation.score
                ).coerceIn(0.0, 1.0),
            timestampNanos = timestampNanos,
        )

        val continuityReasons = continuityDecision.reasons.map {
            "continuity-$it"
        }

        val denseSurface =
            latestDenseSurfaceResult
        val denseSurfaceQualityScale =
            denseSurface?.let {
                (
                    it.meanConfidence * 0.34 +
                        it.trustedPointRatio * 0.28 +
                        it.temporalStability * 0.18 +
                        it.topologyStability * 0.12 +
                        it.viewDiversity * 0.08
                    ).coerceIn(0.20, 1.0)
            } ?: 1.0

        val denseIdentityGateScale =
            latestDenseIdentityGateQuality.coerceIn(0.20, 1.0)

        val cameraDiversityScale =
            (
                0.82 +
                    0.18 *
                        latestCameraContributionPlan
                            .minimumPairwiseDiversity
                            .coerceIn(0.0, 1.0)
            ).coerceIn(0.82, 1.0)

        val supportEvidenceResult =
            distributedSupportEvidenceEngine.evaluate(
                DistributedSupportEvidenceEngine.Input(
                    skeleton = stabilizedSkeleton,
                    selectedCameraIds =
                        latestCameraContributionPlan.selectedDeviceIds
                            .map { it.value }
                            .toSet(),
                    cameraFrames = reprojectionFrames,
                    targetTimestampNanos = timestampNanos,
                    synchronizationConfidence =
                        synchronizationConfidence.coerceIn(0.0, 1.0),
                    cameraMotionRisk =
                        cameraMotionRisk.coerceIn(0.0, 1.0),
                    calibrationReprojectionErrorPx =
                        calibrationProvider()
                            ?.reprojectionErrorPx
                            ?.coerceAtLeast(0.0)
                            ?: 0.0,
                ),
            )

        val ultraPrecisionInput =
            buildUltraPrecisionInput(
                calibration = latestCalibrationSnapshot
                    ?: calibrationProvider()
                    ?: return null,
                cameraFrames = latestCameraFrames,
                cameraCount = cameraCount,
                worldSpaceSkeleton = worldSpaceSkeleton,
                reprojectionScore = reprojectionEvaluation.score,
                supportEvidenceScore =
                    supportEvidenceResult.score
                        .takeIf {
                            supportEvidenceResult.supportingCameraCount > 0
                        },
                geometricInformationP95MajorMeters =
                    geometricObservabilityEvaluation
                        .p95InformationSigmaMajor95Meters,
                spatiotemporalConfidence =
                    spatiotemporalResult.confidence,
                clockSkewNanos = clockSkewNanos,
                evaluationDroppedObservations =
                    evaluation.diagnostics.droppedObservations,
                bodyAcceptedRatio =
                    latestBodyEvidenceSnapshot.acceptedRatio,
                denseIdentityGateQuality =
                    denseIdentityGateScale,
                denseSurfaceTemporalStability =
                    denseSurface?.temporalStability ?: 1.0,
                denseSurfaceViewDiversity =
                    denseSurface?.viewDiversity ?: 1.0,
                denseSubMillimeterGeometryFeasibility =
                    denseSurface
                        ?.meanSubMillimeterFeasibilityScore
                        ?: 0.0,
                denseTailAdjustedFirstOrderDepthErrorMillimeters =
                    denseSurface
                        ?.p95ReprojectionTailAdjustedDepthErrorMillimeters
                        ?: Double.POSITIVE_INFINITY,
                denseTailAdjustedSubMillimeterFeasibilityScore =
                    denseSurface
                        ?.meanTailAdjustedSubMillimeterFeasibilityScore
                        ?: 0.0,
                denseFirstOrderDepthErrorMillimeters =
                    denseSurface
                        ?.meanFirstOrderDepthErrorMillimeters
                        ?: Double.POSITIVE_INFINITY,
                denseMotionStability =
                    denseSurface?.meanMotionStability ?: 0.0,
                denseKinematicInnovationMillimeters =
                    denseSurface
                        ?.meanKinematicInnovationMillimeters
                        ?: 0.0,
            )

        val ultraPrecisionResult =
            ultraPrecisionEvidenceEngine.evaluate(
                policy = ultraPrecisionPolicy,
                input = ultraPrecisionInput,
            )

        val frontierPrecisionResult =
            frontierPrecisionQualificationEngine.evaluate(
                input =
                    FrontierPrecisionQualificationEngine.Input(
                        evidenceSamples =
                            ultraPrecisionInput.empiricalSampleCount,
                        measuredP95Millimeters =
                            ultraPrecisionInput.empiricalP95Millimeters,
                        measuredP99Millimeters =
                            ultraPrecisionInput.empiricalP99Millimeters,
                        externalReferenceP95Millimeters =
                            ultraPrecisionInput.referenceP95Millimeters,
                        externalReferenceUncertaintyMillimeters =
                            ultraPrecisionInput.referenceUncertaintyMillimeters,
                        systemMeasurementUncertaintyMillimeters =
                            (
                                denseSurface
                                    ?.uncertaintyP95Meters
                                    ?.takeIf { it.isFinite() }
                                    ?.times(1000.0)
                            ),
                        matchedReference =
                            ultraPrecisionInput.matchedReference,
                        independentViews =
                            latestCameraFrames
                                .map { it.deviceId }
                                .distinct()
                                .size,
                        baselineMeters =
                            ultraPrecisionInput.cameraBaselineMeters,
                        parallaxDegrees =
                            ultraPrecisionInput.medianParallaxDegrees,
                        calibrationConfidence =
                            ultraPrecisionInput.calibrationConfidence,
                        synchronizationConfidence =
                            ultraPrecisionInput.synchronizationConfidence,
                        viewDiversity =
                            ultraPrecisionInput.viewDiversity,
                        crossViewAgreement =
                            ultraPrecisionInput.crossViewAgreement,
                        occlusionRatio =
                            ultraPrecisionInput.occlusionRatio,
                        identityContinuity =
                            ultraPrecisionInput.identityContinuity,
                        surfaceTemporalStability =
                            ultraPrecisionInput.surfaceTemporalStability,
                        thermalRisk =
                            ultraPrecisionInput.thermalRisk,
                        memoryPressure =
                            ultraPrecisionInput.memoryPressure,
                        timingUncertaintyMicros =
                            (
                                ultraPrecisionInput.synchronizationSkewMilliseconds *
                                    1000.0 +
                                    ultraPrecisionInput.rollingShutterEquivalentMilliseconds *
                                    500.0
                            ).coerceAtLeast(0.0),
                        reprojectionP95Pixels =
                            ultraPrecisionInput.reprojectionErrorPixels,
                        hardFailure =
                            denseSurface?.hardFailure == true,
                    ),
                enabled = ultraPrecisionPolicy.enabled,
            )

        val captureAuthority =
            HighEndCaptureAuthorityEngine().evaluate(
                HighEndCaptureAuthorityEngine.Input(
                    timestampNanos = timestampNanos,
                    connectedCameras = cameraCount,
                    authoritativeCameras =
                        latestCameraContributionPlan.selectedDeviceIds
                            .distinct()
                            .size
                            .coerceAtMost(cameraCount),
                    calibrationConfidence =
                        ultraPrecisionInput.calibrationConfidence,
                    synchronizationConfidence =
                        ultraPrecisionInput.synchronizationConfidence,
                    timingAuthorityConfidence =
                        captureTimingConfidence.coerceIn(0.0, 1.0),
                    temporalConsistency =
                        global4DEvaluation.score.coerceIn(0.0, 1.0),
                    viewDiversity = ultraPrecisionInput.viewDiversity,
                    crossViewAgreement = ultraPrecisionInput.crossViewAgreement,
                    epipolarConsistency =
                        bodySymmetricEpipolarConsistencyScore.coerceIn(0.0, 1.0),
                    bodyMetricEvidenceAuthority =
                        bodyMetricEvidenceAuthority.coerceIn(0.0, 1.0),
                    faceMetricEvidenceAuthority =
                        faceEyeMetricEvidenceAuthority.coerceIn(0.0, 1.0),
                    handMetricEvidenceAuthority =
                        handMetricEvidenceAuthority.coerceIn(0.0, 1.0),
                    faceEpipolarConsistency =
                        faceEpipolarConsistency.coerceIn(0.0, 1.0),
                    handEpipolarConsistency =
                        handEpipolarConsistency.coerceIn(0.0, 1.0),
                    reconstructionUncertaintyMillimeters =
                        geometricObservabilityEvaluation
                            .p95InformationSigmaMajor95Meters
                            .coerceAtLeast(0.0) * 1000.0,
                    motionP95ErrorMillimeters =
                        ultraPrecisionResult.errorBudget.modeledP95Millimeters,
                    motionP99ErrorMillimeters =
                        ultraPrecisionResult.errorBudget.modeledP99Millimeters,
                    temporalLagMilliseconds =
                        abs(clockSkewNanos).toDouble() / 1_000_000.0,
                    rollingShutterRisk =
                        latestCameraFrames
                            .map { it.rollingShutterSkewNanos }
                            .zip(
                                latestCameraFrames.map { it.frameDurationNanos },
                            )
                            .mapNotNull { (skew, duration) ->
                                if (duration <= 0L) null
                                else (
                                    skew.toDouble() /
                                        duration.toDouble()
                                ).coerceIn(0.0, 1.0)
                            }
                            .maxOrNull()
                            ?: 0.0,
                    motionBlurRisk =
                        latestCameraFrames
                            .map { it.estimatedMotionBlurRisk }
                            .maxOrNull()
                            ?.coerceIn(0.0, 1.0)
                            ?: 0.0,
                    cameraMotionRisk =
                        cameraMotionRisk.coerceIn(0.0, 1.0),
                    identityContinuity =
                        latestDenseIdentityGateQuality.coerceIn(0.0, 1.0),
                    occlusionRisk =
                        ultraPrecisionInput.occlusionRatio.coerceIn(0.0, 1.0),
                    contactConsistency =
                        denseContactEvidenceConfidence.coerceIn(0.0, 1.0),
                    observedGeometryRatio =
                        (
                            1.0 - ultraPrecisionInput.occlusionRatio
                        ).coerceIn(0.0, 1.0),
                    inferredGeometryRatio = 0.0,
                    metricScaleUncertaintyRatio =
                        ultraPrecisionInput.metricScaleUncertaintyRatio
                            .coerceIn(0.0, 1.0),
                    empiricalBenchmarkAccepted =
                        ultraPrecisionResult.empiricalEvidenceAvailable,
                    empiricalBenchmarkSampleCount =
                        ultraPrecisionInput.empiricalSampleCount,
                    empiricalP95Millimeters =
                        ultraPrecisionInput.empiricalP95Millimeters,
                    referenceUncertaintyMillimeters =
                        ultraPrecisionInput.referenceUncertaintyMillimeters,
                    referenceConditionsMatched =
                        ultraPrecisionInput.matchedReference,
                    externalReferenceDocumented =
                        ultraPrecisionInput.referenceKind !=
                            UltraPrecisionEvidenceEngine.ReferenceKind.NONE &&
                            ultraPrecisionInput.referenceUncertaintyMillimeters !=
                                null,
                    targetP95Millimeters =
                        frontierPrecisionResult.policy.targetP95Millimeters
                            .coerceAtLeast(0.10),
                    targetRequiresIndependentReference =
                        frontierPrecisionResult.policy.matchedReferenceRequired,
                ),
            )

        val correlatedEvidenceResult =
            correlatedEvidenceFusionEngine.evaluate(
                listOfNotNull(
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "reconstruction",
                        score = evaluation.score,
                        reliability = 0.94,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "reprojection",
                        score = reprojectionEvaluation.score,
                        reliability = 0.92,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY,
                    ),
                    if (
                        supportEvidenceResult.supportingCameraCount > 0
                    ) {
                        CorrelatedEvidenceFusionEngine.Signal(
                            name = "support-evidence",
                            score = supportEvidenceResult.score,
                            reliability =
                                (
                                    0.55 +
                                        (
                                            supportEvidenceResult.supportingCameraCount /
                                                8.0
                                        ).coerceIn(0.0, 1.0) * 0.38
                                ).coerceIn(0.55, 0.93),
                            group =
                                CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY,
                        )
                    } else {
                        null
                    },
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "observability",
                        score = geometricObservabilityEvaluation.score,
                        reliability = 0.90,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.GEOMETRY,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "temporal",
                        score = temporalEvaluation.score,
                        reliability = 0.92,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "spatiotemporal",
                        score = spatiotemporalResult.confidence,
                        reliability = 0.88,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "kinematic",
                        score = kinematicEvaluation.score,
                        reliability = 0.86,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.TEMPORAL,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "uncertainty",
                        score = uncertaintyEvaluation.score,
                        reliability =
                            uncertaintyEvaluation.confidence.coerceIn(0.18, 1.0),
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.UNCERTAINTY,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "rig-drift",
                        score = rigGeometryDriftEvaluation.driftScore,
                        reliability = 0.86,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.RIG,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "camera-diversity",
                        score =
                            latestCameraContributionPlan
                                .minimumPairwiseDiversity
                                .coerceIn(0.0, 1.0),
                        reliability = 0.82,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.RIG,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "physics",
                        score = physicsEvaluation.physicsScore,
                        reliability = 0.84,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.PHYSICS,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "world-space",
                        score = worldSpaceEvaluation.score,
                        reliability = 0.84,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.PHYSICS,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "global-4d",
                        score = global4DEvaluation.score,
                        reliability = 0.90,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.GLOBAL,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "dense-surface",
                        score = denseSurfaceQualityScale,
                        reliability = 0.88,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.DENSE_SURFACE,
                    ),
                    CorrelatedEvidenceFusionEngine.Signal(
                        name = "identity-gate",
                        score = denseIdentityGateScale,
                        reliability = 0.86,
                        group =
                            CorrelatedEvidenceFusionEngine.DependencyGroup.IDENTITY,
                    ),
                ),
            )

        val combinedScore = (
            0.22 * evaluation.score +
                0.14 * temporalEvaluation.score +
                0.12 * reprojectionEvaluation.score +
                0.04 * rigGeometryDriftEvaluation.driftScore +
                0.05 * continuityDecision.score +
                0.04 * uncertaintyEvaluation.score +
                0.04 * kinematicEvaluation.score +
                0.04 * (
                    adaptiveEvidenceSnapshot.cameraWeights.values
                        .takeIf { it.isNotEmpty() }
                        ?.average()
                        ?: 1.0
                    ) +
                0.03 * adaptiveEvidenceSnapshot.globalRigTrust +
                0.06 * spatiotemporalResult.confidence.coerceIn(0.0, 1.0) +
                0.06 * physicsEvaluation.physicsScore +
                0.08 * worldSpaceEvaluation.score +
                0.05 * global4DEvaluation.score
        ).coerceIn(0.0, 1.0)

        val correlationAwareCombinedScore =
            (
                0.58 * combinedScore +
                    0.42 * correlatedEvidenceResult.fusedScore
            ).coerceIn(0.0, 1.0)

        val correlationAwarePenalty =
            (
                0.90 +
                    0.10 *
                        correlatedEvidenceResult.evidenceCoverage
            ).coerceIn(0.90, 1.0)

        val supportAgreementScale =
            if (
                supportEvidenceResult.supportingCameraCount == 0
            ) {
                1.0
            } else {
                (
                    0.97 +
                        0.03 *
                            supportEvidenceResult.score
                ).coerceIn(0.90, 1.0)
            }

        val finalCombinedScore =
            correlationAwareCombinedScore *
                supportAgreementScale *
                correlationAwarePenalty *
                denseIdentityGateScale.coerceIn(0.20, 1.0) *
            cameraDiversityScale *
            (
                0.86 +
                    0.14 *
                        captureAuthority.score.coerceIn(0.0, 1.0)
            ) *
            (
                if (ultraPrecisionPolicy.enabled) {
                    0.95 +
                        ultraPrecisionResult.evidenceScore * 0.05
                } else {
                    1.0
                }
            ).coerceIn(0.95, 1.0)

        val correlatedReasons =
            correlatedEvidenceResult.warnings.map {
                "correlated-evidence-$it"
            }

        val directionalReasons =
            directionalUncertainty.warnings.map {
                "directional-$it"
            }

        val combinedReasons = mutableListOf<String>()
        combinedReasons += reasons
        combinedReasons += continuityReasons
        combinedReasons += correlatedReasons
        combinedReasons += directionalReasons
        if (ultraPrecisionResult.policy.enabled) {
            combinedReasons +=
                ultraPrecisionResult.blockers.map {
                    "ultra-precision-$it"
                }
            combinedReasons +=
                ultraPrecisionResult.warnings.map {
                    "ultra-precision-$it"
                }
        }
        combinedReasons +=
            supportEvidenceResult.warnings.map {
                "support-evidence-$it"
            }
        if (bodyMetricEvidenceAuthority < 0.48) {
            combinedReasons +=
                "body-metric-evidence-authority-low"
        }
        if (bodySymmetricEpipolarConsistencyScore < 0.45) {
            combinedReasons +=
                "body-epipolar-consistency-low"
        }
        combinedReasons +=
            captureAuthority.blockCodes.map {
                "capture-authority-" + it.name
            }
        combinedReasons +=
            captureAuthority.warnings.map {
                "capture-authority-" + it
            }
        val finalReasons = combinedReasons.distinct()

        val directionalQualityScale =
            (
                0.94 +
                    0.06 *
                        directionalUncertainty.confidence
                    .coerceIn(0.0, 1.0)
                ).coerceIn(0.94, 1.0)

        return DetailedResult(
            bodyEvidenceAcceptedRatio =
                latestBodyEvidenceSnapshot.acceptedRatio,
            bodyEvidenceRejectedObservationCount =
                latestBodyEvidenceSnapshot.rejectedObservationCount,
            bodyEvidenceMeanFusionWeight =
                latestBodyEvidenceSnapshot.meanFusionWeight,
            bodyEvidenceMeanUncertaintyMeters =
                latestBodyEvidenceSnapshot.meanUncertaintyMeters,
            bodyEvidenceTimingRiskObservationCount =
                latestBodyEvidenceSnapshot.timingRiskObservationCount,
            bodyEvidenceBoundaryRiskObservationCount =
                latestBodyEvidenceSnapshot.boundaryRiskObservationCount,
            denseSurfaceResult = denseSurface,
            denseSurfacePrecisionQualified =
                denseSurface?.precisionQualified == true,
            denseSurfaceTailAdjustedDepthErrorP95Millimeters =
                denseSurface
                    ?.p95ReprojectionTailAdjustedDepthErrorMillimeters
                    ?: Double.POSITIVE_INFINITY,
            denseSurfaceTailAdjustedFeasibilityScore =
                denseSurface
                    ?.meanTailAdjustedSubMillimeterFeasibilityScore
                    ?: 0.0,
            denseIdentityGateQuality =
                latestDenseIdentityGateQuality,
            denseIdentityGateRejectedCount =
                latestDenseIdentityGateRejectedCount,
            denseIdentityGateEvaluatedPoints =
                latestDenseIdentityGateEvaluatedPoints,
            denseIdentityGateWarnings =
                latestDenseIdentityGateWarnings,
            denseGeometricConditioningScore =
                latestDenseGeometricConditioningScore,
            denseGeometricConditioningMedianParallaxDegrees =
                latestDenseGeometricConditioningMedianParallaxDegrees,
            denseGeometricConditioningMinimumObservationScore =
                latestDenseGeometricConditioningMinimumObservationScore,
            denseGeometricConditioningWarnings =
                latestDenseGeometricConditioningWarnings,
            identityExchangeState =
                latestIdentityExchangeResult.state,
            identityExchangeRisk =
                latestIdentityExchangeResult.exchangeRisk,
            identityExchangeConfidence =
                latestIdentityExchangeResult.productionIdentityConfidence,
            identityExchangeEvaluatedSubjects =
                latestIdentityExchangeResult.evaluatedSubjects,
            identityExchangeHeldSubjects =
                latestIdentityExchangeResult.heldSubjects,
            identityExchangeWarnings =
                latestIdentityExchangeResult.warnings,
            bodyMetricEvidenceAuthority =
                bodyMetricEvidenceAuthority,
            bodySymmetricEpipolarConsistencyScore =
                bodySymmetricEpipolarConsistencyScore,
            bodyEpipolarWarnings =
                if (bodySymmetricEpipolarConsistencyScore < 0.45) {
                    listOf("BODY_EPIPOLAR_EVIDENCE_DEGRADED")
                } else {
                    emptyList()
                },
            faceMetricEvidenceAuthority =
                faceEyeMetricEvidenceAuthority.coerceIn(0.0, 1.0),
            handMetricEvidenceAuthority =
                handMetricEvidenceAuthority.coerceIn(0.0, 1.0),
            faceEpipolarConsistency =
                faceEpipolarConsistency.coerceIn(0.0, 1.0),
            handEpipolarConsistency =
                handEpipolarConsistency.coerceIn(0.0, 1.0),
            crossModalMetricAuthority =
                global4DEvaluation.crossModalMetricAuthority,
            crossModalEpipolarAuthority =
                global4DEvaluation.crossModalEpipolarAuthority,
            captureAuthorityScore = captureAuthority.score,
            captureAuthorityBand = captureAuthority.band,
            captureAuthorityHardBlocked = captureAuthority.hardBlocked,
            captureAuthorityWarnings = captureAuthority.warnings,
            ultraPrecision = ultraPrecisionResult,
            frontierPrecision = frontierPrecisionResult,
            skeleton = continuityDecision.displayFrame,
            diagnostics = evaluation.diagnostics,
            accepted =
                evaluation.accepted &&
                    temporalEvaluation.accepted &&
                    reprojectionEvaluation.accepted &&
                    rigGeometryDriftEvaluation.accepted &&
                    geometricObservabilityEvaluation.accepted &&
                    uncertaintyEvaluation.accepted &&
                    kinematicEvaluation.accepted &&
                    continuityDecision.accepted &&
                    physicsEvaluation.accepted &&
                    worldSpaceEvaluation.accepted &&
                    global4DEvaluation.accepted &&
                    correlatedEvidenceResult.state !=
                        CorrelatedEvidenceFusionEngine.State.REJECTED &&
                    continuityReasons.isEmpty() &&
                    (
                        denseSurface == null ||
                            !denseSurface.hardFailure
                        ) &&
                    latestDenseIdentityGateQuality >= 0.35 &&
                    latestCameraContributionPlan.minimumPairwiseDiversity >=
                        0.10 &&
                    bodyMetricEvidenceAuthority >= 0.16 &&
                    bodySymmetricEpipolarConsistencyScore >= 0.18 &&
                    faceEyeMetricEvidenceAuthority >= 0.16 &&
                    handMetricEvidenceAuthority >= 0.16 &&
                    faceEpipolarConsistency >= 0.18 &&
                    handEpipolarConsistency >= 0.18 &&
                    !captureAuthority.hardBlocked &&
                    (
                        supportEvidenceResult.supportingCameraCount == 0 ||
                            supportEvidenceResult.contradictionRisk <
                                0.72
                    ) &&
                    (
                        !ultraPrecisionPolicy.qualificationMode ||
                            ultraPrecisionResult.qualificationEligible
                        ) &&
                    (
                        !ultraPrecisionPolicy.qualificationMode ||
                            frontierPrecisionResult?.physicallyQualified == true
                        ),
            qualityScore =
                (
                    finalCombinedScore *
                        directionalQualityScale *
                        denseSurfaceQualityScale
                    ).coerceIn(0.0, 1.0),
            rejectionReasons = finalReasons,
            temporalQualityScore = temporalEvaluation.score,
            temporalRejectionReasons = temporalEvaluation.reasons,
            continuityScore = continuityDecision.score,
            continuityAction = continuityDecision.action,
            continuityReasons = continuityDecision.reasons,
            reprojectionQualityScore = reprojectionEvaluation.score,
            reprojectionRejectionReasons = reprojectionEvaluation.warnings,
            reprojectionBottleneck = reprojectionEvaluation.bottleneck,
            rigGeometryDriftScore = rigGeometryDriftEvaluation.driftScore,
            rigGeometryDriftState = rigGeometryDriftEvaluation.state,
            rigGeometryDriftWarnings = rigGeometryDriftEvaluation.warnings,
            geometricObservabilityScore =
                geometricObservabilityEvaluation.score,
            geometricObservabilityWarnings =
                geometricObservabilityEvaluation.warnings,
            geometricInformationConditionScore =
                geometricObservabilityEvaluation.medianInformationConditionScore,
            geometricInformationP95MajorMeters =
                geometricObservabilityEvaluation.p95InformationSigmaMajor95Meters,
            geometricInformationP95DepthMeters =
                geometricObservabilityEvaluation.p95InformationSigmaDepth95Meters,
            geometricObservabilityBottleneck =
                geometricObservabilityEvaluation.bottleneck,
            uncertaintyScore =
                uncertaintyEvaluation.score,
            uncertaintyConfidence =
                uncertaintyEvaluation.confidence,
            meanUncertaintyMeters =
                uncertaintyEvaluation.meanUncertaintyMeters,
            p95UncertaintyMeters =
                uncertaintyEvaluation.p95UncertaintyMeters,
            trustedJointRatio =
                uncertaintyEvaluation.trustedJointRatio,
            uncertaintyBottleneck =
                uncertaintyEvaluation.dominantBottleneck,
            uncertaintyWarnings =
                uncertaintyEvaluation.warnings,
            directionalUncertaintyMajor95Meters =
                directionalUncertainty.major95Meters,
            directionalUncertaintyDepthMeters =
                directionalUncertainty.depthMeters,
            directionalUncertaintyAnisotropy =
                directionalUncertainty.maxAnisotropyRatio,
            directionalUncertaintyConfidence =
                directionalUncertainty.confidence,
            directionalUncertaintyDegradedJointCount =
                directionalUncertainty.degradedJointCount,
            directionalUncertaintyWarnings =
                directionalUncertainty.warnings,
            adaptiveViewEvidenceScore =
                adaptiveEvidenceSnapshot.cameraWeights.values
                    .takeIf { it.isNotEmpty() }
                    ?.average()
                    ?: 1.0,
            adaptiveViewEvidenceGlobalRigTrust =
                adaptiveEvidenceSnapshot.globalRigTrust,
            adaptiveViewEvidenceWarnings =
                adaptiveEvidenceSnapshot.warnings,
            adaptiveViewEvidenceDegradedCameraCount =
                adaptiveEvidenceSnapshot.degradedCameraCount,
            adaptiveViewEvidenceHoldCameraCount =
                adaptiveEvidenceSnapshot.holdCameraCount,
            adaptiveViewEvidenceDominantCamera =
                adaptiveEvidenceSnapshot.dominantCamera?.value
                    ?: "NONE",
            adaptiveViewEvidenceCameraStatuses =
                adaptiveEvidenceSnapshot.cameraStatuses,
            cameraContributionSelectedCount =
                latestCameraContributionPlan.selectedDeviceIds.size,
            cameraContributionExcludedCount =
                latestCameraContributionPlan.excludedDeviceIds.size,
            cameraContributionDiversitySatisfied =
                latestCameraContributionPlan.diversitySatisfied,
            cameraContributionInformationGainScore =
                latestCameraContributionPlan.globalInformationGainScore,
            cameraContributionMinimumPairwiseDiversity =
                latestCameraContributionPlan.minimumPairwiseDiversity,
            cameraContributionViewStarvationRisk =
                latestCameraContributionPlan.viewStarvationRisk,
            cameraContributionReason =
                latestCameraContributionPlan.reason,
            supportEvidenceScore =
                supportEvidenceResult.score,
            supportEvidenceSupportingCameraCount =
                supportEvidenceResult.supportingCameraCount,
            supportEvidenceRejectedCameraCount =
                supportEvidenceResult.rejectedSupportingCameraCount,
            supportEvidenceContradictionRisk =
                supportEvidenceResult.contradictionRisk,
            supportEvidenceSupportingJointCount =
                supportEvidenceResult.supportingJointCount,
            supportEvidenceContradictedJointCount =
                supportEvidenceResult.contradictedJointCount,
            supportEvidenceWarnings =
                supportEvidenceResult.warnings,
            jointViewSelectionScore =
                worldSpaceSkeleton.joints
                    .map { joint ->
                        if (joint.observationCount >= 2) 1.0 else 0.0
                    }
                    .average()
                    .takeIf { it.isFinite() }
                    ?: 0.0,
            jointViewDiversitySatisfiedRatio =
                if (worldSpaceSkeleton.joints.isEmpty()) {
                    0.0
                } else {
                    worldSpaceSkeleton.joints
                        .count { it.observationCount >= 2 }
                        .toDouble() /
                        worldSpaceSkeleton.joints.size.toDouble()
                },
            jointSelectedObservationCount =
                worldSpaceSkeleton.joints.sumOf {
                    it.observationCount.coerceAtLeast(0)
                },
            jointExcludedObservationCount = 0,
            jointViewSelectionReasons =
                if (worldSpaceSkeleton.joints.any { it.observationCount < 2 }) {
                    listOf("INSUFFICIENT_MULTI_VIEW_OBSERVATIONS")
                } else {
                    emptyList()
                },
            spatiotemporalConfidence =
                spatiotemporalResult.confidence,
            spatiotemporalInitialRmsPx =
                spatiotemporalResult.initialRmsPx
                    .takeIf(Double::isFinite)
                    ?: 0.0,
            spatiotemporalFinalRmsPx =
                spatiotemporalResult.finalRmsPx
                    .takeIf(Double::isFinite)
                    ?: 0.0,
            spatiotemporalOffsetsNanos =
                spatiotemporalResult.cameraOffsetsNanos,
            spatiotemporalWarnings =
                spatiotemporalResult.warnings,
            interleavedTemporalConfidence =
                interleavedTemporalResult.frame.temporalConsistency,
            interleavedSpatialAgreement =
                interleavedTemporalResult.frame.spatialAgreement,
            interleavedEffectiveInputRateHz =
                interleavedTemporalResult.frame.effectiveInputRateHz,
            interleavedSupportingCameraCount =
                interleavedTemporalResult.frame.cameraCount,
            interleavedAuthoritative =
                interleavedTemporalResult.frame.authoritative,
            interleavedWarnings =
                interleavedTemporalResult.frame.warnings,
            interleavedSchedulerCoverage =
                latestInterleavedTemporalSamplingSummary.coverageRatio,
            interleavedSchedulerTemporalHealth =
                latestInterleavedTemporalSamplingSummary.temporalHealth,
            interleavedSchedulerEffectiveSamplingMultiplier =
                latestInterleavedTemporalSamplingSummary.effectiveSamplingMultiplier,
            interleavedSchedulerPublishable =
                latestInterleavedTemporalSamplingSummary.latestPlan?.publishable == true,
            interleavedSchedulerWarnings =
                latestInterleavedTemporalSamplingSummary.warnings,
            kinematicScore =
                kinematicEvaluation.score,
            kinematicAccepted =
                kinematicEvaluation.accepted,
            kinematicBottleneck =
                kinematicEvaluation.diagnostics.bottleneck,
            kinematicWarnings =
                kinematicEvaluation.diagnostics.warnings,
            physicsScore =
                physicsEvaluation.physicsScore,
            physicsAccepted =
                physicsEvaluation.accepted,
            physicsMeanBoneResidualMeters =
                physicsEvaluation.meanBoneResidualMeters,
            physicsP95BoneResidualMeters =
                physicsEvaluation.p95BoneResidualMeters,
            physicsMaxBoneResidualMeters =
                physicsEvaluation.maxBoneResidualMeters,
            physicsSymmetryScore =
                physicsEvaluation.symmetryScore,
            physicsFloorContactScore =
                physicsEvaluation.floorContactScore,
            physicsCorrectedJointCount =
                physicsEvaluation.correctedJointCount,
            physicsConstrainedEdgeCount =
                physicsEvaluation.constrainedEdgeCount,
            physicsWarnings =
                physicsEvaluation.warnings,
            worldSpaceConsistencyScore =
                worldSpaceEvaluation.score,
            worldSpaceAccepted =
                worldSpaceEvaluation.accepted,
            worldSpaceTemporalStability =
                worldSpaceEvaluation.temporalStability,
            worldSpaceMotionContinuity =
                worldSpaceEvaluation.motionContinuity,
            worldSpaceRejectedJointCount =
                worldSpaceEvaluation.rejectedJointCount,
            worldSpaceRobustOutlierRatio =
                worldSpaceEvaluation.robustOutlierRatio,
            worldSpaceWarnings =
                worldSpaceEvaluation.warnings,
            global4DScore =
                global4DEvaluation.score,
            global4DAccepted =
                global4DEvaluation.accepted,
            global4DState =
                global4DEvaluation.state,
            global4DPersistentFailure =
                global4DEvaluation.persistentFailure,
            global4DRootMotionConsistency =
                global4DEvaluation.rootMotionConsistency,
            global4DRootVelocityAgreement =
                global4DEvaluation.rootVelocityAgreement,
            global4DSceneConsistency =
                global4DEvaluation.sceneConsistency,
            global4DDirectionalUncertaintyConsistency =
                global4DEvaluation.directionalUncertaintyConsistency,
            global4DFaceEyeConsistency =
                global4DEvaluation.faceEyeConsistency,
            global4DExternalContactConsistency =
                global4DEvaluation.externalContactConsistency,
            global4DGaussianConsistency =
                global4DEvaluation.gaussian4DConsistency,
            global4DGaussianStrainConsistency =
                global4DEvaluation.gaussianStrainConsistency,
            global4DGaussianEvidenceAvailable =
                global4DEvaluation.gaussianEvidenceAvailable,
            global4DWarnings =
                global4DEvaluation.warnings,
        )
    }

    fun configureTargetViewBias(
        direction:
            com.nivukx.nox.core.model.Vec3?,
        weight: Double,
    ) {
        cameraContributionPlanner.configureTargetViewBias(
            direction = direction,
            weight = weight,
        )
    }

    fun configureSubframeAlignmentStrictness(
        strictness: Double,
    ) {
        subframeAlignmentStrictness =
            strictness.coerceIn(0.0, 1.0)
    }

    @Synchronized
    fun reconstructDenseSurface(
        input:
            DenseMultiViewSurfaceReconstructionEngine.FrameInput,
    ):
        DenseMultiViewSurfaceReconstructionEngine.Result =
        denseSurfaceReconstructionEngine.process(input).also {
            latestDenseSurfaceResult = it
        }

    fun denseSurfaceTelemetry():
        DenseMultiViewSurfaceReconstructionEngine.Telemetry =
        denseSurfaceReconstructionEngine.telemetry()

    fun latestDenseSurfaceResult():
        DenseMultiViewSurfaceReconstructionEngine.Result? =
        latestDenseSurfaceResult

    @Synchronized
    fun pushDenseLandmarks(
        deviceId: DeviceId,
        timestampNanos: Long,
        landmarks: List<DenseLandmarkInput>,
    ) {
        pushDenseLandmarkObservations(
            deviceId = deviceId,
            timestampNanos = timestampNanos,
            landmarks = landmarks.map {
                DenseLandmarkObservation(
                    subjectId = it.subjectId,
                    landmarkId = it.landmarkId,
                    normalizedX = it.normalizedX,
                    normalizedY = it.normalizedY,
                    confidence = it.confidence,
                    uncertainty = it.uncertainty,
                    visibility = it.visibility,
                    contactProbability = it.contactProbability,
                    temporalStability = it.temporalStability,
                    crossViewAgreement = it.crossViewAgreement,
                    reprojectionResidualNormalized =
                        it.reprojectionResidualNormalized,
                    predicted = it.predicted,
                    frameHeightPx = it.frameHeightPx,
                    rollingShutterSkewNanos =
                        it.rollingShutterSkewNanos,
                    frameDurationNanos =
                        it.frameDurationNanos,
                    exposureTimeNanos =
                        it.exposureTimeNanos,
                    captureTimingConfidence =
                        it.captureTimingConfidence,
                    probabilitySigmaXNormalized =
                        it.probabilitySigmaXNormalized,
                    probabilitySigmaYNormalized =
                        it.probabilitySigmaYNormalized,
                    probabilityCalibrationScore =
                        it.probabilityCalibrationScore,
                    silhouetteConsistency =
                        it.silhouetteConsistency,
                )
            },
        )
    }

    @Synchronized
    private fun pushDenseLandmarkObservations(
        deviceId: DeviceId,
        timestampNanos: Long,
        landmarks: List<DenseLandmarkObservation>,
    ) {
        if (timestampNanos <= 0L) return

        val history =
            denseLandmarkHistory.getOrPut(deviceId) {
                ArrayDeque()
            }

        if (
            history.isNotEmpty() &&
            timestampNanos < history.last().timestampNanos
        ) {
            return
        }

        history.addLast(
            DenseLandmarkDeviceFrame(
                deviceId = deviceId,
                timestampNanos = timestampNanos,
                landmarks = landmarks.take(1024),
            ),
        )

        while (history.size > 8) {
            history.removeFirst()
        }

        pruneDenseLandmarkHistory(timestampNanos)
    }

    @Synchronized
    fun reconstructBufferedDenseSurface(
        timestampNanos: Long,
        maximumSkewNanos: Long = 25_000_000L,
    ):
        DenseMultiViewSurfaceReconstructionEngine.Result? {
        val snapshot = calibrationProvider() ?: return null
        if (snapshot.intrinsics.isEmpty()) return null

        val frames = denseLandmarkHistory.mapNotNull { (deviceId, history) ->
            history.minByOrNull {
                kotlin.math.abs(
                    it.timestampNanos - timestampNanos,
                )
            }?.takeIf {
                kotlin.math.abs(
                    it.timestampNanos - timestampNanos,
                ) <= maximumSkewNanos
            }
        }

        if (frames.size < 2) return null

        val grouped =
            LinkedHashMap<
                DenseMultiViewSurfaceReconstructionEngine.PointKey,
                MutableList<
                    DenseMultiViewSurfaceReconstructionEngine.PointObservation,
                >,
            >()

        frames.forEach { frame ->
            val intrinsics =
                snapshot.intrinsics[frame.deviceId]
                    ?: return@forEach
            val extrinsics =
                snapshot.extrinsics[frame.deviceId]
                    ?: return@forEach

            val width =
                intrinsics.imageWidth.takeIf { it > 0 } ?: return@forEach
            val height =
                intrinsics.imageHeight.takeIf { it > 0 } ?: return@forEach

            frame.landmarks.forEach { landmark ->
                val confidence =
                    landmark.confidence
                        .coerceIn(0.0, 1.0)
                val visibility =
                    landmark.visibility
                        .coerceIn(0.0, 1.0)
                val uncertainty =
                    landmark.uncertainty
                        .coerceIn(0.0, 1.0)

                val temporalStability =
                    landmark.temporalStability
                        .coerceIn(0.0, 1.0)

                // Older v1-v3 peer frames omit image dimensions; calibrated intrinsic
                // height is a safer fallback than treating the source as a one-pixel image.
                val sourceFrameHeightPx =
                    landmark.frameHeightPx.takeIf { it > 1 } ?: height
                val rollingShutterEvidence =
                    rollingShutterEvidenceEngine.evaluate(
                        RollingShutterEvidenceEngine.Input(
                            yPx =
                                landmark.normalizedY *
                                    sourceFrameHeightPx.toDouble(),
                            frameHeightPx =
                                sourceFrameHeightPx,
                            rollingShutterSkewNanos =
                                landmark.rollingShutterSkewNanos
                                    .coerceAtLeast(0L),
                            frameDurationNanos =
                                landmark.frameDurationNanos
                                    .coerceAtLeast(0L),
                            exposureTimeNanos =
                                landmark.exposureTimeNanos
                                    .coerceAtLeast(0L),
                            captureTimingConfidence =
                                landmark.captureTimingConfidence
                                    .coerceIn(0.0, 1.0),
                        ),
                    )

                val locallyTrustedWeight =
                    DenseEvidencePolicy.observationWeight(
                        confidence = confidence,
                        visibility = visibility,
                        uncertainty = uncertainty,
                        temporalStability = temporalStability,
                        crossViewAgreement = landmark.crossViewAgreement,
                    )

                val weight =
                    (
                        locallyTrustedWeight *
                            rollingShutterEvidence.confidenceScale
                        ).coerceIn(0.001, 1.0)

                val temporalAlignment =
                    (
                        DenseEvidencePolicy.temporalAlignmentConfidence(
                            temporalStability = temporalStability,
                            predicted = landmark.predicted,
                        ) *
                            rollingShutterEvidence.confidenceScale
                        ).coerceIn(0.05, 1.0)

                val point =
                    DenseMultiViewSurfaceReconstructionEngine.PointObservation(
                        key =
                            DenseMultiViewSurfaceReconstructionEngine.PointKey(
                                subjectId =
                                    landmark.subjectId.coerceIn(0, 31),
                                pointId =
                                    landmark.landmarkId.coerceIn(0, 4095),
                            ),
                        observation =
                            TriangulationEngine.Observation(
                                cameraPose = extrinsics.pose,
                                intrinsics = intrinsics,
                                cameraId = frame.deviceId.value,
                                u =
                                    landmark.normalizedX *
                                        width.toDouble(),
                                v =
                                    landmark.normalizedY *
                                        height.toDouble(),
                                weight = weight,
                                temporalAlignmentConfidence =
                                    (
                                        temporalAlignment *
                                            rollingShutterEvidence.confidenceScale
                                        ).coerceIn(0.05, 1.0),
                                temporalUncertaintyNanos =
                                    (
                                        40_000_000.0 *
                                            (1.0 - temporalAlignment)
                                        ).toLong()
                                            .coerceAtLeast(0L),
                                observationUncertainty =
                                    (
                                        uncertainty +
                                            if (landmark.predicted) {
                                                0.16
                                            } else {
                                                0.0
                                            }
                                        ).coerceIn(0.0, 1.0),
                                visibility = visibility,
                                sourceQuality = confidence,
                                crossViewAgreement =
                                    DenseEvidencePolicy.crossViewAgreement(
                                        rawAgreement =
                                            landmark.crossViewAgreement,
                                        acceptedViewCount = 1,
                                    ),
                                temporalStability =
                                    temporalStability,
                                contactProbability =
                                    landmark.contactProbability
                                        .coerceIn(0.0, 1.0),
                                probabilityModes =
                                    buildProbabilityModes(
                                        landmark = landmark,
                                        width = width,
                                        height = height,
                                    ),
                                distributionCalibrationScore =
                                    landmark.probabilityCalibrationScore
                                        .coerceIn(0.0, 1.0),
                            ),
                            semanticConfidence = confidence,
                            visibility = visibility,
                            contactProbability =
                                landmark.contactProbability
                                    .coerceIn(0.0, 1.0),
                            sourceDiversity =
                                if (landmark.predicted) 0.62 else 0.88,
                            silhouetteConsistency =
                                landmark.silhouetteConsistency
                                    .coerceIn(0.0, 1.0),
                    )

                grouped.getOrPut(point.key) {
                    mutableListOf()
                }.add(point)
            }
        }

        var gatedPointCount = 0
        var rejectedPointObservationCount = 0
        var gateScoreSum = 0.0
        var geometricConditioningSum = 0.0
        var geometricConditioningCount = 0
        var geometricParallaxSum = 0.0
        var geometricParallaxCount = 0
        var geometricMinimumObservationScore = 1.0
        val gateWarnings = LinkedHashSet<String>()
        val geometricConditioningWarnings =
            LinkedHashSet<String>()
        val gatedObservations =
            ArrayList<
                DenseMultiViewSurfaceReconstructionEngine.PointObservation,
            >()

        val conditionedGrouped =
            LinkedHashMap<
                DenseMultiViewSurfaceReconstructionEngine.PointKey,
                MutableList<
                    DenseMultiViewSurfaceReconstructionEngine.PointObservation,
                >,
            >()

        grouped.forEach { (pointKey, candidates) ->
            val conditioning =
                perObservationGeometricConditioningEngine.evaluate(
                    candidates.map { it.observation },
                )
            geometricConditioningSum += conditioning.globalConditioningScore
            geometricConditioningCount++
            if (conditioning.medianParallaxDegrees.isFinite()) {
                geometricParallaxSum += conditioning.medianParallaxDegrees
                geometricParallaxCount++
            }
            geometricMinimumObservationScore =
                minOf(
                    geometricMinimumObservationScore,
                    conditioning.minimumObservationConditioningScore,
                )
            geometricConditioningWarnings += conditioning.warnings

            conditionedGrouped[pointKey] =
                candidates.map { candidate ->
                    val score =
                        conditioning.scoreFor(
                            candidate.observation.cameraId,
                        )
                    if (score == null) {
                        candidate
                    } else {
                        candidate.copy(
                            observation =
                                candidate.observation.copy(
                                    weight =
                                        (
                                            candidate.observation.weight *
                                                score.weightMultiplier
                                            ).coerceIn(0.0005, 1.0),
                                    temporalAlignmentConfidence =
                                        (
                                            candidate.observation
                                                .temporalAlignmentConfidence *
                                                (
                                                    0.75 +
                                                        score.conditioningScore *
                                                            0.25
                                                )
                                            ).coerceIn(0.01, 1.0),
                                ),
                            sourceDiversity =
                                maxOf(
                                    candidate.sourceDiversity *
                                        0.72,
                                    score.sourceDiversityScore *
                                        0.28,
                                ).coerceIn(0.0, 1.0),
                        )
                    }
                }.toMutableList()
        }

        latestDenseGeometricConditioningScore =
            if (geometricConditioningCount == 0) {
                0.0
            } else {
                (
                    geometricConditioningSum /
                        geometricConditioningCount.toDouble()
                    ).coerceIn(0.0, 1.0)
            }
        latestDenseGeometricConditioningWarnings =
            geometricConditioningWarnings.toList().take(12)

        latestDenseGeometricConditioningMedianParallaxDegrees =
            if (geometricParallaxCount == 0) {
                0.0
            } else {
                (
                    geometricParallaxSum /
                        geometricParallaxCount.toDouble()
                ).coerceAtLeast(0.0)
            }
        latestDenseGeometricConditioningMinimumObservationScore =
            geometricMinimumObservationScore.coerceIn(0.0, 1.0)

        conditionedGrouped.forEach { (pointKey, candidates) ->
            if (
                gatedPointCount >= 192 ||
                candidates.size < 3
            ) {
                gatedObservations += candidates
                return@forEach
            }

            val gateResult =
                crossViewLandmarkIdentityGateEngine.evaluate(
                    CrossViewLandmarkIdentityGateEngine.Input(
                        timestampNanos = timestampNanos,
                        observations = candidates.map { candidate ->
                            val observation = candidate.observation
                            CrossViewLandmarkIdentityGateEngine.Observation(
                                cameraId = observation.cameraId,
                                timestampNanos = timestampNanos,
                                u = observation.u,
                                v = observation.v,
                                cameraPose = observation.cameraPose,
                                intrinsics = observation.intrinsics,
                                confidence = observation.sourceQuality,
                                uncertainty = observation.observationUncertainty,
                                visibility = observation.visibility,
                                temporalStability =
                                    observation.temporalStability,
                                sourceQuality = observation.sourceQuality,
                            )
                        },
                    ),
                )

            gatedPointCount++
            gateScoreSum += gateResult.identityConsistency
            gateWarnings += gateResult.warnings

            val scoreByCamera =
                gateResult.scores.associateBy {
                    it.cameraId
                }

            val acceptedCameras =
                gateResult.acceptedObservations
                    .map { it.cameraId }
                    .toSet()

            val filtered =
                candidates.mapNotNull { candidate ->
                    val cameraId = candidate.observation.cameraId
                    if (cameraId !in acceptedCameras) {
                        rejectedPointObservationCount++
                        return@mapNotNull null
                    }

                    val confidenceScale =
                        scoreByCamera[cameraId]
                            ?.confidenceScale
                            ?.coerceIn(0.20, 1.0)
                            ?: 1.0

                    candidate.copy(
                        semanticConfidence = (
                            candidate.semanticConfidence *
                                confidenceScale
                            ).coerceIn(0.0, 1.0),
                        sourceDiversity = (
                            candidate.sourceDiversity *
                                confidenceScale
                            ).coerceIn(0.0, 1.0),
                    )
                }

            if (
                filtered.size >= 2 &&
                !gateResult.hardFailure
            ) {
                gatedObservations += filtered
            } else {
                rejectedPointObservationCount += candidates.size
            }

            if (gateResult.hardFailure) {
                gateWarnings +=
                    "dense-identity-gate-hard-failure:" +
                        pointKey.subjectId +
                        ":" +
                        pointKey.pointId
            }
        }

        val observations =
            gatedObservations
                .sortedWith(
                    compareBy<
                        DenseMultiViewSurfaceReconstructionEngine.PointObservation,
                    > {
                        it.key.subjectId
                    }.thenBy {
                        it.key.pointId
                    },
                )

        val identityExchangeEvidence =
            observations
                .groupBy { it.key.subjectId }
                .map { (subjectId, subjectPoints) ->
                    MultiPersonIdentityExchangeRiskEngine.SubjectEvidence(
                        subjectId = subjectId,
                        visibleLandmarks =
                            subjectPoints.count {
                                it.visibility >= 0.35
                            },
                        totalLandmarks =
                            subjectPoints
                                .map { it.key.pointId }
                                .distinct()
                                .size
                                .coerceAtLeast(1),
                        meanConfidence =
                            subjectPoints
                                .map { it.semanticConfidence }
                                .average()
                                .coerceIn(0.0, 1.0),
                        crossViewAgreement =
                            subjectPoints
                                .map {
                                    it.observation.crossViewAgreement
                                }
                                .average()
                                .coerceIn(0.0, 1.0),
                        temporalIdentityConfidence =
                            subjectPoints
                                .map {
                                    it.observation.temporalStability
                                }
                                .average()
                                .coerceIn(0.0, 1.0),
                        contactOverlap =
                            subjectPoints
                                .map { it.contactProbability }
                                .average()
                                .coerceIn(0.0, 1.0),
                        cameraCount =
                            subjectPoints
                                .map { it.observation.cameraId }
                                .distinct()
                                .size
                                .coerceAtLeast(1),
                    )
                }

        val identityExchangeResult =
            multiPersonIdentityExchangeRiskEngine.evaluate(
                timestampNanos = timestampNanos,
                subjects = identityExchangeEvidence,
            )

        latestDenseIdentityGateQuality =
            minOf(
                latestDenseIdentityGateQuality,
                identityExchangeResult.productionIdentityConfidence,
            )
        latestIdentityExchangeResult = identityExchangeResult

        latestDenseIdentityGateWarnings =
            (
                latestDenseIdentityGateWarnings +
                    identityExchangeResult.warnings
                        .map { "identity-exchange-$it" }
            ).distinct().take(16)

        latestDenseIdentityGateEvaluatedPoints =
            gatedPointCount
        latestDenseIdentityGateRejectedCount =
            rejectedPointObservationCount
        latestDenseIdentityGateQuality =
            if (gatedPointCount <= 0) {
                1.0
            } else {
                (
                    gateScoreSum /
                        gatedPointCount.toDouble()
                ).coerceIn(0.0, 1.0)
            }
        latestDenseIdentityGateWarnings =
            gateWarnings.toList().take(12)

        if (observations.isEmpty()) return null

        val result =
            denseSurfaceReconstructionEngine.process(
                DenseMultiViewSurfaceReconstructionEngine.FrameInput(
                    timestampNanos = timestampNanos,
                    points = observations,
                ),
            )

        latestDenseSurfaceResult = result
        return result
    }

    fun resetDenseSurface() {
        denseSurfaceReconstructionEngine.reset()
        denseLandmarkHistory.clear()
        crossViewLandmarkIdentityGateEngine.reset()
        multiPersonIdentityExchangeRiskEngine.reset()
        latestIdentityExchangeResult =
            MultiPersonIdentityExchangeRiskEngine.Result(
                state = MultiPersonIdentityExchangeRiskEngine.State.CLEAR,
                productionIdentityConfidence = 1.0,
                exchangeRisk = 0.0,
                temporalContinuity = 1.0,
                occlusionBurden = 0.0,
                ambiguity = 0.0,
                evaluatedSubjects = 0,
                heldSubjects = 0,
                subjectRisks = emptyList(),
                warnings = emptyList(),
            )
        latestDenseSurfaceResult = null
        latestDenseIdentityGateQuality = 1.0
        latestDenseIdentityGateRejectedCount = 0
        latestDenseIdentityGateWarnings = emptyList()
        latestDenseIdentityGateEvaluatedPoints = 0
    }

    private fun buildProbabilityModes(
        landmark: DenseLandmarkObservation,
        width: Int,
        height: Int,
    ): List<TriangulationEngine.ProbabilityMode> {
        val sigmaX =
            (
                landmark.probabilitySigmaXNormalized.coerceIn(1e-4, 0.25) *
                    width.toDouble()
            ).coerceIn(0.35, 64.0)
        val sigmaY =
            (
                landmark.probabilitySigmaYNormalized.coerceIn(1e-4, 0.25) *
                    height.toDouble()
            ).coerceIn(0.35, 64.0)

        if (!sigmaX.isFinite() || !sigmaY.isFinite()) {
            return emptyList()
        }

        return listOf(
            TriangulationEngine.ProbabilityMode(
                u = landmark.normalizedX * width.toDouble(),
                v = landmark.normalizedY * height.toDouble(),
                sigmaX = sigmaX,
                sigmaY = sigmaY,
                weight = 1.0,
            ),
        )
    }

    private fun pruneDenseLandmarkHistory(
        nowTimestampNanos: Long,
    ) {
        val cutoff =
            nowTimestampNanos - 500_000_000L

        val iterator =
            denseLandmarkHistory.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            while (
                entry.value.isNotEmpty() &&
                entry.value.first().timestampNanos < cutoff
            ) {
                entry.value.removeFirst()
            }

            if (entry.value.isEmpty()) {
                iterator.remove()
            }
        }
    }

    fun reset() {
        buffer.clear()
        adaptiveEvidenceWeightEngine.reset()
        latestCameraContributionPlan =
            MultiViewCameraContributionPlanner.Plan(
                selectedDeviceIds = emptySet(),
                excludedDeviceIds = emptySet(),
                rankedCameras = emptyList(),
                diversitySatisfied = false,
                reason = "reset",
            )
        skeleton.reset()
        temporal.reset()
        continuity.reset()
        reprojectionConsistency.reset()
        rigGeometryDrift.reset()
        geometricObservability.reset()
        uncertaintyPropagation.reset()
        kinematicConstraintEngine.reset()
        physicsConstraintEngine.reset()
        worldSpaceMotionConsistencyEngine.reset()
        global4DConsistencyEngine.reset()
        denseSurfaceReconstructionEngine.reset()
        denseLandmarkHistory.clear()
        crossViewLandmarkIdentityGateEngine.reset()
        latestDenseSurfaceResult = null
        latestDenseIdentityGateQuality = 1.0
        latestDenseIdentityGateRejectedCount = 0
        latestDenseIdentityGateWarnings = emptyList()
        latestDenseIdentityGateEvaluatedPoints = 0
        latestDenseGeometricConditioningScore = 1.0
        latestDenseGeometricConditioningMedianParallaxDegrees = 0.0
        latestDenseGeometricConditioningMinimumObservationScore = 1.0
        latestDenseGeometricConditioningWarnings = emptyList()
        reconstructionHistory.clear()
        latestSpatiotemporalBundleResult = null
        subframeAlignmentStrictness = 0.82
    }

    private fun estimateSpatiotemporalAlignment(
        timestampNanos: Long,
        skeleton: SkeletonFrame,
    ): SpatiotemporalBundleAdjustmentEngine.Result {
        if (skeleton.joints.isEmpty()) {
            return emptySpatiotemporalResult("empty-skeleton-history")
        }

        reconstructionHistory.addLast(skeleton)
        while (reconstructionHistory.size > 8) {
            reconstructionHistory.removeFirst()
        }

        val calibration =
            calibrationProvider()
                ?: return emptySpatiotemporalResult(
                    "calibration-unavailable",
                )

        val snapshots =
            buffer.snapshot()

        val cameras =
            snapshots
                .map { it.deviceId }
                .distinct()
                .mapNotNull { deviceId ->
                    val intrinsics =
                        calibration.intrinsics[deviceId]
                            ?: return@mapNotNull null
                    val extrinsics =
                        calibration.extrinsics[deviceId]
                            ?: return@mapNotNull null
                    SpatiotemporalBundleAdjustmentEngine.Camera(
                        id = deviceId.value,
                        pose = extrinsics.pose,
                        intrinsics = intrinsics,
                        fixed =
                            extrinsics.referenceDevice ==
                                deviceId,
                    )
                }

        if (cameras.size < 2) {
            return emptySpatiotemporalResult(
                "insufficient-calibrated-cameras",
            )
        }

        val oldestHistoryTimestamp =
            reconstructionHistory
                .first()
                .timestampNanos

        val trajectories =
            reconstructionHistory.flatMap { frame ->
                frame.joints.map { joint ->
                    SpatiotemporalBundleAdjustmentEngine.TrajectorySample(
                        pointId = joint.landmarkId,
                        timestampNanos = frame.timestampNanos,
                        position = joint.position,
                    )
                }
            }

        val observations =
            buffer.recentFramesSince(
                oldestHistoryTimestamp,
            ).flatMap { frame ->
                frame.landmarks.map { landmark ->
                    SpatiotemporalBundleAdjustmentEngine.DynamicObservation(
                        cameraId = frame.deviceId.value,
                        pointId = landmark.landmarkId,
                        timestampNanos = frame.timestampNanos,
                        pixel =
                            Vec2(
                                landmark.x,
                                landmark.y,
                            ),
                        confidence =
                            landmark.confidence
                                .toDouble()
                                .coerceIn(
                                    0.0,
                                    1.0,
                                ),
                    )
                }
            }

        val referenceCameraId =
            cameras.firstOrNull { it.fixed }?.id
                ?: cameras.firstOrNull()?.id
                ?: return emptySpatiotemporalResult(
                    "reference-camera-unavailable",
                )

        val rawResult =
            spatiotemporalBundleAdjustmentEngine.optimize(
                cameras = cameras,
                trajectories = trajectories,
                observations = observations,
                referenceCameraId = referenceCameraId,
            )

        val minimumAcceptedConfidence =
            (
                0.25 +
                    0.70 *
                        subframeAlignmentStrictness
                ).coerceIn(0.25, 0.95)

        val result =
            rawResult.copy(
                accepted =
                    rawResult.accepted &&
                        rawResult.confidence >=
                            minimumAcceptedConfidence,
                warnings =
                    if (
                        rawResult.confidence <
                            minimumAcceptedConfidence
                    ) {
                        rawResult.warnings +
                            "subframe-strictness-gate"
                    } else {
                        rawResult.warnings
                    },
            )

        latestSpatiotemporalBundleResult = result
        return result
    }

    private fun emptySpatiotemporalResult(
        warning: String,
    ): SpatiotemporalBundleAdjustmentEngine.Result =
        SpatiotemporalBundleAdjustmentEngine.Result(
            cameraOffsetsNanos = emptyMap(),
            cameraResults = emptyList(),
            initialRmsPx = Double.POSITIVE_INFINITY,
            finalRmsPx = Double.POSITIVE_INFINITY,
            confidence = 0.0,
            accepted = false,
            converged = false,
            warnings = listOf(warning),
        )

    private fun rollingShutterEvidence(
        landmark: com.nivukx.nox.core.model.LandmarkObservation,
    ): RollingShutterEvidenceEngine.Result =
        rollingShutterEvidenceEngine.evaluate(
            RollingShutterEvidenceEngine.Input(
                normalizedRow =
                    landmark.sensorRowNormalized
                        ?: (
                            landmark.y /
                                landmark.frameHeight.coerceAtLeast(1)
                                    .toDouble()
                            ).coerceIn(0.0, 1.0),
                yPx = landmark.y,
                frameHeightPx = landmark.frameHeight.coerceAtLeast(1),
                rollingShutterSkewNanos =
                    landmark.rollingShutterSkewNanos,
                frameDurationNanos =
                    landmark.frameDurationNanos,
                exposureTimeNanos =
                    landmark.exposureTimeNanos,
                captureTimingConfidence =
                    landmark.captureTimingConfidence,
            ),
        )

    private fun buildReprojectionFrames(
        timestampNanos: Long,
        clockSkewNanos: Long,
    ): List<MultiViewReprojectionConsistencyEngine.CameraFrame> {
        val calibration = calibrationProvider() ?: return emptyList()

        return buffer.matchAll(timestampNanos).mapNotNull { frame ->
            val intrinsics = calibration.intrinsics[frame.deviceId]
                ?: return@mapNotNull null
            val extrinsics = calibration.extrinsics[frame.deviceId]
                ?: return@mapNotNull null

            val observations = frame.landmarks
                .filter {
                    it.space ==
                        com.nivukx.nox.core.model.LandmarkSpace.PIXELS
                }
                .map { landmark ->
                    val rollingShutterEvidence =
                        rollingShutterEvidence(landmark)

                    MultiViewReprojectionConsistencyEngine.Observation(
                        landmarkId = landmark.landmarkId,
                        camera = CameraObservation(
                            deviceId = frame.deviceId,
                            point = Vec2(landmark.x, landmark.y),
                            confidence =
                                (
                                    landmark.confidence *
                                        rollingShutterEvidence.confidenceScale
                                    ).coerceIn(1e-4, 1.0),
                            intrinsics = intrinsics,
                            pose = extrinsics.pose,
                            calibrationUncertainty =
                                calibration.uncertaintyProfiles[
                                    frame.deviceId
                                ] ?: CalibrationUncertaintyProfile(),
                            rollingShutterFraction =
                                rollingShutterEvidence.rollingShutterFraction,
                            temporalAlignmentConfidence =
                                (
                                    landmark.captureTimingConfidence *
                                        rollingShutterEvidence.confidenceScale
                                    ).coerceIn(0.0, 1.0),
                            rowTimeOffsetNanos =
                                rollingShutterEvidence.rowOffsetNanos,
                        ),
                    )
                }

            if (observations.isEmpty()) {
                null
            } else {
                MultiViewReprojectionConsistencyEngine.CameraFrame(
                    deviceId = frame.deviceId,
                    timestampNanos = frame.timestampNanos,
                    observations = observations,
                    clockOffsetNanos =
                        if (frame.timestampNanos != timestampNanos) {
                            clockSkewNanos
                        } else {
                            0L
                        },
                )
            }
        }
    }
    private fun buildUltraPrecisionInput(
        calibration: CalibrationSnapshot,
        cameraFrames:
            List<SkeletonReconstructionEngine.CameraFrame>,
        cameraCount: Int,
        worldSpaceSkeleton: SkeletonFrame,
        reprojectionScore: Double,
        supportEvidenceScore: Double? = null,
        geometricInformationP95MajorMeters: Double,
        spatiotemporalConfidence: Double,
        clockSkewNanos: Long,
        evaluationDroppedObservations: Int,
        bodyAcceptedRatio: Double,
        denseIdentityGateQuality: Double,
        denseSurfaceTemporalStability: Double,
        denseSurfaceViewDiversity: Double,
        denseSubMillimeterGeometryFeasibility: Double,
        denseTailAdjustedFirstOrderDepthErrorMillimeters: Double,
        denseTailAdjustedSubMillimeterFeasibilityScore: Double,
        denseFirstOrderDepthErrorMillimeters: Double,
        denseMotionStability: Double,
        denseKinematicInnovationMillimeters: Double,
    ): UltraPrecisionEvidenceEngine.Input {
        val selectedIds =
            cameraFrames.map { it.deviceId }.distinct()

        val centers =
            selectedIds.mapNotNull { id ->
                calibration.extrinsics[id]?.pose?.position
            }

        val baseline =
            if (centers.size < 2) {
                0.0
            } else {
                centers.indices
                    .flatMap { i ->
                        (i + 1 until centers.size).map { j ->
                            centers[i].distanceTo(centers[j])
                        }
                    }
                    .filter { it.isFinite() && it > 0.0 }
                    .maxOrNull()
                    ?: 0.0
            }

        val focalValues =
            selectedIds.mapNotNull { id ->
                calibration.intrinsics[id]?.let {
                    (it.fx + it.fy) * 0.5
                }
            }
        val effectiveFocal =
            focalValues
                .filter { it.isFinite() && it > 0.0 }
                .average()
                .takeIf { it.isFinite() && it > 0.0 }
                ?: 1_000.0

        val workingDistances =
            worldSpaceSkeleton.joints
                .map { it.position.norm() }
                .filter { it.isFinite() && it > 0.0 }
                .sorted()

        val workingDistance =
            if (workingDistances.isEmpty()) {
                2.5
            } else {
                workingDistances[workingDistances.size / 2]
                    .coerceAtLeast(0.25)
            }

        val parallaxAngles = mutableListOf<Double>()
        for (i in 0 until cameraFrames.lastIndex) {
            val aFrame = cameraFrames[i]
            val aById =
                aFrame.observations.associateBy { it.landmarkId }

            for (j in (i + 1 until cameraFrames.size)) {
                val bFrame = cameraFrames[j]
                if (aFrame.deviceId == bFrame.deviceId) continue

                val bById =
                    bFrame.observations.associateBy { it.landmarkId }

                val sharedIds =
                    aById.keys.intersect(bById.keys)
                for (landmarkId in sharedIds.take(96)) {
                    val a =
                        aById[landmarkId] ?: continue
                    val b =
                        bById[landmarkId] ?: continue

                    val rayA = observationRay(
                        a.point,
                        a.intrinsics,
                        a.pose,
                    )
                    val rayB = observationRay(
                        b.point,
                        b.intrinsics,
                        b.pose,
                    )
                    val angle =
                        Math.toDegrees(
                            atan2(
                                rayA.cross(rayB).norm(),
                                rayA.dot(rayB),
                            ),
                        )
                    if (angle.isFinite() && angle > 0.05) {
                        parallaxAngles += angle
                    }
                }
            }
        }

        val medianParallax =
            parallaxAngles.sorted().let {
                when {
                    it.isEmpty() -> 0.0
                    it.size % 2 == 1 -> it[it.size / 2]
                    else ->
                        (it[it.size / 2 - 1] + it[it.size / 2]) * 0.5
                }
            }.coerceIn(0.0, 179.0)

        val meanSpeed =
            worldSpaceSkeleton.joints
                .map { it.velocity.norm() }
                .filter { it.isFinite() }
                .average()
                .takeIf { it.isFinite() }
                ?: 0.0

        val calibrationConfidence =
            (
                exp(
                    -calibration.reprojectionErrorPx
                        .coerceAtLeast(0.0) / 3.0,
                ) *
                    (
                        0.60 +
                            0.40 *
                                spatiotemporalConfidence
                                    .coerceIn(0.0, 1.0)
                    ) *
                    (
                        1.0 /
                            (
                                1.0 +
                                    geometricInformationP95MajorMeters
                                        .coerceAtLeast(0.0) * 20.0
                                )
                        ).coerceIn(0.0, 1.0)
                ).coerceIn(0.0, 1.0)

        val synchronizationConfidence =
            exp(
                -abs(clockSkewNanos).toDouble() /
                    2_500_000.0,
            ).coerceIn(0.0, 1.0)

        val selectedRatio =
            latestCameraContributionPlan
                .selectedDeviceIds
                .size /
                cameraCount.coerceAtLeast(1).toDouble()

        val viewDiversity =
            (
                if (latestCameraContributionPlan.diversitySatisfied) {
                    0.90
                } else {
                    0.45
                } +
                    0.10 * selectedRatio.coerceIn(0.0, 1.0)
                ).coerceIn(0.0, 1.0)

        val droppedRatio =
            evaluationDroppedObservations.toDouble() /
                (
                    evaluationDroppedObservations +
                        worldSpaceSkeleton.joints.size
                    ).coerceAtLeast(1).toDouble()

        val baseCrossViewAgreement =
            (
                reprojectionScore * 0.62 +
                    bodyAcceptedRatio * 0.20 +
                    denseSurfaceViewDiversity * 0.10 +
                    (1.0 - droppedRatio).coerceIn(0.0, 1.0) * 0.08
                ).coerceIn(0.0, 1.0)

        val crossViewAgreement =
            supportEvidenceScore?.let { support ->
                (
                    baseCrossViewAgreement * 0.92 +
                        support.coerceIn(0.0, 1.0) * 0.08
                ).coerceIn(0.0, 1.0)
            } ?: baseCrossViewAgreement

        val occlusionRatio =
            (1.0 - bodyAcceptedRatio).coerceIn(0.0, 1.0)

        val surfaceStability =
            (
                denseSurfaceTemporalStability * 0.72 +
                    denseSurfaceViewDiversity * 0.28
                ).coerceIn(0.0, 1.0)

        return UltraPrecisionEvidenceEngine.Input(
            workingDistanceMeters = workingDistance,
            cameraBaselineMeters = baseline,
            effectiveFocalLengthPixels = effectiveFocal,
            medianParallaxDegrees = medianParallax,
            reprojectionErrorPixels =
                calibration.reprojectionErrorPx.coerceAtLeast(0.0),
            subpixelObservationSigmaPixels =
                max(
                    0.05,
                    (
                        calibration.reprojectionErrorPx * 0.18 +
                            (
                                1.0 -
                                    denseIdentityGateQuality
                            ).coerceAtLeast(0.0) * 0.12
                        ).coerceIn(0.05, 1.5),
                ),
            cameraTranslationSigmaMeters =
                max(
                    0.0005,
                    geometricInformationP95MajorMeters
                        .coerceAtLeast(0.0005),
                ),
            cameraAngularSigmaDegrees = 0.05,
            synchronizationSkewMilliseconds =
                abs(clockSkewNanos).toDouble() / 1_000_000.0,
            rollingShutterEquivalentMilliseconds =
                (
                    (
                        1.0 -
                            spatiotemporalConfidence
                    ).coerceIn(0.0, 1.0) * 2.0
                    ),
            motionSpeedMetersPerSecond = meanSpeed,
            metricScaleUncertaintyRatio =
                (
                    0.0005 +
                        (
                            1.0 -
                                (
                                    calibrationConfidence *
                                        crossViewAgreement
                                    ).coerceIn(0.0, 1.0)
                            ) * 0.0035
                    ).coerceIn(0.0005, 0.02),
            calibrationConfidence = calibrationConfidence,
            synchronizationConfidence =
                synchronizationConfidence,
            exposureIntegrationMilliseconds =
                cameraFrames
                    .map {
                        it.exposureTimeNanos.toDouble() / 1_000_000.0
                    }
                    .filter { it.isFinite() && it > 0.0 }
                    .maxOrNull()
                    ?: 0.0,
            viewDiversity = viewDiversity,
            crossViewAgreement = crossViewAgreement,
            occlusionRatio = occlusionRatio,
            identityContinuity =
                denseIdentityGateQuality.coerceIn(0.0, 1.0),
            surfaceTemporalStability = surfaceStability,
            thermalRisk = 0.0,
            memoryPressure = 0.0,
            subMillimeterGeometryFeasibility =
                denseSubMillimeterGeometryFeasibility
                    .coerceIn(0.0, 1.0),
            tailAdjustedFirstOrderDepthErrorMillimeters =
                denseTailAdjustedFirstOrderDepthErrorMillimeters,
            tailAdjustedSubMillimeterFeasibilityScore =
                denseTailAdjustedSubMillimeterFeasibilityScore
                    .coerceIn(0.0, 1.0),
            firstOrderDepthErrorMillimeters =
                denseFirstOrderDepthErrorMillimeters,
            denseMotionStability =
                denseMotionStability.coerceIn(0.0, 1.0),
            denseKinematicInnovationMillimeters =
                denseKinematicInnovationMillimeters
                    .coerceAtLeast(0.0),
            observationCount =
                worldSpaceSkeleton.joints.size * cameraCount,
            empiricalP95Millimeters =
                ultraPrecisionEmpiricalP95Millimeters,
            empiricalP99Millimeters =
                ultraPrecisionEmpiricalP99Millimeters,
            empiricalSampleCount =
                ultraPrecisionEmpiricalSampleCount,
            matchedReference =
                ultraPrecisionMatchedReference,
            referenceKind =
                ultraPrecisionReferenceKind,
            referenceP95Millimeters =
                ultraPrecisionReferenceP95Millimeters,
            referenceUncertaintyMillimeters =
                ultraPrecisionReferenceUncertaintyMillimeters,
            systemMeasurementUncertaintyMillimeters =
                geometricInformationP95MajorMeters
                    .takeIf { it.isFinite() && it >= 0.0 }
                    ?.times(1000.0),
        )
    }

    private fun observationRay(
        point: Vec2,
        intrinsics:
            com.nivukx.nox.core.model.CameraIntrinsics,
        pose:
            com.nivukx.nox.core.model.Pose,
    ): com.nivukx.nox.core.model.Vec3 =
        pose.rotation.rotate(
            com.nivukx.nox.core.model.Vec3(
                (point.x - intrinsics.cx) /
                    intrinsics.fx.coerceAtLeast(1e-6),
                (point.y - intrinsics.cy) /
                    intrinsics.fy.coerceAtLeast(1e-6),
                1.0,
            ),
        ).normalized()

}
