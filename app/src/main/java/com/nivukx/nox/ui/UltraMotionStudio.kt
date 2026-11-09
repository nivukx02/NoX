package com.nivukx.nox.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.SessionPresentationSettings
import com.nivukx.nox.core.reconstruction.PhysicalPrecisionCapabilityEngine
import com.nivukx.nox.ui.studio.MotionStudioOperatorDeck
import com.nivukx.nox.ui.studio.PostCaptureReconstructionStatusPanel
import com.nivukx.nox.ui.studio.HybridCaptureModePanel
import com.nivukx.nox.ui.studio.ActorDetailBudgetPanel
import com.nivukx.nox.ui.studio.PrecisionWorkflowShelf
import com.nivukx.nox.ui.studio.PrecisionEvidenceSurface
import com.nivukx.nox.ui.studio.ReferenceGradeEvidenceCard
import com.nivukx.nox.ui.studio.StudioTelemetryState
import com.nivukx.nox.ui.studio.LiquidGlassSurface
import com.nivukx.nox.ui.studio.StudioVisualBudgetEngine
import com.nivukx.nox.ui.studio.PhysicalPrecisionAuthorityPanel
import kotlin.math.max
import kotlin.math.min

/**
 * High-end motion-capture operator surface.
 *
 * Critical telemetry stays visible, calibration state is explicit, capture state is persistent,
 * and deep configuration is grouped below the live reconstruction stage.
 */
@Composable
private fun UltraMotionStudioMetricRow(
    label: String,
    value: String,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            label,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            value,
            style = MaterialTheme.typography.labelLarge,
        )
    }
}

@Composable
private fun CrowdIntelligencePanel(
    crowdAppearanceQuality: Double = 0.0,
    meanIdentitySafety: Double,
    interactionPressure: Double,
    differentialMotionScore: Double,
    schedulingCapacity: Double,
    worldLandmarkCoverage: Double,
    worldBoneConsistency: Double,
    worldGeometryReliability: Double,
    segmentationEvidence: Double,
    segmentationCoverage: Double,
    crowdActorSilhouetteSupport: Double = 0.0,
    crowdActorSilhouetteAmbiguity: Double = 0.0,
    personSegmentationConfidence: Double = 0.0,
    personSegmentationBoundaryStability: Double = 0.0,
    personSegmentationLatencyMs: Long = 0L,
    personSegmentationState: String = "DEGRADED",
    personSegmentationDelegate: String = "NONE",
    personSegmentationFallbackCount: Long = 0L,
    bottleneck: String,
    heldActors: Int,
    predictedActors: Int,
    warnings: List<String>,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = true,
        effectScale = 1.0f,
        modifier = modifier,
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        "CROWD INTELLIGENCE",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Per-actor scheduling, interaction pressure and identity safety",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                StatusPill(
                    text = bottleneck.replace('_', ' ').uppercase(),
                    healthy = bottleneck == "NONE",
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    label = "Identity safety",
                    value = "${(meanIdentitySafety * 100.0).toInt()}%",
                    detail = "cross-actor continuity",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Differential motion",
                    value = "${(differentialMotionScore * 100.0).toInt()}%",
                    detail = "camera-compensated",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    label = "Interaction",
                    value = "${(interactionPressure * 100.0).toInt()}%",
                    detail = "occlusion/contact pressure",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Scheduler",
                    value = "${(schedulingCapacity * 100.0).toInt()}%",
                    detail = "detail capacity",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    label = "3D landmarks",
                    value = "${(worldLandmarkCoverage * 100.0).toInt()}%",
                    detail = "body-space coverage",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Bone consistency",
                    value = "${(worldBoneConsistency * 100.0).toInt()}%",
                    detail = "scale-invariant geometry",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "3D prior",
                    value = "${(worldGeometryReliability * 100.0).toInt()}%",
                    detail = "reconstruction prior",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Appearance",
                    value = "${(crowdAppearanceQuality * 100.0).toInt()}%",
                    detail = "track continuity",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    label = "Silhouette",
                    value = "${(segmentationEvidence * 100.0).toInt()}%",
                    detail = "dense support quality",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Foreground",
                    value = "${(segmentationCoverage * 100.0).toInt()}%",
                    detail = "global scene coverage",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Seg. model",
                    value = "${(personSegmentationConfidence * 100.0).toInt()}%",
                    detail =
                        "${personSegmentationState.lowercase()} · " +
                            "${personSegmentationLatencyMs} ms",
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    label = "Actor support",
                    value = "${(crowdActorSilhouetteSupport * 100.0).toInt()}%",
                    detail =
                        "${(crowdActorSilhouetteAmbiguity * 100.0).toInt()}% overlap ambiguity",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                UltraMotionStudioMetricRow(
                    "Predicted actors",
                    predictedActors.toString(),
                )
                UltraMotionStudioMetricRow(
                    "Held actors",
                    heldActors.toString(),
                )
            }

            if (warnings.isNotEmpty()) {
                Text(
                    warnings.take(3).joinToString("  •  "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
fun UltraMotionStudio(
    state: String,
    frameCount: Long,
    inferenceFps: Int,
    poseConfidence: Float,
    reconstructionQuality: Double,
    worldConsistencyScore: Double,
    worldConsistencyWarnings: List<String>,
    gazeConfidence: Double,
    eye3DTemporalConsistencyConfidence: Double,
    eye3DTemporalConsistencyWarnings: List<String>,
    faceEyeConsistencyConfidence: Double,
    faceEyeConsistencyWarnings: List<String>,
    face3DTemporalConsistencyConfidence: Double,
    face3DTemporalConsistencyWarnings: List<String>,
    gazeStability: Double,
    faceHeadYaw: Double,
    motionEventType: String,
    motionEventConfidence: Double,
    faceHeadRoll: Double,
    handTemporalConsistencyConfidence: Double,
    handTemporalConsistencyWarnings: List<String>,
    reconstructionErrorPx: Double,
    reconstructionJoints: Int,
    motionClass: String,
    motionScore: Double,
    commonModeMotionNormalizedPerSecond: Double,
    differentialMotionScore: Double,
    cameraCompensatedMotionScore: Double,
    microMotionScore: Double,
    directionChangeRatio: Double,
    motionMeasurementUncertainty: Double,
    motionTemporalMemoryScore: Double? = null,
    motionEvidenceScore: Double,
    motionEvidenceConfidence: Double,
    motionEvidenceWarnings: List<String>,
    temporalStability: Double,
    occlusionRisk: Double,
    dominantSubject: String,
    subjectLock: String,
    subjectLockConfidence: Double,
    gpuBackend: String,
    gpuLatencyMs: Double,
    calibrationReady: Boolean,
    captureTimingConfidence: Double,
    captureTimingFps: Double,
    captureTimingJitterMillis: Double,
    captureTimingHealthy: Boolean,
    cameraSessionVerification: String = "UNKNOWN",
    cameraSessionSupported: Boolean? = null,
    highSpeedSessionVerification: String = "UNKNOWN",
    highSpeedSessionSupported: Boolean? = null,
    highSpeedSessionActive: Boolean = false,
    highSpeedSelectedFps: Int? = null,
    highSpeedSessionReason: String? = null,
    hardwareReadoutTimestampSupported: Boolean = false,
    hardwareReadoutTimingState: String = "UNKNOWN",
    hardwareReadoutTimingConfidence: Double = 0.0,
    hardwareReadoutP95JitterMillis: Double = Double.POSITIVE_INFINITY,
    hardwareReadoutMedianDelayMillis: Double = Double.NaN,
    latestReadoutTimestampNanos: Long? = null,
    concurrentRigAdmissionState: String = "NOT_APPLICABLE",
    concurrentRigAdmissionScore: Double = 1.0,
    concurrentRigSelectedCameraIds: List<String> = emptyList(),
    concurrentRigBlockers: List<String> = emptyList(),
    concurrentRigWarnings: List<String> = emptyList(),
    thermalExtrinsicDriftState: String = "UNKNOWN",
    thermalExtrinsicDriftConfidence: Double = 0.0,
    thermalExtrinsicTranslationDriftMm: Double = 0.0,
    thermalExtrinsicRotationDriftDegrees: Double = 0.0,
    thermalExtrinsicTemperatureDeltaCelsius: Double = 0.0,
    thermalExtrinsicHardFailure: Boolean = false,
    thermalExtrinsicWarnings: List<String> = emptyList(),
    faceEyeMetricState: String = "NO_DATA",
    faceEyeMetricConfidence: Double = 0.0,
    faceEyeMetricEyeUncertaintyMm: Double = Double.POSITIVE_INFINITY,
    faceEyeMetricWarnings: List<String> = emptyList(),
    faceEyeMetricRecommendations: List<String> = emptyList(),
    multiPersonIdentityContinuity: Double = 0.0,
    multiPersonIdentityReidentifications: Int = 0,
    multiPersonIdentityPredictions: Int = 0,
    multiPerson3DReconstructedCount: Int = 0,
    multiPerson3DDetailedCount: Int = 0,
    multiPerson3DPredictedCount: Int = 0,
    multiPerson3DPredictionMaxAgeMillis: Long = 0L,
    multiPersonObservedCount: Int = 0,
    multiPersonAdmittedCount: Int = 0,
    multiPersonHeldCount: Int = 0,
    multiPersonStarvationRisk: Double = 0.0,
    multiPersonFairnessBoostedCount: Int = 0,
    rigQuarantinedCameraCount: Int = 0,
    crowdMotionMeanIdentitySafety: Double = 0.0,
    crowdMotionInteractionPressure: Double = 0.0,
    crowdMotionDifferentialScore: Double = 0.0,
    crowdMotionSchedulingCapacity: Double = 0.0,
    crowdWorldLandmarkCoverage: Double = 0.0,
    crowdWorldBoneConsistency: Double = 0.0,
    crowdWorldGeometryReliability: Double = 0.0,
    crowdSegmentationEvidence: Double = 0.0,
    crowdSegmentationCoverage: Double = 0.0,
    crowdActorSilhouetteSupport: Double = 0.0,
    crowdActorSilhouetteAmbiguity: Double = 0.0,
    crowdMotionBottleneck: String = "NONE",
    crowdMotionHeldCount: Int = 0,
    crowdMotionPredictedCount: Int = 0,
    crowdMotionWarnings: List<String> = emptyList(),
    cameraCaptureQualityScore: Double,
    cameraCaptureQualityWarnings: List<String>,
    cameraAnalysisDropRatio: Double = 0.0,
    cameraOpticalBlurRisk: Double = 0.0,
    sessionHealthState: String,
    sessionHealthScore: Double,
    sessionHealthBottleneck: String,
    captureIntegrityScore: Double,
    captureIntegrityBand: String,
    captureIntegrityOutputAllowed: Boolean,
    captureIntegrityBottleneck: String,
    captureEvidenceTrend: String,
    captureEvidenceSeverity: String,
    captureEvidenceCurrentScore: Double,
    captureEvidenceRecentScore: Double,
    captureEvidenceSlope: Double,
    captureEvidenceBottleneck: String,
    captureEvidenceWarnings: List<String>,
    adaptiveViewEvidenceScore: Double,
    adaptiveViewEvidenceDegradedCameraCount: Int,
    adaptiveViewEvidenceHoldCameraCount: Int,
    adaptiveViewEvidenceDominantCamera: String,
    adaptiveViewEvidenceWarnings: List<String>,
    spatiotemporalConfidence: Double,
    spatiotemporalInitialRmsPx: Double,
    spatiotemporalFinalRmsPx: Double,
    spatiotemporalWarnings: List<String>,
    multiCameraSyncConfidence: Double,
    multiCameraSyncState: String,
    temporalAlignmentConfidence: Double,
    temporalAlignmentSpreadMillis: Double,
    temporalAlignmentState: String,
    temporalAlignmentWarnings: List<String>,
    geometricObservabilityScore: Double,
    geometricObservabilityWarnings: List<String>,
    geometricInformationConditionScore: Double = 0.0,
    geometricInformationP95MajorMeters: Double = 0.0,
    geometricInformationP95DepthMeters: Double = 0.0,
    reconstructionUncertaintyConfidence: Double,
    reconstructionUncertaintyMeanMeters: Double,
    reconstructionUncertaintyP95Meters: Double,
    reconstructionTrustedJointRatio: Double,
    reconstructionUncertaintyBottleneck: String,
    reconstructionUncertaintyWarnings: List<String>,
    captureReadiness: com.nivukx.nox.core.model.CaptureReadiness?,
    rigCoverageScore: Double,
    denseMeshTemporalConsistencyConfidence: Double,
    denseMeshTemporalConsistencyWarnings: List<String>,
    denseMeshSurfaceConfidence: Double,
    denseLandmarkCoverage: Double = 0.0,
    denseLandmarkVisibleCoverage: Double = 0.0,
    denseLandmarkUncertaintyP95: Double = 1.0,
    denseLandmarkCrossViewAgreement: Double = 0.0,
    denseLandmarkCrossViewEvidenceAvailable: Boolean = false,
    denseLandmarkTemporalStability: Double = 0.0,
    denseLandmarkContactCoverage: Double = 0.0,
    denseLandmarkPredictionBridgeRatio: Double = 0.0,
    denseLandmarkRejectedObservations: Long = 0L,
    denseLandmarkStaleObservations: Long = 0L,
    denseLandmarkNonFiniteObservations: Long = 0L,
    denseLandmarkHardFailure: Boolean = false,
    denseLandmarkContractScore: Double = 0.0,
    denseLandmarkContractState: String = "UNKNOWN",
    denseLandmarkContractAuthoritative: Boolean = false,
    denseLandmarkContractFailure: String = "NONE",
    denseSurfaceTrustedPointRatio: Double = 0.0,
    denseSurfaceUncertaintyP95Meters: Double = Double.POSITIVE_INFINITY,
    denseSurfaceTopologyStability: Double = 0.0,
    denseSurfaceViewDiversity: Double = 0.0,
    denseSurfaceProbabilisticEvidenceScore: Double = 1.0,
    denseSurfaceMeanReprojectionErrorPx: Double = Double.POSITIVE_INFINITY,
    denseSurfaceMeanAppearanceConsistency: Double = 0.0,
    denseSurfaceMeanSilhouetteConsistency: Double = 0.0,
    denseSurfaceMeanMotionBoundaryConfidence: Double = 0.0,
    denseSurfaceMeanLightingStability: Double = 0.0,
    denseSurfaceMeanDynamicProbability: Double = 0.0,
    denseSurfaceMeanTemporalStability: Double = 1.0,
    denseSurfaceTrustedGaussianRatio: Double = 0.0,
    denseSurfaceAdaptiveViewSelectionScore: Double = 0.0,
    denseSurfaceAdaptiveViewDiversityScore: Double = 0.0,
    denseSurfaceAnisotropicWeakAxisConstraint: Double = 0.0,
    denseSurfaceAnisotropicViewConditionScore: Double = 0.0,
    denseSurfaceExcludedWeakViewCount: Int = 0,
    denseSurfaceHardFailure: Boolean = false,
    denseGeometricConditioningScore: Double = 0.0,
    denseGeometricConditioningMedianParallaxDegrees: Double = 0.0,
    denseGeometricConditioningWarnings: List<String> = emptyList(),
    densePrecisionQualified: Boolean? = null,
    denseMeanJerkMetersPerSecond3: Double = Double.NaN,
    denseMotionStability: Double = Double.NaN,
    denseKinematicInnovationMillimeters: Double = Double.NaN,
    denseSubMillimeterFeasibilityScore: Double = Double.NaN,
    denseFirstOrderDepthErrorMillimeters: Double = Double.POSITIVE_INFINITY,
    cameraContributionMinimumPairwiseDiversity: Double = 0.0,
    cameraContributionViewStarvationRisk: Double = 0.0,
    correlatedEvidenceScore: Double = 0.0,
    correlatedEvidenceCoverage: Double = 0.0,
    correlatedEvidenceFailureRisk: Double = 1.0,
    correlatedEvidenceCommonModeRisk: Double = 0.0,
    correlatedEvidenceRedundantSignalCount: Int = 0,
    correlatedEvidenceState: String = "UNKNOWN",
    correlatedEvidenceWarnings: List<String> = emptyList(),
    identityExchangeState: String = "CLEAR",
    identityExchangeRisk: Double = 0.0,
    identityExchangeConfidence: Double = 1.0,
    identityExchangeEvaluatedSubjects: Int = 0,
    identityExchangeHeldSubjects: Int = 0,
    identityExchangeWarnings: List<String> = emptyList(),
    global4DGaussianConsistency: Double = 1.0,
    global4DGaussianStrainConsistency: Double = 1.0,
    global4DGaussianEvidenceAvailable: Boolean = false,
    kinematicConstraintScore: Double,
    kinematicConstraintAccepted: Boolean,
    kinematicConstraintBottleneck: String,
    rigScore: Double,
    lightingLux: Double,
    colorTemperature: Double,
    skeleton: SkeletonFrame?,
    motionProfile: String,
    motionResponsiveness: Double,
    motionSensitivity: Double,
    predictionHorizon: Double,
    occlusionPersistence: Double,
    jitterSuppression: Double,
    reacquisitionRadius: Double,
    rejectAggressiveness: Double,
    velocityResponsiveness: Double,
    accelerationResponsiveness: Double,
    innovationGateElasticity: Double,
    motionThresholdScale: Double,
    adaptiveRegimeStrength: Double = 0.82,
    reconstructionProfile: String,
    reconstructionSoftAccept: Double,
    reconstructionHoldFrames: Int,
    reconstructionHoldAgeMillis: Long,
    reconstructionStabilitySensitivity: Double,
    temporalPredictionHorizon: Double,
    temporalInnovationGateScale: Double,
    temporalUncertaintyGateScale: Double,
    temporalReprojectionGateScale: Double,
    temporalSmoothingResponsiveness: Double,
    temporalConfidenceDecayScale: Double,
    temporalOutlierRetention: Double,
    temporalTopologyToleranceScale: Double,
    motionCustomization: MotionCaptureCustomization,
    sessionPresentationSettings: SessionPresentationSettings,
    recordingCountdown: Int,
    onCaptureToggle: () -> Unit,
    postCaptureState: String = "IDLE",
    postCaptureProgress: Double = 0.0,
    postCaptureProcessedFrames: Long = 0L,
    postCaptureMaxActors: Int = 0,
    onMotionPreset: (String) -> Unit,
    onCustomizationPreset: (String) -> Unit,
    onCustomizationBodyDetail: (Double) -> Unit,
    onCustomizationFaceDetail: (Double) -> Unit,
    onCustomizationEyeDetail: (Double) -> Unit,
    onCustomizationEyeTemporalStrictness: (Double) -> Unit = {},
    onCustomizationHandDetail: (Double) -> Unit,
    onCustomizationHandTemporalStrictness: (Double) -> Unit = {},
    onCustomizationHandGeometryStrictness: (Double) -> Unit = {},
    onCustomizationDepthDetail: (Double) -> Unit,
    onCustomizationDenseMeshDetail: (Double) -> Unit,
    onCustomizationContactDetail: (Double) -> Unit,
    onCustomizationWorldConsistencyDetail: (Double) -> Unit,
    onCustomizationTopologyConstraintStrength: (Double) -> Unit,
    onCustomizationMaxSubjects: (Int) -> Unit,
    onCustomizationOcclusionPredictionFrames: (Int) -> Unit,
    onCustomizationTargetFps: (Int) -> Unit,
    onCustomizationInferenceFps: (Int) -> Unit,
    onCustomizationMicroMotionPreservation: (Double) -> Unit,
    onCustomizationCameraMotionSuppression: (Double) -> Unit,
    onCustomizationFaceEyeFusionWeight: (Double) -> Unit,
    onCustomizationFaceEyeFusionStrictness: (Double) -> Unit,
    onCustomizationTemporalAlignmentStrictness: (Double) -> Unit,
    onCustomizationOutputIntegrityStrictness: (Double) -> Unit,
    onCustomizationPhysicsConstraintStrictness: (Double) -> Unit,
    onCustomizationGlobal4DConsistencyStrictness: (Double) -> Unit,
    onCustomizationSurfaceEvidenceStrictness: (Double) -> Unit,
    onCustomizationMinimumParallaxDegrees: (Double) -> Unit,
    onCustomizationPreferredTemporalSkewMillis: (Double) -> Unit,
    onCustomizationMaximumTemporalSkewMillis: (Double) -> Unit,
    onCustomizationMaximumReprojectionErrorPx: (Double) -> Unit,
    onCustomizationMaximumReconstructionUncertaintyMeters: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
    onSessionClipName: (String) -> Unit,
    onSessionCountdownSeconds: (Int) -> Unit,
    onRequireCalibrationBeforeRecord: (Boolean) -> Unit,
    onMotionResponsiveness: (Double) -> Unit,
    onMotionSensitivity: (Double) -> Unit,
    onPredictionHorizon: (Double) -> Unit,
    onOcclusionPersistence: (Double) -> Unit,
    onJitterSuppression: (Double) -> Unit,
    onReacquisitionRadius: (Double) -> Unit,
    onRejectAggressiveness: (Double) -> Unit,
    onVelocityResponsiveness: (Double) -> Unit,
    onAccelerationResponsiveness: (Double) -> Unit,
    onInnovationGateElasticity: (Double) -> Unit,
    onMotionThresholdScale: (Double) -> Unit,
    onAdaptiveRegimeStrength: (Double) -> Unit = {},
    onReconstructionPreset: (String) -> Unit,
    onReconstructionSoftAccept: (Double) -> Unit,
    onReconstructionHoldFrames: (Int) -> Unit,
    onReconstructionHoldAgeMillis: (Long) -> Unit,
    onReconstructionStabilitySensitivity: (Double) -> Unit,
    onTemporalPredictionHorizon: (Double) -> Unit,
    onTemporalInnovationGateScale: (Double) -> Unit,
    onTemporalUncertaintyGateScale: (Double) -> Unit,
    onTemporalReprojectionGateScale: (Double) -> Unit,
    onTemporalSmoothingResponsiveness: (Double) -> Unit,
    onTemporalConfidenceDecayScale: (Double) -> Unit,
    onTemporalOutlierRetention: (Double) -> Unit,
    onTemporalTopologyToleranceScale: (Double) -> Unit,
    precisionCapability: PhysicalPrecisionCapabilityEngine.Decision =
        PhysicalPrecisionCapabilityEngine.Decision.idle(),
    modifier: Modifier = Modifier,
    crowdAppearanceQuality: Double,
    personSegmentationCoverage: Double,
    personSegmentationConfidence: Double,
    personSegmentationBoundaryStability: Double,
    personSegmentationLatencyMs: Long,
    personSegmentationState: String,
    personSegmentationDelegate: String,
    personSegmentationFallbackCount: Long,
    denseIdentityGateQuality: Double,
    denseIdentityGateRejectedCount: Int,
    denseIdentityGateEvaluatedPoints: Int,
    denseIdentityGateWarnings: List<String>,
    denseGeometricConditioningMinimumObservationScore: Double,
    denseContactConfidence: Double,
    denseContactProtectedLandmarkCount: Int,
    denseContactWarnings: List<String>,
    global4DExternalContactConsistency: Double,) {
    BoxWithConstraints(modifier.fillMaxSize()) {
        val widthTier = studioWidthTier(maxWidth)
        val expanded = widthTier == StudioWidthTier.EXPANDED ||
            widthTier == StudioWidthTier.LARGE
        val stageHeight = when (widthTier) {
            StudioWidthTier.COMPACT -> 500.dp
            StudioWidthTier.MEDIUM -> 560.dp
            StudioWidthTier.EXPANDED -> 620.dp
            StudioWidthTier.LARGE -> 680.dp
        }
        val outerPadding = when (widthTier) {
            StudioWidthTier.COMPACT -> 10.dp
            StudioWidthTier.MEDIUM -> 14.dp
            StudioWidthTier.EXPANDED -> 18.dp
            StudioWidthTier.LARGE -> 24.dp
        }

        var settingsOpen by rememberSaveable { mutableStateOf(false) }
        var showEvidenceMatrix by rememberSaveable { mutableStateOf(true) }
        var showDiagnostics by rememberSaveable { mutableStateOf(true) }
        var showViewportGrid by rememberSaveable { mutableStateOf(true) }
        var showPerformanceTelemetry by rememberSaveable { mutableStateOf(true) }

        val visualBudgetEngine =
            remember {
                StudioVisualBudgetEngine()
            }
        val visualBudget =
            visualBudgetEngine.evaluate(
                StudioVisualBudgetEngine.Input(
                    gpuLatencyMs = gpuLatencyMs,
                    sessionHealth = sessionHealthScore,
                    captureIntegrity = captureIntegrityScore,
                ),
            )
        val glassEffectScale = visualBudget.effectScale

        Column(Modifier.fillMaxSize()) {
            PhysicalPrecisionAuthorityPanel(
                decision = precisionCapability,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 6.dp,
                    ),
            )

            CrowdIntelligencePanel(
                meanIdentitySafety = crowdMotionMeanIdentitySafety,
                interactionPressure = crowdMotionInteractionPressure,
                differentialMotionScore = crowdMotionDifferentialScore,
                schedulingCapacity = crowdMotionSchedulingCapacity,
                worldLandmarkCoverage = crowdWorldLandmarkCoverage,
                worldBoneConsistency = crowdWorldBoneConsistency,
                worldGeometryReliability = crowdWorldGeometryReliability,
                crowdAppearanceQuality = crowdAppearanceQuality,
                segmentationEvidence = crowdSegmentationEvidence,
                segmentationCoverage = crowdSegmentationCoverage,
                crowdActorSilhouetteSupport =
                    crowdActorSilhouetteSupport,
                crowdActorSilhouetteAmbiguity =
                    crowdActorSilhouetteAmbiguity,
                personSegmentationConfidence =
                    personSegmentationConfidence,
                personSegmentationBoundaryStability =
                    personSegmentationBoundaryStability,
                personSegmentationLatencyMs =
                    personSegmentationLatencyMs,
                personSegmentationState =
                    personSegmentationState,
                personSegmentationDelegate =
                    personSegmentationDelegate,
                personSegmentationFallbackCount =
                    personSegmentationFallbackCount,
                bottleneck = crowdMotionBottleneck,
                heldActors = crowdMotionHeldCount,
                predictedActors = crowdMotionPredictedCount,
                warnings = crowdMotionWarnings,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 4.dp,
                    ),
            )

            PrecisionEvidenceSurface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = outerPadding, vertical = 4.dp),
                calibrationReady = calibrationReady,
                synchronized = multiCameraSyncConfidence >= 0.92,
                syncConfidence = multiCameraSyncConfidence,
                reprojectionP95Pixels =
                    reconstructionErrorPx
                        .takeIf(Double::isFinite)
                        ?: Double.POSITIVE_INFINITY,
                reconstructionConfidence = reconstructionQuality,
                reconstructionUncertaintyP95Meters =
                    reconstructionUncertaintyP95Meters,
                geometricObservability = geometricObservabilityScore,
                denseGeometricConditioningScore =
                    denseGeometricConditioningScore,
                denseGeometricConditioningMedianParallaxDegrees =
                    denseGeometricConditioningMedianParallaxDegrees,
                denseGeometricConditioningWarnings =
                    denseGeometricConditioningWarnings,
                denseLandmarkCoverage = denseLandmarkCoverage,
                denseLandmarkCrossViewAgreement =
                    denseLandmarkCrossViewAgreement,
                temporalStability = temporalStability,
                worldConsistency = worldConsistencyScore,
                identityContinuity = multiPersonIdentityContinuity,
                identitySwapRisk =
                    (1.0 - multiPersonIdentityContinuity)
                        .coerceIn(0.0, 1.0),
                faceEyeConfidence = faceEyeMetricConfidence,
                handConfidence =
                    handTemporalConsistencyConfidence
                        .coerceIn(0.0, 1.0),
                cameraCount = concurrentRigSelectedCameraIds
                    .size
                    .coerceAtLeast(if (calibrationReady) 1 else 0),
                predictedObservationRatio =
                    multiPerson3DPredictedCount.toDouble() /
                        max(
                            1,
                            multiPerson3DReconstructedCount +
                                multiPerson3DPredictedCount,
                        ).toDouble(),
                denseSurfaceTrustedPointRatio =
                    denseSurfaceTrustedPointRatio,
                denseSurfaceTopologyStability =
                    denseSurfaceTopologyStability,
                denseSurfaceViewDiversity =
                    denseSurfaceViewDiversity,
                captureIntegrity = captureIntegrityScore,
                sessionHealth = sessionHealthScore,
                selectedActorCount = multiPerson3DReconstructedCount,
                targetActorCount = 50,
            )
            LiquidGlassSurface(
                clear = true,
                effectScale = glassEffectScale,
                modifier = Modifier.fillMaxWidth(),
            ) {
            MotionStudioOperatorDeck(
                    telemetry = StudioTelemetryState(
                        camerasConnected = null,
                        camerasExpected = null,
                        synchronizationConfidence = multiCameraSyncConfidence,
                        calibrationReady = calibrationReady,
                        trackingConfidence = poseConfidence.toDouble(),
                        reconstructionConfidence = reconstructionQuality.coerceIn(0.0, 1.0),
                        motionTemporalMemoryScore = motionTemporalMemoryScore,
                        droppedObservations = null,
                        thermalConstrained = null,
                        denseLandmarkCoverage = denseLandmarkCoverage,
                        denseLandmarkVisibleCoverage =
                            denseLandmarkVisibleCoverage,
                        denseLandmarkUncertaintyP95 =
                            denseLandmarkUncertaintyP95,
                        denseLandmarkCrossViewAgreement =
                            denseLandmarkCrossViewAgreement,
                        denseLandmarkCrossViewEvidenceAvailable =
                            denseLandmarkCrossViewEvidenceAvailable,
                        denseLandmarkTemporalStability =
                            denseLandmarkTemporalStability,
                        denseLandmarkContactCoverage =
                            denseLandmarkContactCoverage,
                        denseLandmarkPredictionBridgeRatio =
                            denseLandmarkPredictionBridgeRatio,
                        denseLandmarkRejectedObservations =
                            denseLandmarkRejectedObservations,
                        denseLandmarkStaleObservations =
                            denseLandmarkStaleObservations,
                        denseLandmarkNonFiniteObservations =
                            denseLandmarkNonFiniteObservations,
                        denseLandmarkHardFailure =
                            denseLandmarkHardFailure,
                        denseLandmarkContractScore =
                            denseLandmarkContractScore,
                        denseLandmarkContractState =
                            denseLandmarkContractState,
                        denseLandmarkContractAuthoritative =
                            denseLandmarkContractAuthoritative,
                        denseLandmarkContractFailure =
                            denseLandmarkContractFailure,
                        denseSurfaceTrustedPointRatio =
                            denseSurfaceTrustedPointRatio,
                        denseSurfaceUncertaintyP95Meters =
                            denseSurfaceUncertaintyP95Meters,
                        denseSurfaceTopologyStability =
                            denseSurfaceTopologyStability,
                        denseSurfaceViewDiversity =
                            denseSurfaceViewDiversity,
                        denseSurfaceProbabilisticEvidenceScore =
                            denseSurfaceProbabilisticEvidenceScore,
                        denseSurfaceAdaptiveViewSelectionScore =
                            denseSurfaceAdaptiveViewSelectionScore,
                        denseSurfaceAdaptiveViewDiversityScore =
                            denseSurfaceAdaptiveViewDiversityScore,
                        denseSurfaceAnisotropicWeakAxisConstraint =
                            denseSurfaceAnisotropicWeakAxisConstraint,
                        denseSurfaceAnisotropicViewConditionScore =
                            denseSurfaceAnisotropicViewConditionScore,
                        denseSurfaceExcludedWeakViewCount =
                            denseSurfaceExcludedWeakViewCount,
                        denseSurfaceMeanReprojectionErrorPx =
                            denseSurfaceMeanReprojectionErrorPx,
                        denseSurfaceMeanAppearanceConsistency =
                            denseSurfaceMeanAppearanceConsistency,
                        denseSurfaceMeanSilhouetteConsistency =
                            denseSurfaceMeanSilhouetteConsistency,
                        denseSurfaceMeanMotionBoundaryConfidence =
                            denseSurfaceMeanMotionBoundaryConfidence,
                        denseSurfaceMeanLightingStability =
                            denseSurfaceMeanLightingStability,
                        denseSurfaceMeanDynamicProbability =
                            denseSurfaceMeanDynamicProbability,
                        denseSurfaceMeanTemporalStability =
                            denseSurfaceMeanTemporalStability,
                        denseSurfaceTrustedGaussianRatio =
                            denseSurfaceTrustedGaussianRatio,
                        denseSurfaceHardFailure =
                            denseSurfaceHardFailure,
                        denseMeanJerkMetersPerSecond3 =
                            denseMeanJerkMetersPerSecond3
                                .takeIf(Double::isFinite),
                        denseMotionStability =
                            denseMotionStability
                                .takeIf(Double::isFinite),
                        denseKinematicInnovationMillimeters =
                            denseKinematicInnovationMillimeters
                                .takeIf(Double::isFinite),
                        denseSubMillimeterFeasibilityScore =
                            denseSubMillimeterFeasibilityScore
                                .takeIf(Double::isFinite),
                        denseFirstOrderDepthErrorMillimeters =
                            denseFirstOrderDepthErrorMillimeters
                                .takeIf(Double::isFinite),
                        densePrecisionQualified =
                            densePrecisionQualified,
                        cameraContributionMinimumPairwiseDiversity =
                            cameraContributionMinimumPairwiseDiversity,
                        cameraContributionViewStarvationRisk =
                            cameraContributionViewStarvationRisk,
                        correlatedEvidenceScore =
                            correlatedEvidenceScore,
                        correlatedEvidenceCoverage =
                            correlatedEvidenceCoverage,
                        correlatedEvidenceFailureRisk =
                            correlatedEvidenceFailureRisk,
                        correlatedEvidenceCommonModeRisk =
                            correlatedEvidenceCommonModeRisk,
                        correlatedEvidenceRedundantSignalCount =
                            correlatedEvidenceRedundantSignalCount,
                        correlatedEvidenceWarnings =
                            correlatedEvidenceWarnings,
                        identityExchangeState =
                            identityExchangeState,
                        identityExchangeRisk =
                            identityExchangeRisk,
                        identityExchangeConfidence =
                            identityExchangeConfidence,
                        identityExchangeEvaluatedSubjects =
                            identityExchangeEvaluatedSubjects,
                        identityExchangeHeldSubjects =
                            identityExchangeHeldSubjects,
                        identityExchangeWarnings =
                            identityExchangeWarnings,
                        global4DGaussianConsistency =
                            global4DGaussianConsistency,
                        global4DGaussianStrainConsistency =
                            global4DGaussianStrainConsistency,
                        global4DGaussianEvidenceAvailable =
                            global4DGaussianEvidenceAvailable,
                        exportReady = captureIntegrityOutputAllowed,
                        nativeEngineHealthy = null,
                        reconstructionUncertaintyP95Meters = reconstructionUncertaintyP95Meters,
                        geometricInformationConditionScore =
                            geometricInformationConditionScore,
                        geometricInformationP95MajorMeters =
                            geometricInformationP95MajorMeters,
                        geometricInformationP95DepthMeters =
                            geometricInformationP95DepthMeters,
                         cameraOpticalBlurRisk =
                             cameraOpticalBlurRisk
                                 .coerceIn(0.0, 1.0),
                        opticalBlurRisk =
                            cameraOpticalBlurRisk
                                .coerceIn(0.0, 1.0),
                        captureAuthorityScore =
                            captureAuthorityScore.coerceIn(0.0, 1.0),
                        captureAuthorityBand =
                            captureAuthorityBand,
                        captureAuthorityHardBlocked =
                            captureAuthorityHardBlocked,
                        captureAuthorityPrecisionQualified =
                            captureAuthorityPrecisionQualified,
                        captureAuthorityWarnings =
                            captureAuthorityWarnings,
                        cameraSessionVerification =
                            cameraSessionVerification,
                        cameraSessionSupported =
                            cameraSessionSupported,
                        highSpeedSessionVerification =
                            highSpeedSessionVerification,
                        highSpeedSessionSupported =
                            highSpeedSessionSupported,
                        highSpeedSessionActive =
                            highSpeedSessionActive,
                        highSpeedSelectedFps =
                            highSpeedSelectedFps,
                        highSpeedSessionReason =
                            highSpeedSessionReason,
                        hardwareReadoutTimestampSupported =
                            hardwareReadoutTimestampSupported,
                        hardwareReadoutTimingState =
                            hardwareReadoutTimingState,
                        hardwareReadoutTimingConfidence =
                            hardwareReadoutTimingConfidence,
                        hardwareReadoutP95JitterMillis =
                            hardwareReadoutP95JitterMillis,
                        hardwareReadoutMedianDelayMillis =
                            hardwareReadoutMedianDelayMillis,
                        latestReadoutTimestampNanos =
                            latestReadoutTimestampNanos,
                        concurrentRigAdmissionState =
                            concurrentRigAdmissionState,
                        concurrentRigAdmissionScore =
                            concurrentRigAdmissionScore,
                        concurrentRigSelectedCameraIds =
                            concurrentRigSelectedCameraIds,
                        concurrentRigBlockers =
                            concurrentRigBlockers,
                        concurrentRigWarnings =
                            concurrentRigWarnings,
                        thermalExtrinsicDriftState =
                            thermalExtrinsicDriftState,
                        thermalExtrinsicDriftConfidence =
                            thermalExtrinsicDriftConfidence,
                        thermalExtrinsicTranslationDriftMm =
                            thermalExtrinsicTranslationDriftMm,
                        thermalExtrinsicRotationDriftDegrees =
                            thermalExtrinsicRotationDriftDegrees,
                        thermalExtrinsicTemperatureDeltaCelsius =
                            thermalExtrinsicTemperatureDeltaCelsius,
                        thermalExtrinsicHardFailure =
                            thermalExtrinsicHardFailure,
                        thermalExtrinsicWarnings =
                            thermalExtrinsicWarnings,
                        faceEyeMetricState =
                            faceEyeMetricState,
                        faceEyeMetricConfidence =
                            faceEyeMetricConfidence,
                        faceEyeMetricEyeUncertaintyMm =
                            faceEyeMetricEyeUncertaintyMm,
                        faceEyeMetricWarnings =
                            faceEyeMetricWarnings,
                        faceEyeMetricRecommendations =
                            faceEyeMetricRecommendations,
                        multiPersonIdentityContinuity =
                            multiPersonIdentityContinuity,
                        multiPersonIdentityReidentifications =
                            multiPersonIdentityReidentifications,
                        multiPersonIdentityPredictions =
                            multiPersonIdentityPredictions,
                        multiPersonObservedCount =
                            multiPersonObservedCount,
                        multiPersonAdmittedCount =
                            multiPersonAdmittedCount,
                        multiPersonHeldCount =
                            multiPersonHeldCount,
                        multiPersonStarvationRisk =
                            multiPersonStarvationRisk,
                        multiPersonFairnessBoostedCount =
                            multiPersonFairnessBoostedCount,
                        multiPerson3DReconstructedCount =
                            multiPerson3DReconstructedCount,
                        multiPerson3DDetailedCount =
                            multiPerson3DDetailedCount,
                        multiPerson3DPredictedCount =
                            multiPerson3DPredictedCount,
                        multiPerson3DPredictionMaxAgeMillis =
                            multiPerson3DPredictionMaxAgeMillis,
                    ),
                    modifier = Modifier.fillMaxWidth(),
                )
    
                }

            if (showPerformanceTelemetry) {
                ReferenceGradeEvidenceCard(
                    modeledP95Millimeters =
                        reconstructionUncertaintyP95Meters
                            .takeIf { it.isFinite() }
                            ?.times(1000.0),
                    synchronizationConfidence =
                        multiCameraSyncConfidence,
                    viewDiversity =
                        geometricObservabilityScore,
                    trackingConfidence =
                        poseConfidence.toDouble(),
                    cameraObservability =
                        geometricObservabilityScore,
                    occlusionRisk = occlusionRisk,
                    temporalConsistency = temporalStability,
                    trustedJointRatio = reconstructionTrustedJointRatio,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(
                            horizontal = outerPadding,
                            vertical = 6.dp,
                        ),
                )
            }

            HighEndOperatorRail(
                state = state,
                poseConfidence = poseConfidence.toDouble(),
                gazeConfidence = gazeConfidence,
                handConfidence = handTemporalConsistencyConfidence,
                geometricObservability = geometricObservabilityScore,
                reconstructionQuality = reconstructionQuality,
                sensorQuality = cameraCaptureQualityScore,
                multiViewEvidence = adaptiveViewEvidenceScore,
                integrityScore = captureIntegrityScore,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 4.dp,
                    ),
            )

            HybridCaptureModePanel(
                settings = motionCustomization,
                onSettingsChanged = { updated ->
                     onAdvancedMotionCustomization { updated }
                 },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 4.dp,
                    ),
            )

            PostCaptureReconstructionStatusPanel(
                state = postCaptureState,
                progress = postCaptureProgress,
                processedFrames = postCaptureProcessedFrames,
                maxActors = postCaptureMaxActors,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 4.dp,
                    ),
            )

            ActorDetailBudgetPanel(
                settings = motionCustomization,
                onSettingsChanged = { updated ->
                     onAdvancedMotionCustomization { updated }
                 },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 4.dp,
                    ),
            )

            LiquidGlassControlLayer(
                state = state,
                motionProfile = motionProfile,
                dominantSubject = dominantSubject,
                calibrationReady = calibrationReady,
                sessionHealthScore = sessionHealthScore,
                captureIntegrityScore = captureIntegrityScore,
                captureIntegrityBand = captureIntegrityBand,
                captureIntegrityOutputAllowed = captureIntegrityOutputAllowed,
                captureIntegrityBottleneck = captureIntegrityBottleneck,
                synchronizationConfidence = multiCameraSyncConfidence,
                onSettingsClick = { settingsOpen = true },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = 14.dp,
                        vertical = 10.dp,
                    ),
            )

            PrecisionWorkflowShelf(
                selectedSystemPreset =
                    when {
                        motionCustomization.inferenceFps >= 50 &&
                            motionCustomization.bodyDetail >= 0.95 ->
                            "QUALITY"
                        motionCustomization.inferenceFps <= 30 &&
                            motionCustomization.bodyDetail < 0.90 ->
                            "LOW_LATENCY"
                        else ->
                            "BALANCED"
                    },
                synchronization = multiCameraSyncConfidence,
                geometricObservability = geometricObservabilityScore,
                densePrecision =
                    when (densePrecisionQualified) {
                        true -> 1.0
                        false -> 0.0
                        null ->
                            denseSubMillimeterFeasibilityScore
                                .takeIf(Double::isFinite)
                    },
                healthScore = sessionHealthScore,
                timingHealthy = captureTimingHealthy,
                cameraDiversity =
                    cameraContributionMinimumPairwiseDiversity,
                cameraStarvationRisk =
                    cameraContributionViewStarvationRisk,
                correlatedEvidence =
                    correlatedEvidenceScore,
                correlatedEvidenceState =
                    correlatedEvidenceState,
                identityExchangeRisk =
                    identityExchangeRisk,
                identityExchangeState =
                    identityExchangeState,
                effectScale = glassEffectScale,
                onPreset = onCustomizationPreset,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 4.dp,
                    ),
            )

            if (showEvidenceMatrix) {
            GlassSignalMatrix(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = outerPadding,
                        vertical = 2.dp,
                    ),
                cells = listOf(
                    GlassSignalCell(
                        label = "Body",
                        value = motionClass.replace('_', ' '),
                        score = poseConfidence.toDouble(),
                        detail = "pose evidence",
                    ),
                    GlassSignalCell(
                        label = "Motion",
                        value = (cameraCompensatedMotionScore * 100.0).toInt().toString() + "%",
                        score = cameraCompensatedMotionScore,
                        detail =
                            "common " +
                                (commonModeMotionNormalizedPerSecond * 100.0).toInt() +
                                " · diff " +
                                (differentialMotionScore * 100.0).toInt(),
                    ),
                    GlassSignalCell(
                        label = "Face",
                        value = "3D ${(face3DTemporalConsistencyConfidence * 100.0).toInt()}%",
                        score = faceEyeConsistencyConfidence,
                        detail = "landmark stability",
                    ),
                    GlassSignalCell(
                        label = "Eyes",
                        value = "Gaze ${(gazeStability * 100.0).toInt()}%",
                        score = gazeConfidence,
                        detail = "gaze + temporal fusion",
                    ),
                    GlassSignalCell(
                        label = "Hands",
                        value = "Temporal ${(handTemporalConsistencyConfidence * 100.0).toInt()}%",
                        score = handTemporalConsistencyConfidence,
                        detail = "articulation continuity",
                    ),
                    GlassSignalCell(
                        label = "Depth",
                        value = "Obs ${(geometricObservabilityScore * 100.0).toInt()}%",
                        score = geometricObservabilityScore,
                        detail = "multi-view geometry",
                    ),
                    GlassSignalCell(
                        label = "Precision",
                        value =
                            when (densePrecisionQualified) {
                                true -> "QUALIFIED"
                                false -> "HOLD"
                                null ->
                                    if (denseSubMillimeterFeasibilityScore.isFinite()) {
                                        (denseSubMillimeterFeasibilityScore * 100.0)
                                            .coerceIn(0.0, 100.0)
                                            .toInt()
                                            .toString() + "%"
                                    } else {
                                        "PENDING"
                                    }
                            },
                        score =
                            when (densePrecisionQualified) {
                                true -> 1.0
                                false -> 0.0
                                null ->
                                    if (denseSubMillimeterFeasibilityScore.isFinite()) {
                                        denseSubMillimeterFeasibilityScore
                                    } else {
                                        0.5
                                    }
                            },
                        detail =
                            if (denseFirstOrderDepthErrorMillimeters.isFinite()) {
                                "1st-order depth " +
                                    "%.2f".format(
                                        denseFirstOrderDepthErrorMillimeters,
                                    ) +
                                    " mm"
                            } else {
                                "geometry evidence pending"
                            },
                    ),
                    GlassSignalCell(
                        label = "3D",
                        value = "Q ${(reconstructionQuality * 100.0).toInt()}%",
                        score = reconstructionQuality,
                        detail = "${reconstructionJoints} joints",
                    ),
                    GlassSignalCell(
                        label = "Kinematic",
                        value =
                            if (kinematicConstraintAccepted) {
                                (kinematicConstraintScore * 100.0).toInt().toString() + "%"
                            } else {
                                "HOLD"
                            },
                        score = kinematicConstraintScore,
                        detail =
                            if (kinematicConstraintBottleneck == "NONE") {
                                "skeletal constraints"
                            } else {
                                kinematicConstraintBottleneck.replace('_', ' ')
                            },
                    ),
                    GlassSignalCell(
                        label = "Surface",
                        value =
                            (denseMeshSurfaceConfidence * 100.0)
                                .toInt()
                                .toString() + "%",
                        score = denseMeshSurfaceConfidence,
                        detail = "dense mesh confidence",
                    ),
                    GlassSignalCell(
                        label = "Sync",
                        value = "${(multiCameraSyncConfidence * 100.0).toInt()}%",
                        score = multiCameraSyncConfidence,
                        detail = "multi-camera clock",
                    ),
                    GlassSignalCell(
                        label = "Sensor",
                        value = "${(cameraCaptureQualityScore * 100.0).toInt()}%",
                        score =
                            (
                                cameraCaptureQualityScore *
                                    (1.0 - cameraAnalysisDropRatio.coerceIn(0.0, 1.0))
                            ).coerceIn(0.0, 1.0),
                        detail =
                            if (cameraAnalysisDropRatio > 0.0) {
                                "analysis drops " +
                                    String.format(
                                        "%.1f%%",
                                        cameraAnalysisDropRatio * 100.0,
                                    )
                            } else {
                                cameraCaptureQualityWarnings.firstOrNull()
                                    ?.replace('-', ' ')
                                    ?.take(30)
                                    ?: "timing + exposure + rolling shutter"
                            },
                    ),
                    GlassSignalCell(
                        label = "Integrity",
                        value = "${(captureIntegrityScore * 100.0).toInt()}%",
                        score = captureIntegrityScore,
                        detail = captureIntegrityBand.replace('_', ' '),
                    ),
                    GlassSignalCell(
                        label = "Trend",
                        value = captureEvidenceTrend,
                        score = captureEvidenceCurrentScore,
                        detail =
                            captureEvidenceBottleneck
                                .takeIf { it != "NONE" }
                                ?.replace('_', ' ')
                                ?: "rolling evidence",
                    ),
                    GlassSignalCell(
                        label = "Quarantine",
                        value =
                            rigQuarantinedCameraCount.toString(),
                        score =
                            if (rigQuarantinedCameraCount == 0) {
                                1.0
                            } else {
                                0.35
                            },
                        detail =
                            if (rigQuarantinedCameraCount == 0) {
                                "all active"
                            } else {
                                "camera(s) held out"
                            },
                    ),
                    GlassSignalCell(
                        label = "Views",
                        value = (adaptiveViewEvidenceScore * 100.0).toInt().toString() + "%",
                        score = adaptiveViewEvidenceScore,
                        detail =
                            when {
                                adaptiveViewEvidenceHoldCameraCount > 0 ->
                                    "HOLD " + adaptiveViewEvidenceHoldCameraCount
                                adaptiveViewEvidenceDegradedCameraCount > 0 ->
                                    "degraded " + adaptiveViewEvidenceDegradedCameraCount
                                adaptiveViewEvidenceDominantCamera != "NONE" ->
                                    "weak " + adaptiveViewEvidenceDominantCamera.takeLast(8)
                                else -> "camera evidence"
                            },
                    ),
                    GlassSignalCell(
                        label = "Timing",
                        value =
                            (spatiotemporalConfidence * 100.0)
                                .toInt()
                                .toString() + "%",
                        score = spatiotemporalConfidence,
                        detail =
                            if (
                                spatiotemporalInitialRmsPx > 0.0 &&
                                spatiotemporalFinalRmsPx > 0.0
                            ) {
                                "RMS %.2f → %.2f px".format(
                                    spatiotemporalInitialRmsPx,
                                    spatiotemporalFinalRmsPx,
                                )
                            } else {
                                spatiotemporalWarnings.firstOrNull()
                                    ?.replace('-', ' ')
                                    ?: "temporal bundle"
                            },
                    ),
                ),
            )
            }


            Box(
                Modifier
                    .fillMaxWidth()
                    .weight(1f),
            ) {
                if (expanded) {
            Row(
                Modifier.fillMaxSize().padding(outerPadding),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                StudioStage(
                    Modifier.weight(1.45f).fillMaxHeight(),
                    state, frameCount, skeleton, reconstructionQuality,
                    worldConsistencyScore, worldConsistencyWarnings,
                    gazeConfidence,
                    eye3DTemporalConsistencyConfidence,
                    eye3DTemporalConsistencyWarnings,
                    faceEyeConsistencyConfidence,
                    faceEyeConsistencyWarnings,
                    face3DTemporalConsistencyConfidence,
                    face3DTemporalConsistencyWarnings,
                    gazeStability, faceHeadYaw, motionEventType,
                    motionEventConfidence, faceHeadRoll,
                    handTemporalConsistencyConfidence,
                    handTemporalConsistencyWarnings,
                    subjectLock, subjectLockConfidence, dominantSubject,
                    motionClass, motionScore,
                     motionMeasurementUncertainty,
                    motionEvidenceScore, motionEvidenceConfidence,
                    motionEvidenceWarnings,
                    temporalStability,
                    occlusionRisk, calibrationReady,
                    captureTimingConfidence, captureTimingFps,
                    captureTimingJitterMillis, captureTimingHealthy,
                    sessionHealthState, sessionHealthScore,
                    sessionHealthBottleneck,
                    multiCameraSyncConfidence, multiCameraSyncState,
                    geometricObservabilityScore,
                    geometricObservabilityWarnings,
                    reconstructionUncertaintyConfidence,
                    reconstructionUncertaintyMeanMeters,
                    reconstructionUncertaintyP95Meters,
                    reconstructionTrustedJointRatio,
                    reconstructionUncertaintyBottleneck,
                    reconstructionUncertaintyWarnings,
                    captureReadiness,
                     rigCoverageScore,
                    denseMeshTemporalConsistencyConfidence,
                    denseMeshTemporalConsistencyWarnings,
                    sessionPresentationSettings,
                    recordingCountdown,
                    onCaptureToggle,
                    showDiagnostics,
                    showViewportGrid,
                )
                StudioOperatorPanel(
                    modifier = Modifier.weight(0.75f).fillMaxHeight(),
                    inferenceFps = inferenceFps,
                    poseConfidence = poseConfidence,
                    reconstructionErrorPx = reconstructionErrorPx,
                    captureIntegrityScore = captureIntegrityScore,
                    captureIntegrityBand = captureIntegrityBand,
                    captureIntegrityOutputAllowed = captureIntegrityOutputAllowed,
                    captureIntegrityBottleneck = captureIntegrityBottleneck,
                    reconstructionJoints = reconstructionJoints,
                    gpuBackend = gpuBackend,
                    gpuLatencyMs = gpuLatencyMs,
                    rigScore = rigScore,
                    lightingLux = lightingLux,
                    colorTemperature = colorTemperature,
                    sessionHealthScore = sessionHealthScore,
                    sessionHealthState = sessionHealthState,
                    sessionHealthBottleneck = sessionHealthBottleneck,
                    synchronizationConfidence = multiCameraSyncConfidence,
                    calibrationReady = calibrationReady,
                    showPerformanceTelemetry = showPerformanceTelemetry,
                )
            }
        } else {
            Column(
                Modifier.fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(outerPadding),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                StudioStage(
                    Modifier.fillMaxWidth().height(stageHeight),
                    state, frameCount, skeleton, reconstructionQuality,
                    worldConsistencyScore, worldConsistencyWarnings,
                    gazeConfidence,
                    eye3DTemporalConsistencyConfidence,
                    eye3DTemporalConsistencyWarnings,
                    faceEyeConsistencyConfidence,
                    faceEyeConsistencyWarnings,
                    face3DTemporalConsistencyConfidence,
                    face3DTemporalConsistencyWarnings,
                    gazeStability, faceHeadYaw, motionEventType,
                    motionEventConfidence, faceHeadRoll,
                    handTemporalConsistencyConfidence,
                    handTemporalConsistencyWarnings,
                    subjectLock, subjectLockConfidence, dominantSubject,
                    motionClass, motionScore,
                     motionMeasurementUncertainty,
                    motionEvidenceScore, motionEvidenceConfidence,
                    motionEvidenceWarnings,
                    temporalStability,
                    occlusionRisk, calibrationReady,
                    captureTimingConfidence, captureTimingFps,
                    captureTimingJitterMillis, captureTimingHealthy,
                    sessionHealthState, sessionHealthScore,
                    sessionHealthBottleneck,
                    multiCameraSyncConfidence, multiCameraSyncState,
                    geometricObservabilityScore,
                    geometricObservabilityWarnings,
                    reconstructionUncertaintyConfidence,
                    reconstructionUncertaintyMeanMeters,
                    reconstructionUncertaintyP95Meters,
                    reconstructionTrustedJointRatio,
                    reconstructionUncertaintyBottleneck,
                    reconstructionUncertaintyWarnings,
                    captureReadiness,
                     rigCoverageScore,
                     denseMeshTemporalConsistencyConfidence,
                     denseMeshTemporalConsistencyWarnings,
                     sessionPresentationSettings,
                     recordingCountdown,
                     onCaptureToggle,
                     showDiagnostics,
                     showViewportGrid,
                     microMotionScore,
                     directionChangeRatio,
                     temporalAlignmentConfidence,
                     temporalAlignmentSpreadMillis,
                     temporalAlignmentState,
                     temporalAlignmentWarnings,
                     adaptiveViewEvidenceWarnings,
                 )
                StudioOperatorPanel(
                    modifier = Modifier.fillMaxWidth(),
                    inferenceFps = inferenceFps,
                    poseConfidence = poseConfidence,
                    reconstructionErrorPx = reconstructionErrorPx,
                    captureIntegrityScore = captureIntegrityScore,
                    captureIntegrityBand = captureIntegrityBand,
                    captureIntegrityOutputAllowed = captureIntegrityOutputAllowed,
                    captureIntegrityBottleneck = captureIntegrityBottleneck,
                    reconstructionJoints = reconstructionJoints,
                    gpuBackend = gpuBackend,
                    gpuLatencyMs = gpuLatencyMs,
                    rigScore = rigScore,
                    lightingLux = lightingLux,
                    colorTemperature = colorTemperature,
                    sessionHealthScore = sessionHealthScore,
                    sessionHealthState = sessionHealthState,
                    sessionHealthBottleneck = sessionHealthBottleneck,
                    synchronizationConfidence = multiCameraSyncConfidence,
                    calibrationReady = calibrationReady,
                    showPerformanceTelemetry = showPerformanceTelemetry,
                )
                }
            }
        }
    }

        if (settingsOpen) {
            androidx.compose.foundation.layout.Box(
                Modifier
                    .fillMaxSize()
                    .background(
                        MaterialTheme.colorScheme.scrim.copy(alpha = 0.72f),
                    )
                    .padding(10.dp),
            ) {
                ProfessionalMotionSettings(
                    modifier = Modifier.fillMaxSize(),
                    motionCustomization = motionCustomization,
                    sessionPresentationSettings = sessionPresentationSettings,
                    motionProfile = motionProfile,
                    motionResponsiveness = motionResponsiveness,
                    motionSensitivity = motionSensitivity,
                    predictionHorizon = predictionHorizon,
                    occlusionPersistence = occlusionPersistence,
                    jitterSuppression = jitterSuppression,
                    reacquisitionRadius = reacquisitionRadius,
                    rejectAggressiveness = rejectAggressiveness,
                    velocityResponsiveness = velocityResponsiveness,
                    accelerationResponsiveness = accelerationResponsiveness,
                    innovationGateElasticity = innovationGateElasticity,
                    motionThresholdScale = motionThresholdScale,
                    adaptiveRegimeStrength = adaptiveRegimeStrength,
                    reconstructionProfile = reconstructionProfile,
                    reconstructionSoftAccept = reconstructionSoftAccept,
                    reconstructionHoldFrames = reconstructionHoldFrames,
                    reconstructionHoldAgeMillis = reconstructionHoldAgeMillis,
                    reconstructionStabilitySensitivity = reconstructionStabilitySensitivity,
                    temporalPredictionHorizon = temporalPredictionHorizon,
                    temporalInnovationGateScale = temporalInnovationGateScale,
                    temporalUncertaintyGateScale = temporalUncertaintyGateScale,
                    temporalReprojectionGateScale = temporalReprojectionGateScale,
                    temporalSmoothingResponsiveness = temporalSmoothingResponsiveness,
                    temporalConfidenceDecayScale = temporalConfidenceDecayScale,
                    temporalOutlierRetention = temporalOutlierRetention,
                    temporalTopologyToleranceScale = temporalTopologyToleranceScale,
                    showEvidenceMatrix = showEvidenceMatrix,
                    showDiagnostics = showDiagnostics,
                    showViewportGrid = showViewportGrid,
                    showPerformanceTelemetry = showPerformanceTelemetry,
                    onShowEvidenceMatrix = { showEvidenceMatrix = it },
                    onShowDiagnostics = { showDiagnostics = it },
                    onShowViewportGrid = { showViewportGrid = it },
                    onShowPerformanceTelemetry = { showPerformanceTelemetry = it },
                    onCustomizationPreset = onCustomizationPreset,
                    onCustomizationBodyDetail = onCustomizationBodyDetail,
                    onCustomizationFaceDetail = onCustomizationFaceDetail,
                    onCustomizationEyeDetail = onCustomizationEyeDetail,
                    onCustomizationEyeTemporalStrictness =
                        onCustomizationEyeTemporalStrictness,
                    onCustomizationHandDetail = onCustomizationHandDetail,
                    onCustomizationHandTemporalStrictness =
                        onCustomizationHandTemporalStrictness,
                    onCustomizationHandGeometryStrictness =
                        onCustomizationHandGeometryStrictness,
                    onCustomizationDepthDetail = onCustomizationDepthDetail,
                    onCustomizationDenseMeshDetail = onCustomizationDenseMeshDetail,
                    onCustomizationContactDetail = onCustomizationContactDetail,
                    onCustomizationWorldConsistencyDetail =
                        onCustomizationWorldConsistencyDetail,
                    onCustomizationTopologyConstraintStrength =
                        onCustomizationTopologyConstraintStrength,
                    onCustomizationMaxSubjects = onCustomizationMaxSubjects,
                    onCustomizationOcclusionPredictionFrames =
                        onCustomizationOcclusionPredictionFrames,
                    onCustomizationTargetFps = onCustomizationTargetFps,
                    onCustomizationInferenceFps = onCustomizationInferenceFps,
                    onCustomizationMicroMotionPreservation =
                        onCustomizationMicroMotionPreservation,
                    onCustomizationCameraMotionSuppression =
                        onCustomizationCameraMotionSuppression,
                    onCustomizationFaceEyeFusionWeight =
                        onCustomizationFaceEyeFusionWeight,
                    onCustomizationFaceEyeFusionStrictness =
                        onCustomizationFaceEyeFusionStrictness,
                    onCustomizationTemporalAlignmentStrictness =
                        onCustomizationTemporalAlignmentStrictness,
                    onCustomizationOutputIntegrityStrictness =
                        onCustomizationOutputIntegrityStrictness,
                    onCustomizationPhysicsConstraintStrictness =
                        onCustomizationPhysicsConstraintStrictness,
                    onCustomizationGlobal4DConsistencyStrictness =
                        onCustomizationGlobal4DConsistencyStrictness,
                    onCustomizationSurfaceEvidenceStrictness =
                        onCustomizationSurfaceEvidenceStrictness,
                    onCustomizationMinimumParallaxDegrees =
                        onCustomizationMinimumParallaxDegrees,
                    onCustomizationPreferredTemporalSkewMillis =
                        onCustomizationPreferredTemporalSkewMillis,
                    onCustomizationMaximumTemporalSkewMillis =
                        onCustomizationMaximumTemporalSkewMillis,
                    onCustomizationMaximumReprojectionErrorPx =
                        onCustomizationMaximumReprojectionErrorPx,
                    onCustomizationMaximumReconstructionUncertaintyMeters =
                        onCustomizationMaximumReconstructionUncertaintyMeters,
                    onAdvancedMotionCustomization =
                        onAdvancedMotionCustomization,
                    onSessionClipName = onSessionClipName,
                    onSessionCountdownSeconds = onSessionCountdownSeconds,
                    onRequireCalibrationBeforeRecord =
                        onRequireCalibrationBeforeRecord,
                    onMotionResponsiveness = onMotionResponsiveness,
                    onMotionSensitivity = onMotionSensitivity,
                    onPredictionHorizon = onPredictionHorizon,
                    onOcclusionPersistence = onOcclusionPersistence,
                    onJitterSuppression = onJitterSuppression,
                    onReacquisitionRadius = onReacquisitionRadius,
                    onRejectAggressiveness = onRejectAggressiveness,
                    onVelocityResponsiveness = onVelocityResponsiveness,
                    onAccelerationResponsiveness = onAccelerationResponsiveness,
                    onInnovationGateElasticity = onInnovationGateElasticity,
                    onMotionThresholdScale = onMotionThresholdScale,
                    onAdaptiveRegimeStrength = onAdaptiveRegimeStrength,
                    onReconstructionPreset = onReconstructionPreset,
                    onReconstructionSoftAccept = onReconstructionSoftAccept,
                    onReconstructionHoldFrames = onReconstructionHoldFrames,
                    onReconstructionHoldAgeMillis =
                        onReconstructionHoldAgeMillis,
                    onReconstructionStabilitySensitivity =
                        onReconstructionStabilitySensitivity,
                    onTemporalPredictionHorizon = onTemporalPredictionHorizon,
                    onTemporalInnovationGateScale =
                        onTemporalInnovationGateScale,
                    onTemporalUncertaintyGateScale =
                        onTemporalUncertaintyGateScale,
                    onTemporalReprojectionGateScale =
                        onTemporalReprojectionGateScale,
                    onTemporalSmoothingResponsiveness =
                        onTemporalSmoothingResponsiveness,
                    onTemporalConfidenceDecayScale =
                        onTemporalConfidenceDecayScale,
                    onTemporalOutlierRetention = onTemporalOutlierRetention,
                    onTemporalTopologyToleranceScale =
                        onTemporalTopologyToleranceScale,
                    onClose = { settingsOpen = false },
                )
            }
        }
    }
}

private enum class StudioWidthTier {
    COMPACT,
    MEDIUM,
    EXPANDED,
    LARGE,
}


@Composable
private fun CaptureControlRail(
    state: String,
    frameCount: Long,
    inferenceFps: Double,
    captureIntegrityScore: Double,
    onCaptureToggle: () -> Unit,
) {
    GlassCard(
        intensity = 0.90f,
        selected = state == "RECORDING",
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Column(
                Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                Text(
                    if (state == "RECORDING") "LIVE MOTION TAKE" else "READY TO CAPTURE",
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    frameCount.toString() + " FRAMES  ·  " +
                        (if (inferenceFps > 0.0) {
                            "%.1f".format(inferenceFps)
                        } else {
                            "--"
                        }) +
                        " FPS  ·  INTEGRITY " +
                        (captureIntegrityScore * 100.0).toInt() +
                        "%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    if (state == "RECORDING") {
                        "Motion evidence is being committed to the synchronized take."
                    } else {
                        "Arm the rig before capture. The control below starts the synchronized motion take."
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            NoxCaptureButton(
                recording = state == "RECORDING",
                enabled = state == "ARMED" || state == "RECORDING",
                onClick = onCaptureToggle,
            )
        }
    }
}

private fun studioWidthTier(
    width: androidx.compose.ui.unit.Dp,
): StudioWidthTier =
    when {
        width >= 1200.dp -> StudioWidthTier.LARGE
        width >= 840.dp -> StudioWidthTier.EXPANDED
        width >= 600.dp -> StudioWidthTier.MEDIUM
        else -> StudioWidthTier.COMPACT
    }

@Composable
private fun StudioOperatorPanel(
    modifier: Modifier,
    inferenceFps: Int,
    poseConfidence: Float,
    reconstructionErrorPx: Double,
    captureIntegrityScore: Double,
    captureIntegrityBand: String,
    captureIntegrityOutputAllowed: Boolean,
    captureIntegrityBottleneck: String,
    reconstructionJoints: Int,
    gpuBackend: String,
    gpuLatencyMs: Double,
    rigScore: Double,
    lightingLux: Double,
    colorTemperature: Double,
    sessionHealthScore: Double,
    sessionHealthState: String,
    sessionHealthBottleneck: String,
    synchronizationConfidence: Double,
    calibrationReady: Boolean,
    showPerformanceTelemetry: Boolean,
) {
    GlassCard(
        modifier = modifier,
        intensity = 0.92f,
    ) {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            SectionTitle(
                "OPERATOR STATUS",
                "Critical runtime evidence only. Deep tuning lives in Settings.",
            )

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Integrity",
                    (captureIntegrityScore * 100.0).toInt().toString() + "%",
                    captureIntegrityBand.replace('_', ' '),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Output",
                    if (captureIntegrityOutputAllowed) "TRUSTED" else "HELD",
                    captureIntegrityBottleneck.replace('_', ' '),
                    Modifier.weight(1f),
                )
            }

            if (showPerformanceTelemetry) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Inference",
                    inferenceFps.toString() + " fps",
                    "adaptive AI schedule",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Pose",
                    (poseConfidence * 100.0).toInt().toString() + "%",
                    reconstructionJoints.toString() + " joints",
                    Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Residual",
                    if (reconstructionErrorPx.isFinite()) {
                        "%.1f px".format(reconstructionErrorPx)
                    } else {
                        "—"
                    },
                    "3D reprojection",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "GPU",
                    "%.1f ms".format(gpuLatencyMs),
                    gpuBackend,
                    Modifier.weight(1f),
                )
            }
            }


            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Rig",
                    (rigScore * 100.0).toInt().toString() + "%",
                    "calibration quality",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Sync",
                    (synchronizationConfidence * 100.0).toInt().toString() + "%",
                    if (calibrationReady) "calibrated" else "relative",
                    Modifier.weight(1f),
                )
            }

            GlassMetric(
                "Lighting",
                lightingLux.toInt().toString() + " lx",
                colorTemperature.toInt().toString() + " K",
                Modifier.fillMaxWidth(),
            )

            GlassGauge(
                "Session health",
                sessionHealthScore.toFloat(),
                Modifier.fillMaxWidth(),
            )

            Text(
                sessionHealthState.replace('_', ' ') +
                    " · " +
                    sessionHealthBottleneck.replace('_', ' '),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            GlassCard(intensity = 0.48f) {
                Text(
                    "Use SETTINGS for capture, tracking, multi-view, reconstruction, temporal, kinematic, face/hand, surface and session controls.",
                    Modifier.padding(13.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun StudioStage(
    modifier: Modifier,
    state: String,
    frameCount: Long,
    skeleton: SkeletonFrame?,
    reconstructionQuality: Double,
    worldConsistencyScore: Double,
    worldConsistencyWarnings: List<String>,
    gazeConfidence: Double,
    eye3DTemporalConsistencyConfidence: Double,
    eye3DTemporalConsistencyWarnings: List<String>,
    faceEyeConsistencyConfidence: Double,
    faceEyeConsistencyWarnings: List<String>,
    face3DTemporalConsistencyConfidence: Double,
    face3DTemporalConsistencyWarnings: List<String>,
    gazeStability: Double,
    faceHeadYaw: Double,
    motionEventType: String,
    motionEventConfidence: Double,
    faceHeadRoll: Double,
    handTemporalConsistencyConfidence: Double,
    handTemporalConsistencyWarnings: List<String>,
    subjectLock: String,
    subjectLockConfidence: Double,
    dominantSubject: String,
    motionClass: String,
    motionScore: Double,
    motionMeasurementUncertainty: Double,
    motionEvidenceScore: Double,
    motionEvidenceConfidence: Double,
    motionEvidenceWarnings: List<String>,
    temporalStability: Double,
    occlusionRisk: Double,
    calibrationReady: Boolean,
    captureTimingConfidence: Double,
    captureTimingFps: Double,
    captureTimingJitterMillis: Double,
    captureTimingHealthy: Boolean,
    sessionHealthState: String,
    sessionHealthScore: Double,
    sessionHealthBottleneck: String,
    multiCameraSyncConfidence: Double,
    multiCameraSyncState: String,
    geometricObservabilityScore: Double,
    geometricObservabilityWarnings: List<String>,
    reconstructionUncertaintyConfidence: Double,
    reconstructionUncertaintyMeanMeters: Double,
    reconstructionUncertaintyP95Meters: Double,
    reconstructionTrustedJointRatio: Double,
    reconstructionUncertaintyBottleneck: String,
    reconstructionUncertaintyWarnings: List<String>,
    captureReadiness: com.nivukx.nox.core.model.CaptureReadiness?,
     rigCoverageScore: Double,
    denseMeshTemporalConsistencyConfidence: Double,
    denseMeshTemporalConsistencyWarnings: List<String>,
    sessionPresentationSettings: SessionPresentationSettings,
    recordingCountdown: Int,
    onCaptureToggle: () -> Unit,
    showDiagnostics: Boolean,
    showViewportGrid: Boolean,
    microMotionScore: Double = 0.0,
    directionChangeRatio: Double = 0.0,
    temporalAlignmentConfidence: Double = 0.0,
    temporalAlignmentSpreadMillis: Double = 0.0,
    temporalAlignmentState: String = "UNKNOWN",
    temporalAlignmentWarnings: List<String> = emptyList(),
    adaptiveViewEvidenceWarnings: List<String> = emptyList(),
) {
    GlassCard(
        modifier = modifier,
        selected = state == "RECORDING",
        intensity = 0.96f,
    ) {
        Column(
            Modifier.fillMaxSize().padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        sessionPresentationSettings.sanitizedClipName(),
                        style = MaterialTheme.typography.headlineSmall,
                    )
                    Text(
                        "RGB-only reconstruction stage",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                StatusPill(
                    text = if (state == "RECORDING") "CAPTURING MOTION" else state,
                    healthy = state != "ERROR",
                )
            }

            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                TelemetryChip("LOCK", subjectLock)
                TelemetryChip("CONF", "${(subjectLockConfidence * 100).toInt()}%")
                TelemetryChip("SUBJECT", dominantSubject)
                TelemetryChip("MOTION", motionClass)
                TelemetryChip("FRAMES", frameCount.toString())
                TelemetryChip(
                    "HEALTH",
                    sessionHealthState.replace('_', ' '),
                )
                TelemetryChip(
                    "SCORE",
                    (sessionHealthScore * 100.0).toInt().toString() + "%",
                )
                TelemetryChip(
                    "BOTTLENECK",
                    sessionHealthBottleneck.uppercase(),
                )
                TelemetryChip(
                    "SYNC",
                    (multiCameraSyncConfidence * 100.0).toInt().toString() + "%",
                )
                TelemetryChip("CAL", if (calibrationReady) "READY" else "RELATIVE")
                 TelemetryChip(
                     "COVERAGE",
                     (rigCoverageScore * 100.0).toInt().toString() + "%",
                 )
                TelemetryChip(
                    "TAKE",
                    sessionPresentationSettings.sanitizedClipName(),
                )
            }

            if (recordingCountdown > 0) {
                CountdownOverlay(recordingCountdown)
            }

            ReconstructionViewport(
                skeleton = skeleton,
                quality = reconstructionQuality,
                showGrid = showViewportGrid,
                modifier = Modifier.fillMaxWidth().weight(1f),
            )

            CaptureControlRail(
                state = state,
                frameCount = frameCount,
                inferenceFps = captureTimingFps,
                captureIntegrityScore = captureReadiness?.captureIntegrityScore
                    ?: reconstructionQuality,
                onCaptureToggle = onCaptureToggle,
            )


            MotionEventPanel(
                eventType = motionEventType,
                confidence = motionEventConfidence,
            )

            ReadinessPanel(captureReadiness)

            if (showDiagnostics) {
            if (worldConsistencyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "WORLD CONSISTENCY",
                    messages = worldConsistencyWarnings.take(3),
                )
            }

            if (faceEyeConsistencyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "FACE / EYE CONSISTENCY",
                    messages = faceEyeConsistencyWarnings.take(3),
                )
            }

            if (face3DTemporalConsistencyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "FACE 3D TEMPORAL GEOMETRY",
                    messages = face3DTemporalConsistencyWarnings.take(3),
                )
            }

            if (eye3DTemporalConsistencyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "EYE 3D TEMPORAL CONSISTENCY",
                    messages = eye3DTemporalConsistencyWarnings.take(3),
                )
            }

            if (handTemporalConsistencyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "HAND TEMPORAL CONSISTENCY",
                    messages = handTemporalConsistencyWarnings.take(3),
                )
            }

            if (denseMeshTemporalConsistencyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "DENSE SURFACE TEMPORAL",
                    messages = denseMeshTemporalConsistencyWarnings.take(3),
                )
            }

            if (motionEvidenceWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "MOTION EVIDENCE FUSION",
                    messages = motionEvidenceWarnings.take(3),
                )
            }

            if (geometricObservabilityWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "3D GEOMETRIC OBSERVABILITY",
                    messages = geometricObservabilityWarnings.take(3),
                )
            }

            if (reconstructionUncertaintyWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "3D METRIC UNCERTAINTY",
                    messages = reconstructionUncertaintyWarnings.take(3),
                )
            }

            if (temporalAlignmentWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "MULTI-CAMERA TEMPORAL ALIGNMENT",
                    messages = temporalAlignmentWarnings.take(3),
                )
            }
            if (adaptiveViewEvidenceWarnings.isNotEmpty()) {
                GlassDiagnosticsStrip(
                    title = "ADAPTIVE CAMERA EVIDENCE",
                    messages = adaptiveViewEvidenceWarnings.take(3),
                )
            }


            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                GlassTelemetryTile(
                    label = "3D certainty",
                    value =
                        (reconstructionUncertaintyConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        if (reconstructionUncertaintyWarnings.isEmpty()) {
                            "P95 " +
                                "%.1f".format(
                                    reconstructionUncertaintyP95Meters * 100.0,
                                ) +
                                " cm · " +
                                (reconstructionTrustedJointRatio * 100.0)
                                    .toInt()
                                    .toString() +
                                "% trusted"
                        } else {
                            reconstructionUncertaintyBottleneck
                        },
                    active =
                        reconstructionUncertaintyConfidence >= 0.55 &&
                            reconstructionTrustedJointRatio >= 0.62,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Dense surface",
                    value =
                        (denseMeshTemporalConsistencyConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        if (denseMeshTemporalConsistencyWarnings.isEmpty()) {
                            "surface stability"
                        } else {
                            denseMeshTemporalConsistencyWarnings.first().take(28)
                        },
                    active =
                        denseMeshTemporalConsistencyConfidence >= 0.62,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Face 3D continuity",
                    value =
                        (face3DTemporalConsistencyConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        if (face3DTemporalConsistencyWarnings.isEmpty()) {
                            "shape-space stable"
                        } else {
                            face3DTemporalConsistencyWarnings.first().take(28)
                        },
                    active =
                        face3DTemporalConsistencyConfidence >= 0.62,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Eye 3D continuity",
                    value =
                        (eye3DTemporalConsistencyConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        if (eye3DTemporalConsistencyWarnings.isEmpty()) {
                            "bilateral geometry stable"
                        } else {
                            eye3DTemporalConsistencyWarnings.first().take(28)
                        },
                    active =
                        eye3DTemporalConsistencyConfidence >= 0.62,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Gaze",
                    value = "${(gazeConfidence * 100.0).toInt()}%",
                    detail =
                        "consistency ${(faceEyeConsistencyConfidence * 100.0).toInt()}% · " +
                            "stability ${(gazeStability * 100.0).toInt()}%",
                    active =
                        gazeConfidence > 0.45 &&
                            faceEyeConsistencyConfidence >= 0.55,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Hand continuity",
                    value =
                        (handTemporalConsistencyConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        if (handTemporalConsistencyWarnings.isEmpty()) {
                            "3D articulation"
                        } else {
                            handTemporalConsistencyWarnings.first().take(28)
                        },
                    active = handTemporalConsistencyConfidence >= 0.55,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Head yaw",
                    value = "${"%.1f".format(faceHeadYaw)}°",
                    detail = "roll ${"%.1f".format(faceHeadRoll)}°",
                    active = gazeConfidence > 0.25,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Motion energy",
                    value = "${(motionScore * 100.0).toInt()}%",
                    detail = motionClass,
                    active = motionScore > 0.08,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Micro-motion",
                    value = "${(microMotionScore * 100.0).toInt()}%",
                    detail = "low-amplitude dynamics",
                    active = microMotionScore >= 0.20,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Direction changes",
                    value = "${(directionChangeRatio * 100.0).toInt()}%",
                    detail = "reversal density",
                    active = directionChangeRatio <= 0.35,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "2D uncertainty",
                    value = "${(motionMeasurementUncertainty * 100.0).toInt()}%",
                    detail = "measurement noise",
                    active = motionMeasurementUncertainty <= 0.50,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Motion evidence",
                    value =
                        (motionEvidenceScore * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        motionEvidenceWarnings.firstOrNull()
                            ?: motionEvidenceConfidence
                                .let { "confidence ${(it * 100.0).toInt()}%" },
                    active =
                        motionEvidenceConfidence >= 0.55 &&
                            motionEvidenceScore >= 0.24,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Temporal lock",
                    value = "${(temporalStability * 100.0).toInt()}%",
                    detail = "continuity",
                    active = temporalStability > 0.60,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Occlusion",
                    value = "${(occlusionRisk * 100.0).toInt()}%",
                    detail = "risk",
                    active = occlusionRisk < 0.35,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Capture clock",
                    value = if (captureTimingFps > 0.0) {
                        "${"%.1f".format(captureTimingFps)} FPS"
                    } else {
                        "--"
                    },
                    detail = if (captureTimingHealthy) {
                        "jitter ${"%.2f".format(captureTimingJitterMillis)} ms"
                    } else {
                        "timing ${(captureTimingConfidence * 100.0).toInt()}%"
                    },
                    active = captureTimingHealthy && captureTimingConfidence >= 0.70,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "3D observability",
                    value =
                        (geometricObservabilityScore * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        geometricObservabilityWarnings.firstOrNull()
                            ?: "baseline + triangulation",
                    active = geometricObservabilityScore >= 0.62,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Session health",
                    value = (sessionHealthScore * 100.0).toInt().toString() + "%",
                    detail = sessionHealthState.replace('_', ' ') +
                        " · " + sessionHealthBottleneck,
                    active =
                        sessionHealthState == "PRODUCTION" &&
                            sessionHealthScore >= 0.78,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "World coherence",
                    value = (worldConsistencyScore * 100.0).toInt().toString() + "%",
                    detail =
                        if (worldConsistencyWarnings.isEmpty()) {
                            "temporal + topology"
                        } else {
                            worldConsistencyWarnings.first().take(26)
                        },
                    active = worldConsistencyScore >= 0.72,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Rig sync",
                    value = (multiCameraSyncConfidence * 100.0).toInt().toString() + "%",
                    detail = multiCameraSyncState.replace('_', ' '),
                    active =
                        multiCameraSyncState == "SYNCHRONIZED" &&
                            multiCameraSyncConfidence >= 0.65,
                    modifier = Modifier.weight(1f),
                )
                GlassTelemetryTile(
                    label = "Temporal alignment",
                    value =
                        if (temporalAlignmentState == "SINGLE_CAMERA") {
                            "N/A"
                        } else {
                            (temporalAlignmentConfidence * 100.0)
                                .toInt()
                                .toString() + "%"
                        },
                    detail =
                        if (temporalAlignmentState == "SINGLE_CAMERA") {
                            "single-camera mode"
                        } else {
                            temporalAlignmentState.replace('_', ' ') +
                                " · spread " +
                                "%.2f".format(temporalAlignmentSpreadMillis) +
                                " ms"
                        },
                    active =
                        temporalAlignmentState == "SINGLE_CAMERA" ||
                            (
                                temporalAlignmentState == "ALIGNED" &&
                                    temporalAlignmentConfidence >= 0.65
                                ),
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

@Composable
private fun ReconstructionViewport(
    skeleton: SkeletonFrame?,
    quality: Double,
    showGrid: Boolean,
    modifier: Modifier,
) {
    var yawDegrees by remember { mutableFloatStateOf(0f) }
    var pitchDegrees by remember { mutableFloatStateOf(-8f) }
    var zoom by remember { mutableFloatStateOf(1.0f) }
    var displayMode by remember { mutableStateOf("SKELETON") }
    val colorScheme = MaterialTheme.colorScheme

    Surface(
        modifier = modifier.pointerInput(Unit) {
            detectTransformGestures { _, pan, zoomChange, rotationChange ->
                yawDegrees =
                    (
                        yawDegrees +
                            pan.x * 0.35f +
                            rotationChange * 0.55f
                        ) % 360f
                pitchDegrees = (
                    pitchDegrees - pan.y * 0.22f
                ).coerceIn(-72f, 72f)
                zoom =
                    (zoom * zoomChange)
                        .coerceIn(0.72f, 1.65f)
            }
        },
        shape = RoundedCornerShape(28.dp),
        color = colorScheme.surfaceContainerLow.copy(alpha = 0.92f),
        tonalElevation = 2.dp,
    ) {
        Box(
            Modifier.fillMaxSize()
                .background(
                    Brush.radialGradient(
                        listOf(
                            colorScheme.primary.copy(alpha = 0.09f),
                            Color.Transparent,
                        ),
                    ),
                ),
        ) {
            Box(Modifier.fillMaxSize()) {
                Canvas(Modifier.fillMaxSize().padding(18.dp)) {
                val center = Offset(size.width * 0.5f, size.height * 0.54f)
                val scale =
                    min(size.width, size.height) *
                        0.32f *
                        zoom.coerceIn(0.72f, 1.65f)
                val yaw = Math.toRadians(yawDegrees.toDouble())
                val pitch = Math.toRadians(pitchDegrees.toDouble())
                val cosYaw = kotlin.math.cos(yaw).toFloat()
                val sinYaw = kotlin.math.sin(yaw).toFloat()
                val cosPitch = kotlin.math.cos(pitch).toFloat()
                val sinPitch = kotlin.math.sin(pitch).toFloat()

                if (showGrid) {
                drawLine(
                    colorScheme.outlineVariant.copy(alpha = 0.30f),
                    Offset(0f, center.y),
                    Offset(size.width, center.y),
                    1.2f,
                )
                drawLine(
                    colorScheme.outlineVariant.copy(alpha = 0.20f),
                    Offset(center.x, 0f),
                    Offset(center.x, size.height),
                    1.2f,
                )

                val gridStep = scale * 0.22f
                for (i in -5..5) {
                    val dx = i * gridStep
                    drawLine(
                        colorScheme.outlineVariant.copy(alpha = 0.09f),
                        Offset(center.x + dx, center.y - scale),
                        Offset(center.x + dx, center.y + scale),
                        1f,
                    )
                }
                }


                val joints = skeleton?.joints.orEmpty().associateBy { it.landmarkId }
                val edges = listOf(
                    11 to 12, 11 to 13, 13 to 15, 12 to 14,
                    14 to 16, 11 to 23, 12 to 24, 23 to 24,
                    23 to 25, 25 to 27, 24 to 26, 26 to 28,
                    0 to 11, 0 to 12,
                )

                fun project(joint: ReconstructionJoint): Offset {
                    val x = joint.position.x.toFloat()
                    val y = joint.position.y.toFloat()
                    val z = joint.position.z.toFloat()

                    val rotatedX = x * cosYaw - z * sinYaw
                    val rotatedZ = x * sinYaw + z * cosYaw
                    val liftedY = y * cosPitch - rotatedZ * sinPitch
                    val depthZ = y * sinPitch + rotatedZ * cosPitch

                    val cameraZ = max(
                        0.20,
                         depthZ.toDouble() + 2.4,
                    )

                    return Offset(
                        center.x +
                            (rotatedX / cameraZ.toFloat()) * scale,
                        center.y -
                            (liftedY / cameraZ.toFloat()) * scale,
                    )
                }

                edges.forEach { pair ->
                    val ja = joints[pair.first] ?: return@forEach
                    val jb = joints[pair.second] ?: return@forEach
                    drawLine(
                        colorScheme.secondary.copy(
                            alpha = min(ja.confidence, jb.confidence)
                                .coerceIn(0.12, 1.0).toFloat(),
                        ),
                        project(ja),
                        project(jb),
                        4f,
                    )
                }

                joints.values.forEach { joint ->
                    val point = project(joint)
                    val confidence = joint.confidence
                         .coerceIn(0.0, 1.0)
                         .toFloat()
                    val speed = joint.velocity.norm().toFloat()
                    val depth = joint.position.z

                    val modeScale = when (displayMode) {
                        "DEPTH" ->
                            (
                                1.45 -
                                    depth.toFloat().coerceIn(-1f, 4f) / 4f
                            ).toFloat()
                        "MOTION" ->
                            1.0f + speed.coerceIn(0f, 3f) / 2.4f
                        "UNCERTAINTY" ->
                            (
                                0.82f +
                                    joint.positionUncertaintyMeters
                                        .coerceIn(0.0, 0.35)
                                        .toFloat() *
                                    1.8f
                                ).coerceIn(0.82f, 1.45f)
                        else ->
                            1.0f
                    }

                    val alpha = when (displayMode) {
                        "CONFIDENCE" -> confidence
                        "DEPTH" ->
                            (
                                1.0f -
                                    depth.toFloat().coerceIn(0f, 4f) / 4f
                                ).coerceIn(0.20f, 1f)
                        "MOTION" ->
                            (
                                0.28f +
                                    speed.coerceIn(0f, 2f) / 2f
                                ).coerceIn(0.20f, 1f)
                        "UNCERTAINTY" ->
                            (
                                1.0f -
                                    joint.positionUncertaintyMeters
                                     .coerceIn(0.0, 0.50)
                                     .toFloat() / 0.50f
                                ).coerceIn(0.16f, 0.96f)
                        else ->
                            confidence.coerceAtLeast(0.20f)
                    }

                    val radius =
                        (
                            5f +
                                confidence.toFloat() * 5f
                            ) * modeScale

                    if (displayMode == "MOTION") {
                        val velocity = joint.velocity
                        val velocityMagnitude = velocity.norm()
                        if (velocityMagnitude.isFinite() && velocityMagnitude > 0.02) {
                            val projectedVelocity =
                                run {
                                     val rotatedX =
                                         velocity.x.toFloat() * cosYaw -
                                             velocity.z.toFloat() * sinYaw
                                     val rotatedZ =
                                         velocity.x.toFloat() * sinYaw +
                                             velocity.z.toFloat() * cosYaw
                                     val liftedY =
                                         velocity.y.toFloat() * cosPitch -
                                             rotatedZ * sinPitch
                                     val cameraZ = max(
                                         0.20,
                                         (velocity.z + joint.position.z + 2.4).coerceAtLeast(0.20),
                                     )
                                     Offset(
                                         rotatedX / cameraZ.toFloat(),
                                         -liftedY / cameraZ.toFloat(),
                                     )
                                 }
                            val tip = point + projectedVelocity * (
                                scale * 0.055f
                            )
                            drawLine(
                                colorScheme.tertiary.copy(
                                    alpha = (
                                        0.24 +
                                            confidence * 0.60
                                        ).coerceIn(0.24, 0.84).toFloat(),
                                ),
                                start = point,
                                end = tip,
                                strokeWidth = (1.6f + velocityMagnitude.toFloat() * 0.85f)
                                    .coerceIn(1.6f, 6.0f),
                            )
                        }
                    }

                    if (displayMode == "UNCERTAINTY") {
                        val uncertaintyRadius =
                            (
                                radius *
                                    (
                                        1.0f +
                                             joint.positionUncertaintyMeters
                                                .coerceIn(0.0, 0.50)
                                                .toFloat() *
                                            2.4f
                                    )
                                ).coerceIn(
                                    radius + 4f,
                                    radius * 3.2f,
                                )

                        drawCircle(
                            colorScheme.tertiary.copy(
                                alpha = (0.10f + alpha * 0.34f)
                                    .coerceIn(0.10f, 0.46f),
                            ),
                            uncertaintyRadius,
                            point,
                            style = Stroke(width = 1.5f),
                        )
                    }

                    drawCircle(
                        colorScheme.primary.copy(
                            alpha = alpha.toFloat(),
                        ),
                        radius + 5f,
                        point,
                        style = Stroke(width = 2f),
                    )
                    drawCircle(
                        colorScheme.primary.copy(
                            alpha = alpha.toFloat(),
                        ),
                        radius,
                        point,
                    )
                }

                drawCircle(
                    colorScheme.secondary.copy(
                        alpha = (0.10 + quality * 0.26).toFloat(),
                    ),
                    scale * 0.78f,
                    center,
                    style = Stroke(width = 1.5f),
                )

                // Subtle horizon rings provide depth without placing a heavy glass material over
                // the reconstruction content itself.
                for (ring in 1..3) {
                    drawCircle(
                        colorScheme.outlineVariant.copy(
                            alpha = 0.05f + quality.toFloat() * 0.015f,
                        ),
                        scale * (0.28f * ring),
                        center,
                        style = Stroke(width = 1f),
                    )
                }
            }

            Row(
                Modifier.align(Alignment.TopStart).padding(14.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                StatusPill("3D SOLVER", quality >= 0.45)
                StatusPill("MOTION ONLY", true)
            }

            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(14.dp)
                    .clickable {
                        displayMode = when (displayMode) {
                            "SKELETON" -> "DEPTH"
                            "DEPTH" -> "CONFIDENCE"
                            "CONFIDENCE" -> "MOTION"
                            "MOTION" -> "UNCERTAINTY"
                            else -> "SKELETON"
                        }
                    },
                shape = RoundedCornerShape(14.dp),
                color = colorScheme.surface.copy(alpha = 0.76f),
            ) {
                Text(
                    displayMode,
                    modifier = Modifier.padding(
                        horizontal = 12.dp,
                        vertical = 8.dp,
                    ),
                    style = MaterialTheme.typography.labelLarge,
                )
            }

            Column(
                Modifier.align(Alignment.BottomStart).padding(14.dp),
            ) {
                Text(
                    "LIVE RECONSTRUCTION",
                    style = MaterialTheme.typography.labelSmall,
                    color = colorScheme.onSurfaceVariant,
                )
                Text(
                    if (skeleton == null) {
                        "Waiting for geometry"
                    } else {
                        "${skeleton.joints.size} joints · " +
                            "${(skeleton.globalConfidence * 100).toInt()}% confidence"
                    },
                    style = MaterialTheme.typography.titleMedium,
                )
            }
        }
    }
}

}

@Composable
private fun StudioInspector(
    modifier: Modifier,
    motionCustomization: MotionCaptureCustomization = MotionCaptureCustomization(),
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit = {},
    rigCoverageScore: Double = 0.0,
    inferenceFps: Int,
    poseConfidence: Float,
    reconstructionErrorPx: Double,
    captureIntegrityScore: Double,
    captureIntegrityBand: String,
    captureIntegrityOutputAllowed: Boolean,
    captureIntegrityBottleneck: String,
    reconstructionJoints: Int,
    gpuBackend: String,
    gpuLatencyMs: Double,
    rigScore: Double,
    lightingLux: Double,
    colorTemperature: Double,
    motionProfile: String,
    motionResponsiveness: Double,
    motionSensitivity: Double,
    predictionHorizon: Double,
    occlusionPersistence: Double,
    jitterSuppression: Double,
    reacquisitionRadius: Double,
    rejectAggressiveness: Double,
    velocityResponsiveness: Double,
    accelerationResponsiveness: Double,
    innovationGateElasticity: Double,
    motionThresholdScale: Double,
    reconstructionProfile: String,
    reconstructionSoftAccept: Double,
    reconstructionHoldFrames: Int,
    reconstructionHoldAgeMillis: Long,
    reconstructionStabilitySensitivity: Double,
    temporalPredictionHorizon: Double,
    temporalInnovationGateScale: Double,
    temporalUncertaintyGateScale: Double,
    temporalReprojectionGateScale: Double,
    temporalSmoothingResponsiveness: Double,
    temporalConfidenceDecayScale: Double,
    temporalOutlierRetention: Double,
    temporalTopologyToleranceScale: Double,
    sessionPresentationSettings: SessionPresentationSettings,
    onMotionPreset: (String) -> Unit,
    onMotionResponsiveness: (Double) -> Unit,
    onMotionSensitivity: (Double) -> Unit,
    onCustomizationMicroMotionPreservation: (Double) -> Unit,
    onCustomizationCameraMotionSuppression: (Double) -> Unit,
    onCustomizationFaceEyeFusionWeight: (Double) -> Unit,
    onCustomizationTemporalAlignmentStrictness: (Double) -> Unit,
    onPredictionHorizon: (Double) -> Unit,
    onOcclusionPersistence: (Double) -> Unit,
    onJitterSuppression: (Double) -> Unit,
    onReacquisitionRadius: (Double) -> Unit,
    onRejectAggressiveness: (Double) -> Unit,
    onVelocityResponsiveness: (Double) -> Unit,
    onAccelerationResponsiveness: (Double) -> Unit,
    onInnovationGateElasticity: (Double) -> Unit,
    onMotionThresholdScale: (Double) -> Unit,
    onReconstructionPreset: (String) -> Unit,
    onReconstructionSoftAccept: (Double) -> Unit,
    onReconstructionHoldFrames: (Int) -> Unit,
    onReconstructionHoldAgeMillis: (Long) -> Unit,
    onReconstructionStabilitySensitivity: (Double) -> Unit,
    onTemporalPredictionHorizon: (Double) -> Unit,
    onTemporalInnovationGateScale: (Double) -> Unit,
    onTemporalUncertaintyGateScale: (Double) -> Unit,
    onTemporalReprojectionGateScale: (Double) -> Unit,
    onTemporalSmoothingResponsiveness: (Double) -> Unit,
    onTemporalConfidenceDecayScale: (Double) -> Unit,
    onTemporalOutlierRetention: (Double) -> Unit,
    onTemporalTopologyToleranceScale: (Double) -> Unit,
    onSessionClipName: (String) -> Unit,
    onSessionCountdownSeconds: (Int) -> Unit,
    onRequireCalibrationBeforeRecord: (Boolean) -> Unit,
    showPerformanceTelemetry: Boolean = false,
    adaptiveRegimeStrength: Double = 0.82,
    onAdaptiveRegimeStrength: (Double) -> Unit = {},
    onCustomizationPreset: (String) -> Unit = {},
    onCustomizationBodyDetail: (Double) -> Unit = {},
    onCustomizationFaceDetail: (Double) -> Unit = {},
    onCustomizationEyeDetail: (Double) -> Unit = {},
    onCustomizationHandDetail: (Double) -> Unit = {},
    onCustomizationHandTemporalStrictness: (Double) -> Unit = {},
    onCustomizationHandGeometryStrictness: (Double) -> Unit = {},
    onCustomizationDepthDetail: (Double) -> Unit = {},
    onCustomizationDenseMeshDetail: (Double) -> Unit = {},
    onCustomizationContactDetail: (Double) -> Unit = {},
    onCustomizationWorldConsistencyDetail: (Double) -> Unit = {},
    onCustomizationTopologyConstraintStrength: (Double) -> Unit = {},
    onCustomizationMaxSubjects: (Int) -> Unit = {},
    onCustomizationOcclusionPredictionFrames: (Int) -> Unit = {},
    onCustomizationTargetFps: (Int) -> Unit = {},
    onCustomizationInferenceFps: (Int) -> Unit = {},
    onCustomizationEyeTemporalStrictness: (Double) -> Unit = {},
    onCustomizationOutputIntegrityStrictness: (Double) -> Unit = {},
    onCustomizationMinimumParallaxDegrees: (Double) -> Unit = {},
    onCustomizationPreferredTemporalSkewMillis: (Double) -> Unit = {},
    onCustomizationMaximumTemporalSkewMillis: (Double) -> Unit = {},
    onCustomizationMaximumReprojectionErrorPx: (Double) -> Unit = {},
    onCustomizationMaximumReconstructionUncertaintyMeters: (Double) -> Unit = {},
) {
    var advancedControls by remember { mutableStateOf(false) }

    Column(
        modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionTitle(
            "Operator telemetry",
            "Critical numbers stay visible. Deep controls stay below.",
        )

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            GlassMetric(
                "Integrity",
                "${(captureIntegrityScore * 100.0).toInt()}%",
                captureIntegrityBand.replace('_', ' '),
                Modifier.weight(1f),
            )
            GlassMetric(
                "Output",
                if (captureIntegrityOutputAllowed) "TRUSTED" else "HELD",
                captureIntegrityBottleneck.replace('_', ' '),
                Modifier.weight(1f),
            )
        }

        GlassGauge(
            "Capture integrity",
            captureIntegrityScore.toFloat(),
        )

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            GlassMetric(
                "Inference",
                "${inferenceFps} fps",
                "AI schedule",
                Modifier.weight(1f),
            )
            GlassMetric(
                "Pose",
                "${(poseConfidence * 100f).toInt()}%",
                "${reconstructionJoints} joints",
                Modifier.weight(1f),
            )
        }

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            GlassMetric(
                "Residual",
                if (reconstructionErrorPx.isFinite()) {
                    "${"%.1f".format(reconstructionErrorPx)} px"
                } else {
                    "—"
                },
                "reprojection",
                Modifier.weight(1f),
            )
            GlassMetric(
                "GPU",
                "${"%.1f".format(gpuLatencyMs)} ms",
                gpuBackend,
                Modifier.weight(1f),
            )
        }

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            GlassMetric(
                "Rig",
                "${(rigScore * 100.0).toInt()}%",
                "calibration quality",
                Modifier.weight(1f),
            )
            GlassMetric(
                "Light",
                "${lightingLux.toInt()} lx",
                "${colorTemperature.toInt()} K",
                Modifier.weight(1f),
            )
        }

        GlassGauge("Pose confidence", poseConfidence)
        GlassGauge("Rig confidence", rigScore.toFloat())
             GlassGauge(
                 "Calibration coverage",
                 rigCoverageScore.toFloat(),
             )

        SectionTitle("Motion profile", motionProfile)
        PresetRow(
            listOf(
                "CINEMATIC",
                "ACTION",
                "PRECISION",
                "CREATURE",
                "FACIAL",
                "HAND_DETAIL",
            ),
            motionProfile,
            onMotionPreset,
        )

        SectionTitle(
            "Advanced capture behavior",
            "These controls modify live inference allocation, temporal fusion and reconstruction trust.",
        )
        ControlSlider(
            "Pose prior strength",
            motionCustomization.posePriorStrength,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        posePriorStrength = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Foot contact strictness",
            motionCustomization.footContactStrictness,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        footContactStrictness = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Head pose strictness",
            motionCustomization.headPoseStrictness,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        headPoseStrictness = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Gaze priority",
            motionCustomization.gazePriority,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        gazePriority = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Depth temporal fusion",
            motionCustomization.depthTemporalFusion,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        depthTemporalFusion = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Camera drift response",
            motionCustomization.cameraDriftResponse,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        cameraDriftResponse = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Thermal quality floor",
            motionCustomization.thermalQualityFloor,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        thermalQualityFloor = value.coerceIn(0.20, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Network jitter tolerance",
            motionCustomization.networkJitterToleranceMillis,
            0f..120f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        networkJitterToleranceMillis = value.coerceIn(0.0, 250.0)
                    }
                }
            },
            1f,
            " ms",
        )
        ControlSlider(
            "Adaptive mesh detail",
            motionCustomization.meshAdaptiveDetail,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        meshAdaptiveDetail = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Retargeting strictness",
            motionCustomization.retargetingStrictness,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        retargetingStrictness = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Scale preservation",
            motionCustomization.scalePreservation,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        scalePreservation = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Occlusion recovery confidence",
            motionCustomization.occlusionRecoveryConfidence,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        occlusionRecoveryConfidence = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Landmark consistency",
            motionCustomization.landmarkConsistencyStrictness,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        landmarkConsistencyStrictness = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Semantic motion sensitivity",
            motionCustomization.semanticMotionSensitivity,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        semanticMotionSensitivity = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Root motion lock",
            motionCustomization.rootMotionLock,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        rootMotionLock = value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )

        ControlSlider("Responsiveness", motionResponsiveness, 0f..1f, onMotionResponsiveness)
        ControlSlider("Sensitivity", motionSensitivity, 0f..1f, onMotionSensitivity)
        ControlSlider(
            "Prediction horizon", predictionHorizon, 0.02f..0.20f,
            onPredictionHorizon, 1000f, " ms",
        )
        ControlSlider(
            "Occlusion persistence", occlusionPersistence, 0f..1f,
            onOcclusionPersistence,
        )
        ControlSlider(
            "Jitter suppression", jitterSuppression, 0f..1f,
            onJitterSuppression,
        )

        SectionTitle(
            "Session",
            "Name the take before recording and control the pre-roll workflow.",
        )
        OutlinedTextField(
            value = sessionPresentationSettings.clipName,
            onValueChange = onSessionClipName,
            label = { Text("Clip name") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        ControlSlider(
            "Recording countdown",
            sessionPresentationSettings.countdownSeconds.toDouble(),
            0f..10f,
            {
                onSessionCountdownSeconds(
                    it.toInt().coerceIn(0, 10),
                )
            },
            1f,
            " s",
        )
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text(
                    "Require calibration",
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    "Block recording until the current rig is calibrated.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(
                checked = sessionPresentationSettings.requireCalibrationBeforeRecord,
                onCheckedChange = onRequireCalibrationBeforeRecord,
            )
        }

        SectionTitle(
            "System tuning",
            "High-level runtime budgets for different capture priorities.",
        )
        PresetRow(
            listOf(
                "QUALITY",
                "BALANCED",
                "LOW_LATENCY",
            ),
            when {
                motionCustomization.inferenceFps >= 50 &&
                    motionCustomization.bodyDetail >= 0.95 ->
                    "QUALITY"
                motionCustomization.inferenceFps <= 30 &&
                    motionCustomization.bodyDetail < 0.90 ->
                    "LOW_LATENCY"
                else ->
                    "BALANCED"
            },
            onCustomizationPreset,
)

        GlassCard(intensity = 0.34f) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    Text(
                        "Expert controls",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Advanced perception, geometry and temporal solver tuning.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = advancedControls,
                    onCheckedChange = { advancedControls = it },
                )
            }
        }

        if (advancedControls) {


        SectionTitle(
            "Perception budget",
            "Runtime detail budgets feed the adaptive inference scheduler.",
        )
        ControlSlider(
            "Body detail",
            motionCustomization.bodyDetail,
            0f..1f,
            onCustomizationBodyDetail,
        )
        ControlSlider(
            "Face detail",
            motionCustomization.faceDetail,
            0f..1f,
            onCustomizationFaceDetail,
        )
        ControlSlider(
            "Eye detail",
            motionCustomization.eyeDetail,
            0f..1f,
            onCustomizationEyeDetail,
        )
        ControlSlider(
            "Hand detail",
            motionCustomization.handDetail,
            0f..1f,
            onCustomizationHandDetail,
        )
        ControlSlider(
            "Hand temporal strictness",
            motionCustomization.handTemporalStrictness,
            0f..1f,
            onCustomizationHandTemporalStrictness,
        )
        ControlSlider(
            "Hand geometry strictness",
            motionCustomization.handGeometryStrictness,
            0f..1f,
            onCustomizationHandGeometryStrictness,
        )
        ControlSlider(
            "Multi-view evidence adaptivity",
            motionCustomization.multiViewEvidenceAdaptivity,
            0f..1f,
            { value ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        multiViewEvidenceAdaptivity =
                            value.coerceIn(0.0, 1.0)
                    }
                }
            },
        )
        ControlSlider(
            "Depth detail",
            motionCustomization.depthDetail,
            0f..1f,
            onCustomizationDepthDetail,
        )
        ControlSlider(
            "Dense mesh detail",
            motionCustomization.denseMeshDetail,
            0f..1f,
            onCustomizationDenseMeshDetail,
        )
        ControlSlider(
            "Contact detail",
            motionCustomization.contactDetail,
            0f..1f,
            onCustomizationContactDetail,
        )
        ControlSlider(
            "World coherence",
            motionCustomization.worldConsistencyDetail,
            0f..1f,
            onCustomizationWorldConsistencyDetail,
        )
        ControlSlider(
            "Topology constraint",
            motionCustomization.topologyConstraintStrength,
            0f..1f,
            onCustomizationTopologyConstraintStrength,
        )
        ControlSlider(
            "Max subjects",
            motionCustomization.maxSubjects.toDouble(),
            1f..64f,
            { onCustomizationMaxSubjects(it.toInt().coerceIn(1, 64)) },
            1f,
            "",
        )
        ControlSlider(
            "Occlusion horizon",
            motionCustomization.maximumOcclusionPredictionFrames.toDouble(),
            0f..30f,
            {
                onCustomizationOcclusionPredictionFrames(
                    it.toInt().coerceIn(0, 30),
                )
            },
            1f,
            " frames",
        )
        ControlSlider(
            "Capture FPS",
            motionCustomization.targetFps.toDouble(),
            15f..120f,
            { onCustomizationTargetFps(it.toInt()) },
            1f,
            " fps",
        )
        ControlSlider(
            "Inference FPS",
            motionCustomization.inferenceFps.toDouble(),
            5f..60f,
            { onCustomizationInferenceFps(it.toInt()) },
            1f,
            " fps",
        )
        ControlSlider(
            "Micro-motion preservation",
            motionCustomization.microMotionPreservation,
            0f..1f,
            onCustomizationMicroMotionPreservation,
        )
        ControlSlider(
            "Camera-motion suppression",
            motionCustomization.cameraMotionSuppression,
            0f..1f,
            onCustomizationCameraMotionSuppression,
        )
        ControlSlider(
            "Face / eye fusion weight",
            motionCustomization.faceEyeFusionWeight,
            0f..1f,
            onCustomizationFaceEyeFusionWeight,
        )
        ControlSlider(
            "3D eye temporal strictness",
            motionCustomization.eyeTemporalStrictness,
            0f..1f,
            onCustomizationEyeTemporalStrictness,
        )
        ControlSlider(
            "Output integrity strictness",
            motionCustomization.outputIntegrityStrictness,
            0f..1f,
            onCustomizationOutputIntegrityStrictness,
        )
        ControlSlider(
            "Temporal alignment strictness",
            motionCustomization.temporalAlignmentStrictness,
            0f..1f,
            onCustomizationTemporalAlignmentStrictness,
        )
        SectionTitle(
            "3D observability",
            "Tune the evidence gates used by calibrated multi-view reconstruction.",
        )
        ControlSlider(
            "Minimum parallax",
            motionCustomization.minimumParallaxDegrees,
            0.20f..2.50f,
            onCustomizationMinimumParallaxDegrees,
            0.05f,
            "°",
        )
        ControlSlider(
            "Preferred temporal skew",
            motionCustomization.preferredTemporalSkewMillis,
            0.50f..15f,
            onCustomizationPreferredTemporalSkewMillis,
            0.50f,
            " ms",
        )
        ControlSlider(
            "Maximum temporal skew",
            motionCustomization.maximumTemporalSkewMillis,
            5f..60f,
            onCustomizationMaximumTemporalSkewMillis,
            0.50f,
            " ms",
        )

        ControlSlider(
            "Maximum reprojection error",
            motionCustomization.maximumReprojectionErrorPx,
            1f..20f,
            onCustomizationMaximumReprojectionErrorPx,
            0.10f,
            " px",
        )
        ControlSlider(
            "Maximum 3D uncertainty",
            motionCustomization.maximumReconstructionUncertaintyMeters,
            0.05f..2f,
            onCustomizationMaximumReconstructionUncertaintyMeters,
            0.05f,
            " m",
        )

        SectionTitle(
            "Acquisition & tracking",
            "Re-localization and measurement rejection remain operator-tunable.",
        )
        ControlSlider(
            "Reacquisition radius",
            reacquisitionRadius,
            0f..1f,
            onReacquisitionRadius,
        )
        ControlSlider(
            "Reject aggressiveness",
            rejectAggressiveness,
            0f..1f,
            onRejectAggressiveness,
        )
        ControlSlider(
            "Velocity responsiveness",
            velocityResponsiveness,
            0f..1f,
            onVelocityResponsiveness,
        )
        ControlSlider(
            "Acceleration responsiveness",
            accelerationResponsiveness,
            0f..1f,
            onAccelerationResponsiveness,
        )
        ControlSlider(
            "Innovation gate elasticity",
            innovationGateElasticity,
            0f..2f,
            onInnovationGateElasticity,
            50f,
            "",
        )
        ControlSlider(
            "Motion threshold scale",
            motionThresholdScale,
            0.25f..2f,
            onMotionThresholdScale,
            50f,
            "x",
        )

        SectionTitle("Reconstruction", reconstructionProfile)
        PresetRow(
            listOf(
                "CINEMATIC",
                "ACTION",
                "PRECISION",
                "CREATURE",
                "FACIAL",
                "HAND_DETAIL",
            ),
            reconstructionProfile,
            onReconstructionPreset,
        )
        ControlSlider(
            "Soft accept threshold",
            reconstructionSoftAccept,
            0f..1f,
            onReconstructionSoftAccept,
        )
        ControlSlider(
            "Hold frames",
            reconstructionHoldFrames.toDouble(),
            1f..60f,
            { onReconstructionHoldFrames(it.toInt().coerceIn(1, 60)) },
            1f,
            " frames",
        )
        ControlSlider(
            "Hold age",
            reconstructionHoldAgeMillis.toDouble(),
            16f..1200f,
            { onReconstructionHoldAgeMillis(it.toLong().coerceIn(16L, 1200L)) },
            1f,
            " ms",
        )
        ControlSlider(
            "Stability sensitivity",
            reconstructionStabilitySensitivity,
            0f..2f,
            onReconstructionStabilitySensitivity,
            50f,
            "x",
        )

        SectionTitle(
            "Temporal solver",
            "Prediction, uncertainty, reprojection and topology gates.",
        )
        ControlSlider(
            "Prediction horizon",
            temporalPredictionHorizon,
            0.01f..0.30f,
            onTemporalPredictionHorizon,
            1000f,
            " ms",
        )
        ControlSlider(
            "Innovation gate",
            temporalInnovationGateScale,
            0f..2f,
            onTemporalInnovationGateScale,
            50f,
            "x",
        )
        ControlSlider(
            "Uncertainty gate",
            temporalUncertaintyGateScale,
            0f..2f,
            onTemporalUncertaintyGateScale,
            50f,
            "x",
        )
        ControlSlider(
            "Reprojection gate",
            temporalReprojectionGateScale,
            0f..2f,
            onTemporalReprojectionGateScale,
            50f,
            "x",
        )
        ControlSlider(
            "Temporal smoothing",
            temporalSmoothingResponsiveness,
            0f..1f,
            onTemporalSmoothingResponsiveness,
        )
        ControlSlider(
            "Confidence decay",
            temporalConfidenceDecayScale,
            0f..2f,
            onTemporalConfidenceDecayScale,
            50f,
            "x",
        )
        ControlSlider(
            "Outlier retention",
            temporalOutlierRetention,
            0f..1f,
            onTemporalOutlierRetention,
        )
        ControlSlider(
            "Topology tolerance",
            temporalTopologyToleranceScale,
            0f..2f,
            onTemporalTopologyToleranceScale,
            50f,
            "x",
        )

        }

        GlassCard {
            Column(
                Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("Capture contract", style = MaterialTheme.typography.titleMedium)
                Text(
                    "RGB is used transiently for perception. Canonical output is motion, geometry, " +
                        "confidence, lighting and synchronized sensor metadata.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    StatusPill("RGB transient", true)
                    StatusPill("Video OFF", true)
                    StatusPill("Motion ON", true)
                }
            }
        }
    }
}

@Composable
private fun PresetRow(
    items: List<String>,
    active: String,
    onSelect: (String) -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items.forEach { item ->
            Surface(
                onClick = { onSelect(item) },
                shape = RoundedCornerShape(16.dp),
                color = if (active == item) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceContainerHigh
                },
            ) {
                Text(
                    item,
                    Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                    style = MaterialTheme.typography.labelMedium,
                )
            }
        }
    }
}

@Composable
private fun ControlSlider(
    label: String,
    value: Double,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Double) -> Unit,
    displayScale: Float = 100f,
    suffix: String = "%",
) {
    GlassCard {
        Column(
            Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(label, style = MaterialTheme.typography.labelLarge)
                Text(
                    "${(value.toFloat() * displayScale).toInt()}$suffix",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Slider(
                value = value.toFloat().coerceIn(range.start, range.endInclusive),
                onValueChange = { onChange(it.toDouble()) },
                valueRange = range,
            )
        }
    }
}

@Composable
private fun GlassDiagnosticsStrip(
    title: String,
    messages: List<String>,
) {
    GlassCard(intensity = 0.46f) {
        Column(
            Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                title,
                style = MaterialTheme.typography.labelLarge,
            )
            messages.forEach { message ->
                Text(
                    "• " + message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ReadinessPanel(
    readiness: com.nivukx.nox.core.model.CaptureReadiness?,
) {
    GlassCard(intensity = 0.80f) {
        Column(
            Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        "CAPTURE PREFLIGHT",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        readiness?.let {
                            if (it.ready) "READY" else "WAITING FOR CONDITIONS"
                        } ?: "ANALYZING",
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
                StatusPill(
                    text = readiness?.let { if (it.ready) "READY" else "CHECK" } ?: "SCAN",
                    healthy = readiness?.ready == true,
                )
            }

            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                readiness?.let {
                    StatusPill("CAMERA", it.cameraReady)
                    StatusPill("MODEL", it.modelReady)
                    StatusPill("THERMAL", it.thermalReady)
                    StatusPill("STORAGE", it.storageReady)
                    StatusPill("POLICY", it.sessionPolicyReady)
                    StatusPill("SUBJECT", it.subjectVisible)
                    StatusPill(
                        "SYNC",
                        it.multiCameraSyncHealthy &&
                            it.multiCameraSyncConfidence >= 0.55,
                    )
                }
            }

            readiness?.let {
                GlassTelemetryTile(
                    label = "Motion quality",
                    value = (it.motionQuality * 100.0).toInt().toString() + "%",
                    detail = "capture evidence",
                    active = it.motionQuality >= 0.50,
                    modifier = Modifier.fillMaxWidth(),
                )

                GlassTelemetryTile(
                    label = "Capture clock",
                    value = if (it.captureTimingFps > 0.0) {
                        "${"%.1f".format(it.captureTimingFps)} FPS"
                    } else {
                        "-- FPS"
                    },
                    detail =
                        "jitter ${"%.2f".format(it.captureTimingJitterMillis)} ms · " +
                            "confidence ${(it.captureTimingConfidence * 100.0).toInt()}%",
                    active =
                        it.captureTimingHealthy &&
                            it.captureTimingConfidence >= 0.60 &&
                            it.captureTimingJitterMillis <= 6.0,
                    modifier = Modifier.fillMaxWidth(),
                )

                GlassTelemetryTile(
                    label = "Rig synchronization",
                    value =
                        (it.multiCameraSyncConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    detail =
                        if (it.multiCameraSyncHealthy) {
                            "clock alignment"
                        } else {
                            "multi-view clock degraded"
                        },
                    active =
                        it.multiCameraSyncHealthy &&
                            it.multiCameraSyncConfidence >= 0.55,
                    modifier = Modifier.fillMaxWidth(),
                )

                if (it.reasons.isNotEmpty()) {
                    Text(
                        it.reasons.take(3).joinToString(" · "),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun MotionEventPanel(
    eventType: String,
    confidence: Double,
) {
    GlassCard(intensity = 0.58f) {
        Row(
            Modifier.fillMaxWidth().padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text(
                    "MOTION EVENT",
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    eventType.replace('_', ' '),
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    "semantic segmentation",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            StatusPill(
                text = "${(confidence * 100.0).toInt()}%",
                healthy = confidence >= 0.55,
            )
        }
    }
}

@Composable
private fun CountdownOverlay(
    seconds: Int,
) {
    Box(
        Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            shape = RoundedCornerShape(32.dp),
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.88f),
            tonalElevation = 8.dp,
        ) {
            Column(
                Modifier.padding(horizontal = 32.dp, vertical = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    "GET READY",
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    seconds.toString(),
                    style = MaterialTheme.typography.displayLarge,
                )
                Text(
                    "capture starts automatically",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
