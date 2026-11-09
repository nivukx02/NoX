package com.nivukx.nox

import android.Manifest
import android.os.Build
import android.os.Bundle
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.adaptive.currentWindowAdaptiveInfoV2
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteScaffoldDefaults
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteType
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.nivukx.nox.core.model.GpuBackend
import com.nivukx.nox.core.model.CaptureState
import com.nivukx.nox.core.model.ProductionCaptureProfile
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.MotionFieldPreviewFrame
import com.nivukx.nox.core.model.MotionFieldGpuState
import com.nivukx.nox.graphics.MotionFieldPreviewSurface
import com.nivukx.nox.ui.UltraMotionStudio
import com.nivukx.nox.ui.GlassAction
import com.nivukx.nox.ui.GlassCard
import com.nivukx.nox.ui.GlassGauge
import com.nivukx.nox.ui.GlassMetric
import com.nivukx.nox.ui.LiquidGlassBackdrop
import com.nivukx.nox.ui.SectionTitle
import com.nivukx.nox.ui.StatusPill
import com.nivukx.nox.ui.TelemetryChip
import com.nivukx.nox.ui.GlassTelemetryTile

private val NoXDarkColors = darkColorScheme(
    primary = Color(0xFF8BB8FF),
    onPrimary = Color(0xFF06101E),
    primaryContainer = Color(0xFF17355C),
    onPrimaryContainer = Color(0xFFD9E8FF),
    secondary = Color(0xFF74E4D4),
    onSecondary = Color(0xFF071412),
    secondaryContainer = Color(0xFF123D39),
    onSecondaryContainer = Color(0xFFD6FFF7),
    background = Color(0xFF05070B),
    onBackground = Color(0xFFF1F5FF),
    surface = Color(0xFF090C12),
    onSurface = Color(0xFFF1F5FF),
    surfaceContainerHigh = Color(0xFF111722),
    surfaceContainerHighest = Color(0xFF171E2B),
    onSurfaceVariant = Color(0xFF9AA8BD),
    outlineVariant = Color(0xFF334057),
    error = Color(0xFFFF7676),
    errorContainer = Color(0xFF4A171C),
)

class MainActivity : ComponentActivity() {
    private val viewModel: NoXViewModel by viewModels()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        val cameraGranted =
            grants[Manifest.permission.CAMERA] == true
        runCatching {
            viewModel.onCameraPermissionResult(cameraGranted)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            val startup =
                remember {
                    try {
                        viewModel to null
                    } catch (error: Throwable) {
                        null to error
                    }
                }
            val startupViewModel = startup.first
            if (startupViewModel != null) {
                NoXApp(startupViewModel)
            } else {
                StartupRecoveryScreen(startup.second)
            }
        }

        val permissions = buildList {
            add(Manifest.permission.CAMERA)

            if (Build.VERSION.SDK_INT <= 30) {
                add(Manifest.permission.ACCESS_FINE_LOCATION)
            }

            if (Build.VERSION.SDK_INT >= 31) {
                add(Manifest.permission.BLUETOOTH_SCAN)
                add(Manifest.permission.BLUETOOTH_CONNECT)
                add(Manifest.permission.BLUETOOTH_ADVERTISE)
            }

            if (Build.VERSION.SDK_INT >= 33) {
                add(Manifest.permission.NEARBY_WIFI_DEVICES)
                add(Manifest.permission.POST_NOTIFICATIONS)
            }

            if (Build.VERSION.SDK_INT >= 37) {
                add(Manifest.permission.ACCESS_LOCAL_NETWORK)
            }
        }.toTypedArray()

        permissionLauncher.launch(permissions)
    }
}

@Composable
private fun NoXApp(
    viewModel: NoXViewModel,
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val devices by viewModel.devices.collectAsStateWithLifecycle()
    val frameCount by viewModel.frameCount.collectAsStateWithLifecycle()
    val postCaptureState by
        viewModel.postCaptureState.collectAsStateWithLifecycle()
    val postCaptureProgress by
        viewModel.postCaptureProgress.collectAsStateWithLifecycle()
    val postCaptureProcessedFrames by
        viewModel.postCaptureProcessedFrames.collectAsStateWithLifecycle()
    val postCaptureMaxActors by
        viewModel.postCaptureMaxActors.collectAsStateWithLifecycle()
    val poseCount by viewModel.poseCount.collectAsStateWithLifecycle()
    val poseConfidence by viewModel.poseConfidence.collectAsStateWithLifecycle()
    val handCount by viewModel.handCount.collectAsStateWithLifecycle()
    val handConfidence by viewModel.handConfidence.collectAsStateWithLifecycle()
    val hand3DConfidence by viewModel.hand3DConfidence.collectAsStateWithLifecycle()
    val handTemporalConsistencyConfidence by
        viewModel.handTemporalConsistencyConfidence.collectAsStateWithLifecycle()
    val handLongitudinalAnatomyConsistency by
        viewModel.handLongitudinalAnatomyConsistency
            .collectAsStateWithLifecycle()
    val handUncertaintyConfidence by
        viewModel.handUncertaintyConfidence
            .collectAsStateWithLifecycle()
    val handUncertaintyRadiusMeters by
        viewModel.handUncertaintyRadiusMeters
            .collectAsStateWithLifecycle()
    val handUncertaintyCoverage by
        viewModel.handUncertaintyCoverage
            .collectAsStateWithLifecycle()
    val handUncertaintyOodRisk by
        viewModel.handUncertaintyOodRisk
            .collectAsStateWithLifecycle()
    val handUncertaintyPolicy by
        viewModel.handUncertaintyPolicy
            .collectAsStateWithLifecycle()
    val handUncertaintyWarnings by
        viewModel.handUncertaintyWarnings
            .collectAsStateWithLifecycle()
    val handTemporalConsistencyWarnings by
        viewModel.handTemporalConsistencyWarnings.collectAsStateWithLifecycle()
    val focusMode by viewModel.focusMode.collectAsStateWithLifecycle()
    val focusConfidence by viewModel.focusConfidence.collectAsStateWithLifecycle()
    val subjectLockState by viewModel.subjectLockState.collectAsStateWithLifecycle()
    val subjectLockConfidence by viewModel.subjectLockConfidence.collectAsStateWithLifecycle()
    val identityContinuityScore by
        viewModel.identityContinuityScore.collectAsStateWithLifecycle()
    val identitySwapRisk by
        viewModel.identitySwapRisk.collectAsStateWithLifecycle()
    val identityAction by
        viewModel.identityAction.collectAsStateWithLifecycle()
    val identityReason by
        viewModel.identityReason.collectAsStateWithLifecycle()

    val identityEvidenceFusionScore by
        viewModel.identityEvidenceFusionScore.collectAsStateWithLifecycle()
    val identitySelectedModalityCount by
        viewModel.identitySelectedModalityCount.collectAsStateWithLifecycle()
    val identityEvidenceWarnings by
        viewModel.identityEvidenceWarnings.collectAsStateWithLifecycle()

    val faceEyeFusionConfidence by
        viewModel.faceEyeFusionConfidence.collectAsStateWithLifecycle()
    val faceEyeSaccadeVelocityDegreesPerSecond by
        viewModel.faceEyeSaccadeVelocityDegreesPerSecond
            .collectAsStateWithLifecycle()
    val faceEyeFusionAccepted by
        viewModel.faceEyeFusionAccepted.collectAsStateWithLifecycle()
    val faceEyeFusionMode by
        viewModel.faceEyeFusionMode.collectAsStateWithLifecycle()
    val faceEyeIndependentEvidenceSupport by
        viewModel.faceEyeIndependentEvidenceSupport
            .collectAsStateWithLifecycle()
    val faceEyeReliabilityTrend by
        viewModel.faceEyeReliabilityTrend
            .collectAsStateWithLifecycle()
    val faceEyeFusionWarnings by
        viewModel.faceEyeFusionWarnings.collectAsStateWithLifecycle()

    val faceEyeMetricState by
        viewModel.faceEyeMetricState.collectAsStateWithLifecycle()
    val faceEyeMetricConfidence by
        viewModel.faceEyeMetricConfidence.collectAsStateWithLifecycle()
    val faceEyeMetricEyeUncertaintyMm by
        viewModel.faceEyeMetricEyeUncertaintyMm.collectAsStateWithLifecycle()
    val faceEyeMetricWarnings by
        viewModel.faceEyeMetricWarnings.collectAsStateWithLifecycle()
    val faceEyeMetricRecommendations by
        viewModel.faceEyeMetricRecommendations.collectAsStateWithLifecycle()
    val multiPersonIdentityContinuity by
        viewModel.multiPersonIdentityContinuity.collectAsStateWithLifecycle()
    val multiPersonIdentityReidentifications by
        viewModel.multiPersonIdentityReidentifications.collectAsStateWithLifecycle()
    val multiPersonIdentityPredictions by
        viewModel.multiPersonIdentityPredictions.collectAsStateWithLifecycle()
    val multiPersonObservedCount by viewModel.multiPersonObservedCount.collectAsStateWithLifecycle()
    val multiPersonAdmittedCount by viewModel.multiPersonAdmittedCount.collectAsStateWithLifecycle()
    val multiPersonHeldCount by viewModel.multiPersonHeldCount.collectAsStateWithLifecycle()
    val multiPersonStarvationRisk by viewModel.multiPersonStarvationRisk.collectAsStateWithLifecycle()
    val multiPersonFairnessBoostedCount by viewModel.multiPersonFairnessBoostedCount.collectAsStateWithLifecycle()
    val multiPerson3DReconstructedCount by viewModel.multiPerson3DReconstructedCount.collectAsStateWithLifecycle()
    val multiPerson3DDetailedCount by viewModel.multiPerson3DDetailedCount.collectAsStateWithLifecycle()
    val multiPerson3DPredictedCount by viewModel.multiPerson3DPredictedCount.collectAsStateWithLifecycle()
    val multiPerson3DPredictionMaxAgeMillis by viewModel.multiPerson3DPredictionMaxAgeMillis.collectAsStateWithLifecycle()
    val crowdActorSilhouetteSupport by
        viewModel.crowdActorSilhouetteSupport.collectAsStateWithLifecycle()
    val crowdActorSilhouetteAmbiguity by
        viewModel.crowdActorSilhouetteAmbiguity.collectAsStateWithLifecycle()
    val personSegmentationCoverage by viewModel.personSegmentationCoverage.collectAsStateWithLifecycle()
    val personSegmentationConfidence by viewModel.personSegmentationConfidence.collectAsStateWithLifecycle()
    val personSegmentationBoundaryStability by viewModel.personSegmentationBoundaryStability.collectAsStateWithLifecycle()
    val personSegmentationLatencyMs by viewModel.personSegmentationLatencyMs.collectAsStateWithLifecycle()
    val personSegmentationState by viewModel.personSegmentationState.collectAsStateWithLifecycle()
    val personSegmentationDelegate by viewModel.personSegmentationDelegate.collectAsStateWithLifecycle()
    val personSegmentationFallbackCount by viewModel.personSegmentationFallbackCount.collectAsStateWithLifecycle()
    val crowdMotionMeanIdentitySafety by viewModel.crowdMotionMeanIdentitySafety.collectAsStateWithLifecycle()
    val crowdMotionInteractionPressure by viewModel.crowdMotionInteractionPressure.collectAsStateWithLifecycle()
    val crowdMotionDifferentialScore by viewModel.crowdMotionDifferentialScore.collectAsStateWithLifecycle()
    val crowdMotionSchedulingCapacity by viewModel.crowdMotionSchedulingCapacity.collectAsStateWithLifecycle()
    val crowdWorldLandmarkCoverage by viewModel.crowdWorldLandmarkCoverage.collectAsStateWithLifecycle()
    val crowdWorldBoneConsistency by viewModel.crowdWorldBoneConsistency.collectAsStateWithLifecycle()
    val crowdWorldGeometryReliability by viewModel.crowdWorldGeometryReliability.collectAsStateWithLifecycle()
    val crowdAppearanceQuality by viewModel.crowdAppearanceQuality.collectAsStateWithLifecycle()
    val crowdSegmentationEvidence by viewModel.crowdSegmentationEvidence.collectAsStateWithLifecycle()
    val crowdSegmentationCoverage by viewModel.crowdSegmentationCoverage.collectAsStateWithLifecycle()
    val crowdMotionBottleneck by viewModel.crowdMotionBottleneck.collectAsStateWithLifecycle()
    val crowdMotionHeldCount by viewModel.crowdMotionHeldCount.collectAsStateWithLifecycle()
    val crowdMotionPredictedCount by viewModel.crowdMotionPredictedCount.collectAsStateWithLifecycle()
    val crowdMotionWarnings by viewModel.crowdMotionWarnings.collectAsStateWithLifecycle()



    val faceEyeCaptureDomainConfidence by
        viewModel.faceEyeCaptureDomainConfidence
            .collectAsStateWithLifecycle()
    val faceEyeLightingConfidence by
        viewModel.faceEyeLightingConfidence
            .collectAsStateWithLifecycle()
    val faceEyeScaleConfidence by
        viewModel.faceEyeScaleConfidence
            .collectAsStateWithLifecycle()

    val faceEyeUncertaintyCalibrationConfidence by
        viewModel.faceEyeUncertaintyCalibrationConfidence
            .collectAsStateWithLifecycle()
    val faceEyeUncertaintyCalibrationReady by
        viewModel.faceEyeUncertaintyCalibrationReady
            .collectAsStateWithLifecycle()
    val faceUncertaintyCalibrationMultiplier by
        viewModel.faceUncertaintyCalibrationMultiplier
            .collectAsStateWithLifecycle()
    val eyeUncertaintyCalibrationMultiplier by
        viewModel.eyeUncertaintyCalibrationMultiplier
            .collectAsStateWithLifecycle()
    val faceUncertaintyCoverage by
        viewModel.faceUncertaintyCoverage
            .collectAsStateWithLifecycle()
    val eyeUncertaintyCoverage by
        viewModel.eyeUncertaintyCoverage
            .collectAsStateWithLifecycle()
    val faceEyeFailureAttribution by
        viewModel.faceEyeFailureAttribution.collectAsStateWithLifecycle()

    val faceEyeGeometricIntegrity by
        viewModel.faceEyeGeometricIntegrity.collectAsStateWithLifecycle()
    val faceEyeFaceFreshnessScore by
        viewModel.faceEyeFaceFreshnessScore.collectAsStateWithLifecycle()
    val faceEyeGazeAgreementScore by
        viewModel.faceEyeGazeAgreementScore.collectAsStateWithLifecycle()
    val faceEyeRayGeometryCondition by
        viewModel.faceEyeRayGeometryCondition
            .collectAsStateWithLifecycle()
    val faceEyeIpdConsistencyScore by
        viewModel.faceEyeIpdConsistencyScore.collectAsStateWithLifecycle()
    val faceEyeOutOfDistributionRisk by
        viewModel.faceEyeOutOfDistributionRisk
            .collectAsStateWithLifecycle()

    val faceEyeShapeConfidence by
        viewModel.faceEyeShapeConfidence.collectAsStateWithLifecycle()
    val faceEyeEyeDynamicConsistency by
        viewModel.faceEyeEyeDynamicConsistency.collectAsStateWithLifecycle()
    val faceEyeHardFailureReasons by
        viewModel.faceEyeHardFailureReasons.collectAsStateWithLifecycle()
    val faceEyeCalibrationDriftScore by
        viewModel.faceEyeCalibrationDriftScore
            .collectAsStateWithLifecycle()
    val faceEyeRecalibrationRecommended by
        viewModel.faceEyeRecalibrationRecommended
            .collectAsStateWithLifecycle()
    val faceEyeCalibrationDriftWarnings by
        viewModel.faceEyeCalibrationDriftWarnings
            .collectAsStateWithLifecycle()

    val adaptiveViewEvidenceGlobalRigTrust by
        viewModel.adaptiveViewEvidenceGlobalRigTrust
            .collectAsStateWithLifecycle()
    val trackLifecycleState by
        viewModel.trackLifecycleState.collectAsStateWithLifecycle()
    val trackLifecycleScore by
        viewModel.trackLifecycleScore.collectAsStateWithLifecycle()
    val trackLifecyclePredictedCount by
        viewModel.trackLifecyclePredictedCount.collectAsStateWithLifecycle()
    val trackLifecycleRecoveringCount by
        viewModel.trackLifecycleRecoveringCount.collectAsStateWithLifecycle()
    val trackLifecycleWarnings by
        viewModel.trackLifecycleWarnings.collectAsStateWithLifecycle()
    val inferenceFps by viewModel.inferenceFps.collectAsStateWithLifecycle()
    val inferenceReason by viewModel.inferenceReason.collectAsStateWithLifecycle()
    val skewNanos by viewModel.skewNanos.collectAsStateWithLifecycle()
    val captureTimingConfidence by viewModel.captureTimingConfidence.collectAsStateWithLifecycle()
    val captureTimingFps by viewModel.captureTimingFps.collectAsStateWithLifecycle()
    val captureTimingJitterMillis by viewModel.captureTimingJitterMillis.collectAsStateWithLifecycle()
    val captureTimingHealthy by viewModel.captureTimingHealthy.collectAsStateWithLifecycle()
    val cameraCaptureQualityScore by viewModel.cameraCaptureQualityScore.collectAsStateWithLifecycle()
    val cameraCaptureQualityWarnings by viewModel.cameraCaptureQualityWarnings.collectAsStateWithLifecycle()
    val cameraOpticalBlurRisk by viewModel.cameraOpticalBlurRisk.collectAsStateWithLifecycle()
    val cameraAnalysisDropRatio by viewModel.cameraAnalysisDropRatio.collectAsStateWithLifecycle()
    val cameraSessionVerification by viewModel.cameraSessionVerification.collectAsStateWithLifecycle()
    val cameraSessionSupported by viewModel.cameraSessionSupported.collectAsStateWithLifecycle()
    val highSpeedSessionVerification by
        viewModel.highSpeedSessionVerification.collectAsStateWithLifecycle()
    val highSpeedSessionSupported by
        viewModel.highSpeedSessionSupported.collectAsStateWithLifecycle()
    val highSpeedSessionActive by
        viewModel.highSpeedSessionActive.collectAsStateWithLifecycle()
    val highSpeedSelectedFps by
        viewModel.highSpeedSelectedFps.collectAsStateWithLifecycle()
    val highSpeedSessionReason by
        viewModel.highSpeedSessionReason.collectAsStateWithLifecycle()
    val hardwareReadoutTimestampSupported by viewModel.hardwareReadoutTimestampSupported.collectAsStateWithLifecycle()
    val latestReadoutTimestampNanos by viewModel.latestReadoutTimestampNanos.collectAsStateWithLifecycle()
    val hardwareReadoutTimingState by viewModel.hardwareReadoutTimingState.collectAsStateWithLifecycle()
    val hardwareReadoutTimingConfidence by viewModel.hardwareReadoutTimingConfidence.collectAsStateWithLifecycle()
    val hardwareReadoutP95JitterMillis by viewModel.hardwareReadoutP95JitterMillis.collectAsStateWithLifecycle()
    val hardwareReadoutMedianDelayMillis by viewModel.hardwareReadoutMedianDelayMillis.collectAsStateWithLifecycle()
    val sessionHealthState by viewModel.sessionHealthState.collectAsStateWithLifecycle()
    val sessionHealthScore by viewModel.sessionHealthScore.collectAsStateWithLifecycle()
    val sessionHealthBottleneck by viewModel.sessionHealthBottleneck.collectAsStateWithLifecycle()
    val captureIntegrityScore by viewModel.captureIntegrityScore.collectAsStateWithLifecycle()
    val captureIntegrityBand by viewModel.captureIntegrityBand.collectAsStateWithLifecycle()
    val captureIntegrityOutputAllowed by viewModel.captureIntegrityOutputAllowed.collectAsStateWithLifecycle()
    val captureAuthorityScore by
        viewModel.captureAuthorityScore.collectAsStateWithLifecycle()
    val captureAuthorityBand by
        viewModel.captureAuthorityBand.collectAsStateWithLifecycle()
    val captureAuthorityHardBlocked by
        viewModel.captureAuthorityHardBlocked.collectAsStateWithLifecycle()
    val captureAuthorityPrecisionQualified by
        viewModel.captureAuthorityPrecisionQualified.collectAsStateWithLifecycle()
    val captureAuthorityWarnings by
        viewModel.captureAuthorityWarnings.collectAsStateWithLifecycle()
    val captureEvidenceTrend by
        viewModel.captureEvidenceTrend.collectAsStateWithLifecycle()
    val captureEvidenceSeverity by
        viewModel.captureEvidenceSeverity.collectAsStateWithLifecycle()
    val captureEvidenceCurrentScore by
        viewModel.captureEvidenceCurrentScore.collectAsStateWithLifecycle()
    val captureEvidenceRecentScore by
        viewModel.captureEvidenceRecentScore.collectAsStateWithLifecycle()
    val captureEvidenceSlope by
        viewModel.captureEvidenceSlope.collectAsStateWithLifecycle()
    val captureEvidenceBottleneck by
        viewModel.captureEvidenceBottleneck.collectAsStateWithLifecycle()
    val captureEvidenceWarnings by
        viewModel.captureEvidenceWarnings.collectAsStateWithLifecycle()
    val adaptiveViewEvidenceScore by
        viewModel.adaptiveViewEvidenceScore.collectAsStateWithLifecycle()
    val adaptiveViewEvidenceDegradedCameraCount by
        viewModel.adaptiveViewEvidenceDegradedCameraCount.collectAsStateWithLifecycle()
    val adaptiveViewEvidenceHoldCameraCount by
        viewModel.adaptiveViewEvidenceHoldCameraCount.collectAsStateWithLifecycle()
    val adaptiveViewEvidenceDominantCamera by
        viewModel.adaptiveViewEvidenceDominantCamera.collectAsStateWithLifecycle()
    val adaptiveViewEvidenceWarnings by
        viewModel.adaptiveViewEvidenceWarnings.collectAsStateWithLifecycle()
    val cameraContributionSelectedCount by
        viewModel.cameraContributionSelectedCount.collectAsStateWithLifecycle()
    val cameraContributionExcludedCount by
        viewModel.cameraContributionExcludedCount.collectAsStateWithLifecycle()
    val cameraContributionDiversitySatisfied by
        viewModel.cameraContributionDiversitySatisfied.collectAsStateWithLifecycle()
    val cameraContributionReason by
        viewModel.cameraContributionReason.collectAsStateWithLifecycle()
    val cameraContributionInformationGainScore by
        viewModel.cameraContributionInformationGainScore
            .collectAsStateWithLifecycle()
    val jointViewSelectionScore by
        viewModel.jointViewSelectionScore.collectAsStateWithLifecycle()
    val jointViewDiversitySatisfiedRatio by
        viewModel.jointViewDiversitySatisfiedRatio.collectAsStateWithLifecycle()
    val jointSelectedObservationCount by
        viewModel.jointSelectedObservationCount.collectAsStateWithLifecycle()
    val jointExcludedObservationCount by
        viewModel.jointExcludedObservationCount.collectAsStateWithLifecycle()
    val jointViewSelectionReasons by
        viewModel.jointViewSelectionReasons.collectAsStateWithLifecycle()
    val rigHealthState by
        viewModel.rigHealthState.collectAsStateWithLifecycle()
    val rigHealthScore by
        viewModel.rigHealthScore.collectAsStateWithLifecycle()

    val rigQuarantinedCameraCount by
        viewModel.rigQuarantinedCameraCount.collectAsStateWithLifecycle()
    val rigHealthWarnings by
        viewModel.rigHealthWarnings.collectAsStateWithLifecycle()
    val rigCameraHealth by
        viewModel.rigCameraHealth.collectAsStateWithLifecycle()
    val spatiotemporalConfidence by
        viewModel.spatiotemporalConfidence.collectAsStateWithLifecycle()
    val spatiotemporalInitialRmsPx by
        viewModel.spatiotemporalInitialRmsPx.collectAsStateWithLifecycle()
    val spatiotemporalFinalRmsPx by
        viewModel.spatiotemporalFinalRmsPx.collectAsStateWithLifecycle()
    val spatiotemporalWarnings by
        viewModel.spatiotemporalWarnings.collectAsStateWithLifecycle()
    val captureIntegrityBottleneck by viewModel.captureIntegrityBottleneck.collectAsStateWithLifecycle()
    val multiCameraSyncConfidence by
        viewModel.multiCameraSyncConfidence.collectAsStateWithLifecycle()
    val multiCameraSyncState by
        viewModel.multiCameraSyncState.collectAsStateWithLifecycle()
    val temporalAlignmentConfidence by
        viewModel.temporalAlignmentConfidence.collectAsStateWithLifecycle()
    val temporalAlignmentSpreadMillis by
        viewModel.temporalAlignmentSpreadMillis.collectAsStateWithLifecycle()
    val temporalAlignmentState by
        viewModel.temporalAlignmentState.collectAsStateWithLifecycle()
    val temporalAlignmentWarnings by
        viewModel.temporalAlignmentWarnings.collectAsStateWithLifecycle()
    val geometricObservabilityScore by
        viewModel.geometricObservabilityScore.collectAsStateWithLifecycle()
    val geometricObservabilityWarnings by
        viewModel.geometricObservabilityWarnings.collectAsStateWithLifecycle()
    val reconstructionUncertaintyConfidence by
        viewModel.reconstructionUncertaintyConfidence.collectAsStateWithLifecycle()
    val reconstructionUncertaintyMeanMeters by
        viewModel.reconstructionUncertaintyMeanMeters.collectAsStateWithLifecycle()
    val reconstructionUncertaintyP95Meters by
        viewModel.reconstructionUncertaintyP95Meters.collectAsStateWithLifecycle()
    val reconstructionTrustedJointRatio by
        viewModel.reconstructionTrustedJointRatio.collectAsStateWithLifecycle()
    val reconstructionUncertaintyBottleneck by
        viewModel.reconstructionUncertaintyBottleneck.collectAsStateWithLifecycle()
    val reconstructionUncertaintyWarnings by
        viewModel.reconstructionUncertaintyWarnings.collectAsStateWithLifecycle()

    val directionalUncertaintyMajor95Meters by
        viewModel.directionalUncertaintyMajor95Meters.collectAsStateWithLifecycle()
    val directionalUncertaintyDepthMeters by
        viewModel.directionalUncertaintyDepthMeters.collectAsStateWithLifecycle()
    val directionalUncertaintyAnisotropy by
        viewModel.directionalUncertaintyAnisotropy.collectAsStateWithLifecycle()
    val directionalUncertaintyConfidence by
        viewModel.directionalUncertaintyConfidence.collectAsStateWithLifecycle()
    val directionalUncertaintyDegradedJointCount by
        viewModel.directionalUncertaintyDegradedJointCount.collectAsStateWithLifecycle()
    val directionalUncertaintyWarnings by
        viewModel.directionalUncertaintyWarnings.collectAsStateWithLifecycle()
    val motionScore by viewModel.motionScore.collectAsStateWithLifecycle()
    val commonModeMotionNormalizedPerSecond by
        viewModel.commonModeMotionNormalizedPerSecond.collectAsStateWithLifecycle()
    val differentialMotionScore by
        viewModel.differentialMotionScore.collectAsStateWithLifecycle()
    val cameraCompensatedMotionScore by
        viewModel.cameraCompensatedMotionScore.collectAsStateWithLifecycle()
    val microMotionScore by
        viewModel.microMotionScore.collectAsStateWithLifecycle()
    val directionChangeRatio by
        viewModel.directionChangeRatio.collectAsStateWithLifecycle()
    val motionMeasurementUncertainty by
        viewModel.motionMeasurementUncertainty.collectAsStateWithLifecycle()
    val motionEvidenceScore by
        viewModel.motionEvidenceScore.collectAsStateWithLifecycle()
    val motionEvidenceConfidence by
        viewModel.motionEvidenceConfidence.collectAsStateWithLifecycle()
    val motionEvidenceWarnings by
        viewModel.motionEvidenceWarnings.collectAsStateWithLifecycle()
    val motionEventType by viewModel.motionEventType.collectAsStateWithLifecycle()
    val motionEventConfidence by viewModel.motionEventConfidence.collectAsStateWithLifecycle()
    val motionDynamicsPhase by
        viewModel.motionDynamicsPhase.collectAsStateWithLifecycle()
    val motionDynamicsScore by
        viewModel.motionDynamicsScore.collectAsStateWithLifecycle()
    val motionContactStability by
        viewModel.motionContactStability.collectAsStateWithLifecycle()
    val motionSymmetryScore by
        viewModel.motionSymmetryScore.collectAsStateWithLifecycle()
    val motionDynamicStress by
        viewModel.motionDynamicStress.collectAsStateWithLifecycle()
    val motionDynamicsUncertainty by
        viewModel.motionDynamicsUncertainty.collectAsStateWithLifecycle()
    val motionTrustedJointRatio by
        viewModel.motionTrustedJointRatio.collectAsStateWithLifecycle()
    val motionMicroMotionRetention by
        viewModel.motionMicroMotionRetention.collectAsStateWithLifecycle()
    val motionLowFrequencyScore by
        viewModel.motionLowFrequencyScore
            .collectAsStateWithLifecycle()
    val motionHighFrequencyScore by
        viewModel.motionHighFrequencyScore
            .collectAsStateWithLifecycle()
    val motionTemporalMemoryScore by
        viewModel.motionTemporalMemoryScore
            .collectAsStateWithLifecycle()
    val motionUncertaintyCoverage by
        viewModel.motionUncertaintyCoverage.collectAsStateWithLifecycle()
    val motionUncertaintyConfidence by
        viewModel.motionUncertaintyConfidence.collectAsStateWithLifecycle()
    val motionUncertaintyOodRisk by
        viewModel.motionUncertaintyOodRisk.collectAsStateWithLifecycle()
    val motionUncertaintyRadiusMeters by
        viewModel.motionUncertaintyRadiusMeters.collectAsStateWithLifecycle()
    val motionUncertaintyCalibrationProgress by
        viewModel.motionUncertaintyCalibrationProgress.collectAsStateWithLifecycle()
    val motionUncertaintyPolicy by
        viewModel.motionUncertaintyPolicy.collectAsStateWithLifecycle()
    val motionUncertaintyWarnings by
        viewModel.motionUncertaintyWarnings.collectAsStateWithLifecycle()
    val motionCustomization by viewModel.motionCustomization.collectAsStateWithLifecycle()
    val rawEnabled by viewModel.rawSidecarEnabled.collectAsStateWithLifecycle()
    val depthEnabled by viewModel.depthSidecarEnabled.collectAsStateWithLifecycle()
    val denseVertices by viewModel.denseMeshVertices.collectAsStateWithLifecycle()
    val denseTriangles by viewModel.denseMeshTriangles.collectAsStateWithLifecycle()
    val denseConfidence by viewModel.denseMeshConfidence.collectAsStateWithLifecycle()
    val liveSkeleton: com.nivukx.nox.core.model.SkeletonFrame? = null
    val liveMotionPreviewFrame by
        viewModel.liveMotionPreviewFrame.collectAsStateWithLifecycle()
    val liveMotionPreviewGpuState by
        viewModel.liveMotionPreviewGpuState.collectAsStateWithLifecycle()
    val sessionPresentationSettings by
        viewModel.sessionPresentationSettings.collectAsStateWithLifecycle()
    val recordingCountdown by
        viewModel.recordingCountdown.collectAsStateWithLifecycle()
    val temporalStability by viewModel.temporalStability.collectAsStateWithLifecycle()
    val denseLandmarkCoverage by
        viewModel.denseLandmarkCoverage.collectAsStateWithLifecycle()
    val denseLandmarkVisibleCoverage by
        viewModel.denseLandmarkVisibleCoverage.collectAsStateWithLifecycle()
    val denseLandmarkUncertaintyP95 by
        viewModel.denseLandmarkUncertaintyP95.collectAsStateWithLifecycle()
    val denseLandmarkCrossViewAgreement by
        viewModel.denseLandmarkCrossViewAgreement.collectAsStateWithLifecycle()
    val denseLandmarkCrossViewEvidenceAvailable by
        viewModel.denseLandmarkCrossViewEvidenceAvailable
            .collectAsStateWithLifecycle()
    val denseLandmarkTemporalStability by
        viewModel.denseLandmarkTemporalStability.collectAsStateWithLifecycle()
    val denseLandmarkContactCoverage by
        viewModel.denseLandmarkContactCoverage.collectAsStateWithLifecycle()
    val denseLandmarkPredictionBridgeRatio by
        viewModel.denseLandmarkPredictionBridgeRatio.collectAsStateWithLifecycle()
    val denseLandmarkRejectedObservations by
        viewModel.denseLandmarkRejectedObservations.collectAsStateWithLifecycle()
    val denseLandmarkStaleObservations by
        viewModel.denseLandmarkStaleObservations.collectAsStateWithLifecycle()
    val denseLandmarkNonFiniteObservations by
        viewModel.denseLandmarkNonFiniteObservations.collectAsStateWithLifecycle()
    val denseLandmarkHardFailure by
        viewModel.denseLandmarkHardFailure.collectAsStateWithLifecycle()
     val denseLandmarkContractScore by
         viewModel.denseLandmarkContractScore.collectAsStateWithLifecycle()
     val denseLandmarkContractState by
         viewModel.denseLandmarkContractState.collectAsStateWithLifecycle()
     val denseLandmarkContractAuthoritative by
         viewModel.denseLandmarkContractAuthoritative.collectAsStateWithLifecycle()
     val denseLandmarkContractFailure by
         viewModel.denseLandmarkContractFailure.collectAsStateWithLifecycle()
    val denseSurfaceTrustedPointRatio by
        viewModel.denseSurfaceTrustedPointRatio.collectAsStateWithLifecycle()
    val denseSurfaceUncertaintyP95Meters by
        viewModel.denseSurfaceUncertaintyP95Meters.collectAsStateWithLifecycle()
    val denseSurfaceTopologyStability by
        viewModel.denseSurfaceTopologyStability.collectAsStateWithLifecycle()
    val denseSurfaceViewDiversity by
        viewModel.denseSurfaceViewDiversity.collectAsStateWithLifecycle()
    val denseSurfaceProbabilisticEvidenceScore by
        viewModel.denseSurfaceProbabilisticEvidenceScore
            .collectAsStateWithLifecycle()
    val denseSurfaceAdaptiveViewSelectionScore by
        viewModel.denseSurfaceAdaptiveViewSelectionScore.collectAsStateWithLifecycle()
    val denseSurfaceAdaptiveViewDiversityScore by
        viewModel.denseSurfaceAdaptiveViewDiversityScore.collectAsStateWithLifecycle()
    val denseSurfaceExcludedWeakViewCount by
        viewModel.denseSurfaceExcludedWeakViewCount.collectAsStateWithLifecycle()
    val denseSurfaceMeanReprojectionErrorPx by
        viewModel.denseSurfaceMeanReprojectionErrorPx
            .collectAsStateWithLifecycle()
    val denseSurfaceHardFailure by
        viewModel.denseSurfaceHardFailure
            .collectAsStateWithLifecycle()
    val denseGeometricConditioningScore by
        viewModel.denseGeometricConditioningScore
            .collectAsStateWithLifecycle()
    val denseGeometricConditioningMedianParallaxDegrees by
        viewModel.denseGeometricConditioningMedianParallaxDegrees
            .collectAsStateWithLifecycle()
    val denseGeometricConditioningMinimumObservationScore by
        viewModel.denseGeometricConditioningMinimumObservationScore
            .collectAsStateWithLifecycle()
    val denseGeometricConditioningWarnings by
        viewModel.denseGeometricConditioningWarnings
            .collectAsStateWithLifecycle()
    val denseIdentityGateQuality by
        viewModel.denseIdentityGateQuality
            .collectAsStateWithLifecycle()
    val denseIdentityGateRejectedCount by
        viewModel.denseIdentityGateRejectedCount
            .collectAsStateWithLifecycle()
    val denseIdentityGateEvaluatedPoints by
        viewModel.denseIdentityGateEvaluatedPoints
            .collectAsStateWithLifecycle()
    val denseIdentityGateWarnings by
        viewModel.denseIdentityGateWarnings
            .collectAsStateWithLifecycle()
    val denseContactConfidence by
        viewModel.denseContactConfidence
            .collectAsStateWithLifecycle()
    val denseContactProtectedLandmarkCount by
        viewModel.denseContactProtectedLandmarkCount
            .collectAsStateWithLifecycle()
    val denseContactWarnings by
        viewModel.denseContactWarnings
            .collectAsStateWithLifecycle()
    val semanticCoverage by viewModel.semanticCoverage.collectAsStateWithLifecycle()
    val attentionConfidence by viewModel.attentionConfidence.collectAsStateWithLifecycle()
    val occlusionRisk by viewModel.occlusionRisk.collectAsStateWithLifecycle()
    val dominantSubject by viewModel.dominantSubject.collectAsStateWithLifecycle()
    val motionClass by viewModel.motionClass.collectAsStateWithLifecycle()
    val semanticMotionScore by
        viewModel.semanticMotionScore.collectAsStateWithLifecycle()
    val cameraMotionRisk by
        viewModel.cameraMotionRisk.collectAsStateWithLifecycle()
    val runtimeBand by viewModel.runtimeBand.collectAsStateWithLifecycle()
    val runtimeQualityScale by
        viewModel.runtimeQualityScale.collectAsStateWithLifecycle()
    val runtimeInferenceScale by
        viewModel.runtimeInferenceScale.collectAsStateWithLifecycle()
    val runtimeProtection by
        viewModel.runtimeProtection.collectAsStateWithLifecycle()
    val runtimeReasons by viewModel.runtimeReasons.collectAsStateWithLifecycle()
    val lightingLux by viewModel.lightingLux.collectAsStateWithLifecycle()
    val colorTemperature by viewModel.colorTemperature.collectAsStateWithLifecycle()
    val gpuBackend by viewModel.gpuBackend.collectAsStateWithLifecycle()
    val concurrentRigAdmissionState by
        viewModel.concurrentRigAdmissionState.collectAsStateWithLifecycle()
    val concurrentRigAdmissionScore by
        viewModel.concurrentRigAdmissionScore.collectAsStateWithLifecycle()
    val concurrentRigSelectedCameraIds by
        viewModel.concurrentRigSelectedCameraIds.collectAsStateWithLifecycle()
    val concurrentRigBlockers by
        viewModel.concurrentRigBlockers.collectAsStateWithLifecycle()
    val concurrentRigWarnings by
        viewModel.concurrentRigWarnings.collectAsStateWithLifecycle()
    val thermalExtrinsicDriftState by
        viewModel.thermalExtrinsicDriftState.collectAsStateWithLifecycle()
    val thermalExtrinsicDriftConfidence by
        viewModel.thermalExtrinsicDriftConfidence.collectAsStateWithLifecycle()
    val thermalExtrinsicTranslationDriftMm by
        viewModel.thermalExtrinsicTranslationDriftMm.collectAsStateWithLifecycle()
    val thermalExtrinsicRotationDriftDegrees by
        viewModel.thermalExtrinsicRotationDriftDegrees.collectAsStateWithLifecycle()
    val thermalExtrinsicTemperatureDeltaCelsius by
        viewModel.thermalExtrinsicTemperatureDeltaCelsius.collectAsStateWithLifecycle()
    val thermalExtrinsicHardFailure by
        viewModel.thermalExtrinsicHardFailure.collectAsStateWithLifecycle()
    val thermalExtrinsicWarnings by
        viewModel.thermalExtrinsicWarnings.collectAsStateWithLifecycle()
    val denseSurfaceMeanAppearanceConsistency by
        viewModel.denseSurfaceMeanAppearanceConsistency.collectAsStateWithLifecycle()
    val denseSurfaceMeanSilhouetteConsistency by
        viewModel.denseSurfaceMeanSilhouetteConsistency.collectAsStateWithLifecycle()
    val denseSurfaceMeanMotionBoundaryConfidence by
        viewModel.denseSurfaceMeanMotionBoundaryConfidence.collectAsStateWithLifecycle()
    val denseSurfaceMeanLightingStability by
        viewModel.denseSurfaceMeanLightingStability.collectAsStateWithLifecycle()
    val showPerformanceTelemetry = true
    val gpuDispatchMillis by viewModel.gpuDispatchMillis.collectAsStateWithLifecycle()
    val reconstructionJoints by viewModel.reconstructionJoints.collectAsStateWithLifecycle()
    val reconstructionError by viewModel.reconstructionErrorPx.collectAsStateWithLifecycle()
    val reconstructionQualityScore by
        viewModel.reconstructionQualityScore.collectAsStateWithLifecycle()
    val worldConsistencyScore by
        viewModel.worldConsistencyScore.collectAsStateWithLifecycle()
    val worldConsistencyWarnings by
        viewModel.worldConsistencyWarnings.collectAsStateWithLifecycle()
    val reconstructionGateReasons by
        viewModel.reconstructionGateReasons.collectAsStateWithLifecycle()
    val face3DPoints by viewModel.face3DPoints.collectAsStateWithLifecycle()
    val face3DError by viewModel.face3DErrorPx.collectAsStateWithLifecycle()
    val eye3DConfidence by viewModel.eye3DConfidence.collectAsStateWithLifecycle()
    val eye3DTemporalConsistencyConfidence by
        viewModel.eye3DTemporalConsistencyConfidence.collectAsStateWithLifecycle()
    val eye3DTemporalConsistencyWarnings by
        viewModel.eye3DTemporalConsistencyWarnings.collectAsStateWithLifecycle()
    val gazeConfidence by viewModel.gazeConfidence.collectAsStateWithLifecycle()
    val gazeStability by viewModel.gazeStability.collectAsStateWithLifecycle()
    val faceEyeConsistencyConfidence by
        viewModel.faceEyeConsistencyConfidence.collectAsStateWithLifecycle()
    val faceEyeConsistencyWarnings by
        viewModel.faceEyeConsistencyWarnings.collectAsStateWithLifecycle()
    val face3DTemporalConsistencyConfidence by
        viewModel.face3DTemporalConsistencyConfidence.collectAsStateWithLifecycle()
    val face3DTemporalConsistencyWarnings by
        viewModel.face3DTemporalConsistencyWarnings.collectAsStateWithLifecycle()
    val faceHeadYaw by viewModel.faceHeadYaw.collectAsStateWithLifecycle()
    val faceHeadRoll by viewModel.faceHeadRoll.collectAsStateWithLifecycle()
    val calibrationReady by viewModel.calibrationReady.collectAsStateWithLifecycle()
    val captureReadiness by viewModel.captureReadiness.collectAsStateWithLifecycle()
    val rigScore by viewModel.rigScore.collectAsStateWithLifecycle()
    val rigReasons by viewModel.rigReasons.collectAsStateWithLifecycle()
    val rigCoverageScore by viewModel.rigCoverageScore.collectAsStateWithLifecycle()
    val denseMeshTemporalConsistencyConfidence by
        viewModel.denseMeshTemporalConsistencyConfidence.collectAsStateWithLifecycle()
    val denseMeshTemporalConsistencyWarnings by
        viewModel.denseMeshTemporalConsistencyWarnings.collectAsStateWithLifecycle()
    val denseMeshSurfaceConfidence by
        viewModel.denseMeshSurfaceConfidence.collectAsStateWithLifecycle()
    val denseTsdfDetailLevel by
        viewModel.denseTsdfDetailLevel.collectAsStateWithLifecycle()
    val denseTsdfComputePressure by
        viewModel.denseTsdfComputePressure.collectAsStateWithLifecycle()
    val denseTsdfSurfaceNormalObservability by
        viewModel.denseTsdfSurfaceNormalObservability
            .collectAsStateWithLifecycle()
    val denseTsdfMeanEvidenceWeight by
        viewModel.denseTsdfMeanEvidenceWeight
            .collectAsStateWithLifecycle()
    val temporalDepthConsistency by
        viewModel.temporalDepthConsistency.collectAsStateWithLifecycle()
    val denseTsdfMeanUncertainty by
        viewModel.denseTsdfMeanUncertainty.collectAsStateWithLifecycle()
    val denseTsdfHighUncertaintyRatio by
        viewModel.denseTsdfHighUncertaintyRatio.collectAsStateWithLifecycle()
    val denseTsdfConfidenceFactor by
        viewModel.denseTsdfConfidenceFactor.collectAsStateWithLifecycle()
    val denseTsdfDecisionReason by
        viewModel.denseTsdfDecisionReason.collectAsStateWithLifecycle()
    val denseMeshExtractionDetail by
        viewModel.denseMeshExtractionDetail.collectAsStateWithLifecycle()
    val denseMeshExtractionPressure by
        viewModel.denseMeshExtractionPressure.collectAsStateWithLifecycle()
    val denseMeshExtractionReason by
        viewModel.denseMeshExtractionReason.collectAsStateWithLifecycle()
    val denseMeshTopologyIntegrity by
        viewModel.denseMeshTopologyIntegrity.collectAsStateWithLifecycle()
    val denseMeshBoundaryEdges by
        viewModel.denseMeshBoundaryEdges.collectAsStateWithLifecycle()
    val denseMeshNonManifoldEdges by
        viewModel.denseMeshNonManifoldEdges.collectAsStateWithLifecycle()
    val denseMeshDuplicateTrianglesRemoved by
        viewModel.denseMeshDuplicateTrianglesRemoved.collectAsStateWithLifecycle()
    val kinematicConstraintScore by
        viewModel.kinematicConstraintScore.collectAsStateWithLifecycle()
    val kinematicConstraintAccepted by
        viewModel.kinematicConstraintAccepted.collectAsStateWithLifecycle()
    val kinematicConstraintBottleneck by
        viewModel.kinematicConstraintBottleneck.collectAsStateWithLifecycle()
    val physicsConstraintScore by
        viewModel.physicsConstraintScore.collectAsStateWithLifecycle()
    val physicsConstraintAccepted by
        viewModel.physicsConstraintAccepted.collectAsStateWithLifecycle()
    val physicsConstraintMeanBoneResidualMeters by
        viewModel.physicsConstraintMeanBoneResidualMeters.collectAsStateWithLifecycle()
    val physicsConstraintP95BoneResidualMeters by
        viewModel.physicsConstraintP95BoneResidualMeters.collectAsStateWithLifecycle()
    val physicsConstraintMaxBoneResidualMeters by
        viewModel.physicsConstraintMaxBoneResidualMeters.collectAsStateWithLifecycle()
    val physicsConstraintSymmetryScore by
        viewModel.physicsConstraintSymmetryScore.collectAsStateWithLifecycle()
    val physicsConstraintFloorContactScore by
        viewModel.physicsConstraintFloorContactScore.collectAsStateWithLifecycle()
    val physicsConstraintCorrectedJointCount by
        viewModel.physicsConstraintCorrectedJointCount.collectAsStateWithLifecycle()
    val physicsConstraintConstrainedEdgeCount by
        viewModel.physicsConstraintConstrainedEdgeCount.collectAsStateWithLifecycle()
    val physicsConstraintWarnings by
        viewModel.physicsConstraintWarnings.collectAsStateWithLifecycle()

    val global4DScore by
        viewModel.global4DScore.collectAsStateWithLifecycle()
    val global4DAccepted by
        viewModel.global4DAccepted.collectAsStateWithLifecycle()
    val global4DState by
        viewModel.global4DState.collectAsStateWithLifecycle()
    val global4DPersistentFailure by
        viewModel.global4DPersistentFailure.collectAsStateWithLifecycle()
    val global4DRootMotionConsistency by
        viewModel.global4DRootMotionConsistency.collectAsStateWithLifecycle()
    val global4DRootVelocityAgreement by
        viewModel.global4DRootVelocityAgreement
            .collectAsStateWithLifecycle()
    val global4DSceneConsistency by
        viewModel.global4DSceneConsistency.collectAsStateWithLifecycle()
    val global4DDirectionalUncertaintyConsistency by
        viewModel.global4DDirectionalUncertaintyConsistency
            .collectAsStateWithLifecycle()
    val global4DFaceEyeConsistency by
        viewModel.global4DFaceEyeConsistency
            .collectAsStateWithLifecycle()
    val global4DExternalContactConsistency by
        viewModel.global4DExternalContactConsistency
            .collectAsStateWithLifecycle()
    val global4DGaussianConsistency by
        viewModel.global4DGaussianConsistency.collectAsStateWithLifecycle()
    val global4DGaussianStrainConsistency by
        viewModel.global4DGaussianStrainConsistency
            .collectAsStateWithLifecycle()
    val global4DGaussianEvidenceAvailable by
        viewModel.global4DGaussianEvidenceAvailable
            .collectAsStateWithLifecycle()
    val global4DWarnings by
        viewModel.global4DWarnings.collectAsStateWithLifecycle()
    val worldSpaceConsistencyScore by
        viewModel.worldSpaceConsistencyScore
            .collectAsStateWithLifecycle()
    val worldSpaceAccepted by
        viewModel.worldSpaceAccepted
            .collectAsStateWithLifecycle()
    val worldSpaceTemporalStability by
        viewModel.worldSpaceTemporalStability
            .collectAsStateWithLifecycle()
    val worldSpaceMotionContinuity by
        viewModel.worldSpaceMotionContinuity
            .collectAsStateWithLifecycle()
    val worldSpaceRejectedJointCount by
        viewModel.worldSpaceRejectedJointCount
            .collectAsStateWithLifecycle()
    val worldSpaceRobustOutlierRatio by
        viewModel.worldSpaceRobustOutlierRatio
            .collectAsStateWithLifecycle()
    val worldSpaceWarnings by
        viewModel.worldSpaceWarnings
            .collectAsStateWithLifecycle()
    val rigRecommendations by viewModel.rigRecommendations.collectAsStateWithLifecycle()
    val error by viewModel.lastError.collectAsStateWithLifecycle()
    val sessions by viewModel.sessionList.collectAsStateWithLifecycle()
    val motionProfile by viewModel.motionProfile.collectAsStateWithLifecycle()
    val motionResponsiveness by viewModel.motionResponsiveness.collectAsStateWithLifecycle()
    val motionSensitivity by viewModel.motionSensitivity.collectAsStateWithLifecycle()
    val predictionHorizon by viewModel.predictionHorizon.collectAsStateWithLifecycle()
    val occlusionPersistence by viewModel.occlusionPersistence.collectAsStateWithLifecycle()
    val reacquisitionRadius by viewModel.reacquisitionRadius.collectAsStateWithLifecycle()
    val rejectAggressiveness by viewModel.rejectAggressiveness.collectAsStateWithLifecycle()
    val jitterSuppression by viewModel.jitterSuppression.collectAsStateWithLifecycle()
    val velocityResponsiveness by viewModel.velocityResponsiveness.collectAsStateWithLifecycle()
    val accelerationResponsiveness by viewModel.accelerationResponsiveness.collectAsStateWithLifecycle()
    val innovationGateElasticity by viewModel.innovationGateElasticity.collectAsStateWithLifecycle()
    val motionThresholdScale by viewModel.motionThresholdScale.collectAsStateWithLifecycle()
    val adaptiveRegimeStrength by viewModel.adaptiveRegimeStrength.collectAsStateWithLifecycle()
    val reconstructionProfile by viewModel.reconstructionProfile.collectAsStateWithLifecycle()
    val reconstructionSoftAccept by viewModel.reconstructionSoftAccept.collectAsStateWithLifecycle()
    val reconstructionHoldFrames by viewModel.reconstructionHoldFrames.collectAsStateWithLifecycle()
    val reconstructionHoldAgeMillis by viewModel.reconstructionHoldAgeMillis.collectAsStateWithLifecycle()
    val reconstructionStabilitySensitivity by
        viewModel.reconstructionStabilitySensitivity.collectAsStateWithLifecycle()
    val temporalPredictionHorizon by
        viewModel.temporalPredictionHorizon.collectAsStateWithLifecycle()
    val temporalInnovationGateScale by
        viewModel.temporalInnovationGateScale.collectAsStateWithLifecycle()
    val temporalUncertaintyGateScale by
        viewModel.temporalUncertaintyGateScale.collectAsStateWithLifecycle()
    val temporalReprojectionGateScale by
        viewModel.temporalReprojectionGateScale.collectAsStateWithLifecycle()
    val temporalSmoothingResponsiveness by
        viewModel.temporalSmoothingResponsiveness.collectAsStateWithLifecycle()
    val temporalConfidenceDecayScale by
        viewModel.temporalConfidenceDecayScale.collectAsStateWithLifecycle()
    val temporalOutlierRetention by
        viewModel.temporalOutlierRetention.collectAsStateWithLifecycle()
    val temporalTopologyToleranceScale by
        viewModel.temporalTopologyToleranceScale.collectAsStateWithLifecycle()

    val productionProfile by
        viewModel.productionProfile.collectAsStateWithLifecycle()
    val productionQualityScore by
        viewModel.productionQualityScore.collectAsStateWithLifecycle()
    val physicalPrecisionCapability by
        viewModel.physicalPrecisionCapability.collectAsStateWithLifecycle()
    val productionPolicy by
        viewModel.productionPolicy.collectAsStateWithLifecycle()
    val productionBottleneck by
        viewModel.productionBottleneck.collectAsStateWithLifecycle()
    val productionBudgetLevel by
        viewModel.productionBudgetLevel.collectAsStateWithLifecycle()
    val productionOutputAllowed by
        viewModel.productionOutputAllowed.collectAsStateWithLifecycle()
    val productionWarnings by
        viewModel.productionWarnings.collectAsStateWithLifecycle()
    val productionDetectorInterval by
        viewModel.productionDetectorInterval.collectAsStateWithLifecycle()
    val productionMeshVertexBudget by
        viewModel.productionMeshVertexBudget.collectAsStateWithLifecycle()
    val productionTsdfStride by
        viewModel.productionTsdfStride.collectAsStateWithLifecycle()
    val productionBundleAdjustmentIterations by
        viewModel.productionBundleAdjustmentIterations.collectAsStateWithLifecycle()

    var tab by remember {
        mutableIntStateOf(0)
    }

    MaterialTheme(
        colorScheme = NoXDarkColors,
    ) {
        LiquidGlassBackdrop {
            BoxWithConstraints(Modifier.fillMaxSize()) {
                val navigationType =
                    NavigationSuiteScaffoldDefaults.navigationSuiteType(
                        currentWindowAdaptiveInfoV2(),
                    )
                val wideLayout =
                    navigationType != NavigationSuiteType.NavigationBar
                Scaffold(
                    containerColor = Color.Transparent,
                    topBar = {
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .height(60.dp)
                                .background(
                                    NoXDarkColors.surface.copy(alpha = 0.78f),
                                )
                                .padding(horizontal = 14.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Image(
                                painter = painterResource(
                                    com.nivukx.nox.R.drawable.nox_logo,
                                ),
                                contentDescription = "NivukxNoX",
                                modifier = Modifier
                                    .width(42.dp)
                                    .height(42.dp)
                                    .clip(RoundedCornerShape(10.dp)),
                                contentScale = ContentScale.Crop,
                            )
                            Spacer(Modifier.width(12.dp))
                            Column {
                                Text(
                                    "NivukxNoX",
                                    style = MaterialTheme.typography.titleMedium,
                                )
                                Text(
                                    "High-End Motion Capture",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    },
                    bottomBar = {
                        if (!wideLayout) {
                            NavigationBar(
                                modifier = Modifier.fillMaxWidth(),
                                containerColor = NoXDarkColors.surface.copy(alpha = 0.78f),
                                tonalElevation = 0.dp,
                            ) {
                                NavigationBarItem(
                                    selected = tab == 0,
                                    onClick = { tab = 0 },
                                    icon = { Text("◉") },
                                    label = { Text("Capture") },
                                )
                                NavigationBarItem(
                                    selected = tab == 1,
                                    onClick = { tab = 1 },
                                    icon = { Text("▤") },
                                    label = { Text("Vault") },
                                )
                                NavigationBarItem(
                                    selected = tab == 2,
                                    onClick = { tab = 2 },
                                    icon = { Text("⌁") },
                                    label = { Text("Diagnostics") },
                                )
                                NavigationBarItem(
                                    selected = tab == 3,
                                    onClick = { tab = 3 },
                                    icon = { Text("⚙") },
                                    label = { Text("Control") },
                                )
                                NavigationBarItem(
                                    selected = tab == 4,
                                    onClick = { tab = 4 },
                                    icon = { Text("✦") },
                                    label = { Text("Studio") },
                                )
                            }
                        }
                    },
                ) { padding ->
                    Row(Modifier.fillMaxSize()) {
                        if (wideLayout) {
                            NavigationRail(
                                containerColor = NoXDarkColors.surface.copy(alpha = 0.62f),
                            ) {
                                NavigationRailItem(
                                    selected = tab == 0,
                                    onClick = { tab = 0 },
                                    icon = { Text("◉") },
                                    label = { Text("Capture") },
                                )
                                NavigationRailItem(
                                    selected = tab == 1,
                                    onClick = { tab = 1 },
                                    icon = { Text("▤") },
                                    label = { Text("Vault") },
                                )
                                NavigationRailItem(
                                    selected = tab == 2,
                                    onClick = { tab = 2 },
                                    icon = { Text("⌁") },
                                    label = { Text("Diagnostics") },
                                )
                                NavigationRailItem(
                                    selected = tab == 3,
                                    onClick = { tab = 3 },
                                    icon = { Text("⚙") },
                                    label = { Text("Control") },
                                )
                                NavigationRailItem(
                                    selected = tab == 4,
                                    onClick = { tab = 4 },
                                    icon = { Text("✦") },
                                    label = { Text("Studio") },
                                )
                            }
                        }
                        Box(
                            Modifier
                                .weight(1f)
                                .fillMaxHeight()
                                .padding(padding),
                        ) {
                            when (tab) {
                    0 -> CaptureScreen(
                        modifier = Modifier.padding(padding),
                        viewModel = viewModel,
                        state = state.name,
                        devices = devices,
                        frames = frameCount,
                        poseCount = poseCount,
                        poseConfidence = poseConfidence,
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
                        handCount = handCount,
                        handConfidence = handConfidence,
                        hand3DConfidence = hand3DConfidence,
                        handLongitudinalAnatomyConsistency =
                            handLongitudinalAnatomyConsistency,
                        handUncertaintyConfidence =
                            handUncertaintyConfidence,
                        handUncertaintyRadiusMeters =
                            handUncertaintyRadiusMeters,
                        handUncertaintyCoverage =
                            handUncertaintyCoverage,
                        handUncertaintyOodRisk =
                            handUncertaintyOodRisk,
                        handUncertaintyPolicy =
                            handUncertaintyPolicy,
                        handUncertaintyWarnings =
                            handUncertaintyWarnings,
                        focusMode = focusMode,
                        focusConfidence = focusConfidence,
                        subjectLockState = subjectLockState,
                        subjectLockConfidence = subjectLockConfidence,
                        inferenceFps = inferenceFps,
                        inferenceReason = inferenceReason,
                        motionScore = motionScore,
                        temporalStability = temporalStability,
                        semanticCoverage = semanticCoverage,
                        attentionConfidence = attentionConfidence,
                        occlusionRisk = occlusionRisk,
                        dominantSubject = dominantSubject,
                        motionClass = motionClass,
                        semanticMotionScore = semanticMotionScore,
                        cameraMotionRisk = cameraMotionRisk,
                        gpuDispatchMillis = gpuDispatchMillis,
                        reconstructionJoints = reconstructionJoints,
                        reconstructionError = reconstructionError,
                        reconstructionQualityScore = reconstructionQualityScore,
                        reconstructionGateReasons = reconstructionGateReasons,
                        face3DPoints = face3DPoints,
                        face3DError = face3DError,
                        eye3DConfidence = eye3DConfidence,
                        eye3DTemporalConsistencyConfidence =
                            eye3DTemporalConsistencyConfidence,
                        eye3DTemporalConsistencyWarnings =
                            eye3DTemporalConsistencyWarnings,
                        rigScore = rigScore,
                        rigReasons = rigReasons,
                        rigCoverageScore = rigCoverageScore,
                        rigRecommendations = rigRecommendations,
                        skewNanos = skewNanos,
                        calibrationReady = calibrationReady,
                        lightingLux = lightingLux,
                        colorTemperature = colorTemperature,
                        gpuBackend = gpuBackend,
                        rawEnabled = rawEnabled,
                        depthEnabled = depthEnabled,
                        denseVertices = denseVertices,
                        denseTriangles = denseTriangles,
                        denseConfidence = denseConfidence,
                         motionCustomization = motionCustomization,
                         liveMotionPreviewFrame = liveMotionPreviewFrame,
                         liveMotionPreviewGpuState = liveMotionPreviewGpuState,
                         runtimeBand = runtimeBand,
                         runtimeQualityScale = runtimeQualityScale,
                         runtimeInferenceScale = runtimeInferenceScale,
                         runtimeProtection = runtimeProtection,
                         runtimeReasons = runtimeReasons,
                                                 error = error,
                    )

                    1 -> SessionsScreen(
                        modifier = Modifier.padding(padding),
                        sessions = sessions,
                    )

                    2 -> DiagnosticsScreen(
                        modifier = Modifier.padding(padding),
                        devices = devices,
                        state = state.name,
                        frames = frameCount,
                        poseCount = poseCount,
                        confidence = poseConfidence,
                        handCount = handCount,
                        handConfidence = handConfidence,
                        hand3DConfidence = hand3DConfidence,
                        focusMode = focusMode,
                        focusConfidence = focusConfidence,
                        inferenceFps = inferenceFps,
                        inferenceReason = inferenceReason,
                        motionScore = motionScore,
                        temporalStability = temporalStability,
                        reconstructionJoints = reconstructionJoints,
                        reconstructionError = reconstructionError,
                        reconstructionQualityScore = reconstructionQualityScore,
                        reconstructionGateReasons = reconstructionGateReasons,
                        face3DPoints = face3DPoints,
                        face3DError = face3DError,
                        eye3DConfidence = eye3DConfidence,
                        skewNanos = skewNanos,
                        calibrationReady = calibrationReady,
                        lightingLux = lightingLux,
                        colorTemperature = colorTemperature,
                        gpuBackend = gpuBackend,
                        rawEnabled = rawEnabled,
                        depthEnabled = depthEnabled,
                        denseVertices = denseVertices,
                        denseTriangles = denseTriangles,
                        denseConfidence = denseConfidence,
                        denseTsdfDetailLevel = denseTsdfDetailLevel,
                        denseTsdfComputePressure = denseTsdfComputePressure,
                        denseTsdfMeanUncertainty = denseTsdfMeanUncertainty,
                        denseTsdfHighUncertaintyRatio = denseTsdfHighUncertaintyRatio,
                        denseTsdfConfidenceFactor = denseTsdfConfidenceFactor,
                        denseTsdfDecisionReason = denseTsdfDecisionReason,
                        denseMeshExtractionDetail = denseMeshExtractionDetail,
                        denseMeshExtractionPressure = denseMeshExtractionPressure,
                        denseMeshExtractionReason = denseMeshExtractionReason,
                        denseMeshTopologyIntegrity =
                            denseMeshTopologyIntegrity,
                        denseMeshBoundaryEdges =
                            denseMeshBoundaryEdges,
                        denseMeshNonManifoldEdges =
                            denseMeshNonManifoldEdges,
                        denseMeshDuplicateTrianglesRemoved =
                            denseMeshDuplicateTrianglesRemoved,
                        identityContinuityScore = identityContinuityScore,
                        identitySwapRisk = identitySwapRisk,
                        identityEvidenceFusionScore = identityEvidenceFusionScore,
                        identitySelectedModalityCount =
                            identitySelectedModalityCount,
                        identityEvidenceWarnings = identityEvidenceWarnings,
                        faceEyeFusionConfidence =
                            faceEyeFusionConfidence,
                        faceEyeSaccadeVelocityDegreesPerSecond =
                            faceEyeSaccadeVelocityDegreesPerSecond,
                        faceEyeFusionAccepted =
                            faceEyeFusionAccepted,
                        faceEyeFusionMode =
                            faceEyeFusionMode,
                        faceEyeIndependentEvidenceSupport =
                            faceEyeIndependentEvidenceSupport,
                        faceEyeReliabilityTrend =
                            faceEyeReliabilityTrend,
                        faceEyeFusionWarnings =
                            faceEyeFusionWarnings,
                        faceEyeUncertaintyCalibrationConfidence =
                            faceEyeUncertaintyCalibrationConfidence,
                        faceEyeUncertaintyCalibrationReady =
                            faceEyeUncertaintyCalibrationReady,
                        faceUncertaintyCalibrationMultiplier =
                            faceUncertaintyCalibrationMultiplier,
                        eyeUncertaintyCalibrationMultiplier =
                            eyeUncertaintyCalibrationMultiplier,
                        faceUncertaintyCoverage =
                            faceUncertaintyCoverage,
                        eyeUncertaintyCoverage =
                            eyeUncertaintyCoverage,
                        faceEyeFailureAttribution =
                            faceEyeFailureAttribution.firstOrNull() ?: "NONE",
                        faceEyeGeometricIntegrity =
                            faceEyeGeometricIntegrity,
                        faceEyeFaceFreshnessScore =
                            faceEyeFaceFreshnessScore,
                        faceEyeGazeAgreementScore =
                            faceEyeGazeAgreementScore,
                        faceEyeRayGeometryCondition =
                            faceEyeRayGeometryCondition.takeIf {
                                frameCount > 0L
                            },
                        faceEyeIpdConsistencyScore =
                            faceEyeIpdConsistencyScore,
                        faceEyeOutOfDistributionRisk =
                            faceEyeOutOfDistributionRisk,
                        faceEyeShapeConfidence =
                            faceEyeShapeConfidence,
                        faceEyeEyeDynamicConsistency =
                            faceEyeEyeDynamicConsistency,
                        faceEyeHardFailureReasons =
                            faceEyeHardFailureReasons,
                        faceEyeCalibrationDriftScore =
                            faceEyeCalibrationDriftScore,
                        faceEyeRecalibrationRecommended =
                            faceEyeRecalibrationRecommended,
                        faceEyeCalibrationDriftWarnings =
                            faceEyeCalibrationDriftWarnings,
                        adaptiveViewEvidenceGlobalRigTrust =
                            adaptiveViewEvidenceGlobalRigTrust,
                        identityAction = identityAction,
                        identityReason = identityReason,
                        trackLifecycleState = trackLifecycleState,
                        trackLifecycleScore = trackLifecycleScore,
                        trackLifecyclePredictedCount =
                            trackLifecyclePredictedCount,
                        trackLifecycleRecoveringCount =
                            trackLifecycleRecoveringCount,
                        trackLifecycleWarnings =
                            trackLifecycleWarnings,
                        rigHealthState = rigHealthState,
                        rigHealthScore = rigHealthScore,
                        rigQuarantinedCameraCount =
                            rigQuarantinedCameraCount,
                        rigHealthWarnings = rigHealthWarnings,
                        rigCameraHealth = rigCameraHealth,
                        cameraContributionSelectedCount =
                            cameraContributionSelectedCount,
                        cameraContributionExcludedCount =
                            cameraContributionExcludedCount,
                        cameraContributionDiversitySatisfied =
                            cameraContributionDiversitySatisfied,
                        cameraContributionReason =
                            cameraContributionReason,
                        cameraContributionInformationGainScore =
                            cameraContributionInformationGainScore,
                        jointViewSelectionScore =
                            jointViewSelectionScore,
                        jointViewDiversitySatisfiedRatio =
                            jointViewDiversitySatisfiedRatio,
                        jointSelectedObservationCount =
                            jointSelectedObservationCount,
                        jointExcludedObservationCount =
                            jointExcludedObservationCount,
                        jointViewSelectionReasons =
                            jointViewSelectionReasons,
                        rigScore = rigScore,
                        rigReasons = rigReasons,
                        rigCoverageScore = rigCoverageScore,
                        rigRecommendations = rigRecommendations,
                        error = error,
                    )

                    3 -> ControlStudioScreen(
                        modifier = Modifier.fillMaxSize(),
                        productionProfile = productionProfile,
                        productionProfiles = viewModel.productionProfiles(),
                        productionQualityScore = productionQualityScore,
                        productionPolicy = productionPolicy,
                        productionBottleneck = productionBottleneck,
                        productionBudgetLevel = productionBudgetLevel,
                        productionOutputAllowed = productionOutputAllowed,
                        productionDetectorInterval = productionDetectorInterval,
                        productionMeshVertexBudget = productionMeshVertexBudget,
                        productionTsdfStride = productionTsdfStride,
                        productionBundleAdjustmentIterations =
                            productionBundleAdjustmentIterations,
                        productionWarnings = productionWarnings,
                        motionCustomization = motionCustomization,
                        liveMotionPreviewFrame = liveMotionPreviewFrame,
                        liveMotionPreviewGpuState = liveMotionPreviewGpuState,
                        runtimeBand = runtimeBand,
                        runtimeQualityScale = runtimeQualityScale,
                        runtimeInferenceScale = runtimeInferenceScale,
                        runtimeProtection = runtimeProtection,
                        runtimeReasons = runtimeReasons,
                        onProductionProfile = {
                            viewModel.applyProductionProfile(it.id)
                        },
                        motionProfile = motionProfile,
                        motionResponsiveness = motionResponsiveness,
                        motionSensitivity = motionSensitivity,
                        predictionHorizon = predictionHorizon,
                        occlusionPersistence = occlusionPersistence,
                        reacquisitionRadius = reacquisitionRadius,
                        rejectAggressiveness = rejectAggressiveness,
                        jitterSuppression = jitterSuppression,
                        reconstructionProfile = reconstructionProfile,
                        reconstructionSoftAccept = reconstructionSoftAccept,
                        reconstructionHoldFrames = reconstructionHoldFrames,
                        reconstructionHoldAgeMillis = reconstructionHoldAgeMillis,
                        reconstructionStabilitySensitivity = reconstructionStabilitySensitivity,
                        onMotionPreset = viewModel::applyMotionPreset,
                        onCustomizationPreset =
                            viewModel::applyCustomizationPreset,
                        onCustomizationBodyDetail = viewModel::setCustomizationBodyDetail,
                        onCustomizationFaceDetail = viewModel::setCustomizationFaceDetail,
                        onCustomizationEyeDetail = viewModel::setCustomizationEyeDetail,
                        onCustomizationEyeTemporalStrictness =
                            viewModel::setCustomizationEyeTemporalStrictness,
                        onCustomizationHandDetail = viewModel::setCustomizationHandDetail,
                        onCustomizationHandTemporalStrictness =
                            viewModel::setCustomizationHandTemporalStrictness,
                        onCustomizationHandGeometryStrictness =
                            viewModel::setCustomizationHandGeometryStrictness,
                        onCustomizationDepthDetail = viewModel::setCustomizationDepthDetail,
                        onCustomizationDenseMeshDetail =
                            viewModel::setCustomizationDenseMeshDetail,
                        onCustomizationContactDetail =
                            viewModel::setCustomizationContactDetail,
                        onCustomizationWorldConsistencyDetail =
                            viewModel::setCustomizationWorldConsistencyDetail,
                        onCustomizationTopologyConstraintStrength =
                            viewModel::setCustomizationTopologyConstraintStrength,
                        onCustomizationMaxSubjects =
                            viewModel::setCustomizationMaxSubjects,
                        onCustomizationOcclusionPredictionFrames =
                            viewModel::setCustomizationOcclusionPredictionFrames,
                        onCustomizationTargetFps =
                            viewModel::setCustomizationTargetFps,
                        onCustomizationInferenceFps =
                            viewModel::setCustomizationInferenceFps,
                        onSessionClipName = viewModel::setSessionClipName,
                        onSessionCountdownSeconds =
                            viewModel::setSessionCountdownSeconds,
                        onRequireCalibrationBeforeRecord =
                            viewModel::setRequireCalibrationBeforeRecord,
                        onMotionResponsiveness = viewModel::setMotionResponsiveness,
                        onMotionSensitivity = viewModel::setMotionSensitivity,
                        onPredictionHorizon = viewModel::setPredictionHorizon,
                        onOcclusionPersistence = viewModel::setOcclusionPersistence,
                        onReacquisitionRadius = viewModel::setReacquisitionRadius,
                        onRejectAggressiveness = viewModel::setRejectAggressiveness,
                        onJitterSuppression = viewModel::setJitterSuppression,
                        onVelocityResponsiveness = viewModel::setVelocityResponsiveness,
                        onAccelerationResponsiveness = viewModel::setAccelerationResponsiveness,
                        onInnovationGateElasticity = viewModel::setInnovationGateElasticity,
                        onMotionThresholdScale = viewModel::setMotionThresholdScale,
                        onAdaptiveRegimeStrength = viewModel::setAdaptiveRegimeStrength,
                        onReconstructionPreset = viewModel::applyReconstructionPreset,
                        onReconstructionSoftAccept = viewModel::setReconstructionSoftAccept,
                        onReconstructionHoldFrames = viewModel::setReconstructionHoldFrames,
                        onReconstructionHoldAgeMillis = viewModel::setReconstructionHoldAgeMillis,
                        onReconstructionStabilitySensitivity =
                            viewModel::setReconstructionStabilitySensitivity,
                        onTemporalPredictionHorizon = viewModel::setTemporalPredictionHorizon,
                        onTemporalInnovationGateScale = viewModel::setTemporalInnovationGateScale,
                        onTemporalUncertaintyGateScale = viewModel::setTemporalUncertaintyGateScale,
                        onTemporalReprojectionGateScale = viewModel::setTemporalReprojectionGateScale,
                        onTemporalSmoothingResponsiveness =
                            viewModel::setTemporalSmoothingResponsiveness,
                        onTemporalConfidenceDecayScale =
                            viewModel::setTemporalConfidenceDecayScale,
                        onTemporalOutlierRetention = viewModel::setTemporalOutlierRetention,
                        onTemporalTopologyToleranceScale =
                            viewModel::setTemporalTopologyToleranceScale,
                        velocityResponsiveness = velocityResponsiveness,
                        accelerationResponsiveness = accelerationResponsiveness,
                        innovationGateElasticity = innovationGateElasticity,
                        motionThresholdScale = motionThresholdScale,
                        adaptiveRegimeStrength = adaptiveRegimeStrength,
                        identityContinuityScore = identityContinuityScore,
                        identitySwapRisk = identitySwapRisk,
                        identityAction = identityAction,
                        temporalPredictionHorizon = temporalPredictionHorizon,
                        temporalInnovationGateScale = temporalInnovationGateScale,
                        temporalUncertaintyGateScale = temporalUncertaintyGateScale,
                        temporalReprojectionGateScale = temporalReprojectionGateScale,
                        temporalSmoothingResponsiveness = temporalSmoothingResponsiveness,
                        temporalConfidenceDecayScale = temporalConfidenceDecayScale,
                        temporalOutlierRetention = temporalOutlierRetention,
                        temporalTopologyToleranceScale = temporalTopologyToleranceScale,
                        motionUncertaintyCoverage = motionUncertaintyCoverage,
                        motionUncertaintyConfidence = motionUncertaintyConfidence,
                        motionUncertaintyOodRisk = motionUncertaintyOodRisk,
                        motionUncertaintyRadiusMeters = motionUncertaintyRadiusMeters,
                        motionUncertaintyCalibrationProgress =
                            motionUncertaintyCalibrationProgress,
                        motionUncertaintyPolicy = motionUncertaintyPolicy,
                        motionUncertaintyWarnings = motionUncertaintyWarnings,
                        motionDynamicsPhase = motionDynamicsPhase,
                        motionDynamicsScore = motionDynamicsScore,
                        motionContactStability = motionContactStability,
                        motionSymmetryScore = motionSymmetryScore,
                        motionDynamicStress = motionDynamicStress,
                        motionDynamicsUncertainty = motionDynamicsUncertainty,
                        motionTrustedJointRatio = motionTrustedJointRatio,
                        motionMicroMotionRetention = motionMicroMotionRetention,
                        physicsConstraintScore = physicsConstraintScore,
                        physicsConstraintAccepted = physicsConstraintAccepted,
                        directionalUncertaintyMajor95Meters =
                            directionalUncertaintyMajor95Meters,
                        directionalUncertaintyDepthMeters =
                            directionalUncertaintyDepthMeters,
                        directionalUncertaintyAnisotropy =
                            directionalUncertaintyAnisotropy,
                        directionalUncertaintyConfidence =
                            directionalUncertaintyConfidence,
                        directionalUncertaintyDegradedJointCount =
                            directionalUncertaintyDegradedJointCount,
                        directionalUncertaintyWarnings =
                            directionalUncertaintyWarnings,
                        physicsConstraintMeanBoneResidualMeters =
                            physicsConstraintMeanBoneResidualMeters,
                        physicsConstraintP95BoneResidualMeters =
                            physicsConstraintP95BoneResidualMeters,
                        physicsConstraintMaxBoneResidualMeters =
                            physicsConstraintMaxBoneResidualMeters,
                        physicsConstraintSymmetryScore =
                            physicsConstraintSymmetryScore,
                        physicsConstraintFloorContactScore =
                            physicsConstraintFloorContactScore,
                        physicsConstraintCorrectedJointCount =
                            physicsConstraintCorrectedJointCount,
                        physicsConstraintConstrainedEdgeCount =
                            physicsConstraintConstrainedEdgeCount,
                        physicsConstraintWarnings =
                            physicsConstraintWarnings,
                        state = state,
                        identityEvidenceFusionScore = identityEvidenceFusionScore,
                        identitySelectedModalityCount = identitySelectedModalityCount,
                        identityEvidenceWarnings = identityEvidenceWarnings,
                        faceEyeFusionConfidence = faceEyeFusionConfidence,
                        faceEyeSaccadeVelocityDegreesPerSecond = faceEyeSaccadeVelocityDegreesPerSecond,
                        faceEyeFusionAccepted = faceEyeFusionAccepted,
                        faceEyeFusionMode = faceEyeFusionMode,
                        faceEyeIndependentEvidenceSupport = faceEyeIndependentEvidenceSupport,
                        faceEyeReliabilityTrend = faceEyeReliabilityTrend,
                        faceEyeFusionWarnings = faceEyeFusionWarnings,
                        faceEyeCaptureDomainConfidence = faceEyeCaptureDomainConfidence,
                        faceEyeLightingConfidence = faceEyeLightingConfidence,
                        faceEyeScaleConfidence = faceEyeScaleConfidence,
                        motionLowFrequencyScore = motionLowFrequencyScore,
                        motionHighFrequencyScore = motionHighFrequencyScore,
                        denseMeshSurfaceConfidence = denseMeshSurfaceConfidence,
                        denseTsdfSurfaceNormalObservability = denseTsdfSurfaceNormalObservability,
                        denseTsdfMeanEvidenceWeight = denseTsdfMeanEvidenceWeight,
                        temporalDepthConsistency = temporalDepthConsistency,
                        global4DScore = global4DScore,
                        global4DAccepted = global4DAccepted,
                        global4DState = global4DState,
                        global4DPersistentFailure = global4DPersistentFailure,
                        global4DRootMotionConsistency = global4DRootMotionConsistency,
                        global4DRootVelocityAgreement = global4DRootVelocityAgreement,
                        global4DSceneConsistency = global4DSceneConsistency,
                        global4DDirectionalUncertaintyConsistency = global4DDirectionalUncertaintyConsistency,
                        global4DFaceEyeConsistency = global4DFaceEyeConsistency,
                        global4DWarnings = global4DWarnings,
                        worldSpaceConsistencyScore = worldSpaceConsistencyScore,
                        worldSpaceAccepted = worldSpaceAccepted,
                        worldSpaceTemporalStability = worldSpaceTemporalStability,
                        worldSpaceMotionContinuity = worldSpaceMotionContinuity,
                        worldSpaceRejectedJointCount = worldSpaceRejectedJointCount,
                        worldSpaceRobustOutlierRatio = worldSpaceRobustOutlierRatio,
                        worldSpaceWarnings = worldSpaceWarnings,
                        error = error,
                    )

                    4 -> UltraMotionStudio(
                        modifier = Modifier.fillMaxSize(),
                        state = state.name,
                        frameCount = frameCount,
                        inferenceFps = inferenceFps,
                        poseConfidence = poseConfidence,
                        reconstructionQuality = reconstructionQualityScore,
                        precisionCapability = physicalPrecisionCapability,
                        worldConsistencyScore = worldConsistencyScore,
                        worldConsistencyWarnings = worldConsistencyWarnings,
                        gazeConfidence = gazeConfidence,
                        eye3DTemporalConsistencyConfidence =
                            eye3DTemporalConsistencyConfidence,
                        eye3DTemporalConsistencyWarnings =
                            eye3DTemporalConsistencyWarnings,
                        faceEyeConsistencyConfidence =
                            faceEyeConsistencyConfidence,
                        faceEyeConsistencyWarnings =
                            faceEyeConsistencyWarnings,
                        face3DTemporalConsistencyConfidence =
                            face3DTemporalConsistencyConfidence,
                        face3DTemporalConsistencyWarnings =
                            face3DTemporalConsistencyWarnings,
                        gazeStability = gazeStability,
                        motionEventType = motionEventType,
                        motionEventConfidence = motionEventConfidence,
                        motionCustomization = motionCustomization,
                        sessionPresentationSettings =
                            sessionPresentationSettings,
                        recordingCountdown = recordingCountdown,
                        onCaptureToggle = viewModel::toggleCapture,
                        postCaptureState = postCaptureState,
                        postCaptureProgress = postCaptureProgress,
                        postCaptureProcessedFrames = postCaptureProcessedFrames,
                        postCaptureMaxActors = postCaptureMaxActors,
                        faceHeadYaw = faceHeadYaw,
                        faceHeadRoll = faceHeadRoll,
                        handTemporalConsistencyConfidence =
                            handTemporalConsistencyConfidence,
                        handTemporalConsistencyWarnings =
                            handTemporalConsistencyWarnings,
                        reconstructionErrorPx = reconstructionError,
                        reconstructionJoints = reconstructionJoints,
                        motionClass = motionClass,
                        motionScore = motionScore,
                        commonModeMotionNormalizedPerSecond =
                            commonModeMotionNormalizedPerSecond,
                        differentialMotionScore =
                            differentialMotionScore,
                        cameraCompensatedMotionScore =
                            cameraCompensatedMotionScore,
                        microMotionScore = microMotionScore,
                        directionChangeRatio = directionChangeRatio,
                        motionMeasurementUncertainty = motionMeasurementUncertainty,
                        motionTemporalMemoryScore =
                            motionTemporalMemoryScore
                                .takeIf { frameCount > 0L },
                        motionEvidenceScore = motionEvidenceScore,
                        motionEvidenceConfidence = motionEvidenceConfidence,
                        motionEvidenceWarnings = motionEvidenceWarnings,
                        temporalStability = temporalStability,
                        occlusionRisk = occlusionRisk,
                        dominantSubject = dominantSubject,
                        subjectLock = subjectLockState,
                        subjectLockConfidence = subjectLockConfidence,
                        gpuBackend = gpuBackend.name,
                        gpuLatencyMs = gpuDispatchMillis,
                        calibrationReady = calibrationReady,
                        captureTimingConfidence = captureTimingConfidence,
                        captureTimingFps = captureTimingFps,
                        captureTimingJitterMillis = captureTimingJitterMillis,
                        captureTimingHealthy = captureTimingHealthy,
                        cameraCaptureQualityScore = cameraCaptureQualityScore,
                        cameraCaptureQualityWarnings = cameraCaptureQualityWarnings,
                         cameraOpticalBlurRisk = cameraOpticalBlurRisk,
                         captureAuthorityScore = captureAuthorityScore,
                         captureAuthorityBand = captureAuthorityBand,
                         captureAuthorityHardBlocked = captureAuthorityHardBlocked,
                         captureAuthorityPrecisionQualified =
                             captureAuthorityPrecisionQualified,
                         captureAuthorityWarnings = captureAuthorityWarnings,
                         cameraAnalysisDropRatio = cameraAnalysisDropRatio,
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
                        latestReadoutTimestampNanos =
                            latestReadoutTimestampNanos,
                        hardwareReadoutTimingState =
                            hardwareReadoutTimingState,
                        hardwareReadoutTimingConfidence =
                            hardwareReadoutTimingConfidence,
                        hardwareReadoutP95JitterMillis =
                            hardwareReadoutP95JitterMillis,
                        hardwareReadoutMedianDelayMillis =
                            hardwareReadoutMedianDelayMillis,
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
                        faceEyeMetricState = faceEyeMetricState,
                        faceEyeMetricConfidence = faceEyeMetricConfidence,
                        faceEyeMetricEyeUncertaintyMm = faceEyeMetricEyeUncertaintyMm,
                        faceEyeMetricWarnings = faceEyeMetricWarnings,
                        faceEyeMetricRecommendations = faceEyeMetricRecommendations,
                        multiPersonIdentityContinuity = multiPersonIdentityContinuity,
                        multiPersonIdentityReidentifications = multiPersonIdentityReidentifications,
                        multiPersonIdentityPredictions = multiPersonIdentityPredictions,
                        multiPerson3DReconstructedCount = multiPerson3DReconstructedCount,
                        multiPerson3DDetailedCount = multiPerson3DDetailedCount,
                        multiPerson3DPredictedCount = multiPerson3DPredictedCount,
                        multiPerson3DPredictionMaxAgeMillis = multiPerson3DPredictionMaxAgeMillis,
                        
                         multiPersonObservedCount = multiPersonObservedCount,
                         multiPersonAdmittedCount = multiPersonAdmittedCount,
                         multiPersonHeldCount = multiPersonHeldCount,
                         multiPersonStarvationRisk = multiPersonStarvationRisk,
                         multiPersonFairnessBoostedCount = multiPersonFairnessBoostedCount,
                         crowdMotionMeanIdentitySafety = crowdMotionMeanIdentitySafety,
                        crowdMotionInteractionPressure = crowdMotionInteractionPressure,
                        crowdMotionDifferentialScore = crowdMotionDifferentialScore,
                        crowdMotionSchedulingCapacity = crowdMotionSchedulingCapacity,
                        crowdWorldLandmarkCoverage = crowdWorldLandmarkCoverage,
                        crowdWorldBoneConsistency = crowdWorldBoneConsistency,
                        crowdWorldGeometryReliability = crowdWorldGeometryReliability,
                        crowdAppearanceQuality = crowdAppearanceQuality,
                        crowdSegmentationEvidence = crowdSegmentationEvidence,
                        crowdSegmentationCoverage = crowdSegmentationCoverage,
                        crowdActorSilhouetteSupport = crowdActorSilhouetteSupport,
                        crowdActorSilhouetteAmbiguity = crowdActorSilhouetteAmbiguity,
                        personSegmentationCoverage =
                            personSegmentationCoverage,
                        personSegmentationConfidence = personSegmentationConfidence,
                        personSegmentationBoundaryStability = personSegmentationBoundaryStability,
                        personSegmentationLatencyMs = personSegmentationLatencyMs,
                        personSegmentationState = personSegmentationState,
                        personSegmentationDelegate = personSegmentationDelegate,
                        personSegmentationFallbackCount = personSegmentationFallbackCount,
                        crowdMotionBottleneck = crowdMotionBottleneck,
                        crowdMotionHeldCount = crowdMotionHeldCount,
                        crowdMotionPredictedCount = crowdMotionPredictedCount,
                        crowdMotionWarnings = crowdMotionWarnings,
                        sessionHealthState = sessionHealthState,
                        sessionHealthScore = sessionHealthScore,
                        sessionHealthBottleneck = sessionHealthBottleneck,
                        captureIntegrityScore = captureIntegrityScore,
                        captureIntegrityBand = captureIntegrityBand,
                        captureIntegrityOutputAllowed = captureIntegrityOutputAllowed,
                        captureEvidenceTrend =
                            captureEvidenceTrend,
                        captureEvidenceSeverity =
                            captureEvidenceSeverity,
                        captureEvidenceCurrentScore =
                            captureEvidenceCurrentScore,
                        captureEvidenceRecentScore =
                            captureEvidenceRecentScore,
                        captureEvidenceSlope =
                            captureEvidenceSlope,
                        captureEvidenceBottleneck =
                            captureEvidenceBottleneck,
                        captureEvidenceWarnings =
                            captureEvidenceWarnings,
                        adaptiveViewEvidenceScore =
                            adaptiveViewEvidenceScore,
                        adaptiveViewEvidenceDegradedCameraCount =
                            adaptiveViewEvidenceDegradedCameraCount,
                        adaptiveViewEvidenceHoldCameraCount =
                            adaptiveViewEvidenceHoldCameraCount,
                        adaptiveViewEvidenceDominantCamera =
                            adaptiveViewEvidenceDominantCamera,
                        adaptiveViewEvidenceWarnings =
                            adaptiveViewEvidenceWarnings,
                        
                         rigQuarantinedCameraCount = rigQuarantinedCameraCount,
                         spatiotemporalConfidence =
                            spatiotemporalConfidence,
                        spatiotemporalInitialRmsPx =
                            spatiotemporalInitialRmsPx,
                        spatiotemporalFinalRmsPx =
                            spatiotemporalFinalRmsPx,
                        spatiotemporalWarnings =
                            spatiotemporalWarnings,
                        captureIntegrityBottleneck = captureIntegrityBottleneck,
                        multiCameraSyncConfidence = multiCameraSyncConfidence,
                        multiCameraSyncState = multiCameraSyncState,
                        temporalAlignmentConfidence = temporalAlignmentConfidence,
                        temporalAlignmentSpreadMillis = temporalAlignmentSpreadMillis,
                        temporalAlignmentState = temporalAlignmentState,
                        temporalAlignmentWarnings = temporalAlignmentWarnings,
                        geometricObservabilityScore =
                            geometricObservabilityScore,
                        geometricObservabilityWarnings =
                            geometricObservabilityWarnings,
                        geometricInformationConditionScore =
                            viewModel.geometricInformationConditionScore.value,
                        geometricInformationP95MajorMeters =
                            viewModel.geometricInformationP95MajorMeters.value,
                        geometricInformationP95DepthMeters =
                            viewModel.geometricInformationP95DepthMeters.value,
                        reconstructionUncertaintyConfidence =
                            reconstructionUncertaintyConfidence,
                        reconstructionUncertaintyMeanMeters =
                            reconstructionUncertaintyMeanMeters,
                        reconstructionUncertaintyP95Meters =
                            reconstructionUncertaintyP95Meters,
                        reconstructionTrustedJointRatio =
                            reconstructionTrustedJointRatio,
                        reconstructionUncertaintyBottleneck =
                            reconstructionUncertaintyBottleneck,
                        reconstructionUncertaintyWarnings =
                            reconstructionUncertaintyWarnings,
                        captureReadiness = captureReadiness,
                        rigCoverageScore = rigCoverageScore,
                        denseMeshTemporalConsistencyConfidence =
                            denseMeshTemporalConsistencyConfidence,
                        denseMeshTemporalConsistencyWarnings =
                            denseMeshTemporalConsistencyWarnings,
                        denseMeshSurfaceConfidence =
                            denseMeshSurfaceConfidence,
                        denseLandmarkCoverage =
                            denseLandmarkCoverage,
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
                            viewModel.denseSurfaceMeanDynamicProbability.value,
                        denseSurfaceMeanTemporalStability =
                            viewModel.denseSurfaceMeanTemporalStability.value,
                        denseSurfaceTrustedGaussianRatio =
                            viewModel.denseSurfaceTrustedGaussianRatio.value,
                        denseSurfaceHardFailure =
                            denseSurfaceHardFailure,
                        denseIdentityGateQuality =
                            denseIdentityGateQuality,
                        denseIdentityGateRejectedCount =
                            denseIdentityGateRejectedCount,
                        denseIdentityGateEvaluatedPoints =
                            denseIdentityGateEvaluatedPoints,
                        denseIdentityGateWarnings =
                            denseIdentityGateWarnings,
                        denseGeometricConditioningScore =
                            denseGeometricConditioningScore,
                        denseGeometricConditioningMinimumObservationScore =
                            denseGeometricConditioningMinimumObservationScore,
                        denseGeometricConditioningMedianParallaxDegrees =
                            denseGeometricConditioningMedianParallaxDegrees,
                        denseGeometricConditioningWarnings =
                            denseGeometricConditioningWarnings,
                        denseContactConfidence =
                            denseContactConfidence,
                        denseContactProtectedLandmarkCount =
                            denseContactProtectedLandmarkCount,
                        denseContactWarnings =
                            denseContactWarnings,
                        global4DExternalContactConsistency =
                            global4DExternalContactConsistency,
                        global4DGaussianConsistency =
                            global4DGaussianConsistency,
                        global4DGaussianStrainConsistency =
                            global4DGaussianStrainConsistency,
                        global4DGaussianEvidenceAvailable =
                            global4DGaussianEvidenceAvailable,
                        kinematicConstraintScore =
                            kinematicConstraintScore,
                        kinematicConstraintAccepted =
                            kinematicConstraintAccepted,
                        kinematicConstraintBottleneck =
                            kinematicConstraintBottleneck,
                        rigScore = rigScore,
                        lightingLux = lightingLux,
                        colorTemperature = colorTemperature,
                        skeleton = liveSkeleton,
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
                        reconstructionProfile = reconstructionProfile,
                        reconstructionSoftAccept = reconstructionSoftAccept,
                        reconstructionHoldFrames = reconstructionHoldFrames,
                        reconstructionHoldAgeMillis = reconstructionHoldAgeMillis,
                        reconstructionStabilitySensitivity = reconstructionStabilitySensitivity,
                        temporalPredictionHorizon = temporalPredictionHorizon,
                        temporalInnovationGateScale = temporalInnovationGateScale,
                        temporalUncertaintyGateScale = temporalUncertaintyGateScale,
                        temporalReprojectionGateScale = temporalReprojectionGateScale,
                        temporalSmoothingResponsiveness =
                            temporalSmoothingResponsiveness,
                        temporalConfidenceDecayScale = temporalConfidenceDecayScale,
                        temporalOutlierRetention = temporalOutlierRetention,
                        temporalTopologyToleranceScale = temporalTopologyToleranceScale,
                        onMotionPreset = viewModel::applyMotionPreset,
                        onCustomizationPreset = viewModel::applyCustomizationPreset,
                        onCustomizationBodyDetail =
                            viewModel::setCustomizationBodyDetail,
                        onCustomizationFaceDetail =
                            viewModel::setCustomizationFaceDetail,
                        onCustomizationEyeDetail =
                            viewModel::setCustomizationEyeDetail,
                        onCustomizationHandDetail =
                            viewModel::setCustomizationHandDetail,
                        onCustomizationDepthDetail =
                            viewModel::setCustomizationDepthDetail,
                        onCustomizationDenseMeshDetail =
                            viewModel::setCustomizationDenseMeshDetail,
                        onCustomizationContactDetail =
                            viewModel::setCustomizationContactDetail,
                        onCustomizationWorldConsistencyDetail =
                            viewModel::setCustomizationWorldConsistencyDetail,
                        onCustomizationTopologyConstraintStrength =
                            viewModel::setCustomizationTopologyConstraintStrength,
                        onCustomizationMaxSubjects =
                            viewModel::setCustomizationMaxSubjects,
                        onCustomizationOcclusionPredictionFrames =
                            viewModel::setCustomizationOcclusionPredictionFrames,
                        onCustomizationTargetFps =
                            viewModel::setCustomizationTargetFps,
                        onCustomizationInferenceFps =
                            viewModel::setCustomizationInferenceFps,
                        onCustomizationMicroMotionPreservation =
                            viewModel::setCustomizationMicroMotionPreservation,
                        onCustomizationCameraMotionSuppression =
                            viewModel::setCustomizationCameraMotionSuppression,
                        onCustomizationFaceEyeFusionWeight =
                            viewModel::setCustomizationFaceEyeFusionWeight,
                         onCustomizationEyeTemporalStrictness =
                             viewModel::setCustomizationEyeTemporalStrictness,
                         onCustomizationHandTemporalStrictness =
                             viewModel::setCustomizationHandTemporalStrictness,
                         onCustomizationHandGeometryStrictness =
                             viewModel::setCustomizationHandGeometryStrictness,
                        onCustomizationFaceEyeFusionStrictness =
                            viewModel::setCustomizationFaceEyeFusionStrictness,
                        onCustomizationTemporalAlignmentStrictness =
                            viewModel::setCustomizationTemporalAlignmentStrictness,
                         onCustomizationOutputIntegrityStrictness =
                             viewModel::setCustomizationOutputIntegrityStrictness,
                         onCustomizationPhysicsConstraintStrictness =
                             viewModel::setCustomizationPhysicsConstraintStrictness,
                         onCustomizationGlobal4DConsistencyStrictness =
                             viewModel::setCustomizationGlobal4DConsistencyStrictness,
                         onCustomizationSurfaceEvidenceStrictness =
                             viewModel::setCustomizationSurfaceEvidenceStrictness,
                         onCustomizationMinimumParallaxDegrees =
                             viewModel::setCustomizationMinimumParallaxDegrees,
                         onCustomizationPreferredTemporalSkewMillis =
                             viewModel::setCustomizationPreferredTemporalSkewMillis,
                         onCustomizationMaximumTemporalSkewMillis =
                             viewModel::setCustomizationMaximumTemporalSkewMillis,
                         onCustomizationMaximumReprojectionErrorPx =
                             viewModel::setCustomizationMaximumReprojectionErrorPx,
                         onCustomizationMaximumReconstructionUncertaintyMeters =
                             viewModel::setCustomizationMaximumReconstructionUncertaintyMeters,
                        onAdvancedMotionCustomization =
                            viewModel::updateAdvancedMotionCustomization,
                        onMotionResponsiveness = viewModel::setMotionResponsiveness,
                        onMotionSensitivity = viewModel::setMotionSensitivity,
                        onPredictionHorizon = viewModel::setPredictionHorizon,
                        onOcclusionPersistence = viewModel::setOcclusionPersistence,
                        onJitterSuppression = viewModel::setJitterSuppression,
                        onReacquisitionRadius = viewModel::setReacquisitionRadius,
                        onRejectAggressiveness = viewModel::setRejectAggressiveness,
                        onVelocityResponsiveness = viewModel::setVelocityResponsiveness,
                        onAccelerationResponsiveness = viewModel::setAccelerationResponsiveness,
                        onInnovationGateElasticity = viewModel::setInnovationGateElasticity,
                        onMotionThresholdScale = viewModel::setMotionThresholdScale,
                        onReconstructionPreset = viewModel::applyReconstructionPreset,
                        onReconstructionSoftAccept = viewModel::setReconstructionSoftAccept,
                        onReconstructionHoldFrames = viewModel::setReconstructionHoldFrames,
                        onReconstructionHoldAgeMillis = viewModel::setReconstructionHoldAgeMillis,
                        onReconstructionStabilitySensitivity =
                            viewModel::setReconstructionStabilitySensitivity,
                        onTemporalPredictionHorizon = viewModel::setTemporalPredictionHorizon,
                        onTemporalInnovationGateScale = viewModel::setTemporalInnovationGateScale,
                        onTemporalUncertaintyGateScale = viewModel::setTemporalUncertaintyGateScale,
                        onTemporalReprojectionGateScale = viewModel::setTemporalReprojectionGateScale,
                        onTemporalSmoothingResponsiveness =
                            viewModel::setTemporalSmoothingResponsiveness,
                        onTemporalConfidenceDecayScale =
                            viewModel::setTemporalConfidenceDecayScale,
                        onTemporalOutlierRetention = viewModel::setTemporalOutlierRetention,
                        onTemporalTopologyToleranceScale =
                            viewModel::setTemporalTopologyToleranceScale,
                        onSessionClipName = viewModel::setSessionClipName,
                        onSessionCountdownSeconds =
                            viewModel::setSessionCountdownSeconds,
                        onRequireCalibrationBeforeRecord =
                            viewModel::setRequireCalibrationBeforeRecord,
                    )

                                        else -> DiagnosticsScreen(
                        modifier = Modifier.padding(padding),
                        devices = devices,
                        state = state.name,
                        frames = frameCount,
                        poseCount = poseCount,
                        confidence = poseConfidence,
                        handCount = handCount,
                        handConfidence = handConfidence,
                        hand3DConfidence = hand3DConfidence,
                        focusMode = focusMode,
                        focusConfidence = focusConfidence,
                        inferenceFps = inferenceFps,
                        inferenceReason = inferenceReason,
                        motionScore = motionScore,
                        temporalStability = temporalStability,
                        reconstructionJoints = reconstructionJoints,
                        reconstructionError = reconstructionError,
                        reconstructionQualityScore = reconstructionQualityScore,
                        reconstructionGateReasons = reconstructionGateReasons,
                        face3DPoints = face3DPoints,
                        face3DError = face3DError,
                        eye3DConfidence = eye3DConfidence,
                        skewNanos = skewNanos,
                        calibrationReady = calibrationReady,
                        lightingLux = lightingLux,
                        colorTemperature = colorTemperature,
                        gpuBackend = gpuBackend,
                        rawEnabled = rawEnabled,
                        depthEnabled = depthEnabled,
                        denseVertices = denseVertices,
                        denseTriangles = denseTriangles,
                        denseConfidence = denseConfidence,
                        denseTsdfDetailLevel = denseTsdfDetailLevel,
                        denseTsdfComputePressure = denseTsdfComputePressure,
                        denseTsdfMeanUncertainty = denseTsdfMeanUncertainty,
                        denseTsdfHighUncertaintyRatio = denseTsdfHighUncertaintyRatio,
                        denseTsdfConfidenceFactor = denseTsdfConfidenceFactor,
                        denseTsdfDecisionReason = denseTsdfDecisionReason,
                        denseMeshExtractionDetail = denseMeshExtractionDetail,
                        denseMeshExtractionPressure = denseMeshExtractionPressure,
                        denseMeshExtractionReason = denseMeshExtractionReason,
                        denseMeshTopologyIntegrity =
                            denseMeshTopologyIntegrity,
                        denseMeshBoundaryEdges =
                            denseMeshBoundaryEdges,
                        denseMeshNonManifoldEdges =
                            denseMeshNonManifoldEdges,
                        denseMeshDuplicateTrianglesRemoved =
                            denseMeshDuplicateTrianglesRemoved,
                        identityContinuityScore = identityContinuityScore,
                        identitySwapRisk = identitySwapRisk,
                        identityEvidenceFusionScore = identityEvidenceFusionScore,
                        identitySelectedModalityCount =
                            identitySelectedModalityCount,
                        identityEvidenceWarnings = identityEvidenceWarnings,
                        faceEyeFusionConfidence =
                            faceEyeFusionConfidence,
                        faceEyeSaccadeVelocityDegreesPerSecond =
                            faceEyeSaccadeVelocityDegreesPerSecond,
                        faceEyeFusionAccepted =
                            faceEyeFusionAccepted,
                        faceEyeFusionMode =
                            faceEyeFusionMode,
                        faceEyeIndependentEvidenceSupport =
                            faceEyeIndependentEvidenceSupport,
                        faceEyeReliabilityTrend =
                            faceEyeReliabilityTrend,
                        faceEyeFusionWarnings =
                            faceEyeFusionWarnings,
                        faceEyeUncertaintyCalibrationConfidence =
                            faceEyeUncertaintyCalibrationConfidence,
                        faceEyeUncertaintyCalibrationReady =
                            faceEyeUncertaintyCalibrationReady,
                        faceUncertaintyCalibrationMultiplier =
                            faceUncertaintyCalibrationMultiplier,
                        eyeUncertaintyCalibrationMultiplier =
                            eyeUncertaintyCalibrationMultiplier,
                        faceUncertaintyCoverage =
                            faceUncertaintyCoverage,
                        eyeUncertaintyCoverage =
                            eyeUncertaintyCoverage,
                        faceEyeFailureAttribution =
                            faceEyeFailureAttribution.firstOrNull() ?: "NONE",
                        faceEyeGeometricIntegrity =
                            faceEyeGeometricIntegrity,
                        faceEyeFaceFreshnessScore =
                            faceEyeFaceFreshnessScore,
                        faceEyeGazeAgreementScore =
                            faceEyeGazeAgreementScore,
                        faceEyeRayGeometryCondition =
                            faceEyeRayGeometryCondition.takeIf {
                                frameCount > 0L
                            },
                        faceEyeIpdConsistencyScore =
                            faceEyeIpdConsistencyScore,
                        faceEyeOutOfDistributionRisk =
                            faceEyeOutOfDistributionRisk,
                        faceEyeShapeConfidence =
                            faceEyeShapeConfidence,
                        faceEyeEyeDynamicConsistency =
                            faceEyeEyeDynamicConsistency,
                        faceEyeHardFailureReasons =
                            faceEyeHardFailureReasons,
                        faceEyeCalibrationDriftScore =
                            faceEyeCalibrationDriftScore,
                        faceEyeRecalibrationRecommended =
                            faceEyeRecalibrationRecommended,
                        faceEyeCalibrationDriftWarnings =
                            faceEyeCalibrationDriftWarnings,
                        adaptiveViewEvidenceGlobalRigTrust =
                            adaptiveViewEvidenceGlobalRigTrust,
                        identityAction = identityAction,
                        identityReason = identityReason,
                        trackLifecycleState = trackLifecycleState,
                        trackLifecycleScore = trackLifecycleScore,
                        trackLifecyclePredictedCount =
                            trackLifecyclePredictedCount,
                        trackLifecycleRecoveringCount =
                            trackLifecycleRecoveringCount,
                        trackLifecycleWarnings =
                            trackLifecycleWarnings,
                        rigHealthState = rigHealthState,
                        rigHealthScore = rigHealthScore,
                        rigQuarantinedCameraCount =
                            rigQuarantinedCameraCount,
                        rigHealthWarnings = rigHealthWarnings,
                        rigCameraHealth = rigCameraHealth,
                        cameraContributionSelectedCount =
                            cameraContributionSelectedCount,
                        cameraContributionExcludedCount =
                            cameraContributionExcludedCount,
                        cameraContributionDiversitySatisfied =
                            cameraContributionDiversitySatisfied,
                        cameraContributionReason =
                            cameraContributionReason,
                        cameraContributionInformationGainScore =
                            cameraContributionInformationGainScore,
                        jointViewSelectionScore =
                            jointViewSelectionScore,
                        jointViewDiversitySatisfiedRatio =
                            jointViewDiversitySatisfiedRatio,
                        jointSelectedObservationCount =
                            jointSelectedObservationCount,
                        jointExcludedObservationCount =
                            jointExcludedObservationCount,
                        jointViewSelectionReasons =
                            jointViewSelectionReasons,
                        rigScore = rigScore,
                        rigReasons = rigReasons,
                        rigCoverageScore = rigCoverageScore,
                        rigRecommendations = rigRecommendations,
                        error = error,
                    )
                        }
                    }
                }
            }
        }
    }
    }
}

@Composable
private fun ControlStudioScreen(
    modifier: Modifier,
    productionProfile: ProductionCaptureProfile,
    productionProfiles: List<ProductionCaptureProfile>,
    productionQualityScore: Double,
    productionPolicy: String,
    productionBottleneck: String,
    productionBudgetLevel: String,
    productionOutputAllowed: Boolean,
    productionDetectorInterval: Int,
    productionMeshVertexBudget: Int,
    productionTsdfStride: Int,
    productionBundleAdjustmentIterations: Int,
    productionWarnings: List<String>,
    onProductionProfile: (ProductionCaptureProfile) -> Unit,
    motionProfile: String,
    motionResponsiveness: Double,
    motionSensitivity: Double,
    predictionHorizon: Double,
    occlusionPersistence: Double,
    reacquisitionRadius: Double,
    rejectAggressiveness: Double,
    jitterSuppression: Double,
    reconstructionProfile: String,
    reconstructionSoftAccept: Double,
    reconstructionHoldFrames: Int,
    reconstructionHoldAgeMillis: Long,
    reconstructionStabilitySensitivity: Double,
    onMotionPreset: (String) -> Unit,
    onMotionResponsiveness: (Double) -> Unit,
    onMotionSensitivity: (Double) -> Unit,
    onPredictionHorizon: (Double) -> Unit,
    onOcclusionPersistence: (Double) -> Unit,
    onReacquisitionRadius: (Double) -> Unit,
    onRejectAggressiveness: (Double) -> Unit,
    onJitterSuppression: (Double) -> Unit,
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
    velocityResponsiveness: Double,
    accelerationResponsiveness: Double,
    innovationGateElasticity: Double,
    motionThresholdScale: Double,
    identityContinuityScore: Double,
    identitySwapRisk: Double,
    identityAction: String,
    temporalPredictionHorizon: Double,
    temporalInnovationGateScale: Double,
    temporalUncertaintyGateScale: Double,
    temporalReprojectionGateScale: Double,
    temporalSmoothingResponsiveness: Double,
    temporalConfidenceDecayScale: Double,
    temporalOutlierRetention: Double,
    temporalTopologyToleranceScale: Double,
    motionUncertaintyCoverage: Double,
    motionUncertaintyConfidence: Double,
    motionUncertaintyOodRisk: Double,
    motionUncertaintyRadiusMeters: Double,
    motionUncertaintyCalibrationProgress: Double,
    motionUncertaintyPolicy: String,
    motionUncertaintyWarnings: List<String>,
    motionDynamicsPhase: String,
    motionDynamicsScore: Double,
    motionContactStability: Double,
    motionSymmetryScore: Double,
    motionDynamicStress: Double,
    motionDynamicsUncertainty: Double,
    motionTrustedJointRatio: Double,
    motionMicroMotionRetention: Double,
    physicsConstraintScore: Double,
    physicsConstraintAccepted: Boolean,
    physicsConstraintMeanBoneResidualMeters: Double,
    physicsConstraintP95BoneResidualMeters: Double,
    physicsConstraintMaxBoneResidualMeters: Double,
    physicsConstraintSymmetryScore: Double,
    physicsConstraintFloorContactScore: Double,
    physicsConstraintCorrectedJointCount: Int,
    physicsConstraintConstrainedEdgeCount: Int,
    physicsConstraintWarnings: List<String>,

    motionCustomization: com.nivukx.nox.core.model.MotionCaptureCustomization,
    liveMotionPreviewFrame: MotionFieldPreviewFrame? = null,
    liveMotionPreviewGpuState: com.nivukx.nox.core.model.MotionFieldGpuState = com.nivukx.nox.core.model.MotionFieldGpuState(),
    runtimeBand: String = "NOMINAL",
    runtimeQualityScale: Double = 1.0,
    runtimeInferenceScale: Double = 1.0,
    runtimeProtection: Boolean = false,
    runtimeReasons: List<String> = emptyList(),
    onCustomizationPreset: (String) -> Unit,
    onCustomizationBodyDetail: (Double) -> Unit,
    onCustomizationFaceDetail: (Double) -> Unit,
    onCustomizationEyeDetail: (Double) -> Unit,
    onCustomizationEyeTemporalStrictness: (Double) -> Unit,
    onCustomizationHandDetail: (Double) -> Unit,
    onCustomizationHandTemporalStrictness: (Double) -> Unit,
    onCustomizationHandGeometryStrictness: (Double) -> Unit,
    onCustomizationDepthDetail: (Double) -> Unit,
    onCustomizationDenseMeshDetail: (Double) -> Unit,
    onCustomizationContactDetail: (Double) -> Unit,
    onCustomizationWorldConsistencyDetail: (Double) -> Unit,
    onCustomizationTopologyConstraintStrength: (Double) -> Unit,
    onCustomizationMaxSubjects: (Int) -> Unit,
    onCustomizationOcclusionPredictionFrames: (Int) -> Unit,
    onCustomizationTargetFps: (Int) -> Unit,
    onCustomizationInferenceFps: (Int) -> Unit,
    onSessionClipName: (String) -> Unit,
    onSessionCountdownSeconds: (Int) -> Unit,
    onRequireCalibrationBeforeRecord: (Boolean) -> Unit,
    onAdaptiveRegimeStrength: (Double) -> Unit,
    adaptiveRegimeStrength: Double,
    directionalUncertaintyMajor95Meters: Double,
    directionalUncertaintyDepthMeters: Double,
    directionalUncertaintyAnisotropy: Double,
    directionalUncertaintyConfidence: Double,
    directionalUncertaintyDegradedJointCount: Int,
    directionalUncertaintyWarnings: List<String>,
    state: CaptureState,
    identityEvidenceFusionScore: Double,
    identitySelectedModalityCount: Int,
    identityEvidenceWarnings: List<String>,
    faceEyeFusionConfidence: Double,
    faceEyeSaccadeVelocityDegreesPerSecond: Double,
    faceEyeFusionAccepted: Boolean,
    faceEyeFusionMode: String,
    faceEyeIndependentEvidenceSupport: Double,
    faceEyeReliabilityTrend: Double,
    faceEyeFusionWarnings: List<String>,
    faceEyeCaptureDomainConfidence: Double,
    faceEyeLightingConfidence: Double,
    faceEyeScaleConfidence: Double,
    motionLowFrequencyScore: Double,
    motionHighFrequencyScore: Double,
    denseMeshSurfaceConfidence: Double,
    denseTsdfSurfaceNormalObservability: Double,
    denseTsdfMeanEvidenceWeight: Double,
    temporalDepthConsistency: Double,
    global4DScore: Double,
    global4DAccepted: Boolean,
    global4DState: String,
    global4DPersistentFailure: Boolean,
    global4DRootMotionConsistency: Double,
    global4DRootVelocityAgreement: Double,
    global4DSceneConsistency: Double,
    global4DDirectionalUncertaintyConsistency: Double,
    global4DFaceEyeConsistency: Double,
    global4DWarnings: List<String>,
    worldSpaceConsistencyScore: Double,
    worldSpaceAccepted: Boolean,
    worldSpaceTemporalStability: Double,
    worldSpaceMotionContinuity: Double,
    worldSpaceRejectedJointCount: Int,
    worldSpaceRobustOutlierRatio: Double,
    worldSpaceWarnings: List<String>,
    error: String?,) {
    LazyColumn(
        modifier = modifier.padding(
            horizontal = 14.dp,
            vertical = 16.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = PaddingValues(bottom = 24.dp),
    ) {
        item {
            SectionTitle(
                "CONTROL STUDIO",
                "Tune motion capture and 3D continuity without leaving the viewfinder workflow.",
            )
        }

        item {
            ReconstructionIntelligencePanel(
                productionQuality = productionQualityScore,
                productionPolicy = productionPolicy,
                productionBottleneck = productionBottleneck,
                productionOutputAllowed = productionOutputAllowed,
                motionPhase = motionDynamicsPhase,
                motionDynamicsScore = motionDynamicsScore,
                dynamicStress = motionDynamicStress,
                identityContinuity = identityContinuityScore,
                identitySwapRisk = identitySwapRisk,
                identityAction = identityAction,
                identityEvidenceFusionScore =
                    identityEvidenceFusionScore,
                identitySelectedModalityCount =
                    identitySelectedModalityCount,
                identityEvidenceWarnings =
                    identityEvidenceWarnings,
                faceEyeFusionConfidence =
                    faceEyeFusionConfidence,
                faceEyeFusionMode = faceEyeFusionMode,
                faceEyeIndependentEvidenceSupport =
                    faceEyeIndependentEvidenceSupport,
                faceEyeReliabilityTrend = faceEyeReliabilityTrend,
                faceEyeSaccadeVelocityDegreesPerSecond =
                    faceEyeSaccadeVelocityDegreesPerSecond,
                faceEyeFusionAccepted = faceEyeFusionAccepted,
                faceEyeFusionWarnings = faceEyeFusionWarnings,
                faceEyeCaptureDomainConfidence =
                    faceEyeCaptureDomainConfidence,
                faceEyeLightingConfidence =
                    faceEyeLightingConfidence,
                faceEyeScaleConfidence =
                    faceEyeScaleConfidence,
                surfaceConfidence = denseMeshSurfaceConfidence,
                denseTsdfSurfaceNormalObservability =
                    denseTsdfSurfaceNormalObservability,
                denseTsdfMeanEvidenceWeight =
                    denseTsdfMeanEvidenceWeight,
                temporalDepthConsistency =
                    temporalDepthConsistency,
                trustedJointRatio = motionTrustedJointRatio,
                physicsConstraintScore = physicsConstraintScore,
                physicsConstraintAccepted = physicsConstraintAccepted,
                physicsConstraintMeanBoneResidualMeters =
                    physicsConstraintMeanBoneResidualMeters,
                physicsConstraintP95BoneResidualMeters =
                    physicsConstraintP95BoneResidualMeters,
                physicsConstraintMaxBoneResidualMeters =
                    physicsConstraintMaxBoneResidualMeters,
                physicsConstraintSymmetryScore =
                    physicsConstraintSymmetryScore,
                physicsConstraintFloorContactScore =
                    physicsConstraintFloorContactScore,
                physicsConstraintCorrectedJointCount =
                    physicsConstraintCorrectedJointCount,
                physicsConstraintConstrainedEdgeCount =
                    physicsConstraintConstrainedEdgeCount,
                physicsConstraintWarnings =
                    physicsConstraintWarnings,
                global4DScore = global4DScore,
                global4DAccepted = global4DAccepted,
                global4DState = global4DState,
                global4DPersistentFailure =
                    global4DPersistentFailure,
                global4DRootMotionConsistency =
                    global4DRootMotionConsistency,
                global4DRootVelocityAgreement =
                    global4DRootVelocityAgreement,
                global4DSceneConsistency =
                    global4DSceneConsistency,
                global4DDirectionalUncertaintyConsistency =
                    global4DDirectionalUncertaintyConsistency,
                global4DFaceEyeConsistency =
                    global4DFaceEyeConsistency,
                global4DWarnings = global4DWarnings,
                worldSpaceConsistencyScore =
                    worldSpaceConsistencyScore,
                worldSpaceAccepted =
                    worldSpaceAccepted,
                worldSpaceTemporalStability =
                    worldSpaceTemporalStability,
                worldSpaceMotionContinuity =
                    worldSpaceMotionContinuity,
                worldSpaceRejectedJointCount =
                    worldSpaceRejectedJointCount,
                worldSpaceRobustOutlierRatio =
                    worldSpaceRobustOutlierRatio,
                worldSpaceWarnings =
                    worldSpaceWarnings,
            )
        }

        item {
            MotionUncertaintyPanel(
                coverage = motionUncertaintyCoverage,
                confidence = motionUncertaintyConfidence,
                oodRisk = motionUncertaintyOodRisk,
                radiusMeters = motionUncertaintyRadiusMeters,
                calibrationProgress = motionUncertaintyCalibrationProgress,
                policy = motionUncertaintyPolicy,
                warnings = motionUncertaintyWarnings,
            )
        }

        item {
            MotionDynamicsPanel(
                phase = motionDynamicsPhase,
                score = motionDynamicsScore,
                contactStability = motionContactStability,
                symmetry = motionSymmetryScore,
                dynamicStress = motionDynamicStress,
                uncertainty = motionDynamicsUncertainty,
                trustedJointRatio = motionTrustedJointRatio,
                microMotionRetention = motionMicroMotionRetention,
                lowFrequencyMotionScore = motionLowFrequencyScore,
                highFrequencyMotionScore = motionHighFrequencyScore,
            )
        }

        item {
            ProductionProfilePanel(
                profile = productionProfile,
                profiles = productionProfiles,
                qualityScore = productionQualityScore,
                policy = productionPolicy,
                bottleneck = productionBottleneck,
                budgetLevel = productionBudgetLevel,
                outputAllowed = productionOutputAllowed,
                detectorInterval = productionDetectorInterval,
                meshVertexBudget = productionMeshVertexBudget,
                tsdfStride = productionTsdfStride,
                bundleAdjustmentIterations =
                    productionBundleAdjustmentIterations,
                warnings = productionWarnings,
                onSelect = onProductionProfile,
            )
        }

        item {
            GlassCard(Modifier.fillMaxWidth(), intensity = 0.98f) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    SectionTitle(
                        "Motion profile",
                        "Presets are starting points. Every parameter remains adjustable.",
                    )
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        listOf("CINEMATIC", "ACTION", "PRECISION").forEach { preset ->
                            GlassAction(
                                text = preset,
                                primary = motionProfile == preset,
                            ) {
                                onMotionPreset(preset)
                            }
                        }
                    }
                    Text(
                        "Current: " + motionProfile,
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item {
            ControlSlider(
                title = "Responsiveness",
                description = "How quickly the tracker follows intentional movement.",
                value = motionResponsiveness.toFloat(),
                range = 0.20f..1.20f,
                valueText = "%.2f".format(motionResponsiveness),
                onValueChange = { onMotionResponsiveness(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Motion sensitivity",
                description = "Amplifies motion energy used by capture decisions and prioritization.",
                value = motionSensitivity.toFloat(),
                range = 0.50f..1.80f,
                valueText = "%.2fx".format(motionSensitivity),
                onValueChange = { onMotionSensitivity(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Prediction horizon",
                description = "Maximum short-term prediction window used during detector gaps.",
                value = predictionHorizon.toFloat(),
                range = 0.04f..0.35f,
                valueText = "%.0f ms".format(predictionHorizon * 1000.0),
                onValueChange = { onPredictionHorizon(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Occlusion persistence",
                description = "How long a temporarily hidden subject remains recoverable.",
                value = occlusionPersistence.toFloat(),
                range = 0.50f..2.50f,
                valueText = "%.2fx".format(occlusionPersistence),
                onValueChange = { onOcclusionPersistence(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Reacquisition radius",
                description = "Spatial window used when an identity reappears after a short loss.",
                value = reacquisitionRadius.toFloat(),
                range = 0.03f..0.22f,
                valueText = "%.0f%%".format(reacquisitionRadius * 100.0),
                onValueChange = { onReacquisitionRadius(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Reject aggressiveness",
                description = "Higher values reject implausible detector jumps more aggressively.",
                value = rejectAggressiveness.toFloat(),
                range = 0.20f..1.00f,
                valueText = "%.2f".format(rejectAggressiveness),
                onValueChange = { onRejectAggressiveness(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Jitter suppression",
                description = "Controls how strongly small temporal detector noise is damped.",
                value = jitterSuppression.toFloat(),
                range = 0.00f..1.00f,
                valueText = "%.2f".format(jitterSuppression),
                onValueChange = { onJitterSuppression(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Velocity responsiveness",
                description = "Adjusts how quickly learned 2D velocity adapts to real detector motion.",
                value = velocityResponsiveness.toFloat(),
                range = 0.55f..1.60f,
                valueText = "%.2fx".format(velocityResponsiveness),
                onValueChange = { onVelocityResponsiveness(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Acceleration responsiveness",
                description = "Controls adaptation speed for second-order motion used during rapid movement.",
                value = accelerationResponsiveness.toFloat(),
                range = 0.55f..1.60f,
                valueText = "%.2fx".format(accelerationResponsiveness),
                onValueChange = { onAccelerationResponsiveness(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Innovation gate elasticity",
                description = "Expands or tightens robust residual gating before a detector jump is rejected.",
                value = innovationGateElasticity.toFloat(),
                range = 0.65f..1.60f,
                valueText = "%.2fx".format(innovationGateElasticity),
                onValueChange = { onInnovationGateElasticity(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Motion classification scale",
                description = "Shifts static, normal, fast and extreme motion thresholds without changing the estimator.",
                value = motionThresholdScale.toFloat(),
                range = 0.70f..1.40f,
                valueText = "%.2fx".format(motionThresholdScale),
                onValueChange = { onMotionThresholdScale(it.toDouble()) },
            )
        }

        item {
            GlassCard(Modifier.fillMaxWidth(), intensity = 0.96f) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(9.dp),
                ) {
                    SectionTitle(
                        "3D reconstruction policy",
                        "Controls presentation continuity after geometric and temporal validation.",
                    )
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        listOf(
                            "CINEMATIC" to "Stable",
                            "ACTION" to "Responsive",
                            "PRECISION" to "Strict",
                        ).forEach { (preset, _) ->
                            GlassAction(
                                text = preset,
                                primary = reconstructionProfile == preset,
                            ) {
                                onReconstructionPreset(preset)
                            }
                        }
                    }
                }
            }
        }

        item {
            ControlSlider(
                title = "3D soft accept threshold",
                description = "Minimum combined reconstruction confidence required for a normal commit.",
                value = reconstructionSoftAccept.toFloat(),
                range = 0.30f..0.95f,
                valueText = "%.0f%%".format(reconstructionSoftAccept * 100.0),
                onValueChange = { onReconstructionSoftAccept(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Hold-last budget",
                description = "How many recent frames may remain on the last trusted reconstruction.",
                value = reconstructionHoldFrames.toFloat(),
                range = 0f..12f,
                valueText = reconstructionHoldFrames.toString() + " frames",
                steps = 11,
                onValueChange = {
                    onReconstructionHoldFrames(
                        it.toInt().coerceIn(0, 12),
                    )
                },
            )
        }

        item {
            ControlSlider(
                title = "Hold age",
                description = "Maximum age of a trusted reconstruction before the UI must reject it.",
                value = reconstructionHoldAgeMillis.toFloat(),
                range = 40f..600f,
                valueText = reconstructionHoldAgeMillis.toString() + " ms",
                steps = 11,
                onValueChange = {
                    onReconstructionHoldAgeMillis(
                        it.toLong().coerceIn(40L, 600L),
                    )
                },
            )
        }

        item {
            ControlSlider(
                title = "Temporal stability sensitivity",
                description = "Weights recent continuity evidence when deciding whether to hold or commit.",
                value = reconstructionStabilitySensitivity.toFloat(),
                range = 0.20f..1.20f,
                valueText = "%.2f".format(reconstructionStabilitySensitivity),
                onValueChange = {
                    onReconstructionStabilitySensitivity(it.toDouble())
                },
            )
        }

        item {
            GlassCard(Modifier.fillMaxWidth(), intensity = 0.96f) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(9.dp),
                ) {
                    SectionTitle(
                        "Temporal 3D model",
                        "Tune the geometric stabilizer itself, not just the final hold/commit layer.",
                    )
                    Text(
                        "All controls remain bounded by the reconstruction safety envelope.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item {
            ControlSlider(
                title = "3D prediction horizon",
                description = "Maximum short-term geometric extrapolation during camera or landmark gaps.",
                value = temporalPredictionHorizon.toFloat(),
                range = 0.02f..0.60f,
                valueText = "%.0f ms".format(temporalPredictionHorizon * 1000.0),
                onValueChange = { onTemporalPredictionHorizon(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Innovation gate scale",
                description = "Controls the spatial residual gate used before a 3D observation is accepted.",
                value = temporalInnovationGateScale.toFloat(),
                range = 0.65f..1.80f,
                valueText = "%.2fx".format(temporalInnovationGateScale),
                onValueChange = { onTemporalInnovationGateScale(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Uncertainty gate scale",
                description = "Controls how measured positional uncertainty widens the temporal acceptance gate.",
                value = temporalUncertaintyGateScale.toFloat(),
                range = 0.50f..1.80f,
                valueText = "%.2fx".format(temporalUncertaintyGateScale),
                onValueChange = { onTemporalUncertaintyGateScale(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Reprojection gate scale",
                description = "Adjusts tolerance for image-space reprojection error before temporal rejection.",
                value = temporalReprojectionGateScale.toFloat(),
                range = 0.50f..1.80f,
                valueText = "%.2fx".format(temporalReprojectionGateScale),
                onValueChange = { onTemporalReprojectionGateScale(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "3D smoothing responsiveness",
                description = "Balances low-latency response against temporal noise suppression in the 3D state.",
                value = temporalSmoothingResponsiveness.toFloat(),
                range = 0.08f..0.92f,
                valueText = "%.2f".format(temporalSmoothingResponsiveness),
                onValueChange = { onTemporalSmoothingResponsiveness(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Confidence decay scale",
                description = "Controls how quickly confidence falls while a joint is predicted instead of observed.",
                value = temporalConfidenceDecayScale.toFloat(),
                range = 0.50f..2.00f,
                valueText = "%.2fx".format(temporalConfidenceDecayScale),
                onValueChange = { onTemporalConfidenceDecayScale(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Outlier retention",
                description = "Keeps a tightly bounded fraction of rejected temporal evidence for continuity analysis.",
                value = temporalOutlierRetention.toFloat(),
                range = 0.00f..0.35f,
                valueText = "%.2f".format(temporalOutlierRetention),
                onValueChange = { onTemporalOutlierRetention(it.toDouble()) },
            )
        }

        item {
            ControlSlider(
                title = "Topology tolerance",
                description = "Scales uncertainty-aware link constraints used to preserve skeleton proportions.",
                value = temporalTopologyToleranceScale.toFloat(),
                range = 0.50f..1.80f,
                valueText = "%.2fx".format(temporalTopologyToleranceScale),
                onValueChange = { onTemporalTopologyToleranceScale(it.toDouble()) },
            )
        }

        item {
            GlassCard(Modifier.fillMaxWidth()) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    Text(
                        "Production guardrails",
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        "These controls tune estimator behaviour. They do not bypass calibration, " +
                            "reprojection checks, synchronization limits, temporal rejection, or " +
                            "hardware capability checks.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        "Presets are persistent across launches and can be fine-tuned at any time.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }
    }
}

@Composable
private fun ControlSlider(
    title: String,
    description: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    valueText: String,
    steps: Int = 0,
    onValueChange: (Float) -> Unit,
) {
    GlassCard(Modifier.fillMaxWidth(), intensity = 0.92f) {
        Column(
            Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(5.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    Modifier.weight(1f),
                ) {
                    Text(
                        title,
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Text(
                        description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    valueText,
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            Slider(
                value = value.coerceIn(range.start, range.endInclusive),
                onValueChange = onValueChange,
                valueRange = range,
                steps = steps,
            )
        }
    }
}

@Composable
private fun CaptureScreen(
    modifier: Modifier,
    viewModel: NoXViewModel,
    state: String,
    devices: List<com.nivukx.nox.core.model.DeviceDescriptor>,
    frames: Long,
    poseCount: Int,
    poseConfidence: Float,
    multiPersonObservedCount: Int,
    multiPersonAdmittedCount: Int,
    multiPersonHeldCount: Int,
    multiPersonStarvationRisk: Double,
    multiPersonFairnessBoostedCount: Int,
    handCount: Int,
    handConfidence: Double,
    hand3DConfidence: Double,
    handLongitudinalAnatomyConsistency: Double = 0.78,
    handUncertaintyConfidence: Double = 1.0,
    handUncertaintyRadiusMeters: Double = 0.0,
    handUncertaintyCoverage: Double = 0.0,
    handUncertaintyOodRisk: Double = 0.0,
    handUncertaintyPolicy: String = "NORMAL",
    handUncertaintyWarnings: List<String> = emptyList(),
    focusMode: String,
    focusConfidence: Double,
    subjectLockState: String,
    subjectLockConfidence: Double,
    inferenceFps: Int,
    inferenceReason: String,
    motionScore: Double,
    temporalStability: Double,
    semanticCoverage: Double,
    attentionConfidence: Double,
    occlusionRisk: Double,
    dominantSubject: String,
    motionClass: String,
    semanticMotionScore: Double,
    cameraMotionRisk: Double,
    gpuDispatchMillis: Double,
    reconstructionJoints: Int,
    reconstructionError: Double,
    reconstructionQualityScore: Double,
    reconstructionGateReasons: List<String>,
    face3DPoints: Int,
    face3DError: Double,
    eye3DConfidence: Double,
    eye3DTemporalConsistencyConfidence: Double,
    eye3DTemporalConsistencyWarnings: List<String>,
    rigScore: Double,
    rigReasons: List<String> = emptyList(),
    rigCoverageScore: Double = 0.0,
    rigRecommendations: List<String> = emptyList(),
    skewNanos: Long,
    calibrationReady: Boolean,
    lightingLux: Double,
    colorTemperature: Double,
    gpuBackend: GpuBackend,
    rawEnabled: Boolean,
    depthEnabled: Boolean,
    denseVertices: Int,
    denseTriangles: Int,
    denseConfidence: Double,
    motionCustomization: MotionCaptureCustomization,
     liveMotionPreviewFrame: MotionFieldPreviewFrame? = null,
     liveMotionPreviewGpuState: MotionFieldGpuState = MotionFieldGpuState(),
     runtimeBand: String = "NOMINAL",
     runtimeQualityScale: Double = 1.0,
     runtimeInferenceScale: Double = 1.0,
     runtimeProtection: Boolean = false,
     runtimeReasons: List<String> = emptyList(),
     error: String?,
) {
    val stability by animateFloatAsState(
        targetValue = temporalStability
            .toFloat()
            .coerceIn(0f, 1f),
        animationSpec = tween(450),
        label = "stability",
    )

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Transparent),
        contentPadding = PaddingValues(
            horizontal = 14.dp,
            vertical = 16.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(13.dp),
    ) {
        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SectionTitle(
                    "NOX / MOTION CAPTURE",
                    "Distributed multi-camera body, hand, face and eye reconstruction",
                )
                StatusPill(
                    text = state,
                    healthy = state != "ERROR",
                )
            }
        }

        item {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(
                        rememberScrollState(),
                    ),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                TelemetryChip(
                    "NODES",
                    devices.count { it.connected }.toString(),
                )
                TelemetryChip(
                    "HANDS",
                    handCount.toString(),
                )
                TelemetryChip(
                    "AF",
                    focusMode,
                )
                TelemetryChip(
                    "AI",
                    inferenceFps.toString() + " FPS",
                )
                TelemetryChip(
                    "DEPTH",
                    if (depthEnabled) "DEPTH16" else "MULTI-VIEW",
                )
                TelemetryChip(
                    "GPU",
                    gpuBackend.name,
                )
            }
        }

        item {
            GlassCard(
                modifier = Modifier.fillMaxWidth(),
                selected = state == "RECORDING",
                intensity = 0.98f,
            ) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(350.dp)
                        .clip(
                            RoundedCornerShape(28.dp),
                        ),
                ) {
                    CameraPreview(
                        viewModel = viewModel,
                        modifier = Modifier.fillMaxSize(),
                        liveMotionPreviewEnabled =
                            motionCustomization.liveMotionPreviewEnabled,
                        liveMotionPreviewFrame = liveMotionPreviewFrame,
                    )

                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(110.dp)
                            .background(
                                Brush.verticalGradient(
                                    listOf(
                                        Color.Black.copy(alpha = 0.50f),
                                        Color.Transparent,
                                    ),
                                ),
                            ),
                    )

                    Row(
                        Modifier
                            .fillMaxWidth()
                            .padding(15.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        StatusPill(
                            if (rawEnabled) "RAW READY" else "RAW OFF",
                            rawEnabled,
                        )

                        StatusPill(
                            if (calibrationReady) "RIG LOCKED" else "RIG OPEN",
                            calibrationReady,
                        )
                    }

                    Surface(
                        Modifier
                            .align(Alignment.BottomStart)
                            .padding(14.dp),
                        shape = RoundedCornerShape(14.dp),
                        color = Color.Black.copy(
                            alpha = 0.64f,
                        ),
                        tonalElevation = 0.dp,
                    ) {
                        Column(
                            Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 8.dp,
                            ),
                            verticalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            Text(
                                frames.toString() +
                                    "F  •  " +
                                    poseCount.toString() +
                                    " KP  •  " +
                                    reconstructionJoints.toString() +
                                    " 3D",
                                color = Color.White,
                                style = MaterialTheme.typography.labelMedium,
                            )
                            if (motionCustomization.liveMotionPreviewEnabled) {
                                val liveFrame = liveMotionPreviewFrame
                                Text(
                                    "LIVE  •  " +
                                        (liveFrame?.actorCount ?: 0) +
                                        " ACTORS  •  " +
                                        (liveMotionPreviewGpuState.backend.name) +
                                        "  •  " +
                                        "%.1f".format(
                                            liveMotionPreviewGpuState.renderLatencyMillis,
                                        ) +
                                        " ms",
                                    color = Color(0xFFD8FFF5),
                                    style = MaterialTheme.typography.labelSmall,
                                )
                                Text(
                                    when {
                                        liveFrame?.stale == true ->
                                            "MOTION STATE STALE"
                                        liveFrame?.predictedToDisplay == true ->
                                            "DISPLAY PREDICTION ACTIVE"
                                        else ->
                                            "SENSOR-TIMED MOTION"
                                    },
                                    color = Color(0xFFB8C8E8),
                                    style = MaterialTheme.typography.labelSmall,
                                )
                            }
                        }
                    }
                }
            }
        }

        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(9.dp),
            ) {
                GlassMetric(
                    "Motion",
                    (motionScore * 100).toInt().toString() + "%",
                    "dynamic energy",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Stability",
                    (stability * 100).toInt().toString() + "%",
                    "temporal",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Hands",
                    handCount.toString(),
                    (handConfidence * 100.0).toInt().toString() + "% conf",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "3D",
                    if (reconstructionJoints == 0) "—"
                    else reconstructionJoints.toString(),
                    (hand3DConfidence * 100.0).toInt().toString() + "% hand 3D",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Anatomy",
                    "%.0f%%".format(
                        handLongitudinalAnatomyConsistency * 100.0,
                    ),
                    "longitudinal",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Hand uncertainty",
                    "%.0f%%".format(
                        handUncertaintyConfidence * 100.0,
                    ),
                    handUncertaintyPolicy +
                        " / %.0f mm".format(
                            handUncertaintyRadiusMeters * 1000.0,
                        ),
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Focus",
                    focusMode,
                    (focusConfidence * 100.0).toInt().toString() + "% target",
                    Modifier.weight(1f),
                )
            }
        }

        item {
            GlassGauge(
                label = "Rig confidence",
                value = rigScore.toFloat(),
                modifier = Modifier.fillMaxWidth(),
            )
        }

        item {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                GlassTelemetryTile(
                    label = "SUBJECT",
                    value = dominantSubject,
                    detail = motionClass,
                    active = dominantSubject != "NONE",
                    modifier = Modifier.width(152.dp),
                )
                GlassTelemetryTile(
                    label = "SUBJECT LOCK",
                    value = subjectLockState,
                    detail = ((subjectLockConfidence * 100.0)
                        .toInt()
                        .coerceIn(0, 100))
                        .toString() + "% confidence",
                    active = subjectLockState == "LOCKED",
                    modifier = Modifier.width(166.dp),
                )
                GlassTelemetryTile(
                    label = "ATTENTION",
                    value = ((attentionConfidence * 100.0).toInt().coerceIn(0, 100)).toString() + "%",
                    detail = "semantic " +
                        ((semanticCoverage * 100.0).toInt().coerceIn(0, 100)).toString() + "%",
                    active = attentionConfidence >= 0.60,
                    modifier = Modifier.width(152.dp),
                )
                GlassTelemetryTile(
                    label = "SEMANTIC MOTION",
                    value = ((semanticMotionScore * 100.0)
                        .toInt()
                        .coerceIn(0, 100))
                        .toString() + "%",
                    detail = motionClass,
                    active = semanticMotionScore >= 0.20,
                    modifier = Modifier.width(166.dp),
                )
                GlassTelemetryTile(
                    label = "CAMERA MOTION",
                    value = ((cameraMotionRisk * 100.0)
                        .toInt()
                        .coerceIn(0, 100))
                        .toString() + "%",
                    detail = "common-mode risk",
                    active = cameraMotionRisk < 0.35,
                    modifier = Modifier.width(166.dp),
                )
                GlassTelemetryTile(
                    label = "RUNTIME",
                    value = runtimeBand,
                    detail = ((runtimeQualityScale * 100.0)
                        .toInt()
                        .coerceIn(0, 110)).toString() + "% budget",
                    active = !runtimeProtection,
                    modifier = Modifier.width(166.dp),
                )
                GlassTelemetryTile(
                    label = "INFERENCE",
                    value = ((runtimeInferenceScale * 100.0)
                        .toInt()
                        .coerceIn(0, 110)).toString() + "%",
                    detail = if (runtimeReasons.isEmpty()) {
                        "adaptive stable"
                    } else {
                        runtimeReasons.first()
                    },
                    active = runtimeInferenceScale >= 0.60,
                    modifier = Modifier.width(166.dp),
                )
                GlassTelemetryTile(
                    label = "OCCLUSION",
                    value = ((occlusionRisk * 100.0).toInt().coerceIn(0, 100)).toString() + "%",
                    detail = "motion-aware",
                    active = occlusionRisk < 0.35,
                    modifier = Modifier.width(152.dp),
                )
                GlassTelemetryTile(
                    label = "GPU",
                    value = if (gpuDispatchMillis > 0.0) {
                        "%.1f ms".format(gpuDispatchMillis)
                    } else {
                        "READY"
                    },
                    detail = gpuBackend.name,
                    active = gpuBackend != GpuBackend.CPU,
                    modifier = Modifier.width(152.dp),
                )
            }
        }

        item {
            GlassCard(
                Modifier.fillMaxWidth(),
            ) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    SectionTitle(
                        "Motion Core",
                        "Live acquisition, perception and metric reconstruction",
                    )
                    MetricRow(
                        "Lighting",
                        if (lightingLux > 0.0) {
                            "%.0f lux".format(lightingLux)
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Color temperature",
                        if (colorTemperature > 0.0) {
                            "%.0f K".format(colorTemperature)
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Hands",
                        handCount.toString() +
                            " / " +
                            (handConfidence * 100.0).toInt().toString() +
                            "% confidence",
                    )
                    MetricRow(
                        "Hand 3D",
                        (hand3DConfidence * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "AI focus",
                        focusMode +
                            " / " +
                            (focusConfidence * 100.0).toInt().toString() +
                            "%",
                    )
                    MetricRow(
                        "Subject lock",
                        subjectLockState +
                            " / " +
                            (subjectLockConfidence * 100.0).toInt().toString() +
                            "%",
                    )
                    MetricRow(
                        "GPU",
                        gpuBackend.name,
                    )
                    MetricRow(
                        "Sidecars",
                        listOfNotNull(
                            "RAW".takeIf { rawEnabled },
                            "DEPTH".takeIf { depthEnabled },
                        ).ifEmpty { listOf("MULTI-VIEW") }.joinToString(
                            " • ",
                        ),
                    )
                    MetricRow(
                        "Dense mesh",
                        denseVertices.toString() +
                            " V / " +
                            denseTriangles.toString() +
                            " T",
                    )
                    MetricRow(
                        "Dense confidence",
                        (denseConfidence * 100.0).toInt().toString() + "%",
                    )
                }
            }
        }

        item {
            GlassCard(
                Modifier.fillMaxWidth(),
            ) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(9.dp),
                ) {
                    SectionTitle(
                        "Rig readiness",
                        "Geometry is accepted only when the evidence supports it.",
                    )
                    MetricRow(
                        "Nodes",
                        devices.count {
                            it.connected
                        }.toString(),
                    )
                    MetricRow(
                        "Pose confidence",
                        (poseConfidence * 100).toInt().toString() + "%",
                    )
                    MetricRow(
                        "Clock skew",
                        (skewNanos / 1_000_000).toString() + " ms",
                    )
                    MetricRow(
                        "3D quality gate",
                        ((reconstructionQualityScore * 100.0)
                            .toInt()
                            .coerceIn(0, 100))
                            .toString() + "% / " +
                            if (reconstructionGateReasons.isEmpty()) {
                                "ACCEPT"
                            } else {
                                "REJECT"
                            },
                    )
                    MetricRow(
                        "3D reprojection",
                        if (reconstructionError.isFinite()) {
                            "%.2f px".format(reconstructionError)
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Face 3D points",
                        face3DPoints.toString(),
                    )
                    MetricRow(
                        "Face reprojection",
                        if (face3DError.isFinite()) {
                            "%.2f px".format(face3DError)
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Eye 3D confidence",
                        (eye3DConfidence * 100.0).toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "View coverage",
                        (rigCoverageScore * 100.0).toInt().toString() + "%",
                    )
                    if (rigRecommendations.isNotEmpty()) {
                        Text(
                            "Coverage guidance",
                            style = MaterialTheme.typography.titleSmall,
                        )
                        rigRecommendations.take(3).forEach { recommendation ->
                            Text(
                                recommendation,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    if (rigReasons.isNotEmpty()) {
                        Spacer(Modifier.height(2.dp))
                        rigReasons.take(3).forEach { reason ->
                            Text(
                                "• " + reason,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }
            }
        }

        item {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(
                        rememberScrollState(),
                    ),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                GlassAction(
                    text = "Discover",
                    enabled = state != "RECORDING",
                ) {
                    viewModel.discover()
                }

                GlassAction(
                    text = "Calibrate",
                    enabled = state != "RECORDING",
                ) {
                    viewModel.calibrate()
                }

                GlassAction(
                    text = "Arm",
                    enabled = state != "RECORDING",
                ) {
                    viewModel.arm()
                }
            }
        }

        item {
            GlassAction(
                text = if (state == "RECORDING") {
                    "STOP SYNCHRONIZED TAKE"
                } else {
                    "START SYNCHRONIZED TAKE"
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = state == "ARMED" ||
                    state == "RECORDING",
                primary = true,
                onClick = viewModel::toggleCapture,
            )
        }

        item {
            Text(
                "MASTER • FRONT • REAR • LEFT • RIGHT • TOP • BOTTOM • FACE • HAND • AUX",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        error?.let { message ->
            item {
                GlassCard(
                    Modifier.fillMaxWidth(),
                ) {
                    Text(
                        message,
                        Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }
    }
}

@Composable
private fun SessionsScreen(
    modifier: Modifier,
    sessions: List<com.nivukx.nox.core.capture.SessionRepository.SessionSummary>,
) {
    LazyColumn(
        modifier
            .fillMaxSize()
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            SectionTitle(
                "Capture Vault",
                "Synchronized takes with integrity validation",
            )
        }

        items(sessions) { session ->
            GlassCard(
                Modifier.fillMaxWidth(),
            ) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Column(
                            Modifier.weight(1f),
                        ) {
                            Text(
                                session.id,
                                style = MaterialTheme.typography.titleMedium,
                            )
                            Text(
                                session.bytes.toString() +
                                    " bytes",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }

                        StatusPill(
                            if (session.complete) {
                                "COMPLETE"
                            } else {
                                "INCOMPLETE"
                            },
                            session.complete,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun DiagnosticsScreen(
    modifier: Modifier,
    devices: List<com.nivukx.nox.core.model.DeviceDescriptor>,
    state: String,
    frames: Long,
    poseCount: Int,
    confidence: Float,
    handCount: Int,
    handConfidence: Double,
    hand3DConfidence: Double,
    focusMode: String,
    focusConfidence: Double,
    inferenceFps: Int,
    inferenceReason: String,
    motionScore: Double,
    temporalStability: Double,
    reconstructionJoints: Int,
    reconstructionError: Double,
    reconstructionQualityScore: Double,
    reconstructionGateReasons: List<String>,
    face3DPoints: Int,
    face3DError: Double,
    eye3DConfidence: Double,
    rigScore: Double,
    identityEvidenceFusionScore: Double = 0.0,
    identitySelectedModalityCount: Int = 0,
    identityEvidenceWarnings: List<String> = emptyList(),
    faceEyeFusionConfidence: Double = 0.0,
    faceEyeSaccadeVelocityDegreesPerSecond: Double = 0.0,
    faceEyeFusionAccepted: Boolean = false,
    faceEyeFusionMode: String = "NONE",
    faceEyeIndependentEvidenceSupport: Double = 0.0,
    faceEyeReliabilityTrend: Double = 0.0,
    faceEyeFusionWarnings: List<String> = emptyList(),
    faceEyeUncertaintyCalibrationConfidence: Double = 0.0,
    faceEyeUncertaintyCalibrationReady: Boolean = false,
    faceUncertaintyCalibrationMultiplier: Double = 1.0,
    eyeUncertaintyCalibrationMultiplier: Double = 1.0,
    faceUncertaintyCoverage: Double = 0.0,
    eyeUncertaintyCoverage: Double = 0.0,
    faceEyeFailureAttribution: String = "NONE",
    faceEyeGeometricIntegrity: Double = 0.0,
    faceEyeFaceFreshnessScore: Double = 0.0,
    faceEyeGazeAgreementScore: Double = 0.0,
    faceEyeRayGeometryCondition: Double? = null,
    faceEyeIpdConsistencyScore: Double = 0.0,
    faceEyeOutOfDistributionRisk: Double = 0.0,
    faceEyeShapeConfidence: Double = 0.0,
    faceEyeEyeDynamicConsistency: Double = 0.0,
    faceEyeHardFailureReasons: List<String> = emptyList(),
    faceEyeCalibrationDriftScore: Double = 0.0,
    faceEyeRecalibrationRecommended: Boolean = false,
    faceEyeCalibrationDriftWarnings: List<String> = emptyList(),
    adaptiveViewEvidenceGlobalRigTrust: Double = 0.0,
    rigQuarantinedCameraCount: Int = 0,
    cameraContributionInformationGainScore: Double = 0.0,
    rigReasons: List<String>,
    rigCoverageScore: Double,
    rigRecommendations: List<String>,
    skewNanos: Long,
    calibrationReady: Boolean,
    lightingLux: Double,
    colorTemperature: Double,
    gpuBackend: GpuBackend,
    rawEnabled: Boolean,
    depthEnabled: Boolean,
    denseVertices: Int,
    denseTriangles: Int,
    denseConfidence: Double,
    denseTsdfDetailLevel: Double,
    denseTsdfComputePressure: Double,
    denseTsdfMeanUncertainty: Double,
    denseTsdfHighUncertaintyRatio: Double,
    denseTsdfConfidenceFactor: Double,
    denseTsdfDecisionReason: String,
    denseMeshExtractionDetail: Double,
    denseMeshExtractionPressure: Double,
    denseMeshExtractionReason: String,
    denseMeshTopologyIntegrity: Double,
    denseMeshBoundaryEdges: Int,
    denseMeshNonManifoldEdges: Int,
    denseMeshDuplicateTrianglesRemoved: Int,
    identityContinuityScore: Double,
    identitySwapRisk: Double,
    identityAction: String,
    identityReason: String,
    trackLifecycleState: String,
    trackLifecycleScore: Double,
    trackLifecyclePredictedCount: Int,
    trackLifecycleRecoveringCount: Int,
    trackLifecycleWarnings: List<String>,
    rigHealthState: String,
    rigHealthScore: Double,
    rigHealthWarnings: List<String>,
    rigCameraHealth:
        List<com.nivukx.nox.core.reconstruction.MultiCameraRigHealthEngine.CameraHealth>,
    cameraContributionSelectedCount: Int,
    cameraContributionExcludedCount: Int,
    cameraContributionDiversitySatisfied: Boolean,
    cameraContributionReason: String,
    jointViewSelectionScore: Double,
    jointViewDiversitySatisfiedRatio: Double,
    jointSelectedObservationCount: Int,
    jointExcludedObservationCount: Int,
    jointViewSelectionReasons: List<String>,
    error: String?,
) {
    LazyColumn(
        modifier
            .fillMaxSize()
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            SectionTitle(
                "Diagnostics",
                "Live pipeline evidence from capture to reconstruction.",
            )
        }

        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                GlassMetric(
                    "State",
                    state,
                    modifier = Modifier.weight(1f),
                )
                GlassMetric(
                    "Stability",
                    (temporalStability * 100).toInt()
                        .toString() + "%",
                    "motion tracker",
                    Modifier.weight(1f),
                )
            }
        }

        item {
            GlassCard(
                Modifier.fillMaxWidth(),
            ) {
                Column(
                    Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    MetricRow(
                        "Frames received",
                        frames.toString(),
                    )
                    MetricRow(
                        "Pose landmarks",
                        poseCount.toString(),
                    )
                    MetricRow(
                        "Hands",
                        handCount.toString(),
                    )
                    MetricRow(
                        "Hand confidence",
                        (handConfidence * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "Hand 3D confidence",
                        (hand3DConfidence * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "Pose confidence",
                        (confidence * 100).toInt().toString() + "%",
                    )
                    MetricRow(
                        "Motion score",
                        (motionScore * 100).toInt().toString() + "%",
                    )
                    MetricRow(
                        "3D joints",
                        reconstructionJoints.toString(),
                    )
                    MetricRow(
                        "3D reprojection",
                        if (reconstructionError.isFinite()) {
                            "%.2f px".format(
                                reconstructionError,
                            )
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Clock skew",
                        skewNanos.toString() + " ns",
                    )
                    MetricRow(
                        "Calibration",
                        if (calibrationReady) {
                            "LOCKED"
                        } else {
                            "OPEN"
                        },
                    )
                    MetricRow(
                        "Face 3D points",
                        face3DPoints.toString(),
                    )
                    MetricRow(
                        "Face reprojection",
                        if (face3DError.isFinite()) {
                            "%.2f px".format(
                                face3DError,
                            )
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Eye 3D confidence",
                        (eye3DConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "Hand 3D confidence",
                        (hand3DConfidence * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "AI focus",
                        focusMode +
                            " / " +
                            (focusConfidence * 100.0)
                                .toInt()
                                .toString() + "%",
                    )
                    MetricRow(
                        "Inference budget",
                        inferenceFps.toString() +
                            " FPS / " +
                            inferenceReason,
                    )
                    MetricRow(
                        "Lighting",
                        if (lightingLux > 0.0) {
                            "%.0f lux".format(lightingLux)
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "Color temperature",
                        if (colorTemperature > 0.0) {
                            "%.0f K".format(colorTemperature)
                        } else {
                            "—"
                        },
                    )
                    MetricRow(
                        "GPU",
                        gpuBackend.name,
                    )
                    MetricRow(
                        "RAW sidecar",
                        if (rawEnabled) "ENABLED" else "OFF",
                    )
                    MetricRow(
                        "Depth sidecar",
                        if (depthEnabled) "ENABLED" else "FALLBACK MULTI-VIEW",
                    )
                    MetricRow(
                        "Dense mesh",
                        denseVertices.toString() +
                            " V / " +
                            denseTriangles.toString() +
                            " T",
                    )
                    MetricRow(
                        "Dense confidence",
                        (denseConfidence * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "TSDF detail",
                        (denseTsdfDetailLevel * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "TSDF compute pressure",
                        (denseTsdfComputePressure * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "TSDF mean SDF uncertainty",
                        "%.3f".format(denseTsdfMeanUncertainty),
                    )
                    MetricRow(
                        "TSDF high-uncertainty voxels",
                        (denseTsdfHighUncertaintyRatio * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "TSDF confidence factor",
                        (denseTsdfConfidenceFactor * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "TSDF policy",
                        denseTsdfDecisionReason.uppercase(),
                    )
                    MetricRow(
                        "Mesh extraction detail",
                        (denseMeshExtractionDetail * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "Mesh extraction pressure",
                        (denseMeshExtractionPressure * 100.0).toInt().toString() + "%",
                    )
                    MetricRow(
                        "Mesh scheduler",
                        denseMeshExtractionReason.uppercase(),
                    )
                    MetricRow(
                        "Mesh topology integrity",
                        (denseMeshTopologyIntegrity * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "Mesh boundary edges",
                        denseMeshBoundaryEdges.toString(),
                    )
                    MetricRow(
                        "Mesh non-manifold edges",
                        denseMeshNonManifoldEdges.toString(),
                    )
                    MetricRow(
                        "Mesh duplicate faces removed",
                        denseMeshDuplicateTrianglesRemoved.toString(),
                    )
                    MetricRow(
                        "Identity continuity",
                        (identityContinuityScore * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "Identity swap risk",
                        (identitySwapRisk * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "Identity policy",
                        identityAction + " / " + identityReason,
                    )
                    MetricRow(
                        "Track lifecycle",
                        trackLifecycleState + " / " +
                            (trackLifecycleScore * 100.0)
                                .toInt()
                                .toString() + "%",
                    )
                    MetricRow(
                        "Track prediction",
                        trackLifecyclePredictedCount.toString(),
                    )
                    MetricRow(
                        "Track recovery",
                        trackLifecycleRecoveringCount.toString(),
                    )
                    MetricRow(
                        "Track warnings",
                        trackLifecycleWarnings.size.toString(),
                    )
                    MetricRow(
                        "Rig health",
                        rigHealthState + " / " +
                            (rigHealthScore * 100.0)
                                .toInt()
                                .toString() + "%",
                    )
                    MetricRow(
                        "Rig warnings",
                        rigHealthWarnings.size.toString(),
                    )
                    MetricRow(
                        "Reconstruction views",
                        cameraContributionSelectedCount.toString() +
                            " selected / " +
                            cameraContributionExcludedCount.toString() +
                            " excluded",
                    )
                    MetricRow(
                        "View diversity",
                        if (cameraContributionDiversitySatisfied) {
                            "SATISFIED"
                        } else {
                            "LIMITED"
                        },
                    )
                    MetricRow(
                        "View planner",
                        cameraContributionReason.uppercase(),
                    )
                    MetricRow(
                        "Joint view score",
                        (jointViewSelectionScore * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "Joint diversity coverage",
                        (jointViewDiversitySatisfiedRatio * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                    MetricRow(
                        "Joint observations",
                        jointSelectedObservationCount.toString() +
                            " selected / " +
                            jointExcludedObservationCount.toString() +
                            " excluded",
                    )
                    if (jointViewSelectionReasons.isNotEmpty()) {
                        MetricRow(
                            "Joint evidence policy",
                            jointViewSelectionReasons.joinToString(" | "),
                        )
                    }
                    MetricRow(
                        "Camera health",
                        rigCameraHealth.count {
                            it.state ==
                                com.nivukx.nox.core.reconstruction.MultiCameraRigHealthEngine.CameraState.HEALTHY
                        }.toString() +
                            "/" +
                            rigCameraHealth.count {
                                it.state !=
                                    com.nivukx.nox.core.reconstruction.MultiCameraRigHealthEngine.CameraState.OFFLINE
                            }.toString(),
                    )
                    MetricRow(
                        "Rig confidence",
                        (rigScore * 100.0)
                            .toInt()
                            .toString() + "%",
                    )
                }
            }
        }

        items(devices) { device ->
            val health =
                rigCameraHealth.firstOrNull {
                    it.deviceId == device.id
                }
            GlassCard(
                Modifier.fillMaxWidth(),
            ) {
                Column(
                    Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(5.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Column {
                            Text(
                                device.displayName,
                                style = MaterialTheme.typography.titleMedium,
                            )
                            Text(
                                device.role.name,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }

                        StatusPill(
                            when {
                                !device.connected -> "OFFLINE"
                                health == null -> "UNKNOWN"
                                else -> health.state.name
                            },
                            device.connected &&
                                health?.state ==
                                    com.nivukx.nox.core.reconstruction.MultiCameraRigHealthEngine.CameraState.HEALTHY,
                        )
                    }

                    Text(
                        "offset " +
                            device.clockOffsetNanos +
                            " ns • RTT " +
                            device.roundTripNanos +
                            " ns",
                    )
                    health?.let {
                        MetricRow(
                            "Rig health",
                            "%.0f%%".format(it.score * 100.0) +
                                " / " +
                                it.state.name,
                        )
                    }
                }
            }
        }

        error?.let {
            item {
                Text(
                    it,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

@Composable
private fun StartupRecoveryScreen(
    error: Throwable?,
) {
    Box(
        Modifier
            .fillMaxSize()
            .background(NoXDarkColors.background),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp),
        ) {
            Image(
                painter = painterResource(
                    com.nivukx.nox.R.drawable.nox_logo,
                ),
                contentDescription = "NivukxNoX",
                modifier = Modifier
                    .width(96.dp)
                    .height(96.dp)
                    .clip(RoundedCornerShape(22.dp)),
                contentScale = ContentScale.Crop,
            )
            Spacer(Modifier.height(20.dp))
            Text(
                "NivukxNoX could not initialize the capture runtime.",
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                error?.message
                    ?.takeIf { it.isNotBlank() }
                    ?: "Initialization failed before the capture surface was ready.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun CameraPreview(
    viewModel: NoXViewModel,
    modifier: Modifier,
    liveMotionPreviewEnabled: Boolean,
    liveMotionPreviewFrame: MotionFieldPreviewFrame?,
) {
    var rendererFailed by remember(liveMotionPreviewEnabled) {
        mutableStateOf(false)
    }

    if (liveMotionPreviewEnabled && !rendererFailed) {
        AndroidView(
            modifier = modifier,
            factory = { context ->
                MotionFieldPreviewSurface(context).apply {
                    setCameraSurfaceListener { surface ->
                        viewModel.setPreviewSurface(surface)
                    }
                    setRendererFailureListener {
                        rendererFailed = true
                        viewModel.setPreviewSurface(null)
                    }
                }
            },
            update = { preview ->
                liveMotionPreviewFrame?.let(preview::submitFrame)
            },
        )
    } else {
        AndroidView(
            modifier = modifier,
            factory = { context ->
                SurfaceView(context).apply {
                    holder.addCallback(
                        object : SurfaceHolder.Callback {
                            override fun surfaceCreated(
                                holder: SurfaceHolder,
                            ) {
                                viewModel.setPreviewSurface(
                                    holder.surface,
                                )
                            }

                            override fun surfaceChanged(
                                holder: SurfaceHolder,
                                format: Int,
                                width: Int,
                                height: Int,
                            ) {
                                viewModel.setPreviewSurface(
                                    holder.surface,
                                )
                            }

                            override fun surfaceDestroyed(
                                holder: SurfaceHolder,
                            ) {
                                viewModel.setPreviewSurface(null)
                            }
                        },
                    )
                }
            },
        )
    }
}

@Composable
private fun MetricRow(
    label: String,
    value: String,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
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
