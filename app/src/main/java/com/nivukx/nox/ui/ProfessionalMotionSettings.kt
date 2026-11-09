package com.nivukx.nox.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import com.nivukx.nox.core.model.MotionFieldGpuMode
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.SessionPresentationSettings

/**
 * Professional motion-capture settings surface.
 *
 * The viewfinder owns the live action. This surface owns configuration. Related controls are
 * deliberately grouped so a production operator does not hunt through a kilometre of sliders.
 */
@Composable
fun ProfessionalMotionSettings(
    modifier: Modifier = Modifier,
    motionCustomization: MotionCaptureCustomization,
    sessionPresentationSettings: SessionPresentationSettings,
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
    adaptiveRegimeStrength: Double,
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
    showEvidenceMatrix: Boolean,
    showDiagnostics: Boolean,
    showViewportGrid: Boolean,
    showPerformanceTelemetry: Boolean,
    onShowEvidenceMatrix: (Boolean) -> Unit,
    onShowDiagnostics: (Boolean) -> Unit,
    onShowViewportGrid: (Boolean) -> Unit,
    onShowPerformanceTelemetry: (Boolean) -> Unit,
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
    onAdaptiveRegimeStrength: (Double) -> Unit,
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
    onClose: () -> Unit,
) {
    var category by remember { mutableStateOf(SettingsCategory.CAPTURE) }

    GlassCard(
        modifier = modifier,
        intensity = 0.98f,
        selected = true,
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    Text(
                        "NOX / SETTINGS",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Production motion-capture control plane",
                        style = MaterialTheme.typography.headlineSmall,
                    )
                    Text(
                        "Every high-impact control is grouped by the subsystem it actually changes.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                GlassAction(
                    text = "DONE",
                    primary = true,
                    modifier = Modifier.width(96.dp),
                    onClick = onClose,
                )
            }

            SettingsCategoryRail(
                selected = category,
                onSelected = { category = it },
            )

            when (category) {
                SettingsCategory.CAMERA -> CameraSettings(
                    motionCustomization = motionCustomization,
                    onAdvancedMotionCustomization =
                        onAdvancedMotionCustomization,
                )

                SettingsCategory.CAPTURE -> CaptureSettings(
                    motionProfile = motionProfile,
                    motionCustomization = motionCustomization,
                    motionResponsiveness = motionResponsiveness,
                    motionSensitivity = motionSensitivity,
                    predictionHorizon = predictionHorizon,
                    occlusionPersistence = occlusionPersistence,
                    jitterSuppression = jitterSuppression,
                    onMotionPreset = onCustomizationPreset,
                    onMotionResponsiveness = onMotionResponsiveness,
                    onMotionSensitivity = onMotionSensitivity,
                    onPredictionHorizon = onPredictionHorizon,
                    onOcclusionPersistence = onOcclusionPersistence,
                    onJitterSuppression = onJitterSuppression,
                    onCustomizationBodyDetail = onCustomizationBodyDetail,
                    onCustomizationDepthDetail = onCustomizationDepthDetail,
                    onCustomizationMaxSubjects = onCustomizationMaxSubjects,
                    onCustomizationOcclusionPredictionFrames =
                        onCustomizationOcclusionPredictionFrames,
                    onCustomizationTargetFps = onCustomizationTargetFps,
                    onCustomizationInferenceFps = onCustomizationInferenceFps,
                    onCustomizationMicroMotionPreservation =
                        onCustomizationMicroMotionPreservation,
                    onCustomizationCameraMotionSuppression =
                        onCustomizationCameraMotionSuppression,
                    onCustomizationOutputIntegrityStrictness =
                        onCustomizationOutputIntegrityStrictness,
                    onAdvancedMotionCustomization =
                        onAdvancedMotionCustomization,
                )

                SettingsCategory.TRACKING -> TrackingSettings(
                    motionCustomization = motionCustomization,
                    reacquisitionRadius = reacquisitionRadius,
                    rejectAggressiveness = rejectAggressiveness,
                    velocityResponsiveness = velocityResponsiveness,
                    accelerationResponsiveness = accelerationResponsiveness,
                    innovationGateElasticity = innovationGateElasticity,
                    motionThresholdScale = motionThresholdScale,
                    onCustomizationBodyDetail = onCustomizationBodyDetail,
                    onCustomizationDepthDetail = onCustomizationDepthDetail,
                    onReacquisitionRadius = onReacquisitionRadius,
                    onRejectAggressiveness = onRejectAggressiveness,
                    onVelocityResponsiveness = onVelocityResponsiveness,
                    onAccelerationResponsiveness = onAccelerationResponsiveness,
                    onInnovationGateElasticity = onInnovationGateElasticity,
                    onMotionThresholdScale = onMotionThresholdScale,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )

                SettingsCategory.MULTI_VIEW -> MultiViewSettings(
                    motionCustomization = motionCustomization,
                    onCustomizationWorldConsistencyDetail =
                        onCustomizationWorldConsistencyDetail,
                    onCustomizationTopologyConstraintStrength =
                        onCustomizationTopologyConstraintStrength,
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
                    onCustomizationPhysicsConstraintStrictness =
                        onCustomizationPhysicsConstraintStrictness,
                    onCustomizationGlobal4DConsistencyStrictness =
                        onCustomizationGlobal4DConsistencyStrictness,
                    onCustomizationSurfaceEvidenceStrictness =
                        onCustomizationSurfaceEvidenceStrictness,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )


                SettingsCategory.FRONTIER -> FrontierCaptureSettings(
                    motionCustomization = motionCustomization,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )

                SettingsCategory.RECONSTRUCTION -> ReconstructionSettings(
                    motionCustomization = motionCustomization,
                    reconstructionProfile = reconstructionProfile,
                    reconstructionSoftAccept = reconstructionSoftAccept,
                    reconstructionHoldFrames = reconstructionHoldFrames,
                    reconstructionHoldAgeMillis = reconstructionHoldAgeMillis,
                    reconstructionStabilitySensitivity =
                        reconstructionStabilitySensitivity,
                    onReconstructionPreset = onReconstructionPreset,
                    onReconstructionSoftAccept = onReconstructionSoftAccept,
                    onReconstructionHoldFrames = onReconstructionHoldFrames,
                    onReconstructionHoldAgeMillis = onReconstructionHoldAgeMillis,
                    onReconstructionStabilitySensitivity =
                        onReconstructionStabilitySensitivity,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )

                SettingsCategory.TEMPORAL -> TemporalSettings(
                    temporalPredictionHorizon = temporalPredictionHorizon,
                    temporalInnovationGateScale = temporalInnovationGateScale,
                    temporalUncertaintyGateScale = temporalUncertaintyGateScale,
                    temporalReprojectionGateScale = temporalReprojectionGateScale,
                    temporalSmoothingResponsiveness = temporalSmoothingResponsiveness,
                    temporalConfidenceDecayScale = temporalConfidenceDecayScale,
                    temporalOutlierRetention = temporalOutlierRetention,
                    temporalTopologyToleranceScale = temporalTopologyToleranceScale,
                    onTemporalPredictionHorizon = onTemporalPredictionHorizon,
                    onTemporalInnovationGateScale = onTemporalInnovationGateScale,
                    onTemporalUncertaintyGateScale = onTemporalUncertaintyGateScale,
                    onTemporalReprojectionGateScale = onTemporalReprojectionGateScale,
                    onTemporalSmoothingResponsiveness =
                        onTemporalSmoothingResponsiveness,
                    onTemporalConfidenceDecayScale =
                        onTemporalConfidenceDecayScale,
                    onTemporalOutlierRetention = onTemporalOutlierRetention,
                    onTemporalTopologyToleranceScale =
                        onTemporalTopologyToleranceScale,
                )

                SettingsCategory.KINEMATICS -> KinematicSettings(
                    motionCustomization = motionCustomization,
                    motionResponsiveness = motionResponsiveness,
                    motionSensitivity = motionSensitivity,
                    adaptiveRegimeStrength = adaptiveRegimeStrength,
                    velocityResponsiveness = velocityResponsiveness,
                    accelerationResponsiveness = accelerationResponsiveness,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                    onMotionResponsiveness = onMotionResponsiveness,
                    onMotionSensitivity = onMotionSensitivity,
                    onAdaptiveRegimeStrength = onAdaptiveRegimeStrength,
                    onVelocityResponsiveness = onVelocityResponsiveness,
                    onAccelerationResponsiveness = onAccelerationResponsiveness,
                    onCustomizationMicroMotionPreservation =
                        onCustomizationMicroMotionPreservation,
                )

                SettingsCategory.FACE_HANDS -> FaceHandSettings(
                    motionCustomization = motionCustomization,
                    onCustomizationFaceDetail = onCustomizationFaceDetail,
                    onCustomizationEyeDetail = onCustomizationEyeDetail,
                    onCustomizationEyeTemporalStrictness =
                        onCustomizationEyeTemporalStrictness,
                    onCustomizationHandDetail = onCustomizationHandDetail,
                    onCustomizationHandTemporalStrictness =
                        onCustomizationHandTemporalStrictness,
                    onCustomizationHandGeometryStrictness =
                        onCustomizationHandGeometryStrictness,
                    onCustomizationFaceEyeFusionWeight =
                        onCustomizationFaceEyeFusionWeight,
                    onCustomizationFaceEyeFusionStrictness =
                        onCustomizationFaceEyeFusionStrictness,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )

                SettingsCategory.SURFACE -> SurfaceSettings(
                    motionCustomization = motionCustomization,
                    onCustomizationDenseMeshDetail = onCustomizationDenseMeshDetail,
                    onCustomizationContactDetail = onCustomizationContactDetail,
                    onCustomizationTopologyConstraintStrength =
                        onCustomizationTopologyConstraintStrength,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )

                SettingsCategory.SESSION -> SessionSettings(
                    sessionPresentationSettings = sessionPresentationSettings,
                    onSessionClipName = onSessionClipName,
                    onSessionCountdownSeconds = onSessionCountdownSeconds,
                    onRequireCalibrationBeforeRecord =
                        onRequireCalibrationBeforeRecord,
                )

                SettingsCategory.SYSTEM -> SystemSettings(
                    motionCustomization = motionCustomization,
                    showEvidenceMatrix = showEvidenceMatrix,
                    showDiagnostics = showDiagnostics,
                    showViewportGrid = showViewportGrid,
                    showPerformanceTelemetry = showPerformanceTelemetry,
                    onShowEvidenceMatrix = onShowEvidenceMatrix,
                    onShowDiagnostics = onShowDiagnostics,
                    onShowViewportGrid = onShowViewportGrid,
                    onShowPerformanceTelemetry = onShowPerformanceTelemetry,
                    onCustomizationPreset = onCustomizationPreset,
                    onAdvancedMotionCustomization = onAdvancedMotionCustomization,
                )
            }

            Spacer(Modifier.height(8.dp))
            Text(
                "NoX remains motion-capture-only. Settings below control perception, reconstruction, synchronization, diagnostics and presentation, not general video recording.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private enum class SettingsCategory(
    val title: String,
    val subtitle: String,
) {
    CAMERA("Camera", "Sensor acquisition"),
    CAPTURE("Capture", "Primary acquisition"),
    TRACKING("Tracking", "Subject lock + detectors"),
    MULTI_VIEW("Multi-view", "Calibration + evidence"),
    RECONSTRUCTION("Reconstruction", "3D solver"),
    FRONTIER("Frontier", "A+B+C control plane"),
    TEMPORAL("Temporal", "Prediction + continuity"),
    KINEMATICS("Kinematics", "Constraints + dynamics"),
    FACE_HANDS("Face + hands", "Fine geometry"),
    SURFACE("Surface", "Dense mesh"),
    SESSION("Session", "Take metadata"),
    SYSTEM("System", "Interface + budgets"),
}

@Composable
private fun SettingsCategoryRail(
    selected: SettingsCategory,
    onSelected: (SettingsCategory) -> Unit,
) {
    GlassCard(intensity = 0.60f) {
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            SettingsCategory.values().forEach { item ->
                Surface(
                    onClick = { onSelected(item) },
                    shape = androidx.compose.foundation.shape.RoundedCornerShape(15.dp),
                    color = if (item == selected) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceContainerHigh
                    },
                    border = androidx.compose.foundation.BorderStroke(
                        1.dp,
                        if (item == selected) {
                            MaterialTheme.colorScheme.primary.copy(alpha = 0.60f)
                        } else {
                            MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.34f)
                        },
                    ),
                ) {
                    Column(
                        Modifier.padding(horizontal = 13.dp, vertical = 9.dp),
                        verticalArrangement = Arrangement.spacedBy(1.dp),
                    ) {
                        Text(
                            item.title.uppercase(),
                            style = MaterialTheme.typography.labelMedium,
                        )
                        Text(
                            item.subtitle,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun CameraSettings(
    motionCustomization: MotionCaptureCustomization,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader(
        "Camera acquisition",
        "Capability-aware sensor controls. The engine clamps requests to real Camera2 limits.",
    )

    SettingsOptionRow(
        title = "Analysis resolution",
        selected = motionCustomization.cameraResolutionPreset.name
            .replace('_', ' '),
        options = listOf(
            "AUTO",
            "HD_720P",
            "FULL_HD_1080P",
            "UHD_4K",
            "MAX_SUPPORTED",
        ),
        onSelected = { value ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraResolutionPreset =
                        runCatching {
                            com.nivukx.nox.core.model.CameraResolutionPreset
                                .valueOf(value)
                        }.getOrDefault(
                            com.nivukx.nox.core.model.CameraResolutionPreset.AUTO,
                        )
                }
            }
        },
    )

    SettingsOptionRow(
        title = "Focus mode",
        selected = motionCustomization.cameraFocusMode.name
            .replace('_', ' '),
        options = listOf(
            "TRACKED_SUBJECT",
            "CONTINUOUS",
            "MANUAL",
        ),
        onSelected = { value ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraFocusMode =
                        runCatching {
                            com.nivukx.nox.core.model.CameraFocusMode
                                .valueOf(value)
                        }.getOrDefault(
                            com.nivukx.nox.core.model.CameraFocusMode.TRACKED_SUBJECT,
                        )
                }
            }
        },
    )

    SettingsOptionRow(
        title = "Exposure mode",
        selected = motionCustomization.cameraExposureMode.name,
        options = listOf("AUTO", "MANUAL"),
        onSelected = { value ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraExposureMode =
                        runCatching {
                            com.nivukx.nox.core.model.CameraExposureMode
                                .valueOf(value)
                        }.getOrDefault(
                            com.nivukx.nox.core.model.CameraExposureMode.AUTO,
                        )
                }
            }
        },
    )

    SettingsOptionRow(
        title = "Anti-banding",
        selected =
            motionCustomization.cameraAntibandingMode.name
                .replace('_', ' '),
        options = listOf(
            "AUTO",
            "HZ_50",
            "HZ_60",
            "OFF",
        ),
        onSelected = { value ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraAntibandingMode =
                        runCatching {
                            com.nivukx.nox.core.model
                                .CameraAntibandingMode
                                .valueOf(value)
                        }.getOrDefault(
                            com.nivukx.nox.core.model
                                .CameraAntibandingMode.AUTO,
                        )
                }
            }
        },
    )

    SettingSlider(
        "Sensor frame rate",
        motionCustomization.targetFps.toDouble(),
        15.0..240.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    targetFps = next.toInt().coerceIn(15, 240)
                    inferenceFps =
                        it.inferenceFps.coerceAtMost(
                            next.toInt().coerceIn(15, 240),
                        )
                }
            }
        },
        numberFormat = "%.0f",
        suffix = " fps",
    )

    SettingSlider(
        "Manual exposure",
        motionCustomization.cameraManualExposureMicros.toDouble(),
        100.0..500_000.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraManualExposureMicros =
                        next.toLong().coerceIn(100L, 500_000L)
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.0f",
        suffix = " µs",
    )

    SettingSlider(
        "Manual ISO",
        motionCustomization.cameraManualIso.toDouble(),
        25.0..12_800.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraManualIso =
                        next.toInt().coerceIn(25, 12_800)
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.0f",
        suffix = "",
    )

    SettingSlider(
        "Exposure compensation",
        motionCustomization.cameraExposureCompensationEv,
        -8.0..8.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraExposureCompensationEv = next
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.1f",
        suffix = " EV",
    )

    SettingSlider(
        "Manual focus distance",
        motionCustomization.cameraManualFocusDistanceDiopters ?: 0.0,
        0.0..20.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraManualFocusDistanceDiopters =
                        next.coerceIn(0.0, 1000.0)
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.2f",
        suffix = " D",
    )

    SettingsSwitch(
        title = "Optical stabilization",
        description = "Use lens OIS only when the camera advertises it. Motion capture defaults to off to preserve geometry.",
        checked = motionCustomization.cameraOisEnabled,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraOisEnabled = enabled
                }
            }
        },
    )

    SettingsSwitch(
        title = "Electronic stabilization",
        description = "Use Camera2 video stabilization. Leave disabled when spatial fidelity is the priority.",
        checked = motionCustomization.cameraElectronicStabilizationEnabled,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraElectronicStabilizationEnabled = enabled
                }
            }
        },
    )

    SettingsSwitch(
        title = "RAW sensor sidecar",
        description = "Persist RAW_SENSOR frames when the camera exposes a compatible stream.",
        checked = motionCustomization.cameraRawSidecarEnabled,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraRawSidecarEnabled = enabled
                }
            }
        },
    )

    SettingsSwitch(
        title = "Depth sidecar",
        description = "Persist DEPTH16 data when available for dense 3D reconstruction.",
        checked = motionCustomization.cameraDepthSidecarEnabled,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraDepthSidecarEnabled = enabled
                }
            }
        },
    )

    SettingsSwitch(
        title = "Auto-exposure lock",
        description = "Freeze AE after the current exposure is established.",
        checked = motionCustomization.cameraAutoExposureLock,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraAutoExposureLock = enabled
                }
            }
        },
    )

    SettingsSwitch(
        title = "Auto-white-balance lock",
        description = "Prevent per-frame AWB changes from becoming a reconstruction nuisance.",
        checked = motionCustomization.cameraAutoWhiteBalanceLock,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraAutoWhiteBalanceLock = enabled
                }
            }
        },
    )


    SettingsSwitch(
        title = "Precise CCT white balance",
        description =
            "Android 16+ camera support: hold a measured color temperature for cross-camera texture consistency.",
        checked =
            motionCustomization.cameraPreciseColorTemperatureEnabled,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraPreciseColorTemperatureEnabled = enabled
                }
            }
        },
    )

    SettingSlider(
        "Color temperature",
        motionCustomization.cameraColorTemperatureKelvin.toDouble(),
        2000.0..10000.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraColorTemperatureKelvin =
                        next.toInt().coerceIn(2000, 10000)
                }
            }
        },
        numberFormat = "%.0f",
        suffix = " K",
    )

    SettingSlider(
        "Color tint",
        motionCustomization.cameraColorTint.toDouble(),
        -100.0..100.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraColorTint =
                        next.toInt().coerceIn(-100, 100)
                }
            }
        },
        numberFormat = "%.0f",
        suffix = "",
    )

    SettingsSwitch(
        title = "Preserve sensor timestamps",
        description = "Keep hardware sensor timing as the authoritative frame clock for synchronization.",
        checked = motionCustomization.cameraPreserveSensorTimestamp,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    cameraPreserveSensorTimestamp = enabled
                }
            }
        },
    )
}

@Composable
private fun CaptureSettings(
    motionProfile: String,
    motionCustomization: MotionCaptureCustomization,
    motionResponsiveness: Double,
    motionSensitivity: Double,
    predictionHorizon: Double,
    occlusionPersistence: Double,
    jitterSuppression: Double,
    onMotionPreset: (String) -> Unit,
    onMotionResponsiveness: (Double) -> Unit,
    onMotionSensitivity: (Double) -> Unit,
    onPredictionHorizon: (Double) -> Unit,
    onOcclusionPersistence: (Double) -> Unit,
    onJitterSuppression: (Double) -> Unit,
    onCustomizationBodyDetail: (Double) -> Unit,
    onCustomizationDepthDetail: (Double) -> Unit,
    onCustomizationMaxSubjects: (Int) -> Unit,
    onCustomizationOcclusionPredictionFrames: (Int) -> Unit,
    onCustomizationTargetFps: (Int) -> Unit,
    onCustomizationInferenceFps: (Int) -> Unit,
    onCustomizationMicroMotionPreservation: (Double) -> Unit,
    onCustomizationCameraMotionSuppression: (Double) -> Unit,
    onCustomizationOutputIntegrityStrictness: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader(
        "Capture",
        "Operator-facing acquisition behavior and runtime capture budgets.",
    )
    SettingsSwitch(
        title = "Live Motion Preview",
        checked = motionCustomization.liveMotionPreviewEnabled,
        onCheckedChange = { enabled ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    liveMotionPreviewEnabled = enabled
                }
            }
        },
        description =
            "GPU-composited body motion, micro-motion, and live 3D reconstruction overlay.",
    )
    SettingsPresetRow(
        items = listOf(
            "AUTO",
            "MAXIMUM",
            "HIGH",
            "BALANCED",
        ),
        active = motionCustomization.liveMotionPreviewGpuMode.name,
        onSelect = { selected ->
            runCatching {
                MotionFieldGpuMode.valueOf(selected)
            }.getOrNull()?.let { mode ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        liveMotionPreviewGpuMode = mode
                    }
                }
            }
        },
    )
    AdvancedSettingSlider(
        "Live preview detail",
        motionCustomization.liveMotionPreviewDetail,
        0.20..1.0,
        onAdvancedMotionCustomization,
        change = { it },
        scale = 100.0,
        numberFormat = "%.0f",
    )
    SettingSlider(
        "Display prediction horizon",
        motionCustomization.liveMotionPreviewPredictionHorizonMillis,
        0.0..50.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    liveMotionPreviewPredictionHorizonMillis = next
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.1f",
        suffix = " ms",
    )
    SettingSlider(
        "Maximum stale preview",
        motionCustomization.liveMotionPreviewMaxStaleMillis,
        16.0..250.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    liveMotionPreviewMaxStaleMillis = next
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.0f",
        suffix = " ms",
    )

    SettingsPresetRow(
        items = listOf("CINEMATIC", "ACTION", "PRECISION", "CREATURE"),
        active = motionProfile,
        onSelect = onMotionPreset,
    )
    SettingSlider(
        "Responsiveness",
        motionResponsiveness,
        0.20..1.20,
        onMotionResponsiveness,
        numberFormat = "%.2f",
    )
    SettingSlider(
        "Sensitivity",
        motionSensitivity,
        0.50..1.80,
        onMotionSensitivity,
        numberFormat = "%.2f",
    )
    SettingSlider(
        "Prediction horizon",
        predictionHorizon,
        0.02..0.20,
        onPredictionHorizon,
        scale = 1000.0,
        suffix = " ms",
    )
    SettingSlider(
        "Occlusion persistence",
        occlusionPersistence,
        0.0..1.0,
        onOcclusionPersistence,
    )
    SettingSlider(
        "Jitter suppression",
        jitterSuppression,
        0.0..1.0,
        onJitterSuppression,
    )
    SettingSlider(
        "Target capture rate",
        motionCustomization.targetFps.toDouble(),
        15.0..240.0,
        { onCustomizationTargetFps(it.toInt().coerceIn(15, 240)) },
        numberFormat = "%.0f",
        suffix = " fps",
    )
    SettingSlider(
        "Inference rate",
        motionCustomization.inferenceFps.toDouble(),
        5.0..60.0,
        { onCustomizationInferenceFps(it.toInt().coerceIn(5, 60)) },
        numberFormat = "%.0f",
        suffix = " fps",
    )
    SettingSlider(
        "Body detail",
        motionCustomization.bodyDetail,
        0.0..1.0,
        onCustomizationBodyDetail,
    )
    SettingSlider(
        "Depth detail",
        motionCustomization.depthDetail,
        0.0..1.0,
        onCustomizationDepthDetail,
    )
    SettingSlider(
        "Micro-motion preservation",
        motionCustomization.microMotionPreservation,
        0.0..1.0,
        onCustomizationMicroMotionPreservation,
    )
    SettingSlider(
        "Camera-motion suppression",
        motionCustomization.cameraMotionSuppression,
        0.0..1.0,
        onCustomizationCameraMotionSuppression,
    )
    SettingSlider(
        "Max subjects",
        motionCustomization.maxSubjects.toDouble(),
        1.0..64.0,
        { onCustomizationMaxSubjects(it.toInt().coerceIn(1, 64)) },
        numberFormat = "%.0f",
        suffix = " subjects",
    )
    AdvancedSettingSlider(
        "Actor identity strictness",
        motionCustomization.actorIdentityStrictness,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        actorIdentityStrictness = it.actorIdentityStrictness
    } }
    AdvancedSettingSlider(
        "Actor occlusion recovery",
        motionCustomization.actorOcclusionRecovery,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        actorOcclusionRecovery = it.actorOcclusionRecovery
    } }
    AdvancedSettingSlider(
        "Cross-camera actor association",
        motionCustomization.actorCrossCameraAssociation,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        actorCrossCameraAssociation = it.actorCrossCameraAssociation
    } }
    AdvancedSettingSlider(
        "Actor micro-motion sensitivity",
        motionCustomization.actorMicroMotionSensitivity,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        actorMicroMotionSensitivity = it.actorMicroMotionSensitivity
    } }
    AdvancedSettingSlider(
        "Temporal burst sensitivity",
        motionCustomization.motionFrequencyBandSensitivity,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        motionFrequencyBandSensitivity = it.motionFrequencyBandSensitivity
    } }
    AdvancedSettingSlider(
        "Micro-motion retention",
        motionCustomization.actorMicroMotionRetention,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        actorMicroMotionRetention = it.actorMicroMotionRetention
    } }
    AdvancedSettingSlider(
        "Subframe timing strictness",
        motionCustomization.actorSubframeTimingStrictness,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        actorSubframeTimingStrictness = it.actorSubframeTimingStrictness
    } }
    SettingSlider(
        "Occlusion horizon",
        motionCustomization.maximumOcclusionPredictionFrames.toDouble(),
        0.0..30.0,
        { onCustomizationOcclusionPredictionFrames(it.toInt().coerceIn(0, 30)) },
        numberFormat = "%.0f",
        suffix = " frames",
    )
    SettingSlider(
        "Output integrity strictness",
        motionCustomization.outputIntegrityStrictness,
        0.0..1.0,
        onCustomizationOutputIntegrityStrictness,
    )
}

@Composable
private fun TrackingSettings(
    motionCustomization: MotionCaptureCustomization,
    reacquisitionRadius: Double,
    rejectAggressiveness: Double,
    velocityResponsiveness: Double,
    accelerationResponsiveness: Double,
    innovationGateElasticity: Double,
    motionThresholdScale: Double,
    onCustomizationBodyDetail: (Double) -> Unit,
    onCustomizationDepthDetail: (Double) -> Unit,
    onReacquisitionRadius: (Double) -> Unit,
    onRejectAggressiveness: (Double) -> Unit,
    onVelocityResponsiveness: (Double) -> Unit,
    onAccelerationResponsiveness: (Double) -> Unit,
    onInnovationGateElasticity: (Double) -> Unit,
    onMotionThresholdScale: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader("Tracking", "Subject acquisition, detector gates and occlusion recovery.")
    SettingSlider("Body detail", motionCustomization.bodyDetail, 0.0..1.0, onCustomizationBodyDetail)
    SettingSlider("Depth detail", motionCustomization.depthDetail, 0.0..1.0, onCustomizationDepthDetail)
    SettingSlider("Reacquisition radius", reacquisitionRadius, 0.0..1.0, onReacquisitionRadius)
    SettingSlider("Reject aggressiveness", rejectAggressiveness, 0.0..1.0, onRejectAggressiveness)
    SettingSlider("Velocity responsiveness", velocityResponsiveness, 0.0..1.0, onVelocityResponsiveness)
    SettingSlider("Acceleration responsiveness", accelerationResponsiveness, 0.0..1.0, onAccelerationResponsiveness)
    SettingSlider("Innovation gate elasticity", innovationGateElasticity, 0.0..2.0, onInnovationGateElasticity, scale = 1.0, numberFormat = "%.2f")
    SettingSlider("Motion threshold", motionThresholdScale, 0.25..2.0, onMotionThresholdScale, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    AdvancedSettingSlider("Occlusion recovery confidence", motionCustomization.occlusionRecoveryConfidence, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            occlusionRecoveryConfidence = it.occlusionRecoveryConfidence
        }
    }
}

@Composable
private fun MultiViewSettings(
    motionCustomization: MotionCaptureCustomization,
    onCustomizationWorldConsistencyDetail: (Double) -> Unit,
    onCustomizationTopologyConstraintStrength: (Double) -> Unit,
    onCustomizationMinimumParallaxDegrees: (Double) -> Unit,
    onCustomizationPreferredTemporalSkewMillis: (Double) -> Unit,
    onCustomizationMaximumTemporalSkewMillis: (Double) -> Unit,
    onCustomizationMaximumReprojectionErrorPx: (Double) -> Unit,
    onCustomizationMaximumReconstructionUncertaintyMeters: (Double) -> Unit,
    onCustomizationPhysicsConstraintStrictness: (Double) -> Unit,
    onCustomizationGlobal4DConsistencyStrictness: (Double) -> Unit,
    onCustomizationSurfaceEvidenceStrictness: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader("Multi-view", "Calibrated camera coverage, synchronization and evidence trust.")
    SettingSlider("World coherence", motionCustomization.worldConsistencyDetail, 0.0..1.0, onCustomizationWorldConsistencyDetail)
    SettingSlider("Temporal alignment strictness", motionCustomization.temporalAlignmentStrictness, 0.0..1.0, { next ->
        onAdvancedMotionCustomization {
            it.copyWith {
                temporalAlignmentStrictness = next
            }
        }
    })
    AdvancedSettingSlider(
        "Sub-frame alignment strictness",
        motionCustomization.subframeAlignmentStrictness,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        subframeAlignmentStrictness = it.subframeAlignmentStrictness
    } }
    AdvancedSettingSlider(
        "Rolling-shutter compensation",
        motionCustomization.rollingShutterCompensation,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        rollingShutterCompensation = it.rollingShutterCompensation
    } }
    AdvancedSettingSlider(
        "Photometric exposure soft limit",
        motionCustomization.photometricExposureShiftSoftLimit,
        0.0..0.30,
        onAdvancedMotionCustomization,
        scale = 1.0,
        numberFormat = "%.2f",
    ) { it.copyWith {
        photometricExposureShiftSoftLimit =
            it.photometricExposureShiftSoftLimit
    } }
    AdvancedSettingSlider(
        "Photometric exposure hard limit",
        motionCustomization.photometricExposureShiftHardLimit,
        (motionCustomization.photometricExposureShiftSoftLimit + 0.01)
            .coerceAtMost(0.80)..0.80,
        onAdvancedMotionCustomization,
        scale = 1.0,
        numberFormat = "%.2f",
    ) { it.copyWith {
        photometricExposureShiftHardLimit =
            it.photometricExposureShiftHardLimit
    } }
    AdvancedSettingSlider(
        "Photometric exposure authority floor",
        motionCustomization.photometricExposureReliabilityFloor,
        0.0..1.0,
        onAdvancedMotionCustomization,
        scale = 1.0,
        numberFormat = "%.2f",
    ) { it.copyWith {
        photometricExposureReliabilityFloor =
            it.photometricExposureReliabilityFloor
    } }
    AdvancedSettingSlider("Depth temporal fusion", motionCustomization.depthTemporalFusion, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            depthTemporalFusion = it.depthTemporalFusion
        }
    }
    SettingSlider("Topology constraint", motionCustomization.topologyConstraintStrength, 0.0..1.0, onCustomizationTopologyConstraintStrength)
    SettingSlider("Minimum parallax", motionCustomization.minimumParallaxDegrees, 0.20..2.50, onCustomizationMinimumParallaxDegrees, scale = 1.0, numberFormat = "%.2f", suffix = "°")
    SettingSlider("Preferred temporal skew", motionCustomization.preferredTemporalSkewMillis, 0.50..15.0, onCustomizationPreferredTemporalSkewMillis, scale = 1.0, numberFormat = "%.2f", suffix = " ms")
    SettingSlider("Maximum temporal skew", motionCustomization.maximumTemporalSkewMillis, 5.0..60.0, onCustomizationMaximumTemporalSkewMillis, scale = 1.0, numberFormat = "%.1f", suffix = " ms")
    SettingSlider("Maximum reprojection error", motionCustomization.maximumReprojectionErrorPx, 1.0..20.0, onCustomizationMaximumReprojectionErrorPx, scale = 1.0, numberFormat = "%.1f", suffix = " px")
    SettingSlider("Maximum 3D uncertainty", motionCustomization.maximumReconstructionUncertaintyMeters, 0.05..2.0, onCustomizationMaximumReconstructionUncertaintyMeters, scale = 1.0, numberFormat = "%.2f", suffix = " m")
    SettingSlider(
        "Physics constraint strictness",
        motionCustomization.physicsConstraintStrictness,
        0.0..1.0,
        onCustomizationPhysicsConstraintStrictness,
    )
    SettingSlider(
        "Global 4D consistency strictness",
        motionCustomization.global4DConsistencyStrictness,
        0.0..1.0,
        onCustomizationGlobal4DConsistencyStrictness,
    )
    SettingSlider(
        "Surface evidence strictness",
        motionCustomization.surfaceEvidenceStrictness,
        0.0..1.0,
        onCustomizationSurfaceEvidenceStrictness,
    )
    AdvancedSettingSlider(
        "Evidence adaptivity",
        motionCustomization.multiViewEvidenceAdaptivity,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        multiViewEvidenceAdaptivity = it.multiViewEvidenceAdaptivity
    } }
    AdvancedSettingSlider(
        "Camera drift response",
        motionCustomization.cameraDriftResponse,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        cameraDriftResponse = it.cameraDriftResponse
    } }
    AdvancedSettingSlider(
        "Network jitter tolerance",
        motionCustomization.networkJitterToleranceMillis,
        0.0..250.0,
        onAdvancedMotionCustomization,
        scale = 1.0,
        numberFormat = "%.0f",
        suffix = " ms",
    ) { it.copyWith {
        networkJitterToleranceMillis = it.networkJitterToleranceMillis
    } }
}

@Composable
private fun ReconstructionSettings(
    motionCustomization: MotionCaptureCustomization,
    reconstructionProfile: String,
    reconstructionSoftAccept: Double,
    reconstructionHoldFrames: Int,
    reconstructionHoldAgeMillis: Long,
    reconstructionStabilitySensitivity: Double,
    onReconstructionPreset: (String) -> Unit,
    onReconstructionSoftAccept: (Double) -> Unit,
    onReconstructionHoldFrames: (Int) -> Unit,
    onReconstructionHoldAgeMillis: (Long) -> Unit,
    onReconstructionStabilitySensitivity: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader("Reconstruction", "3D output acceptance, solver profiles and constrained recovery.")
    SettingsPresetRow(
        items = listOf("CINEMATIC", "ACTION", "PRECISION", "CREATURE", "FACIAL", "HAND_DETAIL"),
        active = reconstructionProfile,
        onSelect = onReconstructionPreset,
    )
    SettingSlider("Soft accept threshold", reconstructionSoftAccept, 0.0..1.0, onReconstructionSoftAccept)
    SettingSlider("Hold frames", reconstructionHoldFrames.toDouble(), 1.0..60.0, { onReconstructionHoldFrames(it.toInt().coerceIn(1, 60)) }, numberFormat = "%.0f", suffix = " frames")
    SettingSlider("Hold age", reconstructionHoldAgeMillis.toDouble(), 16.0..1200.0, { onReconstructionHoldAgeMillis(it.toLong().coerceIn(16, 1200)) }, numberFormat = "%.0f", suffix = " ms")
    SettingSlider("Stability sensitivity", reconstructionStabilitySensitivity, 0.0..2.0, onReconstructionStabilitySensitivity, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    SettingsHeader("Evidence fusion", "Robustly arbitrates multi-view, temporal and kinematic 3D hypotheses without silently treating correlated evidence as independent.")
    SettingSlider(
        "Fusion strength",
        motionCustomization.reconstructionEvidenceFusionStrength,
        0.0..1.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    reconstructionEvidenceFusionStrength = next.coerceIn(0.0, 1.0)
                }
            }
        },
        numberFormat = "%.2f",
        suffix = "x",
    )
    SettingSlider(
        "Disagreement tolerance",
        motionCustomization.reconstructionEvidenceDisagreementToleranceMeters,
        0.02..1.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    reconstructionEvidenceDisagreementToleranceMeters =
                        next.coerceIn(0.02, 1.0)
                }
            }
        },
        scale = 1.0,
        numberFormat = "%.2f",
        suffix = " m",
    )
    SettingSlider(
        "Minimum source diversity",
        motionCustomization.reconstructionEvidenceMinimumSourceDiversity,
        0.0..1.0,
        { next ->
            onAdvancedMotionCustomization {
                it.copyWith {
                    reconstructionEvidenceMinimumSourceDiversity =
                        next.coerceIn(0.0, 1.0)
                }
            }
        },
        numberFormat = "%.2f",
        suffix = "x",
    )
    AdvancedSettingSlider("Pose prior strength", motionCustomization.posePriorStrength, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            posePriorStrength = it.posePriorStrength
        }
    }
    AdvancedSettingSlider("Retargeting strictness", motionCustomization.retargetingStrictness, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            retargetingStrictness = it.retargetingStrictness
        }
    }
    AdvancedSettingSlider("Scale preservation", motionCustomization.scalePreservation, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            scalePreservation = it.scalePreservation
        }
    }
}

@Composable
private fun TemporalSettings(
    temporalPredictionHorizon: Double,
    temporalInnovationGateScale: Double,
    temporalUncertaintyGateScale: Double,
    temporalReprojectionGateScale: Double,
    temporalSmoothingResponsiveness: Double,
    temporalConfidenceDecayScale: Double,
    temporalOutlierRetention: Double,
    temporalTopologyToleranceScale: Double,
    onTemporalPredictionHorizon: (Double) -> Unit,
    onTemporalInnovationGateScale: (Double) -> Unit,
    onTemporalUncertaintyGateScale: (Double) -> Unit,
    onTemporalReprojectionGateScale: (Double) -> Unit,
    onTemporalSmoothingResponsiveness: (Double) -> Unit,
    onTemporalConfidenceDecayScale: (Double) -> Unit,
    onTemporalOutlierRetention: (Double) -> Unit,
    onTemporalTopologyToleranceScale: (Double) -> Unit,
) {
    SettingsHeader("Temporal", "Prediction, robust innovation gates and continuity preservation.")
    SettingSlider("Prediction horizon", temporalPredictionHorizon, 0.01..0.30, onTemporalPredictionHorizon, scale = 1000.0, numberFormat = "%.0f", suffix = " ms")
    SettingSlider("Innovation gate", temporalInnovationGateScale, 0.0..2.0, onTemporalInnovationGateScale, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    SettingSlider("Uncertainty gate", temporalUncertaintyGateScale, 0.0..2.0, onTemporalUncertaintyGateScale, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    SettingSlider("Reprojection gate", temporalReprojectionGateScale, 0.0..2.0, onTemporalReprojectionGateScale, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    SettingSlider("Smoothing responsiveness", temporalSmoothingResponsiveness, 0.0..1.0, onTemporalSmoothingResponsiveness)
    SettingSlider("Confidence decay", temporalConfidenceDecayScale, 0.0..2.0, onTemporalConfidenceDecayScale, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    SettingSlider("Outlier retention", temporalOutlierRetention, 0.0..1.0, onTemporalOutlierRetention)
    SettingSlider("Topology tolerance", temporalTopologyToleranceScale, 0.0..2.0, onTemporalTopologyToleranceScale, scale = 1.0, numberFormat = "%.2f", suffix = "x")
}

@Composable
private fun KinematicSettings(
    motionCustomization: MotionCaptureCustomization,
    motionResponsiveness: Double,
    motionSensitivity: Double,
    adaptiveRegimeStrength: Double,
    velocityResponsiveness: Double,
    accelerationResponsiveness: Double,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
    onMotionResponsiveness: (Double) -> Unit,
    onMotionSensitivity: (Double) -> Unit,
    onAdaptiveRegimeStrength: (Double) -> Unit,
    onVelocityResponsiveness: (Double) -> Unit,
    onAccelerationResponsiveness: (Double) -> Unit,
    onCustomizationMicroMotionPreservation: (Double) -> Unit,
) {
    SettingsHeader("Kinematics", "Human-scale constraints, micro-motion and dynamic response.")
    SettingSlider("Motion responsiveness", motionResponsiveness, 0.0..1.0, onMotionResponsiveness)
    SettingSlider("Motion sensitivity", motionSensitivity, 0.0..2.0, onMotionSensitivity, scale = 1.0, numberFormat = "%.2f", suffix = "x")
    SettingSlider("Adaptive motion regime", adaptiveRegimeStrength, 0.0..1.0, onAdaptiveRegimeStrength, numberFormat = "%.2f", suffix = "x")
    SettingSlider("Velocity responsiveness", velocityResponsiveness, 0.0..1.0, onVelocityResponsiveness)
    SettingSlider("Acceleration responsiveness", accelerationResponsiveness, 0.0..1.0, onAccelerationResponsiveness)
    SettingSlider("Micro-motion preservation", motionCustomization.microMotionPreservation, 0.0..1.0, onCustomizationMicroMotionPreservation)
    AdvancedSettingSlider("Foot contact strictness", motionCustomization.footContactStrictness, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            footContactStrictness = it.footContactStrictness
        }
    }
    AdvancedSettingSlider("Root motion lock", motionCustomization.rootMotionLock, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            rootMotionLock = it.rootMotionLock
        }
    }
    AdvancedSettingSlider("Semantic motion sensitivity", motionCustomization.semanticMotionSensitivity, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            semanticMotionSensitivity = it.semanticMotionSensitivity
        }
    }
}

@Composable
private fun FaceHandSettings(
    motionCustomization: MotionCaptureCustomization,
    onCustomizationFaceDetail: (Double) -> Unit,
    onCustomizationEyeDetail: (Double) -> Unit,
    onCustomizationEyeTemporalStrictness: (Double) -> Unit,
    onCustomizationHandDetail: (Double) -> Unit,
    onCustomizationHandTemporalStrictness: (Double) -> Unit,
    onCustomizationHandGeometryStrictness: (Double) -> Unit,
    onCustomizationFaceEyeFusionWeight: (Double) -> Unit,
    onCustomizationFaceEyeFusionStrictness: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader("Face + hands", "Fine landmark geometry, gaze fusion and articulation trust.")
    SettingSlider("Face detail", motionCustomization.faceDetail, 0.0..1.0, onCustomizationFaceDetail)
    SettingSlider("Eye detail", motionCustomization.eyeDetail, 0.0..1.0, onCustomizationEyeDetail)
    SettingSlider("3D eye temporal strictness", motionCustomization.eyeTemporalStrictness, 0.0..1.0, onCustomizationEyeTemporalStrictness)
    SettingSlider("Face / eye fusion weight", motionCustomization.faceEyeFusionWeight, 0.0..1.0, onCustomizationFaceEyeFusionWeight)
    SettingSlider(
        "Face / eye fusion strictness",
        motionCustomization.faceEyeFusionStrictness,
        0.0..1.0,
        onCustomizationFaceEyeFusionStrictness,
    )
    AdvancedSettingSlider(
        "IPD drift tolerance",
        motionCustomization.faceEyeIpdDriftToleranceMillimeters,
        4.0..30.0,
        onAdvancedMotionCustomization,
        scale = 1.0,
        numberFormat = "%.1f",
        suffix = " mm",
    ) { it.copyWith {
        faceEyeIpdDriftToleranceMillimeters = it.faceEyeIpdDriftToleranceMillimeters
    } }
    SettingSlider("Hand detail", motionCustomization.handDetail, 0.0..1.0, onCustomizationHandDetail)
    SettingSlider("Hand temporal strictness", motionCustomization.handTemporalStrictness, 0.0..1.0, onCustomizationHandTemporalStrictness)
    SettingSlider("Hand geometry strictness", motionCustomization.handGeometryStrictness, 0.0..1.0, onCustomizationHandGeometryStrictness)
    AdvancedSettingSlider(
        "Hand longitudinal anatomy strictness",
        motionCustomization.handLongitudinalAnatomyStrictness,
        0.0..1.0,
        onAdvancedMotionCustomization,
    ) { it.copyWith {
        handLongitudinalAnatomyStrictness =
            it.handLongitudinalAnatomyStrictness
    } }
    AdvancedSettingSlider("Head pose strictness", motionCustomization.headPoseStrictness, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            headPoseStrictness = it.headPoseStrictness
        }
    }
    AdvancedSettingSlider("Gaze priority", motionCustomization.gazePriority, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            gazePriority = it.gazePriority
        }
    }
    AdvancedSettingSlider("Landmark consistency", motionCustomization.landmarkConsistencyStrictness, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            landmarkConsistencyStrictness = it.landmarkConsistencyStrictness
        }
    }
}

@Composable
private fun SurfaceSettings(
    motionCustomization: MotionCaptureCustomization,
    onCustomizationDenseMeshDetail: (Double) -> Unit,
    onCustomizationContactDetail: (Double) -> Unit,
    onCustomizationTopologyConstraintStrength: (Double) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader("Surface", "Dense geometry quality and topology control.")
    SettingSlider("Dense mesh detail", motionCustomization.denseMeshDetail, 0.0..1.0, onCustomizationDenseMeshDetail)
    SettingSlider("Contact detail", motionCustomization.contactDetail, 0.0..1.0, onCustomizationContactDetail)
    SettingSlider("Topology constraint", motionCustomization.topologyConstraintStrength, 0.0..1.0, onCustomizationTopologyConstraintStrength)
    AdvancedSettingSlider("Adaptive mesh detail", motionCustomization.meshAdaptiveDetail, 0.0..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            meshAdaptiveDetail = it.meshAdaptiveDetail
        }
    }
}

@Composable
private fun SessionSettings(
    sessionPresentationSettings: SessionPresentationSettings,
    onSessionClipName: (String) -> Unit,
    onSessionCountdownSeconds: (Int) -> Unit,
    onRequireCalibrationBeforeRecord: (Boolean) -> Unit,
) {
    SettingsHeader("Session", "Take naming, pre-roll and calibration policy.")
    OutlinedTextField(
        value = sessionPresentationSettings.clipName,
        onValueChange = onSessionClipName,
        label = { Text("Take / clip name") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
    SettingSlider(
        "Recording countdown",
        sessionPresentationSettings.countdownSeconds.toDouble(),
        0.0..10.0,
        { onSessionCountdownSeconds(it.toInt().coerceIn(0, 10)) },
        numberFormat = "%.0f",
        suffix = " s",
    )
    SettingsSwitch(
        title = "Require calibration before record",
        description = "Prevent the capture state from starting until the rig is calibrated.",
        checked = sessionPresentationSettings.requireCalibrationBeforeRecord,
        onCheckedChange = onRequireCalibrationBeforeRecord,
    )
}

@Composable
private fun SystemSettings(
    motionCustomization: MotionCaptureCustomization,
    showEvidenceMatrix: Boolean,
    showDiagnostics: Boolean,
    showViewportGrid: Boolean,
    showPerformanceTelemetry: Boolean,
    onShowEvidenceMatrix: (Boolean) -> Unit,
    onShowDiagnostics: (Boolean) -> Unit,
    onShowViewportGrid: (Boolean) -> Unit,
    onShowPerformanceTelemetry: (Boolean) -> Unit,
    onCustomizationPreset: (String) -> Unit,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
) {
    SettingsHeader("System", "High-level quality profiles and operator presentation.")
    SettingsPresetRow(
        items = listOf("QUALITY", "BALANCED", "LOW_LATENCY"),
        active = when {
            motionCustomization.inferenceFps >= 50 && motionCustomization.bodyDetail >= 0.95 -> "QUALITY"
            motionCustomization.inferenceFps <= 30 && motionCustomization.bodyDetail < 0.90 -> "LOW_LATENCY"
            else -> "BALANCED"
        },
        onSelect = onCustomizationPreset,
    )
    AdvancedSettingSlider("Thermal quality floor", motionCustomization.thermalQualityFloor, 0.20..1.0, onAdvancedMotionCustomization) {
        it.copyWith {
            thermalQualityFloor = it.thermalQualityFloor
        }
    }

    SettingsSwitch(
        "Evidence matrix",
        "Show the live confidence matrix above the viewfinder.",
        showEvidenceMatrix,
        onShowEvidenceMatrix,
    )
    SettingsSwitch(
        "Diagnostics",
        "Show subsystem warning strips in the operator stage.",
        showDiagnostics,
        onShowDiagnostics,
    )
    SettingsSwitch(
        "Viewport grid",
        "Show the spatial reference grid behind the reconstructed subject.",
        showViewportGrid,
        onShowViewportGrid,
    )
    SettingsSwitch(
        "Performance telemetry",
        "Keep runtime performance values visible in the preview chrome.",
        showPerformanceTelemetry,
        onShowPerformanceTelemetry,
    )
}

@Composable
private fun SettingsHeader(
    title: String,
    subtitle: String,
) {
    GlassCard(intensity = 0.52f) {
        Column(
            Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(3.dp),
        ) {
            Text(title.uppercase(), style = MaterialTheme.typography.titleMedium)
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun SettingsPresetRow(
    items: List<String>,
    active: String,
    onSelect: (String) -> Unit,
) {
    Row(
        Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items.forEach { item ->
            Surface(
                onClick = { onSelect(item) },
                shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp),
                color = if (item == active) {
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
private fun SettingsSwitch(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    GlassCard(intensity = 0.54f) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 11.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(
                Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(title, style = MaterialTheme.typography.labelLarge)
                Text(
                    description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.width(10.dp))
            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange,
            )
        }
    }
}

@Composable
private fun SettingsOptionRow(
    title: String,
    selected: String,
    options: List<String>,
    onSelected: (String) -> Unit,
) {
    GlassCard(intensity = 0.54f) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    title,
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    selected,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            }

            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                options.forEach { option ->
                    Surface(
                        onClick = { onSelected(option) },
                        shape = androidx.compose.foundation.shape.RoundedCornerShape(15.dp),
                        color =
                            if (selected.replace(' ', '_') == option) {
                                MaterialTheme.colorScheme.primaryContainer
                            } else {
                                MaterialTheme.colorScheme.surfaceContainerHigh
                            },
                    ) {
                        Text(
                            option.replace('_', ' '),
                            Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 9.dp,
                            ),
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingSlider(
    title: String,
    value: Double,
    range: ClosedFloatingPointRange<Double>,
    onValueChange: (Double) -> Unit,
    scale: Double = 100.0,
    numberFormat: String = "%.0f",
    suffix: String = "%",
) {
    GlassCard(intensity = 0.54f) {
        Column(
            Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(title, style = MaterialTheme.typography.labelLarge)
                Text(
                    numberFormat.format(value * scale) + suffix,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            androidx.compose.material3.Slider(
                value = value.toFloat().coerceIn(
                    range.start.toFloat(),
                    range.endInclusive.toFloat(),
                ),
                onValueChange = {
                    onValueChange(
                        it.toDouble().coerceIn(
                            range.start,
                            range.endInclusive,
                        ),
                    )
                },
                valueRange = range.start.toFloat()..range.endInclusive.toFloat(),
            )
        }
    }
}

@Composable
private fun AdvancedSettingSlider(
    title: String,
    value: Double,
    range: ClosedFloatingPointRange<Double>,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
    scale: Double = 100.0,
    numberFormat: String = "%.0f",
    suffix: String = "%",
    change: (MotionCaptureCustomization) -> MotionCaptureCustomization,
) {
    SettingSlider(
        title = title,
        value = value,
        range = range,
        onValueChange = { next ->
            onAdvancedMotionCustomization {
                val current = it
                val updatedBase = change(current)
                applyAdvancedValue(updatedBase, title, next)
            }
        },
        scale = scale,
        numberFormat = numberFormat,
        suffix = suffix,
    )
}

/**
 * Small adapter keeps the shared category components declarative. The actual setter is encoded
 * by the title, but only against fields already present in MotionCaptureCustomization.
 */
private fun applyAdvancedValue(
    customization: MotionCaptureCustomization,
    title: String,
    next: Double,
): MotionCaptureCustomization =
    when (title) {
        "Occlusion recovery confidence" ->
            customization.copyWith {
                occlusionRecoveryConfidence = next
            }
        "Evidence adaptivity" ->
            customization.copyWith {
                multiViewEvidenceAdaptivity = next
            }
        "Camera drift response" ->
            customization.copyWith {
                cameraDriftResponse = next
            }
        "Pose prior strength" ->
            customization.copyWith {
                posePriorStrength = next
            }
        "Retargeting strictness" ->
            customization.copyWith {
                retargetingStrictness = next
            }
        "Scale preservation" ->
            customization.copyWith {
                scalePreservation = next
            }
        "Foot contact strictness" ->
            customization.copyWith {
                footContactStrictness = next
            }
        "Root motion lock" ->
            customization.copyWith {
                rootMotionLock = next
            }
        "Semantic motion sensitivity" ->
            customization.copyWith {
                semanticMotionSensitivity = next
            }
        "Head pose strictness" ->
            customization.copyWith {
                headPoseStrictness = next
            }
        "Gaze priority" ->
            customization.copyWith {
                gazePriority = next
            }
        "Landmark consistency" ->
            customization.copyWith {
                landmarkConsistencyStrictness = next
            }
        "Adaptive mesh detail" ->
            customization.copyWith {
                meshAdaptiveDetail = next
            }
        "Live preview detail" ->
            customization.copyWith {
                liveMotionPreviewDetail = next
            }
        "Thermal quality floor" ->
            customization.copyWith {
                thermalQualityFloor = next
            }
        "Sub-frame alignment strictness" ->
            customization.copyWith {
                subframeAlignmentStrictness = next
            }
        "Rolling-shutter compensation" ->
            customization.copyWith {
                rollingShutterCompensation = next
            }
        else -> customization
    }
