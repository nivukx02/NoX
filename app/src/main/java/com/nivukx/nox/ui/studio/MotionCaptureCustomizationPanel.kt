package com.nivukx.nox.ui.studio

import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.CameraDynamicRangeMode
import com.nivukx.nox.core.model.CameraExposureMode
import com.nivukx.nox.core.model.CameraResolutionPreset
import com.nivukx.nox.core.model.CaptureWorkflowPreset
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings

/**
 * High-fidelity capture intent controls.
 *
 * These controls modify named production settings only. The native/runtime policy remains the
 * authority for whether the selected intent can be sustained under current evidence.
 */
@Composable
fun MotionCaptureCustomizationPanel(
    settings: HighEndMotionCaptureSettings,
    onSettingsChanged: (HighEndMotionCaptureSettings) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = false,
        elevated = false,
        modifier = modifier.fillMaxWidth(),
        contentPadding = 16.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Capture Intelligence",
                style = MaterialTheme.typography.titleMedium,
            )

            Text(
                text = "Operator workflows",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "Presets apply a high-end evidence policy while preserving every underlying control below.",
                style = MaterialTheme.typography.bodySmall,
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(
                        rememberScrollState(),
                    ),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                CaptureWorkflowPreset.entries.forEach { preset ->
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        tonalElevation = 2.dp,
                        modifier = Modifier
                            .width(190.dp)
                            .heightIn(min = 72.dp)
                            .clickable(
                                role = Role.Button,
                            ) {
                                onSettingsChanged(
                                    settings.applyWorkflowPreset(preset),
                                )
                            },
                    ) {
                        Column(
                            modifier = Modifier.padding(
                                horizontal = 10.dp,
                                vertical = 8.dp,
                            ),
                            verticalArrangement = Arrangement.spacedBy(2.dp),
                        ) {
                            Text(
                                text = preset.label,
                                style = MaterialTheme.typography.labelLarge,
                            )
                            Text(
                                text = preset.description,
                                style = MaterialTheme.typography.labelSmall,
                                maxLines = 2,
                            )
                        }
                    }
                }
            }
            Text(
                text = "Tune capture intent without bypassing runtime quality gates.",
                style = MaterialTheme.typography.bodySmall,
            )

            CaptureIntentSummary(settings = settings)

            FrontierSystemsCustomizationPanel(
                settings = settings,
                onSettingsChanged = onSettingsChanged,
                modifier = Modifier.fillMaxWidth(),
            )

            Text(
                text = "Evidence authority policy",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text =
                    "Control how aggressively NoX admits geometry into the authoritative solve. " +
                        "Lower blur tolerance is stricter. Higher geometry and evidence thresholds " +
                        "never manufacture precision; they only raise the admission bar.",
                style = MaterialTheme.typography.bodySmall,
            )
            CapturePrioritySlider(
                label =
                    "Maximum motion-blur risk (" +
                        (settings.maximumMotionBlurRiskForAuthority * 100.0).toInt() +
                        "%)",
                value = settings.maximumMotionBlurRiskForAuthority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            maximumMotionBlurRiskForAuthority = it,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label =
                    "Minimum epipolar consistency (" +
                        (settings.minimumEpipolarConsistencyForAuthority * 100.0).toInt() +
                        "%)",
                value = settings.minimumEpipolarConsistencyForAuthority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            minimumEpipolarConsistencyForAuthority = it,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label =
                    "Minimum metric evidence authority (" +
                        (settings.minimumMetricEvidenceAuthorityForAuthority * 100.0).toInt() +
                        "%)",
                value = settings.minimumMetricEvidenceAuthorityForAuthority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            minimumMetricEvidenceAuthorityForAuthority = it,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label =
                    "Minimum capture authority score (" +
                        (settings.minimumCaptureAuthorityScore * 100.0).toInt() +
                        "%)",
                value = settings.minimumCaptureAuthorityScore,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            minimumCaptureAuthorityScore = it,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label =
                    "Precision authority score (" +
                        (settings.precisionAuthorityMinimumScore * 100.0).toInt() +
                        "%)",
                value = settings.precisionAuthorityMinimumScore,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            precisionAuthorityMinimumScore = it,
                        ),
                    )
                },
            )
            CaptureSampleBudgetSlider(
                value = settings.precisionAuthorityMinimumSamples,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            precisionAuthorityMinimumSamples = it,
                        ),
                    )
                },
            )

            Text(
                text = "Professional camera acquisition",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text =
                    "Tune the physical capture path for motion clarity. Unsupported features are resolved safely by the camera capability engine.",
                style = MaterialTheme.typography.bodySmall,
            )

            Text(
                text = "Resolution",
                style = MaterialTheme.typography.labelLarge,
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(
                        rememberScrollState(),
                    ),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                listOf(
                    CameraResolutionPreset.AUTO,
                    CameraResolutionPreset.FULL_HD_1080P,
                    CameraResolutionPreset.UHD_4K,
                    CameraResolutionPreset.MAX_SUPPORTED,
                ).forEach { preset ->
                    val selected =
                        settings.cameraResolutionPreset == preset
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        tonalElevation =
                            if (selected) 5.dp else 1.dp,
                        modifier =
                            Modifier
                                .width(132.dp)
                                .heightIn(min = 48.dp)
                                .clickable(role = Role.Button) {
                                    onSettingsChanged(
                                        settings.copy(
                                            cameraResolutionPreset = preset,
                                        ),
                                    )
                                },
                    ) {
                        Text(
                            text =
                                when (preset) {
                                    CameraResolutionPreset.AUTO ->
                                        "AUTO"
                                    CameraResolutionPreset.FULL_HD_1080P ->
                                        "1080P"
                                    CameraResolutionPreset.UHD_4K ->
                                        "4K"
                                    CameraResolutionPreset.MAX_SUPPORTED ->
                                        "MAX"
                                    CameraResolutionPreset.HD_720P ->
                                        "720P"
                                },
                            modifier = Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 12.dp,
                            ),
                            style = MaterialTheme.typography.labelLarge,
                        )
                    }
                }
            }

            Text(
                text = "Dynamic range",
                style = MaterialTheme.typography.labelLarge,
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(
                        rememberScrollState(),
                    ),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                listOf(
                    CameraDynamicRangeMode.STANDARD,
                    CameraDynamicRangeMode.TEN_BIT_PREFERRED,
                ).forEach { mode ->
                    val selected =
                        settings.cameraDynamicRangeMode == mode
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        tonalElevation =
                            if (selected) 5.dp else 1.dp,
                        modifier =
                            Modifier
                                .weight(1f)
                                .heightIn(min = 48.dp)
                                .clickable(role = Role.Button) {
                                    onSettingsChanged(
                                        settings.copy(
                                            cameraDynamicRangeMode =
                                                mode,
                                        ),
                                    )
                                },
                    ) {
                        Text(
                            text =
                                if (
                                    mode ==
                                        CameraDynamicRangeMode
                                            .TEN_BIT_PREFERRED
                                ) {
                                    "10-BIT PREFERRED"
                                } else {
                                    "STANDARD"
                                },
                            modifier = Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 12.dp,
                            ),
                            style = MaterialTheme.typography.labelLarge,
                        )
                    }
                }
            }
            Text(
                text =
                    "10-bit is enabled only when the camera advertises a supported dynamic-range profile; otherwise NoX keeps the standard path.",
                style = MaterialTheme.typography.bodySmall,
            )

            Text(
                text = "Exposure",
                style = MaterialTheme.typography.labelLarge,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                listOf(
                    CameraExposureMode.AUTO,
                    CameraExposureMode.MANUAL,
                ).forEach { mode ->
                    val selected =
                        settings.cameraExposureMode == mode
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        tonalElevation =
                            if (selected) 5.dp else 1.dp,
                        modifier =
                            Modifier
                                .weight(1f)
                                .heightIn(min = 48.dp)
                                .clickable(role = Role.Button) {
                                    onSettingsChanged(
                                        settings.copy(
                                            cameraExposureMode = mode,
                                        ),
                                    )
                                },
                    ) {
                        Text(
                            text =
                                if (mode ==
                                    CameraExposureMode.MANUAL
                                ) {
                                    "MANUAL / MOTION"
                                } else {
                                    "AUTO"
                                },
                            modifier = Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 12.dp,
                            ),
                            style = MaterialTheme.typography.labelLarge,
                        )
                    }
                }
            }
            if (android.os.Build.VERSION.SDK_INT >= 36) {
                Surface(
                    shape = RoundedCornerShape(14.dp),
                    tonalElevation =
                        if (
                            settings.cameraExposureMode ==
                                CameraExposureMode.SENSITIVITY_PRIORITY
                        ) {
                            5.dp
                        } else {
                            1.dp
                        },
                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .heightIn(min = 48.dp)
                            .clickable(role = Role.Button) {
                                onSettingsChanged(
                                    settings.copy(
                                        cameraExposureMode =
                                            CameraExposureMode.SENSITIVITY_PRIORITY,
                                    ),
                                )
                            },
                ) {
                    Column(
                        Modifier.padding(
                            horizontal = 12.dp,
                            vertical = 10.dp,
                        ),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Text(
                            "ISO PRIORITY + AE",
                            style = MaterialTheme.typography.labelLarge,
                        )
                        Text(
                            "Android 16 hybrid exposure • AE may adjust shutter while ISO stays fixed",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }


            CapturePrioritySlider(
                label = "Manual exposure time",
                value =
                    (
                        (
                            settings.cameraManualExposureMicros -
                                250L
                        ).coerceIn(0L, 8_333L) /
                            8_083.0
                    ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            cameraManualExposureMicros =
                                250L +
                                    (it * 8_083.0)
                                        .toLong()
                                        .coerceIn(0L, 8_083L),
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Manual sensor gain / ISO",
                value =
                    (
                        (
                            settings.cameraManualIso.toDouble()
                                .coerceIn(100.0, 6400.0) -
                                100.0
                        ) /
                            6300.0
                    ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            cameraManualIso =
                                (
                                    100.0 +
                                        it * 6300.0
                                    ).toInt()
                                        .coerceIn(100, 6400),
                        ),
                    )
                },
            )

            PolicySwitch(
                label = "Preserve hardware sensor timestamps",
                checked = settings.cameraPreserveSensorTimestamp,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            cameraPreserveSensorTimestamp = it,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "Optical stabilization",
                checked = settings.cameraOisEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(cameraOisEnabled = it),
                    )
                },
            )
            PolicySwitch(
                label = "Electronic stabilization",
                checked = settings.cameraElectronicStabilizationEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            cameraElectronicStabilizationEnabled = it,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "RAW sidecar",
                checked = settings.cameraRawSidecarEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            cameraRawSidecarEnabled = it,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "Depth sidecar",
                checked = settings.cameraDepthSidecarEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            cameraDepthSidecarEnabled = it,
                        ),
                    )
                },
            )

            Text(
                text = "Distributed camera fabric",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text =
                    "Keep the global federation broad while bounding the expensive authoritative reconstruction view set. Timing thresholds are applied before precision capture is admitted.",
                style = MaterialTheme.typography.bodySmall,
            )

            CapturePrioritySlider(
                label =
                    "Global camera envelope (" +
                        settings.distributedGlobalCameraEnvelope +
                        ")",
                value =
                    (
                        settings.distributedGlobalCameraEnvelope - 2
                    ) / 126.0,
                onValueChanged = { normalized ->
                    val envelope =
                        (
                            2 +
                                normalized * 126.0
                            ).toInt()
                                .coerceIn(2, 128)

                    onSettingsChanged(
                        settings.copy(
                            distributedGlobalCameraEnvelope =
                                envelope,
                            distributedAuthoritativeCameraBudget =
                                settings.distributedAuthoritativeCameraBudget
                                    .coerceIn(
                                        2,
                                        envelope.coerceAtMost(32),
                                    ),
                            distributedMaximumShards =
                                settings.distributedMaximumShards
                                    .coerceIn(
                                        1,
                                        envelope.coerceAtMost(16),
                                    ),
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label =
                    "Authoritative views (" +
                        settings.distributedAuthoritativeCameraBudget +
                        ")",
                value =
                    (
                        settings.distributedAuthoritativeCameraBudget - 2
                    ).toDouble() /
                        (
                            settings.distributedGlobalCameraEnvelope
                                .coerceAtMost(32) - 2
                        ).coerceAtLeast(1),
                onValueChanged = { normalized ->
                    val maximum =
                        settings.distributedGlobalCameraEnvelope
                            .coerceAtMost(32)

                    onSettingsChanged(
                        settings.copy(
                            distributedAuthoritativeCameraBudget =
                                (
                                    2 +
                                        normalized *
                                            (maximum - 2)
                                    ).toInt()
                                    .coerceIn(2, maximum),
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label =
                    "Camera shards (" +
                        settings.distributedMaximumShards +
                        ")",
                value =
                    (
                        settings.distributedMaximumShards - 1
                    ).toDouble() /
                        (
                            settings.distributedGlobalCameraEnvelope
                                .coerceAtMost(16) - 1
                        ).coerceAtLeast(1),
                onValueChanged = { normalized ->
                    val maximum =
                        settings.distributedGlobalCameraEnvelope
                            .coerceAtMost(16)

                    onSettingsChanged(
                        settings.copy(
                            distributedMaximumShards =
                                (
                                    1 +
                                        normalized *
                                            (maximum - 1)
                                    ).toInt()
                                    .coerceIn(1, maximum),
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Minimum authority health",
                value = settings.distributedMinimumAuthorityHealth,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            distributedMinimumAuthorityHealth = it,
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Minimum authority synchronization",
                value =
                    settings.distributedMinimumAuthoritySynchronization,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            distributedMinimumAuthoritySynchronization =
                                it,
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label =
                    "Maximum temporal residual (" +
                        String.format(
                            "%.1f",
                            settings.distributedMaximumTemporalResidualMs,
                        ) +
                        " ms)",
                value =
                    (
                        settings.distributedMaximumTemporalResidualMs - 1.0
                    ) / 49.0,
                onValueChanged = { normalized ->
                    onSettingsChanged(
                        settings.copy(
                            distributedMaximumTemporalResidualMs =
                                1.0 +
                                    normalized * 49.0,
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label =
                    "Minimum transport headroom (" +
                        (
                            settings.distributedMinimumTransportHeadroom *
                                100.0
                            ).toInt() +
                        "%)",
                value =
                    settings.distributedMinimumTransportHeadroom,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            distributedMinimumTransportHeadroom =
                                it,
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label =
                    "Maximum temporal outliers (" +
                        (
                            settings.distributedMaximumTemporalOutlierRatio *
                                100.0
                            ).toInt() +
                        "%)",
                value =
                    settings.distributedMaximumTemporalOutlierRatio,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            distributedMaximumTemporalOutlierRatio =
                                it,
                        ),
                    )
                },
            )

            CapturePrioritySlider(

                label = "Micro-motion preservation",
                value = settings.preserveMicroMotion,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withMicroMotionPreservation(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Temporal continuity",
                value = settings.temporalContinuityPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalContinuity(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Surface detail",
                value = settings.surfaceDetailPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withSurfaceDetail(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Face + eye detail",
                value = settings.faceEyeDetailPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withFaceEyeDetail(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Hand detail",
                value = settings.handDetailPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withHandDetail(it),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Motion sensitivity",
                value = settings.motionSensitivity,
                onValueChanged = {
                    onSettingsChanged(settings.copy(motionSensitivity = it))
                },
            )
            CapturePrioritySlider(
                label = "Camera-motion suppression",
                value = settings.cameraMotionSuppression,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(cameraMotionSuppression = it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Hand geometry strictness",
                value = settings.handGeometryStrictness,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(handGeometryStrictness = it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Contact constraint",
                value = settings.contactConstraintStrength,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(contactConstraintStrength = it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Dense mesh detail",
                value = settings.denseMeshDetail,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(denseMeshDetail = it),
                    )
                },
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        text = "Semantic silhouette evidence",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        text =
                            "Scene-level person segmentation used as bounded support for dense surface evidence.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Switch(
                    checked = settings.personSegmentationEnabled,
                    onCheckedChange = { enabled ->
                        onSettingsChanged(
                            settings.copy(
                                personSegmentationEnabled = enabled,
                            ),
                        )
                    },
                )
            }

            CapturePrioritySlider(
                label = "Silhouette refresh",
                value =
                    (
                        (
                            settings.personSegmentationCadenceFrames -
                                1
                        ) / 119.0
                    ).coerceIn(0.0, 1.0),
                onValueChanged = { normalized ->
                    onSettingsChanged(
                        settings.copy(
                            personSegmentationCadenceFrames =
                                (1 + normalized * 119.0)
                                    .toInt()
                                    .coerceIn(1, 120),
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Silhouette scheduling priority",
                value = settings.personSegmentationPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            personSegmentationPriority = it,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum parallax",
                value = (settings.minimumParallaxDegrees / 45.0)
                    .coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            minimumParallaxDegrees = it * 45.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Reprojection gate",
                value = ((settings.reprojectionGatePx - 0.25) / 19.75)
                    .coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            reprojectionGatePx = 0.25 + it * 19.75,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "3D uncertainty budget",
                value = (
                    (settings.reconstructionUncertaintyMeters - 0.001) / 0.999
                    ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.copy(
                            reconstructionUncertaintyMeters =
                                0.001 + it * 0.999,
                        ),
                    )
                },
            )
            Text(
                text = "Surface reconstruction contracts",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "Bound dense-surface strain and temporal topology degradation before it can contaminate the reconstructed volume.",
                style = MaterialTheme.typography.bodySmall,
            )
            CapturePrioritySlider(
                label = "Maximum surface strain P95",
                value = (
                    (settings.maximumSurfaceStrainP95Ratio - 0.05) / 0.95
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withSurfaceContract(
                            maximumStrainP95Ratio = 0.05 + it * 0.95,
                            minimumTemporalStability =
                                settings.minimumSurfaceTemporalStability,
                            minimumTopologyContinuity =
                                settings.minimumSurfaceTopologyContinuity,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum surface temporal stability",
                value = settings.minimumSurfaceTemporalStability,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withSurfaceContract(
                            maximumStrainP95Ratio =
                                settings.maximumSurfaceStrainP95Ratio,
                            minimumTemporalStability = it,
                            minimumTopologyContinuity =
                                settings.minimumSurfaceTopologyContinuity,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum surface topology continuity",
                value = settings.minimumSurfaceTopologyContinuity,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withSurfaceContract(
                            maximumStrainP95Ratio =
                                settings.maximumSurfaceStrainP95Ratio,
                            minimumTemporalStability =
                                settings.minimumSurfaceTemporalStability,
                            minimumTopologyContinuity = it,
                        ),
                    )
                },
            )
            Text(
                text = "Dense landmark evidence contracts",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "Tune how aggressively NoX requires dense visibility, low uncertainty, independent views, and temporal stability before authorizing dense geometry.",
                style = MaterialTheme.typography.bodySmall,
            )
            CapturePrioritySlider(
                label = "Dense coverage strictness",
                value = settings.denseLandmarkCoverageStrictness,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withDenseLandmarkContractStrictness(
                            coverage = it,
                            uncertainty =
                                settings.denseLandmarkUncertaintyStrictness,
                            crossView =
                                settings.denseLandmarkCrossViewStrictness,
                            temporal =
                                settings.denseLandmarkTemporalStrictness,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Dense uncertainty strictness",
                value = settings.denseLandmarkUncertaintyStrictness,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withDenseLandmarkContractStrictness(
                            coverage =
                                settings.denseLandmarkCoverageStrictness,
                            uncertainty = it,
                            crossView =
                                settings.denseLandmarkCrossViewStrictness,
                            temporal =
                                settings.denseLandmarkTemporalStrictness,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Dense cross-view strictness",
                value = settings.denseLandmarkCrossViewStrictness,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withDenseLandmarkContractStrictness(
                            coverage =
                                settings.denseLandmarkCoverageStrictness,
                            uncertainty =
                                settings.denseLandmarkUncertaintyStrictness,
                            crossView = it,
                            temporal =
                                settings.denseLandmarkTemporalStrictness,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Dense temporal strictness",
                value = settings.denseLandmarkTemporalStrictness,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withDenseLandmarkContractStrictness(
                            coverage =
                                settings.denseLandmarkCoverageStrictness,
                            uncertainty =
                                settings.denseLandmarkUncertaintyStrictness,
                            crossView =
                                settings.denseLandmarkCrossViewStrictness,
                            temporal = it,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Maximum runtime faults per window",
                value = (
                    settings.maximumRuntimeFaultsPerWindow - 1
                ) / 31.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withRuntimeFaultBudget(
                            1 + (it * 31.0).toInt(),
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Occlusion recovery",
                value = settings.occlusionRecoveryPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withOcclusionRecovery(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Interaction + contact",
                value = settings.interactionContactPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withInteractionContact(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Camera diversity",
                value = settings.cameraDiversityPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withCameraDiversity(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Face expression detail",
                value = settings.faceExpressionPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withFaceExpression(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Hand contact anatomy",
                value = settings.handContactPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withHandContact(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Identity persistence",
                value = settings.identityPersistencePriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withIdentityPersistence(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Capture integrity strictness",
                value = settings.captureIntegrityStrictness,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withCaptureIntegrity(it),
                    )
                },
            )

            PolicySwitch(
                label = "Adaptive quality",
                checked = settings.adaptiveQualityEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.withQualityPolicy(
                            adaptive = it,
                            locked = false,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "Quality lock",
                checked = settings.qualityLockEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.withQualityPolicy(
                            adaptive = false,
                            locked = it,
                        ),
                    )
                },
            )

            Text(
                text = "Operator telemetry",
                style = MaterialTheme.typography.titleSmall,
            )
            PolicySwitch(
                label = "Lighting telemetry",
                checked = settings.lightingTelemetryEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(lightingTelemetryEnabled = it),
                    )
                },
            )
            PolicySwitch(
                label = "Diagnostics overlay",
                checked = settings.diagnosticsOverlayEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(diagnosticsOverlayEnabled = it),
                    )
                },
            )
            PolicySwitch(
                label = "Uncertainty overlay",
                checked = settings.uncertaintyOverlayEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(uncertaintyOverlayEnabled = it),
                    )
                },
            )
            PolicySwitch(
                label = "Camera frusta",
                checked = settings.cameraFrustaEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(cameraFrustaEnabled = it),
                    )
                },
            )

            Text(
                text = "Temporal Supersampling",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "Tune asynchronous multi-camera timing without bypassing reconstruction quality gates.",
                style = MaterialTheme.typography.bodySmall,
            )

            PolicySwitch(
                label = "Interleaved temporal fusion",
                checked = settings.interleavedTemporalFusionEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.withTemporalSupersamplingEnabled(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Temporal supersampling priority",
                value = settings.temporalSupersamplingPriority,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalSupersamplingPriority(it),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Prediction budget",
                value =
                    (
                        settings.temporalFusionPredictionBudgetMillis - 8.0
                    ) / 242.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalFusionPredictionBudget(
                            8.0 + it * 242.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Track retention",
                value =
                    (
                        settings.temporalFusionTrackRetentionMillis - 500.0
                    ) / 9500.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalFusionTrackRetention(
                            500.0 + it * 9500.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Robust outlier gate",
                value =
                    (
                        settings.temporalFusionHardOutlierGate - 1.0
                    ) / 7.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalFusionOutlierGate(
                            1.0 + it * 7.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Confidence half-life",
                value =
                    (
                        settings.temporalFusionConfidenceHalfLifeMillis - 40.0
                    ) / 960.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalFusionConfidenceHalfLife(
                            40.0 + it * 960.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Temporal fusion window",
                value =
                    (
                        settings.interleavedTemporalWindowMillis - 8.0
                    ) / 242.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalFusionWindow(
                            8.0 + it * 242.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Prediction horizon",
                value =
                    settings.interleavedPredictionHorizonMillis / 100.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withPredictionHorizon(it * 100.0),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Maximum sample skew",
                value =
                    (
                        settings.interleavedMaximumSampleSkewMillis - 1.0
                    ) / 99.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withMaximumSampleSkew(
                            1.0 + it * 99.0,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Outlier rejection envelope",
                value =
                    (
                        settings.interleavedOutlierThresholdPixels - 2.0
                    ) / 126.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withOutlierEnvelopePixels(
                            2.0 + it * 126.0,
                        ),
                    )
                },
            )

            Text(
                text = "Reference-grade metrology",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "A 2 mm target is evidence-gated by actual camera geometry, timing and calibration. It is not a visual preset.",
                style = MaterialTheme.typography.bodySmall,
            )
            PolicySwitch(
                label = "Reference-grade 2 mm mode",
                checked = settings.referenceGradePrecisionEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.withReferenceGradePrecision(it),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Reference P95 target",
                value = (
                    (settings.referenceGradeTargetP95ErrorMillimeters - 0.1) / 1.9
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withReferenceGradeContract(
                            targetP95Millimeters = 0.1 + it * 1.9,
                            minimumParallaxDegrees =
                                settings.referenceGradeMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.referenceGradeMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.referenceGradeMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.referenceGradeMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.referenceGradeMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum reference parallax",
                value = (
                    (settings.referenceGradeMinimumParallaxDegrees - 2.0) / 43.0
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withReferenceGradeContract(
                            targetP95Millimeters =
                                settings.referenceGradeTargetP95ErrorMillimeters,
                            minimumParallaxDegrees = 2.0 + it * 43.0,
                            minimumCalibrationConfidence =
                                settings.referenceGradeMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.referenceGradeMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.referenceGradeMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.referenceGradeMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum reference calibration",
                value = settings.referenceGradeMinimumCalibrationConfidence,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withReferenceGradeContract(
                            targetP95Millimeters =
                                settings.referenceGradeTargetP95ErrorMillimeters,
                            minimumParallaxDegrees =
                                settings.referenceGradeMinimumParallaxDegrees,
                            minimumCalibrationConfidence = it,
                            minimumSynchronizationConfidence =
                                settings.referenceGradeMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.referenceGradeMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.referenceGradeMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum reference sync",
                value = settings.referenceGradeMinimumSynchronizationConfidence,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withReferenceGradeContract(
                            targetP95Millimeters =
                                settings.referenceGradeTargetP95ErrorMillimeters,
                            minimumParallaxDegrees =
                                settings.referenceGradeMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.referenceGradeMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence = it,
                            minimumViewDiversity =
                                settings.referenceGradeMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.referenceGradeMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum view diversity",
                value = settings.referenceGradeMinimumViewDiversity,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withReferenceGradeContract(
                            targetP95Millimeters =
                                settings.referenceGradeTargetP95ErrorMillimeters,
                            minimumParallaxDegrees =
                                settings.referenceGradeMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.referenceGradeMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.referenceGradeMinimumSynchronizationConfidence,
                            minimumViewDiversity = it,
                            minimumCrossViewAgreement =
                                settings.referenceGradeMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum cross-view agreement",
                value = settings.referenceGradeMinimumCrossViewAgreement,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withReferenceGradeContract(
                            targetP95Millimeters =
                                settings.referenceGradeTargetP95ErrorMillimeters,
                            minimumParallaxDegrees =
                                settings.referenceGradeMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.referenceGradeMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.referenceGradeMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.referenceGradeMinimumViewDiversity,
                            minimumCrossViewAgreement = it,
                        ),
                    )
                },
            )

            Text(
                text = "Ultra-precision qualification",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "This layer turns the sub-millimeter goal into measurable geometry, timing and reference-evidence gates. It does not claim sub-millimeter accuracy on arbitrary phones.",
                style = MaterialTheme.typography.bodySmall,
            )
            PolicySwitch(
                label = "Enable ultra-precision evidence",
                checked = settings.ultraPrecisionEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.withUltraPrecisionQualification(
                            enabled = it,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "Require empirical qualification",
                checked =
                    settings.ultraPrecisionQualificationMode,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            ultraPrecisionQualificationMode =
                                it && settings.ultraPrecisionEnabled,
                        ),
                    )
                },
            )
            Text(
                text =
                    when {
                        !settings.ultraPrecisionEnabled ->
                            "Precision evidence: inactive"
                        settings.ultraPrecisionQualificationMode ->
                            "Precision evidence: qualification gate active. A matched measured reference is required."
                        else ->
                            "Precision evidence: runtime advisory mode. Geometry and uncertainty remain visible without claiming a pass."
                    },
                style = MaterialTheme.typography.bodySmall,
            )
            CapturePrioritySlider(
                label = "Ultra P95 target (0.05–5.0 mm)",
                value =
                    (
                        (settings.ultraPrecisionTargetP95ErrorMillimeters - 0.05) /
                            4.95
                    ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withUltraPrecisionContract(
                            targetP95Millimeters =
                                0.05 + it * 4.95,
                            minimumBaselineMeters =
                                settings.ultraPrecisionMinimumBaselineMeters,
                            minimumParallaxDegrees =
                                settings.ultraPrecisionMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.ultraPrecisionMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.ultraPrecisionMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.ultraPrecisionMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.ultraPrecisionMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum camera baseline (0.05–2.5 m)",
                value =
                    (
                        settings.ultraPrecisionMinimumBaselineMeters - 0.05
                    ) / 2.45,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withUltraPrecisionContract(
                            targetP95Millimeters =
                                settings.ultraPrecisionTargetP95ErrorMillimeters,
                            minimumBaselineMeters =
                                0.05 + it * 2.45,
                            minimumParallaxDegrees =
                                settings.ultraPrecisionMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.ultraPrecisionMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.ultraPrecisionMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.ultraPrecisionMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.ultraPrecisionMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum parallax",
                value =
                    (
                        settings.ultraPrecisionMinimumParallaxDegrees - 1.0
                    ) / 88.0,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withUltraPrecisionContract(
                            targetP95Millimeters =
                                settings.ultraPrecisionTargetP95ErrorMillimeters,
                            minimumBaselineMeters =
                                settings.ultraPrecisionMinimumBaselineMeters,
                            minimumParallaxDegrees =
                                1.0 + it * 88.0,
                            minimumCalibrationConfidence =
                                settings.ultraPrecisionMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.ultraPrecisionMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.ultraPrecisionMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.ultraPrecisionMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Calibration evidence threshold",
                value =
                    settings.ultraPrecisionMinimumCalibrationConfidence,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withUltraPrecisionContract(
                            targetP95Millimeters =
                                settings.ultraPrecisionTargetP95ErrorMillimeters,
                            minimumBaselineMeters =
                                settings.ultraPrecisionMinimumBaselineMeters,
                            minimumParallaxDegrees =
                                settings.ultraPrecisionMinimumParallaxDegrees,
                            minimumCalibrationConfidence = it,
                            minimumSynchronizationConfidence =
                                settings.ultraPrecisionMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.ultraPrecisionMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.ultraPrecisionMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Synchronization evidence threshold",
                value =
                    settings.ultraPrecisionMinimumSynchronizationConfidence,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withUltraPrecisionContract(
                            targetP95Millimeters =
                                settings.ultraPrecisionTargetP95ErrorMillimeters,
                            minimumBaselineMeters =
                                settings.ultraPrecisionMinimumBaselineMeters,
                            minimumParallaxDegrees =
                                settings.ultraPrecisionMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.ultraPrecisionMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence = it,
                            minimumViewDiversity =
                                settings.ultraPrecisionMinimumViewDiversity,
                            minimumCrossViewAgreement =
                                settings.ultraPrecisionMinimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Cross-view agreement threshold",
                value =
                    settings.ultraPrecisionMinimumCrossViewAgreement,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withUltraPrecisionContract(
                            targetP95Millimeters =
                                settings.ultraPrecisionTargetP95ErrorMillimeters,
                            minimumBaselineMeters =
                                settings.ultraPrecisionMinimumBaselineMeters,
                            minimumParallaxDegrees =
                                settings.ultraPrecisionMinimumParallaxDegrees,
                            minimumCalibrationConfidence =
                                settings.ultraPrecisionMinimumCalibrationConfidence,
                            minimumSynchronizationConfidence =
                                settings.ultraPrecisionMinimumSynchronizationConfidence,
                            minimumViewDiversity =
                                settings.ultraPrecisionMinimumViewDiversity,
                            minimumCrossViewAgreement = it,
                        ),
                    )
                },
            )

            Text(
                text = "Precision contract",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "These are acceptance targets, not cosmetic quality sliders.",
                style = MaterialTheme.typography.bodySmall,
            )
            CapturePrioritySlider(
                label = "Target motion P95 error",
                value = ((settings.targetMotionP95ErrorMillimeters - 0.1) / 49.9)
                    .coerceIn(0.0, 1.0),
                onValueChanged = {
                    val target = 0.1 + it * 49.9
                    onSettingsChanged(
                        settings.withPrecisionContract(
                            target,
                            settings.maximumTemporalLagMilliseconds,
                            settings.maximumDropoutRatio,
                            settings.minimumCalibrationConfidence,
                            settings.minimumViewDiversity,
                            settings.minimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Maximum temporal lag",
                value = ((settings.maximumTemporalLagMilliseconds - 1.0) / 249.0)
                    .coerceIn(0.0, 1.0),
                onValueChanged = {
                    val target = 1.0 + it * 249.0
                    onSettingsChanged(
                        settings.withPrecisionContract(
                            settings.targetMotionP95ErrorMillimeters,
                            target,
                            settings.maximumDropoutRatio,
                            settings.minimumCalibrationConfidence,
                            settings.minimumViewDiversity,
                            settings.minimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Maximum synchronization residual P95",
                value = (
                    (settings.maximumSynchronizationResidualP95Ms - 1.0) / 49.0
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withSynchronizationContract(
                            residualP95Milliseconds = 1.0 + it * 49.0,
                            outlierRatio =
                                settings.maximumSynchronizationOutlierRatio,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Maximum synchronization outlier ratio",
                value = (
                    settings.maximumSynchronizationOutlierRatio / 0.50
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withSynchronizationContract(
                            residualP95Milliseconds =
                                settings.maximumSynchronizationResidualP95Ms,
                            outlierRatio = it * 0.50,
                        ),
                    )
                },
            )
            Text(
                text = "Temporal + interaction contract",
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "These limits control rolling-shutter timing acceptance and hand/object contact evidence.",
                style = MaterialTheme.typography.bodySmall,
            )
            CapturePrioritySlider(
                label = "Maximum rolling-shutter readout",
                value = (
                    (settings.maximumRollingShutterReadoutMilliseconds - 1.0) / 79.0
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalInteractionContract(
                            maximumRollingShutterReadoutMilliseconds =
                                1.0 + it * 79.0,
                            minimumHandObjectContactConfidence =
                                settings.minimumHandObjectContactConfidence,
                            maximumContactPenetrationMillimeters =
                                settings.maximumContactPenetrationMillimeters,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum hand/object contact confidence",
                value = settings.minimumHandObjectContactConfidence,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalInteractionContract(
                            maximumRollingShutterReadoutMilliseconds =
                                settings.maximumRollingShutterReadoutMilliseconds,
                            minimumHandObjectContactConfidence = it,
                            maximumContactPenetrationMillimeters =
                                settings.maximumContactPenetrationMillimeters,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Maximum contact penetration",
                value = (
                    (settings.maximumContactPenetrationMillimeters - 0.5) / 49.5
                ).coerceIn(0.0, 1.0),
                onValueChanged = {
                    onSettingsChanged(
                        settings.withTemporalInteractionContract(
                            maximumRollingShutterReadoutMilliseconds =
                                settings.maximumRollingShutterReadoutMilliseconds,
                            minimumHandObjectContactConfidence =
                                settings.minimumHandObjectContactConfidence,
                            maximumContactPenetrationMillimeters =
                                0.5 + it * 49.5,
                        ),
                    )
                },
            )

            CapturePrioritySlider(
                label = "Maximum dropout",
                value = settings.maximumDropoutRatio.coerceIn(0.0, 0.50) / 0.50,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withPrecisionContract(
                            settings.targetMotionP95ErrorMillimeters,
                            settings.maximumTemporalLagMilliseconds,
                            it * 0.50,
                            settings.minimumCalibrationConfidence,
                            settings.minimumViewDiversity,
                            settings.minimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum calibration confidence",
                value = settings.minimumCalibrationConfidence,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withPrecisionContract(
                            settings.targetMotionP95ErrorMillimeters,
                            settings.maximumTemporalLagMilliseconds,
                            settings.maximumDropoutRatio,
                            it,
                            settings.minimumViewDiversity,
                            settings.minimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum view diversity",
                value = settings.minimumViewDiversity,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withPrecisionContract(
                            settings.targetMotionP95ErrorMillimeters,
                            settings.maximumTemporalLagMilliseconds,
                            settings.maximumDropoutRatio,
                            settings.minimumCalibrationConfidence,
                            it,
                            settings.minimumCrossViewAgreement,
                        ),
                    )
                },
            )
            CapturePrioritySlider(
                label = "Minimum cross-view agreement",
                value = settings.minimumCrossViewAgreement,
                onValueChanged = {
                    onSettingsChanged(
                        settings.withPrecisionContract(
                            settings.targetMotionP95ErrorMillimeters,
                            settings.maximumTemporalLagMilliseconds,
                            settings.maximumDropoutRatio,
                            settings.minimumCalibrationConfidence,
                            settings.minimumViewDiversity,
                            it,
                        ),
                    )
                },
            )

            Text(
                text = "Profile: ${settings.profile().label}",
                style = MaterialTheme.typography.labelMedium,
            )
        }
    }
}

@Composable
private fun CapturePrioritySlider(
    label: String,
    value: Double,
    onValueChanged: (Double) -> Unit,
) {
    Column {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
            )
            Text(
                text = "${(value.coerceIn(0.0, 1.0) * 100.0).toInt()}%",
                style = MaterialTheme.typography.labelMedium,
            )
        }
        Slider(
            value = value.coerceIn(0.0, 1.0).toFloat(),
            onValueChange = { onValueChanged(it.toDouble()) },
        )
    }
}

@Composable
private fun CaptureSampleBudgetSlider(
    value: Int,
    onValueChanged: (Int) -> Unit,
) {
    val minimum = 60
    val maximum = 10_000
    val normalized =
        (
            (value.coerceIn(minimum, maximum) - minimum).toDouble() /
                (maximum - minimum).toDouble()
            ).toFloat()
    Column {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Precision benchmark evidence samples",
                style = MaterialTheme.typography.labelLarge,
            )
            Text(
                text = value.coerceIn(minimum, maximum).toString(),
                style = MaterialTheme.typography.labelMedium,
            )
        }
        Slider(
            value = normalized,
            onValueChange = {
                onValueChanged(
                    minimum +
                        kotlin.math.round(
                            it.coerceIn(0.0f, 1.0f) *
                                (maximum - minimum).toFloat(),
                        ).toInt(),
                )
            },
        )
        Text(
            text = "A precision claim still requires matched reference conditions and measured uncertainty.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun PolicySwitch(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
        )
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
        )
    }
}

@Composable
private fun CaptureIntentSummary(
    settings: HighEndMotionCaptureSettings,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(
                rememberScrollState(),
            ),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        CaptureSummaryChip(
            label = "Capture",
            value = "${settings.targetFps} FPS",
        )
        CaptureSummaryChip(
            label = "Inference",
            value = "${settings.inferenceFps} FPS",
        )
        CaptureSummaryChip(
            label = "3D budget",
            value = String.format(
                "%.1f mm",
                settings.reconstructionUncertaintyMeters * 1000.0,
            ),
        )
        CaptureSummaryChip(
            label = "Parallax",
            value = String.format(
                "%.1f° min",
                settings.minimumParallaxDegrees,
            ),
        )
        CaptureSummaryChip(
            label = "Sync P95",
            value = String.format(
                "%.1f ms",
                settings.maximumSynchronizationResidualP95Ms,
            ),
        )
        CaptureSummaryChip(
            label = "Rig policy",
            value =
                if (settings.referenceGradePrecisionEnabled) {
                    "REFERENCE"
                } else {
                    "ADAPTIVE"
                },
        )
    }
}

@Composable
private fun CaptureSummaryChip(
    label: String,
    value: String,
) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        tonalElevation = 1.dp,
        modifier = Modifier.heightIn(min = 48.dp),
    ) {
        Column(
            modifier = Modifier.padding(
                horizontal = 12.dp,
                vertical = 7.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(1.dp),
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
            )
            Text(
                text = value,
                style = MaterialTheme.typography.labelLarge,
            )
        }
    }
}

