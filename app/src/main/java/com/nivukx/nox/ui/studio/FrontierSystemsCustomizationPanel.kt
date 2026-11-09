package com.nivukx.nox.ui.studio

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import kotlin.math.roundToInt

/**
 * Frontier-specific operator controls.
 *
 * This surface exposes the genuinely new controls without duplicating the broad capture-settings
 * panel. The controls describe policy and evidence thresholds; engines remain responsible for
 * enforcing capability, uncertainty and resource constraints.
 */
@Composable
fun FrontierSystemsCustomizationPanel(
    settings: HighEndMotionCaptureSettings,
    onSettingsChanged: (HighEndMotionCaptureSettings) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        modifier = modifier.fillMaxWidth(),
        contentPadding = 16.dp,
        materialRole =
            StudioMaterialAdaptationEngine.MaterialRole
                .ACTION_CLUSTER,
        underlyingLuminance = 0.32f,
        contentContrast = 0.98f,
        capturePriority = 0.96f,
        operatorFocus = 0.92f,
        plateProtection = 0.92f,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Frontier Capture Fabric",
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text =
                    "128-camera federation, 50-actor scheduling, robust world-space fusion, "
                        + "probabilistic surface state and empirical precision qualification.",
                style = MaterialTheme.typography.bodySmall,
            )

            FrontierSectionTitle("Camera federation")
            FrontierSlider(
                label =
                    "Camera envelope: " +
                        settings.frontierMaximumCameraFederation,
                value =
                    (
                        settings.frontierMaximumCameraFederation - 2
                    ).toDouble() / 126.0,
                onChanged = {
                    val envelope =
                        (
                            2.0 +
                                it.coerceIn(0.0, 1.0) * 126.0
                        )
                            .roundToInt()
                            .coerceIn(2, 128)
                    val authority =
                        settings.frontierAuthorityCameraBudget
                            .coerceIn(
                                2,
                                minOf(32, envelope),
                            )
                    val witness =
                        settings.frontierWitnessCameraBudget
                            .coerceIn(
                                0,
                                envelope - authority,
                            )

                    onSettingsChanged(
                        settings.withFrontierCameraFederationPolicy(
                            authorityBudget = authority,
                            witnessBudget = witness,
                            globalEnvelope = envelope,
                        ),
                    )
                },
            )
            FrontierSlider(
                label =
                    "Authority cameras: " +
                        settings.frontierAuthorityCameraBudget,
                value =
                    (
                        settings.frontierAuthorityCameraBudget - 2
                    ).toDouble() /
                        (
                            minOf(
                                32,
                                settings.frontierMaximumCameraFederation,
                            ) - 2
                        ).coerceAtLeast(1),
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierCameraFederationPolicy(
                            authorityBudget =
                                (
                                    2.0 +
                                        it.coerceIn(0.0, 1.0) *
                                            (
                                                minOf(
                                                    32,
                                                    settings.frontierMaximumCameraFederation,
                                                ) - 2
                                            )
                                    )
                                    .roundToInt(),
                            witnessBudget =
                                settings.frontierWitnessCameraBudget,
                            globalEnvelope =
                                settings.frontierMaximumCameraFederation,
                        ),
                    )
                },
            )
            FrontierSlider(
                label =
                    "Witness cameras: " +
                        settings.frontierWitnessCameraBudget,
                value =
                    settings.frontierWitnessCameraBudget.toDouble() /
                        96.0,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierCameraFederationPolicy(
                            authorityBudget =
                                settings.frontierAuthorityCameraBudget,
                            witnessBudget =
                                (
                                    it.coerceIn(0.0, 1.0) * 96.0
                                )
                                    .roundToInt(),
                            globalEnvelope =
                                settings.frontierMaximumCameraFederation,
                        ),
                    )
                },
            )

            FrontierSectionTitle("World-space robustness")
            FrontierSlider(
                label =
                    "Robust iterations: " +
                        settings.frontierWorldSpaceRobustIterations,
                value =
                    (
                        settings.frontierWorldSpaceRobustIterations - 1
                    ).toDouble() / 7.0,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierWorldSpaceRobustness(
                            iterations =
                                (
                                    1.0 +
                                        it.coerceIn(0.0, 1.0) * 7.0
                                    )
                                    .roundToInt(),
                            huberResidualMillimeters =
                                settings.frontierWorldSpaceHuberResidualMillimeters,
                            maximumConditionEstimate =
                                settings.frontierWorldSpaceMaximumConditionEstimate,
                        ),
                    )
                },
            )
            FrontierSlider(
                label =
                    "Huber residual: " +
                        "%.1f".format(
                            settings.frontierWorldSpaceHuberResidualMillimeters,
                        ) +
                        " mm",
                value =
                    (
                        settings.frontierWorldSpaceHuberResidualMillimeters - 0.5
                    ) / 99.5,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierWorldSpaceRobustness(
                            iterations =
                                settings.frontierWorldSpaceRobustIterations,
                            huberResidualMillimeters =
                                0.5 + it.coerceIn(0.0, 1.0) * 99.5,
                            maximumConditionEstimate =
                                settings.frontierWorldSpaceMaximumConditionEstimate,
                        ),
                    )
                },
            )

            FrontierSectionTitle("Surface state")
            FrontierSlider(
                label =
                    "Patch capacity: " +
                        settings.frontierSurfacePatchCapacity / 1000 +
                        "k",
                value =
                    (
                        settings.frontierSurfacePatchCapacity - 1_024
                    ).toDouble() /
                        (
                            750_000 - 1_024
                        ),
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierSurfaceCapacity(
                            enabled =
                                settings.frontierSurfaceFusionEnabled,
                            patchCapacity =
                                (
                                    1_024.0 +
                                        it.coerceIn(0.0, 1.0) *
                                            (
                                                750_000 - 1_024
                                            )
                                    )
                                    .roundToInt(),
                            historyFrames =
                                settings.frontierSurfaceHistoryFrames,
                        ),
                    )
                },
            )
            FrontierSlider(
                label =
                    "Temporal history: " +
                        settings.frontierSurfaceHistoryFrames +
                        " frames",
                value =
                    (
                        settings.frontierSurfaceHistoryFrames - 2
                    ).toDouble() / 118.0,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierSurfaceCapacity(
                            enabled =
                                settings.frontierSurfaceFusionEnabled,
                            patchCapacity =
                                settings.frontierSurfacePatchCapacity,
                            historyFrames =
                                (
                                    2.0 +
                                        it.coerceIn(0.0, 1.0) * 118.0
                                    )
                                    .roundToInt(),
                        ),
                    )
                },
            )

            FrontierSectionTitle("Calibration authority")
            FrontierSlider(
                label =
                    "Cycle residual target: " +
                        "%.1f".format(
                            settings.frontierCalibrationMaximumCycleResidualMillimeters,
                        ) +
                        " mm",
                value =
                    (
                        settings.frontierCalibrationMaximumCycleResidualMillimeters - 1.0
                    ) / 99.0,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierCalibrationPolicy(
                            graphEnabled =
                                settings.frontierCalibrationGraphEnabled,
                            maximumCycleResidualMillimeters =
                                1.0 + it.coerceIn(0.0, 1.0) * 99.0,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "Calibration graph authority",
                checked = settings.frontierCalibrationGraphEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.withFrontierCalibrationPolicy(
                            graphEnabled = it,
                            maximumCycleResidualMillimeters =
                                settings.frontierCalibrationMaximumCycleResidualMillimeters,
                        ),
                    )
                },
            )
            PolicySwitch(
                label = "Require independent precision reference",
                checked = settings.frontierPrecisionQualificationRequiresReference,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            frontierPrecisionQualificationRequiresReference =
                                it,
                        ),
                    )
                },
            )

            FrontierSectionTitle("Adaptive multi-actor authority")

            PolicySwitch(
                label = "Adaptive authority depth",
                checked = settings.adaptiveAuthorityDepthEnabled,
                onCheckedChange = {
                    onSettingsChanged(
                        settings.copy(
                            adaptiveAuthorityDepthEnabled = it,
                        ),
                    )
                },
            )

            FrontierSlider(
                label =
                    "Authority depth bonus: " +
                        settings.maximumAdaptiveAuthorityBonus,
                value =
                    settings.maximumAdaptiveAuthorityBonus /
                        8.0,
                onChanged = {
                    onSettingsChanged(
                        settings.copy(
                            maximumAdaptiveAuthorityBonus =
                                (
                                    it * 8.0
                                ).roundToInt().coerceIn(0, 8),
                        ),
                    )
                },
            )

            FrontierSlider(
                label =
                    "Crowded-scene overlap threshold: " +
                        "%.0f".format(
                            settings.crowdedSceneOverlapThreshold * 100.0,
                        ) +
                        "%",
                value = settings.crowdedSceneOverlapThreshold,
                onChanged = {
                    onSettingsChanged(
                        settings.copy(
                            crowdedSceneOverlapThreshold =
                                it.coerceIn(0.0, 1.0),
                        ),
                    )
                },
            )

            FrontierSlider(
                label =
                    "Crowded-scene identity boost: " +
                        "%.0f".format(
                            settings.crowdedSceneMatchScoreBoost * 100.0,
                        ) +
                        "%",
                value =
                    settings.crowdedSceneMatchScoreBoost /
                        0.30,
                onChanged = {
                    onSettingsChanged(
                        settings.copy(
                            crowdedSceneMatchScoreBoost =
                                (
                                    it * 0.30
                                ).coerceIn(0.0, 0.30),
                        ),
                    )
                },
            )

            FrontierSlider(
                label =
                    "Maximum authority views / actor: " +
                        settings.maximumAdaptiveAuthorityCamerasPerActor,
                value =
                    (
                        settings.maximumAdaptiveAuthorityCamerasPerActor - 1
                    ).toDouble() / 31.0,
                onChanged = {
                    onSettingsChanged(
                        settings.copy(
                            maximumAdaptiveAuthorityCamerasPerActor =
                                (
                                    1.0 +
                                        it.coerceIn(0.0, 1.0) * 31.0
                                    ).roundToInt()
                                        .coerceIn(1, 32),
                        ),
                    )
                },
            )

            FrontierSectionTitle("Precision geometry")
            FrontierSlider(
                label =
                    "Minimum measured views: " +
                        settings.frontierMinimumObservedViews,
                value =
                    (
                        settings.frontierMinimumObservedViews - 2
                    ).toDouble() / 14.0,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierPrecisionQualification(
                            requiresIndependentReference =
                                settings.frontierPrecisionQualificationRequiresReference,
                            minimumViews =
                                (
                                    2.0 +
                                        it.coerceIn(0.0, 1.0) * 14.0
                                    )
                                    .roundToInt(),
                            minimumParallaxDegrees =
                                settings.frontierMinimumObservedParallaxDegrees,
                        ),
                    )
                },
            )
            FrontierSlider(
                label =
                    "Minimum parallax: " +
                        "%.1f".format(
                            settings.frontierMinimumObservedParallaxDegrees,
                        ) +
                        "°",
                value =
                    (
                        settings.frontierMinimumObservedParallaxDegrees - 1.0
                    ) / 88.0,
                onChanged = {
                    onSettingsChanged(
                        settings.withFrontierPrecisionQualification(
                            requiresIndependentReference =
                                settings.frontierPrecisionQualificationRequiresReference,
                            minimumViews =
                                settings.frontierMinimumObservedViews,
                            minimumParallaxDegrees =
                                1.0 + it.coerceIn(0.0, 1.0) * 88.0,
                        ),
                    )
                },
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FrontierBadge(
                    label = "50 ACTORS",
                    selected = true,
                )
                FrontierBadge(
                    label = "128 CAMERAS",
                    selected =
                        settings.frontierMaximumCameraFederation >= 128,
                )
                FrontierBadge(
                    label = "EMPIRICAL GATE",
                    selected =
                        settings.frontierPrecisionQualificationRequiresReference,
                )
            }
        }
    }
}

@Composable
private fun FrontierSectionTitle(
    title: String,
) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleSmall,
    )
}

@Composable
private fun FrontierSlider(
    label: String,
    value: Double,
    onChanged: (Double) -> Unit,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
        )
        androidx.compose.material3.Slider(
            value = value.coerceIn(0.0, 1.0).toFloat(),
            onValueChange = {
                onChanged(it.toDouble())
            },
        )
    }
}

@Composable
private fun FrontierBadge(
    label: String,
    selected: Boolean,
) {
    Surface(
        tonalElevation =
            if (selected) 6.dp else 2.dp,
        shape =
            androidx.compose.foundation.shape.RoundedCornerShape(
                16.dp,
            ),
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(
                horizontal = 10.dp,
                vertical = 8.dp,
            ),
            style = MaterialTheme.typography.labelSmall,
        )
    }
}
