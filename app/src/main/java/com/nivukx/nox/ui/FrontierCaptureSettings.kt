package com.nivukx.nox.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.MotionCaptureCustomization

@Composable
fun FrontierCaptureSettings(
    motionCustomization: MotionCaptureCustomization,
    onAdvancedMotionCustomization:
        ((MotionCaptureCustomization) -> MotionCaptureCustomization) -> Unit,
    modifier: Modifier = Modifier,
) {
    val activePreset = FrontierCapturePreset.entries.firstOrNull {
        motionCustomization.matchesFrontierPreset(it)
    }

    GlassCard(
        modifier = modifier.fillMaxWidth(),
        intensity = 0.94f,
        selected = true,
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text("FRONTIER CONTROL", style = MaterialTheme.typography.titleLarge)
            Text(
                "High-end motion, geometry, reconstruction and qualification controls.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "ACTIVE PROFILE  ·  ${activePreset?.name?.replace("_", " ") ?: "CUSTOM"}",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                text = "Configured envelope: ${motionCustomization.maxSubjects} actors · ${motionCustomization.distributedGlobalCameraEnvelope} camera nodes · ${motionCustomization.targetFps} fps target / ${motionCustomization.inferenceFps} fps inference",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                text = "Detail budgets: ${motionCustomization.fullDetailActorBudget} full-detail · ${motionCustomization.faceDetailActorBudget} face · ${motionCustomization.eyeDetailActorBudget} eye · ${motionCustomization.handDetailActorBudget} hand · ${motionCustomization.denseSurfaceActorBudget} surface actors",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "Configured budgets are not measured throughput or accuracy. Camera calibration, synchronization, overlapping views, lighting and device performance determine real results. Treat sub-millimetre P95/P99 values as qualification targets until verified against a matched reference.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text("Operating profiles", style = MaterialTheme.typography.labelLarge)
            LazyRow(
                modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(FrontierCapturePreset.entries, key = { it.name }) { preset ->
                    FilterChip(
                        selected = motionCustomization.matchesFrontierPreset(preset),
                        onClick = {
                            onAdvancedMotionCustomization {
                                it.applyFrontierPreset(preset)
                            }
                        },
                        label = { Text(preset.name.replace('_', ' ')) },
                    )
                }
            }

            Text("CAPACITY & CAMERA FEDERATION", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSlider("Target actors", motionCustomization.maxSubjects.toFloat(), 1f..64f, " actors", integerValue = true) { next ->
                onAdvancedMotionCustomization { it.withTargetActorLimit(next.toInt()) }
            }
            FrontierSettingSlider("Full-detail actor budget", motionCustomization.fullDetailActorBudget.toFloat(), 1f..64f, " actors", integerValue = true) { next ->
                onAdvancedMotionCustomization { it.copyWith { fullDetailActorBudget = next.toInt().coerceIn(1,64) } }
            }
            FrontierSettingSlider(
                "Authoritative camera budget",
                motionCustomization.distributedAuthoritativeCameraBudget.toFloat(),
                2f..motionCustomization.distributedGlobalCameraEnvelope.coerceAtMost(32).toFloat(),
                " cameras",
                integerValue = true,
            ) { next ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        distributedAuthoritativeCameraBudget =
                            next.toInt().coerceIn(2, distributedGlobalCameraEnvelope.coerceAtMost(32))
                    }
                }
            }
            FrontierSettingSlider(
                "Global camera envelope",
                motionCustomization.distributedGlobalCameraEnvelope.toFloat(),
                2f..128f,
                " cameras",
                integerValue = true,
            ) { next ->
                onAdvancedMotionCustomization {
                    it.withGlobalCameraEnvelope(next.toInt())
                }
            }
            FrontierSettingSlider(
                "Federation shard budget",
                motionCustomization.distributedMaximumShards.toFloat(),
                1f..motionCustomization.distributedGlobalCameraEnvelope.coerceAtMost(16).toFloat(),
                " shards",
                integerValue = true,
            ) { next ->
                onAdvancedMotionCustomization {
                    it.copyWith {
                        distributedMaximumShards =
                            next.toInt().coerceIn(1, distributedGlobalCameraEnvelope.coerceAtMost(16))
                    }
                }
            }
            FrontierSettingSlider("Minimum authority health", motionCustomization.distributedMinimumAuthorityHealth.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { distributedMinimumAuthorityHealth = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Minimum authority synchronization", motionCustomization.distributedMinimumAuthoritySynchronization.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { distributedMinimumAuthoritySynchronization = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Maximum federation time residual", motionCustomization.distributedMaximumTemporalResidualMs.toFloat(), 1f..50f, " ms") { next ->
                onAdvancedMotionCustomization { it.copyWith { distributedMaximumTemporalResidualMs = next.toDouble().coerceIn(1.0,50.0) } }
            }
            FrontierSettingSlider("Maximum network temporal outliers", motionCustomization.distributedMaximumTemporalOutlierRatio.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { distributedMaximumTemporalOutlierRatio = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Reserved transport headroom", motionCustomization.distributedMinimumTransportHeadroom.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { distributedMinimumTransportHeadroom = next.toDouble().coerceIn(0.0,1.0) } }
            }
            Text("MULTI-VIEW RECONSTRUCTION", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSlider("Minimum triangulation parallax", motionCustomization.minimumParallaxDegrees.toFloat(), 0.1f..10f, "°") { next ->
                onAdvancedMotionCustomization { it.copyWith { minimumParallaxDegrees = next.toDouble().coerceIn(0.1,10.0) } }
            }
            FrontierSettingSlider("Maximum reprojection error", motionCustomization.maximumReprojectionErrorPx.toFloat(), 1f..30f, " px") { next ->
                onAdvancedMotionCustomization { it.copyWith { maximumReprojectionErrorPx = next.toDouble().coerceIn(1.0,30.0) } }
            }
            FrontierSettingSlider("Maximum metric uncertainty", motionCustomization.maximumReconstructionUncertaintyMeters.toFloat(), 0.01f..2f, " m") { next ->
                onAdvancedMotionCustomization { it.copyWith { maximumReconstructionUncertaintyMeters = next.toDouble().coerceIn(0.01,2.0) } }
            }
            FrontierSettingSlider("Multi-view fusion weight", motionCustomization.multiViewFusionWeight.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { multiViewFusionWeight = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Topology constraint strength", motionCustomization.topologyConstraintStrength.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { topologyConstraintStrength = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Global 4D consistency", motionCustomization.global4DConsistencyStrictness.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { global4DConsistencyStrictness = next.toDouble().coerceIn(0.0,1.0) } }
            }

            Text("TEMPORAL MOTION & OCCLUSION", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSlider("Temporal smoothing", motionCustomization.temporalSmoothing.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { temporalSmoothing = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Occlusion persistence", motionCustomization.occlusionPersistence.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { occlusionPersistence = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Micro-motion preservation", motionCustomization.microMotionPreservation.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { microMotionPreservation = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Rolling-shutter compensation", motionCustomization.rollingShutterCompensation.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { rollingShutterCompensation = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Detector refresh cadence", motionCustomization.adaptiveDetectorIntervalFrames.toFloat(), 1f..30f, " frames", integerValue = true) { next ->
                onAdvancedMotionCustomization { it.copyWith { adaptiveDetectorIntervalFrames = next.toInt().coerceIn(1,30) } }
            }
            FrontierSettingSlider("Person segmentation cadence", motionCustomization.personSegmentationCadenceFrames.toFloat(), 1f..30f, " frames", integerValue = true) { next ->
                onAdvancedMotionCustomization { it.copyWith { personSegmentationCadenceFrames = next.toInt().coerceIn(1,30) } }
            }
            FrontierSettingSlider("Occlusion prediction window", motionCustomization.maximumOcclusionPredictionFrames.toFloat(), 0f..60f, " frames", integerValue = true) { next ->
                onAdvancedMotionCustomization { it.copyWith { maximumOcclusionPredictionFrames = next.toInt().coerceIn(0,60) } }
            }

            Text("IDENTITY & CONTINUITY", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSlider("Camera association", motionCustomization.actorCrossCameraAssociation.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { actorCrossCameraAssociation = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Actor identity strictness", motionCustomization.actorIdentityStrictness.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { actorIdentityStrictness = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Occlusion recovery", motionCustomization.actorOcclusionRecovery.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { actorOcclusionRecovery = next.toDouble().coerceIn(0.0,1.0) } }
            }
            Text("BODY, FACE & SURFACE DETAIL", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSlider("Dense mesh detail", motionCustomization.denseMeshDetail.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { denseMeshDetail = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Surface reconstruction priority", motionCustomization.surfaceEvidenceStrictness.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { surfaceEvidenceStrictness = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Face detail", motionCustomization.faceDetail.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { faceDetail = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Eye detail", motionCustomization.eyeDetail.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { eyeDetail = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Gaze priority", motionCustomization.gazePriority.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { gazePriority = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("World consistency", motionCustomization.worldConsistencyDetail.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { worldConsistencyDetail = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Temporal supersampling", motionCustomization.temporalSupersamplingPriority.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { temporalSupersamplingPriority = next.toDouble().coerceIn(0.0,1.0) } }
            }
            Text("PRECISION QUALIFICATION · REFERENCE-BASED", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSlider("Ultra precision P95 target", motionCustomization.ultraPrecisionTargetP95ErrorMillimeters.toFloat(), 0.05f..1.0f, " mm") { next ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionTargetP95ErrorMillimeters = next.toDouble().coerceIn(0.05,1.0) } }
            }
            FrontierSettingSlider("Ultra precision P99 target", motionCustomization.ultraPrecisionTargetP99ErrorMillimeters.toFloat(), 0.10f..2.0f, " mm") { next ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionTargetP99ErrorMillimeters = maxOf(next.toDouble(), ultraPrecisionTargetP95ErrorMillimeters) } }
            }
            FrontierSettingSlider("Minimum cross-view agreement", motionCustomization.ultraPrecisionMinimumCrossViewAgreement.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionMinimumCrossViewAgreement = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Minimum view diversity", motionCustomization.ultraPrecisionMinimumViewDiversity.toFloat(), 0f..1f, "") { next ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionMinimumViewDiversity = next.toDouble().coerceIn(0.0,1.0) } }
            }
            FrontierSettingSlider("Minimum baseline", motionCustomization.ultraPrecisionMinimumBaselineMeters.toFloat(), 0.05f..2.5f, " m") { next ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionMinimumBaselineMeters = next.toDouble().coerceAtLeast(0.05) } }
            }

            Text("EVIDENCE & DEVICE POLICY", style = MaterialTheme.typography.labelLarge)
            FrontierSettingSwitch("Ultra precision enabled", motionCustomization.ultraPrecisionEnabled) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionEnabled = enabled } }
            }
            FrontierSettingSwitch("Reference qualification required", motionCustomization.ultraPrecisionMatchedReferenceRequired) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionMatchedReferenceRequired = enabled } }
            }
            FrontierSettingSwitch("Ultra precision qualification mode", motionCustomization.ultraPrecisionQualificationMode) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { ultraPrecisionQualificationMode = enabled } }
            }
            FrontierSettingSwitch("Correlated evidence protection", motionCustomization.correlatedEvidenceEnabled) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { correlatedEvidenceEnabled = enabled } }
            }
            FrontierSettingSwitch("Correlated evidence qualification", motionCustomization.correlatedEvidenceQualificationMode) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { correlatedEvidenceQualificationMode = enabled } }
            }
            FrontierSettingSwitch("NPU-preferred dense perception", motionCustomization.preferNpuForDensePerception) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { preferNpuForDensePerception = enabled } }
            }
            FrontierSettingSwitch("RAW sensor sidecar", motionCustomization.cameraRawSidecarEnabled) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { cameraRawSidecarEnabled = enabled } }
            }
            FrontierSettingSwitch("Depth sidecar", motionCustomization.cameraDepthSidecarEnabled) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { cameraDepthSidecarEnabled = enabled } }
            }
            FrontierSettingSwitch("Continuous dynamic topology", motionCustomization.dynamicTopology) { enabled ->
                onAdvancedMotionCustomization { it.copyWith { dynamicTopology = enabled } }
            }
        }
    }
}

@Composable
private fun FrontierSettingSlider(
    title: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    suffix: String,
    integerValue: Boolean = false,
    onValue: (Float) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(title, style = MaterialTheme.typography.labelLarge)
            val boundedValue = value.coerceIn(range.start, range.endInclusive)
            val displayValue =
                if (integerValue) {
                    "${boundedValue.toInt()}${suffix}"
                } else {
                    String.format("%.2f%s", boundedValue, suffix)
                }
            Text(
                displayValue,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = onValue,
            valueRange = range,
        )
    }
}

@Composable
private fun FrontierSettingSwitch(
    title: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(title, style = MaterialTheme.typography.labelLarge)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
