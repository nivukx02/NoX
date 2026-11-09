package com.nivukx.nox.ui.studio

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.HighEndMotionCaptureSettings
import com.nivukx.nox.core.model.ProductionCaptureProfile

@Composable
fun MotionStudioOperatorDeck(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
    onCaptureToggle: () -> Unit = {},
    onExport: () -> Unit = {},
    captureProfileId: String = ProductionCaptureProfile.default.id,
    onCaptureProfileChanged: (ProductionCaptureProfile) -> Unit = {},
    customizationSettings: HighEndMotionCaptureSettings =
        HighEndMotionCaptureSettings(
            profileId = captureProfileId,
        ),
    onCustomizationSettingsChanged:
        (HighEndMotionCaptureSettings) -> Unit = {},
    customizationPanelVisible: Boolean = false,
    onCustomizationPanelVisibilityChanged: (Boolean) -> Unit = {},
) {
    var selected by remember { mutableStateOf(StudioSection.RECONSTRUCTION) }
    var query by remember { mutableStateOf("") }
    val filteredSections = remember(query) {
        StudioSection.entries.filter {
            query.isBlank() || it.name
                .replace('_', ' ')
                .contains(query, ignoreCase = true)
        }
    }
    val health = remember(telemetry) {
        StudioHealthModel.summarize(telemetry)
    }
    val captureProfile = remember(captureProfileId) {
        ProductionCaptureProfile.byId(captureProfileId)
    }

    BoxWithConstraints(modifier.fillMaxWidth()) {
        val wide = maxWidth >= 640.dp
        Column(Modifier.fillMaxWidth()) {
            StudioCommandBar(
                telemetry = telemetry,
                query = query,
                onQueryChange = { query = it },
                onCaptureToggle = onCaptureToggle,
                onExport = onExport,
            )

            EvidencePulseStrip(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            FrontierOperatorDeck(
                state = telemetry.toFrontierOperatorState(),
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            DistributedRigStatusPanel(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            TimingAuthorityPanel(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            PrecisionFabricPanel(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            FrontierCaptureCapacityCommandCenter(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            HighEndPrecisionCockpit(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            FrontierPrecisionAuthorityPanel(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            ProductionQualityEvidencePanel(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            EvidenceAuthorityLadderPanel(
                telemetry = telemetry,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            CaptureProfileQuickControls(
                selectedProfileId = captureProfile.id,
                onProfileSelected = onCaptureProfileChanged,
                modifier = Modifier.padding(
                    horizontal = if (wide) 20.dp else 12.dp,
                    vertical = 4.dp,
                ),
            )

            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = if (wide) 20.dp else 12.dp,
                        vertical = 2.dp,
                    ),
                horizontalArrangement = Arrangement.End,
            ) {
                TextButton(
                    onClick = {
                        onCustomizationPanelVisibilityChanged(
                            !customizationPanelVisible,
                        )
                    },
                ) {
                    Text(
                        if (customizationPanelVisible) {
                            "Hide customization"
                        } else {
                            "Customize capture"
                        },
                    )
                }
            }

            if (customizationPanelVisible) {
                MotionCaptureCustomizationPanel(
                    settings = customizationSettings,
                    onSettingsChanged = onCustomizationSettingsChanged,
                    modifier = Modifier.padding(
                        horizontal = if (wide) 20.dp else 12.dp,
                        vertical = 2.dp,
                    ),
                )
            }

            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(
                        horizontal = if (wide) 20.dp else 12.dp,
                        vertical = 4.dp,
                    ),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                filteredSections.forEach { section ->
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        tonalElevation = if (selected == section) 4.dp else 1.dp,
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .selectable(
                                selected = selected == section,
                                onClick = { selected = section },
                                role = Role.Tab,
                            ),
                    ) {
                        Text(
                            text = section.name.replace('_', ' '),
                            style = MaterialTheme.typography.labelMedium,
                            modifier = Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 8.dp,
                            ),
                        )
                    }
                }
            }

            LiquidGlassSurface(
                clear = false,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = if (wide) 20.dp else 12.dp,
                        vertical = 6.dp,
                    ),
            ) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(if (wide) 18.dp else 14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    OperatorHealthHeader(health)
                    StatusLine("Profile", captureProfile.label)

                    if (wide) {
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(18.dp),
                        ) {
                            Column(
                                Modifier.weight(1f),
                                verticalArrangement = Arrangement.spacedBy(7.dp),
                            ) {
                                TelemetryColumn(telemetry)
                            }
                            Column(
                                Modifier.weight(1f),
                                verticalArrangement = Arrangement.spacedBy(7.dp),
                            ) {
                                QualityColumn(telemetry)
                                CapturePathEvidenceColumn(telemetry)
                                PrecisionEvidenceColumn(telemetry)
                                ProductionQualityColumn(telemetry)
                            }
                        }
                    } else {
                        TelemetryColumn(telemetry)
                        QualityColumn(telemetry)
                        CapturePathEvidenceColumn(telemetry)
                        ProductionQualityColumn(telemetry)
                    }

                    Text(
                        text = selected.name.replace('_', ' ') + " surface",
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
            }
        }
    }
}

@Composable
private fun TimingAuthorityPanel(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val supported =
        telemetry.hardwareReadoutTimestampSupported
    val confidence =
        telemetry.hardwareReadoutTimingConfidence
            .coerceIn(0.0, 1.0)
    val p95 =
        telemetry.hardwareReadoutP95JitterMillis
    val medianDelay =
        telemetry.hardwareReadoutMedianDelayMillis

    val precisionReady =
        supported &&
            confidence >= 0.90 &&
            p95.isFinite() &&
            p95 <= 2.0

    LiquidGlassSurface(
        clear = false,
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        "TIMING AUTHORITY",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Hardware readout evidence and temporal confidence",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    if (precisionReady) "REFERENCE-READY"
                    else if (supported) "HARDWARE AVAILABLE"
                    else "SOFTWARE-ALIGNED",
                    style = MaterialTheme.typography.labelMedium,
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                RigMetric(
                    label = "READOUT",
                    value = if (supported) "HW" else "SW",
                    modifier = Modifier.weight(1f),
                )
                RigMetric(
                    label = "CONFIDENCE",
                    value = "${(confidence * 100.0).toInt()}%",
                    modifier = Modifier.weight(1f),
                )
                RigMetric(
                    label = "P95 JITTER",
                    value =
                        if (p95.isFinite()) {
                            "%.2f ms".format(p95)
                        } else {
                            "N/A"
                        },
                    modifier = Modifier.weight(1f),
                )
                RigMetric(
                    label = "MEDIAN DELAY",
                    value =
                        if (medianDelay.isFinite()) {
                            "%.2f ms".format(medianDelay)
                        } else {
                            "N/A"
                        },
                    modifier = Modifier.weight(1f),
                )
            }

            Text(
                if (supported) {
                    "Hardware readout timestamps are measured from the camera capture path. " +
                        "They strengthen temporal authority but do not replace multi-camera " +
                        "clock fitting, rolling-shutter correction or calibrated geometry."
                } else {
                    "This camera does not expose hardware readout timing. NoX keeps software " +
                        "alignment active and visibly lowers timing authority instead of hiding the limitation."
                },
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun DistributedRigStatusPanel(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val connected = telemetry.camerasConnected ?: 0
    val expected = telemetry.camerasExpected ?: connected
    val sync = telemetry.synchronizationConfidence ?: 0.0
    val uncertainty =
        telemetry.reconstructionUncertaintyP95Meters
            ?.takeIf(Double::isFinite)
            ?: Double.POSITIVE_INFINITY

    val admission =
        when {
            connected <= 0 -> "NO RIG"
            sync < 0.50 -> "TIMING BLOCKED"
            telemetry.calibrationReady != true -> "CALIBRATION REQUIRED"
            uncertainty.isFinite() && uncertainty > 0.010 ->
                "PRECISION LIMITED"
            else -> "RIG READY"
        }

    LiquidGlassSurface(
        clear = false,
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        "DISTRIBUTED RIG",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "128-node global evidence envelope",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Text(
                    admission,
                    style = MaterialTheme.typography.labelMedium,
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                RigMetric(
                    label = "CAMERAS",
                    value = "${connected} / ${expected}",
                    modifier = Modifier.weight(1f),
                )
                RigMetric(
                    label = "SYNC",
                    value = "${(sync * 100.0).toInt()}%",
                    modifier = Modifier.weight(1f),
                )
                RigMetric(
                    label = "CAL",
                    value = if (telemetry.calibrationReady == true) "READY" else "HOLD",
                    modifier = Modifier.weight(1f),
                )
                RigMetric(
                    label = "3D P95",
                    value =
                        if (uncertainty.isFinite()) {
                            if (uncertainty < 1.0) {
                                "%.1f mm".format(uncertainty * 1000.0)
                            } else {
                                "%.3f m".format(uncertainty)
                            }
                        } else {
                            "N/A"
                        },
                    modifier = Modifier.weight(1f),
                )
            }

            Text(
                "The rig federation keeps global camera capacity separate from the "
                    + "simultaneous-camera limit of any individual device. Only evidence that "
                    + "passes timing, calibration and geometry authority is promoted into "
                    + "precision reconstruction.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun RigMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 1.dp,
    ) {
        Column(
            Modifier.padding(
                horizontal = 10.dp,
                vertical = 8.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
            )
            Text(
                value,
                style = MaterialTheme.typography.titleSmall,
            )
        }
    }
}

@Composable
private fun OperatorHealthHeader(summary: StudioHealthSummary) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(Modifier.weight(1f)) {
            Text(
                text = when (summary.band) {
                    StudioHealthBand.READY -> "SYSTEM READY"
                    StudioHealthBand.DEGRADED -> "SYSTEM DEGRADED"
                    StudioHealthBand.BLOCKED -> "SYSTEM BLOCKED"
                    StudioHealthBand.UNKNOWN -> "SYSTEM UNKNOWN"
                },
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = "${summary.knownSignalCount}/${summary.totalSignalCount} operator signals known",
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Text(
            text = if (summary.isOperationallyReady) "READY" else summary.band.name,
            style = MaterialTheme.typography.labelLarge,
        )
    }

    summary.blockers.take(2).forEach { blocker ->
        Text(
            text = blocker,
            style = MaterialTheme.typography.bodySmall,
        )
    }

    summary.warnings
        .filterNot { warning -> summary.blockers.contains(warning) }
        .take(3)
        .forEach { warning ->
            Text(
                text = warning,
                style = MaterialTheme.typography.bodySmall,
            )
        }
}


@Composable
private fun CapturePathEvidenceColumn(
    telemetry: StudioTelemetryState,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(7.dp),
    ) {
        StatusLine(
            "Camera Input",
            telemetry.cameraZeroCopyReady?.let {
                if (it) "ZERO-COPY READY" else "COPY PATH"
            },
        )
        StatusLine(
            "Host Copies",
            telemetry.cameraHostCopyCount?.toString(),
        )
        StatusLine(
            "Interleaved Rate",
            telemetry.interleavedEffectiveInputRateHz?.let {
                String.format("%.1f Hz", it)
            },
        )
        StatusLine(
            "Temporal Consistency",
            telemetry.interleavedTemporalConsistency?.percent(),
        )
        StatusLine(
            "Spatial Agreement",
            telemetry.interleavedSpatialAgreement?.percent(),
        )
        StatusLine(
            "Supporting Cameras",
            telemetry.interleavedSupportingCameras?.toString(),
        )
        StatusLine(
            "Interleaved Gate",
            telemetry.interleavedAuthoritative?.let {
                if (it) "AUTHORITATIVE" else "EVIDENCE ONLY"
            },
        )
    }
}

@Composable
private fun PrecisionEvidenceColumn(
    telemetry: StudioTelemetryState,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(7.dp),
    ) {
        StatusLine(
            "P95 Error",
            telemetry.motionP95ErrorMillimeters?.let {
                String.format("%.2f mm", it)
            },
        )
        StatusLine(
            "P99 Error",
            telemetry.motionP99ErrorMillimeters?.let {
                String.format("%.2f mm", it)
            },
        )
        StatusLine(
            "Jitter",
            telemetry.motionJitterMillimeters?.let {
                String.format("%.2f mm", it)
            },
        )
        StatusLine(
            "Lag",
            telemetry.temporalLagMilliseconds?.let {
                String.format("%.1f ms", it)
            },
        )
        StatusLine(
            "Cal. Coverage",
            telemetry.calibrationImageCoverage?.let {
                String.format("%.0f%%", it*100.0)
            },
        )
        StatusLine(
            "View Diversity",
            telemetry.selectedViewDiversity?.let {
                String.format("%.0f%%", it*100.0)
            },
        )
        StatusLine(
            "Cross-view",
            telemetry.crossViewAgreement?.let {
                String.format("%.0f%%", it*100.0)
            },
        )
        StatusLine(
            "Cal. Drift",
            telemetry.calibrationDriftStabilityScore?.let {
                String.format("%.0f%%", it * 100.0)
            },
        )
        StatusLine(
            "Precision Contract",
            productionContractLabel(telemetry),
        )
        StatusLine(
            "Reference Test",
            referenceBenchmarkLabel(telemetry),
        )
    }
}

@Composable
private fun ProductionQualityColumn(
    telemetry: StudioTelemetryState,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(7.dp),
    ) {
        StatusLine(
            "Quality Band",
            telemetry.productionQualityBand,
        )
        StatusLine(
            "Quality Score",
            telemetry.productionQualityScore?.let {
                String.format("%.0f%%", it.coerceIn(0.0, 1.0) * 100.0)
            },
        )
        StatusLine(
            "Contract",
            telemetry.productionQualityContractSatisfied?.let {
                if (it) "SATISFIED" else "NOT MET"
            },
        )
        StatusLine(
            "Rig Observability",
            telemetry.cameraObservabilityScore?.let {
                String.format("%.0f%%", it.coerceIn(0.0, 1.0) * 100.0)
            },
        )
        StatusLine(
            "Camera Session",
            telemetry.cameraSessionVerification?.replace('_', ' '),
        )
        StatusLine(
            "High-speed mode",
            when {
                telemetry.highSpeedSessionActive ->
                    telemetry.highSpeedSelectedFps
                        ?.let { "ACTIVE @ ${it} FPS" }
                        ?: "ACTIVE"
                telemetry.highSpeedSessionSupported == true ->
                    telemetry.highSpeedSelectedFps
                        ?.let { "CAPABLE @ ${it} FPS" }
                        ?: "CAPABLE"
                telemetry.highSpeedSessionSupported == false ->
                    "REJECTED"
                else ->
                    "UNKNOWN"
            },
        )
        StatusLine(
            "High-speed reason",
            telemetry.highSpeedSessionReason?.take(72),
        )
        StatusLine(
            "HW Readout",
            if (telemetry.hardwareReadoutTimestampSupported) {
                "AVAILABLE"
            } else {
                "UNAVAILABLE"
            },
        )
        StatusLine(
            "Face/Eye Metric",
            telemetry.faceEyeMetricState?.replace('_', ' '),
        )
        StatusLine(
            "Identity continuity",
            String.format(
                "%.0f%%",
                telemetry.multiPersonIdentityContinuity
                    .coerceIn(0.0, 1.0) * 100.0,
            ),
        )
        StatusLine(
            "Re-identifications",
            telemetry.multiPersonIdentityReidentifications.toString(),
        )
        StatusLine(
            "Predicted actor tracks",
            telemetry.multiPersonIdentityPredictions.toString(),
        )
        StatusLine(
            "Metric Confidence",
            telemetry.faceEyeMetricConfidence?.let {
                String.format("%.0f%%", it.coerceIn(0.0, 1.0) * 100.0)
            },
        )
        StatusLine(
            "Eye P95",
            telemetry.faceEyeMetricEyeUncertaintyMm?.let {
                if (it.isFinite()) "%.2f mm".format(it) else "N/A"
            },
        )
        StatusLine(
            "Actors",
            telemetry.multiPersonAdmittedCount.toString() +
                " / " +
                telemetry.multiPersonObservedCount.toString(),
        )
        StatusLine(
            "Held actors",
            telemetry.multiPersonHeldCount.toString(),
        )
        StatusLine(
            "Crowd fairness",
            String.format(
                "%.0f%%",
                telemetry.multiPersonStarvationRisk
                    .coerceIn(0.0, 1.0) * 100.0,
            ),
        )
        StatusLine(
            "Selected Cameras",
            telemetry.selectedCameraCount?.toString(),
        )
        StatusLine(
            "Reference Benchmark",
            telemetry.referenceBenchmarkStatus,
        )
        StatusLine(
            "Modeled P95",
            telemetry.modeledPrecisionP95Millimeters?.let {
                String.format("%.2f mm", it)
            },
        )
        StatusLine(
            "Modeled P99",
            telemetry.modeledPrecisionP99Millimeters?.let {
                String.format("%.2f mm", it)
            },
        )
        StatusLine(
            "Metrology Grade",
            telemetry.precisionGrade,
        )
        StatusLine(
            "2 mm Feasible",
            telemetry.referenceGradeFeasible?.let {
                if (it) "YES" else "NO"
            },
        )
        StatusLine(
            "Required Baseline",
            telemetry.requiredBaselineMetersFor2mm?.let {
                if (it.isFinite()) String.format("%.2f m", it) else "INSUFFICIENT"
            },
        )
        StatusLine(
            "Recommended Baseline",
            telemetry.recommendedBaselineMeters?.let {
                if (it.isFinite()) String.format("%.2f m", it) else "INSUFFICIENT"
            },
        )
        StatusLine(
            "Required Focal",
            telemetry.requiredFocalLengthPixelsFor2mm?.let {
                if (it.isFinite()) String.format("%.0f px", it) else "INSUFFICIENT"
            },
        )
        StatusLine(
            "Min. Parallax",
            telemetry.recommendedMinimumParallaxDegrees?.let {
                String.format("%.1f°", it)
            },
        )
        StatusLine(
            "Resource Leak Risk",
            telemetry.resourceLeakRisk?.let {
                String.format("%.0f%%", it.coerceIn(0.0, 1.0) * 100.0)
            },
        )
        StatusLine(
            "Stale Resources",
            telemetry.staleResourceCount?.toString(),
        )
        StatusLine(
            "Resident Memory",
            telemetry.residentResourceBytes?.let {
                formatBytes(it)
            },
        )
        StatusLine(
            "Active Resources",
            telemetry.activeResourceCount?.toString(),
        )
    }
}

@Composable
private fun TelemetryColumn(telemetry: StudioTelemetryState) {
    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
        StatusLine(
            "Cameras",
            telemetry.camerasConnected?.let { connected ->
                val expected = telemetry.camerasExpected?.toString() ?: "?"
                "$connected/$expected"
            },
        )
        StatusLine("Sync", telemetry.synchronizationConfidence?.percent())
        StatusLine(
            "Calibration",
            telemetry.calibrationReady?.let {
                if (it) "READY" else "REQUIRED"
            },
        )
        StatusLine("Tracking", telemetry.trackingConfidence?.percent())
        StatusLine("3D", telemetry.reconstructionConfidence?.percent())
    }
}

@Composable
private fun QualityColumn(telemetry: StudioTelemetryState) {
    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
        StatusLine(
            "Native",
            telemetry.nativeEngineHealthy?.let {
                if (it) "HEALTHY" else "ERROR"
            },
        )
        StatusLine(
            "Thermal",
            telemetry.thermalConstrained?.let {
                if (it) "CONSTRAINED" else "NORMAL"
            },
        )
        StatusLine("Drops", telemetry.droppedObservations?.toString())
        StatusLine(
            "Queue",
            telemetry.queuePressure?.let {
                (it * 100.0).toInt().toString() + "%"
            },
        )
        StatusLine("3D U-P95", telemetry.reconstructionUncertaintyP95Meters?.formatMeters())
        StatusLine(
            "Motion Memory",
            telemetry.motionTemporalMemoryScore?.percent(),
        )
        StatusLine(
            "Dense Trust",
            telemetry.denseSurfaceTrustedPointRatio?.percent(),
        )
        StatusLine(
            "Dense U-P95",
            telemetry.denseSurfaceUncertaintyP95Meters?.formatMeters(),
        )
        StatusLine(
            "Dense Topology",
            telemetry.denseSurfaceTopologyStability?.percent(),
        )
        StatusLine(
            "Dense Views",
            telemetry.denseSurfaceViewDiversity?.percent(),
        )
        StatusLine(
            "Weak Axis",
            telemetry.denseSurfaceAnisotropicWeakAxisConstraint?.percent(),
        )
        StatusLine(
            "2D Mass",
            telemetry.denseSurfaceProbabilisticEvidenceScore?.percent(),
        )
        StatusLine(
            "Axis Condition",
            telemetry.denseSurfaceAnisotropicViewConditionScore?.percent(),
        )
        StatusLine(
            "Dense Reproj",
            telemetry.denseSurfaceMeanReprojectionErrorPx?.let {
                String.format("%.2f px", it)
            },
        )
        StatusLine(
            "Dense Gate",
            telemetry.denseSurfaceHardFailure?.let {
                if (it) "BLOCKED" else "HEALTHY"
            },
        )
        StatusLine(
            "Queue Age",
            telemetry.nativeQueueAgeMicroseconds?.let {
                String.format("%.1f ms", it / 1000.0)
            },
        )
        StatusLine(
            "Worker",
            telemetry.nativeProcessingMicroseconds?.let {
                String.format("%.1f ms", it / 1000.0)
            },
        )
        StatusLine(
            "Capture Overflow",
            telemetry.captureOverflowCount?.toString(),
        )
        StatusLine(
            "TSDF Memory",
            telemetry.tsdfMemoryUtilization?.let {
                String.format("%.0f%%", it * 100.0)
            },
        )
        StatusLine(
            "Export",
            if (StudioUiStateReducer.exportEnabled(telemetry)) "READY" else "HOLD",
        )
    }
}

private fun productionContractLabel(
    telemetry: StudioTelemetryState,
): String {
    val p95 = telemetry.motionP95ErrorMillimeters
    val lag = telemetry.temporalLagMilliseconds
    val calibration = telemetry.calibrationReady
    val diversity = telemetry.selectedViewDiversity
    val agreement = telemetry.crossViewAgreement

    val known = listOfNotNull(
        p95,
        lag,
        calibration,
        diversity,
        agreement,
    )
    if (known.isEmpty()) return "EVIDENCE REQUIRED"

    val pass =
        (p95 == null || p95 <= 5.0) &&
            (lag == null || lag <= 25.0) &&
            (calibration != false) &&
            (diversity == null || diversity >= 0.40) &&
            (agreement == null || agreement >= 0.50)

    return if (pass) "MEETS TARGETS" else "BELOW TARGET"
}

private fun referenceBenchmarkLabel(
    telemetry: StudioTelemetryState,
): String =
    when {
        telemetry.motionP95ErrorMillimeters == null ->
            "NOT MEASURED"
        telemetry.motionP95ErrorMillimeters <= 5.0 ->
            "CONTROLLED TEST REQUIRED"
        else ->
            "RETEST REQUIRED"
    }

private fun formatBytes(bytes: Long): String =
    when {
        bytes < 1024L -> "$bytes B"
        bytes < 1024L * 1024L ->
            String.format("%.1f KB", bytes / 1024.0)
        bytes < 1024L * 1024L * 1024L ->
            String.format("%.1f MB", bytes / (1024.0 * 1024.0))
        else ->
            String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
    }

private fun Double.percent(): String =
    (coerceIn(0.0, 1.0) * 100.0).toInt().toString() + "%"

private fun Double.formatMeters(): String =
    if (this < 1.0) {
        String.format("%.1f cm", this * 100.0)
    } else {
        String.format("%.2f m", this)
    }

@Composable
private fun StatusLine(
    label: String,
    value: String?,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.labelLarge)
        Text(value ?: "N/A", style = MaterialTheme.typography.labelLarge)
    }

private fun StudioTelemetryState.toFrontierOperatorState(): FrontierOperatorState {
    val production = productionQualityBand?.uppercase()
    val authorityBand = captureAuthorityBand?.uppercase()
    val state = when {
        captureAuthorityHardBlocked == true &&
            authorityBand == "BLOCKED" ->
            FrontierOperationalState.EVIDENCE_LIMITED
        captureAuthorityPrecisionQualified == true ||
            production == "PRECISION_QUALIFIED" ||
            (productionQualityContractSatisfied == true &&
                (precisionGrade?.uppercase() == "QUALIFIED" || referenceGradeFeasible == true)) ->
            FrontierOperationalState.PRECISION_QUALIFIED
        thermalConstrained == true -> FrontierOperationalState.THERMAL_LIMITED
        opticalBlurRisk >= 0.60 -> FrontierOperationalState.OPTICAL_LIMITED
        calibrationReady == false -> FrontierOperationalState.CALIBRATION_LIMITED
        synchronizationConfidence != null && synchronizationConfidence < 0.70 -> FrontierOperationalState.TIMING_LIMITED
        nativeEngineHealthy == false -> FrontierOperationalState.DEGRADED
        cameraObservabilityScore != null && cameraObservabilityScore < 0.60 -> FrontierOperationalState.EVIDENCE_LIMITED
        production == "DEGRADED" -> FrontierOperationalState.DEGRADED
        production == "RECOVERING" -> FrontierOperationalState.RECOVERING
        else -> FrontierOperationalState.LIVE
    }
    val actorCount = (multiPersonAdmittedCount.takeIf { it > 0 } ?: multiPersonObservedCount).coerceIn(0, 64)
    val authorityCameraCount = (selectedCameraCount ?: 0).coerceIn(0, 32)
    val witnessEstimate = ((camerasConnected ?: authorityCameraCount) - authorityCameraCount).coerceAtLeast(0).coerceAtMost(96)
    val tracking = (trackingConfidence ?: multiPersonIdentityContinuity).coerceIn(0.0, 1.0)
    val reconstruction = (reconstructionConfidence ?: denseSurfaceProbabilisticEvidenceScore ?: 0.0).coerceIn(0.0, 1.0)
    val timing = (synchronizationConfidence ?: hardwareReadoutTimingConfidence).coerceIn(0.0, 1.0)
    val metricEvidenceAuthority =
        captureAuthorityScore
            ?.takeIf { it.isFinite() }
            ?.coerceIn(0.0, 1.0)
            ?: (
                (denseSurfaceAdaptiveViewSelectionScore ?: 0.0) * 0.60 +
                    (denseSurfaceAdaptiveViewDiversityScore ?: 0.0) * 0.40
                ).coerceIn(0.0, 1.0)
    val memoryBytes = memoryLeaseBytesActive ?: residentResourceBytes ?: 0L
    val memory = (1.0 - memoryBytes.toDouble() / (1L shl 30).toDouble()).coerceIn(0.0, 1.0)
    val opticalBlurRisk =
        maxOf(
            nativeExposureMotionBlurRisk
                ?.takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0,
            cameraOpticalBlurRisk
                ?.takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0,
        )
    val thermal = if (thermalConstrained == true) 0.35 else 0.80
    val p95 = (motionP95ErrorMillimeters ?: modeledPrecisionP95Millimeters ?: 0.0).coerceAtLeast(0.0)
    val observed = (denseSurfaceTrustedPointRatio ?: denseLandmarkVisibleCoverage ?: 0.0).coerceIn(0.0, 1.0)
    return FrontierOperatorState(
        operationalState = state,
        actorCount = actorCount,
        authoritativeCameraCount = authorityCameraCount,
        witnessCameraCount = witnessEstimate,
        trackingConfidence = tracking,
        reconstructionConfidence = reconstruction,
        timingConfidence = timing,
        memoryHeadroom = memory,
        thermalHeadroom = thermal,
        precisionP95Millimeters = p95,
        metricEvidenceAuthority = metricEvidenceAuthority,
        opticalBlurRisk = opticalBlurRisk,
        captureAuthorityScore =
            captureAuthorityScore ?: 0.0,
        captureAuthorityBand =
            captureAuthorityBand ?: "BLOCKED",
        captureAuthorityHardBlocked =
            captureAuthorityHardBlocked ?: true,
        captureAuthorityPrecisionQualified =
            captureAuthorityPrecisionQualified ?: false,
        observedGeometryRatio = observed,
        inferredGeometryRatio = (1.0 - observed).coerceIn(0.0, 1.0),
        warnings = listOfNotNull(
            nativeDropRatio?.takeIf { it > 0.02 }?.let { "native-drop-ratio=" + String.format("%.3f", it) },
            resourceLeakRisk?.takeIf { it > 0.20 }?.let { "resource-leak-risk=" + String.format("%.2f", it) },
            denseLandmarkHardFailure?.takeIf { it }?.let { "dense-landmark-hard-failure" },
            calibrationRecalibrationRequired?.takeIf { it }?.let { "recalibration-required" },
            opticalBlurRisk.takeIf { it > 0.48 }?.let {
                "optical-blur-risk=" + String.format("%.2f", it)
            },
        ),
    )
}
}
