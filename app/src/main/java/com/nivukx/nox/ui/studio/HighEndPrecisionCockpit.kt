package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * Operator-first precision cockpit for the high-end NoX runtime.
 *
 * This surface keeps the operator focused on measurable state rather than decorative certainty.
 * Unknown telemetry stays unknown, qualification stays separate from estimation, and the
 * reconstruction viewport remains a content surface rather than a glass-covered dashboard.
 */
@Composable
fun HighEndPrecisionCockpit(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val connectedCameraCount =
        telemetry.camerasConnected?.coerceIn(0, 128)
    val expectedCameraCount =
        telemetry.camerasExpected?.coerceIn(1, 128)
    val cameraRatio =
        safeRatio(connectedCameraCount, expectedCameraCount ?: 128)
    val actorRatio = (
        telemetry.multiPersonAdmittedCount
            .coerceAtLeast(0)
            .toDouble() / 50.0
        ).coerceIn(0.0, 1.0)

    // This composite includes measured and modeled channels; it is not a physical
    // accuracy guarantee. The separate qualification badge below carries authority status.
    val quality = (
        telemetry.precisionEvidenceScore
            ?: telemetry.precisionFeasibilityScore
            ?: telemetry.productionQualityScore
            ?: 0.0
        ).coerceIn(0.0, 1.0)

    val temporal = (
        telemetry.synchronizationConfidence
            ?: telemetry.interleavedTemporalConsistency
            ?: 0.0
        ).coerceIn(0.0, 1.0)

    val geometry = (
        telemetry.geometricInformationConditionScore
            ?: telemetry.interleavedSpatialAgreement
            ?: telemetry.crossViewAgreement
            ?: telemetry.selectedViewDiversity
            ?: 0.0
        ).coerceIn(0.0, 1.0)

    val identity =
        telemetry.multiPersonIdentityContinuity.coerceIn(0.0, 1.0)

    val surface = (
        telemetry.denseSurfaceMeanTemporalStability
            ?: telemetry.surfaceTemporalStability
            ?: 0.0
        ).coerceIn(0.0, 1.0)

    val resourceSafety = (
        1.0 -
            maxOf(
                telemetry.resourceLeakRisk ?: 0.0,
                telemetry.nativeDropRatio ?: 0.0,
                telemetry.queuePressure ?: 0.0,
            )
        ).coerceIn(0.0, 1.0)

    LiquidGlassSurface(
        modifier = modifier.fillMaxWidth(),
        contentPadding = 14.dp,
        materialRole =
            StudioMaterialAdaptationEngine.MaterialRole.PRIMARY_NAVIGATION,
        underlyingLuminance = 0.30f,
        contentContrast = 0.98f,
        capturePriority = 1.0f,
        operatorFocus = 1.0f,
        plateProtection = 1.0f,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        "PRECISION CONTROL COCKPIT",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Live capture, geometry, identity and reconstruction authority",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Text(
                    precisionState(telemetry),
                    style = MaterialTheme.typography.labelLarge,
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                CockpitMetric(
                    label = "CAMERAS",
                    value =
                        when {
                            connectedCameraCount == null -> "UNKNOWN / 128 cap"
                            expectedCameraCount == null -> "${connectedCameraCount} / 128 cap"
                            else -> "${connectedCameraCount} / ${expectedCameraCount} expected"
                        },
                    progress = cameraRatio,
                    modifier = Modifier.weight(1f),
                )
                CockpitMetric(
                    label = "ACTORS",
                    value =
                        "\${telemetry.multiPersonAdmittedCount.coerceAtLeast(0)} / 50",
                    progress = actorRatio,
                    modifier = Modifier.weight(1f),
                )
                CockpitMetric(
                    label = "QUALITY INDEX",
                    value =
                        "\${(quality * 100.0).roundToInt()}%",
                    progress = quality,
                    modifier = Modifier.weight(1f),
                )
                CockpitMetric(
                    label = "IDENTITY",
                    value =
                        "\${(identity * 100.0).roundToInt()}%",
                    progress = identity,
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    CockpitStatus(
                        label = "Temporal authority",
                        value = temporal,
                        detail =
                            telemetry.synchronizationResidualP95Milliseconds?.let {
                                "\${"%.2f".format(it)} ms P95 residual"
                            } ?: "timing evidence unavailable",
                    )
                    CockpitStatus(
                        label = "Geometry authority",
                        value = geometry,
                        detail =
                            telemetry.reconstructionUncertaintyP95Meters?.let {
                                "\${"%.2f".format(it * 1000.0)} mm P95 uncertainty"
                            } ?: "3D uncertainty unavailable",
                    )
                    CockpitStatus(
                        label = "Surface stability",
                        value = surface,
                        detail =
                            telemetry.surfaceTopologyContinuity?.let {
                                "\${(it * 100.0).roundToInt()}% topology continuity"
                            } ?: "surface topology evidence unavailable",
                    )
                }

                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    StatusLine(
                        "Capture path",
                        when {
                            telemetry.highSpeedSessionActive ->
                                "HIGH-SPEED • \${telemetry.highSpeedSelectedFps ?: 0} FPS"
                            telemetry.cameraZeroCopyReady == true ->
                                "ZERO-COPY READY"
                            else ->
                                "STANDARD"
                        },
                    )
                    StatusLine(
                        "Temporal sampling",
                        telemetry.interleavedEffectiveInputRateHz?.let {
                            "\${"%.1f".format(it)} Hz effective"
                        },
                    )
                    StatusLine(
                        "Identity protection",
                        when {
                            telemetry.identityExchangeState == "HOLD" ||
                                telemetry.identityExchangeState == "HIGH_RISK" ->
                                "HOLD"
                            telemetry.multiPersonHeldCount > 0 ->
                                "\${telemetry.multiPersonHeldCount} SUBJECTS HELD"
                            else ->
                                "STABLE"
                        },
                    )
                    StatusLine(
                        "Resource safety",
                        when {
                            resourceSafety >= 0.90 -> "HEALTHY"
                            resourceSafety >= 0.70 -> "DEGRADED"
                            else -> "PROTECTIVE"
                        },
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                CockpitBadge(
                    label = "READOUT",
                    value =
                        if (telemetry.hardwareReadoutTimestampSupported) {
                            "HARDWARE"
                        } else {
                            "SOFTWARE"
                        },
                    modifier = Modifier.weight(1f),
                )
                CockpitBadge(
                    label = "3D",
                    value =
                        telemetry.reconstructionConfidence?.let {
                            "\${(it * 100.0).roundToInt()}%"
                        } ?: "UNKNOWN",
                    modifier = Modifier.weight(1f),
                )
                CockpitBadge(
                    label = "SURFACE",
                    value =
                        if (surface >= 0.75) "STABLE" else "DEGRADED",
                    modifier = Modifier.weight(1f),
                )
                CockpitBadge(
                    label = "QUALIFICATION",
                    value =
                        when {
                            telemetry.captureAuthorityPrecisionQualified == true ->
                                "QUALIFIED"
                            telemetry.referenceBenchmarkStatus != null ->
                                telemetry.referenceBenchmarkStatus.uppercase()
                            else ->
                                "EVIDENCE"
                        },
                    modifier = Modifier.weight(1f),
                )
            }

            telemetry.captureAuthorityWarnings
                .orEmpty()
                .take(3)
                .forEach { warning ->
                    Text(
                        warning,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
        }
    }
}

@Composable
private fun CockpitMetric(
    label: String,
    value: String,
    progress: Double,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, style = MaterialTheme.typography.labelSmall)
            Text(value, style = MaterialTheme.typography.labelMedium)
        }
        LinearProgressIndicator(
            progress = {
                progress.coerceIn(0.0, 1.0).toFloat()
            },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun CockpitStatus(
    label: String,
    value: Double,
    detail: String,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, style = MaterialTheme.typography.labelMedium)
            Text(
                "\${(value * 100.0).roundToInt()}%",
                style = MaterialTheme.typography.labelMedium,
            )
        }
        Text(
            detail,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun CockpitBadge(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        modifier = modifier,
        clear = true,
        elevated = false,
        contentPadding = 8.dp,
        materialRole =
            StudioMaterialAdaptationEngine.MaterialRole.SECONDARY_TELEMETRY,
        underlyingLuminance = 0.28f,
        contentContrast = 0.98f,
        capturePriority = 1.0f,
        operatorFocus = 0.92f,
        plateProtection = 1.0f,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(1.dp),
        ) {
            Text(label, style = MaterialTheme.typography.labelSmall)
            Text(value, style = MaterialTheme.typography.labelMedium)
        }
    }
}

private fun safeRatio(
    numerator: Int?,
    denominator: Int?,
): Double {
    val top = numerator?.coerceAtLeast(0) ?: 0
    val bottom = denominator?.coerceAtLeast(1) ?: 1
    return (top.toDouble() / bottom.toDouble())
        .coerceIn(0.0, 1.0)
}

private fun precisionState(
    telemetry: StudioTelemetryState,
): String =
    when {
        telemetry.captureAuthorityHardBlocked == true ->
            "BLOCKED"
        telemetry.captureAuthorityPrecisionQualified == true ->
            "QUALIFIED"
        telemetry.captureAuthorityBand != null ->
            telemetry.captureAuthorityBand
        telemetry.productionQualityBand != null ->
            telemetry.productionQualityBand
        else ->
            "EVIDENCE"
    }
