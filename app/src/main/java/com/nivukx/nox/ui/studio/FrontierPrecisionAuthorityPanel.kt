package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlin.math.abs

@Composable
fun FrontierPrecisionAuthorityPanel(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val p95 = telemetry.motionP95ErrorMillimeters
    val p99 = telemetry.motionP99ErrorMillimeters
    val modeledP95 = telemetry.modeledPrecisionP95Millimeters
    val modeledP99 = telemetry.modeledPrecisionP99Millimeters

    val readiness =
        listOfNotNull(
            telemetry.precisionFeasibilityScore,
            telemetry.denseSubMillimeterFeasibilityScore,
            telemetry.productionQualityScore,
        ).averageOrNull() ?: 0.0

    val reference =
        telemetry.referenceBenchmarkStatus
            ?.takeIf { it.isNotBlank() }
            ?: "NO_REFERENCE"

    val grade =
        telemetry.precisionGrade
            ?.takeIf { it.isNotBlank() }
            ?: "UNQUALIFIED"

    val timing =
        telemetry.synchronizationResidualP95Milliseconds
            ?: telemetry.temporalLagMilliseconds

    val timingReady =
        timing == null || timing <= 2.5

    val geometryReady =
        (telemetry.selectedViewDiversity ?: 0.0) >= 0.75 &&
            (telemetry.angularDiversity ?: 0.0) >= 0.60 &&
            (telemetry.crossViewAgreement ?: 0.0) >= 0.90

    val thermalReady =
        telemetry.thermalExtrinsicHardFailure != true &&
            (telemetry.thermalExtrinsicDriftConfidence ?: 1.0) >= 0.75

    val memoryReady =
        (telemetry.resourceLeakRisk ?: 0.0) < 0.40 &&
            (telemetry.queuePressure ?: 0.0) < 0.80

    val authority =
        if (
            telemetry.denseLandmarkHardFailure == true ||
            telemetry.cameraSessionSupported == false
        ) {
            "BLOCKED"
        } else if (
            readiness >= 0.90 &&
            geometryReady &&
            timingReady &&
            thermalReady &&
            memoryReady
        ) {
            "HIGH AUTHORITY"
        } else if (readiness >= 0.70) {
            "EVIDENCE STRONG"
        } else {
            "EVIDENCE LIMITED"
        }

    LiquidGlassSurface(
        clear = false,
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        "FRONTIER PRECISION AUTHORITY",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Measured error, geometry, timing and resource health",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    authority,
                    style = MaterialTheme.typography.labelMedium,
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FrontierMetric(
                    label = "P95",
                    value = formatMm(p95),
                    modifier = Modifier.weight(1f),
                )
                FrontierMetric(
                    label = "P99",
                    value = formatMm(p99),
                    modifier = Modifier.weight(1f),
                )
                FrontierMetric(
                    label = "MODELED",
                    value = formatMm(modeledP95),
                    modifier = Modifier.weight(1f),
                )
                FrontierMetric(
                    label = "GRADE",
                    value = grade,
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FrontierMetric(
                    label = "READINESS",
                    value = "${(readiness * 100.0).toInt()}%",
                    modifier = Modifier.weight(1f),
                )
                FrontierMetric(
                    label = "REFERENCE",
                    value = reference,
                    modifier = Modifier.weight(1f),
                )
                FrontierMetric(
                    label = "VIEWS",
                    value = "${telemetry.selectedCameraCount ?: 0}",
                    modifier = Modifier.weight(1f),
                )
                FrontierMetric(
                    label = "ACTORS",
                    value = telemetry.multiPersonAdmittedCount.toString(),
                    modifier = Modifier.weight(1f),
                )
            }

            Text(
                "Geometry ${status(geometryReady)}  ·  Timing ${status(timingReady)}  ·  Thermal ${status(thermalReady)}  ·  Memory ${status(memoryReady)}",
                style = MaterialTheme.typography.bodySmall,
            )

            val warnings =
                buildList {
                    if (!geometryReady) {
                        add("increase independent viewpoint diversity")
                    }
                    if (!timingReady) {
                        add("tighten inter-camera timing before precision qualification")
                    }
                    if (!thermalReady) {
                        add("thermal drift is reducing extrinsic authority")
                    }
                    if (!memoryReady) {
                        add("resource pressure is preventing maximum detail authority")
                    }
                    if (p95 == null) {
                        add("collect measured reference-aligned error")
                    }
                    if (telemetry.referenceGradeFeasible != true) {
                        add("external reference comparison is not yet qualified")
                    }
                }

            if (warnings.isNotEmpty()) {
                Text(
                    warnings.take(4).joinToString("  ·  "),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (
                p95 != null &&
                modeledP95 != null &&
                p95.isFinite() &&
                modeledP95.isFinite()
            ) {
                Text(
                    "Measured-versus-modeled delta: ${"%.3f".format(abs(p95 - modeledP95))} mm",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (
                p99 != null &&
                modeledP99 != null &&
                p99.isFinite() &&
                modeledP99.isFinite()
            ) {
                Text(
                    "P99 remains separately gated; it is not inferred from P95.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            telemetry.infraredReferenceOutcome
                ?.takeIf { it.isNotBlank() }
                ?.let { outcome ->
                    val qualified = telemetry.infraredReferenceQualified == true
                    LiquidGlassSurface(
                        clear = false,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Column(
                            Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                            ) {
                                Text(
                                    "OPTICAL / INFRARED REFERENCE",
                                    style = MaterialTheme.typography.labelLarge,
                                )
                                Text(
                                    if (qualified) "QUALIFIED" else "INCONCLUSIVE",
                                    style = MaterialTheme.typography.labelMedium,
                                )
                            }
                            Text(
                                outcome.replace('_', ' '),
                                style = MaterialTheme.typography.titleSmall,
                            )
                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                FrontierMetric(
                                    "ΔP95",
                                    formatSignedMm(telemetry.infraredReferenceP95DeltaMillimeters),
                                    Modifier.weight(1f),
                                )
                                FrontierMetric(
                                    "ΔP99",
                                    formatSignedMm(telemetry.infraredReferenceP99DeltaMillimeters),
                                    Modifier.weight(1f),
                                )
                                FrontierMetric(
                                    "CONF",
                                    telemetry.infraredReferenceConfidence?.let { String.format("%d%%", (it * 100.0).toInt()) } ?: "N/A",
                                    Modifier.weight(1f),
                                )
                                FrontierMetric(
                                    "REF U",
                                    formatSignedMm(telemetry.infraredReferenceUncertaintyMillimeters),
                                    Modifier.weight(1f),
                                )
                            }
                            Text(
                                "Negative Δ means lower NoX error. Positive Δ means lower reference error. Tie bands are applied before classification.",
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
private fun FrontierMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    androidx.compose.material3.Surface(
        modifier = modifier,
        shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp),
        tonalElevation = 1.dp,
    ) {
        Column(
            Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(label, style = MaterialTheme.typography.labelSmall)
            Text(value, style = MaterialTheme.typography.titleSmall)
        }
    }
}

private fun formatMm(value: Double?): String = 
    value
        ?.takeIf { it.isFinite() && it >= 0.0 }
        ?.let { "%.2f mm".format(it) }
        ?: "N/A"

private fun status(ready: Boolean): String =
    if (ready) "READY" else "LIMITED"

private fun List<Double>.averageOrNull(): Double? =
    takeIf { it.isNotEmpty() }?.average()

private fun formatSignedMm(value: Double?): String =
    value
        ?.takeIf { it.isFinite() }
        ?.let { String.format("%+.3f mm", it) }
        ?: "N/A"
