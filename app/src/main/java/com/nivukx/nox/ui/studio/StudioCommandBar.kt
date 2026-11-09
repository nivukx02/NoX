package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun StudioCommandBar(
    telemetry: StudioTelemetryState,
    query: String,
    onQueryChange: (String) -> Unit,
    onCaptureToggle: () -> Unit,
    onExport: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = true,
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        "NOX MOTION STUDIO",
                        style = MaterialTheme.typography.labelSmall,
                    )
                    Text(
                        "Production operator surface",
                        style = MaterialTheme.typography.titleSmall,
                    )
                }

                StatusPill(
                    label = "CAMS",
                    value = telemetry.cameraHealth.asPercentOrNa(),
                )
                StatusPill(
                    label = "SYNC",
                    value = telemetry.synchronizationConfidence.asPercentOrNa(),
                )
                StatusPill(
                    label = "RECON",
                    value = telemetry.reconstructionConfidence.asPercentOrNa(),
                )
                StatusPill(
                    label = "SYS",
                    value = telemetry.systemConfidence.asPercentOrNa(),
                )
                HealthPill(health = StudioHealthModel.summarize(telemetry))
            }

            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                OutlinedTextField(
                    value = query,
                    onValueChange = onQueryChange,
                    singleLine = true,
                    label = { Text("Search commands and settings") },
                    modifier = Modifier.weight(1f),
                )

                TextButton(
                    onClick = onCaptureToggle,
                ) {
                    Text("CAPTURE")
                }

                TextButton(
                    onClick = onExport,
                    enabled = StudioUiStateReducer.exportEnabled(telemetry),
                ) {
                    Text("EXPORT")
                }

                Column(
                    horizontalAlignment = Alignment.End,
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        telemetry.calibrationReady?.let {
                            if (it) "CAL READY" else "CAL HOLD"
                        } ?: "CAL N/A",
                        style = MaterialTheme.typography.labelSmall,
                    )
                    Text(
                        telemetry.reconstructionUncertaintyP95Meters
                            ?.let { "P95 " + formatMeters(it) }
                            ?: "P95 N/A",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun HealthPill(
    health: StudioHealthSummary,
) {
    val label = when (health.band) {
        StudioHealthBand.UNKNOWN -> "UNKNOWN"
        StudioHealthBand.BLOCKED -> "BLOCKED"
        StudioHealthBand.DEGRADED -> "DEGRADED"
        StudioHealthBand.READY -> "READY"
    }

    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
        tonalElevation = 1.dp,
    ) {
        Column(
            Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.spacedBy(1.dp),
        ) {
            Text(
                "PIPELINE",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun StatusPill(
    label: String,
    value: String,
) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        tonalElevation = 1.dp,
    ) {
        Row(
            Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.width(2.dp))
            Text(
                value,
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

private fun Double?.asPercentOrNa(): String =
    this?.let {
        (it.coerceIn(0.0, 1.0) * 100.0).toInt().toString() + "%"
    } ?: "N/A"

private fun formatMeters(value: Double): String {
    if (!value.isFinite() || value < 0.0) return "N/A"
    return when {
        value < 1.0 -> "%.0f mm".format(value * 1000.0)
        else -> "%.2f m".format(value)
    }
}
