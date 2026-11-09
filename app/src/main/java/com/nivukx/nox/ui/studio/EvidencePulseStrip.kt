package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Dense operator evidence strip for the live Motion Studio.
 *
 * The strip exposes only evidence already present in native telemetry. It never invents a
 * percentage when the native engine has no authoritative value.
 */
@Composable
fun EvidencePulseStrip(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val signals = listOf(
        EvidencePulseSignal("SYNC", "Synchronization", telemetry.synchronizationConfidence),
        EvidencePulseSignal("RIG", "Calibration", telemetry.calibrationReady?.let { if (it) 1.0 else 0.0 }, true),
        EvidencePulseSignal("BODY", "Tracking", telemetry.trackingConfidence),
        EvidencePulseSignal("3D", "Reconstruction", telemetry.reconstructionConfidence),
        EvidencePulseSignal("OUT", "Export readiness", if (StudioUiStateReducer.exportEnabled(telemetry)) 1.0 else 0.0, true),
    )

    LiquidGlassSurface(
        clear = true,
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text("EVIDENCE PULSE", style = MaterialTheme.typography.labelMedium)
                Text("${signals.count { it.value != null }}/5 SIGNALS", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                signals.forEach { signal ->
                    EvidencePulseCell(signal = signal, modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

private data class EvidencePulseSignal(
    val shortLabel: String,
    val label: String,
    val value: Double?,
    val binary: Boolean = false,
)

@Composable
private fun EvidencePulseCell(
    signal: EvidencePulseSignal,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(signal.shortLabel, style = MaterialTheme.typography.labelSmall)
            Text(signal.displayValue(), style = MaterialTheme.typography.labelSmall)
        }
        LinearProgressIndicator(
            progress = { signal.value?.coerceIn(0.0, 1.0)?.toFloat() ?: 0f },
            modifier = Modifier.fillMaxWidth(),
        )
        Text(signal.label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1)
    }
}

private fun EvidencePulseSignal.displayValue(): String = when {
    value == null -> "N/A"
    binary && value >= 1.0 -> "READY"
    binary -> "HOLD"
    else -> (value.coerceIn(0.0, 1.0) * 100.0).toInt().toString() + "%"
}
