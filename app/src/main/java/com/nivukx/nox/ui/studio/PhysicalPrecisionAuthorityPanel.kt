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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.reconstruction.PhysicalPrecisionCapabilityEngine
import kotlin.math.max

@Composable
fun PhysicalPrecisionAuthorityPanel(
    decision: PhysicalPrecisionCapabilityEngine.Decision,
    modifier: Modifier = Modifier,
) {
    val stateLabel = when (decision.state) {
        PhysicalPrecisionCapabilityEngine.State.NO_DATA -> "NO DATA"
        PhysicalPrecisionCapabilityEngine.State.EVIDENCE_ONLY -> "EVIDENCE ONLY"
        PhysicalPrecisionCapabilityEngine.State.HIGH_PRECISION -> "HIGH PRECISION"
        PhysicalPrecisionCapabilityEngine.State.SUB_MILLIMETER_QUALIFIED -> "SUB-MM QUALIFIED"
        PhysicalPrecisionCapabilityEngine.State.BLOCKED -> "BLOCKED"
    }
    val progress = if (decision.operationalP95Millimeters.isFinite()) {
        (1.0 - decision.operationalP95Millimeters / 2.0).coerceIn(0.0, 1.0)
    } else 0.0

    LiquidGlassPanel(modifier) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text("Precision Authority", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Live uncertainty envelope, not a fabricated accuracy claim",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(stateLabel, style = MaterialTheme.typography.labelLarge)
            }
            LinearProgressIndicator(
                progress = { progress.toFloat() },
                modifier = Modifier.fillMaxWidth(),
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PrecisionMetric(
                    "P95 envelope",
                    if (decision.operationalP95Millimeters.isFinite())
                        "%.3f mm".format(max(0.0, decision.operationalP95Millimeters))
                    else "N/A",
                    Modifier.weight(1f),
                )
                PrecisionMetric(
                    "Evidence",
                    decision.evidenceLevel.name.replace('_', ' '),
                    Modifier.weight(1f),
                )
                PrecisionMetric(
                    "Bottleneck",
                    decision.bottleneck.name.replace('_', ' '),
                    Modifier.weight(1f),
                )
                PrecisionMetric(
                    "Score",
                    "%.2f".format(decision.score),
                    Modifier.weight(0.7f),
                )
            }
            Text(
                when {
                    decision.measuredSubMillimeterQualified ->
                        "External-reference qualification and live conditions satisfy the sub-millimetre gate."
                    decision.liveSubMillimeterCapable ->
                        "Current live geometry is inside the sub-millimetre envelope, but certification remains reference-gated."
                    else ->
                        "Current live conditions are outside the sub-millimetre operating envelope."
                },
                style = MaterialTheme.typography.bodyMedium,
            )
            decision.warnings.take(4).forEach {
                Text(
                    "• " + it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            decision.recommendations.firstOrNull()?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

@Composable
private fun PrecisionMetric(
    label: String,
    value: String,
    modifier: Modifier,
) {
    Column(modifier) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.labelLarge)
    }
}
