package com.nivukx.nox.ui.studio

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Compact operator workflow surface.
 *
 * The main studio should expose intent and evidence before exposing every knob.
 * This shelf intentionally uses existing runtime presets instead of creating a
 * second, divergent settings model.
 */
@Composable
fun PrecisionWorkflowShelf(
    selectedSystemPreset: String,
    synchronization: Double?,
    geometricObservability: Double?,
    densePrecision: Double?,
    healthScore: Double?,
    timingHealthy: Boolean?,
    cameraDiversity: Double? = null,
    cameraStarvationRisk: Double? = null,
    correlatedEvidence: Double? = null,
    correlatedEvidenceState: String? = null,
    identityExchangeRisk: Double? = null,
    identityExchangeState: String? = null,
    effectScale: Float = 0.82f,
    onPreset: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = true,
        effectScale = effectScale.coerceIn(0.35f, 1.0f),
        modifier = modifier,
    ) {
        Column(
            modifier = Modifier.padding(
                horizontal = 12.dp,
                vertical = 10.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        text = "Capture intent",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Text(
                        text = "Choose the operating priority, then watch evidence.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                StatusBadge(
                    label = healthLabel(healthScore),
                    score = healthScore,
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                listOf(
                    "QUALITY" to "Quality",
                    "BALANCED" to "Balanced",
                    "LOW_LATENCY" to "Low latency",
                ).forEach { (id, label) ->
                    WorkflowChip(
                        label = label,
                        selected = selectedSystemPreset == id,
                        onClick = { onPreset(id) },
                    )
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                MetricPill(
                    label = "Sync",
                    value = percentage(synchronization),
                    detail = "multi-camera clock",
                )
                MetricPill(
                    label = "Geometry",
                    value = percentage(geometricObservability),
                    detail = "camera information",
                )
                MetricPill(
                    label = "Precision",
                    value =
                        when {
                            densePrecision == null -> "PENDING"
                            densePrecision >= 0.999 -> "QUALIFIED"
                            else -> percentage(densePrecision)
                        },
                    detail = "dense 3D evidence",
                )
                MetricPill(
                    label = "Views",
                    value = percentage(cameraDiversity),
                    detail =
                        when {
                            cameraStarvationRisk == null -> "camera geometry"
                            cameraStarvationRisk > 0.72 -> "starvation risk high"
                            cameraStarvationRisk > 0.45 -> "camera dependence rising"
                            else -> "diverse evidence"
                        },
                )
                MetricPill(
                    label = "Evidence",
                    value =
                        when (correlatedEvidenceState) {
                            "STRONG" -> percentage(correlatedEvidence)
                            "GUARDED" -> "GUARDED"
                            "DEGRADED" -> "DEGRADED"
                            "REJECTED" -> "HOLD"
                            else -> percentage(correlatedEvidence)
                        },
                    detail =
                        when {
                            correlatedEvidenceState == "REJECTED" ->
                                "correlated quality gate"
                            else ->
                                "independent evidence"
                        },
                )
                MetricPill(
                    label = "Identity",
                    value =
                        when (identityExchangeState) {
                            "HOLD", "HIGH_RISK" -> "HOLD"
                            "GUARDED" -> "GUARDED"
                            else -> percentage(
                                (1.0 - (identityExchangeRisk ?: 0.0))
                                    .coerceIn(0.0, 1.0),
                            )
                        },
                    detail = "multi-person continuity",
                )
                MetricPill(
                    label = "Timing",
                    value =
                        when (timingHealthy) {
                            true -> "HEALTHY"
                            false -> "HOLD"
                            null -> "UNKNOWN"
                        },
                    detail = "capture timing",
                )
            }
        }
    }
}

@Composable
private fun WorkflowChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(14.dp),
        tonalElevation = if (selected) 4.dp else 1.dp,
        color =
            if (selected) {
                MaterialTheme.colorScheme.primaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f)
            },
        contentColor =
            if (selected) {
                MaterialTheme.colorScheme.onPrimaryContainer
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier.padding(
                horizontal = 13.dp,
                vertical = 8.dp,
            ),
        )
    }
}

@Composable
private fun MetricPill(
    label: String,
    value: String,
    detail: String,
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 0.dp,
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.45f),
    ) {
        Column(
            modifier = Modifier.padding(
                horizontal = 11.dp,
                vertical = 8.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(1.dp),
        ) {
            Text(
                text = label.uppercase(),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = detail,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun StatusBadge(
    label: String,
    score: Double?,
) {
    val alpha = when {
        score == null -> 0.50f
        score >= 0.80 -> 0.82f
        score >= 0.50 -> 0.68f
        else -> 0.54f
    }

    Surface(
        shape = RoundedCornerShape(999.dp),
        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = alpha),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            modifier = Modifier.padding(
                horizontal = 10.dp,
                vertical = 6.dp,
            ),
        )
    }
}

private fun healthLabel(score: Double?): String =
    when {
        score == null -> "UNKNOWN"
        score >= 0.80 -> "READY"
        score >= 0.50 -> "DEGRADED"
        else -> "HOLD"
    }

private fun percentage(value: Double?): String =
    value
        ?.takeIf { it.isFinite() }
        ?.coerceIn(0.0, 1.0)
        ?.let { "%.0f%%".format(it * 100.0) }
        ?: "UNKNOWN"
