package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** Operator-facing evidence ladder that exposes which authority layer limits a capture. */
@Composable
fun EvidenceAuthorityLadderPanel(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val stages = listOf(
        EvidenceStage(
            label = "SYNC",
            score = telemetry.synchronizationConfidence ?: 0.0,
            detail = telemetry.synchronizationResidualP95Milliseconds
                ?.let { value -> "P95 %.2f ms".format(value) }
                ?: "no synchronized sample",
        ),
        EvidenceStage(
            label = "CALIBRATION",
            score = telemetry.calibrationReady
                ?.let { ready -> if (ready) 1.0 else 0.0 }
                ?: (telemetry.cameraObservabilityScore ?: 0.0),
            detail = telemetry.referenceBenchmarkStatus ?: "rig state unavailable",
        ),
        EvidenceStage(
            label = "VIEWS",
            score = telemetry.selectedViewDiversity
                ?: telemetry.denseSurfaceViewDiversity
                ?: 0.0,
            detail = telemetry.selectedCameraCount
                ?.let { count -> "$count selected cameras" }
                ?: "view diversity unavailable",
        ),
        EvidenceStage(
            label = "3D",
            score = telemetry.reconstructionConfidence
                ?: telemetry.denseSurfaceProbabilisticEvidenceScore
                ?: 0.0,
            detail = telemetry.reconstructionUncertaintyP95Meters
                ?.let { value -> "P95 %.3f m".format(value) }
                ?: "metric uncertainty unavailable",
        ),
        EvidenceStage(
            label = "SURFACE",
            score = telemetry.denseSurfaceTrustedPointRatio
                ?: telemetry.denseSurfaceProbabilisticEvidenceScore
                ?: telemetry.denseLandmarkCoverage
                ?: 0.0,
            detail = telemetry.denseSurfaceUncertaintyP95Meters
                ?.let { value -> "P95 %.3f m".format(value) }
                ?: "surface evidence unavailable",
        ),
        EvidenceStage(
            label = "FACE/EYE",
            score = telemetry.faceEyeMetricConfidence ?: 0.0,
            detail = telemetry.faceEyeMetricState ?: "face-eye authority unavailable",
        ),
    )

    val limiter = stages.minByOrNull { stage -> stage.score }

    Surface(
        modifier = modifier,
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        contentColor = MaterialTheme.colorScheme.onSurface,
        shape = androidx.compose.foundation.shape.RoundedCornerShape(22.dp),
        tonalElevation = 2.dp,
        shadowElevation = 3.dp,
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        "EVIDENCE AUTHORITY",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Measure first. Infer only where measured evidence is insufficient.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    limiter?.let { stage -> "LIMITER ${stage.label}" } ?: "NO DATA",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                stages.take(3).forEach { stage ->
                    GlassGauge(
                        label = stage.label,
                        value = stage.score.toFloat().coerceIn(0f, 1f),
                        modifier = Modifier.weight(1f),
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                stages.drop(3).forEach { stage ->
                    GlassGauge(
                        label = stage.label,
                        value = stage.score.toFloat().coerceIn(0f, 1f),
                        modifier = Modifier.weight(1f),
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                stages.forEach(::EvidenceStageRow)
            }
        }
    }
}

private data class EvidenceStage(
    val label: String,
    val score: Double,
    val detail: String,
)

@Composable
private fun EvidenceStageRow(stage: EvidenceStage) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 2.dp, vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(stage.label, style = MaterialTheme.typography.labelSmall)
        Text(
            "${(stage.score.coerceIn(0.0, 1.0) * 100.0).toInt()}% · ${stage.detail}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
