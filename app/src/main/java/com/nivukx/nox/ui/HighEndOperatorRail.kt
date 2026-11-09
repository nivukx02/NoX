package com.nivukx.nox.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * Persistent operator-stage telemetry rail.
 *
 * The rail belongs to the content layer rather than the navigation glass layer. It exposes the
 * health of the capture-to-output pipeline in a compact sequence so an operator can identify the
 * current bottleneck without opening deep diagnostics.
 */
@Composable
fun HighEndOperatorRail(
    state: String,
    poseConfidence: Double,
    gazeConfidence: Double,
    handConfidence: Double,
    geometricObservability: Double,
    reconstructionQuality: Double,
    sensorQuality: Double,
    multiViewEvidence: Double,
    integrityScore: Double,
    modifier: Modifier = Modifier,
) {
    val authorityScore =
        minOf(
            sensorQuality.coerceIn(0.0, 1.0),
            geometricObservability.coerceIn(0.0, 1.0),
            reconstructionQuality.coerceIn(0.0, 1.0),
            multiViewEvidence.coerceIn(0.0, 1.0),
            integrityScore.coerceIn(0.0, 1.0),
        )

    val stages = listOf(
        OperatorStage(
            label = "Authority",
            score = authorityScore,
            detail = "evidence gate",
        ),
        OperatorStage(
            label = "Capture",
            score = sensorQuality,
            detail = state,
        ),
        OperatorStage(
            label = "Pose",
            score = poseConfidence,
            detail = "body",
        ),
        OperatorStage(
            label = "Face / Eye",
            score = gazeConfidence,
            detail = "gaze",
        ),
        OperatorStage(
            label = "Hands",
            score = handConfidence,
            detail = "articulation",
        ),
        OperatorStage(
            label = "Geometry",
            score = geometricObservability,
            detail = "observability",
        ),
        OperatorStage(
            label = "3D Solve",
            score = reconstructionQuality,
            detail = "reconstruction",
        ),
        OperatorStage(
            label = "Multi-view",
            score = multiViewEvidence,
            detail = "view evidence",
        ),
        OperatorStage(
            label = "Output",
            score = integrityScore,
            detail = "integrity",
        ),
    )

    Surface(
        modifier = modifier,
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.76f),
        shape = RoundedCornerShape(18.dp),
        tonalElevation = 2.dp,
        shadowElevation = 3.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 10.dp, vertical = 9.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            stages.forEachIndexed { index, stage ->
                OperatorStageCell(stage)

                if (index != stages.lastIndex) {
                    Spacer(
                        modifier = Modifier
                            .height(28.dp)
                            .clip(RoundedCornerShape(100.dp))
                            .background(
                                MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
                            )
                    )
                }
            }
        }
    }
}

private data class OperatorStage(
    val label: String,
    val score: Double,
    val detail: String,
)

@Composable
private fun OperatorStageCell(
    stage: OperatorStage,
) {
    val normalized = stage.score
        .coerceIn(0.0, 1.0)
    val percent = (normalized * 100.0).roundToInt()
    val stateLabel = when {
        normalized >= 0.90 -> "LOCKED"
        normalized >= 0.75 -> "GOOD"
        normalized >= 0.55 -> "DEGRADED"
        normalized >= 0.35 -> "HOLD"
        else -> "BLOCKED"
    }

    Column(
        modifier = Modifier.padding(horizontal = 3.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text(
            text = stage.label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
        )

        Text(
            text = percent.toString() + "%",
            style = MaterialTheme.typography.titleSmall,
        )

        Text(
            text = stateLabel,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary,
            maxLines = 1,
        )

        Text(
            text = stage.detail,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(3.dp)
                .clip(RoundedCornerShape(100.dp))
                .background(
                    MaterialTheme.colorScheme.surfaceContainerHighest,
                ),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(normalized.toFloat())
                    .height(3.dp)
                    .clip(RoundedCornerShape(100.dp))
                    .background(MaterialTheme.colorScheme.primary),
            )
        }
    }
}
