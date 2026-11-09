package com.nivukx.nox

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.ProductionCaptureProfile
import com.nivukx.nox.ui.GlassAction
import com.nivukx.nox.ui.GlassCard
import com.nivukx.nox.ui.GlassMetric
import com.nivukx.nox.ui.SectionTitle
import com.nivukx.nox.ui.StatusPill

/**
 * Operator-facing production policy surface.
 *
 * This exposes policy and evidence rather than implementation details. The
 * operator can choose a workload target, inspect the active budget, and see
 * whether the runtime has accepted, held, or degraded the requested envelope.
 */
@Composable
fun ProductionProfilePanel(
    profile: ProductionCaptureProfile,
    profiles: List<ProductionCaptureProfile>,
    qualityScore: Double,
    policy: String,
    bottleneck: String,
    budgetLevel: String,
    outputAllowed: Boolean,
    detectorInterval: Int,
    meshVertexBudget: Int,
    tsdfStride: Int,
    bundleAdjustmentIterations: Int,
    warnings: List<String>,
    onSelect: (ProductionCaptureProfile) -> Unit,
    modifier: Modifier = Modifier,
) {
    GlassCard(
        modifier.fillMaxWidth(),
        intensity = 1.0f,
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                SectionTitle(
                    "PRODUCTION PROFILE",
                    "Explicit quality envelope, compute budget, and runtime gating.",
                )
                StatusPill(
                    if (outputAllowed) "OUTPUT READY" else "HOLD",
                    outputAllowed,
                )
            }

            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                profiles.forEach { candidate ->
                    GlassAction(
                        text = candidate.label.uppercase(),
                        primary = candidate.id == profile.id,
                    ) {
                        onSelect(candidate)
                    }
                }
            }

            Text(
                profile.description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Quality",
                    "%.0f%%".format(qualityScore * 100.0),
                    policy,
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Budget",
                    budgetLevel,
                    bottleneck,
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Tier",
                    profile.qualityTier.name,
                    profile.tsdfResolutionClass,
                    Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Capture",
                    "\${profile.targetCaptureFps} FPS",
                    "input clock",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Inference",
                    "\${profile.inferenceFps} FPS",
                    "runtime target",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Subjects",
                    profile.maxSubjects.toString(),
                    profile.subjectTrackingMode.name,
                    Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Mesh",
                    "%,d V".format(meshVertexBudget),
                    "\${profile.denseMeshHz} Hz",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "TSDF",
                    "x\$tsdfStride",
                    "\${profile.tsdfVoxelSizeMeters * 1000.0} mm",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Bundle",
                    bundleAdjustmentIterations.toString(),
                    "iterations",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Detector",
                    "every \$detectorInterval",
                    "frames",
                    Modifier.weight(1f),
                )
            }

            Column(
                Modifier
                    .fillMaxWidth()
                    .widthIn(min = 220.dp),
                verticalArrangement = Arrangement.spacedBy(5.dp),
            ) {
                Text(
                    "Face \${profile.faceDetailPolicy.name} • " +
                        "Eyes \${profile.eyeTrackingPolicy.name} • " +
                        "Hands \${profile.handTrackingPolicy.name}",
                    style = MaterialTheme.typography.labelMedium,
                )
                Text(
                    "Surface \${profile.surfacePolicy.name} • " +
                        "Topology \${profile.topologyPolicy.name} • " +
                        "GPU \${profile.gpuExecutionPolicy.name}",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (warnings.isNotEmpty()) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        "Runtime notes",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    warnings.take(6).forEach { warning ->
                        Text(
                            "• \$warning",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}
