package com.nivukx.nox

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.ui.GlassCard
import com.nivukx.nox.ui.GlassMetric
import com.nivukx.nox.ui.SectionTitle
import com.nivukx.nox.ui.StatusPill

@Composable
fun MotionDynamicsPanel(
    phase: String,
    score: Double,
    contactStability: Double,
    symmetry: Double,
    dynamicStress: Double,
    uncertainty: Double,
    trustedJointRatio: Double,
    microMotionRetention: Double,
    lowFrequencyMotionScore: Double = 0.0,
    highFrequencyMotionScore: Double = 0.0,
    modifier: Modifier = Modifier,
) {
    GlassCard(
        modifier.fillMaxWidth(),
        intensity = 0.98f,
    ) {
        Column(
            Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                SectionTitle(
                    "DYNAMICS INTELLIGENCE",
                    "Kinematic phase and confidence evidence from reconstructed 3D motion.",
                )
                StatusPill(
                    phase,
                    score >= 0.60,
                )
            }

            Text(
                "The dynamics engine is downstream of temporal reconstruction. " +
                    "It does not invent missing joints; it scores the physical consistency " +
                    "of the state already reconstructed.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Dynamics",
                    "%.0f%%".format(score * 100.0),
                    phase,
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Contact",
                    "%.0f%%".format(contactStability * 100.0),
                    "lower body",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Symmetry",
                    "%.0f%%".format(symmetry * 100.0),
                    "bilateral",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Stress",
                    "%.0f%%".format(dynamicStress * 100.0),
                    "jerk regime",
                    Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Uncertainty",
                    "%.0f%%".format(uncertainty * 100.0),
                    "position",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Trusted joints",
                    "%.0f%%".format(trustedJointRatio * 100.0),
                    ">= confidence gate",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Micro-motion",
                    "%.0f%%".format(microMotionRetention * 100.0),
                    "preserved",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Sustained",
                    "%.0f%%".format(lowFrequencyMotionScore * 100.0),
                    "low-frequency",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Articulation burst",
                    "%.0f%%".format(highFrequencyMotionScore * 100.0),
                    if (highFrequencyMotionScore >= 0.68) {
                        "priority"
                    } else {
                        "stable"
                    },
                    Modifier.weight(1f),
                )
            }
        }
    }
}
