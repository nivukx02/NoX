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
fun MotionUncertaintyPanel(
    coverage: Double,
    confidence: Double,
    oodRisk: Double,
    radiusMeters: Double,
    calibrationProgress: Double,
    policy: String,
    warnings: List<String>,
    modifier: Modifier = Modifier,
) {
    val stable =
        policy == "NORMAL" &&
            confidence >= 0.65 &&
            oodRisk < 0.50

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
                    "MOTION UNCERTAINTY",
                    "Empirical joint envelopes, calibration progress, and out-of-distribution risk.",
                )
                StatusPill(policy, stable)
            }

            Text(
                "NoX keeps uncertainty explicit. The envelope is empirical and calibration-dependent; " +
                    "it is not presented as a universal statistical guarantee.",
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
                    "Coverage",
                    "%.0f%%".format(coverage * 100.0),
                    "empirical",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Confidence",
                    "%.0f%%".format(confidence * 100.0),
                    "calibrated",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "OOD risk",
                    "%.0f%%".format(oodRisk * 100.0),
                    "motion",
                    Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GlassMetric(
                    "Mean radius",
                    "%.3fm".format(radiusMeters),
                    "joint envelope",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Calibration",
                    "%.0f%%".format(calibrationProgress * 100.0),
                    "ready",
                    Modifier.weight(1f),
                )
                GlassMetric(
                    "Warnings",
                    warnings.size.toString(),
                    "active",
                    Modifier.weight(1f),
                )
            }

            val guidance =
                when {
                    oodRisk >= 0.74 ->
                        "Hold long-horizon extrapolation and reacquire observed motion before trusting the solve."
                    coverage < 0.85 ->
                        "Calibration coverage is below the production target. Preserve more adjacent measured frames."
                    calibrationProgress < 0.35 ->
                        "The uncertainty model is still warming up. Treat early envelopes as provisional."
                    else ->
                        "Calibrated envelopes are available for downstream weighting and export validation."
                }

            Text(
                guidance,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
