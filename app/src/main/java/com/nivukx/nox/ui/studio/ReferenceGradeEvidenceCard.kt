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
import kotlin.math.roundToInt

@Composable
fun ReferenceGradeEvidenceCard(
    modeledP95Millimeters: Double?,
    synchronizationConfidence: Double?,
    synchronizationResidualP95Milliseconds: Double? = null,
    synchronizationOutlierRatio: Double? = null,
    viewDiversity: Double?,
    trackingConfidence: Double?,
    cameraObservability: Double?,
    measuredP95Millimeters: Double? = null,
    measuredP99Millimeters: Double? = null,
    externalReferenceUncertaintyMillimeters: Double? = null,
    empiricalBenchmarkAccepted: Boolean = false,
    matchedCaptureConditions: Boolean = false,
    externalReferenceDocumented: Boolean = false,
    occlusionRisk: Double? = null,
    temporalConsistency: Double? = null,
    trustedJointRatio: Double? = null,
    handObjectContactConfidence: Double? = null,
    interactionCollisionConfidence: Double? = null,
    maximumContactPenetrationMillimeters: Double? = null,
    metricScaleProvenance: String? = null,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        modifier = modifier,
        clear = false,
        elevated = true,
        contentPadding = 16.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "REFERENCE-GRADE EVIDENCE",
                style = MaterialTheme.typography.labelLarge,
            )
            Text(
                text = "2 mm target • measured qualification stays separate from modeled feasibility",
                style = MaterialTheme.typography.bodySmall,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                EvidenceMetric(
                    label = "Model P95",
                    value = modeledP95Millimeters?.let {
                        String.format("%.2f mm", it)
                    } ?: "NO MODEL",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Measured P95",
                    value = measuredP95Millimeters?.let {
                        String.format("%.2f mm", it)
                    } ?: "PENDING",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Measured P99",
                    value = measuredP99Millimeters?.let {
                        String.format("%.2f mm", it)
                    } ?: "PENDING",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                EvidenceMetric(
                    label = "Sync",
                    value = synchronizationConfidence?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Sync P95",
                    value = synchronizationResidualP95Milliseconds?.let {
                        String.format("%.2f ms", it)
                    } ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Sync outliers",
                    value = synchronizationOutlierRatio?.let {
                        String.format("%.1f%%", it * 100.0)
                    } ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Views",
                    value = viewDiversity?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Tracking",
                    value = trackingConfidence?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Rig",
                    value = cameraObservability?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                EvidenceMetric(
                    label = "Visibility trust",
                    value = occlusionRisk?.let {
                        ((1.0 - it.coerceIn(0.0, 1.0)) * 100.0)
                            .roundToInt()
                            .toString() + "%"
                    } ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Temporal",
                    value = temporalConsistency?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Trusted joints",
                    value = trustedJointRatio?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Metric provenance",
                    value = metricScaleProvenance ?: "NOT CONNECTED",
                    modifier = Modifier.weight(1.4f),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                EvidenceMetric(
                    label = "Hand/object",
                    value = handObjectContactConfidence?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Collision",
                    value = interactionCollisionConfidence?.asPercent() ?: "—",
                    modifier = Modifier.weight(1f),
                )
                EvidenceMetric(
                    label = "Penetration",
                    value = maximumContactPenetrationMillimeters?.let {
                        String.format("%.2f mm", it)
                    } ?: "—",
                    modifier = Modifier.weight(1f),
                )
            }

            EvidenceMetric(
                label = "Reference uncertainty",
                value = externalReferenceUncertaintyMillimeters?.let {
                    String.format("%.2f mm", it)
                } ?: "DOCUMENTATION REQUIRED",
                modifier = Modifier.fillMaxWidth(),
            )

            val qualified =
                empiricalBenchmarkAccepted &&
                    matchedCaptureConditions &&
                    externalReferenceDocumented &&
                    measuredP95Millimeters != null &&
                    measuredP99Millimeters != null &&
                    externalReferenceUncertaintyMillimeters != null

            Text(
                text = if (qualified) {
                    "QUALIFIED: empirical benchmark evidence supplied"
                } else {
                    "BLOCKED: empirical benchmark evidence required before a 2 mm claim"
                },
                style = MaterialTheme.typography.labelLarge,
                color = if (qualified) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.error
                },
            )

            if (!qualified) {
                Text(
                    text =
                        "Modeled geometry is advisory. No infrared-superiority or sub-2 mm " +
                            "real-world claim is emitted from architecture alone.",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }
    }
}

@Composable
private fun EvidenceMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleSmall,
        )
    }
}

private fun Double.asPercent(): String =
    "${(coerceIn(0.0, 1.0) * 100.0).roundToInt()}%"
