package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

@Composable
fun PrecisionFabricPanel(
    telemetry: StudioTelemetryState,
    modifier: Modifier = Modifier,
) {
    val denseTrust =
        telemetry.denseSurfaceTrustedPointRatio
            ?: telemetry.denseLandmarkCoverage
            ?: 0.0
    val uncertainty =
        telemetry.denseSurfaceUncertaintyP95Meters
            ?: telemetry.reconstructionUncertaintyP95Meters
            ?: Double.POSITIVE_INFINITY
    val synchronization =
        telemetry.synchronizationResidualP95Milliseconds
            ?: telemetry.temporalLagMilliseconds
            ?: Double.POSITIVE_INFINITY
    val geometry =
        telemetry.geometricInformationConditionScore
            ?: 0.0
    val contact =
        telemetry.denseContactConfidence
            ?: telemetry.denseLandmarkContactCoverage
            ?: 0.0
    val temporalInnovation =
        telemetry.denseTemporalInnovationP95 ?: 0.0
    val viewDiversity =
        telemetry.denseViewDiversity ?: 0.0
    val geometryConfidence =
        telemetry.denseGeometricConfidence ?: 0.0

    val motionDirection =
        telemetry.nativeMotionDirectionConsistency
            ?: 0.0
    val microMotion =
        telemetry.nativeMotionMicroMotionRetention
            ?: 0.0
    val dynamicStability =
        telemetry.nativeMotionDynamicStability
            ?: 0.0
    val motionInnovation =
        telemetry.nativeMotionInnovationP95Pixels
            ?: 0.0

    val gaussian4DConsistency =
        telemetry.global4DGaussianConsistency
            ?: 1.0
    val gaussian4DStrainConsistency =
        telemetry.global4DGaussianStrainConsistency
            ?: 1.0
    val gaussian4DEvidenceAvailable =
        telemetry.global4DGaussianEvidenceAvailable
            ?: false

    val measuredBand =
        denseTrust >= 0.90 &&
            geometry >= 0.75 &&
            geometryConfidence >= 0.70 &&
            viewDiversity >= 0.30 &&
            contact >= 0.60 &&
            temporalInnovation <= 0.55 &&
            uncertainty.isFinite() &&
            uncertainty <= 0.001 &&
            synchronization.isFinite() &&
            synchronization <= 2.0 &&
            motionDirection >= 0.80 &&
            microMotion >= 0.75 &&
            dynamicStability >= 0.80 &&
            (
                !gaussian4DEvidenceAvailable ||
                    (
                        gaussian4DConsistency >= 0.85 &&
                            gaussian4DStrainConsistency >= 0.85
                    )
            )

    val state =
        when {
            measuredBand -> "MEASURED PRECISION BAND"
            denseTrust >= 0.70 && geometry >= 0.55 ->
                "HIGH-END / EVIDENCE LIMITED"
            else -> "EVIDENCE INSUFFICIENT"
        }

    LiquidGlassSurface(
        modifier =
            modifier
                .fillMaxWidth()
                .semantics {
                    contentDescription = "Precision Fabric. $state."
                },
        clear = true,
        elevated = true,
        contentPadding = 16.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(
                    Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        text = "Precision Fabric",
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        text =
                            "Dense geometry, temporal consistency and metrology evidence",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    text = state,
                    style = MaterialTheme.typography.labelMedium,
                    color =
                        if (measuredBand) {
                            Color(0xFF8EF0C8)
                        } else {
                            MaterialTheme.colorScheme.primary
                        },
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PrecisionMetric(
                    label = "Dense Trust",
                    value =
                        (denseTrust.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Geometry",
                    value =
                        (geometry.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Contact",
                    value =
                        (contact.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Geometry Trust",
                    value =
                        (geometryConfidence.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PrecisionMetric(
                    label = "Rig",
                    value =
                        if (
                            telemetry.camerasConnected != null &&
                            telemetry.camerasExpected != null
                        ) {
                            "${telemetry.camerasConnected}/${telemetry.camerasExpected}"
                        } else {
                            "N/A"
                        },
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Calibration",
                    value =
                        telemetry.calibrationImageCoverage?.let {
                            String.format("%.0f%%", it * 100.0)
                        } ?: "N/A",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Sync P95",
                    value =
                        telemetry.synchronizationResidualP95Milliseconds?.let {
                            String.format("%.2f ms", it)
                        } ?: "N/A",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Drift",
                    value =
                        telemetry.calibrationDriftStabilityScore?.let {
                            String.format("%.0f%%", it * 100.0)
                        } ?: "N/A",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PrecisionMetric(
                    label = "Motion Trust",
                    value =
                        (
                            (
                                motionDirection * 0.36 +
                                    microMotion * 0.24 +
                                    dynamicStability * 0.40
                            ).coerceIn(0.0, 1.0) * 100.0
                        ).toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "P95 Innovation",
                    value =
                        "%.2f px".format(
                            motionInnovation.coerceAtLeast(0.0),
                        ),
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "View Diversity",
                    value =
                        (viewDiversity.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Temporal P95",
                    value =
                        (temporalInnovation.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                PrecisionMetric(
                    label = "4D Gaussian",
                    value =
                        (
                            gaussian4DConsistency.coerceIn(0.0, 1.0) *
                                100.0
                        ).toInt().toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Deformation",
                    value =
                        (
                            gaussian4DStrainConsistency.coerceIn(0.0, 1.0) *
                                100.0
                        ).toInt().toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Gaussian Trust",
                    value =
                        (
                            (telemetry.denseSurfaceTrustedGaussianRatio ?: 0.0)
                                .coerceIn(0.0, 1.0) * 100.0
                        ).toInt().toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Dynamic",
                    value =
                        (
                            (telemetry.denseSurfaceMeanDynamicProbability ?: 0.0)
                                .coerceIn(0.0, 1.0) * 100.0
                        ).toInt().toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Surface Temporal",
                    value =
                        (
                            (telemetry.denseSurfaceMeanTemporalStability ?: 0.0)
                                .coerceIn(0.0, 1.0) * 100.0
                        ).toInt().toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "P95 Uncertainty",
                    value =
                        if (uncertainty.isFinite()) {
                            String.format(
                                "%.2f mm",
                                uncertainty * 1000.0,
                            )
                        } else {
                            "N/A"
                        },
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Dynamic Stability",
                    value =
                        (dynamicStability.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() + "%",
                    modifier = Modifier.weight(1f),
                )
                PrecisionMetric(
                    label = "Sync P95",
                    value =
                        if (synchronization.isFinite()) {
                            String.format(
                                "%.2f ms",
                                synchronization,
                            )
                        } else {
                            "N/A"
                        },
                    modifier = Modifier.weight(1f),
                )
                Spacer(Modifier.weight(1f))
            }

            Surface(
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.40f),
                shape = MaterialTheme.shapes.medium,
            ) {
                Text(
                    modifier = Modifier.padding(
                        horizontal = 12.dp,
                        vertical = 10.dp,
                    ),
                    text =
                        if (measuredBand) {
                            "Traceable precision gate satisfied by current evidence. Output remains governed by measurement confidence, not display confidence."
                        } else {
                            "Sub-millimetre output is intentionally unasserted until calibration, synchronization, geometry and uncertainty evidence agree."
                        },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun PrecisionMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.52f),
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(
            Modifier.padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleSmall,
            )
        }
    }
}
