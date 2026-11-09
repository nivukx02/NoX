package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.reconstruction.PrecisionEvidenceFieldEngine
import com.nivukx.nox.ui.SectionTitle
import java.util.Locale

/**
 * Operator-facing evidence field for dense markerless reconstruction.
 *
 * Glass is retained as a functional control material. Dense telemetry stays on readable surfaces.
 */
@Composable
fun PrecisionEvidenceSurface(
    modifier: Modifier = Modifier,
    calibrationReady: Boolean,
    synchronized: Boolean,
    syncConfidence: Double,
    reprojectionP95Pixels: Double,
    reconstructionConfidence: Double,
    reconstructionUncertaintyP95Meters: Double,
    geometricObservability: Double,
    denseGeometricConditioningScore: Double = 0.0,
    denseGeometricConditioningMinimumObservationScore: Double = 1.0,
    denseGeometricConditioningMedianParallaxDegrees: Double = 0.0,
    denseGeometricConditioningWarnings: List<String> = emptyList(),
    denseLandmarkCoverage: Double,
    denseLandmarkCrossViewAgreement: Double,
    temporalStability: Double,
    worldConsistency: Double,
    identityContinuity: Double,
    identitySwapRisk: Double,
    faceEyeConfidence: Double,
    handConfidence: Double,
    cameraCount: Int,
    predictedObservationRatio: Double,
    denseSurfaceTrustedPointRatio: Double,
    denseSurfaceTopologyStability: Double,
    denseSurfaceViewDiversity: Double,
    captureIntegrity: Double,
    sessionHealth: Double,
    selectedActorCount: Int,
    targetActorCount: Int = 50,
) {
    val runtimeHeadroom =
        minOf(
            captureIntegrity.coerceIn(0.0, 1.0),
            sessionHealth.coerceIn(0.0, 1.0),
        )

    val predictedRatio =
        predictedObservationRatio.coerceIn(0.0, 1.0)

    val precisionEngine =
        remember { PrecisionEvidenceFieldEngine() }

    val effectiveSyncConfidence =
        if (synchronized) {
            syncConfidence.coerceIn(0.0, 1.0)
        } else {
            minOf(
                syncConfidence.coerceIn(0.0, 1.0),
                0.35,
            )
        }

    val precision =
        precisionEngine.evaluate(
            PrecisionEvidenceFieldEngine.Input(
                timestampNanos = 0L,
                cameraCount = cameraCount,
                calibrationConfidence =
                    if (calibrationReady) 1.0 else 0.0,
                reprojectionP95Pixels =
                    reprojectionP95Pixels.takeIf(Double::isFinite)
                        ?: Double.POSITIVE_INFINITY,
                synchronizationConfidence =
                    effectiveSyncConfidence,
                synchronizationJitterMillis = 0.0,
                crossViewEvidence =
                    denseLandmarkCrossViewAgreement
                        .coerceIn(0.0, 1.0),
                viewDiversity =
                    denseSurfaceViewDiversity
                        .coerceIn(0.0, 1.0),
                geometricObservability =
                    geometricObservability
                        .coerceIn(0.0, 1.0),
                geometricConditionScore =
                    listOf(
                        denseGeometricConditioningScore
                            .coerceIn(0.0, 1.0),
                        denseGeometricConditioningMinimumObservationScore
                            .coerceIn(0.0, 1.0),
                        denseSurfaceTopologyStability
                            .coerceIn(0.0, 1.0),
                    ).minOrNull() ?: 0.0,
                directionalMajor95Meters =
                    reconstructionUncertaintyP95Meters
                        .takeIf(Double::isFinite)
                        ?.coerceAtLeast(0.0)
                        ?: Double.POSITIVE_INFINITY,
                directionalDepth95Meters =
                    reconstructionUncertaintyP95Meters
                        .takeIf(Double::isFinite)
                        ?.coerceAtLeast(0.0)
                        ?: Double.POSITIVE_INFINITY,
                reconstructionUncertaintyP95Meters =
                    reconstructionUncertaintyP95Meters
                        .takeIf(Double::isFinite)
                        ?.coerceAtLeast(0.0)
                        ?: Double.POSITIVE_INFINITY,
                denseSurfaceUncertaintyP95Meters =
                    reconstructionUncertaintyP95Meters
                        .takeIf(Double::isFinite)
                        ?.coerceAtLeast(0.0)
                        ?: Double.POSITIVE_INFINITY,
                temporalStability =
                    temporalStability.coerceIn(0.0, 1.0),
                worldConsistency =
                    worldConsistency.coerceIn(0.0, 1.0),
                identityContinuity =
                    identityContinuity.coerceIn(0.0, 1.0),
                identitySwapRisk =
                    identitySwapRisk.coerceIn(0.0, 1.0),
                bodyConfidence =
                    reconstructionConfidence.coerceIn(0.0, 1.0),
                faceEyeConfidence =
                    faceEyeConfidence.coerceIn(0.0, 1.0),
                handConfidence =
                    handConfidence.coerceIn(0.0, 1.0),
                thermalHeadroom =
                    captureIntegrity.coerceIn(0.0, 1.0),
                computeHeadroom =
                    sessionHealth.coerceIn(0.0, 1.0),
                predictedObservationRatio = predictedRatio,
            ),
        )

    val geometry = precision.scoreForDisplay(
        precision.geometryDisplayScore(
            denseLandmarkCoverage = denseLandmarkCoverage,
            denseLandmarkCrossViewAgreement =
                denseLandmarkCrossViewAgreement,
            geometricObservability =
                geometricObservability,
        ),
    )

    val time = precision.timeDisplayScore(
        syncConfidence = effectiveSyncConfidence,
        temporalStability = temporalStability,
        worldConsistency = worldConsistency,
    )

    val identity = precision.identityDisplayScore(
        identityContinuity = identityContinuity,
        identitySwapRisk = identitySwapRisk,
        predictedObservationRatio = predictedRatio,
    )

    val surface = denseSurfaceTrustedPointRatio
        .coerceIn(0.0, 1.0)
        .coerceAtMost(denseSurfaceTopologyStability.coerceIn(0.0, 1.0))
        .coerceAtMost(denseSurfaceViewDiversity.coerceIn(0.0, 1.0))

    val runtime = runtimeHeadroom

    val overall = precision.score

    val uncertaintyMillimeters =
        precision.modeledP95Millimeters
            .takeIf(Double::isFinite)
            ?.coerceAtLeast(0.0)

    val state = when (precision.band) {
        PrecisionEvidenceFieldEngine.Band.BLOCKED ->
            precision.bottleneck.name.replace('_', ' ')
        PrecisionEvidenceFieldEngine.Band.CONDITIONAL ->
            "CONDITIONAL"
        PrecisionEvidenceFieldEngine.Band.OPERATIONAL ->
            "OPERATIONAL"
        PrecisionEvidenceFieldEngine.Band.REFERENCE_CANDIDATE ->
            "REFERENCE CANDIDATE"
        PrecisionEvidenceFieldEngine.Band.EMPIRICALLY_QUALIFIED ->
            "EMPIRICALLY QUALIFIED"
    }
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(24.dp),
        tonalElevation = 2.dp,
        shadowElevation = 8.dp,
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    SectionTitle("Precision Evidence Field")
                    Text(
                        text = "Independent geometry, timing, identity, surface and runtime evidence.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Surface(
                    shape = RoundedCornerShape(999.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerHighest,
                ) {
                    Text(
                        text =
                            state +
                                "  " +
                                (overall * 100.0).toInt() +
                                "%",
                        modifier = Modifier.padding(
                            horizontal = 12.dp,
                            vertical = 7.dp,
                        ),
                        style = MaterialTheme.typography.labelMedium,
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                EvidenceCell("GEOMETRY", geometry)
                EvidenceCell("TIME", time)
                EvidenceCell("IDENTITY", identity)
                EvidenceCell("SURFACE", surface)
                EvidenceCell("RUNTIME", runtime)
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                EvidenceValue(
                    title = "MODELED P95",
                    value =
                        uncertaintyMillimeters?.let {
                            String.format(
                                Locale.US,
                                "%.3f mm",
                                it,
                            )
                        } ?: "UNAVAILABLE",
                )
                EvidenceValue(
                    title = "GEOMETRIC CONDITION",
                    value =
                        (denseGeometricConditioningScore
                            .coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() +
                            "%  ·  " +
                            String.format(
                                Locale.US,
                                "%.1f°",
                                denseGeometricConditioningMedianParallaxDegrees
                                    .coerceAtLeast(0.0),
                            ),
                )
                EvidenceValue(
                    title = "CROSS-VIEW",
                    value =
                        (denseLandmarkCrossViewAgreement
                            .coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() +
                            "%",
                )
                EvidenceValue(
                    title = "ACTORS",
                    value =
                        selectedActorCount.toString() +
                            " / " +
                            targetActorCount.toString(),
                )
                EvidenceValue(
                    title = "FACE / EYE",
                    value =
                        (faceEyeConfidence.coerceIn(0.0, 1.0) * 100.0)
                            .toInt()
                            .toString() +
                            "%",
                )
            }

            Text(
                text =
                    when {
                        state == "REFERENCE CANDIDATE" ->
                            "Model-derived evidence is inside the configured candidate gate. External matched-reference qualification remains separate."
                        cameraCount < 2 ->
                            "Single-view motion remains useful, but metric depth is under-observed without calibrated multi-view geometry."
                        geometry < 0.65 -> {
                            val warning =
                                denseGeometricConditioningWarnings
                                    .firstOrNull()
                                    ?.let { "  [$it]" }
                                    .orEmpty()
                            "Increase geometric diversity or recover weak cameras before trusting dense 3D surface detail." +
                                warning
                        }
                        time < 0.80 ->
                            "Temporal alignment is currently a dominant risk. Preserve sensor timestamps and reduce clock residual."
                        identity < 0.65 ->
                            "Identity continuity is degraded. Preserve longer appearance and motion history."
                        uncertaintyMillimeters != null &&
                            uncertaintyMillimeters > 1.0 ->
                            "The modeled P95 uncertainty exceeds the configured 1 mm target."
                        else ->
                            "Live reconstruction is operational. Precision is shown as evidence, not as an unmeasured physical claim."
                    },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private fun PrecisionEvidenceFieldEngine.Result.scoreForDisplay(
    value: Double,
): Double = value.coerceIn(0.0, 1.0)

private fun PrecisionEvidenceFieldEngine.Result.geometryDisplayScore(
    denseLandmarkCoverage: Double,
    denseLandmarkCrossViewAgreement: Double,
    geometricObservability: Double,
): Double =
    listOf(
        denseLandmarkCoverage.coerceIn(0.0, 1.0),
        denseLandmarkCrossViewAgreement.coerceIn(0.0, 1.0),
        geometricObservability.coerceIn(0.0, 1.0),
    ).minOrNull() ?: 0.0

private fun PrecisionEvidenceFieldEngine.Result.timeDisplayScore(
    syncConfidence: Double,
    temporalStability: Double,
    worldConsistency: Double,
): Double =
    listOf(
        syncConfidence.coerceIn(0.0, 1.0),
        temporalStability.coerceIn(0.0, 1.0),
        worldConsistency.coerceIn(0.0, 1.0),
    ).minOrNull() ?: 0.0

private fun PrecisionEvidenceFieldEngine.Result.identityDisplayScore(
    identityContinuity: Double,
    identitySwapRisk: Double,
    predictedObservationRatio: Double,
): Double =
    listOf(
        identityContinuity.coerceIn(0.0, 1.0),
        1.0 - identitySwapRisk.coerceIn(0.0, 1.0),
        1.0 - predictedObservationRatio.coerceIn(0.0, 1.0),
    ).minOrNull() ?: 0.0

@Composable
private fun EvidenceCell(
    title: String,
    score: Double,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(8.dp),
            color = MaterialTheme.colorScheme.surfaceContainerLowest,
        ) {
            Text(
                text =
                    (score.coerceIn(0.0, 1.0) * 100.0)
                        .toInt()
                        .toString() +
                        "%",
                modifier = Modifier.padding(
                    horizontal = 8.dp,
                    vertical = 6.dp,
                ),
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun EvidenceValue(
    title: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceContainerLowest,
    ) {
        Column(
            modifier = Modifier.padding(11.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = title,
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
