package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

enum class FrontierOperationalState {
    LIVE,
    DEGRADED,
    RECOVERING,
    EVIDENCE_LIMITED,
    CALIBRATION_LIMITED,
    TIMING_LIMITED,
    THERMAL_LIMITED,
    OPTICAL_LIMITED,
    PRECISION_QUALIFIED,
}

data class FrontierOperatorState(
    val operationalState: FrontierOperationalState,
    val actorCount: Int,
    val authoritativeCameraCount: Int,
    val witnessCameraCount: Int,
    val trackingConfidence: Double,
    val reconstructionConfidence: Double,
    val timingConfidence: Double,
    val memoryHeadroom: Double,
    val thermalHeadroom: Double,
    val precisionP95Millimeters: Double,
    val observedGeometryRatio: Double,
    val inferredGeometryRatio: Double,
    val metricEvidenceAuthority: Double = 0.0,
    val opticalBlurRisk: Double = 0.0,
    val captureAuthorityScore: Double = 0.0,
    val captureAuthorityBand: String = "BLOCKED",
    val captureAuthorityHardBlocked: Boolean = true,
    val captureAuthorityPrecisionQualified: Boolean = false,
    val warnings: List<String> = emptyList(),
) {
    init {
        require(actorCount >= 0 && actorCount <= 64)
        require(authoritativeCameraCount in 0..32)
        require(witnessCameraCount in 0..96)
        listOf(trackingConfidence,reconstructionConfidence,timingConfidence,memoryHeadroom,thermalHeadroom,observedGeometryRatio,inferredGeometryRatio).forEach { require(it in 0.0..1.0) }
        require(precisionP95Millimeters.isFinite() && precisionP95Millimeters >= 0.0)
        require(metricEvidenceAuthority in 0.0..1.0)
        require(opticalBlurRisk in 0.0..1.0)
        require(captureAuthorityScore in 0.0..1.0)
        require(captureAuthorityBand.isNotBlank())
    }
}

@Composable
fun FrontierOperatorDeck(state: FrontierOperatorState, modifier: Modifier = Modifier, onStateFilter: ((FrontierOperationalState) -> Unit)? = null) {
    LiquidGlassSurface(
        modifier = modifier.widthIn(min = 320.dp, max = 760.dp),
        materialRole =
            StudioMaterialAdaptationEngine.MaterialRole.PRIMARY_INTERACTION,
        contentPadding = StudioGlassTokens.sectionPadding,
        operatorFocus = 1.0f,
        accessibilityLabel =
            "Frontier motion capture control and evidence status",
        accessibilityStateDescription =
            state.operationalState.name.replace('_', ' ') +
                "; metric evidence " +
                percent(state.metricEvidenceAuthority),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column {
                    Text("FRONTIER CONTROL", style = MaterialTheme.typography.labelLarge)
                    Text(state.operationalState.name.replace('_', ' '), style = MaterialTheme.typography.titleMedium)
                }
                FilterChip(selected = state.operationalState == FrontierOperationalState.PRECISION_QUALIFIED, onClick = { onStateFilter?.invoke(state.operationalState) }, label = { Text("Authority") })
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FrontierMetric("Actors", state.actorCount.toString(), Modifier.weight(1f))
                FrontierMetric("Cameras", "${state.authoritativeCameraCount}/${state.witnessCameraCount}", Modifier.weight(1f))
                FrontierMetric("Tracking", percent(state.trackingConfidence), Modifier.weight(1f))
                FrontierMetric("3D", percent(state.reconstructionConfidence), Modifier.weight(1f))
            }
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FrontierMetric(
                    "Authority",
                    percent(state.captureAuthorityScore) +
                        " · " +
                        state.captureAuthorityBand.replace('_', ' '),
                    Modifier.weight(1f),
                )
                FrontierMetric(
                    "Evidence",
                    percent(state.metricEvidenceAuthority),
                    Modifier.weight(1f),
                )
                FrontierMetric(
                    "Timing",
                    percent(state.timingConfidence),
                    Modifier.weight(1f),
                )
                FrontierMetric(
                    "Blur",
                    percent(state.opticalBlurRisk),
                    Modifier.weight(1f),
                )
                FrontierMetric("Memory", percent(state.memoryHeadroom), Modifier.weight(1f))
                FrontierMetric("Thermal", percent(state.thermalHeadroom), Modifier.weight(1f))
                FrontierMetric("P95", "%.3f mm".format(state.precisionP95Millimeters), Modifier.weight(1f))
            }
            Text("Observed ${percent(state.observedGeometryRatio)} · Inferred ${percent(state.inferredGeometryRatio)}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (state.warnings.isNotEmpty()) Text(state.warnings.take(4).joinToString(" • "), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        }
    }
}

@Composable
private fun FrontierMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier.semantics {
            contentDescription = "$label: $value"
        },
        shape = RoundedCornerShape(14.dp),
        tonalElevation = 1.dp,
    ) {
        Column(
            modifier = Modifier.padding(
                horizontal = 10.dp,
                vertical = 8.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
            )
            Text(
                value,
                style = MaterialTheme.typography.titleSmall,
            )
        }
    }
}
private fun percent(value: Double): String = "${(value * 100.0).toInt()}%"