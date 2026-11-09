package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.ReconstructionRenderSettings
import com.nivukx.nox.core.model.ReconstructionVisualizationMode

@Composable
fun ReconstructionInspector(
    settings: ReconstructionRenderSettings,
    onSettingsChange: (ReconstructionRenderSettings) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassPanel(modifier) {
        Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Reconstruction", style = MaterialTheme.typography.titleMedium)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                ReconstructionVisualizationMode.entries.take(3).forEach { mode ->
                    FilterChip(
                        selected = settings.mode == mode,
                        onClick = { onSettingsChange(settings.copy(mode = mode)) },
                        label = { Text(mode.name.replace('_', ' ')) },
                    )
                }
            }
            Text("Mesh opacity " + (settings.meshOpacity * 100).toInt() + "%")
            Slider(value = settings.meshOpacity.toFloat(), onValueChange = { onSettingsChange(settings.copy(meshOpacity = it.toDouble())) }, valueRange = 0.05f..1f)
            Text("Skeleton " + settings.skeletonLineWidthPx + " px")
            Slider(value = settings.skeletonLineWidthPx.toFloat(), onValueChange = { onSettingsChange(settings.copy(skeletonLineWidthPx = it.toDouble())) }, valueRange = 0.5f..8f)
            Text(
                text = "Uncertainty " + if (settings.showUncertainty) "visible" else "hidden",
                style = MaterialTheme.typography.labelLarge,
            )
        }
    }
}