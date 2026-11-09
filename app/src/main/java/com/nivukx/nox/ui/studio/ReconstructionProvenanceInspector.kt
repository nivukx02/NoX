package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

data class FrontierReconstructionProvenanceView(val actorId: Int, val regionId: Int, val observedRatio: Double, val constrainedRatio: Double, val propagatedRatio: Double, val learnedRatio: Double, val generatedRatio: Double, val confidence: Double)

@Composable fun ReconstructionProvenanceInspector(value: FrontierReconstructionProvenanceView, modifier: Modifier = Modifier) {
    LiquidGlassSurface(modifier = modifier, contentPadding = 12.dp) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Reconstruction provenance", style = MaterialTheme.typography.titleMedium)
            Text("Actor ${value.actorId} · Region ${value.regionId}", style = MaterialTheme.typography.bodySmall)
            FrontierProvenanceRow("Observed", value.observedRatio)
            FrontierProvenanceRow("Constrained", value.constrainedRatio)
            FrontierProvenanceRow("Propagated", value.propagatedRatio)
            FrontierProvenanceRow("Learned prior", value.learnedRatio)
            FrontierProvenanceRow("Generated", value.generatedRatio)
            Text("Confidence ${(value.confidence * 100).toInt()}%", style = MaterialTheme.typography.labelMedium)
        }
    }
}

@Composable private fun FrontierProvenanceRow(label: String, value: Double) {
    Row(Modifier, horizontalArrangement = Arrangement.SpaceBetween) { Text(label, style = MaterialTheme.typography.labelMedium); Text("${(value.coerceIn(0.0,1.0) * 100.0).toInt()}%", style = MaterialTheme.typography.labelSmall) }
}