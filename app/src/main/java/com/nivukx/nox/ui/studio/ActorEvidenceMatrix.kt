package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

data class FrontierActorEvidenceRow(val actorId: Int, val confidence: Double, val uncertainty: Double, val bodyEveryFrames: Int, val faceEveryFrames: Int, val eyeEveryFrames: Int, val handEveryFrames: Int, val surfaceEveryFrames: Int, val occluded: Boolean, val interactionRisk: Double)

@Composable fun ActorEvidenceMatrix(rows: List<FrontierActorEvidenceRow>, modifier: Modifier = Modifier) {
    LiquidGlassSurface(modifier = modifier, contentPadding = 12.dp) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Actor evidence matrix", style = MaterialTheme.typography.titleMedium)
            Text("50-actor target · 64-actor hard ceiling", style = MaterialTheme.typography.bodySmall)
            LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                items(rows.sortedBy { it.actorId }.take(64), key = { it.actorId }) { row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("A${row.actorId}", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(0.7f))
                        Text("${(row.confidence * 100).toInt()}%", modifier = Modifier.weight(0.8f))
                        Text("U ${(row.uncertainty * 100).toInt()}%", modifier = Modifier.weight(0.9f))
                        Text("B${row.bodyEveryFrames}", modifier = Modifier.weight(0.5f))
                        Text("F${row.faceEveryFrames}", modifier = Modifier.weight(0.5f))
                        Text("E${row.eyeEveryFrames}", modifier = Modifier.weight(0.5f))
                        Text("H${row.handEveryFrames}", modifier = Modifier.weight(0.5f))
                        Text(if (row.occluded) "OCC" else "CLEAR", color = if (row.occluded) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}