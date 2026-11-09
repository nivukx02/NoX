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

data class FrontierCameraInspectorRow(val cameraId: String, val sector: Int, val authority: String, val quality: Double, val synchronization: Double, val calibration: Double, val coverage: Double, val quarantined: Boolean)

@Composable fun CameraFederationInspector(rows: List<FrontierCameraInspectorRow>, modifier: Modifier = Modifier) {
    LiquidGlassSurface(modifier = modifier, contentPadding = 12.dp) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Camera federation", style = MaterialTheme.typography.titleMedium)
            Text("${rows.count { it.authority == "AUTHORITATIVE" }} authoritative · ${rows.count { it.authority == "WITNESS" }} witness · ${rows.count { it.quarantined }} quarantined", style = MaterialTheme.typography.bodySmall)
            LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                items(rows.take(128), key = { it.cameraId }) { row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(row.cameraId, style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
                        Text("S${row.sector}", style = MaterialTheme.typography.labelSmall)
                        Text(row.authority, style = MaterialTheme.typography.labelSmall)
                        Text("${(row.quality * 100).toInt()}%", color = if (row.quarantined) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}