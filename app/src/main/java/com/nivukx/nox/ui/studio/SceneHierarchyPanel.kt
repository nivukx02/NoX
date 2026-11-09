package com.nivukx.nox.ui.studio

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

data class SceneHierarchyItem(
    val id: String,
    val label: String,
    val depth: Int = 0,
    val enabled: Boolean = true,
)

@Composable
fun SceneHierarchyPanel(
    items: List<SceneHierarchyItem>,
    selectedId: String?,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassPanel(modifier) {
        Column(Modifier.fillMaxWidth().padding(10.dp)) {
            Text("Scene", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(6.dp))
            items.forEach { item ->
                Row(
                    Modifier.fillMaxWidth().clickable { onSelect(item.id) }.padding(
                        start = (8 + item.depth * 14).dp, top = 8.dp, bottom = 8.dp, end = 8.dp
                    )
                ) {
                    Text(
                        text = item.label + if (item.enabled) "" else " · hidden",
                        style = if (selectedId == item.id) MaterialTheme.typography.labelLarge
                        else MaterialTheme.typography.labelMedium,
                    )
                }
            }
        }
    }
}