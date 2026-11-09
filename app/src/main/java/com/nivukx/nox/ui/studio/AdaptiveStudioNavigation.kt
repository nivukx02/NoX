package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.PermanentNavigationDrawer
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

@Composable
fun AdaptiveStudioNavigation(
    selected: StudioSection,
    onSelected: (StudioSection) -> Unit,
    largeLayout: Boolean,
) {
    if (largeLayout) {
        PermanentNavigationDrawer(
            drawerContent = {
                NavigationRail(
                    modifier = Modifier.fillMaxHeight(),
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                ) {
                    StudioSection.entries.forEach { section ->
                        NavigationRailItem(
                            selected = section == selected,
                            onClick = { onSelected(section) },
                            label = { Text(section.name.replace('_', ' ')) },
                            icon = {},
                        )
                    }
                }
            },
        ) {}
    } else {
        NavigationRail {
            StudioSection.entries.take(7).forEach { section ->
                NavigationRailItem(
                    selected = section == selected,
                    onClick = { onSelected(section) },
                    label = { Text(section.name.replace('_', ' ')) },
                    icon = {},
                )
            }
        }
    }
}