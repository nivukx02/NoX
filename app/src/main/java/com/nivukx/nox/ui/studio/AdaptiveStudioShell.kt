package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun AdaptiveStudioShell(
    largeLayout: Boolean,
    selected: StudioSection,
    onSelected: (StudioSection) -> Unit,
    inspector: @Composable () -> Unit,
    content: @Composable () -> Unit,
) {
    Row(Modifier.fillMaxSize()) {
        LiquidGlassSurface(
            Modifier.width(if (largeLayout) 88.dp else 72.dp),
            clear = true,
        ) {
            AdaptiveStudioNavigation(
                selected = selected,
                onSelected = onSelected,
                largeLayout = largeLayout,
            )
        }
        Box(Modifier.weight(1f).fillMaxSize()) {
            content()
        }
        if (largeLayout) {
            LiquidGlassSurface(
                Modifier.width(320.dp),
                clear = false,
            ) {
                inspector()
            }
        }
    }
}
