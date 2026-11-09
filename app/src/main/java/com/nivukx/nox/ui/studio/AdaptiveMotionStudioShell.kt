package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.material3.adaptive.currentWindowAdaptiveInfoV2
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteScaffold
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteScaffoldDefaults

@Composable
fun AdaptiveMotionStudioShell(
    navigationState: StudioNavigationState,
    onNavigationChange: (StudioNavigationState) -> Unit,
    content: @Composable () -> Unit,
    inspector: (@Composable () -> Unit)? = null,
) {
    val adaptiveInfo = currentWindowAdaptiveInfoV2()
    val destinations = MotionStudioDestination.entries

    NavigationSuiteScaffold(
        navigationSuiteItems = {
            destinations.forEach { destination ->
                item(
                    selected = navigationState.selected == destination,
                    onClick = {
                        onNavigationChange(
                            StudioNavigationReducer.select(navigationState, destination),
                        )
                    },
                    icon = {
                        Text(
                            destination.shortLabel.take(1),
                            modifier = Modifier.semantics {
                                contentDescription =
                                    destination.contentDescription
                            },
                        )
                    },
                    label = { Text(destination.label) },
                )
            }
        },
        layoutType =
            NavigationSuiteScaffoldDefaults.navigationSuiteType(
                adaptiveInfo,
            ),
    ) {
        Row(Modifier.fillMaxSize().padding(StudioGlassTokens.contentPadding)) {
            Box(Modifier.weight(1f)) { content() }
            if (navigationState.inspectorVisible && inspector != null) {
                LiquidGlassPanel(
                    Modifier
                        .padding(start = 12.dp)
                        .widthIn(min = 280.dp, max = StudioGlassTokens.inspectorWidth + 120.dp),
                ) {
                    inspector()
                }
            }
        }
    }
}
