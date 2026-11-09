package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.ActorDetailPriorityMode
import com.nivukx.nox.core.model.MotionCaptureCustomization

/**
 * Crowded-scene detail allocation controls.
 *
 * The operator controls how many actors may receive each expensive modality. The canonical body
 * identity graph always retains admitted actors; these budgets only influence optional detail work.
 */
@Composable
fun ActorDetailBudgetPanel(
    settings: MotionCaptureCustomization,
    onSettingsChanged: (MotionCaptureCustomization) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = false,
        elevated = true,
        modifier = modifier.fillMaxWidth(),
        contentPadding = 14.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            Text(
                text = "Crowd Detail Allocation",
                style = MaterialTheme.typography.titleMedium,
            )

            Text(
                text =
                    "All body identities stay active. These budgets control how many actors receive the most expensive face, eye, hand and dense-surface processing.",
                style = MaterialTheme.typography.bodySmall,
            )

            ActorDetailModeRow(
                mode = settings.actorDetailPriorityMode,
                onModeChanged = {
                    onSettingsChanged(settings.copyWith {
                        actorDetailPriorityMode = it
                    })
                },
            )

            BudgetSlider("Full-detail actors", settings.fullDetailActorBudget, settings.maxSubjects) { value ->
                onSettingsChanged(settings.copyWith {
                    fullDetailActorBudget = value
                })
            }
            BudgetSlider("Face-detail actors", settings.faceDetailActorBudget, settings.maxSubjects) { value ->
                onSettingsChanged(settings.copyWith {
                    faceDetailActorBudget = value
                })
            }
            BudgetSlider("Eye-detail actors", settings.eyeDetailActorBudget, settings.maxSubjects) { value ->
                onSettingsChanged(settings.copyWith {
                    eyeDetailActorBudget = value
                })
            }
            BudgetSlider("Hand-detail actors", settings.handDetailActorBudget, settings.maxSubjects) { value ->
                onSettingsChanged(settings.copyWith {
                    handDetailActorBudget = value
                })
            }
            BudgetSlider("Dense-surface actors", settings.denseSurfaceActorBudget, settings.maxSubjects) { value ->
                onSettingsChanged(settings.copyWith {
                    denseSurfaceActorBudget = value
                })
            }

            Text(
                text = "Priority mode: ${settings.actorDetailPriorityMode.name.replace('_', ' ')}",
                style = MaterialTheme.typography.labelMedium,
            )
        }
    }
}

@Composable
private fun ActorDetailModeRow(
    mode: ActorDetailPriorityMode,
    onModeChanged: (ActorDetailPriorityMode) -> Unit,
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
    ) {
        ActorDetailPriorityMode.entries.forEach { entry ->
            StudioChoiceChip(
                selected = entry == mode,
                text = entry.name.replace('_', ' '),
                onClick = { onModeChanged(entry) },
            )
        }
    }
}

@Composable
private fun BudgetSlider(
    label: String,
    value: Int,
    maxValue: Int,
    onValueChanged: (Int) -> Unit,
) {
    val safeMax = maxValue.coerceIn(1, 64)
    val safeValue = value.coerceIn(1, safeMax)

    Column(
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
            )
            Text(
                text = "$safeValue / $safeMax",
                style = MaterialTheme.typography.labelMedium,
            )
        }

        Slider(
            value = safeValue.toFloat(),
            onValueChange = { onValueChanged(it.toInt().coerceIn(1, safeMax)) },
            valueRange = 1f..safeMax.toFloat(),
            steps = (safeMax - 2).coerceAtLeast(0),
        )
    }
}
