package com.nivukx.nox.ui.studio

import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.CaptureQualityTier
import com.nivukx.nox.core.model.ProductionCaptureProfile

/**
 * Compact operator control for switching complete capture policies without opening advanced
 * settings. Every profile maps to an existing production policy, so the control never invents
 * a second set of hidden tuning values.
 */
@Composable
fun CaptureProfileQuickControls(
    selectedProfileId: String,
    onProfileSelected: (ProductionCaptureProfile) -> Unit,
    modifier: Modifier = Modifier,
    profiles: List<ProductionCaptureProfile> = ProductionCaptureProfile.all(),
) {
    Row(
        modifier
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        profiles.forEach { profile ->
            val selected = profile.id == selectedProfileId
            Surface(
                shape = RoundedCornerShape(16.dp),
                tonalElevation = if (selected) 4.dp else 1.dp,
                shadowElevation = if (selected) 2.dp else 0.dp,
                modifier = Modifier.clickable {
                    onProfileSelected(profile)
                },
            ) {
                Column(
                    Modifier.padding(
                        horizontal = 13.dp,
                        vertical = 9.dp,
                    ),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        text = profile.label,
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        text = profile.qualityTier.displayLabel(),
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }
    }
}

private fun CaptureQualityTier.displayLabel(): String =
    when (this) {
        CaptureQualityTier.BALANCED -> "Balanced"
        CaptureQualityTier.HIGH -> "High"
        CaptureQualityTier.ULTRA -> "Ultra"
        CaptureQualityTier.REFERENCE -> "Reference"
    }
