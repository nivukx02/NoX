package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Operator-facing state surface for the deferred solve.
 *
 * The panel reports process state and measured frame/actor counters rather than using a fake
 * percentage derived from time. Progress is based on actual processed/decode coverage.
 */
@Composable
fun PostCaptureReconstructionStatusPanel(
    state: String,
    progress: Double,
    processedFrames: Long,
    maxActors: Int,
    modifier: Modifier = Modifier,
) {
    if (state == "IDLE" || state == "NOT_AVAILABLE" || state == "DISABLED") {
        return
    }

    LiquidGlassSurface(
        clear = false,
        elevated = true,
        modifier = modifier.fillMaxWidth(),
        contentPadding = 14.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = "Post-Capture Reconstruction",
                    style = MaterialTheme.typography.titleSmall,
                )
                Text(
                    text = state.replace('_', ' '),
                    style = MaterialTheme.typography.labelLarge,
                )
            }

            LinearProgressIndicator(
                progress = { progress.coerceIn(0.0, 1.0).toFloat() },
                modifier = Modifier.fillMaxWidth(),
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = "Frames $processedFrames",
                    style = MaterialTheme.typography.labelMedium,
                )
                Text(
                    text = "Peak actors $maxActors",
                    style = MaterialTheme.typography.labelMedium,
                )
                Text(
                    text = "${(progress.coerceIn(0.0, 1.0) * 100.0).toInt()}%",
                    style = MaterialTheme.typography.labelMedium,
                )
            }

            Text(
                text =
                    when (state) {
                        "QUEUED" ->
                            "Visual evidence is retained. The heavy reconstruction pass is queued."
                        "PROCESSING" ->
                            "Recorded visual evidence is being re-entered through the multimodal perception graph."
                        "COMPLETE" ->
                            "Deferred perception pass completed with the reported frame coverage."
                        "DEGRADED" ->
                            "The pass completed with reduced evidence coverage or warnings."
                        "CANCELLED" ->
                            "Deferred processing was cancelled before full evidence coverage."
                        "FAILED" ->
                            "Deferred processing failed. The captured evidence remains available for diagnosis."
                        else ->
                            "Deferred reconstruction state is available in the session diagnostics."
                    },
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}
