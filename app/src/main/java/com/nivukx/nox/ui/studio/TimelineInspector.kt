package com.nivukx.nox.ui.studio

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

data class TimelineInspectorState(
    val playheadSeconds: Double,
    val durationSeconds: Double,
    val loopStartSeconds: Double,
    val loopEndSeconds: Double,
    val frameRate: Double = 60.0,
    val precisionQualified: Boolean = false,
    val precisionP95Millimeters: Double? = null,
)

@Composable
fun TimelineInspector(
    state: TimelineInspectorState,
    onSeek: (Double) -> Unit,
    modifier: Modifier = Modifier,
) {
    val safeDuration =
        state.durationSeconds.coerceAtLeast(0.001)
    val safeFps =
        state.frameRate.coerceIn(1.0, 240.0)

    var playhead by remember(state.playheadSeconds) {
        mutableFloatStateOf(
            state.playheadSeconds.coerceIn(0.0, safeDuration).toFloat(),
        )
    }

    fun seekByFrames(frames: Int) {
        val next =
            (
                playhead.toDouble() +
                    frames.toDouble() / safeFps
                ).coerceIn(0.0, safeDuration)
        playhead = next.toFloat()
        onSeek(next)
    }

    Column(
        modifier,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(
                    "Timeline",
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    formatTimecode(playhead.toDouble(), safeFps) +
                        " / " +
                        formatTimecode(safeDuration, safeFps),
                    style = MaterialTheme.typography.labelMedium,
                )
            }
            Surface(
                tonalElevation = 1.dp,
                shape = androidx.compose.foundation.shape.RoundedCornerShape(10.dp),
            ) {
                Text(
                    if (state.precisionQualified) {
                        "PRECISION AUTHORITY"
                    } else {
                        "EVIDENCE MODE"
                    },
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(
                        horizontal = 10.dp,
                        vertical = 6.dp,
                    ),
                )
            }
        }

        Slider(
            value = playhead.coerceIn(0f, safeDuration.toFloat()),
            onValueChange = {
                playhead = it
                onSeek(it.toDouble())
            },
            valueRange = 0f..safeDuration.toFloat(),
        )

        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Button(onClick = { seekByFrames(-1) }) {
                Text("−1F")
            }
            Button(onClick = { seekByFrames(1) }) {
                Text("+1F")
            }
            Button(onClick = { seekByFrames(-10) }) {
                Text("−10F")
            }
            Button(onClick = { seekByFrames(10) }) {
                Text("+10F")
            }
            Text(
                "FPS " + String.format("%.3f", safeFps),
                style = MaterialTheme.typography.labelMedium,
            )
        }

        Text(
            "Loop " +
                formatTimecode(state.loopStartSeconds.coerceIn(0.0, safeDuration), safeFps) +
                " to " +
                formatTimecode(state.loopEndSeconds.coerceIn(0.0, safeDuration), safeFps),
            style = MaterialTheme.typography.labelSmall,
        )

        Text(
            state.precisionP95Millimeters?.let {
                "Measured P95 " + String.format("%.3f mm", it)
            } ?: "Measured P95 unavailable",

            style = MaterialTheme.typography.labelSmall,
        )
    }
}

private fun formatTimecode(seconds: Double, frameRate: Double): String {
    val safeSeconds =
        seconds.takeIf { it.isFinite() }?.coerceAtLeast(0.0) ?: 0.0
    val nominalFramesPerSecond =
        kotlin.math.round(
            frameRate.coerceAtLeast(1.0),
        ).toLong().coerceAtLeast(1L)
    val totalFrames =
        kotlin.math.round(
            safeSeconds *
                frameRate.coerceAtLeast(1.0),
        ).toLong()
    val frame =
        totalFrames %
            nominalFramesPerSecond
    val totalSeconds =
        totalFrames /
            nominalFramesPerSecond
    val second = totalSeconds % 60L
    val minute = (totalSeconds / 60L) % 60L
    val hour = totalSeconds / 3600L
    return String.format("%02d:%02d:%02d:%02d", hour, minute, second, frame)
}
