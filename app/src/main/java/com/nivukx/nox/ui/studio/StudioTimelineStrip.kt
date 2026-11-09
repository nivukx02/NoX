package com.nivukx.nox.ui.studio

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun StudioTimelineStrip(
    currentFrame: Long,
    frameCount: Long,
    markers: List<Long>,
    modifier: Modifier = Modifier,
) {
    Row(modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).height(54.dp)) {
        Canvas(Modifier.fillMaxWidth().height(54.dp)) {
            val safeFrames = frameCount.coerceAtLeast(1L)
            val frameWidth = (size.width / 60f).coerceAtLeast(18f)
            val y = size.height * 0.55f
            val totalWidth = frameWidth * safeFrames
            drawLine(Color.White, Offset(0f, y), Offset(totalWidth, y), strokeWidth = 1.5f)
            for (marker in markers) {
                if (marker !in 0..safeFrames) continue
                val x = frameWidth * marker
                drawLine(Color.White, Offset(x, y - 12f), Offset(x, y + 12f), strokeWidth = 2f, cap = StrokeCap.Round)
            }
            val playhead = frameWidth * currentFrame.coerceIn(0L, safeFrames)
            drawLine(Color.White, Offset(playhead, 0f), Offset(playhead, size.height), strokeWidth = 2.5f)
        }
    }
}