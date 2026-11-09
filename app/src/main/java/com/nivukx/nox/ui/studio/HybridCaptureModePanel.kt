package com.nivukx.nox.ui.studio

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nivukx.nox.core.model.CaptureEvidenceMode
import com.nivukx.nox.core.model.MotionCaptureCustomization
import com.nivukx.nox.core.model.RecordedVideoCodec
import com.nivukx.nox.core.model.RecordedVideoEvidenceProfile

/**
 * High-end capture acquisition surface.
 *
 * The panel makes the architectural distinction visible to the operator:
 * live motion is an immediate estimate, while recorded evidence enables a substantially more
 * expensive offline reconstruction pass.
 */
@Composable
fun HybridCaptureModePanel(
    settings: MotionCaptureCustomization,
    onSettingsChanged: (MotionCaptureCustomization) -> Unit,
    modifier: Modifier = Modifier,
) {
    LiquidGlassSurface(
        clear = false,
        elevated = true,
        modifier = modifier.fillMaxWidth(),
        contentPadding = 16.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                text = "Acquisition & Reconstruction",
                style = MaterialTheme.typography.titleMedium,
            )

            Text(
                text =
                    "Choose whether NoX produces only live motion, records visual evidence for a deferred solve, or exposes both paths together.",
                style = MaterialTheme.typography.bodySmall,
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                CaptureEvidenceMode.entries.forEach { mode ->
                    val selected =
                        settings.captureEvidenceMode == mode

                    Surface(
                        shape = RoundedCornerShape(18.dp),
                        tonalElevation =
                            if (selected) 5.dp else 1.dp,
                        modifier = Modifier
                            .width(210.dp)
                            .padding(vertical = 2.dp),
                        onClick = {
                            onSettingsChanged(
                                settings.copyWith {
                                    captureEvidenceMode = mode
                                    recordedVideoEvidenceProfile =
                                        if (
                                            mode ==
                                                CaptureEvidenceMode.LIVE_MOTION_ONLY
                                        ) {
                                            settings.recordedVideoEvidenceProfile
                                        } else {
                                            settings.recordedVideoEvidenceProfile
                                                ?: RecordedVideoEvidenceProfile.liveUltra()
                                        }
                                },
                            )
                        },
                    ) {
                        Column(
                            modifier = Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 10.dp,
                            ),
                            verticalArrangement = Arrangement.spacedBy(3.dp),
                        ) {
                            Text(
                                text =
                                    when (mode) {
                                        CaptureEvidenceMode.LIVE_MOTION_ONLY ->
                                            "LIVE MOTION"
                                        CaptureEvidenceMode.RECORD_AND_POSTPROCESS ->
                                            "RECORD → RECONSTRUCT"
                                        CaptureEvidenceMode.HYBRID_LIVE_AND_POST ->
                                            "HYBRID LIVE + POST"
                                    },
                                style = MaterialTheme.typography.labelLarge,
                            )
                            Text(
                                text =
                                    when (mode) {
                                        CaptureEvidenceMode.LIVE_MOTION_ONLY ->
                                            "Low-latency body/face/eye/hand state."
                                        CaptureEvidenceMode.RECORD_AND_POSTPROCESS ->
                                            "Preserve visual evidence for the deferred solve."
                                        CaptureEvidenceMode.HYBRID_LIVE_AND_POST ->
                                            "Live operator feedback plus the full post-capture path."
                                    },
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    }
                }
            }

            Text(
                text =
                    "Target subjects: ${settings.maxSubjects} / 50 production target / 64 hard ceiling",
                style = MaterialTheme.typography.labelLarge,
            )

            Slider(
                value = settings.maxSubjects.toFloat(),
                onValueChange = {
                    onSettingsChanged(
                        settings.copyWith {
                            maxSubjects =
                                it.toInt()
                                    .coerceIn(1, 64)
                        },
                    )
                },
                valueRange = 1f..64f,
                steps = 62,
            )

            Text(
                text =
                    "Runtime admission is bounded. Body tracking is retained first; optional dense surface, hand, eye and visual-evidence workloads are throttled according to CPU/GPU/NPU/thermal/memory headroom.",
                style = MaterialTheme.typography.bodySmall,
            )

            val video =
                settings.recordedVideoEvidenceProfile

            if (
                settings.captureEvidenceMode !=
                    CaptureEvidenceMode.LIVE_MOTION_ONLY &&
                video != null
            ) {
                Text(
                    text =
                        "Evidence ${video.width}×${video.height} @ ${video.fps} FPS · ${video.codec.operatorLabel()} · ${video.bitrateBitsPerSecond / 1_000_000} Mbps",
                    style = MaterialTheme.typography.labelMedium,
                )

                Slider(
                    value = video.fps.toFloat(),
                    onValueChange = { next ->
                        val fps =
                            next.toInt()
                                .coerceIn(24, 240)
                        onSettingsChanged(
                            settings.copyWith {
                                recordedVideoEvidenceProfile =
                                    video.copy(fps = fps)
                            },
                        )
                    },
                    valueRange = 24f..240f,
                    steps = 95,
                )
            }

            Text(
                text =
                    when (settings.captureEvidenceMode) {
                        CaptureEvidenceMode.LIVE_MOTION_ONLY ->
                            "No video file is created. Structured motion remains the canonical capture artifact."
                        CaptureEvidenceMode.RECORD_AND_POSTPROCESS ->
                            "The camera encoder records evidence while the reconstruction result is generated after capture."
                        CaptureEvidenceMode.HYBRID_LIVE_AND_POST ->
                            "The operator receives live motion immediately; the recorded evidence remains available for refinement."
                    },
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

private fun RecordedVideoCodec.operatorLabel(): String =
    when (this) {
        RecordedVideoCodec.HEVC -> "HEVC"
        RecordedVideoCodec.AVC -> "AVC"
        RecordedVideoCodec.AV1 -> "AV1"
    }
