package com.nivukx.nox.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlin.math.PI
import kotlin.math.sin

private val GlassShape = RoundedCornerShape(28.dp)

@Composable
fun LiquidGlassBackdrop(
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    val transition = rememberInfiniteTransition(
        label = "liquidMotion",
    )

    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = (PI * 2f).toFloat(),
        animationSpec = infiniteRepeatable(
            tween(
                durationMillis = 14_000,
                easing = FastOutSlowInEasing,
            ),
        ),
        label = "liquidPhase",
    )

    Box(
        modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primary.copy(alpha = 0.16f),
                        Color.Transparent,
                    ),
                    center = Offset(
                        0.30f + sin(phase) * 0.08f,
                        0.18f + sin(phase * 0.7f) * 0.06f,
                    ),
                    radius = 720f,
                ),
            )
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.secondary.copy(alpha = 0.10f),
                        Color.Transparent,
                    ),
                    center = Offset(
                        0.72f + sin(phase * 0.8f) * 0.06f,
                        0.78f + sin(phase * 0.55f) * 0.05f,
                    ),
                    radius = 620f,
                ),
            ),
    ) {
        content()
    }
}

@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    selected: Boolean = false,
    intensity: Float = 1f,
    content: @Composable () -> Unit,
) {
    val borderAlpha by animateFloatAsState(
        targetValue = if (selected) 0.78f else 0.28f,
        animationSpec = spring(),
        label = "glassBorder",
    )

    val shimmerPhase =
        if (selected) {
            val shimmer = rememberInfiniteTransition(
                label = "selectedGlassShimmer",
            )
            val phase by shimmer.animateFloat(
                initialValue = 0f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    tween(
                        durationMillis = 4_800,
                        easing = FastOutSlowInEasing,
                    ),
                ),
                label = "selectedGlassShimmerPhase",
            )
            phase
        } else {
            0f
        }

    Surface(
        modifier = modifier
            .shadow(
                if (selected) 28.dp else 14.dp,
                GlassShape,
                clip = false,
            )
            .clip(GlassShape)
            .background(
                Brush.linearGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.surface.copy(
                            alpha = (0.88f * intensity).coerceIn(0f, 1f),
                        ),
                        MaterialTheme.colorScheme.surfaceContainerHighest.copy(
                            alpha = (0.62f * intensity).coerceIn(0f, 1f),
                        ),
                        MaterialTheme.colorScheme.surface.copy(
                            alpha = (0.78f * intensity).coerceIn(0f, 1f),
                        ),
                    ),
                    start = Offset(0f, 0f),
                    end = Offset(
                        1200f +
                            (
                                if (selected) shimmerPhase
                                else 0f
                            ) * 240f,
                        760f,
                    ),
                ),
            )
            .border(
                BorderStroke(
                    1.dp,
                    Brush.linearGradient(
                        colors = listOf(
                            Color.White.copy(alpha = borderAlpha),
                            MaterialTheme.colorScheme.primary.copy(
                                alpha = borderAlpha * 0.38f,
                            ),
                            Color.Transparent,
                        ),
                    ),
                ),
                GlassShape,
            )
            .graphicsLayer {
                alpha = 0.995f
            },
        color = Color.Transparent,
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(
                            Color.White.copy(alpha = 0.035f),
                            Color.Transparent,
                        ),
                    ),
                ),
        ) {
            content()
        }
    }
}


@Composable
fun LiquidGlassControlLayer(
    state: String,
    motionProfile: String,
    dominantSubject: String,
    calibrationReady: Boolean,
    sessionHealthScore: Double,
    captureIntegrityScore: Double,
    captureIntegrityBand: String,
    captureIntegrityOutputAllowed: Boolean,
    captureIntegrityBottleneck: String,
    synchronizationConfidence: Double,
    onSettingsClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    val pulse by animateFloatAsState(
        targetValue = if (
            sessionHealthScore >= 0.70 &&
                synchronizationConfidence >= 0.70
        ) {
            1.0f
        } else {
            0.72f
        },
        animationSpec = tween(
            durationMillis = 420,
            easing = FastOutSlowInEasing,
        ),
        label = "controlLayerPulse",
    )

    Surface(
        modifier = modifier
            .height(68.dp)
            .semantics(mergeDescendants = true) {
                contentDescription =
                    "NoX Motion Studio. " +
                        "State $state. " +
                        "Motion profile $motionProfile. " +
                        "Subject $dominantSubject. " +
                        "Calibration " +
                        if (calibrationReady) "ready" else "required" +
                        ". Session health " +
                        (sessionHealthScore * 100.0).toInt() +
                        " percent. Synchronization " +
                        (synchronizationConfidence * 100.0).toInt() +
                        " percent. " +
                        "Integrity " +
                        (captureIntegrityScore * 100.0).toInt() +
                        " percent. Band " + captureIntegrityBand + ". " +
                        "Output " + if (captureIntegrityOutputAllowed) "allowed" else "held" + ". " +
                        "Bottleneck " + captureIntegrityBottleneck + "."
            }
            .shadow(
                elevation = 18.dp,
                shape = RoundedCornerShape(24.dp),
                clip = false,
            )
            .clip(RoundedCornerShape(24.dp))
            .border(
                BorderStroke(
                    1.dp,
                    Brush.linearGradient(
                        listOf(
                            Color.White.copy(alpha = 0.34f),
                            MaterialTheme.colorScheme.primary.copy(alpha = 0.20f),
                            Color.Transparent,
                        ),
                    ),
                ),
                RoundedCornerShape(24.dp),
            ),
        color = MaterialTheme.colorScheme.surface.copy(
            alpha = 0.74f,
        ),
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .horizontalScroll(
                    rememberScrollState(),
                )
                .padding(
                    horizontal = 16.dp,
                    vertical = 10.dp,
                ),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Column(
                modifier = Modifier.widthIn(min = 150.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(
                    "NOX MOTION STUDIO",
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    motionProfile
                        .replace('_', ' ')
                        .lowercase()
                        .replaceFirstChar {
                            if (it.isLowerCase()) it.titlecase()
                            else it.toString()
                        },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            TelemetryChip(
                label = "SUBJECT",
                value = dominantSubject
                    .replace('_', ' ')
                    .uppercase(),
            )

            TelemetryChip(
                label = "RIG",
                value = (synchronizationConfidence * 100.0)
                    .toInt()
                    .coerceIn(0, 100)
                    .toString() + "%",
            )


            TelemetryChip(
                label = "INTEGRITY",
                value = (captureIntegrityScore * 100.0)
                    .toInt()
                    .coerceIn(0, 100)
                    .toString() + "%",
            )

            StatusPill(
                text = if (calibrationReady) {
                    "CALIBRATED"
                } else {
                    "CALIBRATION"
                },
                healthy = calibrationReady,
            )

            Box(
                modifier = Modifier
                    .width(7.dp)
                    .height(7.dp)
                    .clip(RoundedCornerShape(50))
                    .background(
                        MaterialTheme.colorScheme.primary.copy(
                            alpha = pulse,
                        ),
                    ),
            )

            StatusPill(
                text = captureIntegrityBand.replace('_', ' ').uppercase(),
                healthy = captureIntegrityOutputAllowed,
            )

            StatusPill(
                text = state
                    .replace('_', ' ')
                    .uppercase(),
                healthy =
                    sessionHealthScore >= 0.55,
            )

            onSettingsClick?.let { openSettings ->
                Surface(
                    onClick = openSettings,
                    modifier = Modifier.height(44.dp),
                    shape = RoundedCornerShape(15.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerHigh.copy(
                        alpha = 0.78f,
                    ),
                    border = BorderStroke(
                        1.dp,
                        MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.42f),
                    ),
                ) {
                    Box(
                        Modifier.padding(horizontal = 13.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            "SETTINGS",
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }
                }
            }
        }
    }
}

data class GlassSignalCell(
    val label: String,
    val value: String,
    val score: Double,
    val detail: String,
)

@Composable
fun GlassSignalMatrix(
    cells: List<GlassSignalCell>,
    modifier: Modifier = Modifier,
) {
    val strongCount by remember(cells) {
        derivedStateOf {
            cells.count { it.score >= 0.70 }
        }
    }

    GlassCard(
        modifier = modifier,
        intensity = 0.80f,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    horizontal = 14.dp,
                    vertical = 12.dp,
                ),
            verticalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(
                        "LIVE EVIDENCE MATRIX",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "Independent confidence surfaces for capture decisions",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Text(
                    "$strongCount/${cells.size} STRONG",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                cells.forEach { cell ->
                    val score = cell.score.coerceIn(0.0, 1.0)
                    val active = score >= 0.55

                    GlassCard(
                        modifier = Modifier.widthIn(
                            min = 142.dp,
                            max = 180.dp,
                        ),
                        selected = active && score >= 0.80,
                        intensity = if (score >= 0.70) 0.92f else 0.72f,
                    ) {
                        Column(
                            modifier = Modifier.padding(
                                horizontal = 12.dp,
                                vertical = 11.dp,
                            ),
                            verticalArrangement = Arrangement.spacedBy(5.dp),
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement =
                                    Arrangement.SpaceBetween,
                                verticalAlignment =
                                    Alignment.CenterVertically,
                            ) {
                                Text(
                                    cell.label.uppercase(),
                                    style = MaterialTheme.typography.labelSmall,
                                    color =
                                        MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                Text(
                                    "${(score * 100.0).toInt()}%",
                                    style = MaterialTheme.typography.labelMedium,
                                )
                            }

                            Text(
                                cell.value,
                                style = MaterialTheme.typography.titleMedium,
                            )

                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(5.dp)
                                    .clip(RoundedCornerShape(50))
                                    .background(
                                        MaterialTheme.colorScheme
                                            .surfaceContainerHighest
                                            .copy(alpha = 0.72f),
                                    ),
                            ) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth(score.toFloat())
                                        .height(5.dp)
                                        .clip(RoundedCornerShape(50))
                                        .background(
                                            Brush.horizontalGradient(
                                                listOf(
                                                    MaterialTheme.colorScheme.primary,
                                                    MaterialTheme.colorScheme.secondary,
                                                ),
                                            ),
                                        ),
                                )
                            }

                            Text(
                                cell.detail,
                                style = MaterialTheme.typography.labelSmall,
                                color =
                                    MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun GlassMetric(
    label: String,
    value: String,
    detail: String? = null,
    modifier: Modifier = Modifier,
) {
    GlassCard(modifier) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(5.dp),
        ) {
            Text(
                label.uppercase(),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                value,
                style = MaterialTheme.typography.headlineSmall,
            )
            detail?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
fun GlassGauge(
    label: String,
    value: Float,
    modifier: Modifier = Modifier,
) {
    val progress by animateFloatAsState(
        targetValue = value.coerceIn(0f, 1f),
        animationSpec = tween(
            durationMillis = 450,
            easing = FastOutSlowInEasing,
        ),
        label = "gauge",
    )

    GlassCard(
        modifier = modifier,
        intensity = 0.92f,
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    label.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    (progress * 100f).toInt().toString() + "%",
                    style = MaterialTheme.typography.labelLarge,
                )
            }

            Box(
                Modifier
                    .fillMaxWidth()
                    .height(7.dp)
                    .clip(RoundedCornerShape(50))
                    .background(
                        MaterialTheme.colorScheme.surfaceContainerHighest.copy(
                            alpha = 0.82f,
                        ),
                    ),
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(progress)
                        .height(7.dp)
                        .clip(RoundedCornerShape(50))
                        .background(
                            Brush.horizontalGradient(
                                listOf(
                                    MaterialTheme.colorScheme.primary,
                                    MaterialTheme.colorScheme.secondary,
                                ),
                            ),
                        ),
                )
            }
        }
    }
}

@Composable
fun GlassAction(
    text: String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    primary: Boolean = false,
    onClick: () -> Unit,
) {
    val container = if (primary) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.surfaceContainerHigh
    }

    Surface(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(56.dp),
        shape = RoundedCornerShape(19.dp),
        color = container.copy(
            alpha = if (enabled) {
                if (primary) 0.92f else 0.74f
            } else {
                0.38f
            },
        ),
        border = BorderStroke(
            1.dp,
            MaterialTheme.colorScheme.outlineVariant.copy(
                alpha = if (primary) 0.70f else 0.42f,
            ),
        ),
    ) {
        Box(
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text,
                style = MaterialTheme.typography.labelLarge,
            )
        }
    }
}

@Composable
fun StatusPill(
    text: String,
    healthy: Boolean,
    modifier: Modifier = Modifier,
) {
    val color = if (healthy) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.errorContainer
    }

    val border = if (healthy) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.error
    }

    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(50),
        color = color.copy(alpha = 0.80f),
        border = BorderStroke(
            1.dp,
            border.copy(alpha = 0.50f),
        ),
    ) {
        Text(
            text,
            modifier = Modifier.padding(
                horizontal = 12.dp,
                vertical = 7.dp,
            ),
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
fun SectionTitle(
    title: String,
    subtitle: String? = null,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text(
            title,
            style = MaterialTheme.typography.titleLarge,
        )

        subtitle?.let {
            Text(
                it,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
fun GlassTelemetryTile(
    label: String,
    value: String,
    detail: String? = null,
    active: Boolean = false,
    modifier: Modifier = Modifier,
) {
    val pulse = rememberInfiniteTransition(
        label = "telemetryPulse",
    )

    val pulsePhase by pulse.animateFloat(
        initialValue = 0.78f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            tween(
                durationMillis = 1600,
                easing = FastOutSlowInEasing,
            ),
        ),
        label = "telemetryPulsePhase",
    )

    GlassCard(
        modifier = modifier.semantics(mergeDescendants = true) {
            contentDescription =
                buildString {
                    append(label)
                    append(": ")
                    append(value)
                    detail?.let {
                        append(". ")
                        append(it)
                    }
                }
        },
        selected = active,
        intensity = 0.86f,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    horizontal = 14.dp,
                    vertical = 12.dp,
                ),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Box(
                modifier = Modifier
                    .width(8.dp)
                    .height(8.dp)
                    .clip(RoundedCornerShape(50))
                    .background(
                        if (active) {
                            MaterialTheme.colorScheme.secondary.copy(
                                alpha = pulsePhase,
                            )
                        } else {
                            MaterialTheme.colorScheme.outlineVariant.copy(
                                alpha = 0.72f,
                            )
                        },
                    ),
            )

            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(
                    label.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    value,
                    style = MaterialTheme.typography.titleMedium,
                )
                detail?.let {
                    Text(
                        it,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
fun NoxCaptureButton(
    recording: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val container = if (recording) {
        MaterialTheme.colorScheme.errorContainer
    } else {
        MaterialTheme.colorScheme.primaryContainer
    }
    Surface(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier
            .height(78.dp)
            .width(78.dp)
            .shadow(
                elevation = if (recording) 22.dp else 16.dp,
                shape = androidx.compose.foundation.shape.CircleShape,
                clip = false,
            ),
        shape = androidx.compose.foundation.shape.CircleShape,
        color = container.copy(alpha = if (enabled) 0.96f else 0.42f),
        border = BorderStroke(
            2.dp,
            if (recording) {
                MaterialTheme.colorScheme.error
            } else {
                MaterialTheme.colorScheme.primary
            }.copy(alpha = 0.78f),
        ),
    ) {
        Box(
            Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center,
        ) {
            Surface(
                modifier = Modifier
                    .height(if (recording) 30.dp else 38.dp)
                    .width(if (recording) 30.dp else 38.dp),
                shape = if (recording) {
                    RoundedCornerShape(8.dp)
                } else {
                    androidx.compose.foundation.shape.CircleShape
                },
                color = if (recording) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.primary
                },
            ) {}
        }
    }
}

@Composable
fun TelemetryChip(
    label: String,
    value: String,
) {
    Surface(
        modifier = Modifier.semantics(mergeDescendants = true) {
            contentDescription = "$label: $value"
        },
        shape = RoundedCornerShape(15.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHigh.copy(
            alpha = 0.66f,
        ),
        border = BorderStroke(
            1.dp,
            MaterialTheme.colorScheme.outlineVariant.copy(
                alpha = 0.36f,
            ),
        ),
    ) {
        Row(
            modifier = Modifier.padding(
                horizontal = 11.dp,
                vertical = 8.dp,
            ),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                label.uppercase(),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.width(7.dp))
            Text(
                value,
                style = MaterialTheme.typography.labelMedium,
            )
        }
    }
}
