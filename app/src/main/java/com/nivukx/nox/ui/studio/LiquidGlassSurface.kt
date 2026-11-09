package com.nivukx.nox.ui.studio

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.matchParentSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

private fun liquidGlassShape(): Shape = RoundedCornerShape(24.dp)

@Composable
fun LiquidGlassSurface(
    modifier: Modifier = Modifier,
    clear: Boolean = false,
    elevated: Boolean = true,
    effectScale: Float = 1.0f,
    contentPadding: Dp = 0.dp,
    materialRole: StudioMaterialAdaptationEngine.MaterialRole =
        StudioMaterialAdaptationEngine.MaterialRole.SECONDARY_TELEMETRY,
    underlyingLuminance: Float = 0.50f,
    contentContrast: Float = 0.90f,
    capturePriority: Float = 1.0f,
    operatorFocus: Float = 1.0f,
    plateProtection: Float = 1.0f,
    reduceTransparency: Boolean = false,
    reduceMotion: Boolean = false,
    accessibilityLabel: String? = null,
    accessibilityRole: Role? = null,
    accessibilityStateDescription: String? = null,
    content: @Composable BoxScope.() -> Unit,
) {
    val shape = remember { liquidGlassShape() }
    val scheme = MaterialTheme.colorScheme
    val materialPolicy = remember(
        materialRole,
        underlyingLuminance,
        contentContrast,
        capturePriority,
        operatorFocus,
        plateProtection,
        reduceTransparency,
        reduceMotion,
    ) {
        StudioMaterialAdaptationEngine().evaluate(
            role = materialRole,
            input = StudioMaterialAdaptationEngine.Input(
                underlyingLuminance = underlyingLuminance.coerceIn(0.0f, 1.0f).toDouble(),
                contentContrast = contentContrast.coerceIn(0.0f, 1.0f).toDouble(),
                capturePriority = capturePriority.coerceIn(0.0f, 1.0f).toDouble(),
                operatorFocus = operatorFocus.coerceIn(0.0f, 1.0f).toDouble(),
                reduceTransparency = reduceTransparency,
                reduceMotion = reduceMotion,
            ),
        )
    }

    val scale = (
        effectScale.coerceIn(0.35f, 1.0f) *
            materialPolicy.effectScale.toFloat()
        ).coerceIn(0.30f, 1.0f)

    val baseAlpha = (
        materialPolicy.materialAlpha.toFloat() *
            (if (clear) 0.74f else 1.0f)
        ).coerceIn(0.22f, 0.90f)

    val container = if (clear) {
        scheme.surface.copy(
            alpha = (
                0.36f + baseAlpha * 0.32f
                ).coerceIn(0.30f, 0.58f),
        )
    } else {
        scheme.surfaceContainer.copy(
            alpha = (
                0.50f + baseAlpha * 0.48f
                ).coerceIn(0.62f, 0.92f),
        )
    }

    val edge = scheme.outline.copy(
        alpha = (
            materialPolicy.borderAlpha.toFloat() *
                (if (clear) 0.92f else 0.78f)
            ).coerceIn(0.10f, 0.38f),
    )
    val innerEdge = scheme.onSurface.copy(
        alpha = (
            (if (clear) 0.11f else 0.075f) *
                (0.62f + 0.38f * scale) +
                materialPolicy.contentContrastBoost.toFloat() * 0.08f
            ).coerceIn(0.04f, 0.16f),
    )
    val topSpecular = scheme.onSurface.copy(
        alpha = (
            materialPolicy.specularAlpha.toFloat() *
                scale
            ).coerceIn(0.02f, 0.16f),
    )
    val lowerShade = scheme.onSurface.copy(
        alpha = (
            (if (clear) 0.035f else 0.055f) *
                (0.75f + 0.25f * scale)
            ).coerceIn(0.02f, 0.06f),
    )

    val plateAlpha = (
        (
            if (clear) 0.08f else 0.12f
        ) +
            materialPolicy.contentContrastBoost.toFloat() * 0.34f +
            (1.0f - plateProtection.coerceIn(0.0f, 1.0f)) * 0.10f +
            if (reduceTransparency) 0.08f else 0.0f
        ).coerceIn(0.06f, 0.34f)

    val readabilityPlate = scheme.surface.copy(
        alpha = plateAlpha,
    )

    val surfaceBrush = remember(
        clear,
        scale,
        topSpecular,
        lowerShade,
    ) {
        Brush.verticalGradient(
            0.0f to topSpecular,
            0.22f to Color.Transparent,
            0.72f to Color.Transparent,
            1.0f to lowerShade,
        )
    }

    val ambientBrush = remember(
        clear,
        scale,
        scheme.primary,
        scheme.tertiary,
    ) {
        Brush.linearGradient(
            colors = listOf(
                scheme.primary.copy(
                    alpha = (
                        if (clear) 0.045f else 0.025f
                    ) * scale,
                ),
                Color.Transparent,
                scheme.tertiary.copy(
                    alpha = (
                        if (clear) 0.030f else 0.018f
                    ) * scale,
                ),
            ),
        )
    }

    val shadowElevation =
        if (elevated) {
            (
                8.0f *
                    scale *
                    materialPolicy.animationScale.toFloat()
                ).dp
        } else {
            2.dp
        }

    Surface(
        modifier = modifier
            .semantics(mergeDescendants = true) {
                accessibilityLabel?.let { contentDescription = it }
                accessibilityRole?.let { role = it }
                accessibilityStateDescription?.let {
                    stateDescription = it
                }
            }
            .clip(shape)
            .border(
                BorderStroke(1.dp, edge),
                shape,
            ),
        shape = shape,
        color = container,
        contentColor = scheme.onSurface,
        tonalElevation = if (elevated) 4.dp else 1.dp,
        shadowElevation = shadowElevation,
    ) {
        Box(
            modifier = Modifier
                .background(ambientBrush)
                .background(surfaceBrush)
                .border(
                    BorderStroke(0.75.dp, innerEdge),
                    shape,
                )
                .padding(contentPadding),
        ) {
            Box(
                modifier = Modifier
                    .matchParentSize()
                    .background(
                        readabilityPlate,
                        shape,
                    ),
            )
            Box(
                modifier = Modifier.matchParentSize(),
                content = content,
            )
        }
    }
}
