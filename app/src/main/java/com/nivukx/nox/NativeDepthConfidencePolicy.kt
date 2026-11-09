package com.nivukx.nox

/**
 * Combines frame-level confidence with adaptive TSDF weighting without violating the native
 * observation-confidence contract. Pixel-level DEPTH16 confidence is still carried separately.
 */
internal object NativeDepthConfidencePolicy {
    fun resolve(
        frameConfidence: Double,
        adaptiveScale: Double,
    ): Double {
        val boundedFrameConfidence =
            frameConfidence
                .takeIf { it.isFinite() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0
        val boundedAdaptiveScale =
            adaptiveScale
                .takeIf { it.isFinite() }
                ?.coerceIn(0.10, 1.50)
                ?: return 0.0

        return (boundedFrameConfidence * boundedAdaptiveScale)
            .coerceIn(0.0, 1.0)
    }
}
