package com.nivukx.nox.core.capture

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Selects a motion-capture analysis mode from measured per-resolution frame-duration evidence.
 *
 * Camera2 exposes frame-rate ranges and output durations through different capability surfaces.
 * This optimizer never assumes that the highest advertised FPS is available at the chosen size.
 */
class CameraMotionModeOptimizer(
    private val config: Config = Config(),
) {

    data class Config(
        val targetWidth: Int = 1920,
        val targetHeight: Int = 1080,
        val minimumFps: Int = 15,
        val maximumFps: Int = 240,
        val temporalWeight: Double = 0.68,
        val spatialWeight: Double = 0.32,
    ) {
        init {
            require(targetWidth > 0)
            require(targetHeight > 0)
            require(minimumFps >= 1)
            require(maximumFps >= minimumFps)
            require(temporalWeight >= 0.0)
            require(spatialWeight >= 0.0)
            require(temporalWeight + spatialWeight > 0.0)
        }
    }

    data class Candidate(
        val width: Int,
        val height: Int,
        val maximumFps: Int,
    ) {
        init {
            require(width > 0)
            require(height > 0)
            require(maximumFps >= 1)
        }

        val pixelCount: Long
            get() = width.toLong() * height.toLong()
    }

    data class Selection(
        val width: Int,
        val height: Int,
        val fps: Int,
        val temporalScore: Double,
        val spatialScore: Double,
        val combinedScore: Double,
    )

    fun select(
        candidates: List<Candidate>,
        advertisedMaximumFps: Int,
    ): Selection? {
        if (candidates.isEmpty()) return null

        val boundedAdvertisedFps =
            advertisedMaximumFps.coerceIn(
                config.minimumFps,
                config.maximumFps,
            )

        val valid =
            candidates
                .map { candidate ->
                    candidate.copy(
                        maximumFps =
                            min(
                                candidate.maximumFps,
                                boundedAdvertisedFps,
                            ),
                    )
                }
                .filter {
                    it.maximumFps >= config.minimumFps
                }

        if (valid.isEmpty()) return null

        val maxPixels =
            valid.maxOf { it.pixelCount }
                .coerceAtLeast(1L)
                .toDouble()

        val targetPixels =
            config.targetWidth.toDouble() *
                config.targetHeight.toDouble()

        val weightSum =
            (config.temporalWeight + config.spatialWeight)
                .coerceAtLeast(1e-9)

        return valid
            .map { candidate ->
                val temporalScore =
                    (
                        candidate.maximumFps.toDouble() /
                            boundedAdvertisedFps.toDouble()
                    ).coerceIn(0.0, 1.0)

                val targetDistance =
                    abs(
                        candidate.pixelCount.toDouble() -
                            targetPixels,
                    ) /
                        max(
                            targetPixels,
                            maxPixels,
                        )

                val spatialScore =
                    (
                        1.0 -
                            targetDistance
                    ).coerceIn(0.0, 1.0)

                val combinedScore =
                    (
                        config.temporalWeight *
                            temporalScore +
                            config.spatialWeight *
                                spatialScore
                    ) / weightSum

                Selection(
                    width = candidate.width,
                    height = candidate.height,
                    fps = candidate.maximumFps,
                    temporalScore = temporalScore,
                    spatialScore = spatialScore,
                    combinedScore = combinedScore,
                )
            }
            .maxWithOrNull(
                compareBy<Selection> { it.combinedScore }
                    .thenBy { it.fps }
                    .thenBy { it.width * it.height },
            )
    }
}
