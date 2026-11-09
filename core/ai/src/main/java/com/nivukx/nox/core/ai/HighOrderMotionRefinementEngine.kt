package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Higher-order temporal evidence layer for the motion front-end.
 *
 * The detector itself remains the source of observations. This layer evaluates the dynamics
 * surrounding those observations with robust statistics so that isolated detector spikes, camera
 * motion, vibration and weak-confidence micro motion do not get promoted to "real" body motion.
 *
 * It intentionally changes confidence/evidence, not coordinates. Geometry stays downstream and
 * remains responsible for world-space reconstruction.
 */
class HighOrderMotionRefinementEngine(
    private val windowSize: Int = 48,
) : AutoCloseable {

    data class Assessment(
        val refinedQuality: MotionCaptureEngine.Quality,
        val robustMotionScore: Double,
        val microMotionIntegrity: Double,
        val cameraSuppressionIntegrity: Double,
        val temporalPredictionTrust: Double,
        val outlierRate: Double,
        val dynamicRegimeConfidence: Double,
        val spectralStability: Double,
        val evidenceWarnings: List<String>,
    )

    private data class Sample(
        val motionScore: Double,
        val speed: Double,
        val acceleration: Double,
        val jerk: Double,
        val jitter: Double,
        val confidence: Double,
        val occlusion: Double,
        val vibrationRisk: Double,
    )

    private val history = ArrayDeque<Sample>()

    @Synchronized
    fun refine(
        quality: MotionCaptureEngine.Quality,
        context: MotionCaptureEngine.MotionContext,
    ): Assessment {
        val sample = Sample(
            motionScore = finite(quality.motionScore),
            speed = finite(quality.meanSpeedNormalizedPerSecond),
            acceleration = finite(
                quality.meanAccelerationNormalizedPerSecond2,
            ),
            jerk = finite(quality.meanJerkNormalizedPerSecond3),
            jitter = finite(quality.jitterNormalized),
            confidence = finite(quality.meanConfidence),
            occlusion = finite(quality.occlusionRatio),
            vibrationRisk = finite(quality.vibrationRisk),
        )

        history.addLast(sample)
        while (history.size > windowSize) {
            history.removeFirst()
        }

        val robustMotion = robustConsensus { it.motionScore }
        val robustSpeed = robustConsensus { it.speed }
        val robustAcceleration = robustConsensus { it.acceleration }
        val robustJerk = robustConsensus { it.jerk }
        val robustJitter = robustConsensus { it.jitter }
        val robustConfidence = robustConsensus { it.confidence }
        val robustOcclusion = robustConsensus { it.occlusion }
        val robustVibration = robustConsensus { it.vibrationRisk }

        val motionOutlierRate =
            robustOutlierRate { it.motionScore }
        val dynamicsOutlierRate =
            (
                robustOutlierRate { it.speed } +
                    robustOutlierRate { it.acceleration } +
                    robustOutlierRate { it.jerk }
                ) / 3.0

        val cameraSuppressionIntegrity =
            (
                1.0 -
                    context.cameraDominance * 0.62 -
                    context.rollingShutterRisk * 0.22 -
                    context.sensorConfidence
                        .let { 1.0 - it }
                        .coerceIn(0.0, 1.0) * 0.16
                ).coerceIn(0.0, 1.0)

        val microMotionIntegrity =
            (
                quality.microMotionScore * 0.54 +
                    (
                        1.0 -
                            quality.meanMeasurementUncertaintyScore
                    ).coerceIn(0.0, 1.0) * 0.24 +
                    quality.temporalStability * 0.22
                ).coerceIn(0.0, 1.0)

        val temporalPredictionTrust =
            (
                quality.temporalStability * 0.40 +
                    (1.0 - quality.rejectedMeasurementRatio) * 0.25 +
                    quality.dynamicsConfidence * 0.20 +
                    cameraSuppressionIntegrity * 0.15
                ).coerceIn(0.0, 1.0)

        val spectralStability =
            (
                quality.spectralConfidence * 0.52 +
                    (1.0 - quality.vibrationRisk) * 0.28 +
                    quality.periodicMotionScore * 0.20
                ).coerceIn(0.0, 1.0)

        val robustMotionScore =
            (
                robustMotion * 0.32 +
                    normalizeRobustSpeed(robustSpeed) * 0.18 +
                    normalizeRobustAcceleration(robustAcceleration) * 0.14 +
                    normalizeRobustJerk(robustJerk) * 0.08 +
                    (1.0 - robustJitter) * 0.10 +
                    robustConfidence * 0.10 +
                    (1.0 - robustOcclusion) * 0.04 +
                    (1.0 - robustVibration) * 0.04
                ).coerceIn(0.0, 1.0)

        val evidenceWarnings = buildList {
            if (motionOutlierRate > 0.22) {
                add("high-order-motion-spikes")
            }
            if (dynamicsOutlierRate > 0.20) {
                add("high-order-dynamics-outliers")
            }
            if (cameraSuppressionIntegrity < 0.55) {
                add("camera-motion-dominant")
            }
            if (microMotionIntegrity < 0.55) {
                add("micro-motion-evidence-weak")
            }
            if (spectralStability < 0.45 && quality.spectralConfidence >= 0.45) {
                add("spectral-instability")
            }
            if (quality.predictedLandmarks > quality.trackedLandmarks) {
                add("prediction-dominant-frame")
            }
        }.distinct()

        val adjustedConfidence =
            (
                quality.meanConfidence * 0.62 +
                    robustConfidence * 0.18 +
                    temporalPredictionTrust * 0.10 +
                    cameraSuppressionIntegrity * 0.10 -
                    motionOutlierRate * 0.10
                ).coerceIn(0.0, 1.0)

        val adjustedTemporalStability =
            (
                quality.temporalStability * 0.56 +
                    temporalPredictionTrust * 0.24 +
                    cameraSuppressionIntegrity * 0.20
                ).coerceIn(0.0, 1.0)

        val adjustedUncertainty =
            (
                quality.meanMeasurementUncertaintyScore * 0.62 +
                    motionOutlierRate * 0.18 +
                    (1.0 - cameraSuppressionIntegrity) * 0.10 +
                    (1.0 - microMotionIntegrity) * 0.10
                ).coerceIn(0.0, 1.0)

        val refinedQuality = quality.copy(
            motionScore = (
                quality.motionScore * 0.48 +
                    robustMotionScore * 0.32 +
                    quality.cameraCompensatedMotionScore *
                        cameraSuppressionIntegrity * 0.20
                ).coerceIn(0.0, 1.0),
            meanConfidence = adjustedConfidence,
            temporalStability = adjustedTemporalStability,
            meanMeasurementUncertaintyScore = adjustedUncertainty,
            dynamicsConfidence = (
                quality.dynamicsConfidence * 0.58 +
                    robustConfidence * 0.18 +
                    (1.0 - dynamicsOutlierRate) * 0.14 +
                    spectralStability * 0.10
                ).coerceIn(0.0, 1.0),
            jitterNormalized = (
                quality.jitterNormalized * 0.70 +
                    robustJitter * 0.20 +
                    motionOutlierRate * 0.10
                ).coerceIn(0.0, 1.0),
            rejectedMeasurementRatio = (
                quality.rejectedMeasurementRatio * 0.82 +
                    motionOutlierRate * 0.18
                ).coerceIn(0.0, 1.0),
            microMotionScore = (
                quality.microMotionScore * 0.68 +
                    microMotionIntegrity * 0.32
                ).coerceIn(0.0, 1.0),
            vibrationRisk = (
                quality.vibrationRisk * 0.70 +
                    robustVibration * 0.20 +
                    (1.0 - spectralStability) * 0.10
                ).coerceIn(0.0, 1.0),
        )

        return Assessment(
            refinedQuality = refinedQuality,
            robustMotionScore = robustMotionScore,
            microMotionIntegrity = microMotionIntegrity,
            cameraSuppressionIntegrity = cameraSuppressionIntegrity,
            temporalPredictionTrust = temporalPredictionTrust,
            outlierRate = (
                motionOutlierRate * 0.55 +
                    dynamicsOutlierRate * 0.45
                ).coerceIn(0.0, 1.0),
            dynamicRegimeConfidence = (
                robustConfidence * 0.45 +
                    temporalPredictionTrust * 0.35 +
                    spectralStability * 0.20
                ).coerceIn(0.0, 1.0),
            spectralStability = spectralStability,
            evidenceWarnings = evidenceWarnings,
        )
    }

    private fun robustConsensus(
        selector: (Sample) -> Double,
    ): Double {
        val values = history.map(selector)
            .filter { it.isFinite() }
            .sorted()

        if (values.isEmpty()) return 0.0

        val median = median(values)
        val deviations = values
            .map { abs(it - median) }
            .sorted()
        val mad = median(deviations).coerceAtLeast(1e-9)
        val scale = 1.4826 * mad

        if (scale < 1e-8) return median

        var weightedSum = 0.0
        var weightSum = 0.0

        for (value in values) {
            val robustZ = abs(value - median) / scale
            val weight =
                if (robustZ <= 2.5) {
                    1.0
                } else {
                    2.5 / max(2.5, robustZ)
                }
            weightedSum += value * weight
            weightSum += weight
        }

        return if (weightSum > 0.0) {
            weightedSum / weightSum
        } else {
            median
        }
    }

    private fun robustOutlierRate(
        selector: (Sample) -> Double,
    ): Double {
        val values = history.map(selector)
            .filter { it.isFinite() }
            .sorted()

        if (values.size < 5) return 0.0

        val median = median(values)
        val mad = median(
            values.map { abs(it - median) }.sorted(),
        ).coerceAtLeast(1e-9)
        val scale = 1.4826 * mad

        if (scale < 1e-8) {
            val minValue = values.first()
            val maxValue = values.last()
            return if (abs(maxValue - minValue) <= 1e-6) 0.0 else 0.10
        }

        val threshold = 3.5 * scale
        return values.count {
            abs(it - median) > threshold
        }.toDouble() / values.size.toDouble()
    }

    private fun normalizeRobustSpeed(value: Double): Double =
        (1.0 - expDecay(value, 1.5)).coerceIn(0.0, 1.0)

    private fun normalizeRobustAcceleration(value: Double): Double =
        (1.0 - expDecay(value, 8.0)).coerceIn(0.0, 1.0)

    private fun normalizeRobustJerk(value: Double): Double =
        (1.0 - expDecay(value, 50.0)).coerceIn(0.0, 1.0)

    private fun expDecay(
        value: Double,
        scale: Double,
    ): Double =
        kotlin.math.exp(
            -value.coerceAtLeast(0.0) /
                scale.coerceAtLeast(1e-6),
        )

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val middle = values.size / 2
        return if (values.size % 2 == 0) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    }

    private fun finite(value: Double): Double =
        if (value.isFinite()) {
            value.coerceAtLeast(0.0)
        } else {
            min(1.0, value.takeUnless { it.isNaN() }?.let { abs(it) } ?: 0.0)
        }

    @Synchronized
    override fun close() {
        history.clear()
    }
}
