package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.ImuSample
import com.nivukx.nox.core.model.VioPose
import kotlin.math.abs

/**
 * Stateful bridge between raw IMU timing and visual/VIO timing.
 *
 * An accepted alignment result is smoothed into a bounded EWMA. The estimated offset is diagnostic
 * evidence for sensor fusion and scheduling; authoritative sensor timestamps remain untouched.
 */
class CameraImuSynchronizationEngine(
    private val estimatorConfig:
        ImuVisualMotionAlignmentEngine.Config =
        ImuVisualMotionAlignmentEngine.Config(),
    private val offsetSmoothingAlpha: Double = 0.34,
) {

    init {
        require(offsetSmoothingAlpha in 0.05..0.95)
    }

    data class Result(
        val estimatedOffsetNanos: Double,
        val instantaneousOffsetNanos: Double,
        val confidence: Double,
        val correlation: Double,
        val accepted: Boolean,
        val warnings: List<String>,
        val sampleCount: Int,
    )

    private val estimator =
        ImuVisualMotionAlignmentEngine(
            estimatorConfig,
        )

    private var estimatedOffsetNanos = 0.0
    private var confidence = 0.0

    @Synchronized
    fun estimate(
        imu: List<ImuSample>,
        visual: List<VioPose>,
    ): Result {
        val imuSamples =
            imu
                .filter {
                    it.timestampNanos >= 0L
                }
                .sortedBy {
                    it.timestampNanos
                }
                .map {
                    ImuVisualMotionAlignmentEngine.Sample(
                        timestampNanos =
                            it.timestampNanos,
                        angularSpeedRadiansPerSecond =
                            kotlin.math.sqrt(
                                it.x * it.x +
                                    it.y * it.y +
                                    it.z * it.z,
                            ),
                    )
                }

        val visualSamples =
            visual
                .filter {
                    it.timestampNanos >= 0L &&
                        it.confidence > 0.0 &&
                        it.angularVelocityRadiansPerSecond.x
                            .isFinite() &&
                        it.angularVelocityRadiansPerSecond.y
                            .isFinite() &&
                        it.angularVelocityRadiansPerSecond.z
                            .isFinite()
                }
                .sortedBy {
                    it.timestampNanos
                }
                .map {
                    val angular =
                        it.angularVelocityRadiansPerSecond

                    ImuVisualMotionAlignmentEngine.Sample(
                        timestampNanos =
                            it.timestampNanos,
                        angularSpeedRadiansPerSecond =
                            angular.norm(),
                    )
                }

        val alignment =
            estimator.estimate(
                imu = imuSamples,
                visual = visualSamples,
            )

        if (
            alignment.accepted &&
            alignment.offsetNanos.isFinite()
        ) {
            estimatedOffsetNanos +=
                (
                    alignment.offsetNanos -
                        estimatedOffsetNanos
                    ) *
                    offsetSmoothingAlpha

            confidence =
                if (confidence <= 0.0) {
                    alignment.confidence.coerceIn(0.0, 1.0)
                } else {
                    (
                        confidence * 0.30 +
                            maxOf(
                                alignment.confidence,
                                alignment.correlation * 0.72,
                            ) * 0.70
                        ).coerceIn(
                        0.0,
                        1.0,
                    )
                }
        } else {
            confidence *= 0.90
        }

        return Result(
            estimatedOffsetNanos =
                if (confidence >= 0.20) {
                    estimatedOffsetNanos
                } else {
                    0.0
                },
            instantaneousOffsetNanos =
                if (alignment.accepted) {
                    alignment.offsetNanos
                } else {
                    0.0
                },
            confidence = confidence,
            correlation = alignment.correlation,
            accepted =
                alignment.accepted &&
                    confidence >= 0.30,
            warnings =
                alignment.warnings +
                    listOfNotNull(
                        if (
                            confidence < 0.30 &&
                            abs(
                                estimatedOffsetNanos,
                            ) > 0.0
                        ) {
                            "camera-imu-offset-confidence-is-low"
                        } else {
                            null
                        },
                    ),
            sampleCount =
                minOf(
                    imuSamples.size,
                    visualSamples.size,
                ),
        )
    }

    @Synchronized
    fun reset() {
        estimatedOffsetNanos = 0.0
        confidence = 0.0
    }

    @Synchronized
    fun currentOffsetNanos(): Double =
        estimatedOffsetNanos

    @Synchronized
    fun currentConfidence(): Double =
        confidence
}
