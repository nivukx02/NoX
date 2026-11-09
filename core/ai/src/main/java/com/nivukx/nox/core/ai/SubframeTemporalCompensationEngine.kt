package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.roundToLong
import kotlin.math.min

/**
 * Deterministic timing-evidence layer for high-end temporal motion inference.
 *
 * A camera frame is not an instantaneous observation: rolling-shutter readout, exposure integration,
 * frame cadence, sensor timestamp jitter, and inter-device skew all contribute to the effective time
 * represented by a pixel landmark. This engine converts those quantities into a conservative
 * motion-context correction.
 *
 * It does not invent timestamps. It only derives an effective exposure-center offset and uncertainty
 * from supplied camera metadata. The downstream motion estimator can then reduce temporal trust when
 * the capture conditions are poorly qualified.
 */
class SubframeTemporalCompensationEngine(
    private val configuration: Configuration = Configuration(),
) {
    data class Configuration(
        val maximumSupportedRollingSkewNanos: Long = 40_000_000L,
        val maximumSupportedExposureNanos: Long = 50_000_000L,
        val referenceJitterNanos: Long = 350_000L,
        val minimumTrustScale: Double = 0.25,
        val highMotionRiskThreshold: Double = 0.72,
    ) {
        init {
            require(maximumSupportedRollingSkewNanos > 0L)
            require(maximumSupportedExposureNanos > 0L)
            require(referenceJitterNanos > 0L)
            require(minimumTrustScale in 0.0..1.0)
            require(highMotionRiskThreshold in 0.0..1.0)
        }
    }

    data class Input(
        val frameTimestampNanos: Long,
        val frameDurationNanos: Long,
        val exposureTimeNanos: Long,
        val rollingShutterSkewNanos: Long,
        val captureTimingConfidence: Double,
        val sensorConfidence: Double,
        val cameraMotionRisk: Double,
        val globalMotionConfidence: Double,
    ) {
        init {
            require(frameTimestampNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(rollingShutterSkewNanos >= 0L)
            require(captureTimingConfidence in 0.0..1.0)
            require(sensorConfidence in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(globalMotionConfidence in 0.0..1.0)
        }
    }

    data class Result(
        val sourceTimestampNanos: Long,
        val effectiveExposureCenterTimestampNanos: Long,
        val effectiveTimestampOffsetNanos: Long,
        val rollingShutterRisk: Double,
        val exposureBlurRisk: Double,
        val timestampJitterRisk: Double,
        val timingUncertaintyNanos: Long,
        val timingConfidence: Double,
        val measurementTrustScale: Double,
        val highMotionTimingRisk: Boolean,
        val warnings: List<String>,
    )

    fun compensate(input: Input): Result {
        val frameDuration =
            input.frameDurationNanos.coerceAtLeast(1L)
        val exposure =
            input.exposureTimeNanos
                .coerceIn(0L, configuration.maximumSupportedExposureNanos)

        val exposureCenterOffset =
            exposure / 2L

        val rollingOffset =
            min(
                input.rollingShutterSkewNanos / 2L,
                configuration.maximumSupportedRollingSkewNanos / 2L,
            )

        val signedOffset =
            exposureCenterOffset + rollingOffset

        val effectiveTimestamp =
            safeAdd(
                input.frameTimestampNanos,
                signedOffset,
            )

        val rollingRatio =
            (
                input.rollingShutterSkewNanos.toDouble() /
                    configuration.maximumSupportedRollingSkewNanos.toDouble()
                ).coerceIn(0.0, 1.0)

        val exposureRatio =
            (
                exposure.toDouble() /
                    frameDuration.toDouble()
                ).coerceIn(0.0, 1.0)

        val exposureBlurRisk =
            smoothStep(
                exposureRatio,
                0.10,
                0.65,
            )

        val timestampJitterRisk =
            (
                1.0 -
                    input.captureTimingConfidence
                ).coerceIn(0.0, 1.0) *
                0.72

        val timingUncertainty =
            max(
                configuration.referenceJitterNanos.toDouble(),
                (
                    configuration.referenceJitterNanos +
                        input.rollingShutterSkewNanos * 0.24 +
                        exposure * 0.18
                    ) *
                    (
                        1.0 +
                            (1.0 - input.sensorConfidence) * 1.8
                    ),
            )
                .roundToLong()
                .coerceIn(
                    configuration.referenceJitterNanos,
                    configuration.maximumSupportedRollingSkewNanos * 2L,
                )

        val motionCoupling =
            (
                input.cameraMotionRisk *
                    (0.62 + 0.38 * input.globalMotionConfidence)
                ).coerceIn(0.0, 1.0)

        val rollingRisk =
            (
                rollingRatio * 0.62 +
                    exposureBlurRisk * 0.18 +
                    timestampJitterRisk * 0.20
                ).coerceIn(0.0, 1.0)

        val trust =
            (
                input.captureTimingConfidence * 0.34 +
                    input.sensorConfidence * 0.28 +
                    (1.0 - rollingRisk) * 0.22 +
                    (1.0 - motionCoupling) * 0.16
                ).coerceIn(
                    configuration.minimumTrustScale,
                    1.0,
                )

        val highMotionTimingRisk =
            (
                rollingRisk >=
                    configuration.highMotionRiskThreshold
                    &&
                    (
                        motionCoupling >= 0.45 ||
                            exposureBlurRisk >= 0.72
                    )
            )

        val warnings =
            buildList {
                if (input.captureTimingConfidence < 0.80) {
                    add("SUBFRAME_TIMING_CONFIDENCE_LOW")
                }
                if (input.sensorConfidence < 0.80) {
                    add("SUBFRAME_SENSOR_CONFIDENCE_LOW")
                }
                if (rollingRatio >= 0.75) {
                    add("ROLLING_SHUTTER_SKEW_HIGH")
                }
                if (exposureBlurRisk >= 0.70) {
                    add("EXPOSURE_INTEGRATION_HIGH")
                }
                if (highMotionTimingRisk) {
                    add("HIGH_MOTION_SUBFRAME_TIMING_RISK")
                }
            }

        return Result(
            sourceTimestampNanos =
                input.frameTimestampNanos,
            effectiveExposureCenterTimestampNanos =
                effectiveTimestamp,
            effectiveTimestampOffsetNanos =
                signedOffset,
            rollingShutterRisk = rollingRisk,
            exposureBlurRisk = exposureBlurRisk,
            timestampJitterRisk = timestampJitterRisk,
            timingUncertaintyNanos =
                timingUncertainty,
            timingConfidence =
                trust,
            measurementTrustScale =
                trust,
            highMotionTimingRisk =
                highMotionTimingRisk,
            warnings =
                warnings.distinct(),
        )
    }

    private fun smoothStep(
        value: Double,
        edge0: Double,
        edge1: Double,
    ): Double {
        if (value <= edge0) return 0.0
        if (value >= edge1) return 1.0
        val t =
            (
                (value - edge0) /
                    (edge1 - edge0)
                ).coerceIn(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    }

    private fun safeAdd(
        timestampNanos: Long,
        offsetNanos: Long,
    ): Long =
        if (offsetNanos > 0L &&
            timestampNanos > Long.MAX_VALUE - offsetNanos
        ) {
            Long.MAX_VALUE
        } else {
            timestampNanos + offsetNanos
        }
}
