package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Measurement-oriented timing authority for rolling-shutter cameras.
 *
 * Camera2 exposes the sensor timestamp at the start of exposure of the first row and,
 * when available, the rolling-shutter skew across the active image. Treating a frame as if
 * every pixel were acquired at one instant introduces a systematic temporal error during fast
 * motion. This engine converts those observables into an effective exposure-center timestamp,
 * per-row timing offset, motion-blur burden and a bounded confidence factor.
 *
 * It intentionally performs no undocumented de-warping. Geometry stages consume the timing
 * evidence and decide how much correction their own motion model can support.
 */
class ExposureMidpointTimingEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumFrameDurationNanos: Long = 1_000_000L,
        val minimumExposureNanos: Long = 100_000L,
        val minimumConfidenceScale: Double = 0.05,
        val readoutToExposurePenalty: Double = 0.55,
        val rowOffsetPenaltyScale: Double = 1.10,
        val exposureBlurPenaltyScale: Double = 0.70,
        val motionConditioningThreshold: Double = 0.20,
        val hardReadoutRatio: Double = 1.25,
    ) {
        init {
            require(minimumFrameDurationNanos > 0L)
            require(minimumExposureNanos > 0L)
            require(minimumConfidenceScale in 0.0..1.0)
            require(readoutToExposurePenalty >= 0.0)
            require(rowOffsetPenaltyScale >= 0.0)
            require(exposureBlurPenaltyScale >= 0.0)
            require(motionConditioningThreshold in 0.0..1.0)
            require(hardReadoutRatio > 0.0)
        }
    }

    data class Input(
        val firstRowExposureStartNanos: Long,
        val frameDurationNanos: Long,
        val exposureTimeNanos: Long,
        val rollingShutterSkewNanos: Long,
        val rowFraction: Double,
        val captureTimingConfidence: Double,
        val activeArrayHeightPx: Int = 0,
        val viewportTopPx: Int = 0,
        val viewportHeightPx: Int = 0,
    ) {
        init {
            require(firstRowExposureStartNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(rollingShutterSkewNanos >= 0L)
            require(rowFraction.isFinite() && rowFraction in 0.0..1.0)
            require(captureTimingConfidence in 0.0..1.0)
            require(activeArrayHeightPx >= 0)
            require(viewportTopPx >= 0)
            require(viewportHeightPx >= 0)
            if (activeArrayHeightPx > 0 && viewportHeightPx > 0) {
                require(viewportTopPx < activeArrayHeightPx)
                require(viewportTopPx + viewportHeightPx <= activeArrayHeightPx)
            }
        }
    }

    data class Result(
        val sensorRowFraction: Double,
        val exposureStartNanos: Long,
        val exposureMidpointNanos: Long,
        val effectiveCaptureNanos: Long,
        val rowOffsetNanos: Long,
        val exposureHalfWindowNanos: Long,
        val readoutRatio: Double,
        val exposureToFrameRatio: Double,
        val rowTemporalBurden: Double,
        val motionBlurBurden: Double,
        val timingUncertaintyNanos: Long,
        val confidenceScale: Double,
        val hardFailure: Boolean,
        val warnings: List<String>,
    )

    fun evaluate(input: Input): Result {
        val row = input.rowFraction.coerceIn(0.0, 1.0)
        val viewportAware =
            input.activeArrayHeightPx > 0 &&
                input.viewportHeightPx > 0

        val sensorRow =
            if (viewportAware) {
                (
                    input.viewportTopPx.toDouble() +
                        row * input.viewportHeightPx.toDouble()
                    ) /
                    input.activeArrayHeightPx.toDouble()
            } else {
                row
            }.coerceIn(0.0, 1.0)

        val streamRow =
            if (viewportAware) {
                (
                    sensorRow -
                        input.viewportTopPx.toDouble() /
                            input.activeArrayHeightPx.toDouble()
                    ) /
                    (
                        input.viewportHeightPx.toDouble() /
                            input.activeArrayHeightPx.toDouble()
                    ).coerceAtLeast(1e-9)
            } else {
                row
            }.coerceIn(0.0, 1.0)

        val frameDuration =
            max(
                input.frameDurationNanos,
                config.minimumFrameDurationNanos,
            )
        val exposure =
            max(
                input.exposureTimeNanos,
                config.minimumExposureNanos,
            )

        val readout =
            input.rollingShutterSkewNanos.toDouble() *
                if (viewportAware) {
                    input.viewportHeightPx.toDouble() /
                        input.activeArrayHeightPx.toDouble()
                } else {
                    1.0
                }

        val readoutNanos =
            readout.toLong().coerceAtLeast(0L)

        val rowOffset =
            (
                (streamRow - 0.5) *
                    readoutNanos.toDouble()
            ).toLong()

        val exposureStart =
            input.firstRowExposureStartNanos +
                rowOffset

        val midpoint =
            exposureStart +
                exposure / 2L

        val readoutRatio =
            readoutNanos.toDouble() /
                frameDuration.toDouble()

        val exposureFrameRatio =
            exposure.toDouble() /
                frameDuration.toDouble()

        val rowBurden =
            (
                abs(rowOffset.toDouble()) /
                    max(
                        frameDuration,
                        exposure,
                    ).toDouble()
            ).coerceIn(0.0, 4.0)

        val blurBurden =
            (
                exposure.toDouble() /
                    max(
                        frameDuration,
                        config.minimumFrameDurationNanos,
                    ).toDouble()
            ).coerceIn(0.0, 4.0)

        val readoutBurden =
            (
                readoutNanos.toDouble() /
                    max(
                        exposure,
                        config.minimumExposureNanos,
                    ).toDouble()
            ).coerceIn(0.0, 64.0)

        val readoutPenalty =
            exp(
                -config.readoutToExposurePenalty *
                    max(
                        0.0,
                        readoutBurden -
                            config.motionConditioningThreshold,
                    ),
            )

        val rowPenalty =
            exp(
                -config.rowOffsetPenaltyScale *
                    rowBurden,
            )

        val blurPenalty =
            exp(
                -config.exposureBlurPenaltyScale *
                    max(
                        0.0,
                        blurBurden -
                            config.motionConditioningThreshold,
                    ),
            )

        val confidence =
            (
                input.captureTimingConfidence *
                    readoutPenalty *
                    rowPenalty *
                    blurPenalty
                ).coerceIn(
                    config.minimumConfidenceScale,
                    1.0,
                )

        val uncertainty =
            (
                max(
                    exposure,
                    input.rollingShutterSkewNanos,
                ) / 2L +
                    abs(rowOffset) +
                    (
                        frameDuration.toDouble() *
                            (1.0 - input.captureTimingConfidence)
                    ).toLong()
                ).coerceAtMost(
                    Long.MAX_VALUE / 4L,
                )

        val hardFailure =
            input.captureTimingConfidence < 0.05 ||
                readoutRatio > config.hardReadoutRatio ||
                !midpoint.isFiniteTimestamp()

        val warnings =
            buildList {
                if (input.captureTimingConfidence < 0.60) {
                    add("CAPTURE_TIMING_CONFIDENCE_DEGRADED")
                }
                if (input.rollingShutterSkewNanos == 0L) {
                    add("ROLLING_SHUTTER_SKEW_UNAVAILABLE")
                }
                if (readoutRatio > 1.0) {
                    add("READOUT_LONGER_THAN_FRAME_PERIOD")
                }
                if (blurBurden > 0.50) {
                    add("EXPOSURE_TIME_MOTION_BLUR_BURDEN_HIGH")
                }
                if (abs(rowOffset) > frameDuration / 4L) {
                    add("ROW_TIME_OFFSET_DOMINATES_FRAME_TIMING")
                }
                if (hardFailure) {
                    add("TIMING_AUTHORITY_HARD_FAILURE")
                }
            }

        return Result(
            sensorRowFraction = sensorRow,
            exposureStartNanos = exposureStart,
            exposureMidpointNanos = midpoint,
            effectiveCaptureNanos = midpoint,
            rowOffsetNanos = rowOffset,
            exposureHalfWindowNanos = exposure / 2L,
            readoutRatio = readoutRatio.coerceAtLeast(0.0),
            exposureToFrameRatio =
                exposureFrameRatio.coerceAtLeast(0.0),
            rowTemporalBurden = rowBurden.coerceIn(0.0, 1.0),
            motionBlurBurden = blurBurden.coerceIn(0.0, 1.0),
            timingUncertaintyNanos = uncertainty,
            confidenceScale = confidence,
            hardFailure = hardFailure,
            warnings = warnings,
        )
    }

    private fun Long.isFiniteTimestamp(): Boolean =
        this >= 0L
}
