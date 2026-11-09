package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp

/**
 * Converts measured Camera2 rolling-shutter timing into bounded per-observation evidence.
 *
 * The result is deliberately expressed as a fraction of the frame interval, not as an
 * arbitrary pixel correction. Downstream geometry can therefore preserve the physical timing
 * information while deciding how strongly to trust an observation.
 */
class RollingShutterEvidenceEngine(
    private val config: Config = Config(),
) {
    private val exposureTimingEngine =
        ExposureMidpointTimingEngine()
    data class Config(
        val minimumFrameDurationNanos: Long = 1_000_000L,
        val maximumReadoutRatio: Double = 1.0,
        val rowPenaltyScale: Double = 1.15,
        val minimumConfidenceScale: Double = 0.08,
    ) {
        init {
            require(minimumFrameDurationNanos > 0L)
            require(maximumReadoutRatio > 0.0)
            require(rowPenaltyScale > 0.0)
            require(minimumConfidenceScale in 0.0..1.0)
        }
    }

    data class Input(
        val yPx: Double,
        val frameHeightPx: Int,
        val normalizedRow: Double? = null,
        val rollingShutterSkewNanos: Long,
        val frameDurationNanos: Long,
        val exposureTimeNanos: Long,
        val captureTimingConfidence: Double,
        /**
         * Full active-array height when available from Camera2 characteristics.
         * Zero means unavailable and preserves the frame-space timing path.
         */
        val activeArrayHeightPx: Int = 0,
        /**
         * Vertical crop of the sensor used by this output stream.
         */
        val viewportTopPx: Int = 0,
        val viewportHeightPx: Int = 0,
    ) {
        init {
            require(yPx.isFinite())
            require(frameHeightPx > 0)
            normalizedRow?.let {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(rollingShutterSkewNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(exposureTimeNanos >= 0L)
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
        val normalizedRow: Double,
        val rowOffsetNanos: Long,
        val rollingShutterFraction: Double,
        val readoutRatio: Double,
        val rowTemporalBurden: Double,
        val confidenceScale: Double,
        val warnings: List<String>,
        val exposureMidpointOffsetNanos: Long = 0L,
        val effectiveCaptureOffsetNanos: Long = 0L,
        val timingUncertaintyNanos: Long = 0L,
        val motionBlurBurden: Double = 0.0,
    )

    fun evaluate(input: Input): Result {
        val frameRow = (
            input.normalizedRow
                ?: (
                    input.yPx /
                        input.frameHeightPx.toDouble()
                    )
            ).coerceIn(0.0, 1.0)

        val timingAuthority = exposureTimingEngine.evaluate(
            ExposureMidpointTimingEngine.Input(
                firstRowExposureStartNanos = 0L,
                frameDurationNanos = input.frameDurationNanos,
                exposureTimeNanos = input.exposureTimeNanos,
                rollingShutterSkewNanos = input.rollingShutterSkewNanos,
                rowFraction = frameRow,
                captureTimingConfidence = input.captureTimingConfidence,
                activeArrayHeightPx = input.activeArrayHeightPx,
                viewportTopPx = input.viewportTopPx,
                viewportHeightPx = input.viewportHeightPx,
            ),
        )

        val viewportAware =
            input.activeArrayHeightPx > 0 &&
                input.viewportHeightPx > 0

        val baseSkewNanos = input.rollingShutterSkewNanos
        val scaledSkewNanos =
            if (viewportAware) {
                (
                    baseSkewNanos.toDouble() *
                        input.viewportHeightPx.toDouble() /
                        input.activeArrayHeightPx.toDouble()
                    ).toLong()
            } else {
                baseSkewNanos
            }

        val sensorNormalizedRow =
            if (viewportAware) {
                (
                    input.viewportTopPx.toDouble() +
                        frameRow * input.viewportHeightPx.toDouble()
                    ) / input.activeArrayHeightPx.toDouble()
            } else {
                frameRow
            }.coerceIn(0.0, 1.0)

        val row = if (viewportAware) {
            (
                sensorNormalizedRow -
                    input.viewportTopPx.toDouble() /
                        input.activeArrayHeightPx.toDouble()
                ) / (
                    input.viewportHeightPx.toDouble() /
                        input.activeArrayHeightPx.toDouble()
                    ).coerceAtLeast(1e-9)
        } else {
            frameRow
        }.coerceIn(0.0, 1.0)

        if (scaledSkewNanos == 0L) {
            return Result(
                normalizedRow = row,
                rowOffsetNanos = 0L,
                rollingShutterFraction = 0.0,
                readoutRatio = 0.0,
                rowTemporalBurden = 0.0,
                confidenceScale =
                    (
                        input.captureTimingConfidence *
                            timingAuthority.confidenceScale
                        ).coerceIn(
                            config.minimumConfidenceScale,
                            1.0,
                        ),
                warnings = (
                    listOf("ROLLING_SHUTTER_SKEW_UNAVAILABLE") +
                        timingAuthority.warnings
                ).distinct(),
                exposureMidpointOffsetNanos =
                    timingAuthority.rowOffsetNanos +
                        timingAuthority.exposureHalfWindowNanos,
                effectiveCaptureOffsetNanos =
                    timingAuthority.rowOffsetNanos +
                        timingAuthority.exposureHalfWindowNanos,
                timingUncertaintyNanos =
                    timingAuthority.timingUncertaintyNanos,
                motionBlurBurden =
                    timingAuthority.motionBlurBurden,
            )
        }

        val centeredRow = (row - 0.5) * 2.0
        val rowOffset = (
            centeredRow *
                0.5 *
                scaledSkewNanos.toDouble()
        ).toLong()

        val frameDuration = maxOf(
            input.frameDurationNanos,
            config.minimumFrameDurationNanos,
        ).toDouble()
        val exposure = input.exposureTimeNanos.toDouble()

        val readoutRatio = (
            scaledSkewNanos.toDouble() / frameDuration
        ).coerceAtLeast(0.0)

        val timingDenominator = maxOf(
            frameDuration,
            exposure,
            config.minimumFrameDurationNanos.toDouble(),
        )

        val rowTemporalBurden = (
            abs(rowOffset.toDouble()) / timingDenominator
        ).coerceAtLeast(0.0)

        val readoutExcess = (
            readoutRatio - config.maximumReadoutRatio
        ).coerceAtLeast(0.0)

        val rowPenalty = exp(
            -config.rowPenaltyScale * rowTemporalBurden
        ).coerceIn(config.minimumConfidenceScale, 1.0)

        val excessPenalty = exp(
            -2.0 * readoutExcess
        ).coerceIn(config.minimumConfidenceScale, 1.0)

        val confidence = (
            input.captureTimingConfidence *
                rowPenalty *
                excessPenalty *
                timingAuthority.confidenceScale
        ).coerceIn(config.minimumConfidenceScale, 1.0)

        val fractionDenominator = maxOf(
            frameDuration,
            scaledSkewNanos.toDouble(),
            1.0,
        )

        val rollingFraction = (
            rowOffset.toDouble() / fractionDenominator
        ).coerceIn(-1.0, 1.0)

        val warnings = buildList {
            addAll(timingAuthority.warnings)
            if (input.captureTimingConfidence < 0.60) {
                add("CAPTURE_TIMING_CONFIDENCE_DEGRADED")
            }
            if (readoutRatio > config.maximumReadoutRatio) {
                add("ROLLING_SHUTTER_READOUT_RATIO_HIGH")
            }
            if (rowTemporalBurden > 0.35) {
                add("ROW_TIME_OFFSET_HIGH")
            }
        }

        return Result(
            normalizedRow = row,
            rowOffsetNanos = rowOffset,
            rollingShutterFraction = rollingFraction,
            readoutRatio = readoutRatio.coerceAtMost(Double.MAX_VALUE),
            rowTemporalBurden = rowTemporalBurden.coerceIn(0.0, 1.0),
            confidenceScale = confidence,
            warnings = warnings.distinct(),
            exposureMidpointOffsetNanos =
                timingAuthority.exposureMidpointNanos,
            effectiveCaptureOffsetNanos =
                timingAuthority.effectiveCaptureNanos,
            timingUncertaintyNanos =
                timingAuthority.timingUncertaintyNanos,
            motionBlurBurden =
                timingAuthority.motionBlurBurden,
        )
    }
}
