package com.nivukx.nox.core.capture

import com.nivukx.nox.core.model.CameraFocusMode
import com.nivukx.nox.core.model.FrameTelemetry
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * Evaluates the trustworthiness of camera metadata driving the motion pipeline.
 *
 * This does not judge image aesthetics. It measures whether the timing, exposure, rolling-shutter
 * and focus metadata are sufficiently coherent for geometric reconstruction and temporal fusion.
 */
class CameraCaptureQualityEngine {

    data class Result(
        val timestampNanos: Long,
        val score: Double,
        val cadenceFps: Double,
        val exposureFraction: Double,
        val rollingShutterFraction: Double,
        val focusConfidence: Double,
        val hardwareReadoutTimingConfidence: Double = 0.0,
        val readoutDelayNanos: Long? = null,
        val metadataCompleteness: Double,
        val analysisDropRatio: Double = 0.0,
        val opticalBlurRisk: Double = 0.0,
        val hardFailure: Boolean,
        val warnings: List<String>,
    )

    fun evaluate(
        telemetry: FrameTelemetry,
        targetFps: Int,
        focusMode: CameraFocusMode,
        analysisDropRatio: Double = 0.0,
    ): Result {
        require(
            analysisDropRatio.isFinite() &&
                analysisDropRatio in 0.0..1.0,
        )
        val frameDuration = telemetry.frameDurationNanos
        val exposure = telemetry.exposureTimeNanos
        val skew = telemetry.rollingShutterSkewNanos

        val cadenceFps =
            if (frameDuration > 0L) {
                1_000_000_000.0 / frameDuration.toDouble()
            } else {
                0.0
            }

        val exposureFraction =
            if (frameDuration > 0L) {
                exposure.toDouble() / frameDuration.toDouble()
            } else {
                Double.POSITIVE_INFINITY
            }

        val rollingFraction =
            if (frameDuration > 0L) {
                abs(skew).toDouble() / frameDuration.toDouble()
            } else {
                Double.POSITIVE_INFINITY
            }

        val cadenceError =
            if (cadenceFps > 0.0) {
                abs(cadenceFps - targetFps.toDouble()) /
                    max(targetFps.toDouble(), 1.0)
            } else {
                1.0
            }

        val cadenceScore =
            exp(-cadenceError * 7.0)
                .coerceIn(0.0, 1.0)

        val exposureScore =
            when {
                !exposureFraction.isFinite() ||
                    exposureFraction <= 0.0 -> 0.0

                exposureFraction <= 0.50 -> 1.0

                exposureFraction <= 0.85 ->
                    (1.0 - (exposureFraction - 0.50) / 0.35)
                        .coerceIn(0.20, 1.0)

                else ->
                    (1.0 - (exposureFraction - 0.85) / 0.75)
                        .coerceIn(0.05, 0.20)
            }

        val rollingScore =
            when {
                !rollingFraction.isFinite() -> 0.0
                rollingFraction <= 0.20 -> 1.0
                rollingFraction <= 0.65 ->
                    (1.0 - (rollingFraction - 0.20) / 0.45)
                        .coerceIn(0.25, 1.0)
                else ->
                    (1.0 - (rollingFraction - 0.65) / 1.0)
                        .coerceIn(0.05, 0.25)
            }

        val readoutDelayNanos =
            telemetry.readoutTimestampNanos?.let {
                it - telemetry.sensorTimestampNanos
            }

        val hardwareReadoutTimingConfidence =
            when {
                telemetry.readoutTimestampNanos == null -> 0.0
                telemetry.readoutTimestampNanos < telemetry.sensorTimestampNanos ->
                    0.10
                telemetry.exposureTimeNanos > 0L &&
                    readoutDelayNanos != null &&
                    readoutDelayNanos < telemetry.exposureTimeNanos ->
                    0.35
                else -> 0.98
            }

        val focusConfidence =
            when (focusMode) {
                CameraFocusMode.MANUAL ->
                    if (telemetry.lensFocusDistanceDiopters != null) {
                        1.0
                    } else {
                        0.30
                    }

                CameraFocusMode.CONTINUOUS,
                CameraFocusMode.TRACKED_SUBJECT ->
                    when {
                        telemetry.lensFocusDistanceDiopters == null &&
                            telemetry.afState == null -> 0.55

                        telemetry.lensFocusDistanceDiopters != null &&
                            telemetry.afState != null -> 1.0

                        else -> 0.76
                    }
            }

        val completeness =
            listOf(
                telemetry.sensorTimestampNanos > 0L,
                telemetry.monotonicTimestampNanos > 0L,
                telemetry.frameDurationNanos > 0L,
                telemetry.exposureTimeNanos > 0L,
                telemetry.sensorSensitivityIso > 0,
                telemetry.afState != null,
                telemetry.aeState != null,
                telemetry.awbState != null,
            ).count { it }.toDouble() / 8.0

        val opticalBlurRisk =
            (1.0 - exposureScore)
                .coerceIn(0.0, 1.0)

        val warnings = mutableListOf<String>()

        if (cadenceFps <= 0.0) {
            warnings += "camera-frame-duration-invalid"
        } else if (cadenceError > 0.12) {
            warnings += "camera-cadence-below-target"
        }

        if (exposureFraction > 0.85) {
            warnings += "exposure-window-high-motion-blur-risk"
        }

        if (rollingFraction > 0.65) {
            warnings += "rolling-shutter-dominant"
        }

        if (focusMode == CameraFocusMode.MANUAL &&
            telemetry.lensFocusDistanceDiopters == null
        ) {
            warnings += "manual-focus-distance-unavailable"
        }

        if (completeness < 0.75) {
            warnings += "camera-metadata-incomplete"
        }

        if (analysisDropRatio > 0.02) {
            warnings += "analysis-backpressure-present"
        }
        if (analysisDropRatio > 0.10) {
            warnings += "analysis-backpressure-high"
        }

        if (telemetry.readoutTimestampNanos == null) {
            warnings += "hardware-readout-timestamp-unavailable"
        } else if (hardwareReadoutTimingConfidence < 0.50) {
            warnings += "hardware-readout-timestamp-invalid"
        }

        val score =
            (
                cadenceScore * 0.28 +
                    exposureScore * 0.20 +
                    rollingScore * 0.20 +
                    focusConfidence * 0.14 +
                    completeness * 0.10 +
                    (
                        if (telemetry.readoutTimestampNanos == null) {
                            0.72
                        } else {
                            hardwareReadoutTimingConfidence
                        }
                    ) * 0.07 +
                    (
                        1.0 -
                            analysisDropRatio
                    ).coerceIn(0.0, 1.0) * 0.10
            ).coerceIn(0.0, 1.0)

        val hardFailure =
            frameDuration <= 0L ||
                exposure <= 0L ||
                telemetry.sensorTimestampNanos <= 0L ||
                score < 0.20 ||
                analysisDropRatio > 0.25

        return Result(
            timestampNanos = telemetry.sensorTimestampNanos,
            score = score,
            cadenceFps = cadenceFps,
            exposureFraction = exposureFraction.coerceAtLeast(0.0),
            rollingShutterFraction = rollingFraction.coerceAtLeast(0.0),
            focusConfidence = focusConfidence,
            analysisDropRatio = analysisDropRatio,
            opticalBlurRisk = opticalBlurRisk,
            hardwareReadoutTimingConfidence =
                hardwareReadoutTimingConfidence,
            readoutDelayNanos = readoutDelayNanos,
            metadataCompleteness = completeness,
            hardFailure = hardFailure,
            warnings = warnings.distinct().take(8),
        )
    }

    fun reset() = Unit
}
