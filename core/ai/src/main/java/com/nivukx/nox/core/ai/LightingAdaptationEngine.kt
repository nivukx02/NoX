
package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.CameraTelemetryState
import com.nivukx.nox.core.model.LightingStabilityBand
import com.nivukx.nox.core.model.LightingState
import com.nivukx.nox.core.model.PerceptionSource
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Derives capture-quality lighting state from camera telemetry.
 *
 * NoX records lighting metadata, not image frames. This engine therefore turns exposure, ISO,
 * frame duration, focus and frame-to-frame telemetry into a compact lighting state that can guide
 * inference budgets and reconstruction confidence without persisting RGB content.
 */
class LightingAdaptationEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val lowLux: Double = 8.0,
        val highLux: Double = 25_000.0,
        val targetExposureDuty: Double = 0.55,
        val dutyTolerance: Double = 0.22,
        val cctMinimumKelvin: Double = 1800.0,
        val cctMaximumKelvin: Double = 15_000.0,
        val flickerWindowFrames: Int = 18,
        val flickerVarianceThreshold: Double = 0.035,
        val temporalHalfLifeFrames: Double = 8.0,
    )

    data class Sample(
        val timestampNanos: Long,
        val lux: Double,
        val cctKelvin: Double,
        val exposureTimeNanos: Long,
        val frameDurationNanos: Long,
        val iso: Int,
        val focusConfidence: Double,
        val cameraMotionRisk: Double,
    )

    data class Adaptation(
        val lighting: LightingState,
        val inferenceScale: Double,
        val depthScale: Double,
        val faceScale: Double,
        val eyeScale: Double,
        val handScale: Double,
        val warnings: List<String>,
        val diagnostics: List<String>,
    )

    private val luxHistory = ArrayDeque<Double>()
    private val cctHistory = ArrayDeque<Double>()
    private var lastTimestampNanos = Long.MIN_VALUE
    private var previousLux = Double.NaN

    @Synchronized
    fun update(
        timestampNanos: Long,
        telemetry: CameraTelemetryState,
        estimatedLux: Double? = null,
    ): Adaptation {
        val lux = (
            estimatedLux
                ?.takeIf { it.isFinite() && it >= 0.0 }
                ?: inferLux(telemetry)
            ).coerceIn(0.0, 100_000.0)

        val cct = inferCct(
            previous = cctHistory.lastOrNull(),
            telemetry = telemetry,
        )

        pushHistory(luxHistory, lux)
        pushHistory(cctHistory, cct)

        val duty = if (telemetry.frameDurationNanos > 0L) {
            telemetry.exposureTimeNanos.toDouble() /
                telemetry.frameDurationNanos.toDouble()
        } else {
            0.0
        }

        val dutyRisk = (
            abs(duty - config.targetExposureDuty) /
                config.dutyTolerance.coerceAtLeast(1e-4)
            ).coerceIn(0.0, 1.0)

        val lowLightRisk = when {
            lux <= 0.0 -> 1.0
            lux < config.lowLux -> (
                ln(config.lowLux / lux.coerceAtLeast(0.1)) /
                    ln(config.lowLux / 0.1)
                ).coerceIn(0.0, 1.0)
            else -> 0.0
        }

        val highlightRisk = when {
            lux <= config.highLux -> 0.0
            else -> (
                ln(lux / config.highLux) /
                    ln(100_000.0 / config.highLux)
                ).coerceIn(0.0, 1.0)
        }

        val cctRisk = when {
            cct < config.cctMinimumKelvin -> (
                (config.cctMinimumKelvin - cct) /
                    config.cctMinimumKelvin
                ).coerceIn(0.0, 1.0)
            cct > config.cctMaximumKelvin -> (
                (cct - config.cctMaximumKelvin) /
                    config.cctMaximumKelvin
                ).coerceIn(0.0, 1.0)
            else -> 0.0
        }

        val flickerRisk = estimateFlickerRisk()
        val exposureRisk = (
            dutyRisk * 0.34 +
                lowLightRisk * 0.30 +
                highlightRisk * 0.22 +
                flickerRisk * 0.14
            ).coerceIn(0.0, 1.0)

        val shadowRisk = (
            lowLightRisk * 0.72 +
                telemetry.cameraMotionRisk * 0.08 +
                (1.0 - telemetry.focusConfidence) * 0.20
            ).coerceIn(0.0, 1.0)

        val dynamicRangePressure = (
            exposureRisk * 0.62 +
                highlightRisk * 0.20 +
                shadowRisk * 0.18
            ).coerceIn(0.0, 1.0)

        val stability = stabilityBand(
            timestampNanos = timestampNanos,
            lux = lux,
            cct = cct,
            flickerRisk = flickerRisk,
        )

        val confidence = (
            telemetry.focusConfidence.coerceIn(0.0, 1.0) * 0.35 +
                (1.0 - exposureRisk) * 0.35 +
                (1.0 - dynamicRangePressure) * 0.20 +
                (1.0 - cctRisk) * 0.10
            ).coerceIn(0.0, 1.0)

        val lighting = LightingState(
            timestampNanos = timestampNanos,
            lux = lux,
            correlatedColorTemperatureKelvin = cct,
            exposureCompensation = exposureCompensation(duty),
            highlightRisk = highlightRisk,
            shadowRisk = shadowRisk,
            flickerRisk = flickerRisk,
            dynamicRangePressure = dynamicRangePressure,
            stability = stability,
            confidence = confidence,
            source = PerceptionSource.SENSOR,
        )

        val scale = (
            1.0 -
                dynamicRangePressure * 0.48 -
                telemetry.cameraMotionRisk * 0.16
            ).coerceIn(0.25, 1.0)

        val faceScale = (
            scale *
                (1.0 - highlightRisk * 0.14)
            ).coerceIn(0.20, 1.0)

        val eyeScale = (
            scale *
                (1.0 - flickerRisk * 0.32) *
                (1.0 - lowLightRisk * 0.22)
            ).coerceIn(0.15, 1.0)

        val handScale = (
            scale *
                (1.0 - shadowRisk * 0.26)
            ).coerceIn(0.20, 1.0)

        val warnings = buildList {
            if (lowLightRisk > 0.55) add("low-light perception pressure")
            if (highlightRisk > 0.55) add("highlight clipping risk")
            if (flickerRisk > 0.45) add("possible lighting flicker")
            if (dutyRisk > 0.60) add("exposure duty outside preferred motion range")
            if (cctRisk > 0.45) add("color temperature outside calibrated operating band")
            if (telemetry.focusConfidence < 0.40) add("focus confidence degraded")
        }

        val diagnostics = listOf(
            "lux=" + format(lux),
            "cctKelvin=" + format(cct),
            "exposureDuty=" + format(duty),
            "lowLightRisk=" + format(lowLightRisk),
            "highlightRisk=" + format(highlightRisk),
            "shadowRisk=" + format(shadowRisk),
            "flickerRisk=" + format(flickerRisk),
            "dynamicRangePressure=" + format(dynamicRangePressure),
            "inferenceScale=" + format(scale),
        )

        lastTimestampNanos = timestampNanos
        previousLux = lux

        return Adaptation(
            lighting = lighting,
            inferenceScale = scale,
            depthScale = (
                scale *
                    (1.0 - lowLightRisk * 0.24)
                ).coerceIn(0.18, 1.0),
            faceScale = faceScale,
            eyeScale = eyeScale,
            handScale = handScale,
            warnings = warnings,
            diagnostics = diagnostics,
        )
    }

    @Synchronized
    fun reset() {
        luxHistory.clear()
        cctHistory.clear()
        lastTimestampNanos = Long.MIN_VALUE
        previousLux = Double.NaN
    }

    private fun inferLux(
        telemetry: CameraTelemetryState,
    ): Double {
        val isoFactor = (telemetry.sensitivityIso.coerceAtLeast(50) / 100.0)
        val duty = if (telemetry.frameDurationNanos > 0L) {
            telemetry.exposureTimeNanos.toDouble() /
                telemetry.frameDurationNanos.toDouble()
        } else {
            0.25
        }

        val focusFactor = (
            0.45 +
                telemetry.focusConfidence.coerceIn(0.0, 1.0) * 0.55
            )

        return (
            950.0 *
                isoFactor.coerceAtLeast(0.25) *
                duty.coerceIn(0.01, 1.0) *
                focusFactor
            ).coerceIn(0.5, 80_000.0)
    }

    private fun inferCct(
        previous: Double?,
        telemetry: CameraTelemetryState,
    ): Double {
        val baseline = previous
            ?.takeIf { it.isFinite() }
            ?: 5200.0

        val isoInfluence = (
            telemetry.sensitivityIso.coerceIn(50, 3200) /
                3200.0
            ) * 900.0

        val focusInfluence = (
            telemetry.focusConfidence.coerceIn(0.0, 1.0) - 0.5
        ) * 420.0

        return (
            baseline * 0.78 +
                (5200.0 + isoInfluence + focusInfluence) * 0.22
            ).coerceIn(
                config.cctMinimumKelvin * 0.70,
                config.cctMaximumKelvin * 1.25,
            )
    }

    private fun exposureCompensation(
        duty: Double,
    ): Double =
        (
            (config.targetExposureDuty - duty) /
                config.targetExposureDuty.coerceAtLeast(0.01)
            ).coerceIn(-2.0, 2.0)

    private fun estimateFlickerRisk(): Double {
        if (luxHistory.size < 5) return 0.15

        val values = luxHistory.toList()
        val mean = values.average().coerceAtLeast(0.1)
        val variance = values
            .map { value ->
                val normalized = (value - mean) / mean
                normalized * normalized
            }
            .average()

        val alternating = values
            .zipWithNext()
            .map { (a, b) ->
                abs(b - a) / max(1.0, mean)
            }
            .average()

        return (
            variance / config.flickerVarianceThreshold.coerceAtLeast(1e-5) * 0.64 +
                alternating * 0.36
            ).coerceIn(0.0, 1.0)
    }

    private fun stabilityBand(
        timestampNanos: Long,
        lux: Double,
        cct: Double,
        flickerRisk: Double,
    ): LightingStabilityBand {
        if (lastTimestampNanos == Long.MIN_VALUE) {
            return LightingStabilityBand.UNKNOWN
        }

        val dtMillis = (
            timestampNanos - lastTimestampNanos
            ).coerceAtLeast(0L) / 1e6

        if (dtMillis <= 0.0) return LightingStabilityBand.UNKNOWN

        val luxDelta = if (previousLux.isFinite()) {
            abs(lux - previousLux) /
                max(1.0, previousLux)
        } else {
            0.0
        }

        return when {
            flickerRisk > 0.62 || luxDelta > 0.35 ->
                LightingStabilityBand.CHALLENGING
            flickerRisk > 0.38 || luxDelta > 0.18 ||
                cctHistory.zipWithNext().any {
                    abs(it.second - it.first) > 800.0
                } ->
                LightingStabilityBand.CHANGING
            luxDelta < 0.06 ->
                LightingStabilityBand.STABLE
            else ->
                LightingStabilityBand.CHANGING
        }
    }

    private fun pushHistory(
        target: ArrayDeque<Double>,
        value: Double,
    ) {
        target.addLast(value)
        while (target.size > config.flickerWindowFrames) {
            target.removeFirst()
        }
    }

    private fun format(value: Double): String =
        if (!value.isFinite()) {
            "nan"
        } else {
            "%.4f".format(java.util.Locale.US, value)
        }
}
