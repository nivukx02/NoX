package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.CaptureTimingEvidence
import com.nivukx.nox.core.model.FrameTelemetry
kotlin.math.abs
kotlin.math.max

/**
 * Converts Camera2 timing telemetry into an explicit exposure-centered timing authority record.
 * Sensor timestamps remain the primary clock anchor; hardware readout timing and rolling-shutter skew
 * are treated as corroborating evidence and uncertainty contributors rather than magic constants.
 */
class ExposureMidpointTimingAuthorityEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumExposureNanos: Long = 500_000_000L,
        val maximumRollingShutterSkewNanos: Long = 100_000_000L,
        val maximumReadoutOffsetNanos: Long = 100_000_000L,
        val minimumTimingUncertaintyNanos: Double = 50.0,
        val hardwareUncertaintyNanos: Double = 100.0,
        val softwareUncertaintyNanos: Double = 250_000.0,
    ) {
        init {
            require(maximumExposureNanos > 0L)
            require(maximumRollingShutterSkewNanos >= 0L)
            require(maximumReadoutOffsetNanos >= 0L)
            require(minimumTimingUncertaintyNanos >= 0.0)
            require(hardwareUncertaintyNanos >= 0.0)
            require(softwareUncertaintyNanos >= hardwareUncertaintyNanos)
        }
    }

    enum class Authority {
        HARDWARE_AND_SENSOR,
        SENSOR_AND_READOUT_INFERRED,
        SENSOR_ONLY,
        INVALID,
    }

    data class Result(
        val evidence: CaptureTimingEvidence?,
        val authority: Authority,
        val readoutOffsetNanos: Long?,
        val timingConfidence: Double,
        val warnings: List<String>,
    ) {
        val valid: Boolean get() = evidence != null && authority != Authority.INVALID
    }

    @Synchronized
    fun resolve(telemetry: FrameTelemetry, rowCount: Int = 1): Result {
        if (telemetry.sensorTimestampNanos < 0L ||
            telemetry.monotonicTimestampNanos < 0L ||
            telemetry.exposureTimeNanos < 0L ||
            telemetry.frameDurationNanos < 0L ||
            telemetry.rollingShutterSkewNanos < 0L ||
            telemetry.exposureTimeNanos > config.maximumExposureNanos ||
            telemetry.rollingShutterSkewNanos > config.maximumRollingShutterSkewNanos ||
            rowCount <= 0
        ) {
            return invalid("invalid capture timing metadata")
        }

        val start = telemetry.sensorTimestampNanos
        val end = safeAdd(start, telemetry.exposureTimeNanos)
            ?: return invalid("exposure timestamp overflow")
        val midpoint = start + telemetry.exposureTimeNanos / 2L
        val readout = telemetry.readoutTimestampNanos
        val readoutOffset = readout?.let {
            safeSubtract(it, start)
        }

        val warnings = ArrayList<String>(4)
        var uncertainty =
            if (readout != null) config.hardwareUncertaintyNanos
            else config.softwareUncertaintyNanos

        if (readoutOffset != null && abs(readoutOffset) > config.maximumReadoutOffsetNanos) {
            warnings += "readout-anchor-offset-outside-authority-window"
            uncertainty += abs(readoutOffset).toDouble() * 0.25
        }

        if (telemetry.rollingShutterSkewNanos > 0L) {
            uncertainty += telemetry.rollingShutterSkewNanos.toDouble() / 2.0
        }

        val exposureFraction =
            if (telemetry.frameDurationNanos > 0L) {
                telemetry.exposureTimeNanos.toDouble() / telemetry.frameDurationNanos.toDouble()
            } else 0.0
        if (exposureFraction > 1.0 + 1e-9) {
            warnings += "exposure-longer-than-frame-duration"
            uncertainty += (exposureFraction - 1.0) * 1_000_000.0
        }

        val effectiveUncertainty =
            max(config.minimumTimingUncertaintyNanos, uncertainty)
        val confidence =
            (1.0 / (1.0 + effectiveUncertainty / 1_000_000.0))
                .coerceIn(0.0, 1.0)

        val authority = when {
            readout != null && warnings.none { it.contains("outside-authority") } ->
                Authority.HARDWARE_AND_SENSOR
            readout != null ->
                Authority.SENSOR_AND_READOUT_INFERRED
            else ->
                Authority.SENSOR_ONLY
        }

        val evidence = CaptureTimingEvidence(
            sensorTimestampNanos = telemetry.sensorTimestampNanos,
            exposureStartNanos = start,
            exposureMidpointNanos = midpoint,
            exposureEndNanos = end,
            readoutDurationNanos = telemetry.rollingShutterSkewNanos,
            rowTimeNanos =
                if (rowCount <= 1) 0.0
                else telemetry.rollingShutterSkewNanos.toDouble() / (rowCount - 1).toDouble(),
            timingUncertaintyNanos = effectiveUncertainty,
            hardwareAuthoritative = authority == Authority.HARDWARE_AND_SENSOR,
            source = "camera2-${authority.name.lowercase()}",
        )

        return Result(
            evidence = evidence,
            authority = authority,
            readoutOffsetNanos = readoutOffset,
            timingConfidence = confidence,
            warnings = warnings,
        )
    }

    private fun invalid(message: String): Result =
        Result(
            evidence = null,
            authority = Authority.INVALID,
            readoutOffsetNanos = null,
            timingConfidence = 0.0,
            warnings = listOf(message),
        )

    private fun safeAdd(a: Long, b: Long): Long? {
        if (b < 0L || a > Long.MAX_VALUE - b) return null
        return a + b
    }

    private fun safeSubtract(a: Long, b: Long): Long? {
        if (b < 0L || a < Long.MIN_VALUE + b) return null
        return a - b
    }
}