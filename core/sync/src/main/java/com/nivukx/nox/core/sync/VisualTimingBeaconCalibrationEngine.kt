package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Detector-neutral visual timing-beacon calibration for heterogeneous distributed camera rigs.
 *
 * A timing beacon is assumed to encode a global event timestamp in the captured image. The beacon
 * observation is converted from sensor-start time to an exposure-center / row-corrected time before
 * the existing robust affine clock fitter is applied. This makes the calibration useful for RGB,
 * infrared, action cameras, and other sensors without coupling this engine to a particular detector.
 *
 * The engine intentionally does not inspect pixels. A platform-specific beacon detector is
 * responsible for decoding the timestamp and reporting confidence plus the observed image row.
 *
 * Production properties:
 * - bounded per-camera observation history;
 * - explicit decode confidence filtering;
 * - exposure-center correction;
 * - rolling-shutter row-time correction;
 * - iterative residual rejection followed by refitting;
 * - affine clock offset + drift estimation;
 * - deterministic result ordering;
 * - fail-closed admission for insufficient evidence;
 * - no mutation of source timestamps.
 */
class VisualTimingBeaconCalibrationEngine(
    private val config: Config = Config(),
    private val clockFitter: RecordedCameraClockFitEngine =
        RecordedCameraClockFitEngine(),
) {

    data class Config(
        val minimumSamplesPerCamera: Int = 8,
        val maximumObservationsPerCamera: Int = 256,
        val minimumDecodeConfidence: Double = 0.65,
        val maximumExposureNanos: Long = 100_000_000L,
        val maximumRollingShutterNanos: Long = 100_000_000L,
        val maximumAcceptedResidualNanos: Long = 6_000_000L,
        val huberThresholdNanos: Double = 1_500_000.0,
        val maximumClockFitRmsNanos: Double = 8_000_000.0,
        val minimumFitConfidence: Double = 0.55,
        val maximumRejectedRatio: Double = 0.35,
        val maximumFitIterations: Int = 4,
    ) {
        init {
            require(minimumSamplesPerCamera >= 3)
            require(maximumObservationsPerCamera >= minimumSamplesPerCamera)
            require(minimumDecodeConfidence in 0.0..1.0)
            require(maximumExposureNanos > 0L)
            require(maximumRollingShutterNanos >= 0L)
            require(maximumAcceptedResidualNanos > 0L)
            require(huberThresholdNanos > 0.0)
            require(maximumClockFitRmsNanos > huberThresholdNanos)
            require(minimumFitConfidence in 0.0..1.0)
            require(maximumRejectedRatio in 0.0..1.0)
            require(maximumFitIterations in 1..8)
        }
    }

    data class Observation(
        val deviceId: DeviceId,
        val deviceTimestampNanos: Long,
        val beaconTimestampNanos: Long,
        val exposureTimeNanos: Long,
        val rollingShutterSkewNanos: Long,
        val normalizedRow: Double = 0.5,
        val decodeConfidence: Double = 1.0,
    ) {
        init {
            require(deviceTimestampNanos >= 0L)
            require(beaconTimestampNanos >= 0L)
            require(exposureTimeNanos >= 0L)
            require(rollingShutterSkewNanos >= 0L)
            require(exposureTimeNanos <= 100_000_000L)
            require(rollingShutterSkewNanos <= 100_000_000L)
            require(normalizedRow.isFinite() && normalizedRow in 0.0..1.0)
            require(
                decodeConfidence.isFinite() &&
                    decodeConfidence in 0.0..1.0,
            )
        }
    }

    data class CorrectedObservation(
        val source: Observation,
        val correctedDeviceTimestampNanos: Long,
        val rowOffsetNanos: Long,
    )

    data class DeviceCalibration(
        val deviceId: DeviceId,
        val fit: RecordedCameraClockFitEngine.FitResult,
        val acceptedObservationCount: Int,
        val rejectedObservationCount: Int,
        val decodeConfidence: Double,
        val residualP95Nanos: Double,
        val residualMaxNanos: Double,
        val rejectionRatio: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    ) {
        val synchronizationConfidence: Double
            get() =
                if (accepted) {
                    (
                        fit.synchronizationConfidence *
                            decodeConfidence *
                            (
                                1.0 -
                                    rejectionRatio
                            )
                    ).coerceIn(0.0, 1.0)
                } else {
                    0.0
                }
    }

    data class CalibrationResult(
        val perDevice: List<DeviceCalibration>,
        val totalObservationCount: Int,
        val acceptedDeviceCount: Int,
        val rejectedDeviceCount: Int,
        val globalConfidence: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    ) {
        val deviceCount: Int
            get() = perDevice.size
    }

    fun calibrate(
        observations: List<Observation>,
    ): CalibrationResult {
        if (observations.isEmpty()) {
            return CalibrationResult(
                perDevice = emptyList(),
                totalObservationCount = 0,
                acceptedDeviceCount = 0,
                rejectedDeviceCount = 0,
                globalConfidence = 0.0,
                accepted = false,
                warnings =
                    listOf(
                        "CLOCK_BEACON_NO_OBSERVATIONS",
                    ),
            )
        }

        val grouped =
            observations
                .filter {
                    it.decodeConfidence >=
                        config.minimumDecodeConfidence
                }
                .groupBy { it.deviceId }
                .toSortedMap(
                    compareBy { it.value },
                )

        val perDevice =
            grouped.map { (deviceId, samples) ->
                calibrateDevice(
                    deviceId = deviceId,
                    observations = samples,
                )
            }

        val acceptedDevices =
            perDevice.count { it.accepted }

        val globalConfidence =
            if (perDevice.isEmpty()) {
                0.0
            } else {
                perDevice
                    .map { it.synchronizationConfidence }
                    .average()
                    .coerceIn(0.0, 1.0)
            }

        val warnings =
            buildList {
                if (
                    grouped.size <
                        observations
                            .map { it.deviceId }
                            .distinct()
                            .size
                ) {
                    add(
                        "BEACON_DECODE_CONFIDENCE_FILTERED_OBSERVATIONS",
                    )
                }
                if (acceptedDevices < 2) {
                    add(
                        "CLOCK_BEACON_MULTI_DEVICE_AUTHORITY_UNAVAILABLE",
                    )
                }
                perDevice
                    .flatMap { it.warnings }
                    .distinct()
                    .take(12)
                    .forEach(::add)
            }.distinct()

        return CalibrationResult(
            perDevice = perDevice,
            totalObservationCount = observations.size,
            acceptedDeviceCount = acceptedDevices,
            rejectedDeviceCount =
                perDevice.count { !it.accepted },
            globalConfidence = globalConfidence,
            accepted =
                acceptedDevices >=
                    min(2, perDevice.size) &&
                    globalConfidence >=
                        config.minimumFitConfidence,
            warnings = warnings,
        )
    }

    fun correct(
        observation: Observation,
    ): CorrectedObservation {
        val exposureCenter =
            observation.exposureTimeNanos / 2L

        val rowOffset =
            (
                (
                    observation.normalizedRow - 0.5
                    ) *
                    observation.rollingShutterSkewNanos
                        .toDouble()
                )
                .toLong()

        val corrected =
            saturatingAdd(
                saturatingAdd(
                    observation.deviceTimestampNanos,
                    exposureCenter,
                ),
                rowOffset,
            )

        return CorrectedObservation(
            source = observation,
            correctedDeviceTimestampNanos =
                corrected.coerceIn(
                    0L,
                    Long.MAX_VALUE,
                ),
            rowOffsetNanos = rowOffset,
        )
    }

    fun convertDeviceCaptureToMaster(
        observation: Observation,
        fit: RecordedCameraClockFitEngine.FitResult,
    ): Long =
        clockFitter.convertDeviceToMaster(
            deviceTimestampNanos =
                correct(observation)
                    .correctedDeviceTimestampNanos,
            fit = fit,
        )

    private fun saturatingAdd(
        first: Long,
        second: Long,
    ): Long =
        when {
            second > 0L &&
                first > Long.MAX_VALUE - second ->
                Long.MAX_VALUE
            second < 0L &&
                first < Long.MIN_VALUE - second ->
                Long.MIN_VALUE
            else ->
                first + second
        }

    private fun calibrateDevice(
        deviceId: DeviceId,
        observations: List<Observation>,
    ): DeviceCalibration {
        val bounded =
            observations
                .asSequence()
                .filter { it.deviceId == deviceId }
                .filter {
                    it.exposureTimeNanos <=
                        config.maximumExposureNanos
                }
                .filter {
                    it.rollingShutterSkewNanos <=
                        config.maximumRollingShutterNanos
                }
                .sortedBy {
                    it.deviceTimestampNanos
                }
                .take(config.maximumObservationsPerCamera)
                .map(::correct)
                .toList()

        if (bounded.size < config.minimumSamplesPerCamera) {
            return rejectedDevice(
                deviceId = deviceId,
                observationCount = bounded.size,
                warning =
                    "CLOCK_BEACON_SAMPLE_COUNT_INSUFFICIENT",
            )
        }

        var active =
            bounded

        var fit =
            fit(
                observations = active,
            )

        var iteration = 0
        while (
            iteration <
                config.maximumFitIterations
        ) {
            val residuals =
                active.map { observation ->
                    residualNanos(
                        observation = observation,
                        fit = fit,
                    )
                }

            val inliers =
                active.filterIndexed { index, _ ->
                    residuals[index] <=
                        config.maximumAcceptedResidualNanos
                }

            if (
                inliers.size < config.minimumSamplesPerCamera ||
                inliers.size == active.size
            ) {
                break
            }

            active = inliers
            fit = fit(active)
            iteration += 1
        }

        val finalResiduals =
            active.map { observation ->
                residualNanos(
                    observation = observation,
                    fit = fit,
                )
            }

        val allResiduals =
            bounded.map { observation ->
                residualNanos(
                    observation = observation,
                    fit = fit,
                )
            }

        val rejectedCount =
            bounded.size -
                active.size

        val rejectionRatio =
            rejectedCount.toDouble() /
                bounded.size
                    .coerceAtLeast(1)
                    .toDouble()

        val decodeConfidence =
            active
                .map { it.source.decodeConfidence }
                .average()
                .coerceIn(0.0, 1.0)

        val residualP95 =
            percentile(
                allResiduals.sorted(),
                0.95,
            )

        val residualMax =
            allResiduals.maxOrNull() ?: 0.0

        val warnings =
            buildList {
                if (
                    rejectedCount > 0
                ) {
                    add("CLOCK_BEACON_OUTLIERS_REJECTED")
                }
                if (
                    bounded.size >=
                        config.maximumObservationsPerCamera
                ) {
                    add(
                        "CLOCK_BEACON_OBSERVATION_BUDGET_TRUNCATED",
                    )
                }
                if (
                    rejectionRatio >
                        config.maximumRejectedRatio
                ) {
                    add(
                        "CLOCK_BEACON_REJECTION_RATIO_HIGH",
                    )
                }
                if (
                    fit.residualRmsNanos >
                        config.maximumClockFitRmsNanos
                ) {
                    add("CLOCK_BEACON_FIT_RESIDUAL_RMS_HIGH")
                }
                if (
                    residualP95 >
                        config.maximumAcceptedResidualNanos
                ) {
                    add("CLOCK_BEACON_RESIDUAL_P95_HIGH")
                }
            }.distinct()

        val accepted =
            fit.accepted &&
                fit.confidence >=
                    config.minimumFitConfidence &&
                active.size >=
                    config.minimumSamplesPerCamera &&
                rejectionRatio <=
                    config.maximumRejectedRatio &&
                residualP95 <=
                    config.maximumAcceptedResidualNanos

        return DeviceCalibration(
            deviceId = deviceId,
            fit = fit,
            acceptedObservationCount =
                active.size,
            rejectedObservationCount =
                rejectedCount,
            decodeConfidence = decodeConfidence,
            residualP95Nanos = residualP95,
            residualMaxNanos = residualMax,
            rejectionRatio = rejectionRatio,
            accepted = accepted,
            warnings = warnings,
        )
    }

    private fun fit(
        observations: List<CorrectedObservation>,
    ): RecordedCameraClockFitEngine.FitResult =
        clockFitter.fit(
            samples =
                observations.map {
                    RecordedCameraClockFitEngine.Sample(
                        deviceTimestampNanos =
                            it.correctedDeviceTimestampNanos,
                        masterTimestampNanos =
                            it.source.beaconTimestampNanos,
                        weight =
                            it.source.decodeConfidence
                                .coerceIn(0.01, 1.0),
                    )
                },
            minimumSamples =
                config.minimumSamplesPerCamera,
            huberThresholdNanos =
                config.huberThresholdNanos,
            maximumAcceptedRmsNanos =
                config.maximumClockFitRmsNanos,
        )

    private fun residualNanos(
        observation: CorrectedObservation,
        fit: RecordedCameraClockFitEngine.FitResult,
    ): Double {
        val predicted =
            clockFitter.convertDeviceToMaster(
                deviceTimestampNanos =
                    observation.correctedDeviceTimestampNanos,
                fit = fit,
            )
        return abs(
            observation.source.beaconTimestampNanos
                .toDouble() -
                predicted.toDouble(),
        )
    }

    private fun rejectedDevice(
        deviceId: DeviceId,
        observationCount: Int,
        warning: String,
    ): DeviceCalibration {
        val fit =
            RecordedCameraClockFitEngine.FitResult(
                accepted = false,
                sampleCount = observationCount,
                clockOffsetNanos = 0L,
                clockDriftPpm = 0.0,
                residualRmsNanos = Double.POSITIVE_INFINITY,
                residualP95Nanos = Double.POSITIVE_INFINITY,
                residualMaxNanos = Double.POSITIVE_INFINITY,
                confidence = 0.0,
                warnings = listOf(warning),
            )

        return DeviceCalibration(
            deviceId = deviceId,
            fit = fit,
            acceptedObservationCount = 0,
            rejectedObservationCount = observationCount,
            decodeConfidence = 0.0,
            residualP95Nanos = Double.POSITIVE_INFINITY,
            residualMaxNanos = Double.POSITIVE_INFINITY,
            rejectionRatio = 1.0,
            accepted = false,
            warnings = listOf(warning),
        )
    }

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val position =
            probability.coerceIn(0.0, 1.0) *
                (values.size - 1)
                    .toDouble()
        val lower =
            position.toInt()
        val upper =
            min(
                values.lastIndex,
                lower + 1,
            )
        val fraction =
            position -
                lower.toDouble()

        return values[lower] *
            (1.0 - fraction) +
            values[upper] *
            fraction
    }
}
