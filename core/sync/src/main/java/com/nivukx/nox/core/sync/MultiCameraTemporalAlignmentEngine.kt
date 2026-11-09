package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FrameStamp
import com.nivukx.nox.core.model.RollingShutterModel
import kotlin.math.abs
import kotlin.math.min

/**
 * Production temporal-alignment layer for distributed motion capture.
 *
 * Camera timestamps are not automatically interchangeable across devices. Android documents that
 * SENSOR_TIMESTAMP represents the start of exposure, while cross-device comparability depends on
 * the timestamp source and hardware synchronization support. This engine makes that distinction
 * explicit instead of silently treating every nanosecond value as globally comparable.
 *
 * Responsibilities:
 * - normalize exposure-center timestamps;
 * - compensate row-dependent rolling-shutter timing;
 * - estimate per-device clock offsets from paired samples;
 * - reject unknown or unsafe timestamp domains;
 * - compute frame-group temporal spread;
 * - expose deterministic confidence and actionable diagnostics;
 * - retain short-term offset history without per-frame matrix allocations.
 *
 * The original capture timestamp is never overwritten. The alignment result is a separate provenance
 * record so downstream reconstruction can decide whether to compensate, hold, or reject a frame.
 */
class MultiCameraTemporalAlignmentEngine(
    private val config: Config = Config(),
) {
    enum class TimestampDomain {
        SENSOR,
        REALTIME,
        MONOTONIC,
        UNKNOWN,
    }

    data class Config(
        val historyPerDevice: Int = 32,
        val ewmaAlpha: Double = 0.22,
        val preferredResidualNanos: Long = 8_000_000L,
        val maximumOffsetNanos: Long = 40_000_000L,
        val preferredSpreadNanos: Long = 3_000_000L,
        val maximumSpreadNanos: Long = 18_000_000L,
        val maximumExposureNanos: Long = 100_000_000L,
        val minimumConfidence: Double = 0.35,
        val unknownDomainPenalty: Double = 0.34,
        val approximateDomainPenalty: Double = 0.18,
        val rollingShutterCompensationWeight: Double = 1.0,
        val maximumPeerClockDriftPpm: Double = 500.0,
        val peerClockDriftAlpha: Double = 0.18,
    ) {
        init {
            require(historyPerDevice >= 4)
            require(ewmaAlpha in 0.01..1.0)
            require(preferredResidualNanos > 0L)
            require(maximumOffsetNanos > preferredResidualNanos)
            require(preferredSpreadNanos in 0L until maximumSpreadNanos)
            require(maximumSpreadNanos > 0L)
            require(maximumExposureNanos > 0L)
            require(minimumConfidence in 0.0..1.0)
            require(unknownDomainPenalty in 0.0..1.0)
            require(approximateDomainPenalty in 0.0..1.0)
            require(rollingShutterCompensationWeight in 0.0..1.0)
            require(maximumPeerClockDriftPpm > 0.0)
            require(peerClockDriftAlpha in 0.01..1.0)
        }
    }

    data class FrameInput(
        val deviceId: DeviceId,
        val stamp: FrameStamp,
        val domain: TimestampDomain,
        val hardwareSynchronized: Boolean = false,
        val rollingShutter: RollingShutterModel? = null,
        val sourceIsComparableAcrossDevices: Boolean = true,
        val normalizedRow: Double = 0.5,
    ) {
        init {
            require(stamp.sensorTimestampNanos >= 0L)
            require(stamp.monotonicTimestampNanos >= 0L)
            require(stamp.exposureTimeNanos >= 0L)
            require(normalizedRow in 0.0..1.0)
        }
    }

    data class AlignmentSample(
        val deviceId: DeviceId,
        val correctedTimestampNanos: Long,
        val rawTimestampNanos: Long,
        val estimatedClockOffsetNanos: Long,
        val exposureCenterOffsetNanos: Long,
        val rollingShutterOffsetNanos: Long,
        val confidence: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    data class GroupResult(
        val referenceTimestampNanos: Long,
        val accepted: Boolean,
        val confidence: Double,
        val spreadNanos: Long,
        val medianOffsetNanos: Long,
        val maximumAbsoluteOffsetNanos: Long,
        val samples: List<AlignmentSample>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val acceptedDeviceCount: Int
            get() = samples.count { it.accepted }

        val rejectedDeviceCount: Int
            get() = samples.count { !it.accepted }

        val temporalQuality: Double
            get() = confidence
    }

    data class DeviceTelemetry(
        val deviceId: DeviceId,
        val samples: Int,
        val offsetEwmaNanos: Double,
        val offsetJitterNanos: Double,
        val confidence: Double,
        val timestampDomain: TimestampDomain,
        val hardwareSynchronized: Boolean,
        val subframeOffsetEwmaNanos: Double = 0.0,
        val subframeConfidence: Double = 0.0,
        val beaconCalibrationConfidence: Double = 0.0,
        val beaconClockDriftPpm: Double = 0.0,
        val peerClockDriftPpm: Double = 0.0,
    )

    private data class DeviceState(
        var offsetEwmaNanos: Double = 0.0,
        var jitterEwmaNanos: Double = 0.0,
        var samples: Int = 0,
        var confidence: Double = 0.0,
        var domain: TimestampDomain = TimestampDomain.UNKNOWN,
        var hardwareSynchronized: Boolean = false,
        var subframeOffsetEwmaNanos: Double = 0.0,
        var subframeConfidence: Double = 0.0,
        var subframeJitterEwmaNanos: Double = 0.0,
        var subframeEvidenceSamples: Int = 0,
        var beaconOffsetNanos: Double? = null,
        var beaconDriftPpm: Double = 0.0,
        var beaconConfidence: Double = 0.0,
        var peerClockDriftPpm: Double = 0.0,
        var lastOffsetMeasurementTimestampNanos: Long = Long.MIN_VALUE,
        val residualHistory: ArrayDeque<Long> = ArrayDeque(),
        val offsetHistory: ArrayDeque<Pair<Long, Long>> = ArrayDeque(),
    )

    private val states = LinkedHashMap<DeviceId, DeviceState>()

    @Synchronized
    fun alignGroup(
        frames: List<FrameInput>,
        referenceDevice: DeviceId? = null,
    ): GroupResult {
        if (frames.isEmpty()) {
            return GroupResult(
                referenceTimestampNanos = 0L,
                accepted = false,
                confidence = 0.0,
                spreadNanos = Long.MAX_VALUE,
                medianOffsetNanos = 0L,
                maximumAbsoluteOffsetNanos = Long.MAX_VALUE,
                samples = emptyList(),
                warnings = listOf("no frames supplied"),
                recommendations = listOf(
                    "Provide at least two synchronized camera frames.",
                ),
            )
        }

        val deduplicated = frames
            .distinctBy { it.deviceId }
            .sortedBy { it.deviceId.value }

        val reference = chooseReference(
            frames = deduplicated,
            explicit = referenceDevice,
        )

        val referenceTime =
            correctedExposureTimestamp(reference)

        val estimates = deduplicated.map { frame ->
            alignFrame(
                frame = frame,
                referenceTimestampNanos = referenceTime,
                referenceFrame = reference,
            )
        }

        val accepted = estimates.filter { it.accepted }
        val offsets = accepted.map {
            it.correctedTimestampNanos - referenceTime
        }
        val absoluteOffsets = offsets.map(::abs)

        val spread =
            if (absoluteOffsets.isEmpty()) {
                Long.MAX_VALUE
            } else {
                absoluteOffsets.maxOrNull()!! -
                    absoluteOffsets.minOrNull()!!
            }

        val medianOffset = medianLong(offsets)
        val maximumAbsoluteOffset =
            absoluteOffsets.maxOrNull() ?: Long.MAX_VALUE

        val domainConfidence =
            domainConfidence(deduplicated)

        val spreadConfidence = inverse(
            value = spread.toDouble(),
            target = config.preferredSpreadNanos.toDouble(),
            maximum = config.maximumSpreadNanos.toDouble(),
        )

        val offsetConfidence = inverse(
            value = maximumAbsoluteOffset.toDouble(),
            target = config.preferredResidualNanos.toDouble(),
            maximum = config.maximumOffsetNanos.toDouble(),
        )

        val robustSpreadNanos =
            medianAbsoluteDeviationLong(offsets)

        val robustSpreadConfidence =
            inverse(
                value = robustSpreadNanos.toDouble(),
                target = 0.0,
                maximum = config.preferredSpreadNanos.toDouble(),
            )

        val acceptanceRatio =
            accepted.size.toDouble() /
                deduplicated.size
                    .coerceAtLeast(1)
                    .toDouble()

        val confidence = (
            0.24 * domainConfidence +
                0.24 * spreadConfidence +
                0.18 * offsetConfidence +
                0.18 * acceptanceRatio +
                0.16 * robustSpreadConfidence
            ).coerceIn(0.0, 1.0)

        val warnings = buildWarnings(
            frames = deduplicated,
            estimates = estimates,
            referenceTimestampNanos = referenceTime,
            spreadNanos = spread,
            maximumAbsoluteOffsetNanos = maximumAbsoluteOffset,
            confidence = confidence,
        )

        val recommendations = buildRecommendations(
            frames = deduplicated,
            spreadNanos = spread,
            maximumAbsoluteOffsetNanos = maximumAbsoluteOffset,
            acceptedDeviceCount = accepted.size,
        )

        return GroupResult(
            referenceTimestampNanos = referenceTime,
            accepted =
                accepted.size >= 2 &&
                    confidence >= config.minimumConfidence &&
                    spread <= config.maximumSpreadNanos,
            confidence = confidence,
            spreadNanos = spread,
            medianOffsetNanos = medianOffset,
            maximumAbsoluteOffsetNanos = maximumAbsoluteOffset,
            samples = estimates,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun ingestPair(
        reference: FrameInput,
        sample: FrameInput,
    ): DeviceTelemetry {
        val measuredOffset =
            correctedExposureTimestamp(sample) -
                correctedExposureTimestamp(reference)

        val state =
            states.getOrPut(sample.deviceId) {
                DeviceState()
            }

        updateState(
            state = state,
            measuredOffsetNanos = measuredOffset,
            frame = sample,
        )

        return telemetry(sample.deviceId)
    }

    @Synchronized
    fun applySubframeOffsetEvidence(
        deviceId: DeviceId,
        offsetNanos: Double,
        confidence: Double,
    ): DeviceTelemetry {
        require(offsetNanos.isFinite())
        require(confidence in 0.0..1.0)

        val state =
            states.getOrPut(deviceId) {
                DeviceState()
            }

        if (state.subframeEvidenceSamples == 0) {
            state.subframeOffsetEwmaNanos = offsetNanos
            state.subframeJitterEwmaNanos = 0.0
            state.subframeConfidence = confidence
        } else {
            val alpha =
                (
                    0.16 +
                        0.52 * confidence
                    ).coerceIn(
                    0.16,
                    0.68,
                )

            val residual =
                offsetNanos -
                    state.subframeOffsetEwmaNanos

            state.subframeOffsetEwmaNanos +=
                residual * alpha

            state.subframeJitterEwmaNanos =
                state.subframeJitterEwmaNanos *
                    0.82 +
                    abs(residual) *
                    0.18

            state.subframeConfidence =
                (
                    state.subframeConfidence *
                        0.30 +
                        confidence *
                        0.70
                    ).coerceIn(
                    0.0,
                    1.0,
                )
        }
        state.subframeEvidenceSamples += 1

        return telemetry(deviceId)
    }

    /**
     * Applies a detector-neutral visual timing-beacon calibration to a live camera state.
     *
     * The beacon fit supplies an absolute affine device->master clock model. Existing peer-clock
     * estimation remains active, but the beacon contribution is blended by confidence so a weak
     * visual decode cannot abruptly move a live stream.
     */
    @Synchronized
    fun applyVisualTimingBeaconCalibration(
        calibration: VisualTimingBeaconCalibrationEngine.DeviceCalibration,
    ): DeviceTelemetry {
        val state =
            states.getOrPut(calibration.deviceId) {
                DeviceState()
            }

        if (!calibration.accepted) {
            state.beaconConfidence *= 0.90
            return telemetry(calibration.deviceId)
        }

        state.beaconOffsetNanos =
            calibration.fit.clockOffsetNanos.toDouble()
        state.beaconDriftPpm =
            calibration.fit.clockDriftPpm
        state.beaconConfidence =
            calibration.synchronizationConfidence
                .coerceIn(0.0, 1.0)

        return telemetry(calibration.deviceId)
    }

    @Synchronized
    fun applySubframeAlignment(
        deviceId: DeviceId,
        result: SubframeMotionAlignmentEngine.Result,
    ): DeviceTelemetry {
        val state =
            states.getOrPut(deviceId) {
                DeviceState()
            }

        if (result.accepted &&
            result.offsetNanos.isFinite()
        ) {
            if (state.subframeEvidenceSamples == 0) {
                state.subframeOffsetEwmaNanos = result.offsetNanos
                state.subframeJitterEwmaNanos = 0.0
                state.subframeConfidence = result.confidence
                state.subframeEvidenceSamples = 1
                return telemetry(deviceId)
            }

            val alpha =
                (
                    0.16 +
                        0.52 * result.confidence
                    ).coerceIn(
                        0.16,
                        0.68,
                    )

            val residual =
                result.offsetNanos -
                    state.subframeOffsetEwmaNanos

            state.subframeOffsetEwmaNanos +=
                residual * alpha

            state.subframeJitterEwmaNanos =
                state.subframeJitterEwmaNanos *
                    0.82 +
                    abs(residual) *
                    0.18

            state.subframeConfidence =
                (
                    state.subframeConfidence *
                        0.30 +
                        result.confidence *
                        0.70
                    ).coerceIn(
                        0.0,
                        1.0,
                    )
            state.subframeEvidenceSamples += 1
        } else {
            state.subframeConfidence *= 0.94
        }

        return telemetry(deviceId)
    }

    @Synchronized
    fun telemetry(
        deviceId: DeviceId,
    ): DeviceTelemetry {
        val state =
            states[deviceId]
                ?: return DeviceTelemetry(
                    deviceId = deviceId,
                    samples = 0,
                    offsetEwmaNanos = 0.0,
                    offsetJitterNanos = 0.0,
                    confidence = 0.0,
                    timestampDomain = TimestampDomain.UNKNOWN,
                    hardwareSynchronized = false,
                    subframeOffsetEwmaNanos = 0.0,
                    subframeConfidence = 0.0,
                )

        return DeviceTelemetry(
            deviceId = deviceId,
            samples = state.samples,
            offsetEwmaNanos = state.offsetEwmaNanos,
            offsetJitterNanos = state.jitterEwmaNanos,
            confidence = state.confidence,
            timestampDomain = state.domain,
            hardwareSynchronized = state.hardwareSynchronized,
            subframeOffsetEwmaNanos =
                state.subframeOffsetEwmaNanos,
            subframeConfidence =
                state.subframeConfidence,
            beaconCalibrationConfidence =
                state.beaconConfidence,
            beaconClockDriftPpm =
                state.beaconDriftPpm,
            peerClockDriftPpm =
                state.peerClockDriftPpm,
        )
    }

    @Synchronized
    fun allTelemetry(): List<DeviceTelemetry> =
        states.keys
            .sortedBy { it.value }
            .map(::telemetry)

    @Synchronized
    fun reset() {
        states.clear()
    }

    private fun chooseReference(
        frames: List<FrameInput>,
        explicit: DeviceId?,
    ): FrameInput {
        explicit?.let { id ->
            frames.firstOrNull { it.deviceId == id }?.let {
                return it
            }
        }

        return frames.maxByOrNull { frame ->
            val domainScore =
                when (frame.domain) {
                    TimestampDomain.SENSOR -> 1.00
                    TimestampDomain.REALTIME -> 0.92
                    TimestampDomain.MONOTONIC -> 0.78
                    TimestampDomain.UNKNOWN -> 0.25
                }

            val hardwareScore =
                if (frame.hardwareSynchronized) {
                    0.20
                } else {
                    0.0
                }

            val comparabilityScore =
                if (frame.sourceIsComparableAcrossDevices) {
                    0.10
                } else {
                    0.0
                }

            domainScore +
                hardwareScore +
                comparabilityScore
        } ?: frames.first()
    }

    private fun alignFrame(
        frame: FrameInput,
        referenceTimestampNanos: Long,
        referenceFrame: FrameInput,
    ): AlignmentSample {
        val corrected =
            correctedExposureTimestamp(frame)

        val state =
            if (frame.deviceId == referenceFrame.deviceId) {
                null
            } else {
                states.getOrPut(frame.deviceId) {
                    DeviceState()
                }
            }

        val estimatedOffset =
            if (state == null) {
                0L
            } else {
                val peerElapsedNanos =
                    if (state.lastOffsetMeasurementTimestampNanos == Long.MIN_VALUE) {
                        0.0
                    } else {
                        corrected.toDouble() -
                            state.lastOffsetMeasurementTimestampNanos.toDouble()
                    }

                val peerDriftProjection =
                    peerElapsedNanos *
                        state.peerClockDriftPpm *
                        1e-6

                val peerOffset =
                    state.offsetEwmaNanos +
                        peerDriftProjection +
                        (
                            state.subframeOffsetEwmaNanos *
                                state.subframeConfidence
                        )

                val beaconOffset =
                    state.beaconOffsetNanos?.let { clockOffset ->
                        val slope =
                            1.0 +
                                state.beaconDriftPpm *
                                    1e-6
                        val predictedMaster =
                            clockOffset +
                                corrected.toDouble() *
                                    slope
                        corrected.toDouble() -
                            predictedMaster
                    }

                val beaconBlend =
                    state.beaconConfidence
                        .coerceIn(0.0, 1.0)

                (
                    if (beaconOffset != null) {
                        peerOffset *
                            (1.0 - beaconBlend) +
                            beaconOffset *
                            beaconBlend
                    } else {
                        peerOffset
                    }
                ).toLong()
            }

        if (state != null) {
            updateState(
                state = state,
                measuredOffsetNanos =
                    corrected - referenceTimestampNanos,
                frame = frame,
            )
        }

        val correctedTimestamp =
            corrected - estimatedOffset

        val residual =
            correctedTimestamp -
                referenceTimestampNanos

        val residualConfidence =
            inverse(
                value = abs(residual).toDouble(),
                target = config.preferredResidualNanos.toDouble(),
                maximum = config.maximumOffsetNanos.toDouble(),
            )

        val domainConfidence =
            frameDomainConfidence(frame)

        val hardwareConfidence =
            if (frame.hardwareSynchronized) {
                1.0
            } else {
                0.78
            }

        val comparabilityConfidence =
            if (frame.sourceIsComparableAcrossDevices) {
                1.0
            } else {
                1.0 -
                    config.approximateDomainPenalty
            }

        val rollingConfidence =
            if (frame.rollingShutter == null) {
                0.92
            } else {
                val readout =
                    frame.rollingShutter.readoutTimeNanos
                if (readout <= 0L) {
                    0.45
                } else {
                    (
                        1.0 -
                            (
                                readout.toDouble() /
                                    config.maximumExposureNanos
                            ).coerceIn(0.0, 1.0) *
                            0.24
                        ).coerceIn(0.0, 1.0)
                }
            }

        val confidence = (
            0.32 * residualConfidence +
                0.28 * domainConfidence +
                0.18 * hardwareConfidence +
                0.10 * comparabilityConfidence +
                0.12 * rollingConfidence
            ).coerceIn(0.0, 1.0)

        val warnings =
            buildFrameWarnings(
                frame = frame,
                residualNanos = residual,
                confidence = confidence,
            )

        val accepted =
            confidence >= config.minimumConfidence &&
                abs(residual) <=
                config.maximumOffsetNanos &&
                frame.domain != TimestampDomain.UNKNOWN

        return AlignmentSample(
            deviceId = frame.deviceId,
            correctedTimestampNanos = correctedTimestamp,
            rawTimestampNanos = frame.rawTimestamp(),
            estimatedClockOffsetNanos = estimatedOffset,
            exposureCenterOffsetNanos =
                exposureCenterOffset(frame),
            rollingShutterOffsetNanos =
                rollingShutterOffset(frame),
            confidence = confidence,
            accepted = accepted,
            warnings = warnings,
        )
    }

    private fun updateState(
        state: DeviceState,
        measuredOffsetNanos: Long,
        frame: FrameInput,
    ) {
        val sample =
            measuredOffsetNanos.toDouble()

        val residual =
            if (state.samples == 0) {
                sample
            } else {
                sample -
                    state.offsetEwmaNanos
            }

        val alpha = config.ewmaAlpha

        state.offsetEwmaNanos =
            if (state.samples == 0) {
                sample
            } else {
                state.offsetEwmaNanos *
                    (1.0 - alpha) +
                    sample * alpha
            }

        state.jitterEwmaNanos =
            if (state.samples == 0) {
                0.0
            } else {
                state.jitterEwmaNanos *
                    (1.0 - alpha) +
                    abs(residual) * alpha
            }

        state.residualHistory.addLast(
            abs(measuredOffsetNanos),
        )

        while (
            state.residualHistory.size >
            config.historyPerDevice
        ) {
            state.residualHistory.removeFirst()
        }

        val measurementTimestamp =
            frame.timestampForAlignment()

        state.offsetHistory.addLast(
            measurementTimestamp to measuredOffsetNanos,
        )

        while (
            state.offsetHistory.size >
            min(config.historyPerDevice, 16)
        ) {
            state.offsetHistory.removeFirst()
        }

        robustClockDriftPpm(state)?.let { measuredDrift ->
            state.peerClockDriftPpm =
                state.peerClockDriftPpm *
                    (1.0 - config.peerClockDriftAlpha) +
                    measuredDrift *
                    config.peerClockDriftAlpha
        }

        state.peerClockDriftPpm =
            state.peerClockDriftPpm.coerceIn(
                -config.maximumPeerClockDriftPpm,
                config.maximumPeerClockDriftPpm,
            )

        state.lastOffsetMeasurementTimestampNanos =
            measurementTimestamp

        val jitterScore =
            inverse(
                value = state.jitterEwmaNanos,
                target = 0.0,
                maximum = config.maximumOffsetNanos.toDouble(),
            )

        val robustJitterScore =
            inverse(
                value =
                    medianAbsoluteDeviationLong(
                        state.residualHistory.toList(),
                    ).toDouble(),
                target = 0.0,
                maximum = config.maximumOffsetNanos.toDouble(),
            )

        state.domain = frame.domain
        state.hardwareSynchronized =
            frame.hardwareSynchronized
        state.samples += 1
        state.confidence = (
            0.50 * inverse(
                value = abs(state.offsetEwmaNanos),
                target = config.preferredResidualNanos.toDouble(),
                maximum = config.maximumOffsetNanos.toDouble(),
            ) +
                0.24 * jitterScore +
                0.26 * robustJitterScore
            ).coerceIn(0.0, 1.0)
    }

    private fun correctedExposureTimestamp(
        frame: FrameInput,
    ): Long =
        frame.timestampForAlignment()

    private fun exposureCenterOffset(
        frame: FrameInput,
    ): Long =
        frame.stamp.exposureTimeNanos
            .coerceIn(
                0L,
                config.maximumExposureNanos,
            ) / 2L

    private fun rollingShutterOffset(
        frame: FrameInput,
    ): Long {
        val model =
            frame.rollingShutter
                ?: return 0L

        val rowOffset =
            (
                (frame.normalizedRow - 0.5) *
                    model.readoutTimeNanos *
                    model.directionSign
                ).toLong()

        return (
            rowOffset *
                config.rollingShutterCompensationWeight
            ).toLong()
    }

    private fun FrameInput.timestampForAlignment(): Long {
        val raw =
            when (domain) {
                TimestampDomain.SENSOR ->
                    stamp.sensorTimestampNanos
                TimestampDomain.REALTIME ->
                    stamp.monotonicTimestampNanos
                TimestampDomain.MONOTONIC ->
                    stamp.monotonicTimestampNanos
                TimestampDomain.UNKNOWN ->
                    stamp.monotonicTimestampNanos
            }

        return raw +
            exposureCenterOffset(this) +
            rollingShutterOffset(this)
    }

    private fun FrameInput.rawTimestamp(): Long =
        when (domain) {
            TimestampDomain.SENSOR ->
                stamp.sensorTimestampNanos
            TimestampDomain.REALTIME,
            TimestampDomain.MONOTONIC,
            TimestampDomain.UNKNOWN ->
                stamp.monotonicTimestampNanos
        }

    private fun frameDomainConfidence(
        frame: FrameInput,
    ): Double =
        when (frame.domain) {
            TimestampDomain.SENSOR ->
                if (frame.sourceIsComparableAcrossDevices) {
                    1.0
                } else {
                    0.78
                }

            TimestampDomain.REALTIME ->
                if (frame.sourceIsComparableAcrossDevices) {
                    0.96
                } else {
                    0.78
                }

            TimestampDomain.MONOTONIC ->
                if (frame.sourceIsComparableAcrossDevices) {
                    0.88
                } else {
                    0.72
                }

            TimestampDomain.UNKNOWN ->
                config.unknownDomainPenalty
        }

    private fun domainConfidence(
        frames: List<FrameInput>,
    ): Double =
        if (frames.isEmpty()) {
            0.0
        } else {
            frames.map(::frameDomainConfidence).average()
        }

    private fun buildFrameWarnings(
        frame: FrameInput,
        residualNanos: Long,
        confidence: Double,
    ): List<String> = buildList {
        if (frame.domain == TimestampDomain.UNKNOWN) {
            add("timestamp-domain-unknown")
        }

        if (!frame.sourceIsComparableAcrossDevices) {
            add(
                "timestamps-not-guaranteed-cross-device-comparable",
            )
        }

        if (
            abs(residualNanos) >
            config.preferredResidualNanos
        ) {
            add("temporal-offset-above-preferred-bound")
        }

        if (
            abs(residualNanos) >
            config.maximumOffsetNanos
        ) {
            add("temporal-offset-above-hard-bound")
        }

        if (
            frame.rollingShutter != null &&
                frame.rollingShutter.readoutTimeNanos >
                config.maximumExposureNanos
        ) {
            add(
                "rolling-shutter-readout-exceeds-configured-bound",
            )
        }

        if (confidence < config.minimumConfidence) {
            add("alignment-confidence-low")
        }

        if (
            frame.stamp.exposureTimeNanos >
            config.maximumExposureNanos
        ) {
            add("exposure-time-exceeds-configured-bound")
        }
    }.distinct()

    private fun buildWarnings(
        frames: List<FrameInput>,
        estimates: List<AlignmentSample>,
        referenceTimestampNanos: Long,
        spreadNanos: Long,
        maximumAbsoluteOffsetNanos: Long,
        confidence: Double,
    ): List<String> = buildList {
        if (frames.size < 2) {
            add("single-camera-group")
        }

        if (
            estimates.count { it.accepted } <
            min(2, frames.size)
        ) {
            add("insufficient-accepted-camera-frames")
        }

        if (
            estimates.any {
                it.warnings.any { warning ->
                    warning == "timestamp-domain-unknown"
                }
            }
        ) {
            add("timestamp-domain-unknown")
        }


        if (
            spreadNanos >
            config.preferredSpreadNanos
        ) {
            add("frame-spread-above-preferred-bound")
        }

        if (
            spreadNanos >
            config.maximumSpreadNanos
        ) {
            add("frame-spread-above-hard-bound")
        }

        if (
            maximumAbsoluteOffsetNanos >
            config.preferredResidualNanos
        ) {
            add("clock-offset-above-preferred-bound")
        }

        if (
            maximumAbsoluteOffsetNanos >
            config.maximumOffsetNanos
        ) {
            add("clock-offset-above-hard-bound")
        }

        val robustSpreadNanos =
            medianAbsoluteDeviationLong(
                estimates
                    .filter { it.accepted }
                    .map {
                        it.correctedTimestampNanos -
                            referenceTimestampNanos
                    },
            )

        if (
            robustSpreadNanos >
            config.preferredSpreadNanos
        ) {
            add("robust-temporal-jitter-above-preferred-bound")
        }

        if (
            domainConfidence(frames) <
            1.0 - config.unknownDomainPenalty
        ) {
            add("timestamp-domain-confidence-reduced")
        }

        if (confidence < config.minimumConfidence) {
            add("group-alignment-confidence-low")
        }
    }.distinct()

    private fun buildRecommendations(
        frames: List<FrameInput>,
        spreadNanos: Long,
        maximumAbsoluteOffsetNanos: Long,
        acceptedDeviceCount: Int,
    ): List<String> = buildList {
        if (acceptedDeviceCount < 2) {
            add(
                "Acquire a second independently validated camera frame before triangulating.",
            )
        }

        if (
            frames.any {
                it.domain ==
                    TimestampDomain.UNKNOWN
            }
        ) {
            add(
                "Prefer a sensor or documented cross-device-comparable timebase.",
            )
        }

        if (
            frames.any {
                !it.sourceIsComparableAcrossDevices
            }
        ) {
            add(
                "Measure a per-device clock offset during the session.",
            )
        }

        if (
            spreadNanos >
            config.preferredSpreadNanos
        ) {
            add(
                "Tighten frame pairing before multi-view reconstruction.",
            )
        }

        if (
            maximumAbsoluteOffsetNanos >
            config.preferredResidualNanos
        ) {
            add(
                "Run a synchronization exchange again before calibration.",
            )
        }

        if (
            frames.any {
                val model = it.rollingShutter
                model != null &&
                    model.readoutTimeNanos > 0L
            }
        ) {
            add(
                "Apply row-aware rolling-shutter compensation to fast-moving subjects.",
            )
        }
    }.distinct()

    private fun robustClockDriftPpm(
        state: DeviceState,
    ): Double? {
        val samples =
            state.offsetHistory.toList()
        if (samples.size < 3) return null

        val slopes =
            ArrayList<Double>(
                samples.size * (samples.size - 1) / 2,
            )

        for (i in 0 until samples.lastIndex) {
            val first = samples[i]
            for (j in i + 1 until samples.size) {
                val second = samples[j]
                val dtNanos =
                    second.first.toDouble() -
                        first.first.toDouble()
                if (dtNanos <= 1.0) continue

                val doNanos =
                    second.second.toDouble() -
                        first.second.toDouble()

                val ppm =
                    doNanos /
                        dtNanos *
                        1e6

                if (ppm.isFinite()) {
                    slopes += ppm
                }
            }
        }

        if (slopes.isEmpty()) return null
        slopes.sort()
        val middle = slopes.size / 2
        return if (slopes.size % 2 == 0) {
            (slopes[middle - 1] + slopes[middle]) * 0.5
        } else {
            slopes[middle]
        }
    }

    private fun inverse(
        value: Double,
        target: Double,
        maximum: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= target) return 1.0
        if (value >= maximum) return 0.0
        return (
            1.0 -
                (value - target) /
                (maximum - target).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun medianLong(
        values: List<Long>,
    ): Long {
        if (values.isEmpty()) return 0L
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2L
        } else {
            sorted[middle]
        }
    }

    private fun medianAbsoluteDeviationLong(
        values: List<Long>,
    ): Long {
        if (values.size < 3) return 0L
        val median = medianLong(values)
        val deviations = values.map { value ->
            abs(value - median)
        }
        return medianLong(deviations)
    }
}
