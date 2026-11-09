package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Evaluates multi-device synchronization quality for motion capture.
 *
 * Synchronization is treated as a geometric input, not just a networking metric. A camera with a
 * beautiful calibration transform is still harmful to 3D reconstruction when its clock offset,
 * round-trip latency, packet loss or sample age make the observations temporally incompatible.
 *
 * The engine maintains a short EWMA history per peer and reports both absolute quality and
 * cross-device agreement. The output is deliberately independent of the wire protocol so it can be
 * fed by Nearby Connections, Wi-Fi, BLE-assisted transport, replay logs or offline validation.
 */
class MultiCameraSynchronizationQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumAbsoluteOffsetNanos: Long = 25_000_000L,
        val targetAbsoluteOffsetNanos: Long = 3_000_000L,
        val maximumRoundTripNanos: Long = 120_000_000L,
        val targetRoundTripNanos: Long = 35_000_000L,
        val maximumPacketLoss: Double = 0.10,
        val targetPacketLoss: Double = 0.015,
        val maximumSampleAgeMillis: Double = 650.0,
        val targetSampleAgeMillis: Double = 90.0,
        val maximumPeerSpreadNanos: Long = 18_000_000L,
        val historyPerPeer: Int = 24,
        val ewmaAlpha: Double = 0.24,
        val minimumPeerConfidence: Double = 0.30,
        val maximumSubframeOffsetNanos: Double = 6_000_000.0,
        val targetSubframeOffsetNanos: Double = 1_000_000.0,
        val minimumSubframeConfidence: Double = 0.70,
        val precisionContractEnabled: Boolean = false,
        val precisionMaximumAbsoluteOffsetNanos: Long = 1_000_000L,
        val precisionMaximumPeerSpreadNanos: Long = 750_000L,
        val precisionMaximumSubframeOffsetNanos: Double = 500_000.0,
        val precisionMinimumSubframeConfidence: Double = 0.90,
        val precisionMinimumConfidence: Double = 0.90,
        val precisionMaximumClockDriftPpm: Double = 80.0,
        val precisionMinimumStableEvaluations: Int = 3,
        val warningMaximumClockDriftPpm: Double = 150.0,
    ) {
        init {
            require(maximumAbsoluteOffsetNanos > targetAbsoluteOffsetNanos)
            require(targetAbsoluteOffsetNanos >= 0L)
            require(maximumRoundTripNanos > targetRoundTripNanos)
            require(targetRoundTripNanos >= 0L)
            require(maximumPacketLoss in 0.0..1.0)
            require(targetPacketLoss in 0.0..maximumPacketLoss)
            require(maximumSampleAgeMillis > targetSampleAgeMillis)
            require(targetSampleAgeMillis >= 0.0)
            require(maximumPeerSpreadNanos > 0L)
            require(historyPerPeer >= 4)
            require(ewmaAlpha in 0.01..1.0)
            require(minimumPeerConfidence in 0.0..1.0)
            require(
                maximumSubframeOffsetNanos >
                    targetSubframeOffsetNanos
            )
            require(targetSubframeOffsetNanos >= 0.0)
            require(minimumSubframeConfidence in 0.0..1.0)
            require(precisionMaximumAbsoluteOffsetNanos > 0L)
            require(precisionMaximumPeerSpreadNanos > 0L)
            require(precisionMaximumSubframeOffsetNanos > 0.0)
            require(precisionMinimumSubframeConfidence in 0.0..1.0)
            require(precisionMinimumConfidence in 0.0..1.0)
            require(precisionMaximumClockDriftPpm > 0.0)
            require(precisionMinimumStableEvaluations >= 1)
            require(warningMaximumClockDriftPpm >= precisionMaximumClockDriftPpm)
        }
    }

    data class PeerSample(
        val deviceId: DeviceId,
        val offsetNanos: Long,
        val roundTripNanos: Long,
        val packetLoss: Double,
        val sampleAgeMillis: Double,
        val subframeOffsetNanos: Double = 0.0,
        val subframeConfidence: Double = 1.0,
    ) {
        init {
            require(roundTripNanos >= 0L)
            require(packetLoss in 0.0..1.0)
            require(sampleAgeMillis >= 0.0)
            require(subframeOffsetNanos.isFinite())
            require(subframeConfidence in 0.0..1.0)
        }
    }

    data class PeerQuality(
        val deviceId: DeviceId,
        val offsetNanos: Long,
        val offsetJitterNanos: Long,
        val roundTripNanos: Long,
        val packetLoss: Double,
        val sampleAgeMillis: Double,
        val confidence: Double,
        val healthy: Boolean,
        val subframeOffsetNanos: Double,
        val subframeConfidence: Double,
        val warnings: List<String>,
        val clockDriftPpm: Double = 0.0,
    )

    enum class State {
        NO_PEERS,
        SYNCHRONIZED,
        DEGRADED,
        UNSAFE,
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val confidence: Double,
        val meanAbsoluteOffsetNanos: Long,
        val medianAbsoluteOffsetNanos: Long,
        val maximumPeerSpreadNanos: Long,
        val meanRoundTripNanos: Long,
        val meanPacketLoss: Double,
        val meanSampleAgeMillis: Double,
        val meanSubframeOffsetNanos: Double,
        val meanSubframeConfidence: Double,
        val peers: List<PeerQuality>,
        val warnings: List<String>,
        val recommendations: List<String>,
        val precisionQualified: Boolean = false,
        val precisionBlockers: List<String> = emptyList(),
        val maximumClockDriftPpm: Double = 0.0,
        val clockGraphConfidence: Double? = null,
        val clockGraphCoverage: Double? = null,
        val clockGraphCycleConsistency: Double? = null,
        val clockGraphAccepted: Boolean? = null,
        val precisionQualifiedStreak: Int = 0,
    ) {
        val healthy: Boolean
            get() = state == State.SYNCHRONIZED || state == State.NO_PEERS

        val unsafe: Boolean
            get() = state == State.UNSAFE
    }

    private data class TimedOffset(
        val timestampNanos: Long,
        val offsetNanos: Long,
    )

    private data class PeerState(
        var offsetEwmaNanos: Double = 0.0,
        var jitterEwmaNanos: Double = 0.0,
        var lastOffsetNanos: Long = 0L,
        var timestampNanos: Long = Long.MIN_VALUE,
        val history: ArrayDeque<Long> = ArrayDeque(),
        val timedOffsets: ArrayDeque<TimedOffset> = ArrayDeque(),
    )

    private val states = HashMap<DeviceId, PeerState>()
    private var precisionQualifiedStreak = 0

    @Synchronized
    fun evaluate(
        timestampNanos: Long,
        samples: List<PeerSample>,
        clockGraph: DistributedClockAuthorityEngine.Result? = null,
    ): Result {
        if (samples.isEmpty()) {
            precisionQualifiedStreak = 0
            return Result(
                timestampNanos = timestampNanos,
                state = State.NO_PEERS,
                confidence = 1.0,
                meanAbsoluteOffsetNanos = 0L,
                medianAbsoluteOffsetNanos = 0L,
                maximumPeerSpreadNanos = 0L,
                meanRoundTripNanos = 0L,
                meanPacketLoss = 0.0,
                meanSampleAgeMillis = 0.0,
                meanSubframeOffsetNanos = 0.0,
                meanSubframeConfidence = 1.0,
                peers = emptyList(),
                warnings =
                    if (config.precisionContractEnabled) {
                        listOf(
                            "precision-sync-no-peer-evidence",
                        )
                    } else {
                        emptyList()
                    },
                recommendations =
                    if (config.precisionContractEnabled) {
                        listOf(
                            "Connect and align at least two camera peers before precision qualification.",
                        )
                    } else {
                        emptyList()
                    },
                precisionQualified =
                    !config.precisionContractEnabled,
                precisionBlockers =
                    if (config.precisionContractEnabled) {
                        listOf("precision-sync-no-peer-evidence")
                    } else {
                        emptyList()
                    },
                maximumClockDriftPpm = 0.0,
                clockGraphConfidence = clockGraph?.confidence,
                clockGraphCoverage = clockGraph?.graphCoverage,
                clockGraphCycleConsistency =
                    clockGraph?.cycleConsistency,
                clockGraphAccepted = clockGraph?.accepted,
                precisionQualifiedStreak = precisionQualifiedStreak,
            )
        }

        val peerQuality = samples
            .distinctBy { it.deviceId }
            .map { sample ->
                evaluatePeer(
                    timestampNanos = timestampNanos,
                    sample = sample,
                )
            }

        val offsets = peerQuality
            .map { abs(it.offsetNanos) }
            .sorted()

        val spread =
            if (offsets.isEmpty()) {
                0L
            } else {
                offsets.last() -
                    offsets.first()
            }

        val meanOffset =
            if (offsets.isEmpty()) {
                0L
            } else {
                offsets
                    .map { it.toDouble() }
                    .average()
                    .toLong()
            }

        val medianOffset =
            medianLong(offsets)

        val meanRoundTrip =
            peerQuality
                .map { it.roundTripNanos }
                .averageLongOrZero()
                .toLong()

        val meanLoss =
            peerQuality
                .map { it.packetLoss }
                .averageOrZero()

        val meanAge =
            peerQuality
                .map { it.sampleAgeMillis }
                .averageOrZero()

        val meanSubframeOffset =
            peerQuality
                .map {
                    abs(
                        it.subframeOffsetNanos,
                    )
                }
                .averageOrZero()

        val meanSubframeConfidence =
            peerQuality
                .map {
                    it.subframeConfidence
                }
                .averageOrZero()

        val maximumClockDriftPpm =
            peerQuality
                .map { it.clockDriftPpm }
                .filter(Double::isFinite)
                .maxOrNull()
                ?.coerceAtLeast(0.0)
                ?: 0.0

        val clockDriftScore =
            inverse(
                value = maximumClockDriftPpm,
                target = config.precisionMaximumClockDriftPpm * 0.35,
                maximum = config.warningMaximumClockDriftPpm,
            )

        val offsetScore =
            inverse(
                value = meanOffset.toDouble(),
                target = config.targetAbsoluteOffsetNanos.toDouble(),
                maximum = config.maximumAbsoluteOffsetNanos.toDouble(),
            )

        val spreadScore =
            inverse(
                value = spread.toDouble(),
                target = (
                    config.targetAbsoluteOffsetNanos * 2L
                ).toDouble(),
                maximum = config.maximumPeerSpreadNanos.toDouble(),
            )

        val roundTripScore =
            inverse(
                value = meanRoundTrip.toDouble(),
                target = config.targetRoundTripNanos.toDouble(),
                maximum = config.maximumRoundTripNanos.toDouble(),
            )

        val packetLossScore =
            inverse(
                value = meanLoss,
                target = config.targetPacketLoss,
                maximum = config.maximumPacketLoss,
            )

        val ageScore =
            inverse(
                value = meanAge,
                target = config.targetSampleAgeMillis,
                maximum = config.maximumSampleAgeMillis,
            )

        val subframeOffsetScore =
            inverse(
                value = meanSubframeOffset,
                target = config.targetSubframeOffsetNanos,
                maximum = config.maximumSubframeOffsetNanos,
            )

        val subframeConfidenceScore =
            meanSubframeConfidence
                .coerceIn(0.0, 1.0)

        val peerScore =
            peerQuality
                .map { it.confidence }
                .averageOrZero()

        val localConfidence = (
            offsetScore * 0.24 +
                spreadScore * 0.14 +
                roundTripScore * 0.12 +
                packetLossScore * 0.11 +
                ageScore * 0.09 +
                subframeOffsetScore * 0.14 +
                subframeConfidenceScore * 0.08 +
                peerScore * 0.06 +
                clockDriftScore * 0.02
            ).coerceIn(0.0, 1.0)

        val graphConfidence =
            clockGraph?.confidence?.coerceIn(0.0, 1.0)

        val confidence =
            if (graphConfidence != null) {
                (
                    localConfidence * 0.72 +
                        graphConfidence * 0.28
                ).coerceIn(0.0, 1.0)
            } else {
                localConfidence
            }

        val measuredPrecisionBlockers = buildList {
            if (!config.precisionContractEnabled) {
                // Standard mode deliberately remains less restrictive.
            } else {
                if (meanOffset >
                    config.precisionMaximumAbsoluteOffsetNanos
                ) {
                    add("precision-clock-offset-exceeds-one-millisecond")
                }
                if (spread >
                    config.precisionMaximumPeerSpreadNanos
                ) {
                    add("precision-peer-spread-exceeds-750-microseconds")
                }
                if (meanSubframeOffset >
                    config.precisionMaximumSubframeOffsetNanos
                ) {
                    add("precision-subframe-offset-exceeds-500-microseconds")
                }
                if (
                    meanSubframeConfidence <
                        config.precisionMinimumSubframeConfidence
                ) {
                    add("precision-subframe-confidence-below-contract")
                }
                if (
                    confidence <
                        config.precisionMinimumConfidence
                ) {
                    add("precision-synchronization-confidence-below-contract")
                }
                if (
                    maximumClockDriftPpm >
                        config.precisionMaximumClockDriftPpm
                ) {
                    add("precision-clock-drift-exceeds-contract")
                }
                if (
                    clockGraph != null &&
                    (
                        !clockGraph.accepted ||
                            clockGraph.graphCoverage < 0.90 ||
                            clockGraph.cycleConsistency < 0.85
                    )
                ) {
                    add(
                        "precision-multi-master-clock-consensus-below-contract",
                    )
                }
            }
        }

        if (config.precisionContractEnabled) {
            precisionQualifiedStreak =
                if (measuredPrecisionBlockers.isEmpty()) {
                    min(
                        config.precisionMinimumStableEvaluations,
                        precisionQualifiedStreak + 1,
                    )
                } else {
                    0
                }
        } else {
            precisionQualifiedStreak = 0
        }

        val precisionBlockers = buildList {
            addAll(measuredPrecisionBlockers)
            if (
                config.precisionContractEnabled &&
                precisionQualifiedStreak <
                    config.precisionMinimumStableEvaluations
            ) {
                add("precision-sync-stability-window-incomplete")
            }
        }.distinct()

        val precisionQualified =
            !config.precisionContractEnabled ||
                (
                    precisionBlockers.isEmpty() &&
                        (
                            clockGraph == null ||
                                (
                                    clockGraph.accepted &&
                                        clockGraph.graphCoverage >= 0.90 &&
                                        clockGraph.cycleConsistency >= 0.85
                                )
                        )
                )

        val warnings = buildList {
            precisionBlockers.forEach {
                add(it)
            }
            if (meanOffset >
                config.targetAbsoluteOffsetNanos
            ) {
                add("mean clock offset exceeds target")
            }

            if (spread >
                config.maximumPeerSpreadNanos
            ) {
                add("peer clock spread exceeds capture envelope")
            }

            if (meanRoundTrip >
                config.maximumRoundTripNanos
            ) {
                add("network round-trip time is too high")
            }

            if (meanLoss >
                config.maximumPacketLoss
            ) {
                add("packet loss exceeds synchronization target")
            }

            if (meanAge >
                config.maximumSampleAgeMillis
            ) {
                add("synchronization samples are stale")
            }

            if (
                meanSubframeOffset >
                    config.maximumSubframeOffsetNanos
            ) {
                add("sub-frame temporal offset exceeds capture envelope")
            }

            if (
                meanSubframeConfidence <
                    config.minimumSubframeConfidence
            ) {
                add("sub-frame temporal confidence is too low")
            }
            if (maximumClockDriftPpm > config.warningMaximumClockDriftPpm) {
                add("clock drift exceeds the temporal warning envelope")
            }
            if (
                clockGraph != null &&
                !clockGraph.accepted
            ) {
                add(
                    "multi-master clock graph is not reference-consistent",
                )
            }

            peerQuality
                .filter { !it.healthy }
                .take(4)
                .forEach {
                    add(
                        "peer " +
                            it.deviceId.value.takeLast(8) +
                            " degraded: " +
                            it.warnings.joinToString("; "),
                    )
                }
        }

        val recommendations = buildList {
            if (offsetScore < 0.60) {
                add(
                    "Refresh clock probes before the next multi-camera take.",
                )
            }

            if (spreadScore < 0.60) {
                add(
                    "Re-synchronize all cameras from the same master clock.",
                )
            }

            if (roundTripScore < 0.60) {
                add(
                    "Reduce network contention or move peers to a lower-latency transport.",
                )
            }

            if (packetLossScore < 0.60) {
                add(
                    "Stabilize peer connectivity before relying on dense multi-view reconstruction.",
                )
            }

            if (ageScore < 0.60) {
                add(
                    "Increase synchronization probe cadence so capture metadata stays fresh.",
                )
            }

            if (
                subframeOffsetScore <
                    0.60 ||
                subframeConfidenceScore <
                    config.minimumSubframeConfidence
            ) {
                add(
                    "Run sub-frame motion alignment before trusting metric multi-view output.",
                )
            }

            if (confidence < 0.45) {
                add(
                    "Do not treat the rig as metric-synchronized until clock quality recovers.",
                )
            }
        }

        val state = when {
            precisionBlockers.isNotEmpty() ->
                State.UNSAFE

            confidence < 0.35 ->
                State.UNSAFE

            confidence < 0.65 ||
                warnings.isNotEmpty() ->
                State.DEGRADED

            else ->
                State.SYNCHRONIZED
        }

        return Result(
            timestampNanos = timestampNanos,
            state = state,
            confidence = confidence,
            meanAbsoluteOffsetNanos = meanOffset,
            precisionQualified = precisionQualified,
            precisionBlockers = precisionBlockers.distinct(),
            maximumClockDriftPpm = maximumClockDriftPpm,
            clockGraphConfidence = clockGraph?.confidence,
            clockGraphCoverage = clockGraph?.graphCoverage,
            clockGraphCycleConsistency =
                clockGraph?.cycleConsistency,
            clockGraphAccepted = clockGraph?.accepted,
            precisionQualifiedStreak = precisionQualifiedStreak,
            medianAbsoluteOffsetNanos = medianOffset,
            maximumPeerSpreadNanos = spread,
            meanRoundTripNanos = meanRoundTrip,
            meanPacketLoss = meanLoss,
            meanSampleAgeMillis = meanAge,
            meanSubframeOffsetNanos = meanSubframeOffset,
            meanSubframeConfidence = meanSubframeConfidence,
            peers = peerQuality,
            warnings = warnings.distinct(),
            recommendations = recommendations.distinct(),
        )
    }

    @Synchronized
    fun reset() {
        states.clear()
        precisionQualifiedStreak = 0
    }

    private fun evaluatePeer(
        timestampNanos: Long,
        sample: PeerSample,
    ): PeerQuality {
        val state =
            states.getOrPut(sample.deviceId) {
                PeerState(
                    offsetEwmaNanos =
                        sample.offsetNanos.toDouble(),
                    lastOffsetNanos =
                        sample.offsetNanos,
                )
            }

        val timestampMonotonic =
            state.timestampNanos == Long.MIN_VALUE ||
                timestampNanos >= state.timestampNanos

        val offsetDelta =
            abs(
                sample.offsetNanos -
                    state.lastOffsetNanos,
            )

        state.offsetEwmaNanos =
            state.offsetEwmaNanos * (
                1.0 - config.ewmaAlpha
                ) +
                sample.offsetNanos *
                config.ewmaAlpha

        state.jitterEwmaNanos =
            state.jitterEwmaNanos * (
                1.0 - config.ewmaAlpha
                ) +
                offsetDelta *
                config.ewmaAlpha

        state.lastOffsetNanos =
            sample.offsetNanos

        if (timestampMonotonic) {
            state.timestampNanos = timestampNanos
            state.history.addLast(sample.offsetNanos)
            state.timedOffsets.addLast(
                TimedOffset(
                    timestampNanos = timestampNanos,
                    offsetNanos = sample.offsetNanos,
                ),
            )
        }

        while (state.history.size > config.historyPerPeer) {
            state.history.removeFirst()
        }

        while (state.timedOffsets.size > config.historyPerPeer) {
            state.timedOffsets.removeFirst()
        }

        val clockDriftPpm =
            estimateClockDriftPpm(state.timedOffsets)

        val driftScore =
            inverse(
                value = clockDriftPpm,
                target = config.precisionMaximumClockDriftPpm * 0.35,
                maximum = config.warningMaximumClockDriftPpm,
            )

        val offsetScore =
            inverse(
                value = abs(state.offsetEwmaNanos),
                target = config.targetAbsoluteOffsetNanos.toDouble(),
                maximum = config.maximumAbsoluteOffsetNanos.toDouble(),
            )

        val jitterScore =
            inverse(
                value = state.jitterEwmaNanos,
                target = (
                    config.targetAbsoluteOffsetNanos * 0.75
                ).toDouble(),
                maximum = config.maximumAbsoluteOffsetNanos.toDouble(),
            )

        val roundTripScore =
            inverse(
                value = sample.roundTripNanos.toDouble(),
                target = config.targetRoundTripNanos.toDouble(),
                maximum = config.maximumRoundTripNanos.toDouble(),
            )

        val lossScore =
            inverse(
                value = sample.packetLoss,
                target = config.targetPacketLoss,
                maximum = config.maximumPacketLoss,
            )

        val ageScore =
            inverse(
                value = sample.sampleAgeMillis,
                target = config.targetSampleAgeMillis,
                maximum = config.maximumSampleAgeMillis,
            )

        val subframeOffsetScore =
            inverse(
                value =
                    abs(
                        sample.subframeOffsetNanos,
                    ),
                target =
                    config.targetSubframeOffsetNanos,
                maximum =
                    config.maximumSubframeOffsetNanos,
            )

        val subframeConfidenceScore =
            sample.subframeConfidence
                .coerceIn(0.0, 1.0)

        val confidence = (
            offsetScore * 0.24 +
                jitterScore * 0.13 +
                roundTripScore * 0.13 +
                lossScore * 0.16 +
                ageScore * 0.14 +
                subframeOffsetScore * 0.10 +
                subframeConfidenceScore * 0.08 +
                driftScore * 0.02
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (!timestampMonotonic) {
                add("out-of-order-timestamp")
            }
            if (offsetScore < 0.60) {
                add("offset")
            }
            if (jitterScore < 0.60) {
                add("jitter")
            }
            if (roundTripScore < 0.60) {
                add("round-trip")
            }
            if (lossScore < 0.60) {
                add("packet-loss")
            }
            if (ageScore < 0.60) {
                add("stale")
            }
            if (
                subframeOffsetScore <
                    0.60
            ) {
                add("sub-frame-offset")
            }
            if (
                subframeConfidenceScore <
                    config.minimumSubframeConfidence
            ) {
                add("sub-frame-confidence")
            }
            if (
                clockDriftPpm >
                    config.warningMaximumClockDriftPpm
            ) {
                add("clock-drift")
            }
        }

        return PeerQuality(
            deviceId = sample.deviceId,
            offsetNanos = sample.offsetNanos,
            offsetJitterNanos =
                state.jitterEwmaNanos
                    .toLong()
                    .coerceAtLeast(0L),
            roundTripNanos = sample.roundTripNanos,
            packetLoss = sample.packetLoss,
            sampleAgeMillis = sample.sampleAgeMillis,
            confidence = confidence,
            healthy =
                confidence >=
                    config.minimumPeerConfidence &&
                    warnings.isEmpty(),
            subframeOffsetNanos =
                sample.subframeOffsetNanos,
            subframeConfidence =
                sample.subframeConfidence,
            warnings = warnings,
            clockDriftPpm = clockDriftPpm,
        )
    }

    private fun estimateClockDriftPpm(
        history: ArrayDeque<TimedOffset>,
    ): Double {
        if (history.size < 3) return 0.0

        val first = history.first().timestampNanos
        val points = history.map {
            (
                (it.timestampNanos - first).coerceAtLeast(0L)
                    .toDouble() / 1_000_000_000.0
            ) to it.offsetNanos.toDouble()
        }

        val meanX = points.map { it.first }.average()
        val meanY = points.map { it.second }.average()

        var numerator = 0.0
        var denominator = 0.0
        for ((x, y) in points) {
            val dx = x - meanX
            numerator += dx * (y - meanY)
            denominator += dx * dx
        }

        if (!denominator.isFinite() || denominator <= 1e-12) {
            return 0.0
        }

        // Offset slope is nanoseconds/second. One ppm corresponds to
        // 1 microsecond/second = 1000 nanoseconds/second.
        val nanosecondsPerSecond = numerator / denominator
        return abs(nanosecondsPerSecond / 1000.0)
            .takeIf(Double::isFinite)
            ?.coerceAtMost(1_000_000.0)
            ?: Double.POSITIVE_INFINITY
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
                (
                    value - target
                ) /
                    (
                        maximum - target
                    ).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun medianLong(
        values: List<Long>,
    ): Long {
        if (values.isEmpty()) return 0L
        return if (values.size % 2 == 0) {
            (
                values[values.size / 2 - 1] +
                    values[values.size / 2]
                ) / 2L
        } else {
            values[values.size / 2]
        }
    }

    private fun List<Long>.averageLongOrZero(): Double =
        if (isEmpty()) 0.0
        else map(Long::toDouble).average()

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0
        else average()
}
