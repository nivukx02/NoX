package com.nivukx.nox.core.network

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.exp

/**
 * Bounded per-peer transport quality estimator.
 *
 * PING/PONG provides an application-visible round-trip measurement without assuming the peer clocks
 * are already synchronized. The estimator keeps only a small rolling sample window and turns RTT,
 * jitter and timeout loss into a bounded score suitable for distributed route selection.
 */
class PeerTransportQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumPeers: Int = 256,
        val maximumRttSamplesPerPeer: Int = 64,
        val maximumOutstandingProbesPerPeer: Int = 32,
        val targetRttMillis: Double = 8.0,
        val maximumRttMillis: Double = 250.0,
        val targetJitterMillis: Double = 2.0,
        val maximumJitterMillis: Double = 40.0,
        val timeoutNanos: Long = 250_000_000L,
        val ewmaAlpha: Double = 0.24,
        val minimumPayloadThroughputMbps: Double = 5.0,
        val targetPayloadThroughputMbps: Double = 80.0,
        val maximumPayloadThroughputMbps: Double = 500.0,
        val maximumPayloadSamplesPerPeer: Int = 64,
        val maximumOutstandingPayloadsPerPeer: Int = 128,
        val freshnessHalfLifeNanos: Long = 400_000_000L,
    ) {
        init {
            require(maximumPeers in 1..1024)
            require(maximumRttSamplesPerPeer in 8..512)
            require(maximumOutstandingProbesPerPeer in 2..128)
            require(targetRttMillis > 0.0)
            require(maximumRttMillis > targetRttMillis)
            require(targetJitterMillis > 0.0)
            require(maximumJitterMillis > targetJitterMillis)
            require(timeoutNanos > 0L)
            require(ewmaAlpha in 0.01..1.0)
            require(minimumPayloadThroughputMbps > 0.0)
            require(
                targetPayloadThroughputMbps >
                    minimumPayloadThroughputMbps,
            )
            require(
                maximumPayloadThroughputMbps >
                    targetPayloadThroughputMbps,
            )
            require(maximumPayloadSamplesPerPeer in 8..512)
            require(maximumOutstandingPayloadsPerPeer in 8..1024)
            require(freshnessHalfLifeNanos > 0L)
        }
    }

    data class Quality(
        val endpoint: String,
        val rttMedianMillis: Double,
        val rttP95Millis: Double,
        val jitterRmsMillis: Double,
        val packetLossRatio: Double,
        val successfulProbes: Long,
        val timedOutProbes: Long,
        val outstandingProbes: Int,
        val transportScore: Double,
        val payloadThroughputMbps: Double,
        val payloadFailureRatio: Double,
        val successfulPayloads: Long,
        val failedPayloads: Long,
        val lastSeenAgeMillis: Double,
        val freshnessScore: Double,
    )

    private data class PayloadState(
        val startedAtNanos: Long,
        val totalBytes: Long,
    )

    private data class PeerState(
        val sentAtNanosBySequence:
            LinkedHashMap<Long, Long> =
                LinkedHashMap(),
        val rttSamplesNanos:
            ArrayDeque<Long> =
                ArrayDeque(),
        var successfulProbes: Long = 0L,
        var timedOutProbes: Long = 0L,
        var jitterEwmaNanos: Double = 0.0,
        var previousRttNanos: Long? = null,
        var lastSeenNanos: Long = 0L,
        val payloadsById: LinkedHashMap<Long, PayloadState> = LinkedHashMap(),
        val payloadThroughputSamplesMbps: ArrayDeque<Double> = ArrayDeque(),
        var successfulPayloads: Long = 0L,
        var failedPayloads: Long = 0L,
    )

    private val peers =
        LinkedHashMap<String, PeerState>(
            16,
            0.75f,
            true,
        )

    @Synchronized
    fun observePayloadStarted(
        endpoint: String,
        payloadId: Long,
        totalBytes: Long,
        startedAtNanos: Long,
    ) {
        require(endpoint.isNotBlank())
        require(payloadId >= 0L)
        require(totalBytes > 0L)
        require(startedAtNanos >= 0L)

        val state = stateFor(endpoint)
        state.payloadsById[payloadId] =
            PayloadState(
                startedAtNanos = startedAtNanos,
                totalBytes = totalBytes,
            )

        while (
            state.payloadsById.size >
                config.maximumOutstandingPayloadsPerPeer
        ) {
            state.payloadsById.entries.firstOrNull()?.key?.let {
                state.payloadsById.remove(it)
            } ?: break
        }
    }

    @Synchronized
    fun observePayloadProgress(
        endpoint: String,
        payloadId: Long,
        bytesTransferred: Long,
        totalBytes: Long,
        status: Int,
        observedAtNanos: Long,
    ) {
        require(endpoint.isNotBlank())
        require(payloadId >= 0L)
        require(bytesTransferred >= 0L)
        require(totalBytes >= 0L)
        require(observedAtNanos >= 0L)

        val state = stateFor(endpoint)
        val payload = state.payloadsById[payloadId]

        if (payload == null) {
            if (status == 3) {
                state.payloadsById[payloadId] =
                    PayloadState(
                        startedAtNanos = observedAtNanos,
                        totalBytes = totalBytes,
                    )
            } else if (status == 2 || status == 4) {
                state.failedPayloads += 1L
            }
            return
        }

        val durationNanos =
            (observedAtNanos - payload.startedAtNanos)
                .coerceAtLeast(1L)

        val completed =
            status == 1 || status == 2 || status == 4

        if (!completed) return

        state.payloadsById.remove(payloadId)

        if (status == 1 && bytesTransferred > 0L) {
            val effectiveBytes =
                min(
                    bytesTransferred,
                    totalBytes.takeIf { it > 0L } ?: bytesTransferred,
                )
            val throughputMbps =
                effectiveBytes.toDouble() *
                    8.0 /
                    durationNanos.toDouble() /
                    1_000.0

            state.payloadThroughputSamplesMbps
                .addLast(throughputMbps.coerceIn(0.0, config.maximumPayloadThroughputMbps * 4.0))

            while (
                state.payloadThroughputSamplesMbps.size >
                    config.maximumPayloadSamplesPerPeer
            ) {
                state.payloadThroughputSamplesMbps.removeFirst()
            }

            state.successfulPayloads += 1L
        } else {
            state.failedPayloads += 1L
        }

        state.lastSeenNanos = observedAtNanos
    }

    @Synchronized
    fun observePingSent(
        endpoint: String,
        sequence: Long,
        sentAtNanos: Long,
    ) {
        require(endpoint.isNotBlank())
        require(sequence >= 0L)
        require(sentAtNanos >= 0L)

        val state =
            stateFor(endpoint)

        state.sentAtNanosBySequence[sequence] =
            sentAtNanos

        while (
            state.sentAtNanosBySequence.size >
                config.maximumOutstandingProbesPerPeer
        ) {
            val oldest =
                state.sentAtNanosBySequence
                    .entries
                    .firstOrNull()
                    ?.key
            if (oldest != null) {
                state.sentAtNanosBySequence.remove(oldest)
            } else {
                break
            }
        }
    }

    @Synchronized
    fun observePong(
        endpoint: String,
        sequence: Long,
        receivedAtNanos: Long,
    ) {
        require(endpoint.isNotBlank())
        require(sequence >= 0L)
        require(receivedAtNanos >= 0L)

        val state =
            stateFor(endpoint)

        val sentAt =
            state.sentAtNanosBySequence
                .remove(sequence)
                ?: return

        val rtt =
            receivedAtNanos -
                sentAt

        if (rtt < 0L) return

        state.successfulProbes += 1L
        state.lastSeenNanos =
            receivedAtNanos

        state.rttSamplesNanos.addLast(rtt)
        while (
            state.rttSamplesNanos.size >
                config.maximumRttSamplesPerPeer
        ) {
            state.rttSamplesNanos.removeFirst()
        }

        val previous =
            state.previousRttNanos

        if (previous != null) {
            val delta =
                abs(
                    rtt -
                        previous,
                ).toDouble()
            state.jitterEwmaNanos =
                config.ewmaAlpha * delta +
                    (
                        1.0 -
                            config.ewmaAlpha
                    ) *
                    state.jitterEwmaNanos
        }

        state.previousRttNanos =
            rtt
    }

    @Synchronized
    fun observePingTimeout(
        endpoint: String,
        sequence: Long,
        nowNanos: Long,
    ) {
        require(endpoint.isNotBlank())
        require(sequence >= 0L)
        require(nowNanos >= 0L)

        val state =
            stateFor(endpoint)

        val sentAt =
            state.sentAtNanosBySequence
                .remove(sequence)
                ?: return

        if (
            nowNanos -
                sentAt <
                config.timeoutNanos
        ) {
            return
        }

        state.timedOutProbes += 1L
        state.lastSeenNanos =
            nowNanos
    }

    @Synchronized
    fun expireOutstanding(
        nowNanos: Long,
    ) {
        require(nowNanos >= 0L)

        peers.values.forEach { state ->
            val iterator =
                state.sentAtNanosBySequence
                    .entries
                    .iterator()

            while (iterator.hasNext()) {
                val entry = iterator.next()
                if (
                    nowNanos -
                        entry.value >=
                        config.timeoutNanos
                ) {
                    iterator.remove()
                    state.timedOutProbes += 1L
                }
            }

            val payloadIterator =
                state.payloadsById.entries.iterator()
            while (payloadIterator.hasNext()) {
                val entry = payloadIterator.next()
                if (
                    nowNanos -
                        entry.value.startedAtNanos >=
                        config.timeoutNanos
                ) {
                    payloadIterator.remove()
                    state.failedPayloads += 1L
                }
            }
        }
    }

    @Synchronized
    fun estimatedPayloadThroughputMbps(
        endpoint: String,
    ): Double {
        require(endpoint.isNotBlank())

        val state =
            peers[endpoint]
                ?: return 0.0

        return state.payloadThroughputSamplesMbps
            .averageOrZero()
            .coerceIn(
                0.0,
                config.maximumPayloadThroughputMbps,
            )
    }

    private fun ArrayDeque<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    @Synchronized
    fun snapshot(
        endpoint: String,
        observedAtNanos: Long = System.nanoTime(),
    ): Quality {
        require(observedAtNanos >= 0L)
        require(endpoint.isNotBlank())

        val state =
            peers[endpoint]
                ?: return Quality(
                    endpoint = endpoint,
                    rttMedianMillis =
                        Double.POSITIVE_INFINITY,
                    rttP95Millis =
                        Double.POSITIVE_INFINITY,
                    jitterRmsMillis =
                        Double.POSITIVE_INFINITY,
                    packetLossRatio = 1.0,
                    successfulProbes = 0L,
                    timedOutProbes = 0L,
                    outstandingProbes = 0,
                    transportScore = 0.0,
                    payloadThroughputMbps = 0.0,
                    payloadFailureRatio = 1.0,
                    successfulPayloads = 0L,
                    failedPayloads = 0L,
                    lastSeenAgeMillis = Double.POSITIVE_INFINITY,
                    freshnessScore = 0.0,
                )

        val rtt =
            state.rttSamplesNanos
                .sorted()

        val median =
            percentile(
                rtt,
                0.50,
            )

        val p95 =
            percentile(
                rtt,
                0.95,
            )

        val jitter =
            state.jitterEwmaNanos

        val attempts =
            (
                state.successfulProbes +
                    state.timedOutProbes
            ).coerceAtLeast(1L)

        val loss =
            state.timedOutProbes.toDouble() /
                attempts.toDouble()

        val rttScore =
            if (rtt.isEmpty()) {
                0.0
            } else {
                (
                    1.0 -
                        max(
                            0.0,
                            p95 / 1_000_000.0 -
                                config.targetRttMillis,
                        ) /
                        (
                            config.maximumRttMillis -
                                config.targetRttMillis
                        )
                ).coerceIn(0.0, 1.0)
            }

        val jitterScore =
            (
                1.0 -
                    max(
                        0.0,
                        jitter / 1_000_000.0 -
                            config.targetJitterMillis,
                    ) /
                    (
                        config.maximumJitterMillis -
                            config.targetJitterMillis
                    )
            ).coerceIn(0.0, 1.0)

        val lossScore =
            (
                1.0 -
                    loss
            ).coerceIn(0.0, 1.0)

        val payloadThroughput =
            state.payloadThroughputSamplesMbps
                .sorted()
                .let { values ->
                    if (values.isEmpty()) 0.0
                    else percentileDouble(values, 0.50)
                }

        val payloadAttempts =
            (
                state.successfulPayloads +
                    state.failedPayloads
            ).coerceAtLeast(1L)

        val payloadFailureRatio =
            state.failedPayloads.toDouble() /
                payloadAttempts.toDouble()

        val throughputScore =
            if (payloadThroughput <= 0.0) {
                0.45
            } else if (
                payloadThroughput <=
                    config.minimumPayloadThroughputMbps
            ) {
                0.20
            } else {
                (
                    (
                        payloadThroughput -
                            config.minimumPayloadThroughputMbps
                    ) /
                        (
                            config.targetPayloadThroughputMbps -
                                config.minimumPayloadThroughputMbps
                        )
                ).coerceIn(0.0, 1.0)
            }

        val payloadLossScore =
            (
                1.0 -
                    payloadFailureRatio
            ).coerceIn(0.0, 1.0)

        val lastSeenAgeNanos =
            if (state.lastSeenNanos > 0L) {
                (
                    observedAtNanos -
                        state.lastSeenNanos
                    ).coerceAtLeast(0L)
            } else {
                Long.MAX_VALUE
            }

        val freshnessScore =
            if (lastSeenAgeNanos == Long.MAX_VALUE) {
                0.0
            } else {
                exp(
                    -lastSeenAgeNanos.toDouble() /
                        config.freshnessHalfLifeNanos.toDouble(),
                ).coerceIn(0.0, 1.0)
            }

        val score =
            (
                rttScore * 0.32 +
                    jitterScore * 0.18 +
                    lossScore * 0.18 +
                    throughputScore * 0.16 +
                    payloadLossScore * 0.07 +
                    freshnessScore * 0.09
            ).coerceIn(0.0, 1.0)

        return Quality(
            endpoint = endpoint,
            rttMedianMillis =
                median / 1_000_000.0,
            rttP95Millis =
                p95 / 1_000_000.0,
            jitterRmsMillis =
                jitter / 1_000_000.0,
            packetLossRatio = loss,
            successfulProbes =
                state.successfulProbes,
            timedOutProbes =
                state.timedOutProbes,
            outstandingProbes =
                state.sentAtNanosBySequence.size,
            transportScore = score,
            payloadThroughputMbps =
                payloadThroughput,
            payloadFailureRatio =
                payloadFailureRatio,
            successfulPayloads =
                state.successfulPayloads,
            failedPayloads =
                state.failedPayloads,
            lastSeenAgeMillis =
                if (lastSeenAgeNanos == Long.MAX_VALUE) {
                    Double.POSITIVE_INFINITY
                } else {
                    lastSeenAgeNanos.toDouble() / 1_000_000.0
                },
            freshnessScore =
                freshnessScore,
        )
    }

    @Synchronized
    fun outstandingPayloadCount(
        endpoint: String,
    ): Int =
        peers[endpoint]
            ?.payloadsById
            ?.size
            ?: 0

    @Synchronized
    fun outstandingProbeCount(
        endpoint: String,
    ): Int =
        peers[endpoint]
            ?.sentAtNanosBySequence
            ?.size
            ?: 0

    @Synchronized
    fun clearEndpoint(
        endpoint: String,
    ) {
        require(endpoint.isNotBlank())
        peers.remove(endpoint)
    }

    @Synchronized
    fun clear() {
        peers.clear()
    }

    private fun stateFor(
        endpoint: String,
    ): PeerState {
        val existing =
            peers[endpoint]
        if (existing != null) return existing

        while (
            peers.size >=
                config.maximumPeers
        ) {
            peers.entries
                .firstOrNull()
                ?.key
                ?.let(peers::remove)
        }

        return PeerState().also {
            peers[endpoint] = it
        }
    }

    private fun percentileDouble(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        val position =
            probability.coerceIn(0.0, 1.0) *
                (values.size - 1)
        val lower = position.toInt()
        val upper = min(values.lastIndex, lower + 1)
        val fraction = position - lower.toDouble()
        return values[lower] * (1.0 - fraction) +
            values[upper] * fraction
    }

    private fun percentile(
        values: List<Long>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        val position =
            probability.coerceIn(0.0, 1.0) *
                (values.size - 1)
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

        return values[lower].toDouble() *
            (1.0 - fraction) +
            values[upper].toDouble() *
                fraction
    }
}
