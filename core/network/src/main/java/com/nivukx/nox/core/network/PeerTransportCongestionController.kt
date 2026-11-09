package com.nivukx.nox.core.network

import kotlin.math.max
import kotlin.math.min

/**
 * Per-peer weighted byte-budget controller for heterogeneous distributed capture links.
 *
 * Unlike packet-count shedding, this controller accounts for actual payload bytes and measured
 * throughput. A reserved critical lane protects synchronization, control, and reconstruction-critical
 * evidence from bursty low-priority traffic. It never blocks the caller; admission is deterministic
 * and fail-fast so the capture pipeline can choose a lower-cost evidence representation.
 */
class PeerTransportCongestionController(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumPeers: Int = 256,
        val bootstrapThroughputMbps: Double = 80.0,
        val minimumThroughputMbps: Double = 5.0,
        val maximumThroughputMbps: Double = 1_000.0,
        val burstSeconds: Double = 0.20,
        val criticalReserveFraction: Double = 0.45,
        val highShare: Double = 0.28,
        val normalShare: Double = 0.20,
        val normalBurstMultiplier: Double = 1.35,
        val lowShare: Double = 0.07,
        val maximumIdleAgeNanos: Long = 5_000_000_000L,
    ) {
        init {
            require(maximumPeers in 1..1024)
            require(
                bootstrapThroughputMbps >=
                    minimumThroughputMbps,
            )
            require(
                maximumThroughputMbps >
                    bootstrapThroughputMbps,
            )
            require(burstSeconds in 0.01..2.0)
            require(criticalReserveFraction in 0.10..0.80)
            require(highShare in 0.05..0.80)
            require(normalShare in 0.05..0.80)
            require(lowShare in 0.01..0.50)
            require(
                criticalReserveFraction +
                    highShare +
                    normalShare +
                    lowShare <= 1.0,
            )
            require(normalBurstMultiplier in 1.0..4.0)
            require(maximumIdleAgeNanos > 0L)
        }
    }

    private data class Bucket(
        var tokensBytes: Double,
        val capacityBytes: Double,
    ) {
        fun refill(
            bytesPerSecond: Double,
            elapsedSeconds: Double,
        ) {
            tokensBytes =
                min(
                    capacityBytes,
                    tokensBytes +
                        bytesPerSecond *
                        elapsedSeconds,
                )
        }

        fun tryConsume(
            bytes: Int,
        ): Boolean {
            if (bytes <= 0) return true
            val requested =
                bytes.toDouble()
            if (tokensBytes + 1e-9 < requested) {
                return false
            }
            tokensBytes -= requested
            return true
        }

        fun refund(bytes: Int) {
            if (bytes <= 0) return
            tokensBytes =
                min(
                    capacityBytes,
                    tokensBytes + bytes.toDouble(),
                )
        }
    }

    private data class State(
        var lastUpdateNanos: Long,
        var measuredThroughputMbps: Double,
        val critical: Bucket,
        val high: Bucket,
        val normal: Bucket,
        val low: Bucket,
    )

    private val states =
        LinkedHashMap<String, State>(
            32,
            0.75f,
            true,
        )

    @Synchronized
    fun tryAcquire(
        endpoint: String,
        type: NoXWireProtocol.Type,
        packetBytes: Int,
        measuredThroughputMbps: Double,
        reconstructionCritical: Boolean = false,
        nowNanos: Long = System.nanoTime(),
    ): Boolean {
        require(endpoint.isNotBlank())
        require(packetBytes >= 0)
        require(nowNanos >= 0L)

        pruneIdle(nowNanos)

        val throughput =
            when {
                measuredThroughputMbps.isFinite() &&
                    measuredThroughputMbps > 0.0 ->
                    measuredThroughputMbps.coerceIn(
                        config.minimumThroughputMbps,
                        config.maximumThroughputMbps,
                    )
                else ->
                    config.bootstrapThroughputMbps
            }

        val state =
            states[endpoint]
                ?: createState(
                    endpoint = endpoint,
                    throughputMbps = throughput,
                    nowNanos = nowNanos,
                )

        val elapsedSeconds =
            (
                nowNanos -
                    state.lastUpdateNanos
                ).coerceAtLeast(0L)
                .toDouble() /
                1_000_000_000.0

        refill(
            state = state,
            throughputMbps = throughput,
            elapsedSeconds = elapsedSeconds,
        )

        state.lastUpdateNanos = nowNanos
        state.measuredThroughputMbps =
            throughput

        val trafficClass =
            PeerTransportSheddingPolicy()
                .classify(type)

        return when {
            reconstructionCritical ||
                trafficClass ==
                PeerTransportSheddingPolicy.Class.CRITICAL ->
                state.critical.tryConsume(packetBytes)

            trafficClass ==
                PeerTransportSheddingPolicy.Class.HIGH ->
                state.high.tryConsume(packetBytes)

            trafficClass ==
                PeerTransportSheddingPolicy.Class.LOW ->
                state.low.tryConsume(packetBytes)

            else ->
                state.normal.tryConsume(packetBytes)
        }
    }

    @Synchronized
    fun refund(
        endpoint: String,
        type: NoXWireProtocol.Type,
        packetBytes: Int,
        reconstructionCritical: Boolean = false,
    ) {
        require(endpoint.isNotBlank())
        require(packetBytes >= 0)

        val state = states[endpoint]
            ?: return

        val trafficClass =
            PeerTransportSheddingPolicy()
                .classify(type)

        val bucket =
            when {
                reconstructionCritical ||
                    trafficClass ==
                    PeerTransportSheddingPolicy.Class.CRITICAL ->
                    state.critical

                trafficClass ==
                    PeerTransportSheddingPolicy.Class.HIGH ->
                    state.high

                trafficClass ==
                    PeerTransportSheddingPolicy.Class.LOW ->
                    state.low

                else ->
                    state.normal
            }

        bucket.refund(packetBytes)
    }

    @Synchronized
    fun resetEndpoint(endpoint: String) {
        require(endpoint.isNotBlank())
        states.remove(endpoint)
    }

    @Synchronized
    fun reset() {
        states.clear()
    }

    @Synchronized
    fun stateCount(): Int =
        states.size

    @Synchronized
    fun measuredThroughputMbps(
        endpoint: String,
    ): Double =
        states[endpoint]?.measuredThroughputMbps
            ?: config.bootstrapThroughputMbps

    private fun createState(
        endpoint: String,
        throughputMbps: Double,
        nowNanos: Long,
    ): State {
        require(endpoint.isNotBlank())

        if (
            states.size >=
                config.maximumPeers
        ) {
            states.entries
                .firstOrNull()
                ?.key
                ?.let(states::remove)
        }

        val bytesPerSecond =
            throughputMbps *
                125_000.0

        val burstBytes =
            bytesPerSecond *
                config.burstSeconds

        fun bucket(
            share: Double,
            multiplier: Double = 1.0,
        ): Bucket {
            val capacity =
                max(
                    1_024.0,
                    burstBytes *
                        share *
                        multiplier,
                )
            return Bucket(
                tokensBytes = capacity,
                capacityBytes = capacity,
            )
        }

        return State(
            lastUpdateNanos = nowNanos,
            measuredThroughputMbps = throughputMbps,
            critical =
                bucket(
                    config.criticalReserveFraction,
                ),
            high =
                bucket(
                    config.highShare,
                ),
            normal =
                bucket(
                    config.normalShare,
                    config.normalBurstMultiplier,
                ),
            low =
                bucket(
                    config.lowShare,
                ),
        ).also {
            states[endpoint] = it
        }
    }

    private fun refill(
        state: State,
        throughputMbps: Double,
        elapsedSeconds: Double,
    ) {
        if (elapsedSeconds <= 0.0) return

        val bytesPerSecond =
            throughputMbps *
                125_000.0

        state.critical.refill(
            bytesPerSecond *
                config.criticalReserveFraction,
            elapsedSeconds,
        )
        state.high.refill(
            bytesPerSecond *
                config.highShare,
            elapsedSeconds,
        )
        state.normal.refill(
            bytesPerSecond *
                config.normalShare,
            elapsedSeconds,
        )
        state.low.refill(
            bytesPerSecond *
                config.lowShare,
            elapsedSeconds,
        )
    }

    private fun pruneIdle(
        nowNanos: Long,
    ) {
        val iterator =
            states.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (
                nowNanos -
                    entry.value.lastUpdateNanos >=
                    config.maximumIdleAgeNanos
            ) {
                iterator.remove()
            }
        }
    }
}
