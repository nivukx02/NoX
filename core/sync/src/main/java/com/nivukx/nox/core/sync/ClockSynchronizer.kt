package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Robust monotonic clock estimator.
 *
 * Offset is estimated from multiple NTP-style exchanges and smoothed using RTT-weighted samples.
 * A linear drift term is also estimated from the recent window, which matters on long takes where
 * two phone oscillators can slowly walk apart.
 */
class ClockSynchronizer(
    private val alpha: Double = 0.2,
    private val windowSize: Int = 12,
) {
    data class Sample(
        val t1: Long,
        val t2: Long,
        val t3: Long,
        val t4: Long,
    ) {
        val roundTripNanos: Long get() = ((t4 - t1) - (t3 - t2)).coerceAtLeast(0L)
        val offsetNanos: Long
            get() {
                val first = t2 - t1
                val second = t3 - t4
                return averageSafely(
                    first,
                    second,
                )
            }

        val localMidpointNanos: Long
            get() =
                t1 +
                    (t4 - t1) / 2L
    }

    data class Estimate(
        val offsetNanos: Long = 0L,
        val roundTripNanos: Long = Long.MAX_VALUE,
        val jitterNanos: Long = Long.MAX_VALUE,
        val confidence: Double = 0.0,
        val driftPpm: Double = 0.0,
    )

    private data class State(
        val samples: ArrayDeque<Sample> = ArrayDeque(),
        var filteredOffset: Double = 0.0,
        var initialized: Boolean = false,
    )

    private val states = HashMap<DeviceId, State>()

    @Synchronized
    fun observe(deviceId: DeviceId, sample: Sample): Estimate {
        require(sample.t4 >= sample.t1)
        require(sample.t3 >= sample.t2)

        val state = states.getOrPut(deviceId) { State() }
        if (state.samples.size >= windowSize) state.samples.removeFirst()
        state.samples.addLast(sample)

        val medianOffset = weightedMedianOffset(state.samples)
        if (!state.initialized) {
            state.filteredOffset = medianOffset
            state.initialized = true
        } else {
            state.filteredOffset += alpha * (medianOffset - state.filteredOffset)
        }
        return estimate(deviceId)
    }

    @Synchronized
    fun estimate(deviceId: DeviceId): Estimate {
        val state = states[deviceId] ?: return Estimate()
        if (state.samples.isEmpty()) return Estimate()

        val latest = state.samples.last()
        val rtts = state.samples.map { it.roundTripNanos.toDouble() }
        val jitter =
            robustJitterNanos(
                state.samples.map {
                    it.offsetNanos.toDouble()
                },
            )
        val rtt =
            weightedRttNanos(
                state.samples,
            )
        val driftPpm =
            robustDriftPpm(
                state.samples,
            )

        val confidence = (
            1.0 /
                (
                    1.0 +
                        rtt / 1_000_000.0 +
                        jitter / 1_000_000.0
                    ) *
                (
                    1.0 -
                        (
                            abs(driftPpm) /
                                2500.0
                        ).coerceIn(0.0, 0.8)
                    ) *
                (
                    1.0 -
                        (
                            jitter /
                                4_000_000.0
                        ).coerceIn(0.0, 0.65) *
                        0.22
                    )
            ).coerceIn(0.0, 1.0)

        return Estimate(
            offsetNanos = state.filteredOffset.toLong(),
            roundTripNanos = rtt.toLong(),
            jitterNanos = jitter.toLong(),
            confidence = confidence,
            driftPpm = driftPpm,
        )
    }

    @Synchronized
    fun convertMasterToPeer(deviceId: DeviceId, masterTimeNanos: Long): Long {
        val state = states[deviceId] ?: return masterTimeNanos
        val estimate = estimate(deviceId)
        val anchor = state.samples.firstOrNull()?.localMidpointNanos ?: masterTimeNanos
        val dt = masterTimeNanos - anchor
        val driftCorrection = dt * estimate.driftPpm / 1_000_000.0
        return (masterTimeNanos + estimate.offsetNanos + driftCorrection).toLong()
    }

    @Synchronized
    fun convertPeerToMaster(deviceId: DeviceId, peerTimeNanos: Long): Long {
        val state = states[deviceId] ?: return peerTimeNanos
        val estimate = estimate(deviceId)
        val anchorPeer = (state.samples.firstOrNull()?.localMidpointNanos ?: peerTimeNanos) +
            estimate.offsetNanos
        val dt = peerTimeNanos - anchorPeer
        val driftCorrection = dt * estimate.driftPpm / 1_000_000.0
        return (peerTimeNanos - estimate.offsetNanos - driftCorrection).toLong()
    }

    @Synchronized
    fun remove(deviceId: DeviceId) { states.remove(deviceId) }

    @Synchronized
    fun clear() { states.clear() }

    @Synchronized
    fun diagnostics(): String =
        states.entries.joinToString("\n") { (device, _) ->
            val e = estimate(device)
            device.value + ": offset=" + e.offsetNanos +
                "ns rtt=" + e.roundTripNanos +
                "ns jitter=" + e.jitterNanos +
                "ns drift=" + "%.3f".format(e.driftPpm) +
                "ppm confidence=" + "%.3f".format(e.confidence)
        }

    private fun averageSafely(
        a: Long,
        b: Long,
    ): Long =
        (a and b) +
            ((a xor b) shr 1)
    private fun weightedMedianOffset(samples: Collection<Sample>): Double {
        val ordered = samples.sortedBy { it.offsetNanos }
        val total = ordered.sumOf { 1.0 / (1.0 + it.roundTripNanos.toDouble()) }
        var cumulative = 0.0
        for (sample in ordered) {
            cumulative += 1.0 / (1.0 + sample.roundTripNanos.toDouble())
            if (cumulative >= total * 0.5) return sample.offsetNanos.toDouble()
        }
        return ordered.last().offsetNanos.toDouble()
    }

    private fun robustDriftPpm(
        samples: Collection<Sample>,
    ): Double {
        if (samples.size < 4) return 0.0

        val ordered =
            samples.sortedBy {
                it.localMidpointNanos
            }

        val slopes =
            ArrayList<Double>()

        for (i in ordered.indices) {
            for (j in i + 1 until ordered.size) {
                val dt =
                    ordered[j].localMidpointNanos -
                        ordered[i].localMidpointNanos

                if (dt <= 0L) continue

                val doffset =
                    ordered[j].offsetNanos -
                        ordered[i].offsetNanos

                val slope =
                    doffset.toDouble() /
                        dt.toDouble() *
                        1_000_000.0

                if (slope.isFinite()) {
                    slopes += slope
                }
            }
        }

        if (slopes.isEmpty()) return 0.0

        val median =
            median(slopes)

        return median.coerceIn(
            -10_000.0,
            10_000.0,
        )
    }

    private fun robustJitterNanos(
        offsets: List<Double>,
    ): Double {
        if (offsets.size < 2) return 0.0

        val center =
            median(offsets)

        val deviations =
            offsets.map {
                abs(it - center)
            }

        val mad =
            median(deviations)

        return (
            mad * 1.4826
        ).coerceAtLeast(0.0)
    }

    private fun weightedRttNanos(
        samples: Collection<Sample>,
    ): Double {
        if (samples.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        var numerator = 0.0
        var denominator = 0.0

        samples.forEach { sample ->
            val rtt =
                sample.roundTripNanos
                    .coerceAtLeast(0L)
                    .toDouble()
            val weight =
                1.0 /
                    (
                        1.0 +
                            rtt /
                                1_000_000.0
                        )

            numerator += rtt * weight
            denominator += weight
        }

        return if (denominator > 0.0) {
            numerator / denominator
        } else {
            Double.POSITIVE_INFINITY
        }
    }

    private fun median(
        values: List<Double>,
    ): Double {
        if (values.isEmpty()) return 0.0

        val sorted =
            values.sorted()

        val middle =
            sorted.size / 2

        return if (
            sorted.size % 2 == 0
        ) {
            (
                sorted[middle - 1] +
                    sorted[middle]
                ) * 0.5
        } else {
            sorted[middle]
        }
    }
}
