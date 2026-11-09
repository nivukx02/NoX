package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertTrue

class PeerTransportQualityEngineTest {

    @Test
    fun stablePongCadenceProducesHighTransportScore() {
        val engine = PeerTransportQualityEngine()
        val endpoint = "relay-a"

        repeat(20) { index ->
            val sequence = (index + 1).toLong()
            val sent = 1_000_000_000L + index * 20_000_000L

            engine.observePingSent(
                endpoint = endpoint,
                sequence = sequence,
                sentAtNanos = sent,
            )
            engine.observePong(
                endpoint = endpoint,
                sequence = sequence,
                receivedAtNanos = sent + 2_000_000L,
            )
        }

        val quality =
            engine.snapshot(endpoint)

        assertTrue(quality.rttP95Millis < 3.0)
        assertTrue(quality.packetLossRatio < 0.01)
        assertTrue(quality.transportScore > 0.90)
    }

    @Test
    fun timeoutsBecomeMeasuredPacketLoss() {
        val engine = PeerTransportQualityEngine()
        val endpoint = "relay-b"

        repeat(8) { index ->
            val sequence = (index + 1).toLong()
            val sent =
                2_000_000_000L +
                    index * 30_000_000L

            engine.observePingSent(
                endpoint,
                sequence,
                sent,
            )

            if (index % 2 == 0) {
                engine.observePingTimeout(
                    endpoint,
                    sequence,
                    sent + 300_000_000L,
                )
            } else {
                engine.observePong(
                    endpoint,
                    sequence,
                    sent + 4_000_000L,
                )
            }
        }

        val quality =
            engine.snapshot(endpoint)

        assertTrue(quality.packetLossRatio >= 0.49)
        assertTrue(quality.transportScore < 0.75)
    }

    @Test
    fun outstandingProbeBudgetIsBounded() {
        val engine =
            PeerTransportQualityEngine(
                PeerTransportQualityEngine.Config(
                    maximumOutstandingProbesPerPeer = 4,
                ),
            )
        val endpoint = "relay-c"

        repeat(20) { index ->
            engine.observePingSent(
                endpoint = endpoint,
                sequence = index.toLong(),
                sentAtNanos =
                    3_000_000_000L +
                        index * 1_000_000L,
            )
        }

        assertTrue(
            engine.outstandingProbeCount(endpoint) <= 4,
        )
    }
}
