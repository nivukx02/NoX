package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertTrue

class PeerTransportFreshnessTest {

    @Test
    fun recentPeerEvidenceHasHighFreshness() {
        val engine =
            PeerTransportQualityEngine(
                PeerTransportQualityEngine.Config(
                    freshnessHalfLifeNanos = 500_000_000L,
                ),
            )

        engine.observePingSent(
            endpoint = "cam-a",
            sequence = 1L,
            sentAtNanos = 1_000_000_000L,
        )
        engine.observePong(
            endpoint = "cam-a",
            sequence = 1L,
            receivedAtNanos = 1_004_000_000L,
        )

        val quality =
            engine.snapshot(
                endpoint = "cam-a",
                observedAtNanos = 1_050_000_000L,
            )

        assertTrue(quality.freshnessScore > 0.90)
        assertTrue(quality.lastSeenAgeMillis < 0.10)
    }

    @Test
    fun stalePeerEvidenceLosesFreshnessWithoutDeletingHistory() {
        val engine =
            PeerTransportQualityEngine(
                PeerTransportQualityEngine.Config(
                    freshnessHalfLifeNanos = 100_000_000L,
                ),
            )

        engine.observePingSent(
            endpoint = "cam-b",
            sequence = 2L,
            sentAtNanos = 1_000_000_000L,
        )
        engine.observePong(
            endpoint = "cam-b",
            sequence = 2L,
            receivedAtNanos = 1_005_000_000L,
        )

        val quality =
            engine.snapshot(
                endpoint = "cam-b",
                observedAtNanos = 1_505_000_000L,
            )

        assertTrue(quality.lastSeenAgeMillis > 400.0)
        assertTrue(quality.freshnessScore < 0.02)
        assertTrue(quality.successfulProbes == 1L)
    }
}
