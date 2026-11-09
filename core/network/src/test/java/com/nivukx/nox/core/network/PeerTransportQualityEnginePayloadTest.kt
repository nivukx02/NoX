package com.nivukx.nox.core.network

import kotlin.test.Test
import kotlin.test.assertTrue

class PeerTransportQualityEnginePayloadTest {

    @Test
    fun successfulPayloadTransferImprovesMeasuredThroughputConfidence() {
        val engine =
            PeerTransportQualityEngine(
                PeerTransportQualityEngine.Config(
                    minimumPayloadThroughputMbps = 5.0,
                    targetPayloadThroughputMbps = 80.0,
                    maximumPayloadThroughputMbps = 500.0,
                ),
            )

        engine.observePayloadStarted(
            endpoint = "cam-a",
            payloadId = 77L,
            totalBytes = 8_000_000L,
            startedAtNanos = 1_000_000_000L,
        )
        engine.observePayloadProgress(
            endpoint = "cam-a",
            payloadId = 77L,
            bytesTransferred = 8_000_000L,
            totalBytes = 8_000_000L,
            status = 1,
            observedAtNanos = 1_100_000_000L,
        )

        val quality =
            engine.snapshot("cam-a")

        assertTrue(
            quality.payloadThroughputMbps > 50.0,
        )
        assertTrue(
            quality.transportScore > 0.70,
        )
    }

    @Test
    fun failedLargePayloadLowersTransportScore() {
        val engine =
            PeerTransportQualityEngine()

        engine.observePayloadStarted(
            endpoint = "cam-b",
            payloadId = 91L,
            totalBytes = 12_000_000L,
            startedAtNanos = 2_000_000_000L,
        )
        engine.observePayloadProgress(
            endpoint = "cam-b",
            payloadId = 91L,
            bytesTransferred = 1_000_000L,
            totalBytes = 12_000_000L,
            status = 2,
            observedAtNanos = 2_500_000_000L,
        )

        val quality =
            engine.snapshot("cam-b")

        assertTrue(
            quality.payloadFailureRatio > 0.0,
        )
        assertTrue(
            quality.transportScore < 0.75,
        )
    }
}
