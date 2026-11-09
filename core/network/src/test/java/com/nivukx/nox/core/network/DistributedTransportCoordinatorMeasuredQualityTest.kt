package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertTrue

class DistributedTransportCoordinatorMeasuredQualityTest {

    @Test
    fun measuredPayloadQualityOverridesOptimisticStaticBandwidth() {
        val coordinator =
            DistributedTransportCoordinator()

        val staticNode =
            DistributedTransportCoordinator.Node(
                deviceId = DeviceId("cam-a"),
                shardId = 0,
                relayCapable = true,
                computeHeadroom = 0.95,
                bandwidthHeadroom = 0.95,
                thermalHeadroom = 0.95,
                timingAuthority = 0.95,
                packetLoss = 0.01,
                queueDepth = 1,
            )

        val poorMeasuredQuality =
            PeerTransportQualityEngine.Quality(
                endpoint = "cam-a",
                rttMedianMillis = 12.0,
                rttP95Millis = 22.0,
                jitterRmsMillis = 2.0,
                packetLossRatio = 0.30,
                successfulProbes = 10L,
                timedOutProbes = 4L,
                outstandingProbes = 0,
                transportScore = 0.22,
                payloadThroughputMbps = 3.0,
                payloadFailureRatio = 0.25,
                successfulPayloads = 3L,
                failedPayloads = 1L,
            )

        val enriched =
            coordinator
                .enrichNodesWithTransportQuality(
                    nodes = listOf(staticNode),
                    qualities =
                        mapOf(
                            DeviceId("cam-a") to
                                poorMeasuredQuality,
                        ),
                )
                .single()

        assertTrue(
            enriched.health < staticNode.health,
        )
        assertTrue(
            enriched.measuredTransportScore == 0.22,
        )
        assertTrue(
            enriched.packetLoss >= 0.30,
        )
    }
}
