package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals

class DistributedTransportFreshnessTest {

    @Test
    fun staleCriticalRouteDoesNotSend() {
        val engine = DistributedTransportCoordinator()

        val source = NodeFactory.node(
            id = "cam-a",
            shard = 0,
            freshness = 0.05,
        )
        val destination = NodeFactory.node(
            id = "hub",
            shard = 0,
            freshness = 0.98,
        )

        val plan =
            engine.plan(
                nodes = listOf(source, destination),
                links =
                    listOf(
                        DistributedTransportCoordinator.Link(
                            from = DeviceId("cam-a"),
                            to = DeviceId("hub"),
                            latencyMillis = 1.0,
                            packetLoss = 0.01,
                            throughputMbps = 200.0,
                            bidirectional = true,
                        ),
                    ),
                destination = DeviceId("hub"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 1L,
                        bytes = 4096,
                        deadlineMillis = 20.0,
                        informationScore = 0.95,
                        reconstructionCritical = true,
                        sourceDeviceId = DeviceId("cam-a"),
                    ),
            )

        assertEquals(
            DistributedTransportCoordinator.Action.DROP,
            plan.action,
        )
    }

    private object NodeFactory {
        fun node(
            id: String,
            shard: Int,
            freshness: Double,
        ) =
            DistributedTransportCoordinator.Node(
                deviceId = DeviceId(id),
                shardId = shard,
                relayCapable = true,
                computeHeadroom = 0.95,
                bandwidthHeadroom = 0.95,
                thermalHeadroom = 0.95,
                timingAuthority = 0.95,
                packetLoss = 0.01,
                queueDepth = 1,
                measuredTransportScore = 0.95,
                transportFreshness = freshness,
            )
    }
}
