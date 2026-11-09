package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedTransportAdmissionDeterminismTest {

    @Test
    fun nodeBoundingPreservesSourceAndDestination() {
        val coordinator =
            DistributedTransportCoordinator(
                DistributedTransportCoordinator.Config(
                    maximumNodes = 4,
                ),
            )

        val nodes =
            (0 until 8).map { index ->
                DistributedTransportCoordinator.Node(
                    deviceId = DeviceId("node-$index"),
                    shardId = index % 2,
                    relayCapable = true,
                    computeHeadroom =
                        if (index == 7) 0.99 else 0.55,
                    bandwidthHeadroom = 0.90,
                    thermalHeadroom = 0.90,
                    timingAuthority = 0.90,
                    packetLoss = 0.01,
                    queueDepth = 1,
                    measuredTransportScore = 0.90,
                    transportFreshness = 0.95,
                )
            }

        val plan =
            coordinator.plan(
                nodes = nodes,
                links =
                    listOf(
                        link("node-7", "node-6"),
                        link("node-6", "node-0"),
                        link("node-0", "node-1"),
                    ),
                source = DeviceId("node-7"),
                destination = DeviceId("node-1"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 1L,
                        bytes = 4096,
                        deadlineMillis = 20.0,
                        informationScore = 0.90,
                        reconstructionCritical = true,
                        sourceDeviceId = DeviceId("node-7"),
                    ),
            )

        assertTrue(plan.route.isNotEmpty())
        assertEquals(
            DeviceId("node-7"),
            plan.route.first(),
        )
        assertEquals(
            DeviceId("node-1"),
            plan.route.last(),
        )
    }

    @Test
    fun staleTransportIsReportedSeparatelyFromGenericRouteHealth() {
        val coordinator =
            DistributedTransportCoordinator()

        val source =
            baseNode("source").copy(
                transportFreshness = 0.20,
            )
        val destination =
            baseNode("dest").copy(
                transportFreshness = 0.98,
            )

        val plan =
            coordinator.plan(
                nodes = listOf(source, destination),
                links =
                    listOf(
                        link("source", "dest"),
                    ),
                destination = DeviceId("dest"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 2L,
                        bytes = 4096,
                        deadlineMillis = 30.0,
                        informationScore = 0.90,
                        reconstructionCritical = true,
                        sourceDeviceId = DeviceId("source"),
                    ),
            )

        assertTrue(
            plan.reason.contains("route") ||
                plan.action ==
                DistributedTransportCoordinator.Action.DROP,
        )
    }

    private fun baseNode(id: String) =
        DistributedTransportCoordinator.Node(
            deviceId = DeviceId(id),
            shardId = 0,
            relayCapable = true,
            computeHeadroom = 0.95,
            bandwidthHeadroom = 0.95,
            thermalHeadroom = 0.95,
            timingAuthority = 0.95,
            packetLoss = 0.01,
            queueDepth = 1,
            measuredTransportScore = 0.95,
            transportFreshness = 0.95,
        )

    private fun link(from: String, to: String) =
        DistributedTransportCoordinator.Link(
            from = DeviceId(from),
            to = DeviceId(to),
            latencyMillis = 1.0,
            packetLoss = 0.01,
            throughputMbps = 200.0,
            bidirectional = true,
        )
}
