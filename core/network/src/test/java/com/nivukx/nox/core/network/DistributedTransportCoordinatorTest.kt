package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DistributedTransportCoordinatorTest {

    @Test
    fun selectsHealthyLowLatencyShardRelay() {
        val coordinator = DistributedTransportCoordinator()

        val result =
            coordinator.plan(
                nodes =
                    listOf(
                        node("cam-a", shard = 1, bandwidthMbps = 80.0),
                        node("relay-a", shard = 1, bandwidthMbps = 400.0, relay = true),
                        node("relay-b", shard = 2, bandwidthMbps = 600.0, relay = true),
                        node("master", shard = 0, bandwidthMbps = 900.0, relay = true),
                    ),
                links =
                    listOf(
                        link("cam-a", "relay-a", 2.0, 0.01, 100.0),
                        link("relay-a", "master", 3.0, 0.01, 100.0),
                        link("cam-a", "relay-b", 12.0, 0.03, 70.0),
                        link("relay-b", "master", 2.0, 0.01, 500.0),
                    ),
                destination = DeviceId("master"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 1L,
                        bytes = 40_000,
                        sourceDeviceId = DeviceId("cam-a"),
                        deadlineMillis = 25.0,
                        informationScore = 0.92,
                        reconstructionCritical = true,
                    ),
            )

        assertTrue(result.accepted)
        assertEquals(
            listOf(
                DeviceId("relay-a"),
                DeviceId("master"),
            ),
            result.route,
        )
    }

    @Test
    fun criticalEvidenceRejectsAPathThatMissesDeadline() {
        val coordinator = DistributedTransportCoordinator()

        val result =
            coordinator.plan(
                nodes =
                    listOf(
                        node("cam-a", shard = 1, bandwidthMbps = 10.0),
                        node("relay", shard = 1, bandwidthMbps = 50.0, relay = true),
                        node("master", shard = 0, bandwidthMbps = 500.0, relay = true),
                    ),
                links =
                    listOf(
                        link("cam-a", "relay", 20.0, 0.20, 2.0),
                        link("relay", "master", 20.0, 0.20, 2.0),
                    ),
                destination = DeviceId("master"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 2L,
                        bytes = 120_000,
                        sourceDeviceId = DeviceId("cam-a"),
                        deadlineMillis = 15.0,
                        informationScore = 0.99,
                        reconstructionCritical = true,
                    ),
            )

        assertTrue(!result.accepted)
        assertTrue(
            result.reason.contains("deadline"),
        )
    }

    @Test
    fun lowInformationEvidenceCanBeDeferredUnderBandwidthPressure() {
        val coordinator = DistributedTransportCoordinator()

        val result =
            coordinator.plan(
                nodes =
                    listOf(
                        node("cam-a", shard = 1, bandwidthMbps = 20.0),
                        node("master", shard = 0, bandwidthMbps = 40.0, relay = true),
                    ),
                links =
                    listOf(
                        link("cam-a", "master", 15.0, 0.10, 8.0),
                    ),
                destination = DeviceId("master"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 3L,
                        bytes = 90_000,
                        sourceDeviceId = DeviceId("cam-a"),
                        deadlineMillis = 20.0,
                        informationScore = 0.15,
                        reconstructionCritical = false,
                    ),
            )

        assertTrue(result.accepted)
        assertTrue(result.action != DistributedTransportCoordinator.Action.DROP)
        assertTrue(
            result.action ==
                DistributedTransportCoordinator.Action.SUMMARIZE ||
                result.action ==
                DistributedTransportCoordinator.Action.DEFER,
        )
    }

    @Test
    fun relayFanoutIsBounded() {
        val coordinator =
            DistributedTransportCoordinator(
                DistributedTransportCoordinator.Config(
                    maximumRelayFanout = 4,
                ),
            )

        val nodes =
            (0 until 8).map { index ->
                node(
                    id = "cam-$index",
                    shard = index / 2,
                    bandwidthMbps = 100.0,
                )
            } +
                node(
                    id = "master",
                    shard = 0,
                    bandwidthMbps = 900.0,
                    relay = true,
                )

        val links =
            (0 until 8).map { index ->
                link(
                    "cam-$index",
                    "master",
                    4.0,
                    0.02,
                    100.0,
                )
            }

        val results =
            coordinator.planAll(
                nodes = nodes,
                links = links,
                destination = DeviceId("master"),
                evidences =
                    List(8) { index ->
                        DistributedTransportCoordinator.Evidence(
                            evidenceId = index.toLong(),
                            bytes = 20_000,
                            deadlineMillis = 40.0,
                            informationScore = 0.90,
                            reconstructionCritical = true,
                        )
                    },
            )

        assertTrue(
            results.count {
                it.action == DistributedTransportCoordinator.Action.SEND
            } <= 4,
        )
    }

    private fun node(
        id: String,
        shard: Int,
        bandwidthMbps: Double,
        relay: Boolean = false,
    ) =
        DistributedTransportCoordinator.Node(
            deviceId = DeviceId(id),
            shardId = shard,
            relayCapable = relay,
            computeHeadroom = 0.90,
            bandwidthHeadroom = bandwidthMbps / 1_000.0,
            thermalHeadroom = 0.90,
            timingAuthority = 0.95,
            packetLoss = 0.01,
            queueDepth = 2,
        )

    private fun link(
        from: String,
        to: String,
        latencyMillis: Double,
        packetLoss: Double,
        throughputMbps: Double,
    ) =
        DistributedTransportCoordinator.Link(
            from = DeviceId(from),
            to = DeviceId(to),
            latencyMillis = latencyMillis,
            packetLoss = packetLoss,
            throughputMbps = throughputMbps,
            bidirectional = true,
        )
    @Test
    fun sameShardRelayIsPreferredWhenQualityIsOtherwiseComparable() {
        val coordinator =
            DistributedTransportCoordinator(
                DistributedTransportCoordinator.Config(
                    sameShardRelayBonus = 0.18,
                ),
            )

        val result =
            coordinator.plan(
                nodes =
                    listOf(
                        node("cam-a", shard = 3, bandwidthMbps = 300.0),
                        node("relay-local", shard = 3, bandwidthMbps = 300.0, relay = true),
                        node("relay-remote", shard = 4, bandwidthMbps = 300.0, relay = true),
                        node("master", shard = 0, bandwidthMbps = 600.0, relay = true),
                    ),
                links =
                    listOf(
                        link("cam-a", "relay-local", 4.0, 0.01, 300.0),
                        link("relay-local", "master", 4.0, 0.01, 300.0),
                        link("cam-a", "relay-remote", 4.0, 0.01, 300.0),
                        link("relay-remote", "master", 4.0, 0.01, 300.0),
                    ),
                destination = DeviceId("master"),
                evidence =
                    DistributedTransportCoordinator.Evidence(
                        evidenceId = 15L,
                        sourceDeviceId = DeviceId("cam-a"),
                        bytes = 10_000,
                        deadlineMillis = 50.0,
                        informationScore = 0.95,
                        reconstructionCritical = true,
                    ),
            )

        assertEquals(
            listOf(
                DeviceId("relay-local"),
                DeviceId("master"),
            ),
            result.route,
        )
    }

    @Test
    fun loadAwareRoutingSpreadsHighValueEvidence() {
        val coordinator =
            DistributedTransportCoordinator(
                DistributedTransportCoordinator.Config(
                    maximumRelayFanout = 8,
                    routeRelayLoadWeight = 0.24,
                ),
            )

        val nodes =
            listOf(
                node("cam-a", 1, 300.0),
                node("cam-b", 2, 300.0),
                node("relay-a", 1, 600.0, true),
                node("relay-b", 2, 600.0, true),
                node("master", 0, 1_000.0, true),
            )

        val links =
            listOf(
                link("cam-a", "relay-a", 2.0, 0.01, 300.0),
                link("relay-a", "master", 2.0, 0.01, 300.0),
                link("cam-a", "relay-b", 2.0, 0.01, 300.0),
                link("relay-b", "master", 2.0, 0.01, 300.0),
                link("cam-b", "relay-a", 2.0, 0.01, 300.0),
                link("cam-b", "relay-b", 2.0, 0.01, 300.0),
            )

        val plans =
            coordinator.planAll(
                nodes = nodes,
                links = links,
                destination = DeviceId("master"),
                evidences =
                    listOf(
                        DistributedTransportCoordinator.Evidence(
                            evidenceId = 101L,
                            sourceDeviceId = DeviceId("cam-a"),
                            bytes = 20_000,
                            deadlineMillis = 50.0,
                            informationScore = 0.96,
                            reconstructionCritical = true,
                        ),
                        DistributedTransportCoordinator.Evidence(
                            evidenceId = 102L,
                            sourceDeviceId = DeviceId("cam-b"),
                            bytes = 20_000,
                            deadlineMillis = 50.0,
                            informationScore = 0.96,
                            reconstructionCritical = true,
                        ),
                    ),
            )

        assertEquals(
            2,
            plans.count {
                it.action ==
                    DistributedTransportCoordinator.Action.SEND
            },
        )
        assertTrue(
            plans.map { it.route.firstOrNull()?.value }
                .distinct()
                .size >= 2,
        )
    }

}
