package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

/**
 * Hierarchical transport planner for large distributed capture fabrics.
 *
 * The capture graph and the network graph are deliberately separate. A camera may remain a pure
 * sensor while another device acts as a shard relay/aggregator. The planner chooses bounded paths
 * using measured latency, throughput, packet loss, queue depth, compute headroom, thermal headroom,
 * and timing authority.
 *
 * Design goals:
 * - support 128+ distributed capture nodes without requiring a central star topology;
 * - keep core evidence local whenever it is more informative than the available link can safely carry;
 * - bound path length, relay fan-out, route memory, and candidate expansion;
 * - make reconstruction-critical evidence fail closed rather than silently degrade;
 * - prefer deterministic routes for reproducible session behavior;
 * - allow lower-value evidence to be summarized or deferred under pressure.
 *
 * This class plans transport. It does not own sockets or perform I/O.
 */
class DistributedTransportCoordinator(
    private val config: Config = Config(),
) {

    enum class Action {
        SEND,
        SUMMARIZE,
        DEFER,
        DROP,
    }

    data class Config(
        val maximumNodes: Int = 256,
        val maximumLinks: Int = 2048,
        val maximumPathHops: Int = 4,
        val maximumRouteCandidates: Int = 64,
        val maximumRelayFanout: Int = 8,
        val minimumLinkThroughputMbps: Double = 2.0,
        val maximumLinkPacketLoss: Double = 0.35,
        val maximumQueueDepth: Int = 64,
        val minimumTimingAuthority: Double = 0.55,
        val minimumCriticalTransportFreshness: Double = 0.55,
        val criticalInformationThreshold: Double = 0.70,
        val summarizeInformationThreshold: Double = 0.35,
        val summarizeSizeRatio: Double = 0.32,
        val minimumDeadlineSlackMillis: Double = 1.5,
        val routeLatencyWeight: Double = 0.38,
        val routeLossWeight: Double = 0.18,
        val routeThroughputWeight: Double = 0.18,
        val routeQueueWeight: Double = 0.10,
        val routeNodeWeight: Double = 0.08,
        val routeTimingWeight: Double = 0.08,
        val routeRelayLoadWeight: Double = 0.10,
        val sameShardRelayBonus: Double = 0.10,
    ) {
        init {
            require(maximumNodes in 2..1024)
            require(maximumLinks >= maximumNodes)
            require(maximumPathHops in 1..12)
            require(maximumRouteCandidates in 1..512)
            require(maximumRelayFanout in 1..64)
            require(minimumLinkThroughputMbps > 0.0)
            require(maximumLinkPacketLoss in 0.0..1.0)
            require(maximumQueueDepth >= 1)
            require(minimumTimingAuthority in 0.0..1.0)
            require(minimumCriticalTransportFreshness in 0.0..1.0)
            require(criticalInformationThreshold in 0.0..1.0)
            require(
                summarizeInformationThreshold in
                    0.0..criticalInformationThreshold,
            )
            require(summarizeSizeRatio in 0.05..1.0)
            require(minimumDeadlineSlackMillis >= 0.0)
            require(routeRelayLoadWeight in 0.0..0.50)
            require(sameShardRelayBonus in 0.0..0.50)
        }
    }

    data class Node(
        val deviceId: DeviceId,
        val shardId: Int,
        val relayCapable: Boolean,
        val computeHeadroom: Double,
        val bandwidthHeadroom: Double,
        val thermalHeadroom: Double,
        val timingAuthority: Double,
        val packetLoss: Double,
        val queueDepth: Int,
        val measuredTransportScore: Double? = null,
        val transportFreshness: Double = 1.0,
    ) {
        init {
            require(shardId >= 0)
            require(computeHeadroom in 0.0..1.0)
            require(bandwidthHeadroom in 0.0..1.0)
            require(thermalHeadroom in 0.0..1.0)
            require(timingAuthority in 0.0..1.0)
            require(packetLoss in 0.0..1.0)
            require(queueDepth >= 0)
            measuredTransportScore?.let {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(transportFreshness.isFinite() && transportFreshness in 0.0..1.0)
        }

        val effectiveBandwidthHeadroom: Double
            get() =
                measuredTransportScore
                    ?.let {
                        (
                            bandwidthHeadroom * 0.40 +
                                it * 0.60
                        ).coerceIn(0.0, 1.0)
                    }
                    ?: bandwidthHeadroom

        val health: Double
            get() =
                (
                    computeHeadroom * 0.21 +
                        effectiveBandwidthHeadroom * 0.22 +
                        thermalHeadroom * 0.14 +
                        timingAuthority * 0.27 +
                        transportFreshness * 0.05 +
                        (
                            1.0 -
                                packetLoss
                        ) * 0.12
                    ).coerceIn(0.0, 1.0)
    }

    data class Link(
        val from: DeviceId,
        val to: DeviceId,
        val latencyMillis: Double,
        val packetLoss: Double,
        val throughputMbps: Double,
        val bidirectional: Boolean,
    ) {
        init {
            require(latencyMillis >= 0.0)
            require(packetLoss in 0.0..1.0)
            require(throughputMbps > 0.0)
        }
    }

    data class Evidence(
        val evidenceId: Long = 0L,
        val bytes: Int,
        val deadlineMillis: Double,
        val informationScore: Double,
        val reconstructionCritical: Boolean = false,
        val sourceDeviceId: DeviceId,
        val sourcePriority: Double = 0.5,
        val resumable: Boolean = true,
    ) {
        init {
            require(evidenceId >= 0L)
            require(bytes >= 0)
            require(deadlineMillis >= 0.0)
            require(informationScore in 0.0..1.0)
            require(sourceDeviceId.value.isNotBlank())
            require(sourcePriority in 0.0..1.0)
        }
    }

    data class Route(
        val nodes: List<DeviceId>,
        val totalLatencyMillis: Double,
        val totalTransferMillis: Double,
        val totalCost: Double,
        val bottleneckThroughputMbps: Double,
        val aggregatePacketLoss: Double,
        val minimumNodeHealth: Double,
        val minimumTimingAuthority: Double,
        val minimumTransportFreshness: Double = 0.0,
        val deadlineSlackMillis: Double,
        val accepted: Boolean,
    ) {
        val hopCount: Int
            get() =
                (
                    nodes.size - 1
                ).coerceAtLeast(0)
    }

    data class Plan(
        val evidenceId: Long,
        val action: Action,
        val route: List<DeviceId>,
        val estimatedLatencyMillis: Double,
        val estimatedBytes: Int,
        val informationScore: Double,
        val reason: String,
        val routeHealth: Double,
        val timingAuthority: Double,
        val transportFreshness: Double = 0.0,
    ) {
        val accepted: Boolean
            get() =
                action != Action.DROP
    }

    private data class Edge(
        val from: DeviceId,
        val to: DeviceId,
        val link: Link,
    )

    private data class QueueNode(
        val node: DeviceId,
        val cost: Double,
        val path: List<DeviceId>,
        val latencyMillis: Double,
        val transferMillis: Double,
        val throughputMbps: Double,
        val packetLoss: Double,
        val nodeHealth: Double,
        val timingAuthority: Double,
        val transportFreshness: Double,
    )

    fun enrichNodesWithTransportQuality(
        nodes: List<Node>,
        qualities: Map<DeviceId, PeerTransportQualityEngine.Quality>,
    ): List<Node> =
        nodes.map { node ->
            qualities[node.deviceId]
                ?.let { quality ->
                    node.copy(
                        measuredTransportScore =
                            quality.transportScore,
                        packetLoss =
                            maxOf(
                                node.packetLoss,
                                quality.packetLossRatio,
                            ).coerceIn(0.0, 1.0),
                        transportFreshness =
                            quality.freshnessScore.coerceIn(0.0, 1.0),
                        bandwidthHeadroom =
                            node.bandwidthHeadroom
                                .coerceIn(0.0, 1.0),
                    )
                }
                ?: node
        }

    fun plan(
        nodes: List<Node>,
        links: List<Link>,
        source: DeviceId? = null,
        destination: DeviceId,
        evidence: Evidence,
        relayLoads: Map<DeviceId, Int> = emptyMap(),
    ): Plan {
        val resolvedSource =
            source ?: evidence.sourceDeviceId

        val distinctNodes =
            nodes.distinctBy { it.deviceId }

        val endpointIds =
            setOf(
                resolvedSource,
                destination,
            )

        val boundedNodes =
            (
                distinctNodes.filter {
                    it.deviceId in endpointIds
                } +
                    distinctNodes
                        .filterNot {
                            it.deviceId in endpointIds
                        }
                        .sortedWith(
                            compareByDescending<Node> {
                                it.health
                            }.thenByDescending {
                                it.timingAuthority
                            }.thenByDescending {
                                it.transportFreshness
                            }.thenByDescending {
                                it.effectiveBandwidthHeadroom
                            }.thenBy {
                                it.deviceId.value
                            },
                        )
            )
                .distinctBy { it.deviceId }
                .take(config.maximumNodes)

        val nodeMap =
            boundedNodes.associateBy { it.deviceId }

        val sourceNode = nodeMap[resolvedSource]
        val destinationNode =
            nodeMap[destination]

        if (
            sourceNode == null ||
                destinationNode == null
        ) {
            return Plan(
                evidenceId = evidence.evidenceId,
                action =
                    if (
                        evidence.reconstructionCritical
                    ) {
                        Action.DROP
                    } else {
                        Action.DEFER
                    },
                route = emptyList(),
                estimatedLatencyMillis =
                    Double.POSITIVE_INFINITY,
                estimatedBytes = evidence.bytes,
                informationScore = evidence.informationScore,
                reason = "source-or-destination-unavailable",
                routeHealth = 0.0,
                timingAuthority = 0.0,
            )
        }

        if (resolvedSource == destination) {
            return Plan(
                evidenceId = evidence.evidenceId,
                action = Action.SEND,
                route = listOf(resolvedSource),
                estimatedLatencyMillis = 0.0,
                estimatedBytes = evidence.bytes,
                informationScore = evidence.informationScore,
                reason = "source-is-destination",
                routeHealth =
                    min(
                        sourceNode.health,
                        destinationNode.health,
                    ),
                timingAuthority =
                    min(
                        sourceNode.timingAuthority,
                        destinationNode.timingAuthority,
                    ),
            )
        }

        val route =
            findBestRoute(
                nodes = nodeMap,
                links = links
                    .distinctBy {
                        it.from to it.to
                    }
                    .sortedWith(
                        compareByDescending<Link> {
                            it.from == resolvedSource ||
                                it.to == resolvedSource ||
                                it.from == destination ||
                                it.to == destination
                        }.thenBy {
                            it.latencyMillis
                        }.thenByDescending {
                            it.throughputMbps
                        }.thenBy {
                            it.packetLoss
                        }.thenBy {
                            it.from.value
                        }.thenBy {
                            it.to.value
                        },
                    )
                    .take(config.maximumLinks),
                source = resolvedSource,
                destination = destination,
                bytes = evidence.bytes,
                deadlineMillis = evidence.deadlineMillis,
                relayLoads = relayLoads,
            )

        if (route == null) {
            return fallbackPlan(
                evidence = evidence,
                reason = "no-admissible-route",
            )
        }

        val slack =
            route.deadlineSlackMillis

        val deadlineSafe =
            slack >=
                config.minimumDeadlineSlackMillis

        val criticalTimingSafe =
            route.minimumTimingAuthority >=
                config.minimumTimingAuthority

        val healthSafe =
            route.minimumNodeHealth >=
                0.45

        val evidenceFreshnessSafe =
            route.minimumTransportFreshness >=
                config.minimumCriticalTransportFreshness

        if (
            evidence.reconstructionCritical &&
                !deadlineSafe
        ) {
            return fallbackPlan(
                evidence = evidence,
                reason =
                    "critical-evidence-route-misses-deadline",
            )
        }

        if (
            evidence.reconstructionCritical &&
                (
                    !criticalTimingSafe ||
                        !healthSafe ||
                        !evidenceFreshnessSafe
                    )
        ) {
            return fallbackPlan(
                evidence = evidence,
                reason =
                    "critical-evidence-route-health-or-timing-low",
            )
        }

        return when {
            deadlineSafe &&
                healthSafe &&
                criticalTimingSafe &&
                evidenceFreshnessSafe ->
                Plan(
                    evidenceId = evidence.evidenceId,
                    action = Action.SEND,
                    route = route.nodes.drop(1),
                    estimatedLatencyMillis =
                        route.totalLatencyMillis +
                            route.totalTransferMillis,
                    estimatedBytes = evidence.bytes,
                    informationScore =
                        evidence.informationScore,
                    reason = "healthy-hierarchical-route",
                    routeHealth =
                        route.minimumNodeHealth,
                    timingAuthority =
                        route.minimumTimingAuthority,
                    transportFreshness =
                        route.minimumTransportFreshness,
                )

            !evidence.reconstructionCritical &&
                evidence.informationScore <=
                    config.summarizeInformationThreshold &&
                evidence.resumable &&
                route.totalTransferMillis >
                    max(
                        0.0,
                        evidence.deadlineMillis * 0.40,
                    ) ->
                Plan(
                    evidenceId = evidence.evidenceId,
                    action = Action.SUMMARIZE,
                    route = route.nodes.drop(1),
                    estimatedLatencyMillis =
                        route.totalLatencyMillis +
                            route.totalTransferMillis *
                                config.summarizeSizeRatio,
                    estimatedBytes =
                        (
                            evidence.bytes *
                                config.summarizeSizeRatio
                        ).toInt()
                            .coerceAtLeast(1),
                    informationScore =
                        evidence.informationScore,
                    reason =
                        "low-information-evidence-summarized",
                    routeHealth =
                        route.minimumNodeHealth,
                    timingAuthority =
                        route.minimumTimingAuthority,
                )

            !evidence.reconstructionCritical &&
                evidence.resumable &&
                !deadlineSafe &&
                route.deadlineSlackMillis > -10.0 ->
                Plan(
                    evidenceId = evidence.evidenceId,
                    action = Action.DEFER,
                    route = route.nodes.drop(1),
                    estimatedLatencyMillis =
                        route.totalLatencyMillis,
                    estimatedBytes = evidence.bytes,
                    informationScore =
                        evidence.informationScore,
                    reason = "deferred-for-route-headroom",
                    routeHealth =
                        route.minimumNodeHealth,
                    timingAuthority =
                        route.minimumTimingAuthority,
                )

            evidence.reconstructionCritical ->
                fallbackPlan(
                    evidence = evidence,
                    reason = "critical-route-quality-insufficient",
                )

            else ->
                Plan(
                    evidenceId = evidence.evidenceId,
                    action = Action.DROP,
                    route = route.nodes.drop(1),
                    estimatedLatencyMillis =
                        route.totalLatencyMillis +
                            route.totalTransferMillis,
                    estimatedBytes = evidence.bytes,
                    informationScore =
                        evidence.informationScore,
                    reason = "noncritical-evidence-route-insufficient",
                    routeHealth =
                        route.minimumNodeHealth,
                    timingAuthority =
                        route.minimumTimingAuthority,
                )
        }
    }

    fun planAll(
        nodes: List<Node>,
        links: List<Link>,
        destination: DeviceId,
        evidences: List<Evidence>,
    ): List<Plan> {
        val bounded =
            evidences
                .sortedWith(
                    compareByDescending<Evidence> {
                        if (it.reconstructionCritical) 1 else 0
                    }.thenByDescending {
                        it.informationScore
                    }.thenBy {
                        it.deadlineMillis
                    }.thenBy {
                        it.evidenceId
                    },
                )

        val sentByRelay =
            HashMap<DeviceId, Int>()

        return bounded.map { evidence ->
            val plan =
                plan(
                    nodes = nodes,
                    links = links,
                    source =
                        evidence.sourceDeviceId,
                    destination = destination,
                    evidence = evidence,
                    relayLoads = sentByRelay,
                )

            val relayHops =
                plan.route.dropLast(1)

            val saturatedRelay =
                relayHops.firstOrNull {
                    (sentByRelay[it] ?: 0) >=
                        config.maximumRelayFanout
                }

            if (
                saturatedRelay != null &&
                plan.action == Action.SEND
            ) {
                plan.copy(
                    action =
                        if (
                            evidence.reconstructionCritical
                        ) {
                            Action.DEFER
                        } else {
                            Action.DROP
                        },
                    reason =
                        "relay-fanout-limit-" +
                            saturatedRelay.value,
                )
            } else {
                relayHops.forEach { relay ->
                    sentByRelay[relay] =
                        (
                            sentByRelay[relay] ?: 0
                        ) + 1
                }
                plan
            }
        }
    }

    private fun nodeHealthScore(
        node: Node,
    ): Double {
        val queuePenalty =
            (
                node.queueDepth.toDouble() /
                    config.maximumQueueDepth
            ).coerceIn(0.0, 1.0) * 0.06

        val freshnessPenalty =
            (
                1.0 -
                    node.transportFreshness
            ).coerceIn(0.0, 1.0) * 0.12

        return (
            node.health -
                queuePenalty -
                freshnessPenalty
        ).coerceIn(0.0, 1.0)
    }

    private fun fallbackPlan(
        evidence: Evidence,
        reason: String,
    ): Plan =
        Plan(
            evidenceId = evidence.evidenceId,
            action =
                if (
                    evidence.reconstructionCritical
                ) {
                    Action.DROP
                } else if (
                    evidence.informationScore >=
                        config.criticalInformationThreshold
                ) {
                    Action.DEFER
                } else {
                    Action.SUMMARIZE
                },
            route = emptyList(),
            estimatedLatencyMillis =
                Double.POSITIVE_INFINITY,
            estimatedBytes = evidence.bytes,
            informationScore =
                evidence.informationScore,
            reason = reason,
            routeHealth = 0.0,
            timingAuthority = 0.0,
            transportFreshness = 0.0,
        )

    private fun findBestRoute(
        nodes: Map<DeviceId, Node>,
        links: List<Link>,
        source: DeviceId,
        destination: DeviceId,
        bytes: Int,
        deadlineMillis: Double,
        relayLoads: Map<DeviceId, Int> = emptyMap(),
    ): Route? {
        val adjacency =
            HashMap<DeviceId, MutableList<Edge>>()

        links.forEach { link ->
            val toNode =
                nodes[link.to]
                    ?: return@forEach

            if (
                link.throughputMbps <
                    config.minimumLinkThroughputMbps ||
                link.packetLoss >
                    config.maximumLinkPacketLoss ||
                toNode.queueDepth >
                    config.maximumQueueDepth
            ) {
                return@forEach
            }

            adjacency
                .getOrPut(link.from) {
                    ArrayList()
                }
                .add(
                    Edge(
                        from = link.from,
                        to = link.to,
                        link = link,
                    ),
                )

            if (link.bidirectional) {
                adjacency
                    .getOrPut(link.to) {
                        ArrayList()
                    }
                    .add(
                        Edge(
                            from = link.to,
                            to = link.from,
                            link =
                                link.copy(
                                    from = link.to,
                                    to = link.from,
                                ),
                        ),
                    )
            }
        }

        val queue =
            PriorityQueue(
                compareBy<QueueNode> {
                    it.cost
                }.thenBy {
                    it.path.joinToString("/")
                },
            )

        queue.add(
            QueueNode(
                node = source,
                cost = 0.0,
                path = listOf(source),
                latencyMillis = 0.0,
                transferMillis = 0.0,
                throughputMbps =
                    Double.POSITIVE_INFINITY,
                packetLoss = 0.0,
                nodeHealth =
                    nodes[source]?.let(::nodeHealthScore) ?: 0.0,
                timingAuthority =
                    nodes[source]
                        ?.timingAuthority
                        ?: 0.0,
                transportFreshness =
                    nodes[source]
                        ?.transportFreshness
                        ?: 0.0,
            ),
        )

        val bestCost =
            HashMap<DeviceId, Double>()
        var expansions = 0

        while (
            queue.isNotEmpty() &&
                expansions <
                config.maximumRouteCandidates
        ) {
            val current =
                queue.poll()
            expansions += 1

            if (
                current.node ==
                    destination
            ) {
                val totalTransport =
                    current.transferMillis
                val totalLatency =
                    current.latencyMillis
                val slack =
                    deadlineMillis -
                        totalTransport -
                        totalLatency

                if (
                    current.path.size - 1 >
                        config.maximumPathHops
                ) {
                    continue
                }

                return Route(
                    nodes = current.path,
                    totalLatencyMillis = totalLatency,
                    totalTransferMillis =
                        totalTransport,
                    totalCost = current.cost,
                    bottleneckThroughputMbps =
                        if (
                            current.throughputMbps
                                .isFinite()
                        ) {
                            current.throughputMbps
                        } else {
                            0.0
                        },
                    aggregatePacketLoss =
                        current.packetLoss,
                    minimumNodeHealth =
                        current.nodeHealth,
                    minimumTimingAuthority =
                        current.timingAuthority,
                    minimumTransportFreshness =
                        current.transportFreshness,
                    deadlineSlackMillis =
                        slack,
                    accepted =
                        slack >=
                            -config.minimumDeadlineSlackMillis,
                )
            }

            val knownBest =
                bestCost[current.node]
            if (
                knownBest != null &&
                    current.cost >
                    knownBest + 1e-9
            ) {
                continue
            }
            bestCost[current.node] =
                current.cost

            if (
                current.path.size - 1 >=
                    config.maximumPathHops
            ) {
                continue
            }

            adjacency[current.node]
                .orEmpty()
                .sortedWith(
                    compareBy<Edge> {
                        it.to.value
                    },
                )
                .forEach { edge ->
                    if (
                        current.path.contains(
                            edge.to,
                        )
                    ) {
                        return@forEach
                    }

                    val destinationNode =
                        nodes[edge.to]
                            ?: return@forEach

                    if (
                        destinationNode.queueDepth >
                            config.maximumQueueDepth
                    ) {
                        return@forEach
                    }

                    if (
                        edge.to != destination &&
                        !destinationNode.relayCapable
                    ) {
                        return@forEach
                    }

                    val transferMillis =
                        if (bytes <= 0) {
                            0.0
                        } else {
                            (
                                bytes.toDouble() * 8.0 /
                                    (
                                        edge.link.throughputMbps *
                                            1_000_000.0
                                    )
                            ) * 1000.0
                        }

                    val queuePenalty =
                        destinationNode.queueDepth.toDouble() /
                            config.maximumQueueDepth
                                .toDouble()

                    val throughputScore =
                        (
                            1.0 -
                                config.minimumLinkThroughputMbps /
                                    max(
                                        edge.link.throughputMbps,
                                        config.minimumLinkThroughputMbps,
                                    )
                        ).coerceIn(0.0, 1.0)

                    val lossScore =
                        (
                            1.0 -
                                edge.link.packetLoss
                        ).coerceIn(0.0, 1.0)

                    val nodeScore =
                        nodeHealthScore(destinationNode)

                    val timingScore =
                        destinationNode.timingAuthority

                    val sourceShard =
                        nodes[source]?.shardId

                    val shardLocality =
                        if (
                            destinationNode.relayCapable &&
                                sourceShard != null &&
                                destinationNode.shardId ==
                                sourceShard
                        ) {
                            config.sameShardRelayBonus
                        } else {
                            0.0
                        }

                    val relayLoadPenalty =
                        if (
                            edge.to != destination &&
                                destinationNode.relayCapable
                        ) {
                            (
                                relayLoads[edge.to]
                                    ?.toDouble()
                                    ?: 0.0
                            ) /
                                config.maximumRelayFanout
                                    .toDouble()
                        } else {
                            0.0
                        }

                    val hopPenalty =
                        (
                            current.path.size -
                                1
                        ).toDouble() /
                            config.maximumPathHops
                                .toDouble()

                    val edgeCost =
                        edge.link.latencyMillis /
                            20.0 *
                            config.routeLatencyWeight +
                            (
                                1.0 -
                                    lossScore
                                ) *
                                config.routeLossWeight +
                            (
                                1.0 -
                                    throughputScore
                                ) *
                                config.routeThroughputWeight +
                            queuePenalty *
                                config.routeQueueWeight +
                            (
                                1.0 -
                                    nodeScore
                                ) *
                                config.routeNodeWeight +
                            (
                                1.0 -
                                    timingScore
                                ) *
                                config.routeTimingWeight +
                            relayLoadPenalty *
                                config.routeRelayLoadWeight +
                            hopPenalty * 0.04 -
                            shardLocality

                    queue.add(
                        QueueNode(
                            node = edge.to,
                            cost =
                                current.cost +
                                    edgeCost,
                            path =
                                current.path +
                                    edge.to,
                            latencyMillis =
                                current.latencyMillis +
                                    edge.link.latencyMillis,
                            transferMillis =
                                current.transferMillis +
                                    transferMillis,
                            throughputMbps =
                                min(
                                    current.throughputMbps,
                                    edge.link.throughputMbps,
                                ),
                            packetLoss =
                                1.0 -
                                    (
                                        1.0 -
                                            current.packetLoss
                                        ) *
                                        (
                                            1.0 -
                                                edge.link.packetLoss
                                        ),
                            nodeHealth =
                                min(
                                    current.nodeHealth,
                                    nodeHealthScore(
                                        destinationNode,
                                    ),
                                ),
                            timingAuthority =
                                min(
                                    current.timingAuthority,
                                    destinationNode.timingAuthority,
                                ),
                            transportFreshness =
                                min(
                                    current.transportFreshness,
                                    destinationNode.transportFreshness,
                                ),
                        ),
                    )
                }
        }

        return null
    }
}
