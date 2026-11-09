package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Robust graph-level clock authority for distributed multi-camera capture.
 *
 * An edge encodes:
 *
 *     clock(to) = clock(from) + offsetNanos
 *
 * Multiple anchors are solved jointly. The highest-quality node is used only as a gauge constraint,
 * not as an unquestioned timing oracle. Redundant cycles expose inconsistent measurements, while
 * disconnected or weakly timed nodes remain non-authoritative.
 *
 * The solver is intentionally bounded to the distributed camera envelope and is suitable for
 * synchronization epochs rather than per-frame processing.
 */
class DistributedClockAuthorityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumNodes: Int = 128,
        val maximumEdges: Int = 1024,
        val maximumIterations: Int = 32,
        val convergenceToleranceNanos: Double = 250.0,
        val minimumNodeAuthority: Double = 0.80,
        val minimumEdgeConfidence: Double = 0.55,
        val maximumCycleResidualNanos: Long = 2_500_000L,
        val minimumCycleConsistency: Double = 0.85,
        val maximumResidualNanos: Long = 6_000_000L,
        val maximumAuthorityNodes: Int = 4,
        val robustHuberDeltaNanos: Double = 1_250_000.0,
    ) {
        init {
            require(maximumNodes in 2..128)
            require(maximumEdges >= maximumNodes)
            require(maximumIterations in 4..128)
            require(convergenceToleranceNanos > 0.0)
            require(minimumNodeAuthority in 0.0..1.0)
            require(minimumEdgeConfidence in 0.0..1.0)
            require(maximumCycleResidualNanos > 0L)
            require(minimumCycleConsistency in 0.0..1.0)
            require(maximumResidualNanos >= maximumCycleResidualNanos)
            require(maximumAuthorityNodes in 1..16)
            require(robustHuberDeltaNanos > 0.0)
        }
    }

    data class Node(
        val deviceId: DeviceId,
        val timingAuthority: Double,
        val timestampDomain:
            MultiCameraTemporalAlignmentEngine.TimestampDomain =
            MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC,
        val hardwareSynchronized: Boolean = false,
        val visualBeaconConfidence: Double = 0.0,
    ) {
        init {
            require(deviceId.value.isNotBlank())
            require(timingAuthority.isFinite() && timingAuthority in 0.0..1.0)
            require(visualBeaconConfidence in 0.0..1.0)
        }
    }

    data class Edge(
        val from: DeviceId,
        val to: DeviceId,
        val offsetNanos: Long,
        val residualNanos: Long = 0L,
        val confidence: Double = 1.0,
    ) {
        init {
            require(from != to)
            require(offsetNanos in -5_000_000_000L..5_000_000_000L)
            require(residualNanos >= 0L)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class NodeEstimate(
        val deviceId: DeviceId,
        val offsetNanos: Long,
        val authority: Double,
        val residualRmsNanos: Double,
        val connected: Boolean,
        val authoritative: Boolean,
    )

    data class Result(
        val offsets: Map<DeviceId, Long>,
        val estimates: List<NodeEstimate>,
        val authorityNodes: List<DeviceId>,
        val cycleConsistency: Double,
        val graphCoverage: Double,
        val residualP95Nanos: Double,
        val confidence: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    fun solve(
        nodes: List<Node>,
        edges: List<Edge>,
    ): Result {
        val boundedNodes =
            nodes
                .distinctBy { it.deviceId }
                .take(config.maximumNodes)

        if (boundedNodes.isEmpty()) {
            return Result(
                offsets = emptyMap(),
                estimates = emptyList(),
                authorityNodes = emptyList(),
                cycleConsistency = 0.0,
                graphCoverage = 0.0,
                residualP95Nanos = Double.POSITIVE_INFINITY,
                confidence = 0.0,
                accepted = false,
                warnings =
                    listOf(
                        "CLOCK_GRAPH_NO_NODES",
                    ),
            )
        }

        val nodeMap =
            boundedNodes.associateBy { it.deviceId }

        val usableEdges =
            edges
                .filter { edge ->
                    nodeMap.containsKey(edge.from) &&
                        nodeMap.containsKey(edge.to) &&
                        edge.confidence >=
                            config.minimumEdgeConfidence
                }
                .distinctBy {
                    it.from to it.to
                }
                .take(config.maximumEdges)

        if (usableEdges.isEmpty()) {
            return Result(
                offsets =
                    boundedNodes
                        .associate { it.deviceId to 0L },
                estimates =
                    boundedNodes.map {
                        NodeEstimate(
                            deviceId = it.deviceId,
                            offsetNanos = 0L,
                            authority = 0.0,
                            residualRmsNanos =
                                Double.POSITIVE_INFINITY,
                            connected = false,
                            authoritative = false,
                        )
                    },
                authorityNodes = emptyList(),
                cycleConsistency = 0.0,
                graphCoverage =
                    1.0 /
                        boundedNodes.size.toDouble(),
                residualP95Nanos =
                    Double.POSITIVE_INFINITY,
                confidence = 0.0,
                accepted = false,
                warnings =
                    listOf(
                        "CLOCK_GRAPH_NO_USABLE_EDGES",
                    ),
            )
        }

        val degree =
            HashMap<DeviceId, Int>()
        usableEdges.forEach { edge ->
            degree[edge.from] =
                (degree[edge.from] ?: 0) + 1
            degree[edge.to] =
                (degree[edge.to] ?: 0) + 1
        }

        val graphAuthorityCandidates =
            boundedNodes
                .filter {
                    (degree[it.deviceId] ?: 0) > 0 &&
                        it.timingAuthority >=
                            config.minimumNodeAuthority
                }
                .sortedByDescending {
                    it.timingAuthority
                }

        val authorityNodes =
            graphAuthorityCandidates
                .take(config.maximumAuthorityNodes)
                .map { it.deviceId }

        val gauge =
            (
                authorityNodes.firstOrNull()
                    ?: boundedNodes
                        .maxByOrNull {
                            it.timingAuthority
                        }
                    ?: boundedNodes.first()
            ).deviceId

        val index =
            boundedNodes
                .mapIndexed { i, node ->
                    node.deviceId to i
                }
                .toMap()

        val offsets =
            DoubleArray(boundedNodes.size)

        for (iteration in 0 until config.maximumIterations) {
            var maximumDelta = 0.0

            boundedNodes.forEach { node ->
                if (node.deviceId == gauge) return@forEach

                val i =
                    index.getValue(node.deviceId)

                var numerator = 0.0
                var denominator = 0.0

                usableEdges.forEach { edge ->
                    when (node.deviceId) {
                        edge.to -> {
                            val neighbor =
                                offsets[
                                    index.getValue(edge.from)
                                ]
                            val expected =
                                neighbor +
                                    edge.offsetNanos
                                        .toDouble()
                            val residual =
                                expected -
                                    offsets[i]
                            val weight =
                                robustWeight(
                                    residual,
                                    edge.confidence,
                                    edge.residualNanos,
                                )
                            numerator +=
                                expected * weight
                            denominator += weight
                        }

                        edge.from -> {
                            val neighbor =
                                offsets[
                                    index.getValue(edge.to)
                                ]
                            val expected =
                                neighbor -
                                    edge.offsetNanos
                                        .toDouble()
                            val residual =
                                expected -
                                    offsets[i]
                            val weight =
                                robustWeight(
                                    residual,
                                    edge.confidence,
                                    edge.residualNanos,
                                )
                            numerator +=
                                expected * weight
                            denominator += weight
                        }

                        else -> Unit
                    }
                }

                if (denominator > 0.0) {
                    val next =
                        numerator / denominator
                    maximumDelta =
                        max(
                            maximumDelta,
                            abs(
                                next -
                                    offsets[i],
                            ),
                        )
                    offsets[i] =
                        next
                }
            }

            if (
                maximumDelta <=
                    config.convergenceToleranceNanos
            ) {
                break
            }
        }

        offsets[index.getValue(gauge)] = 0.0

        val edgeResiduals =
            usableEdges.map { edge ->
                abs(
                    offsets[
                        index.getValue(edge.to)
                    ] -
                        offsets[
                            index.getValue(edge.from)
                        ] -
                        edge.offsetNanos.toDouble(),
                )
            }

        val residualP95 =
            percentile(
                edgeResiduals.sorted(),
                0.95,
            )

        val spanningTreeKeys =
            spanningTreeEdgeKeys(
                gauge = gauge,
                edges = usableEdges,
            )

        val cycleEdges =
            usableEdges.filter { edge ->
                undirectedEdgeKey(
                    edge.from,
                    edge.to,
                ) !in spanningTreeKeys
            }

        val cycleResidualP95 =
            if (cycleEdges.isEmpty()) {
                0.0
            } else {
                percentile(
                    cycleEdges
                        .map { edge ->
                            abs(
                                offsets[
                                    index.getValue(edge.to)
                                ] -
                                    offsets[
                                        index.getValue(edge.from)
                                    ] -
                                    edge.offsetNanos.toDouble()
                            )
                        }
                        .sorted(),
                    0.95,
                )
            }

        val cycleConsistency =
            (
                1.0 -
                    cycleResidualP95 /
                        config.maximumCycleResidualNanos
                            .toDouble()
            ).coerceIn(0.0, 1.0)

        val connectedNodes =
            connectedComponent(
                gauge = gauge,
                edges = usableEdges,
            )

        val graphCoverage =
            connectedNodes.size.toDouble() /
                boundedNodes.size
                    .toDouble()
                    .coerceAtLeast(1.0)

        val estimates =
            boundedNodes.map { node ->
                val nodeResiduals =
                    usableEdges
                        .asSequence()
                        .filter {
                            it.from == node.deviceId ||
                                it.to == node.deviceId
                        }
                        .map {
                            abs(
                                offsets[
                                    index.getValue(it.to)
                                ] -
                                    offsets[
                                        index.getValue(it.from)
                                    ] -
                                    it.offsetNanos
                                        .toDouble()
                            )
                        }
                        .toList()

                val rms =
                    if (nodeResiduals.isEmpty()) {
                        Double.POSITIVE_INFINITY
                    } else {
                        kotlin.math.sqrt(
                            nodeResiduals
                                .map { it * it }
                                .average(),
                        )
                    }

                val residualScore =
                    if (rms.isFinite()) {
                        (
                            1.0 -
                                rms /
                                    config.maximumResidualNanos
                                        .toDouble()
                        ).coerceIn(0.0, 1.0)
                    } else {
                        0.0
                    }

                val timestampSourceScore =
                    timestampSourceScore(node)

                val connectivityScore =
                    if (
                        connectedNodes.contains(
                            node.deviceId,
                        )
                    ) {
                        1.0
                    } else {
                        0.0
                    }

                val authority =
                    (
                        node.timingAuthority * 0.50 +
                            timestampSourceScore * 0.15 +
                            residualScore * 0.25 +
                            connectivityScore * 0.10
                    ).coerceIn(0.0, 1.0)

                NodeEstimate(
                    deviceId = node.deviceId,
                    offsetNanos =
                        offsets[
                            index.getValue(node.deviceId)
                        ].toLong(),
                    authority = authority,
                    residualRmsNanos = rms,
                    connected =
                        connectivityScore > 0.0,
                    authoritative =
                        authority >=
                            config.minimumNodeAuthority &&
                            connectedNodes.contains(
                                node.deviceId,
                            ),
                )
            }

        val finalAuthorities =
            estimates
                .filter { it.authoritative }
                .sortedByDescending {
                    it.authority
                }
                .take(config.maximumAuthorityNodes)
                .map { it.deviceId }

        val authorityScore =
            if (estimates.isEmpty()) {
                0.0
            } else {
                finalAuthorities
                    .map { id ->
                        estimates
                            .first { it.deviceId == id }
                            .authority
                    }
                    .averageOrZero()
            }

        val residualScore =
            (
                1.0 -
                    residualP95 /
                        config.maximumResidualNanos
                            .toDouble()
            ).coerceIn(0.0, 1.0)

        val confidence =
            (
                authorityScore * 0.35 +
                    cycleConsistency * 0.30 +
                    graphCoverage * 0.20 +
                    residualScore * 0.15
            ).coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (
                    finalAuthorities.size < 2
                ) {
                    add("CLOCK_GRAPH_SINGLE_AUTHORITY")
                }
                if (
                    cycleConsistency <
                        config.minimumCycleConsistency
                ) {
                    add("CLOCK_GRAPH_CYCLE_INCONSISTENCY")
                }
                if (
                    graphCoverage < 0.90
                ) {
                    add("CLOCK_GRAPH_COVERAGE_INCOMPLETE")
                }
                if (
                    residualP95 >
                        config.maximumResidualNanos
                ) {
                    add("CLOCK_GRAPH_RESIDUAL_HIGH")
                }
                if (
                    authorityScore <
                        config.minimumNodeAuthority
                ) {
                    add("CLOCK_GRAPH_AUTHORITY_WEAK")
                }
            }.distinct()

        return Result(
            offsets =
                estimates.associate {
                    it.deviceId to it.offsetNanos
                },
            estimates = estimates,
            authorityNodes = finalAuthorities,
            cycleConsistency = cycleConsistency,
            graphCoverage = graphCoverage,
            residualP95Nanos = residualP95,
            confidence = confidence,
            accepted =
                connectedNodes.size >= 2 &&
                    finalAuthorities.isNotEmpty() &&
                    cycleConsistency >=
                        config.minimumCycleConsistency &&
                    residualP95 <=
                        config.maximumResidualNanos &&
                    confidence >=
                        config.minimumNodeAuthority,
            warnings = warnings,
        )
    }

    private fun timestampSourceScore(
        node: Node,
    ): Double {
        val domainScore =
            when (node.timestampDomain) {
                MultiCameraTemporalAlignmentEngine.TimestampDomain.SENSOR ->
                    if (node.hardwareSynchronized) 1.0 else 0.84
                MultiCameraTemporalAlignmentEngine.TimestampDomain.REALTIME ->
                    0.72
                MultiCameraTemporalAlignmentEngine.TimestampDomain.MONOTONIC ->
                    0.78
                MultiCameraTemporalAlignmentEngine.TimestampDomain.UNKNOWN ->
                    0.25
            }

        val hardwareScore =
            if (node.hardwareSynchronized) 1.0 else 0.70

        return (
            domainScore * 0.60 +
                hardwareScore * 0.20 +
                node.visualBeaconConfidence.coerceIn(0.0, 1.0) * 0.20
            ).coerceIn(0.0, 1.0)
    }

    private fun robustWeight(
        residualNanos: Double,
        confidence: Double,
        measuredResidualNanos: Long,
    ): Double {
        val robust =
            if (
                abs(residualNanos) <=
                    config.robustHuberDeltaNanos
            ) {
                1.0
            } else {
                config.robustHuberDeltaNanos /
                    abs(residualNanos)
            }

        val measuredResidualWeight =
            (
                1.0 -
                    measuredResidualNanos.toDouble() /
                        config.maximumResidualNanos
            ).coerceIn(0.05, 1.0)

        return (
            robust *
                confidence.coerceIn(0.01, 1.0) *
                measuredResidualWeight
            ).coerceAtLeast(1e-6)
    }

    private fun edgeKey(
        edge: Edge,
    ): String =
        edge.from.value + "->" + edge.to.value

    private fun undirectedEdgeKey(
        first: DeviceId,
        second: DeviceId,
    ): String =
        if (first.value <= second.value) {
            first.value + "|" + second.value
        } else {
            second.value + "|" + first.value
        }

    private fun spanningTreeEdgeKeys(
        gauge: DeviceId,
        edges: List<Edge>,
    ): Set<String> {
        val adjacency =
            HashMap<DeviceId, MutableList<Edge>>()

        edges.forEach { edge ->
            adjacency
                .getOrPut(edge.from) { ArrayList() }
                .add(edge)
            adjacency
                .getOrPut(edge.to) { ArrayList() }
                .add(
                    edge.copy(
                        from = edge.to,
                        to = edge.from,
                        offsetNanos = -edge.offsetNanos,
                    ),
                )
        }

        val visited =
            LinkedHashSet<DeviceId>()
        val queue =
            ArrayDeque<DeviceId>()
        val tree =
            LinkedHashSet<String>()

        queue.add(gauge)
        visited.add(gauge)

        while (queue.isNotEmpty()) {
            val current =
                queue.removeFirst()

            adjacency[current]
                .orEmpty()
                .sortedBy {
                    it.to.value
                }
                .forEach { edge ->
                    if (visited.add(edge.to)) {
                        tree +=
                            undirectedEdgeKey(
                                current,
                                edge.to,
                            )
                        queue.add(edge.to)
                    }
                }
        }

        return tree
    }

    private fun connectedComponent(
        gauge: DeviceId,
        edges: List<Edge>,
    ): Set<DeviceId> {
        val adjacency =
            HashMap<DeviceId, MutableList<DeviceId>>()

        edges.forEach { edge ->
            adjacency
                .getOrPut(edge.from) {
                    ArrayList()
                }
                .add(edge.to)
            adjacency
                .getOrPut(edge.to) {
                    ArrayList()
                }
                .add(edge.from)
        }

        val visited =
            LinkedHashSet<DeviceId>()
        val queue =
            ArrayDeque<DeviceId>()
        queue.add(gauge)
        visited.add(gauge)

        while (queue.isNotEmpty()) {
            val current = queue.removeFirst()
            adjacency[current]
                .orEmpty()
                .sortedBy { it.value }
                .forEach { next ->
                    if (visited.add(next)) {
                        queue.add(next)
                    }
                }
        }

        return visited
    }

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        if (values.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        val position =
            probability.coerceIn(0.0, 1.0) *
                (values.size - 1)

        val lower =
            position.toInt()
        val upper =
            min(
                values.lastIndex,
                lower + 1,
            )
        val fraction =
            position -
                lower.toDouble()

        return values[lower] *
            (1.0 - fraction) +
            values[upper] *
            fraction
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
