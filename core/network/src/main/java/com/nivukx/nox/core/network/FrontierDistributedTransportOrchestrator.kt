package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId

/**
 * Unified distributed transport control plane.
 *
 * Combines topology selection, frontier bandwidth allocation, peer-quality enrichment and routed
 * evidence planning while keeping the established transport components authoritative.
 */
class FrontierDistributedTransportOrchestrator(
    private val transportBudget: FrontierCameraTransportBudgetEngine = FrontierCameraTransportBudgetEngine(),
    private val peerQuality: PeerTransportQualityEngine = PeerTransportQualityEngine(),
    private val routedTransport: DistributedTransportCoordinator = DistributedTransportCoordinator(),
) : AutoCloseable {

    private var closed = false

    private fun ensureOpen() {
        check(!closed) {
            "FrontierDistributedTransportOrchestrator is closed"
        }
    }
    data class Result(
        val topology: NearbyTransportTopologySelector.Mode,
        val budget: FrontierCameraTransportBudgetEngine.Result,
        val enrichedNodes: List<DistributedTransportCoordinator.Node>,
        val warnings: List<String>,
    )

    @Synchronized
    fun evaluate(
        channels: List<FrontierCameraTransportBudgetEngine.CameraChannel>,
        linkCapacityKbps: Double,
        timestampNanos: Long,
        expectedPeerCount: Int = channels.size,
        nodes: List<DistributedTransportCoordinator.Node> = emptyList(),
        qualities: Map<DeviceId, PeerTransportQualityEngine.Quality> = emptyMap(),
        mode: NearbyTransportTopologySelector.Mode = NearbyTransportTopologySelector.Mode.AUTO,
    ): Result {
        ensureOpen()
        require(expectedPeerCount >= 1)
        val topology = NearbyTransportTopologySelector.resolve(
            mode = mode,
            expectedPeerCount = expectedPeerCount,
        )
        val budget = transportBudget.allocate(
            channels = channels,
            linkCapacityKbps = linkCapacityKbps,
            timestampNanos = timestampNanos,
        )
        val enrichedNodes = routedTransport.enrichNodesWithTransportQuality(
            nodes = nodes,
            qualities = qualities,
        )
        return Result(
            topology = topology,
            budget = budget,
            enrichedNodes = enrichedNodes,
            warnings = buildList {
                addAll(budget.warnings)
                if (topology == NearbyTransportTopologySelector.Mode.CLUSTER && expectedPeerCount < 16) {
                    add("cluster-topology-used-for-small-federation")
                }
                if (enrichedNodes.any { it.health < 0.45 }) {
                    add("distributed-node-health-low")
                }
            }.distinct(),
        )
    }

    @Synchronized
    fun route(
        nodes: List<DistributedTransportCoordinator.Node>,
        links: List<DistributedTransportCoordinator.Link>,
        destination: DeviceId,
        evidence: DistributedTransportCoordinator.Evidence,
        source: DeviceId? = null,
        relayLoads: Map<DeviceId, Int> = emptyMap(),
        qualities: Map<DeviceId, PeerTransportQualityEngine.Quality> = emptyMap(),
    ): DistributedTransportCoordinator.Plan {
        ensureOpen()
        val enriched = routedTransport.enrichNodesWithTransportQuality(nodes, qualities)
        return routedTransport.plan(
            nodes = enriched,
            links = links,
            source = source,
            destination = destination,
            evidence = evidence,
            relayLoads = relayLoads,
        )
    }

    @Synchronized
    fun observePingSent(endpoint: String, sequence: Long, timestampNanos: Long) {
        ensureOpen()
        peerQuality.observePingSent(endpoint, sequence, timestampNanos)
    }

    @Synchronized
    fun observePong(endpoint: String, sequence: Long, timestampNanos: Long) {
        ensureOpen()
        peerQuality.observePong(endpoint, sequence, timestampNanos)
    }

    @Synchronized
    fun observePayloadStarted(
        endpoint: String,
        payloadId: Long,
        totalBytes: Long,
        timestampNanos: Long,
    ) {
        ensureOpen()
        peerQuality.observePayloadStarted(endpoint, payloadId, totalBytes, timestampNanos)
    }

    @Synchronized
    fun observePayloadProgress(
        endpoint: String,
        payloadId: Long,
        bytesTransferred: Long,
        totalBytes: Long,
        status: Int,
        timestampNanos: Long,
    ) {
        ensureOpen()
        peerQuality.observePayloadProgress(
            endpoint,
            payloadId,
            bytesTransferred,
            totalBytes,
            status,
            timestampNanos,
        )
    }

    @Synchronized
    fun peerQuality(endpoint: String, nowNanos: Long): PeerTransportQualityEngine.Quality {
        ensureOpen()
        return peerQuality.snapshot(endpoint, nowNanos)
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        peerQuality.clear()
    }
}
