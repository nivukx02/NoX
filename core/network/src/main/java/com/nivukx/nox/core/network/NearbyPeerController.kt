package com.nivukx.nox.core.network

import android.content.Context
import android.os.Build
import com.google.android.gms.nearby.Nearby
import com.google.android.gms.nearby.connection.AdvertisingOptions
import com.google.android.gms.nearby.connection.ConnectionInfo
import com.google.android.gms.nearby.connection.ConnectionLifecycleCallback
import com.google.android.gms.nearby.connection.ConnectionResolution
import com.google.android.gms.nearby.connection.DiscoveredEndpointInfo
import com.google.android.gms.nearby.connection.DiscoveryOptions
import com.google.android.gms.nearby.connection.EndpointDiscoveryCallback
import com.google.android.gms.nearby.connection.Payload
import com.google.android.gms.nearby.connection.PayloadCallback
import com.google.android.gms.nearby.connection.PayloadTransferUpdate
import com.google.android.gms.nearby.connection.Strategy
import com.nivukx.nox.core.model.DeviceId
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

class NearbyPeerController(
    context: Context,
    private val serviceId: String = "com.nivukx.nox.mesh.v2",
    private val localDeviceId: DeviceId? = null,
    private val transportTopology: TransportTopology =
        TransportTopology.STAR,
    private val peerAdmissionPolicy: PeerAdmissionPolicy =
        PeerAdmissionPolicy(),
    private val peerCapacityPolicy: NearbyPeerCapacityPolicy =
        NearbyPeerCapacityPolicy(),
    private val expectedPeerCount: Int = 2,
    private val autoClusterThreshold: Int = 32,
    private val localPeerCapabilities:
        PeerCapabilitiesWireCodec.Capabilities =
        PeerCapabilitiesWireCodec.localDefaults(),
) {

    enum class TransportTopology(
        val strategyName: String,
    ) {
        STAR("P2P_STAR"),
        CLUSTER("P2P_CLUSTER"),
        AUTO("P2P_AUTO"),
    }

    companion object {
        // Nearby byte payloads are constrained to 32 KiB in the supported API contract.
        // The NoX wire envelope uses the same ceiling so transport and protocol validation agree.
        const val MAX_NEARBY_BYTES_PAYLOAD = 32 * 1024
        private const val MAX_FRAGMENT_HEADER_BYTES = 28 + 96
        private const val NOX_HEADER_BYTES = 36
        private const val MAX_DEPTH_FRAGMENT_HEADER_BYTES =
            32 + 64 + 96
    }

    init {
        require(expectedPeerCount >= 1)
        require(autoClusterThreshold in 2..128)
    }

    data class PeerPacket(val endpoint: DeviceId, val payload: ByteArray)

    data class DenseLandmarkFrame(
        val endpoint: DeviceId,
        val frame: DenseLandmarkWireCodec.Frame,
    )

    data class MultiActorPoseFrame(
        val endpoint: DeviceId,
        val frame: MultiActorPoseWireCodec.Frame,
    )

    data class FaceMeshFrame(
        val endpoint: DeviceId,
        val frame: com.nivukx.nox.core.model.FaceMeshFrame,
    )

    data class TimingBeaconObservation(
        val endpoint: DeviceId,
        val observation:
            TimingBeaconObservationWireCodec.Observation,
    )

    data class DepthFrame(
        val endpoint: DeviceId,
        val frame: DepthWireCodec.Frame,
    )

    private val client = Nearby.getConnectionsClient(context.applicationContext)
    private val peers = ConcurrentHashMap<String, DeviceId>()
    private val peerCapabilities =
        ConcurrentHashMap<String, PeerCapabilitiesWireCodec.Capabilities>()
    private val connectingEndpoints =
        ConcurrentHashMap.newKeySet<String>()
    private val replayGuard = PacketReplayGuard()
    private val peerTransportQuality =
        PeerTransportQualityEngine()
    private val transportSheddingPolicy =
        PeerTransportSheddingPolicy()
    private val transportCongestionController =
        PeerTransportCongestionController()
    private val controlSequence =
        AtomicLong(1L)
    private val _connected = MutableStateFlow<Set<DeviceId>>(emptySet())
    private val _pendingPeerRequests =
        MutableStateFlow<List<PeerAdmissionPolicy.Request>>(
            emptyList(),
        )
    private val _packets =
        MutableSharedFlow<PeerPacket>(
            extraBufferCapacity = 256,
        )
    private val _denseLandmarkFrames =
        MutableSharedFlow<DenseLandmarkFrame>(
            extraBufferCapacity = 64,
        )
    private val denseLandmarkReassembler =
        DenseLandmarkFragmentReassembler()
    private val multiActorPoseReassembler =
        MultiActorPoseFragmentReassembler()
    private val faceMeshReassembler =
        FaceMeshFragmentReassembler()
    private val depthFragmentReassembler =
        DepthFragmentReassembler()
    private val fragmentRetransmissionCache =
        EvidenceFragmentRetransmissionCache()
    private val fragmentReplyRouteRegistry =
        FragmentReplyRouteRegistry()

    val connected: StateFlow<Set<DeviceId>> =
        _connected.asStateFlow()
    val pendingPeerRequests:
        StateFlow<List<PeerAdmissionPolicy.Request>> =
        _pendingPeerRequests.asStateFlow()
    val packets: SharedFlow<PeerPacket> =
        _packets.asSharedFlow()
    val denseLandmarkFrames: SharedFlow<DenseLandmarkFrame> =
        _denseLandmarkFrames.asSharedFlow()
    private val _multiActorPoseFrames =
        MutableSharedFlow<MultiActorPoseFrame>(
            extraBufferCapacity = 64,
        )
    val multiActorPoseFrames: SharedFlow<MultiActorPoseFrame> =
        _multiActorPoseFrames.asSharedFlow()
    private val _faceMeshFrames =
        MutableSharedFlow<FaceMeshFrame>(
            extraBufferCapacity = 32,
        )
    val faceMeshFrames: SharedFlow<FaceMeshFrame> =
        _faceMeshFrames.asSharedFlow()
    private val _timingBeaconObservations =
        MutableSharedFlow<TimingBeaconObservation>(
            extraBufferCapacity = 256,
        )
    val timingBeaconObservations:
        SharedFlow<TimingBeaconObservation> =
        _timingBeaconObservations.asSharedFlow()
    private val _depthFrames =
        MutableSharedFlow<DepthFrame>(
            extraBufferCapacity = 32,
        )
    val depthFrames: SharedFlow<DepthFrame> =
        _depthFrames.asSharedFlow()

    fun peerCapabilities(
        endpoint: DeviceId,
    ): PeerCapabilitiesWireCodec.Capabilities =
        peerCapabilities[endpoint.value]
            ?: PeerCapabilitiesWireCodec.legacyFallback()

    fun allPeerCapabilities():
        Map<DeviceId, PeerCapabilitiesWireCodec.Capabilities> =
        peerCapabilities
            .mapKeys { (endpoint, _) -> DeviceId(endpoint) }
            .toMap()

    private val payloadCallback = object : PayloadCallback() {
        override fun onPayloadReceived(endpointId: String, payload: Payload) {
            if (payload.type != Payload.Type.BYTES) return
            val bytes = payload.asBytes() ?: return
            if (bytes.size > MAX_NEARBY_BYTES_PAYLOAD) return

            if (!replayGuard.accept(endpointId, bytes)) {
                return
            }

            val packet =
                runCatching {
                    NoXWireProtocol.decode(bytes)
                }.getOrNull()
                ?: return

            expireFragmentAssembliesAndRequestNacks(
                System.nanoTime(),
            )

            dispatchPacket(
                endpointId = endpointId,
                packet = packet,
            )

            _packets.tryEmit(
                PeerPacket(
                    endpoint = DeviceId(endpointId),
                    payload = bytes,
                ),
            )
        }

        override fun onPayloadTransferUpdate(
            endpointId: String,
            update: PayloadTransferUpdate,
        ) {
            peerTransportQuality.observePayloadProgress(
                endpoint = endpointId,
                payloadId = update.payloadId,
                bytesTransferred = update.bytesTransferred,
                totalBytes = update.totalBytes,
                status = update.status,
                observedAtNanos = System.nanoTime(),
            )
        }
    }

    private val callback = object : ConnectionLifecycleCallback() {
        override fun onConnectionInitiated(
            endpointId: String,
            connectionInfo: ConnectionInfo,
        ) {
            val result =
                runCatching {
                    peerAdmissionPolicy.request(
                        endpoint =
                            DeviceId(endpointId),
                        authenticationDigits =
                            connectionInfo
                                .getAuthenticationDigits(),
                        endpointName =
                            connectionInfo
                                .getEndpointName(),
                        nowNanos =
                            System.nanoTime(),
                    )
                }.getOrElse {
                    return
                }

            _pendingPeerRequests.value =
                peerAdmissionPolicy.pendingRequests(
                    System.nanoTime(),
                )

            when {
                result.acceptImmediately -> {
                    if (
                        !peerCapacityPolicy.tryReserveConnected(
                            endpointId,
                        )
                    ) {
                        peerAdmissionPolicy.reject(
                            DeviceId(endpointId),
                        )
                        _pendingPeerRequests.value =
                            peerAdmissionPolicy.pendingRequests(
                                System.nanoTime(),
                            )
                        client.rejectConnection(
                            endpointId,
                        )
                    } else {
                        client.acceptConnection(
                            endpointId,
                            payloadCallback,
                        )
                    }
                }

                result.decision ==
                    PeerAdmissionPolicy.Decision.REJECT -> {
                    peerAdmissionPolicy.reject(
                        DeviceId(endpointId),
                    )
                    _pendingPeerRequests.value =
                        peerAdmissionPolicy.pendingRequests(
                            System.nanoTime(),
                        )
                    client.rejectConnection(
                        endpointId,
                    )
                }

                else -> Unit
            }
        }

        override fun onConnectionResult(
            endpointId: String,
            result: ConnectionResolution,
        ) {
            connectingEndpoints.remove(endpointId)
            if (!result.status.isSuccess) {
                peerCapacityPolicy.release(endpointId)
                return
            }

            if (
                !peerCapacityPolicy.tryReserveConnected(
                    endpointId,
                )
            ) {
                runCatching {
                    client.rejectConnection(endpointId)
                }
                return
            }

            peers[endpointId] = DeviceId(endpointId)
            publishPeers()

            val now = System.nanoTime()
            send(
                endpoint = DeviceId(endpointId),
                packet =
                    NoXWireProtocol.encode(
                        type = NoXWireProtocol.Type.HELLO,
                        sequence = now,
                        timestampNanos = now,
                        payload =
                            PeerCapabilitiesWireCodec.encode(
                                localPeerCapabilities,
                            ),
                    ),
            )
        }

        override fun onDisconnected(endpointId: String) {
            peers.remove(endpointId)
            connectingEndpoints.remove(endpointId)
            peerCapacityPolicy.release(endpointId)
            replayGuard.clear(endpointId)
            denseLandmarkReassembler.clearEndpoint(endpointId)
            multiActorPoseReassembler.clearEndpoint(endpointId)
            faceMeshReassembler.clearEndpoint(endpointId)
            depthFragmentReassembler.clearEndpoint(endpointId)
            fragmentRetransmissionCache.clearEndpoint(endpointId)
            fragmentReplyRouteRegistry.clearEndpoint(endpointId)
            peerTransportQuality.clearEndpoint(endpointId)
            transportCongestionController.resetEndpoint(
                endpointId,
            )
            peerCapabilities.remove(endpointId)
            peerAdmissionPolicy.reject(
                DeviceId(endpointId),
            )
            _pendingPeerRequests.value =
                peerAdmissionPolicy.pendingRequests(
                    System.nanoTime(),
                )
            publishPeers()
        }
    }

    fun resolvedTransportTopology(): TransportTopology =
        when (
            NearbyTransportTopologySelector.resolve(
                mode =
                    when (transportTopology) {
                        TransportTopology.STAR ->
                            NearbyTransportTopologySelector.Mode.STAR
                        TransportTopology.CLUSTER ->
                            NearbyTransportTopologySelector.Mode.CLUSTER
                        TransportTopology.AUTO ->
                            NearbyTransportTopologySelector.Mode.AUTO
                    },
                expectedPeerCount = expectedPeerCount,
                config =
                    NearbyTransportTopologySelector.Config(
                        autoClusterThreshold =
                            autoClusterThreshold,
                    ),
            )
        ) {
            NearbyTransportTopologySelector.Mode.STAR ->
                TransportTopology.STAR
            NearbyTransportTopologySelector.Mode.CLUSTER ->
                TransportTopology.CLUSTER
            NearbyTransportTopologySelector.Mode.AUTO ->
                TransportTopology.STAR
        }

    private fun TransportTopology.toNearbyStrategy(): Strategy =
        when (this) {
            TransportTopology.STAR ->
                Strategy.P2P_STAR
            TransportTopology.CLUSTER ->
                Strategy.P2P_CLUSTER
            TransportTopology.AUTO ->
                resolvedTransportTopology()
                    .toNearbyStrategy()
        }

    fun startAdvertising(deviceName: String) {
        client.startAdvertising(
            deviceName,
            serviceId,
            callback,
            AdvertisingOptions.Builder()
                .setStrategy(
                    transportTopology.toNearbyStrategy(),
                )
                .build(),
        )
    }

    fun startDiscovery() {
        client.startDiscovery(
            serviceId,
            object : EndpointDiscoveryCallback() {
                override fun onEndpointFound(
                    endpointId: String,
                    info: DiscoveredEndpointInfo,
                ) {
                    if (
                        peers.containsKey(endpointId) ||
                        connectingEndpoints.contains(endpointId)
                    ) {
                        return
                    }

                    if (
                        !peerCapacityPolicy.tryReserveConnecting(
                            endpointId,
                        )
                    ) {
                        return
                    }

                    connectingEndpoints += endpointId

                    runCatching {
                        client.requestConnection(
                            "NoX-" + Build.MODEL,
                            endpointId,
                            callback,
                        )
                    }.onFailure {
                        connectingEndpoints.remove(endpointId)
                        peerCapacityPolicy.releaseConnecting(
                            endpointId,
                        )
                    }
                }

                override fun onEndpointLost(
                    endpointId: String,
                ) = Unit
            },
            DiscoveryOptions.Builder()
                .setStrategy(
                    transportTopology.toNearbyStrategy(),
                )
                .build(),
        )
    }

    fun approvePeerConnection(
        endpoint: DeviceId,
        authenticationDigits: String,
    ): Boolean {
        val accepted =
            peerAdmissionPolicy.approve(
                endpoint = endpoint,
                authenticationDigits =
                    authenticationDigits,
                nowNanos =
                    System.nanoTime(),
            )

        _pendingPeerRequests.value =
            peerAdmissionPolicy.pendingRequests(
                System.nanoTime(),
            )

        if (!accepted) {
            return false
        }

        if (
            !peerCapacityPolicy.tryReserveConnected(
                endpoint.value,
            )
        ) {
            return false
        }

        client.acceptConnection(
            endpoint.value,
            payloadCallback,
        )
        return true
    }

    fun rejectPeerConnection(
        endpoint: DeviceId,
    ): Boolean {
        val rejected =
            peerAdmissionPolicy.reject(
                endpoint,
            )

        _pendingPeerRequests.value =
            peerAdmissionPolicy.pendingRequests(
                System.nanoTime(),
            )

        if (rejected) {
            client.rejectConnection(
                endpoint.value,
            )
        }

        return rejected
    }

    fun trustPeer(
        endpoint: DeviceId,
    ) {
        peerAdmissionPolicy.trust(endpoint)
    }

    fun revokePeerTrust(
        endpoint: DeviceId,
    ) {
        peerAdmissionPolicy.revoke(endpoint)
    }

    fun probePeer(
        endpoint: DeviceId,
    ): Long {
        require(peers.containsKey(endpoint.value)) {
            "Cannot probe a peer that is not connected"
        }

        val now =
            System.nanoTime()
        val sequence =
            controlSequence.getAndIncrement()

        peerTransportQuality.observePingSent(
            endpoint = endpoint.value,
            sequence = sequence,
            sentAtNanos = now,
        )

        send(
            endpoint = endpoint,
            packet =
                NoXWireProtocol.encode(
                    type = NoXWireProtocol.Type.PING,
                    sequence = sequence,
                    timestampNanos = now,
                ),
        )

        return sequence
    }

    fun probeAllPeers(): List<Long> =
        peers.values
            .sortedBy { it.value }
            .map(::probePeer)

    fun transportQuality(
        endpoint: DeviceId,
    ): PeerTransportQualityEngine.Quality {
        peerTransportQuality.expireOutstanding(
            System.nanoTime(),
        )
        return peerTransportQuality.snapshot(
            endpoint.value,
        )
    }

    fun sendRouted(
        route: DistributedTransportCoordinator.Route,
        origin: DeviceId,
        destination: DeviceId,
        packetSequence: Long,
        timestampNanos: Long,
        payload: ByteArray,
        reconstructionCritical: Boolean = false,
    ): Boolean {
        val localOrigin =
            localDeviceId
        if (
            localOrigin != null &&
            origin != localOrigin
        ) {
            return false
        }

        require(
            route.nodes.lastOrNull() == destination,
        )
        require(
            route.nodes.isNotEmpty(),
        )

        return sendRoutedAlongNodes(
            routeNodes = route.nodes,
            origin = origin,
            destination = destination,
            packetSequence = packetSequence,
            timestampNanos = timestampNanos,
            payload = payload,
            reconstructionCritical =
                reconstructionCritical,
        )
    }

    private fun sendRoutedAlongNodes(
        routeNodes: List<DeviceId>,
        origin: DeviceId,
        destination: DeviceId,
        packetSequence: Long,
        timestampNanos: Long,
        payload: ByteArray,
        reconstructionCritical: Boolean,
    ): Boolean {
        val localOrigin =
            localDeviceId
        if (
            localOrigin != null &&
            origin != localOrigin
        ) {
            return false
        }

        require(routeNodes.isNotEmpty())
        require(routeNodes.distinct().size == routeNodes.size)
        require(routeNodes.lastOrNull() == destination)

        val nextHop =
            routeNodes.firstOrNull()
                ?: return false

        if (!peers.containsKey(nextHop.value)) {
            return false
        }

        val packet =
            DistributedTransportEnvelopeCodec.encode(
                origin = origin,
                destination = destination,
                route = routeNodes,
                hopIndex = 0,
                ttl = routeNodes.size,
                packetSequence = packetSequence,
                timestampNanos = timestampNanos,
                payload = payload,
                reconstructionCritical =
                    reconstructionCritical,
            )

        return send(
            endpoint = nextHop,
            packet = packet,
            reconstructionCritical =
                reconstructionCritical,
        )
    }

    private fun sendCachedRoutedFragment(
        routeNodes: List<DeviceId>,
        origin: DeviceId,
        destination: DeviceId,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
        chunkIndex: Int,
        chunkCount: Int,
        packetSequence: Long,
        timestampNanos: Long,
        payload: ByteArray,
        reconstructionCritical: Boolean,
    ): Boolean {
        val sent =
            sendRoutedAlongNodes(
                routeNodes = routeNodes,
                origin = origin,
                destination = destination,
                packetSequence = packetSequence,
                timestampNanos = timestampNanos,
                payload = payload,
                reconstructionCritical =
                    reconstructionCritical,
            )

        if (sent) {
            fragmentRetransmissionCache.remember(
                endpoint = destination.value,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
                chunkIndex = chunkIndex,
                chunkCount = chunkCount,
                packet = payload,
                reconstructionCritical =
                    reconstructionCritical,
                nowNanos = System.nanoTime(),
            )
            fragmentReplyRouteRegistry.remember(
                endpoint = destination.value,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
                route = routeNodes,
                nowNanos = System.nanoTime(),
            )
        }

        return sent
    }

    fun sendRoutedDepthFrame(
        route: DistributedTransportCoordinator.Route,
        origin: DeviceId,
        destination: DeviceId,
        frame: com.nivukx.nox.core.model.DepthFrame,
        sourceDeviceId: DeviceId,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
    ): Boolean {
        val routeBudget =
            DistributedTransportEnvelopeCodec
                .maxPayloadBytesForRoute(
                    origin = origin,
                    destination = destination,
                    route = route.nodes,
                )

        val fragmentBudget =
            (
                routeBudget -
                    NOX_HEADER_BYTES -
                    MAX_DEPTH_FRAGMENT_HEADER_BYTES
            )
            .coerceAtLeast(4_096)
            .coerceAtMost(
                MAX_NEARBY_BYTES_PAYLOAD -
                    NOX_HEADER_BYTES -
                    MAX_DEPTH_FRAGMENT_HEADER_BYTES,
            )

        return runCatching {
            val payload =
                requireNotNull(
                    DepthWireCodec.encode(
                        deviceId = sourceDeviceId,
                        frame = frame,
                    ),
                )

            if (
                !peerCapabilities(destination)
                    .supportsDepthFragmentation
            ) {
                val rawPacket =
                    NoXWireProtocol.encode(
                        type =
                            NoXWireProtocol.Type.DEPTH_FRAME,
                        sequence = packetSequenceStart,
                        timestampNanos = frame.timestampNanos,
                        payload = payload,
                    )

                check(rawPacket.size <= routeBudget) {
                    "Destination lacks depth fragmentation and routed depth exceeds packet budget"
                }

                return@runCatching sendRouted(
                    route = route,
                    origin = origin,
                    destination = destination,
                    packetSequence =
                        packetSequenceStart,
                    timestampNanos =
                        frame.timestampNanos,
                    payload = rawPacket,
                    reconstructionCritical =
                        reconstructionCritical,
                )
            }

            DepthChunkedWireCodec
                .encode(
                    deviceId = sourceDeviceId,
                    payload = payload,
                    streamId = streamId,
                    frameSequence = frameSequence,
                    maximumPayloadBytes = fragmentBudget,
                )
                .forEachIndexed { index, chunk ->
                    val packetSequence =
                        fragmentPacketSequence(
                            packetSequenceStart,
                            index,
                        )
                    val packet =
                        DepthFragmentWireCodec.encode(
                            chunk = chunk,
                            packetSequence = packetSequence,
                        )

                    check(packet.size <= routeBudget) {
                        "Depth routed fragment exceeds route payload budget"
                    }

                    check(
                        sendCachedRoutedFragment(
                            routeNodes = route.nodes,
                            origin = origin,
                            destination = destination,
                            evidenceType =
                                NoXWireProtocol.Type.DEPTH_FRAGMENT,
                            streamId = streamId,
                            frameSequence = frameSequence,
                            chunkIndex = chunk.chunkIndex,
                            chunkCount = chunk.chunkCount,
                            packetSequence = packetSequence,
                            timestampNanos = chunk.timestampNanos,
                            payload = packet,
                            reconstructionCritical =
                                reconstructionCritical,
                        ),
                    )
                }

            true
        }.getOrDefault(false)
    }

    fun sendRoutedDenseLandmarkFrame(
        route: DistributedTransportCoordinator.Route,
        origin: DeviceId,
        destination: DeviceId,
        frame: DenseLandmarkWireCodec.Frame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
        wireVersion: DenseLandmarkChunkedWireCodec.FrameWireVersion =
            DenseLandmarkChunkedWireCodec.FrameWireVersion.V4_TIMING_METADATA,
    ): Boolean {
        val routeBudget =
            DistributedTransportEnvelopeCodec
                .maxPayloadBytesForRoute(
                    origin = origin,
                    destination = destination,
                    route = route.nodes,
                )

        val streamBytes =
            streamId.toByteArray(Charsets.UTF_8).size
        val fragmentBudget =
            (
                routeBudget -
                    36 -
                    28 -
                    streamBytes
            ).coerceAtLeast(1_024)

        return runCatching {
            DenseLandmarkChunkedWireCodec
                .encode(
                    frame = frame,
                    streamId = streamId,
                    frameSequence = frameSequence,
                    wireVersion = wireVersion,
                    maximumPayloadBytes =
                        fragmentBudget
                            .coerceAtMost(
                                routeBudget,
                            )
                            .coerceAtMost(
                                MAX_NEARBY_BYTES_PAYLOAD -
                                    NOX_HEADER_BYTES -
                                    MAX_FRAGMENT_HEADER_BYTES,
                            ),
                )
                .forEachIndexed { index, chunk ->
                    val packet =
                        DenseLandmarkFragmentWireCodec.encode(
                            chunk = chunk,
                            packetSequence =
                                fragmentPacketSequence(packetSequenceStart, index),
                        )
                    check(
                        packet.size <= routeBudget,
                    ) {
                        "Dense landmark routed fragment exceeds route payload budget"
                    }
                    check(
                        sendCachedRoutedFragment(
                            routeNodes = route.nodes,
                            origin = origin,
                            destination = destination,
                            evidenceType =
                                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                            streamId = streamId,
                            frameSequence = frameSequence,
                            chunkIndex = chunk.chunkIndex,
                            chunkCount = chunk.chunkCount,
                            packetSequence =
                                fragmentPacketSequence(packetSequenceStart, index),
                            timestampNanos =
                                chunk.timestampNanos,
                            payload = packet,
                            reconstructionCritical =
                                reconstructionCritical,
                        ),
                    )
                }
            true
        }.getOrDefault(false)
    }

    fun sendRoutedMultiActorPoseFrame(
        route: DistributedTransportCoordinator.Route,
        origin: DeviceId,
        destination: DeviceId,
        frame: MultiActorPoseWireCodec.Frame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
    ): Boolean {
        val routeBudget =
            DistributedTransportEnvelopeCodec
                .maxPayloadBytesForRoute(
                    origin = origin,
                    destination = destination,
                    route = route.nodes,
                )

        val streamBytes =
            streamId.toByteArray(Charsets.UTF_8).size
        val fragmentBudget =
            (
                routeBudget -
                    36 -
                    24 -
                    streamBytes
            ).coerceAtLeast(1_024)

        return runCatching {
            MultiActorPoseChunkedWireCodec
                .encode(
                    frame = frame,
                    streamId = streamId,
                    frameSequence = frameSequence,
                    maximumPayloadBytes =
                        fragmentBudget
                            .coerceAtMost(
                                routeBudget,
                            )
                            .coerceAtMost(
                                MAX_NEARBY_BYTES_PAYLOAD -
                                    NOX_HEADER_BYTES -
                                    24 -
                                    96,
                            ),
                )
                .forEachIndexed { index, chunk ->
                    val packet =
                        MultiActorPoseFragmentWireCodec.encode(
                            chunk = chunk,
                            packetSequence =
                                fragmentPacketSequence(packetSequenceStart, index),
                        )
                    check(
                        packet.size <= routeBudget,
                    ) {
                        "Multi-actor routed fragment exceeds route payload budget"
                    }
                    check(
                        sendCachedRoutedFragment(
                            routeNodes = route.nodes,
                            origin = origin,
                            destination = destination,
                            evidenceType =
                                NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                            streamId = streamId,
                            frameSequence = frameSequence,
                            chunkIndex = chunk.chunkIndex,
                            chunkCount = chunk.chunkCount,
                            packetSequence =
                                fragmentPacketSequence(packetSequenceStart, index),
                            timestampNanos =
                                chunk.timestampNanos,
                            payload = packet,
                            reconstructionCritical =
                                reconstructionCritical,
                        ),
                    )
                }
            true
        }.getOrDefault(false)
    }

    fun sendRoutedFaceMeshFrame(
        route: DistributedTransportCoordinator.Route,
        origin: DeviceId,
        destination: DeviceId,
        frame: com.nivukx.nox.core.model.FaceMeshFrame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
    ): Boolean {
        val routeBudget =
            DistributedTransportEnvelopeCodec
                .maxPayloadBytesForRoute(
                    origin = origin,
                    destination = destination,
                    route = route.nodes,
                )

        val streamBytes =
            streamId.toByteArray(Charsets.UTF_8).size
        val fragmentBudget =
            (
                routeBudget -
                    36 -
                    24 -
                    streamBytes
            ).coerceAtLeast(4_096)

        return runCatching {
            FaceMeshChunkedWireCodec
                .encode(
                    frame = frame,
                    streamId = streamId,
                    frameSequence = frameSequence,
                    maximumPayloadBytes =
                        fragmentBudget
                            .coerceAtMost(
                                45 * 1024,
                            )
                            .coerceAtMost(
                                MAX_NEARBY_BYTES_PAYLOAD -
                                    NOX_HEADER_BYTES -
                                    24 -
                                    96,
                            ),
                )
                .forEachIndexed { index, chunk ->
                    val packet =
                        FaceMeshFragmentWireCodec.encode(
                            chunk = chunk,
                            packetSequence =
                                fragmentPacketSequence(packetSequenceStart, index),
                        )
                    check(packet.size <= routeBudget) {
                        "Face mesh routed fragment exceeds route payload budget"
                    }
                    check(
                        sendCachedRoutedFragment(
                            routeNodes = route.nodes,
                            origin = origin,
                            destination = destination,
                            evidenceType =
                                NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                            streamId = streamId,
                            frameSequence = frameSequence,
                            chunkIndex = chunk.chunkIndex,
                            chunkCount = chunk.chunkCount,
                            packetSequence =
                                fragmentPacketSequence(packetSequenceStart, index),
                            timestampNanos =
                                chunk.timestampNanos,
                            payload = packet,
                            reconstructionCritical =
                                reconstructionCritical,
                        ),
                    )
                }
            true
        }.getOrDefault(false)
    }

    fun sendFaceMeshFrame(
        endpoint: DeviceId,
        frame: com.nivukx.nox.core.model.FaceMeshFrame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
    ) {
        require(packetSequenceStart >= 0L)

        FaceMeshChunkedWireCodec
            .encode(
                frame = frame,
                streamId = streamId,
                frameSequence = frameSequence,
                wireVersion = wireVersion,
                maximumPayloadBytes =
                    MAX_NEARBY_BYTES_PAYLOAD -
                        NOX_HEADER_BYTES -
                        24 -
                        96,
            )
            .forEachIndexed { index, chunk ->
                check(
                    sendCachedFragment(
                        endpoint = endpoint,
                        evidenceType =
                            NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                        streamId = streamId,
                        frameSequence = frameSequence,
                        chunkIndex = chunk.chunkIndex,
                        chunkCount = chunk.chunkCount,
                        packet =
                            FaceMeshFragmentWireCodec.encode(
                                chunk = chunk,
                                packetSequence =
                                    fragmentPacketSequence(packetSequenceStart, index),
                            ),
                        reconstructionCritical =
                            reconstructionCritical,
                    ),
                )
            }
    }

    fun sendMultiActorPoseFrame(
        endpoint: DeviceId,
        frame: MultiActorPoseWireCodec.Frame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
    ) {
        require(packetSequenceStart >= 0L)

        MultiActorPoseChunkedWireCodec
            .encode(
                frame = frame,
                streamId = streamId,
                frameSequence = frameSequence,
                maximumPayloadBytes =
                    MAX_NEARBY_BYTES_PAYLOAD -
                        NOX_HEADER_BYTES -
                        24 -
                        96,
            )
            .forEachIndexed { index, chunk ->
                require(
                    packetSequenceStart <=
                        Long.MAX_VALUE -
                            index.toLong(),
                ) {
                    "Multi-actor fragment packet sequence overflow"
                }
                check(
                    sendCachedFragment(
                        endpoint = endpoint,
                        evidenceType =
                            NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                        streamId = streamId,
                        frameSequence = frameSequence,
                        chunkIndex = chunk.chunkIndex,
                        chunkCount = chunk.chunkCount,
                        packet =
                            MultiActorPoseFragmentWireCodec.encode(
                                chunk = chunk,
                                packetSequence =
                                    fragmentPacketSequence(packetSequenceStart, index),
                            ),
                        reconstructionCritical =
                            reconstructionCritical,
                    ),
                ) {
                    "Failed to queue multi-actor pose fragment"
                }
            }
    }

    fun sendRoutedTimingBeaconObservation(
        route: DistributedTransportCoordinator.Route,
        origin: DeviceId,
        destination: DeviceId,
        observation: TimingBeaconObservationWireCodec.Observation,
        packetSequence: Long,
    ): Boolean {
        val inner =
            TimingBeaconObservationWireCodec.encode(
                observation = observation,
                packetSequence = packetSequence,
            )

        return sendRouted(
            route = route,
            origin = origin,
            destination = destination,
            packetSequence = packetSequence,
            timestampNanos =
                observation.deviceTimestampNanos,
            payload = inner,
            reconstructionCritical = true,
        )
    }

    fun sendTimingBeaconObservation(
        endpoint: DeviceId,
        observation:
            TimingBeaconObservationWireCodec.Observation,
        packetSequence: Long,
    ): Boolean =
        send(
            endpoint = endpoint,
            packet =
                TimingBeaconObservationWireCodec.encode(
                    observation = observation,
                    packetSequence = packetSequence,
                ),
        )

    fun sendDepthFrame(
        endpoint: DeviceId,
        frame: com.nivukx.nox.core.model.DepthFrame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
    ) {
        require(packetSequenceStart >= 0L)

        val sourceDeviceId =
            localDeviceId
                ?: error(
                    "Depth frame requires a local source identity",
                )

        val payload =
            DepthWireCodec.encode(
                deviceId = sourceDeviceId,
                frame = frame,
            )
                ?: error(
                    "Depth payload exceeds the bounded metric-depth envelope",
                )

        val peer =
            peerCapabilities(endpoint)

        if (!peer.supportsDepthFragmentation) {
            val rawPacket =
                runCatching {
                    NoXWireProtocol.encode(
                        type =
                            NoXWireProtocol.Type.DEPTH_FRAME,
                        sequence = packetSequenceStart,
                        timestampNanos = frame.timestampNanos,
                        payload = payload,
                    )
                }.getOrNull()
                    ?: error(
                        "Unable to construct legacy metric-depth packet",
                    )

            check(
                rawPacket.size <=
                    MAX_NEARBY_BYTES_PAYLOAD,
            ) {
                "Peer lacks depth fragmentation and the depth frame exceeds the legacy packet budget"
            }

            check(
                send(
                    endpoint = endpoint,
                    packet = rawPacket,
                    reconstructionCritical =
                        reconstructionCritical,
                ),
            ) {
                "Failed to send legacy metric-depth packet"
            }
            return
        }

        DepthChunkedWireCodec
            .encode(
                deviceId = sourceDeviceId,
                payload = payload,
                streamId = streamId,
                frameSequence = frameSequence,
                maximumPayloadBytes =
                    MAX_NEARBY_BYTES_PAYLOAD -
                        NOX_HEADER_BYTES -
                        MAX_DEPTH_FRAGMENT_HEADER_BYTES,
            )
            .forEachIndexed { index, chunk ->
                val packet =
                    DepthFragmentWireCodec.encode(
                        chunk = chunk,
                        packetSequence =
                            fragmentPacketSequence(
                                packetSequenceStart,
                                index,
                            ),
                    )
                check(
                    sendCachedFragment(
                        endpoint = endpoint,
                        evidenceType =
                            NoXWireProtocol.Type.DEPTH_FRAGMENT,
                        streamId = streamId,
                        frameSequence = frameSequence,
                        chunkIndex = chunk.chunkIndex,
                        chunkCount = chunk.chunkCount,
                        packet = packet,
                        reconstructionCritical =
                            reconstructionCritical,
                    ),
                ) {
                    "Failed to queue depth fragment"
                }
            }
    }

    fun sendDenseLandmarkFrame(
        endpoint: DeviceId,
        frame: DenseLandmarkWireCodec.Frame,
        streamId: String,
        frameSequence: Long,
        packetSequenceStart: Long,
        reconstructionCritical: Boolean = false,
        wireVersion: DenseLandmarkChunkedWireCodec.FrameWireVersion =
            DenseLandmarkChunkedWireCodec.FrameWireVersion.V4_TIMING_METADATA,
    ) {
        require(packetSequenceStart >= 0L)

        DenseLandmarkChunkedWireCodec
            .encode(
                frame = frame,
                streamId = streamId,
                frameSequence = frameSequence,
                maximumPayloadBytes =
                    MAX_NEARBY_BYTES_PAYLOAD -
                        NOX_HEADER_BYTES -
                        28 -
                        96,
            )
            .forEachIndexed { index, chunk ->
                require(
                    packetSequenceStart <=
                        Long.MAX_VALUE -
                            index.toLong(),
                ) {
                    "Dense landmark fragment packet sequence overflow"
                }
                check(
                    sendCachedFragment(
                        endpoint = endpoint,
                        evidenceType =
                            NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                        streamId = streamId,
                        frameSequence = frameSequence,
                        chunkIndex = chunk.chunkIndex,
                        chunkCount = chunk.chunkCount,
                        packet =
                            DenseLandmarkFragmentWireCodec.encode(
                                chunk = chunk,
                                packetSequence =
                                    fragmentPacketSequence(packetSequenceStart, index),
                            ),
                        reconstructionCritical =
                            reconstructionCritical,
                    ),
                ) {
                    "Failed to queue dense landmark fragment"
                }
            }
    }

    fun send(
        endpoint: DeviceId,
        packet: ByteArray,
        reconstructionCritical: Boolean = false,
    ): Boolean {
        if (
            !peers.containsKey(endpoint.value) ||
            packet.size > MAX_NEARBY_BYTES_PAYLOAD
        ) {
            return false
        }

        val packetType =
            packetTypeOf(packet)

        if (
            transportSheddingPolicy.shouldShed(
                type = packetType,
                packetBytes = packet.size,
                outstandingPayloads =
                    peerTransportQuality
                        .outstandingPayloadCount(
                            endpoint.value,
                        ),
                reconstructionCritical =
                    reconstructionCritical,
            )
        ) {
            return false
        }

        val nowNanos = System.nanoTime()
        val measuredThroughputMbps =
            peerTransportQuality
                .estimatedPayloadThroughputMbps(
                    endpoint.value,
                )

        if (
            !transportCongestionController.tryAcquire(
                endpoint = endpoint.value,
                type = packetType,
                packetBytes = packet.size,
                measuredThroughputMbps =
                    measuredThroughputMbps,
                reconstructionCritical =
                    reconstructionCritical,
                nowNanos = nowNanos,
            )
        ) {
            return false
        }

        val payload =
            Payload.fromBytes(packet)

        peerTransportQuality.observePayloadStarted(
            endpoint = endpoint.value,
            payloadId = payload.id,
            totalBytes = packet.size.toLong(),
            startedAtNanos = nowNanos,
        )

        return runCatching {
            client.sendPayload(
                endpoint.value,
                payload,
            )
            true
        }.getOrElse {
            peerTransportQuality.observePayloadProgress(
                endpoint = endpoint.value,
                payloadId = payload.id,
                bytesTransferred = 0L,
                totalBytes = packet.size.toLong(),
                status =
                    PayloadTransferUpdate.Status.FAILURE,
                observedAtNanos = System.nanoTime(),
            )
            transportCongestionController.refund(
                endpoint = endpoint.value,
                type = packetType,
                packetBytes = packet.size,
                reconstructionCritical =
                    reconstructionCritical,
            )
            false
        }
    }

    fun broadcast(packet: ByteArray) {
        if (packet.size > MAX_NEARBY_BYTES_PAYLOAD) return

        peers.values
            .sortedBy { it.value }
            .forEach { endpoint ->
                send(
                    endpoint = endpoint,
                    packet = packet,
                )
            }
    }

    fun stop() {
        client.stopAdvertising()
        client.stopDiscovery()
        client.stopAllEndpoints()
        peers.clear()
        connectingEndpoints.clear()
        peerCapacityPolicy.clear()
        replayGuard.clearAll()
        denseLandmarkReassembler.clear()
        multiActorPoseReassembler.clear()
        faceMeshReassembler.clear()
        fragmentRetransmissionCache.clear()
        fragmentReplyRouteRegistry.clear()
        peerTransportQuality.clear()
        transportCongestionController.reset()
        peerAdmissionPolicy.clearPendingRequests()
        _pendingPeerRequests.value = emptyList()
        publishPeers()
    }

    private fun processPeerCapabilities(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
    ) {
        val capabilities =
            if (packet.payload.isEmpty()) {
                PeerCapabilitiesWireCodec.legacyFallback()
            } else {
                PeerCapabilitiesWireCodec.decode(
                    packet.payload,
                ) ?: return
            }

        peerCapabilities[endpointId] =
            capabilities
    }

    private fun respondToPing(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
    ) {
        if (!peers.containsKey(endpointId)) return

        send(
            endpoint = DeviceId(endpointId),
            packet =
                NoXWireProtocol.encode(
                    type = NoXWireProtocol.Type.PONG,
                    sequence = packet.sequence,
                    timestampNanos = System.nanoTime(),
                ),
        )
    }

    private fun dispatchPacket(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
        emitRawPacket: Boolean = false,
        replyRoute: List<DeviceId>? = null,
    ) {
        when (packet.type) {
            NoXWireProtocol.Type.HELLO ->
                processPeerCapabilities(
                    endpointId = endpointId,
                    packet = packet,
                )

            NoXWireProtocol.Type.PING ->
                respondToPing(
                    endpointId = endpointId,
                    packet = packet,
                )

            NoXWireProtocol.Type.PONG ->
                peerTransportQuality.observePong(
                    endpoint = endpointId,
                    sequence = packet.sequence,
                    receivedAtNanos = System.nanoTime(),
                )

            NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT ->
                processDenseLandmarkFragment(
                    endpointId = endpointId,
                    packet = packet,
                    replyRoute = replyRoute,
                )

            NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT ->
                processMultiActorPoseFragment(
                    endpointId = endpointId,
                    packet = packet,
                    replyRoute = replyRoute,
                )

            NoXWireProtocol.Type.FACE_MESH_FRAGMENT ->
                processFaceMeshFragment(
                    endpointId = endpointId,
                    packet = packet,
                    replyRoute = replyRoute,
                )

            NoXWireProtocol.Type.DEPTH_FRAGMENT ->
                processDepthFragment(
                    endpointId = endpointId,
                    packet = packet,
                    replyRoute = replyRoute,
                )

            NoXWireProtocol.Type.TIMING_BEACON_OBSERVATION ->
                processTimingBeaconObservation(
                    endpointId = endpointId,
                    packet = packet,
                    replyRoute = replyRoute,
                )

            NoXWireProtocol.Type.EVIDENCE_FRAGMENT_FEEDBACK ->
                processFragmentFeedback(
                    endpointId = endpointId,
                    packet = packet,
                    replyRoute = replyRoute,
                )

            NoXWireProtocol.Type.ROUTED_EVIDENCE_FRAME ->
                processRoutedEvidence(
                    endpointId = endpointId,
                    packet = packet,
                )

            else -> Unit
        }

        if (emitRawPacket) {
            _packets.tryEmit(
                PeerPacket(
                    endpoint = DeviceId(endpointId),
                    payload = packet.payload,
                ),
            )
        }
    }

    private fun processRoutedEvidence(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
    ) {
        val envelope =
            runCatching {
                DistributedTransportEnvelopeCodec.decode(packet)
            }.getOrNull()
                ?: return

        val local =
            localDeviceId
                ?: return

        if (local == envelope.destination) {
            val innerPacket =
                runCatching {
                    NoXWireProtocol.decode(
                        envelope.payload,
                    )
                }.getOrNull()
                    ?: return

            val replyRoute =
                if (
                    innerPacket.type ==
                        NoXWireProtocol.Type.EVIDENCE_FRAGMENT_FEEDBACK
                ) {
                    envelope.route
                        .asReversed()
                        .drop(1)
                        .plus(envelope.origin)
                } else {
                    envelope.route
                        .asReversed()
                        .drop(1)
                        .plus(envelope.origin)
                }

            rememberFragmentReplyRoute(
                endpoint = envelope.origin.value,
                packet = innerPacket,
                route = replyRoute,
                nowNanos = System.nanoTime(),
            )

            dispatchPacket(
                endpointId =
                    envelope.origin.value,
                packet = innerPacket,
                emitRawPacket = true,
                replyRoute = replyRoute,
            )
            return
        }

        if (
            !DistributedTransportEnvelopeCodec.canAdvance(
                envelope = envelope,
                localDeviceId = local,
            )
        ) {
            return
        }

        val advanced =
            runCatching {
                DistributedTransportEnvelopeCodec.advance(
                    envelope = envelope,
                    localDeviceId = local,
                )
            }.getOrNull()
                ?: return

        val nextHop =
            advanced.route[
                advanced.hopIndex
            ]

        val forwarded =
            DistributedTransportEnvelopeCodec.encode(
                origin = advanced.origin,
                destination = advanced.destination,
                route = advanced.route,
                hopIndex = advanced.hopIndex,
                ttl = advanced.ttl,
                packetSequence =
                    advanced.packetSequence,
                timestampNanos =
                    advanced.timestampNanos,
                payload = advanced.payload,
                reconstructionCritical =
                    advanced.reconstructionCritical,
            )

        send(
            endpoint = nextHop,
            packet = forwarded,
            reconstructionCritical =
                advanced.reconstructionCritical,
        )
    }

    private fun processFaceMeshFragment(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
        replyRoute: List<DeviceId>? = null,
    ) {
        val chunk =
            runCatching {
                FaceMeshFragmentWireCodec.decode(packet)
            }.getOrNull()
                ?: return

        val frame =
            runCatching {
                faceMeshReassembler.accept(
                    endpoint = endpointId,
                    chunk = chunk,
                    nowNanos = System.nanoTime(),
                )
            }.getOrNull()
                ?: return

        if (frame != null) {
            _faceMeshFrames.tryEmit(
                FaceMeshFrame(
                    endpoint = DeviceId(endpointId),
                    frame = frame,
                ),
            )
            emitAck(
                endpointId = endpointId,
                evidenceType =
                    NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                streamId = chunk.streamId,
                frameSequence = chunk.frameSequence,
                chunkCount = chunk.chunkCount,
                replyRoute = replyRoute,
            )
        }
    }

    private fun processMultiActorPoseFragment(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
        replyRoute: List<DeviceId>? = null,
    ) {
        val chunk =
            runCatching {
                MultiActorPoseFragmentWireCodec.decode(packet)
            }.getOrNull()
                ?: return

        val frame =
            runCatching {
                multiActorPoseReassembler.accept(
                    endpoint = endpointId,
                    chunk = chunk,
                    nowNanos = System.nanoTime(),
                )
            }.getOrNull()
                ?: return

        if (frame != null) {
            _multiActorPoseFrames.tryEmit(
                MultiActorPoseFrame(
                    endpoint = DeviceId(endpointId),
                    frame = frame,
                ),
            )
            emitAck(
                endpointId = endpointId,
                evidenceType =
                    NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                streamId = chunk.streamId,
                frameSequence = chunk.frameSequence,
                chunkCount = chunk.chunkCount,
                replyRoute = replyRoute,
            )
        }
    }

    private fun processDepthFragment(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
        replyRoute: List<DeviceId>? = null,
    ) {
        val chunk =
            runCatching {
                DepthFragmentWireCodec.decode(packet)
            }.getOrNull()
                ?: return

        val payload =
            runCatching {
                depthFragmentReassembler.accept(
                    endpoint = endpointId,
                    chunk = chunk,
                    nowNanos = System.nanoTime(),
                )
            }.getOrNull()
                ?: return

        if (payload != null) {
            val frame =
                runCatching {
                    DepthWireCodec.decode(payload)
                }.getOrNull()
                    ?: return

            if (frame.deviceId != chunk.deviceId) {
                return
            }

            _depthFrames.tryEmit(
                DepthFrame(
                    endpoint = DeviceId(endpointId),
                    frame = frame,
                ),
            )

            emitAck(
                endpointId = endpointId,
                evidenceType =
                    NoXWireProtocol.Type.DEPTH_FRAGMENT,
                streamId = chunk.streamId,
                frameSequence = chunk.frameSequence,
                chunkCount = chunk.chunkCount,
                replyRoute = replyRoute,
            )
        }
    }

    private fun processTimingBeaconObservation(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
    ) {
        val observation =
            runCatching {
                TimingBeaconObservationWireCodec.decode(
                    packet,
                )
            }.getOrNull()
                ?: return

        _timingBeaconObservations.tryEmit(
            TimingBeaconObservation(
                endpoint = DeviceId(endpointId),
                observation = observation,
            ),
        )
    }

    private fun processDenseLandmarkFragment(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
        replyRoute: List<DeviceId>? = null,
    ) {
        val chunk =
            runCatching {
                DenseLandmarkFragmentWireCodec.decode(packet)
            }.getOrNull()
                ?: return

        val frame =
            runCatching {
                denseLandmarkReassembler.accept(
                    endpoint = endpointId,
                    chunk = chunk,
                    nowNanos = System.nanoTime(),
                )
            }.getOrNull()
                ?: return

        if (frame != null) {
            _denseLandmarkFrames.tryEmit(
                DenseLandmarkFrame(
                    endpoint = DeviceId(endpointId),
                    frame = frame,
                ),
            )
            emitAck(
                endpointId = endpointId,
                evidenceType =
                    NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                streamId = chunk.streamId,
                frameSequence = chunk.frameSequence,
                chunkCount = chunk.chunkCount,
                replyRoute = replyRoute,
            )
        }
    }

    private fun fragmentPacketSequence(
        packetSequenceStart: Long,
        index: Int,
    ): Long {
        require(packetSequenceStart >= 0L)
        require(index >= 0)
        val offset = index.toLong()
        require(
            packetSequenceStart <=
                Long.MAX_VALUE - offset,
        ) {
            "Fragment packet sequence overflow"
        }
        return packetSequenceStart + offset
    }

    private fun packetTypeOf(
        packet: ByteArray,
    ): NoXWireProtocol.Type =
        runCatching {
            require(packet.size >= 12)
            java.nio.ByteBuffer
                .wrap(packet)
                .order(java.nio.ByteOrder.BIG_ENDIAN)
                .apply {
                    position(8)
                }
                .int
                .let { code ->
                    NoXWireProtocol.Type.entries
                        .firstOrNull { it.code == code }
                        ?: NoXWireProtocol.Type.ERROR
                }
        }.getOrDefault(
            NoXWireProtocol.Type.ERROR,
        )

    private fun sendCachedFragment(
        endpoint: DeviceId,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
        chunkIndex: Int,
        chunkCount: Int,
        packet: ByteArray,
        reconstructionCritical: Boolean,
    ): Boolean {
        val sent =
            send(
                endpoint = endpoint,
                packet = packet,
                reconstructionCritical =
                    reconstructionCritical,
            )

        if (sent) {
            fragmentRetransmissionCache.remember(
                endpoint = endpoint.value,
                evidenceType = evidenceType,
                streamId = streamId,
                frameSequence = frameSequence,
                chunkIndex = chunkIndex,
                chunkCount = chunkCount,
                packet = packet,
                reconstructionCritical =
                    reconstructionCritical,
                nowNanos = System.nanoTime(),
            )
        }

        return sent
    }

    private fun sendFragmentFeedback(
        endpoint: String,
        feedback:
            EvidenceFragmentFeedbackWireCodec.Feedback,
        replyRoute: List<DeviceId>? = null,
    ) {
        val packet =
            EvidenceFragmentFeedbackWireCodec.encode(
                feedback = feedback,
                packetSequence =
                    controlSequence.getAndIncrement(),
                timestampNanos =
                    System.nanoTime(),
            )

        val route =
            replyRoute
                ?.takeIf { it.isNotEmpty() }

        if (route != null) {
            sendRoutedAlongNodes(
                routeNodes = route,
                origin =
                    localDeviceId
                        ?: return,
                destination = route.last(),
                packetSequence =
                    NoXWireProtocol.decode(packet).sequence,
                timestampNanos =
                    feedbackTimestamp(packet),
                payload = packet,
                reconstructionCritical = true,
            )
        } else {
            send(
                endpoint = DeviceId(endpoint),
                packet = packet,
                reconstructionCritical = true,
            )
        }
    }

    private fun feedbackTimestamp(
        packet: ByteArray,
    ): Long =
        NoXWireProtocol.decode(packet).timestampNanos

    private fun emitAck(
        endpointId: String,
        evidenceType: NoXWireProtocol.Type,
        streamId: String,
        frameSequence: Long,
        chunkCount: Int,
        replyRoute: List<DeviceId>? = null,
    ) {
        sendFragmentFeedback(
            endpoint = endpointId,
            feedback =
                EvidenceFragmentFeedbackWireCodec.Feedback(
                    direction =
                        EvidenceFragmentFeedbackWireCodec.Direction.ACK,
                    evidenceType = evidenceType,
                    streamId = streamId,
                    frameSequence = frameSequence,
                    chunkCount = chunkCount,
                    receivedMask =
                        (1 shl chunkCount) - 1,
                ),
            replyRoute = replyRoute,
        )
    }

    private fun emitExpiredFragmentNacks(
        nowNanos: Long,
    ) {
        denseLandmarkReassembler
            .expireWithFeedback(nowNanos)
            .forEach { state ->
                sendFragmentFeedback(
                    endpoint = state.endpoint,
                    feedback =
                        EvidenceFragmentFeedbackWireCodec.Feedback(
                            direction =
                                EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                            evidenceType =
                                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            chunkCount = state.chunkCount,
                            receivedMask = state.receivedMask,
                        ),
                    replyRoute =
                        fragmentReplyRouteRegistry.find(
                            endpoint = state.endpoint,
                            evidenceType =
                                NoXWireProtocol.Type.DENSE_LANDMARK_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            nowNanos = nowNanos,
                        ),
                )
            }

        multiActorPoseReassembler
            .expireWithFeedback(nowNanos)
            .forEach { state ->
                sendFragmentFeedback(
                    endpoint = state.endpoint,
                    feedback =
                        EvidenceFragmentFeedbackWireCodec.Feedback(
                            direction =
                                EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                            evidenceType =
                                NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            chunkCount = state.chunkCount,
                            receivedMask = state.receivedMask,
                        ),
                    replyRoute =
                        fragmentReplyRouteRegistry.find(
                            endpoint = state.endpoint,
                            evidenceType =
                                NoXWireProtocol.Type.MULTI_ACTOR_POSE_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            nowNanos = nowNanos,
                        ),
                )
            }

        depthFragmentReassembler
            .expireWithFeedback(nowNanos)
            .forEach { state ->
                sendFragmentFeedback(
                    endpoint = state.endpoint,
                    feedback =
                        EvidenceFragmentFeedbackWireCodec.Feedback(
                            direction =
                                EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                            evidenceType =
                                NoXWireProtocol.Type.DEPTH_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            chunkCount = state.chunkCount,
                            receivedMask = state.receivedMask,
                        ),
                    replyRoute =
                        fragmentReplyRouteRegistry.find(
                            endpoint = state.endpoint,
                            evidenceType =
                                NoXWireProtocol.Type.DEPTH_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            nowNanos = nowNanos,
                        ),
                )
            }

        faceMeshReassembler
            .expireWithFeedback(nowNanos)
            .forEach { state ->
                sendFragmentFeedback(
                    endpoint = state.endpoint,
                    feedback =
                        EvidenceFragmentFeedbackWireCodec.Feedback(
                            direction =
                                EvidenceFragmentFeedbackWireCodec.Direction.NACK,
                            evidenceType =
                                NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            chunkCount = state.chunkCount,
                            receivedMask = state.receivedMask,
                        ),
                    replyRoute =
                        fragmentReplyRouteRegistry.find(
                            endpoint = state.endpoint,
                            evidenceType =
                                NoXWireProtocol.Type.FACE_MESH_FRAGMENT,
                            streamId = state.streamId,
                            frameSequence = state.frameSequence,
                            nowNanos = nowNanos,
                        ),
                )
            }

        fragmentRetransmissionCache.expire(nowNanos)
    }

    private fun processFragmentFeedback(
        endpointId: String,
        packet: NoXWireProtocol.Packet,
        replyRoute: List<DeviceId>? = null,
    ) {
        val feedback =
            runCatching {
                EvidenceFragmentFeedbackWireCodec.decode(
                    packet,
                )
            }.getOrNull()
                ?: return

        fragmentRetransmissionCache
            .handle(
                endpoint = endpointId,
                feedback = feedback,
                nowNanos = System.nanoTime(),
                nextPacketSequence = {
                    controlSequence.getAndIncrement()
                },
            )
            .forEach { retransmission ->
                if (
                    replyRoute != null &&
                        replyRoute.isNotEmpty()
                ) {
                    sendRoutedAlongNodes(
                        routeNodes = replyRoute,
                        origin =
                            localDeviceId
                                ?: return@forEach,
                        destination =
                            replyRoute.last(),
                        packetSequence =
                            NoXWireProtocol.decode(
                                retransmission.packet,
                            ).sequence,
                        timestampNanos =
                            NoXWireProtocol.decode(
                                retransmission.packet,
                            ).timestampNanos,
                        payload =
                            retransmission.packet,
                        reconstructionCritical =
                            retransmission.reconstructionCritical,
                    )
                } else {
                    send(
                        endpoint = DeviceId(endpointId),
                        packet = retransmission.packet,
                        reconstructionCritical =
                            retransmission.reconstructionCritical,
                    )
                }
            }
    }

    private fun publishPeers() {
        _connected.value = peers.values.toSet()
    }
}
