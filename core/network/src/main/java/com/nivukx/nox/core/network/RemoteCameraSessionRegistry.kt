package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceCapabilities
import com.nivukx.nox.core.model.DeviceId
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.max

/**
 * Bounded registry for remote camera session identities.
 *
 * Camera identity and network endpoint are deliberately separate. Reconnection may change an
 * endpoint without creating a new camera identity, while an endpoint may not silently claim
 * another active camera.
 */
class RemoteCameraSessionRegistry(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumCameras: Int = 128,
        val minimumTimingAuthorityScore: Double = 0.0,
        val maximumHeartbeatHistory: Int = 32,
    ) {
        init {
            require(maximumCameras in 1..128)
            require(
                minimumTimingAuthorityScore.isFinite() &&
                    minimumTimingAuthorityScore in 0.0..1.0,
            )
            require(maximumHeartbeatHistory in 1..256)
        }
    }

    data class CameraNode(
        val cameraId: DeviceId,
        val endpoint: UdpRemoteEvidenceTransport.Endpoint,
        val capabilities: DeviceCapabilities,
        val timingAuthorityScore: Double = 0.0,
        val lastHeartbeatNanos: Long = 0L,
        val generation: Long = 0L,
        val tags: Set<String> = emptySet(),
        /** Latest remote packet-loss estimate as a ratio; null means telemetry is unavailable. */
        val packetLossRatio: Double? = null,
        /** Latest remote link latency estimate; null means telemetry is unavailable. */
        val latencyMillis: Double? = null,
        /** Requested evidence bandwidth; null selects the coordinator's authority-based default. */
        val requestedKbps: Double? = null,
    ) {
        init {
            require(cameraId.value.isNotBlank())
            require(
                timingAuthorityScore.isFinite() &&
                    timingAuthorityScore in 0.0..1.0,
            )
            require(lastHeartbeatNanos >= 0L)
            require(generation >= 0L)
            require(tags.all { it.isNotBlank() })
            require(
                packetLossRatio == null ||
                    (packetLossRatio.isFinite() && packetLossRatio in 0.0..1.0),
            )
            require(
                latencyMillis == null ||
                    (latencyMillis.isFinite() && latencyMillis >= 0.0),
            )
            require(
                requestedKbps == null ||
                    (requestedKbps.isFinite() && requestedKbps >= 0.0),
            )
        }
    }

    data class Snapshot(
        val cameraCount: Int,
        val healthyCameraCount: Int,
        val averageTimingAuthorityScore: Double,
        val endpointCollisionCount: Long,
        val duplicateIdentityCount: Long,
        val generation: Long,
        val cameras: List<CameraNode>,
    )

    private val lock = ReentrantReadWriteLock(true)
    private val nodes = LinkedHashMap<DeviceId, CameraNode>()
    private val endpointToCamera =
        LinkedHashMap<UdpRemoteEvidenceTransport.Endpoint, DeviceId>()
    private val heartbeatHistory =
        LinkedHashMap<DeviceId, ArrayDeque<Long>>()
    /** Starts a grace clock when a never-seen heartbeat is first observed by expiry sweeps. */
    private val missingHeartbeatSinceNanos = LinkedHashMap<DeviceId, Long>()

    private var endpointCollisionCount = 0L
    private var duplicateIdentityCount = 0L
    private var generation = 0L
    private var closed = false

    fun register(node: CameraNode) {
        lock.write {
            requireOpenLocked()

            require(
                node.timingAuthorityScore >=
                    config.minimumTimingAuthorityScore,
            ) {
                "Remote camera timing authority is below registry admission threshold"
            }

            val existing = nodes[node.cameraId]
            val endpointOwner = endpointToCamera[node.endpoint]

            if (
                endpointOwner != null &&
                    endpointOwner != node.cameraId
            ) {
                endpointCollisionCount =
                    safeIncrement(endpointCollisionCount)
                throw IllegalStateException(
                    "Endpoint " +
                        node.endpoint +
                        " is already owned by " +
                        endpointOwner,
                )
            }

            if (existing == null) {
                if (nodes.size >= config.maximumCameras) {
                    throw IllegalStateException(
                        "Remote camera registry capacity " +
                            config.maximumCameras +
                            " reached",
                    )
                }
            } else {
                duplicateIdentityCount =
                    safeIncrement(duplicateIdentityCount)
                if (existing.endpoint != node.endpoint) {
                    endpointToCamera.remove(existing.endpoint)
                }
            }

            generation = safeIncrement(generation)

            val normalized =
                node.copy(
                    generation =
                        max(
                            node.generation,
                            generation,
                        ),
                )

            nodes[node.cameraId] = normalized
            endpointToCamera[node.endpoint] = node.cameraId
            missingHeartbeatSinceNanos.remove(node.cameraId)

            recordHeartbeat(
                cameraId = node.cameraId,
                timestampNanos = node.lastHeartbeatNanos,
            )
        }
    }

    fun heartbeat(
        cameraId: DeviceId,
        timestampNanos: Long,
    ): Boolean =
        lock.write {
            requireOpenLocked()
            require(timestampNanos >= 0L)

            val current = nodes[cameraId]
                ?: return@write false

            generation = safeIncrement(generation)
            nodes[cameraId] =
                current.copy(
                    lastHeartbeatNanos =
                        max(
                            current.lastHeartbeatNanos,
                            timestampNanos,
                        ),
                    generation = generation,
                )
            if (timestampNanos > 0L) {
                missingHeartbeatSinceNanos.remove(cameraId)
            }

            recordHeartbeat(
                cameraId = cameraId,
                timestampNanos = timestampNanos,
            )
            true
        }

    /**
     * Atomically records an observed link-quality sample with the camera heartbeat.
     *
     * This keeps transport allocation inputs current without requiring callers to re-register
     * the camera, which would incorrectly inflate duplicate-identity diagnostics.
     */
    fun recordNetworkTelemetry(
        cameraId: DeviceId,
        packetLossRatio: Double,
        latencyMillis: Double,
        requestedKbps: Double,
        timestampNanos: Long,
    ): Boolean =
        lock.write {
            requireOpenLocked()
            require(timestampNanos >= 0L)
            require(
                packetLossRatio.isFinite() &&
                    packetLossRatio in 0.0..1.0,
            ) {
                "Remote packet-loss ratio must be finite and in [0, 1]"
            }
            require(
                latencyMillis.isFinite() &&
                    latencyMillis >= 0.0,
            ) {
                "Remote link latency must be finite and non-negative"
            }
            require(
                requestedKbps.isFinite() &&
                    requestedKbps >= 0.0,
            ) {
                "Requested transport bitrate must be finite and non-negative"
            }

            val current =
                nodes[cameraId]
                    ?: return@write false
            if (timestampNanos < current.lastHeartbeatNanos) {
                return@write false
            }

            generation = safeIncrement(generation)
            nodes[cameraId] =
                current.copy(
                    lastHeartbeatNanos =
                        max(
                            current.lastHeartbeatNanos,
                            timestampNanos,
                        ),
                    generation = generation,
                    packetLossRatio = packetLossRatio,
                    latencyMillis = latencyMillis,
                    requestedKbps = requestedKbps,
                )

            if (timestampNanos > 0L) {
                missingHeartbeatSinceNanos.remove(cameraId)
            }
            recordHeartbeat(
                cameraId = cameraId,
                timestampNanos = timestampNanos,
            )
            true
        }

    fun update(node: CameraNode): Boolean =
        lock.write {
            requireOpenLocked()
            if (node.cameraId !in nodes) {
                return@write false
            }
            register(node)
            true
        }

    fun remove(cameraId: DeviceId): Boolean =
        lock.write {
            requireOpenLocked()
            val removed =
                nodes.remove(cameraId)
                    ?: return@write false

            endpointToCamera.remove(removed.endpoint)
            heartbeatHistory.remove(cameraId)
            missingHeartbeatSinceNanos.remove(cameraId)
            generation = safeIncrement(generation)
            true
        }

    fun find(cameraId: DeviceId): CameraNode? =
        lock.read {
            requireOpenLocked()
            nodes[cameraId]
        }

    fun findByEndpoint(
        endpoint: UdpRemoteEvidenceTransport.Endpoint,
    ): CameraNode? =
        lock.read {
            requireOpenLocked()
            endpointToCamera[endpoint]?.let(nodes::get)
        }

    fun all(): List<CameraNode> =
        lock.read {
            requireOpenLocked()
            nodes.values.toList()
        }

    fun size(): Int =
        lock.read {
            requireOpenLocked()
            nodes.size
        }

    fun contains(cameraId: DeviceId): Boolean =
        lock.read {
            requireOpenLocked()
            cameraId in nodes
        }

    fun expire(
        nowNanos: Long,
        maximumSilenceNanos: Long,
    ): List<DeviceId> =
        lock.write {
            requireOpenLocked()
            require(nowNanos >= 0L)
            require(maximumSilenceNanos > 0L)

            val expired =
                nodes.values
                    .asSequence()
                    .filter { node ->
                        if (node.lastHeartbeatNanos > 0L) {
                            missingHeartbeatSinceNanos.remove(node.cameraId)
                            nowNanos >= node.lastHeartbeatNanos &&
                                nowNanos - node.lastHeartbeatNanos >=
                                    maximumSilenceNanos
                        } else {
                            val firstObservedNanos =
                                missingHeartbeatSinceNanos.getOrPut(node.cameraId) {
                                    nowNanos
                                }
                            nowNanos >= firstObservedNanos &&
                                nowNanos - firstObservedNanos >= maximumSilenceNanos
                        }
                    }
                    .sortedBy { node ->
                        if (node.lastHeartbeatNanos > 0L) {
                            node.lastHeartbeatNanos
                        } else {
                            missingHeartbeatSinceNanos[node.cameraId] ?: nowNanos
                        }
                    }
                    .map { it.cameraId }
                    .toList()

            expired.forEach { cameraId ->
                val node = nodes.remove(cameraId)
                if (node != null) {
                    endpointToCamera.remove(node.endpoint)
                    heartbeatHistory.remove(cameraId)
                    missingHeartbeatSinceNanos.remove(cameraId)
                    generation = safeIncrement(generation)
                }
            }

            expired
        }

    fun snapshot(
        nowNanos: Long,
        maximumSilenceNanos: Long = 5_000_000_000L,
    ): Snapshot {
        val cameras = all()
        val healthy =
            cameras.count { node ->
                node.lastHeartbeatNanos > 0L &&
                    nowNanos >= node.lastHeartbeatNanos &&
                    nowNanos - node.lastHeartbeatNanos <
                        maximumSilenceNanos
            }

        return Snapshot(
            cameraCount = cameras.size,
            healthyCameraCount = healthy,
            averageTimingAuthorityScore =
                cameras
                    .map { it.timingAuthorityScore }
                    .averageOrZero(),
            endpointCollisionCount = endpointCollisionCount(),
            duplicateIdentityCount = duplicateIdentityCount(),
            generation = currentGeneration(),
            cameras = cameras,
        )
    }

    fun endpointOwner(
        endpoint: UdpRemoteEvidenceTransport.Endpoint,
    ): DeviceId? =
        lock.read {
            requireOpenLocked()
            endpointToCamera[endpoint]
        }

    fun heartbeatHistory(
        cameraId: DeviceId,
    ): List<Long> =
        lock.read {
            requireOpenLocked()
            heartbeatHistory[cameraId]?.toList().orEmpty()
        }

    fun endpointCollisionCount(): Long =
        lock.read {
            requireOpenLocked()
            endpointCollisionCount
        }

    fun duplicateIdentityCount(): Long =
        lock.read {
            requireOpenLocked()
            duplicateIdentityCount
        }

    fun currentGeneration(): Long =
        lock.read {
            requireOpenLocked()
            generation
        }

    fun clear() {
        lock.write {
            requireOpenLocked()
            nodes.clear()
            endpointToCamera.clear()
            heartbeatHistory.clear()
            missingHeartbeatSinceNanos.clear()
            endpointCollisionCount = 0L
            duplicateIdentityCount = 0L
            generation = safeIncrement(generation)
        }
    }

    fun close() {
        lock.write {
            if (closed) return
            nodes.clear()
            endpointToCamera.clear()
            heartbeatHistory.clear()
            missingHeartbeatSinceNanos.clear()
            closed = true
        }
    }

    private fun recordHeartbeat(
        cameraId: DeviceId,
        timestampNanos: Long,
    ) {
        if (timestampNanos <= 0L) return

        val history =
            heartbeatHistory.getOrPut(cameraId) {
                ArrayDeque()
            }

        if (history.lastOrNull() != timestampNanos) {
            history.addLast(timestampNanos)
        }

        while (
            history.size >
                config.maximumHeartbeatHistory
        ) {
            history.removeFirst()
        }
    }

    private fun requireOpenLocked() {
        check(!closed) {
            "Remote camera session registry is closed"
        }
    }

    private fun safeIncrement(value: Long): Long =
        if (value == Long.MAX_VALUE) {
            Long.MAX_VALUE
        } else {
            value + 1L
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
