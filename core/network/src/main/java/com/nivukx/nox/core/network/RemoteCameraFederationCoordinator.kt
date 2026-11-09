package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

/**
 * Production coordinator for dynamic 128-camera remote capture federations.
 *
 * It composes the remote identity registry with the existing transport budget engine. The
 * coordinator owns no image data, so camera admission/retirement remains cheap even when a rig
 * contains 50 active subjects and 100+ camera nodes.
 */
class RemoteCameraFederationCoordinator(
    private val config: Config = Config(),
    private val registry:
        RemoteCameraSessionRegistry =
        RemoteCameraSessionRegistry(
            RemoteCameraSessionRegistry.Config(
                maximumCameras = config.maximumCameras,
            ),
        ),
    private val transportBudget:
        FrontierCameraTransportBudgetEngine =
        FrontierCameraTransportBudgetEngine(
            FrontierCameraTransportBudgetEngine.Config(
                maximumCameras = config.maximumCameras,
                authoritativeCameras =
                    config.maximumAuthoritativeCameras,
                targetActors = config.targetActors,
            ),
        ),
) : AutoCloseable {

    data class Config(
        val maximumCameras: Int = 128,
        val maximumAuthoritativeCameras: Int = 32,
        val targetActors: Int = 50,
        val hardActorCeiling: Int = 64,
        val maximumSilenceNanos: Long = 5_000_000_000L,
        val unknownPacketLossEstimate: Double = 0.05,
        val unknownLatencyEstimateMillis: Double = 75.0,
        val defaultAuthoritativeRequestedKbps: Double = 1_800.0,
        val defaultWitnessRequestedKbps: Double = 180.0,
    ) {
        init {
            require(maximumCameras in 2..128)
            require(
                maximumAuthoritativeCameras in
                    2..maximumCameras,
            )
            require(targetActors in 1..hardActorCeiling)
            require(hardActorCeiling in 1..64)
            require(maximumSilenceNanos > 0L)
            require(
                unknownPacketLossEstimate.isFinite() &&
                    unknownPacketLossEstimate in 0.0..1.0,
            )
            require(
                unknownLatencyEstimateMillis.isFinite() &&
                    unknownLatencyEstimateMillis >= 0.0,
            )
            require(
                defaultAuthoritativeRequestedKbps.isFinite() &&
                    defaultAuthoritativeRequestedKbps > 0.0,
            )
            require(
                defaultWitnessRequestedKbps.isFinite() &&
                    defaultWitnessRequestedKbps > 0.0,
            )
        }
    }

    data class FederationSnapshot(
        val registry: RemoteCameraSessionRegistry.Snapshot,
        val transport: FrontierCameraTransportBudgetEngine.Result?,
        val retiredCameraIds: List<DeviceId>,
    )

    private val lock = ReentrantReadWriteLock(true)
    private var lastSnapshot =
        FederationSnapshot(
            registry = registry.snapshot(0L),
            transport = null,
            retiredCameraIds = emptyList(),
        )
    private var closed = false

    fun admit(
        node: RemoteCameraSessionRegistry.CameraNode,
    ) {
        lock.write {
            requireOpenLocked()
            registry.register(node)
            lastSnapshot =
                lastSnapshot.copy(
                    registry = registry.snapshot(
                        nowNanos = node.lastHeartbeatNanos,
                        maximumSilenceNanos =
                            config.maximumSilenceNanos,
                    ),
                    retiredCameraIds = emptyList(),
                )
        }
    }

    fun heartbeat(
        cameraId: DeviceId,
        timestampNanos: Long,
    ): Boolean =
        lock.write {
            requireOpenLocked()
            val accepted =
                registry.heartbeat(
                    cameraId = cameraId,
                    timestampNanos = timestampNanos,
                )
            if (accepted) {
                lastSnapshot =
                    lastSnapshot.copy(
                        registry = registry.snapshot(
                            nowNanos = timestampNanos,
                            maximumSilenceNanos =
                                config.maximumSilenceNanos,
                        ),
                        retiredCameraIds = emptyList(),
                    )
            }
            accepted
        }

    /**
     * Publishes the latest observed network quality for a remote camera.
     *
     * The previous transport allocation is invalidated because its queue/rate decisions were
     * based on older link measurements. Callers must allocate again before treating a transport
     * budget as current.
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
            val accepted =
                registry.recordNetworkTelemetry(
                    cameraId = cameraId,
                    packetLossRatio = packetLossRatio,
                    latencyMillis = latencyMillis,
                    requestedKbps = requestedKbps,
                    timestampNanos = timestampNanos,
                )
            if (accepted) {
                lastSnapshot =
                    lastSnapshot.copy(
                        registry = registry.snapshot(
                            nowNanos = timestampNanos,
                            maximumSilenceNanos = config.maximumSilenceNanos,
                        ),
                        transport = null,
                        retiredCameraIds = emptyList(),
                    )
            }
            accepted
        }

    fun retireStale(
        nowNanos: Long,
    ): List<DeviceId> =
        lock.write {
            requireOpenLocked()
            val retired =
                registry.expire(
                    nowNanos = nowNanos,
                    maximumSilenceNanos =
                        config.maximumSilenceNanos,
                )
            lastSnapshot =
                lastSnapshot.copy(
                    registry = registry.snapshot(
                        nowNanos = nowNanos,
                        maximumSilenceNanos =
                            config.maximumSilenceNanos,
                    ),
                    retiredCameraIds = retired,
                )
            retired
        }

    fun remove(
        cameraId: DeviceId,
    ): Boolean =
        lock.write {
            requireOpenLocked()
            val removed = registry.remove(cameraId)
            if (removed) {
                lastSnapshot =
                    lastSnapshot.copy(
                        registry = registry.snapshot(
                            nowNanos =
                                lastSnapshot.registry.cameras
                                    .maxOfOrNull {
                                        it.lastHeartbeatNanos
                                    }
                                    ?: 0L,
                            maximumSilenceNanos =
                                config.maximumSilenceNanos,
                        ),
                        retiredCameraIds =
                            listOf(cameraId),
                    )
            }
            removed
        }

    fun allocateTransport(
        linkCapacityKbps: Double,
        authoritativeCameraIds: Set<DeviceId>,
        actorImportanceByCamera:
            Map<DeviceId, Map<Int, Double>> = emptyMap(),
        timestampNanos: Long,
    ): FrontierCameraTransportBudgetEngine.Result =
        lock.write {
            requireOpenLocked()

            val channels =
                registry.all().map { camera ->
                    FrontierCameraTransportBudgetEngine.CameraChannel(
                        cameraId = camera.cameraId.value,
                        authoritative =
                            camera.cameraId in
                                authoritativeCameraIds,
                        quality =
                            camera.timingAuthorityScore,
                        packetLoss =
                            camera.packetLossRatio
                                ?: config.unknownPacketLossEstimate,
                        latencyMillis =
                            camera.latencyMillis
                                ?: config.unknownLatencyEstimateMillis,
                        requestedKbps =
                            camera.requestedKbps
                                ?: if (
                                    camera.cameraId in
                                        authoritativeCameraIds
                                ) {
                                    config.defaultAuthoritativeRequestedKbps
                                } else {
                                    config.defaultWitnessRequestedKbps
                                },
                        actorImportance =
                            actorImportanceByCamera[
                                camera.cameraId
                            ].orEmpty(),
                    )
                }

            val budgetResult =
                transportBudget.allocate(
                    channels = channels,
                    linkCapacityKbps = linkCapacityKbps,
                    timestampNanos = timestampNanos,
                )
            val incompleteNetworkTelemetry =
                registry.all().any {
                    it.packetLossRatio == null || it.latencyMillis == null
                }
            val result =
                if (incompleteNetworkTelemetry) {
                    budgetResult.copy(
                        warnings =
                            (
                                budgetResult.warnings +
                                    "remote-camera-network-telemetry-incomplete"
                            ).distinct(),
                    )
                } else {
                    budgetResult
                }

            lastSnapshot =
                lastSnapshot.copy(
                    registry = registry.snapshot(
                        nowNanos = timestampNanos,
                        maximumSilenceNanos =
                            config.maximumSilenceNanos,
                    ),
                    transport = result,
                    retiredCameraIds = emptyList(),
                )

            result
        }

    fun find(
        cameraId: DeviceId,
    ): RemoteCameraSessionRegistry.CameraNode? =
        lock.read {
            requireOpenLocked()
            registry.find(cameraId)
        }

    fun snapshot(): FederationSnapshot =
        lock.read {
            lastSnapshot
        }

    fun size(): Int =
        lock.read {
            requireOpenLocked()
            registry.size()
        }

    override fun close() {
        lock.write {
            if (closed) return

            val prior =
                registry.snapshot(
                    nowNanos =
                        lastSnapshot.registry.cameras
                            .maxOfOrNull {
                                it.lastHeartbeatNanos
                            }
                            ?: 0L,
                    maximumSilenceNanos =
                        config.maximumSilenceNanos,
                )

            registry.close()
            closed = true

            lastSnapshot =
                FederationSnapshot(
                    registry =
                        RemoteCameraSessionRegistry.Snapshot(
                            cameraCount = 0,
                            healthyCameraCount = 0,
                            averageTimingAuthorityScore =
                                prior.averageTimingAuthorityScore,
                            endpointCollisionCount =
                                prior.endpointCollisionCount,
                            duplicateIdentityCount =
                                prior.duplicateIdentityCount,
                            generation =
                                prior.generation,
                            cameras = emptyList(),
                        ),
                    transport = null,
                    retiredCameraIds =
                        prior.cameras.map {
                            it.cameraId
                        },
                )
        }
    }

    private fun requireOpenLocked() {
        check(!closed) {
            "Remote camera federation coordinator is closed"
        }
    }
}
