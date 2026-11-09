package com.nivukx.nox.core.network

import kotlin.math.max
import kotlin.math.min

/**
 * Distributed camera transport budgeter for up to 128 cameras and a 50-actor workload.
 *
 * Authoritative cameras receive reserved transport first. Witness cameras keep low-rate evidence
 * paths alive so they can be promoted later without reconnecting. Total bandwidth is hard-bounded.
 */
class FrontierCameraTransportBudgetEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumCameras: Int = 128,
        val authoritativeCameras: Int = 32,
        val targetActors: Int = 50,
        val hardActorCeiling: Int = 64,
        val minimumAuthoritativeKbps: Double = 1_800.0,
        val minimumWitnessKbps: Double = 180.0,
        val maximumKbpsPerCamera: Double = 20_000.0,
        val queueBudgetBytes: Long = 32L * 1024L * 1024L,
        val packetLossTarget: Double = 0.03,
        val latencyTargetMillis: Double = 30.0,
        val timingReserveRatio: Double = 0.12,
        val identityReserveRatio: Double = 0.18,
        val geometryReserveRatio: Double = 0.42,
        val surfaceReserveRatio: Double = 0.24,
        val diagnosticsReserveRatio: Double = 0.04,
    ) {
        init {
            require(maximumCameras in 2..128)
            require(authoritativeCameras in 2..maximumCameras)
            require(targetActors in 1..hardActorCeiling)
            require(hardActorCeiling in 1..64)
            require(minimumAuthoritativeKbps > 0.0)
            require(minimumWitnessKbps > 0.0)
            require(maximumKbpsPerCamera >= minimumAuthoritativeKbps)
            require(queueBudgetBytes > 0L)
            require(packetLossTarget in 0.0..1.0)
            require(latencyTargetMillis > 0.0)
            val reserveTotal =
                timingReserveRatio +
                    identityReserveRatio +
                    geometryReserveRatio +
                    surfaceReserveRatio +
                    diagnosticsReserveRatio
            require(timingReserveRatio in 0.0..1.0)
            require(identityReserveRatio in 0.0..1.0)
            require(geometryReserveRatio in 0.0..1.0)
            require(surfaceReserveRatio in 0.0..1.0)
            require(diagnosticsReserveRatio in 0.0..1.0)
            require(kotlin.math.abs(reserveTotal - 1.0) < 1e-6)
        }
    }

    data class CameraChannel(
        val cameraId: String,
        val authoritative: Boolean,
        val quality: Double,
        val packetLoss: Double,
        val latencyMillis: Double,
        val requestedKbps: Double,
        val actorImportance: Map<Int, Double> = emptyMap(),
    ) {
        init {
            require(cameraId.isNotBlank())
            require(quality in 0.0..1.0)
            require(packetLoss in 0.0..1.0)
            require(latencyMillis >= 0.0 && latencyMillis.isFinite())
            require(requestedKbps >= 0.0 && requestedKbps.isFinite())
            require(actorImportance.keys.all { it in 0..64 })
            require(actorImportance.values.all { it.isFinite() && it in 0.0..1.0 })
        }
    }

    enum class TransportTier {
        FULL_EVIDENCE,
        REDUCED_EVIDENCE,
        WITNESS_HEARTBEAT,
        QUARANTINED,
    }

    data class Allocation(
        val cameraId: String,
        val tier: TransportTier,
        val allocatedKbps: Double,
        val queueBytes: Long,
        val perActorKbps: Map<Int, Double>,
        val timingKbps: Double = 0.0,
        val identityKbps: Double = 0.0,
        val geometryKbps: Double = 0.0,
        val surfaceKbps: Double = 0.0,
        val diagnosticsKbps: Double = 0.0,
        val reason: String,
    )

    data class Result(
        val allocations: List<Allocation>,
        val totalAllocatedKbps: Double,
        val totalQueueBytes: Long,
        val overloaded: Boolean,
        val authoritativeUnderfunded: Int,
        val witnessUnderfunded: Int,
        val networkHealth: Double,
        val warnings: List<String>,
    )

    fun allocate(
        channels: List<CameraChannel>,
        linkCapacityKbps: Double,
        timestampNanos: Long,
    ): Result {
        require(linkCapacityKbps > 0.0 && linkCapacityKbps.isFinite())
        require(timestampNanos >= 0L)
        val unique = channels.distinctBy { it.cameraId }.take(config.maximumCameras)
        if (unique.isEmpty()) {
            return Result(emptyList(),0.0,0L,true,0,0,0.0,listOf("no-camera-channels"))
        }

        val minimumDemand = unique.associateWith {
            if (it.authoritative) config.minimumAuthoritativeKbps else config.minimumWitnessKbps
        }
        val minimumTotal = minimumDemand.values.sum().coerceAtLeast(1.0)
        val minimumGrant = min(linkCapacityKbps, minimumTotal)
        val minimumScale = minimumGrant / minimumTotal
        val weights = unique.associateWith { max(0.05, priority(it) * it.requestedKbps.coerceAtLeast(1.0)) }
        val weightTotal = weights.values.sum().coerceAtLeast(1.0)
        val granted = linkedMapOf<String, Double>()
        unique.forEach { channel ->
            granted[channel.cameraId] = minimumDemand[channel]!! * minimumScale
        }
        var remaining = max(0.0, linkCapacityKbps - minimumGrant)
        if (remaining > 1e-9) {
            unique.forEach { channel ->
                if (remaining <= 1e-9) return@forEach
                val current = granted[channel.cameraId] ?: 0.0
                val cap = (channel.requestedKbps.coerceAtMost(config.maximumKbpsPerCamera) - current).coerceAtLeast(0.0)
                if (cap <= 0.0) return@forEach
                val share = min(cap, remaining * (weights[channel]!! / weightTotal))
                granted[channel.cameraId] = current + share
            }
            remaining = max(0.0, linkCapacityKbps - granted.values.sum())
        }

        val allocations = unique.sortedWith(
            compareByDescending<CameraChannel> { it.authoritative }.thenByDescending { priority(it) }.thenBy { it.cameraId }
        ).map { channel ->
            val minimum = minimumDemand[channel]!!
            val requested = (granted[channel.cameraId] ?: 0.0).coerceIn(0.0, config.maximumKbpsPerCamera)
            val linkHealth = transportHealth(channel)
            val tier = when {
                linkHealth < 0.30 -> TransportTier.QUARANTINED
                channel.authoritative && requested >= minimum * 0.92 -> TransportTier.FULL_EVIDENCE
                requested >= minimum * 1.75 -> TransportTier.REDUCED_EVIDENCE
                else -> TransportTier.WITNESS_HEARTBEAT
            }
            val effective = when (tier) {
                TransportTier.QUARANTINED -> 0.0
                TransportTier.FULL_EVIDENCE -> requested
                TransportTier.REDUCED_EVIDENCE -> requested * 0.62
                TransportTier.WITNESS_HEARTBEAT -> min(requested, minimum)
            }
            val evidence = evidenceClassBudget(effective)
            Allocation(
                cameraId = channel.cameraId,
                tier = tier,
                allocatedKbps = effective,
                queueBytes = min(
                    config.queueBudgetBytes / unique.size.coerceAtLeast(1).toLong(),
                    bytesForKbps(effective, 0.35)
                ),
                perActorKbps = actorBudget(effective, channel.actorImportance),
                timingKbps = evidence.timingKbps,
                identityKbps = evidence.identityKbps,
                geometryKbps = evidence.geometryKbps,
                surfaceKbps = evidence.surfaceKbps,
                diagnosticsKbps = evidence.diagnosticsKbps,
                reason = reason(tier, linkHealth),
            )
        }

        val totalAllocated = allocations.sumOf { it.allocatedKbps }.coerceAtMost(linkCapacityKbps)
        val totalQueue = allocations.sumOf { it.queueBytes }.coerceAtMost(config.queueBudgetBytes)
        val authUnderfunded = allocations.count {
            val source = unique.first { source -> source.cameraId == it.cameraId }
            source.authoritative && it.allocatedKbps + 1e-9 < config.minimumAuthoritativeKbps
        }
        val witnessUnderfunded = allocations.count {
            val source = unique.first { source -> source.cameraId == it.cameraId }
            !source.authoritative && it.allocatedKbps + 1e-9 < config.minimumWitnessKbps
        }
        val health = unique.map { transportHealth(it) }.average().coerceIn(0.0,1.0)
        val overloaded =
            linkCapacityKbps + 1e-6 < minimumTotal ||
                totalQueue >= config.queueBudgetBytes

        return Result(
            allocations = allocations,
            totalAllocatedKbps = totalAllocated,
            totalQueueBytes = totalQueue,
            overloaded = overloaded,
            authoritativeUnderfunded = authUnderfunded,
            witnessUnderfunded = witnessUnderfunded,
            networkHealth = health,
            warnings = buildList {
                if (linkCapacityKbps < minimumTotal) add("frontier-network-capacity-below-minimum-evidence")
                if (authUnderfunded > 0) add("authoritative-camera-transport-underfunded")
                if (witnessUnderfunded > 0) add("witness-camera-transport-underfunded")
                if (unique.any { transportHealth(it) < 0.45 }) add("low-quality-camera-links-present")
                if (unique.count { it.authoritative } > config.authoritativeCameras) add("authoritative-camera-count-over-policy")
                if (totalQueue >= config.queueBudgetBytes) add("frontier-network-queue-cap-reached")
            }.distinct(),
        )
    }

    private fun priority(channel: CameraChannel): Double {
        val actor = if (channel.actorImportance.isEmpty()) 0.50 else channel.actorImportance.values.average()
        return (channel.quality * 0.34 + transportHealth(channel) * 0.36 + actor * 0.18 + if (channel.authoritative) 0.12 else 0.02).coerceIn(0.0,1.0)
    }

    private fun transportHealth(channel: CameraChannel): Double {
        val loss = (1.0 - channel.packetLoss / max(config.packetLossTarget,1e-6)).coerceIn(0.0,1.0)
        val latency = (1.0 - channel.latencyMillis / max(config.latencyTargetMillis,1e-6)).coerceIn(0.0,1.0)
        return (channel.quality * 0.46 + loss * 0.30 + latency * 0.24).coerceIn(0.0,1.0)
    }

    private data class EvidenceClassBudget(
        val timingKbps: Double,
        val identityKbps: Double,
        val geometryKbps: Double,
        val surfaceKbps: Double,
        val diagnosticsKbps: Double,
    )

    private fun evidenceClassBudget(
        totalKbps: Double,
    ): EvidenceClassBudget =
        EvidenceClassBudget(
            timingKbps = totalKbps * config.timingReserveRatio,
            identityKbps = totalKbps * config.identityReserveRatio,
            geometryKbps = totalKbps * config.geometryReserveRatio,
            surfaceKbps = totalKbps * config.surfaceReserveRatio,
            diagnosticsKbps = totalKbps * config.diagnosticsReserveRatio,
        )

    private fun actorBudget(totalKbps: Double, importance: Map<Int, Double>): Map<Int, Double> {
        val actors = importance.filterKeys { it in 0 until config.hardActorCeiling }.ifEmpty { (0 until config.targetActors).associateWith { 1.0 } }
        val totalWeight = actors.values.sum().coerceAtLeast(1e-9)
        return actors.entries.associate { (actor, score) -> actor to totalKbps * score / totalWeight }
    }

    private fun bytesForKbps(kbps: Double, seconds: Double): Long = (kbps * 1000.0 / 8.0 * seconds).toLong().coerceAtLeast(0L)

    private fun reason(tier: TransportTier, health: Double): String = when (tier) {
        TransportTier.FULL_EVIDENCE -> "authoritative-high-fidelity"
        TransportTier.REDUCED_EVIDENCE -> "bandwidth-adaptive-detail"
        TransportTier.WITNESS_HEARTBEAT -> "witness-low-rate-evidence"
        TransportTier.QUARANTINED -> "transport-health-quarantine"
    }
}
