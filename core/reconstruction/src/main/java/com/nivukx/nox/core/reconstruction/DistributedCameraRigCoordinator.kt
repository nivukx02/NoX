package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.calibration.CaptureRigObservabilityEngine
import kotlin.math.atan2
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Hierarchical coordinator for distributed RGB camera rigs.
 *
 * A single Android device is intentionally not treated as a 100-camera host. Android's concurrent
 * camera APIs are device-capability bounded, so large rigs are represented as a federation of
 * independently timestamped camera nodes. This coordinator turns that federation into one bounded
 * reconstruction evidence contract.
 *
 * Design invariants:
 * - all input cameras are evaluated up to a hard bounded global capacity;
 * - local shard selection prevents a dense cluster from starving distant viewpoints;
 * - global selection is deterministic and geometry-aware;
 * - the expensive dense solver receives only a bounded authoritative subset;
 * - rejected cameras remain visible as evidence and are never silently deleted;
 * - a degraded shard lowers authority without poisoning healthy shards;
 * - duplicate camera identities are rejected before any geometry is computed;
 * - non-finite telemetry is converted into explicit quarantine state;
 * - selection churn is limited by deterministic hysteresis;
 * - this class does not claim physical accuracy. Accuracy remains an empirical benchmark contract.
 *
 * The coordinator is deliberately model-agnostic. It accepts the same CameraEvidence contract used
 * by the existing rig observability engine, which means it can be fed by Camera2, CameraX, network
 * peers, replay sessions, external capture nodes, or future transport implementations.
 */
class DistributedCameraRigCoordinator(
    config: Config = Config(),
) {

    @Volatile
    private var config: Config = config

    data class Config(
        val maximumGlobalCameras: Int = 128,
        val maximumShards: Int = 16,
        // Keep global federation at 128 nodes while allowing a larger, still-bounded authoritative view set.
        val targetSelectedCameras: Int = 32,
        val maximumSelectedCameras: Int = 32,
        val localShardCapacity: Int = 8,
        val minimumCamerasPerShard: Int = 2,
        val shardHealthWeight: Double = 0.42,
        val shardDiversityWeight: Double = 0.24,
        val shardInformationWeight: Double = 0.18,
        val shardTemporalWeight: Double = 0.16,
        val volumetricGeometryWeight: Double = 0.18,
        val globalSelectionHysteresis: Double = 0.08,
        val minimumAuthoritativeHealth: Double = 0.35,
        val minimumAuthoritativeSync: Double = 0.55,
        val minimumAuthoritativeThermalHeadroom: Double = 0.35,
        val minimumAuthoritativeDriftStability: Double = 0.60,
        val maximumTemporalResidualMillis: Double = 12.0,
        val maximumTemporalOutlierRatio: Double = 0.20,
        val maximumDuplicateRatio: Double = 0.10,
        val minimumAuthoritativeTransportHeadroom: Double = 0.45,
        val transportSelectionWeight: Double = 0.10,
    ) {
        init {
            require(maximumGlobalCameras >= 1)
            require(maximumShards >= 1)
            require(targetSelectedCameras >= 1)
            require(maximumSelectedCameras >= targetSelectedCameras)
            require(targetSelectedCameras <= maximumGlobalCameras)
            require(maximumSelectedCameras <= maximumGlobalCameras)
            require(localShardCapacity >= 1)
            require(
                (
                    maximumGlobalCameras +
                        localShardCapacity -
                        1
                    ) / localShardCapacity <=
                    maximumShards
            )
            require(minimumCamerasPerShard >= 1)
            require(shardHealthWeight >= 0.0)
            require(shardDiversityWeight >= 0.0)
            require(shardInformationWeight >= 0.0)
            require(shardTemporalWeight >= 0.0)
            require(volumetricGeometryWeight in 0.0..1.0)
            require(
                shardHealthWeight +
                    shardDiversityWeight +
                    shardInformationWeight +
                    shardTemporalWeight >
                    0.0,
            )
            require(globalSelectionHysteresis in 0.0..0.30)
            require(minimumAuthoritativeHealth in 0.0..1.0)
            require(minimumAuthoritativeSync in 0.0..1.0)
            require(minimumAuthoritativeThermalHeadroom in 0.0..1.0)
            require(minimumAuthoritativeDriftStability in 0.0..1.0)
            require(maximumTemporalResidualMillis > 0.0)
            require(maximumTemporalOutlierRatio in 0.0..1.0)
            require(maximumDuplicateRatio in 0.0..1.0)
            require(minimumAuthoritativeTransportHeadroom in 0.0..1.0)
            require(transportSelectionWeight in 0.0..1.0)
        }
    }

    enum class Authority {
        AUTHORITATIVE,
        SUPPORTING,
        DEGRADED,
        QUARANTINED,
    }

    data class CameraNode(
        val evidence: CaptureRigObservabilityEngine.CameraEvidence,
        val authority: Authority,
        val shardId: Int,
        val score: Double,
        val reason: String,
    )

    data class Shard(
        val id: Int,
        val cameraIds: List<String>,
        val selectedCameraIds: List<String>,
        val health: Double,
        val temporalHealth: Double,
        val informationScore: Double,
        val geometryScore: Double,
        val authority: Authority,
        val warnings: List<String>,
    )

    data class Plan(
        val timestampNanos: Long,
        val acceptedCameraCount: Int,
        val selectedCameraCount: Int,
        val quarantinedCameraCount: Int,
        val supportingCameraCount: Int,
        val shardCount: Int,
        val selectedCameraIds: List<String>,
        val selectedCameraAuthorities: Map<String, Authority>,
        val shards: List<Shard>,
        val globalHealth: Double,
        val globalTemporalHealth: Double,
        val globalInformationScore: Double,
        val globalGeometryScore: Double,
        val volumetricCoverageScore: Double = 0.0,
        val globalTimingDeterminismScore: Double = 0.0,
        val hardwareTimingCameraCount: Int = 0,
        val calibratedLogicalSyncCameraCount: Int = 0,
        val minimumPairwiseDiversity: Double,
        val cameraStarvationRisk: Double,
        val duplicateRatio: Double,
        val authoritative: Boolean,
        val warnings: List<String>,
    ) {
        val hasSufficientEvidence: Boolean
            get() =
                selectedCameraCount >= 2 &&
                    globalHealth >= 0.50 &&
                    globalTemporalHealth >= 0.50 &&
                    globalGeometryScore >= 0.35 &&
                    warnings.none { it == "camera-capacity-exceeded" }

        val precisionBlocked: Boolean
            get() =
                !authoritative ||
                    globalTemporalHealth < 0.65 ||
                    globalGeometryScore < 0.55 ||
                    minimumPairwiseDiversity < 0.20
    }

    private data class ShardCandidate(
        val id: Int,
        val cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
        val score: Double,
        val health: Double,
        val temporalHealth: Double,
        val information: Double,
        val geometry: Double,
    )

    private data class SpatialKey(
        val sector: CaptureRigObservabilityEngine.Sector,
        val radialBand: Int,
        val verticalBand: Int,
    )

    private var observability =
        CaptureRigObservabilityEngine(
            maximumCameras = config.maximumGlobalCameras,
        )

    private val previousSelection = LinkedHashSet<String>()

    @Synchronized
    fun configure(next: Config) {
        if (config == next) return
        config = next
        observability =
            CaptureRigObservabilityEngine(
                maximumCameras = next.maximumGlobalCameras,
            )
        previousSelection.clear()
    }

    @Synchronized
    fun plan(
        timestampNanos: Long,
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
    ): Plan {
        require(timestampNanos >= 0L)

        if (cameras.isEmpty()) {
            previousSelection.clear()
            return emptyPlan(timestampNanos, listOf("no-camera-evidence"))
        }

        val distinct = cameras
            .distinctBy { it.cameraId }

        val duplicateCount = max(0, cameras.size - distinct.size)
        val duplicateRatio =
            duplicateCount.toDouble() /
                cameras.size.coerceAtLeast(1).toDouble()

        val capacityExceeded =
            distinct.size > config.maximumGlobalCameras

        val bounded = distinct
            .sortedWith(
                compareByDescending<CaptureRigObservabilityEngine.CameraEvidence> { it.health }
                    .thenByDescending { it.synchronizationConfidence }
                    .thenBy { it.cameraId },
            )
            .take(config.maximumGlobalCameras)

        val nodes = classifyNodes(bounded)
        val shardGroups = partition(nodes)

        val shardCandidates =
            shardGroups.mapIndexed { index, group ->
                scoreShard(index, group)
            }

        val selected =
            selectGlobal(shardCandidates)

        val selectedIds =
            selected
                .flatMap { it.cameras }
                .map { it.cameraId }
                .distinct()
                .take(config.maximumSelectedCameras)

        val selectedSet = selectedIds.toSet()

        val selectedEvidence =
            bounded.filter { it.cameraId in selectedSet }

        val pairwise =
            observability.pairwise(selectedEvidence)

        val minimumPairwiseDiversity =
            pairwise
                .map {
                    (
                        (it.angularSeparationDegrees / 90.0)
                            .coerceIn(0.0, 1.0) * 0.55 +
                            baselineScore(it.baselineMeters) * 0.45
                    ).coerceIn(0.0, 1.0)
                }
                .minOrNull()
                ?: 0.0

        val globalHealth =
            weightedAverage(selectedEvidence.map { it.health })

        val globalTemporalHealth =
            weightedAverage(selectedEvidence.map { it.temporalTimingHealth })

        val globalInformation =
            weightedAverage(
                selectedEvidence.map {
                    informationScore(it)
                },
            )

        val globalGeometry =
            geometryScore(selectedEvidence)

        val volumetricCoverage =
            volumetricCoverageScore(selectedEvidence)

        val globalTimingDeterminism =
            if (selectedEvidence.isEmpty()) {
                0.0
            } else {
                selectedEvidence
                    .map { it.hardwareTimingHealth }
                    .average()
                    .coerceIn(0.0, 1.0)
            }

        val authorities =
            selectedEvidence.associate { evidence ->
                evidence.cameraId to authorityFor(evidence)
            }

        val quarantined =
            nodes.count { it.authority == Authority.QUARANTINED }

        val supporting =
            selectedEvidence.count {
                authorities[it.cameraId] == Authority.SUPPORTING
            }

        val starvationRisk =
            starvationRisk(
                all = bounded,
                selected = selectedSet,
            )

        val warnings =
            buildList {
                if (capacityExceeded) {
                    add("camera-capacity-exceeded")
                }
                if (duplicateRatio > config.maximumDuplicateRatio) {
                    add("camera-duplicate-ratio-high")
                }
                if (selectedIds.size < min(2, bounded.size)) {
                    add("selected-camera-count-insufficient")
                }
                if (globalTemporalHealth < 0.65) {
                    add("global-camera-timing-degraded")
                }
                if (globalGeometry < 0.55) {
                    add("global-camera-geometry-degraded")
                }
                if (volumetricCoverage < 0.08) {
                    add("camera-volumetric-coverage-low")
                }
                if (minimumPairwiseDiversity < 0.20) {
                    add("selected-camera-diversity-low")
                }
                if (globalTimingDeterminism < 0.70) {
                    add("hardware-timing-determinism-low")
                }
                if (shardCandidates.any {
                        it.cameras.size >= config.minimumCamerasPerShard &&
                            it.score < config.minimumAuthoritativeHealth
                    }
                ) {
                    add("one-or-more-camera-shards-degraded")
                }
                if (quarantined > 0) {
                    add("camera-quarantine-active")
                }
                if (selectedEvidence.any {
                        it.thermalHeadroom <
                            config.minimumAuthoritativeThermalHeadroom
                    }) {
                    add("authoritative-thermal-headroom-limited")
                }
                if (selectedEvidence.any {
                        it.transportHeadroom <
                            config.minimumAuthoritativeTransportHeadroom
                    }) {
                    add("authoritative-transport-headroom-limited")
                }
                if (selectedEvidence.any {
                        it.calibrationDriftStability <
                            config.minimumAuthoritativeDriftStability
                    }) {
                    add("authoritative-calibration-drift-limited")
                }
                if (selectedEvidence.any {
                        it.motionBlurRisk?.let { risk ->
                            risk > 0.60
                        } == true
                    }) {
                    add("authoritative-optical-blur-limited")
                }
                if (starvationRisk > 0.50) {
                    add("camera-selection-starvation-risk")
                }
            }.distinct()

        val authoritative =
            selectedIds.size >= 2 &&
                globalHealth >= config.minimumAuthoritativeHealth &&
                globalTemporalHealth >= config.minimumAuthoritativeSync &&
                globalGeometry >= 0.50 &&
                globalTimingDeterminism >= 0.60 &&
                duplicateRatio <= config.maximumDuplicateRatio &&
                minimumPairwiseDiversity >= 0.15

        val finalWarnings =
            if (authoritative) {
                warnings
            } else {
                (warnings + "distributed-rig-authority-blocked").distinct()
            }

        previousSelection.clear()
        previousSelection.addAll(selectedIds)

        return Plan(
            timestampNanos = timestampNanos,
            acceptedCameraCount = bounded.size,
            selectedCameraCount = selectedIds.size,
            quarantinedCameraCount = quarantined,
            supportingCameraCount = supporting,
            shardCount = shardCandidates.size,
            selectedCameraIds = selectedIds,
            selectedCameraAuthorities = authorities,
            shards =
                shardCandidates.map { shard ->
                    Shard(
                        id = shard.id,
                        cameraIds = shard.cameras.map { it.cameraId },
                        selectedCameraIds =
                            shard.cameras
                                .map { it.cameraId }
                                .filter { it in selectedSet },
                        health = shard.health,
                        temporalHealth = shard.temporalHealth,
                        informationScore = shard.information,
                        geometryScore = shard.geometry,
                        authority =
                            shard.cameras
                                .map { authorityFor(it) }
                                .maxByOrNull { it.ordinal }
                                ?: Authority.QUARANTINED,
                        warnings = shardWarnings(shard),
                    )
                },
            globalHealth = globalHealth,
            globalTemporalHealth = globalTemporalHealth,
            globalInformationScore = globalInformation,
            globalGeometryScore = globalGeometry,
            volumetricCoverageScore = volumetricCoverage,
            globalTimingDeterminismScore = globalTimingDeterminism,
            hardwareTimingCameraCount =
                selectedEvidence.count {
                    it.supportsHardwareReadoutTimestamp
                },
            calibratedLogicalSyncCameraCount =
                selectedEvidence.count {
                    it.logicalSensorSyncCalibrated
                },
            minimumPairwiseDiversity = minimumPairwiseDiversity,
            cameraStarvationRisk = starvationRisk,
            duplicateRatio = duplicateRatio,
            authoritative = authoritative,
            warnings = finalWarnings,
        )
    }

    fun configuration(): Config = config

    fun close() {
        previousSelection.clear()
    }

    private fun classifyNodes(
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
    ): List<CameraNode> {
        return cameras.map { evidence ->
            val authority = authorityFor(evidence)
            val temporal =
                evidence.temporalTimingHealth
            val information =
                informationScore(evidence)
            val geometry =
                geometrySingleScore(evidence)
            val base =
                evidence.health * 0.34 +
                    temporal * 0.20 +
                    information * 0.17 +
                    geometry * 0.19 +
                    evidence.transportHeadroom *
                        config.transportSelectionWeight

            val hysteresis =
                if (evidence.cameraId in previousSelection) {
                    config.globalSelectionHysteresis
                } else {
                    0.0
                }

            CameraNode(
                evidence = evidence,
                authority = authority,
                shardId = -1,
                score = (base + hysteresis).coerceIn(0.0, 1.0),
                reason =
                    when (authority) {
                        Authority.AUTHORITATIVE -> "measured-evidence-healthy"
                        Authority.SUPPORTING -> "useful-supporting-evidence"
                        Authority.DEGRADED -> "evidence-below-authority-target"
                        Authority.QUARANTINED -> "timing-or-calibration-contract-breached"
                    },
            )
        }
    }

    private fun partition(
        nodes: List<CameraNode>,
    ): List<List<CaptureRigObservabilityEngine.CameraEvidence>> {
        if (nodes.isEmpty()) return emptyList()

        val centerX =
            nodes.map { it.evidence.positionMeters.x }.average()
        val centerY =
            nodes.map { it.evidence.positionMeters.y }.average()
        val centerZ =
            nodes.map { it.evidence.positionMeters.z }.average()

        val groups =
            linkedMapOf<SpatialKey, MutableList<CaptureRigObservabilityEngine.CameraEvidence>>()

        nodes
            .sortedBy { it.evidence.cameraId }
            .forEach { node ->
                val p = node.evidence.positionMeters
                val dx = p.x - centerX
                val dz = p.z - centerZ
                val radial =
                    kotlin.math.sqrt(dx * dx + dz * dz)
                val key =
                    SpatialKey(
                        sector =
                            node.evidence.sector
                                ?: inferSector(node.evidence),
                        radialBand =
                            floor(radial / 2.0)
                                .toInt()
                                .coerceIn(0, 7),
                        verticalBand =
                            when {
                                p.y - centerY > 1.0 -> 2
                                p.y - centerY < -1.0 -> 0
                                else -> 1
                            },
                    )
                groups
                    .getOrPut(key) { mutableListOf() }
                    .add(node.evidence)
            }

        /*
         * Preserve every bounded camera in the federation metadata. 16 shards x 8 cameras exactly
         * covers the 128-node envelope, so partitioning must never discard spatial groups.
         */
        val orderedCameras =
            groups.values
                .flatten()
                .sortedWith(
                    compareBy<CaptureRigObservabilityEngine.CameraEvidence> {
                        spatialOrderingKey(
                            it,
                            centerX,
                            centerY,
                            centerZ,
                        )
                    }.thenByDescending {
                        it.health
                    }.thenBy {
                        it.cameraId
                    },
                )

        val shards =
            orderedCameras
                .chunked(config.localShardCapacity)

        require(
            shards.size <= config.maximumShards &&
                shards.flatten().map { it.cameraId }.distinct().size ==
                    nodes.size,
        ) {
            "Distributed camera partition unexpectedly dropped bounded evidence"
        }

        return shards
    }

    private fun spatialOrderingKey(
        evidence: CaptureRigObservabilityEngine.CameraEvidence,
        centerX: Double,
        centerY: Double,
        centerZ: Double,
    ): String {
        val p = evidence.positionMeters
        val dx = p.x - centerX
        val dz = p.z - centerZ
        val radial =
            kotlin.math.sqrt(dx * dx + dz * dz)
        val radialBand =
            floor(radial / 2.0).toInt().coerceIn(0, 7)
        val verticalBand =
            when {
                p.y - centerY > 1.0 -> 2
                p.y - centerY < -1.0 -> 0
                else -> 1
            }
        val sector =
            evidence.sector
                ?: inferSector(evidence)
        return sector.ordinal.toString() +
            ":" + radialBand +
            ":" + verticalBand
    }

    private fun scoreShard(
        id: Int,
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
    ): ShardCandidate {
        val health =
            weightedAverage(cameras.map { it.health })
        val temporal =
            weightedAverage(cameras.map { it.temporalTimingHealth })
        val information =
            weightedAverage(cameras.map { informationScore(it) })
        val geometry =
            geometryScore(cameras)

        val diversity =
            if (cameras.size < 2) {
                0.0
            } else {
                observability
                    .pairwise(cameras)
                    .map {
                        (
                            (it.angularSeparationDegrees / 90.0)
                                .coerceIn(0.0, 1.0) * 0.55 +
                                baselineScore(it.baselineMeters) * 0.45
                        ).coerceIn(0.0, 1.0)
                    }
                    .average()
            }

        val score =
            health * config.shardHealthWeight +
                diversity * config.shardDiversityWeight +
                information * config.shardInformationWeight +
                temporal * config.shardTemporalWeight

        return ShardCandidate(
            id = id,
            cameras = cameras,
            score = score.coerceIn(0.0, 1.0),
            health = health,
            temporalHealth = temporal,
            information = information,
            geometry = geometry,
        )
    }

    private fun selectGlobal(
        shards: List<ShardCandidate>,
    ): List<ShardCandidate> {
        if (shards.isEmpty()) return emptyList()

        val target =
            min(
                config.maximumSelectedCameras,
                max(
                    config.targetSelectedCameras,
                    shards.count {
                        it.cameras.size >= config.minimumCamerasPerShard
                    },
                ),
            )

        val ranked =
            shards
                .map { shard ->
                    shard.copy(
                        cameras =
                            shard.cameras
                                .filter {
                                    authorityFor(it) != Authority.QUARANTINED
                                },
                    )
                }
                .filter { it.cameras.isNotEmpty() }
                .sortedWith(
                    compareByDescending<ShardCandidate> { it.score }
                        .thenBy { it.id },
                )

        if (ranked.isEmpty()) return emptyList()

        /*
         * Phase one enforces spatial coverage. Phase two fills the remaining budget by the same
         * deterministic camera score used by the authority layer. This prevents a high-end rig
         * from spending the complete 32-camera authority budget on a small part of a 128-camera
         * federation just because those cameras have slightly higher local health.
         */
        val selectedByShard =
            linkedMapOf<Int, MutableList<CaptureRigObservabilityEngine.CameraEvidence>>()
        var remaining = target

        ranked
            .asSequence()
            .mapNotNull { shard ->
                val best =
                    shard.cameras
                        .sortedWith(cameraSelectionComparator())
                        .firstOrNull()
                        ?: return@mapNotNull null
                if (
                    authorityFor(best) == Authority.DEGRADED &&
                    shard.cameras.size == 1
                ) {
                    null
                } else {
                    shard to best
                }
            }
            .forEach { (shard, best) ->
                if (remaining > 0) {
                    selectedByShard
                        .getOrPut(shard.id) { mutableListOf() }
                        .add(best)
                    remaining -= 1
                }
            }

        if (remaining > 0) {
            val alreadySelected =
                selectedByShard.values
                    .flatten()
                    .map { it.cameraId }
                    .toSet()

            val fillCandidates =
                ranked
                    .flatMap { shard ->
                        shard.cameras.map { shard.id to it }
                    }
                    .filter { (_, camera) ->
                        camera.cameraId !in alreadySelected
                    }
                    .sortedWith(
                        compareByDescending<
                            Pair<Int, CaptureRigObservabilityEngine.CameraEvidence>
                        > { (_, camera) ->
                            cameraSelectionScore(camera)
                        }.thenBy { (_, camera) ->
                            camera.cameraId
                        },
                    )

            for ((shardId, camera) in fillCandidates) {
                if (remaining <= 0) break
                val bucket =
                    selectedByShard.getOrPut(shardId) {
                        mutableListOf()
                    }
                if (bucket.size >= config.localShardCapacity) continue
                bucket += camera
                remaining -= 1
            }
        }

        return ranked.mapNotNull { shard ->
            val selected =
                selectedByShard[shard.id]
                    .orEmpty()
                    .take(config.localShardCapacity)
            if (selected.isEmpty()) {
                null
            } else {
                shard.copy(cameras = selected)
            }
        }
    }

    private fun cameraSelectionScore(
        camera: CaptureRigObservabilityEngine.CameraEvidence,
    ): Double {
        val hysteresis =
            if (camera.cameraId in previousSelection) {
                config.globalSelectionHysteresis
            } else {
                0.0
            }

        val optical =
            1.0 -
                (
                    camera.motionBlurRisk
                        ?.coerceIn(0.0, 1.0)
                        ?: 0.0
                    )

        val opticalAttenuatedHealth =
            camera.health *
                (
                    0.72 +
                        0.28 * optical
                    ).coerceIn(0.72, 1.0)

        return (
            opticalAttenuatedHealth * 0.44 +
                camera.temporalTimingHealth * 0.17 +
                informationScore(camera) * 0.15 +
                camera.dynamicParallaxScore * 0.10 +
                camera.occlusionCoverage * 0.06 +
                optical * 0.05 +
                hysteresis * 0.03
            ).coerceIn(0.0, 1.0)
    }

    private fun cameraSelectionComparator():
        Comparator<CaptureRigObservabilityEngine.CameraEvidence> =
        compareByDescending<
            CaptureRigObservabilityEngine.CameraEvidence
        > { camera ->
            cameraSelectionScore(camera)
        }.thenBy { it.cameraId }

    private fun authorityFor(
        camera: CaptureRigObservabilityEngine.CameraEvidence,
    ): Authority {
        if (!camera.active) return Authority.QUARANTINED
        if (camera.calibrationConfidence < 0.35) return Authority.QUARANTINED
        if (camera.synchronizationConfidence < 0.35) return Authority.QUARANTINED
        if (
            camera.temporalResidualP95Milliseconds >
                config.maximumTemporalResidualMillis ||
            camera.temporalOutlierRatio >
                config.maximumTemporalOutlierRatio
        ) {
            return Authority.QUARANTINED
        }
        if (camera.motionBlurRisk?.let { it > 0.92 } == true) {
            return Authority.QUARANTINED
        }

        val health = camera.health
        return when {
            health >= config.minimumAuthoritativeHealth &&
                camera.synchronizationConfidence >=
                config.minimumAuthoritativeSync &&
                camera.thermalHeadroom >=
                    config.minimumAuthoritativeThermalHeadroom &&
                camera.calibrationDriftStability >=
                    config.minimumAuthoritativeDriftStability &&
                camera.transportHeadroom >=
                    config.minimumAuthoritativeTransportHeadroom ->
                Authority.AUTHORITATIVE
            health >= 0.45 ->
                Authority.SUPPORTING
            else ->
                Authority.DEGRADED
        }
    }

    private fun informationScore(
        camera: CaptureRigObservabilityEngine.CameraEvidence,
    ): Double =
        (
            camera.surfaceInformationGain * 0.34 +
                camera.occlusionCoverage * 0.28 +
                camera.dynamicParallaxScore * 0.22 +
                camera.trackedObjectCoverage * 0.16
        ).coerceIn(0.0, 1.0)

    private fun volumetricCoverageScore(
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
    ): Double {
        if (cameras.size < 4) return 0.0

        val xs = cameras.map { it.positionMeters.x }
        val ys = cameras.map { it.positionMeters.y }
        val zs = cameras.map { it.positionMeters.z }

        val spanX =
            (xs.maxOrNull() ?: 0.0) -
                (xs.minOrNull() ?: 0.0)
        val spanY =
            (ys.maxOrNull() ?: 0.0) -
                (ys.minOrNull() ?: 0.0)
        val spanZ =
            (zs.maxOrNull() ?: 0.0) -
                (zs.minOrNull() ?: 0.0)

        val diagonal =
            kotlin.math.sqrt(
                spanX * spanX +
                    spanY * spanY +
                    spanZ * spanZ,
            )

        if (!diagonal.isFinite() || diagonal <= 1e-6) {
            return 0.0
        }

        val normalizedX =
            (spanX / diagonal).coerceIn(0.0, 1.0)
        val normalizedY =
            (spanY / diagonal).coerceIn(0.0, 1.0)
        val normalizedZ =
            (spanZ / diagonal).coerceIn(0.0, 1.0)

        return (
            normalizedX *
                normalizedY *
                normalizedZ
        ).let(kotlin.math.cbrt).coerceIn(0.0, 1.0)
    }

    private fun geometrySingleScore(
        camera: CaptureRigObservabilityEngine.CameraEvidence,
    ): Double =
        (
            camera.calibrationConfidence * 0.30 +
                camera.trackingConfidence * 0.18 +
                camera.centroidConnectivity * 0.12 +
                camera.dynamicParallaxScore * 0.18 +
                camera.occlusionCoverage * 0.12 +
                camera.surfaceInformationGain * 0.10
        ).coerceIn(0.0, 1.0)

    private fun geometryScore(
        cameras: List<CaptureRigObservabilityEngine.CameraEvidence>,
    ): Double {
        if (cameras.isEmpty()) return 0.0
        val single =
            cameras.map(::geometrySingleScore).average()
        val pairwise =
            if (cameras.size < 2) {
                0.0
            } else {
                observability
                    .pairwise(cameras)
                    .map {
                        (
                            (it.angularSeparationDegrees / 90.0)
                                .coerceIn(0.0, 1.0) * 0.60 +
                                baselineScore(it.baselineMeters) * 0.40
                        ).coerceIn(0.0, 1.0)
                    }
                    .average()
            }
        val volumetric =
            volumetricCoverageScore(cameras)

        val configuredVolumeWeight =
            config.volumetricGeometryWeight
                .coerceIn(0.0, 1.0)

        val remainingWeight =
            (1.0 - configuredVolumeWeight)
                .coerceAtLeast(0.0)

        return (
            single * remainingWeight * 0.60 +
                pairwise * remainingWeight * 0.40 +
                volumetric * configuredVolumeWeight
        ).coerceIn(0.0, 1.0)
    }

    private fun weightedAverage(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        return values
            .filter { it.isFinite() }
            .map { it.coerceIn(0.0, 1.0) }
            .average()
            .coerceIn(0.0, 1.0)
    }

    private fun baselineScore(
        baselineMeters: Double,
    ): Double =
        when {
            baselineMeters < 0.05 ->
                (baselineMeters / 0.05 * 0.20).coerceIn(0.0, 0.20)
            baselineMeters < 0.25 ->
                0.20 +
                    (baselineMeters - 0.05) /
                    0.20 * 0.50
            baselineMeters < 1.0 ->
                0.70 +
                    (baselineMeters - 0.25) /
                    0.75 * 0.30
            else -> 1.0
        }.coerceIn(0.0, 1.0)

    private fun starvationRisk(
        all: List<CaptureRigObservabilityEngine.CameraEvidence>,
        selected: Set<String>,
    ): Double {
        if (all.isEmpty()) return 0.0
        val excludedHealthy =
            all.count {
                it.cameraId !in selected &&
                    it.health >= config.minimumAuthoritativeHealth
            }
        return (
            excludedHealthy.toDouble() /
                all.size.toDouble()
        ).coerceIn(0.0, 1.0)
    }

    private fun shardWarnings(
        shard: ShardCandidate,
    ): List<String> =
        buildList {
            if (shard.health < config.minimumAuthoritativeHealth) {
                add("shard-health-degraded")
            }
            if (shard.temporalHealth < config.minimumAuthoritativeSync) {
                add("shard-timing-degraded")
            }
            if (shard.geometry < 0.50) {
                add("shard-geometry-degraded")
            }
            if (shard.cameras.size < config.minimumCamerasPerShard) {
                add("shard-underpopulated")
            }
        }

    private fun inferSector(
        camera: CaptureRigObservabilityEngine.CameraEvidence,
    ): CaptureRigObservabilityEngine.Sector {
        val p = camera.positionMeters
        if (p.y > 2.2) return CaptureRigObservabilityEngine.Sector.TOP
        if (p.y < 0.4) return CaptureRigObservabilityEngine.Sector.BOTTOM

        val angle =
            Math.toDegrees(
                atan2(p.x, p.z),
            )

        return when {
            angle >= -22.5 && angle < 22.5 ->
                CaptureRigObservabilityEngine.Sector.FRONT
            angle < 67.5 ->
                CaptureRigObservabilityEngine.Sector.FRONT_RIGHT
            angle < 112.5 ->
                CaptureRigObservabilityEngine.Sector.RIGHT
            angle < 157.5 ->
                CaptureRigObservabilityEngine.Sector.REAR_RIGHT
            angle >= 157.5 || angle < -157.5 ->
                CaptureRigObservabilityEngine.Sector.REAR
            angle >= -157.5 && angle < -112.5 ->
                CaptureRigObservabilityEngine.Sector.REAR_LEFT
            angle >= -112.5 && angle < -67.5 ->
                CaptureRigObservabilityEngine.Sector.LEFT
            else ->
                CaptureRigObservabilityEngine.Sector.FRONT_LEFT
        }
    }

    private fun emptyPlan(
        timestampNanos: Long,
        warnings: List<String>,
    ): Plan =
        Plan(
            timestampNanos = timestampNanos,
            acceptedCameraCount = 0,
            selectedCameraCount = 0,
            quarantinedCameraCount = 0,
            supportingCameraCount = 0,
            shardCount = 0,
            selectedCameraIds = emptyList(),
            selectedCameraAuthorities = emptyMap(),
            shards = emptyList(),
            globalHealth = 0.0,
            globalTemporalHealth = 0.0,
            globalInformationScore = 0.0,
            globalGeometryScore = 0.0,
            minimumPairwiseDiversity = 0.0,
            cameraStarvationRisk = 0.0,
            duplicateRatio = 0.0,
            authoritative = false,
            warnings = warnings,
        )
}
