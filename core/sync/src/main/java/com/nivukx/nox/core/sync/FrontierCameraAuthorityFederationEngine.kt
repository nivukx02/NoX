package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.Vec3
import kotlin.math.max

/** Selects a bounded authoritative camera set from a distributed federation.
 * Global participation may reach 128 cameras, while expensive reconstruction remains capped at 32.
 */
class FrontierCameraAuthorityFederationEngine(private val config: Config = Config()) {
    data class Config(
        val globalCameraLimit: Int = 128,
        val authoritativeLimit: Int = 32,
        val maximumWitnessCount: Int = 96,
        val minimumHealth: Double = 0.35,
        val minimumSync: Double = 0.55,
        val quarantineAfterBadFrames: Int = 3,
        val maximumMotionBlurRiskForAuthority: Double = 0.72,
        val authorityHysteresisScoreDrop: Double = 0.08,
    ) {
        init {
            require(globalCameraLimit in 2..128)
            require(authoritativeLimit in 2..32)
            require(maximumWitnessCount in 0..(globalCameraLimit - authoritativeLimit))
            require(authoritativeLimit <= globalCameraLimit)
            require(minimumHealth in 0.0..1.0)
            require(minimumSync in 0.0..1.0)
            require(quarantineAfterBadFrames >= 1)
            require(maximumMotionBlurRiskForAuthority in 0.0..1.0)
            require(authorityHysteresisScoreDrop in 0.0..0.5)
        }
    }

    data class CameraEvidence(
        val cameraId: String,
        val sector: Int,
        val health: Double,
        val synchronization: Double,
        val calibration: Double,
        val coverage: Double,
        val actorVisibility: Double,
        val thermalHeadroom: Double,
        val networkHeadroom: Double,
        val freshness: Double,
        val consecutiveBadFrames: Int = 0,
        val calibrationDriftRisk: Double = 0.0,
        val poseUncertaintyMeters: Double = 0.0,
        val poseAngularUncertaintyDegrees: Double = 0.0,
        val actorVisibilityById: Map<Int, Double> = emptyMap(),
        /** Optional frame-level optical motion-blur risk. Null preserves legacy adapter behavior. */
        val motionBlurRisk: Double? = null,
        val positionMeters: Vec3? = null,
        val forward: Vec3? = null,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(sector >= 0)
            listOf(health, synchronization, calibration, coverage, actorVisibility, thermalHeadroom, networkHeadroom, freshness).forEach { require(it in 0.0..1.0) }
            require(consecutiveBadFrames >= 0)
            require(calibrationDriftRisk.isFinite() && calibrationDriftRisk in 0.0..1.0)
            require(poseUncertaintyMeters.isFinite() && poseUncertaintyMeters >= 0.0)
            require(poseAngularUncertaintyDegrees.isFinite() && poseAngularUncertaintyDegrees >= 0.0)
            motionBlurRisk?.let {
                require(it.isFinite() && it in 0.0..1.0)
            }
            positionMeters?.let {
                require(
                    it.x.isFinite() &&
                        it.y.isFinite() &&
                        it.z.isFinite(),
                )
            }
            forward?.let {
                require(
                    it.x.isFinite() &&
                        it.y.isFinite() &&
                        it.z.isFinite(),
                )
            }
            if (positionMeters != null || forward != null) {
                require(positionMeters != null && forward != null)
                require(forward.norm() > 1e-9)
            }
            require(actorVisibilityById.keys.all { it >= 0 })
            require(actorVisibilityById.values.all { it.isFinite() && it in 0.0..1.0 })
        }

        fun eligible(config: Config): Boolean =
            health >= config.minimumHealth &&
                synchronization >= config.minimumSync &&
                calibration > 0.0 &&
                freshness > 0.0 &&
                consecutiveBadFrames < config.quarantineAfterBadFrames &&
                (
                    motionBlurRisk == null ||
                        motionBlurRisk <=
                            config.maximumMotionBlurRiskForAuthority
                )

        fun score(actorId: Int? = null): Double {
            val visibility = actorId?.let { actorVisibilityById[it] ?: actorVisibility } ?: actorVisibility
            val drift = (1.0 - calibrationDriftRisk).coerceIn(0.0, 1.0)
            val translationStability =
                kotlin.math.exp(-poseUncertaintyMeters / 0.050).coerceIn(0.0, 1.0)
            val angularStability =
                kotlin.math.exp(-poseAngularUncertaintyDegrees / 1.50).coerceIn(0.0, 1.0)
            val opticalStability =
                motionBlurRisk
                    ?.let {
                        (1.0 - it).coerceIn(0.0, 1.0)
                    }
                    ?: 1.0
            return 0.15 * health +
                0.14 * synchronization +
                0.16 * calibration +
                0.13 * coverage +
                0.10 * visibility +
                0.07 * thermalHeadroom +
                0.04 * networkHeadroom +
                0.03 * freshness +
                0.07 * drift +
                0.03 * translationStability +
                0.02 * angularStability +
                0.06 * opticalStability
        }
    }

    private val previousAuthorityIdsByActor =
        LinkedHashMap<Int?, Set<String>>()

    data class Result(
        val authoritative: List<CameraEvidence>,
        val witnesses: List<CameraEvidence>,
        val quarantined: List<CameraEvidence>,
        val excluded: List<CameraEvidence>,
        val sectorCoverage: Set<Int>,
        val authorityScore: Double,
    )

    @Synchronized
    fun select(cameras: List<CameraEvidence>): Result =
        selectInternal(cameras, null)

    @Synchronized
    fun selectForActor(
        actorId: Int,
        cameras: List<CameraEvidence>,
    ): Result {
        require(actorId >= 0)
        return selectInternal(cameras, actorId)
    }

    private fun selectInternal(
        cameras: List<CameraEvidence>,
        actorId: Int?,
    ): Result {
        val unique = cameras.distinctBy { it.cameraId }.take(config.globalCameraLimit)
        val quarantined = unique.filter { it.consecutiveBadFrames >= config.quarantineAfterBadFrames }
        val eligible = unique.filter { it !in quarantined && it.eligible(config) }
        val selected = ArrayList<CameraEvidence>(config.authoritativeLimit)
        val previousAuthorityIds = previousAuthorityIdsByActor[actorId].orEmpty()
        val bySector = eligible.groupBy { it.sector }
        val rankScore: (CameraEvidence) -> Double = { camera ->
            camera.score(actorId) +
                if (camera.cameraId in previousAuthorityIds) {
                    config.authorityHysteresisScoreDrop
                } else {
                    0.0
                }
        }
        bySector.keys.sorted().forEach { sector ->
            if (selected.size >= config.authoritativeLimit) return@forEach
            bySector[sector]
                .orEmpty()
                .sortedWith(compareByDescending<CameraEvidence> { rankScore(it) }.thenBy { it.cameraId })
                .firstOrNull()
                ?.let { selected += it }
        }
        eligible
            .sortedWith(compareByDescending<CameraEvidence> { rankScore(it) }.thenBy { it.cameraId })
            .forEach {
                if (selected.size >= config.authoritativeLimit) return@forEach
                if (it !in selected) selected += it
            }
        val witnesses = eligible
            .asSequence()
            .filter { it !in selected }
            .sortedWith(compareByDescending<CameraEvidence> { it.score(actorId) }.thenBy { it.cameraId })
            .take(config.maximumWitnessCount)
            .toList()
        previousAuthorityIdsByActor[actorId] = selected.map { it.cameraId }.toSet()
        while (previousAuthorityIdsByActor.size > 65) {
            previousAuthorityIdsByActor.entries.firstOrNull()?.let { previousAuthorityIdsByActor.remove(it.key) }
                ?: break
        }
        val used = (selected + witnesses).toSet()
        val excluded = unique.filter { it !in used && it !in quarantined }
        val score = if (selected.isEmpty()) 0.0 else selected.map { it.score(actorId) }.average()
        return Result(
            selected,
            witnesses,
            quarantined,
            excluded,
            selected.map { it.sector }.toSet(),
            score.coerceIn(0.0,1.0),
        )
    }

    /** Clears per-actor authority hysteresis at a rig/session lifecycle boundary. */
    @Synchronized
    fun reset() {
        previousAuthorityIdsByActor.clear()
    }
}

private fun score(camera: FrontierCameraAuthorityFederationEngine.CameraEvidence): Double = max(0.0, camera.score())