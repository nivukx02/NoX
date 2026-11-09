package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.min

/**
 * Deterministic geometry-aware camera rig planner.
 *
 * More cameras are not automatically more useful. This engine rewards calibration health,
 * temporal trust, spatial baseline, angular diversity, and sector coverage while penalizing
 * redundant viewpoints. Selection is replayable for benchmark runs.
 */
class CaptureRigObservabilityEngine(
    private val maximumCameras: Int = 128,
) {

    companion object {
        private val recoveryPolicy =
            CameraAutoHealPolicyEngine()
    }
    enum class Sector {
        FRONT, FRONT_RIGHT, RIGHT, REAR_RIGHT,
        REAR, REAR_LEFT, LEFT, FRONT_LEFT, TOP, BOTTOM, UNKNOWN,
    }

    data class CameraEvidence(
        val cameraId: String,
        val role: DeviceRole,
        val positionMeters: Vec3,
        val forwardVector: Vec3,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val trackingConfidence: Double = 1.0,
        val thermalHeadroom: Double = 1.0,
        val transportHeadroom: Double = 1.0,
        val active: Boolean = true,
        val sector: Sector? = null,
        val imageErrorPx: Double = 0.0,
        val centroidConnectivity: Double = 1.0,
        val temperatureDeltaCelsius: Double = 0.0,
        val calibrationDriftStability: Double = 1.0,
        val trackedObjectCoverage: Double = 1.0,
        val calibrationAgeSeconds: Double = 0.0,
        val surfaceInformationGain: Double = 1.0,
        val occlusionCoverage: Double = 1.0,
        val dynamicParallaxScore: Double = 1.0,
        val temporalResidualP95Milliseconds: Double = 0.0,
        val temporalOutlierRatio: Double = 0.0,
        val hardwareReadoutTimingConfidence: Double = 0.0,
        val hardwareReadoutP95JitterMilliseconds: Double = 0.0,
        val logicalSensorSyncCalibrated: Boolean = false,
        val photometricConsistency: Double = 1.0,
        val whiteBalanceShift: Double = 0.0,
        /** Optional frame-level optical motion-blur risk. Null means the adapter has no estimate. */
        val motionBlurRisk: Double? = null,
        /** Optional native subpixel retention evidence. Null means unavailable. */
        val subpixelMotionRetention: Double? = null,
    ) {
        init {
            require(cameraId.isNotBlank())
            listOf(
                calibrationConfidence,
                synchronizationConfidence,
                trackingConfidence,
                thermalHeadroom,
                transportHeadroom,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            listOf(
                imageErrorPx,
                centroidConnectivity,
                temperatureDeltaCelsius,
                calibrationDriftStability,
                trackedObjectCoverage,
                calibrationAgeSeconds,
                surfaceInformationGain,
                occlusionCoverage,
                dynamicParallaxScore,
                temporalResidualP95Milliseconds,
                temporalOutlierRatio,
                hardwareReadoutTimingConfidence,
                hardwareReadoutP95JitterMilliseconds,
                photometricConsistency,
                whiteBalanceShift,
            ).forEach {
                require(it.isFinite())
            }
            require(imageErrorPx >= 0.0)
            require(centroidConnectivity in 0.0..1.0)
            require(temperatureDeltaCelsius >= 0.0)
            require(calibrationDriftStability in 0.0..1.0)
            require(trackedObjectCoverage in 0.0..1.0)
            require(calibrationAgeSeconds >= 0.0)
            require(surfaceInformationGain in 0.0..1.0)
            require(occlusionCoverage in 0.0..1.0)
            require(dynamicParallaxScore in 0.0..1.0)
            require(temporalResidualP95Milliseconds >= 0.0)
            require(temporalOutlierRatio in 0.0..1.0)
            require(hardwareReadoutTimingConfidence in 0.0..1.0)
            require(hardwareReadoutP95JitterMilliseconds >= 0.0)
            require(photometricConsistency in 0.0..1.0)
            require(whiteBalanceShift >= 0.0 && whiteBalanceShift <= 1.0)
            require(
                motionBlurRisk == null ||
                    (
                        motionBlurRisk.isFinite() &&
                            motionBlurRisk in 0.0..1.0
                    ),
            )
            require(
                subpixelMotionRetention == null ||
                    (
                        subpixelMotionRetention.isFinite() &&
                            subpixelMotionRetention in 0.0..1.0
                    ),
            )
            require(positionMeters.x.isFinite())
            require(positionMeters.y.isFinite())
            require(positionMeters.z.isFinite())
            require(forwardVector.x.isFinite())
            require(forwardVector.y.isFinite())
            require(forwardVector.z.isFinite())
        }

        private val opticalHealth: Double
            get() =
                motionBlurRisk
                    ?.let {
                        (
                            1.0 -
                                0.72 * it
                        ).coerceIn(0.0, 1.0)
                    }
                    ?: 1.0

        val health: Double
            get() {
                val weighted =
                    calibrationConfidence * 0.22 +
                        synchronizationConfidence * 0.17 +
                        trackingConfidence * 0.10 +
                        opticalHealth * 0.05 +
                        (subpixelMotionRetention ?: 1.0) * 0.04 +
                        thermalHeadroom * 0.05 +
                        transportHeadroom * 0.05 +
                        imageErrorHealth * 0.07 +
                        centroidConnectivity * 0.05 +
                        calibrationDriftStability * 0.04 +
                        trackedObjectCoverage * 0.04 +
                        surfaceInformationGain * 0.05 +
                        occlusionCoverage * 0.05 +
                        dynamicParallaxScore * 0.05 +
                        temporalTimingHealth * 0.04 +
                        photometricHealth * 0.04
                return (
                    weighted / 1.08
                ).coerceIn(0.0, 1.0)
            }

        val temporalTimingHealth: Double
            get() = (
                (
                    1.0 -
                        (temporalResidualP95Milliseconds / 12.0)
                            .coerceIn(0.0, 1.0)
                ) * 0.56 +
                    (
                        1.0 -
                            temporalOutlierRatio
                    ).coerceIn(0.0, 1.0) * 0.24 +
                    hardwareTimingHealth * 0.20
            ).coerceIn(0.0, 1.0)

        val hardwareTimingHealth: Double
            get() {
                val jitterScore =
                    (
                        1.0 -
                            hardwareReadoutP95JitterMilliseconds / 2.5
                    ).coerceIn(0.0, 1.0)

                val readoutScore =
                    if (hardwareReadoutTimingConfidence > 0.0) {
                        hardwareReadoutTimingConfidence
                    } else {
                        0.68
                    }

                val logicalScore =
                    if (logicalSensorSyncCalibrated) 1.0 else 0.78

                return (
                    readoutScore * 0.55 +
                        jitterScore * 0.25 +
                        logicalScore * 0.20
                    ).coerceIn(0.0, 1.0)
            }

        val photometricHealth: Double
            get() =
                (
                    photometricConsistency * 0.70 +
                        (
                            1.0 -
                                whiteBalanceShift
                        ).coerceIn(0.0, 1.0) * 0.30
                    ).coerceIn(0.0, 1.0)

        val imageErrorHealth: Double
            get() =
                (1.0 -
                    (imageErrorPx / 8.0)
                        .coerceIn(0.0, 1.0))
                    .coerceIn(0.0, 1.0)

        val recoveryAction: CameraAutoHealPolicyEngine.Action
            get() =
                recoveryPolicy.evaluate(
                    CameraAutoHealPolicyEngine.Input(
                        cameraId = cameraId,
                        calibrationConfidence =
                            calibrationConfidence,
                        imageErrorPx = imageErrorPx,
                        centroidConnectivity =
                            centroidConnectivity,
                        temperatureDeltaCelsius =
                            temperatureDeltaCelsius,
                        calibrationDriftStability =
                            calibrationDriftStability,
                        trackedObjectCoverage =
                            trackedObjectCoverage,
                        calibrationAgeSeconds =
                            calibrationAgeSeconds,
                    ),
                ).action

        val normalizedForward: Vec3
            get() = forwardVector.normalized()
    }

    data class Pairwise(
        val first: String,
        val second: String,
        val baselineMeters: Double,
        val angularSeparationDegrees: Double,
        val utility: Double,
    )

    data class Health(
        val activeCameras: Int,
        val meanHealth: Double,
        val minimumHealth: Double,
        val meanBaselineMeters: Double,
        val angularDiversity: Double,
        val sectorCoverage: Double,
        val roleDiversity: Double,
        val redundancy: Double,
        val observabilityScore: Double,
        val warnings: List<String>,
        val recoveryActions:
            Map<String, CameraAutoHealPolicyEngine.Action> =
            emptyMap(),
    )

    data class Selection(
        val selected: List<CameraEvidence>,
        val rejected: List<CameraEvidence>,
        val marginalGain: Map<String, Double>,
        val objectiveScore: Double,
        val warnings: List<String>,
    )

    fun analyze(cameras: List<CameraEvidence>): Health {
        val allDistinct = cameras.asSequence()
            .filter { it.active }
            .distinctBy { it.cameraId }
            .take(maximumCameras)
            .toList()

        val active = allDistinct.filter {
            it.recoveryAction !=
                CameraAutoHealPolicyEngine.Action.HOLD &&
                it.recoveryAction !=
                CameraAutoHealPolicyEngine.Action.RECALIBRATION_REQUIRED
        }

        if (active.isEmpty()) {
            return Health(
                activeCameras = 0,
                meanHealth = 0.0,
                minimumHealth = 0.0,
                meanBaselineMeters = 0.0,
                angularDiversity = 0.0,
                sectorCoverage = 0.0,
                roleDiversity = 0.0,
                redundancy = 0.0,
                observabilityScore = 0.0,
                warnings = buildList {
                    add("no-active-cameras")
                    if (allDistinct.any {
                        it.recoveryAction ==
                            CameraAutoHealPolicyEngine.Action.RECALIBRATION_REQUIRED
                    }) {
                        add("all-cameras-require-recalibration")
                    }
                    if (allDistinct.any {
                        it.recoveryAction ==
                            CameraAutoHealPolicyEngine.Action.HOLD
                    }) {
                        add("all-cameras-held")
                    }
                },
                recoveryActions =
                    allDistinct.associate {
                        it.cameraId to it.recoveryAction
                    },
            )
        }

        val pairs = pairwise(active)
        val health = active.map { it.health }
        val baselines = pairs.map { it.baselineMeters }
        val angular = angularDiversity(pairs)
        val sectors = sectorCoverage(active)
        val roles = roleDiversity(active)
        val redundancy = redundancy(pairs)
        val meanBaseline = baselines.averageOrZero()

        val informationGain =
            active.map {
                (
                    it.surfaceInformationGain * 0.45 +
                        it.occlusionCoverage * 0.30 +
                        it.dynamicParallaxScore * 0.25
                    ).coerceIn(0.0, 1.0)
            }.average()
        val score = (
            health.average() * 0.30 +
                angular * 0.20 +
                sectors * 0.15 +
                roles * 0.08 +
                informationGain * 0.12 +
                (1.0 - redundancy) * 0.08 +
                baselineQuality(meanBaseline) * 0.07
            ).coerceIn(0.0, 1.0)

        return Health(
            activeCameras = active.size,
            meanHealth = health.average(),
            minimumHealth = health.minOrNull() ?: 0.0,
            meanBaselineMeters = meanBaseline,
            angularDiversity = angular,
            sectorCoverage = sectors,
            roleDiversity = roles,
            redundancy = redundancy,
            observabilityScore = score,
            warnings = buildList {
                if (health.average() < 0.65) {
                    add("mean-camera-health-below-target")
                }
                if ((health.minOrNull() ?: 0.0) < 0.40) {
                    add("camera-health-critical")
                }
                if (angular < 0.40) {
                    add("view-angle-diversity-low")
                }
                if (sectors < 0.50) {
                    add("capture-sector-coverage-low")
                }
                if (roles < 0.50) {
                    add("camera-role-diversity-low")
                }
                if (redundancy > 0.75 && active.size > 3) {
                    add("camera-redundancy-high")
                }
                if (informationGain < 0.50) {
                    add("camera-information-gain-low")
                }
                if (active.any {
                    it.temporalResidualP95Milliseconds > 12.0 ||
                        it.temporalOutlierRatio > 0.20
                }) {
                    add("camera-temporal-contract-breach")
                }
                if (active.any {
                    it.hardwareTimingHealth < 0.60
                }) {
                    add("hardware-timing-authority-low")
                }
                if (active.any {
                    it.recoveryAction ==
                        CameraAutoHealPolicyEngine.Action.HEAL_CANDIDATE
                }) {
                    add("camera-auto-heal-candidates-present")
                }
                if (active.any {
                    it.recoveryAction ==
                        CameraAutoHealPolicyEngine.Action.RECALIBRATION_REQUIRED
                }) {
                    add("camera-recalibration-required")
                }
            },
            recoveryActions =
                allDistinct.associate {
                    it.cameraId to it.recoveryAction
                },
        )
    }

    fun select(
        cameras: List<CameraEvidence>,
        requestedCount: Int,
    ): Selection {
        require(requestedCount >= 1)

        val candidates = cameras.asSequence()
            .filter { it.active }
            .distinctBy { it.cameraId }
            .sortedWith(
                compareByDescending<CameraEvidence> { it.health }
                    .thenBy { it.cameraId },
            )
            .take(maximumCameras)
            .toList()

        if (candidates.isEmpty()) {
            return Selection(
                selected = emptyList(),
                rejected = emptyList(),
                marginalGain = emptyMap(),
                objectiveScore = 0.0,
                warnings = listOf("no-active-cameras"),
            )
        }

        val target = min(requestedCount, candidates.size)
        val selected = mutableListOf<CameraEvidence>()
        val gains = linkedMapOf<String, Double>()

        while (selected.size < target) {
            val best = candidates
                .asSequence()
                .filterNot { candidate ->
                    selected.any { it.cameraId == candidate.cameraId }
                }
                .map { candidate ->
                    candidate to marginalUtility(candidate, selected)
                }
                .maxWithOrNull(
                    compareBy<Pair<CameraEvidence, Double>> { it.second }
                        .thenByDescending { it.first.health }
                        .thenBy { it.first.cameraId },
                )
                ?: break

            selected += best.first
            gains[best.first.cameraId] = best.second
        }

        val rejected = candidates.filterNot { camera ->
            selected.any { it.cameraId == camera.cameraId }
        }
        val objective = if (selected.isEmpty()) {
            0.0
        } else {
            (
                selected.map { it.health }.average() * 0.40 +
                    angularDiversity(pairwise(selected)) * 0.30 +
                    sectorCoverage(selected) * 0.18 +
                    roleDiversity(selected) * 0.12
                ).coerceIn(0.0, 1.0)
        }

        val warnings = buildList {
            if (selected.size < target) add("selection-underfilled")
            if (selected.size > 1 &&
                angularDiversity(pairwise(selected)) < 0.40
            ) {
                add("selected-rig-angular-diversity-low")
            }
            if (sectorCoverage(selected) < 0.50) {
                add("selected-rig-sector-coverage-low")
            }
            if (selected.any { it.health < 0.40 }) {
                add("selected-rig-contains-low-health-camera")
            }
        }

        return Selection(
            selected = selected,
            rejected = rejected,
            marginalGain = gains,
            objectiveScore = objective,
            warnings = warnings,
        )
    }

    fun pairwise(cameras: List<CameraEvidence>): List<Pairwise> {
        val output = ArrayList<Pairwise>()
        for (firstIndex in cameras.indices) {
            for (secondIndex in firstIndex + 1 until cameras.size) {
                val first = cameras[firstIndex]
                val second = cameras[secondIndex]
                val baseline = first.positionMeters.distanceTo(
                    second.positionMeters,
                )
                val angle = Math.toDegrees(
                    acos(
                        first.normalizedForward
                            .dot(second.normalizedForward)
                            .coerceIn(-1.0, 1.0),
                    ),
                )
                output += Pairwise(
                    first = first.cameraId,
                    second = second.cameraId,
                    baselineMeters = baseline,
                    angularSeparationDegrees = angle,
                    utility = (
                        baselineQuality(baseline) * 0.52 +
                            (angle / 90.0).coerceIn(0.0, 1.0) * 0.48
                        ).coerceIn(0.0, 1.0),
                )
            }
        }
        return output
    }

    private fun marginalUtility(
        candidate: CameraEvidence,
        selected: List<CameraEvidence>,
    ): Double {
        if (selected.isEmpty()) return candidate.health

        val bestGeometry = selected.maxOf { other ->
            val baseline = candidate.positionMeters.distanceTo(
                other.positionMeters,
            )
            val angle = Math.toDegrees(
                acos(
                    candidate.normalizedForward
                        .dot(other.normalizedForward)
                        .coerceIn(-1.0, 1.0),
                ),
            )
            baselineQuality(baseline) * 0.52 +
                (angle / 90.0).coerceIn(0.0, 1.0) * 0.48
        }

        val candidateSector = candidate.sector ?: inferSector(candidate)
        val sectorNovelty =
            if (selected.any {
                (it.sector ?: inferSector(it)) == candidateSector
            }) {
                0.20
            } else {
                1.0
            }

        val roleNovelty =
            if (selected.any { it.role == candidate.role }) 0.25 else 1.0

        val informationGain =
            candidate.surfaceInformationGain * 0.45 +
                candidate.occlusionCoverage * 0.30 +
                candidate.dynamicParallaxScore * 0.25
        val temporalTrust =
            candidate.temporalTimingHealth

        return (
            bestGeometry * 0.38 +
                sectorNovelty * 0.12 +
                roleNovelty * 0.06 +
                informationGain * 0.18 +
                temporalTrust * 0.08 +
                candidate.health * 0.18
            ).coerceIn(0.0, 1.0)
    }

    private fun angularDiversity(pairs: List<Pairwise>): Double =
        if (pairs.isEmpty()) 0.0 else pairs.map {
            (it.angularSeparationDegrees / 180.0).coerceIn(0.0, 1.0)
        }.average()

    private fun sectorCoverage(cameras: List<CameraEvidence>): Double {
        if (cameras.isEmpty()) return 0.0
        return (
            cameras.map { it.sector ?: inferSector(it) }
                .filter { it != Sector.UNKNOWN }
                .distinct()
                .size
                .toDouble() / 8.0
            ).coerceIn(0.0, 1.0)
    }

    private fun roleDiversity(cameras: List<CameraEvidence>): Double {
        if (cameras.isEmpty()) return 0.0
        return (
            cameras.map { it.role }.distinct().size.toDouble() /
                min(8, cameras.size)
            ).coerceIn(0.0, 1.0)
    }

    private fun redundancy(pairs: List<Pairwise>): Double {
        if (pairs.isEmpty()) return 0.0
        return pairs.count {
            it.angularSeparationDegrees < 15.0 &&
                abs(it.baselineMeters) < 0.25
        }.toDouble() / pairs.size
    }

    private fun baselineQuality(baselineMeters: Double): Double =
        when {
            baselineMeters < 0.05 ->
                (baselineMeters / 0.05 * 0.20).coerceIn(0.0, 0.20)
            baselineMeters < 0.25 ->
                0.20 + (baselineMeters - 0.05) / 0.20 * 0.50
            baselineMeters < 1.0 ->
                0.70 + (baselineMeters - 0.25) / 0.75 * 0.30
            else -> 1.0
        }.coerceIn(0.0, 1.0)

    private fun inferSector(camera: CameraEvidence): Sector {
        val p = camera.positionMeters
        if (p.y > 2.2) return Sector.TOP
        if (p.y < 0.4) return Sector.BOTTOM

        val angle = Math.toDegrees(atan2(p.x, p.z))
        return when {
            angle >= -22.5 && angle < 22.5 -> Sector.FRONT
            angle < 67.5 -> Sector.FRONT_RIGHT
            angle < 112.5 -> Sector.RIGHT
            angle < 157.5 -> Sector.REAR_RIGHT
            angle >= 157.5 || angle < -157.5 -> Sector.REAR
            angle >= -157.5 && angle < -112.5 -> Sector.REAR_LEFT
            angle >= -112.5 && angle < -67.5 -> Sector.LEFT
            else -> Sector.FRONT_LEFT
        }
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
