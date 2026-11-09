package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.Vec3
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Produces placement recommendations for large camera federations.
 *
 * The planner is deterministic and geometry-first. It does not assume a camera can be installed
 * at an arbitrary coordinate; it emits candidate positions and a justification so an operator or
 * deployment controller can decide which physical mounts are actually available.
 */
class AdaptiveCameraPlacementPlanner(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumRecommendations: Int = 96,
        val minimumAzimuthSectors: Int = 16,
        val radialLayers: List<Double> = listOf(1.8, 3.0, 5.0, 8.0),
        val verticalOffsets: List<Double> = listOf(-1.2, 0.0, 1.2),
        val preferredHorizontalDistanceMeters: Double = 3.0,
        val minimumHorizontalBaselineMeters: Double = 0.30,
        val preferredHorizontalBaselineMeters: Double = 0.90,
        val minimumCandidateSeparationMeters: Double = 0.45,
        val maximumCandidateDistanceMeters: Double = 18.0,
        val occlusionHotspotWeight: Double = 0.42,
        val sectorCoverageWeight: Double = 0.28,
        val baselineWeight: Double = 0.18,
        val depthConditioningWeight: Double = 0.12,
    ) {
        init {
            require(maximumRecommendations in 1..96)
            require(minimumAzimuthSectors in 4..64)
            require(radialLayers.isNotEmpty() && radialLayers.all { it > 0.0 })
            require(verticalOffsets.isNotEmpty() && verticalOffsets.all { it.isFinite() })
            require(preferredHorizontalDistanceMeters > 0.0)
            require(minimumHorizontalBaselineMeters > 0.0)
            require(preferredHorizontalBaselineMeters >= minimumHorizontalBaselineMeters)
            require(minimumCandidateSeparationMeters > 0.0)
            require(maximumCandidateDistanceMeters > 0.0)
            listOf(
                occlusionHotspotWeight,
                sectorCoverageWeight,
                baselineWeight,
                depthConditioningWeight,
            ).forEach { require(it in 0.0..1.0) }
        }
    }

    data class ExistingCamera(
        val id: String,
        val positionMeters: Vec3,
        val forward: Vec3,
        val sector: Int,
        val quality: Double = 1.0,
    ) {
        init {
            require(id.isNotBlank())
            require(positionMeters.finite())
            require(forward.finite() && forward.norm() > 1e-9)
            require(sector >= 0)
            require(quality in 0.0..1.0)
        }
    }

    data class OcclusionHotspot(
        val positionMeters: Vec3,
        val radiusMeters: Double,
        val severity: Double,
    ) {
        init {
            require(positionMeters.finite())
            require(radiusMeters > 0.0)
            require(severity in 0.0..1.0)
        }
    }

    data class Request(
        val subjectCenterMeters: Vec3,
        val subjectRadiusMeters: Double,
        val occupiedSectors: Set<Int>,
        val cameras: List<ExistingCamera>,
        val hotspots: List<OcclusionHotspot> = emptyList(),
        val requiredAdditionalCameras: Int,
    ) {
        init {
            require(subjectCenterMeters.finite())
            require(subjectRadiusMeters > 0.0)
            require(occupiedSectors.all { it >= 0 })
            require(requiredAdditionalCameras >= 0)
        }
    }

    data class Placement(
        val positionMeters: Vec3,
        val forward: Vec3,
        val sector: Int,
        val radialDistanceMeters: Double,
        val score: Double,
        val baselineScore: Double,
        val hotspotScore: Double,
        val sectorCoverageScore: Double,
        val depthConditioningScore: Double,
        val reason: String,
    )

    data class Result(
        val placements: List<Placement>,
        val uncoveredSectorCount: Int,
        val estimatedCoverageGain: Double,
        val estimatedDepthConditioningGain: Double,
        val warnings: List<String>,
    )

    fun plan(request: Request): Result {
        if (request.requiredAdditionalCameras == 0) {
            return Result(
                placements = emptyList(),
                uncoveredSectorCount = 0,
                estimatedCoverageGain = 0.0,
                estimatedDepthConditioningGain = 0.0,
                warnings = emptyList(),
            )
        }

        val targetCount =
            request.requiredAdditionalCameras
                .coerceIn(
                    1,
                    config.maximumRecommendations,
                )

        val occupied =
            request.cameras
                .map { it.sector }
                .toSet() +
                request.occupiedSectors

        val candidates =
            buildCandidates(request)
                .sortedWith(
                    compareByDescending<Placement> { it.score }
                        .thenBy { it.sector }
                        .thenBy { it.radialDistanceMeters },
                )

        val selected = ArrayList<Placement>(targetCount)
        for (candidate in candidates) {
            if (selected.size >= targetCount) break

            val separated =
                selected.all {
                    it.positionMeters.distanceTo(
                        candidate.positionMeters,
                    ) >= config.minimumCandidateSeparationMeters
                }

            if (!separated) continue

            val sectorRepeated =
                selected.count {
                    it.sector == candidate.sector
                }

            if (
                sectorRepeated > 0 &&
                occupied.isNotEmpty() &&
                candidate.sector !in uncoveredSectors(occupied)
            ) {
                continue
            }

            selected += candidate
        }

        if (selected.size < targetCount) {
            candidates.forEach { candidate ->
                if (selected.size >= targetCount) return@forEach

                val separated =
                    selected.all {
                        it.positionMeters.distanceTo(
                            candidate.positionMeters,
                        ) >=
                            config.minimumCandidateSeparationMeters * 0.72
                    }

                if (separated && candidate !in selected) {
                    selected += candidate
                }
            }
        }

        val uncovered =
            uncoveredSectors(
                occupied +
                    selected.map { it.sector },
            )

        val coverageGain =
            (
                selected
                    .map { it.sectorCoverageScore }
                    .averageOrZero()
            ).coerceIn(0.0, 1.0)

        val depthGain =
            (
                selected
                    .map { it.depthConditioningScore }
                    .averageOrZero()
            ).coerceIn(0.0, 1.0)

        val warnings =
            buildList {
                if (selected.size < targetCount) {
                    add("placement-candidate-space-limited")
                }
                if (uncovered.isNotEmpty()) {
                    add("subject-sector-coverage-remains-incomplete")
                }
                if (request.hotspots.isNotEmpty() &&
                    selected.none { it.hotspotScore >= 0.65 }
                ) {
                    add("hotspot-coverage-gain-limited")
                }
                if (
                    selected.none {
                        it.baselineScore >= 0.80
                    }
                ) {
                    add("stereo-baseline-gain-limited")
                }
            }

        return Result(
            placements = selected,
            uncoveredSectorCount = uncovered.size,
            estimatedCoverageGain = coverageGain,
            estimatedDepthConditioningGain = depthGain,
            warnings = warnings.distinct(),
        )
    }

    private fun buildCandidates(
        request: Request,
    ): List<Placement> {
        val candidates = ArrayList<Placement>()
        val center = request.subjectCenterMeters

        config.radialLayers.forEachIndexed { layerIndex, radius ->
            if (radius > config.maximumCandidateDistanceMeters) return@forEachIndexed

            config.verticalOffsets.forEach { vertical ->
                for (sector in 0 until config.minimumAzimuthSectors) {
                    val theta =
                        2.0 *
                            PI *
                            sector.toDouble() /
                            config.minimumAzimuthSectors.toDouble()

                    val position =
                        center +
                            Vec3(
                                radius * cos(theta),
                                vertical,
                                radius * sin(theta),
                            )

                    if (
                        request.cameras.any {
                            it.positionMeters.distanceTo(position) <
                                config.minimumCandidateSeparationMeters
                        }
                    ) {
                        continue
                    }

                    val forward =
                        (center - position)
                            .normalized()

                    val hotspotScore =
                        hotspotScore(
                            position,
                            request.hotspots,
                        )

                    val sectorScore =
                        sectorCoverageScore(
                            sector,
                            request,
                        )

                    val baselineScore =
                        baselineScore(
                            position,
                            request,
                        )

                    val depthConditioning =
                        depthConditioningScore(
                            position,
                            request,
                        )

                    val layerBonus =
                        when (layerIndex) {
                            0 -> 0.10
                            1 -> 0.08
                            2 -> 0.03
                            else -> 0.0
                        }

                    val score =
                        (
                            config.occlusionHotspotWeight *
                                hotspotScore +
                                config.sectorCoverageWeight *
                                    sectorScore +
                                config.baselineWeight *
                                    baselineScore +
                                config.depthConditioningWeight *
                                    depthConditioning +
                                layerBonus
                        )
                            .coerceIn(0.0, 1.0)

                    val reason =
                        when {
                            hotspotScore >= 0.70 ->
                                "occlusion-hotspot"
                            sectorScore >= 0.80 &&
                                sector !in request.occupiedSectors ->
                                "uncovered-sector"
                            baselineScore >= 0.80 ->
                                "baseline-strengthening"
                            depthConditioning >= 0.80 ->
                                "depth-conditioning"
                            else ->
                                "coverage-balancing"
                        }

                    candidates +=
                        Placement(
                            positionMeters = position,
                            forward = forward,
                            sector = sector,
                            radialDistanceMeters = radius,
                            score = score,
                            baselineScore = baselineScore,
                            hotspotScore = hotspotScore,
                            sectorCoverageScore = sectorScore,
                            depthConditioningScore = depthConditioning,
                            reason = reason,
                        )
                }
            }
        }

        return candidates
    }

    private fun hotspotScore(
        position: Vec3,
        hotspots: List<OcclusionHotspot>,
    ): Double {
        if (hotspots.isEmpty()) return 0.0

        return hotspots
            .sumOf { hotspot ->
                val distance =
                    position.distanceTo(
                        hotspot.positionMeters,
                    )
                val spatial =
                    (
                        1.0 -
                            distance /
                                max(
                                    hotspot.radiusMeters *
                                        5.0,
                                    0.001,
                                )
                    ).coerceIn(0.0, 1.0)

                spatial *
                    hotspot.severity
            )
            .coerceIn(0.0, 1.0)
    }

    private fun sectorCoverageScore(
        sector: Int,
        request: Request,
    ): Double {
        val missing =
            sector !in request.occupiedSectors
        if (missing) return 1.0

        val neighboringMissing =
            (listOf(
                (sector - 1 + config.minimumAzimuthSectors) %
                    config.minimumAzimuthSectors,
                (sector + 1) %
                    config.minimumAzimuthSectors,
            ))
                .count {
                    it !in request.occupiedSectors
                }

        return (
            0.20 +
                neighboringMissing *
                    0.35
            )
                .coerceIn(0.0, 1.0)
    }

    private fun baselineScore(
        position: Vec3,
        request: Request,
    ): Double {
        if (request.cameras.isEmpty()) return 0.75

        val nearest =
            request.cameras
                .map {
                    it.positionMeters.distanceTo(
                        position,
                    )
                }
                .minOrNull()
                ?: return 0.0

        val preferred =
            config.preferredHorizontalBaselineMeters

        val normalized =
            (
                nearest /
                    preferred
            ).coerceIn(0.0, 1.0)

        if (
            nearest <
                config.minimumHorizontalBaselineMeters
        ) {
            return 0.10
        }

        return normalized
    }

    private fun depthConditioningScore(
        position: Vec3,
        request: Request,
    ): Double {
        if (request.cameras.size < 2) return 0.45

        val radial =
            position.distanceTo(
                request.subjectCenterMeters,
            )

        val preferred =
            config.preferredHorizontalDistanceMeters

        val distanceScore =
            (
                1.0 -
                    abs(radial - preferred) /
                        max(
                            preferred,
                            1e-6,
                        )
            )
                .coerceIn(0.0, 1.0)

        val angularSeparation =
            request.cameras
                .map {
                    angularSeparationDegrees(
                        (it.positionMeters - request.subjectCenterMeters)
                            .normalized(),
                        (position - request.subjectCenterMeters)
                            .normalized(),
                    )
                }
                .averageOrZero()

        val angularScore =
            (
                angularSeparation /
                    90.0
            ).coerceIn(0.0, 1.0)

        return (
            0.55 * distanceScore +
                0.45 * angularScore
            ).coerceIn(0.0, 1.0)
    }

    private fun uncoveredSectors(
        occupied: Set<Int>,
    ): Set<Int> =
        (0 until config.minimumAzimuthSectors)
            .filter { it !in occupied }
            .toSet()

    private fun angularSeparationDegrees(
        first: Vec3,
        second: Vec3,
    ): Double =
        Math.toDegrees(
            kotlin.math.acos(
                first.normalized()
                    .dot(second.normalized())
                    .coerceIn(-1.0, 1.0),
            ),
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun Vec3.finite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()
}

