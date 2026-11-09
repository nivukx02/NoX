package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import kotlin.math.max
import kotlin.math.min

/** Bounded dynamic Gaussian surface refinement. Geometry wins over priors and inferred completion. */
class DynamicGaussianSurfaceRefinementEngine(
    private val config: Config = Config(),
    private val sdfFlowRegularizer: SurfaceSdfFlowRegularizationEngine = SurfaceSdfFlowRegularizationEngine(),
) {
    data class Config(
        val maximumGaussians: Int = 16_384,
        val maximumDeformationMeters: Double = 0.20,
        val minimumConfidence: Double = 0.20,
        val surfaceFlowWeight: Double = 0.55,
        val covarianceInflationOnInference: Double = 1.35,
        val minimumVisibility: Double = 0.12,
        val occlusionPenaltyStrength: Double = 0.72,
        val photometricWeight: Double = 0.10,
        val normalConsistencyWeight: Double = 0.10,
        val neighborhoodRadiusMeters: Double = 0.045,
        val maximumNeighborhoodCandidates: Int = 12,
        val neighborhoodPositionWeight: Double = 0.18,
        val neighborhoodNormalWeight: Double = 0.16,
        val spatialDeduplicationMeters: Double = 0.004,
        val minimumAuthorityForOutput: Double = 0.08,
        val uncertaintyPenaltyMeters: Double = 0.08,
    )

    data class SurfacePoint(
        val id: Long,
        val positionMeters: Vec3,
        val normal: Vec3,
        val covariance: UncertaintyEnvelope,
        val confidence: Double,
        val origin: EvidenceOrigin,
        val flowDeltaMeters: Vec3 = Vec3(0.0,0.0,0.0),
        val signedDistanceDeltaMeters: Double? = null,
        val visibility: Double = 1.0,
        val occlusionRisk: Double = 0.0,
        val photometricConsistency: Double = 1.0,
        val normalConsistency: Double = 1.0,
        val independentCameraCount: Int = 1,
        val timestampNanos: Long = 0L,
        val semanticRegionId: Int = 0,
        val temporalStability: Double = 1.0,
        val silhouetteConsistency: Double = 1.0,
        val motionBoundaryConfidence: Double = 1.0,
    )

    data class Gaussian(
        val id: Long,
        val positionMeters: Vec3,
        val normal: Vec3,
        val covariance: UncertaintyEnvelope,
        val opacity: Double,
        val authority: Double,
        val origin: EvidenceOrigin,
        val cameraSupport: Int = 0,
        val neighborhoodSupport: Int = 0,
        val temporalStability: Double = 0.0,
        val silhouetteConsistency: Double = 0.0,
        val motionBoundaryConfidence: Double = 0.0,
    )

    fun refine(points: List<SurfacePoint>): List<Gaussian> {
        val candidates = points
            .asSequence()
            .filter {
                it.confidence >= config.minimumConfidence &&
                    it.visibility >= config.minimumVisibility &&
                    finite(it.positionMeters) &&
                    finite(it.normal) &&
                    it.normal.norm() > 1e-9
            }
            .sortedWith(
                compareByDescending<SurfacePoint> { authority(it) }
                    .thenBy { it.timestampNanos }
                    .thenBy { it.semanticRegionId }
                    .thenBy { it.id },
            )
            .take(config.maximumGaussians * 2)
            .toList()

        val deduplicated = deduplicate(candidates)
        val spatialIndex = buildSpatialIndex(deduplicated)
        return deduplicated
            .mapNotNull { point ->
                refinePoint(
                    point = point,
                    neighborhood = neighborhood(
                        point = point,
                        spatialIndex = spatialIndex,
                    ),
                )
            }
            .filter { it.authority >= config.minimumAuthorityForOutput }
            .sortedWith(
                compareByDescending<Gaussian> { it.authority }
                    .thenBy { it.id },
            )
            .take(config.maximumGaussians)
            .toList()
    }

    private fun refinePoint(
        point: SurfacePoint,
        neighborhood: List<SurfacePoint>,
    ): Gaussian? {
        val primaryNormal = point.normal.normalized()
        val neighborhoodNormal = robustNeighborhoodNormal(
            primary = primaryNormal,
            neighborhood = neighborhood,
        )
        val flow = point.flowDeltaMeters
            .clampNorm(config.maximumDeformationMeters)

        val regularizedFlow =
            point.signedDistanceDeltaMeters?.let { sdfDelta ->
                sdfFlowRegularizer
                    .regularize(
                        listOf(
                            SurfaceSdfFlowRegularizationEngine.Constraint(
                                id = point.id,
                                positionMeters = point.positionMeters,
                                normal = primaryNormal,
                                flowDeltaMeters = flow,
                                signedDistanceDeltaMeters = sdfDelta,
                                uncertaintyMeters =
                                    kotlin.math.sqrt(
                                        point.covariance
                                            .principalVarianceUpperBound()
                                            .coerceAtLeast(0.0),
                                    ),
                                confidence = point.confidence,
                            ),
                        ),
                    )
                    .singleOrNull()
                    ?.regularizedFlowMeters
            } ?: flow

        val dynamicWeight = (
            point.confidence *
                point.temporalStability *
                point.motionBoundaryConfidence *
                config.surfaceFlowWeight
            ).coerceIn(0.0, 1.0)
        val basePosition =
            point.positionMeters + regularizedFlow * dynamicWeight

        val neighborhoodTarget =
            robustNeighborhoodPosition(
                primary = basePosition,
                neighborhood = neighborhood,
            )
        val neighborhoodWeight = (
            config.neighborhoodPositionWeight *
                neighborhoodStrength(neighborhood.size)
            ).coerceIn(0.0, 0.35)
        val finalPosition =
            basePosition +
                (
                    neighborhoodTarget - basePosition
                ).clampNorm(
                    config.neighborhoodRadiusMeters * 0.5,
                ) * neighborhoodWeight

        val finalNormal =
            (
                primaryNormal *
                    (1.0 - config.neighborhoodNormalWeight.coerceIn(0.0, 0.35)) +
                    neighborhoodNormal *
                    config.neighborhoodNormalWeight.coerceIn(0.0, 0.35)
                ).normalized()

        val inferred =
            point.origin == EvidenceOrigin.LEARNED_PRIOR ||
                point.origin == EvidenceOrigin.GENERATIVE_COMPLETION
        val covariance =
            if (inferred) {
                inflate(point.covariance)
            } else {
                point.covariance
            }

        val authorityScore =
            authority(
                point = point,
                neighborhoodSupport = neighborhood.size,
            )

        if (
            authorityScore < config.minimumAuthorityForOutput ||
                !finite(finalPosition) ||
                !finite(finalNormal) ||
                finalNormal.norm() < 1e-9
        ) {
            return null
        }

        val opacity = (
            authorityScore *
                (0.64 + 0.36 * point.visibility.coerceIn(0.0, 1.0)) *
                (
                    1.0 -
                        config.occlusionPenaltyStrength *
                        point.occlusionRisk.coerceIn(0.0, 1.0)
                ) *
                (
                    1.0 -
                        config.photometricWeight +
                        config.photometricWeight *
                        point.photometricConsistency.coerceIn(0.0, 1.0)
                ) *
                (
                    1.0 -
                        config.normalConsistencyWeight +
                        config.normalConsistencyWeight *
                        point.normalConsistency.coerceIn(0.0, 1.0)
                ) *
                (0.85 + 0.15 * point.silhouetteConsistency) *
                (0.85 + 0.15 * point.temporalStability)
            ).coerceIn(0.0, 1.0)

        return Gaussian(
            id = point.id,
            positionMeters = finalPosition,
            normal = finalNormal,
            covariance = covariance,
            opacity = opacity,
            authority = authorityScore,
            origin = point.origin,
            cameraSupport = point.independentCameraCount,
            neighborhoodSupport = neighborhood.size,
            temporalStability = point.temporalStability,
            silhouetteConsistency = point.silhouetteConsistency,
            motionBoundaryConfidence = point.motionBoundaryConfidence,
        )
    }

    private fun deduplicate(
        points: List<SurfacePoint>,
    ): List<SurfacePoint> {
        if (points.isEmpty()) return emptyList()
        val cellSize = config.spatialDeduplicationMeters
        val cells = LinkedHashMap<GridKey, SurfacePoint>()

        points.forEach { point ->
            val key = GridKey(
                x = kotlin.math.floor(point.positionMeters.x / cellSize).toLong(),
                y = kotlin.math.floor(point.positionMeters.y / cellSize).toLong(),
                z = kotlin.math.floor(point.positionMeters.z / cellSize).toLong(),
                semanticRegionId = point.semanticRegionId,
            )
            val existing = cells[key]
            cells[key] =
                if (
                    existing == null ||
                    authority(point) > authority(existing)
                ) {
                    point
                } else {
                    existing
                }
        }

        return cells.values.toList()
    }

    private data class GridKey(
        val x: Long,
        val y: Long,
        val z: Long,
        val semanticRegionId: Int,
    )

    private data class NeighborhoodKey(
        val x: Long,
        val y: Long,
        val z: Long,
        val semanticRegionId: Int,
    )

    private fun buildSpatialIndex(
        points: List<SurfacePoint>,
    ): Map<NeighborhoodKey, List<SurfacePoint>> {
        val cellSize =
            config.neighborhoodRadiusMeters.coerceAtLeast(1e-4)
        val buckets = LinkedHashMap<NeighborhoodKey, MutableList<SurfacePoint>>()
        points.forEach { point ->
            val key = NeighborhoodKey(
                x = kotlin.math.floor(
                    point.positionMeters.x / cellSize,
                ).toLong(),
                y = kotlin.math.floor(
                    point.positionMeters.y / cellSize,
                ).toLong(),
                z = kotlin.math.floor(
                    point.positionMeters.z / cellSize,
                ).toLong(),
                semanticRegionId = point.semanticRegionId,
            )
            buckets.getOrPut(key) { ArrayList() }.add(point)
        }
        return buckets.mapValues { (_, values) ->
            values.sortedWith(
                compareByDescending<SurfacePoint> { authority(it) }
                    .thenBy { it.id },
            )
        }
    }

    private fun neighborhood(
        point: SurfacePoint,
        spatialIndex: Map<NeighborhoodKey, List<SurfacePoint>>,
    ): List<SurfacePoint> {
        val cellSize =
            config.neighborhoodRadiusMeters.coerceAtLeast(1e-4)
        val center = NeighborhoodKey(
            x = kotlin.math.floor(
                point.positionMeters.x / cellSize,
            ).toLong(),
            y = kotlin.math.floor(
                point.positionMeters.y / cellSize,
            ).toLong(),
            z = kotlin.math.floor(
                point.positionMeters.z / cellSize,
            ).toLong(),
            semanticRegionId = point.semanticRegionId,
        )
        val candidates = ArrayList<Pair<SurfacePoint, Double>>()

        for (dx in -1L..1L) {
            for (dy in -1L..1L) {
                for (dz in -1L..1L) {
                    val key = center.copy(
                        x = center.x + dx,
                        y = center.y + dy,
                        z = center.z + dz,
                    )
                    spatialIndex[key].orEmpty().forEach { candidate ->
                        if (candidate.id == point.id) return@forEach
                        val distance =
                            candidate.positionMeters.distanceTo(
                                point.positionMeters,
                            )
                        if (
                            distance.isFinite() &&
                                distance <= config.neighborhoodRadiusMeters
                        ) {
                            candidates += candidate to distance
                        }
                    }
                }
            }
        }

        return candidates
            .sortedWith(
                compareBy<Pair<SurfacePoint, Double>> { it.second }
                    .thenByDescending { authority(it.first) }
                    .thenBy { it.first.id },
            )
            .take(config.maximumNeighborhoodCandidates)
            .map { it.first }
    }

    private fun robustNeighborhoodPosition(
        primary: Vec3,
        neighborhood: List<SurfacePoint>,
    ): Vec3 {
        if (neighborhood.isEmpty()) return primary
        val candidates = neighborhood + listOf(
            SurfacePoint(
                id = Long.MIN_VALUE,
                positionMeters = primary,
                normal = Vec3(0.0, 0.0, 1.0),
                covariance = UncertaintyEnvelope(1.0, 1.0, 1.0),
                confidence = 1.0,
                origin = EvidenceOrigin.OBSERVED,
            ),
        )

        var estimate = primary
        repeat(2) {
            var sum = Vec3(0.0, 0.0, 0.0)
            var total = 0.0
            candidates.forEach { candidate ->
                val error =
                    candidate.positionMeters.distanceTo(estimate)
                val robust = 1.0 /
                    (1.0 + (
                        error /
                            config.neighborhoodRadiusMeters
                        ).let { it * it })
                val weight =
                    authority(candidate) * robust
                if (
                    weight.isFinite() &&
                        weight > 1e-9
                ) {
                    sum += candidate.positionMeters * weight
                    total += weight
                }
            }
            if (total > 1e-9) {
                estimate = sum / total
            }
        }
        return estimate
    }

    private fun robustNeighborhoodNormal(
        primary: Vec3,
        neighborhood: List<SurfacePoint>,
    ): Vec3 {
        if (neighborhood.isEmpty()) return primary
        var sum = primary
        neighborhood.forEach { candidate ->
            val normal = candidate.normal.normalized()
            val alignment = normal.dot(primary)
                .coerceIn(-1.0, 1.0)
            if (alignment >= 0.0) {
                sum += normal * (
                    authority(candidate) *
                        alignment
                )
            }
        }
        return if (sum.norm() < 1e-9) {
            primary
        } else {
            sum.normalized()
        }
    }

    private fun neighborhoodStrength(count: Int): Double =
        (
            kotlin.math.sqrt(
                min(
                    count,
                    config.maximumNeighborhoodCandidates,
                ).toDouble(),
            ) / kotlin.math.sqrt(8.0)
        ).coerceIn(0.0, 1.0)

    private fun authority(
        point: SurfacePoint,
        neighborhoodSupport: Int = 0,
    ): Double {
        val uncertaintyMeters =
            kotlin.math.sqrt(
                point.covariance
                    .principalVarianceUpperBound()
                    .coerceAtLeast(0.0),
            )
        val uncertaintyScore = (
            1.0 -
                uncertaintyMeters /
                config.uncertaintyPenaltyMeters
            ).coerceIn(0.0, 1.0)
        val cameraScore = (
            kotlin.math.sqrt(
                point.independentCameraCount
                    .coerceAtLeast(1)
                    .toDouble(),
            ) / kotlin.math.sqrt(8.0)
        ).coerceIn(0.0, 1.0)
        val neighborhoodScore =
            neighborhoodStrength(neighborhoodSupport)
        val provenanceScore =
            when (point.origin) {
                EvidenceOrigin.OBSERVED -> 1.0
                EvidenceOrigin.GEOMETRICALLY_CONSTRAINED -> 0.88
                EvidenceOrigin.TEMPORALLY_PROPAGATED -> 0.66
                EvidenceOrigin.LEARNED_PRIOR -> 0.48
                EvidenceOrigin.GENERATIVE_COMPLETION -> 0.32
            }
        return (
            0.30 * point.confidence +
                0.15 * point.covariance.confidence +
                0.11 * point.visibility.coerceIn(0.0, 1.0) +
                0.09 * (1.0 - point.occlusionRisk.coerceIn(0.0, 1.0)) +
                0.07 * point.photometricConsistency.coerceIn(0.0, 1.0) +
                0.07 * point.normalConsistency.coerceIn(0.0, 1.0) +
                0.07 * cameraScore +
                0.05 * point.temporalStability.coerceIn(0.0, 1.0) +
                0.04 * point.silhouetteConsistency.coerceIn(0.0, 1.0) +
                0.03 * point.motionBoundaryConfidence.coerceIn(0.0, 1.0) +
                0.06 * uncertaintyScore +
                0.06 * neighborhoodScore
            ).coerceIn(0.0, 1.0) *
            provenanceScore
    }

    private fun inflate(c: UncertaintyEnvelope) = UncertaintyEnvelope(
        c.xxMeters2 * config.covarianceInflationOnInference,
        c.yyMeters2 * config.covarianceInflationOnInference,
        c.zzMeters2 * config.covarianceInflationOnInference,
        c.xyMeters2 * config.covarianceInflationOnInference,
        c.xzMeters2 * config.covarianceInflationOnInference,
        c.yzMeters2 * config.covarianceInflationOnInference,
        c.confidence * 0.80,
    )

    private fun finite(v: Vec3) = v.x.isFinite() && v.y.isFinite() && v.z.isFinite()
}

private fun Vec3.clampNorm(maxNorm: Double): Vec3 { val n = norm(); return if (n <= maxNorm || n < 1e-12) this else this * (maxNorm / n) }