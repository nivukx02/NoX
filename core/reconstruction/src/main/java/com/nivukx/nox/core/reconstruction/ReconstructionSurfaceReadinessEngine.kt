package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DenseMeshQuality
import kotlin.math.exp
import kotlin.math.max

/**
 * Converts dense-surface evidence into a single auditable production gate.
 *
 * This layer does not reconstruct geometry and does not smooth away bad evidence. It consumes the
 * independent geometry, temporal, uncertainty, reprojection, view-support, scale, lighting and
 * occlusion signals produced by the existing reconstruction stack and determines whether the
 * current surface is safe to present as production-grade output.
 */
class ReconstructionSurfaceReadinessEngine(
    private val config: Config = Config(),
) {
    enum class Band {
        UNKNOWN,
        BLOCKED,
        DEGRADED,
        READY,
    }

    data class Config(
        val readyThreshold: Double = 0.78,
        val degradedThreshold: Double = 0.55,
        val geometryBlockThreshold: Double = 0.25,
        val uncertaintyBlockMultiplier: Double = 4.0,
        val reprojectionBlockMultiplier: Double = 4.0,
        val minimumVertexCount: Int = 32,
        val minimumTriangleCount: Int = 16,
    ) {
        init {
            require(readyThreshold in 0.0..1.0)
            require(degradedThreshold in 0.0..readyThreshold)
            require(geometryBlockThreshold in 0.0..degradedThreshold)
            require(uncertaintyBlockMultiplier > 1.0)
            require(reprojectionBlockMultiplier > 1.0)
            require(minimumVertexCount >= 3)
            require(minimumTriangleCount >= 1)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val meshQuality: DenseMeshQuality?,
        val surfaceConfidence: Double?,
        val temporalConfidence: Double?,
        val uncertaintyP95Meters: Double?,
        val uncertaintyBudgetMeters: Double,
        val reprojectionErrorPx: Double?,
        val reprojectionBudgetPx: Double,
        val supportingViewConfidence: Double?,
        val scaleConfidence: Double?,
        val metricScaleAvailable: Boolean = true,
        val lightingConfidence: Double? = null,
        val occlusionConfidence: Double? = null,
        val upstreamConfidence: Double? = null,
        val canonicalSurfaceConfidence: Double = 1.0,
        val canonicalSurfaceAccepted: Boolean = true,
        val canonicalCorrespondenceCoverage: Double = 1.0,
        val canonicalEdgeStretchScore: Double = 1.0,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(uncertaintyBudgetMeters.isFinite() && uncertaintyBudgetMeters > 0.0)
            require(reprojectionBudgetPx.isFinite() && reprojectionBudgetPx > 0.0)
            listOfNotNull(
                surfaceConfidence,
                temporalConfidence,
                supportingViewConfidence,
                scaleConfidence,
                lightingConfidence,
                occlusionConfidence,
                upstreamConfidence,
                canonicalSurfaceConfidence,
                canonicalCorrespondenceCoverage,
                canonicalEdgeStretchScore,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            uncertaintyP95Meters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            reprojectionErrorPx?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Breakdown(
        val geometry: Double,
        val surface: Double,
        val temporal: Double,
        val uncertainty: Double,
        val reprojection: Double,
        val viewSupport: Double,
        val scale: Double,
        val lighting: Double,
        val occlusion: Double,
        val upstream: Double,
        val canonicalSurface: Double,
    ) {
        val finalScore: Double
            get() =
                weightedMean(
                    listOf(
                        geometry to 0.18,
                        surface to 0.16,
                        temporal to 0.16,
                        uncertainty to 0.16,
                        reprojection to 0.10,
                        viewSupport to 0.08,
                        scale to 0.05,
                        lighting to 0.05,
                        occlusion to 0.04,
                        upstream to 0.02,
                        canonicalSurface to 0.10,
                    ),
                )

        val weakestCriticalSignal: Double
            get() = minOf(
                geometry,
                surface,
                temporal,
                uncertainty,
                reprojection,
                canonicalSurface,
            )

        private companion object {
            fun weightedMean(values: List<Pair<Double, Double>>): Double {
                val denominator = values.sumOf { it.second }.coerceAtLeast(1e-9)
                return values.sumOf { (value, weight) ->
                    value.coerceIn(0.0, 1.0) * weight
                } / denominator
            }
        }
    }

    data class Result(
        val timestampNanos: Long,
        val band: Band,
        val accepted: Boolean,
        val score: Double,
        val breakdown: Breakdown,
        val blockers: List<String>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val isOperatorReady: Boolean
            get() = band == Band.READY && accepted
    }

    fun evaluate(input: Input): Result {
        val quality = input.meshQuality
        if (quality == null) {
            return Result(
                timestampNanos = input.timestampNanos,
                band = Band.UNKNOWN,
                accepted = false,
                score = 0.0,
                breakdown = Breakdown(
                    geometry = 0.0,
                    surface = 0.0,
                    temporal = 0.0,
                    uncertainty = 0.0,
                    reprojection = 0.0,
                    viewSupport = normalizeOptional(input.supportingViewConfidence),
                    scale = normalizeOptional(input.scaleConfidence),
                    lighting = normalizeOptional(input.lightingConfidence),
                    occlusion = normalizeOptional(input.occlusionConfidence),
                    upstream = normalizeOptional(input.upstreamConfidence),
                    canonicalSurface =
                        input.canonicalSurfaceConfidence.coerceIn(0.0, 1.0),
                ),
                blockers = listOf("geometry evidence is unavailable"),
                warnings = missingEvidenceWarnings(input),
                recommendations = listOf(
                    "Wait for a valid dense surface observation before presenting production geometry.",
                ),
            )
        }

        val geometry = geometryScore(quality)
        val surface = normalizeOptional(input.surfaceConfidence)
        val temporal = temporalScore(
            quality = quality,
            external = input.temporalConfidence,
        )
        val uncertainty = uncertaintyScore(
            p95 = input.uncertaintyP95Meters,
            budget = input.uncertaintyBudgetMeters,
        )
        val reprojection = reprojectionScore(
            errorPx = input.reprojectionErrorPx ?: quality.meanReprojectionErrorPx,
            budgetPx = input.reprojectionBudgetPx,
        )
        val viewSupport = normalizeOptional(input.supportingViewConfidence)
        val scale = normalizeOptional(input.scaleConfidence)
        val lighting = normalizeOptional(input.lightingConfidence)
        val occlusion = normalizeOptional(input.occlusionConfidence)
        val upstream = normalizeOptional(input.upstreamConfidence)
        val canonicalSurface =
            (
                input.canonicalSurfaceConfidence * 0.62 +
                    input.canonicalCorrespondenceCoverage * 0.20 +
                    input.canonicalEdgeStretchScore * 0.18
                ).coerceIn(0.0, 1.0)

        val breakdown = Breakdown(
            geometry = geometry,
            surface = surface,
            temporal = temporal,
            uncertainty = uncertainty,
            reprojection = reprojection,
            viewSupport = viewSupport,
            scale = scale,
            lighting = lighting,
            occlusion = occlusion,
            upstream = upstream,
            canonicalSurface = canonicalSurface,
        )

        val blockers = buildList {
            if (quality.vertexCount < config.minimumVertexCount) {
                add("geometry support is below production minimum")
            }
            if (quality.triangleCount < config.minimumTriangleCount) {
                add("surface topology is below production minimum")
            }
            if (geometry < config.geometryBlockThreshold) {
                add("geometry confidence is below the hard quality gate")
            }
            if (
                surface < config.geometryBlockThreshold ||
                    quality.geometricConfidence <
                    config.geometryBlockThreshold
            ) {
                add("surface geometry evidence is below the hard quality gate")
            }
            if (
                geometry <
                    config.geometryBlockThreshold ||
                quality.geometricConfidence <
                    config.geometryBlockThreshold
            ) {
                add("raw geometric confidence is below the hard quality gate")
            }
            if (
                input.uncertaintyP95Meters != null &&
                input.uncertaintyP95Meters >
                    input.uncertaintyBudgetMeters *
                    config.uncertaintyBlockMultiplier
            ) {
                add("uncertainty P95 exceeds the production block threshold")
            }
            if (
                input.reprojectionErrorPx != null &&
                input.reprojectionErrorPx >
                    input.reprojectionBudgetPx *
                    config.reprojectionBlockMultiplier
            ) {
                add("reprojection error exceeds the production block threshold")
            }
            if (
                quality.geometricConfidence < config.geometryBlockThreshold &&
                quality.normalCoverage < 0.40
            ) {
                add("surface normal coverage is insufficient")
            }
            if (!input.canonicalSurfaceAccepted) {
                add("canonical surface deformation evidence was rejected")
            }
            if (
                input.canonicalCorrespondenceCoverage <
                    config.degradedThreshold
            ) {
                add("canonical surface correspondence coverage is insufficient")
            }
        }

        val warnings = buildList {
            if (!quality.reprojectionEvidenceAvailable) {
                add("reprojection evidence is not independently measured")
            }
            if (!quality.temporalEvidenceAvailable) {
                add("temporal surface evidence is incomplete")
            }
            if (!quality.watertightnessEvidenceAvailable) {
                add("topology closure evidence is incomplete")
            }
            if (surface < 0.65) {
                add("surface confidence is degraded")
            }
            if (temporal < 0.65) {
                add("temporal surface stability is degraded")
            }
            if (uncertainty < 0.65) {
                add("uncertainty is elevated")
            }
            if (reprojection < 0.65) {
                add("reprojection consistency is degraded")
            }
            if (viewSupport < 0.60) {
                add("multi-view support is limited")
            }
            if (!input.metricScaleAvailable) {
                add("metric scale is unavailable; geometry remains relative or prior-scaled")
            }
            if (lighting < 0.60) {
                add("lighting evidence may be insufficient for stable surface appearance")
            }
            if (canonicalSurface < 0.65) {
                add("canonical surface deformation confidence is degraded")
            }
        }.distinct()

        val recommendations = buildList {
            if (viewSupport < 0.60) {
                add("Increase camera coverage or subject parallax before committing the take.")
            }
            if (uncertainty < 0.65) {
                add("Increase depth evidence density and preserve calibrated scale where available.")
            }
            if (reprojection < 0.65) {
                add("Re-check calibration and synchronization before accepting surface geometry.")
            }
            if (temporal < 0.65) {
                add("Keep acquisition continuous to rebuild temporal surface history.")
            }
            if (!input.metricScaleAvailable) {
                add("Add calibrated multi-view or metric depth evidence when absolute dimensions matter.")
            }
            if (lighting < 0.60) {
                add("Prefer stable illumination and avoid large exposure changes during the take.")
            }
            if (canonicalSurface < 0.65) {
                add("Validate canonical temporal deformation before propagating mesh changes across frames.")
            }
            if (isEmpty()) {
                add("Surface evidence is internally consistent for the current capture conditions.")
            }
        }

        val missingCriticalEvidence =
            input.surfaceConfidence == null ||
                input.temporalConfidence == null ||
                input.uncertaintyP95Meters == null ||
                input.reprojectionErrorPx == null
 ||
            (!input.canonicalSurfaceAccepted &&
                input.canonicalSurfaceConfidence < 1.0)

        val score = breakdown.finalScore
        val band = when {
            blockers.isNotEmpty() -> Band.BLOCKED
            missingCriticalEvidence && score < config.readyThreshold -> Band.UNKNOWN
            score >= config.readyThreshold &&
                breakdown.weakestCriticalSignal >= 0.62 ->
                Band.READY
            score >= config.degradedThreshold ->
                Band.DEGRADED
            else ->
                Band.BLOCKED
        }

        return Result(
            timestampNanos = input.timestampNanos,
            band = band,
            accepted =
                band == Band.READY ||
                    (band == Band.DEGRADED && breakdown.weakestCriticalSignal >= 0.45),
            score = score,
            breakdown = breakdown,
            blockers = blockers,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun geometryScore(
        quality: DenseMeshQuality,
    ): Double =
        (
            quality.geometricConfidence.coerceIn(0.0, 1.0) * 0.50 +
                quality.normalCoverage.coerceIn(0.0, 1.0) * 0.24 +
                quality.watertightnessScore.coerceIn(0.0, 1.0) * 0.16 +
                topologyCountScore(quality.vertexCount, quality.triangleCount) * 0.10
        ).coerceIn(0.0, 1.0)

    private fun temporalScore(
        quality: DenseMeshQuality,
        external: Double?,
    ): Double {
        val internal = quality.temporalStability.coerceIn(0.0, 1.0)
        val externalScore = external ?: internal
        return (internal * 0.70 + externalScore * 0.30)
            .coerceIn(0.0, 1.0)
    }

    private fun uncertaintyScore(
        p95: Double?,
        budget: Double,
    ): Double {
        if (p95 == null) return 0.0
        if (!p95.isFinite() || p95 < 0.0) return 0.0
        return exp(-p95 / max(budget, 1e-6))
            .coerceIn(0.0, 1.0)
    }

    private fun reprojectionScore(
        errorPx: Double,
        budgetPx: Double,
    ): Double {
        if (!errorPx.isFinite() || errorPx < 0.0) return 0.0
        return exp(-errorPx / max(budgetPx, 1e-6))
            .coerceIn(0.0, 1.0)
    }

    private fun topologyCountScore(
        vertices: Int,
        triangles: Int,
    ): Double {
        val vertexScore =
            (vertices.toDouble() / config.minimumVertexCount.toDouble())
                .coerceIn(0.0, 1.0)
        val triangleScore =
            (triangles.toDouble() / config.minimumTriangleCount.toDouble())
                .coerceIn(0.0, 1.0)
        return (vertexScore * 0.55 + triangleScore * 0.45)
            .coerceIn(0.0, 1.0)
    }

    private fun normalizeOptional(value: Double?): Double =
        value?.coerceIn(0.0, 1.0) ?: 0.50

    private fun missingEvidenceWarnings(input: Input): List<String> =
        buildList {
            if (input.surfaceConfidence == null) {
                add("surface confidence evidence is unavailable")
            }
            if (input.temporalConfidence == null) {
                add("temporal confidence evidence is unavailable")
            }
            if (input.uncertaintyP95Meters == null) {
                add("uncertainty P95 is unavailable")
            }
            if (input.reprojectionErrorPx == null) {
                add("reprojection error is unavailable")
            }
        }
}
