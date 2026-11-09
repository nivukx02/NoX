package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ProductionCaptureProfile
import kotlin.math.max
import kotlin.math.min

/**
 * Adaptive reconstruction-detail authority.
 *
 * The renderer and reconstruction engines should spend compute on the geometry that materially
 * improves metric evidence. This policy turns uncertainty, motion dynamics, camera coverage,
 * thermal headroom, memory pressure and GPU latency into a deterministic detail envelope.
 *
 * It never increases quality above a declared profile's physical or computational limits and never
 * uses visual-detail language as a proxy for metric accuracy.
 */
class AdaptiveReconstructionDetailEngine(
    private val config: Config = Config(),
) {
    enum class Tier {
        REFERENCE,
        ULTRA,
        HIGH,
        BALANCED,
        SAFE_HOLD,
    }

    data class Config(
        val referenceUncertaintyMm: Double = 1.2,
        val ultraUncertaintyMm: Double = 4.0,
        val highUncertaintyMm: Double = 12.0,
        val maximumGpuLatencyMs: Double = 28.0,
        val maximumDynamicStress: Double = 0.92,
        val minimumThermalHeadroom: Double = 0.22,
        val minimumMemoryHeadroom: Double = 0.20,
        val minimumRigConfidence: Double = 0.42,
        val uncertaintyWeight: Double = 0.30,
        val geometryWeight: Double = 0.18,
        val thermalWeight: Double = 0.14,
        val memoryWeight: Double = 0.10,
        val gpuWeight: Double = 0.10,
        val dynamicsWeight: Double = 0.08,
        val viewWeight: Double = 0.10,
    ) {
        init {
            require(referenceUncertaintyMm > 0.0)
            require(ultraUncertaintyMm > referenceUncertaintyMm)
            require(highUncertaintyMm > ultraUncertaintyMm)
            require(maximumGpuLatencyMs > 0.0)
            require(maximumDynamicStress in 0.0..1.0)
            require(minimumThermalHeadroom in 0.0..1.0)
            require(minimumMemoryHeadroom in 0.0..1.0)
            require(minimumRigConfidence in 0.0..1.0)
            val sum = (
                uncertaintyWeight + geometryWeight + thermalWeight +
                    memoryWeight + gpuWeight + dynamicsWeight + viewWeight
                )
            require(sum > 0.0)
        }
    }

    data class Input(
        val profile: ProductionCaptureProfile,
        val p95UncertaintyMeters: Double,
        val geometricObservability: Double,
        val thermalHeadroom: Double,
        val memoryHeadroom: Double,
        val gpuLatencyMs: Double,
        val dynamicStress: Double,
        val rigConfidence: Double,
        val viewDiversity: Double,
        val captureTimingConfidence: Double,
    ) {
        init {
            require(p95UncertaintyMeters.isFinite() && p95UncertaintyMeters >= 0.0)
            listOf(
                geometricObservability,
                thermalHeadroom,
                memoryHeadroom,
                dynamicStress,
                rigConfidence,
                viewDiversity,
                captureTimingConfidence,
            ).forEach { require(it in 0.0..1.0) }
            require(gpuLatencyMs >= 0.0 || gpuLatencyMs.isNaN())
        }
    }

    data class RegionInput(
        val regionId: Long,
        val p95UncertaintyMeters: Double,
        val geometricObservability: Double,
        val dynamicStress: Double,
        val viewDiversity: Double,
        val semanticImportance: Double,
        val curvatureScore: Double,
        val actorPresent: Boolean = true,
    ) {
        init {
            require(regionId >= 0L)
            require(p95UncertaintyMeters.isFinite() && p95UncertaintyMeters >= 0.0)
            require(
                listOf(
                    geometricObservability,
                    dynamicStress,
                    viewDiversity,
                    semanticImportance,
                    curvatureScore,
                ).all { it.isFinite() && it in 0.0..1.0 },
            )
        }
    }

    data class RegionBudget(
        val regionId: Long,
        val detailWeight: Double,
        val relativeVoxelScale: Double,
        val relativeVertexAllocation: Double,
        val tier: Tier,
        val reasons: List<String>,
    )

    /**
     * Computes a spatial detail field without changing the global resource ceiling.
     * High curvature, actor presence and semantic importance preserve detail; high uncertainty,
     * low observability and dynamic instability reduce allocation.
     */
    fun evaluateRegions(
        input: Input,
        regions: List<RegionInput>,
    ): List<RegionBudget> {
        if (regions.isEmpty()) return emptyList()
        val global = evaluate(input)
        val weights = regions.associateWith { region ->
            val certainty = (
                1.0 - (
                    region.p95UncertaintyMeters /
                        input.profile.maxMeanUncertaintyMeters
                ).coerceIn(0.0, 2.0) * 0.55
            ).coerceIn(0.0,1.0)
            (
                certainty * 0.30 +
                    region.geometricObservability * 0.16 +
                    region.viewDiversity * 0.12 +
                    (1.0 - region.dynamicStress) * 0.10 +
                    region.semanticImportance * 0.14 +
                    region.curvatureScore * 0.12 +
                    if (region.actorPresent) 0.06 else 0.0
                ).coerceIn(0.02,1.0)
        }
        val weightSum = weights.values.sum().coerceAtLeast(1e-9)

        regions.sortedBy { it.regionId }.map { region ->
            val normalized = (weights[region]!! / weightSum * regions.size)
                .coerceIn(0.20, 2.50)
            val weight = (
                global.score * 0.65 +
                    weights[region]!! * 0.35
                ).coerceIn(0.0,1.0)
            val voxelScale = (
                global.effectiveVoxelSizeMeters /
                    input.profile.tsdfVoxelSizeMeters
            )
            val tier = when {
                weight >= 0.84 && region.curvatureScore >= 0.70 &&
                    region.geometricObservability >= 0.75 -> Tier.REFERENCE
                weight >= 0.68 -> Tier.ULTRA
                weight >= 0.52 -> Tier.HIGH
                weight >= 0.30 -> Tier.BALANCED
                else -> Tier.SAFE_HOLD
            }
            val tierFactor = when (tier) {
                Tier.REFERENCE -> 1.00
                Tier.ULTRA -> 0.82
                Tier.HIGH -> 0.66
                Tier.BALANCED -> 0.48
                Tier.SAFE_HOLD -> 0.20
            }
            RegionBudget(
                regionId = region.regionId,
                detailWeight = weight,
                relativeVoxelScale = (voxelScale / tierFactor).coerceIn(1.0,8.0),
                relativeVertexAllocation = (
                    normalized *
                        tierFactor *
                        (0.60 + 0.40 * input.memoryHeadroom)
                ).coerceIn(0.03,4.0),
                tier = tier,
                reasons = buildList {
                    if (region.curvatureScore >= 0.70) add("high-curvature")
                    if (region.semanticImportance >= 0.70) add("high-semantic-value")
                    if (region.actorPresent) add("actor-region")
                    if (region.p95UncertaintyMeters >
                        input.profile.maxMeanUncertaintyMeters) {
                        add("uncertainty-reduced")
                    }
                    if (region.geometricObservability < 0.55) add("observability-reduced")
                    if (region.dynamicStress > 0.75) add("dynamic-stress-reduced")
                }.distinct(),
            )
        }
    }

    data class Budget(
        val tier: Tier,
        val effectiveVoxelSizeMeters: Double,
        val effectiveTruncationMeters: Double,
        val meshVertexBudget: Int,
        val denseMeshHz: Int,
        val faceMeshHz: Int,
        val eyeTrackingHz: Int,
        val handTrackingHz: Int,
        val bundleAdjustmentIterations: Int,
        val score: Double,
        val preserveMetricEvidence: Boolean,
        val holdOutput: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(input: Input): Budget {
        val uncertaintyMm = input.p95UncertaintyMeters * 1000.0

        val uncertaintyScore = inverse(
            uncertaintyMm,
            config.highUncertaintyMm,
            config.referenceUncertaintyMm,
        )
        val geometryScore = input.geometricObservability
        val thermalScore = input.thermalHeadroom
        val memoryScore = input.memoryHeadroom
        val gpuScore = inverse(
            input.gpuLatencyMs.takeIf { it.isFinite() }
                ?: config.maximumGpuLatencyMs,
            config.maximumGpuLatencyMs * 0.45,
            config.maximumGpuLatencyMs,
        )
        val dynamicsScore = (
            1.0 - input.dynamicStress * 0.70 +
                input.captureTimingConfidence * 0.30
        ).coerceIn(0.0, 1.0)
        val viewScore = (
            input.viewDiversity * 0.70 +
                input.rigConfidence * 0.20 +
                input.geometricObservability * 0.10
        ).coerceIn(0.0, 1.0)

        val score = (
            uncertaintyScore * config.uncertaintyWeight +
                geometryScore * config.geometryWeight +
                thermalScore * config.thermalWeight +
                memoryScore * config.memoryWeight +
                gpuScore * config.gpuWeight +
                dynamicsScore * config.dynamicsWeight +
                viewScore * config.viewWeight
            ).coerceIn(0.0, 1.0)

        val hardHold =
            uncertaintyMm > max(
                input.profile.maxMeanUncertaintyMeters * 1000.0 * 2.5,
                config.highUncertaintyMm * 2.0,
            ) ||
                thermalScore < config.minimumThermalHeadroom ||
                memoryScore < config.minimumMemoryHeadroom ||
                input.rigConfidence < config.minimumRigConfidence ||
                input.dynamicStress > config.maximumDynamicStress ||
                input.captureTimingConfidence < 0.30

        val tier = when {
            hardHold -> Tier.SAFE_HOLD
            uncertaintyMm <= config.referenceUncertaintyMm &&
                score >= 0.82 ->
                Tier.REFERENCE
            uncertaintyMm <= config.ultraUncertaintyMm &&
                score >= 0.68 ->
                Tier.ULTRA
            uncertaintyMm <= config.highUncertaintyMm &&
                score >= 0.54 ->
                Tier.HIGH
            else ->
                Tier.BALANCED
        }

        val tierScale = when (tier) {
            Tier.REFERENCE -> 1.00
            Tier.ULTRA -> 0.86
            Tier.HIGH -> 0.70
            Tier.BALANCED -> 0.50
            Tier.SAFE_HOLD -> 0.22
        }

        val voxelScale = (
            1.0 / tierScale.coerceAtLeast(0.22)
        ).coerceIn(1.0, 4.5)

        val baseVoxel = input.profile.tsdfVoxelSizeMeters
        val voxel = (
            baseVoxel * voxelScale
        ).coerceAtMost(0.080)
            .coerceAtLeast(0.001)

        val truncation = max(
            input.profile.tsdfTruncationMeters,
            voxel * 4.0,
        )

        val vertexScale = (
            tierScale *
                (
                    0.70 +
                        0.30 * input.memoryHeadroom
                ) *
                (
                    0.72 +
                        0.28 * input.gpuLatencyMs.let {
                            inverse(
                                it.takeIf(Double::isFinite)
                                    ?: config.maximumGpuLatencyMs,
                                config.maximumGpuLatencyMs * 0.45,
                                config.maximumGpuLatencyMs,
                            )
                        }
                )
            ).coerceIn(0.12, 1.0)

        val meshBudget = (
            input.profile.meshVertexBudget *
                vertexScale
            ).toInt().coerceIn(4_000, input.profile.meshVertexBudget)

        val frequencyScale = (
            tierScale *
                (0.78 + 0.22 * input.thermalHeadroom) *
                (0.72 + 0.28 * input.memoryHeadroom)
            ).coerceIn(0.12, 1.0)

        val denseMeshHz =
            max(1, (input.profile.denseMeshHz * frequencyScale).toInt())
        val faceMeshHz =
            max(1, (input.profile.faceMeshHz * frequencyScale).toInt())
        val eyeHz =
            max(1, (input.profile.eyeTrackingHz * frequencyScale).toInt())
        val handHz =
            max(1, (input.profile.handTrackingHz * frequencyScale).toInt())
        val bundleIterations =
            max(
                1,
                (input.profile.bundleAdjustmentIterations *
                    (
                        0.52 +
                            0.48 * tierScale
                        )
                    ).toInt(),
            )

        val warnings = buildList {
            if (uncertaintyMm > config.highUncertaintyMm) {
                add("P95_RECONSTRUCTION_UNCERTAINTY_HIGH")
            }
            if (geometryScore < 0.60) {
                add("GEOMETRIC_OBSERVABILITY_REDUCED")
            }
            if (thermalScore < 0.50) {
                add("THERMAL_HEADROOM_REDUCED")
            }
            if (memoryScore < 0.50) {
                add("MEMORY_HEADROOM_REDUCED")
            }
            if (gpuScore < 0.50) {
                add("GPU_LATENCY_REDUCED")
            }
            if (viewScore < 0.60) {
                add("MULTI_VIEW_DIVERSITY_REDUCED")
            }
            if (tier == Tier.SAFE_HOLD) {
                add("RECONSTRUCTION_DETAIL_HELD_FOR_EVIDENCE_SAFETY")
            }
        }.distinct()

        val recommendations = buildList {
            if (uncertaintyMm > config.highUncertaintyMm) {
                add("Increase calibrated multi-view overlap or improve landmark uncertainty before increasing mesh density.")
            }
            if (viewScore < 0.60) {
                add("Add an independent camera angle rather than simply raising inference frequency.")
            }
            if (thermalScore < 0.50) {
                add("Use a short high-detail capture window and then return to a sustained tier.")
            }
            if (memoryScore < 0.50) {
                add("Prefer bounded surface buffers and avoid duplicate frame copies.")
            }
            if (tier == Tier.SAFE_HOLD) {
                add("Hold metric geometry output until capture evidence returns above the configured floor.")
            }
        }.distinct()

        return Budget(
            tier = tier,
            effectiveVoxelSizeMeters = voxel,
            effectiveTruncationMeters = truncation,
            meshVertexBudget = meshBudget,
            denseMeshHz = denseMeshHz,
            faceMeshHz = faceMeshHz,
            eyeTrackingHz = eyeHz,
            handTrackingHz = handHz,
            bundleAdjustmentIterations = bundleIterations,
            score = score,
            preserveMetricEvidence = tier != Tier.SAFE_HOLD,
            holdOutput = tier == Tier.SAFE_HOLD,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun inverse(
        value: Double,
        low: Double,
        high: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= low) return 1.0
        if (value >= high) return 0.0
        return (
            1.0 -
                (value - low) /
                    (high - low).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }
}
