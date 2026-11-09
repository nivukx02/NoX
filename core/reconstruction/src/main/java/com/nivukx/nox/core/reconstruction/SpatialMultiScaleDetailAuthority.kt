package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min

/**
 * Spatial multi-scale detail authority for dense 3D reconstruction.
 *
 * The allocator decides where additional reconstruction effort is justified. It does not invent
 * geometry and it never changes the metric point itself. Instead, it assigns a bounded detail band
 * and compute share to each reconstructed region using already-measured evidence:
 *
 *  - geometric complexity,
 *  - motion-boundary evidence,
 *  - semantic importance,
 *  - uncertainty-reduction value,
 *  - view information gain,
 *  - temporal stability.
 *
 * Cost grows non-linearly with detail level. This prevents a high-detail request from consuming the
 * entire frame budget when thousands of points are present.
 */
class SpatialMultiScaleDetailAuthority(
    private val config: Config = Config(),
) {
    enum class Band {
        COARSE,
        STANDARD,
        FINE,
        ULTRA,
    }

    data class Config(
        val maximumRegions: Int = 16_384,
        val minimumDetailBudgetUnits: Int = 4_096,
        val standardCostUnits: Int = 2,
        val fineCostUnits: Int = 4,
        val ultraCostUnits: Int = 8,
        val coarseCostUnits: Int = 1,
        val coarseThreshold: Double = 0.30,
        val standardThreshold: Double = 0.48,
        val fineThreshold: Double = 0.70,
        val ultraThreshold: Double = 0.88,
        val minimumTemporalStability: Double = 0.30,
        val uncertaintySoftLimitMeters: Double = 0.080,
        val uncertaintyHardLimitMeters: Double = 0.220,
        val minimumViewInformation: Double = 0.25,
        val complexityWeight: Double = 0.24,
        val motionWeight: Double = 0.17,
        val semanticWeight: Double = 0.12,
        val uncertaintyWeight: Double = 0.17,
        val viewInformationWeight: Double = 0.18,
        val temporalWeight: Double = 0.12,
    ) {
        init {
            require(maximumRegions >= 1)
            require(minimumDetailBudgetUnits >= 1)
            require(coarseCostUnits in 1..128)
            require(standardCostUnits > coarseCostUnits)
            require(fineCostUnits > standardCostUnits)
            require(ultraCostUnits > fineCostUnits)
            require(
                coarseThreshold >= 0.0 &&
                    coarseThreshold < standardThreshold,
            )
            require(standardThreshold < fineThreshold)
            require(fineThreshold < ultraThreshold)
            require(ultraThreshold <= 1.0)
            require(minimumTemporalStability in 0.0..1.0)
            require(uncertaintySoftLimitMeters > 0.0)
            require(uncertaintyHardLimitMeters > uncertaintySoftLimitMeters)
            require(minimumViewInformation in 0.0..1.0)
            require(complexityWeight >= 0.0)
            require(motionWeight >= 0.0)
            require(semanticWeight >= 0.0)
            require(uncertaintyWeight >= 0.0)
            require(viewInformationWeight >= 0.0)
            require(temporalWeight >= 0.0)
            require(
                complexityWeight +
                    motionWeight +
                    semanticWeight +
                    uncertaintyWeight +
                    viewInformationWeight +
                    temporalWeight > 0.0,
            )
        }
    }

    data class RegionEvidence(
        val regionId: Long,
        val geometricComplexity: Double,
        val motionBoundary: Double,
        val semanticImportance: Double,
        val uncertaintyMeters: Double,
        val viewInformationGain: Double,
        val temporalStability: Double,
        /** Combined metric-view evidence authority supplied by the upstream reconstruction solve. */
        val metricEvidenceAuthority: Double = 1.0,
    ) {
        init {
            require(regionId >= 0L)
            listOf(
                geometricComplexity,
                motionBoundary,
                semanticImportance,
                viewInformationGain,
                temporalStability,
                metricEvidenceAuthority,
            ).forEach {
                require(it in 0.0..1.0)
            }
            require(
                uncertaintyMeters.isFinite() &&
                    uncertaintyMeters >= 0.0,
            )
        }
    }

    data class Assignment(
        val regionId: Long,
        val band: Band,
        val priority: Double,
        val costUnits: Int,
        val computeShare: Double,
        val admittedForRefinement: Boolean,
        val holdReason: String? = null,
    )

    data class Result(
        val assignments: List<Assignment>,
        val requestedRegionCount: Int,
        val admittedRegionCount: Int,
        val coarseRegionCount: Int,
        val standardRegionCount: Int,
        val fineRegionCount: Int,
        val ultraRegionCount: Int,
        val usedBudgetUnits: Int,
        val budgetUnits: Int,
        val budgetUtilization: Double,
        val meanPriority: Double,
        val ultraRatio: Double,
        val fineOrHigherRatio: Double,
        val holdOutput: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun allocate(
        regions: Collection<RegionEvidence>,
        availableBudgetUnits: Int,
    ): Result {
        require(availableBudgetUnits >= 1)

        val bounded = regions
            .distinctBy { it.regionId }
            .take(config.maximumRegions)

        if (bounded.isEmpty()) {
            return emptyResult()
        }

        val budgetUnits =
            max(
                config.minimumDetailBudgetUnits,
                availableBudgetUnits,
            )

        val ranked =
            bounded
                .map { evidence ->
                    evidence to priority(evidence)
                }
                .sortedWith(
                    compareByDescending<Pair<RegionEvidence, Double>> {
                        it.second
                    }.thenBy {
                        it.first.regionId
                    },
                )

        val assignments = ArrayList<Assignment>(ranked.size)
        var used = 0

        ranked.forEach { (evidence, score) ->
            val preferredBand =
                preferredBand(
                    score = score,
                    evidence = evidence,
                )
            val preferredCost =
                cost(preferredBand)

            if (used + preferredCost <= budgetUnits) {
                used += preferredCost
                assignments += Assignment(
                    regionId = evidence.regionId,
                    band = preferredBand,
                    priority = score,
                    costUnits = preferredCost,
                    computeShare = 0.0,
                    admittedForRefinement = true,
                )
            } else {
                val fallbackBand =
                    listOf(
                        Band.FINE,
                        Band.STANDARD,
                        Band.COARSE,
                    )
                        .filter { band ->
                            cost(band) <=
                                budgetUnits - used
                        }
                        .maxByOrNull(::cost)
                        ?: Band.COARSE
                val fallbackCost = cost(fallbackBand)

                if (
                    fallbackCost <= budgetUnits - used &&
                    score >= config.coarseThreshold
                ) {
                    used += fallbackCost
                    assignments += Assignment(
                        regionId = evidence.regionId,
                        band = fallbackBand,
                        priority = score,
                        costUnits = fallbackCost,
                        computeShare = 0.0,
                        admittedForRefinement = true,
                    )
                } else {
                    assignments += Assignment(
                        regionId = evidence.regionId,
                        band = Band.COARSE,
                        priority = score,
                        costUnits = 0,
                        computeShare = 0.0,
                        admittedForRefinement = false,
                        holdReason = "spatial-detail-budget-exhausted",
                    )
                }
            }
        }

        val totalWeightedPriority =
            assignments.sumOf {
                it.priority * max(it.costUnits, 1)
            }.coerceAtLeast(1e-9)

        val normalized =
            assignments.map { assignment ->
                assignment.copy(
                    computeShare =
                        (
                            assignment.priority *
                                max(
                                    assignment.costUnits,
                                    1,
                                )
                        ) /
                            totalWeightedPriority,
                )
            }

        val coarse =
            normalized.count {
                it.band == Band.COARSE
            }
        val standard =
            normalized.count {
                it.band == Band.STANDARD
            }
        val fine =
            normalized.count {
                it.band == Band.FINE
            }
        val ultra =
            normalized.count {
                it.band == Band.ULTRA
            }

        val admitted =
            normalized.count {
                it.admittedForRefinement
            }
        val regionCount =
            normalized.size
        val utilization =
            used.toDouble() /
                budgetUnits.toDouble().coerceAtLeast(1.0)

        val meanPriority =
            normalized
                .map { it.priority }
                .average()
                .coerceIn(0.0, 1.0)

        val ultraRatio =
            ultra.toDouble() /
                regionCount.toDouble().coerceAtLeast(1.0)
        val fineOrHigherRatio =
            (fine + ultra).toDouble() /
                regionCount.toDouble().coerceAtLeast(1.0)

        val holdOutput =
            admitted < max(
                1,
                regionCount / 8,
            ) ||
                (
                    meanPriority < config.coarseThreshold &&
                        regionCount >= 16
                )

        val warnings = buildList {
            if (bounded.size < regions.distinctBy { it.regionId }.size) {
                add("SPATIAL_DETAIL_REGION_CAP_APPLIED")
            }
            if (utilization >= 0.98) {
                add("SPATIAL_DETAIL_BUDGET_NEAR_SATURATION")
            }
            if (admitted < regionCount * 0.50) {
                add("SPATIAL_DETAIL_ADMISSION_LOW")
            }
            if (ultraRatio > 0.40) {
                add("SPATIAL_DETAIL_ULTRA_REGION_RATIO_HIGH")
            }
            if (
                normalized.any {
                    it.holdReason ==
                        "spatial-detail-budget-exhausted"
                }
            ) {
                add("SPATIAL_DETAIL_BUDGET_EXHAUSTED_FOR_LOW_PRIORITY_REGIONS")
            }
            if (holdOutput) {
                add("SPATIAL_DETAIL_OUTPUT_HELD_FOR_EVIDENCE_COVERAGE")
            }
        }.distinct()

        val recommendations = buildList {
            if (utilization >= 0.90) {
                add("Increase GPU or memory headroom before raising the spatial fine-detail budget.")
            }
            if (fineOrHigherRatio < 0.20) {
                add("Prioritize camera coverage and stable observations before increasing global mesh density.")
            }
            if (ultraRatio > 0.40) {
                add("Limit ultra-detail to regions with measurable geometric complexity rather than the whole surface.")
            }
            if (holdOutput) {
                add("Hold metric surface refinement until enough regions have independently supported evidence.")
            }
        }.distinct()

        return Result(
            assignments = normalized,
            requestedRegionCount = regionCount,
            admittedRegionCount = admitted,
            coarseRegionCount = coarse,
            standardRegionCount = standard,
            fineRegionCount = fine,
            ultraRegionCount = ultra,
            usedBudgetUnits = used,
            budgetUnits = budgetUnits,
            budgetUtilization = utilization.coerceIn(0.0, 1.0),
            meanPriority = meanPriority,
            ultraRatio = ultraRatio,
            fineOrHigherRatio = fineOrHigherRatio,
            holdOutput = holdOutput,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun priority(
        evidence: RegionEvidence,
    ): Double {
        val uncertaintyScore =
            inverseUncertainty(
                evidence.uncertaintyMeters,
            )
        val temporalScore =
            (
                evidence.temporalStability /
                    config.minimumTemporalStability.coerceAtLeast(1e-6)
            ).coerceIn(0.0, 1.0)

        val base =
            (
                evidence.geometricComplexity *
                    config.complexityWeight +
                    evidence.motionBoundary *
                    config.motionWeight +
                    evidence.semanticImportance *
                    config.semanticWeight +
                    uncertaintyScore *
                    config.uncertaintyWeight +
                    evidence.viewInformationGain *
                    config.viewInformationWeight +
                    temporalScore *
                    config.temporalWeight
                ).coerceIn(0.0, 1.0)

        return (
            base *
                (
                    0.76 +
                        0.24 *
                            evidence.metricEvidenceAuthority
                )
            ).coerceIn(0.0, 1.0)
    }

    private fun preferredBand(
        score: Double,
        evidence: RegionEvidence,
    ): Band =
        when {
            evidence.temporalStability <
                config.minimumTemporalStability ->
                Band.COARSE
            evidence.viewInformationGain <
                config.minimumViewInformation ||
                evidence.metricEvidenceAuthority < 0.30 ->
                Band.COARSE
            score >= config.ultraThreshold ->
                Band.ULTRA
            score >= config.fineThreshold ->
                Band.FINE
            score >= config.standardThreshold ->
                Band.STANDARD
            else ->
                Band.COARSE
        }

    private fun inverseUncertainty(
        value: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= config.uncertaintySoftLimitMeters) return 1.0
        if (value >= config.uncertaintyHardLimitMeters) return 0.0
        return (
            1.0 -
                (
                    value -
                        config.uncertaintySoftLimitMeters
                    ) /
                    (
                        config.uncertaintyHardLimitMeters -
                            config.uncertaintySoftLimitMeters
                    ).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun cost(
        band: Band,
    ): Int =
        when (band) {
            Band.COARSE -> config.coarseCostUnits
            Band.STANDARD -> config.standardCostUnits
            Band.FINE -> config.fineCostUnits
            Band.ULTRA -> config.ultraCostUnits
        }

    private fun emptyResult(): Result =
        Result(
            assignments = emptyList(),
            requestedRegionCount = 0,
            admittedRegionCount = 0,
            coarseRegionCount = 0,
            standardRegionCount = 0,
            fineRegionCount = 0,
            ultraRegionCount = 0,
            usedBudgetUnits = 0,
            budgetUnits = 0,
            budgetUtilization = 0.0,
            meanPriority = 0.0,
            ultraRatio = 0.0,
            fineOrHigherRatio = 0.0,
            holdOutput = true,
            warnings = listOf("SPATIAL_DETAIL_NO_REGIONS"),
            recommendations = listOf(
                "Collect valid reconstructed regions before allocating dense refinement compute.",
            ),
        )
}
