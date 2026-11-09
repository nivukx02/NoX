package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min

/**
 * Representation-level authority arbitration for reconstruction.
 *
 * No single representation is universally optimal. TSDF is strong for metric occupancy and stable
 * surfaces, Gaussian state is strong for view-dependent appearance and dynamic surfels, probabilistic
 * patches are strong for uncertainty-aware temporal continuity, while learned completion is useful
 * only where direct evidence is absent. This engine makes that trade-off explicit and bounded.
 */
class ReconstructionRepresentationAuthorityEngine(
    private val config: Config = Config(),
) {

    enum class Representation {
        TSDF,
        GAUSSIAN,
        PROBABILISTIC_SURFACE,
        LEARNED_COMPLETION,
    }

    data class Evidence(
        val representation: Representation,
        val coverage: Double,
        val metricConfidence: Double,
        val uncertaintyMeters: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val appearanceConfidence: Double,
        val dynamicCompatibility: Double,
        val memoryCostMegabytes: Double,
        val gpuCostNormalized: Double,
        val directlyMeasuredRatio: Double,
        val epipolarConsistency: Double = 1.0,
        val metricEvidenceAuthority: Double = directlyMeasuredRatio,
    ) {
        init {
            listOf(
                coverage,
                metricConfidence,
                temporalStability,
                topologyStability,
                appearanceConfidence,
                dynamicCompatibility,
                directlyMeasuredRatio,
                epipolarConsistency,
                metricEvidenceAuthority,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
            require(memoryCostMegabytes.isFinite() && memoryCostMegabytes >= 0.0)
            require(gpuCostNormalized.isFinite() && gpuCostNormalized >= 0.0)
        }
    }

    data class Decision(
        val primary: Representation,
        val secondary: List<Representation>,
        val confidence: Double,
        val scoreGap: Double,
        val evidenceLimited: Boolean,
        val learnedOnly: Boolean,
        val reasons: List<String>,
    )

    data class Result(
        val decision: Decision,
        val rankedScores: List<Pair<Representation, Double>>,
        val totalMemoryMegabytes: Double,
        val normalizedGpuCost: Double,
        val directlyMeasuredRatio: Double,
        val warnings: List<String>,
    )

    data class Config(
        val maximumMemoryMegabytes: Double = 3_072.0,
        val maximumGpuCostNormalized: Double = 0.90,
        val minimumDirectMeasurementForMetricAuthority: Double = 0.55,
        val uncertaintyReferenceMeters: Double = 0.002,
        val scoreMarginForSecondary: Double = 0.10,
        val learnedAuthorityCeiling: Double = 0.42,
    ) {
        init {
            require(maximumMemoryMegabytes > 0.0)
            require(maximumGpuCostNormalized > 0.0)
            require(minimumDirectMeasurementForMetricAuthority in 0.0..1.0)
            require(uncertaintyReferenceMeters > 0.0)
            require(scoreMarginForSecondary >= 0.0)
            require(learnedAuthorityCeiling in 0.0..1.0)
        }
    }

    fun decide(
        evidence: Collection<Evidence>,
        memoryHeadroom: Double,
        gpuHeadroom: Double,
        dynamicScene: Boolean,
    ): Result {
        require(memoryHeadroom in 0.0..1.0)
        require(gpuHeadroom in 0.0..1.0)

        if (evidence.isEmpty()) {
            return Result(
                decision =
                    Decision(
                        primary = Representation.PROBABILISTIC_SURFACE,
                        secondary = emptyList(),
                        confidence = 0.0,
                        scoreGap = 0.0,
                        evidenceLimited = true,
                        learnedOnly = false,
                        reasons =
                            listOf(
                                "no-reconstruction-representation-evidence",
                            ),
                    ),
                rankedScores = emptyList(),
                totalMemoryMegabytes = 0.0,
                normalizedGpuCost = 0.0,
                directlyMeasuredRatio = 0.0,
                warnings =
                    listOf(
                        "reconstruction-representation-evidence-missing",
                    ),
            )
        }

        val availableBudget =
            (
                0.25 * memoryHeadroom +
                    0.45 * gpuHeadroom +
                    0.30 *
                        min(
                            memoryHeadroom,
                            gpuHeadroom,
                        )
                )
                .coerceIn(0.05, 1.0)

        val totalMemory =
            evidence.sumOf {
                it.memoryCostMegabytes
            }

        val totalGpuCost =
            evidence.sumOf {
                it.gpuCostNormalized
            }

        val directlyMeasured =
            evidence
                .map {
                    it.directlyMeasuredRatio * it.coverage
                }
                .sum()
                .div(
                    evidence.sumOf { it.coverage }
                        .coerceAtLeast(1e-6),
                )
                .coerceIn(0.0, 1.0)

        val ranked =
            evidence
                .distinctBy { it.representation }
                .map { item ->
                    val uncertaintyScore =
                        (
                            1.0 -
                                item.uncertaintyMeters /
                                    config.uncertaintyReferenceMeters
                        )
                            .coerceIn(0.0, 1.0)

                    val memoryPenalty =
                        (
                            item.memoryCostMegabytes /
                                config.maximumMemoryMegabytes
                        )
                            .coerceIn(0.0, 1.0)

                    val gpuPenalty =
                        (
                            item.gpuCostNormalized /
                                config.maximumGpuCostNormalized
                        )
                            .coerceIn(0.0, 1.0)

                    val dynamicBonus =
                        if (
                            dynamicScene &&
                            item.representation ==
                                Representation.PROBABILISTIC_SURFACE
                        ) {
                            0.16
                        } else {
                            0.0
                        }

                    val metricBonus =
                        if (
                            item.directlyMeasuredRatio >=
                                config.minimumDirectMeasurementForMetricAuthority
                        ) {
                            0.10
                        } else {
                            0.0
                        }

                    val geometricAuthorityPenalty =
                        (
                            (1.0 - item.epipolarConsistency) * 0.08 +
                                (1.0 - item.metricEvidenceAuthority) * 0.10
                            ).coerceIn(0.0, 0.18)

                    val learnedPenalty =
                        if (
                            item.representation ==
                                Representation.LEARNED_COMPLETION
                        ) {
                            (
                                config.learnedAuthorityCeiling
                            )
                                .coerceAtMost(
                                    item.directlyMeasuredRatio +
                                        0.15,
                                )
                        } else {
                            0.0
                        }

                    val score =
                        (
                            0.18 * item.coverage +
                                0.18 * item.metricConfidence +
                                0.16 * uncertaintyScore +
                                0.12 * item.temporalStability +
                                0.10 * item.topologyStability +
                                0.10 * item.appearanceConfidence +
                                0.08 * item.dynamicCompatibility +
                                 0.06 * item.directlyMeasuredRatio +
                                 0.07 * item.epipolarConsistency +
                                 0.09 * item.metricEvidenceAuthority +
                                 dynamicBonus +
                                 metricBonus -
                                 geometricAuthorityPenalty -
                                 0.06 * memoryPenalty -
                                0.06 * gpuPenalty -
                                learnedPenalty * 0.08
                            )
                            .coerceIn(0.0, 1.0)

                    item.representation to score
                }
                .sortedWith(
                    compareByDescending<Pair<Representation, Double>> {
                        it.second
                    }.thenBy {
                        it.first.ordinal
                    },
                )

        val resourceLimited =
            totalMemory >
                config.maximumMemoryMegabytes *
                    max(
                        0.05,
                        memoryHeadroom,
                    ) ||
                totalGpuCost >
                    config.maximumGpuCostNormalized *
                        max(
                            0.05,
                            gpuHeadroom,
                        )

        val adjustedRanked =
            if (!resourceLimited) {
                ranked
            } else {
                ranked.map { (representation, score) ->
                    val evidenceItem =
                        evidence.firstOrNull {
                            it.representation == representation
                        }
                    val pressure =
                        (
                            evidenceItem?.memoryCostMegabytes?.div(
                                config.maximumMemoryMegabytes *
                                    max(
                                        0.05,
                                        memoryHeadroom,
                                    ),
                            )
                                ?: 0.0
                        ) +
                            (
                                evidenceItem?.gpuCostNormalized?.div(
                                    config.maximumGpuCostNormalized *
                                        max(
                                            0.05,
                                            gpuHeadroom,
                                        ),
                                )
                                    ?: 0.0
                            )

                    representation to
                        (
                            score -
                                0.05 *
                                    pressure.coerceIn(
                                        0.0,
                                        2.0,
                                    )
                            )
                                .coerceIn(0.0, 1.0)
                }
                    .sortedWith(
                        compareByDescending<Pair<Representation, Double>> {
                            it.second
                        }.thenBy {
                            it.first.ordinal
                        },
                    )
            }

        val primary =
            adjustedRanked.first().first
        val primaryScore =
            adjustedRanked.first().second
        val secondScore =
            adjustedRanked.getOrNull(1)?.second ?: 0.0

        val secondary =
            adjustedRanked
                .drop(1)
                .take(2)
                .filter {
                    primaryScore -
                        it.second <=
                        config.scoreMarginForSecondary
                }
                .map {
                    it.first
                }

        val confidence =
            (
                primaryScore *
                    (
                        0.70 +
                            0.30 * availableBudget
                    )
                )
                .coerceIn(0.0, 1.0)

        val learnedOnly =
            primary ==
                Representation.LEARNED_COMPLETION

        val evidenceLimited =
            directlyMeasured <
                config.minimumDirectMeasurementForMetricAuthority ||
                primaryScore < 0.55

        val reasons =
            buildList {
                if (resourceLimited) add("representation-resource-pressure")
                if (dynamicBonusFor(primary, dynamicScene)) {
                    add("dynamic-scene-temporal-authority")
                }
                if (
                    directlyMeasured <
                        config.minimumDirectMeasurementForMetricAuthority
                ) {
                    add("direct-metric-evidence-limited")
                }
                if (
                    primary ==
                        Representation.GAUSSIAN
                ) {
                    add("appearance-and-dynamic-surface-authority")
                }
                if (
                    primary ==
                        Representation.TSDF
                ) {
                    add("metric-occupancy-and-topology-authority")
                }
                if (
                    primary ==
                        Representation.PROBABILISTIC_SURFACE
                ) {
                    add("uncertainty-and-temporal-surface-authority")
                }
                if (learnedOnly) {
                    add("learned-completion-must-not-be-treated-as-direct-measurement")
                }
            }

        val warnings =
            buildList {
                if (resourceLimited) {
                    add("reconstruction-representation-resource-pressure")
                }
                if (evidenceLimited) {
                    add("reconstruction-representation-evidence-limited")
                }
                if (learnedOnly) {
                    add("reconstruction-primary-representation-learned-only")
                }
                if (primaryScore - secondScore < 0.06) {
                    add("reconstruction-representation-authority-close")
                }
            }

        return Result(
            decision =
                Decision(
                    primary = primary,
                    secondary = secondary,
                    confidence = confidence,
                    scoreGap = primaryScore - secondScore,
                    evidenceLimited = evidenceLimited,
                    learnedOnly = learnedOnly,
                    reasons = reasons.distinct(),
                ),
            rankedScores = adjustedRanked,
            totalMemoryMegabytes = totalMemory,
            normalizedGpuCost = totalGpuCost,
            directlyMeasuredRatio = directlyMeasured,
            warnings = warnings.distinct(),
        )
    }

    private fun dynamicBonusFor(
        representation: Representation,
        dynamicScene: Boolean,
    ): Boolean =
        dynamicScene &&
            representation ==
                Representation.PROBABILISTIC_SURFACE
}
