package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.Vec3
import kotlin.math.pow
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production-grade arbitration layer for heterogeneous 3D reconstruction evidence.
 *
 * A calibrated multi-view solution, a temporal prior, an inertial prior, a monocular-depth
 * estimate, and a kinematic constraint are not independent in the statistical sense. Treating
 * them as equally independent can make a fused point look more certain than the measurements
 * justify. This engine therefore performs four explicit operations:
 *
 * 1. Sanitise candidate geometry and reject non-finite evidence.
 * 2. Convert confidence and uncertainty into bounded source weights.
 * 3. Robustly solve a consensus point with Huber-style spatial residual weighting.
 * 4. Inflate uncertainty and reduce confidence when hypotheses disagree.
 *
 * Correlation groups are used to prevent multiple estimates derived from the same upstream
 * evidence family from overwhelming genuinely independent observations.
 *
 * This is deliberately deterministic and dependency-free so it can run in the hot reconstruction
 * path without a model download or an external service.
 */
class MultiEvidence3DReconstructionEngine(
    config: Config = Config(),
) : AutoCloseable {

    @Volatile
    private var runtimeConfig: Config = config

    private val temporalPriorEngine =
        HighEndTemporalFusionEngine()

    @Synchronized
    fun configure(next: Config) {
        runtimeConfig = next
    }

    fun currentConfig(): Config = runtimeConfig

    enum class Source {
        PRIMARY,
        MULTI_VIEW,
        MONOCULAR_DEPTH,
        TEMPORAL_PRIOR,
        INERTIAL_PRIOR,
        BODY_SPACE_PRIOR,
        KINEMATIC_CONSTRAINT,
    }

    data class Config(
        val maxIterations: Int = 8,
        val uncertaintyScaleMeters: Double = 0.055,
        val disagreementScaleMeters: Double = 0.12,
        val huberDeltaMeters: Double = 0.085,
        val outlierDistanceMeters: Double = 0.28,
        val minimumCandidateConfidence: Double = 0.05,
        val minimumAcceptedConfidence: Double = 0.22,
        val maximumAcceptedUncertaintyMeters: Double = 1.50,
        val correlationGroupCapacity: Double = 1.35,
        val disagreementConfidencePenalty: Double = 0.48,
        val minimumSourceDiversity: Double = 0.20,
    ) {
        init {
            require(maxIterations in 2..32)
            require(uncertaintyScaleMeters > 0.0)
            require(disagreementScaleMeters > 0.0)
            require(huberDeltaMeters > 0.0)
            require(outlierDistanceMeters > huberDeltaMeters)
            require(minimumCandidateConfidence in 0.0..1.0)
            require(minimumAcceptedConfidence in 0.0..1.0)
            require(maximumAcceptedUncertaintyMeters > 0.0)
            require(correlationGroupCapacity > 0.0)
            require(disagreementConfidencePenalty in 0.0..1.0)
            require(minimumSourceDiversity in 0.0..1.0)
        }
    }

    data class Candidate(
        val joint: ReconstructionJoint,
        val source: Source,
        val correlationGroup: String,
    ) {
        init {
            require(correlationGroup.isNotBlank())
            require(joint.confidence.isFinite())
            require(joint.confidence in 0.0..1.0)
            require(joint.positionUncertaintyMeters.isFinite())
            require(joint.positionUncertaintyMeters >= 0.0)
        }
    }

    data class TemporalFusionResult(
        val result: Result,
        val temporalSummary: HighEndTemporalFusionEngine.FrameSummary,
        val temporalPriorUsed: Boolean,
    )

    data class Result(
        val position: Vec3,
        val velocity: Vec3,
        val confidence: Double,
        val positionUncertaintyMeters: Double,
        val disagreementMeters: Double,
        val sourceDiversity: Double,
        val outlierCount: Int,
        val acceptedCandidateCount: Int,
        val usedPrimaryOnly: Boolean,
        val contributors: List<Source>,
    ) {
        val accepted: Boolean
            get() =
                position.x.isFinite() &&
                    position.y.isFinite() &&
                    position.z.isFinite() &&
                    velocity.x.isFinite() &&
                    velocity.y.isFinite() &&
                    velocity.z.isFinite() &&
                    confidence >= 0.0 &&
                    positionUncertaintyMeters.isFinite()
    }

    private data class WeightedCandidate(
        val candidate: Candidate,
        val baseWeight: Double,
        val groupWeightCap: Double,
    )

    private data class FusionState(
        val position: Vec3,
        val velocity: Vec3,
        val weights: List<Double>,
        val disagreementMeters: Double,
        val outlierCount: Int,
    )

    @Synchronized
    fun fuseWithTemporalPrior(
        timestampNanos: Long,
        primary: ReconstructionJoint,
        candidates: List<Candidate>,
    ): TemporalFusionResult {
        require(timestampNanos >= 0L)

        val observations =
            buildList {
                add(
                    HighEndTemporalFusionEngine.Observation(
                        trackId = primary.landmarkId,
                        timestampNanos = timestampNanos,
                        point =
                            HighEndTemporalFusionEngine.Point3(
                                primary.position.x,
                                primary.position.y,
                                primary.position.z,
                            ),
                        confidence = primary.confidence,
                        cameraId = "primary",
                        covarianceTrace =
                            primary.positionUncertaintyMeters.coerceIn(
                                0.0,
                                1.0,
                            ),
                    ),
                )
                candidates.forEachIndexed { index, candidate ->
                    add(
                        HighEndTemporalFusionEngine.Observation(
                            trackId = candidate.joint.landmarkId,
                            timestampNanos = timestampNanos,
                            point =
                                HighEndTemporalFusionEngine.Point3(
                                    candidate.joint.position.x,
                                    candidate.joint.position.y,
                                    candidate.joint.position.z,
                                ),
                            confidence = candidate.joint.confidence,
                            cameraId =
                                candidate.source.name +
                                    ":" +
                                    candidate.correlationGroup +
                                    ":" +
                                    index,
                            covarianceTrace =
                                candidate.joint
                                    .positionUncertaintyMeters
                                    .coerceIn(0.0, 1.0),
                        ),
                    )
                }
            }

        var reset = false
        val summary =
            try {
                temporalPriorEngine.update(
                    timestampNanos = timestampNanos,
                    observations = observations,
                )
            } catch (error: IllegalArgumentException) {
                if (!error.message.orEmpty().contains("strictly increasing")) {
                    throw error
                }
                reset = true
                temporalPriorEngine.reset()
                temporalPriorEngine.update(
                    timestampNanos = timestampNanos,
                    observations = observations,
                )
            }

        val snapshot =
            temporalPriorEngine.snapshot(primary.landmarkId)
                ?: return TemporalFusionResult(
                    result = fuse(
                        timestampNanos = timestampNanos,
                        primary = primary,
                        candidates = candidates,
                    ),
                    temporalSummary = summary,
                    temporalPriorUsed = false,
                )

        val temporalPrior =
            primary.copy(
                position =
                    Vec3(
                        snapshot.position.x,
                        snapshot.position.y,
                        snapshot.position.z,
                    ),
                velocity =
                    Vec3(
                        snapshot.velocity.x,
                        snapshot.velocity.y,
                        snapshot.velocity.z,
                    ),
                confidence =
                    min(
                        primary.confidence,
                        snapshot.confidence,
                    ),
                positionUncertaintyMeters =
                    primary.positionUncertaintyMeters *
                        (
                            1.0 +
                                (
                                    1.0 -
                                        snapshot.confidence
                                    ).coerceIn(0.0, 1.0) * 0.50
                            ),
            )

        val result =
            fuse(
                timestampNanos = timestampNanos,
                primary = primary,
                candidates =
                    candidates +
                        Candidate(
                            joint = temporalPrior,
                            source = Source.TEMPORAL_PRIOR,
                            correlationGroup = "high-end-temporal",
                        ),
            )

        return TemporalFusionResult(
            result = result,
            temporalSummary = summary,
            temporalPriorUsed =
                !reset || result.accepted,
        )
    }

    fun fuse(
        timestampNanos: Long,
        primary: ReconstructionJoint,
        candidates: List<Candidate>,
    ): Result {
        require(timestampNanos >= 0L)

        val primaryCandidate =
            Candidate(
                joint = primary,
                source = Source.PRIMARY,
                correlationGroup = "primary",
            )

        val validCandidates =
            (listOf(primaryCandidate) + candidates)
                .asSequence()
                .filter(::isUsable)
                .distinctBy {
                    candidateKey(
                        it,
                        timestampNanos,
                    )
                }
                .toList()

        if (validCandidates.isEmpty()) {
            return fallback(primary)
        }

        if (validCandidates.size == 1) {
            return fallback(primary).copy(
                position = primary.position,
                velocity = primary.velocity,
                confidence = primary.confidence,
                positionUncertaintyMeters =
                    primary.positionUncertaintyMeters,
                contributors = listOf(Source.PRIMARY),
            )
        }

        val weighted =
            validCandidates.map { candidate ->
                WeightedCandidate(
                    candidate = candidate,
                    baseWeight = baseWeight(candidate),
                    groupWeightCap =
                        runtimeConfig.correlationGroupCapacity,
                )
            }

        val normalizedBase =
            normalizeGroupWeights(weighted)

        val initial =
            robustConsensus(
                primary = primary.position,
                candidates = weighted.mapIndexed { index, item ->
                    item.candidate to normalizedBase[index]
                },
            )

        var state = FusionState(
            position = initial.position,
            velocity = initial.velocity,
            weights = initial.weights,
            disagreementMeters = initial.disagreementMeters,
            outlierCount = initial.outlierCount,
        )

        repeat(runtimeConfig.maxIterations) {
            val refined =
                robustConsensus(
                    primary = state.position,
                    candidates = weighted.mapIndexed { index, item ->
                        Pair(
                            item.candidate,
                            normalizedBase[index] *
                                residualWeight(
                                distanceMeters =
                                    state.position.distanceTo(
                                        item.candidate.joint.position,
                                    ),
                                candidate = item.candidate,
                                ),
                        )
                    },
                )

            val positionShift =
                state.position.distanceTo(refined.position)

            state =
                FusionState(
                    position = refined.position,
                    velocity = refined.velocity,
                    weights = refined.weights,
                    disagreementMeters = refined.disagreementMeters,
                    outlierCount = refined.outlierCount,
                )

            if (positionShift < 1e-5) {
                return@repeat
            }
        }

        val acceptedIndices =
            state.weights.indices.filter {
                state.weights[it] > 1e-5
            }

        val acceptedCandidates =
            acceptedIndices.map { index ->
                weighted[index].candidate
            }

        val diversity =
            sourceDiversity(acceptedCandidates)

        val consensus =
            consensusQuality(
                disagreementMeters = state.disagreementMeters,
            )

        val evidenceConfidence =
            weightedConfidence(
                candidates = weighted,
                weights = state.weights,
            )

        val uncertainty =
            fusedUncertainty(
                weighted = weighted,
                weights = state.weights,
                disagreementMeters = state.disagreementMeters,
                consensus = consensus,
            )

        val confidence =
            (
                evidenceConfidence *
                    consensus *
                    diversityPenalty(diversity)
                ).coerceIn(0.0, 1.0)

        val contributors =
            acceptedCandidates
                .sortedBy { it.source.ordinal }
                .map { it.source }
                .distinct()

        val accepted =
            state.position.isFinite() &&
                confidence >= runtimeConfig.minimumAcceptedConfidence &&
                uncertainty <= runtimeConfig.maximumAcceptedUncertaintyMeters

        val finalPosition =
            if (accepted || primary.position.isFinite()) {
                state.position
            } else {
                primary.position
            }

        return Result(
            position = finalPosition,
            velocity = state.velocity,
            confidence =
                if (accepted) {
                    confidence
                } else {
                    min(
                        primary.confidence,
                        confidence * 0.82,
                    ).coerceIn(0.0, 1.0)
                },
            positionUncertaintyMeters =
                uncertainty.coerceAtLeast(
                    primary.positionUncertaintyMeters,
                ),
            disagreementMeters = state.disagreementMeters,
            sourceDiversity = diversity,
            outlierCount = state.outlierCount,
            acceptedCandidateCount = acceptedCandidates.size,
            usedPrimaryOnly =
                acceptedCandidates.size <= 1 ||
                    diversity < runtimeConfig.minimumSourceDiversity,
            contributors =
                if (contributors.isEmpty()) {
                    listOf(Source.PRIMARY)
                } else {
                    contributors
                },
        )
    }

    private fun robustConsensus(
        primary: Vec3,
        candidates: List<Pair<Candidate, Double>>,
    ): FusionState {
        val finite =
            candidates.filter {
                it.first.joint.position.isFinite() &&
                    it.second.isFinite() &&
                    it.second > 1e-8
            }

        if (finite.isEmpty()) {
            return FusionState(
                position = primary,
                velocity = Vec3(0.0, 0.0, 0.0),
                weights = candidates.map { 0.0 },
                disagreementMeters = 0.0,
                outlierCount = 0,
            )
        }

        val weightSum =
            finite.sumOf { it.second }.coerceAtLeast(1e-9)

        var position =
            finite.sumOf { pair ->
                pair.first.joint.position.x * pair.second
            } / weightSum

        var y =
            finite.sumOf { pair ->
                pair.first.joint.position.y * pair.second
            } / weightSum

        var z =
            finite.sumOf { pair ->
                pair.first.joint.position.z * pair.second
            } / weightSum

        var center = Vec3(position, y, z)

        repeat(3) {
            val robustWeights =
                finite.map { (candidate, base) ->
                    base *
                        residualWeight(
                            center.distanceTo(
                                candidate.joint.position,
                            ),
                            candidate,
                        )
                }

            val sum =
                robustWeights.sum().coerceAtLeast(1e-9)

            position =
                finite.indices.sumOf { index ->
                    finite[index].first.joint.position.x *
                        robustWeights[index]
                } / sum

            y =
                finite.indices.sumOf { index ->
                    finite[index].first.joint.position.y *
                        robustWeights[index]
                } / sum

            z =
                finite.indices.sumOf { index ->
                    finite[index].first.joint.position.z *
                        robustWeights[index]
                } / sum

            center =
                Vec3(position, y, z)
        }

        val finalWeights =
            candidates.map { (candidate, base) ->
                base *
                    residualWeight(
                        center.distanceTo(
                            candidate.joint.position,
                        ),
                        candidate,
                    )
            }

        val normalized =
            normalizeWeights(finalWeights)

        val velocityWeightSum =
            normalized.sum().coerceAtLeast(1e-9)

        val velocity =
            candidates.indices.fold(
                Vec3(0.0, 0.0, 0.0),
            ) { accumulator, index ->
                accumulator +
                    candidates[index].first.joint.velocity *
                    (
                        normalized[index] /
                            velocityWeightSum
                        )
            }

        val disagreements =
            candidates.mapIndexed { index, pair ->
                center.distanceTo(pair.first.joint.position) *
                    normalized[index]
            }

        val centerDistances =
            candidates.map { (candidate, _) ->
                center.distanceTo(candidate.joint.position)
            }

        val nonOutlierIndices =
            centerDistances
                .mapIndexed { index, distance ->
                    index to distance
                }
                .filter {
                    it.second <= runtimeConfig.outlierDistanceMeters * 2.0
                }
                .map { it.first }

        val pairwiseSpread =
            nonOutlierIndices
                .flatMapIndexed { index, firstIndex ->
                    nonOutlierIndices.drop(index + 1).map { secondIndex ->
                        candidates[firstIndex].first.joint.position.distanceTo(
                            candidates[secondIndex].first.joint.position,
                        )
                    }
                }
                .maxOrNull()
                ?: 0.0

        val weightedDisagreement =
            if (disagreements.isEmpty()) {
                0.0
            } else {
                disagreements.sum() /
                    normalized.sum().coerceAtLeast(1e-9)
            }

        val disagreement =
            maxOf(
                weightedDisagreement,
                pairwiseSpread,
            )

        val outliers =
            candidates.count {
                center.distanceTo(it.first.joint.position) >
                    runtimeConfig.outlierDistanceMeters
            }

        return FusionState(
            position = center,
            velocity = velocity,
            weights = normalized,
            disagreementMeters = disagreement,
            outlierCount = outliers,
        )
    }

    private fun baseWeight(candidate: Candidate): Double {
        val confidence =
            candidate.joint.confidence
                .coerceIn(
                    runtimeConfig.minimumCandidateConfidence,
                    1.0,
                )

        val uncertainty =
            candidate.joint.positionUncertaintyMeters
                .coerceIn(0.0, 4.0)

        val uncertaintyWeight =
            exp(
                -uncertainty /
                    runtimeConfig.uncertaintyScaleMeters
            ).coerceIn(0.02, 1.0)

        val sourceWeight =
            when (candidate.source) {
                Source.PRIMARY -> 1.00
                Source.MULTI_VIEW -> 1.08
                Source.KINEMATIC_CONSTRAINT -> 0.94
                Source.TEMPORAL_PRIOR -> 0.82
                Source.INERTIAL_PRIOR -> 0.68
                Source.BODY_SPACE_PRIOR -> 0.42
                Source.MONOCULAR_DEPTH -> 0.64
            }

        return (
            kotlin.math.exp(1.35 * kotlin.math.ln(confidence.coerceAtLeast(1e-6))) *
                uncertaintyWeight *
                sourceWeight
            ).coerceAtLeast(1e-6)
    }

    private fun normalizeGroupWeights(
        candidates: List<WeightedCandidate>,
    ): List<Double> {
        if (candidates.isEmpty()) return emptyList()

        val groupTotals =
            candidates.groupingBy {
                it.candidate.correlationGroup
            }.fold(0.0) { total, item ->
                total + item.baseWeight
            }

        return candidates.map { candidate ->
            val total =
                groupTotals[candidate.candidate.correlationGroup]
                    ?.coerceAtLeast(1e-9)
                    ?: 1.0

            val capFactor =
                min(
                    1.0,
                    candidate.groupWeightCap / total,
                )

            candidate.baseWeight * capFactor
        }.let(::normalizeWeights)
    }

    private fun residualWeight(
        distanceMeters: Double,
        candidate: Candidate,
    ): Double {
        if (!distanceMeters.isFinite()) return 0.0

        val huber =
            if (distanceMeters <= runtimeConfig.huberDeltaMeters) {
                1.0
            } else {
                runtimeConfig.huberDeltaMeters /
                    distanceMeters.coerceAtLeast(
                        runtimeConfig.huberDeltaMeters,
                    )
            }

        val consensus =
            exp(
                -distanceMeters /
                    runtimeConfig.disagreementScaleMeters
            ).coerceIn(0.02, 1.0)

        val uncertaintyProtection =
            (
                1.0 -
                    candidate.joint
                        .positionUncertaintyMeters
                        .coerceIn(0.0, 1.0) *
                    0.10
                ).coerceIn(0.85, 1.0)

        return (
            huber *
                (
                    0.55 +
                        0.45 * consensus
                    ) *
                uncertaintyProtection
            ).coerceIn(0.0, 1.0)
    }

    private fun weightedConfidence(
        candidates: List<WeightedCandidate>,
        weights: List<Double>,
    ): Double {
        if (candidates.isEmpty()) return 0.0
        val total =
            weights.sum().coerceAtLeast(1e-9)

        return candidates.indices.sumOf { index ->
            candidates[index].candidate.joint.confidence *
                weights[index]
        } / total
    }

    private fun fusedUncertainty(
        weighted: List<WeightedCandidate>,
        weights: List<Double>,
        disagreementMeters: Double,
        consensus: Double,
    ): Double {
        if (weighted.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        val total =
            weights.sum().coerceAtLeast(1e-9)

        val measurementUncertainty =
            weighted.indices.sumOf { index ->
                weighted[index]
                    .candidate
                    .joint
                    .positionUncertaintyMeters
                    .coerceAtLeast(0.0) *
                    weights[index]
            } / total

        return (
            measurementUncertainty +
                disagreementMeters *
                    (
                        0.65 +
                            0.35 *
                                (
                                    1.0 -
                                        consensus
                                    )
                        ) +
                max(
                    0.0,
                    1.0 - consensus,
                ) *
                    0.015
            ).coerceAtLeast(1e-4)
    }

    private fun consensusQuality(
        disagreementMeters: Double,
    ): Double =
        exp(
            -disagreementMeters /
                runtimeConfig.disagreementScaleMeters
        ).coerceIn(0.0, 1.0)

    private fun sourceDiversity(
        candidates: List<Candidate>,
    ): Double {
        if (candidates.isEmpty()) return 0.0

        val groups =
            candidates
                .map { it.correlationGroup }
                .distinct()
                .size

        return (
            groups.toDouble() /
                3.0
            ).coerceIn(0.0, 1.0)
    }

    private fun diversityPenalty(
        diversity: Double,
    ): Double =
        if (diversity >= runtimeConfig.minimumSourceDiversity) {
            1.0
        } else {
            (
                0.82 +
                    diversity * 0.18
                ).coerceIn(0.0, 1.0)
        }

    private fun isUsable(candidate: Candidate): Boolean {
        val joint = candidate.joint
        return (
            joint.position.isFinite() &&
                joint.velocity.isFinite() &&
                joint.confidence.isFinite() &&
                joint.confidence >=
                    runtimeConfig.minimumCandidateConfidence &&
                joint.positionUncertaintyMeters.isFinite() &&
                joint.positionUncertaintyMeters >= 0.0
            )
    }

    private fun candidateKey(
        candidate: Candidate,
        timestampNanos: Long,
    ): String =
        listOf(
            timestampNanos,
            candidate.source.name,
            candidate.correlationGroup,
            candidate.joint.position.x,
            candidate.joint.position.y,
            candidate.joint.position.z,
        ).joinToString("|")

    private fun normalizeWeights(
        weights: List<Double>,
    ): List<Double> {
        val sum =
            weights.filter(Double::isFinite)
                .sum()
                .coerceAtLeast(1e-12)

        return weights.map {
            if (it.isFinite() && it > 0.0) {
                it / sum
            } else {
                0.0
            }
        }
    }

    @Synchronized
    override fun close() {
        temporalPriorEngine.close()
    }

    private fun fallback(
        primary: ReconstructionJoint,
    ): Result =
        Result(
            position = primary.position,
            velocity = primary.velocity,
            confidence = primary.confidence,
            positionUncertaintyMeters =
                primary.positionUncertaintyMeters,
            disagreementMeters = 0.0,
            sourceDiversity = 1.0,
            outlierCount = 0,
            acceptedCandidateCount = 1,
            usedPrimaryOnly = true,
            contributors = listOf(Source.PRIMARY),
        )

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()

    private fun Double.pow(
        exponent: Double,
    ): Double =
        java.lang.Math.pow(this, exponent)
}
