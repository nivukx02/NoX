package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Higher-order cross-view actor hypothesis authority.
 *
 * Pairwise links are useful for candidate generation, but pairwise errors can form mutually
 * compatible chains that describe no single actor. This layer evaluates compact 3-view and
 * higher-order hypotheses after the existing graph stage. It deliberately keeps the expensive
 * combinatorics bounded: only the strongest views in each candidate are admitted, and only a bounded
 * number of triplets is scored.
 *
 * The result is a soft evidence score. It never invents an identity, 3D location, or appearance
 * descriptor. The existing calibrated geometry and local track identifiers remain authoritative
 * evidence sources.
 */
class MultiViewActorHypothesisConsensusEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumViewsForHigherOrder: Int = 3,
        val maximumCandidateViews: Int = 12,
        val maximumTriplets: Int = 220,
        val minimumPairAggregate: Double = 0.52,
        val minimumTripletScore: Double = 0.58,
        val geometryToleranceMeters: Double = 1.35,
        val maximumDepthDisagreementMeters: Double = 0.35,
        val maximumClockSkewMillis: Double = 8.0,
        val scoreConsistencyWeight: Double = 0.30,
        val temporalWeight: Double = 0.20,
        val geometryWeight: Double = 0.28,
        val scaleWeight: Double = 0.12,
        val depthWeight: Double = 0.10,
        val confidenceWeight: Double = 0.08,
    ) {
        init {
            require(minimumViewsForHigherOrder >= 3)
            require(maximumCandidateViews >= minimumViewsForHigherOrder)
            require(maximumTriplets >= 1)
            require(minimumPairAggregate in 0.0..1.0)
            require(minimumTripletScore in 0.0..1.0)
            require(geometryToleranceMeters > 0.0)
            require(maximumDepthDisagreementMeters > 0.0)
            require(maximumClockSkewMillis > 0.0)
            require(
                scoreConsistencyWeight +
                    temporalWeight +
                    geometryWeight +
                    scaleWeight +
                    depthWeight +
                    confidenceWeight > 0.0,
            )
        }
    }

    data class Result(
        val viewCount: Int,
        val evaluatedViewCount: Int,
        val evaluatedTripletCount: Int,
        val acceptedTripletCount: Int,
        val consensusScore: Double,
        val ambiguityRisk: Double,
        val minimumTripletScore: Double,
        val temporalConsistency: Double,
        val geometryConsistency: Double,
        val scaleConsistency: Double,
        val depthConsistency: Double,
        val confidenceConsistency: Double,
        val warnings: List<String>,
    ) {
        val higherOrderAvailable: Boolean
            get() = evaluatedViewCount >= 3 && evaluatedTripletCount > 0

        val authoritative: Boolean
            get() =
                !higherOrderAvailable ||
                    (
                        consensusScore >= 0.58 &&
                            acceptedTripletCount > 0 &&
                            ambiguityRisk <= 0.42
                    )
    }

    fun evaluate(
        members: List<MultiCameraSubjectAssociationEngine.Observation>,
    ): Result {
        if (members.isEmpty()) {
            return Result(
                viewCount = 0,
                evaluatedViewCount = 0,
                evaluatedTripletCount = 0,
                acceptedTripletCount = 0,
                consensusScore = 0.0,
                ambiguityRisk = 1.0,
                minimumTripletScore = 0.0,
                temporalConsistency = 0.0,
                geometryConsistency = 0.0,
                scaleConsistency = 0.0,
                depthConsistency = 0.0,
                confidenceConsistency = 0.0,
                warnings = listOf("higher-order-candidate-empty"),
            )
        }

        val ranked = members
            .distinctBy { it.deviceId }
            .sortedWith(
                compareByDescending<MultiCameraSubjectAssociationEngine.Observation> {
                    it.confidence
                }.thenBy { it.deviceId.value }
                    .thenBy { it.localTrackId },
            )
            .take(config.maximumCandidateViews)

        if (ranked.size < config.minimumViewsForHigherOrder) {
            return Result(
                viewCount = members.map { it.deviceId }.distinct().size,
                evaluatedViewCount = ranked.size,
                evaluatedTripletCount = 0,
                acceptedTripletCount = 0,
                consensusScore = 1.0,
                ambiguityRisk = 0.0,
                minimumTripletScore = 1.0,
                temporalConsistency =
                    temporalConsistency(ranked),
                geometryConsistency =
                    geometryConsistency(ranked),
                scaleConsistency =
                    scaleConsistency(ranked),
                depthConsistency =
                    depthConsistency(ranked),
                confidenceConsistency =
                    confidenceConsistency(ranked),
                warnings = listOf("higher-order-evidence-unavailable-two-view"),
            )
        }

        val triplets = ArrayList<TripletScore>(
            min(
                config.maximumTriplets,
                ranked.size * (ranked.size - 1) * (ranked.size - 2) / 6,
            ),
        )

        loop@ for (a in 0 until ranked.lastIndex - 1) {
            for (b in a + 1 until ranked.lastIndex) {
                for (c in b + 1 until ranked.size) {
                    triplets += scoreTriplet(
                        ranked[a],
                        ranked[b],
                        ranked[c],
                    )
                    if (triplets.size >= config.maximumTriplets) {
                        break@loop
                    }
                }
            }
        }

        if (triplets.isEmpty()) {
            return Result(
                viewCount = members.map { it.deviceId }.distinct().size,
                evaluatedViewCount = ranked.size,
                evaluatedTripletCount = 0,
                acceptedTripletCount = 0,
                consensusScore = 0.0,
                ambiguityRisk = 1.0,
                minimumTripletScore = 0.0,
                temporalConsistency = temporalConsistency(ranked),
                geometryConsistency = geometryConsistency(ranked),
                scaleConsistency = scaleConsistency(ranked),
                depthConsistency = depthConsistency(ranked),
                confidenceConsistency = confidenceConsistency(ranked),
                warnings = listOf("higher-order-triplet-generation-failed"),
            )
        }

        val ordered = triplets
            .sortedByDescending { it.score }
        val accepted = ordered.filter {
            it.score >= config.minimumTripletScore &&
                it.minimumPairAggregate >= config.minimumPairAggregate
        }

        /*
         * The median of the best-supported hypotheses is less fragile than using only the single
         * strongest triplet. A single accidental triple should not become identity authority.
         */
        val topCount = max(
            1,
            min(
                8,
                ordered.size,
            ),
        )
        val topScores = ordered
            .take(topCount)
            .map { it.score }
            .sorted()

        val consensus =
            median(topScores)
                .coerceIn(0.0, 1.0)

        val ambiguity =
            (
                1.0 -
                    consensus
            ).coerceIn(0.0, 1.0)

        val minimumTriplet =
            ordered.minOfOrNull { it.score }
                ?: 0.0

        val warnings = buildList {
            if (accepted.isEmpty()) {
                add("higher-order-actor-hypothesis-not-converged")
            }
            if (consensus < 0.58) {
                add("higher-order-actor-consensus-low")
            }
            if (ambiguity > 0.42) {
                add("higher-order-identity-ambiguity-high")
            }
            if (depthConsistency(ranked) < 0.50) {
                add("higher-order-depth-consistency-low")
            }
            if (accepted.size < max(1, triplets.size / 4)) {
                add("higher-order-triplet-support-sparse")
            }
            if (ranked.size < members.map { it.deviceId }.distinct().size) {
                add("higher-order-view-budget-truncated")
            }
        }.distinct()

        return Result(
            viewCount = members.map { it.deviceId }.distinct().size,
            evaluatedViewCount = ranked.size,
            evaluatedTripletCount = triplets.size,
            acceptedTripletCount = accepted.size,
            consensusScore = consensus,
            ambiguityRisk = ambiguity,
            minimumTripletScore = minimumTriplet,
            temporalConsistency = temporalConsistency(ranked),
            geometryConsistency = geometryConsistency(ranked),
            scaleConsistency = scaleConsistency(ranked),
            depthConsistency = depthConsistency(ranked),
            confidenceConsistency = confidenceConsistency(ranked),
            warnings = warnings,
        )
    }

    private data class TripletScore(
        val score: Double,
        val minimumPairAggregate: Double,
        val temporal: Double,
        val geometry: Double,
        val scale: Double,
        val confidence: Double,
        val depth: Double,
    )

    private fun scoreTriplet(
        a: MultiCameraSubjectAssociationEngine.Observation,
        b: MultiCameraSubjectAssociationEngine.Observation,
        c: MultiCameraSubjectAssociationEngine.Observation,
    ): TripletScore {
        val pairs = listOf(
            a to b,
            a to c,
            b to c,
        )

        // Pair scoring is intentionally kept aligned with the parent association contract.
        val pairAggregates = pairs.map { (first, second) ->
            pairScore(first, second)
        }

        val temporal = temporalConsistency(listOf(a, b, c))
        val geometry = geometryConsistency(listOf(a, b, c))
        val scale = scaleConsistency(listOf(a, b, c))
        val depth = depthConsistency(listOf(a, b, c))
        val confidence = confidenceConsistency(listOf(a, b, c))
        val consistency = pairAggregates
            .map { it.coerceIn(0.0, 1.0) }
            .let { values ->
                if (values.isEmpty()) 0.0 else {
                    val mean = values.average()
                    val spread = values
                        .map { abs(it - mean) }
                        .average()
                    (mean * 0.75 + (1.0 - spread) * 0.25)
                        .coerceIn(0.0, 1.0)
                }
            }

        val score = (
            consistency * config.scoreConsistencyWeight +
                temporal * config.temporalWeight +
                geometry * config.geometryWeight +
                scale * config.scaleWeight +
                depth * config.depthWeight +
                confidence * config.confidenceWeight
            ).coerceIn(0.0, 1.0)

        return TripletScore(
            score = score,
            minimumPairAggregate =
                pairAggregates.minOrNull()?.coerceIn(0.0, 1.0) ?: 0.0,
            temporal = temporal,
            geometry = geometry,
            scale = scale,
            confidence = confidence,
            depth = depth,
        )
    }

    /**
     * Mirrors the public pair-admission contract without coupling this engine to the parent engine's
     * private implementation. The terms intentionally remain conservative.
     */
    private fun pairScore(
        a: MultiCameraSubjectAssociationEngine.Observation,
        b: MultiCameraSubjectAssociationEngine.Observation,
    ): Double {
        if (a.subjectType != b.subjectType) return 0.0

        val time = exp(
            -abs(a.timestampNanos - b.timestampNanos).toDouble() /
                (config.maximumClockSkewMillis * 1e6).coerceAtLeast(1.0),
        ).coerceIn(0.0, 1.0)

        val geometry =
            calibratedGeometryScore(a, b)

        val depth =
            calibratedDepthScore(a, b)

        val scale =
            scaleScore(a, b)

        val motion =
            exp(
                -(a.velocity - b.velocity).norm(),
            ).coerceIn(0.0, 1.0)

        val confidence =
            min(a.confidence, b.confidence)
                .coerceIn(0.0, 1.0)

        val calibratedPair =
            a.calibratedCenterMeters != null &&
                b.calibratedCenterMeters != null
        val geometricTerm =
            if (calibratedPair) {
                geometry
            } else {
                0.34
            }

        return (
            0.12 +
                geometricTerm * 0.30 +
                scale * 0.14 +
                motion * 0.18 +
                depth * 0.10 +
                confidence * 0.08 +
                time * 0.08
            ).coerceIn(0.0, 1.0)
    }

    private fun calibratedGeometryScore(
        a: MultiCameraSubjectAssociationEngine.Observation,
        b: MultiCameraSubjectAssociationEngine.Observation,
    ): Double {
        val pa = a.calibratedCenterMeters ?: return 0.0
        val pb = b.calibratedCenterMeters ?: return 0.0
        val distance2d = pa.distanceTo(pb)
        val depthScore = calibratedDepthScore(a, b)
        val distance3d =
            if (
                a.calibratedDepthMeters != null &&
                b.calibratedDepthMeters != null
            ) {
                kotlin.math.sqrt(
                    distance2d * distance2d +
                        (
                            a.calibratedDepthMeters -
                                b.calibratedDepthMeters
                        ) *
                        (
                            a.calibratedDepthMeters -
                                b.calibratedDepthMeters
                        )
                )
            } else {
                distance2d
            }
        (
            (
                1.0 -
                    distance3d /
                        config.geometryToleranceMeters
            ).coerceIn(0.0, 1.0) * 0.78 +
                depthScore * 0.22
        ).coerceIn(0.0, 1.0)
    }

    private fun scaleScore(
        a: MultiCameraSubjectAssociationEngine.Observation,
        b: MultiCameraSubjectAssociationEngine.Observation,
    ): Double {
        val aw = a.bounds.width.coerceAtLeast(1.0)
        val bw = b.bounds.width.coerceAtLeast(1.0)
        val ratio = max(aw, bw) /
            min(aw, bw).coerceAtLeast(1e-6)
        return exp(
            -abs(kotlin.math.ln(ratio)),
        ).coerceIn(0.0, 1.0)
    }

    private fun temporalConsistency(
        observations: List<MultiCameraSubjectAssociationEngine.Observation>,
    ): Double {
        if (observations.size < 2) return 1.0
        var worst = 0.0
        for (i in 0 until observations.lastIndex) {
            for (j in i + 1 until observations.size) {
                worst = max(
                    worst,
                    abs(
                        observations[i].timestampNanos -
                            observations[j].timestampNanos,
                    ).toDouble(),
                )
            }
        }
        return exp(
            -worst /
                (config.maximumClockSkewMillis * 1e6)
                    .coerceAtLeast(1.0),
        ).coerceIn(0.0, 1.0)
    }

    private fun geometryConsistency(
        observations: List<MultiCameraSubjectAssociationEngine.Observation>,
    ): Double {
        val points = observations.mapNotNull {
            it.calibratedCenterMeters
        }
        if (points.size < 2) return 0.62

        var maximum = 0.0
        for (i in 0 until points.lastIndex) {
            for (j in i + 1 until points.size) {
                maximum = max(
                    maximum,
                    points[i].distanceTo(points[j]),
                )
            }
        }

        val mean = points
            .map { point ->
                points
                    .filter { other -> other !== point }
                    .minOfOrNull { other -> point.distanceTo(other) }
                    ?: 0.0
            }
            .average()

        return (
            1.0 -
                (
                    max(0.0, maximum - mean) /
                        config.geometryToleranceMeters
                )
            ).coerceIn(0.0, 1.0)
    }

    private fun calibratedDepthScore(
        a: MultiCameraSubjectAssociationEngine.Observation,
        b: MultiCameraSubjectAssociationEngine.Observation,
    ): Double {
        val da = a.calibratedDepthMeters ?: return 0.58
        val db = b.calibratedDepthMeters ?: return 0.58
        val disagreement =
            abs(da - db)
        return (
            1.0 -
                disagreement /
                    config.maximumDepthDisagreementMeters
        ).coerceIn(0.0, 1.0)
    }

    private fun depthConsistency(
        observations: List<MultiCameraSubjectAssociationEngine.Observation>,
    ): Double {
        val depths =
            observations.mapNotNull {
                it.calibratedDepthMeters
            }
        if (depths.size < 2) return 0.58

        val medianDepth = median(depths)
        val maximumDeviation =
            depths.maxOfOrNull {
                abs(it - medianDepth)
            } ?: 0.0

        return (
            1.0 -
                maximumDeviation /
                    config.maximumDepthDisagreementMeters
        ).coerceIn(0.0, 1.0)
    }

    private fun scaleConsistency(
        observations: List<MultiCameraSubjectAssociationEngine.Observation>,
    ): Double {
        if (observations.size < 2) return 1.0
        val widths = observations.map {
            it.bounds.width.coerceAtLeast(1.0)
        }
        val mean = widths.average().coerceAtLeast(1e-6)
        val meanAbsoluteRelativeDeviation =
            widths
                .map { abs(it - mean) / mean }
                .average()

        return (
            1.0 -
                meanAbsoluteRelativeDeviation / 0.80
            ).coerceIn(0.0, 1.0)
    }

    private fun confidenceConsistency(
        observations: List<MultiCameraSubjectAssociationEngine.Observation>,
    ): Double {
        if (observations.isEmpty()) return 0.0
        val values = observations.map { it.confidence.coerceIn(0.0, 1.0) }
        val mean = values.average()
        val deviation = values.map { abs(it - mean) }.average()
        return (mean * 0.75 + (1.0 - deviation) * 0.25)
            .coerceIn(0.0, 1.0)
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

}
