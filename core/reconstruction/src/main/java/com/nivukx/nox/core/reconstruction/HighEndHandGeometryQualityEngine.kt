package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Hand3DFrame
import com.nivukx.nox.core.model.Hand3DLandmark
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * Production hand-geometry quality stage.
 *
 * Hand3DReconstructionEngine is responsible for correspondence, triangulation and basic bone
 * stabilization. This engine deliberately sits after reconstruction and evaluates whether the
 * resulting geometry is physically coherent enough to be trusted by downstream motion capture.
 *
 * The engine is non-mutating. It never "fixes" landmarks to make a poor reconstruction look good.
 * A consumer can use the returned score to gate output, request denser observations, or surface a
 * diagnostic to the operator.
 */
class HighEndHandGeometryQualityEngine(
    initialConfig: Config = Config(),
) {
    @Volatile
    private var config: Config = initialConfig

    data class Config(
        val minimumLandmarkConfidence: Double = 0.20,
        val segmentToleranceRatio: Double = 0.28,
        val hardSegmentToleranceRatio: Double = 0.55,
        val palmPlanarityTolerance: Double = 0.035,
        val temporalInnovationSoftMeters: Double = 0.045,
        val temporalInnovationHardMeters: Double = 0.16,
        val temporalConfidenceRetention: Double = 0.72,
        val articulationWeight: Double = 0.18,
        val segmentWeight: Double = 0.28,
        val palmWeight: Double = 0.18,
        val temporalWeight: Double = 0.22,
        val coverageWeight: Double = 0.14,
        val metricEvidenceWeight: Double = 0.14,
        val longitudinalBaselineLearningRate: Double = 0.06,
        val longitudinalBaselineMinimumSamples: Int = 3,
        val longitudinalBaselineUpdateToleranceRatio: Double = 0.18,
        val longitudinalBaselineWeight: Double = 0.42,
    ) {
        init {
            require(minimumLandmarkConfidence in 0.0..1.0)
            require(segmentToleranceRatio > 0.0)
            require(hardSegmentToleranceRatio > segmentToleranceRatio)
            require(palmPlanarityTolerance > 0.0)
            require(temporalInnovationSoftMeters > 0.0)
            require(temporalInnovationHardMeters > temporalInnovationSoftMeters)
            require(temporalConfidenceRetention in 0.0..1.0)
            require(
                articulationWeight +
                    segmentWeight +
                    palmWeight +
                    temporalWeight +
                    coverageWeight +
                    metricEvidenceWeight > 0.0,
            )
            require(longitudinalBaselineLearningRate in 0.0..1.0)
            require(longitudinalBaselineMinimumSamples >= 1)
            require(longitudinalBaselineUpdateToleranceRatio > 0.0)
            require(longitudinalBaselineWeight in 0.0..1.0)
        }
    }

    data class HandResult(
        val handId: Long,
        val confidence: Double,
        val segmentConsistency: Double,
        val longitudinalSegmentConsistency: Double = 0.78,
        val palmPlanarity: Double,
        val articulation: Double,
        val temporalStability: Double,
        val landmarkCoverage: Double,
        val innovationMeters: Double,
        val metricEvidenceAuthority: Double = 0.0,
        val epipolarConsistencyScore: Double = 1.0,
        val strictSubMillimeterQualified: Boolean = false,
        val hardFailure: Boolean,
        val warnings: List<String>,
    )

    data class Result(
        val timestampNanos: Long,
        val confidence: Double,
        val handCount: Int,
        val hardFailureCount: Int,
        val meanSegmentConsistency: Double,
        val meanLongitudinalSegmentConsistency: Double = 0.78,
        val meanPalmPlanarity: Double,
        val meanArticulation: Double,
        val temporalStability: Double,
        val landmarkCoverage: Double,
        val meanInnovationMeters: Double,
        val meanMetricEvidenceAuthority: Double = 0.0,
        val meanEpipolarConsistencyScore: Double = 1.0,
        val warnings: List<String>,
        val hands: List<HandResult>,
    ) {
        val accepted: Boolean
            get() = confidence >= 0.45 &&
                hardFailureCount == 0

        val bottleneck: String?
            get() = hands
                .flatMap { hand ->
                    hand.warnings.map { hand.handId to it }
                }
                .groupingBy { it.second }
                .eachCount()
                .maxByOrNull { it.value }
                ?.key
    }

    private data class TemporalState(
        val positions: Map<Int, Vec3>,
        val timestampNanos: Long,
    )

    private data class SegmentBaseline(
        var lengthMeters: Double,
        var samples: Int,
        var confidence: Double,
    )

    private val previous =
        HashMap<Long, TemporalState>()

    private val baselines =
        HashMap<Pair<Long, Pair<Int, Int>>, SegmentBaseline>()

    fun evaluate(frame: Hand3DFrame): Result {
        val results = frame.hands.map { hand ->
            evaluateHand(
                frameTimestampNanos = frame.timestampNanos,
                hand = hand,
            )
        }

        if (results.isEmpty()) {
            return Result(
                timestampNanos = frame.timestampNanos,
                confidence = 0.0,
                handCount = 0,
                hardFailureCount = 1,
                meanSegmentConsistency = 0.0,
                meanLongitudinalSegmentConsistency = 0.0,
                meanPalmPlanarity = 0.0,
                meanArticulation = 0.0,
                temporalStability = 0.0,
                landmarkCoverage = 0.0,
                meanInnovationMeters = config.temporalInnovationHardMeters,
                meanMetricEvidenceAuthority = 0.0,
                meanEpipolarConsistencyScore = 0.0,
                warnings = listOf("hand-geometry-no-valid-hands"),
                hands = emptyList(),
            )
        }

        val confidence =
            results.map { it.confidence }.average().coerceIn(0.0, 1.0)

        val warnings =
            results.flatMap { it.warnings }
                .distinct()
                .take(12)

        return Result(
            timestampNanos = frame.timestampNanos,
            confidence = confidence,
            handCount = results.size,
            hardFailureCount = results.count { it.hardFailure },
            meanSegmentConsistency =
                results.map { it.segmentConsistency }.average(),
            meanLongitudinalSegmentConsistency =
                results.map {
                    it.longitudinalSegmentConsistency
                }.average(),
            meanPalmPlanarity =
                results.map { it.palmPlanarity }.average(),
            meanArticulation =
                results.map { it.articulation }.average(),
            temporalStability =
                results.map { it.temporalStability }.average(),
            landmarkCoverage =
                results.map { it.landmarkCoverage }.average(),
            meanInnovationMeters =
                results.map { it.innovationMeters }.average(),
            meanMetricEvidenceAuthority =
                results.map { it.metricEvidenceAuthority }.average(),
            meanEpipolarConsistencyScore =
                results.map { it.epipolarConsistencyScore }.average(),
            warnings = warnings,
            hands = results,
        )
    }

    @Synchronized
    fun configureStrictness(strictness: Double) {
        val normalized =
            strictness.coerceIn(0.0, 1.0)

        config =
            Config(
                minimumLandmarkConfidence =
                    (
                        0.16 +
                            normalized * 0.16
                        ).coerceIn(0.16, 0.32),
                segmentToleranceRatio =
                    (
                        0.32 -
                            normalized * 0.12
                        ).coerceIn(0.20, 0.32),
                hardSegmentToleranceRatio =
                    (
                        0.64 -
                            normalized * 0.18
                        ).coerceIn(0.46, 0.64),
                palmPlanarityTolerance =
                    (
                        0.045 -
                            normalized * 0.015
                        ).coerceIn(0.030, 0.045),
                temporalInnovationSoftMeters =
                    (
                        0.055 -
                            normalized * 0.020
                        ).coerceIn(0.035, 0.055),
                temporalInnovationHardMeters =
                    (
                        0.20 -
                            normalized * 0.060
                        ).coerceIn(0.14, 0.20),
                temporalConfidenceRetention =
                    (
                        0.80 -
                            normalized * 0.10
                        ).coerceIn(0.70, 0.80),
                articulationWeight = 0.18,
                segmentWeight =
                    (
                        0.24 +
                            normalized * 0.08
                        ).coerceIn(0.24, 0.32),
                palmWeight = 0.18,
                temporalWeight = 0.22,
                coverageWeight =
                    (
                        0.18 -
                            normalized * 0.06
                        ).coerceIn(0.12, 0.18),
                longitudinalBaselineLearningRate =
                    (
                        0.055 -
                            normalized * 0.025
                        ).coerceIn(0.030, 0.055),
                longitudinalBaselineMinimumSamples =
                    (
                        3 +
                            (normalized * 3.0).toInt()
                    ).coerceIn(3, 6),
                longitudinalBaselineUpdateToleranceRatio =
                    (
                        0.22 -
                            normalized * 0.08
                        ).coerceIn(0.14, 0.22),
                longitudinalBaselineWeight =
                    (
                        0.22 +
                            normalized * 0.32
                        ).coerceIn(0.22, 0.54),
            )

        reset()
    }

    fun configuration(): Config = config

    fun reset() {
        previous.clear()
        baselines.clear()
    }

    private fun evaluateHand(
        frameTimestampNanos: Long,
        hand: com.nivukx.nox.core.model.Hand3DResult,
    ): HandResult {
        val visible = hand.landmarks.filter {
            it.confidence >= config.minimumLandmarkConfidence &&
                it.position.x.isFinite() &&
                it.position.y.isFinite() &&
                it.position.z.isFinite()
        }

        val coverage =
            (visible.size / 21.0).coerceIn(0.0, 1.0)

        val segment =
            segmentConsistency(
                handId = hand.handId,
                landmarks = visible,
            )
        val palm = palmPlanarity(visible)
        val articulation = articulationQuality(visible)
        val temporal = temporalQuality(
            handId = hand.handId,
            timestampNanos = frameTimestampNanos,
            landmarks = visible,
        )

        val longitudinalSegment =
            segment.longitudinalScore

        val combinedSegmentScore =
            (
                segment.score * (
                    1.0 -
                        config.longitudinalBaselineWeight
                    ) +
                    longitudinalSegment *
                        config.longitudinalBaselineWeight
                ).coerceIn(0.0, 1.0)

        val metricEvidenceAuthority =
            hand.metricEvidenceAuthority.coerceIn(0.0, 1.0)
        val epipolarConsistencyScore =
            hand.symmetricEpipolarConsistencyScore.coerceIn(0.0, 1.0)
        val strictSubMillimeterQualified =
            hand.strictSubMillimeterQualified &&
                hand.landmarks.size == 21 &&
                hand.meanUncertaintyMeters <= 0.0010 &&
                hand.reprojectionErrorPx <= 0.75 &&
                metricEvidenceAuthority >= 0.85 &&
                epipolarConsistencyScore >= 0.85

        val rawConfidence = (
            combinedSegmentScore * config.segmentWeight +
                palm * config.palmWeight +
                articulation * config.articulationWeight +
                temporal.score * config.temporalWeight +
                coverage * config.coverageWeight +
                metricEvidenceAuthority *
                    config.metricEvidenceWeight
        ).div(
            config.segmentWeight +
                config.palmWeight +
                config.articulationWeight +
                config.temporalWeight +
                config.coverageWeight +
                config.metricEvidenceWeight,
        ).coerceIn(0.0, 1.0)

        val hardFailure =
            segment.hardFailure ||
                segment.longitudinalHardFailure ||
                temporal.hardFailure ||
                epipolarConsistencyScore < 0.18

        val confidence =
            if (hardFailure) min(rawConfidence, 0.70) else rawConfidence

        val warnings = buildList {
            if (coverage < 0.70) {
                add("hand-landmark-coverage-low")
            }
            if (segment.score < 0.55) {
                add("hand-segment-consistency-low")
            }
            if (segment.hardFailure) {
                add("hand-segment-consistency-hard-failure")
            }
            if (longitudinalSegment < 0.55) {
                add("hand-longitudinal-anatomy-consistency-low")
            }
            if (segment.longitudinalHardFailure) {
                add("hand-longitudinal-anatomy-hard-failure")
            }
            if (palm < 0.55) {
                add("hand-palm-plane-unstable")
            }
            if (articulation < 0.55) {
                add("hand-articulation-inconsistent")
            }
            if (temporal.score < 0.55) {
                add("hand-temporal-innovation-high")
            }
            if (temporal.hardFailure) {
                add("hand-temporal-innovation-hard-failure")
            }
            if (metricEvidenceAuthority < 0.45) {
                add("hand-metric-evidence-authority-low")
            }
            if (epipolarConsistencyScore < 0.45) {
                add("hand-epipolar-consistency-low")
            }
            if (epipolarConsistencyScore < 0.18) {
                add("hand-epipolar-consistency-hard-failure")
            }
            if (
                !strictSubMillimeterQualified &&
                    hand.confidence >= 0.85
            ) {
                add(
                    "hand-submillimeter-metric-output-not-qualified",
                )
            }
        }.distinct()

        return HandResult(
            handId = hand.handId,
            confidence = confidence,
            segmentConsistency = segment.score,
            palmPlanarity = palm,
            articulation = articulation,
            temporalStability = temporal.score,
            landmarkCoverage = coverage,
            innovationMeters = temporal.innovation,
            metricEvidenceAuthority = metricEvidenceAuthority,
            epipolarConsistencyScore = epipolarConsistencyScore,
            strictSubMillimeterQualified =
                strictSubMillimeterQualified,
            hardFailure = hardFailure,
            longitudinalSegmentConsistency =
                longitudinalSegment,
            warnings = warnings,
        )
    }

    private data class SegmentResult(
        val score: Double,
        val hardFailure: Boolean,
        val longitudinalScore: Double,
        val longitudinalHardFailure: Boolean,
    )

    private fun segmentConsistency(
        handId: Long,
        landmarks: List<Hand3DLandmark>,
    ): SegmentResult {
        val map = landmarks.associateBy { it.index }
        val chains = listOf(
            listOf(0, 1, 2, 3, 4),
            listOf(0, 5, 6, 7, 8),
            listOf(0, 9, 10, 11, 12),
            listOf(0, 13, 14, 15, 16),
            listOf(0, 17, 18, 19, 20),
        )

        val deviations = mutableListOf<Double>()
        val longitudinalDeviations = mutableListOf<Double>()
        var hardCount = 0
        var longitudinalHardCount = 0

        chains.forEach { chain ->
            val lengths =
                chain.zipWithNext().mapNotNull { (a, b) ->
                    val pa = map[a]?.position
                        ?: return@mapNotNull null
                    val pb = map[b]?.position
                        ?: return@mapNotNull null
                    val distance = pa.distanceTo(pb)
                    distance.takeIf {
                        it.isFinite() && it > 1e-5
                    }
                }

            if (lengths.size < 2) return@forEach

            val reference = median(lengths)
            if (!reference.isFinite() || reference <= 1e-6) {
                return@forEach
            }

            lengths.forEachIndexed { index, length ->
                val deviation =
                    abs(length - reference) /
                        reference
                deviations += deviation
                if (deviation > config.hardSegmentToleranceRatio) {
                    hardCount += 1
                }

                val start = chain[index]
                val end = chain[index + 1]
                val startLandmark = map[start]
                val endLandmark = map[end]
                if (
                    startLandmark != null &&
                    endLandmark != null &&
                    startLandmark.confidence >=
                        config.minimumLandmarkConfidence &&
                    endLandmark.confidence >=
                        config.minimumLandmarkConfidence
                ) {
                    val key = handId to (start to end)
                    val baseline = baselines[key]
                    if (baseline == null) {
                        if (
                            deviation <=
                                config.longitudinalBaselineUpdateToleranceRatio *
                                    0.72 &&
                            startLandmark.confidence >= 0.84 &&
                            endLandmark.confidence >= 0.84
                        ) {
                            baselines[key] =
                                SegmentBaseline(
                                    lengthMeters = length,
                                    samples = 1,
                                    confidence =
                                        (
                                            startLandmark.confidence *
                                                endLandmark.confidence
                                            ).coerceIn(0.0, 1.0),
                                )
                        }
                    } else {
                        val baselineDeviation =
                            abs(length - baseline.lengthMeters) /
                                baseline.lengthMeters
                                    .coerceAtLeast(1e-6)
                        if (
                            baseline.samples >=
                                config.longitudinalBaselineMinimumSamples
                        ) {
                            longitudinalDeviations +=
                                baselineDeviation
                        }

                        if (
                            baselineDeviation >
                                config.hardSegmentToleranceRatio
                        ) {
                            longitudinalHardCount += 1
                        }

                        if (
                            baselineDeviation <=
                                config.longitudinalBaselineUpdateToleranceRatio
                        ) {
                            val alpha =
                                (
                                    config.longitudinalBaselineLearningRate *
                                        (
                                            startLandmark.confidence +
                                                endLandmark.confidence
                                            ) * 0.5
                                    ).coerceIn(0.0, 1.0)
                            baseline.lengthMeters =
                                baseline.lengthMeters *
                                    (1.0 - alpha) +
                                    length * alpha
                            baseline.samples += 1
                            baseline.confidence =
                                (
                                    baseline.confidence * 0.94 +
                                        (
                                            startLandmark.confidence *
                                                endLandmark.confidence
                                            ).coerceIn(0.0, 1.0) * 0.06
                                    ).coerceIn(0.0, 1.0)
                        }
                    }
                }
            }
        }

        if (deviations.size < 8) {
            return SegmentResult(
                score = 0.0,
                hardFailure = true,
                longitudinalScore = 0.0,
                longitudinalHardFailure = false,
            )
        }

        val meanDeviation = deviations.average()
        val score =
            (
                1.0 -
                    meanDeviation /
                    config.segmentToleranceRatio
            ).coerceIn(0.0, 1.0)

        val longitudinalScore =
            if (longitudinalDeviations.isEmpty()) {
                0.78
            } else {
                (
                    1.0 -
                        longitudinalDeviations.average() /
                            config.segmentToleranceRatio
                    ).coerceIn(0.0, 1.0)
            }

        return SegmentResult(
            score = score,
            hardFailure = hardCount >= 2,
            longitudinalScore = longitudinalScore,
            longitudinalHardFailure = longitudinalHardCount >= 2,
        )
    }

    private fun palmPlanarity(
        landmarks: List<Hand3DLandmark>,
    ): Double {
        val map = landmarks.associateBy { it.index }
        val wrist = map[0]?.position ?: return 0.0
        val indexMcp = map[5]?.position ?: return 0.0
        val middleMcp = map[9]?.position ?: return 0.0
        val pinkyMcp = map[17]?.position ?: return 0.0

        val a = indexMcp - wrist
        val b = pinkyMcp - wrist
        val normal = a.cross(b)
        val norm = normal.norm()

        if (!norm.isFinite() || norm < 1e-6) {
            return 0.0
        }

        val unit = normal * (1.0 / norm)
        val points = listOf(
            map[1]?.position,
            indexMcp,
            middleMcp,
            map[13]?.position,
            pinkyMcp,
        ).filterNotNull()

        if (points.size < 4) return 0.0

        val residuals = points.map { point ->
            abs((point - wrist).dot(unit))
        }
        val meanResidual = residuals.average()
        return (
            1.0 -
                meanResidual /
                config.palmPlanarityTolerance
        ).coerceIn(0.0, 1.0)
    }

    private fun articulationQuality(
        landmarks: List<Hand3DLandmark>,
    ): Double {
        val map = landmarks.associateBy { it.index }
        val fingerTriples = listOf(
            Triple(1, 2, 3),
            Triple(2, 3, 4),
            Triple(5, 6, 7),
            Triple(6, 7, 8),
            Triple(9, 10, 11),
            Triple(10, 11, 12),
            Triple(13, 14, 15),
            Triple(14, 15, 16),
            Triple(17, 18, 19),
            Triple(18, 19, 20),
        )

        val values = fingerTriples.mapNotNull { (a, b, c) ->
            val pa = map[a]?.position ?: return@mapNotNull null
            val pb = map[b]?.position ?: return@mapNotNull null
            val pc = map[c]?.position ?: return@mapNotNull null

            val ab = (pa - pb).normalized()
            val cb = (pc - pb).normalized()
            val dot = ab.dot(cb).coerceIn(-1.0, 1.0)
            val angle = acos(dot)
            val straightness =
                abs(angle) / Math.PI
            straightness.coerceIn(0.0, 1.0)
        }

        return if (values.isEmpty()) 0.0 else values.average()
    }

    private data class TemporalResult(
        val score: Double,
        val innovation: Double,
        val hardFailure: Boolean,
    )

    private fun temporalQuality(
        handId: Long,
        timestampNanos: Long,
        landmarks: List<Hand3DLandmark>,
    ): TemporalResult {
        val positions = landmarks.associate {
            it.index to it.position
        }

        val old = previous[handId]
        previous[handId] =
            TemporalState(
                positions = positions,
                timestampNanos = timestampNanos,
            )

        if (old == null) {
            return TemporalResult(
                score = config.temporalConfidenceRetention,
                innovation = 0.0,
                hardFailure = false,
            )
        }

        val dt = (
            timestampNanos - old.timestampNanos
        ).toDouble() / 1e9

        if (!dt.isFinite() || dt <= 0.0 || dt > 0.5) {
            return TemporalResult(
                score = 0.35,
                innovation = config.temporalInnovationHardMeters,
                hardFailure = true,
            )
        }

        val shared = positions.keys.intersect(old.positions.keys)
        if (shared.size < 8) {
            return TemporalResult(
                score = 0.38,
                innovation = config.temporalInnovationHardMeters,
                hardFailure = true,
            )
        }

        val innovations = shared.map {
            positions.getValue(it).distanceTo(
                old.positions.getValue(it),
            )
        }

        val mean = innovations.average()
        val maxInnovation = innovations.maxOrNull() ?: mean

        val score =
            if (mean <= config.temporalInnovationSoftMeters) {
                1.0 -
                    0.25 *
                    (mean / config.temporalInnovationSoftMeters)
            } else {
                0.75 *
                    (
                        1.0 -
                            (
                                (
                                    mean -
                                        config.temporalInnovationSoftMeters
                                    ) /
                                    (
                                        config.temporalInnovationHardMeters -
                                            config.temporalInnovationSoftMeters
                                        )
                                ).coerceIn(0.0, 1.0)
                        )
            }

        return TemporalResult(
            score = score.coerceIn(0.0, 1.0),
            innovation = mean,
            hardFailure =
                maxInnovation >
                    config.temporalInnovationHardMeters,
        )
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
        }
    }
}
