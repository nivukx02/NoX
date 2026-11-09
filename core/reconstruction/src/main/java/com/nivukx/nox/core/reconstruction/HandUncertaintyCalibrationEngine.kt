package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Hand3DFrame
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Empirical uncertainty calibration for metric 3D hand reconstruction.
 *
 * The hand reconstruction pipeline already reports geometric uncertainty, reprojection quality,
 * temporal innovation and confidence. This engine calibrates those signals against recent
 * empirical residuals so downstream systems can distinguish "a number was produced" from
 * "the number has historically behaved as expected".
 *
 * It deliberately does not invent geometry or change hand landmarks. It only produces a
 * reliability surface:
 * - empirical residual coverage;
 * - uncertainty radius;
 * - OOD risk;
 * - per-hand calibration confidence;
 * - bounded dynamic/occlusion inflation;
 * - hysteretic production policy.
 *
 * Current-frame residuals are evaluated against history that existed before the current frame.
 * This prevents pathological samples from widening their own calibration distribution.
 */
class HandUncertaintyCalibrationEngine(
    initialConfig: Config = Config(),
) {

    enum class Policy {
        NORMAL,
        CAUTION,
        HOLD,
        REJECT,
    }

    data class Config(
        val targetCoverage: Double = 0.95,
        val minimumCalibrationSamples: Int = 6,
        val maximumResidualSamples: Int = 96,
        val residualFallbackMeters: Double = 0.018,
        val maximumRadiusMeters: Double = 0.22,
        val warningRadiusMeters: Double = 0.08,
        val dynamicInflationWeight: Double = 0.16,
        val occlusionInflationWeight: Double = 0.22,
        val synchronizationInflationWeight: Double = 0.10,
        val oodThreshold: Double = 0.82,
        val holdOodThreshold: Double = 0.62,
        val historyAlpha: Double = 0.20,
        val confidenceWeight: Double = 0.36,
        val empiricalCoverageWeight: Double = 0.28,
        val residualWeight: Double = 0.20,
        val trendWeight: Double = 0.16,
        val minimumHandConfidence: Double = 0.20,
        val warningPolicyScore: Double = 0.60,
        val holdPolicyScore: Double = 0.42,
    ) {
        init {
            require(targetCoverage in 0.50..0.999)
            require(minimumCalibrationSamples >= 2)
            require(maximumResidualSamples >= minimumCalibrationSamples)
            require(residualFallbackMeters > 0.0)
            require(maximumRadiusMeters > warningRadiusMeters)
            require(warningRadiusMeters > 0.0)
            require(dynamicInflationWeight >= 0.0)
            require(occlusionInflationWeight >= 0.0)
            require(synchronizationInflationWeight >= 0.0)
            require(oodThreshold in 0.0..1.0)
            require(holdOodThreshold in 0.0..oodThreshold)
            require(historyAlpha in 0.0..1.0)
            require(confidenceWeight >= 0.0)
            require(empiricalCoverageWeight >= 0.0)
            require(residualWeight >= 0.0)
            require(trendWeight >= 0.0)
            require(minimumHandConfidence in 0.0..1.0)
            require(holdPolicyScore in 0.0..warningPolicyScore)
            require(warningPolicyScore <= 1.0)
        }
    }

    data class HandEnvelope(
        val handId: Long,
        val radiusMeters: Double,
        val empiricalCoverage: Double,
        val normalizedInnovation: Double,
        val outOfDistributionRisk: Double,
        val measurementUncertaintyMeters: Double,
        val confidence: Double,
        val calibrationReady: Boolean,
        val predictionOnly: Boolean = false,
        val warning: String? = null,
    )

    data class Result(
        val timestampNanos: Long,
        val targetCoverage: Double,
        val calibrationReady: Boolean,
        val measuredHandCount: Int,
        val calibratedHandCount: Int,
        val meanRadiusMeters: Double,
        val p95RadiusMeters: Double,
        val empiricalCoverage: Double,
        val uncertaintyConfidence: Double,
        val outOfDistributionRisk: Double,
        val dynamicInflation: Double,
        val occlusionInflation: Double,
        val synchronizationInflation: Double,
        val calibrationProgress: Double,
        val policy: Policy,
        val warnings: List<String>,
        val recommendations: List<String>,
        val envelopes: List<HandEnvelope>,
    )

    private data class State(
        val residuals: ArrayDeque<Double> = ArrayDeque(),
        var smoothedScore: Double = 0.0,
        var lastCoverage: Double = 0.0,
        var goodFrames: Int = 0,
        var badFrames: Int = 0,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
    )

    private var config = initialConfig
    private val states = HashMap<Long, State>()
    private var lastResult: Result? = null

    @Synchronized
    fun configure(newConfig: Config) {
        if (newConfig == config) return
        config = newConfig
        states.values.forEach {
            while (it.residuals.size > config.maximumResidualSamples) {
                it.residuals.removeFirst()
            }
        }
    }

    fun configuration(): Config = config

    fun snapshot(): Result? = lastResult

    @Synchronized
    fun evaluate(
        frame: Hand3DFrame,
        dynamicStress: Double = 0.0,
        occlusionRisk: Double = 0.0,
        synchronizationConfidence: Double = 1.0,
    ): Result {
        lastResult?.let { previous ->
            if (frame.timestampNanos <= previous.timestampNanos) {
                return previous.copy(
                    warnings =
                        (
                            previous.warnings +
                                "stale-hand-frame-rejected"
                            ).distinct(),
                )
            }
        }
        if (frame.hands.isEmpty()) {
            return blocked(
                timestampNanos = frame.timestampNanos,
                warning = "hand-uncertainty-no-reconstructed-hands",
            )
        }

        val dynamicInflation =
            (
                dynamicStress.coerceIn(0.0, 1.0) *
                    config.dynamicInflationWeight
                ).coerceIn(0.0, 0.40)

        val occlusionInflation =
            (
                occlusionRisk.coerceIn(0.0, 1.0) *
                    config.occlusionInflationWeight
                ).coerceIn(0.0, 0.45)

        val synchronizationInflation =
            (
                (
                    1.0 -
                        synchronizationConfidence.coerceIn(0.0, 1.0)
                    ) *
                    config.synchronizationInflationWeight
                ).coerceIn(0.0, 0.30)

        val envelopes =
            frame.hands.map { hand ->
                evaluateHand(
                    hand = hand,
                    timestampNanos = frame.timestampNanos,
                    dynamicInflation = dynamicInflation,
                    occlusionInflation = occlusionInflation,
                    synchronizationInflation = synchronizationInflation,
                )
            }

        val calibrated =
            envelopes.filter { it.calibrationReady }

        val meanRadius =
            envelopes
                .map { it.radiusMeters }
                .average()
                .coerceAtLeast(0.0)

        val p95Radius =
            percentile(
                envelopes.map { it.radiusMeters },
                0.95,
            )

        val empiricalCoverage =
            if (calibrated.isEmpty()) {
                0.0
            } else {
                calibrated.map { it.empiricalCoverage }.average()
            }

        val confidence =
            if (envelopes.isEmpty()) {
                0.0
            } else {
                envelopes.map { it.confidence }.average()
            }

        val oodRisk =
            envelopes
                .maxOfOrNull { it.outOfDistributionRisk }
                ?: 0.0

        val calibrationProgress =
            (
                states.values
                    .maxOfOrNull {
                        min(
                            1.0,
                            it.residuals.size.toDouble() /
                                config.minimumCalibrationSamples.toDouble(),
                        )
                    }
                    ?: 0.0
                ).coerceIn(0.0, 1.0)

        val rawPolicyScore =
            (
                confidence * config.confidenceWeight +
                    empiricalCoverage * config.empiricalCoverageWeight +
                    (
                        1.0 -
                            (
                                meanRadius /
                                    config.maximumRadiusMeters
                            ).coerceIn(0.0, 1.0)
                        ) * config.residualWeight +
                    (
                        if (oodRisk < config.holdOodThreshold) {
                            1.0
                        } else {
                            (
                                1.0 -
                                    oodRisk
                                ).coerceIn(0.0, 1.0)
                        }
                        ) * config.trendWeight
                ).coerceIn(0.0, 1.0)

        val smoothed =
            states.values
                .map { it.smoothedScore }
                .filter { it.isFinite() && it > 0.0 }
                .takeIf { it.isNotEmpty() }
                ?.average()
                ?.coerceIn(0.0, 1.0)
                ?: rawPolicyScore

        val policy =
            when {
                oodRisk >= config.oodThreshold -> Policy.REJECT
                smoothed < config.holdPolicyScore -> Policy.HOLD
                smoothed < config.warningPolicyScore -> Policy.CAUTION
                else -> Policy.NORMAL
            }

        val warnings =
            buildWarnings(
                envelopes = envelopes,
                meanRadius = meanRadius,
                empiricalCoverage = empiricalCoverage,
                uncertaintyConfidence = confidence,
                oodRisk = oodRisk,
                policy = policy,
            )

        val recommendations =
            buildRecommendations(
                policy = policy,
                meanRadius = meanRadius,
                empiricalCoverage = empiricalCoverage,
                oodRisk = oodRisk,
                calibrationProgress = calibrationProgress,
            )

        val result =
            Result(
                timestampNanos = frame.timestampNanos,
                targetCoverage = config.targetCoverage,
                calibrationReady =
                    calibrated.size >=
                        min(
                            frame.hands.size,
                            1,
                        ),
                measuredHandCount = frame.hands.size,
                calibratedHandCount = calibrated.size,
                meanRadiusMeters = meanRadius,
                p95RadiusMeters = p95Radius,
                empiricalCoverage = empiricalCoverage.coerceIn(0.0, 1.0),
                uncertaintyConfidence = confidence.coerceIn(0.0, 1.0),
                outOfDistributionRisk = oodRisk.coerceIn(0.0, 1.0),
                dynamicInflation = dynamicInflation,
                occlusionInflation = occlusionInflation,
                synchronizationInflation = synchronizationInflation,
                calibrationProgress = calibrationProgress,
                policy = policy,
                warnings = warnings,
                recommendations = recommendations,
                envelopes = envelopes,
            )

        lastResult = result
        return result
    }

    @Synchronized
    fun reset() {
        states.clear()
        lastResult = null
    }

    private fun evaluateHand(
        hand: com.nivukx.nox.core.model.Hand3DResult,
        timestampNanos: Long,
        dynamicInflation: Double,
        occlusionInflation: Double,
        synchronizationInflation: Double,
    ): HandEnvelope {
        val state =
            states.getOrPut(hand.handId) {
                State()
            }

        val confidence =
            hand.confidence.coerceIn(0.0, 1.0)
        val measurementUncertainty =
            hand.meanUncertaintyMeters
                .coerceAtLeast(0.0)

        val estimatedResidual =
            (
                hand.reprojectionErrorPx.coerceAtLeast(0.0) / 100.0 +
                    measurementUncertainty
                ).coerceAtLeast(
                    config.residualFallbackMeters,
                )

        val history =
            state.residuals.toList()

        val medianResidual =
            median(history)
        val robustScale =
            robustScale(
                history = history,
                fallback = config.residualFallbackMeters,
            )

        val normalizedInnovation =
            if (history.size < 4) {
                estimatedResidual /
                    config.residualFallbackMeters
            } else {
                abs(
                    estimatedResidual -
                        medianResidual,
                ) /
                    robustScale.coerceAtLeast(1e-6)
            }

        val outOfDistributionRisk =
            if (history.size < 4) {
                (
                    normalizedInnovation /
                        5.0
                    ).coerceIn(0.0, 1.0)
            } else {
                (
                    1.0 -
                        exp(
                            -normalizedInnovation /
                                3.0,
                        )
                    ).coerceIn(0.0, 1.0)
            }

        val localCoverage =
            if (history.size < config.minimumCalibrationSamples) {
                0.0
            } else {
                empiricalCoverage(
                    history = history,
                    threshold =
                        estimatedResidual,
                )
            }

        val targetQuantile =
            if (history.size >= config.minimumCalibrationSamples) {
                percentile(
                    history,
                    config.targetCoverage,
                )
            } else {
                config.residualFallbackMeters
            }

        val radius =
            (
                max(
                    measurementUncertainty,
                    targetQuantile,
                ) *
                    (
                        1.0 +
                            dynamicInflation +
                            occlusionInflation +
                            synchronizationInflation
                        )
                ).coerceIn(
                    0.0,
                    config.maximumRadiusMeters,
                )

        val calibrationReady =
            history.size >=
                config.minimumCalibrationSamples

        val temporalTrend =
            (
                if (history.size >= 4) {
                    1.0 -
                        max(
                            0.0,
                            estimatedResidual -
                                medianResidual,
                        ) /
                        robustScale.coerceAtLeast(1e-6)
                } else {
                    0.72
                }
                ).coerceIn(0.0, 1.0)

        val handConfidence =
            (
                confidence * 0.44 +
                    (
                        1.0 -
                            measurementUncertainty /
                                config.maximumRadiusMeters
                        ).coerceIn(0.0, 1.0) * 0.26 +
                    (
                        if (calibrationReady) {
                            localCoverage
                        } else {
                            0.68
                        }
                        ) * 0.18 +
                    temporalTrend * 0.12
                ).coerceIn(0.0, 1.0)

        if (
            history.size < 4 ||
            outOfDistributionRisk < config.oodThreshold
        ) {
            addResidual(
                state = state,
                residual = estimatedResidual,
                timestampNanos = timestampNanos,
            )
        } else {
            state.lastTimestampNanos =
                max(
                    state.lastTimestampNanos,
                    timestampNanos,
                )
        }

        val warning =
            when {
                outOfDistributionRisk >= config.oodThreshold ->
                    "hand-uncertainty-out-of-distribution"
                radius >= config.warningRadiusMeters ->
                    "hand-uncertainty-radius-elevated"
                !calibrationReady ->
                    "hand-uncertainty-calibration-warming"
                localCoverage < config.targetCoverage * 0.82 ->
                    "hand-uncertainty-coverage-below-target"
                else -> null
            }

        return HandEnvelope(
            handId = hand.handId,
            radiusMeters = radius,
            empiricalCoverage = localCoverage,
            normalizedInnovation = normalizedInnovation,
            outOfDistributionRisk = outOfDistributionRisk,
            measurementUncertaintyMeters = measurementUncertainty,
            confidence = handConfidence,
            calibrationReady = calibrationReady,
            warning = warning,
        )
    }

    private fun addResidual(
        state: State,
        residual: Double,
        timestampNanos: Long,
    ) {
        state.residuals.addLast(
            residual.coerceAtLeast(0.0),
        )

        while (
            state.residuals.size >
                config.maximumResidualSamples
        ) {
            state.residuals.removeFirst()
        }

        state.lastCoverage =
            if (state.residuals.size >=
                config.minimumCalibrationSamples
            ) {
                percentile(
                    state.residuals.toList(),
                    config.targetCoverage,
                )
            } else {
                0.0
            }

        val score =
            (
                1.0 -
                    residual /
                        config.maximumRadiusMeters
                ).coerceIn(0.0, 1.0)

        state.smoothedScore =
            if (state.smoothedScore <= 0.0) {
                score
            } else {
                state.smoothedScore *
                    (1.0 - config.historyAlpha) +
                    score * config.historyAlpha
            }

        if (score >= config.warningPolicyScore) {
            state.goodFrames += 1
            state.badFrames =
                max(0, state.badFrames - 1)
        } else {
            state.badFrames += 1
            state.goodFrames = 0
        }

        state.lastTimestampNanos =
            max(
                state.lastTimestampNanos,
                timestampNanos,
            )
    }

    private fun buildWarnings(
        envelopes: List<HandEnvelope>,
        meanRadius: Double,
        empiricalCoverage: Double,
        uncertaintyConfidence: Double,
        oodRisk: Double,
        policy: Policy,
    ): List<String> =
        buildList {
            envelopes.mapNotNull { it.warning }
                .distinct()
                .forEach(::add)

            if (meanRadius >= config.warningRadiusMeters) {
                add("hand-uncertainty-mean-radius-elevated")
            }

            if (
                empiricalCoverage > 0.0 &&
                empiricalCoverage <
                    config.targetCoverage * 0.88
            ) {
                add("hand-uncertainty-empirical-coverage-low")
            }

            if (uncertaintyConfidence < 0.55) {
                add("hand-uncertainty-confidence-low")
            }

            if (oodRisk >= config.holdOodThreshold) {
                add("hand-uncertainty-ood-risk-elevated")
            }

            if (policy == Policy.HOLD) {
                add("hand-uncertainty-production-hold")
            }

            if (policy == Policy.REJECT) {
                add("hand-uncertainty-production-reject")
            }
        }.distinct()

    private fun buildRecommendations(
        policy: Policy,
        meanRadius: Double,
        empiricalCoverage: Double,
        oodRisk: Double,
        calibrationProgress: Double,
    ): List<String> =
        buildList {
            if (calibrationProgress < 1.0) {
                add(
                    "Continue stable multi-view hand observations until empirical uncertainty calibration matures.",
                )
            }
            if (meanRadius >= config.warningRadiusMeters) {
                add(
                    "Increase camera coverage and preserve more independent hand observations before trusting fine finger motion.",
                )
            }
            if (
                empiricalCoverage > 0.0 &&
                empiricalCoverage < config.targetCoverage * 0.88
            ) {
                add(
                    "Increase multi-view hand parallax or reduce occlusion before exporting high-precision articulation.",
                )
            }
            if (oodRisk >= config.holdOodThreshold) {
                add(
                    "Treat the hand uncertainty stream as out-of-distribution and prevent it from dominating identity continuity.",
                )
            }
            if (policy == Policy.HOLD || policy == Policy.REJECT) {
                add(
                    "Hold high-precision hand output until repeated evidence returns inside the calibrated envelope.",
                )
            }
        }.distinct().take(8)

    private fun empiricalCoverage(
        history: List<Double>,
        threshold: Double,
    ): Double {
        if (history.isEmpty()) return 0.0
        return history.count {
            it <= threshold
        }.toDouble() /
            history.size.toDouble()
    }

    private fun robustScale(
        history: List<Double>,
        fallback: Double,
    ): Double {
        if (history.isEmpty()) return fallback
        val center = median(history)
        val deviations =
            history.map {
                abs(it - center)
            }
        val mad = median(deviations)
        return max(
            fallback * 0.20,
            center + 1.4826 * mad,
        ).coerceAtLeast(1e-6)
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

    private fun percentile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        if (sorted.size == 1) return sorted.first()
        val position =
            fraction.coerceIn(0.0, 1.0) *
                (sorted.size - 1)
        val lower = position.toInt()
        val upper = min(
            lower + 1,
            sorted.lastIndex,
        )
        val blend = position - lower
        return sorted[lower] * (1.0 - blend) +
            sorted[upper] * blend
    }

    private fun blocked(
        timestampNanos: Long,
        warning: String,
    ): Result {
        val result =
            Result(
                timestampNanos = timestampNanos,
                targetCoverage = config.targetCoverage,
                calibrationReady = false,
                measuredHandCount = 0,
                calibratedHandCount = 0,
                meanRadiusMeters =
                    config.maximumRadiusMeters,
                p95RadiusMeters =
                    config.maximumRadiusMeters,
                empiricalCoverage = 0.0,
                uncertaintyConfidence = 0.0,
                outOfDistributionRisk = 1.0,
                dynamicInflation = 0.0,
                occlusionInflation = 0.0,
                synchronizationInflation = 0.0,
                calibrationProgress = 0.0,
                policy = Policy.REJECT,
                warnings = listOf(warning),
                recommendations =
                    listOf(
                        "Hold hand output until reconstructed metric hands are available.",
                    ),
                envelopes = emptyList(),
            )
        lastResult = result
        return result
    }
}
