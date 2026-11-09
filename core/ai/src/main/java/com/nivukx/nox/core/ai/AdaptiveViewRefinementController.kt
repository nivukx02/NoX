package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Selective high-cost refinement scheduler for difficult camera viewpoints.
 *
 * The controller keeps a constant-memory temporal state buffer, estimates viewpoint difficulty,
 * left/right ambiguity, short-term continuity and occlusion pressure, and only requests an expensive
 * refinement pass when the expected information gain exceeds its compute cost.
 *
 * This is a policy engine, not a neural model. It never fabricates a refinement result.
 */
class AdaptiveViewRefinementController(
    private val config: Config = Config(),
) {
    data class Config(
        val historyCapacity: Int = 32,
        val refinementThreshold: Double = 0.62,
        val forcedRefinementThreshold: Double = 0.86,
        val recoveryThreshold: Double = 0.64,
        val minimumFramesBetweenRefinements: Int = 2,
        val temporalContinuityWeight: Double = 0.17,
        val viewpointDifficultyWeight: Double = 0.33,
        val ambiguityWeight: Double = 0.18,
        val occlusionWeight: Double = 0.20,
        val confidenceLossWeight: Double = 0.12,
        val minimumExpectedGainToCostRatio: Double = 0.60,
        val maximumRefinementCost: Double = 1.35,
    ) {
        init {
            require(historyCapacity >= 8)
            require(refinementThreshold in 0.0..1.0)
            require(forcedRefinementThreshold >= refinementThreshold)
            require(recoveryThreshold >= refinementThreshold)
            require(minimumFramesBetweenRefinements >= 0)
            require(temporalContinuityWeight >= 0.0)
            require(viewpointDifficultyWeight >= 0.0)
            require(ambiguityWeight >= 0.0)
            require(occlusionWeight >= 0.0)
            require(confidenceLossWeight >= 0.0)
            require(
                minimumExpectedGainToCostRatio > 0.0
            )
            require(maximumRefinementCost > 0.0)
            require(
                temporalContinuityWeight +
                    viewpointDifficultyWeight +
                    ambiguityWeight +
                    occlusionWeight +
                    confidenceLossWeight > 0.0,
            )
        }
    }

    data class FrameEvidence(
        val frameIndex: Long,
        val timestampNanos: Long,
        val viewpointChallenge: Double,
        val leftRightAmbiguity: Double,
        val occlusionRisk: Double,
        val confidence: Double,
        val temporalStability: Double,
        val motionEnergy: Double = 0.0,
    ) {
        init {
            require(frameIndex >= 0L)
            require(timestampNanos >= 0L)
            listOf(
                viewpointChallenge,
                leftRightAmbiguity,
                occlusionRisk,
                confidence,
                temporalStability,
                motionEnergy,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
        }
    }

    data class Decision(
        val requestRefinement: Boolean,
        val forced: Boolean,
        val recovery: Boolean,
        val score: Double,
        val viewpointChallenge: Double,
        val ambiguity: Double,
        val occlusion: Double,
        val temporalContinuity: Double,
        val confidenceLoss: Double,
        val expectedGain: Double,
        val computeCost: Double,
        val efficiencyRatio: Double = 0.0,
        val reason: String,
        val retainedHistoryFrames: Int,
    )

    private var history: Array<FrameEvidence?> =
        arrayOfNulls(config.historyCapacity)
    private var historySize = 0
    private var historyCursor = 0
    private var lastRefinementFrame = Long.MIN_VALUE
    private var lastTimestampNanos = Long.MIN_VALUE

    @Synchronized
    fun evaluate(evidence: FrameEvidence): Decision {
        val continuity = temporalContinuity(evidence)
        val confidenceLoss = confidenceLoss(evidence)
        val score = weightedScore(
            evidence = evidence,
            continuity = continuity,
            confidenceLoss = confidenceLoss,
        )

        val framesSinceRefinement =
            if (lastRefinementFrame == Long.MIN_VALUE) {
                Int.MAX_VALUE
            } else {
                (evidence.frameIndex - lastRefinementFrame)
                    .coerceAtLeast(0L)
                    .coerceAtMost(Int.MAX_VALUE.toLong())
                    .toInt()
            }

        val forced =
            score >= config.forcedRefinementThreshold ||
                evidence.confidence <= 0.25 &&
                evidence.viewpointChallenge >= 0.55

        val recovery =
            score >= config.recoveryThreshold &&
                (
                    evidence.occlusionRisk >= 0.62 ||
                        confidenceLoss >= 0.34
                    )

        val expectedGain = (
            score *
                (0.55 + 0.45 * evidence.motionEnergy) *
                (1.0 + confidenceLoss * 0.55)
            ).coerceIn(0.0, 1.5)

        val computeCost = (
            0.48 +
                0.32 * evidence.viewpointChallenge +
                0.20 * evidence.motionEnergy
            ).coerceIn(
                0.0,
                config.maximumRefinementCost,
            )

        val gainCostRatio =
            if (computeCost <= 1e-6) {
                Double.POSITIVE_INFINITY
            } else {
                expectedGain / computeCost
            }

        val valueJustifiesRefinement =
            gainCostRatio >=
                config.minimumExpectedGainToCostRatio

        val spacingSatisfied =
            framesSinceRefinement >=
                config.minimumFramesBetweenRefinements

        val request =
            (
                forced ||
                    (
                        (recovery || score >= config.refinementThreshold) &&
                            valueJustifiesRefinement
                        )
                ) &&
                spacingSatisfied

        if (request) {
            lastRefinementFrame = evidence.frameIndex
        }

        append(evidence)
        lastTimestampNanos = evidence.timestampNanos

        val reason = when {
            forced -> "forced_view_refinement"
            recovery -> "occlusion_or_confidence_recovery"
            request -> "selective_view_refinement"
            evidence.viewpointChallenge >= 0.70 ->
                "challenging_view_but_temporally_stable"
            evidence.occlusionRisk >= 0.70 ->
                "occlusion_present_but_refinement_rate_limited"
            !valueJustifiesRefinement ->
                "refinement_value_below_compute_cost"
            else -> "temporal_state_sufficient"
        }

        return Decision(
            requestRefinement = request,
            forced = forced,
            recovery = recovery,
            score = score,
            viewpointChallenge = evidence.viewpointChallenge,
            ambiguity = evidence.leftRightAmbiguity,
            occlusion = evidence.occlusionRisk,
            temporalContinuity = continuity,
            confidenceLoss = confidenceLoss,
            expectedGain = expectedGain,
            computeCost = computeCost,
            efficiencyRatio = gainCostRatio,
            reason = reason,
            retainedHistoryFrames = historySize,
        )
    }

    @Synchronized
    fun reset() {
        history.fill(null)
        historySize = 0
        historyCursor = 0
        lastRefinementFrame = Long.MIN_VALUE
        lastTimestampNanos = Long.MIN_VALUE
    }

    @Synchronized
    fun snapshot(): List<FrameEvidence> {
        if (historySize == 0) return emptyList()
        val oldest =
            (historyCursor - historySize + config.historyCapacity) %
                config.historyCapacity
        return buildList(historySize) {
            repeat(historySize) { offset ->
                val index =
                    (oldest + offset) %
                        config.historyCapacity
                history[index]?.let(::add)
            }
        }
    }

    private fun weightedScore(
        evidence: FrameEvidence,
        continuity: Double,
        confidenceLoss: Double,
    ): Double {
        val total =
            config.temporalContinuityWeight +
                config.viewpointDifficultyWeight +
                config.ambiguityWeight +
                config.occlusionWeight +
                config.confidenceLossWeight

        val raw =
            config.temporalContinuityWeight * (1.0 - continuity) +
                config.viewpointDifficultyWeight *
                    evidence.viewpointChallenge +
                config.ambiguityWeight *
                    evidence.leftRightAmbiguity +
                config.occlusionWeight *
                    evidence.occlusionRisk +
                config.confidenceLossWeight *
                    confidenceLoss +
                0.18 *
                    min(
                        evidence.viewpointChallenge,
                        evidence.leftRightAmbiguity,
                    )

        return (raw / total).coerceIn(0.0, 1.0)
    }

    private fun temporalContinuity(
        evidence: FrameEvidence,
    ): Double {
        val previous = latest()
            ?: return evidence.temporalStability

        val dtSeconds =
            ((evidence.timestampNanos - previous.timestampNanos)
                .coerceAtLeast(1L) / 1e9)
                .coerceIn(1e-4, 1.0)

        val confidenceDelta =
            abs(evidence.confidence - previous.confidence)

        val motionDelta =
            abs(evidence.motionEnergy - previous.motionEnergy)

        val temporalDelta =
            abs(evidence.temporalStability - previous.temporalStability)

        val instantContinuity = (
            1.0 -
                confidenceDelta * 0.44 -
                motionDelta * 0.22 -
                temporalDelta * 0.20 -
                min(1.0, dtSeconds * 4.0) * 0.14
            ).coerceIn(0.0, 1.0)

        val historyContinuity = recentAverage { frame ->
            (
                1.0 -
                    abs(evidence.confidence - frame.confidence) * 0.36 -
                    abs(evidence.viewpointChallenge - frame.viewpointChallenge) * 0.24 -
                    abs(evidence.occlusionRisk - frame.occlusionRisk) * 0.22 -
                    abs(evidence.leftRightAmbiguity - frame.leftRightAmbiguity) * 0.18
                ).coerceIn(0.0, 1.0)
        }

        return (
            instantContinuity * 0.72 +
                historyContinuity * 0.28
            ).coerceIn(0.0, 1.0)
    }

    private fun confidenceLoss(
        evidence: FrameEvidence,
    ): Double {
        val recent = recentAverage { it.confidence }
        return if (historySize == 0) {
            1.0 - evidence.confidence
        } else {
            (recent - evidence.confidence)
                .coerceIn(-1.0, 1.0)
                .let { delta -> max(0.0, delta) }
                .coerceIn(0.0, 1.0)
        }
    }

    private fun append(evidence: FrameEvidence) {
        history[historyCursor] = evidence
        historyCursor =
            (historyCursor + 1) %
                config.historyCapacity
        historySize =
            min(config.historyCapacity, historySize + 1)
    }

    private fun latest(): FrameEvidence? {
        if (historySize == 0) return null
        val index =
            (historyCursor - 1 + config.historyCapacity) %
                config.historyCapacity
        return history[index]
    }

    private fun recentAverage(
        transform: (FrameEvidence) -> Double,
    ): Double {
        if (historySize == 0) return 0.0
        var sum = 0.0
        var count = 0
        val limit = min(historySize, 8)
        repeat(limit) { offset ->
            val index =
                (historyCursor - 1 - offset + config.historyCapacity * 2) %
                    config.historyCapacity
            history[index]?.let {
                sum += transform(it)
                count += 1
            }
        }
        return if (count == 0) 0.0 else sum / count.toDouble()
    }
}
