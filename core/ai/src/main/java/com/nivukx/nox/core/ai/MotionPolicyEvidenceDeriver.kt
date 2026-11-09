package com.nivukx.nox.core.ai

import kotlin.math.abs

/**
 * Derives conservative view and identity ambiguity signals from measured perception evidence.
 *
 * The result is intentionally independent of a particular detector. It converts already measured
 * confidence, occlusion, semantic coverage, temporal stability, and camera-motion signals into the
 * normalized evidence consumed by the motion orchestration policy.
 */
object MotionPolicyEvidenceDeriver {

    data class Result(
        val viewpointChallenge: Double,
        val leftRightAmbiguity: Double,
    ) {
        init {
            require(viewpointChallenge in 0.0..1.0)
            require(leftRightAmbiguity in 0.0..1.0)
        }
    }

    fun derive(
        assessment: AdvancedMotionFusionEngine.Assessment,
        motionQuality: MotionCaptureEngine.Quality,
    ): Result {
        val occlusionRisk = assessment.occlusionRisk.coerceIn(0.0, 1.0)
        val semanticCoverage = assessment.semanticCoverage.coerceIn(0.0, 1.0)
        val cameraMotionRisk =
            assessment.semanticMotion.cameraMotionRisk.coerceIn(0.0, 1.0)
        val temporalStability =
            motionQuality.temporalStability.coerceIn(0.0, 1.0)
        val attentionConfidence =
            assessment.attentionConfidence.coerceIn(0.0, 1.0)
        val meanConfidence =
            motionQuality.meanConfidence.coerceIn(0.0, 1.0)
        val occlusionRatio =
            motionQuality.occlusionRatio.coerceIn(0.0, 1.0)

        val viewpointChallenge = (
            occlusionRisk * 0.46 +
                (1.0 - semanticCoverage) * 0.18 +
                cameraMotionRisk * 0.18 +
                (1.0 - temporalStability) * 0.18
            ).coerceIn(0.0, 1.0)

        val residualConfidenceDisagreement =
            abs(
                meanConfidence -
                    attentionConfidence,
            ).coerceIn(0.0, 1.0)

        val leftRightAmbiguity = (
            occlusionRisk * 0.46 +
                occlusionRatio * 0.20 +
                (1.0 - attentionConfidence) * 0.15 +
                (1.0 - meanConfidence) * 0.10 +
                residualConfidenceDisagreement * 0.09
            ).coerceIn(0.0, 1.0)

        return Result(
            viewpointChallenge = viewpointChallenge,
            leftRightAmbiguity = leftRightAmbiguity,
        )
    }
}
