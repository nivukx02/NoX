package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.ConfidenceBreakdown
import com.nivukx.nox.core.model.MotionQualityGate
import com.nivukx.nox.core.model.ReconstructionConfidenceBand
import com.nivukx.nox.core.model.SubjectType
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Unified evidence-based quality engine.
 *
 * NoX must not publish a pretty skeleton simply because a detector returned coordinates. This
 * engine makes acceptance depend on independent evidence: detector confidence, geometry, temporal
 * continuity, topology stability, depth/scale evidence, occlusion, and reconstruction residuals.
 *
 * The output is deliberately diagnostic. A downstream recorder may choose to preserve degraded
 * frames for forensic editing, while a live avatar driver can reject them.
 */
class MotionQualityEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumCaptureConfidence: Double = 0.34,
        val minimumLiveConfidence: Double = 0.50,
        val maximumAcceptedUncertaintyMeters: Double = 0.35,
        val maximumLiveUncertaintyMeters: Double = 0.25,
        val maximumAcceptedReprojectionPx: Double = 8.0,
        val maximumLiveReprojectionPx: Double = 5.0,
        val minimumTemporalStability: Double = 0.35,
        val minimumLiveTemporalStability: Double = 0.56,
        val occlusionPenaltyScale: Double = 0.18,
        val topologyPenaltyScale: Double = 0.15,
        val predictionPenaltyScale: Double = 0.08,
        val residualScalePx: Double = 6.0,
    )

    data class Input(
        val timestampNanos: Long,
        val subjectType: SubjectType,
        val detectorConfidence: Double,
        val geometryConfidence: Double,
        val temporalStability: Double,
        val topologyStability: Double,
        val occlusionRisk: Double,
        val depthConfidence: Double,
        val scaleConfidence: Double,
        val meanUncertaintyMeters: Double,
        val reprojectionErrorPx: Double,
        val predictedFraction: Double,
        val visibleNodes: Int,
        val totalNodes: Int,
        val cameraMotionRisk: Double = 0.0,
        val modelFresh: Boolean = true,
    )

    data class Result(
        val acceptedForMotionRecording: Boolean,
        val acceptedForLiveAvatar: Boolean,
        val confidence: Double,
        val breakdown: ConfidenceBreakdown,
        val band: ReconstructionConfidenceBand,
        val gate: MotionQualityGate,
        val warnings: List<String>,
        val rejectionReasons: List<String>,
    ) {
        val usable: Boolean
            get() = band != ReconstructionConfidenceBand.UNTRUSTED
    }

    fun evaluate(input: Input): Result {
        val detector = sanitize(input.detectorConfidence)
        val geometry = sanitize(input.geometryConfidence)
        val temporal = sanitize(input.temporalStability)
        val topology = sanitize(input.topologyStability)
        val occlusion = 1.0 - sanitize(input.occlusionRisk)
        val depth = sanitize(input.depthConfidence)
        val scale = sanitize(input.scaleConfidence)

        val visibility =
            if (input.totalNodes <= 0) 0.0
            else (
                input.visibleNodes.toDouble() /
                    input.totalNodes.toDouble()
                ).coerceIn(0.0, 1.0)

        val residualConfidence =
            exp(
                -input.reprojectionErrorPx.coerceAtLeast(0.0) /
                    config.residualScalePx.coerceAtLeast(0.1),
            ).coerceIn(0.0, 1.0)

        val predictionPenalty =
            (
                1.0 -
                    config.predictionPenaltyScale *
                    sanitize(input.predictedFraction)
                ).coerceIn(0.0, 1.0)

        val cameraPenalty =
            (
                1.0 -
                    0.10 * sanitize(input.cameraMotionRisk)
                ).coerceIn(0.0, 1.0)

        val raw = ConfidenceBreakdown.weighted(
            detector = detector,
            geometry = geometry * visibility * cameraPenalty,
            temporal = temporal * predictionPenalty,
            topology = topology,
            occlusion = occlusion,
            depth = depth,
            scale = scale,
        )

        val confidence = (
            raw.final *
                (0.58 + 0.42 * residualConfidence)
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (input.occlusionRisk > 0.45) add("occlusion risk elevated")
            if (input.predictedFraction > 0.30) add("many nodes are prediction-only")
            if (input.cameraMotionRisk > 0.50) add("camera motion compensation active")
            if (!input.modelFresh) add("one or more perception stages are using cached evidence")
            if (input.scaleConfidence < 0.40) add("metric scale is weak or estimated")
            if (input.meanUncertaintyMeters > config.maximumLiveUncertaintyMeters) {
                add("uncertainty above live-avatar threshold")
            }
            if (input.reprojectionErrorPx > config.maximumLiveReprojectionPx) {
                add("reprojection residual above live threshold")
            }
        }

        val gate = MotionQualityGate.evaluate(
            confidence = confidence,
            uncertaintyMeters = input.meanUncertaintyMeters,
            reprojectionErrorPx = input.reprojectionErrorPx,
            temporalStability = temporal,
        )

        val rejectionReasons = gate.reasons.toMutableList()

        if (input.visibleNodes <= 0) {
            rejectionReasons += "no visible nodes"
        }

        if (input.totalNodes > 0 && visibility < 0.25) {
            rejectionReasons += "visibility coverage below 25%"
        }

        if (input.subjectType == SubjectType.UNKNOWN && confidence < 0.50) {
            rejectionReasons += "unknown topology without enough evidence"
        }

        if (input.meanUncertaintyMeters > config.maximumAcceptedUncertaintyMeters) {
            rejectionReasons += "uncertainty exceeds motion-recording gate"
        }

        if (input.reprojectionErrorPx > config.maximumAcceptedReprojectionPx) {
            rejectionReasons += "reprojection exceeds motion-recording gate"
        }

        val acceptedForMotion =
            rejectionReasons.isEmpty() &&
                confidence >= config.minimumCaptureConfidence

        val acceptedForLive =
            acceptedForMotion &&
                confidence >= config.minimumLiveConfidence &&
                input.meanUncertaintyMeters <= config.maximumLiveUncertaintyMeters &&
                input.reprojectionErrorPx <= config.maximumLiveReprojectionPx &&
                temporal >= config.minimumLiveTemporalStability

        return Result(
            acceptedForMotionRecording = acceptedForMotion,
            acceptedForLiveAvatar = acceptedForLive,
            confidence = confidence,
            breakdown = raw.copy(final = confidence),
            band = gate.confidenceBand,
            gate = gate,
            warnings = warnings.distinct(),
            rejectionReasons = rejectionReasons.distinct(),
        )
    }

    fun combine(
        primary: Result,
        secondary: Result,
        primaryWeight: Double = 0.70,
    ): Result {
        val p = primaryWeight.coerceIn(0.0, 1.0)
        val s = 1.0 - p
        val confidence = (
            primary.confidence * p +
                secondary.confidence * s
            ).coerceIn(0.0, 1.0)

        val live =
            primary.acceptedForLiveAvatar &&
                secondary.acceptedForLiveAvatar &&
                confidence >= config.minimumLiveConfidence

        val motion =
            primary.acceptedForMotionRecording &&
                secondary.acceptedForMotionRecording &&
                confidence >= config.minimumCaptureConfidence

        val warnings =
            (primary.warnings + secondary.warnings).distinct()

        val reasons =
            (primary.rejectionReasons + secondary.rejectionReasons).distinct()

        val band =
            when {
                confidence >= 0.90 -> ReconstructionConfidenceBand.EXCELLENT
                confidence >= 0.75 -> ReconstructionConfidenceBand.GOOD
                confidence >= 0.55 -> ReconstructionConfidenceBand.USABLE
                confidence >= 0.35 -> ReconstructionConfidenceBand.DEGRADED
                else -> ReconstructionConfidenceBand.UNTRUSTED
            }

        return primary.copy(
            acceptedForMotionRecording = motion,
            acceptedForLiveAvatar = live,
            confidence = confidence,
            band = band,
            warnings = warnings,
            rejectionReasons = reasons,
        )
    }

    private fun sanitize(value: Double): Double =
        if (!value.isFinite()) 0.0 else value.coerceIn(0.0, 1.0)
}
