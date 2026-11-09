package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Motion-aware reliability weighting for human observations used as bundle-adjustment evidence.
 *
 * It is intentionally a soft constraint layer around the existing calibrated BA solver. Dynamic,
 * uncertain, or camera-compensation-inconsistent motion is downweighted rather than turned into
 * artificial geometry.
 */
class HumanAwareBundleAdjustmentEvidenceEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val minimumEvidenceConfidence: Double = 0.25,
        val minimumMotionReliability: Double = 0.20,
        val uncertaintyFloorMeters: Double = 0.005,
        val maximumResidualMeters: Double = 0.75,
        val dynamicProbabilityPenalty: Double = 0.55,
        val cameraMotionCompensationMinimum: Double = 0.35,
        val temporalSpanPenaltyPerFrame: Double = 0.012,
    ) {
        init {
            require(minimumEvidenceConfidence in 0.0..1.0)
            require(minimumMotionReliability in 0.0..1.0)
            require(uncertaintyFloorMeters > 0.0)
            require(maximumResidualMeters > uncertaintyFloorMeters)
            require(dynamicProbabilityPenalty in 0.0..1.0)
            require(cameraMotionCompensationMinimum in 0.0..1.0)
            require(temporalSpanPenaltyPerFrame >= 0.0)
        }
    }

    data class Evidence(
        val cameraId: String,
        val pointId: Int,
        val predictedMotionMeters: Vec3,
        val observedMotionMeters: Vec3,
        val motionUncertaintyMeters: Double,
        val confidence: Double,
        val dynamicProbability: Double,
        val cameraMotionCompensationConfidence: Double,
        val landmarkStability: Double = 0.75,
        val temporalSpanFrames: Int = 1,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(pointId >= 0)
            require(finite(predictedMotionMeters))
            require(finite(observedMotionMeters))
            require(motionUncertaintyMeters.isFinite() && motionUncertaintyMeters >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(dynamicProbability.isFinite() && dynamicProbability in 0.0..1.0)
            require(
                cameraMotionCompensationConfidence.isFinite() &&
                    cameraMotionCompensationConfidence in 0.0..1.0,
            )
            require(landmarkStability.isFinite() && landmarkStability in 0.0..1.0)
            require(temporalSpanFrames >= 1)
        }
    }

    data class WeightedObservation(
        val cameraId: String,
        val pointId: Int,
        val weight: Double,
        val motionReliability: Double,
        val residualMeters: Double,
        val reason: String,
    )

    data class Result(
        val weights: List<WeightedObservation>,
        val acceptedCount: Int,
        val rejectedCount: Int,
        val meanReliability: Double,
        val warnings: List<String>,
    )

    fun evaluate(
        observations: List<BundleAdjustmentEngine.Observation>,
        evidence: List<Evidence>,
    ): Result {
        val byObservation = evidence
            .groupBy { it.cameraId to it.pointId }
            .mapValues { (_, values) ->
                values.maxByOrNull { reliability(it) }!!
            }

        val weights = observations.map { observation ->
            val ev = byObservation[observation.cameraId to observation.pointId]
            if (ev == null) {
                WeightedObservation(
                    cameraId = observation.cameraId,
                    pointId = observation.pointId,
                    weight = (observation.confidence * 0.50).coerceIn(0.0, 1.0),
                    motionReliability = 0.50,
                    residualMeters = 0.0,
                    reason = "no-human-motion-evidence",
                )
            } else {
                val reliability = reliability(ev)
                WeightedObservation(
                    cameraId = observation.cameraId,
                    pointId = observation.pointId,
                    weight = (observation.confidence * reliability).coerceIn(0.0, 1.0),
                    motionReliability = reliability,
                    residualMeters = ev.predictedMotionMeters.distanceTo(ev.observedMotionMeters),
                    reason =
                        if (reliability >= config.minimumMotionReliability) {
                            "motion-consistent"
                        } else {
                            "motion-evidence-rejected"
                        },
                )
            }
        }

        val accepted = weights.count { it.weight >= config.minimumEvidenceConfidence }
        return Result(
            weights = weights,
            acceptedCount = accepted,
            rejectedCount = weights.size - accepted,
            meanReliability = weights.map { it.motionReliability }.averageOrZero(),
            warnings = buildList {
                if (accepted == 0 && weights.isNotEmpty()) {
                    add("all-human-motion-bundle-evidence-rejected")
                }
                if (weights.any { it.reason == "motion-evidence-rejected" }) {
                    add("human-motion-reliability-reduced-bundle-weight")
                }
                if (weights.any { it.reason == "no-human-motion-evidence" }) {
                    add("partial-human-motion-evidence-coverage")
                }
            }.distinct(),
        )
    }

    private fun reliability(evidence: Evidence): Double {
        val residual = evidence.predictedMotionMeters.distanceTo(evidence.observedMotionMeters)
        if (!residual.isFinite() || residual > config.maximumResidualMeters) return 0.0

        val sigma = max(config.uncertaintyFloorMeters, evidence.motionUncertaintyMeters)
        val consistency = exp(-0.5 * (residual / sigma) * (residual / sigma))
        val dynamicPenalty =
            1.0 - config.dynamicProbabilityPenalty * evidence.dynamicProbability
        val cameraCompensation =
            if (evidence.cameraMotionCompensationConfidence >= config.cameraMotionCompensationMinimum) {
                evidence.cameraMotionCompensationConfidence
            } else {
                evidence.cameraMotionCompensationConfidence * 0.60
            }
        val temporalPenalty =
            (1.0 - config.temporalSpanPenaltyPerFrame *
                (evidence.temporalSpanFrames - 1).coerceAtLeast(0))
                .coerceIn(0.45, 1.0)
        return (
            evidence.confidence *
                evidence.landmarkStability *
                consistency *
                dynamicPenalty.coerceIn(0.0, 1.0) *
                cameraCompensation.coerceIn(0.0, 1.0) *
                temporalPenalty
        ).coerceIn(0.0, 1.0)
    }

    private fun finite(value: Vec3): Boolean =
        value.x.isFinite() && value.y.isFinite() && value.z.isFinite()
}

private fun Iterable<Double>.averageOrZero(): Double =
    if (none()) 0.0 else average()

private fun Vec3.distanceTo(other: Vec3): Double =
    sqrt(
        (x - other.x) * (x - other.x) +
            (y - other.y) * (y - other.y) +
            (z - other.z) * (z - other.z),
    )
