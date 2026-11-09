package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionDiagnostics
import com.nivukx.nox.core.model.SkeletonFrame
import kotlin.math.abs
import kotlin.math.exp

/**
 * Runtime reconstruction quality gate.
 *
 * A 3D result is not considered trustworthy just because triangulation returned a point. The gate
 * combines image-space reprojection, calibrated multi-view geometry, clock skew, joint uncertainty,
 * confidence and temporal consistency. It returns a deterministic score and explicit reasons so the
 * UI/session log can distinguish "no geometry" from "geometry rejected".
 */
class ReconstructionQualityAnalyzer(
    private val minimumJoints: Int = 5,
    private val minimumCameras: Int = 2,
    private val maximumMeanReprojectionErrorPx: Double = 6.0,
    private val maximumMedianReprojectionErrorPx: Double = 5.0,
    private val maximumClockSkewNanos: Long = 25_000_000L,
    private val maximumMeanUncertaintyMeters: Double = 0.25,
    private val maximumJointUncertaintyMeters: Double = 0.75,
    private val minimumConfidence: Double = 0.30,
) {

    data class Evaluation(
        val diagnostics: ReconstructionDiagnostics,
        val score: Double,
        val accepted: Boolean,
        val reasons: List<String>,
    )

    fun analyze(
        frame: SkeletonFrame,
        camerasUsed: Int,
        clockSkewNanos: Long,
        droppedObservations: Int = 0,
    ): ReconstructionDiagnostics {
        val errors = frame.joints
            .map { it.reprojectionErrorPx }
            .filter { it.isFinite() }
            .sorted()

        val mean = if (errors.isEmpty()) {
            Double.POSITIVE_INFINITY
        } else {
            errors.average()
        }

        val median = if (errors.isEmpty()) {
            Double.POSITIVE_INFINITY
        } else {
            errors[errors.size / 2]
        }

        return ReconstructionDiagnostics(
            jointsTriangulated = frame.joints.size,
            meanReprojectionErrorPx = mean,
            medianReprojectionErrorPx = median,
            clockSkewNanos = abs(clockSkewNanos),
            droppedObservations = droppedObservations.coerceAtLeast(0),
            camerasUsed = camerasUsed.coerceAtLeast(0),
        )
    }

    fun evaluate(
        frame: SkeletonFrame,
        camerasUsed: Int,
        clockSkewNanos: Long,
        droppedObservations: Int = 0,
    ): Evaluation {
        val diagnostics = analyze(
            frame = frame,
            camerasUsed = camerasUsed,
            clockSkewNanos = clockSkewNanos,
            droppedObservations = droppedObservations,
        )

        val uncertainties = frame.joints.map {
            it.positionUncertaintyMeters
        }.filter { it.isFinite() && it >= 0.0 }

        val meanUncertainty = if (uncertainties.isEmpty()) {
            Double.POSITIVE_INFINITY
        } else {
            uncertainties.average()
        }

        val maxUncertainty = uncertainties.maxOrNull()
            ?: Double.POSITIVE_INFINITY

        val meanJointConfidence = if (frame.joints.isEmpty()) {
            0.0
        } else {
            frame.joints.map { it.confidence }.average()
        }

        val errors = frame.joints
            .map { it.reprojectionErrorPx }
            .filter { it.isFinite() && it >= 0.0 }
            .sorted()

        val inlierThresholdPx =
            maximumMeanReprojectionErrorPx * 1.50
        val inlierRatio = if (errors.isEmpty()) {
            0.0
        } else {
            errors.count { it <= inlierThresholdPx }.toDouble() /
                errors.size.toDouble()
        }

        val confidenceWeightedError = if (frame.joints.isEmpty()) {
            Double.POSITIVE_INFINITY
        } else {
            val weighted = frame.joints.mapNotNull { joint ->
                val error = joint.reprojectionErrorPx
                val confidence = joint.confidence
                if (!error.isFinite() || !confidence.isFinite()) {
                    null
                } else {
                    error.coerceAtLeast(0.0) *
                        confidence.coerceIn(0.05, 1.0)
                }
            }
            val weights = frame.joints.mapNotNull { joint ->
                if (!joint.reprojectionErrorPx.isFinite() ||
                    !joint.confidence.isFinite()
                ) {
                    null
                } else {
                    joint.confidence.coerceIn(0.05, 1.0)
                }
            }
            if (weighted.isEmpty() || weights.sum() <= 1e-9) {
                Double.POSITIVE_INFINITY
            } else {
                weighted.sum() / weights.sum()
            }
        }

        val errorScore = exp(
            -(
                diagnostics.meanReprojectionErrorPx /
                    maximumMeanReprojectionErrorPx.coerceAtLeast(0.1)
            )
        ).coerceIn(0.0, 1.0)

        val weightedErrorScore = exp(
            -(
                confidenceWeightedError /
                    maximumMeanReprojectionErrorPx.coerceAtLeast(0.1)
            )
        ).coerceIn(0.0, 1.0)

        val inlierScore = inlierRatio.coerceIn(0.0, 1.0)

        val medianScore = exp(
            -(
                diagnostics.medianReprojectionErrorPx /
                    maximumMedianReprojectionErrorPx.coerceAtLeast(0.1)
                )
        ).coerceIn(0.0, 1.0)

        val uncertaintyScore = exp(
            -(
                meanUncertainty /
                    maximumMeanUncertaintyMeters.coerceAtLeast(1e-3)
                )
        ).coerceIn(0.0, 1.0)

        val confidenceScore = meanJointConfidence
            .coerceIn(0.0, 1.0)

        val coverageScore = (
            diagnostics.jointsTriangulated /
                (minimumJoints * 2.0).coerceAtLeast(1.0)
            ).coerceIn(0.0, 1.0)

        val cameraScore = (
            diagnostics.camerasUsed /
                4.0
            ).coerceIn(0.0, 1.0)

        val skewScore = (
            1.0 -
                diagnostics.clockSkewNanos.toDouble() /
                    maximumClockSkewNanos.coerceAtLeast(1L)
            ).coerceIn(0.0, 1.0)

        val temporalScore = frame.globalConfidence
            .coerceIn(0.0, 1.0)

        val score = (
            0.16 * errorScore +
                0.10 * weightedErrorScore +
                0.09 * medianScore +
                0.18 * uncertaintyScore +
                0.16 * confidenceScore +
                0.09 * coverageScore +
                0.06 * inlierScore +
                0.06 * cameraScore +
                0.05 * skewScore +
                0.05 * temporalScore
            ).coerceIn(0.0, 1.0)

        val reasons = mutableListOf<String>()

        if (diagnostics.jointsTriangulated < minimumJoints) {
            reasons += "insufficient-joint-coverage"
        }
        if (diagnostics.camerasUsed < minimumCameras) {
            reasons += "insufficient-camera-count"
        }
        if (
            !diagnostics.meanReprojectionErrorPx.isFinite() ||
            diagnostics.meanReprojectionErrorPx >
                maximumMeanReprojectionErrorPx
        ) {
            reasons += "mean-reprojection-error"
        }
        if (
            !diagnostics.medianReprojectionErrorPx.isFinite() ||
            diagnostics.medianReprojectionErrorPx >
                maximumMedianReprojectionErrorPx
        ) {
            reasons += "median-reprojection-error"
        }
        if (
            diagnostics.clockSkewNanos >
                maximumClockSkewNanos
        ) {
            reasons += "clock-skew"
        }
        if (
            meanUncertainty > maximumMeanUncertaintyMeters
        ) {
            reasons += "mean-depth-uncertainty"
        }
        if (
            maxUncertainty > maximumJointUncertaintyMeters
        ) {
            reasons += "joint-depth-uncertainty"
        }
        if (
            meanJointConfidence < minimumConfidence
        ) {
            reasons += "low-joint-confidence"
        }
        if (inlierRatio < 0.65) {
            reasons += "low-reprojection-inlier-ratio"
        }
        if (frame.globalConfidence < minimumConfidence) {
            reasons += "low-global-confidence"
        }
        if (droppedObservations > diagnostics.jointsTriangulated) {
            reasons += "excess-dropped-observations"
        }

        val accepted = reasons.isEmpty() &&
            score >= 0.55

        if (!accepted && reasons.isEmpty()) {
            reasons += "quality-score-below-threshold"
        }

        return Evaluation(
            diagnostics = diagnostics,
            score = score,
            accepted = accepted,
            reasons = reasons,
        )
    }

    fun accepts(
        diagnostics: ReconstructionDiagnostics,
        maxMeanReprojectionErrorPx: Double =
            maximumMeanReprojectionErrorPx,
        maxClockSkewNanos: Long =
            maximumClockSkewNanos,
        minimumJoints: Int = this.minimumJoints,
    ): Boolean =
        diagnostics.jointsTriangulated >= minimumJoints &&
            diagnostics.meanReprojectionErrorPx <=
                maxMeanReprojectionErrorPx &&
            diagnostics.clockSkewNanos <=
                maxClockSkewNanos &&
            diagnostics.camerasUsed >= minimumCameras
}
