package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DynamicTopology
import com.nivukx.nox.core.model.MonocularReconstructionFrame
import com.nivukx.nox.core.model.MonocularReconstructionJoint
import com.nivukx.nox.core.model.ScaleSource
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Refines monocular metric scale from reconstructed articulated geometry.
 *
 * Relative monocular depth is fundamentally scale-ambiguous. This engine uses only topology edges
 * that already carry explicit preferred metric lengths, estimates a robust scale ratio, rejects
 * outliers with a median/MAD gate, and refuses to override trusted calibrated metric depth.
 */
class AnthropometricScaleRefinementEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumEdges: Int = 3,
        val minimumEdgeConfidence: Double = 0.24,
        val minimumObservedLengthMeters: Double = 0.015,
        val minimumScaleRatio: Double = 0.72,
        val maximumScaleRatio: Double = 1.38,
        val smoothingAlpha: Double = 0.28,
        val robustSigma: Double = 3.5,
        val maximumRelativeScaleJump: Double = 0.12,
        val trustedMetricConfidence: Double = 0.84,
    )

    data class EdgeEvidence(
        val edgeId: String,
        val observedLengthMeters: Double,
        val expectedLengthMeters: Double,
        val ratio: Double,
        val confidence: Double,
        val inlier: Boolean,
    )

    data class Result(
        val frame: MonocularReconstructionFrame,
        val appliedScale: Double,
        val estimatedScaleRatio: Double,
        val confidence: Double,
        val source: ScaleSource,
        val refined: Boolean,
        val evidence: List<EdgeEvidence>,
        val diagnostics: List<String>,
    )

    fun refine(
        frame: MonocularReconstructionFrame,
        topology: DynamicTopology,
    ): Result {
        if (
            frame.scaleSource == ScaleSource.CALIBRATED_METRIC &&
            frame.scaleConfidence >= config.trustedMetricConfidence
        ) {
            return Result(
                frame = frame,
                appliedScale = 1.0,
                estimatedScaleRatio = 1.0,
                confidence = frame.scaleConfidence,
                source = frame.scaleSource,
                refined = false,
                evidence = emptyList(),
                diagnostics = listOf(
                    "trusted calibrated metric scale preserved",
                ),
            )
        }

        if (
            frame.scaleSource == ScaleSource.DEPTH_SENSOR &&
            frame.scaleConfidence >= config.trustedMetricConfidence
        ) {
            return Result(
                frame = frame,
                appliedScale = 1.0,
                estimatedScaleRatio = 1.0,
                confidence = frame.scaleConfidence,
                source = frame.scaleSource,
                refined = false,
                evidence = emptyList(),
                diagnostics = listOf(
                    "trusted depth-sensor metric scale preserved",
                ),
            )
        }

        val joints = frame.joints.associateBy { it.landmarkId }
        val rawEvidence = topology.edges.mapNotNull { edge ->
            val a = joints[edge.a] ?: return@mapNotNull null
            val b = joints[edge.b] ?: return@mapNotNull null

            val observed = a.positionCameraMeters.distanceTo(
                b.positionCameraMeters,
            )

            if (
                !observed.isFinite() ||
                observed < config.minimumObservedLengthMeters ||
                edge.preferredLengthMeters <= 0.0
            ) {
                return@mapNotNull null
            }

            val confidence = min(
                a.confidence,
                b.confidence,
            ).coerceIn(0.0, 1.0) *
                edge.stiffness.coerceIn(0.0, 1.0)

            if (confidence < config.minimumEdgeConfidence) {
                return@mapNotNull null
            }

            val ratio = (
                edge.preferredLengthMeters /
                    observed
            ).coerceIn(
                config.minimumScaleRatio,
                config.maximumScaleRatio,
            )

            EdgeEvidence(
                edgeId = edge.a.toString() + ":" + edge.b,
                observedLengthMeters = observed,
                expectedLengthMeters = edge.preferredLengthMeters,
                ratio = ratio,
                confidence = confidence,
                inlier = true,
            )
        }

        if (rawEvidence.size < config.minimumEdges) {
            return Result(
                frame = frame,
                appliedScale = 1.0,
                estimatedScaleRatio = 1.0,
                confidence = frame.scaleConfidence * 0.82,
                source = frame.scaleSource,
                refined = false,
                evidence = rawEvidence,
                diagnostics = listOf(
                    "insufficient metric topology edges for scale refinement",
                    "edges=" + rawEvidence.size,
                ),
            )
        }

        val ratios = rawEvidence.map { it.ratio }
        val median = median(ratios)
        val mad = median(
            ratios.map { abs(it - median) },
        ).coerceAtLeast(1e-5)

        val threshold = config.robustSigma * 1.4826 * mad
        val evidence = rawEvidence.map { sample ->
            sample.copy(
                inlier =
                    abs(sample.ratio - median) <=
                        max(
                            threshold,
                            0.035,
                        ),
            )
        }

        val inliers = evidence.filter { it.inlier }
        if (inliers.size < config.minimumEdges) {
            return Result(
                frame = frame,
                appliedScale = 1.0,
                estimatedScaleRatio = 1.0,
                confidence = frame.scaleConfidence * 0.68,
                source = frame.scaleSource,
                refined = false,
                evidence = evidence,
                diagnostics = listOf(
                    "robust scale gate rejected topology evidence",
                    "inliers=" + inliers.size,
                ),
            )
        }

        val weightedRatio = (
            inliers.sumOf { it.ratio * it.confidence } /
                inliers.sumOf { it.confidence }.coerceAtLeast(1e-6)
            ).coerceIn(
                config.minimumScaleRatio,
                config.maximumScaleRatio,
            )

        val boundedDelta = (
            weightedRatio - 1.0
        ).coerceIn(
            -config.maximumRelativeScaleJump,
            config.maximumRelativeScaleJump,
        )

        val appliedScale =
            1.0 +
                boundedDelta *
                config.smoothingAlpha.coerceIn(0.0, 1.0)

        val refinedJoints = if (
            abs(appliedScale - 1.0) < 1e-4
        ) {
            frame.joints
        } else {
            frame.joints.map { joint ->
                scaleJoint(
                    joint = joint,
                    root = frame.rootPositionMeters,
                    scale = appliedScale,
                )
            }
        }

        val confidence = (
            frame.scaleConfidence * 0.42 +
                inliers.map { it.confidence }.average() * 0.38 +
                inliers.size.toDouble() /
                    evidence.size.toDouble().coerceAtLeast(1.0) * 0.20
            ).coerceIn(0.0, 1.0)

        val refinedFrame =
            if (abs(appliedScale - 1.0) < 1e-4) {
                frame
            } else {
                frame.copy(
                    joints = refinedJoints,
                    rootPositionMeters =
                        frame.rootPositionMeters.lerp(
                            root = frame.rootPositionMeters,
                            targetScale = appliedScale,
                        ),
                    globalScaleMeters =
                        frame.globalScaleMeters * appliedScale,
                    scaleSource = ScaleSource.ANTHROPOMETRIC_REFINED,
                    scaleConfidence = confidence,
                )
            }

        return Result(
            frame = refinedFrame,
            appliedScale = appliedScale,
            estimatedScaleRatio = weightedRatio,
            confidence = confidence,
            source = refinedFrame.scaleSource,
            refined = abs(appliedScale - 1.0) >= 1e-4,
            evidence = evidence,
            diagnostics = listOf(
                "inputScaleSource=" + frame.scaleSource.name,
                "estimatedScaleRatio=" + format(weightedRatio),
                "appliedScale=" + format(appliedScale),
                "inliers=" + inliers.size,
                "evidence=" + evidence.size,
                "confidence=" + format(confidence),
            ),
        )
    }

    private fun scaleJoint(
        joint: MonocularReconstructionJoint,
        root: com.nivukx.nox.core.model.Vec3,
        scale: Double,
    ): MonocularReconstructionJoint {
        val position =
            root +
                (joint.positionCameraMeters - root) * scale

        return joint.copy(
            positionCameraMeters = position,
            velocityCameraMetersPerSecond =
                joint.velocityCameraMetersPerSecond * scale,
            accelerationCameraMetersPerSecond2 =
                joint.accelerationCameraMetersPerSecond2 * scale,
            uncertaintyMeters =
                (
                    joint.uncertaintyMeters * abs(scale)
                ).coerceAtLeast(0.001),
            depthMeters =
                position.z,
        )
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 1.0
        return if (sorted.size % 2 == 0) {
            (
                sorted[sorted.size / 2 - 1] +
                    sorted[sorted.size / 2]
                ) * 0.5
        } else {
            sorted[sorted.size / 2]
        }
    }

    private fun format(value: Double): String =
        "%.5f".format(
            java.util.Locale.US,
            value,
        )

    private fun com.nivukx.nox.core.model.Vec3.lerp(
        root: com.nivukx.nox.core.model.Vec3,
        targetScale: Double,
    ): com.nivukx.nox.core.model.Vec3 =
        root + (this - root) * targetScale
}
