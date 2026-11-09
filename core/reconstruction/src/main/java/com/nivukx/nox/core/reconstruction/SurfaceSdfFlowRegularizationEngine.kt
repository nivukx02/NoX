package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max

/**
 * Surface-flow regularizer inspired by SDF-flow formulations for dynamic 4D reconstruction.
 *
 * The normal component of a Gaussian's motion is compared with an externally estimated signed
 * distance change. High-residual constraints are downweighted instead of forcing the surface to
 * follow a bad flow estimate.
 */
class SurfaceSdfFlowRegularizationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val sigmaMeters: Double = 0.015,
        val minimumConfidence: Double = 0.20,
        val maximumFlowMeters: Double = 0.20,
        val blendWeight: Double = 0.70,
    ) {
        init {
            require(sigmaMeters > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(maximumFlowMeters > 0.0)
            require(blendWeight in 0.0..1.0)
        }
    }

    data class Constraint(
        val id: Long,
        val positionMeters: Vec3,
        val normal: Vec3,
        val flowDeltaMeters: Vec3,
        val signedDistanceDeltaMeters: Double,
        val uncertaintyMeters: Double,
        val confidence: Double,
    ) {
        init {
            require(id >= 0L)
            require(finite(positionMeters))
            require(finite(normal) && normal.norm() > 1e-9)
            require(finite(flowDeltaMeters))
            require(signedDistanceDeltaMeters.isFinite())
            require(uncertaintyMeters.isFinite() && uncertaintyMeters >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class Result(
        val id: Long,
        val originalFlowMeters: Vec3,
        val regularizedFlowMeters: Vec3,
        val normalResidualMeters: Double,
        val constraintWeight: Double,
        val accepted: Boolean,
        val warning: String?,
    )

    fun regularize(constraints: List<Constraint>): List<Result> =
        constraints.map { constraint ->
            val normal = constraint.normal.normalized()
            val boundedFlow = constraint.flowDeltaMeters.clampNorm(config.maximumFlowMeters)
            val normalComponent = boundedFlow.dot(normal)
            val residual = normalComponent - constraint.signedDistanceDeltaMeters
            val sigma = max(
                config.sigmaMeters,
                constraint.uncertaintyMeters,
            )
            val gaussianWeight = exp(-0.5 * (residual / sigma) * (residual / sigma))
            val weight = (
                constraint.confidence.coerceIn(0.0,1.0) *
                    gaussianWeight
                ).coerceIn(0.0,1.0)
            val targetNormalDelta = constraint.signedDistanceDeltaMeters
                .coerceIn(-config.maximumFlowMeters, config.maximumFlowMeters)
            val correctedNormal = normalComponent +
                (targetNormalDelta - normalComponent) *
                weight *
                config.blendWeight
            val tangential = boundedFlow - normal * normalComponent
            val regularized = (
                tangential + normal * correctedNormal
            ).clampNorm(config.maximumFlowMeters)
            Result(
                id = constraint.id,
                originalFlowMeters = boundedFlow,
                regularizedFlowMeters = regularized,
                normalResidualMeters = kotlin.math.abs(residual),
                constraintWeight = weight,
                accepted = constraint.confidence >= config.minimumConfidence && weight >= config.minimumConfidence,
                warning = if (weight < config.minimumConfidence) "sdf-flow-constraint-downweighted" else null,
            )
        }

    private fun finite(v: Vec3): Boolean =
        v.x.isFinite() && v.y.isFinite() && v.z.isFinite()
}

private fun Vec3.clampNorm(maxNorm: Double): Vec3 {
    val n = norm()
    return if (n <= maxNorm || n < 1e-12) this else this * (maxNorm / n)
}
