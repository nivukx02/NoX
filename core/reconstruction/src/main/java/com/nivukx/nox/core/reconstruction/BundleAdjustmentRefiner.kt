package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraObservation
import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.absoluteValue
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Robust multi-view bundle-adjustment-style point refiner.
 *
 * This is deliberately joint-local rather than a monolithic global optimizer. Each reconstructed
 * joint is refined against every synchronized camera observation using a damped Gauss-Newton
 * update, Huber robustification, analytic Brown-Conrady projection Jacobians, camera-model
 * distortion, cheirality checks, and uncertainty propagation. That gives the reconstruction path a
 * second geometric pass after triangulation
 * without allocating large matrices for an entire skeleton on a mobile device.
 *
 * The optimizer never treats "smooth" as synonymous with "correct". A candidate that reduces
 * average reprojection error but becomes geometrically ill-conditioned is rejected. Likewise,
 * outliers are down-weighted rather than allowed to drag a valid 3D point across the scene.
 */
class BundleAdjustmentRefiner(
    private val config: Config = Config(),
) {

    data class Config(
        val iterations: Int = 5,
        val damping: Double = 1e-3,
        val dampingGrowth: Double = 3.0,
        val dampingFloor: Double = 1e-6,
        val huberThresholdPx: Double = 3.0,
        val maximumObservationResidualPx: Double = 48.0,
        val minimumDepthMeters: Double = 0.08,
        val maximumDepthMeters: Double = 200.0,
        val minimumAcceptedConfidence: Double = 0.10,
        val minimumConditionScore: Double = 1e-5,
        val maximumStepMeters: Double = 0.35,
        val minimumImprovementPx: Double = 0.002,
        val priorWeight: Double = 0.02,
        val uncertaintyFloorMeters: Double = 0.001,
        val uncertaintyCeilingMeters: Double = 2.5,
    ) {
        init {
            require(iterations in 1..20)
            require(damping > 0.0)
            require(dampingGrowth >= 1.0)
            require(dampingFloor > 0.0)
            require(huberThresholdPx > 0.0)
            require(maximumObservationResidualPx > huberThresholdPx)
            require(minimumDepthMeters > 0.0)
            require(maximumDepthMeters > minimumDepthMeters)
            require(minimumAcceptedConfidence in 0.0..1.0)
            require(minimumConditionScore > 0.0)
            require(maximumStepMeters > 0.0)
            require(minimumImprovementPx >= 0.0)
            require(priorWeight >= 0.0)
            require(uncertaintyFloorMeters > 0.0)
            require(uncertaintyCeilingMeters >= uncertaintyFloorMeters)
        }
    }

    data class Result(
        val joint: ReconstructionJoint,
        val accepted: Boolean,
        val initialErrorPx: Double,
        val finalErrorPx: Double,
        val observationsUsed: Int,
        val inliers: Int,
        val rejectedOutliers: Int,
        val conditionScore: Double,
        val stepMeters: Double,
        val uncertaintyMeters: Double,
        val reason: String,
    )

    private data class Projection(
        val x: Double,
        val y: Double,
        val depth: Double,
        val jacobian: Matrix2x3,
    )

    private data class Residual(
        val ex: Double,
        val ey: Double,
        val weight: Double,
        val projection: Projection,
        val observation: CameraObservation,
    )

    private data class Matrix3x3(
        val m00: Double = 0.0,
        val m01: Double = 0.0,
        val m02: Double = 0.0,
        val m10: Double = 0.0,
        val m11: Double = 0.0,
        val m12: Double = 0.0,
        val m20: Double = 0.0,
        val m21: Double = 0.0,
        val m22: Double = 0.0,
    ) {
        operator fun plus(other: Matrix3x3): Matrix3x3 =
            Matrix3x3(
                m00 + other.m00, m01 + other.m01, m02 + other.m02,
                m10 + other.m10, m11 + other.m11, m12 + other.m12,
                m20 + other.m20, m21 + other.m21, m22 + other.m22,
            )

        operator fun times(scale: Double): Matrix3x3 =
            Matrix3x3(
                m00 * scale, m01 * scale, m02 * scale,
                m10 * scale, m11 * scale, m12 * scale,
                m20 * scale, m21 * scale, m22 * scale,
            )

        fun addDiagonal(value: Double): Matrix3x3 =
            Matrix3x3(
                m00 + value, m01, m02,
                m10, m11 + value, m12,
                m20, m21, m22 + value,
            )
    }

    private data class Matrix2x3(
        val x0: Double,
        val x1: Double,
        val x2: Double,
        val y0: Double,
        val y1: Double,
        val y2: Double,
    )

    private data class Vector3(
        val x: Double,
        val y: Double,
        val z: Double,
    ) {
        operator fun plus(other: Vector3): Vector3 =
            Vector3(x + other.x, y + other.y, z + other.z)

        operator fun times(scale: Double): Vector3 =
            Vector3(x * scale, y * scale, z * scale)

        fun toVec3(): Vec3 = Vec3(x, y, z)
    }

    fun refine(
        joint: ReconstructionJoint,
        observations: List<CameraObservation>,
        priorPosition: Vec3 = joint.position,
    ): Result {
        val valid = observations
                        .filter { it.confidence >= config.minimumAcceptedConfidence }
            .filter { finite(it) }
            .distinctBy { it.deviceId }

        if (valid.size < 2) {
            return rejected(
                joint = joint,
                error = joint.reprojectionErrorPx,
                used = valid.size,
                reason = "insufficient-independent-views",
            )
        }

        val initialMetrics = evaluate(
            position = joint.position,
            observations = valid,
            priorPosition = priorPosition,
        )

        if (!initialMetrics.acceptedGeometry) {
            return rejected(
                joint = joint,
                error = initialMetrics.errorPx,
                used = initialMetrics.used,
                reason = initialMetrics.reason,
                conditionScore = initialMetrics.conditionScore,
            )
        }

        var current = joint.position
        var best = current
        var bestError = initialMetrics.errorPx
        var bestCondition = initialMetrics.conditionScore
        var bestInliers = initialMetrics.inliers
        var bestUsed = initialMetrics.used
        var bestUncertainty = initialMetrics.uncertaintyMeters
        var damping = config.damping
        var totalStep = 0.0

        repeat(config.iterations) {
            val metrics = linearize(
                position = current,
                observations = valid,
                priorPosition = priorPosition,
            )

            if (!metrics.acceptedGeometry) {
                damping *= config.dampingGrowth
                return@repeat
            }

            val regularized = metrics.hessian.addDiagonal(damping)
            val delta = solve(
                matrix = regularized,
                rhs = metrics.gradient,
            ) ?: run {
                damping *= config.dampingGrowth
                return@repeat
            }

            var step = delta.toVec3()
            val rawStepMeters = step.norm()
            if (!rawStepMeters.isFinite()) {
                damping *= config.dampingGrowth
                return@repeat
            }

            if (rawStepMeters > config.maximumStepMeters) {
                step *= config.maximumStepMeters / rawStepMeters
            }

            val forwardCandidate = current + step
            val forwardMetrics = evaluate(
                position = forwardCandidate,
                observations = valid,
                priorPosition = priorPosition,
            )

            val backwardCandidate = current - step
            val backwardMetrics = evaluate(
                position = backwardCandidate,
                observations = valid,
                priorPosition = priorPosition,
            )

            val (candidate, candidateMetrics, acceptedStep) =
                when {
                    forwardMetrics.acceptedGeometry &&
                        (
                            !backwardMetrics.acceptedGeometry ||
                                forwardMetrics.errorPx <= backwardMetrics.errorPx
                        ) ->
                        Triple(
                            forwardCandidate,
                            forwardMetrics,
                            Vector3(step.x, step.y, step.z),
                        )

                    backwardMetrics.acceptedGeometry ->
                        Triple(
                            backwardCandidate,
                            backwardMetrics,
                            Vector3(-step.x, -step.y, -step.z),
                        )

                    else ->
                        Triple(current, forwardMetrics, Vector3(0.0, 0.0, 0.0))
                }

            if (
                candidateMetrics.acceptedGeometry &&
                candidateMetrics.errorPx + config.minimumImprovementPx < bestError
            ) {
                current = candidate
                best = candidate
                bestError = candidateMetrics.errorPx
                bestCondition = candidateMetrics.conditionScore
                bestInliers = candidateMetrics.inliers
                bestUsed = candidateMetrics.used
                bestUncertainty = candidateMetrics.uncertaintyMeters
                totalStep += acceptedStep.toVec3().norm()
                damping = max(config.dampingFloor, damping / config.dampingGrowth)
            } else {
                damping *= config.dampingGrowth
            }
        }

        val improvement = initialMetrics.errorPx - bestError
        val accepted = (
            improvement >= config.minimumImprovementPx ||
                initialMetrics.errorPx <= joint.reprojectionErrorPx + config.minimumImprovementPx
        ) && bestCondition >= config.minimumConditionScore &&
            best.inFrontOfMajority(valid)

        if (!accepted) {
            return rejected(
                joint = joint,
                error = initialMetrics.errorPx,
                used = initialMetrics.used,
                reason = "refinement-rejected-by-geometric-gate",
                conditionScore = bestCondition,
                uncertaintyMeters = initialMetrics.uncertaintyMeters,
            )
        }

        val confidenceBoost = (
            0.52 * exp(-bestError / 4.0) +
                0.28 * (bestInliers.toDouble() / bestUsed.coerceAtLeast(1)) +
                0.20 * bestCondition
        ).coerceIn(0.0, 1.0)

        val finalConfidence = (
            joint.confidence * 0.58 +
                confidenceBoost * 0.42
        ).coerceIn(0.0, 1.0)

        val finalVelocity = if (totalStep <= 1e-9) {
            joint.velocity
        } else {
            val alpha = (totalStep / config.maximumStepMeters)
                .coerceIn(0.0, 1.0)
            joint.velocity.lerp(
                (best - joint.position) * (1.0 / max(alpha, 1e-6)),
                0.25 * alpha,
            )
        }

        val refined = joint.copy(
            position = best,
            velocity = finalVelocity,
            confidence = finalConfidence,
            reprojectionErrorPx = bestError,
            observationCount = bestUsed,
            positionUncertaintyMeters = bestUncertainty
                .coerceIn(config.uncertaintyFloorMeters, config.uncertaintyCeilingMeters),
        )

        return Result(
            joint = refined,
            accepted = true,
            initialErrorPx = initialMetrics.errorPx,
            finalErrorPx = bestError,
            observationsUsed = bestUsed,
            inliers = bestInliers,
            rejectedOutliers = valid.size - bestInliers,
            conditionScore = bestCondition,
            stepMeters = totalStep,
            uncertaintyMeters = refined.positionUncertaintyMeters,
            reason = if (improvement > 1.0) {
                "large-geometry-correction"
            } else {
                "robust-multiview-refinement"
            },
        )
    }

    private data class Metrics(
        val acceptedGeometry: Boolean,
        val errorPx: Double,
        val used: Int,
        val inliers: Int,
        val uncertaintyMeters: Double,
        val conditionScore: Double,
        val hessian: Matrix3x3 = Matrix3x3(),
        val gradient: Vector3 = Vector3(0.0, 0.0, 0.0),
        val reason: String = "ok",
    )

    private fun linearize(
        position: Vec3,
        observations: List<CameraObservation>,
        priorPosition: Vec3,
    ): Metrics {
        val residuals = observations.mapNotNull { buildResidual(position, it) }
        if (residuals.size < 2) {
            return Metrics(
                acceptedGeometry = false,
                errorPx = Double.POSITIVE_INFINITY,
                used = residuals.size,
                inliers = 0,
                uncertaintyMeters = config.uncertaintyCeilingMeters,
                conditionScore = 0.0,
                reason = "too-few-projectable-observations",
            )
        }

        var hessian = Matrix3x3()
        var gradient = Vector3(0.0, 0.0, 0.0)
        var weightedError = 0.0
        var weightSum = 0.0
        var inliers = 0
        var maxResidual = 0.0
        val residualMagnitudes =
            residuals.map { hypot(it.ex, it.ey) }.sorted()
        val medianResidual =
            residualMagnitudes[residualMagnitudes.lastIndex / 2]
        val adaptiveInlierThreshold =
            max(
                config.huberThresholdPx * 2.0,
                medianResidual * 1.75,
            )

        residuals.forEach { residual ->
            val j = residual.projection.jacobian
            val w = residual.weight

            hessian += Matrix3x3(
                j.x0*j.x0 + j.y0*j.y0,
                j.x0*j.x1 + j.y0*j.y1,
                j.x0*j.x2 + j.y0*j.y2,
                j.x1*j.x0 + j.y1*j.y0,
                j.x1*j.x1 + j.y1*j.y1,
                j.x1*j.x2 + j.y1*j.y2,
                j.x2*j.x0 + j.y2*j.y0,
                j.x2*j.x1 + j.y2*j.y1,
                j.x2*j.x2 + j.y2*j.y2,
            ) * w

            gradient += Vector3(
                j.x0*residual.ex + j.y0*residual.ey,
                j.x1*residual.ex + j.y1*residual.ey,
                j.x2*residual.ex + j.y2*residual.ey,
            ) * w

            val magnitude = hypot(
                residual.ex,
                residual.ey,
            )
            weightedError += magnitude * magnitude * w
            weightSum += w
            maxResidual = max(maxResidual, magnitude)
            if (magnitude <= adaptiveInlierThreshold) {
                inliers += 1
            }
        }

        val prior = position - priorPosition
        val priorW = config.priorWeight
        hessian += Matrix3x3(
            priorW, 0.0, 0.0,
            0.0, priorW, 0.0,
            0.0, 0.0, priorW,
        )
        gradient += Vector3(
            prior.x * priorW,
            prior.y * priorW,
            prior.z * priorW,
        )

        val condition = conditioningScore(hessian)
        val rms = sqrt(
            weightedError /
                weightSum.coerceAtLeast(1e-9),
        )

        val uncertainty = uncertaintyFromHessian(
            hessian = hessian,
            reprojectionErrorPx = rms,
        )

        val robustSupport =
            inliers >= 2 &&
                inliers.toDouble() /
                    residuals.size.toDouble() >=
                    0.50
        val accepted = (
            (
                maxResidual <= config.maximumObservationResidualPx ||
                    robustSupport
                ) &&
                condition >= config.minimumConditionScore &&
                uncertainty <= config.uncertaintyCeilingMeters &&
                residuals.size >= 2
        )

        return Metrics(
            acceptedGeometry = accepted,
            errorPx = rms,
            used = residuals.size,
            inliers = inliers,
            uncertaintyMeters = uncertainty,
            conditionScore = condition,
            hessian = hessian,
            gradient = gradient,
            reason = when {
                maxResidual > config.maximumObservationResidualPx &&
                    !robustSupport ->
                    "observation-residual-too-large"
                condition < config.minimumConditionScore ->
                    "ill-conditioned-bundle"
                uncertainty > config.uncertaintyCeilingMeters ->
                    "uncertainty-too-large"
                else -> "ok"
            },
        )
    }

    private fun evaluate(
        position: Vec3,
        observations: List<CameraObservation>,
        priorPosition: Vec3,
    ): Metrics {
        val linearized = linearize(
            position = position,
            observations = observations,
            priorPosition = priorPosition,
        )

        return linearized.copy(
            hessian = Matrix3x3(),
            gradient = Vector3(0.0, 0.0, 0.0),
        )
    }

    private fun buildResidual(
        position: Vec3,
        observation: CameraObservation,
    ): Residual? {
        val projection = project(
            position = position,
            observation = observation,
        ) ?: return null

        val ex = observation.point.x - projection.x
        val ey = observation.point.y - projection.y
        val magnitude = hypot(ex, ey)

        if (!magnitude.isFinite()) return null

        val robust = huberWeight(magnitude)
        val sourceWeight = (
            observation.confidence.coerceIn(0.0, 1.0) *
                exp(-observation.rollingShutterFraction.absoluteValue * 0.5)
        ).coerceAtLeast(0.05)

        return Residual(
            ex = ex,
            ey = ey,
            weight = robust * sourceWeight,
            projection = projection,
            observation = observation,
        )
    }

    private fun project(
        position: Vec3,
        observation: CameraObservation,
    ): Projection? {
        val rotation = observation.pose.rotation.normalized()
        val cameraPoint = rotation
            .conjugate()
            .rotate(position - observation.pose.position)

        val z = cameraPoint.z
        if (!z.isFinite() || z <= config.minimumDepthMeters || z >= config.maximumDepthMeters) {
            return null
        }

        val x = cameraPoint.x / z
        val y = cameraPoint.y / z
        val radial2 = x*x + y*y
        val radial = (
            1.0 +
                observation.intrinsics.k1 * radial2 +
                observation.intrinsics.k2 * radial2 * radial2 +
                observation.intrinsics.k3 * radial2 * radial2 * radial2
        )
        val tx = (
            2.0 * observation.intrinsics.p1 * x * y +
                observation.intrinsics.p2 * (radial2 + 2.0 * x*x)
        )
        val ty = (
            observation.intrinsics.p1 * (radial2 + 2.0 * y*y) +
                2.0 * observation.intrinsics.p2 * x * y
        )
        val xd = x * radial + tx
        val yd = y * radial + ty
        val u = observation.intrinsics.fx * xd + observation.intrinsics.cx
        val v = observation.intrinsics.fy * yd + observation.intrinsics.cy

        val rotationWorldToCamera =
            rotation
                .conjugate()
                .normalized()

        val radialDerivative =
            observation.intrinsics.k1 +
                2.0 *
                    observation.intrinsics.k2 *
                    radial2 +
                3.0 *
                    observation.intrinsics.k3 *
                    radial2 *
                    radial2

        val dXDistortedDx =
            radial +
                2.0 *
                    x *
                    x *
                    radialDerivative +
                2.0 *
                    observation.intrinsics.p1 *
                    y +
                6.0 *
                    observation.intrinsics.p2 *
                    x

        val dXDistortedDy =
            2.0 *
                x *
                y *
                radialDerivative +
                2.0 *
                    observation.intrinsics.p1 *
                    x +
                2.0 *
                    observation.intrinsics.p2 *
                    y

        val dYDistortedDx =
            2.0 *
                x *
                y *
                radialDerivative +
                2.0 *
                    observation.intrinsics.p1 *
                    x +
                2.0 *
                    observation.intrinsics.p2 *
                    y

        val dYDistortedDy =
            radial +
                2.0 *
                    y *
                    y *
                    radialDerivative +
                6.0 *
                    observation.intrinsics.p1 *
                    y +
                2.0 *
                    observation.intrinsics.p2 *
                    x

        val inverseDepth =
            1.0 / z

        val dXNormDz =
            -x *
                inverseDepth

        val dYNormDz =
            -y *
                inverseDepth

        val duDx =
            observation.intrinsics.fx *
                dXDistortedDx *
                inverseDepth

        val duDy =
            observation.intrinsics.fx *
                dXDistortedDy *
                inverseDepth

        val duDz =
            observation.intrinsics.fx *
                (
                    dXDistortedDx *
                        dXNormDz +
                        dXDistortedDy *
                        dYNormDz
                )

        val dvDx =
            observation.intrinsics.fy *
                dYDistortedDx *
                inverseDepth

        val dvDy =
            observation.intrinsics.fy *
                dYDistortedDy *
                inverseDepth

        val dvDz =
            observation.intrinsics.fy *
                (
                    dYDistortedDx *
                        dXNormDz +
                        dYDistortedDy *
                        dYNormDz
                )

        val worldX =
            rotationWorldToCamera
                .rotate(
                    Vec3(
                        1.0,
                        0.0,
                        0.0,
                    ),
                )
        val worldY =
            rotationWorldToCamera
                .rotate(
                    Vec3(
                        0.0,
                        1.0,
                        0.0,
                    ),
                )
        val worldZ =
            rotationWorldToCamera
                .rotate(
                    Vec3(
                        0.0,
                        0.0,
                        1.0,
                    ),
                )

        val jacobian =
            Matrix2x3(
                x0 =
                    duDx * worldX.x +
                        duDy * worldX.y +
                        duDz * worldX.z,
                x1 =
                    duDx * worldY.x +
                        duDy * worldY.y +
                        duDz * worldY.z,
                x2 =
                    duDx * worldZ.x +
                        duDy * worldZ.y +
                        duDz * worldZ.z,
                y0 =
                    dvDx * worldX.x +
                        dvDy * worldX.y +
                        dvDz * worldX.z,
                y1 =
                    dvDx * worldY.x +
                        dvDy * worldY.y +
                        dvDz * worldY.z,
                y2 =
                    dvDx * worldZ.x +
                        dvDy * worldZ.y +
                        dvDz * worldZ.z,
            )

        return Projection(
            x = u,
            y = v,
            depth = z,
            jacobian = jacobian,
        )
    }

    private fun huberWeight(
        residualPx: Double,
    ): Double {
        val r = residualPx.absoluteValue
        if (r <= config.huberThresholdPx) return 1.0
        return config.huberThresholdPx / r.coerceAtLeast(1e-9)
    }

    private fun conditioningScore(
        matrix: Matrix3x3,
    ): Double {
        val eigenvalues =
            symmetricEigenvalues(
                matrix,
            )

        val finitePositive =
            eigenvalues
                .filter {
                    it.isFinite() &&
                        it > 1e-12
                }

        val weakest =
            finitePositive.minOrNull()
                ?: return 0.0
        val strongest =
            finitePositive.maxOrNull()
                ?: return 0.0

        if (
            strongest <= 1e-12 ||
            weakest <= 1e-12
        ) {
            return 0.0
        }

        return (
            weakest /
                strongest
            ).coerceIn(0.0, 1.0)
    }

    private fun uncertaintyFromHessian(
        hessian: Matrix3x3,
        reprojectionErrorPx: Double,
    ): Double {
        val eigenvalues =
            symmetricEigenvalues(
                hessian,
            )

        val weakest =
            eigenvalues
                .filter {
                    it.isFinite() &&
                        it > 1e-12
                }
                .minOrNull()
                ?: return config.uncertaintyCeilingMeters

        val scale =
            max(
                reprojectionErrorPx / 2.0,
                0.5,
            )

        return (
            scale /
                sqrt(
                    weakest,
                )
            ).coerceIn(
                config.uncertaintyFloorMeters,
                config.uncertaintyCeilingMeters,
            )
    }

    /**
     * Jacobi diagonalization for the symmetric 3x3 normal matrix.
     *
     * The matrix is symmetric by construction (JᵀWJ plus diagonal damping/prior terms). Full
     * eigenvalue conditioning captures cross-axis correlation that diagonal-only heuristics miss.
     * Six sweeps are more than enough for this tiny matrix and avoid pulling in a heavyweight
     * linear-algebra dependency on-device.
     */
    private fun symmetricEigenvalues(
        matrix: Matrix3x3,
    ): List<Double> {
        var a00 = matrix.m00
        var a01 = matrix.m01
        var a02 = matrix.m02
        var a11 = matrix.m11
        var a12 = matrix.m12
        var a22 = matrix.m22

        repeat(8) {
            val p: Int
            val q: Int
            val apq: Double

            when {
                abs(a01) >= abs(a02) &&
                    abs(a01) >= abs(a12) -> {
                    p = 0
                    q = 1
                    apq = a01
                }

                abs(a02) >= abs(a12) -> {
                    p = 0
                    q = 2
                    apq = a02
                }

                else -> {
                    p = 1
                    q = 2
                    apq = a12
                }
            }

            if (
                !apq.isFinite() ||
                abs(apq) <= 1e-12
            ) {
                return@repeat
            }

            val app =
                when (p) {
                    0 -> a00
                    1 -> a11
                    else -> a22
                }
            val aqq =
                when (q) {
                    0 -> a00
                    1 -> a11
                    else -> a22
                }

            val theta =
                0.5 *
                    kotlin.math.atan2(
                        2.0 * apq,
                        aqq - app,
                    )
            val cosTheta =
                kotlin.math.cos(theta)
            val sinTheta =
                kotlin.math.sin(theta)

            fun rotate(
                aip: Double,
                aiq: Double,
            ): Pair<Double, Double> =
                (
                    aip * cosTheta -
                        aiq * sinTheta
                    ) to
                    (
                        aip * sinTheta +
                            aiq * cosTheta
                        )

            if (p == 0 && q == 1) {
                val a02Old = a02
                val a12Old = a12
                a00 =
                    app * cosTheta * cosTheta +
                        aqq * sinTheta * sinTheta -
                        2.0 * apq *
                        sinTheta *
                        cosTheta
                a11 =
                    app * sinTheta * sinTheta +
                        aqq * cosTheta * cosTheta +
                        2.0 * apq *
                        sinTheta *
                        cosTheta
                a01 = 0.0
                val r02 = rotate(a02Old, a12Old)
                a02 = r02.first
                a12 = r02.second
            } else if (p == 0 && q == 2) {
                val a01Old = a01
                val a12Old = a12
                a00 =
                    app * cosTheta * cosTheta +
                        aqq * sinTheta * sinTheta -
                        2.0 * apq *
                        sinTheta *
                        cosTheta
                a22 =
                    app * sinTheta * sinTheta +
                        aqq * cosTheta * cosTheta +
                        2.0 * apq *
                        sinTheta *
                        cosTheta
                a02 = 0.0
                val r01 = rotate(a01Old, a12Old)
                a01 = r01.first
                a12 = r01.second
            } else {
                val a01Old = a01
                val a02Old = a02
                a11 =
                    app * cosTheta * cosTheta +
                        aqq * sinTheta * sinTheta -
                        2.0 * apq *
                        sinTheta *
                        cosTheta
                a22 =
                    app * sinTheta * sinTheta +
                        aqq * cosTheta * cosTheta +
                        2.0 * apq *
                        sinTheta *
                        cosTheta
                a12 = 0.0
                val r01 = rotate(a01Old, a02Old)
                a01 = r01.first
                a02 = r01.second
            }
        }

        return listOf(
            a00,
            a11,
            a22,
        )
    }

    private fun solve(
        matrix: Matrix3x3,
        rhs: Vector3,
    ): Vector3? {
        val a = Array(3) { DoubleArray(4) }
        a[0][0] = matrix.m00
        a[0][1] = matrix.m01
        a[0][2] = matrix.m02
        a[0][3] = rhs.x
        a[1][0] = matrix.m10
        a[1][1] = matrix.m11
        a[1][2] = matrix.m12
        a[1][3] = rhs.y
        a[2][0] = matrix.m20
        a[2][1] = matrix.m21
        a[2][2] = matrix.m22
        a[2][3] = rhs.z

        for (column in 0..2) {
            var pivot = column
            for (row in column + 1..2) {
                if (abs(a[row][column]) > abs(a[pivot][column])) {
                    pivot = row
                }
            }

            if (abs(a[pivot][column]) < 1e-12) {
                return null
            }

            if (pivot != column) {
                val temp = a[column]
                a[column] = a[pivot]
                a[pivot] = temp
            }

            val pivotValue = a[column][column]
            for (j in column..3) {
                a[column][j] /= pivotValue
            }

            for (row in 0..2) {
                if (row == column) continue
                val factor = a[row][column]
                for (j in column..3) {
                    a[row][j] -= factor * a[column][j]
                }
            }
        }

        return Vector3(
            a[0][3],
            a[1][3],
            a[2][3],
        )
    }

    private fun rejected(
        joint: ReconstructionJoint,
        error: Double,
        used: Int,
        reason: String,
        conditionScore: Double = 0.0,
        uncertaintyMeters: Double = Double.NaN,
    ): Result {
        val safeUncertainty = if (uncertaintyMeters.isFinite()) {
            uncertaintyMeters
        } else {
            max(
                joint.positionUncertaintyMeters,
                config.uncertaintyCeilingMeters,
            )
        }.coerceIn(
            config.uncertaintyFloorMeters,
            config.uncertaintyCeilingMeters,
        )

        return Result(
            joint = joint.copy(
                positionUncertaintyMeters = safeUncertainty,
            ),
            accepted = false,
            initialErrorPx = error,
            finalErrorPx = error,
            observationsUsed = used,
            inliers = 0,
            rejectedOutliers = max(0, used),
            conditionScore = conditionScore,
            stepMeters = 0.0,
            uncertaintyMeters = safeUncertainty,
            reason = reason,
        )
    }

    private fun finite(
        observation: CameraObservation,
    ): Boolean =
        observation.point.x.isFinite() &&
            observation.point.y.isFinite() &&
            observation.confidence.isFinite() &&
            observation.rollingShutterFraction.isFinite()

    private fun Vec3.norm(): Double =
        sqrt(x*x + y*y + z*z)

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() && y.isFinite() && z.isFinite()

    private fun Vec3.inFrontOfMajority(
        observations: List<CameraObservation>,
    ): Boolean {
        var positive = 0
        var valid = 0
        observations.forEach { observation ->
            val cameraPoint = observation.pose.rotation
                .normalized()
                .conjugate()
                .rotate(this - observation.pose.position)
            if (cameraPoint.z.isFinite()) {
                valid += 1
                if (cameraPoint.z > config.minimumDepthMeters) {
                    positive += 1
                }
            }
        }
        return positive >= max(1, (valid * 0.5).toInt())
    }

    private fun hypot(a: Double, b: Double): Double =
        sqrt(a*a + b*b)

    private operator fun Vector3.times(scale: Double): Vector3 =
        Vector3(x * scale, y * scale, z * scale)

    private operator fun Vector3.plus(other: Vector3): Vector3 =
        Vector3(x + other.x, y + other.y, z + other.z)
}
