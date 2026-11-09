package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationUncertaintyProfile
import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Directional uncertainty estimator for metric multi-view triangulation.
 *
 * This engine turns local reprojection sensitivity into a bounded 3D covariance approximation.
 * Unlike a single scalar uncertainty, the result exposes the principal uncertainty axes and a
 * depth-aligned sigma. This is especially important when a point is well localized laterally but
 * poorly constrained in depth because the camera baseline and rays are nearly collinear.
 *
 * The estimator is deliberately deterministic and model-consistent. It uses the same analytic
 * Brown-Conrady projection Jacobian as the primary triangulation solver, retaining finite-difference
 * differentiation only as a defensive fallback for degenerate calibration states. It consumes only
 * the calibrated cameras, reconstructed point, measured image points, and robust residual scale.
 * It does not invent a posterior that is unsupported by the geometry.
 */
class TriangulationUncertaintyEngine(
    private val config: Config = Config(),
    private val calibrationPropagationEngine:
        CalibrationUncertaintyPropagationEngine =
        CalibrationUncertaintyPropagationEngine(),
) {

    data class Config(
        val finiteDifferenceStepMeters: Double = 1e-4,
        val minimumEigenvalue: Double = 1e-9,
        val maximumSigmaMeters: Double = 3.0,
        val confidenceZScore: Double = 1.96,
        val minimumVariancePixels2: Double = 0.25,
        val maximumVariancePixels2: Double = 400.0,
        val damping: Double = 1e-6,
        val jacobianWeightFloor: Double = 1e-4,
        val anisotropyWarningRatio: Double = 4.0,
        val anisotropyHardRatio: Double = 12.0,
    ) {
        init {
            require(finiteDifferenceStepMeters > 0.0)
            require(minimumEigenvalue > 0.0)
            require(maximumSigmaMeters > 0.0)
            require(confidenceZScore > 0.0)
            require(minimumVariancePixels2 > 0.0)
            require(maximumVariancePixels2 > minimumVariancePixels2)
            require(damping > 0.0)
            require(jacobianWeightFloor > 0.0)
            require(anisotropyWarningRatio >= 1.0)
            require(anisotropyHardRatio > anisotropyWarningRatio)
        }
    }

    data class Input(
        val point: Vec3,
        val observations: List<Observation>,
        val residualsPx: List<Double> = emptyList(),
        val robustResidualScalePx: Double = 1.0,
    ) {
        init {
            require(point.x.isFinite() && point.y.isFinite() && point.z.isFinite())
            require(observations.size >= 2)
            require(robustResidualScalePx.isFinite())
            require(robustResidualScalePx > 0.0)
        }
    }

    data class Observation(
        val cameraPose: Pose,
        val intrinsics: CameraIntrinsics,
        val u: Double,
        val v: Double,
        val weight: Double = 1.0,
        val confidence: Double = 1.0,
        val temporalAlignmentConfidence: Double = 1.0,
        val crossViewIndependence: Double = 1.0,
        val calibrationUncertainty:
            CalibrationUncertaintyProfile = CalibrationUncertaintyProfile(),
    ) {
        init {
            require(weight >= 0.0)
            require(confidence in 0.0..1.0)
            require(temporalAlignmentConfidence in 0.0..1.0)
            require(
                crossViewIndependence.isFinite() &&
                    crossViewIndependence in 0.0..1.0,
            )
        }
    }

    data class Result(
        val covariance: Array<DoubleArray>,
        val sigmaMajorMeters: Double,
        val sigmaMiddleMeters: Double,
        val sigmaMinorMeters: Double,
        val sigmaDepthMeters: Double,
        val uncertainty95MajorMeters: Double,
        val anisotropyRatio: Double,
        val depthDominance: Double,
        val geometricConditionScore: Double,
        val confidence: Double,
        val viewIndependenceScore: Double = 1.0,
        val calibrationUncertaintyMeters: Double = 0.0,
        val calibrationUncertaintyApplied: Boolean = false,
        val numericJacobianFallbackCount: Int = 0,
        val valid: Boolean,
        val warnings: List<String>,
    )

    fun estimate(input: Input): Result {
        val validObservations =
            input.observations.filter {
                it.weight > 0.0 &&
                    it.intrinsics.fx > 1e-6 &&
                    it.intrinsics.fy > 1e-6 &&
                    it.temporalAlignmentConfidence > 0.0
            }

        if (validObservations.size < 2) {
            return invalid("uncertainty: insufficient calibrated observations")
        }

        val normal = Array(3) { DoubleArray(3) }
        var numericJacobianFallbackCount = 0

        validObservations.forEach { observation ->
            val projection =
                project(
                    point = input.point,
                    observation = observation,
                ) ?: return@forEach

            val jacobianResolution =
                analyticJacobian(
                    point = input.point,
                    observation = observation,
                )?.let {
                    JacobianResolution(
                        jacobian = it,
                        usedNumericalFallback = false,
                    )
                }
                    ?: JacobianResolution(
                        jacobian =
                            numericJacobian(
                                point = input.point,
                                observation = observation,
                            ),
                        usedNumericalFallback = true,
                    )

            if (jacobianResolution.usedNumericalFallback) {
                numericJacobianFallbackCount += 1
            }

            val jacobian =
                jacobianResolution.jacobian

            val residualError =
                sqrt(
                    (
                        observation.u - projection.x
                        ) * (
                        observation.u - projection.x
                        ) +
                        (
                            observation.v - projection.y
                            ) * (
                            observation.v - projection.y
                        ),
                )

            val robustWeight =
                residualWeight(
                    residualError,
                    input.robustResidualScalePx,
                )

            /*
             * Cross-view independence is an information-quality factor, not a cosmetic score.
             * Views sharing a correlated sensor, clock, crop or inference failure must not receive
             * the same statistical credit as genuinely independent observations.
             */
            val independenceWeight =
                (
                    0.35 +
                        0.65 *
                            observation.crossViewIndependence
                                .coerceIn(0.0, 1.0)
                ).coerceIn(0.35, 1.0)

            val cameraWeight =
                (
                    observation.weight *
                        observation.confidence *
                        observation.temporalAlignmentConfidence *
                        independenceWeight *
                        robustWeight
                    ).coerceAtLeast(
                        config.jacobianWeightFloor,
                    )

            accumulateNormal(
                normal = normal,
                jacobian = jacobian,
                weight = cameraWeight,
            )
        }

        for (axis in 0..2) {
            normal[axis][axis] += config.damping
        }

        val inverse =
            invertSymmetric3(normal)
                ?: return invalid(
                    "uncertainty: normal matrix is singular",
                )

        val residualVariance =
            estimateResidualVariance(
                input = input,
                validObservationCount = validObservations.size,
            )

        val covariance =
            inverse.map { row ->
                DoubleArray(3) { column ->
                    (
                        row[column] *
                            residualVariance
                        ).coerceIn(
                        -config.maximumSigmaMeters * config.maximumSigmaMeters,
                        config.maximumSigmaMeters * config.maximumSigmaMeters,
                    )
                }
            }.toTypedArray()

        /*
         * Calibration uncertainty is an explicit world-space covariance envelope. We select the
         * dominant per-camera envelope rather than dividing shared/systematic calibration error
         * by the number of views.
         */
        val calibrationEnvelope =
            validObservations
                .map { observation ->
                    calibrationPropagationEngine.propagate(
                        pointWorld = input.point,
                        cameraPose = observation.cameraPose,
                        profile = observation.calibrationUncertainty,
                    )
                }
                .filter { it.applied }
                .maxByOrNull {
                    it.equivalentSigmaMeters
                }

        if (calibrationEnvelope != null) {
            for (row in 0..2) {
                for (column in 0..2) {
                    covariance[row][column] = (
                        covariance[row][column] +
                            calibrationEnvelope.covariance[row][column]
                        ).coerceAtMost(
                        config.maximumSigmaMeters *
                            config.maximumSigmaMeters,
                    )
                }
            }
        }

        val calibrationSigma =
            calibrationEnvelope?.equivalentSigmaMeters ?: 0.0

        val eigenvalues =
            symmetricEigenvalues(covariance)

        val sigma =
            eigenvalues.map {
                sqrt(
                    it.coerceIn(
                        config.minimumEigenvalue,
                        config.maximumSigmaMeters *
                            config.maximumSigmaMeters,
                    ),
                )
            }.sortedDescending()

        val sigmaMajor =
            sigma.getOrElse(0) {
                config.maximumSigmaMeters
            }
        val sigmaMiddle =
            sigma.getOrElse(1) {
                sigmaMajor
            }
        val sigmaMinor =
            sigma.getOrElse(2) {
                sigmaMiddle
            }

        val viewingAxis =
            meanViewingDirection(validObservations)

        val sigmaDepth =
            projectedSigma(
                covariance = covariance,
                direction = viewingAxis,
            )

        val anisotropy =
            (
                sigmaMajor /
                    sigmaMinor.coerceAtLeast(
                        config.minimumEigenvalue,
                    )
                ).coerceIn(
                1.0,
                1e6,
            )

        val depthDominance =
            (
                sigmaDepth /
                    sigmaMajor.coerceAtLeast(
                        config.minimumEigenvalue,
                    )
                ).coerceIn(
                0.0,
                1.0,
            )

        val geometricCondition =
            (
                sigmaMinor /
                    sigmaMajor.coerceAtLeast(
                        config.minimumEigenvalue,
                    )
                ).coerceIn(
                0.0,
                1.0,
            )

        val uncertaintyConfidence =
            (
                1.0 -
                    sigmaMajor /
                        config.maximumSigmaMeters
                ).coerceIn(
                0.0,
                1.0,
            )

        val effectiveIndependence =
            weightedIndependence(
                validObservations,
            )

        val independenceAdjustedConfidence =
            (
                uncertaintyConfidence *
                    (
                        0.72 +
                            0.28 *
                                effectiveIndependence
                    )
            ).coerceIn(0.0, 1.0)

        val warnings =
            buildWarnings(
                sigmaMajor = sigmaMajor,
                sigmaDepth = sigmaDepth,
                anisotropy = anisotropy,
                conditionScore = geometricCondition,
            ) +
                buildIndependenceWarnings(
                    effectiveIndependence,
                ) +
                buildCalibrationWarnings(
                calibrationSigma = calibrationSigma,
                observations = validObservations,
            )

        return Result(
            covariance = covariance,
            sigmaMajorMeters = sigmaMajor,
            sigmaMiddleMeters = sigmaMiddle,
            sigmaMinorMeters = sigmaMinor,
            sigmaDepthMeters = sigmaDepth,
            uncertainty95MajorMeters =
                (
                    sigmaMajor *
                        config.confidenceZScore
                    ).coerceAtMost(
                    config.maximumSigmaMeters *
                        config.confidenceZScore,
                ),
            anisotropyRatio = anisotropy,
            depthDominance = depthDominance,
            geometricConditionScore = geometricCondition,
            confidence = independenceAdjustedConfidence,
            viewIndependenceScore = effectiveIndependence,
            calibrationUncertaintyMeters = calibrationSigma,
            calibrationUncertaintyApplied = calibrationSigma > 0.0,
            numericJacobianFallbackCount =
                numericJacobianFallbackCount,
            valid = warnings.none {
                it.contains("hard")
            },
            warnings =
                (
                    warnings +
                        if (
                            numericJacobianFallbackCount > 0
                        ) {
                            listOf(
                                "analytic-jacobian-numerical-fallback-used",
                            )
                        } else {
                            emptyList()
                        }
                ).distinct(),
        )
    }

    private fun weightedIndependence(
        observations: List<Observation>,
    ): Double {
        if (observations.isEmpty()) return 0.0

        var weightedSum = 0.0
        var totalWeight = 0.0

        observations.forEach { observation ->
            val weight =
                (
                    observation.weight.coerceAtLeast(0.0) *
                        observation.confidence.coerceIn(0.0, 1.0)
                )
                    .coerceAtLeast(1e-6)

            weightedSum +=
                weight *
                    observation.crossViewIndependence
                        .coerceIn(0.0, 1.0)
            totalWeight += weight
        }

        return if (totalWeight > 0.0) {
            (weightedSum / totalWeight)
                .coerceIn(0.0, 1.0)
        } else {
            0.0
        }
    }

    private fun buildIndependenceWarnings(
        independence: Double,
    ): List<String> =
        buildList {
            if (independence < 0.55) {
                add("view-independence-low")
            }
            if (independence < 0.35) {
                add("view-independence-hard-risk")
            }
        }

    private fun buildCalibrationWarnings(
        calibrationSigma: Double,
        observations: List<Observation>,
    ): List<String> = buildList {
        val profiled = observations.count {
            it.calibrationUncertainty.hasEvidence
        }
        if (profiled > 0) {
            add("calibration-uncertainty-envelope-applied")
            if (calibrationSigma >= 0.010) {
                add("calibration-uncertainty-material-to-metric-result")
            }
            if (profiled < observations.size) {
                add("calibration-uncertainty-incomplete-camera-coverage")
            }
        }
    }.distinct()

    private fun residualWeight(
        residualPx: Double,
        robustScalePx: Double,
    ): Double {
        if (!residualPx.isFinite()) return 0.02

        val normalized =
            residualPx /
                robustScalePx.coerceAtLeast(
                    config.minimumVariancePixels2,
                )

        return (
            1.0 /
                (
                    1.0 +
                        normalized * normalized
                    )
            ).coerceIn(
            0.02,
            1.0,
        )
    }

    private fun estimateResidualVariance(
        input: Input,
        validObservationCount: Int,
    ): Double {
        val explicit =
            input.residualsPx
                .filter { it.isFinite() && it >= 0.0 }

        val meanResidualSquared =
            if (explicit.isNotEmpty()) {
                explicit.map {
                    it * it
                }.average()
            } else {
                input.robustResidualScalePx *
                    input.robustResidualScalePx
            }

        val degreesOfFreedom =
            (
                validObservationCount * 2 - 3
            ).coerceAtLeast(1)

        val dofScale =
            validObservationCount.toDouble() /
                degreesOfFreedom.toDouble()

        return (
            meanResidualSquared *
                dofScale
            ).coerceIn(
            config.minimumVariancePixels2,
            config.maximumVariancePixels2,
        )
    }

    private data class JacobianResolution(
        val jacobian: Array<DoubleArray>,
        val usedNumericalFallback: Boolean,
    )

    private fun analyticJacobian(
        point: Vec3,
        observation: Observation,
    ): Array<DoubleArray>? {
        val inverse =
            observation.cameraPose.rotation
                .conjugate()
                .normalized()

        val cameraPoint =
            inverse.rotate(
                point -
                    observation.cameraPose.position,
            )

        val z = cameraPoint.z
        if (!z.isFinite() || z <= 1e-6) return null

        val x = cameraPoint.x / z
        val y = cameraPoint.y / z
        if (!x.isFinite() || !y.isFinite()) return null

        val r2 = x * x + y * y
        val r4 = r2 * r2
        val r6 = r4 * r2

        val k1 = observation.intrinsics.k1
        val k2 = observation.intrinsics.k2
        val k3 = observation.intrinsics.k3
        val p1 = observation.intrinsics.p1
        val p2 = observation.intrinsics.p2

        val radial =
            1.0 +
                k1 * r2 +
                k2 * r4 +
                k3 * r6
        if (!radial.isFinite()) return null

        val radialSlope =
            k1 +
                2.0 * k2 * r2 +
                3.0 * k3 * r4

        val dRadialDx =
            2.0 * x * radialSlope
        val dRadialDy =
            2.0 * y * radialSlope

        val dTxDx =
            2.0 * p1 * y +
                6.0 * p2 * x
        val dTxDy =
            2.0 * p1 * x +
                2.0 * p2 * y
        val dTyDx =
            2.0 * p1 * x +
                2.0 * p2 * y
        val dTyDy =
            6.0 * p1 * y +
                2.0 * p2 * x

        val dDistortedXDx =
            radial +
                x * dRadialDx +
                dTxDx
        val dDistortedXDy =
            x * dRadialDy +
                dTxDy
        val dDistortedYDx =
            y * dRadialDx +
                dTyDx
        val dDistortedYDy =
            radial +
                y * dRadialDy +
                dTyDy

        val invZ = 1.0 / z
        val invZ2 = invZ * invZ

        val worldToCameraX =
            inverse.rotate(
                Vec3(1.0, 0.0, 0.0),
            )
        val worldToCameraY =
            inverse.rotate(
                Vec3(0.0, 1.0, 0.0),
            )
        val worldToCameraZ =
            inverse.rotate(
                Vec3(0.0, 0.0, 1.0),
            )

        fun normalizedDerivative(
            dCamera: Vec3,
        ): Vec2 =
            Vec2(
                (
                    dCamera.x * z -
                        cameraPoint.x *
                        dCamera.z
                    ) * invZ2,
                (
                    dCamera.y * z -
                        cameraPoint.y *
                        dCamera.z
                    ) * invZ2,
            )

        val dNormX =
            listOf(
                normalizedDerivative(
                    worldToCameraX,
                ),
                normalizedDerivative(
                    worldToCameraY,
                ),
                normalizedDerivative(
                    worldToCameraZ,
                ),
            )

        val duDx =
            dDistortedXDx * dNormX[0].x +
                dDistortedXDy * dNormX[0].y
        val duDy =
            dDistortedXDx * dNormX[1].x +
                dDistortedXDy * dNormX[1].y
        val duDz =
            dDistortedXDx * dNormX[2].x +
                dDistortedXDy * dNormX[2].y

        val dvDx =
            dDistortedYDx * dNormX[0].x +
                dDistortedYDy * dNormX[0].y
        val dvDy =
            dDistortedYDx * dNormX[1].x +
                dDistortedYDy * dNormX[1].y
        val dvDz =
            dDistortedYDx * dNormX[2].x +
                dDistortedYDy * dNormX[2].y

        val jacobian =
            arrayOf(
                doubleArrayOf(
                    observation.intrinsics.fx * duDx,
                    observation.intrinsics.fx * duDy,
                    observation.intrinsics.fx * duDz,
                ),
                doubleArrayOf(
                    observation.intrinsics.fy * dvDx,
                    observation.intrinsics.fy * dvDy,
                    observation.intrinsics.fy * dvDz,
                ),
            )

        if (
            jacobian.any { row ->
                row.any { value ->
                    !value.isFinite()
                }
            }
        ) {
            return null
        }

        return jacobian
    }

    private fun numericJacobian(
        point: Vec3,
        observation: Observation,
    ): Array<DoubleArray> {
        val h =
            config.finiteDifferenceStepMeters

        fun projectShift(
            axis: Int,
            sign: Double,
        ): Vec2 {
            val shifted =
                when (axis) {
                    0 -> point + Vec3(sign * h, 0.0, 0.0)
                    1 -> point + Vec3(0.0, sign * h, 0.0)
                    else -> point + Vec3(0.0, 0.0, sign * h)
                }

            return project(
                point = shifted,
                observation = observation,
            ) ?: Vec2(
                observation.u,
                observation.v,
            )
        }

        val columns =
            (0..2).map { axis ->
                val plus =
                    projectShift(axis, 1.0)
                val minus =
                    projectShift(axis, -1.0)

                Vec2(
                    x =
                        (
                            plus.x - minus.x
                            ) /
                            (2.0 * h),
                    y =
                        (
                            plus.y - minus.y
                            ) /
                            (2.0 * h),
                )
            }

        return arrayOf(
            doubleArrayOf(
                columns[0].x,
                columns[1].x,
                columns[2].x,
            ),
            doubleArrayOf(
                columns[0].y,
                columns[1].y,
                columns[2].y,
            ),
        )
    }

    private fun accumulateNormal(
        normal: Array<DoubleArray>,
        jacobian: Array<DoubleArray>,
        weight: Double,
    ) {
        for (r in 0..2) {
            for (c in 0..2) {
                normal[r][c] +=
                    weight * (
                        jacobian[0][r] *
                            jacobian[0][c] +
                            jacobian[1][r] *
                            jacobian[1][c]
                        )
            }
        }
    }

    private fun project(
        point: Vec3,
        observation: Observation,
    ): Vec2? {
        val inverse =
            observation.cameraPose.rotation
                .conjugate()
                .normalized()

        val cameraPoint =
            inverse.rotate(
                point -
                    observation.cameraPose.position,
            )

        if (
            !cameraPoint.isFinite() ||
            cameraPoint.z <= 1e-6
        ) {
            return null
        }

        var x =
            cameraPoint.x /
                cameraPoint.z
        var y =
            cameraPoint.y /
                cameraPoint.z

        val r2 =
            x * x + y * y

        val radial =
            1.0 +
                observation.intrinsics.k1 * r2 +
                observation.intrinsics.k2 * r2 * r2 +
                observation.intrinsics.k3 * r2 * r2 * r2

        val dx =
            2.0 * observation.intrinsics.p1 * x * y +
                observation.intrinsics.p2 *
                (
                    r2 +
                        2.0 * x * x
                    )

        val dy =
            observation.intrinsics.p1 *
                (
                    r2 +
                        2.0 * y * y
                    ) +
                2.0 * observation.intrinsics.p2 * x * y

        x =
            x * radial + dx
        y =
            y * radial + dy

        return Vec2(
            x * observation.intrinsics.fx +
                observation.intrinsics.cx,
            y * observation.intrinsics.fy +
                observation.intrinsics.cy,
        )
    }

    private fun meanViewingDirection(
        observations: List<Observation>,
    ): Vec3 {
        var direction =
            Vec3(0.0, 0.0, 0.0)

        observations.forEach {
            direction +=
                it.cameraPose.rotation
                    .rotate(
                        Vec3(
                            0.0,
                            0.0,
                            1.0,
                        ),
                    )
                    .normalized()
        }

        return direction.normalized()
            .takeIf { it.isFinite() && it.norm() > 1e-6 }
            ?: Vec3(0.0, 0.0, 1.0)
    }

    private fun projectedSigma(
        covariance: Array<DoubleArray>,
        direction: Vec3,
    ): Double {
        val vector =
            doubleArrayOf(
                direction.x,
                direction.y,
                direction.z,
            )

        var variance = 0.0

        for (row in 0..2) {
            for (column in 0..2) {
                variance +=
                    vector[row] *
                        covariance[row][column] *
                        vector[column]
            }
        }

        return sqrt(
            variance.coerceAtLeast(
                config.minimumEigenvalue,
            ),
        ).coerceAtMost(
            config.maximumSigmaMeters,
        )
    }

    private fun invertSymmetric3(
        matrix: Array<DoubleArray>,
    ): Array<DoubleArray>? {
        val determinant =
            determinant3(matrix)

        if (
            !determinant.isFinite() ||
            abs(determinant) < 1e-12
        ) {
            return null
        }

        val inverse =
            Array(3) {
                DoubleArray(3)
            }

        inverse[0][0] =
            cofactor(matrix, 0, 0) /
                determinant
        inverse[0][1] =
            cofactor(matrix, 1, 0) /
                determinant
        inverse[0][2] =
            cofactor(matrix, 2, 0) /
                determinant
        inverse[1][0] =
            cofactor(matrix, 0, 1) /
                determinant
        inverse[1][1] =
            cofactor(matrix, 1, 1) /
                determinant
        inverse[1][2] =
            cofactor(matrix, 2, 1) /
                determinant
        inverse[2][0] =
            cofactor(matrix, 0, 2) /
                determinant
        inverse[2][1] =
            cofactor(matrix, 1, 2) /
                determinant
        inverse[2][2] =
            cofactor(matrix, 2, 2) /
                determinant

        return inverse
    }

    private fun determinant3(
        matrix: Array<DoubleArray>,
    ): Double =
        matrix[0][0] *
            (
                matrix[1][1] *
                    matrix[2][2] -
                    matrix[1][2] *
                    matrix[2][1]
                ) -
            matrix[0][1] *
            (
                matrix[1][0] *
                    matrix[2][2] -
                    matrix[1][2] *
                    matrix[2][0]
                ) +
            matrix[0][2] *
            (
                matrix[1][0] *
                    matrix[2][1] -
                    matrix[1][1] *
                    matrix[2][0]
                )

    private fun cofactor(
        matrix: Array<DoubleArray>,
        row: Int,
        column: Int,
    ): Double {
        val rows =
            (0..2).filter { it != row }
        val columns =
            (0..2).filter { it != column }

        return (
            matrix[rows[0]][columns[0]] *
                matrix[rows[1]][columns[1]] -
                matrix[rows[0]][columns[1]] *
                matrix[rows[1]][columns[0]]
            ) *
            if ((row + column) % 2 == 0) {
                1.0
            } else {
                -1.0
            }
    }

    private fun symmetricEigenvalues(
        matrix: Array<DoubleArray>,
    ): List<Double> {
        val a =
            Array(3) { row ->
                doubleArrayOf(
                    matrix[row][0],
                    matrix[row][1],
                    matrix[row][2],
                )
            }

        repeat(18) {
            var p = 0
            var q = 1
            var largest =
                abs(a[p][q])

            for (i in 0..2) {
                for (j in i + 1..2) {
                    if (abs(a[i][j]) > largest) {
                        largest =
                            abs(a[i][j])
                        p = i
                        q = j
                    }
                }
            }

            if (largest < 1e-10) {
                return@repeat
            }

            val angle =
                0.5 *
                    kotlin.math.atan2(
                        2.0 * a[p][q],
                        a[q][q] -
                            a[p][p],
                    )
            val cosine =
                kotlin.math.cos(angle)
            val sine =
                kotlin.math.sin(angle)

            for (k in 0..2) {
                val apk = a[p][k]
                val aqk = a[q][k]
                a[p][k] =
                    cosine * apk -
                        sine * aqk
                a[q][k] =
                    sine * apk +
                        cosine * aqk
            }

            for (k in 0..2) {
                val akp = a[k][p]
                val akq = a[k][q]
                a[k][p] =
                    cosine * akp -
                        sine * akq
                a[k][q] =
                    sine * akp +
                        cosine * akq
            }
        }

        return listOf(
            a[0][0],
            a[1][1],
            a[2][2],
        )
            .map {
                it.coerceAtLeast(
                    config.minimumEigenvalue,
                )
            }
            .sortedDescending()
    }

    private fun buildWarnings(
        sigmaMajor: Double,
        sigmaDepth: Double,
        anisotropy: Double,
        conditionScore: Double,
    ): List<String> = buildList {
        if (
            sigmaMajor >=
                config.maximumSigmaMeters * 0.72
        ) {
            add("uncertainty-major-axis-large")
        }

        if (
            sigmaDepth >=
                config.maximumSigmaMeters * 0.60
        ) {
            add("uncertainty-depth-axis-large")
        }

        if (
            anisotropy >=
                config.anisotropyWarningRatio
        ) {
            add("uncertainty-highly-anisotropic")
        }

        if (
            anisotropy >=
                config.anisotropyHardRatio
        ) {
            add("uncertainty-anisotropy-hard")
        }

        if (
            conditionScore <= 0.14
        ) {
            add("uncertainty-geometry-ill-conditioned")
        }
    }.distinct()

    private fun invalid(
        warning: String,
    ): Result =
        Result(
            covariance =
                Array(3) {
                    DoubleArray(3)
                },
            sigmaMajorMeters =
                config.maximumSigmaMeters,
            sigmaMiddleMeters =
                config.maximumSigmaMeters,
            sigmaMinorMeters =
                config.maximumSigmaMeters,
            sigmaDepthMeters =
                config.maximumSigmaMeters,
            uncertainty95MajorMeters =
                config.maximumSigmaMeters *
                    config.confidenceZScore,
            anisotropyRatio =
                config.anisotropyHardRatio,
            depthDominance = 1.0,
            geometricConditionScore = 0.0,
            confidence = 0.0,
            valid = false,
            warnings = listOf(warning),
        )

    private fun Vec3.isFinite(): Boolean =
        x.isFinite() &&
            y.isFinite() &&
            z.isFinite()
}
