package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Sparse camera-rig bundle adjustment for NoX calibration.
 *
 * The reference camera remains fixed. Other cameras are optimized in six degrees of freedom:
 * translation (meters) + a small-angle rotation vector. 3D calibration points are re-triangulated
 * between iterations, while camera updates minimize robust reprojection error. The normal path uses
 * an analytic SE(3) Brown-Conrady projection Jacobian; central finite differences remain available
 * as an observable defensive fallback.
 *
 * This is deliberately a calibration-stage optimizer rather than a per-frame tracker. It is much
 * more expensive than the runtime fusion path and should be run on a calibration take.
 */
class BundleAdjustmentEngine(
    private val maxIterations: Int = 12,
    private val huberDeltaPx: Double = 2.5,
    private val initialDamping: Double = 1e-3,
    private val maximumTranslationStepMeters: Double = 0.06,
    private val maximumRotationStepRadians: Double = 0.035,
) {
    init {
        require(maxIterations >= 1)
        require(huberDeltaPx > 0.0)
        require(initialDamping > 0.0)
        require(maximumTranslationStepMeters > 0.0)
        require(maximumRotationStepRadians > 0.0)
    }
    data class Camera(
        val id: String,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val fixed: Boolean = false,
    )

    data class Point(
        val id: Int,
        val position: Vec3,
    )

    data class Observation(
        val cameraId: String,
        val pointId: Int,
        val pixel: Vec2,
        val confidence: Double = 1.0,
    )

    data class Result(
        val cameras: List<Camera>,
        val points: List<Point>,
        val initialRmsPx: Double,
        val finalRmsPx: Double,
        val iterations: Int,
        val converged: Boolean,
        val numericalJacobianFallbackCount: Int = 0,
    )

    data class HumanAwareResult(
        val bundle: Result,
        val motionEvidence: HumanAwareBundleAdjustmentEvidenceEngine.Result,
    )

    fun optimizeHumanAware(
        inputCameras: List<Camera>,
        inputPoints: List<Point>,
        observations: List<Observation>,
        motionEvidence: List<HumanAwareBundleAdjustmentEvidenceEngine.Evidence>,
    ): HumanAwareResult {
        val evidenceEngine = HumanAwareBundleAdjustmentEvidenceEngine()
        val weighted = evidenceEngine.evaluate(
            observations = observations,
            evidence = motionEvidence,
        )
        val weightedObservations = observations.mapIndexed { index, observation ->
            val result = weighted.weights[index]
            observation.copy(confidence = result.weight)
        }
        return HumanAwareResult(
            bundle = optimize(
                inputCameras = inputCameras,
                inputPoints = inputPoints,
                observations = weightedObservations,
            ),
            motionEvidence = weighted,
        )
    }

    fun optimize(
        inputCameras: List<Camera>,
        inputPoints: List<Point>,
        observations: List<Observation>,
    ): Result {
        require(inputCameras.any { it.fixed }) {
            "Bundle adjustment requires a fixed reference camera"
        }
        if (inputCameras.size < 2 || inputPoints.isEmpty() || observations.isEmpty()) {
            val error = rms(inputCameras, inputPoints, observations)
            return Result(
                cameras = inputCameras,
                points = inputPoints,
                initialRmsPx = error,
                finalRmsPx = error,
                iterations = 0,
                converged = true,
                numericalJacobianFallbackCount = 0,
            )
        }

        var cameras = inputCameras.associateBy { it.id }.toMutableMap()
        var points = inputPoints.associateBy { it.id }.toMutableMap()
        var damping = initialDamping
        val initial = rms(cameras.values.toList(), points.values.toList(), observations)
        var current = initial
        var acceptedIterations = 0
        var converged = false
        var numericalJacobianFallbackCount = 0

        repeat(maxIterations) {
            points = retriangulate(
                cameras = cameras.values.toList(),
                seedPoints = points.values.toList(),
                observations = observations,
            ).associateBy { it.id }.toMutableMap()

            val deltas = HashMap<String, DoubleArray>()

            cameras.values.filter { !it.fixed }.forEach { camera ->
                val system = buildNormalEquation(
                    camera = camera,
                    cameras = cameras,
                    points = points,
                    observations = observations,
                ) ?: return@forEach

                numericalJacobianFallbackCount +=
                    system.numericalJacobianFallbackCount

                val delta = solvePositiveDefinite(
                    system.hessian,
                    system.gradient,
                    damping,
                ) ?: return@forEach

                val bounded =
                    boundStep(delta)

                if (bounded.all { abs(it) < 1e-9 }) {
                    return@forEach
                }

                deltas[camera.id] = bounded
            }

            if (deltas.isEmpty()) {
                converged = true
                return@repeat
            }

            val trial = cameras.mapValuesTo(mutableMapOf()) { (_, camera) ->
                val delta = deltas[camera.id]
                if (delta == null) camera else applyDelta(camera, delta)
            }

            val trialError = rms(
                trial.values.toList(),
                points.values.toList(),
                observations,
            )

            if (trialError + 1e-5 < current) {
                cameras = trial
                current = trialError
                damping = max(damping * 0.55, 1e-9)
                acceptedIterations += 1

                if (abs(initial - current) < 1e-4 || trialError < 0.20) {
                    converged = true
                    return@repeat
                }
            } else {
                damping = min(damping * 4.0, 1e6)
            }
        }

        return Result(
            cameras = cameras.values.sortedBy { it.id },
            points = points.values.sortedBy { it.id },
            initialRmsPx = initial,
            finalRmsPx = current,
            iterations = acceptedIterations,
            converged = converged,
            numericalJacobianFallbackCount =
                numericalJacobianFallbackCount,
        )
    }

    private data class NormalEquation(
        val hessian: Array<DoubleArray>,
        val gradient: DoubleArray,
        val numericalJacobianFallbackCount: Int = 0,
    )

    private fun buildNormalEquation(
        camera: Camera,
        cameras: Map<String, Camera>,
        points: Map<Int, Point>,
        observations: List<Observation>,
    ): NormalEquation? {
        val h = Array(6) { DoubleArray(6) }
        val g = DoubleArray(6)
        var numericalJacobianFallbackCount = 0
        val relevant = observations.filter {
            it.cameraId == camera.id && points.containsKey(it.pointId) && it.confidence > 1e-4
        }
        if (relevant.size < 3) return null

        relevant.forEach { observation ->
            val point = points[observation.pointId] ?: return@forEach
            val base = project(point.position, camera) ?: return@forEach
            val residual = doubleArrayOf(
                observation.pixel.x - base.x,
                observation.pixel.y - base.y,
            )

            val residualNorm =
                sqrt(
                    residual[0] * residual[0] +
                        residual[1] * residual[1],
                )
            val robustWeight =
                observation.confidence.coerceIn(0.02, 1.0) *
                    robustWeight(
                        residualNorm,
                    )

            val jacobianResolution =
                analyticPoseJacobian(
                    camera = camera,
                    point = point.position,
                )?.let {
                    PoseJacobianResolution(
                        jacobian = it,
                        usedNumericalFallback = false,
                    )
                } ?: PoseJacobianResolution(
                    jacobian =
                        numericalJacobian(
                            camera,
                            point.position,
                        ),
                    usedNumericalFallback = true,
                )

            if (jacobianResolution.usedNumericalFallback) {
                numericalJacobianFallbackCount += 1
            }

            val jacobian =
                jacobianResolution.jacobian

            for (row in 0..1) {
                for (i in 0 until 6) {
                    g[i] += robustWeight * jacobian[row][i] * residual[row]
                    for (j in 0 until 6) {
                        h[i][j] += robustWeight * jacobian[row][i] * jacobian[row][j]
                    }
                }
            }
        }

        return NormalEquation(
            hessian = h,
            gradient = g,
            numericalJacobianFallbackCount =
                numericalJacobianFallbackCount,
        )
    }

    private data class PoseJacobianResolution(
        val jacobian: Array<DoubleArray>,
        val usedNumericalFallback: Boolean,
    )

    /**
     * Analytic derivative of the calibrated projection with respect to the six-DoF camera update.
     *
     * Translation follows the optimizer's applyDelta convention directly. Rotation is the first
     * order derivative of the left-multiplicative small-angle quaternion update used by applyDelta.
     * The Brown-Conrady distortion derivative is identical to the primary triangulation model.
     */
    private fun analyticPoseJacobian(
        camera: Camera,
        point: Vec3,
    ): Array<DoubleArray>? {
        val inverse =
            camera.pose.rotation
                .conjugate()
                .normalized()

        val cameraPoint =
            inverse.rotate(
                point -
                    camera.pose.position,
            )

        val z = cameraPoint.z
        if (!z.isFinite() || z <= 1e-8) return null

        val x = cameraPoint.x / z
        val y = cameraPoint.y / z
        if (!x.isFinite() || !y.isFinite()) return null

        val r2 = x * x + y * y
        val r4 = r2 * r2
        val r6 = r4 * r2

        val k = camera.intrinsics
        val radial =
            1.0 +
                k.k1 * r2 +
                k.k2 * r4 +
                k.k3 * r6
        if (!radial.isFinite()) return null

        val radialSlope =
            k.k1 +
                2.0 * k.k2 * r2 +
                3.0 * k.k3 * r4

        val dRadialDx =
            2.0 * x * radialSlope
        val dRadialDy =
            2.0 * y * radialSlope

        val dTxDx =
            2.0 * k.p1 * y +
                6.0 * k.p2 * x
        val dTxDy =
            2.0 * k.p1 * x +
                2.0 * k.p2 * y
        val dTyDx =
            2.0 * k.p1 * x +
                2.0 * k.p2 * y
        val dTyDy =
            6.0 * k.p1 * y +
                2.0 * k.p2 * x

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

        val invZ2 =
            1.0 /
                (z * z)

        val cameraBasis =
            arrayOf(
                inverse.rotate(Vec3(1.0, 0.0, 0.0)),
                inverse.rotate(Vec3(0.0, 1.0, 0.0)),
                inverse.rotate(Vec3(0.0, 0.0, 1.0)),
            )

        /*
         * The optimizer updates:
         *   C' = C + dt
         *   R' = Exp(dr) R
         * so the camera-space point derivative is:
         *   dPc/dt = -R^-1 e
         *   dPc/dr = R^-1 (Pc x e)
         */
        val cameraPointRotationBasis =
            arrayOf(
                cameraPoint.cross(Vec3(1.0, 0.0, 0.0)),
                cameraPoint.cross(Vec3(0.0, 1.0, 0.0)),
                cameraPoint.cross(Vec3(0.0, 0.0, 1.0)),
            )

        val derivatives =
            Array(6) { index ->
                if (index < 3) {
                    cameraBasis[index] * -1.0
                } else {
                    inverse.rotate(
                        cameraPointRotationBasis[
                            index - 3
                        ],
                    )
                }
            }

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

        val normalizedDerivatives =
            derivatives.map(::normalizedDerivative)

        val du =
            DoubleArray(6)
        val dv =
            DoubleArray(6)

        for (index in 0 until 6) {
            val dn = normalizedDerivatives[index]
            du[index] =
                k.fx *
                    (
                        dDistortedXDx * dn.x +
                            dDistortedXDy * dn.y
                    )
            dv[index] =
                k.fy *
                    (
                        dDistortedYDx * dn.x +
                            dDistortedYDy * dn.y
                    )
        }

        if (
            du.any { !it.isFinite() } ||
                dv.any { !it.isFinite() }
        ) {
            return null
        }

        return arrayOf(
            du,
            dv,
        )
    }

    private fun numericalJacobian(
        camera: Camera,
        point: Vec3,
    ): Array<DoubleArray> {
        val epsT = 1e-4
        val epsR = 1e-5
        val result =
            Array(2) {
                DoubleArray(6)
            }

        for (axis in 0 until 6) {
            val delta =
                DoubleArray(6)
            delta[axis] =
                if (axis < 3) {
                    epsT
                } else {
                    epsR
                }

            val plus =
                applyDelta(
                    camera,
                    delta,
                )

            val negativeDelta =
                delta.copyOf()
                    .also {
                        it[axis] =
                            -delta[axis]
                    }

            val minus =
                applyDelta(
                    camera,
                    negativeDelta,
                )

            val projectedPlus =
                project(
                    point,
                    plus,
                ) ?: continue
            val projectedMinus =
                project(
                    point,
                    minus,
                ) ?: continue

            val denominator =
                2.0 * delta[axis]

            result[0][axis] =
                (
                    projectedPlus.x -
                        projectedMinus.x
                    ) /
                    denominator
            result[1][axis] =
                (
                    projectedPlus.y -
                        projectedMinus.y
                    ) /
                    denominator
        }

        return result
    }

    private fun solvePositiveDefinite(
        hessian: Array<DoubleArray>,
        gradient: DoubleArray,
        damping: Double,
    ): DoubleArray? {
        val n = 6
        val a = Array(n) { row ->
            DoubleArray(n + 1).also { line ->
                for (col in 0 until n) {
                    line[col] = hessian[row][col]
                }
                line[row] += damping * (abs(line[row]).coerceAtLeast(1.0))
                line[n] = gradient[row]
            }
        }

        for (col in 0 until n) {
            var pivot = col
            for (row in col + 1 until n) {
                if (abs(a[row][col]) > abs(a[pivot][col])) pivot = row
            }
            if (abs(a[pivot][col]) < 1e-10) return null

            val tmp = a[col]
            a[col] = a[pivot]
            a[pivot] = tmp

            val divisor = a[col][col]
            for (c in col until n + 1) a[col][c] /= divisor

            for (row in 0 until n) {
                if (row == col) continue
                val factor = a[row][col]
                if (abs(factor) < 1e-14) continue
                for (c in col until n + 1) {
                    a[row][c] -= factor * a[col][c]
                }
            }
        }

        return DoubleArray(n) { a[it][n] }
    }

    private fun applyDelta(camera: Camera, delta: DoubleArray): Camera {
        require(delta.size == 6)
        val translation = Vec3(delta[0], delta[1], delta[2])
        val rotation = smallAngleQuaternion(
            Vec3(delta[3], delta[4], delta[5]),
        )

        return camera.copy(
            pose = Pose(
                position = camera.pose.position + translation,
                rotation = (rotation * camera.pose.rotation).normalized(),
            ),
        )
    }

    private fun smallAngleQuaternion(v: Vec3): Quat {
        val angle = v.norm()
        if (angle < 1e-12) return Quat.IDENTITY
        val half = angle * 0.5
        val s = kotlin.math.sin(half) / angle
        return Quat(
            kotlin.math.cos(half),
            v.x * s,
            v.y * s,
            v.z * s,
        ).normalized()
    }

    private fun retriangulate(
        cameras: List<Camera>,
        seedPoints: List<Point>,
        observations: List<Observation>,
    ): List<Point> {
        val cameraMap = cameras.associateBy { it.id }
        val pointGroups = observations.groupBy { it.pointId }

        return seedPoints.map { seed ->
            val obs = pointGroups[seed.id].orEmpty()
            val usable = obs.mapNotNull { observation ->
                val camera = cameraMap[observation.cameraId] ?: return@mapNotNull null
                TriangulationEngine.Observation(
                    cameraPose = camera.pose,
                    intrinsics = camera.intrinsics,
                    cameraId = observation.cameraId,
                    u = observation.pixel.x,
                    v = observation.pixel.y,
                    weight = observation.confidence,
                )
            }

            val reconstructed = TriangulationEngine(
                minParallaxDegrees = 0.1,
                huberDeltaPx = huberDeltaPx,
            ).robust(usable)?.point

            seed.copy(position = reconstructed ?: seed.position)
        }
    }

    private fun rms(
        cameras: List<Camera>,
        points: List<Point>,
        observations: List<Observation>,
    ): Double {
        val cameraMap = cameras.associateBy { it.id }
        val pointMap = points.associateBy { it.id }
        val residuals = observations.mapNotNull { observation ->
            val camera = cameraMap[observation.cameraId] ?: return@mapNotNull null
            val point = pointMap[observation.pointId] ?: return@mapNotNull null
            val projected = project(point.position, camera) ?: return@mapNotNull null
            val dx = projected.x - observation.pixel.x
            val dy = projected.y - observation.pixel.y
            sqrt(dx * dx + dy * dy)
        }

        return if (residuals.isEmpty()) Double.POSITIVE_INFINITY
        else sqrt(residuals.map { it * it }.average())
    }

    private fun robustWeight(
        error: Double,
    ): Double {
        if (!error.isFinite() || error < 0.0) return 0.0

        val delta =
            huberDeltaPx.coerceAtLeast(1e-6)
        val normalized =
            error /
                delta

        val huber =
            if (normalized <= 1.0) {
                1.0
            } else {
                1.0 /
                    normalized
            }

        val cauchy =
            1.0 /
                (
                    1.0 +
                        (error /
                            (delta * 1.7)) *
                        (error /
                            (delta * 1.7))
                )

        val tukeyRadius =
            error /
                (delta * 3.5)

        val tukey =
            if (tukeyRadius < 1.0) {
                val oneMinus =
                    1.0 -
                        tukeyRadius *
                        tukeyRadius
                oneMinus *
                    oneMinus
            } else {
                0.01
            }

        return (
            0.36 * huber +
                0.36 * cauchy +
                0.28 * tukey
            ).coerceIn(0.0, 1.0)
    }

    private fun boundStep(
        delta: DoubleArray,
    ): DoubleArray {
        require(delta.size == 6)

        val translationNorm =
            sqrt(
                delta[0] * delta[0] +
                    delta[1] * delta[1] +
                    delta[2] * delta[2],
            )
        val rotationNorm =
            sqrt(
                delta[3] * delta[3] +
                    delta[4] * delta[4] +
                    delta[5] * delta[5],
            )

        val translationScale =
            if (
                translationNorm > maximumTranslationStepMeters
            ) {
                maximumTranslationStepMeters /
                    translationNorm
            } else {
                1.0
            }

        val rotationScale =
            if (
                rotationNorm > maximumRotationStepRadians
            ) {
                maximumRotationStepRadians /
                    rotationNorm
            } else {
                1.0
            }

        val out =
            delta.copyOf()

        if (translationScale < 1.0) {
            for (index in 0 until 3) {
                out[index] *=
                    translationScale
            }
        }

        if (rotationScale < 1.0) {
            for (index in 3 until 6) {
                out[index] *=
                    rotationScale
            }
        }

        return out
    }

    private fun project(point: Vec3, camera: Camera): Vec2? {
        val cameraPoint = camera.pose.rotation.conjugate()
            .normalized()
            .rotate(point - camera.pose.position)

        if (cameraPoint.z <= 1e-8) return null

        var x = cameraPoint.x / cameraPoint.z
        var y = cameraPoint.y / cameraPoint.z
        val r2 = x * x + y * y
        val k = camera.intrinsics
        val radial = 1.0 + k.k1 * r2 + k.k2 * r2 * r2 + k.k3 * r2 * r2 * r2
        val dx = 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x)
        val dy = k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y

        x = x * radial + dx
        y = y * radial + dy

        return Vec2(
            x * k.fx + k.cx,
            y * k.fy + k.cy,
        )
    }
}
