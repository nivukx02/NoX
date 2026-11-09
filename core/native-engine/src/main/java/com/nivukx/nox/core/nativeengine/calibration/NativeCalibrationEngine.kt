package com.nivukx.nox.core.nativeengine.calibration

import com.nivukx.nox.core.model.CameraIntrinsics
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Quat
import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3

/**
 * Stateless JNI boundary for the native high-end calibration solver.
 *
 * The bridge is deliberately bounded to the distributed camera envelope and a finite observation
 * budget. Native calibration is a calibration-stage operation, not a per-frame hot-path operation.
 */
class NativeCalibrationEngine {

    data class Observation(
        val worldPoint: Vec3,
        val pixel: Vec2,
        val weight: Double = 1.0,
    ) {
        init {
            require(worldPoint.x.isFinite())
            require(worldPoint.y.isFinite())
            require(worldPoint.z.isFinite())
            require(pixel.x.isFinite())
            require(pixel.y.isFinite())
            require(weight.isFinite() && weight >= 0.0)
        }
    }

    data class CameraInput(
        val cameraId: String,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val fixed: Boolean = false,
        val observations: List<Observation>,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(intrinsics.fx > 0.0 && intrinsics.fx.isFinite())
            require(intrinsics.fy > 0.0 && intrinsics.fy.isFinite())
            require(intrinsics.imageWidth > 0)
            require(intrinsics.imageHeight > 0)
            require(observations.isNotEmpty())
        }
    }

    data class CameraResult(
        val cameraId: String,
        val intrinsics: CameraIntrinsics,
        val pose: Pose,
        val reprojectionRmsPx: Double,
        val conditioning: Double,
        val confidence: Double,
        val initialReprojectionRmsPx: Double,
        val intrinsicRefinementImprovementPx: Double,
        val imageCoverage: Double,
        val depthCoverage: Double,
        val observationCoverage: Double,
        val intrinsicsRefined: Boolean,
        val intrinsicJacobianFallbackCount: Int,
        val translationJacobianFallbackCount: Int,
        val valid: Boolean,
    )

    data class Result(
        val cameras: List<CameraResult>,
        val residualRmsPx: Double,
        val confidence: Double,
        val imageCoverage: Double,
        val depthCoverage: Double,
        val observationCoverage: Double,
        val valid: Boolean,
    )

    fun solve(
        cameras: List<CameraInput>,
        config: NativeCalibrationConfig = NativeCalibrationConfig(),
    ): Result {
        check(isAvailable()) {
            "Native calibration library is unavailable"
        }
        require(cameras.size in 2..MAX_CAMERAS)
        require(cameras.count { it.fixed } == 1) {
            "Native calibration requires exactly one fixed reference camera"
        }

        val observationCount = cameras.sumOf { it.observations.size }
        require(observationCount in 1..MAX_OBSERVATIONS)

        val ids = cameras.map { it.cameraId }.toTypedArray()
        require(ids.toSet().size == ids.size) {
            "Camera IDs must be unique"
        }

        val intrinsics = DoubleArray(cameras.size * INTRINSIC_STRIDE)
        val poses = DoubleArray(cameras.size * POSE_STRIDE)
        val fixed = BooleanArray(cameras.size)

        cameras.forEachIndexed { index, camera ->
            val i = index * INTRINSIC_STRIDE
            intrinsics[i] = camera.intrinsics.fx
            intrinsics[i + 1] = camera.intrinsics.fy
            intrinsics[i + 2] = camera.intrinsics.cx
            intrinsics[i + 3] = camera.intrinsics.cy
            intrinsics[i + 4] = camera.intrinsics.k1
            intrinsics[i + 5] = camera.intrinsics.k2
            intrinsics[i + 6] = camera.intrinsics.p1
            intrinsics[i + 7] = camera.intrinsics.p2
            intrinsics[i + 8] = camera.intrinsics.k3
            intrinsics[i + 9] = camera.intrinsics.imageWidth.toDouble()
            intrinsics[i + 10] = camera.intrinsics.imageHeight.toDouble()

            val p = index * POSE_STRIDE
            val q = camera.pose.rotation.normalized()
            poses[p] = q.w
            poses[p + 1] = q.x
            poses[p + 2] = q.y
            poses[p + 3] = q.z
            poses[p + 4] = camera.pose.position.x
            poses[p + 5] = camera.pose.position.y
            poses[p + 6] = camera.pose.position.z
            fixed[index] = camera.fixed
        }

        val observationCameraIndices = IntArray(observationCount)
        val observationData = DoubleArray(observationCount * OBSERVATION_STRIDE)
        var observationIndex = 0

        cameras.forEachIndexed { cameraIndex, camera ->
            camera.observations.forEach { observation ->
                observationCameraIndices[observationIndex] = cameraIndex
                val o = observationIndex * OBSERVATION_STRIDE
                observationData[o] = observation.worldPoint.x
                observationData[o + 1] = observation.worldPoint.y
                observationData[o + 2] = observation.worldPoint.z
                observationData[o + 3] = observation.pixel.x
                observationData[o + 4] = observation.pixel.y
                observationData[o + 5] = observation.weight
                observationIndex += 1
            }
        }

        val output = nativeSolve(
            ids = ids,
            intrinsics = intrinsics,
            poses = poses,
            fixed = fixed,
            observationCameraIndices = observationCameraIndices,
            observationData = observationData,
            huberDeltaPx = config.huberDeltaPx,
            maxIterations = config.maxIterations,
            minimumObservationsPerCamera = config.minimumObservationsPerCamera,
            refineIntrinsics = config.refineIntrinsics,
            refineDistortion = config.refineDistortion,
            focalPriorSigmaRatio = config.focalPriorSigmaRatio,
            principalPointPriorSigmaRatio = config.principalPointPriorSigmaRatio,
            distortionPriorSigma = config.distortionPriorSigma,
            refinementDamping = config.refinementDamping,
            refinementIterations = config.refinementIterations,
        ) ?: error("Native calibration returned no result")

        return parse(output, cameras)
    }

    private fun parse(output: DoubleArray, inputs: List<CameraInput>): Result {
        val expected = GLOBAL_STRIDE + inputs.size * CAMERA_STRIDE
        require(output.size == expected) {
            "Unexpected native calibration output size=" + output.size + ", expected=" + expected
        }

        var offset = 0
        val residualRms = output[offset++]
        val confidence = output[offset++]
        val imageCoverage = output[offset++]
        val depthCoverage = output[offset++]
        val observationCoverage = output[offset++]
        val valid = output[offset++] >= 0.5

        val cameras = inputs.mapIndexed { index, input ->
            val resultOffset = offset + index * CAMERA_STRIDE
            CameraResult(
                cameraId = input.cameraId,
                intrinsics = CameraIntrinsics(
                    fx = output[resultOffset + 12],
                    fy = output[resultOffset + 13],
                    cx = output[resultOffset + 14],
                    cy = output[resultOffset + 15],
                    k1 = output[resultOffset + 16],
                    k2 = output[resultOffset + 17],
                    p1 = output[resultOffset + 18],
                    p2 = output[resultOffset + 19],
                    k3 = output[resultOffset + 20],
                    imageWidth = input.intrinsics.imageWidth,
                    imageHeight = input.intrinsics.imageHeight,
                ),
                pose = Pose(
                    position = Vec3(
                        output[resultOffset + 25],
                        output[resultOffset + 26],
                        output[resultOffset + 27],
                    ),
                    rotation = Quat(
                        output[resultOffset + 21],
                        output[resultOffset + 22],
                        output[resultOffset + 23],
                        output[resultOffset + 24],
                    ).normalized(),
                ),
                reprojectionRmsPx = output[resultOffset],
                conditioning = output[resultOffset + 1],
                confidence = output[resultOffset + 2],
                initialReprojectionRmsPx = output[resultOffset + 3],
                intrinsicRefinementImprovementPx = output[resultOffset + 4],
                imageCoverage = output[resultOffset + 5],
                depthCoverage = output[resultOffset + 6],
                observationCoverage = output[resultOffset + 7],
                intrinsicsRefined = output[resultOffset + 8] >= 0.5,
                intrinsicJacobianFallbackCount = output[resultOffset + 9].toInt().coerceAtLeast(0),
                translationJacobianFallbackCount = output[resultOffset + 10].toInt().coerceAtLeast(0),
                valid = output[resultOffset + 11] >= 0.5,
            )
        }

        return Result(
            cameras = cameras,
            residualRmsPx = residualRms,
            confidence = confidence,
            imageCoverage = imageCoverage,
            depthCoverage = depthCoverage,
            observationCoverage = observationCoverage,
            valid = valid,
        )
    }

    companion object {
        private const val MAX_CAMERAS = 128
        private const val MAX_OBSERVATIONS = 32_768
        private const val INTRINSIC_STRIDE = 11
        private const val POSE_STRIDE = 7
        private const val OBSERVATION_STRIDE = 6
        private const val GLOBAL_STRIDE = 6
        private const val CAMERA_STRIDE = 28

        private val nativeAvailable: Boolean =
            runCatching {
                System.loadLibrary("nox_native_engine")
            }.isSuccess

        fun isAvailable(): Boolean =
            nativeAvailable

        private external fun nativeSolve(
            ids: Array<String>,
            intrinsics: DoubleArray,
            poses: DoubleArray,
            fixed: BooleanArray,
            observationCameraIndices: IntArray,
            observationData: DoubleArray,
            huberDeltaPx: Double,
            maxIterations: Int,
            minimumObservationsPerCamera: Int,
            refineIntrinsics: Boolean,
            refineDistortion: Boolean,
            focalPriorSigmaRatio: Double,
            principalPointPriorSigmaRatio: Double,
            distortionPriorSigma: Double,
            refinementDamping: Double,
            refinementIterations: Int,
        ): DoubleArray?
    }
}