package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.ReconstructionMesh
import com.nivukx.nox.core.model.FaceContourFrame
import kotlin.math.abs

/**
 * Distributed facial geometry buffer and reconstruction bridge.
 *
 * Each phone contributes synchronized ordered contours. A face mesh is generated only after the
 * requested timestamp has at least two calibrated camera views.
 */
class FaceMultiViewFusionEngine(
    private val historyPerDevice: Int = 12,
    private val maxTimestampSkewNanos: Long = 35_000_000L,
    private val maximumDevices: Int = 128,
    private val reconstructor: Face3DReconstructionEngine = Face3DReconstructionEngine(),
) {
    init {
        require(historyPerDevice in 2..256)
        require(maxTimestampSkewNanos > 0L)
        require(maximumDevices in 16..128)
    }
    data class DeviceFrame(
        val deviceId: DeviceId,
        val frame: FaceContourFrame,
    )

    data class Result(
        val mesh: ReconstructionMesh,
        val points: Int,
        val meanReprojectionErrorPx: Double,
        val confidence: Double,
        val pointsBySemanticId: Map<Int, Vec3> = emptyMap(),
        val meanParallaxDegrees: Double = 0.0,
        val meanUncertaintyMeters: Double = 0.0,
        val independentViewCount: Int = 0,
        val precisionQualified: Boolean = false,
        val meanMetricEvidenceAuthority: Double = 0.0,
        val meanEpipolarConsistencyScore: Double = 1.0,
        val epipolarWarnings: List<String> = emptyList(),
    )

    private val history = HashMap<DeviceId, ArrayDeque<FaceContourFrame>>()

    @Synchronized
    fun push(deviceId: DeviceId, frame: FaceContourFrame) {
        val queue = history.getOrPut(deviceId) { ArrayDeque() }
        while (queue.size >= historyPerDevice) queue.removeFirst()
        queue.addLast(frame)
        enforceDeviceCapacity()
    }

    @Synchronized
    fun reconstruct(
        timestampNanos: Long,
        calibration: CalibrationSnapshot,
    ): Result? {
        val frames = history.mapNotNull { (deviceId, queue) ->
            val frame = queue.minByOrNull { abs(it.timestampNanos - timestampNanos) }
                ?: return@mapNotNull null
            if (abs(frame.timestampNanos - timestampNanos) > maxTimestampSkewNanos) {
                return@mapNotNull null
            }
            if (frame.captureTimingConfidence < 0.15) {
                return@mapNotNull null
            }
            if (!calibration.intrinsics.containsKey(deviceId) ||
                !calibration.extrinsics.containsKey(deviceId)
            ) {
                return@mapNotNull null
            }
            Face3DReconstructionEngine.CameraFrame(
                deviceId = deviceId,
                timestampNanos = frame.timestampNanos,
                face = frame,
                captureTimingConfidence =
                    minOf(
                        frame.captureTimingConfidence,
                        calibration.timingAuthorityConfidence
                            .takeIf { it > 0.0 }
                            ?: 1.0,
                    ),
                calibrationUncertainty =
                    calibration.uncertaintyProfiles[deviceId]
                        ?: com.nivukx.nox.core.model.CalibrationUncertaintyProfile(),
            )
        }

        if (frames.size < 2) return null

        val result = reconstructor.reconstruct(
            targetTimestampNanos = timestampNanos,
            frames = frames,
            intrinsics = calibration.intrinsics,
            extrinsics =
                calibration.extrinsics.mapValues { (_, value) -> value.pose },
            calibrationUncertaintyProfiles =
                calibration.uncertaintyProfiles,
        ) ?: return null

        return Result(
            mesh = result.mesh,
            points = result.validPointCount,
            meanReprojectionErrorPx = result.meanReprojectionErrorPx,
            confidence = result.mesh.confidence,
            pointsBySemanticId = result.pointsBySemanticId,
            meanParallaxDegrees = result.meanParallaxDegrees,
            meanUncertaintyMeters = result.meanUncertaintyMeters,
            independentViewCount = result.independentViewCount,
            precisionQualified = result.precisionQualified,
            meanMetricEvidenceAuthority =
                result.meanMetricEvidenceAuthority,
            meanEpipolarConsistencyScore =
                result.meanEpipolarConsistencyScore,
            epipolarWarnings =
                result.epipolarWarnings,
        )
    }

    @Synchronized
    private fun enforceDeviceCapacity() {
        while (history.size > maximumDevices) {
            val victim = history.minByOrNull { (_, queue) ->
                queue.lastOrNull()?.timestampNanos ?: Long.MIN_VALUE
            }?.key ?: return
            history.remove(victim)
        }
    }

    @Synchronized
    fun clear() {
        history.clear()
    }
}
