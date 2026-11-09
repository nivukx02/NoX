package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.CalibrationSnapshot
import kotlin.math.abs

/**
 * Runtime gate that prevents degraded calibration from silently feeding metric reconstruction.
 *
 * A distributed motion rig is only as good as its weakest geometry link, so quality is measured in
 * terms useful to the reconstruction layer: camera count, clock skew, reprojection residual and
 * minimum baseline/parallax.
 */
class RigQualityGate(
    private val maxReprojectionErrorPx: Double = 4.0,
    private val maxClockSkewNanos: Long = 25_000_000L,
    private val minBaselineMeters: Double = 0.05,
    private val maxBaselineMeters: Double = 20.0,
) {
    data class CameraMetric(
        val id: String,
        val baselineMeters: Double,
        val reprojectionErrorPx: Double,
        val synchronized: Boolean,
    )

    data class Report(
        val accepted: Boolean,
        val score: Double,
        val reasons: List<String>,
        val cameras: Int,
        val worstReprojectionErrorPx: Double,
        val worstClockSkewNanos: Long,
        val baselineRangeMeters: ClosedFloatingPointRange<Double>?,
    )

    fun evaluate(
        calibration: CalibrationSnapshot,
        cameraPositionsMeters: Map<String, com.nivukx.nox.core.model.Vec3>,
        clockSkewNanosByCamera: Map<String, Long>,
    ): Report {
        val reasons = mutableListOf<String>()
        val cameraCount = calibration.extrinsics.size
        if (cameraCount < 2) reasons += "At least two calibrated cameras are required"

        val cameraPoints = cameraPositionsMeters.values.toList()
        val baselines = buildList {
            for (i in 0 until cameraPoints.size) {
                for (j in i + 1 until cameraPoints.size) {
                    val distance = cameraPoints[i].distanceTo(cameraPoints[j])
                    if (distance > 1e-6) add(distance)
                }
            }
        }

        val minBaseline = baselines.minOrNull()
        val maxBaseline = baselines.maxOrNull()

        if (minBaseline != null && minBaseline < minBaselineMeters) {
            reasons += "Some camera baselines are too small for stable depth"
        }
        if (maxBaseline != null && maxBaseline > maxBaselineMeters) {
            reasons += "Rig baseline exceeds configured operating envelope"
        }

        val worstError = calibration.reprojectionErrorPx
        if (worstError > maxReprojectionErrorPx) {
            reasons += "Calibration reprojection error exceeds threshold"
        }

        val worstSkew = clockSkewNanosByCamera.values.maxOrNull() ?: Long.MAX_VALUE
        if (worstSkew > maxClockSkewNanos) {
            reasons += "Clock synchronization is outside the reconstruction envelope"
        }

        val geometryScore = when {
            minBaseline == null -> 0.0
            minBaseline < minBaselineMeters -> 0.35
            else -> 1.0
        }

        val reprojectionScore =
            (1.0 - worstError / maxReprojectionErrorPx).coerceIn(0.0, 1.0)
        val syncScore =
            (1.0 - worstSkew.toDouble() / maxClockSkewNanos).coerceIn(0.0, 1.0)

        val score = (0.45 * geometryScore + 0.35 * reprojectionScore + 0.20 * syncScore)
            .coerceIn(0.0, 1.0)

        return Report(
            accepted = reasons.isEmpty(),
            score = score,
            reasons = reasons,
            cameras = cameraCount,
            worstReprojectionErrorPx = worstError,
            worstClockSkewNanos = worstSkew,
            baselineRangeMeters = if (minBaseline != null && maxBaseline != null) {
                minBaseline..maxBaseline
            } else {
                null
            },
        )
    }
}
