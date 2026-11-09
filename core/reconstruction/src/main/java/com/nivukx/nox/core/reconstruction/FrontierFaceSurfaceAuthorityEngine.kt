package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.CalibrationSnapshot
import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.FaceMeshFrame
import com.nivukx.nox.core.model.PrecisionAuthoritySnapshot

/**
 * Frontier authority wrapper around the existing metric face mesh solver.
 * It adds explicit evidence sufficiency and qualification boundaries without changing the underlying solver.
 */
class FrontierFaceSurfaceAuthorityEngine(
    private val fusion: FaceMeshMultiViewFusionEngine = FaceMeshMultiViewFusionEngine(),
    private val minimumIndependentDevices: Int = 3,
    private val minimumAcceptedPointRatio: Double = 0.72,
    private val maximumTailDepthErrorMillimeters: Double = 1.0,
) : AutoCloseable {

    data class Result(
        val metric: FaceMeshMultiViewFusionEngine.Result?,
        val precisionAuthority: PrecisionAuthoritySnapshot,
        val authoritative: Boolean,
        val provenanceWarnings: List<String>,
    )

    @Synchronized
    fun push(deviceId: DeviceId, frame: FaceMeshFrame) = fusion.push(deviceId, frame)

    @Synchronized
    fun reconstruct(timestampNanos: Long, calibration: CalibrationSnapshot): Result {
        val metric = fusion.reconstruct(timestampNanos, calibration)
        if (metric == null) return unavailableResult()

        val sourceDevices = metric.sourceDevices.distinct()
        val tailDepth = metric.p95ReprojectionTailAdjustedDepthErrorMillimeters
        val tailPass = tailDepth.isFinite() && tailDepth <= maximumTailDepthErrorMillimeters
        val sufficientViews = sourceDevices.size >= minimumIndependentDevices
        val acceptedRatioPass = metric.acceptedPointRatio >= minimumAcceptedPointRatio
        val temporalPass = metric.meanTemporalStability >= 0.55
        val modeledFeasible =
            metric.meanSubMillimeterFeasibilityScore >= 0.50 && tailPass && temporalPass
        val authority = sufficientViews && acceptedRatioPass && temporalPass &&
            metric.confidence >= 0.70 && tailPass
        val snapshot = PrecisionAuthoritySnapshot(
            p50ErrorMeters = metric.meanUncertaintyMeters.coerceAtLeast(0.0),
            p95ErrorMeters = (metric.meanUncertaintyMeters * 1.50).coerceAtLeast(0.0),
            p99ErrorMeters = (metric.meanUncertaintyMeters * 2.25).coerceAtLeast(0.0),
            bootstrapHalfWidthMeters = (metric.meanUncertaintyMeters * 0.25).coerceAtLeast(0.0),
            independentViewCount = sourceDevices.size,
            viewDiversity = metric.points.map { point ->
                (point.sourceCount.toDouble() /
                    sourceDevices.size.coerceAtLeast(1).toDouble())
            }.average().coerceIn(0.0,1.0),
            reprojectionP95Pixels = metric.p95ReprojectionErrorPx.coerceAtLeast(0.0),
            timingResidualNanosP95 = 0.0,
            calibrationResidualPixelsP95 = metric.meanReprojectionErrorPx.coerceAtLeast(0.0),
            empiricalSampleCount = metric.points.size,
            modeledSubMillimeterFeasible = modeledFeasible,
            matchedReferenceQualified = false,
            provenanceFingerprint = "face-${metric.timestampNanos}-${sourceDevices.sortedBy { it.value }.joinToString(",") { it.value }}-${metric.points.size}",
        )
        return Result(metric, snapshot, authority, buildList {
            if (!sufficientViews) add("face-independent-view-evidence-insufficient")
            if (!acceptedRatioPass) add("face-accepted-point-ratio-below-authority")
            if (!tailPass) add("face-tail-depth-error-above-qualification-limit")
            if (!temporalPass) add("face-temporal-stability-below-authority")
        })
    }

    private fun unavailableResult(): Result = Result(
        metric = null,
        precisionAuthority = PrecisionAuthoritySnapshot(
            p50ErrorMeters = 1.0e6,
            p95ErrorMeters = 1.0e6,
            p99ErrorMeters = 1.0e6,
            bootstrapHalfWidthMeters = 1.0e6,
            independentViewCount = 0,
            viewDiversity = 0.0,
            reprojectionP95Pixels = 1.0e6,
            timingResidualNanosP95 = 1.0e12,
            calibrationResidualPixelsP95 = 1.0e6,
            empiricalSampleCount = 0,
            modeledSubMillimeterFeasible = false,
            matchedReferenceQualified = false,
            provenanceFingerprint = "face-no-metric-result",
        ),
        authoritative = false,
        provenanceWarnings = listOf("face-metric-reconstruction-unavailable"),
    )

    override fun close() = fusion.close()
}