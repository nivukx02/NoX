package com.nivukx.nox.core.model

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

enum class EvidenceOrigin(val authorityRank: Int) {
    OBSERVED(5),
    GEOMETRICALLY_CONSTRAINED(4),
    TEMPORALLY_PROPAGATED(3),
    LEARNED_PRIOR(2),
    GENERATIVE_COMPLETION(1),
    DERIVED(0),
}

data class CaptureTimingEvidence(
    val sensorTimestampNanos: Long,
    val exposureStartNanos: Long,
    val exposureMidpointNanos: Long,
    val exposureEndNanos: Long,
    val readoutDurationNanos: Long,
    val rowTimeNanos: Double,
    val timingUncertaintyNanos: Double,
    val hardwareAuthoritative: Boolean,
    val source: String,
) {
    init {
        require(sensorTimestampNanos >= 0L)
        require(exposureStartNanos >= 0L)
        require(exposureMidpointNanos >= exposureStartNanos)
        require(exposureEndNanos >= exposureMidpointNanos)
        require(readoutDurationNanos >= 0L)
        require(rowTimeNanos.isFinite() && rowTimeNanos >= 0.0)
        require(timingUncertaintyNanos.isFinite() && timingUncertaintyNanos >= 0.0)
        require(source.isNotBlank())
        require(exposureMidpointNanos == midpoint(exposureStartNanos, exposureEndNanos))
    }

    private fun midpoint(start: Long, end: Long): Long =
        start + (end - start) / 2L

    val exposureDurationNanos: Long
        get() = exposureEndNanos - exposureStartNanos

    fun rowTimestamp(rowIndex: Int, rowCount: Int): Double {
        require(rowIndex >= 0)
        require(rowCount > 0)
        val normalizedRow =
            if (rowCount == 1) 0.5 else rowIndex.toDouble() / (rowCount - 1).toDouble()
        return exposureMidpointNanos +
            (normalizedRow - 0.5) * readoutDurationNanos.toDouble()
    }
}

data class UncertaintyEnvelope(
    val xxMeters2: Double,
    val yyMeters2: Double,
    val zzMeters2: Double,
    val xyMeters2: Double = 0.0,
    val xzMeters2: Double = 0.0,
    val yzMeters2: Double = 0.0,
    val confidence: Double = 1.0,
) {
    init {
        val values = doubleArrayOf(xxMeters2, yyMeters2, zzMeters2, xyMeters2, xzMeters2, yzMeters2, confidence)
        require(values.all { it.isFinite() })
        require(xxMeters2 >= 0.0 && yyMeters2 >= 0.0 && zzMeters2 >= 0.0)
        require(confidence in 0.0..1.0)
        require(correlationBound(xyMeters2, xxMeters2, yyMeters2))
        require(correlationBound(xzMeters2, xxMeters2, zzMeters2))
        require(correlationBound(yzMeters2, yyMeters2, zzMeters2))
        require(determinant() >= -1e-12)
    }

    private fun correlationBound(covariance: Double, a: Double, b: Double): Boolean =
        abs(covariance) <= sqrt(max(0.0, a * b)) + 1e-12

    fun determinant(): Double =
        xxMeters2 * (yyMeters2 * zzMeters2 - yzMeters2 * yzMeters2) -
            xyMeters2 * (xyMeters2 * zzMeters2 - yzMeters2 * xzMeters2) +
            xzMeters2 * (xyMeters2 * yzMeters2 - yyMeters2 * xzMeters2)

    fun principalVarianceUpperBound(): Double {
        val rowX = abs(xyMeters2) + abs(xzMeters2)
        val rowY = abs(xyMeters2) + abs(yzMeters2)
        val rowZ = abs(xzMeters2) + abs(yzMeters2)
        return max(xxMeters2 + rowX, max(yyMeters2 + rowY, zzMeters2 + rowZ))
    }
}

data class ObservationProvenance(
    val cameraId: String,
    val calibrationFingerprint: String,
    val timingFingerprint: String,
    val modelFingerprint: String,
    val origin: EvidenceOrigin,
    val captureTimestampNanos: Long,
    val sequenceNumber: Long,
    val freshnessNanos: Long,
) {
    init {
        require(cameraId.isNotBlank())
        require(calibrationFingerprint.isNotBlank())
        require(timingFingerprint.isNotBlank())
        require(modelFingerprint.isNotBlank())
        require(captureTimestampNanos >= 0L)
        require(sequenceNumber >= 0L)
        require(freshnessNanos >= 0L)
    }

    fun compatibilityKey(): String =
        listOf(cameraId, calibrationFingerprint, timingFingerprint, modelFingerprint, origin.name).joinToString("|")
}

data class WorldSpaceProposal(
    val actorId: Int,
    val pointId: Int,
    val timestampNanos: Long,
    val positionMeters: Vec3,
    val velocityMetersPerSecond: Vec3 = Vec3(0.0, 0.0, 0.0),
    val accelerationMetersPerSecond2: Vec3 = Vec3(0.0, 0.0, 0.0),
    val uncertainty: UncertaintyEnvelope,
    val confidence: Double,
    val geometricSupport: Double,
    val temporalSupport: Double,
    val learnedSupport: Double,
    val independentCameraCount: Int,
    val viewDiversity: Double,
    val origin: EvidenceOrigin,
    val provenance: List<ObservationProvenance>,
) {
    init {
        require(actorId >= 0)
        require(pointId >= 0)
        require(timestampNanos >= 0L)
        require(positionMeters.x.isFinite() && positionMeters.y.isFinite() && positionMeters.z.isFinite())
        require(velocityMetersPerSecond.x.isFinite() && velocityMetersPerSecond.y.isFinite() && velocityMetersPerSecond.z.isFinite())
        require(accelerationMetersPerSecond2.x.isFinite() && accelerationMetersPerSecond2.y.isFinite() && accelerationMetersPerSecond2.z.isFinite())
        require(confidence in 0.0..1.0)
        require(geometricSupport in 0.0..1.0)
        require(temporalSupport in 0.0..1.0)
        require(learnedSupport in 0.0..1.0)
        require(independentCameraCount >= 0)
        require(viewDiversity in 0.0..1.0)
        require(provenance.distinctBy { it.cameraId }.size == provenance.size)
        if (origin == EvidenceOrigin.OBSERVED) require(independentCameraCount >= 1)
    }

    val authorityScore: Double
        get() =
            (confidence * 0.36 +
                geometricSupport * 0.30 +
                temporalSupport * 0.16 +
                viewDiversity * 0.12 +
                uncertainty.confidence * 0.06) *
                (origin.authorityRank / 5.0)
}

data class ReconstructionRegionProvenance(
    val actorId: Int,
    val regionId: Int,
    val observedRatio: Double,
    val geometricallyConstrainedRatio: Double,
    val temporallyPropagatedRatio: Double,
    val learnedPriorRatio: Double,
    val generativeCompletionRatio: Double,
    val confidence: Double,
) {
    init {
        require(actorId >= 0)
        require(regionId >= 0)
        val ratios = doubleArrayOf(observedRatio, geometricallyConstrainedRatio, temporallyPropagatedRatio, learnedPriorRatio, generativeCompletionRatio)
        require(ratios.all { it.isFinite() && it in 0.0..1.0 })
        require(confidence in 0.0..1.0)
        require(ratios.sum() <= 1.0 + 1e-9)
    }

    val measuredRatio: Double
        get() = observedRatio + geometricallyConstrainedRatio

    val inferredRatio: Double
        get() = temporallyPropagatedRatio + learnedPriorRatio + generativeCompletionRatio

    val authorityWeightedRatio: Double
        get() = observedRatio +
            geometricallyConstrainedRatio * 0.85 +
            temporallyPropagatedRatio * 0.65 +
            learnedPriorRatio * 0.45 +
            generativeCompletionRatio * 0.25
}

data class PrecisionAuthoritySnapshot(
    val p50ErrorMeters: Double,
    val p95ErrorMeters: Double,
    val p99ErrorMeters: Double,
    val bootstrapHalfWidthMeters: Double,
    val independentViewCount: Int,
    val viewDiversity: Double,
    val reprojectionP95Pixels: Double,
    val timingResidualNanosP95: Double,
    val calibrationResidualPixelsP95: Double,
    val empiricalSampleCount: Int,
    val modeledSubMillimeterFeasible: Boolean,
    val matchedReferenceQualified: Boolean,
    val provenanceFingerprint: String,
) {
    init {
        val values = doubleArrayOf(p50ErrorMeters, p95ErrorMeters, p99ErrorMeters, bootstrapHalfWidthMeters, reprojectionP95Pixels, timingResidualNanosP95, calibrationResidualPixelsP95)
        require(values.all { it.isFinite() && it >= 0.0 })
        require(p50ErrorMeters <= p95ErrorMeters + 1e-12)
        require(p95ErrorMeters <= p99ErrorMeters + 1e-12)
        require(independentViewCount >= 0)
        require(viewDiversity in 0.0..1.0)
        require(empiricalSampleCount >= 0)
        require(provenanceFingerprint.isNotBlank())
    }

    fun physicallyQualified(maximumP95Meters: Double, maximumP99Meters: Double, minimumViews: Int, minimumSamples: Int): Boolean =
        matchedReferenceQualified &&
            empiricalSampleCount >= minimumSamples &&
            independentViewCount >= minimumViews &&
            p95ErrorMeters <= maximumP95Meters &&
            p99ErrorMeters <= maximumP99Meters
}

