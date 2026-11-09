package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Robust camera timing model for heterogeneous and rolling-shutter capture.
 *
 * The model estimates:
 *  - camera clock offset against a shared capture timeline;
 *  - clock drift in parts-per-million;
 *  - row-dependent temporal bias caused by rolling-shutter scanout;
 *  - exposure-window midpoint bias;
 *  - robust residual statistics and an explicit evidence confidence.
 *
 * This is evidence plumbing, not a magic synchronizer. A camera is only considered temporally
 * trustworthy when enough monotonic samples exist, the robust residuals are bounded, and the fitted
 * parameters remain physically plausible.
 */
class SpatiotemporalCalibrationEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        // Global distributed-rig identity envelope. Per-solve selection remains bounded downstream.
        val maximumCameras: Int = 128,
        val maximumSamplesPerCamera: Int = 256,
        val minimumSamplesForFit: Int = 6,
        val maximumDriftPartsPerMillion: Double = 2_000.0,
        val maximumRowTimeMilliseconds: Double = 80.0,
        val robustResidualScaleMilliseconds: Double = 1.5,
        val warningResidualP95Milliseconds: Double = 4.0,
        val hardResidualP95Milliseconds: Double = 12.0,
        val hardOutlierRatio: Double = 0.20,
        val confidenceFloor: Double = 0.50,
        val staleToleranceNanos: Long = 0L,
    ) {
        init {
            require(maximumCameras > 0)
            require(maximumSamplesPerCamera >= minimumSamplesForFit)
            require(minimumSamplesForFit >= 3)
            require(maximumDriftPartsPerMillion > 0.0)
            require(maximumRowTimeMilliseconds > 0.0)
            require(robustResidualScaleMilliseconds > 0.0)
            require(warningResidualP95Milliseconds > 0.0)
            require(hardResidualP95Milliseconds >= warningResidualP95Milliseconds)
            require(hardOutlierRatio in 0.0..1.0)
            require(confidenceFloor in 0.0..1.0)
            require(staleToleranceNanos >= 0L)
        }
    }

    data class Observation(
        val cameraId: String,
        val sequence: Long,
        val localTimestampNanos: Long,
        val referenceTimestampNanos: Long,
        val normalizedRow: Double = 0.5,
        val exposureDurationNanos: Long = 0L,
        val valid: Boolean = true,
    ) {
        init {
            require(cameraId.isNotBlank())
            require(sequence >= 0L)
            require(localTimestampNanos >= 0L)
            require(referenceTimestampNanos >= 0L)
            require(normalizedRow.isFinite() && normalizedRow in 0.0..1.0)
            require(exposureDurationNanos >= 0L)
        }
    }

    data class TimestampCorrection(
        val localTimestampNanos: Long,
        val correctedReferenceTimestampNanos: Long,
        val correctionMilliseconds: Double,
        val residualUncertaintyMilliseconds: Double,
        val rowOffsetMilliseconds: Double,
        val exposureMidpointOffsetMilliseconds: Double,
        val confidence: Double,
    )

    data class Result(
        val cameraId: String,
        val sampleCount: Int,
        val clockOffsetMilliseconds: Double,
        val clockDriftPartsPerMillion: Double,
        val rowReadoutMilliseconds: Double,
        val exposureMidpointBiasMilliseconds: Double,
        val residualP50Milliseconds: Double,
        val residualP95Milliseconds: Double,
        val residualP99Milliseconds: Double,
        val outlierRatio: Double,
        val temporalConfidence: Double,
        val monotonicInput: Boolean,
        val staleEvidenceRejected: Int,
        val physicallyPlausible: Boolean,
        val valid: Boolean,
        val warningCodes: List<String>,
        private val clockOffsetMillisecondsModel: Double = clockOffsetMilliseconds,
        private val clockDriftPartsPerMillionModel: Double =
            clockDriftPartsPerMillion,
        private val modelOriginTimestampNanos: Long = 0L,
    ) {
        fun correctTimestamp(
            localTimestampNanos: Long,
            normalizedRow: Double = 0.5,
            exposureDurationNanos: Long = 0L,
        ): TimestampCorrection {
            require(localTimestampNanos >= 0L)
            require(normalizedRow.isFinite() && normalizedRow in 0.0..1.0)
            require(exposureDurationNanos >= 0L)

            val elapsedSeconds =
                (
                    localTimestampNanos -
                        modelOriginTimestampNanos
                    ).toDouble() / 1_000_000_000.0

            val rowOffset =
                rowReadoutMilliseconds *
                    normalizedRow
            val exposureOffset =
                exposureMidpointBiasMilliseconds *
                    (
                        (
                            exposureDurationNanos / 1_000_000_000.0
                        ) / 0.020
                    ).coerceIn(0.0, 2.0) -
                    exposureMidpointBiasMilliseconds

            val correction =
                clockOffsetMillisecondsModel +
                    clockDriftPartsPerMillionModel *
                    elapsedSeconds / 1_000_000.0 +
                    rowOffset +
                    exposureOffset

            return TimestampCorrection(
                localTimestampNanos = localTimestampNanos,
                correctedReferenceTimestampNanos =
                    (
                        localTimestampNanos +
                            correction * 1_000_000.0
                        ).toLong().coerceAtLeast(0L),
                correctionMilliseconds = correction,
                residualUncertaintyMilliseconds =
                    residualP95Milliseconds.coerceAtLeast(0.0),
                rowOffsetMilliseconds = rowOffset,
                exposureMidpointOffsetMilliseconds =
                    exposureOffset,
                confidence = subFrameConfidence,
            )
        }

        fun predictedResidualMilliseconds(
            localTimestampNanos: Long,
            normalizedRow: Double = 0.5,
            exposureDurationNanos: Long = 0L,
        ): Double =
            correctTimestamp(
                localTimestampNanos = localTimestampNanos,
                normalizedRow = normalizedRow,
                exposureDurationNanos = exposureDurationNanos,
            ).residualUncertaintyMilliseconds

        val hardFailure: Boolean
            get() =
                !valid ||
                    residualP95Milliseconds > 12.0 ||
                    outlierRatio > 0.20

        val subFrameConfidence: Double
            get() = (
                temporalConfidence *
                    (1.0 - min(1.0, residualP95Milliseconds / 12.0) * 0.50)
                ).coerceIn(0.0, 1.0)
    }

    private data class Sample(
        val sequence: Long,
        val localSeconds: Double,
        val referenceSeconds: Double,
        val centeredSeconds: Double,
        val row: Double,
        val exposureSeconds: Double,
    )

    private data class CameraState(
        val samples: ArrayDeque<Sample>,
        var lastSequence: Long = Long.MIN_VALUE,
        var lastReferenceTimestampNanos: Long = Long.MIN_VALUE,
        var lastLocalTimestampNanos: Long = Long.MIN_VALUE,
        var staleEvidenceRejected: Int = 0,
    )

    private val states = LinkedHashMap<String, CameraState>()

    @Synchronized
    fun add(observation: Observation): Boolean {
        if (!observation.valid) return false

        val state = states.getOrPut(observation.cameraId) {
            if (states.size >= config.maximumCameras) {
                val eldest = states.entries.firstOrNull()?.key
                if (eldest != null) states.remove(eldest)
            }
            CameraState(ArrayDeque(config.maximumSamplesPerCamera))
        }

        val referenceTimestamp = observation.referenceTimestampNanos
        val localTimestamp = observation.localTimestampNanos

        val stale =
            observation.sequence <= state.lastSequence ||
                (
                    state.lastReferenceTimestampNanos != Long.MIN_VALUE &&
                        referenceTimestamp +
                        config.staleToleranceNanos <
                        state.lastReferenceTimestampNanos
                    ) ||
                (
                    state.lastLocalTimestampNanos != Long.MIN_VALUE &&
                        localTimestamp +
                        config.staleToleranceNanos <
                        state.lastLocalTimestampNanos
                    )

        if (stale) {
            state.staleEvidenceRejected += 1
            return false
        }

        val localSeconds = localTimestamp / 1_000_000_000.0
        val referenceSeconds = referenceTimestamp / 1_000_000_000.0
        val exposureSeconds = observation.exposureDurationNanos / 1_000_000_000.0

        state.samples.addLast(
            Sample(
                sequence = observation.sequence,
                localSeconds = localSeconds,
                referenceSeconds = referenceSeconds,
                centeredSeconds = 0.0,
                row = observation.normalizedRow,
                exposureSeconds = exposureSeconds,
            ),
        )
        while (state.samples.size > config.maximumSamplesPerCamera) {
            state.samples.removeFirst()
        }

        state.lastSequence = observation.sequence
        state.lastReferenceTimestampNanos = referenceTimestamp
        state.lastLocalTimestampNanos = localTimestamp
        return true
    }

    @Synchronized
    fun fit(cameraId: String): Result {
        val state = states[cameraId]
            ?: return blocked(cameraId, "camera-state-missing")

        if (state.samples.size < config.minimumSamplesForFit) {
            return blocked(
                cameraId,
                "insufficient-temporal-samples",
                state.staleEvidenceRejected,
            )
        }

        val raw = state.samples.toList()
        val localOrigin = raw.map { it.localSeconds }.average()
        val samples = raw.map {
            it.copy(
                centeredSeconds = it.localSeconds - localOrigin,
            )
        }

        val baseFit = robustFit(samples, initialWeights(samples.size))

        val residuals = samples.map { sample ->
            val prediction =
                baseFit.offsetMilliseconds / 1_000.0 +
                    (baseFit.driftPartsPerMillion / 1_000_000.0) *
                    sample.centeredSeconds +
                    baseFit.rowReadoutMilliseconds / 1_000.0 *
                    sample.row +
                    baseFit.exposureBiasMilliseconds / 1_000.0 *
                    normalizedExposure(sample.exposureSeconds)

            (sample.referenceSeconds - sample.localSeconds - prediction) * 1_000.0
        }

        val scale = robustScale(residuals)
            .coerceAtLeast(config.robustResidualScaleMilliseconds)
        val finalWeights = residuals.map { residual ->
            val normalized = abs(residual) / scale
            if (normalized <= 1.0) {
                1.0
            } else {
                1.0 / normalized
            }
        }

        val refinedFit = robustFit(samples, finalWeights)
        val refinedResiduals = samples.map { sample ->
            val prediction =
                refinedFit.offsetMilliseconds / 1_000.0 +
                    (refinedFit.driftPartsPerMillion / 1_000_000.0) *
                    sample.centeredSeconds +
                    refinedFit.rowReadoutMilliseconds / 1_000.0 *
                    sample.row +
                    refinedFit.exposureBiasMilliseconds / 1_000.0 *
                    normalizedExposure(sample.exposureSeconds)

            (sample.referenceSeconds - sample.localSeconds - prediction) * 1_000.0
        }

        val absResiduals = refinedResiduals.map(::abs).sorted()
        val p50 = quantile(absResiduals, 0.50)
        val p95 = quantile(absResiduals, 0.95)
        val p99 = quantile(absResiduals, 0.99)

        val outlierThreshold = max(
            config.hardResidualP95Milliseconds / 2.0,
            scale * 3.0,
        )
        val outliers = refinedResiduals.count {
            abs(it) > outlierThreshold
        }
        val outlierRatio = outliers.toDouble() / refinedResiduals.size

        val physicallyPlausible =
            abs(refinedFit.driftPartsPerMillion) <=
                config.maximumDriftPartsPerMillion &&
                refinedFit.rowReadoutMilliseconds in
                -config.maximumRowTimeMilliseconds..
                config.maximumRowTimeMilliseconds &&
                refinedFit.exposureBiasMilliseconds.isFinite()

        val residualQuality = (
            1.0 -
                p95 / config.hardResidualP95Milliseconds
            ).coerceIn(0.0, 1.0)
        val outlierQuality = (1.0 - outlierRatio).coerceIn(0.0, 1.0)
        val parameterQuality = if (physicallyPlausible) 1.0 else 0.0
        val sampleQuality = (
            raw.size.toDouble() /
                (config.minimumSamplesForFit * 4.0)
        ).coerceIn(0.0, 1.0)

        val confidence = (
            residualQuality * 0.50 +
                outlierQuality * 0.20 +
                parameterQuality * 0.20 +
                sampleQuality * 0.10
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (p95 > config.warningResidualP95Milliseconds) {
                add("temporal-residual-p95-elevated")
            }
            if (p95 > config.hardResidualP95Milliseconds) {
                add("temporal-residual-p95-hard-failure")
            }
            if (outlierRatio > config.hardOutlierRatio) {
                add("temporal-outlier-ratio-hard-failure")
            } else if (outlierRatio > 0.08) {
                add("temporal-outlier-ratio-elevated")
            }
            if (abs(refinedFit.driftPartsPerMillion) > 500.0) {
                add("clock-drift-elevated")
            }
            if (abs(refinedFit.rowReadoutMilliseconds) > 20.0) {
                add("rolling-shutter-readout-elevated")
            }
            if (!physicallyPlausible) {
                add("temporal-model-physically-implausible")
            }
            if (state.staleEvidenceRejected > 0) {
                add("stale-temporal-evidence-rejected")
            }
        }

        val valid =
            monotonic(samples) &&
                physicallyPlausible &&
                p95 <= config.hardResidualP95Milliseconds &&
                outlierRatio <= config.hardOutlierRatio &&
                confidence >= config.confidenceFloor

        return Result(
            cameraId = cameraId,
            sampleCount = samples.size,
            clockOffsetMilliseconds = refinedFit.offsetMilliseconds,
            clockDriftPartsPerMillion = refinedFit.driftPartsPerMillion,
            rowReadoutMilliseconds = refinedFit.rowReadoutMilliseconds,
            exposureMidpointBiasMilliseconds =
                refinedFit.exposureBiasMilliseconds,
            residualP50Milliseconds = p50,
            residualP95Milliseconds = p95,
            residualP99Milliseconds = p99,
            outlierRatio = outlierRatio,
            temporalConfidence = confidence,
            monotonicInput = monotonic(samples),
            staleEvidenceRejected = state.staleEvidenceRejected,
            physicallyPlausible = physicallyPlausible,
            valid = valid,
            warningCodes = warnings.distinct(),
            modelOriginTimestampNanos =
                (localOrigin * 1_000_000_000.0)
                    .toLong()
                    .coerceAtLeast(0L),
        )
    }

    @Synchronized
    fun fitAll(): List<Result> =
        states.keys.toList().map(::fit)

    @Synchronized
    fun reset(cameraId: String? = null) {
        if (cameraId == null) {
            states.clear()
        } else {
            states.remove(cameraId)
        }
    }

    @Synchronized
    override fun close() {
        states.clear()
    }

    private data class Fit(
        val offsetMilliseconds: Double,
        val driftPartsPerMillion: Double,
        val rowReadoutMilliseconds: Double,
        val exposureBiasMilliseconds: Double,
    )

    private fun robustFit(
        samples: List<Sample>,
        weights: List<Double>,
    ): Fit {
        require(samples.isNotEmpty())
        require(samples.size == weights.size)

        val values =
            samples.map { sample ->
                doubleArrayOf(
                    1.0,
                    sample.centeredSeconds,
                    sample.row,
                    normalizedExposure(sample.exposureSeconds),
                )
            }

        val activeColumns =
            (1..3).filter { column ->
                val columnValues = values.map { it[column] }
                val minimum = columnValues.minOrNull() ?: 0.0
                val maximum = columnValues.maxOrNull() ?: 0.0
                maximum - minimum > 1e-12
            }

        val columns = listOf(0) + activeColumns
        val dimension = columns.size
        val normal = Array(dimension) { DoubleArray(dimension + 1) }

        samples.forEachIndexed { index, sample ->
            val x = values[index]
            val weight = weights[index].coerceIn(0.001, 1.0)
            val target =
                sample.referenceSeconds -
                    sample.localSeconds

            for (r in 0 until dimension) {
                val xr = x[columns[r]]
                for (c in 0 until dimension) {
                    normal[r][c] +=
                        weight *
                            xr *
                            x[columns[c]]
                }
                normal[r][dimension] +=
                    weight *
                        xr *
                        target
            }
        }

        val solution = solveLinearSystem(normal)
            ?: return Fit(0.0, 0.0, 0.0, 0.0)

        val full = DoubleArray(4)
        columns.forEachIndexed { index, column ->
            full[column] = solution[index]
        }

        return Fit(
            offsetMilliseconds = full[0] * 1_000.0,
            driftPartsPerMillion = full[1] * 1_000_000.0,
            rowReadoutMilliseconds = full[2] * 1_000.0,
            exposureBiasMilliseconds = full[3] * 1_000.0,
        )
    }

    private fun solveLinearSystem(
        matrix: Array<DoubleArray>,
    ): DoubleArray? {
        val dimension = matrix.size
        if (dimension == 0) return null
        val a =
            Array(dimension) { row ->
                DoubleArray(dimension + 1) { column ->
                    matrix[row][column]
                }
            }

        for (pivot in 0 until dimension) {
            var best = pivot
            for (candidate in pivot + 1 until dimension) {
                if (
                    kotlin.math.abs(a[candidate][pivot]) >
                    kotlin.math.abs(a[best][pivot])
                ) {
                    best = candidate
                }
            }

            if (kotlin.math.abs(a[best][pivot]) < 1e-12) {
                return null
            }

            if (best != pivot) {
                val temp = a[pivot]
                a[pivot] = a[best]
                a[best] = temp
            }

            val divisor = a[pivot][pivot]
            for (column in pivot until dimension + 1) {
                a[pivot][column] /= divisor
            }

            for (row in 0 until dimension) {
                if (row == pivot) continue
                val factor = a[row][pivot]
                if (kotlin.math.abs(factor) < 1e-18) continue
                for (column in pivot until dimension + 1) {
                    a[row][column] -=
                        factor *
                            a[pivot][column]
                }
            }
        }

        return DoubleArray(dimension) { a[it][dimension] }
    }

    private fun initialWeights(size: Int): List<Double> =
        List(size) { 1.0 }

    private fun robustScale(values: List<Double>): Double {
        if (values.isEmpty()) return config.robustResidualScaleMilliseconds
        val median = quantile(values.map(::abs).sorted(), 0.50)
        return median * 1.4826
    }

    private fun quantile(
        sortedValues: List<Double>,
        probability: Double,
    ): Double {
        if (sortedValues.isEmpty()) return Double.POSITIVE_INFINITY
        val position =
            ceil(probability.coerceIn(0.0, 1.0) * sortedValues.size)
                .toInt()
                .coerceAtLeast(1) - 1
        return sortedValues[position.coerceIn(0, sortedValues.lastIndex)]
    }

    private fun normalizedExposure(exposureSeconds: Double): Double =
        if (exposureSeconds <= 0.0) {
            0.0
        } else {
            (exposureSeconds / 0.020).coerceIn(0.0, 2.0) - 1.0
        }

    private fun monotonic(samples: List<Sample>): Boolean =
        samples.zipWithNext().all { (first, second) ->
            second.sequence > first.sequence &&
                second.localSeconds >= first.localSeconds &&
                second.referenceSeconds >= first.referenceSeconds
        }

    private fun blocked(
        cameraId: String,
        reason: String,
        staleEvidenceRejected: Int = 0,
    ): Result =
        Result(
            cameraId = cameraId,
            sampleCount = 0,
            clockOffsetMilliseconds = 0.0,
            clockDriftPartsPerMillion = 0.0,
            rowReadoutMilliseconds = 0.0,
            exposureMidpointBiasMilliseconds = 0.0,
            residualP50Milliseconds = Double.POSITIVE_INFINITY,
            residualP95Milliseconds = Double.POSITIVE_INFINITY,
            residualP99Milliseconds = Double.POSITIVE_INFINITY,
            outlierRatio = 1.0,
            temporalConfidence = 0.0,
            monotonicInput = false,
            staleEvidenceRejected = staleEvidenceRejected,
            physicallyPlausible = false,
            valid = false,
            warningCodes = listOf(reason),
        )
}
