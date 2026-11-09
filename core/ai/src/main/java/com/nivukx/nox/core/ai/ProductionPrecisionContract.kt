package com.nivukx.nox.core.ai

import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Evidence-driven production precision contract for NoX.
 *
 * The contract target is deliberately stricter than the normal mobile profile:
 * P95 < 2 mm under a controlled, matched benchmark cohort.
 *
 * This is an acceptance target, not a claim that every phone, subject, speed, distance,
 * lighting condition or camera topology will achieve it. A run is only reference-qualified
 * when its uncertainty budget, synchronization, calibration and benchmark conditions pass.
 */
class ProductionPrecisionContract(
    private val maximumSamples: Int = 16384,
    private val maximumFailureEvents: Int = 4096,
) : AutoCloseable {

    enum class Region {
        ROOT,
        SPINE,
        HEAD,
        FACE,
        EYES,
        LEFT_ARM,
        RIGHT_ARM,
        LEFT_HAND,
        RIGHT_HAND,
        LEFT_LEG,
        RIGHT_LEG,
        FEET,
        SURFACE,
        FULL_BODY,
    }

    enum class ReferenceSystem {
        VICON,
        OTHER_MARKER_OPTICAL,
        ROBOTIC_GROUND_TRUTH,
        HIGH_PRECISION_CALIBRATION_RIG,
        NONE,
    }

    enum class FailureKind {
        CALIBRATION,
        SYNCHRONIZATION,
        GEOMETRY,
        TEMPORAL,
        OCCLUSION,
        IDENTITY,
        SURFACE,
        MEMORY,
        THERMAL,
        OUTPUT_INTEGRITY,
        REFERENCE_MISMATCH,
    }

    data class Contract(
        val p95TargetMillimeters: Double = 1.75,
        val p99LimitMillimeters: Double = 3.50,
        val maximumMeanLagMilliseconds: Double = 12.0,
        val maximumP95LagMilliseconds: Double = 16.7,
        val maximumDropoutRatio: Double = 0.01,
        val maximumIdentitySwapRatio: Double = 0.002,
        val minimumSynchronizationConfidence: Double = 0.90,
        val minimumCalibrationConfidence: Double = 0.90,
        val minimumTrackingConfidence: Double = 0.88,
        val minimumCrossViewAgreement: Double = 0.70,
        val minimumViewDiversity: Double = 0.60,
        val maximumCalibrationDriftRisk: Double = 0.10,
        val maximumRollingShutterRisk: Double = 0.20,
        val maximumTopologyRisk: Double = 0.05,
        val minimumSurfaceTemporalStability: Double = 0.90,
        val maximumThermalRisk: Double = 0.65,
        val maximumMemoryPressure: Double = 0.82,
        val maximumQueueAgeMilliseconds: Double = 12.0,
        val maximumProcessingMilliseconds: Double = 10.0,
    ) {
        init {
            require(
                p95TargetMillimeters.isFinite() &&
                    p95TargetMillimeters in 0.10..10.0,
            )
            require(
                p99LimitMillimeters.isFinite() &&
                    p99LimitMillimeters >= p95TargetMillimeters,
            )
            require(
                maximumMeanLagMilliseconds.isFinite() &&
                    maximumMeanLagMilliseconds > 0.0,
            )
            require(
                maximumP95LagMilliseconds >=
                    maximumMeanLagMilliseconds,
            )
            listOf(
                maximumDropoutRatio,
                maximumIdentitySwapRatio,
                minimumSynchronizationConfidence,
                minimumCalibrationConfidence,
                minimumTrackingConfidence,
                minimumCrossViewAgreement,
                minimumViewDiversity,
                maximumCalibrationDriftRisk,
                maximumRollingShutterRisk,
                maximumTopologyRisk,
                minimumSurfaceTemporalStability,
                maximumThermalRisk,
                maximumMemoryPressure,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(maximumQueueAgeMilliseconds > 0.0)
            require(maximumProcessingMilliseconds > 0.0)
        }

        companion object {
            fun controlledSubTwoMillimeter(): Contract =
                Contract(
                    p95TargetMillimeters = 1.75,
                    p99LimitMillimeters = 3.50,
                    maximumMeanLagMilliseconds = 12.0,
                    maximumP95LagMilliseconds = 16.7,
                    maximumDropoutRatio = 0.01,
                    maximumIdentitySwapRatio = 0.002,
                    minimumSynchronizationConfidence = 0.90,
                    minimumCalibrationConfidence = 0.90,
                    minimumTrackingConfidence = 0.88,
                    minimumCrossViewAgreement = 0.70,
                    minimumViewDiversity = 0.60,
                    maximumCalibrationDriftRisk = 0.10,
                    maximumRollingShutterRisk = 0.20,
                    maximumTopologyRisk = 0.05,
                    minimumSurfaceTemporalStability = 0.90,
                    maximumThermalRisk = 0.65,
                    maximumMemoryPressure = 0.82,
                    maximumQueueAgeMilliseconds = 12.0,
                    maximumProcessingMilliseconds = 10.0,
                )
        }
    }

    data class Measurement(
        val timestampNanos: Long,
        val region: Region,
        val errorMillimeters: Double,
        val uncertaintyMillimeters: Double,
        val lagMilliseconds: Double,
        val visibility: Double,
        val confidence: Double,
        val occluded: Boolean,
        val identityStable: Boolean,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val crossViewAgreement: Double,
        val rollingShutterRisk: Double,
        val calibrationDriftRisk: Double,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(errorMillimeters.isFinite() && errorMillimeters >= 0.0)
            require(uncertaintyMillimeters.isFinite() && uncertaintyMillimeters >= 0.0)
            require(lagMilliseconds.isFinite() && lagMilliseconds >= 0.0)
            listOf(
                visibility,
                confidence,
                calibrationConfidence,
                synchronizationConfidence,
                crossViewAgreement,
                rollingShutterRisk,
                calibrationDriftRisk,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
        }

        val conservativeErrorMillimeters: Double
            get() = errorMillimeters + uncertaintyMillimeters
    }

    data class ReferenceMetadata(
        val system: ReferenceSystem,
        val referenceP95Millimeters: Double,
        val referenceUncertaintyMillimeters: Double,
        val conditionsMatched: Boolean,
        val referenceSampleCount: Int,
        val sameCaptureVolume: Boolean,
        val sameMotionCohort: Boolean,
        val sameFrameTiming: Boolean,
        val sameCalibrationWindow: Boolean,
    ) {
        init {
            require(referenceP95Millimeters.isFinite() && referenceP95Millimeters >= 0.0)
            require(
                referenceUncertaintyMillimeters.isFinite() &&
                    referenceUncertaintyMillimeters >= 0.0,
            )
            require(referenceSampleCount >= 0)
        }

        val independentlyComparable: Boolean
            get() =
                conditionsMatched &&
                    sameCaptureVolume &&
                    sameMotionCohort &&
                    sameFrameTiming &&
                    sameCalibrationWindow &&
                    referenceSampleCount >= 120 &&
                    referenceUncertaintyMillimeters > 0.0 &&
                    system != ReferenceSystem.NONE
    }

    data class RegionReport(
        val region: Region,
        val sampleCount: Int,
        val p50Millimeters: Double,
        val p95Millimeters: Double,
        val p99Millimeters: Double,
        val maximumMillimeters: Double,
        val meanLagMilliseconds: Double,
        val p95LagMilliseconds: Double,
        val dropoutRatio: Double,
        val occlusionRatio: Double,
        val identityStability: Double,
        val meanConfidence: Double,
        val meanUncertaintyMillimeters: Double,
    )

    data class ReferenceComparison(
        val referenceSystem: ReferenceSystem,
        val comparable: Boolean,
        val ultraPrecisionComparable: Boolean = false,
        val systemP95PlusUncertaintyMillimeters: Double,
        val referenceP95PlusUncertaintyMillimeters: Double,
        val separationMarginMillimeters: Double,
        val systemHasLowerMeasuredError: Boolean,
        val inconclusiveReason: String?,
    )

    data class FailureEvent(
        val timestampNanos: Long,
        val kind: FailureKind,
        val severity: Int,
        val message: String,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(severity in 1..5)
            require(message.isNotBlank())
        }
    }

    data class Report(
        val sampleCount: Int,
        val overall: RegionReport,
        val perRegion: Map<Region, RegionReport>,
        val lowVisibilityP95Millimeters: Double,
        val occludedP95Millimeters: Double,
        val controlledSubTwoMillimeterGate: Boolean,
        val controlledUltraPrecisionGate: Boolean = false,
        val contractFailures: List<String>,
        val failureEventCount: Int,
        val referenceComparison: ReferenceComparison?,
    )

    private val measurements = ArrayDeque<Measurement>()
    private val failures = ArrayDeque<FailureEvent>()
    @Volatile
    private var closed = false

    @Synchronized
    fun addMeasurement(value: Measurement) {
        check(!closed) { "Precision contract is closed" }
        while (measurements.size >= maximumSamples) {
            measurements.removeFirst()
        }
        measurements.addLast(value)
    }

    @Synchronized
    fun addFailure(event: FailureEvent) {
        check(!closed)
        while (failures.size >= maximumFailureEvents) {
            failures.removeFirst()
        }
        failures.addLast(event)
    }

    @Synchronized
    fun snapshotSampleCount(): Int = measurements.size

    @Synchronized
    fun complete(
        contract: Contract = Contract.controlledSubTwoMillimeter(),
        reference: ReferenceMetadata? = null,
    ): Report {
        check(!closed)
        require(measurements.isNotEmpty()) {
            "At least one precision measurement is required"
        }

        val rows = measurements.toList()
        val overall = regionReport(Region.FULL_BODY, rows)
        val perRegion = rows
            .groupBy { it.region }
            .mapValues { (region, values) ->
                regionReport(region, values)
            }

        val lowVisibility = rows.filter { it.visibility < 0.50 }
        val occluded = rows.filter { it.occluded }

        val failuresFound = buildList {
            if (overall.p95Millimeters > contract.p95TargetMillimeters) {
                add("p95-geometric-error")
            }
            if (overall.p99Millimeters > contract.p99LimitMillimeters) {
                add("p99-geometric-error")
            }
            if (overall.meanLagMilliseconds >
                contract.maximumMeanLagMilliseconds
            ) {
                add("mean-temporal-lag")
            }
            if (overall.p95LagMilliseconds >
                contract.maximumP95LagMilliseconds
            ) {
                add("p95-temporal-lag")
            }
            if (overall.dropoutRatio > contract.maximumDropoutRatio) {
                add("dropout-ratio")
            }
            if (
                1.0 - overall.identityStability >
                contract.maximumIdentitySwapRatio
            ) {
                add("identity-instability")
            }
            if (
                rows.map { it.calibrationConfidence }.average() <
                contract.minimumCalibrationConfidence
            ) {
                add("calibration-confidence")
            }
            if (
                rows.map { it.synchronizationConfidence }.average() <
                contract.minimumSynchronizationConfidence
            ) {
                add("synchronization-confidence")
            }
            if (
                rows.map { it.crossViewAgreement }.average() <
                contract.minimumCrossViewAgreement
            ) {
                add("cross-view-agreement")
            }
            if (
                rows.map { it.rollingShutterRisk }.average() >
                contract.maximumRollingShutterRisk
            ) {
                add("rolling-shutter-risk")
            }
            if (
                rows.map { it.calibrationDriftRisk }.average() >
                contract.maximumCalibrationDriftRisk
            ) {
                add("calibration-drift-risk")
            }
        }

        val referenceComparison = reference?.let { compareToReference(overall, it) }
        val referenceGate = referenceComparison?.let {
            it.comparable && it.systemHasLowerMeasuredError
        } ?: true

        return Report(
            sampleCount = rows.size,
            overall = overall,
            perRegion = perRegion,
            lowVisibilityP95Millimeters =
                if (lowVisibility.isEmpty()) 0.0
                else quantile(
                    lowVisibility.map { it.conservativeErrorMillimeters },
                    0.95,
                ),
            occludedP95Millimeters =
                if (occluded.isEmpty()) 0.0
                else quantile(
                    occluded.map { it.conservativeErrorMillimeters },
                    0.95,
                ),
            controlledSubTwoMillimeterGate =
                failuresFound.isEmpty() &&
                    overall.p95Millimeters <=
                    contract.p95TargetMillimeters &&
                    referenceGate,
            controlledUltraPrecisionGate =
                failuresFound.isEmpty() &&
                    rows.size >= 256 &&
                    overall.p95Millimeters < 1.0 &&
                    overall.p99Millimeters <= 2.5 &&
                    referenceComparison?.ultraPrecisionComparable == true,
            contractFailures = failuresFound.distinct(),
            failureEventCount = failures.size,
            referenceComparison = referenceComparison,
        )
    }

    @Synchronized
    fun reset() {
        check(!closed)
        measurements.clear()
        failures.clear()
    }

    override fun close() {
        synchronized(this) {
            closed = true
            measurements.clear()
            failures.clear()
        }
    }

    private fun compareToReference(
        system: RegionReport,
        reference: ReferenceMetadata,
    ): ReferenceComparison {
        if (!reference.independentlyComparable) {
            return ReferenceComparison(
                referenceSystem = reference.system,
                comparable = false,
                ultraPrecisionComparable = false,
                systemP95PlusUncertaintyMillimeters =
                    system.p95Millimeters,
                referenceP95PlusUncertaintyMillimeters =
                    reference.referenceP95Millimeters +
                        reference.referenceUncertaintyMillimeters,
                separationMarginMillimeters = 0.0,
                systemHasLowerMeasuredError = false,
                inconclusiveReason =
                    when {
                        !reference.conditionsMatched ->
                            "reference-conditions-not-matched"
                        reference.referenceSampleCount < 120 ->
                            "reference-sample-count-too-small"
                        reference.referenceUncertaintyMillimeters <= 0.0 ->
                            "reference-uncertainty-missing"
                        !reference.sameCaptureVolume ->
                            "capture-volume-mismatch"
                        !reference.sameMotionCohort ->
                            "motion-cohort-mismatch"
                        !reference.sameFrameTiming ->
                            "frame-timing-mismatch"
                        !reference.sameCalibrationWindow ->
                            "calibration-window-mismatch"
                        else -> "reference-not-comparable"
                    },
            )
        }

        val systemUpper = system.p95Millimeters
        val referenceUpper =
            reference.referenceP95Millimeters +
                reference.referenceUncertaintyMillimeters
        val separation =
            referenceUpper -
                systemUpper

        return ReferenceComparison(
            referenceSystem = reference.system,
            comparable = true,
            ultraPrecisionComparable =
                reference.referenceSampleCount >= 240 &&
                    reference.referenceP95Millimeters < 1.0 &&
                    reference.referenceUncertaintyMillimeters <= 0.25,
            systemP95PlusUncertaintyMillimeters = systemUpper,
            referenceP95PlusUncertaintyMillimeters = referenceUpper,
            separationMarginMillimeters = separation,
            systemHasLowerMeasuredError = separation > 0.0,
            inconclusiveReason = if (separation <= 0.0) {
                "confidence-bounds-overlap"
            } else {
                null
            },
        )
    }

    private fun regionReport(
        region: Region,
        rows: List<Measurement>,
    ): RegionReport {
        require(rows.isNotEmpty())
        val errors = rows.map { it.conservativeErrorMillimeters }
        val lags = rows.map { it.lagMilliseconds }
        return RegionReport(
            region = region,
            sampleCount = rows.size,
            p50Millimeters = quantile(errors, 0.50),
            p95Millimeters = quantile(errors, 0.95),
            p99Millimeters = quantile(errors, 0.99),
            maximumMillimeters = errors.maxOrNull() ?: 0.0,
            meanLagMilliseconds = lags.average(),
            p95LagMilliseconds = quantile(lags, 0.95),
            dropoutRatio =
                rows.count { it.confidence < 0.50 }
                    .toDouble() / rows.size,
            occlusionRatio =
                rows.count { it.occluded }
                    .toDouble() / rows.size,
            identityStability =
                rows.count { it.identityStable }
                    .toDouble() / rows.size,
            meanConfidence =
                rows.map { it.confidence }.average(),
            meanUncertaintyMillimeters =
                rows.map { it.uncertaintyMillimeters }.average(),
        )
    }

    private fun quantile(
        values: List<Double>,
        percentile: Double,
    ): Double {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        val index = percentile.coerceIn(0.0, 1.0) *
            (sorted.lastIndex)
        val lower = index.toInt()
        val upper = min(lower + 1, sorted.lastIndex)
        val alpha = index - lower
        return sorted[lower] * (1.0 - alpha) +
            sorted[upper] * alpha
    }

    /**
     * Robust scale estimate used for diagnostics. It is intentionally not used to erase outliers
     * from P95/P99 acceptance statistics.
     */
    fun robustScaleMillimeters(): Double {
        synchronized(this) {
            if (measurements.size < 2) return 0.0
            val values = measurements.map { it.conservativeErrorMillimeters }
            val median = quantile(values, 0.50)
            val deviations = values.map { abs(it - median) }
            return 1.4826 * quantile(deviations, 0.50)
        }
    }

    fun uncertaintyFromComponents(
        triangulationMillimeters: Double,
        calibrationMillimeters: Double,
        temporalMillimeters: Double,
        rollingShutterMillimeters: Double,
    ): Double {
        val components = listOf(
            triangulationMillimeters,
            calibrationMillimeters,
            temporalMillimeters,
            rollingShutterMillimeters,
        )
        require(components.all { it.isFinite() && it >= 0.0 })
        return sqrt(
            triangulationMillimeters * triangulationMillimeters +
                calibrationMillimeters * calibrationMillimeters +
                temporalMillimeters * temporalMillimeters +
                rollingShutterMillimeters * rollingShutterMillimeters,
        )
    }
}
