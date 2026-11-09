package com.nivukx.nox.core.ai

import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Production quality authority for motion-capture accuracy, runtime safety and controlled
 * reference benchmarking.
 *
 * It deliberately separates requested fidelity from measured fidelity. External comparisons
 * remain inconclusive until capture conditions and reference uncertainty are supplied.
 */
class ProductionMotionCaptureQualityOrchestrator(
    private val maximumBenchmarkSamples: Int = 8192,
    private val maximumHealthEvents: Int = 4096,
) : AutoCloseable {

    enum class QualityBand {
        REFERENCE,
        ULTRA,
        HIGH,
        DEGRADED,
        BLOCKED,
    }

    enum class ReferenceClass {
        MARKER_OPTICAL,
        INFRARED_OPTICAL,
        RGBD,
        ROBOTICS_GROUND_TRUTH,
        OTHER,
    }

    enum class MotionRegion {
        ROOT,
        TORSO,
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
        FULL_BODY,
    }

    enum class Scenario {
        STATIC,
        STANDARD_MOTION,
        FAST_MOTION,
        OCCLUSION_HEAVY,
        INTERACTION,
    }

    enum class FailureSeverity {
        INFO,
        WARNING,
        ERROR,
        CRITICAL,
    }

    data class QualityContract(
        val targetP95ErrorMillimeters: Double = 5.0,
        val maximumP99ErrorMillimeters: Double = 12.0,
        val maximumCatastrophicErrorMillimeters: Double = 15.0,
        val maximumCatastrophicErrorRatio: Double = 0.001,
        val maximumTemporalLagMilliseconds: Double = 25.0,
        val maximumDropoutRatio: Double = 0.02,
        val minimumSynchronizationConfidence: Double = 0.75,
        val minimumTimingAuthorityConfidence: Double = 0.0,
        val maximumTimingAuthorityResidualMilliseconds: Double = 50.0,
        val requireReferenceGradeTimingAuthority: Boolean = false,
        val minimumCalibrationConfidence: Double = 0.75,
        val minimumTrackingConfidence: Double = 0.75,
        val minimumCrossViewAgreement: Double = 0.50,
        val minimumViewDiversity: Double = 0.40,
        val minimumIdentityContinuity: Double = 0.80,
        val maximumNativeQueueAgeMilliseconds: Double = 25.0,
        val maximumNativeProcessingMilliseconds: Double = 12.0,
        val maximumThermalRisk: Double = 0.82,
        val maximumMemoryPressure: Double = 0.90,
        val maximumCrashRatePerMinute: Double = 0.02,
        val maximumMeshTopologyRisk: Double = 0.08,
        val minimumSurfaceTemporalStability: Double = 0.80,
        val minimumFaceEyeConfidence: Double = 0.80,
        val minimumHand3dConfidence: Double = 0.78,
    ) {
        init {
            require(targetP95ErrorMillimeters > 0.0)
            require(maximumP99ErrorMillimeters >= targetP95ErrorMillimeters)
            require(
                maximumCatastrophicErrorMillimeters >=
                    maximumP99ErrorMillimeters,
            )
            require(
                maximumCatastrophicErrorRatio.isFinite() &&
                    maximumCatastrophicErrorRatio >= 0.0 &&
                    maximumCatastrophicErrorRatio <= 1.0,
            )
            require(maximumTemporalLagMilliseconds > 0.0)
            require(
                maximumTimingAuthorityResidualMilliseconds > 0.0
            )
            require(maximumNativeQueueAgeMilliseconds > 0.0)
            require(maximumNativeProcessingMilliseconds > 0.0)
            require(maximumCrashRatePerMinute >= 0.0)
            listOf(
                maximumDropoutRatio,
                minimumSynchronizationConfidence,
                minimumTimingAuthorityConfidence,
                minimumCalibrationConfidence,
                minimumTrackingConfidence,
                minimumCrossViewAgreement,
                minimumViewDiversity,
                minimumIdentityContinuity,
                maximumThermalRisk,
                maximumMemoryPressure,
                maximumMeshTopologyRisk,
                minimumSurfaceTemporalStability,
                minimumFaceEyeConfidence,
                minimumHand3dConfidence,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
        }
    }

    data class Evidence(
        val timestampNanos: Long,
        val synchronizationConfidence: Double = 1.0,
        val timingAuthorityConfidence: Double = 1.0,
        val timingAuthorityResidualP95Milliseconds: Double = 0.0,
        val timingAuthorityReferenceGrade: Boolean = true,
        val calibrationConfidence: Double = 1.0,
        val trackingConfidence: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val identityContinuity: Double = 1.0,
        val p95ErrorMillimeters: Double = 0.0,
        val p99ErrorMillimeters: Double = 0.0,
        val temporalLagMilliseconds: Double = 0.0,
        val dropoutRatio: Double = 0.0,
        val nativeQueueAgeMilliseconds: Double = 0.0,
        val nativeProcessingMilliseconds: Double = 0.0,
        val thermalRisk: Double = 0.0,
        val memoryPressure: Double = 0.0,
        val gpuHeadroom: Double = 1.0,
        val npuHeadroom: Double = 1.0,
        val cpuHeadroom: Double = 1.0,
        val crashRatePerMinute: Double = 0.0,
        val meshTopologyRisk: Double = 0.0,
        val surfaceTemporalStability: Double = 1.0,
        val faceEyeConfidence: Double = 1.0,
        val hand3dConfidence: Double = 1.0,
        val calibrationDriftRisk: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val lightingConfidence: Double = 1.0,
        val activeCameras: Int = 1,
        val activeSubjects: Int = 1,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(activeCameras >= 0)
            require(activeSubjects >= 0)
            listOf(
                synchronizationConfidence,
                timingAuthorityConfidence,
                calibrationConfidence,
                trackingConfidence,
                crossViewAgreement,
                viewDiversity,
                identityContinuity,
                dropoutRatio,
                thermalRisk,
                memoryPressure,
                gpuHeadroom,
                npuHeadroom,
                cpuHeadroom,
                meshTopologyRisk,
                surfaceTemporalStability,
                faceEyeConfidence,
                hand3dConfidence,
                calibrationDriftRisk,
                rollingShutterRisk,
                occlusionRisk,
                lightingConfidence,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            listOf(
                p95ErrorMillimeters,
                p99ErrorMillimeters,
                temporalLagMilliseconds,
                nativeQueueAgeMilliseconds,
                nativeProcessingMilliseconds,
                crashRatePerMinute,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
        }

        val computeHeadroom: Double
            get() = minOf(
                gpuHeadroom,
                npuHeadroom,
                cpuHeadroom,
                1.0 - memoryPressure,
            )

        val evidenceConfidence: Double
            get() = (
                synchronizationConfidence * 0.16 +
                    calibrationConfidence * 0.16 +
                    trackingConfidence * 0.16 +
                    crossViewAgreement * 0.12 +
                    viewDiversity * 0.08 +
                    identityContinuity * 0.10 +
                    surfaceTemporalStability * 0.08 +
                    faceEyeConfidence * 0.05 +
                    hand3dConfidence * 0.05 +
                    lightingConfidence * 0.04
                ).coerceIn(0.0, 1.0)
    }

    data class ExecutionBudget(
        val detectorDutyCycle: Double,
        val depthDutyCycle: Double,
        val triangulationDutyCycle: Double,
        val temporalFusionDutyCycle: Double,
        val faceEyeDutyCycle: Double,
        val handDutyCycle: Double,
        val surfaceDutyCycle: Double,
        val diagnosticsDutyCycle: Double,
        val cameraSelectionBudget: Int,
        val maximumConcurrentWorkers: Int,
        val recommendedFrameStride: Int,
        val adaptiveScale: Double,
        val thermalReserve: Double,
        val memoryReserve: Double,
    ) {
        init {
            listOf(
                detectorDutyCycle,
                depthDutyCycle,
                triangulationDutyCycle,
                temporalFusionDutyCycle,
                faceEyeDutyCycle,
                handDutyCycle,
                surfaceDutyCycle,
                diagnosticsDutyCycle,
                adaptiveScale,
                thermalReserve,
                memoryReserve,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            require(cameraSelectionBudget >= 1)
            require(maximumConcurrentWorkers >= 1)
            require(recommendedFrameStride >= 1)
        }
    }

    data class Decision(
        val band: QualityBand,
        val captureAllowed: Boolean,
        val contractSatisfied: Boolean,
        val hardSafetyBlock: Boolean,
        val score: Double,
        val confidence: Double,
        val adaptiveScale: Double,
        val blockingReasons: List<String>,
        val warningReasons: List<String>,
        val execution: ExecutionBudget,
    ) {
        val needsAttention: Boolean
            get() = hardSafetyBlock || warningReasons.isNotEmpty()

        val isReferenceGrade: Boolean
            get() = band == QualityBand.REFERENCE && contractSatisfied
    }

    data class BenchmarkSample(
        val timestampNanos: Long,
        val region: MotionRegion,
        val errorMillimeters: Double,
        val temporalLagMilliseconds: Double,
        val visibility: Double = 1.0,
        val occluded: Boolean = false,
        val interaction: Boolean = false,
        val fastMotion: Boolean = false,
        val multiPersonOcclusion: Boolean = false,
        val identityExchangeRisk: Double = 0.0,
        val identityStable: Boolean = true,
        val confidence: Double = 1.0,
        val referenceErrorMillimeters: Double? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(errorMillimeters.isFinite() && errorMillimeters >= 0.0)
            require(temporalLagMilliseconds.isFinite() && temporalLagMilliseconds >= 0.0)
            require(visibility.isFinite() && visibility in 0.0..1.0)
            require(identityExchangeRisk.isFinite() && identityExchangeRisk in 0.0..1.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            referenceErrorMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class RegionStatistics(
        val region: MotionRegion,
        val sampleCount: Int,
        val meanErrorMillimeters: Double,
        val medianErrorMillimeters: Double,
        val p95ErrorMillimeters: Double,
        val p99ErrorMillimeters: Double,
        val catastrophicErrorCount: Int,
        val catastrophicErrorRatio: Double,
        val maxErrorMillimeters: Double,
        val meanLagMilliseconds: Double,
        val dropoutRatio: Double,
        val occludedSampleRatio: Double,
        val identityStableRatio: Double,
        val confidence: Double,
    )

    data class ReferenceComparison(
        val referenceClass: ReferenceClass,
        val conditionsMatched: Boolean,
        val referenceP95ErrorMillimeters: Double,
        val referenceUncertaintyMillimeters: Double,
        val systemP95ErrorMillimeters: Double,
        val systemUncertaintyMillimeters: Double,
        val combinedUncertaintyMillimeters: Double,
        val systemP95ConfidenceUpperMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val referenceP95ConfidenceLowerMillimeters: Double =
            Double.POSITIVE_INFINITY,
        val statisticalSeparationMarginMillimeters: Double =
            Double.NEGATIVE_INFINITY,
        val statisticallyLowerErrorThanReference: Boolean,
        val inconclusiveReason: String?,
    )

    data class ScenarioStatistics(
        val scenario: Scenario,
        val sampleCount: Int,
        val p50Millimeters: Double,
        val p95Millimeters: Double,
        val p99Millimeters: Double,
        val meanLagMilliseconds: Double,
        val dropoutRatio: Double,
        val occludedRatio: Double,
        val identityStableRatio: Double,
        val meanReferenceErrorMillimeters: Double?,
    )

    data class ScenarioReferenceComparison(
        val scenario: Scenario,
        val sampleCount: Int,
        val systemP95Millimeters: Double,
        val referenceP95Millimeters: Double,
        val combinedUncertaintyMillimeters: Double,
        val statisticallyLowerErrorThanReference: Boolean,
        val inconclusiveReason: String?,
    )

    data class BenchmarkReport(
        val sampleCount: Int,
        val overall: RegionStatistics,
        val perRegion: Map<MotionRegion, RegionStatistics>,
        val scenarioStatistics: Map<Scenario, ScenarioStatistics>,
        val scenarioReferenceComparisons:
            Map<Scenario, ScenarioReferenceComparison>,
        val referenceComparison: ReferenceComparison?,
        val lowVisibilityP95ErrorMillimeters: Double,
        val occludedP95ErrorMillimeters: Double,
        val identityStableRatio: Double,
        val catastrophicErrorCount: Int,
        val catastrophicErrorRatio: Double,
        val qualityGate: Boolean,
    )

    data class HealthEvent(
        val timestampNanos: Long,
        val severity: FailureSeverity,
        val code: String,
        val recoverable: Boolean,
    )

    data class HealthSnapshot(
        val eventCount: Int,
        val criticalCount: Int,
        val errorCount: Int,
        val warningCount: Int,
        val unrecoverableCount: Int,
        val recentCodes: List<String>,
        val healthy: Boolean,
    )

    private val benchmarkSamples = ArrayDeque<BenchmarkSample>()
    private val healthEvents = ArrayDeque<HealthEvent>()

    @Volatile
    private var closed = false

    @Volatile
    private var latestDecision: Decision? = null

    @Synchronized
    fun evaluate(
        evidence: Evidence,
        contract: QualityContract = QualityContract(),
        qualityLock: Boolean = false,
    ): Decision {
        check(!closed) { "Quality orchestrator is closed" }

        val blockers = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        if (evidence.synchronizationConfidence < 0.35) {
            blockers += "sync-confidence-critical"
        } else if (
            evidence.synchronizationConfidence <
                contract.minimumSynchronizationConfidence
        ) {
            warnings += "sync-below-contract"
        }

        if (
            evidence.timingAuthorityConfidence <
                contract.minimumTimingAuthorityConfidence
        ) {
            warnings += "timing-authority-below-contract"
        }

        if (
            evidence.timingAuthorityResidualP95Milliseconds >
                contract.maximumTimingAuthorityResidualMilliseconds
        ) {
            warnings += "timing-authority-residual-above-contract"
        }

        if (
            contract.requireReferenceGradeTimingAuthority &&
            !evidence.timingAuthorityReferenceGrade
        ) {
            warnings += "timing-authority-reference-grade-required"
        }

        if (evidence.calibrationConfidence < 0.35) {
            blockers += "calibration-confidence-critical"
        } else if (
            evidence.calibrationConfidence <
                contract.minimumCalibrationConfidence
        ) {
            warnings += "calibration-below-contract"
        }

        if (evidence.trackingConfidence < 0.30) {
            blockers += "tracking-confidence-critical"
        } else if (
            evidence.trackingConfidence <
                contract.minimumTrackingConfidence
        ) {
            warnings += "tracking-below-contract"
        }

        if (
            evidence.crashRatePerMinute >
                contract.maximumCrashRatePerMinute * 4.0
        ) {
            blockers += "crash-rate-unsafe"
        } else if (
            evidence.crashRatePerMinute >
                contract.maximumCrashRatePerMinute
        ) {
            warnings += "crash-rate-above-contract"
        }

        if (evidence.memoryPressure >= 0.98) {
            blockers += "memory-pressure-critical"
        } else if (
            evidence.memoryPressure >
                contract.maximumMemoryPressure
        ) {
            warnings += "memory-pressure-high"
        }

        if (evidence.thermalRisk >= 0.97) {
            blockers += "thermal-risk-critical"
        } else if (evidence.thermalRisk > contract.maximumThermalRisk) {
            warnings += "thermal-risk-high"
        }

        if (
            evidence.dropoutRatio >
                contract.maximumDropoutRatio * 4.0
        ) {
            blockers += "dropout-rate-unsafe"
        } else if (
            evidence.dropoutRatio >
                contract.maximumDropoutRatio
        ) {
            warnings += "dropout-above-contract"
        }

        if (
            evidence.p95ErrorMillimeters >
                contract.maximumP99ErrorMillimeters * 1.8
        ) {
            blockers += "geometric-error-critical"
        } else if (
            evidence.p95ErrorMillimeters >
                contract.targetP95ErrorMillimeters
        ) {
            warnings += "p95-error-above-contract"
        }

        if (
            evidence.p99ErrorMillimeters >
                contract.maximumP99ErrorMillimeters
        ) {
            warnings += "p99-error-above-contract"
        }

        if (
            evidence.temporalLagMilliseconds >
                contract.maximumTemporalLagMilliseconds * 2.0
        ) {
            blockers += "temporal-lag-critical"
        } else if (
            evidence.temporalLagMilliseconds >
                contract.maximumTemporalLagMilliseconds
        ) {
            warnings += "temporal-lag-above-contract"
        }

        if (
            evidence.nativeQueueAgeMilliseconds >
                contract.maximumNativeQueueAgeMilliseconds * 2.0
        ) {
            blockers += "native-queue-critical"
        } else if (
            evidence.nativeQueueAgeMilliseconds >
                contract.maximumNativeQueueAgeMilliseconds
        ) {
            warnings += "native-queue-residence-high"
        }

        if (
            evidence.nativeProcessingMilliseconds >
                contract.maximumNativeProcessingMilliseconds * 2.0
        ) {
            blockers += "native-processing-critical"
        } else if (
            evidence.nativeProcessingMilliseconds >
                contract.maximumNativeProcessingMilliseconds
        ) {
            warnings += "native-processing-high"
        }

        if (evidence.viewDiversity < contract.minimumViewDiversity) {
            warnings += "view-diversity-below-contract"
        }

        if (
            evidence.crossViewAgreement <
                contract.minimumCrossViewAgreement
        ) {
            warnings += "cross-view-agreement-below-contract"
        }

        if (
            evidence.identityContinuity <
                contract.minimumIdentityContinuity
        ) {
            warnings += "identity-continuity-below-contract"
        }

        if (evidence.meshTopologyRisk > contract.maximumMeshTopologyRisk) {
            warnings += "mesh-topology-risk-high"
        }

        if (
            evidence.surfaceTemporalStability <
                contract.minimumSurfaceTemporalStability
        ) {
            warnings += "surface-temporal-stability-low"
        }

        if (evidence.faceEyeConfidence < contract.minimumFaceEyeConfidence) {
            warnings += "face-eye-confidence-below-contract"
        }

        if (evidence.hand3dConfidence < contract.minimumHand3dConfidence) {
            warnings += "hand-3d-confidence-below-contract"
        }

        if (evidence.calibrationDriftRisk > 0.35) {
            warnings += "calibration-drift-risk-rising"
        }

        if (evidence.rollingShutterRisk > 0.55) {
            warnings += "rolling-shutter-risk-rising"
        }

        if (evidence.occlusionRisk > 0.65) {
            warnings += "occlusion-risk-high"
        }

        val contractSatisfied =
            blockers.isEmpty() &&
                evidence.p95ErrorMillimeters <=
                contract.targetP95ErrorMillimeters &&
                evidence.p99ErrorMillimeters <=
                contract.maximumP99ErrorMillimeters &&
                evidence.temporalLagMilliseconds <=
                contract.maximumTemporalLagMilliseconds &&
                evidence.dropoutRatio <=
                contract.maximumDropoutRatio &&
                evidence.synchronizationConfidence >=
                contract.minimumSynchronizationConfidence &&
                evidence.timingAuthorityConfidence >=
                contract.minimumTimingAuthorityConfidence &&
                evidence.timingAuthorityResidualP95Milliseconds <=
                contract.maximumTimingAuthorityResidualMilliseconds &&
                (
                    !contract.requireReferenceGradeTimingAuthority ||
                        evidence.timingAuthorityReferenceGrade
                ) &&
                evidence.calibrationConfidence >=
                contract.minimumCalibrationConfidence &&
                evidence.trackingConfidence >=
                contract.minimumTrackingConfidence &&
                evidence.crossViewAgreement >=
                contract.minimumCrossViewAgreement &&
                evidence.viewDiversity >=
                contract.minimumViewDiversity &&
                evidence.identityContinuity >=
                contract.minimumIdentityContinuity &&
                evidence.surfaceTemporalStability >=
                contract.minimumSurfaceTemporalStability &&
                evidence.faceEyeConfidence >=
                contract.minimumFaceEyeConfidence &&
                evidence.hand3dConfidence >=
                contract.minimumHand3dConfidence

        if (qualityLock && !contractSatisfied) {
            warnings += "quality-lock-contract-not-satisfied"
        }

        val pressure = max(
            max(evidence.thermalRisk, evidence.memoryPressure),
            1.0 - min(
                evidence.gpuHeadroom,
                min(evidence.npuHeadroom, evidence.cpuHeadroom),
            ),
        ).coerceIn(0.0, 1.0)

        val errorPressure = (
            evidence.p95ErrorMillimeters /
                contract.targetP95ErrorMillimeters
        ).coerceIn(0.0, 3.0) / 3.0

        val score = (
            evidence.evidenceConfidence * 0.72 +
                (1.0 - errorPressure) * 0.16 +
                (1.0 - pressure) * 0.12
            ).coerceIn(0.0, 1.0)

        val adaptiveScale = when {
            blockers.isNotEmpty() -> 0.50
            pressure <= 0.18 -> 1.0
            pressure >= 0.90 -> 0.58
            else ->
                1.0 - ((pressure - 0.18) / 0.72) * 0.42
        }.coerceIn(0.50, 1.0)

        val computeScale = min(
            evidence.gpuHeadroom,
            min(
                evidence.npuHeadroom,
                min(evidence.cpuHeadroom, 1.0 - evidence.memoryPressure),
            ),
        ).coerceIn(0.0, 1.0)

        val execution = ExecutionBudget(
            detectorDutyCycle =
                (0.76 * adaptiveScale).coerceIn(0.35, 1.0),
            depthDutyCycle =
                (0.72 * adaptiveScale +
                    evidence.viewDiversity * 0.18).coerceIn(0.32, 1.0),
            triangulationDutyCycle =
                (0.84 * adaptiveScale +
                    evidence.crossViewAgreement * 0.16)
                    .coerceIn(0.40, 1.0),
            temporalFusionDutyCycle =
                (0.92 * adaptiveScale +
                    evidence.identityContinuity * 0.08)
                    .coerceIn(0.55, 1.0),
            faceEyeDutyCycle =
                (0.90 * adaptiveScale +
                    evidence.faceEyeConfidence * 0.10)
                    .coerceIn(0.50, 1.0),
            handDutyCycle =
                (0.86 * adaptiveScale +
                    evidence.hand3dConfidence * 0.14)
                    .coerceIn(0.45, 1.0),
            surfaceDutyCycle =
                (0.82 * adaptiveScale +
                    evidence.surfaceTemporalStability * 0.18)
                    .coerceIn(0.40, 1.0),
            diagnosticsDutyCycle =
                (0.35 + evidence.evidenceConfidence * 0.45)
                    .coerceIn(0.20, 0.80),
            cameraSelectionBudget =
                max(1, evidence.activeCameras),
            maximumConcurrentWorkers =
                (1 + 15.0 * computeScale)
                    .toInt()
                    .coerceIn(1, 16),
            recommendedFrameStride =
                when {
                    blockers.isNotEmpty() -> 3
                    pressure >= 0.65 -> 2
                    else -> 1
                },
            adaptiveScale = adaptiveScale,
            thermalReserve =
                (1.0 - evidence.thermalRisk).coerceIn(0.0, 1.0),
            memoryReserve =
                (1.0 - evidence.memoryPressure).coerceIn(0.0, 1.0),
        )

        val band = when {
            blockers.isNotEmpty() -> QualityBand.BLOCKED
            contractSatisfied && score >= 0.90 -> QualityBand.REFERENCE
            score >= 0.78 -> QualityBand.ULTRA
            score >= 0.60 -> QualityBand.HIGH
            else -> QualityBand.DEGRADED
        }

        return Decision(
            band = band,
            captureAllowed = blockers.isEmpty(),
            contractSatisfied = contractSatisfied,
            hardSafetyBlock = blockers.isNotEmpty(),
            score = score,
            confidence = evidence.evidenceConfidence,
            adaptiveScale = adaptiveScale,
            blockingReasons = blockers.distinct(),
            warningReasons = warnings.distinct(),
            execution = execution,
        ).also {
            latestDecision = it
        }
    }

    fun latestDecision(): Decision? = latestDecision

    @Synchronized
    fun addBenchmarkSample(sample: BenchmarkSample) {
        check(!closed) { "Quality orchestrator is closed" }
        while (benchmarkSamples.size >= maximumBenchmarkSamples) {
            benchmarkSamples.removeFirst()
        }
        benchmarkSamples.addLast(sample)
    }

    @Synchronized
    fun completeBenchmark(
        referenceClass: ReferenceClass? = null,
        referenceP95ErrorMillimeters: Double? = null,
        referenceUncertaintyMillimeters: Double = 0.0,
        conditionsMatched: Boolean = false,
        contract: QualityContract = QualityContract(),
    ): BenchmarkReport {
        check(!closed) { "Quality orchestrator is closed" }
        require(referenceUncertaintyMillimeters >= 0.0)
        require(
            referenceP95ErrorMillimeters == null ||
                referenceP95ErrorMillimeters >= 0.0,
        )

        val rows = benchmarkSamples.toList()
        require(rows.isNotEmpty()) { "Benchmark requires samples" }

        val overall = stats(
            MotionRegion.FULL_BODY,
            rows,
            contract.maximumCatastrophicErrorMillimeters,
        )
        val regions = rows
            .groupBy { it.region }
            .mapValues { entry ->
                stats(
                    entry.key,
                    entry.value,
                    contract.maximumCatastrophicErrorMillimeters,
                )
            }

        val lowVisibilityRows = rows.filter { it.visibility < 0.50 }
        val occludedRows = rows.filter { it.occluded }
        val stableRatio =
            rows.count { it.identityStable }.toDouble() / rows.size
        val catastrophicCount =
            rows.count {
                it.errorMillimeters >=
                    contract.maximumCatastrophicErrorMillimeters
            }
        val catastrophicRatio =
            catastrophicCount.toDouble() / rows.size

        val comparison =
            if (
                referenceClass != null &&
                referenceP95ErrorMillimeters != null
            ) {
                val systemUncertainty =
                    robustUncertainty(rows.map { it.errorMillimeters })
                val combined = sqrt(
                    systemUncertainty * systemUncertainty +
                        referenceUncertaintyMillimeters *
                        referenceUncertaintyMillimeters,
                )
                val reason = when {
                    !conditionsMatched ->
                        "reference-conditions-not-matched"
                    rows.size < 120 ->
                        "benchmark-too-small-for-reference-separation"
                    referenceUncertaintyMillimeters <= 0.0 ->
                        "reference-uncertainty-not-supplied"
                    else -> null
                }
                val referenceLower =
                    referenceP95ErrorMillimeters -
                        referenceUncertaintyMillimeters

                val systemUpper =
                    overall.p95ConfidenceUpperMillimeters +
                        systemUncertainty

                val separationMargin =
                    referenceLower -
                        systemUpper

                ReferenceComparison(
                    referenceClass = referenceClass,
                    conditionsMatched = conditionsMatched,
                    referenceP95ErrorMillimeters =
                        referenceP95ErrorMillimeters,
                    referenceUncertaintyMillimeters =
                        referenceUncertaintyMillimeters,
                    systemP95ErrorMillimeters =
                        overall.p95ErrorMillimeters,
                    systemUncertaintyMillimeters =
                        systemUncertainty,
                    combinedUncertaintyMillimeters = combined,
                    systemP95ConfidenceUpperMillimeters =
                        systemUpper,
                    referenceP95ConfidenceLowerMillimeters =
                        referenceLower,
                    statisticalSeparationMarginMillimeters =
                        separationMargin,
                    statisticallyLowerErrorThanReference =
                        reason == null &&
                            separationMargin > 0.0,
                    inconclusiveReason =
                        when {
                            reason != null -> reason
                            separationMargin <= 0.0 ->
                                "reference-separation-overlaps-statistical-confidence"
                            else -> null
                        },
                )
            } else {
                null
            }

        val gate =
            overall.p95ErrorMillimeters <=
                contract.targetP95ErrorMillimeters &&
                overall.p99ErrorMillimeters <=
                contract.maximumP99ErrorMillimeters &&
                catastrophicRatio <=
                contract.maximumCatastrophicErrorRatio &&
                overall.meanLagMilliseconds <=
                contract.maximumTemporalLagMilliseconds &&
                overall.dropoutRatio <=
                contract.maximumDropoutRatio &&
                stableRatio >= contract.minimumIdentityContinuity

        val scenarioStatistics =
            rows
                .groupBy(::scenarioOf)
                .mapValues { (scenario, scenarioRows) ->
                    scenarioStatistics(
                        scenario = scenario,
                        rows = scenarioRows,
                    )
                }

        val scenarioReferenceComparisons =
            if (
                referenceClass != null &&
                referenceP95ErrorMillimeters != null &&
                referenceUncertaintyMillimeters > 0.0 &&
                conditionsMatched
            ) {
                rows
                    .groupBy(::scenarioOf)
                    .mapValues { (scenario, scenarioRows) ->
                        scenarioReferenceComparison(
                            scenario = scenario,
                            rows = scenarioRows,
                        )
                    }
            } else {
                emptyMap()
            }

        return BenchmarkReport(
            sampleCount = rows.size,
            overall = overall,
            perRegion = regions,
            scenarioStatistics = scenarioStatistics,
            scenarioReferenceComparisons =
                scenarioReferenceComparisons,
            referenceComparison = comparison,
            lowVisibilityP95ErrorMillimeters =
                if (lowVisibilityRows.isEmpty()) {
                    0.0
                } else {
                    quantile(
                        lowVisibilityRows.map {
                            it.errorMillimeters
                        },
                        0.95,
                    )
                },
            occludedP95ErrorMillimeters =
                if (occludedRows.isEmpty()) {
                    0.0
                } else {
                    quantile(
                        occludedRows.map {
                            it.errorMillimeters
                        },
                        0.95,
                    )
                },
            identityStableRatio = stableRatio,
            catastrophicErrorCount = catastrophicCount,
            catastrophicErrorRatio = catastrophicRatio,
            qualityGate = gate,
        )
    }

    private fun scenarioOf(
        sample: BenchmarkSample,
    ): Scenario =
        when {
            sample.multiPersonOcclusion ->
                Scenario.OCCLUSION_HEAVY
            sample.occluded &&
                sample.visibility < 0.65 ->
                Scenario.OCCLUSION_HEAVY
            sample.interaction ->
                Scenario.INTERACTION
            sample.fastMotion ->
                Scenario.FAST_MOTION
            sample.temporalLagMilliseconds <= 4.0 &&
                sample.errorMillimeters <= 1.0 ->
                Scenario.STATIC
            else ->
                Scenario.STANDARD_MOTION
        }

    private fun scenarioStatistics(
        scenario: Scenario,
        rows: List<BenchmarkSample>,
    ): ScenarioStatistics {
        val errors =
            rows.map { it.errorMillimeters }
        val referenceErrors =
            rows.mapNotNull {
                if (it.referenceErrorMillimeters?.isFinite() == true) {
                    it.referenceErrorMillimeters
                } else {
                    null
                }
            }

        return ScenarioStatistics(
            scenario = scenario,
            sampleCount = rows.size,
            p50Millimeters =
                quantile(errors, 0.50),
            p95Millimeters =
                quantile(errors, 0.95),
            p99Millimeters =
                quantile(errors, 0.99),
            meanLagMilliseconds =
                rows.map {
                    it.temporalLagMilliseconds
                }.average(),
            dropoutRatio =
                rows.count {
                    it.confidence < 0.50
                }.toDouble() /
                    rows.size.coerceAtLeast(1),
            occludedRatio =
                rows.count {
                    it.occluded
                }.toDouble() /
                    rows.size.coerceAtLeast(1),
            identityStableRatio =
                rows.count {
                    it.identityStable
                }.toDouble() /
                    rows.size.coerceAtLeast(1),
            meanReferenceErrorMillimeters =
                referenceErrors
                    .takeIf { it.isNotEmpty() }
                    ?.average(),
        )
    }

    private fun scenarioReferenceComparison(
        scenario: Scenario,
        rows: List<BenchmarkSample>,
    ): ScenarioReferenceComparison {
        val pairs =
            rows.mapNotNull { row ->
                val reference =
                    row.referenceErrorMillimeters
                        ?.takeIf {
                            it.isFinite() && it >= 0.0
                        }
                    ?: return@mapNotNull null
                row.errorMillimeters to reference
            }

        if (pairs.size < 20) {
            return ScenarioReferenceComparison(
                scenario = scenario,
                sampleCount = pairs.size,
                systemP95Millimeters =
                    pairs
                        .map { it.first }
                        .takeIf { it.isNotEmpty() }
                        ?.let { quantile(it, 0.95) }
                        ?: 0.0,
                referenceP95Millimeters =
                    pairs
                        .map { it.second }
                        .takeIf { it.isNotEmpty() }
                        ?.let { quantile(it, 0.95) }
                        ?: 0.0,
                combinedUncertaintyMillimeters =
                    Double.POSITIVE_INFINITY,
                statisticallyLowerErrorThanReference = false,
                inconclusiveReason =
                    "scenario-reference-sample-count-too-small",
            )
        }

        val system =
            pairs.map { it.first }
        val reference =
            pairs.map { it.second }

        val systemP95 =
            quantile(system, 0.95)
        val referenceP95 =
            quantile(reference, 0.95)
        val systemUncertainty =
            robustUncertainty(system)
        val referenceUncertainty =
            robustUncertainty(reference)
        val combined =
            sqrt(
                systemUncertainty * systemUncertainty +
                    referenceUncertainty * referenceUncertainty,
            )
        val systemUpper =
            empiricalQuantileConfidenceBounds(
                sorted = system.sorted(),
                quantile = 0.95,
            ).second +
                systemUncertainty
        val referenceLower =
            empiricalQuantileConfidenceBounds(
                sorted = reference.sorted(),
                quantile = 0.95,
            ).first -
                referenceUncertainty
        val margin =
            referenceLower -
                systemUpper

        return ScenarioReferenceComparison(
            scenario = scenario,
            sampleCount = pairs.size,
            systemP95Millimeters = systemP95,
            referenceP95Millimeters = referenceP95,
            combinedUncertaintyMillimeters = combined,
            statisticallyLowerErrorThanReference =
                margin > 0.0,
            inconclusiveReason =
                if (margin <= 0.0) {
                    "scenario-error-separation-overlaps-statistical-confidence"
                } else {
                    null
                },
        )
    }

    @Synchronized
    fun reportHealthEvent(event: HealthEvent) {
        check(!closed) { "Quality orchestrator is closed" }
        while (healthEvents.size >= maximumHealthEvents) {
            healthEvents.removeFirst()
        }
        healthEvents.addLast(event)
    }

    @Synchronized
    fun healthSnapshot(): HealthSnapshot {
        val rows = healthEvents.toList()
        return HealthSnapshot(
            eventCount = rows.size,
            criticalCount =
                rows.count { it.severity == FailureSeverity.CRITICAL },
            errorCount =
                rows.count { it.severity == FailureSeverity.ERROR },
            warningCount =
                rows.count { it.severity == FailureSeverity.WARNING },
            unrecoverableCount =
                rows.count { !it.recoverable },
            recentCodes =
                rows.takeLast(16).map { it.code },
            healthy =
                rows.none {
                    it.severity == FailureSeverity.CRITICAL &&
                        !it.recoverable
                },
        )
    }

    @Synchronized
    fun resetBenchmark() {
        benchmarkSamples.clear()
    }

    override fun close() {
        closed = true
        benchmarkSamples.clear()
        healthEvents.clear()
        latestDecision = null
    }

    private fun stats(
        region: MotionRegion,
        rows: List<BenchmarkSample>,
        catastrophicThresholdMillimeters: Double,
    ): RegionStatistics {
        val errors = rows.map { it.errorMillimeters }
        val p95 =
            quantile(errors, 0.95)
        val p99 =
            quantile(errors, 0.99)
        val p95Bounds =
            empiricalQuantileConfidenceBounds(
                sorted = errors.sorted(),
                quantile = 0.95,
            )
        val p99Bounds =
            empiricalQuantileConfidenceBounds(
                sorted = errors.sorted(),
                quantile = 0.99,
            )

        return RegionStatistics(
            region = region,
            sampleCount = rows.size,
            meanErrorMillimeters = errors.average(),
            medianErrorMillimeters = quantile(errors, 0.50),
            p95ErrorMillimeters = p95,
            p99ErrorMillimeters = p99,
            p95ConfidenceLowerMillimeters =
                p95Bounds.first,
            p95ConfidenceUpperMillimeters =
                p95Bounds.second,
            p99ConfidenceLowerMillimeters =
                p99Bounds.first,
            p99ConfidenceUpperMillimeters =
                p99Bounds.second,
            catastrophicErrorCount =
                errors.count {
                    it >= catastrophicThresholdMillimeters
                },
            catastrophicErrorRatio =
                errors.count {
                    it >= catastrophicThresholdMillimeters
                }.toDouble() / errors.size,
            maxErrorMillimeters = errors.maxOrNull() ?: 0.0,
            meanLagMilliseconds =
                rows.map { it.temporalLagMilliseconds }.average(),
            dropoutRatio =
                rows.count { it.confidence < 0.50 }
                    .toDouble() / rows.size,
            occludedSampleRatio =
                rows.count { it.occluded }.toDouble() / rows.size,
            identityStableRatio =
                rows.count { it.identityStable }.toDouble() / rows.size,
            confidence = rows.map { it.confidence }.average(),
        )
    }

    /**
     * Distribution-free confidence bounds for empirical quantiles.
     * The DKW inequality supplies a conservative CDF band without assuming Gaussian mocap error.
     */
    private fun empiricalQuantileConfidenceBounds(
        sorted: List<Double>,
        quantile: Double,
        alpha: Double = 0.05,
    ): Pair<Double, Double> {
        if (sorted.isEmpty()) {
            return Double.POSITIVE_INFINITY to
                Double.POSITIVE_INFINITY
        }

        val epsilon =
            sqrt(
                ln(2.0 / alpha) /
                    (2.0 * sorted.size.toDouble()),
            ).coerceIn(0.0, 0.49)

        return (
            quantile(
                sorted,
                (quantile - epsilon)
                    .coerceIn(0.0, 1.0),
            ) to
                quantile(
                    sorted,
                    (quantile + epsilon)
                        .coerceIn(0.0, 1.0),
                )
        )
    }

    private fun robustUncertainty(values: List<Double>): Double {
        if (values.size < 2) return 0.0
        val median = quantile(values, 0.50)
        val mad =
            quantile(
                values.map { abs(it - median) },
                0.50,
            )
        return (1.4826 * mad / sqrt(values.size.toDouble()))
            .coerceAtLeast(0.0)
    }

    private fun quantile(
        values: List<Double>,
        q: Double,
    ): Double {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        val position =
            q.coerceIn(0.0, 1.0) * sorted.lastIndex
        val lower = position.toInt()
        val upper = min(lower + 1, sorted.lastIndex)
        val alpha = position - lower
        return sorted[lower] * (1.0 - alpha) +
            sorted[upper] * alpha
    }
}
