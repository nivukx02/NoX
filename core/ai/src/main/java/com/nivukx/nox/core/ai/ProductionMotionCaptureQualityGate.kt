package com.nivukx.nox.core.ai

import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Evidence authority for production motion capture.
 *
 * Separates requested quality from measured quality and provides bounded runtime decisions.
 * External optical/infrared comparisons remain inconclusive until the test conditions and
 * reference uncertainty are explicitly supplied.
 */
class ProductionMotionCaptureQualityGate(
    private val maxBenchmarkSamples: Int = 8192,
    private val maxHealthEvents: Int = 2048,
) : AutoCloseable {

    init {
        require(maxBenchmarkSamples in 1..100_000) {
            "Benchmark sample retention must stay within the bounded memory envelope"
        }
        require(maxHealthEvents in 1..32_768) {
            "Health-event retention must stay within the bounded memory envelope"
        }
    }

    enum class Band { REFERENCE, ULTRA, HIGH, DEGRADED, BLOCKED }
    enum class ReferenceSystem { INFRARED_OPTICAL, MARKER_OPTICAL, RGBD, OTHER }
    enum class Region {
        ROOT, TORSO, HEAD, FACE, EYES,
        LEFT_ARM, RIGHT_ARM, LEFT_HAND, RIGHT_HAND,
        LEFT_LEG, RIGHT_LEG, FEET, FULL_BODY,
    }

    data class Contract(
        val p95TargetMillimeters: Double = 5.0,
        val p99MaximumMillimeters: Double = 12.0,
        val lagMaximumMilliseconds: Double = 25.0,
        val dropoutMaximum: Double = 0.02,
        val synchronizationMinimum: Double = 0.75,
        val synchronizationResidualP95MaximumMilliseconds: Double = 12.0,
        val synchronizationOutlierMaximum: Double = 0.20,
        val calibrationMinimum: Double = 0.75,
        val trackingMinimum: Double = 0.75,
        val crossViewMinimum: Double = 0.50,
        val viewDiversityMinimum: Double = 0.40,
        val minimumActiveCameras: Int = 1,
        val timingAuthorityMinimum: Double = 0.0,
        val timingAuthorityResidualP95MaximumMilliseconds: Double = 50.0,
        val focusConditioningMinimum: Double = 0.0,
        val referenceFocusConditioningMinimum: Double = 0.90,
        val identityMinimum: Double = 0.80,
        val queueAgeMaximumMilliseconds: Double = 25.0,
        val processingMaximumMilliseconds: Double = 12.0,
        val thermalRiskMaximum: Double = 0.82,
        val memoryPressureMaximum: Double = 0.90,
        val meshTopologyMinimum: Double = 0.92,
        val surfaceTemporalMinimum: Double = 0.80,
        val faceEyeMinimum: Double = 0.80,
        val handMinimum: Double = 0.78,
        val calibrationDriftMaximum: Double = 0.35,
        val rollingShutterMaximum: Double = 0.55,
        val occlusionRiskMaximum: Double = 0.65,
        val global4DTemporalMinimum: Double = 0.80,
        val contactConsistencyMinimum: Double = 0.80,
        val denseLandmarkCoverageMinimum: Double = 0.50,
        val denseLandmarkVisibleCoverageMinimum: Double = 0.50,
        val denseLandmarkUncertaintyP95Maximum: Double = 0.35,
        val denseLandmarkCrossViewMinimum: Double = 0.55,
        val denseLandmarkTemporalMinimum: Double = 0.55,
        val denseLandmarkContactCoverageMinimum: Double = 0.20,
        val denseLandmarkPredictionBridgeMaximum: Double = 0.70,
        val metricScaleDriftMaximumPercentPerMinute: Double = 1.0,
        val catastrophicErrorMaximumMillimeters: Double = 75.0,
        val catastrophicTailMaximumRatio: Double = 0.001,
        val minimumBenchmarkSamples: Int = 600,
        val requireDownstreamQualityForContract: Boolean = true,
    ) {
        init {
            require(p95TargetMillimeters > 0.0)
            require(p99MaximumMillimeters >= p95TargetMillimeters)
            require(lagMaximumMilliseconds > 0.0)
            require(queueAgeMaximumMilliseconds > 0.0)
            require(minimumActiveCameras in 1..128)
            require(minimumBenchmarkSamples in 120..100_000)
            require(timingAuthorityMinimum in 0.0..1.0)
            require(timingAuthorityResidualP95MaximumMilliseconds > 0.0)
            require(focusConditioningMinimum in 0.0..1.0)
            require(referenceFocusConditioningMinimum in 0.0..1.0)
            require(processingMaximumMilliseconds > 0.0)
            require(synchronizationResidualP95MaximumMilliseconds > 0.0)
            require(synchronizationOutlierMaximum in 0.0..1.0)
            listOf(
                p95TargetMillimeters,
                p99MaximumMillimeters,
                lagMaximumMilliseconds,
                synchronizationResidualP95MaximumMilliseconds,
                timingAuthorityResidualP95MaximumMilliseconds,
                queueAgeMaximumMilliseconds,
                processingMaximumMilliseconds,
                denseLandmarkUncertaintyP95Maximum,
                metricScaleDriftMaximumPercentPerMinute,
                catastrophicErrorMaximumMillimeters,
            ).forEach { threshold ->
                require(threshold.isFinite() && threshold > 0.0) {
                    "Quality thresholds must be finite and positive"
                }
            }
            require(catastrophicTailMaximumRatio.isFinite() && catastrophicTailMaximumRatio in 0.0..1.0) {
                "Catastrophic-tail ratio must be finite and in [0, 1]"
            }
            listOf(
                dropoutMaximum,
                synchronizationMinimum,
                calibrationMinimum,
                trackingMinimum,
                crossViewMinimum,
                viewDiversityMinimum,
                identityMinimum,
                thermalRiskMaximum,
                memoryPressureMaximum,
                meshTopologyMinimum,
                surfaceTemporalMinimum,
                faceEyeMinimum,
                handMinimum,
                calibrationDriftMaximum,
                rollingShutterMaximum,
                occlusionRiskMaximum,
                global4DTemporalMinimum,
                contactConsistencyMinimum,
                denseLandmarkCoverageMinimum,
                denseLandmarkVisibleCoverageMinimum,
                denseLandmarkCrossViewMinimum,
                denseLandmarkTemporalMinimum,
                denseLandmarkContactCoverageMinimum,
                denseLandmarkPredictionBridgeMaximum,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
        }
    }

    data class Evidence(
        val timestampNanos: Long,
        val synchronizationConfidence: Double = 1.0,
        val synchronizationResidualP95Milliseconds: Double = 0.0,
        val synchronizationOutlierRatio: Double = 0.0,
        val calibrationConfidence: Double = 1.0,
        val trackingConfidence: Double = 1.0,
        val crossViewAgreement: Double = 1.0,
        val viewDiversity: Double = 1.0,
        val identityContinuity: Double = 1.0,
        val p95ErrorMillimeters: Double = 0.0,
        val p99ErrorMillimeters: Double = 0.0,
        val temporalLagMilliseconds: Double = 0.0,
        val dropoutRatio: Double = 0.0,
        val queueAgeMilliseconds: Double = 0.0,
        val processingMilliseconds: Double = 0.0,
        val thermalRisk: Double = 0.0,
        val memoryPressure: Double = 0.0,
        val gpuHeadroom: Double = 1.0,
        val npuHeadroom: Double = 1.0,
        val cpuHeadroom: Double = 1.0,
        val meshTopologyIntegrity: Double = 1.0,
        val surfaceTemporalStability: Double = 1.0,
        val faceEyeConfidence: Double = 1.0,
        val handConfidence: Double = 1.0,
        val calibrationDriftRisk: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val occlusionRisk: Double = 0.0,
        val metricScaleDriftPercentPerMinute: Double = 0.0,
        val global4DTemporalConsistency: Double = 1.0,
        val contactConsistency: Double = 1.0,
        val denseLandmarkCoverage: Double = 1.0,
        val denseLandmarkVisibleCoverage: Double = 1.0,
        val denseLandmarkUncertaintyP95: Double = 0.0,
        val denseLandmarkCrossViewAgreement: Double = 1.0,
        val denseLandmarkTemporalStability: Double = 1.0,
        val denseLandmarkContactCoverage: Double = 1.0,
        val denseLandmarkPredictionBridgeRatio: Double = 0.0,
        val timingAuthorityConfidence: Double = 0.0,
        val timingAuthorityResidualP95Milliseconds: Double = 0.0,
        val focusConditioningScore: Double = 1.0,
        val focusConditioningReferenceCompatible: Boolean = true,
        val denseLandmarkCrossViewEvidenceAvailable: Boolean = false,
        val denseLandmarkHardFailure: Boolean = false,
        val activeCameras: Int = 1,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(activeCameras >= 0)
            listOf(
                synchronizationConfidence,
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
                meshTopologyIntegrity,
                surfaceTemporalStability,
                faceEyeConfidence,
                handConfidence,
                calibrationDriftRisk,
                rollingShutterRisk,
                occlusionRisk,
                denseLandmarkCoverage,
                denseLandmarkVisibleCoverage,
                denseLandmarkCrossViewAgreement,
                denseLandmarkTemporalStability,
                denseLandmarkContactCoverage,
                timingAuthorityConfidence,
                focusConditioningScore,
            ).forEach { require(it.isFinite() && it in 0.0..1.0) }
            require(
                synchronizationResidualP95Milliseconds.isFinite() &&
                    synchronizationResidualP95Milliseconds >= 0.0,
            )
            require(
                synchronizationOutlierRatio.isFinite() &&
                    synchronizationOutlierRatio in 0.0..1.0,
            )
            listOf(
                p95ErrorMillimeters,
                p99ErrorMillimeters,
                timingAuthorityResidualP95Milliseconds,
                temporalLagMilliseconds,
                queueAgeMilliseconds,
                processingMilliseconds,
                denseLandmarkUncertaintyP95,
                denseLandmarkPredictionBridgeRatio,
            ).forEach { require(it.isFinite() && it >= 0.0) }
        }

        val computeHeadroom: Double
            get() = minOf(
                gpuHeadroom,
                npuHeadroom,
                cpuHeadroom,
                1.0 - memoryPressure,
            )

        val evidenceScore: Double
            get() = (
                synchronizationConfidence * 0.08 +
                    (
                        1.0 -
                            (
                                synchronizationResidualP95Milliseconds /
                                    4.0
                            ).coerceIn(0.0, 1.0)
                        ) * 0.03 +
                    (
                        1.0 -
                            synchronizationOutlierRatio * 2.5
                        ).coerceIn(0.0, 1.0) * 0.02 +
                    calibrationConfidence * 0.13 +
                    trackingConfidence * 0.13 +
                    crossViewAgreement * 0.10 +
                    viewDiversity * 0.07 +
                    identityContinuity * 0.08 +
                    surfaceTemporalStability * 0.06 +
                    faceEyeConfidence * 0.05 +
                    handConfidence * 0.05 +
                    meshTopologyIntegrity * 0.05 +
                    (1.0 - calibrationDriftRisk) * 0.02 +
                    (1.0 - rollingShutterRisk) * 0.01 +
                    (1.0 - occlusionRisk) * 0.01 +
                    global4DTemporalConsistency * 0.05 +
                    contactConsistency * 0.04 +
                    focusConditioningScore * 0.025 +
                    denseLandmarkCoverage * 0.035 +
                    denseLandmarkVisibleCoverage * 0.025 +
                    denseLandmarkCrossViewAgreement * 0.035 +
                    denseLandmarkTemporalStability * 0.030 +
                    denseLandmarkContactCoverage * 0.015 +
                    (1.0 - denseLandmarkUncertaintyP95) * 0.030 +
                    (1.0 - denseLandmarkPredictionBridgeRatio) * 0.010 +
                    (1.0 - (metricScaleDriftPercentPerMinute /
                        5.0).coerceIn(0.0, 1.0)) * 0.05
                ).coerceIn(0.0, 1.0)
    }

    data class RuntimeBudget(
        val detectorScale: Double,
        val depthScale: Double,
        val triangulationScale: Double,
        val temporalFusionScale: Double,
        val faceEyeScale: Double,
        val handScale: Double,
        val surfaceScale: Double,
        val diagnosticsScale: Double,
        val recommendedCameraSelectionCount: Int,
        val thermalReserve: Double,
        val memoryReserve: Double,
    )

    data class Decision(
        val band: Band,
        val allowed: Boolean,
        val contractSatisfied: Boolean,
        val hardBlock: Boolean,
        val score: Double,
        val confidence: Double,
        val adaptiveScale: Double,
        val warnings: List<String>,
        val blockers: List<String>,
        val recommendedFrameStride: Int,
        val recommendedWorkers: Int,
        val runtimeBudget: RuntimeBudget = RuntimeBudget(
            detectorScale = 1.0,
            depthScale = 1.0,
            triangulationScale = 1.0,
            temporalFusionScale = 1.0,
            faceEyeScale = 1.0,
            handScale = 1.0,
            surfaceScale = 1.0,
            diagnosticsScale = 0.50,
            recommendedCameraSelectionCount = 1,
            thermalReserve = 1.0,
            memoryReserve = 1.0,
        ),
    )

    data class BenchmarkSample(
        val timestampNanos: Long,
        val region: Region,
        val systemErrorMillimeters: Double,
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
            require(systemErrorMillimeters.isFinite() && systemErrorMillimeters >= 0.0)
            require(temporalLagMilliseconds.isFinite() && temporalLagMilliseconds >= 0.0)
            require(visibility.isFinite() && visibility in 0.0..1.0)
            require(identityExchangeRisk.isFinite() && identityExchangeRisk in 0.0..1.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            referenceErrorMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class Statistics(
        val sampleCount: Int,
        val meanMillimeters: Double,
        val medianMillimeters: Double,
        val p95Millimeters: Double,
        val p99Millimeters: Double,
        val maxMillimeters: Double,
        val meanLagMilliseconds: Double,
        val dropoutRatio: Double,
        val occludedRatio: Double,
        val identityStableRatio: Double,
    )

    data class ReferenceComparison(
        val system: ReferenceSystem,
        val matchedConditions: Boolean,
        val referenceP95Millimeters: Double,
        val referenceUncertaintyMillimeters: Double,
        val systemP95Millimeters: Double,
        val systemUncertaintyMillimeters: Double,
        val statisticallySeparated: Boolean,
        val inconclusiveReason: String?,
        val combinedUncertaintyMillimeters: Double = 0.0,
        val effectMarginMillimeters: Double = 0.0,
        val separationRatio: Double = 0.0,
        val pairedSampleCount: Int = 0,
        val pairedSystemWinRate: Double = 0.0,
        val pairedMedianDeltaMillimeters: Double = 0.0,
    )

    data class BenchmarkReport(
        val sampleCount: Int,
        val overall: Statistics,
        val perRegion: Map<Region, Statistics>,
        val lowVisibilityP95Millimeters: Double,
        val occludedP95Millimeters: Double,
        val referenceComparison: ReferenceComparison?,
        val gatePassed: Boolean,
        val lowVisibilityRatio: Double = 0.0,
        val severeOcclusionRatio: Double = 0.0,
        val identityFailureRatio: Double = 0.0,
        val interactionP95Millimeters: Double = 0.0,
        val fastMotionP95Millimeters: Double = 0.0,
        val multiPersonOcclusionP95Millimeters: Double = 0.0,
        val interactionSampleRatio: Double = 0.0,
        val fastMotionSampleRatio: Double = 0.0,
        val multiPersonOcclusionSampleRatio: Double = 0.0,
        val identityExchangeFailureRatio: Double = 0.0,
        val catastrophicTailRatio: Double = 0.0,
        val catastrophicErrorMaximumMillimeters: Double = 0.0,
    )

    data class HealthEvent(
        val timestampNanos: Long,
        val code: String,
        val critical: Boolean,
        val recoverable: Boolean,
    )

    data class Health(
        val events: Int,
        val critical: Int,
        val unrecoverable: Int,
        val latestCodes: List<String>,
        val healthy: Boolean,
    )

    private val benchmark = ArrayDeque<BenchmarkSample>(maxBenchmarkSamples)
    private val health = ArrayDeque<HealthEvent>(maxHealthEvents)
    @Volatile private var closed = false
    @Volatile private var latest: Decision? = null

    @Synchronized
    fun evaluate(evidence: Evidence, contract: Contract = Contract()): Decision {
        check(!closed)

        val warnings = mutableListOf<String>()
        val blockers = mutableListOf<String>()

        fun gate(
            value: Double,
            threshold: Double,
            warningCode: String,
            blockCode: String? = null,
        ) {
            if (blockCode != null && value < threshold && value < 0.35) {
                blockers += blockCode
            } else if (value < threshold) {
                warnings += warningCode
            }
        }

        gate(
            evidence.synchronizationConfidence,
            contract.synchronizationMinimum,
            "sync-below-contract",
            "sync-confidence-critical",
        )
        if (
            evidence.synchronizationResidualP95Milliseconds >
                contract.synchronizationResidualP95MaximumMilliseconds
        ) {
            warnings += "sync-residual-p95-above-contract"
        }
        if (
            evidence.synchronizationOutlierRatio >
                contract.synchronizationOutlierMaximum
        ) {
            warnings += "sync-outlier-ratio-above-contract"
        }
        if (
            evidence.synchronizationResidualP95Milliseconds >
                contract.synchronizationResidualP95MaximumMilliseconds * 2.0 ||
            evidence.synchronizationOutlierRatio >
                contract.synchronizationOutlierMaximum * 2.0
        ) {
            blockers += "sync-evidence-critical"
        }
        gate(
            evidence.calibrationConfidence,
            contract.calibrationMinimum,
            "calibration-below-contract",
            "calibration-confidence-critical",
        )
        gate(
            evidence.trackingConfidence,
            contract.trackingMinimum,
            "tracking-below-contract",
            "tracking-confidence-critical",
        )

        if (evidence.activeCameras < contract.minimumActiveCameras) {
            blockers += "active-camera-count-below-contract"
        }
        if (
            evidence.focusConditioningScore <
                contract.focusConditioningMinimum
        ) {
            warnings += "focus-conditioning-below-contract"
        }

        if (
            evidence.focusConditioningScore <
                contract.focusConditioningMinimum * 0.60 &&
            contract.focusConditioningMinimum > 0.0
        ) {
            blockers += "focus-conditioning-critical"
        }

        if (
            contract.referenceFocusConditioningMinimum >
                0.0 &&
            evidence.focusConditioningScore <
                contract.referenceFocusConditioningMinimum
        ) {
            warnings += "focus-conditioning-below-reference-target"
        }

        if (
            !evidence.focusConditioningReferenceCompatible &&
            contract.referenceFocusConditioningMinimum > 0.0
        ) {
            blockers += "focus-state-mismatched-for-reference"
        }

        if (
            evidence.timingAuthorityConfidence <
                contract.timingAuthorityMinimum
        ) {
            warnings += "timing-authority-below-contract"
        }
        if (
            evidence.timingAuthorityResidualP95Milliseconds >
                contract.timingAuthorityResidualP95MaximumMilliseconds
        ) {
            warnings += "timing-authority-residual-above-contract"
        }
        if (
            evidence.timingAuthorityConfidence <
                contract.timingAuthorityMinimum * 0.60 &&
            contract.timingAuthorityMinimum > 0.0
        ) {
            blockers += "timing-authority-critical"
        }
        if (evidence.crossViewAgreement < contract.crossViewMinimum) {
            warnings += "cross-view-agreement-below-contract"
        }
        if (evidence.viewDiversity < contract.viewDiversityMinimum) {
            warnings += "view-diversity-below-contract"
        }
        if (evidence.identityContinuity < contract.identityMinimum) {
            warnings += "identity-continuity-below-contract"
        }

        if (evidence.p95ErrorMillimeters > contract.p95TargetMillimeters) {
            warnings += "p95-error-above-contract"
        }
        if (evidence.p95ErrorMillimeters > contract.p99MaximumMillimeters * 1.8) {
            blockers += "geometric-error-critical"
        }
        if (evidence.p99ErrorMillimeters > contract.p99MaximumMillimeters) {
            warnings += "p99-error-above-contract"
        }

        if (evidence.temporalLagMilliseconds > contract.lagMaximumMilliseconds) {
            warnings += "temporal-lag-above-contract"
        }
        if (evidence.temporalLagMilliseconds > contract.lagMaximumMilliseconds * 2.0) {
            blockers += "temporal-lag-critical"
        }
        if (evidence.dropoutRatio > contract.dropoutMaximum) {
            warnings += "dropout-above-contract"
        }
        if (evidence.dropoutRatio > contract.dropoutMaximum * 4.0) {
            blockers += "dropout-critical"
        }

        if (evidence.queueAgeMilliseconds > contract.queueAgeMaximumMilliseconds) {
            warnings += "native-queue-residence-high"
        }
        if (evidence.queueAgeMilliseconds > contract.queueAgeMaximumMilliseconds * 2.0) {
            blockers += "native-queue-critical"
        }
        if (evidence.processingMilliseconds > contract.processingMaximumMilliseconds) {
            warnings += "native-processing-high"
        }
        if (evidence.processingMilliseconds > contract.processingMaximumMilliseconds * 2.0) {
            blockers += "native-processing-critical"
        }

        if (evidence.thermalRisk > contract.thermalRiskMaximum) {
            warnings += "thermal-risk-high"
        }
        if (evidence.thermalRisk >= 0.97) {
            blockers += "thermal-risk-critical"
        }
        if (evidence.memoryPressure > contract.memoryPressureMaximum) {
            warnings += "memory-pressure-high"
        }
        if (evidence.memoryPressure >= 0.98) {
            blockers += "memory-pressure-critical"
        }

        if (evidence.meshTopologyIntegrity < contract.meshTopologyMinimum) {
            warnings += "mesh-topology-risk"
        }
        if (evidence.surfaceTemporalStability < contract.surfaceTemporalMinimum) {
            warnings += "surface-temporal-instability"
        }
        if (evidence.faceEyeConfidence < contract.faceEyeMinimum) {
            warnings += "face-eye-confidence-low"
        }
        if (evidence.handConfidence < contract.handMinimum) {
            warnings += "hand-confidence-low"
        }
        if (evidence.calibrationDriftRisk > contract.calibrationDriftMaximum) {
            warnings += "calibration-drift-risk-rising"
        }
        if (evidence.rollingShutterRisk > contract.rollingShutterMaximum) {
            warnings += "rolling-shutter-risk-rising"
        }
        if (evidence.occlusionRisk > contract.occlusionRiskMaximum) {
            warnings += "occlusion-risk-high"
        }
        if (evidence.global4DTemporalConsistency < 0.80) {
            warnings += "global-4d-temporal-consistency-low"
        }
        if (evidence.contactConsistency < contract.contactConsistencyMinimum) {
            warnings += "contact-consistency-low"
        }
        if (evidence.denseLandmarkHardFailure) {
            blockers += "dense-landmark-hard-failure"
        }
        if (
            evidence.denseLandmarkCoverage <
                contract.denseLandmarkCoverageMinimum
        ) {
            blockers += "dense-landmark-coverage-critical"
        } else if (
            evidence.denseLandmarkCoverage <
                contract.denseLandmarkCoverageMinimum + 0.10
        ) {
            warnings += "dense-landmark-coverage-low"
        }
        if (
            evidence.denseLandmarkVisibleCoverage <
                contract.denseLandmarkVisibleCoverageMinimum
        ) {
            blockers += "dense-landmark-visible-coverage-critical"
        }
        if (
            evidence.denseLandmarkUncertaintyP95 >
                contract.denseLandmarkUncertaintyP95Maximum
        ) {
            blockers += "dense-landmark-uncertainty-critical"
        }
        if (
            evidence.denseLandmarkCrossViewEvidenceAvailable &&
                evidence.denseLandmarkCrossViewAgreement <
                    contract.denseLandmarkCrossViewMinimum
        ) {
            blockers += "dense-landmark-cross-view-critical"
        }
        if (
            evidence.denseLandmarkTemporalStability <
                contract.denseLandmarkTemporalMinimum
        ) {
            warnings += "dense-landmark-temporal-instability"
        }
        if (
            evidence.denseLandmarkContactCoverage <
                contract.denseLandmarkContactCoverageMinimum
        ) {
            warnings += "dense-landmark-contact-sparse"
        }
        if (
            evidence.denseLandmarkPredictionBridgeRatio >
                contract.denseLandmarkPredictionBridgeMaximum
        ) {
            warnings += "dense-landmark-prediction-heavy"
        }
        if (
            evidence.metricScaleDriftPercentPerMinute >
                contract.metricScaleDriftMaximumPercentPerMinute
        ) {
            warnings += "metric-scale-drift-high"
        }

        val contractSatisfied =
            blockers.isEmpty() &&
                evidence.p95ErrorMillimeters <= contract.p95TargetMillimeters &&
                evidence.p99ErrorMillimeters <= contract.p99MaximumMillimeters &&
                evidence.temporalLagMilliseconds <= contract.lagMaximumMilliseconds &&
                evidence.dropoutRatio <= contract.dropoutMaximum &&
                evidence.synchronizationConfidence >= contract.synchronizationMinimum &&
                evidence.synchronizationResidualP95Milliseconds <=
                    contract.synchronizationResidualP95MaximumMilliseconds &&
                evidence.synchronizationOutlierRatio <=
                    contract.synchronizationOutlierMaximum &&
                evidence.calibrationConfidence >= contract.calibrationMinimum &&
                evidence.trackingConfidence >= contract.trackingMinimum &&
                evidence.crossViewAgreement >= contract.crossViewMinimum &&
                evidence.viewDiversity >= contract.viewDiversityMinimum &&
                evidence.activeCameras >= contract.minimumActiveCameras &&
                evidence.timingAuthorityConfidence >= contract.timingAuthorityMinimum &&
                evidence.timingAuthorityResidualP95Milliseconds <=
                    contract.timingAuthorityResidualP95MaximumMilliseconds &&
                evidence.focusConditioningScore >=
                    if (contract.referenceFocusConditioningMinimum > 0.0) {
                        contract.referenceFocusConditioningMinimum
                    } else {
                        contract.focusConditioningMinimum
                    } &&
                (
                    contract.referenceFocusConditioningMinimum <= 0.0 ||
                        evidence.focusConditioningReferenceCompatible
                ) &&
                evidence.identityContinuity >= contract.identityMinimum &&
                (
                    !contract.requireDownstreamQualityForContract ||
                        (
                            evidence.meshTopologyIntegrity >= contract.meshTopologyMinimum &&
                            evidence.surfaceTemporalStability >= contract.surfaceTemporalMinimum &&
                            evidence.faceEyeConfidence >= contract.faceEyeMinimum &&
                            evidence.handConfidence >= contract.handMinimum &&
                            evidence.calibrationDriftRisk <= contract.calibrationDriftMaximum &&
                            evidence.rollingShutterRisk <= contract.rollingShutterMaximum &&
                            evidence.occlusionRisk <= contract.occlusionRiskMaximum &&
                            evidence.global4DTemporalConsistency >= contract.global4DTemporalMinimum &&
                            evidence.contactConsistency >= contract.contactConsistencyMinimum &&
                            evidence.denseLandmarkCoverage >= contract.denseLandmarkCoverageMinimum &&
                            evidence.denseLandmarkVisibleCoverage >= contract.denseLandmarkVisibleCoverageMinimum &&
                            evidence.denseLandmarkUncertaintyP95 <= contract.denseLandmarkUncertaintyP95Maximum &&
                            evidence.denseLandmarkTemporalStability >= contract.denseLandmarkTemporalMinimum &&
                            evidence.denseLandmarkContactCoverage >= contract.denseLandmarkContactCoverageMinimum &&
                            evidence.denseLandmarkPredictionBridgeRatio <= contract.denseLandmarkPredictionBridgeMaximum &&
                            (
                                !evidence.denseLandmarkCrossViewEvidenceAvailable ||
                                    evidence.denseLandmarkCrossViewAgreement >=
                                        contract.denseLandmarkCrossViewMinimum
                                ) &&
                            !evidence.denseLandmarkHardFailure &&
                            evidence.metricScaleDriftPercentPerMinute <=
                                contract.metricScaleDriftMaximumPercentPerMinute
                        )
                )

        val pressure = maxOf(
            evidence.thermalRisk,
            evidence.memoryPressure,
            1.0 - evidence.computeHeadroom,
        ).coerceIn(0.0, 1.0)
        val accuracyPressure = (
            evidence.p95ErrorMillimeters /
                contract.p95TargetMillimeters
        ).coerceIn(0.0, 3.0) / 3.0
        val score = (
            evidence.evidenceScore * 0.76 +
                (1.0 - accuracyPressure) * 0.14 +
                (1.0 - pressure) * 0.10
        ).coerceIn(0.0, 1.0)

        val scale = when {
            blockers.isNotEmpty() -> 0.50
            pressure >= 0.90 -> 0.58
            pressure >= 0.65 -> 0.72
            pressure >= 0.45 -> 0.86
            else -> 1.0
        }

        val computeScale = evidence.computeHeadroom.coerceIn(0.0, 1.0)
        val cameraSelectionCount = evidence.activeCameras
            .coerceAtLeast(1)
            .coerceAtMost(32)
        val runtimeBudget = RuntimeBudget(
            detectorScale = (0.78 * scale).coerceIn(0.35, 1.0),
            depthScale = (
                0.72 * scale +
                    evidence.viewDiversity * 0.18
            ).coerceIn(0.32, 1.0),
            triangulationScale = (
                0.84 * scale +
                    evidence.crossViewAgreement * 0.16
            ).coerceIn(0.40, 1.0),
            temporalFusionScale = (
                0.92 * scale +
                    evidence.identityContinuity * 0.08
            ).coerceIn(0.55, 1.0),
            faceEyeScale = (
                0.90 * scale +
                    evidence.faceEyeConfidence * 0.10
            ).coerceIn(0.50, 1.0),
            handScale = (
                0.86 * scale +
                    evidence.handConfidence * 0.14
            ).coerceIn(0.45, 1.0),
            surfaceScale = (
                0.82 * scale +
                    evidence.surfaceTemporalStability * 0.18
            ).coerceIn(0.40, 1.0),
            diagnosticsScale = (
                0.35 + evidence.evidenceScore * 0.45
            ).coerceIn(0.20, 0.80),
            recommendedCameraSelectionCount = cameraSelectionCount,
            thermalReserve = (1.0 - evidence.thermalRisk)
                .coerceIn(0.0, 1.0),
            memoryReserve = (1.0 - evidence.memoryPressure)
                .coerceIn(0.0, 1.0),
        )

        val decision = Decision(
            band = when {
                blockers.isNotEmpty() -> Band.BLOCKED
                contractSatisfied && score >= 0.90 -> Band.REFERENCE
                score >= 0.78 -> Band.ULTRA
                score >= 0.60 -> Band.HIGH
                else -> Band.DEGRADED
            },
            allowed = blockers.isEmpty(),
            contractSatisfied = contractSatisfied,
            hardBlock = blockers.isNotEmpty(),
            score = score,
            confidence = evidence.evidenceScore,
            adaptiveScale = scale,
            warnings = warnings.distinct(),
            blockers = blockers.distinct(),
            recommendedFrameStride = when {
                blockers.isNotEmpty() -> 3
                pressure >= 0.65 -> 2
                else -> 1
            },
            recommendedWorkers = (
                1.0 + 15.0 * computeScale
            ).toInt().coerceIn(1, 16),
            runtimeBudget = runtimeBudget,
        )
        latest = decision
        return decision
    }

    fun latestDecision(): Decision? = latest

    @Synchronized
    fun addBenchmarkSample(sample: BenchmarkSample) {
        check(!closed)
        while (benchmark.size >= maxBenchmarkSamples) benchmark.removeFirst()
        benchmark.addLast(sample)
    }

    @Synchronized
    fun benchmarkSize(): Int = benchmark.size

    @Synchronized
    fun completeBenchmark(
        referenceSystem: ReferenceSystem? = null,
        referenceP95Millimeters: Double? = null,
        referenceUncertaintyMillimeters: Double = 0.0,
        matchedConditions: Boolean = false,
        contract: Contract = Contract(),
    ): BenchmarkReport {
        check(!closed)
        require(referenceUncertaintyMillimeters >= 0.0)
        val rows = benchmark.toList()
        require(rows.isNotEmpty())

        val overall = statistics(rows)
        val byRegion = rows.groupBy { it.region }
            .mapValues { statistics(it.value) }
        val lowVisibility = rows.filter { it.visibility < 0.5 }
        val occluded = rows.filter { it.occluded }
        val interaction = rows.filter { it.interaction }
        val fastMotion = rows.filter { it.fastMotion }
        val multiPersonOcclusion =
            rows.filter { it.multiPersonOcclusion }
        val identityExchangeRows =
            rows.filter { it.identityExchangeRisk >= 0.50 }

        val catastrophicTailRatio =
            rows.count {
                it.systemErrorMillimeters >
                    contract.catastrophicErrorMaximumMillimeters
            }.toDouble() /
                rows.size.toDouble()

        val pairedRows = rows.filter {
            it.referenceErrorMillimeters != null
        }
        val pairedWinRate =
            if (pairedRows.isEmpty()) {
                0.0
            } else {
                pairedRows.count {
                    it.systemErrorMillimeters < it.referenceErrorMillimeters!!
                }.toDouble() / pairedRows.size
            }
        val pairedMedianDelta =
            if (pairedRows.isEmpty()) {
                0.0
            } else {
                quantile(
                    pairedRows.map {
                        it.systemErrorMillimeters -
                            it.referenceErrorMillimeters!!
                    },
                    0.50,
                )
            }

        val reference = if (
            referenceSystem != null &&
            referenceP95Millimeters != null
        ) {
            val systemUncertainty = bootstrapP95StandardError(
                rows.map { it.systemErrorMillimeters },
            )
            val combinedUncertainty = sqrt(
                systemUncertainty * systemUncertainty +
                    referenceUncertaintyMillimeters *
                    referenceUncertaintyMillimeters,
            )
            val effectMargin =
                referenceP95Millimeters - overall.p95Millimeters
            val separationRatio =
                if (combinedUncertainty <= 1e-12) {
                    if (effectMargin > 0.0) {
                        Double.POSITIVE_INFINITY
                    } else {
                        0.0
                    }
                } else {
                    effectMargin / combinedUncertainty
                }
            val reason = when {
                !matchedConditions -> "reference-conditions-not-matched"
                rows.size < 120 -> "benchmark-too-small"
                referenceUncertaintyMillimeters <= 0.0 ->
                    "reference-uncertainty-not-supplied"
                else -> null
            }
            ReferenceComparison(
                system = referenceSystem,
                matchedConditions = matchedConditions,
                referenceP95Millimeters = referenceP95Millimeters,
                referenceUncertaintyMillimeters =
                    referenceUncertaintyMillimeters,
                systemP95Millimeters = overall.p95Millimeters,
                systemUncertaintyMillimeters = systemUncertainty,
                statisticallySeparated =
                    reason == null &&
                        separationRatio >= 1.96,
                inconclusiveReason = reason,
                combinedUncertaintyMillimeters =
                    combinedUncertainty,
                effectMarginMillimeters = effectMargin,
                separationRatio = separationRatio,
                pairedSampleCount = pairedRows.size,
                pairedSystemWinRate = pairedWinRate,
                pairedMedianDeltaMillimeters = pairedMedianDelta,
            )
        } else {
            null
        }

        val referenceGateSatisfied =
            reference?.let {
                it.matchedConditions &&
                    it.statisticallySeparated
            } ?: true

        val gatePassed =
            rows.size >= contract.minimumBenchmarkSamples &&
                overall.p95Millimeters <= contract.p95TargetMillimeters &&
                overall.p99Millimeters <= contract.p99MaximumMillimeters &&
                overall.meanLagMilliseconds <= contract.lagMaximumMilliseconds &&
                overall.dropoutRatio <= contract.dropoutMaximum &&
                overall.identityStableRatio >= contract.identityMinimum &&
                catastrophicTailRatio <= contract.catastrophicTailMaximumRatio &&
                overall.maxMillimeters <= contract.catastrophicErrorMaximumMillimeters &&
                referenceGateSatisfied

        return BenchmarkReport(
            sampleCount = rows.size,
            overall = overall,
            perRegion = byRegion,
            lowVisibilityP95Millimeters =
                lowVisibility.takeIf { it.isNotEmpty() }
                    ?.let { statistics(it).p95Millimeters } ?: 0.0,
            occludedP95Millimeters =
                occluded.takeIf { it.isNotEmpty() }
                    ?.let { statistics(it).p95Millimeters } ?: 0.0,
            referenceComparison = reference,
            gatePassed = gatePassed,
            lowVisibilityRatio =
                rows.count { it.visibility < 0.50 }
                    .toDouble() / rows.size,
            severeOcclusionRatio =
                rows.count { it.occluded }
                    .toDouble() / rows.size,
            identityFailureRatio =
                rows.count { !it.identityStable }
                    .toDouble() / rows.size,
            interactionP95Millimeters =
                interaction.takeIf { it.isNotEmpty() }
                    ?.let { statistics(it).p95Millimeters } ?: 0.0,
            fastMotionP95Millimeters =
                fastMotion.takeIf { it.isNotEmpty() }
                    ?.let { statistics(it).p95Millimeters } ?: 0.0,
            multiPersonOcclusionP95Millimeters =
                multiPersonOcclusion.takeIf { it.isNotEmpty() }
                    ?.let { statistics(it).p95Millimeters } ?: 0.0,
            interactionSampleRatio =
                interaction.size.toDouble() / rows.size,
            fastMotionSampleRatio =
                fastMotion.size.toDouble() / rows.size,
            multiPersonOcclusionSampleRatio =
                multiPersonOcclusion.size.toDouble() / rows.size,
            identityExchangeFailureRatio =
                if (identityExchangeRows.isEmpty()) {
                    0.0
                } else {
                    identityExchangeRows.count { !it.identityStable }
                        .toDouble() / identityExchangeRows.size
                },
            catastrophicTailRatio = catastrophicTailRatio,
            catastrophicErrorMaximumMillimeters =
                overall.maxMillimeters,
        )
    }

    @Synchronized
    fun resetBenchmark() {
        benchmark.clear()
    }

    @Synchronized
    fun recordHealth(event: HealthEvent) {
        check(!closed)
        while (health.size >= maxHealthEvents) health.removeFirst()
        health.addLast(event)
    }

    @Synchronized
    fun health(): Health {
        val rows = health.toList()
        return Health(
            events = rows.size,
            critical = rows.count { it.critical },
            unrecoverable = rows.count { it.critical && !it.recoverable },
            latestCodes = rows.takeLast(16).map { it.code }.distinct(),
            healthy = rows.none { it.critical && !it.recoverable },
        )
    }

    override fun close() {
        closed = true
        benchmark.clear()
        health.clear()
        latest = null
    }

    private fun statistics(rows: List<BenchmarkSample>): Statistics {
        val errors = rows.map { it.systemErrorMillimeters }
        return Statistics(
            sampleCount = rows.size,
            meanMillimeters = errors.average(),
            medianMillimeters = quantile(errors, 0.50),
            p95Millimeters = quantile(errors, 0.95),
            p99Millimeters = quantile(errors, 0.99),
            maxMillimeters = errors.maxOrNull() ?: 0.0,
            meanLagMilliseconds =
                rows.map { it.temporalLagMilliseconds }.average(),
            dropoutRatio =
                rows.count { it.confidence < 0.50 }.toDouble() / rows.size,
            occludedRatio =
                rows.count { it.occluded }.toDouble() / rows.size,
            identityStableRatio =
                rows.count { it.identityStable }.toDouble() / rows.size,
        )
    }

    /**
     * Estimates uncertainty of the reported P95 itself rather than the mean of the
     * underlying error samples. This matters because the qualification boundary is
     * explicitly a tail-quantile claim.
     *
     * The resampling stream is deterministic so benchmark reports are reproducible
     * across devices and repeated qualification runs. It is deliberately bounded:
     * benchmark completion is an operator/qualification path, not the per-frame path.
     */
    private fun bootstrapP95StandardError(
        values: List<Double>,
        replicates: Int = 512,
    ): Double {
        if (values.size < 2) return 0.0
        require(replicates > 0)

        var state = 0x9E3779B9.toInt()
        val estimates = ArrayList<Double>(replicates)

        repeat(replicates) {
            val resample = ArrayList<Double>(values.size)
            repeat(values.size) {
                state = xorshift32(state)
                val index = (state.toUInt().toLong() and 0x7FFFFFFFL)
                    .mod(values.size.toLong())
                    .toInt()
                resample += values[index]
            }
            estimates += quantile(resample, 0.95)
        }

        return robustStandardError(estimates)
    }

    private fun xorshift32(input: Int): Int {
        var value = input
        value = value xor (value shl 13)
        value = value xor (value ushr 17)
        value = value xor (value shl 5)
        return value
    }

    private fun robustStandardError(values: List<Double>): Double {
        if (values.size < 2) return 0.0
        val median = quantile(values, 0.50)
        val mad = quantile(
            values.map { abs(it - median) },
            0.50,
        )
        return (1.4826 * mad / sqrt(values.size.toDouble()))
            .coerceAtLeast(0.0)
    }

    private fun quantile(values: List<Double>, q: Double): Double {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        val position = q.coerceIn(0.0, 1.0) * sorted.lastIndex
        val lower = position.toInt()
        val upper = (lower + 1).coerceAtMost(sorted.lastIndex)
        val alpha = position - lower
        return sorted[lower] * (1.0 - alpha) + sorted[upper] * alpha
    }
}
