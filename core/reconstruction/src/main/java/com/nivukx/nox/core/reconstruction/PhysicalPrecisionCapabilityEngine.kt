package com.nivukx.nox.core.reconstruction

import java.util.ArrayDeque

/**
 * Live physical-precision admission layer.
 *
 * This is an operational uncertainty envelope, not a claim of measured accuracy.
 * Sub-millimetre qualification requires both live evidence and a matched external
 * reference cohort. No confidence score is converted directly into a distance.
 */
class PhysicalPrecisionCapabilityEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class State { NO_DATA, EVIDENCE_ONLY, HIGH_PRECISION, SUB_MILLIMETER_QUALIFIED, BLOCKED }

    enum class Bottleneck {
        NONE, CALIBRATION, SYNCHRONIZATION, TEMPORAL_ALIGNMENT,
        GEOMETRIC_OBSERVABILITY, CROSS_VIEW_DIVERSITY,
        RECONSTRUCTION_UNCERTAINTY, SURFACE_UNCERTAINTY,
        MOTION_UNCERTAINTY, CAPTURE_TIMING, RUNTIME_HEALTH, EMPIRICAL_REFERENCE,
    }

    enum class EvidenceLevel { NONE, LIVE_ONLY, EMPIRICAL, EMPIRICAL_AND_LIVE }

    data class Config(
        val highPrecisionBoundMillimeters: Double = 2.0,
        val subMillimeterBoundMillimeters: Double = 1.0,
        val calibrationMinimum: Double = 0.96,
        val synchronizationMinimum: Double = 0.96,
        val temporalAlignmentMinimum: Double = 0.94,
        val crossViewAgreementMinimum: Double = 0.86,
        val viewDiversityMinimum: Double = 0.78,
        val captureTimingMinimum: Double = 0.93,
        val minimumCameraCountForSubMillimeter: Int = 3,
        val minimumCalibrationUncertaintyCoverage: Double = 0.90,
        val minimumCalibrationUncertaintyEvidenceConfidence: Double = 0.85,
        val maximumTemporalSpreadMillis: Double = 4.0,
        val maximumCaptureJitterMillis: Double = 3.0,
        val minimumIdentityContinuity: Double = 0.92,
        val minimumHardwareReadoutTimingConfidence: Double = 0.90,
        val maximumHardwareReadoutP95JitterMillis: Double = 2.0,
        val empiricalP95LimitMillimeters: Double = 1.0,
        val empiricalReferenceLimitMillimeters: Double = 0.25,
        val minimumEmpiricalSamples: Int = 240,
        val minimumIndependentSessions: Int = 3,
        val maximumHistory: Int = 128,    ) {
        init {
            require(highPrecisionBoundMillimeters.isFinite() && highPrecisionBoundMillimeters > 0.0) {
                "High-precision bound must be finite and positive"
            }
            require(
                subMillimeterBoundMillimeters.isFinite() &&
                    subMillimeterBoundMillimeters > 0.0 &&
                    subMillimeterBoundMillimeters <= highPrecisionBoundMillimeters,
            ) {
                "Sub-millimetre bound must be positive and no greater than the high-precision bound"
            }

            listOf(
                calibrationMinimum,
                synchronizationMinimum,
                temporalAlignmentMinimum,
                crossViewAgreementMinimum,
                viewDiversityMinimum,
                captureTimingMinimum,
                minimumCalibrationUncertaintyCoverage,
                minimumCalibrationUncertaintyEvidenceConfidence,
                minimumIdentityContinuity,
                minimumHardwareReadoutTimingConfidence,
            ).forEach { threshold ->
                require(threshold.isFinite() && threshold in 0.0..1.0) {
                    "Precision evidence thresholds must be finite values in [0, 1]"
                }
            }

            require(minimumCameraCountForSubMillimeter in 2..128) {
                "Sub-millimetre qualification requires a bounded multi-camera configuration"
            }
            require(maximumTemporalSpreadMillis.isFinite() && maximumTemporalSpreadMillis > 0.0) {
                "Maximum temporal spread must be finite and positive"
            }
            require(maximumCaptureJitterMillis.isFinite() && maximumCaptureJitterMillis > 0.0) {
                "Maximum capture jitter must be finite and positive"
            }
            require(
                maximumHardwareReadoutP95JitterMillis.isFinite() &&
                    maximumHardwareReadoutP95JitterMillis > 0.0,
            ) {
                "Hardware readout jitter limit must be finite and positive"
            }
            require(empiricalP95LimitMillimeters.isFinite() && empiricalP95LimitMillimeters > 0.0) {
                "Empirical P95 limit must be finite and positive"
            }
            require(
                empiricalReferenceLimitMillimeters.isFinite() &&
                    empiricalReferenceLimitMillimeters > 0.0,
            ) {
                "Reference uncertainty limit must be finite and positive"
            }
            require(minimumEmpiricalSamples in 1..1_000_000) {
                "Empirical sample requirement is outside the supported bound"
            }
            require(minimumIndependentSessions in 1..10_000) {
                "Independent-session requirement is outside the supported bound"
            }
            require(maximumHistory in 1..4096) {
                "Precision history must be bounded between 1 and 4096 decisions"
            }
        }
    }

    data class Input(
        val timestampNanos: Long,
        val reconstructionUncertaintyP95Meters: Double,
        val directionalMajor95Meters: Double,
        val directionalDepth95Meters: Double,
        val geometricInformationP95MajorMeters: Double,
        val geometricInformationP95DepthMeters: Double,
        val denseSurfaceUncertaintyP95Meters: Double,
        val calibrationConfidence: Double,
        val calibrationUncertaintyCoverage: Double = 0.0,
        val calibrationUncertaintyEvidenceConfidence: Double = 0.0,
        val synchronizationConfidence: Double,
        val temporalAlignmentConfidence: Double,
        val temporalAlignmentSpreadMillis: Double,
        val crossViewAgreement: Double,
        val viewDiversity: Double,
        val cameraInformationGain: Double,
        val cameraCount: Int,
        val motionUncertaintyConfidence: Double,
        val motionUncertaintyRadiusMeters: Double,
        val captureTimingConfidence: Double,
        val captureTimingJitterMillis: Double,
        val opticalSignalQuality: Double = 1.0,
        val temporalStability: Double,
        val occlusionRisk: Double,
        val globalRigTrust: Double = 1.0,
        val global4DScore: Double = 1.0,
        val global4DRootVelocityAgreement: Double = 1.0,
        val global4DDirectionalUncertaintyConsistency: Double = 1.0,
        val global4DFaceEyeConsistency: Double = 1.0,
        val runtimeHealthy: Boolean,
        val benchmarkArmed: Boolean,
        val empiricalP95Millimeters: Double? = null,
        val empiricalReferenceUncertaintyMillimeters: Double? = null,
        val empiricalSampleCount: Int = 0,
        val empiricalIndependentSessions: Int = 0,
        val identityContinuity: Double = 1.0,
        val hardwareReadoutAvailable: Boolean = false,
        val hardwareReadoutTimingConfidence: Double = 1.0,
        val hardwareReadoutP95JitterMillis: Double = 0.0,
        val spatialDetailHold: Boolean = false,
        val spatialDetailConfidence: Double = 1.0,
    ) {
        init {
            require(timestampNanos >= 0L)
            listOf(
                reconstructionUncertaintyP95Meters, directionalMajor95Meters,
                directionalDepth95Meters, geometricInformationP95MajorMeters,
                geometricInformationP95DepthMeters, denseSurfaceUncertaintyP95Meters,
                calibrationConfidence, calibrationUncertaintyCoverage,
                calibrationUncertaintyEvidenceConfidence, synchronizationConfidence,
                temporalAlignmentConfidence,
                temporalAlignmentSpreadMillis, crossViewAgreement, viewDiversity,
                cameraInformationGain, motionUncertaintyConfidence, motionUncertaintyRadiusMeters,
                captureTimingConfidence, captureTimingJitterMillis, opticalSignalQuality,
                temporalStability, occlusionRisk, globalRigTrust, global4DScore,
                global4DRootVelocityAgreement, global4DDirectionalUncertaintyConsistency,
                global4DFaceEyeConsistency, identityContinuity, hardwareReadoutTimingConfidence,
                hardwareReadoutP95JitterMillis, spatialDetailConfidence,
            ).forEach { require(it.isFinite()) }
            listOf(
                calibrationConfidence, calibrationUncertaintyCoverage,
                calibrationUncertaintyEvidenceConfidence, synchronizationConfidence,
                temporalAlignmentConfidence, crossViewAgreement, viewDiversity,
                cameraInformationGain, motionUncertaintyConfidence,
                captureTimingConfidence, temporalStability, occlusionRisk,
                opticalSignalQuality, globalRigTrust, global4DScore,
                global4DRootVelocityAgreement, global4DDirectionalUncertaintyConsistency,
                global4DFaceEyeConsistency, identityContinuity,
                hardwareReadoutTimingConfidence, spatialDetailConfidence,
            ).forEach { require(it in 0.0..1.0) }
            listOf(
                reconstructionUncertaintyP95Meters,
                directionalMajor95Meters,
                directionalDepth95Meters,
                geometricInformationP95MajorMeters,
                geometricInformationP95DepthMeters,
                denseSurfaceUncertaintyP95Meters,
                temporalAlignmentSpreadMillis,
                motionUncertaintyRadiusMeters,
                captureTimingJitterMillis,
                hardwareReadoutP95JitterMillis,
            ).forEach { value ->
                require(value >= 0.0) {
                    "Uncertainty, residual and jitter magnitudes cannot be negative"
                }
            }
            require(calibrationUncertaintyCoverage in 0.0..1.0)
            require(calibrationUncertaintyEvidenceConfidence in 0.0..1.0)
            require(cameraCount >= 0)
            require(empiricalSampleCount >= 0)
            require(empiricalIndependentSessions >= 0)
            empiricalP95Millimeters?.let { require(it.isFinite() && it >= 0.0) }
            empiricalReferenceUncertaintyMillimeters?.let { require(it.isFinite() && it >= 0.0) }
        }
    }

    data class UncertaintyBudget(
        val reconstructionMillimeters: Double,
        val directionalMillimeters: Double,
        val geometricMillimeters: Double,
        val surfaceMillimeters: Double,
        val motionMillimeters: Double,
        val temporalPenaltyMillimeters: Double,
        val evidenceInflation: Double,
        val operationalP95Millimeters: Double,
        val headroomToSubMillimeterMillimeters: Double,
        val headroomToHighPrecisionMillimeters: Double,
    )

    data class Evidence(
        val calibration: Double,
        val synchronization: Double,
        val temporalAlignment: Double,
        val crossViewAgreement: Double,
        val viewDiversity: Double,
        val cameraInformationGain: Double,
        val captureTiming: Double,
        val motionUncertainty: Double,
        val runtimeHealth: Double,
        val cameraCount: Int,
        val calibrationUncertaintyCoverage: Double = 0.0,
        val calibrationUncertaintyEvidenceConfidence: Double = 0.0,
        val calibrationUncertaintyReadiness: Double = 0.0,
    )

    data class Decision(
        val timestampNanos: Long,
        val state: State,
        val bottleneck: Bottleneck,
        val evidenceLevel: EvidenceLevel,
        val score: Double,
        val operationalP95Millimeters: Double,
        val liveSubMillimeterCapable: Boolean,
        val measuredSubMillimeterQualified: Boolean,
        val captureAdmission: Boolean,
        val benchmarkReady: Boolean,
        val budget: UncertaintyBudget,
        val evidence: Evidence,
        val warnings: List<String>,
        val recommendations: List<String>,
        val diagnostics: List<String>,
        val qualificationProtocolStatus: String = "NOT_READY",
        val qualificationProtocolScore: Double = 0.0,
        val qualificationProtocolReady: Boolean = false,
        val qualificationProtocolFailedRequirements: Int = 0,
    ) {
        companion object {
            fun idle(): Decision = Decision(
                timestampNanos = 0L, state = State.NO_DATA, bottleneck = Bottleneck.NONE,
                evidenceLevel = EvidenceLevel.NONE, score = 0.0,
                operationalP95Millimeters = Double.POSITIVE_INFINITY,
                liveSubMillimeterCapable = false, measuredSubMillimeterQualified = false,
                captureAdmission = false, benchmarkReady = false,
                budget = UncertaintyBudget(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY),
                evidence = Evidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
                warnings = listOf("NO_LIVE_PRECISION_EVIDENCE"),
                recommendations = listOf("Start a calibrated motion session before precision admission."),
                diagnostics = listOf("state=NO_DATA"),
            )
        }
    }

    private val history = ArrayDeque<Decision>()
    private val correlatedBudgetEngine = CorrelatedUncertaintyBudgetEngine()
    private val qualificationProtocolEngine =
        PrecisionQualificationProtocolEngine()
    private var closed = false

    @Synchronized
    fun evaluate(input: Input): Decision {
        check(!closed)
        val reconstructionMm = positive(input.reconstructionUncertaintyP95Meters * 1000.0)
        val directionalMm = maxOf(positive(input.directionalMajor95Meters * 1000.0),
            positive(input.directionalDepth95Meters * 1000.0))
        val geometricMm = maxOf(positive(input.geometricInformationP95MajorMeters * 1000.0),
            positive(input.geometricInformationP95DepthMeters * 1000.0))
        val surfaceMm = positive(input.denseSurfaceUncertaintyP95Meters * 1000.0)
        val motionMm = positive(input.motionUncertaintyRadiusMeters * 1000.0)
        val temporalPenalty =
            (input.temporalAlignmentSpreadMillis * 0.08 +
                input.captureTimingJitterMillis * 0.12).coerceIn(0.0, 5.0)
        val dominantGeometry =
            maxOf(reconstructionMm, directionalMm, geometricMm, surfaceMm)

        val spatialP95Multiplier = 2.7954834829
        val correlatedBudget =
            correlatedBudgetEngine.combine(
                CorrelatedUncertaintyBudgetEngine.Input(
                    reconstructionMm = reconstructionMm / spatialP95Multiplier,
                    directionalMm = directionalMm / spatialP95Multiplier,
                    geometricMm = geometricMm / spatialP95Multiplier,
                    surfaceMm = surfaceMm / spatialP95Multiplier,
                    motionMm = motionMm,
                    temporalMm = temporalPenalty,
                    observability = input.cameraInformationGain,
                    calibration = input.calibrationConfidence,
                    synchronization = input.synchronizationConfidence,
                    temporalAlignment = input.temporalAlignmentConfidence,
                    crossViewAgreement = input.crossViewAgreement,
                    viewDiversity = input.viewDiversity,
                    captureTiming = input.captureTimingConfidence,
                    occlusionRisk = input.occlusionRisk,
                ),
            )

        val evidenceInflation = correlatedBudget.evidenceInflation
        val operationalP95 = correlatedBudget.finalP95Millimeters
        val runtimeHealth = if (input.runtimeHealthy) 1.0 else 0.0
        val evidence = Evidence(
            calibration = input.calibrationConfidence,
            synchronization = input.synchronizationConfidence,
            temporalAlignment = input.temporalAlignmentConfidence,
            crossViewAgreement = input.crossViewAgreement,
            viewDiversity = input.viewDiversity,
            cameraInformationGain = input.cameraInformationGain,
            captureTiming = input.captureTimingConfidence,
            motionUncertainty = input.motionUncertaintyConfidence,
            runtimeHealth = runtimeHealth,
            cameraCount = input.cameraCount,
            calibrationUncertaintyCoverage =
                input.calibrationUncertaintyCoverage,
            calibrationUncertaintyEvidenceConfidence =
                input.calibrationUncertaintyEvidenceConfidence,
            calibrationUncertaintyReadiness =
                (
                    input.calibrationUncertaintyCoverage *
                        input.calibrationUncertaintyEvidenceConfidence
                    ).coerceIn(0.0, 1.0),
        )
        val componentScore = listOf(
            input.calibrationConfidence,
            input.synchronizationConfidence,
            input.temporalAlignmentConfidence,
            input.crossViewAgreement,
            input.viewDiversity,
            input.cameraInformationGain,
            input.captureTimingConfidence,
            input.motionUncertaintyConfidence,
            input.calibrationUncertaintyCoverage,
            input.calibrationUncertaintyEvidenceConfidence,
            1.0 - input.occlusionRisk,
            input.opticalSignalQuality,
            input.identityContinuity,
            input.spatialDetailConfidence,
            if (
                input.hardwareReadoutAvailable
            ) {
                input.hardwareReadoutTimingConfidence
            } else {
                1.0
            },
            runtimeHealth,
        ).minOrNull() ?: 0.0
        val score = (inverse(operationalP95, config.highPrecisionBoundMillimeters) * 0.38 +
            componentScore * 0.34 +
            input.temporalStability * 0.16 +
            input.opticalSignalQuality * 0.12
        ).coerceIn(0.0, 1.0)
        val liveSubMillimeterCapable = runtimeHealth > 0.5 &&
            operationalP95 <= config.subMillimeterBoundMillimeters &&
            input.calibrationConfidence >= config.calibrationMinimum &&
            input.synchronizationConfidence >= config.synchronizationMinimum &&
            input.temporalAlignmentConfidence >= config.temporalAlignmentMinimum &&
            input.temporalAlignmentSpreadMillis <= config.maximumTemporalSpreadMillis &&
            input.crossViewAgreement >= config.crossViewAgreementMinimum &&
            input.viewDiversity >= config.viewDiversityMinimum &&
            input.captureTimingConfidence >= config.captureTimingMinimum &&
            input.captureTimingJitterMillis <= config.maximumCaptureJitterMillis &&
            input.calibrationUncertaintyCoverage >=
                config.minimumCalibrationUncertaintyCoverage &&
            input.calibrationUncertaintyEvidenceConfidence >=
                config.minimumCalibrationUncertaintyEvidenceConfidence &&
            input.cameraCount >= config.minimumCameraCountForSubMillimeter &&
            input.motionUncertaintyConfidence >= 0.92 &&
            input.temporalStability >= 0.90 &&
            input.opticalSignalQuality >= 0.82 &&
            input.identityContinuity >= config.minimumIdentityContinuity &&
            !input.spatialDetailHold &&
            (
                !input.hardwareReadoutAvailable ||
                    (
                        input.hardwareReadoutTimingConfidence >=
                            config.minimumHardwareReadoutTimingConfidence &&
                        input.hardwareReadoutP95JitterMillis <=
                            config.maximumHardwareReadoutP95JitterMillis
                    )
            )
        val benchmarkReady = input.benchmarkArmed &&
            input.empiricalSampleCount >= config.minimumEmpiricalSamples &&
            input.empiricalIndependentSessions >= config.minimumIndependentSessions &&
            input.empiricalP95Millimeters?.let { it <= config.empiricalP95LimitMillimeters } == true &&
            input.empiricalReferenceUncertaintyMillimeters?.let { it <= config.empiricalReferenceLimitMillimeters } == true
        val measuredQualified = benchmarkReady && liveSubMillimeterCapable
        val state = when {
            !hasGeometry(reconstructionMm, directionalMm, geometricMm) -> State.NO_DATA
            !input.runtimeHealthy -> State.BLOCKED
            measuredQualified -> State.SUB_MILLIMETER_QUALIFIED
            operationalP95 <= config.highPrecisionBoundMillimeters && componentScore >= 0.75 -> State.HIGH_PRECISION
            else -> State.EVIDENCE_ONLY
        }
        val bottleneck = bottleneck(input, operationalP95, runtimeHealth, liveSubMillimeterCapable, measuredQualified)
        val warnings = buildList {
            if (!input.runtimeHealthy) add("RUNTIME_HEALTH_BLOCKS_PRECISION_ADMISSION")
            if (input.calibrationConfidence < config.calibrationMinimum) add("CALIBRATION_UNCERTAINTY_EXCEEDS_PRECISION_CONTRACT")
            if (
                input.calibrationUncertaintyCoverage <
                    config.minimumCalibrationUncertaintyCoverage
            ) {
                add("CALIBRATION_UNCERTAINTY_COVERAGE_INCOMPLETE")
            }
            if (
                input.calibrationUncertaintyEvidenceConfidence <
                    config.minimumCalibrationUncertaintyEvidenceConfidence
            ) {
                add("CALIBRATION_UNCERTAINTY_EVIDENCE_CONFIDENCE_LOW")
            }
            if (input.synchronizationConfidence < config.synchronizationMinimum) add("CLOCK_SYNCHRONIZATION_IS_NOT_PRECISION_GRADE")
            if (input.temporalAlignmentConfidence < config.temporalAlignmentMinimum) add("TEMPORAL_ALIGNMENT_IS_NOT_PRECISION_GRADE")
            if (input.temporalAlignmentSpreadMillis > config.maximumTemporalSpreadMillis) add("TEMPORAL_SPREAD_EXCEEDS_SUB_MILLIMETER_BUDGET")
            if (input.crossViewAgreement < config.crossViewAgreementMinimum) add("CROSS_VIEW_AGREEMENT_IS_BELOW_SUB_MILLIMETER_GATE")
            if (input.viewDiversity < config.viewDiversityMinimum) add("VIEW_DIVERSITY_IS_BELOW_SUB_MILLIMETER_GATE")
            if (input.captureTimingJitterMillis > config.maximumCaptureJitterMillis) add("CAPTURE_JITTER_IS_ABOVE_SUB_MILLIMETER_GATE")
            if (input.cameraCount < config.minimumCameraCountForSubMillimeter) add("INSUFFICIENT_MULTI_VIEW_COVERAGE_FOR_SUB_MILLIMETER_GATE")
            if (input.opticalSignalQuality < 0.82) add("OPTICAL_SIGNAL_QUALITY_IS_BELOW_SUB_MILLIMETER_GATE")
            if (
                input.identityContinuity <
                    config.minimumIdentityContinuity
            ) {
                add("MULTI_PERSON_IDENTITY_CONTINUITY_IS_BELOW_SUB_MILLIMETER_GATE")
            }
            if (
                input.spatialDetailHold
            ) {
                add("SPATIAL_DETAIL_AUTHORITY_IS_HOLDING_PRECISION_REFINEMENT")
            }
            if (
                input.hardwareReadoutAvailable &&
                (
                    input.hardwareReadoutTimingConfidence <
                        config.minimumHardwareReadoutTimingConfidence ||
                    input.hardwareReadoutP95JitterMillis >
                        config.maximumHardwareReadoutP95JitterMillis
                )
            ) {
                add("HARDWARE_READOUT_TIMING_IS_BELOW_PRECISION_GATE")
            }
            if (dominantGeometry > config.highPrecisionBoundMillimeters) add("LIVE_GEOMETRIC_UNCERTAINTY_EXCEEDS_HIGH_PRECISION_BOUND")
            if (!benchmarkReady) add("SUB_MILLIMETER_CLAIM_REQUIRES_MATCHED_EXTERNAL_REFERENCE")
            if (benchmarkReady && !liveSubMillimeterCapable) add("EMPIRICAL_SUB_MILLIMETER_RESULT_IS_NOT_VALID_FOR_CURRENT_LIVE_REGIME")
        }.distinct()
        val recommendations = recommendations(input, bottleneck, operationalP95, benchmarkReady).distinct()
        val diagnostics = listOf(
            "state=" + state.name,
            "bottleneck=" + bottleneck.name,
            "operationalP95Mm=" + format(operationalP95),
            "geometryP95Mm=" + format(dominantGeometry),
            "dominantBudgetComponent=" + correlatedBudget.dominantComponent.name,
            "correlatedBudgetP95Mm=" + format(correlatedBudget.finalP95Millimeters),
            "correlationsApplied=" + correlatedBudget.correlationsApplied,
            "evidenceInflation=" + format(evidenceInflation),
            "precisionScore=" + format(score),
            "liveSubMillimeterCapable=" + liveSubMillimeterCapable,
            "measuredSubMillimeterQualified=" + measuredQualified,
            "benchmarkReady=" + benchmarkReady,
            "cameraCount=" + input.cameraCount,
            "calibrationUncertaintyCoverage=" +
                format(input.calibrationUncertaintyCoverage),
            "calibrationUncertaintyEvidenceConfidence=" +
                format(input.calibrationUncertaintyEvidenceConfidence),
            "calibrationUncertaintyReadiness=" +
                format(
                    evidence.calibrationUncertaintyReadiness,
                ),
        )
        val decision = Decision(
            timestampNanos = input.timestampNanos, state = state, bottleneck = bottleneck,
            evidenceLevel = when { measuredQualified -> EvidenceLevel.EMPIRICAL_AND_LIVE
                benchmarkReady -> EvidenceLevel.EMPIRICAL else -> EvidenceLevel.LIVE_ONLY },
            score = score, operationalP95Millimeters = operationalP95,
            liveSubMillimeterCapable = liveSubMillimeterCapable, measuredSubMillimeterQualified = measuredQualified,
            captureAdmission = state == State.SUB_MILLIMETER_QUALIFIED || state == State.HIGH_PRECISION,
            benchmarkReady = benchmarkReady,
            budget = UncertaintyBudget(reconstructionMm, directionalMm, geometricMm, surfaceMm, motionMm,
                temporalPenalty, evidenceInflation, operationalP95,
                config.subMillimeterBoundMillimeters - operationalP95,
                config.highPrecisionBoundMillimeters - operationalP95),
            evidence = evidence, warnings = warnings, recommendations = recommendations, diagnostics = diagnostics,
        )
        history.addLast(decision)
        while (history.size > config.maximumHistory) history.removeFirst()
        return decision
    }

    @Synchronized
    fun evaluateWithReferenceBenchmark(
        liveInput: Input,
        benchmark: ReferenceMetrologyBenchmarkEngine.Result,
    ): Decision {
        val benchmarkUsable =
            benchmark.validSamples >= config.minimumEmpiricalSamples &&
                benchmark.independentSessions >= config.minimumIndependentSessions &&
                benchmark.qualificationReady &&
                benchmark.global.p95Millimeters.isFinite() &&
                benchmark.referenceUncertaintyP95Millimeters.isFinite()

        val decision = evaluate(
            liveInput.copy(
                benchmarkArmed = benchmarkUsable,
                empiricalP95Millimeters =
                    benchmark.global.p95Millimeters
                        .takeIf(Double::isFinite)
                        ?.coerceAtLeast(0.0),
                empiricalReferenceUncertaintyMillimeters =
                    benchmark.referenceUncertaintyP95Millimeters
                        .takeIf(Double::isFinite)
                        ?.coerceAtLeast(0.0),
                empiricalSampleCount = benchmark.validSamples,
                empiricalIndependentSessions =
                    benchmark.independentSessions,
            ),
        )

        val protocol =
            qualificationProtocolEngine.evaluate(
                benchmark = benchmark,
                liveQualified = decision.liveSubMillimeterCapable,
            )

        val enriched = decision.copy(
            qualificationProtocolStatus =
                protocol.status.name,
            qualificationProtocolScore =
                protocol.score,
            qualificationProtocolReady =
                protocol.qualificationReady,
            qualificationProtocolFailedRequirements =
                protocol.failedRequirements,
            warnings =
                (
                    decision.warnings +
                        protocol.warnings
                    ).distinct().take(24),
            recommendations =
                (
                    decision.recommendations +
                        protocol.recommendations
                    ).distinct().take(24),
            diagnostics =
                (
                    decision.diagnostics +
                        listOf(
                            "qualificationProtocolStatus=" +
                                protocol.status.name,
                            "qualificationProtocolScore=" +
                                format(protocol.score),
                            "qualificationProtocolReady=" +
                                protocol.qualificationReady,
                            "qualificationProtocolFailedRequirements=" +
                                protocol.failedRequirements,
                        )
                    ).distinct(),
        )

        if (history.isNotEmpty()) {
            history.removeLast()
            history.addLast(enriched)
        }
        return enriched
    }

    @Synchronized
    fun latest(): Decision = history.lastOrNull() ?: Decision.idle()

    @Synchronized
    fun recent(limit: Int = 16): List<Decision> = history.toList().takeLast(limit.coerceIn(1, config.maximumHistory))

    @Synchronized
    fun reset() { check(!closed); history.clear() }

    @Synchronized
    override fun close() { if (!closed) { history.clear(); closed = true } }

    private fun hasGeometry(a: Double, b: Double, c: Double): Boolean =
        listOf(a, b, c).any { it.isFinite() && it > 0.0 }

    private fun bottleneck(
        input: Input, operationalP95: Double, runtimeHealth: Double,
        liveSubMillimeterCapable: Boolean, measuredQualified: Boolean,
    ): Bottleneck {
        if (!input.runtimeHealthy) return Bottleneck.RUNTIME_HEALTH
        if (input.calibrationConfidence < config.calibrationMinimum ||
            input.calibrationUncertaintyCoverage <
                config.minimumCalibrationUncertaintyCoverage ||
            input.calibrationUncertaintyEvidenceConfidence <
                config.minimumCalibrationUncertaintyEvidenceConfidence
        ) return Bottleneck.CALIBRATION
        if (input.synchronizationConfidence < config.synchronizationMinimum) return Bottleneck.SYNCHRONIZATION
        if (input.temporalAlignmentConfidence < config.temporalAlignmentMinimum ||
            input.temporalAlignmentSpreadMillis > config.maximumTemporalSpreadMillis) return Bottleneck.TEMPORAL_ALIGNMENT
        if (input.viewDiversity < config.viewDiversityMinimum ||
            input.crossViewAgreement < config.crossViewAgreementMinimum) return Bottleneck.CROSS_VIEW_DIVERSITY
        if (input.geometricInformationP95MajorMeters * 1000.0 > config.highPrecisionBoundMillimeters) return Bottleneck.GEOMETRIC_OBSERVABILITY
        if (input.denseSurfaceUncertaintyP95Meters * 1000.0 > config.highPrecisionBoundMillimeters) return Bottleneck.SURFACE_UNCERTAINTY
        if (input.motionUncertaintyConfidence < 0.85) return Bottleneck.MOTION_UNCERTAINTY
        if (input.captureTimingConfidence < config.captureTimingMinimum ||
            input.captureTimingJitterMillis > config.maximumCaptureJitterMillis) return Bottleneck.CAPTURE_TIMING
        if (operationalP95 > config.highPrecisionBoundMillimeters) return Bottleneck.RECONSTRUCTION_UNCERTAINTY
        if (runtimeHealth < 0.5) return Bottleneck.RUNTIME_HEALTH
        if (!measuredQualified && !liveSubMillimeterCapable) return Bottleneck.EMPIRICAL_REFERENCE
        return Bottleneck.NONE
    }

    private fun recommendations(
        input: Input, bottleneck: Bottleneck, operationalP95: Double, benchmarkReady: Boolean,
    ): List<String> = buildList {
        when (bottleneck) {
            Bottleneck.CALIBRATION -> add(
                "Re-run multi-view calibration and attach measurement-derived uncertainty profiles to every participating camera.",
            )
            Bottleneck.SYNCHRONIZATION -> add("Synchronize device clocks and reject stale frames.")
            Bottleneck.TEMPORAL_ALIGNMENT -> add("Tighten sub-frame alignment and reduce camera skew.")
            Bottleneck.CROSS_VIEW_DIVERSITY -> add("Increase azimuth/elevation coverage and camera overlap.")
            Bottleneck.GEOMETRIC_OBSERVABILITY -> add("Increase parallax with a wider calibrated baseline.")
            Bottleneck.SURFACE_UNCERTAINTY -> add("Increase dense-surface evidence and temporal persistence.")
            Bottleneck.MOTION_UNCERTAINTY -> add("Reduce unsupported prediction during fast or occluded motion.")
            Bottleneck.CAPTURE_TIMING -> add("Lower capture jitter and frame skew.")
            Bottleneck.RECONSTRUCTION_UNCERTAINTY -> add("Increase trustworthy cross-view evidence before forcing a solve.")
            Bottleneck.RUNTIME_HEALTH -> add("Relieve thermal, memory or queue pressure before precision capture.")
            Bottleneck.EMPIRICAL_REFERENCE -> add("Run a matched external-reference cohort before making a sub-millimetre claim.")
            Bottleneck.NONE -> Unit
        }
        if (operationalP95 > config.subMillimeterBoundMillimeters) add("Live uncertainty is above the sub-millimetre budget.")
        if (!benchmarkReady) add("Keep sub-millimetre status evidence-only until reference uncertainty is measured.")
        if (input.cameraCount < config.minimumCameraCountForSubMillimeter) add(
            "Use at least " + config.minimumCameraCountForSubMillimeter + " synchronized views for sub-millimetre admission.",
        )
    }

    private fun positive(value: Double): Double = if (value.isFinite() && value >= 0.0) value else 0.0

    private fun inverse(value: Double, limit: Double): Double =
        if (!value.isFinite() || limit <= 0.0) 0.0 else (1.0 - value / limit).coerceIn(0.0, 1.0)

    private fun format(value: Double): String = if (value.isFinite()) "%.4f".format(value) else "INF"
}