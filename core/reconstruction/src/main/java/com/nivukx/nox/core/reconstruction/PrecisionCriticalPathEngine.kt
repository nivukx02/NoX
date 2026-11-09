package com.nivukx.nox.core.reconstruction

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Attributes the current motion-capture precision budget to concrete measurable causes and
 * produces bounded mitigation guidance.
 *
 * This is intentionally a decision layer, not a magical accuracy booster. It can recommend
 * geometry/timing/calibration changes, but it cannot claim those changes actually achieved the
 * requested error until measured evidence confirms them.
 */
class PrecisionCriticalPathEngine {

    enum class Term {
        IMAGE_REPROJECTION,
        OBSERVATION_SAMPLING,
        STEREO_GEOMETRY,
        DEPTH_SENSOR,
        CAMERA_POSE,
        TEMPORAL_TIMING,
        ROLLING_SHUTTER,
        CALIBRATION,
        METRIC_SCALE,
        RECONSTRUCTION_PRIOR,
        CORRELATED_RUNTIME,
    }

    enum class Priority {
        CRITICAL,
        HIGH,
        MEDIUM,
        LOW,
        SATISFIED,
    }

    data class Input(
        val metrology: MetricPrecisionBudgetEngine.Decision,
        val targetP95Millimeters: Double,
        val targetP99Millimeters: Double,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val viewDiversity: Double,
        val crossViewAgreement: Double,
        val identityContinuity: Double,
        val temporalLagMilliseconds: Double,
        val rollingShutterRisk: Double,
        val occlusionRisk: Double,
        val metricScaleDriftPercentPerMinute: Double,
        val thermalRisk: Double,
        val memoryPressure: Double,
        val surfaceTemporalStability: Double,
        val empiricalP95Millimeters: Double? = null,
        val empiricalP99Millimeters: Double? = null,
    ) {
        init {
            require(targetP95Millimeters > 0.0 && targetP95Millimeters.isFinite())
            require(targetP99Millimeters >= targetP95Millimeters)
            listOf(
                calibrationConfidence,
                synchronizationConfidence,
                viewDiversity,
                crossViewAgreement,
                identityContinuity,
                rollingShutterRisk,
                occlusionRisk,
                thermalRisk,
                memoryPressure,
                surfaceTemporalStability,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(
                temporalLagMilliseconds.isFinite() &&
                    temporalLagMilliseconds >= 0.0,
            )
            require(
                metricScaleDriftPercentPerMinute.isFinite() &&
                    metricScaleDriftPercentPerMinute >= 0.0,
            )
            empiricalP95Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
            empiricalP99Millimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class TermBudget(
        val term: Term,
        val sigmaMillimeters: Double,
        val shareOfVariance: Double,
        val priority: Priority,
        val measurable: Boolean,
        val reason: String,
    ) {
        init {
            require(sigmaMillimeters.isFinite() && sigmaMillimeters >= 0.0)
            require(shareOfVariance.isFinite() && shareOfVariance in 0.0..1.0)
        }
    }

    data class Mitigation(
        val term: Term,
        val action: String,
        val expectedEffect: String,
        val requiredEvidence: String,
        val operatorVisible: Boolean = true,
    )

    data class Decision(
        val targetP95Millimeters: Double,
        val modeledP95Millimeters: Double,
        val modeledP99Millimeters: Double,
        val residualP95BudgetMillimeters: Double,
        val residualP99BudgetMillimeters: Double,
        val dominantTerm: Term?,
        val dominantPriority: Priority,
        val terms: List<TermBudget>,
        val mitigations: List<Mitigation>,
        val referenceGradeFeasible: Boolean,
        val empiricalEvidenceSupportsTarget: Boolean,
        val requiresMeasuredCertification: Boolean,
        val blocked: Boolean,
        val reasons: List<String>,
    ) {
        val precisionBudgetHealthy: Boolean
            get() =
                !blocked &&
                    residualP95BudgetMillimeters >= 0.0 &&
                    residualP99BudgetMillimeters >= 0.0

        val topTerms: List<TermBudget>
            get() = terms
                .sortedByDescending {
                    it.shareOfVariance
                }
                .take(5)
    }

    fun evaluate(input: Input): Decision {
        val c = input.metrology.contribution
        val variances = linkedMapOf(
            Term.IMAGE_REPROJECTION to c.imageReprojectionMillimeters.squared(),
            Term.OBSERVATION_SAMPLING to
                c.observationSamplingMillimeters.squared(),
            Term.STEREO_GEOMETRY to c.stereoDepthMillimeters.squared(),
            Term.DEPTH_SENSOR to c.depthSensorMillimeters.squared(),
            Term.CAMERA_POSE to c.cameraPoseMillimeters.squared(),
            Term.TEMPORAL_TIMING to c.timingMillimeters.squared(),
            Term.ROLLING_SHUTTER to c.rollingShutterMillimeters.squared(),
            Term.CALIBRATION to c.calibrationMillimeters.squared(),
            Term.METRIC_SCALE to c.scaleMillimeters.squared(),
            Term.RECONSTRUCTION_PRIOR to c.reconstructionPriorMillimeters.squared(),
        )

        val baseVariance = variances.values.sum().coerceAtLeast(1e-12)
        val correlationSigma = correlatedRuntimeSigma(input)
        val totalVariance = baseVariance + correlationSigma.squared()

        val modelSigma = sqrt(totalVariance)
        val modeledP95 = modelSigma * 1.96
        val modeledP99 = modelSigma * 2.576

        val terms = variances.map { (term, variance) ->
            termBudget(
                term = term,
                variance = variance,
                totalVariance = totalVariance,
                input = input,
            )
        }.toMutableList()

        if (correlationSigma > 0.0) {
            terms += TermBudget(
                term = Term.CORRELATED_RUNTIME,
                sigmaMillimeters = correlationSigma,
                shareOfVariance =
                    (correlationSigma.squared() / totalVariance)
                        .coerceIn(0.0, 1.0),
                priority = priorityFor(
                    correlationSigma * 1.96,
                    input.targetP95Millimeters,
                ),
                measurable = true,
                reason = "Correlated timing, occlusion, rig and drift effects.",
            )
        }

        val dominant = terms.maxByOrNull { it.shareOfVariance }
        val residualP95 = input.targetP95Millimeters - modeledP95
        val residualP99 = input.targetP99Millimeters - modeledP99

        val empiricalSupports = input.empiricalP95Millimeters?.let {
            it <= input.targetP95Millimeters
        } == true && input.empiricalP99Millimeters?.let {
            it <= input.targetP99Millimeters
        } == true

        val requiresMeasuredCertification =
            input.targetP95Millimeters <= 2.0

        val blocked =
            input.metrology.hardBlock ||
                modeledP95 > input.targetP95Millimeters * 1.35 ||
                modeledP99 > input.targetP99Millimeters * 1.35

        val reasons = buildList {
            if (blocked) add("precision-budget-exceeded")
            if (residualP95 < 0.0) add("p95-budget-negative")
            if (residualP99 < 0.0) add("p99-budget-negative")
            if (input.calibrationConfidence < 0.92) {
                add("calibration-not-reference-grade")
            }
            if (input.synchronizationConfidence < 0.92) {
                add("synchronization-not-reference-grade")
            }
            if (input.viewDiversity < 0.65) {
                add("view-diversity-not-reference-grade")
            }
            if (input.crossViewAgreement < 0.75) {
                add("cross-view-agreement-not-reference-grade")
            }
            if (input.identityContinuity < 0.90) {
                add("identity-continuity-not-reference-grade")
            }
            if (input.rollingShutterRisk > 0.35) {
                add("rolling-shutter-regime-dominant")
            }
            if (input.occlusionRisk > 0.45) {
                add("occlusion-regime-dominant")
            }
            if (input.metricScaleDriftPercentPerMinute > 0.10) {
                add("metric-scale-drift-too-high")
            }
            if (input.thermalRisk > 0.65) {
                add("thermal-headroom-below-reference")
            }
            if (input.memoryPressure > 0.82) {
                add("memory-headroom-below-reference")
            }
            if (input.surfaceTemporalStability < 0.90) {
                add("surface-stability-below-reference")
            }
            if (requiresMeasuredCertification && !empiricalSupports) {
                add("sub-2mm-target-requires-empirical-certification")
            }
        }

        return Decision(
            targetP95Millimeters = input.targetP95Millimeters,
            modeledP95Millimeters = modeledP95,
            modeledP99Millimeters = modeledP99,
            residualP95BudgetMillimeters = residualP95,
            residualP99BudgetMillimeters = residualP99,
            dominantTerm = dominant?.term,
            dominantPriority =
                dominant?.priority ?: Priority.LOW,
            terms = terms.sortedByDescending {
                it.shareOfVariance
            },
            mitigations = mitigationsFor(
                input = input,
                dominant = dominant?.term,
                terms = terms,
            ),
            referenceGradeFeasible =
                input.metrology.referenceGradeFeasible &&
                    modeledP95 <= 2.0 &&
                    modeledP99 <= 5.0,
            empiricalEvidenceSupportsTarget = empiricalSupports,
            requiresMeasuredCertification = requiresMeasuredCertification,
            blocked = blocked,
            reasons = reasons.distinct(),
        )
    }

    private fun termBudget(
        term: Term,
        variance: Double,
        totalVariance: Double,
        input: Input,
    ): TermBudget {
        val sigma = sqrt(variance.coerceAtLeast(0.0))
        return TermBudget(
            term = term,
            sigmaMillimeters = sigma,
            shareOfVariance =
                (variance / totalVariance).coerceIn(0.0, 1.0),
            priority = priorityFor(
                errorContributionP95 = sigma * 1.96,
                targetP95 = input.targetP95Millimeters,
            ),
            measurable = true,
            reason = reasonFor(term, input),
        )
    }

    private fun priorityFor(
        errorContributionP95: Double,
        targetP95: Double,
    ): Priority {
        val share = (
            errorContributionP95 /
                max(targetP95, 0.001)
        ).coerceIn(0.0, 2.0)

        return when {
            share >= 0.70 -> Priority.CRITICAL
            share >= 0.40 -> Priority.HIGH
            share >= 0.18 -> Priority.MEDIUM
            share >= 0.05 -> Priority.LOW
            else -> Priority.SATISFIED
        }
    }

    private fun correlatedRuntimeSigma(
        input: Input,
    ): Double {
        val timing = (
            input.temporalLagMilliseconds *
                0.10 +
                input.rollingShutterRisk *
                input.targetP95Millimeters *
                0.12
        )
        val occlusion = (
            input.occlusionRisk *
                max(
                    1.0,
                    input.empiricalP95Millimeters ?: input.targetP95Millimeters,
                ) * 0.16
        )
        val rig = (
            (1.0 - input.crossViewAgreement) *
                max(
                    1.0,
                    input.empiricalP95Millimeters ?: input.targetP95Millimeters,
                ) * 0.20
        )
        val drift =
            input.metricScaleDriftPercentPerMinute * 0.35
        val thermal =
            input.thermalRisk * 0.18
        val memory =
            input.memoryPressure * 0.12
        val surface =
            (1.0 - input.surfaceTemporalStability) * 0.40

        return sqrt(
            timing.squared() +
                occlusion.squared() +
                rig.squared() +
                drift.squared() +
                thermal.squared() +
                memory.squared() +
                surface.squared(),
        )
    }

    private fun reasonFor(
        term: Term,
        input: Input,
    ): String =
        when (term) {
            Term.IMAGE_REPROJECTION ->
                "Detector localization and lens-model residual dominate image-space uncertainty."
            Term.OBSERVATION_SAMPLING ->
                "Sub-pixel localization, photon statistics, read noise and quantization limit observation resolution."
            Term.STEREO_GEOMETRY ->
                "Baseline, focal length and parallax dominate metric depth sensitivity."
            Term.DEPTH_SENSOR ->
                "Depth-source noise is material to the metric 3D budget."
            Term.CAMERA_POSE ->
                "Camera pose translation/rotation uncertainty propagates into world-space joints."
            Term.TEMPORAL_TIMING ->
                "Temporal skew converts motion into spatial reconstruction error."
            Term.ROLLING_SHUTTER ->
                "Row timing during motion creates structured spatial distortion."
            Term.CALIBRATION ->
                "Calibration confidence and residuals leave systematic geometry error."
            Term.METRIC_SCALE ->
                "Metric scale drift contaminates every world-space measurement."
            Term.RECONSTRUCTION_PRIOR ->
                "Priors stabilize under ambiguity but impose model-dependent residual error."
            Term.CORRELATED_RUNTIME ->
                "Runtime conditions correlate otherwise independent error terms."
        }

    private fun mitigationsFor(
        input: Input,
        dominant: Term?,
        terms: List<TermBudget>,
    ): List<Mitigation> {
        val actions = mutableListOf<Mitigation>()

        fun add(
            term: Term,
            action: String,
            effect: String,
            evidence: String,
        ) {
            actions += Mitigation(
                term = term,
                action = action,
                expectedEffect = effect,
                requiredEvidence = evidence,
            )
        }

        val ordered = listOfNotNull(dominant) +
            terms.map { it.term }.filter { it != dominant }

        for (term in ordered.distinct()) {
            when (term) {
                Term.IMAGE_REPROJECTION -> add(
                    term,
                    "Increase effective focal sampling or tighten keypoint residual gating; prefer higher-resolution observations when thermally sustainable.",
                    "Reduces angular localization error before triangulation.",
                    "Measured reprojection RMS/P95 and calibrated intrinsics.",
                )
                Term.OBSERVATION_SAMPLING -> add(
                    term,
                    "Reduce the observation noise floor with stronger sub-pixel localization, better signal-to-noise, controlled exposure and calibrated sensor readout.",
                    "Reduces the irreducible image-space measurement floor before multi-view geometry.",
                    "Measured localization sigma, sensor noise characterization and residual histograms.",
                )
                Term.STEREO_GEOMETRY -> add(
                    term,
                    "Increase useful baseline and parallax while retaining subject coverage; prefer geometry-aware camera selection over simply adding cameras.",
                    "Improves metric depth conditioning.",
                    "Baseline, ray-angle distribution and selected-view conditioning.",
                )
                Term.DEPTH_SENSOR -> add(
                    term,
                    "Fuse depth only when its confidence and cross-view consistency beat the triangulated uncertainty.",
                    "Avoids replacing a better geometric estimate with noisy depth.",
                    "Per-pixel depth confidence and cross-view residuals.",
                )
                Term.CAMERA_POSE -> add(
                    term,
                    "Tighten camera pose refinement and recalibrate when residual slope or thermal drift rises.",
                    "Reduces world-space transformation error.",
                    "Pose covariance, reprojection residual and drift telemetry.",
                )
                Term.TEMPORAL_TIMING -> add(
                    term,
                    "Tighten hardware/clock alignment and use timestamp-aware interpolation rather than frame-index matching.",
                    "Reduces velocity-to-position conversion error.",
                    "Clock skew P95, temporal residual and measured lag.",
                )
                Term.ROLLING_SHUTTER -> add(
                    term,
                    "Use row-time-aware correction and suppress high-risk views when readout distortion dominates.",
                    "Reduces motion-dependent spatial warping.",
                    "Readout fraction, row timestamp model and residual improvement.",
                )
                Term.CALIBRATION -> add(
                    term,
                    "Enter recalibration workflow when residual slope or uncertainty crosses the reference budget.",
                    "Reduces systematic bias across the capture volume.",
                    "Calibration confidence, residual trend and control-point coverage.",
                )
                Term.METRIC_SCALE -> add(
                    term,
                    "Stabilize metric scale with cross-camera scale constraints and bounded drift correction.",
                    "Prevents accumulated world-space scale drift.",
                    "Scale ratio, drift-per-minute and reference-length residual.",
                )
                Term.RECONSTRUCTION_PRIOR -> add(
                    term,
                    "Increase multi-view evidence before increasing prior strength; keep priors uncertainty-aware.",
                    "Prevents plausible-looking but metrically wrong geometry.",
                    "Cross-view agreement and uncertainty reduction.",
                )
                Term.CORRELATED_RUNTIME -> add(
                    term,
                    "Reduce the dominant shared failure source before adding model complexity or smoothing.",
                    "Prevents correlated errors from surviving robust fusion.",
                    "Change-pointed runtime telemetry across the affected cameras.",
                )
            }
        }

        return actions.distinctBy {
            it.term
        }.take(10)
    }

    private fun Double.squared(): Double = this * this
}
