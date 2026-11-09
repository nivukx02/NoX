package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-end metrology and reproducibility layer for motion-capture evaluation.
 *
 * This is measurement infrastructure, not an accuracy generator. It deliberately separates:
 * - point error,
 * - random repeatability,
 * - between-session reproducibility,
 * - temporal lag,
 * - systematic bias,
 * - regime-specific degradation,
 * - uncertainty of the external reference,
 * - confidence intervals around reported quantiles.
 *
 * A low P95 without a confidence interval can be misleading for small datasets. A low average
 * without regime stratification can hide catastrophic failures during occlusion. A comparison
 * against an external reference without a reference uncertainty budget can also create false
 * precision. This engine keeps those dimensions explicit.
 *
 * State is bounded, all input is finite-checked, and every certification decision is conservative.
 */
class HighEndMetrologyBenchmarkEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class ReferenceKind {
        OPTICAL_MARKER_REFERENCE,
        INFRARED_MARKER_REFERENCE,
        MULTI_CAMERA_REFERENCE,
        SYNTHETIC_GROUND_TRUTH,
        MANUAL_REFERENCE,
    }

    enum class MotionRegime {
        STATIC,
        SLOW,
        NORMAL,
        FAST,
        EXTREME,
    }

    enum class OcclusionRegime {
        NONE,
        MILD,
        MODERATE,
        SEVERE,
    }

    enum class LightingRegime {
        CONTROLLED,
        VARIABLE,
        CHALLENGING,
    }

    enum class GateState {
        PASS,
        FAIL,
        UNKNOWN,
    }

    data class Config(
        val minimumSamples: Int = 120,
        val minimumSamplesForSubMillimeterEvidence: Int = 240,
        val minimumIndependentSessions: Int = 3,
        val confidenceLevel: Double = 0.95,
        val bootstrapIterations: Int = 512,
        val maximumSamples: Int = 500_000,
        val maximumJoints: Int = 512,
        val maximumRegimeBuckets: Int = 64,
        val maximumDiagnosticHistory: Int = 256,
        val p50TargetMm: Double = 2.0,
        val p95TargetMm: Double = 2.0,
        val p99TargetMm: Double = 5.0,
        val subMillimeterP95TargetMm: Double = 1.0,
        val subMillimeterP99TargetMm: Double = 2.5,
        val referenceUncertaintyTargetMm: Double = 0.50,
        val subMillimeterReferenceUncertaintyTargetMm: Double = 0.25,
        val repeatabilityTargetMm: Double = 1.0,
        val subMillimeterRepeatabilityTargetMm: Double = 0.50,
        val reproducibilityTargetMm: Double = 2.0,
        val lagTargetMs: Double = 25.0,
        val dropoutTarget: Double = 0.02,
    ) {
        init {
            require(minimumSamples > 0)
            require(minimumSamplesForSubMillimeterEvidence >= minimumSamples)
            require(minimumIndependentSessions >= 2)
            require(confidenceLevel in 0.80..0.999)
            require(bootstrapIterations >= 64)
            require(maximumSamples >= minimumSamples)
            require(maximumJoints > 0)
            require(maximumRegimeBuckets > 0)
            require(maximumDiagnosticHistory >= 16)
            require(p50TargetMm > 0.0)
            require(p95TargetMm > 0.0)
            require(p99TargetMm >= p95TargetMm)
            require(subMillimeterP95TargetMm > 0.0)
            require(subMillimeterP99TargetMm >= subMillimeterP95TargetMm)
            require(referenceUncertaintyTargetMm > 0.0)
            require(subMillimeterReferenceUncertaintyTargetMm > 0.0)
            require(repeatabilityTargetMm > 0.0)
            require(subMillimeterRepeatabilityTargetMm > 0.0)
            require(reproducibilityTargetMm > 0.0)
            require(lagTargetMs > 0.0)
            require(dropoutTarget in 0.0..1.0)
        }
    }

    data class Sample(
        val sessionId: String,
        val timestampNanos: Long,
        val jointId: Int,
        val estimate: Vec3,
        val reference: Vec3,
        val confidence: Double = 1.0,
        val visible: Boolean = true,
        val motionRegime: MotionRegime = MotionRegime.NORMAL,
        val occlusionRegime: OcclusionRegime = OcclusionRegime.NONE,
        val lightingRegime: LightingRegime = LightingRegime.CONTROLLED,
        val referenceUncertaintyMm: Double? = null,
        val estimateScale: Double? = null,
        val referenceScale: Double? = null,
    ) {
        init {
            require(sessionId.isNotBlank())
            require(timestampNanos >= 0L)
            require(jointId >= 0)
            require(estimate.x.isFinite() && estimate.y.isFinite() && estimate.z.isFinite())
            require(reference.x.isFinite() && reference.y.isFinite() && reference.z.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
            referenceUncertaintyMm?.let {
                require(it.isFinite() && it >= 0.0)
            }
            estimateScale?.let {
                require(it.isFinite() && it > 0.0)
            }
            referenceScale?.let {
                require(it.isFinite() && it > 0.0)
            }
        }
    }

    data class SurfaceSample(
        val sessionId: String,
        val timestampNanos: Long,
        val estimate: Vec3,
        val reference: Vec3,
        val confidence: Double = 1.0,
        val referenceUncertaintyMm: Double? = null,
        val estimateNormal: Vec3? = null,
        val referenceNormal: Vec3? = null,
    ) {
        init {
            require(sessionId.isNotBlank())
            require(timestampNanos >= 0L)
            require(estimate.x.isFinite() && estimate.y.isFinite() && estimate.z.isFinite())
            require(reference.x.isFinite() && reference.y.isFinite() && reference.z.isFinite())
            require(confidence in 0.0..1.0)
            referenceUncertaintyMm?.let {
                require(it.isFinite() && it >= 0.0)
            }
            estimateNormal?.let { require(it.x.isFinite() && it.y.isFinite() && it.z.isFinite()) }
            referenceNormal?.let { require(it.x.isFinite() && it.y.isFinite() && it.z.isFinite()) }
        }
    }

    data class Input(
        val captureId: String,
        val referenceKind: ReferenceKind,
        val samples: List<Sample>,
        val surfaceSamples: List<SurfaceSample> = emptyList(),
        val matchedReferenceConditions: Boolean = false,
        val referenceUncertaintyMm: Double? = null,
    ) {
        init {
            require(captureId.isNotBlank())
            require(samples.isNotEmpty())
            referenceUncertaintyMm?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class ConfidenceInterval(
        val lower: Double,
        val center: Double,
        val upper: Double,
        val confidenceLevel: Double,
    ) {
        init {
            require(lower.isFinite())
            require(center.isFinite())
            require(upper.isFinite())
            require(lower <= center)
            require(center <= upper)
            require(confidenceLevel in 0.0..0.9999)
        }
    }

    data class JointMetrics(
        val jointId: Int,
        val sampleCount: Int,
        val validCount: Int,
        val dropoutRatio: Double,
        val meanErrorMm: Double,
        val rmsErrorMm: Double,
        val p50ErrorMm: Double,
        val p95ErrorMm: Double,
        val p99ErrorMm: Double,
        val p95ConfidenceInterval: ConfidenceInterval?,
        val p99ConfidenceInterval: ConfidenceInterval?,
        val biasVectorMm: Vec3,
        val biasMagnitudeMm: Double,
        val repeatabilityMm: Double,
        val worstMotionRegimeP95Mm: Double,
        val worstOcclusionRegimeP95Mm: Double,
    )

    data class BlandAltman(
        val sampleCount: Int,
        val meanBiasMm: Double,
        val standardDeviationMm: Double,
        val lowerAgreementMm: Double,
        val upperAgreementMm: Double,
    ) {
        val spanMm: Double
            get() = upperAgreementMm - lowerAgreementMm
    }

    data class Reproducibility(
        val sessionCount: Int,
        val withinSessionRepeatabilityMm: Double,
        val betweenSessionReproducibilityMm: Double,
        val sessionMeansMm: Map<String, Double>,
        val sessionBiasMm: Map<String, Double>,
    )

    data class RegimeMetrics(
        val key: String,
        val sampleCount: Int,
        val p50Mm: Double,
        val p95Mm: Double,
        val p99Mm: Double,
        val dropoutRatio: Double,
        val repeatabilityMm: Double,
    )

    data class SurfaceMetrics(
        val sampleCount: Int,
        val validCount: Int,
        val p50PointErrorMm: Double,
        val p95PointErrorMm: Double,
        val p99PointErrorMm: Double,
        val meanNormalErrorDegrees: Double?,
    )

    data class Aggregate(
        val sampleCount: Int,
        val validCount: Int,
        val jointCount: Int,
        val sessionCount: Int,
        val dropoutRatio: Double,
        val meanErrorMm: Double,
        val rmsErrorMm: Double,
        val p50ErrorMm: Double,
        val p95ErrorMm: Double,
        val p99ErrorMm: Double,
        val p95ConfidenceInterval: ConfidenceInterval?,
        val p99ConfidenceInterval: ConfidenceInterval?,
        val maxErrorMm: Double,
        val biasMagnitudeMm: Double,
        val repeatabilityMm: Double,
        val reproducibilityMm: Double,
        val lagMs: Double,
        val scaleDriftPercentPerMinute: Double?,
        val referenceUncertaintyMm: Double?,
        val blandAltman: BlandAltman?,
        val surface: SurfaceMetrics?,
    )

    data class Gate(
        val name: String,
        val state: GateState,
        val measured: Double?,
        val upperConfidenceBound: Double?,
        val limit: Double?,
        val unit: String,
        val reason: String,
    )

    data class Certification(
        val referenceKind: ReferenceKind,
        val empiricalEvidenceSufficient: Boolean,
        val matchedConditions: Boolean,
        val referenceUncertaintyKnown: Boolean,
        val twoMillimeterEvidence: Boolean,
        val oneMillimeterEvidence: Boolean,
        val hardBlocked: Boolean,
        val confidence: Double,
        val gates: List<Gate>,
        val blockers: List<String>,
        val warnings: List<String>,
    )

    data class Report(
        val captureId: String,
        val referenceKind: ReferenceKind,
        val aggregate: Aggregate,
        val jointMetrics: List<JointMetrics>,
        val regimeMetrics: List<RegimeMetrics>,
        val reproducibility: Reproducibility,
        val blandAltman: BlandAltman?,
        val surface: SurfaceMetrics?,
        val certification: Certification,
        val diagnostics: List<String>,
    )

    data class Comparison(
        val p50DeltaMm: Double,
        val p95DeltaMm: Double,
        val p99DeltaMm: Double,
        val repeatabilityDeltaMm: Double,
        val reproducibilityDeltaMm: Double,
        val biasDeltaMm: Double,
        val lagDeltaMs: Double,
        val note: String,
    )

    private val diagnosticHistory = ArrayDeque<String>()
    private var closed = false

    @Synchronized
    fun evaluate(input: Input): Report {
        check(!closed)

        val samples = input.samples
            .asSequence()
            .filter { it.confidence > 0.0 }
            .sortedWith(
                compareBy<Sample> { it.timestampNanos }
                    .thenBy { it.sessionId }
                    .thenBy { it.jointId },
            )
            .take(config.maximumSamples)
            .toList()

        require(samples.isNotEmpty())

        val jointGroups = samples
            .groupBy { it.jointId }
            .toSortedMap()
            .entries
            .take(config.maximumJoints)

        val jointMetrics = jointGroups.map {
            evaluateJoint(
                it.key,
                it.value,
            )
        }

        val validSamples = samples.filter {
            it.visible &&
                it.confidence >= 0.35
        }

        val errors = validSamples.map(::errorMm)
        val ci95 = bootstrapConfidenceInterval(
            values = errors,
            captureId = input.captureId + ":p95",
            statistic = { percentile(it, 0.95) },
        )
        val ci99 = bootstrapConfidenceInterval(
            values = errors,
            captureId = input.captureId + ":p99",
            statistic = { percentile(it, 0.99) },
        )

        val aggregate = Aggregate(
            sampleCount = samples.size,
            validCount = validSamples.size,
            jointCount = jointGroups.size,
            sessionCount = samples.map { it.sessionId }.distinct().size,
            dropoutRatio = 1.0 -
                validSamples.size.toDouble() /
                samples.size.coerceAtLeast(1).toDouble(),
            meanErrorMm = mean(errors),
            rmsErrorMm = rms(errors),
            p50ErrorMm = percentile(errors, 0.50),
            p95ErrorMm = percentile(errors, 0.95),
            p99ErrorMm = percentile(errors, 0.99),
            p95ConfidenceInterval = ci95,
            p99ConfidenceInterval = ci99,
            maxErrorMm = errors.maxOrNull() ?: Double.POSITIVE_INFINITY,
            biasMagnitudeMm = biasMagnitude(validSamples),
            repeatabilityMm = withinSessionRepeatability(validSamples),
            reproducibilityMm =
                betweenSessionReproducibility(validSamples),
            lagMs = estimateLagMs(validSamples),
            scaleDriftPercentPerMinute = scaleDrift(validSamples),
            referenceUncertaintyMm =
                referenceUncertainty(
                    input = input,
                    samples = samples,
                ),
            blandAltman = blandAltman(validSamples),
            surface =
                evaluateSurface(input.surfaceSamples),
        )

        val regimeMetrics =
            evaluateRegimes(samples)

        val reproducibility =
            evaluateReproducibility(validSamples)

        val certification =
            certify(
                input = input,
                aggregate = aggregate,
                reproducibility = reproducibility,
            )

        val diagnostics = buildList {
            add("samples=" + samples.size)
            add("valid=" + validSamples.size)
            add("sessions=" + aggregate.sessionCount)
            add("p95=" + metric(aggregate.p95ErrorMm))
            add("p99=" + metric(aggregate.p99ErrorMm))
            add(
                "p95-ci=" +
                    formatInterval(aggregate.p95ConfidenceInterval),
            )
            add(
                "p99-ci=" +
                    formatInterval(aggregate.p99ConfidenceInterval),
            )
            add("repeatability=" + metric(aggregate.repeatabilityMm))
            add("reproducibility=" + metric(aggregate.reproducibilityMm))
            add("lagMs=" + metric(aggregate.lagMs))
            if (aggregate.referenceUncertaintyMm != null) {
                add(
                    "referenceUncertaintyMm=" +
                        metric(aggregate.referenceUncertaintyMm),
                )
            }
            certification.warnings.take(8).forEach {
                add("warning=" + it)
            }
        }

        diagnostics.forEach(::appendDiagnostic)

        return Report(
            captureId = input.captureId,
            referenceKind = input.referenceKind,
            aggregate = aggregate,
            jointMetrics = jointMetrics,
            regimeMetrics = regimeMetrics,
            reproducibility = reproducibility,
            blandAltman = aggregate.blandAltman,
            surface = aggregate.surface,
            certification = certification,
            diagnostics = diagnostics,
        )
    }

    @Synchronized
    fun compare(
        a: Report,
        b: Report,
    ): Comparison =
        Comparison(
            p50DeltaMm =
                a.aggregate.p50ErrorMm -
                    b.aggregate.p50ErrorMm,
            p95DeltaMm =
                a.aggregate.p95ErrorMm -
                    b.aggregate.p95ErrorMm,
            p99DeltaMm =
                a.aggregate.p99ErrorMm -
                    b.aggregate.p99ErrorMm,
            repeatabilityDeltaMm =
                a.aggregate.repeatabilityMm -
                    b.aggregate.repeatabilityMm,
            reproducibilityDeltaMm =
                a.aggregate.reproducibilityMm -
                    b.aggregate.reproducibilityMm,
            biasDeltaMm =
                a.aggregate.biasMagnitudeMm -
                    b.aggregate.biasMagnitudeMm,
            lagDeltaMs =
                a.aggregate.lagMs -
                    b.aggregate.lagMs,
            note =
                "Descriptive comparison only. Matched capture conditions and external-reference uncertainty remain part of interpretation.",
        )

    @Synchronized
    fun diagnostics(): List<String> =
        diagnosticHistory.toList()

    @Synchronized
    fun reset() {
        check(!closed)
        diagnosticHistory.clear()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        diagnosticHistory.clear()
        closed = true
    }

    private fun evaluateJoint(
        jointId: Int,
        samples: List<Sample>,
    ): JointMetrics {
        val valid = samples.filter {
            it.visible &&
                it.confidence >= 0.35
        }

        val errors = valid.map(::errorMm)
        val p95Ci =
            bootstrapConfidenceInterval(
                values = errors,
                captureId = "joint:" + jointId + ":p95",
                statistic = { percentile(it, 0.95) },
            )
        val p99Ci =
            bootstrapConfidenceInterval(
                values = errors,
                captureId = "joint:" + jointId + ":p99",
                statistic = { percentile(it, 0.99) },
            )

        val bias =
            biasVector(valid)

        val worstMotion =
            samples.groupBy { it.motionRegime }
                .values
                .map { percentile(it.filter { s -> s.visible }.map(::errorMm), 0.95) }
                .maxOrNull()
                ?: Double.POSITIVE_INFINITY

        val worstOcclusion =
            samples.groupBy { it.occlusionRegime }
                .values
                .map { percentile(it.filter { s -> s.visible }.map(::errorMm), 0.95) }
                .maxOrNull()
                ?: Double.POSITIVE_INFINITY

        return JointMetrics(
            jointId = jointId,
            sampleCount = samples.size,
            validCount = valid.size,
            dropoutRatio =
                1.0 -
                    valid.size.toDouble() /
                    samples.size.coerceAtLeast(1).toDouble(),
            meanErrorMm = mean(errors),
            rmsErrorMm = rms(errors),
            p50ErrorMm = percentile(errors, 0.50),
            p95ErrorMm = percentile(errors, 0.95),
            p99ErrorMm = percentile(errors, 0.99),
            p95ConfidenceInterval = p95Ci,
            p99ConfidenceInterval = p99Ci,
            biasVectorMm = bias,
            biasMagnitudeMm = vectorLength(bias),
            repeatabilityMm = withinSessionRepeatability(valid),
            worstMotionRegimeP95Mm = worstMotion,
            worstOcclusionRegimeP95Mm = worstOcclusion,
        )
    }

    private fun evaluateSurface(
        samples: List<SurfaceSample>,
    ): SurfaceMetrics? {
        if (samples.isEmpty()) return null

        val valid = samples.filter {
            it.confidence >= 0.35
        }

        val errors = valid.map {
            distanceMm(
                it.estimate,
                it.reference,
            )
        }

        val normalErrors = valid.mapNotNull {
            val a = normalize(it.estimateNormal)
            val b = normalize(it.referenceNormal)
            if (a == null || b == null) null
            else angleDegrees(a, b)
        }

        return SurfaceMetrics(
            sampleCount = samples.size,
            validCount = valid.size,
            p50PointErrorMm = percentile(errors, 0.50),
            p95PointErrorMm = percentile(errors, 0.95),
            p99PointErrorMm = percentile(errors, 0.99),
            meanNormalErrorDegrees =
                normalErrors.takeIf { it.isNotEmpty() }
                    ?.let(::mean),
        )
    }

    private fun evaluateRegimes(
        samples: List<Sample>,
    ): List<RegimeMetrics> {
        val byMotion =
            samples.groupBy { "motion:" + it.motionRegime.name }
        val byOcclusion =
            samples.groupBy { "occlusion:" + it.occlusionRegime.name }
        val byLighting =
            samples.groupBy { "lighting:" + it.lightingRegime.name }

        return (
            byMotion +
                byOcclusion +
                byLighting
            )
            .entries
            .take(config.maximumRegimeBuckets)
            .map { (key, group) ->
                val valid = group.filter {
                    it.visible &&
                        it.confidence >= 0.35
                }
                val errors = valid.map(::errorMm)
                RegimeMetrics(
                    key = key,
                    sampleCount = group.size,
                    p50Mm = percentile(errors, 0.50),
                    p95Mm = percentile(errors, 0.95),
                    p99Mm = percentile(errors, 0.99),
                    dropoutRatio =
                        1.0 -
                            valid.size.toDouble() /
                            group.size.coerceAtLeast(1).toDouble(),
                    repeatabilityMm =
                        withinSessionRepeatability(valid),
                )
            }
            .sortedBy { it.key }
    }

    private fun evaluateReproducibility(
        samples: List<Sample>,
    ): Reproducibility {
        val groups =
            samples.groupBy { it.sessionId }

        val sessionMeans =
            groups.mapValues { (_, values) ->
                mean(values.map(::errorMm))
            }

        val sessionBiases =
            groups.mapValues { (_, values) ->
                vectorLength(
                    biasVector(values),
                )
            }

        val within =
            withinSessionRepeatability(samples)

        val sessionMeanList =
            sessionMeans.values.filter(Double::isFinite)

        val between =
            if (sessionMeanList.size < 2) {
                0.0
            } else {
                rms(
                    sessionMeanList.map { it - mean(sessionMeanList) },
                )
            }

        return Reproducibility(
            sessionCount = groups.size,
            withinSessionRepeatabilityMm = within,
            betweenSessionReproducibilityMm = between,
            sessionMeansMm = sessionMeans,
            sessionBiasMm = sessionBiases,
        )
    }

    private fun certify(
        input: Input,
        aggregate: Aggregate,
        reproducibility: Reproducibility,
    ): Certification {
        val blockers = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        val evidenceSufficient =
            aggregate.sampleCount >= config.minimumSamples &&
                reproducibility.sessionCount >=
                    config.minimumIndependentSessions &&
                aggregate.p95ConfidenceInterval != null &&
                aggregate.p99ConfidenceInterval != null

        val referenceUncertaintyKnown =
            aggregate.referenceUncertaintyMm != null

        val p95Upper =
            aggregate.p95ConfidenceInterval?.upper
        val p99Upper =
            aggregate.p99ConfidenceInterval?.upper

        val twoMm =
            evidenceSufficient &&
                input.matchedReferenceConditions &&
                referenceUncertaintyKnown &&
                p95Upper != null &&
                p99Upper != null &&
                p95Upper <= config.p95TargetMm &&
                p99Upper <= config.p99TargetMm &&
                aggregate.repeatabilityMm <=
                    config.repeatabilityTargetMm &&
                reproducibility.betweenSessionReproducibilityMm <=
                    config.reproducibilityTargetMm &&
                aggregate.dropoutRatio <=
                    config.dropoutTarget &&
                (aggregate.referenceUncertaintyMm
                    ?: Double.POSITIVE_INFINITY) <=
                    config.referenceUncertaintyTargetMm

        val oneMm =
            evidenceSufficient &&
                aggregate.sampleCount >=
                    config.minimumSamplesForSubMillimeterEvidence &&
                reproducibility.sessionCount >=
                    config.minimumIndependentSessions &&
                input.matchedReferenceConditions &&
                referenceUncertaintyKnown &&
                p95Upper != null &&
                p99Upper != null &&
                p95Upper <=
                    config.subMillimeterP95TargetMm &&
                p99Upper <=
                    config.subMillimeterP99TargetMm &&
                aggregate.repeatabilityMm <=
                    config.subMillimeterRepeatabilityTargetMm &&
                (aggregate.referenceUncertaintyMm
                    ?: Double.POSITIVE_INFINITY) <=
                    config.subMillimeterReferenceUncertaintyTargetMm

        if (!evidenceSufficient) {
            warnings += "empirical-evidence-insufficient"
        }
        if (!input.matchedReferenceConditions) {
            warnings += "reference-conditions-not-matched"
        }
        if (!referenceUncertaintyKnown) {
            warnings += "reference-uncertainty-unknown"
        }
        if (aggregate.referenceUncertaintyMm != null &&
            aggregate.referenceUncertaintyMm >
                config.referenceUncertaintyTargetMm
        ) {
            blockers +=
                "reference-uncertainty-too-large-for-two-millimeter-gate"
        }
        if (p95Upper != null &&
            p95Upper > config.p95TargetMm
        ) {
            blockers +=
                "p95-upper-confidence-bound-exceeds-two-millimeter-target"
        }
        if (p99Upper != null &&
            p99Upper > config.p99TargetMm
        ) {
            warnings +=
                "p99-upper-confidence-bound-exceeds-five-millimeter-target"
        }
        if (aggregate.repeatabilityMm >
            config.repeatabilityTargetMm
        ) {
            warnings +=
                "within-session-repeatability-exceeds-target"
        }
        if (reproducibility.betweenSessionReproducibilityMm >
            config.reproducibilityTargetMm
        ) {
            warnings +=
                "between-session-reproducibility-exceeds-target"
        }
        if (aggregate.dropoutRatio > config.dropoutTarget) {
            warnings += "dropout-exceeds-target"
        }

        val score = (
            lowerBetter(
                p95Upper ?: Double.POSITIVE_INFINITY,
                config.p95TargetMm * 2.5,
            ) * 0.34 +
                lowerBetter(
                    p99Upper ?: Double.POSITIVE_INFINITY,
                    config.p99TargetMm * 2.5,
                ) * 0.16 +
                lowerBetter(
                    aggregate.repeatabilityMm,
                    config.repeatabilityTargetMm * 2.0,
                ) * 0.16 +
                lowerBetter(
                    reproducibility.betweenSessionReproducibilityMm,
                    config.reproducibilityTargetMm * 2.0,
                ) * 0.14 +
                lowerBetter(
                    aggregate.dropoutRatio,
                    max(config.dropoutTarget * 4.0, 0.01),
                ) * 0.10 +
                lowerBetter(
                    aggregate.lagMs,
                    config.lagTargetMs * 2.0,
                ) * 0.10
            ).coerceIn(0.0, 1.0)

        return Certification(
            referenceKind = input.referenceKind,
            empiricalEvidenceSufficient = evidenceSufficient,
            matchedConditions = input.matchedReferenceConditions,
            referenceUncertaintyKnown = referenceUncertaintyKnown,
            twoMillimeterEvidence = twoMm,
            oneMillimeterEvidence = oneMm,
            hardBlocked = blockers.isNotEmpty(),
            confidence = score,
            gates = listOf(
                gate(
                    name = "P95 upper confidence bound",
                    measured = p95Upper,
                    limit = config.p95TargetMm,
                    unit = "mm",
                    samples = aggregate.sampleCount,
                    reason =
                        "Quantile CI upper bound must remain inside target.",
                ),
                gate(
                    name = "P99 upper confidence bound",
                    measured = p99Upper,
                    limit = config.p99TargetMm,
                    unit = "mm",
                    samples = aggregate.sampleCount,
                    reason =
                        "Tail error must remain inside target.",
                ),
                gate(
                    name = "Within-session repeatability",
                    measured = aggregate.repeatabilityMm,
                    limit = config.repeatabilityTargetMm,
                    unit = "mm",
                    samples = aggregate.sampleCount,
                    reason =
                        "Short-term repeatability must remain bounded.",
                ),
                gate(
                    name = "Between-session reproducibility",
                    measured =
                        reproducibility.betweenSessionReproducibilityMm,
                    limit = config.reproducibilityTargetMm,
                    unit = "mm",
                    samples = aggregate.sampleCount,
                    reason =
                        "Independent sessions must remain mutually consistent.",
                ),
                gate(
                    name = "Reference uncertainty",
                    measured = aggregate.referenceUncertaintyMm,
                    limit = config.referenceUncertaintyTargetMm,
                    unit = "mm",
                    samples = aggregate.sampleCount,
                    reason =
                        "A certification claim cannot exceed the quality of its reference.",
                ),
            ),
            blockers = blockers.distinct(),
            warnings = warnings.distinct(),
        )
    }

    private fun gate(
        name: String,
        measured: Double?,
        limit: Double,
        unit: String,
        samples: Int,
        reason: String,
    ): Gate {
        if (
            samples < config.minimumSamples ||
            measured == null ||
            !measured.isFinite()
        ) {
            return Gate(
                name = name,
                state = GateState.UNKNOWN,
                measured = measured,
                upperConfidenceBound = measured,
                limit = limit,
                unit = unit,
                reason = reason,
            )
        }

        return Gate(
            name = name,
            state =
                if (measured <= limit) {
                    GateState.PASS
                } else {
                    GateState.FAIL
                },
            measured = measured,
            upperConfidenceBound = measured,
            limit = limit,
            unit = unit,
            reason = reason,
        )
    }

    private fun bootstrapConfidenceInterval(
        values: List<Double>,
        captureId: String,
        statistic: (List<Double>) -> Double,
    ): ConfidenceInterval? {
        val finite = values.filter(Double::isFinite)
        if (finite.size < config.minimumSamples) {
            return null
        }

        val seed =
            deterministicSeed(captureId)

        val rng =
            DeterministicRng(seed)

        val estimates =
            DoubleArray(config.bootstrapIterations)

        val sample =
            ArrayList<Double>(finite.size)

        var index = 0
        while (index < config.bootstrapIterations) {
            sample.clear()
            repeat(finite.size) {
                sample += finite[rng.nextInt(finite.size)]
            }
            estimates[index] =
                statistic(sample)
            index++
        }

        estimates.sort()
        val alpha =
            (1.0 - config.confidenceLevel) * 0.5

        val lower =
            percentile(
                estimates.asList(),
                alpha,
            )

        val center =
            statistic(finite)

        val upper =
            percentile(
                estimates.asList(),
                1.0 - alpha,
            )

        return ConfidenceInterval(
            lower = lower,
            center = center,
            upper = upper,
            confidenceLevel = config.confidenceLevel,
        )
    }

    private fun estimateLagMs(
        samples: List<Sample>,
    ): Double {
        if (samples.size < config.minimumSamples) {
            return 0.0
        }

        val grouped =
            samples.groupBy { it.jointId }

        val errorsByOffset =
            linkedMapOf<Long, Double>()

        val offsets =
            listOf(
                -50_000_000L,
                -40_000_000L,
                -30_000_000L,
                -20_000_000L,
                -10_000_000L,
                0L,
                10_000_000L,
                20_000_000L,
                30_000_000L,
                40_000_000L,
                50_000_000L,
            )

        for (offset in offsets) {
            val pairs =
                grouped.values.flatMap { joint ->
                    val sorted = joint.sortedBy { it.timestampNanos }
                    val out =
                        ArrayList<Double>()

                    for (estimate in sorted) {
                        val targetTime =
                            estimate.timestampNanos + offset

                        val nearest =
                            sorted.minByOrNull {
                                abs(
                                    it.timestampNanos -
                                        targetTime,
                                )
                            } ?: continue

                        out +=
                            distanceMm(
                                estimate.reference,
                                nearest.reference,
                            )
                    }
                    out
                }

            if (pairs.isNotEmpty()) {
                errorsByOffset[offset] =
                    mean(pairs)
            }
        }

        val best =
            errorsByOffset.minByOrNull { it.value }
                ?.key
                ?: 0L

        return abs(best).toDouble() / 1_000_000.0
    }

    private fun withinSessionRepeatability(
        samples: List<Sample>,
    ): Double {
        val groups =
            samples.groupBy {
                it.sessionId
            }

        val pairRms =
            groups.values.mapNotNull { session ->
                val sorted =
                    session.filter { it.visible }
                        .sortedBy { it.timestampNanos }

                if (sorted.size < 3) {
                    null
                } else {
                    val deltas =
                        sorted.zipWithNext().map {
                            distanceMm(
                                it.first.estimate,
                                it.second.estimate,
                            )
                        }
                    rms(deltas)
                }
            }

        return pairRms.takeIf { it.isNotEmpty() }
            ?.let(::rms)
            ?: 0.0
    }

    private fun betweenSessionReproducibility(
        samples: List<Sample>,
    ): Double {
        val groups =
            samples.groupBy {
                it.sessionId
            }

        if (groups.size < 2) return 0.0

        val sessionMean =
            groups.mapValues { (_, values) ->
                mean(
                    values.map(::errorMm),
                )
            }

        val means =
            sessionMean.values.filter(Double::isFinite)

        if (means.size < 2) return 0.0

        val center =
            mean(means)

        return rms(
            means.map {
                it - center
            },
        )
    }

    private fun biasVector(
        samples: List<Sample>,
    ): Vec3 {
        val valid =
            samples.filter {
                it.visible &&
                    it.confidence >= 0.35
            }

        if (valid.isEmpty()) {
            return Vec3(
                0.0,
                0.0,
                0.0,
            )
        }

        return Vec3(
            valid.map {
                it.estimate.x - it.reference.x
            }.average() * 1000.0,
            valid.map {
                it.estimate.y - it.reference.y
            }.average() * 1000.0,
            valid.map {
                it.estimate.z - it.reference.z
            }.average() * 1000.0,
        )
    }

    private fun biasMagnitude(
        samples: List<Sample>,
    ): Double =
        vectorLength(
            biasVector(samples),
        )

    private fun withinSessionRepeatabilityForJoint(
        samples: List<Sample>,
    ): Double =
        withinSessionRepeatability(samples)

    private fun referenceUncertainty(
        input: Input,
        samples: List<Sample>,
    ): Double? {
        input.referenceUncertaintyMm?.let {
            return it
        }

        val values =
            samples.mapNotNull {
                it.referenceUncertaintyMm
            }

        return values.takeIf {
            it.isNotEmpty()
        }?.let { median(it, 0.0) }
    }

    private fun scaleDrift(
        samples: List<Sample>,
    ): Double? {
        val values =
            samples.mapNotNull {
                val estimate =
                    it.estimateScale ?: return@mapNotNull null
                val reference =
                    it.referenceScale ?: return@mapNotNull null

                val ratio =
                    (estimate / reference)
                        .takeIf(Double::isFinite)
                        ?: return@mapNotNull null

                it.timestampNanos /
                    1_000_000_000.0 to ratio
            }

        if (values.size < 4) return null

        val meanX =
            values.map { it.first }.average()

        val meanY =
            values.map { it.second }.average()

        var numerator = 0.0
        var denominator = 0.0

        for ((x, y) in values) {
            val dx = x - meanX
            val dy = y - meanY
            numerator += dx * dy
            denominator += dx * dx
        }

        if (denominator <= 1e-12) return null

        return abs(
            numerator /
                denominator,
        ) * 60.0 * 100.0
    }

    private fun blandAltman(
        samples: List<Sample>,
    ): BlandAltman? {
        val valid =
            samples.filter {
                it.visible &&
                    it.confidence >= 0.35
            }

        if (valid.size < config.minimumSamples) {
            return null
        }

        val differences =
            valid.map { errorMm(it) }

        val meanDifference =
            mean(differences)

        val sd =
            standardDeviation(differences)

        return BlandAltman(
            sampleCount = valid.size,
            meanBiasMm = meanDifference,
            standardDeviationMm = sd,
            lowerAgreementMm =
                meanDifference - 1.96 * sd,
            upperAgreementMm =
                meanDifference + 1.96 * sd,
        )
    }

    private fun errorMm(
        sample: Sample,
    ): Double =
        distanceMm(
            sample.estimate,
            sample.reference,
        )

    private fun distanceMm(
        a: Vec3,
        b: Vec3,
    ): Double =
        sqrt(
            (a.x - b.x).let { it * it } +
                (a.y - b.y).let { it * it } +
                (a.z - b.z).let { it * it },
        ) * 1000.0

    private fun normalize(
        value: Vec3?,
    ): Vec3? {
        value ?: return null
        val magnitude =
            vectorLength(value)

        if (
            !magnitude.isFinite() ||
            magnitude <= 1e-9
        ) {
            return null
        }

        return Vec3(
            value.x / magnitude,
            value.y / magnitude,
            value.z / magnitude,
        )
    }

    private fun angleDegrees(
        a: Vec3,
        b: Vec3,
    ): Double =
        Math.toDegrees(
            acos(
                (
                    a.x * b.x +
                        a.y * b.y +
                        a.z * b.z
                    ).coerceIn(-1.0, 1.0),
            ),
        )

    private fun vectorLength(
        value: Vec3,
    ): Double =
        sqrt(
            value.x * value.x +
                value.y * value.y +
                value.z * value.z,
        )

    private fun standardDeviation(
        values: List<Double>,
    ): Double {
        val finite =
            values.filter(Double::isFinite)

        if (finite.size < 2) {
            return 0.0
        }

        val center =
            finite.average()

        return sqrt(
            finite.sumOf {
                val delta =
                    it - center
                delta * delta
            } /
                (finite.size - 1).toDouble(),
        )
    }

    private fun mean(
        values: List<Double>,
    ): Double =
        values.filter(Double::isFinite)
            .takeIf { it.isNotEmpty() }
            ?.average()
            ?: Double.POSITIVE_INFINITY

    private fun rms(
        values: List<Double>,
    ): Double {
        val finite =
            values.filter(Double::isFinite)

        if (finite.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        return sqrt(
            finite.sumOf {
                it * it
            } /
                finite.size.toDouble(),
        )
    }

    private fun percentile(
        values: List<Double>,
        probability: Double,
    ): Double {
        val sorted =
            values.filter(Double::isFinite)
                .sorted()

        if (sorted.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        if (sorted.size == 1) {
            return sorted.first()
        }

        val position =
            probability.coerceIn(0.0, 1.0) *
                (sorted.size - 1).toDouble()

        val lower =
            position.toInt()

        val upper =
            min(
                lower + 1,
                sorted.lastIndex,
            )

        val fraction =
            position - lower

        return sorted[lower] +
            (sorted[upper] - sorted[lower]) *
            fraction
    }

    private fun lowerBetter(
        value: Double,
        limit: Double,
    ): Double =
        if (
            !value.isFinite() ||
            limit <= 0.0
        ) {
            0.0
        } else {
            (
                1.0 -
                    value /
                    limit
                ).coerceIn(0.0, 1.0)
        }

    private fun deterministicSeed(
        value: String,
    ): Long {
        var hash = 1469598103934665603L
        for (byte in value.encodeToByteArray()) {
            hash = hash xor byte.toLong()
            hash *= 1099511628211L
        }
        return hash
    }

    private class DeterministicRng(
        seed: Long,
    ) {
        private var state =
            seed xor -7046029254386353131L

        fun nextLong(): Long {
            var z =
                state + -7046029254386353131L
            state = z
            z =
                (z xor (z ushr 30)) *
                    -4658895280553007687L
            z =
                (z xor (z ushr 27)) *
                    -7723592293110705685L
            return z xor (z ushr 31)
        }

        fun nextInt(
            bound: Int,
        ): Int {
            require(bound > 0)
            var value =
                nextLong() ushr 1
            val remainder =
                value % bound.toLong()
            if (
                value -
                remainder +
                (bound - 1) >= 0
            ) {
                return remainder.toInt()
            }
            while (true) {
                value =
                    nextLong() ushr 1
                val candidate =
                    value % bound.toLong()
                if (
                    value -
                    candidate +
                    (bound - 1) >= 0
                ) {
                    return candidate.toInt()
                }
            }
        }
    }

    private fun metric(
        value: Double,
    ): String =
        "%.4f".format(
            value.coerceIn(
                -9999.0,
                9999.0,
            ),
        )

    private fun formatInterval(
        interval: ConfidenceInterval?,
    ): String =
        interval?.let {
            "[" +
                metric(it.lower) +
                "," +
                metric(it.upper) +
                "]"
        } ?: "UNKNOWN"

    private fun appendDiagnostic(
        value: String,
    ) {
        diagnosticHistory.addLast(value)
        while (
            diagnosticHistory.size >
            config.maximumDiagnosticHistory
        ) {
            diagnosticHistory.removeFirst()
        }
    }
}
