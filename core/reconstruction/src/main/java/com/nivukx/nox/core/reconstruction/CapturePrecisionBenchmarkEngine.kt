package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Deterministic production benchmark for quantitative motion-capture accuracy.
 *
 * This is an evaluation layer, not a model. It makes residual accuracy, jitter, dropout,
 * temporal lag and scale drift explicit so capture quality can be regression-tested and
 * compared against a documented external reference.
 */
class CapturePrecisionBenchmarkEngine(
    private val config: Config = Config(),
) {
    enum class ReferenceType {
        OPTICAL_REFERENCE,
        MARKER_REFERENCE,
        ACTIVE_OPTICAL_MARKER_REFERENCE,
        SYNTHETIC_GROUND_TRUTH,
        MANUAL_REFERENCE,
    }

    enum class GateStatus { PASS, FAIL, UNKNOWN }

    data class Config(
        val p50LimitMm: Double = 2.0,
        val p95LimitMm: Double = 5.0,
        val p99LimitMm: Double = 8.0,
        val jitterLimitMm: Double = 2.5,
        val dropoutLimit: Double = 0.02,
        val lagLimitMs: Double = 25.0,
        val scaleDriftLimitPercentPerMinute: Double = 0.50,
        val minimumConfidence: Double = 0.35,
        val minimumSamples: Int = 24,
        val maximumSamples: Int = 400_000,
        val maximumJoints: Int = 512,
        val maximumReferenceGapNanos: Long = 40_000_000L,
        val lagSearchNanos: Long = 60_000_000L,
        val lagStepNanos: Long = 2_000_000L,
        val lagCoarseStepNanos: Long = 8_000_000L,
        val lagFineRadiusSteps: Int = 2,
    ) {
        init {
            require(p50LimitMm > 0.0)
            require(p95LimitMm >= p50LimitMm)
            require(p99LimitMm >= p95LimitMm)
            require(jitterLimitMm > 0.0)
            require(dropoutLimit in 0.0..1.0)
            require(lagLimitMs > 0.0)
            require(scaleDriftLimitPercentPerMinute > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(minimumSamples > 0)
            require(maximumSamples >= minimumSamples)
            require(maximumJoints > 0)
            require(maximumReferenceGapNanos > 0L)
            require(lagSearchNanos >= lagStepNanos)
            require(lagStepNanos > 0L)
            require(
                lagCoarseStepNanos >= lagStepNanos &&
                    lagCoarseStepNanos <= lagSearchNanos,
            )
            require(lagFineRadiusSteps in 1..8)
        }

        companion object {
            fun ultraPrecision(): Config =
                Config(
                    p50LimitMm = 0.50,
                    p95LimitMm = 0.75,
                    p99LimitMm = 1.75,
                    jitterLimitMm = 0.50,
                    dropoutLimit = 0.01,
                    lagLimitMs = 5.0,
                    scaleDriftLimitPercentPerMinute = 0.25,
                    minimumConfidence = 0.75,
                    minimumSamples = 600,
                    maximumSamples = 400_000,
                    maximumJoints = 512,
                    maximumReferenceGapNanos = 10_000_000L,
                    lagSearchNanos = 30_000_000L,
                    lagStepNanos = 500_000L,
                )
        }
    }

    data class JointSample(
        val timestampNanos: Long,
        val jointId: Int,
        val estimate: Vec3,
        val reference: Vec3,
        val confidence: Double,
        val visible: Boolean = true,
        val estimatedScale: Double? = null,
        val referenceScale: Double? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(jointId >= 0)
            requireFinite(estimate)
            requireFinite(reference)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            estimatedScale?.let { require(it.isFinite() && it > 0.0) }
            referenceScale?.let { require(it.isFinite() && it > 0.0) }
        }
    }

    data class SurfaceSample(
        val timestampNanos: Long,
        val estimate: Vec3,
        val reference: Vec3,
        val estimateNormal: Vec3? = null,
        val referenceNormal: Vec3? = null,
        val confidence: Double = 1.0,
        val visible: Boolean = true,
    ) {
        init {
            require(timestampNanos >= 0L)
            requireFinite(estimate)
            requireFinite(reference)
            require(confidence in 0.0..1.0)
            estimateNormal?.let(::requireFinite)
            referenceNormal?.let(::requireFinite)
        }
    }

    data class Input(
        val captureId: String,
        val referenceType: ReferenceType,
        val samples: List<JointSample>,
        val surfaceSamples: List<SurfaceSample> = emptyList(),
        val referenceUncertaintyMillimeters: Double? = null,
    ) {
        init {
            require(captureId.isNotBlank())
            require(samples.isNotEmpty())
            referenceUncertaintyMillimeters?.let {
                require(it.isFinite() && it >= 0.0)
            }
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
        val maxErrorMm: Double,
        val jitterRmsMm: Double,
        val scaleRatio: Double?,
    )

    data class SurfaceMetrics(
        val sampleCount: Int,
        val validCount: Int,
        val meanPointErrorMm: Double,
        val p95PointErrorMm: Double,
        val p99PointErrorMm: Double,
        val maxPointErrorMm: Double,
        val meanNormalErrorDegrees: Double?,
    )

    data class Aggregate(
        val sampleCount: Int,
        val validCount: Int,
        val jointCount: Int,
        val dropoutRatio: Double,
        val meanErrorMm: Double,
        val rmsErrorMm: Double,
        val p50ErrorMm: Double,
        val p95ErrorMm: Double,
        val p99ErrorMm: Double,
        val maxErrorMm: Double,
        val jitterRmsMm: Double,
        val temporalLagMs: Double,
        val scaleRatio: Double?,
        val scaleDriftPercentPerMinute: Double?,
        val surface: SurfaceMetrics?,
    )

    data class Gate(
        val name: String,
        val status: GateStatus,
        val measured: Double?,
        val limit: Double?,
        val unit: String,
    )

    data class Report(
        val captureId: String,
        val referenceType: ReferenceType,
        val referenceUncertaintyMillimeters: Double?,
        val aggregate: Aggregate,
        val jointMetrics: List<JointMetrics>,
        val gates: List<Gate>,
        val confidence: Double,
        val accepted: Boolean,
        val warnings: List<String>,
    )

    data class Comparison(
        val referenceType: ReferenceType,
        val deltaP50Mm: Double,
        val deltaP95Mm: Double,
        val deltaP99Mm: Double,
        val deltaJitterMm: Double,
        val deltaDropout: Double,
        val deltaLagMs: Double,
        val deltaScaleDrift: Double?,
        val note: String,
    )

    private data class Timed(
        val timestampNanos: Long,
        val point: Vec3,
        val confidence: Double,
    )

    /**
     * Strict comparative verdict for external reference runs.
     *
     * "BETTER" is emitted only when both reports have the same reference family, documented
     * reference uncertainty, enough samples, and NoX improves every primary metric used by the
     * capture contract. Otherwise the verdict is QUALIFIED, EQUIVALENT, INCONCLUSIVE, or WORSE.
     *
     * This deliberately avoids equating a modeled target with a physical guarantee.
     */
    enum class ReferenceVerdict {
        BETTER,
        EQUIVALENT,
        WORSE,
        INCONCLUSIVE,
    }

    data class ReferenceComparison(
        val verdict: ReferenceVerdict,
        val p95DeltaMm: Double,
        val p99DeltaMm: Double,
        val jitterDeltaMm: Double,
        val dropoutDelta: Double,
        val lagDeltaMs: Double,
        val uncertaintyAware: Boolean,
        val sampleComparable: Boolean,
        val rationale: List<String>,
    )

    fun qualifySubMillimeter(
        input: Input,
        temporalLagMs: Double?,
        protocol: SubMillimeterQualificationProtocol =
            SubMillimeterQualificationProtocol(),
    ): SubMillimeterQualificationProtocol.QualificationReport {
        require(input.samples.isNotEmpty())
        require(
            input.referenceType ==
                ReferenceType.MARKER_REFERENCE ||
                input.referenceType ==
                ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE ||
                input.referenceType ==
                ReferenceType.OPTICAL_REFERENCE
        )

        return protocol.qualify(
            samples =
                input.samples.map { sample ->
                    SubMillimeterQualificationProtocol.Sample(
                        timestampNanos = sample.timestampNanos,
                        estimate = sample.estimate,
                        reference = sample.reference,
                        referenceUncertaintyMm =
                            input.referenceUncertaintyMillimeters
                                ?: Double.POSITIVE_INFINITY,
                        confidence = sample.confidence,
                        visible = sample.visible,
                    )
                },
            temporalLagMs = temporalLagMs,
        )
    }

    fun compareAgainstReference(
        nox: Report,
        reference: Report,
        maximumEquivalentDeltaMm: Double = 0.02,
        maximumEquivalentDeltaRatio: Double = 0.02,
    ): ReferenceComparison {
        require(maximumEquivalentDeltaMm >= 0.0)
        require(maximumEquivalentDeltaRatio >= 0.0)

        val sampleComparable =
            nox.aggregate.sampleCount >= config.minimumSamples &&
                reference.aggregate.sampleCount >= config.minimumSamples

        val uncertaintyAware =
            reference.referenceUncertaintyMillimeters
                ?.isFinite() == true &&
                reference.referenceUncertaintyMillimeters
                    ?.let { it >= 0.0 } == true

        val sameReferenceFamily =
            nox.referenceType == reference.referenceType &&
                (
                    nox.referenceType ==
                        ReferenceType.OPTICAL_REFERENCE ||
                        nox.referenceType ==
                        ReferenceType.MARKER_REFERENCE ||
                    nox.referenceType ==
                        ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE
                )

        val p95Delta =
            nox.aggregate.p95ErrorMm -
                reference.aggregate.p95ErrorMm
        val p99Delta =
            nox.aggregate.p99ErrorMm -
                reference.aggregate.p99ErrorMm
        val jitterDelta =
            nox.aggregate.jitterRmsMm -
                reference.aggregate.jitterRmsMm
        val dropoutDelta =
            nox.aggregate.dropoutRatio -
                reference.aggregate.dropoutRatio
        val lagDelta =
            nox.aggregate.temporalLagMs -
                reference.aggregate.temporalLagMs

        if (
            !sampleComparable ||
            !uncertaintyAware ||
            !sameReferenceFamily
        ) {
            return ReferenceComparison(
                verdict = ReferenceVerdict.INCONCLUSIVE,
                p95DeltaMm = p95Delta,
                p99DeltaMm = p99Delta,
                jitterDeltaMm = jitterDelta,
                dropoutDelta = dropoutDelta,
                lagDeltaMs = lagDelta,
                uncertaintyAware = uncertaintyAware,
                sampleComparable = sampleComparable,
                rationale =
                    buildList {
                        if (!sampleComparable) {
                            add("reference-sample-count-not-comparable")
                        }
                        if (!uncertaintyAware) {
                            add("reference-uncertainty-not-documented")
                        }
                        if (!sameReferenceFamily) {
                            add("reference-family-mismatch")
                        }
                    },
            )
        }

        val referenceScale =
            max(
                reference.aggregate.p95ErrorMm,
                1e-6,
            )

        val equivalentDelta =
            max(
                maximumEquivalentDeltaMm,
                referenceScale *
                    maximumEquivalentDeltaRatio,
            )

        val p95Better = p95Delta < -equivalentDelta
        val p99Better = p99Delta < -equivalentDelta
        val jitterBetter = jitterDelta < -equivalentDelta
        val dropoutBetter =
            dropoutDelta <
                -maximumEquivalentDeltaRatio
        val lagBetter =
            lagDelta <
                -maximumEquivalentDeltaRatio

        val p95Worse = p95Delta > equivalentDelta
        val p99Worse = p99Delta > equivalentDelta
        val jitterWorse = jitterDelta > equivalentDelta
        val dropoutWorse =
            dropoutDelta >
                maximumEquivalentDeltaRatio
        val lagWorse =
            lagDelta >
                maximumEquivalentDeltaRatio

        val allBetter =
            p95Better &&
                p99Better &&
                jitterBetter &&
                dropoutBetter &&
                lagBetter
        val allWorse =
            p95Worse &&
                p99Worse &&
                jitterWorse &&
                dropoutWorse &&
                lagWorse

        val verdict =
            when {
                allBetter -> ReferenceVerdict.BETTER
                allWorse -> ReferenceVerdict.WORSE
                listOf(
                    p95Delta,
                    p99Delta,
                    jitterDelta,
                    dropoutDelta,
                    lagDelta,
                ).all {
                    abs(it) <= equivalentDelta
                } -> ReferenceVerdict.EQUIVALENT
                else -> ReferenceVerdict.INCONCLUSIVE
            }

        val rationale =
            buildList {
                add("matched-reference-family")
                add("reference-uncertainty-documented")
                if (p95Better) add("p95-improved")
                if (p99Better) add("p99-improved")
                if (jitterBetter) add("jitter-improved")
                if (dropoutBetter) add("dropout-improved")
                if (lagBetter) add("lag-improved")
                if (p95Worse) add("p95-worse")
                if (p99Worse) add("p99-worse")
                if (jitterWorse) add("jitter-worse")
                if (dropoutWorse) add("dropout-worse")
                if (lagWorse) add("lag-worse")
            }

        return ReferenceComparison(
            verdict = verdict,
            p95DeltaMm = p95Delta,
            p99DeltaMm = p99Delta,
            jitterDeltaMm = jitterDelta,
            dropoutDelta = dropoutDelta,
            lagDeltaMs = lagDelta,
            uncertaintyAware = uncertaintyAware,
            sampleComparable = sampleComparable,
            rationale = rationale,
        )
    }

    fun evaluate(input: Input): Report {
        val bounded = sanitize(input)
        val joints = bounded.samples
            .groupBy { it.jointId }
            .toSortedMap()
            .entries
            .take(config.maximumJoints)
            .map { evaluateJoint(it.key, it.value) }

        val valid = bounded.samples.filter {
            it.visible && it.confidence >= config.minimumConfidence
        }
        val errors = valid.map { distance(it.estimate, it.reference) * 1000.0 }
        val aggregate = Aggregate(
            sampleCount = bounded.samples.size,
            validCount = valid.size,
            jointCount = joints.size,
            dropoutRatio = 1.0 - valid.size.toDouble() /
                bounded.samples.size.coerceAtLeast(1).toDouble(),
            meanErrorMm = mean(errors),
            rmsErrorMm = rms(errors),
            p50ErrorMm = percentile(errors, 0.50),
            p95ErrorMm = percentile(errors, 0.95),
            p99ErrorMm = percentile(errors, 0.99),
            maxErrorMm = errors.maxOrNull() ?: Double.POSITIVE_INFINITY,
            jitterRmsMm = jitter(valid),
            temporalLagMs = lag(valid),
            scaleRatio = scaleRatio(valid),
            scaleDriftPercentPerMinute = scaleDrift(valid),
            surface = evaluateSurface(bounded.surfaceSamples),
        )

        val gates = gates(aggregate, bounded.samples.size)
        val known = gates.filter { it.status != GateStatus.UNKNOWN }
        val gateScore = if (known.isEmpty()) 0.0
        else known.count { it.status == GateStatus.PASS }.toDouble() / known.size

        val confidence = (
            lowerBetter(aggregate.p95ErrorMm, config.p95LimitMm * 2.0) * 0.40 +
                lowerBetter(aggregate.dropoutRatio, max(config.dropoutLimit * 4.0, 0.01)) * 0.15 +
                lowerBetter(aggregate.temporalLagMs, config.lagLimitMs * 2.0) * 0.15 +
                gateScore * 0.15 +
                joints.count { it.validCount > 0 }.toDouble() /
                    joints.size.coerceAtLeast(1).toDouble() * 0.15
            ).coerceIn(0.0, 1.0)

        val warnings = buildList {
            if (bounded.samples.size < config.minimumSamples) {
                add("Insufficient samples for an authoritative precision gate.")
            }
            gates.filter { it.status == GateStatus.FAIL }
                .forEach { add(it.name + " exceeds the configured target.") }
            if (
                bounded.referenceType == ReferenceType.OPTICAL_REFERENCE ||
                bounded.referenceType == ReferenceType.MARKER_REFERENCE ||
                bounded.referenceType ==
                    ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE
            ) {
                if (bounded.referenceUncertaintyMillimeters == null) {
                    add("External reference uncertainty is not documented.")
                } else {
                    add(
                        "External reference uncertainty: " +
                            bounded.referenceUncertaintyMillimeters +
                            " mm.",
                    )
                }
            }
        }

        val externalReference =
            bounded.referenceType == ReferenceType.OPTICAL_REFERENCE ||
                bounded.referenceType == ReferenceType.MARKER_REFERENCE ||
                bounded.referenceType ==
                    ReferenceType.ACTIVE_OPTICAL_MARKER_REFERENCE
        val accepted =
            bounded.samples.size >= config.minimumSamples &&
                gates.none { it.status == GateStatus.FAIL } &&
                known.any { it.status == GateStatus.PASS } &&
                (!externalReference ||
                    bounded.referenceUncertaintyMillimeters != null) &&
                confidence >= 0.70

        return Report(
            captureId = bounded.captureId,
            referenceType = bounded.referenceType,
            referenceUncertaintyMillimeters =
                bounded.referenceUncertaintyMillimeters,
            aggregate = aggregate,
            jointMetrics = joints,
            gates = gates,
            confidence = confidence,
            accepted = accepted,
            warnings = warnings,
        )
    }

    fun compare(estimate: Report, reference: Report): Comparison {
        val a = estimate.aggregate
        val b = reference.aggregate
        return Comparison(
            referenceType = reference.referenceType,
            deltaP50Mm = a.p50ErrorMm - b.p50ErrorMm,
            deltaP95Mm = a.p95ErrorMm - b.p95ErrorMm,
            deltaP99Mm = a.p99ErrorMm - b.p99ErrorMm,
            deltaJitterMm = a.jitterRmsMm - b.jitterRmsMm,
            deltaDropout = a.dropoutRatio - b.dropoutRatio,
            deltaLagMs = a.temporalLagMs - b.temporalLagMs,
            deltaScaleDrift =
                if (a.scaleDriftPercentPerMinute != null &&
                    b.scaleDriftPercentPerMinute != null
                ) {
                    a.scaleDriftPercentPerMinute -
                        b.scaleDriftPercentPerMinute
                } else null,
            note = "Comparison is descriptive; reference uncertainty and matched capture conditions remain part of interpretation.",
        )
    }

    private fun sanitize(input: Input): Input {
        val samples = input.samples.asSequence()
            .sortedWith(compareBy<JointSample> { it.timestampNanos }.thenBy { it.jointId })
            .take(config.maximumSamples)
            .toList()
        require(samples.isNotEmpty())
        val ids = samples.map { it.jointId }.distinct().take(config.maximumJoints).toSet()
        return input.copy(
            samples = samples.filter { it.jointId in ids },
            surfaceSamples = input.surfaceSamples
                .sortedBy { it.timestampNanos }
                .take(config.maximumSamples.coerceAtMost(120_000)),
        )
    }

    private fun evaluateJoint(id: Int, samples: List<JointSample>): JointMetrics {
        val valid = samples.filter {
            it.visible && it.confidence >= config.minimumConfidence
        }
        val errors = valid.map { distance(it.estimate, it.reference) * 1000.0 }
        return JointMetrics(
            jointId = id,
            sampleCount = samples.size,
            validCount = valid.size,
            dropoutRatio = 1.0 - valid.size.toDouble() / samples.size.coerceAtLeast(1),
            meanErrorMm = mean(errors),
            rmsErrorMm = rms(errors),
            p50ErrorMm = percentile(errors, 0.50),
            p95ErrorMm = percentile(errors, 0.95),
            p99ErrorMm = percentile(errors, 0.99),
            maxErrorMm = errors.maxOrNull() ?: Double.POSITIVE_INFINITY,
            jitterRmsMm = jitter(valid),
            scaleRatio = scaleRatio(valid),
        )
    }

    private fun evaluateSurface(samples: List<SurfaceSample>): SurfaceMetrics? {
        if (samples.isEmpty()) return null
        val valid = samples.filter {
            it.visible && it.confidence >= config.minimumConfidence
        }
        val errors = valid.map { distance(it.estimate, it.reference) * 1000.0 }
        val normal = valid.mapNotNull {
            val a = normalize(it.estimateNormal)
            val b = normalize(it.referenceNormal)
            if (a == null || b == null) null else angleDegrees(a, b)
        }
        return SurfaceMetrics(
            sampleCount = samples.size,
            validCount = valid.size,
            meanPointErrorMm = mean(errors),
            p95PointErrorMm = percentile(errors, 0.95),
            p99PointErrorMm = percentile(errors, 0.99),
            maxPointErrorMm = errors.maxOrNull() ?: Double.POSITIVE_INFINITY,
            meanNormalErrorDegrees = normal.takeIf { it.isNotEmpty() }?.let(::mean),
        )
    }

    private fun gates(a: Aggregate, samples: Int): List<Gate> = listOf(
        gate("P50 position error", a.p50ErrorMm, config.p50LimitMm, "mm", samples),
        gate("P95 position error", a.p95ErrorMm, config.p95LimitMm, "mm", samples),
        gate("P99 position error", a.p99ErrorMm, config.p99LimitMm, "mm", samples),
        gate("Jitter RMS", a.jitterRmsMm, config.jitterLimitMm, "mm", samples),
        gate("Dropout ratio", a.dropoutRatio, config.dropoutLimit, "ratio", samples),
        gate("Temporal lag", a.temporalLagMs, config.lagLimitMs, "ms", samples),
        gate(
            "Scale drift",
            a.scaleDriftPercentPerMinute,
            config.scaleDriftLimitPercentPerMinute,
            "%/min",
            samples,
        ),
    )

    private fun gate(
        name: String,
        value: Double?,
        limit: Double,
        unit: String,
        samples: Int,
    ): Gate = when {
        samples < config.minimumSamples || value == null || !value.isFinite() ->
            Gate(name, GateStatus.UNKNOWN, value, limit, unit)
        value <= limit ->
            Gate(name, GateStatus.PASS, value, limit, unit)
        else ->
            Gate(name, GateStatus.FAIL, value, limit, unit)
    }

    private fun jitter(samples: List<JointSample>): Double {
        if (samples.size < 3) return 0.0
        val residuals = samples.sortedBy { it.timestampNanos }
            .map { it.estimate - it.reference }
        return rms(residuals.zipWithNext().map { distance(it.first, it.second) * 1000.0 })
    }

    private fun lag(samples: List<JointSample>): Double {
        if (samples.size < config.minimumSamples) return 0.0

        val series =
            samples
                .groupBy { it.jointId }
                .values

        fun scoreAt(
            candidateLag: Long,
        ): Double {
            var sum = 0.0
            var weight = 0.0

            for (joint in series) {
                val estimates =
                    joint
                        .sortedBy { it.timestampNanos }
                        .map {
                            Timed(
                                it.timestampNanos +
                                    candidateLag,
                                it.estimate,
                                it.confidence,
                            )
                        }

                val refs =
                    joint
                        .sortedBy { it.timestampNanos }
                        .map {
                            Timed(
                                it.timestampNanos,
                                it.reference,
                                it.confidence,
                            )
                        }

                for ((estimate, reference) in
                    pairs(
                        estimates,
                        refs,
                    )
                ) {
                    val gap =
                        abs(
                            estimate.timestampNanos -
                                reference.timestampNanos,
                        )

                    if (
                        gap >
                            config.maximumReferenceGapNanos
                    ) {
                        continue
                    }

                    val gapWeight =
                        (
                            1.0 -
                                gap.toDouble() /
                                    config.maximumReferenceGapNanos
                        ).coerceIn(
                            0.0,
                            1.0,
                        )

                    val confidenceWeight =
                        min(
                            estimate.confidence,
                            reference.confidence,
                        )

                    val residual =
                        distance(
                            estimate.point,
                            reference.point,
                        )

                    val robustScale =
                        0.004 +
                            residual /
                                10.0

                    val robustWeight =
                        (
                            1.0 /
                                (
                                    1.0 +
                                        (
                                            residual /
                                                robustScale
                                        ) *
                                        (
                                            residual /
                                                robustScale
                                        )
                                )
                            ).coerceIn(
                                0.02,
                                1.0,
                            )

                    val totalWeight =
                        confidenceWeight *
                            gapWeight *
                            robustWeight

                    sum += residual * totalWeight
                    weight += totalWeight
                }
            }

            return if (weight > 1e-9) {
                sum / weight
            } else {
                Double.POSITIVE_INFINITY
            }
        }

        var bestLag = 0L
        var bestScore = Double.POSITIVE_INFINITY

        var coarse =
            -config.lagSearchNanos

        while (
            coarse <=
                config.lagSearchNanos
        ) {
            val score =
                scoreAt(coarse)

            if (
                score.isFinite() &&
                score <
                    bestScore
            ) {
                bestScore = score
                bestLag = coarse
            }

            coarse +=
                config.lagCoarseStepNanos
        }

        val radius =
            config.lagFineRadiusSteps

        for (step in -radius..radius) {
            val candidateLag =
                (
                    bestLag +
                        step *
                        config.lagStepNanos
                ).coerceIn(
                    -config.lagSearchNanos,
                    config.lagSearchNanos,
                )

            val score =
                scoreAt(candidateLag)

            if (
                score.isFinite() &&
                score <
                    bestScore
            ) {
                bestScore = score
                bestLag = candidateLag
            }
        }

        val step =
            config.lagStepNanos

        if (
            bestLag -
                step >=
                -config.lagSearchNanos &&
            bestLag +
                step <=
                config.lagSearchNanos
        ) {
            val left =
                scoreAt(
                    bestLag - step,
                )

            val center =
                scoreAt(
                    bestLag,
                )

            val right =
                scoreAt(
                    bestLag + step,
                )

            val denominator =
                left -
                    2.0 * center +
                    right

            if (
                denominator.isFinite() &&
                abs(denominator) >
                    1e-12
            ) {
                val fraction =
                    (
                        0.5 *
                            (left - right) /
                            denominator
                        ).coerceIn(
                            -0.5,
                            0.5,
                        )

                val refinedLag =
                    (
                        bestLag +
                            (
                                fraction *
                                    step
                            ).toLong()
                        ).coerceIn(
                            -config.lagSearchNanos,
                            config.lagSearchNanos,
                        )

                val refinedScore =
                    scoreAt(
                        refinedLag,
                    )

                if (
                    refinedScore.isFinite() &&
                    refinedScore <=
                        bestScore
                ) {
                    bestLag = refinedLag
                }
            }
        }

        return abs(
            bestLag,
        ).toDouble() /
            1_000_000.0
    }

    private fun pairs(
        a: List<Timed>,
        b: List<Timed>,
    ): List<Pair<Timed, Timed>> {
        if (a.isEmpty() || b.isEmpty()) return emptyList()
        val out = ArrayList<Pair<Timed, Timed>>(min(a.size, b.size))
        var cursor = 0
        for (item in a) {
            while (cursor + 1 < b.size && b[cursor + 1].timestampNanos <= item.timestampNanos) {
                cursor++
            }
            val left = b.getOrNull(cursor)
            val right = b.getOrNull(cursor + 1)
            val nearest = listOfNotNull(left, right).minByOrNull {
                abs(it.timestampNanos - item.timestampNanos)
            } ?: continue
            out += item to nearest
        }
        return out
    }

    private fun scaleRatio(samples: List<JointSample>): Double? {
        val ratios = samples.mapNotNull {
            val a = it.estimatedScale ?: return@mapNotNull null
            val b = it.referenceScale ?: return@mapNotNull null
            (a / b).takeIf(Double::isFinite)
        }
        return ratios.takeIf { it.isNotEmpty() }?.let(::median)
    }

    private fun scaleDrift(
        samples: List<JointSample>,
    ): Double? {
        val values =
            samples
                .mapNotNull {
                    val estimated =
                        it.estimatedScale
                            ?: return@mapNotNull null
                    val reference =
                        it.referenceScale
                            ?: return@mapNotNull null

                    val ratio =
                        (
                            estimated /
                                reference
                            ).takeIf(Double::isFinite)
                            ?: return@mapNotNull null

                    it.timestampNanos /
                        1_000_000_000.0 to
                        ratio
                }

        if (values.size < 4) return null

        val slopes =
            ArrayList<Double>()

        for (i in values.indices) {
            for (j in i + 1 until values.size) {
                val dt =
                    values[j].first -
                        values[i].first

                if (dt <= 0.0) continue

                val slope =
                    (
                        values[j].second -
                            values[i].second
                        ) /
                        dt

                if (slope.isFinite()) {
                    slopes += slope
                }
            }
        }

        if (slopes.isEmpty()) return null

        return abs(
            median(slopes),
        ) *
            60.0 *
            100.0
    }

    private fun lowerBetter(value: Double, fullCredit: Double): Double =
        if (!value.isFinite() || fullCredit <= 0.0) 0.0
        else (1.0 - value / fullCredit).coerceIn(0.0, 1.0)

    private fun mean(values: List<Double>): Double =
        values.filter(Double::isFinite).takeIf { it.isNotEmpty() }?.average()
            ?: Double.POSITIVE_INFINITY

    private fun rms(values: List<Double>): Double {
        val finite = values.filter(Double::isFinite)
        if (finite.isEmpty()) return Double.POSITIVE_INFINITY
        return sqrt(finite.sumOf { it * it } / finite.size.toDouble())
    }

    private fun percentile(values: List<Double>, probability: Double): Double {
        val finite = values.filter(Double::isFinite).sorted()
        if (finite.isEmpty()) return Double.POSITIVE_INFINITY
        if (finite.size == 1) return finite[0]
        val p = probability.coerceIn(0.0, 1.0) * (finite.size - 1)
        val lower = p.toInt()
        val upper = min(lower + 1, finite.lastIndex)
        return finite[lower] + (finite[upper] - finite[lower]) * (p - lower)
    }

    private fun median(values: List<Double>): Double = percentile(values, 0.50)

    private fun distance(a: Vec3, b: Vec3): Double =
        sqrt(
            (a.x - b.x) * (a.x - b.x) +
                (a.y - b.y) * (a.y - b.y) +
                (a.z - b.z) * (a.z - b.z),
        )

    private fun normalize(v: Vec3?): Vec3? {
        v ?: return null
        val length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if (!length.isFinite() || length <= 1e-9) return null
        return Vec3(v.x / length, v.y / length, v.z / length)
    }

    private fun angleDegrees(a: Vec3, b: Vec3): Double =
        Math.toDegrees(
            kotlin.math.acos(
                (
                    a.x * b.x + a.y * b.y + a.z * b.z
                    ).coerceIn(-1.0, 1.0),
            ),
        )

    private companion object {
        fun requireFinite(v: Vec3) {
            require(v.x.isFinite() && v.y.isFinite() && v.z.isFinite())
        }
    }
}
