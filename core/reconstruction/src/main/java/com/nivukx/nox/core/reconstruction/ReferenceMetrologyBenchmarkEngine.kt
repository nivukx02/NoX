package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3

import kotlin.math.sqrt
import kotlin.math.abs

/**
 * Reference-based motion-capture metrology engine.
 *
 * This engine compares NoX estimates against an independent measured reference. It is intentionally
 * separate from the live precision gate: a live confidence score cannot certify physical accuracy.
 *
 * The engine supports:
 * - Euclidean position error and axis-wise trueness;
 * - RMSE, mean absolute error and empirical P95/P99 tails;
 * - dynamic versus quasi-static regimes;
 * - occluded versus visible observations;
 * - dropout and invalid-observation accounting;
 * - subject/joint/session stratification;
 * - repeatability across independent sessions;
 * - reference-system uncertainty propagation.
 *
 * Results are measurement summaries. Qualification policy remains a separate decision layer.
 */
class ReferenceMetrologyBenchmarkEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumValidSamples: Int = 240,
        val minimumIndependentSessions: Int = 3,
        val p95TargetMillimeters: Double = 1.0,
        val p99TargetMillimeters: Double = 1.5,
        val maximumDropoutFraction: Double = 0.02,
        val maximumOccludedP95Millimeters: Double = 2.5,
        val minimumRepeatabilitySamplesPerSession: Int = 30,
        val maximumStoredSamples: Int = 250_000,
        val maximumStoredSessions: Int = 128,
    ) {
        init {
            require(minimumValidSamples > 0)
            require(minimumIndependentSessions >= 2)
            require(p95TargetMillimeters > 0.0)
            require(p99TargetMillimeters >= p95TargetMillimeters)
            require(maximumDropoutFraction in 0.0..1.0)
            require(maximumOccludedP95Millimeters > 0.0)
            require(minimumRepeatabilitySamplesPerSession > 0)
            require(maximumStoredSamples >= minimumValidSamples)
            require(maximumStoredSessions >= minimumIndependentSessions)
        }
    }

    data class Sample(
        val sessionId: String,
        val timestampNanos: Long,
        val subjectId: String,
        val jointId: Int,
        val estimated: Vec3,
        val reference: Vec3,
        val referenceUncertaintyMillimeters: Double,
        val estimatedConfidence: Double,
        val occluded: Boolean = false,
        val dynamic: Boolean = false,
        val validEstimate: Boolean = true,
    ) {
        init {
            require(sessionId.isNotBlank())
            require(timestampNanos >= 0L)
            require(subjectId.isNotBlank())
            require(jointId >= 0)
            require(referenceUncertaintyMillimeters.isFinite())
            require(referenceUncertaintyMillimeters >= 0.0)
            require(estimatedConfidence.isFinite())
            require(estimatedConfidence in 0.0..1.0)
        }
    }

    data class AxisError(
        val xMillimeters: Double,
        val yMillimeters: Double,
        val zMillimeters: Double,
        val meanSignedXMillimeters: Double,
        val meanSignedYMillimeters: Double,
        val meanSignedZMillimeters: Double,
    )

    data class Distribution(
        val count: Int,
        val meanMillimeters: Double,
        val meanAbsoluteMillimeters: Double,
        val rmseMillimeters: Double,
        val p50Millimeters: Double,
        val p95Millimeters: Double,
        val p99Millimeters: Double,
        val maximumMillimeters: Double,
        val standardDeviationMillimeters: Double,
        val truenessMillimeters: Double,
    )

    data class SessionResult(
        val sessionId: String,
        val sampleCount: Int,
        val validCount: Int,
        val dropoutFraction: Double,
        val distribution: Distribution,
        val repeatabilityMillimeters: Double,
        val qualifiedAtP95: Boolean,
        val warnings: List<String>,
    )

    data class JointResult(
        val subjectId: String,
        val jointId: Int,
        val sampleCount: Int,
        val distribution: Distribution,
        val visibleDistribution: Distribution,
        val occludedDistribution: Distribution,
        val axisError: AxisError,
    )

    data class Result(
        val totalSamples: Int,
        val validSamples: Int,
        val invalidSamples: Int,
        val dropoutFraction: Double,
        val independentSessions: Int,
        val subjects: Int,
        val joints: Int,
        val global: Distribution,
        val visible: Distribution,
        val occluded: Distribution,
        val axisError: AxisError,
        val sessionResults: List<SessionResult>,
        val jointResults: List<JointResult>,
        val referenceUncertaintyP95Millimeters: Double,
        val combinedP95Millimeters: Double,
        val repeatabilityP95Millimeters: Double,
        val dynamicP95Millimeters: Double,
        val staticP95Millimeters: Double,
        val dynamicSampleCount: Int,
        val staticSampleCount: Int,
        val qualificationReady: Boolean,
        val qualificationReasons: List<String>,
        val warnings: List<String>,
    ) {
        companion object {
            fun empty(): Result =
                Result(
                    totalSamples = 0,
                    validSamples = 0,
                    invalidSamples = 0,
                    dropoutFraction = 1.0,
                    independentSessions = 0,
                    subjects = 0,
                    joints = 0,
                    global = emptyDistribution(),
                    visible = emptyDistribution(),
                    occluded = emptyDistribution(),
                    axisError = zeroAxis(),
                    sessionResults = emptyList(),
                    jointResults = emptyList(),
                    referenceUncertaintyP95Millimeters = 0.0,
                    combinedP95Millimeters = 0.0,
                    repeatabilityP95Millimeters = 0.0,
                    dynamicP95Millimeters = 0.0,
                    staticP95Millimeters = 0.0,
                    dynamicSampleCount = 0,
                    staticSampleCount = 0,
                    qualificationReady = false,
                    qualificationReasons = listOf("NO_REFERENCE_SAMPLES"),
                    warnings = listOf("NO_REFERENCE_SAMPLES"),
                )

            fun emptyDistribution(): Distribution =
                Distribution(
                    count = 0,
                    meanMillimeters = 0.0,
                    meanAbsoluteMillimeters = 0.0,
                    rmseMillimeters = 0.0,
                    p50Millimeters = 0.0,
                    p95Millimeters = 0.0,
                    p99Millimeters = 0.0,
                    maximumMillimeters = 0.0,
                    standardDeviationMillimeters = 0.0,
                    truenessMillimeters = 0.0,
                )

            fun zeroAxis(): AxisError =
                AxisError(
                    xMillimeters = 0.0,
                    yMillimeters = 0.0,
                    zMillimeters = 0.0,
                    meanSignedXMillimeters = 0.0,
                    meanSignedYMillimeters = 0.0,
                    meanSignedZMillimeters = 0.0,
                )

        }
    }

    private val samples = ArrayDeque<Sample>()
    private var lastResult = Result.empty()

    @Synchronized
    fun add(sample: Sample) {
        if (samples.size >= config.maximumStoredSamples) {
            samples.removeFirst()
        }
        samples.addLast(sample)
        lastResult = Result.empty()
    }

    @Synchronized
    fun addAll(values: Iterable<Sample>) {
        values.forEach { add(it) }
    }

    @Synchronized
    fun removeSession(sessionId: String): Int {
        if (sessionId.isBlank()) return 0
        val before = samples.size
        samples.removeIf { it.sessionId == sessionId }
        lastResult = Result.empty()
        return before - samples.size
    }

    @Synchronized
    fun evaluate(): Result {
        if (samples.isEmpty()) {
            lastResult = Result.empty()
            return lastResult
        }

        val total = samples.size
        val valid = samples.filter { it.validEstimate }
        val invalidCount = total - valid.size
        val dropoutFraction =
            invalidCount.toDouble() /
                total.toDouble().coerceAtLeast(1.0)

        val globalErrors = valid.map(::errorRecord)
        val visibleErrors = valid.filterNot { it.occluded }.map(::errorRecord)
        val occludedErrors = valid.filter { it.occluded }.map(::errorRecord)

        val dynamic = valid.filter { it.dynamic }.map(::errorRecord)
        val static = valid.filterNot { it.dynamic }.map(::errorRecord)

        val sessions = valid.map { it.sessionId }.toSet()
        val subjects = valid.map { it.subjectId }.toSet()
        val joints =
            valid
                .map { it.jointId }
                .toSet()

        val sessionResults =
            sessions
                .map { sessionId ->
                    evaluateSession(
                        sessionId = sessionId,
                        totalSamples = samples.count {
                            it.sessionId == sessionId
                        },
                        validSamples =
                            valid.count {
                                it.sessionId == sessionId
                            },
                        errors =
                            valid
                                .filter { it.sessionId == sessionId }
                                .map(::errorRecord),
                    )
                }
                .sortedBy { it.sessionId }

        val jointResults =
            valid
                .groupBy {
                    it.subjectId to it.jointId
                }
                .map { (key, values) ->
                    val allErrors = values.map(::errorRecord)
                    val visibleForJoint =
                        values
                            .filterNot { it.occluded }
                            .map(::errorRecord)
                    val occludedForJoint =
                        values
                            .filter { it.occluded }
                            .map(::errorRecord)

                    JointResult(
                        subjectId = key.first,
                        jointId = key.second,
                        sampleCount = values.size,
                        distribution = distribution(allErrors),
                        visibleDistribution =
                            distribution(visibleForJoint),
                        occludedDistribution =
                            distribution(occludedForJoint),
                        axisError = axisError(values),
                    )
                }
                .sortedWith(
                    compareBy<JointResult>(
                        { it.subjectId },
                        { it.jointId },
                    ),
                )

        val repeatabilities =
            sessionResults
                .map { it.repeatabilityMillimeters }
                .filter { it.isFinite() }

        val repeatabilityP95 =
            percentile(
                repeatabilities,
                0.95,
            )

        val referenceUncertaintyP95 =
            percentile(
                valid
                    .map {
                        it.referenceUncertaintyMillimeters
                    },
                0.95,
            )

        val global = distribution(globalErrors)
        val visibleDistribution = distribution(visibleErrors)
        val occludedDistribution = distribution(occludedErrors)

        val combinedP95 =
            rss(
                global.p95Millimeters,
                referenceUncertaintyP95,
            )

        val qualificationReasons =
            buildList {
                if (
                    valid.size <
                    config.minimumValidSamples
                ) {
                    add(
                        "INSUFFICIENT_VALID_REFERENCE_SAMPLES",
                    )
                }
                if (
                    sessions.size <
                    config.minimumIndependentSessions
                ) {
                    add(
                        "INSUFFICIENT_INDEPENDENT_SESSIONS",
                    )
                }
                if (
                    dropoutFraction >
                    config.maximumDropoutFraction
                ) {
                    add(
                        "DROPOUT_FRACTION_EXCEEDS_LIMIT",
                    )
                }
                if (
                    global.p95Millimeters >
                    config.p95TargetMillimeters
                ) {
                    add(
                        "GLOBAL_P95_EXCEEDS_TARGET",
                    )
                }
                if (
                    global.p99Millimeters >
                    config.p99TargetMillimeters
                ) {
                    add(
                        "GLOBAL_P99_EXCEEDS_TARGET",
                    )
                }
                if (
                    occludedDistribution.count > 0 &&
                    occludedDistribution.p95Millimeters >
                    config.maximumOccludedP95Millimeters
                ) {
                    add(
                        "OCCLUDED_P95_EXCEEDS_TARGET",
                    )
                }
                if (
                    referenceUncertaintyP95 >
                    config.p95TargetMillimeters * 0.25
                ) {
                    add(
                        "REFERENCE_UNCERTAINTY_TOO_LARGE",
                    )
                }
                if (
                    repeatabilityP95 >
                    config.p95TargetMillimeters
                ) {
                    add(
                        "INTER_SESSION_REPEATABILITY_EXCEEDS_TARGET",
                    )
                }
                if (
                    sessionResults.any {
                        it.validCount <
                            config.minimumRepeatabilitySamplesPerSession
                    }
                ) {
                    add(
                        "ONE_OR_MORE_SESSIONS_HAVE_INSUFFICIENT_REPEATABILITY_SAMPLES",
                    )
                }
            }.distinct()

        val warnings =
            buildList {
                if (invalidCount > 0) {
                    add("INVALID_OR_DROPPED_ESTIMATES_PRESENT")
                }
                if (
                    global.p99Millimeters >
                    global.p95Millimeters * 1.75
                ) {
                    add(
                        "LONG_TAIL_ERROR_DISTRIBUTION_REQUIRES_DYNAMIC_ANALYSIS",
                    )
                }
                if (
                    dynamic.isNotEmpty() &&
                    dynamicP95(dynamic) >
                    staticP95(static).coerceAtLeast(1e-6) * 2.0
                ) {
                    add("DYNAMIC_REGIME_ERROR_DOMINATES_STATIC_REGIME")
                }
                if (
                    occludedErrors.isNotEmpty() &&
                    occludedDistribution.p95Millimeters >
                    visibleDistribution.p95Millimeters * 1.5
                ) {
                    add(
                        "OCCLUSION_SIGNIFICANTLY_INCREASES_ERROR",
                    )
                }
                if (
                    global.truenessMillimeters >
                    config.p95TargetMillimeters
                ) {
                    add(
                        "SYSTEMATIC_TREND_EXCEEDS_PRECISION_TARGET",
                    )
                }
            }.distinct()

        val result =
            Result(
                totalSamples = total,
                validSamples = valid.size,
                invalidSamples = invalidCount,
                dropoutFraction = dropoutFraction,
                independentSessions = sessions.size,
                subjects = subjects.size,
                joints = joints.size,
                global = global,
                visible = visibleDistribution,
                occluded = occludedDistribution,
                axisError = axisError(valid),
                sessionResults = sessionResults,
                jointResults = jointResults,
                referenceUncertaintyP95Millimeters =
                    referenceUncertaintyP95,
                combinedP95Millimeters = combinedP95,
                repeatabilityP95Millimeters =
                    repeatabilityP95,
                dynamicP95Millimeters =
                    dynamicP95(dynamic),
                staticP95Millimeters =
                    staticP95(static),
                dynamicSampleCount = dynamic.size,
                staticSampleCount = static.size,
                qualificationReady =
                    qualificationReasons.isEmpty(),
                qualificationReasons = qualificationReasons,
                warnings = warnings,
            )

        lastResult = result
        return result
    }

    @Synchronized
    fun latest(): Result = lastResult

    @Synchronized
    fun sampleCount(): Int = samples.size

    @Synchronized
    fun clear() {
        samples.clear()
        lastResult = Result.empty()
    }

    private data class ErrorRecord(
        val magnitudeMillimeters: Double,
        val xMillimeters: Double,
        val yMillimeters: Double,
        val zMillimeters: Double,
        val referenceUncertaintyMillimeters: Double,
    )

    private fun errorRecord(sample: Sample): ErrorRecord {
        val dx =
            (sample.estimated.x - sample.reference.x) * 1000.0
        val dy =
            (sample.estimated.y - sample.reference.y) * 1000.0
        val dz =
            (sample.estimated.z - sample.reference.z) * 1000.0

        val magnitude =
            sqrt(
                dx * dx +
                    dy * dy +
                    dz * dz,
            )

        return ErrorRecord(
            magnitudeMillimeters =
                if (magnitude.isFinite()) magnitude else Double.POSITIVE_INFINITY,
            xMillimeters =
                if (dx.isFinite()) dx else 0.0,
            yMillimeters =
                if (dy.isFinite()) dy else 0.0,
            zMillimeters =
                if (dz.isFinite()) dz else 0.0,
            referenceUncertaintyMillimeters =
                sample.referenceUncertaintyMillimeters,
        )
    }

    private fun evaluateSession(
        sessionId: String,
        totalSamples: Int,
        validSamples: Int,
        errors: List<ErrorRecord>,
    ): SessionResult {
        val dropout =
            (
                totalSamples - validSamples
            ).coerceAtLeast(0)
                .toDouble() /
                totalSamples.toDouble().coerceAtLeast(1.0)

        val distribution =
            distribution(errors)

        val repeatability =
            repeatability(errors)

        val warnings =
            buildList {
                if (dropout > config.maximumDropoutFraction) {
                    add("SESSION_DROPOUT_HIGH")
                }
                if (validSamples < config.minimumRepeatabilitySamplesPerSession) {
                    add("SESSION_REPEATABILITY_SAMPLE_COUNT_LOW")
                }
                if (
                    distribution.p95Millimeters >
                    config.p95TargetMillimeters
                ) {
                    add("SESSION_P95_ABOVE_TARGET")
                }
            }.distinct()

        return SessionResult(
            sessionId = sessionId,
            sampleCount = totalSamples,
            validCount = validSamples,
            dropoutFraction = dropout,
            distribution = distribution,
            repeatabilityMillimeters = repeatability,
            qualifiedAtP95 =
                distribution.p95Millimeters <=
                    config.p95TargetMillimeters,
            warnings = warnings,
        )
    }

    private fun repeatability(
        errors: List<ErrorRecord>,
    ): Double {
        if (errors.size < 2) return 0.0
        val mean =
            errors
                .map { it.magnitudeMillimeters }
                .average()
        val variance =
            errors
                .map {
                    val d =
                        it.magnitudeMillimeters - mean
                    d * d
                }
                .average()
        return sqrt(
            maxOf(
                0.0,
                variance,
            ),
        )
    }

    private fun distribution(
        errors: List<ErrorRecord>,
    ): Distribution {
        if (errors.isEmpty()) {
            return emptyDistribution()
        }

        val finite =
            errors
                .map { it.magnitudeMillimeters }
                .filter { it.isFinite() }
                .sorted()

        if (finite.isEmpty()) {
            return emptyDistribution()
        }

        val mean =
            finite.average()

        val meanAbsolute =
            finite
                .map { abs(it) }
                .average()

        val rmse =
            sqrt(
                finite
                    .map { it * it }
                    .average(),
            )

        val variance =
            finite
                .map {
                    val d =
                        it - mean
                    d * d
                }
                .average()

        val trueness =
            abs(mean)

        return Distribution(
            count = finite.size,
            meanMillimeters = mean,
            meanAbsoluteMillimeters = meanAbsolute,
            rmseMillimeters = rmse,
            p50Millimeters = percentile(finite, 0.50),
            p95Millimeters = percentile(finite, 0.95),
            p99Millimeters = percentile(finite, 0.99),
            maximumMillimeters = finite.last(),
            standardDeviationMillimeters =
                sqrt(
                    maxOf(
                        0.0,
                        variance,
                    ),
                ),
            truenessMillimeters = trueness,
        )
    }

    private fun axisError(
        values: List<Sample>,
    ): AxisError {
        if (values.isEmpty()) {
            return zeroAxis()
        }

        val records = values.map(::errorRecord)
        val x = records.map { it.xMillimeters }.average()
        val y = records.map { it.yMillimeters }.average()
        val z = records.map { it.zMillimeters }.average()

        return AxisError(
            xMillimeters =
                sqrt(
                    records
                        .map {
                            it.xMillimeters * it.xMillimeters
                        }
                        .average(),
                ),
            yMillimeters =
                sqrt(
                    records
                        .map {
                            it.yMillimeters * it.yMillimeters
                        }
                        .average(),
                ),
            zMillimeters =
                sqrt(
                    records
                        .map {
                            it.zMillimeters * it.zMillimeters
                        }
                        .average(),
                ),
            meanSignedXMillimeters = x,
            meanSignedYMillimeters = y,
            meanSignedZMillimeters = z,
        )
    }

    private fun dynamicP95(
        values: List<ErrorRecord>,
    ): Double =
        percentile(
            values.map { it.magnitudeMillimeters },
            0.95,
        )

    private fun staticP95(
        values: List<ErrorRecord>,
    ): Double =
        percentile(
            values.map { it.magnitudeMillimeters },
            0.95,
        )

    private fun percentile(
        values: List<Double>,
        q: Double,
    ): Double {
        val finite =
            values
                .filter { it.isFinite() }
                .sorted()

        if (finite.isEmpty()) return 0.0
        if (finite.size == 1) return finite.first()

        val position =
            q.coerceIn(0.0, 1.0) *
                (finite.size - 1).toDouble()

        val lower = position.toInt()
        val upper =
            minOf(
                finite.lastIndex,
                lower + 1,
            )
        val alpha = position - lower

        return finite[lower] * (1.0 - alpha) +
            finite[upper] * alpha
    }

    private fun rss(
        a: Double,
        b: Double,
    ): Double =
        sqrt(
            a * a +
                b * b,
        )

    private fun emptyDistribution(): Distribution =
        Distribution(
            count = 0,
            meanMillimeters = 0.0,
            meanAbsoluteMillimeters = 0.0,
            rmseMillimeters = 0.0,
            p50Millimeters = 0.0,
            p95Millimeters = 0.0,
            p99Millimeters = 0.0,
            maximumMillimeters = 0.0,
            standardDeviationMillimeters = 0.0,
            truenessMillimeters = 0.0,
        )

    private fun zeroAxis(): AxisError =
        AxisError(
            xMillimeters = 0.0,
            yMillimeters = 0.0,
            zMillimeters = 0.0,
            meanSignedXMillimeters = 0.0,
            meanSignedYMillimeters = 0.0,
            meanSignedZMillimeters = 0.0,
        )
}
