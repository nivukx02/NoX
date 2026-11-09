package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.max

/**
 * Truth-preserving qualification protocol for practical sub-millimetre claims.
 *
 * A capture run is not called sub-millimetre merely because one aggregate percentile is small.
 * The protocol stratifies samples by measured motion regime and requires independent reference
 * quality plus minimum evidence in every enabled regime.
 *
 * This is a qualification gate, not a guarantee of physical accuracy.
 */
class SubMillimeterQualificationProtocol(
    private val config: Config = Config.strictProduction(),
) {

    enum class MotionRegime {
        STATIC,
        SLOW,
        DYNAMIC,
    }

    data class Config(
        val maximumReferenceUncertaintyMm: Double = 0.25,
        val staticSpeedMetersPerSecond: Double = 0.02,
        val dynamicSpeedMetersPerSecond: Double = 1.00,
        val maximumStaticP95ErrorMm: Double = 0.50,
        val maximumSlowP95ErrorMm: Double = 0.75,
        val maximumDynamicP95ErrorMm: Double = 1.00,
        val maximumStaticP99ErrorMm: Double = 0.80,
        val maximumSlowP99ErrorMm: Double = 1.20,
        val maximumDynamicP99ErrorMm: Double = 1.50,
        val maximumJitterMm: Double = 0.50,
        val maximumDropoutRatio: Double = 0.01,
        val minimumSamplesPerRegime: Int = 120,
        val minimumIndependentReferenceSessions: Int = 1,
        val maximumLagMs: Double = 5.0,
    ) {
        init {
            require(maximumReferenceUncertaintyMm > 0.0)
            require(staticSpeedMetersPerSecond > 0.0)
            require(dynamicSpeedMetersPerSecond > staticSpeedMetersPerSecond)
            require(maximumStaticP95ErrorMm > 0.0)
            require(maximumSlowP95ErrorMm >= maximumStaticP95ErrorMm)
            require(maximumDynamicP95ErrorMm >= maximumSlowP95ErrorMm)
            require(maximumStaticP99ErrorMm >= maximumStaticP95ErrorMm)
            require(maximumSlowP99ErrorMm >= maximumSlowP95ErrorMm)
            require(maximumDynamicP99ErrorMm >= maximumDynamicP95ErrorMm)
            require(maximumJitterMm > 0.0)
            require(maximumDropoutRatio in 0.0..1.0)
            require(minimumSamplesPerRegime > 0)
            require(minimumIndependentReferenceSessions >= 1)
            require(maximumLagMs > 0.0)
        }

        companion object {
            fun strictProduction(): Config = Config()

            fun labQualification(): Config =
                Config(
                    maximumReferenceUncertaintyMm = 0.15,
                    maximumStaticP95ErrorMm = 0.40,
                    maximumSlowP95ErrorMm = 0.60,
                    maximumDynamicP95ErrorMm = 0.90,
                    maximumStaticP99ErrorMm = 0.65,
                    maximumSlowP99ErrorMm = 1.00,
                    maximumDynamicP99ErrorMm = 1.35,
                    maximumJitterMm = 0.35,
                    maximumDropoutRatio = 0.005,
                    minimumSamplesPerRegime = 300,
                    minimumIndependentReferenceSessions = 3,
                    maximumLagMs = 3.0,
                )
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val estimate: Vec3,
        val reference: Vec3,
        val referenceUncertaintyMm: Double,
        val confidence: Double,
        val referenceSessionId: String = "default",
        val visible: Boolean = true,
        val sourceVelocityMetersPerSecond: Double? = null,
    ) {
        init {
            require(timestampNanos >= 0L)
            requireFinite(estimate)
            requireFinite(reference)
            require(referenceUncertaintyMm.isFinite() && referenceUncertaintyMm >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(referenceSessionId.isNotBlank())
            sourceVelocityMetersPerSecond?.let {
                require(it.isFinite() && it >= 0.0)
            }
        }
    }

    data class RegimeMetrics(
        val regime: MotionRegime,
        val sampleCount: Int,
        val validCount: Int,
        val dropoutRatio: Double,
        val p95ErrorMm: Double?,
        val p99ErrorMm: Double?,
        val jitterMm: Double?,
        val accepted: Boolean,
        val reasons: List<String>,
    )

    data class QualificationReport(
        val referenceUncertaintyMm: Double?,
        val referenceSessionCount: Int,
        val regimeMetrics: List<RegimeMetrics>,
        val overallDropoutRatio: Double,
        val temporalLagMs: Double?,
        val accepted: Boolean,
        val reasons: List<String>,
    )

    fun classifySpeed(speedMetersPerSecond: Double): MotionRegime {
        require(speedMetersPerSecond.isFinite() && speedMetersPerSecond >= 0.0)
        return when {
            speedMetersPerSecond < config.staticSpeedMetersPerSecond ->
                MotionRegime.STATIC
            speedMetersPerSecond < config.dynamicSpeedMetersPerSecond ->
                MotionRegime.SLOW
            else ->
                MotionRegime.DYNAMIC
        }
    }

    fun qualify(
        samples: List<Sample>,
        temporalLagMs: Double?,
    ): QualificationReport {
        require(samples.isNotEmpty())
        temporalLagMs?.let {
            require(it.isFinite() && it >= 0.0)
        }

        val referenceUncertainty =
            samples.maxOfOrNull { it.referenceUncertaintyMm }
        val referenceSessionCount =
            samples.map { it.referenceSessionId }.distinct().size

        val byRegime =
            MotionRegime.entries.associateWith {
                ArrayList<Sample>()
            }

        var previous: Sample? = null
        var totalSeen = 0
        var totalValid = 0

        samples
            .sortedBy { it.timestampNanos }
            .forEach { sample ->
                val speed =
                    sample.sourceVelocityMetersPerSecond
                        ?: previous?.let {
                            val dtSeconds =
                                (
                                    sample.timestampNanos -
                                        it.timestampNanos
                                ).toDouble() /
                                    1_000_000_000.0

                            if (dtSeconds <= 0.0) {
                                0.0
                            } else {
                                sample.reference.distanceTo(it.reference) /
                                    dtSeconds
                            }
                        }
                        ?: 0.0

                byRegime.getValue(classifySpeed(speed)).add(sample)
                totalSeen++
                if (sample.visible && sample.confidence > 0.0) {
                    totalValid++
                }
                previous = sample
            }

        val regimeReports =
            MotionRegime.entries.map { regime ->
                evaluateRegime(
                    regime = regime,
                    samples = byRegime.getValue(regime),
                )
            }

        val overallDropout =
            1.0 -
                totalValid.toDouble() /
                max(1, totalSeen).toDouble()

        val globalReasons =
            buildList {
                if (
                    referenceUncertainty == null ||
                        referenceUncertainty >
                        config.maximumReferenceUncertaintyMm
                ) {
                    add("REFERENCE_UNCERTAINTY_ABOVE_SUB_MM_CONTRACT")
                }

                if (referenceSessionCount < config.minimumIndependentReferenceSessions) {
                    add("INSUFFICIENT_INDEPENDENT_REFERENCE_SESSIONS")
                }

                if (overallDropout > config.maximumDropoutRatio) {
                    add("OVERALL_DROPOUT_ABOVE_SUB_MM_CONTRACT")
                }

                temporalLagMs?.let {
                    if (it > config.maximumLagMs) {
                        add("TEMPORAL_LAG_ABOVE_SUB_MM_CONTRACT")
                    }
                } ?: add("TEMPORAL_LAG_UNKNOWN")

                regimeReports
                    .filter { it.sampleCount > 0 && !it.accepted }
                    .forEach { regime ->
                        addAll(
                            regime.reasons.map { reason ->
                                regime.regime.name + ":" + reason
                            },
                        )
                    }
            }.distinct()

        val requiredRegimes =
            regimeReports.filter { it.sampleCount > 0 }

        val accepted =
            requiredRegimes.isNotEmpty() &&
                requiredRegimes.all { it.accepted } &&
                globalReasons.isEmpty()

        return QualificationReport(
            referenceUncertaintyMm = referenceUncertainty,
            referenceSessionCount = referenceSessionCount,
            regimeMetrics = regimeReports,
            overallDropoutRatio = overallDropout,
            temporalLagMs = temporalLagMs,
            accepted = accepted,
            reasons = globalReasons,
        )
    }

    private fun evaluateRegime(
        regime: MotionRegime,
        samples: List<Sample>,
    ): RegimeMetrics {
        val valid =
            samples.filter {
                it.visible &&
                    it.confidence > 0.0
            }

        val errors =
            valid
                .map {
                    it.estimate.distanceTo(it.reference) *
                        1_000.0
                }
                .filter { it.isFinite() && it >= 0.0 }

        val dropout =
            1.0 -
                valid.size.toDouble() /
                max(1, samples.size).toDouble()

        val sortedErrors = errors.sorted()
        val p95 = percentile(sortedErrors, 0.95)
        val p99 = percentile(sortedErrors, 0.99)
        val jitter =
            temporalJitterMm(
                valid.sortedBy { it.timestampNanos },
            )

        val reasons =
            buildList {
                if (samples.size < config.minimumSamplesPerRegime) {
                    add("INSUFFICIENT_REGIME_SAMPLES")
                }

                if (dropout > config.maximumDropoutRatio) {
                    add("REGIME_DROPOUT_HIGH")
                }

                if (
                    p95 != null &&
                    p95 > p95Limit(regime)
                ) {
                    add("REGIME_P95_ABOVE_CONTRACT")
                }

                if (
                    p99 != null &&
                    p99 > p99Limit(regime)
                ) {
                    add("REGIME_P99_ABOVE_CONTRACT")
                }

                if (
                    jitter != null &&
                    jitter > config.maximumJitterMm
                ) {
                    add("REGIME_JITTER_ABOVE_CONTRACT")
                }
            }.distinct()

        return RegimeMetrics(
            regime = regime,
            sampleCount = samples.size,
            validCount = valid.size,
            dropoutRatio = dropout,
            p95ErrorMm = p95,
            p99ErrorMm = p99,
            jitterMm = jitter,
            accepted =
                samples.size >=
                    config.minimumSamplesPerRegime &&
                    reasons.isEmpty(),
            reasons = reasons,
        )
    }

    private fun p95Limit(regime: MotionRegime): Double =
        when (regime) {
            MotionRegime.STATIC ->
                config.maximumStaticP95ErrorMm
            MotionRegime.SLOW ->
                config.maximumSlowP95ErrorMm
            MotionRegime.DYNAMIC ->
                config.maximumDynamicP95ErrorMm
        }

    private fun p99Limit(regime: MotionRegime): Double =
        when (regime) {
            MotionRegime.STATIC ->
                config.maximumStaticP99ErrorMm
            MotionRegime.SLOW ->
                config.maximumSlowP99ErrorMm
            MotionRegime.DYNAMIC ->
                config.maximumDynamicP99ErrorMm
        }

    private fun percentile(
        sorted: List<Double>,
        probability: Double,
    ): Double? {
        if (sorted.isEmpty()) return null
        val index =
            (
                (sorted.size - 1) *
                    probability
            ).toInt()

        return sorted[
            index.coerceIn(
                0,
                sorted.lastIndex,
            )
        ]
    }

    private fun temporalJitterMm(
        samples: List<Sample>,
    ): Double? {
        if (samples.size < 2) return null

        var sumSquared = 0.0
        var count = 0

        for (index in 1..samples.lastIndex) {
            val previous = samples[index - 1]
            val current = samples[index]
            val delta =
                (
                    current.estimate -
                        previous.estimate
                ).distanceTo(
                    Vec3(0.0, 0.0, 0.0),
                ) -
                    (
                        current.reference -
                            previous.reference
                    ).distanceTo(
                        Vec3(0.0, 0.0, 0.0),
                    )

            if (delta.isFinite()) {
                sumSquared +=
                    delta * delta *
                    1_000_000.0
                count++
            }
        }

        return if (count == 0) {
            null
        } else {
            kotlin.math.sqrt(
                sumSquared /
                    count.toDouble(),
            )
        }
    }

    private fun requireFinite(value: Vec3) {
        require(
            value.x.isFinite() &&
                value.y.isFinite() &&
                value.z.isFinite(),
        )
    }
}
