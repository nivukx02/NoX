package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Rolling runtime evidence ledger for production motion capture.
 *
 * Capture failures are often gradual. A camera can drift, temporal alignment can worsen, facial
 * evidence can decay, or surface confidence can erode for several seconds before any single frame
 * crosses a hard gate. This engine keeps a bounded timeline of quality evidence and produces a
 * trend-aware snapshot that can be displayed to an operator or attached to a session record.
 *
 * It never invents evidence. Every metric is supplied by an upstream engine and is clamped to a
 * known numerical range. The trend state is derived from measured history only.
 */
class ProductionCaptureEvidenceLedger(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class Trend {
        IMPROVING,
        STABLE,
        DEGRADING,
        CRITICAL,
        UNKNOWN,
    }

    enum class Severity {
        NONE,
        INFO,
        WARNING,
        HOLD,
        CRITICAL,
    }

    data class Config(
        val capacity: Int = 180,
        val minimumSamplesForTrend: Int = 8,
        val recentWindowSamples: Int = 24,
        val baselineWindowSamples: Int = 72,
        val warningSlopePerSample: Double = -0.004,
        val criticalSlopePerSample: Double = -0.012,
        val warningScore: Double = 0.64,
        val holdScore: Double = 0.48,
        val criticalScore: Double = 0.30,
        val confidenceWeight: Double = 0.62,
        val stabilityWeight: Double = 0.20,
        val continuityWeight: Double = 0.18,
        val maxGapSeconds: Double = 0.35,
    ) {
        init {
            require(capacity >= 16)
            require(minimumSamplesForTrend in 3..capacity)
            require(recentWindowSamples in 3..capacity)
            require(baselineWindowSamples in 3..capacity)
            require(warningSlopePerSample < 0.0)
            require(criticalSlopePerSample < warningSlopePerSample)
            require(warningScore in 0.0..1.0)
            require(holdScore in 0.0..warningScore)
            require(criticalScore in 0.0..holdScore)
            require(confidenceWeight >= 0.0)
            require(stabilityWeight >= 0.0)
            require(continuityWeight >= 0.0)
            require(
                confidenceWeight +
                    stabilityWeight +
                    continuityWeight > 0.0,
            )
            require(maxGapSeconds > 0.0)
        }
    }

    data class Sample(
        val timestampNanos: Long,
        val integrity: Double,
        val reconstruction: Double,
        val geometry: Double,
        val temporal: Double,
        val synchronization: Double,
        val calibration: Double,
        val bodyTracking: Double,
        val faceEye: Double,
        val faceEyeAccepted: Boolean = true,
        val faceEyeReliabilityTrend: Double = 0.0,
        val faceEyeCalibrationDriftScore: Double = 0.0,
        val faceEyeCalibrationDriftPersistent: Boolean = false,
        val hand: Double,
        val kinematic: Double,
        val surface: Double,
        val physics: Double = 1.0,
        val global4D: Double = 1.0,
        val global4DAccepted: Boolean = true,
        val global4DPersistentFailure: Boolean = false,
        val global4DDirectionalUncertaintyConsistency: Double = 1.0,
        val global4DFaceEyeConsistency: Double = 1.0,
        val metricAuthority: Double = 1.0,
        val epipolarAuthority: Double = 1.0,
        val motionInnovationScore: Double = 1.0,
        val outputAllowed: Boolean,
        val hardFailure: Boolean,
    ) {
        init {
            require(timestampNanos >= 0L)
            listOf(
                integrity,
                reconstruction,
                geometry,
                temporal,
                synchronization,
                calibration,
                bodyTracking,
                faceEye,
                hand,
                kinematic,
                surface,
                physics,
                global4D,
                global4DDirectionalUncertaintyConsistency,
                global4DFaceEyeConsistency,
                metricAuthority,
                epipolarAuthority,
                motionInnovationScore,
            ).forEach {
                require(it in 0.0..1.0)
            }
            require(faceEyeReliabilityTrend in -1.0..1.0)
        }
    }

    data class ComponentTrend(
        val name: String,
        val current: Double,
        val recentMean: Double,
        val baselineMean: Double,
        val delta: Double,
        val slopePerSample: Double,
        val direction: Trend,
        val severity: Severity,
    )

    data class Snapshot(
        val timestampNanos: Long,
        val sampleCount: Int,
        val currentScore: Double,
        val recentMeanScore: Double,
        val baselineMeanScore: Double,
        val trend: Trend,
        val severity: Severity,
        val scoreSlopePerSample: Double,
        val stabilityScore: Double,
        val continuityScore: Double,
        val degradationRatio: Double,
        val heldFrameRatio: Double,
        val hardFailureRatio: Double,
        val metricAuthorityScore: Double = 0.0,
        val epipolarAuthorityScore: Double = 0.0,
        val motionInnovationScore: Double = 0.0,
        val dominantComponent: ComponentTrend?,
        val components: List<ComponentTrend>,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val trusted: Boolean
            get() =
                severity == Severity.NONE ||
                    severity == Severity.INFO

        val shouldHold: Boolean
            get() =
                severity == Severity.HOLD ||
                    severity == Severity.CRITICAL
    }

    private val samples = ArrayDeque<Sample>()
    private var latest: Snapshot? = null

    @Synchronized
    fun append(sample: Sample): Snapshot {
        samples.addLast(sanitize(sample))
        while (samples.size > config.capacity) {
            samples.removeFirst()
        }

        val snapshot = computeSnapshot()
        latest = snapshot
        return snapshot
    }

    @Synchronized
    fun latest(): Snapshot? = latest

    @Synchronized
    fun size(): Int = samples.size

    @Synchronized
    fun recentSamples(limit: Int = config.recentWindowSamples): List<Sample> =
        samples.takeLast(limit.coerceIn(1, samples.size.coerceAtLeast(1)))

    @Synchronized
    fun reset() {
        samples.clear()
        latest = null
    }

    override fun close() = reset()

    private fun computeSnapshot(): Snapshot {
        val list = samples.toList()
        val latestSample = list.lastOrNull()

        if (latestSample == null) {
            return emptySnapshot()
        }

        if (list.size < config.minimumSamplesForTrend) {
            val current = compositeScore(latestSample)
            return Snapshot(
                timestampNanos = latestSample.timestampNanos,
                sampleCount = list.size,
                currentScore = current,
                recentMeanScore = current,
                baselineMeanScore = current,
                trend = Trend.UNKNOWN,
                severity = severityFor(
                    current = current,
                    trend = Trend.UNKNOWN,
                    hardFailure = latestSample.hardFailure,
                ),
                scoreSlopePerSample = 0.0,
                stabilityScore = 1.0,
                continuityScore = 1.0,
                degradationRatio = if (latestSample.outputAllowed) 0.0 else 1.0,
                heldFrameRatio = if (latestSample.outputAllowed) 0.0 else 1.0,
                hardFailureRatio = if (latestSample.hardFailure) 1.0 else 0.0,
                dominantComponent = null,
                components = componentSnapshots(list),
                warnings = listOf(
                    "evidence-ledger-acquiring-baseline",
                ),
                recommendations = listOf(
                    "Continue capture briefly so the rolling quality baseline can stabilize.",
                ),
            )
        }

        val recent = list.takeLast(
            min(config.recentWindowSamples, list.size),
        )
        val baseline = list.takeLast(
            min(config.baselineWindowSamples, list.size),
        )

        val recentScores = recent.map(::compositeScore)
        val baselineScores = baseline.map(::compositeScore)
        val currentScore = compositeScore(latestSample)
        val recentMean = recentScores.averageOrZero()
        val baselineMean = baselineScores.averageOrZero()

        val slope = linearSlope(recentScores)
        val stability = stabilityScore(recentScores)
        val continuity = continuityScore(recent)
        val degradation =
            (
                1.0 -
                    currentScore /
                    baselineMean.coerceAtLeast(0.08)
            ).coerceIn(0.0, 1.0)

        val heldRatio = recent.count { !it.outputAllowed }
            .toDouble() / recent.size.toDouble()
        val hardRatio = recent.count { it.hardFailure }
            .toDouble() / recent.size.toDouble()

        val metricAuthorityScore =
            latestSample.metricAuthority.coerceIn(
                0.0,
                1.0,
            )
        val epipolarAuthorityScore =
            latestSample.epipolarAuthority.coerceIn(
                0.0,
                1.0,
            )
        val motionInnovationScore =
            latestSample.motionInnovationScore.coerceIn(
                0.0,
                1.0,
            )

        val trend = classifyTrend(
            slope = slope,
            current = currentScore,
            recentMean = recentMean,
            baselineMean = baselineMean,
        )

        val components = componentSnapshots(list)
        val dominant = components
            .filter { it.severity != Severity.NONE }
            .minByOrNull { it.recentMean }

        val severity = severityFor(
            current = currentScore,
            trend = trend,
            hardFailure = latestSample.hardFailure,
        ).let { base ->
            when {
                hardRatio >= 0.40 -> maxSeverity(base, Severity.CRITICAL)
                heldRatio >= 0.55 -> maxSeverity(base, Severity.HOLD)
                slope <= config.criticalSlopePerSample ->
                    maxSeverity(base, Severity.WARNING)
                else -> base
            }
        }

        val warnings = buildWarnings(
            current = currentScore,
            recentMean = recentMean,
            baselineMean = baselineMean,
            trend = trend,
            slope = slope,
            stability = stability,
            continuity = continuity,
            degradation = degradation,
            heldRatio = heldRatio,
            hardRatio = hardRatio,
            dominant = dominant,
        )

        val recommendations = buildRecommendations(
            trend = trend,
            severity = severity,
            continuity = continuity,
            stability = stability,
            heldRatio = heldRatio,
            hardRatio = hardRatio,
            dominant = dominant,
        )

        return Snapshot(
            timestampNanos = latestSample.timestampNanos,
            sampleCount = list.size,
            currentScore = currentScore,
            recentMeanScore = recentMean,
            baselineMeanScore = baselineMean,
            trend = trend,
            severity = severity,
            scoreSlopePerSample = slope,
            stabilityScore = stability,
            continuityScore = continuity,
            degradationRatio = degradation,
            heldFrameRatio = heldRatio,
            hardFailureRatio = hardRatio,
            metricAuthorityScore =
                metricAuthorityScore,
            epipolarAuthorityScore =
                epipolarAuthorityScore,
            motionInnovationScore =
                motionInnovationScore,
            dominantComponent = dominant,
            components = components,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun componentSnapshots(
        allSamples: List<Sample>,
    ): List<ComponentTrend> {
        if (allSamples.isEmpty()) return emptyList()

        val recent = allSamples.takeLast(
            min(config.recentWindowSamples, allSamples.size),
        )
        val baseline = allSamples.takeLast(
            min(config.baselineWindowSamples, allSamples.size),
        )

        val extractors:
            List<Pair<String, (Sample) -> Double>> =
            listOf(
                "integrity" to { sample: Sample -> sample.integrity },
                "reconstruction" to { sample: Sample -> sample.reconstruction },
                "geometry" to { sample: Sample -> sample.geometry },
                "temporal" to { sample: Sample -> sample.temporal },
                "synchronization" to { sample: Sample -> sample.synchronization },
                "calibration" to { sample: Sample -> sample.calibration },
                "body-tracking" to { sample: Sample -> sample.bodyTracking },
                "face-eye" to { sample: Sample -> sample.faceEye },
                "face-eye-trend" to { sample: Sample ->
                    faceEyeReliabilityTrendScore(
                        sample.faceEyeReliabilityTrend,
                    )
                },
                "face-eye-calibration" to { sample: Sample ->
                    (
                        1.0 -
                            sample.faceEyeCalibrationDriftScore
                    ).coerceIn(0.0, 1.0) *
                        if (sample.faceEyeCalibrationDriftPersistent) {
                            0.42
                        } else {
                            1.0
                        }
                },
                "hand" to { sample: Sample -> sample.hand },
                "kinematic" to { sample: Sample -> sample.kinematic },
                "surface" to { sample: Sample -> sample.surface },
                "physics" to { sample: Sample -> sample.physics },
                "global-4d" to { sample: Sample -> sample.global4D },
                "global-4d-directional-uncertainty" to { sample: Sample ->
                    sample.global4DDirectionalUncertaintyConsistency
                },
                "global-4d-face-eye" to { sample: Sample ->
                    sample.global4DFaceEyeConsistency
                },
                "metric-authority" to { sample: Sample ->
                    sample.metricAuthority
                },
                "epipolar-authority" to { sample: Sample ->
                    sample.epipolarAuthority
                },
                "motion-innovation" to { sample: Sample ->
                    sample.motionInnovationScore
                },
            )

        return extractors.map { (name, extractor) ->
            val current = extractor(recent.last())
            val recentValues = recent.map(extractor)
            val baselineValues = baseline.map(extractor)
            val recentMean = recentValues.averageOrZero()
            val baselineMean = baselineValues.averageOrZero()
            val slope = linearSlope(recentValues)
            val delta = recentMean - baselineMean

            ComponentTrend(
                name = name,
                current = current,
                recentMean = recentMean,
                baselineMean = baselineMean,
                delta = delta,
                slopePerSample = slope,
                direction = classifyComponentTrend(
                    delta = delta,
                    slope = slope,
                ),
                severity = componentSeverity(
                    current = current,
                    slope = slope,
                ),
            )
        }
    }

    private fun faceEyeReliabilityTrendScore(
        trend: Double,
    ): Double {
        val bounded = trend.coerceIn(-1.0, 1.0)
        return when {
            bounded >= 0.0 ->
                (0.88 + bounded * 0.12).coerceIn(0.0, 1.0)
            bounded >= -0.08 ->
                (0.82 + (bounded + 0.08) / 0.08 * 0.06)
                    .coerceIn(0.0, 1.0)
            bounded >= -0.35 ->
                (
                    0.82 -
                        (-bounded - 0.08) / 0.27 * 0.42
                    ).coerceIn(0.34, 0.82)
            else ->
                (
                    0.40 -
                        (-bounded - 0.35) / 0.65 * 0.32
                    ).coerceIn(0.08, 0.40)
        }
    }

    private fun compositeScore(sample: Sample): Double =
        (
            sample.integrity * config.confidenceWeight * 0.56 +
                weightedMean(
                    listOf(
                        sample.reconstruction,
                        sample.geometry,
                        sample.temporal,
                        sample.synchronization,
                        sample.calibration,
                        sample.bodyTracking,
                        (
                            sample.faceEye *
                                if (sample.faceEyeAccepted) {
                                    1.0
                                } else {
                                    0.58
                                }
                        ).coerceIn(0.0, 1.0),
                        (
                            1.0 -
                                sample.faceEyeCalibrationDriftScore
                            ).coerceIn(0.0, 1.0) *
                            if (sample.faceEyeCalibrationDriftPersistent) {
                                0.42
                            } else {
                                1.0
                            },
                        sample.hand,
                        sample.kinematic,
                        sample.surface,
                        sample.physics,
                        sample.global4D,
                    ),
                ) * config.confidenceWeight * 0.44 +
                continuityProxy(sample) *
                    config.continuityWeight +
                sample.temporal *
                    config.stabilityWeight
        ).div(
            config.confidenceWeight +
                config.continuityWeight +
                config.stabilityWeight,
        ).coerceIn(0.0, 1.0)

    private fun weightedMean(values: List<Double>): Double =
        if (values.isEmpty()) 0.0 else values.average()

    private fun continuityProxy(sample: Sample): Double =
        when {
            sample.hardFailure -> 0.20
            !sample.outputAllowed -> 0.54
            else -> 0.90
        }

    private fun classifyTrend(
        slope: Double,
        current: Double,
        recentMean: Double,
        baselineMean: Double,
    ): Trend =
        when {
            current < config.criticalScore &&
                slope <= config.criticalSlopePerSample ->
                Trend.CRITICAL

            slope <= config.criticalSlopePerSample ->
                Trend.DEGRADING

            slope <= config.warningSlopePerSample ->
                Trend.DEGRADING

            recentMean >
                baselineMean + 0.025 &&
                slope > 0.0 ->
                Trend.IMPROVING

            abs(recentMean - baselineMean) <= 0.025 ->
                Trend.STABLE

            recentMean < baselineMean ->
                Trend.DEGRADING

            else ->
                Trend.STABLE
        }

    private fun classifyComponentTrend(
        delta: Double,
        slope: Double,
    ): Trend =
        when {
            slope <= config.warningSlopePerSample ||
                delta <= -0.02 ->
                Trend.DEGRADING

            slope >= abs(config.warningSlopePerSample) &&
                delta >= 0.03 ->
                Trend.IMPROVING

            abs(delta) <= 0.03 ->
                Trend.STABLE

            delta > 0.0 -> Trend.IMPROVING
            else -> Trend.DEGRADING
        }

    private fun componentSeverity(
        current: Double,
        slope: Double,
    ): Severity =
        when {
            current < config.criticalScore -> Severity.CRITICAL
            current < config.holdScore -> Severity.HOLD
            current < config.warningScore -> Severity.WARNING
            slope <= config.criticalSlopePerSample -> Severity.WARNING
            else -> Severity.NONE
        }

    private fun severityFor(
        current: Double,
        trend: Trend,
        hardFailure: Boolean,
    ): Severity =
        when {
            hardFailure -> Severity.CRITICAL
            current < config.criticalScore -> Severity.CRITICAL
            current < config.holdScore -> Severity.HOLD
            current < config.warningScore -> Severity.WARNING
            trend == Trend.CRITICAL -> Severity.WARNING
            trend == Trend.DEGRADING -> Severity.INFO
            else -> Severity.NONE
        }

    private fun maxSeverity(
        a: Severity,
        b: Severity,
    ): Severity =
        if (severityRank(a) >= severityRank(b)) a else b

    private fun severityRank(value: Severity): Int =
        when (value) {
            Severity.NONE -> 0
            Severity.INFO -> 1
            Severity.WARNING -> 2
            Severity.HOLD -> 3
            Severity.CRITICAL -> 4
        }

    private fun stabilityScore(values: List<Double>): Double {
        if (values.size < 2) return 1.0

        val mean = values.average()
        val variance =
            values.sumOf { (it - mean) * (it - mean) } /
                values.size.toDouble()

        val deviation = kotlin.math.sqrt(
            variance.coerceAtLeast(0.0),
        )

        return (
            1.0 -
                deviation / 0.25
        ).coerceIn(0.0, 1.0)
    }

    private fun continuityScore(
        values: List<Sample>,
    ): Double {
        if (values.size < 2) return 1.0

        var good = 0
        var total = 0

        values.zipWithNext().forEach { (a, b) ->
            total += 1
            val dt =
                (
                    b.timestampNanos -
                        a.timestampNanos
                    ).toDouble() / 1e9

            if (
                dt > 0.0 &&
                dt <= config.maxGapSeconds &&
                b.timestampNanos > a.timestampNanos
            ) {
                good += 1
            }
        }

        return if (total == 0) {
            1.0
        } else {
            good.toDouble() / total.toDouble()
        }
    }

    private fun linearSlope(values: List<Double>): Double {
        if (values.size < 2) return 0.0

        val n = values.size.toDouble()
        val meanX = (n - 1.0) * 0.5
        val meanY = values.average()

        var numerator = 0.0
        var denominator = 0.0

        values.forEachIndexed { index, value ->
            val dx = index.toDouble() - meanX
            numerator += dx * (value - meanY)
            denominator += dx * dx
        }

        if (denominator <= 1e-9) return 0.0
        return numerator / denominator
    }

    private fun buildWarnings(
        current: Double,
        recentMean: Double,
        baselineMean: Double,
        trend: Trend,
        slope: Double,
        stability: Double,
        continuity: Double,
        degradation: Double,
        heldRatio: Double,
        hardRatio: Double,
        dominant: ComponentTrend?,
    ): List<String> = buildList {
        if (trend == Trend.DEGRADING) {
            add("capture-quality-trend-degrading")
        }
        if (trend == Trend.CRITICAL) {
            add("capture-quality-trend-critical")
        }
        if (current < config.warningScore) {
            add("capture-quality-current-below-warning-floor")
        }
        if (recentMean < baselineMean - 0.06) {
            add("capture-quality-recent-window-below-baseline")
        }
        if (slope <= config.warningSlopePerSample) {
            add("capture-quality-negative-slope")
        }
        if (stability < 0.55) {
            add("capture-quality-unstable")
        }
        if (continuity < 0.92) {
            add("capture-timeline-continuity-degraded")
        }
        if (degradation > 0.20) {
            add("capture-quality-degradation-significant")
        }
        if (heldRatio > 0.25) {
            add("trusted-output-hold-ratio-elevated")
        }
        if (hardRatio > 0.08) {
            add("hard-failure-ratio-elevated")
        }
        if (samples.lastOrNull()?.faceEyeAccepted == false) {
            add("face-eye-fusion-rejected")
        }
        if (samples.lastOrNull()?.faceEyeCalibrationDriftPersistent == true) {
            add("face-eye-calibration-drift-persistent")
        }
        if (
            samples.lastOrNull()?.global4DAccepted == false ||
            samples.lastOrNull()?.global4DPersistentFailure == true
        ) {
            add("global-4d-consistency-rejected")
        }
        samples.lastOrNull()?.let { sample ->
            if (sample.metricAuthority < 0.55) {
                add("metric-authority-below-trust-floor")
            }
            if (sample.epipolarAuthority < 0.55) {
                add("epipolar-authority-below-trust-floor")
            }
            if (sample.motionInnovationScore < 0.50) {
                add("motion-model-innovation-degraded")
            }
        }
        dominant?.let {
            if (it.severity != Severity.NONE) {
                add("dominant-quality-bottleneck-" + it.name)
            }
        }
    }.distinct()

    private fun buildRecommendations(
        trend: Trend,
        severity: Severity,
        continuity: Double,
        stability: Double,
        heldRatio: Double,
        hardRatio: Double,
        dominant: ComponentTrend?,
    ): List<String> = buildList {
        if (trend == Trend.IMPROVING) {
            add("Maintain current rig placement and capture conditions.")
        }

        if (trend == Trend.STABLE &&
            severity == Severity.NONE
        ) {
            add("Quality trend is stable; continue the current capture configuration.")
        }

        if (continuity < 0.92) {
            add("Investigate timestamp gaps or dropped inference frames before the next take.")
        }

        if (stability < 0.55) {
            add("Inspect lighting, motion blur, camera movement, and synchronization stability.")
        }

        if (heldRatio > 0.25) {
            add("Resolve the dominant evidence bottleneck before committing more trusted motion.")
        }

        if (hardRatio > 0.08) {
            add("Review hard-failure frames and recalibrate or reposition affected cameras.")
        }

        val latest = samples.lastOrNull()
        if (latest != null) {
            if (latest.metricAuthority < 0.55) {
                add("Increase measured metric evidence before trusting fine 3D geometry.")
            }
            if (latest.epipolarAuthority < 0.55) {
                add("Increase independent camera agreement and reject contradictory views.")
            }
            if (latest.motionInnovationScore < 0.50) {
                add("Inspect motion-model residuals for blur, identity swaps, or timing error.")
            }
        }

        if (dominant != null) {
            when (dominant.name) {
                "geometry" ->
                    add("Increase multi-view coverage or improve camera baseline quality.")
                "synchronization" ->
                    add("Tighten multi-camera clock alignment and inspect network jitter.")
                "calibration" ->
                    add("Refresh calibration and verify camera poses remain fixed.")
                "face-eye" ->
                    add("Keep the face inside the high-confidence capture region.")
                "face-eye-calibration" ->
                    add("Recalibrate binocular face/eye geometry before trusting metric gaze or facial landmarks.")
                "kinematic" ->
                    add("Review skeletal residuals, foot contacts, and root-motion constraints.")
                "physics" ->
                    add("Review bone-length residuals, dynamics envelopes, symmetry, and support contacts before trusting metric output.")
                "surface" ->
                    add("Increase dense-surface observation before trusting fine geometry.")
                "temporal" ->
                    add("Reduce dropped frames and verify temporal continuity.")
            }
        }
    }.distinct().take(6)

    private fun sanitize(sample: Sample): Sample =
        sample.copy(
            integrity = finite(sample.integrity),
            reconstruction = finite(sample.reconstruction),
            geometry = finite(sample.geometry),
            temporal = finite(sample.temporal),
            synchronization = finite(sample.synchronization),
            calibration = finite(sample.calibration),
            bodyTracking = finite(sample.bodyTracking),
            faceEye = finite(sample.faceEye),
            faceEyeAccepted = sample.faceEyeAccepted,
            hand = finite(sample.hand),
            kinematic = finite(sample.kinematic),
            surface = finite(sample.surface),
            physics = finite(sample.physics),
            global4D = finite(sample.global4D),
            global4DDirectionalUncertaintyConsistency =
                finite(sample.global4DDirectionalUncertaintyConsistency),
            global4DFaceEyeConsistency =
                finite(sample.global4DFaceEyeConsistency),
            metricAuthority =
                finite(sample.metricAuthority),
            epipolarAuthority =
                finite(sample.epipolarAuthority),
            motionInnovationScore =
                finite(sample.motionInnovationScore),
        )

    private fun finite(value: Double): Double =
        if (value.isFinite()) {
            value.coerceIn(0.0, 1.0)
        } else {
            0.0
        }

    private fun emptySnapshot(): Snapshot =
        Snapshot(
            timestampNanos = 0L,
            sampleCount = 0,
            currentScore = 0.0,
            recentMeanScore = 0.0,
            baselineMeanScore = 0.0,
            trend = Trend.UNKNOWN,
            severity = Severity.CRITICAL,
            scoreSlopePerSample = 0.0,
            stabilityScore = 0.0,
            continuityScore = 0.0,
            degradationRatio = 1.0,
            heldFrameRatio = 1.0,
            hardFailureRatio = 1.0,
            dominantComponent = null,
            components = emptyList(),
            warnings = listOf("capture-evidence-ledger-empty"),
            recommendations = listOf(
                "Start a capture before evaluating rolling evidence.",
            ),
        )

    private fun DoubleArray.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}

private fun List<Double>.averageOrZero(): Double =
    if (isEmpty()) 0.0 else average()
