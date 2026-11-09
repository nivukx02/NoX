package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.LandmarkObservation
import com.nivukx.nox.core.model.LandmarkSpace
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Uncertainty-adaptive subpixel landmark refinement.
 *
 * This layer sits between detector output and dense reconstruction. It does not claim physical
 * accuracy that the sensor, optics, calibration or detector cannot support. It reduces avoidable
 * temporal jitter, compensates bounded rolling-shutter timing, rejects implausible innovations,
 * and carries explicit uncertainty into downstream evidence fusion.
 */
class SubpixelLandmarkRefinementEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val maximumLandmarks: Int = 4096,
        val maximumHistoryFrames: Int = 8,
        val staleStateMillis: Double = 900.0,
        val temporalHalfLifeMillis: Double = 90.0,
        val maximumCorrectionNormalized: Double = 0.028,
        val hardInnovationNormalized: Double = 0.22,
        val softInnovationNormalized: Double = 0.045,
        val maximumVelocityNormalizedPerSecond: Double = 12.0,
        val maximumAccelerationNormalizedPerSecond2: Double = 180.0,
        val maximumJerkNormalizedPerSecond3: Double = 1600.0,
        val minimumDetectorConfidence: Double = 0.05,
        val hardConfidenceFloor: Double = 0.01,
        val minimumGain: Double = 0.08,
        val maximumGain: Double = 0.92,
        val processNoiseBase: Double = 0.00002,
        val processNoiseMotionScale: Double = 0.00045,
        val measurementNoiseFloor: Double = 0.000002,
        val measurementNoiseCeiling: Double = 0.004,
        val rollingShutterScale: Double = 1.0,
        val microMotionThresholdNormalizedPerSecond: Double = 0.0015,
        val predictionHorizonMillis: Double = 80.0,
        val diagnosticHistoryLimit: Int = 128,
        val photometricMatchRadiusNormalized: Double = 0.075,
        val photometricAgreementScaleNormalized: Double = 0.015,
        val photometricEvidenceWeight: Double = 0.24,
        val photometricHardDisagreementNormalized: Double = 0.10,
    ) {
        init {
            require(maximumLandmarks > 0)
            require(maximumHistoryFrames >= 2)
            require(staleStateMillis > temporalHalfLifeMillis)
            require(temporalHalfLifeMillis > 0.0)
            require(maximumCorrectionNormalized > 0.0)
            require(hardInnovationNormalized > maximumCorrectionNormalized)
            require(softInnovationNormalized > 0.0)
            require(maximumVelocityNormalizedPerSecond > 0.0)
            require(maximumAccelerationNormalizedPerSecond2 > 0.0)
            require(maximumJerkNormalizedPerSecond3 > 0.0)
            require(minimumDetectorConfidence in 0.0..1.0)
            require(hardConfidenceFloor in 0.0..minimumDetectorConfidence)
            require(minimumGain in 0.0..1.0)
            require(maximumGain in 0.0..1.0)
            require(minimumGain <= maximumGain)
            require(processNoiseBase >= 0.0)
            require(processNoiseMotionScale >= 0.0)
            require(measurementNoiseFloor > 0.0)
            require(measurementNoiseCeiling > measurementNoiseFloor)
            require(rollingShutterScale >= 0.0)
            require(microMotionThresholdNormalizedPerSecond >= 0.0)
            require(predictionHorizonMillis >= 0.0)
            require(diagnosticHistoryLimit >= 8)
            require(photometricMatchRadiusNormalized > 0.0)
            require(photometricAgreementScaleNormalized > 0.0)
            require(photometricEvidenceWeight in 0.0..0.9)
            require(photometricHardDisagreementNormalized >
                photometricAgreementScaleNormalized)
        }
    }

    enum class RefinementState {
        INITIALIZING,
        OBSERVED,
        REFINED,
        MICRO_MOTION,
        HIGH_MOTION,
        LOW_CONFIDENCE,
        PREDICTED,
        REJECTED,
        STALE,
    }

    data class PhotometricHint(
        val normalizedX: Double,
        val normalizedY: Double,
        val normalizedDx: Double,
        val normalizedDy: Double,
        val confidence: Double,
        val gradientAgreement: Double,
        val textureStrength: Double,
        val forwardBackwardConsistency: Double,
        val subpixelRefined: Boolean = false,
    ) {
        init {
            require(normalizedX.isFinite())
            require(normalizedY.isFinite())
            require(normalizedDx.isFinite())
            require(normalizedDy.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(gradientAgreement.isFinite() && gradientAgreement in 0.0..1.0)
            require(textureStrength.isFinite() && textureStrength in 0.0..1.0)
            require(
                forwardBackwardConsistency.isFinite() &&
                    forwardBackwardConsistency in 0.0..1.0,
            )
        }
    }

    private data class PhotometricAssessment(
        val available: Boolean,
        val confidence: Double,
        val agreement: Double,
        val observedMotionError: Double,
        val expectedDx: Double,
        val expectedDy: Double,
        val hardDisagreement: Boolean,
    )

    data class RefinedLandmark(
        val landmarkId: Int,
        val timestampNanos: Long,
        val x: Double,
        val y: Double,
        val confidence: Double,
        val uncertaintyNormalized: Double,
        val innovationNormalized: Double,
        val correctionDistanceNormalized: Double,
        val rollingShutterCorrectionNormalized: Double,
        val temporalStability: Double,
        val velocityNormalizedPerSecond: Double,
        val accelerationNormalizedPerSecond2: Double,
        val microMotionScore: Double,
        val probabilitySigmaXNormalized: Double = 0.0,
        val probabilitySigmaYNormalized: Double = 0.0,
        val probabilityCalibrationScore: Double = 1.0,
        val state: RefinementState,
        val source: LandmarkObservation,
        val diagnostics: List<String>,
    )

    data class FrameResult(
        val timestampNanos: Long,
        val observations: List<RefinedLandmark>,
        val inputCount: Int,
        val acceptedCount: Int,
        val rejectedCount: Int,
        val predictedCount: Int,
        val nonFiniteCount: Int,
        val staleStateCount: Int,
        val meanConfidence: Double,
        val meanUncertaintyNormalized: Double,
        val uncertaintyP95Normalized: Double,
        val meanInnovationNormalized: Double,
        val maxCorrectionNormalized: Double,
        val meanRollingShutterCorrectionNormalized: Double,
        val temporalStability: Double,
        val microMotionPreservation: Double,
        val hardFailure: Boolean,
        val warnings: List<String>,
        val diagnostics: List<String>,
    ) {
        val authoritative: Boolean
            get() =
                !hardFailure &&
                    acceptedCount > 0 &&
                    meanConfidence >= 0.50 &&
                    uncertaintyP95Normalized <= 0.35 &&
                    temporalStability >= 0.50
    }

    data class Telemetry(
        val activeStates: Int,
        val latestTimestampNanos: Long?,
        val meanConfidence: Double,
        val uncertaintyP95Normalized: Double,
        val maxCorrectionNormalized: Double,
        val temporalStability: Double,
        val microMotionPreservation: Double,
        val rejectedCount: Long,
        val nonFiniteCount: Long,
        val staleStateCount: Long,
        val hardFailures: Long,
    )

    private data class LandmarkState(
        val landmarkId: Int,
        var timestampNanos: Long = Long.MIN_VALUE,
        var lastObservedTimestampNanos: Long = Long.MIN_VALUE,
        var x: Double = 0.0,
        var y: Double = 0.0,
        var vx: Double = 0.0,
        var vy: Double = 0.0,
        var ax: Double = 0.0,
        var ay: Double = 0.0,
        var covarianceX: Double = 0.002,
        var covarianceY: Double = 0.002,
        var uncertainty: Double = 0.50,
        var confidence: Double = 0.0,
        var stability: Double = 0.0,
        var lastInnovation: Double = 0.0,
        var observedFrames: Long = 0L,
        var predictedFrames: Long = 0L,
        var rejectedFrames: Long = 0L,
    )

    private val states = LinkedHashMap<Int, LandmarkState>()
    private val history = ArrayDeque<FrameResult>()
    private val diagnosticHistory = ArrayDeque<String>()
    private var closed = false
    private var lastFrameTimestampNanos = Long.MIN_VALUE
    private var latestResult: FrameResult? = null
    private var cumulativeRejected = 0L
    private var cumulativeNonFinite = 0L
    private var cumulativeStaleStates = 0L
    private var cumulativeHardFailures = 0L

    @Synchronized
    fun refine(
        observations: List<LandmarkObservation>,
        timestampNanos: Long = observations.maxOfOrNull { it.timestampNanos } ?: 0L,
        photometricHints: List<PhotometricHint> = emptyList(),
    ): FrameResult {
        check(!closed) { "SubpixelLandmarkRefinementEngine is closed" }

        if (
            timestampNanos < 0L ||
            (
                lastFrameTimestampNanos != Long.MIN_VALUE &&
                    timestampNanos < lastFrameTimestampNanos
                )
        ) {
            cumulativeHardFailures++
            appendDiagnostic("frame timestamp reversal rejected before subpixel refinement")
            val failed = emptyResult(
                timestampNanos.coerceAtLeast(0L),
                "Frame timestamp contract failed.",
                true,
            )
            latestResult = failed
            return failed
        }

        lastFrameTimestampNanos = timestampNanos
        val staleRemoved = prune(timestampNanos)
        val deduplicated = LinkedHashMap<Int, LandmarkObservation>()
        var nonFiniteCount = 0
        var rejectedCount = 0
        var duplicateCount = 0

        for (observation in observations) {
            if (!observationIsFinite(observation)) {
                nonFiniteCount++
                cumulativeNonFinite++
                continue
            }
            if (observation.timestampNanos > timestampNanos) {
                rejectedCount++
                cumulativeRejected++
                continue
            }
            if (observation.confidence < config.hardConfidenceFloor) {
                rejectedCount++
                cumulativeRejected++
                continue
            }
            val previous = deduplicated[observation.landmarkId]
            if (previous == null) {
                deduplicated[observation.landmarkId] = observation
            } else {
                duplicateCount++
                deduplicated[observation.landmarkId] = when {
                    observation.confidence > previous.confidence -> observation
                    observation.timestampNanos > previous.timestampNanos -> observation
                    else -> previous
                }
            }
        }

        val refined = ArrayList<RefinedLandmark>(deduplicated.size + states.size)
        for (observation in deduplicated.values) {
            val state = states.getOrPut(observation.landmarkId) {
                LandmarkState(observation.landmarkId)
            }
            refined +=
                refineObservation(
                    observation = observation,
                    state = state,
                    timestampNanos = timestampNanos,
                    photometricHints = photometricHints,
                )
        }

        if (config.predictionHorizonMillis > 0.0) {
            for ((landmarkId, state) in states) {
                if (landmarkId in deduplicated.keys) continue
                predictMissing(state, timestampNanos)?.let(refined::add)
            }
        }

        // Enforce the state-count ceiling after admitting the current frame.
        // This prevents a large single-frame burst from temporarily exceeding
        // the bounded in-memory landmark state budget.
        prune(timestampNanos)

        refined.sortBy { it.landmarkId }
        val accepted = refined.count {
            it.state != RefinementState.REJECTED &&
                it.state != RefinementState.STALE
        }
        val predicted = refined.count { it.state == RefinementState.PREDICTED }

        val confidenceValues = refined.map { it.confidence.coerceIn(0.0, 1.0) }
        val uncertaintyValues = refined.map { it.uncertaintyNormalized.coerceIn(0.0, 1.0) }
        val innovationValues = refined.map { it.innovationNormalized.coerceAtLeast(0.0) }
        val correctionValues = refined.map { it.correctionDistanceNormalized.coerceAtLeast(0.0) }
        val rollingValues = refined.map { it.rollingShutterCorrectionNormalized.coerceAtLeast(0.0) }
        val meanConfidence = confidenceValues.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val meanUncertainty = uncertaintyValues.takeIf { it.isNotEmpty() }?.average() ?: 1.0
        val uncertaintyP95 = percentile(uncertaintyValues, 0.95)
        val meanInnovation = innovationValues.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val maxCorrection = correctionValues.maxOrNull() ?: 0.0
        val meanRolling = rollingValues.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val temporalStability = refined.map { it.temporalStability }.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val microMotionPreservation = refined.map { it.microMotionScore }.takeIf { it.isNotEmpty() }?.average() ?: 0.0

        val warnings = buildList {
            if (meanConfidence < 0.45) add("subpixel refinement confidence is degraded")
            if (uncertaintyP95 > 0.35) add("subpixel uncertainty P95 is elevated")
            if (meanInnovation > config.softInnovationNormalized) add("landmark innovation is above the soft motion band")
            if (maxCorrection > config.maximumCorrectionNormalized * 0.80) add("some landmark corrections approach the configured cap")
            if (meanRolling > 0.01) add("rolling-shutter correction is materially active")
            if (predicted > 0) add("short observation gaps are being bridged by bounded prediction")
            if (duplicateCount > 0) add("duplicate landmark observations were suppressed")
            if (nonFiniteCount > 0) add("non-finite detector observations were rejected")
            if (staleRemoved > 0) add("stale landmark state was pruned")
        }

        val hardFailure =
            (accepted == 0 && observations.isNotEmpty()) ||
                (nonFiniteCount > 0 && accepted == 0) ||
                uncertaintyP95 >= 0.95
        if (hardFailure) cumulativeHardFailures++

        val diagnostics = listOf(
            "input=" + observations.size,
            "deduplicated=" + deduplicated.size,
            "accepted=" + accepted,
            "rejected=" + rejectedCount,
            "predicted=" + predicted,
            "uncertaintyP95=" + metric(uncertaintyP95),
            "meanInnovation=" + metric(meanInnovation),
            "maxCorrection=" + metric(maxCorrection),
            "rollingCorrection=" + metric(meanRolling),
            "temporalStability=" + metric(temporalStability),
            "microMotion=" + metric(microMotionPreservation),
        )
        diagnostics.forEach(::appendDiagnostic)

        val result = FrameResult(
            timestampNanos = timestampNanos,
            observations = refined,
            inputCount = observations.size,
            acceptedCount = accepted,
            rejectedCount = rejectedCount + refined.count { it.state == RefinementState.REJECTED },
            predictedCount = predicted,
            nonFiniteCount = nonFiniteCount,
            staleStateCount = staleRemoved,
            meanConfidence = meanConfidence,
            meanUncertaintyNormalized = meanUncertainty,
            uncertaintyP95Normalized = uncertaintyP95,
            meanInnovationNormalized = meanInnovation,
            maxCorrectionNormalized = maxCorrection,
            meanRollingShutterCorrectionNormalized = meanRolling,
            temporalStability = temporalStability,
            microMotionPreservation = microMotionPreservation,
            hardFailure = hardFailure,
            warnings = warnings.distinct(),
            diagnostics = diagnostics,
        )

        latestResult = result
        history.addLast(result)
        while (history.size > config.maximumHistoryFrames) history.removeFirst()
        return result
    }

    @Synchronized
    fun latest(): FrameResult? = latestResult

    @Synchronized
    fun diagnostics(): List<String> = diagnosticHistory.toList()

    @Synchronized
    fun telemetry(): Telemetry =
        Telemetry(
            activeStates = states.size,
            latestTimestampNanos = latestResult?.timestampNanos,
            meanConfidence = latestResult?.meanConfidence ?: 0.0,
            uncertaintyP95Normalized = latestResult?.uncertaintyP95Normalized ?: 1.0,
            maxCorrectionNormalized = latestResult?.maxCorrectionNormalized ?: 0.0,
            temporalStability = latestResult?.temporalStability ?: 0.0,
            microMotionPreservation = latestResult?.microMotionPreservation ?: 0.0,
            rejectedCount = cumulativeRejected,
            nonFiniteCount = cumulativeNonFinite,
            staleStateCount = cumulativeStaleStates,
            hardFailures = cumulativeHardFailures,
        )

    @Synchronized
    fun reset() {
        check(!closed) { "SubpixelLandmarkRefinementEngine is closed" }
        states.clear()
        history.clear()
        diagnosticHistory.clear()
        lastFrameTimestampNanos = Long.MIN_VALUE
        latestResult = null
        cumulativeRejected = 0L
        cumulativeNonFinite = 0L
        cumulativeStaleStates = 0L
        cumulativeHardFailures = 0L
    }

    @Synchronized
    override fun close() {
        if (closed) return
        reset()
        closed = true
    }

    private fun refineObservation(
        observation: LandmarkObservation,
        state: LandmarkState,
        timestampNanos: Long,
        photometricHints: List<PhotometricHint>,
    ): RefinedLandmark {
        val ageMillis =
            (timestampNanos - observation.timestampNanos).coerceAtLeast(0L).toDouble() / 1_000_000.0
        val freshness = temporalFreshness(ageMillis)
        val rawX = normalizedX(observation)
        val rawY = normalizedY(observation)
        val rolling = rollingShutterCorrection(observation, state)
        val correctedX = rawX - rolling.first
        val correctedY = rawY - rolling.second

        if (!correctedX.isFinite() || !correctedY.isFinite()) {
            state.rejectedFrames++
            return rejected(observation, state, timestampNanos, "rolling-shutter correction became non-finite")
        }

        if (state.timestampNanos == Long.MIN_VALUE) {
            state.timestampNanos = timestampNanos
            state.lastObservedTimestampNanos = timestampNanos
            state.x = correctedX
            state.y = correctedY
            state.confidence = observation.confidence
            state.uncertainty = initialUncertainty(observation)
            state.stability = (observation.confidence * freshness).coerceIn(0.0, 1.0)
            state.observedFrames++
            return RefinedLandmark(
                landmarkId = observation.landmarkId,
                timestampNanos = timestampNanos,
                x = correctedX,
                y = correctedY,
                confidence = observation.confidence,
                uncertaintyNormalized = state.uncertainty,
                innovationNormalized = 0.0,
                correctionDistanceNormalized = distance(correctedX, correctedY, rawX, rawY),
                rollingShutterCorrectionNormalized = magnitude(rolling.first, rolling.second),
                temporalStability = state.stability,
                velocityNormalizedPerSecond = 0.0,
                accelerationNormalizedPerSecond2 = 0.0,
                microMotionScore = 0.15,
                probabilitySigmaXNormalized =
                    sqrt(state.covarianceX).coerceIn(1e-4, 0.25),
                probabilitySigmaYNormalized =
                    sqrt(state.covarianceY).coerceIn(1e-4, 0.25),
                probabilityCalibrationScore =
                    (1.0 - state.uncertainty).coerceIn(0.0, 1.0),
                state = RefinementState.INITIALIZING,
                source = observation,
                diagnostics = listOf("initial detector observation accepted as temporal anchor"),
            )
        }

        val dt = (timestampNanos - state.timestampNanos).coerceAtLeast(1L).toDouble() / 1_000_000_000.0
        if (!dt.isFinite() || dt <= 1e-6 || dt > 1.0) {
            state.rejectedFrames++
            return rejected(observation, state, timestampNanos, "invalid temporal delta rejected")
        }

        val predicted = predict(state, dt)
        val photometric =
            assessPhotometricEvidence(
                observation = observation,
                currentX = correctedX,
                currentY = correctedY,
                state = state,
                predicted = predicted,
                hints = photometricHints,
            )
        val innovation = distance(correctedX, correctedY, predicted.first, predicted.second)
        if (innovation > config.hardInnovationNormalized) {
            state.rejectedFrames++
            state.uncertainty = (state.uncertainty * 0.76 + min(1.0, innovation / config.hardInnovationNormalized) * 0.24).coerceIn(0.0, 1.0)
            return RefinedLandmark(
                landmarkId = observation.landmarkId,
                timestampNanos = timestampNanos,
                x = predicted.first,
                y = predicted.second,
                confidence = (state.confidence * temporalFreshness(dt * 1000.0) * 0.68).coerceIn(0.0, 1.0),
                uncertaintyNormalized = state.uncertainty,
                innovationNormalized = innovation,
                correctionDistanceNormalized = distance(predicted.first, predicted.second, rawX, rawY).coerceAtMost(config.maximumCorrectionNormalized),
                rollingShutterCorrectionNormalized = magnitude(rolling.first, rolling.second),
                temporalStability = (state.stability * 0.78).coerceIn(0.0, 1.0),
                velocityNormalizedPerSecond = velocityMagnitude(state),
                accelerationNormalizedPerSecond2 = accelerationMagnitude(state),
                microMotionScore = microMotionScore(state),
                probabilitySigmaXNormalized =
                    sqrt(state.covarianceX).coerceIn(1e-4, 0.25),
                probabilitySigmaYNormalized =
                    sqrt(state.covarianceY).coerceIn(1e-4, 0.25),
                probabilityCalibrationScore =
                    (1.0 - state.uncertainty).coerceIn(0.0, 1.0),
                state = RefinementState.REJECTED,
                source = observation,
                diagnostics = listOf("innovation exceeded the hard temporal gate", "bounded prediction preserved continuity"),
            )
        }

        val measurementNoise = measurementNoise(observation, freshness)
        val processNoise =
            config.processNoiseBase +
                config.processNoiseMotionScale *
                    motionMagnitude(state)

        val photometricReliability =
            if (photometric.available) {
                (
                    0.35 +
                        0.65 *
                            photometric.agreement
                ) *
                    photometric.confidence
            } else {
                1.0
            }

        val photometricNoiseScale =
            if (photometric.available) {
                (
                    1.0 +
                        config.photometricEvidenceWeight *
                            (
                                1.0 -
                                    photometricReliability
                            ) *
                            4.0
                    ).coerceIn(1.0, 4.0)
            } else {
                1.0
            }

        val timingNoiseScale =
            (
                1.0 +
                    (
                        1.0 -
                            observation.captureTimingQuality
                    ) *
                        3.0
                ).coerceIn(1.0, 4.0)

        val combinedMeasurementNoise =
            (
                measurementNoise *
                    photometricNoiseScale *
                    timingNoiseScale
            ).coerceAtMost(
                config.measurementNoiseCeiling,
            )

        val gainX =
            adaptiveGain(
                state.covarianceX,
                combinedMeasurementNoise +
                    processNoise,
            )
        val gainY =
            adaptiveGain(
                state.covarianceY,
                combinedMeasurementNoise +
                    processNoise,
            )
        val innovationWeight =
            robustInnovationWeight(innovation)

        val photometricGainScale =
            if (photometric.available) {
                (
                    0.25 +
                        0.75 *
                            photometric.agreement *
                            photometric.confidence
                ).coerceIn(0.25, 1.0)
            } else {
                1.0
            }

        val effectiveGainX =
            (
                gainX *
                    innovationWeight *
                    photometricGainScale
            ).coerceIn(
                config.minimumGain,
                config.maximumGain,
            )
        val effectiveGainY =
            (
                gainY *
                    innovationWeight *
                    photometricGainScale
            ).coerceIn(
                config.minimumGain,
                config.maximumGain,
            )

        val refinedX = predicted.first + (correctedX - predicted.first) * effectiveGainX
        val refinedY = predicted.second + (correctedY - predicted.second) * effectiveGainY
        val correction = distance(refinedX, refinedY, rawX, rawY).coerceAtMost(config.maximumCorrectionNormalized)
        val measurementStability = exp(-innovation / config.softInnovationNormalized.coerceAtLeast(1e-9)).coerceIn(0.0, 1.0)
        val temporalStability = (state.stability * 0.62 + observation.confidence * 0.18 + measurementStability * 0.20).coerceIn(0.0, 1.0)

        updateKinematics(state, refinedX, refinedY, timestampNanos)
        state.covarianceX = ((1.0 - effectiveGainX) * state.covarianceX + processNoise).coerceIn(config.measurementNoiseFloor, config.measurementNoiseCeiling)
        state.covarianceY = ((1.0 - effectiveGainY) * state.covarianceY + processNoise).coerceIn(config.measurementNoiseFloor, config.measurementNoiseCeiling)

        val residualUncertainty =
            sqrt(
                state.covarianceX +
                    state.covarianceY,
            ).coerceIn(0.0, 1.0)

        val detectorConfidence =
            observation.confidence.coerceIn(
                config.hardConfidenceFloor,
                1.0,
            )

        val photometricConfidence =
            if (photometric.available) {
                (
                    photometric.confidence *
                        (
                            0.45 +
                                0.55 *
                                    photometric.agreement
                        )
                ).coerceIn(0.0, 1.0)
            } else {
                0.70
            }

        val confidence = (
            detectorConfidence * 0.46 +
                temporalStability * 0.26 +
                freshness * 0.08 +
                (1.0 - residualUncertainty) * 0.07 +
                photometricConfidence * 0.13
            ).coerceIn(0.0, 1.0)

        state.confidence = state.confidence * 0.45 + confidence * 0.55
        val disagreementInflation =
            if (
                photometric.available &&
                photometric.hardDisagreement
            ) {
                0.16
            } else if (
                photometric.available
            ) {
                (
                    1.0 -
                        photometric.agreement
                ) * 0.08
            } else {
                0.0
            }

        state.uncertainty =
            (
                state.uncertainty * 0.42 +
                    residualUncertainty * 0.50 +
                    disagreementInflation
            ).coerceIn(0.0, 1.0)
        state.stability = temporalStability
        state.timestampNanos = timestampNanos
        state.lastObservedTimestampNanos = timestampNanos
        state.lastInnovation = innovation
        state.observedFrames++

        val velocity = velocityMagnitude(state)
        val acceleration = accelerationMagnitude(state)
        val microMotion = microMotionScore(state)
        val refinementState = when {
            microMotion >= 0.65 && velocity <= config.maximumVelocityNormalizedPerSecond * 0.20 -> RefinementState.MICRO_MOTION
            acceleration >= config.maximumAccelerationNormalizedPerSecond2 * 0.30 -> RefinementState.HIGH_MOTION
            observation.confidence < 0.25 -> RefinementState.LOW_CONFIDENCE
            correction > 0.0002 -> RefinementState.REFINED
            else -> RefinementState.OBSERVED
        }

        val diagnostics = buildList {
            if (innovationWeight < 0.65) add("robust innovation weighting reduced detector influence")
            if (rolling.first != 0.0 || rolling.second != 0.0) add("rolling-shutter timing correction applied")
            if (observation.captureTimingQuality < 0.75) {
                add("capture timing quality widened measurement uncertainty")
            }
            if (photometric.available) {
                add(
                    "photometric motion evidence conditioned landmark refinement"
                )
            }
            if (
                photometric.available &&
                photometric.hardDisagreement
            ) {
                add(
                    "photometric motion disagreed with detector displacement; correction gain reduced"
                )
            }
            if (microMotion >= 0.50) add("micro-motion preservation active")
            if (refinementState == RefinementState.HIGH_MOTION) add("high-motion process noise is active")
            if (residualUncertainty > 0.35) add("refined landmark uncertainty is elevated")
        }

        return RefinedLandmark(
            landmarkId = observation.landmarkId,
            timestampNanos = timestampNanos,
            x = refinedX.coerceIn(-2.0, 3.0),
            y = refinedY.coerceIn(-2.0, 3.0),
            confidence = state.confidence.coerceIn(0.0, 1.0),
            uncertaintyNormalized = state.uncertainty.coerceIn(0.0, 1.0),
            innovationNormalized = innovation,
            correctionDistanceNormalized = correction,
            rollingShutterCorrectionNormalized = magnitude(rolling.first, rolling.second),
            temporalStability = temporalStability,
            velocityNormalizedPerSecond = velocity,
            accelerationNormalizedPerSecond2 = acceleration,
            microMotionScore = microMotion,
            probabilitySigmaXNormalized =
                sqrt(state.covarianceX).coerceIn(1e-4, 0.25),
            probabilitySigmaYNormalized =
                sqrt(state.covarianceY).coerceIn(1e-4, 0.25),
            probabilityCalibrationScore =
                (1.0 - state.uncertainty).coerceIn(0.0, 1.0),
            state = refinementState,
            source = observation,
            diagnostics = diagnostics,
        )
    }

    private fun assessPhotometricEvidence(
        observation: LandmarkObservation,
        currentX: Double,
        currentY: Double,
        state: LandmarkState,
        predicted: Pair<Double, Double>,
        hints: List<PhotometricHint>,
    ): PhotometricAssessment {
        if (hints.isEmpty()) {
            return PhotometricAssessment(
                available = false,
                confidence = 0.0,
                agreement = 1.0,
                observedMotionError = 0.0,
                expectedDx = 0.0,
                expectedDy = 0.0,
                hardDisagreement = false,
            )
        }

        val distanceThreshold =
            config.photometricMatchRadiusNormalized

        val candidate =
            hints
                .mapNotNull { hint ->
                    val dx =
                        hint.normalizedX -
                            currentX
                    val dy =
                        hint.normalizedY -
                            currentY
                    val distance =
                        sqrt(
                            dx * dx +
                                dy * dy,
                        )
                    if (
                        !distance.isFinite() ||
                        distance > distanceThreshold
                    ) {
                        null
                    } else {
                        hint to distance
                    }
                }
                .minByOrNull { it.second }
                ?: return PhotometricAssessment(
                    available = false,
                    confidence = 0.0,
                    agreement = 1.0,
                    observedMotionError = 0.0,
                    expectedDx = 0.0,
                    expectedDy = 0.0,
                    hardDisagreement = false,
                )

        val (hint, _) = candidate
        val observedDx =
            currentX -
                state.x
        val observedDy =
            currentY -
                state.y

        val mismatch =
            sqrt(
                square(
                    observedDx -
                        hint.normalizedDx
                ) +
                    square(
                        observedDy -
                            hint.normalizedDy
                    ),
            )

        val agreement =
            exp(
                -mismatch /
                    config.photometricAgreementScaleNormalized
            ).coerceIn(0.0, 1.0)

        val confidence =
            (
                hint.confidence * 0.42 +
                    hint.gradientAgreement * 0.20 +
                    hint.textureStrength * 0.14 +
                    hint.forwardBackwardConsistency * 0.24
                ).coerceIn(0.0, 1.0)

        val hardDisagreement =
            mismatch >
                config.photometricHardDisagreementNormalized &&
                confidence >= 0.70

        return PhotometricAssessment(
            available = true,
            confidence = confidence,
            agreement = agreement,
            observedMotionError = mismatch,
            expectedDx = hint.normalizedDx,
            expectedDy = hint.normalizedDy,
            hardDisagreement = hardDisagreement,
        )
    }

    private fun predictMissing(
        state: LandmarkState,
        timestampNanos: Long,
    ): RefinedLandmark? {
        if (state.lastObservedTimestampNanos == Long.MIN_VALUE) return null
        val ageMillis = (timestampNanos - state.lastObservedTimestampNanos).coerceAtLeast(0L).toDouble() / 1_000_000.0
        if (ageMillis > config.predictionHorizonMillis) return null
        val predicted = predict(state, ageMillis / 1000.0)
        state.predictedFrames++
        state.timestampNanos = timestampNanos
        val decay = temporalFreshness(ageMillis) * 0.78
        val uncertainty = (state.uncertainty + ageMillis / config.predictionHorizonMillis.coerceAtLeast(1.0) * 0.30).coerceIn(0.0, 1.0)
        return RefinedLandmark(
            landmarkId = state.landmarkId,
            timestampNanos = timestampNanos,
            x = predicted.first,
            y = predicted.second,
            confidence = (state.confidence * decay).coerceIn(0.0, 1.0),
            uncertaintyNormalized = uncertainty,
            innovationNormalized = 0.0,
            correctionDistanceNormalized = 0.0,
            rollingShutterCorrectionNormalized = 0.0,
            temporalStability = (state.stability * decay).coerceIn(0.0, 1.0),
            velocityNormalizedPerSecond = velocityMagnitude(state),
            accelerationNormalizedPerSecond2 = accelerationMagnitude(state),
            microMotionScore = microMotionScore(state),
            probabilitySigmaXNormalized =
                sqrt(state.covarianceX).coerceIn(1e-4, 0.25),
            probabilitySigmaYNormalized =
                sqrt(state.covarianceY).coerceIn(1e-4, 0.25),
            probabilityCalibrationScore =
                (1.0 - uncertainty).coerceIn(0.0, 1.0),
            state = RefinementState.PREDICTED,
            source = LandmarkObservation(
                landmarkId = state.landmarkId,
                x = predicted.first,
                y = predicted.second,
                confidence = (state.confidence * decay).coerceIn(0.0, 1.0),
                timestampNanos = timestampNanos,
                space = LandmarkSpace.NORMALIZED,
                captureTimingConfidence = state.confidence.coerceIn(0.0, 1.0),
            ),
            diagnostics = listOf("short detector gap bridged by bounded constant-acceleration prediction"),
        )
    }

    private fun rollingShutterCorrection(
        observation: LandmarkObservation,
        state: LandmarkState,
    ): Pair<Double, Double> {
        if (
            observation.rollingShutterSkewNanos <= 0L ||
            state.timestampNanos == Long.MIN_VALUE
        ) {
            return 0.0 to 0.0
        }
        val sensorRowNormalized =
            observation.sensorRowNormalized ?: return 0.0 to 0.0
        val rowOffsetSeconds =
            (
                (
                    sensorRowNormalized.coerceIn(0.0, 1.0) -
                        0.5
                    ) *
                    observation.rollingShutterSkewNanos.toDouble()
                ) /
                1_000_000_000.0
        val dx = state.vx * rowOffsetSeconds * config.rollingShutterScale.coerceIn(0.0, 1.0)
        val dy = state.vy * rowOffsetSeconds * config.rollingShutterScale.coerceIn(0.0, 1.0)
        val m = magnitude(dx, dy)
        if (!m.isFinite()) return 0.0 to 0.0
        if (m <= config.maximumCorrectionNormalized) return dx to dy
        val scale = config.maximumCorrectionNormalized / m.coerceAtLeast(1e-9)
        return dx * scale to dy * scale
    }

    private fun updateKinematics(
        state: LandmarkState,
        x: Double,
        y: Double,
        timestampNanos: Long,
    ) {
        if (state.timestampNanos == Long.MIN_VALUE) {
            state.x = x
            state.y = y
            state.timestampNanos = timestampNanos
            return
        }
        val dt = (timestampNanos - state.timestampNanos).coerceAtLeast(1L).toDouble() / 1_000_000_000.0
        if (!dt.isFinite() || dt <= 1e-6) return
        val newVx = ((x - state.x) / dt).coerceIn(-config.maximumVelocityNormalizedPerSecond, config.maximumVelocityNormalizedPerSecond)
        val newVy = ((y - state.y) / dt).coerceIn(-config.maximumVelocityNormalizedPerSecond, config.maximumVelocityNormalizedPerSecond)
        val newAx = ((newVx - state.vx) / dt).coerceIn(-config.maximumAccelerationNormalizedPerSecond2, config.maximumAccelerationNormalizedPerSecond2)
        val newAy = ((newVy - state.vy) / dt).coerceIn(-config.maximumAccelerationNormalizedPerSecond2, config.maximumAccelerationNormalizedPerSecond2)
        val deltaAx = (newAx - state.ax).coerceIn(-config.maximumJerkNormalizedPerSecond3 * dt, config.maximumJerkNormalizedPerSecond3 * dt)
        val deltaAy = (newAy - state.ay).coerceIn(-config.maximumJerkNormalizedPerSecond3 * dt, config.maximumJerkNormalizedPerSecond3 * dt)
        state.ax = state.ax * 0.72 + (state.ax + deltaAx) * 0.28
        state.ay = state.ay * 0.72 + (state.ay + deltaAy) * 0.28
        state.vx = state.vx * 0.62 + newVx * 0.38
        state.vy = state.vy * 0.62 + newVy * 0.38
        state.x = x
        state.y = y
    }

    private fun predict(state: LandmarkState, dtSeconds: Double): Pair<Double, Double> {
        val maxDt = (config.predictionHorizonMillis / 1000.0).coerceAtLeast(0.0)
        val dt = dtSeconds.coerceIn(0.0, maxDt)
        val speed = velocityMagnitude(state).coerceAtLeast(1e-9)
        val acceleration = accelerationMagnitude(state).coerceAtLeast(1e-9)
        val speedScale = min(1.0, config.maximumVelocityNormalizedPerSecond / speed)
        val accelerationScale = min(1.0, config.maximumAccelerationNormalizedPerSecond2 / acceleration)
        val vx = state.vx * speedScale
        val vy = state.vy * speedScale
        val ax = state.ax * accelerationScale
        val ay = state.ay * accelerationScale
        return (state.x + vx * dt + 0.5 * ax * dt * dt) to
            (state.y + vy * dt + 0.5 * ay * dt * dt)
    }

    private fun adaptiveGain(covariance: Double, measurementNoise: Double): Double {
        val denominator = covariance + measurementNoise
        if (!denominator.isFinite() || denominator <= 1e-12) return config.minimumGain
        return (covariance / denominator).coerceIn(config.minimumGain, config.maximumGain)
    }

    private fun robustInnovationWeight(innovation: Double): Double {
        if (!innovation.isFinite()) return 0.05
        val normalized = innovation / config.softInnovationNormalized.coerceAtLeast(1e-9)
        if (normalized <= 1.0) return 1.0
        return (1.0 / normalized).coerceIn(0.05, 1.0)
    }

    private fun measurementNoise(observation: LandmarkObservation, freshness: Double): Double {
        val confidence = observation.confidence.coerceIn(config.hardConfidenceFloor, 1.0)
        val timing = observation.captureTimingConfidence.coerceIn(0.0, 1.0)
        val raw = config.measurementNoiseFloor +
            (1.0 - confidence) * 0.0015 +
            (1.0 - timing) * 0.0008 +
            (1.0 - freshness) * 0.0006
        return raw.coerceIn(config.measurementNoiseFloor, config.measurementNoiseCeiling)
    }

    private fun initialUncertainty(observation: LandmarkObservation): Double {
        val confidenceLoss = (1.0 - observation.confidence.coerceIn(0.0, 1.0)) * 0.68
        val timingLoss = (1.0 - observation.captureTimingConfidence.coerceIn(0.0, 1.0)) * 0.20
        val rollingLoss = if (observation.rollingShutterSkewNanos > 0L) 0.08 else 0.02
        return (confidenceLoss + timingLoss + rollingLoss).coerceIn(0.0, 1.0)
    }

    private fun microMotionScore(state: LandmarkState): Double {
        val speed = velocityMagnitude(state)
        val threshold = config.microMotionThresholdNormalizedPerSecond.coerceAtLeast(1e-6)
        val normalized = speed / (threshold * 8.0)
        return when {
            normalized <= 0.05 -> 0.15
            normalized <= 0.30 -> 0.72
            normalized <= 1.0 -> 1.0
            normalized <= 2.5 -> 0.76
            else -> 0.40
        }
    }

    private fun motionMagnitude(state: LandmarkState): Double =
        (
            velocityMagnitude(state) / config.maximumVelocityNormalizedPerSecond +
                accelerationMagnitude(state) / config.maximumAccelerationNormalizedPerSecond2
            ) / 2.0
            .coerceIn(0.0, 1.0)

    private fun velocityMagnitude(state: LandmarkState): Double = sqrt(square(state.vx) + square(state.vy))
    private fun accelerationMagnitude(state: LandmarkState): Double = sqrt(square(state.ax) + square(state.ay))

    private fun normalizedX(observation: LandmarkObservation): Double =
        when (observation.space) {
            LandmarkSpace.NORMALIZED -> observation.x
            LandmarkSpace.PIXELS -> observation.x / observation.frameWidth.coerceAtLeast(1).toDouble()
            else -> observation.x
        }

    private fun normalizedY(observation: LandmarkObservation): Double =
        when (observation.space) {
            LandmarkSpace.NORMALIZED -> observation.y
            LandmarkSpace.PIXELS -> observation.y / observation.frameHeight.coerceAtLeast(1).toDouble()
            else -> observation.y
        }

    private fun observationIsFinite(observation: LandmarkObservation): Boolean =
        observation.timestampNanos >= 0L &&
            observation.landmarkId >= 0 &&
            observation.x.isFinite() &&
            observation.y.isFinite() &&
            observation.confidence.isFinite() &&
            observation.captureTimingConfidence.isFinite() &&
            observation.sensorRowNormalized?.let { it.isFinite() } ?: true

    private fun temporalFreshness(ageMillis: Double): Double {
        if (!ageMillis.isFinite()) return 0.0
        if (ageMillis <= 0.0) return 1.0
        return exp(-ln(2.0) * ageMillis / config.temporalHalfLifeMillis).coerceIn(0.0, 1.0)
    }

    private fun prune(timestampNanos: Long): Int {
        val cutoff = timestampNanos - (config.staleStateMillis * 1_000_000.0).toLong().coerceAtLeast(0L)
        val iterator = states.entries.iterator()
        var removed = 0
        while (iterator.hasNext()) {
            val state = iterator.next().value
            if (state.lastObservedTimestampNanos != Long.MIN_VALUE && state.lastObservedTimestampNanos < cutoff) {
                iterator.remove()
                removed++
            }
        }
        cumulativeStaleStates += removed.toLong()
        while (states.size > config.maximumLandmarks) {
            val oldest = states.entries.minByOrNull {
                it.value.lastObservedTimestampNanos.takeIf { ts -> ts != Long.MIN_VALUE } ?: Long.MIN_VALUE
            }?.key ?: break
            states.remove(oldest)
            removed++
        }
        return removed
    }

    private fun rejected(observation: LandmarkObservation, state: LandmarkState, timestampNanos: Long, diagnostic: String): RefinedLandmark =
        RefinedLandmark(
            landmarkId = observation.landmarkId,
            timestampNanos = timestampNanos,
            x = state.x,
            y = state.y,
            confidence = (state.confidence * 0.74).coerceIn(0.0, 1.0),
            uncertaintyNormalized = (state.uncertainty * 0.86 + 0.14).coerceIn(0.0, 1.0),
            innovationNormalized = state.lastInnovation,
            correctionDistanceNormalized = 0.0,
            rollingShutterCorrectionNormalized = 0.0,
            temporalStability = (state.stability * 0.70).coerceIn(0.0, 1.0),
            velocityNormalizedPerSecond = velocityMagnitude(state),
            accelerationNormalizedPerSecond2 = accelerationMagnitude(state),
            microMotionScore = microMotionScore(state),
            state = RefinementState.REJECTED,
            source = observation,
            diagnostics = listOf(diagnostic),
        )

    private fun percentile(values: List<Double>, quantile: Double): Double {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 1.0
        if (sorted.size == 1) return sorted.first()
        val position = quantile.coerceIn(0.0, 1.0) * (sorted.size - 1).toDouble()
        val lower = position.toInt()
        val upper = min(sorted.lastIndex, lower + 1)
        val fraction = position - lower
        return (sorted[lower] + (sorted[upper] - sorted[lower]) * fraction).coerceAtLeast(0.0)
    }

    private fun magnitude(x: Double, y: Double): Double = sqrt(square(x) + square(y))
    private fun distance(x1: Double, y1: Double, x2: Double, y2: Double): Double = magnitude(x1 - x2, y1 - y2)
    private fun square(value: Double): Double = value * value
    private fun metric(value: Double): String = "%.5f".format(value.coerceIn(-99.0, 99.0))

    private fun appendDiagnostic(message: String) {
        diagnosticHistory.addLast(message)
        while (diagnosticHistory.size > config.diagnosticHistoryLimit) diagnosticHistory.removeFirst()
    }

    private fun emptyResult(timestampNanos: Long, warning: String, hardFailure: Boolean): FrameResult =
        FrameResult(
            timestampNanos = timestampNanos,
            observations = emptyList(),
            inputCount = 0,
            acceptedCount = 0,
            rejectedCount = 0,
            predictedCount = 0,
            nonFiniteCount = 0,
            staleStateCount = 0,
            meanConfidence = 0.0,
            meanUncertaintyNormalized = 1.0,
            uncertaintyP95Normalized = 1.0,
            meanInnovationNormalized = 0.0,
            maxCorrectionNormalized = 0.0,
            meanRollingShutterCorrectionNormalized = 0.0,
            temporalStability = 0.0,
            microMotionPreservation = 0.0,
            hardFailure = hardFailure,
            warnings = listOf(warning),
            diagnostics = listOf("subpixel landmark refinement produced no trustworthy result"),
        )
}