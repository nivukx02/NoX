package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * High-end temporal motion estimator for 2D detector landmarks.
 *
 * The engine is deliberately independent from any specific AI model. It converts noisy detector
 * measurements into a timestamped, uncertainty-aware motion state that downstream triangulation can
 * consume without confusing "smooth" with "correct".
 *
 * Pipeline:
 *  1. constant-acceleration prediction;
 *  2. robust innovation gating;
 *  3. confidence-weighted measurement correction;
 *  4. adaptive velocity/acceleration learning;
 *  5. bounded occlusion prediction;
 *  6. jerk / jitter / stability estimation;
 *  7. capture-readiness classification.
 *
 * Coordinates are normalized to [0,1] and all dynamics are expressed per second.
 */
class MotionCaptureEngine(
    private val maximumPredictionGapNanos: Long = 180_000_000L,
    private val maxLandmarkSpeedNormalized: Double = 6.0,
    private val maxLandmarkAccelerationNormalized: Double = 40.0,
    private val minimumMeasurementConfidence: Double = 0.08,
    private val hardResidualNormalized: Double = 0.22,
    private val predictionConfidenceFloor: Double = 0.16,
    private val reacquisitionDistanceNormalized: Double = 0.11,
    private val reacquisitionMinimumConfidence: Double = 0.34,
    private val maximumStateAgeNanos: Long = 900_000_000L,
    private val maximumTrackedLandmarks: Int = 4_096,
) : AutoCloseable {

    companion object {
        /**
         * Converts measured Camera2 rolling-shutter metadata into a bounded motion-risk signal.
         *
         * This signal is deliberately conservative. A high readout/frame ratio increases risk,
         * while poor timing confidence adds additional system-level uncertainty. It never creates
         * a positive motion measurement.
         */
        fun rollingShutterRiskFromTelemetry(
            rollingShutterSkewNanos: Long,
            frameDurationNanos: Long,
            captureTimingConfidence: Double,
        ): Double {
            require(rollingShutterSkewNanos >= 0L)
            require(frameDurationNanos >= 0L)
            require(captureTimingConfidence in 0.0..1.0)

            if (rollingShutterSkewNanos == 0L) {
                return 0.0
            }

            val frameDuration = maxOf(
                frameDurationNanos,
                1_000_000L,
            ).toDouble()
            val readoutRatio = (
                rollingShutterSkewNanos.toDouble() /
                    frameDuration
            ).coerceAtLeast(0.0)

            val readoutRisk = (
                readoutRatio / 0.75
            ).coerceIn(0.0, 1.0)

            val timingRisk =
                1.0 - captureTimingConfidence.coerceIn(0.0, 1.0)

            return (
                0.78 * readoutRisk +
                    0.22 * timingRisk
                ).coerceIn(0.0, 1.0)
        }
    }

    @Volatile
    private var tuning = MotionCaptureTuning()

    fun configure(newTuning: MotionCaptureTuning) {
        tuning = newTuning
    }

    fun tuning(): MotionCaptureTuning = tuning

    enum class MotionClass {
        STATIC,
        MICRO_MOTION,
        NORMAL,
        FAST,
        EXTREME,
    }

    /**
     * External evidence used to distinguish subject motion from camera/system motion.
     *
     * These values modulate temporal trust; they do not replace landmark measurements.
     */
    data class MotionContext(
        val cameraMotionRisk: Double = 0.0,
        val sensorConfidence: Double = 1.0,
        val rollingShutterRisk: Double = 0.0,
        val globalMotionConfidence: Double = 0.0,
        /**
         * Optical integration blur risk derived from measured exposure/frame timing.
         * Kept separate from rolling-shutter risk because global exposure blur can corrupt
         * correspondence even when row timing is benign.
         */
        val motionBlurRisk: Double = 0.0,
    ) {
        init {
            require(cameraMotionRisk in 0.0..1.0)
            require(sensorConfidence in 0.0..1.0)
            require(rollingShutterRisk in 0.0..1.0)
            require(globalMotionConfidence in 0.0..1.0)
            require(motionBlurRisk in 0.0..1.0)
        }

        val cameraDominance: Double
            get() = (
                cameraMotionRisk *
                    (0.55 + 0.45 * sensorConfidence) *
                    (0.70 + 0.30 * globalMotionConfidence)
                ).coerceIn(0.0, 1.0)

        val measurementTrustScale: Double
            get() = (
                1.0 -
                    cameraDominance * 0.48 -
                    rollingShutterRisk * 0.18 -
                    motionBlurRisk * 0.24
                ).coerceIn(0.28, 1.0)
    }

    data class LandmarkQuality(
        val id: Int,
        val confidence: Double,
        val innovationNormalized: Double,
        val speedNormalizedPerSecond: Double,
        val accelerationNormalizedPerSecond2: Double,
        val jerkNormalizedPerSecond3: Double,
        val measurementUncertaintyScore: Double,
        val predicted: Boolean,
        val rejectedMeasurement: Boolean,
    )

    data class Quality(
        val timestampNanos: Long,
        val motionScore: Double,
        val motionClass: MotionClass,
        val tuningResponsiveness: Double = 0.82,
        val predictionHorizonSeconds: Double = 0.18,
        val reacquisitionRadiusNormalized: Double = 0.11,

        val meanSpeedNormalizedPerSecond: Double,
        val maxSpeedNormalizedPerSecond: Double,
        val meanAccelerationNormalizedPerSecond2: Double,
        val maxAccelerationNormalizedPerSecond2: Double,
        val meanJerkNormalizedPerSecond3: Double,
        val jitterNormalized: Double,
        val trackedLandmarks: Int,
        val predictedLandmarks: Int,
        val rejectedLandmarks: Int,
        val meanConfidence: Double,
        val occlusionRatio: Double,
        val temporalStability: Double,
        val spatialSpread: Double,
        val stationaryRatio: Double,
        val dynamicsConfidence: Double,
        val meanInnovationNormalized: Double,
        val rejectedMeasurementRatio: Double,
        val directionChangeRatio: Double = 0.0,
        val microMotionScore: Double = 0.0,
        val meanMeasurementUncertaintyScore: Double = 0.0,
        val motionBlurRisk: Double = 0.0,
        val commonModeMotionNormalizedPerSecond: Double = 0.0,
        val differentialMotionScore: Double = 0.0,
        val cameraCompensatedMotionScore: Double = 0.0,
        val dominantFrequencyHz: Double = 0.0,
        val periodicMotionScore: Double = 0.0,
        val vibrationRisk: Double = 0.0,
        val spectralConfidence: Double = 0.0,
        val stateCapacityEvictions: Long = 0L,
    ) {
        val healthy: Boolean
            get() =
                trackedLandmarks >= 2 &&
                    meanConfidence >= 0.35 &&
                    occlusionRatio <= 0.40 &&
                    rejectedMeasurementRatio <= 0.45

        val captureSafe: Boolean
            get() =
                healthy &&
                    jitterNormalized <= 0.75 &&
                    temporalStability >= 0.35 &&
                    dynamicsConfidence >= 0.30 &&
                    meanMeasurementUncertaintyScore <= 0.72 &&
                    motionBlurRisk <= 0.90 &&
                    (
                        spectralConfidence < 0.45 ||
                            vibrationRisk <= 0.82
                        )

        val captureReadiness: Double
            get() =
                (
                    0.28 * meanConfidence +
                        0.24 * temporalStability +
                        0.18 * (1.0 - occlusionRatio) +
                        0.16 * dynamicsConfidence +
                        0.10 * (1.0 - jitterNormalized) +
                        0.03 * (1.0 - meanMeasurementUncertaintyScore) +
                        0.04 * (1.0 - motionBlurRisk) +
                        0.04 * (
                            if (spectralConfidence >= 0.45) {
                                1.0 - vibrationRisk
                            } else {
                                0.75
                            }
                        )
                    ).coerceIn(0.0, 1.0)
    }

    data class Refined(
        val result: PoseDetectionEngine.Result,
        val quality: Quality,
        val landmarkQuality: List<LandmarkQuality>,
    )

    private data class State(
        var name: String,
        var x: Double,
        var y: Double,
        var vx: Double,
        var vy: Double,
        var ax: Double,
        var ay: Double,
        var jerk: Double,
        var confidence: Double,
        var residualEma: Double,
        var speedEma: Double,
        var accelerationEma: Double,
        var lastObservedNanos: Long,
        var lastTimestampNanos: Long,
        var observationCount: Long,
        var rejectedCount: Long,
        var stableFrames: Int,
    )

    private val states = HashMap<Int, State>()
    private var lastTimestampNanos = 0L
    private var lastRefined: Refined? = null
    private var stateCapacityEvictions = 0L

    private val temporalMotionSpectrumAnalyzer =
        TemporalMotionSpectrumAnalyzer()

    private val motionRegimeAdaptiveController =
        MotionRegimeAdaptiveController()

    @Volatile
    private var adaptivePolicy =
        motionRegimeAdaptiveController.identityPolicy("initial")

    private val motionSpectrumHistory =
        java.util.ArrayDeque<
            TemporalMotionSpectrumAnalyzer.Sample
            >()

    @Synchronized
    fun refine(
        input: PoseDetectionEngine.Result,
        context: MotionContext = MotionContext(),
    ): Refined {
        val timestamp = input.timestampNanos
        if (
            lastTimestampNanos != 0L &&
            timestamp <= lastTimestampNanos
        ) {
            val previous = lastRefined
            if (previous != null) {
                return previous
            }
            reset()
        }
        pruneExpiredStates(timestamp)
        enforceStateCapacity()
        val width = input.imageWidth.coerceAtLeast(1).toDouble()
        val height = input.imageHeight.coerceAtLeast(1).toDouble()
        val dt = sampleDeltaSeconds(timestamp)
        val cfg = tuning
        val policy = adaptivePolicy
        val configuredPredictionGapNanos = (
            cfg.predictionHorizonSeconds *
                policy.predictionHorizonScale *
                1_000_000_000.0
        ).toLong().coerceIn(40_000_000L, 350_000_000L)
        val effectivePredictionGapNanos =
            minOf(
                configuredPredictionGapNanos
                    .coerceAtLeast(1_000_000L),
                maximumPredictionGapNanos
                    .coerceAtLeast(1_000_000L)
                    .coerceAtMost(350_000_000L),
            )

        val currentIds = input.landmarks.map { it.id }.toHashSet()
        val output = ArrayList<PoseDetectionEngine.Landmark>(
            input.landmarks.size + states.size,
        )
        val quality = ArrayList<LandmarkQuality>(
            input.landmarks.size + states.size,
        )
        val reacquisitionCandidateIds = states.keys.toHashSet()

        var rejected = 0
        var predictedCount = 0
        var directionChanges = 0
        var microMotionSamples = 0

        val speeds = ArrayList<Double>()
        val accelerations = ArrayList<Double>()
        val jerks = ArrayList<Double>()
        val innovations = ArrayList<Double>()
        val confidences = ArrayList<Double>()
        val measurementUncertainties = ArrayList<Double>()

        input.landmarks.forEach { measurement ->
            val confidence =
                measurement.confidence
                    .toDouble()
                    .coerceIn(0.0, 1.0)

            if (confidence < cfg.minimumConfidence) {
                return@forEach
            }

            val existingById = states[measurement.id]
            val reacquired =
                if (existingById == null) {
                    findReacquisitionState(
                        measurement = measurement,
                        width = width,
                        height = height,
                        timestampNanos = timestamp,
                        candidateIds = reacquisitionCandidateIds,
                    )
                } else {
                    null
                }

            if (existingById == null && reacquired != null) {
                states.remove(reacquired.first)
                states[measurement.id] = reacquired.second
            }

            val state =
                existingById
                    ?: reacquired?.second
                    ?: State(
                        name = measurement.name,
                        x = (measurement.x / width)
                            .toDouble()
                            .coerceIn(0.0, 1.0),
                        y = (measurement.y / height)
                            .toDouble()
                            .coerceIn(0.0, 1.0),
                        vx = 0.0,
                        vy = 0.0,
                        ax = 0.0,
                        ay = 0.0,
                        jerk = 0.0,
                        confidence = confidence,
                        residualEma = 0.0,
                        speedEma = 0.0,
                        accelerationEma = 0.0,
                        lastObservedNanos = timestamp,
                        lastTimestampNanos = timestamp,
                        observationCount = 1L,
                        rejectedCount = 0L,
                        stableFrames = 1,
                    ).also {
                        states[measurement.id] = it
                    }

            val localDt =
                (
                    (
                        timestamp -
                            state.lastTimestampNanos
                        ).coerceAtLeast(1L) / 1e9
                    ).coerceIn(
                        1.0 / 240.0,
                        0.25,
                    )

            val predictionX =
                (
                    state.x +
                        state.vx * localDt +
                        0.5 * state.ax *
                        localDt * localDt
                    ).coerceIn(0.0, 1.0)

            val predictionY =
                (
                    state.y +
                        state.vy * localDt +
                        0.5 * state.ay *
                        localDt * localDt
                    ).coerceIn(0.0, 1.0)

            val measuredX =
                (
                    measurement.x /
                        width
                    ).toDouble().coerceIn(0.0, 1.0)

            val measuredY =
                (
                    measurement.y /
                        height
                    ).toDouble().coerceIn(0.0, 1.0)

            val residualX =
                measuredX -
                    predictionX

            val residualY =
                measuredY -
                    predictionY

            val innovation =
                hypot(
                    residualX,
                    residualY,
                )

            val speedPrior =
                hypot(
                    state.vx,
                    state.vy,
                )

            val predictedTravel =
                speedPrior *
                    localDt

            val adaptiveGate =
                (
                    hardResidualNormalized *
                        cfg.innovationGateElasticity *
                        policy.innovationGateScale *
                        (
                            1.0 +
                                (1.0 - cfg.rejectAggressiveness) * 0.34 +
                                context.cameraDominance * 0.42 +
                                context.rollingShutterRisk * 0.16 +
                        context.motionBlurRisk * 0.10
                        ) +
                        min(
                            0.10,
                            predictedTravel *
                                (0.45 + 0.40 * cfg.responsiveness),
                        ) +
                        (1.0 - confidence) * 0.10
                    ).coerceIn(
                        0.045,
                        0.46,
                    )

            val rejectedMeasurement =
                innovation > adaptiveGate &&
                    confidence < 0.82

            if (rejectedMeasurement) {
                rejected += 1
                state.rejectedCount += 1L
            }

            val robustWeight =
                if (rejectedMeasurement) {
                    0.025
                } else {
                    val normalizedResidual =
                        innovation /
                            max(
                                adaptiveGate,
                                1e-6,
                            )

                    val huberWeight =
                        if (
                            normalizedResidual <=
                                1.0
                        ) {
                            1.0
                        } else {
                            1.0 /
                                normalizedResidual
                        }

                    (
                        0.16 +
                            0.72 *
                            confidence
                        ) *
                        huberWeight *
                        context.measurementTrustScale
                }.coerceIn(
                    0.02,
                    0.92,
                )

            val correctedX =
                (
                    predictionX +
                        residualX *
                        robustWeight
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

            val correctedY =
                (
                    predictionY +
                        residualY *
                        robustWeight
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

            val previousVx =
                state.vx
            val previousVy =
                state.vy
            val previousAx =
                state.ax
            val previousAy =
                state.ay

            val measuredVx =
                (
                    correctedX -
                        state.x
                    ) /
                    localDt

            val measuredVy =
                (
                    correctedY -
                        state.y
                    ) /
                    localDt

            val velocityGain =
                (
                    (
                        0.08 +
                            0.58 * confidence +
                            0.14 *
                            min(
                            speedPrior /
                                2.0,
                            1.0,
                            )
                        ) *
                        (0.62 + 0.46 * cfg.responsiveness) *
                        cfg.velocityResponsiveness *
                        policy.velocityResponsivenessScale *
                        (
                            1.0 -
                                context.cameraDominance * 0.36 -
                                context.rollingShutterRisk * 0.12
                            ).coerceIn(0.38, 1.0)
                    ).coerceIn(
                        0.06,
                        0.94,
                    )

            state.vx +=
                (
                    measuredVx -
                        state.vx
                    ) *
                    velocityGain

            state.vy +=
                (
                    measuredVy -
                        state.vy
                    ) *
                    velocityGain

            state.vx =
                state.vx.coerceIn(
                    -maxLandmarkSpeedNormalized,
                    maxLandmarkSpeedNormalized,
                )
            state.vy =
                state.vy.coerceIn(
                    -maxLandmarkSpeedNormalized,
                    maxLandmarkSpeedNormalized,
                )

            val measuredAx =
                (
                    state.vx -
                        previousVx
                    ) /
                    localDt

            val measuredAy =
                (
                    state.vy -
                        previousVy
                    ) /
                    localDt

            val accelerationGain =
                (
                    (
                        0.10 +
                            0.32 *
                            confidence
                        ) *
                        cfg.accelerationResponsiveness *
                            policy.accelerationResponsivenessScale
                    ).coerceIn(
                        0.08,
                        0.72,
                    )

            state.ax +=
                (
                    measuredAx -
                        state.ax
                    ) *
                    accelerationGain

            state.ay +=
                (
                    measuredAy -
                        state.ay
                    ) *
                    accelerationGain

            val accelerationMagnitude =
                hypot(
                    state.ax,
                    state.ay,
                )

            if (
                accelerationMagnitude >
                    maxLandmarkAccelerationNormalized
            ) {
                val scale =
                    maxLandmarkAccelerationNormalized /
                        accelerationMagnitude

                state.ax *= scale
                state.ay *= scale
            }

            val jerkMagnitude =
                hypot(
                    (state.ax - previousAx) /
                        localDt,
                    (state.ay - previousAy) /
                        localDt,
                )

            state.jerk +=
                (
                    jerkMagnitude -
                        state.jerk
                    ) *
                    0.18

            state.x = correctedX
            state.y = correctedY
            state.name = measurement.name
            state.lastTimestampNanos =
                timestamp
            state.lastObservedNanos =
                timestamp
            state.observationCount += 1L

            val jitterSuppression = (
                0.54 +
                    0.42 *
                        cfg.jitterSuppression *
                        policy.jitterSuppressionScale
            ).coerceIn(0.35, 0.94)

            val residualRetention = jitterSuppression
            state.residualEma =
                state.residualEma *
                    residualRetention +
                    innovation *
                    (1.0 - residualRetention)

            state.speedEma =
                state.speedEma *
                    0.72 +
                    hypot(
                        state.vx,
                        state.vy,
                    ) * 0.28

            state.accelerationEma =
                state.accelerationEma *
                    0.70 +
                    hypot(
                        state.ax,
                        state.ay,
                    ) * 0.30

            state.confidence =
                if (rejectedMeasurement) {
                    state.confidence *
                        0.76
                } else {
                    (
                        state.confidence *
                            0.34 +
                            confidence *
                            0.66
                        ).coerceIn(
                            0.0,
                            1.0,
                        )
                }

            if (
                !rejectedMeasurement &&
                innovation <
                    adaptiveGate * 0.75
            ) {
                state.stableFrames += 1
            } else {
                state.stableFrames =
                    (
                        state.stableFrames -
                            1
                        ).coerceAtLeast(0)
            }

            val innovationConfidence =
                (
                    1.0 -
                        innovation /
                        max(
                            adaptiveGate,
                            1e-6,
                        )
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

            val finalConfidence =
                (
                    0.58 * confidence +
                        0.24 *
                        state.confidence +
                        0.18 *
                        innovationConfidence
                    ).coerceIn(
                        0.0,
                        1.0,
                    )

            val measurementUncertaintyScore =
                (
                    0.42 *
                        (1.0 - innovationConfidence) +
                        0.38 *
                        (1.0 - finalConfidence) +
                        if (rejectedMeasurement) 0.20 else 0.0
                    ).coerceIn(0.0, 1.0)

            measurementUncertainties += measurementUncertaintyScore

            output +=
                PoseDetectionEngine.Landmark(
                    id = measurement.id,
                    name = measurement.name,
                    x =
                        (state.x * width)
                            .toFloat(),
                    y =
                        (state.y * height)
                            .toFloat(),
                    confidence =
                        finalConfidence
                            .toFloat(),
                )

            val speed =
                hypot(
                    state.vx,
                    state.vy,
                )

            val acceleration =
                hypot(
                    state.ax,
                    state.ay,
                )

            val previousSpeed =
                hypot(
                    previousVx,
                    previousVy,
                )

            val currentSpeed = speed

            val directionDot =
                if (
                    previousSpeed > 0.012 &&
                        currentSpeed > 0.012
                ) {
                    (
                        previousVx * state.vx +
                            previousVy * state.vy
                    ) /
                        (
                            previousSpeed *
                                currentSpeed
                        ).coerceAtLeast(1e-9)
                } else {
                    1.0
                }

            val measuredSpeed = hypot(measuredVx, measuredVy)
            val measuredDirectionDot =
                if (
                    previousSpeed > 0.0015 &&
                        measuredSpeed > 0.0015
                ) {
                    (
                        previousVx * measuredVx +
                            previousVy * measuredVy
                        ) /
                        (
                            previousSpeed *
                                measuredSpeed
                            ).coerceAtLeast(1e-9)
                } else {
                    1.0
                }

            if (
                (
                    previousSpeed > 0.0015 &&
                        currentSpeed > 0.0015 &&
                        directionDot < 0.0
                    ) ||
                    (
                        previousSpeed > 0.0015 &&
                            measuredSpeed > 0.0015 &&
                            measuredDirectionDot < 0.0
                        )
            ) {
                directionChanges += 1
            }

            if (
                measuredSpeed >= 0.0005 &&
                    measuredSpeed <= 0.25 &&
                    currentSpeed <= 0.25 &&
                    acceleration < 12.0 &&
                    !rejectedMeasurement
            ) {
                microMotionSamples += 1
            }

            speeds += speed
            accelerations += acceleration
            jerks += state.jerk
            innovations += innovation
            confidences += finalConfidence

            quality +=
                LandmarkQuality(
                    id = measurement.id,
                    confidence = finalConfidence,
                    innovationNormalized = innovation,
                    speedNormalizedPerSecond = speed,
                    accelerationNormalizedPerSecond2 = acceleration,
                    jerkNormalizedPerSecond3 = state.jerk,
                    measurementUncertaintyScore =
                        measurementUncertaintyScore,
                    predicted = false,
                    rejectedMeasurement =
                        rejectedMeasurement,
                )
        }

        states.entries
            .filter { (id, state) ->
                id !in currentIds &&
                    timestamp -
                        state.lastObservedNanos <=
                        effectivePredictionGapNanos
            }
            .forEach { (id, state) ->
                val predictionDt =
                    (
                        (
                            timestamp -
                                state.lastTimestampNanos
                            ).coerceAtLeast(1L) /
                            1e9
                        ).coerceIn(
                            1.0 / 240.0,
                            0.20,
                        )

                state.x =
                    (
                        state.x +
                            state.vx *
                            predictionDt +
                            0.5 *
                            state.ax *
                            predictionDt *
                            predictionDt
                        ).coerceIn(
                            0.0,
                            1.0,
                        )

                state.y =
                    (
                        state.y +
                            state.vy *
                            predictionDt +
                            0.5 *
                            state.ay *
                            predictionDt *
                            predictionDt
                        ).coerceIn(
                            0.0,
                            1.0,
                        )

                val decay =
                    exp(
                        -(
                            3.2 /
                                cfg.occlusionPersistence.coerceAtLeast(0.35)
                        ) * predictionDt,
                    )

                state.confidence =
                    (
                        state.confidence *
                            decay
                        ).coerceIn(
                            0.0,
                            1.0,
                        )

                state.lastTimestampNanos =
                    timestamp

                predictedCount += 1

                val predictedConfidence =
                    (
                        state.confidence *
                            0.82
                        ).coerceIn(
                            0.0,
                            1.0,
                        )

                val predictionUncertaintyScore =
                    (
                        0.72 *
                            (1.0 - predictedConfidence) +
                            0.28 *
                                (
                                    predictionDt / 0.20
                                ).coerceIn(0.0, 1.0)
                        ).coerceIn(0.0, 1.0)

                measurementUncertainties +=
                    predictionUncertaintyScore

                if (
                    predictedConfidence >=
                        predictionConfidenceFloor
                ) {
                    output +=
                        PoseDetectionEngine.Landmark(
                            id = id,
                            name = state.name,
                            x =
                                (
                                    state.x *
                                        width
                                    ).toFloat(),
                            y =
                                (
                                    state.y *
                                        height
                                    ).toFloat(),
                            confidence =
                                predictedConfidence
                                    .toFloat(),
                        )
                }

                val speed =
                    hypot(
                        state.vx,
                        state.vy,
                    )

                val acceleration =
                    hypot(
                        state.ax,
                        state.ay,
                    )

                speeds += speed
                accelerations +=
                    acceleration
                jerks += state.jerk
                innovations +=
                    state.residualEma
                confidences +=
                    predictedConfidence

                quality +=
                    LandmarkQuality(
                        id = id,
                        confidence =
                            predictedConfidence,
                        innovationNormalized =
                            state.residualEma,
                        speedNormalizedPerSecond =
                            speed,
                        accelerationNormalizedPerSecond2 =
                            acceleration,
                        jerkNormalizedPerSecond3 =
                            state.jerk,
                        measurementUncertaintyScore =
                            predictionUncertaintyScore,
                        predicted = true,
                        rejectedMeasurement = false,
                    )
            }

        states.entries
            .filter {
                timestamp -
                    it.value.lastObservedNanos >
                    effectivePredictionGapNanos
            }
            .toList()
            .forEach {
                states.remove(it.key)
            }

        val tracked =
            quality.count {
                it.confidence >=
                    predictionConfidenceFloor
            }

        val meanSpeed =
            speeds.averageOrZero()

        val maxSpeed =
            speeds.maxOrNull() ?: 0.0

        val meanAcceleration =
            accelerations.averageOrZero()

        val maxAcceleration =
            accelerations.maxOrNull()
                ?: 0.0

        val meanJerk =
            jerks.averageOrZero()

        val meanInnovation =
            innovations.averageOrZero()

        val meanConfidence =
            confidences.averageOrZero()

        val rejectedRatio =
            (
                rejected.toDouble() /
                    input.landmarks.size
                        .coerceAtLeast(1)
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val occlusionRatio =
            (
                predictedCount.toDouble() /
                    tracked.coerceAtLeast(1)
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val jitter =
            states.values
                .map { state ->
                    state.residualEma
                }
                .averageOrZero()
                .coerceIn(
                    0.0,
                    1.0,
                )

        val spatialSpread =
            normalizedSpread(
                output,
                width,
                height,
            )

        val stationaryRatio =
            if (quality.isEmpty()) {
                1.0
            } else {
                (
                    quality.count {
                        !it.predicted &&
                            it.speedNormalizedPerSecond <
                                0.025
                    }.toDouble() /
                        quality.size
                    ).coerceIn(
                        0.0,
                        1.0,
                    )
            }

        val temporalStability =
            (
                meanConfidence *
                    (1.0 - jitter * (0.62 + 0.38 * cfg.jitterSuppression)) *
                    (1.0 - 0.55 * occlusionRatio) *
                    (1.0 - 0.45 * rejectedRatio) *
                    context.sensorConfidence *
                    (1.0 - 0.30 * context.rollingShutterRisk)
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val rawMotionScore =
            (
                1.0 -
                    exp(
                        -(
                            meanSpeed * 4.6 * cfg.motionSensitivity +
                                maxSpeed * 1.4 * cfg.motionSensitivity +
                                meanAcceleration * 0.28
                            ),
                    )
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val motionScore =
            (
                rawMotionScore *
                    (
                        1.0 -
                            context.cameraDominance * 0.52 -
                            context.rollingShutterRisk * 0.14
                        )
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val dynamicsConfidence =
            (
                meanConfidence *
                    (1.0 -
                        min(
                            1.0,
                            meanJerk /
                                40.0,
                        )
                    ) *
                    (1.0 - 0.55 * rejectedRatio) *
                    context.sensorConfidence *
                    (1.0 - 0.34 * context.cameraDominance)
                ).coerceIn(
                    0.0,
                    1.0,
                )

        val directionChangeRatio =
            (
                directionChanges.toDouble() /
                    quality.size.coerceAtLeast(1)
            ).coerceIn(0.0, 1.0)

        val binaryMicroMotionSignal =
            (
                microMotionSamples.toDouble() /
                    quality.size.coerceAtLeast(1)
            ).coerceIn(0.0, 1.0)
        val continuousMicroMotionSignal =
            (
                speeds.count {
                    it >= 0.0001 &&
                        it <= 0.50
                }.toDouble() /
                    speeds.size.coerceAtLeast(1)
            ).coerceIn(0.0, 1.0)

        val microMotionScore =
            max(
                binaryMicroMotionSignal,
                continuousMicroMotionSignal * 0.85,
            ) *
                meanConfidence *
                (1.0 - jitter)
            .coerceIn(0.0, 1.0)

        val meanMeasurementUncertaintyScore =
            measurementUncertainties.averageOrZero()
                .coerceIn(0.0, 1.0)

        val activeStates =
            states.values
                .filter { state ->
                    state.confidence >=
                        cfg.minimumConfidence * 0.75 &&
                        state.lastTimestampNanos == timestamp
                }

        val medianVelocityX =
            activeStates
                .map { it.vx }
                .medianOrZero()
        val medianVelocityY =
            activeStates
                .map { it.vy }
                .medianOrZero()

        val commonModeMotion =
            hypot(
                medianVelocityX,
                medianVelocityY,
            ).coerceAtLeast(0.0)

        val differentialMotion =
            activeStates
                .map {
                    hypot(
                        it.vx - medianVelocityX,
                        it.vy - medianVelocityY,
                    )
                }
                .averageOrZero()
                .coerceAtLeast(0.0)

        val differentialMotionScore =
            (
                1.0 -
                    exp(
                        -differentialMotion * 5.0,
                    )
                ).coerceIn(0.0, 1.0)

        val cameraCompensatedMotionScore =
            (
                rawMotionScore *
                    (
                        1.0 -
                            context.cameraDominance * 0.62 *
                                (
                                    1.0 -
                                        differentialMotionScore * 0.72
                                    )
                        )
                ).coerceIn(0.0, 1.0)

        motionSpectrumHistory.addLast(
            TemporalMotionSpectrumAnalyzer.Sample(
                timestampNanos = timestamp,
                magnitude = meanSpeed,
            ),
        )
        while (
            motionSpectrumHistory.size >
                96
        ) {
            motionSpectrumHistory.removeFirst()
        }

        val spectral =
            temporalMotionSpectrumAnalyzer.analyze(
                motionSpectrumHistory.toList(),
            )

        val spectralDynamicsConfidence =
            if (spectral.valid) {
                (
                    dynamicsConfidence *
                        (
                            0.90 +
                                0.10 *
                                    spectral.spectralConfidence
                            )
                    ).coerceIn(
                    0.0,
                    1.0,
                )
            } else {
                dynamicsConfidence
            }

        val qualityResult =
            Quality(
                timestampNanos = timestamp,
                motionScore = motionScore,
                motionClass =
                    classifyMotion(
                        score = motionScore,
                        maxSpeed = maxSpeed,
                        maxAcceleration =
                            maxAcceleration,
                        thresholdScale =
                            cfg.motionThresholdScale,
                    ),
                tuningResponsiveness = cfg.responsiveness,
                predictionHorizonSeconds =
                    cfg.predictionHorizonSeconds,
                reacquisitionRadiusNormalized =
                    cfg.reacquisitionRadius,
                meanSpeedNormalizedPerSecond =
                    meanSpeed,
                maxSpeedNormalizedPerSecond =
                    maxSpeed.coerceAtMost(
                        maxLandmarkSpeedNormalized,
                    ),
                meanAccelerationNormalizedPerSecond2 =
                    meanAcceleration,
                maxAccelerationNormalizedPerSecond2 =
                    maxAcceleration.coerceAtMost(
                        maxLandmarkAccelerationNormalized,
                    ),
                meanJerkNormalizedPerSecond3 =
                    meanJerk,
                jitterNormalized =
                    jitter,
                trackedLandmarks =
                    tracked,
                predictedLandmarks =
                    predictedCount,
                rejectedLandmarks =
                    rejected,
                meanConfidence =
                    meanConfidence,
                occlusionRatio =
                    occlusionRatio,
                temporalStability =
                    temporalStability,
                spatialSpread =
                    spatialSpread,
                stationaryRatio =
                    stationaryRatio,
                dynamicsConfidence =
                    spectralDynamicsConfidence,
                meanInnovationNormalized =
                    meanInnovation,
                rejectedMeasurementRatio =
                    rejectedRatio,
                directionChangeRatio =
                    directionChangeRatio,
                microMotionScore =
                    microMotionScore,
                meanMeasurementUncertaintyScore =
                    meanMeasurementUncertaintyScore,
                motionBlurRisk =
                    context.motionBlurRisk.coerceIn(0.0, 1.0),
                dominantFrequencyHz =
                    spectral.dominantFrequencyHz,
                periodicMotionScore =
                    spectral.periodicMotionScore,
                vibrationRisk =
                    spectral.vibrationRisk,
                spectralConfidence =
                    spectral.spectralConfidence,
                commonModeMotionNormalizedPerSecond =
                    commonModeMotion,
                differentialMotionScore =
                    differentialMotionScore,
                cameraCompensatedMotionScore =
                    cameraCompensatedMotionScore,
                stateCapacityEvictions = stateCapacityEvictions,
            )

        adaptivePolicy =
            motionRegimeAdaptiveController.blend(
                current =
                    motionRegimeAdaptiveController.identityPolicy(
                        reason = "adaptive baseline",
                    ),
                target =
                    motionRegimeAdaptiveController.resolve(
                        quality = qualityResult,
                        context = context,
                    ),
                alpha = cfg.adaptiveRegimeStrength,
            )

        enforceStateCapacity()
        lastTimestampNanos = timestamp

        val refined =
            Refined(
                result =
                    input.copy(
                    landmarks =
                        output
                            .distinctBy {
                                it.id
                            }
                            .sortedBy {
                                it.id
                            },
                    confidence =
                        meanConfidence
                            .toFloat()
                            .coerceIn(
                                0f,
                                1f,
                            ),
                ),
            quality = qualityResult,
                landmarkQuality =
                    quality.sortedBy {
                        it.id
                    },
            )
        lastRefined = refined
        return refined
    }

    private fun findReacquisitionState(
        measurement: PoseDetectionEngine.Landmark,
        width: Double,
        height: Double,
        timestampNanos: Long,
        candidateIds: Set<Int>,
    ): Pair<Int, State>? {
        val cfg = tuning
        val measuredX = (
            measurement.x.toDouble() / width
            ).coerceIn(0.0, 1.0)
        val measuredY = (
            measurement.y.toDouble() / height
            ).coerceIn(0.0, 1.0)

        var best: Pair<Int, State>? = null
        var bestScore = Double.POSITIVE_INFINITY

        states.forEach { (id, candidate) ->
            if (id == measurement.id || id !in candidateIds) return@forEach
            if (timestampNanos - candidate.lastObservedNanos >
                maximumStateAgeNanos
            ) {
                return@forEach
            }
            if (candidate.confidence < (
                reacquisitionMinimumConfidence *
                    (0.72 + 0.50 * cfg.rejectAggressiveness) *
                    adaptivePolicy.reacquisitionScale
                ).coerceIn(0.18, 0.92)
            ) {
                return@forEach
            }

            val ageSeconds =
                (
                    timestampNanos -
                        candidate.lastTimestampNanos
                    ).coerceAtLeast(0L) / 1e9
            val predictionDt =
                ageSeconds.coerceIn(
                    0.0,
                    (
                        maximumPredictionGapNanos.toDouble() *
                            adaptivePolicy.predictionHorizonScale
                    ) / 1e9,
                )
            val predictedX =
                (
                    candidate.x +
                        candidate.vx * predictionDt +
                        0.5 *
                        candidate.ax *
                        predictionDt *
                        predictionDt
                    ).coerceIn(0.0, 1.0)
            val predictedY =
                (
                    candidate.y +
                        candidate.vy * predictionDt +
                        0.5 *
                        candidate.ay *
                        predictionDt *
                        predictionDt
                    ).coerceIn(0.0, 1.0)

            val distance = hypot(
                measuredX - predictedX,
                measuredY - predictedY,
            )

            if (distance > (
                reacquisitionDistanceNormalized *
                    adaptivePolicy.reacquisitionScale *
                    (0.68 + 0.78 * cfg.reacquisitionRadius / 0.11)
                ).coerceIn(0.03, 0.22)
            ) {
                return@forEach
            }

            val namePenalty =
                if (
                    candidate.name.isBlank() ||
                    measurement.name.isBlank() ||
                    candidate.name.equals(
                        measurement.name,
                        ignoreCase = true,
                    )
                ) {
                    0.0
                } else {
                    0.035
                }

            val agePenalty =
                (ageSeconds * 0.018).coerceIn(0.0, 0.08)
            val residualPenalty =
                (candidate.residualEma * 0.12).coerceIn(0.0, 0.08)
            val confidenceBonus =
                (candidate.confidence * 0.025).coerceIn(0.0, 0.025)
            val score =
                distance +
                    namePenalty +
                    agePenalty +
                    residualPenalty -
                    confidenceBonus
            if (score < bestScore) {
                bestScore = score
                best = id to candidate
            }
        }

        return best
    }

    @Synchronized
    private fun enforceStateCapacity() {
        if (states.size <= maximumTrackedLandmarks) return
        val overflow = states.size - maximumTrackedLandmarks
        stateCapacityEvictions += overflow.toLong()
        states.entries
            .sortedWith(
                compareBy<MutableMap.MutableEntry<Int, State>> {
                    it.value.confidence
                }.thenBy {
                    it.value.lastObservedNanos
                }.thenBy {
                    it.key
                },
            )
            .take(overflow)
            .map { it.key }
            .forEach(states::remove)
    }

    @Synchronized
    private fun pruneExpiredStates(timestampNanos: Long) {
        states.entries.removeIf { (_, state) ->
            timestampNanos - state.lastObservedNanos >
                maximumStateAgeNanos
        }
    }

    @Synchronized
    fun reset() {
        states.clear()
        motionSpectrumHistory.clear()
        adaptivePolicy =
            motionRegimeAdaptiveController.identityPolicy("reset")
        lastTimestampNanos = 0L
        lastRefined = null
        stateCapacityEvictions = 0L
    }

    private fun sampleDeltaSeconds(
        timestampNanos: Long,
    ): Double {
        if (lastTimestampNanos == 0L) {
            return 1.0 / 30.0
        }

        return (
            (
                timestampNanos -
                    lastTimestampNanos
                ).coerceAtLeast(1L) /
                1e9
            ).coerceIn(
                1.0 / 240.0,
                0.25,
            )
    }

    private fun classifyMotion(
        score: Double,
        maxSpeed: Double,
        maxAcceleration: Double,
        thresholdScale: Double = 1.0,
    ): MotionClass {
        val scale = thresholdScale.coerceIn(0.70, 1.40)
        val staticThreshold = 0.05 * scale
        val microThreshold = 0.18 * scale
        val normalThreshold = 0.55 * scale
        val fastThreshold = 0.82 * scale
        val normalSpeed = 1.25 * scale
        val fastAcceleration = 8.0 * scale

        return when {
            score < staticThreshold ->
                MotionClass.STATIC

            score < microThreshold ->
                MotionClass.MICRO_MOTION

            score < normalThreshold &&
                maxSpeed < normalSpeed ->
                MotionClass.NORMAL

            score < fastThreshold &&
                maxAcceleration < fastAcceleration ->
                MotionClass.FAST

            else ->
                MotionClass.EXTREME
        }
    }

    private fun normalizedSpread(
        landmarks:
        List<PoseDetectionEngine.Landmark>,
        width: Double,
        height: Double,
    ): Double {
        if (landmarks.size < 2) {
            return 0.0
        }

        val centerX =
            landmarks
                .map { it.x / width }
                .average()

        val centerY =
            landmarks
                .map { it.y / height }
                .average()

        return landmarks
            .map {
                hypot(
                    it.x / width -
                        centerX,
                    it.y / height -
                        centerY,
                )
            }
            .average()
            .coerceIn(
                0.0,
                1.0,
            )
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun List<Double>.medianOrZero(): Double {
        val values =
            filter(Double::isFinite).sorted()
        if (values.isEmpty()) return 0.0
        val middle = values.size / 2
        return if (values.size % 2 == 0) {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        }
    }

    override fun close() {
        reset()
    }
}
