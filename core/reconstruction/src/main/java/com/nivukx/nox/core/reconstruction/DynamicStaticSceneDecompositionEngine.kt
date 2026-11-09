package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.acos
import kotlin.math.max

/**
 * Soft dynamic/static scene decomposition for post-capture reconstruction.
 *
 * The goal is to prevent moving subjects from contaminating camera-graph refinement and
 * static-scene TSDF integration. The classifier keeps a temporal state per sample and exposes
 * probabilities instead of a brittle binary mask.
 */
class DynamicStaticSceneDecompositionEngine(
    private var config: Config = Config(),
) {
    data class Config(
        val maximumSamples: Int = 10000,
        val staticResidualMeters: Double = 0.018,
        val dynamicResidualMeters: Double = 0.095,
        val staticVelocityMetersPerSecond: Double = 0.035,
        val dynamicVelocityMetersPerSecond: Double = 0.22,
        val cameraMotionCompensation: Double = 0.88,
        val temporalAlpha: Double = 0.22,
        val confidenceAlpha: Double = 0.18,
        val hysteresisMargin: Double = 0.07,
        val staleSeconds: Double = 0.50,
        val dynamicPersistence: Int = 3,
        val staticPersistence: Int = 4,
        val maximumHistory: Int = 10000,
        val staticNormalChangeDegrees: Double = 4.0,
        val dynamicNormalChangeDegrees: Double = 25.0,
    ) {
        init {
            require(maximumSamples >= 1)
            require(staticResidualMeters > 0.0)
            require(dynamicResidualMeters > staticResidualMeters)
            require(staticVelocityMetersPerSecond >= 0.0)
            require(dynamicVelocityMetersPerSecond > staticVelocityMetersPerSecond)
            require(cameraMotionCompensation in 0.0..1.0)
            require(temporalAlpha in 0.01..1.0)
            require(confidenceAlpha in 0.01..1.0)
            require(hysteresisMargin >= 0.0)
            require(staleSeconds > 0.0)
            require(dynamicPersistence >= 1)
            require(staticPersistence >= 1)
            require(maximumHistory >= 1)
            require(staticNormalChangeDegrees >= 0.0)
            require(dynamicNormalChangeDegrees > staticNormalChangeDegrees)
        }
    }

    data class Sample(
        val id: Long,
        val position: Vec3,
        val confidence: Double = 1.0,
        val timestampNanos: Long,
        val normal: Vec3? = null,
    ) {
        init {
            require(id >= 0L)
            require(confidence in 0.0..1.0)
            require(timestampNanos > 0L)
        }
    }

    enum class Label { UNKNOWN, STATIC, DYNAMIC }

    data class SampleState(
        val id: Long,
        val staticProbability: Double,
        val dynamicProbability: Double,
        val residualMeters: Double,
        val velocityMetersPerSecond: Double,
        val motionCoherence: Double,
        val normalChangeDegrees: Double = 0.0,
        val confidence: Double,
        val label: Label,
        val staticStreak: Int,
        val dynamicStreak: Int,
        val stale: Boolean,
    )

    data class Result(
        val timestampNanos: Long,
        val staticCoverage: Double,
        val dynamicCoverage: Double,
        val unknownCoverage: Double,
        val staticConfidence: Double,
        val dynamicConfidence: Double,
        val decompositionConfidence: Double,
        val meanNormalChangeDegrees: Double = 0.0,
        val states: List<SampleState>,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class HistorySample(
        var timestampNanos: Long,
        var position: Vec3,
        var staticProbability: Double,
        var dynamicProbability: Double,
        var staticStreak: Int,
        var dynamicStreak: Int,
        var normal: Vec3?,
    )

    private val history = LinkedHashMap<Long, HistorySample>()
    private var lastTimestampNanos = Long.MIN_VALUE

    @Synchronized
    fun configure(value: Config) { config = value }

    fun configuration(): Config = config

    @Synchronized
    fun reset() {
        history.clear()
        lastTimestampNanos = Long.MIN_VALUE
    }

    @Synchronized
    fun evaluate(
        timestampNanos: Long,
        samples: List<Sample>,
        cameraMotionMetersPerSecond: Vec3 = Vec3(0.0, 0.0, 0.0),
    ): Result {
        require(timestampNanos > 0L)

        if (lastTimestampNanos != Long.MIN_VALUE && timestampNanos <= lastTimestampNanos) {
            reset()
        }

        val dtSeconds = if (lastTimestampNanos == Long.MIN_VALUE) {
            1.0 / 30.0
        } else {
            (timestampNanos - lastTimestampNanos).coerceAtLeast(1L) / 1e9
        }

        val bounded = samples
            .asSequence()
            .filter {
                it.position.x.isFinite() &&
                    it.position.y.isFinite() &&
                    it.position.z.isFinite()
            }
            .sortedByDescending { it.confidence }
            .take(config.maximumSamples)
            .toList()

        val states = ArrayList<SampleState>(bounded.size)
        val cameraDisplacement = scale(
            cameraMotionMetersPerSecond,
            config.cameraMotionCompensation * dtSeconds,
        )

        for (sample in bounded) {
            val previous = history[sample.id]

            val predictedPosition = previous?.let {
                add(it.position, cameraDisplacement)
            }

            val rawResidual = previous?.let {
                sample.position.distanceTo(it.position)
            } ?: 0.0

            val compensatedResidual = previous?.let {
                sample.position.distanceTo(predictedPosition!!)
            } ?: 0.0

            val velocity = if (previous == null) {
                0.0
            } else {
                compensatedResidual / dtSeconds.coerceAtLeast(1e-4)
            }

            val velocityScore = ramp(
                velocity,
                config.staticVelocityMetersPerSecond,
                config.dynamicVelocityMetersPerSecond,
            )

            val residualScore = ramp(
                compensatedResidual,
                config.staticResidualMeters,
                config.dynamicResidualMeters,
            )

            val normalChangeDegrees =
                if (
                    previous?.normal != null &&
                        sample.normal != null &&
                        previous.normal!!.norm() > 1e-6 &&
                        sample.normal.norm() > 1e-6
                ) {
                    Math.toDegrees(
                        acos(
                            previous.normal!!
                                .normalized()
                                .dot(
                                    sample.normal
                                        .normalized(),
                                )
                                .coerceIn(-1.0, 1.0),
                        ),
                    ).coerceAtLeast(0.0)
                } else {
                    0.0
                }

            val normalChangeScore =
                ramp(
                    normalChangeDegrees,
                    config.staticNormalChangeDegrees,
                    config.dynamicNormalChangeDegrees,
                )

            val motionScore =
                if (
                    previous?.normal != null &&
                        sample.normal != null
                ) {
                    (
                        velocityScore * 0.46 +
                            residualScore * 0.34 +
                            normalChangeScore * 0.20
                        ).coerceIn(0.0, 1.0)
                } else {
                    (
                        velocityScore * 0.56 +
                            residualScore * 0.44
                        ).coerceIn(0.0, 1.0)
                }

            val confidencePenalty = (
                1.0 - sample.confidence
            ).coerceIn(0.0, 1.0)

            val dynamicMeasurement = (
                motionScore * (1.0 - config.confidenceAlpha) +
                    confidencePenalty * config.confidenceAlpha
            ).coerceIn(0.0, 1.0)

            val instantaneousDynamicEvidence = maxOf(
                motionScore,
                velocityScore,
                residualScore,
                normalChangeScore,
            ).coerceIn(0.0, 1.0)

            val smoothedDynamicProbability = if (previous == null) {
                dynamicMeasurement
            } else {
                previous.dynamicProbability * (1.0 - config.temporalAlpha) +
                    dynamicMeasurement * config.temporalAlpha
            }

            val dynamicProbability =
                if (instantaneousDynamicEvidence >= 0.90) {
                    maxOf(smoothedDynamicProbability, 0.68)
                } else {
                    smoothedDynamicProbability
                }

            val staticProbability = (1.0 - dynamicProbability).coerceIn(0.0, 1.0)

            val stateHistory = if (previous == null) {
                HistorySample(
                    timestampNanos = timestampNanos,
                    position = sample.position,
                    staticProbability = staticProbability,
                    dynamicProbability = dynamicProbability,
                    staticStreak = if (staticProbability >= 0.62) 1 else 0,
                    dynamicStreak = if (dynamicProbability >= 0.62) 1 else 0,
                    normal = sample.normal,
                ).also { history[sample.id] = it }
            } else {
                previous.position = sample.position
                previous.staticProbability = staticProbability
                previous.dynamicProbability = dynamicProbability
                previous.normal = sample.normal

                if (dynamicProbability >= 0.62) {
                    previous.dynamicStreak += 1
                    previous.staticStreak = max(0, previous.staticStreak - 1)
                } else if (staticProbability >= 0.62) {
                    previous.staticStreak += 1
                    previous.dynamicStreak = max(0, previous.dynamicStreak - 1)
                } else {
                    previous.staticStreak = max(0, previous.staticStreak - 1)
                    previous.dynamicStreak = max(0, previous.dynamicStreak - 1)
                }
                previous
            }

            val stale =
                timestampNanos - stateHistory.timestampNanos >
                    (config.staleSeconds * 1e9).toLong()

            val label = when {
                stateHistory.dynamicStreak >= config.dynamicPersistence -> Label.DYNAMIC
                stateHistory.staticStreak >= config.staticPersistence -> Label.STATIC
                kotlin.math.abs(dynamicProbability - staticProbability) <
                    config.hysteresisMargin -> Label.UNKNOWN
                dynamicProbability >= 0.58 -> Label.DYNAMIC
                staticProbability >= 0.58 -> Label.STATIC
                else -> Label.UNKNOWN
            }

            stateHistory.timestampNanos = timestampNanos

            states += SampleState(
                id = sample.id,
                staticProbability = staticProbability,
                dynamicProbability = dynamicProbability,
                residualMeters = rawResidual,
                velocityMetersPerSecond = velocity,
                motionCoherence = (1.0 - motionScore).coerceIn(0.0, 1.0),
                normalChangeDegrees = normalChangeDegrees,
                confidence = sample.confidence,
                label = label,
                staticStreak = stateHistory.staticStreak,
                dynamicStreak = stateHistory.dynamicStreak,
                stale = stale,
            )
        }

        trimHistory()
        lastTimestampNanos = timestampNanos

        val static = states.filter { it.label == Label.STATIC }
        val dynamic = states.filter { it.label == Label.DYNAMIC }
        val unknown = states.filter { it.label == Label.UNKNOWN }

        val staticCoverage = static.size.toDouble() / states.size.coerceAtLeast(1)
        val dynamicCoverage = dynamic.size.toDouble() / states.size.coerceAtLeast(1)
        val unknownCoverage = unknown.size.toDouble() / states.size.coerceAtLeast(1)

        val staticConfidence = static.map {
            it.staticProbability * it.confidence
        }.averageOrZero()

        val dynamicConfidence = dynamic.map {
            it.dynamicProbability * it.confidence
        }.averageOrZero()

        val decompositionConfidence = (
            (1.0 - unknownCoverage) * 0.34 +
                staticConfidence * 0.33 +
                dynamicConfidence * 0.33
        ).coerceIn(0.0, 1.0)

        val meanNormalChangeDegrees =
            if (states.isEmpty()) {
                0.0
            } else {
                states.map {
                    it.normalChangeDegrees
                }.average()
            }

        val warnings = buildList {
            if (unknownCoverage > 0.38) add("dynamic-static-classification-ambiguous")
            if (dynamicCoverage > 0.72) add("dynamic-geometry-dominates-observation")
            if (staticCoverage < 0.20) add("insufficient-static-support-for-camera-refinement")
            if (meanNormalChangeDegrees >= config.dynamicNormalChangeDegrees) {
                add("surface-normal-change-dominates-observation")
            }
            if (cameraMotionMetersPerSecond.norm() >
                config.dynamicVelocityMetersPerSecond * 2.0
            ) {
                add("camera-motion-compensation-near-limit")
            }
        }.distinct()

        val recommendations = buildList {
            if (unknownCoverage > 0.38) {
                add("Preserve ambiguous samples for joint optimization instead of forcing a hard static mask.")
            }
            if (staticCoverage < 0.20) {
                add("Delay static-scene bundle refinement until stable background observations accumulate.")
            }
            if (dynamicCoverage > 0.72) {
                add("Downweight dynamic human regions during camera graph refinement and TSDF reintegration.")
            }
            if (decompositionConfidence >= 0.80) {
                add("Use soft static probabilities as weights for downstream scene refinement.")
            }
        }.distinct()

        return Result(
            timestampNanos = timestampNanos,
            staticCoverage = staticCoverage,
            dynamicCoverage = dynamicCoverage,
            unknownCoverage = unknownCoverage,
            staticConfidence = staticConfidence,
            dynamicConfidence = dynamicConfidence,
            decompositionConfidence = decompositionConfidence,
            meanNormalChangeDegrees = meanNormalChangeDegrees,
            states = states,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun trimHistory() {
        while (history.size > config.maximumHistory) {
            val oldest = history.entries.minByOrNull { it.value.timestampNanos } ?: break
            history.remove(oldest.key)
        }
    }

    private fun ramp(value: Double, low: Double, high: Double): Double {
        if (!value.isFinite()) return 1.0
        if (value <= low) return 0.0
        if (value >= high) return 1.0
        return ((value - low) / (high - low).coerceAtLeast(1e-9)).coerceIn(0.0, 1.0)
    }

    private fun add(a: Vec3, b: Vec3): Vec3 =
        Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

    private fun scale(a: Vec3, scalar: Double): Vec3 =
        Vec3(a.x * scalar, a.y * scalar, a.z * scalar)

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
