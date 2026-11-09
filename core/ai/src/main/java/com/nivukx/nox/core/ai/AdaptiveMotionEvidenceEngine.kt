package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Multi-scale motion evidence engine for markerless capture.
 *
 * It estimates motion without assuming that detector displacement equals subject motion.
 * Common-mode motion, rolling-shutter risk, micro-motion, acceleration and temporal freshness are
 * separated so downstream reconstruction can decide how much to trust the observation.
 */
class AdaptiveMotionEvidenceEngine(
    private val maximumLandmarkStates: Int = 4096,
    private val maximumHistorySamples: Int = 128,
    private val maximumStateAgeNanos: Long = 1_000_000_000L,
) : AutoCloseable {

    data class Landmark(
        val id: Int,
        val x: Double,
        val y: Double,
        val confidence: Double,
    ) {
        init {
            require(id >= 0)
            require(x.isFinite() && y.isFinite())
            require(confidence.isFinite() && confidence in 0.0..1.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val landmarks: List<Landmark>,
        val cameraMotionRisk: Double = 0.0,
        val rollingShutterRisk: Double = 0.0,
        val captureTimingConfidence: Double = 1.0,
        val occlusionRatio: Double = 0.0,
        val identityContinuity: Double = 1.0,
        val nominalFrameRateHz: Double = 30.0,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(cameraMotionRisk in 0.0..1.0)
            require(rollingShutterRisk in 0.0..1.0)
            require(captureTimingConfidence in 0.0..1.0)
            require(occlusionRatio in 0.0..1.0)
            require(identityContinuity in 0.0..1.0)
            require(nominalFrameRateHz.isFinite() && nominalFrameRateHz in 1.0..240.0)
        }
    }

    enum class MotionClass { STATIC, MICRO, NORMAL, FAST, EXTREME }

    data class Evidence(
        val timestampNanos: Long,
        val motionClass: MotionClass,
        val globalEnergy: Double,
        val differentialEnergy: Double,
        val commonModeEnergy: Double,
        val meanSpeedPerSecond: Double,
        val maxSpeedPerSecond: Double,
        val meanAccelerationPerSecond2: Double,
        val maxAccelerationPerSecond2: Double,
        val microMotionScore: Double,
        val jitterScore: Double,
        val vibrationRisk: Double,
        val cameraMotionRisk: Double,
        val rollingShutterRisk: Double,
        val occlusionRisk: Double,
        val identityContinuity: Double,
        val temporalConfidence: Double,
        val frequencyProxyHz: Double,
        val periodicMotionScore: Double,
        val trackedLandmarks: Int,
        val predictedRisk: Double,
        val confidence: Double,
        val warnings: List<String>,
    ) {
        val healthy: Boolean
            get() = confidence >= 0.50 && temporalConfidence >= 0.40
    }

    private data class State(
        var timestampNanos: Long = Long.MIN_VALUE,
        var x: Double = 0.0,
        var y: Double = 0.0,
        var vx: Double = 0.0,
        var vy: Double = 0.0,
        var ax: Double = 0.0,
        var ay: Double = 0.0,
        var confidence: Double = 0.0,
    )

    private val states = LinkedHashMap<Int, State>()
    private val energyHistory = ArrayDeque<Double>()
    private val frequencyHistory = ArrayDeque<Double>()
    private var closed = false
    @Volatile private var latest: Evidence? = null

    @Synchronized
    fun process(input: Input): Evidence {
        check(!closed)
        if (input.timestampNanos < (latest?.timestampNanos ?: Long.MIN_VALUE)) {
            return latest ?: emptyEvidence(input.timestampNanos, listOf("timestamp-reversal"))
        }

        val observations = input.landmarks
            .filter { it.confidence >= 0.05 }
            .take(4096)

        val velocities = ArrayList<Double>(observations.size)
        val accelerations = ArrayList<Double>(observations.size)
        val displacements = ArrayList<Double>(observations.size)
        val currentVx = ArrayList<Double>(observations.size)
        val currentVy = ArrayList<Double>(observations.size)

        for (landmark in observations) {
            val state = states.getOrPut(landmark.id) { State() }
            val dt = if (state.timestampNanos == Long.MIN_VALUE) {
                0.0
            } else {
                ((input.timestampNanos - state.timestampNanos).coerceAtLeast(1L) / 1e9)
                    .coerceIn(1e-4, 0.5)
            }
            if (dt > 0.0) {
                val dx = landmark.x - state.x
                val dy = landmark.y - state.y
                val speedX = dx / dt
                val speedY = dy / dt
                val speed = hypot(speedX, speedY)
                val accelerationX = (speedX - state.vx) / dt
                val accelerationY = (speedY - state.vy) / dt
                val acceleration = hypot(accelerationX, accelerationY)

                velocities += speed * landmark.confidence
                accelerations += acceleration * landmark.confidence
                displacements += hypot(dx, dy) * landmark.confidence
                currentVx += speedX
                currentVy += speedY

                state.vx = 0.65 * state.vx + 0.35 * speedX
                state.vy = 0.65 * state.vy + 0.35 * speedY
                state.ax = 0.60 * state.ax + 0.40 * accelerationX
                state.ay = 0.60 * state.ay + 0.40 * accelerationY
            }
            state.x = landmark.x
            state.y = landmark.y
            state.confidence = landmark.confidence
            state.timestampNanos = input.timestampNanos
        }

        pruneStates(input.timestampNanos)

        val velocity = weightedMean(velocities)
        val acceleration = weightedMean(accelerations)
        val displacement = weightedMean(displacements)
        val commonVx = median(currentVx)
        val commonVy = median(currentVy)
        val commonMode = hypot(commonVx, commonVy)
        val differential = currentVx.indices.map { index ->
            hypot(currentVx[index] - commonVx, currentVy[index] - commonVy)
        }.averageOrZero()

        val globalEnergy = normalized(
            0.55 * velocity / 2.5 +
                0.30 * acceleration / 18.0 +
                0.15 * displacement / 0.04,
        )
        val differentialEnergy = normalized(
            0.68 * differential / 2.5 +
                0.32 * acceleration / 18.0,
        )
        val commonModeEnergy = normalized(commonMode / 2.5)
        val microMotionScore = normalized(
            (velocity / 0.30) * 0.65 +
                (displacement / 0.012) * 0.35,
        )
        val jitterScore = (
            abs(acceleration - velocity * 6.0) / 30.0
        ).coerceIn(0.0, 1.0)

        energyHistory.addLast(globalEnergy)
        frequencyHistory.addLast(differentialEnergy)
        trimHistory()

        val frequencyProxy = estimateFrequencyHz(input.nominalFrameRateHz)
        val periodicity = periodicityScore()
        val temporalConfidence = (
            input.captureTimingConfidence * 0.30 +
                (1.0 - input.rollingShutterRisk) * 0.12 +
                input.identityContinuity * 0.20 +
                (1.0 - input.occlusionRatio) * 0.20 +
                min(1.0, observations.size / 16.0) * 0.18
            ).coerceIn(0.0, 1.0)

        val vibrationRisk = (
            jitterScore * 0.48 +
                periodicity * 0.22 +
                input.rollingShutterRisk * 0.18 +
                commonModeEnergy * 0.12
            ).coerceIn(0.0, 1.0)

        val predictedRisk = (
            input.occlusionRatio * 0.45 +
                input.rollingShutterRisk * 0.20 +
                (1.0 - input.captureTimingConfidence) * 0.15 +
                (1.0 - input.identityContinuity) * 0.20
            ).coerceIn(0.0, 1.0)

        val differentialTrust = (
            differentialEnergy * 0.52 +
                microMotionScore * 0.18 +
                (1.0 - input.cameraMotionRisk) * 0.16 +
                (1.0 - vibrationRisk) * 0.14
            ).coerceIn(0.0, 1.0)

        val confidence = (
            temporalConfidence * 0.34 +
                differentialTrust * 0.32 +
                (1.0 - predictedRisk) * 0.16 +
                (1.0 - input.cameraMotionRisk) * 0.10 +
                (1.0 - input.rollingShutterRisk) * 0.08
            ).coerceIn(0.0, 1.0)

        val motionClass = when {
            differentialEnergy >= 0.82 || acceleration >= 28.0 -> MotionClass.EXTREME
            differentialEnergy >= 0.55 || velocity >= 2.0 -> MotionClass.FAST
            differentialEnergy >= 0.22 || velocity >= 0.65 -> MotionClass.NORMAL
            microMotionScore >= 0.20 -> MotionClass.MICRO
            else -> MotionClass.STATIC
        }

        val warnings = buildList {
            if (input.cameraMotionRisk >= 0.60) add("camera-motion-dominant")
            if (input.rollingShutterRisk >= 0.55) add("rolling-shutter-risk")
            if (input.occlusionRatio >= 0.50) add("heavy-occlusion")
            if (input.captureTimingConfidence < 0.60) add("timing-confidence-low")
            if (vibrationRisk >= 0.65) add("vibration-risk-high")
            if (commonModeEnergy > differentialEnergy + 0.15) add("common-mode-dominant")
            if (observations.size < 4) add("sparse-landmark-support")
        }

        return Evidence(
            timestampNanos = input.timestampNanos,
            motionClass = motionClass,
            globalEnergy = globalEnergy,
            differentialEnergy = differentialEnergy,
            commonModeEnergy = commonModeEnergy,
            meanSpeedPerSecond = velocity,
            maxSpeedPerSecond = velocities.maxOrNull() ?: 0.0,
            meanAccelerationPerSecond2 = acceleration,
            maxAccelerationPerSecond2 = accelerations.maxOrNull() ?: 0.0,
            microMotionScore = microMotionScore,
            jitterScore = jitterScore,
            vibrationRisk = vibrationRisk,
            cameraMotionRisk = input.cameraMotionRisk,
            rollingShutterRisk = input.rollingShutterRisk,
            occlusionRisk = input.occlusionRatio,
            identityContinuity = input.identityContinuity,
            temporalConfidence = temporalConfidence,
            frequencyProxyHz = frequencyProxy,
            periodicMotionScore = periodicity,
            trackedLandmarks = observations.size,
            predictedRisk = predictedRisk,
            confidence = confidence,
            warnings = warnings.distinct(),
        ).also { latest = it }
    }

    fun latest(): Evidence? = latest

    @Synchronized
    fun reset() {
        states.clear()
        energyHistory.clear()
        frequencyHistory.clear()
        latest = null
    }

    override fun close() {
        reset()
        closed = true
    }

    private fun commonModeEstimate(
        landmarks: List<Landmark>,
        timestampNanos: Long,
    ): Double {
        if (landmarks.isEmpty()) return 0.0
        val speeds = ArrayList<Double>(landmarks.size)
        for (landmark in landmarks) {
            states[landmark.id]?.let { state ->
                if (state.timestampNanos == timestampNanos) {
                    speeds += hypot(state.vx, state.vy) * landmark.confidence
                }
            }
        }
        return weightedMean(speeds)
    }

    private fun estimateFrequencyHz(frameRateHz: Double): Double {
        if (energyHistory.size < 12) return 0.0
        val values = energyHistory.toList()
        var crossings = 0
        var previous = values.first() - 0.5
        for (value in values.drop(1)) {
            val current = value - 0.5
            if ((previous < 0.0 && current >= 0.0) ||
                (previous > 0.0 && current <= 0.0)
            ) {
                crossings++
            }
            previous = current
        }
        return (crossings.toDouble() / max(1, values.size - 1)) * frameRateHz * 0.5
    }

    private fun periodicityScore(): Double {
        if (frequencyHistory.size < 8) return 0.0
        val values = frequencyHistory.toList()
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        val stability = 1.0 - sqrt(variance).coerceIn(0.0, 1.0)
        return (stability * 0.70 + min(1.0, mean * 1.8) * 0.30)
            .coerceIn(0.0, 1.0)
    }

    private fun pruneStates(timestampNanos: Long) {
        val iterator = states.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (timestampNanos - entry.value.timestampNanos > maximumStateAgeNanos) {
                iterator.remove()
            }
        }
        while (states.size > maximumLandmarkStates) {
            states.entries.firstOrNull()?.let { states.remove(it.key) } ?: break
        }
    }

    private fun trimHistory() {
        while (energyHistory.size > maximumHistorySamples) energyHistory.removeFirst()
        while (frequencyHistory.size > maximumHistorySamples) frequencyHistory.removeFirst()
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private fun weightedMean(values: List<Double>): Double =
        if (values.isEmpty()) 0.0 else values.average()

    private fun normalized(value: Double): Double =
        value.coerceAtLeast(0.0).coerceAtMost(1.0)

    private fun emptyEvidence(
        timestampNanos: Long,
        warnings: List<String>,
    ) = Evidence(
        timestampNanos = timestampNanos,
        motionClass = MotionClass.STATIC,
        globalEnergy = 0.0,
        differentialEnergy = 0.0,
        commonModeEnergy = 0.0,
        meanSpeedPerSecond = 0.0,
        maxSpeedPerSecond = 0.0,
        meanAccelerationPerSecond2 = 0.0,
        maxAccelerationPerSecond2 = 0.0,
        microMotionScore = 0.0,
        jitterScore = 0.0,
        vibrationRisk = 0.0,
        cameraMotionRisk = 1.0,
        rollingShutterRisk = 1.0,
        occlusionRisk = 1.0,
        identityContinuity = 0.0,
        temporalConfidence = 0.0,
        frequencyProxyHz = 0.0,
        periodicMotionScore = 0.0,
        trackedLandmarks = 0,
        predictedRisk = 1.0,
        confidence = 0.0,
        warnings = warnings,
    )
}
