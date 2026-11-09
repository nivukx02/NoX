package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Production runtime governor for the multimodal perception graph.
 *
 * This engine does not perform detection itself. It makes the capture pipeline decide how much
 * compute to spend on the current frame based on subject motion, camera motion, tracking quality,
 * thermal pressure and measured inference latency.
 *
 * Design goals:
 * - preserve temporal continuity before maximizing detector frequency;
 * - avoid oscillating between quality levels;
 * - protect latency under thermal/memory pressure;
 * - spend additional compute on fast or uncertain motion;
 * - degrade gracefully instead of abruptly disabling perception stages.
 */
class ProductionPerceptionRuntimeGovernor(
    private val targetFrameLatencyMillis: Double = 33.333,
    private val hardLatencyMillis: Double = 52.0,
    private val minimumQualityScale: Double = 0.34,
    private val recoveryMarginMillis: Double = 5.0,
    private val decisionHysteresis: Double = 0.08,
) {

    enum class RuntimeBand {
        ULTRA,
        HIGH,
        BALANCED,
        CONSERVATIVE,
        PROTECTIVE,
    }

    data class Decision(
        val timestampNanos: Long,
        val band: RuntimeBand,
        val qualityScale: Double,
        val inferenceScale: Double,
        val posePriority: Double,
        val facePriority: Double,
        val eyePriority: Double,
        val handPriority: Double,
        val objectPriority: Double,
        val contourPriority: Double,
        val expectedFrameBudgetMillis: Double,
        val emergencyThermalProtection: Boolean,
        val cameraMotionSuppression: Double,
        val motionAmplification: Double,
        val temporalHoldSeconds: Double,
        val confidence: Double,
        val reasons: List<String>,
    )

    data class Telemetry(
        val timestampNanos: Long,
        val measuredLatencyMillis: Double,
        val latencyEwmaMillis: Double,
        val systemLoad: Double,
        val motionPressure: Double,
        val cameraMotionPressure: Double,
        val thermalPressure: Double,
        val qualityScale: Double,
        val band: RuntimeBand,
        val decisionCount: Long,
        val overloadCount: Long,
        val protectionCount: Long,
    )

    private var latencyEwmaMillis = targetFrameLatencyMillis
    private var systemLoad = 0.0
    private var thermalPressure = 0.0
    private var memoryPressure = 0.0
    private var gpuPressure = 0.0
    private var inferencePressure = 0.0
    private var lastDecision: Decision? = null
    private var lastTimestampNanos = Long.MIN_VALUE
    private var decisionCount = 0L
    private var overloadCount = 0L
    private var protectionCount = 0L

    fun updateSystemLoad(
        thermalScale: Double,
        memoryScale: Double,
        gpuScale: Double,
        inferenceScale: Double,
    ) {
        thermalPressure = pressureFromScale(thermalScale)
        memoryPressure = pressureFromScale(memoryScale)
        gpuPressure = pressureFromScale(gpuScale)
        inferencePressure = pressureFromScale(inferenceScale)

        systemLoad = (
            0.38 * thermalPressure +
                0.20 * memoryPressure +
                0.24 * gpuPressure +
                0.18 * inferencePressure
            ).coerceIn(0.0, 1.0)
    }

    fun recordInferenceDuration(durationMillis: Double) {
        if (!durationMillis.isFinite() || durationMillis < 0.0) return
        val measured = durationMillis.coerceIn(0.0, 2500.0)
        latencyEwmaMillis = blend(
            current = latencyEwmaMillis,
            measured = measured,
            alpha = if (measured > hardLatencyMillis) 0.30 else 0.12,
        )

        if (measured > hardLatencyMillis) {
            overloadCount += 1L
        }
    }

    @Synchronized
    fun decide(
        timestampNanos: Long,
        motion: SemanticMotionEstimator.GlobalMotion,
        trackedConfidence: Double = motion.confidence,
    ): Decision {
        require(timestampNanos >= 0L) {
            "timestampNanos must be non-negative"
        }

        val motionPressure = (
            0.48 * motion.globalEnergy +
                0.24 * motion.relativeMotionEnergy +
                0.18 * normalized(motion.maxSpeedNormalizedPerSecond, 4.5) +
                0.10 * normalized(motion.maxAccelerationNormalizedPerSecond2, 24.0)
            ).coerceIn(0.0, 1.0)

        val cameraMotionPressure = motion.cameraMotionRisk.coerceIn(0.0, 1.0)
        val latencyPressure = normalized(
            max(0.0, latencyEwmaMillis - targetFrameLatencyMillis),
            max(
                1.0,
                hardLatencyMillis - targetFrameLatencyMillis,
            ),
        )

        val recovery = if (
            latencyEwmaMillis <
                targetFrameLatencyMillis - recoveryMarginMillis
        ) {
            0.12
        } else {
            0.0
        }

        val thermalEmergency = thermalPressure >= 0.82 ||
            memoryPressure >= 0.92 ||
            latencyEwmaMillis >= hardLatencyMillis * 1.55

        val desiredQuality = (
            1.0 +
                0.46 * motionPressure +
                0.18 * motion.relativeMotionEnergy +
                0.12 * trackedConfidence -
                0.52 * systemLoad -
                0.42 * latencyPressure +
                recovery
            ).coerceIn(minimumQualityScale, 1.55)

        val band = chooseBand(
            desiredQuality = desiredQuality,
            motionPressure = motionPressure,
            cameraMotionPressure = cameraMotionPressure,
            thermalEmergency = thermalEmergency,
        )

        val bandScale = when (band) {
            RuntimeBand.ULTRA -> 1.0
            RuntimeBand.HIGH -> 0.90
            RuntimeBand.BALANCED -> 0.76
            RuntimeBand.CONSERVATIVE -> 0.60
            RuntimeBand.PROTECTIVE -> 0.42
        }

        val blendedQualityScale = (
            0.60 * desiredQuality +
                0.40 * bandScale
            ).coerceIn(minimumQualityScale, 1.10)
        val qualityScale =
            if (band == RuntimeBand.PROTECTIVE) {
                min(0.72, blendedQualityScale)
            } else {
                blendedQualityScale
            }

        val latencyPenalty = (
            1.0 - latencyPressure * 0.38
        ).coerceIn(0.50, 1.0)

        val motionBoost = (
            1.0 +
                motionPressure * 0.34 +
                motion.globalEnergy * 0.16
            ).coerceIn(1.0, 1.42)

        val cameraSuppression = (
            cameraMotionPressure * 0.72
        ).coerceIn(0.0, 0.72)

        val posePriority = stagePriority(
            base = 1.0,
            qualityScale = qualityScale,
            motionBoost = motionBoost,
            latencyPenalty = latencyPenalty,
            pressure = systemLoad,
        )

        val facePriority = stagePriority(
            base = 0.90,
            qualityScale = qualityScale,
            motionBoost = motionBoost,
            latencyPenalty = latencyPenalty,
            pressure = systemLoad,
        )

        val eyePriority = stagePriority(
            base = 0.98,
            qualityScale = qualityScale,
            motionBoost = 1.0 + motionPressure * 0.44,
            latencyPenalty = latencyPenalty,
            pressure = systemLoad,
        )

        val handPriority = stagePriority(
            base = 0.74,
            qualityScale = qualityScale,
            motionBoost = 1.0 + motionPressure * 0.58,
            latencyPenalty = latencyPenalty,
            pressure = systemLoad,
        )

        val objectPriority = stagePriority(
            base = 0.86,
            qualityScale = qualityScale,
            motionBoost = 1.0 + motionPressure * 0.52,
            latencyPenalty = latencyPenalty,
            pressure = systemLoad,
        )

        val contourPriority = stagePriority(
            base = 0.52,
            qualityScale = qualityScale,
            motionBoost = 1.0 + motionPressure * 0.25,
            latencyPenalty = latencyPenalty,
            pressure = systemLoad,
        )

        val expectedBudget = (
            targetFrameLatencyMillis /
                qualityScale.coerceAtLeast(0.1)
        ).coerceIn(
            targetFrameLatencyMillis,
            hardLatencyMillis * 1.15,
        )

        val holdSeconds = (
            0.10 +
                0.12 * cameraMotionPressure +
                0.15 * systemLoad +
                0.10 * latencyPressure
            ).coerceIn(0.08, 0.42)

        val confidence = (
            0.34 * motion.confidence +
                0.30 * (1.0 - systemLoad) +
                0.22 * latencyPenalty +
                0.14 * trackedConfidence
            ).coerceIn(0.0, 1.0)

        val reasons = buildReasons(
            motionPressure = motionPressure,
            cameraMotionPressure = cameraMotionPressure,
            latencyPressure = latencyPressure,
            thermalEmergency = thermalEmergency,
            systemLoad = systemLoad,
        )

        if (thermalEmergency) {
            protectionCount += 1L
        }

        decisionCount += 1L

        val candidate = Decision(
            timestampNanos = timestampNanos,
            band = band,
            qualityScale = qualityScale,
            inferenceScale = (
                qualityScale * latencyPenalty
            ).coerceIn(minimumQualityScale, 1.10),
            posePriority = posePriority,
            facePriority = facePriority,
            eyePriority = eyePriority,
            handPriority = handPriority,
            objectPriority = objectPriority,
            contourPriority = contourPriority,
            expectedFrameBudgetMillis = expectedBudget,
            emergencyThermalProtection = thermalEmergency,
            cameraMotionSuppression = cameraSuppression,
            motionAmplification = motionBoost,
            temporalHoldSeconds = holdSeconds,
            confidence = confidence,
            reasons = reasons,
        )

        val stabilized = stabilize(candidate)
        lastDecision = stabilized
        lastTimestampNanos = timestampNanos
        return stabilized
    }

    fun currentDecision(): Decision =
        lastDecision ?: Decision(
            timestampNanos = 0L,
            band = RuntimeBand.HIGH,
            qualityScale = 0.88,
            inferenceScale = 0.88,
            posePriority = 0.96,
            facePriority = 0.84,
            eyePriority = 0.92,
            handPriority = 0.72,
            objectPriority = 0.84,
            contourPriority = 0.48,
            expectedFrameBudgetMillis = targetFrameLatencyMillis,
            emergencyThermalProtection = false,
            cameraMotionSuppression = 0.0,
            motionAmplification = 1.0,
            temporalHoldSeconds = 0.10,
            confidence = 0.65,
            reasons = listOf("initial-runtime-policy"),
        )

    fun telemetry(): Telemetry {
        val decision = currentDecision()
        return Telemetry(
            timestampNanos = lastTimestampNanos.coerceAtLeast(0L),
            measuredLatencyMillis = latencyEwmaMillis,
            latencyEwmaMillis = latencyEwmaMillis,
            systemLoad = systemLoad,
            motionPressure = decision.motionAmplification
                .minus(1.0)
                .coerceIn(0.0, 1.0),
            cameraMotionPressure = decision.cameraMotionSuppression,
            thermalPressure = thermalPressure,
            qualityScale = decision.qualityScale,
            band = decision.band,
            decisionCount = decisionCount,
            overloadCount = overloadCount,
            protectionCount = protectionCount,
        )
    }

    @Synchronized
    fun reset() {
        latencyEwmaMillis = targetFrameLatencyMillis
        systemLoad = 0.0
        thermalPressure = 0.0
        memoryPressure = 0.0
        gpuPressure = 0.0
        inferencePressure = 0.0
        lastDecision = null
        lastTimestampNanos = Long.MIN_VALUE
        decisionCount = 0L
        overloadCount = 0L
        protectionCount = 0L
    }

    private fun chooseBand(
        desiredQuality: Double,
        motionPressure: Double,
        cameraMotionPressure: Double,
        thermalEmergency: Boolean,
    ): RuntimeBand {
        if (thermalEmergency) return RuntimeBand.PROTECTIVE

        val previous = lastDecision?.band
        val desired = when {
            desiredQuality >= 0.96 &&
                motionPressure >= 0.35 &&
                systemLoad < 0.28 ->
                RuntimeBand.ULTRA
            desiredQuality >= 0.80 ->
                RuntimeBand.HIGH
            desiredQuality >= 0.62 ->
                RuntimeBand.BALANCED
            desiredQuality >= 0.46 ->
                RuntimeBand.CONSERVATIVE
            else ->
                RuntimeBand.PROTECTIVE
        }

        if (previous == null) return desired

        val previousScale = bandScale(previous)
        val desiredScale = bandScale(desired)
        val delta = abs(previousScale - desiredScale)

        if (delta < decisionHysteresis) {
            return previous
        }

        if (
            desiredScale > previousScale &&
            (
                systemLoad > 0.46 ||
                    latencyEwmaMillis >
                        targetFrameLatencyMillis + recoveryMarginMillis
            )
        ) {
            return previous
        }

        if (
            desiredScale < previousScale &&
            cameraMotionPressure > 0.82 &&
            motionPressure > 0.58
        ) {
            // Preserve additional compute during difficult motion unless thermal protection is
            // genuinely required. Sudden downshifts during fast movement are a classic way to
            // manufacture tracker failure and then blame "AI".
            return previous
        }

        return desired
    }

    private fun stabilize(candidate: Decision): Decision {
        val previous = lastDecision ?: return candidate
        if (candidate.emergencyThermalProtection) return candidate

        val smoothing = if (
            candidate.band == previous.band
        ) {
            0.36
        } else {
            0.72
        }

        return candidate.copy(
            qualityScale = blend(
                previous.qualityScale,
                candidate.qualityScale,
                smoothing,
            ),
            inferenceScale = blend(
                previous.inferenceScale,
                candidate.inferenceScale,
                smoothing,
            ),
            posePriority = blend(
                previous.posePriority,
                candidate.posePriority,
                smoothing,
            ),
            facePriority = blend(
                previous.facePriority,
                candidate.facePriority,
                smoothing,
            ),
            eyePriority = blend(
                previous.eyePriority,
                candidate.eyePriority,
                smoothing,
            ),
            handPriority = blend(
                previous.handPriority,
                candidate.handPriority,
                smoothing,
            ),
            objectPriority = blend(
                previous.objectPriority,
                candidate.objectPriority,
                smoothing,
            ),
            contourPriority = blend(
                previous.contourPriority,
                candidate.contourPriority,
                smoothing,
            ),
        )
    }

    private fun stagePriority(
        base: Double,
        qualityScale: Double,
        motionBoost: Double,
        latencyPenalty: Double,
        pressure: Double,
    ): Double =
        (
            base *
                qualityScale *
                motionBoost *
                latencyPenalty *
                (1.0 - pressure * 0.30)
            ).coerceIn(0.18, 1.20)

    private fun buildReasons(
        motionPressure: Double,
        cameraMotionPressure: Double,
        latencyPressure: Double,
        thermalEmergency: Boolean,
        systemLoad: Double,
    ): List<String> {
        val reasons = ArrayList<String>()

        if (motionPressure >= 0.62) {
            reasons += "high-subject-motion"
        } else if (motionPressure >= 0.30) {
            reasons += "moderate-subject-motion"
        }

        if (cameraMotionPressure >= 0.62) {
            reasons += "camera-motion-compensation"
        }

        if (latencyPressure >= 0.50) {
            reasons += "latency-pressure"
        }

        if (systemLoad >= 0.60) {
            reasons += "system-load"
        }

        if (thermalEmergency) {
            reasons += "thermal-protection"
        }

        return reasons
    }

    private fun pressureFromScale(scale: Double): Double =
        (1.0 - scale.coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)

    private fun normalized(value: Double, scale: Double): Double =
        if (!value.isFinite()) {
            1.0
        } else {
            (
                value.coerceAtLeast(0.0) /
                    scale.coerceAtLeast(1e-6)
                ).coerceIn(0.0, 1.0)
        }

    private fun bandScale(band: RuntimeBand): Double =
        when (band) {
            RuntimeBand.ULTRA -> 1.0
            RuntimeBand.HIGH -> 0.88
            RuntimeBand.BALANCED -> 0.73
            RuntimeBand.CONSERVATIVE -> 0.58
            RuntimeBand.PROTECTIVE -> 0.42
        }

    private fun blend(
        current: Double,
        measured: Double,
        alpha: Double,
    ): Double {
        val a = alpha.coerceIn(0.0, 1.0)
        return current * (1.0 - a) + measured * a
    }
}
