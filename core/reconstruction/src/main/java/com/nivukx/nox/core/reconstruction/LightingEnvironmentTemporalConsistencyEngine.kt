package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.LightingEstimate
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.ln

/**
 * Validates camera-derived lighting continuity without claiming absolute photometric accuracy.
 *
 * The base lighting estimator is an instantaneous relative signal. This layer adds temporal
 * hysteresis for exposure flicker, abrupt color-temperature changes, light-direction changes,
 * confidence collapse, and recovery after unstable lighting.
 */
class LightingEnvironmentTemporalConsistencyEngine(
    private val config: Config = Config(),
) {
    enum class State { UNKNOWN, STABLE, TRANSITION, FLICKER, DEGRADED }

    data class Config(
        val maximumGapSeconds: Double = 0.60,
        val illuminanceLogWarning: Double = 0.24,
        val illuminanceLogReject: Double = 0.62,
        val cctRelativeWarning: Double = 0.12,
        val cctRelativeReject: Double = 0.30,
        val directionWarningDegrees: Double = 24.0,
        val directionRejectDegrees: Double = 55.0,
        val minimumConfidence: Double = 0.28,
        val warningScore: Double = 0.68,
        val degradedScore: Double = 0.48,
        val historyAlpha: Double = 0.16,
        val persistentFlickerFrames: Int = 4,
        val persistentDegradedFrames: Int = 5,
        val recoveryFrames: Int = 5,
    ) {
        init {
            require(maximumGapSeconds > 0.0)
            require(illuminanceLogReject > illuminanceLogWarning)
            require(cctRelativeReject > cctRelativeWarning)
            require(directionRejectDegrees > directionWarningDegrees)
            require(minimumConfidence in 0.0..1.0)
            require(degradedScore <= warningScore)
            require(historyAlpha in 0.0..1.0)
            require(persistentFlickerFrames >= 1)
            require(persistentDegradedFrames >= 1)
            require(recoveryFrames >= 1)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val confidence: Double,
        val continuityScore: Double,
        val illuminanceChangeRatio: Double,
        val colorTemperatureChangeRatio: Double,
        val directionChangeDegrees: Double,
        val persistentFlickerFrames: Int,
        val persistentDegradedFrames: Int,
        val recoveryFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class History(
        var timestampNanos: Long,
        var estimate: LightingEstimate,
        var continuityEma: Double,
        var confidenceEma: Double,
        var flickerFrames: Int,
        var degradedFrames: Int,
        var recoveryFrames: Int,
        var lastState: State,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(estimate: LightingEstimate): Result {
        val confidence = estimate.confidence.coerceIn(0.0, 1.0)
        val previous = history

        if (previous == null) {
            val state =
                if (confidence < config.minimumConfidence) {
                    State.DEGRADED
                } else {
                    State.UNKNOWN
                }
            history = History(
                timestampNanos = estimate.timestampNanos,
                estimate = estimate,
                continuityEma = confidence,
                confidenceEma = confidence,
                flickerFrames = 0,
                degradedFrames = if (state == State.DEGRADED) 1 else 0,
                recoveryFrames = 0,
                lastState = state,
            )
            return result(
                estimate = estimate,
                state = state,
                confidence = confidence,
                continuity = confidence,
                illuminationRatio = 1.0,
                cctRatio = 1.0,
                directionDelta = 0.0,
            )
        }

        val dt = (estimate.timestampNanos - previous.timestampNanos)
            .coerceAtLeast(1L) / 1e9
        if (dt > config.maximumGapSeconds) {
            history = History(
                timestampNanos = estimate.timestampNanos,
                estimate = estimate,
                continuityEma = confidence * 0.90,
                confidenceEma = confidence * 0.90,
                flickerFrames = 0,
                degradedFrames = if (confidence < config.minimumConfidence) 1 else 0,
                recoveryFrames = 0,
                lastState = if (confidence < config.minimumConfidence) {
                    State.DEGRADED
                } else {
                    State.UNKNOWN
                },
            )
            return result(
                estimate = estimate,
                state = history!!.lastState,
                confidence = confidence * 0.90,
                continuity = confidence * 0.90,
                illuminationRatio = 1.0,
                cctRatio = 1.0,
                directionDelta = 0.0,
                extraWarnings = listOf("lighting-temporal-history-reset-after-gap"),
            )
        }

        val illuminationRatio = ratio(
            estimate.illuminanceLux.coerceAtLeast(0.01),
            previous.estimate.illuminanceLux.coerceAtLeast(0.01),
        )
        val illuminationDelta = abs(ln(illuminationRatio.coerceIn(0.05, 20.0)))
        val cctRatio = ratio(
            estimate.colorTemperatureKelvin.coerceAtLeast(1800.0),
            previous.estimate.colorTemperatureKelvin.coerceAtLeast(1800.0),
        )
        val cctDelta = abs(cctRatio - 1.0)
        val directionDelta = directionDegrees(
            previous.estimate.directionalLight,
            estimate.directionalLight,
        )

        val illuminationPenalty = penalty(
            illuminationDelta,
            config.illuminanceLogWarning,
            config.illuminanceLogReject,
        )
        val cctPenalty = penalty(
            cctDelta,
            config.cctRelativeWarning,
            config.cctRelativeReject,
        )
        val directionPenalty = penalty(
            directionDelta,
            config.directionWarningDegrees,
            config.directionRejectDegrees,
        )

        val continuity = (
            (1.0 - illuminationPenalty) * 0.38 +
                (1.0 - cctPenalty) * 0.18 +
                (1.0 - directionPenalty) * 0.18 +
                previous.estimate.confidence * 0.12 +
                confidence * 0.14
        ).coerceIn(0.0, 1.0)

        val flicker = illuminationDelta >= config.illuminanceLogWarning &&
            previous.estimate.confidence >= config.minimumConfidence &&
            confidence >= config.minimumConfidence
        val degraded = continuity < config.degradedScore ||
            confidence < config.minimumConfidence

        previous.flickerFrames = if (flicker) {
            previous.flickerFrames + 1
        } else {
            max(0, previous.flickerFrames - 1)
        }
        previous.degradedFrames = if (degraded) {
            previous.degradedFrames + 1
        } else {
            max(0, previous.degradedFrames - 1)
        }
        previous.recoveryFrames = if (degraded) {
            0
        } else {
            previous.recoveryFrames + 1
        }

        val alpha = config.historyAlpha
        previous.continuityEma =
            previous.continuityEma * (1.0 - alpha) + continuity * alpha
        previous.confidenceEma =
            previous.confidenceEma * (1.0 - alpha) + confidence * alpha
        previous.timestampNanos = estimate.timestampNanos
        previous.estimate = estimate

        val transition =
            illuminationPenalty > 0.0 ||
                cctPenalty > 0.0 ||
                directionPenalty > 0.0

        val state = when {
            previous.degradedFrames >= config.persistentDegradedFrames ->
                State.DEGRADED
            previous.flickerFrames >= config.persistentFlickerFrames ->
                State.FLICKER
            previous.lastState == State.FLICKER ||
                previous.lastState == State.DEGRADED ->
                if (
                    previous.recoveryFrames >= config.recoveryFrames &&
                    previous.continuityEma >= config.warningScore
                ) {
                    State.STABLE
                } else {
                    previous.lastState
                }
            transition -> State.TRANSITION
            previous.continuityEma >= config.warningScore -> State.STABLE
            else -> State.TRANSITION
        }
        previous.lastState = state

        return result(
            estimate = estimate,
            state = state,
            confidence = previous.confidenceEma,
            continuity = previous.continuityEma,
            illuminationRatio = illuminationRatio,
            cctRatio = cctRatio,
            directionDelta = directionDelta,
            extraWarnings = buildList {
                if (illuminationDelta >= config.illuminanceLogWarning) {
                    add("illuminance-changed-abruptly")
                }
                if (cctDelta >= config.cctRelativeWarning) {
                    add("color-temperature-changed-abruptly")
                }
                if (directionDelta >= config.directionWarningDegrees) {
                    add("dominant-light-direction-changed-abruptly")
                }
                if (confidence < config.minimumConfidence) {
                    add("lighting-estimate-confidence-below-production-floor")
                }
            },
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    private fun result(
        estimate: LightingEstimate,
        state: State,
        confidence: Double,
        continuity: Double,
        illuminationRatio: Double,
        cctRatio: Double,
        directionDelta: Double,
        extraWarnings: List<String> = emptyList(),
    ): Result {
        val warnings = buildList {
            when (state) {
                State.UNKNOWN -> add("lighting-temporal-history-initializing")
                State.STABLE -> Unit
                State.TRANSITION ->
                    add("lighting-environment-undergoing-measured-transition")
                State.FLICKER -> {
                    add("persistent-lighting-flicker-detected")
                    add("lighting-confidence-reduced-for-temporal-reconstruction")
                }
                State.DEGRADED -> {
                    add("lighting-temporal-quality-degraded")
                    add("lighting-derived-reconstruction-evidence-should-be-held")
                }
            }
            addAll(extraWarnings)
        }.distinct()

        val recommendations = buildList {
            if (state == State.FLICKER) {
                add("stabilize scene illumination or exposure before trusting relighting changes")
            }
            if (abs(ln(illuminationRatio.coerceIn(0.05, 20.0))) >=
                config.illuminanceLogWarning
            ) {
                add("check auto-exposure and avoid rapid exposure compensation changes")
            }
            if (abs(cctRatio - 1.0) >= config.cctRelativeWarning) {
                add("verify mixed-light sources before treating color changes as physical relighting")
            }
            if (directionDelta >= config.directionWarningDegrees) {
                add("verify camera motion and shadow motion before accepting a new dominant-light direction")
            }
            if (confidence < config.minimumConfidence) {
                add("hold lighting influence and preserve the last trusted lighting state")
            }
        }.distinct()

        return Result(
            timestampNanos = estimate.timestampNanos,
            state = state,
            confidence = confidence.coerceIn(0.0, 1.0),
            continuityScore = continuity.coerceIn(0.0, 1.0),
            illuminanceChangeRatio = illuminationRatio,
            colorTemperatureChangeRatio = cctRatio,
            directionChangeDegrees = directionDelta,
            persistentFlickerFrames = history?.flickerFrames ?: 0,
            persistentDegradedFrames = history?.degradedFrames ?: 0,
            recoveryFrames = history?.recoveryFrames ?: 0,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun penalty(value: Double, warning: Double, reject: Double): Double {
        if (!value.isFinite()) return 1.0
        if (value <= warning) return 0.0
        if (value >= reject) return 1.0
        return ((value - warning) / (reject - warning)).coerceIn(0.0, 1.0)
    }

    private fun ratio(current: Double, previous: Double): Double =
        if (!current.isFinite() || !previous.isFinite() || previous <= 1e-9) {
            1.0
        } else {
            (current / previous).coerceIn(0.05, 20.0)
        }

    private fun directionDegrees(
        a: com.nivukx.nox.core.model.Vec3,
        b: com.nivukx.nox.core.model.Vec3,
    ): Double {
        val na = a.normalized()
        val nb = b.normalized()
        if (na.norm() < 1e-6 || nb.norm() < 1e-6) return 90.0
        return Math.toDegrees(acos(na.dot(nb).coerceIn(-1.0, 1.0)))
            .coerceIn(0.0, 180.0)
    }
}