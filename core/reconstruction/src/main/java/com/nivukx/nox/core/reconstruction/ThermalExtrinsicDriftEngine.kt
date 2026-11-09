package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Thermal-conditioned extrinsic drift authority.
 *
 * Mobile cameras can change pose relative to a rig as mounts, adhesive, clamps, tripods and device
 * temperature change. This engine does not assume drift is present; it estimates it from repeated
 * pose observations and calibration anchors, tracks temperature correlation, and emits explicit
 * recalibration evidence.
 *
 * It is intentionally camera-local and bounded so a single unstable device cannot contaminate the
 * whole rig's state. A downstream rig-health engine can use the per-camera result to hold a camera
 * out of primary reconstruction while leaving other cameras active.
 */
class ThermalExtrinsicDriftEngine(
    private val config: Config = Config(),
) {
    enum class State {
        UNKNOWN,
        STABLE,
        WATCH,
        RECALIBRATE,
        BLOCKED,
    }

    data class Config(
        val historyPerCamera: Int = 64,
        val minimumSamples: Int = 6,
        val warningTranslationMm: Double = 0.35,
        val hardTranslationMm: Double = 1.50,
        val warningRotationDegrees: Double = 0.12,
        val hardRotationDegrees: Double = 0.60,
        val warningTemperatureDeltaCelsius: Double = 8.0,
        val hardTemperatureDeltaCelsius: Double = 20.0,
        val warningDriftRateMmPerCelsius: Double = 0.025,
        val hardDriftRateMmPerCelsius: Double = 0.080,
        val warningAngularRateDegPerCelsius: Double = 0.010,
        val hardAngularRateDegPerCelsius: Double = 0.030,
        val thermalPathTemperatureToleranceCelsius: Double = 1.5,
        val warningThermalPathHysteresisMm: Double = 0.25,
        val hardThermalPathHysteresisMm: Double = 1.00,
        val minimumCalibrationConfidence: Double = 0.82,
        val minimumTemperatureSpanForRateCelsius: Double = 4.0,
        val hysteresisAlpha: Double = 0.18,
        val stableThreshold: Double = 0.82,
        val watchThreshold: Double = 0.62,
        val recalibrateThreshold: Double = 0.40,
        val stateEnterFrames: Int = 2,
        val recoveryFrames: Int = 3,
    )

    data class Sample(
        val cameraId: DeviceId,
        val timestampNanos: Long,
        val temperatureCelsius: Double,
        val calibratedPose: Pose,
        val observedPose: Pose,
        val calibrationConfidence: Double,
        val mountIntegrity: Double = 1.0,
        val translationEvidenceAvailable: Boolean = true,
        val rotationEvidenceAvailable: Boolean = true,
    ) {
        init {
            require(timestampNanos >= 0L)
            require(temperatureCelsius.isFinite())
            require(calibrationConfidence in 0.0..1.0)
            require(mountIntegrity in 0.0..1.0)
        }
    }

    data class Result(
        val cameraId: DeviceId,
        val timestampNanos: Long,
        val state: State,
        val confidence: Double,
        val translationDriftMm: Double,
        val rotationDriftDegrees: Double,
        val temperatureDeltaCelsius: Double,
        val translationDriftRateMmPerCelsius: Double,
        val rotationDriftRateDegPerCelsius: Double,
        val temperatureSpanConfidence: Double,
        val thermalCorrelation: Double,
        val mountIntegrity: Double,
        val samples: Int,
        val thermalPathHysteresisMm: Double = 0.0,
        val thermalPathHysteresisDegrees: Double = 0.0,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class CameraState(
        val samples: ArrayDeque<Sample> = ArrayDeque(),
        var confidenceEma: Double = 1.0,
        var stableState: State = State.UNKNOWN,
        var degradingFrames: Int = 0,
        var recoveringFrames: Int = 0,
        var lastResult: Result? = null,
    )

    private val states = linkedMapOf<DeviceId, CameraState>()

    @Synchronized
    fun observe(sample: Sample): Result {
        val state = states.getOrPut(sample.cameraId) { CameraState() }
        if (state.samples.isNotEmpty() &&
            sample.timestampNanos < state.samples.last().timestampNanos
        ) {
            return state.lastResult ?: unknown(sample)
        }

        state.samples.addLast(sample)
        while (state.samples.size > config.historyPerCamera) {
            state.samples.removeFirst()
        }

        val rawResult =
            evaluate(sample.cameraId, state)

        state.confidenceEma =
            state.confidenceEma * (1.0 - config.hysteresisAlpha) +
                rawResult.confidence * config.hysteresisAlpha

        state.stableState =
            transitionState(
                current = state.stableState,
                proposed = rawResult.state,
                state = state,
                hardFailure =
                    rawResult.state == State.BLOCKED,
            )

        val stableResult =
            rawResult.copy(
                state = state.stableState,
                confidence =
                    state.confidenceEma.coerceIn(
                        0.0,
                        1.0,
                    ),
            )
        state.lastResult = stableResult
        return stableResult
    }

    @Synchronized
    fun evaluate(cameraId: DeviceId): Result =
        states[cameraId]?.let { state ->
            state.lastResult ?: evaluate(
                cameraId,
                state,
            )
        } ?: Result(
            cameraId = cameraId,
            timestampNanos = 0L,
            state = State.UNKNOWN,
            confidence = 0.0,
            translationDriftMm = 0.0,
            rotationDriftDegrees = 0.0,
            temperatureDeltaCelsius = 0.0,
            translationDriftRateMmPerCelsius = 0.0,
            rotationDriftRateDegPerCelsius = 0.0,
            thermalCorrelation = 0.0,
            mountIntegrity = 0.0,
            samples = 0,
            warnings = listOf("NO_THERMAL_DRIFT_HISTORY"),
            recommendations = listOf(
                "Collect repeated pose and temperature observations before judging thermal stability.",
            ),
        )

    @Synchronized
    fun all(): List<Result> =
        states.keys.sortedBy { it.value }.mapNotNull {
            states[it]?.lastResult
        }

    @Synchronized
    fun clear() {
        states.clear()
    }

    private fun transitionState(
        current: State,
        proposed: State,
        state: CameraState,
        hardFailure: Boolean,
    ): State {
        if (hardFailure) {
            state.degradingFrames = 0
            state.recoveringFrames = 0
            return State.BLOCKED
        }

        val currentRank =
            stateSeverity(current)
        val proposedRank =
            stateSeverity(proposed)

        return when {
            current == State.UNKNOWN -> {
                state.degradingFrames = 0
                state.recoveringFrames = 0
                proposed
            }

            proposedRank > currentRank -> {
                state.recoveringFrames = 0
                state.degradingFrames += 1
                if (
                    state.degradingFrames >=
                        config.stateEnterFrames
                ) {
                    state.degradingFrames = 0
                    proposed
                } else {
                    current
                }
            }

            proposedRank < currentRank -> {
                state.degradingFrames = 0
                state.recoveringFrames += 1
                if (
                    state.recoveringFrames >=
                        config.recoveryFrames
                ) {
                    state.recoveringFrames = 0
                    proposed
                } else {
                    current
                }
            }

            else -> {
                state.degradingFrames = 0
                state.recoveringFrames = 0
                current
            }
        }
    }

    private fun stateSeverity(
        state: State,
    ): Int =
        when (state) {
            State.UNKNOWN -> 0
            State.STABLE -> 1
            State.WATCH -> 2
            State.RECALIBRATE -> 3
            State.BLOCKED -> 4
        }

    private fun evaluate(
        cameraId: DeviceId,
        state: CameraState,
    ): Result {
        val samples = state.samples
        val latest = samples.lastOrNull()
            ?: return evaluate(cameraId)

        if (samples.size < config.minimumSamples) {
            return Result(
                cameraId = cameraId,
                timestampNanos = latest.timestampNanos,
                state = State.UNKNOWN,
                confidence = 0.35,
                translationDriftMm = driftTranslationMm(latest),
                rotationDriftDegrees = driftRotationDegrees(latest),
                temperatureDeltaCelsius = 0.0,
                translationDriftRateMmPerCelsius = 0.0,
                rotationDriftRateDegPerCelsius = 0.0,
                temperatureSpanConfidence = 0.0,
                thermalCorrelation = 0.0,
                mountIntegrity = latest.mountIntegrity,
                samples = samples.size,
                warnings = listOf("INSUFFICIENT_THERMAL_DRIFT_SAMPLES"),
                recommendations = listOf(
                    "Keep the camera mechanically fixed and collect a longer thermal history.",
                ),
            )
        }

        val latestTranslation = driftTranslationMm(latest)
        val latestRotation = driftRotationDegrees(latest)
        val referenceTemperature =
            samples.first().temperatureCelsius
        val temperatureDelta =
            latest.temperatureCelsius - referenceTemperature

        val temperatureSpan =
            (
                samples.maxOf { it.temperatureCelsius } -
                    samples.minOf { it.temperatureCelsius }
            ).coerceAtLeast(0.0)

        val temperatureSpanConfidence =
            (
                temperatureSpan /
                    config.minimumTemperatureSpanForRateCelsius
            ).coerceIn(0.0, 1.0)

        val translationRate =
            robustAbsoluteSlope(
                samples.map { sample ->
                    sample.temperatureCelsius to
                        driftTranslationMm(sample)
                },
            )

        val rotationRate =
            robustAbsoluteSlope(
                samples.map { sample ->
                    sample.temperatureCelsius to
                        driftRotationDegrees(sample)
                },
            )

        val effectiveTranslationRate =
            translationRate *
                temperatureSpanConfidence

        val effectiveRotationRate =
            rotationRate *
                temperatureSpanConfidence

        val thermalPathHysteresis =
            estimateThermalPathHysteresis(samples)

        val thermalCorrelation =
            signedCorrelation(
                samples.map { sample ->
                    sample.temperatureCelsius to
                        driftTranslationMm(sample)
                },
            )

        val thermalPathHysteresisScore =
            inverse(
                thermalPathHysteresis.translationMm,
                config.warningThermalPathHysteresisMm,
                config.hardThermalPathHysteresisMm,
            )

        val translationScore =
            inverse(
                latestTranslation,
                config.warningTranslationMm,
                config.hardTranslationMm,
            )
        val rotationScore =
            inverse(
                latestRotation,
                config.warningRotationDegrees,
                config.hardRotationDegrees,
            )
        val temperatureScore =
            inverse(
                abs(temperatureDelta),
                config.warningTemperatureDeltaCelsius,
                config.hardTemperatureDeltaCelsius,
            )
        val translationRateScore =
            inverse(
                effectiveTranslationRate,
                config.warningDriftRateMmPerCelsius,
                config.hardDriftRateMmPerCelsius,
            )
        val rotationRateScore =
            inverse(
                effectiveRotationRate,
                config.warningAngularRateDegPerCelsius,
                config.hardAngularRateDegPerCelsius,
            )
        val calibrationScore =
            samples.map { it.calibrationConfidence }.average()
        val mountScore =
            samples.map { it.mountIntegrity }.average()
        val translationEvidenceRatio =
            samples.count { it.translationEvidenceAvailable }.toDouble() /
                samples.size.toDouble().coerceAtLeast(1.0)
        val rotationEvidenceRatio =
            samples.count { it.rotationEvidenceAvailable }.toDouble() /
                samples.size.toDouble().coerceAtLeast(1.0)

        val confidence = (
            translationScore * 0.22 * translationEvidenceRatio +
                rotationScore * 0.20 * rotationEvidenceRatio +
                temperatureScore * 0.10 +
                translationRateScore * 0.18 * translationEvidenceRatio +
                rotationRateScore * 0.12 * rotationEvidenceRatio +
                thermalPathHysteresisScore * 0.06 +
                temperatureSpanConfidence * 0.05 * (
                    translationEvidenceRatio +
                        rotationEvidenceRatio
                ) * 0.5 +
                calibrationScore * 0.10 +
                mountScore * 0.08 +
                (
                    (translationEvidenceRatio + rotationEvidenceRatio) * 0.05
                )
            ).coerceIn(0.0, 1.0)

        val hardFailure =
            (
                latest.translationEvidenceAvailable &&
                    latestTranslation >= config.hardTranslationMm
                ) ||
                (
                    latest.rotationEvidenceAvailable &&
                        latestRotation >= config.hardRotationDegrees
                    ) ||
                (
                    latest.translationEvidenceAvailable &&
                        effectiveTranslationRate >=
                            config.hardDriftRateMmPerCelsius
                    ) ||
                (
                    latest.rotationEvidenceAvailable &&
                        effectiveRotationRate >=
                            config.hardAngularRateDegPerCelsius
                    ) ||
                thermalPathHysteresis.translationMm >=
                    config.hardThermalPathHysteresisMm ||
                abs(temperatureDelta) >= config.hardTemperatureDeltaCelsius ||
                calibrationScore < config.minimumCalibrationConfidence ||
                mountScore < 0.30

        val stateClass = when {
            hardFailure -> State.BLOCKED
            confidence < config.recalibrateThreshold -> State.RECALIBRATE
            confidence < config.watchThreshold -> State.WATCH
            confidence < config.stableThreshold -> State.WATCH
            else -> State.STABLE
        }

        val warnings = buildList {
            if (
                latest.translationEvidenceAvailable &&
                    latestTranslation > config.warningTranslationMm
            ) {
                add("EXTRINSIC_TRANSLATION_DRIFT_ELEVATED")
            } else if (!latest.translationEvidenceAvailable) {
                add("TRANSLATION_DRIFT_UNOBSERVED")
            }
            if (
                latest.rotationEvidenceAvailable &&
                    latestRotation > config.warningRotationDegrees
            ) {
                add("EXTRINSIC_ROTATION_DRIFT_ELEVATED")
            } else if (!latest.rotationEvidenceAvailable) {
                add("ROTATION_DRIFT_UNOBSERVED")
            }
            if (translationEvidenceRatio < 0.75) {
                add("TRANSLATION_DRIFT_EVIDENCE_INCOMPLETE")
            }
            if (rotationEvidenceRatio < 0.75) {
                add("ROTATION_DRIFT_EVIDENCE_INCOMPLETE")
            }
            if (abs(temperatureDelta) > config.warningTemperatureDeltaCelsius) {
                add("THERMAL_DELTA_ELEVATED")
            }
            if (
                latest.translationEvidenceAvailable &&
                    effectiveTranslationRate > config.warningDriftRateMmPerCelsius
            ) {
                add("THERMAL_TRANSLATION_COEFFICIENT_ELEVATED")
            }
            if (
                latest.rotationEvidenceAvailable &&
                    effectiveRotationRate > config.warningAngularRateDegPerCelsius
            ) {
                add("THERMAL_ROTATION_COEFFICIENT_ELEVATED")
            }
            if (calibrationScore < config.minimumCalibrationConfidence) {
                add("CALIBRATION_CONFIDENCE_BELOW_THERMAL_AUTHORITY_FLOOR")
            }
            if (mountScore < 0.70) {
                add("MOUNT_INTEGRITY_DEGRADED")
            }
            if (abs(thermalCorrelation) > 0.65) {
                add("THERMAL_DRIFT_CORRELATION_STRONG")
            }
            if (
                thermalPathHysteresis.translationMm >
                    config.warningThermalPathHysteresisMm
            ) {
                add("THERMAL_PATH_HYSTERESIS_ELEVATED")
            }
        }.distinct()

        val recommendations = buildList {
            if (stateClass == State.RECALIBRATE || stateClass == State.BLOCKED) {
                add("Re-run multi-view extrinsic calibration before reference-grade capture.")
            }
            if (
                latest.translationEvidenceAvailable &&
                    translationRate > config.warningDriftRateMmPerCelsius
            ) {
                add("Allow the camera to thermally settle or compensate only from measured rig-specific drift.")
            }
            if (latestRotation > config.warningRotationDegrees) {
                add("Inspect mechanical mounting and lens/device seating for movement.")
            }
            if (abs(temperatureDelta) > config.warningTemperatureDeltaCelsius) {
                add("Record the calibration temperature and repeat the reference pose after thermal stabilization.")
            }
            if (mountScore < 0.70) {
                add("Mechanically secure the camera before trusting prior extrinsics.")
            }
            if (
                thermalPathHysteresis.translationMm >
                    config.warningThermalPathHysteresisMm
            ) {
                add("Repeat calibration across heating and cooling paths; do not trust a single-temperature extrinsic fit.")
            }
        }.distinct()

        return Result(
            cameraId = cameraId,
            timestampNanos = latest.timestampNanos,
            state = stateClass,
            confidence = confidence,
            translationDriftMm = latestTranslation,
            rotationDriftDegrees = latestRotation,
            temperatureDeltaCelsius = temperatureDelta,
            translationDriftRateMmPerCelsius = effectiveTranslationRate,
            rotationDriftRateDegPerCelsius = effectiveRotationRate,
            temperatureSpanConfidence = temperatureSpanConfidence,
            thermalCorrelation = thermalCorrelation,
            thermalPathHysteresisMm =
                thermalPathHysteresis.translationMm,
            thermalPathHysteresisDegrees =
                thermalPathHysteresis.rotationDegrees,
            mountIntegrity = mountScore,
            samples = samples.size,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private data class ThermalPathHysteresis(
        val translationMm: Double,
        val rotationDegrees: Double,
    )

    private fun estimateThermalPathHysteresis(
        samples: Collection<Sample>,
    ): ThermalPathHysteresis {
        if (samples.size < 4) {
            return ThermalPathHysteresis(
                translationMm = 0.0,
                rotationDegrees = 0.0,
            )
        }

        val ordered = samples.toList()
        val direction = IntArray(ordered.size)

        for (index in 1 until ordered.size) {
            val delta =
                ordered[index].temperatureCelsius -
                    ordered[index - 1].temperatureCelsius
            direction[index] =
                when {
                    delta > 0.05 -> 1
                    delta < -0.05 -> -1
                    else -> 0
                }
        }

        var maxTranslation = 0.0
        var maxRotation = 0.0

        for (i in ordered.indices) {
            if (direction[i] == 0) continue
            for (j in i + 1 until ordered.size) {
                if (
                    direction[j] == 0 ||
                    direction[j] == direction[i]
                ) continue
                if (
                    abs(
                        ordered[i].temperatureCelsius -
                            ordered[j].temperatureCelsius,
                    ) >
                        config.thermalPathTemperatureToleranceCelsius
                ) continue

                maxTranslation =
                    max(
                        maxTranslation,
                        abs(
                            driftTranslationMm(ordered[i]) -
                                driftTranslationMm(ordered[j]),
                        ),
                    )
                maxRotation =
                    max(
                        maxRotation,
                        abs(
                            driftRotationDegrees(ordered[i]) -
                                driftRotationDegrees(ordered[j]),
                        ),
                    )
            }
        }

        return ThermalPathHysteresis(
            translationMm = maxTranslation,
            rotationDegrees = maxRotation,
        )
    }

    private fun driftTranslationMm(sample: Sample): Double =
        sample.calibratedPose.position
            .distanceTo(sample.observedPose.position) * 1000.0

    private fun driftRotationDegrees(sample: Sample): Double {
        val a = sample.calibratedPose.rotation.normalized()
        val b = sample.observedPose.rotation.normalized()
        val dot = (
            a.w * b.w +
                a.x * b.x +
                a.y * b.y +
                a.z * b.z
            ).coerceIn(-1.0, 1.0)
        return Math.toDegrees(
            2.0 * acos(abs(dot).coerceIn(0.0, 1.0)),
        )
    }

    private fun robustAbsoluteSlope(
        pairs: List<Pair<Double, Double>>,
    ): Double {
        if (pairs.size < 2) return 0.0

        val slopes =
            pairs.indices.flatMap { i ->
                (i + 1 until pairs.size).mapNotNull { j ->
                    val dx =
                        pairs[j].first -
                            pairs[i].first
                    val dy =
                        pairs[j].second -
                            pairs[i].second
                    if (abs(dx) <= 1e-9) {
                        null
                    } else {
                        dy / dx
                    }
                }
            }

        if (slopes.isEmpty()) return 0.0

        return abs(
            median(
                slopes,
            ),
        )
    }

    private fun median(
        values: List<Double>,
    ): Double {
        if (values.isEmpty()) return 0.0
        val sorted =
            values.sorted()
        val middle =
            sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (
                sorted[middle - 1] +
                    sorted[middle]
            ) * 0.5
        } else {
            sorted[middle]
        }
    }

    private fun signedCorrelation(
        pairs: List<Pair<Double, Double>>,
    ): Double {
        if (pairs.size < 3) return 0.0
        val meanX = pairs.map { it.first }.average()
        val meanY = pairs.map { it.second }.average()
        var xy = 0.0
        var xx = 0.0
        var yy = 0.0
        pairs.forEach { (x, y) ->
            val dx = x - meanX
            val dy = y - meanY
            xy += dx * dy
            xx += dx * dx
            yy += dy * dy
        }
        if (xx <= 1e-12 || yy <= 1e-12) return 0.0
        return (xy / kotlin.math.sqrt(xx * yy)).coerceIn(-1.0, 1.0)
    }

    private fun inverse(
        value: Double,
        warning: Double,
        hard: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= warning) return 1.0
        if (value >= hard) return 0.0
        return (
            1.0 -
                (value - warning) /
                    (hard - warning).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun unknown(sample: Sample): Result =
        Result(
            cameraId = sample.cameraId,
            timestampNanos = sample.timestampNanos,
            state = State.UNKNOWN,
            confidence = 0.0,
            translationDriftMm = 0.0,
            rotationDriftDegrees = 0.0,
            temperatureDeltaCelsius = 0.0,
            translationDriftRateMmPerCelsius = 0.0,
            rotationDriftRateDegPerCelsius = 0.0,
            thermalCorrelation = 0.0,
            thermalPathHysteresisMm = 0.0,
            thermalPathHysteresisDegrees = 0.0,
            mountIntegrity = sample.mountIntegrity,
            samples = 0,
            warnings = listOf("OUT_OF_ORDER_THERMAL_SAMPLE"),
            recommendations = listOf(
                "Preserve monotonic capture timestamps per camera.",
            ),
        )
}
