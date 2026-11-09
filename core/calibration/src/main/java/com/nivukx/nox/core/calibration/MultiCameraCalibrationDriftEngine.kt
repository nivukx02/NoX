package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Pose
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Long-session multi-camera calibration drift monitor.
 *
 * Calibration is a temporal contract rather than a one-time checkbox. This engine tracks camera
 * translation, orientation, reprojection residuals, thermal stress, vibration risk and pairwise
 * baseline changes after a rig has been accepted. It never mutates the baseline automatically.
 * Instead it exposes a bounded trust score and explicit hold/recalibration decisions.
 */
class MultiCameraCalibrationDriftEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    data class Config(
        val translationWarningMeters: Double = 0.006,
        val translationHardMeters: Double = 0.025,
        val rotationWarningDegrees: Double = 0.35,
        val rotationHardDegrees: Double = 1.50,
        val reprojectionWarningPx: Double = 1.5,
        val reprojectionHardPx: Double = 4.0,
        val baselineDriftWarningRatio: Double = 0.010,
        val baselineDriftHardRatio: Double = 0.040,
        val minimumTrust: Double = 0.12,
        val preferredTrust: Double = 0.90,
        val historyLength: Int = 48,
        val smoothingAlpha: Double = 0.18,
        val recoveryAlpha: Double = 0.055,
        val thermalPenalty: Double = 0.18,
        val vibrationPenalty: Double = 0.20,
        val temporalPenalty: Double = 0.24,
        val minimumTemporalAlignmentConfidence: Double = 0.70,
        val maximumSubframeOffsetMillis: Double = 6.0,
        val reprojectionPenalty: Double = 0.28,
        val geometryPenalty: Double = 0.34,
        val recoverySamples: Int = 12,
        val minimumHardFailureConfidence: Double = 0.65,
        val persistentHardFailureSamples: Int = 3,
    )

    companion object {
        /**
         * Qualification envelope for metrology-oriented capture.
         *
         * These thresholds are intentionally stricter than normal production capture. Passing this
         * profile is still not a claim of sub-millimeter world-space accuracy; it only means the
         * measured calibration drift envelope is tight enough to enter an external metrology
         * qualification protocol.
         */
        fun metrologyQualification(): Config =
            Config(
                translationWarningMeters = 0.00075,
                translationHardMeters = 0.0030,
                rotationWarningDegrees = 0.05,
                rotationHardDegrees = 0.25,
                reprojectionWarningPx = 0.50,
                reprojectionHardPx = 1.50,
                baselineDriftWarningRatio = 0.0010,
                baselineDriftHardRatio = 0.0050,
                minimumTrust = 0.08,
                preferredTrust = 0.96,
                historyLength = 96,
                smoothingAlpha = 0.12,
                recoveryAlpha = 0.035,
                thermalPenalty = 0.24,
                vibrationPenalty = 0.24,
                temporalPenalty = 0.30,
                minimumTemporalAlignmentConfidence = 0.90,
                maximumSubframeOffsetMillis = 0.50,
                reprojectionPenalty = 0.34,
                geometryPenalty = 0.42,
                recoverySamples = 24,
            )
    }

    data class BaselineCamera(
        val deviceId: DeviceId,
        val pose: Pose,
    ) {
        init {
            require(deviceId.value.isNotBlank())
            require(pose.position.x.isFinite())
            require(pose.position.y.isFinite())
            require(pose.position.z.isFinite())
            require(pose.rotation.x.isFinite())
            require(pose.rotation.y.isFinite())
            require(pose.rotation.z.isFinite())
            require(pose.rotation.w.isFinite())
        }
    }

    data class CurrentCamera(
        val deviceId: DeviceId,
        val pose: Pose,
        val reprojectionErrorPx: Double = 0.0,
        val confidence: Double = 1.0,
        val thermalRisk: Double = 0.0,
        val vibrationRisk: Double = 0.0,
        val temporalAlignmentConfidence: Double = 1.0,
        val subframeOffsetMillis: Double = 0.0,
    ) {
        init {
            require(reprojectionErrorPx.isFinite() && reprojectionErrorPx >= 0.0)
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(thermalRisk.isFinite() && thermalRisk in 0.0..1.0)
            require(vibrationRisk.isFinite() && vibrationRisk in 0.0..1.0)
            require(
                temporalAlignmentConfidence.isFinite() &&
                    temporalAlignmentConfidence in 0.0..1.0
            )
            require(subframeOffsetMillis.isFinite() && subframeOffsetMillis >= 0.0)
        }
    }

    data class CameraDrift(
        val deviceId: DeviceId,
        val translationMeters: Double,
        val rotationDegrees: Double,
        val reprojectionErrorPx: Double,
        val translationTrust: Double,
        val rotationTrust: Double,
        val reprojectionTrust: Double,
        val trust: Double,
        val samples: Long,
        val degraded: Boolean,
        val hardFailure: Boolean,
        val warnings: List<String>,
    )

    data class PairDrift(
        val a: DeviceId,
        val b: DeviceId,
        val baselineMeters: Double,
        val baselineDriftMeters: Double,
        val baselineDriftRatio: Double,
        val relativeRotationDegrees: Double,
        val trust: Double,
        val degraded: Boolean,
        val hardFailure: Boolean,
        val warnings: List<String>,
    )

    data class Snapshot(
        val timestampNanos: Long,
        val rigTrust: Double,
        val geometryTrust: Double,
        val thermalTrust: Double,
        val vibrationTrust: Double,
        val temporalTrust: Double,
        val cameraDrift: List<CameraDrift>,
        val pairDrift: List<PairDrift>,
        val degraded: Boolean,
        val hardFailure: Boolean,
        val recalibrationRecommended: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class State(
        var samples: Long = 0L,
        var translationEma: Double = 0.0,
        var rotationEma: Double = 0.0,
        var reprojectionEma: Double = 0.0,
        var thermalEma: Double = 0.0,
        var vibrationEma: Double = 0.0,
        var temporalConfidenceEma: Double = 1.0,
        var subframeOffsetEmaMillis: Double = 0.0,
        var trustEma: Double = 1.0,
        var consecutiveGoodSamples: Int = 0,
        var consecutiveHardFailureEvidence: Int = 0,
        var lastTimestampNanos: Long = 0L,
        val translationHistory: ArrayDeque<Double> = ArrayDeque(),
        val rotationHistory: ArrayDeque<Double> = ArrayDeque(),
    )

    private var baseline: Map<DeviceId, BaselineCamera> = emptyMap()
    private val states = LinkedHashMap<DeviceId, State>()
    private var latestSnapshot: Snapshot? = null

    init {
        require(config.translationWarningMeters > 0.0)
        require(config.translationHardMeters > config.translationWarningMeters)
        require(config.rotationWarningDegrees > 0.0)
        require(config.rotationHardDegrees > config.rotationWarningDegrees)
        require(config.reprojectionWarningPx > 0.0)
        require(config.reprojectionHardPx > config.reprojectionWarningPx)
        require(config.baselineDriftWarningRatio > 0.0)
        require(config.baselineDriftHardRatio > config.baselineDriftWarningRatio)
        require(config.minimumTrust in 0.0..1.0)
        require(config.preferredTrust >= config.minimumTrust)
        require(config.preferredTrust <= 1.0)
        require(config.historyLength >= 8)
        require(config.smoothingAlpha in 0.0..1.0)
        require(config.recoveryAlpha in 0.0..1.0)
        require(config.thermalPenalty in 0.0..1.0)
        require(config.vibrationPenalty in 0.0..1.0)
        require(config.temporalPenalty in 0.0..1.0)
        require(
            config.minimumTemporalAlignmentConfidence in 0.0..1.0
        )
        require(config.maximumSubframeOffsetMillis > 0.0)
        require(config.reprojectionPenalty in 0.0..1.0)
        require(config.geometryPenalty in 0.0..1.0)
        require(config.recoverySamples >= 1)
        require(
            config.minimumHardFailureConfidence in 0.0..1.0
        )
        require(config.persistentHardFailureSamples >= 1)
    }

    @Synchronized
    fun establishBaseline(
        cameras: List<BaselineCamera>,
        timestampNanos: Long,
    ) {
        require(cameras.isNotEmpty())
        baseline = cameras.associateBy { it.deviceId }
        states.clear()
        cameras.forEach {
            states[it.deviceId] = State(
                lastTimestampNanos = timestampNanos,
            )
        }
        latestSnapshot = Snapshot(
            timestampNanos = timestampNanos,
            rigTrust = 1.0,
            geometryTrust = 1.0,
            thermalTrust = 1.0,
            vibrationTrust = 1.0,
            temporalTrust = 1.0,
            cameraDrift = emptyList(),
            pairDrift = emptyList(),
            degraded = false,
            hardFailure = false,
            recalibrationRecommended = false,
            warnings = emptyList(),
            recommendations = emptyList(),
        )
    }

    @Synchronized
    fun observe(
        timestampNanos: Long,
        cameras: List<CurrentCamera>,
    ): Snapshot {
        if (baseline.isEmpty()) {
            return uninitializedSnapshot(timestampNanos).also {
                latestSnapshot = it
            }
        }

        val current = cameras.associateBy { it.deviceId }
        val cameraResults = baseline.values.mapNotNull { reference ->
            current[reference.deviceId]?.let {
                evaluateCamera(reference, it, timestampNanos)
            }
        }

        val pairResults = evaluatePairs(current)
        val cameraTrust = cameraResults.averageCameraTrust()
        val geometryTrust = pairResults.averagePairTrust()
        val thermalTrust = inverseRisk(
            cameraResults.map { stateFor(it.deviceId).thermalEma }.averageOrZero(),
        )
        val vibrationTrust = inverseRisk(
            cameraResults.map { stateFor(it.deviceId).vibrationEma }.averageOrZero(),
        )
        val temporalTrust =
            current.values
                .filter { baseline.containsKey(it.deviceId) }
                .map {
                    it.temporalAlignmentConfidence
                        .coerceIn(0.0, 1.0)
                }
                .averageOrZero()
                .coerceIn(0.0, 1.0)

        val availabilityTrust = (
            current.keys.intersect(baseline.keys).size.toDouble() /
                baseline.size.toDouble().coerceAtLeast(1.0)
        ).coerceIn(0.0, 1.0)

        val rigTrust = (
            cameraTrust * 0.42 +
                geometryTrust * 0.31 +
                thermalTrust * 0.07 +
                vibrationTrust * 0.08 +
                temporalTrust * 0.12 +
                availabilityTrust * 0.08
            ).coerceIn(config.minimumTrust, 1.0)

        val hardFailure =
            cameraResults.any { it.hardFailure } ||
                pairResults.any { it.hardFailure } ||
                availabilityTrust < 0.50

        val temporalMisalignment =
            temporalTrust <
                config.minimumTemporalAlignmentConfidence ||
                current.values.any {
                    baseline.containsKey(it.deviceId) &&
                        abs(it.subframeOffsetMillis) >
                            config.maximumSubframeOffsetMillis
                }

        val degraded =
            hardFailure ||
                temporalMisalignment ||
                rigTrust < config.preferredTrust

        val recalibrationRecommended =
            hardFailure ||
                temporalMisalignment ||
                rigTrust < 0.72 ||
                geometryTrust < 0.68

        return Snapshot(
            timestampNanos = timestampNanos,
            rigTrust = rigTrust,
            geometryTrust = geometryTrust,
            thermalTrust = thermalTrust,
            vibrationTrust = vibrationTrust,
            temporalTrust = temporalTrust,
            cameraDrift = cameraResults,
            pairDrift = pairResults,
            degraded = degraded,
            hardFailure = hardFailure,
            recalibrationRecommended = recalibrationRecommended,
            warnings = buildWarnings(
                cameraResults,
                pairResults,
                rigTrust,
                geometryTrust,
                hardFailure,
            ),
            recommendations = buildRecommendations(
                cameraResults,
                pairResults,
                rigTrust,
                recalibrationRecommended,
            ),
        ).also {
            latestSnapshot = it
        }
    }

    @Synchronized
    fun latest(): Snapshot? = latestSnapshot

    @Synchronized
    fun rigTrust(): Double =
        latestSnapshot?.rigTrust
            ?: if (baseline.isNotEmpty()) 1.0 else 0.0

    @Synchronized
    fun cameraTrust(deviceId: DeviceId): Double =
        latestSnapshot?.cameraDrift
            ?.firstOrNull { it.deviceId == deviceId }
            ?.trust
            ?: if (baseline.containsKey(deviceId)) 1.0 else 0.0

    @Synchronized
    fun shouldHoldMultiViewOutput(): Boolean =
        latestSnapshot?.let {
            it.hardFailure ||
                it.rigTrust < 0.45 ||
                it.geometryTrust < 0.40
        } ?: true

    @Synchronized
    fun reset() {
        baseline = emptyMap()
        states.clear()
        latestSnapshot = null
    }

    override fun close() = reset()

    private fun evaluateCamera(
        reference: BaselineCamera,
        current: CurrentCamera,
        timestampNanos: Long,
    ): CameraDrift {
        val state = stateFor(reference.deviceId)
        val translation =
            reference.pose.position.distanceTo(current.pose.position)
        val rotation =
            quaternionAngleDegrees(
                reference.pose,
                current.pose,
            )
        val reprojection =
            current.reprojectionErrorPx.coerceAtLeast(0.0)
        val thermal =
            current.thermalRisk.coerceIn(0.0, 1.0)
        val vibration =
            current.vibrationRisk.coerceIn(0.0, 1.0)
        val temporalAlignment =
            current.temporalAlignmentConfidence
                .coerceIn(0.0, 1.0)
        val subframeOffset =
            abs(
                current.subframeOffsetMillis,
            )
        val confidence =
            current.confidence.coerceIn(0.0, 1.0)

        state.translationEma =
            ema(
                previous = state.translationEma,
                current = translation,
                first = state.samples == 0L,
            )
        state.rotationEma =
            ema(
                previous = state.rotationEma,
                current = rotation,
                first = state.samples == 0L,
            )
        state.reprojectionEma =
            ema(
                previous = state.reprojectionEma,
                current = reprojection,
                first = state.samples == 0L,
            )
        state.thermalEma =
            ema(
                previous = state.thermalEma,
                current = thermal,
                first = false,
            )
        state.vibrationEma =
            ema(
                previous = state.vibrationEma,
                current = vibration,
                first = false,
            )

        state.temporalConfidenceEma =
            ema(
                previous =
                    state.temporalConfidenceEma,
                current =
                    temporalAlignment,
                first =
                    state.samples == 0L,
            )
        state.subframeOffsetEmaMillis =
            ema(
                previous =
                    state.subframeOffsetEmaMillis,
                current =
                    subframeOffset,
                first =
                    state.samples == 0L,
            )

        push(state.translationHistory, translation)
        push(state.rotationHistory, rotation)

        val translationTrust =
            thresholdTrust(
                state.translationEma,
                config.translationWarningMeters,
                config.translationHardMeters,
            )
        val rotationTrust =
            thresholdTrust(
                state.rotationEma,
                config.rotationWarningDegrees,
                config.rotationHardDegrees,
            )
        val reprojectionTrust =
            thresholdTrust(
                state.reprojectionEma,
                config.reprojectionWarningPx,
                config.reprojectionHardPx,
            )

        val targetTrust = (
            translationTrust * 0.26 +
                rotationTrust * 0.26 +
                reprojectionTrust * 0.28 +
                (1.0 - thermal * config.thermalPenalty) * 0.06 +
                (1.0 - vibration * config.vibrationPenalty) * 0.08 +
                state.temporalConfidenceEma * 0.08 +
                (
                    if (
                        state.subframeOffsetEmaMillis >
                            config.maximumSubframeOffsetMillis
                    ) {
                        1.0 -
                            config.temporalPenalty
                    } else {
                        1.0
                    }
                ).coerceIn(0.0, 1.0) * 0.06 +
                confidence * 0.06
            ).coerceIn(0.0, 1.0)

        val alpha =
            if (targetTrust >= state.trustEma) {
                config.recoveryAlpha
            } else {
                config.smoothingAlpha
            }

        state.trustEma =
            (
                state.trustEma +
                    (targetTrust - state.trustEma) * alpha
                ).coerceIn(config.minimumTrust, 1.0)

        state.consecutiveGoodSamples =
            if (targetTrust >= 0.82) {
                state.consecutiveGoodSamples + 1
            } else {
                0
            }

        state.samples += 1L
        state.lastTimestampNanos = timestampNanos

        val robustTranslation =
            robustGate(
                state.translationHistory,
                config.translationWarningMeters,
            )
        val robustRotation =
            robustGate(
                state.rotationHistory,
                config.rotationWarningDegrees,
            )

        val hardEvidence =
            state.translationEma >= config.translationHardMeters ||
                state.rotationEma >= config.rotationHardDegrees ||
                state.reprojectionEma >= config.reprojectionHardPx ||
                robustTranslation >= config.translationHardMeters ||
                robustRotation >= config.rotationHardDegrees

        if (
            confidence >=
                config.minimumHardFailureConfidence &&
                hardEvidence
        ) {
            state.consecutiveHardFailureEvidence += 1
        } else if (!hardEvidence) {
            state.consecutiveHardFailureEvidence = 0
        }

        val hardFailure =
            hardEvidence &&
                (
                    confidence >=
                        config.minimumHardFailureConfidence ||
                    state.consecutiveHardFailureEvidence >=
                        config.persistentHardFailureSamples
                )

        val degraded =
            hardFailure ||
                state.trustEma < config.preferredTrust

        return CameraDrift(
            deviceId = reference.deviceId,
            translationMeters = translation,
            rotationDegrees = rotation,
            reprojectionErrorPx = reprojection,
            translationTrust = translationTrust,
            rotationTrust = rotationTrust,
            reprojectionTrust = reprojectionTrust,
            trust = state.trustEma,
            samples = state.samples,
            degraded = degraded,
            hardFailure = hardFailure,
            warnings =
                cameraWarnings(
                    reference.deviceId,
                    state,
                    thermal,
                    vibration,
                ) +
                    when {
                        state.temporalConfidenceEma <
                            config.minimumTemporalAlignmentConfidence ->
                            listOf(
                                "temporal alignment confidence is low",
                                "recalibration recommended for temporal alignment",
                            )
                        state.subframeOffsetEmaMillis >
                            config.maximumSubframeOffsetMillis ->
                            listOf(
                                "sub-frame offset exceeds drift envelope",
                            )
                        else -> emptyList()
                    },
        )
    }

    private fun evaluatePairs(
        current: Map<DeviceId, CurrentCamera>,
    ): List<PairDrift> {
        val ids = baseline.keys
            .filter { current.containsKey(it) }
            .sortedBy { it.value }

        val result = mutableListOf<PairDrift>()

        for (i in 0 until ids.lastIndex) {
            for (j in i + 1 until ids.size) {
                val a = ids[i]
                val b = ids[j]
                val baseA = baseline.getValue(a)
                val baseB = baseline.getValue(b)
                val nowA = current.getValue(a)
                val nowB = current.getValue(b)

                val baseVector =
                    baseB.pose.position - baseA.pose.position
                val currentVector =
                    nowB.pose.position - nowA.pose.position

                val baseLength = baseVector.norm()
                val currentLength = currentVector.norm()
                val driftMeters =
                    abs(currentLength - baseLength)

                val driftRatio =
                    if (baseLength < 1e-6) 1.0
                    else driftMeters / baseLength

                val relativeRotation =
                    relativeRotationDegrees(
                        baseA.pose,
                        baseB.pose,
                        nowA.pose,
                        nowB.pose,
                    )

                val baselineTrust =
                    thresholdTrust(
                        driftRatio,
                        config.baselineDriftWarningRatio,
                        config.baselineDriftHardRatio,
                    )
                val rotationTrust =
                    thresholdTrust(
                        relativeRotation,
                        config.rotationWarningDegrees,
                        config.rotationHardDegrees,
                    )
                val individualTrust =
                    min(
                        cameraTrust(a),
                        cameraTrust(b),
                    )

                val trust = (
                    baselineTrust * 0.46 +
                        rotationTrust * 0.32 +
                        individualTrust * 0.22
                    ).coerceIn(
                    config.minimumTrust,
                    1.0,
                )

                val hardFailure =
                    (
                        driftRatio >= config.baselineDriftHardRatio ||
                            relativeRotation >= config.rotationHardDegrees
                    ) &&
                        nowA.confidence >=
                            config.minimumHardFailureConfidence &&
                        nowB.confidence >=
                            config.minimumHardFailureConfidence

                result += PairDrift(
                    a = a,
                    b = b,
                    baselineMeters = baseLength,
                    baselineDriftMeters = driftMeters,
                    baselineDriftRatio = driftRatio,
                    relativeRotationDegrees = relativeRotation,
                    trust = trust,
                    degraded =
                        hardFailure ||
                            trust < config.preferredTrust,
                    hardFailure = hardFailure,
                    warnings = pairWarnings(
                        a,
                        b,
                        driftRatio,
                        relativeRotation,
                    ),
                )
            }
        }

        return result
    }

    private fun quaternionAngleDegrees(
        a: Pose,
        b: Pose,
    ): Double {
        val aw = a.rotation.normalized()
        val bw = b.rotation.normalized()
        val dot = abs(
            aw.w * bw.w +
                aw.x * bw.x +
                aw.y * bw.y +
                aw.z * bw.z,
        ).coerceIn(-1.0, 1.0)
        return Math.toDegrees(2.0 * acos(dot))
    }

    private fun relativeRotationDegrees(
        baseA: Pose,
        baseB: Pose,
        currentA: Pose,
        currentB: Pose,
    ): Double {
        val baseRelative =
            baseA.rotation.conjugate() * baseB.rotation
        val currentRelative =
            currentA.rotation.conjugate() * currentB.rotation
        val dot = abs(
            baseRelative.normalized().w * currentRelative.normalized().w +
                baseRelative.normalized().x * currentRelative.normalized().x +
                baseRelative.normalized().y * currentRelative.normalized().y +
                baseRelative.normalized().z * currentRelative.normalized().z,
        ).coerceIn(-1.0, 1.0)
        return Math.toDegrees(2.0 * acos(dot))
    }

    private fun stateFor(id: DeviceId): State =
        states.getOrPut(id) { State() }

    private fun ema(
        previous: Double,
        current: Double,
        first: Boolean,
    ): Double =
        if (first) current
        else previous +
            (current - previous) * config.smoothingAlpha

    private fun thresholdTrust(
        value: Double,
        warning: Double,
        hard: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= warning) return 1.0
        if (value >= hard) return 0.10
        val t = (
            value - warning
        ) / (hard - warning).coerceAtLeast(1e-9)
        return (1.0 - t * 0.90).coerceIn(0.10, 1.0)
    }

    private fun robustGate(
        values: Collection<Double>,
        base: Double,
    ): Double {
        if (values.isEmpty()) return base
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return base
        val median = median(sorted)
        val deviations = sorted.map { abs(it - median) }.sorted()
        val mad = median(deviations)
        return max(
            base,
            median + mad * 1.4826 * 3.5,
        )
    }

    private fun median(values: List<Double>): Double =
        if (values.isEmpty()) 0.0
        else if (values.size % 2 == 0) {
            (
                values[values.size / 2 - 1] +
                    values[values.size / 2]
                ) * 0.5
        } else {
            values[values.size / 2]
        }

    private fun push(
        queue: ArrayDeque<Double>,
        value: Double,
    ) {
        queue.addLast(value.coerceAtLeast(0.0))
        while (queue.size > config.historyLength) {
            queue.removeFirst()
        }
    }

    private fun inverseRisk(risk: Double): Double =
        (1.0 - risk.coerceIn(0.0, 1.0))
            .coerceIn(0.0, 1.0)

    private fun List<CameraDrift>.averageCameraTrust(): Double =
        if (isEmpty()) 0.0 else averageOf { it.trust }

    private fun List<PairDrift>.averagePairTrust(): Double =
        if (isEmpty()) {
            if (baseline.size <= 1) 1.0 else 0.0
        } else {
            averageOf { it.trust }
        }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else filter { it.isFinite() }.average()

    private fun <T> Collection<T>.averageOf(
        selector: (T) -> Double,
    ): Double =
        if (isEmpty()) 0.0
        else map(selector).average().coerceIn(0.0, 1.0)

    private fun cameraWarnings(
        deviceId: DeviceId,
        state: State,
        thermal: Double,
        vibration: Double,
    ): List<String> = buildList {
        if (state.translationEma > config.translationWarningMeters) {
            add("camera " + deviceId.value + " translation drift elevated")
        }
        if (state.rotationEma > config.rotationWarningDegrees) {
            add("camera " + deviceId.value + " rotation drift elevated")
        }
        if (state.reprojectionEma > config.reprojectionWarningPx) {
            add("camera " + deviceId.value + " reprojection residual elevated")
        }
        if (thermal > 0.70) {
            add("camera " + deviceId.value + " thermal risk is high")
        }
        if (vibration > 0.65) {
            add("camera " + deviceId.value + " vibration risk is high")
        }
        if (
            state.consecutiveGoodSamples >= config.recoverySamples &&
            state.trustEma >= config.preferredTrust
        ) {
            add("camera " + deviceId.value + " trust recovered")
        }
    }.distinct()

    private fun pairWarnings(
        a: DeviceId,
        b: DeviceId,
        ratio: Double,
        rotation: Double,
    ): List<String> = buildList {
        if (ratio > config.baselineDriftWarningRatio) {
            add("pair " + a.value + "/" + b.value + " baseline changed")
        }
        if (rotation > config.rotationWarningDegrees) {
            add("pair " + a.value + "/" + b.value + " orientation changed")
        }
        if (ratio >= config.baselineDriftHardRatio) {
            add("pair " + a.value + "/" + b.value + " baseline hard failure")
        }
    }.distinct()

    private fun buildWarnings(
        camera: List<CameraDrift>,
        pairs: List<PairDrift>,
        rigTrust: Double,
        geometryTrust: Double,
        hardFailure: Boolean,
    ): List<String> = buildList {
        if (hardFailure) {
            add("hard calibration drift detected")
        }
        if (rigTrust < config.preferredTrust) {
            add("rig trust below preferred production level")
        }
        if (geometryTrust < 0.68) {
            add("multi-camera geometry trust degraded")
        }
        camera.filter { it.hardFailure }.forEach {
            add("camera " + it.deviceId.value + " exceeds hard drift limits")
        }
        pairs.filter { it.hardFailure }.forEach {
            add(
                "camera pair " + it.a.value + "/" + it.b.value +
                    " exceeds hard drift limits",
            )
        }
        if (
            camera.any {
                it.warnings.any { warning ->
                    warning.contains("temporal", ignoreCase = true) ||
                        warning.contains("sub-frame", ignoreCase = true)
                }
            }
        ) {
            add("temporal alignment drift detected")
            add("recalibration recommended for temporal alignment")
        }
    }.distinct()

    private fun buildRecommendations(
        camera: List<CameraDrift>,
        pairs: List<PairDrift>,
        rigTrust: Double,
        recalibrationRecommended: Boolean,
    ): List<String> = buildList {
        if (recalibrationRecommended) {
            add("Run live rig recalibration before trusting new multi-view metric output.")
        }
        camera.filter {
            it.translationMeters > config.translationWarningMeters ||
                it.rotationDegrees > config.rotationWarningDegrees
        }.forEach {
            add(
                "Inspect physical mounting and camera alignment for " +
                    it.deviceId.value + ".",
            )
        }
        pairs.filter {
            it.baselineDriftRatio > config.baselineDriftWarningRatio
        }.forEach {
            add(
                "Revalidate the " +
                    it.a.value + "/" + it.b.value +
                    " baseline before high-speed capture.",
            )
        }
        if (rigTrust < 0.45) {
            add("Hold multi-view metric output and retain monocular motion as fallback.")
        } else if (rigTrust < 0.72) {
            add("Reduce multi-view contribution until rig trust recovers.")
        } else {
            add("Rig remains suitable for normal multi-view operation.")
        }
    }.distinct()

    private fun uninitializedSnapshot(
        timestampNanos: Long,
    ): Snapshot =
        Snapshot(
            timestampNanos = timestampNanos,
            rigTrust = 0.0,
            geometryTrust = 0.0,
            thermalTrust = 0.0,
            vibrationTrust = 0.0,
            temporalTrust = 0.0,
            cameraDrift = emptyList(),
            pairDrift = emptyList(),
            degraded = true,
            hardFailure = true,
            recalibrationRecommended = true,
            warnings = listOf("drift baseline is not initialized"),
            recommendations = listOf(
                "Establish a trusted calibration baseline before multi-view reconstruction.",
            ),
        )
}
