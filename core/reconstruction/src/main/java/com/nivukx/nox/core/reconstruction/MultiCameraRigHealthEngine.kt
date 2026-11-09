package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceDescriptor
import com.nivukx.nox.core.model.DeviceId
import kotlin.math.abs
import kotlin.math.exp

/**
 * Production operator-facing health model for a distributed mobile capture rig.
 *
 * This layer deliberately does not invent camera quality. It combines measured transport timing,
 * packet loss, synchronization/capture confidence, calibration status, and the causal per-camera
 * geometry trust maintained by AdaptiveMultiViewEvidenceWeightEngine.
 */
class MultiCameraRigHealthEngine(
    private val config: Config = Config(),
) {
    private val faultContainment = CameraFaultContainmentEngine()


    enum class State {
        UNKNOWN,
        HEALTHY,
        WATCH,
        HOLD,
        BLOCKED,
    }

    enum class CameraState {
        UNKNOWN,
        HEALTHY,
        WATCH,
        HOLD,
        OFFLINE,
    }

    data class Config(
        val healthyThreshold: Double = 0.82,
        val watchThreshold: Double = 0.62,
        val holdThreshold: Double = 0.42,
        val packetLossScale: Double = 0.12,
        val clockOffsetScaleMillis: Double = 5.0,
        val roundTripScaleMillis: Double = 60.0,
        val hardPacketLoss: Double = 0.35,
        val hardClockOffsetMillis: Double = 20.0,
        val hardRoundTripMillis: Double = 450.0,
    ) {
        init {
            require(healthyThreshold in 0.0..1.0)
            require(watchThreshold in 0.0..healthyThreshold)
            require(holdThreshold in 0.0..watchThreshold)
            require(packetLossScale > 0.0)
            require(clockOffsetScaleMillis > 0.0)
            require(roundTripScaleMillis > 0.0)
            require(hardPacketLoss in 0.0..1.0)
            require(hardClockOffsetMillis > 0.0)
            require(hardRoundTripMillis > 0.0)
        }
    }

    data class Input(
        val devices: List<DeviceDescriptor>,
        val calibrationReady: Boolean,
        val calibrationConfidence: Double,
        val synchronizationConfidence: Double,
        val captureTimingConfidence: Double,
        val cameraStatuses:
            List<AdaptiveMultiViewEvidenceWeightEngine.CameraStatus> = emptyList(),
    )

    data class CameraHealth(
        val deviceId: DeviceId,
        val state: CameraState,
        val score: Double,
        val transportScore: Double,
        val clockScore: Double,
        val geometryScore: Double,
        val packetLoss: Double,
        val clockOffsetMillis: Double,
        val roundTripMillis: Double,
        val geometryTrust: Double,
        val quarantined: Boolean = false,
        val quarantineFailureStreak: Int = 0,
        val quarantineRecoveryStreak: Int = 0,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    data class Result(
        val state: State,
        val overallScore: Double,
        val onlineCameraCount: Int,
        val blockedCameraCount: Int,
        val quarantinedCameraCount: Int = 0,
        val cameras: List<CameraHealth>,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    fun evaluate(input: Input): Result {
        val online = input.devices.filter { it.connected }
        val statusByCamera =
            input.cameraStatuses.associateBy { it.deviceId }

        val cameras = input.devices.map { device ->
            evaluateCamera(
                device = device,
                geometryStatus = statusByCamera[device.id],
                calibrationConfidence =
                    input.calibrationConfidence.coerceIn(0.0, 1.0),
                synchronizationConfidence =
                    input.synchronizationConfidence.coerceIn(0.0, 1.0),
                captureTimingConfidence =
                    input.captureTimingConfidence.coerceIn(0.0, 1.0),
            )
        }

        val onlineScores =
            cameras.filter {
                it.state != CameraState.OFFLINE &&
                    !it.quarantined
            }.map { it.score }

        val quarantinedCount =
            cameras.count { it.quarantined }

        val networkScore =
            averageOrZero(onlineScores)

        val weakestCameraScore =
            onlineScores.minOrNull()
                ?.coerceIn(0.0, 1.0)
                ?: 0.0

        val dispersionPenalty =
            if (onlineScores.size <= 1) {
                0.0
            } else {
                (
                    onlineScores.average() -
                        weakestCameraScore
                    ).coerceIn(0.0, 1.0) * 0.12
            }

        val globalInfrastructureScore = (
            input.synchronizationConfidence.coerceIn(0.0, 1.0) * 0.52 +
                input.captureTimingConfidence.coerceIn(0.0, 1.0) * 0.28 +
                input.calibrationConfidence.coerceIn(0.0, 1.0) * 0.20
            ).coerceIn(0.0, 1.0)

        val overallScore = (
            networkScore * 0.50 +
                weakestCameraScore * 0.22 +
                globalInfrastructureScore * 0.28 -
                dispersionPenalty
            ).coerceIn(0.0, 1.0)

        val blocked = cameras.count {
            it.state == CameraState.HOLD &&
                !it.quarantined
        }
        val quarantined = cameras.count { it.quarantined }
        val activeCameraCount = cameras.count {
            it.state != CameraState.OFFLINE &&
                !it.quarantined
        }

        val state = when {
            online.isEmpty() -> State.UNKNOWN
            activeCameraCount < 2 -> State.BLOCKED
            blocked > 0 -> State.BLOCKED
            !input.calibrationReady ||
                input.calibrationConfidence < 0.50 -> State.HOLD
            quarantined > 0 ->
                when {
                    overallScore >= config.watchThreshold ->
                        State.WATCH
                    overallScore >= config.holdThreshold ->
                        State.HOLD
                    else ->
                        State.BLOCKED
                }
            overallScore >= config.healthyThreshold &&
                cameras.none {
                    it.state == CameraState.WATCH
                } ->
                State.HEALTHY
            overallScore >= config.watchThreshold ->
                State.WATCH
            overallScore >= config.holdThreshold ->
                State.HOLD
            else ->
                State.BLOCKED
        }

        val warnings = buildWarnings(
            state = state,
            cameras = cameras,
            input = input,
        )

        val recommendations = buildRecommendations(
            state = state,
            cameras = cameras,
            input = input,
        )

        return Result(
            state = state,
            overallScore = overallScore,
            onlineCameraCount = online.size,
            blockedCameraCount = blocked,
            quarantinedCameraCount = quarantined,
            cameras = cameras,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    private fun evaluateCamera(
        device: DeviceDescriptor,
        geometryStatus:
            AdaptiveMultiViewEvidenceWeightEngine.CameraStatus?,
        calibrationConfidence: Double,
        synchronizationConfidence: Double,
        captureTimingConfidence: Double,
    ): CameraHealth {
        if (!device.connected) {
            return CameraHealth(
                deviceId = device.id,
                state = CameraState.OFFLINE,
                score = 0.0,
                transportScore = 0.0,
                clockScore = 0.0,
                geometryScore = geometryStatus?.trust ?: 0.0,
                packetLoss = device.packetLoss.coerceIn(0.0, 1.0),
                clockOffsetMillis =
                    abs(device.clockOffsetNanos) / 1e6,
                roundTripMillis =
                    device.roundTripNanos.coerceAtLeast(0L) / 1e6,
                geometryTrust = geometryStatus?.trust ?: 0.0,
                quarantined = true,
                quarantineFailureStreak = 0,
                quarantineRecoveryStreak = 0,
                warnings = listOf("camera-offline"),
                recommendations = listOf(
                    "restore the camera connection before the next metric capture",
                ),
            )
        }

        val packetLoss =
            device.packetLoss.coerceIn(0.0, 1.0)
        val clockOffsetMillis =
            abs(device.clockOffsetNanos) / 1e6
        val roundTripMillis =
            device.roundTripNanos.coerceAtLeast(0L) / 1e6

        val packetScore = exp(
            -packetLoss / config.packetLossScale,
        ).coerceIn(0.0, 1.0)

        val clockScore = exp(
            -clockOffsetMillis /
                config.clockOffsetScaleMillis,
        ).coerceIn(0.0, 1.0)

        val rttScore = exp(
            -roundTripMillis /
                config.roundTripScaleMillis,
        ).coerceIn(0.0, 1.0)

        val transportScore = (
            packetScore * 0.48 +
                clockScore * 0.28 +
                rttScore * 0.24
            ).coerceIn(0.0, 1.0)

        val geometryTrust =
            geometryStatus?.trust?.coerceIn(0.0, 1.0)
                ?: calibrationConfidence

        val geometryScore = (
            geometryTrust * 0.64 +
                calibrationConfidence * 0.20 +
                synchronizationConfidence * 0.10 +
                captureTimingConfidence * 0.06
            ).coerceIn(0.0, 1.0)

        val score = (
            transportScore * 0.46 +
                geometryScore * 0.54
            ).coerceIn(0.0, 1.0)

        val hardFailure =
            packetLoss >= config.hardPacketLoss ||
                clockOffsetMillis >= config.hardClockOffsetMillis ||
                roundTripMillis >= config.hardRoundTripMillis ||
                geometryStatus?.state ==
                    AdaptiveMultiViewEvidenceWeightEngine.State.HOLD

        val rawState = when {
            hardFailure -> CameraState.HOLD
            score >= config.healthyThreshold -> CameraState.HEALTHY
            score >= config.watchThreshold -> CameraState.WATCH
            else -> CameraState.HOLD
        }

        val containment =
            faultContainment.observe(
                CameraFaultContainmentEngine.Observation(
                    cameraId = device.id,
                    score = score,
                    hardFailure = hardFailure,
                    connected = device.connected,
                ),
            )

        val state =
            if (containment.state ==
                CameraFaultContainmentEngine.State.QUARANTINED
            ) {
                CameraState.HOLD
            } else {
                rawState
            }

        val warnings = buildList {
            if (packetLoss > 0.08) {
                add("packet-loss-elevated")
            }
            if (clockOffsetMillis > 3.0) {
                add("clock-offset-elevated")
            }
            if (roundTripMillis > 80.0) {
                add("network-round-trip-elevated")
            }
            if (geometryTrust < 0.55) {
                add("geometry-trust-degraded")
            }
            if (
                geometryStatus?.state ==
                    AdaptiveMultiViewEvidenceWeightEngine.State.DEGRADED
            ) {
                add("adaptive-reconstruction-weight-reduced")
            }
            if (
                geometryStatus?.state ==
                    AdaptiveMultiViewEvidenceWeightEngine.State.HOLD
            ) {
                add("camera-held-out-of-primary-reconstruction")
            }
        }.distinct()

        val recommendations = buildList {
            if (packetLoss > 0.08) {
                add("stabilize the peer link before increasing capture density")
            }
            if (clockOffsetMillis > 3.0) {
                add("re-run clock synchronization before metric reconstruction")
            }
            if (roundTripMillis > 80.0) {
                add("reduce network contention or move the camera closer to the peer")
            }
            if (geometryTrust < 0.55) {
                add("inspect this camera for mount movement, blur, focus or calibration drift")
            }
            if (state == CameraState.HOLD) {
                add("exclude this camera from primary reconstruction until evidence recovers")
            }
        }.distinct()

        return CameraHealth(
            deviceId = device.id,
            state = state,
            score = score,
            transportScore = transportScore,
            clockScore = clockScore,
            geometryScore = geometryScore,
            packetLoss = packetLoss,
            clockOffsetMillis = clockOffsetMillis,
            roundTripMillis = roundTripMillis,
            geometryTrust = geometryTrust,
            quarantined =
                containment.state ==
                    CameraFaultContainmentEngine.State.QUARANTINED,
            quarantineFailureStreak =
                containment.failureStreak,
            quarantineRecoveryStreak =
                containment.recoveryStreak,
            warnings =
                (
                    warnings +
                        containment.warnings
                ).distinct(),
            recommendations =
                (
                    recommendations +
                        containment.recommendations
                ).distinct(),
        )
    }

    private fun buildWarnings(
        state: State,
        cameras: List<CameraHealth>,
        input: Input,
    ): List<String> = buildList {
        if (!input.calibrationReady) {
            add("rig-calibration-not-ready")
        }
        if (input.synchronizationConfidence < 0.70) {
            add("rig-synchronization-below-production-threshold")
        }
        if (input.captureTimingConfidence < 0.70) {
            add("capture-timing-confidence-below-production-threshold")
        }
        if (
            cameras.isNotEmpty() &&
            cameras.filter {
                it.state != CameraState.OFFLINE
            }.minOfOrNull { it.score }?.let { it < 0.45 } == true
        ) {
            add("weakest-online-camera-below-metric-trust-floor")
        }
        if (
            cameras.count {
                it.state == CameraState.HOLD
            } >
                maxOf(1, input.devices.count { it.connected } / 3)
        ) {
            add("multiple-online-cameras-held-out-of-primary-rig")
        }
        cameras.filter { it.warnings.isNotEmpty() }.forEach { camera ->
            add(
                camera.deviceId.value +
                    ":" +
                    camera.warnings.joinToString(","),
            )
        }
        when (state) {
            State.UNKNOWN -> add("rig-health-awaiting-online-cameras")
            State.HEALTHY -> Unit
            State.WATCH -> add("rig-health-degraded-monitor-before-capture")
            State.HOLD -> add("metric-output-should-be-held-until-rig-health-recovers")
            State.BLOCKED -> add("rig-health-blocks-trusted-metric-capture")
        }
    }.distinct()

    fun reset() {
        faultContainment.reset()
    }

    private fun buildRecommendations(
        state: State,
        cameras: List<CameraHealth>,
        input: Input,
    ): List<String> = buildList {
        when (state) {
            State.UNKNOWN ->
                add("connect at least two cameras and complete calibration")
            State.HEALTHY -> Unit
            State.WATCH ->
                add("watch the rig health matrix before starting another take")
            State.HOLD ->
                add("hold metric reconstruction and recover the lowest-trust cameras")
            State.BLOCKED ->
                add("do not accept metric output until rig calibration and camera health recover")
        }

        if (!input.calibrationReady) {
            add("complete multi-camera calibration before metric capture")
        }
        val quarantined = cameras.count { it.quarantined }
        if (quarantined > 0) {
            if (input.devices.count { it.connected } - quarantined >= 2) {
                add("maintain the remaining active cameras while the quarantined device recovers")
            } else {
                add("recover or replace a quarantined camera before metric multi-view capture")
            }
        }


        cameras
            .sortedBy { it.score }
            .take(2)
            .filter { it.state != CameraState.OFFLINE }
            .forEach {
                add(
                    "prioritize " +
                        it.deviceId.value +
                        " for recovery (score=" +
                        "%.2f".format(it.score) +
                        ")",
                )
            }
    }.distinct()

    private fun averageOrZero(values: List<Double>): Double =
        if (values.isEmpty()) 0.0 else values.average()
}
