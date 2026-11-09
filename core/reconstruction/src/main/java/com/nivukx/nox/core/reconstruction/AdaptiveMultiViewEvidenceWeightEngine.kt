package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.ln

/**
 * Adaptive per-camera/per-joint evidence weighting for multi-view reconstruction.
 *
 * Frame N reprojection diagnostics influence frame N+1. This causal design lets NoX reduce the
 * influence of a camera that repeatedly disagrees with calibrated geometry without ever using the
 * current output to retroactively justify the current input.
 */
class AdaptiveMultiViewEvidenceWeightEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val neutralWeight: Double = 1.0,
        val minimumWeight: Double = 0.14,
        val maximumWeight: Double = 1.0,
        val warningResidualPx: Double = 5.0,
        val hardResidualPx: Double = 18.0,
        val residualScalePx: Double = 4.5,
        val residualEmaAlpha: Double = 0.22,
        val trustEmaAlpha: Double = 0.16,
        val recoveryAlpha: Double = 0.06,
        val minimumEvidenceSamples: Int = 3,
        val staleAfterFrames: Int = 12,
        val staleDecayPerFrame: Double = 0.055,
        val cameraFailureThreshold: Double = 0.30,
        val cameraHoldThreshold: Double = 0.22,
        val confidenceFloor: Double = 0.05,
        val globalCalibrationWeight: Double = 0.56,
        val globalSynchronizationWeight: Double = 0.44,
        val globalTrustEmaAlpha: Double = 0.18,
        val globalTrustMinimumWeight: Double = 0.24,
        val globalTrustWarningThreshold: Double = 0.62,
        // Match the distributed rig envelope so evidence trust can be maintained per camera
        // without forcing the dense solver to ingest every camera simultaneously.
        val maximumCameras: Int = 128,
        val maximumJointStatesPerCamera: Int = 4_096,
        val jointStatePruneAfterFrames: Int = 120,
    ) {
        init {
            require(neutralWeight in 0.0..1.0)
            require(minimumWeight in 0.0..1.0)
            require(maximumWeight in 0.0..1.0)
            require(maximumWeight >= minimumWeight)
            require(warningResidualPx > 0.0)
            require(hardResidualPx > warningResidualPx)
            require(residualScalePx > 0.0)
            require(residualEmaAlpha in 0.0..1.0)
            require(trustEmaAlpha in 0.0..1.0)
            require(recoveryAlpha in 0.0..1.0)
            require(minimumEvidenceSamples >= 1)
            require(staleAfterFrames >= 1)
            require(staleDecayPerFrame in 0.0..1.0)
            require(cameraFailureThreshold in 0.0..1.0)
            require(cameraHoldThreshold in 0.0..1.0)
            require(confidenceFloor in 0.0..1.0)
            require(globalCalibrationWeight >= 0.0)
            require(globalSynchronizationWeight >= 0.0)
            require(globalCalibrationWeight + globalSynchronizationWeight > 0.0)
            require(globalTrustEmaAlpha in 0.0..1.0)
            require(globalTrustMinimumWeight in 0.0..1.0)
            require(globalTrustWarningThreshold in 0.0..1.0)
            require(maximumCameras >= 2)
            require(maximumJointStatesPerCamera >= 64)
            require(jointStatePruneAfterFrames >= staleAfterFrames)
        }
    }

    data class Tuning(
        val adaptivity: Double = 0.84,
        val minimumViewTrust: Double = 0.20,
        val staleRecovery: Double = 0.72,
        val disagreementSensitivity: Double = 0.84,
    ) {
        init {
            require(adaptivity in 0.0..1.0)
            require(minimumViewTrust in 0.0..1.0)
            require(staleRecovery in 0.0..1.0)
            require(disagreementSensitivity in 0.0..1.0)
        }
    }

    data class ViewKey(
        val deviceId: DeviceId,
        val jointId: Int,
    )

    data class CameraStatus(
        val deviceId: DeviceId,
        val trust: Double,
        val observationCount: Int,
        val degradedJointCount: Int,
        val hardFailureJointCount: Int,
        val meanResidualPx: Double,
        val state: State,
    )

    enum class State {
        NEUTRAL,
        HEALTHY,
        DEGRADED,
        HOLD,
        RECOVERING,
        UNKNOWN,
    }

    data class Snapshot(
        val cameraWeights: Map<DeviceId, Double>,
        val jointWeights: Map<ViewKey, Double>,
        val cameraStatuses: List<CameraStatus>,
        val degradedCameraCount: Int,
        val holdCameraCount: Int,
        val dominantCamera: DeviceId?,
        val warnings: List<String>,
        val globalRigTrust: Double = 1.0,
    )

    private data class JointState(
        var trust: Double = 1.0,
        var residualEmaPx: Double = 0.0,
        var evidenceSamples: Int = 0,
        var lastFrameIndex: Long = Long.MIN_VALUE,
        var staleFrames: Int = 0,
    )

    private data class CameraState(
        var trust: Double = 1.0,
        var residualEmaPx: Double = 0.0,
        var observationCount: Int = 0,
        var degradedJointCount: Int = 0,
        var hardFailureJointCount: Int = 0,
        var lastFrameIndex: Long = Long.MIN_VALUE,
        val jointStates: MutableMap<Int, JointState> = mutableMapOf(),
    )

    private var tuning = Tuning()
    private var frameIndex = 0L
    private var globalRigTrust = 1.0
    private var lastGlobalRigTarget = 1.0
    private val cameras = mutableMapOf<DeviceId, CameraState>()

    @Synchronized
    fun configure(value: Tuning) {
        tuning = value
    }

    @Synchronized
    fun currentTuning(): Tuning = tuning

    fun configuration(): Config = config

    @Synchronized
    fun beginFrame() {
        frameIndex += 1L

        pruneStaleState()

        cameras.values.forEach { camera ->
            val age =
                if (camera.lastFrameIndex == Long.MIN_VALUE) {
                    Int.MAX_VALUE
                } else {
                    (frameIndex - camera.lastFrameIndex)
                        .coerceAtLeast(0L)
                        .coerceAtMost(Int.MAX_VALUE.toLong())
                        .toInt()
                }

            if (age >= config.staleAfterFrames) {
                camera.jointStates.values.forEach { joint ->
                    joint.staleFrames += 1

                    val recovery =
                        config.recoveryAlpha *
                            tuning.staleRecovery *
                            (1.0 - tuning.adaptivity * 0.40)

                    joint.trust =
                        (
                            joint.trust +
                                (1.0 - joint.trust) * recovery
                        ).coerceIn(
                            config.minimumWeight,
                            1.0,
                        )
                }

                val cameraRecovery =
                    config.recoveryAlpha *
                        tuning.staleRecovery

                camera.trust =
                    (
                        camera.trust +
                            (1.0 - camera.trust) *
                                cameraRecovery
                    ).coerceIn(
                        config.minimumWeight,
                        1.0,
                    )
            }
        }
    }

    @Synchronized
    fun weight(
        deviceId: DeviceId,
        jointId: Int,
        observationConfidence: Double,
    ): Double {
        val base =
            observationConfidence
                .coerceIn(
                    config.confidenceFloor,
                    1.0,
                )

        val camera =
            cameras[deviceId]
                ?: return base

        val joint =
            camera.jointStates[jointId]
                ?: return (
                    base *
                        lerp(
                            1.0,
                            camera.trust,
                            tuning.adaptivity,
                        )
                ).coerceIn(
                    base * config.minimumWeight,
                    base * config.maximumWeight,
                )

        val stalePenalty =
            if (joint.staleFrames <= 0) {
                1.0
            } else {
                (
                    0.50 +
                        0.50 * exp(
                            -joint.staleFrames *
                                config.staleDecayPerFrame,
                        )
                    ).coerceIn(
                    0.50,
                    1.0,
                )
            }

        val adaptive =
            lerp(
                1.0,
                joint.trust,
                tuning.adaptivity,
            )

        val adaptiveFloor =
            maxOf(
                config.minimumWeight,
                tuning.minimumViewTrust * tuning.adaptivity,
            ).coerceIn(
                config.minimumWeight,
                config.maximumWeight,
            )

        val globalWeight =
            lerp(
                1.0,
                globalRigTrust,
                0.72,
            ).coerceIn(
                config.globalTrustMinimumWeight,
                1.0,
            )

        return (
            base *
                adaptive *
                stalePenalty *
                globalWeight
            ).coerceIn(
                base * adaptiveFloor,
                base * config.maximumWeight,
            )
    }

    @Synchronized
    fun update(
        result: MultiViewReprojectionConsistencyEngine.Result,
        calibrationConfidence: Double = 1.0,
        synchronizationConfidence: Double = 1.0,
    ): Snapshot {
        val calibrationTrust =
            calibrationConfidence.coerceIn(0.0, 1.0)
        val synchronizationTrust =
            synchronizationConfidence.coerceIn(0.0, 1.0)
        val globalTarget =
            (
                calibrationTrust *
                    config.globalCalibrationWeight +
                    synchronizationTrust *
                    config.globalSynchronizationWeight
                ) /
                (
                    config.globalCalibrationWeight +
                        config.globalSynchronizationWeight
                    ).coerceAtLeast(1e-9)

        lastGlobalRigTarget = globalTarget.coerceIn(0.0, 1.0)
        globalRigTrust =
            ema(
                old = globalRigTrust,
                value = lastGlobalRigTarget,
                alpha = config.globalTrustEmaAlpha.coerceIn(0.02, 0.50),
            )

        result.diagnostics.forEach { joint ->
            joint.views.forEach { view ->
                if (!cameras.containsKey(view.deviceId) &&
                    cameras.size >= config.maximumCameras
                ) {
                    evictLeastUsefulCamera()
                }

                val camera =
                    cameras.getOrPut(view.deviceId) {
                        CameraState()
                    }

                if (!camera.jointStates.containsKey(joint.jointId) &&
                    camera.jointStates.size >=
                        config.maximumJointStatesPerCamera
                ) {
                    evictOldestJointState(camera)
                }

                val state =
                    camera.jointStates.getOrPut(joint.jointId) {
                        JointState()
                    }

                val residual =
                    if (view.reprojectionErrorPx.isFinite()) {
                        view.reprojectionErrorPx
                            .coerceIn(
                                0.0,
                                config.hardResidualPx * 4.0,
                            )
                    } else {
                        config.hardResidualPx * 2.0
                    }

                val residualScore =
                    exp(
                        -residual /
                            config.residualScalePx,
                    ).coerceIn(0.0, 1.0)

                val inImageScore =
                    if (view.inImage) 1.0 else 0.0

                val robustScore =
                    view.robustWeight
                        .coerceIn(0.0, 1.0)

                val observationScore =
                    view.observationConfidence
                        .coerceIn(0.0, 1.0)

                val viewScore =
                    (
                        0.58 * residualScore +
                            0.17 * robustScore +
                            0.15 * observationScore +
                            0.10 * inImageScore
                    ).coerceIn(0.0, 1.0)

                val rejectionPenalty =
                    if (view.rejected) 0.55 else 1.0

                val targetTrust =
                    (
                        viewScore *
                            rejectionPenalty
                    ).coerceIn(
                        0.02,
                        1.0,
                    )

                val trustAlpha =
                    (
                        config.trustEmaAlpha *
                            (0.55 +
                                tuning.adaptivity * 0.45)
                    ).coerceIn(
                        0.04,
                        0.42,
                    )

                state.residualEmaPx =
                    ema(
                        old = state.residualEmaPx,
                        value = residual,
                        alpha =
                            config.residualEmaAlpha
                                .coerceIn(0.04, 0.42),
                    )

                val disagreementScale =
                    (
                        0.75 +
                            tuning.disagreementSensitivity * 0.50
                    ).coerceIn(
                        0.75,
                        1.25,
                    )

                state.trust =
                    ema(
                        old = state.trust,
                        value = (
                            targetTrust /
                                disagreementScale
                        ).coerceIn(
                            0.02,
                            1.0,
                        ),
                        alpha = trustAlpha,
                    )

                state.evidenceSamples += 1
                state.lastFrameIndex = frameIndex
                state.staleFrames = 0

                camera.residualEmaPx =
                    ema(
                        old = camera.residualEmaPx,
                        value = residual,
                        alpha =
                            config.residualEmaAlpha
                                .coerceIn(0.04, 0.42),
                    )
                camera.lastFrameIndex = frameIndex
                camera.observationCount += 1

                if (
                    view.rejected ||
                    residual > config.warningResidualPx
                ) {
                    camera.degradedJointCount += 1
                }

                if (
                    view.rejected &&
                    residual >= config.hardResidualPx
                ) {
                    camera.hardFailureJointCount += 1
                }
            }
        }

        cameras.forEach { (_, state) ->
            val usableJointTrust =
                state.jointStates.values
                    .filter {
                        it.evidenceSamples >=
                            config.minimumEvidenceSamples
                    }
                    .map { it.trust }

            if (usableJointTrust.isNotEmpty()) {
                val jointMean =
                    usableJointTrust.average()

                val cameraTarget =
                    (
                        jointMean * 0.78 +
                            (
                                1.0 -
                                    state.residualEmaPx /
                                        config.residualScalePx
                            ).coerceIn(0.0, 1.0) * 0.22
                    ).coerceIn(
                        config.minimumWeight,
                        1.0,
                    )

                state.trust =
                    ema(
                        old = state.trust,
                        value = cameraTarget,
                        alpha =
                            config.trustEmaAlpha
                                .coerceIn(0.04, 0.42),
                    )
            }
        }

        return snapshot()
    }

    @Synchronized
    fun snapshot(): Snapshot {
        val cameraWeights =
            cameras.mapValues { (_, state) ->
                effectiveCameraTrust(state)
            }

        val jointWeights =
            buildMap {
                cameras.forEach { (deviceId, camera) ->
                    camera.jointStates.forEach { (jointId, state) ->
                        put(
                            ViewKey(
                                deviceId = deviceId,
                                jointId = jointId,
                            ),
                            effectiveJointTrust(
                                camera = camera,
                                joint = state,
                            ),
                        )
                    }
                }
            }

        val statuses =
            cameras.map { (deviceId, state) ->
                CameraStatus(
                    deviceId = deviceId,
                    trust = effectiveCameraTrust(state),
                    observationCount = state.observationCount,
                    degradedJointCount =
                        state.degradedJointCount,
                    hardFailureJointCount =
                        state.hardFailureJointCount,
                    meanResidualPx =
                        state.residualEmaPx,
                    state = classify(state),
                )
            }.sortedBy {
                it.deviceId.value
            }

        val degraded =
            statuses.count {
                it.state == State.DEGRADED ||
                    it.state == State.HOLD
            }

        val hold =
            statuses.count {
                it.state == State.HOLD
            }

        val dominant =
            statuses
                .minByOrNull { it.trust }
                ?.deviceId

        val warnings =
            buildList {
                statuses.forEach { status ->
                    when (status.state) {
                        State.HOLD ->
                            add(
                                "camera-\${status.deviceId.value}-evidence-hold",
                            )

                        State.DEGRADED ->
                            add(
                                "camera-\${status.deviceId.value}-evidence-degraded",
                            )

                        else -> Unit
                    }
                }

                if (hold > 0) {
                    add("multi-view-camera-evidence-hold")
                } else if (degraded > 0) {
                    add("multi-view-camera-evidence-degraded")
                }
                if (
                    globalRigTrust < config.globalTrustWarningThreshold ||
                    lastGlobalRigTarget < config.globalTrustWarningThreshold
                ) {
                    add("multi-view-global-rig-evidence-degraded")
                }
            }.distinct().take(12)

        return Snapshot(
            cameraWeights = cameraWeights,
            jointWeights = jointWeights,
            cameraStatuses = statuses,
            degradedCameraCount = degraded,
            holdCameraCount = hold,
            dominantCamera = dominant,
            warnings = warnings,
            globalRigTrust = globalRigTrust,
        )
    }

    private fun pruneStaleState() {
        val cutoff =
            frameIndex -
                config.jointStatePruneAfterFrames.toLong()
        val cameraIterator = cameras.entries.iterator()
        while (cameraIterator.hasNext()) {
            val entry = cameraIterator.next()
            val camera = entry.value
            val jointIterator = camera.jointStates.entries.iterator()
            while (jointIterator.hasNext()) {
                val joint = jointIterator.next().value
                if (joint.lastFrameIndex != Long.MIN_VALUE &&
                    joint.lastFrameIndex < cutoff
                ) {
                    jointIterator.remove()
                }
            }
            if (camera.lastFrameIndex != Long.MIN_VALUE &&
                camera.lastFrameIndex < cutoff &&
                camera.jointStates.isEmpty()
            ) {
                cameraIterator.remove()
            }
        }
    }

    private fun evictLeastUsefulCamera() {
        val victim =
            cameras.entries
                .minByOrNull { entry ->
                    effectiveCameraTrust(entry.value) *
                        (1.0 + ln(entry.value.observationCount.toDouble()).coerceAtLeast(0.0) * 0.05)
                }
                ?.key
        if (victim != null) cameras.remove(victim)
    }

    private fun evictOldestJointState(camera: CameraState) {
        val victim =
            camera.jointStates.entries
                .minByOrNull { (_, state) ->
                    state.lastFrameIndex
                }
                ?.key
        if (victim != null) camera.jointStates.remove(victim)
    }

    @Synchronized
    fun reset() {
        cameras.clear()
        frameIndex = 0L
        globalRigTrust = 1.0
    }

    @Synchronized
    fun cameraTrust(
        deviceId: DeviceId,
    ): Double =
        cameras[deviceId]
            ?.let(::effectiveCameraTrust)
            ?: config.neutralWeight

    @Synchronized
    fun jointTrust(
        deviceId: DeviceId,
        jointId: Int,
    ): Double =
        cameras[deviceId]
            ?.jointStates
            ?.get(jointId)
            ?.let { state ->
                effectiveJointTrust(
                    camera = cameras.getValue(deviceId),
                    joint = state,
                )
            }
            ?: config.neutralWeight

    private fun effectiveCameraTrust(
        state: CameraState,
    ): Double =
        lerp(
            1.0,
            state.trust,
            tuning.adaptivity,
        ).coerceIn(
            maxOf(
                config.minimumWeight,
                tuning.minimumViewTrust * tuning.adaptivity,
            ),
            config.maximumWeight,
        )

    private fun effectiveJointTrust(
        camera: CameraState,
        joint: JointState,
    ): Double {
        val cameraTrust =
            effectiveCameraTrust(camera)

        val stalePenalty =
            if (joint.staleFrames <= 0) {
                1.0
            } else {
                exp(
                    -joint.staleFrames *
                        config.staleDecayPerFrame,
                ).coerceIn(
                    0.15,
                    1.0,
                )
            }

        return (
            lerp(
                1.0,
                joint.trust,
                tuning.adaptivity,
            ) *
                cameraTrust *
                stalePenalty
            ).coerceIn(
                config.minimumWeight,
                config.maximumWeight,
            )
    }

    private fun classify(
        state: CameraState,
    ): State {
        if (state.lastFrameIndex == Long.MIN_VALUE) {
            return State.UNKNOWN
        }

        if (state.trust <= config.cameraHoldThreshold) {
            return State.HOLD
        }

        if (state.trust < config.cameraFailureThreshold) {
            return State.DEGRADED
        }

        return if (
            state.trust > 0.82 &&
            state.degradedJointCount == 0
        ) {
            State.HEALTHY
        } else {
            State.RECOVERING
        }
    }

    private fun ema(
        old: Double,
        value: Double,
        alpha: Double,
    ): Double =
        (
            old *
                (1.0 - alpha.coerceIn(0.0, 1.0)) +
                value *
                alpha.coerceIn(0.0, 1.0)
        ).coerceIn(
            0.0,
            1.0,
        )

    private fun lerp(
        a: Double,
        b: Double,
        t: Double,
    ): Double =
        a +
            (b - a) *
            t.coerceIn(0.0, 1.0)
}
