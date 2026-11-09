package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Deterministic workload scheduler for sustained high-end motion capture.
 *
 * The scheduler protects the motion contract by separating mandatory capture stages from optional
 * fidelity stages. It reacts to queue pressure, GPU/CPU load, thermal headroom, memory pressure,
 * active camera count, and recent stage latency. Optional work can be reduced or deferred, but
 * synchronization and primary motion tracking are never silently sacrificed.
 *
 * This is intentionally policy-only. Native/Render implementations decide how to realize REDUCED
 * or DEFER decisions. The scheduler owns no threads, buffers, coroutines, or GPU handles.
 */
class ProductionCaptureStageScheduler(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class Stage {
        SENSOR_INGESTION,
        TIMESTAMP_ALIGNMENT,
        PRIMARY_2D_TRACKING,
        MULTI_VIEW_ASSOCIATION,
        METRIC_TRIANGULATION,
        TEMPORAL_FUSION,
        FACE_EYE,
        HAND_3D,
        DENSE_SURFACE,
        CONTACT_PHYSICS,
        GLOBAL_4D,
        EXPORT,
    }

    enum class Action {
        RUN_FULL,
        RUN_REDUCED,
        DEFER,
        HOLD,
    }

    data class Config(
        val maximumStages: Int = Stage.entries.size,
        val targetFrameBudgetMilliseconds: Double = 16.67,
        val hardFrameBudgetMilliseconds: Double = 33.34,
        val queueReducedThreshold: Double = 0.62,
        val queueDeferThreshold: Double = 0.82,
        val gpuReducedThreshold: Double = 0.68,
        val gpuDeferThreshold: Double = 0.86,
        val cpuReducedThreshold: Double = 0.72,
        val cpuDeferThreshold: Double = 0.90,
        val memoryReducedThreshold: Double = 0.72,
        val memoryDeferThreshold: Double = 0.90,
        val thermalReducedThreshold: Double = 0.72,
        val thermalDeferThreshold: Double = 0.88,
        val confirmationFrames: Int = 3,
        val recoveryFrames: Int = 4,
        val stageLatencyWindow: Int = 12,
        val maximumDeferredFrames: Int = 4,
    ) {
        init {
            require(maximumStages in 4..Stage.entries.size)
            require(targetFrameBudgetMilliseconds > 0.0)
            require(hardFrameBudgetMilliseconds >= targetFrameBudgetMilliseconds)
            listOf(
                queueReducedThreshold,
                queueDeferThreshold,
                gpuReducedThreshold,
                gpuDeferThreshold,
                cpuReducedThreshold,
                cpuDeferThreshold,
                memoryReducedThreshold,
                memoryDeferThreshold,
                thermalReducedThreshold,
                thermalDeferThreshold,
            ).forEach {
                require(it in 0.0..1.0)
            }
            require(queueDeferThreshold >= queueReducedThreshold)
            require(gpuDeferThreshold >= gpuReducedThreshold)
            require(cpuDeferThreshold >= cpuReducedThreshold)
            require(memoryDeferThreshold >= memoryReducedThreshold)
            require(thermalDeferThreshold >= thermalReducedThreshold)
            require(confirmationFrames >= 1)
            require(recoveryFrames >= 1)
            require(stageLatencyWindow >= 4)
            require(maximumDeferredFrames >= 1)
        }
    }

    data class RuntimePressure(
        val timestampNanos: Long,
        val queuePressure: Double,
        val gpuPressure: Double,
        val cpuPressure: Double,
        val memoryPressure: Double,
        val thermalPressure: Double,
        val estimatedFrameTimeMilliseconds: Double,
        val activeCameraCount: Int,
        val motionContractStrictness: Double = 0.85,
    ) {
        init {
            require(timestampNanos >= 0L)
            listOf(
                queuePressure,
                gpuPressure,
                cpuPressure,
                memoryPressure,
                thermalPressure,
                motionContractStrictness,
            ).forEach {
                require(it.isFinite() && it in 0.0..1.0)
            }
            require(estimatedFrameTimeMilliseconds.isFinite())
            require(activeCameraCount >= 0)
        }
    }

    data class StageTelemetry(
        val lastLatencyMilliseconds: Double = 0.0,
        val recentMeanLatencyMilliseconds: Double = 0.0,
        val recentP95LatencyMilliseconds: Double = 0.0,
        val lastAction: Action = Action.RUN_FULL,
        val deferredFrames: Int = 0,
    )

    data class StageDecision(
        val stage: Stage,
        val action: Action,
        val qualityScale: Double,
        val budgetShare: Double,
        val reason: String,
        val mandatory: Boolean,
        val deferredFrames: Int,
    )

    data class FrameDecision(
        val timestampNanos: Long,
        val actions: Map<Stage, StageDecision>,
        val totalBudgetShare: Double,
        val predictedFrameTimeMilliseconds: Double,
        val coreMotionProtected: Boolean,
        val fidelityReduced: Boolean,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val captureSafe: Boolean
            get() =
                coreMotionProtected &&
                    actions[Stage.SENSOR_INGESTION]?.action != Action.HOLD &&
                    actions[Stage.TIMESTAMP_ALIGNMENT]?.action != Action.HOLD &&
                    actions[Stage.PRIMARY_2D_TRACKING]?.action != Action.HOLD

        val optionalStagesDeferred: Int
            get() =
                actions.values.count {
                    !it.mandatory &&
                        it.action == Action.DEFER
                }
    }

    private data class StageState(
        var telemetry: StageTelemetry = StageTelemetry(),
        var pressureCount: Int = 0,
        var recoveryCount: Int = 0,
    )

    private val mandatoryStages = setOf(
        Stage.SENSOR_INGESTION,
        Stage.TIMESTAMP_ALIGNMENT,
        Stage.PRIMARY_2D_TRACKING,
        Stage.MULTI_VIEW_ASSOCIATION,
        Stage.METRIC_TRIANGULATION,
        Stage.TEMPORAL_FUSION,
    )

    private val baseWeights = mapOf(
        Stage.SENSOR_INGESTION to 0.14,
        Stage.TIMESTAMP_ALIGNMENT to 0.10,
        Stage.PRIMARY_2D_TRACKING to 0.14,
        Stage.MULTI_VIEW_ASSOCIATION to 0.08,
        Stage.METRIC_TRIANGULATION to 0.13,
        Stage.TEMPORAL_FUSION to 0.12,
        Stage.FACE_EYE to 0.06,
        Stage.HAND_3D to 0.06,
        Stage.DENSE_SURFACE to 0.08,
        Stage.CONTACT_PHYSICS to 0.03,
        Stage.GLOBAL_4D to 0.04,
        Stage.EXPORT to 0.02,
    )

    private val states =
        Stage.entries.associateWith { StageState() }.toMutableMap()

    private var lastDecision: FrameDecision? = null

    @Synchronized
    fun recordStageLatency(
        stage: Stage,
        latencyMilliseconds: Double,
    ) {
        require(latencyMilliseconds.isFinite())
        require(latencyMilliseconds >= 0.0)

        val state = states[stage] ?: return
        val old = state.telemetry
        val mean =
            if (old.recentMeanLatencyMilliseconds <= 0.0) {
                latencyMilliseconds
            } else {
                old.recentMeanLatencyMilliseconds * 0.75 +
                    latencyMilliseconds * 0.25
            }

        val pseudoP95 =
            max(
                latencyMilliseconds,
                old.recentP95LatencyMilliseconds * 0.82 +
                    latencyMilliseconds * 0.18,
            )

        state.telemetry = old.copy(
            lastLatencyMilliseconds = latencyMilliseconds,
            recentMeanLatencyMilliseconds = mean,
            recentP95LatencyMilliseconds = pseudoP95,
        )
    }

    @Synchronized
    fun decide(
        pressure: RuntimePressure,
    ): FrameDecision {
        val stageLimit =
            min(
                config.maximumStages,
                Stage.entries.size,
            )

        val severe =
            pressure.queuePressure >= config.queueDeferThreshold ||
                pressure.gpuPressure >= config.gpuDeferThreshold ||
                pressure.cpuPressure >= config.cpuDeferThreshold ||
                pressure.memoryPressure >= config.memoryDeferThreshold ||
                pressure.thermalPressure >= config.thermalDeferThreshold ||
                pressure.estimatedFrameTimeMilliseconds >=
                config.hardFrameBudgetMilliseconds

        val reduced =
            pressure.queuePressure >= config.queueReducedThreshold ||
                pressure.gpuPressure >= config.gpuReducedThreshold ||
                pressure.cpuPressure >= config.cpuReducedThreshold ||
                pressure.memoryPressure >= config.memoryReducedThreshold ||
                pressure.thermalPressure >= config.thermalReducedThreshold ||
                pressure.estimatedFrameTimeMilliseconds >
                config.targetFrameBudgetMilliseconds

        val decisions = linkedMapOf<Stage, StageDecision>()
        var totalBudget = 0.0
        var predictedFrameTime =
            pressure.estimatedFrameTimeMilliseconds
        var hysteresisConfirming = false
        var recoveryHysteresisActive = false

        Stage.entries.take(stageLimit).forEach { stage ->
            val state = states[stage] ?: StageState()
            val mandatory = stage in mandatoryStages
            val weight = baseWeights[stage] ?: 0.05
            val stagePressure = pressureForStage(stage, pressure)
            val stageSevere =
                stagePressure >= config.deferThreshold(stage) ||
                    pressure.estimatedFrameTimeMilliseconds >=
                    config.hardFrameBudgetMilliseconds
            val stageReduced =
                stagePressure >= config.reducedThreshold(stage) ||
                    pressure.estimatedFrameTimeMilliseconds >
                    config.targetFrameBudgetMilliseconds

            if (stageSevere) {
                state.pressureCount =
                    config.confirmationFrames
                state.recoveryCount = 0
            } else if (stageReduced) {
                state.pressureCount =
                    (state.pressureCount + 1)
                        .coerceAtMost(config.confirmationFrames)
                state.recoveryCount = 0
            } else {
                state.recoveryCount =
                    (state.recoveryCount + 1)
                        .coerceAtMost(config.recoveryFrames)
                state.pressureCount = 0
            }

            val pressureConfirmed =
                stageSevere ||
                    state.pressureCount >= config.confirmationFrames
            val recoveryConfirmed =
                !stageReduced &&
                    !stageSevere &&
                    state.recoveryCount >= config.recoveryFrames

            var action = Action.RUN_FULL
            var qualityScale = 1.0
            var reason = "within-budget"

            if (mandatory) {
                when {
                    stageSevere &&
                        stage in setOf(
                            Stage.TIMESTAMP_ALIGNMENT,
                            Stage.PRIMARY_2D_TRACKING,
                            Stage.METRIC_TRIANGULATION,
                        ) -> {
                        action = Action.RUN_REDUCED
                        qualityScale =
                            (1.0 -
                                stagePressure * 0.22)
                                .coerceIn(0.72, 1.0)
                        reason = "protect-core-motion-contract"
                    }

                    stageReduced -> {
                        action = Action.RUN_REDUCED
                        qualityScale =
                            (1.0 -
                                stagePressure * 0.18)
                                .coerceIn(0.78, 1.0)
                        reason = "protect-core-throughput"
                    }
                }
            } else {
                when {
                    stageSevere -> {
                        action =
                            if (
                                state.telemetry.deferredFrames >=
                                config.maximumDeferredFrames ||
                                pressure.motionContractStrictness > 0.92 &&
                                stage == Stage.GLOBAL_4D
                            ) {
                                Action.RUN_REDUCED
                            } else {
                                Action.DEFER
                            }
                        qualityScale =
                            if (action == Action.DEFER) {
                                0.0
                            } else {
                                0.55
                            }
                        reason = "resource-pressure-shed-optional"
                    }

                    stageReduced -> {
                        action = Action.RUN_REDUCED
                        qualityScale = (
                            1.0 -
                                stagePressure * 0.50
                            ).coerceIn(0.42, 0.90)
                        reason = "resource-pressure-reduced"
                    }
                }
            }

            if (
                action == Action.DEFER &&
                state.telemetry.deferredFrames >=
                config.maximumDeferredFrames
            ) {
                action = Action.RUN_REDUCED
                qualityScale = 0.52
                reason = "defer-budget-reached"
            }

            if (!mandatory) {
                val previousAction =
                    state.telemetry.lastAction

                if (
                    stageReduced &&
                    !stageSevere &&
                    !pressureConfirmed &&
                    previousAction == Action.RUN_FULL
                ) {
                    action = Action.RUN_FULL
                    qualityScale = 1.0
                    reason = "pressure-confirmation-hysteresis"
                }

                if (
                    !stageReduced &&
                    !stageSevere &&
                    previousAction != Action.RUN_FULL &&
                    !recoveryConfirmed
                ) {
                    action = Action.RUN_REDUCED
                    qualityScale =
                        min(
                            qualityScale.coerceAtLeast(0.55),
                            0.82,
                        )
                    reason = "recovery-hysteresis-ramp"
                }
            if (
                stageReduced &&
                !stageSevere &&
                !pressureConfirmed
            ) {
                hysteresisConfirming = true
            }
            if (
                !stageReduced &&
                !stageSevere &&
                previousAction != Action.RUN_FULL &&
                !recoveryConfirmed
            ) {
                recoveryHysteresisActive = true
            }
            }

            val latencyPressure =
                (
                    state.telemetry.recentP95LatencyMilliseconds /
                        config.targetFrameBudgetMilliseconds
                    ).coerceIn(0.0, 2.0)

            if (
                action == Action.RUN_FULL &&
                latencyPressure > 1.15 &&
                !mandatory
            ) {
                action = Action.RUN_REDUCED
                qualityScale =
                    min(qualityScale, 0.68)
                reason = "historical-stage-latency-high"
            }

            if (action == Action.DEFER) {
                state.telemetry =
                    state.telemetry.copy(
                        lastAction = Action.DEFER,
                        deferredFrames =
                            (state.telemetry.deferredFrames + 1)
                                .coerceAtMost(
                                    config.maximumDeferredFrames,
                                ),
                    )
            } else {
                state.telemetry =
                    state.telemetry.copy(
                        lastAction = action,
                        deferredFrames = 0,
                    )
            }

            decisions[stage] = StageDecision(
                stage = stage,
                action = action,
                qualityScale = qualityScale,
                budgetShare = weight *
                    if (action == Action.DEFER) 0.0 else 1.0,
                reason = reason,
                mandatory = mandatory,
                deferredFrames =
                    state.telemetry.deferredFrames,
            )

            totalBudget += decisions[stage]!!.budgetShare
            predictedFrameTime +=
                state.telemetry.lastLatencyMilliseconds *
                    (
                        if (action == Action.RUN_FULL) 1.0
                        else if (action == Action.RUN_REDUCED) 0.58
                        else 0.0
                    )
        }

        val protectedCore =
            mandatoryStages.all { stage ->
                val decision = decisions[stage]
                decision != null &&
                    decision.action != Action.HOLD
            }

        val fidelityReduced =
            decisions.values.any {
                !it.mandatory &&
                    it.action != Action.RUN_FULL
            }

        val warnings = buildList {
            if (severe) {
                add("capture-stage-pressure-severe")
            } else if (reduced) {
                add("capture-stage-pressure-elevated")
            }
            if (fidelityReduced) {
                add("optional-fidelity-stages-reduced")
            }
            if (hysteresisConfirming) {
                add("capture-stage-pressure-hysteresis-confirming")
            }
            if (recoveryHysteresisActive) {
                add("capture-stage-recovery-hysteresis-active")
            }
            if (!protectedCore) {
                add("core-motion-contract-not-protected")
            }
            if (pressure.estimatedFrameTimeMilliseconds >
                config.hardFrameBudgetMilliseconds
            ) {
                add("frame-time-budget-hard-overrun")
            }
            if (pressure.activeCameraCount >
                8 &&
                pressure.gpuPressure > 0.78
            ) {
                add("high-camera-count-gpu-pressure")
            }
        }

        val recommendations = buildList {
            if (pressure.thermalPressure >=
                config.thermalReducedThreshold
            ) {
                add("Allow thermal recovery before restoring dense surface stages.")
            }
            if (pressure.memoryPressure >=
                config.memoryReducedThreshold
            ) {
                add("Release transient reconstruction resources before restoring optional stages.")
            }
            if (pressure.queuePressure >=
                config.queueReducedThreshold
            ) {
                add("Drain capture backlog before increasing optional inference frequency.")
            }
            if (pressure.gpuPressure >=
                config.gpuReducedThreshold
            ) {
                add("Preserve timestamp alignment and primary motion while reducing expensive rendering or surface work.")
            }
        }.distinct().take(6)

        val decision = FrameDecision(
            timestampNanos = pressure.timestampNanos,
            actions = decisions,
            totalBudgetShare = totalBudget.coerceIn(0.0, 1.0),
            predictedFrameTimeMilliseconds =
                predictedFrameTime.coerceAtLeast(0.0),
            coreMotionProtected = protectedCore,
            fidelityReduced = fidelityReduced,
            warnings = warnings,
            recommendations = recommendations,
        )

        lastDecision = decision
        return decision
    }

    private fun pressureForStage(
        stage: Stage,
        pressure: RuntimePressure,
    ): Double =
        when (stage) {
            Stage.SENSOR_INGESTION,
            Stage.TIMESTAMP_ALIGNMENT,
            Stage.PRIMARY_2D_TRACKING,
            Stage.MULTI_VIEW_ASSOCIATION,
            Stage.METRIC_TRIANGULATION,
            Stage.TEMPORAL_FUSION -> maxOf(
                pressure.queuePressure,
                pressure.cpuPressure,
                pressure.memoryPressure,
                pressure.thermalPressure,
            )

            Stage.FACE_EYE -> maxOf(
                pressure.gpuPressure,
                pressure.cpuPressure,
                pressure.memoryPressure,
                pressure.thermalPressure,
            )

            Stage.HAND_3D -> maxOf(
                pressure.gpuPressure,
                pressure.cpuPressure,
                pressure.memoryPressure,
            )

            Stage.DENSE_SURFACE -> maxOf(
                pressure.gpuPressure,
                pressure.memoryPressure,
                pressure.thermalPressure,
            )

            Stage.CONTACT_PHYSICS -> maxOf(
                pressure.cpuPressure,
                pressure.memoryPressure,
            )

            Stage.GLOBAL_4D -> maxOf(
                pressure.cpuPressure,
                pressure.gpuPressure,
                pressure.memoryPressure,
                pressure.thermalPressure,
            )

            Stage.EXPORT -> maxOf(
                pressure.queuePressure,
                pressure.memoryPressure,
            )
        }

    private fun Config.reducedThreshold(stage: Stage): Double =
        when (stage) {
            Stage.SENSOR_INGESTION,
            Stage.TIMESTAMP_ALIGNMENT,
            Stage.PRIMARY_2D_TRACKING,
            Stage.MULTI_VIEW_ASSOCIATION,
            Stage.METRIC_TRIANGULATION,
            Stage.TEMPORAL_FUSION ->
                minOf(queueReducedThreshold, cpuReducedThreshold)

            Stage.FACE_EYE,
            Stage.HAND_3D ->
                minOf(gpuReducedThreshold, cpuReducedThreshold)

            Stage.DENSE_SURFACE ->
                minOf(gpuReducedThreshold, memoryReducedThreshold)

            Stage.CONTACT_PHYSICS ->
                minOf(cpuReducedThreshold, memoryReducedThreshold)

            Stage.GLOBAL_4D ->
                minOf(
                    cpuReducedThreshold,
                    gpuReducedThreshold,
                )

            Stage.EXPORT ->
                minOf(
                    queueReducedThreshold,
                    memoryReducedThreshold,
                )
        }

    private fun Config.deferThreshold(stage: Stage): Double =
        when (stage) {
            Stage.SENSOR_INGESTION,
            Stage.TIMESTAMP_ALIGNMENT,
            Stage.PRIMARY_2D_TRACKING,
            Stage.MULTI_VIEW_ASSOCIATION,
            Stage.METRIC_TRIANGULATION,
            Stage.TEMPORAL_FUSION ->
                minOf(queueDeferThreshold, cpuDeferThreshold)

            Stage.FACE_EYE,
            Stage.HAND_3D ->
                minOf(gpuDeferThreshold, cpuDeferThreshold)

            Stage.DENSE_SURFACE ->
                minOf(gpuDeferThreshold, memoryDeferThreshold)

            Stage.CONTACT_PHYSICS ->
                minOf(cpuDeferThreshold, memoryDeferThreshold)

            Stage.GLOBAL_4D ->
                minOf(
                    cpuDeferThreshold,
                    gpuDeferThreshold,
                )

            Stage.EXPORT ->
                minOf(
                    queueDeferThreshold,
                    memoryDeferThreshold,
                )
        }

    @Synchronized
    fun latestDecision(): FrameDecision? = lastDecision

    @Synchronized
    fun stageTelemetry(stage: Stage): StageTelemetry =
        states[stage]?.telemetry ?: StageTelemetry()

    @Synchronized
    fun reset() {
        states.values.forEach {
            it.telemetry = StageTelemetry()
            it.pressureCount = 0
            it.recoveryCount = 0
        }
        lastDecision = null
    }

    @Synchronized
    override fun close() {
        reset()
    }
}
