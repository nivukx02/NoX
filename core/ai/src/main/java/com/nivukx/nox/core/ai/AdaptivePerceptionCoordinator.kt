package com.nivukx.nox.core.ai

import kotlin.math.max
import kotlin.math.min

/**
 * Per-stage inference budget controller.
 *
 * This controller is intentionally model-agnostic. It does not own camera frames and never decides
 * whether a captured frame is recorded. It only decides which perception stages should consume the
 * current analysis frame and how old a cached result may be before it is considered stale.
 *
 * Design goals:
 * - preserve full-rate capture while reducing redundant neural work;
 * - increase temporal sampling when subject motion or occlusion increases;
 * - decrease expensive contour/object/hand work under thermal or memory pressure;
 * - make every downgrade explicit through a reason string;
 * - keep stage state deterministic and testable without Android hardware.
 */
class AdaptivePerceptionCoordinator(
    private val captureFps: Int = 30,
    private val minimumFps: Int = 5,
    private val maxCacheAgeNanos: Long = 320_000_000L,
) {

    enum class Stage {
        POSE,
        FACE,
        EYE,
        HAND,
        OBJECT,
        CONTOUR,
        SEGMENTATION,
    }

    data class SystemLoad(
        val thermalScale: Double = 1.0,
        val memoryScale: Double = 1.0,
        val gpuScale: Double = 1.0,
        val inferenceScale: Double = 1.0,
    ) {
        val combinedScale: Double
            get() = min(
                thermalScale.coerceIn(0.15, 1.0),
                min(
                    memoryScale.coerceIn(0.15, 1.0),
                    min(
                        gpuScale.coerceIn(0.15, 1.0),
                        inferenceScale.coerceIn(0.15, 1.0),
                    ),
                ),
            )
    }

    data class StageDecision(
        val stage: Stage,
        val process: Boolean,
        val requestedIntervalFrames: Int,
        val effectiveIntervalFrames: Int,
        val cacheMaxAgeNanos: Long,
        val reason: String,
    )

    data class Plan(
        val timestampNanos: Long,
        val frameIndex: Long,
        val motionClass: MotionCaptureEngine.MotionClass,
        val motionReadiness: Double,
        val systemLoad: SystemLoad,
        val decisions: Map<Stage, StageDecision>,
        val overallInferenceScale: Double,
        val reason: String,
    ) {
        fun shouldProcess(stage: Stage): Boolean =
            decisions[stage]?.process == true

        fun maxAge(stage: Stage): Long =
            decisions[stage]?.cacheMaxAgeNanos ?: 0L
    }

    data class Telemetry(
        val frameIndex: Long,
        val skippedFrames: Long,
        val processedByStage: Map<Stage, Long>,
        val skippedByStage: Map<Stage, Long>,
        val ewmaStageMillis: Map<Stage, Double>,
    )

    private data class StageState(
        var lastProcessedTimestampNanos: Long = Long.MIN_VALUE,
        var lastProcessedFrame: Long = Long.MIN_VALUE,
        var ewmaMillis: Double = 0.0,
        var processed: Long = 0L,
        var skipped: Long = 0L,
    )

    private val states = Stage.entries.associateWith { StageState() }.toMutableMap()
    private var frameIndex = 0L
    private var skippedFrames = 0L
    private var previousMotionClass = MotionCaptureEngine.MotionClass.STATIC
    private var previousReadiness = 0.0
    private var load = SystemLoad()
    private var segmentationEnabled = true
    private var segmentationCadenceFrames = 4

    @Synchronized
    fun configureSegmentation(
        enabled: Boolean,
        cadenceFrames: Int,
    ) {
        segmentationEnabled = enabled
        segmentationCadenceFrames =
            cadenceFrames.coerceIn(1, 120)
    }

    private fun latencyBudgetMillis(stage: Stage): Double =
        when (stage) {
            Stage.POSE -> 12.0
            Stage.FACE -> 10.0
            Stage.EYE -> 8.0
            Stage.HAND -> 9.0
            Stage.OBJECT -> 7.0
            Stage.CONTOUR -> 12.0
            Stage.SEGMENTATION -> 75.0
        }

    private fun latencyPressure(stage: Stage): Double {
        val ewma = states.getValue(stage).ewmaMillis
        if (ewma <= 0.0) return 0.0

        return (
            ewma /
                latencyBudgetMillis(stage)
            ).coerceIn(0.0, 4.0)
    }

    @Synchronized
    fun updateSystemLoad(
        thermalScale: Double,
        memoryScale: Double,
        gpuScale: Double = 1.0,
        inferenceScale: Double = 1.0,
    ) {
        load = SystemLoad(
            thermalScale = thermalScale,
            memoryScale = memoryScale,
            gpuScale = gpuScale,
            inferenceScale = inferenceScale,
        )
    }

    @Synchronized
    fun beginFrame(
        timestampNanos: Long,
        motionQuality: MotionCaptureEngine.Quality,
    ): Plan {
        frameIndex += 1L
        previousMotionClass = motionQuality.motionClass
        previousReadiness = motionQuality.captureReadiness

        val loadScale = load.combinedScale
        val motionBoost = when (motionQuality.motionClass) {
            MotionCaptureEngine.MotionClass.EXTREME -> 1.45
            MotionCaptureEngine.MotionClass.FAST -> 1.25
            MotionCaptureEngine.MotionClass.NORMAL -> 1.05
            MotionCaptureEngine.MotionClass.MICRO_MOTION -> 0.90
            MotionCaptureEngine.MotionClass.STATIC -> 0.78
        }

        val occlusionBoost = 1.0 +
            motionQuality.occlusionRatio.coerceIn(0.0, 1.0) * 0.35
        val readinessBoost = when {
            motionQuality.captureReadiness >= 0.82 -> 1.10
            motionQuality.captureReadiness >= 0.60 -> 1.03
            motionQuality.captureReadiness <= 0.28 -> 0.84
            else -> 1.0
        }

        val activityScale = (
            motionBoost *
                occlusionBoost *
                readinessBoost *
                loadScale
            ).coerceIn(0.25, 1.65)

        val baseIntervals = when {
            activityScale >= 1.25 -> mapOf(
                Stage.POSE to 1,
                Stage.FACE to 1,
                Stage.EYE to 1,
                Stage.HAND to 1,
                Stage.OBJECT to 1,
                Stage.CONTOUR to 2,
                Stage.SEGMENTATION to 6,
            )

            activityScale >= 0.92 -> mapOf(
                Stage.POSE to 1,
                Stage.FACE to 1,
                Stage.EYE to 1,
                Stage.HAND to 1,
                Stage.OBJECT to 2,
                Stage.CONTOUR to 2,
                Stage.SEGMENTATION to 4,
            )

            activityScale >= 0.65 -> mapOf(
                Stage.POSE to 1,
                Stage.FACE to 1,
                Stage.EYE to 1,
                Stage.HAND to 2,
                Stage.OBJECT to 2,
                Stage.CONTOUR to 3,
                Stage.SEGMENTATION to 3,
            )

            else -> mapOf(
                Stage.POSE to 1,
                Stage.FACE to 2,
                Stage.EYE to 2,
                Stage.HAND to 3,
                Stage.OBJECT to 3,
                Stage.CONTOUR to 4,
                Stage.SEGMENTATION to 8,
            )
        }

        val effectiveIntervals = baseIntervals.mapValues { (stage, interval) ->
            val pressurePenalty = when {
                loadScale < 0.42 -> 2
                loadScale < 0.65 -> 1
                else -> 0
            }
            val stageLatencyPenalty = when {
                latencyPressure(stage) >= 2.0 -> 2
                latencyPressure(stage) >= 1.35 -> 1
                else -> 0
            }
            when {
                interval == 1 &&
                    stageLatencyPenalty == 0 &&
                    pressurePenalty == 0 -> 1
                else ->
                    (
                        interval +
                            pressurePenalty +
                            stageLatencyPenalty
                        ).coerceAtMost(
                            (captureFps / minimumFps).coerceAtLeast(1),
                        )
            }
        }

        val decisions = Stage.entries.associateWith { stage ->
            val interval =
                effectiveIntervals.getValue(stage)
            val configuredInterval =
                if (stage == Stage.SEGMENTATION) {
                    maxOf(
                        interval,
                        segmentationCadenceFrames,
                    )
                } else {
                    interval
                }
            val stageState = states.getValue(stage)
            val cadenceDue =
                stageState.lastProcessedFrame ==
                    Long.MIN_VALUE ||
                    frameIndex -
                        stageState.lastProcessedFrame >=
                        configuredInterval

            val process =
                if (stage == Stage.SEGMENTATION) {
                    segmentationEnabled &&
                        cadenceDue
                } else {
                    cadenceDue
                }

            if (!process) {
                stageState.skipped += 1L
                skippedFrames += 1L
            } else {
                stageState.processed += 1L
                stageState.lastProcessedFrame = frameIndex
                stageState.lastProcessedTimestampNanos = timestampNanos
            }

            val maxAge = when (stage) {
                Stage.POSE -> 120_000_000L
                Stage.FACE -> 180_000_000L
                Stage.EYE -> 160_000_000L
                Stage.HAND -> 220_000_000L
                Stage.OBJECT -> 240_000_000L
                Stage.CONTOUR -> 300_000_000L
                Stage.SEGMENTATION -> 280_000_000L
            }.coerceAtMost(maxCacheAgeNanos)

            StageDecision(
                stage = stage,
                process = process,
                requestedIntervalFrames = baseIntervals.getValue(stage),
                effectiveIntervalFrames =
                    configuredInterval,
                cacheMaxAgeNanos = maxAge,
                reason = when {
                    process &&
                        interval == 1 &&
                        latencyPressure(stage) < 1.35 ->
                        "full-rate"
                    process &&
                        latencyPressure(stage) >= 1.35 ->
                        "stage-latency-refresh"
                    process -> "adaptive-refresh"
                    !process &&
                        latencyPressure(stage) >= 1.35 ->
                        "cached-stage-latency"
                    stage == Stage.SEGMENTATION &&
                        !segmentationEnabled ->
                        "disabled-by-capture-policy"
                    else -> "cached-temporal"
                },
            )
        }

        val globalReason = when {
            load.thermalScale < 0.50 -> "thermal-budget"
            load.memoryScale < 0.50 -> "memory-budget"
            motionQuality.motionClass == MotionCaptureEngine.MotionClass.EXTREME -> "extreme-motion"
            motionQuality.motionClass == MotionCaptureEngine.MotionClass.FAST -> "fast-motion"
            activityScale < 0.70 -> "low-activity"
            else -> "balanced"
        }

        return Plan(
            timestampNanos = timestampNanos,
            frameIndex = frameIndex,
            motionClass = motionQuality.motionClass,
            motionReadiness = motionQuality.captureReadiness,
            systemLoad = load,
            decisions = decisions,
            overallInferenceScale = activityScale,
            reason = globalReason,
        )
    }

    @Synchronized
    fun recordStageDuration(
        stage: Stage,
        elapsedMillis: Double,
    ) {
        val state = states.getValue(stage)
        val duration = elapsedMillis.coerceAtLeast(0.0)
        state.ewmaMillis =
            if (state.ewmaMillis <= 0.0) duration
            else state.ewmaMillis * 0.85 + duration * 0.15
    }

    @Synchronized
    fun telemetry(): Telemetry =
        Telemetry(
            frameIndex = frameIndex,
            skippedFrames = skippedFrames,
            processedByStage = states.mapValues { it.value.processed },
            skippedByStage = states.mapValues { it.value.skipped },
            ewmaStageMillis = states.mapValues { it.value.ewmaMillis },
        )

    @Synchronized
    fun reset() {
        frameIndex = 0L
        skippedFrames = 0L
        previousMotionClass = MotionCaptureEngine.MotionClass.STATIC
        previousReadiness = 0.0
        Stage.entries.forEach { states.getValue(it).let {
            it.lastProcessedTimestampNanos = Long.MIN_VALUE
            it.lastProcessedFrame = Long.MIN_VALUE
            it.ewmaMillis = 0.0
            it.processed = 0L
            it.skipped = 0L
        } }
    }
}
