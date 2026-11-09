package com.nivukx.nox.core.reconstruction

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Stateful runtime monitor for detecting persistent disagreement between a calibrated camera rig
 * and the geometry observed during capture.
 *
 * This is intentionally a residual monitor, not a fake online bundle-adjustment system. It uses
 * measured multi-view reprojection evidence and applies hysteresis so transient blur, motion,
 * synchronization loss, or timing degradation are not mistaken for physical rig movement.
 */
class LiveRigGeometryDriftEngine(
    private val config: Config = Config(),
) {

    enum class State {
        UNKNOWN,
        STABLE,
        WATCH,
        AMBIGUOUS,
        REVALIDATE,
        BLOCKED,
    }

    data class Config(
        val watchResidualPx: Double = 3.5,
        val revalidateResidualPx: Double = 5.5,
        val blockResidualPx: Double = 9.0,
        val watchP95Px: Double = 8.0,
        val revalidateP95Px: Double = 13.0,
        val blockP95Px: Double = 20.0,
        val minimumInlierRatio: Double = 0.62,
        val watchInlierRatio: Double = 0.76,
        val minimumMultiViewScore: Double = 0.50,
        val watchMultiViewScore: Double = 0.70,
        val minimumCameraCount: Int = 2,
        val historyAlpha: Double = 0.14,
        val growthWatchRatio: Double = 2.4,
        val growthRevalidateRatio: Double = 4.0,
        val persistentWatchFrames: Int = 7,
        val persistentRevalidateFrames: Int = 12,
        val persistentBlockFrames: Int = 20,
        val recoveryFrames: Int = 14,
        val ambiguityCameraMotionRisk: Double = 0.66,
        val ambiguitySyncConfidence: Double = 0.58,
        val ambiguityTimingConfidence: Double = 0.58,
        val ambiguityReleaseFrames: Int = 6,
        val maximumClockSkewMillis: Double = 6.0,
    ) {
        init {
            require(watchResidualPx > 0.0)
            require(revalidateResidualPx > watchResidualPx)
            require(blockResidualPx > revalidateResidualPx)
            require(watchP95Px > 0.0)
            require(revalidateP95Px > watchP95Px)
            require(blockP95Px > revalidateP95Px)
            require(minimumInlierRatio in 0.0..1.0)
            require(watchInlierRatio >= minimumInlierRatio)
            require(minimumMultiViewScore in 0.0..1.0)
            require(watchMultiViewScore >= minimumMultiViewScore)
            require(minimumCameraCount >= 2)
            require(historyAlpha in 0.0..1.0)
            require(growthWatchRatio > 1.0)
            require(growthRevalidateRatio > growthWatchRatio)
            require(persistentWatchFrames >= 1)
            require(persistentRevalidateFrames >= persistentWatchFrames)
            require(persistentBlockFrames >= persistentRevalidateFrames)
            require(recoveryFrames >= 1)
            require(ambiguityCameraMotionRisk in 0.0..1.0)
            require(ambiguitySyncConfidence in 0.0..1.0)
            require(ambiguityTimingConfidence in 0.0..1.0)
            require(ambiguityReleaseFrames >= 1)
            require(maximumClockSkewMillis > 0.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val calibrationReprojectionErrorPx: Double,
        val liveMedianResidualPx: Double,
        val liveP95ResidualPx: Double,
        val liveMaximumResidualPx: Double,
        val liveInlierRatio: Double,
        val liveMultiViewScore: Double,
        val cameraCount: Int,
        val clockSkewMillis: Double,
        val cameraMotionRisk: Double = 0.0,
        val synchronizationConfidence: Double = 1.0,
        val captureTimingConfidence: Double = 1.0,
    )

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val accepted: Boolean,
        val driftScore: Double,
        val baselineResidualPx: Double,
        val smoothedMedianResidualPx: Double,
        val smoothedP95ResidualPx: Double,
        val residualGrowthRatio: Double,
        val inlierRatio: Double,
        val multiViewScore: Double,
        val ambiguityScore: Double,
        val persistentWatchFrames: Int,
        val persistentRevalidateFrames: Int,
        val persistentBlockFrames: Int,
        val recoveryFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val requiresRecalibration: Boolean
            get() = state == State.REVALIDATE || state == State.BLOCKED

        val fatal: Boolean
            get() = state == State.BLOCKED

        val bottleneck: String
            get() = when (state) {
                State.UNKNOWN -> "INITIALIZING"
                State.STABLE -> "NONE"
                State.WATCH -> "GEOMETRY_RESIDUAL"
                State.AMBIGUOUS -> "CAPTURE_AMBIGUITY"
                State.REVALIDATE -> "CALIBRATION_DRIFT"
                State.BLOCKED -> "CALIBRATION_INVALID"
            }
    }

    private data class History(
        var timestampNanos: Long,
        var residualEmaPx: Double,
        var p95EmaPx: Double,
        var inlierEma: Double,
        var scoreEma: Double,
        var lastState: State,
        var watchFrames: Int,
        var revalidateFrames: Int,
        var blockFrames: Int,
        var recoveryFrames: Int,
        var clearAmbiguityFrames: Int,
    )

    private var history: History? = null

    @Synchronized
    fun update(input: Input): Result {
        val median = sanitize(input.liveMedianResidualPx)
        val p95 = sanitize(input.liveP95ResidualPx)
        val maximum = sanitize(input.liveMaximumResidualPx)
        val calibration = sanitize(input.calibrationReprojectionErrorPx)
        val inlier = input.liveInlierRatio.coerceIn(0.0, 1.0)
        val score = input.liveMultiViewScore.coerceIn(0.0, 1.0)
        val clockHealth = (
            1.0 -
                input.clockSkewMillis.coerceAtLeast(0.0) /
                    config.maximumClockSkewMillis
        ).coerceIn(0.0, 1.0)

        val ambiguityScore = (
            0.40 * input.cameraMotionRisk.coerceIn(0.0, 1.0) +
                0.30 * (1.0 - input.synchronizationConfidence.coerceIn(0.0, 1.0)) +
                0.20 * (1.0 - input.captureTimingConfidence.coerceIn(0.0, 1.0)) +
                0.10 * (1.0 - clockHealth)
        ).coerceIn(0.0, 1.0)

        val ambiguous =
            input.cameraMotionRisk >= config.ambiguityCameraMotionRisk ||
                input.synchronizationConfidence <
                    config.ambiguitySyncConfidence ||
                input.captureTimingConfidence <
                    config.ambiguityTimingConfidence

        val stateHistory = history
        if (stateHistory == null) {
            history = History(
                timestampNanos = input.timestampNanos,
                residualEmaPx = median,
                p95EmaPx = p95,
                inlierEma = inlier,
                scoreEma = score,
                lastState = State.UNKNOWN,
                watchFrames = 0,
                revalidateFrames = 0,
                blockFrames = 0,
                recoveryFrames = 0,
                clearAmbiguityFrames =
                    if (ambiguous) 0 else 1,
            )
        } else {
            updateHistory(
                stateHistory,
                input,
                median,
                p95,
                inlier,
                score,
                ambiguous,
            )
        }

        val current = history!!
        val baseline = calibration.coerceAtLeast(0.75)
        val growthRatio = median / baseline
        val state = resolveState(
            history = current,
            input = input,
            median = median,
            p95 = p95,
            maximum = maximum,
            inlier = inlier,
            score = score,
            growthRatio = growthRatio,
            ambiguous = ambiguous,
        )
        current.lastState = state

        val residualHealth = (
            0.54 * exp(-median / config.watchResidualPx) +
                0.28 * exp(-p95 / config.watchP95Px) +
                0.18 * exp(-maximum / config.blockResidualPx)
        ).coerceIn(0.0, 1.0)

        val growthPenalty =
            ((growthRatio - 1.0) /
                (config.growthRevalidateRatio - 1.0))
                .coerceIn(0.0, 1.0)

        val driftEvidence = (
            (1.0 - residualHealth) * 0.58 +
                growthPenalty * 0.16 +
                (1.0 - inlier) * 0.12 +
                (1.0 - score) * 0.08 +
                ambiguityScore * 0.06
        ).coerceIn(0.0, 1.0)

        val driftScore = (
            1.0 -
                driftEvidence
        ).coerceIn(0.0, 1.0)

        val warnings = warningsFor(
            state,
            median,
            p95,
            maximum,
            growthRatio,
            inlier,
            score,
            ambiguityScore,
            current,
        )

        val recommendations = recommendationsFor(
            state,
            median,
            p95,
            inlier,
            score,
            ambiguityScore,
            input.clockSkewMillis,
            calibration,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            state = state,
            accepted =
                state != State.BLOCKED &&
                    input.cameraCount >= config.minimumCameraCount &&
                    inlier >= config.minimumInlierRatio &&
                    score >= config.minimumMultiViewScore,
            driftScore = driftScore,
            baselineResidualPx = calibration,
            smoothedMedianResidualPx = current.residualEmaPx,
            smoothedP95ResidualPx = current.p95EmaPx,
            residualGrowthRatio = growthRatio.coerceAtLeast(0.0),
            inlierRatio = inlier,
            multiViewScore = score,
            ambiguityScore = ambiguityScore,
            persistentWatchFrames = current.watchFrames,
            persistentRevalidateFrames = current.revalidateFrames,
            persistentBlockFrames = current.blockFrames,
            recoveryFrames = current.recoveryFrames,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun state(): State = history?.lastState ?: State.UNKNOWN

    @Synchronized
    fun reset() {
        history = null
    }

    private fun resolveState(
        history: History,
        input: Input,
        median: Double,
        p95: Double,
        maximum: Double,
        inlier: Double,
        score: Double,
        growthRatio: Double,
        ambiguous: Boolean,
    ): State {
        if (input.cameraCount < config.minimumCameraCount) {
            return State.UNKNOWN
        }

        if (ambiguous) {
            return if (history.lastState == State.BLOCKED) {
                State.AMBIGUOUS
            } else {
                State.AMBIGUOUS
            }
        }

        if (
            history.lastState == State.BLOCKED &&
            history.recoveryFrames < config.recoveryFrames
        ) {
            return State.BLOCKED
        }

        if (maximum >= config.blockResidualPx ||
            p95 >= config.blockP95Px
        ) {
            return if (
                history.blockFrames >= config.persistentBlockFrames
            ) {
                State.BLOCKED
            } else {
                State.REVALIDATE
            }
        }

        if (
            median >= config.revalidateResidualPx ||
                p95 >= config.revalidateP95Px ||
                growthRatio >= config.growthRevalidateRatio
        ) {
            return if (
                history.revalidateFrames >=
                    config.persistentRevalidateFrames
            ) {
                State.REVALIDATE
            } else {
                State.WATCH
            }
        }

        if (
            median >= config.watchResidualPx ||
                p95 >= config.watchP95Px ||
                inlier < config.minimumInlierRatio ||
                score < config.minimumMultiViewScore ||
                growthRatio >= config.growthWatchRatio
        ) {
            return if (
                history.watchFrames >=
                    config.persistentWatchFrames
            ) {
                State.WATCH
            } else {
                State.STABLE
            }
        }

        if (
            history.lastState == State.AMBIGUOUS &&
                history.clearAmbiguityFrames <
                    config.ambiguityReleaseFrames
        ) {
            return State.AMBIGUOUS
        }

        return State.STABLE
    }

    private fun updateHistory(
        history: History,
        input: Input,
        median: Double,
        p95: Double,
        inlier: Double,
        score: Double,
        ambiguous: Boolean,
    ) {
        val alpha = config.historyAlpha
        history.residualEmaPx =
            history.residualEmaPx * (1.0 - alpha) +
                median * alpha
        history.p95EmaPx =
            history.p95EmaPx * (1.0 - alpha) +
                p95 * alpha
        history.inlierEma =
            history.inlierEma * (1.0 - alpha) +
                inlier * alpha
        history.scoreEma =
            history.scoreEma * (1.0 - alpha) +
                score * alpha
        history.timestampNanos = input.timestampNanos

        if (ambiguous) {
            history.clearAmbiguityFrames = 0
        } else {
            history.clearAmbiguityFrames += 1
        }

        if (
            median >= config.watchResidualPx ||
                p95 >= config.watchP95Px ||
                inlier < config.minimumInlierRatio ||
                score < config.minimumMultiViewScore
        ) {
            history.watchFrames += 1
        } else {
            history.watchFrames = max(0, history.watchFrames - 1)
        }

        if (
            median >= config.revalidateResidualPx ||
                p95 >= config.revalidateP95Px
        ) {
            history.revalidateFrames += 1
        } else {
            history.revalidateFrames =
                max(0, history.revalidateFrames - 1)
        }

        if (
            median >= config.blockResidualPx ||
                p95 >= config.blockP95Px
        ) {
            history.blockFrames += 1
        } else {
            history.blockFrames = max(0, history.blockFrames - 1)
        }

        if (
            !ambiguous &&
                median < config.watchResidualPx &&
                p95 < config.watchP95Px &&
                inlier >= config.watchInlierRatio &&
                score >= config.watchMultiViewScore
        ) {
            history.recoveryFrames += 1
        } else {
            history.recoveryFrames = 0
        }
    }

    private fun warningsFor(
        state: State,
        median: Double,
        p95: Double,
        maximum: Double,
        growthRatio: Double,
        inlier: Double,
        score: Double,
        ambiguity: Double,
        history: History,
    ): List<String> = buildList {
        when (state) {
            State.UNKNOWN ->
                add("geometry-monitor-awaiting-multi-view-evidence")
            State.STABLE -> Unit
            State.WATCH -> {
                add("live-rig-geometry-residual-elevated")
                if (growthRatio >= config.growthWatchRatio) {
                    add("runtime-residual-growth-exceeds-calibration-envelope")
                }
                if (inlier < config.watchInlierRatio) {
                    add("multi-view-inlier-coverage-degraded")
                }
            }
            State.AMBIGUOUS -> {
                add("rig-geometry-drift-ambiguous-during-capture-instability")
                if (ambiguity >= 0.65) {
                    add("motion-or-synchronization-quality-limits-drift-attribution")
                }
            }
            State.REVALIDATE -> {
                add("rig-calibration-revalidation-required")
                add("persistent-runtime-geometry-residual-exceeds-calibration-envelope")
                if (history.revalidateFrames >= config.persistentRevalidateFrames) {
                    add("persistent-reprojection-degradation-confirmed")
                }
            }
            State.BLOCKED -> {
                add("fatal:rig-calibration-no-longer-trusted")
                add("fatal:persistent-runtime-geometry-residual-exceeded-hard-limit")
            }
        }

        if (median >= config.watchResidualPx) {
            add("median-runtime-reprojection-error-elevated")
        }
        if (p95 >= config.watchP95Px) {
            add("upper-tail-runtime-reprojection-error-elevated")
        }
        if (maximum >= config.blockResidualPx) {
            add("runtime-maximum-residual-outside-production-envelope")
        }
        if (score < config.minimumMultiViewScore) {
            add("multi-view-score-below-production-threshold")
        }
    }.distinct()

    private fun recommendationsFor(
        state: State,
        median: Double,
        p95: Double,
        inlier: Double,
        score: Double,
        ambiguity: Double,
        clockSkewMillis: Double,
        calibrationResidual: Double,
    ): List<String> = buildList {
        if (state == State.AMBIGUOUS) {
            add("stabilize camera motion and peer timing before attributing residuals to rig geometry")
        }

        if (clockSkewMillis > config.maximumClockSkewMillis) {
            add("tighten inter-camera clock alignment before geometry revalidation")
        }

        if (inlier < config.minimumInlierRatio) {
            add("inspect low-confidence views for blur, occlusion, focus loss or exposure instability")
        }

        if (score < config.minimumMultiViewScore) {
            add("hold metric output until multi-view confidence recovers")
        }

        if (
            median >= config.watchResidualPx ||
                p95 >= config.watchP95Px
        ) {
            add("verify that no phone, tripod or mount moved after calibration")
        }

        if (state == State.REVALIDATE || state == State.BLOCKED) {
            add("rerun multi-camera calibration before accepting another metric capture segment")
        }

        if (calibrationResidual > config.watchResidualPx) {
            add("replace the calibration snapshot with a lower-residual reference take")
        }

        if (ambiguity > 0.50) {
            add("use a stable capture window to separate transient capture faults from geometry drift")
        }
    }.distinct()

    private fun sanitize(value: Double): Double =
        if (value.isFinite()) {
            min(max(value, 0.0), 100.0)
        } else {
            100.0
        }
}
