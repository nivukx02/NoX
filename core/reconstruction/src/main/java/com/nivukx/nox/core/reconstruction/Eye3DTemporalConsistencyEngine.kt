package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.EyeSide
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max

/**
 * Bilateral temporal gate for 3D eye reconstruction.
 *
 * It validates measured multi-view eye geometry without suppressing genuine saccades. A rapid gaze
 * transition is only downgraded when bilateral agreement, triangulation confidence, uncertainty,
 * or temporal evidence cannot support the transition.
 */
class Eye3DTemporalConsistencyEngine(
    config: Config = Config(),
) {
    @Volatile
    private var config: Config = config


    enum class State {
        UNKNOWN,
        STABLE,
        RAPID_GAZE,
        BILATERAL_MISMATCH,
        DEGRADED,
    }

    data class Config(
        val maximumGapSeconds: Double = 0.20,
        val minimumConfidence: Double = 0.28,
        val warningConfidence: Double = 0.68,
        val rejectConfidence: Double = 0.42,
        val baselineWarningRatio: Double = 0.055,
        val baselineRejectRatio: Double = 0.13,
        val gazeWarningDegrees: Double = 18.0,
        val gazeRejectDegrees: Double = 38.0,
        val rapidGazeVelocityDegreesPerSecond: Double = 720.0,
        val extremeGazeVelocityDegreesPerSecond: Double = 1440.0,
        val gazeAccelerationWarningDegreesPerSecond2: Double = 6000.0,
        val historyAlpha: Double = 0.18,
        val persistentWarningFrames: Int = 3,
        val persistentRejectFrames: Int = 6,
        val recoveryFrames: Int = 4,
    ) {
        init {
            require(maximumGapSeconds > 0.0)
            require(minimumConfidence in 0.0..1.0)
            require(rejectConfidence in 0.0..warningConfidence)
            require(baselineWarningRatio > 0.0)
            require(baselineRejectRatio > baselineWarningRatio)
            require(gazeWarningDegrees > 0.0)
            require(gazeRejectDegrees > gazeWarningDegrees)
            require(rapidGazeVelocityDegreesPerSecond > 0.0)
            require(
                extremeGazeVelocityDegreesPerSecond >
                    rapidGazeVelocityDegreesPerSecond,
            )
            require(gazeAccelerationWarningDegreesPerSecond2 > 0.0)
            require(historyAlpha in 0.0..1.0)
            require(persistentWarningFrames >= 1)
            require(persistentRejectFrames >= persistentWarningFrames)
            require(recoveryFrames >= 1)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val left: Eye3DReconstructionEngine.Result?,
        val right: Eye3DReconstructionEngine.Result?,
    ) {
        init {
            require(timestampNanos > 0L)
            require(left == null || left.eye == EyeSide.LEFT)
            require(right == null || right.eye == EyeSide.RIGHT)
        }
    }

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val confidence: Double,
        val bilateralConfidence: Double,
        val temporalConfidence: Double,
        val leftConfidence: Double,
        val rightConfidence: Double,
        val interEyeDistanceMeters: Double?,
        val interEyeDistanceChangeRatio: Double?,
        val gazeAngleDifferenceDegrees: Double?,
        val meanGazeVelocityDegreesPerSecond: Double,
        val gazeAccelerationDegreesPerSecond2: Double,
        val baselineScore: Double,
        val gazeAgreementScore: Double,
        val triangulationScore: Double,
        val persistentWarningFrames: Int,
        val persistentRejectFrames: Int,
        val recoveryFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val accepted: Boolean
            get() =
                state != State.DEGRADED

        val degraded: Boolean
            get() = state == State.DEGRADED
    }

    private data class History(
        var timestampNanos: Long,
        var baselineMeters: Double?,
        var previousMeanVelocity: Double,
        var confidenceEma: Double,
        var temporalEma: Double,
        var warningFrames: Int,
        var rejectFrames: Int,
        var recoveryFrames: Int,
        var lastState: State,
    )

    private var history: History? = null
    private var lastResult: Result? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        lastResult?.let { previous ->
            if (input.timestampNanos <= previous.timestampNanos) {
                return previous.copy(
                    warnings =
                        (
                            previous.warnings +
                                "stale-eye-frame-rejected"
                            ).distinct(),
                )
            }
        }

        val left = input.left
        val right = input.right

        if (left == null && right == null) {
            reset()
            return emptyResult(input.timestampNanos)
        }

        val leftConfidence =
            left?.confidence?.coerceIn(0.0, 1.0) ?: 0.0
        val rightConfidence =
            right?.confidence?.coerceIn(0.0, 1.0) ?: 0.0

        val leftWeight =
            if (right != null) {
                0.50
            } else {
                0.76
            }
        val rightWeight =
            if (left != null) {
                0.50
            } else {
                0.0
            }

        val triangulationScore = (
            eyeScore(left) * leftWeight +
                eyeScore(right) * rightWeight
        ).coerceIn(0.0, 1.0)

        val baselineMeters =
            if (left != null && right != null) {
                left.centerWorld.distanceTo(right.centerWorld)
            } else {
                null
            }

        val previous = history
        val dtSeconds =
            if (previous == null) {
                0.0
            } else {
                (
                    input.timestampNanos -
                        previous.timestampNanos
                ).coerceAtLeast(1L) / 1e9
            }

        if (previous != null && dtSeconds > config.maximumGapSeconds) {
            resetHistory(input.timestampNanos, baselineMeters)
        }

        val activeHistory = history
        val baselineChangeRatio =
            if (
                baselineMeters == null ||
                    activeHistory?.baselineMeters == null ||
                    activeHistory.baselineMeters!! <= 1e-8
            ) {
                null
            } else {
                abs(
                    baselineMeters -
                        activeHistory.baselineMeters!!,
                ) / activeHistory.baselineMeters!!
            }

        val gazeAngleDifference =
            if (left != null && right != null) {
                Math.toDegrees(
                    acos(
                        left.gazeDirectionWorld
                            .normalized()
                            .dot(
                                right.gazeDirectionWorld
                                    .normalized(),
                            )
                            .coerceIn(-1.0, 1.0),
                    ),
                )
            } else {
                null
            }

        val meanVelocity = listOfNotNull(
            left?.angularVelocityDegreesPerSecond,
            right?.angularVelocityDegreesPerSecond,
        ).averageOrZero()

        val acceleration =
            if (dtSeconds <= 1e-4) {
                0.0
            } else {
                abs(
                    meanVelocity -
                        (activeHistory?.previousMeanVelocity ?: 0.0),
                ) / dtSeconds
            }

        val baselineScore = baselineScore(baselineChangeRatio)
        val agreementScore = gazeAgreementScore(gazeAngleDifference)
        val velocityScore = velocityScore(meanVelocity)
        val accelerationScore = accelerationScore(acceleration)

        val eyeConfidenceAverage =
            if (left != null && right != null) {
                (leftConfidence + rightConfidence) * 0.5
            } else {
                max(leftConfidence, rightConfidence) * 0.76
            }

        val bilateralConfidence = (
            eyeConfidenceAverage * 0.40 +
                agreementScore * 0.22 +
                baselineScore * 0.18 +
                triangulationScore * 0.20
        ).coerceIn(0.0, 1.0)

        val temporalConfidenceRaw = (
            velocityScore * 0.34 +
                accelerationScore * 0.18 +
                eyeConfidenceAverage * 0.20 +
                triangulationScore * 0.28
        ).coerceIn(0.0, 1.0)

        val rawConfidence = (
            eyeConfidenceAverage * 0.34 +
                bilateralConfidence * 0.38 +
                temporalConfidenceRaw * 0.28
        ).coerceIn(0.0, 1.0)

        val strictnessPressure =
            (
                (26.0 - config.gazeWarningDegrees) / 8.0
            ).coerceIn(0.0, 1.0)

        val confidence =
            (
                rawConfidence *
                    (1.0 - strictnessPressure * 0.16)
            ).coerceIn(0.0, 1.0)

        val warningNow =
            confidence < config.warningConfidence ||
                baselineChangeRatio != null &&
                baselineChangeRatio >= config.baselineWarningRatio ||
                gazeAngleDifference != null &&
                gazeAngleDifference >= config.gazeWarningDegrees ||
                meanVelocity >= config.rapidGazeVelocityDegreesPerSecond ||
                acceleration >= config.gazeAccelerationWarningDegreesPerSecond2

        val rejectNow =
            confidence < config.rejectConfidence ||
                baselineChangeRatio != null &&
                baselineChangeRatio >= config.baselineRejectRatio ||
                gazeAngleDifference != null &&
                gazeAngleDifference >= config.gazeRejectDegrees

        updateHistory(
            timestampNanos = input.timestampNanos,
            baselineMeters = baselineMeters,
            meanVelocity = meanVelocity,
            confidence = confidence,
            temporalConfidence = temporalConfidenceRaw,
            warningNow = warningNow,
            rejectNow = rejectNow,
        )

        val current = history!!
        val rapidGaze =
            meanVelocity >= config.rapidGazeVelocityDegreesPerSecond ||
                acceleration >=
                config.gazeAccelerationWarningDegreesPerSecond2
        val bilateralMismatch =
            gazeAngleDifference != null &&
                gazeAngleDifference >= config.gazeWarningDegrees

        val state = resolveState(
            current = current,
            rapidGaze = rapidGaze,
            bilateralMismatch = bilateralMismatch,
        )
        current.lastState = state

        val result = Result(
            timestampNanos = input.timestampNanos,
            state = state,
            confidence = current.confidenceEma,
            bilateralConfidence = bilateralConfidence,
            temporalConfidence = current.temporalEma,
            leftConfidence = leftConfidence,
            rightConfidence = rightConfidence,
            interEyeDistanceMeters = baselineMeters,
            interEyeDistanceChangeRatio = baselineChangeRatio,
            gazeAngleDifferenceDegrees = gazeAngleDifference,
            meanGazeVelocityDegreesPerSecond = meanVelocity,
            gazeAccelerationDegreesPerSecond2 = acceleration,
            baselineScore = baselineScore,
            gazeAgreementScore = agreementScore,
            triangulationScore = triangulationScore,
            persistentWarningFrames = current.warningFrames,
            persistentRejectFrames = current.rejectFrames,
            recoveryFrames = current.recoveryFrames,
            warnings = warnings(
                state = state,
                resultConfidence = current.confidenceEma,
                baselineChangeRatio = baselineChangeRatio,
                gazeAngleDifference = gazeAngleDifference,
                velocity = meanVelocity,
                acceleration = acceleration,
                left = left,
                right = right,
                history = current,
            ),
            recommendations = recommendations(
                state = state,
                baselineChangeRatio = baselineChangeRatio,
                gazeAngleDifference = gazeAngleDifference,
                velocity = meanVelocity,
                acceleration = acceleration,
                triangulationScore = triangulationScore,
            ),
        )
        lastResult = result
        return result
    }

    @Synchronized
    fun configureStrictness(strictness: Double) {
        val normalized = strictness.coerceIn(0.0, 1.0)
        config = Config(
            maximumGapSeconds = 0.20,
            minimumConfidence = 0.36 - normalized * 0.10,
            warningConfidence = 0.60 + normalized * 0.12,
            rejectConfidence = 0.34 + normalized * 0.12,
            baselineWarningRatio =
                0.090 - normalized * 0.045,
            baselineRejectRatio =
                0.180 - normalized * 0.065,
            gazeWarningDegrees =
                26.0 - normalized * 8.0,
            gazeRejectDegrees =
                48.0 - normalized * 16.0,
            rapidGazeVelocityDegreesPerSecond =
                880.0 - normalized * 160.0,
            extremeGazeVelocityDegreesPerSecond =
                1600.0 - normalized * 140.0,
            gazeAccelerationWarningDegreesPerSecond2 =
                7000.0 - normalized * 1200.0,
            historyAlpha = 0.18,
            persistentWarningFrames =
                (5.0 - normalized * 2.0)
                    .toInt()
                    .coerceAtLeast(2),
            persistentRejectFrames =
                (8.0 - normalized * 4.0)
                    .toInt()
                    .coerceAtLeast(3),
            recoveryFrames =
                (5.0 - normalized * 1.0)
                    .toInt()
                    .coerceAtLeast(3),
        )
        history = null
    }

    @Synchronized
    fun reset() {
        history = null
        lastResult = null
    }

    private fun updateHistory(
        timestampNanos: Long,
        baselineMeters: Double?,
        meanVelocity: Double,
        confidence: Double,
        temporalConfidence: Double,
        warningNow: Boolean,
        rejectNow: Boolean,
    ) {
        val current = history ?: History(
            timestampNanos = timestampNanos,
            baselineMeters = baselineMeters,
            previousMeanVelocity = meanVelocity,
            confidenceEma = confidence,
            temporalEma = temporalConfidence,
            warningFrames = 0,
            rejectFrames = 0,
            recoveryFrames = 0,
            lastState = State.UNKNOWN,
        )

        val alpha = config.historyAlpha
        current.timestampNanos = timestampNanos
        current.baselineMeters =
            baselineMeters ?: current.baselineMeters
        current.previousMeanVelocity = meanVelocity
        current.confidenceEma =
            current.confidenceEma * (1.0 - alpha) +
                confidence * alpha
        current.temporalEma =
            current.temporalEma * (1.0 - alpha) +
                temporalConfidence * alpha
        current.warningFrames =
            if (warningNow) {
                current.warningFrames + 1
            } else {
                max(0, current.warningFrames - 1)
            }
        current.rejectFrames =
            if (rejectNow) {
                current.rejectFrames + 1
            } else {
                max(0, current.rejectFrames - 1)
            }
        current.recoveryFrames =
            if (!warningNow && !rejectNow) {
                current.recoveryFrames + 1
            } else {
                0
            }
        history = current
    }

    private fun resolveState(
        current: History,
        rapidGaze: Boolean,
        bilateralMismatch: Boolean,
    ): State =
        when {
            current.rejectFrames >= config.persistentRejectFrames ->
                State.DEGRADED
            bilateralMismatch -> State.BILATERAL_MISMATCH
            rapidGaze -> State.RAPID_GAZE
            current.lastState == State.DEGRADED &&
                current.recoveryFrames < config.recoveryFrames ->
                State.DEGRADED
            current.confidenceEma >= config.warningConfidence &&
                current.temporalEma >= config.warningConfidence ->
                State.STABLE
            else -> State.DEGRADED
        }

    private fun eyeScore(
        result: Eye3DReconstructionEngine.Result?,
    ): Double {
        if (result == null) return 0.0

        val uncertaintyScore = exp(
            -result.centerUncertaintyMeters / 0.015,
        ).coerceIn(0.0, 1.0)
        val angularScore = (
            1.0 - result.angularSpreadDegrees / 22.0
        ).coerceIn(0.0, 1.0)
        val parallaxScore =
            if (result.parallaxDegrees <= 0.0) {
                0.72
            } else {
                (
                    1.0 -
                        kotlin.math.exp(
                            -result.parallaxDegrees / 1.5,
                        )
                    ).coerceIn(0.0, 1.0)
            }

        return (
            result.confidence * 0.48 +
                uncertaintyScore * 0.24 +
                angularScore * 0.16 +
                parallaxScore * 0.12
        ).coerceIn(0.0, 1.0)
    }

    private fun baselineScore(changeRatio: Double?): Double =
        when {
            changeRatio == null -> 0.70
            changeRatio <= config.baselineWarningRatio -> 1.0
            changeRatio >= config.baselineRejectRatio -> 0.0
            else ->
                1.0 -
                    (
                        changeRatio -
                            config.baselineWarningRatio
                    ) /
                    (
                        config.baselineRejectRatio -
                            config.baselineWarningRatio
                    )
        }.coerceIn(0.0, 1.0)

    private fun gazeAgreementScore(angleDegrees: Double?): Double =
        when {
            angleDegrees == null -> 0.68
            angleDegrees <= config.gazeWarningDegrees -> 1.0
            angleDegrees >= config.gazeRejectDegrees -> 0.0
            else ->
                1.0 -
                    (
                        angleDegrees -
                            config.gazeWarningDegrees
                    ) /
                    (
                        config.gazeRejectDegrees -
                            config.gazeWarningDegrees
                    )
        }.coerceIn(0.0, 1.0)

    private fun velocityScore(velocity: Double): Double =
        when {
            velocity <= config.rapidGazeVelocityDegreesPerSecond ->
                1.0
            velocity >= config.extremeGazeVelocityDegreesPerSecond ->
                0.0
            else ->
                1.0 -
                    (
                        velocity -
                            config.rapidGazeVelocityDegreesPerSecond
                    ) /
                    (
                        config.extremeGazeVelocityDegreesPerSecond -
                            config.rapidGazeVelocityDegreesPerSecond
                    )
        }.coerceIn(0.0, 1.0)

    private fun accelerationScore(acceleration: Double): Double =
        (
            1.0 -
                acceleration /
                    config.gazeAccelerationWarningDegreesPerSecond2
        ).coerceIn(0.0, 1.0)

    private fun resetHistory(
        timestampNanos: Long,
        baselineMeters: Double?,
    ) {
        history = History(
            timestampNanos = timestampNanos,
            baselineMeters = baselineMeters,
            previousMeanVelocity = 0.0,
            confidenceEma = config.minimumConfidence,
            temporalEma = config.minimumConfidence,
            warningFrames = 0,
            rejectFrames = 0,
            recoveryFrames = 0,
            lastState = State.UNKNOWN,
        )
    }

    private fun warnings(
        state: State,
        resultConfidence: Double,
        baselineChangeRatio: Double?,
        gazeAngleDifference: Double?,
        velocity: Double,
        acceleration: Double,
        left: Eye3DReconstructionEngine.Result?,
        right: Eye3DReconstructionEngine.Result?,
        history: History,
    ): List<String> = buildList {
        when (state) {
            State.UNKNOWN -> add("eye-temporal-history-initializing")
            State.STABLE -> Unit
            State.RAPID_GAZE -> add("rapid-gaze-transition-detected")
            State.BILATERAL_MISMATCH -> {
                add("left-right-gaze-geometry-disagrees")
                add("eye-camera-consensus-degraded")
            }
            State.DEGRADED -> {
                add("3d-eye-temporal-consistency-degraded")
                add("eye-geometry-output-should-be-held")
            }
        }

        if (baselineChangeRatio != null &&
            baselineChangeRatio >= config.baselineWarningRatio
        ) {
            add("inter-eye-baseline-changed-abruptly")
        }

        if (gazeAngleDifference != null &&
            gazeAngleDifference >= config.gazeWarningDegrees
        ) {
            add("left-right-gaze-geometry-disagrees")
            add("cross-eye-gaze-angle-disagreement-elevated")
        }

        if (velocity >= config.rapidGazeVelocityDegreesPerSecond) {
            add("gaze-angular-velocity-is-high")
        }

        if (acceleration >= config.gazeAccelerationWarningDegreesPerSecond2) {
            add("gaze-angular-acceleration-is-high")
        }

        if (left?.centerUncertaintyMeters ?: 0.0 > 0.03) {
            add("left-eye-center-uncertainty-elevated")
        }

        if (right?.centerUncertaintyMeters ?: 0.0 > 0.03) {
            add("right-eye-center-uncertainty-elevated")
        }

        if (history.warningFrames >= config.persistentWarningFrames) {
            add("persistent-eye-temporal-instability")
        }

        if (history.rejectFrames >= config.persistentRejectFrames) {
            add("persistent-eye-geometry-rejection")
        }

        if (resultConfidence < config.minimumConfidence) {
            add("eye-temporal-confidence-below-production-floor")
        }
    }.distinct()

    private fun recommendations(
        state: State,
        baselineChangeRatio: Double?,
        gazeAngleDifference: Double?,
        velocity: Double,
        acceleration: Double,
        triangulationScore: Double,
    ): List<String> = buildList {
        if (state == State.BILATERAL_MISMATCH) {
            add("verify face-camera calibration and the active face track")
        }
        if (baselineChangeRatio != null &&
            baselineChangeRatio >= config.baselineWarningRatio
        ) {
            add("hold facial scale before accepting a rapid vergence change")
        }
        if (gazeAngleDifference != null &&
            gazeAngleDifference >= config.gazeWarningDegrees
        ) {
            add("verify both eye observations belong to the same face")
        }
        if (velocity >= config.rapidGazeVelocityDegreesPerSecond) {
            add("retain rapid gaze transitions only while multi-view confidence remains high")
        }
        if (acceleration >= config.gazeAccelerationWarningDegreesPerSecond2) {
            add("increase eye-camera temporal stability before trusting gaze acceleration")
        }
        if (triangulationScore < 0.55) {
            add("increase face-camera baseline or remove low-confidence eye views")
        }
    }.distinct()

    private fun emptyResult(timestampNanos: Long): Result =
        Result(
            timestampNanos = timestampNanos,
            state = State.DEGRADED,
            confidence = 0.0,
            bilateralConfidence = 0.0,
            temporalConfidence = 0.0,
            leftConfidence = 0.0,
            rightConfidence = 0.0,
            interEyeDistanceMeters = null,
            interEyeDistanceChangeRatio = null,
            gazeAngleDifferenceDegrees = null,
            meanGazeVelocityDegreesPerSecond = 0.0,
            gazeAccelerationDegreesPerSecond2 = 0.0,
            baselineScore = 0.0,
            gazeAgreementScore = 0.0,
            triangulationScore = 0.0,
            persistentWarningFrames = 0,
            persistentRejectFrames = 0,
            recoveryFrames = 0,
            warnings = listOf("no-3d-eye-observations-available"),
            recommendations = listOf(
                "Hold 3D eye output until stable multi-view eye observations are available.",
            ),
        )
}

private fun List<Double>.averageOrZero(): Double =
    if (isEmpty()) 0.0 else average()
