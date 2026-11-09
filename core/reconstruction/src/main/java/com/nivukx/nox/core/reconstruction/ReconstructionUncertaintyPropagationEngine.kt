package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import kotlin.math.exp
import kotlin.math.max

/**
 * Conservative metric uncertainty propagation for reconstructed 3D skeletons.
 *
 * This class does not synthesize a covariance matrix from missing measurements. It starts with
 * the joint-level metric uncertainty produced by calibrated reconstruction and inflates it using
 * observable downstream failure modes: reprojection quality, temporal stability, geometric
 * observability, rig drift, synchronization, timing, camera motion, and occlusion.
 */
class ReconstructionUncertaintyPropagationEngine(
    private val config: Config = Config(),
) {
    enum class State {
        UNKNOWN,
        STABLE,
        WATCH,
        DEGRADED,
        BLOCKED,
    }

    data class Config(
        val unknownJointUncertaintyMeters: Double = 0.060,
        val warningJointUncertaintyMeters: Double = 0.085,
        val rejectJointUncertaintyMeters: Double = 0.180,
        val warningMeanUncertaintyMeters: Double = 0.050,
        val rejectMeanUncertaintyMeters: Double = 0.120,
        val minimumTrustedJointRatio: Double = 0.62,
        val targetTrustedJointRatio: Double = 0.86,
        val minimumConfidence: Double = 0.32,
        val warningScore: Double = 0.70,
        val rejectScore: Double = 0.46,
        val historyAlpha: Double = 0.18,
        val persistentWarningFrames: Int = 4,
        val persistentBlockedFrames: Int = 8,
        val recoveryFrames: Int = 5,
        val maximumInflation: Double = 8.0,
        val reprojectionWeight: Double = 0.72,
        val temporalWeight: Double = 0.44,
        val observabilityWeight: Double = 0.68,
        val driftWeight: Double = 0.64,
        val timingWeight: Double = 0.42,
        val syncWeight: Double = 0.46,
        val cameraMotionWeight: Double = 0.26,
        val occlusionWeight: Double = 0.34,
    ) {
        init {
            require(unknownJointUncertaintyMeters > 0.0)
            require(warningJointUncertaintyMeters > 0.0)
            require(rejectJointUncertaintyMeters > warningJointUncertaintyMeters)
            require(warningMeanUncertaintyMeters > 0.0)
            require(rejectMeanUncertaintyMeters > warningMeanUncertaintyMeters)
            require(minimumTrustedJointRatio in 0.0..1.0)
            require(targetTrustedJointRatio >= minimumTrustedJointRatio)
            require(targetTrustedJointRatio <= 1.0)
            require(minimumConfidence in 0.0..1.0)
            require(rejectScore in 0.0..warningScore)
            require(historyAlpha in 0.0..1.0)
            require(persistentWarningFrames >= 1)
            require(persistentBlockedFrames >= persistentWarningFrames)
            require(recoveryFrames >= 1)
            require(maximumInflation >= 1.0)
            require(reprojectionWeight >= 0.0)
            require(temporalWeight >= 0.0)
            require(observabilityWeight >= 0.0)
            require(driftWeight >= 0.0)
            require(timingWeight >= 0.0)
            require(syncWeight >= 0.0)
            require(cameraMotionWeight >= 0.0)
            require(occlusionWeight >= 0.0)
        }
    }

    data class Input(
        val timestampNanos: Long,
        val skeleton: SkeletonFrame,
        val reprojectionQuality: Double,
        val temporalQuality: Double,
        val geometricObservability: Double,
        val rigGeometryDriftScore: Double,
        val synchronizationConfidence: Double,
        val captureTimingConfidence: Double,
        val cameraMotionRisk: Double,
        val occlusionRisk: Double,
    ) {
        init {
            require(timestampNanos > 0L)
            require(reprojectionQuality in 0.0..1.0)
            require(temporalQuality in 0.0..1.0)
            require(geometricObservability in 0.0..1.0)
            require(rigGeometryDriftScore in 0.0..1.0)
            require(synchronizationConfidence in 0.0..1.0)
            require(captureTimingConfidence in 0.0..1.0)
            require(cameraMotionRisk in 0.0..1.0)
            require(occlusionRisk in 0.0..1.0)
        }
    }

    data class JointUncertainty(
        val landmarkId: Int,
        val baseUncertaintyMeters: Double,
        val effectiveUncertaintyMeters: Double,
        val confidence: Double,
        val observationCount: Int,
        val reprojectionErrorPx: Double,
        val trusted: Boolean,
        val reasons: List<String>,
    )

    data class Result(
        val timestampNanos: Long,
        val state: State,
        val accepted: Boolean,
        val score: Double,
        val confidence: Double,
        val meanUncertaintyMeters: Double,
        val p95UncertaintyMeters: Double,
        val worstUncertaintyMeters: Double,
        val trustedJointRatio: Double,
        val trustedJointCount: Int,
        val jointCount: Int,
        val inflationFactor: Double,
        val dominantBottleneck: String,
        val jointUncertainty: List<JointUncertainty>,
        val persistentWarningFrames: Int,
        val persistentBlockedFrames: Int,
        val recoveryFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    ) {
        val degraded: Boolean
            get() = state == State.DEGRADED || state == State.BLOCKED
    }

    private data class History(
        var timestampNanos: Long,
        var scoreEma: Double,
        var confidenceEma: Double,
        var meanUncertaintyEma: Double,
        var trustedRatioEma: Double,
        var warningFrames: Int,
        var blockedFrames: Int,
        var recoveryFrames: Int,
        var lastState: State,
    )

    private var history: History? = null

    @Synchronized
    fun evaluate(input: Input): Result {
        val joints = input.skeleton.joints
        if (joints.isEmpty()) {
            return blockedResult(
                input.timestampNanos,
                "no-reconstruction-joints-available",
            )
        }

        val inflation = computeInflationFactor(input)
        val perJoint = joints.map { joint ->
            evaluateJoint(joint, input, inflation)
        }
        val uncertainties = perJoint
            .map { it.effectiveUncertaintyMeters }
            .filter { it.isFinite() }
            .sorted()

        if (uncertainties.isEmpty()) {
            return blockedResult(
                input.timestampNanos,
                "joint-uncertainty-values-are-non-finite",
            )
        }

        val mean = uncertainties.average()
        val p95 = quantile(uncertainties, 0.95)
        val worst = uncertainties.last()

        val trustedCount = perJoint.count { it.trusted }
        val trustedRatio = trustedCount.toDouble() /
            perJoint.size.toDouble()

        val uncertaintyScore = (
            scoreFromUncertainty(mean) * 0.32 +
                scoreFromUncertainty(p95) * 0.32 +
                scoreFromUncertainty(worst) * 0.14 +
                trustedRatio * 0.22
        ).coerceIn(0.0, 1.0)

        val upstreamScore = (
            input.reprojectionQuality * 0.22 +
                input.temporalQuality * 0.18 +
                input.geometricObservability * 0.18 +
                input.rigGeometryDriftScore * 0.16 +
                input.synchronizationConfidence * 0.10 +
                input.captureTimingConfidence * 0.08 +
                (1.0 - input.cameraMotionRisk) * 0.04 +
                (1.0 - input.occlusionRisk) * 0.04
        ).coerceIn(0.0, 1.0)

        val score = (
            uncertaintyScore * 0.70 +
                upstreamScore * 0.30
        ).coerceIn(0.0, 1.0)

        val confidence = (
            input.skeleton.globalConfidence.coerceIn(0.0, 1.0) * 0.34 +
                score * 0.40 +
                trustedRatio * 0.18 +
                input.reprojectionQuality * 0.08
        ).coerceIn(0.0, 1.0)

        val blockedNow =
            mean >= config.rejectMeanUncertaintyMeters ||
                p95 >= config.rejectJointUncertaintyMeters ||
                trustedRatio < config.minimumTrustedJointRatio ||
                score < config.rejectScore

        val warningNow =
            blockedNow ||
                mean >= config.warningMeanUncertaintyMeters ||
                p95 >= config.warningJointUncertaintyMeters ||
                trustedRatio < config.targetTrustedJointRatio ||
                score < config.warningScore

        updateHistory(
            timestampNanos = input.timestampNanos,
            meanUncertainty = mean,
            trustedRatio = trustedRatio,
            score = score,
            confidence = confidence,
            warningNow = warningNow,
            blockedNow = blockedNow,
        )

        val current = history!!
        val state = resolveState(current, warningNow, blockedNow)
        current.lastState = state

        val bottleneck = bottleneck(
            input = input,
            meanUncertainty = mean,
            p95Uncertainty = p95,
            trustedRatio = trustedRatio,
        )

        return Result(
            timestampNanos = input.timestampNanos,
            state = state,
            accepted =
                state != State.BLOCKED &&
                    trustedRatio >= config.minimumTrustedJointRatio &&
                    score >= config.rejectScore,
            score = current.scoreEma,
            confidence = current.confidenceEma,
            meanUncertaintyMeters = mean,
            p95UncertaintyMeters = p95,
            worstUncertaintyMeters = worst,
            trustedJointRatio = trustedRatio,
            trustedJointCount = trustedCount,
            jointCount = perJoint.size,
            inflationFactor = inflation,
            dominantBottleneck = bottleneck,
            jointUncertainty = perJoint,
            persistentWarningFrames = current.warningFrames,
            persistentBlockedFrames = current.blockedFrames,
            recoveryFrames = current.recoveryFrames,
            warnings = buildWarnings(
                state = state,
                input = input,
                meanUncertainty = mean,
                p95Uncertainty = p95,
                trustedRatio = trustedRatio,
                inflation = inflation,
                history = current,
            ),
            recommendations = buildRecommendations(
                state = state,
                input = input,
                meanUncertainty = mean,
                p95Uncertainty = p95,
                trustedRatio = trustedRatio,
                bottleneck = bottleneck,
            ),
        )
    }

    @Synchronized
    fun reset() {
        history = null
    }

    private fun evaluateJoint(
        joint: ReconstructionJoint,
        input: Input,
        inflation: Double,
    ): JointUncertainty {
        val explicit = joint.positionUncertaintyMeters.isFinite() &&
            joint.positionUncertaintyMeters > 1e-6
        val base = if (explicit) {
            joint.positionUncertaintyMeters
        } else {
            config.unknownJointUncertaintyMeters
        }

        val confidencePenalty =
            (1.0 - joint.confidence.coerceIn(0.0, 1.0)) * 0.85

        val reprojectionPenalty =
            if (joint.reprojectionErrorPx.isFinite()) {
                (joint.reprojectionErrorPx / 8.0)
                    .coerceIn(0.0, 1.5) * 0.55
            } else {
                0.85
            }

        val observationPenalty = (
            1.0 -
                (joint.observationCount.toDouble() / 4.0)
                    .coerceIn(0.0, 1.0)
        ) * 0.40

        val localInflation = (
            1.0 +
                confidencePenalty +
                reprojectionPenalty +
                observationPenalty
        ).coerceIn(1.0, config.maximumInflation)

        val effective = (
            base *
                inflation *
                localInflation
        ).coerceAtMost(
            config.rejectJointUncertaintyMeters *
                config.maximumInflation,
        )

        val reasons = buildList {
            if (!explicit) add("metric-uncertainty-not-explicit")
            if (joint.confidence < 0.55) {
                add("joint-confidence-limited")
            }
            if (
                joint.reprojectionErrorPx.isFinite() &&
                joint.reprojectionErrorPx > 4.0
            ) {
                add("joint-reprojection-residual-elevated")
            }
            if (joint.observationCount < 3) {
                add("joint-supported-by-few-views")
            }
            if (input.geometricObservability < 0.62) {
                add("global-geometric-observability-limited")
            }
            if (input.synchronizationConfidence < 0.65) {
                add("multi-camera-synchronization-limited")
            }
        }.distinct()

        return JointUncertainty(
            landmarkId = joint.landmarkId,
            baseUncertaintyMeters = base,
            effectiveUncertaintyMeters = effective,
            confidence = joint.confidence,
            observationCount = joint.observationCount,
            reprojectionErrorPx = joint.reprojectionErrorPx,
            trusted =
                effective <= config.warningJointUncertaintyMeters &&
                    joint.confidence >= config.minimumConfidence,
            reasons = reasons,
        )
    }

    private fun computeInflationFactor(input: Input): Double {
        val penalty =
            (1.0 - input.reprojectionQuality) *
                config.reprojectionWeight +
                (1.0 - input.temporalQuality) *
                config.temporalWeight +
                (1.0 - input.geometricObservability) *
                config.observabilityWeight +
                (1.0 - input.rigGeometryDriftScore) *
                config.driftWeight +
                (1.0 - input.captureTimingConfidence) *
                config.timingWeight +
                (1.0 - input.synchronizationConfidence) *
                config.syncWeight +
                input.cameraMotionRisk *
                config.cameraMotionWeight +
                input.occlusionRisk *
                config.occlusionWeight

        return (1.0 + penalty).coerceIn(
            1.0,
            config.maximumInflation,
        )
    }

    private fun scoreFromUncertainty(
        uncertaintyMeters: Double,
    ): Double =
        exp(
            -uncertaintyMeters /
                config.warningJointUncertaintyMeters,
        ).coerceIn(0.0, 1.0)

    private fun updateHistory(
        timestampNanos: Long,
        meanUncertainty: Double,
        trustedRatio: Double,
        score: Double,
        confidence: Double,
        warningNow: Boolean,
        blockedNow: Boolean,
    ) {
        val current = history ?: History(
            timestampNanos = timestampNanos,
            scoreEma = score,
            confidenceEma = confidence,
            meanUncertaintyEma = meanUncertainty,
            trustedRatioEma = trustedRatio,
            warningFrames = 0,
            blockedFrames = 0,
            recoveryFrames = 0,
            lastState = State.UNKNOWN,
        )

        val alpha = config.historyAlpha
        current.timestampNanos = timestampNanos
        current.scoreEma = current.scoreEma * (1.0 - alpha) + score * alpha
        current.confidenceEma =
            current.confidenceEma * (1.0 - alpha) +
                confidence * alpha
        current.meanUncertaintyEma =
            current.meanUncertaintyEma * (1.0 - alpha) +
                meanUncertainty * alpha
        current.trustedRatioEma =
            current.trustedRatioEma * (1.0 - alpha) +
                trustedRatio * alpha

        current.warningFrames =
            if (warningNow) current.warningFrames + 1
            else max(0, current.warningFrames - 1)
        current.blockedFrames =
            if (blockedNow) current.blockedFrames + 1
            else max(0, current.blockedFrames - 1)
        current.recoveryFrames =
            if (!warningNow && !blockedNow) current.recoveryFrames + 1
            else 0

        history = current
    }

    private fun resolveState(
        current: History,
        warningNow: Boolean,
        blockedNow: Boolean,
    ): State =
        when {
            current.blockedFrames >= config.persistentBlockedFrames ->
                State.BLOCKED
            blockedNow ->
                State.DEGRADED
            warningNow ->
                State.WATCH
            current.lastState == State.BLOCKED &&
                current.recoveryFrames < config.recoveryFrames ->
                State.DEGRADED
            current.scoreEma >= config.warningScore ->
                State.STABLE
            else ->
                State.WATCH
        }

    private fun bottleneck(
        input: Input,
        meanUncertainty: Double,
        p95Uncertainty: Double,
        trustedRatio: Double,
    ): String {
        val candidates = listOf(
            "JOINT_UNCERTAINTY" to
                meanUncertainty / config.rejectMeanUncertaintyMeters,
            "P95_UNCERTAINTY" to
                p95Uncertainty / config.rejectJointUncertaintyMeters,
            "REPROJECTION" to (1.0 - input.reprojectionQuality),
            "OBSERVABILITY" to (1.0 - input.geometricObservability),
            "RIG_DRIFT" to (1.0 - input.rigGeometryDriftScore),
            "SYNC" to (1.0 - input.synchronizationConfidence),
            "TIMING" to (1.0 - input.captureTimingConfidence),
            "CAMERA_MOTION" to input.cameraMotionRisk,
            "OCCLUSION" to input.occlusionRisk,
            "TRUSTED_COVERAGE" to (1.0 - trustedRatio),
        )
        return candidates.maxByOrNull { it.second }?.first ?: "NONE"
    }

    private fun buildWarnings(
        state: State,
        input: Input,
        meanUncertainty: Double,
        p95Uncertainty: Double,
        trustedRatio: Double,
        inflation: Double,
        history: History,
    ): List<String> = buildList {
        when (state) {
            State.UNKNOWN ->
                add("uncertainty-propagation-history-initializing")
            State.STABLE -> Unit
            State.WATCH ->
                add("3d-reconstruction-uncertainty-elevated")
            State.DEGRADED -> {
                add("3d-reconstruction-uncertainty-degraded")
                add("metric-output-should-be-held")
            }
            State.BLOCKED -> {
                add("fatal:3d-reconstruction-uncertainty-exceeded-production-envelope")
                add("fatal:metric-output-not-trustworthy")
            }
        }
        if (meanUncertainty >= config.warningMeanUncertaintyMeters) {
            add("mean-joint-uncertainty-elevated")
        }
        if (p95Uncertainty >= config.warningJointUncertaintyMeters) {
            add("p95-joint-uncertainty-elevated")
        }
        if (trustedRatio < config.targetTrustedJointRatio) {
            add("trusted-joint-coverage-below-target")
        }
        if (trustedRatio < config.minimumTrustedJointRatio) {
            add("trusted-joint-coverage-below-production-floor")
        }
        if (inflation >= 2.2) {
            add("uncertainty-inflation-factor-high")
        }
        if (input.geometricObservability < 0.62) {
            add("geometric-observability-is-limiting-3d-certainty")
        }
        if (input.synchronizationConfidence < 0.65) {
            add("synchronization-is-limiting-3d-certainty")
        }
        if (input.captureTimingConfidence < 0.65) {
            add("capture-timing-is-limiting-3d-certainty")
        }
        if (history.warningFrames >= config.persistentWarningFrames) {
            add("persistent-3d-uncertainty-instability")
        }
        if (history.blockedFrames >= config.persistentBlockedFrames) {
            add("persistent-3d-uncertainty-blocked")
        }
    }.distinct()

    private fun buildRecommendations(
        state: State,
        input: Input,
        meanUncertainty: Double,
        p95Uncertainty: Double,
        trustedRatio: Double,
        bottleneck: String,
    ): List<String> = buildList {
        when (bottleneck) {
            "REPROJECTION" ->
                add("improve 2d localization or reject cameras contributing high reprojection residuals")
            "OBSERVABILITY" ->
                add("increase multi-camera baseline diversity and keep the subject inside well-conditioned triangulation regions")
            "RIG_DRIFT" ->
                add("revalidate the camera rig before accepting another metric segment")
            "SYNC" ->
                add("tighten inter-camera clock alignment")
            "TIMING" ->
                add("restore stable frame cadence before trusting high-frequency 3d motion")
            "CAMERA_MOTION" ->
                add("stabilize the camera rig or improve inertial compensation")
            "OCCLUSION" ->
                add("increase view coverage and preserve redundant subject observations")
            "TRUSTED_COVERAGE" ->
                add("hold low-confidence joints and increase camera coverage")
            "JOINT_UNCERTAINTY", "P95_UNCERTAINTY" ->
                add("retain the last trusted metric pose until joint uncertainty recovers")
        }
        if (
            meanUncertainty >= config.warningMeanUncertaintyMeters ||
                p95Uncertainty >= config.warningJointUncertaintyMeters
        ) {
            add("preserve the uncertainty envelope downstream instead of treating all joints as equally precise")
        }
        if (trustedRatio < config.minimumTrustedJointRatio) {
            add("retain per-joint uncertainty in exports and hold low-confidence joints")
        }
        if (state == State.BLOCKED) {
            add("re-run calibration and reacquire a multi-view segment before resuming metric capture")
        }
        if (input.geometricObservability < 0.62) {
            add("prioritize camera placement and baseline geometry before increasing model complexity")
        }
    }.distinct()

    private fun blockedResult(
        timestampNanos: Long,
        message: String,
    ): Result =
        Result(
            timestampNanos = timestampNanos,
            state = State.BLOCKED,
            accepted = false,
            score = 0.0,
            confidence = 0.0,
            meanUncertaintyMeters = Double.POSITIVE_INFINITY,
            p95UncertaintyMeters = Double.POSITIVE_INFINITY,
            worstUncertaintyMeters = Double.POSITIVE_INFINITY,
            trustedJointRatio = 0.0,
            trustedJointCount = 0,
            jointCount = 0,
            inflationFactor = config.maximumInflation,
            dominantBottleneck = "JOINT_UNCERTAINTY",
            jointUncertainty = emptyList(),
            persistentWarningFrames = history?.warningFrames ?: 0,
            persistentBlockedFrames = history?.blockedFrames ?: 0,
            recoveryFrames = history?.recoveryFrames ?: 0,
            warnings = listOf("fatal:$message"),
            recommendations = listOf(
                "Hold metric 3D output until a valid uncertainty envelope is available.",
            ),
        )

    private fun quantile(
        values: List<Double>,
        fraction: Double,
    ): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        if (values.size == 1) return values.first()
        val position = fraction.coerceIn(0.0, 1.0) *
            (values.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, values.lastIndex)
        val blend = position - lower
        return values[lower] * (1.0 - blend) +
            values[upper] * blend
    }
}
