package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.SkeletonFrame
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Temporal commit controller for reconstructed 3D skeletons.
 *
 * Multi-view triangulation can return a numerically valid but temporally implausible frame. This
 * controller adds a final continuity decision after geometric and temporal quality checks.
 *
 * The important distinction is intentional:
 * - COMMIT means the current geometry becomes the stable reconstruction;
 * - HOLD_LAST means the current geometry is rejected for presentation, while a very recent stable
 *   reconstruction may be retained as the display frame;
 * - REJECT means no stable frame is allowed to hide the failure.
 *
 * This prevents one bad camera packet or synchronization spike from turning into a visible 3D
 * "teleport" without inventing measurements.
 */
class ReconstructionContinuityController(
    private val softAcceptScore: Double = 0.52,
    private val hardRejectScore: Double = 0.28,
    private val maxMeanJointJumpMeters: Double = 0.32,
    private val maxJointSpeedMetersPerSecond: Double = 11.0,
    private val holdMaxFrames: Int = 4,
    private val holdMaxAgeNanos: Long = 160_000_000L,
) {

    @Volatile
    private var tuning = ReconstructionContinuityTuning(
        softAcceptScore = softAcceptScore,
        hardRejectScore = hardRejectScore,
        maxJointJumpMeters = maxMeanJointJumpMeters,
        maxJointSpeedMetersPerSecond = maxJointSpeedMetersPerSecond,
        holdFrames = holdMaxFrames,
        holdAgeMillis = holdMaxAgeNanos / 1_000_000L,
        stabilitySensitivity = 0.65,
    )

    fun configure(newTuning: ReconstructionContinuityTuning) {
        tuning = newTuning
    }

    fun tuning(): ReconstructionContinuityTuning = tuning

    enum class Action {
        COMMIT,
        HOLD_LAST,
        REJECT,
    }

    data class Decision(
        val displayFrame: SkeletonFrame,
        val accepted: Boolean,
        val action: Action,
        val score: Double,
        val meanJointJumpMeters: Double,
        val maxJointSpeedMetersPerSecond: Double,
        val stableAgeNanos: Long,
        val reasons: List<String>,
    )

    private var stableFrame: SkeletonFrame? = null
    private var stableScore = 0.0
    private var stableTimestampNanos = Long.MIN_VALUE
    private var heldFrames = 0

    @Synchronized
    fun evaluate(
        candidate: SkeletonFrame,
        qualityAccepted: Boolean,
        qualityScore: Double,
        timestampNanos: Long = candidate.timestampNanos,
    ): Decision {
        require(timestampNanos >= 0L) {
            "timestampNanos must be non-negative"
        }

        val cfg = tuning
        val previous = stableFrame

        if (previous == null) {
            if (
                qualityAccepted &&
                qualityScore >= cfg.softAcceptScore &&
                candidate.globalConfidence >= 0.30
            ) {
                stableFrame = candidate
                stableScore = qualityScore
                stableTimestampNanos = timestampNanos
                heldFrames = 0

                return Decision(
                    displayFrame = candidate,
                    accepted = true,
                    action = Action.COMMIT,
                    score = qualityScore,
                    meanJointJumpMeters = 0.0,
                    maxJointSpeedMetersPerSecond = 0.0,
                    stableAgeNanos = 0L,
                    reasons = emptyList(),
                )
            }

            return Decision(
                displayFrame = candidate,
                accepted = false,
                action = Action.REJECT,
                score = qualityScore,
                meanJointJumpMeters = 0.0,
                maxJointSpeedMetersPerSecond = 0.0,
                stableAgeNanos = 0L,
                reasons = listOf("no-stable-baseline"),
            )
        }

        val delta = compare(previous, candidate, timestampNanos)
        val reasons = ArrayList<String>()

        if (!qualityAccepted) {
            reasons += "quality-gate"
        }

        if (qualityScore < cfg.softAcceptScore) {
            reasons += "soft-quality-threshold"
        }

        if (qualityScore < cfg.hardRejectScore) {
            reasons += "hard-quality-threshold"
        }

        if (delta.meanJumpMeters > cfg.maxJointJumpMeters) {
            reasons += "mean-joint-discontinuity"
        }

        if (delta.maxSpeedMetersPerSecond > cfg.maxJointSpeedMetersPerSecond) {
            reasons += "joint-speed-discontinuity"
        }

        val structurallySafe =
            candidate.joints.isNotEmpty() &&
                delta.meanJumpMeters <= maxMeanJointJumpMeters &&
                delta.maxSpeedMetersPerSecond <=
                    maxJointSpeedMetersPerSecond

        val commitEligible =
            qualityAccepted &&
                qualityScore >= cfg.softAcceptScore &&
                candidate.globalConfidence >= 0.30 &&
                structurallySafe &&
                reasons.isEmpty()

        if (commitEligible) {
            stableFrame = candidate
            stableScore = blend(
                current = stableScore,
                measured = qualityScore,
                alpha = 0.34,
            )
            stableTimestampNanos = timestampNanos
            heldFrames = 0

            return Decision(
                displayFrame = candidate,
                accepted = true,
                action = Action.COMMIT,
                score = blend(
                    current = stableScore,
                    measured = qualityScore,
                    alpha = 0.25,
                ),
                meanJointJumpMeters = delta.meanJumpMeters,
                maxJointSpeedMetersPerSecond =
                    delta.maxSpeedMetersPerSecond,
                stableAgeNanos = 0L,
                reasons = emptyList(),
            )
        }

        val stableAge = (
            timestampNanos - stableTimestampNanos
        ).coerceAtLeast(0L)

        val holdEligible =
            hardRejectScore <= qualityScore &&
                heldFrames < cfg.holdFrames &&
                stableAge <= cfg.holdAgeMillis * 1_000_000L &&
                structurallySafe &&
                previous.joints.isNotEmpty()

        if (holdEligible) {
            heldFrames += 1

            val decay = exp(
                -stableAge.toDouble() /
                    holdMaxAgeNanos.toDouble().coerceAtLeast(1.0),
            )

            return Decision(
                displayFrame = previous.copy(
                    timestampNanos = timestampNanos,
                    globalConfidence = (
                        previous.globalConfidence *
                            0.94 *
                            decay
                        ).coerceIn(0.0, 1.0),
                ),
                accepted = false,
                action = Action.HOLD_LAST,
                score = (
                    0.60 * stableScore +
                        0.40 * qualityScore
                    ).coerceIn(0.0, 1.0),
                meanJointJumpMeters = delta.meanJumpMeters,
                maxJointSpeedMetersPerSecond =
                    delta.maxSpeedMetersPerSecond,
                stableAgeNanos = stableAge,
                reasons = (
                    reasons +
                        "held-last-stable-reconstruction"
                ).distinct(),
            )
        }

        heldFrames = 0

        return Decision(
            displayFrame = candidate,
            accepted = false,
            action = Action.REJECT,
            score = qualityScore.coerceIn(0.0, 1.0),
            meanJointJumpMeters = delta.meanJumpMeters,
            maxJointSpeedMetersPerSecond =
                delta.maxSpeedMetersPerSecond,
            stableAgeNanos = stableAge,
            reasons = (
                reasons +
                    "no-temporal-continuity"
            ).distinct(),
        )
    }

    fun stableSnapshot(): SkeletonFrame? = stableFrame

    fun stableScore(): Double = stableScore

    @Synchronized
    fun reset() {
        stableFrame = null
        stableScore = 0.0
        stableTimestampNanos = Long.MIN_VALUE
        heldFrames = 0
    }

    private data class Delta(
        val comparedJoints: Int,
        val meanJumpMeters: Double,
        val maxSpeedMetersPerSecond: Double,
    )

    private fun compare(
        previous: SkeletonFrame,
        candidate: SkeletonFrame,
        timestampNanos: Long,
    ): Delta {
        if (previous.joints.isEmpty() || candidate.joints.isEmpty()) {
            return Delta(
                comparedJoints = 0,
                meanJumpMeters = Double.POSITIVE_INFINITY,
                maxSpeedMetersPerSecond = Double.POSITIVE_INFINITY,
            )
        }

        val previousById = previous.joints.associateBy {
            it.landmarkId
        }

        val dt = (
            timestampNanos - previous.timestampNanos
        ).coerceAtLeast(1L) / 1e9

        var count = 0
        var sum = 0.0
        var maxSpeed = 0.0

        candidate.joints.forEach { current ->
            val old = previousById[current.landmarkId]
                ?: return@forEach

            val jump = current.position.distanceTo(
                old.position,
            )

            if (!jump.isFinite()) {
                return@forEach
            }

            count += 1
            sum += jump
            maxSpeed = max(
                maxSpeed,
                jump / dt.coerceAtLeast(1.0 / 240.0),
            )
        }

        if (count == 0) {
            return Delta(
                comparedJoints = 0,
                meanJumpMeters = Double.POSITIVE_INFINITY,
                maxSpeedMetersPerSecond = Double.POSITIVE_INFINITY,
            )
        }

        return Delta(
            comparedJoints = count,
            meanJumpMeters = (
                sum / count.toDouble()
            ).coerceAtLeast(0.0),
            maxSpeedMetersPerSecond = maxSpeed,
        )
    }

    private fun blend(
        current: Double,
        measured: Double,
        alpha: Double,
    ): Double {
        val a = min(1.0, max(0.0, alpha))
        return current * (1.0 - a) + measured * a
    }
}
