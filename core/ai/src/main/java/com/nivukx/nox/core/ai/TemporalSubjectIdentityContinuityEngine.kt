package com.nivukx.nox.core.ai

import kotlin.math.abs
import kotlin.math.max

/**
 * Temporal identity gate for multi-subject motion capture.
 *
 * The detector's strongest current candidate is not automatically treated as the active actor.
 * This gate keeps identity decisions temporally coherent, explicitly models short occlusions, and
 * requires sustained evidence before switching actors. It is deliberately model-agnostic so it can
 * sit after any detector/tracker implementation without pretending that a transient score spike is
 * ground truth.
 */
class TemporalSubjectIdentityContinuityEngine(
    private val config: Config = Config(),
) {

    enum class Action {
        SEARCH,
        ACQUIRE,
        KEEP,
        HOLD,
        SWITCH,
        RECOVER,
    }

    data class Config(
        val switchConfidenceMargin: Double = 0.18,
        val holdAmbiguityThreshold: Double = 0.58,
        val maximumRecoveryGapNanos: Long = 420_000_000L,
        val minimumSwitchEvidenceFrames: Int = 3,
        val minimumKeepConfidence: Double = 0.48,
        val recoveryConfidenceFloor: Double = 0.30,
        val ambiguityRiskWeight: Double = 0.34,
        val viewpointRiskWeight: Double = 0.16,
        val temporalInstabilityWeight: Double = 0.18,
        val occlusionRiskWeight: Double = 0.17,
        val competingSubjectWeight: Double = 0.15,
    ) {
        init {
            require(switchConfidenceMargin in 0.0..1.0)
            require(holdAmbiguityThreshold in 0.0..1.0)
            require(maximumRecoveryGapNanos > 0L)
            require(minimumSwitchEvidenceFrames >= 1)
            require(minimumKeepConfidence in 0.0..1.0)
            require(recoveryConfidenceFloor in 0.0..1.0)
            val weights = listOf(
                ambiguityRiskWeight,
                viewpointRiskWeight,
                temporalInstabilityWeight,
                occlusionRiskWeight,
                competingSubjectWeight,
            )
            require(weights.all { it >= 0.0 && it <= 1.0 })
            require(weights.sum() <= 1.0 + 1e-9)
        }
    }

    data class FrameEvidence(
        val timestampNanos: Long,
        val candidateTrackId: Long?,
        val candidateConfidence: Double,
        val subjectCount: Int,
        val occlusionRisk: Double,
        val temporalStability: Double,
        val leftRightAmbiguity: Double,
        val viewpointChallenge: Double,
    ) {
        init {
            require(subjectCount >= 0)
            require(timestampNanos >= 0L)
            require(candidateConfidence.isFinite())
            require(occlusionRisk.isFinite())
            require(temporalStability.isFinite())
            require(leftRightAmbiguity.isFinite())
            require(viewpointChallenge.isFinite())
        }
    }

    data class Result(
        val selectedTrackId: Long?,
        val candidateTrackId: Long?,
        val action: Action,
        val continuityScore: Double,
        val identitySwapRisk: Double,
        val predictedContinuation: Boolean,
        val switchEvidenceFrames: Int,
        val lastStableTrackAgeNanos: Long?,
        val warnings: List<String>,
        val reason: String,
    ) {
        val lockTrusted: Boolean
            get() = selectedTrackId != null && continuityScore >= 0.55
    }

    private var activeTrackId: Long? = null
    private var activeConfidence = 0.0
    private var lastStableTimestampNanos: Long? = null
    private var switchCandidateTrackId: Long? = null
    private var switchEvidenceFrames = 0

    @Synchronized
    fun evaluate(input: FrameEvidence): Result {
        val candidateId = input.candidateTrackId
        val candidateConfidence = input.candidateConfidence.coerceIn(0.0, 1.0)
        val occlusionRisk = input.occlusionRisk.coerceIn(0.0, 1.0)
        val temporalStability = input.temporalStability.coerceIn(0.0, 1.0)
        val ambiguity = input.leftRightAmbiguity.coerceIn(0.0, 1.0)
        val viewpoint = input.viewpointChallenge.coerceIn(0.0, 1.0)

        if (activeTrackId == null) {
            resetSwitchEvidence()
            if (candidateId == null || candidateConfidence < config.recoveryConfidenceFloor) {
                return Result(
                    selectedTrackId = null,
                    candidateTrackId = candidateId,
                    action = Action.SEARCH,
                    continuityScore = 0.0,
                    identitySwapRisk = identitySwapRisk(input),
                    predictedContinuation = false,
                    switchEvidenceFrames = 0,
                    lastStableTrackAgeNanos = null,
                    warnings = listOf("subject-identity-not-established"),
                    reason = "no-stable-subject-identity",
                )
            }

            activeTrackId = candidateId
            activeConfidence = candidateConfidence
            lastStableTimestampNanos = input.timestampNanos
            return Result(
                selectedTrackId = candidateId,
                candidateTrackId = candidateId,
                action = Action.ACQUIRE,
                continuityScore = continuityScore(
                    candidateConfidence = candidateConfidence,
                    temporalStability = temporalStability,
                    occlusionRisk = occlusionRisk,
                    ambiguity = ambiguity,
                    viewpoint = viewpoint,
                ),
                identitySwapRisk = 0.0,
                predictedContinuation = false,
                switchEvidenceFrames = 0,
                lastStableTrackAgeNanos = 0L,
                warnings = emptyList(),
                reason = "initial-subject-acquisition",
            )
        }

        val stableId = activeTrackId
        val age = lastStableTimestampNanos?.let {
            max(0L, input.timestampNanos - it)
        }

        if (candidateId == null) {
            resetSwitchEvidence()
            val recoverable = age != null &&
                age <= config.maximumRecoveryGapNanos &&
                activeConfidence >= config.recoveryConfidenceFloor

            if (recoverable) {
                activeConfidence = (
                    activeConfidence * 0.88 +
                        temporalStability * 0.07 +
                        (1.0 - occlusionRisk) * 0.05
                    ).coerceIn(0.0, 1.0)

                return Result(
                    selectedTrackId = stableId,
                    candidateTrackId = null,
                    action = Action.RECOVER,
                    continuityScore = (
                        activeConfidence * 0.54 +
                            temporalStability * 0.20 +
                            (1.0 - occlusionRisk) * 0.12 +
                            (1.0 - viewpoint) * 0.08 +
                            (1.0 - ambiguity) * 0.06
                        ).coerceIn(0.0, 1.0),
                    identitySwapRisk = (
                        0.25 + occlusionRisk * 0.55 + viewpoint * 0.20
                    ).coerceIn(0.0, 1.0),
                    predictedContinuation = true,
                    switchEvidenceFrames = 0,
                    lastStableTrackAgeNanos = age,
                    warnings = listOf("subject-temporarily-occluded"),
                    reason = "short-gap-prediction-keeps-last-stable-identity",
                )
            }

            activeTrackId = null
            activeConfidence = 0.0
            lastStableTimestampNanos = null
            return Result(
                selectedTrackId = null,
                candidateTrackId = null,
                action = Action.SEARCH,
                continuityScore = 0.0,
                identitySwapRisk = 0.0,
                predictedContinuation = false,
                switchEvidenceFrames = 0,
                lastStableTrackAgeNanos = age,
                warnings = listOf("subject-identity-recovery-window-expired"),
                reason = "stable-identity-no-longer-recoverable",
            )
        }

        if (candidateId == stableId) {
            resetSwitchEvidence()
            activeConfidence = (
                activeConfidence * 0.52 +
                    candidateConfidence * 0.32 +
                    temporalStability * 0.16
                ).coerceIn(0.0, 1.0)
            lastStableTimestampNanos = input.timestampNanos

            return Result(
                selectedTrackId = stableId,
                candidateTrackId = candidateId,
                action = Action.KEEP,
                continuityScore = continuityScore(
                    candidateConfidence = candidateConfidence,
                    temporalStability = temporalStability,
                    occlusionRisk = occlusionRisk,
                    ambiguity = ambiguity,
                    viewpoint = viewpoint,
                ),
                identitySwapRisk = (
                    ambiguity * config.ambiguityRiskWeight +
                        viewpoint * config.viewpointRiskWeight +
                        (1.0 - temporalStability) * config.temporalInstabilityWeight +
                        occlusionRisk * config.occlusionRiskWeight
                    ).coerceIn(0.0, 1.0),
                predictedContinuation = false,
                switchEvidenceFrames = 0,
                lastStableTrackAgeNanos = 0L,
                warnings = if (candidateConfidence < config.minimumKeepConfidence) {
                    listOf("active-subject-confidence-reduced")
                } else {
                    emptyList()
                },
                reason = "candidate-matches-stable-identity",
            )
        }

        val competingConfidence = candidateConfidence
        val margin = competingConfidence - activeConfidence
        val swapRisk = identitySwapRisk(input)
        val highlyAmbiguous = ambiguity >= config.holdAmbiguityThreshold
        val requiresConservativeHold =
            highlyAmbiguous ||
                swapRisk >= 0.58 ||
                (
                    input.subjectCount > 1 &&
                        candidateConfidence < activeConfidence + config.switchConfidenceMargin
                    )

        if (requiresConservativeHold && margin < config.switchConfidenceMargin) {
            resetSwitchEvidence()
            return Result(
                selectedTrackId = stableId,
                candidateTrackId = candidateId,
                action = Action.HOLD,
                continuityScore = (
                    0.46 * activeConfidence +
                        0.24 * temporalStability +
                        0.16 * (1.0 - swapRisk) +
                        0.14 * (1.0 - ambiguity)
                    ).coerceIn(0.0, 1.0),
                identitySwapRisk = swapRisk,
                predictedContinuation = true,
                switchEvidenceFrames = 0,
                lastStableTrackAgeNanos = age,
                warnings = listOf(
                    "competing-subject-identity-ambiguous",
                    "track-switch-held-for-temporal-evidence",
                ),
                reason = "hold-active-identity-until-cross-frame-evidence-improves",
            )
        }

        if (switchCandidateTrackId == candidateId) {
            switchEvidenceFrames += 1
        } else {
            switchCandidateTrackId = candidateId
            switchEvidenceFrames = 1
        }

        val sufficientEvidence =
            switchEvidenceFrames >= config.minimumSwitchEvidenceFrames &&
                margin >= config.switchConfidenceMargin &&
                candidateConfidence >= config.minimumKeepConfidence

        if (sufficientEvidence) {
            activeTrackId = candidateId
            activeConfidence = candidateConfidence
            lastStableTimestampNanos = input.timestampNanos
            resetSwitchEvidence()

            return Result(
                selectedTrackId = candidateId,
                candidateTrackId = candidateId,
                action = Action.SWITCH,
                continuityScore = (
                    0.48 * candidateConfidence +
                        0.22 * temporalStability +
                        0.16 * (1.0 - ambiguity) +
                        0.14 * (1.0 - occlusionRisk)
                    ).coerceIn(0.0, 1.0),
                identitySwapRisk = (
                    swapRisk * 0.50
                ).coerceIn(0.0, 1.0),
                predictedContinuation = false,
                switchEvidenceFrames = config.minimumSwitchEvidenceFrames,
                lastStableTrackAgeNanos = 0L,
                warnings = listOf("subject-identity-switched-after-sustained-evidence"),
                reason = "competing-track-sustained-confidence-margin",
            )
        }

        return Result(
            selectedTrackId = stableId,
            candidateTrackId = candidateId,
            action = Action.HOLD,
            continuityScore = (
                activeConfidence * 0.44 +
                    temporalStability * 0.24 +
                    candidateConfidence * 0.12 +
                    (1.0 - swapRisk) * 0.20
                ).coerceIn(0.0, 1.0),
            identitySwapRisk = swapRisk,
            predictedContinuation = true,
            switchEvidenceFrames = switchEvidenceFrames,
            lastStableTrackAgeNanos = age,
            warnings = listOf("pending-subject-identity-switch"),
            reason = "candidate-needs-more-temporal-evidence",
        )
    }

    @Synchronized
    fun activeTrack(): Long? = activeTrackId

    @Synchronized
    fun reset() {
        activeTrackId = null
        activeConfidence = 0.0
        lastStableTimestampNanos = null
        resetSwitchEvidence()
    }

    private fun resetSwitchEvidence() {
        switchCandidateTrackId = null
        switchEvidenceFrames = 0
    }

    private fun continuityScore(
        candidateConfidence: Double,
        temporalStability: Double,
        occlusionRisk: Double,
        ambiguity: Double,
        viewpoint: Double,
    ): Double = (
        candidateConfidence * 0.42 +
            temporalStability * 0.24 +
            (1.0 - occlusionRisk) * 0.14 +
            (1.0 - ambiguity) * 0.10 +
            (1.0 - viewpoint) * 0.10
        ).coerceIn(0.0, 1.0)

    private fun identitySwapRisk(input: FrameEvidence): Double {
        val ambiguity = input.leftRightAmbiguity.coerceIn(0.0, 1.0)
        val viewpoint = input.viewpointChallenge.coerceIn(0.0, 1.0)
        val temporalInstability =
            1.0 - input.temporalStability.coerceIn(0.0, 1.0)
        val occlusion = input.occlusionRisk.coerceIn(0.0, 1.0)
        val competingSubjects = if (input.subjectCount > 1) {
            1.0
        } else {
            0.0
        }

        return (
            ambiguity * config.ambiguityRiskWeight +
                viewpoint * config.viewpointRiskWeight +
                temporalInstability * config.temporalInstabilityWeight +
                occlusion * config.occlusionRiskWeight +
                competingSubjects * config.competingSubjectWeight
            ).coerceIn(0.0, 1.0)
    }
}
