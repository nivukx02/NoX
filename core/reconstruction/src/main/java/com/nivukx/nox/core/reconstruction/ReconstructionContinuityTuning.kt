package com.nivukx.nox.core.reconstruction

/**
 * Runtime reconstruction continuity policy.
 *
 * The policy changes how aggressively the presentation layer commits or holds a reconstructed
 * frame. Geometric validity remains governed by the reconstruction quality gate.
 */
data class ReconstructionContinuityTuning(
    val softAcceptScore: Double = 0.52,
    val hardRejectScore: Double = 0.28,
    val maxJointJumpMeters: Double = 0.32,
    val maxJointSpeedMetersPerSecond: Double = 11.0,
    val holdFrames: Int = 4,
    val holdAgeMillis: Long = 160L,
    val stabilitySensitivity: Double = 0.65,
) {
    init {
        require(softAcceptScore in 0.30..0.95)
        require(hardRejectScore in 0.10..softAcceptScore)
        require(maxJointJumpMeters in 0.05..1.25)
        require(maxJointSpeedMetersPerSecond in 2.0..40.0)
        require(holdFrames in 0..12)
        require(holdAgeMillis in 40L..600L)
        require(stabilitySensitivity in 0.20..1.20)
    }

    companion object {
        fun cinematic(): ReconstructionContinuityTuning =
            ReconstructionContinuityTuning(
                softAcceptScore = 0.58,
                hardRejectScore = 0.34,
                maxJointJumpMeters = 0.24,
                maxJointSpeedMetersPerSecond = 8.5,
                holdFrames = 6,
                holdAgeMillis = 220L,
                stabilitySensitivity = 0.86,
            )

        fun liveAction(): ReconstructionContinuityTuning =
            ReconstructionContinuityTuning(
                softAcceptScore = 0.46,
                hardRejectScore = 0.22,
                maxJointJumpMeters = 0.42,
                maxJointSpeedMetersPerSecond = 16.0,
                holdFrames = 3,
                holdAgeMillis = 140L,
                stabilitySensitivity = 0.52,
            )

        fun precision(): ReconstructionContinuityTuning =
            ReconstructionContinuityTuning(
                softAcceptScore = 0.64,
                hardRejectScore = 0.38,
                maxJointJumpMeters = 0.18,
                maxJointSpeedMetersPerSecond = 7.0,
                holdFrames = 5,
                holdAgeMillis = 200L,
                stabilitySensitivity = 1.0,
            )

        fun creature(): ReconstructionContinuityTuning =
            ReconstructionContinuityTuning(
                softAcceptScore = 0.50,
                hardRejectScore = 0.24,
                maxJointJumpMeters = 0.50,
                maxJointSpeedMetersPerSecond = 20.0,
                holdFrames = 4,
                holdAgeMillis = 170L,
                stabilitySensitivity = 0.58,
            )

        fun facial(): ReconstructionContinuityTuning =
            ReconstructionContinuityTuning(
                softAcceptScore = 0.62,
                hardRejectScore = 0.40,
                maxJointJumpMeters = 0.16,
                maxJointSpeedMetersPerSecond = 6.5,
                holdFrames = 7,
                holdAgeMillis = 240L,
                stabilitySensitivity = 0.94,
            )

        fun handDetail(): ReconstructionContinuityTuning =
            ReconstructionContinuityTuning(
                softAcceptScore = 0.54,
                hardRejectScore = 0.30,
                maxJointJumpMeters = 0.14,
                maxJointSpeedMetersPerSecond = 7.5,
                holdFrames = 5,
                holdAgeMillis = 180L,
                stabilitySensitivity = 0.82,
            )

    }
}
