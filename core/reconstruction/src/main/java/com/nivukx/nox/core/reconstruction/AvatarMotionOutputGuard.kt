
package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.AvatarPose
import com.nivukx.nox.core.model.ReconstructionConfidenceBand
import com.nivukx.nox.core.model.SkeletonFrame

/**
 * Separates "motion was reconstructed" from "motion is safe to drive an avatar with".
 *
 * A production mocap system should continue preserving forensic motion records even when a live
 * avatar update is rejected. This guard therefore never mutates the source motion frame.
 */
class AvatarMotionOutputGuard(
    private val config: Config = Config(),
) {

    data class Config(
        val minimumConfidence: Double = 0.55,
        val minimumTemporalStability: Double = 0.56,
        val maximumReprojectionErrorPx: Double = 5.0,
        val maximumUncertaintyMeters: Double = 0.25,
        val minimumUncertaintyConfidence: Double = 0.55,
        val minimumTrustedJointRatio: Double = 0.62,
        val maximumP95UncertaintyMeters: Double = 0.18,
        val maximumPredictedFraction: Double = 0.42,
        val holdFrames: Int = 3,
        val decayPerRejectedFrame: Double = 0.18,
    )

    data class Input(
        val skeleton: SkeletonFrame,
        val avatarPose: AvatarPose,
        val confidenceBand: ReconstructionConfidenceBand,
        val reprojectionErrorPx: Double,
        val meanUncertaintyMeters: Double,
        val uncertaintyConfidence: Double = 1.0,
        val p95UncertaintyMeters: Double = 0.0,
        val trustedJointRatio: Double = 1.0,
        val uncertaintyBottleneck: String = "NONE",
    )

    data class Decision(
        val accepted: Boolean,
        val pose: AvatarPose?,
        val confidence: Double,
        val reasons: List<String>,
        val usedHold: Boolean,
        val heldFrames: Int,
    )

    private var lastAccepted: AvatarPose? = null
    private var heldFrames = 0

    @Synchronized
    fun evaluate(input: Input): Decision {
        val frame = input.skeleton
        val predictedFraction =
            if (frame.joints.isEmpty()) {
                1.0
            } else {
                frame.joints.count {
                    it.confidence < 0.30 ||
                        it.positionUncertaintyMeters >
                            config.maximumUncertaintyMeters * 1.5
                }.toDouble() / frame.joints.size.toDouble()
            }

        val reasons = mutableListOf<String>()

        if (frame.globalConfidence < config.minimumConfidence) {
            reasons += "skeleton confidence below live-avatar gate"
        }

        if (input.reprojectionErrorPx > config.maximumReprojectionErrorPx) {
            reasons += "reprojection residual above live-avatar gate"
        }

        if (input.meanUncertaintyMeters > config.maximumUncertaintyMeters) {
            reasons += "metric uncertainty above live-avatar gate"
        }

        if (input.uncertaintyConfidence < config.minimumUncertaintyConfidence) {
            reasons += "global reconstruction uncertainty confidence below live-avatar gate"
        }

        if (input.p95UncertaintyMeters > config.maximumP95UncertaintyMeters) {
            reasons += "P95 reconstruction uncertainty above live-avatar gate"
        }

        if (input.trustedJointRatio < config.minimumTrustedJointRatio) {
            reasons += "trusted-joint coverage below live-avatar gate"
        }

        if (input.uncertaintyBottleneck != "NONE") {
            reasons += "uncertainty bottleneck: " + input.uncertaintyBottleneck
        }

        if (predictedFraction > config.maximumPredictedFraction) {
            reasons += "predicted-node fraction above live-avatar gate"
        }

        if (input.confidenceBand == ReconstructionConfidenceBand.UNTRUSTED) {
            reasons += "confidence band is untrusted"
        }

        val accepted =
            reasons.isEmpty() &&
                frame.joints.isNotEmpty() &&
                frame.globalConfidence.isFinite()

        if (accepted) {
            heldFrames = 0
            val pose = input.avatarPose
            lastAccepted = pose

            return Decision(
                accepted = true,
                pose = pose,
                confidence = (
                    frame.globalConfidence *
                        (1.0 - predictedFraction * 0.20) *
                        input.uncertaintyConfidence.coerceIn(0.0, 1.0) *
                        (
                            0.70 +
                                input.trustedJointRatio.coerceIn(0.0, 1.0) *
                                    0.30
                        )
                    ).coerceIn(0.0, 1.0),
                reasons = if (pose == null) {
                    listOf("avatar retargeting produced no valid bones")
                } else {
                    emptyList()
                },
                usedHold = false,
                heldFrames = 0,
            )
        }

        if (lastAccepted != null && heldFrames < config.holdFrames) {
            heldFrames += 1
            val decayed = config.decayPerRejectedFrame *
                heldFrames.toDouble()

            return Decision(
                accepted = true,
                pose = lastAccepted?.copy(
                    confidence = (
                        lastAccepted?.confidence ?: 0.0
                    ) * (1.0 - decayed).coerceAtLeast(0.0),
                ),
                confidence = (
                    frame.globalConfidence *
                        (1.0 - decayed).coerceAtLeast(0.0)
                    ).coerceIn(0.0, 1.0),
                reasons = reasons.distinct(),
                usedHold = true,
                heldFrames = heldFrames,
            )
        }

        heldFrames = 0

        return Decision(
            accepted = false,
            pose = null,
            confidence = frame.globalConfidence.coerceIn(0.0, 1.0),
            reasons = reasons.distinct(),
            usedHold = false,
            heldFrames = 0,
        )
    }

    @Synchronized
    fun reset() {
        lastAccepted = null
        heldFrames = 0
    }

}
