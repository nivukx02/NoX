package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.HandObservation
import com.nivukx.nox.core.model.HandSide
import com.nivukx.nox.core.model.Vec2
import kotlin.math.max
import kotlin.math.min

/**
 * Associates detector-level hands to global actor tracks without requiring unsupported MediaPipe ROI
 * processing.
 *
 * The association uses wrist/hand position, handedness, actor bounds, temporal velocity and
 * confidence. It produces an explicit "unassigned" state when evidence is ambiguous.
 */
class MultiActorHandAssociationEngine {

    data class Actor(
        val actorId: Long,
        val center: Vec2,
        val boundsLeft: Double,
        val boundsTop: Double,
        val boundsRight: Double,
        val boundsBottom: Double,
        val velocityXNormalizedPerSecond: Double,
        val velocityYNormalizedPerSecond: Double,
        val confidence: Double,
    )

    data class Assignment(
        val actorId: Long?,
        val handId: Long,
        val side: HandSide,
        val score: Double,
        val accepted: Boolean,
        val reason: String,
    )

    data class Result(
        val assignments: List<Assignment>,
        val unassignedHands: Int,
        val warnings: List<String>,
    )

    fun associate(
        actors: List<Actor>,
        hands: List<HandObservation>,
        normalizedDistanceGate: Double = 0.50,
        minimumScore: Double = 0.46,
    ): Result {
        require(normalizedDistanceGate > 0.0)
        require(minimumScore in 0.0..1.0)

        val candidates =
            hands.flatMap { hand ->
                val wrist =
                    hand.wrist
                if (wrist == null) {
                    emptyList()
                } else {
                    actors.map { actor ->
                        val wristX =
                            wrist.normalizedX.coerceIn(
                                -1.0,
                                2.0,
                            )
                        val wristY =
                            wrist.normalizedY.coerceIn(
                                -1.0,
                                2.0,
                            )
                        val inside =
                            wristX in actor.boundsLeft..actor.boundsRight &&
                                wristY in actor.boundsTop..actor.boundsBottom

                        val dx =
                            wristX -
                                actor.center.x
                        val dy =
                            wristY -
                                actor.center.y

                        val distance =
                            kotlin.math.hypot(
                                dx,
                                dy,
                            )

                        val spatial =
                            (
                                1.0 -
                                    distance /
                                    normalizedDistanceGate
                                ).coerceIn(0.0, 1.0)

                        val containment =
                            if (inside) 1.0 else 0.0

                        val velocityAgreement =
                            velocityAgreement(
                                hand = hand,
                                actor = actor,
                            )

                        val handedness =
                            if (hand.side ==
                                HandSide.UNKNOWN
                            ) {
                                0.45
                            } else {
                                0.70 +
                                    0.30 *
                                    hand.handednessConfidence
                            }

                        val score =
                            (
                                spatial * 0.36 +
                                    containment * 0.22 +
                                    velocityAgreement * 0.18 +
                                    handedness * 0.10 +
                                    hand.temporalStability.coerceIn(
                                        0.0,
                                        1.0,
                                    ) * 0.08 +
                                    hand.confidence.coerceIn(
                                        0.0,
                                        1.0,
                                    ) * 0.06
                                ).coerceIn(0.0, 1.0)

                        Triple(
                            hand,
                            actor,
                            score,
                        )
                    }
                }
            }.sortedByDescending { it.third }

        val usedHands =
            HashSet<Long>()
        val usedActors =
            HashSet<Long>()
        val assignments =
            ArrayList<Assignment>()

        candidates.forEach { (hand, actor, score) ->
            if (hand.handId in usedHands ||
                actor.actorId in usedActors
            ) {
                return@forEach
            }

            if (score < minimumScore) {
                return@forEach
            }

            usedHands += hand.handId
            usedActors += actor.actorId

            assignments +=
                Assignment(
                    actorId = actor.actorId,
                    handId = hand.handId,
                    side = hand.side,
                    score = score,
                    accepted = true,
                    reason = "GEOMETRY_TEMPORAL_ASSOCIATION",
                )
        }

        hands
            .filter {
                it.handId !in usedHands
            }
            .forEach { hand ->
                assignments +=
                    Assignment(
                        actorId = null,
                        handId = hand.handId,
                        side = hand.side,
                        score = 0.0,
                        accepted = false,
                        reason = "HAND_ACTOR_ASSOCIATION_AMBIGUOUS",
                    )
            }

        val unmatched =
            assignments.count {
                !it.accepted
            }

        val warnings =
            buildList {
                if (hands.isNotEmpty() && actors.isEmpty()) {
                    add("NO_ACTOR_TRACKS_AVAILABLE_FOR_HAND_ASSOCIATION")
                }
                if (unmatched > 0) {
                    add("ONE_OR_MORE_HANDS_LEFT_UNASSIGNED")
                }
            }

        return Result(
            assignments = assignments,
            unassignedHands = unmatched,
            warnings = warnings,
        )
    }

    private fun velocityAgreement(
        hand: HandObservation,
        actor: Actor,
    ): Double {
        val hx =
            hand.wristVelocityMetersPerSecond.x
        val hy =
            hand.wristVelocityMetersPerSecond.y

        val ax =
            actor.velocityXNormalizedPerSecond
        val ay =
            actor.velocityYNormalizedPerSecond

        val handNorm =
            max(
                1e-6,
                kotlin.math.hypot(hx, hy),
            )
        val actorNorm =
            max(
                1e-6,
                kotlin.math.hypot(ax, ay),
            )

        val dot =
            (
                hx * ax +
                    hy * ay
                ) /
                (handNorm * actorNorm)

        val directionAgreement =
            ((dot + 1.0) * 0.5)
                .coerceIn(0.0, 1.0)

        /*
         * Hand velocity is reported in meters/second while actor velocity is reported in normalized
         * image units/second. Their magnitudes are therefore not directly comparable. Only direction
         * is fused, with an activity-presence term that is intentionally scale-free.
         */
        val handActive =
            (
                handNorm /
                    (handNorm + 0.25)
                ).coerceIn(0.0, 1.0)
        val actorActive =
            (
                actorNorm /
                    (actorNorm + 0.10)
                ).coerceIn(0.0, 1.0)

        val activityAgreement =
            min(
                handActive,
                actorActive,
            )

        return (
            directionAgreement * 0.78 +
                activityAgreement * 0.22
            ).coerceIn(0.0, 1.0)
    }
}
