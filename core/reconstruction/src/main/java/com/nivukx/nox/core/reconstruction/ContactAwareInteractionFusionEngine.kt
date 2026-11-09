package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Contact-aware human/object motion fusion.
 *
 * Recent markerless + inertial research highlights contact as a useful high-level signal for
 * switching between kinematic and inertial reasoning during human-object interaction. NoX keeps
 * that concept model-agnostic: detectors may provide contact probability, object pose, hand pose
 * and inertial confidence, while this engine produces a bounded interaction state and explicit
 * relative-motion residual.
 *
 * Contact is never treated as proof of rigid attachment. The engine requires temporal persistence,
 * spatial proximity and relative-motion agreement before promoting a contact to a stable constraint.
 */
class ContactAwareInteractionFusionEngine(
    private val config: Config = Config(),
) {
    enum class ContactState {
        UNKNOWN,
        APPROACHING,
        CONTACT_CANDIDATE,
        STABLE_CONTACT,
        SEPARATING,
        LOST,
    }

    enum class ReasoningMode {
        KINEMATIC,
        INERTIAL,
        FUSED_CONTACT,
        HOLD,
    }

    data class Config(
        val enterContactProbability: Double = 0.78,
        val stableContactProbability: Double = 0.88,
        val exitContactProbability: Double = 0.34,
        val maximumDistanceMeters: Double = 0.12,
        val stableDistanceMeters: Double = 0.055,
        val maximumRelativeSpeedMetersPerSecond: Double = 0.65,
        val stableRelativeSpeedMetersPerSecond: Double = 0.18,
        val minimumContactConfidence: Double = 0.58,
        val minimumInertialConfidence: Double = 0.50,
        val minimumTemporalConsistency: Double = 0.64,
        val contactEnterFrames: Int = 3,
        val contactStableFrames: Int = 3,
        val contactExitFrames: Int = 3,
        val historyPerPair: Int = 24,
        val residualScaleMeters: Double = 0.025,
    ) {
        init {
            require(enterContactProbability in 0.0..1.0)
            require(stableContactProbability >= enterContactProbability)
            require(exitContactProbability < enterContactProbability)
            require(maximumDistanceMeters > stableDistanceMeters)
            require(maximumRelativeSpeedMetersPerSecond >
                stableRelativeSpeedMetersPerSecond)
            require(minimumContactConfidence in 0.0..1.0)
            require(minimumInertialConfidence in 0.0..1.0)
            require(minimumTemporalConsistency in 0.0..1.0)
            require(contactEnterFrames >= 1)
            require(contactStableFrames >= 1)
            require(contactExitFrames >= 1)
            require(historyPerPair >= 4)
            require(residualScaleMeters > 0.0)
        }
    }

    data class HandState(
        val handId: String,
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val confidence: Double,
    )

    data class ObjectState(
        val objectId: String,
        val timestampNanos: Long,
        val positionMeters: Vec3,
        val velocityMetersPerSecond: Vec3,
        val confidence: Double,
    )

    data class ContactObservation(
        val hand: HandState,
        val trackedObject: ObjectState,
        val contactProbability: Double,
        val inertialConfidence: Double,
        val temporalConsistency: Double,
        val detectorAgreement: Double = 1.0,
    ) {
        init {
            require(contactProbability in 0.0..1.0)
            require(inertialConfidence in 0.0..1.0)
            require(temporalConsistency in 0.0..1.0)
            require(detectorAgreement in 0.0..1.0)
        }
    }

    data class PairResult(
        val handId: String,
        val objectId: String,
        val timestampNanos: Long,
        val state: ContactState,
        val reasoningMode: ReasoningMode,
        val contactConfidence: Double,
        val distanceMeters: Double,
        val relativeSpeedMetersPerSecond: Double,
        val relativePositionResidualMeters: Double,
        val fusedConfidence: Double,
        val enterFrames: Int,
        val stableFrames: Int,
        val exitFrames: Int,
        val warnings: List<String>,
        val recommendations: List<String>,
    )

    private data class PairState(
        var state: ContactState = ContactState.UNKNOWN,
        var enterFrames: Int = 0,
        var stableFrames: Int = 0,
        var exitFrames: Int = 0,
        var confidenceEma: Double = 0.0,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
        val residualHistory: ArrayDeque<Double> = ArrayDeque(),
    )

    private val pairStates = linkedMapOf<String, PairState>()

    @Synchronized
    fun evaluate(input: ContactObservation): PairResult {
        val key = input.hand.handId + "::" + input.trackedObject.objectId
        val state = pairStates.getOrPut(key) { PairState() }
        val timestamp = max(input.hand.timestampNanos, input.trackedObject.timestampNanos)

        if (
            state.lastTimestampNanos != Long.MIN_VALUE &&
            timestamp < state.lastTimestampNanos
        ) {
            return PairResult(
                handId = input.hand.handId,
                objectId = input.trackedObject.objectId,
                timestampNanos = timestamp,
                state = state.state,
                reasoningMode = ReasoningMode.HOLD,
                contactConfidence = state.confidenceEma,
                distanceMeters = Double.POSITIVE_INFINITY,
                relativeSpeedMetersPerSecond = Double.POSITIVE_INFINITY,
                relativePositionResidualMeters = Double.POSITIVE_INFINITY,
                fusedConfidence = 0.0,
                enterFrames = state.enterFrames,
                stableFrames = state.stableFrames,
                exitFrames = state.exitFrames,
                warnings = listOf("OUT_OF_ORDER_CONTACT_SAMPLE"),
                recommendations = listOf(
                    "Preserve monotonic hand/object timestamps before enforcing contact constraints.",
                ),
            )
        }

        state.lastTimestampNanos = timestamp

        val distance = input.hand.positionMeters
            .distanceTo(input.trackedObject.positionMeters)
        val relativeVelocity = input.hand.velocityMetersPerSecond -
            input.trackedObject.velocityMetersPerSecond
        val relativeSpeed = relativeVelocity.norm()

        val distanceScore = inverse(
            distance,
            config.stableDistanceMeters,
            config.maximumDistanceMeters,
        )
        val relativeSpeedScore = inverse(
            relativeSpeed,
            config.stableRelativeSpeedMetersPerSecond,
            config.maximumRelativeSpeedMetersPerSecond,
        )
        val temporalScore = input.temporalConsistency
        val sensorScore = (
            input.inertialConfidence * 0.50 +
                input.detectorAgreement * 0.50
            ).coerceIn(0.0, 1.0)

        val contactEvidence = (
            input.contactProbability * 0.40 +
                distanceScore * 0.20 +
                relativeSpeedScore * 0.16 +
                temporalScore * 0.12 +
                sensorScore * 0.12
            ).coerceIn(0.0, 1.0)

        state.confidenceEma =
            if (state.enterFrames == 0 &&
                state.stableFrames == 0 &&
                state.exitFrames == 0 &&
                state.confidenceEma == 0.0
            ) {
                contactEvidence
            } else {
                state.confidenceEma * 0.72 +
                    contactEvidence * 0.28
            }

        val candidate =
            contactEvidence >= config.enterContactProbability &&
                distance <= config.maximumDistanceMeters &&
                relativeSpeed <= config.maximumRelativeSpeedMetersPerSecond &&
                input.temporalConsistency >=
                    config.minimumTemporalConsistency

        val stableCandidate =
            contactEvidence >= config.stableContactProbability &&
                distance <= config.stableDistanceMeters &&
                relativeSpeed <= config.stableRelativeSpeedMetersPerSecond &&
                input.inertialConfidence >= config.minimumInertialConfidence &&
                input.detectorAgreement >=
                    config.minimumContactConfidence

        val separated =
            input.contactProbability <= config.exitContactProbability ||
                distance > config.maximumDistanceMeters ||
                relativeSpeed > config.maximumRelativeSpeedMetersPerSecond

        when {
            stableCandidate -> {
                state.enterFrames += 1
                state.stableFrames += 1
                state.exitFrames = 0
            }
            candidate -> {
                state.enterFrames += 1
                state.stableFrames =
                    max(0, state.stableFrames - 1)
                state.exitFrames = 0
            }
            separated -> {
                state.exitFrames += 1
                state.enterFrames =
                    max(0, state.enterFrames - 1)
                state.stableFrames =
                    max(0, state.stableFrames - 1)
            }
            else -> {
                state.enterFrames =
                    max(0, state.enterFrames - 1)
                state.stableFrames =
                    max(0, state.stableFrames - 1)
                state.exitFrames =
                    max(0, state.exitFrames - 1)
            }
        }

        state.state =
            when (state.state) {
                ContactState.UNKNOWN ->
                    if (candidate) {
                        ContactState.CONTACT_CANDIDATE
                    } else {
                        ContactState.UNKNOWN
                    }

                ContactState.APPROACHING ->
                    if (candidate && state.enterFrames >=
                        config.contactEnterFrames) {
                        ContactState.CONTACT_CANDIDATE
                    } else if (separated) {
                        ContactState.LOST
                    } else {
                        ContactState.APPROACHING
                    }

                ContactState.CONTACT_CANDIDATE ->
                    if (stableCandidate &&
                        state.stableFrames >= config.contactStableFrames) {
                        ContactState.STABLE_CONTACT
                    } else if (separated &&
                        state.exitFrames >= config.contactExitFrames) {
                        ContactState.SEPARATING
                    } else {
                        ContactState.CONTACT_CANDIDATE
                    }

                ContactState.STABLE_CONTACT ->
                    if (separated &&
                        state.exitFrames >= config.contactExitFrames) {
                        ContactState.SEPARATING
                    } else if (stableCandidate) {
                        ContactState.STABLE_CONTACT
                    } else {
                        ContactState.CONTACT_CANDIDATE
                    }

                ContactState.SEPARATING ->
                    if (candidate &&
                        state.enterFrames >= config.contactEnterFrames) {
                        ContactState.CONTACT_CANDIDATE
                    } else if (state.exitFrames >= config.contactExitFrames) {
                        ContactState.LOST
                    } else {
                        ContactState.SEPARATING
                    }

                ContactState.LOST ->
                    if (candidate) {
                        ContactState.APPROACHING
                    } else {
                        ContactState.LOST
                    }
            }

        val residual = distance
        state.residualHistory.addLast(residual)
        while (state.residualHistory.size > config.historyPerPair) {
            state.residualHistory.removeFirst()
        }

        val residualRobustWeight = robustWeight(
            median(state.residualHistory.toList()),
        )

        val fusedConfidence = (
            state.confidenceEma * 0.52 +
                input.hand.confidence * 0.16 +
                input.trackedObject.confidence * 0.12 +
                sensorScore * 0.10 +
                residualRobustWeight * 0.10
            ).coerceIn(0.0, 1.0)

        val reasoningMode =
            when {
                state.state == ContactState.STABLE_CONTACT &&
                    input.inertialConfidence >= config.minimumInertialConfidence &&
                    input.temporalConsistency >=
                    config.minimumTemporalConsistency ->
                    ReasoningMode.FUSED_CONTACT

                input.inertialConfidence >= config.minimumInertialConfidence ->
                    ReasoningMode.INERTIAL

                input.hand.confidence >=
                    config.minimumContactConfidence ->
                    ReasoningMode.KINEMATIC

                else ->
                    ReasoningMode.HOLD
            }

        val warnings = buildList {
            if (distance > config.maximumDistanceMeters) {
                add("HAND_OBJECT_DISTANCE_OUTSIDE_CONTACT_ENVELOPE")
            }
            if (relativeSpeed > config.maximumRelativeSpeedMetersPerSecond) {
                add("HAND_OBJECT_RELATIVE_SPEED_TOO_HIGH_FOR_RIGID_CONTACT")
            }
            if (input.temporalConsistency <
                config.minimumTemporalConsistency) {
                add("CONTACT_TEMPORAL_CONSISTENCY_LOW")
            }
            if (input.inertialConfidence <
                config.minimumInertialConfidence) {
                add("OBJECT_INERTIAL_EVIDENCE_LOW")
            }
            if (input.detectorAgreement <
                config.minimumContactConfidence) {
                add("CONTACT_DETECTOR_AGREEMENT_LOW")
            }
            if (state.state == ContactState.CONTACT_CANDIDATE) {
                add("CONTACT_CANDIDATE_REQUIRES_PERSISTENCE")
            }
            if (reasoningMode == ReasoningMode.HOLD) {
                add("CONTACT_REASONING_HELD")
            }
        }.distinct()

        val recommendations = buildList {
            if (state.state == ContactState.CONTACT_CANDIDATE) {
                add("Maintain tracking for additional frames before imposing a rigid hand-object constraint.")
            }
            if (input.inertialConfidence < config.minimumInertialConfidence) {
                add("Increase object inertial confidence or fall back to unconstrained kinematic tracking.")
            }
            if (relativeSpeed > config.stableRelativeSpeedMetersPerSecond) {
                add("Do not lock object and hand motion until relative velocity stabilizes.")
            }
            if (distance > config.stableDistanceMeters) {
                add("Increase spatial evidence around the hand-object contact region.")
            }
        }.distinct()

        return PairResult(
            handId = input.hand.handId,
            objectId = input.trackedObject.objectId,
            timestampNanos = timestamp,
            state = state.state,
            reasoningMode = reasoningMode,
            contactConfidence = state.confidenceEma,
            distanceMeters = distance,
            relativeSpeedMetersPerSecond = relativeSpeed,
            relativePositionResidualMeters = residual,
            fusedConfidence = fusedConfidence,
            enterFrames = state.enterFrames,
            stableFrames = state.stableFrames,
            exitFrames = state.exitFrames,
            warnings = warnings,
            recommendations = recommendations,
        )
    }

    @Synchronized
    fun reset() {
        pairStates.clear()
    }

    @Synchronized
    fun activePairCount(): Int =
        pairStates.count {
            it.value.state == ContactState.STABLE_CONTACT ||
                it.value.state == ContactState.CONTACT_CANDIDATE
        }

    private fun inverse(
        value: Double,
        preferred: Double,
        hard: Double,
    ): Double {
        if (!value.isFinite()) return 0.0
        if (value <= preferred) return 1.0
        if (value >= hard) return 0.0
        return (
            1.0 -
                (value - preferred) /
                    (hard - preferred).coerceAtLeast(1e-9)
            ).coerceIn(0.0, 1.0)
    }

    private fun robustWeight(residual: Double): Double {
        if (!residual.isFinite()) return 0.0
        val u = residual / config.residualScaleMeters
        return (
            if (u <= 1.0) {
                val t = 1.0 - u * u
                t * t
            } else {
                1.0 /
                    (1.0 + u * u)
            }
            ).coerceIn(0.04, 1.0)
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }
}
