package com.nivukx.nox.core.reconstruction

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Contact-aware temporal reconstruction evidence for hands interacting with scene objects.
 *
 * The engine deliberately separates visual proximity from physical contact. A pair can be close
 * without touching, can enter a shallow uncertain overlap because of depth noise, and can remain
 * in contact while one view is temporarily occluded. The state machine therefore tracks:
 * distance, approach speed, normal alignment, penetration, contact age, impact energy, hysteresis,
 * and evidence continuity.
 *
 * This is a bounded evidence engine. It does not hallucinate hidden geometry and it never converts
 * a low-confidence intersection into a production-grade collision claim.
 */
class ContactAwareReconstructionEngine(
    private val config: Config = Config(),
) : AutoCloseable {

    enum class State {
        FREE,
        APPROACHING,
        CONTACT,
        SEPARATING,
        OCCLUDED,
        STALE,
    }

    data class Config(
        val maximumPairs: Int = 512,
        val maximumSamplesPerPair: Int = 96,
        val distanceSigmaMillimeters: Double = 22.0,
        val contactDistanceMillimeters: Double = 18.0,
        val releaseDistanceMillimeters: Double = 32.0,
        val maximumPenetrationMillimeters: Double = 15.0,
        val contactEnterScore: Double = 0.64,
        val contactReleaseScore: Double = 0.34,
        val impactVelocityMetersPerSecond: Double = 0.55,
        val maximumDeltaSeconds: Double = 0.25,
        val stateHalfLifeSeconds: Double = 0.22,
        val minimumConfidenceForClaim: Double = 0.72,
    ) {
        init {
            require(maximumPairs > 0)
            require(maximumSamplesPerPair >= 4)
            require(distanceSigmaMillimeters > 0.0)
            require(contactDistanceMillimeters > 0.0)
            require(releaseDistanceMillimeters >= contactDistanceMillimeters)
            require(maximumPenetrationMillimeters >= 0.0)
            require(contactEnterScore in 0.0..1.0)
            require(contactReleaseScore in 0.0..contactEnterScore)
            require(impactVelocityMetersPerSecond >= 0.0)
            require(maximumDeltaSeconds > 0.0)
            require(stateHalfLifeSeconds > 0.0)
            require(minimumConfidenceForClaim in 0.0..1.0)
        }
    }

    data class InteractionObservation(
        val handId: Long,
        val objectId: Long,
        val timestampNanos: Long,
        val distanceMillimeters: Double,
        val relativeSpeedMetersPerSecond: Double,
        val normalAlignment: Double,
        val penetrationMillimeters: Double = 0.0,
        val handConfidence: Double = 1.0,
        val objectConfidence: Double = 1.0,
        val visible: Boolean = true,
    ) {
        init {
            require(handId >= 0L)
            require(objectId >= 0L)
            require(timestampNanos >= 0L)
            listOf(
                distanceMillimeters,
                relativeSpeedMetersPerSecond,
                penetrationMillimeters,
                handConfidence,
                objectConfidence,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
            require(normalAlignment.isFinite() && normalAlignment in -1.0..1.0)
            require(handConfidence <= 1.0)
            require(objectConfidence <= 1.0)
        }
    }

    data class Result(
        val handId: Long,
        val objectId: Long,
        val timestampNanos: Long,
        val state: State,
        val contactScore: Double,
        val interactionConfidence: Double,
        val contactAgeSeconds: Double,
        val separationAgeSeconds: Double,
        val peakApproachVelocityMetersPerSecond: Double,
        val peakPenetrationMillimeters: Double,
        val impactScore: Double,
        val temporalConsistency: Double,
        val collisionConfidence: Double,
        val staleEvidenceRejected: Int,
        val hardPenetrationViolation: Boolean,
        val warnings: List<String>,
    ) {
        val isTrustedContact: Boolean
            get() =
                state == State.CONTACT &&
                    contactScore >= 0.64 &&
                    interactionConfidence >= 0.72 &&
                    !hardPenetrationViolation

        val safeForFusion: Boolean
            get() =
                state != State.STALE &&
                    !hardPenetrationViolation &&
                    interactionConfidence >= 0.45
    }

    data class Aggregate(
        val timestampNanos: Long,
        val activePairCount: Int,
        val contactPairCount: Int,
        val approachingPairCount: Int,
        val occludedPairCount: Int,
        val meanContactScore: Double,
        val meanInteractionConfidence: Double,
        val stableContactRatio: Double,
        val maximumPenetrationMillimeters: Double,
        val maximumImpactScore: Double,
        val temporalConsistency: Double,
        val collisionConfidence: Double,
        val staleEvidenceRejected: Int,
        val trustedContactPairs: List<Pair<Long, Long>>,
        val warnings: List<String>,
    ) {
        val productionContactConfidence: Double
            get() = min(
                min(meanInteractionConfidence, temporalConsistency),
                collisionConfidence,
            )
    }

    private data class Sample(
        val timestampNanos: Long,
        val distanceMillimeters: Double,
        val approachVelocityMetersPerSecond: Double,
        val normalAlignment: Double,
        val penetrationMillimeters: Double,
        val confidence: Double,
    )

    private data class PairState(
        val handId: Long,
        val objectId: Long,
        val samples: ArrayDeque<Sample>,
        var lastTimestampNanos: Long = Long.MIN_VALUE,
        var state: State = State.FREE,
        var contactAgeSeconds: Double = 0.0,
        var separationAgeSeconds: Double = 0.0,
        var peakApproachVelocityMetersPerSecond: Double = 0.0,
        var peakPenetrationMillimeters: Double = 0.0,
        var impactScore: Double = 0.0,
        var temporalConsistency: Double = 0.0,
        var contactScore: Double = 0.0,
        var interactionConfidence: Double = 0.0,
        var collisionConfidence: Double = 0.0,
        var staleEvidenceRejected: Int = 0,
        var missingSeconds: Double = 0.0,
    )

    private val pairStates = LinkedHashMap<PairKey, PairState>()
    private var lastTimestampNanos: Long = Long.MIN_VALUE

    private data class PairKey(
        val handId: Long,
        val objectId: Long,
    )

    @Synchronized
    fun evaluate(
        observations: List<InteractionObservation>,
        timestampNanos: Long,
    ): Aggregate {
        require(timestampNanos >= 0L)

        if (
            lastTimestampNanos != Long.MIN_VALUE &&
            timestampNanos <= lastTimestampNanos
        ) {
            return aggregate(
                timestampNanos = lastTimestampNanos,
                warnings = listOf("stale-frame-rejected"),
            )
        }

        val seen = HashSet<PairKey>()
        val warnings = mutableListOf<String>()
        var staleRejected = 0

        observations
            .asSequence()
            .filter { it.visible }
            .take(config.maximumPairs)
            .forEach { observation ->
                val key = PairKey(observation.handId, observation.objectId)
                seen += key
                val state = pairStates.getOrPut(key) {
                    if (pairStates.size >= config.maximumPairs) {
                        val victim = pairStates.minByOrNull {
                            it.value.interactionConfidence
                        }?.key
                        if (victim != null) pairStates.remove(victim)
                    }
                    PairState(
                        handId = observation.handId,
                        objectId = observation.objectId,
                        samples = ArrayDeque(config.maximumSamplesPerPair),
                    )
                }

                val firstSample =
                    state.lastTimestampNanos == Long.MIN_VALUE
                val deltaSeconds =
                    if (firstSample) {
                        0.0
                    } else {
                        (timestampNanos - state.lastTimestampNanos) /
                            1_000_000_000.0
                    }

                if (
                    !firstSample &&
                    (
                        !deltaSeconds.isFinite() ||
                            deltaSeconds <= 0.0 ||
                            deltaSeconds > config.maximumDeltaSeconds
                        )
                ) {
                    state.staleEvidenceRejected += 1
                    staleRejected += 1
                    warnings += "stale-or-invalid-interaction-sample"
                    return@forEach
                }

                updateState(
                    state = state,
                    observation = observation,
                    deltaSeconds = deltaSeconds,
                )
                state.lastTimestampNanos = timestampNanos
                state.missingSeconds = 0.0
            }

        pairStates.forEach { (_, state) ->
            if (PairKey(state.handId, state.objectId) !in seen) {
                state.missingSeconds += when {
                    lastTimestampNanos == Long.MIN_VALUE -> 0.0
                    else ->
                        (timestampNanos - kotlin.math.max(
                            lastTimestampNanos,
                            state.lastTimestampNanos.takeIf {
                                it != Long.MIN_VALUE
                            } ?: timestampNanos,
                        )) / 1_000_000_000.0
                }.coerceIn(0.0, config.maximumDeltaSeconds)

                decayMissingState(state)
            }
        }

        lastTimestampNanos = timestampNanos
        staleRejected += pairStates.values.sumOf { it.staleEvidenceRejected }

        return aggregate(
            timestampNanos = timestampNanos,
            warnings = warnings.distinct(),
        )
    }

    @Synchronized
    fun result(
        handId: Long,
        objectId: Long,
    ): Result? =
        pairStates[PairKey(handId, objectId)]?.let(::toResult)

    @Synchronized
    fun allResults(): List<Result> =
        pairStates.values.map(::toResult)

    @Synchronized
    fun reset(
        handId: Long? = null,
        objectId: Long? = null,
    ) {
        if (handId == null && objectId == null) {
            pairStates.clear()
            lastTimestampNanos = Long.MIN_VALUE
            return
        }

        val victims = pairStates.keys.filter {
            (handId == null || it.handId == handId) &&
                (objectId == null || it.objectId == objectId)
        }
        victims.forEach(pairStates::remove)
    }

    @Synchronized
    override fun close() {
        pairStates.clear()
        lastTimestampNanos = Long.MIN_VALUE
    }

    private fun updateState(
        state: PairState,
        observation: InteractionObservation,
        deltaSeconds: Double,
    ) {
        val confidence = (
            observation.handConfidence *
                observation.objectConfidence
            ).coerceIn(0.0, 1.0)

        val distanceScore = exp(
            -0.5 *
                (observation.distanceMillimeters /
                    config.distanceSigmaMillimeters).let { it * it },
        ).coerceIn(0.0, 1.0)

        val normalScore =
            ((observation.normalAlignment + 1.0) / 2.0)
                .coerceIn(0.0, 1.0)

        val penetrationScore =
            if (observation.penetrationMillimeters <=
                config.maximumPenetrationMillimeters
            ) {
                1.0 -
                    observation.penetrationMillimeters /
                    config.maximumPenetrationMillimeters.coerceAtLeast(0.001)
            } else {
                0.0
            }

        val speedScore = (
            observation.relativeSpeedMetersPerSecond /
                config.impactVelocityMetersPerSecond.coerceAtLeast(0.001)
            ).coerceIn(0.0, 1.0)

        val rawContactScore = (
            distanceScore * 0.42 +
                normalScore * 0.18 +
                penetrationScore * 0.12 +
                (1.0 - speedScore) * 0.08 +
                confidence * 0.20
            ).coerceIn(0.0, 1.0)

        val approach = max(
            0.0,
            observation.relativeSpeedMetersPerSecond,
        )

        state.contactScore =
            if (state.contactScore <= 0.0) {
                rawContactScore
            } else {
                state.contactScore * 0.72 + rawContactScore * 0.28
            }

        state.interactionConfidence =
            if (state.interactionConfidence <= 0.0) {
                confidence
            } else {
                state.interactionConfidence * 0.78 +
                    confidence * 0.22
            }

        state.peakApproachVelocityMetersPerSecond = max(
            state.peakApproachVelocityMetersPerSecond,
            approach,
        )
        state.peakPenetrationMillimeters = max(
            state.peakPenetrationMillimeters,
            observation.penetrationMillimeters,
        )

        val impact = (
            speedScore *
                distanceScore *
                confidence
            ).coerceIn(0.0, 1.0)
        state.impactScore =
            max(
                state.impactScore * exp(
                    -deltaSeconds /
                        config.stateHalfLifeSeconds,
                ),
                impact,
            )

        val continuityEvidence =
            (1.0 - min(
                1.0,
                deltaSeconds / config.maximumDeltaSeconds,
            ) * 0.35) *
                confidence
        state.temporalConsistency =
            if (state.temporalConsistency <= 0.0) {
                continuityEvidence
            } else {
                state.temporalConsistency * 0.82 +
                    continuityEvidence * 0.18
            }

        state.collisionConfidence =
            (
                state.contactScore * 0.40 +
                    penetrationScore * 0.16 +
                    normalScore * 0.16 +
                    state.temporalConsistency * 0.18 +
                    confidence * 0.10
                ).coerceIn(0.0, 1.0)

        val penetrationViolation =
            observation.penetrationMillimeters >
                config.maximumPenetrationMillimeters

        when (state.state) {
            State.FREE -> {
                state.state =
                    if (state.contactScore >= 0.42) {
                        State.APPROACHING
                    } else {
                        State.FREE
                    }
                state.contactAgeSeconds = 0.0
                state.separationAgeSeconds = 0.0
            }

            State.APPROACHING,
            State.SEPARATING,
            State.OCCLUDED,
            -> {
                if (state.contactScore >= config.contactEnterScore &&
                    !penetrationViolation
                ) {
                    state.state = State.CONTACT
                    state.contactAgeSeconds += deltaSeconds
                    state.separationAgeSeconds = 0.0
                } else if (state.contactScore < config.contactReleaseScore) {
                    state.state = State.SEPARATING
                    state.separationAgeSeconds += deltaSeconds
                    state.contactAgeSeconds = 0.0
                } else {
                    state.state = State.APPROACHING
                }
            }

            State.CONTACT -> {
                if (penetrationViolation) {
                    state.state = State.OCCLUDED
                } else if (
                    state.contactScore <= config.contactReleaseScore &&
                    observation.distanceMillimeters >
                    config.releaseDistanceMillimeters
                ) {
                    state.state = State.SEPARATING
                    state.separationAgeSeconds += deltaSeconds
                    state.contactAgeSeconds = 0.0
                } else {
                    state.contactAgeSeconds += deltaSeconds
                    state.separationAgeSeconds = 0.0
                }
            }

            State.STALE -> {
                if (state.contactScore >= config.contactEnterScore) {
                    state.state = State.CONTACT
                }
            }
        }
    }

    private fun decayMissingState(state: PairState) {
        val decay = exp(
            -state.missingSeconds /
                config.stateHalfLifeSeconds,
        ).coerceIn(0.0, 1.0)

        state.contactScore *= decay
        state.interactionConfidence *= decay
        state.temporalConsistency *= decay
        state.collisionConfidence *= decay
        state.impactScore *= decay

        when {
            state.missingSeconds > config.maximumDeltaSeconds * 2.0 ->
                state.state = State.STALE

            state.state == State.CONTACT ->
                state.state = State.OCCLUDED

            state.state == State.APPROACHING ->
                state.state = State.OCCLUDED
        }
    }

    private fun aggregate(
        timestampNanos: Long,
        warnings: List<String>,
    ): Aggregate {
        val active = pairStates.values.filter {
            it.state != State.STALE ||
                it.interactionConfidence > 0.05
        }

        val contact = active.count { it.state == State.CONTACT }
        val approaching =
            active.count { it.state == State.APPROACHING }
        val occluded =
            active.count { it.state == State.OCCLUDED }

        val meanContact =
            active.map { it.contactScore }.averageOrZero()
        val meanConfidence =
            active.map { it.interactionConfidence }.averageOrZero()
        val stableRatio =
            if (active.isEmpty()) {
                0.0
            } else {
                contact.toDouble() / active.size
            }

        val maxPenetration =
            active.maxOfOrNull { it.peakPenetrationMillimeters }
                ?: 0.0
        val maxImpact =
            active.maxOfOrNull { it.impactScore } ?: 0.0
        val temporal =
            active.map { it.temporalConsistency }.averageOrZero()
        val collision =
            active.map { it.collisionConfidence }.averageOrZero()

        val trusted =
            active.asSequence()
                .map(::toResult)
                .filter { it.isTrustedContact }
                .map { it.handId to it.objectId }
                .take(config.maximumPairs)
                .toList()

        val aggregateWarnings = buildList {
            addAll(warnings)
            if (active.isEmpty()) add("no-active-contact-evidence")
            if (maxPenetration > config.maximumPenetrationMillimeters) {
                add("penetration-limit-exceeded")
            }
            if (occluded > 0) add("contact-occlusion-active")
            if (trusted.isEmpty() && contact > 0) {
                add("contact-present-but-untrusted")
            }
        }.distinct()

        return Aggregate(
            timestampNanos = timestampNanos,
            activePairCount = active.size,
            contactPairCount = contact,
            approachingPairCount = approaching,
            occludedPairCount = occluded,
            meanContactScore = meanContact,
            meanInteractionConfidence = meanConfidence,
            stableContactRatio = stableRatio,
            maximumPenetrationMillimeters = maxPenetration,
            maximumImpactScore = maxImpact,
            temporalConsistency = temporal,
            collisionConfidence = collision,
            staleEvidenceRejected =
                pairStates.values.sumOf { it.staleEvidenceRejected },
            trustedContactPairs = trusted,
            warnings = aggregateWarnings,
        )
    }

    private fun toResult(state: PairState): Result =
        Result(
            handId = state.handId,
            objectId = state.objectId,
            timestampNanos = state.lastTimestampNanos.coerceAtLeast(0L),
            state = state.state,
            contactScore = state.contactScore.coerceIn(0.0, 1.0),
            interactionConfidence =
                state.interactionConfidence.coerceIn(0.0, 1.0),
            contactAgeSeconds =
                state.contactAgeSeconds.coerceAtLeast(0.0),
            separationAgeSeconds =
                state.separationAgeSeconds.coerceAtLeast(0.0),
            peakApproachVelocityMetersPerSecond =
                state.peakApproachVelocityMetersPerSecond.coerceAtLeast(0.0),
            peakPenetrationMillimeters =
                state.peakPenetrationMillimeters.coerceAtLeast(0.0),
            impactScore = state.impactScore.coerceIn(0.0, 1.0),
            temporalConsistency =
                state.temporalConsistency.coerceIn(0.0, 1.0),
            collisionConfidence =
                state.collisionConfidence.coerceIn(0.0, 1.0),
            staleEvidenceRejected = state.staleEvidenceRejected,
            hardPenetrationViolation =
                state.peakPenetrationMillimeters >
                    config.maximumPenetrationMillimeters,
            warnings = buildList {
                if (state.state == State.OCCLUDED) {
                    add("interaction-occluded")
                }
                if (state.state == State.STALE) {
                    add("interaction-stale")
                }
                if (state.staleEvidenceRejected > 0) {
                    add("stale-interaction-evidence-rejected")
                }
                if (
                    state.interactionConfidence <
                    config.minimumConfidenceForClaim
                ) {
                    add("interaction-confidence-below-claim-threshold")
                }
            },
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()
}
