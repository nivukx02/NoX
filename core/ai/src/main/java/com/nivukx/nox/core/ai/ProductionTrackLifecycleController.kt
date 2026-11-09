package com.nivukx.nox.core.ai

import kotlin.math.exp

/**
 * Production lifecycle controller layered on top of SubjectTrackManager.
 *
 * SubjectTrackManager performs identity-aware geometric association. This controller owns the
 * temporal contract around those tracks: tentative acquisition, confirmation, short occlusion,
 * evidence-backed recovery, and explicit loss. It never invents a new identity and never revives
 * a stale track merely because a detector score spikes.
 */
class ProductionTrackLifecycleController(
    private val config: Config = Config(),
) {

    enum class State {
        NEW,
        TENTATIVE,
        CONFIRMED,
        OCCLUDED,
        RECOVERING,
        LOST,
        RETIRED,
    }

    data class Config(
        val minimumConfirmationFrames: Int = 3,
        val maximumOcclusionNanos: Long = 700_000_000L,
        val recoveryWindowNanos: Long = 1_200_000_000L,
        val retirementNanos: Long = 2_500_000_000L,
        val minimumConfidence: Double = 0.35,
        val recoveryConfidenceFloor: Double = 0.30,
        val confirmedConfidenceFloor: Double = 0.50,
        val temporalWeight: Double = 0.28,
        val confidenceWeight: Double = 0.36,
        val missWeight: Double = 0.16,
        val velocityStabilityWeight: Double = 0.08,
        val stateMemoryWeight: Double = 0.12,
    ) {
        init {
            require(minimumConfirmationFrames >= 1)
            require(maximumOcclusionNanos > 0L)
            require(recoveryWindowNanos >= maximumOcclusionNanos)
            require(retirementNanos >= recoveryWindowNanos)
            require(minimumConfidence in 0.0..1.0)
            require(recoveryConfidenceFloor in 0.0..1.0)
            require(confirmedConfidenceFloor in 0.0..1.0)
            val weights = listOf(
                temporalWeight,
                confidenceWeight,
                missWeight,
                velocityStabilityWeight,
                stateMemoryWeight,
            )
            require(weights.all { it >= 0.0 && it <= 1.0 })
            require(weights.sum() > 0.0)
        }
    }

    data class TrackEvidence(
        val trackId: Long,
        val timestampNanos: Long,
        val confidence: Double,
        val missedFrames: Int,
        val ageFrames: Long,
        val temporalStability: Double,
        val velocityMagnitude: Double,
    ) {
        init {
            require(trackId >= 0L)
            require(timestampNanos >= 0L)
            require(confidence.isFinite())
            require(missedFrames >= 0)
            require(ageFrames >= 0L)
            require(temporalStability.isFinite())
            require(velocityMagnitude.isFinite())
        }
    }

    data class Result(
        val states: Map<Long, State>,
        val continuityScores: Map<Long, Double>,
        val predictedTrackIds: Set<Long>,
        val recoveringTrackIds: Set<Long>,
        val confirmedTrackIds: Set<Long>,
        val dominantTrackId: Long?,
        val dominantState: State,
        val continuityScore: Double,
        val warnings: List<String>,
    )

    private data class Memory(
        var state: State,
        var lastTimestampNanos: Long,
        var lastSeenTimestampNanos: Long,
        var consecutiveStrongFrames: Int,
        var recoveryEvidenceFrames: Int,
        var lastConfidence: Double,
        var lastTemporalStability: Double,
        var lastVelocityMagnitude: Double,
        var missingSinceNanos: Long?,
    )

    private val memories = LinkedHashMap<Long, Memory>()

    @Synchronized
    fun update(
        timestampNanos: Long,
        tracks: List<TrackEvidence>,
        evidenceFresh: Boolean = true,
    ): Result {
        require(timestampNanos >= 0L)

        val staleEvidenceIds =
            tracks
                .asSequence()
                .filter { it.timestampNanos <= timestampNanos }
                .groupBy { it.trackId }
                .filter { (trackId, evidenceList) ->
                    val newestEvidence =
                        evidenceList.maxOfOrNull {
                            it.timestampNanos
                        } ?: Long.MIN_VALUE
                    val memory = memories[trackId]
                    memory != null &&
                        newestEvidence <=
                            memory.lastTimestampNanos
                }
                .keys

        val uniqueTracks =
            tracks
                .filter { it.timestampNanos <= timestampNanos }
                .filter { it.trackId !in staleEvidenceIds }
                .distinctBy { it.trackId }

        val seenIds = uniqueTracks.map { it.trackId }.toHashSet()
        val protectedFromMissingIds = seenIds.toHashSet()

        if (evidenceFresh) {
            uniqueTracks.forEach { evidence ->
                updateMemory(
                    evidence = evidence,
                    timestampNanos = timestampNanos,
                )
            }
        } else {
            memories.values.forEach { memory ->
                memory.lastTimestampNanos =
                    timestampNanos
            }
        }

        if (evidenceFresh) {
            memories.entries
                .filterNot { protectedFromMissingIds.contains(it.key) }
                .forEach { (id, memory) ->
                val age =
                    timestampNanos - memory.lastSeenTimestampNanos

                if (memory.missingSinceNanos == null) {
                    memory.missingSinceNanos =
                        timestampNanos
                }

                if (
                    age >
                    config.retirementNanos
                ) {
                    memory.state = State.RETIRED
                } else if (
                    age >
                    config.maximumOcclusionNanos
                ) {
                    memory.state = State.LOST
                } else {
                    memory.state =
                        State.OCCLUDED
                }

                memory.lastTimestampNanos =
                    timestampNanos

                memories[id] = memory
            }
        }

        memories.entries
            .removeIf {
                it.value.state == State.RETIRED
            }

        val states =
            memories.mapValues { it.value.state }

        val continuityScores =
            memories.mapValues { (_, memory) ->
                continuityScore(memory)
            }

        val predicted =
            states.filterValues {
                it == State.OCCLUDED
            }.keys

        val recovering =
            states.filterValues {
                it == State.RECOVERING
            }.keys

        val confirmed =
            states.filterValues {
                it == State.CONFIRMED ||
                    it == State.RECOVERING ||
                    it == State.OCCLUDED
            }.keys

        val dominant =
            memories
                .entries
                .maxByOrNull {
                    continuityScore(it.value) *
                        when (it.value.state) {
                            State.CONFIRMED -> 1.08
                            State.RECOVERING -> 1.04
                            State.OCCLUDED -> 0.88
                            State.TENTATIVE -> 0.72
                            State.NEW -> 0.50
                            State.LOST -> 0.35
                            State.RETIRED -> 0.0
                        }
                }

        val warnings =
            buildWarnings(
                states = states,
                continuityScores = continuityScores,
            ).toMutableList()

        if (!evidenceFresh && memories.isNotEmpty()) {
            warnings += "tracking-evidence-cached-for-scheduler-budget"
        }
        if (staleEvidenceIds.isNotEmpty()) {
            warnings += "stale-track-evidence-rejected"
        }

        return Result(
            states = states,
            continuityScores = continuityScores,
            predictedTrackIds = predicted,
            recoveringTrackIds = recovering,
            confirmedTrackIds = confirmed,
            dominantTrackId = dominant?.key,
            dominantState = dominant?.value?.state ?: State.NEW,
            continuityScore =
                dominant?.value
                    ?.let(::continuityScore)
                    ?.coerceIn(0.0, 1.0)
                    ?: 0.0,
            warnings = warnings,
        )
    }

    @Synchronized
    fun reset() {
        memories.clear()
    }

    @Synchronized
    fun snapshot(): Result =
        buildResultForCurrentState()

    private fun updateMemory(
        evidence: TrackEvidence,
        timestampNanos: Long,
    ) {
        val memory =
            memories[evidence.trackId]

        if (memory == null) {
            memories[evidence.trackId] =
                Memory(
                    state = State.TENTATIVE,
                    lastTimestampNanos = timestampNanos,
                    lastSeenTimestampNanos = timestampNanos,
                    consecutiveStrongFrames =
                        if (
                            evidence.confidence >=
                            config.confirmedConfidenceFloor
                        ) {
                            1
                        } else {
                            0
                        },
                    recoveryEvidenceFrames = 0,
                    lastConfidence =
                        evidence.confidence.coerceIn(0.0, 1.0),
                    lastTemporalStability =
                        evidence.temporalStability.coerceIn(0.0, 1.0),
                    lastVelocityMagnitude =
                        evidence.velocityMagnitude.coerceAtLeast(0.0),
                    missingSinceNanos = null,
                )
            return
        }

        val previouslyOccluded =
            memory.missingSinceNanos != null ||
                memory.state == State.OCCLUDED ||
                memory.state == State.LOST

        val confidence =
            evidence.confidence.coerceIn(0.0, 1.0)

        val dtNanos =
            (
                timestampNanos -
                    memory.lastSeenTimestampNanos
            ).coerceAtLeast(0L)

        val stable =
            evidence.temporalStability
                .coerceIn(0.0, 1.0)

        val strong =
            confidence >=
                config.confirmedConfidenceFloor &&
                stable >= 0.40

        if (strong) {
            memory.consecutiveStrongFrames += 1
        } else {
            memory.consecutiveStrongFrames =
                (
                    memory.consecutiveStrongFrames -
                        1
                    ).coerceAtLeast(0)
        }

        if (previouslyOccluded) {
            memory.recoveryEvidenceFrames =
                if (
                    confidence >=
                        config.recoveryConfidenceFloor &&
                    stable >= 0.32
                ) {
                    memory.recoveryEvidenceFrames + 1
                } else {
                    0
                }

            memory.state =
                when {
                    memory.recoveryEvidenceFrames >=
                        2 ->
                        if (
                            dtNanos <=
                                config.recoveryWindowNanos
                        ) {
                            State.RECOVERING
                        } else {
                            State.TENTATIVE
                        }

                    dtNanos <= config.maximumOcclusionNanos ->
                        State.OCCLUDED

                    dtNanos <= config.recoveryWindowNanos ->
                        State.LOST

                    else ->
                        State.LOST
                }
        } else {
            memory.state =
                when {
                    evidence.missedFrames >= 4 ->
                        State.LOST

                    evidence.missedFrames > 0 &&
                        dtNanos <=
                        config.maximumOcclusionNanos ->
                        State.OCCLUDED

                    memory.consecutiveStrongFrames >=
                        config.minimumConfirmationFrames ->
                        State.CONFIRMED

                    confidence >=
                        config.minimumConfidence ->
                        State.TENTATIVE

                    else ->
                        State.NEW
                }
        }

        memory.lastTimestampNanos =
            timestampNanos
        memory.lastSeenTimestampNanos =
            timestampNanos
        memory.lastConfidence =
            confidence
        memory.lastTemporalStability =
            stable
        memory.lastVelocityMagnitude =
            evidence.velocityMagnitude.coerceAtLeast(0.0)
        memory.missingSinceNanos = null
    }

    private fun continuityScore(
        memory: Memory,
    ): Double {
        val missPenalty =
            when (memory.state) {
                State.OCCLUDED ->
                    0.42
                State.RECOVERING ->
                    0.18
                State.LOST ->
                    0.78
                State.TENTATIVE ->
                    0.16
                State.NEW ->
                    0.35
                State.CONFIRMED ->
                    0.0
                State.RETIRED ->
                    1.0
            }

        val velocityStability =
            exp(
                -memory.lastVelocityMagnitude.coerceAtLeast(0.0) *
                    1.35,
            ).coerceIn(0.0, 1.0)

        val temporalStability =
            memory.lastTemporalStability
                .coerceIn(0.0, 1.0)

        return (
            memory.lastConfidence *
                config.confidenceWeight +
                (
                    1.0 -
                        missPenalty
                    ).coerceIn(0.0, 1.0) *
                config.stateMemoryWeight +
                velocityStability *
                config.velocityStabilityWeight +
                (
                    1.0 -
                        missPenalty * 0.35
                    ).coerceIn(0.0, 1.0) *
                config.missWeight +
                temporalStability *
                config.temporalWeight
            ).coerceIn(0.0, 1.0)
    }

    private fun buildWarnings(
        states: Map<Long, State>,
        continuityScores: Map<Long, Double>,
    ): List<String> {
        val warnings =
            ArrayList<String>()

        if (states.values.any { it == State.OCCLUDED }) {
            warnings += "track-occlusion-prediction-active"
        }

        if (states.values.any { it == State.RECOVERING }) {
            warnings += "track-reacquisition-in-progress"
        }

        if (states.values.any { it == State.LOST }) {
            warnings += "track-loss-detected"
        }

        if (
            continuityScores.values.any {
                it < 0.45
            }
        ) {
            warnings += "track-continuity-below-production-floor"
        }

        return warnings.distinct()
    }

    private fun buildResultForCurrentState(): Result {
        val states =
            memories.mapValues { it.value.state }

        val continuity =
            memories.mapValues { continuityScore(it.value) }

        val dominant =
            memories.entries.maxByOrNull {
                continuity[it.key] ?: 0.0
            }

        return Result(
            states = states,
            continuityScores = continuity,
            predictedTrackIds =
                states.filterValues {
                    it == State.OCCLUDED
                }.keys,
            recoveringTrackIds =
                states.filterValues {
                    it == State.RECOVERING
                }.keys,
            confirmedTrackIds =
                states.filterValues {
                    it == State.CONFIRMED ||
                        it == State.RECOVERING ||
                        it == State.OCCLUDED
                }.keys,
            dominantTrackId = dominant?.key,
            dominantState =
                dominant?.value?.state ?: State.NEW,
            continuityScore =
                dominant?.key?.let {
                    continuity[it] ?: 0.0
                } ?: 0.0,
            warnings = buildWarnings(
                states = states,
                continuityScores = continuity,
            ),
        )
    }
}
