package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import kotlin.math.exp
import kotlin.math.max

/**
 * Cross-modal identity continuity engine.
 *
 * This stage consumes body, face, hands, and eyes as independent evidence streams.
 * Missing evidence is treated as missing, never synthesized. Association is solved globally
 * across a bounded set of active tracks, then stabilized with temporal hysteresis.
 */
class CrossModalIdentityFusionEngine(
    private var config: Config = Config(),
) {
    enum class Action { SEARCH, HOLD, ASSIGN, RECOVER, SWITCH, REJECT }

    data class Config(
        // Bounded crowd envelope aligned with the 50-actor target and 64-actor hard ceiling.
        val maximumTracks: Int = 64,
        val maximumObservations: Int = 128,
        val bodyWeight: Double = 0.34,
        val faceWeight: Double = 0.24,
        val handWeight: Double = 0.18,
        val eyeWeight: Double = 0.10,
        val motionWeight: Double = 0.10,
        val missingModalityPenalty: Double = 0.14,
        val staleModalityPenalty: Double = 0.18,
        val predictionPenalty: Double = 0.11,
        val occlusionPenalty: Double = 0.12,
        val switchCost: Double = 0.24,
        val holdCost: Double = 0.34,
        val rejectCost: Double = 0.92,
        val assignmentGate: Double = 0.74,
        val switchGate: Double = 0.56,
        val recoveryGate: Double = 0.66,
        val historyAlpha: Double = 0.22,
        val velocityAlpha: Double = 0.28,
        val maximumAgeSeconds: Double = 0.80,
        val maximumBodyResidualMeters: Double = 1.40,
        val maximumFaceResidualMeters: Double = 0.55,
        val maximumHandResidualMeters: Double = 0.85,
        val maximumEyeResidualMeters: Double = 0.40,
        val maximumVelocityMetersPerSecond: Double = 8.0,
        val hysteresisMargin: Double = 0.08,
    ) {
        init {
            require(maximumTracks in 1..64)
            require(maximumObservations in 1..128)
            require(maximumObservations >= maximumTracks)
            require(bodyWeight + faceWeight + handWeight + eyeWeight + motionWeight > 0.0)
            require(assignmentGate in 0.0..1.0)
            require(switchGate in 0.0..1.0)
            require(recoveryGate in 0.0..1.0)
            require(historyAlpha in 0.01..1.0)
            require(velocityAlpha in 0.01..1.0)
            require(maximumAgeSeconds > 0.0)
            require(maximumBodyResidualMeters > 0.0)
            require(maximumFaceResidualMeters > 0.0)
            require(maximumHandResidualMeters > 0.0)
            require(maximumEyeResidualMeters > 0.0)
            require(maximumVelocityMetersPerSecond > 0.0)
            require(hysteresisMargin >= 0.0)
        }
    }

    data class Observation(
        val observationId: Long,
        val timestampNanos: Long,
        val bodyRoot: Vec3? = null,
        val faceCenter: Vec3? = null,
        val leftHand: Vec3? = null,
        val rightHand: Vec3? = null,
        val leftEye: Vec3? = null,
        val rightEye: Vec3? = null,
        val scaleMeters: Double? = null,
        val confidence: Double = 1.0,
        val predicted: Boolean = false,
        val occluded: Boolean = false,
        val freshnessSeconds: Double = 0.0,
        val evidenceProfile:
            Map<CrossModalEvidenceArbitrationEngine.Modality,
                CrossModalEvidenceArbitrationEngine.ModalityEvidence> =
            emptyMap(),
    ) {
        init {
            require(observationId >= 0L)
            require(timestampNanos > 0L)
            require(scaleMeters == null || scaleMeters > 0.0)
            require(confidence in 0.0..1.0)
            require(freshnessSeconds >= 0.0)
        }

        fun modalityCount(): Int =
            listOf(bodyRoot, faceCenter, leftHand, rightHand, leftEye, rightEye).count { it != null }
    }

    data class TrackSeed(
        val trackId: Long,
        val timestampNanos: Long,
        val bodyRoot: Vec3? = null,
        val faceCenter: Vec3? = null,
        val leftHand: Vec3? = null,
        val rightHand: Vec3? = null,
        val leftEye: Vec3? = null,
        val rightEye: Vec3? = null,
        val scaleMeters: Double? = null,
        val confidence: Double = 0.5,
    )

    data class TrackAssignment(
        val trackId: Long,
        val observationId: Long?,
        val cost: Double,
        val confidence: Double,
        val spatialScore: Double,
        val motionScore: Double,
        val crossModalScore: Double,
        val freshnessScore: Double,
        val action: Action,
        val predicted: Boolean,
        val occluded: Boolean,
        val diagnostics: List<String>,
        val evidenceFusionScore: Double = 0.0,
        val selectedEvidenceModalityCount: Int = 0,
        val selectedEvidenceModalities: List<String> = emptyList(),
    )

    data class Result(
        val timestampNanos: Long,
        val assignments: List<TrackAssignment>,
        val continuityScore: Double,
        val swapRisk: Double,
        val unmatchedObservationIds: List<Long>,
        val orphanTrackIds: List<Long>,
        val action: Action,
        val reason: String,
        val warnings: List<String>,
        val recommendations: List<String>,
        val evidenceFusionScore: Double = 0.0,
        val selectedEvidenceModalityCount: Int = 0,
        val evidenceWarnings: List<String> = emptyList(),
    ) {
        val assignedTrackCount: Int
            get() = assignments.count { it.observationId != null }

        val recoveredTrackCount: Int
            get() = assignments.count { it.action == Action.RECOVER }

        val switchedTrackCount: Int
            get() = assignments.count { it.action == Action.SWITCH }
    }

    private data class TrackState(
        val trackId: Long,
        var timestampNanos: Long,
        var bodyRoot: Vec3?,
        var faceCenter: Vec3?,
        var leftHand: Vec3?,
        var rightHand: Vec3?,
        var leftEye: Vec3?,
        var rightEye: Vec3?,
        var scaleMeters: Double?,
        var confidence: Double,
        var velocity: Vec3 = ZERO,
        var ageSeconds: Double = 0.0,
        var missedFrames: Int = 0,
    )

    private data class PairMetrics(
        val cost: Double,
        val spatialScore: Double,
        val motionScore: Double,
        val crossModalScore: Double,
        val freshnessScore: Double,
        val evidenceFusionScore: Double,
        val selectedEvidenceModalityCount: Int,
        val selectedEvidenceModalities: List<String>,
        val diagnostics: List<String>,
    )

    private var states = LinkedHashMap<Long, TrackState>()
    private val evidenceArbitrationEngine =
        CrossModalEvidenceArbitrationEngine()
    private var nextTrackId = 1L
    private var lastTimestampNanos = Long.MIN_VALUE
    private var lastAssignments: Map<Long, Long> = emptyMap()

    @Synchronized
    fun configure(value: Config) { config = value }

    fun configuration(): Config = config

    fun evidenceArbitrationConfiguration():
        CrossModalEvidenceArbitrationEngine.Config =
        evidenceArbitrationEngine.configuration()

    @Synchronized
    fun configureEvidenceArbitration(
        value: CrossModalEvidenceArbitrationEngine.Config,
    ) {
        evidenceArbitrationEngine.configure(value)
    }

    @Synchronized
    fun reset() {
        states.clear()
        nextTrackId = 1L
        lastTimestampNanos = Long.MIN_VALUE
        lastAssignments = emptyMap()
        evidenceArbitrationEngine.reset()
    }

    @Synchronized
    fun seed(seeds: List<TrackSeed>) {
        states.clear()
        nextTrackId = 1L
        for (seed in seeds.take(config.maximumTracks)) {
            nextTrackId = max(nextTrackId, seed.trackId + 1L)
            states[seed.trackId] = TrackState(
                trackId = seed.trackId,
                timestampNanos = seed.timestampNanos,
                bodyRoot = seed.bodyRoot,
                faceCenter = seed.faceCenter,
                leftHand = seed.leftHand,
                rightHand = seed.rightHand,
                leftEye = seed.leftEye,
                rightEye = seed.rightEye,
                scaleMeters = seed.scaleMeters,
                confidence = seed.confidence.coerceIn(0.0, 1.0),
            )
        }
        lastTimestampNanos = seeds.maxOfOrNull { it.timestampNanos } ?: Long.MIN_VALUE
    }

    @Synchronized
    fun update(
        timestampNanos: Long,
        observations: List<Observation>,
    ): Result {
        require(timestampNanos > 0L)

        if (lastTimestampNanos != Long.MIN_VALUE && timestampNanos <= lastTimestampNanos) {
            reset()
        }

        val boundedObservations = observations
            .asSequence()
            .filter { it.timestampNanos <= timestampNanos }
            .sortedWith(
                compareByDescending<Observation> { it.confidence }
                    .thenByDescending { it.modalityCount() }
                    .thenBy { it.observationId },
            )
            .take(config.maximumObservations)
            .toList()

        val currentStates = states.values
            .filter { timestampNanos - it.timestampNanos <= nanos(config.maximumAgeSeconds) }
            .take(config.maximumTracks)

        val dtSeconds = if (lastTimestampNanos == Long.MIN_VALUE) {
            1.0 / 30.0
        } else {
            (timestampNanos - lastTimestampNanos).coerceAtLeast(1L) / 1e9
        }

        val arbitrationByObservation =
            boundedObservations.associate { observation ->
                observation.observationId to
                    evidenceArbitrationEngine.arbitrate(
                        arbitrationInput(observation),
                    )
            }

        val pairMatrix = Array(currentStates.size) { row ->
            Array(boundedObservations.size) { column ->
                val observation = boundedObservations[column]
                scorePair(
                    currentStates[row],
                    observation,
                    dtSeconds,
                    arbitrationByObservation[observation.observationId]!!,
                )
            }
        }

        val solution = solveGlobalAssignment(currentStates, boundedObservations, pairMatrix)
        val assignments = mutableListOf<TrackAssignment>()
        val usedObservations = mutableSetOf<Long>()
        val usedTracks = mutableSetOf<Long>()

        for ((row, observationIndex) in solution) {
            val state = currentStates[row]
            val observation = boundedObservations[observationIndex]
            val metrics = pairMatrix[row][observationIndex]
            val previousObservationId = lastAssignments[state.trackId]
            val switched = previousObservationId != null &&
                previousObservationId != observation.observationId

            val baseAction = when {
                metrics.cost <= 1.0 - config.assignmentGate -> Action.ASSIGN
                switched && metrics.cost <= 1.0 - config.switchGate -> Action.SWITCH
                metrics.cost <= 1.0 - config.recoveryGate -> Action.RECOVER
                else -> Action.REJECT
            }

            val action = if (
                baseAction == Action.SWITCH &&
                metrics.cost > 1.0 - config.switchGate - config.hysteresisMargin
            ) {
                Action.HOLD
            } else {
                baseAction
            }

            if (action == Action.REJECT) continue

            usedTracks += state.trackId
            usedObservations += observation.observationId
            updateState(state, observation, timestampNanos, dtSeconds, action)

            assignments += TrackAssignment(
                trackId = state.trackId,
                observationId = observation.observationId,
                cost = metrics.cost,
                confidence = assignmentConfidence(metrics),
                spatialScore = metrics.spatialScore,
                motionScore = metrics.motionScore,
                crossModalScore = metrics.crossModalScore,
                freshnessScore = metrics.freshnessScore,
                action = action,
                predicted = observation.predicted,
                occluded = observation.occluded,
                diagnostics = metrics.diagnostics,
                evidenceFusionScore = metrics.evidenceFusionScore,
                selectedEvidenceModalityCount =
                    metrics.selectedEvidenceModalityCount,
                selectedEvidenceModalities =
                    metrics.selectedEvidenceModalities,
            )
        }

        for (state in currentStates) {
            if (state.trackId in usedTracks) continue

            state.missedFrames += 1
            state.ageSeconds += dtSeconds
            state.confidence *= 0.94

            assignments += TrackAssignment(
                trackId = state.trackId,
                observationId = null,
                cost = config.holdCost.coerceAtMost(1.0),
                confidence = (state.confidence * 0.72).coerceIn(0.0, 1.0),
                spatialScore = 0.0,
                motionScore = motionScoreFromState(state),
                crossModalScore = 0.0,
                freshnessScore = 0.0,
                action = if (state.missedFrames <= 2) Action.HOLD else Action.SEARCH,
                predicted = true,
                occluded = true,
                diagnostics = listOf("observation-missing-for-track", "prediction-only-state"),
            )
        }

        val unmatchedBeforeAcquisition = boundedObservations
            .filter { it.observationId !in usedObservations }

        val availableSlots =
            (config.maximumTracks - states.size)
                .coerceAtLeast(0)

        val acquired = unmatchedBeforeAcquisition
            .take(availableSlots)
            .map { observation ->
                val trackId = nextTrackId++
                val state = newStateFromObservation(
                    trackId = trackId,
                    observation = observation,
                    timestampNanos = timestampNanos,
                )
                states[trackId] = state
                usedObservations += observation.observationId

                val evidence =
                    arbitrationByObservation[observation.observationId]!!

                TrackAssignment(
                    trackId = trackId,
                    observationId = observation.observationId,
                    cost = (
                        1.0 -
                            evidence.fusionConfidence
                        ).coerceIn(0.0, 1.0),
                    confidence = (
                        observation.confidence * 0.74 +
                            evidence.fusionConfidence * 0.26
                        ).coerceIn(0.0, 1.0),
                    spatialScore = 1.0,
                    motionScore = 1.0,
                    crossModalScore =
                        evidence.fusionConfidence,
                    freshnessScore =
                        (
                            1.0 -
                                observation.freshnessSeconds /
                                    config.maximumAgeSeconds
                        ).coerceIn(0.0, 1.0),
                    action = Action.ASSIGN,
                    predicted = observation.predicted,
                    occluded = observation.occluded,
                    diagnostics = (
                        listOf(
                            "new-cross-modal-track-acquired",
                            "identity-state-created-from-observation",
                        ) +
                            evidence.warnings.map {
                                "evidence-$it"
                            }
                        ).distinct(),
                    evidenceFusionScore =
                        evidence.fusionConfidence,
                    selectedEvidenceModalityCount =
                        evidence.selectedModalityCount,
                    selectedEvidenceModalities =
                        evidence.modalities
                            .filter { it.selected }
                            .map { it.modality.name }
                            .sorted(),
                )
            }

        assignments += acquired

        val unmatched = boundedObservations
            .map { it.observationId }
            .filter { it !in usedObservations }

        val orphanTracks = currentStates
            .map { it.trackId }
            .filter { it !in usedTracks }

        val continuity = continuityScore(assignments, boundedObservations.size)
        val swapRisk = swapRisk(assignments, boundedObservations.size)

        val evidenceResults =
            arbitrationByObservation.values.toList()

        val evidenceFusionScore =
            if (evidenceResults.isEmpty()) {
                0.0
            } else {
                evidenceResults
                    .map { it.fusionConfidence }
                    .average()
            }

        val selectedEvidenceModalityCount =
            evidenceResults.sumOf {
                it.selectedModalityCount
            }

        val evidenceWarnings =
            evidenceResults
                .flatMap { it.warnings }
                .distinct()

        val action = when {
            assignments.any { it.action == Action.SWITCH } -> Action.SWITCH
            assignments.any { it.action == Action.RECOVER } -> Action.RECOVER
            assignments.any { it.action == Action.ASSIGN } -> Action.ASSIGN
            assignments.any { it.action == Action.HOLD } -> Action.HOLD
            else -> Action.SEARCH
        }

        val warnings = buildList {
            evidenceWarnings.forEach {
                add("evidence-$it")
            }
            if (swapRisk >= 0.45) add("cross-modal-identity-swap-risk-elevated")
            if (continuity < 0.55) add("cross-modal-identity-continuity-below-target")
            if (unmatched.isNotEmpty()) add("observations-unmatched=" + unmatched.size)
            if (currentStates.any { it.ageSeconds > config.maximumAgeSeconds * 0.70 }) {
                add("track-state-aging")
            }
            if (assignments.count { it.predicted } > assignments.size / 2) {
                add("prediction-dominant-identity-state")
            }
        }.distinct()

        val recommendations = buildList {
            if (swapRisk >= 0.45) {
                add("Increase lateral camera coverage before trusting identity switches.")
            }
            if (continuity < 0.55) {
                add("Preserve temporally adjacent frames and retain face or hand observations during occlusion.")
            }
            if (unmatched.isNotEmpty()) {
                add("Preserve unassigned actor observations for the next association window.")
            }
            if (assignments.any { it.action == Action.RECOVER }) {
                add("Keep the recovered track in a bounded reacquisition state until multimodal agreement improves.")
            }
        }.distinct()

        retireOldStates(timestampNanos)
        lastAssignments = assignments
            .filter { it.observationId != null }
            .associate { it.trackId to requireNotNull(it.observationId) }
        lastTimestampNanos = timestampNanos

        return Result(
            timestampNanos = timestampNanos,
            assignments = assignments.sortedBy { it.trackId },
            continuityScore = continuity,
            swapRisk = swapRisk,
            unmatchedObservationIds = unmatched,
            orphanTrackIds = orphanTracks,
            action = action,
            reason = reasonFor(action, continuity, swapRisk, unmatched.size),
            warnings = warnings,
            recommendations = recommendations,
            evidenceFusionScore = evidenceFusionScore,
            selectedEvidenceModalityCount =
                selectedEvidenceModalityCount,
            evidenceWarnings = evidenceWarnings,
        )
    }

    /**
     * Deterministic minimum-cost maximum-cardinality assignment.
     *
     * A valid match receives a large negative cardinality bonus, so the optimizer first prefers
     * more accepted pairs and then minimizes their real association cost. Invalid pairs are given
     * an effectively unreachable cost. Dummy rows/columns represent unassigned tracks/observations.
     *
     * This replaces exponential branch-and-bound and keeps the association stage predictable as
     * multi-person capture scales beyond the default small rig.
     */
    private fun solveGlobalAssignment(
        states: List<TrackState>,
        observations: List<Observation>,
        costs: Array<Array<PairMetrics>>,
    ): List<Pair<Int, Int>> {
        if (states.isEmpty() || observations.isEmpty()) return emptyList()

        val dimension = max(states.size, observations.size)
        val invalidCost = 1_000_000.0
        val cardinalityBonus = 10_000.0
        val matrix = Array(dimension) { DoubleArray(dimension) }

        for (row in 0 until dimension) {
            for (column in 0 until dimension) {
                matrix[row][column] = when {
                    row >= states.size || column >= observations.size -> 0.0
                    costs[row][column].cost >= config.rejectCost -> invalidCost
                    else ->
                        costs[row][column].cost -
                            cardinalityBonus +
                            row * 1e-9 +
                            column * 1e-12
                }
            }
        }

        val assignment = hungarian(matrix)
        return assignment.toList()
            .mapIndexedNotNull { row, column ->
                if (
                    row >= states.size ||
                    column !in observations.indices
                ) {
                    null
                } else {
                    val metrics = costs[row][column]
                    if (metrics.cost >= config.rejectCost) {
                        null
                    } else {
                        row to column
                    }
                }
            }
    }

    /**
     * Square Hungarian algorithm for deterministic minimum-cost assignment.
     *
     * Complexity is O(n^3) with bounded n from the production identity configuration.
     */
    private fun hungarian(
        matrix: Array<DoubleArray>,
    ): IntArray {
        val n = matrix.size
        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

        for (row in 1..n) {
            p[0] = row
            var column0 = 0
            val minv = DoubleArray(n + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(n + 1)

            do {
                used[column0] = true
                val row0 = p[column0]
                var delta = Double.POSITIVE_INFINITY
                var column1 = 0

                for (column in 1..n) {
                    if (used[column]) continue

                    val current =
                        matrix[row0 - 1][column - 1] -
                            u[row0] -
                            v[column]

                    if (current < minv[column]) {
                        minv[column] = current
                        way[column] = column0
                    }

                    if (
                        minv[column] < delta ||
                        minv[column] == delta &&
                            column < column1
                    ) {
                        delta = minv[column]
                        column1 = column
                    }
                }

                for (column in 0..n) {
                    if (used[column]) {
                        u[p[column]] += delta
                        v[column] -= delta
                    } else {
                        minv[column] -= delta
                    }
                }

                column0 = column1
            } while (p[column0] != 0)

            do {
                val previousColumn = way[column0]
                p[column0] = p[previousColumn]
                column0 = previousColumn
            } while (column0 != 0)
        }

        val assignment = IntArray(n) { -1 }
        for (column in 1..n) {
            val row = p[column]
            if (row in 1..n) {
                assignment[row - 1] = column - 1
            }
        }
        return assignment
    }

    private fun arbitrationInput(
        observation: Observation,
    ): CrossModalEvidenceArbitrationEngine.Input {
        fun evidenceOrDefault(
            modality:
                CrossModalEvidenceArbitrationEngine.Modality,
            measured: Boolean,
        ): CrossModalEvidenceArbitrationEngine.ModalityEvidence? {
            if (!measured) return null

            return observation.evidenceProfile[modality]
                ?: CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                    modality = modality,
                    confidence = observation.confidence,
                    uncertaintyMeters = 0.03,
                    redundancy = 0.10,
                    freshnessSeconds = observation.freshnessSeconds,
                    oodRisk = 0.0,
                    predicted = observation.predicted,
                    predictionOnly = observation.predicted,
                    occluded = observation.occluded,
                )
        }

        return CrossModalEvidenceArbitrationEngine.Input(
            timestampNanos = observation.timestampNanos,
            modalities = listOfNotNull(
                evidenceOrDefault(
                    CrossModalEvidenceArbitrationEngine.Modality.BODY,
                    observation.bodyRoot != null,
                ),
                evidenceOrDefault(
                    CrossModalEvidenceArbitrationEngine.Modality.FACE,
                    observation.faceCenter != null,
                ),
                evidenceOrDefault(
                    CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND,
                    observation.leftHand != null,
                ),
                evidenceOrDefault(
                    CrossModalEvidenceArbitrationEngine.Modality.RIGHT_HAND,
                    observation.rightHand != null,
                ),
                evidenceOrDefault(
                    CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT,
                    observation.leftEye != null,
                ),
                evidenceOrDefault(
                    CrossModalEvidenceArbitrationEngine.Modality.EYE_RIGHT,
                    observation.rightEye != null,
                ),
            ),
        )
    }

    private fun scorePair(
        state: TrackState,
        observation: Observation,
        dtSeconds: Double,
        evidence: CrossModalEvidenceArbitrationEngine.Result,
    ): PairMetrics {
        val predictedBody = state.bodyRoot?.let {
            add(it, scale(state.velocity, dtSeconds.coerceAtMost(config.maximumAgeSeconds)))
        }

        val selectedModalities =
            evidence.modalities
                .filter { it.selected }
                .map { it.modality }
                .toSet()

        val body =
            normalizedResidual(
                predictedBody,
                observation.bodyRoot,
                config.maximumBodyResidualMeters,
            ).takeIf {
                CrossModalEvidenceArbitrationEngine.Modality.BODY in
                    selectedModalities
            }

        val face =
            normalizedResidual(
                state.faceCenter,
                observation.faceCenter,
                config.maximumFaceResidualMeters,
            ).takeIf {
                CrossModalEvidenceArbitrationEngine.Modality.FACE in
                    selectedModalities
            }

        val leftHand =
            normalizedResidual(
                state.leftHand,
                observation.leftHand,
                config.maximumHandResidualMeters,
            ).takeIf {
                CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND in
                    selectedModalities
            }

        val rightHand =
            normalizedResidual(
                state.rightHand,
                observation.rightHand,
                config.maximumHandResidualMeters,
            ).takeIf {
                CrossModalEvidenceArbitrationEngine.Modality.RIGHT_HAND in
                    selectedModalities
            }

        val leftEye =
            normalizedResidual(
                state.leftEye,
                observation.leftEye,
                config.maximumEyeResidualMeters,
            ).takeIf {
                CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT in
                    selectedModalities
            }

        val rightEye =
            normalizedResidual(
                state.rightEye,
                observation.rightEye,
                config.maximumEyeResidualMeters,
            ).takeIf {
                CrossModalEvidenceArbitrationEngine.Modality.EYE_RIGHT in
                    selectedModalities
            }

        val spatialSamples = listOfNotNull(
            body,
            face,
            leftHand,
            rightHand,
            leftEye,
            rightEye,
        )
        val spatialScore = if (spatialSamples.isEmpty()) {
            0.0
        } else {
            exp(-spatialSamples.average() * 2.7).coerceIn(0.0, 1.0)
        }

        val motionResidual = if (predictedBody != null && observation.bodyRoot != null) {
            predictedBody.distanceTo(observation.bodyRoot)
        } else {
            0.0
        }
        val motionScore = exp(
            -motionResidual / config.maximumBodyResidualMeters,
        ).coerceIn(0.0, 1.0)

        val pairedModalities = listOf(
            CrossModalEvidenceArbitrationEngine.Modality.FACE to
                (state.faceCenter to observation.faceCenter),
            CrossModalEvidenceArbitrationEngine.Modality.LEFT_HAND to
                (state.leftHand to observation.leftHand),
            CrossModalEvidenceArbitrationEngine.Modality.RIGHT_HAND to
                (state.rightHand to observation.rightHand),
            CrossModalEvidenceArbitrationEngine.Modality.EYE_LEFT to
                (state.leftEye to observation.leftEye),
            CrossModalEvidenceArbitrationEngine.Modality.EYE_RIGHT to
                (state.rightEye to observation.rightEye),
        ).filter {
            it.first in selectedModalities &&
                it.second.first != null &&
                it.second.second != null
        }

        val crossModalResidual = pairedModalities.map { (_, pair) ->
            requireNotNull(pair.first).distanceTo(
                requireNotNull(pair.second),
            )
        }.averageOrZero()

        val crossModalScore = exp(
            -crossModalResidual / config.maximumFaceResidualMeters.coerceAtLeast(0.05),
        ).coerceIn(0.0, 1.0)

        val modalityAvailability = observation.modalityCount() / 6.0
        val freshnessScore = (
            1.0 - observation.freshnessSeconds / config.maximumAgeSeconds
        ).coerceIn(0.0, 1.0)

        val missingPenalty =
            (1.0 - modalityAvailability) * config.missingModalityPenalty
        val stalePenalty =
            observation.freshnessSeconds * config.staleModalityPenalty
        val predictionPenalty =
            if (observation.predicted) config.predictionPenalty else 0.0
        val occlusionPenalty =
            if (observation.occluded) config.occlusionPenalty else 0.0
        val switchPenalty =
            lastAssignments[state.trackId]
                ?.takeIf { it != observation.observationId }
                ?.let { config.switchCost }
                ?: 0.0

        val evidenceContribution =
            evidence.fusionConfidence.coerceIn(0.0, 1.0)

        val score = (
            spatialScore * config.bodyWeight +
                crossModalScore *
                    (
                        config.faceWeight +
                            config.handWeight +
                            config.eyeWeight
                        ) +
                motionScore * config.motionWeight +
                freshnessScore * 0.06 +
                observation.confidence * 0.06 +
                evidenceContribution * 0.10
            ).coerceIn(0.0, 1.0)

        val cost = (
            1.0 - score +
                (
                    1.0 -
                        evidenceContribution
                    ) * 0.08 +
                missingPenalty +
                stalePenalty +
                predictionPenalty +
                occlusionPenalty +
                switchPenalty
            ).coerceIn(0.0, 2.0)

        return PairMetrics(
            cost = cost,
            spatialScore = spatialScore,
            motionScore = motionScore,
            crossModalScore = crossModalScore,
            freshnessScore = freshnessScore,
            evidenceFusionScore = evidenceContribution,
            selectedEvidenceModalityCount =
                evidence.selectedModalityCount,
            selectedEvidenceModalities =
                evidence.modalities
                    .filter { it.selected }
                    .map { it.modality.name }
                    .sorted(),
            diagnostics = buildList {
                if (observation.bodyRoot == null) add("body-evidence-missing")
                if (observation.faceCenter == null) add("face-evidence-missing")
                if (observation.leftHand == null) add("left-hand-evidence-missing")
                if (observation.rightHand == null) add("right-hand-evidence-missing")
                if (observation.leftEye == null || observation.rightEye == null) {
                    add("binocular-eye-evidence-partial")
                }
                if (observation.predicted) add("prediction-derived-observation")
                if (observation.occluded) add("occlusion-flagged-observation")
                if (observation.freshnessSeconds > 0.20) add("stale-evidence")
                if (switchPenalty > 0.0) add("identity-switch-penalty-active")
                evidence.warnings.forEach {
                    add("cross-modal-evidence-$it")
                }
                if (selectedModalities.isEmpty()) {
                    add("cross-modal-evidence-no-selected-modality")
                }
                if (evidence.fusionConfidence < 0.45) {
                    add("cross-modal-evidence-confidence-low")
                }
            },
        )
    }

    private fun updateState(
        state: TrackState,
        observation: Observation,
        timestampNanos: Long,
        dtSeconds: Double,
        action: Action,
    ) {
        val oldBody = state.bodyRoot
        if (oldBody != null && observation.bodyRoot != null) {
            val measuredVelocity = scale(
                subtract(observation.bodyRoot, oldBody),
                1.0 / dtSeconds.coerceAtLeast(1e-4),
            )
            state.velocity = blend(
                state.velocity,
                clampMagnitude(measuredVelocity, config.maximumVelocityMetersPerSecond),
                config.velocityAlpha,
            )
        } else if (observation.bodyRoot != null) {
            state.velocity = ZERO
        }

        fun fuse(previous: Vec3?, incoming: Vec3?, alpha: Double): Vec3? =
            when {
                incoming == null -> previous
                previous == null -> incoming
                else -> blend(previous, incoming, alpha)
            }

        val alpha = when (action) {
            Action.RECOVER -> 0.30
            Action.SWITCH -> 0.18
            Action.HOLD -> 0.12
            Action.ASSIGN -> 0.64
            else -> 0.48
        }

        state.bodyRoot = fuse(state.bodyRoot, observation.bodyRoot, alpha)
        state.faceCenter = fuse(state.faceCenter, observation.faceCenter, alpha)
        state.leftHand = fuse(state.leftHand, observation.leftHand, alpha)
        state.rightHand = fuse(state.rightHand, observation.rightHand, alpha)
        state.leftEye = fuse(state.leftEye, observation.leftEye, alpha)
        state.rightEye = fuse(state.rightEye, observation.rightEye, alpha)
        state.scaleMeters = observation.scaleMeters ?: state.scaleMeters
        state.timestampNanos = timestampNanos
        state.confidence = (
            state.confidence * (1.0 - config.historyAlpha) +
                observation.confidence * config.historyAlpha
        ).coerceIn(0.0, 1.0)
        state.ageSeconds = 0.0
        state.missedFrames = 0
    }

    private fun newStateFromObservation(
        trackId: Long,
        observation: Observation,
        timestampNanos: Long,
    ): TrackState =
        TrackState(
            trackId = trackId,
            timestampNanos = timestampNanos,
            bodyRoot = observation.bodyRoot,
            faceCenter = observation.faceCenter,
            leftHand = observation.leftHand,
            rightHand = observation.rightHand,
            leftEye = observation.leftEye,
            rightEye = observation.rightEye,
            scaleMeters = observation.scaleMeters,
            confidence = observation.confidence.coerceIn(0.0, 1.0),
            velocity = ZERO,
            ageSeconds = 0.0,
            missedFrames = 0,
        )

    private fun assignmentConfidence(metrics: PairMetrics): Double =
        (
            metrics.spatialScore * 0.32 +
                metrics.motionScore * 0.18 +
                metrics.crossModalScore * 0.18 +
                metrics.freshnessScore * 0.08 +
                metrics.evidenceFusionScore * 0.16 +
                (1.0 - metrics.cost.coerceIn(0.0, 1.0)) * 0.08
            ).coerceIn(0.0, 1.0)

    private fun continuityScore(assignments: List<TrackAssignment>, observationCount: Int): Double {
        if (assignments.isEmpty()) return 0.0
        val assigned = assignments.filter { it.observationId != null }
        val assignmentRatio = assigned.size.toDouble() /
            max(1, max(assignments.size, observationCount))
        val meanConfidence = assigned.map { it.confidence }.averageOrZero()
        val stability = assigned.count {
            it.action == Action.ASSIGN || it.action == Action.HOLD
        }.toDouble() / assigned.size.coerceAtLeast(1)
        val recoveryPenalty = assigned.count { it.action == Action.RECOVER } * 0.04
        val switchPenalty = assigned.count { it.action == Action.SWITCH } * 0.10
        return (
            assignmentRatio * 0.30 +
                meanConfidence * 0.46 +
                stability * 0.24 -
                recoveryPenalty -
                switchPenalty
            ).coerceIn(0.0, 1.0)
    }

    private fun swapRisk(assignments: List<TrackAssignment>, observationCount: Int): Double {
        if (assignments.isEmpty() || observationCount <= 1) return 0.0
        val switchRatio = assignments.count { it.action == Action.SWITCH }.toDouble() /
            assignments.size.coerceAtLeast(1)
        val lowConfidenceRatio = assignments.count { it.confidence < 0.55 }.toDouble() /
            assignments.size.coerceAtLeast(1)
        val weakCrossModalRatio = assignments.count { it.crossModalScore < 0.40 }.toDouble() /
            assignments.size.coerceAtLeast(1)
        return (
            switchRatio * 0.55 +
                lowConfidenceRatio * 0.25 +
                weakCrossModalRatio * 0.20
            ).coerceIn(0.0, 1.0)
    }

    private fun retireOldStates(timestampNanos: Long) {
        val iterator = states.iterator()
        while (iterator.hasNext()) {
            val state = iterator.next().value
            if (timestampNanos - state.timestampNanos > nanos(config.maximumAgeSeconds)) {
                iterator.remove()
            }
        }
    }

    private fun reasonFor(
        action: Action,
        continuity: Double,
        swapRisk: Double,
        unmatchedCount: Int,
    ): String = when {
        action == Action.SWITCH -> "identity-switch-candidate-requires-confirmation"
        action == Action.RECOVER -> "identity-track-reacquiring-from-temporal-memory"
        swapRisk >= 0.45 -> "multimodal-disagreement-increases-swap-risk"
        continuity >= 0.82 -> "multimodal-identity-continuity-stable"
        unmatchedCount > 0 -> "some-observations-remain-unmatched"
        else -> "identity-continuity-within-production-bounds"
    }

    private fun normalizedResidual(a: Vec3?, b: Vec3?, maximum: Double): Double? =
        if (a == null || b == null) null else (
            a.distanceTo(b) / maximum.coerceAtLeast(1e-6)
        ).coerceIn(0.0, 2.0)

    private fun motionScoreFromState(state: TrackState): Double =
        exp(-state.velocity.norm() / config.maximumVelocityMetersPerSecond)
            .coerceIn(0.0, 1.0) * state.confidence.coerceIn(0.0, 1.0)

    private fun nanos(seconds: Double): Long = (seconds * 1e9).toLong().coerceAtLeast(1L)

    private fun clampMagnitude(value: Vec3, maximum: Double): Vec3 {
        val length = value.norm()
        if (!length.isFinite() || length <= maximum) return value
        return scale(value, maximum / length.coerceAtLeast(1e-9))
    }

    private fun add(a: Vec3, b: Vec3): Vec3 =
        Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

    private fun subtract(a: Vec3, b: Vec3): Vec3 =
        Vec3(a.x - b.x, a.y - b.y, a.z - b.z)

    private fun scale(a: Vec3, scalar: Double): Vec3 =
        Vec3(a.x * scalar, a.y * scalar, a.z * scalar)

    private fun blend(a: Vec3, b: Vec3, alpha: Double): Vec3 =
        Vec3(
            a.x + (b.x - a.x) * alpha,
            a.y + (b.y - a.y) * alpha,
            a.z + (b.z - a.z) * alpha,
        )

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else average()

    private companion object {
        val ZERO = Vec3(0.0, 0.0, 0.0)
    }
}
