package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Bounded global multi-actor identity assignment in metric world space.
 *
 * The engine keeps the frontier hypothesis layer independent from detector implementation and uses:
 * - geometry against the previous world-space prediction;
 * - explicit depth consistency when a metric depth observation exists;
 * - appearance evidence as a secondary identity cue;
 * - detector confidence;
 * - hard contradiction gates for physically incompatible candidates;
 * - deterministic Hungarian assignment for one-to-one primary identity;
 * - bounded alternate hypotheses for downstream ambiguity handling.
 *
 * Identity assignment is evidence, not ground truth. Uncalibrated or weak observations cannot silently
 * acquire high-authority metric identity.
 */
class MultiActorHypothesisAssociationEngine(
    private val config: Config = Config(),
) {
    data class Config(
        val maximumActors: Int = 64,
        val maximumObservations: Int = 256,
        val maximumHypothesesPerActor: Int = 3,
        val minimumAcceptanceScore: Double = 0.30,
        val ambiguityMargin: Double = 0.08,
        val hardGeometryContradictionDistanceMeters: Double = 0.90,
        val hardDepthContradictionMeters: Double = 0.45,
        val geometryWeight: Double = 0.55,
        val appearanceWeight: Double = 0.25,
        val confidenceWeight: Double = 0.10,
        val depthWeight: Double = 0.10,
        val velocityWeight: Double = 0.10,
        val detectorHintBonus: Double = 0.12,
        val minimumAppearanceQualityForHardGate: Double = 0.75,
        val hardAppearanceMismatchScore: Double = 0.20,
        val unmatchedCost: Double = 0.70,
    ) {
        init {
            require(maximumActors in 1..64)
            require(maximumObservations in 1..512)
            require(maximumHypothesesPerActor in 1..8)
            require(minimumAcceptanceScore in 0.0..1.0)
            require(ambiguityMargin in 0.0..0.5)
            require(hardGeometryContradictionDistanceMeters > 0.0)
            require(hardDepthContradictionMeters > 0.0)
            require(geometryWeight >= 0.0 && appearanceWeight >= 0.0)
            require(confidenceWeight >= 0.0 && depthWeight >= 0.0)
            require(velocityWeight >= 0.0)
            require(
                geometryWeight + appearanceWeight +
                    confidenceWeight + depthWeight + velocityWeight > 0.0
            )
            require(detectorHintBonus >= 0.0)
            require(minimumAppearanceQualityForHardGate in 0.0..1.0)
            require(hardAppearanceMismatchScore in 0.0..1.0)
            require(unmatchedCost in 0.0..1.0)
        }
    }

    data class ActorObservation(
        val detectionId: Long,
        val actorHint: Int?,
        val timestampNanos: Long,
        val centerMeters: Vec3,
        val appearanceScoreByActor: Map<Int, Double> = emptyMap(),
        val depthMeters: Double? = null,
        val confidence: Double = 1.0,
        val appearanceQuality: Double = 0.0,
        val velocityMetersPerSecond: Vec3? = null,
    ) {
        init {
            require(detectionId >= 0L)
            require(timestampNanos >= 0L)
            require(finite(centerMeters))
            actorHint?.let { require(it in 0..63) }
            depthMeters?.let {
                require(it.isFinite() && it > 0.0)
            }
            require(confidence.isFinite() && confidence in 0.0..1.0)
            require(appearanceQuality.isFinite() && appearanceQuality in 0.0..1.0)
            require(appearanceScoreByActor.keys.all { it >= 0 })
            require(
                appearanceScoreByActor.values.all {
                    it.isFinite() && it in 0.0..1.0
                }
            )
            velocityMetersPerSecond?.let {
                require(finite(it))
            }
        }
    }

    data class ActorTrack(
        val actorId: Int,
        val detectionId: Long,
        val score: Double,
        val alternate: Boolean,
    )

    data class Result(
        val accepted: List<ActorTrack>,
        val rejectedDetectionIds: Set<Long>,
        val ambiguousDetectionIds: Set<Long>,
        val primaryAssignmentCount: Int = accepted.count { !it.alternate },
        val alternateHypothesisCount: Int = accepted.count { it.alternate },
        val assignmentObjective: Double = 0.0,
        val globallyOptimized: Boolean = true,
        val warnings: List<String> = emptyList(),
    )

    private data class Candidate(
        val observationIndex: Int,
        val observation: ActorObservation,
        val actorId: Int,
        val score: Double,
    )

    fun associate(
        observations: List<ActorObservation>,
        predictions: Map<Int, Vec3>,
        predictedVelocityByActor: Map<Int, Vec3> = emptyMap(),
    ): Result {
        require(predictions.keys.all { it in 0 until config.maximumActors })
        require(predictedVelocityByActor.keys.all { it in 0 until config.maximumActors })
        require(predictedVelocityByActor.values.all(::finite))

        val bounded = observations
            .distinctBy { it.detectionId }
            .sortedWith(
                compareByDescending<ActorObservation> { it.confidence }
                    .thenBy { it.detectionId }
            )

        val overflowRejected = bounded
            .drop(config.maximumObservations)
            .map { it.detectionId }
            .toMutableSet()

        val work = bounded
            .take(config.maximumObservations)

        if (work.isEmpty()) {
            return Result(
                accepted = emptyList(),
                rejectedDetectionIds = overflowRejected,
                ambiguousDetectionIds = emptySet(),
                warnings = if (overflowRejected.isNotEmpty()) {
                    listOf("identity-observation-budget-truncated")
                } else {
                    emptyList()
                },
            )
        }

        val candidates = buildCandidates(
            observations = work,
            predictions = predictions,
            predictedVelocityByActor = predictedVelocityByActor,
        )
        if (candidates.isEmpty()) {
            overflowRejected += work.map { it.detectionId }
            return Result(
                accepted = emptyList(),
                rejectedDetectionIds = overflowRejected,
                ambiguousDetectionIds = emptySet(),
                warnings = listOf("identity-no-plausible-candidates"),
            )
        }

        val byObservation = candidates.groupBy { it.observationIndex }
        val actorIds = candidates
            .map { it.actorId }
            .distinct()
            .sorted()

        val ambiguous = LinkedHashSet<Long>()
        byObservation.forEach { (_, options) ->
            val sorted = options.sortedWith(
                compareByDescending<Candidate> { it.score }
                    .thenBy { it.actorId }
            )
            val best = sorted.getOrNull(0)
            val second = sorted.getOrNull(1)
            if (
                best != null &&
                second != null &&
                best.score >= config.minimumAcceptanceScore &&
                best.score - second.score < config.ambiguityMargin
            ) {
                ambiguous += best.observation.detectionId
            }
        }

        val matrixSize = max(work.size, actorIds.size)
        val unmatchedCost = config.unmatchedCost
        val forbiddenCost = 1_000_000.0
        val cost = Array(matrixSize) {
            DoubleArray(matrixSize) { unmatchedCost }
        }

        for (row in work.indices) {
            for (column in actorIds.indices) {
                val actorId = actorIds[column]
                val candidate = byObservation[row]
                    .orEmpty()
                    .firstOrNull { it.actorId == actorId }
                cost[row][column] = when {
                    candidate == null -> forbiddenCost
                    candidate.score < config.minimumAcceptanceScore -> forbiddenCost
                    else -> 1.0 - candidate.score
                }
            }
        }

        val assignment = hungarianMinimum(cost)

        val primaries = ArrayList<ActorTrack>()
        val occupiedActors = LinkedHashSet<Int>()
        var objective = 0.0

        assignment.forEachIndexed { row, column ->
            if (row !in work.indices || column !in actorIds.indices) {
                return@forEachIndexed
            }

            val actorId = actorIds[column]
            val candidate = byObservation[row]
                .orEmpty()
                .firstOrNull { it.actorId == actorId }
                ?: return@forEachIndexed

            if (candidate.score < config.minimumAcceptanceScore) {
                return@forEachIndexed
            }

            val assignedCost = cost[row][column]
            if (!assignedCost.isFinite() || assignedCost >= forbiddenCost * 0.5) {
                return@forEachIndexed
            }

            primaries += ActorTrack(
                actorId = actorId,
                detectionId = candidate.observation.detectionId,
                score = candidate.score,
                alternate = false,
            )
            occupiedActors += actorId
            objective += candidate.score
        }

        val alternates = ArrayList<ActorTrack>()
        primaries.sortedWith(
            compareBy<ActorTrack> { it.detectionId }
                .thenByDescending { it.score }
        ).forEach { primary ->
            val observationIndex = work.indexOfFirst {
                it.detectionId == primary.detectionId
            }
            if (observationIndex < 0) return@forEach

            byObservation[observationIndex]
                .orEmpty()
                .asSequence()
                .filter { it.actorId != primary.actorId }
                .filter { it.actorId !in occupiedActors }
                .filter {
                    it.score >= (
                        config.minimumAcceptanceScore * 0.85
                    )
                }
                .sortedWith(
                    compareByDescending<Candidate> { it.score }
                        .thenBy { it.actorId }
                )
                .take(
                    (config.maximumHypothesesPerActor - 1)
                        .coerceAtLeast(0)
                )
                .forEach { candidate ->
                    alternates += ActorTrack(
                        actorId = candidate.actorId,
                        detectionId = candidate.observation.detectionId,
                        score = candidate.score,
                        alternate = true,
                    )
                }
        }

        val acceptedDetectionIds = primaries
            .map { it.detectionId }
            .toSet()

        work.forEach { observation ->
            if (observation.detectionId !in acceptedDetectionIds) {
                overflowRejected += observation.detectionId
            }
        }

        val warnings = buildList {
            if (overflowRejected.isNotEmpty()) {
                add("identity-observations-rejected")
            }
            if (ambiguous.isNotEmpty()) {
                add("identity-ambiguity-present")
            }
            if (actorIds.size > config.maximumActors) {
                add("identity-actor-budget-truncated")
            }
            if (alternates.isNotEmpty()) {
                add("identity-alternate-hypotheses-present")
            }
        }.distinct()

        return Result(
            accepted = (
                primaries + alternates
                ).sortedWith(
                    compareBy<ActorTrack> { it.detectionId }
                        .thenBy { it.alternate }
                        .thenBy { it.actorId }
                ),
            rejectedDetectionIds = overflowRejected,
            ambiguousDetectionIds = ambiguous,
            primaryAssignmentCount = primaries.size,
            alternateHypothesisCount = alternates.size,
            assignmentObjective = objective,
            globallyOptimized = true,
            warnings = warnings,
        )
    }

    private fun buildCandidates(
        observations: List<ActorObservation>,
        predictions: Map<Int, Vec3>,
        predictedVelocityByActor: Map<Int, Vec3>,
    ): List<Candidate> {
        val result = ArrayList<Candidate>()

        observations.forEachIndexed { observationIndex, observation ->
            val actorPool: List<Int> =
                if (observation.actorHint != null && observation.actorHint in 0 until config.maximumActors) {
                    listOf(observation.actorHint)
                } else {
                    (
                        predictions.keys +
                            observation.appearanceScoreByActor.keys
                        )
                        .filter { it in 0 until config.maximumActors }
                        .distinct()
                        .sorted()
                }

            actorPool.forEach { actorId ->
                val prediction = predictions[actorId]

                val geometryScore =
                    prediction?.let {
                        val distance =
                            observation.centerMeters.distanceTo(it)
                        if (
                            !distance.isFinite() ||
                            distance >
                                config.hardGeometryContradictionDistanceMeters
                        ) {
                            null
                        } else {
                            1.0 -
                                distance /
                                config.hardGeometryContradictionDistanceMeters
                        }
                    } ?: 0.35

                if (geometryScore == null) return@forEach

                val depthScore =
                    observation.depthMeters?.let { measuredDepth ->
                        prediction?.let { predicted ->
                            val difference =
                                abs(measuredDepth - predicted.z)
                            if (
                                !difference.isFinite() ||
                                difference >
                                    config.hardDepthContradictionMeters
                            ) {
                                null
                            } else {
                                1.0 -
                                    difference /
                                    config.hardDepthContradictionMeters
                            }
                        } ?: 0.35
                    } ?: 0.35

                if (depthScore == null) return@forEach

                val velocityScore =
                    if (
                        observation.velocityMetersPerSecond != null &&
                        predictedVelocityByActor[actorId] != null
                    ) {
                        val difference =
                            observation.velocityMetersPerSecond
                                .distanceTo(
                                    predictedVelocityByActor
                                        .getValue(actorId),
                                )
                        (
                            1.0 -
                                difference /
                                12.0
                        ).coerceIn(0.0, 1.0)
                    } else {
                        0.35
                    }

                val appearance =
                    observation.appearanceScoreByActor[actorId] ?: 0.35

                if (
                    observation.appearanceQuality >=
                        config.minimumAppearanceQualityForHardGate &&
                    actorId !in observation.appearanceScoreByActor
                ) {
                    return@forEach
                }

                if (
                    observation.appearanceQuality >=
                        config.minimumAppearanceQualityForHardGate &&
                    appearance <
                        config.hardAppearanceMismatchScore
                ) {
                    return@forEach
                }

                val weighted =
                    (
                        geometryScore * config.geometryWeight +
                            appearance * config.appearanceWeight +
                            observation.confidence *
                            config.confidenceWeight +
                            depthScore * config.depthWeight +
                            velocityScore * config.velocityWeight
                        ) / (
                            config.geometryWeight +
                                config.appearanceWeight +
                                config.confidenceWeight +
                                config.depthWeight +
                                config.velocityWeight
                            )

                val hintBonus =
                    if (observation.actorHint == actorId) {
                        config.detectorHintBonus
                    } else {
                        0.0
                    }

                val score =
                    (
                        weighted + hintBonus
                    ).coerceIn(0.0, 1.0)

                result += Candidate(
                    observationIndex = observationIndex,
                    observation = observation,
                    actorId = actorId,
                    score = score,
                )
            }
        }

        return result
            .sortedWith(
                compareBy<Candidate> { it.observationIndex }
                    .thenBy { it.actorId }
            )
    }

    private fun hungarianMinimum(
        input: Array<DoubleArray>,
    ): IntArray {
        val n = input.size
        if (n == 0) return IntArray(0)

        val m = input.first().size
        require(m == n)

        val u = DoubleArray(n + 1)
        val v = DoubleArray(m + 1)
        val p = IntArray(m + 1)
        val way = IntArray(m + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = DoubleArray(m + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(m + 1)

            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..m) {
                    if (used[j]) continue

                    val cij = input[i0 - 1][j - 1]
                    val current =
                        cij - u[i0] - v[j]

                    if (current < minv[j]) {
                        minv[j] = current
                        way[j] = j0
                    }

                    if (
                        minv[j] < delta ||
                        (
                            minv[j] == delta &&
                                j < j1
                            )
                    ) {
                        delta = minv[j]
                        j1 = j
                    }
                }

                for (j in 0..m) {
                    if (used[j]) {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }

                j0 = j1
            } while (p[j0] != 0)

            do {
                val j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while (j0 != 0)
        }

        return IntArray(n) { row ->
            var column = -1
            for (j in 1..m) {
                if (p[j] == row + 1) {
                    column = j - 1
                    break
                }
            }
            column
        }
    }

    private fun Vec3.distanceTo(other: Vec3): Double =
        sqrt(
            (x - other.x) * (x - other.x) +
                (y - other.y) * (y - other.y) +
                (z - other.z) * (z - other.z),
        )

    private fun finite(v: Vec3): Boolean =
        v.x.isFinite() && v.y.isFinite() && v.z.isFinite()
}

private fun Vec3.distanceTo(other: Vec3): Double =
    max(
        0.0,
        sqrt(
            (x - other.x) * (x - other.x) +
                (y - other.y) * (y - other.y) +
                (z - other.z) * (z - other.z),
        ),
    )
