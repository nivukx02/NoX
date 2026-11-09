package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Bounded global one-to-one identity continuation for rig-level subject components.
 *
 * This is deliberately separate from pixel-level association. Camera-local observations are first
 * assembled into current rig components; this engine then decides which previous rig identities
 * continue into the new frame. One previous rig identity can be assigned at most once, preventing
 * local "best previous track" reuse from creating identity duplication during crossings.
 *
 * Evidence channels:
 * - semantic type compatibility;
 * - calibrated metric-center continuity when available upstream;
 * - motion-vector continuity;
 * - per-camera local-track continuity;
 * - confidence and evidence-count support.
 *
 * The matcher fails closed for weak identity continuity. New rig identities are created only when
 * no prior identity clears the configured continuation gate.
 */
class GlobalRigIdentityAssignmentEngine(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumTracks: Int = 64,
        val minimumMatchScore: Double = 0.46,
        val centerScaleMeters: Double = 0.50,
        val velocityScaleMetersPerSecond: Double = 1.75,
        val semanticWeight: Double = 0.16,
        val centerWeight: Double = 0.34,
        val velocityWeight: Double = 0.24,
        val localTrackWeight: Double = 0.18,
        val confidenceWeight: Double = 0.08,
        val ambiguityMargin: Double = 0.08,
    ) {
        init {
            require(maximumTracks in 1..128)
            require(minimumMatchScore in 0.0..1.0)
            require(centerScaleMeters > 0.0)
            require(velocityScaleMetersPerSecond > 0.0)
            listOf(
                semanticWeight,
                centerWeight,
                velocityWeight,
                localTrackWeight,
                confidenceWeight,
            ).forEach {
                require(it.isFinite() && it >= 0.0)
            }
            require(
                semanticWeight +
                    centerWeight +
                    velocityWeight +
                    localTrackWeight +
                    confidenceWeight >
                    0.0,
            )
            require(ambiguityMargin >= 0.0)
        }
    }

    data class Track(
        val rigTrackId: Long,
        val subjectType: SubjectType,
        val center: Vec2,
        val centerIsMetric: Boolean = false,
        val velocity: Vec2,
        val confidence: Double,
        val observationCount: Int,
        val localTrackIds: Map<String, Long> = emptyMap(),
    ) {
        init {
            require(rigTrackId >= 0L)
            require(
                center.x.isFinite() &&
                    center.y.isFinite(),
            )
            require(
                velocity.x.isFinite() &&
                    velocity.y.isFinite(),
            )
            require(confidence in 0.0..1.0)
            require(observationCount >= 0)
            require(localTrackIds.values.all { it >= 0L })
        }
    }

    data class Assignment(
        val currentIndex: Int,
        val rigTrackId: Long,
        val score: Double,
        val margin: Double,
    )

    data class Result(
        val assignments: List<Assignment>,
        val unmatchedCurrentIndices: List<Int>,
        val unmatchedPreviousIds: List<Long>,
        val globalConfidence: Double,
        val ambiguityRisk: Double,
        val boundedInput: Boolean,
        val diagnostics: List<String>,
    )

    fun assign(
        current: List<Track>,
        previous: List<Track>,
    ): Result {
        // Preserve caller order so assignment indices remain stable across the integration
        // boundary. The caller already produces a deterministic component ordering.
        val currentBounded =
            current.take(config.maximumTracks)

        val previousBounded =
            previous
                .filter { it.rigTrackId > 0L }
                .distinctBy { it.rigTrackId }
                .sortedBy { it.rigTrackId }
                .take(config.maximumTracks)

        val boundedInput =
            currentBounded.size != current.size ||
                previousBounded.size !=
                previous
                    .filter { it.rigTrackId > 0L }
                    .distinctBy { it.rigTrackId }
                    .size

        if (
            currentBounded.isEmpty() ||
            previousBounded.isEmpty()
        ) {
            return Result(
                assignments = emptyList(),
                unmatchedCurrentIndices =
                    current.indices.toList(),
                unmatchedPreviousIds =
                    previousBounded.map { it.rigTrackId },
                globalConfidence = 0.0,
                ambiguityRisk = 0.0,
                boundedInput = boundedInput,
                diagnostics =
                    listOf(
                        if (currentBounded.isEmpty()) {
                            "rig-identity-no-current-components"
                        } else {
                            "rig-identity-no-previous-identities"
                        },
                    ),
            )
        }

        val n =
            max(
                currentBounded.size,
                previousBounded.size,
            )

        val matrix =
            Array(n) {
                DoubleArray(n) {
                    0.0
                }
            }

        for (currentIndex in currentBounded.indices) {
            val now =
                currentBounded[currentIndex]
            for (
                previousIndex in previousBounded.indices
            ) {
                val before =
                    previousBounded[previousIndex]

                matrix[currentIndex][previousIndex] =
                    pairScore(
                        current = now,
                        previous = before,
                    )
            }
        }

        val assignment =
            hungarian(
                matrix = matrix,
            )

        val accepted =
            ArrayList<Assignment>()
        val usedPrevious =
            HashSet<Long>()
        val unmatchedCurrent =
            mutableListOf<Int>().apply {
                if (currentBounded.size < current.size) {
                    addAll(
                        currentBounded.size until current.size,
                    )
                }
            }

        for (currentIndex in currentBounded.indices) {
            val previousIndex =
                assignment[currentIndex]

            if (
                previousIndex !in
                previousBounded.indices
            ) {
                unmatchedCurrent += currentIndex
                continue
            }

            val score =
                matrix[currentIndex][previousIndex]

            val margin =
                assignmentMargin(
                    currentIndex = currentIndex,
                    assignedPreviousIndex =
                        previousIndex,
                    matrix = matrix,
                )

            if (
                score >= config.minimumMatchScore &&
                margin >= 0.0
            ) {
                val rigTrackId =
                    previousBounded[previousIndex]
                        .rigTrackId
                accepted +=
                    Assignment(
                        currentIndex = currentIndex,
                        rigTrackId = rigTrackId,
                        score = score,
                        margin = margin,
                    )
                usedPrevious += rigTrackId
            } else {
                unmatchedCurrent += currentIndex
            }
        }

        val unmatchedPrevious =
            previousBounded
                .map { it.rigTrackId }
                .filter { it !in usedPrevious }

        val assignmentQuality =
            accepted
                .map { assignment ->
                    assignment.score *
                        (
                            0.74 +
                                0.26 *
                                    (
                                        assignment.margin /
                                            config.ambiguityMargin
                                                .coerceAtLeast(0.08)
                                    ).coerceIn(0.0, 1.0)
                        )
                }
                .average()
                .takeIf { accepted.isNotEmpty() }
                ?: 0.0

        val unmatchedCurrentRatio =
            unmatchedCurrent.size.toDouble() /
                current.size.coerceAtLeast(1).toDouble()

        val coverageConfidence =
            (
                1.0 -
                    unmatchedCurrentRatio
            ).coerceIn(0.0, 1.0)

        val globalConfidence =
            (
                assignmentQuality * 0.82 +
                    coverageConfidence * 0.18
            ).coerceIn(0.0, 1.0)

        val assignmentAmbiguity =
            if (accepted.isEmpty()) {
                0.0
            } else {
                accepted
                    .map {
                        (
                            1.0 -
                                (
                                    it.margin /
                                        config.ambiguityMargin
                                            .coerceAtLeast(0.08)
                                ).coerceIn(0.0, 1.0)
                        )
                    }
                    .average()
                    .coerceIn(0.0, 1.0)
            }

        val ambiguityRisk =
            (
                assignmentAmbiguity * 0.80 +
                    unmatchedCurrentRatio * 0.20
            ).coerceIn(0.0, 1.0)

        val diagnostics =
            buildList {
                add(
                    "rig-identity-current=" +
                        currentBounded.size,
                )
                add(
                    "rig-identity-previous=" +
                        previousBounded.size,
                )
                add(
                    "rig-identity-assigned=" +
                        accepted.size,
                )
                add(
                    "rig-identity-unmatched-current=" +
                        unmatchedCurrent.size,
                )
                add(
                    "rig-identity-unmatched-previous=" +
                        unmatchedPrevious.size,
                )
                add(
                    "rig-identity-confidence=" +
                        format(globalConfidence),
                )
                add(
                    "rig-identity-ambiguity=" +
                        format(ambiguityRisk),
                )
                if (boundedInput) {
                    add(
                        "rig-identity-input-bounded",
                    )
                }
                if (
                    unmatchedCurrentRatio >= 0.20
                ) {
                    add(
                        "rig-identity-current-coverage-degraded",
                    )
                }
                if (
                    ambiguityRisk >= 0.40
                ) {
                    add(
                        "rig-identity-exchange-risk-elevated",
                    )
                }
            }

        return Result(
            assignments =
                accepted.sortedBy { it.currentIndex },
            unmatchedCurrentIndices =
                unmatchedCurrent,
            unmatchedPreviousIds =
                unmatchedPrevious,
            globalConfidence =
                globalConfidence,
            ambiguityRisk =
                ambiguityRisk,
            boundedInput =
                boundedInput,
            diagnostics =
                diagnostics,
        )
    }

    private fun pairScore(
        current: Track,
        previous: Track,
    ): Double {
        if (
            current.subjectType !=
            previous.subjectType
        ) {
            return 0.0
        }

        val semantic =
            1.0

        val center =
            if (
                current.centerIsMetric &&
                    previous.centerIsMetric
            ) {
                val centerDistance =
                    current.center.distanceTo(
                        previous.center,
                    )
                exp(
                    -centerDistance /
                        config.centerScaleMeters,
                ).coerceIn(0.0, 1.0)
            } else {
                0.55
            }

        val velocityDelta =
            (
                current.velocity -
                    previous.velocity
                ).norm()

        val velocity =
            exp(
                -velocityDelta /
                    config.velocityScaleMetersPerSecond,
            ).coerceIn(0.0, 1.0)

        val sharedLocalTracks =
            if (
                current.localTrackIds.isEmpty() ||
                previous.localTrackIds.isEmpty()
            ) {
                0.0
            } else {
                val shared =
                    current.localTrackIds
                        .asSequence()
                        .count { (device, localId) ->
                            previous.localTrackIds[device] ==
                                localId
                        }
                val denominator =
                    max(
                        current.localTrackIds.size,
                        previous.localTrackIds.size,
                    )
                (
                    shared.toDouble() /
                        denominator
                            .coerceAtLeast(1)
                            .toDouble()
                ).coerceIn(0.0, 1.0)
            }

        val confidence =
            min(
                current.confidence,
                previous.confidence,
            ).coerceIn(0.0, 1.0)

        return (
            semantic * config.semanticWeight +
                center * config.centerWeight +
                velocity * config.velocityWeight +
                sharedLocalTracks *
                config.localTrackWeight +
                confidence *
                config.confidenceWeight
            ) /
            (
                config.semanticWeight +
                    config.centerWeight +
                    config.velocityWeight +
                    config.localTrackWeight +
                    config.confidenceWeight
            ).coerceAtLeast(1e-9)
                .coerceIn(0.0, 1.0)
    }

    private fun assignmentMargin(
        currentIndex: Int,
        assignedPreviousIndex: Int,
        matrix: Array<DoubleArray>,
    ): Double {
        val assigned =
            matrix[currentIndex][assignedPreviousIndex]

        val alternative =
            matrix[currentIndex]
                .filterIndexed {
                    index, _ ->
                    index !=
                        assignedPreviousIndex
                }
                .maxOrNull()
                ?: 0.0

        return (
            assigned -
                alternative
            ).coerceIn(
                -1.0,
                1.0,
            )
    }

    /**
     * Maximum-weight Hungarian assignment with zero dummy cells.
     */
    private fun hungarian(
        matrix: Array<DoubleArray>,
    ): IntArray {
        val n = matrix.size
        if (n == 0) {
            return IntArray(0)
        }

        val u =
            DoubleArray(n + 1)
        val v =
            DoubleArray(n + 1)
        val p =
            IntArray(n + 1)
        val way =
            IntArray(n + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv =
                DoubleArray(n + 1) {
                    Double.POSITIVE_INFINITY
                }
            val used =
                BooleanArray(n + 1)

            do {
                used[j0] = true
                val i0 = p[j0]
                var delta =
                    Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) continue

                    val cost =
                        -matrix[i0 - 1][j - 1]

                    val cur =
                        cost -
                            u[i0] -
                            v[j]

                    if (
                        cur < minv[j] ||
                        (
                            cur ==
                                minv[j] &&
                                j < j1
                            )
                    ) {
                        minv[j] = cur
                        way[j] = j0
                    }

                    if (
                        minv[j] < delta ||
                        (
                            minv[j] ==
                                delta &&
                                j < j1
                            )
                    ) {
                        delta = minv[j]
                        j1 = j
                    }
                }

                for (j in 0..n) {
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

        val result =
            IntArray(n) { -1 }

        for (j in 1..n) {
            if (p[j] != 0) {
                result[p[j] - 1] =
                    j - 1
            }
        }

        return result
    }

    private fun format(
        value: Double,
    ): String =
        "%.3f".format(
            java.util.Locale.US,
            value.coerceIn(0.0, 1.0),
        )
}
