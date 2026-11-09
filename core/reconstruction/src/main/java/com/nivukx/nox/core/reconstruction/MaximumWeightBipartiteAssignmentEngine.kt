package com.nivukx.nox.core.reconstruction

import kotlin.math.max

/**
 * Bounded maximum-weight one-to-one assignment for dense multi-person identity fusion.
 *
 * This is intentionally deterministic. Missing candidate edges are represented by a forbidden
 * score, while zero-weight dummy assignments allow observations or references to remain unmatched.
 *
 * Complexity is O(n^3) with n <= max(leftCount, rightCount) after bounded input admission.
 */
class MaximumWeightBipartiteAssignmentEngine(
    private val config: Config = Config(),
) {
    data class Config(
        // Supports the production 64-track / 128-observation cross-modal identity envelope.
        val maxItems: Int = 128,
        val forbiddenScore: Double = -1_000_000.0,
        val dummyScore: Double = 0.0,
        val tieBreakEpsilon: Double = 1e-9,
    ) {
        init {
            require(maxItems >= 1)
            require(forbiddenScore.isFinite() && forbiddenScore < -1.0)
            require(dummyScore.isFinite())
            require(tieBreakEpsilon > 0.0)
        }
    }

    data class Candidate(
        val leftIndex: Int,
        val rightId: Long,
        val score: Double,
    ) {
        init {
            require(leftIndex >= 0)
            require(rightId >= 0L)
            require(score.isFinite())
        }
    }

    data class Match(
        val leftIndex: Int,
        val rightId: Long,
        val score: Double,
    )

    data class Result(
        val matches: List<Match>,
        val unmatchedLeft: List<Int>,
        val unmatchedRight: List<Long>,
        val objectiveScore: Double,
        val matrixSize: Int,
        val boundedInput: Boolean,
    )

    fun assign(
        leftCount: Int,
        rightIds: List<Long>,
        candidates: List<Candidate>,
        minimumAcceptedScore: Double = Double.NEGATIVE_INFINITY,
    ): Result {
        require(leftCount >= 0)
        require(rightIds.distinct().size == rightIds.size)
        require(minimumAcceptedScore.isFinite() || minimumAcceptedScore == Double.NEGATIVE_INFINITY)

        if (leftCount == 0 || rightIds.isEmpty()) {
            return Result(
                matches = emptyList(),
                unmatchedLeft = (0 until leftCount).toList(),
                unmatchedRight = rightIds.distinct().sorted(),
                objectiveScore = 0.0,
                matrixSize = 0,
                boundedInput = false,
            )
        }

        val boundedLeftCount = leftCount.coerceAtMost(config.maxItems)
        val boundedRights = rightIds.take(config.maxItems)
        val boundedInput =
            boundedLeftCount != leftCount ||
                boundedRights.size != rightIds.size

        val rightIndexById = boundedRights
            .withIndex()
            .associate { it.value to it.index }
        val candidateScoreByPair =
            HashMap<Pair<Int, Long>, Double>(
                candidates.size.coerceAtMost(config.maxItems * config.maxItems),
            )
        candidates.forEach { candidate ->
            if (candidate.leftIndex !in 0 until boundedLeftCount) return@forEach
            if (candidate.rightId !in rightIndexById) return@forEach
            val key = candidate.leftIndex to candidate.rightId
            candidateScoreByPair[key] =
                max(
                    candidateScoreByPair[key] ?: Double.NEGATIVE_INFINITY,
                    candidate.score,
                )
        }

        val n = max(boundedLeftCount, boundedRights.size)

        val matrix = Array(n) { DoubleArray(n) { config.dummyScore } }

        /*
         * Real-left to real-right cells start forbidden. Candidate scores then replace them.
         * Dummy rows/columns retain dummyScore, letting the optimizer reject weak matches instead
         * of being forced into a bad identity assignment.
         */
        for (candidate in candidates) {
            if (candidate.leftIndex !in 0 until boundedLeftCount) continue
            val rightIndex = rightIndexById[candidate.rightId] ?: continue
            val score = candidate.score + config.tieBreakEpsilon *
                (
                    (boundedRights.size - rightIndex).toDouble() * 1e-3 +
                        (boundedLeftCount - candidate.leftIndex).toDouble() * 1e-6
                    )
            matrix[candidate.leftIndex][rightIndex] =
                max(
                    matrix[candidate.leftIndex][rightIndex],
                    score,
                )
        }

        /*
         * A standard Hungarian minimization implementation is applied to negative weights.
         * The square matrix includes explicit dummy rows and columns.
         */
        val assignment = hungarian(
            matrix = matrix,
            forbiddenScore = config.forbiddenScore,
        )

        val accepted = ArrayList<Match>()
        val usedRight = HashSet<Long>()
        var objective = 0.0

        for (left in 0 until boundedLeftCount) {
            val right = assignment[left]
            if (right !in boundedRights.indices) continue
            val rightId = boundedRights[right]
            val score = matrix[left][right]

            /*
             * Reject dummy/forbidden outcomes and preserve the caller's semantic threshold.
             * The epsilon is removed from the externally visible score.
             */
            val raw =
                candidateScoreByPair[left to rightId]
                    ?: Double.NEGATIVE_INFINITY

            if (
                raw.isFinite() &&
                raw >= minimumAcceptedScore &&
                raw > config.dummyScore
            ) {
                accepted += Match(
                    leftIndex = left,
                    rightId = rightId,
                    score = raw,
                )
                usedRight += rightId
                objective += raw
            }
        }

        val matchedLeft = accepted.map { it.leftIndex }.toSet()
        val unmatchedLeft =
            (0 until boundedLeftCount)
                .filter { it !in matchedLeft }

        val unmatchedRight =
            boundedRights.filter { it !in usedRight }

        return Result(
            matches = accepted.sortedBy { it.leftIndex },
            unmatchedLeft = unmatchedLeft,
            unmatchedRight = unmatchedRight,
            objectiveScore = objective,
            matrixSize = n,
            boundedInput = boundedInput,
        )
    }

    /**
     * Hungarian algorithm for maximum-weight assignment through cost = -weight.
     *
     * Potentials use one-based indexing, which avoids special cases for zero rows/columns.
     */
    private fun hungarian(
        matrix: Array<DoubleArray>,
        forbiddenScore: Double,
    ): IntArray {
        val n = matrix.size
        if (n == 0) return IntArray(0)

        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = DoubleArray(n + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(n + 1)

            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) continue

                    val weight = matrix[i0 - 1][j - 1]
                    val cost =
                        if (weight <= forbiddenScore) {
                            1e12
                        } else {
                            -weight
                        }

                    val cur = cost - u[i0] - v[j]
                    if (cur < minv[j]) {
                        minv[j] = cur
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

        val rowToColumn = IntArray(n) { -1 }
        for (j in 1..n) {
            if (p[j] != 0) {
                rowToColumn[p[j] - 1] = j - 1
            }
        }
        return rowToColumn
    }
}
