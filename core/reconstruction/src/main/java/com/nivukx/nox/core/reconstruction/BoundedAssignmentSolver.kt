package com.nivukx.nox.core.reconstruction

import kotlin.math.max

/**
 * Bounded global assignment solver for multi-view correspondences.
 *
 * NoX often has <= 2 hands, but the implementation supports up to four groups and four candidates.
 * Greedy matching can swap identities when hands cross. This solver evaluates the complete bounded
 * assignment space with an explicit unmatched penalty, so identity switches are minimized without
 * introducing an unbounded combinatorial algorithm into a mobile real-time path.
 */
class BoundedAssignmentSolver(
    private val maxItems: Int = 4,
) {

    data class Match(
        val itemIndex: Int,
        val candidateIndex: Int,
        val cost: Double,
    )

    data class Result(
        val matches: List<Match>,
        val totalCost: Double,
        val unmatchedItems: Int,
        val unmatchedCandidates: Int,
    )

    fun solve(
        costs: Array<DoubleArray>,
        unmatchedItemPenalty: Double,
        maxAcceptedCost: Double,
    ): Result {
        require(costs.size <= maxItems) {
            "Too many assignment items"
        }

        val candidateCount =
            costs.maxOfOrNull { it.size } ?: 0

        require(candidateCount <= maxItems) {
            "Too many assignment candidates"
        }

        if (costs.isEmpty() || candidateCount == 0) {
            return Result(
                matches = emptyList(),
                totalCost =
                    unmatchedItemPenalty *
                        costs.size,
                unmatchedItems = costs.size,
                unmatchedCandidates = candidateCount,
            )
        }

        val padded = Array(costs.size) { row ->
            DoubleArray(candidateCount) { column ->
                costs[row]
                    .getOrNull(column)
                    ?.takeIf(Double::isFinite)
                    ?: Double.POSITIVE_INFINITY
            }
        }

        var bestCost = Double.POSITIVE_INFINITY
        var best = IntArray(costs.size) { -1 }
        val used = BooleanArray(candidateCount)
        val assignment = IntArray(costs.size) { -1 }

        fun search(
            row: Int,
            runningCost: Double,
        ) {
            if (runningCost >= bestCost) {
                return
            }

            if (row >= padded.size) {
                bestCost = runningCost
                best = assignment.copyOf()
                return
            }

            search(
                row = row + 1,
                runningCost =
                    runningCost +
                        unmatchedItemPenalty,
            )

            for (column in 0 until candidateCount) {
                if (used[column]) continue

                val cost = padded[row][column]
                if (!cost.isFinite()) continue
                if (cost > maxAcceptedCost) continue

                used[column] = true
                assignment[row] = column

                search(
                    row = row + 1,
                    runningCost = runningCost + cost,
                )

                assignment[row] = -1
                used[column] = false
            }
        }

        search(
            row = 0,
            runningCost = 0.0,
        )

        val matches = best.indices.mapNotNull { itemIndex ->
            val candidateIndex = best[itemIndex]
            if (candidateIndex < 0) {
                null
            } else {
                Match(
                    itemIndex = itemIndex,
                    candidateIndex = candidateIndex,
                    cost = padded[itemIndex][candidateIndex],
                )
            }
        }

        return Result(
            matches = matches,
            totalCost = bestCost,
            unmatchedItems =
                costs.size -
                    matches.size,
            unmatchedCandidates =
                max(
                    0,
                    candidateCount -
                        matches.size,
                ),
        )
    }
}
