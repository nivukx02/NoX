package com.nivukx.nox.core.ai

import kotlin.math.max

/**
 * Bounded Hungarian maximum-weight assignment for local multi-person tracking.
 *
 * The tracker supplies only geometrically gated candidate edges. Dummy assignments remain at zero,
 * allowing detections and tracks to stay unmatched rather than forcing identity teleportation.
 */
class BoundedGlobalAssignmentEngine(
    private val maxItems: Int = 64,
    private val forbiddenWeight: Double = -1_000_000.0,
) {
    data class Edge(
        val row: Int,
        val column: Int,
        val score: Double,
    )

    data class Match(
        val row: Int,
        val column: Int,
        val score: Double,
    )

    data class Result(
        val matches: List<Match>,
        val bounded: Boolean,
        val matrixSize: Int,
    )

    init {
        require(maxItems >= 1)
        require(forbiddenWeight < -1.0)
    }

    fun assign(
        rowCount: Int,
        columnCount: Int,
        edges: List<Edge>,
        minimumScore: Double,
    ): Result {
        require(rowCount >= 0)
        require(columnCount >= 0)
        require(minimumScore.isFinite())

        if (rowCount == 0 || columnCount == 0) {
            return Result(
                matches = emptyList(),
                bounded = false,
                matrixSize = 0,
            )
        }

        val rows = rowCount.coerceAtMost(maxItems)
        val columns = columnCount.coerceAtMost(maxItems)
        val bounded =
            rows != rowCount ||
                columns != columnCount
        val size = max(rows, columns)

        val matrix = Array(size) {
            DoubleArray(size)
        }
        val raw = HashMap<Long, Double>()

        edges.forEach { edge ->
            if (
                edge.row !in 0 until rows ||
                edge.column !in 0 until columns ||
                !edge.score.isFinite() ||
                edge.score < minimumScore
            ) {
                return@forEach
            }
            val adjusted =
                edge.score +
                    deterministicTieBreak(
                        edge.row,
                        edge.column,
                        rows,
                        columns,
                    )
            if (
                adjusted >
                    matrix[edge.row][edge.column]
            ) {
                matrix[edge.row][edge.column] = adjusted
                raw[key(edge.row, edge.column)] =
                    edge.score
            }
        }

        /*
         * Cells without candidate edges must be forbidden rather than zero. Otherwise Hungarian
         * can consume a real row/column with a fake zero edge and distort the maximum assignment
         * when positive candidates are sparse.
         */
        for (row in 0 until rows) {
            for (column in 0 until columns) {
                if (
                    matrix[row][column] == 0.0 &&
                    raw[key(row, column)] == null
                ) {
                    matrix[row][column] =
                        forbiddenWeight
                }
            }
        }

        val assignment = hungarianMax(matrix)
        val matches = buildList {
            for (row in 0 until rows) {
                val column =
                    assignment.getOrNull(row)
                        ?: continue
                if (column !in 0 until columns) continue
                val score =
                    raw[key(row, column)]
                        ?: continue
                if (score >= minimumScore) {
                    add(
                        Match(
                            row = row,
                            column = column,
                            score = score,
                        ),
                    )
                }
            }
        }.sortedBy { it.row }

        return Result(
            matches = matches,
            bounded = bounded,
            matrixSize = size,
        )
    }

    private fun hungarianMax(
        weights: Array<DoubleArray>,
    ): IntArray {
        val n = weights.size
        if (n == 0) return IntArray(0)

        val u = DoubleArray(n + 1)
        val v = DoubleArray(n + 1)
        val p = IntArray(n + 1)
        val way = IntArray(n + 1)

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
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) continue
                    val weight =
                        weights[i0 - 1][j - 1]
                    val cost =
                        if (
                            weight <=
                                forbiddenWeight / 2.0
                        ) {
                            1e12
                        } else {
                            -weight
                        }
                    val cur =
                        cost -
                            u[i0] -
                            v[j]
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

        val result = IntArray(n) { -1 }
        for (column in 1..n) {
            if (p[column] != 0) {
                result[p[column] - 1] = column - 1
            }
        }
        return result
    }

    private fun deterministicTieBreak(
        row: Int,
        column: Int,
        rowCount: Int,
        columnCount: Int,
    ): Double =
        (
            (rowCount - row).toDouble() * 1e-9 +
                (columnCount - column).toDouble() * 1e-12
        )

    private fun key(
        row: Int,
        column: Int,
    ): Long =
        (
            row.toLong() shl 32
        ) xor (
            column.toLong() and 0xffffffffL
        )
}
