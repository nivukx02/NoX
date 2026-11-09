package com.nivukx.nox.core.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BoundedAssignmentSolverTest {

    @Test
    fun choosesGlobalMinimumInsteadOfGreedyLocalMinimum() {
        val solver = BoundedAssignmentSolver()

        val result =
            solver.solve(
                costs = arrayOf(
                    doubleArrayOf(1.0, 2.0),
                    doubleArrayOf(1.1, 9.0),
                ),
                unmatchedItemPenalty = 20.0,
                maxAcceptedCost = 10.0,
            )

        assertEquals(1, result.matches[0].candidateIndex)
        assertEquals(0, result.matches[1].candidateIndex)
        assertEquals(3.1, result.totalCost, 0.000001)
    }

    @Test
    fun rejectsExpensiveMatchesWhenUnmatchedIsCheaper() {
        val solver = BoundedAssignmentSolver()

        val result =
            solver.solve(
                costs = arrayOf(
                    doubleArrayOf(25.0),
                ),
                unmatchedItemPenalty = 5.0,
                maxAcceptedCost = 30.0,
            )

        assertTrue(result.matches.isEmpty())
        assertEquals(1, result.unmatchedItems)
        assertEquals(5.0, result.totalCost, 0.000001)
    }

    @Test
    fun handlesEmptyInputsDeterministically() {
        val solver = BoundedAssignmentSolver()

        val result =
            solver.solve(
                costs = emptyArray(),
                unmatchedItemPenalty = 4.0,
                maxAcceptedCost = 8.0,
            )

        assertTrue(result.matches.isEmpty())
        assertEquals(0.0, result.totalCost, 0.000001)
    }
}
