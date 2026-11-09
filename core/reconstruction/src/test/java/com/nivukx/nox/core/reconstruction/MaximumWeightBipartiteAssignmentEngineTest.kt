package com.nivukx.nox.core.reconstruction

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MaximumWeightBipartiteAssignmentEngineTest {

    @Test
    fun optimizerFindsGlobalMaximumInsteadOfGreedyLocalChoice() {
        val engine = MaximumWeightBipartiteAssignmentEngine()

        val result = engine.assign(
            leftCount = 2,
            rightIds = listOf(10L, 20L),
            candidates = listOf(
                MaximumWeightBipartiteAssignmentEngine.Candidate(0, 10L, 0.91),
                MaximumWeightBipartiteAssignmentEngine.Candidate(0, 20L, 0.89),
                MaximumWeightBipartiteAssignmentEngine.Candidate(1, 10L, 0.90),
            ),
            minimumAcceptedScore = 0.50,
        )

        assertEquals(
            listOf(
                MaximumWeightBipartiteAssignmentEngine.Match(0, 20L, 0.89),
                MaximumWeightBipartiteAssignmentEngine.Match(1, 10L, 0.90),
            ),
            result.matches,
        )
        assertEquals(1.79, result.objectiveScore, 1e-9)
    }

    @Test
    fun weakCandidateCanRemainUnmatchedThroughDummyAssignment() {
        val engine = MaximumWeightBipartiteAssignmentEngine()

        val result = engine.assign(
            leftCount = 2,
            rightIds = listOf(10L, 20L),
            candidates = listOf(
                MaximumWeightBipartiteAssignmentEngine.Candidate(0, 10L, 0.91),
                MaximumWeightBipartiteAssignmentEngine.Candidate(1, 20L, 0.30),
            ),
            minimumAcceptedScore = 0.50,
        )

        assertEquals(
            listOf(
                MaximumWeightBipartiteAssignmentEngine.Match(0, 10L, 0.91),
            ),
            result.matches,
        )
        assertTrue(1 in result.unmatchedLeft)
        assertTrue(20L in result.unmatchedRight)
    }

    @Test
    fun assignmentIsDeterministicForTies() {
        val engine = MaximumWeightBipartiteAssignmentEngine()

        val candidates = listOf(
            MaximumWeightBipartiteAssignmentEngine.Candidate(0, 10L, 0.80),
            MaximumWeightBipartiteAssignmentEngine.Candidate(0, 20L, 0.80),
            MaximumWeightBipartiteAssignmentEngine.Candidate(1, 10L, 0.80),
            MaximumWeightBipartiteAssignmentEngine.Candidate(1, 20L, 0.80),
        )

        val first = engine.assign(2, listOf(10L, 20L), candidates)
        val second = engine.assign(2, listOf(10L, 20L), candidates)

        assertEquals(first.matches, second.matches)
        assertEquals(first.objectiveScore, second.objectiveScore, 1e-9)
    }

    @Test
    fun inputIsBoundedRatherThanGrowingWithoutLimit() {
        val engine = MaximumWeightBipartiteAssignmentEngine(
            MaximumWeightBipartiteAssignmentEngine.Config(maxItems = 2),
        )

        val result = engine.assign(
            leftCount = 4,
            rightIds = listOf(1L, 2L, 3L),
            candidates = emptyList(),
        )

        assertTrue(result.boundedInput)
        assertEquals(2, result.matrixSize)
    }
}
