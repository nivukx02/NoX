package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class BoundedGlobalAssignmentEngineTest {

    @Test
    fun findsGlobalAssignment() {
        val engine = BoundedGlobalAssignmentEngine()
        val result = engine.assign(
            rowCount = 2,
            columnCount = 2,
            edges = listOf(
                BoundedGlobalAssignmentEngine.Edge(0, 0, 0.91),
                BoundedGlobalAssignmentEngine.Edge(0, 1, 0.89),
                BoundedGlobalAssignmentEngine.Edge(1, 0, 0.90),
            ),
            minimumScore = 0.50,
        )

        assertEquals(
            listOf(
                BoundedGlobalAssignmentEngine.Match(0, 1, 0.89),
                BoundedGlobalAssignmentEngine.Match(1, 0, 0.90),
            ),
            result.matches,
        )
    }

    @Test
    fun weakEdgesRemainUnmatched() {
        val engine = BoundedGlobalAssignmentEngine()
        val result = engine.assign(
            rowCount = 2,
            columnCount = 2,
            edges = listOf(
                BoundedGlobalAssignmentEngine.Edge(0, 0, 0.80),
                BoundedGlobalAssignmentEngine.Edge(1, 1, 0.20),
            ),
            minimumScore = 0.50,
        )
        assertEquals(1, result.matches.size)
        assertTrue(result.matrixSize == 2)
    }

    @Test
    fun capacityIsBounded() {
        val engine =
            BoundedGlobalAssignmentEngine(maxItems = 3)
        val result = engine.assign(
            rowCount = 8,
            columnCount = 8,
            edges = emptyList(),
            minimumScore = 0.1,
        )
        assertTrue(result.bounded)
        assertEquals(3, result.matrixSize)
    }
}
