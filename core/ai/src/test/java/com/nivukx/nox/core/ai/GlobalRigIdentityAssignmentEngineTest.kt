package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import com.nivukx.nox.core.model.Vec2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GlobalRigIdentityAssignmentEngineTest {

    @Test
    fun oneToOneAssignmentPreservesCrossingIdentities() {
        val engine =
            GlobalRigIdentityAssignmentEngine(
                GlobalRigIdentityAssignmentEngine.Config(
                    minimumMatchScore = 0.45,
                ),
            )

        val previous =
            listOf(
                track(
                    rigTrackId = 10L,
                    center = Vec2(0.42, 0.0),
                    velocity = Vec2(0.55, 0.0),
                    localTrackIds = mapOf("a" to 1L, "b" to 11L),
                ),
                track(
                    rigTrackId = 20L,
                    center = Vec2(0.58, 0.0),
                    velocity = Vec2(-0.55, 0.0),
                    localTrackIds = mapOf("a" to 2L, "b" to 12L),
                ),
            )

        val current =
            listOf(
                track(
                    rigTrackId = 0L,
                    center = Vec2(0.46, 0.0),
                    velocity = Vec2(0.55, 0.0),
                    localTrackIds = mapOf("a" to 102L, "b" to 112L),
                ),
                track(
                    rigTrackId = 0L,
                    center = Vec2(0.54, 0.0),
                    velocity = Vec2(-0.55, 0.0),
                    localTrackIds = mapOf("a" to 101L, "b" to 111L),
                ),
            )

        val result =
            engine.assign(
                current = current,
                previous = previous,
            )

        assertEquals(2, result.assignments.size)
        val first =
            result.assignments.first {
                it.currentIndex == 0
            }
        val second =
            result.assignments.first {
                it.currentIndex == 1
            }

        assertEquals(10L, first.rigTrackId)
        assertEquals(20L, second.rigTrackId)
        assertTrue(result.globalConfidence > 0.55)
    }

    @Test
    fun onePreviousIdentityCannotBeAssignedTwice() {
        val engine =
            GlobalRigIdentityAssignmentEngine()

        val previous =
            listOf(
                track(
                    rigTrackId = 7L,
                    center = Vec2(0.50, 0.0),
                    velocity = Vec2(0.0, 0.0),
                    localTrackIds = emptyMap(),
                ),
            )

        val current =
            listOf(
                track(
                    rigTrackId = 0L,
                    center = Vec2(0.50, 0.0),
                    velocity = Vec2(0.0, 0.0),
                    localTrackIds = emptyMap(),
                ),
                track(
                    rigTrackId = 0L,
                    center = Vec2(0.51, 0.0),
                    velocity = Vec2(0.0, 0.0),
                    localTrackIds = emptyMap(),
                ),
            )

        val result =
            engine.assign(
                current = current,
                previous = previous,
            )

        assertEquals(1, result.assignments.size)
        assertEquals(1, result.unmatchedCurrentIndices.size)
    }

    @Test
    fun semanticTypeMismatchIsForbidden() {
        val engine =
            GlobalRigIdentityAssignmentEngine()

        val result =
            engine.assign(
                current =
                    listOf(
                        track(
                            rigTrackId = 0L,
                            center = Vec2(0.5, 0.0),
                            velocity = Vec2(0.0, 0.0),
                            localTrackIds = emptyMap(),
                            subjectType = SubjectType.ANIMAL,
                        ),
                    ),
                previous =
                    listOf(
                        track(
                            rigTrackId = 9L,
                            center = Vec2(0.5, 0.0),
                            velocity = Vec2(0.0, 0.0),
                            localTrackIds = emptyMap(),
                            subjectType = SubjectType.HUMAN,
                        ),
                    ),
            )

        assertTrue(result.assignments.isEmpty())
        assertEquals(listOf(0), result.unmatchedCurrentIndices)
    }

    private fun track(
        rigTrackId: Long,
        center: Vec2,
        velocity: Vec2,
        localTrackIds: Map<String, Long>,
        subjectType: SubjectType = SubjectType.HUMAN,
    ) =
        GlobalRigIdentityAssignmentEngine.Track(
            rigTrackId = rigTrackId,
            subjectType = subjectType,
            center = center,
            velocity = velocity,
            confidence = 0.94,
            observationCount = localTrackIds.size.coerceAtLeast(2),
            localTrackIds = localTrackIds,
        )
    @Test
    fun truncation_reports_unmatched_current_components_and_reduces_confidence() {
        val engine = GlobalRigIdentityAssignmentEngine(
            GlobalRigIdentityAssignmentEngine.Config(
                maximumTracks = 64,
            ),
        )

        val previous = listOf(
            track(
                rigTrackId = 1L,
                center = Vec2(0.5, 0.0),
                velocity = Vec2(0.0, 0.0),
                localTrackIds = mapOf("a" to 1L),
            ),
        )

        val current =
            (0 until 66).map { index ->
                track(
                    rigTrackId = 0L,
                    center = Vec2(
                        0.45 + index * 0.001,
                        0.0,
                    ),
                    velocity = Vec2(0.0, 0.0),
                    localTrackIds = emptyMap(),
                )
            }

        val result = engine.assign(
            current = current,
            previous = previous,
        )

        assertTrue(result.boundedInput)
        assertTrue(result.unmatchedCurrentIndices.contains(64))
        assertTrue(result.unmatchedCurrentIndices.contains(65))
        assertTrue(result.globalConfidence < 0.90)
    }
}
