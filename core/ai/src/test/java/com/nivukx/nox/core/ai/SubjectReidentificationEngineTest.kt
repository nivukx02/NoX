package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.SubjectType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SubjectReidentificationEngineTest {

    private fun track(
        id: Long,
        x: Float,
        y: Float,
        confidence: Double = 0.92,
    ) =
        SubjectTrackManager.Track(
            id = id,
            subjectType = SubjectType.HUMAN,
            left = x - 0.10f,
            top = y - 0.20f,
            right = x + 0.10f,
            bottom = y + 0.20f,
            velocityX = 0.0f,
            velocityY = 0.0f,
            confidence = confidence,
            ageFrames = 5L,
            missedFrames = 0,
            lastTimestampNanos = 1_000_000L,
        )

    @Test
    fun stableIdentitySurvivesDetectorTrackIdChange() {
        val engine = SubjectReidentificationEngine()

        val first = engine.update(
            listOf(track(10L, 0.40f, 0.50f)),
            1_000_000L,
        )
        val stableId = first.identities.single().stableId

        val second = engine.update(
            listOf(track(77L, 0.402f, 0.502f)),
            1_033_333_333L,
        )

        assertEquals(1, second.identities.size)
        assertEquals(
            stableId,
            second.identities.single().stableId,
        )
        assertTrue(second.reassociatedCount >= 1)
    }
    @Test
    fun stale_timestamp_is_idempotent_and_does_not_mutate_identity_state() {
        val engine = SubjectReidentificationEngine()

        val first =
            engine.update(
                listOf(
                    track(
                        10L,
                        0.40f,
                        0.50f,
                    ),
                ),
                1_000_000_000L,
            )
        val before = first.identities.single()

        val stale =
            engine.update(
                listOf(
                    track(
                        99L,
                        0.90f,
                        0.90f,
                        confidence = 0.99,
                    ),
                ),
                900_000_000L,
            )

        assertEquals(first, stale)
        assertEquals(
            before.stableId,
            stale.identities.single().stableId,
        )
        assertEquals(
            before.currentTrackId,
            stale.identities.single().currentTrackId,
        )
        assertEquals(
            before.centerX,
            stale.identities.single().centerX,
            0.0,
        )
    }

    @Test
    fun global_assignment_preserves_velocity_consistent_identity_during_crossing() {
        val engine = SubjectReidentificationEngine(
            SubjectReidentificationEngine.Config(
                maximumMissedFrames = 6,
                minimumMatchScore = 0.32,
            ),
        )

        val first = engine.update(
            listOf(
                track(10L, 0.30f, 0.50f).copy(velocityX = 0.18f),
                track(20L, 0.70f, 0.50f).copy(velocityX = -0.18f),
            ),
            1_000_000_000L,
        )

        val leftActor =
            first.minBy { it.centerX }.stableId
        val rightActor =
            first.maxBy { it.centerX }.stableId

        val second = engine.update(
            listOf(
                track(31L, 0.58f, 0.50f).copy(velocityX = 0.18f),
                track(32L, 0.42f, 0.50f).copy(velocityX = -0.18f),
            ),
            1_033_333_333L,
        )

        assertEquals(2, second.identities.size)
        assertEquals(
            leftActor,
            second.single { it.currentTrackId == 31L }.stableId,
        )
        assertEquals(
            rightActor,
            second.single { it.currentTrackId == 32L }.stableId,
        )
    }
}
