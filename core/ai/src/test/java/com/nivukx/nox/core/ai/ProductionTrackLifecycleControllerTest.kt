package com.nivukx.nox.core.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductionTrackLifecycleControllerTest {

    private val controller =
        ProductionTrackLifecycleController(
            ProductionTrackLifecycleController.Config(
                minimumConfirmationFrames = 2,
                maximumOcclusionNanos = 300_000_000L,
                recoveryWindowNanos = 700_000_000L,
                minimumConfidence = 0.35,
                recoveryConfidenceFloor = 0.30,
            ),
        )

    private fun evidence(
        id: Long,
        timestamp: Long,
        confidence: Double,
        missed: Int,
        age: Long,
        stability: Double = 0.90,
    ) =
        ProductionTrackLifecycleController.TrackEvidence(
            trackId = id,
            timestampNanos = timestamp,
            confidence = confidence,
            missedFrames = missed,
            ageFrames = age,
            temporalStability = stability,
            velocityMagnitude = 0.08,
        )

    @Test
    fun strongTrackTransitionsFromNewToConfirmed() {
        var result = controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 7L,
                    timestamp = 100L,
                    confidence = 0.92,
                    missed = 0,
                    age = 1L,
                ),
            ),
        )
        assertEquals(
            ProductionTrackLifecycleController.State.TENTATIVE,
            result.states[7L],
        )

        result = controller.update(
            timestampNanos = 200L,
            tracks = listOf(
                evidence(
                    id = 7L,
                    timestamp = 200L,
                    confidence = 0.91,
                    missed = 0,
                    age = 2L,
                ),
            ),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.CONFIRMED,
            result.states[7L],
        )
        assertTrue(result.continuityScore > 0.75)
    }

    @Test
    fun briefOcclusionPreservesTrackAndMarksPredictedState() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 3L,
                    timestamp = 100L,
                    confidence = 0.90,
                    missed = 0,
                    age = 4L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 180_000_000L,
            tracks = listOf(
                evidence(
                    id = 3L,
                    timestamp = 180_000_000L,
                    confidence = 0.68,
                    missed = 2,
                    age = 6L,
                    stability = 0.82,
                ),
            ),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.OCCLUDED,
            result.states[3L],
        )
        assertTrue(
            result.predictedTrackIds.contains(3L),
        )
    }

    @Test
    fun reacquisitionAfterOcclusionRequiresRecoveryEvidence() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 9L,
                    timestamp = 100L,
                    confidence = 0.91,
                    missed = 0,
                    age = 5L,
                ),
            ),
        )
        controller.update(
            timestampNanos = 150_000_000L,
            tracks = listOf(
                evidence(
                    id = 9L,
                    timestamp = 150_000_000L,
                    confidence = 0.61,
                    missed = 1,
                    age = 6L,
                ),
            ),
        )

        controller.update(
            timestampNanos = 200_000_000L,
            tracks = listOf(
                evidence(
                    id = 9L,
                    timestamp = 200_000_000L,
                    confidence = 0.89,
                    missed = 0,
                    age = 7L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 250_000_000L,
            tracks = listOf(
                evidence(
                    id = 9L,
                    timestamp = 250_000_000L,
                    confidence = 0.91,
                    missed = 0,
                    age = 8L,
                ),
            ),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.RECOVERING,
            result.states[9L],
        )
        assertTrue(result.recoveringTrackIds.contains(9L))
    }

    @Test
    fun staleTrackBecomesLostWithoutBeingInstantlyReused() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 12L,
                    timestamp = 100L,
                    confidence = 0.88,
                    missed = 0,
                    age = 6L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 900_000_000L,
            tracks = listOf(
                evidence(
                    id = 12L,
                    timestamp = 900_000_000L,
                    confidence = 0.38,
                    missed = 5,
                    age = 11L,
                ),
            ),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.LOST,
            result.states[12L],
        )
        assertTrue(result.warnings.any { it.contains("loss") })
    }

    @Test
    fun cachedSchedulerEvidenceDoesNotCountAsFreshObservation() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 33L,
                    timestamp = 100L,
                    confidence = 0.92,
                    missed = 0,
                    age = 4L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 150_000_000L,
            tracks = listOf(
                evidence(
                    id = 33L,
                    timestamp = 100L,
                    confidence = 0.92,
                    missed = 0,
                    age = 4L,
                ),
            ),
            evidenceFresh = false,
        )

        assertEquals(
            ProductionTrackLifecycleController.State.TENTATIVE,
            result.states[33L],
        )
        assertTrue(
            result.warnings.contains(
                "tracking-evidence-cached-for-scheduler-budget",
            ),
        )
        assertTrue(result.predictedTrackIds.isEmpty())
    }

    @Test
    fun emptyFrameDoesNotEraseRecentTrackImmediately() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 21L,
                    timestamp = 100L,
                    confidence = 0.90,
                    missed = 0,
                    age = 4L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 150_000_000L,
            tracks = emptyList(),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.OCCLUDED,
            result.states[21L],
        )
        assertTrue(result.predictedTrackIds.contains(21L))
    }

    @Test
    fun occlusionBudgetEndsPredictionBeforeRecoveryWindow() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 44L,
                    timestamp = 100L,
                    confidence = 0.93,
                    missed = 0,
                    age = 1L,
                ),
            ),
        )

        controller.update(
            timestampNanos = 200L,
            tracks = listOf(
                evidence(
                    id = 44L,
                    timestamp = 200L,
                    confidence = 0.92,
                    missed = 0,
                    age = 2L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 550_000_000L,
            tracks = emptyList(),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.LOST,
            result.states[44L],
        )
        assertFalse(result.predictedTrackIds.contains(44L))
    }

    @Test
    fun cachedSchedulerEvidenceNeverTurnsConfirmedTrackIntoOcclusion() {
        controller.update(
            timestampNanos = 100L,
            tracks = listOf(
                evidence(
                    id = 33L,
                    timestamp = 100L,
                    confidence = 0.94,
                    missed = 0,
                    age = 1L,
                ),
            ),
        )

        controller.update(
            timestampNanos = 200L,
            tracks = listOf(
                evidence(
                    id = 33L,
                    timestamp = 200L,
                    confidence = 0.93,
                    missed = 0,
                    age = 2L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 400_000_000L,
            tracks = emptyList(),
            evidenceFresh = false,
        )

        assertEquals(
            ProductionTrackLifecycleController.State.CONFIRMED,
            result.states[33L],
        )
        assertFalse(result.predictedTrackIds.contains(33L))
        assertTrue(
            result.warnings.contains(
                "tracking-evidence-cached-for-scheduler-budget",
            ),
        )
    }
    @Test
    fun continuityScoreReflectsObservedTemporalStability() {
        controller.update(
            timestampNanos = 1L,
            tracks = listOf(
                evidence(
                    id = 55L,
                    timestamp = 1L,
                    confidence = 0.90,
                    missed = 0,
                    age = 1L,
                    stability = 0.98,
                ),
            ),
        )
        val stableResult =
            controller.update(
                timestampNanos = 2L,
                tracks = listOf(
                    evidence(
                        id = 55L,
                        timestamp = 2L,
                        confidence = 0.90,
                        missed = 0,
                        age = 2L,
                        stability = 0.98,
                    ),
                ),
            )

        controller.reset()

        controller.update(
            timestampNanos = 1L,
            tracks = listOf(
                evidence(
                    id = 55L,
                    timestamp = 1L,
                    confidence = 0.90,
                    missed = 0,
                    age = 1L,
                    stability = 0.18,
                ),
            ),
        )
        val unstableResult =
            controller.update(
                timestampNanos = 2L,
                tracks = listOf(
                    evidence(
                        id = 55L,
                        timestamp = 2L,
                        confidence = 0.90,
                        missed = 0,
                        age = 2L,
                        stability = 0.18,
                    ),
                ),
            )

        assertTrue(
            stableResult.continuityScore >
                unstableResult.continuityScore + 0.15,
        )
    }


    @Test
    fun stale_track_evidence_cannot_advance_lifecycle_state() {
        controller.update(
            timestampNanos = 1_000L,
            tracks = listOf(
                evidence(
                    id = 77L,
                    timestamp = 1_000L,
                    confidence = 0.92,
                    missed = 0,
                    age = 1L,
                ),
            ),
        )
        controller.update(
            timestampNanos = 2_000L,
            tracks = listOf(
                evidence(
                    id = 77L,
                    timestamp = 2_000L,
                    confidence = 0.93,
                    missed = 0,
                    age = 2L,
                ),
            ),
        )

        val result = controller.update(
            timestampNanos = 3_000L,
            tracks = listOf(
                evidence(
                    id = 77L,
                    timestamp = 1_500L,
                    confidence = 0.95,
                    missed = 0,
                    age = 100L,
                ),
            ),
        )

        assertEquals(
            ProductionTrackLifecycleController.State.OCCLUDED,
            result.states[77L],
        )
        assertTrue(
            result.warnings.contains(
                "stale-track-evidence-rejected",
            ),
        )
    }

}
