package com.nivukx.nox.core.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TemporalSubjectIdentityContinuityEngineTest {

    @Test
    fun retainsLockedTrack_whenACompetingTrackAppearsBriefly() {
        val engine = TemporalSubjectIdentityContinuityEngine(
            TemporalSubjectIdentityContinuityEngine.Config(
                switchConfidenceMargin = 0.24,
                holdAmbiguityThreshold = 0.50,
                maximumRecoveryGapNanos = 300_000_000L,
            ),
        )

        val first = engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 0L,
                candidateTrackId = 10L,
                candidateConfidence = 0.94,
                subjectCount = 1,
                occlusionRisk = 0.05,
                temporalStability = 0.96,
                leftRightAmbiguity = 0.05,
                viewpointChallenge = 0.05,
            ),
        )
        assertEquals(10L, first.selectedTrackId)

        val competing = engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 33_000_000L,
                candidateTrackId = 20L,
                candidateConfidence = 0.93,
                subjectCount = 2,
                occlusionRisk = 0.42,
                temporalStability = 0.78,
                leftRightAmbiguity = 0.86,
                viewpointChallenge = 0.72,
            ),
        )

        assertEquals(10L, competing.selectedTrackId)
        assertEquals(
            TemporalSubjectIdentityContinuityEngine.Action.HOLD,
            competing.action,
        )
        assertTrue(competing.identitySwapRisk > 0.60)
    }

    @Test
    fun switchesAfterSustainedEvidence_whenPreviousTrackIsWeak() {
        val engine = TemporalSubjectIdentityContinuityEngine(
            TemporalSubjectIdentityContinuityEngine.Config(
                switchConfidenceMargin = 0.12,
                minimumSwitchEvidenceFrames = 2,
                holdAmbiguityThreshold = 0.80,
            ),
        )

        engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 0L,
                candidateTrackId = 3L,
                candidateConfidence = 0.48,
                subjectCount = 2,
                occlusionRisk = 0.70,
                temporalStability = 0.40,
                leftRightAmbiguity = 0.20,
                viewpointChallenge = 0.20,
            ),
        )

        val firstCandidate = engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 40_000_000L,
                candidateTrackId = 4L,
                candidateConfidence = 0.86,
                subjectCount = 2,
                occlusionRisk = 0.20,
                temporalStability = 0.88,
                leftRightAmbiguity = 0.08,
                viewpointChallenge = 0.10,
            ),
        )
        assertEquals(
            TemporalSubjectIdentityContinuityEngine.Action.HOLD,
            firstCandidate.action,
        )

        val switched = engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 80_000_000L,
                candidateTrackId = 4L,
                candidateConfidence = 0.88,
                subjectCount = 2,
                occlusionRisk = 0.16,
                temporalStability = 0.91,
                leftRightAmbiguity = 0.05,
                viewpointChallenge = 0.08,
            ),
        )

        assertEquals(4L, switched.selectedTrackId)
        assertEquals(
            TemporalSubjectIdentityContinuityEngine.Action.SWITCH,
            switched.action,
        )
    }

    @Test
    fun recoversLastTrack_duringShortOcclusion() {
        val engine = TemporalSubjectIdentityContinuityEngine()

        engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 1_000L,
                candidateTrackId = 7L,
                candidateConfidence = 0.90,
                subjectCount = 1,
                occlusionRisk = 0.05,
                temporalStability = 0.92,
                leftRightAmbiguity = 0.05,
                viewpointChallenge = 0.05,
            ),
        )

        val recovered = engine.evaluate(
            TemporalSubjectIdentityContinuityEngine.FrameEvidence(
                timestampNanos = 80_000_000L,
                candidateTrackId = null,
                candidateConfidence = 0.0,
                subjectCount = 0,
                occlusionRisk = 0.92,
                temporalStability = 0.62,
                leftRightAmbiguity = 0.10,
                viewpointChallenge = 0.18,
            ),
        )

        assertEquals(7L, recovered.selectedTrackId)
        assertEquals(
            TemporalSubjectIdentityContinuityEngine.Action.RECOVER,
            recovered.action,
        )
        assertTrue(recovered.predictedContinuation)
    }
}
