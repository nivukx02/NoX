package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class OcclusionContactEvidenceEngineTest {

    private fun joint(
        id: Int,
        x: Double,
        y: Double,
        confidence: Double = 1.0,
        visible: Boolean = true,
    ) = OcclusionContactEvidenceEngine.JointObservation(
        landmarkId = id,
        x = x,
        y = y,
        confidence = confidence,
        visible = visible,
    )

    @Test
    fun visible_landmarks_receive_full_visibility_weight() {
        val engine = OcclusionContactEvidenceEngine()
        val result = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 1_000_000_000L,
                actorId = 1L,
                landmarks = listOf(
                    joint(0, 0.10, 0.20),
                    joint(1, 0.20, 0.30),
                    joint(2, 0.30, 0.40),
                ),
            ),
        )

        assertEquals(1.0, result.meanVisibility, 1e-9)
        assertTrue(result.criticalOcclusionRatio == 0.0)
        assertTrue(result.captureTrust > 0.80)
    }

    @Test
    fun persistent_occlusion_inflates_uncertainty_and_limits_prediction() {
        val engine = OcclusionContactEvidenceEngine(
            OcclusionContactEvidenceEngine.Config(
                maximumPredictionMillis = 80.0,
                hardOcclusionRatio = 0.55,
            ),
        )

        engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 1_000_000_000L,
                actorId = 1L,
                landmarks = listOf(
                    joint(0, 0.10, 0.20),
                    joint(1, 0.20, 0.30),
                    joint(2, 0.30, 0.40),
                    joint(3, 0.40, 0.50),
                ),
            ),
        )

        val result = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 1_040_000_000L,
                actorId = 1L,
                landmarks = listOf(
                    joint(0, 0.10, 0.20),
                    joint(1, 0.20, 0.30, visible = false, confidence = 0.20),
                    joint(2, 0.30, 0.40, visible = false, confidence = 0.12),
                    joint(3, 0.40, 0.50, visible = false, confidence = 0.08),
                ),
            ),
        )

        assertTrue(result.criticalOcclusionRatio >= 0.50)
        assertTrue(result.predictionBudgetRemainingMillis < 80.0)
        assertTrue(result.visibilityUncertaintyMultiplier > 1.0)
        assertTrue(
            result.warnings.any { it.contains("persistent-occlusion") },
        )
    }

    @Test
    fun contact_clusters_preserve_conflicting_limbs_without_forcing_identity_merge() {
        val engine = OcclusionContactEvidenceEngine()

        val result = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 2_000_000_000L,
                actorId = 7L,
                landmarks = listOf(
                    joint(10, 0.20, 0.50),
                    joint(11, 0.24, 0.52),
                    joint(12, 0.50, 0.50),
                    joint(13, 0.54, 0.52),
                ),
                contacts = listOf(
                    OcclusionContactEvidenceEngine.ContactObservation(
                        firstLandmarkId = 10,
                        secondLandmarkId = 11,
                        distanceNormalized = 0.04,
                        confidence = 0.95,
                    ),
                    OcclusionContactEvidenceEngine.ContactObservation(
                        firstLandmarkId = 12,
                        secondLandmarkId = 13,
                        distanceNormalized = 0.04,
                        confidence = 0.90,
                    ),
                ),
            ),
        )

        assertEquals(2, result.contactClusterCount)
        assertTrue(result.contactPreservationScore > 0.80)
        assertTrue(result.identityMergeSuppression > 0.70)
    }

    @Test
    fun timestamp_regression_does_not_rewrite_actor_evidence() {
        val engine = OcclusionContactEvidenceEngine()

        val first = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 3_000_000_000L,
                actorId = 9L,
                landmarks = listOf(joint(0, 0.1, 0.1)),
            ),
        )

        val stale = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 2_000_000_000L,
                actorId = 9L,
                landmarks = listOf(
                    joint(
                        0,
                        0.9,
                        0.9,
                        confidence = 0.01,
                        visible = false,
                    ),
                ),
            ),
        )

        assertEquals(first.timestampNanos, stale.timestampNanos)
        assertEquals(first.meanVisibility, stale.meanVisibility, 1e-9)
        assertTrue(stale.warnings.any { it.contains("timestamp-reversal") })
    }

    @Test
    fun capacity_is_bounded_for_large_landmark_sets() {
        val engine = OcclusionContactEvidenceEngine(
            OcclusionContactEvidenceEngine.Config(
                maximumLandmarks = 32,
                maximumContacts = 16,
            ),
        )

        val result = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 4_000_000_000L,
                actorId = 1L,
                landmarks = (0 until 128).map {
                    joint(
                        id = it,
                        x = it / 128.0,
                        y = 0.5,
                    )
                },
                contacts = (0 until 64).map {
                    OcclusionContactEvidenceEngine.ContactObservation(
                        firstLandmarkId = it,
                        secondLandmarkId = it + 1,
                        distanceNormalized = 0.05,
                        confidence = 1.0,
                    )
                },
            ),
        )

        assertTrue(result.processedLandmarkCount <= 32)
        assertTrue(result.processedContactCount <= 16)
    }

    @Test
    fun close_releases_state() {
        val engine = OcclusionContactEvidenceEngine()
        engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 5_000_000_000L,
                actorId = 1L,
                landmarks = listOf(joint(0, 0.1, 0.1)),
            ),
        )
        engine.close()

        assertEquals(0, engine.trackedActorCount())
    }

    @Test
    fun periodic_motion_and_momentum_increase_bounded_occlusion_recovery_prior() {
        val engine = OcclusionContactEvidenceEngine(
            OcclusionContactEvidenceEngine.Config(
                maximumPredictionMillis = 80.0,
            ),
        )

        val result = engine.evaluate(
            OcclusionContactEvidenceEngine.Input(
                timestampNanos = 6_000_000_000L,
                actorId = 22L,
                landmarks = listOf(
                    joint(0, 0.1, 0.2),
                    joint(1, 0.2, 0.3, confidence = 0.15, visible = false),
                    joint(2, 0.3, 0.4, confidence = 0.10, visible = false),
                ),
                periodicMotionScore = 0.90,
                motionFrequencyHz = 1.25,
                momentumConfidence = 0.88,
            ),
        )

        assertTrue(result.frequencyRecoveryPrior > 0.10)
        assertTrue(result.momentumRecoveryPrior > 0.10)
        assertTrue(result.predictionBudgetRemainingMillis <= 80.0)
        assertTrue(result.captureTrust in 0.0..1.0)
    }

    
    @Test
    fun physicalEvidenceAndIdentitySafetyRemainIndependentOfContact() {
        val engine = OcclusionContactEvidenceEngine()

        val result =
            engine.evaluate(
                OcclusionContactEvidenceEngine.Input(
                    timestampNanos = 1_000_000_000L,
                    actorId = 7L,
                    landmarks = listOf(
                        OcclusionContactEvidenceEngine.JointObservation(
                            landmarkId = 1,
                            x = 0.40,
                            y = 0.50,
                            confidence = 0.95,
                            crossViewSupport = 0.94,
                            depthConfidence = 0.96,
                            appearanceConsistency = 0.92,
                            motionBoundaryConfidence = 0.90,
                            temporalInnovationScore = 0.93,
                        ),
                        OcclusionContactEvidenceEngine.JointObservation(
                            landmarkId = 2,
                            x = 0.44,
                            y = 0.50,
                            confidence = 0.90,
                            crossViewSupport = 0.88,
                            depthConfidence = 0.91,
                            appearanceConsistency = 0.90,
                            motionBoundaryConfidence = 0.88,
                            temporalInnovationScore = 0.91,
                        ),
                    ),
                    contacts = listOf(
                        OcclusionContactEvidenceEngine.ContactObservation(
                            firstLandmarkId = 1,
                            secondLandmarkId = 2,
                            distanceNormalized = 0.03,
                            confidence = 0.94,
                        ),
                    ),
                ),
            )

        assertTrue(result.meanEvidenceQuality > 0.75)
        assertTrue(result.identitySafetyScore > 0.62)
        assertTrue(
            result.warnings.none {
                it == "identity-safety-evidence-degraded"
            },
        )
        assertTrue(
            result.landmarkEvidence.all {
                it.evidenceQuality > 0.70
            },
        )

        engine.close()
    }

}
