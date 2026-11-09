package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.ReconstructionJoint
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiEvidence3DReconstructionEngineTest {

    @Test
    fun consensusSuppressesSeparatedOutlier() {
        val engine = MultiEvidence3DReconstructionEngine()
        val result = engine.fuse(
            timestampNanos = 1_000_000_000L,
            primary = joint(
                x = 1.00,
                y = 2.00,
                z = 3.00,
                confidence = 0.92,
                uncertainty = 0.025,
            ),
            candidates = listOf(
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(1.01, 2.01, 3.01, 0.86, 0.030),
                    source = MultiEvidence3DReconstructionEngine.Source.MULTI_VIEW,
                    correlationGroup = "geometry",
                ),
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(1.00, 1.99, 3.02, 0.80, 0.040),
                    source = MultiEvidence3DReconstructionEngine.Source.TEMPORAL_PRIOR,
                    correlationGroup = "temporal",
                ),
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(4.20, 0.20, -2.0, 0.99, 0.005),
                    source = MultiEvidence3DReconstructionEngine.Source.MONOCULAR_DEPTH,
                    correlationGroup = "depth",
                ),
            ),
        )

        assertTrue(result.accepted)
        assertTrue(result.position.distanceTo(Vec3(1.0, 2.0, 3.0)) < 0.10)
        assertTrue(result.outlierCount >= 1)
        assertTrue(result.sourceDiversity >= 0.50)
    }

    @Test
    fun highDisagreementReducesConfidenceAndRaisesUncertainty() {
        val engine = MultiEvidence3DReconstructionEngine()
        val result = engine.fuse(
            timestampNanos = 2_000_000_000L,
            primary = joint(0.0, 0.0, 2.0, 0.72, 0.06),
            candidates = listOf(
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(0.30, 0.0, 2.0, 0.70, 0.08),
                    source = MultiEvidence3DReconstructionEngine.Source.MULTI_VIEW,
                    correlationGroup = "geometry",
                ),
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(-0.25, 0.0, 2.0, 0.68, 0.07),
                    source = MultiEvidence3DReconstructionEngine.Source.INERTIAL_PRIOR,
                    correlationGroup = "motion",
                ),
            ),
        )

        assertTrue(result.positionUncertaintyMeters > 0.06)
        assertTrue(result.confidence < 0.75)
        assertTrue(result.disagreementMeters > 0.15)
    }

    @Test
    fun primaryIsRetainedWhenNoIndependentEvidenceExists() {
        val engine = MultiEvidence3DReconstructionEngine()
        val primary = joint(0.2, 0.3, 1.1, 0.91, 0.02)

        val result = engine.fuse(
            timestampNanos = 3_000_000_000L,
            primary = primary,
            candidates = emptyList(),
        )

        assertEquals(primary.position, result.position)
        assertEquals(primary.confidence, result.confidence, 1e-9)
        assertTrue(result.usedPrimaryOnly)
    }


    @Test
    fun temporal_prior_path_provides_3d_temporal_evidence_without_dropping_primary() {
        val engine = MultiEvidence3DReconstructionEngine()
        val first = engine.fuseWithTemporalPrior(
            timestampNanos = 10_000_000_000L,
            primary = joint(
                x = 0.0,
                y = 0.0,
                z = 2.0,
                confidence = 0.95,
                uncertainty = 0.02,
            ),
            candidates = listOf(
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(0.01, 0.0, 2.0, 0.90, 0.025),
                    source =
                        MultiEvidence3DReconstructionEngine.Source.MULTI_VIEW,
                    correlationGroup = "camera-a",
                ),
            ),
        )
        val second = engine.fuseWithTemporalPrior(
            timestampNanos = 10_016_666_667L,
            primary = joint(
                x = 0.02,
                y = 0.0,
                z = 2.0,
                confidence = 0.95,
                uncertainty = 0.02,
            ),
            candidates = listOf(
                MultiEvidence3DReconstructionEngine.Candidate(
                    joint = joint(0.025, 0.0, 2.0, 0.92, 0.02),
                    source =
                        MultiEvidence3DReconstructionEngine.Source.MULTI_VIEW,
                    correlationGroup = "camera-b",
                ),
            ),
        )

        assertTrue(first.result.accepted)
        assertTrue(second.result.accepted)
        assertTrue(first.temporalPriorUsed)
        assertTrue(second.temporalPriorUsed)
        assertTrue(second.temporalSummary.activeTracks >= 1)
        assertTrue(
            second.result.position.distanceTo(
                Vec3(0.02, 0.0, 2.0),
            ) < 0.10,
        )
        engine.close()
    }

    @Test
    fun temporal_prior_path_recovers_after_timestamp_regression() {
        val engine = MultiEvidence3DReconstructionEngine()
        engine.fuseWithTemporalPrior(
            timestampNanos = 20_000_000_000L,
            primary = joint(1.0, 0.0, 2.0, 0.95, 0.02),
            candidates = emptyList(),
        )

        val recovered = engine.fuseWithTemporalPrior(
            timestampNanos = 19_000_000_000L,
            primary = joint(0.98, 0.0, 2.0, 0.94, 0.02),
            candidates = emptyList(),
        )

        assertTrue(recovered.result.accepted)
        assertTrue(recovered.temporalSummary.activeTracks >= 1)
        engine.close()
    }

    private fun joint(
        x: Double,
        y: Double,
        z: Double,
        confidence: Double,
        uncertainty: Double,
    ): ReconstructionJoint =
        ReconstructionJoint(
            landmarkId = 7,
            position = Vec3(x, y, z),
            velocity = Vec3(0.0, 0.0, 0.0),
            confidence = confidence,
            reprojectionErrorPx = 1.5,
            observationCount = 4,
            positionUncertaintyMeters = uncertainty,
        )
}
