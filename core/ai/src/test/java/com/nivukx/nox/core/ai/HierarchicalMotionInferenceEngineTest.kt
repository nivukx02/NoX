package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.EvidenceOrigin
import com.nivukx.nox.core.model.UncertaintyEnvelope
import com.nivukx.nox.core.model.Vec3
import com.nivukx.nox.core.model.WorldSpaceProposal
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HierarchicalMotionInferenceEngineTest {
    private fun proposal(time: Long, x: Double, confidence: Double = 0.9) = WorldSpaceProposal(
        actorId = 1, pointId = 2, timestampNanos = time, positionMeters = Vec3(x,0.0,0.0),
        uncertainty = UncertaintyEnvelope(1e-4,1e-4,1e-4,confidence=confidence),
        confidence = confidence, geometricSupport = 0.9, temporalSupport = 0.8, learnedSupport = 0.0,
        independentCameraCount = 3, viewDiversity = 0.8, origin = EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,
        provenance = emptyList(),
    )

    @Test
    fun repeatedMeasurementsProduceWorldVelocity() {
        val engine = HierarchicalMotionInferenceEngine()
        engine.process(1_000_000_000L, listOf(proposal(1_000_000_000L,0.0)))
        val result = engine.process(1_100_000_000L, listOf(proposal(1_100_000_000L,0.1)))
        assertEquals(1, result.activeTrackCount)
        assertTrue(result.states.first().velocityMetersPerSecond.x > 0.4)
        engine.close()
    }

    @Test
    fun consistentHighSpeedMeasurementRemainsResponsive() {
        val engine =
            HierarchicalMotionInferenceEngine()
        engine.process(
            1_000_000_000L,
            listOf(
                proposal(
                    1_000_000_000L,
                    0.0,
                    confidence = 0.96,
                ),
            ),
        )
        val result =
            engine.process(
                1_050_000_000L,
                listOf(
                    proposal(
                        1_050_000_000L,
                        0.10,
                        confidence = 0.96,
                    ),
                ),
            )
        val state = result.states.first()
        assertTrue(state.measurementBlend > 0.65)
        assertTrue(state.innovationScore > 0.55)
        assertTrue(state.velocityMetersPerSecond.x > 0.50)
        engine.close()
    }

    @Test
    fun catastrophicMeasurementSpikeIsRobustlyDownweighted() {
        val engine =
            HierarchicalMotionInferenceEngine()
        engine.process(
            2_000_000_000L,
            listOf(
                proposal(
                    2_000_000_000L,
                    0.0,
                    confidence = 0.98,
                ),
            ),
        )
        engine.process(
            2_050_000_000L,
            listOf(
                proposal(
                    2_050_000_000L,
                    0.02,
                    confidence = 0.98,
                ),
            ),
        )
        val result =
            engine.process(
                2_066_000_000L,
                listOf(
                    proposal(
                        2_066_000_000L,
                        5.0,
                        confidence = 0.98,
                    ),
                ),
            )
        val state = result.states.first()
        assertTrue(
            result.robustOutlierCount >= 1,
        )
        assertTrue(
            result.warnings.contains(
                "robust-innovation-downweighting-active",
            ),
        )
        assertTrue(state.measurementBlend < 0.40)
        assertTrue(state.innovationScore < 0.50)
        assertTrue(
            state.positionMeters.x <
                5.0 * 0.40,
        )
        engine.close()
    }

    @Test
    fun weakGeometricEvidenceCannotOutvoteStrongMeasuredEvidence() {
        fun measured(
            x: Double,
            confidence: Double,
            geometry: Double,
            diversity: Double,
        ) =
            WorldSpaceProposal(
                actorId = 9,
                pointId = 4,
                timestampNanos = 3_000_000_000L,
                positionMeters = Vec3(x, 0.0, 0.0),
                uncertainty =
                    UncertaintyEnvelope(
                        0.004,
                        0.004,
                        0.004,
                        confidence = confidence,
                    ),
                confidence = confidence,
                geometricSupport = geometry,
                temporalSupport = 0.90,
                learnedSupport = 0.0,
                independentCameraCount = 3,
                viewDiversity = diversity,
                origin =
                    EvidenceOrigin.GEOMETRICALLY_CONSTRAINED,
                provenance = emptyList(),
            )

        val engine =
            HierarchicalMotionInferenceEngine()

        engine.process(
            3_000_000_000L,
            listOf(
                measured(
                    x = 0.0,
                    confidence = 0.95,
                    geometry = 0.95,
                    diversity = 0.90,
                ),
            ),
        )
        val result =
            engine.process(
                3_050_000_000L,
                listOf(
                    measured(
                        x = 0.02,
                        confidence = 0.98,
                        geometry = 0.15,
                        diversity = 0.20,
                    ),
                    measured(
                        x = 0.10,
                        confidence = 0.72,
                        geometry = 0.98,
                        diversity = 0.96,
                    ),
                ),
            )

        val state = result.states.first()
        assertTrue(
            state.positionMeters.x > 0.06,
        )
        engine.close()
    }

    @Test
    fun missingObservationProducesBoundedRecoveryHypotheses() {
        val engine = HierarchicalMotionInferenceEngine(
            HierarchicalMotionInferenceEngine.Config(maximumHypothesesPerTrack = 3),
        )
        engine.process(1_000_000_000L, listOf(proposal(1_000_000_000L,0.0)))
        engine.process(1_100_000_000L, listOf(proposal(1_100_000_000L,0.1)))
        val result = engine.process(1_150_000_000L, emptyList())
        assertTrue(result.recoveredTrackCount == 1)
        assertTrue(result.states.count { it.hypothesisRank > 0 } == 2)
        assertTrue(result.states.all { it.positionMeters.x.isFinite() && it.positionMeters.y.isFinite() && it.positionMeters.z.isFinite() })
        engine.close()
    }

    @Test
    fun staleProposalIsNotReinjectedIntoNewerState() {
        val engine = HierarchicalMotionInferenceEngine()
        engine.process(2_000_000_000L, listOf(proposal(2_000_000_000L,0.0)))
        val result = engine.process(2_100_000_000L, listOf(proposal(2_000_000_000L,99.0)))
        assertTrue(result.droppedProposalCount > 0)
        assertEquals(0.0, result.states.first().positionMeters.x, 0.001)
        engine.close()
    }
}