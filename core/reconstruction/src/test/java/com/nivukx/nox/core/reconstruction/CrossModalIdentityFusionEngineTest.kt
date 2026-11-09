package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CrossModalIdentityFusionEngineTest {

    @Test
    fun assignsCloseMultimodalObservationToSeededTrack() {
        val engine = CrossModalIdentityFusionEngine()
        val timestamp = 1_000_000_000L

        engine.seed(
            listOf(
                CrossModalIdentityFusionEngine.TrackSeed(
                    trackId = 7L,
                    timestampNanos = timestamp,
                    bodyRoot = Vec3(0.0, 1.0, 0.0),
                    faceCenter = Vec3(0.0, 1.55, 0.0),
                    leftHand = Vec3(-0.45, 1.05, 0.0),
                    rightHand = Vec3(0.45, 1.05, 0.0),
                    leftEye = Vec3(-0.03, 1.58, 0.06),
                    rightEye = Vec3(0.03, 1.58, 0.06),
                    confidence = 0.95,
                ),
            ),
        )

        val result = engine.update(
            timestampNanos = timestamp + 33_000_000L,
            observations = listOf(
                CrossModalIdentityFusionEngine.Observation(
                    observationId = 41L,
                    timestampNanos = timestamp + 33_000_000L,
                    bodyRoot = Vec3(0.01, 1.01, 0.0),
                    faceCenter = Vec3(0.01, 1.55, 0.0),
                    leftHand = Vec3(-0.44, 1.04, 0.01),
                    rightHand = Vec3(0.44, 1.04, -0.01),
                    leftEye = Vec3(-0.02, 1.58, 0.06),
                    rightEye = Vec3(0.04, 1.58, 0.06),
                    confidence = 0.94,
                ),
            ),
        )

        assertEquals(1, result.assignedTrackCount)
        assertEquals(41L, result.assignments.single().observationId)
        assertTrue(result.continuityScore > 0.50)
        assertTrue(result.swapRisk < 0.50)
    }

    @Test
    fun firstObservationAcquiresProductionTrack() {
        val engine = CrossModalIdentityFusionEngine()
        val timestamp = 3_000_000_000L

        val result = engine.update(
            timestampNanos = timestamp,
            observations = listOf(
                CrossModalIdentityFusionEngine.Observation(
                    observationId = 100L,
                    timestampNanos = timestamp,
                    bodyRoot = Vec3(0.0, 1.0, 0.0),
                    faceCenter = Vec3(0.0, 1.55, 0.0),
                    confidence = 0.93,
                ),
            ),
        )

        assertEquals(1, result.assignedTrackCount)
        assertEquals(0, result.unmatchedObservationIds.size)
        assertEquals(100L, result.assignments.single().observationId)
        assertTrue(result.assignments.single().action ==
            CrossModalIdentityFusionEngine.Action.ASSIGN)
    }

    @Test
    fun unreliableFaceEvidenceDoesNotDragIdentityAssociation() {
        val engine = CrossModalIdentityFusionEngine()
        val timestamp = 4_000_000_000L

        engine.seed(
            listOf(
                CrossModalIdentityFusionEngine.TrackSeed(
                    trackId = 3L,
                    timestampNanos = timestamp,
                    bodyRoot = Vec3(0.0, 1.0, 0.0),
                    faceCenter = Vec3(0.0, 1.55, 0.0),
                    leftHand = Vec3(-0.4, 1.05, 0.0),
                    rightHand = Vec3(0.4, 1.05, 0.0),
                    confidence = 0.94,
                ),
            ),
        )

        val result =
            engine.update(
                timestampNanos = timestamp + 33_000_000L,
                observations = listOf(
                    CrossModalIdentityFusionEngine.Observation(
                        observationId = 91L,
                        timestampNanos = timestamp + 33_000_000L,
                        bodyRoot = Vec3(0.01, 1.01, 0.0),
                        faceCenter = Vec3(0.80, 1.55, 0.0),
                        leftHand = Vec3(-0.39, 1.04, 0.0),
                        rightHand = Vec3(0.39, 1.04, 0.0),
                        confidence = 0.94,
                        evidenceProfile = mapOf(
                            CrossModalEvidenceArbitrationEngine.Modality.FACE to
                                CrossModalEvidenceArbitrationEngine.ModalityEvidence(
                                    modality =
                                        CrossModalEvidenceArbitrationEngine.Modality.FACE,
                                    confidence = 0.92,
                                    uncertaintyMeters = 0.04,
                                    oodRisk = 0.97,
                                ),
                        ),
                    ),
                ),
            )

        assertTrue(result.assignedTrackCount == 1)
        assertTrue(
            result.evidenceFusionScore > 0.45,
        )
        assertTrue(
            result.warnings.any {
                it.contains("deprioritized") ||
                    it.contains("out-of-distribution")
            },
        )
        assertTrue(
            result.assignments
                .single()
                .diagnostics
                .any {
                    it.contains("evidence") ||
                        it.contains("deprioritized")
                },
        )
    }

    @Test
    fun identityAssignmentReportsSelectedEvidenceModalities() {
        val engine = CrossModalIdentityFusionEngine()
        val timestamp = 6_000_000_000L

        engine.seed(
            listOf(
                CrossModalIdentityFusionEngine.TrackSeed(
                    trackId = 12L,
                    timestampNanos = timestamp,
                    bodyRoot = Vec3(0.0, 1.0, 0.0),
                    faceCenter = Vec3(0.0, 1.55, 0.0),
                    confidence = 0.94,
                ),
            ),
        )

        val result =
            engine.update(
                timestampNanos = timestamp + 33_000_000L,
                observations = listOf(
                    CrossModalIdentityFusionEngine.Observation(
                        observationId = 121L,
                        timestampNanos = timestamp + 33_000_000L,
                        bodyRoot = Vec3(0.01, 1.0, 0.0),
                        faceCenter = Vec3(0.01, 1.55, 0.0),
                        leftEye = Vec3(-0.03, 1.58, 0.06),
                        rightEye = Vec3(0.03, 1.58, 0.06),
                        confidence = 0.94,
                    ),
                ),
            )

        val assignment =
            result.assignments.single {
                it.observationId == 121L
            }

        assertTrue(
            assignment.selectedEvidenceModalities
                .contains("BODY"),
        )
        assertTrue(
            assignment.selectedEvidenceModalities.isNotEmpty(),
        )
    }

    @Test
    fun globalAssignmentPreventsDuplicateObservationUse() {
        val engine = CrossModalIdentityFusionEngine()
        val timestamp = 2_000_000_000L

        engine.seed(
            listOf(
                CrossModalIdentityFusionEngine.TrackSeed(
                    trackId = 1L,
                    timestampNanos = timestamp,
                    bodyRoot = Vec3(0.0, 1.0, 0.0),
                    faceCenter = Vec3(0.0, 1.5, 0.0),
                    confidence = 0.9,
                ),
                CrossModalIdentityFusionEngine.TrackSeed(
                    trackId = 2L,
                    timestampNanos = timestamp,
                    bodyRoot = Vec3(2.0, 1.0, 0.0),
                    faceCenter = Vec3(2.0, 1.5, 0.0),
                    confidence = 0.9,
                ),
            ),
        )

        val result = engine.update(
            timestampNanos = timestamp + 33_000_000L,
            observations = listOf(
                CrossModalIdentityFusionEngine.Observation(
                    observationId = 10L,
                    timestampNanos = timestamp + 33_000_000L,
                    bodyRoot = Vec3(0.02, 1.0, 0.0),
                    faceCenter = Vec3(0.01, 1.5, 0.0),
                    confidence = 0.92,
                ),
                CrossModalIdentityFusionEngine.Observation(
                    observationId = 11L,
                    timestampNanos = timestamp + 33_000_000L,
                    bodyRoot = Vec3(2.01, 1.0, 0.0),
                    faceCenter = Vec3(2.01, 1.5, 0.0),
                    confidence = 0.91,
                ),
            ),
        )

        val assigned = result.assignments.mapNotNull { it.observationId }
        assertEquals(2, assigned.distinct().size)
        assertTrue(result.continuityScore > 0.45)
    }
}
