package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.MotionCaptureCustomization
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class ActorDetailAllocationEngineTest {

    private fun evidence(
        actorId: Long,
    ) =
        ActorDetailAllocationEngine.ActorEvidence(
            actorId = actorId,
            confidence = 0.92,
            visibleRatio = 0.88,
            motionEnergy = 0.54,
            interactionPriority = 0.40,
            identityContinuity = 0.91,
            occlusionRisk = 0.12,
            screenAreaRatio = 0.18,
        )

    @Test
    fun skipped_actor_receives_detail_through_bounded_aging_fairness() {
        val engine = ActorDetailAllocationEngine()
        val customization =
            MotionCaptureCustomization.qualityFirst().copy(
                maxSubjects = 2,
                fullDetailActorBudget = 1,
                faceDetailActorBudget = 1,
                eyeDetailActorBudget = 1,
                handDetailActorBudget = 1,
                denseSurfaceActorBudget = 1,
            )

        val first =
            engine.allocate(
                evidence = listOf(
                    evidence(1L),
                    evidence(2L),
                ),
                customization = customization,
            )

        assertEquals(
            setOf(1L),
            first.fullDetailActorIds,
        )

        val second =
            engine.allocate(
                evidence = listOf(
                    evidence(1L),
                    evidence(2L),
                ),
                customization = customization,
            )

        assertEquals(
            setOf(2L),
            second.fullDetailActorIds,
        )
        assertEquals(
            2,
            second.allocations.size,
        )
        assertTrue(
            second.warnings.none {
                it == "DETAIL_ALLOCATOR_SCENE_EXCEEDS_GLOBAL_SUBJECT_LIMIT"
            },
        )
    }
}
