package com.nivukx.nox.core.ai

import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Test

class MultiActorPoseLandmarkerEngineTest {
    @Test
    fun multi_actor_configuration_accepts_full_32_subject_capacity() {
        val config = MultiActorPoseLandmarkerEngine.Config(maxPoses = 32)

        assertEquals(32, config.maxPoses)
        assertTrue(config.minDetectionConfidence in 0f..1f)
    }

    @Test
    fun association_configuration_matches_high_end_actor_limit() {
        val config = MultiActorPoseAssociationEngine.Config(maxActors = 32)

        assertEquals(32, config.maxActors)
    }
}
